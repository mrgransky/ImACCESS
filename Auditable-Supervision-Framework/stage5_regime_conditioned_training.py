import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

print(f"sys_path: {sys.path}")

from utils import *
import clip
from clip_peft import get_injected_peft_clip, get_adapter_peft_clip
from loss import compute_loss_masks, diagnose_train_val_coverage
from stage5_dataset_loader import get_stage5_dataloaders
from stage5_regime_aware_loss import (
	compute_regime_aware_contrastive_loss,
	REGIME_LAMBDA_I2T,
	REGIME_LAMBDA_T2I,
	REGIME_LAMBDA_REPEL,
	VALID_REGIMES,
	SKIP_REGIMES,
)

SUPPORTED_PEFT = {
	"lora", "lora_plus", "dora", "rslora", "ia3", "vera",   # injected PEFT (get_injected_peft_clip)
	"tip_adapter", "tip_adapter_f",                         # adapter PEFT  (get_adapter_peft_clip)
	"clip_adapter_v", "clip_adapter_t", "clip_adapter_vt",  # adapter PEFT  (get_adapter_peft_clip)
	"probe",                                                # linear probe
	"full",                                                 # full fine-tuning
}

# 1. REGIME EPOCH DIAGNOSTICS
def log_regime_epoch_stats(
	regime_counters: Dict[str, int],
	loss_by_regime:  Dict[str, List[float]],
	epoch:           int,
	split:           str = "TRAIN",
	verbose:         bool = True,
) -> Dict[str, Any]:
	"""
	Emit per-regime loss and sample-count statistics for one epoch.
	Returns a flat dict suitable for JSON serialisation into metrics log.
	"""
	total_samples = sum(regime_counters.values())
	stats = {"epoch": epoch, "split": split, "total_samples": total_samples}
	
	if verbose:
		print(f"[{split}] Regime distribution")
	
	for regime in sorted(regime_counters.keys()):
		count   = regime_counters[regime]
		pct     = count / max(total_samples, 1) * 100
		losses  = loss_by_regime.get(regime, [])
		avg_loss = float(np.mean(losses)) if losses else float("nan")
		stats[f"{regime}_count"]    = count
		stats[f"{regime}_pct"]      = round(pct, 2)
		stats[f"{regime}_avg_loss"] = round(avg_loss, 6)
		if verbose:
			print(
				f"  ├─ {regime:<20s}: {count:>6,} ({pct:5.1f}%) "
				f"avg_loss={avg_loss:.6f}"
			)
	
	if verbose:
		print(f"  └─ Total samples : {total_samples:,}")
	
	return stats

# 2. EVALUATION (mirrors lora_finetune_multi_label eval block)
@torch.no_grad()
def evaluate(
	model:            torch.nn.Module,
	val_loader:       DataLoader,
	all_class_embeds: torch.Tensor,
	criterion_i2t:    torch.nn.BCEWithLogitsLoss,
	criterion_t2i:    torch.nn.BCEWithLogitsLoss,
	active_mask:      torch.Tensor,
	head_mask:        torch.Tensor,
	rare_mask:        torch.Tensor,
	temperature:      float,
	device:           torch.device,
	epoch:            int,
	verbose:          bool = True,
) -> Dict[str, float]:
	"""
	Validation pass.  Returns a metrics dict with:
		val_loss, val_loss_i2t, val_loss_t2i, val_loss_repel,
		val_map_all, val_map_head, val_map_rare,
		val_p@1, val_p@5, val_ndcg@5
	"""
	model.eval()
	num_classes = all_class_embeds.shape[0]
	if verbose:
		print(f"\n[VALIDATION EPOCH {epoch}] {len(val_loader.dataset)} samples (batch size: {val_loader.batch_size})")

	total_loss = total_i2t = total_t2i = total_repel = 0.0
	n_batches  = 0
	all_scores  = []
	all_targets = []
	regime_counters: Dict[str, int]        = {}
	loss_by_regime:  Dict[str, List[float]] = {}
	class_embeds = torch.nn.functional.normalize(all_class_embeds, dim=-1).to(device)

	for batch in val_loader:
		if not batch:
			continue

		images     = batch["image"].to(device)
		label_vec  = batch["label_vec"].to(device)
		hn_vec     = batch["hn_vec"].to(device)
		w_pos      = batch["w_pos"].to(device)
		w_neg      = batch["w_neg"].to(device)
		regimes    = batch["regime"]

		loss, l_i2t, l_t2i, l_repel = compute_regime_aware_contrastive_loss(
			model=model,
			images=images,
			all_class_embeds=class_embeds,
			label_vectors=label_vec,
			hn_vectors=hn_vec,
			regimes=regimes,
			w_pos_raw=w_pos,
			w_neg_raw=w_neg,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			temperature=temperature,
			loss_weights={
				"i2t":   REGIME_LAMBDA_I2T,
				"t2i":   REGIME_LAMBDA_T2I,
				"repel": REGIME_LAMBDA_REPEL,
			},
			split="VAL",
			verbose=verbose,
		)

		total_loss  += loss.item()
		total_i2t   += l_i2t.item()
		total_t2i   += l_t2i.item()
		total_repel += l_repel.item()
		n_batches   += 1

		# Regime counters for val diagnostics
		for r in regimes:
				regime_counters[r] = regime_counters.get(r, 0) + 1
				loss_by_regime.setdefault(r, []).append(loss.item())

		# Retrieval scores: cosine sim image → class
		image_embeds = torch.nn.functional.normalize(model.encode_image(images), dim=-1).float()
		scores = torch.matmul(image_embeds, class_embeds.T)  # [B, C]
		all_scores.append(scores.cpu())
		all_targets.append(label_vec.cpu())

	# Aggregate retrieval metrics
	all_scores  = torch.cat(all_scores,  dim=0) # [N_val, C]
	all_targets = torch.cat(all_targets, dim=0) # [N_val, C]

	metrics = _compute_retrieval_metrics(
		scores=all_scores,
		targets=all_targets,
		active_mask=active_mask.cpu(),
		head_mask=head_mask.cpu(),
		rare_mask=rare_mask.cpu(),
	)

	n_b = max(n_batches, 1)
	metrics.update(
		{
			"val_loss":       total_loss  / n_b,
			"val_loss_i2t":   total_i2t   / n_b,
			"val_loss_t2i":   total_t2i   / n_b,
			"val_loss_repel": total_repel / n_b,
		}
	)

	if verbose:
		log_regime_epoch_stats(regime_counters, loss_by_regime, epoch, split="VAL")
		print(
				f"Epoch {epoch} [VAL] "
				f"loss={metrics['val_loss']:.6f} "
				f"(i2t={metrics['val_loss_i2t']:.4f} "
				f"t2i={metrics['val_loss_t2i']:.4f} "
				f"repel={metrics['val_loss_repel']:.4f}) | "
				f"mAP={metrics.get('val_map_all', float('nan')):.4f} "
				f"P@1={metrics.get('val_p@1', float('nan')):.4f} "
				f"nDCG@5={metrics.get('val_ndcg@5', float('nan')):.4f}"
		)

	return metrics

def _compute_retrieval_metrics(
		scores:      torch.Tensor,   # [N, C]
		targets:     torch.Tensor,   # [N, C]
		active_mask: torch.Tensor,   # [C] bool
		head_mask:   torch.Tensor,   # [C] bool
		rare_mask:   torch.Tensor,   # [C] bool
) -> Dict[str, float]:
		"""
		Compute mAP, P@K, nDCG@K over active classes.
		Head / rare sub-metrics computed over the respective class subsets.
		"""
		# Restrict to active classes only
		s = scores[:, active_mask]   # [N, C_active]
		t = targets[:, active_mask]  # [N, C_active]

		map_all  = _mean_average_precision(s, t)
		p_at_1   = _precision_at_k(s, t, k=1)
		p_at_5   = _precision_at_k(s, t, k=5)
		ndcg_at5 = _ndcg_at_k(s, t, k=5)

		# Head / rare sub-metrics (within active classes)
		head_sub = head_mask & active_mask
		rare_sub = rare_mask & active_mask

		map_head = _mean_average_precision(scores[:, head_sub], targets[:, head_sub]) \
				if head_sub.sum() > 0 else float("nan")
		map_rare = _mean_average_precision(scores[:, rare_sub], targets[:, rare_sub]) \
				if rare_sub.sum() > 0 else float("nan")

		return {
				"val_map_all":  map_all,
				"val_map_head": map_head,
				"val_map_rare": map_rare,
				"val_p@1":      p_at_1,
				"val_p@5":      p_at_5,
				"val_ndcg@5":   ndcg_at5,
		}

def _mean_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
		"""Macro-averaged AP over classes (column-wise AP)."""
		if targets.sum() == 0:
				return float("nan")
		C = scores.shape[1]
		aps = []
		for c in range(C):
				pos = targets[:, c].sum().item()
				if pos == 0:
						continue
				order  = scores[:, c].argsort(descending=True)
				hits   = targets[order, c].float()
				prec   = hits.cumsum(0) / torch.arange(1, len(hits) + 1, dtype=torch.float32)
				ap     = (prec * hits).sum().item() / pos
				aps.append(ap)
		return float(np.mean(aps)) if aps else float("nan")

def _precision_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
		"""Sample-averaged P@K."""
		topk = scores.topk(min(k, scores.shape[1]), dim=1).indices   # [N, k]
		hits = targets.gather(1, topk).float()
		return float(hits.mean().item())

def _ndcg_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> float:
		"""Sample-averaged nDCG@K."""
		N = scores.shape[0]
		topk_idx = scores.topk(min(k, scores.shape[1]), dim=1).indices  # [N, k]
		gains    = targets.gather(1, topk_idx).float()                   # [N, k]
		discounts = torch.log2(
				torch.arange(2, gains.shape[1] + 2, dtype=torch.float32)
		)                    # [k]
		dcg  = (gains / discounts).sum(dim=1)                    # [N]
		# Ideal DCG: sort targets descending, take top-k
		ideal_gains = targets.float().sort(dim=1, descending=True).values[:, :k]
		idcg = (ideal_gains / discounts[:ideal_gains.shape[1]]).sum(dim=1)
		ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
		return float(ndcg.mean().item())

# 3. CLASS EMBEDDING BUILDER
@torch.no_grad()
def build_class_embeddings(
		model:      torch.nn.Module,
		label_dict: Dict[str, int],
		device:     torch.device,
		batch_size: int = 256,
		verbose:    bool = True,
) -> torch.Tensor:
		"""
		Encode all canonical class names with the (frozen) text encoder.
		Returns [C, D] FP32 tensor on CPU.

		Rebuilt at the start of each epoch so LoRA text-encoder updates
		are reflected without stale embeddings.
		"""
		model.eval()
		labels_sorted = [lbl for lbl, _ in sorted(label_dict.items(), key=lambda x: x[1])]
		all_embeds    = []

		for i in range(0, len(labels_sorted), batch_size):
				chunk  = labels_sorted[i : i + batch_size]
				tokens = clip.tokenize(chunk).to(device)
				embeds = model.encode_text(tokens).float().cpu()
				all_embeds.append(embeds)

		class_embeds = torch.cat(all_embeds, dim=0)   # [C, D]

		if verbose:
				print(
						f"[EMBEDDINGS] {class_embeds.shape[0]} classes | "
						f"dim={class_embeds.shape[1]} | "
						f"norm range [{class_embeds.norm(dim=-1).min():.3f}, "
						f"{class_embeds.norm(dim=-1).max():.3f}]"
				)

		return class_embeds

# 4. PEFT SETUP  (mirrors lora_finetune_multi_label)
def setup_peft(
	model:       torch.nn.Module,
	peft_method: str,
	peft_config: Optional[Dict] = None,
	verbose:     bool = True,
) -> Tuple[torch.nn.Module, List[Dict]]:
	"""
	Apply PEFT to model and return (model, optimizer_param_groups).
	Uses the custom clip_peft.py implementations exclusively — no dependency
	on the HuggingFace `peft` package.
	peft_config keys (all optional, sensible defaults provided):
			rank            : LoRA / DoRA / VeRA / rsLoRA rank          (default: 16)
			alpha           : LoRA scaling factor                        (default: 32)
			dropout         : adapter dropout rate                       (default: 0.05)
			lr_multiplier   : B-matrix LR multiplier for lora_plus       (default: 16.0)
			target_text_modules   : text-encoder module names to inject  (default: see below)
			target_vision_modules : vision-encoder module names to inject (default: see below)
			quantized             : use bitsandbytes quantisation         (default: False)
			quantization_bits     : 4 or 8                               (default: 8)
			compute_dtype         : torch dtype for quantised compute     (default: torch.float16)
			# adapter-specific (tip_adapter / tip_adapter_f)
			initial_beta    : Tip-Adapter temperature                    (default: 1.0)
			initial_alpha   : Tip-Adapter scaling                        (default: 1.0)
			# adapter-specific (clip_adapter_*)
			bottleneck_dim  : CLIP-Adapter bottleneck dimension          (default: 64)
			activation      : CLIP-Adapter activation ('relu'/'gelu')    (default: 'relu')
	"""
	if peft_config is None:
			peft_config = {}
	peft_method = peft_method.lower()
	assert peft_method in SUPPORTED_PEFT, f"[PEFT] Unknown: '{peft_method}'. Choose: {SUPPORTED_PEFT}"
	rank          = peft_config.get("rank",    16)
	alpha         = peft_config.get("alpha",   32)
	dropout       = peft_config.get("dropout", 0.05)
	lr_multiplier = peft_config.get("lr_multiplier", 16.0)   # for lora_plus
	quantized         = peft_config.get("quantized",         False)
	quantization_bits = peft_config.get("quantization_bits", 8)
	compute_dtype     = peft_config.get("compute_dtype",     torch.float16)
	# Default target modules — mirrors lora_finetune_multi_label()
	default_text_modules   = ["in_proj", "out_proj", "c_fc", "c_proj"]
	default_vision_modules = ["in_proj", "out_proj", "c_fc", "c_proj"]
	target_text_modules   = peft_config.get("target_text_modules",   default_text_modules)
	target_vision_modules = peft_config.get("target_vision_modules", default_vision_modules)
	if verbose:
		print(f"\n[PEFT] method={peft_method} | rank={rank} | alpha={alpha} | dropout={dropout}")
	
	# ── Injected PEFT methods (get_injected_peft_clip) ────
	# lora, lora_plus, rslora, dora, vera, ia3
	if peft_method in {"lora", "lora_plus", "dora", "rslora", "vera", "ia3"}:
			# lora_plus passes a lambda multiplier; others pass None
			lora_plus_lambda = lr_multiplier if peft_method == "lora_plus" else None
			model = get_injected_peft_clip(
					clip_model=model,
					method=peft_method,
					rank=rank,
					alpha=alpha,
					dropout=dropout,
					lora_plus_lambda=lora_plus_lambda,
					target_text_modules=target_text_modules,
					target_vision_modules=target_vision_modules,
					quantized=quantized,
					quantization_bits=quantization_bits,
					compute_dtype=compute_dtype,
					verbose=verbose,
			)
			if peft_method == "lora_plus":
					# LoRA+: A matrices use base LR, B matrices use lr_multiplier × base LR
					lora_a_params = [p for n, p in model.named_parameters()
							if p.requires_grad and "lora_A" in n]
					lora_b_params = [p for n, p in model.named_parameters()
							if p.requires_grad and "lora_B" in n]
					param_groups = [
							{"params": lora_a_params, "lr_multiplier": 1.0},
							{"params": lora_b_params, "lr_multiplier": lr_multiplier},
					]
			else:
					trainable = [p for p in model.parameters() if p.requires_grad]
					param_groups = [{"params": trainable}]
			if verbose:
					n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
					n_total     = sum(p.numel() for p in model.parameters())
					print(
							f"[PEFT][{peft_method}] trainable: {n_trainable:,} / {n_total:,} "
							f"({100*n_trainable/max(n_total,1):.3f}%)"
					)
	
	# ── Adapter PEFT methods (get_adapter_peft_clip) ────
	# tip_adapter, tip_adapter_f, clip_adapter_v, clip_adapter_t, clip_adapter_vt
	elif peft_method in {"tip_adapter", "tip_adapter_f", "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"}:
		is_clip_adapter = peft_method.startswith("clip_adapter")
		is_tip_adapter  = peft_method.startswith("tip_adapter")
		adapter_kwargs = dict(
			clip_model=model,
			method=peft_method,
			verbose=verbose,
		)
		if is_tip_adapter:
			adapter_kwargs["initial_beta"]  = peft_config.get("initial_beta",  1.0)
			adapter_kwargs["initial_alpha"] = peft_config.get("initial_alpha", 1.0)
		
		if is_clip_adapter:
			adapter_kwargs["bottleneck_dim"] = peft_config.get("bottleneck_dim", 64)
			adapter_kwargs["activation"]     = peft_config.get("activation",     "relu")
		
		model = get_adapter_peft_clip(**adapter_kwargs)
		trainable = [p for p in model.parameters() if p.requires_grad]
		param_groups = [{"params": trainable}]
		
		if verbose:
				n_trainable = sum(p.numel() for p in trainable)
				n_total     = sum(p.numel() for p in model.parameters())
				print(
						f"[PEFT][{peft_method}] trainable: {n_trainable:,} / {n_total:,} "
						f"({100*n_trainable/max(n_total,1):.3f}%)"
				)

	# ── Linear probe ────
	elif peft_method == "probe":
		# Freeze everything, then unfreeze only the final projection parameters
		for p in model.parameters():
			p.requires_grad_(False)
		
		# Unfreeze visual.proj and text_projection (both are nn.Parameter in CLIP)
		if hasattr(model.visual, "proj") and isinstance(model.visual.proj, torch.nn.Parameter):
			model.visual.proj.requires_grad_(True)
		
		if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
			model.text_projection.requires_grad_(True)
		
		trainable = [p for p in model.parameters() if p.requires_grad]
		param_groups = [{"params": trainable}]
		
		if verbose:
			n_trainable = sum(p.numel() for p in trainable)
			print(f"[PEFT][probe] trainable params: {n_trainable:,}")
	
	# ── Full fine-tuning ────
	elif peft_method == "full":
		for p in model.parameters():
			p.requires_grad_(True)
		param_groups = [{"params": list(model.parameters())}]
		if verbose:
			n = sum(p.numel() for p in model.parameters())
			print(f"[PEFT][full] trainable params: {n:,}")
	
	return model, param_groups

# 5. CHECKPOINT HELPERS
def save_checkpoint(
	model:torch.nn.Module,
	optimizer:torch.optim.Optimizer,
	scheduler,
	epoch:int,
	metrics:Dict[str, float],
	label_dict:Dict[str, int],
	checkpoints_file_path:str,
	verbose:bool=False,
):
	state = {
		"epoch": epoch,
		"metrics": metrics,
		"label_dict": label_dict,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict() if scheduler else None,
	}
	torch.save(state, checkpoints_file_path)
	if verbose:
		print(f"[Checkpoint] Saved {checkpoints_file_path} (epoch={epoch} val_loss={metrics.get('val_loss', float('nan')):.6f})")

def load_checkpoint(
	ckpt_path:str,
	model:torch.nn.Module,
	optimizer:Optional[torch.optim.Optimizer] = None,
	scheduler=None,
	device:torch.device = torch.device("cpu"),
	verbose:bool = True,
) -> Tuple[int, Dict]:
	assert os.path.isfile(ckpt_path), f"[load_checkpoint] Not found: {ckpt_path}"
	state = torch.load(ckpt_path, map_location=device)
	model.load_state_dict(state["model_state_dict"])
	if optimizer and "optimizer_state_dict" in state:
			optimizer.load_state_dict(state["optimizer_state_dict"])
	if scheduler and state.get("scheduler_state_dict"):
			scheduler.load_state_dict(state["scheduler_state_dict"])
	epoch   = state.get("epoch", 0)
	metrics = state.get("metrics", {})
	if verbose:
			print(f"[load_checkpoint] Resumed from epoch {epoch} | {ckpt_path}")
	return epoch, metrics

# 6. MASTER TRAINING FUNCTION
def regime_conditioned_finetune(
	# ── Data ────
	metadata_fpth:      str,
	checkpoints_dir:    str,
	supervision_fpth:   str, # auditable_supervision_matrix.parquet
	id_col:             str = "doc_url",
	text_col:           str = "multimodal_labels",
	# ── Model ────
	clip_model_name:    str = "ViT-L/14",
	peft_method:        str = "lora",
	peft_config:        Optional[Dict] = None,
	resume_ckpt:        Optional[str]  = None,
	# ── Training ────
	num_epochs:         int   = 30,
	batch_size:         int   = 128,
	num_workers:        int   = 4,
	learning_rate:      float = 1e-4,
	weight_decay:       float = 1e-4,
	temperature:        float = 0.07,
	grad_clip:          float = 1.0,
	warmup_epochs:      int   = 2,
	patience:           int   = 7,
	# ── Loss ────
	pw_mode:            str   = "sqrt",
	pw_max_cap:         Optional[float] = 50.0,
	loss_weights:       Optional[Dict[str, float]] = None,
	# ── Misc ────
	seed:               int   = 42,
	verbose:            bool  = False,
) -> Dict[str, Any]:
	"""
	Regime-conditioned fine-tuning of a CLIP dual-encoder.
	Drop-in replacement for lora_finetune_multi_label() with two key additions:
			1. Regime-aware loss (ω_pos, ω_neg, repulsion arm)
			2. Per-epoch regime diagnostics logged to metrics JSON
	Returns
	----
	results : dict with keys
			best_epoch, best_val_loss, best_metrics,
			all_train_metrics, all_val_metrics,
			label_dict, checkpoints_dir
	"""
	# ── Reproducibility ────
	set_seeds(seed=seed)

	# ── Setup ────
	DATASET_DIRECTORY = os.path.dirname(metadata_fpth)
	os.makedirs(checkpoints_dir, exist_ok=True)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# ── Model + PEFT ────
	model, _ = clip.load(
		name=clip_model_name,
		device=device,
		jit=False, # training or finetuning => jit=False
		random_weights=False, # finetuning => random_weights=False
		dropout=0.0,
		download_root=get_model_directory(path=DATASET_DIRECTORY),
	)
	model.name = clip_model_name # Custom attribute to store model name
	model_name = model.__class__.__name__
	model_arch = re.sub(r'[/@]', '_', model.name) if hasattr(model, 'name') else 'unknown_arch'
	input_resolution = getattr(model.visual, "input_resolution", None)
	if verbose:
		print(f"[Stage5] input_resolution: {input_resolution}")

	ckpt_fname = f"{model_name}_{model_arch}_{peft_method}_checkpoint.pt"
	ckpt_fpath = os.path.join(checkpoints_dir, ckpt_fname)

	model, param_groups = setup_peft(
		model=model,
		peft_method=peft_method,
		peft_config=peft_config,
		verbose=verbose,
	)
	model = model.to(device)

	print(f"\n[Stage5] Regime-Conditioned Fine-Tuning: {model.__class__.__name__} {clip_model_name}")
	print(f"  ├─ input resolution : {input_resolution}")
	print(f"  ├─ PEFT             : {peft_method}")
	print(f"  ├─ Supervision      : {supervision_fpth}")
	print(f"  ├─ Device           : {device}")
	print(f"  ├─ Epochs           : {num_epochs}")
	print(f"  ├─ Batch size       : {batch_size}")
	print(f"  ├─ LR               : {learning_rate}")
	print(f"  ├─ Dataset          : {DATASET_DIRECTORY}")
	print(f"  └─ Checkpoints      : {checkpoints_dir}")

	# ── DataLoaders ────
	train_loader, val_loader = get_stage5_dataloaders(
		metadata_fpth=metadata_fpth,
		supervision_fpth=supervision_fpth,
		batch_size=batch_size,
		num_workers=num_workers,
		input_resolution=input_resolution,
		id_col=id_col,
		text_col=text_col,
		verbose=verbose,
	)
	label_dict  = train_loader.dataset.label_dict
	num_classes = len(label_dict)

	# ── Loss masks (Axis 1: class-level balance) ────
	loss_masks = compute_loss_masks(
		loader=train_loader,
		num_classes=num_classes,
		device=device,
		pw_mode=pw_mode,
		pw_max_cap=pw_max_cap,
		verbose=verbose,
	)
	active_mask = loss_masks["active_mask"]
	head_mask   = loss_masks["head_mask"]
	rare_mask   = loss_masks["rare_mask"]
	pos_weight  = loss_masks["pos_weight"]
	
	# Train/val coverage diagnostic
	diagnose_train_val_coverage(
		train_freq=loss_masks["train_freq"],
		validation_loader=val_loader,
		num_classes=num_classes,
		verbose=verbose,
	)
	
	# ── Criteria ────
	criterion_i2t = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none",)
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction="none")
	

	# ── Optimizer ────
	# Resolve per-group LR for LoRA+ (lr_multiplier stored in group dict)
	for grp in param_groups:
		mult = grp.pop("lr_multiplier", 1.0)
		grp.setdefault("lr", learning_rate * mult)
		grp.setdefault("weight_decay", weight_decay)
	optimizer = torch.optim.AdamW(param_groups)

	# ── Scheduler: linear warmup → cosine decay ────
	total_steps   = num_epochs * len(train_loader)
	warmup_steps  = warmup_epochs * len(train_loader)
	def lr_lambda(step: int) -> float:
		if step < warmup_steps:
			return float(step) / max(warmup_steps, 1)
		progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	# ── Resume ────
	start_epoch = 0
	if resume_ckpt:
		start_epoch, _ = load_checkpoint(
			ckpt_path=resume_ckpt,
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			device=device,
			verbose=verbose,
		)
		start_epoch += 1
	# ── AMP scaler ────
	use_amp = torch.cuda.is_available()
	scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

	# Main training loop
	best_val_loss    = float("inf")
	best_epoch       = -1
	best_metrics     = {}
	patience_counter = 0
	all_train_metrics: List[Dict] = []
	all_val_metrics:   List[Dict] = []
	start_time = time.time()
	for epoch in range(start_epoch, num_epochs):
		t0 = time.time()
		model.train()
		print(f"\n[Epoch {epoch+1}/{num_epochs}]")
		# Rebuild class embeddings each epoch (text encoder may have been updated)
		all_class_embeds = build_class_embeddings(
			model=model,
			label_dict=label_dict,
			device=device,
			verbose=verbose,
		).to(device)
		epoch_loss = epoch_i2t = epoch_t2i = epoch_repel = 0.0
		n_batches  = 0
		regime_counters: Dict[str, int]        = {}
		loss_by_regime:  Dict[str, List[float]] = {}
		for batch_idx, batch in enumerate(train_loader):
			if not batch:
				continue
			
			images    = batch["image"].to(device, non_blocking=True)
			label_vec = batch["label_vec"].to(device, non_blocking=True)
			hn_vec    = batch["hn_vec"].to(device, non_blocking=True)
			w_pos     = batch["w_pos"].to(device, non_blocking=True)
			w_neg     = batch["w_neg"].to(device, non_blocking=True)
			regimes   = batch["regime"]
			
			optimizer.zero_grad(set_to_none=True)
			
			with torch.cuda.amp.autocast(enabled=use_amp):
				loss, l_i2t, l_t2i, l_repel = compute_regime_aware_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vec,
					hn_vectors=hn_vec,
					regimes=regimes,
					w_pos_raw=w_pos,
					w_neg_raw=w_neg,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					split="TRAIN",
					verbose=verbose,
				)
			
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad],
				max_norm=grad_clip,
			)
			
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			epoch_loss  += loss.item()
			epoch_i2t   += l_i2t.item()
			epoch_t2i   += l_t2i.item()
			epoch_repel += l_repel.item()
			n_batches   += 1
			
			# Regime tracking
			for r in regimes:
				regime_counters[r] = regime_counters.get(r, 0) + 1
				loss_by_regime.setdefault(r, []).append(loss.item())
			
			if verbose and (batch_idx % max(1, len(train_loader) // 5) == 0):
				lr_now = scheduler.get_last_lr()[0]
				print(
					f"\t[{batch_idx:04d}/{len(train_loader):04d}] "
					f"loss={loss.item():.6f} "
					f"(i2t={l_i2t.item():.4f} t2i={l_t2i.item():.4f} repel={l_repel.item():.4f}) "
					f"lr={lr_now:.3e}"
				)
		
		# ── End-of-epoch train stats ────
		n_b = max(n_batches, 1)
		train_metrics = {
			"epoch":            epoch,
			"train_loss":       epoch_loss  / n_b,
			"train_loss_i2t":   epoch_i2t   / n_b,
			"train_loss_t2i":   epoch_t2i   / n_b,
			"train_loss_repel": epoch_repel / n_b,
			"lr":               scheduler.get_last_lr()[0],
		}
		regime_stats = log_regime_epoch_stats(
			regime_counters, 
			loss_by_regime, 
			epoch, 
			split="TRAIN", 
			verbose=verbose,
		)
		train_metrics.update(regime_stats)
		all_train_metrics.append(train_metrics)
		# ── Validation ────
		val_metrics = evaluate(
			model=model,
			val_loader=val_loader,
			all_class_embeds=all_class_embeds,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			head_mask=head_mask,
			rare_mask=rare_mask,
			temperature=temperature,
			device=device,
			epoch=epoch,
			verbose=verbose,
		)
		val_metrics["epoch"] = epoch
		all_val_metrics.append(val_metrics)
		print(f"[ELAPSED] {time.time() - t0:.2f}s")

		# ── Early stopping + checkpoint ────
		val_loss = val_metrics["val_loss"]
		if val_loss < best_val_loss:
			best_val_loss    = val_loss
			best_epoch       = epoch
			best_metrics     = {**train_metrics, **val_metrics}
			patience_counter = 0
			save_checkpoint(
				model=model,
				optimizer=optimizer,
				scheduler=scheduler,
				epoch=epoch,
				metrics=best_metrics,
				label_dict=label_dict,
				checkpoints_file_path=ckpt_fpath,
				verbose=verbose,
			)
		else:
			patience_counter += 1
			if verbose:
				print(
					f"[Stage5][Epoch {epoch}] No improvement "
					f"({patience_counter}/{patience}). "
					f"Best val_loss={best_val_loss:.6f} @ epoch {best_epoch}"
				)
			if patience_counter >= patience:
				print(f"\n[Stage5] Early stopping triggered at epoch {epoch}.")
				break

		# Persist metrics JSON
		ft_metrics_fpath = ckpt_fpath.replace("checkpoint.pt", "ft_metrics.json") 
		with open(ft_metrics_fpath, "w") as f:
			json.dump(
				{
					"train": all_train_metrics,
					"val":   all_val_metrics,
					"best":  best_metrics,
				},
				f,
				indent=2,
				ensure_ascii=False,
			)

	print(f"\n{'='*80}")
	print(f"Training complete, Total Elapsed time: {time.time() - start_time:.1f} sec")
	print(f"  ├─ Best epoch    : {best_epoch}")
	print(f"  ├─ Best val_loss : {best_val_loss:.6f}")
	print(f"  ├─ mAP (all)     : {best_metrics.get('val_map_all',  float('nan')):.4f}")
	print(f"  ├─ mAP (head)    : {best_metrics.get('val_map_head', float('nan')):.4f}")
	print(f"  ├─ mAP (rare)    : {best_metrics.get('val_map_rare', float('nan')):.4f}")
	print(f"  ├─ P@1           : {best_metrics.get('val_p@1',      float('nan')):.4f}")
	print(f"  ├─ nDCG@5        : {best_metrics.get('val_ndcg@5',   float('nan')):.4f}")
	print(f"  └─ Outputs       : {checkpoints_dir}")
	print(f"{'='*80}\n")

	return {
		"best_epoch":        best_epoch,
		"best_val_loss":     best_val_loss,
		"best_metrics":      best_metrics,
		"all_train_metrics": all_train_metrics,
		"all_val_metrics":   all_val_metrics,
		"label_dict":        label_dict,
		"checkpoints_dir":        checkpoints_dir,
	}