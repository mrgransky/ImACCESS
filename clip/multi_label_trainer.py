import json

from utils import *
from early_stopper import EarlyStopping
import clip
from clip_peft import get_injected_peft_clip, get_adapter_peft_clip
from probe import get_probe_clip
from loss import *
from evals import *
import visualize as viz

def get_text_embeddings(
	model: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	dtype: torch.dtype,
	verbose: bool = False,
):
	model.eval()
	device = next(model.parameters()).device
	all_class_embeds = []
	text_batch_size = loader.batch_size
	# This class order MUST be exactly the order used by get_validation_metrics()
	# to construct class-text embeddings and similarity-matrix columns/rows.
	try:
		class_names = loader.dataset.dataset.classes
	except AttributeError:
		try:
			class_names = loader.dataset.unique_labels
		except AttributeError as error:
			raise AttributeError(
				"Could not recover validation class names. "
				"The shared protocol can only be mapped if the class-name order "
				"corresponding to the similarity-matrix columns is available."
			) from error

	num_classes = len(class_names)

	if verbose:
		print(f"\nPre-encoding {num_classes} label(s) (batch_size: {text_batch_size})")
	with torch.inference_mode(), torch.amp.autocast(
			device_type=device.type, 
			enabled=torch.cuda.is_available(),
			dtype=dtype,
		):
			for i in range(0, num_classes, text_batch_size):
				end_idx = min(i + text_batch_size, num_classes)
				batch_class_names = class_names[i:end_idx]

				batch_class_texts = clip.tokenize(batch_class_names).to(device)
				batch_embeds = model.encode_text(batch_class_texts)
				batch_embeds = torch.nn.functional.normalize(batch_embeds.float(), dim=-1)
				all_class_embeds.append(batch_embeds.cpu()) # Move to CPU immediately to save GPU memory
				
				# Clean up
				del batch_class_texts, batch_embeds
				torch.cuda.empty_cache()
	
	all_class_embeds = torch.cat(all_class_embeds, dim=0).to(device).detach()
	
	if verbose:
		print(f"All {num_classes} classes Embeddings (frozen text encoder)")
		print(f"   ├─ {type(all_class_embeds)}")
		print(f"   ├─ {all_class_embeds.shape}")
		print(f"   ├─ {all_class_embeds.dtype}")
		print(f"   └─ {all_class_embeds.device}")

	return all_class_embeds

def print_final_evaluation_summary(
	mode: str,
	model_arch: str,
	mdl_fpth: str,
	method: Optional[str] = None,
	pw_mode: str = "log",
	actual_trained_epochs:Optional[int]=None,
	model: Optional[torch.nn.Module] = None,
	frozen_params: Optional[int] = None,
	early_stopping: Optional[Any] = None,
	eval_results: Optional[Dict[str, Any]] = None,
	model_source: Optional[str] = None,
	final_tiered_i2t: Optional[Dict[str, Any]] = None,
	final_tiered_t2i: Optional[Dict[str, Any]] = None,
	final_shared_tiered_i2t: Optional[Dict[str, Any]] = None,
	final_shared_tiered_t2i: Optional[Dict[str, Any]] = None,
	extra_lines: Optional[List[str]] = None,
	verbose: bool = True,
) -> None:
	if not verbose:
		return

	# Extract tiered metrics and source from eval_results dict if provided
	if eval_results is not None:
		if model_source is None:
			model_source = eval_results.get("model_loaded_from", "unknown")
		if final_tiered_i2t is None:
			final_tiered_i2t = eval_results.get("tiered_i2t", {})
		if final_tiered_t2i is None:
			final_tiered_t2i = eval_results.get("tiered_t2i", {})
		if final_shared_tiered_i2t is None:
			final_shared_tiered_i2t = eval_results.get("shared_tiered_i2t", {})
		if final_shared_tiered_t2i is None:
			final_shared_tiered_t2i = eval_results.get("shared_tiered_t2i", {})

	final_tiered_i2t = final_tiered_i2t or {}
	final_tiered_t2i = final_tiered_t2i or {}
	final_shared_tiered_i2t = final_shared_tiered_i2t or {}
	final_shared_tiered_t2i = final_shared_tiered_t2i or {}

	# Compute parameter count if model is passed and frozen_params wasn't explicitly supplied
	if frozen_params is None and model is not None:
		frozen_params = sum(p.numel() for p in model.parameters())

	print(f"{'='*50}")
	print(f"{mode.upper()} Final evaluation from: {model_source}")
	if method:
		print(f"  Method: {method}")
	print(f"  pw_mode: {pw_mode}")
	if frozen_params is not None:
		print(f"  {model_arch} frozen params: {frozen_params:,}")
	
	if extra_lines:
		for line in extra_lines:
			print(f"  {line}")

	if actual_trained_epochs:
		print(f"  Epochs trained: {actual_trained_epochs}")
	if early_stopping is not None and hasattr(early_stopping, "get_best_score"):
		print(f"  Best val loss: {early_stopping.get_best_score():.6f} @ Epoch {early_stopping.get_best_epoch() + 1}")
	print(f"  Best model: {mdl_fpth}")

	# print each retrieval block cleanly
	def _print_tier_block(title: str, metrics: Dict[str, Any], k: int = 10):
		k = str(k)
		if metrics:
			print(title)
			for tier, m in metrics.items():
				m_ap = m.get("mAP", {}).get(k, 0) if isinstance(m.get("mAP"), dict) else 0
				rec = m.get("Recall", {}).get(k, 0) if isinstance(m.get("Recall"), dict) else 0
				print(f"  {tier:8s} mAP@{k}={m_ap:.3f}  R@{k}={rec:.3f}")

	print(f"{'='*50}")
	_print_tier_block("[Tiered] I2T Retrieval", final_tiered_i2t)
	_print_tier_block("\n[Tiered] T2I Retrieval", final_tiered_t2i)
	_print_tier_block("\n[Shared Tiered] I2T Retrieval", final_shared_tiered_i2t)
	_print_tier_block("\n[Shared Tiered] T2I Retrieval", final_shared_tiered_t2i)
	print(f"{'='*50}")

def zero_shot_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	loss_weights: Dict[str, float] = None,
	temperature: float = 1.0, # not change ranking when it is a positive scalar
	pw_mode: str = "log",
	verbose: bool = True,
) -> Dict:
	"""
	Evaluate Zero-Shot CLIP under two explicitly separate tier protocols:

	1. Per-run/original tiers:
	   - Derived from this run's training labels via compute_loss_masks().
	   - Retained for continuity with the original analysis.
	   - Uses adaptive validation-support filtering.

	2. Shared R1-C tiers:
	   - Derived from the fixed shared_vocab_tier_spec.json.
	   - Uses the common vocabulary and common head/rare partitions shared
	     across LLM, VLM, and multimodal supervision conditions.
	   - Uses min_val_support=1; no run-specific adaptive tier alteration.

	Returns a dictionary matching evaluate_best_model()'s result structure,
	including both original and shared tiered I2T/T2I results.
	"""

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name


	# 1. Calculate parameter counts
	# just for printing purpose of frozen and trainable params:
	# Freeze all parameters for Zero-Shot Evaluation
	for param in model.parameters():
		param.requires_grad = False

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = total_params - trainable_params

	# 2. Calculate percentages
	trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0
	frozen_pct = (frozen_params / total_params) * 100 if total_params > 0 else 0

	model_arch = (
		re.sub(r"[/@]", "-", model.name)
		if hasattr(model, "name")
		else "unknown_arch"
	)
	model_name = model.__class__.__name__

	mode = inspect.stack()[0].function
	mode = re.sub(r"_multi_label", "", mode)

	if verbose:
		print(f"\n{'=' * 70}")
		print(f"{mode.upper()} | {model_name} {model_arch} | {dataset_name}")
		print(f"{'=' * 70}")

	# ── Zero-shot checkpoint: base model state, no optimisation performed ──
	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"bs_{train_loader.batch_size}_"
		f"pw_{pw_mode}_"
		f"temp_{temperature}.pth",
	)

	checkpoint = {
		"epoch": 0,
		"model_state_dict": model.state_dict(),
		"best_val_loss": None,
		"strategy": "zero_shot",
		"temperature": temperature,
		"model_architecture": model_arch,
	}
	torch.save(checkpoint, mdl_fpth)

	if verbose:
		print(f"[SAVED] {mdl_fpth}")

	per_run_masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,
		device=device,
		verbose=verbose,
	)

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=per_run_masks["active_mask"],
		head_mask=per_run_masks["head_mask"],
		rare_mask=per_run_masks["rare_mask"],
		early_stopping=None,
		checkpoint_path=None,
		finetune_strategy=None,
		device=device,
		cache_dir=results_dir,
		temperature=temperature,
		use_fixed_masks=True,
		topk_values=topk_values,
		embeddings_cache=None,
		lora_params=None,
		class_embeds_override=None,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	return best_model_eval_results

def probe_multi_label(
	model: torch.nn.Module,
	train_loader,
	validation_loader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	topk_values: List[int],
	patience: int = 10,
	min_delta: float = 1e-4,
	cumulative_delta: float = 5e-3,
	minimum_epochs: int = 20,
	loss_weights: Dict[str, float] = None,
	volatility_threshold: float = 15.0,
	slope_threshold: float = 1e-4,
	pairwise_imp_threshold: float = 1e-4,
	probe_hidden_dim: int = None, # None: Linear or MLP (if not None)
	probe_dropout: float = 0.1,
	cache_features: bool = True,
	temperature: float = 1.0,
	pw_mode: str = "log",
	verbose: bool = True,
):
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_multi_label', '', mode)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Temperature: {temperature} (raw BCE logits — scale learned by W & b)")
		print(f"   ├─ Objective  : Unidirectional I2T BCEWithLogitsLoss (closed-set; no T2I loss)")
	
	amp_dtype = torch.float32
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# FREEZE ALL CLIP PARAMETERS
	for param in model.parameters():
		param.requires_grad = False
	model.eval()

	# Compute training-set loss masks & pos_weight first so MultiLabelProbe.__init__()
	# initializes both W (zero-shot text embeddings) and b (weighted-BCE priors) in one step.
	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,
		device=device,
		verbose=verbose,
	)

	pos_weight = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask = masks["head_mask"]
	rare_mask = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	# Build probe (wraps frozen CLIP, initializes trainable W from text embeds and b from priors)
	probe = get_probe_clip(
		clip_model=model,
		validation_loader=validation_loader,
		device=torch.device(device),
		hidden_dim=probe_hidden_dim, # if activated, it creates MLP probe (Not used in our experiments)
		dropout=probe_dropout,
		zero_shot_init=True, # faster convergence
		loss_masks=masks,    # initializes bias to weighted-BCE equilibrium priors inside __init__
		verbose=verbose,
	)

	# Criteria
	# For the probe, loss is computed directly on logits [B, C] — same shape
	# as i2t_sim — so criterion_i2t with pos_weight applies directly.
	# No criterion_t2i needed: the probe has no T2I direction.
	criterion = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,
		reduction='none',
	)
	if verbose:
		print(f"\n{criterion.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# Separate weight vs bias param groups: do NOT apply weight decay to biases
	# so prior-initialized biases (esp. rare & inactive) are not pulled back toward 0.
	decay_params = [p for n, p in probe.named_parameters() if p.requires_grad and not n.endswith("bias")]
	no_decay_params = [p for n, p in probe.named_parameters() if p.requires_grad and n.endswith("bias")]
	optimizer_param_groups = [
		{"params": decay_params, "weight_decay": weight_decay},
		{"params": no_decay_params, "weight_decay": 0.0},
	]
	optimizer = torch.optim.AdamW(
		params=optimizer_param_groups,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  ├─ Weight group (wd={weight_decay}): {sum(p.numel() for p in decay_params):,} params")
		print(f"  └─ Bias group   (wd=0.0): {sum(p.numel() for p in no_decay_params):,} params")

	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")
	
	# Probe head runs in FP32 for exact gradients on small per-class BCE signals & large negative biases;
	# GradScaler is only ever needed for FP16 (never BF16 or FP32).
	use_probe_amp = False
	use_grad_scaler = use_probe_amp and torch.cuda.is_available() and (amp_dtype == torch.float16)
	scaler = torch.amp.GradScaler(
		device=device,
		enabled=use_grad_scaler,
		init_scale=2**15,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) | Probe head precision: torch.float32 (feature extraction AMP: {amp_dtype})")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ backbone_amp_dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		# f"{probe.probe_type}_" # LinearProbe or MLPProbe
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	# Feature caching
	# wrap feature extraction in torch.no_grad() to avoid building
	# a computation graph for 74K images — CLIP is frozen, gradients useless.
	mean_train_img_dir = None
	if cache_features:
		print("Pre-extracting image features (CLIP frozen — runs once)...")
		model.eval()

		def extract_features(loader, desc):
			feats, lbls = [], []

			with torch.no_grad(), torch.amp.autocast(
					device_type=torch.device(device).type, 
					enabled=torch.cuda.is_available(),
					dtype=amp_dtype,
				):
					for images, _, label_vectors in tqdm(loader, desc=desc):
						images = images.to(device, non_blocking=True)

						emb = model.encode_image(images).float()
						emb = torch.nn.functional.normalize(emb, dim=-1)

						feats.append(emb.cpu())
						lbls.append(label_vectors.cpu())

			return torch.cat(feats, dim=0), torch.cat(lbls, dim=0)
		
		train_feats, train_lbls = extract_features(train_loader, "Train features")
		val_feats, val_lbls = extract_features(validation_loader, "Val features")

		# Compute mean CLIP image direction u_xbar to monitor anti-mean collapse during training
		mean_train_img_vec = train_feats.mean(dim=0, keepdim=True)
		mean_train_img_norm = mean_train_img_vec.norm().item()
		mean_train_img_dir = torch.nn.functional.normalize(mean_train_img_vec, dim=-1).to(device)

		print(
			f"Cached — train: {train_feats.shape} ({train_feats.dtype}), "
			f"val: {val_feats.shape} ({val_feats.dtype}) | "
			f"||mean(x_train)||_2 = {mean_train_img_norm:.4f} (image cone concentration)"
		)
		
		train_feature_loader = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(train_feats, train_lbls),
			batch_size=train_loader.batch_size,
			shuffle=True,
			num_workers=0,
		)

		val_feature_loader = torch.utils.data.DataLoader(
			torch.utils.data.TensorDataset(val_feats, val_lbls),
			batch_size=validation_loader.batch_size,
			shuffle=False,
			num_workers=0,
		)
	
	training_losses = list()
	validation_losses = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		probe.train()  # only the linear head
		model.eval() # CLIP stays frozen, no gradients
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss = 0.0
		num_batches = 0
		
		# Choose data source
		data_iter = train_feature_loader if cache_features else train_loader
		
		for bidx, batch_data in enumerate(data_iter):
			if cache_features:
				# Using cached FP32 L2-normalized features
				image_embeds, label_vectors = batch_data
				image_embeds = image_embeds.to(device, non_blocking=True).float()
				label_vectors = label_vectors.to(device, non_blocking=True).float()
			else:
				images, _, label_vectors = batch_data
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				
				# Extract image embeddings (frozen)
				with torch.no_grad(), torch.amp.autocast(
					device_type=torch.device(device).type,
					enabled=torch.cuda.is_available(),
					dtype=amp_dtype,
				):
					image_embeds = model.encode_image(images).float() # [B, embed_dim]
					image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

			optimizer.zero_grad(set_to_none=True)
			
			# Run linear probe head in FP32 to avoid BF16/FP16 quantization on small per-class BCE gradients
			with torch.amp.autocast(
				device_type=torch.device(device).type, 
				enabled=use_probe_amp,
				dtype=amp_dtype,
			):
				# logits = (image_embeds @ W.T + b) / temperature
				logits = probe.probe(image_embeds) / temperature # [B, num_classes]

				# Apply pos_weight BCE, mask zero-count classes
				loss_raw = criterion(logits, label_vectors)  # [B, num_classes], reduction='none'

				if active_mask.any():
					loss = loss_raw[:, active_mask].mean()
				else:
					# Fallback to 0.0 loss with gradients enabled if no active classes exist
					loss = torch.tensor(0.0, device=device, requires_grad=True) 

			# Check for NaN loss
			if torch.isnan(loss):
				print(f"[WARNING] loss={loss} detected at epoch {epoch+1}, batch {bidx+1} => Skipping...")
				continue
			
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			if bidx % max(5, int(print_every * 0.07)) == 0 or bidx + 1 == len(data_iter):
				label_gradients = probe.weight.grad # [num_classes, embed_dim]
				bias_gradients = probe.bias.grad    # [num_classes]
				label_gradients_per_class_norm = label_gradients.norm(dim=1) # [num_classes]
				inact_w_grad_max = label_gradients_per_class_norm[~active_mask].max().item() if (~active_mask).any() else 0.0
				b_grad_abs = bias_gradients.abs() if bias_gradients is not None else torch.zeros_like(label_gradients_per_class_norm)
				print(
					f"\t\t[GRAD] batch{bidx+1:4d} W_grad_norm(active) "
					f"range: ({label_gradients_per_class_norm[active_mask].min().item():.2e}, {label_gradients_per_class_norm[active_mask].max().item():.2e}) "
					f"μ={label_gradients_per_class_norm[active_mask].mean().item():.2e} | "
					f"b_grad(active): range: ({b_grad_abs[active_mask].min().item():.2e}, {b_grad_abs[active_mask].max().item():.2e}) μ={b_grad_abs[active_mask].mean().item():.2e} | "
					f"inactive_max={inact_w_grad_max:.1e}"
				)

			if bidx % print_every == 0 or bidx + 1 == len(data_iter):
				print(f"\tBatch[{bidx+1:4d}/{len(data_iter)}] Loss: {loss.item():.6f}")

			torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			epoch_loss  += loss.item()
			num_batches += 1
		
		# Calculate average losses
		avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
		training_losses.append(avg_loss)
		learning_rates_history.append(optimizer.param_groups[0]['lr'])
		weight_decays_history.append(optimizer.param_groups[0]['weight_decay'])

		with torch.no_grad():
			W_init = probe._W_init.float()
			b_init = probe._b_init.float() if hasattr(probe, "_b_init") and probe._b_init.numel() > 0 else torch.zeros_like(probe.bias.data).float()
			W = probe.weight.data.float()
			b = probe.bias.data.float()

			W_label_norms = W.norm(dim=1)
			W_init_label_norms = W_init.norm(dim=1)
			b_drift = (b - b_init).abs()

			cos_W_vs_Winit = torch.nn.functional.cosine_similarity(W, W_init, dim=1)
			active_head_mask = head_mask & active_mask
			active_rare_mask = rare_mask & active_mask
			inactive_mask = ~active_mask

			cos_active   = cos_W_vs_Winit[active_mask]
			cos_head     = cos_W_vs_Winit[active_head_mask]
			cos_rare     = cos_W_vs_Winit[active_rare_mask]
			cos_inactive = cos_W_vs_Winit[inactive_mask]

			print(f"\n[W & b DRIFT — Epoch {epoch+1}] W_init: {tuple(W_init.shape)} W: {tuple(W.shape)} b: {tuple(b.shape)}")
			print(f"  b | ALL      (n={b.numel():5d}) (min, max): ({b.min():.4f}, {b.max():.4f}) μ±σ: {b.mean():.4f} ± {b.std():.4f}")
			print(f"  b | ACTIVE   (n={active_mask.sum().item():5d}) (min, max): ({b[active_mask].min():.4f}, {b[active_mask].max():.4f}) μ±σ: {b[active_mask].mean():.4f} ± {b[active_mask].std():.4f} | mean|Δb|={b_drift[active_mask].mean():.4f}")
			if active_head_mask.any():
				print(f"  b | HEAD     (n={active_head_mask.sum().item():5d}) (min, max): ({b[active_head_mask].min():.4f}, {b[active_head_mask].max():.4f}) μ±σ: {b[active_head_mask].mean():.4f} ± {b[active_head_mask].std():.4f} | mean|Δb|={b_drift[active_head_mask].mean():.4f}")
			if active_rare_mask.any():
				print(f"  b | RARE     (n={active_rare_mask.sum().item():5d}) (min, max): ({b[active_rare_mask].min():.4f}, {b[active_rare_mask].max():.4f}) μ±σ: {b[active_rare_mask].mean():.4f} ± {b[active_rare_mask].std():.4f} | mean|Δb|={b_drift[active_rare_mask].mean():.4f}")
			if inactive_mask.any():
				print(f"  b | INACTIVE (n={inactive_mask.sum().item():5d}) (min, max): ({b[inactive_mask].min():.4f}, {b[inactive_mask].max():.4f}) | max|Δb|={b_drift[inactive_mask].max():.2e}")
			print(f"  ||W|| | ACTIVE   (min, max): ({W_label_norms[active_mask].min():.4f}, {W_label_norms[active_mask].max():.4f}) μ±σ: {W_label_norms[active_mask].mean():.4f} ± {W_label_norms[active_mask].std():.4f}")
			if active_head_mask.any():
				print(f"  ||W|| | HEAD     (min, max): ({W_label_norms[active_head_mask].min():.4f}, {W_label_norms[active_head_mask].max():.4f}) μ±σ: {W_label_norms[active_head_mask].mean():.4f} ± {W_label_norms[active_head_mask].std():.4f}")
			if active_rare_mask.any():
				print(f"  ||W|| | RARE     (min, max): ({W_label_norms[active_rare_mask].min():.4f}, {W_label_norms[active_rare_mask].max():.4f}) μ±σ: {W_label_norms[active_rare_mask].mean():.4f} ± {W_label_norms[active_rare_mask].std():.4f}")
			if inactive_mask.any():
				print(f"  ||W|| | INACTIVE (min, max): ({W_label_norms[inactive_mask].min():.4f}, {W_label_norms[inactive_mask].max():.4f}) μ±σ: {W_label_norms[inactive_mask].mean():.4f} ± {W_label_norms[inactive_mask].std():.4f}")
			print(f"  cos(W, W_init) | ACTIVE   (n={cos_active.numel():5d}) μ±σ: {cos_active.mean():.4f} ± {cos_active.std():.4f} (min={cos_active.min():.4f})")
			if cos_head.numel() > 0:
				print(f"  cos(W, W_init) | HEAD     (n={cos_head.numel():5d}) μ±σ: {cos_head.mean():.4f} ± {cos_head.std():.4f} | <0.0: {(cos_head < 0).sum().item()} | <0.5: {(cos_head < 0.5).sum().item()}")
			if cos_rare.numel() > 0:
				print(f"  cos(W, W_init) | RARE     (n={cos_rare.numel():5d}) μ±σ: {cos_rare.mean():.4f} ± {cos_rare.std():.4f} | <0.0: {(cos_rare < 0).sum().item()} | <0.5: {(cos_rare < 0.5).sum().item()}")
			if cos_inactive.numel() > 0:
				print(f"  cos(W, W_init) | INACTIVE (n={cos_inactive.numel():5d}) μ±σ: {cos_inactive.mean():.4f} ± {cos_inactive.std():.4f}")
			print(f"  cos(W, W_init) < 0.0 : {(cos_W_vs_Winit < 0).sum().item()} / {W.shape[0]} (active: {(cos_active < 0).sum().item()} / {cos_active.numel()})")
			print(f"  cos(W, W_init) < 0.5 : {(cos_W_vs_Winit < 0.5).sum().item()} / {W.shape[0]} (active: {(cos_active < 0.5).sum().item()} / {cos_active.numel()})")

			if mean_train_img_dir is not None:
				cos_W_vs_mean_img = torch.nn.functional.cosine_similarity(W, mean_train_img_dir.expand_as(W), dim=1)
				head_anti = f"{cos_W_vs_mean_img[active_head_mask].mean():.4f}" if active_head_mask.any() else "N/A"
				rare_anti = f"{cos_W_vs_mean_img[active_rare_mask].mean():.4f}" if active_rare_mask.any() else "N/A"
				inact_anti = f"{cos_W_vs_mean_img[inactive_mask].mean():.4f}" if inactive_mask.any() else "N/A"
				print(
					f"  cos(W, u_xbar) [anti-mean check] | "
					f"HEAD: {head_anti} | "
					f"RARE: {rare_anti} | "
					f"INACTIVE: {inact_anti} "
					f"(strongly negative ≈ -0.8 indicates anti-mean collapse)"
				)

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")	

		probe.probe.eval()
		val_loss = 0.0
		val_preds_active_list, val_labels_active_list = list(), list()
		val_pos_logit_sum, val_pos_logit_sq_sum, val_pos_count, val_pos_gt0 = 0.0, 0.0, 0, 0
		val_neg_logit_sum, val_neg_logit_sq_sum, val_neg_count, val_neg_gt0 = 0.0, 0.0, 0, 0
		val_inact_fp_count, val_inact_total = 0, 0
		val_inact_logit_max = float("-inf")
		
		with torch.no_grad():
			val_iter = val_feature_loader if cache_features else validation_loader
			
			for batch_data in val_iter:
				if cache_features:
					image_embeds, label_vectors = batch_data
					image_embeds = image_embeds.to(device, non_blocking=True).float()
					label_vectors = label_vectors.to(device, non_blocking=True).float()
				else:
					images, _, label_vectors = batch_data										
					images = images.to(device, non_blocking=True)
					label_vectors = label_vectors.to(device, non_blocking=True).float()
					
					# Extract features
					with torch.amp.autocast(
						device_type=torch.device(device).type,
						enabled=torch.cuda.is_available(),
						dtype=amp_dtype,
					):
						image_embeds = model.encode_image(images).float()
						image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
				
				# Get predictions from probe [Validation] in FP32
				logits = probe.probe(image_embeds) / temperature

				loss_raw = criterion(logits, label_vectors)
				val_loss += loss_raw[:, active_mask].mean().item()

				# Track active-class predictions & logit separation diagnostics
				logits_active = logits[:, active_mask]
				labels_active = label_vectors[:, active_mask]
				preds_active = (torch.sigmoid(logits_active) > 0.5).float()
				val_preds_active_list.append(preds_active.cpu())
				val_labels_active_list.append(labels_active.cpu())

				pos_m = labels_active > 0.5
				neg_m = ~pos_m
				if pos_m.any():
					lp = logits_active[pos_m]
					val_pos_logit_sum += lp.sum().item()
					val_pos_logit_sq_sum += (lp * lp).sum().item()
					val_pos_count += lp.numel()
					val_pos_gt0 += (lp > 0).sum().item()
				if neg_m.any():
					ln = logits_active[neg_m]
					val_neg_logit_sum += ln.sum().item()
					val_neg_logit_sq_sum += (ln * ln).sum().item()
					val_neg_count += ln.numel()
					val_neg_gt0 += (ln > 0).sum().item()
				if (~active_mask).any():
					logits_inact = logits[:, ~active_mask]
					val_inact_fp_count += (logits_inact > 0).sum().item()
					val_inact_total += logits_inact.numel()
					val_inact_logit_max = max(val_inact_logit_max, logits_inact.max().item())
		
		avg_val_loss = val_loss / len(val_iter)
		validation_losses.append(avg_val_loss)

		# Calculate multi-label classification metrics on ACTIVE classes only
		val_preds = torch.cat(val_preds_active_list, dim=0)
		val_labels = torch.cat(val_labels_active_list, dim=0)
		
		hamming = hamming_loss(val_labels.numpy(), val_preds.numpy())
		f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average='weighted', zero_division=0)
		exact_match = (val_preds == val_labels).all(dim=1).float().mean().item()
		partial_match = (val_preds == val_labels).float().mean().item()

		pos_mean = val_pos_logit_sum / max(val_pos_count, 1)
		pos_std = max(0.0, (val_pos_logit_sq_sum / max(val_pos_count, 1)) - pos_mean ** 2) ** 0.5
		neg_mean = val_neg_logit_sum / max(val_neg_count, 1)
		neg_std = max(0.0, (val_neg_logit_sq_sum / max(val_neg_count, 1)) - neg_mean ** 2) ** 0.5

		epoch_metrics = {
			"val_loss":       avg_val_loss,
			"hamming_loss":   hamming,
			"f1_score":       f1,
			"exact_match_acc": exact_match,
			"partial_acc":    partial_match,
		}

		full_val_loss_acc_metrics_all_epochs.append(epoch_metrics)

		if cache_features:
			with torch.no_grad():
				sample_feats = val_feats[:512].to(device).float()     # [512, 768] val image features
				sample_labels = val_lbls[:512].to(device).float()     # [512, C] multi-hot
				
				W_norm = torch.nn.functional.normalize(probe.weight.float(), dim=-1) # [C, emb_dim]
				matched_class_vecs = torch.zeros(sample_feats.shape[0], W_norm.shape[1], device=device)

				for i in range(sample_feats.shape[0]):
					pos_idx = sample_labels[i].nonzero(as_tuple=True)[0]
					if pos_idx.numel() > 0:
						matched_class_vecs[i] = W_norm[pos_idx].mean(dim=0)
				
				matched_class_vecs = torch.nn.functional.normalize(matched_class_vecs, dim=-1)
				sample_feats_norm = torch.nn.functional.normalize(sample_feats, dim=-1)
				cos_sim = torch.nn.functional.cosine_similarity(
					sample_feats_norm, 
					matched_class_vecs, 
					dim=1
				).mean().item()

				# Quick probe-logit Top-5 hit rate on active classes (tracks actual classifier ranking)
				sample_logits = (probe.probe(sample_feats) / temperature)
				sample_logits[:, ~active_mask] = float("-inf")
				eff_k5 = min(5, max(int(active_mask.sum().item()), 1))
				top5_idx = sample_logits.topk(eff_k5, dim=1).indices
				top5_mask = torch.zeros_like(sample_logits, dtype=torch.bool).scatter_(1, top5_idx, True)
				val_sample_hit5 = (top5_mask & sample_labels.bool()).any(dim=1).float().mean().item()
		else:
			cos_sim = None
			val_sample_hit5 = None
		
		print(
			f"\nEpoch {epoch+1}:\n"
			f"  [LOSS] {mode.upper()} — Train: {avg_loss:.6f} Val: {avg_val_loss:.6f}\n"
			f"  [VAL ACTIVE ({active_mask.sum().item():,} cls)] "
			f"Hamming: {hamming:.6f} | Weighted F1: {f1:.4f} | ExactMatch: {exact_match:.4f} | PartialAcc: {partial_match:.6f}\n"
			f"  [VAL LOGITS] pos(y=1): μ±σ={pos_mean:+.3f}±{pos_std:.3f} (frac>0: {val_pos_gt0/max(val_pos_count,1):.3f}) | "
			f"neg(y=0,active): μ±σ={neg_mean:+.3f}±{neg_std:.3f} (frac>0: {val_neg_gt0/max(val_neg_count,1):.5f})\n"
			f"  [VAL INACTIVE] FP(>0): {val_inact_fp_count}/{val_inact_total} | max_logit: {val_inact_logit_max:.3f}\n"
			f"  LR: {scheduler.get_last_lr()}"
		)

		if cos_sim is not None:
			print(f"  Val CosSim (W): {cos_sim:.4f} | Val Probe-Logit Hit@5 (512-sample): {val_sample_hit5:.4f}")
		
		if early_stopping.should_stop(
			current_value=avg_val_loss,
			model=probe, # saves probe head {weight, bias}
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f} sec")
	
	print(f"Total Training Time: {time.time() - train_start_time:.1f} sec")
	
	# Final evaluation
	# Pass probe (not model) with use_probe_logits=True so evaluate_best_model()
	# reloads the best checkpoint first and then scores exact trained probe logits
	# (x @ W.T + b) / temperature — preserving learned weight norms and biases,
	# and masking untrained inactive classes in Alignment Score & full-set metrics.
	best_model_eval_results = evaluate_best_model(
		model=probe, # wraps frozen CLIP + trained linear head (W, b)
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		temperature=temperature,
		class_embeds_override=None,
		use_probe_logits=True,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	# Update model path
	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	probe_trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		extra_lines=[
			f"Probe trainable params: {probe_trainable_params:,} ({probe.probe_type}, temp={temperature})",
			"Scoring: exact probe logits (x @ W.T + b) / temp [T2I = closed-set logit transpose, not open-vocab]",
		],
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		# f"{probe.probe_type}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}"
	)
	
	# Plotting
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}
	
	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)
	
	return best_model_eval_results

def full_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
	pw_mode: str = "log",
	verbose: bool=True,
):
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
		
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"{mode.upper()}-FT")
		print(f"  ├─ {model_name} {model_arch}")
		print(f"  ├─ {dataset_name}")
		print(f"  ├─ Epochs: {num_epochs}  Batch size: {train_loader.batch_size}  Device: {type(device)} {device}")
		print(f"  ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}  Patience: {patience}")
		print(f"  ├─ Loss weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		print(f"  ├─ Temperature: {temperature}")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		# check with cuda capability
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16

		if verbose:
			print(f"  └─ {gpu_name} {total_mem:.2f}GB VRAM cuda capability: {cuda_capability}")
	
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]

	if non_zero_dropouts:
		print(f"\nNon-zero dropout detected in base {model_name} {model_arch} during {mode}:")
		print(non_zero_dropouts)
		print()

	# Freeze all parameters first
	for param in model.parameters():
		param.requires_grad = False
	
	# Unfreeze only vision encoder parameters
	for param in model.visual.parameters():
		param.requires_grad = True

	print("\nModel parameters before training:")
	for n, p in model.named_parameters():
		print(f"{n:<60}{p.requires_grad:<5}{p.dtype}\t{p.shape}")
	print("="*130)

	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,
		# pw_max_cap=100.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N 					= masks["N"]
	train_freq 	= masks["train_freq"]

	# sys.exit()
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none',)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# Optimizer
	full_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=full_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in full_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 2048 (Conservative start)
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdelta_{min_delta:.1e}_"
		f"cdelta_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	
	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True)#.float()
			
			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx + 1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f} T2I: {batch_loss_t2i:.6f})"
				)

		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

		# clear cache before validation
		torch.cuda.empty_cache()
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Train — Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		topk_values=topk_values,
		temperature=temperature,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)
	
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)
	
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	return best_model_eval_results

def lora_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	lora_rank: int,
	lora_alpha: float,
	lora_dropout: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	loss_weights: Dict[str, float] = None,
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	pw_mode: str = "log", # orig: sqrt
	verbose: bool = True,
):
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# ── Dropout check 
	non_zero_dropouts = [
			(name, module.p)
			for name, module in model.named_modules()
			if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts:
			dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
			assert False, (
					f"Non-zero dropout in base model during LoRA: {dropout_info}\n"
					"Set dropout=0.0 in clip.load() before LoRA injection."
			)
	
	mode = re.sub(r'_finetune_multi_label', '', inspect.stack()[0].function)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	if verbose:
		print(f"\n{mode.upper()}")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		# check with cuda capability
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		
		if verbose:
			print(f"   └─ {gpu_name} | {total_mem:.1f}GB VRAM | cuda capability: {cuda_capability}")
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	# LoRA injection — vision encoder only
	# Text encoder stays frozen and un-injected, consistent with full fine-tuning.
	# all_class_embeds pre-computed from frozen text encoder remains valid.
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # no LoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode, # orig: sqrt
		# pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N 					= masks["N"]
	train_freq 	= masks["train_freq"]

	# sys.exit()

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")
	
	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# Optimizer — LoRA parameters only
	lora_params = [p for p in model.parameters() if p.requires_grad]	
	optimizer = torch.optim.AdamW(
		params=lora_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in lora_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 2048 (Conservative start)
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_loa_{lora_alpha}_lod_{lora_dropout}_"
		f"temp_{temperature}_bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_pat_{patience}_"
		f"mdt_{min_delta:.1e}_cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}.pth"
	)

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss_total = epoch_loss_i2t = epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			optimizer.zero_grad(set_to_none=True)
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad], 
				max_norm=1.0
			)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			epoch_loss_total += total_loss.item()
			epoch_loss_i2t   += loss_i2t.item()
			epoch_loss_t2i   += loss_t2i.item()
			num_batches += 1
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\tBatch {bidx+1}/{len(train_loader)}")
				print(
					f"\t\t[LOSS] Total: {total_loss.item():.6f} "
					f"(I2T: {loss_i2t.item():.6f} T2I: {loss_t2i.item():.6f})"
				)
				b_norms = [
					p.data.norm().item()
					for n, p in model.named_parameters()
					if p.requires_grad and "lora_B" in n
				]
				if b_norms:
					b_norms_t = torch.tensor(b_norms)
					print(
						f"\t\t[NORM] lora_B: "
						f"(min, max): ({b_norms_t.min():.4f}, {b_norms_t.max():.4f}) "
						f"μ±σ: {b_norms_t.mean():.4f} ± {b_norms_t.std():.4f}"
					)

		avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total)
		training_losses_breakdown["total"].append(avg_total)
		training_losses_breakdown["i2t"].append(avg_i2t)
		training_losses_breakdown["t2i"].append(avg_t2i)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_metrics = validation_results["full_metrics"]
		img2txt_metrics  = validation_results["img2txt_metrics"]
		txt2img_metrics  = validation_results["txt2img_metrics"]
		full_val_loss_acc_metrics_all_epochs.append(full_val_metrics)
		cos_sim = full_val_metrics.get("cosine_similarity")
		align_score = full_val_metrics.get("alignment_score")
		img2txt_metrics_all_epochs.append(img2txt_metrics)
		txt2img_metrics_all_epochs.append(txt2img_metrics)
		i2t_map10 = img2txt_metrics.get("mAP", {}).get("10", float("nan"))
		t2i_map10 = txt2img_metrics.get("mAP", {}).get("10", float("nan"))
		i2t_r10 = img2txt_metrics.get("Recall", {}).get("10", float("nan"))
		t2i_r10 = txt2img_metrics.get("Recall", {}).get("10", float("nan"))
		print(
			f"\nEpoch {epoch+1}:\n"
			f"  [LOSS] — {mode.upper()}-FT Train: {avg_total:.4f} (I2T: {avg_i2t}, T2I: {avg_t2i}) Val: {current_val_loss}\n"
			f"  I2T    — mAP@10: {i2t_map10:.4f}  R@10: {i2t_r10:.4f}\n"
			f"  T2I    — mAP@10: {t2i_map10:.4f}  R@10: {t2i_r10:.4f}\n"
			f"  LR     — {scheduler.get_last_lr()[0]}"
		)
		if align_score is not None:
			print(f"  Embed — AlignScore@5: {align_score:.4f}")
		elif cos_sim is not None:
			print(f"  Embed — CosSim: {cos_sim:.4f}")
		else:
			print(f"  Embed — AlignScore: N/A")
		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break
		if early_stopping.should_stop(
				current_value=current_val_loss,
				model=model,
				epoch=epoch,
				optimizer=optimizer,
				scheduler=scheduler,
				checkpoint_path=mdl_fpth,
		):
				print(
					f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
					f"@ epoch {early_stopping.get_best_epoch()+1}"
				)
				break
		
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total elapsed: {time.time()-train_start_time:.1f}s")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_temp_{temperature}_"
		f"bs_{train_loader.batch_size}_lor_{lora_rank}_loa_{lora_alpha}_lod_{lora_dropout}"
	)

	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def lora_plus_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	lora_rank: int,
	lora_alpha: float,
	lora_dropout: float,
	lora_plus_lambda: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	quantization_bits: int=8,
	quantized: bool=False,
	loss_weights: Dict[str, float]=None,
	temperature: float = 0.07,
	pw_mode: str = "log", # orig: "sqrt"
	verbose: bool=False,
):
	"""
	LoRA+ fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label LoRA+ version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	6. Differential learning rates for LoRA A (base) and LoRA B (λ * base)
	
	Args:
		model: CLIP model to fine-tune with LoRA+
		train_loader: Training torch.utils.data.DataLoader (must provide multi-label vectors)
		validation_loader: Validation torch.utils.data.DataLoader  
		num_epochs: Number of training epochs
		print_every: Print loss every N batches
		learning_rate: Base learning rate for LoRA A parameters
		weight_decay: Weight decay for regularization
		device: Training device (cuda/cpu)
		results_dir: Directory to save results
		lora_rank: LoRA rank parameter
		lora_alpha: LoRA alpha parameter
		lora_dropout: LoRA dropout parameter
		patience: Early stopping patience
		min_delta: Minimum change for improvement
		cumulative_delta: Cumulative delta for early stopping
		minimum_epochs: Minimum epochs before early stopping
		volatility_threshold: Threshold for validation loss volatility
		slope_threshold: Threshold for validation loss slope
		pairwise_imp_threshold: Threshold for pairwise improvement
		topk_values: K values for evaluation metrics
		quantization_bits: Bits for quantization (4 or 8)
		lora_plus_lambda: Learning rate multiplier for LoRA B parameters
		quantized: Whether to use quantized base weights (QLoRA+)
		loss_weights: Optional weights for I2T and T2I losses
		temperature: Temperature scaling for similarities
		verbose: Enable detailed logging
	"""	
	window_size = minimum_epochs + 1	
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Check for non-zero dropout in the base model
	dropout_values = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA+ fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA+ fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)
	
	# Dataset and model setup
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)
	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	if verbose:
		print(f"\n{mode.upper()} {model_name} {model_arch}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ Batch size: {train_loader.batch_size}  Device: {type(device)} {device}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")
		
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability(device)
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   ├─ {gpu_name} {device}")
			print(f"   ├─ AMP dtype: {amp_dtype}")
			print(f"   ├─ Total Memory: {gpu_total_mem:.2f}GB")
			print(f"   └─ CUDA Capability: {cuda_capability}")
	
	# Apply LoRA+ to the model
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		lora_plus_lambda=lora_plus_lambda,
		target_text_modules=[], # no LoRA+ in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	# Separate LoRA A and B parameters for differential LR
	lora_A_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n]
	lora_B_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_B" in n]

	# Sanity check — nothing should fall through
	all_lora_params = set(id(p) for p in lora_A_params + lora_B_params)
	ungrouped = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in all_lora_params]
	if ungrouped:
		raise ValueError(f"LoRA+: trainable params not in lora_A or lora_B groups: {ungrouped}")

	if verbose:
		print(f"\n{mode.upper()} Parameter Groups:")
		print(f"  ├─ lora_A params: {len(lora_A_params)} tensors")
		print(f"  ├─ lora_B params: {len(lora_B_params)} tensors")
		print(f"  ├─ All lora params: {len(all_lora_params)} tensors")
		print(f"  ├─ Unassigned params: {len(ungrouped)} tensors")
		print(f"  └─ LR multiplier (λ): {lora_plus_lambda}")
	
	# Optimizer with differential learning rates (Hayou et al., ICML 2024: symmetric weight decay on A and B)
	lora_A_lr = learning_rate
	lora_A_wd = weight_decay

	lora_B_lr = learning_rate * lora_plus_lambda
	lora_B_wd = weight_decay
	lora_plus_params = [
		{
			'params': lora_A_params,
			'lr': lora_A_lr,
			'weight_decay': lora_A_wd,
		},
		{
			'params': lora_B_params,
			'lr': lora_B_lr,
			'weight_decay': lora_B_wd,
		},
	]

	optimizer = torch.optim.AdamW(
		params=lora_plus_params,
		betas=(0.9, 0.98),
		eps=1e-6,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ LR: lora_A = {lora_A_lr} lora_B = {lora_B_lr} (λ={lora_plus_lambda})")
		print(f"  ├─ WD: lora_A = {lora_A_wd} lora_B = {lora_B_wd}")
		print(f"  └─ Params: lora_A: {sum(p.numel() for p in lora_A_params):,} lora_B: {sum(p.numel() for p in lora_B_params):,}")

		get_parameters_info(
			model=model, 
			mode=mode, 
			optimizer=optimizer, 
			verbose=verbose
		)

		get_parameters_info_orig(model=model, mode=mode)

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	is_ampere_or_newer = cuda_capability[0] >= 8

	if is_ampere_or_newer:
		B_MAX_NORM = 100.0
		amp_enabled = True
		scaler = torch.amp.GradScaler(
			device=device,
			init_scale=2**15,
			growth_factor=2.0,
			backoff_factor=0.5,
			growth_interval=2000,
		)
	else:
		B_MAX_NORM = 25.0
		amp_enabled = False  # AMP disabled — FP16 on V100 is unstable for ViT-L LoRA+
		scaler = None

	if verbose:
		print(f"\nAMP Configuration:")
		print(f"  ├─ Enabled: {amp_enabled}")
		print(f"  ├─ dtype: {amp_dtype if amp_enabled else 'FP32 (AMP disabled)'}")
		print(f"  ├─ B_MAX_NORM: {B_MAX_NORM}")
		if scaler is not None:
			scaler_state = scaler.state_dict()
			print(f"  ├─ GradScaler init_scale: {scaler_state.get('scale', 'N/A')}")
			print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
			print(f"  └─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		else:
			print(f"  └─ GradScaler: disabled")

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode, #orig: "sqrt",
		# pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N 					= masks["N"]
	train_freq 	= masks["train_freq"]

	# sys.exit()

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none',)

	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_A_{lora_A_lr:.1e}_"
		f"wd_A_{lora_A_wd:.1e}_"
		f"lr_B_{lora_B_lr:.1e}_"
		f"wd_B_{lora_B_wd}_"
		f"B_norm_max_{B_MAX_NORM}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"lmbd_{lora_plus_lambda}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	
	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()
	
	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data
			
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			if torch.isnan(images).any():
				if verbose:
					print(f"[WARNING] Corrupted image detected in batch {bidx+1}. Skipping.")
				continue

			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(
				device_type=device.type,
				enabled=amp_enabled,#torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				# Multi-label contrastive loss computation
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch
			
			# Backward Pass
			if scaler is not None:
				scaler.scale(total_loss).backward()
			else:
				total_loss.backward()
			
			# Gradient Clipping (Vital for LoRA+)
			if scaler is not None:
				scaler.unscale_(optimizer)

			grad_norm = torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad], 
				max_norm=1.0
			)

			# Grad norm check — post-clipping
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\tBatch {bidx+1}/{len(train_loader)}")
				grad_norms_A, grad_norms_B = [], []
				for name, param in model.named_parameters():
					if param.requires_grad and param.grad is not None:
						gn = param.grad.norm().item()
						if "lora_A" in name:
							grad_norms_A.append(gn)
						elif "lora_B" in name:
							grad_norms_B.append(gn)
				
				if grad_norms_A and grad_norms_B:
					print(
						f"\t\t[GRAD] (min, max) "
						f"A: ({min(grad_norms_A):.6f}, {max(grad_norms_A):.6f}) "
						f"B: ({min(grad_norms_B):.6f}, {max(grad_norms_B):.6f})"
					)

			# Guard: skip optimizer step if grads are still corrupt post-unscale
			# fires only on FP16 (V100 and older) due to overflow.
			# On BF16/Ampere this is defensive dead code — kept for correctness.
			if torch.isnan(grad_norm) or torch.isinf(grad_norm):
				print(f"[WARNING] Corrupt grad_norm at epoch {epoch+1} batch {bidx+1}: {grad_norm} | isnan(): {torch.isnan(grad_norm)} | isinf(): {torch.isinf(grad_norm)}")

				# purge Adam state for any param whose grad was NaN/Inf BEFORE clearing grads
				for group in optimizer.param_groups:
					for p in group['params']:
						if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
							if p in optimizer.state:
								optimizer.state[p] = {}

				optimizer.zero_grad(set_to_none=True)

				if scaler is not None:
					if cuda_capability[0] < 8:
						if hasattr(scaler, '_scale') and scaler._scale is not None:
							# scaler._scale.mul_(0.5)
							current_scale = scaler._scale.item()
							new_scale = max(current_scale * 0.5, 1.0)  # never go below 1.0
							scaler._scale.fill_(new_scale)

					scaler.update()  # must still call to reset internal state

					if verbose:
						print(f"\t\t[Scaler] scale after corrupt grad recovery={scaler.get_scale():.1f}")

				continue

			# Step the optimizer (Generic for Scaler or None)
			if scaler is not None:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()

			# Clip B matrix norms (magnitude control)
			clipped_count = 0
			clipped_details = []
			with torch.no_grad():
				for name, param in model.named_parameters():
					if param.requires_grad and "lora_B" in name:
						norm = param.data.norm()
						if norm > B_MAX_NORM:
							clipped_count += 1
							clipped_details.append((name, norm.item()))
							param.data.mul_(B_MAX_NORM / norm)

			# Only log clipping details at print_every intervals, not every batch
			if clipped_count > 0:
				print(f"\t\t[BATCH {bidx+1} B-NORM] « {clipped_count} layer(s) » clipped:")
				for cname, cnorm in clipped_details:
					print(f"\t\t\t{cname:<80}{cnorm:.6f} → {B_MAX_NORM}")

			scheduler.step()

			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\t[LOSS] Total: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
				)
		
		# Detect full NaN cascade — entire epoch was skipped
		if num_batches == 0:
			print(f"[CRITICAL] Epoch {epoch+1}: zero valid batches — full NaN cascade detected.")
			# Clear corrupted Adam momentum buffers
			for group in optimizer.param_groups:
				for p in group['params']:
					if p in optimizer.state:
						optimizer.state[p] = {}

			# Restore best weights if available
			if early_stopping.best_weights is not None:
				early_stopping._restore_best_weights(model)
				print(f"  Restored weights from epoch {early_stopping.get_best_epoch()+1}.")
			else:
				print(f"  No checkpoint available. Aborting.")

			break

		# average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)
		
		# Track learning rates (now we have multiple parameter groups)
		# for i, g in enumerate(optimizer.param_groups):
		# (['params', 'lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay', 'initial_lr'])
		# 	print(i, g.keys())

		learning_rates_history.append([group['lr'] for group in optimizer.param_groups])
		weight_decays_history.append([group['weight_decay'] for group in optimizer.param_groups])
		
		if verbose:
			print(f"{len(learning_rates_history[-1])} LR groups: {learning_rates_history[-1]}")
			print(f"{len(weight_decays_history[-1])} WD groups: {weight_decays_history[-1]}")
		
		# Weight health check before validation
		healthy, A_norms, B_norms = check_lora_weight_health(
			model=model, 
			optimizer=optimizer,
			verbose=verbose,
		)

		if not healthy:
			# Clear corrupted Adam states first
			for group in optimizer.param_groups:
					for p in group['params']:
							if p in optimizer.state:
									optimizer.state[p] = {}
			# Then restore weights
			if early_stopping.best_weights is not None:
					early_stopping._restore_best_weights(model)
			break

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		# empty cache:
		torch.cuda.empty_cache()

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
				"lora_plus_lambda": lora_plus_lambda,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Train - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break


		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break
		
		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Elapsed Time: {time.time() - train_start_time:.1f}s")
	
	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
			"lora_plus_lambda": lora_plus_lambda,
		},
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)
	
	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		extra_lines=[
			f"Rank: {lora_rank} Alpha: {lora_alpha} Dropout: {lora_dropout} Lambda: {lora_plus_lambda} B_MAX_NORM: {B_MAX_NORM}"
		],
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_A_{lora_A_lr:.1e}_"
		f"wd_A_{lora_A_wd:.1e}_"
		f"lmbd_{lora_plus_lambda}_"
		f"lr_B_{lora_B_lr:.1e}_"
		f"wd_B_{lora_B_wd}_"
		f"B_norm_max_{B_MAX_NORM}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}"
	)
	
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}
	
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)
	
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)
	
	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	return best_model_eval_results

def rslora_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	lora_rank: int,
	lora_alpha: float,
	lora_dropout: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	loss_weights: Dict[str, float] = None,
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	pw_mode: str = "log", # orig: "sqrt"
	verbose: bool = True,
):
	"""
	rsLoRA multi-label fine-tuning for CLIP.

	Identical to lora_finetune_multi_label() in every respect except that
	LoRA's standard scaling  α/r  is replaced by the rank-stabilised variant
		α / √r   (rsLoRA, Kalajdzievski 2023)

	This keeps gradient norms bounded as rank grows, which is particularly
	useful when sweeping lora_rank or when training at higher ranks (r ≥ 16)
	where standard LoRA can exhibit loss spikes or slower convergence.

	All parameters are identical to lora_finetune_multi_label(); the only
	internal difference is passing method="rslora" to get_injected_peft_clip(),
	which sets self.scale = alpha / (rank ** 0.5) inside LoRALinear.
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# ── Dropout check
	non_zero_dropouts = [
		(name, module.p)
		for name, module in model.named_modules()
		if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
		assert False, (
			f"Non-zero dropout in base model during rsLoRA: {dropout_info}\n"
			"Set dropout=0.0 in clip.load() before rsLoRA injection."
		)

	mode = re.sub(r'_finetune_multi_label', '', inspect.stack()[0].function)  # → "rslora"
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	if verbose:
		print(f"\n{mode.upper()}")
		print(f"   ├─ {dataset_name} | {model_name} {model_arch}")
		print(f"   ├─ Rank           : {lora_rank}")
		print(f"   ├─ Alpha (input)  : {lora_alpha}")
		print(f"   ├─ Dropout        : {lora_dropout}")
		print(f"   ├─ Batch size     : {train_loader.batch_size}")
		print(f"   ├─ Device         : {type(device)} {device}")
		print(f"   ├─ Temperature    : {temperature}")
		print(f"   ├─ Loss Weights   : I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[],               # no rsLoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	
	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode, # orig: "sqrt",
		# pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N           = masks["N"]
	train_freq  = masks["train_freq"]

	# Criteria
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item()}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item()}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min()}, {train_freq.max()}]")

	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none')
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	#  Optimizer — rsLoRA parameters only 
	rslora_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=rslora_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in rslora_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 32768, default 
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_pat_{patience}_"
		f"mdt_{min_delta:.1e}_cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}.pth"
	)

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss_total = epoch_loss_i2t = epoch_loss_t2i = 0.0
		num_batches = 0
		nan_loss_count = 0      # track NaN loss skips
		nan_grad_count = 0      # track NaN gradient skips

		for bidx, batch_data in enumerate(train_loader):
			images, _, label_vectors = batch_data
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				nan_loss_count += 1
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				
				# Abort epoch if NaN cascade is unrecoverable
				if nan_loss_count > len(train_loader) // 2:
					raise RuntimeError(
						f"[ABORT] NaN loss in >{len(train_loader)//2} batches "
						f"All {len(train_loader)} batches produced NaN/Inf loss. "
						f"Reduce learning_rate or lora_alpha and rerun."
					)
				
				continue

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)

			# Check for NaN/Inf gradients after unscaling
			has_bad_grad = any(
				not torch.isfinite(p.grad).all()
				for p in model.parameters()
				if p.requires_grad and p.grad is not None
			)
			
			if has_bad_grad:
				nan_grad_count += 1
				if nan_grad_count % 50 == 1:
					print(f"Warning: NaN/Inf gradient: e{epoch+1} b{bidx+1} (total skipped: {nan_grad_count}). Skipping step.")
				optimizer.zero_grad(set_to_none=True)
				scaler.update()
				continue
			
			torch.nn.utils.clip_grad_norm_(
				[p for p in model.parameters() if p.requires_grad],
				max_norm=1.0,
			)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			epoch_loss_total += total_loss.item()
			epoch_loss_i2t   += loss_i2t.item()
			epoch_loss_t2i   += loss_t2i.item()
			num_batches += 1

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx+1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {total_loss.item():.6f} "
					f"(I2T: {loss_i2t.item():.6f} T2I: {loss_t2i.item():.6f})"
				)

		if num_batches == 0:
			raise RuntimeError(
				f"[ABORT] Epoch {epoch+1}: zero valid batches. "
				f"All {len(train_loader)} batches produced NaN/Inf loss. "
				f"Reduce learning_rate or lora_alpha and rerun."
			)

		# Report skips at end of epoch
		if nan_loss_count > 0 or nan_grad_count > 0:
			print(
				f"[Epoch {epoch+1}] Skipped batches — "
				f"NaN loss: {nan_loss_count}, NaN grad: {nan_grad_count} "
				f"/ {len(train_loader)} total"
			)

		avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0
		training_losses.append(avg_total)
		training_losses_breakdown["total"].append(avg_total)
		training_losses_breakdown["i2t"].append(avg_i2t)
		training_losses_breakdown["t2i"].append(avg_t2i)
		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)

		full_val_metrics = validation_results["full_metrics"]
		img2txt_metrics  = validation_results["img2txt_metrics"]
		txt2img_metrics  = validation_results["txt2img_metrics"]
		full_val_loss_acc_metrics_all_epochs.append(full_val_metrics)
		img2txt_metrics_all_epochs.append(img2txt_metrics)
		txt2img_metrics_all_epochs.append(txt2img_metrics)

		i2t_map10 = img2txt_metrics.get("mAP", {}).get("10", float("nan"))
		t2i_map10 = txt2img_metrics.get("mAP", {}).get("10", float("nan"))
		i2t_r10   = img2txt_metrics.get("Recall", {}).get("10", float("nan"))
		t2i_r10   = txt2img_metrics.get("Recall", {}).get("10", float("nan"))
		cos_sim   = full_val_metrics.get("cosine_similarity")
		align_score = full_val_metrics.get("alignment_score")
		print(
			f"\nEpoch {epoch+1}:\n"
			f"  [LOSS] — {mode.upper()}-FT Train: {avg_total} (I2T: {avg_i2t}, T2I: {avg_t2i}) Val: {current_val_loss}\n"
			f"  I2T    — mAP@10: {i2t_map10:.4f}  R@10: {i2t_r10:.4f}\n"
			f"  T2I    — mAP@10: {t2i_map10:.4f}  R@10: {t2i_r10:.4f}\n"
			f"  LR     — {scheduler.get_last_lr()[0]}"
		)
		if align_score is not None:
			print(f"  Embed — AlignScore@5: {align_score:.4f}")
		elif cos_sim is not None:
			print(f"  Embed — CosSim: {cos_sim:.4f}")
		else:
			print(f"  Embed — AlignScore: N/A")


		# ── Training health check ────────────────────────────────────────────
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{mode}] Total elapsed: {time.time()-train_start_time:.1f}s")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	file_base_name = (
		f"{mode}_"
		f"{model_arch}_ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)

	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def dora_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	lora_rank: int,
	lora_alpha: float,
	lora_dropout: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	quantization_bits: int = 8,
	quantized: bool = False,
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,
	pw_mode: str = "log",
	verbose: bool = True,
):
	"""
	DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning for multi-label datasets.
	
	DoRA decomposes pre-trained weights into magnitude and direction components,
	applying LoRA to the directional component while keeping magnitude trainable.
	This improves learning capacity and stability without inference overhead.
	
	Reference: DoRA: Weight-Decomposed Low-Rank Adaptation, ICML'24
	"""
	
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	window_size = minimum_epochs + 1

	# Validate dropout configuration
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during DoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for DoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during DoRA fine-tuning "
			"which gives you more control over DoRA-specific regularization without affecting the base model.\n"
		)

	# Early stopping setup
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha}")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Inject DoRA into model
	model = get_injected_peft_clip(
		clip_model=model,
		method=mode, # "dora"
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # no DoRA in text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode, # orig: "sqrt",
		# pw_max_cap=50.0,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N 					= masks["N"]
	train_freq 	= masks["train_freq"]

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)

	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none',)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# Optimizer setup: exclude 1D DoRA magnitude vectors from weight decay (only decay 2D LoRA A/B matrices)
	dora_matrix_params = [p for n, p in model.named_parameters() if p.requires_grad and "magnitude" not in n]
	dora_magnitude_params = [p for n, p in model.named_parameters() if p.requires_grad and "magnitude" in n]
	dora_params = dora_matrix_params + dora_magnitude_params
	optimizer = torch.optim.AdamW(
		params=[
			{"params": dora_matrix_params, "weight_decay": weight_decay},
			{"params": dora_magnitude_params, "weight_decay": 0.0},
		],
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in dora_params):,} (matrices: {sum(p.numel() for p in dora_matrix_params):,} wd={weight_decay} | magnitude: {sum(p.numel() for p in dora_magnitude_params):,} wd=0.0)")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  └─ Eps: {optimizer.defaults['eps']}")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 2048 (Conservative start)
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	# Model checkpoint path
	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, (images, _, label_vectors) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True)

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx + 1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {total_loss.item():.7f} "
					f"(I2T: {loss_i2t.item():.7f} T2I: {loss_t2i.item():.7f})"
				)
			
			epoch_loss_total += total_loss.item()
			epoch_loss_i2t += loss_i2t.item()
			epoch_loss_t2i += loss_t2i.item()
			num_batches += 1
		
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)
		
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode.upper()}: Train - Total: {avg_total_loss} (I2T: {avg_i2t_loss}, T2I: {avg_t2i_loss}) Val: {current_val_loss}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')


		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = train_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Train Cache: {cache_stats}")

		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = validation_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Validation Cache: {cache_stats}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Training Elapsed Time: {time.time() - train_start_time:.1f}s")

	# Final evaluation
	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}"
	)
	
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	# Plot loss breakdown (I2T vs T2I)
	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def vera_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	lora_rank: int,
	lora_alpha: float,
	lora_dropout: float,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	pw_mode: str = "log", # orig: log
	verbose: bool = False,
):
	"""
	VeRA fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label VeRA version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Key differences from LoRA multi-label version:
	1. Uses VeRA adaptation (frozen random matrices + trainable scaling vectors)
	2. Significantly fewer trainable parameters than LoRA
	3. Shared random matrices across all layers
	
	Args:
		model: CLIP model to fine-tune with VeRA
		train_loader: Training torch.utils.data.DataLoader (must provide multi-label vectors)
		validation_loader: Validation torch.utils.data.DataLoader  
		num_epochs: Number of training epochs
		print_every: Print loss every N batches
		learning_rate: Learning rate for VeRA parameters
		weight_decay: Weight decay for regularization
		device: Training device (cuda/cpu)
		results_dir: Directory to save results
		lora_rank: VeRA rank parameter (for compatibility)
		lora_alpha: VeRA alpha parameter (for compatibility, not used in VeRA)
		lora_dropout: VeRA dropout parameter
		patience: Early stopping patience
		min_delta: Minimum change for improvement
		cumulative_delta: Cumulative delta for early stopping
		minimum_epochs: Minimum epochs before early stopping
		volatility_threshold: Volatility threshold for early stopping
		slope_threshold: Slope threshold for early stopping
		pairwise_imp_threshold: Pairwise improvement threshold
		topk_values: K values for evaluation metrics
		loss_weights: Optional weights for I2T and T2I losses
		temperature: Temperature scaling for similarities
		quantization_bits: Quantization bits (4 or 8)
		quantized: Whether to use quantized base weights
		verbose: Print detailed information
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for VeRA
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during VeRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for VeRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during VeRA fine-tuning "
			"which gives you more control over VeRA-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ Rank: {lora_rank}")
		print(f"   ├─ Alpha: {lora_alpha} (not used in VeRA)")
		print(f"   ├─ Dropout: {lora_dropout}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=lora_rank,
		alpha=lora_alpha,
		dropout=lora_dropout,
		target_text_modules=[], # Frozen text encoder
		target_vision_modules=["in_proj", "out_proj", "c_fc", "c_proj"],
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)

	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,# orig: "log",
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask = masks["head_mask"]
	rare_mask = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(
		reduction='none',
	)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")

	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# Optimizer: 1D VeRA scaling vectors (lambda_d initialized to 1.0, lambda_b to 0.0) must not be decayed toward 0
	vera_params = [p for p in model.parameters() if p.requires_grad]

	optimizer = torch.optim.AdamW(
		params=vera_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=0.0,
	)

	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in vera_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: 0.0 (disabled for 1D VeRA scaling vectors; CLI weight_decay={weight_decay})")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 2048 (Conservative start)
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{'quantized_' + str(quantization_bits) + 'bit_' if quantized else ''}"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)

			images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
						
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch
			
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx + 1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f} T2I: {batch_loss_t2i:.6f})"
				)
		
		# Calculate average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")
		
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode.upper()}-FT: Training - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Validation: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')
		# print(f"   ├─ [VeRA SCALING VECTORS]:")
		# for name, module in model.named_modules():
		# 	if hasattr(module, 'lambda_b') and hasattr(module, 'lambda_d'):
		# 		lb = module.lambda_b.data
		# 		ld = module.lambda_d.data
		# 		print(f"\t{name:<60s}λ_b: mean={lb.mean():.6f} std={lb.std():.6f} max={lb.abs().max():.6f} λ_d: mean={ld.mean():.6f} std={ld.std():.6f} max={ld.abs().max():.6f}")

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[VeRA SCALING VECTORS — Post-Training Summary]")
	lambda_b_stats = []
	lambda_d_stats = []
	for name, module in model.named_modules():
		if hasattr(module, 'lambda_b') and hasattr(module, 'lambda_d'):
			lb = module.lambda_b.data.cpu()
			ld = module.lambda_d.data.cpu()
			lambda_b_stats.append(lb)
			lambda_d_stats.append(ld)
			print(
				f"  {name:<50s} "
				f"λ_b: mean={lb.mean():.6f} std={lb.std():.6f} max={lb.abs().max():.6f} | "
				f"λ_d: mean={ld.mean():.6f} std={ld.std():.6f}"
			)

	# Global summary
	glb_lb = torch.cat(lambda_b_stats)
	glb_ld = torch.cat(lambda_d_stats)

	lb_mean_abs = glb_lb.abs().mean().item()
	lb_std = glb_lb.std().item()
	ld_mean_delta = (glb_ld - 1.0).abs().mean().item()

	print(f"\n[GLOBAL SUMMARY — VeRA SCALING VECTORS]")
	print(f"  ▶ Gating Signal (λ_b): Mean |λ_b| = {lb_mean_abs} std={lb_std}")
	print(f"  ▶ Refine Signal (λ_d): Mean Δλ_d  = {ld_mean_delta} std={glb_ld.std()}")
	print(f"  ▶ λ_b — max={glb_lb.abs().max():.6f} nonzero={(glb_lb.abs() > 1e-6).sum().item()} / {glb_lb.numel()} ({100*(glb_lb.abs() > 1e-6).sum().item()/glb_lb.numel():.1f}%)")
	print(f"  ▶ λ_d — max={glb_ld.abs().max():.6f} delta_from_init={ld_mean_delta}")
	print(f"\n[DIAGNOSIS]")
	if lb_mean_abs < 1e-7:
		print("  ❌ CRITICAL: λ_b is effectively zero. The VeRA path is CLOSED. Check gradients/LR.")
	elif lb_mean_abs < 1e-4:
		print("  ⚠️ WARNING: λ_b is very weak. Model is barely using the adapted path.")
	else:
		print("  ✅ SUCCESS: λ_b is active. The model is successfully 'opening the gate' to VeRA.")
			
	if ld_mean_delta < 1e-7:
		print("  ⚠️ NOTE: λ_d hasn't moved from 1.0. Rank-space refinement is inactive.")
	elif ld_mean_delta > 0.5:
		print("  ⚠️ NOTE: λ_d has shifted significantly (>0.5). High adaptation in rank-space.")
	else:
		print("  ✅ SUCCESS: λ_d is refining the rank-space directions.")
	
	print(f"\n[{mode.upper()}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params={
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout,
		},
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def ia3_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
	temperature: float = 0.07,
	quantization_bits: int = 8,
	quantized: bool = False,
	pw_mode: str = "log",
	verbose: bool = True,
):
	"""
	(IA)³ fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label (IA)³ version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Args:
			model: CLIP model to fine-tune with (IA)³
			train_loader: Training torch.utils.data.DataLoader (must provide multi-label vectors)
			validation_loader: Validation torch.utils.data.DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate for (IA)³ parameters
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			volatility_threshold: Volatility threshold for early stopping
			slope_threshold: Slope threshold for early stopping
			pairwise_imp_threshold: Pairwise improvement threshold for early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			quantization_bits: Bits for quantization (if quantized=True)
			quantized: Whether to use quantized base weights
			verbose: Enable detailed logging
	"""
	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for (IA)³
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during (IA)³ fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for (IA)³ practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during (IA)³ fine-tuning "
			"which gives you more control over (IA)³-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
		
	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ Model      : {model_name} {model_arch}")
		print(f"   ├─ Dataset    : {dataset_name}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Device     : {type(device)} {device}")
		print(f"   ├─ Learning rate: {learning_rate}  Weight decay: {weight_decay}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss Weights: I2T={loss_weights['i2t']}, T2I={loss_weights['t2i']}")
		if quantized:
			print(f"   ├─ Using Quantization: {quantization_bits}-bit")

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	model = get_injected_peft_clip(
		clip_model=model,
		method=mode,
		rank=1,  # Not used for (IA)³, but required by function signature
		alpha=1.0,  # Not used for (IA)³, but required by function signature
		dropout=0.0,  # Not used for (IA)³, but required by function signature
		target_text_modules=[], # no (IA)³ in text encoder
		target_vision_modules=["in_proj", "c_proj"],  # Liu et al. (NeurIPS 2022): attention K, V (in_proj) + post-GELU FFN (c_proj)
		quantization_bits=quantization_bits,
		quantized=quantized,
		verbose=verbose,
	).to(device)
	get_parameters_info(model=model, mode=mode, verbose=verbose)
	get_parameters_info_orig(model=model, mode=mode)

	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode, # orig: "log",
		device=device,
		verbose=verbose,
	)
	pos_weight = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask = masks["head_mask"]
	rare_mask = masks["rare_mask"]
	N = masks["N"]
	train_freq = masks["train_freq"]

	# Criteria
	# I2T: pos_weight applies — rows are images, cols are classes
	criterion_i2t = torch.nn.BCEWithLogitsLoss(
		pos_weight=pos_weight,   # [num_classes], broadcasts over last dim correctly
		reduction='none',
	)
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}")
		print(f"   ├─ pos_weight: {type(pos_weight)} {pos_weight.shape} {pos_weight.dtype} {pos_weight.device} range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ samples: {N}")
		print(f"   ├─ Active classes (freq > 0): {active_mask.sum().item():,}")
		print(f"   ├─ active_mask: {type(active_mask)} {active_mask.shape} {active_mask.dtype} {active_mask.device} True count: {active_mask.sum().item():,}")
		print(f"   └─ train_freq: {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device} range: [{train_freq.min():.2f}, {train_freq.max():.2f}]")

	# T2I: no pos_weight — rows are classes, cols are batch images
	# The imbalance is already corrected via I2T; T2I provides directional symmetry
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none',)
	if verbose:
		print(f"\n[T2I] {criterion_t2i.__class__.__name__}")
		print(f"   └─ no pos_weight (imbalance already corrected by I2T)")
	
	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# Optimizer: multiplicative (IA)³ scaling vectors (initialized to 1.0) must not be decayed toward 0.0
	ia3_params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.AdamW(
		params=ia3_params,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=0.0,
	)
	
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in ia3_params):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: 0.0 (disabled for multiplicative (IA)³ scales; CLI weight_decay={weight_decay})")

	# Scheduler
	# approximate T_max: N epochs * minimum_epochs
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**15,      # 2048 (Conservative start)
		growth_factor=2.0,     # Smoother growth than default 2.0
		backoff_factor=0.5,    # Standard
		growth_interval=2000,  # Keep scale stable longer
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'N/A'}")
		print()

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	validation_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	learning_rates_history = list()
	weight_decays_history = list()
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images, _, label_vectors = batch_data

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
						
			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available(),
				dtype=amp_dtype,
			):
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				# no zero_grad needed here — already done above
				continue # skip this batch
			
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx + 1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f} T2I: {batch_loss_t2i:.6f})"
				)
		
		# Calculate average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")
		
		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			verbose=verbose,
		)
		
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score   = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {mode}-FT: Training - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Validation: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')
		
		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=mode,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}")
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = train_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Train Cache: {cache_stats}")

		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cache_stats = validation_loader.dataset.get_cache_stats()
		# 	if cache_stats is not None:
		# 		print(f"Validation Cache: {cache_stats}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")
	
	print(f"[{mode}] Total Training Time: {time.time() - train_start_time:.1f} sec")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		temperature=temperature,
		topk_values=topk_values,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)


	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{mode}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}"
	)
	
	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)

	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def clip_adapter_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	clip_adapter_method: str,  # "clip_adapter_v", "clip_adapter_t", "clip_adapter_vt"
	topk_values: List[int],
	bottleneck_dim: int = 256,
	activation: str = "relu",
	patience: int = 7,
	min_delta: float = 1e-4,
	cumulative_delta: float = 1e-3,
	minimum_epochs: int = 10,
	volatility_threshold: float = 0.02,
	slope_threshold: float = 1e-3,
	pairwise_imp_threshold: float = 0.01,
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,
	pw_mode: str = "log",
	verbose: bool = True,
):
	"""
	Fine-tunes a CLIP model using CLIP-Adapter for multi-label datasets.

	Three variants controlled by clip_adapter_method:
		- "clip_adapter_v"  : adapter on vision encoder only  → text encoder fully frozen
													→ all_class_embeds pre-encoded once, reused every epoch
		- "clip_adapter_t"  : adapter on text encoder only    → vision encoder fully frozen
													→ all_class_embeds re-encoded every epoch (adapter changes)
		- "clip_adapter_vt" : adapters on both encoders       → same as clip_adapter_t
													→ all_class_embeds re-encoded every epoch

	Loss:
		- criterion_i2t: BCEWithLogitsLoss with pos_weight  [B, C] → active_mask applied
		- criterion_t2i: BCEWithLogitsLoss plain             [C, B] → active_mask applied
	"""

	# Validate method
	valid_methods = ("clip_adapter_v", "clip_adapter_t", "clip_adapter_vt")
	assert clip_adapter_method in valid_methods, (
		f"clip_adapter_method must be one of {valid_methods}, got '{clip_adapter_method}'"
	)
	text_adapter_active = clip_adapter_method in ("clip_adapter_t", "clip_adapter_vt")
	visual_adapter_active = clip_adapter_method in ("clip_adapter_v", "clip_adapter_vt")
	device = torch.device(device)

	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	# Dropout check (warning only — CLIP-Adapter tolerates dropout) 
	non_zero_dropouts = [
		(name, module.p)
		for name, module in model.named_modules()
		if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if non_zero_dropouts and verbose:
		dropout_info = ", ".join([f"{n}: p={p}" for n, p in non_zero_dropouts])
		print(f"[WARNING] Non-zero dropout in base model: {dropout_info}")

	if verbose:
		print(f"{mode.upper()} [Multi-Label] variant: {clip_adapter_method} bottleneck={bottleneck_dim} act={activation}")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ {model_name} {model_arch}")
		print(f"   ├─ Batch size : {train_loader.batch_size}")
		print(f"   ├─ Temperature: {temperature}")
		print(f"   ├─ Loss weights: {loss_weights}")
		print(f"   ├─ Visual adapter active: {visual_adapter_active}")
		print(f"   ├─ Text adapter active  : {text_adapter_active}")

	amp_dtype = torch.float32
	if torch.cuda.is_available() and device.type == "cuda":
		gpu_name = torch.cuda.get_device_name(device)
		gpu_mem  = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# Early stopping
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	# Inject adapter (post-projection multimodal space, convex residual ratio=0.2 per Gao et al. IJCV 2023)
	model = get_adapter_peft_clip(
		clip_model=model,
		method=clip_adapter_method,
		bottleneck_dim=bottleneck_dim,
		activation=activation,
		verbose=verbose,
	).to(device)
	get_parameters_info(model=model, mode=clip_adapter_method, verbose=verbose)
	get_parameters_info_orig(model=model, mode=clip_adapter_method)

	if verbose:
		trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
		frozen = [(n, p.numel()) for n, p in model.named_parameters() if not p.requires_grad]
		print(f"Trainable tensors: {len(trainable)} | Frozen tensors: {len(frozen)}")
		for name, numel in trainable:
			print(f"  {name}: {numel:,}")
		print(f"[TOTAL] trainable: {sum(n for _,n in trainable):,} frozen: {sum(n for _,n in frozen):,}")

	# Loss masks (respect passed pw_mode)
	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N           = masks["N"]
	train_freq  = masks["train_freq"]

	# Criteria
	criterion_i2t = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none')
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__}  pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"   ├─ N={N} active={active_mask.sum().item():,}")
		print(f"[T2I] {criterion_t2i.__class__.__name__} no pos_weight")

	# Optimizer — exclude 0D/1D parameters (biases, LayerNorm, scalar ratios) from weight decay
	decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
	no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
	adapter_params = decay_params + no_decay_params
	if not adapter_params:
		raise ValueError("No trainable parameters found. Check get_adapter_peft_clip.")

	param_groups = []
	if decay_params:
		param_groups.append({"params": decay_params, "weight_decay": weight_decay})
	if no_decay_params:
		param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

	optimizer = torch.optim.AdamW(
		params=param_groups,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__}")
		print(f"  ├─ Params: {sum(p.numel() for p in adapter_params):,} (decay: {sum(p.numel() for p in decay_params):,}, no_decay: {sum(p.numel() for p in no_decay_params):,})")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: {weight_decay}")

	# Scheduler
	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_loader)
	ANNEALING_RATIO = 1e-2 # 1% of initial LR
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	# GradScaler is ONLY needed for FP16 (never BF16 or FP32)
	use_grad_scaler = torch.cuda.is_available() and device.type == "cuda" and (amp_dtype == torch.float16)
	scaler = torch.amp.GradScaler(
		device=device.type,
		enabled=use_grad_scaler,
		init_scale=2**15,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	if verbose:
		print(f"\n{scaler.__class__.__name__} (enabled: {scaler.is_enabled()}) for AMP training")
		scaler_state = scaler.state_dict()
		print(f"  ├─ init_scale: {scaler_state.get('scale', 'N/A')}")
		print(f"  ├─ growth_factor: {scaler_state.get('growth_factor', 'N/A')}")
		print(f"  ├─ backoff_factor: {scaler_state.get('backoff_factor', 'N/A')}")
		print(f"  ├─ growth_interval: {scaler_state.get('growth_interval', 'N/A')}")
		print(f"  └─ dtype: {amp_dtype if torch.cuda.is_available() else 'torch.float32'}")
		print()

	# Checkpoint path
	mdl_fpth = os.path.join(
		results_dir,
		f"{clip_adapter_method}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"cbd_{bottleneck_dim}_act_{activation}_temp_{temperature}_"
		f"mep_{minimum_epochs}_pat_{patience}_mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	# Pre-encode frozen class text features ONCE:
	# - For clip_adapter_v: L2-normalized frozen text embeddings reused every batch.
	# - For clip_adapter_t / clip_adapter_vt: unnormalized post-projection base text features [C, D]
	#   computed once from frozen CLIP text encoder; model.text_adapter(base_class_feats) is applied
	#   per batch with gradients (avoids re-running the 12-layer text transformer over C classes).
	if not text_adapter_active:
		all_class_embeds = get_text_embeddings(
			model=model,
			loader=validation_loader,
			dtype=amp_dtype,
			verbose=verbose,
		)
		base_class_feats = None
	else:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except AttributeError:
			class_names = validation_loader.dataset.unique_labels
		model.eval()
		text_batch_size = validation_loader.batch_size
		base_chunks = []
		with torch.no_grad(), torch.amp.autocast(
			device_type=device.type,
			enabled=torch.cuda.is_available() and device.type == "cuda",
			dtype=amp_dtype,
		):
			for i in range(0, len(class_names), text_batch_size):
				batch_names = class_names[i : i + text_batch_size]
				batch_tokens = clip.tokenize(batch_names).to(device)
				batch_base = model.encode_base_text(batch_tokens).float()
				base_chunks.append(batch_base.cpu())
				del batch_tokens, batch_base
		base_class_feats = torch.cat(base_chunks, dim=0).to(device).detach()
		all_class_embeds = None
		if verbose:
			print(f"\n>> Text adapter active — cached frozen base text features: {tuple(base_class_feats.shape)} ({base_class_feats.dtype})")

	def _get_adapted_class_embeds_no_grad() -> torch.Tensor:
		model.text_adapter.eval()
		with torch.no_grad():
			adapted = model.text_adapter(base_class_feats).float()
			return torch.nn.functional.normalize(adapted, dim=-1)

	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history  = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()

		# Keep frozen CLIP backbone in eval mode (protects BatchNorm/Dropout); only adapters in train mode
		model.eval()
		if visual_adapter_active:
			model.visual.adapter.train()
		if text_adapter_active:
			model.text_adapter.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")

		epoch_loss_total = 0.0
		epoch_loss_i2t   = 0.0
		epoch_loss_t2i   = 0.0
		num_batches      = 0

		for bidx, (images, _, label_vectors) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)

			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()

			with torch.amp.autocast(
				device_type=device.type, 
				enabled=torch.cuda.is_available() and device.type == "cuda",
				dtype=amp_dtype,
			):
				if text_adapter_active:
					# Pass frozen base text features through trainable text_adapter WITH gradients
					batch_class_embeds = model.text_adapter(base_class_feats)
					batch_class_embeds = torch.nn.functional.normalize(batch_class_embeds.float(), dim=-1)
				else:
					batch_class_embeds = all_class_embeds  # pre-encoded, detached, FP32

				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=batch_class_embeds,
					label_vectors=label_vectors,
					criterion_i2t=criterion_i2t,
					criterion_t2i=criterion_t2i,
					active_mask=active_mask,
					temperature=temperature,
					loss_weights=loss_weights,
					verbose=verbose,
				)

			# Check for NaN loss
			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} "
					f"loss_i2t: {loss_i2t} "
					f"loss_t2i: {loss_t2i}"
				)
				continue # skip this batch

			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()

			epoch_loss_total += total_loss.item()
			epoch_loss_i2t   += loss_i2t.item()
			epoch_loss_t2i   += loss_t2i.item()
			num_batches      += 1

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\tBatch[{bidx+1:6d}/{len(train_loader)}] "
					f"[LOSS] Total: {total_loss.item():.6f} "
					f"(I2T: {loss_i2t.item():.6f} "
					f"T2I: {loss_t2i.item():.6f})"
				)

		avg_total = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t   = epoch_loss_i2t   / num_batches if num_batches > 0 else 0.0
		avg_t2i   = epoch_loss_t2i   / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total)
		training_losses_breakdown["total"].append(avg_total)
		training_losses_breakdown["i2t"].append(avg_i2t)
		training_losses_breakdown["t2i"].append(avg_t2i)
		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		# For validation loss: use freshly adapted text embeddings when text adapter is active
		val_class_embeds = _get_adapted_class_embeds_no_grad() if text_adapter_active else all_class_embeds

		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion_i2t=criterion_i2t,
			criterion_t2i=criterion_t2i,
			active_mask=active_mask,
			device=device,
			all_class_embeds=val_class_embeds,
			temperature=temperature,
			verbose=verbose,
		)
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=clip_adapter_method,
			cache_dir=results_dir,
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			class_embeds_override=val_class_embeds if text_adapter_active else None,
			verbose=verbose,
		)

		full_metrics = validation_results["full_metrics"]
		cos_sim = full_metrics.get("cosine_similarity")
		align_score = full_metrics.get("alignment_score")
		
		retrieval = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"],
		}

		full_val_loss_acc_metrics_all_epochs.append(full_metrics)
		img2txt_metrics_all_epochs.append(retrieval["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {clip_adapter_method}-FT: Train: {avg_total:.6f} (I2T: {avg_i2t:.6f}, T2I: {avg_t2i:.6f})  Val: {current_val_loss:.6f}\n'
			f'   ├─ LR: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ [I2T] Accuracy: {full_metrics.get("img2txt_topk_acc")}\n'
			f'   ├─ [T2I] Accuracy: {full_metrics.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')


		print(
			f"   ├─ [I2T] mAP={retrieval['img2txt'].get('mAP',{})}  R={retrieval['img2txt'].get('Recall',{})}\n"
			f"   ├─ [T2I] mAP={retrieval['txt2img'].get('mAP',{})}  R={retrieval['txt2img'].get('Recall',{})}"
		)
		if full_metrics.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_metrics["hamming_loss"]:.4f}')
			print(f'   ├─ PartialAcc: {full_metrics["partial_acc"]:.4f}')
			print(f'   └─ F1: {full_metrics["f1_score"]:.4f}')

		# ── Training health check ────────────────────────────────────────────
		# Run after epoch 1 and at mid-warmup — all signals now available
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=clip_adapter_method,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break

		# # Cache stats
		# if hasattr(train_loader.dataset, 'get_cache_stats'):
		# 	cs = train_loader.dataset.get_cache_stats()
		# 	if cs: print(f"Train cache: {cs}")
		
		# if hasattr(validation_loader.dataset, 'get_cache_stats'):
		# 	cs = validation_loader.dataset.get_cache_stats()
		# 	if cs: print(f"Val cache: {cs}")

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time()-train_and_val_st_time:.1f}s")

	print(f"[{clip_adapter_method}] Total Training Elapsed Time: {time.time()-train_start_time:.1f} sec")

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=clip_adapter_method,
		device=device,
		cache_dir=results_dir,
		lora_params=None,
		topk_values=topk_values,
		temperature=temperature,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print_final_evaluation_summary(
		mode=mode,
		method=clip_adapter_method,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{clip_adapter_method}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_"
		f"cbd_{bottleneck_dim}_act_{activation}_temp_{temperature}"
	)

	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(viz_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(viz_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	viz.plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"],
	)
	
	viz.plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)
	
	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)
	
	viz.plot_hyperparameter_evolution(
		eta_min=eta_min,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		fname=plot_paths["hp_evol"],
	)

	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)

	return best_model_eval_results

def tip_adapter_finetune_multi_label(
	model: torch.nn.Module,
	train_loader: torch.utils.data.DataLoader,
	validation_loader: torch.utils.data.DataLoader,
	num_epochs: int,
	print_every: int,
	learning_rate: float,
	weight_decay: float,
	device: str,
	results_dir: str,
	tier_spec_path: str,
	tip_adapter_method: str, # "tip_adapter" or "tip_adapter_f"
	patience: int,
	min_delta: float,
	cumulative_delta: float,
	minimum_epochs: int,
	volatility_threshold: float,
	slope_threshold: float,
	pairwise_imp_threshold: float,
	topk_values: List[int],
	initial_beta: float = 5.5,
	initial_alpha: float = 1.0,
	support_shots: int = 16,  # Target support samples per class (deduplicated across multi-label images)
	temperature: float = 0.07,
	loss_weights: Dict[str, float] = None,
	pw_mode: str = "log",
	verbose: bool = True,
):
	"""
	Adapts a frozen CLIP model using Tip-Adapter (training-free) or Tip-Adapter-F (fine-tuned cache keys)
	for multi-label classification & retrieval, faithful to Zhang et al. (ECCV 2022, Eq. 4-6):

		clip_logits  = (f_norm @ W_c.T) / temperature
		cache_logits = exp(-beta * (1 - f_norm @ K_norm.T)) @ L_val
		tip_logits   = clip_logits + alpha * cache_logits

	Multi-label adaptations:
		- Entire CLIP visual and text encoders remain completely frozen.
		- Support set images are deduplicated (each multi-label image enters the cache at most once,
		  selected greedily so every active class receives up to `support_shots` coverage).
		- Cache values L_val ∈ R^{M × C} are column-normalized by per-class support counts so head,
		  body, and rare classes share identical [0, 1] cache affinity scale while inactive classes
		  remain 0 and fall back cleanly to zero-shot CLIP.
		- Tip-Adapter: zero gradient steps; tunes (alpha, beta) via validation grid search.
		- Tip-Adapter-F: unfreezes and fine-tunes ONLY the cache keys K ∈ R^{M × D} via AdamW +
		  cosine schedule on bidirectional multi-label BCE loss, then refines (alpha, beta) via
		  validation grid search (accepting only if val loss improves).
	"""
	valid_methods = ("tip_adapter", "tip_adapter_f")
	assert tip_adapter_method in valid_methods, (
		f"tip_adapter_method must be one of {valid_methods}, got '{tip_adapter_method}'"
	)
	device = torch.device(device)

	window_size = minimum_epochs + 1
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [
		(name, module.p)
		for name, module in model.named_modules()
		if isinstance(module, torch.nn.Dropout) and module.p > 0
	]
	if verbose and non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		print(f"[Tip-Adapter Multi-Label] WARNING: Non-zero dropout detected in base model: {dropout_info}.")

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
	)

	# Dataset and class metadata (consistent order with compute_loss_masks / get_text_embeddings / evals)
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	try:
		class_names = validation_loader.dataset.dataset.classes
	except AttributeError:
		try:
			class_names = validation_loader.dataset.unique_labels
		except AttributeError:
			class_names = train_loader.dataset.unique_labels

	num_classes = len(class_names)

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	if verbose:
		print(f"\n{mode.upper()} [Multi-Label]")
		print(f"   ├─ {dataset_name}")
		print(f"   ├─ train: {len(train_loader.dataset):,} samples | val: {len(validation_loader.dataset):,} samples")
		print(f"   ├─ {model_name} {model_arch} | {tip_adapter_method}")
		print(f"   ├─ Beta[init]: {initial_beta} | Alpha[init]: {initial_alpha} | Support Shots: {support_shots}")
		print(f"   ├─ Batch size: {train_loader.batch_size} | Temperature: {temperature} | Loss Weights: {loss_weights}")
		print(f"   ├─ Device    : {type(device)} {device}")

	amp_dtype = torch.float32
	if torch.cuda.is_available() and device.type == "cuda":
		gpu_name = torch.cuda.get_device_name(device)
		gpu_total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		cuda_capability = torch.cuda.get_device_capability()
		if cuda_capability[0] >= 8 and torch.cuda.is_bf16_supported():
			amp_dtype = torch.bfloat16
		else:
			amp_dtype = torch.float16
		if verbose:
			print(f"   └─ {gpu_name} | {gpu_total_mem:.2f}GB VRAM | cuda capability: {cuda_capability}")

	# 1. Compute loss masks first (so we know active classes & pos_weight)
	masks = compute_loss_masks(
		train_loader=train_loader,
		validation_loader=validation_loader,
		pw_mode=pw_mode,
		device=device,
		verbose=verbose,
	)
	pos_weight  = masks["pos_weight"]
	active_mask = masks["active_mask"]
	head_mask   = masks["head_mask"]
	rare_mask   = masks["rare_mask"]
	N           = masks["N"]
	train_freq  = masks["train_freq"]

	criterion_i2t = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
	criterion_t2i = torch.nn.BCEWithLogitsLoss(reduction='none')
	if verbose:
		print(f"\n[I2T] {criterion_i2t.__class__.__name__} pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}] | active={active_mask.sum().item():,}")
		print(f"[T2I] {criterion_t2i.__class__.__name__} (no pos_weight)")

	# 2. Pre-extract frozen CLIP visual features ONCE for train & val (entire visual encoder is frozen!)
	if verbose:
		print("-" * 100)
		print("[FROZEN FEATURE EXTRACTION] Extracting L2-normalized CLIP visual features once for train & val...")

	model.eval()
	def _extract_frozen_features(loader, desc: str) -> Tuple[torch.Tensor, torch.Tensor]:
		feats_chunks, lbls_chunks = [], []
		with torch.no_grad(), torch.amp.autocast(
			device_type=device.type,
			enabled=torch.cuda.is_available() and device.type == "cuda",
			dtype=amp_dtype,
		):
			for images, _, label_vectors in tqdm(loader, desc=desc, disable=not verbose):
				images = images.to(device, non_blocking=True)
				emb = model.encode_image(images).float()
				emb = torch.nn.functional.normalize(emb, dim=-1)
				feats_chunks.append(emb.cpu())
				lbls_chunks.append(label_vectors.cpu())
		return torch.cat(feats_chunks, dim=0), torch.cat(lbls_chunks, dim=0)

	train_feats, train_lbls = _extract_frozen_features(train_loader, "Tip-Adapter train features")
	val_feats, val_lbls = _extract_frozen_features(validation_loader, "Tip-Adapter val features")

	# 3. Build DEDUPLICATED multi-label support cache (rare-first greedy selection)
	# Sort training samples by the minimum training frequency of their active labels (rarest first)
	# so rare classes always secure their shots cleanly, and each unique image enters the cache AT MOST ONCE.
	n_train_samples = train_feats.shape[0]
	train_freq_cpu = train_freq.cpu()
	sample_min_freq = torch.full((n_train_samples,), float('inf'), dtype=torch.float32)
	for idx in range(n_train_samples):
		active_c = torch.where(train_lbls[idx] > 0)[0]
		if active_c.numel() > 0:
			sample_min_freq[idx] = train_freq_cpu[active_c].min()

	# Deterministic order: rarest-label samples first
	ordered_indices = torch.argsort(sample_min_freq, stable=True).tolist()
	class_shot_counts = torch.zeros(num_classes, dtype=torch.long)
	selected_indices = []
	n_active_total = int((train_freq_cpu > 0).sum().item())

	for idx in ordered_indices:
		active_c = torch.where(train_lbls[idx] > 0)[0]
		if active_c.numel() == 0:
			continue
		if (class_shot_counts[active_c] < support_shots).any():
			selected_indices.append(idx)
			class_shot_counts[active_c] += 1
			if int((class_shot_counts >= support_shots).sum().item()) >= n_active_total:
				break

	if selected_indices:
		sel_idx_tensor = torch.tensor(selected_indices, dtype=torch.long)
		support_features = train_feats[sel_idx_tensor].clone().to(device)
		support_label_vectors = train_lbls[sel_idx_tensor].clone().to(device)
	else:
		feat_dim = train_feats.shape[1]
		support_features = torch.empty(0, feat_dim, device=device, dtype=torch.float32)
		support_label_vectors = torch.empty(0, num_classes, device=device, dtype=torch.float32)

	covered_classes = int((class_shot_counts > 0).sum().item())
	if verbose:
		print(f"[DEDUPLICATED SUPPORT CACHE]")
		print(f"   ├─ Unique cached images : {support_features.shape[0]:,} / {n_train_samples:,} train samples")
		print(f"   ├─ Covered classes      : {covered_classes:,} / {n_active_total:,} active ({num_classes:,} total)")
		print(f"   ├─ Support features     : {tuple(support_features.shape)} ({support_features.dtype})")
		print(f"   └─ Support labels       : {tuple(support_label_vectors.shape)} ({support_label_vectors.dtype})")
		print("-" * 100)

	# 4. Pre-encode frozen class text embeddings [C, D] in FP32
	all_class_embeds = get_text_embeddings(
		model=model,
		loader=validation_loader,
		dtype=amp_dtype,
		verbose=verbose,
	)

	# 5. Attach faithful logit-space Tip-Adapter / Tip-Adapter-F head to frozen CLIP
	model = get_adapter_peft_clip(
		clip_model=model,
		method=tip_adapter_method,
		initial_beta=initial_beta,
		initial_alpha=initial_alpha,
		verbose=verbose,
	).to(device)

	adapter_module = model.tip_adapter
	adapter_module.set_cache(
		support_features=support_features,
		support_labels=support_label_vectors,
		text_features=all_class_embeds,
		normalize_by_class_count=True,
	)

	# Calibrate global scalar logit_bias to multi-label prior log-odds so non-shift-invariant
	# BCEWithLogitsLoss starts around negative prior logits (preserving exact I2T & T2I rankings)
	n_calib = min(512, train_feats.shape[0])
	if n_calib > 0:
		adapter_module.calibrate_logit_bias(
			sample_features=train_feats[:n_calib],
			active_mask=active_mask,
			train_freq=train_freq,
			num_samples=N,
			temperature=temperature,
		)

	if verbose:
		cache_stats = adapter_module.get_memory_footprint()
		print(f"\n[CACHE] {tip_adapter_method}:")
		print(f"  ├─ Unique cache entries : {cache_stats.get('cache_size', 0):,}")
		print(f"  ├─ Trainable key params : {cache_stats.get('trainable_params', 0):,}")
		print(f"  ├─ Cache memory         : {cache_stats.get('cache_memory_mb', 0):.4f} MB")
		print(f"  └─ Total head memory    : {cache_stats.get('total_memory_mb', 0):.4f} MB")

	get_parameters_info(model=model, mode=tip_adapter_method, verbose=verbose)
	get_parameters_info_orig(model=model, mode=tip_adapter_method)

	# Feature dataloaders for fast FP32 training & validation
	train_feature_loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(train_feats, train_lbls),
		batch_size=train_loader.batch_size,
		shuffle=True,
		num_workers=0,
	)
	val_feature_loader = torch.utils.data.DataLoader(
		torch.utils.data.TensorDataset(val_feats, val_lbls),
		batch_size=validation_loader.batch_size,
		shuffle=False,
		num_workers=0,
	)

	@torch.no_grad()
	def _compute_tip_val_loss(alpha_override: Optional[float] = None, beta_override: Optional[float] = None) -> float:
		adapter_module.eval()
		max_batches = max(50, len(val_feature_loader) // 10)
		total_loss = 0.0
		total_samples = 0
		for bidx, (v_feats, v_lbls) in enumerate(val_feature_loader):
			if bidx >= max_batches:
				break
			bs = v_feats.size(0)
			if bs == 0:
				continue
			v_feats = v_feats.to(device, non_blocking=True).float()
			v_lbls = v_lbls.to(device, non_blocking=True).float()
			tip_logits = adapter_module(
				v_feats,
				temperature=temperature,
				alpha=alpha_override,
				beta=beta_override,
			)
			batch_loss, _, _ = compute_multilabel_logit_loss(
				logits=tip_logits,
				label_vectors=v_lbls,
				criterion_i2t=criterion_i2t,
				criterion_t2i=criterion_t2i,
				active_mask=active_mask,
				loss_weights=loss_weights,
			)
			total_loss += batch_loss.item() * bs
			total_samples += bs
		return total_loss / total_samples if total_samples > 0 else 0.0

	@torch.no_grad()
	def _compute_tip_val_map_at_k(
		alpha_override: Optional[float] = None,
		beta_override: Optional[float] = None,
		k: int = 10,
	) -> float:
		"""Fast vectorized validation I2T mAP@k on active classes for ranking-metric HP search."""
		adapter_module.eval()
		max_batches = max(50, len(val_feature_loader) // 10)
		ap_sum = 0.0
		valid_count = 0
		ranks = torch.arange(1, k + 1, device=device, dtype=torch.float32).unsqueeze(0)
		for bidx, (v_feats, v_lbls) in enumerate(val_feature_loader):
			if bidx >= max_batches:
				break
			if v_feats.size(0) == 0:
				continue
			vf = v_feats.to(device, non_blocking=True).float()
			vl = v_lbls.to(device, non_blocking=True).float()
			scores = adapter_module(vf, temperature=temperature, alpha=alpha_override, beta=beta_override)
			eff_k = min(k, scores.shape[1])
			topk_idx = scores.topk(eff_k, dim=1).indices
			hits = vl.gather(1, topk_idx)
			cum_hits = hits.cumsum(dim=1)
			prec_at_r = cum_hits / ranks[:, :eff_k]
			denom = vl.sum(dim=1).clamp(max=eff_k)
			valid = denom > 0
			if valid.any():
				ap = (prec_at_r * hits).sum(dim=1)[valid] / denom[valid]
				ap_sum += ap.sum().item()
				valid_count += int(valid.sum().item())
		return ap_sum / valid_count if valid_count > 0 else 0.0

	@torch.no_grad()
	def _grid_search_alpha_beta(k: int = 10) -> Tuple[float, float, float]:
		"""
		Validation grid search over (alpha, beta) maximizing validation I2T mAP@k
		(faithful to Zhang et al., ECCV 2022 Sec. 4.1 search_hp, which optimizes validation
		accuracy/ranking rather than raw loss so alpha is never shrunk just to shift logit scale).
		"""
		adapter_module.eval()
		max_batches = max(50, len(val_feature_loader) // 10)
		alpha_grid = sorted({float(initial_alpha), 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0})
		beta_grid = sorted({float(initial_beta), 1.0, 2.0, 3.5, 5.5, 8.0, 12.0, 20.0})

		# Gather validation slice batches + zero-shot clip_logits ONCE
		val_batches = []
		for bidx, (v_feats, v_lbls) in enumerate(val_feature_loader):
			if bidx >= max_batches:
				break
			if v_feats.size(0) == 0:
				continue
			vf_gpu = v_feats.to(device, non_blocking=True).float()
			vl_gpu = v_lbls.to(device, non_blocking=True).float()
			c_logits = adapter_module.compute_clip_logits(vf_gpu, temperature=temperature)
			val_batches.append((vf_gpu, vl_gpu, c_logits))

		ranks = torch.arange(1, k + 1, device=device, dtype=torch.float32).unsqueeze(0)
		best_score = -1.0
		best_alpha = float(adapter_module.alpha.item())
		best_beta = float(adapter_module.beta.item())

		for b_val in beta_grid:
			# Compute temperature-scaled cache_logits once per beta across all validation batches
			cache_logits_list = [
				adapter_module.compute_cache_logits(vf_gpu, beta=b_val, temperature=temperature)
				for vf_gpu, _, _ in val_batches
			]
			for a_val in alpha_grid:
				ap_sum = 0.0
				valid_n = 0
				for (_, vl_gpu, c_logits), cache_l in zip(val_batches, cache_logits_list):
					t_logits = c_logits + a_val * cache_l
					eff_k = min(k, t_logits.shape[1])
					topk_idx = t_logits.topk(eff_k, dim=1).indices
					hits = vl_gpu.gather(1, topk_idx)
					cum_hits = hits.cumsum(dim=1)
					prec_at_r = cum_hits / ranks[:, :eff_k]
					denom = vl_gpu.sum(dim=1).clamp(max=eff_k)
					valid = denom > 0
					if valid.any():
						ap = (prec_at_r * hits).sum(dim=1)[valid] / denom[valid]
						ap_sum += ap.sum().item()
						valid_n += int(valid.sum().item())
				avg_map = ap_sum / valid_n if valid_n > 0 else 0.0
				if avg_map > best_score:
					best_score = avg_map
					best_alpha = a_val
					best_beta = b_val

		if verbose:
			print(
				f"[GRID SEARCH] Searched {len(beta_grid)} β × {len(alpha_grid)} α = {len(beta_grid)*len(alpha_grid)} configs "
				f"→ best α={best_alpha}, β={best_beta} (val I2T mAP@{k}={best_score:.4f})"
			)
		return best_alpha, best_beta, best_score

	# Trainable parameters: empty for training-free tip_adapter; [cache_keys] for tip_adapter_f
	trainable_parameters = [p for p in model.parameters() if p.requires_grad]

	if tip_adapter_method == "tip_adapter":
		num_epochs = 0
	elif not trainable_parameters:
		raise ValueError("Tip-Adapter-F has no trainable cache_keys. Check support set construction and set_cache().")

	mdl_fpth = os.path.join(
		results_dir,
		f"{tip_adapter_method}_"
		f"{model_arch}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"init_alpha_{initial_alpha}_"
		f"init_beta_{initial_beta}_"
		f"shots_{support_shots}_"
		f"temp_{temperature}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pw_{pw_mode}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	viz_dir = os.path.join(results_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)

	# ─────────────────────────────────────────────────────────────────────────
	# BRANCH A: Training-Free Tip-Adapter (Zhang et al., ECCV 2022, Sec. 3.1)
	# ─────────────────────────────────────────────────────────────────────────
	if num_epochs == 0 or not trainable_parameters:
		if verbose:
			print(f"\n[{tip_adapter_method}] Training-free mode — running validation grid search over (α, β)...")

		best_alpha, best_beta, best_val_map = _grid_search_alpha_beta()
		adapter_module.set_alpha_beta(best_alpha, best_beta)
		if n_calib > 0:
			adapter_module.calibrate_logit_bias(
				sample_features=train_feats[:n_calib],
				active_mask=active_mask,
				train_freq=train_freq,
				num_samples=N,
				temperature=temperature,
			)
		best_val_loss = _compute_tip_val_loss()
		actual_trained_epochs = 0

		# Save checkpoint so downstream inference can load the tuned Tip-Adapter cache
		torch.save(
			{
				"epoch": 0,
				"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
				"best_val_loss": best_val_loss,
				"best_val_map10": best_val_map,
				"alpha": best_alpha,
				"beta": best_beta,
			},
			mdl_fpth,
		)
		if verbose:
			print(f"[{tip_adapter_method}] Saved training-free cache checkpoint → {mdl_fpth}")

		best_model_eval_results = evaluate_best_model(
			model=model,
			validation_loader=validation_loader,
			active_mask=active_mask,
			head_mask=head_mask,
			rare_mask=rare_mask,
			early_stopping=None,
			checkpoint_path=mdl_fpth,
			finetune_strategy=tip_adapter_method,
			device=device,
			cache_dir=results_dir,
			embeddings_cache=(val_feats, None),
			topk_values=topk_values,
			temperature=temperature,
			use_tip_logits=True,
			use_fixed_masks=True,
			tier_spec_path=tier_spec_path,
			verbose=verbose,
		)

		print_final_evaluation_summary(
			mode=mode,
			method=tip_adapter_method,
			model_arch=model_arch,
			model=model,
			actual_trained_epochs=actual_trained_epochs,
			early_stopping=None,
			mdl_fpth=mdl_fpth,
			pw_mode=pw_mode,
			eval_results=best_model_eval_results,
			extra_lines=[
				f"Tip-Adapter (Training-Free): M={adapter_module.cache_keys.shape[0]:,} unique cache keys | "
				f"grid-searched α={best_alpha}, β={best_beta}, val_mAP@10={best_val_map:.4f}, val_loss={best_val_loss:.6f}",
			],
			verbose=verbose,
		)

		file_base_name = (
			f"{tip_adapter_method}_"
			f"{model_arch}_"
			f"training_free_"
			f"bs_{train_loader.batch_size}_"
			f"beta_{best_beta}_"
			f"alpha_{best_alpha}_"
			f"shots_{support_shots}_"
			f"temp_{temperature}"
		)
		viz.plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
			text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
			fname=os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		)
		return best_model_eval_results

	# ─────────────────────────────────────────────────────────────────────────
	# BRANCH B: Tip-Adapter-F (Fine-Tuned Cache Keys — Zhang et al., ECCV 2022, Sec. 3.2)
	# ─────────────────────────────────────────────────────────────────────────
	# Use weight_decay=0.0 on unit-sphere-normalized cache_keys and scalar logit_bias
	# (radial weight decay on L2-normalized keys only causes artificial norm shrinkage).
	optimizer = torch.optim.AdamW(
		params=trainable_parameters,
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=0.0,
	)
	if verbose:
		print(f"\n{optimizer.__class__.__name__} optimizer (Tip-Adapter-F cache keys + scalar logit_bias)")
		print(f"  ├─ Params: {sum(p.numel() for p in trainable_parameters):,}")
		print(f"  ├─ LR: {learning_rate}")
		print(f"  ├─ Betas: {optimizer.defaults['betas']}")
		print(f"  ├─ Eps: {optimizer.defaults['eps']}")
		print(f"  └─ Weight Decay: 0.0 (disabled for unit-normalized keys & scalar bias)")

	estimated_epochs = 2 * minimum_epochs
	T_max = estimated_epochs * len(train_feature_loader)
	ANNEALING_RATIO = 1e-2
	eta_min = learning_rate * ANNEALING_RATIO
	scheduler = build_cosine_scheduler(
		optimizer=optimizer,
		T_max=T_max,
		floor=ANNEALING_RATIO,
	)
	if verbose:
		print(f"\n{scheduler.__class__.__name__}")
		print(f"  ├─ minimum_epochs = {minimum_epochs}")
		print(f"  ├─ estimated_epochs = {estimated_epochs} ({estimated_epochs/minimum_epochs:.1f}x minimum_epochs)")
		print(f"  ├─ T_max = {T_max} steps [({estimated_epochs} estimated epochs x {len(train_feature_loader)} batches/epoch)]")
		print(f"  └─ eta_min = {eta_min} ({ANNEALING_RATIO*100}% of initial LR={learning_rate})")

	# Tip-Adapter-F head trains cleanly in pure FP32 (no GradScaler needed)
	training_losses = []
	validation_losses = []
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	learning_rates_history = []
	weight_decays_history = []
	alphas = []
	betas = []
	train_start_time = time.time()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.eval()          # Keep frozen CLIP backbone in eval mode
		adapter_module.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")

		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0

		for bidx, (batch_feats, label_vectors) in enumerate(train_feature_loader):
			optimizer.zero_grad(set_to_none=True)
			batch_feats = batch_feats.to(device, non_blocking=True).float()
			label_vectors = label_vectors.to(device, non_blocking=True).float()

			tip_logits = adapter_module(batch_feats, temperature=temperature)
			total_loss, loss_i2t, loss_t2i = compute_multilabel_logit_loss(
				logits=tip_logits,
				label_vectors=label_vectors,
				criterion_i2t=criterion_i2t,
				criterion_t2i=criterion_t2i,
				active_mask=active_mask,
				loss_weights=loss_weights,
			)

			if torch.isnan(total_loss) or torch.isinf(total_loss):
				print(
					f"[WARNING] e{epoch+1} b{bidx+1}: "
					f"total_loss: {total_loss} loss_i2t: {loss_i2t} loss_t2i: {loss_t2i}"
				)
				continue

			total_loss.backward()
			torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
			optimizer.step()
			scheduler.step()

			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()

			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1

			if bidx % print_every == 0 or bidx + 1 == len(train_feature_loader):
				print(
					f"\tBatch[{bidx + 1:6d}/{len(train_feature_loader)}] "
					f"[LOSS] Total: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f} T2I: {batch_loss_t2i:.6f})"
				)

		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0

		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		learning_rates_history.append([optimizer.param_groups[0]['lr']])
		weight_decays_history.append([optimizer.param_groups[0]['weight_decay']])

		print(f"[Epoch {epoch+1}] Training elapsed time: {time.time()-train_and_val_st_time:.1f}s\nValidating...")

		current_val_loss = _compute_tip_val_loss()
		validation_losses.append(current_val_loss)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
			finetune_strategy=tip_adapter_method,
			cache_dir=results_dir,
			embeddings_cache=(val_feats, None),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
			use_tip_logits=True,
			verbose=verbose,
		)

		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		cos_sim = full_val_loss_acc_metrics_per_epoch.get("cosine_similarity")
		align_score = full_val_loss_acc_metrics_per_epoch.get("alignment_score")

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"],
		}

		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])

		print(
			f'\nEpoch {epoch+1}:\n'
			f'   ├─ [LOSS] {tip_adapter_method.upper()}: Train - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f}) Val: {current_val_loss:.6f}\n'
			f'   ├─ Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n'
			f'   ├─ Multi-label Validation Accuracy Metrics:\n'
			f'      ├─ [I2T] {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'      └─ [T2I] {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		if align_score is not None:
			print(f'   ├─ Embed — AlignScore@5: {align_score:.4f}')
		elif cos_sim is not None:
			print(f'   ├─ Embed — CosSim: {cos_sim:.4f}')
		else:
			print(f'   ├─ Embed — AlignScore: N/A')

		print(f"   ├─ Retrieval Metrics:")
		print(
			f"      ├─ [I2T] mAP {retrieval_metrics_per_epoch['img2txt'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['img2txt'].get('Recall', {})}"
		)
		print(
			f"      └─ [T2I] mAP: {retrieval_metrics_per_epoch['txt2img'].get('mAP', {})}, "
			f"Recall: {retrieval_metrics_per_epoch['txt2img'].get('Recall', {})}"
		)

		print(
			f"   ├─ α: {adapter_module.alpha.item():.5f}, β: {adapter_module.beta.item():.5f}, "
			f"logit_bias: {adapter_module.logit_bias.item():.4f}"
		)
		alphas.append(adapter_module.alpha.item())
		betas.append(adapter_module.beta.item())

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'   ├─ Hamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'   ├─ Partial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'   └─ F1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')

		# Training health check
		if epoch in {0, minimum_epochs // 2}:
			should_abort = check_training_health(
				model=model,
				epoch=epoch,
				mode=tip_adapter_method,
				training_losses=training_losses,
				validation_losses=validation_losses,
				align_score=align_score,
				temperature=temperature,
				learning_rate=learning_rate,
				verbose=verbose,
			)
			if should_abort:
				print(f"[{mode.upper()}] Aborting at epoch {epoch+1} due to broken gradient signal.")
				break

		# Early stopping check
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\n[EARLY STOPPING] best loss: {early_stopping.get_best_score():.6f} "
				f"@ epoch {early_stopping.get_best_epoch()+1}"
			)
			break

		print(f"[Epoch {epoch+1} ELAPSED TIME (Train + Validation)]: {time.time() - train_and_val_st_time:.1f}s")

	print(f"[{tip_adapter_method}] Total Training Elapsed Time: {time.time() - train_start_time:.1f} sec")

	# Post-training (alpha, beta) validation grid search maximizing val I2T mAP@10 (Zhang et al., ECCV 2022, Sec. 3.2)
	if os.path.exists(mdl_fpth):
		ckpt = torch.load(mdl_fpth, map_location=device)
		model.load_state_dict(ckpt["model_state_dict"], strict=True)
		pre_search_map = _compute_tip_val_map_at_k(k=10)
		gs_alpha, gs_beta, gs_map = _grid_search_alpha_beta(k=10)
		if gs_map > pre_search_map + 1e-6:
			adapter_module.set_alpha_beta(gs_alpha, gs_beta)
			ckpt["model_state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
			ckpt["best_val_map10"] = gs_map
			ckpt["alpha"] = gs_alpha
			ckpt["beta"] = gs_beta
			torch.save(ckpt, mdl_fpth)
			if verbose:
				print(
					f"[{tip_adapter_method}] Post-training grid search improved val I2T mAP@10 "
					f"{pre_search_map:.4f} → {gs_map:.4f} (α={gs_alpha}, β={gs_beta}); updated checkpoint."
				)
		elif verbose:
			print(
				f"[{tip_adapter_method}] Keeping initial α={adapter_module.alpha.item():.4f}, "
				f"β={adapter_module.beta.item():.4f} (val I2T mAP@10={pre_search_map:.4f})."
			)

	best_model_eval_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		active_mask=active_mask,
		head_mask=head_mask,
		rare_mask=rare_mask,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=tip_adapter_method,
		device=device,
		cache_dir=results_dir,
		embeddings_cache=(val_feats, None),
		topk_values=topk_values,
		lora_params=None,
		temperature=temperature,
		use_tip_logits=True,
		use_fixed_masks=True,
		tier_spec_path=tier_spec_path,
		verbose=verbose,
	)

	actual_trained_epochs = len(training_losses)
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs,
	)

	print_final_evaluation_summary(
		mode=mode,
		method=tip_adapter_method,
		model_arch=model_arch,
		model=model,
		actual_trained_epochs=actual_trained_epochs,
		early_stopping=early_stopping,
		mdl_fpth=mdl_fpth,
		pw_mode=pw_mode,
		eval_results=best_model_eval_results,
		extra_lines=[
			f"Tip-Adapter-F: M={adapter_module.cache_keys.shape[0]:,} trainable cache keys | "
			f"final α={adapter_module.alpha.item():.4f}, β={adapter_module.beta.item():.4f}",
		],
		verbose=verbose,
	)

	# Generate plots
	file_base_name = (
		f"{tip_adapter_method}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"beta_{adapter_module.beta.item()}_"
		f"alpha_{adapter_module.alpha.item()}_"
		f"shots_{support_shots}_"
		f"temp_{temperature}"
	)

	plot_paths = {
		"losses": os.path.join(viz_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(viz_dir, f"{file_base_name}_losses_breakdown.png"),
		"retrieval_per_epoch": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(viz_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		"alpha_beta_evol": os.path.join(viz_dir, f"{file_base_name}_alpha_beta_evol.png"),
		"hp_evol": os.path.join(viz_dir, f"{file_base_name}_hp_evol.png"),
	}

	if actual_trained_epochs > 0:
		viz.plot_multilabel_loss_breakdown(
			training_losses_breakdown=training_losses_breakdown,
			filepath=plot_paths["losses_breakdown"],
		)
		viz.plot_retrieval_metrics_per_epoch(
			dataset_name=dataset_name,
			image_to_text_metrics_list=img2txt_metrics_all_epochs,
			text_to_image_metrics_list=txt2img_metrics_all_epochs,
			fname=plot_paths["retrieval_per_epoch"],
		)
		if learning_rates_history and weight_decays_history:
			viz.plot_hyperparameter_evolution(
				eta_min=eta_min,
				learning_rates=learning_rates_history,
				weight_decays=weight_decays_history,
				fname=plot_paths["hp_evol"],
			)

	viz.plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_model_eval_results["img2txt_metrics"],
		text_to_image_metrics=best_model_eval_results["txt2img_metrics"],
		fname=plot_paths["retrieval_best"],
	)
	viz.plot_train_val_losses(
		train_losses=training_losses,
		val_losses=validation_losses,
		fname=plot_paths["losses"],
	)
	if alphas and betas:
		viz.plot_alpha_beta_evolution(
			alphas=alphas,
			betas=betas,
			fname=plot_paths["alpha_beta_evol"],
		)

	return best_model_eval_results
