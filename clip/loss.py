
from utils import *

def diagnose_train_val_coverage(
	train_freq: torch.Tensor,
	validation_loader,
	num_classes: int,
	verbose: bool = True,
) -> torch.Tensor:
	"""
	Returns val_freq tensor and prints coverage diagnostic.
	Call once after compute_loss_masks() in each fine-tuning function.
	"""
	val_freq = torch.zeros(num_classes, dtype=torch.float32)

	for raw in validation_loader.dataset.labels:
		try:
			for lbl in ast.literal_eval(raw):
				if lbl in validation_loader.dataset.label_dict:
					val_freq[validation_loader.dataset.label_dict[lbl]] += 1
		except (ValueError, SyntaxError):
			pass

	val_active   = (val_freq > 0)
	train_active = (train_freq > 0)
	train_only   = (train_active & ~val_active).sum().item()
	val_only     = (val_active & ~train_active).sum().item()
	both_active  = (train_active & val_active).sum().item()
	neither      = (~train_active & ~val_active).sum().item()

	if verbose:
		print(f"\n[COVERAGE DIAGNOSTIC]")
		print(f"  ├─ Total train labels         : {len(train_freq>0)}")
		print(f"  ├─ Total val labels           : {len(val_freq>0)}")
		print(f"  ├─ Active in both train & val : {both_active}")
		print(f"  ├─ Active in train only       : {train_only}")
		print(f"  ├─ Active in val only         : {val_only}")
		print(f"  └─ Inactive in both           : {neither}")

		if train_only > 0:
			print(f"\n{train_only} labels trained on but absent from val.")

		if val_only > 0:
			print(
				f"{val_only} labels ONLY present in val with no training samples => "
				f"pos_weight defaults to 1.0 for these."
			)
		
		# Train-only class frequency analysis
		if train_only > 0:
			train_only_mask  = (train_active & ~val_active)
			train_only_freqs = train_freq[train_only_mask]
			print(f"\n[ANALYSIS] Train-only label frequency")
			print(f"  ├─ Count            : {train_only_mask.sum().item():,}")
			print(f"  ├─ Freq (min, max)  : ({train_only_freqs.min():.1f}, {train_only_freqs.max():.1f})")
			print(f"  ├─ Freq (mean, std) : ({train_only_freqs.mean():.1f}, {train_only_freqs.std():.1f}) median: {train_only_freqs.median():.1f}")
			print(f"  ├─ label(s) freq=1  : {(train_only_freqs == 1).sum().item():,}")
			print(f"  ├─ label(s) freq≤5  : {(train_only_freqs <= 5).sum().item():,}")
			print(f"  └─ label(s) freq>10 : {(train_only_freqs > 10).sum().item():,}")
		
		# Val-only class frequency analysis
		if val_only > 0:
			val_only_mask  = (val_active & ~train_active)
			val_only_freqs = val_freq[val_only_mask]
			print(f"\n[ANALYSIS] Val-only label frequency")
			print(f"  ├─ Count            : {val_only_mask.sum().item():,}")
			print(f"  ├─ Freq (min, max)  : ({val_only_freqs.min():.1f}, {val_only_freqs.max():.1f})")
			print(f"  └─ Freq (mean, std) : ({val_only_freqs.mean():.1f}, {val_only_freqs.std():.1f}) median: {val_only_freqs.median():.1f}")
			print(
				f" [NOTE]: these labels will appear in rare tier evaluation "
				f"but model has no positive training signal for them."
			)
	
	return val_freq

def compute_loss_masks(
	train_loader: DataLoader,
	validation_loader: DataLoader,
	num_classes: int,
	device: torch.device,
	pw_mode: str = "log", # "log" | "sqrt" | "linear"
	pw_max_cap: Optional[float]=None,
	pareto_threshold: float = 0.8,
	rare_percentile: float = 0.2, # bottom X% of active classes by frequency → rare
	verbose: bool = True,
) -> Dict[str, torch.Tensor]:
	"""
	Compute training loss weights and evaluation tier masks from training label frequencies.
	Two concerns are kept strictly separate:
		1. pos_weight  — loss weighting, depends on pw_mode (training only)
		2. head/rare   — evaluation tiers, based purely on frequency (all strategies)
	Args:
			loader     : DataLoader whose .dataset has .labels and .label_dict
			num_classes      : Total number of classes (including inactive)
			device           : Target device for returned tensors
			pareto_threshold : Cumulative frequency fraction defining "head" classes (default 80%)
			rare_percentile  : Bottom fraction of active classes by frequency → "rare" (default 20%)
			pw_mode          : Loss weighting strategy:
													 "log"    → log1p(ratio)          range ~[0, 11]   probe/adapters/IA3/VeRA
													 "sqrt"   → sqrt(ratio).clamp(max=pw_max_cap)  range ~[1, cap=50] LoRA/LoRA+/DoRA/RSLora
													 "linear" → ratio.clamp(pw_max_cap) range ~[1, cap=100] full fine-tuning
			pw_max_cap       : Hard cap for "linear" mode (ignored otherwise)
			verbose          : Print summary statistics
	Returns dict with keys:
			pos_weight   [num_classes] float32  cuda — for BCEWithLogitsLoss (training only)
			active_mask  [num_classes] bool     cuda — freq > 0
			head_mask    [num_classes] bool     cuda — Pareto top classes by cumulative frequency
			rare_mask    [num_classes] bool     cuda — bottom rare_percentile of active classes
			train_freq   [num_classes] float32  cpu  — raw per-class counts
			N            int                        — total training samples
	"""
	
	# 1. Count label frequencies
	N = len(train_loader.dataset)

	train_freq = torch.zeros(num_classes, dtype=torch.float32)
	for i, raw in enumerate(train_loader.dataset.labels):
		# print(i, raw) # 522 ['railroad', 'train', 'station']
		try:
			for lbl in ast.literal_eval(raw):
				if lbl in train_loader.dataset.label_dict:
					idx = train_loader.dataset.label_dict[lbl]
					# print(f"\t{lbl}, {idx}") # railroad, 107
					train_freq[idx] += 1
		except (ValueError, SyntaxError):
			pass

	if train_freq.sum() == 0:
		raise ValueError(
			f"No valid labels found in dataset. Check that:\n"
			f"  1. train_loader.dataset.labels contains valid data\n"
			f"  2. train_loader.dataset.label_dict is populated\n"
			f"  3. Labels in dataset match keys in label_dict"
		)

	if verbose:
		print(f"\n[LOSS MASKING] Train Frequency:")
		print(f"  ├─ {type(train_freq)} {train_freq.shape} {train_freq.dtype} {train_freq.device}")
		print(f"  ├─ (min, max): ({train_freq.min()}, {train_freq.max()}) sum: {train_freq.sum()}")
		print(f"  └─ mean: {train_freq.mean():.2f}, std: {train_freq.std():.2f}, median: {train_freq.median()}")

	# 2. active_mask — classes with at least one training example
	active_mask = (train_freq > 0).to(device)
	ratio = (N - train_freq) / train_freq.clamp(min=1)

	if verbose:
		print(f"\n[LOSS MASKING] Raw Ratio:")
		print(f"  ├─ {type(ratio)} {ratio.shape} {ratio.dtype} {ratio.device}")
		print(f"  ├─ (min, max): ({ratio.min():.2f}, {ratio.max():.2f}) sum: {ratio.sum():.2f}")
		print(f"  └─ mean: {ratio.mean():.2f}, std: {ratio.std():.2f} median: {ratio.median()}")

	# 3. pos_weight — training loss weighting only
	if pw_mode == "log":
		scaled = torch.log1p(ratio)
	elif pw_mode == "sqrt":
		scaled = torch.sqrt(ratio)
	elif pw_mode == "linear":
		scaled = ratio
	else:
		raise ValueError(f"Unknown pw_mode '{pw_mode}'. Choose from: 'log', 'sqrt', 'linear'.")

	if pw_max_cap:
		if verbose:
			print(f"pw_max_cap: {pw_max_cap}")
		scaled = scaled.clamp(min=1.0, max=pw_max_cap)

	if verbose:
		print(f"\n[LOSS MASKING] Scaled: (pw_mode: {pw_mode})")
		print(f"  ├─ {type(scaled)} {scaled.shape} {scaled.dtype} {scaled.device}")
		print(f"  ├─ (min, max): ({scaled.min():.2f}, {scaled.max():.2f}) sum: {scaled.sum():.2f}")
		print(f"  └─ mean: {scaled.mean():.2f}, std: {scaled.std():.2f} median: {scaled.median()}")

	# inactive classes always get weight 1.0 (they are masked out in the loss anyway)
	pos_weight = torch.where(
		train_freq > 0,
		scaled,
		torch.ones(num_classes),
	).to(device)

	# 4. head_mask — Pareto top classes by cumulative frequency
	# "head" = fewest classes that together account for pareto_threshold of all occurrences
	sorted_freq, sorted_idx = torch.sort(train_freq, descending=True)
	cumsum = sorted_freq.cumsum(0)
	pareto_cutoff = int((cumsum <= cumsum[-1] * pareto_threshold).sum().item()) + 1
	head_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
	head_mask[sorted_idx[:pareto_cutoff]] = True

	# 5. rare_mask — bottom rare_percentile of ACTIVE classes by frequency ─
	# Fully decoupled from pos_weight — stable across all strategies and zero-shot
	active_freq = train_freq[active_mask.cpu()]   # CPU tensor, active classes only
	if active_freq.numel() > 1:
		freq_threshold = torch.quantile(active_freq, rare_percentile)
		rare_mask = ((train_freq <= freq_threshold) & (train_freq > 0)).to(device)
	else:
		# degenerate dataset — no rare classes
		rare_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
	
	train_loader_name = getattr(train_loader, 'name', 'UNNAMED_LOADER')
	if verbose:
		print(f"\n[LOSS MASKING] Label frequencies {train_loader_name}")
		print(f"  ├─ Total samples:            {N:,}")
		print(f"  ├─ Total labels:             {num_classes:,}")
		print(f"  ├─ Train freq:               [{train_freq.min():.1f}, {train_freq.max():.1f}] μ={train_freq.mean():.1f} σ={train_freq.std():.1f} {type(train_freq)} {train_freq.shape} {train_freq.dtype}, {train_freq.device}")
		print(f"  ├─ Raw Ratio:                [{ratio.min():.1f}, {ratio.max():.1f}] μ={ratio.mean():.1f} σ={ratio.std():.1f} {type(ratio)} {ratio.shape}")
		print(f"  ├─ pw_mode:                  {pw_mode}")
		if pw_max_cap:
			print(f"  ├─ pw_max_cap:               {pw_max_cap}")
		print(f"  ├─ Scaled:                   [{scaled.min():.1f}, {scaled.max():.1f}] μ={scaled.mean():.1f} σ={scaled.std():.1f} {type(scaled)} {scaled.shape}")
		print(f"  ├─ pos_weight:               [{pos_weight.min():.1f}, {pos_weight.max():.1f}] μ={pos_weight.mean():.1f} σ={pos_weight.std():.1f} {type(pos_weight)} {pos_weight.shape}")
		print(f"  ├─ Active classes (freq>0):  {active_mask.sum().item():,} / {num_classes:,} {type(active_mask)} {active_mask.shape}")
		print(f"  ├─ Head  (Pareto {pareto_threshold}):       {head_mask.sum().item():,}")
		print(f"  └─ Rare  (bottom {rare_percentile}):       {rare_mask.sum().item():,}")

	diagnose_train_val_coverage(
		train_freq=train_freq,
		validation_loader=validation_loader,
		num_classes=num_classes,
		verbose=verbose,
	)

	return {
		"active_mask": active_mask,
		"head_mask":   head_mask, # smallest label set covering 80% of training-label occurrences
		"rare_mask":   rare_mask, # labels at or below 20th percentile of positive training frequencies
		"train_freq":  train_freq,
		"pos_weight":  pos_weight,
		"N":           N,
	}

def compute_multilabel_validation_loss(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	criterion_i2t,      # BCEWithLogitsLoss with pos_weight, reduction='none'
	criterion_t2i,      # BCEWithLogitsLoss plain, reduction='none'
	active_mask,        # [num_classes] bool
	device: str,
	all_class_embeds: torch.Tensor,
	temperature: float,
	verbose: bool = False,
) -> float:

	model.eval()
	total_loss = 0.0
	total_samples = 0
	
	max_batches = max(50, len(validation_loader) // 10)
	if verbose:
		print(f"\nMultilabel validation loss")
		print(f"  {type(model)} {model.name}")
		print(f"  {validation_loader.name} {len(validation_loader)} batches")
		print(f"  max_batches: {max_batches}")
		print(f"  active_mask: {active_mask.shape} {active_mask.sum()}/{len(active_mask)}")
		print(f"  all_class_embeds: {all_class_embeds.shape} {all_class_embeds.device}")

	with torch.no_grad():
		for batch_idx, (images, _, label_vectors) in enumerate(validation_loader):
			if batch_idx >= max_batches:
				break
			
			batch_size = images.size(0)
			if batch_size == 0:  # Skip empty batches
				continue
					
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Encode images
			image_embeds = model.encode_image(images)
			image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
			
			# Cast to FP32 before similarity + BCE computation
			image_embeds = image_embeds.float()
			# all_class_embeds = all_class_embeds.float() # already FP32 — no cast needed

			# Compute similarities
			i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
			t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature
			
			# Compute losses
			i2t_targets = label_vectors
			t2i_targets = label_vectors.T
			
			i2t_loss_raw = criterion_i2t(i2t_similarities, i2t_targets) # [B, C]
			loss_i2t = i2t_loss_raw[:, active_mask].mean()

			t2i_loss_raw = criterion_t2i(t2i_similarities, t2i_targets) # [C, B]
			loss_t2i = t2i_loss_raw[active_mask, :].mean()

			batch_loss = 0.5 * (loss_i2t + loss_t2i)
			
			# Correct accumulation
			total_loss += batch_loss.item() * batch_size
			total_samples += batch_size
	
	avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

	return avg_loss

def compute_multilabel_contrastive_loss(
	model,
	images,
	all_class_embeds,
	label_vectors,
	criterion_i2t, # with pos_weight
	criterion_t2i, # without pos_weight
	active_mask,
	temperature,
	loss_weights=None,
	verbose=False,
):
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}

	# # Embeddings (may be FP16 from autocast)
	image_embeds = torch.nn.functional.normalize(model.encode_image(images), dim=-1)
	class_embeds = torch.nn.functional.normalize(all_class_embeds, dim=-1)

	# Cast to FP32 BEFORE similarity computation
	# prevents FP16 overflow in BCEWithLogitsLoss
	image_embeds = image_embeds.float()
	# class_embeds = class_embeds.float() # already FP32 — no cast needed

	# I2T: [batch_size, num_classes]
	i2t_sim = torch.matmul(image_embeds, class_embeds.T) / temperature
	i2t_loss_raw = criterion_i2t(i2t_sim, label_vectors.float())  # [batch, C]
	loss_i2t = i2t_loss_raw[:, active_mask].mean()

	# T2I: [num_classes, batch_size]
	t2i_sim = torch.matmul(class_embeds, image_embeds.T) / temperature
	t2i_loss_raw = criterion_t2i(t2i_sim, label_vectors.T.float())  # [C, batch]
	loss_t2i = t2i_loss_raw[active_mask, :].mean()

	# Total loss
	total_loss = loss_weights["i2t"] * loss_i2t + loss_weights["t2i"] * loss_t2i

	return total_loss, loss_i2t, loss_t2i