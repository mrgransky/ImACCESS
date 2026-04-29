
from utils import *

class LossAnalyzer:
	def __init__(self, epochs, train_loss, val_loss):
		self.epochs = np.array(epochs)
		self.train_loss = np.array(train_loss)
		self.val_loss = np.array(val_loss)
		self.ema_window = 10
		self.ema_threshold = 1e-3
			
	def sma(self, data, window):
		return pd.Series(data).rolling(window=window, min_periods=1).mean().values
	
	def ema(self, data, window, alpha=None):
		if alpha is None:
			alpha = 2.0 / (window + 1)
		
		ema = np.zeros_like(data)
		ema[0] = data[0]
		
		for i in range(1, len(data)):
			ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
		
		return ema
	
	def plot_analysis(self, windows=[5, 10, 20], fpth='loss_analysis.png', figsize=(11, 7)):
		fpth = fpth.replace("_loss_analyzer.png", "")
		cols = plt.cm.tab10(np.linspace(0, 1, len(windows) + 1))

		# Plot 1: Training Loss - SMA
		plt.figure(figsize=figsize)
		plt.plot(self.epochs, self.train_loss, alpha=0.3, label='Raw', color="#493C66")
		for i, window in enumerate(windows):
			sma = self.sma(self.train_loss, window)
			plt.plot(self.epochs, sma, label=f'SMA-{window}', linewidth=1, color=cols[i])
		
		plt.title('Training Loss - SMA')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_train_sma.png', dpi=200, bbox_inches='tight')
		plt.close()
		
		# Plot 2: Training Loss - EMA
		plt.figure(figsize=figsize)
		plt.plot(self.epochs, self.train_loss, alpha=0.3, label='Raw', color="#493C66")
		for i, window in enumerate(windows):
			ema = self.ema(self.train_loss, window)
			plt.plot(self.epochs, ema, label=f'EMA-{window}', linewidth=1, color=cols[i])
		
		plt.title('Training Loss - EMA')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_train_ema.png', dpi=200, bbox_inches='tight')
		plt.close()

		# Plot 3: Validation Loss - SMA
		plt.figure(figsize=figsize)
		plt.plot(self.epochs, self.val_loss, alpha=0.3, label='Raw', color="#493C66")
		for i, window in enumerate(windows):
			sma = self.sma(self.val_loss, window)
			plt.plot(self.epochs, sma, linestyle='--', label=f'SMA-{window}', linewidth=2.5, color=cols[i])
		
		plt.title('Validation Loss - SMA')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_val_sma.png', dpi=200, bbox_inches='tight')
		plt.close()
		
		# Plot 4: Validation Loss - EMA
		plt.figure(figsize=figsize)
		plt.plot(self.epochs, self.val_loss, alpha=0.3, label='Raw', color="#493C66")
		for i, window in enumerate(windows):
			ema = self.ema(self.val_loss, window)
			plt.plot(self.epochs, ema, linestyle='-', label=f'EMA-{window}', linewidth=1, color=cols[i])
		
		plt.title('Validation Loss - EMA')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_val_ema.png', dpi=200, bbox_inches='tight')
		plt.close()

		# Plot 5: Combined Smoothed Comparison
		plt.figure(figsize=figsize)
		train_ema = self.ema(self.train_loss, 10)
		val_ema = self.ema(self.val_loss, 10)
		plt.plot(self.epochs, train_ema, label='Training EMA-10', linewidth=2.5)
		plt.plot(self.epochs, val_ema, label='Validation EMA-10', linewidth=2.5)
		
		plt.title('Smoothed Comparison (EMA-10)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_smoothed_comp.png', dpi=200, bbox_inches='tight')
		plt.close()
		
		# Plot 6: Overfitting Detection
		plt.figure(figsize=figsize)
		gap = val_ema - train_ema
		# print(f"Train EMA: {train_ema}\nVal EMA: {val_ema}\nGap: {gap}")
		plt.plot(self.epochs, gap, color="#000000", linewidth=1.5, label='Val - Train Gap')
		# plt.plot(self.epochs, np.zeros_like(gap), color="#000000", linewidth=1.5, linestyle='--', label='Zero Line')
		plt.plot(self.epochs, val_ema, label='Validation EMA-10', linewidth=1.5, linestyle='--', color="#C77203")
		plt.plot(self.epochs, train_ema, label='Training EMA-10', linewidth=1.5, linestyle='--', color="#0025FA")
		plt.axhline(y=0, color="#838282", linestyle='-', alpha=0.5)
		plt.fill_between(self.epochs, gap, 0, where=(gap > 0), alpha=0.3, color="#FD5C5C", label='Overfitting Zone')
		
		plt.title('Overfitting Detection (Val - Train EMA-10)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss Difference')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_overfit_gap.png', dpi=200, bbox_inches='tight')
		plt.close()
		
		# Plot 7: Volatility Analysis
		plt.figure(figsize=figsize)
		val_volatility = pd.Series(self.val_loss).rolling(window=10).std()
		train_volatility = pd.Series(self.train_loss).rolling(window=10).std()
		plt.plot(self.epochs, val_volatility, label='Val Volatility', linewidth=1.5)
		plt.plot(self.epochs, train_volatility, label='Train Volatility', linewidth=1.5)
		
		plt.title('Loss Volatility Analysis')
		plt.xlabel('Epochs')
		plt.ylabel('Loss Volatility (Std Dev)')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_volatility.png', dpi=200, bbox_inches='tight')
		plt.close()
		
	def get_training_signals(self):
		signals = {}
		
		# Best epoch using EMA
		val_ema = self.ema(self.val_loss, self.ema_window)
		best_idx = np.argmin(val_ema)
		signals['best_epoch'] = self.epochs[best_idx]
		signals['best_loss'] = val_ema[best_idx]
		
		# Overfitting gap
		train_ema = self.ema(self.train_loss, self.ema_window)
		signals['overfitting_gap'] = val_ema[-1] - train_ema[-1] # (positive = over‑fit)
		
		# Recent trend
		recent_trend = np.mean(np.diff(val_ema[-self.ema_window:]))
		signals['recent_trend'] = recent_trend # (positive = improving)
		
		# Recommendations
		if recent_trend > self.ema_threshold:
			signals['recommendation'] = f"STOP - Validation loss increasing ({recent_trend}) > {self.ema_threshold}"
		elif recent_trend > -self.ema_threshold:
			signals['recommendation'] = f"CAUTION - Loss plateauing ({recent_trend}) > -{self.ema_threshold}"
		else:
			signals['recommendation'] = f"CONTINUE - Still improving ({recent_trend})"
				
		return signals

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
		print(f"\nTrain/Val Class Coverage")
		print(f"  ├─ Active in both train and val : {both_active:,}")
		print(f"  ├─ Active in train only         : {train_only:,}")
		print(f"  ├─ Active in val only           : {val_only:,}")
		print(f"  └─ Inactive in both             : {neither:,}")
		if train_only > 0:
			print(f"\n  ⚠  {train_only} classes trained on but absent from val.")
		if val_only > 0:
			print(
				f"\n  ⚠  {val_only} classes in val with no training samples — "
				f"pos_weight defaults to 1.0 for these."
			)
		
		# Train-only class frequency analysis
		if train_only > 0:
			train_only_mask  = (train_active & ~val_active)
			train_only_freqs = train_freq[train_only_mask]
			print(f"\n  [Train-only class frequency analysis]")
			print(f"  ├─ Count                : {train_only_mask.sum().item():,}")
			print(f"  ├─ Frequency min        : {train_only_freqs.min():.0f}")
			print(f"  ├─ Frequency max        : {train_only_freqs.max():.0f}")
			print(f"  ├─ Frequency mean       : {train_only_freqs.mean():.1f}")
			print(f"  ├─ Frequency median     : {train_only_freqs.median():.1f}")
			print(f"  ├─ Classes with freq=1  : {(train_only_freqs == 1).sum().item():,}")
			print(f"  ├─ Classes with freq≤5  : {(train_only_freqs <= 5).sum().item():,}")
			print(f"  └─ Classes with freq>10 : {(train_only_freqs > 10).sum().item():,}")
		
		# Val-only class frequency analysis
		if val_only > 0:
			val_only_mask  = (val_active & ~train_active)
			val_only_freqs = val_freq[val_only_mask]
			print(f"\n  [Val-only class frequency analysis]")
			print(f"  ├─ Count                : {val_only_mask.sum().item():,}")
			print(f"  ├─ Val frequency min    : {val_only_freqs.min():.0f}")
			print(f"  ├─ Val frequency max    : {val_only_freqs.max():.0f}")
			print(f"  └─ Val frequency mean   : {val_only_freqs.mean():.1f}")
			print(
				f"  Note: these classes will appear in rare tier evaluation "
				f"but model has no positive training signal for them."
			)
	
	return val_freq

def compute_loss_masks(
	train_loader: DataLoader,
	num_classes: int,
	device: torch.device,
	pw_mode: str = "log",                # "log" | "sqrt" | "linear"
	pw_max_cap: Optional[float]=None,
	pareto_threshold: float = 0.8,
	rare_percentile: float = 0.2,       # bottom X% of active classes by frequency → rare
	verbose: bool = True,
) -> Dict[str, torch.Tensor]:
	"""
	Compute training loss weights and evaluation tier masks from training label frequencies.
	Two concerns are kept strictly separate:
		1. pos_weight  — loss weighting, depends on pw_mode (training only)
		2. head/rare   — evaluation tiers, based purely on frequency (all strategies)
	Args:
			train_loader     : DataLoader whose .dataset has .labels and .label_dict
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
	print(f"N: {N}")

	train_freq = torch.zeros(num_classes, dtype=torch.float32)
	for i, raw in enumerate(train_loader.dataset.labels):
		print(i, raw) # 522 ['railroad', 'train', 'station']
		try:
			for lbl in ast.literal_eval(raw):
				if lbl in train_loader.dataset.label_dict:
					idx = train_loader.dataset.label_dict[lbl]
					print(f"\t{lbl}, {idx}") # railroad, 107
					train_freq[idx] += 1
		except (ValueError, SyntaxError):
			pass
	print(f"train_freq: {type(train_freq)} {train_freq.shape} (min, max): ({train_freq.min():.2f}, {train_freq.max():.2f}) mean: {train_freq.mean():.2f} std: {train_freq.std():.2f}")

	# 2. active_mask — classes with at least one training example
	active_mask = (train_freq > 0).to(device)
	ratio = (N - train_freq) / train_freq.clamp(min=1)

	print(f"raw ratio: {type(ratio)} {ratio.shape} (min, max): ({ratio.min():.2f}, {ratio.max():.2f}) mean: {ratio.mean():.2f} std: {ratio.std():.2f}")

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

	print(f"scaled: {type(scaled)} {scaled.shape} (min, max): ({scaled.min():.2f}, {scaled.max():.2f}) mean: {scaled.mean():.2f} std: {scaled.std():.2f}")

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
	
	loader_name = getattr(train_loader, 'name', 'UNNAMED_LOADER')
	if verbose:
		print(f"\nLabel frequencies {loader_name}")
		print(f"  ├─ samples (N):              {N:,}")
		print(f"  ├─ classes (C):              {num_classes:,}")
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

	return {
		"active_mask": active_mask,
		"head_mask":   head_mask,
		"rare_mask":   rare_mask,
		"train_freq":  train_freq,
		"pos_weight":  pos_weight,
		"N":           N,
	}

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