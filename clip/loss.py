
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

def compute_loss_masks(
	train_loader: DataLoader,
	num_classes: int,
	device: torch.device,
	pareto_threshold: float = 0.80,
	pw_rare_threshold: float = 20.0,
	verbose: bool = True,
) -> Dict[str, torch.Tensor]:
	"""
	Compute pos_weight, active_mask, head_mask, and rare_mask from
	training set label frequencies. Called internally by every
	fine-tuning function — no need to pass masks from outside.
	Returns dict with keys:
			pos_weight   [num_classes] float32  — for BCEWithLogitsLoss
			active_mask  [num_classes] bool     — freq > 0
			head_mask    [num_classes] bool     — Pareto 80% head classes
			rare_mask    [num_classes] bool     — pos_weight > pw_rare_threshold
			train_freq   [num_classes] float32  — raw frequencies (CPU)
			N            int                    — total training samples
	"""
	print("\nLabel frequencies from training set")
	train_freq = torch.zeros(num_classes, dtype=torch.float32)
	N = len(train_loader.dataset)
	for raw in train_loader.dataset.labels:
			try:
					for lbl in ast.literal_eval(raw):
							if lbl in train_loader.dataset.label_dict:
									train_freq[train_loader.dataset.label_dict[lbl]] += 1
			except (ValueError, SyntaxError):
					pass

	# pos_weight — capped at 1000 to avoid float16 overflow
	pos_weight = torch.where(
			train_freq > 0,
			((N - train_freq) / train_freq.clamp(min=1)).clamp(max=1000.0),
			torch.ones(num_classes),
	).to(device)

	# active_mask — classes with at least one training example
	active_mask = (train_freq > 0).to(device)

	# head_mask — Pareto classes covering pareto_threshold of occurrences
	sorted_freq, sorted_idx = torch.sort(train_freq, descending=True)
	cumsum = sorted_freq.cumsum(0)
	pareto_cutoff = (cumsum <= cumsum[-1] * pareto_threshold).sum().item() + 1
	head_indices = sorted_idx[:pareto_cutoff]
	head_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
	head_mask[head_indices] = True

	# rare_mask — learnable but imbalanced classes
	rare_mask = (pos_weight > pw_rare_threshold) & active_mask
	if verbose:
		print(f"  ├─ Total samples (N):        {N:,}")
		print(f"  ├─ pos_weight range:         [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
		print(f"  ├─ Active classes (freq>0):  {active_mask.sum().item():,} / {num_classes:,}")
		print(f"  ├─ Head (Pareto {pareto_threshold:.0%}):        {head_mask.sum().item():,}")
		print(f"  └─ Rare (pw>{pw_rare_threshold:.0f}):           {rare_mask.sum().item():,}")

	return {
			"pos_weight":  pos_weight,
			"active_mask": active_mask,
			"head_mask":   head_mask,
			"rare_mask":   rare_mask,
			"train_freq":  train_freq,
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

		image_embeds = torch.nn.functional.normalize(model.encode_image(images), dim=-1)
		class_embeds = torch.nn.functional.normalize(all_class_embeds, dim=-1)

		# I2T: [batch_size, num_classes]
		i2t_sim = torch.matmul(image_embeds, class_embeds.T) / temperature
		i2t_loss_raw = criterion_i2t(i2t_sim, label_vectors.float())  # [batch, C]
		loss_i2t = i2t_loss_raw[:, active_mask].mean()

		# T2I: [num_classes, batch_size]
		t2i_sim = torch.matmul(class_embeds, image_embeds.T) / temperature
		t2i_loss_raw = criterion_t2i(t2i_sim, label_vectors.T.float())  # [C, batch]
		loss_t2i = t2i_loss_raw[active_mask, :].mean()

		total_loss = loss_weights["i2t"] * loss_i2t + loss_weights["t2i"] * loss_t2i
		return total_loss, loss_i2t, loss_t2i

class LabelSmoothingBCELoss(torch.nn.Module):
	def __init__(self, smoothing: float = 0.1):
		super().__init__()
		self.smoothing = smoothing
			
	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			logits: [batch_size, num_classes] - raw logits
			targets: [batch_size, num_classes] - binary targets (0 or 1)
		"""
		# Apply label smoothing
		# Positive labels: 1 -> (1 - smoothing)
		# Negative labels: 0 -> smoothing
		smooth_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
		
		# Apply BCE loss with logits
		loss = F.binary_cross_entropy_with_logits(logits, smooth_targets)
		return loss