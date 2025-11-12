
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

def compute_multilabel_contrastive_loss(
		model: torch.nn.Module,
		images: torch.Tensor,
		all_class_embeds: torch.Tensor,
		label_vectors: torch.Tensor,
		criterion: torch.nn.Module,
		temperature: float,
		loss_weights: Dict[str, float] = None,
		verbose: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Compute bidirectional multi-label contrastive loss.
	
	Args:
		model: CLIP model
		images: [batch_size, 3, 224, 224]
		all_class_embeds: [num_classes, embed_dim] - pre-computed text embeddings
		label_vectors: [batch_size, num_classes] - binary label matrix
		criterion: Loss function (BCEWithLogitsLoss)
		temperature: Temperature scaling for similarities
		loss_weights: Weights for I2T and T2I losses
		verbose: Print debug info
	Returns:
		Tuple of (total_loss, i2t_loss, t2i_loss)
	"""
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	batch_size, num_classes = label_vectors.shape
	if verbose:
		print(f"batch_size: {batch_size}, num_classes: {num_classes}")

	# Encode images
	image_embeds = model.encode_image(images)  # [batch_size, embed_dim]
	image_embeds = F.normalize(image_embeds, dim=-1)
	if verbose:
		print(f"image_embeds: {image_embeds.shape} {image_embeds.dtype} {image_embeds.device}")
	
	all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	if verbose:
		print(f"all_class_embeds: {all_class_embeds.shape} {all_class_embeds.dtype} {all_class_embeds.device}")

	# ================================
	# Image-to-Text Loss
	# ================================
	# Compute similarity matrix: [batch_size, num_classes]
	i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
	
	if verbose:
		print(f"i2t_similarities: {i2t_similarities.shape} {i2t_similarities.dtype} {i2t_similarities.device}")

	# I2T targets: label_vectors directly [batch_size, num_classes]
	i2t_targets = label_vectors.float()
	
	# Compute I2T loss
	loss_i2t = criterion(i2t_similarities, i2t_targets)
	
	# ================================
	# Text-to-Image Loss  
	# ================================
	# Compute similarity matrix: [num_classes, batch_size]
	t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature

	if verbose:
		print(f"t2i_similarities: {t2i_similarities.shape} {t2i_similarities.dtype} {t2i_similarities.device}")
	
	# T2I targets: transpose of label_vectors [num_classes, batch_size]
	t2i_targets = label_vectors.T.float()
	
	# Compute T2I loss
	loss_t2i = criterion(t2i_similarities, t2i_targets)

	total_loss = (loss_weights["i2t"] * loss_i2t) + (loss_weights["t2i"] * loss_t2i)

	if verbose:
		print(f"loss_i2t: {loss_i2t.item()} loss_t2i: {loss_t2i.item()} total_loss: {total_loss.item()}")
		print(f"requires_grad total_loss: {total_loss.requires_grad} loss_i2t: {loss_i2t.requires_grad} loss_t2i: {loss_t2i.requires_grad}")
		print("-"*60)
	
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