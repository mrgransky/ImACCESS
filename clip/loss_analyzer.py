
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
	
	def plot_analysis(self, windows=[5, 10, 20], fpth='loss_analysis.png', figsize=(10, 6)):
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
		plt.savefig(f'{fpth}_smoothed_comparison.png', dpi=200, bbox_inches='tight')
		plt.close()
		
		# Plot 6: Overfitting Detection
		plt.figure(figsize=figsize)
		gap = val_ema - train_ema
		plt.plot(self.epochs, gap, color="#000000", linewidth=1.5, label='Val - Train Gap')
		plt.axhline(y=0, color="#000000", linestyle='-', alpha=0.5)
		plt.fill_between(self.epochs, gap, 0, where=(gap > 0), alpha=0.3, color="#FF0000", label='Overfitting Zone')
		
		plt.title('Overfitting Detection (Val - Train EMA-10)')
		plt.xlabel('Epochs')
		plt.ylabel('Loss Difference')
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.savefig(f'{fpth}_overfitting_gap.png', dpi=200, bbox_inches='tight')
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

'''
	# Example usage:
	analyzer = LossAnalyzer(epochs, train_loss, val_loss)
	analyzer.plot_analysis()
	signals = analyzer.get_training_signals()
'''
