
from utils import *

class LossAnalyzer:
	def __init__(self, epochs, train_loss, val_loss):
		self.epochs = np.array(epochs)
		self.train_loss = np.array(train_loss)
		self.val_loss = np.array(val_loss)			
			
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
	
	def plot_analysis(self, windows=[5, 10, 20], fpth='loss_analysis.png'):
		fig, axes = plt.subplots(2, 3, figsize=(18, 10))
		
		# Training loss with moving averages
		axes[0,0].plot(self.epochs, self.train_loss, alpha=0.3, label='Raw', color='gray')
		for i, window in enumerate(windows):
			sma = self.sma(self.train_loss, window)
			ema = self.ema(self.train_loss, window)
			axes[0,0].plot(self.epochs, sma, label=f'SMA-{window}', linewidth=2)
			axes[0,1].plot(self.epochs, ema, label=f'EMA-{window}', linewidth=2)
		
		axes[0,0].set_title('Training Loss - SMA')
		axes[0,0].legend()
		axes[0,0].grid(True, alpha=0.3)
		
		axes[0,1].plot(self.epochs, self.train_loss, alpha=0.3, label='Raw', color='gray')
		axes[0,1].set_title('Training Loss - EMA')
		axes[0,1].legend()
		axes[0,1].grid(True, alpha=0.3)
		
		# Validation loss with moving averages
		axes[0,2].plot(self.epochs, self.val_loss, alpha=0.3, label='Raw', color='gray')
		for window in windows:
			sma = self.sma(self.val_loss, window)
			axes[0,2].plot(self.epochs, sma, label=f'SMA-{window}', linewidth=2)
		axes[0,2].set_title('Validation Loss - SMA')
		axes[0,2].legend()
		axes[0,2].grid(True, alpha=0.3)
		
		# Combined smooth view
		train_ema = self.ema(self.train_loss, 10)
		val_ema = self.ema(self.val_loss, 10)
		axes[1,0].plot(self.epochs, train_ema, label='Training EMA-10', linewidth=3)
		axes[1,0].plot(self.epochs, val_ema, label='Validation EMA-10', linewidth=3)
		axes[1,0].set_title('Smoothed Comparison')
		axes[1,0].legend()
		axes[1,0].grid(True, alpha=0.3)
		
		# Overfitting detection
		gap = val_ema - train_ema
		axes[1,1].plot(self.epochs, gap, color='purple', linewidth=2)
		axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
		axes[1,1].fill_between(self.epochs, gap, 0, where=(gap > 0), alpha=0.3, color='red')
		axes[1,1].set_title('Overfitting Gap (Val - Train)')
		axes[1,1].grid(True, alpha=0.3)
		
		# Volatility analysis
		val_volatility = pd.Series(self.val_loss).rolling(window=10).std()
		train_volatility = pd.Series(self.train_loss).rolling(window=10).std()
		axes[1,2].plot(self.epochs, val_volatility, label='Val Volatility', linewidth=2)
		axes[1,2].plot(self.epochs, train_volatility, label='Train Volatility', linewidth=2)
		axes[1,2].set_title('Loss Volatility')
		axes[1,2].legend()
		axes[1,2].grid(True, alpha=0.3)
		
		plt.savefig(fpth, dpi=200, bbox_inches='tight')
	
	def get_training_signals(self, window=10, threshold=1e-3):
		signals = {}
		
		# Best epoch using EMA
		val_ema = self.ema(self.val_loss, window)
		best_idx = np.argmin(val_ema)
		signals['best_epoch'] = self.epochs[best_idx]
		signals['best_loss'] = val_ema[best_idx]
		
		# Overfitting gap
		train_ema = self.ema(self.train_loss, window)
		signals['overfitting_gap'] = val_ema[-1] - train_ema[-1] # (positive = over‑fit)
		
		# Recent trend
		recent_trend = np.mean(np.diff(val_ema[-window:]))
		signals['recent_trend'] = recent_trend # (positive = improving)
		
		# Recommendations
		if recent_trend > threshold:
			signals['recommendation'] = f"STOP - Validation loss increasing ({recent_trend}) > {threshold}"
		elif recent_trend > -threshold:
			signals['recommendation'] = f"CAUTION - Loss plateauing ({recent_trend}) > -{threshold}"
		else:
			signals['recommendation'] = f"CONTINUE - Still improving ({recent_trend}) < {threshold}"
				
		return signals

'''
	# Example usage:
	analyzer = LossAnalyzer(epochs, train_loss, val_loss)
	analyzer.plot_analysis()
	signals = analyzer.get_training_signals()
'''
