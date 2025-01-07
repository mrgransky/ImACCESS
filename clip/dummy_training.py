import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class EarlyStopping:
	def __init__(
			self,
			patience: int = 5, # epochs to wait before stopping the training
			min_delta: float = 1e-3, # minimum difference between new and old loss to count as improvement
			cumulative_delta: float = 0.01,
			window_size: int = 5,
			mode: str = 'min',
			min_epochs: int = 5,
			restore_best_weights: bool = True,
		):
		"""
		Args:
			patience: Number of epochs to wait before early stopping
			min_delta: Minimum change in monitored value to qualify as an improvement
			cumulative_delta: Minimum cumulative improvement over window_size epochs
			window_size: Size of the window for tracking improvement trends
			mode: 'min' for loss, 'max' for metrics like accuracy
			min_epochs: Minimum number of epochs before early stopping can trigger
			restore_best_weights: Whether to restore model to best weights when stopped
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights
		
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.value_history = []
		self.improvement_history = []
		
		self.sign = 1 if mode == 'min' else -1
	
	def is_improvement(self, current_value: float) -> bool:
		if self.best_score is None:
			return True
		improvement = (self.best_score - current_value) * self.sign
		return improvement > self.min_delta
	
	def calculate_trend(self) -> float:
		"""Calculate improvement trend over window"""
		if len(self.value_history) < self.window_size:
			return float('inf') if self.mode == 'min' else float('-inf')
		window = self.value_history[-self.window_size:]
		if self.mode == 'min':
			return sum(window[i] - window[i+1] for i in range(len(window)-1))
		return sum(window[i+1] - window[i] for i in range(len(window)-1))
	
	def should_stop(self, current_value: float, model: nn.Module, epoch: int) -> bool:
		"""
		Enhanced stopping decision based on multiple criteria
		"""
		self.value_history.append(current_value)

		# Before minimum epochs: [0-min_epochs]:
		# 1. Check if current_value is better than best_score
		# 2. If so, update best_score and best_weights
		# 3. Continue training
		if epoch < self.min_epochs:
			if self.best_score is None or current_value * self.sign < self.best_score * self.sign:
				self.best_score = current_value
				self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
				if self.restore_best_weights:
					self.best_weights = copy.deepcopy(model.state_dict())
			return False

		# 4. After minimum epochs: [min_epochs-end]:
		if self.is_improvement(current_value):
			self.best_score = current_value
			self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
			if self.restore_best_weights:
				self.best_weights = copy.deepcopy(model.state_dict())
			self.counter = 0
			self.improvement_history.append(True)
		else:
			self.counter += 1
			self.improvement_history.append(False)
		
		# Calculate trend over window
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		
		# Decision logic combining multiple factors
		should_stop = False
		
		# if no improvements by min_delta over counter exceeds paitience:
		if self.counter >= self.patience:
			should_stop = True
		
		# Check if stuck in local optimum
		if len(self.improvement_history) >= self.window_size:
			recent_improvements = sum(self.improvement_history[-self.window_size:])
			if recent_improvements == 0 and cumulative_improvement < self.cumulative_delta:
				should_stop = True
		
		# If stopping, restore best weights if configured
		if should_stop and self.restore_best_weights and self.best_weights is not None:
			model.load_state_dict(self.best_weights)
			self.stopped_epoch = epoch
		
		return should_stop
	
	def get_best_score(self) -> float:
		return self.best_score
	
	def get_stopped_epoch(self) -> int:
		return self.stopped_epoch

# Define a simple neural network
class SimpleNN(nn.Module):
		def __init__(self):
				super(SimpleNN, self).__init__()
				self.fc = nn.Linear(10, 1)

		def forward(self, x):
				return self.fc(x)

# Initialize model, loss function, optimizer, and early stopping
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
mdl_fpth = os.path.join("results", f"dummy_train.pth")

early_stopping = EarlyStopping(
	patience=10,          # Wait for 10 epochs without improvement before stopping
	min_delta=1e-4,       # Consider an improvement only if the change is greater than 0.0001
	cumulative_delta=5e-3,# Cumulative improvement over the window should be greater than 0.005
	window_size=10,       # Consider the last 10 epochs for cumulative trend
	mode='min',           # Minimize loss
	min_epochs=20,        # Ensure at least 20 epochs of training
	restore_best_weights=True  # Restore model weights to the best epoch
)

# Dummy data
data = torch.randn(int(1e+3), int(1e+1))
targets = torch.randn(int(1e+3), 1)
nEpochs = 100

# Training loop
for epoch in range(nEpochs):
		model.train()
		optimizer.zero_grad()
		outputs = model(data)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		print(f"Epoch {epoch}, Loss: {loss.item()}")

		# Early stopping check
		if early_stopping.should_stop(loss.item(), model, epoch):
			print(f"Early stopping at epoch {epoch}")
			print(f"Best: {early_stopping.get_best_score()} @ Epoch: {early_stopping.get_stopped_epoch()}")
			break