from utils import *
from dataset_loader import get_dataloaders
from trainer import finetune, train

parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--num_workers', '-nw', type=int, default=18, help='Number of CPUs [def: max cpus]')
parser.add_argument('--num_epochs', '-ne', type=int, default=7, help='Number of epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
parser.add_argument('--print_every', type=int, default=150, help='Print loss')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune'], default='finetune', help='Choose mode (train/finetune)')

args, unknown = parser.parse_known_args()
args.device = torch.device(args.device)
print(args)

# run in pouta:
# train from scratch:
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -ne 32 -lr 1e-5 -wd 1e-3 --print_every 100 -nw 40 --device "cuda:2" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_train.out &

# finetune:
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -ne 128  -lr 1e-4 -wd 1e-2 --print_every 150 -nw 40 --device "cuda:2" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_ft_vitb32.out &

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
		# Don't stop before minimum epochs
		if epoch < self.min_epochs:
			if self.best_score is None or current_value * self.sign < self.best_score * self.sign:
				self.best_score = current_value
				self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
				if self.restore_best_weights:
					self.best_weights = copy.deepcopy(model.state_dict())
			return False
		
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
		
		# Check primary patience criterion
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

def get_dataset(dataset_dir:str="/path/to/dataset"):
	train_dataset = pd.read_csv(os.path.join(dataset_dir, f"train_metadata.csv"))
	val_dataset = pd.read_csv(os.path.join(dataset_dir, f"val_metadata.csv"))
	return train_dataset, val_dataset

def main():
	set_seeds()
	print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
	model, preprocess = load_model(
		model_name=args.model_name,
		device=args.device,
		jit=False,
	)
	train_dataset, validation_dataset = get_dataset(dataset_dir=args.dataset_dir)
	train_loader, validation_loader = get_dataloaders(
		train_dataset=train_dataset, 
		val_dataset=validation_dataset, 
		preprocess=preprocess,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	print(f"Train Loader: {len(train_loader)} batches, Validation Loader: {len(validation_loader)} batches")
	early_stopping = EarlyStopping(
		patience=5, # 5 epochs without improvement before stopping
		min_delta=1e-4,
		cumulative_delta=5e-3,
		window_size=5, # 
		mode='min', # 'min' for loss, 'max' for accuracy
		min_epochs=3, # Minimum epochs before early stopping can be triggered
		restore_best_weights=True,
	)
	# visualize_(dataloader=train_loader, num_samples=5)
	if args.mode == "finetune":
		finetune(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.num_epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			early_stopping=early_stopping,
			early_stopping_patience=5,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=os.path.basename(args.dataset_dir),
			device=args.device,
			results_dir=os.path.join(args.dataset_dir, "results")
		)
	elif args.mode == "train":
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.num_epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			early_stopping=early_stopping,
			early_stopping_patience=5,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=os.path.basename(args.dataset_dir),
			device=args.device,
			results_dir=os.path.join(args.dataset_dir, "results")
		)
	else:
		raise ValueError("Invalid mode. Choose between 'train' or 'finetune'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)