from utils import *
from dataset_loader import get_dataloaders
from trainer import finetune, train
import copy

parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--num_workers', '-nw', type=int, default=18, help='Number of CPUs [def: max cpus]')
parser.add_argument('--epochs', '-e', type=int, default=7, help='Number of epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
parser.add_argument('--print_every', type=int, default=150, help='Print loss')
parser.add_argument('--model_name', '-md', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune'], default='finetune', help='Choose mode (train/finetune)')
parser.add_argument('--window_size', '-ws', type=int, default=5, help='Windows size for early stopping and progressive freezing')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
parser.add_argument('--minimum_epochs', type=int, default=20, help='Early stopping minimum epochs')

args, unknown = parser.parse_known_args()
args.device = torch.device(args.device)
print(args)

# run in pouta:
# train from scratch:
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -ne 32 -lr 1e-5 -wd 1e-3 --print_every 100 -nw 40 --device "cuda:2" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_train.out &

# finetune:
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -ne 128  -lr 1e-4 -wd 1e-2 --print_every 150 -nw 40 --device "cuda:2" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_ft_vitb32.out &

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
	# visualize_(dataloader=train_loader, num_samples=5)
	if args.mode == "finetune":
		finetune(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=os.path.basename(args.dataset_dir),
			device=args.device,
			results_dir=os.path.join(args.dataset_dir, "results"),
			window_size=args.window_size, 						# early stopping & progressive unfreezing
			patience=args.patience, 									# early stopping
			min_delta=args.minimum_delta, 						# early stopping & progressive unfreezing
			cumulative_delta=args.cumulative_delta, 	# early stopping
			minimum_epochs=args.minimum_epochs, 			# early stopping
		)
	elif args.mode == "train":
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			dataset_name=os.path.basename(args.dataset_dir),
			device=args.device,
			results_dir=os.path.join(args.dataset_dir, "results"),
			window_size=args.window_size, 						# early stopping
			patience=args.patience, 									# early stopping
			min_delta=args.minimum_delta, 						# early stopping
			cumulative_delta=args.cumulative_delta, 	# early stopping
			minimum_epochs=args.minimum_epochs, 			# early stopping
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