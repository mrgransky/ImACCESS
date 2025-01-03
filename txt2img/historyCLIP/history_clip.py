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
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 260 -ne 32 -lr 1e-5 -wd 1e-3 --print_every 100 -nw 40 --device "cuda:2" -m "train" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_train.out &

# finetune:
# $ nohup python -u history_clip.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 262 -ne 128  -lr 5e-4 -wd 1e-3 --print_every 150 -nw 40 --device "cuda:2" -md "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_ft_vitb32.out &

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
			num_epochs=args.num_epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			model_name=args.model_name,
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