from utils import *
from dataset_loader import get_dataloaders
from trainer import finetune, train

parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs [def: max cpus]')
parser.add_argument('--epochs', '-e', type=int, default=7, help='Number of epochs')
parser.add_argument('--batch_size', '-bs', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3, help='Weight decay [def: 5e-4]')
parser.add_argument('--print_every', type=int, default=250, help='Print loss')
parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune'], default='finetune', help='Choose mode (train/finetune)')
parser.add_argument('--window_size', '-ws', type=int, default=5, help='Windows size for early stopping and progressive freezing')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
parser.add_argument('--minimum_epochs', type=int, default=20, help='Early stopping minimum epochs')
parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the model')
parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	
args, unknown = parser.parse_known_args()
args.device = torch.device(args.device)
print(args)

# run in local:
# $ nohup python -u history_clip_trainer.py -ddir /home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 128 -e 32 -lr 1e-5 -wd 1e-3 --print_every 200 -nw 12 -m train -a "ViT-B/32" > logs/europeana_train.out &

# run in pouta:
# train from scratch:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -e 32 -lr 1e-5 -wd 1e-3 --print_every 200 -nw 40 --device "cuda:2" -m train -a "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_train.out &

# finetune:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORICAL_ARCHIVES -bs 256 -e 32  -lr 1e-4 -wd 1e-3 --print_every 200 -nw 40 --device "cuda:3" -m finetune -a "ViT-B/32" > /media/volume/ImACCESS/trash/historyCLIP_ft.out &

@measure_execution_time
def main():
	set_seeds()
	print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]

	model, preprocess = clip.load(
		name=args.model_architecture,
		device=args.device, 
		jit=False, # training or finetuning => jit=False
		random_weights=True if args.mode == 'train' else False, 
		dropout=args.dropout,
	)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	print(f"Model: {model.__class__.__name__} loaded with {model.name} architecture on {args.device} device")
	
	train_loader, validation_loader = get_dataloaders(
		dataset_dir=args.dataset_dir,
		sampling=args.sampling,
		preprocess=preprocess,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)
	print(f"Train Loader: {len(train_loader)} batches, Number of unique classes: {train_loader.dataset.num_classes}")
	print(f"Validation Loader: {len(validation_loader)} batches, Number of unique classes: {validation_loader.dataset.num_classes}")
	# for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
	# 	print(images.shape, tokenized_labels.shape, labels_indices.shape, labels_indices)
	# 	break

	# for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
	# 	print(images.shape, tokenized_labels.shape, labels_indices.shape, labels_indices)
	# 	break

	# visualize_(dataloader=train_loader, num_samples=5)
	# return
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
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			device=args.device,
			results_dir=os.path.join(args.dataset_dir, "results"),
			window_size=args.window_size, 						# early stopping
			patience=args.patience, 									# early stopping
			min_delta=args.minimum_delta, 						# early stopping
			cumulative_delta=args.cumulative_delta, 	# early stopping
			minimum_epochs=args.minimum_epochs, 			# early stopping
			TOP_K_VALUES=args.topK_values,
		)
	else:
		raise ValueError("Invalid mode. Choose between 'train' or 'finetune'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))