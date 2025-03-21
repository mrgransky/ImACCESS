import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from dataset_loader import get_dataloaders
from trainer import train, pretrain, full_finetune, lora_finetune, progressive_unfreeze_finetune
from visualize import visualize_samples, visualize_, plot_all_pretrain_metrics

# run in local:
# $ nohup python -u history_clip_trainer.py -ddir /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 32 -e 100 -lr 1e-5 -wd 1e-1 --print_every 200 -nw 12 -m finetune -fts progressive -a "ViT-B/32" > logs/europeana_ft_progressive.out &

# run in pouta:

# pretrain:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 > /media/volume/ImACCESS/trash/europeana_pretrained.out &

# train from scratch:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 32 -e 100 -lr 1e-4 -wd 1e-1 --print_every 200 -nw 50 --device "cuda:0" -m train -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/smu_train.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 256 -e 100 -lr 1e-4 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:1" -m train -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/wwii_train.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 256 -e 100 -lr 1e-5 -wd 1e-3 --print_every 200 -nw 50 --device "cuda:2" -m train -a "ViT-B/32" -do 0.1 > /media/volume/ImACCESS/trash/europeana_train.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 256 -e 100 -lr 1e-5 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:0" -m train -a "ViT-B/32" -do 0.1 > /media/volume/ImACCESS/trash/na_train.out &

# finetune [full]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 100 -lr 5e-6 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_full.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 256 -e 100 -lr 5e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:2" -m finetune -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/wwii_ft_full.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 256 -e 100 -lr 5e-5 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:2" -m finetune -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/europeana_ft_full.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 256 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:0" -m finetune -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/na_ft_full.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 128 -e 100 -lr 1e-5 -wd 1e-2 --print_every 250 -nw 50 --device "cuda:3" -m finetune -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/history_xN_ft_full.out &

# finetune [lora]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 100 -lr 5e-6 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -fts lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_lora.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 256 -e 100 -lr 5e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m finetune -fts lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/wwii_ft_lora.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 256 -e 100 -lr 5e-5 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:2" -m finetune -fts lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/europeana_ft_lora.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 256 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:1" -m finetune  -fts lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/na_ft_lora.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 128 -e 100 -lr 1e-5 -wd 1e-2 --print_every 250 -nw 50 --device "cuda:1" -m finetune -fts lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/history_xN_ft_lora.out &

# finetune [progressive unfreezing]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 32 -e 150 -lr 1e-4 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_progressive.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 512 -e 150 -lr 5e-5 -wd 1e-1 --print_every 100 -nw 50 --device "cuda:2" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/wwii_ft_progressive.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 512 -e 150 -lr 5e-5 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:3" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/europeana_ft_progressive.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 256 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:1" -m finetune  -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/na_ft_progressive.out &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 128 -e 100 -lr 1e-5 -wd 1e-2 --print_every 250 -nw 50 --device "cuda:3" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/history_xN_ft_progressive.out &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=16, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--epochs', '-e', type=int, default=9, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-6, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=100, help='Print loss')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune', 'pretrain'], default='pretrain', help='Choose mode (train/finetune)')
	parser.add_argument('--finetune_strategy', '-fts', type=str, choices=['full', 'lora', 'progressive'], default='full', help='Fine-tuning strategy (full/lora/progressive) when mode is finetune')
	parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (used if finetune_strategy=lora)')
	parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha (used if finetune_strategy=lora)')
	parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout (used if finetune_strategy=lora)')
	parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--minimum_epochs', type=int, default=20, help='Early stopping minimum epochs')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the model')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	set_seeds(seed=42)

	print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]

	print(f">> CLIP Model Architecture: {args.model_architecture}...")
	model_config = get_config(architecture=args.model_architecture, dropout=args.dropout,)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))

	model, preprocess = clip.load(
		name=args.model_architecture,
		device=args.device, 
		jit=False, # training or finetuning => jit=False
		random_weights=True if args.mode == 'train' else False, 
		dropout=args.dropout,
		download_root=get_model_directory(path=args.dataset_dir),
	)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	model_name = model.__class__.__name__
	print(f"Loaded {model_name} {model.name} in {args.device}")
	
	train_loader, validation_loader = get_dataloaders(
		dataset_dir=args.dataset_dir,
		sampling=args.sampling,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=model_config["image_resolution"],
		preprocess=None, # preprocess,
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)
	# visualize_(dataloader=validation_loader, batches=4, num_samples=7)
	# visualize_samples(validation_loader, validation_loader.dataset, num_samples=5)
	
	if args.mode == "finetune":
		if args.finetune_strategy == "full":
			full_finetune(
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
				patience=args.patience, 									# early stopping
				min_delta=args.minimum_delta, 						# early stopping & progressive unfreezing
				cumulative_delta=args.cumulative_delta, 	# early stopping
				minimum_epochs=args.minimum_epochs, 			# early stopping
				TOP_K_VALUES=args.topK_values,
			)
		elif args.finetune_strategy == "lora":
			lora_finetune(
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
				lora_rank=args.lora_rank,
				lora_alpha=args.lora_alpha,
				lora_dropout=args.lora_dropout,
				patience=args.patience, 									# early stopping	& progressive unfreezing
				min_delta=args.minimum_delta, 						# early stopping & progressive unfreezing
				cumulative_delta=args.cumulative_delta, 	# early stopping
				minimum_epochs=args.minimum_epochs, 			# early stopping
				TOP_K_VALUES=args.topK_values,
			)
		elif args.finetune_strategy == 'progressive':
			progressive_unfreeze_finetune(
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
				patience=10,									# early stopping
				min_delta=1e-4,								# early stopping
				cumulative_delta=5e-3,				# early stopping
				minimum_epochs=20,						# early stopping
				top_k_values=args.topK_values,
			)
		else:
			raise ValueError(f"Invalid mode: {args.mode}")
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
			patience=args.patience, 									# early stopping
			min_delta=args.minimum_delta, 						# early stopping
			cumulative_delta=args.cumulative_delta, 	# early stopping
			minimum_epochs=args.minimum_epochs, 			# early stopping
			TOP_K_VALUES=args.topK_values,
		)
	elif args.mode == "pretrain":
		all_img2txt_metrics = {}
		all_txt2img_metrics = {}
		available_models = clip.available_models()[::-1]#[:4] # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		for model_arch in available_models:
			print(f"Evaluating pre-trained model: {model_arch}")
			model_config = get_config(architecture=model_arch, dropout=args.dropout,)
			print(json.dumps(model_config, indent=4, ensure_ascii=False))

			model, preprocess = clip.load(
				name=model_arch,
				device=args.device,
				random_weights=False,
				download_root=get_model_directory(path=args.dataset_dir),
				dropout=args.dropout,
			)
			model = model.float()
			model.name = model_arch  # Custom attribute to store model name
			print(f"Model: {model.__class__.__name__} loaded with {model.name} architecture on {args.device} device")
			train_loader, validation_loader = get_dataloaders(
				dataset_dir=args.dataset_dir,
				sampling=args.sampling,
				batch_size=args.batch_size,
				num_workers=args.num_workers,
				input_resolution=model_config["image_resolution"],
				preprocess=None, # preprocess,
			)
			print_loader_info(loader=train_loader, batch_size=args.batch_size)
			print_loader_info(loader=validation_loader, batch_size=args.batch_size)

			img2txt_metrics, txt2img_metrics = pretrain(
					model=model,
					validation_loader=validation_loader,
					results_dir=os.path.join(args.dataset_dir, "results"),
					device=args.device,
					TOP_K_VALUES=args.topK_values,
			)
			all_img2txt_metrics[model_arch] = img2txt_metrics
			all_txt2img_metrics[model_arch] = txt2img_metrics
			del model  # Clean up memory
			torch.cuda.empty_cache()
		# Pass all metrics to the new visualization function
		plot_all_pretrain_metrics(
			dataset_name=os.path.basename(args.dataset_dir),
			img2txt_metrics_dict=all_img2txt_metrics,
			txt2img_metrics_dict=all_txt2img_metrics,
			topK_values=args.topK_values,
			fname=os.path.join(args.dataset_dir, "results", f"{os.path.basename(args.dataset_dir)}_x{len(available_models)}_pretrained_clip_retrieval_metrics.png"),
		)
	else:
		raise ValueError("Invalid mode. Choose between 'pretrain', 'train', 'finetune'!")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))