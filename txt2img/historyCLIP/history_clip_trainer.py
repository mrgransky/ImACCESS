import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from historical_dataset_loader import get_dataloaders
from trainer import train, pretrain, full_finetune, lora_finetune, progressive_unfreeze_finetune, evaluate_best_model
from visualize import visualize_samples, visualize_, plot_all_pretrain_metrics, plot_comparison_metrics

# $ python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=10))))"

# run in local:
# $ nohup python -u history_clip_trainer.py -ddir /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 100 -lr 1e-5 -wd 1e-1 --print_every 200 -nw 12 -m finetune -fts progressive -a "ViT-B/32" > logs/europeana_ft_progressive.txt &

# Pouta:
# pretrain:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 --device "cuda:0" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 --device "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 --device "cuda:2" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 --device "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 16 --device "cuda:0" --log_dir /media/volume/ImACCESS/trash &

# finetune [full]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 16 -e 100 -lr 1e-5 -wd 1e-2 --print_every 50 -nw 50 --device "cuda:3" -m finetune -a "ViT-L/14" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:2" -m finetune -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 16 -e 100 -lr 5e-5 -wd 1e-1 --print_every 200 -nw 50 --device "cuda:2" -m finetune -a "ViT-L/14" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 100 -lr 5e-5 -wd 1e-1 --print_every 100 -nw 50 --device "cuda:0" -m finetune -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 64 -e 100 -lr 1e-5 -wd 1e-1 --print_every 250 -nw 50 --device "cuda:3" -m finetune -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &

# finetune [lora]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:1" -m finetune -fts lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 150 -lr 1e-4 -wd 1e-1 --print_every 100 -nw 50 --device "cuda:3" -m finetune -fts lora --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-1 --print_every 200 -nw 50 --device "cuda:0" -m finetune -fts lora --lora_rank 8 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:1" -m finetune  -fts lora --lora_rank 4 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 250 -nw 50 --device "cuda:1" -m finetune -fts lora --lora_rank 4 --lora_alpha 32 --lora_dropout 0.0 -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &

# finetune [progressive unfreezing]:
# using for loop:
# $ for lr in $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=6))))"); do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 50 -nw 50 --device 'cuda:3' -m finetune -fts progressive -a 'ViT-B/32' -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_progressive_lr_${lr}.txt & done

# $ for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4; do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_progressive_lr_${lr}.txt & done
# $ for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4; do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 100 -nw 50 --device "cuda:1" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/wwii_ft_progressive_lr_${lr}.txt & done
# $ for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4; do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 50 -nw 50 --device "cuda:1" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/europeana_ft_progressive_lr_${lr}.txt & done
# $ for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4; do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 100 -lr $lr -wd 1e-1 --print_every 100 -nw 50 --device "cuda:2" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/na_ft_progressive_lr_${lr}.txt & done
# $ for lr in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4; do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 64 -e 100 -lr $lr -wd 1e-1 --print_every 250 -nw 50 --device "cuda:3" -m finetune -fts progressive -a "ViT-B/32" -do 0.0 > /media/volume/ImACCESS/trash/history_xN_ft_progressive_lr_${lr}.txt & done

# using one command:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -e 150 -bs 64 -lr 1e-5 -wd 1e-2 --print_every 10 -nw 10 --device "cuda:1" -m finetune -fts progressive -a "ViT-L/14" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 5e-6 -wd 1e-2 --print_every 50 -nw 50 --device "cuda:2" -m finetune -fts progressive -a "ViT-B/16" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 150 -lr 5e-6 -wd 1e-2 --print_every 100 -nw 10 --device "cuda:2" -m finetune -fts progressive -a "ViT-L/14" -do 0.1 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 100 -lr 1e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:1" -m finetune  -fts progressive -a "ViT-B/16" -do 0.1 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 32 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 24 --device "cuda:3" -m finetune -fts progressive -a "ViT-B/16" -do 0.2 --log_dir /media/volume/ImACCESS/trash &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=16, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--epochs', '-e', type=int, default=9, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=100, help='Print loss')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune', 'pretrain', 'compare'], default='pretrain', help='Choose mode (train/finetune/pretrain/compare)')
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
	parser.add_argument('--log_dir', type=str, default=None, help='Directory to store log files (if not specified, logs will go to stdout)')
	parser.add_argument('--checkpoint_path', '-cp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)

	# Original stdout/stderr
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	log_file = None

	try:
		if args.log_dir:
			os.makedirs(args.log_dir, exist_ok=True)
			dataset_name = os.path.basename(args.dataset_dir)
			arch_name = args.model_architecture.replace('/', '').replace('@', '_')

			log_file_base_name = f"{dataset_name}_{args.mode}_{args.finetune_strategy}_{arch_name}_bs_{args.batch_size}_ep_{args.epochs}_lr_{args.learning_rate}_wd_{args.weight_decay}_do_{args.dropout}_logs" 

			if args.finetune_strategy == "pretrain":
				log_file_base_name = f"{dataset_name}_{args.mode}_{arch_name}_logs"

			if args.finetune_strategy == "lora":
				log_file_base_name += f"_lora_rank_{args.lora_rank}_lora_alpha_{args.lora_alpha}_lora_dropout_{args.lora_dropout}"
			
			log_file_path = os.path.join(args.log_dir, f"{log_file_base_name}.txt")

			log_file = open(log_file_path, 'w')
			sys.stdout = log_file
			sys.stderr = log_file
			print(f"Log file opened: {log_file.name} | {log_file_path}")

		print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
		print_args_table(args=args, parser=parser)
		set_seeds(seed=42)
		# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
		RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results")
		os.makedirs(RESULT_DIRECTORY, exist_ok=True)

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
		
		# Adaptive window size
		window_size = get_adaptive_window_size(
			loader=train_loader,
			min_window=5,
			max_window=20,
		)

		if args.mode == "finetune":
			if args.finetune_strategy == "full":
				full_finetune(
					model=model,
					train_loader=train_loader,
					validation_loader=validation_loader,
					num_epochs=args.epochs,
					print_every=args.print_every,
					learning_rate=args.learning_rate,
					weight_decay=args.weight_decay,
					device=args.device,
					results_dir=RESULT_DIRECTORY,
					window_size=window_size, 									# early stopping & progressive unfreezing
					patience=args.patience, 									# early stopping
					min_delta=args.minimum_delta, 						# early stopping & progressive unfreezing
					cumulative_delta=args.cumulative_delta, 	# early stopping
					minimum_epochs=args.minimum_epochs, 			# early stopping
					topk_values=args.topK_values,
				)
			elif args.finetune_strategy == "lora":
				lora_finetune(
					model=model,
					train_loader=train_loader,
					validation_loader=validation_loader,
					num_epochs=args.epochs,
					print_every=args.print_every,
					learning_rate=args.learning_rate,
					weight_decay=args.weight_decay,
					device=args.device,
					results_dir=RESULT_DIRECTORY,
					window_size=window_size, 									# early stopping & progressive unfreezing
					lora_rank=args.lora_rank,
					lora_alpha=args.lora_alpha,
					lora_dropout=args.lora_dropout,
					patience=args.patience, 									# early stopping	& progressive unfreezing
					min_delta=args.minimum_delta, 						# early stopping & progressive unfreezing
					cumulative_delta=args.cumulative_delta, 	# early stopping
					minimum_epochs=args.minimum_epochs, 			# early stopping
					topk_values=args.topK_values,
				)
			elif args.finetune_strategy == 'progressive':
				progressive_unfreeze_finetune(
					model=model,
					train_loader=train_loader,
					validation_loader=validation_loader,
					num_epochs=args.epochs,
					print_every=args.print_every,
					initial_learning_rate=args.learning_rate,
					initial_weight_decay=args.weight_decay,
					device=args.device,
					results_dir=RESULT_DIRECTORY,
					window_size=window_size, 									# early stopping & progressive unfreezing
					patience=args.patience, 									# early stopping
					min_delta=args.minimum_delta, 						# early stopping
					cumulative_delta=args.cumulative_delta, 	# early stopping
					minimum_epochs=args.minimum_epochs, 			# early stopping
					topk_values=args.topK_values,
				)
			else:
				raise ValueError(f"Invalid mode: {args.mode}! Choose between 'full', 'lora', 'progressive'!")
		elif args.mode == "train":
			train(
				model=model,
				train_loader=train_loader,
				validation_loader=validation_loader,
				num_epochs=args.epochs,
				print_every=args.print_every,
				learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				device=args.device,
				results_dir=RESULT_DIRECTORY,
				window_size=window_size, 									# early stopping & progressive unfreezing
				patience=args.patience, 									# early stopping
				min_delta=args.minimum_delta, 						# early stopping
				cumulative_delta=args.cumulative_delta, 	# early stopping
				minimum_epochs=args.minimum_epochs, 			# early stopping
				topk_values=args.topK_values,
			)
		elif args.mode == "pretrain":
			all_img2txt_metrics = {}
			all_txt2img_metrics = {}
			all_vit_encoders = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
			all_available_clip_encoders = clip.available_models() # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
			for model_arch in all_vit_encoders:
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
						results_dir=RESULT_DIRECTORY,
						device=args.device,
						topk_values=args.topK_values,
				)
				all_img2txt_metrics[model_arch] = img2txt_metrics
				all_txt2img_metrics[model_arch] = txt2img_metrics
				del model  # Clean up memory
				torch.cuda.empty_cache()
			plot_all_pretrain_metrics(
				dataset_name=validation_loader.name,
				img2txt_metrics_dict=all_img2txt_metrics,
				txt2img_metrics_dict=all_txt2img_metrics,
				topK_values=args.topK_values,
				results_dir=RESULT_DIRECTORY,
			)
		elif args.mode == "compare":
			if args.checkpoint_path is None:
				raise ValueError("Please provide a checkpoint path for comparison!")

			if not os.path.exists(args.checkpoint_path):
				raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist!")

			if args.finetune_strategy not in args.checkpoint_path:
				raise ValueError(f"Checkpoint path {args.checkpoint_path} does not match the assigned finetune strategy: « {args.finetune_strategy} »!")

			if re.sub(r'[/@]', '-', args.model_architecture) not in args.checkpoint_path:
				raise ValueError(f"Checkpoint path {args.checkpoint_path} does not match the assigned model architecture: « {args.model_architecture} »!")

			# Step 1: Compute pretrained model metrics
			print(f">> Computing metrics for pretrained {args.model_architecture}...")
			pretrained_img2txt, pretrained_txt2img = pretrain(
				model=model,
				validation_loader=validation_loader,
				results_dir=RESULT_DIRECTORY,
				device=args.device,
				topk_values=args.topK_values,
			)
			pretrained_img2txt_dict = {args.model_architecture: pretrained_img2txt}
			pretrained_txt2img_dict = {args.model_architecture: pretrained_txt2img}
			print(f">> Pretrained model metrics computed successfully.")

			# Step 2: Load and evaluate finetuned model
			print(f">> Loading finetuned model from {args.checkpoint_path}...")
						
			try:
				# Load the model
				checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
				if 'model_state_dict' in checkpoint:
					model.load_state_dict(checkpoint['model_state_dict'])
				else:
					model.load_state_dict(checkpoint)
				
				# Evaluate finetuned model
				criterion = torch.nn.CrossEntropyLoss()
				evaluation_results = evaluate_best_model(
					model=model,
					validation_loader=validation_loader,
					criterion=criterion,
					early_stopping=None,
					checkpoint_path=args.checkpoint_path,
					device=args.device,
					topk_values=args.topK_values,
					verbose=True
				)
				
				finetuned_img2txt_dict = {args.model_architecture: evaluation_results["img2txt_metrics"]}
				finetuned_txt2img_dict = {args.model_architecture: evaluation_results["txt2img_metrics"]}

				plot_comparison_metrics(
					dataset_name=validation_loader.name,
					pretrained_img2txt_dict=pretrained_img2txt_dict,
					pretrained_txt2img_dict=pretrained_txt2img_dict,
					finetuned_img2txt_dict=finetuned_img2txt_dict,
					finetuned_txt2img_dict=finetuned_txt2img_dict,
					model_name=args.model_architecture,
					finetune_strategy=args.finetune_strategy,
					topK_values=args.topK_values,
					results_dir=RESULT_DIRECTORY,
				)
					
			except Exception as e:
				print(f"Error loading or evaluating finetuned model: {e}")
				traceback.print_exc()

		else:
			raise ValueError(f"Invalid mode: {args.mode}. Choose between: 'pretrain', 'train', 'finetune', 'compare'!")

		print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
	finally:
		if log_file:
			log_file.flush()
			sys.stdout = original_stdout
			sys.stderr = original_stderr
			log_file.close()

if __name__ == "__main__":
	main()