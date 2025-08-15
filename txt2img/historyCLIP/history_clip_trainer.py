import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
from utils import *
from trainer import (
	train, pretrain, 
	full_finetune_single_label, 
	lora_finetune_single_label, 
	progressive_finetune_single_label, 
	full_finetune_multi_label,
	lora_finetune_multi_label,
	progressive_finetune_multi_label,
)
from visualize import visualize_samples, visualize_, plot_all_pretrain_metrics, plot_comparison_metrics_merged, plot_comparison_metrics_split
from historical_dataset_loader import get_single_label_dataloaders, get_multi_label_dataloaders


# $ python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=10))))"

# run in local:
# $ python history_clip_trainer.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -e 3 -m finetune -fts progressive -dt multi_label
# $ nohup python -u history_clip_trainer.py -ddir /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 100 -lr 1e-5 -wd 1e-1 --print_every 200 -nw 12 -m finetune -fts progressive -a "ViT-B/32" > logs/europeana_ft_progressive.txt &

# Pouta:
# pretrain:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -dv "cuda:0" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -dv "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -dv "cuda:2" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -dv "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 16 -dv "cuda:0" --log_dir /media/volume/ImACCESS/trash &

# finetune [full]:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 200 -lr 1e-5 -wd 1e-2 --print_every 10 -nw 32 -dv "cuda:0" -m finetune -fts full -dt single_label -a "ViT-B/32" -do 0.1 -mep 10 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-6 -wd 1e-2 --print_every 200 -nw 12 -dv "cuda:0" -m finetune -fts full -dt multi_label -a "ViT-B/32" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 12 -dv "cuda:2" -m finetune -fts full -dt multi_label -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 100 -lr 5e-6 -wd 1e-2 --print_every 100 -nw 32 -dv "cuda:3" -m finetune -fts full -dt multi_label  -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 256 -e 150 -lr 5e-4 -wd 1e-2 --print_every 750 -nw 50 -dv "cuda:1" -m finetune -fts full -dt single_label -a "ViT-B/32" -do 0.1 --log_dir /media/volume/ImACCESS/trash &

# finetune [lora]: alpha = 2x rank
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 200 -lr 1e-5 -wd 1e-2 --print_every 10 -nw 32 -dv "cuda:1" -m finetune -fts lora -lor 32 -loa 64.0 -lod 0.1 -dt single_label -a "ViT-B/32" -mep 10 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 200 -nw 50 -dv "cuda:1" -m finetune -fts lora -lor 4 -loa 32.0 -lod 0.05 -dt multi_label -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 150 -lr 1e-4 -wd 1e-1 --print_every 100 -nw 12 -dv "cuda:0" -m finetune -fts lora -lor 8 -loa 32.0 -lod 0.05 -dt multi_label -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 -dv "cuda:1" -m finetune  -fts lora -lor 4 -loa 32.0 -lod 0.0 -dt multi_label -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 64 -e 150 -lr 5e-6 -wd 1e-2 --print_every 750 -nw 40 -dv "cuda:2" -m finetune -fts lora -lor 64 -loa 128.0 -lod 0.05 -dt multi_label -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &

# finetune [progressive unfreezing]:
# using for loop:
# $ for lr in $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=6))))"); do nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 50 -nw 50 -dv 'cuda:3' -m finetune -fts progressive -a 'ViT-B/32' -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_progressive_lr_${lr}.txt & done

# using one command:
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -e 200 -bs 64 -lr 1e-5 -wd 1e-2 --print_every 12 -nw 32 -dv "cuda:2" -m finetune -fts progressive -dt multi_label -a "ViT-B/32" -do 0.1 -mep 6 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 50 -nw 50 -dv "cuda:2" -m finetune -fts progressive -dt multi_label -a "ViT-B/32" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 32 -e 150 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 12 -dv "cuda:1" -m finetune -fts progressive -dt multi_label -a "ViT-B/32" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 32 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 -dv "cuda:0" -m finetune -fts progressive -dt multi_label -a "ViT-L/14" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u history_clip_trainer.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 256 -e 100 -mep 11 -pat 5 -lr 5e-5 -wd 1e-2 -nw 32 -dv "cuda:3" -m finetune -fts progressive -dt single_label -a "ViT-L/14@336px" -do 0.1 --log_dir /media/volume/ImACCESS/trash &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--dataset_type', '-dt', type=str, choices=['single_label', 'multi_label'], default='single_label', help='Dataset type (single_label/multi_label)')
	parser.add_argument('--device', '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--epochs', '-e', type=int, default=3, help='Number of epochs')
	parser.add_argument('--num_workers', '-nw', type=int, default=4, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch size for training')
	parser.add_argument('--cache_size', '-cs', type=int, default=None, help='Cache size for dataloader (in number of samples)')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-3]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 5e-4]')
	parser.add_argument('--print_every', type=int, default=500, help='Print loss')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--mode', '-m', type=str, choices=['train', 'finetune', 'pretrain'], required=True, help='Choose mode (train/finetune/pretrain)')
	parser.add_argument('--finetune_strategy', '-fts', type=str, choices=['full', 'lora', 'progressive'], default=None, help='Fine-tuning strategy (full/lora/progressive) when mode is finetune')
	parser.add_argument('--lora_rank', '-lor', type=int, default=None, help='LoRA rank (used if finetune_strategy=lora)')
	parser.add_argument('--lora_alpha', '-loa', type=float, default=None, help='LoRA alpha (used if finetune_strategy=lora)')
	parser.add_argument('--lora_dropout', '-lod', type=float, default=None, help='LoRA dropout (used if finetune_strategy=lora)')
	parser.add_argument('--minimum_epochs', '-mep', type=int, default=9, help='Early stopping minimum epochs')
	parser.add_argument('--patience', '-pat', type=int, default=7, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the model')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	parser.add_argument('--log_dir', type=str, default=None, help='Directory to store log files (if not specified, logs will go to stdout)')
	parser.add_argument('--use_lamb', '-lamb', action='store_true', help='Use LAMB optimizer instead of AdamW')
	# parser.add_argument('--checkpoint_path', '-cp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	args.dataset_dir = os.path.normpath(args.dataset_dir)
	# Original stdout/stderr
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	log_file = None

	try:
		if args.log_dir:
			os.makedirs(args.log_dir, exist_ok=True)
			dataset_name = os.path.basename(args.dataset_dir)
			arch_name = args.model_architecture.replace('/', '').replace('@', '_')
			log_file_base_name = (
				f"{dataset_name}_{args.dataset_type}_"
				f"{args.mode}_"
				f"{args.finetune_strategy}_"
				f"{arch_name}_"
				f"bs_{args.batch_size}_"
				f"nw_{args.num_workers}_"
				f"ep_{args.epochs}_"
				f"mep_{args.minimum_epochs}_"
				f"pat_{args.patience}_"
				f"mdelta_{args.minimum_delta:.1e}_"
				f"cdelta_{args.cumulative_delta:.1e}_"
				f"lr_{args.learning_rate:.1e}_"
				f"wd_{args.weight_decay:.1e}_"
				f"do_{args.dropout}"
			)

			if args.finetune_strategy == "pretrain":
				log_file_base_name = f"_{dataset_name}_{args.dataset_type}_{args.mode}_{arch_name}"

			if args.finetune_strategy == "lora":
				log_file_base_name += f"_lor_{args.lora_rank}_loa_{args.lora_alpha}_lod_{args.lora_dropout}"
			
			if args.use_lamb:
				log_file_base_name += "_lamb"

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
		RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results_{args.dataset_type}")
		os.makedirs(RESULT_DIRECTORY, exist_ok=True)

		print(f">> CLIP Model Architecture: {args.model_architecture}...")
		model_config = get_config(architecture=args.model_architecture, dropout=args.dropout,)
		print(json.dumps(model_config, indent=4, ensure_ascii=False))

		model, _ = clip.load(
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
		dataset_functions = {
			'single_label': get_single_label_dataloaders,
			'multi_label': get_multi_label_dataloaders
		}
		train_loader, validation_loader = dataset_functions[args.dataset_type](
			dataset_dir=args.dataset_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			input_resolution=model_config["image_resolution"],
			# cache_size=args.cache_size,
		)
		print_loader_info(loader=train_loader, batch_size=args.batch_size)
		print_loader_info(loader=validation_loader, batch_size=args.batch_size)
		# visualize_(dataloader=validation_loader, batches=4, num_samples=7)
		# visualize_samples(validation_loader, validation_loader.dataset, num_samples=5)
		
		# Adaptive window size
		window_size = get_adaptive_window_size(
			loader=train_loader,
			min_window=5,
			max_window=min(args.minimum_epochs, 20),
		)
		finetune_functions = {
			'single_label': {
				'full': full_finetune_single_label,
				'lora': lora_finetune_single_label,
				'progressive': progressive_finetune_single_label,
			},
			'multi_label': {
				'full': full_finetune_multi_label,
				'lora': lora_finetune_multi_label,
				'progressive': progressive_finetune_multi_label,
			}
		}
		if args.mode == "finetune":
			finetune_functions[args.dataset_type][args.finetune_strategy](
				model=model,
				train_loader=train_loader,
				validation_loader=validation_loader,
				num_epochs=args.epochs,
				print_every=args.print_every,
				learning_rate=args.learning_rate,
				weight_decay=args.weight_decay,
				device=args.device,
				results_dir=RESULT_DIRECTORY,
				window_size=window_size,
				patience=args.patience,
				min_delta=args.minimum_delta,
				cumulative_delta=args.cumulative_delta,
				minimum_epochs=args.minimum_epochs,
				topk_values=args.topK_values,
				use_lamb=args.use_lamb,
				**(
					{
						'lora_rank': args.lora_rank,
						'lora_alpha': args.lora_alpha,
						'lora_dropout': args.lora_dropout
					} if args.finetune_strategy == 'lora' else {})
				)
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
		else:
			raise ValueError(f"Invalid mode: {args.mode}. Choose between: 'pretrain', 'train', 'finetune', 'quantitative'!")

		print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
	finally:
		if log_file:
			log_file.flush()
			sys.stdout = original_stdout
			sys.stderr = original_stderr
			log_file.close()

if __name__ == "__main__":
	cleanup_old_temp_dirs()
	main()