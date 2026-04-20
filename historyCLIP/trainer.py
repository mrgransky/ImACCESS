import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from single_label_trainer import *
from multi_label_trainer import *

from historyXN_dataset_loader import get_single_label_dataloaders, get_multi_label_dataloaders

# $ python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=10))))"

# run in local:
# single-label:
# $ python trainer.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_single_label.csv -fts progressive -mphbs 3 -mepph 5 -bs 128 -lr 3e-4 -wd 1e-2 -tnp 8
# $ nohup python -u trainer.py -csv /home/farid/datasets/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_single_label.csv -bs 64 -e 100 -lr 1e-5 -wd 1e-1 --print_every 200 -nw 12 -fts progressive -a "ViT-B/32" -mphbs 3 -mepph 5 -tnp 8 > logs/europeana_ft_progressive.txt &

# multi-label:
# $ python trainer.py -csv /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_multimodal.csv -c multimodal_canonical_labels -stg full

# Pouta:
# pretrain:
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_single_label.csv -bs 64 -dv "cuda:0" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_single_label.csv -bs 64 -dv "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_single_label.csv -bs 64 -dv "cuda:2" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata_single_label.csv -bs 64 -dv "cuda:1" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_single_label.csv -bs 16 -dv "cuda:0" --log_dir /media/volume/ImACCESS/trash &

# finetune [full]:
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 200 -lr 5e-6 -wd 1e-2 --print_every 10 -nw 32 -dv "cuda:0" -fts full -dt single_label -a "ViT-B/32" -do 0.0 -mep 7 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-6 -wd 1e-2 --print_every 200 -nw 12 -dv "cuda:0" -fts full -a "ViT-B/32" -do 0.05 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 12 -dv "cuda:2" -fts full -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 100 -lr 5e-6 -wd 1e-2 --print_every 100 -nw 32 -dv "cuda:3" -fts full  -a "ViT-B/32" -do 0.0 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 256 -e 150 -lr 5e-4 -wd 1e-2 --print_every 750 -nw 50 -dv "cuda:1" -fts full -dt single_label -a "ViT-B/32" -do 0.1 --log_dir /media/volume/ImACCESS/trash &

# finetune [lora]: alpha = 2x rank
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 256 -e 200 -lr 1e-6 -wd 1e-2 -nw 32 -dv "cuda:1" -fts lora -lor 32 -loa 64.0 -lod 0.1 -dt single_label -a "ViT-B/32" -mep 7 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 200 -nw 50 -dv "cuda:1" -fts lora -lor 4 -loa 32.0 -lod 0.05 -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 64 -e 150 -lr 1e-4 -wd 1e-1 --print_every 100 -nw 12 -dv "cuda:0" -fts lora -lor 8 -loa 32.0 -lod 0.05 -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 -dv "cuda:1"  -fts lora -lor 4 -loa 32.0 -lod 0.0 -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_multi_label_multimodal.csv -bs 64 -e 150 -lr 5e-6 -wd 1e-2 --print_every 750 -nw 40 -dv "cuda:2" -fts lora -lor 64 -loa 128.0 -lod 0.05 -a "ViT-B/32" --log_dir /media/volume/ImACCESS/trash &

# finetune [progressive unfreezing]:
# using for loop:
# $ for lr in $(python -c "import numpy as np; print(' '.join(map(str, np.logspace(-6, -4, num=6))))"); do nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -bs 64 -e 150 -lr $lr -wd 1e-1 --print_every 50 -nw 50 -dv 'cuda:3' -fts progressive -a 'ViT-B/32' -do 0.0 > /media/volume/ImACCESS/trash/smu_ft_progressive_lr_${lr}.txt & done

# using one command:
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31 -e 110 -bs 256 -lr 3e-4 -wd 1e-2 -nw 32 -dv "cuda:3" -fts progressive -mep 7 -pat 3 -mepph 5 -mphbs 3 -tnp 8 -a "ViT-L/14@336px" --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31 -bs 64 -e 150 -lr 1e-5 -wd 1e-2 --print_every 50 -nw 50 -dv "cuda:2" -fts progressive -a "ViT-B/32" -do 0.05 -mphbs 3 -mepph 5 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -bs 32 -e 150 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 12 -dv "cuda:1" -fts progressive -a "ViT-B/32" -do 0.05 -mphbs 3 -mepph 5 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31 -bs 32 -e 100 -lr 1e-5 -wd 1e-2 --print_every 100 -nw 50 -dv "cuda:0" -fts progressive -a "ViT-L/14" -do 0.05 -mphbs 3 -mepph 5 --log_dir /media/volume/ImACCESS/trash &
# $ nohup python -u trainer.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -bs 128 -e 100 -mep 7 -pat 3 -lr 5e-4 -wd 1e-2 -nw 32 -dv "cuda:3" -fts progressive -mphbs 3 -mepph 5 -tnp 8 --print_every 2500 --log_dir /media/volume/ImACCESS/trash &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")

	# Common
	parser.add_argument('--metadata_csv', '-csv', type=str, required=True, help='Metadata CSV file')
	parser.add_argument('--column', '-c', type=str, choices=['llm_canonical_labels', 'vlm_canonical_labels', 'multimodal_canonical_labels'], required=True, help='Column for loading label')	
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--strategy', '-stg', type=str, choices=['full', 'lora', 'rslora', 'lora_plus', 'dora', 'vera', 'ia3', 'progressive', 'adapter', 'baseline'], default=None, help='Strategy')
	parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5, help='learning rate [def: 5e-5]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 1e-2]')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the model')
	parser.add_argument('--device', '-dv', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=12, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	parser.add_argument('--log_dir', type=str, default=None, help='Directory to store log files (if not specified, logs will go to stdout)')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--print_every', type=int, default=500, help='Print loss')
	parser.add_argument('--temperature', '-t', type=float, default=0.07, help='Temperature [def: 0.07]')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

	# Early stopping
	parser.add_argument('--minimum_epochs', '-mep', type=int, default=7, help='Early stopping minimum epochs')
	parser.add_argument('--patience', '-pat', type=int, default=3, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--volatility_threshold', '-vth', type=float, default=5.0, help='Volatility threshold for early stopping')
	parser.add_argument('--slope_threshold', '-slth', type=float, default=1e-4, help='Slope threshold for early stopping')
	parser.add_argument('--pairwise_imp_threshold', '-pith', type=float, default=1e-4, help='Pairwise improvement threshold for early stopping')

	# LoRA, LoRA+, DoRA, VeRA, RSLoRA
	parser.add_argument('--lora_rank', '-lor', type=int, default=None, help='LoRA rank (used if strategy=lora)')
	parser.add_argument('--lora_alpha', '-loa', type=float, default=None, help='LoRA alpha (used if strategy=lora)')
	parser.add_argument('--lora_dropout', '-lod', type=float, default=None, help='LoRA dropout (used if strategy=lora)')

	# LoRA+
	parser.add_argument('--lora_plus_lambda', '-lmbd', type=float, default=None, help='LoRA+ lambda multiplier (used if strategy=lora_plus)')

	# Progressive
	parser.add_argument('--min_phases_before_stopping', '-mphbs', type=int, default=None, help='Minimum number of phases before stopping (used if strategy=progressive)')
	parser.add_argument('--min_epochs_per_phase', '-mepph', type=int, default=None, help='Minimum number of epochs per phase (used if strategy=progressive)')
	parser.add_argument('--total_num_phases', '-tnp', type=int, default=None, help='Total number of phases (used if strategy=progressive)')

	# Adapter-based FT
	parser.add_argument('--adapter_method', '-am', type=str, choices=['clip_adapter_v', 'clip_adapter_t', 'clip_adapter_vt', 'tip_adapter', 'tip_adapter_f'], default=None, help='Adapter method (used if strategy=adapter)')

	# Baselines
	parser.add_argument('--baseline_method', '-bm', type=str, choices=['zero_shot', 'probe'], default=None, help='Baseline method')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)

	DATASET_DIRECTORY = os.path.dirname(args.metadata_csv)
	dataset_name = os.path.basename(DATASET_DIRECTORY)
	dataset_type = "single_label" if "single_label" in args.metadata_csv else "multi_label"
	# Original stdout/stderr
	original_stdout = sys.stdout
	original_stderr = sys.stderr
	log_file = None

	if not args.strategy:
		raise ValueError("strategy must be specified (example: -stg lora)")

	LORA_FAMILY_STRATEGIES = ('lora', 'rslora', 'dora', 'lora_plus', 'vera')
	
	if args.strategy in LORA_FAMILY_STRATEGIES:
		assert args.lora_rank is not None, "lora_rank must be specified for LoRA-family strategies"
		assert args.lora_alpha is not None, "lora_alpha must be specified for LoRA-family strategies"
		assert args.lora_dropout is not None, "lora_dropout must be specified for LoRA-family strategies"

	if args.strategy == "lora_plus":
		assert args.lora_plus_lambda is not None, "lora_plus_lambda must be specified for lora_plus finetuning (example: -lmbd 32.0)"

	if args.strategy == "adapter":
		assert args.adapter_method is not None, "adapter_method must be specified for adapter-based finetuning (example: -am clip_adapter_v)"

	try:
		if args.log_dir:
			os.makedirs(args.log_dir, exist_ok=True)
			
			arch_name = args.model_architecture.replace('/', '').replace('@', '_')
			log_file_base_name = (
				f"{dataset_name}_{dataset_type}_"
				f"{args.strategy}_"
				f"{arch_name}_"
				f"nw_{args.num_workers}_"
				f"ep_{args.epochs}_"
				f"mep_{args.minimum_epochs}_"
				f"pat_{args.patience}_"
				f"mdelta_{args.minimum_delta:.1e}_"
				f"cdelta_{args.cumulative_delta:.1e}_"
				f"lr_{args.learning_rate:.1e}_"
				f"wd_{args.weight_decay:.1e}_"
				f"temp_{args.temperature}_"
				f"bs_{args.batch_size}_"
				f"do_{args.dropout}"
			)

			if args.strategy in LORA_FAMILY_STRATEGIES:
				log_file_base_name += f"_lor_{args.lora_rank}_loa_{args.lora_alpha}_lod_{args.lora_dropout}"
			
			if args.strategy == "lora_plus":
				log_file_base_name += f"_lmbd_{args.lora_plus_lambda}"			

			log_file_path = os.path.join(args.log_dir, f"{log_file_base_name}.txt")

			log_file = open(log_file_path, 'w')
			sys.stdout = log_file
			sys.stderr = log_file
			print(f"Log file opened: {log_file.name} | {log_file_path}")

		print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
		print_args_table(args=args, parser=parser)
		print(args)
		set_seeds(seed=42)
		# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		# print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
		# RESULT_DIRECTORY = os.path.join(DATASET_DIRECTORY, f"{dataset_type}") # multi_label
		RESULT_DIRECTORY = os.path.join(DATASET_DIRECTORY, f"{args.column}") # multimodal_canonical_labels
		os.makedirs(RESULT_DIRECTORY, exist_ok=True)

		print(f">> CLIP Model Architecture: {args.model_architecture}...")
		model_config = get_config(
			architecture=args.model_architecture, 
			dropout=args.dropout,
		)
		print(json.dumps(model_config, indent=4, ensure_ascii=False))

		model, _ = clip.load(
			name=args.model_architecture,
			device=args.device, 
			jit=False, # training or finetuning => jit=False
			random_weights=False, 
			dropout=args.dropout,
			download_root=get_model_directory(path=DATASET_DIRECTORY),
		)
		model = model.float() # Convert model parameters to FP32
		model.name = args.model_architecture  # Custom attribute to store model name
		model_name = model.__class__.__name__
		print(f"Loaded {model_name} {model.name} in {args.device}")
		dataset_functions = {
			'single_label': get_single_label_dataloaders,
			'multi_label': get_multi_label_dataloaders
		}
		train_loader, validation_loader = dataset_functions[dataset_type](
			metadata_fpth=args.metadata_csv,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			input_resolution=model_config["image_resolution"],
			col=args.column,
		)

		print_loader_info(loader=train_loader)
		print_loader_info(loader=validation_loader)

		# viz.visualize_samples(validation_loader, num_samples=5)
		
		finetune_functions = {
			'single_label': {
				'full': full_finetune_single_label,
				'probe': probe_single_label,
				'lora': lora_finetune_single_label,
				'lora_plus': lora_plus_finetune_single_label,
				'ia3': ia3_finetune_single_label,
				'dora': dora_finetune_single_label,
				'vera': vera_finetune_single_label,
				'progressive': progressive_finetune_single_label,
				'adapter': clip_adapter_finetune_single_label if args.adapter_method and args.adapter_method.startswith('clip_adapter') else tip_adapter_finetune_single_label,
			},
			'multi_label': {
				'full': full_finetune_multi_label,
				'lora': lora_finetune_multi_label,
				'rslora': rslora_finetune_multi_label,
				'lora_plus': lora_plus_finetune_multi_label,
				'ia3': ia3_finetune_multi_label,
				'dora': dora_finetune_multi_label,
				'vera': vera_finetune_multi_label,
				'baseline': probe_multi_label if args.baseline_method and args.baseline_method.startswith('probe') else zero_shot_multi_label,
				'adapter': clip_adapter_finetune_multi_label if args.adapter_method and args.adapter_method.startswith('clip_adapter') else tip_adapter_finetune_multi_label,
			}
		}
		finetune_functions[dataset_type][args.strategy](
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			patience=args.patience,
			min_delta=args.minimum_delta,
			cumulative_delta=args.cumulative_delta,
			minimum_epochs=args.minimum_epochs,
			volatility_threshold=args.volatility_threshold,
			slope_threshold=args.slope_threshold,
			pairwise_imp_threshold=args.pairwise_imp_threshold,
			topk_values=args.topK_values,
			print_every=args.print_every,
			temperature=args.temperature,
			**(
					{
						'lora_rank': args.lora_rank,
						'lora_alpha': args.lora_alpha,
						'lora_dropout': args.lora_dropout
					} if args.strategy in LORA_FAMILY_STRATEGIES else {}
				),
			**(
					{
						'min_phases_before_stopping': args.min_phases_before_stopping,
						'min_epochs_per_phase': args.min_epochs_per_phase,
						'total_num_phases': args.total_num_phases,
					} if args.strategy == 'progressive' else {}
					),
			**(
					{
						'probe_dropout': args.probe_dropout,
					} if args.strategy == 'probe' else {}
				),
			**(
					{
						'lora_plus_lambda': args.lora_plus_lambda,
					} if args.strategy == 'lora_plus' else {}
				),
			**(
					{
						'clip_adapter_method': args.adapter_method,
						'bottleneck_dim': 256,
						'activation': 'relu',
					} if args.strategy == 'adapter' and args.adapter_method and args.adapter_method.startswith('clip_adapter') else {}
				),
			**(
					{
						'tip_adapter_method': args.adapter_method,
						'initial_beta': 1.0,
						'initial_alpha': 1.0,
						'support_shots': 16,
					} if args.strategy == 'adapter' and args.adapter_method and args.adapter_method.startswith('tip_adapter') else {}
				),
		)
		
		# # Clean up any available JSON/PT files before finishing
		# json_files = glob.glob(os.path.join(RESULT_DIRECTORY, "*.json"))
		# pt_files = glob.glob(os.path.join(RESULT_DIRECTORY, "*.pt"))
		# cleanup_files = json_files + pt_files
		# if cleanup_files:
		# 	print(f"Cleaning up {len(cleanup_files)} file(s) from {RESULT_DIRECTORY}:")
		# 	for f in cleanup_files:
		# 		print(f)
		# 		try:
		# 			os.remove(f)
		# 		except Exception as e:
		# 			print(f"Warning: Failed to remove {f}: {e}")

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