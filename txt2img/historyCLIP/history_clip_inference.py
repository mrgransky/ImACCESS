from functools import cache
import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from historical_dataset_loader import get_dataloaders, get_preprocess
from model import get_lora_clip
from trainer import pretrain, evaluate_best_model
from visualize import (
	plot_image_to_texts_stacked_horizontal_bar, 
	plot_text_to_images, 
	plot_image_to_texts_pretrained, 
	plot_comparison_metrics_split, 
	plot_comparison_metrics_merged, 
	plot_text_to_images_merged, 
	plot_image_to_texts_separate_horizontal_bars
)

# "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg"
# "https://pbs.twimg.com/media/Gowu5zDaYAAZ2YK?format=jpg"
# "https://pbs.twimg.com/media/Go0qRhvWEAAIxpn?format=png"

# # run in local for all fine-tuned models:
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -qi "https://pbs.twimg.com/media/Gowu5zDaYAAZ2YK?format=jpg" -ql "military personnel" -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_full_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_actual_epochs_15_dropout_0.0_lr_1.0e-05_wd_1.0e-02_bs_64_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_progressive_unfreeze_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_dropout_0.0_init_lr_1.0e-05_init_wd_1.0e-02_bs_64_best_model.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_lora_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_actual_epochs_15_lr_1.0e-05_wd_1.0e-02_lora_rank_32_lora_alpha_64.0_lora_dropout_0.05_bs_64_best_model.pth -lor 32 -loa 64.0 -lod 0.05

# # run in pouta for all fine-tuned models:
# $ nohup python -u history_clip_inference.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -qi "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg" -ql "aircraft" --device "cuda:2" -k 5 -fcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_full_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_actual_epochs_21_dropout_0.1_lr_1.0e-05_wd_1.0e-01_bs_64_best_model.pth -pcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_progressive_unfreeze_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_dropout_0.1_init_lr_1.0e-05_init_wd_1.0e-02_bs_64_best_model.pth -lcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_lora_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_110_lr_1.0e-05_wd_1.0e-02_lora_rank_64_lora_alpha_128.0_lora_dropout_0.05_bs_64_best_model.pth -lor 64 -loa 128.0 -lod 0.05 > /media/volume/ImACCESS/trash/history_clip_inference.txt &

# # run in Puhti:
# $ python history_clip_inference.py -ddir /scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02 -fcp /scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/results/WWII_1939-09-01_1945-09-02_full_finetune_CLIP_ViT-B-32_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_150_do_0.05_lr_5.0e-05_wd_1.0e-02_bs_64_best_model.pth -pcp /scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/results/WWII_1939-09-01_1945-09-02_progressive_unfreeze_finetune_CLIP_ViT-B-32_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_150_do_0.05_init_lr_5.0e-05_init_wd_1.0e-02_bs_64_best_model.pth -lcp /scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/results/WWII_1939-09-01_1945-09-02_lora_finetune_CLIP_ViT-B-32_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_150_lr_5.0e-05_wd_1.0e-02_lora_rank_8_lora_alpha_16.0_lora_dropout_0.05_bs_64_best_model.pth -lor 8 -loa 16.0 -lod 0.05


@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=16, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--query_image', '-qi', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
	parser.add_argument('--full_checkpoint', '-fcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--lora_checkpoint', '-lcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--progressive_checkpoint', '-pcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--lora_rank', '-lor', type=int, default=None, help='LoRA rank (used if finetune_strategy=lora)')
	parser.add_argument('--lora_alpha', '-loa', type=float, default=None, help='LoRA alpha (used if finetune_strategy=lora)')
	parser.add_argument('--lora_dropout', '-lod', type=float, default=None, help='LoRA dropout (used if finetune_strategy=lora)')
	parser.add_argument('--topK_values', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	set_seeds(seed=42)

	assert args.query_image is not None, "query_image must be provided for qualitative mode"
	assert args.query_label is not None, "query_label must be provided for qualitative mode"
	assert args.topK is not None, "topK must be provided for qualitative mode"

	if args.lora_checkpoint is not None:
		if not os.path.exists(args.lora_checkpoint):
			raise ValueError(f"Checkpoint path {args.lora_checkpoint} does not exist!")
		if args.lora_rank is None or args.lora_alpha is None or args.lora_dropout is None:
			raise ValueError("Please provide LoRA parameters for comparison!")
		if f"_lora_rank_{args.lora_rank}" not in args.lora_checkpoint:
			raise ValueError("LoRA rank in checkpoint path does not match provided LoRA rank!")
		if f"_lora_alpha_{args.lora_alpha}" not in args.lora_checkpoint:
			raise ValueError("LoRA alpha in checkpoint path does not match provided LoRA alpha!")
		if f"_lora_dropout_{args.lora_dropout}" not in args.lora_checkpoint:
			raise ValueError("LoRA dropout in checkpoint path does not match provided LoRA dropout!") 

	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
	RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results")
	CACHE_DIRECTORY = os.path.join(RESULT_DIRECTORY, "inference_cache")
	os.makedirs(RESULT_DIRECTORY, exist_ok=True)
	os.makedirs(CACHE_DIRECTORY, exist_ok=True)
	models_to_plot = {}
	print(f">> CLIP model configuration: {args.model_architecture}...")
	model_config = get_config(architecture=args.model_architecture)
	print(json.dumps(model_config, indent=4, ensure_ascii=False))
	pretrained_model, pretrained_preprocess = clip.load(
		name=args.model_architecture,
		device=args.device,
		download_root=get_model_directory(path=args.dataset_dir),
	)
	pretrained_model = pretrained_model.float() # Convert model parameters to FP32
	pretrained_model_name = pretrained_model.__class__.__name__ # CLIP
	pretrained_model.name = args.model_architecture # ViT-B/32
	pretrained_model_arch = re.sub(r'[/@]', '-', args.model_architecture)

	if not all(pretrained_model_arch in checkpoint for checkpoint in [args.full_checkpoint, args.lora_checkpoint, args.progressive_checkpoint]):
		raise ValueError("Checkpoint path does not match the assigned model architecture!")

	models_to_plot["pretrained"] = pretrained_model

	train_loader, validation_loader = get_dataloaders(
		dataset_dir=args.dataset_dir,
		sampling=args.sampling,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=model_config["image_resolution"],
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)
	criterion = torch.nn.CrossEntropyLoss()

	customized_preprocess = get_preprocess(
		dataset_dir=args.dataset_dir, 
		input_resolution=model_config["image_resolution"],
	)

	# for all finetuned models(+ pre-trained):
	finetuned_checkpoint_paths = {
		"full": args.full_checkpoint,
		"lora": args.lora_checkpoint,
		"progressive": args.progressive_checkpoint,
	}
	print(json.dumps(finetuned_checkpoint_paths, indent=4, ensure_ascii=False))

	# Load Fine-tuned Models
	fine_tuned_models = {}
	finetuned_img2txt_dict = {args.model_architecture: {}}
	finetuned_txt2img_dict = {args.model_architecture: {}}
	for ft_name, ft_path in finetuned_checkpoint_paths.items():
		if ft_path and os.path.exists(ft_path):
			print(f">> Loading Fine-tuned Model: {ft_name} from {ft_path}...")
			model, _ = clip.load(
				name=args.model_architecture,
				device=args.device,
				download_root=get_model_directory(path=args.dataset_dir),
			)
			if ft_name == "lora":
				model = get_lora_clip(
					clip_model=model,
					lora_rank=args.lora_rank,
					lora_alpha=args.lora_alpha,
					lora_dropout=args.lora_dropout,
					verbose=False,
				)
				model.to(args.device)
			
			model = model.float()
			model.name = args.model_architecture
			checkpoint = torch.load(ft_path, map_location=args.device)
			if 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'])
			else:
				model.load_state_dict(checkpoint)
			model = model.float()
			fine_tuned_models[ft_name] = model

			# Evaluate finetuned model
			evaluation_results = evaluate_best_model(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				early_stopping=None,
				checkpoint_path=finetuned_checkpoint_paths.get(ft_name, None),
				finetune_strategy=ft_name,
				device=args.device,
				cache_dir=CACHE_DIRECTORY,
				topk_values=args.topK_values,
				verbose=True,
				clean_cache=False, # don't clean cache for all models [to speedup]
			)
			finetuned_img2txt_dict[args.model_architecture][ft_name] = evaluation_results["img2txt_metrics"]
			finetuned_txt2img_dict[args.model_architecture][ft_name] = evaluation_results["txt2img_metrics"]
		else:
			print(f"WARNING: Fine-tuned model not found at {ft_path}. Skipping {ft_name}")
	models_to_plot.update(fine_tuned_models)
	print(f">> Fine-tuned models loaded successfully.")
	# print(f"finetuned_img2txt_dict:")
	# print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
	# print(f"finetuned_txt2img_dict:")
	# print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))

	####################################### Qualitative Analysis #######################################
	if args.query_image is not None:
		plot_image_to_texts_pretrained(
			best_pretrained_model=pretrained_model,
			validation_loader=validation_loader,
			# preprocess=pretrained_preprocess, # customized_preprocess,
			preprocess=customized_preprocess,
			img_path=args.query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)
		plot_image_to_texts_stacked_horizontal_bar(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=args.query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)
		plot_image_to_texts_separate_horizontal_bars(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=args.query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)

	if args.query_label is not None:
		plot_text_to_images(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			query_text=args.query_label,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			cache_dir=CACHE_DIRECTORY,
		)
		plot_text_to_images_merged(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			query_text=args.query_label,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			cache_dir=CACHE_DIRECTORY,
		)
	####################################### Qualitative Analysis #######################################


	####################################### Quantitative Analysis #######################################
	finetune_strategies = []
	if args.full_checkpoint is not None:
		finetune_strategies.append("full")
	if args.lora_checkpoint is not None:
		finetune_strategies.append("lora")
	if args.progressive_checkpoint is not None:
		finetune_strategies.append("progressive")
	if len(finetune_strategies) == 0:
		raise ValueError("Please provide at least one checkpoint for comparison!")
	print(f">> All available finetune strategies: {finetune_strategies}")

	print(f">> Computing metrics for pretrained {args.model_architecture}...")
	pretrained_img2txt_dict = {args.model_architecture: {}}
	pretrained_txt2img_dict = {args.model_architecture: {}}
	pretrained_img2txt, pretrained_txt2img = pretrain(
		model=pretrained_model,
		validation_loader=validation_loader,
		results_dir=RESULT_DIRECTORY,
		cache_dir=CACHE_DIRECTORY,
		device=args.device,
		topk_values=args.topK_values,
		verbose=False,
	)
	pretrained_img2txt_dict[args.model_architecture] = pretrained_img2txt
	pretrained_txt2img_dict[args.model_architecture] = pretrained_txt2img
	print(f">> Pretrained model metrics computed successfully. [for Quantitative Analysis]")
	# print(f"pretrained_img2txt_dict:")
	# print(json.dumps(pretrained_img2txt_dict, indent=4, ensure_ascii=False))
	# print(f"pretrained_txt2img_dict:")
	# print(json.dumps(pretrained_txt2img_dict, indent=4, ensure_ascii=False))

	plot_comparison_metrics_split(
		dataset_name=validation_loader.name,
		pretrained_img2txt_dict=pretrained_img2txt_dict,
		pretrained_txt2img_dict=pretrained_txt2img_dict,
		finetuned_img2txt_dict=finetuned_img2txt_dict,
		finetuned_txt2img_dict=finetuned_txt2img_dict,
		model_name=args.model_architecture,
		finetune_strategies=finetune_strategies,
		topK_values=args.topK_values,
		results_dir=RESULT_DIRECTORY,
	)
	plot_comparison_metrics_merged(
		dataset_name=validation_loader.name,
		pretrained_img2txt_dict=pretrained_img2txt_dict,
		pretrained_txt2img_dict=pretrained_txt2img_dict,
		finetuned_img2txt_dict=finetuned_img2txt_dict,
		finetuned_txt2img_dict=finetuned_txt2img_dict,
		model_name=args.model_architecture,
		finetune_strategies=finetune_strategies,
		topK_values=args.topK_values,
		results_dir=RESULT_DIRECTORY,
	)
	####################################### Quantitative Analysis #######################################

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
