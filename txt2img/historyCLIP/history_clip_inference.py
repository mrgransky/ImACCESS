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
# "https://pbs.twimg.com/media/Go2T7FJbIAApElq?format=jpg"

# # run in local for all fine-tuned models with image and label:
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -qi "https://pbs.twimg.com/media/Go0qRhvWEAAIxpn?format=png" -ql "military personnel" -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_full_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_actual_epochs_15_dropout_0.0_lr_1.0e-05_wd_1.0e-02_bs_64_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_progressive_unfreeze_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_dropout_0.0_init_lr_1.0e-05_init_wd_1.0e-02_bs_64_best_model.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/SMU_1900-01-01_1970-12-31_lora_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_15_actual_epochs_15_lr_1.0e-05_wd_1.0e-02_lora_rank_32_lora_alpha_64.0_lora_dropout_0.05_bs_64_best_model.pth

# # Local | All fine-tuned models (head, torso, tail):
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/full_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_dropout_0.05_lr_1.0e-05_wd_1.0e-02_bs_64_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/progressive_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_dropout_0.05_ilr_1.0e-05_iwd_1.0e-02_bs_64_best_model_last_phase_0_flr_1.0e-05_fwd_0.01.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/lora_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_lr_1.0e-05_wd_1.0e-02_lor_64_loa_128.0_lod_0.05_bs_64_best_model.pth

# # run in pouta for all fine-tuned models:
# $ nohup python -u history_clip_inference.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -nw 32 --device "cuda:2" -k 5 -bs 256 -fcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_full_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_actual_epochs_21_dropout_0.1_lr_1.0e-05_wd_1.0e-01_bs_64_best_model.pth -pcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_progressive_unfreeze_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_dropout_0.1_init_lr_1.0e-05_init_wd_1.0e-02_bs_64_best_model.pth -lcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/HISTORY_X4_lora_finetune_CLIP_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_110_lr_1.0e-05_wd_1.0e-02_lora_rank_64_lora_alpha_128.0_lora_dropout_0.05_bs_64_best_model.pth > /media/volume/ImACCESS/trash/history_clip_inference.txt &

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=8, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--batch_size', '-bs', type=int, default=128, help='Batch size for training')
	parser.add_argument('--query_image', '-qi', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')
	parser.add_argument('--full_checkpoint', '-fcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--lora_checkpoint', '-lcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--progressive_checkpoint', '-pcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--topK_values', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	set_seeds(seed=42)
	RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results")
	CACHE_DIRECTORY = os.path.join(RESULT_DIRECTORY, "inference_cache")
	os.makedirs(RESULT_DIRECTORY, exist_ok=True)
	os.makedirs(CACHE_DIRECTORY, exist_ok=True)

	if args.full_checkpoint is not None:
		assert os.path.exists(args.full_checkpoint), f"full_checkpoint {args.full_checkpoint} does not exist!"
	if args.lora_checkpoint is not None:
		assert os.path.exists(args.lora_checkpoint), f"lora_checkpoint {args.lora_checkpoint} does not exist!"
	if args.progressive_checkpoint is not None:
		assert os.path.exists(args.progressive_checkpoint), f"progressive_checkpoint {args.progressive_checkpoint} does not exist!"
	if args.lora_checkpoint is not None:
		params = get_lora_params(args.lora_checkpoint)
		if params:
			print(f">> {args.lora_checkpoint}\n\tLoRA parameters: {params}")
			args.lora_rank = params['lora_rank']
			args.lora_alpha = params['lora_alpha']
			args.lora_dropout = params['lora_dropout']
		else:
			raise ValueError("LoRA parameters not found in the provided checkpoint path!")

	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	# print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
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

	# ####################################### Randomly select one sample from validation set #######################################
	# if args.query_image is None or args.query_label is None:
	# 	print("One or both of query_image and query_label not provided. Selecting a random sample from validation set...")
	# 	validation_dataset = validation_loader.dataset
	# 	# Use a random seed based on the current time
	# 	rng = random.Random(int(time.time() * 1000))  # Seed with millisecond timestamp		
	# 	random_idx = rng.randint(0, len(validation_dataset) - 1)
	# 	# random_idx = random.randint(0, len(validation_dataset) - 1) # reproducible
	# 	random_sample = validation_dataset.data_frame.iloc[random_idx]
	# 	if args.query_image is None:
	# 		args.query_image = random_sample['img_path']
	# 		print(f"Selected random image: {args.query_image}")
	# 	if args.query_label is None:
	# 		args.query_label = random_sample['label']
	# 		print(f"Selected random label: {args.query_label}")
	# ####################################### Randomly select one sample from validation set #######################################

	# Systematic selection of samples from validation set: Head, Torso, Tail
	if args.query_image is None or args.query_label is None:
		print("One or both of query_image and query_label not provided. Selecting samples from validation set...")
		i2t_samples, t2i_samples = select_qualitative_samples(
			metadata_path=os.path.join(args.dataset_dir, "metadata.csv"),
			metadata_train_path=os.path.join(args.dataset_dir, "metadata_train.csv"),
			metadata_val_path=os.path.join(args.dataset_dir, "metadata_val.csv"),
			num_samples_per_segment=5 # Select 5 samples per segment
		)
		if i2t_samples and t2i_samples:
			QUERY_IMAGES = [sample['image_path'] for sample in i2t_samples]
			QUERY_LABELS = [sample['label'] for sample in t2i_samples]
		else:
			raise ValueError("No samples selected from validation set!")
	else:
		QUERY_IMAGES = [args.query_image]
		QUERY_LABELS = [args.query_label]
	print(len(QUERY_IMAGES), QUERY_IMAGES)
	print()
	print(len(QUERY_LABELS), QUERY_LABELS)

	# for all finetuned models(+ pre-trained):
	finetuned_checkpoint_paths = {
		"full": args.full_checkpoint,
		"lora": args.lora_checkpoint,
		"progressive": args.progressive_checkpoint,
	}
	print(json.dumps(finetuned_checkpoint_paths, indent=4, ensure_ascii=False))

	# Load Fine-tuned Models
	print("Loading Fine-tuned Models [takes a while]...")
	ft_start = time.time()
	fine_tuned_models = {}
	finetuned_img2txt_dict = {args.model_architecture: {}}
	finetuned_txt2img_dict = {args.model_architecture: {}}
	for ft_name, ft_path in finetuned_checkpoint_paths.items():
		if ft_path and os.path.exists(ft_path):
			model, _ = clip.load(name=args.model_architecture, device=args.device, download_root=get_model_directory(path=args.dataset_dir))
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
			model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
			fine_tuned_models[ft_name] = model
	print(f"Fine-tuned Models loaded in {time.time() - ft_start:.5f} sec")
	models_to_plot.update(fine_tuned_models)

	print("Computing Model Embeddings [sequentially]...")
	mdl_emb_start = time.time()
	embeddings_cache = {}
	for strategy, model in models_to_plot.items():
		embeddings, paths = compute_model_embeddings(
			strategy=strategy,
			model=model,
			loader=validation_loader,
			device=args.device,
			cache_dir=CACHE_DIRECTORY,
			lora_rank=args.lora_rank if strategy == "lora" else None,
			lora_alpha=args.lora_alpha if strategy == "lora" else None,
			lora_dropout=args.lora_dropout if strategy == "lora" else None,
		)
		embeddings_cache[strategy] = (embeddings, paths)
	print(f"Model Embeddings computed in {time.time() - mdl_emb_start:.5f} sec")

	# Evaluate fine-tuned models
	for ft_name, ft_path in finetuned_checkpoint_paths.items():
		if ft_name in fine_tuned_models:
			evaluation_results = evaluate_best_model(
				model=fine_tuned_models[ft_name],
				validation_loader=validation_loader,
				criterion=criterion,
				early_stopping=None,
				checkpoint_path=ft_path,
				finetune_strategy=ft_name,
				device=args.device,
				cache_dir=CACHE_DIRECTORY,
				topk_values=args.topK_values,
				verbose=True,
				clean_cache=False,
				embeddings_cache=embeddings_cache[ft_name],
				max_in_batch_samples=None, # get_max_samples(batch_size=args.batch_size, N=10, device=args.device),
				lora_params={
					"lora_rank": args.lora_rank,
					"lora_alpha": args.lora_alpha,
					"lora_dropout": args.lora_dropout,
				} if ft_name == "lora" else None,
			)
			finetuned_img2txt_dict[args.model_architecture][ft_name] = evaluation_results["img2txt_metrics"]
			finetuned_txt2img_dict[args.model_architecture][ft_name] = evaluation_results["txt2img_metrics"]

	####################################### Qualitative Analysis #######################################
	print(f"Qualitative Analysis".center(160, " "))
	for query_image in QUERY_IMAGES:
		print(f">> Query Image: {query_image}")
		plot_image_to_texts_pretrained(
			best_pretrained_model=pretrained_model,
			validation_loader=validation_loader,
			# preprocess=pretrained_preprocess, # customized_preprocess,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)
		plot_image_to_texts_stacked_horizontal_bar(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)
		plot_image_to_texts_separate_horizontal_bars(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			img_path=query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)

	for query_label in QUERY_LABELS:
		print(f">> Query Label: {query_label}")
		plot_text_to_images(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			query_text=query_label,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			cache_dir=CACHE_DIRECTORY,
			embeddings_cache=embeddings_cache,
		)
		plot_text_to_images_merged(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=customized_preprocess,
			query_text=query_label,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			cache_dir=CACHE_DIRECTORY,
			embeddings_cache=embeddings_cache,
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
		embeddings_cache=embeddings_cache["pretrained"],
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
	multiprocessing.set_start_method('spawn')
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))