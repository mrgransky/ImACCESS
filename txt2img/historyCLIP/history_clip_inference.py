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
from visualize import plot_image_to_texts_stacked_horizontal_bar, plot_text_to_images, plot_image_to_texts_pretrained, plot_comparison_metrics_split

def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--mode', '-m', type=str, choices=['quantitative', 'qualitative'], required=True, help='Choose mode (qualitative/quantitative)')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=16, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--query_image', '-qi', type=str, default="/home/farid/datasets/TEST_IMGs/5968_115463.jpg", help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default="aircraft", help='image path for zero shot classification')
	parser.add_argument('--topK', '-k', type=int, default=5, help='TopK results')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	set_seeds(seed=42)
	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	print(clip.available_models()) # ViT-[size]/[patch_size][@resolution] or RN[depth]x[width_multiplier]
	RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results")
	os.makedirs(RESULT_DIRECTORY, exist_ok=True)

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
	pretrained_model.name = args.model_architecture
	pretrained_model_arch = re.sub(r'[/@]', '-', args.model_architecture)


	# Define paths to fine-tuned models
	base_path = os.path.join(args.dataset_dir, "results")
	finetuned_checkpoint_paths = {
		"full": os.path.join(base_path, f"{os.path.basename(args.dataset_dir)}_full_finetune_{pretrained_model_name}_{pretrained_model_arch}_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_9_do_0.0_lr_1.0e-04_wd_1.0e-02_bs_64_best_model.pth"),
		"lora": os.path.join(base_path, f"{os.path.basename(args.dataset_dir)}_lora_finetune_{pretrained_model_name}_{pretrained_model_arch}_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_9_lr_1.0e-04_wd_1.0e-02_lora_rank_8_lora_alpha_16.0_lora_dropout_0.05_bs_64_best_model.pth"),
		"progressive": os.path.join(base_path, f"{os.path.basename(args.dataset_dir)}_progressive_unfreeze_finetune_{pretrained_model_name}_{pretrained_model_arch}_opt_AdamW_sch_OneCycleLR_loss_CrossEntropyLoss_scaler_GradScaler_init_epochs_9_do_0.0_init_lr_1.0e-04_init_wd_1.0e-02_bs_64_best_model.pth"),
	}

	# Load Fine-tuned Models
	fine_tuned_models = {}
	for ft_name, ft_path in finetuned_checkpoint_paths.items():
		if os.path.exists(ft_path):
			print(f">> Loading Fine-tuned Model: {ft_name} from {ft_path}...")
			model, _ = clip.load(
				name=args.model_architecture,
				device=args.device,
				download_root=get_model_directory(path=args.dataset_dir),
			)
			if ft_name == "lora":
				model = get_lora_clip(
					clip_model=model,
					lora_rank=8,
					lora_alpha=16.0,
					lora_dropout=0.05,
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
		else:
			print(f"WARNING: Fine-tuned model not found at {ft_path}. Skipping {ft_name}")

	train_loader, validation_loader = get_dataloaders(
		dataset_dir=args.dataset_dir,
		sampling=args.sampling,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		input_resolution=model_config["image_resolution"],
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	# Prepare list of models for plotting
	models_to_plot = {"pretrained": pretrained_model}
	models_to_plot.update(fine_tuned_models)

	customized_preprocess = get_preprocess(dataset_dir=args.dataset_dir, input_resolution=model_config["image_resolution"])

	# 1. Compute pretrained model metrics
	print(f">> Computing metrics for pretrained {args.model_architecture}...")
	pretrained_img2txt, pretrained_txt2img = pretrain(
		model=pretrained_model,
		validation_loader=validation_loader,
		results_dir=RESULT_DIRECTORY,
		device=args.device,
		topk_values=[1, 3, 5, 10, 15, 20],
	)
	pretrained_img2txt_dict = {args.model_architecture: pretrained_img2txt}
	pretrained_txt2img_dict = {args.model_architecture: pretrained_txt2img}
	print(f">> Pretrained model metrics computed successfully.")

	# 2. Evaluate finetuned model
	criterion = torch.nn.CrossEntropyLoss()
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=None,
		checkpoint_path=finetuned_checkpoint_paths.get("full", None),
		device=args.device,
		topk_values=[1, 3, 5, 10, 15, 20],
		verbose=True
	)
	finetuned_img2txt_dict = {args.model_architecture: evaluation_results["img2txt_metrics"]}
	finetuned_txt2img_dict = {args.model_architecture: evaluation_results["txt2img_metrics"]}

	# 3. Plot qualitative results
	if args.mode == "qualitative":
		plot_image_to_texts_pretrained(
			best_pretrained_model=pretrained_model,
			validation_loader=validation_loader,
			preprocess=pretrained_preprocess, # customized_preprocess,
			img_path=args.query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)

		plot_image_to_texts_stacked_horizontal_bar(
			models=models_to_plot,
			validation_loader=validation_loader,
			preprocess=pretrained_preprocess,
			img_path=args.query_image,
			topk=args.topK,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
		)
	elif args.mode == "quantitative":


		plot_comparison_metrics_split(
			dataset_name=validation_loader.name,
			pretrained_img2txt_dict=pretrained_img2txt_dict,
			pretrained_txt2img_dict=pretrained_txt2img_dict,
			finetuned_img2txt_dict=finetuned_img2txt_dict,
			finetuned_txt2img_dict=finetuned_txt2img_dict,
			model_name=args.model_architecture,
			finetune_strategy="full",
			topK_values=[1, 3, 5, 10, 15, 20],
			results_dir=RESULT_DIRECTORY,
		)
	else:
		raise ValueError(f"Invalid mode: {args.mode}. Choose between: 'qualitative', 'quantitative'!")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))