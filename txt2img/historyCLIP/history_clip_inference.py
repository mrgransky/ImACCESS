import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

from utils import *
from historical_dataset_loader import get_single_label_dataloaders, get_multi_label_dataloaders, get_preprocess
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
# https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg

# # run in local for all fine-tuned models with image and label:
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -qi "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg" -ql "military personnel" -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/full_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_24_dropout_0.0_lr_1.0e-05_wd_1.0e-02_bs_64_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/progressive_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_dropout_0.0_ilr_1.0e-05_iwd_1.0e-02_bs_64_best_model_last_phase_1_flr_6.9e-06_fwd_0.01030392170778281.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results/lora_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_lr_1.0e-05_wd_1.0e-02_lor_64_loa_128.0_lod_0.05_bs_64_best_model.pth

# ################ Local ################ 
# All fine-tuned models (head, torso, tail) 
# Single-label:
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_single_label/full_single_label_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_16_dropout_0.0_lr_1.0e-05_wd_1.0e-02_bs_32_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_single_label/progressive_single_label_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_25_dropout_0.0_ilr_1.0e-05_iwd_1.0e-02_bs_32_best_model_last_phase_1_flr_6.9e-06_fwd_0.010306122448979592.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_single_label/lora_single_label_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_25_actual_eps_18_lr_1.0e-05_wd_1.0e-02_lor_8_loa_16.0_lod_0.05_bs_32_best_model.pth

# Multi-label:
# $ python history_clip_inference.py -ddir /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31 -dt multi_label -fcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_multi_label/full_multi_label_ViT-B-32_AdamW_OneCycleLR_BCEWithLogitsLoss_GradScaler_ieps_25_actual_eps_17_dropout_0.0_lr_1.0e-05_wd_1.0e-02_temp_0.07_bs_16_best_model.pth -pcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_multi_label/progressive_multi_label_ViT-B-32_AdamW_OneCycleLR_BCEWithLogitsLoss_GradScaler_ieps_25_actual_eps_25_dropout_0.0_ilr_1.0e-05_iwd_1.0e-02_temp_0.07_bs_16_best_model_last_phase_1_flr_5.6e-06_fwd_0.010306122448979592.pth -lcp /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/results_multi_label/lora_multi_label_ViT-B-32_AdamW_OneCycleLR_BCEWithLogitsLoss_GradScaler_ieps_25_actual_eps_17_lr_1.0e-05_wd_1.0e-02_lor_8_loa_16.0_lod_0.05_temp_0.07_bs_16_best_model.pth
# ################ Local ################ 

# # run in pouta for all fine-tuned models:
# $ nohup python -u history_clip_inference.py -ddir /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4 -nw 32 --device "cuda:2" -k 5 -bs 256 -fcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/full_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_110_actual_eps_23_dropout_0.1_lr_5.0e-06_wd_1.0e-02_bs_64_best_model.pth -pcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/progressive_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_110_actual_eps_84_dropout_0.1_ilr_5.0e-06_iwd_1.0e-02_bs_64_best_model_last_phase_3_flr_2.3e-06_fwd_0.012021761646381529.pth -lcp /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/results/lora_ViT-B-32_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_ieps_110_actual_eps_26_lr_5.0e-06_wd_1.0e-02_lor_64_loa_128.0_lod_0.05_bs_64_best_model.pth > /media/volume/ImACCESS/trash/history_clip_inference.txt &

def evaluate_pretrained_clip_multilabel(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		device: str,
		topk_values: List[int],
		verbose: bool = True,
		embeddings_cache: tuple = None,
):
		"""
		Multi-label baseline evaluation for pre-trained CLIP.
		Only used when dataset_type == 'multi_label'
		"""
		if verbose:
				print("Evaluating pre-trained CLIP for multi-label classification...")
		
		model.eval()
		
		# Get class information
		try:
				class_names = validation_loader.dataset.unique_labels
				num_classes = len(class_names)
		except AttributeError:
				class_names = validation_loader.dataset.dataset.classes
				num_classes = len(class_names)
		
		if verbose:
				print(f"Multi-label evaluation: {num_classes} classes")
		
		# Use embeddings cache if available
		if embeddings_cache is not None:
				all_image_embeds, all_labels = embeddings_cache
				all_image_embeds = all_image_embeds.to(device).float()
				all_labels = all_labels.to(device)
				if verbose:
						print("Using cached embeddings for evaluation")
		else:
				# Compute embeddings from scratch
				if verbose:
						print("Computing embeddings from scratch...")
				all_image_embeds, all_labels = _compute_image_embeddings_multilabel(
						model, validation_loader, device, verbose
				)
		
		# Pre-encode all class texts
		all_class_texts = clip.tokenize(class_names).to(device)
		with torch.no_grad():
				all_class_embeds = model.encode_text(all_class_texts)
				all_class_embeds = F.normalize(all_class_embeds, dim=-1)
		
		# Compute similarities [num_samples, num_classes]
		with torch.no_grad():
				similarities = torch.matmul(all_image_embeds, all_class_embeds.T)
				probabilities = torch.sigmoid(similarities)
		
		# Find optimal threshold
		if verbose:
				print("Finding optimal threshold for multi-label classification...")
		
		optimal_threshold = _find_optimal_multilabel_threshold(
				probabilities, all_labels, validation_split=0.2, verbose=verbose
		)
		
		# Apply threshold to get predictions
		predictions = (probabilities > optimal_threshold).float()
		
		# Compute multi-label metrics
		metrics = _compute_multilabel_baseline_metrics(
				predictions, all_labels, similarities, class_names, topk_values, device
		)
		
		if verbose:
				print(f"Optimal threshold: {optimal_threshold:.4f}")
				print(f"Baseline metrics computed for {num_classes} classes")
		
		return metrics

def _compute_image_embeddings_multilabel(model, validation_loader, device, verbose):
		"""Helper function for multi-label embedding computation."""
		all_image_embeds = []
		all_labels = []
		
		model.eval()
		iterator = tqdm(validation_loader, desc="Encoding images") if verbose else validation_loader
		
		for images, _, labels in iterator:
				images = images.to(device, non_blocking=True)
				
				with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
						image_embeds = model.encode_image(images)
				
				image_embeds = F.normalize(image_embeds.float(), dim=-1)
				all_image_embeds.append(image_embeds.cpu())
				all_labels.append(labels.cpu())
		
		all_image_embeds = torch.cat(all_image_embeds, dim=0)
		all_labels = torch.cat(all_labels, dim=0)
		
		return all_image_embeds.to(device), all_labels.to(device)

def _find_optimal_multilabel_threshold(
		probabilities: torch.Tensor, 
		labels: torch.Tensor, 
		validation_split: float = 0.2,
		verbose: bool = True
) -> float:
		"""Find optimal threshold for multi-label classification."""
		try:
				from sklearn.metrics import f1_score
		except ImportError:
				print("Warning: sklearn not available, using frequency-based threshold")
				sparsity = labels.float().mean().item()
				return max(0.1, min(0.9, sparsity))
		
		num_samples = probabilities.shape[0]
		val_size = int(num_samples * validation_split)
		
		if val_size < 10:
				# Fallback: use label frequency as threshold
				sparsity = labels.float().mean().item()
				threshold = max(0.1, min(0.9, sparsity))
				if verbose:
						print(f"Using frequency-based threshold: {threshold:.4f}")
				return threshold
		
		# Split data for threshold validation
		val_probs = probabilities[:val_size]
		val_labels = labels[:val_size]
		
		best_threshold = 0.5
		best_f1 = 0.0
		
		# Test different thresholds
		thresholds = torch.linspace(0.05, 0.95, 19)
		for threshold in thresholds:
				predictions = (val_probs > threshold).float()
				try:
						f1 = f1_score(
								val_labels.cpu().numpy(), 
								predictions.cpu().numpy(), 
								average='weighted',
								zero_division=0
						)
						if f1 > best_f1:
								best_f1 = f1
								best_threshold = threshold.item()
				except:
						continue
		
		if verbose:
				print(f"Best F1: {best_f1:.4f} at threshold: {best_threshold:.4f}")
		
		return best_threshold

def _compute_multilabel_baseline_metrics(
		predictions: torch.Tensor,
		labels: torch.Tensor, 
		similarities: torch.Tensor,
		class_names: List[str],
		topk_values: List[int],
		device: str
) -> Dict:
		"""Compute comprehensive multi-label baseline metrics."""
		try:
				from sklearn.metrics import hamming_loss, f1_score
				
				# Basic multi-label metrics
				hamming = hamming_loss(labels.cpu().numpy(), predictions.cpu().numpy())
				
				# F1 Score
				f1 = f1_score(
						labels.cpu().numpy(), 
						predictions.cpu().numpy(), 
						average='weighted',
						zero_division=0
				)
		except ImportError:
				print("Warning: sklearn not available, using basic metrics only")
				hamming = ((predictions != labels).float().sum() / (labels.shape[0] * labels.shape[1])).item()
				f1 = 0.0
		
		# Exact match accuracy (all labels must match)
		exact_match = (predictions == labels).all(dim=1).float().mean().item()
		
		# Partial accuracy (at least one correct prediction)
		partial_acc = ((predictions * labels).sum(dim=1) > 0).float().mean().item()
		
		# Top-K metrics
		topk_metrics = {}
		for k in topk_values:
				if k <= len(class_names):
						topk_indices = similarities.topk(k, dim=1)[1]
						topk_preds = torch.zeros_like(similarities).scatter_(1, topk_indices, 1.0)
						
						# Subset accuracy: all true labels are in top-k
						subset_acc = (labels <= topk_preds).all(dim=1).float().mean().item()
						topk_metrics[str(k)] = subset_acc
		
		# Average cosine similarity with true labels
		num_samples, num_classes = similarities.shape
		matched_similarities = []
		
		for i in range(num_samples):
				true_indices = torch.where(labels[i] == 1)[0]
				if len(true_indices) > 0:
						# Average similarity to all true classes
						avg_sim = similarities[i, true_indices].mean().item()
						matched_similarities.append(avg_sim)
		
		avg_cosine_sim = np.mean(matched_similarities) if matched_similarities else 0.0
		
		return {
				"hamming_loss": float(hamming),
				"exact_match_acc": float(exact_match),
				"partial_acc": float(partial_acc),
				"f1_score": float(f1),
				"topk_acc": topk_metrics,
				"cosine_similarity": float(avg_cosine_sim),
				"baseline_type": "similarity_threshold"
		}

def get_multi_label_head_torso_tail_samples(
		metadata_path: str,
		metadata_train_path: str,
		metadata_val_path: str,
		num_samples_per_segment: int = 5,
) -> Tuple[List[Dict], List[str]]:
		"""
		Sample from head, torso, tail distributions for multi-label datasets.
		Only used when dataset_type == 'multi_label'
		"""
		try:
				import ast
				# Load metadata
				df_val = pd.read_csv(metadata_val_path)
				
				# For multi-label, we need to count label frequencies differently
				all_labels = []
				for labels_str in df_val['multimodal_labels']:
						try:
								labels = ast.literal_eval(labels_str)
								all_labels.extend(labels)
						except:
								continue
				
				# Count label frequencies
				label_counts = pd.Series(all_labels).value_counts()
				
				# Define head, torso, tail based on frequency distribution
				total_labels = len(label_counts)
				head_threshold = int(total_labels * 0.2)  # Top 20%
				tail_threshold = int(total_labels * 0.8)   # Bottom 20%
				
				head_labels = set(label_counts.head(head_threshold).index)
				tail_labels = set(label_counts.tail(total_labels - tail_threshold).index)
				torso_labels = set(label_counts.index) - head_labels - tail_labels
				
				# Sample images and labels
				i2t_samples = []
				t2i_samples = []
				
				# Sample images with head, torso, tail labels
				for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
						segment_samples = []
						
						for _, row in df_val.iterrows():
								try:
										row_labels = set(ast.literal_eval(row['multimodal_labels']))
										if row_labels & segment_labels:  # If any intersection
												segment_samples.append({
														'image_path': row['img_path'],
														'labels': list(row_labels),
														'segment': segment_name
												})
								except:
										continue
						
						# Randomly sample from this segment
						if len(segment_samples) >= num_samples_per_segment:
								sampled = random.sample(segment_samples, num_samples_per_segment)
								i2t_samples.extend(sampled)
				
				# Sample text queries from head, torso, tail
				for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
						segment_label_list = list(segment_labels)
						if len(segment_label_list) >= num_samples_per_segment:
								sampled_labels = random.sample(segment_label_list, num_samples_per_segment)
								t2i_samples.extend(sampled_labels)
				
				return i2t_samples, t2i_samples
				
		except Exception as e:
				print(f"Error in multi-label sampling: {e}")
				return [], []

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Historical Archives Dataset")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='DATASET directory')
	parser.add_argument('--dataset_type', '-dt', type=str, choices=['single_label', 'multi_label'], default='single_label', help='Dataset type (single_label/multi_label)')
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=4, help='Number of CPUs [def: max cpus]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--sampling', '-s', type=str, default="stratified_random", choices=["stratified_random", "kfold_stratified"], help='Sampling method')
	parser.add_argument('--batch_size', '-bs', type=int, default=16, help='Batch size for training')
	parser.add_argument('--query_image', '-qi', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--query_label', '-ql', type=str, default=None, help='image path for zero shot classification')
	parser.add_argument('--topK', '-k', type=int, default=3, help='TopK results')
	parser.add_argument('--full_checkpoint', '-fcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--lora_checkpoint', '-lcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--progressive_checkpoint', '-pcp', type=str, default=None, help='Path to finetuned model checkpoint for comparison')
	parser.add_argument('--topK_values', type=int, nargs='+', default=[1, 3, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	set_seeds(seed=42)
	RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"results_{args.dataset_type}")
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

	if args.dataset_type == "multi_label":
		train_loader, validation_loader = get_multi_label_dataloaders(
			dataset_dir=args.dataset_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			input_resolution=model_config["image_resolution"],
		)
		criterion = torch.nn.BCEWithLogitsLoss()
	else:
		train_loader, validation_loader = get_single_label_dataloaders(
			dataset_dir=args.dataset_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			input_resolution=model_config["image_resolution"],
		)
		criterion = torch.nn.CrossEntropyLoss()
	print(f">> dataset: {args.dataset_type} => criterion: {criterion.__class__.__name__}")
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	customized_preprocess = get_preprocess(
		dataset_dir=args.dataset_dir, 
		input_resolution=model_config["image_resolution"],
	)

	# Systematic selection of samples from validation set: Head, Torso, Tail
	if args.query_image is None or args.query_label is None:
		print("Selecting samples from validation set...")
		if args.dataset_type == "multi_label":
			i2t_samples, t2i_samples = get_multi_label_head_torso_tail_samples(
				metadata_path=os.path.join(args.dataset_dir, "metadata_multimodal.csv"),
				metadata_train_path=os.path.join(args.dataset_dir, "metadata_multimodal_train.csv"),
				metadata_val_path=os.path.join(args.dataset_dir, "metadata_multimodal_val.csv"),
				num_samples_per_segment=5,
			)
			if i2t_samples and t2i_samples:
				QUERY_IMAGES = [sample['image_path'] for sample in i2t_samples]
				QUERY_LABELS = t2i_samples  # Already a list of strings
			else:
				raise ValueError("No multi-label samples selected!")
		else:
			i2t_samples, t2i_samples = get_single_label_head_torso_tail_samples(
				metadata_path=os.path.join(args.dataset_dir, "metadata.csv"),
				metadata_train_path=os.path.join(args.dataset_dir, "metadata_train.csv"),
				metadata_val_path=os.path.join(args.dataset_dir, "metadata_val.csv"),
				num_samples_per_segment=10,
			)
			if i2t_samples and t2i_samples:
				QUERY_IMAGES = [sample['image_path'] for sample in i2t_samples]
				QUERY_LABELS = [sample['label'] for sample in t2i_samples]
			else:
				raise ValueError("No single-label samples selected!")
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
	# multiprocessing.set_start_method('spawn')
	main()