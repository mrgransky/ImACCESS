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

def _compute_similarities_chunked(
		image_embeds: torch.Tensor,
		class_embeds: torch.Tensor,
		chunk_size: int = 100,
		device: str = "cuda",
		temperature: float = 0.07
	) -> torch.Tensor:
	num_images = image_embeds.size(0)
	num_classes = class_embeds.size(0)
	
	# Pre-allocate result tensor on CPU to save GPU memory
	similarities = torch.zeros(num_images, num_classes, dtype=torch.float32, device=device,)
	
	for i in range(0, num_images, chunk_size):
		end_i = min(i + chunk_size, num_images)
		
		# Move only the current chunk to GPU
		img_chunk = image_embeds[i:end_i].to(device)
		
		# Compute similarity for this chunk
		with torch.no_grad():
			chunk_sim = torch.matmul(img_chunk, class_embeds.T) / temperature
		
		# Store result on CPU
		similarities[i:end_i] = chunk_sim.cpu()
		
		# Clean up GPU memory
		del img_chunk, chunk_sim
		torch.cuda.empty_cache()
	
	return similarities.to(device, non_blocking=True)

def _compute_image_embeddings_multilabel(
		model, 
		validation_loader, 
		device, 
		verbose,
		max_samples: int = None
	):
	all_image_embeds = []
	all_labels = []
	processed_samples = 0
	
	model.eval()
	iterator = tqdm(validation_loader, desc="Encoding images") if verbose else validation_loader
	
	for batch_idx, (images, _, labels) in enumerate(iterator):
		# Check if we've processed enough samples
		if max_samples and processed_samples >= max_samples:
			break
		
		# Adjust batch size if we're near the limit
		if max_samples and processed_samples + images.size(0) > max_samples:
			remaining = max_samples - processed_samples
			images = images[:remaining]
			labels = labels[:remaining]
		
		images = images.to(device, non_blocking=True)
		torch.cuda.empty_cache()
		
		try:
			with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
				with torch.no_grad():
					image_embeds = model.encode_image(images)
			image_embeds = F.normalize(image_embeds.float(), dim=-1)
			# Move to CPU immediately to free GPU memory
			all_image_embeds.append(image_embeds.cpu())
			all_labels.append(labels.cpu())
			processed_samples += images.size(0)			
			del images, image_embeds
			torch.cuda.empty_cache()
		except torch.cuda.OutOfMemoryError:
			print(f"OOM at batch {batch_idx}, reducing batch size...")
			batch_size = images.size(0)
			chunk_size = max(1, batch_size // 2)
			for i in range(0, batch_size, chunk_size):
				end_idx = min(i + chunk_size, batch_size)
				img_chunk = images[i:end_idx].to(device, non_blocking=True)
				label_chunk = labels[i:end_idx]				
				torch.cuda.empty_cache()
				with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
					with torch.no_grad():
						chunk_embeds = model.encode_image(img_chunk)
				chunk_embeds = F.normalize(chunk_embeds.float(), dim=-1)
				all_image_embeds.append(chunk_embeds.cpu())
				all_labels.append(label_chunk.cpu())
				processed_samples += img_chunk.size(0)
				del img_chunk, chunk_embeds
				torch.cuda.empty_cache()
			del images
			torch.cuda.empty_cache()	
	all_image_embeds = torch.cat(all_image_embeds, dim=0)
	all_labels = torch.cat(all_labels, dim=0)
	if verbose:
		print(f"Processed {processed_samples} samples, embedding shape: {all_image_embeds.shape}")
	return all_image_embeds.to(device, non_blocking=True), all_labels.to(device, non_blocking=True)

def pretrain_multilabel(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		device: torch.device,
		results_dir: str,
		cache_dir: str = None,
		topk_values: List[int] = [1, 3, 5],
		verbose: bool = True,
		max_samples: int = None,
		temperature: float = 0.07
	) -> Tuple[Dict, Dict]:
	model_name = model.__class__.__name__
	model_arch = re.sub(r"[/@]", "_", model.name)
	if cache_dir is None:
		cache_dir = results_dir
	
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name
	
	if verbose:
		print(f"Pretrain Multi-label Evaluation {dataset_name} {model_name} - {model_arch} {device}".center(170, "-"))

	# Get class information
	try:
		class_names = validation_loader.dataset.unique_labels
		num_classes = len(class_names)
	except AttributeError:
		class_names = validation_loader.dataset.dataset.classes
		num_classes = len(class_names)
	
	if verbose:
		print(f"Multi-label evaluation: {num_classes} classes")
	
	# Use the memory-efficient embedding computation
	all_image_embeds, all_labels = _compute_image_embeddings_multilabel(
		model,
		validation_loader,
		device,
		verbose,
		max_samples=max_samples,
	)
	
	# Pre-encode all class texts
	all_class_texts = clip.tokenize(class_names).to(device, non_blocking=True)
	with torch.no_grad():
		all_class_embeds = model.encode_text(all_class_texts)
		all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	
	# Clear cache before similarity computation
	torch.cuda.empty_cache()
	if verbose:
		print(f"Computing Image-to-Text similarities with temperature={temperature}")
	# Compute similarities in chunks
	i2t_similarities = _compute_similarities_chunked(
		all_image_embeds, 
		all_class_embeds, 
		chunk_size=100,
		device=device,
		temperature=temperature
	)
	if verbose:
		print(f"Computing Text-to-Image similarities with temperature={temperature}")
	t2i_similarities = _compute_similarities_chunked(
		all_class_embeds,
		all_image_embeds, 
		chunk_size=100,
		device=device,
		temperature=temperature
	)
	
	# Compute retrieval metrics compatible with plotting functions
	img2txt_metrics = _compute_multilabel_retrieval_metrics(
		similarity_matrix=i2t_similarities,
		query_labels=all_labels,
		candidate_labels=torch.arange(num_classes, device=device),
		topK_values=topk_values,
		mode="Image-to-Text",
		verbose=verbose
	)
	
	txt2img_metrics = _compute_multilabel_retrieval_metrics(
		similarity_matrix=t2i_similarities,
		query_labels=torch.arange(num_classes, device=device),
		candidate_labels=all_labels,
		topK_values=topk_values,
		mode="Text-to-Image",
		verbose=verbose
	)
	
	if verbose:
		print("Image to Text Metrics: ")
		print(json.dumps(img2txt_metrics, indent=2, ensure_ascii=False))
		print("Text to Image Metrics: ")
		print(json.dumps(txt2img_metrics, indent=2, ensure_ascii=False))

	# Create plot
	retrieval_metrics_best_model_fpth = os.path.join(
		results_dir, 
		f"{dataset_name}_pretrained_{model_name}_{model_arch}_retrieval_metrics_img2txt_txt2img.png"
	)
	
	# Import plotting function (assuming it exists)
	try:
		from visualize import plot_retrieval_metrics_best_model
		plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=img2txt_metrics,
			text_to_image_metrics=txt2img_metrics,
			fname=retrieval_metrics_best_model_fpth,
			best_model_name=f"Pretrained {model_name} {model_arch}",
		)
	except ImportError:
		print("Warning: Could not import plotting function")

	return img2txt_metrics, txt2img_metrics

def _compute_multilabel_retrieval_metrics(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		topK_values: List[int],
		mode: str = "Image-to-Text",
		verbose: bool = True
) -> Dict:
	"""
	Compute retrieval metrics for multi-label classification that are compatible 
	with the plotting functions (mP, mAP, Recall).
	"""
	if verbose:
		print(f"Computing retrieval metrics for {mode}")
	
	metrics = {"mP": {}, "mAP": {}, "Recall": {}}
	
	num_queries, num_candidates = similarity_matrix.shape
	device = similarity_matrix.device
	
	# Get top-K indices for all queries
	all_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
	
	for K in topK_values:
		if K > num_candidates:
			continue
			
		top_k_indices = all_sorted_indices[:, :K]
		
		# Compute correctness mask for multi-label
		if mode == "Image-to-Text":
			# query_labels: [num_images, num_classes], candidate_labels: [num_classes]
			correct_mask = _compute_multilabel_i2t_correctness(
				top_k_indices, query_labels, K
			)
		else:  # Text-to-Image
			# query_labels: [num_classes], candidate_labels: [num_images, num_classes]
			correct_mask = _compute_multilabel_t2i_correctness(
				top_k_indices, candidate_labels, K
			)
		
		# Compute metrics
		# mP: Mean Precision - average precision across all queries
		metrics["mP"][str(K)] = correct_mask.float().mean().item()
		
		# Recall: Fraction of queries that retrieved at least one relevant item
		metrics["Recall"][str(K)] = correct_mask.any(dim=1).float().mean().item()
		
		# mAP: Mean Average Precision
		ap_scores = []
		for i in range(num_queries):
			relevant_mask = correct_mask[i]
			if relevant_mask.any():
				# Calculate AP for this query
				relevant_positions = torch.where(relevant_mask)[0].float() + 1  # 1-indexed
				precisions = torch.arange(1, len(relevant_positions) + 1, device=device).float() / relevant_positions
				ap = precisions.mean().item()
			else:
				ap = 0.0
			ap_scores.append(ap)
		
		metrics["mAP"][str(K)] = np.mean(ap_scores)
	
	return metrics

def _compute_multilabel_i2t_correctness(
		top_k_indices: torch.Tensor,
		query_labels: torch.Tensor,
		K: int
	) -> torch.Tensor:
	num_images = top_k_indices.shape[0]
	device = top_k_indices.device
	correct_mask = torch.zeros(num_images, K, device=device, dtype=torch.bool)

	for i in range(num_images):
		true_class_indices = torch.where(query_labels[i] == 1)[0]

		if len(true_class_indices) > 0:
			retrieved_classes = top_k_indices[i]
			correct_retrievals = torch.isin(retrieved_classes, true_class_indices)
			correct_mask[i] = correct_retrievals

	return correct_mask

def _compute_multilabel_t2i_correctness(
		top_k_indices: torch.Tensor,
		candidate_labels: torch.Tensor,
		K: int
) -> torch.Tensor:
	"""
	Compute correctness mask for Text-to-Image multi-label retrieval.
	
	Args:
		top_k_indices: [num_classes, K] - top K image indices for each class
		candidate_labels: [num_images, num_classes] - multi-hot labels for each image
		K: number of top retrievals
		
	Returns:
		correct_mask: [num_classes, K] - binary mask indicating correct retrievals
	"""
	num_classes = top_k_indices.shape[0]
	device = top_k_indices.device
	
	correct_mask = torch.zeros(num_classes, K, device=device, dtype=torch.bool)
	
	for class_idx in range(num_classes):
		# Get images that have this class
		images_with_class = torch.where(candidate_labels[:, class_idx] == 1)[0]
		
		if len(images_with_class) > 0:
			# Check which of the top-K retrieved images actually have this class
			retrieved_images = top_k_indices[class_idx]  # [K]
			correct_retrievals = torch.isin(retrieved_images, images_with_class)
			correct_mask[class_idx] = correct_retrievals
	
	return correct_mask

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
	parser.add_argument('--temperature', '-t', type=float, default=0.07, help='Temperature for evaluation')

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

	print(f"Temperature used in training: 0.07")
	print(f"Temperature used in evaluation: {getattr(args, 'temperature', 'Not set')}")


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


	print("\n" + "="*80)
	print("DEBUGGING: Multi-label Ground Truth Extraction")
	print("="*80)
	
	# Check if ground truth extraction is correct
	for sample in validation_loader:
			images, _, labels = sample
			print(f"Batch size: {images.shape[0]}")
			print(f"Image shape: {images.shape}")
			print(f"Label shape: {labels.shape}")  # Should be [batch_size, num_classes]
			print(f"Labels dtype: {labels.dtype}")
			print(f"Sample labels for first image: {labels[0]}")  # Should show multiple 1s for multi-label
			print(f"Number of positive labels in first sample: {labels[0].sum().item()}")
			print(f"Non-zero label indices: {torch.where(labels[0] == 1)[0].tolist()}")
			
			# Also check a few more samples in the batch
			if images.shape[0] > 1:
					print(f"Sample labels for second image: {labels[1]}")
					print(f"Number of positive labels in second sample: {labels[1].sum().item()}")
			
			break  # Only check first batch
	
	print("="*80)
	print("END DEBUGGING")
	print("="*80 + "\n")


	customized_preprocess = get_preprocess(
		dataset_dir=args.dataset_dir, 
		input_resolution=model_config["image_resolution"],
	)

	print("\n" + "="*80)
	print("DEBUGGING: Text Embeddings")
	print("="*80)

	# Test text encoding with pretrained model
	test_labels = ['railroad', 'cannon', 'mountain']
	test_texts = clip.tokenize(test_labels).to(args.device)
	with torch.no_grad():
			text_embeds = pretrained_model.encode_text(test_texts)
			text_embeds = F.normalize(text_embeds, dim=-1)
			
	print(f"Text embeddings shape: {text_embeds.shape}")
	print(f"Text embeddings norm: {text_embeds.norm(dim=-1)}")  # Should be ~1.0
	print(f"Text similarity matrix:\n{torch.mm(text_embeds, text_embeds.T)}")  # Should show reasonable similarities

	print("="*80 + "\n")


	# Systematic selection of samples from validation set: Head, Torso, Tail
	if args.query_image is None or args.query_label is None:
		print("Selecting samples from validation set...")
		if args.dataset_type == "multi_label":
			i2t_samples, t2i_samples = get_multi_label_head_torso_tail_samples(
				metadata_path=os.path.join(args.dataset_dir, "metadata_multimodal.csv"),
				metadata_train_path=os.path.join(args.dataset_dir, "metadata_multimodal_train.csv"),
				metadata_val_path=os.path.join(args.dataset_dir, "metadata_multimodal_val.csv"),
				num_samples_per_segment=2,
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
				num_samples_per_segment=5,
			)
			if i2t_samples and t2i_samples:
				QUERY_IMAGES = [sample['image_path'] for sample in i2t_samples]
				QUERY_LABELS = [sample['label'] for sample in t2i_samples]
			else:
				raise ValueError("No single-label samples selected!")
	else:
		QUERY_IMAGES = [args.query_image]
		QUERY_LABELS = [args.query_label]

	print("QUERY IMAGES & LABELS".center(160, "-"))
	print(len(QUERY_IMAGES), QUERY_IMAGES)
	print(len(QUERY_LABELS), QUERY_LABELS)
	print("QUERY IMAGES & LABELS".center(160, "-"))

	# for all finetuned models(+ pre-trained):
	finetuned_checkpoint_paths = {
		"full": args.full_checkpoint,
		"lora": args.lora_checkpoint,
		"progressive": args.progressive_checkpoint,
	}
	print(f">> Loading {len(finetuned_checkpoint_paths)} Fine-tuned Models [takes a while]...")
	print(json.dumps(finetuned_checkpoint_paths, indent=4, ensure_ascii=False))
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
	print(f">> {len(fine_tuned_models)} Fine-tuned Models loaded in {time.time() - ft_start:.5f} sec")
	models_to_plot.update(fine_tuned_models)

	print("Computing Model Embeddings".center(160, "-"))
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
	print(f"Model Embeddings computed in {time.time() - mdl_emb_start:.5f} sec".center(160, " "))

	print(f"Evaluating {len(fine_tuned_models)} Fine-tuned Models".center(160, "-"))
	ft_eval_start = time.time()
	for ft_name, ft_path in finetuned_checkpoint_paths.items():
		if ft_name in fine_tuned_models:
			print(f"\n>> Fine-tuning strategy: {ft_name}")
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
				temperature=args.temperature,
			)
			finetuned_img2txt_dict[args.model_architecture][ft_name] = evaluation_results["img2txt_metrics"]
			finetuned_txt2img_dict[args.model_architecture][ft_name] = evaluation_results["txt2img_metrics"]
	print(f"{len(fine_tuned_models)} Fine-tuned Models evaluated in {time.time() - ft_eval_start:.5f} sec".center(160, "-"))

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
	if args.dataset_type == "multi_label":
		max_eval_samples = min(500, len(validation_loader.dataset))
		pretrained_img2txt, pretrained_txt2img = pretrain_multilabel(
			model=pretrained_model,
			validation_loader=validation_loader,
			device=args.device,
			results_dir=RESULT_DIRECTORY,
			cache_dir=CACHE_DIRECTORY,
			topk_values=args.topK_values,
			verbose=True,
			max_samples=max_eval_samples,
			temperature=args.temperature,
		)
		pretrained_img2txt_dict[args.model_architecture] = pretrained_img2txt
		pretrained_txt2img_dict[args.model_architecture] = pretrained_txt2img
	else:
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