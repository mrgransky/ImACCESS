from utils import *
from model import get_lora_clip, LAMB, SingleLabelLinearProbe, MultiLabelProbe, get_probe_clip
from early_stopper import EarlyStopping
from loss_analyzer import LossAnalyzer

from visualize import (
	plot_loss_accuracy_metrics, 
	plot_retrieval_metrics_best_model, 
	plot_retrieval_metrics_per_epoch, 
	plot_all_pretrain_metrics,
	plot_multilabel_loss_breakdown,
	collect_progressive_training_history,
	plot_phase_transition_analysis,
	plot_phase_transition_analysis_individual
)

if USER == "farid":
	from visualize import build_arch_flowchart

def compute_multilabel_validation_loss(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		temperature: float = 0.07,
		max_batches: int = None,
		all_class_embeds: torch.Tensor = None,
	) -> float:
	model.eval()
	total_loss = 0.0
	total_samples = 0
	
	# Get class embeddings
	if all_class_embeds is None:
		try:
			class_names = validation_loader.dataset.unique_labels
		except:
			try:
				class_names = validation_loader.dataset.dataset.classes
			except:
				raise ValueError("Could not extract class names from validation loader")
		
		all_class_texts = clip.tokenize(class_names).to(device, non_blocking=True)
		with torch.no_grad():
			all_class_embeds = model.encode_text(all_class_texts)
			all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	
	with torch.no_grad():
		for batch_idx, (images, _, label_vectors) in enumerate(validation_loader):
			if max_batches and batch_idx >= max_batches:
					break
			
			batch_size = images.size(0)
			if batch_size == 0:  # Skip empty batches
				continue
					
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Encode images
			image_embeds = model.encode_image(images)
			image_embeds = F.normalize(image_embeds, dim=-1)
			
			# Compute similarities
			i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
			t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature
			
			# Compute losses
			i2t_targets = label_vectors
			t2i_targets = label_vectors.T
			
			loss_i2t = criterion(i2t_similarities, i2t_targets)
			loss_t2i = criterion(t2i_similarities, t2i_targets)
			batch_loss = 0.5 * (loss_i2t + loss_t2i)
			
			# Correct accumulation
			total_loss += batch_loss.item() * batch_size
			total_samples += batch_size
	
	avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
	return avg_loss

def compute_multilabel_inbatch_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		topK_values: List[int],
		max_samples: int = 384,
		temperature: float = 0.07
	) -> Dict:
	model.eval()
	
	try:
		class_names = validation_loader.dataset.unique_labels
	except:
		try:
			class_names = validation_loader.dataset.dataset.classes
		except:
			raise ValueError("Could not extract class names from validation loader")
	
	all_class_texts = clip.tokenize(class_names).to(device)
	with torch.no_grad():
			all_class_embeds = model.encode_text(all_class_texts)
			all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	
	total_loss = 0.0
	processed_batches = 0
	total_samples = 0
	
	# Metrics storage
	img2txt_topk_hits = {k: 0 for k in topK_values}
	txt2img_topk_hits = {k: 0 for k in topK_values}
	img2txt_total_possible = {k: 0 for k in topK_values}
	txt2img_total_possible = {k: 0 for k in topK_values}
	
	cosine_similarities = list()
	
	with torch.no_grad():
		for batch_idx, (images, _, label_vectors) in enumerate(validation_loader):
			if total_samples >= max_samples:
				break
					
			batch_size = images.size(0)
			if total_samples + batch_size > max_samples:
				effective_batch_size = max_samples - total_samples
				images = images[:effective_batch_size]
				label_vectors = label_vectors[:effective_batch_size]
				batch_size = effective_batch_size
			
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Encode images
			image_embeds = model.encode_image(images)
			image_embeds = F.normalize(image_embeds, dim=-1)
			
			# Compute similarities
			i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
			
			# Compute validation loss
			i2t_targets = label_vectors.float()
			t2i_targets = label_vectors.T.float()
			t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature
			
			loss_i2t = criterion(i2t_similarities, i2t_targets)
			loss_t2i = criterion(t2i_similarities, t2i_targets)
			batch_loss = 0.5 * (loss_i2t + loss_t2i)
			total_loss += batch_loss.item()
			
			# ================================
			# Image-to-Text Top-K Accuracy
			# ================================
			for k in topK_values:
				if k > len(class_names):
					continue
						
				# Get top-k predicted classes for each image
				topk_indices = torch.topk(i2t_similarities, k=k, dim=1)[1]  # [batch_size, k]
				
				for img_idx in range(batch_size):
					# Get ground truth classes for this image
					true_classes = torch.where(label_vectors[img_idx] == 1)[0]
					
					if len(true_classes) > 0:
						# Get predicted classes
						pred_classes = topk_indices[img_idx]
						
						# Count how many predicted classes are correct
						hits = torch.isin(pred_classes, true_classes).sum().item()
						img2txt_topk_hits[k] += hits
						
						# Total possible hits is min(k, number_of_true_classes)
						img2txt_total_possible[k] += min(k, len(true_classes))
			
			# ================================
			# Text-to-Image Top-K Accuracy
			# ================================ 
			for k in topK_values:
				if k > batch_size:
					continue
						
				# For each class, get top-k similar images
				topk_indices = torch.topk(t2i_similarities, k=k, dim=1)[1]  # [num_classes, k]
				
				for class_idx in range(len(class_names)):
					# Find images that have this class
					images_with_class = torch.where(label_vectors[:, class_idx] == 1)[0]
					
					if len(images_with_class) > 0:
						# Get top-k retrieved images for this class
						retrieved_images = topk_indices[class_idx]
						
						# Count how many retrieved images actually have this class
						hits = torch.isin(retrieved_images, images_with_class).sum().item()
						txt2img_topk_hits[k] += hits
						
						# Total possible hits is min(k, number_of_images_with_class)
						txt2img_total_possible[k] += min(k, len(images_with_class))
			
			# Compute cosine similarities between matched pairs
			for img_idx in range(batch_size):
				true_classes = torch.where(label_vectors[img_idx] == 1)[0]
				if len(true_classes) > 0:
					# Average similarity to all true classes
					img_embed = image_embeds[img_idx]
					class_embeds = all_class_embeds[true_classes]
					similarities = F.cosine_similarity(img_embed.unsqueeze(0), class_embeds, dim=1)
					cosine_similarities.append(similarities.mean().item())
			
			processed_batches += 1
			total_samples += batch_size
	
	# Calculate final metrics
	avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
	
	img2txt_topk_acc = {}
	for k in topK_values:
		if img2txt_total_possible[k] > 0:
			img2txt_topk_acc[str(k)] = img2txt_topk_hits[k] / img2txt_total_possible[k]
		else:
			img2txt_topk_acc[str(k)] = 0.0
	
	txt2img_topk_acc = {}
	for k in topK_values:
		if txt2img_total_possible[k] > 0:
			txt2img_topk_acc[str(k)] = txt2img_topk_hits[k] / txt2img_total_possible[k]
		else:
			txt2img_topk_acc[str(k)] = 0.0
	
	# Overall accuracy (average across all K values)
	img2txt_acc = np.mean(list(img2txt_topk_acc.values())) if img2txt_topk_acc else 0.0
	txt2img_acc = np.mean(list(txt2img_topk_acc.values())) if txt2img_topk_acc else 0.0
	
	avg_cosine_sim = np.mean(cosine_similarities) if cosine_similarities else 0.0
	
	return {
		"val_loss": float(avg_loss),
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": img2txt_topk_acc,
		"txt2img_topk_acc": txt2img_topk_acc,
		"cosine_similarity": float(avg_cosine_sim)
	}

def compute_direct_in_batch_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		topK_values: List[int],
		max_samples: int = 384,
		temperature: float = 0.07,
	) -> Dict:

	model.eval()
	total_loss = 0.0
	processed_batches = 0
	total_samples = 0
	
	# Check if this is multi-label by inspecting the dataset
	sample_batch = next(iter(validation_loader))
	is_multilabel = len(sample_batch) == 3 and len(sample_batch[2].shape) == 2
	
	if is_multilabel:
		print("Multi-label dataset detected - skipping in-batch metrics computation")
		multi_label_in_batch_metrics = compute_multilabel_inbatch_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topK_values,
			max_samples=max_samples,
			temperature=temperature
		)
		return multi_label_in_batch_metrics

	print("Single-label dataset detected - computing in-batch metrics")
	total_loss = 0.0
	total_img2txt_correct = 0
	total_txt2img_correct = 0
	processed_batches = 0
	total_samples = 0
	cosine_similarities = list()
	
	# Get class information
	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	
	n_classes = len(class_names)
	valid_k_values = [k for k in topK_values if k <= n_classes]
	
	# Top-K accuracy tracking
	img2txt_topk_accuracy = {k: 0 for k in valid_k_values}
	txt2img_topk_accuracy = {k: 0 for k in topK_values}
	
	with torch.no_grad():
		for bidx, batch in enumerate(validation_loader):
			try:
				images, tokenized_labels, labels_indices = batch
				if total_samples >= max_samples:
					break
				
				batch_size = images.size(0)
				if total_samples + batch_size > max_samples:
					effective_batch_size = max_samples - total_samples
					images = images[:effective_batch_size]
					tokenized_labels = tokenized_labels[:effective_batch_size]
					labels_indices = labels_indices[:effective_batch_size]
					batch_size = effective_batch_size
				
				images = images.to(device, non_blocking=True)
				tokenized_labels = tokenized_labels.to(device, non_blocking=True)
				
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					logits_per_image, logits_per_text = model(images, tokenized_labels)
					
					# Check if logits match expected dimensions for single-label
					if logits_per_image.shape != (batch_size, batch_size):
						print(f"Warning: Unexpected logits shape: {logits_per_image.shape}")
						continue
					
					ground_truth = torch.arange(start=0, end=batch_size, device=device)
					
					# Compute loss
					if isinstance(criterion, torch.nn.CrossEntropyLoss):
						loss_img = criterion(logits_per_image, ground_truth)
						loss_txt = criterion(logits_per_text, ground_truth)
						batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
						total_loss += batch_loss
				
				# ================================
				# ACCURACY COMPUTATION
				# ================================
				
				# Image-to-Text accuracy (top-1)
				img2txt_preds = torch.argmax(logits_per_image, dim=1)
				img2txt_correct = (img2txt_preds == ground_truth).sum().item()
				total_img2txt_correct += img2txt_correct
				
				# Text-to-Image accuracy (top-1)
				txt2img_preds = torch.argmax(logits_per_text, dim=1)
				txt2img_correct = (txt2img_preds == ground_truth).sum().item()
				total_txt2img_correct += txt2img_correct
				
				# Top-K accuracy for Image-to-Text
				for k in valid_k_values:
					topk_preds = torch.topk(logits_per_image, k=min(k, batch_size), dim=1)[1]
					correct_topk = torch.isin(ground_truth.unsqueeze(1), topk_preds).any(dim=1).sum().item()
					img2txt_topk_accuracy[k] += correct_topk
				
				# Top-K accuracy for Text-to-Image
				for k in topK_values:
					if k <= batch_size:  # Can't have more K than batch size
						topk_preds = torch.topk(logits_per_text, k=min(k, batch_size), dim=1)[1]
						correct_topk = torch.isin(ground_truth.unsqueeze(1), topk_preds).any(dim=1).sum().item()
						txt2img_topk_accuracy[k] += correct_topk
				
				# Cosine similarity between matched pairs
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					image_embeds = model.encode_image(images)
					text_embeds = model.encode_text(tokenized_labels)
				
				image_embeds = F.normalize(image_embeds, dim=-1)
				text_embeds = F.normalize(text_embeds, dim=-1)
				
				# Cosine similarity (diagonal elements = matched pairs)
				cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().numpy()
				cosine_similarities.extend(cos_sim.tolist())
				
				processed_batches += 1
				total_samples += batch_size
					
			except Exception as e:
				print(f"Warning: Error processing batch {bidx}: {e}")
				continue
	
	# ================================
	# CALCULATE FINAL METRICS
	# ================================
	if total_samples == 0:
		print("Warning: No samples processed")
		return {
			"val_loss": 0.0,
			"img2txt_acc": 0.0,
			"txt2img_acc": 0.0,
			"img2txt_topk_acc": {str(k): 0.0 for k in valid_k_values},
			"txt2img_topk_acc": {str(k): 0.0 for k in topK_values},
			"cosine_similarity": 0.0
		}
	
	# Average loss
	avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
	
	# Top-1 accuracy
	img2txt_acc = total_img2txt_correct / total_samples
	txt2img_acc = total_txt2img_correct / total_samples
	
	# Top-K accuracy
	img2txt_topk_acc = {str(k): v / total_samples for k, v in img2txt_topk_accuracy.items()}
	txt2img_topk_acc = {str(k): v / total_samples for k, v in txt2img_topk_accuracy.items()}
	
	# Average cosine similarity
	avg_cos_sim = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
	
	return {
		"val_loss": float(avg_loss),
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": img2txt_topk_acc,
		"txt2img_topk_acc": txt2img_topk_acc,
		"cosine_similarity": float(avg_cos_sim)
	}

def get_optimal_threshold_multilabel(
		probs: torch.Tensor, 
		labels: torch.Tensor, 
		validation_split: float = 0.1
) -> float:
		"""
		Find optimal threshold for multi-label classification using validation split.
		
		Args:
				probs: Prediction probabilities [num_samples, num_classes]
				labels: Ground truth labels [num_samples, num_classes] 
				validation_split: Fraction of data to use for threshold selection
				
		Returns:
				Optimal threshold value
		"""
		num_samples = probs.shape[0]
		val_size = int(num_samples * validation_split)
		
		if val_size < 10:  # Too few samples for validation
				# Use label sparsity as fallback with better bounds
				sparsity = labels.float().mean().item()
				return max(0.1, min(0.9, sparsity))
		
		# Split data
		val_probs = probs[:val_size]
		val_labels = labels[:val_size]
		
		best_threshold = 0.5
		best_f1 = 0.0
		
		# Test different thresholds
		for threshold in torch.linspace(0.1, 0.9, 17):
				predictions = (val_probs > threshold).float()
				try:
						# Calculate F1 score (handles multi-label)
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
		
		return best_threshold

def chunked_similarity_computation(
		query_embeddings: torch.Tensor,
		candidate_embeddings: torch.Tensor,
		chunk_size: int = 1000,
		temperature: float = 0.07
	) -> torch.Tensor:
	num_queries = query_embeddings.shape[0]
	num_candidates = candidate_embeddings.shape[0]
	device = query_embeddings.device
	
	# Pre-allocate result tensor
	similarity_matrix = torch.zeros(
		num_queries, 
		num_candidates, 
		device=device, 
		dtype=torch.float32
	)
	
	for i in range(0, num_queries, chunk_size):
		end_i = min(i + chunk_size, num_queries)
		query_chunk = query_embeddings[i:end_i]
		
		# Compute similarity for this chunk
		chunk_similarities = torch.mm(query_chunk, candidate_embeddings.T) / temperature
		similarity_matrix[i:end_i] = chunk_similarities
	
	return similarity_matrix

def compute_multilabel_mrr(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		mode: str = "Image-to-Text"
) -> float:
		"""
		Compute Mean Reciprocal Rank for multi-label scenarios correctly.
		
		Args:
				similarity_matrix: [num_queries, num_candidates]
				query_labels: [num_queries, num_classes] for multi-label
				candidate_labels: [num_candidates, num_classes] or [num_classes] for classes
				mode: "Image-to-Text" or "Text-to-Image"
				
		Returns:
				Mean Reciprocal Rank score
		"""
		num_queries = similarity_matrix.shape[0]
		device = similarity_matrix.device
		
		# Get sorted indices (highest similarity first)
		sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
		
		reciprocal_ranks = list()
		
		if mode == "Image-to-Text":
				# For each image query, find ranks of relevant text classes
				for i in range(num_queries):
						true_class_indices = torch.where(query_labels[i] == 1)[0]
						if len(true_class_indices) == 0:
								continue
						
						# Find the rank of the highest-ranked relevant class
						ranks_of_relevant = list()
						for class_idx in true_class_indices:
								# Find where this class appears in the sorted results
								rank_positions = (sorted_indices[i] == class_idx).nonzero(as_tuple=True)[0]
								if len(rank_positions) > 0:
										ranks_of_relevant.append(rank_positions[0].item() + 1)  # 1-indexed
						
						if ranks_of_relevant:
								# Use the best (lowest) rank among all relevant classes
								best_rank = min(ranks_of_relevant)
								reciprocal_ranks.append(1.0 / best_rank)
		
		else:  # Text-to-Image
				# For each class query, find ranks of relevant images
				for i in range(num_queries):
						class_idx = i  # Assuming query i corresponds to class i
						
						# Find images that have this class
						if len(candidate_labels.shape) == 2:  # Multi-label candidates
								relevant_images = torch.where(candidate_labels[:, class_idx] == 1)[0]
						else:  # Single-label candidates  
								relevant_images = torch.where(candidate_labels == class_idx)[0]
						
						if len(relevant_images) == 0:
								continue
						
						# Find rank of highest-ranked relevant image
						ranks_of_relevant = list()
						for img_idx in relevant_images:
								rank_positions = (sorted_indices[i] == img_idx).nonzero(as_tuple=True)[0]
								if len(rank_positions) > 0:
										ranks_of_relevant.append(rank_positions[0].item() + 1)
						
						if ranks_of_relevant:
								best_rank = min(ranks_of_relevant)
								reciprocal_ranks.append(1.0 / best_rank)
		
		return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_retrieval_metrics_from_similarity(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		topK_values: List[int],
		mode: str = "Image-to-Text",
		class_counts: Optional[torch.Tensor] = None,
		max_k: Optional[int] = None,
		cache_dir: str = None,
		cache_key: str = None,
		is_training: bool = False,
		verbose: bool = False,
		chunk_size: int = 1000,
	) -> Dict:
	"""
	Compute retrieval metrics (mP, mAP, Recall) with memory optimization and proper multi-label support.
	
	Args:
			similarity_matrix: [num_queries, num_candidates]
			query_labels: [num_queries] for single-label or [num_queries, num_classes] for multi-label
			candidate_labels: [num_candidates] for single-label or [num_candidates, num_classes] for multi-label  
			topK_values: List of K values for evaluation
			mode: "Image-to-Text" or "Text-to-Image"
			class_counts: Number of samples per class (for single-label recall)
			max_k: Maximum K to consider
			cache_dir: Cache directory
			cache_key: Cache identifier
			is_training: Skip caching if True
			verbose: Print progress
			chunk_size: Chunk size for memory optimization
			
	Returns:
			Dictionary with mP, mAP, and Recall metrics
	"""
	if verbose:
		print(f"Computing retrieval metrics (mP, mAP, Recall) for {mode}")
	
	num_queries, num_candidates = similarity_matrix.shape
	device = similarity_matrix.device
	
	# Validate inputs
	if query_labels.dim() not in [1, 2] or candidate_labels.dim() not in [1, 2]:
		raise ValueError("Labels must be 1D (single-label) or 2D (multi-label)")
	
	# Determine if multi-label based on labels dimensionality
	is_multi_label = (
		len(candidate_labels.shape) == 2 if mode == "Text-to-Image" 
		else len(query_labels.shape) == 2
	)
	
	if verbose:
		print(
			f"{'Multi-label' if is_multi_label else 'Single-label'} Dataset |"
			f"Similarity matrix: {similarity_matrix.shape} | "
			f"Query labels: {query_labels.shape} | "
			f"Candidate labels: {candidate_labels.shape}"
		)
	
	# Check cache
	cache_file = None
	if cache_dir and cache_key and not is_training:
		cache_file = os.path.join(cache_dir, f"{cache_key}_retrieval_metrics.json")
		if os.path.exists(cache_file):
			try:
				if verbose:
					print(f"Loading cached metrics from {cache_file}")
				with open(cache_file, 'r') as f:
					return json.load(f)
			except Exception as e:
				if verbose:
					print(f"Cache loading failed: {e}. Computing metrics.")
	
	# Validate and filter K values
	valid_K_values = [K for K in topK_values if K <= (max_k or num_candidates)]
	if not valid_K_values:
		raise ValueError("No valid K values provided")
	
	# Get top-K indices for all queries (memory efficient)
	all_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
	
	metrics = {"mP": {}, "mAP": {}, "Recall": {}}
	
	for K in valid_K_values:
		top_k_indices = all_sorted_indices[:, :K]
		
		# Compute correctness mask based on dataset type
		if is_multi_label:
			correct_mask = compute_multilabel_correctness(
				top_k_indices, 
				query_labels, 
				candidate_labels, 
				mode, 
				K, 
				chunk_size,
			)
		else:
			correct_mask = compute_singlelabel_correctness(
				top_k_indices, 
				query_labels, 
				candidate_labels, 
				K,
			)
		
		# Compute metrics
		metrics["mP"][str(K)] = correct_mask.float().mean().item()
		
		# Compute Recall
		if mode == "Image-to-Text":
			metrics["Recall"][str(K)] = correct_mask.any(dim=1).float().mean().item()
		else:  # Text-to-Image
			if is_multi_label:
				# For multi-label: recall = retrieved_relevant / total_relevant
				num_classes = candidate_labels.shape[1]
				relevant_counts = torch.sum(candidate_labels, dim=0)  # [num_classes]
				
				total_recall = 0.0
				valid_classes = 0
				
				for class_idx in range(num_classes):
					if relevant_counts[class_idx] > 0:
						class_retrieved = correct_mask[class_idx].sum().item()
						class_recall = class_retrieved / relevant_counts[class_idx].item()
						total_recall += class_recall
						valid_classes += 1
				
				metrics["Recall"][str(K)] = total_recall / max(1, valid_classes)
			else:
				# Single-label: use class counts
				if class_counts is None:
					raise ValueError("class_counts required for single-label text-to-image")
				
				relevant_counts = class_counts[query_labels]
				recalled = correct_mask.sum(dim=1).float()
				metrics["Recall"][str(K)] = (recalled / relevant_counts.clamp(min=1)).mean().item()
		
		# Compute mAP (vectorized)
		positions = torch.arange(1, K + 1, device=device).float().unsqueeze(0)
		cumulative_correct = correct_mask.float().cumsum(dim=1)
		precisions = cumulative_correct / positions
		
		# AP = sum(precision * relevance) / num_relevant
		ap_scores = (precisions * correct_mask.float()).sum(dim=1) / correct_mask.sum(dim=1).clamp(min=1)
		metrics["mAP"][str(K)] = ap_scores.nanmean().item()
	
	# Save cache
	if cache_file and not is_training:
		try:
			os.makedirs(cache_dir, exist_ok=True)
			with open(cache_file, 'w') as f:
				json.dump(metrics, f)
			if verbose:
				print(f"Cached metrics to {cache_file}")
		except Exception as e:
			if verbose:
				print(f"Cache saving failed: {e}")
	
	return metrics

def compute_multilabel_correctness(
		top_k_indices: torch.Tensor,
		query_labels: torch.Tensor, 
		candidate_labels: torch.Tensor,
		mode: str,
		K: int,
		chunk_size: int = 1000
) -> torch.Tensor:
		"""
		Compute correctness mask for multi-label scenarios with memory optimization.
		"""
		num_queries = top_k_indices.shape[0]
		device = top_k_indices.device
		
		correct_mask = torch.zeros(num_queries, K, device=device, dtype=torch.bool)
		
		if mode == "Image-to-Text":
				# Process in chunks to save memory
				for i in range(0, num_queries, chunk_size):
						end_i = min(i + chunk_size, num_queries)
						chunk_indices = top_k_indices[i:end_i]  # [chunk_size, K]
						chunk_queries = query_labels[i:end_i]   # [chunk_size, num_classes]
						
						# For each query in chunk, check if retrieved classes are relevant
						for j, (retrieved_classes, true_classes) in enumerate(zip(chunk_indices, chunk_queries)):
								true_class_indices = torch.where(true_classes == 1)[0]
								# Check which retrieved classes are in true classes
								chunk_correct = torch.isin(retrieved_classes, true_class_indices)
								correct_mask[i + j] = chunk_correct
								
		else:  # Text-to-Image
				# For each class query, check if retrieved samples have the class
				for i in range(num_queries):
						class_idx = i  # Assuming query i corresponds to class i
						retrieved_samples = top_k_indices[i]  # [K]
						
						if class_idx < candidate_labels.shape[1]:
								# Check which retrieved samples have this class
								has_class = candidate_labels[retrieved_samples, class_idx]  # [K]
								correct_mask[i] = has_class
		
		return correct_mask

def compute_singlelabel_correctness(
		top_k_indices: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor, 
		K: int
) -> torch.Tensor:
		"""
		Compute correctness mask for single-label scenarios.
		"""
		# Get retrieved labels
		retrieved_labels = candidate_labels[top_k_indices]  # [num_queries, K]
		
		# Expand query labels to match
		query_labels_expanded = query_labels.unsqueeze(1).expand(-1, K)  # [num_queries, K]
		
		# Check correctness
		correct_mask = (retrieved_labels == query_labels_expanded)
		
		return correct_mask

@torch.no_grad()
def get_validation_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		topK_values: List[int],
		cache_dir: str,
		finetune_strategy: str = None,
		chunk_size: int = 1024,
		verbose: bool = True,
		max_in_batch_samples: Optional[int] = None,
		force_recompute: bool = False,
		embeddings_cache: tuple = None,
		lora_params: Optional[Dict] = None,
		is_training: bool = False,
		model_hash: str = None,
		temperature: float = 0.07,
	) -> Dict:

	if verbose:
		print("Computing validation metrics (in-batch, full-set, retrieval)...")
	
	model.eval()
	torch.cuda.empty_cache()
	start_time = time.time()
	
	if finetune_strategy is None:
		finetune_strategy = "pretrained"
	
	model_class_name = model.__class__.__name__
	model_arch_name = getattr(model, 'name', 'unknown_arch')
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	num_workers = getattr(validation_loader, 'num_workers', 'unknown_num_workers')
	
	try:
		class_names = validation_loader.dataset.dataset.classes
	except AttributeError:
		class_names = validation_loader.dataset.unique_labels
	
	n_classes = len(class_names)
	num_samples = len(validation_loader.dataset)
	
	# if verbose:
	# 	print(f"Dataset: {dataset_name}, Label(s): {n_classes}, Samples: {num_samples}")
	
	cache_file = os.path.join(
		cache_dir,
		f"{dataset_name}_{finetune_strategy}_bs_{validation_loader.batch_size}_"
		f"nw_{num_workers}_{model_class_name}_{model_arch_name.replace('/', '_')}_"
		f"validation_embeddings.pt"
	)
	if model_hash:
		cache_file = cache_file.replace(".pt", f"_{model_hash}.pt")
	
	# Step 1: Compute in-batch metrics if requested
	in_batch_metrics = None
	if max_in_batch_samples is not None:
		if verbose:
			print(f"Computing in-batch metrics with {max_in_batch_samples} samples...")
		in_batch_metrics = compute_direct_in_batch_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topK_values,
			max_samples=max_in_batch_samples,
			temperature=temperature,
		)
	
	# Step 2: Load or compute embeddings
	cache_loaded = False
	
	# Try to use provided cache first
	if not is_training and embeddings_cache is not None:
		all_image_embeds, _ = embeddings_cache
		all_labels = _prepare_labels_tensor(validation_loader, num_samples, n_classes, device)
		cache_loaded = True
		if verbose:
			print("Using provided embeddings cache")
	
	# Try to load from file cache
	elif not is_training and os.path.exists(cache_file) and not force_recompute:
		if verbose:
			print(f"Loading cached embeddings from {cache_file}")
		try:
			cached = torch.load(cache_file, map_location='cpu')
			all_image_embeds = cached['image_embeds']
			cached_labels = cached['labels']
			
			# Validate cache compatibility
			expected_labels = _prepare_labels_tensor(validation_loader, num_samples, n_classes, device)
			
			if _validate_cache_compatibility(cached_labels, expected_labels):
				all_labels = cached_labels.to(device)
				cache_loaded = True
				if verbose:
					print("Cache validation successful")
			else:
				if verbose:
					print("Cache incompatible, recomputing...")
				cache_loaded = False
						
		except Exception as e:
			if verbose:
				print(f"Cache loading failed: {e}. Recomputing...")
			cache_loaded = False
	
	if not cache_loaded:
		if verbose:
			print("Computing embeddings from scratch [takes a while]", end="...")
		t0 = time.time()
		all_image_embeds, all_labels = _compute_image_embeddings(
			model=model, 
			validation_loader=validation_loader, 
			device=device,
			# verbose=verbose, # too much verbosity
		)
		if verbose:
			print(f"Elapsed: {time.time() - t0:.1f} s")
		
		# Save to cache if not training
		if not is_training:
			try:
				os.makedirs(cache_dir, exist_ok=True)
				torch.save({
					'image_embeds': all_image_embeds.cpu(),
					'labels': all_labels.cpu()
				}, cache_file)
				if verbose:
					print(f"Saved embeddings to cache: {cache_file}")
			except Exception as e:
				if verbose:
					print(f"Cache saving failed: {e}")
	
	# Step 3: Compute text embeddings
	if verbose:
		print("Computing text embeddings...")
	
	text_inputs = clip.tokenize(class_names).to(device)
	with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
		class_text_embeds = model.encode_text(text_inputs)
	class_text_embeds = F.normalize(class_text_embeds.float(), dim=-1)
	
	# Move to device and ensure proper types
	device_image_embeds = all_image_embeds.to(device).float()
	device_class_text_embeds = class_text_embeds.to(device).float()
	device_labels = all_labels.to(device)
	
	# Step 4: Compute similarity matrices (chunked for memory efficiency)
	if verbose:
		print("Computing similarity matrices...")
	
	i2t_similarity = chunked_similarity_computation(
		device_image_embeds, 
		device_class_text_embeds, 
		chunk_size=chunk_size,
		temperature=temperature
	)
	t2i_similarity = chunked_similarity_computation(
		device_class_text_embeds, 
		device_image_embeds, 
		chunk_size=chunk_size,
		temperature=temperature
	)
	
	# Step 5: Compute full-set metrics
	if verbose:
		print("Computing full-set metrics...")
	
	full_metrics = compute_full_set_metrics_from_cache(
		i2t_similarity=i2t_similarity,
		t2i_similarity=t2i_similarity,
		labels=device_labels,
		n_classes=n_classes,
		topK_values=topK_values,
		device=device,
		device_image_embeds=device_image_embeds,
		device_class_text_embeds=device_class_text_embeds,
		chunk_size=chunk_size
	)
	
	# Step 6: Compute retrieval metrics
	if verbose:
		print("Computing retrieval metrics...")
	
	cache_key_base = f"{dataset_name}_{finetune_strategy}_{model_class_name}_{model_arch_name.replace('/', '_')}"
	if lora_params:
		cache_key_base += f"_lora_r{lora_params['lora_rank']}_a{lora_params['lora_alpha']}_d{lora_params['lora_dropout']}"
	
	img2txt_metrics = compute_retrieval_metrics_from_similarity(
		similarity_matrix=i2t_similarity,
		query_labels=device_labels,
		candidate_labels=torch.arange(n_classes, device=device),
		topK_values=topK_values,
		mode="Image-to-Text",
		cache_dir=cache_dir,
		cache_key=f"{cache_key_base}_img2txt",
		is_training=is_training,
		verbose=verbose,
		chunk_size=chunk_size,
	)
	
	# Prepare class counts for single-label datasets
	class_counts = None
	if len(device_labels.shape) == 1:  # Single-label
		class_counts = torch.bincount(device_labels.long(), minlength=n_classes)
	
	txt2img_metrics = compute_retrieval_metrics_from_similarity(
		similarity_matrix=t2i_similarity,
		query_labels=torch.arange(n_classes, device=device),
		candidate_labels=device_labels,
		topK_values=topK_values,
		mode="Text-to-Image",
		class_counts=class_counts,
		cache_dir=cache_dir,
		cache_key=f"{cache_key_base}_txt2img",
		is_training=is_training,
		verbose=verbose,
		chunk_size=chunk_size,
	)
	
	if verbose:
		print(f">> Validation completed in {time.time() - start_time:.2f}s")
	
	return {
		"in_batch_metrics": in_batch_metrics,
		"full_metrics": full_metrics,
		"img2txt_metrics": img2txt_metrics,
		"txt2img_metrics": txt2img_metrics
	}

def compute_full_set_metrics_from_cache(
		i2t_similarity: torch.Tensor,
		t2i_similarity: torch.Tensor,
		labels: torch.Tensor,
		n_classes: int,
		topK_values: List[int],
		device: str,
		device_image_embeds: torch.Tensor,
		device_class_text_embeds: torch.Tensor,
		chunk_size: int = 1000,
) -> Dict:
		"""
		Compute comprehensive full-set metrics with proper multi-label support and memory optimization.
		"""
		num_samples = i2t_similarity.shape[0]
		
		# Determine if multi-label
		is_multi_label = len(labels.shape) == 2 and labels.shape[1] == n_classes
		
		# Validate shapes
		if is_multi_label:
				assert labels.shape == (num_samples, n_classes), \
						f"Multi-label shape mismatch: {labels.shape} vs expected ({num_samples}, {n_classes})"
		else:
				assert labels.shape == (num_samples,), \
						f"Single-label shape mismatch: {labels.shape} vs expected ({num_samples},)"
				# Ensure proper integer type for indexing
				labels = labels.long()
		
		valid_k_values = [k for k in topK_values if k <= n_classes]
		
		# Compute Image-to-Text metrics
		if is_multi_label:
				img2txt_acc, img2txt_topk_acc = _compute_multilabel_i2t_accuracy(
						i2t_similarity, labels, valid_k_values, n_classes
				)
		else:
				img2txt_acc, img2txt_topk_acc = _compute_singlelabel_i2t_accuracy(
						i2t_similarity, labels, valid_k_values
				)
		
		# Compute Text-to-Image metrics  
		txt2img_topk_acc = _compute_t2i_accuracy(
				t2i_similarity, labels, topK_values, is_multi_label, num_samples, n_classes
		)
		txt2img_acc = txt2img_topk_acc.get(1, 0.0)
		
		# Compute MRR
		if is_multi_label:
				img2txt_mrr = compute_multilabel_mrr(
						i2t_similarity, labels, torch.arange(n_classes, device=device), "Image-to-Text"
				)
		else:
				img2txt_mrr = _compute_singlelabel_mrr(i2t_similarity, labels)
		
		# Compute cosine similarity between matched pairs
		cos_sim = _compute_matched_cosine_similarity(
				device_image_embeds, device_class_text_embeds, labels, is_multi_label
		)
		
		# Additional multi-label metrics
		hamming_loss = None
		partial_acc = None 
		f1_score_val = None
		
		if is_multi_label:
				# Get optimal threshold for predictions
				i2t_probs = torch.sigmoid(i2t_similarity)
				threshold = get_optimal_threshold_multilabel(i2t_probs, labels)
				
				# Make predictions
				i2t_preds = (i2t_probs > threshold).float()
				
				# Compute additional metrics
				hamming_loss = (i2t_preds != labels).float().mean().item()
				partial_acc = (i2t_preds == labels).float().mean().item()
				
				try:
						f1_score_val = f1_score(
								labels.cpu().numpy(), 
								i2t_preds.cpu().numpy(), 
								average='weighted',
								zero_division=0
						)
				except:
						f1_score_val = 0.0
		
		return {
				"img2txt_acc": float(img2txt_acc),
				"txt2img_acc": float(txt2img_acc), 
				"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
				"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
				"mean_reciprocal_rank": float(img2txt_mrr),
				"cosine_similarity": float(cos_sim),
				"hamming_loss": float(hamming_loss) if hamming_loss is not None else None,
				"partial_acc": float(partial_acc) if partial_acc is not None else None,
				"f1_score": float(f1_score_val) if f1_score_val is not None else None,
		}

def _prepare_labels_tensor(
		validation_loader: DataLoader, 
		num_samples: int, 
		n_classes: int, 
		device: str,
	) -> torch.Tensor:
	"""
	Prepare labels tensor for validation metrics computation.
	Handles both single-label and multi-label datasets.
	"""
	dataset = validation_loader.dataset
	
	# Detect dataset type by checking for multi-label specific attributes
	# is_multi_label = (
	# 	hasattr(dataset, 'label_dict') or 
	# 	hasattr(dataset, '_num_classes') or
	# 	'MultiLabel' in dataset.__class__.__name__
	# )
	is_multi_label = (
		(hasattr(dataset, 'label_dict') 
	 and dataset.label_dict is not None) 
	 or 'MultiLabel' in dataset.__class__.__name__
	) and not hasattr(dataset, 'labels_int')

	if is_multi_label:
		# Multi-label dataset - create label vectors [num_samples, num_classes]
		all_labels = torch.zeros(num_samples, n_classes, dtype=torch.float32)
		
		for i in range(num_samples):
			try:
				# Method 1: Use pre-computed label vectors from DataFrame
				if hasattr(dataset, 'data_frame') and 'label_vector' in dataset.data_frame.columns:
					label_vector = dataset.data_frame.iloc[i]['label_vector']
					if isinstance(label_vector, np.ndarray):
						all_labels[i] = torch.tensor(label_vector, dtype=torch.float32)
					elif isinstance(label_vector, torch.Tensor):
						all_labels[i] = label_vector.clone().detach().float()
					else:
						# Fallback to method 2
						raise ValueError("Invalid label_vector type")
				
				# Method 2: Get from dataset's __getitem__ method
				elif hasattr(dataset, '__getitem__'):
					try:
						_, _, label_vector = dataset[i]
						if isinstance(label_vector, torch.Tensor) and label_vector.shape == (n_classes,):
							all_labels[i] = label_vector.float()
						else:
							raise ValueError("Invalid label vector from __getitem__")
					except:
						# Fallback to method 3
						raise ValueError("Could not get label from __getitem__")
				
				# Method 3: Parse from string representation (fallback)
				else:
					if hasattr(dataset, 'labels') and hasattr(dataset, 'label_dict'):
						labels_str = dataset.labels[i]
						import ast
						labels = ast.literal_eval(labels_str)
						for label in labels:
							if label in dataset.label_dict:
								all_labels[i][dataset.label_dict[label]] = 1.0
					else:
						raise ValueError("Cannot extract labels from multi-label dataset")
						
			except Exception as e:
				print(f"Warning: Error processing sample {i}: {e}")
				# Leave as zeros for this sample
				continue
	
	else:
		# Single-label dataset - use integer labels [num_samples]
		if not hasattr(dataset, 'labels_int'):
			raise AttributeError(
				f"Single-label dataset {type(dataset)} missing 'labels_int' attribute. "
				"This attribute should contain integer class indices."
			)
		
		# Validate first sample to determine tensor type
		sample_label = dataset.labels_int[0]
		if isinstance(sample_label, (int, np.integer)):
			# Single-label: use long dtype for proper indexing
			all_labels = torch.zeros(num_samples, dtype=torch.long)
			for i in range(num_samples):
				all_labels[i] = dataset.labels_int[i]
		else:
			raise ValueError(f"Unexpected label type in single-label dataset: {type(sample_label)}")
	
	return all_labels.to(device)

def _validate_cache_compatibility(cached_labels: torch.Tensor, expected_labels: torch.Tensor) -> bool:
		"""Validate that cached labels are compatible with expected format."""
		if cached_labels.shape != expected_labels.shape:
				return False
		if cached_labels.dtype != expected_labels.dtype:
				return False
		return True

def _compute_image_embeddings(
		model: torch.nn.Module,
		validation_loader: DataLoader, 
		device: torch.device, 
		verbose: bool=False, 
		max_batches=None,
	):
	all_image_embeds, all_labels = list(), []
	model = model.to(device)
	model.eval()

	iterator = tqdm(validation_loader, desc="Encoding images") if verbose else validation_loader

	batch_count = 0
	with torch.no_grad():
		for images, _, labels_indices in iterator:
			if max_batches and batch_count >= max_batches:
				print(f"Stopping at batch {batch_count} due to max_batches limit")
				break

			if batch_count % 50 == 0:
				high_mem = monitor_memory_usage(operation_name=f"Batch {batch_count}")
				if high_mem:
					torch.cuda.empty_cache()

			images = images.to(device, non_blocking=True)
			if device.type == "cuda":
				images = images.half()

			with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
				image_embeds = model.encode_image(images)

			# Normalize and offload to CPU
			image_embeds = F.normalize(image_embeds, dim=-1).cpu()
			all_image_embeds.append(image_embeds)
			all_labels.append(labels_indices.cpu())

			# Explicit cleanup
			del images, image_embeds, labels_indices
			if batch_count % 100 == 0:
				torch.cuda.empty_cache()

			batch_count += 1

	if not all_image_embeds:
		raise RuntimeError("No image embeddings computed — possible failure in all batches.")

	all_image_embeds = torch.cat(all_image_embeds, dim=0)
	all_labels = torch.cat(all_labels, dim=0)

	return all_image_embeds, all_labels

def _compute_multilabel_i2t_accuracy(i2t_similarity, labels, valid_k_values, n_classes):
		"""Compute image-to-text accuracy for multi-label datasets."""
		# Use optimal thresholding
		i2t_probs = torch.sigmoid(i2t_similarity)
		threshold = get_optimal_threshold_multilabel(i2t_probs, labels)
		i2t_preds = (i2t_probs > threshold).float()
		
		# Exact match accuracy (all labels must match)
		img2txt_acc = (i2t_preds == labels).all(dim=1).float().mean().item()
		
		# Top-K accuracy
		img2txt_topk_acc = {}
		for k in valid_k_values:
				# For each sample, check if top-k predictions include all true labels
				topk_indices = i2t_similarity.topk(k, dim=1)[1]
				topk_preds = torch.zeros_like(i2t_probs).scatter_(1, topk_indices, 1.0)
				
				# Check if all true labels are in top-k (subset relation)
				correct = (labels <= topk_preds).all(dim=1).float().mean().item()
				img2txt_topk_acc[k] = correct
		
		return img2txt_acc, img2txt_topk_acc

def _compute_singlelabel_i2t_accuracy(i2t_similarity, labels, valid_k_values):
		"""Compute image-to-text accuracy for single-label datasets."""
		img2txt_preds = torch.argmax(i2t_similarity, dim=1)
		img2txt_acc = (img2txt_preds == labels).float().mean().item()
		
		img2txt_topk_acc = {}
		for k in valid_k_values:
				topk_indices = i2t_similarity.topk(k, dim=1)[1]
				correct = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
				img2txt_topk_acc[k] = correct
		
		return img2txt_acc, img2txt_topk_acc

def _compute_t2i_accuracy(t2i_similarity, labels, topK_values, is_multi_label, num_samples, n_classes):
		"""Compute text-to-image accuracy."""
		txt2img_topk_acc = {}
		
		for k in topK_values:
				effective_k = min(k, num_samples)
				topk_indices = t2i_similarity.topk(effective_k, dim=1)[1]
				
				class_correct = 0
				for class_idx in range(n_classes):
						retrieved_samples = topk_indices[class_idx]
						
						if is_multi_label:
								retrieved_labels = labels[retrieved_samples]
								if retrieved_labels[:, class_idx].any():
										class_correct += 1
						else:
								retrieved_labels = labels[retrieved_samples]
								if class_idx in retrieved_labels:
										class_correct += 1
				
				txt2img_topk_acc[k] = class_correct / n_classes
		
		return txt2img_topk_acc

def _compute_singlelabel_mrr(i2t_similarity, labels):
		"""Compute MRR for single-label datasets."""
		ranks = i2t_similarity.argsort(dim=-1, descending=True)
		rr_indices = ranks.eq(labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
		return (1.0 / rr_indices.float()).mean().item()

def _compute_matched_cosine_similarity(image_embeds, text_embeds, labels, is_multi_label):
		"""Compute cosine similarity between matched image-text pairs."""
		if is_multi_label:
				# For multi-label, average text embeddings of true classes
				matched_text_embeds = torch.zeros_like(image_embeds)
				for i in range(len(labels)):
						positive_indices = torch.where(labels[i] == 1)[0]
						if positive_indices.numel() > 0:
								matched_text_embeds[i] = text_embeds[positive_indices].mean(dim=0)
						else:
								matched_text_embeds[i] = text_embeds.mean(dim=0)
		else:
				# For single-label, direct indexing
				matched_text_embeds = text_embeds[labels]
		
		cos_sim = F.cosine_similarity(image_embeds, matched_text_embeds, dim=1)
		return cos_sim.mean().item()

def evaluate_best_model(
		model,
		validation_loader,
		criterion,
		early_stopping,
		checkpoint_path,
		finetune_strategy,
		device,
		cache_dir: str,
		topk_values: list[int] = [1, 5, 10],
		verbose: bool = True,
		clean_cache: bool = True,
		embeddings_cache=None,
		max_in_batch_samples: int = 384,
		lora_params = None,
		temperature: float = 0.07,
	):
	model_source = "current"
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	if verbose:
		print(f"Evaluating best model on {dataset_name} {finetune_strategy} criterion: {criterion.__class__.__name__}")

	if os.path.exists(checkpoint_path):
		if verbose:
			print(f"Loading best model weights {checkpoint_path} for final evaluation...")
		try:
			checkpoint = torch.load(checkpoint_path, map_location=device)
			if 'model_state_dict' in checkpoint:
				state_dict = checkpoint['model_state_dict']
				key_mappings = {
					'clip_model.': 'clip.',  # Map clip_model.* to clip.*
					'probe.clip_model.': 'clip.',  # Map probe.clip_model.* to clip.*
					'probe.probe.': 'probe.',  # Map probe.probe.* to probe.*
				}				
				translated_state_dict = translate_state_dict_keys(state_dict, key_mappings)
				try:
					model.load_state_dict(translated_state_dict, strict=False)
					best_epoch = checkpoint.get('epoch', 'unknown')
					if verbose:
						print(f"Loaded weights from checkpoint (epoch {best_epoch+1}): best_val_loss: {checkpoint.get('best_val_loss', 'unknown')}")
					model_source = "checkpoint"
				except Exception as e:
					if verbose:
						print(f"Translated state dict loading failed: {e}")
						print("Attempting flexible loading with strict=False...")
					# Fall back to partial loading
					missing_keys, unexpected_keys = model.load_state_dict(translated_state_dict, strict=False)
					if verbose and (missing_keys or unexpected_keys):
						print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
					model_source = "checkpoint_partial"		
			elif isinstance(checkpoint, dict) and 'epoch' not in checkpoint:
				# Handle direct state dictionary
				translated_state_dict = translate_state_dict_keys(checkpoint, key_mappings)
				model.load_state_dict(translated_state_dict, strict=False)
				if verbose:
					print("Loaded weights from direct state dictionary")
				model_source = "checkpoint"
			else:
				if verbose:
					print("Warning: Loaded file format not recognized as a model checkpoint.")
		except Exception as e:
			if verbose:
				print(f"<!> Error loading checkpoint:\n{e}")
				print("Proceeding with current model weights.")
	else:
		if verbose:
				print(f"Checkpoint not found at {checkpoint_path}. Proceeding with current model weights.")

	if model_source == "current" and early_stopping and early_stopping.restore_best_weights and early_stopping.best_weights is not None:
		try:
			if verbose:
				print(f"Loading weights from early stopping (epoch {early_stopping.best_epoch+1})")
			model.load_state_dict({k: v.to(device, non_blocking=True) for k, v in early_stopping.best_weights.items()})
			model_source = "early_stopping"
		except Exception as e:
			if verbose:
				print(f"Error loading weights from early stopping: {e}")
				print("Proceeding with current model weights.")

	param_count = sum(p.numel() for p in model.parameters())
	if verbose:
		print(f"Model ready for evaluation. Parameters: {param_count:,}")
	
	model.eval()
	
	if verbose:
		print("Performing final evaluation on the best model...")
	
	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		device=device,
		topK_values=topk_values,
		finetune_strategy=finetune_strategy,
		cache_dir=cache_dir,
		verbose=verbose,
		max_in_batch_samples=max_in_batch_samples,
		embeddings_cache=embeddings_cache,
		lora_params=lora_params,
		is_training=False,  # Use cache for final evaluation/inference
		model_hash=get_model_hash(model),
	)
	in_batch_metrics = validation_results["in_batch_metrics"]
	full_metrics = validation_results["full_metrics"]
	retrieval_metrics = {
		"img2txt": validation_results["img2txt_metrics"],
		"txt2img": validation_results["txt2img_metrics"]
	}
	
	if clean_cache:
		cleanup_embedding_cache(
			dataset_name=dataset_name,
			cache_dir=cache_dir,
			finetune_strategy=finetune_strategy,
			batch_size=validation_loader.batch_size,
			num_workers=validation_loader.num_workers,
			model_name=model.__class__.__name__,
			model_arch=model.name if hasattr(model, 'name') else 'unknown_arch'
		)
	return {
		"in_batch_metrics": in_batch_metrics,
		"full_metrics": full_metrics,
		"img2txt_metrics": retrieval_metrics["img2txt"],
		"txt2img_metrics": retrieval_metrics["txt2img"],
		"model_loaded_from": model_source
	}

def get_status(
		model,
		phase,
		layers_to_unfreeze,
		cache=None,
	):
	# Compute parameter statistics
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	total_params = trainable_params + frozen_params

	# Count unique layers based on group membership
	layer_groups = get_layer_groups(model)
	all_layers = set()
	for group, layers in layer_groups.items():
		for layer in layers:
			all_layers.add(layer)  # Use exact layer names from groups
	total_layers = len(all_layers)

	# Count unique frozen layers
	frozen_layers = 0
	unfrozen_layers = set(layers_to_unfreeze)  # Layers to unfreeze in this phase
	for layer in all_layers:
		# Check if any parameter in this layer is frozen
		is_frozen = all(not p.requires_grad for name, p in model.named_parameters() if layer in name)
		if is_frozen and layer not in unfrozen_layers:
			frozen_layers += 1

	# Update category breakdown
	category_breakdown = {}
	for group, layers in layer_groups.items():
		frozen_in_group = sum(1 for layer in layers if all(not p.requires_grad for name, p in model.named_parameters() if layer in name) and layer not in unfrozen_layers)
		total_in_group = len(layers)
		category_breakdown[group] = (frozen_in_group, total_in_group)

	# Cache results if provided
	if cache is not None:
		cache[f"phase_{phase}"] = {"trainable": trainable_params, "frozen": frozen_params}

	# Print detailed status using tabulate
	headers = ["Metric", "Value"]
	param_stats = [
		["Phase #", f"{phase}"],
		["Total Parameters", f"{total_params:,}"],
		["Trainable Parameters", f"{trainable_params:,} ({trainable_params/total_params*100:.2f}%)"],
		["Frozen Parameters", f"{frozen_params:,} ({frozen_params/total_params*100:.2f}%)"]
	]
	layer_stats = [
		["Total Layers", total_layers],
		["Frozen Layers", f"{frozen_layers} ({frozen_layers/total_layers*100:.2f}%)"]
	]
	category_stats = [[group, f"{frozen}/{total} ({frozen/total*100:.2f}%)"] for group, (frozen, total) in category_breakdown.items()]

	print(tabulate.tabulate(param_stats, headers=headers, tablefmt="pretty", colalign=("left", "left")))
	print("\nLayer Statistics:")
	print(tabulate.tabulate(layer_stats, headers=headers, tablefmt="pretty", colalign=("left", "left")))
	print("\nLayer Category Breakdown:")
	print(tabulate.tabulate(category_stats, headers=["Category", "Frozen/Total (Percentage)"], tablefmt="pretty", colalign=("left", "left")))

def get_num_transformer_blocks(
		model: torch.nn.Module
	) -> tuple:
	if not hasattr(model, 'visual'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'visual' attribute.")

	if not hasattr(model, 'transformer'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'transformer' attribute.")

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Count visual transformer blocks
	visual_blocks = 0
	if is_vit:
		if not hasattr(model.visual, 'transformer') or not hasattr(model.visual.transformer, 'resblocks'):
			raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'visual.transformer.resblocks' attribute.")
		visual_blocks = len(model.visual.transformer.resblocks)
	elif is_resnet:
		# ResNet models use 'layer1', 'layer2', etc.
		visual_layers = [attr for attr in dir(model.visual) if attr.startswith('layer') and attr[5:].isdigit()]
		visual_blocks = len(visual_layers)
		if visual_blocks == 0:
			print(f"Model {model.name} is a ResNet but no 'visual.layerX' blocks found. Visual blocks set to 0.")
	else:
		raise ValueError(f"Unsupported architecture {model.name}. Expected ViT or ResNet.")

	# Count text transformer blocks
	text_blocks = 0
	if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
		text_blocks = len(model.transformer.resblocks)
	else:
		print(f"Model {model.name} lacks 'transformer.resblocks'. Text blocks set to 0.")

	# print(f">> {model.__class__.__name__} {model.name}: Visual Transformer blocks: {visual_blocks}, Text Transformer blocks: {text_blocks}")
	return visual_blocks, text_blocks

def get_layer_groups(
		model: torch.nn.Module
	) -> dict:
	vis_nblocks, txt_nblocks = get_num_transformer_blocks(model=model)

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Visual transformer or CNN blocks
	visual_blocks = list()
	if is_vit and vis_nblocks > 0:
		visual_blocks = [f'visual.transformer.resblocks.{i}' for i in range(vis_nblocks)]
	elif is_resnet and vis_nblocks > 0:
		visual_blocks = [f'visual.layer{i+1}' for i in range(vis_nblocks)]
	else:
		print(f"No visual blocks defined for model {model.name}")

	# Text transformer blocks
	text_blocks = [f'transformer.resblocks.{i}' for i in range(txt_nblocks)] if txt_nblocks > 0 else []
	if txt_nblocks == 0:
		print(f"No text transformer blocks defined for model {model.name}")

	"""
		ViT architecture (patch embedding → transformer blocks → projection)
		- Frontend (Lower Layers): initial layers responsible for converting raw inputs (images or text) into a format suitable for transformer blocks.
		- Transformer Blocks (Intermediate Layers): core layers that perform feature extraction and contextualization (self-attention mechanisms).
		- Projection (Output Layers): final layers that map the transformer outputs to the shared embedding space and compute similarity scores.
	"""

	def hasattr_nested(obj, attr_string):
		attrs = attr_string.split('.')
		for attr in attrs:
			if hasattr(obj, attr):
				obj = getattr(obj, attr)
			else:
				return False
		return True

	layer_groups = {
		'visual_frontend': [
			'visual.conv1', # patch embedding (ViT) or first conv layer (ResNet)
			'visual.class_embedding' if is_vit else 'visual.bn1', # CLS token for ViT, bn1 for ResNet
			'visual.positional_embedding' if is_vit else 'visual.relu', # positional embedding for ViT, relu for ResNet
		],
		'visual_transformer': visual_blocks,
		'text_frontend': [ # Converts tokenized text into embeddings (token_embedding) then adds positional information (positional_embedding).
			'token_embedding',
			'positional_embedding',
		],
		'text_transformer': text_blocks,
		'projections': [], # Start with an empty list
		# 'projections': [
		# 	'visual.proj', # Projects visual transformer’s output (e.g., the CLS token embedding) into the shared space.
		# 	'visual.ln_post' if is_vit else 'visual.attnpool',  # ln_post for ViT, attnpool for ResNet
		# 	'text_projection', # Projects the text transformer’s output into the shared space.
		# 	'logit_scale', # learnable scalar that scales the cosine similarities between image and text embeddings during contrastive loss computation.
		# ],
	}

	# Conditionally add projection layers only if they
	# exist in the model and are supported by the architecture
	projection_candidates = {
		'visual.proj': True, # Always check for this
		'visual.ln_post': is_vit,
		'visual.attnpool': is_resnet,
		'text_projection': True,
		'logit_scale': True,
	}

	for layer_name, should_check in projection_candidates.items():
		if should_check and hasattr_nested(model, layer_name):
			layer_groups['projections'].append(layer_name)

	# Also filter frontend layers that might not exist in all models
	layer_groups['visual_frontend'] = [
		name 
		for name in layer_groups['visual_frontend'] 
		if hasattr_nested(model, name)
	]

	layer_groups['text_frontend'] = [
		name 
		for name in layer_groups['text_frontend'] 
		if hasattr_nested(model, name)
	]

	return layer_groups

def unfreeze_layers(
		model: torch.nn.Module,
		strategy: Dict[int, List[str]],
		phase: int,
		cache: Dict[int, List[str]],
	):
	print(f"Applying unfreeze strategy for Phase {phase}...")
	
	# 1. Get the layers to unfreeze at this phase
	layers_to_unfreeze = strategy[phase]

	# 2. Unfreeze the layers
	# Assumes layer names in layers_to_unfreeze are prefixes of parameter names
	# (e.g., 'visual.transformer.resblocks.0' matches 'visual.transformer.resblocks.0.attn.in_proj_weight')
	for name, param in model.named_parameters():
		param.requires_grad = False # Freeze all layers first
		if any(ly in name for ly in layers_to_unfreeze): # Unfreeze layers in the list
			param.requires_grad = True

	# 3. Cache the frozen layers
	get_status(
		model=model,
		phase=phase,
		layers_to_unfreeze=layers_to_unfreeze,
		cache=cache,
	)

def get_unfreeze_schedule(
		model: torch.nn.Module,
		num_phases: int,
		layer_groups_to_unfreeze: List[str]=['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
	) -> Dict[int, List[str]]:

	unfreeze_fractions = np.linspace(0, 1, num_phases).tolist()
	print(f"Getting unfreeze schedule for {model.name} for {len(unfreeze_fractions)} phases".center(120, "-"))
	print(f"Layer groups to unfreeze: {layer_groups_to_unfreeze}")
	print(f"Unfreeze fractions: {unfreeze_fractions}")
	
	
	# Validate input
	if not all(0.0 <= p <= 1.0 for p in unfreeze_fractions):
		raise ValueError("Unfreeze fractions must be between 0.0 and 1.0.")

	if not all(g in ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'] for g in layer_groups_to_unfreeze):
		raise ValueError("Invalid layer group specified. Accepted: visual_frontend, visual_transformer, text_frontend, text_transformer, projections.")

	def create_layer_table(num_layers: int, layer_type: str) -> str:
		table_data = list()
		for i, pct in enumerate(unfreeze_fractions):
			label = f"{int(pct * 100)}%" if pct != 0.0 and pct != 1.0 else ("None" if pct == 0.0 else "All")
			table_data.append([
				i,
				label,
				f"{int(pct * num_layers)}/{num_layers}",
				f"{(pct * 100):.0f}%"
			])

		return (
			f"\n{layer_type} Transformer Layer Unfreezing Schedule:\n"
			+ tabulate.tabulate(
				table_data,
				headers=["#", "Phase Type", "Unfrozen Layers", "Unfrozen Percentage(%)"],
				tablefmt="grid"
			)
		)

	layer_groups = get_layer_groups(model=model)
	print(f"\nLayer groups:\n{json.dumps(layer_groups, indent=2)}")

	# Explicitly get all layer groups from the model definition
	projections = layer_groups.get('projections', [])
	visual_transformer = layer_groups.get('visual_transformer', [])
	text_transformer = layer_groups.get('text_transformer', [])
	visual_frontend = layer_groups.get('visual_frontend', [])
	text_frontend = layer_groups.get('text_frontend', [])
	
	total_v_layers = len(visual_transformer)
	total_t_layers = len(text_transformer)
	print(f"Total visual transformer layers: {total_v_layers}")
	print(f"Total text transformer layers: {total_t_layers}")

	if total_v_layers == 0 and total_t_layers == 0:
		raise ValueError("No transformer blocks found in visual or text encoders. Cannot create unfreezing schedule.")

	schedule = {}
	for phase, unfreeze_pct in enumerate(unfreeze_fractions):
		# Start with an empty list for this phase's layers
		layers_to_unfreeze_for_phase = list()
		
		# 1. ALWAYS include projections if they are in the target groups
		if 'projections' in layer_groups_to_unfreeze:
			layers_to_unfreeze_for_phase.extend(projections)
			
		# 2. Add transformer layers based on percentage
		if 'visual_transformer' in layer_groups_to_unfreeze:
			v_unfreeze_count = int(unfreeze_pct * total_v_layers)
			if v_unfreeze_count > 0:
				layers_to_unfreeze_for_phase.extend(visual_transformer[-v_unfreeze_count:])

		if 'text_transformer' in layer_groups_to_unfreeze:
			t_unfreeze_count = int(unfreeze_pct * total_t_layers)
			if t_unfreeze_count > 0:
				layers_to_unfreeze_for_phase.extend(text_transformer[-t_unfreeze_count:])
				
		# 3. Add frontend layers ONLY at the final phase (100%)
		if unfreeze_pct == 1.0:
			if 'visual_frontend' in layer_groups_to_unfreeze:
				layers_to_unfreeze_for_phase.extend(visual_frontend)
			if 'text_frontend' in layer_groups_to_unfreeze:
				layers_to_unfreeze_for_phase.extend(text_frontend)
		
		# Use a set to remove duplicates (e.g., if a layer name is in multiple groups)
		# and convert back to a sorted list for consistent ordering.
		schedule[phase] = sorted(list(set(layers_to_unfreeze_for_phase)))

		# --- Enhanced Logging for verification ---
		print(f"Phase {phase} (unfreeze_pct: {unfreeze_pct*100:.2f}%): {len(schedule[phase])} layers to unfreeze:")
		for i, layer in enumerate(schedule[phase]):
			print(f"\t{i:02d} {layer}")

	print(create_layer_table(total_v_layers, "Visual"))
	print(create_layer_table(total_t_layers, "Text"))

	print("-"*120)
	return schedule

def compute_embedding_drift(
		model: torch.nn.Module, 
		val_subset: torch.utils.data.DataLoader, 
		pretrained_embeds: torch.Tensor, 
		device: torch.device, 
		phase: int, 
		epoch: int,
	):
	"""
	Embedding Drift = 1 - cosine_similarity, 
	measures how far the current image embeddings 
	moved from their original, pre-trained positions. 
	0.0: no change, 
	1.0: they are now orthogonal (completely different).
	
	In summary, the ideal Embedding Drift curve:
		Starts near zero.
		Shows small, controlled increases during early-to-mid phases 
		that are inversely correlated with validation loss (drift goes up, loss goes down).
		Plateaus in later phases, indicating that the foundational knowledge is being preserved.
	"""
	model.eval()
	with torch.no_grad():
		imgs = next(iter(val_subset)) # torch.Size([batch_size, channels, height, width ])
		imgs = imgs.to(device)
		new_embeds = model.encode_image(imgs)
		new_embeds = F.normalize(new_embeds, dim=-1)
		drift = F.cosine_similarity(new_embeds, pretrained_embeds[:new_embeds.size(0)].to(device), dim=-1)
		mean_drift = 1 - drift.mean().item()
	print(f"[DEBUG] Embedding Drift | Phase {phase} | Epoch {epoch}: {mean_drift}")
	return mean_drift

def create_differential_optimizer_groups(
		model: torch.nn.Module,
		base_lr: float,
		base_wd: float,
		optimizer_hyperparams: dict,
		lr_multipliers: dict=None,
	) -> List[Dict]:

	if lr_multipliers is None:
		lr_multipliers = {
			'projections': 0.1,					# orig: 0.5 # Already trained - conservative
			'text_transformer': 1.0,		# orig: 0.1 # New layers - full rate
			'visual_transformer': 1.0,	# orig: 0.1 # New layers - full rate
			'text_frontend': 0.01,
			'visual_frontend': 0.01,
		}

	print("\n>> Creating Optimizer Groups with Differential LRs")
	print(f"Base LR: {base_lr:<20}Base WD: {base_wd}")
	# print(f"LR Multipliers:\n{json.dumps(lr_multipliers, indent=2)}")

	adamw_defaults = {
		'lr': base_lr,
		'weight_decay': base_wd,
		'eps': optimizer_hyperparams.get('eps', 1e-8),  # Adjusted default to match AdamW
		'betas': optimizer_hyperparams.get('betas', (0.9, 0.999)), # Adjusted default to match AdamW
		'amsgrad': False,
		'maximize': False,
		'foreach': None,
		'capturable': False,
		'differentiable': False,
		'fused': None,
		'decoupled_weight_decay': True,
	}
	
	layer_groups_map = get_layer_groups(model)	

	assigned_params = set()
	param_groups = list()

	for group_name, layer_prefixes in layer_groups_map.items():
		group_params_list = list()
		for prefix in layer_prefixes:
			for name, param in model.named_parameters():
				if name.startswith(prefix) and param.requires_grad and param not in assigned_params:
					group_params_list.append(param)
					assigned_params.add(param)
		
		if group_params_list:
			lr_multiplier = lr_multipliers.get(group_name, 0.1)
			
			group_dict = adamw_defaults.copy()
			group_dict['params'] = group_params_list
			group_dict['lr'] = base_lr * lr_multiplier
			param_groups.append(group_dict)
			
			print(
				f"{group_name:<22}"
				f"Parameters: {len(group_params_list):<5}"
				f"LR Multiplier: {lr_multiplier:<10}"
				f"Final LR: {group_dict['lr']}"
			)

	remaining_params = [
		p 
		for p in model.parameters() 
		if p.requires_grad and p not in assigned_params
	]

	if remaining_params:
		print("  - Group: 'remaining_params' (unclassified)")
		group_dict = adamw_defaults.copy()
		group_dict['params'] = remaining_params
		group_dict['lr'] = base_lr * 0.01
		param_groups.append(group_dict)

	return param_groups

def should_transition_to_next_phase(
		current_phase: int,
		losses: List[float],
		window: int,
		epochs_in_phase: int,
		min_epochs_per_phase: int,
		num_phases: int,
		best_loss: Optional[float],
		best_loss_threshold: float,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		accuracies: Optional[List[float]]=None, # Added optional accuracy list
		accuracy_plateau_threshold: float=1e-3, # Threshold for accuracy stagnation
		plateau_threshold: float = 1e-2,				# 1% improvement threshold
	) -> bool:
	
	# Must meet minimum epochs requirement
	if epochs_in_phase < min_epochs_per_phase:
		return False

	# Don't transition from final phase
	if current_phase >= num_phases - 1:
		return False

	# Need enough history to detect trends
	if len(losses) < window:
		print(f"<!> Insufficient loss data ({len(losses)} < {window}) for phase transition.")
		return False
	
	# Simple plateau detection: compare recent average to older average
	recent_window = losses[-window:]
	older_window = losses[-(window*2):-window] if len(losses) >= window*2 else losses[:-window]
	
	if not older_window:  # Fallback if not enough history
		older_window = losses[:len(losses)//2]
		recent_window = losses[len(losses)//2:]
	
	recent_avg = np.mean(recent_window)
	older_avg = np.mean(older_window)
	
	improvement = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
	is_plateau = improvement < plateau_threshold

	# Simple volatility check - high volatility suggests instability (overshooting with spiking)
	recent_std = np.std(recent_window)
	recent_cv = (recent_std / abs(recent_avg)) * 100 if recent_avg != 0 else 0
	high_volatility = recent_cv > volatility_threshold  # X% coefficient of variation
	
	print(f"\n=== SIMPLE TRANSITION CHECK (Phase {current_phase}) ===")
	print(f"{len(losses)} losses: {losses}")
	print(f"\t>> min: {min(losses)} max: {max(losses)} range: {max(losses) - min(losses)} std: {np.std(losses)}")
	print(f"Recent window ({len(recent_window)} epochs, requested {window}): {recent_window}")
	print(f"Older average: {older_avg} → Recent average: {recent_avg}")
	print(f"Improvement: {improvement} (threshold: {plateau_threshold})")
	print(f"Volatility: {recent_cv:.2f}% (high if > {volatility_threshold}%)")
	print(f"Plateau: {is_plateau}")

	# Transition logic
	should_transition = False
	reasons = list()
	if is_plateau and not high_volatility:
		should_transition = True
		reasons.append(f"Meaningful plateau detected ({improvement} < {plateau_threshold})")
	elif high_volatility:
		reasons.append(f"High volatility ({recent_cv:.2f}%) - continuing current phase for stability")
	else:
		reasons.append(f"Still learning ({improvement} improvement > threshold: {plateau_threshold})")
	
	if should_transition:
		print(f"\t>>> TRANSITION RECOMMENDED: {', '.join(reasons)}")
	else:
		print(f"\t>>> CONTINUE CURRENT PHASE: {', '.join(reasons)}")
	
	print("=" * 50)
	return should_transition

def handle_phase_transition(
		current_phase: int, 
		optimizer: torch.optim.Optimizer,
	) -> Tuple[int, float, float]:
	next_phase = current_phase + 1	
	new_lr = optimizer.param_groups[0]['lr'] * 0.8  # 20% reduction
	new_wd = optimizer.param_groups[0]['weight_decay'] * 1.1  # 10% increase
	return next_phase, new_lr, new_wd

def progressive_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		patience: int,															# patience for epochs without improvement
		min_delta: float,														# Make slightly less sensitive than default
		cumulative_delta: float,										# Keep cumulative check reasonable
		minimum_epochs: int,												# Minimum epochs before ANY early stop
		min_epochs_per_phase: int,									# Minimum epochs within a phase before transition check
		volatility_threshold: float,								# Allow slightly more volatility
		slope_threshold: float, 										# Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float,							# Stricter requirement for pairwise improvement
		min_phases_before_stopping: int,						# Ensure significant unfreezing before global stop
		accuracy_plateau_threshold: float = 5e-4,		# For phase transition based on accuracy
		topk_values: list[int]=None,
		layer_groups_to_unfreeze: list[str]=None,
		use_lamb: bool=False,
		verbose: bool=False,
	):
	if layer_groups_to_unfreeze is None:
		layer_groups_to_unfreeze = ['visual_transformer', 'text_transformer', 'projections'] # key layers
	if topk_values is None:
		topk_values = [1, 5, 10, 15, 20]

	initial_learning_rate = learning_rate
	initial_weight_decay = weight_decay
	window_size = min_epochs_per_phase + 5
	total_num_phases = min_phases_before_stopping + 5

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		min_phases_before_stopping=min_phases_before_stopping,
	)
	early_stopping_triggered = False

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{CLUSTER} | {HOST} | {gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	if USER == "farid":
		build_arch_flowchart(
			model,
			filename=os.path.join(results_dir, f"{model_arch}_flowchart"),
			format="png",      # you can also use "svg" for a vector image
			view=False,         # opens the image automatically (optional)
			rankdir="LR"       # top‑to‑bottom (feel free to try "LR")
		)

	if verbose:
		for n, m in model.named_modules():
			print(f"{n:<60} {type(m).__name__:<50} Training: {m.training:<10} Weights Grad: {m.weight.requires_grad if hasattr(m, 'weight') else ''}")
		print("-"*100)

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		print(f"Non-zero dropout detected in {model_name} {model_arch}:")
		for i, v in enumerate(non_zero_dropouts):
			print(f"{i:02d} {v[0]:<60}p={v[1]}")
		print("-"*100)
	else:
		print(f"No non-zero dropout detected in {model_name} {model_arch}")

	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		num_phases=total_num_phases,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	layer_cache = {} # Cache for layer status (optional, used by get_status)

	# unfreeze layers for Phase 0 to correctly initialize the optimizer
	unfreeze_layers(
		model=model,
		strategy=unfreeze_schedule,
		phase=0,
		cache=layer_cache,
	)
	
	optimizer = None
	scheduler = None

	criterion = torch.nn.CrossEntropyLoss()
	print(f"Using {criterion.__class__.__name__} as the loss function")

	scaler = torch.amp.GradScaler(device=device) # For mixed precision
	print(f"Using {scaler.__class__.__name__} for mixed precision training")

	mdl_fpth = os.path.join(
		results_dir,
		# f"{dataset_name}_"
		f"{mode}_"
		f"{model_arch}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mphb4stp_{min_phases_before_stopping}"
		f".pth"
	)

	# Embedding drift: get a fixed batch of validation data and original embeddings
	val_subset_loader = DataLoader(
		validation_loader.dataset, 
		batch_size=validation_loader.batch_size,
		shuffle=False
	)

	fixed_val_batch = next(iter(val_subset_loader))

	with torch.no_grad():
		model.eval()
		initial_images, _, _ = fixed_val_batch
		initial_images = initial_images.to(device)
		pretrained_embeds = model.encode_image(initial_images)
		pretrained_embeds = F.normalize(pretrained_embeds, dim=-1)
	
	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = list()
	validation_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list() # History of [in-batch] validation metrics dicts per epoch
	full_val_loss_acc_metrics_all_epochs = list() # History of [full] validation metrics dicts per epoch
	
	planned_next_lr = initial_learning_rate # What we PLAN to use in next phase initialization
	planned_next_wd = initial_weight_decay # What we PLAN to use in next phase initialization

	learning_rates_history = list()
	weight_decays_history = list()
	phases_history = list()
	phase_transitions_epochs = list()
	embedding_drift_history = list()
	
	train_start_time = time.time()
	print(f"Training: {num_epochs} epochs, {total_num_phases} phases, {min_epochs_per_phase} min epochs per phase".center(170, "-"))

	for epoch in range(num_epochs):
		print(f"Epoch {epoch+1}/{num_epochs} Phase {current_phase}/{total_num_phases}")
		epoch_start_time = time.time()
		torch.cuda.empty_cache()

		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)

		if epochs_in_current_phase == 0:
			# 1. Create optimizer for the current phase
			trainable_params = [p for p in model.parameters() if p.requires_grad]
			
			if not trainable_params:
				raise ValueError("No trainable parameters found!")
			
			optimizer = torch.optim.AdamW(
				trainable_params,
				lr=planned_next_lr, 					# Use planned LR as starting point
				weight_decay=planned_next_wd,	# Use planned WD as starting point
				betas=(0.9, 0.98),
				eps=1e-6
			)

			# # 2. Configure the main scheduler to take over *after* the warm-up
			# # minimum LR to be X% of what PLANNED LR is for this phase: planned_next_lr * X 
			# if current_phase >= 3:
			# 	eta_min = planned_next_lr * 0.2   # Very conservative for final phases
			# 	cycle_description = "conservative"
			# elif current_phase >= 2:
			# 	eta_min = planned_next_lr * 0.1   # Moderate cycling for mid phases
			# 	cycle_description = "moderate" 
			# else:
			# 	eta_min = planned_next_lr * 0.01  # Aggressive cycling for early phases
			# 	cycle_description = "aggressive"
			
			# # 3. Adaptive cycle length based on remaining epochs and phase
			# remaining_epochs = num_epochs - epoch
			# if current_phase == 0: T_0 = 15  																	# Longer cycles for initial learning
			# elif remaining_epochs < 20: T_0 = max(3, remaining_epochs // 2)  	# Shorter cycles for final phases
			# else: T_0 = max(5, remaining_epochs // 3)													# Balanced cycles for mid phases
			
			T_0 = 10 # 10 epochs
			eta_min = planned_next_lr * 0.10 # 10% of planned_next_lr
			cycle_description = "simple"

			# 4. Create phase-optimized scheduler
			scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
				optimizer=optimizer,
				T_0=T_0,
				T_mult=1,
				eta_min=eta_min,
				last_epoch=-1
			)
			
			print(f"Phase {current_phase} scheduler with {cycle_description} cycling")
			print(f"  ├─ T_0 = {T_0} epochs")
			print(f"  ├─ LR: eta_min: {eta_min} ====>>> PLANNED: {planned_next_lr}")
			print(f"  ├─ Amplitude ratio: {(planned_next_lr/eta_min)}x")
			print(f"  └─ Main scheduler ({scheduler.__class__.__name__}) configured.")

		model.train()
		epoch_train_loss = 0.0
		num_train_batches = len(train_loader)
		trainable_params_exist = any(p.requires_grad for p in model.parameters())

		for bidx, batch_data in enumerate(train_loader):
			images, tokenized_labels, _ = batch_data # Adjust unpacking as needed
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				batch_loss = 0.5 * (loss_img + loss_txt)

			scheduler.step()

			scaler.scale(batch_loss).backward()
			scaler.unscale_(optimizer) # Unscale before clipping
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()

			batch_loss_item = batch_loss.item()
			epoch_train_loss += batch_loss_item

			if bidx % print_every == 0:
				print(f"\tBatch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item}")
			elif bidx == num_train_batches - 1 and batch_loss_item > 0:
				print(f"\tBatch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item}")
			else:
				pass

		avg_training_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 and trainable_params_exist else 0.0
		training_losses.append(avg_training_loss)
		if hasattr(early_stopping, "train_loss_history"):
			early_stopping.train_loss_history.append(avg_training_loss)

		current_lr = optimizer.param_groups[0]['lr']
		current_wd = optimizer.param_groups[0]['weight_decay']

		learning_rates_history.append(current_lr)    # Record what was ACTUALLY used
		weight_decays_history.append(current_wd)     # Record what was ACTUALLY used
		phases_history.append(current_phase)         # Record current phase

		print("="*100)
		print(f"Epoch {epoch+1} current_phase: {current_phase} current_epochs_in_phase: {epochs_in_current_phase}")
		print(f"[ACTUAL] current_lr: {current_lr} current_wd: {current_wd}")
		print(f"[PLANNED] planned_next_lr: {planned_next_lr} planned_next_wd: {planned_next_wd}")
		print("="*100)

		drift_value = compute_embedding_drift(
			model, 
			fixed_val_batch, # Pass the fixed batch
			pretrained_embeds, 
			device, 
			current_phase, 
			epoch + 1
		)
		embedding_drift_history.append(drift_value)

		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)

		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]

		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]
		validation_losses.append(current_val_loss)

		print(
			f'\t[LOSS] Training: {avg_training_loss} Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		should_trans = should_transition_to_next_phase(
			current_phase=current_phase,
			losses=validation_losses,
			window=window_size,
			epochs_in_phase=epochs_in_current_phase,
			min_epochs_per_phase=min_epochs_per_phase,
			num_phases=total_num_phases,
			best_loss=early_stopping.get_best_score(), # Use best score from early stopping state
			best_loss_threshold=min_delta, # Use min_delta for closeness check
			volatility_threshold=volatility_threshold,
			slope_threshold=slope_threshold, # Use positive threshold for worsening loss
			pairwise_imp_threshold=pairwise_imp_threshold,
			# accuracies=val_accs, # Pass average accuracy
			# accuracy_plateau_threshold=accuracy_plateau_threshold,
		)
		if should_trans:
			phase_transitions_epochs.append(epoch)
			current_phase, planned_next_lr, planned_next_wd = handle_phase_transition(current_phase, optimizer)
			print(f"Transition to Phase {current_phase} @ epoch {epoch+1}: New planned LR: {planned_next_lr}, New WD: {planned_next_wd}")
			epochs_in_current_phase = 0 # Reset phase epoch counter
			early_stopping.reset() # <<< CRITICAL: Reset early stopping state for the new phase
			validation_losses = list() # Reset validation losses for the new phase
		else:
			epochs_in_current_phase += 1

		training_should_stop = early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
			current_phase=current_phase
		)
		if training_should_stop:
			print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
			early_stopping_triggered = True
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break
		print(f"Epoch {epoch+1} total elapsed time: {time.time() - epoch_start_time:.1f} sec.".center(170, " "))

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"Training Finished Total Epochs Run: {epoch + 1} Elapsed_t: {total_training_time:.1f} sec".center(170, "-"))
	print(f"Final Phase Reached: {current_phase}")
	print(f"Best Validation Loss Achieved: {early_stopping.get_best_score()} @ Epoch {early_stopping.get_best_epoch() + 1}")

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics as needed
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]

	model_source = evaluation_results["model_loaded_from"]
	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		# f"{dataset_name}_"
		f"{mode}_"
		f"{CLUSTER}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"ep_{actual_trained_epochs}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}_"
		f"mep_{minimum_epochs}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mphb4stp_{min_phases_before_stopping}_"
		f"vt_{volatility_threshold}_"
		f"slp_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"fph_{current_phase}"
	)

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs,
		additional_info={
			'fph': current_phase,
			'flr': planned_next_lr,
			'fwd': planned_next_wd
		}
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retr_perEP.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retr_perK.png"),
		"progressive_dynamics": os.path.join(results_dir, f"{file_base_name}_dyn.png"),
		"phase_analysis": os.path.join(results_dir, f"{file_base_name}_ph_anls.png"),
		"unfreeze_heatmap": os.path.join(results_dir, f"{file_base_name}_unf_hmap.png"),
		"training_summary": os.path.join(results_dir, f"{file_base_name}_train_summary.txt"),
		"loss_evolution": os.path.join(results_dir, f"{file_base_name}_loss_evol.png"),
		"lr_evolution": os.path.join(results_dir, f"{file_base_name}_lr_evol.png"),
		"wd_evolution": os.path.join(results_dir, f"{file_base_name}_wd_evol.png"),
		"phase_efficiency": os.path.join(results_dir, f"{file_base_name}_ph_eff.png"),
		"hyperparameter_correlation": os.path.join(results_dir, f"{file_base_name}_hypp_corr.png"),
		"trainable_layers": os.path.join(results_dir, f"{file_base_name}_train_lyrs.png"),
		"grad_norm": os.path.join(results_dir, f"{file_base_name}_grad_norm.png"),
		"loss_volatility": os.path.join(results_dir, f"{file_base_name}_loss_vol.png"),
		"loss_analyzer": os.path.join(results_dir, f"{file_base_name}_loss_analyzer.png"),
	}

	training_history = collect_progressive_training_history(
		training_losses=training_losses,
		in_batch_metrics_all_epochs=in_batch_loss_acc_metrics_all_epochs,
		learning_rates=learning_rates_history,
		weight_decays=weight_decays_history,
		phases=phases_history,
		embedding_drifts=embedding_drift_history,
		phase_transitions=phase_transitions_epochs,
		early_stop_epoch=epoch+1 if early_stopping_triggered else None,
		best_epoch=early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else None
	)

	analysis_results = plot_phase_transition_analysis(
		training_history=training_history,
		file_path=plot_paths["phase_analysis"],
	)

	analyzer = LossAnalyzer(
		epochs=training_history['epochs'], 
		train_loss=training_history['train_losses'], 
		val_loss=training_history['val_losses']
	)
	analyzer.plot_analysis(fpth=plot_paths["loss_analyzer"])
	signals = analyzer.get_training_signals()
	print(f"\nEMA signal summary: {signals}\n")

	plot_phase_transition_analysis_individual(
		training_history=training_history,
		file_path=plot_paths["phase_analysis"],
	)

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return in_batch_loss_acc_metrics_all_epochs

def full_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		# window_size: int,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float, 
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		use_lamb: bool = False,
	):
	window_size = minimum_epochs + 4
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for full finetune, but for consistency
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__  # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name


	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) | batch_size: {train_loader.batch_size} | {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Extract dropout value from the model (if any)
	dropout_val = None
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	if dropout_val is None:
		dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found

	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"Non-zero dropout detected in base {model_name} {model_arch} during {mode}:")
	print(non_zero_dropouts)

	# Unfreeze all layers for full fine-tuning:
	for name, param in model.named_parameters():
		param.requires_grad = True # all parameters are trainable

	get_parameters_info(model=model, mode=mode)

	if use_lamb:
		param_names = {id(p): n for n, p in model.named_parameters()}
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			weight_decay=weight_decay,
		)
		optimizer.param_names = param_names
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	criterion = torch.nn.CrossEntropyLoss()

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)

			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stabilize training
			scaler.step(optimizer)
			scaler.update()
			scheduler.step() # Update learning rate

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item()}")

				if hasattr(optimizer, 'get_trust_ratio_stats'):
					trust_stats = optimizer.get_trust_ratio_stats()
					if trust_stats:
						print(f"\t\tTrust Ratio Summary:")
						print(f"\t\t{trust_stats}\n")
						optimizer.adaptive_lr_adjustment()

						# Periodic visualization
						if epoch % 10 == 0:
							optimizer.visualize_stats(save_path=os.path.join(results_dir, f"trust_ratio_epoch_{epoch}"))

			epoch_loss += total_loss.item()

		if hasattr(optimizer, 'get_trust_ratio_stats'):
			trust_stats = optimizer.get_trust_ratio_stats()
			if trust_stats:
				print(f"\nEpoch {epoch+1}: Trust Ratio Summary:")
				print(trust_stats)
				print("-"*120)

		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		# all metrics in one using caching mechanism:
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {in_batch_loss_acc_metrics_per_epoch["val_loss"]}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		# --- Early Stopping Check ---
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(170, "-"))
	
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics as needed
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]

	model_source = evaluation_results["model_loaded_from"]
	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		# f"{dataset_name}_"
		f"{mode}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
	)

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=retrieval_metrics_fpth,
	)

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def lora_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		# window_size: int,
		lora_rank: int,
		lora_alpha: float,
		lora_dropout: float,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		use_lamb: bool = False,
	):
	window_size = minimum_epochs + 4
	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold, # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for LoRA finetune, but for consistency
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} | Rank: {lora_rank} | Alpha: {lora_alpha} | Dropout: {lora_dropout} | {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Apply LoRA to the model
	model = get_lora_clip(
		clip_model=model,
		lora_rank=lora_rank,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout,
		verbose=True,
	)
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	if use_lamb:
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		print(f">> Validating Epoch {epoch+1} ...")
		# all metrics in one using caching mechanism:
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
		)
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}'
			f'(Training): {avg_training_loss} '
			f'Validation(in-batch): {current_val_loss}\n'
			f'\tValidation Top-k Accuracy:\n'
			f'\tIn-batch:\n'
			f'\t\t[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\t[text retrieval per image]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t[image retrieval per text]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		print(f"Image-to-Text Retrieval:\n\t{retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval:\n\t{retrieval_metrics_per_epoch['txt2img']}")

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping triggered at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics as needed
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]

	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]
	print(f"Final evaluation used model weights from: {model_source}")
	
	print("--- Final Metrics [In-batch Validation] ---")
	print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
	print("--- Final Metrics [Full Validation Set] ---")
	print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
	print("--- Image-to-Text Retrieval ---")
	print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
	print("--- Text-to-Image Retrieval ---")
	print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		# f"{dataset_name}_"
		f"{mode}_"
		# f"{optimizer.__class__.__name__}_"
		# f"{scheduler.__class__.__name__}_"
		# f"{criterion.__class__.__name__}_"
		# f"{scaler.__class__.__name__}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		# mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		# cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		# mean_reciprocal_rank_file_path=plot_paths["mrr"],
		# cosine_similarity_file_path=plot_paths["cs"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

def probe_finetune_single_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		learning_rate: float,
		weight_decay: float,
		probe_dropout: float,
		device: str,
		results_dir: str,
		window_size: int,
		print_every: int,
		patience: int,
		min_delta: float,
		cumulative_delta: float,
		minimum_epochs: int,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		probe_hidden_dim: int = None,  # Optional hidden layer for MLP probe
		use_lamb: bool = False,
	):
	"""
	Enhanced Linear Probing fine-tuning with robust ViT support.
	
	This method:
	1. Automatically fixes ViT positional embedding mismatches
	2. Freezes all CLIP parameters (vision and text encoders)
	3. Extracts features from the frozen CLIP model
	4. Trains a linear classifier (or shallow MLP) on top of these features
	
	The probe can be:
	- Simple linear layer: CLIP features -> num_classes
	- Two-layer MLP: CLIP features -> hidden_dim -> num_classes (if probe_hidden_dim is specified)
	"""

	# Inspect the model for dropout layers
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1, # Not really needed for linear probe, but for consistency
	)
	
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_single_label', '', mode)
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) | batch_size: {train_loader.batch_size} | {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))
	
	# Get number of classes
	try:
		class_names = validation_loader.dataset.dataset.classes
		num_classes = len(class_names)
	except AttributeError:
		class_names = validation_loader.dataset.unique_labels
		num_classes = len(class_names)
	
	print(f"Number of Labels: {num_classes}")
	
	# =====================================
	# STEP 1: FREEZE ALL CLIP PARAMETERS AND CREATE PROBE
	# =====================================
	for param in model.parameters():
		param.requires_grad = False # Freeze all CLIP parameters(no fine-tuning)
	
	get_parameters_info(model=model, mode=mode)

	# Create the robust linear probe that handles everything automatically
	print("\nCreating probe model...")
	
	probe = get_probe_clip(
			clip_model=model,
			validation_loader=validation_loader,
			device=torch.device(device),
			# hidden_dim=256,  # Optional: creates MLP probe
			dropout=probe_dropout,
			zero_shot_init=True,
			verbose=True
	)
	print(f"DEBUG: Probe created successfully")
	print(f"Multi-label dataset? {isinstance(probe, MultiLabelProbe)}")

	clip_dim = probe.input_dim  # Get the detected feature dimension
	probe_params = sum(p.numel() for p in probe.parameters())

	print(f"DEBUG: Probe class: {probe.__class__.__name__}")
	print(f"DEBUG: Probe input_dim: {probe.input_dim}")
	print(f"DEBUG: Probe num_classes: {probe.num_classes}")


	print(f"CLIP output dimension: {clip_dim}")
	print(f"Probe type: {probe.probe_type} | Probe parameters: {probe_params:,}")
	
	# Debug: Print probe architecture details
	if probe.probe_type == "Linear":
		print(f"Probe weight shape: {probe.probe.weight.shape}")
		print(f"Probe bias shape: {probe.probe.bias.shape}")
	else:
		print(f"MLP Probe architecture: {probe.probe}")
	
	# =====================================
	# STEP 2: SETUP TRAINING
	# =====================================
	
	# Only optimize probe parameters
	if use_lamb:
		optimizer = LAMB(
			params=probe.parameters(),
			lr=learning_rate,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=probe.parameters(),
			lr=learning_rate,
			betas=(0.9, 0.999),
			eps=1e-8,
			weight_decay=weight_decay,
		)
	print(f"Using {optimizer.__class__.__name__} for optimization")
	
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=num_epochs,
		eta_min=learning_rate * 0.01, # 1% of initial LR
	)
	print(f"Using {scheduler.__class__.__name__} for learning rate scheduling")
	
	criterion = torch.nn.CrossEntropyLoss()
	print(f"Using {criterion.__class__.__name__} as the loss function")
	
	scaler = torch.amp.GradScaler(device=device)
	print(f"Using {scaler.__class__.__name__} for mixed precision training")
	
	mdl_fpth = os.path.join(
		results_dir,
		f"{probe.probe_type.lower()}_probe_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"hdim_{probe_hidden_dim}_"
		f"pdo_{probe_dropout}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	
	print(f"Best model will be saved in: {mdl_fpth}")
	
	# =====================================
	# STEP 3: EXTRACT AND CACHE FEATURES (OPTIONAL)
	# =====================================
	# For efficiency, we can pre-extract all features once
	print("Extracting features from frozen CLIP model...")
	
	def extract_features(loader, model, device):
			"""Extract features from frozen CLIP model"""
			features = list()
			labels = list()
			
			model.eval()
			with torch.no_grad():
					for images, _, label_indices in tqdm(loader, desc="Extracting features"):
							images = images.to(device, non_blocking=True)
							# Extract CLIP features
							image_features = model.encode_image(images)
							image_features = F.normalize(image_features, dim=-1)
							
							features.append(image_features.cpu())
							labels.append(label_indices.cpu())
			
			return torch.cat(features, dim=0), torch.cat(labels, dim=0)
	
	# Extract features once (optional - can be skipped for online extraction)
	train_features, train_labels = extract_features(train_loader, model, device)
	val_features, val_labels = extract_features(validation_loader, model, device)
	
	print(f"Extracted features - Train: {train_features.shape}, Val: {val_features.shape}")
	print(f"DEBUG: Extracted train features shape: {train_features.shape}")
	print(f"DEBUG: Extracted val features shape: {val_features.shape}")
	print(f"DEBUG: Train labels shape: {train_labels.shape}")
	print(f"DEBUG: Val labels shape: {val_labels.shape}")	


	# Create feature datasets
	from torch.utils.data import TensorDataset
	train_feature_dataset = TensorDataset(train_features, train_labels)
	val_feature_dataset = TensorDataset(val_features, val_labels)
	
	# Create new dataloaders for features
	train_feature_loader = DataLoader(
			train_feature_dataset,
			batch_size=train_loader.batch_size,
			shuffle=True,
			num_workers=0,  # Features are already in memory
			pin_memory=False
	)
	
	val_feature_loader = DataLoader(
			val_feature_dataset,
			batch_size=validation_loader.batch_size,
			shuffle=False,
			num_workers=0,
			pin_memory=False
	)
	
	# =====================================
	# STEP 4: TRAINING LOOP
	# =====================================
	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	
	for epoch in range(num_epochs):
			train_and_val_st_time = time.time()
			torch.cuda.empty_cache()
			
			# Training
			probe.train()
			print(f"Epoch [{epoch + 1}/{num_epochs}]")
			epoch_loss = 0.0
			correct = 0
			total = 0
			
			for bidx, (features, labels) in enumerate(train_feature_loader):
					features = features.to(device, non_blocking=True)
					labels = labels.to(device, non_blocking=True)
					
					optimizer.zero_grad()
					
					with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
							# Forward pass through probe only
							logits = probe(features)
							loss = criterion(logits, labels)
					
					scaler.scale(loss).backward()
					torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
					scaler.step(optimizer)
					scaler.update()
					
					# Track accuracy
					_, predicted = torch.max(logits.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()
					
					if bidx % print_every == 0 or bidx + 1 == len(train_feature_loader):
							print(f"\t\tBatch [{bidx + 1}/{len(train_feature_loader)}] Loss: {loss.item():.6f}")
					
					epoch_loss += loss.item()
			
			avg_training_loss = epoch_loss / len(train_feature_loader)
			train_accuracy = 100 * correct / total
			training_losses.append(avg_training_loss)
			
			print(f"Training Loss: {avg_training_loss:.6f}, Training Accuracy: {train_accuracy:.2f}%")
			
			# Validation
			probe.eval()
			val_loss = 0.0
			correct = 0
			total = 0
			
			with torch.no_grad():
					for features, labels in val_feature_loader:
							features = features.to(device, non_blocking=True)
							labels = labels.to(device, non_blocking=True)
							
							logits = probe(features)
							loss = criterion(logits, labels)
							
							val_loss += loss.item()
							
							_, predicted = torch.max(logits.data, 1)
							total += labels.size(0)
							correct += (predicted == labels).sum().item()
			
			avg_val_loss = val_loss / len(val_feature_loader)
			val_accuracy = 100 * correct / total
			
			print(f"Validation Loss: {avg_val_loss:.6f}, Validation Accuracy: {val_accuracy:.2f}%")
			
			# For compatibility with existing code, create metrics dict
			in_batch_metrics = {
					"val_loss": avg_val_loss,
					"accuracy": val_accuracy / 100,  # Convert to fraction
			}
			in_batch_loss_acc_metrics_all_epochs.append(in_batch_metrics)
			
			# Update scheduler
			scheduler.step()
			
			# Early stopping
			if early_stopping.should_stop(
					current_value=avg_val_loss,
					model=probe,  # Save probe weights, not CLIP
					epoch=epoch,
					optimizer=optimizer,
					scheduler=scheduler,
					checkpoint_path=mdl_fpth,
			):
					print(f"\nEarly stopping at epoch {epoch + 1}")
					break
			
			print(f"Epoch {epoch+1} Duration: {time.time() - train_and_val_st_time:.2f} sec".center(170, "-"))
	
	print(f"[{mode}] Total Training Time: {time.time() - train_start_time:.1f} sec".center(170, "-"))
	
	# Continue with the rest of your original code (evaluation, plotting, etc.)
	# [Keep all the remaining parts of your original function unchanged]
	
	# =====================================
	# STEP 5: FINAL EVALUATION
	# =====================================
	# Load best probe weights
	if os.path.exists(mdl_fpth):
		print(f"Loading best probe weights from {mdl_fpth}")
		checkpoint = torch.load(mdl_fpth, map_location=device)
		if 'model_state_dict' in checkpoint:
			probe.load_state_dict(checkpoint['model_state_dict'])
		else:
			probe.load_state_dict(checkpoint)
	
	# Create combined model for evaluation
	class CLIPWithProbe(torch.nn.Module):
			def __init__(self, clip_model, probe):
					super().__init__()
					self.clip = clip_model
					self.probe = probe
					# Copy necessary attributes from CLIP
					self.visual = clip_model.visual
					self.encode_image = clip_model.encode_image
					self.encode_text = clip_model.encode_text
					self.name = getattr(clip_model, 'name', 'unknown')
			
			def forward(self, images, texts):
					# For compatibility with evaluation code
					image_features = self.clip.encode_image(images)
					image_features = F.normalize(image_features, dim=-1)
					
					# Get logits from probe
					logits = self.probe(image_features)
					
					# For CLIP-style evaluation, we need to return image-text similarity
					# We'll use the probe's logits as a proxy
					batch_size = images.shape[0]
					num_classes = logits.shape[1]
					
					# Create pseudo-similarity matrix
					# This is a hack for compatibility - ideally evaluation should be adapted
					if batch_size == num_classes:
							return logits, logits.T
					else:
							# Pad or truncate to make square matrix
							min_dim = min(batch_size, num_classes)
							logits_per_image = logits[:min_dim, :min_dim]
							logits_per_text = logits_per_image.T
							return logits_per_image, logits_per_text
	
	# Combine CLIP and probe for evaluation
	combined_model = CLIPWithProbe(model, probe)
	
	# Run evaluation with the combined model
	evaluation_results = evaluate_best_model(
		model=combined_model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(
			batch_size=validation_loader.batch_size, 
			N=10, 
			device=device
		),
	)
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	print(f"Final evaluation used model weights from: {evaluation_results['model_loaded_from']}")
	print("--- Final Metrics [In-batch Validation] ---")
	print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
	print("--- Final Metrics [Full Validation Set] ---")
	print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
	print("--- Image-to-Text Retrieval ---")
	print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
	print("--- Text-to-Image Retrieval ---")
	print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))
	# Generate plots
	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)
	
	file_base_name = (
		# f"{dataset_name}_"
		f"{mode}_{probe.probe_type}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}"
	)
	
	# Update model path
	mdl_fpth = get_updated_model_name(
			original_path=mdl_fpth,
			actual_epochs=actual_trained_epochs
	)
	
	print(f"Best model will be renamed to: {mdl_fpth}")
	
	# Print final summary
	best_val_loss = early_stopping.get_best_score() or 0.0
	print("\n" + "="*80)
	print("ENHANCED LINEAR PROBE TRAINING SUMMARY")
	print("="*80)
	print(f"Method: {mode}")
	print(f"Model: {getattr(model, 'name', 'Unknown')}")
	print(f"Probe Type: {probe.probe_type}")
	print(f"Probe Parameters: {probe_params:,}")
	print(f"CLIP Parameters (frozen): {sum(p.numel() for p in model.parameters()):,}")
	print(f"Total Epochs: {actual_trained_epochs}")
	print(f"Best Validation Loss: {best_val_loss}")
	print(f"Best Epoch: {early_stopping.get_best_epoch() + 1}")
	print("="*80)
	plot_paths = {
			"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
			"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
			"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
			"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
			"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
			"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
			"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}
	plot_loss_accuracy_metrics(
			dataset_name=dataset_name,
			train_losses=training_losses,
			val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			losses_file_path=plot_paths["losses"],
			in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
			in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
			full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
			full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)
	
	plot_retrieval_metrics_per_epoch(
			dataset_name=dataset_name,
			image_to_text_metrics_list=img2txt_metrics_all_epochs,
			text_to_image_metrics_list=txt2img_metrics_all_epochs,
			fname=plot_paths["retrieval_per_epoch"],
	)
	
	plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=final_img2txt_metrics,
			text_to_image_metrics=final_txt2img_metrics,
			fname=plot_paths["retrieval_best"],
	)
	return in_batch_loss_acc_metrics_all_epochs

def full_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		volatility_threshold: float = 15.0,
		slope_threshold: float = 1e-4, 
		pairwise_imp_threshold: float = 1e-4,
		use_lamb: bool = False,
	):
	"""
	Full fine-tuning for multi-label CLIP classification.
	
	Key changes from single-label version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Proper multi-label evaluation metrics
	4. Custom loss computation for contrastive multi-label learning
	
	Args:
			model: CLIP model to fine-tune
			train_loader: Training DataLoader (must provide multi-label vectors)
			validation_loader: Validation DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			window_size: Window size for early stopping
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			label_smoothing: Label smoothing factor (0.0 = no smoothing)
	"""
	
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))

	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))
	
	try:
		num_classes = len(validation_loader.dataset.unique_labels)
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		num_classes = len(validation_loader.dataset.dataset.classes)
		class_names = validation_loader.dataset.dataset.classes
	print(f"Multi-label dataset: {num_classes} classes")
	
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]

	print(f"\nNon-zero dropout detected in base {model_name} {model_arch} during {mode}:")
	print(non_zero_dropouts)
	print()

	# Unfreeze all layers for full fine-tuning
	for name, param in model.named_parameters():
		param.requires_grad = True

	get_parameters_info(model=model, mode=mode)

	# NEW: Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	print(f"Using {criterion.__class__.__name__} for multi-label classification")
	# Pre-encode all class texts (for efficiency)
	print(f"Pre-encoding {num_classes} class texts...")
	all_class_texts = clip.tokenize(class_names).to(device)
	with torch.no_grad():
			model.eval()
			all_class_embeds = model.encode_text(all_class_texts)
			all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	model.train()

	if use_lamb:
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdelta_{min_delta:.1e}_"
		f"cdelta_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")
	training_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	for epoch in range(num_epochs):
			train_and_val_st_time = time.time()
			torch.cuda.empty_cache()
			model.train()
			print(f"Epoch [{epoch + 1}/{num_epochs}]")
			
			epoch_loss_total = 0.0
			epoch_loss_i2t = 0.0
			epoch_loss_t2i = 0.0
			num_batches = 0
			for bidx, batch_data in enumerate(train_loader):
				if len(batch_data) == 3:
					images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
				else:
					raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
				batch_size = images.size(0)
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				# Validate label_vectors shape
				if label_vectors.shape != (batch_size, num_classes):
					raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")
				optimizer.zero_grad(set_to_none=True)
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					# ================================
					# CRITICAL CHANGE: Multi-label loss computation
					# ================================
					total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
						model=model,
						images=images,
						all_class_embeds=all_class_embeds,
						label_vectors=label_vectors,
						criterion=criterion,
						temperature=temperature,
						loss_weights=loss_weights
					)
				# Check for NaN loss
				if torch.isnan(total_loss):
					print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
					continue
				scaler.scale(total_loss).backward()
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()

				# Track losses
				batch_loss_total = total_loss.item()
				batch_loss_i2t = loss_i2t.item()
				batch_loss_t2i = loss_t2i.item()
				
				epoch_loss_total += batch_loss_total
				epoch_loss_i2t += batch_loss_i2t
				epoch_loss_t2i += batch_loss_t2i
				num_batches += 1
				if bidx % print_every == 0 or bidx + 1 == len(train_loader):
					print(
						f"\t\tBatch [{bidx + 1}/{len(train_loader)}] "
						f"Total Loss: {batch_loss_total:.6f} "
						f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
					)

			avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
			avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
			avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0

			training_losses.append(avg_total_loss)
			training_losses_breakdown["total"].append(avg_total_loss)
			training_losses_breakdown["i2t"].append(avg_i2t_loss)
			training_losses_breakdown["t2i"].append(avg_t2i_loss)

			print(f">> Training completed in {time.time() - train_and_val_st_time:.2f} sec. Validating Epoch {epoch+1}")

			current_val_loss = compute_multilabel_validation_loss(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				device=device,
				all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
				temperature=temperature,
				max_batches=10
			)
			validation_results = get_validation_metrics(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,  # Now uses BCEWithLogitsLoss
				device=device,
				topK_values=topk_values,
				finetune_strategy=mode,
				cache_dir=results_dir,
				verbose=True,
				max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
				is_training=True,
				model_hash=get_model_hash(model),
				temperature=temperature,
			)
			
			in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
			in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss
			full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
			retrieval_metrics_per_epoch = {
				"img2txt": validation_results["img2txt_metrics"],
				"txt2img": validation_results["txt2img_metrics"]
			}
			in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
			full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
			img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
			txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
			current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]
			print(
				f'@ Epoch {epoch + 1}:\n'
				f'\t[LOSS] {mode}:\n'
				f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
				f'\t\tValidation: {current_val_loss:.6f}\n'
				f'\tMulti-label Validation Metrics:\n'
				f'\t\tIn-batch Top-K Accuracy:\n'
				f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
				f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
				f'\t\tFull Validation Set:\n'
				f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
				f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
			)

			if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
				print(f'\tMulti-label Metrics:')
				print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
				print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
				print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
			
			print(f"\tRetrieval Metrics:")
			print(
				f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
				f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
			)
			print(
				f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
				f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
			)

			if hasattr(train_loader.dataset, 'get_cache_stats'):
				print(f"#"*100)
				cache_stats = train_loader.dataset.get_cache_stats()
				if cache_stats is not None:
					print(f"Train Cache Stats: {cache_stats}")

			if hasattr(validation_loader.dataset, 'get_cache_stats'):
				cache_stats = validation_loader.dataset.get_cache_stats()
				if cache_stats is not None:
					print(f"Validation Cache Stats: {cache_stats}")
				print(f"#"*100)

			if early_stopping.should_stop(
				current_value=current_val_loss,
				model=model,
				epoch=epoch,
				optimizer=optimizer,
				scheduler=scheduler,
				checkpoint_path=mdl_fpth,
			):
				print(
					f"\nEarly stopping at epoch {epoch + 1} "
					f"with best loss: {early_stopping.get_best_score()} "
					f"obtained in epoch {early_stopping.get_best_epoch()+1}")
				break
			print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f}s".center(150, "="))
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	# ================================
	# FINAL EVALUATION
	# ================================
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]

	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")

	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)
	
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	# ================================
	# PLOTTING: Enhanced for multi-label
	# ================================
	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}
	# Plot training loss breakdown
	plot_multilabel_loss_breakdown(
			training_losses_breakdown=training_losses_breakdown,
			filepath=plot_paths["losses_breakdown"]
	)
	plot_loss_accuracy_metrics(
			dataset_name=dataset_name,
			train_losses=training_losses,
			val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
			full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
			losses_file_path=plot_paths["losses"],
			in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
			in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
			full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
			full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)
	plot_retrieval_metrics_per_epoch(
			dataset_name=dataset_name,
			image_to_text_metrics_list=img2txt_metrics_all_epochs,
			text_to_image_metrics_list=txt2img_metrics_all_epochs,
			fname=plot_paths["retrieval_per_epoch"],
	)
	plot_retrieval_metrics_best_model(
			dataset_name=dataset_name,
			image_to_text_metrics=final_img2txt_metrics,
			text_to_image_metrics=final_txt2img_metrics,
			fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def progressive_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int = 10,
		patience: int = 10,
		min_delta: float = 1e-4,  # Make slightly less sensitive than default
		cumulative_delta: float = 5e-3,  # Keep cumulative check reasonable
		minimum_epochs: int = 20,  # Minimum epochs before ANY early stop
		min_epochs_per_phase: int = 5,  # Minimum epochs within a phase before transition check
		volatility_threshold: float = 15.0,  # Allow slightly more volatility
		slope_threshold: float = 1e-4,  # Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float = 1e-4,  # Stricter requirement for pairwise improvement
		accuracy_plateau_threshold: float = 5e-4,  # For phase transition based on accuracy
		min_phases_before_stopping: int = 3,  # Ensure significant unfreezing before global stop
		topk_values: list[int] = [1, 5, 10],
		layer_groups_to_unfreeze: list[str] = ['visual_transformer', 'text_transformer', 'projections'],  # Focus on key layers
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		use_lamb: bool = False,
	):
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	initial_learning_rate = learning_rate
	initial_weight_decay = weight_decay

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		min_phases_before_stopping=min_phases_before_stopping,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	try:
		num_classes = len(validation_loader.dataset.unique_labels)
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		num_classes = len(validation_loader.dataset.dataset.classes)
		class_names = validation_loader.dataset.dataset.classes
	print(f"Multi-label progressive fine-tuning: {num_classes} classes")
	# print(f"Class names sample:\n{class_names}")

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	# dropout layers inspection:
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during {mode} fine-tuning:")
	print(non_zero_dropouts)
	print()

	# Get the detailed layer unfreeze schedule
	total_num_phases = min_phases_before_stopping + 5
	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		num_phases=total_num_phases,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	print(f"Using {criterion.__class__.__name__} for multi-label classification")

	# Pre-encode all class texts (for efficiency)
	print(f"Pre-encoding {num_classes} class texts...")
	all_class_texts = clip.tokenize(class_names).to(device)
	with torch.no_grad():
		model.eval()
		all_class_embeds = model.encode_text(all_class_texts)
		all_class_embeds = F.normalize(all_class_embeds, dim=-1)

	model.train()
	if use_lamb:
		optimizer = LAMB(
			params=filter(lambda p: p.requires_grad, model.parameters()),  # Initially might be empty if phase 0 has no unfrozen layers
			lr=initial_learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=initial_weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=filter(lambda p: p.requires_grad, model.parameters()),  # Initially might be empty if phase 0 has no unfrozen layers
			lr=initial_learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=initial_weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=initial_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,  # Standard pct_start
		anneal_strategy='cos'  # Cosine annealing
	)

	print(f"Using {scheduler.__class__.__name__} for learning rate scheduling")
	print(f"Using {criterion.__class__.__name__} as the loss function")

	scaler = torch.amp.GradScaler(device=device) # automatic mixed precision
	print(f"Using {scaler.__class__.__name__} for automatic mixed precision training")

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}_"
		f"mepph_{min_epochs_per_phase}_"
		f"mpbs_{min_phases_before_stopping}_"
		f".pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = list()  # History of average training loss per epoch
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}  # Multi-label loss breakdown
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()  # History of [in-batch] validation metrics dicts per epoch
	full_val_loss_acc_metrics_all_epochs = list()  # History of [full] validation metrics dicts per epoch
	best_val_loss = None  # Track the absolute best validation loss
	layer_cache = {}  # Cache for layer status (optional, used by get_status)
	last_lr = initial_learning_rate  # Track current LR
	last_wd = initial_weight_decay  # Track current WD
	phase_just_changed = False  # Flag to signal optimizer refresh needed

	# --- Main Training Loop ---
	train_start_time = time.time()

	for epoch in range(num_epochs):
		epoch_start_time = time.time()
		print(f"Epoch {epoch+1}/{num_epochs} Phase {current_phase}/{total_num_phases} current LR: {last_lr} current WD: {last_wd}")
		torch.cuda.empty_cache()
		
		# --- Phase Transition Check ---
		# Check only if enough epochs *overall* and *within the phase* have passed,
		# and if we are not already in the last phase.
		if (epoch >= minimum_epochs and  # Overall min epochs check
			epochs_in_current_phase >= min_epochs_per_phase and
			current_phase < total_num_phases - 1 and
			len(early_stopping.value_history) >= window_size):
			print(f"Checking phase transition ({epochs_in_current_phase} elapsed epochs in phase {current_phase})")

			val_losses = early_stopping.value_history
			
			# For multi-label, we can use average accuracy across I2T and T2I
			val_accs_in_batch = list()
			for m in in_batch_loss_acc_metrics_all_epochs:
				i2t_acc = m.get('img2txt_acc', 0.0)
				t2i_acc = m.get('txt2img_acc', 0.0)
				avg_acc = (i2t_acc + t2i_acc) / 2.0
				val_accs_in_batch.append(avg_acc)

			should_trans = should_transition_to_next_phase(
				current_phase=current_phase,
				losses=val_losses,
				window=window_size,
				best_loss=early_stopping.get_best_score(),  # Use best score from early stopping state
				best_loss_threshold=min_delta,  # Use min_delta for closeness check
				volatility_threshold=volatility_threshold,
				slope_threshold=slope_threshold,  # Use positive threshold for worsening loss
				pairwise_imp_threshold=pairwise_imp_threshold,
				accuracies=val_accs_in_batch,  # Pass average accuracy for multi-label
				accuracy_plateau_threshold=accuracy_plateau_threshold,
			)
			
			if should_trans:
				current_phase, last_lr, last_wd = handle_phase_transition(
					current_phase=current_phase,
					last_lr=last_lr,
					last_wd=last_wd,
				)
				epochs_in_current_phase = 0  # Reset phase epoch counter
				early_stopping.reset()  # <<< CRITICAL: Reset early stopping state for the new phase
				print(f"Transitioned to Phase {current_phase}. Early stopping reset.")

				phase_just_changed = True  # Signal that optimizer needs refresh after unfreeze
				print(f"Phase transition triggered. Optimizer/Scheduler refresh pending after unfreeze.")
				print(f"Current Phase: {current_phase}")

		# --- Unfreeze Layers for Current Phase ---
		print(f"Applying unfreeze strategy for Phase {current_phase}...")
		# Ensure layers are correctly frozen/unfrozen *before* optimizer step
		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)
		
		if phase_just_changed or epoch == 0:
			print("Refreshing optimizer parameter groups...")
			optimizer.param_groups.clear()
			optimizer.add_param_group(
				{
					'params': [p for p in model.parameters() if p.requires_grad],
					'lr': last_lr,  # Use the new LR
					'weight_decay': last_wd,  # Use the new WD
				}
			)
			print(f"Optimizer parameter groups refreshed. LR set to {last_lr}, WD set to {last_wd}.")
			print("Re-initializing OneCycleLR scheduler for new phase/start...")
			steps_per_epoch = len(train_loader)
			# Schedule over remaining epochs (more adaptive)
			# max: Ensure scheduler_epochs is at least 1
			scheduler_epochs = max(1, num_epochs - epoch)
			scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=last_lr,  # Use the new LR as the peak for the new cycle
				steps_per_epoch=steps_per_epoch,
				epochs=scheduler_epochs,
				pct_start=0.1,  # Consider if this needs adjustment in later phases
				anneal_strategy='cos',
				# last_epoch = -1 # Ensures it starts fresh
			)
			print(f"Scheduler re-initialized with max_lr={last_lr} for {scheduler_epochs} epochs.")
			phase_just_changed = False  # Reset the flag

		# --- Training Epoch ---
		model.train()
		epoch_train_loss_total = 0.0
		epoch_train_loss_i2t = 0.0
		epoch_train_loss_t2i = 0.0
		num_train_batches = len(train_loader)
		num_processed_batches = 0
		
		trainable_params_exist = any(p.requires_grad for p in model.parameters())
		if not trainable_params_exist:
			print("Warning: No trainable parameters found for the current phase. Skipping training steps.")
		else:
			for bidx, batch_data in enumerate(train_loader):
				if len(batch_data) == 3:
					images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
				else:
					raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
				
				batch_size = images.size(0)
				images = images.to(device, non_blocking=True)
				label_vectors = label_vectors.to(device, non_blocking=True).float()
				
				# Validate label_vectors shape
				if label_vectors.shape != (batch_size, num_classes):
					raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")

				optimizer.zero_grad(set_to_none=True)
				
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					# Multi-label contrastive loss computation
					total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
						model=model,
						images=images,
						all_class_embeds=all_class_embeds,
						label_vectors=label_vectors,
						criterion=criterion,
						temperature=temperature,
						loss_weights=loss_weights
					)
				
				if torch.isnan(total_loss):
					print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
					continue  # Skip optimizer step if loss is NaN
				
				scaler.scale(total_loss).backward()
				scaler.unscale_(optimizer)  # Unscale before clipping
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()  # Step the scheduler
				
				# Track losses
				batch_loss_total = total_loss.item()
				batch_loss_i2t = loss_i2t.item()
				batch_loss_t2i = loss_t2i.item()
				
				epoch_train_loss_total += batch_loss_total
				epoch_train_loss_i2t += batch_loss_i2t
				epoch_train_loss_t2i += batch_loss_t2i
				num_processed_batches += 1

				if bidx % print_every == 0:
					print(
						f"\tBatch [{bidx+1}/{num_train_batches}] "
						f"Total Loss: {batch_loss_total:.6f} "
						f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
					)
				elif bidx == num_train_batches - 1 and batch_loss_total > 0:
					print(
						f"\tBatch [{bidx+1}/{num_train_batches}] "
						f"Total Loss: {batch_loss_total:.6f} "
						f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
					)

		# Calculate average losses
		avg_total_loss = epoch_train_loss_total / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		avg_i2t_loss = epoch_train_loss_i2t / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		avg_t2i_loss = epoch_train_loss_t2i / num_processed_batches if num_processed_batches > 0 and trainable_params_exist else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		# --- Validation ---
		print(f">> Training Completed in {time.time() - epoch_start_time:.2f} sec. Validating Epoch: {epoch+1}")

		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			max_batches=10
		)

		# Get comprehensive validation metrics
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,  # Now uses BCEWithLogitsLoss
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss  # Use computed validation loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)

		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"#"*100)
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"#"*100)
				print(f"Validation Cache Stats: {cache_stats}")
		
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
			current_phase=current_phase
		):
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break  # Exit the main training loop

		# --- End of Epoch ---
		epochs_in_current_phase += 1

		if epoch+1 > minimum_epochs: 
			print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
		print(f"Epoch {epoch+1} Elapsed_t: {time.time()-epoch_start_time:.2f} sec".center(170, "-"))

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"\n--- Training Finished ---")
	print(f"Total Epochs Run: {epoch + 1}")
	print(f"Final Phase Reached: {current_phase}")
	print(f"Best Validation Loss Achieved: {early_stopping.get_best_score()} @ Epoch {early_stopping.get_best_epoch() + 1}")
	print(f"Total Training Time: {total_training_time:.2f}s")

	# Final evaluation with best model
	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=True,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics as needed
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]

	model_source = evaluation_results["model_loaded_from"]
	print(f"Final evaluation used model weights from: {model_source}")

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"last_phase_{current_phase}_"
		f"ep_{actual_trained_epochs}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}_"
		f"temp_{temperature}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}"
	)
	if last_lr is not None:
		file_base_name += f"_final_lr_{last_lr:.1e}"

	if last_wd is not None:
		file_base_name += f"_final_wd_{last_wd:.1e}"

	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth,
		actual_epochs=actual_trained_epochs,
		additional_info={
			'last_phase': current_phase,
			'flr': last_lr,
			'fwd': last_wd
		}
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	# Plot training loss breakdown (specific to multi-label)
	plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def lora_finetune_multi_label(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int,
		lora_rank: int,
		lora_alpha: float,
		lora_dropout: float,
		verbose: bool = True,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		topk_values: List[int] = [1, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,  # For balancing I2T and T2I losses
		temperature: float = 0.07,  # Temperature for contrastive learning
		label_smoothing: float = 0.0,  # Label smoothing for multi-label
		volatility_threshold: float = 15.0,
		slope_threshold: float = 1e-4, 
		pairwise_imp_threshold: float = 1e-4,
		use_lamb: bool = False,
	):
	"""
	LoRA fine-tuning for multi-label CLIP classification.
	
	Key differences from single-label LoRA version:
	1. Uses BCEWithLogitsLoss instead of CrossEntropyLoss
	2. Handles bidirectional multi-label targets (I2T and T2I)
	3. Pre-encodes class embeddings for efficiency
	4. Uses multi-label specific loss computation
	5. Proper multi-label evaluation metrics
	
	Args:
			model: CLIP model to fine-tune with LoRA
			train_loader: Training DataLoader (must provide multi-label vectors)
			validation_loader: Validation DataLoader  
			num_epochs: Number of training epochs
			print_every: Print loss every N batches
			learning_rate: Learning rate for LoRA parameters
			weight_decay: Weight decay for regularization
			device: Training device (cuda/cpu)
			results_dir: Directory to save results
			window_size: Window size for early stopping
			lora_rank: LoRA rank parameter
			lora_alpha: LoRA alpha parameter
			lora_dropout: LoRA dropout parameter
			patience: Early stopping patience
			min_delta: Minimum change for improvement
			cumulative_delta: Cumulative delta for early stopping
			minimum_epochs: Minimum epochs before early stopping
			topk_values: K values for evaluation metrics
			loss_weights: Optional weights for I2T and T2I losses
			temperature: Temperature scaling for similarities
			label_smoothing: Label smoothing factor (0.0 = no smoothing)
	"""
	
	# Set default loss weights
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	# Inspect the model for dropout layers and validate for LoRA
	dropout_values = list()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	# Check for non-zero dropout in the base model
	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	if non_zero_dropouts:
		dropout_info = ", ".join([f"{name}: p={p}" for name, p in non_zero_dropouts])
		assert False, (
			f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during LoRA fine-tuning:"
			f"\n{dropout_info}\n"
			"This adds stochasticity and noise to the frozen base model, which is unconventional for LoRA practices.\n"
			"Fix: Set dropout=0.0 in clip.load() to enforce a deterministic base model behavior during LoRA fine-tuning "
			"which gives you more control over LoRA-specific regularization without affecting the base model.\n"
		)

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',  # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
		volatility_threshold=volatility_threshold,
		slope_threshold=slope_threshold,  # Positive slope is bad for loss
		pairwise_imp_threshold=pairwise_imp_threshold,
		# min_phases_before_stopping=1,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune_multi_label', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} | Rank: {lora_rank} | Alpha: {lora_alpha} | Dropout: {lora_dropout} | {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Get dataset information
	try:
		num_classes = len(validation_loader.dataset.unique_labels)
		class_names = validation_loader.dataset.unique_labels
	except AttributeError:
		num_classes = len(validation_loader.dataset.dataset.classes)
		class_names = validation_loader.dataset.dataset.classes
	print(f"Multi-label LoRA fine-tuning: {num_classes} classes")
	# print(f"Class names sample: {class_names[:5]}...")

	# Apply LoRA to the model
	model = get_lora_clip(
		clip_model=model,
		lora_rank=lora_rank,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout
	)
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	# Use BCEWithLogitsLoss for multi-label classification
	if label_smoothing > 0:
		print(f"Using label smoothing: {label_smoothing}")
		criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
	else:
		criterion = torch.nn.BCEWithLogitsLoss()
	print(f"Using {criterion.__class__.__name__} for multi-label classification")

	# Pre-encode all class texts (for efficiency)
	print(f"Pre-encoding {num_classes} class texts...")
	all_class_texts = clip.tokenize(class_names).to(device)
	with torch.no_grad():
		model.eval()
		all_class_embeds = model.encode_text(all_class_texts)
		all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	model.train()

	if use_lamb:
		optimizer = LAMB(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)
	else:
		optimizer = torch.optim.AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate,
			betas=(0.9, 0.98),
			eps=1e-6,
			weight_decay=weight_decay,
		)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	scaler = torch.amp.GradScaler(device=device)

	mdl_fpth = os.path.join(
		results_dir,
		f"{mode}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"mep_{minimum_epochs}_"
		f"pat_{patience}_"
		f"mdt_{min_delta:.1e}_"
		f"cdt_{cumulative_delta:.1e}_"
		f"vt_{volatility_threshold}_"
		f"st_{slope_threshold:.1e}_"
		f"pit_{pairwise_imp_threshold:.1e}"
		f".pth"
	)

	training_losses = list()
	training_losses_breakdown = {"i2t": [], "t2i": [], "total": []}
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	full_val_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None

	for epoch in range(num_epochs):
		train_and_val_st_time = time.time()
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		
		epoch_loss_total = 0.0
		epoch_loss_i2t = 0.0
		epoch_loss_t2i = 0.0
		num_batches = 0
		
		for bidx, batch_data in enumerate(train_loader):
			if len(batch_data) == 3:
				images, _, label_vectors = batch_data  # Ignore tokenized_labels, use pre-encoded
			else:
				raise ValueError(f"Expected 3 items from DataLoader, got {len(batch_data)}")
			
			batch_size = images.size(0)
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Validate label_vectors shape
			if label_vectors.shape != (batch_size, num_classes):
				raise ValueError(f"Label vectors shape {label_vectors.shape} doesn't match expected ({batch_size}, {num_classes})")

			optimizer.zero_grad(set_to_none=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				# Multi-label contrastive loss computation
				total_loss, loss_i2t, loss_t2i = compute_multilabel_contrastive_loss(
					model=model,
					images=images,
					all_class_embeds=all_class_embeds,
					label_vectors=label_vectors,
					criterion=criterion,
					temperature=temperature,
					loss_weights=loss_weights
				)
			
			# Check for NaN loss
			if torch.isnan(total_loss):
				print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
				continue
			
			scaler.scale(total_loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			# Track losses
			batch_loss_total = total_loss.item()
			batch_loss_i2t = loss_i2t.item()
			batch_loss_t2i = loss_t2i.item()
			
			epoch_loss_total += batch_loss_total
			epoch_loss_i2t += batch_loss_i2t
			epoch_loss_t2i += batch_loss_t2i
			num_batches += 1
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(
					f"\t\tBatch [{bidx + 1}/{len(train_loader)}] "
					f"Total Loss: {batch_loss_total:.6f} "
					f"(I2T: {batch_loss_i2t:.6f}, T2I: {batch_loss_t2i:.6f})"
				)
		
		# Calculate average losses
		avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
		avg_i2t_loss = epoch_loss_i2t / num_batches if num_batches > 0 else 0.0
		avg_t2i_loss = epoch_loss_t2i / num_batches if num_batches > 0 else 0.0
		
		training_losses.append(avg_total_loss)
		training_losses_breakdown["total"].append(avg_total_loss)
		training_losses_breakdown["i2t"].append(avg_i2t_loss)
		training_losses_breakdown["t2i"].append(avg_t2i_loss)

		print(f">> Validating Epoch {epoch+1} ...")
		
		# Compute validation loss using the same multi-label loss function
		current_val_loss = compute_multilabel_validation_loss(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			all_class_embeds=all_class_embeds,  # Reuse pre-encoded embeddings
			temperature=temperature,
			max_batches=10
		)

		# Get comprehensive validation metrics
		validation_results = get_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,  # Now uses BCEWithLogitsLoss
			device=device,
			topK_values=topk_values,
			finetune_strategy=mode,
			cache_dir=results_dir,
			verbose=True,
			lora_params={
				"lora_rank": lora_rank,
				"lora_alpha": lora_alpha,
				"lora_dropout": lora_dropout,
			},
			max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
			is_training=True,
			model_hash=get_model_hash(model),
			temperature=temperature,
		)
		
		in_batch_loss_acc_metrics_per_epoch = validation_results["in_batch_metrics"]
		in_batch_loss_acc_metrics_per_epoch["val_loss"] = current_val_loss  # Use computed validation loss
		full_val_loss_acc_metrics_per_epoch = validation_results["full_metrics"]
		retrieval_metrics_per_epoch = {
			"img2txt": validation_results["img2txt_metrics"],
			"txt2img": validation_results["txt2img_metrics"]
		}

		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		full_val_loss_acc_metrics_all_epochs.append(full_val_loss_acc_metrics_per_epoch)
		img2txt_metrics_all_epochs.append(retrieval_metrics_per_epoch["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics_per_epoch["txt2img"])
		
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}:\n'
			f'\t\tTraining - Total: {avg_total_loss:.6f} (I2T: {avg_i2t_loss:.6f}, T2I: {avg_t2i_loss:.6f})\n'
			f'\t\tValidation: {current_val_loss:.6f}\n'
			f'\tMulti-label Validation Metrics:\n'
			f'\t\tIn-batch Top-K Accuracy:\n'
			f'\t\t\t[Image→Text]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}\n'
			f'\t\tFull Validation Set:\n'
			f'\t\t\t[Image→Text]: {full_val_loss_acc_metrics_per_epoch.get("img2txt_topk_acc")}\n'
			f'\t\t\t[Text→Image]: {full_val_loss_acc_metrics_per_epoch.get("txt2img_topk_acc")}'
		)
		
		if full_val_loss_acc_metrics_per_epoch.get("hamming_loss") is not None:
			print(f'\tMulti-label Metrics:')
			print(f'\t\tHamming Loss: {full_val_loss_acc_metrics_per_epoch.get("hamming_loss", "N/A"):.4f}')
			print(f'\t\tPartial Accuracy: {full_val_loss_acc_metrics_per_epoch.get("partial_acc", "N/A"):.4f}')
			print(f'\t\tF1 Score: {full_val_loss_acc_metrics_per_epoch.get("f1_score", "N/A"):.4f}')
		
		print(f"\tRetrieval Metrics:")
		print(
			f"\t\tImage-to-Text: mAP@10={retrieval_metrics_per_epoch['img2txt'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['img2txt'].get('Recall', {}).get('10', 'N/A')}"
		)
		print(
			f"\t\tText-to-Image: mAP@10={retrieval_metrics_per_epoch['txt2img'].get('mAP', {}).get('10', 'N/A')}, "
			f"Recall@10={retrieval_metrics_per_epoch['txt2img'].get('Recall', {}).get('10', 'N/A')}"
		)

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping triggered at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break

		print(f"Epoch {epoch+1} Duration [Train + Validation]: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
	
	print(f"[{mode}] Total Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	evaluation_results = evaluate_best_model(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		early_stopping=early_stopping,
		checkpoint_path=mdl_fpth,
		finetune_strategy=mode,
		device=device,
		cache_dir=results_dir,
		topk_values=topk_values,
		verbose=verbose,
		max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
	)

	# Access individual metrics
	final_metrics_in_batch = evaluation_results["in_batch_metrics"]
	final_metrics_full = evaluation_results["full_metrics"]
	final_img2txt_metrics = evaluation_results["img2txt_metrics"]
	final_txt2img_metrics = evaluation_results["txt2img_metrics"]
	model_source = evaluation_results["model_loaded_from"]

	if verbose:
		print(f"Final evaluation used model weights from: {model_source}")
		print("--- Final Metrics [In-batch Validation] ---")
		print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
		print("--- Final Metrics [Full Validation Set] ---")
		print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
		print("--- Image-to-Text Retrieval ---")
		print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
		print("--- Text-to-Image Retrieval ---")
		print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))

	print("\nGenerating result plots...")
	actual_trained_epochs = len(training_losses)

	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"temp_{temperature}_"
		f"bs_{train_loader.batch_size}_"
		f"lor_{lora_rank}_"
		f"loa_{lora_alpha}_"
		f"lod_{lora_dropout}"
	)
	
	mdl_fpth = get_updated_model_name(
		original_path=mdl_fpth, 
		actual_epochs=actual_trained_epochs
	)

	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_multilabel_loss_breakdown(
		training_losses_breakdown=training_losses_breakdown,
		filepath=plot_paths["losses_breakdown"]
	)

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
		full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

	return final_metrics_in_batch, final_metrics_full, final_img2txt_metrics, final_txt2img_metrics

def probe_finetune_multi_label(
		model: torch.nn.Module,
		train_loader,
		validation_loader,
		num_epochs: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int,
		verbose: bool = True,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		topk_values: List[int] = [1, 3, 5, 10, 15, 20],
		loss_weights: Dict[str, float] = None,
		temperature: float = 0.07,
		label_smoothing: float = 0.0,
		volatility_threshold: float = 15.0,
		slope_threshold: float = 1e-4,
		pairwise_imp_threshold: float = 1e-4,
		use_lamb: bool = False,
		probe_hidden_dim: int = None,  # Optional: add hidden layer
		probe_dropout: float = 0.1,
		cache_features: bool = True,  # Optional: cache features for efficiency
):
		"""
		Enhanced Linear probing fine-tuning for multi-label CLIP classification with robust ViT support.
		Automatically handles different ViT architectures and fixes positional embedding issues.
		"""
		# Set default loss weights
		if loss_weights is None:
				loss_weights = {"i2t": 0.5, "t2i": 0.5}

		early_stopping = EarlyStopping(
				patience=patience,
				min_delta=min_delta,
				cumulative_delta=cumulative_delta,
				window_size=window_size,
				mode='min',
				min_epochs=minimum_epochs,
				restore_best_weights=True,
				volatility_threshold=volatility_threshold,
				slope_threshold=slope_threshold,
				pairwise_imp_threshold=pairwise_imp_threshold,
				# min_phases_before_stopping=1,
		)

		try:
				dataset_name = validation_loader.dataset.dataset.__class__.__name__
		except AttributeError:
				dataset_name = validation_loader.dataset.dataset_name

		mode = inspect.stack()[0].function
		mode = re.sub(r'_finetune_multi_label', '', mode)
		model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
		model_name = model.__class__.__name__

		print(f"{mode} | {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
		if torch.cuda.is_available():
				gpu_name = torch.cuda.get_device_name(device)
				total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
				print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

		# Get dataset information
		try:
				num_classes = len(validation_loader.dataset.unique_labels)
				class_names = validation_loader.dataset.unique_labels
		except:
				num_classes = len(validation_loader.dataset.dataset.classes)
				class_names = validation_loader.dataset.dataset.classes
		print(f"Multi-label Linear Probe fine-tuning: {num_classes} classes")

		# =====================================
		# STEP 1: FREEZE ALL CLIP PARAMETERS AND CREATE ROBUST PROBE
		# =====================================
		for param in model.parameters():
				param.requires_grad = False

		print("\nCreating robust probe model...")
		probe = get_probe_clip(
			clip_model=model,
			validation_loader=validation_loader,
			device=torch.device(device),
			# hidden_dim=256,  # Optional: creates MLP probe
			dropout=probe_dropout,
			zero_shot_init=True, # faster convergence
			verbose=True
		)
		print(f"Multi-label dataset: {isinstance(probe, MultiLabelProbe)}")

		embed_dim = probe.input_dim  # Get the detected feature dimension
		probe_params = sum(p.numel() for p in probe.parameters())
		probe_type = probe.probe_type

		print(f"CLIP embedding dimension: {embed_dim}")
		print(f"Probe type: {probe_type} | Parameters: {probe_params:,}")

		# Use BCEWithLogitsLoss for multi-label classification
		if label_smoothing > 0:
				print(f"Using label smoothing: {label_smoothing}")
				criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
		else:
				criterion = torch.nn.BCEWithLogitsLoss()
		print(f"Using {criterion.__class__.__name__} for multi-label classification")

		# Pre-encode all class texts (for evaluation)
		print(f"Pre-encoding {num_classes} class texts...")
		all_class_texts = clip.tokenize(class_names).to(device)
		with torch.no_grad():
				model.eval()
				all_class_embeds = model.encode_text(all_class_texts)
				all_class_embeds = F.normalize(all_class_embeds, dim=-1)

		# Optimizer setup
		if use_lamb:
				optimizer = LAMB(
						params=probe.parameters(),
						lr=learning_rate,
						betas=(0.9, 0.98),
						eps=1e-6,
						weight_decay=weight_decay,
				)
		else:
				optimizer = torch.optim.AdamW(
						params=probe.parameters(),
						lr=learning_rate,
						betas=(0.9, 0.98),
						eps=1e-6,
						weight_decay=weight_decay,
				)

		scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=learning_rate,
				steps_per_epoch=len(train_loader),
				epochs=num_epochs,
				pct_start=0.1,
				anneal_strategy='cos',
		)

		scaler = torch.amp.GradScaler(device=device)

		mdl_fpth = os.path.join(
				results_dir,
				f"{mode}_"
				f"{model_arch}_"
				f"{optimizer.__class__.__name__}_"
				f"{scheduler.__class__.__name__}_"
				f"{criterion.__class__.__name__}_"
				f"probe_{probe_type}_"
				f"ieps_{num_epochs}_"
				f"lr_{learning_rate:.1e}_"
				f"wd_{weight_decay:.1e}_"
				f"temp_{temperature}_"
				f"bs_{train_loader.batch_size}_"
				f"mep_{minimum_epochs}_"
				f"pat_{patience}_"
				f"mdt_{min_delta:.1e}_"
				f"cdt_{cumulative_delta:.1e}_"
				f"vt_{volatility_threshold}_"
				f"st_{slope_threshold:.1e}_"
				f"pit_{pairwise_imp_threshold:.1e}"
				f".pth"
		)

		# Optional: Cache features for efficiency
		train_features_cache = None
		val_features_cache = None
		
		if cache_features:
				print("Pre-extracting features for efficient training...")
				
				# Extract training features
				train_features = list()
				train_labels = list()
				model.eval()
				with torch.no_grad():
						for batch_data in tqdm(train_loader, desc="Extracting train features"):
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								image_embeds = model.encode_image(images)
								image_embeds = F.normalize(image_embeds, dim=-1)
								
								train_features.append(image_embeds.cpu())
								train_labels.append(label_vectors.cpu())
				
				train_features_cache = (torch.cat(train_features, dim=0), torch.cat(train_labels, dim=0))
				
				# Extract validation features
				val_features = list()
				val_labels = list()
				with torch.no_grad():
						for batch_data in tqdm(validation_loader, desc="Extracting val features"):
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								image_embeds = model.encode_image(images)
								image_embeds = F.normalize(image_embeds, dim=-1)
								
								val_features.append(image_embeds.cpu())
								val_labels.append(label_vectors.cpu())
				
				val_features_cache = (torch.cat(val_features, dim=0), torch.cat(val_labels, dim=0))
				
				print(f"Cached features - Train: {train_features_cache[0].shape}, Val: {val_features_cache[0].shape}")
				
				# Create feature dataloaders
				from torch.utils.data import TensorDataset
				train_feature_dataset = TensorDataset(train_features_cache[0], train_features_cache[1])
				val_feature_dataset = TensorDataset(val_features_cache[0], val_features_cache[1])
				
				train_feature_loader = DataLoader(
						train_feature_dataset,
						batch_size=train_loader.batch_size,
						shuffle=True,
						num_workers=0
				)
				val_feature_loader = DataLoader(
						val_feature_dataset,
						batch_size=validation_loader.batch_size,
						shuffle=False,
						num_workers=0
				)

		training_losses = list()
		training_losses_breakdown = {"total": []}
		img2txt_metrics_all_epochs = list()
		txt2img_metrics_all_epochs = list()
		in_batch_loss_acc_metrics_all_epochs = list()
		full_val_loss_acc_metrics_all_epochs = list()
		train_start_time = time.time()

		for epoch in range(num_epochs):
				train_and_val_st_time = time.time()
				torch.cuda.empty_cache()
				probe.train()

				print(f"Epoch [{epoch + 1}/{num_epochs}]")
				
				epoch_loss_total = 0.0
				num_batches = 0
				
				# Choose data source
				data_loader = train_feature_loader if cache_features else train_loader
				
				for bidx, batch_data in enumerate(data_loader):
						if cache_features:
								# Using cached features
								image_embeds, label_vectors = batch_data
								image_embeds = image_embeds.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
						else:
								# Extract features on-the-fly
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
								
								# Extract image embeddings (frozen)
								with torch.no_grad():
										image_embeds = model.encode_image(images)
										image_embeds = F.normalize(image_embeds, dim=-1)
						
						optimizer.zero_grad(set_to_none=True)
						
						with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
								# Linear probe forward (multi-label logits)
								logits = probe(image_embeds)
								
								# Multi-label loss
								loss = criterion(logits, label_vectors)
						
						# Check for NaN loss
						if torch.isnan(loss):
								print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
								continue
						
						scaler.scale(loss).backward()
						scaler.unscale_(optimizer)
						torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
						scaler.step(optimizer)
						scaler.update()
						scheduler.step()
						
						# Track losses
						batch_loss_total = loss.item()
						epoch_loss_total += batch_loss_total
						num_batches += 1
						
						if bidx % print_every == 0 or bidx + 1 == len(data_loader):
								print(f"\t\tBatch [{bidx + 1}/{len(data_loader)}] Loss: {batch_loss_total:.6f}")
				
				# Calculate average losses
				avg_total_loss = epoch_loss_total / num_batches if num_batches > 0 else 0.0
				training_losses.append(avg_total_loss)
				training_losses_breakdown["total"].append(avg_total_loss)

				print(f">> Validating Epoch {epoch+1} ...")
				
				# Validation with probe
				probe.eval()
				val_loss = 0.0
				val_preds = list()
				val_labels_list = list()
				
				with torch.no_grad():
						data_loader = val_feature_loader if cache_features else validation_loader
						
						for batch_data in data_loader:
								if cache_features:
										image_embeds, label_vectors = batch_data
										image_embeds = image_embeds.to(device, non_blocking=True)
										label_vectors = label_vectors.to(device, non_blocking=True).float()
								else:
										if len(batch_data) == 3:
												images, _, label_vectors = batch_data
										else:
												raise ValueError(f"Expected 3 items, got {len(batch_data)}")
										
										images = images.to(device, non_blocking=True)
										label_vectors = label_vectors.to(device, non_blocking=True).float()
										
										# Extract features
										image_embeds = model.encode_image(images)
										image_embeds = F.normalize(image_embeds, dim=-1)
								
								# Get predictions from probe
								logits = probe(image_embeds)
								loss = criterion(logits, label_vectors)
								val_loss += loss.item()
								
								# Store predictions for metrics
								probs = torch.sigmoid(logits)
								preds = (probs > 0.5).float()
								val_preds.append(preds.cpu())
								val_labels_list.append(label_vectors.cpu())
				
				avg_val_loss = val_loss / len(data_loader)
				
				# Calculate multi-label metrics
				val_preds = torch.cat(val_preds, dim=0)
				val_labels = torch.cat(val_labels_list, dim=0)
				
				hamming = hamming_loss(val_labels.numpy(), val_preds.numpy())
				f1 = f1_score(val_labels.numpy(), val_preds.numpy(), average='weighted', zero_division=0)
				exact_match = (val_preds == val_labels).all(dim=1).float().mean().item()
				partial_match = (val_preds == val_labels).float().mean().item()
				
				print(f"Validation - Loss: {avg_val_loss:.6f}, Hamming: {hamming:.4f}, F1: {f1:.4f}, Exact Match: {exact_match:.4f}")
				
				# Create metrics for compatibility
				current_val_loss = avg_val_loss
				
				# Simple in-batch metrics
				in_batch_metrics = {
						"val_loss": avg_val_loss,
						"hamming_loss": hamming,
						"f1_score": f1,
						"exact_match_acc": exact_match,
						"partial_acc": partial_match,
				}
				in_batch_loss_acc_metrics_all_epochs.append(in_batch_metrics)
				full_val_loss_acc_metrics_all_epochs.append(in_batch_metrics)

				if early_stopping.should_stop(
						current_value=current_val_loss,
						model=probe,  # Save probe weights
						epoch=epoch,
						optimizer=optimizer,
						scheduler=scheduler,
						checkpoint_path=mdl_fpth,
				):
						print(f"\nEarly stopping at epoch {epoch + 1}")
						break

				print(f"Epoch {epoch+1} Duration: {time.time() - train_and_val_st_time:.2f} sec".center(150, "="))
		
		print(f"[{mode}] Total Time: {time.time() - train_start_time:.1f} sec".center(170, "-"))

		# Load best probe weights
		if os.path.exists(mdl_fpth):
				print(f"Loading best probe weights from {mdl_fpth}")
				checkpoint = torch.load(mdl_fpth, map_location=device)
				if 'model_state_dict' in checkpoint:
						probe.load_state_dict(checkpoint['model_state_dict'])
				else:
						probe.load_state_dict(checkpoint)

		# Final evaluation
		print("\nFinal Evaluation:")
		probe.eval()
		final_preds = list()
		final_labels = list()
		
		with torch.no_grad():
				data_loader = val_feature_loader if cache_features else validation_loader
				
				for batch_data in data_loader:
						if cache_features:
								image_embeds, label_vectors = batch_data
								image_embeds = image_embeds.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
						else:
								if len(batch_data) == 3:
										images, _, label_vectors = batch_data
								else:
										raise ValueError(f"Expected 3 items, got {len(batch_data)}")
								
								images = images.to(device, non_blocking=True)
								label_vectors = label_vectors.to(device, non_blocking=True).float()
								
								image_embeds = model.encode_image(images)
								image_embeds = F.normalize(image_embeds, dim=-1)
						
						logits = probe(image_embeds)
						probs = torch.sigmoid(logits)
						preds = (probs > 0.5).float()
						
						final_preds.append(preds.cpu())
						final_labels.append(label_vectors.cpu())
		
		final_preds = torch.cat(final_preds, dim=0)
		final_labels = torch.cat(final_labels, dim=0)
		
		# Final metrics
		final_hamming = hamming_loss(final_labels.numpy(), final_preds.numpy())
		final_f1 = f1_score(final_labels.numpy(), final_preds.numpy(), average='weighted', zero_division=0)
		final_exact = (final_preds == final_labels).all(dim=1).float().mean().item()
		final_partial = (final_preds == final_labels).float().mean().item()
		best_val_loss = early_stopping.get_best_score() or 0.0

		print("\n" + "="*80)
		print("ENHANCED LINEAR PROBE MULTI-LABEL TRAINING SUMMARY")
		print("="*80)
		print(f"Method: {mode}")
		print(f"Model: {getattr(model, 'name', 'Unknown')}")
		print(f"Probe Type: {probe_type}")
		print(f"Probe Parameters: {probe_params:,}")
		print(f"CLIP Parameters (frozen): {sum(p.numel() for p in model.parameters()):,}")
		print(f"Total Epochs: {len(training_losses)}")
		print(f"Best Val Loss: {best_val_loss}")
		print(f"Best Epoch: {early_stopping.get_best_epoch() + 1}")
		print("-"*80)
		print("Final Metrics:")
		print(f"  Hamming Loss: {final_hamming:.4f}")
		print(f"  F1 Score: {final_f1:.4f}")
		print(f"  Exact Match: {final_exact:.4f}")
		print(f"  Partial Match: {final_partial:.4f}")
		print("="*80)

		evaluation_results = evaluate_best_model(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				early_stopping=early_stopping,
				checkpoint_path=mdl_fpth,
				finetune_strategy=mode,
				device=device,
				cache_dir=results_dir,
				topk_values=topk_values,
				verbose=verbose,
				max_in_batch_samples=get_max_samples(batch_size=validation_loader.batch_size, N=10, device=device),
		)

		# Access individual metrics
		final_metrics_in_batch = evaluation_results["in_batch_metrics"]
		final_metrics_full = evaluation_results["full_metrics"]
		final_img2txt_metrics = evaluation_results["img2txt_metrics"]
		final_txt2img_metrics = evaluation_results["txt2img_metrics"]

		if verbose:
				print(f"Final evaluation used model weights from: {evaluation_results['model_loaded_from']}")
				print("--- Final Metrics [In-batch Validation] ---")
				print(json.dumps(final_metrics_in_batch, indent=2, ensure_ascii=False))
				print("--- Final Metrics [Full Validation Set] ---")
				print(json.dumps(final_metrics_full, indent=2, ensure_ascii=False))
				print("--- Image-to-Text Retrieval ---")
				print(json.dumps(final_img2txt_metrics, indent=2, ensure_ascii=False))
				print("--- Text-to-Image Retrieval ---")
				print(json.dumps(final_txt2img_metrics, indent=2, ensure_ascii=False))

		print("\nGenerating result plots...")
		actual_trained_epochs = len(training_losses)

		file_base_name = (
				f"{dataset_name}_"
				f"{mode}_"
				f"{optimizer.__class__.__name__}_"
				f"{scheduler.__class__.__name__}_"
				f"{criterion.__class__.__name__}_"
				f"{scaler.__class__.__name__}_"
				f"{model_name}_"
				f"{model_arch}_"
				f"ep_{actual_trained_epochs}_"
				f"lr_{learning_rate:.1e}_"
				f"wd_{weight_decay:.1e}_"
				f"temp_{temperature}_"
				f"bs_{train_loader.batch_size}"
		)
		
		# Update model path
		mdl_fpth = get_updated_model_name(
				original_path=mdl_fpth, 
				actual_epochs=actual_trained_epochs
		)
		
		print(f"Model renamed to: {mdl_fpth}")

		# Plotting
		plot_paths = {
				"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
				"losses_breakdown": os.path.join(results_dir, f"{file_base_name}_losses_breakdown.png"),
				"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
				"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
				"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
				"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
				"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
				"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
		}

		plot_multilabel_loss_breakdown(
				training_losses_breakdown=training_losses_breakdown,
				filepath=plot_paths["losses_breakdown"]
		)

		plot_loss_accuracy_metrics(
				dataset_name=dataset_name,
				train_losses=training_losses,
				val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				full_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in full_val_loss_acc_metrics_all_epochs],
				losses_file_path=plot_paths["losses"],
				in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
				in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
				full_topk_val_acc_i2t_fpth=plot_paths["full_val_topk_i2t"],
				full_topk_val_acc_t2i_fpth=plot_paths["full_val_topk_t2i"],
		)

		plot_retrieval_metrics_per_epoch(
				dataset_name=dataset_name,
				image_to_text_metrics_list=img2txt_metrics_all_epochs,
				text_to_image_metrics_list=txt2img_metrics_all_epochs,
				fname=plot_paths["retrieval_per_epoch"],
		)

		plot_retrieval_metrics_best_model(
				dataset_name=dataset_name,
				image_to_text_metrics=final_img2txt_metrics,
				text_to_image_metrics=final_txt2img_metrics,
				fname=plot_paths["retrieval_best"],
		)

		return in_batch_loss_acc_metrics_all_epochs

def compute_multilabel_contrastive_loss(
		model: torch.nn.Module,
		images: torch.Tensor,
		all_class_embeds: torch.Tensor,
		label_vectors: torch.Tensor,
		criterion: torch.nn.Module,
		temperature: float = 0.07,
		loss_weights: Dict[str, float] = None
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Compute bidirectional multi-label contrastive loss.
	
	Args:
			model: CLIP model
			images: [batch_size, 3, 224, 224]
			all_class_embeds: [num_classes, embed_dim] - pre-computed text embeddings
			label_vectors: [batch_size, num_classes] - binary label matrix
			criterion: Loss function (BCEWithLogitsLoss)
			temperature: Temperature scaling for similarities
			loss_weights: Weights for I2T and T2I losses
			
	Returns:
			Tuple of (total_loss, i2t_loss, t2i_loss)
	"""
	if loss_weights is None:
		loss_weights = {"i2t": 0.5, "t2i": 0.5}
	
	batch_size, num_classes = label_vectors.shape
	
	# Encode images
	image_embeds = model.encode_image(images)  # [batch_size, embed_dim]
	
	# Normalize embeddings
	image_embeds = F.normalize(image_embeds, dim=-1)
	all_class_embeds = F.normalize(all_class_embeds, dim=-1)
	
	# ================================
	# Image-to-Text Loss
	# ================================
	# Compute similarity matrix: [batch_size, num_classes]
	i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
	
	# I2T targets: label_vectors directly [batch_size, num_classes]
	i2t_targets = label_vectors.float()
	
	# Compute I2T loss
	loss_i2t = criterion(i2t_similarities, i2t_targets)
	
	# ================================
	# Text-to-Image Loss  
	# ================================
	# Compute similarity matrix: [num_classes, batch_size]
	t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature
	
	# T2I targets: transpose of label_vectors [num_classes, batch_size]
	t2i_targets = label_vectors.T.float()
	
	# Compute T2I loss
	loss_t2i = criterion(t2i_similarities, t2i_targets)
	
	# ================================
	# Combine losses
	# ================================
	total_loss = loss_weights["i2t"] * loss_i2t + loss_weights["t2i"] * loss_t2i
	
	return total_loss, loss_i2t, loss_t2i

class LabelSmoothingBCELoss(torch.nn.Module):
	def __init__(self, smoothing: float = 0.1):
		super().__init__()
		self.smoothing = smoothing
			
	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""
		Args:
				logits: [batch_size, num_classes] - raw logits
				targets: [batch_size, num_classes] - binary targets (0 or 1)
		"""
		# Apply label smoothing
		# Positive labels: 1 -> (1 - smoothing)
		# Negative labels: 0 -> smoothing
		smooth_targets = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
		
		# Apply BCE loss with logits
		loss = F.binary_cross_entropy_with_logits(logits, smooth_targets)
		return loss

def train(
		model:torch.nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int,
		print_every:int,
		learning_rate:float,
		weight_decay:float,
		device:torch.device,
		results_dir: str,
		window_size: int,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		topk_values:List[int]=[1, 5, 10, 15, 20],
	):

	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min', # Monitoring validation loss
		min_epochs=minimum_epochs,
		restore_best_weights=True,
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__ # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function	
	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) | batch_size: {train_loader.batch_size} | {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	dropout_val = None
	for name, module in model.named_modules():
		# print(f"{name}: {type(module).__name__}")
		if isinstance(module, torch.nn.Dropout):
			# print(f"{name}.p: {module.p}")
			dropout_val = module.p
			break
	if dropout_val is None:
		dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found (unlikely in your case)

	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers (train from scratch) initialized with random weights
		# print(f"{name} requires_grad: {param.requires_grad}")

	get_parameters_info(model=model, mode=mode)

	optimizer = torch.optim.AdamW(
		params=[p for p in model.parameters() if p.requires_grad], # Only optimizes parameters that require gradients
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1, # percentage of the cycle (in number of steps) spent increasing the learning rate
		anneal_strategy='cos', # cos/linear annealing
	)
	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	mdl_fpth = os.path.join(
		results_dir,
		# f"{dataset_name}_"
		f"{mode}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"do_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}"
		f".pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = list()
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list()
	train_start_time = time.time()
	# print(torch.cuda.memory_summary(device=device))
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	for epoch in range(num_epochs):
		torch.cuda.empty_cache() # Clear GPU memory cache
		model.train() # dropout is active, units are dropped with specified probability (e.g., p=0.1)
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			# torch.Size([batch_size, 3, 224, 224]), torch.Size([batch_size, 77]), torch.Size([batch_size])
			# print(bidx, images.shape, tokenized_labels.shape, labels_indices.shape)
			optimizer.zero_grad() # Clear gradients from previous batch
			images = images.to(device, non_blocking=True) # torch.Size([b, 3, 224, 224]),
			tokenized_labels = tokenized_labels.to(device, non_blocking=True) # torch.Size([b, 77])
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()): # # Automatic Mixed Precision (AMP) backpropagation:
				logits_per_image, logits_per_text = model(images, tokenized_labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)
			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # stabilize training if exploding gradients
			scaler.step(optimizer)
			scaler.update()
			scheduler.step() # Update learning rate
			if bidx%print_every==0 or bidx+1==len(train_loader):
				print(
					f"\t\tBatch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)

		# Compute traditional loss/accuracy metrics on validation set
		in_batch_loss_acc_metrics_per_epoch = get_in_batch_validation_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topk_values,
		)
		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		print(
			f'@ Epoch {epoch+1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {in_batch_loss_acc_metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_acc")}'
		)

		# Compute retrieval-based metrics
		retrieval_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=topk_values,
		)
		img2txt_metrics_all_epochs.append(retrieval_metrics["img2txt"])
		txt2img_metrics_all_epochs.append(retrieval_metrics["txt2img"])

		# ############################## Early stopping ##############################
		current_val_loss = in_batch_loss_acc_metrics_per_epoch["val_loss"]

		if hasattr(train_loader.dataset, 'get_cache_stats'):
			print(f"#"*100)
			cache_stats = train_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Train Cache Stats: {cache_stats}")

		if hasattr(validation_loader.dataset, 'get_cache_stats'):
			cache_stats = validation_loader.dataset.get_cache_stats()
			if cache_stats is not None:
				print(f"Validation Cache Stats: {cache_stats}")
			print(f"#"*100)

		# Early stopping check
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			optimizer=optimizer,
			scheduler=scheduler,
			checkpoint_path=mdl_fpth,
		):
			print(
				f"\nEarly stopping at epoch {epoch + 1} "
				f"with best loss: {early_stopping.get_best_score()} "
				f"obtained in epoch {early_stopping.get_best_epoch()+1}")
			break
		# ############################## Early stopping ##############################
		print("-"*170)

	print(f"Elapsed_t: {time.time()-train_start_time:.1f} sec".center(170, "-"))
	file_base_name = (
		f"{dataset_name}_"
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{len(training_losses)}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"do_{dropout_val}"
	)

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_batch_topk_i2t_acc.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_batch_topk_t2i_acc.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_i2t_acc.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_t2i_acc.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_i2t_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		in_batch_topk_val_accuracy_t2i_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		in_batch_topk_val_acc_i2t_fpth=plot_paths["in_batch_val_topk_i2t"],
		in_batch_topk_val_acc_t2i_fpth=plot_paths["in_batch_val_topk_t2i"],
		mean_reciprocal_rank_file_path=plot_paths["mrr"],
		cosine_similarity_file_path=plot_paths["cs"],
	)

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_all_epochs,
		text_to_image_metrics_list=txt2img_metrics_all_epochs,
		fname=retrieval_metrics_fpth,
	)

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=final_img2txt_metrics,
		text_to_image_metrics=final_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def pretrain(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		device: torch.device,
		results_dir: str,
		cache_dir: str=None,
		topk_values: List=[1, 3, 5],
		verbose:bool=True,
		embeddings_cache=None,
	):
	model_name = model.__class__.__name__
	model_arch = re.sub(r"[/@]", "_", model.name)
	if cache_dir is None:
		cache_dir = results_dir
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name
	if verbose:
		print(f"Pretrain Evaluation {dataset_name} {model_name} - {model_arch} {device}".center(170, "-"))

	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		criterion=torch.nn.CrossEntropyLoss(),
		device=device,
		topK_values=topk_values,
		cache_dir=cache_dir,
		verbose=True,
		embeddings_cache=embeddings_cache,
		is_training=False,
		model_hash=get_model_hash(model),
	)
	# in_batch_metrics = validation_results["in_batch_metrics"]
	# full_metrics = validation_results["full_metrics"]
	retrieval_metrics = {
		"img2txt": validation_results["img2txt_metrics"],
		"txt2img": validation_results["txt2img_metrics"]
	}
	img2txt_metrics = retrieval_metrics["img2txt"]
	txt2img_metrics = retrieval_metrics["txt2img"]

	if verbose:
		print("Image to Text Metrics: ")
		print(json.dumps(img2txt_metrics, indent=2, ensure_ascii=False))
		print("Text to Image Metrics: ")
		print(json.dumps(txt2img_metrics, indent=2, ensure_ascii=False))

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{dataset_name}_pretrained_{model_name}_{model_arch}_retrieval_metrics_img2txt_txt2img.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=img2txt_metrics,
		text_to_image_metrics=txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
		best_model_name=f"Pretrained {model_name} {model_arch}",
	)

	return img2txt_metrics, txt2img_metrics