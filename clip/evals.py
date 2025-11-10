from utils import *

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
		temperature: float = 0.07,
		verbose: bool = False,
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
	if verbose:
		print(f"Pre-encoding class {len(class_names)} texts => {type(all_class_texts)} {all_class_texts.shape}")
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
		verbose: bool = False,
	) -> Dict:

	model.eval()
	total_loss = 0.0
	processed_batches = 0
	total_samples = 0
	
	# Check if this is multi-label by inspecting the dataset
	sample_batch = next(iter(validation_loader))
	is_multilabel = len(sample_batch) == 3 and len(sample_batch[2].shape) == 2
	
	if is_multilabel:
		if verbose:
			print("Multi-label dataset detected - skipping in-batch metrics computation")
		multi_label_in_batch_metrics = compute_multilabel_inbatch_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=topK_values,
			max_samples=max_samples,
			temperature=temperature,
			verbose=verbose,
		)
		return multi_label_in_batch_metrics

	if verbose:
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
			verbose=verbose,
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
