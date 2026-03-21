from utils import *

if USER == "farid":
	from visualize import build_arch_flowchart

def check_lora_weight_health(model, epoch, verbose=True):
		issues = []
		stats = {"A": {}, "B": {}}
		for name, param in model.named_parameters():
				if not param.requires_grad:
						continue
				group = "A" if "lora_A" in name else "B" if "lora_B" in name else None
				if group is None:
						continue
				has_nan = torch.isnan(param.data).any().item()
				has_inf = torch.isinf(param.data).any().item()
				norm = param.data.norm().item()
				if has_nan or has_inf:
						issues.append(f"  ✗ {name}: nan={has_nan} inf={has_inf} norm={norm:.4e}")
				stats[group][name] = norm
		
		A_norms = list(stats["A"].values())
		B_norms = list(stats["B"].values())
		
		if verbose:
				print(f"\n[Weight Health — Epoch {epoch+1}]")
				if A_norms:
						print(f"  lora_A norms — min={min(A_norms):.4e} max={max(A_norms):.4e} mean={sum(A_norms)/len(A_norms):.4e}")
				if B_norms:
						print(f"  lora_B norms — min={min(B_norms):.4e} max={max(B_norms):.4e} mean={sum(B_norms)/len(B_norms):.4e}")
				if issues:
						print(f"  !! {len(issues)} corrupted tensors:")
						for issue in issues[:10]:  # cap at 10
								print(issue)
				else:
						print(f"  ✓ All weights healthy")
		
		return len(issues) == 0, A_norms, B_norms

def check_training_health(
		model,
		epoch,
		mode,
		training_losses,
		validation_losses,
		align_score,
		temperature,        # ← pass in so the abort message can report it
		learning_rate,      # ← pass in for diagnostic message
		verbose=True,
):
		issues = []

		# ── Signal 1: Loss flatline (universal) ──────────────────────────────
		if len(training_losses) >= 2:
				loss_delta = abs(training_losses[0] - training_losses[-1])
				relative_delta = loss_delta / (training_losses[0] + 1e-8)
				if relative_delta < 0.005:
						issues.append(
								f"Loss flatline: {training_losses[0]:.6f} → "
								f"{training_losses[-1]:.6f} "
								f"({relative_delta*100:.3f}% change)"
						)

		# ── Signal 2: AlignScore frozen (universal) ───────────────────────────
		if (
				align_score is not None
				and align_score == align_score  # not NaN
				and epoch >= 1
				and align_score < 0.005
		):
				issues.append(
						f"AlignScore@5 critically low: {align_score:.6f}"
				)

		# ── Signal 3: Method-specific parameter movement ──────────────────────
		mode_lower = mode.lower()

		if "probe" in mode_lower:
				# Probe W should diverge from zero-shot initialisation
				w_norms = [
						p.data.norm().item()
						for n, p in model.named_parameters()
						if p.requires_grad and "weight" in n
				]
				if w_norms and max(w_norms) < 1e-03:
						issues.append(
								f"Linear probe W has not moved: max norm={max(w_norms):.2e}"
						)

		elif "full" in mode_lower:
				# Full FT — check that at least some vision layers have nonzero gradients
				# Use weight norm change as proxy since gradients are only available
				# during backward — compare to pretrained norms which should be ~O(1)
				# This signal is weak for Full FT so we skip Signal 3 and rely on 1+2
				pass  # Signals 1 and 2 are sufficient for Full FT

		elif "lora" in mode_lower or "dora" in mode_lower:
				B_norms = [
						p.data.norm().item()
						for n, p in model.named_parameters()
						if p.requires_grad and "lora_B" in n
				]
				if B_norms:
						mean_B = sum(B_norms) / len(B_norms)
						if mean_B < 1e-03:
								issues.append(
										f"LoRA B matrices static: mean norm={mean_B:.2e} < 1e-03"
								)

		elif "vera" in mode_lower:
				lb_vals = [
						p.data.abs().mean().item()
						for n, p in model.named_parameters()
						if p.requires_grad and "lambda_b" in n
				]
				if lb_vals:
						lb_mean = sum(lb_vals) / len(lb_vals)
						if lb_mean < 5e-04:
								issues.append(
										f"VeRA λ_b static: mean|λ_b|={lb_mean:.2e} < 5e-04"
								)

		elif "ia3" in mode_lower:
				deltas = [
						(p.data - 1.0).abs().mean().item()
						for n, p in model.named_parameters()
						if p.requires_grad and "scaling" in n
				]
				if deltas:
						delta_mean = sum(deltas) / len(deltas)
						if delta_mean < 5e-04:
								issues.append(
										f"IA³ scaling vectors static: "
										f"mean|s-1|={delta_mean:.2e} < 5e-04"
								)

		elif "adapter" in mode_lower or "tip" in mode_lower:
				adapter_norms = [
						p.data.norm().item()
						for n, p in model.named_parameters()
						if p.requires_grad
				]
				if adapter_norms:
						mean_norm = sum(adapter_norms) / len(adapter_norms)
						if mean_norm < 1e-03:
								issues.append(
										f"Adapter weights static: mean norm={mean_norm:.2e} < 1e-03"
								)

		# ── Decision ─────────────────────────────────────────────────────────
		should_abort = len(issues) >= 2

		if verbose:
				print(f"\n{'─'*60}")
				print(f"[Training Health Check — Epoch {epoch+1} | {mode.upper()}]")
				print(f"  Config: temperature={temperature} lr={learning_rate:.1e}")
				if issues:
						for issue in issues:
								print(f"  ⚠  {issue}")
				else:
						print(f"  ✓  All signals healthy — training proceeding normally")

				if should_abort:
						print(f"\n  ❌ ABORT: {len(issues)}/3 signals indicate broken gradient.")
						print(f"  Most likely causes given your config:")
						if temperature >= 0.5:
								print(f"    → temperature={temperature} is too high for "
											f"4667-class L2-normalised BCE — use 0.07")
						if learning_rate < 1e-05:
								print(f"    → learning_rate={learning_rate:.1e} may be too low")
						print(f"  Aborting to save GPU time.")
				else:
						print(f"  ✓  Training healthy — continuing.")
				print(f"{'─'*60}\n")

		return should_abort

def compute_tiered_retrieval_metrics(
	similarity_matrix: torch.Tensor,
	query_labels: torch.Tensor,
	topK_values: List[int],
	head_mask: torch.Tensor,
	rare_mask: torch.Tensor,
	active_mask: torch.Tensor,
	mode: str = "Image-to-Text",
	min_val_support: int = 10,
	verbose: bool = False,
) -> Dict:
	if verbose:
		print(f"[{mode}]")
		print(f"  ├─ Similarity matrix: {similarity_matrix.shape} {similarity_matrix.device}")
		print(f"  ├─ Query labels: {query_labels.shape} {query_labels.device}")
		print(f"  ├─ Head mask: {head_mask.shape} {head_mask.device}")
		print(f"  ├─ Rare mask: {rare_mask.shape} {rare_mask.device}")
		print(f"  └─ Active mask: {active_mask.shape} {active_mask.device}")

	tiers = {
		"overall": active_mask,
		"head":    head_mask & active_mask,
		"rare":    rare_mask & active_mask,
	}
	results = {}
	for tier_name, tier_mask in tiers.items():
		tier_indices = torch.where(tier_mask)[0]
		
		if mode == "Image-to-Text":
			# Queries are images, candidates are text (one per class)
			tier_sim = similarity_matrix[:, tier_indices] # [N_images, N_tier]
			tier_query_labels = query_labels[:, tier_indices] # [N_images, N_tier]
			tier_candidate_labels = torch.arange(len(tier_indices), device=similarity_matrix.device) # [N_tier] — class identity
		else:  # Text-to-Image
			# Queries are text (one per class), candidates are images
			if tier_name == "rare":
				val_support = query_labels[:, tier_indices].sum(dim=0)  # [N_tier]
				supported = val_support >= min_val_support
				tier_indices = tier_indices[supported]
			
			tier_sim = similarity_matrix[tier_indices, :]          # [N_tier, N_images]
			tier_query_labels = torch.arange(len(tier_indices), device=similarity_matrix.device) # [N_tier] — class identity
			
			# slice image labels to tier classes only
			tier_candidate_labels = query_labels[:, tier_indices]   # [N_images, N_tier]
		
		tier_metrics = compute_retrieval_metrics_from_similarity(
			similarity_matrix=tier_sim,
			query_labels=tier_query_labels,
			candidate_labels=tier_candidate_labels,
			topK_values=topK_values,
			mode=mode,
			verbose=verbose,
		)
		
		results[tier_name] = tier_metrics
		
		if verbose:
			print(
				f"  [{tier_name.upper():8s}] "
				f"mAP@10={tier_metrics['mAP'].get('10', 0):.4f}  "
				f"R@10={tier_metrics['Recall'].get('10', 0):.4f}  "
				f"({tier_indices.shape[0]} classes)"
			)
	
	return results

def compute_multilabel_validation_loss(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	criterion_i2t,      # BCEWithLogitsLoss with pos_weight, reduction='none'
	criterion_t2i,      # BCEWithLogitsLoss plain, reduction='none'
	active_mask,        # [num_classes] bool
	device: str,
	all_class_embeds: torch.Tensor,
	temperature: float,
	verbose: bool = False,
) -> float:
	model.eval()
	total_loss = 0.0
	total_samples = 0
	
	max_batches = max(50, len(validation_loader) // 10)
	if verbose:
		print(f"\nMultilabel validation loss:")
		print(f"  {type(model)} {model.name}")
		print(f"  {validation_loader.name} {len(validation_loader)} batches")
		print(f"  max_batches: {max_batches}")
		print(f"  active_mask: {active_mask.shape} {active_mask.sum()}/{len(active_mask)}")
		print(f"  all_class_embeds: {all_class_embeds.shape} {all_class_embeds.device}")

	with torch.no_grad():
		for batch_idx, (images, _, label_vectors) in enumerate(validation_loader):
			if batch_idx >= max_batches:
				break
			
			batch_size = images.size(0)
			if batch_size == 0:  # Skip empty batches
				continue
					
			images = images.to(device, non_blocking=True)
			label_vectors = label_vectors.to(device, non_blocking=True).float()
			
			# Encode images
			image_embeds = model.encode_image(images)
			image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
			
			# Compute similarities
			i2t_similarities = torch.matmul(image_embeds, all_class_embeds.T) / temperature
			t2i_similarities = torch.matmul(all_class_embeds, image_embeds.T) / temperature
			
			# Compute losses
			i2t_targets = label_vectors
			t2i_targets = label_vectors.T
			
			i2t_loss_raw = criterion_i2t(i2t_similarities, i2t_targets) # [B, C]
			loss_i2t = i2t_loss_raw[:, active_mask].mean()

			t2i_loss_raw = criterion_t2i(t2i_similarities, t2i_targets) # [C, B]
			loss_t2i = t2i_loss_raw[active_mask, :].mean()

			batch_loss = 0.5 * (loss_i2t + loss_t2i)
			
			# Correct accumulation
			total_loss += batch_loss.item() * batch_size
			total_samples += batch_size
	
	avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

	return avg_loss

def chunked_similarity_computation(
		query_embeddings: torch.Tensor,
		candidate_embeddings: torch.Tensor,
		temperature: float,
		chunk_size: int = 1000,
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
		chunk_size: int = 1000,
		verbose: bool = False,
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
						# candidate_labels: [N_images, N_tier] — already sliced to tier classes
						# num_queries == N_tier (one query per class)
						relevant_counts = candidate_labels.sum(dim=0).float()        # [num_queries]
						retrieved_counts = correct_mask.float().sum(dim=1)           # [num_queries]
						valid = relevant_counts > 0
						recall_per_query = torch.where(
								valid,
								retrieved_counts / relevant_counts.clamp(min=1),
								torch.zeros_like(retrieved_counts),
						)
						metrics["Recall"][str(K)] = recall_per_query[valid].mean().item() if valid.any() else 0.0
				else:
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
	device: str,
	topK_values: List[int],
	cache_dir: str,
	temperature: float,
	finetune_strategy: str = None,
	chunk_size: int = 1024,
	force_recompute: bool = False,
	embeddings_cache: tuple = None,
	lora_params: Optional[Dict] = None,
	is_training: bool = False,
	model_hash: str = None,
	class_embeds_override: Optional[torch.Tensor] = None,
	verbose: bool = True,
) -> Dict:

	if verbose:
		print("\nComputing validation metrics")
		print(f"└─ Temperatur: {temperature}")

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
	
	if verbose:
		print(f"Dataset: {dataset_name}, Label(s): {n_classes}, Samples: {num_samples}")
	
	cache_file = os.path.join(
		cache_dir,
		f"{dataset_name}_{finetune_strategy}_bs_{validation_loader.batch_size}_"
		f"nw_{num_workers}_{model_class_name}_{model_arch_name.replace('/', '_')}_"
		f"validation_embeddings.pt"
	)

	if model_hash:
		cache_file = cache_file.replace(".pt", f"_{model_hash}.pt")
	
	# Step 2: Load or compute embeddings
	cache_loaded = False
	
	# Try to use provided cache first
	if not is_training and embeddings_cache is not None:
		all_image_embeds, _ = embeddings_cache
		all_labels = _prepare_labels_tensor(validation_loader, num_samples, n_classes, device)
		cache_loaded = True
		if verbose:
			print(f"Embeddings cache loaded from provided embeddings_cache: {type(embeddings_cache)}")	
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
			print("Computing embeddings from scratch [takes a while] ...")
		t0 = time.time()
		all_image_embeds, all_labels = _compute_image_embeddings(
			model=model, 
			validation_loader=validation_loader, 
			device=device,
			verbose=verbose,
		)
		if verbose:
			print(f"Elapsed: {time.time() - t0:.1f} s")
			
	# Step 3: Compute class embeddings
	if class_embeds_override is not None:
		# Use probe's trained W instead of frozen text encoder
		class_text_embeds = torch.nn.functional.normalize(class_embeds_override, dim=-1).to(device).float()
		if verbose:
			print(f"class_text_embeds [probe W override]: {class_text_embeds.shape} {class_text_embeds.dtype} {class_text_embeds.device}")
	else:
		# Standard path: encode class names with frozen text encoder
		text_batch_size = validation_loader.batch_size
		if verbose:
			print(f"Pre-encoding {n_classes} classes in batch_size: {text_batch_size}")
		class_text_embeds = []
		model.eval()
		with torch.no_grad():
			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
				for i in range(0, n_classes, text_batch_size):
					end_idx = min(i + text_batch_size, n_classes)
					batch_class_names = class_names[i:end_idx]
					batch_class_texts = clip.tokenize(batch_class_names).to(device)
					batch_embeds = model.encode_text(batch_class_texts)
					batch_embeds = torch.nn.functional.normalize(batch_embeds, dim=-1)
					class_text_embeds.append(batch_embeds.cpu())
					del batch_class_texts, batch_embeds
					torch.cuda.empty_cache()
		class_text_embeds = torch.cat(class_text_embeds, dim=0).to(device)
		if verbose:
			print(f"class_text_embeds: {type(class_text_embeds)} {class_text_embeds.shape} {class_text_embeds.dtype} {class_text_embeds.device}")

	# Step 4: Compute similarity matrices (chunked for memory efficiency)
	device_image_embeds = all_image_embeds.to(device, non_blocking=True)
	device_class_text_embeds = class_text_embeds.to(device, non_blocking=True)
	device_labels = all_labels.to(device, non_blocking=True)

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
	if verbose:
		print(f"Similarity matrices: I2T {i2t_similarity.shape}, T2I {t2i_similarity.shape}")

	# Step 5: Compute full-set metrics
	full_metrics = compute_full_set_metrics_from_cache(
		i2t_similarity=i2t_similarity,
		t2i_similarity=t2i_similarity,
		labels=device_labels,
		n_classes=n_classes,
		topK_values=topK_values,
		device=device,
		device_image_embeds=device_image_embeds,
		device_class_text_embeds=device_class_text_embeds,
		temperature=temperature,
		chunk_size=chunk_size,
		verbose=verbose,
	)
	
	# Step 6: Compute retrieval metrics
	cache_key_base = f"{dataset_name}_{finetune_strategy}_{model_class_name}_{model_arch_name.replace('/', '_')}"
	if lora_params:
		lora_rank = lora_params.get("lora_rank")
		lora_alpha = lora_params.get("lora_alpha")
		lora_dropout = lora_params.get("lora_dropout")
		# LoRA+ has additional parameters
		lora_plus_lambda = lora_params.get("lora_plus_lambda", None)
		cache_key_base += f"_lora_r_{lora_rank}_a_{lora_alpha}_d_{lora_dropout}"
		if lora_plus_lambda is not None:
			cache_key_base += f"_lmbd_{lora_plus_lambda}"
	
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
		# print(f"{type(model)} I2T: {type(img2txt_metrics)} T2I: {type(txt2img_metrics)}")
		print(f"\nValidation Elapsed Time: {time.time() - start_time:.1f}s")
	
	return {
		"full_metrics": full_metrics,
		"img2txt_metrics": img2txt_metrics,
		"txt2img_metrics": txt2img_metrics,
		"i2t_similarity": i2t_similarity,
		"t2i_similarity": t2i_similarity,
		"device_labels": device_labels,
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
		temperature: float,
		chunk_size: int = 1000,
		verbose: bool = False,
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
		
		# Alignment score
		if is_multi_label:
			alignment_score = get_multilabel_alignment_score(
				image_embeds=device_image_embeds,
				all_class_embeds=device_class_text_embeds,
				labels=labels,
				temperature=temperature,
				topk=5,
				verbose=verbose,
			)
			cos_sim = None  # no longer computed for multi-label
		else:
				alignment_score = None
				cos_sim = get_matched_cosine_similarity(
						image_embeds=device_image_embeds,
						text_embeds=device_class_text_embeds,
						labels=labels,
						is_multi_label=False,
						verbose=verbose,
				)

		# Additional multi-label metrics
		hamming_loss = None
		partial_acc = None 
		f1_score_val = None
		
		if is_multi_label:
			# Get optimal threshold for predictions
			i2t_probs = torch.sigmoid(i2t_similarity)
			threshold = 0.5
			
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
				"cosine_similarity": float(cos_sim) if cos_sim is not None else None,
				"alignment_score": float(alignment_score) if alignment_score is not None else None,
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

	batch_count = 0
	with torch.no_grad():
		for images, _, labels_indices in validation_loader:
			if max_batches and batch_count >= max_batches:
				print(f"Stopping at batch {batch_count} due to max_batches limit")
				break

			if batch_count % 50 == 0:
				high_mem = monitor_memory_usage(operation_name=f"Batch {batch_count}")
				if high_mem:
					torch.cuda.empty_cache()

			images = images.to(device, non_blocking=True)
			if device.type == "cuda":
				images = images.half() # 

			with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
				image_embeds = model.encode_image(images)
				image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

			# offload to CPU
			image_embeds = image_embeds.cpu()

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
		threshold = 0.5
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

def get_matched_cosine_similarity(
		image_embeds,
		text_embeds,
		labels,
		is_multi_label,
		verbose=False,  # set True for one epoch to diagnose
):
		"""Compute cosine similarity between matched image-text pairs."""

		# ── Debug: input tensor health ────────────────────────────────────────
		if verbose:
				print(f"\n[CosSim DEBUG] Input shapes and health")
				print(f"  image_embeds  : {image_embeds.shape} dtype={image_embeds.dtype} device={image_embeds.device}")
				print(f"  text_embeds   : {text_embeds.shape} dtype={text_embeds.dtype} device={text_embeds.device}")
				print(f"  labels        : {labels.shape} dtype={labels.dtype}")
				print(f"  image_embeds  — nan={torch.isnan(image_embeds).any().item()} inf={torch.isinf(image_embeds).any().item()}")
				print(f"  text_embeds   — nan={torch.isnan(text_embeds).any().item()} inf={torch.isinf(text_embeds).any().item()}")

				# Check normalisation — CLIP embeddings should be unit norm
				img_norms = image_embeds.norm(dim=1)
				txt_norms = text_embeds.norm(dim=1)
				print(f"\n  image_embeds norms — min={img_norms.min():.6f} max={img_norms.max():.6f} mean={img_norms.mean():.6f}")
				print(f"  text_embeds  norms — min={txt_norms.min():.6f} max={txt_norms.max():.6f} mean={txt_norms.mean():.6f}")
				print(f"  Are image_embeds normalised (norm≈1)? {torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-3)}")
				print(f"  Are text_embeds  normalised (norm≈1)? {torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-3)}")

				# Check value ranges
				print(f"\n  image_embeds values — min={image_embeds.min():.6f} max={image_embeds.max():.6f} mean={image_embeds.mean():.6f}")
				print(f"  text_embeds  values — min={text_embeds.min():.6f} max={text_embeds.max():.6f} mean={text_embeds.mean():.6f}")

				# Label statistics
				if is_multi_label:
						pos_per_sample = labels.sum(dim=1)
						print(f"\n  Labels — positives per sample: min={pos_per_sample.min().item():.0f} max={pos_per_sample.max().item():.0f} mean={pos_per_sample.mean().item():.2f}")
						print(f"  Samples with zero positive labels: {(pos_per_sample == 0).sum().item()}")

		# ── Build matched text embeddings ─────────────────────────────────────
		if is_multi_label:
				matched_text_embeds = torch.zeros_like(image_embeds)
				positive_counts = []
				for i in range(len(labels)):
						positive_indices = torch.where(labels[i] == 1)[0]
						if positive_indices.numel() > 0:
								matched_text_embeds[i] = text_embeds[positive_indices].mean(dim=0)
								positive_counts.append(positive_indices.numel())
						else:
								matched_text_embeds[i] = text_embeds.mean(dim=0)
								positive_counts.append(0)
		else:
				matched_text_embeds = text_embeds[labels]
				positive_counts = [1] * len(labels)

		# ── Debug: matched embeddings health ─────────────────────────────────
		if verbose:
				matched_norms = matched_text_embeds.norm(dim=1)
				print(f"\n  matched_text_embeds — shape={matched_text_embeds.shape}")
				print(f"  matched_text_embeds — nan={torch.isnan(matched_text_embeds).any().item()} inf={torch.isinf(matched_text_embeds).any().item()}")
				print(f"  matched_text_embeds norms — min={matched_norms.min():.6f} max={matched_norms.max():.6f} mean={matched_norms.mean():.6f}")
				print(f"  NOTE: averaged embeddings are NOT unit norm even if inputs are")
				print(f"  Positive label counts — min={min(positive_counts)} max={max(positive_counts)} mean={sum(positive_counts)/len(positive_counts):.2f}")

				# Raw dot products before normalisation — key diagnostic
				# If dot products are negative, the embeddings are fundamentally misaligned
				dot_products = (image_embeds * matched_text_embeds).sum(dim=1)
				print(f"\n  Raw dot products (before norm division):")
				print(f"    min={dot_products.min():.6f} max={dot_products.max():.6f} mean={dot_products.mean():.6f}")
				print(f"    negative fraction: {(dot_products < 0).float().mean().item():.3f}")

				# Per-sample cosine similarity distribution
				img_n = torch.nn.functional.normalize(image_embeds, dim=1)
				txt_n = torch.nn.functional.normalize(matched_text_embeds, dim=1)
				per_sample_cos = (img_n * txt_n).sum(dim=1)
				print(f"\n  Per-sample CosSim (normalised):")
				print(f"    min={per_sample_cos.min():.6f} max={per_sample_cos.max():.6f} mean={per_sample_cos.mean():.6f}")
				print(f"    positive fraction: {(per_sample_cos > 0).float().mean().item():.3f}")
				print(f"    >0.1 fraction    : {(per_sample_cos > 0.1).float().mean().item():.3f}")
				print(f"    <-0.1 fraction   : {(per_sample_cos < -0.1).float().mean().item():.3f}")

				# Check if the issue is averaging — compare single-label vs averaged
				# Take first sample with exactly 1 positive label as reference
				single_label_samples = [i for i, c in enumerate(positive_counts) if c == 1]
				multi_label_samples  = [i for i, c in enumerate(positive_counts) if c > 1]
				if single_label_samples:
						sl_cos = per_sample_cos[single_label_samples]
						print(f"\n  Single-positive samples ({len(single_label_samples)} samples):")
						print(f"    CosSim — min={sl_cos.min():.6f} max={sl_cos.max():.6f} mean={sl_cos.mean():.6f}")
				if multi_label_samples:
						ml_cos = per_sample_cos[multi_label_samples]
						print(f"  Multi-positive samples ({len(multi_label_samples)} samples):")
						print(f"    CosSim — min={ml_cos.min():.6f} max={ml_cos.max():.6f} mean={ml_cos.mean():.6f}")

				# Check text_embeds passed in — are these class embeddings or something else?
				print(f"\n  text_embeds origin check:")
				print(f"    shape[0]={text_embeds.shape[0]} — expected num_classes=4669")
				print(f"    shape[1]={text_embeds.shape[1]} — expected embed_dim=768")

		# ── Guard: NaN/Inf check ──────────────────────────────────────────────
		if torch.isnan(image_embeds).any() or torch.isnan(matched_text_embeds).any():
				if verbose:
						print(f"\n  [GUARD] NaN detected — returning float('nan')")
				return float('nan')

		# ── Guard: zero-norm mask ─────────────────────────────────────────────
		img_norms = image_embeds.norm(dim=1, keepdim=True)
		txt_norms = matched_text_embeds.norm(dim=1, keepdim=True)
		valid_mask = (img_norms.squeeze() > 1e-8) & (txt_norms.squeeze() > 1e-8)

		if valid_mask.sum() == 0:
				if verbose:
						print(f"\n  [GUARD] No valid pairs — returning float('nan')")
				return float('nan')

		# cos_sim = torch.nn.functional.cosine_similarity(
		# 	image_embeds[valid_mask],
		# 	matched_text_embeds[valid_mask],
		# 	dim=1,
		# )
		# Normalise matched embeddings before cosine similarity
		img_n = torch.nn.functional.normalize(image_embeds[valid_mask], dim=1)
		txt_n = torch.nn.functional.normalize(matched_text_embeds[valid_mask], dim=1)
		cos_sim = torch.nn.functional.cosine_similarity(img_n, txt_n, dim=1)

		result = cos_sim.mean().item()

		if verbose:
				print(f"\n  Final CosSim (valid pairs={valid_mask.sum().item()}/{len(valid_mask)}): {result:.6f}")

		return result

def get_multilabel_alignment_score(
	image_embeds: torch.Tensor,       # [N, D] — L2 normalised
	all_class_embeds: torch.Tensor,   # [C, D] — L2 normalised
	labels: torch.Tensor,             # [N, C] — binary, long
	temperature: float, # 0.07 only for Zero-Shot CLIP,
	topk: int = 5,
	verbose: bool = False,
) -> float:
	"""
	Fraction of samples where at least one true class ranks in top-K
	by cosine similarity. Meaningful and interpretable for multi-label data.
	Returns value in [0, 1]:
			0.0 — no image has any true class in its top-K retrieved classes
			1.0 — every image has at least one true class in its top-K
	"""
	
	# Guard: NaN/Inf check with detailed diagnostics
	image_has_nan = torch.isnan(image_embeds).any()
	image_has_inf = torch.isinf(image_embeds).any()
	class_has_nan = torch.isnan(all_class_embeds).any()
	class_has_inf = torch.isinf(all_class_embeds).any()
	
	if image_has_nan or class_has_nan or image_has_inf or class_has_inf:
		if verbose:
			print(f"\n  [GUARD] NaN/Inf detected in embeddings:")
			print(f"    image_embeds  — NaN: {image_has_nan.item()} | Inf: {image_has_inf.item()}")
			print(f"    class_embeds  — NaN: {class_has_nan.item()} | Inf: {class_has_inf.item()}")
			
			if image_has_nan:
				nan_mask = torch.isnan(image_embeds)
				nan_rows = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
				nan_cols = nan_mask.any(dim=0).nonzero(as_tuple=True)[0]
				print(f"    image_embeds NaN locations:")
				print(f"      Affected samples (rows): {nan_rows.tolist()[:10]} {'...' if len(nan_rows) > 10 else ''} (total: {len(nan_rows)})")
				print(f"      Affected dimensions (cols): {nan_cols.tolist()[:10]} {'...' if len(nan_cols) > 10 else ''} (total: {len(nan_cols)})")
			
			if class_has_nan:
				nan_mask = torch.isnan(all_class_embeds)
				nan_rows = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
				nan_cols = nan_mask.any(dim=0).nonzero(as_tuple=True)[0]
				print(f"    class_embeds NaN locations:")
				print(f"      Affected classes (rows): {nan_rows.tolist()[:10]} {'...' if len(nan_rows) > 10 else ''} (total: {len(nan_rows)})")
				print(f"      Affected dimensions (cols): {nan_cols.tolist()[:10]} {'...' if len(nan_cols) > 10 else ''} (total: {len(nan_cols)})")
			
			if image_has_inf:
				inf_mask = torch.isinf(image_embeds)
				inf_count = inf_mask.sum().item()
				print(f"    image_embeds Inf count: {inf_count}")
			
			if class_has_inf:
				inf_mask = torch.isinf(all_class_embeds)
				inf_count = inf_mask.sum().item()
				print(f"    class_embeds Inf count: {inf_count}")
		
		return float('nan')
	
	# [N, C] cosine similarity (no temperature for ranking — temperature distorts topk)
	# Normalise both sides before ranking
	image_embeds_n = torch.nn.functional.normalize(image_embeds, dim=1)
	all_class_embeds_n = torch.nn.functional.normalize(all_class_embeds, dim=1)
	logits = (image_embeds_n @ all_class_embeds_n.T) / temperature
	
	# Clamp topk to available classes
	effective_k = min(topk, logits.shape[1])
	
	# [N, K] top-K class indices per image
	topk_indices = logits.topk(effective_k, dim=1).indices
	
	# Vectorised hit detection — no Python loop
	# Scatter top-K indices into a binary hit matrix [N, C]
	topk_mask = torch.zeros_like(logits, dtype=torch.bool)
	topk_mask.scatter_(1, topk_indices, True)
	
	# A hit occurs when any true class appears in the top-K mask
	# labels is long — cast to bool for AND operation
	hits = (topk_mask & labels.bool()).any(dim=1)  # [N]
	score = hits.float().mean().item()
	
	if verbose:
		print(f"\n[Alignment Score @ top-{effective_k}] Temperature: {temperature}")
		print(f"  Samples with ≥1 true class in top-{effective_k}: {hits.sum().item()} / {len(hits)}")
		print(f"  Alignment score: {score:.6f}")
		
		# Additional breakdown by number of positive labels
		pos_counts = labels.sum(dim=1).long()
		for n_pos in sorted(pos_counts.unique().tolist()):
			mask = pos_counts == n_pos
			if mask.sum() > 0:
				group_score = hits[mask].float().mean().item()
				print(f"  └─ {n_pos} positive labels ({mask.sum().item()} samples): {group_score}")
	
	return score

def evaluate_best_model(
	model,
	validation_loader,
	active_mask,
	head_mask,
	rare_mask,
	early_stopping,
	checkpoint_path,
	finetune_strategy,
	device,
	cache_dir: str,
	temperature: float,
	topk_values: list[int] = [1, 5, 10],
	clean_cache: bool = True,
	embeddings_cache=None,
	lora_params: Optional[Dict] = None,
	class_embeds_override: Optional[torch.Tensor] = None,
	verbose: bool = True,
):
	model_source = "current"
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	if verbose:
		print(f"Evaluating best {type(model)} on {dataset_name} | Strategy: {finetune_strategy}")

	if checkpoint_path is not None and os.path.exists(checkpoint_path):
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
			if checkpoint_path is None:
				print("No checkpoint path provided. Proceeding with current model weights.")
			else:
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

	if verbose:
		param_count = sum(p.numel() for p in model.parameters())
		print(f">> {type(model)} Parameters: {param_count:,}")
		
	validation_results = get_validation_metrics(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=topk_values,
		finetune_strategy=finetune_strategy,
		cache_dir=cache_dir,
		embeddings_cache=embeddings_cache,
		lora_params=lora_params,
		is_training=False,  # Use cache for final evaluation/inference
		model_hash=get_model_hash(model),
		temperature=temperature,
		class_embeds_override=class_embeds_override,
		verbose=verbose,
	)
	full_metrics = validation_results["full_metrics"]
	i2t_similarity = validation_results["i2t_similarity"]
	t2i_similarity = validation_results["t2i_similarity"]
	device_labels  = validation_results["device_labels"]

	if verbose:
		print("\nComputing tiered retrieval metrics (Overall / Head / Rare)...")

	tiered_i2t = compute_tiered_retrieval_metrics(
			similarity_matrix=i2t_similarity,
			query_labels=device_labels,
			topK_values=topk_values,
			head_mask=head_mask,
			rare_mask=rare_mask,
			active_mask=active_mask,
			mode="Image-to-Text",
			verbose=verbose,
	)
	tiered_t2i = compute_tiered_retrieval_metrics(
			similarity_matrix=t2i_similarity,
			query_labels=device_labels,
			topK_values=topk_values,
			head_mask=head_mask,
			rare_mask=rare_mask,
			active_mask=active_mask,
			mode="Text-to-Image",
			min_val_support=5,
			verbose=verbose,
	)

	# ── Clean up large tensors immediately after use ─────────────
	del i2t_similarity, t2i_similarity
	torch.cuda.empty_cache()
	
	# if clean_cache:
	# 	cleanup_embedding_cache(
	# 		dataset_name=dataset_name,
	# 		cache_dir=cache_dir,
	# 		finetune_strategy=finetune_strategy,
	# 		batch_size=validation_loader.batch_size,
	# 		num_workers=validation_loader.num_workers,
	# 		model_name=model.__class__.__name__,
	# 		model_arch=model.name if hasattr(model, 'name') else 'unknown_arch'
	# 	)

	return {
		"full_metrics":      full_metrics,
		"img2txt_metrics":   validation_results["img2txt_metrics"],
		"txt2img_metrics":   validation_results["txt2img_metrics"],
		"tiered_i2t":        tiered_i2t,
		"tiered_t2i":        tiered_t2i,
		"model_loaded_from": model_source,
	}
