from utils import *
from model import get_lora_clip
from visualize import plot_loss_accuracy_metrics, plot_retrieval_metrics_best_model, plot_retrieval_metrics_per_epoch, plot_all_pretrain_metrics

def cleanup_embedding_cache(
		dataset_name: str,
		cache_dir: str,
		finetune_strategy: str,
		batch_size: int,
		model_name: str,
		model_arch: str,
		num_workers: int,
	):
	base_name = os.path.join(
			cache_dir,
			f"{dataset_name}_"
			f"{finetune_strategy}_"
			f"bs_{batch_size}_"
			f"nw_{num_workers}_"
			f"{model_name}_"
			f"{re.sub(r'[/@]', '_', model_arch)}_"
			f"validation_embeddings"
	)
	cache_files = glob.glob(f"{base_name}.pt") + glob.glob(f"{base_name}_*.pt")
	if cache_files:
			print(f"Found {len(cache_files)} cache file(s) to clean up.")
			for cache_file in cache_files:
					try:
							os.remove(cache_file)
							print(f"Successfully removed cache file: {cache_file}")
					except Exception as e:
							print(f"Warning: Failed to remove cache file {cache_file}: {e}")
	else:
			print(f"No cache files found for {base_name}*.pt")

def cleanup_embedding_cache_old(
		dataset_name: str,
		cache_dir: str,
		finetune_strategy: str, 
		batch_size: int, 
		model_name: str, 
		model_arch: str,
		num_workers: int,
	):

	cache_file = os.path.join(
		cache_dir,
		f"{dataset_name}_"
		f"{finetune_strategy}_"
		f"bs_{batch_size}_"
		f"nw_{num_workers}_"
		f"{model_name}_"
		f"{re.sub(r'[/@]', '_', model_arch)}_"
		f"validation_embeddings.pt"
	)	
	if os.path.exists(cache_file):
		try:
			os.remove(cache_file)
			print(f"Successfully removed cache file: {cache_file}")
		except Exception as e:
			print(f"Warning: Failed to remove cache file {cache_file}: {e}")
	else:
		print(f"Cache file: {cache_file} does not exist. No cleanup.")

def get_model_hash(model: torch.nn.Module) -> str:
		"""
		Generate a hash of model parameters to detect when model weights have changed.
		This is used to determine if cached embeddings need to be recomputed.
		
		Args:
				model: The model to hash
				
		Returns:
				String hash of model parameters
		"""
		hasher = hashlib.md5()
		# Only hash a subset of parameters for efficiency on very large models
		param_sample = []
		for i, param in enumerate(model.parameters()):
				if i % 10 == 0:  # Sample every 10th parameter
						param_sample.append(param.data.cpu().numpy().mean())  # Just use the mean for speed
		
		hasher.update(str(param_sample).encode())
		return hasher.hexdigest()

def compute_direct_in_batch_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str,
		topK_values: List[int],
		max_samples: int = 384  # Increased to align with batch size 128
) -> Dict:
		model.eval()
		total_loss = 0.0
		total_img2txt_correct = 0
		total_txt2img_correct = 0
		processed_batches = 0
		total_samples = 0
		cosine_similarities = []
		
		try:
				class_names = validation_loader.dataset.dataset.classes
		except:
				class_names = validation_loader.dataset.unique_labels
		
		n_classes = len(class_names)
		valid_k_values = [k for k in topK_values if k <= n_classes]
		
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
										
										ground_truth = torch.arange(batch_size, device=device)
										
										loss_img = criterion(logits_per_image, ground_truth)
										loss_txt = criterion(logits_per_text, ground_truth)
										batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
										total_loss += batch_loss
								
								img2txt_preds = torch.argmax(logits_per_image, dim=1)
								img2txt_correct = (img2txt_preds == ground_truth).sum().item()
								total_img2txt_correct += img2txt_correct
								
								txt2img_preds = torch.argmax(logits_per_text, dim=1)
								txt2img_correct = (txt2img_preds == ground_truth).sum().item()
								total_txt2img_correct += txt2img_correct
								
								# Vectorized top-K accuracy
								for k in valid_k_values:
										topk_preds = torch.topk(logits_per_image, k=min(k, batch_size), dim=1)[1]
										img2txt_topk_accuracy[k] += torch.isin(ground_truth.unsqueeze(1), topk_preds).any(dim=1).sum().item()
								
								for k in topK_values:
										topk_preds = torch.topk(logits_per_text, k=min(k, batch_size), dim=1)[1]
										txt2img_topk_accuracy[k] += torch.isin(ground_truth.unsqueeze(1), topk_preds).any(dim=1).sum().item()
								
								with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
										image_embeds = model.encode_image(images)
										text_embeds = model.encode_text(tokenized_labels)
								
								image_embeds = F.normalize(image_embeds, dim=-1)
								text_embeds = F.normalize(text_embeds, dim=-1)
								
								# Vectorized cosine similarity
								cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().numpy()
								cosine_similarities.extend(cos_sim.tolist())
								
								processed_batches += 1
								total_samples += batch_size
						
						except Exception as e:
								print(f"Warning: Error processing batch {bidx}: {e}")
								continue
				
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
				
				avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
				img2txt_acc = total_img2txt_correct / total_samples if total_samples > 0 else 0.0
				txt2img_acc = total_txt2img_correct / total_samples if total_samples > 0 else 0.0
				
				img2txt_topk_acc = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()} if total_samples > 0 else {k: 0.0 for k in valid_k_values}
				txt2img_topk_acc = {k: v / total_samples for k, v in txt2img_topk_accuracy.items()} if total_samples > 0 else {k: 0.0 for k in topK_values}
				
				avg_cos_sim = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0
				
				return {
						"val_loss": float(avg_loss),
						"img2txt_acc": float(img2txt_acc),
						"txt2img_acc": float(txt2img_acc),
						"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
						"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
						"cosine_similarity": float(avg_cos_sim)
				}

def compute_ap(
		i: int, 
		correct_mask: torch.Tensor, 
		query_labels: torch.Tensor, 
		class_counts: Optional[torch.Tensor], 
		mode: str, 
		K: int,
	) -> float:
	correct = correct_mask[i]
	if correct.any():
		relevant_positions = torch.where(correct)[0]
		precisions = []
		cumulative_correct = 0
		for pos in relevant_positions:
			cumulative_correct += 1
			precision_at_pos = cumulative_correct / (pos.item() + 1)
			precisions.append(precision_at_pos)
		if mode == "Image-to-Text":
			R = 1
		else:
			R = class_counts[query_labels[i]].item()
		if R > 0:
			return sum(precisions) / min(R, K)
	return 0.0

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
		verbose: bool = True,
) -> Dict:
		num_queries, num_candidates = similarity_matrix.shape
		device = similarity_matrix.device
		
		# Check cache only if not training
		cache_file = None
		if cache_dir and cache_key and not is_training:
				cache_file = os.path.join(cache_dir, f"{cache_key}_retrieval_metrics.json")
				if os.path.exists(cache_file):
						try:
								if verbose:
										print(f"Loading cached retrieval metrics from {cache_file}")
								with open(cache_file, 'r') as f:
										return json.load(f)
						except Exception as e:
								print(f"Error loading cache: {e}. Computing metrics.")
		
		if verbose:
				print(f"Computing retrieval metrics for {mode} (cache skipped: is_training={is_training})")
		
		if max_k is not None:
				valid_K_values = [K for K in topK_values if K <= max_k]
		else:
				valid_K_values = topK_values
		
		metrics = {"mP": {}, "mAP": {}, "Recall": {}}
		
		all_sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
		
		for K in valid_K_values:
				top_k_indices = all_sorted_indices[:, :K]
				retrieved_labels = candidate_labels[top_k_indices]
				true_labels_expanded = query_labels.unsqueeze(1).expand(-1, K)
				correct_mask = (retrieved_labels == true_labels_expanded)
				
				metrics["mP"][str(K)] = correct_mask.float().mean(dim=1).mean().item()
				
				if mode == "Image-to-Text":
						metrics["Recall"][str(K)] = correct_mask.any(dim=1).float().mean().item()
				else:
						relevant_counts = class_counts[query_labels]
						metrics["Recall"][str(K)] = (correct_mask.sum(dim=1) / relevant_counts.clamp(min=1)).mean().item()
				
				# Vectorized AP
				positions = torch.arange(1, K + 1, device=device).float().unsqueeze(0).expand(num_queries, K)
				cumulative_correct = correct_mask.float().cumsum(dim=1)
				precisions = cumulative_correct / positions
				ap = (precisions * correct_mask.float()).sum(dim=1) / correct_mask.sum(dim=1).clamp(min=1)
				metrics["mAP"][str(K)] = ap.nanmean().item()
		
		# Save to cache only if not training
		if cache_dir and cache_key and not is_training:
				try:
						os.makedirs(cache_dir, exist_ok=True)
						with open(cache_file, 'w') as f:
								json.dump(metrics, f)
						if verbose:
								print(f"Saved metrics to {cache_file}")
				except Exception as e:
						print(f"Warning: Failed to save cache {e}")
		
		return metrics

@torch.no_grad()
def get_validation_metrics(
		model: torch.nn.Module,
		validation_loader: torch.utils.data.DataLoader,
		criterion: torch.nn.Module,
		device: torch.device,
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
) -> Dict:
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

		cache_file = os.path.join(
				cache_dir,
				f"{dataset_name}_{finetune_strategy}_bs_{validation_loader.batch_size}_nw_{num_workers}_{model_class_name}_{re.sub(r'[/@]', '_', model_arch_name)}_validation_embeddings.pt"
		)
		if model_hash:
				cache_file = cache_file.replace(".pt", f"_{model_hash}.pt")

		# Step 1: In-batch metrics (small subset)
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
						max_samples=max_in_batch_samples
				)

		# Step 2: Load or compute embeddings
		cache_loaded = False
		if not is_training and embeddings_cache is not None:
				all_image_embeds, _ = embeddings_cache
				all_labels = torch.tensor(
						[validation_loader.dataset.labels_int[i] for i in range(len(validation_loader.dataset))],
						device='cpu'
				)
				cache_loaded = True
				if verbose:
						print("Loaded embeddings from provided cache.")
		elif not is_training and os.path.exists(cache_file) and not force_recompute:
				if verbose:
						print(f"Loading cached embeddings from {cache_file}")
				cached = torch.load(cache_file, map_location='cpu')
				all_image_embeds = cached['image_embeds']
				all_labels = cached['labels']
				cache_loaded = True

		if not cache_loaded or is_training:
				if verbose:
						print("Computing embeddings from scratch...")
				all_image_embeds = []
				all_labels = []

				model = model.to(device)
				model.eval()

				for images, _, labels_indices in tqdm(validation_loader, desc="Encoding images"):
						images = images.to(device, non_blocking=True)
						with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
								image_embeds = model.encode_image(images)
						image_embeds = F.normalize(image_embeds.float(), dim=-1)  # Ensure float32 after normalization
						all_image_embeds.append(image_embeds.cpu())
						all_labels.extend(labels_indices.cpu().tolist())

				all_image_embeds = torch.cat(all_image_embeds, dim=0)
				all_labels = torch.tensor(all_labels, device='cpu')

				if not is_training:
						os.makedirs(cache_dir, exist_ok=True)
						torch.save({'image_embeds': all_image_embeds, 'labels': all_labels}, cache_file)
						if verbose:
								print(f"Saved embeddings to {cache_file}")

		# Step 3: Compute text embeddings
		text_inputs = clip.tokenize(class_names).to(device)
		with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
				class_text_embeds = model.encode_text(text_inputs)
		class_text_embeds = F.normalize(class_text_embeds.float(), dim=-1).cpu()  # Ensure float32 after normalization

		# Step 4: Compute similarity matrices
		device_image_embeds = all_image_embeds.to(device).float()
		device_class_text_embeds = class_text_embeds.to(device).float()
		device_labels = all_labels.to(device)

		i2t_similarity = device_image_embeds @ device_class_text_embeds.T
		t2i_similarity = device_class_text_embeds @ device_image_embeds.T

		# Step 5: Full-set metrics
		full_metrics = compute_full_set_metrics_from_cache(
				i2t_similarity=i2t_similarity,
				t2i_similarity=t2i_similarity,
				labels=device_labels,
				n_classes=n_classes,
				topK_values=topK_values,
				device=device
		)

		# Step 6: Retrieval metrics
		cache_key_base = f"{dataset_name}_{finetune_strategy}_{model_class_name}_{re.sub(r'[/@]', '_', model_arch_name)}"
		if lora_params:
				cache_key_base += f"_lora_rank_{lora_params['lora_rank']}_lora_alpha_{lora_params['lora_alpha']}_lora_dropout_{lora_params['lora_dropout']}"

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
		)

		class_counts = torch.bincount(device_labels, minlength=n_classes)
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
		)

		if verbose:
				print(f"Validation evaluation completed in {time.time() - start_time:.2f} sec")

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
		device: str
	) -> Dict:
	# Filter valid K values
	valid_k_values = [k for k in topK_values if k <= n_classes]
	
	# Image-to-text accuracy metrics
	img2txt_preds = torch.argmax(i2t_similarity, dim=1)
	img2txt_acc = (img2txt_preds == labels).float().mean().item()
	
	# Image-to-text top-K accuracy
	img2txt_topk_acc = {}
	for k in valid_k_values:
			topk_indices = i2t_similarity.topk(k, dim=1)[1]
			correct = (topk_indices == labels.unsqueeze(1)).any(dim=1)
			img2txt_topk_acc[k] = correct.float().mean().item()
	
	# Text-to-image top-K accuracy
	txt2img_topk_acc = {}
	for k in topK_values:
			class_correct = 0
			effective_k = min(k, i2t_similarity.size(0))
			
			topk_indices = t2i_similarity.topk(effective_k, dim=1)[1]
			for class_idx in range(n_classes):
					retrieved_labels = labels[topk_indices[class_idx]]
					if class_idx in retrieved_labels:
							class_correct += 1
			
			txt2img_topk_acc[k] = class_correct / n_classes
	
	# Set top-1 text-to-image accuracy
	txt2img_acc = txt2img_topk_acc.get(1, 0.0)
	
	# Compute MRR (Mean Reciprocal Rank)
	ranks = i2t_similarity.argsort(dim=1, descending=True)
	rr_indices = ranks.eq(labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
	img2txt_mrr = (1.0 / rr_indices.float()).mean().item()
	
	# Compute cosine similarity (between corresponding image-text pairs)
	# We don't have direct pairs in this context, so using MRR as substitute
	
	# Return metrics in the expected format
	return {
		"img2txt_acc": float(img2txt_acc),
		"txt2img_acc": float(txt2img_acc),
		"img2txt_topk_acc": {str(k): float(v) for k, v in img2txt_topk_acc.items()},
		"txt2img_topk_acc": {str(k): float(v) for k, v in txt2img_topk_acc.items()},
		"mean_reciprocal_rank": float(img2txt_mrr),
		"cosine_similarity": 0.0  # Using placeholder since we don't have direct pairs
	}

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
	):
	model_source = "current"
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	
	if os.path.exists(checkpoint_path):
		if verbose:
			print(f"\nLoading best model weights from {checkpoint_path} for final evaluation...")
		try:
			checkpoint = torch.load(checkpoint_path, map_location=device)
			if 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'])
				best_epoch = checkpoint.get('epoch', 'unknown')
				if verbose:
					print(f"Loaded weights from checkpoint (epoch {best_epoch+1})")
				model_source = "checkpoint"
			elif isinstance(checkpoint, dict) and 'epoch' not in checkpoint:
				model.load_state_dict(checkpoint)
				if verbose:
					print("Loaded weights from direct state dictionary")
				model_source = "checkpoint"
			else:
				if verbose:
					print("Warning: Loaded file format not recognized as a model checkpoint.")
		except Exception as e:
			if verbose:
				print(f"Error loading checkpoint: {e}")
	
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
		print("\nPerforming final evaluation on the best model...")
	
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

def checkpoint_best_model(
		model,
		optimizer,
		scheduler,
		current_val_loss,
		best_val_loss,
		early_stopping,
		checkpoint_path,
		epoch,
		current_phase=None,
		img2txt_metrics=None,
		txt2img_metrics=None,
	):
	"""
	Checkpoint the model when performance improves, with comprehensive state saving.
	
	This function evaluates whether the current model represents an improvement
	over previous checkpoints and saves the model state if it does. It uses a
	combination of early stopping criteria and direct validation loss comparison.
	
	Args:
			model: The model to checkpoint
			optimizer: The optimizer used for training
			scheduler: The learning rate scheduler
			current_val_loss: The current validation loss
			best_val_loss: The best validation loss observed so far (None if first evaluation)
			early_stopping: The early stopping object used for tracking improvement
			checkpoint_path: Path where the checkpoint should be saved
			epoch: Current epoch number (0-indexed)
			current_phase: Current phase number for progressive training (optional)
			img2txt_metrics: Image-to-text retrieval metrics for current evaluation (optional)
			txt2img_metrics: Text-to-image retrieval metrics for current evaluation (optional)
	
	Returns:
			tuple: (
					updated_best_val_loss: The new best validation loss after this check,
					final_img2txt_metrics: Image-to-text metrics if model improved, unchanged otherwise,
					final_txt2img_metrics: Text-to-image metrics if model improved, unchanged otherwise
			)
	"""
	# Initialize return values - will remain unchanged unless model improves
	final_img2txt_metrics = img2txt_metrics
	final_txt2img_metrics = txt2img_metrics
	
	# Create baseline checkpoint dictionary (will be updated if needed)
	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"scheduler_state_dict": scheduler.state_dict(),
		"best_val_loss": best_val_loss,
	}
	
	# Add phase information if available (for progressive training)
	if current_phase is not None:
		checkpoint["phase"] = current_phase
	
	# --- Simplified Improvement Detection Logic ---
	model_improved = False
	
	# Case 1: First evaluation (no previous best)
	if best_val_loss is None:
		print(f"Initial best model (loss {current_val_loss:.5f})")
		best_val_loss = current_val_loss
		model_improved = True
	
	# Case 2: Early stopping detects improvement
	elif early_stopping.is_improvement(current_val_loss):
		print(f"*** New Best Validation Loss Found: {current_val_loss:.6f} (Epoch {epoch+1}) ***")
		best_val_loss = current_val_loss
		model_improved = True
	
	# Case 3: Fallback - direct comparison with minimum delta
	# This handles cases where early stopping might not be properly configured
	elif current_val_loss < best_val_loss - early_stopping.min_delta:
		print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
		best_val_loss = current_val_loss
		model_improved = True
	
	# --- Save Improved Model ---
	if model_improved:
		# Cache best weights to avoid potential race condition
		current_best_weights = None
		if early_stopping.restore_best_weights and early_stopping.best_weights is not None:
			# Make a reference copy to avoid potential race condition
			current_best_weights = early_stopping.best_weights
		
		# Update the best validation loss in the checkpoint
		checkpoint["best_val_loss"] = best_val_loss
		
		# Determine which weights to save
		if current_best_weights is not None:
			# Use the weights cached by early stopping
			checkpoint["model_state_dict"] = current_best_weights
			best_epoch = getattr(early_stopping, 'best_epoch', 0)
			print(f"Best model weights (from epoch {best_epoch+1}) saved to {checkpoint_path}")
		else:
			# Use current model weights
			checkpoint["model_state_dict"] = model.state_dict()
			print(f"Best model weights (current epoch {epoch+1}) saved to {checkpoint_path}")
		
		# Save the checkpoint
		try:
			torch.save(checkpoint, checkpoint_path)
		except Exception as e:
			print(f"Warning: Failed to save checkpoint to {checkpoint_path}: {e}")
		
		# Update metrics return values if available
		if img2txt_metrics is not None:
			final_img2txt_metrics = img2txt_metrics
		if txt2img_metrics is not None:
			final_txt2img_metrics = txt2img_metrics
	
	return best_val_loss, final_img2txt_metrics, final_txt2img_metrics

def compute_slope(losses: List[float]) -> float:
	"""Computes the slope of the best-fit line for a list of losses."""
	if len(losses) < 2: # Need at least two points for a slope
		print("Warning: compute_slope called with less than 2 points. Returning 0.")
		return 0.0
	x = np.arange(len(losses))
	A = np.vstack([x, np.ones(len(x))]).T
	try:
		# Use np.linalg.lstsq for linear regression
		m, _ = np.linalg.lstsq(A, np.array(losses), rcond=None)[0]
		return m
	except np.linalg.LinAlgError:
		print("Warning: Least squares failed in compute_slope, returning slope 0.")
		return 0.0 # Handle potential numerical issues

class EarlyStopping:
	def __init__(
			self,
			patience: int = 5,             				# How many epochs to wait for improvement before stopping
			min_delta: float = 1e-3,       				# Minimum change needed to count as an improvement
			cumulative_delta: float = 0.01,				# Minimum total improvement over window_size needed
			window_size: int = 5,          				# How many recent epochs to consider for trend analysis
			mode: str = 'min',             				# 'min' (decrease is better, e.g., loss) or 'max' (increase is better, e.g., accuracy)
			min_epochs: int = 5,           				# Minimum total epochs before stopping can EVER occur
			restore_best_weights: bool = True, 		# Load best weights back when stopping?
			volatility_threshold: float = 10.0, 	# Stop if % volatility in window exceeds this
			slope_threshold: float = 0.0,  				# Stop if slope worsens beyond this threshold (e.g., >0 for loss)
			pairwise_imp_threshold: float = 5e-3, # Stop if avg improvement between adjacent epochs is below this
			min_phases_before_stopping: int = 3, 	# Minimum training phases to complete before stopping
		):

		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights
		self.volatility_threshold = volatility_threshold
		self.slope_threshold = slope_threshold
		self.pairwise_imp_threshold = pairwise_imp_threshold
		self.min_phases_before_stopping = min_phases_before_stopping
		self.sign = 1 if mode == 'min' else -1 # Multiplier for improvement calculation
		self.reset() # set up the initial internal state variables
		print(
			f"EarlyStopping Initialized: "
			f"Patience={patience}, "
			f"MinDelta={min_delta}, "
			f"CumulativeDelta={cumulative_delta}, "
			f"Window={window_size}, "
			f"MinEpochs={min_epochs}, "
			f"MinPhases={min_phases_before_stopping} (if applicable)"
		)

	def reset(self):
		print("--- EarlyStopping state reset, Essential for starting fresh or resetting between training phases ---")
		# Best score (metric value) observed so far
		self.best_score = None
		# state_dict of the model when best_score was achieved (if restore_best_weights is True)
		self.best_weights = None
		# Counter for consecutive epochs without improvement
		self.counter = 0
		# The epoch number when improvement was last observed
		self.stopped_epoch = 0
		# The epoch number when the absolute best_score was achieved
		self.best_epoch = 0
		# List storing the history of the monitored metric values (e.g., validation losses)
		self.value_history = []
		# List storing boolean flags indicating if improvement occurred in each epoch
		self.improvement_history = []
		# Track the current training phase (set by should_stop)
		self.current_phase = 0

	def compute_volatility(self, window: List[float]) -> float:
		"""Computes the coefficient of variation (volatility) as a percentage."""
		if not window or len(window) < 2:
			return 0.0
		mean_val = np.mean(window)
		std_val = np.std(window)
		return (std_val / abs(mean_val)) * 100 if mean_val != 0 else 0.0

	def is_improvement(self, current_value: float) -> bool:
		"""Checks if the current value is an improvement over the best score."""
		# If no best_score exists yet (first epoch), it's always an improvement.
		if self.best_score is None:
			return True # First epoch is always an improvement
		# Calculate improvement based on mode ('min' or 'max')
		# - If mode='min' (sign=1): improvement = best_score - current_value. Positive if current < best.
		# - If mode='max' (sign=-1): improvement = -(best_score - current_value) = current_value - best_score. Positive if current > best.
		improvement = (self.best_score - current_value) * self.sign
		return improvement > self.min_delta

	def should_stop(
			self,
			current_value: float,
			model: torch.nn.Module,
			epoch: int,
			current_phase: Optional[int] = None,
		) -> bool:

		# --- Update State ---
		self.value_history.append(current_value)
		phase_info = f", Phase {current_phase}" if current_phase is not None else ""
		print(f"\n--- EarlyStopping Check (Epoch {epoch+1}{phase_info}) ---")
		print(f"Current Validation Loss: {current_value}")

		# --- Initial Checks ---
		# 1. Minimum Epochs Check: Don't stop if fewer than min_epochs have run.
		if epoch < self.min_epochs:
			print(f"Skipping early stopping check (epoch {epoch+1} < min_epochs {self.min_epochs})")
			return False # Continue training

		# --- Improvement Tracking ---
		# 2. Check if the current value is an improvement over the best score seen so far.
		improved = self.is_improvement(current_value)
		if improved:
			print(f"\tImprovement detected! Best: {self.best_score if self.best_score is not None else 'N/A'} -> {current_value} (delta: {self.min_delta})")
			self.best_score = current_value         # Update the best score
			self.best_epoch = epoch                 # Record the epoch number of this best score
			self.stopped_epoch = epoch              # Update the epoch where improvement last happened
			self.counter = 0                        # Reset the patience counter
			self.improvement_history.append(True)   # Record improvement in history
			if self.restore_best_weights:
				print("\tSaving best model weights...")
				# Use CPU state_dict to save memory if possible, clone to avoid issues
				self.best_weights = {k: v.clone().cpu().detach() for k, v in model.state_dict().items()}
		else:
			self.counter += 1                       # Increment the patience counter
			self.improvement_history.append(False)  # Record lack of improvement
			print(f"\tNo improvement detected. Best: {self.best_score:.6f}. Patience counter: {self.counter}/{self.patience}")

		# --- Window-Based Metric Calculation ---
		# 3. Check if enough history exists for window-based calculations.
		if len(self.value_history) < self.window_size:
			print(f"\tNot enough history ({len(self.value_history)} < {self.window_size}) for window-based checks.")
			# Even without window metrics, check if patience is exceeded *and* min phases are done.
			patience_exceeded = self.counter >= self.patience
			phase_constraint_met = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
			if patience_exceeded:
				if phase_constraint_met:
					print(f"EARLY STOPPING TRIGGERED (Phase {current_phase} >= {self.min_phases_before_stopping}): Patience ({self.counter}/{self.patience}) exceeded.")
					return True
				else:
					print(f"\tPatience ({self.counter}/{self.patience}) exceeded, but delaying stop (Phase {current_phase} < {self.min_phases_before_stopping})")
					return False
			if self.counter >= self.patience and current_phase >= self.min_phases_before_stopping:
				print(f"EARLY STOPPING TRIGGERED (Phase {current_phase} >= {self.min_phases_before_stopping}): Patience ({self.counter}/{self.patience}) exceeded.")
				return True
			return False # Not enough history for other checks, and patience/phase condition not met

		# If enough history exists, proceed with window calculations:
		last_window = self.value_history[-self.window_size:]
		print(f"\tWindow ({self.window_size} epochs): {last_window}")

		# Calculate metrics over the window:
		# a) Slope Check
		slope = compute_slope(last_window) # Use global function
		print(f"\tSlope over window: {slope:.5f} (Threshold: > {self.slope_threshold})")
		# b) Volatility Check
		volatility = self.compute_volatility(last_window)
		print(f"\tVolatility over window: {volatility:.2f}% (Threshold: >= {self.volatility_threshold}%)")
		# c) Average Pairwise Improvement: Calculate the average change between adjacent epochs.
		# (last_window[i] - last_window[i+1]) * self.sign
		# ensures positive values mean improvement regardless of 'min' or 'max' mode.
		pairwise_diffs = [(last_window[i] - last_window[i+1]) * self.sign for i in range(len(last_window)-1)]
		pairwise_imp_avg = np.mean(pairwise_diffs) if pairwise_diffs else 0.0
		print(f"\tAvg Pairwise Improvement over window: {pairwise_imp_avg:.5f} (Threshold: < {self.pairwise_imp_threshold})")
		# d) Closeness to Best: Check if the current value is already very close to the best score.
		close_to_best = abs(current_value - self.best_score) < self.min_delta if self.best_score is not None else False
		print(f"\tClose to best score ({self.best_score:.6f}): {close_to_best}")
		# e) Cumulative Improvement: Check Check total improvement from the start to the end of the window.
		window_start_value = self.value_history[-self.window_size]
		window_end_value = self.value_history[-1]
		# Calculate improvement based on mode, then take absolute value for threshold check
		cumulative_improvement_signed = (window_start_value - window_end_value) * self.sign
		cumulative_improvement_abs = abs(cumulative_improvement_signed)
		print(f"\tCumulative Improvement over window: {cumulative_improvement_signed} (Threshold for lack of improvement: < {self.cumulative_delta})")
		# ----- Combine Stopping Criteria -----
		# 4. Check if any stopping conditions are met.
		stop_reason = []
		# Reason 1: Patience exceeded
		if self.counter >= self.patience:
			stop_reason.append(f"Patience ({self.counter}/{self.patience})")
		# Reason 2: High Volatility indicates instability
		if volatility >= self.volatility_threshold:
			stop_reason.append(f"High volatility ({volatility:.2f}%)")
		# Reason 3: Worsening Trend (Slope)
		# Check if the slope is moving in the 'wrong' direction beyond the threshold.
		# The condition `(slope * self.sign) < (-self.slope_threshold * self.sign)` handles both 'min' and 'max' modes.
		# E.g., for 'min' mode (sign=1) & slope_threshold=0, this is `slope < 0`, which seems wrong.
		# Let's rethink: We want to stop if slope indicates worsening.
		# For 'min' mode (loss), worsening means slope > slope_threshold (e.g., > 0).
		# For 'max' mode (accuracy), worsening means slope < slope_threshold (e.g., < 0).
		# Let's simplify the condition:
		is_worsening = False
		if self.mode == 'min' and slope > self.slope_threshold: is_worsening = True
		elif self.mode == 'max' and slope < self.slope_threshold: is_worsening = True
		if is_worsening:
			stop_reason.append(f"Worsening slope ({slope:.5f})")
		# Reason 4: Stagnation (Low Pairwise Improvement AND Not Close to Best)
		# Stop if average improvement per step is low, unless we are already very near the best score found.
		if pairwise_imp_avg < self.pairwise_imp_threshold and not close_to_best:
			stop_reason.append(f"Low pairwise improvement ({pairwise_imp_avg:.5f}) & not close to best")
		# Reason 5: Lack of significant cumulative improvement over the window
		# Stop if the total improvement over the whole window is below the threshold.
		if cumulative_improvement_abs < self.cumulative_delta:
			stop_reason.append(f"Low cumulative improvement ({cumulative_improvement_abs:.5f})")

		# --- Final Decision ---
		should_trigger_stop = bool(stop_reason)
		should_really_stop = False

		if should_trigger_stop:
			reason_str = ', '.join(stop_reason)
			# Apply phase check ONLY if current_phase is provided
			phase_constraint_met = (current_phase is None) or (current_phase >= self.min_phases_before_stopping)
			if phase_constraint_met:
				print(f"EARLY STOPPING TRIGGERED: {reason_str}")
				should_really_stop = True
			else: # Phase constraint is active and not met
				print(f"\tStopping condition met ({reason_str}), but delaying stop (Phase {current_phase} < {self.min_phases_before_stopping})")
		else:
			print("\tNo stopping conditions met.")

		# --- Restore Best Weights (if stopping) ---
		# 6. load the best saved weights back into the model.
		if should_really_stop and self.restore_best_weights:
			if self.best_weights is not None:
				try:
					# Get device from model's parameters instead of assuming model.device exists
					target_device = next(model.parameters()).device
					print(f"Restoring model weights from best epoch {self.best_epoch + 1} (score: {self.best_score:.6f})")
					# Load state dict, ensuring tensors are moved to the correct device
					model.load_state_dict({k: v.to(target_device) for k, v in self.best_weights.items()})
				except Exception as e:
					print(f"Error restoring model weights: {e}! Skipping weight restoration.")
			else:
				print("Warning: restore_best_weights is True, but no best weights were saved.")
		return should_really_stop

	def get_status(self) -> Dict[str, Any]:
		"""Returns the current status of the early stopper."""
		status = {
			"best_score": self.best_score,
			"best_epoch": self.best_epoch + 1 if self.best_score is not None else 0,
			"patience_counter": self.counter,
			"value_history_len": len(self.value_history)
		}
		if len(self.value_history) >= self.window_size:
			last_window = self.value_history[-self.window_size:]
			status["volatility_window"] = self.compute_volatility(last_window)
			status["slope_window"] = compute_slope(last_window)
		else:
			status["volatility_window"] = None
			status["slope_window"] = None
		return status

	def get_best_score(self) -> Optional[float]:
		return self.best_score

	def get_best_epoch(self) -> int:
		return self.best_epoch # 0-based

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

def get_num_transformer_blocks(model: torch.nn.Module) -> tuple:
	# Ensure the model has the required attributes
	if not hasattr(model, 'visual'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'visual' attribute.")

	if not hasattr(model, 'transformer'):
		raise ValueError(f"{model.__class__.__name__} ({model.name}) lacks 'transformer' attribute.")

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Count visual blocks
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

def get_layer_groups(model: torch.nn.Module) -> dict:
	vis_nblocks, txt_nblocks = get_num_transformer_blocks(model=model)

	# Determine model type
	is_vit = "ViT" in model.name
	is_resnet = "RN" in model.name

	# Visual transformer or CNN blocks
	visual_blocks = []
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

	layer_groups = {
		'visual_frontend': [
			'visual.conv1',  # patch embedding (ViT) or first conv layer (ResNet)
			'visual.class_embedding' if is_vit else 'visual.bn1',  # CLS token for ViT, bn1 for ResNet
			'visual.positional_embedding' if is_vit else 'visual.relu',  # positional embedding for ViT, relu for ResNet
		],
		'visual_transformer': visual_blocks,
		'text_frontend': [ # Converts tokenized text into embeddings (token_embedding) then adds positional information (positional_embedding).
			'token_embedding',
			'positional_embedding',
		],
		'text_transformer': text_blocks,
		'projections': [
			'visual.proj', # Projects visual transformer’s output (e.g., the CLS token embedding) into the shared space.
			'visual.ln_post' if is_vit else 'visual.attnpool',  # ln_post for ViT, attnpool for ResNet
			'text_projection', # Projects the text transformer’s output into the shared space.
			'logit_scale', # learnable scalar that scales the cosine similarities between image and text embeddings during contrastive loss computation.
		],
	}

	return layer_groups

def unfreeze_layers(
		model: torch.nn.Module,
		strategy: Dict[int, List[str]],
		phase: int,
		cache: Dict[int, List[str]],
	):
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
		unfreeze_percentages: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], # Start at 0% unfrozen, increase to 100%
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
		max_trainable_params: Optional[int] = None,
	) -> Dict[int, List[str]]:

	# Validate input
	if not all(0.0 <= p <= 1.0 for p in unfreeze_percentages):
		raise ValueError("Unfreeze percentages must be between 0.0 and 1.0.")

	if not all(g in ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'] for g in layer_groups_to_unfreeze):
		raise ValueError("Invalid layer group specified. Accepted: visual_frontend, visual_transformer, text_frontend, text_transformer, projections.")

	layer_groups = get_layer_groups(model=model)
	selected_groups = {group: layer_groups[group] for group in layer_groups_to_unfreeze if group in layer_groups}

	if not selected_groups:
		raise ValueError("No valid layer groups found for freezing.")

	# Calculate total layers for visual and text components
	total_v_layers = len(selected_groups.get('visual_transformer', []))
	total_t_layers = len(selected_groups.get('text_transformer', []))
	total_p_layers = len(selected_groups.get('projections', []))

	if total_v_layers == 0 and total_t_layers == 0:
		raise ValueError("No transformer blocks found in visual or text encoders. Cannot create unfreezing schedule.")

	display_percentages = sorted(unfreeze_percentages)  # Ascending order for table
	def create_layer_table(num_layers: int, layer_type: str) -> str:
		table_data = []
		for i, pct in enumerate(display_percentages):
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
				headers=["#", "Phase Type", "Unfrozen Layers", "Percentage"],
				tablefmt="grid"
			)
		)

	print(create_layer_table(total_v_layers, "Visual"))
	print(create_layer_table(total_t_layers, "Text"))

	schedule = {}
	all_transformer_layers = selected_groups.get('visual_transformer', []) + selected_groups.get('text_transformer', [])
	base_layers = sum([selected_groups.get(group, []) for group in ['visual_frontend', 'text_frontend']], [])
	for phase, unfreeze_pct in enumerate(unfreeze_percentages):
		# Calculate number of layers to unfreeze
		v_unfreeze_count = int(unfreeze_pct * total_v_layers)
		t_unfreeze_count = int(unfreeze_pct * total_t_layers)
		p_unfreeze_count = int(unfreeze_pct * total_p_layers)

		# Unfreeze from last to first to prioritize high-level feature adaptation
		v_transformers_to_unfreeze = selected_groups.get('visual_transformer', [])[-v_unfreeze_count:] if v_unfreeze_count > 0 else []
		t_transformers_to_unfreeze = selected_groups.get('text_transformer', [])[-t_unfreeze_count:] if t_unfreeze_count > 0 else []
		projections_to_unfreeze = selected_groups.get('projections', []) # always unfrozen from Phase 0 to allow early adaptation of the output space.

		frontend_layers_to_unfreeze = base_layers if unfreeze_pct == 1.0 else []
		layers_to_unfreeze = v_transformers_to_unfreeze + t_transformers_to_unfreeze + projections_to_unfreeze + frontend_layers_to_unfreeze
		schedule[phase] = layers_to_unfreeze

		print(f"Phase {phase} (unfreeze_pct={unfreeze_pct}): {len(layers_to_unfreeze)} layers to unfreeze")

	print(f"\nUnfreeze Schedule contains {len(schedule)} different phases:\n{[f'phase {phase}: {len(layers)} layers' for phase, layers in schedule.items()]}\n")
	print(json.dumps(schedule, indent=2, ensure_ascii=False))
	print("-"*50)
	return schedule

def should_transition_phase(
		losses: List[float],
		window: int,
		best_loss: Optional[float],
		best_loss_threshold: float,
		volatility_threshold: float,
		slope_threshold: float,
		pairwise_imp_threshold: float,
		accuracies: Optional[List[float]]=None, # Added optional accuracy list
		accuracy_plateau_threshold: float = 1e-3 # Threshold for accuracy stagnation
	) -> bool:

	print(f"\n--- Phase Transition Check (Window: {window}) ---")

	if len(losses) < window:
		print(f"<!> Insufficient loss data ({len(losses)} < {window}) for phase transition.")
		return False

	# --- Loss Analysis ---
	# Coefficient of Variation = (Standard Deviation / |Mean|) * 100
	last_window_losses = losses[-window:]
	current_loss = last_window_losses[-1]
	mean_loss = np.mean(last_window_losses)
	std_loss = np.std(last_window_losses)
	loss_volatility = (std_loss / abs(mean_loss)) * 100 if mean_loss != 0 else 0.0

	# Calculate Average Pairwise Loss Improvement:
	#    - Computes the difference between each adjacent epoch's loss within the window.
	#    - `loss[i] - loss[i+1]` means a positive value indicates loss DECREASED (improvement).
	loss_pairwise_diffs = [last_window_losses[i] - last_window_losses[i+1] for i in range(len(last_window_losses)-1)]
	#    - Average these differences to get the typical improvement per step in the window.
	loss_pairwise_imp_avg = np.mean(loss_pairwise_diffs) if loss_pairwise_diffs else 0.0

	# Calculate Loss Slope:
	#    - Fits a line to the losses in the window and gets the slope.
	#    - Positive slope means loss is generally increasing (worsening).
	#    - Negative slope means loss is generally decreasing (improving).
	loss_slope = compute_slope(last_window_losses) # Use global function

	# Check Closeness to Best Loss:
	#    - Determines if the current loss is already very near the absolute best loss ever recorded.
	#    - Handles the case where best_loss might still be None (early in training).
	close_to_best = best_loss is not None and abs(current_loss - best_loss) < best_loss_threshold

	print(f"Loss Window: {last_window_losses}")
	print(f"Current Loss: {current_loss:.6f} | Best Loss: {best_loss if best_loss is not None else 'N/A'} | Close: {close_to_best} (Thresh: {best_loss_threshold})")
	print(f"Loss Volatility: {loss_volatility:.2f}% (Thresh: >= {volatility_threshold}%)")
	print(f"Loss Slope: {loss_slope:.5f} (Thresh: > {slope_threshold})")
	print(f"Avg Pairwise Loss Improvement: {loss_pairwise_imp_avg:.5f} (Thresh: < {pairwise_imp_threshold})")

	# --- Accuracy Analysis (Optional) ---
	accuracy_plateau = False
	if accuracies is not None:
		if len(accuracies) >= window:
			last_window_acc = accuracies[-window:]
			# Calculate Average Pairwise Accuracy Improvement:
			#     - `acc[i+1] - acc[i]` means a positive value indicates accuracy INCREASED (improvement).
			acc_pairwise_diffs = [last_window_acc[i+1] - last_window_acc[i] for i in range(len(last_window_acc)-1)]
			acc_pairwise_imp_avg = np.mean(acc_pairwise_diffs) if acc_pairwise_diffs else 0.0
			# Determine Accuracy Plateau: If the average improvement is below the threshold, accuracy has likely stalled.
			accuracy_plateau = acc_pairwise_imp_avg < accuracy_plateau_threshold
			print(f"Accuracy Window: {last_window_acc}")
			print(f"Avg Pairwise Acc Improvement: {acc_pairwise_imp_avg:.5f} (Plateau Thresh: < {accuracy_plateau_threshold}) => Plateau: {accuracy_plateau}")
		else:
			print(f"<!> Insufficient accuracy data ({len(accuracies)} < {window}) for plateau check.")
	else:
		print("Accuracy data not provided, skipping accuracy plateau check.")

	# --- Transition Logic ---
	transition = False
	reasons = []

	# Reason 1: Loss is highly volatile (unstable)
	if loss_volatility >= volatility_threshold:
		transition = True
		reasons.append(f"High loss volatility ({loss_volatility:.2f}%)")

	# Reason 2: Loss trend is worsening (slope > threshold)
	if loss_slope > slope_threshold:
		transition = True
		reasons.append(f"Worsening loss slope ({loss_slope:.5f})")

	# Reason 3: Loss improvement has stagnated AND not close to best
	if loss_pairwise_imp_avg < pairwise_imp_threshold and not close_to_best:
		transition = True
		reasons.append(f"Low loss improvement ({loss_pairwise_imp_avg:.5f}) & not close to best")

	# Reason 4: Accuracy has plateaued (if available)
	if accuracy_plateau:
		transition = True
		reasons.append("Accuracy plateau detected")

	if transition:
		print(f"==>> PHASE TRANSITION RECOMMENDED: {', '.join(reasons)}")
	else:
		print("==>> No phase transition needed: Stable progress or close to best.")
	return transition

def handle_phase_transition(
		current_phase: int,
		initial_lr: float,
		initial_wd: float,
		max_phases: int,
		window_size: int,
		current_loss: float,
		best_loss: Optional[float],
	) -> Tuple[int, float, float]:

	# --- 1. Calculate Loss Stability Factor ---
	if best_loss is None or best_loss <= 0:
		loss_stability_factor = 1.0
	else:
		loss_stability_factor = min(max(0.5, current_loss / best_loss), 2.0)

	# --- 2. Calculate Window Factor ---
	window_factor = max(0.5, min(1.5, 10 / window_size))

	# --- 3. Determine Next Phase Index and Phase Factor ---
	next_phase = current_phase + 1
	if next_phase >= max_phases:
		next_phase = max_phases - 1
		phase_factor = 0.1
		print(f"<!> Already in final phase ({current_phase}). Applying fixed LR reduction.")
	else:
		phase_progress = next_phase / max(1, max_phases - 1)
		phase_factor = 0.75 ** phase_progress

	# --- 4. Calculate New Learning Rate ---
	new_lr = initial_lr * phase_factor * loss_stability_factor * window_factor
	min_allowable_lr = initial_lr * 1e-3
	new_lr = max(new_lr, min_allowable_lr)

	# --- 5. Calculate New Weight Decay with Dynamic Max Factor ---
	wd_phase_progress = min(1.0, next_phase / max(1, max_phases - 1))

	# Dynamically determine max_wd_increase_factor based on context
	max_wd_increase_factor = 1.0 + (wd_phase_progress * 1.5) + ((1 - loss_stability_factor) * 1.0)

	# Calculate the total possible increase range
	wd_increase_range = initial_wd * (max_wd_increase_factor - 1.0)

	# Calculate the new weight decay based on linear progression
	new_wd = initial_wd + (wd_increase_range * wd_phase_progress)

	# Add a maximum cap (which is now redundant but kept for clarity)
	max_allowable_wd = initial_wd * max_wd_increase_factor
	new_wd = min(new_wd, max_allowable_wd)

	print(f"\n--- Phase Transition Occurred (Moving to Phase {next_phase}) ---")
	print(f"Previous Phase: {current_phase}")
	print(f"Factors -> Loss Stability: {loss_stability_factor:.3f}, Window Factor: {window_factor:.3f}, Phase Factor: {phase_factor:.3f}")
	print(f"Calculated New LR: {new_lr:.3e} (min allowable: {min_allowable_lr:.3e})")
	print(f"WD Factors -> Phase Progress: {wd_phase_progress:.2f}, Dynamic Max Increase Factor: {max_wd_increase_factor:.2f}")
	print(f"Calculated New WD: {new_wd:.3e} (initial: {initial_wd:.3e})")

	return next_phase, new_lr, new_wd

def get_unfreeze_pcts_hybrid(
		model: torch.nn.Module,
		train_loader: DataLoader,
		min_phases: int,
		max_phases: int,
	):

	vis_nblocks, txt_nblocks = get_num_transformer_blocks(model=model)
	total_transformer_layers = vis_nblocks + txt_nblocks
	layers_per_phase = 2 # Unfreezing 1 layer per modality per phase
	baseline_phases = total_transformer_layers // layers_per_phase + 1
	print(f"Baseline Phases (with total_transformer_layers: {total_transformer_layers}): {baseline_phases}")
	dataset_size = len(train_loader.dataset)
	dataset_phases = int(5 + np.log10(dataset_size))
	print(f"Dataset Size: {dataset_size}: Phases: {dataset_phases}")
	num_phases = max(
		min_phases,
		min(
			max_phases,
			min(
				baseline_phases,
				dataset_phases,
			)
		)
	)
	unfreeze_pcts = np.linspace(0, 1, num_phases).tolist()
	print(f"Unfreeze Schedule contains {len(unfreeze_pcts)} different phases:\n{unfreeze_pcts}")
	return unfreeze_pcts

def progressive_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		print_every: int,
		initial_learning_rate: float,
		initial_weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int=10,
		patience: int = 10,
		min_delta: float = 1e-4, # Make slightly less sensitive than default
		cumulative_delta: float = 5e-3, # Keep cumulative check reasonable
		minimum_epochs: int = 20, # Minimum epochs before ANY early stop
		min_epochs_per_phase: int = 5, # Minimum epochs within a phase before transition check
		volatility_threshold: float = 15.0, # Allow slightly more volatility
		slope_threshold: float = 1e-4, # Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float = 1e-4, # Stricter requirement for pairwise improvement
		accuracy_plateau_threshold: float = 5e-4, # For phase transition based on accuracy
		min_phases_before_stopping: int = 3, # Ensure significant unfreezing before global stop
		topk_values: list[int] = [1, 5, 10],
		layer_groups_to_unfreeze: list[str] = ['visual_transformer', 'text_transformer', 'projections'], # Focus on key layers
		unfreeze_percentages: Optional[List[float]] = None, # Allow passing custom percentages
	):

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

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	# Inspect the model for dropout layers
	dropout_values = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"\nNon-zero dropout detected in base {model.__class__.__name__} {model.name} during {mode} fine-tuning:")
	print(non_zero_dropouts)
	print()

	# Determine unfreeze schedule percentages
	if unfreeze_percentages is None:
		unfreeze_percentages = get_unfreeze_pcts_hybrid(
			model=model,
			train_loader=train_loader,
			min_phases=max(4, min_phases_before_stopping + 1), # Ensure enough phases
			max_phases=15, # Cap the number of phases
		)

	# Get the detailed layer unfreeze schedule
	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		unfreeze_percentages=unfreeze_percentages,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	max_phases = len(unfreeze_schedule)

	optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()), # Initially might be empty if phase 0 has no unfrozen layers
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
		pct_start=0.1, # Standard pct_start
		anneal_strategy='cos' # Cosine annealing
	)
	print(f"Using {scheduler.__class__.__name__} for learning rate scheduling")

	criterion = torch.nn.CrossEntropyLoss()
	print(f"Using {criterion.__class__.__name__} as the loss function")

	scaler = torch.amp.GradScaler(device=device) # For mixed precision
	print(f"Using {scaler.__class__.__name__} for mixed precision training")

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
		f"dropout_{dropout_val}_"
		f"ilr_{initial_learning_rate:.1e}_"
		f"iwd_{initial_weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"best_model.pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = [] # History of average training loss per epoch
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list() # History of [in-batch] validation metrics dicts per epoch
	full_val_loss_acc_metrics_all_epochs = list() # History of [full] validation metrics dicts per epoch
	best_val_loss = None # Track the absolute best validation loss
	layer_cache = {} # Cache for layer status (optional, used by get_status)
	last_lr = initial_learning_rate # Track current LR
	last_wd = initial_weight_decay # Track current WD
	phase_just_changed = False # Flag to signal optimizer refresh needed

	# --- Main Training Loop ---
	train_start_time = time.time()

	for epoch in range(num_epochs):
		epoch_start_time = time.time()
		print(f"\n=== Epoch {epoch+1}/{num_epochs} Phase {current_phase} current LR: {last_lr:.3e} current WD: {last_wd:.3e}) ===")
		torch.cuda.empty_cache()
		# --- Phase Transition Check ---
		# Check only if enough epochs *overall* and *within the phase* have passed,
		# and if we are not already in the last phase.
		if (epoch >= minimum_epochs and # Overall min epochs check
			epochs_in_current_phase >= min_epochs_per_phase and
			current_phase < max_phases - 1 and
			len(early_stopping.value_history) >= window_size):
			print(f"Checking for phase transition (Epochs in phase: {epochs_in_current_phase})")

		val_losses = early_stopping.value_history
		val_accs_in_batch = [m.get('img2txt_acc', 0.0) + m.get('txt2img_acc', 0.0) / 2.0 for m in in_batch_loss_acc_metrics_all_epochs]
		val_accs_full = [m.get('img2txt_acc', 0.0) + m.get('txt2img_acc', 0.0) / 2.0 for m in full_val_loss_acc_metrics_all_epochs]

		should_trans = should_transition_phase(
			losses=val_losses,
			window=window_size,
			best_loss=early_stopping.get_best_score(), # Use best score from early stopping state
			best_loss_threshold=min_delta, # Use min_delta for closeness check
			volatility_threshold=volatility_threshold,
			slope_threshold=slope_threshold, # Use positive threshold for worsening loss
			pairwise_imp_threshold=pairwise_imp_threshold,
			# accuracies=val_accs, # Pass average accuracy
			# accuracy_plateau_threshold=accuracy_plateau_threshold,
		)
		if should_trans:
			current_phase, last_lr, last_wd = handle_phase_transition(
				current_phase=current_phase,
				initial_lr=initial_learning_rate,
				initial_wd=initial_weight_decay,
				max_phases=max_phases,
				window_size=window_size,
				current_loss=val_losses[-1],
				best_loss=early_stopping.get_best_score(),
			)
			epochs_in_current_phase = 0 # Reset phase epoch counter
			early_stopping.reset() # <<< CRITICAL: Reset early stopping state for the new phase
			print(f"Transitioned to Phase {current_phase}. Early stopping reset.")

			phase_just_changed = True # Signal that optimizer needs refresh after unfreeze
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
					'lr': last_lr, # Use the new LR
					'weight_decay': last_wd, # Use the new WD
				}
			)
			print(f"Optimizer parameter groups refreshed. LR set to {last_lr:.3e}, WD set to {last_wd:.3e}.")
			print("Re-initializing OneCycleLR scheduler for new phase/start...")
			steps_per_epoch = len(train_loader)
			# Schedule over remaining epochs (more adaptive)
			# max: Ensure scheduler_epochs is at least 1
			scheduler_epochs = max(1, num_epochs - epoch)
			scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=last_lr, # Use the new LR as the peak for the new cycle
				steps_per_epoch=steps_per_epoch,
				epochs=scheduler_epochs,
				pct_start=0.1, # Consider if this needs adjustment in later phases
				anneal_strategy='cos',
				# last_epoch = -1 # Ensures it starts fresh
			)
			print(f"Scheduler re-initialized with max_lr={last_lr:.3e} for {scheduler_epochs} epochs.")
			phase_just_changed = False # Reset the flag

		# --- Training Epoch ---
		model.train()
		epoch_train_loss = 0.0
		num_train_batches = len(train_loader)
		trainable_params_exist = any(p.requires_grad for p in model.parameters())
		if not trainable_params_exist:
			print("Warning: No trainable parameters found for the current phase. Skipping training steps.")
		else:
			for bidx, batch_data in enumerate(train_loader):
				images, tokenized_labels, _ = batch_data # Adjust unpacking as needed
				images = images.to(device, non_blocking=True)
				tokenized_labels = tokenized_labels.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
					logits_per_image, logits_per_text = model(images, tokenized_labels)
					ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
					loss_img = criterion(logits_per_image, ground_truth)
					loss_txt = criterion(logits_per_text, ground_truth)
					batch_loss = 0.5 * (loss_img + loss_txt)
				if torch.isnan(batch_loss):
					print(f"Warning: NaN loss detected at epoch {epoch+1}, batch {bidx+1}. Skipping batch.")
					continue # Skip optimizer step if loss is NaN
				scaler.scale(batch_loss).backward()
				scaler.unscale_(optimizer) # Unscale before clipping
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step() # Step the scheduler
				batch_loss_item = batch_loss.item()
				epoch_train_loss += batch_loss_item

				if bidx % print_every == 0:
					print(f"\tBatch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item:.6f}")
				elif bidx == num_train_batches - 1 and batch_loss_item > 0:
					print(f"\tBatch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item:.6f}")
				else:
					pass

		avg_training_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 and trainable_params_exist else 0.0
		training_losses.append(avg_training_loss)

		# --- Validation ---
		print(f"Epoch: {epoch+1} validation...")

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
			f'@ Epoch {epoch + 1}:\n'
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
		print(f"Retrieval Metrics:\n")
		print(f"Image-to-Text Retrieval: {retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval: {retrieval_metrics_per_epoch['txt2img']}")

		# --- Checkpointing Best Model ---
		best_val_loss, final_img2txt_metrics, final_txt2img_metrics = checkpoint_best_model(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			current_val_loss=current_val_loss,
			best_val_loss=best_val_loss,
			early_stopping=early_stopping,
			checkpoint_path=mdl_fpth,
			epoch=epoch,
			current_phase=current_phase,
			img2txt_metrics=retrieval_metrics_per_epoch["img2txt"],
			txt2img_metrics=retrieval_metrics_per_epoch["txt2img"],
		)

		# --- Early Stopping Check ---
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			current_phase=current_phase
		):
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break # Exit the main training loop

		# --- End of Epoch ---
		epochs_in_current_phase += 1
		epoch_duration = time.time() - epoch_start_time
		print(f"Epoch {epoch+1} Duration: {epoch_duration:.2f}s")
		if epoch+1 > minimum_epochs: 
			print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
		print("-" * 80)

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"\n--- Training Finished ---")
	print(f"Total Epochs Run: {epoch + 1}")
	print(f"Final Phase Reached: {current_phase}")
	print(f"Best Validation Loss Achieved: {early_stopping.get_best_score()} @ Epoch {early_stopping.get_best_epoch() + 1}")
	print(f"Total Training Time: {total_training_time:.2f}s")

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
		f"{model_name}_"
		f"{model_arch}_"
		f"last_phase_{current_phase}_"
		f"ep_{actual_trained_epochs}_"
		f"bs_{train_loader.batch_size}_"
		f"dropout_{dropout_val}_"
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
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_img2txt_accuracy.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_txt2img_accuracy.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_img2txt_accuracy.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_txt2img_accuracy.png"),
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

	return in_batch_loss_acc_metrics_all_epochs # Return history for potential further analysis

def lora_finetune(
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
		lora_rank: int = 8,
		lora_alpha: float = 16.0,
		lora_dropout: float = 0.05,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		topk_values: List[int] = [1, 5, 10, 15, 20],
	):

	# Inspect the model for dropout layers
	dropout_values = []
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
	)

	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name

	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune', '', mode)

	model_arch = re.sub(r'[/@]', '-', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_name = model.__class__.__name__

	print(f"{mode} {model_name} {model_arch} {dataset_name} batch_size: {train_loader.batch_size} {type(device)} {device}".center(160, "-"))
	if torch.cuda.is_available():
		gpu_name = torch.cuda.get_device_name(device)
		total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3) # GB
		print(f"{gpu_name} | {total_mem:.2f}GB VRAM".center(160, " "))

	# for name, param in model.named_parameters():
	# 	print(f"{name} => {param.shape} {param.requires_grad}")

	# Apply LoRA to the model
	model = get_lora_clip(
		clip_model=model,
		lora_rank=lora_rank,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout
	)
	model.to(device)
	get_parameters_info(model=model, mode=mode)

	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad],
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = lr_scheduler.OneCycleLR(
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
		# f"{dataset_name}_"
		f"{mode}_"
		# f"{model_name}_"
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
		f"bs_{train_loader.batch_size}_"
		f"best_model.pth"
	)

	training_losses = []
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None
	in_batch_loss_acc_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []

	for epoch in range(num_epochs):
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

		print(f">> Validation for epoch {epoch+1}...")
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
			f'@ Epoch {epoch + 1}:\n'
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
		print(f"Retrieval Metrics:\n")
		print(f"Image-to-Text Retrieval: {retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval: {retrieval_metrics_per_epoch['txt2img']}")

		# Use our unified checkpointing function
		best_val_loss, final_img2txt_metrics, final_txt2img_metrics = checkpoint_best_model(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			current_val_loss=current_val_loss,
			best_val_loss=best_val_loss,
			early_stopping=early_stopping,
			checkpoint_path=mdl_fpth,
			epoch=epoch,
			img2txt_metrics=retrieval_metrics_per_epoch.get("img2txt"),
			txt2img_metrics=retrieval_metrics_per_epoch.get("txt2img")
		)

		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
		):
			print(f"\nEarly stopping triggered at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
			break

		print("-" * 140)
	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

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
		f"{dataset_name}_"
		f"{mode}_"
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"lora_rank_{lora_rank}_"
		f"lora_alpha_{lora_alpha}_"
		f"lora_dropout_{lora_dropout}"
	)
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_img2txt_accuracy.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_txt2img_accuracy.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_img2txt_accuracy.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_txt2img_accuracy.png"),
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

def full_finetune(
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
		dataset_name = validation_loader.dataset.dataset.__class__.__name__  # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune', '', mode)
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
		dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found (unlikely in your case)

	# Inspect the model for dropout layers
	dropout_values = []
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_values.append((name, module.p))

	non_zero_dropouts = [(name, p) for name, p in dropout_values if p > 0]
	print(f"\nNon-zero dropout detected in base {model_name} {model_arch} during {mode}:")
	print(non_zero_dropouts)
	print()

	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers for fine-tuning, all parammeters are trainable

	get_parameters_info(model=model, mode=mode)

	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad],
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = lr_scheduler.OneCycleLR(
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
		# f"{dataset_name}_"
		f"{mode}_"
		# f"{model_name}_"
		f"{model_arch}_"
		f"{optimizer.__class__.__name__}_"
		f"{scheduler.__class__.__name__}_"
		f"{criterion.__class__.__name__}_"
		f"{scaler.__class__.__name__}_"
		f"ieps_{num_epochs}_"
		f"dropout_{dropout_val}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"best_model.pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = []
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	in_batch_loss_acc_metrics_all_epochs = []
	full_val_loss_acc_metrics_all_epochs = []
	train_start_time = time.time()
	best_val_loss = float('inf')
	final_img2txt_metrics = None
	final_txt2img_metrics = None

	for epoch in range(num_epochs):
		torch.cuda.empty_cache()  # Clear GPU memory cache
		model.train()  # Enable dropout and training mode
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)

			with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()): # Automatic Mixed Precision (AMP)
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(start=0, end=len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)

			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stabilize training
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()  # Update learning rate

			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"\t\tBatch [{bidx + 1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")

			epoch_loss += total_loss.item()

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
			f'@ Epoch {epoch + 1}:\n'
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
		print(f"Retrieval Metrics:\n")
		print(f"Image-to-Text Retrieval: {retrieval_metrics_per_epoch['img2txt']}")
		print(f"Text-to-Image Retrieval: {retrieval_metrics_per_epoch['txt2img']}")

		
		# --- Checkpointing Best Model ---
		best_val_loss, final_img2txt_metrics, final_txt2img_metrics = checkpoint_best_model(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			current_val_loss=current_val_loss,
			best_val_loss=best_val_loss,
			early_stopping=early_stopping,
			checkpoint_path=mdl_fpth,
			epoch=epoch,
			img2txt_metrics=retrieval_metrics_per_epoch["img2txt"],
			txt2img_metrics=retrieval_metrics_per_epoch["txt2img"]
		)

		# --- Early Stopping Check ---
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
		):
			print(f"\nEarly stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score()}")
			break
		print("-" * 140)
	
	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

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
		f"{model_name}_"
		f"{model_arch}_"
		f"ep_{actual_trained_epochs}_"
		f"lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"dropout_{dropout_val}"
	)
	mdl_fpth = get_updated_model_name(original_path=mdl_fpth, actual_epochs=actual_trained_epochs)
	print(f"Best model will be renamed to: {mdl_fpth}")

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_img2txt_accuracy.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_txt2img_accuracy.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_img2txt_accuracy.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_txt2img_accuracy.png"),
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
		dataset_name = validation_loader.dataset.dataset_name #
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	mode = re.sub(r'_finetune', '', mode)
	
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

	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad], # Only optimizes parameters that require gradients
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = lr_scheduler.OneCycleLR(
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
		f"bs_{train_loader.batch_size}_"
		f"best_model.pth"
	)
	print(f"Best model will be saved in: {mdl_fpth}")

	training_losses = []
	img2txt_metrics_all_epochs = []
	txt2img_metrics_all_epochs = []
	in_batch_loss_acc_metrics_all_epochs = []
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

		best_val_loss, final_img2txt_metrics, final_txt2img_metrics = checkpoint_best_model(
			model=model,
			optimizer=optimizer,
			scheduler=scheduler,
			current_val_loss=current_val_loss,
			best_val_loss=best_val_loss,
			early_stopping=early_stopping,
			checkpoint_path=mdl_fpth,
			epoch=epoch,
			img2txt_metrics=img2txt_metrics,
			txt2img_metrics=txt2img_metrics
		)

		# Early stopping check
		if early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
		):
			print(f"\nEarly stopping at epoch {epoch+1}. Best loss: {early_stopping.get_best_score():.5f}")
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
		f"dropout_{dropout_val}"
	)

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"in_batch_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_img2txt_accuracy.png"),
		"in_batch_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_in_batch_topk_txt2img_accuracy.png"),
		"full_val_topk_i2t": os.path.join(results_dir, f"{file_base_name}_full_topk_img2txt_accuracy.png"),
		"full_val_topk_t2i": os.path.join(results_dir, f"{file_base_name}_full_topk_txt2img_accuracy.png"),
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