from utils import *
from datasets_loader import get_dataloaders
from visualize import plot_loss_accuracy, plot_retrieval_metrics_best_model, plot_retrieval_metrics_per_epoch

# train cifar100 from scratch:
# $ nohup python -u trainer.py -d cifar100 -bs 512 -e 250 -lr 5e-5 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m "train" -a "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune cifar100:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 256 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m "finetune" -a "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_ft.out &

# train imagenet from scratch:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 100 -lr 1e-5 -wd 1e-3 --print_every 2500 -nw 50 --device "cuda:1" -m "train" -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_train.out &

# finetune imagenet:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 100 -lr 1e-5 -wd 1e-3 --print_every 2500 -nw 50 --device "cuda:0" -m "finetune" -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_ft.out &

class EarlyStopping:
	def __init__(
			self,
			patience: int = 5, # epochs to wait before stopping the training
			min_delta: float = 1e-3, # minimum difference between new and old loss to count as improvement
			cumulative_delta: float = 0.01,
			window_size: int = 5,
			mode: str = 'min',
			min_epochs: int = 5,
			restore_best_weights: bool = True,
		):
		"""
		Args:
			patience: Number of epochs to wait before early stopping
			min_delta: Minimum change in monitored value to qualify as an improvement
			cumulative_delta: Minimum cumulative improvement over window_size epochs
			window_size: Size of the window for tracking improvement trends
			mode: 'min' for loss, 'max' for metrics like accuracy
			min_epochs: Minimum number of epochs before early stopping can trigger
			restore_best_weights: Whether to restore model to best weights when stopped
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights
		
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.value_history = []
		self.improvement_history = []		
		self.sign = 1 if mode == 'min' else -1
		self.best_epoch = 0,

	def is_improvement(self, current_value: float) -> bool:
		if self.best_score is None:
			return True
		improvement = (self.best_score - current_value) * self.sign
		return improvement > self.min_delta
	
	def calculate_trend(self) -> float:
		"""
			Calculate improvement trend over window.
			Returns inf (mode='min') or -inf (mode='max') if history is shorter than window_size,
			effectively disabling window-based stopping until enough epochs have passed.
		"""
		if len(self.value_history) < self.window_size:
			return float('inf') if self.mode == 'min' else float('-inf')
		window = self.value_history[-self.window_size:]
		if self.mode == 'min':
			return sum(window[i] - window[i+1] for i in range(len(window)-1))
		return sum(window[i+1] - window[i] for i in range(len(window)-1))
	
	# def should_stop(self, current_value: float, model: torch.nn.Module, epoch: int) -> bool:
	# 	"""
	# 	Enhanced stopping decision based on multiple criteria
	# 	"""
	# 	self.value_history.append(current_value)
	# 	# Don't stop before minimum epochs
	# 	if epoch < self.min_epochs:
	# 		if self.best_score is None or current_value * self.sign < self.best_score * self.sign:
	# 			self.best_score = current_value
	# 			self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
	# 			if self.restore_best_weights:
	# 				self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
	# 		return False
		
	# 	if self.is_improvement(current_value):
	# 		self.best_score = current_value
	# 		self.stopped_epoch = epoch  # Update stopped_epoch when a new best score is achieved
	# 		self.best_epoch = epoch  # Update best_epoch when a new best score is achieved
	# 		if self.restore_best_weights:
	# 			self.best_weights = copy.deepcopy(model.state_dict())
	# 		self.counter = 0
	# 		self.improvement_history.append(True)
	# 	else:
	# 		self.counter += 1
	# 		self.improvement_history.append(False)
		
	# 	# Calculate trend over window
	# 	trend = self.calculate_trend()
	# 	cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		
	# 	# Decision logic combining multiple factors
	# 	should_stop = False
		
	# 	# Check primary patience criterion
	# 	if self.counter >= self.patience:
	# 		should_stop = True
		
	# 	# Check if stuck in local optimum
	# 	if len(self.improvement_history) >= self.window_size:
	# 		recent_improvements = sum(self.improvement_history[-self.window_size:])
	# 		if recent_improvements == 0 and cumulative_improvement < self.cumulative_delta:
	# 			should_stop = True
		
	# 	# If stopping, restore best weights if configured
	# 	if should_stop and self.restore_best_weights and self.best_weights:
	# 		model.load_state_dict(self.best_weights)
	# 		self.stopped_epoch = epoch # Update stopped_epoch when stopping
		
	# 	return should_stop

	def should_stop(self, current_value: float, model: torch.nn.Module, epoch: int) -> bool:
		"""
		Enhanced stopping decision based on multiple criteria.
		
		Args:
				current_value: Current value of the monitored metric (e.g., validation loss).
				model: The model being trained.
				epoch: Current epoch number.
		
		Returns:
				bool: Whether to stop training.
		"""
		self.value_history.append(current_value)
		if epoch < self.min_epochs:
				print(f"Epoch {epoch}: Skipping early stopping (min_epochs={self.min_epochs}).")
				return False
		
		if self.is_improvement(current_value):
				print(f"Epoch {epoch}: Improvement detected (current={current_value}, best={self.best_score}).")
				self.best_score = current_value
				self.stopped_epoch = epoch
				if self.restore_best_weights:
						self.best_weights = copy.deepcopy(model.state_dict())
				self.counter = 0
				self.improvement_history.append(True)
		else:
				print(f"Epoch {epoch}: No improvement (current={current_value}, best={self.best_score}).")
				self.counter += 1
				self.improvement_history.append(False)
		
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		print(f"Trend: {trend}, Cumulative Improvement: {cumulative_improvement}")
		
		should_stop = False
		if self.counter >= self.patience:
				print(f"Early stopping triggered (patience={self.patience}).")
				should_stop = True
		
		if len(self.improvement_history) >= self.window_size:
				recent_improvements = sum(self.improvement_history[-self.window_size:])
				if recent_improvements == 0 and cumulative_improvement < self.cumulative_delta:
						print("Early stopping triggered (local optimum detected).")
						should_stop = True
		
		if should_stop and self.restore_best_weights and self.best_weights is not None:
				model.load_state_dict(self.best_weights)
				print("Restored best model weights.")
		
		return should_stop

	def get_best_score(self) -> float:
		return self.best_score
	
	def get_stopped_epoch(self) -> int:
		return self.stopped_epoch

	def get_best_epoch(self) -> int:
		return self.best_epoch

def evaluate_retrieval_performance(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	device: str="cuda:0",
	topK_values: List=[1, 3, 5],
	):
	print(f"Evaluating retrieval performance on {device}".center(60, '='))
	model.eval()
	image_embeddings = []
	image_labels = []
	class_names = []
	
	# Generate text embeddings for all class names once
	dataset = validation_loader.dataset.dataset
	class_names = dataset.classes
	n_classes = len(class_names)
	
	with torch.no_grad():
		# Encode class names to text embeddings
		text_inputs = clip.tokenize(texts=class_names).to(device, non_blocking=True)
		class_text_embeddings = model.encode_text(text_inputs)
		class_text_embeddings = class_text_embeddings / class_text_embeddings.norm(dim=-1, keepdim=True)
		
		# Collect image embeddings and their labels
		for bidx, (images, _, class_indices) in enumerate(validation_loader):
			images = images.to(device, non_blocking=True)
			class_indices = class_indices.to(device, non_blocking=True)
			
			image_embeds = model.encode_image(images)
			image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
			
			image_embeddings.append(image_embeds.cpu().numpy())
			image_labels.extend(class_indices.cpu().numpy())

	# Aggregate and normalize embeddings
	image_embeddings = np.concatenate(image_embeddings, axis=0)
	image_labels = np.array(image_labels)
	class_text_embeddings = class_text_embeddings.cpu().numpy()
	
	# Compute similarity matrix
	similarity_matrix = image_embeddings @ class_text_embeddings.T

	image_to_text_metrics = get_retrieval_metrics(
		similarity_matrix=similarity_matrix,
		query_labels=image_labels,
		candidate_labels=np.arange(n_classes),
		topK_values=topK_values,
		mode="Image-to-Text",
		class_counts=None,  # No class counts for Image-to-Text
		max_k=n_classes,  # Pass max_k for Image-to-Text to limit K to the number of classes
	)
	
	text_to_image_metrics = get_retrieval_metrics(
		similarity_matrix=class_text_embeddings @ image_embeddings.T,
		query_labels=np.arange(n_classes),
		candidate_labels=image_labels,
		topK_values=topK_values,
		mode="Text-to-Image",
		class_counts=np.bincount(image_labels), # Count number of occurrences of each value in array of non-negative ints.
		max_k=None,  # No limit on K for Text-to-Image
	)

	return image_to_text_metrics, text_to_image_metrics

def get_retrieval_metrics(
	similarity_matrix: np.ndarray,
	query_labels: np.ndarray,
	candidate_labels: np.ndarray,
	topK_values: List[int] = [1, 3, 5],
	mode: str ="Image-to-Text",
	class_counts: np.ndarray = None,
	max_k: int = None,  # New parameter to limit K values (None for no limit)
	):
	num_queries, num_candidates = similarity_matrix.shape
	assert num_queries == len(query_labels), "Number of queries must match labels"
	
	num_classes = len(np.unique(candidate_labels)) # unique values in candidate_labels

	# Filter topK_values based on max_k and num_classes
	if max_k is not None:
		valid_K_values = [K for K in topK_values if K <= max_k]
		print(f"max_k: {max_k} | Valid K values: {topK_values}")
	else:
		valid_K_values = topK_values # No limit on K values
	# valid_K_values = [K for K in topK_values if K <= num_classes]

	if len(valid_K_values) < len(topK_values):
		print(f"<!> Warning: K values: ({set(topK_values) - set(valid_K_values)}) exceed the number of classes ({num_classes}). => ignored!")
	
	metrics = {
		"mP": {},
		"mAP": {},
		"Recall": {},
	}
	
	for K in valid_K_values:
		top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :K]
		
		precision, recall, ap = [], [], []
		for i in range(num_queries):
			true_label = query_labels[i]
			retrieved_labels = candidate_labels[top_k_indices[i]]
			correct = np.sum(retrieved_labels == true_label)
			
			# 1. Precision @ K
			precision.append(correct / K)
			
			# 2. Compute Recall@K with division by zero protection
			if mode == "Image-to-Text":
				relevant_count = 1  # Single relevant item per query [single label per image]
			else:
				relevant_count = class_counts[true_label] if class_counts is not None else 0
					
			if relevant_count == 0:
				recall.append(0.0)
			else:
				recall.append(correct / relevant_count)

			# 3. Compute AP@K with proper normalization
			relevant_positions = np.where(retrieved_labels == true_label)[0]
			p_at = []
			cumulative_correct = 0

			for pos in relevant_positions:
				if pos < K:  # Only consider positions within top-K
					cumulative_correct += 1
					precision_at_rank = cumulative_correct / (pos + 1)  # pos is 0-based
					p_at.append(precision_at_rank)

			# Determine normalization factor
			if mode == "Image-to-Text":
				R = 1  # Always 1 relevant item for image-to-text
			else:
				R = class_counts[true_label] if class_counts is not None else 0
					
			# Handle queries with no relevant items
			if R == 0:
				ap.append(0.0)
				continue
					
			if len(p_at) == 0:
				ap.append(0.0)
			else:
				ap.append(sum(p_at) / min(R, K)) # Normalize by min(R, K)

		# Store metrics for this K
		metrics["mP"][str(K)] = np.mean(precision)
		metrics["mAP"][str(K)] = np.mean(ap)
		metrics["Recall"][str(K)] = np.mean(recall)
	
	return metrics

def evaluate_loss_and_accuracy(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str = "cuda",
		topK_values: List = [1, 3, 5],
	):
	"""
	Evaluate the CLIP model's performance on a validation dataset, computing loss, accuracy,
	top-K accuracy, mean reciprocal rank (MRR), and cosine similarity between embeddings.
	Args:
			model: CLIP model instance
			validation_loader: DataLoader for validation data
			criterion: Loss function (e.g., CrossEntropyLoss)
			device: Device to run evaluation on ("cuda" or "cpu")
			topK_values: List of K values for top-K accuracy computation
	Returns:
			Tuple of (avg_val_loss, img2txt_acc, txt2img_acc, img2txt_topk_accuracy, 
								txt2img_topk_accuracy, mean_reciprocal_rank, cosine_sim_mean)
	"""
	# Check device availability and fallback to CPU if CUDA is not available
	if not torch.cuda.is_available() and device.startswith("cuda"):
		print("CUDA not available, falling back to CPU.")
		device = "cpu"
	model.eval()
	total_loss = 0
	total_img2txt_correct = 0
	total_txt2img_correct = 0
	num_batches = len(validation_loader)
	total_samples = len(validation_loader.dataset)
	# Determine number of classes and valid K values for Image-to-Text
	try:
		num_classes = len(validation_loader.dataset.dataset.classes)
	except AttributeError:
		print("Warning: Could not determine number of classes from dataset. Using a default or dataset-specific method.")
		num_classes = len(set(validation_loader.dataset.dataset.labels)) if hasattr(validation_loader.dataset.dataset, 'labels') else 10  # Default to 10 (e.g., CIFAR10)
	if num_classes <= 0:
		raise ValueError("Number of classes must be positive.")
	# Valid K values for Image-to-Text (limited by num_classes)
	valid_img2txt_k_values = [K for K in topK_values if K <= num_classes]
	if len(valid_img2txt_k_values) < len(topK_values):
		print(f"Warning: Some K values ({set(topK_values) - set(valid_img2txt_k_values)}) exceed the number of classes ({num_classes}) for Image-to-Text. They will be ignored.")
	# Valid K values for Text-to-Image (no limit, use all topK_values)
	valid_txt2img_k_values = topK_values
	img2txt_topk_accuracy = {k: 0 for k in valid_img2txt_k_values}
	txt2img_topk_accuracy = {k: 0 for k in valid_txt2img_k_values}
	reciprocal_ranks = []
	cosine_similarities = []
	with torch.no_grad():
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
			images, tokenized_labels = images.to(device, non_blocking=True), tokenized_labels.to(device, non_blocking=True)  # [batch_size, 3, 224, 224], [batch_size, 77]
			batch_size = images.size(0)
			
			# Forward pass to get logits
			logits_per_image, logits_per_text = model(images, tokenized_labels)  # [batch_size, batch_size]
			
			# Ground Truth
			correct_labels = torch.arange(start=0, end=batch_size, dtype=torch.long, device=device)
			
			# Validation Loss: Average of both losses
			loss_img = criterion(logits_per_image, correct_labels)
			loss_txt = criterion(logits_per_text, correct_labels)
			batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
			total_loss += batch_loss
			
			# Predictions for Top-1 Accuracy
			pred_lbl_per_img_idxs = torch.argmax(input=logits_per_image, dim=1)  # [batch_size]
			pred_img_per_lbl_idxs = torch.argmax(input=logits_per_text, dim=1)  # [batch_size]
			
			# Top-1 Accuracy
			img2txt_correct = (pred_lbl_per_img_idxs == correct_labels).sum().item()
			txt2img_correct = (pred_img_per_lbl_idxs == correct_labels).sum().item()
			total_img2txt_correct += img2txt_correct
			total_txt2img_correct += txt2img_correct
			
			# Top-K Accuracy for Image-to-Text
			for k in valid_img2txt_k_values:
				effective_k = min(k, batch_size)  # Ensure k is not greater than batch_size
				topk_predicted_labels_values, topk_predicted_labels_idxs = torch.topk(input=logits_per_image, k=effective_k, dim=1)
				img2txt_topk_accuracy[k] += (topk_predicted_labels_idxs == correct_labels.unsqueeze(1)).any(dim=1).sum().item()
			
			# Top-K Accuracy for Text-to-Image
			for k in valid_txt2img_k_values:
				effective_k = min(k, batch_size)  # Ensure k is not greater than batch_size
				topk_predicted_images_values, topk_predicted_images_idxs = torch.topk(input=logits_per_text, k=effective_k, dim=1)
				txt2img_topk_accuracy[k] += (topk_predicted_images_idxs == correct_labels.unsqueeze(1)).any(dim=1).sum().item()
			
			# Mean Reciprocal Rank (MRR) for Image-to-Text
			ranks = logits_per_image.argsort(dim=1, descending=True)
			rr_indices = ranks.eq(correct_labels.view(-1, 1)).nonzero(as_tuple=True)[1] + 1  # +1 for rank
			rr_indices_inv = (1.0 / rr_indices).cpu().numpy()
			reciprocal_ranks.extend(rr_indices_inv)

			# Cosine Similarity using raw embeddings (more accurate)
			image_embeddings = model.encode_image(images)  # Step 1: Compute embeddings for images
			image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)  # Normalize
			text_embeddings = model.encode_text(tokenized_labels)  # Step 2: Compute embeddings for text labels
			text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)  # Normalize
			cos_sim = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1).cpu().numpy()
			cosine_similarities.extend(cos_sim)

	# Compute average metrics
	print(f"Dataset: {validation_loader.dataset.dataset.__class__.__name__} | {validation_loader.name} | Total Samples: {total_samples} | Num Batches: {num_batches}")
	avg_val_loss = total_loss / num_batches
	img2txt_acc = total_img2txt_correct / total_samples
	txt2img_acc = total_txt2img_correct / total_samples
	img2txt_topk_accuracy = {k: v / total_samples for k, v in img2txt_topk_accuracy.items()}
	txt2img_topk_accuracy = {k: v / total_samples for k, v in txt2img_topk_accuracy.items()}
	mean_reciprocal_rank = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
	cosine_sim_mean = np.mean(cosine_similarities) if cosine_similarities else 0.0

	# Convert to native Python types
	metrics = {
		"val_loss": float(avg_val_loss),  # Convert to Python float
		"img2txt_acc": float(img2txt_acc),  # Convert to Python float
		"txt2img_acc": float(txt2img_acc),  # Convert to Python float
		"img2txt_topk_acc": {k: float(v) for k, v in img2txt_topk_accuracy.items()},  # Convert each value
		"txt2img_topk_acc": {k: float(v) for k, v in txt2img_topk_accuracy.items()},  # Convert each value
		"mean_reciprocal_rank": float(mean_reciprocal_rank),  # Convert NumPy float32 to Python float
		"cosine_similarity": float(cosine_sim_mean),  # Convert NumPy float32 to Python float
	}

	return metrics

def count_clip_layers(model):
		"""
		Count total number of layers in CLIP model
		"""
		total_layers = 0
		unique_layers = set()
		
		# Count each named parameter's layer (get base layer name without .weight/.bias)
		for name, _ in model.named_parameters():
				# Split the name and take everything except the last part (weight/bias)
				layer_name = '.'.join(name.split('.')[:-1])
				if layer_name:  # Avoid empty strings
						unique_layers.add(layer_name)
		
		total_layers = len(unique_layers)
		
		# Detailed breakdown
		visual_transformer_blocks = len([l for l in unique_layers if 'visual.transformer.resblocks' in l])
		text_transformer_blocks = len([l for l in unique_layers if 'transformer.resblocks' in l])
		projection_layers = len([l for l in unique_layers if any(x in l for x in ['visual.proj', 'text_projection', 'visual.ln_post'])])
		frontend_layers = len([l for l in unique_layers if any(x in l for x in ['visual.conv1', 'visual.class_embedding', 'positional_embedding', 'token_embedding'])])
		
		print(f"\nCLIP Layer Statistics:")
		print(f"Total unique layers: {total_layers}")
		print(f"Visual transformer blocks: {visual_transformer_blocks}")
		print(f"Text transformer blocks: {text_transformer_blocks}")
		print(f"Projection layers: {projection_layers}")
		print(f"Frontend layers: {frontend_layers}")
		return total_layers

def get_status(
	model,
	current_phase=0,
	layers_to_freeze=[],
	total_layers=0,
	):
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)    
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	freeze_percentage = (len(layers_to_freeze) / total_layers) * 100 if total_layers > 0 else 0
	unfreeze_percentage = 100 - freeze_percentage
	print("\n" + "="*60)
	print(f"Model Status - Phase {current_phase}".center(60))
	print("="*60)
	print("\nParameter Statistics:")
	print(f"{'Total:':<25} {total_params:,}")
	print(f"{'Trainable:':<25} {trainable_params:,} ({trainable_percent:.2f}%)")
	print(f"{'Frozen:':<25} {frozen_params:,} ({frozen_percent:.2f}%)")
	print("\nLayer Statistics:")
	print(f"{'Total:':<25} {total_layers}")
	print(f"{'Frozen:':<25} {len(layers_to_freeze)} ({freeze_percentage:.1f}%)")
	print(f"{'Trainable:':<25} {total_layers - len(layers_to_freeze)} ({unfreeze_percentage:.1f}%)")
	print("\n" + "="*60 + "\n")

def get_num_vit_blocks(model):
	if not hasattr(model, 'visual') or not hasattr(model.visual, 'transformer'):
		raise ValueError("Model structure not compatible - missing visual transformer")
	vis_transformer = model.visual.transformer
	txt_transformer = model.transformer
	return len(vis_transformer.resblocks), len(txt_transformer.resblocks)

def get_layer_groups(nv:int=12, nt:int=12):
	layer_groups = {
		'visual_frontend': [
			'visual.conv1', # patch embedding
			'visual.class_embedding', # CLS token
			'visual.positional_embedding', # positional embedding
			# 'visual.ln_pre' # 
		],
		'visual_transformer': [f'visual.transformer.resblocks.{i}' for i in range(nv)],
		'text_frontend': ['token_embedding','positional_embedding'],
		'text_transformer': [f'transformer.resblocks.{i}' for i in range(nt)],
		'projections': [
			'visual.proj', # final normalization before projection
			'visual.ln_post',
			'text_projection',
			'logit_scale', # Temperature parameter
		],
	}
	return layer_groups

def get_progressive_freeze_schedule(layer_groups:dict):
	total_v_layers = len(layer_groups['visual_transformer'])
	total_t_layers = len(layer_groups['text_transformer'])
	print(f"Total visual layers: {total_v_layers} | 80%: {int(0.8*total_v_layers)} 60%: {int(0.6*total_v_layers)} 40%: {int(0.4*total_v_layers)}")
	print(f"Total text layers: {total_t_layers} | 80%: {int(0.8*total_t_layers)} 60%: {int(0.6*total_t_layers)} 40%: {int(0.4*total_t_layers)}")
	schedule = [
		# Phase 0: Freeze all layers except the projection layers:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'] + layer_groups['text_frontend'] + layer_groups['text_transformer'],
		# Phase 1: Freeze 80% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.8*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.8*total_t_layers)],
		# Phase 2: freeze 60% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.6*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.6*total_t_layers)],
		# Phase 3: freeze 40% of transformer blocks:
		layer_groups['visual_frontend'] + layer_groups['visual_transformer'][:int(0.4*total_v_layers)] + layer_groups['text_frontend'] + layer_groups['text_transformer'][:int(0.4*total_t_layers)],
		# Phase 4: freeze only (visual + text) frontends
		layer_groups['visual_frontend'] + layer_groups['text_frontend']
	]
	return schedule

def freeze_(layers, model):
	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers first
		if any(ly in name for ly in layers): # Freeze layers in the list
			param.requires_grad = False

def should_transition_phase(
	losses:List[float],
	th: float=1e-4,
	window:int=3,
	) -> bool:
	if len(losses) < window:
		return False # Not enough data to make a decision
	last_window_losses = losses[-window:]
	avg_loss = sum(last_window_losses) / window
	relative_change = abs(last_window_losses[-1] - avg_loss) / avg_loss # Relative change in loss
	transition_required: bool = relative_change < th
	return transition_required

def handle_phase_transition(current_phase, initial_lr, max_phases):
	if current_phase >= max_phases - 1:
		return current_phase, initial_lr * (0.1 ** current_phase)
	new_phase = current_phase + 1
	new_lr = initial_lr * (0.1 ** new_phase) # Reduce learning rate by 10x
	print(f"<!> Plateau detected! Transitioning to Phase {new_phase} with learning rate {new_lr:.1e}")
	return new_phase, new_lr

def finetune(
		model:torch.nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=7,
		nw:int=10,
		print_every:int=150,
		model_name:str="ViT-B/32",
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		device:str="cuda",
		results_dir:str="results",
		window_size:int=10,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES=[1, 5, 10, 15, 20],
	):
	early_stopping = EarlyStopping(
		patience=patience,									# Wait for 10 epochs without improvement before stopping
		min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
		cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
		window_size=window_size,						# Consider the last 10 epochs for cumulative trend
		mode='min',													# Minimize loss
		min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
		restore_best_weights=True						# Restore model weights to the best epoch
	)
	os.makedirs(results_dir, exist_ok=True)
	mode = "finetune"
	dataset_name = train_loader.dataset.dataset.__class__.__name__ # Get the name of the dataset class
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

	total_layers = count_clip_layers(model)
	vis_nblocks, txt_nblocks = get_num_vit_blocks(model)
	print(f"[Transformer Blocks] Vision: {vis_nblocks} | Text: {txt_nblocks}")
	layer_groups = get_layer_groups(nv=vis_nblocks, nt=txt_nblocks,)
	total_v_layers = len(layer_groups['visual_transformer'])
	total_t_layers = len(layer_groups['text_transformer'])
	print(f"[Layer Groups] Visual: {total_v_layers} | Text: {total_t_layers}")
	freeze_schedule = get_progressive_freeze_schedule(layer_groups) # progressive freezing based on validation loss plateau
	print(f"Freeze Schedule[{len(freeze_schedule)}]:\n{json.dumps(freeze_schedule, indent=2)}")
	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_clip.pth"
	)
	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)
	training_losses, val_losses = [], []
	val_acc_img2txt_list = []
	val_acc_txt2img_list = []
	img2txt_topk_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	current_phase = 0
	plateau_threshold = min_delta # ensure parameter consistency
	initial_learning_rate = learning_rate # Store the initial value
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	ft_st = time.time()
	torch.cuda.empty_cache() # Clear GPU memory cache
	# Training loop:
	for epoch in range(num_epochs):
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		# Adaptive Progressive Layer Freezing Schedule:
		if len(val_losses) > 1: # 2 epochs needed to compare
			should_transition = should_transition_phase(
				losses=val_losses,
				th=plateau_threshold,
				window=window_size,
			)
			if should_transition:
				print(f"Plateau detected @ Epoch: {epoch+1} Transitioning from phase: {current_phase} to next phase.")
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(freeze_schedule)
				)
		layers_to_freeze = freeze_schedule[current_phase]
		freeze_(layers=layers_to_freeze, model=model)
		get_status(model, current_phase, layers_to_freeze, total_layers)
		# optimizer = AdamW(
		optimizer = AdamW(
			params=[p for p in model.parameters() if p.requires_grad],
			lr=learning_rate, # potentially update learning rate based on phase
			betas=(0.9, 0.98),
			eps=1e-8,
			weight_decay=weight_decay,
		)
		scheduler = lr_scheduler.OneCycleLR(
			optimizer=optimizer,
			max_lr=learning_rate,
			steps_per_epoch=len(train_loader),
			epochs=num_epochs - epoch,  # Adjust for remaining epochs
			pct_start=0.1,
			anneal_strategy='cos',
		)
		epoch_loss = 0.0
		for bidx, (images, labels) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True) # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			with torch.amp.autocast(device_type=device.type): # # Automatic Mixed Precision (AMP) backpropagation:
				logits_per_image, logits_per_text = model(images, labels) # torch.Size([batch_size, batch_size]) torch.Size([batch_size, batch_size])
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
					f"\t\tBatch [{bidx+1}/{len(train_loader)}] "
					f"Loss: {total_loss.item():.7f}",
				)
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		# print(f"Average {mode} Loss: {avg_training_loss:.7f} ")
		training_losses.append(avg_training_loss)
		(
			avg_valid_loss,
			img2txt_val_acc,
			txt2img_val_acc,
			img2txt_topk_accuracy,
			mean_reciprocal_rank, 
			cosine_sim_mean,
		) = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion, 
			device=device,
			topK_values=TOP_K_VALUES,
		)
		val_losses.append(avg_valid_loss)
		val_acc_img2txt_list.append(img2txt_val_acc)
		val_acc_txt2img_list.append(txt2img_val_acc)
		img2txt_topk_accuracy_list.append([img2txt_topk_accuracy[k] for k in TOP_K_VALUES])
		mean_reciprocal_rank_list.append(mean_reciprocal_rank)
		cosine_similarity_list.append(cosine_sim_mean)
		print(
			f'@ Epoch: {epoch+1}\n'
			f'\t[Loss] {mode}: {avg_training_loss:.7f} | Valid: {avg_valid_loss:.9f}\n'
			f'\tValid Acc [text retrieval per image]: {img2txt_val_acc} '
			f'[image retrieval per text]: {txt2img_val_acc}'
		)
		# Compute retrieval-based metrics
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		print(f"Image-to-text retrieval metrics:")
		print(json.dumps(img2txt_metrics, indent=4, ensure_ascii=False))
		print(f"Text-to-image retrieval metrics:")
		print(json.dumps(txt2img_metrics, indent=4, ensure_ascii=False))
		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)
		# ############################## Early stopping ##############################
		if early_stopping.should_stop(avg_valid_loss, model, epoch):
			print(
				f'\nEarly stopping triggered at epoch {epoch+1}\t'
				f'Best validation loss: {early_stopping.get_best_score():.5f} @ Epoch {early_stopping.get_stopped_epoch()+1}\n'
			)
			break
		else:
			print(f"Saving best model in {mdl_fpth} for best validation loss: {avg_valid_loss:.9f}")
			torch.save(model.state_dict(), mdl_fpth)
		# ############################## Early stopping ##############################
		print("-"*170)
	print(f"{mode} Elapsed_t: {time.time()-ft_st:.1f} sec".center(160, " "))

	losses_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_losses_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_val_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_top_k_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	mrr_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_mrr_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	cs_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_cs_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")

	plot_loss_accuracy(
		train_losses=training_losses,
		val_losses=val_losses,
		val_acc_img2txt_list=val_acc_img2txt_list,
		val_acc_txt2img_list=val_acc_txt2img_list,
		img2txt_topk_accuracy_list=img2txt_topk_accuracy_list,
		mean_reciprocal_rank_list=mean_reciprocal_rank_list,
		cosine_similarity_list=cosine_similarity_list,
		losses_file_path=losses_fpth,
		accuracy_file_path=val_acc_fpth,
		topk_accuracy_file_path=topk_acc_fpth,
		mean_reciprocal_rank_file_path=mrr_fpth,
		cosine_similarity_file_path=cs_fpth,
	)

	retrieval_metrics_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_name)}_retrieval_metrics_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	plot_retrieval_metrics_per_epoch(
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=retrieval_metrics_fpth,
	)

def train(
		model:torch.nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int=5,
		nw:int=10,
		print_every:int=150,
		learning_rate:float=1e-5,
		weight_decay:float=1e-3,
		device:str="cuda",
		results_dir:str="results",
		window_size:int=10,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES=[1, 5, 10, 15, 20],
	):
	early_stopping = EarlyStopping(
		patience=patience,									# Wait for 10 epochs without improvement before stopping
		min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
		cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
		window_size=window_size,						# Consider the last 10 epochs for cumulative trend
		mode='min',													# Minimize loss
		min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
		restore_best_weights=True						# Restore model weights to the best epoch
	)
	dataset_name = validation_loader.dataset.dataset.__class__.__name__ # CIFAR10, ImageNet, etc.
	os.makedirs(results_dir, exist_ok=True)
	mode = "train"
	model_arch = model.name
	print(f"{mode} {model.__class__.__name__} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))

	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers (train from scratch)

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)	
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	print(
		f"[Model Parameters Statictics] Total: {total_params:,} "
		f"Trainable: {trainable_params:,} ({trainable_percent:.2f}%) "
		f"Frozen: {frozen_params:,} ({frozen_percent:.2f}%)"
		.center(160, "-")
	)
	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_clip.pth"
	)
	# optimizer = AdamW(
	optimizer = AdamW(
		params=[p for p in model.parameters() if p.requires_grad],# Only optimizes parameters that require gradients
		lr=learning_rate,
		betas=(0.9,0.98),
		eps=1e-8,
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
	training_losses, val_losses = [], []
	val_acc_img2txt_list = []
	val_acc_txt2img_list = []
	img2txt_topk_accuracy_list = []
	mean_reciprocal_rank_list = []
	cosine_similarity_list = []
	precision_list, recall_list, f1_list = [], [], []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	metrics_for_all_epochs = []
	train_start_time = time.time()
	print(torch.cuda.memory_summary(device=device))
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None
	for epoch in range(num_epochs):
		torch.cuda.empty_cache() # Clear GPU memory cache
		model.train()
		print(f"Epoch [{epoch+1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			# torch.Size([64, 3, 224, 224]), torch.Size([64, 77]), torch.Size([64])
			optimizer.zero_grad() # Clear gradients from previous batch
			images, tokenized_labels = images.to(device, non_blocking=True), tokenized_labels.to(device, non_blocking=True) # torch.Size([b, 3, 224, 224]), torch.Size([b, 77])
			with torch.amp.autocast(device_type=device.type): # # Automatic Mixed Precision (AMP) backpropagation:
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
		metrics_per_epoch = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		print(
			f'@ Epoch {epoch+1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tValid Acc [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
		)

		# Compute retrieval-based metrics
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)
		torch.cuda.empty_cache() # free up GPU memory
		# ############################## Early stopping ##############################
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
		}

		# Check if this is the best model so far
		if current_val_loss < best_val_loss - early_stopping.min_delta:
			print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)  # Save best weights
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics

		# Early stopping check
		if early_stopping.should_stop(current_val_loss, model, epoch):
			print(f"\nEarly stopping at epoch {epoch+1}. Best loss: {early_stopping.get_best_score():.5f}")
			
			# Final evaluation with restored best weights
			final_metrics = evaluate_loss_and_accuracy(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				device=device,
				topK_values=TOP_K_VALUES
			)
			
			final_img2txt, final_txt2img = evaluate_retrieval_performance(
				model=model,
				validation_loader=validation_loader,
				device=device,
				topK_values=TOP_K_VALUES
			)
			
			# Update metrics to match restored weights
			metrics_per_epoch = final_metrics
			img2txt_metrics = final_img2txt
			txt2img_metrics = final_txt2img
			
			# Ensure we keep track of absolute best metrics
			if final_metrics["val_loss"] < best_val_loss:
				best_val_loss = final_metrics["val_loss"]
				checkpoint.update({"best_val_loss": best_val_loss})
				best_img2txt_metrics = final_img2txt
				best_txt2img_metrics = final_txt2img
				torch.save(checkpoint, mdl_fpth)
			break
		# ############################## Early stopping ##############################
		print("-"*170)

	print(f"Elapsed_t: {time.time()-train_start_time:.1f} sec".center(150, "-"))

	losses_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_losses_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	val_acc_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_top1_accuracy_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	img2txt_topk_accuracy_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_img2txt_topk_accuracy_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	txt2img_topk_accuracy_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_txt2img_topk_accuracy_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	topk_acc_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_topk_acc_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	mrr_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_mrr_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	cs_fpth = os.path.join(results_dir, f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_cs_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png")
	plot_loss_accuracy(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
		val_acc_img2txt_list=[metrics["img2txt_acc"] for metrics in metrics_for_all_epochs],
		val_acc_txt2img_list=[metrics["txt2img_acc"] for metrics in metrics_for_all_epochs],
		img2txt_topk_accuracy_list=[metrics["img2txt_topk_acc"] for metrics in metrics_for_all_epochs],
		txt2img_topk_accuracy_list=[metrics["txt2img_topk_acc"] for metrics in metrics_for_all_epochs],
		mean_reciprocal_rank_list=[metrics["mean_reciprocal_rank"] for metrics in metrics_for_all_epochs],
		cosine_similarity_list=[metrics["cosine_similarity"] for metrics in metrics_for_all_epochs],
		losses_file_path=losses_fpth,
		accuracy_file_path=val_acc_fpth,
		img2txt_topk_accuracy_file_path=img2txt_topk_accuracy_fpth,
		txt2img_topk_accuracy_file_path=txt2img_topk_accuracy_fpth,
		mean_reciprocal_rank_file_path=mrr_fpth,
		cosine_similarity_file_path=cs_fpth,
	)

	retrieval_metrics_fpth = os.path.join(
		results_dir, 
		f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_retrieval_metrics_per_epoch_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png"
	)
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=retrieval_metrics_fpth,
	)

	retrieval_metrics_best_model_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_mode_{mode}_{re.sub('/', '', model_arch)}_retrieval_metrics_best_model_per_k_ep_{len(training_losses)}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_{train_loader.batch_size}_bs.png"
	)
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def pretrain(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	device: str="cuda:0",
	TOP_K_VALUES: List=[1, 3, 5],
	):
	print("Pretrain Evaluation".center(150, "-"))
	model_name = model.__class__.__name__
	model_arch = model.name.replace("/","_")
	print(f"Model: {model_name} - {model_arch}") # CLIP - ViT-B/32
	# 1. evaluate_retrieval_performance
	img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=TOP_K_VALUES,
	)
	print("Image to Text Metrics: ")
	print(json.dumps(img2txt_metrics, indent=4))
	print("Text to Image Metrics: ")
	print(json.dumps(txt2img_metrics, indent=4))

	# 2. plot_retrieval_metrics_best_model
	retrieval_metrics_best_model_fpth = os.path.join(f"{validation_loader.name}_retrieval_metrics_pretrained_{model_name}_{model_arch}.png")
	plot_retrieval_metrics_best_model(
		image_to_text_metrics=img2txt_metrics,
		text_to_image_metrics=txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
		best_model_name=f"Pretrained {model_name} {model_arch}",
	)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Balanced Dataset")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs')
	parser.add_argument('--epochs', '-e', type=int, default=12, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-4]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 1e-3]')
	parser.add_argument('--print_every', type=int, default=250, help='Print loss')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP model name')
	parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet'], default='cifar100', help='Choose dataset (CIFAR10/cifar100)')
	parser.add_argument('--mode', '-m', type=str, choices=['pretrain', 'train', 'finetune'], default='pretrain', help='Choose mode (pretrain/train/finetune)')
	parser.add_argument('--window_size', '-ws', type=int, default=5, help='Windows size for early stopping and progressive freezing')
	parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--minimum_epochs', type=int, default=20, help='Early stopping minimum epochs')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 5, 10, 15, 20], help='Top K values for retrieval metrics')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(args)
	set_seeds()
	print(clip.available_models())

	# model, preprocess = clip.load(name=args.model_architecture, device=args.device, jit=False) # training or finetuning => jit=False
	model, preprocess = clip.load_from_scratch(name=args.model_architecture, device=args.device)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	print(f"Model: {model.__class__.__name__} loaded with {model.name} architecture")
	train_loader, validation_loader = get_dataloaders(
		dataset_name=args.dataset,
		batch_size=args.batch_size,
		nw=args.num_workers,
		USER=os.environ.get('USER'),
	)
	print(f"Train Loader[{train_loader.name}]: {len(train_loader)} batches, Validation Loader[{validation_loader.name}]: {len(validation_loader)} batches")
	for bi, batch in enumerate(train_loader):
		print(f"Batch {bi+1}/{len(train_loader)}: contains {len(batch)} element(s): {[elem.shape for elem in batch]}")
		break
	# visualize_(dataloader=train_loader, num_samples=5)
	# return
	if args.mode == 'finetune':
		finetune(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results"),
			window_size=args.window_size, 	# early stopping & progressive unfreezing
			patience=10, 										# early stopping
			min_delta=1e-4, 								# early stopping & progressive unfreezing
			cumulative_delta=5e-3, 					# early stopping
			minimum_epochs=20, 							# early stopping
			TOP_K_VALUES=args.topK_values,
		)
	elif args.mode == 'train':
		train(
			model=model,
			train_loader=train_loader,
			validation_loader=validation_loader,
			num_epochs=args.epochs,
			nw=args.num_workers,
			print_every=args.print_every,
			learning_rate=args.learning_rate,
			weight_decay=args.weight_decay,
			device=args.device,
			results_dir=os.path.join(args.dataset, "results"),
			window_size=args.window_size, # early stopping
			patience=10,									# early stopping
			min_delta=1e-4,								# early stopping
			cumulative_delta=5e-3,				# early stopping
			minimum_epochs=20,						# early stopping
			TOP_K_VALUES=args.topK_values,
		)
	elif args.mode == "pretrain":
		pretrain(
			model=model,
			validation_loader=validation_loader,
			device=args.device,
			TOP_K_VALUES=args.topK_values,			
		)
	else:
		raise ValueError("Invalid mode. Choose either 'finetune' or 'train'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))