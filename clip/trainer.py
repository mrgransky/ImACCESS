from utils import *
from model import get_lora_clip
from visualize import plot_loss_accuracy, plot_retrieval_metrics_best_model, plot_retrieval_metrics_per_epoch, plot_all_pretrain_metrics

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

		# Validate inputs
		if mode not in ["min", "max"]:
			raise ValueError(f"Invalid mode: {mode}. Must be 'min' or 'max'.")
		if window_size < 0:
			raise ValueError(f"Invalid window_size: {window_size}. Must be ≥ 0.")

		self.patience = patience
		self.min_delta = min_delta
		self.cumulative_delta = cumulative_delta
		self.window_size = window_size
		self.mode = mode
		self.min_epochs = min_epochs
		self.restore_best_weights = restore_best_weights

		self.sign = 1 if mode == 'min' else -1
		self.reset()
		
	def reset(self):
		self.best_score = None
		self.best_weights = None
		self.counter = 0
		self.stopped_epoch = 0
		self.value_history = []
		self.improvement_history = []
		self.best_epoch = 0

	def is_improvement(
			self,
			current_value: float,
		) -> bool:
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
		if self.window_size == 0:
			return float("inf") if self.mode == "min" else float("-inf")

		if len(self.value_history) < self.window_size:
			return float('inf') if self.mode == 'min' else float('-inf')

		window = self.value_history[-self.window_size:]

		# # Calculate the trend over the window:
		# if self.mode == 'min':
		# 	return sum(window[i] - window[i+1] for i in range(len(window)-1))
		# return sum(window[i+1] - window[i] for i in range(len(window)-1))

		# simple trend calculation:
		if self.mode == 'min':
			return window[0] - window[-1]  # Total improvement over the window
		else:
			return window[-1] - window[0] # For accuracy-like metrics, we want to see the increase
	
	def should_stop(
			self,
			current_value: float, 
			model: torch.nn.Module, 
			epoch: int,
		) -> bool:
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
			print(f"Epoch {epoch+1}: Still less than minimum epochs. Skipping early stopping (min_epochs={self.min_epochs})")
			return False
		
		if self.is_improvement(current_value):
			print(f"\tImprovement detected (current: {current_value}, best: {self.best_score})")
			self.best_score = current_value
			self.stopped_epoch = epoch
			self.best_epoch = epoch
			if self.restore_best_weights:
				# self.best_weights = copy.deepcopy(model.state_dict())
				self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
			self.counter = 0
			self.improvement_history.append(True)
		else:
			print(
				f"\tNO improvement detected! (current={current_value}, best={self.best_score}) "
				f"absolute difference: {abs(current_value - self.best_score)} "
				f"=> Incrementing counter (current counter={self.counter})"
			)
			self.counter += 1
			self.improvement_history.append(False)
		
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		print(f">> Trend: {trend} | Cumulative Improvement: {cumulative_improvement}")
		
		should_stop = False
		if self.counter >= self.patience:
			print(f"Early stopping can be triggered since validation loss fails to improve for (patience={self.patience}) epochs.")
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
		device: str = "cuda:0",
		topK_values: List[int] = [1, 3, 5],
	):
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} - {model_arch} Retrieval Performance [{dataset_name}]: {topK_values}...")
	model.eval()  # dropout is disabled, ensuring deterministic outputs
	image_embeddings = []
	image_labels = []
	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	n_classes = len(class_names)
	torch.cuda.empty_cache()  # Clear GPU memory cache
	with torch.no_grad():
		text_inputs = clip.tokenize(texts=class_names).to(device, non_blocking=True)
		class_text_embeddings = model.encode_text(text_inputs)
		class_text_embeddings = class_text_embeddings / class_text_embeddings.norm(dim=-1, keepdim=True)
		for bidx, (images, _, class_indices) in enumerate(validation_loader):
			images = images.to(device, non_blocking=True)
			class_indices = class_indices.to(device, non_blocking=True)
			image_embeds = model.encode_image(images)
			image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
			image_embeddings.append(image_embeds.cpu())
			image_labels.extend(class_indices.cpu())
	# Aggregate and normalize embeddings
	image_embeddings = torch.cat(image_embeddings, dim=0)
	image_labels = torch.tensor(image_labels)
	class_text_embeddings = class_text_embeddings.cpu()
	similarity_matrix = image_embeddings @ class_text_embeddings.T
	# print("Similarity matrix stats:")
	# print(
	# 		type(similarity_matrix),
	# 		similarity_matrix.shape,
	# 		similarity_matrix.dtype,
	# 		similarity_matrix.min(),
	# 		similarity_matrix.max(),
	# 		similarity_matrix.mean(),
	# 		similarity_matrix.std(),
	# )
	# print(similarity_matrix[:10, :10])  # ensure values are reasonable (e.g., -1 to 1).

	image_to_text_metrics = get_retrieval_metrics(
		similarity_matrix=similarity_matrix,
		query_labels=image_labels,
		candidate_labels=torch.arange(n_classes),
		topK_values=topK_values,
		mode="Image-to-Text",
		class_counts=None,  # No class counts for Image-to-Text
		max_k=n_classes,  # Pass max_k for Image-to-Text to limit K to the number of classes
	)

	text_to_image_metrics = get_retrieval_metrics(
		similarity_matrix=class_text_embeddings @ image_embeddings.T,
		query_labels=torch.arange(n_classes),
		candidate_labels=image_labels,
		topK_values=topK_values,
		mode="Text-to-Image",
		class_counts=torch.bincount(image_labels),  # Count number of occurrences of each value in array of non-negative ints.
		max_k=None,  # No limit on K for Text-to-Image
	)
	
	return image_to_text_metrics, text_to_image_metrics

def get_retrieval_metrics(
		similarity_matrix: torch.Tensor,
		query_labels: torch.Tensor,
		candidate_labels: torch.Tensor,
		topK_values: List[int] = [1, 3, 5],
		mode: str = "Image-to-Text",
		class_counts: torch.Tensor = None,
		max_k: int = None, # limit K values (None for no limit)
	):

	num_queries, num_candidates = similarity_matrix.shape
	assert num_queries == len(query_labels), "Number of queries must match labels"
	num_classes = len(torch.unique(candidate_labels))
	if max_k is not None:
		valid_K_values = [K for K in topK_values if K <= max_k]
	else:
		valid_K_values = topK_values  # No limit on K values
	if len(valid_K_values) < len(topK_values):
		print(f"\t<!> Warning: K values: ({set(topK_values) - set(valid_K_values)}) exceed the number of classes ({num_classes}) => ignored!")
	metrics = {
		"mP": {},
		"mAP": {},
		"Recall": {},
	}
	for K in valid_K_values:
		top_k_indices = torch.argsort(-similarity_matrix, dim=1)[:, :K]
		precision, recall, ap = [], [], []
		for i in range(num_queries):
			true_label = query_labels[i]
			retrieved_labels = candidate_labels[top_k_indices[i]]
			correct = (retrieved_labels == true_label).sum().item()
			# 1. Precision @ K
			precision.append(correct / K)
			# 2. Compute Recall@K with division by zero protection
			if mode == "Image-to-Text":
				relevant_count = 1  # Single relevant item per query [single label per image]
			else:
				relevant_count = (
					class_counts[true_label].item()
					if class_counts is not None
					else 0
				)
			if relevant_count == 0:
				recall.append(0.0)
			else:
				recall.append(correct / relevant_count)
			# 3. Compute AP@K with proper normalization
			relevant_positions = torch.where(retrieved_labels == true_label)[0]
			p_at = []
			cumulative_correct = 0
			for pos in relevant_positions:
				if pos < K:  # Only consider positions within top-K
					cumulative_correct += 1
					precision_at_rank = cumulative_correct / (pos + 1)  # pos is 0-based
					p_at.append(precision_at_rank)
			# Determine normalization factor
			if mode == "Image-to-Text":
				R = 1 # Always 1 relevant item for image-to-text
			else:
				R = (
					class_counts[true_label].item()
					if class_counts is not None
					else 0
				)
			# Handle queries with no relevant items
			if R == 0:
				ap.append(0.0)
				continue
			if len(p_at) == 0:
				ap.append(0.0)
			else:
				ap.append(sum(p_at) / min(R, K))  # Normalize by min(R, K)
		# Store metrics for this K
		metrics["mP"][str(K)] = torch.tensor(precision).mean().item()
		metrics["mAP"][str(K)] = torch.tensor(ap).mean().item()
		metrics["Recall"][str(K)] = torch.tensor(recall).mean().item()
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
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} - {model_arch} [Loss & Accuracy] [{dataset_name}]: {topK_values}...")

	model.eval()
	total_loss = 0
	total_img2txt_correct = 0
	total_txt2img_correct = 0
	num_batches = len(validation_loader)
	total_samples = len(validation_loader.dataset)

	try:
		class_names = validation_loader.dataset.dataset.classes
	except:
		class_names = validation_loader.dataset.unique_labels
	num_classes = len(class_names)

	if num_classes <= 0:
		raise ValueError("Number of classes must be positive.")

	# Valid K values for Image-to-Text (limited by num_classes)
	valid_img2txt_k_values = [K for K in topK_values if K <= num_classes]
	if len(valid_img2txt_k_values) < len(topK_values):
		print(f"\t<!> Warning: K values ({set(topK_values) - set(valid_img2txt_k_values)}) exceed the number of classes ({num_classes}) for Image-to-Text. => ignored.")

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
	# print(f"Dataset: {validation_loader.dataset.dataset.__class__.__name__} | {validation_loader.name} | Total Samples: {total_samples} | Num Batches: {num_batches}")
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

def get_unfreeze_schedule(
		model: torch.nn.Module,
		unfreeze_percentages: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], # Start at 0% unfrozen, increase to 100%
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
		max_trainable_params: Optional[int] = None
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

def unfreeze_layers(
		model: torch.nn.Module, 
		strategy: list,
		phase: int,
		cache: dict = None,
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

def should_transition_phase(
		losses: List[float],
		accuracies: List[float] = None,
		loss_threshold: float = 5e-3,
		accuracy_threshold: float = 1e-3,
		best_loss_threshold: float = 1e-3,
		window: int = 10,
		best_loss: Optional[float] = None,
	) -> bool:
	"""
	Determines if a phase transition is needed based on loss and accuracy trends.
	
	Args:
		losses: List of training losses per epoch.
		accuracies: Optional list of validation accuracies per epoch.
		loss_threshold: Minimum cumulative loss improvement to avoid plateau.
		accuracy_threshold: Minimum cumulative accuracy improvement to avoid plateau.
		best_loss_threshold: Threshold for closeness to best loss.
		window: Number of epochs to evaluate trends over.
		best_loss: Optional best loss achieved so far.
	
	Returns:
		bool: True if phase transition is required, False otherwise.
	"""

	if len(losses) < window:
		print(f"<!> Not enough loss data ({len(losses)} < {window} windows) to evaluate phase transition.")
		return False
	
	# Loss analysis
	last_window_losses = losses[-window:]
	stability = 
	cv = (np.std(last_window_losses) / np.mean(last_window_losses)) * 100 if mean != 0 else 0  # Avoid division by zero
	cumulative_loss_improvement = last_window_losses[0] - last_window_losses[-1]  # Positive = improvement
	loss_trend = last_window_losses[-1] - last_window_losses[0]  # Positive = worsening
	close_to_best = best_loss is not None and abs(last_window_losses[-1] - best_loss) < best_loss_threshold
	loss_plateau = abs(cumulative_loss_improvement) < loss_threshold
	sustained_improvement = cumulative_loss_improvement > loss_threshold  # Significant continuous improvement

	pairwise_improvements = [last_window_losses[i] - last_window_losses[i+1] for i in range(len(last_window_losses)-1)]
	average_improvement = np.mean(pairwise_improvements) if pairwise_improvements else 0.0 # positive → improvement
	
	# Accuracy analysis
	acc_plateau = False
	cumulative_acc_improvement = None
	if accuracies is not None and len(accuracies) >= window:
		last_window_accs = accuracies[-window:]
		cumulative_acc_improvement = last_window_accs[-1] - last_window_accs[0]  # Positive = improvement
		acc_plateau = abs(cumulative_acc_improvement) < accuracy_threshold

	# Detailed debugging prints
	print(f"Phase transition:")
	print(f"Losses[{len(last_window_losses)}={window} Windows]: Coefficient of Variation: {cv}\n{last_window_losses}")
	print(
		f"\t|Cumulative loss improvement| = {abs(cumulative_loss_improvement)} "
		f"=> Loss plateau (<{loss_threshold}): {loss_plateau}"
	)
	print(
		f"\tCumulative loss improvement(first-last) = {cumulative_loss_improvement} "
		f"=> Sustained Improvement (>{loss_threshold}): {sustained_improvement}"
	)
	print(f"\tLoss trend(last-first): {loss_trend} (>0 worsening: {loss_trend > 0})")
	print(
		f"\tCurrent loss: {last_window_losses[-1]} best loss: {best_loss if best_loss is not None else 'N/A'} | "
		f"Close to best loss (absolute diff) [<{best_loss_threshold}]: {close_to_best} "
	)
	print(f"pairwise_improvements[{len(pairwise_improvements)}]:\n{pairwise_improvements}")
	print(f"\tAverage pairwise lost improvement: {average_improvement}")
	
	if accuracies is not None and len(accuracies) >= window:
		print(f"\t{window} Window accuracies: {last_window_accs}")
		print(f"\tCumulative accuracy improvement: {cumulative_acc_improvement} (threshold: {accuracy_threshold})")
		print(f"\tAccuracy plateau: {acc_plateau}")
	# else:
	# 	print("\t<!> Accuracy data unavailable; relying solely on loss for phase transition.")
	
	transition = False

	# Transition logic
	if loss_plateau:
		if loss_trend > 0:
			transition = True
			print(f"\t>> Decision: Transition due to active loss deterioration, regardless of proximity to best loss! (loss_trend: {loss_trend} > 0)")
		elif close_to_best is False and sustained_improvement is False:
			transition = True
			print("\t>> Decision: Transition due to stagnation without proximity to best loss")
		else:
			print(
				f"\t>> Decision: No transition! Close to best loss ({close_to_best}) | Sustained improvement ({sustained_improvement})"
				f"For transition: Close to best loss must be False and sustained improvement must be False."
			)
	elif acc_plateau:
		transition = True
		print("\t>> Decision: Transition due to accuracy plateau")

	print(f"==>> Phase Transition Required? {transition}")
	return transition

def handle_phase_transition(
		current_phase: int,
		initial_lr: float,
		max_phases: int,
		scheduler,
		window_size: int,  # Add window_size parameter
		current_loss: float,  # Add current validation loss
		best_loss: float,  # Add best observed loss
	):

	# Calculate adaptive LR reduction factor
	loss_ratio = current_loss / best_loss if best_loss > 0 else 1.0
	window_factor = max(0.5, min(2.0, 10/window_size))  # 0.5-2.0 range
	
	if current_phase >= max_phases - 1:
		# Final phase uses dynamic minimum LR
		new_lr = initial_lr * (0.1 * window_factor)
		print(f"Final phase LR: {new_lr:.2e} (window factor: {window_factor})")
		return current_phase, new_lr
	
	# Dynamic phase-based reduction with window consideration
	phase_factor = 0.8 ** (current_phase / (window_size/5))  # Scaled by window
	new_lr = initial_lr * max(0.05, phase_factor * loss_ratio)  # Never <5% of initial
	
	# Update scheduler
	if isinstance(scheduler, lr_scheduler.OneCycleLR):
		scheduler.max_lr = new_lr
		scheduler.base_lrs = [new_lr] * len(scheduler.base_lrs)
	
	for param_group in scheduler.optimizer.param_groups:
		param_group['lr'] = new_lr
	
	print(
		f"Phase {current_phase+1} LR: {new_lr:.2e} | "
		f"Factors: phase={phase_factor}, window={window_factor}, "
		f"loss={loss_ratio}"
	)	
	return current_phase + 1, new_lr

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

def get_adaptive_window_size(
		loader: DataLoader,
		min_window: int,
		max_window: int,
	) -> int:

		n_samples = len(loader.dataset)
		try:
			class_names = loader.dataset.dataset.classes
		except:
			class_names = loader.dataset.unique_labels
		n_classes = len(class_names)
		
		# Base window on dataset complexity
		complexity_factor = np.log10(n_samples * n_classes)
		
		# Bounded exponential scaling
		window = int(
			min(
				max_window, 
				max(
					min_window, 
					np.round(complexity_factor * 3)
				)
			)
		)
		
		print(f"Adaptive window: {window} | Samples: {n_samples} | Classes: {n_classes}")
		return window

def progressive_unfreeze_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		nw: int,
		print_every: int,
		initial_learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		patience: int = 10,
		min_delta: float = 1e-3,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		top_k_values: List[int] = [1, 5, 10, 15, 20],
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
	) -> Dict[str, any]:

	# Input validation
	if not train_loader or not validation_loader:
		raise ValueError("Train and validation loaders must not be empty.")
	
	# Adaptive window size
	window_size =get_adaptive_window_size(
		loader=train_loader,
		min_window=5,
		max_window=20,
	)

	# Initialize early stopping
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True,
	)
	
	# Extract dataset name
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = getattr(validation_loader.dataset, 'dataset_name', 'Unknown')
	
	# Create results directory
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	model_arch = model.name
	model_name = model.__class__.__name__
	
	# just for debugging:
	# for name, param in model.named_parameters():
	# 	print(f"{name} => {param.shape} {param.requires_grad}")
	
	print(f"{mode} {model_name} {model_arch} {dataset_name} {num_epochs} Epoch(s) {device} [x{nw} cores]".center(160, "-"))

	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))
	
	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
	
	unfreeze_percentages = get_unfreeze_pcts_hybrid(
		model=model,
		train_loader=train_loader,
		min_phases=7,
		max_phases=15,
	)

	unfreeze_schedule = get_unfreeze_schedule(
		model=model,
		unfreeze_percentages=unfreeze_percentages,
		layer_groups_to_unfreeze=layer_groups_to_unfreeze,
	)

	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"dropout_{dropout_val}_init_lr_{initial_learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)
	
	criterion = torch.nn.CrossEntropyLoss()

	scaler = torch.amp.GradScaler(
		device=device,
		init_scale=2**16,
		growth_factor=2.0,
		backoff_factor=0.5,
		growth_interval=2000,
	)

	optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()),
		lr=initial_learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)

	scheduler = lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=initial_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1,
		anneal_strategy='cos',
	)

	training_losses = []
	metrics_for_all_epochs = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	global_patience = int(minimum_epochs * 1.5) # Total epochs without improvement across phases
	global_counter = 0
	global_best_loss = float('inf')
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None
	current_phase = 0
	epochs_in_current_phase = 0
	min_epochs_per_phase = 7

	# global minimum number of epochs that must be completed 
	# before any phase transition can occur:
	min_epochs_before_transition = min(window_size, minimum_epochs)

	min_phases_before_stopping = 3 # ensure model progresses through at least 3 phases (unfreezing 60% of transformer blocks) before early stopping can trigger
	layer_cache = {} # Cache for layer freezing status
	learning_rate = None

	train_start_time = time.time()
	for epoch in range(num_epochs):
		torch.cuda.empty_cache()
		epochs_in_current_phase += 1 # Increment the epoch counter for the current phase
		print(f"Epoch [{epoch+1}/{num_epochs}] GPU Memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
		print(
			f"epoch: {epoch} min_epochs_before_transition: {min_epochs_before_transition} "
			f"epoch >= min_epochs_before_transition: {epoch >= min_epochs_before_transition}\n"
			f"epochs_in_current_phase: {epochs_in_current_phase} min_epochs_per_phase: {min_epochs_per_phase} "
			f"epochs_in_current_phase >= min_epochs_per_phase: {epochs_in_current_phase >= min_epochs_per_phase}\n"
			f"epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase: "
			f"{epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase}"
		)

		# Phase transition logic
		if epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase:
			img2txt_accs = [metrics["img2txt_acc"] for metrics in metrics_for_all_epochs]
			txt2img_accs = [metrics["txt2img_acc"] for metrics in metrics_for_all_epochs]
			avg_accs = [(img + txt) / 2 for img, txt in zip(img2txt_accs, txt2img_accs)]

			should_transition = should_transition_phase(
				losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
				accuracies=None,#avg_accs,
				loss_threshold=1e-2,
				accuracy_threshold=5e-5,
				best_loss_threshold=1e-3,
				window=window_size,
				best_loss=best_val_loss,
			)

			if should_transition:
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(unfreeze_schedule),
					scheduler=scheduler,
					window_size=window_size,
					current_loss=metrics_for_all_epochs[-1]["val_loss"],
					best_loss=best_val_loss,
				)
				epochs_in_current_phase = 0  # Reset the counter after transitioning
				early_stopping.reset() # Reset early stopping after phase transition (between phases)

				# Update optimizer with new learning rate
				for param_group in optimizer.param_groups:
					param_group['lr'] = learning_rate

				scheduler.base_lrs = [learning_rate] * len(scheduler.base_lrs)
				scheduler.max_lr = learning_rate

		# Unfreeze layers for current phase
		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)
		
		model.train()
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True) # Clear gradients at start of each epoch

			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=True): # Automatic Mixed Precision (AMP) backpropagation
				logits_per_image, logits_per_text = model(images, tokenized_labels)
				ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
				loss_img = criterion(logits_per_image, ground_truth)
				loss_txt = criterion(logits_per_text, ground_truth)
				total_loss = 0.5 * (loss_img + loss_txt)

			scaler.scale(total_loss).backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
			scheduler.step()
			
			if bidx % print_every == 0 or bidx + 1 == len(train_loader):
				print(f"Batch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
			
			epoch_loss += total_loss.item()
		avg_training_loss = epoch_loss / len(train_loader)
		training_losses.append(avg_training_loss)
		
		# Evaluate on validation set
		metrics_per_epoch = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=top_k_values,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		
		# Compute retrieval metrics
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=top_k_values,
		)

		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
		)

		# Checkpointing
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
		}
		if current_val_loss < best_val_loss - early_stopping.min_delta:
			# print(f"New best model found: (current loss: {current_val_loss} < best loss: {best_val_loss})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics

		# Early stopping (per-phase)
		if early_stopping.should_stop(current_val_loss, model, epoch):
			if current_phase >= min_phases_before_stopping:
				print(f"[Per Phase] Early stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score()}")
				break
			else:
				print(
					f"[Per Phase] Early stopping condition met at epoch {epoch + 1}! "
					f"but delaying until minimum phases ({min_phases_before_stopping}) are reached. "
					f"Current phase: {current_phase}"
				)
		
		# Global Early Stopping
		if current_val_loss < global_best_loss - min_delta:
			global_best_loss = current_val_loss
			global_counter = 0
		else:
			global_counter += 1
		if global_counter >= global_patience and current_phase >= min_phases_before_stopping:
			print(f"Global early stopping triggered after (global_patience={global_patience}) epochs without improvement.")
			break

		print("-" * 140)

	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	file_base_name = (
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}"
		f"_wd_{weight_decay:.1e}"
		f"_bs_{train_loader.batch_size}_"
		f"dropout_{dropout_val}_"
		f"_init_lr_{initial_learning_rate:.1e}"
	)
	if learning_rate is not None:
		file_base_name += f"_final_lr_{learning_rate:.1e}"


	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"val_acc": os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png"),
		"img2txt_topk": os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png"),
		"txt2img_topk": os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

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
		losses_file_path=plot_paths["losses"],
		accuracy_file_path=plot_paths["val_acc"],
		img2txt_topk_accuracy_file_path=plot_paths["img2txt_topk"],
		txt2img_topk_accuracy_file_path=plot_paths["txt2img_topk"],
		mean_reciprocal_rank_file_path=plot_paths["mrr"],
		cosine_similarity_file_path=plot_paths["cs"],
	)

	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=plot_paths["retrieval_per_epoch"],
	)

	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=plot_paths["retrieval_best"],
	)

def lora_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		nw: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		lora_rank: int = 8,
		lora_alpha: float = 16.0,
		lora_dropout: float = 0.05,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		TOP_K_VALUES: List[int] = [1, 5, 10, 15, 20],
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

	window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches

	# Early stopping setup (same as finetune())
	early_stopping = EarlyStopping(
		patience=patience,
		min_delta=min_delta,
		cumulative_delta=cumulative_delta,
		window_size=window_size,
		mode='min',
		min_epochs=minimum_epochs,
		restore_best_weights=True
	)
	
	# Dataset and directory setup (same as finetune())
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except AttributeError:
		dataset_name = validation_loader.dataset.dataset_name
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	model_arch = model.name
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) | {type(device)} {device} [x{nw} cores]".center(160, "-"))

	for name, param in model.named_parameters():
		print(f"{name} => {param.shape} {param.requires_grad}")

	# Apply LoRA to the model
	model = get_lora_clip(
		clip_model=model,
		lora_rank=lora_rank,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout
	)
	model.to(device)
	# Get dropout value (same as finetune())
	get_parameters_info(model=model, mode=mode)

	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"lora_rank_{lora_rank}_alpha_{lora_alpha}_lora_dropout_{lora_dropout}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)

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

	training_losses = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	metrics_for_all_epochs = []
	train_start_time = time.time()
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None

	for epoch in range(num_epochs):
		torch.cuda.empty_cache()
		model.train()
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad(set_to_none=True)
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=True):
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

		metrics_per_epoch = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		
		img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		img2txt_metrics_list.append(img2txt_metrics)
		txt2img_metrics_list.append(txt2img_metrics)

		# Early stopping and checkpointing (same as finetune())
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
		}
		if current_val_loss < best_val_loss - early_stopping.min_delta:
			print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics
		if early_stopping.should_stop(current_val_loss, model, epoch):
			print(f"\nEarly stopping triggered at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
			break
		print("-" * 140)
	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	file_base_name = (
		f"{dataset_name}_{mode}_{re.sub('/', '', model_arch)}_"
		f"alpha_{lora_alpha}_lora_dropout_{lora_dropout}_"
		f"ep_{len(training_losses)}_lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_rank_{lora_rank}"
	)

	losses_fpth = os.path.join(results_dir, f"{file_base_name}_losses.png")
	val_acc_fpth = os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png")
	img2txt_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png")
	txt2img_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png")
	mrr_fpth = os.path.join(results_dir, f"{file_base_name}_mrr.png")
	cs_fpth = os.path.join(results_dir, f"{file_base_name}_cos_sim.png")	

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

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=retrieval_metrics_fpth,
	)
	
	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def full_finetune(
		model: torch.nn.Module,
		train_loader: DataLoader,
		validation_loader: DataLoader,
		num_epochs: int,
		nw: int,
		print_every: int,
		learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		TOP_K_VALUES: List[int] = [1, 5, 10, 15, 20],
	):

	window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches

	early_stopping = EarlyStopping(
			patience=patience,  # Wait for 10 epochs without improvement before stopping
			min_delta=min_delta,  # Consider an improvement only if the change is greater than 0.0001
			cumulative_delta=cumulative_delta,  # Cumulative improvement over the window should be greater than 0.005
			window_size=window_size,  # Consider the last 10 epochs for cumulative trend
			mode='min',  # Minimize loss
			min_epochs=minimum_epochs,  # Ensure at least 20 epochs of training
			restore_best_weights=True  # Restore model weights to the best epoch
	)
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__  # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	model_arch = model.name
	model_name = model.__class__.__name__

	print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) | {type(device)} {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))
	
	# Extract dropout value from the model (if any)
	dropout_val = None
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break

	if dropout_val is None:
			dropout_val = 0.0  # Default to 0.0 if no Dropout layers are found (unlikely in your case)

	for name, param in model.named_parameters():
		param.requires_grad = True # Unfreeze all layers for fine-tuning, all parammeters are trainable

	get_parameters_info(model=model, mode=mode)

	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)
	
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

	# Lists to store metrics
	training_losses = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	metrics_for_all_epochs = []
	train_start_time = time.time()
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None

	for epoch in range(num_epochs):
		torch.cuda.empty_cache()  # Clear GPU memory cache
		model.train()  # Enable dropout and training mode
		print(f"Epoch [{epoch + 1}/{num_epochs}]")
		epoch_loss = 0.0
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
			optimizer.zero_grad() # Clear gradients from previous batch
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			
			with torch.amp.autocast(device_type=device.type, enabled=True): # Automatic Mixed Precision (AMP)
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

		# Evaluate on validation set
		metrics_per_epoch = evaluate_loss_and_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=TOP_K_VALUES,
		)
		metrics_for_all_epochs.append(metrics_per_epoch)
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
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

		# Early stopping
		current_val_loss = metrics_per_epoch["val_loss"]
		checkpoint = {
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
		}

		if current_val_loss < best_val_loss - early_stopping.min_delta:
			print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics

		if early_stopping.should_stop(current_val_loss, model, epoch):
			print(f"\nEarly stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
			final_metrics = evaluate_loss_and_accuracy(
				model=model,
				validation_loader=validation_loader,
				criterion=criterion,
				device=device,
				topK_values=TOP_K_VALUES,
			)

			final_img2txt, final_txt2img = evaluate_retrieval_performance(
				model=model,
				validation_loader=validation_loader,
				device=device,
				topK_values=TOP_K_VALUES,
			)

			metrics_per_epoch = final_metrics
			img2txt_metrics = final_img2txt
			txt2img_metrics = final_txt2img

			if final_metrics["val_loss"] < best_val_loss:
				best_val_loss = final_metrics["val_loss"]
				checkpoint.update({"best_val_loss": best_val_loss})
				best_img2txt_metrics = final_img2txt
				best_txt2img_metrics = final_txt2img
				torch.save(checkpoint, mdl_fpth)
			break
		print("-" * 140)
	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))
	
	file_base_name = (
		f"{dataset_name}_{mode}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}_lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_do_{dropout_val}"
	)

	losses_fpth = os.path.join(results_dir, f"{file_base_name}_losses.png")
	val_acc_fpth = os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png")
	img2txt_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png")
	txt2img_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png")
	mrr_fpth = os.path.join(results_dir, f"{file_base_name}_mrr.png")
	cs_fpth = os.path.join(results_dir, f"{file_base_name}_cos_sim.png")

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

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=retrieval_metrics_fpth,
	)
	
	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def train(
		model:torch.nn.Module,
		train_loader:DataLoader,
		validation_loader:DataLoader,
		num_epochs:int,
		nw:int,
		print_every:int,
		learning_rate:float,
		weight_decay:float,
		device:torch.device,
		results_dir:str,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES:List[int]=[1, 5, 10, 15, 20],
	):
	window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches

	early_stopping = EarlyStopping(
		patience=patience,									# Wait for 10 epochs without improvement before stopping
		min_delta=min_delta,								# Consider an improvement only if the change is greater than 0.0001
		cumulative_delta=cumulative_delta,	# Cumulative improvement over the window should be greater than 0.005
		window_size=window_size,						# Consider the last 10 epochs for cumulative trend
		mode='min',													# Minimize loss
		min_epochs=minimum_epochs,					# Ensure at least 20 epochs of training
		restore_best_weights=True						# Restore model weights to the best epoch
	)
	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__ # CIFAR10, ImageNet, etc.
	except AttributeError as e:
		dataset_name = validation_loader.dataset.dataset_name # 
	os.makedirs(results_dir, exist_ok=True)
	mode = inspect.stack()[0].function
	model_arch = model.name
	model_name = model.__class__.__name__
	print(f"{mode} {model_name} {model_arch} « {dataset_name} » {num_epochs} Epoch(s) | {type(device)} {device} [x{nw} cores]".center(160, "-"))
	if torch.cuda.is_available():
		print(f"{torch.cuda.get_device_name(device)}".center(160, " "))

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

	mdl_fpth = os.path.join(
		results_dir,
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)

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
	training_losses = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	metrics_for_all_epochs = []
	train_start_time = time.time()
	# print(torch.cuda.memory_summary(device=device))
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None
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
			with torch.amp.autocast(device_type=device.type, enabled=True): # # Automatic Mixed Precision (AMP) backpropagation:
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
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
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

	print(f"Elapsed_t: {time.time()-train_start_time:.1f} sec".center(170, "-"))
	file_base_name = (
		f"{dataset_name}_{mode}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}_lr_{learning_rate:.1e}_"
		f"wd_{weight_decay:.1e}_bs_{train_loader.batch_size}_do_{dropout_val}"
	)
	losses_fpth = os.path.join(results_dir, f"{file_base_name}_losses.png")
	val_acc_fpth = os.path.join(results_dir, f"{file_base_name}_top1_accuracy.png")
	img2txt_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png")
	txt2img_topk_accuracy_fpth = os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png")
	mrr_fpth = os.path.join(results_dir, f"{file_base_name}_mrr.png")
	cs_fpth = os.path.join(results_dir, f"{file_base_name}_cos_sim.png")
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

	retrieval_metrics_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png")
	plot_retrieval_metrics_per_epoch(
		dataset_name=dataset_name,
		image_to_text_metrics_list=img2txt_metrics_list,
		text_to_image_metrics_list=txt2img_metrics_list,
		fname=retrieval_metrics_fpth,
	)

	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=best_img2txt_metrics,
		text_to_image_metrics=best_txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
	)

def pretrain(
	model: torch.nn.Module,
	validation_loader: DataLoader,
	results_dir: str,
	device: torch.device,
	TOP_K_VALUES: List=[1, 3, 5],
	):
	model_name = model.__class__.__name__
	model_arch = re.sub(r"[/@]", "_", model.name)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name

	os.makedirs(results_dir, exist_ok=True)

	print(f"Pretrain Evaluation {dataset_name} {model_name} - {model_arch} {device}".center(170, "-"))
	
	# 1. evaluate_retrieval_performance
	img2txt_metrics, txt2img_metrics = evaluate_retrieval_performance(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=TOP_K_VALUES,
	)

	print("Image to Text Metrics: ")
	print(json.dumps(img2txt_metrics, indent=2, ensure_ascii=False))

	print("Text to Image Metrics: ")
	print(json.dumps(txt2img_metrics, indent=2, ensure_ascii=False))

	# 2. plot_retrieval_metrics_best_model
	retrieval_metrics_best_model_fpth = os.path.join(results_dir, f"{validation_loader.name}_retrieval_metrics_pretrained_{model_name}_{model_arch}.png")
	plot_retrieval_metrics_best_model(
		dataset_name=dataset_name,
		image_to_text_metrics=img2txt_metrics,
		text_to_image_metrics=txt2img_metrics,
		fname=retrieval_metrics_best_model_fpth,
		best_model_name=f"Pretrained {model_name} {model_arch}",
	)
	return img2txt_metrics, txt2img_metrics