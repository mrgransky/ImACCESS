from utils import *
from datasets_loader import get_dataloaders
from model import get_lora_clip
from visualize import plot_loss_accuracy, plot_retrieval_metrics_best_model, plot_retrieval_metrics_per_epoch, plot_all_pretrain_metrics

# train cifar100 from scratch:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 100 -nw 50 --device "cuda:3" -m train -a "ViT-B/32" -do 0.1 > /media/volume/ImACCESS/trash/cifar100_train.out &

# finetune cifar100:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m finetune -a "ViT-B/32" > /media/volume/ImACCESS/trash/cifar100_ft.out &

# finetune cifar100 with lora:
# $ nohup python -u trainer.py -d cifar100 -bs 256 -e 250 -lr 1e-4 -wd 1e-3 --print_every 100 -nw 50 --device "cuda:2" -m finetune -fts "lora"  -a "ViT-B/32" --lora > /media/volume/ImACCESS/trash/cifar100_ft_lora.out &

# finetune cifar100 with progressive unfreezing:
# $ nohup python -u trainer.py -d cifar100 -bs 128 -e 250 -lr 1e-4 -wd 1e-2 --print_every 200 -nw 50 --device "cuda:2" -m finetune -fts progressive -a "ViT-B/32"  > /media/volume/ImACCESS/trash/cifar100_ft_progressive.out &

# finetune svhn with progressive unfreezing:
# $ nohup python -u trainer.py -d svhn -bs 512 -e 250 -lr 1e-5 -wd 1e-1 --print_every 50 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" > /media/volume/ImACCESS/trash/svhn_ft_progreessive.out &

# finetune imagenet [full]:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 2500 -nw 50 --device "cuda:0" -m finetune -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_ft.out &

# finetune imagenet with progressive unfreezing:
# $ nohup python -u trainer.py -d imagenet -bs 256 -e 250 -lr 1e-4 -wd 1e-2 --print_every 2500 -nw 50 --device "cuda:0" -m finetune -fts progressive -a "ViT-B/32" > /media/volume/ImACCESS/trash/imagenet_prog_unfreeze_ft.out &

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
			print(f"Epoch {epoch+1}: Improvement detected (current={current_value}, best={self.best_score}).")
			self.best_score = current_value
			self.stopped_epoch = epoch
			self.best_epoch = epoch
			if self.restore_best_weights:
				# self.best_weights = copy.deepcopy(model.state_dict())
				self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
			self.counter = 0
			self.improvement_history.append(True)
		else:
			print(f"Epoch {epoch+1}: No improvement (current={current_value:.4f}, best={self.best_score:.4f}).")
			self.counter += 1
			self.improvement_history.append(False)
		
		trend = self.calculate_trend()
		cumulative_improvement = abs(trend) if len(self.value_history) >= self.window_size else float('inf')
		print(f">> Trend: {trend:.7f} | Cumulative Improvement: {cumulative_improvement:.7f}")
		
		should_stop = False
		if self.counter >= self.patience:
			print(f"Early stopping triggered! validation loss fails to improve for (patience={self.patience}) epochs.")
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

def get_status(model, phase, layers_to_unfreeze, cache=None):
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

	current_epoch = len(losses) + 1  # Assume epochs start at 1 for user-friendly logging
	if len(losses) < window:
		print(f"Epoch {current_epoch}: Not enough epochs ({len(losses)} < {window}) to evaluate phase transition.")
		return False
	
	# Loss analysis
	last_window_losses = losses[-window:]
	cumulative_loss_improvement = last_window_losses[0] - last_window_losses[-1]  # Positive = improvement
	loss_plateau = abs(cumulative_loss_improvement) < loss_threshold
	loss_trend = last_window_losses[-1] - last_window_losses[0]  # Positive = worsening
	close_to_best = best_loss is not None and abs(last_window_losses[-1] - best_loss) < best_loss_threshold
	sustained_improvement = cumulative_loss_improvement > loss_threshold  # Significant improvement

	# Accuracy analysis
	acc_plateau = False
	cumulative_acc_improvement = None
	if accuracies is not None and len(accuracies) >= window:
		last_window_accs = accuracies[-window:]
		cumulative_acc_improvement = last_window_accs[-1] - last_window_accs[0]  # Positive = improvement
		acc_plateau = abs(cumulative_acc_improvement) < accuracy_threshold

	# Detailed debugging prints
	print(f"Phase transition evaluation [epoch {current_epoch}]:")
	print(f"\t{window} Window losses: {last_window_losses}")
	print(f"\tCumulative loss improvement: {cumulative_loss_improvement:.6f} (threshold: {loss_threshold:.6f})")
	print(f"\tLoss plateau: {loss_plateau}")
	print(f"\tLoss trend: {loss_trend:.6f} (>0 means worsening)")
	print(f"\tClose to best loss: {close_to_best} (current: {last_window_losses[-1]:.6f}, best: {best_loss if best_loss is not None else 'N/A'}, threshold: {best_loss_threshold:.6f})")
	print(f"\tSustained loss improvement: {sustained_improvement}")
	
	if accuracies is not None and len(accuracies) >= window:
		print(f"\t{window} Window accuracies: {last_window_accs}")
		print(f"\tCumulative accuracy improvement: {cumulative_acc_improvement:.6f} (threshold: {accuracy_threshold:.6f})")
		print(f"\tAccuracy plateau: {acc_plateau}")
	else:
		print("\tAccuracy data not available for phase transition evaluation.")
	
	# Transition logic
	transition = False
	if loss_plateau:
		if loss_trend > 0:
			transition = True
			print("\tDecision: Transition due to active loss deterioration")
		elif not close_to_best and not sustained_improvement:
			transition = True
			print("\tDecision: Transition due to stagnation without proximity to best loss")
	elif acc_plateau:
		transition = True
		print("\tDecision: Transition due to accuracy plateau")
	print(f"==>> Phase Transition Required? {transition}")
	return transition

def handle_phase_transition(
		current_phase: int,
		initial_lr: float,
		max_phases: int,
		scheduler,
	):

	# 1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 3.125e-6
	if current_phase >= max_phases - 1:
		new_lr = initial_lr * (0.5 ** current_phase)  # Consistent 2x reduction
		return current_phase, new_lr

	new_phase = current_phase + 1
	new_lr = initial_lr * (0.5 ** new_phase)  # Reduce by 2x per phase
	# update schuler max_lr:
	scheduler.max_lr = new_lr
	for param_group in scheduler.optimizer.param_groups:
		param_group['lr'] = new_lr
	print(f"\tTransitioning to Phase {new_phase} with new learning rate {new_lr:.1e}")

	return new_phase, new_lr

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

def progressive_unfreeze_finetune(
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
		top_k_values: List[int] = [1, 5, 10, 15, 20],
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
		min_epochs_before_transition: int = 5,
		accumulation_steps: int = 4,  # New parameter for gradient accumulation
) -> Dict[str, any]:
		# Input validation
		if not train_loader or not validation_loader:
				raise ValueError("Train and validation loaders must not be empty.")
		
		window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches
		print(f"Training batch: {len(train_loader)}, window_size: {window_size}")

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
				f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
		)
		
		# Initialize training components
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
				lr=learning_rate,
				betas=(0.9, 0.98),
				eps=1e-8,
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

		training_losses = []
		metrics_for_all_epochs = []
		img2txt_metrics_list = []
		txt2img_metrics_list = []
		train_start_time = time.time()
		best_val_loss = float('inf')
		best_img2txt_metrics = None
		best_txt2img_metrics = None
		current_phase = 0
		epochs_in_current_phase = 0
		min_epochs_per_phase = 5
		max_epochs_per_phase = 15
		initial_learning_rate = learning_rate
		min_phases_before_stopping = 3  # Ensure model progresses through at least 3 phases before early stopping
		layer_cache = {}  # Cache for layer freezing status

		# Effective batch size and micro-batch size
		effective_batch_size = train_loader.batch_size
		micro_batch_size = effective_batch_size // accumulation_steps
		if micro_batch_size == 0:
				micro_batch_size = 1
				accumulation_steps = effective_batch_size  # Adjust accumulation steps if batch size is too small
		print(f"Effective Batch Size: {effective_batch_size}, Micro-Batch Size: {micro_batch_size}, Accumulation Steps: {accumulation_steps}")

		for epoch in range(num_epochs):
				torch.cuda.empty_cache()
				print(f"Epoch [{epoch+1}/{num_epochs}] GPU Memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
				epochs_in_current_phase += 1

				# Phase transition logic
				if epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase:
						img2txt_accs = [metrics["img2txt_acc"] for metrics in metrics_for_all_epochs]
						txt2img_accs = [metrics["txt2img_acc"] for metrics in metrics_for_all_epochs]
						avg_accs = [(img + txt) / 2 for img, txt in zip(img2txt_accs, txt2img_accs)]

						should_transition = should_transition_phase(
								losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
								accuracies=avg_accs,
								loss_threshold=min_delta * 2,
								accuracy_threshold=1e-4,
								best_loss_threshold=min_delta * 5,
								window=window_size,
								best_loss=best_val_loss,
						)

						if should_transition:
								current_phase, learning_rate = handle_phase_transition(
										current_phase=current_phase,
										initial_lr=initial_learning_rate,
										max_phases=len(unfreeze_schedule),
										scheduler=scheduler,
								)
								epochs_in_current_phase = 0  # Reset the counter after transitioning
				
				# Unfreeze layers for current phase
				unfreeze_layers(
						model=model,
						strategy=unfreeze_schedule,
						phase=current_phase,
						cache=layer_cache,
				)
				
				# Update optimizer with new learning rate
				for param_group in optimizer.param_groups:
						param_group['lr'] = learning_rate
				
				model.train()
				epoch_loss = 0.0
				optimizer.zero_grad(set_to_none=True)  # Clear gradients at the start of the epoch

				for bidx, (images, tokenized_labels, labels_indices) in enumerate(train_loader):
						# Split the batch into micro-batches
						num_samples = images.size(0)
						micro_batches = []
						for micro_idx in range(0, num_samples, micro_batch_size):
								micro_end = min(micro_idx + micro_batch_size, num_samples)
								micro_images = images[micro_idx:micro_end].to(device, non_blocking=True)
								micro_tokenized_labels = tokenized_labels[micro_idx:micro_end].to(device, non_blocking=True)
								micro_batches.append((micro_images, micro_tokenized_labels))

						# Process each micro-batch
						for micro_idx, (micro_images, micro_tokenized_labels) in enumerate(micro_batches):
								with torch.amp.autocast(device_type=device.type, enabled=True):
										logits_per_image, logits_per_text = model(micro_images, micro_tokenized_labels)
										ground_truth = torch.arange(len(micro_images), dtype=torch.long, device=device)
										loss_img = criterion(logits_per_image, ground_truth)
										loss_txt = criterion(logits_per_text, ground_truth)
										total_loss = 0.5 * (loss_img + loss_txt)

								# Scale the loss to account for accumulation (normalize by accumulation_steps)
								scaled_loss = total_loss / accumulation_steps
								scaler.scale(scaled_loss).backward()

								# Accumulate gradients
								if (micro_idx + 1) % accumulation_steps == 0 or micro_idx == len(micro_batches) - 1:
										torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
										scaler.step(optimizer)
										scaler.update()
										optimizer.zero_grad(set_to_none=True)  # Clear gradients after optimization step
										scheduler.step()

								epoch_loss += total_loss.item() * micro_images.size(0)  # Accumulate loss for reporting

						if bidx % print_every == 0 or bidx + 1 == len(train_loader):
								print(f"Batch [{bidx+1}/{len(train_loader)}] Loss: {total_loss.item():.7f}")
				
				avg_training_loss = epoch_loss / len(train_loader.dataset)  # Normalize by total samples
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
						print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
						best_val_loss = current_val_loss
						checkpoint.update({"best_val_loss": best_val_loss})
						torch.save(checkpoint, mdl_fpth)
						best_img2txt_metrics = img2txt_metrics
						best_txt2img_metrics = txt2img_metrics

				# Early stopping
				if early_stopping.should_stop(current_val_loss, model, epoch):
						if current_phase >= min_phases_before_stopping:
								print(f"Early stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
								break
						else:
								print(f"Early stopping condition met at epoch {epoch + 1}! but delaying until minimum phases ({min_phases_before_stopping}) are reached. Current phase: {current_phase}")
				print("-" * 140)

		print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

		file_base_name = (
				f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
				f"ep_{len(training_losses)}_init_lr_{initial_learning_rate:.1e}_final_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
				f"bs_{train_loader.batch_size}_dropout_{dropout_val}"
		)

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

def progressive_unfreeze_finetune_original(
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
		top_k_values: List[int] = [1, 5, 10, 15, 20],
		layer_groups_to_unfreeze: List[str] = ['visual_frontend', 'visual_transformer', 'text_frontend', 'text_transformer', 'projections'],
		min_epochs_before_transition: int = 5,
	) -> Dict[str, any]:
	# Input validation
	if not train_loader or not validation_loader:
		raise ValueError("Train and validation loaders must not be empty.")
	
	window_size = max(5, int(0.1 * len(train_loader)))  # 10% of training batches
	print(f"training batch: {len(train_loader)}, window_size: {window_size}")

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
		f"dropout_{dropout_val}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pth"
	)
	
	# Initialize training components
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
		lr=learning_rate,
		betas=(0.9, 0.98),
		eps=1e-8,
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

	training_losses = []
	metrics_for_all_epochs = []
	img2txt_metrics_list = []
	txt2img_metrics_list = []
	train_start_time = time.time()
	best_val_loss = float('inf')
	best_img2txt_metrics = None
	best_txt2img_metrics = None
	current_phase = 0
	epochs_in_current_phase = 0
	min_epochs_per_phase = 5
	max_epochs_per_phase = 15
	initial_learning_rate = learning_rate
	min_phases_before_stopping = 3 # ensure model progresses through at least 3 phases (unfreezing 60% of transformer blocks) before early stopping can trigger
	layer_cache = {} # Cache for layer freezing status

	for epoch in range(num_epochs):
		torch.cuda.empty_cache()
		print(f"Epoch [{epoch+1}/{num_epochs}] GPU Memory usage: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
		epochs_in_current_phase += 1 # Increment the epoch counter for the current phase

		# Phase transition logic
		if epoch >= min_epochs_before_transition and epochs_in_current_phase >= min_epochs_per_phase:
			img2txt_accs = [metrics["img2txt_acc"] for metrics in metrics_for_all_epochs]
			txt2img_accs = [metrics["txt2img_acc"] for metrics in metrics_for_all_epochs]
			avg_accs = [(img + txt) / 2 for img, txt in zip(img2txt_accs, txt2img_accs)]

			should_transition = should_transition_phase(
				losses=[metrics["val_loss"] for metrics in metrics_for_all_epochs],
				# accuracies=[metrics["img2txt_acc"] for metrics in metrics_for_all_epochs],
				accuracies=avg_accs,
				# loss_threshold=cumulative_delta, # Align with early stopping cumulative_delta
				loss_threshold=min_delta * 2, # More tolerant threshold
				accuracy_threshold=1e-4,
				best_loss_threshold=min_delta * 5,#5e-3,
				window=window_size,
				best_loss=best_val_loss,
			)

			if should_transition:
				current_phase, learning_rate = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=len(unfreeze_schedule),
					scheduler=scheduler,
				)
				epochs_in_current_phase = 0  # Reset the counter after transitioning
				# # TODO: Reset early stopping between phases
				# early_stopping.reset()
		
		# Unfreeze layers for current phase
		unfreeze_layers(
			model=model,
			strategy=unfreeze_schedule,
			phase=current_phase,
			cache=layer_cache,
		)
		
		# Update optimizer with new learning rate
		for param_group in optimizer.param_groups:
			param_group['lr'] = learning_rate
		
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
			print(f"New best model found (loss {current_val_loss:.5f} < {best_val_loss:.5f})")
			best_val_loss = current_val_loss
			checkpoint.update({"best_val_loss": best_val_loss})
			torch.save(checkpoint, mdl_fpth)
			best_img2txt_metrics = img2txt_metrics
			best_txt2img_metrics = txt2img_metrics

		# Early stopping
		if early_stopping.should_stop(current_val_loss, model, epoch):
			if current_phase >= min_phases_before_stopping:
				print(f"Early stopping at epoch {epoch + 1}. Best loss: {early_stopping.get_best_score():.5f}")
				break
			else:
				print(f"Early stopping condition met at epoch {epoch + 1}! but delaying until minimum phases ({min_phases_before_stopping}) are reached. Current phase: {current_phase}")
		print("-" * 140)

	print(f"Elapsed_t: {time.time() - train_start_time:.1f} sec".center(170, "-"))

	file_base_name = (
		f"{dataset_name}_{mode}_{model_name}_{re.sub('/', '', model_arch)}_"
		f"ep_{len(training_losses)}_init_lr_{initial_learning_rate:.1e}_final_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_dropout_{dropout_val}"
	)

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

	print(f"Pretrain Evaluation {dataset_name} with {validation_loader.dataset.__class__.__name__} samples| {model_name} - {model_arch} | {device}".center(170, "-"))
	
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

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="FineTune CLIP for Balanced Dataset. Note: 'train' mode always initializes with random weights, while 'pretrain' and 'finetune' use pre-trained OpenAI weights.")
	parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
	parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs')
	parser.add_argument('--epochs', '-e', type=int, default=12, help='Number of epochs')
	parser.add_argument('--batch_size', '-bs', type=int, default=8, help='Batch size for training')
	parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='small learning rate for better convergence [def: 1e-4]')
	parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='Weight decay [def: 1e-3]')
	parser.add_argument('--print_every', type=int, default=250, help='Print every [def: 250]')
	parser.add_argument('--model_architecture', '-a', type=str, default="ViT-B/32", help='CLIP Architecture (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)')
	parser.add_argument('--dataset', '-d', type=str, choices=['cifar10', 'cifar100', 'cinic10', 'imagenet', 'svhn'], default='cifar100', help='Choose dataset (CIFAR10/cifar100)')
	parser.add_argument('--mode', '-m', type=str, choices=['pretrain', 'train', 'finetune'], default='pretrain', help='Choose mode (pretrain/train/finetune)')
	parser.add_argument('--finetune_strategy', '-fts', type=str, choices=['full', 'lora', 'progressive'], default='full', help='Fine-tuning strategy (full/lora/progressive) when mode is finetune')
	parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank (used if finetune_strategy=lora)')
	parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha (used if finetune_strategy=lora)')
	parser.add_argument('--lora_dropout', type=float, default=0.0, help='Regularizes trainable LoRA parameters, [primary focus of fine-tuning]')
	parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
	parser.add_argument('--minimum_delta', '-mdelta', type=float, default=1e-4, help='Min delta for early stopping & progressive freezing [Platueau threshhold]')
	parser.add_argument('--cumulative_delta', '-cdelta', type=float, default=5e-3, help='Cumulative delta for early stopping')
	parser.add_argument('--minimum_epochs', type=int, default=15, help='Early stopping minimum epochs')
	parser.add_argument('--topK_values', '-k', type=int, nargs='+', default=[1, 5, 10, 15, 20], help='Top K values for retrieval metrics')
	parser.add_argument('--dropout', '-do', type=float, default=0.0, help='Dropout rate for the base model')

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print(type(args.device), args.device, torch.cuda.device_count(), args.device.index)

	print_args_table(args=args, parser=parser)
	set_seeds()
	print(clip.available_models()) # List all available CLIP models
	# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	print(f">> CLIP Model Architecture: {args.model_architecture}...")
	model_config = get_config(architecture=args.model_architecture, dropout=args.dropout,)

	print(json.dumps(model_config, indent=4, ensure_ascii=False))
	model, preprocess = clip.load(
		name=args.model_architecture,
		device=args.device, 
		jit=False, # training or finetuning => jit=False
		random_weights=True if args.mode == 'train' else False, 
		dropout=args.dropout,
	)
	model = model.float() # Convert model parameters to FP32
	model.name = args.model_architecture  # Custom attribute to store model name
	print(f"Model: {model.__class__.__name__} loaded with {model.name} architecture on {args.device} device")
	# print(model.visual.conv1.weight[0, 0, 0])  # Random value (not zeros or pretrained values)
	# print(f"embed_dim: {model.text_projection.size(0)}, transformer_width: {model.text_projection.size(1)}")

	train_loader, validation_loader = get_dataloaders(
		dataset_name=args.dataset,
		batch_size=args.batch_size,
		nw=args.num_workers,
		USER=os.environ.get('USER'),
		input_resolution=model_config["image_resolution"],
		preprocess=None,# preprocess,
	)
	print_loader_info(loader=train_loader, batch_size=args.batch_size)
	print_loader_info(loader=validation_loader, batch_size=args.batch_size)

	# visualize_(dataloader=train_loader, num_samples=5)

	if args.mode == 'finetune':
		if args.finetune_strategy == 'full':
			full_finetune(
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
				patience=10, 										# early stopping
				min_delta=1e-4, 								# early stopping & progressive unfreezing
				cumulative_delta=5e-3, 					# early stopping
				minimum_epochs=20, 							# early stopping
				TOP_K_VALUES=args.topK_values,
			)
		elif args.finetune_strategy == 'lora':
			lora_finetune(
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
				lora_rank=args.lora_rank,
				lora_alpha=args.lora_alpha,
				lora_dropout=args.lora_dropout,
				patience=args.patience,
				min_delta=args.minimum_delta,
				cumulative_delta=args.cumulative_delta,
				minimum_epochs=args.minimum_epochs,
				TOP_K_VALUES=args.topK_values,
			)
		elif args.finetune_strategy == 'progressive':
			progressive_unfreeze_finetune(
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
				patience=10,									# early stopping
				min_delta=1e-4,								# early stopping
				cumulative_delta=5e-3,				# early stopping and progressive unfreezing
				minimum_epochs=20,						# early stopping
				top_k_values=args.topK_values,
			)
		else:
			raise ValueError(f"Invalid mode: {args.mode}")
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
			patience=10,									# early stopping
			min_delta=1e-4,								# early stopping
			cumulative_delta=5e-3,				# early stopping
			minimum_epochs=20,						# early stopping
			TOP_K_VALUES=args.topK_values,
		)
	elif args.mode == "pretrain":
		all_img2txt_metrics = {}
		all_txt2img_metrics = {}
		available_models = clip.available_models() # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		for model_arch in available_models[::-1]:
			print(f"Evaluating pre-trained {model_arch}")
			model_config = get_config(architecture=model_arch, dropout=args.dropout,)
			print(json.dumps(model_config, indent=4, ensure_ascii=False))

			model, preprocess = clip.load(
				name=model_arch,
				device=args.device,
				random_weights=False,
				dropout=args.dropout,
			)
			model = model.float()
			model.name = model_arch  # Custom attribute to store model name
			print(f"{model.__class__.__name__} - {model_arch} loaded successfully")
			train_loader, validation_loader = get_dataloaders(
				dataset_name=args.dataset,
				batch_size=args.batch_size,
				nw=args.num_workers,
				USER=os.environ.get('USER'),
				input_resolution=model_config["image_resolution"],
				preprocess=None,
				# preprocess=preprocess,
			)
			print_loader_info(loader=train_loader, batch_size=args.batch_size)
			print_loader_info(loader=validation_loader, batch_size=args.batch_size)

			img2txt_metrics, txt2img_metrics = pretrain(
				model=model,
				validation_loader=validation_loader,
				results_dir=os.path.join(args.dataset, "results"),
				device=args.device,
				TOP_K_VALUES=args.topK_values,
			)
			all_img2txt_metrics[model_arch] = img2txt_metrics
			all_txt2img_metrics[model_arch] = txt2img_metrics
			del model  # Clean up memory
			torch.cuda.empty_cache()
		# Pass all metrics to the new visualization function
		plot_all_pretrain_metrics(
			dataset_name=args.dataset,
			img2txt_metrics_dict=all_img2txt_metrics,
			txt2img_metrics_dict=all_txt2img_metrics,
			results_dir=os.path.join(args.dataset, "results"),
			topK_values=args.topK_values,
		)
	else:
		raise ValueError("Invalid mode. Choose either 'finetune' or 'train'.")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))