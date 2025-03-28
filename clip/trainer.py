from utils import *
from model import get_lora_clip
from visualize import plot_loss_accuracy_metrics, plot_retrieval_metrics_best_model, plot_retrieval_metrics_per_epoch, plot_all_pretrain_metrics

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
	
	def should_stop(self, current_value: float, model: torch.nn.Module, epoch: int, current_phase: int) -> bool:
		# --- Update State ---
		self.value_history.append(current_value)
		self.current_phase = current_phase # Update internal phase tracker
		print(f"\n--- EarlyStopping Check (Epoch {epoch+1}, Phase {current_phase}) ---")
		print(f"Current Value: {current_value:.6f}")
		# --- Initial Checks ---
		# 1. Minimum Epochs Check: Don't stop if fewer than min_epochs have run.
		if epoch < self.min_epochs:
			print(f"Skipping early stopping check (epoch {epoch+1} < min_epochs {self.min_epochs})")
			return False # Continue training
		# --- Improvement Tracking ---
		# 2. Check if the current value is an improvement over the best score seen so far.
		improved = self.is_improvement(current_value)
		if improved:
				print(f"\tImprovement detected! Best: {self.best_score if self.best_score is not None else 'N/A'} -> {current_value:.6f} (delta: {self.min_delta})")
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
		print(f"\tCumulative Improvement over window: {cumulative_improvement_signed:.5f} (Threshold for lack of imp: < {self.cumulative_delta})")
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
		if self.mode == 'min' and slope > self.slope_threshold:
			is_worsening = True
		elif self.mode == 'max' and slope < self.slope_threshold:
			is_worsening = True
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
		# 5. Decide whether to actually stop based on reasons and minimum phases.
		should_really_stop = False
		if stop_reason:
			reason_str = ', '.join(stop_reason)
			# Check if the minimum number of training phases has been completed.
			if current_phase >= self.min_phases_before_stopping:
				print(f"EARLY STOPPING TRIGGERED (Phase {current_phase} >= {self.min_phases_before_stopping}): {reason_str}")
				should_really_stop = True
			else:
				print(f"\tStopping condition met ({reason_str}), but delaying stop (Phase {current_phase} < {self.min_phases_before_stopping})")
		else:
			print("\tNo stopping conditions met.")
		# --- Restore Best Weights (if stopping) ---
		# 6. If stopping and configured, load the best saved weights back into the model.
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
			"current_phase": self.current_phase,
			"value_history_len": len(self.value_history)
		}
		if len(self.value_history) >= self.window_size:
			last_window = self.value_history[-self.window_size:]
			status["volatility_window"] = self.compute_volatility(last_window)
			status["slope_window"] = compute_slope(last_window) # Use global
		else:
			status["volatility_window"] = None
			status["slope_window"] = None
		return status
	
	def get_best_score(self) -> Optional[float]:
		return self.best_score
	
	def get_best_epoch(self) -> int:
		return self.best_epoch # 0-based

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
	torch.cuda.empty_cache() # Clear GPU memory cache

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

def get_loss_accuracy_metrics(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str = "cuda",
		topK_values: List = [1, 5, 10, 15, 20],
	):
	"""
	Evaluate the CLIP model's performance on the full validation set.
	Computes loss per batch and accuracy over all samples.
	"""
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} - {model_arch} [Full Loss & Accuracy] [{dataset_name}]: {topK_values}...")
	model.eval()
	total_loss = 0
	num_batches = len(validation_loader)
	total_samples = len(validation_loader.dataset)
	all_image_embeds = []
	all_text_embeds = []
	metrics = {}
	# Step 1: Collect embeddings and compute per-batch loss
	with torch.no_grad():
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(validation_loader):
			images = images.to(device, non_blocking=True)
			tokenized_labels = tokenized_labels.to(device, non_blocking=True)
			batch_size = images.size(0)
			# Forward pass
			logits_per_image, logits_per_text = model(images, tokenized_labels)
			image_embeds = model.encode_image(images)  # Extract normalized embeddings
			text_embeds = model.encode_text(tokenized_labels)
			image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
			text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
			# Compute batch loss
			ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
			loss_img = criterion(logits_per_image, ground_truth)
			loss_txt = criterion(logits_per_text, ground_truth)
			batch_loss = 0.5 * (loss_img.item() + loss_txt.item())
			total_loss += batch_loss
			# Collect embeddings
			all_image_embeds.append(image_embeds.cpu())
			all_text_embeds.append(text_embeds.cpu())
	# Average loss
	avg_val_loss = total_loss / num_batches
	metrics = {"val_loss": float(avg_val_loss)}
	# Step 2: Concatenate all embeddings
	all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [total_samples, embed_dim]
	all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [total_samples, embed_dim]
	all_image_embeds = all_image_embeds.to(device)
	all_text_embeds = all_text_embeds.to(device)
	# Step 3: Compute full similarity matrix
	similarities_img2txt = all_image_embeds @ all_text_embeds.T  # [total_samples, total_samples]
	# Step 4: Compute full validation set accuracy
	ground_truth = torch.arange(total_samples, device=device)

	# Image-to-Text Retrieval
	for k in topK_values:
		topk_img2txt = similarities_img2txt.topk(k, dim=1).indices
		img2txt_acc_k = (topk_img2txt == ground_truth.view(-1, 1)).any(dim=1).float().mean().item()
		metrics[f"img2txt_acc@{k}"] = float(img2txt_acc_k)

	# Text-to-Image Retrieval
	for k in topK_values:
		topk_txt2img = similarities_img2txt.T.topk(k, dim=1).indices
		txt2img_acc_k = (topk_txt2img == ground_truth.view(-1, 1)).any(dim=1).float().mean().item()
		metrics[f"txt2img_acc@{k}"] = float(txt2img_acc_k)

	return metrics

def get_in_batch_loss_accuracy(
		model: torch.nn.Module,
		validation_loader: DataLoader,
		criterion: torch.nn.Module,
		device: str = "cuda",
		topK_values: List = [1, 3, 5],
	):
	dataset_name = validation_loader.name
	model_name = model.__class__.__name__
	model_arch = model.name
	print(f">> Evaluating {model_name} - {model_arch} [in-batch Loss & Accuracy] [{dataset_name}]: {topK_values}...")

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
		accuracies: Optional[List[float]], # Added optional accuracy list
		window: int,
		best_loss: Optional[float],
		best_loss_threshold: float = 1e-3,
		volatility_threshold: float = 10.0,
		slope_threshold: float = 0.0,
		pairwise_imp_threshold: float = 5e-3,
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
		current_phase: int,                # Input: The index of the phase just completed (0-based)
		initial_lr: float,                 # Input: The LR the training started with
		max_phases: int,                   # Input: Total number of phases defined (e.g., length of unfreeze_schedule)
		window_size: int,                  # Input: Window size used for analysis (affects window_factor)
		current_loss: float,               # Input: Validation loss from the most recent epoch
		best_loss: Optional[float]         # Input: Best validation loss seen so far (can be None)
	) -> Tuple[int, float]:                # Output: (new_phase_index, new_learning_rate)
	
	# --- 1. Calculate Loss Ratio ---
	# This factor scales the LR based on how the current loss compares to the best loss.
	if best_loss is None or best_loss <= 0:
		# If no best loss yet, or best loss is invalid, use a neutral ratio of 1.0.
		loss_ratio = 1.0
	else:
		# Calculate ratio: current / best.
		# Clamp the ratio between 0.5 and 2.0 to prevent extreme scaling effects
		# if the current loss is drastically different from the best loss. Adds stability.
		loss_ratio = min(max(0.5, current_loss / best_loss), 2.0)
	
	# --- 2. Calculate Window Factor ---
	# Scales LR based on the window size. The formula 10 / window_size means
	# smaller windows result in a potentially larger factor (up to 1.5).
	# This might imply wanting more aggressive LR changes if decisions are based on shorter histories.
	# Clamped between 0.5 and 1.5 for stability. This is a tuning parameter.
	window_factor = max(0.5, min(1.5, 10 / window_size))
	
	# --- 3. Determine Next Phase Index and Phase Factor ---
	next_phase = current_phase + 1 # Tentative next phase index
	# Check if we are transitioning beyond the last defined phase
	if next_phase >= max_phases:
		# Stay in the last defined phase (index = max_phases - 1).
		next_phase = max_phases - 1
		# Apply a fixed, aggressive LR reduction factor since transition was triggered
		# even in the final phase, suggesting strong need for LR drop.
		phase_factor = 0.1
		print(f"<!> Already in final phase ({current_phase}). Applying fixed LR reduction.")
	else:
		# Calculate progress through the phases (approx. 0 to 1).
		# `max(1, max_phases - 1)` prevents division by zero if max_phases is 1.
		phase_progress = next_phase / max(1, max_phases - 1)
		# Calculate an exponential decay factor based on phase progress.
		# As training progresses (phase_progress -> 1), the factor decreases (towards 0.75).
		# This generally reduces the LR more significantly in later phases. Base 0.75 is a tuning choice.
		phase_factor = 0.75 ** phase_progress
	
	# --- 4. Calculate New Learning Rate ---
	# Combine the initial LR with all calculated scaling factors.
	new_lr = initial_lr * phase_factor * loss_ratio * window_factor
	# Define a floor for the learning rate (e.g., 0.1% of the initial LR)
	# to prevent it from becoming extremely small or zero.
	min_allowable_lr = initial_lr * 1e-3
	# Enforce the minimum learning rate.
	new_lr = max(new_lr, min_allowable_lr)
	
	print(f"\n--- Phase Transition Occurred (Moving to Phase {next_phase}) ---")
	print(f"Previous Phase: {current_phase}")
	print(f"Factors -> Loss Ratio: {loss_ratio:.3f}, Window Factor: {window_factor:.3f}, Phase Factor: {phase_factor:.3f}")
	print(f"Calculated New LR: {new_lr:.3e} (min allowable: {min_allowable_lr:.3e})")
		
	# Return the index of the phase we are *entering* and the new learning rate.
	return next_phase, new_lr

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
		nw: int, # num_workers for DataLoader
		print_every: int, # Print frequency within epoch
		initial_learning_rate: float,
		weight_decay: float,
		device: str,
		results_dir: str,
		window_size: int,
		patience: int = 10,
		min_delta: float = 1e-4, # Make slightly less sensitive than default
		cumulative_delta: float = 5e-3, # Keep cumulative check reasonable
		minimum_epochs: int = 15, # Minimum epochs before ANY early stop
		min_epochs_per_phase: int = 5, # Minimum epochs within a phase before transition check
		volatility_threshold: float = 15.0, # Allow slightly more volatility
		slope_threshold: float = 1e-4, # Allow very slightly positive slope before stopping/transitioning
		pairwise_imp_threshold: float = 1e-4, # Stricter requirement for pairwise improvement
		accuracy_plateau_threshold: float = 5e-4, # For phase transition based on accuracy
		min_phases_before_stopping: int = 3, # Ensure significant unfreezing before global stop
		top_k_values: list[int] = [1, 5, 10],
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
		min_phases_before_stopping=min_phases_before_stopping
	)

	try:
		dataset_name = validation_loader.dataset.dataset.__class__.__name__
	except:
		dataset_name = validation_loader.dataset.dataset_name
	mode_name = inspect.stack()[0].function
	model_arch = re.sub('/', '', model.name) if hasattr(model, 'name') else 'unknown_arch'
	model_class_name = model.__class__.__name__
	base_filename = f"{dataset_name}_{mode_name}_{model_class_name}_{model_arch}"
	best_model_path = os.path.join(results_dir, f"{base_filename}_best_model.pth")

	# Find dropout value
	dropout_val = 0.0
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Dropout):
			dropout_val = module.p
			break
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
	# Optimizer and Scheduler
	# Filter parameters dynamically based on requires_grad, which changes per phase
	optimizer = AdamW(
		params=filter(lambda p: p.requires_grad, model.parameters()), # Initially might be empty if phase 0 has no unfrozen layers
		lr=initial_learning_rate,
		betas=(0.9, 0.98),
		eps=1e-6,
		weight_decay=weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer=optimizer,
		max_lr=initial_learning_rate,
		steps_per_epoch=len(train_loader),
		epochs=num_epochs,
		pct_start=0.1, # Standard pct_start
		anneal_strategy='cos' # Cosine annealing
	)

	criterion = torch.nn.CrossEntropyLoss()
	scaler = torch.amp.GradScaler(device=device) # For mixed precision

	# --- Training State ---
	current_phase = 0
	epochs_in_current_phase = 0
	training_losses = [] # History of average training loss per epoch
	img2txt_metrics_all_epochs = list()
	txt2img_metrics_all_epochs = list()
	in_batch_loss_acc_metrics_all_epochs = list() # History of [in-batch] validation metrics dicts per epoch
	loss_acc_metrics_all_epochs = list() # History of validation metrics dicts per epoch
	best_val_loss = None # Track the absolute best validation loss
	layer_cache = {} # Cache for layer status (optional, used by get_status)
	last_lr = initial_learning_rate # Track current LR
	phase_just_changed = False # Flag to signal optimizer refresh needed
	print(f"\nStarting Training: {base_filename}...")
	print(f"Epochs: {num_epochs} | Device: {device} | Initial LR: {initial_learning_rate:.2e}")
	print(f"Optimizer: AdamW | Scheduler: OneCycleLR | Criterion: CrossEntropyLoss")
	print(f"Early Stopping: Patience={patience}, MinDelta={min_delta}, cimulativeDelta={cumulative_delta}, Window={window_size}, MinEpochs={minimum_epochs}, MinPhases={min_phases_before_stopping}")
	print(f"Phase Transition: Window={window_size}, MinEpochsInPhase={min_epochs_per_phase}")
	print("-" * 80)

	# --- Main Training Loop ---
	train_start_time = time.time()
	for epoch in range(num_epochs):
		epoch_start_time = time.time()
		print(f"\n=== Epoch {epoch+1}/{num_epochs} | Phase {current_phase} | LR: {last_lr:.3e} ===")
		torch.cuda.empty_cache()
		# --- Phase Transition Check ---
		# Check only if enough epochs *overall* and *within the phase* have passed,
		# and if we are not already in the last phase.
		if (epoch >= minimum_epochs and # Overall min epochs check
			epochs_in_current_phase >= min_epochs_per_phase and
			current_phase < max_phases - 1 and
			len(early_stopping.value_history) >= window_size):
			print(f"Checking for phase transition (Epochs in phase: {epochs_in_current_phase})")
			# Extract necessary data for should_transition_phase
			val_losses = early_stopping.value_history

			val_accs = [m.get('img2txt_acc', 0.0) + m.get('txt2img_acc', 0.0) / 2.0 for m in in_batch_loss_acc_metrics_all_epochs] if in_batch_loss_acc_metrics_all_epochs else None
			should_trans = should_transition_phase(
				losses=val_losses,
				accuracies=val_accs, # Pass average accuracy
				window=window_size,
				best_loss=early_stopping.get_best_score(), # Use best score from early stopping state
				best_loss_threshold=min_delta, # Use min_delta for closeness check
				volatility_threshold=volatility_threshold,
				slope_threshold=slope_threshold, # Use positive threshold for worsening loss
				pairwise_imp_threshold=pairwise_imp_threshold,
				accuracy_plateau_threshold=accuracy_plateau_threshold
			)
			if should_trans:
				current_phase, last_lr = handle_phase_transition(
					current_phase=current_phase,
					initial_lr=initial_learning_rate,
					max_phases=max_phases,
					window_size=window_size,
					current_loss=val_losses[-1],
					best_loss=early_stopping.get_best_score()
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
			cache=layer_cache # Optional cache
		)
		if phase_just_changed or epoch == 0:
			print("Refreshing optimizer parameter groups...")
			optimizer.param_groups.clear()
			optimizer.add_param_group({'params': [p for p in model.parameters() if p.requires_grad], 'lr': last_lr})
			print(f"Optimizer parameter groups refreshed. LR set to {last_lr:.3e}.")
			
			# --- Re-initialize Scheduler (Option B Implementation) ---
			print("Re-initializing OneCycleLR scheduler for new phase/start...")
			steps_per_epoch = len(train_loader)
			# Decide on remaining epochs for the new schedule
			# Option 1: Schedule over all original epochs (simpler, might finish annealing early)
			# scheduler_epochs = num_epochs
			# Option 2: Schedule over remaining epochs (more adaptive)
			scheduler_epochs = num_epochs - epoch
			# Ensure scheduler_epochs is at least 1
			scheduler_epochs = max(1, scheduler_epochs)
			scheduler = torch.optim.lr_scheduler.OneCycleLR(
				optimizer=optimizer,
				max_lr=last_lr, # Use the new LR as the peak for the new cycle
				steps_per_epoch=steps_per_epoch,
				epochs=scheduler_epochs, # Schedule over remaining or total epochs
				pct_start=0.1, # Consider if this needs adjustment in later phases
				anneal_strategy='cos',
				# last_epoch = -1 # Ensures it starts fresh
			)
			print(f"Scheduler re-initialized with max_lr={last_lr:.3e} for {scheduler_epochs} epochs.")
			# --- End Scheduler Re-initialization ---
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
				# Assuming batch_data unpacks correctly
				images, tokenized_labels, _ = batch_data # Adjust unpacking as needed
				images = images.to(device, non_blocking=True)
				tokenized_labels = tokenized_labels.to(device, non_blocking=True)
				optimizer.zero_grad(set_to_none=True)
				with torch.amp.autocast(device_type=device.type, enabled=True):
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
				if bidx % print_every == 0 or bidx + 1 == num_train_batches:
					print(f"\tBatch [{bidx+1}/{num_train_batches}] Loss: {batch_loss_item:.6f}")
		avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 and trainable_params_exist else 0.0
		training_losses.append(avg_epoch_train_loss)
		
		# --- Validation ---
		print(f"Epoch: {epoch+1} validation...")
		# Compute in-batch loss/accuracy metrics on validation set:
		in_batch_loss_acc_metrics_per_epoch = get_in_batch_loss_accuracy(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=top_k_values
		)
		in_batch_loss_acc_metrics_all_epochs.append(in_batch_loss_acc_metrics_per_epoch)
		loss_acc_metrics_per_epoch = get_loss_accuracy_metrics(
			model=model,
			validation_loader=validation_loader,
			criterion=criterion,
			device=device,
			topK_values=top_k_values
		)
		loss_acc_metrics_all_epochs.append(loss_acc_metrics_per_epoch)

		# Compute retrieval-based metrics
		img2txt_metrics_per_epoch, txt2img_metrics_per_epoch = evaluate_retrieval_performance(
			model=model,
			validation_loader=validation_loader,
			device=device,
			topK_values=top_k_values,
		)
		img2txt_metrics_all_epochs.append(img2txt_metrics_per_epoch)
		txt2img_metrics_all_epochs.append(txt2img_metrics_per_epoch)

		current_val_loss = in_batch_loss_acc_metrics_per_epoch.get("val_loss", float('inf')) # Handle missing key safely
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode_name}'
			f'(Training): {avg_epoch_train_loss} '
			f'Validation(in-batch): {in_batch_loss_acc_metrics_per_epoch.get("val_loss", float("inf"))} '
			f'Validation(full): {loss_acc_metrics_per_epoch.get("val_loss", float("inf"))}\n'
			f'\tValidation Accuracy:\n'
			f'\tIn-batch: (Top-1) '
			f'[text retrieval per image]: {in_batch_loss_acc_metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {in_batch_loss_acc_metrics_per_epoch.get("txt2img_acc")}\n'
			f'\tFull Validation Set:\n'
			f'\t\tText retrieval per image (img2txt):\n'
			f'\t\t\t' + '\n\t\t\t'.join([f'accuracy@{k}: {loss_acc_metrics_per_epoch.get(f"img2txt_acc@{k}", "N/A"):.4f}' for k in top_k_values]) + '\n'
			f'\t\tImage retrieval per text (txt2img):\n'
			f'\t\t\t' + '\n\t\t\t'.join([f'accuracy@{k}: {loss_acc_metrics_per_epoch.get(f"txt2img_acc@{k}", "N/A"):.4f}' for k in top_k_values])
		)

		# --- Checkpointing Best Model ---
		# Use the best_score from EarlyStopping state as it's more robust
		current_best_from_stopper = early_stopping.get_best_score()
		if current_best_from_stopper is not None and current_val_loss <= current_best_from_stopper :
			# This logic relies on early_stopping.is_improvement having updated best_score
			# Let's refine this slightly for clarity:
			if early_stopping.is_improvement(current_val_loss) or best_val_loss is None: # Checks min_delta
				if best_val_loss is None or current_val_loss < best_val_loss: # Update absolute best
					print(f"*** New Best Validation Loss Found: {current_val_loss:.6f} (Epoch {epoch+1}) ***")
					best_val_loss = current_val_loss
					# Save the actual best model state (could be from stopper's cache or current model)
					if early_stopping.restore_best_weights and early_stopping.best_weights is not None:
						# Save the state dict stored by early stopping
						torch.save(early_stopping.best_weights, best_model_path)
						print(f"Best model weights (from epoch {early_stopping.best_epoch+1}) saved to {best_model_path}")
					else:
						# Save current model state if not restoring or no weights saved yet
						torch.save(model.state_dict(), best_model_path)
						print(f"Best model weights (current epoch {epoch+1}) saved to {best_model_path}")

		# --- Early Stopping Check ---
		stop_training = early_stopping.should_stop(
			current_value=current_val_loss,
			model=model,
			epoch=epoch,
			current_phase=current_phase
		)

		if stop_training:
			print(f"--- Training stopped early at epoch {epoch+1} ---")
			break # Exit the main training loop

		# --- End of Epoch ---
		epochs_in_current_phase += 1
		epoch_duration = time.time() - epoch_start_time
		print(f"Epoch {epoch+1} Duration: {epoch_duration:.2f}s")
		print(f"EarlyStopping Status:\n{json.dumps(early_stopping.get_status(), indent=2, ensure_ascii=False)}")
		print("-" * 80)

	# --- End of Training ---
	total_training_time = time.time() - train_start_time
	print(f"\n--- Training Finished ---")
	print(f"Total Epochs Run: {epoch + 1}")
	print(f"Final Phase Reached: {current_phase}")
	# print(f"Best Validation Loss Achieved: {early_stopping.get_best_score():.6f} at Epoch {early_stopping.get_best_epoch() + 1}")
	print(f"Total Training Time: {total_training_time:.2f}s")
	# --- Final Evaluation & Plotting ---
	# Load the best model weights for final evaluation
	if early_stopping.restore_best_weights and os.path.exists(best_model_path):
		print(f"\nLoading best model weights from {best_model_path} for final evaluation...")
		# Load weights carefully, handling potential device mismatches if needed
		best_weights_loaded = torch.load(best_model_path, map_location=device)
		# Check if it's a state_dict or the full stopper cache
		if isinstance(best_weights_loaded, dict) and not any(k.startswith('best_score') for k in best_weights_loaded.keys()): # Heuristic for state_dict
			model.load_state_dict(best_weights_loaded)
		elif isinstance(best_weights_loaded, dict) and 'model_state_dict' in best_weights_loaded: # Check if it's a checkpoint dict
			model.load_state_dict(best_weights_loaded['model_state_dict'])
		else:
			print("Warning: Loaded best model file format not recognized as state_dict. Using weights stored in EarlyStopping if available.")
			if early_stopping.best_weights is not None:
				model.load_state_dict({k: v.to(device) for k, v in early_stopping.best_weights.items()})

	print("\nPerforming final evaluation on the best model...")
	final_metrics = get_in_batch_loss_accuracy(
		model=model,
		validation_loader=validation_loader,
		criterion=criterion,
		device=device,
		topK_values=top_k_values,
	)

	final_img2txt_metrics, final_txt2img_metrics = evaluate_retrieval_performance(
		model=model,
		validation_loader=validation_loader,
		device=device,
		topK_values=top_k_values,
	)

	print("\n--- Final Metrics[Validation Loss & Accuracy] (Best Model) ---")
	print(json.dumps(final_metrics, indent=2))

	print("Image-to-Text Retrieval:")
	print(json.dumps(final_img2txt_metrics, indent=2))

	print("Text-to-Image Retrieval:")
	print(json.dumps(final_txt2img_metrics, indent=2))

	print("\nGenerating result plots...")

	file_base_name = (
		f"{dataset_name}_"
		f"{model_class_name}_"
		f"{re.sub('/', '', model_arch)}_"
		f"{mode_name}_"
		f"last_phase_{current_phase}_"
		f"ep_{len(training_losses)}_"
		f"wd_{weight_decay:.1e}_"
		f"bs_{train_loader.batch_size}_"
		f"dropout_{dropout_val}_"
		f"init_lr_{initial_learning_rate:.1e}"
	)

	if last_lr is not None:
		file_base_name += f"_final_lr_{last_lr:.1e}"

	plot_paths = {
		"losses": os.path.join(results_dir, f"{file_base_name}_losses.png"),
		"val_acc": os.path.join(results_dir, f"{file_base_name}_in_batch_top1_accuracy_t2i_i2t.png"),
		"img2txt_topk": os.path.join(results_dir, f"{file_base_name}_img2txt_topk_accuracy.png"),
		"txt2img_topk": os.path.join(results_dir, f"{file_base_name}_txt2img_topk_accuracy.png"),
		"mrr": os.path.join(results_dir, f"{file_base_name}_mrr.png"),
		"cs": os.path.join(results_dir, f"{file_base_name}_cos_sim.png"),
		"retrieval_per_epoch": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_per_epoch.png"),
		"retrieval_best": os.path.join(results_dir, f"{file_base_name}_retrieval_metrics_best_model_per_k.png"),
	}

	plot_loss_accuracy_metrics(
		dataset_name=dataset_name,
		train_losses=training_losses,
		val_losses=[m.get("val_loss", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		val_acc_img2txt_list=[m.get("img2txt_acc", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		val_acc_txt2img_list=[m.get("txt2img_acc", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		img2txt_topk_accuracy_list=[m.get("img2txt_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		txt2img_topk_accuracy_list=[m.get("txt2img_topk_acc", {}) for m in in_batch_loss_acc_metrics_all_epochs],
		mean_reciprocal_rank_list=[m.get("mean_reciprocal_rank", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		cosine_similarity_list=[m.get("cosine_similarity", float('nan')) for m in in_batch_loss_acc_metrics_all_epochs],
		losses_file_path=plot_paths["losses"],
		accuracy_file_path=plot_paths["val_acc"],
		img2txt_topk_accuracy_file_path=plot_paths["img2txt_topk"],
		txt2img_topk_accuracy_file_path=plot_paths["txt2img_topk"],
		mean_reciprocal_rank_file_path=plot_paths["mrr"],
		cosine_similarity_file_path=plot_paths["cs"],
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
		nw: int,
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

		metrics_per_epoch = get_in_batch_loss_accuracy(
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
		print(
			f'@ Epoch {epoch + 1}:\n'
			f'\t[LOSS] {mode}: {avg_training_loss:.5f} | Valid: {metrics_per_epoch.get("val_loss"):.8f}\n'
			f'\tIn-batch Validation Accuracy [text retrieval per image]: {metrics_per_epoch.get("img2txt_acc")} '
			f'[image retrieval per text]: {metrics_per_epoch.get("txt2img_acc")}'
		)

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

	plot_loss_accuracy_metrics(
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
		window_size: int,
		patience: int = 10,
		min_delta: float = 1e-4,
		cumulative_delta: float = 5e-3,
		minimum_epochs: int = 20,
		TOP_K_VALUES: List[int] = [1, 5, 10, 15, 20],
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
		metrics_per_epoch = get_in_batch_loss_accuracy(
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
			final_metrics = get_in_batch_loss_accuracy(
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

	plot_loss_accuracy_metrics(
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
		results_dir: str,
		window_size: int,
		patience:int=10,
		min_delta:float=1e-4,
		cumulative_delta:float=5e-3,
		minimum_epochs:int=20,
		TOP_K_VALUES:List[int]=[1, 5, 10, 15, 20],
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
		metrics_per_epoch = get_in_batch_loss_accuracy(
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
			final_metrics = get_in_batch_loss_accuracy(
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
	plot_loss_accuracy_metrics(
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