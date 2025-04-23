from utils import *
import matplotlib.gridspec as gridspec

def plot_image_to_texts_separate_horizontal_bars(
		models: dict,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(15, 6),  # Adjusted for multiple subplots
		dpi: int = 300,  # Increased for publication quality
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	pretrained_model_arch = models.get("pretrained").name
	print(f"{len(models)} strategies for {dataset_name} {pretrained_model_arch}")
	
	# Prepare labels
	try:
		labels = validation_loader.dataset.dataset.classes
	except AttributeError:
		labels = validation_loader.dataset.unique_labels
	n_labels = len(labels)
	if topk > n_labels:
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	
	# Load and preprocess image
	try:
		img = Image.open(img_path).convert("RGB")
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content)).convert("RGB")
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	image_tensor = preprocess(img).unsqueeze(0).to(device)
	# Check if img_path is in the validation set and get ground-truth label if available
	ground_truth_label = None
	validation_dataset = validation_loader.dataset
	if hasattr(validation_dataset, 'data_frame') and 'img_path' in validation_dataset.data_frame.columns:
		matching_rows = validation_dataset.data_frame[validation_dataset.data_frame['img_path'] == img_path]
		if not matching_rows.empty:
			ground_truth_label = matching_rows['label'].iloc[0]
			print(f"Ground truth label for {img_path}: {ground_truth_label}")
	# Compute predictions for each model
	model_predictions = {}
	model_topk_labels = {}
	model_topk_probs = {}
	for model_name, model in models.items():
		model.eval()
		print(f"[Image-to-text(s)] {model_name} Zero-Shot Image Classification Query: {img_path}".center(200, " "))
		t0 = time.time()
		with torch.no_grad():
			image_features = model.encode_image(image_tensor)
			labels_features = model.encode_text(tokenized_labels_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			labels_features /= labels_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
		
		# Store full probabilities for all labels
		all_probs = similarities.squeeze().cpu().numpy()
		model_predictions[model_name] = all_probs
		
		# Get top-k labels and probabilities for this model
		topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
		topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
		topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
		topk_pred_labels = [labels[i] for i in topk_pred_indices]
		
		# Sort by descending probability
		sorted_indices = np.argsort(topk_pred_probs)[::-1]
		model_topk_labels[model_name] = [topk_pred_labels[i] for i in sorted_indices]
		model_topk_probs[model_name] = topk_pred_probs[sorted_indices]
		print(f"Top-{topk} predicted labels for {model_name}: {model_topk_labels[model_name]}")
		print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))
	# Create subplot grid: 1 row, (1 + len(models)) columns
	num_strategies = len(models)
	
	# Get image dimensions for dynamic sizing
	img_width, img_height = img.size
	aspect_ratio = img_height / img_width
	
	# Calculate figure dimensions based on image aspect ratio
	img_subplot_width = 3  # Base width for image subplot in inches
	img_subplot_height = img_subplot_width * aspect_ratio  # Maintain image aspect ratio
	
	# Adjust model subplot widths based on image height
	model_subplot_width = 3  # Base width for model subplots
	model_subplot_height = img_subplot_height  # Match height with image
	
	# Calculate total figure dimensions
	fig_width = img_subplot_width + (model_subplot_width * num_strategies)
	fig_height = max(4, model_subplot_height)  # Ensure minimum height
	
	# Create figure with calculated dimensions
	fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
	
	# Create grid with appropriate width ratios
	# First column for image, remaining columns for models
	width_ratios = [img_subplot_width/model_subplot_width] + [1] * num_strategies
	gs = gridspec.GridSpec(1, 1 + num_strategies, width_ratios=width_ratios, wspace=0.02)
	# Subplot 1: Query Image
	ax0 = plt.subplot(gs[0])
	ax0.imshow(img)
	ax0.axis('off')
	ax0.set_title("")
	title_text = f"Query Image\nGT: {ground_truth_label.capitalize()}" if ground_truth_label else "Query"
	ax0.text(
		0.5,  # x position (center)
		-0.05,  # y position (just below the image)
		title_text,
		fontsize=9,
		fontweight='bold',
		ha='center',
		va='top',
		transform=ax0.transAxes  # Use axes coordinates
	)
	# Define colors consistent with plot_comparison_metrics_split/merged
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	colors = [pretrained_colors.get(pretrained_model_arch, '#000000')] + list(strategy_colors.values())
	print(f"colors: {colors}")
	# Subplots for each model
	all_strategies = list(models.keys())
	axes = []
	# Create subplots iteratively to avoid referencing 'axes' before assignment
	for model_idx in range(num_strategies):
			if model_idx == 0:
					# First subplot shares x-axis with ax0 (image subplot)
					ax = plt.subplot(gs[model_idx + 1])
			else:
					# Subsequent subplots share x-axis with the first model subplot (axes[0])
					ax = plt.subplot(gs[model_idx + 1], sharex=axes[0])
			axes.append(ax)
	# Create a list of handles for the legend
	legend_handles = []
	for model_idx, (model_name, ax) in enumerate(zip(all_strategies, axes)):
			y_pos = np.arange(topk)#*0.8  # Multiply by factor < 1 to reduce spacing
			sorted_probs = model_topk_probs[model_name]
			sorted_labels = model_topk_labels[model_name]
			# Plot horizontal bars and create a handle for the legend
			bars = ax.barh(
				y_pos,
				sorted_probs,
				height=0.5,
				color=colors[model_idx],
				edgecolor='white',
				alpha=0.9,
				label=f"CLIP {pretrained_model_arch}" if model_name == "pretrained" else model_name.upper()
			)
			legend_handles.append(bars)
			ax.invert_yaxis()  # Highest probs on top
			# Hide y-axis ticks and labels
			ax.set_yticks([])
			ax.set_yticklabels([])  # Empty labels
			# Set specific x-axis limits and ticks
			ax.set_xlim(0, 1)
			ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
			ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=7)
			ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#888888')  # Subtler grid
			ax.tick_params(axis='x', labelsize=8)
			# Annotate bars with labels and probabilities
			for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
					formatted_label = label.replace('_', ' ').title()
					ax.text(
							prob + 0.01 if prob < 0.5 else prob - 0.01,
							i,
							f"{formatted_label}\n({prob:.2f})",
							va='center',
							ha='right' if prob > 0.5 else 'left',
							fontsize=7,
							color='black',
							fontweight='bold' if prob == max(sorted_probs) else 'normal',
					)
			for spine in ax.spines.values():
					spine.set_color('black')
	# Add a legend at the top of the figure
	fig.legend(
		legend_handles,
		[handle.get_label() for handle in legend_handles],
		fontsize=11,
		loc='upper center',
		ncol=len(legend_handles),
		bbox_to_anchor=(0.5, 0.98),
		bbox_transform=fig.transFigure,
		frameon=True,
		shadow=True,
		fancybox=True,
		edgecolor='black',
		facecolor='white',
	)
	fig.text(
		0.5,  # x position (center of figure)
		0.02,  # y position (near bottom of figure)
		"Probability",
		ha='center',  # horizontal alignment
		va='center',  # vertical alignment
		fontsize=12,
		fontweight='bold'
	)
	plt.tight_layout() 
	# Add a border around the entire figure
	for spine in fig.gca().spines.values():
			spine.set_visible(True)
			spine.set_color('black')
			spine.set_linewidth(1.0)
	# Save plot
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	file_name = os.path.join(
		results_dir,
		f'{dataset_name}_'
		f'Top{topk}_labels_'
		f'image_{img_hash}_'
		f"{'gt_' + ground_truth_label.replace(' ', '-') + '_' if ground_truth_label else ''}"
		f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
		f'separate_bar_image_to_text.png'
	)
	plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved visualization to: {file_name}")

def plot_image_to_texts_stacked_horizontal_bar(
		models: dict,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(8, 6),
		dpi: int = 250,
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	print(f"num_strategies: {len(models)}")
	try:
		labels = validation_loader.dataset.dataset.classes
	except AttributeError:
		labels = validation_loader.dataset.unique_labels
	n_labels = len(labels)
	if topk > n_labels:
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
		return
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)
	try:
		img = Image.open(img_path).convert("RGB")
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content)).convert("RGB")
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return
	image_tensor = preprocess(img).unsqueeze(0).to(device)

	# Compute predictions for each model
	model_predictions = {}
	pretrained_topk_labels = []  # To store the top-k labels from the pre-trained model
	pretrained_topk_probs = []  # To store the corresponding probabilities for sorting
	for model_name, model in models.items():
		model.eval()
		print(f"[Image-to-text(s)] {model_name} Zero-Shot Image Classification of image: {img_path}".center(200, " "))
		t0 = time.time()
		with torch.no_grad():
			image_features = model.encode_image(image_tensor)
			labels_features = model.encode_text(tokenized_labels_tensor)
			image_features /= image_features.norm(dim=-1, keepdim=True)
			labels_features /= labels_features.norm(dim=-1, keepdim=True)
			similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
		
		# Store full probabilities for all labels
		all_probs = similarities.squeeze().cpu().numpy()
		model_predictions[model_name] = all_probs
		# If this is the pre-trained model, get its top-k labels and probabilities
		if model_name == "pretrained":
			topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
			topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
			topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
			pretrained_topk_labels = [labels[i] for i in topk_pred_indices]
			pretrained_topk_probs = topk_pred_probs
			print(f"Top-{topk} predicted labels for pretrained model: {pretrained_topk_labels}")
		print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

	# Sort the pre-trained model's top-k labels by their probabilities (descending)
	sorted_indices = np.argsort(pretrained_topk_probs)[::-1]  # Descending order
	pretrained_topk_labels = [pretrained_topk_labels[i] for i in sorted_indices]

	num_labels = len(pretrained_topk_labels)
	num_strategies = len(models)
	plot_data = np.zeros((num_labels, num_strategies))  # Rows: labels, Columns: models
	all_strategies = list(models.keys())

	for model_idx, (model_name, probs) in enumerate(model_predictions.items()):
		for label_idx, label in enumerate(pretrained_topk_labels):
			# Find the index of this label in the full label list
			label_full_idx = labels.index(label)
			plot_data[label_idx, model_idx] = probs[label_full_idx]

	fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
	bar_width = 0.21
	y_pos = np.arange(num_labels)
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	pretrained_model_arch = models.get("pretrained").name
	colors = [pretrained_colors.get(pretrained_model_arch, '#000000')] + list(strategy_colors.values())
	print(f"colors: {colors}")
	winning_model_per_label = np.argmax(plot_data, axis=1)

	for model_idx, model_name in enumerate(all_strategies):
			if model_name == "pretrained":
				model_name = f"{model_name.capitalize()} {pretrained_model_arch}"
			offset = (model_idx - num_strategies / 2) * bar_width
			bars = ax.barh(
				y_pos + offset,
				plot_data[:, model_idx],
				height=bar_width,
				label=model_name.split('_')[-1].replace('finetune', '').capitalize() if '_' in model_name else model_name,
				color=colors[model_idx],
				edgecolor='white',
				alpha=0.85,
			)
			for i, bar in enumerate(bars):
				width = bar.get_width()
				if width > 0.01:
					is_winner = (model_idx == winning_model_per_label[i])
					ax.text(
						width + 0.01,
						bar.get_y() + bar.get_height() / 2,
						f"{width:.2f}",
						va='center',
						fontsize=8,
						color='black',
						fontweight='bold' if is_winner else 'normal',
						alpha=0.85,
					)
	ax.set_yticks(y_pos)
	ax.set_yticklabels([label.replace('_', ' ').title() for label in pretrained_topk_labels], fontsize=11)
	ax.set_xlim(0, 1.02)
	ax.set_xlabel("Probability", fontsize=10)
	# ax.set_title(f"Top-{topk} Predictions (Pre-trained Baseline)", fontsize=12, fontweight='bold')
	ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='black',)
	ax.tick_params(axis='x', labelsize=12)
	ax.legend(
		fontsize=9,
		loc='best',
		ncol=len(models),
		frameon=True,
		facecolor='white',
		shadow=True,
		fancybox=True,
	)
	for spine in ax.spines.values():
		spine.set_color('black')
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	file_name = os.path.join(
			results_dir,
			f'{dataset_name}_'
			f'Top{topk}_labels_'
			f'image_{img_hash}_'
			f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
			f'stacked_bar_image_to_text.png'
	)
	plt.tight_layout()
	plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved visualization to: {file_name}")

	# Save the original image separately using same hash (for visual comparison)
	fig_img, ax_img = plt.subplots(figsize=(4, 4), dpi=dpi)
	ax_img.imshow(img)
	ax_img.axis('off')
	img_file_name = os.path.join(results_dir, f'{dataset_name}_query_original_image_{img_hash}.png')
	plt.tight_layout()
	plt.savefig(img_file_name, bbox_inches='tight', dpi=dpi)
	plt.close()
	print(f"Saved original image to: {img_file_name}")

def plot_image_to_texts_pretrained(
		best_pretrained_model: torch.nn.Module,
		validation_loader: DataLoader,
		preprocess,
		img_path: str,
		topk: int,
		device: str,
		results_dir: str,
		figure_size=(13, 7),
		dpi: int = 300,
	):
	dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
	best_pretrained_model_name = best_pretrained_model.__class__.__name__
	best_pretrained_model_arch = re.sub(r'[/@]', '-', best_pretrained_model.name)
	best_pretrained_model.eval()
	print(f"[Image-to-text(s)] {best_pretrained_model_name} {best_pretrained_model_arch} Zero-Shot Image Classification of image: {img_path}".center(200, " "))
	t0 = time.time()
	try:
		labels = validation_loader.dataset.dataset.classes
	except AttributeError:
		labels = validation_loader.dataset.unique_labels
	n_labels = len(labels)
	if topk > n_labels:
		print(f"ERROR: requested Top-{topk} labeling is greater than number of labels ({n_labels}) => EXIT...")
		return
	# Tokenize the labels and move to device
	tokenized_labels_tensor = clip.tokenize(texts=labels).to(device)

	try:
		img = Image.open(img_path).convert("RGB")
	except FileNotFoundError:
		try:
			response = requests.get(img_path)
			response.raise_for_status()
			img = Image.open(BytesIO(response.content)).convert("RGB")
		except requests.exceptions.RequestException as e:
			print(f"ERROR: failed to load image from {img_path} => {e}")
			return

	# Preprocess image
	image_tensor = preprocess(img).unsqueeze(0).to(device)
	
	# Encode and compute similarity
	with torch.no_grad():
		image_features = best_pretrained_model.encode_image(image_tensor)
		labels_features = best_pretrained_model.encode_text(tokenized_labels_tensor)
		image_features /= image_features.norm(dim=-1, keepdim=True)
		labels_features /= labels_features.norm(dim=-1, keepdim=True)
		similarities = (100.0 * image_features @ labels_features.T).softmax(dim=-1)
	
	# Get top-k predictions
	topk_pred_probs, topk_pred_labels_idx = similarities.topk(topk, dim=-1)
	topk_pred_probs = topk_pred_probs.squeeze().cpu().numpy()
	topk_pred_indices = topk_pred_labels_idx.squeeze().cpu().numpy()
	topk_pred_labels = [labels[i] for i in topk_pred_indices]
	print(f"Top-{topk} predicted labels: {topk_pred_labels}")

	# Sort predictions by descending probability
	sorted_indices = topk_pred_probs.argsort()[::-1]
	sorted_probs = topk_pred_probs[sorted_indices]
	print(sorted_probs)

	sorted_labels = [topk_pred_labels[i] for i in sorted_indices]

	# Hash image path for unique filename
	img_hash = hashlib.sha256(img_path.encode()).hexdigest()[:8]
	file_name = os.path.join(
			results_dir,
			f'{dataset_name}_'
			f'Top{topk}_labels_'
			f'image_{img_hash}_'
			f"{re.sub(r'[/@]', '-', best_pretrained_model_arch)}_pretrained_"
			f'bar_image_to_text.png'
	)
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}

	# Plot
	fig = plt.figure(figsize=figure_size, dpi=dpi)
	gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.01)
	# Subplot 1: Image
	ax0 = plt.subplot(gs[0])
	ax0.imshow(img)
	ax0.axis('off')
	ax0.set_title("Query Image", fontsize=12)

	# Subplot 2: Horizontal bar plot
	ax1 = plt.subplot(gs[1])
	y_pos = range(topk)
	ax1.barh(y_pos, sorted_probs, color=pretrained_colors.get(best_pretrained_model.name, '#000000'), edgecolor='white')
	ax1.invert_yaxis()  # Highest probs on top
	ax1.set_yticks([])  # Hide y-axis ticks
	ax1.set_xlim(0, 1)
	ax1.set_xlabel("Probability", fontsize=11)
	ax1.set_title(f"Top-{topk} Predicted Labels", fontsize=10)
	ax1.grid(False)
	# ax1.grid(True, axis='x', linestyle='--', alpha=0.5, color='black')
	for spine in ax1.spines.values():
		spine.set_edgecolor('black')

	# Annotate bars on the right with labels and probs
	for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
		ax1.text(prob + 0.02, i, f"{label} ({prob:.2f})", va='center', fontsize=8, color='black', fontweight='bold', backgroundcolor='white', alpha=0.8)
	
	plt.tight_layout()
	plt.savefig(file_name, bbox_inches='tight')
	plt.close()
	print(f"Saved visualization to: {file_name}")
	print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(160, "-"))

def plot_text_to_images_merged(
		models: Dict,
		validation_loader: torch.utils.data.DataLoader,
		preprocess,
		query_text: str,
		topk: int,
		device: str,
		results_dir: str,
		cache_dir: str = None,
		embeddings_cache: Dict = None,
		dpi: int = 300,
		print_every: int = 250
):
		"""
		Plot top-K images retrieved for a given text query across multiple models in a merged grid.
		
		Args:
				models: Dictionary mapping strategy names to CLIP models
				validation_loader: DataLoader for validation images
				preprocess: Preprocessing function for images
				query_text: Text query for retrieval
				topk: Number of top images to retrieve
				device: Device for computation (e.g., 'cuda:0')
				results_dir: Directory to save visualization
				cache_dir: Directory to store cached embeddings (optional)
				embeddings_cache: Dictionary mapping strategies to (embeddings, image_paths) tuples
				dpi: Resolution for saved figure
				print_every: Interval for progress logging
		"""
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		img_hash = hashlib.sha256(query_text.encode()).hexdigest()[:8]
		if cache_dir is None:
				cache_dir = results_dir
		
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		
		pretrained_model_arch = models.get("pretrained").name
		tokenized_query = clip.tokenize([query_text]).to(device)
		
		model_results = {}
		num_strategies = len(models)
		all_strategies = list(models.keys())
		
		for strategy, model in models.items():
				print(f"Processing strategy: {strategy} {model.__class__.__name__} {pretrained_model_arch}".center(160, " "))
				model.eval()
				print(f"[Text-to-image(s) (merged)] {strategy} query: '{query_text}'".center(160, " "))
				
				# Use cached embeddings
				if embeddings_cache is not None and strategy in embeddings_cache:
						all_image_embeddings, image_paths = embeddings_cache[strategy]
						print(f"Using precomputed embeddings for {strategy}")
				else:
						print(f"No cached embeddings found for {strategy}. Computing from scratch...")
						cache_file = os.path.join(
								cache_dir, 
								f"{dataset_name}_{strategy}_{model.__class__.__name__}_{re.sub(r'[/@]', '-', pretrained_model_arch)}_embeddings.pt"
						)
						
						all_image_embeddings = None
						image_paths = []
						
						if os.path.exists(cache_file):
								print(f"Loading cached embeddings from {cache_file}")
								try:
										cached_data = torch.load(cache_file, map_location='cpu')
										all_image_embeddings = cached_data['embeddings'].to(device, dtype=torch.float32)
										image_paths = cached_data.get('image_paths', [])
										print(f"Successfully loaded {len(all_image_embeddings)} cached embeddings")
								except Exception as e:
										print(f"Error loading cached embeddings: {e}")
										all_image_embeddings = None
						
						if all_image_embeddings is None:
								print("Computing image embeddings (this may take a while)...")
								image_embeddings_list = []
								image_paths = []
								
								dataset = validation_loader.dataset
								has_img_path = hasattr(dataset, 'images') and isinstance(dataset.images, (list, tuple))
								for batch_idx, batch in enumerate(validation_loader):
										images = batch[0]
										if has_img_path:
												start_idx = batch_idx * validation_loader.batch_size
												batch_paths = []
												for i in range(len(images)):
														global_idx = start_idx + i
														if global_idx < len(dataset):
																batch_paths.append(dataset.images[global_idx])
														else:
																batch_paths.append(f"missing_path_{global_idx}")
										else:
												batch_paths = [f"batch_{batch_idx}_img_{i}" for i in range(len(images))]
										
										image_paths.extend(batch_paths)
										
										if not isinstance(images, torch.Tensor) or len(images.shape) != 4:
												print(f"Warning: Invalid image tensor in batch {batch_idx}")
												continue
										
										with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
												images = images.to(device)
												image_features = model.encode_image(images)
												image_features /= image_features.norm(dim=-1, keepdim=True)
												image_embeddings_list.append(image_features.cpu().to(torch.float32))
										
										if (batch_idx + 1) % print_every == 0:
												print(f"Processed {batch_idx + 1}/{len(validation_loader)} batches")
								
								if image_embeddings_list:
										all_image_embeddings = torch.cat(image_embeddings_list, dim=0).to(device, dtype=torch.float32)
										print(f"Computed {len(all_image_embeddings)} image embeddings for {dataset_name} | « {strategy} » strategy")
										
										try:
												torch.save({
														'embeddings': all_image_embeddings.cpu(),
														'image_paths': image_paths
												}, cache_file)
												print(f"Saved embeddings to {cache_file}")
										except Exception as e:
												print(f"Warning: Failed to save embeddings cache: {e}")
								else:
										print("Error: No valid image embeddings were collected")
										continue
				
				# Compute similarities between text query and all images
				with torch.no_grad():
						text_features = model.encode_text(tokenized_query)
						text_features = F.normalize(text_features, dim=-1).to(torch.float32)
						all_image_embeddings = all_image_embeddings.to(torch.float32)
						similarities = (100.0 * text_features @ all_image_embeddings.T).softmax(dim=-1)
						
						effective_topk = min(topk, len(all_image_embeddings))
						topk_scores, topk_indices = torch.topk(similarities.squeeze(), effective_topk)
						topk_scores = topk_scores.cpu().numpy()
						topk_indices = topk_indices.cpu().numpy()
				
				# Retrieve ground-truth labels from the dataset
				dataset = validation_loader.dataset
				try:
						if hasattr(dataset, 'label') and isinstance(dataset.label, (list, np.ndarray)):
								ground_truth_labels = dataset.label
						elif hasattr(dataset, 'labels') and isinstance(dataset.labels, (list, np.ndarray)):
								ground_truth_labels = dataset.labels
						else:
								raise AttributeError("Dataset does not have accessible 'label' or 'labels' attribute")
						topk_ground_truth_labels = [ground_truth_labels[idx] for idx in topk_indices]
				except (AttributeError, IndexError) as e:
						print(f"Warning: Could not retrieve ground-truth labels: {e}")
						topk_ground_truth_labels = [f"Unknown GT {idx}" for idx in topk_indices]
				
				# Store results for this model
				model_results[strategy] = {
						'topk_scores': topk_scores,
						'topk_indices': topk_indices,
						'image_paths': image_paths,
						'ground_truth_labels': topk_ground_truth_labels
				}
		
		# Create a figure with a larger figure size to accommodate the borders
		fig_width = effective_topk * 3.2
		fig_height = num_strategies * 4
		fig, axes = plt.subplots(
				nrows=num_strategies,
				ncols=effective_topk,
				figsize=(fig_width, fig_height),
				constrained_layout=True,
		)
		fig.suptitle(
				f"Query: '{query_text}' Top-{effective_topk} Images Across Models",
				fontsize=13,
				fontweight='bold',
		)
		
		# If there's only one model or topk=1, adjust axes to be 2D
		if num_strategies == 1:
				axes = [axes]
		if effective_topk == 1:
				axes = [[ax] for ax in axes]
		
		# Plot images for each model
		for row_idx, strategy in enumerate(all_strategies):
				# Get border color for this model
				if strategy == 'pretrained':
						model = models[strategy]
						border_color = pretrained_colors.get(model.name, '#745555')
				else:
						border_color = strategy_colors.get(strategy, '#000000')
				
				# Get top-k results for this model
				result = model_results[strategy]
				topk_scores = result['topk_scores']
				topk_indices = result['topk_indices']
				image_paths = result['image_paths']
				topk_ground_truth_labels = result['ground_truth_labels']
				dataset = validation_loader.dataset
				
				# Plot each image in the row
				for col_idx, (idx, score, gt_label) in enumerate(zip(topk_indices, topk_scores, topk_ground_truth_labels)):
						ax = axes[row_idx][col_idx]
						try:
								img_path = image_paths[idx]
								if os.path.exists(img_path):
										img = Image.open(img_path).convert('RGB')
										ax.imshow(img)
								else:
										if hasattr(dataset, '__getitem__'):
												sample = dataset[idx]
												if len(sample) >= 3:
														img = sample[0]
												else:
														raise ValueError(f"Unexpected dataset structure at index {idx}: {sample}")
												
												if isinstance(img, torch.Tensor):
														img = img.cpu().numpy()
														if img.shape[0] in [1, 3]:
																img = img.transpose(1, 2, 0)
														mean = np.array([0.5126933455467224, 0.5045100450515747, 0.48094621300697327])
														std = np.array([0.276103675365448, 0.2733437418937683, 0.27065524458885193])
														img = img * std + mean
														img = np.clip(img, 0, 1)
												ax.imshow(img)
								ax.set_title(f"Top-{col_idx+1} (Score: {score:.4f})\nGT: {gt_label}", fontsize=10)
						
						except Exception as e:
								print(f"Warning: Could not display image {idx} for model {strategy}: {e}")
								ax.imshow(np.ones((224, 224, 3)) * 0.5)
								ax.set_title(f"Top-{col_idx+1} (Score: {score:.4f})\nGT: Unknown", fontsize=10)
						
						# Remove default spines
						for spine in ax.spines.values():
								spine.set_visible(False)
						
						ax.axis('off')
				
				# Add model name label on the left side of the row
				axes[row_idx][0].text(
						-0.15,
						0.5,
						strategy.upper() if strategy != 'pretrained' else f"{strategy.capitalize()} {pretrained_model_arch}",
						transform=axes[row_idx][0].transAxes,
						fontsize=14,
						fontweight='bold',
						va='center',
						ha='right',
						rotation=90,
						color=border_color,
				)
		
		# Save the visualization
		file_name = os.path.join(
				results_dir,
				f"{dataset_name}_"
				f"Top{effective_topk}_images_"
				f"{img_hash}_"
				f"Q_{re.sub(' ', '_', query_text)}_"
				f"{'_'.join(all_strategies)}_"
				f"{re.sub(r'[/@]', '-', pretrained_model_arch)}_"
				f"t2i_merged.png"
		)
		
		plt.tight_layout()
		plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
		plt.close()
		print(f"Saved visualization to: {file_name}")

def plot_text_to_images(
				models, 
				validation_loader, 
				preprocess, 
				query_text, 
				topk, 
				device, 
				results_dir, 
				cache_dir=None, 
				embeddings_cache=None, 
				dpi=300,
):
		plt_start_time = time.time()
		dataset_name = getattr(validation_loader, 'name', 'unknown_dataset')
		img_hash = hashlib.sha256(query_text.encode()).hexdigest()[:8]
		if cache_dir is None:
				cache_dir = results_dir
		tokenized_query = clip.tokenize([query_text]).to(device)
		
		for strategy, model in models.items():
				print(f"Processing strategy: {strategy} ".center(160, " "))
				if strategy == 'pretrained':
						model_arch = re.sub(r'[/@]', '-', model.name)
						print(f"{model.__class__.__name__} {model_arch}".center(160, " "))
				model.eval()
				print(f"[Text-to-image(s)] strategy: {strategy} Query: '{query_text}'".center(160, " "))
				
				# Get top-k images
				all_image_embeddings, image_paths = embeddings_cache[strategy]
				all_image_embeddings = all_image_embeddings.to(device, dtype=torch.float32)
				
				with torch.no_grad():
						text_features = model.encode_text(tokenized_query).to(torch.float32)
						text_features = F.normalize(text_features, dim=-1)
						similarities = (100.0 * text_features @ all_image_embeddings.T).softmax(dim=-1)
						effective_topk = min(topk, len(all_image_embeddings))
						topk_scores, topk_indices = torch.topk(similarities.squeeze(), effective_topk)
						topk_scores = topk_scores.cpu().numpy()
						topk_indices = topk_indices.cpu().numpy()
				
				# Get ground truth labels
				dataset = validation_loader.dataset
				try:
						ground_truth_labels = dataset.labels
						topk_ground_truth_labels = [ground_truth_labels[idx] for idx in topk_indices]
				except (AttributeError, IndexError) as e:
						print(f"Warning: Could not retrieve ground-truth labels: {e}")
						topk_ground_truth_labels = [f"Unknown GT {idx}" for idx in topk_indices]
				
				# Load all images first
				topk_images = []
				for idx in topk_indices:
						try:
								img_path = image_paths[idx]
								if os.path.exists(img_path):
										img = Image.open(img_path).convert('RGB')
								else:
										sample = dataset[idx]
										if len(sample) >= 3:
												img = sample[0]
										else:
												raise ValueError(f"Unexpected dataset structure at index {idx}")
										
										if isinstance(img, torch.Tensor):
												img = img.cpu().numpy()
												if img.shape[0] in [1, 3]:
														img = img.transpose(1, 2, 0)
												mean = np.array([0.5754663102194626, 0.564594860510725, 0.5443646108296668])
												std = np.array([0.2736517370426002, 0.26753170455186887, 0.2619102890668636])
												img = img * std + mean
												img = np.clip(img, 0, 1)
												img = (img * 255).astype(np.uint8)
												img = Image.fromarray(img)
								topk_images.append(img)
						except Exception as e:
								print(f"Warning: Could not load image {idx}: {e}")
								blank_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
								topk_images.append(Image.fromarray(blank_img))
				
				# Title height in pixels
				title_height = 100  # Increased to accommodate longer GT text
				
				# First determine dimensions
				heights = [img.height for img in topk_images]
				widths = [img.width for img in topk_images]
				
				# Use the same aspect ratio for all images
				max_height = max(heights)
				# Resize images to have same height
				for i in range(len(topk_images)):
						if topk_images[i].height != max_height:
								w = int(topk_images[i].width * (max_height / topk_images[i].height))
								topk_images[i] = topk_images[i].resize((w, max_height), Image.LANCZOS)
				
				# Update widths after resizing
				widths = [img.width for img in topk_images]
				
				# Create a composite image
				total_width = sum(widths)
				composite = Image.new('RGB', (total_width, max_height + title_height), color='white')
				
				# Add each image
				x_offset = 0
				for i, img in enumerate(topk_images):
						composite.paste(img, (x_offset, title_height))  # Leave space at top for text
						x_offset += img.width
				
				try:
						# Try to load fonts with different sizes for different text elements
						title_font = ImageFont.truetype("arial.ttf", 20)  # For "Top-N"
						score_font = ImageFont.truetype("arial.ttf", 18)  # For "Score: X.XXX"
						gt_font = ImageFont.truetype("arial.ttf", 16)     # For "GT: Label text"
				except IOError:
						try:
								# Try another common font
								title_font = ImageFont.truetype("DejaVuSans.ttf", 20)
								score_font = ImageFont.truetype("DejaVuSans.ttf", 18)
								gt_font = ImageFont.truetype("DejaVuSans.ttf", 16)
						except IOError:
								# Fall back to default font if no fonts available
								title_font = score_font = gt_font = ImageFont.load_default()
				
				draw = ImageDraw.Draw(composite)
				
				# Add a subtle dividing line between title area and image
				draw.line([(0, title_height-2), (total_width, title_height-2)], fill="#DDDDDD", width=1)
				
				# Add text for each image with proper vertical alignment
				x_offset = 0
				for i, (score, gt_label, img) in enumerate(zip(topk_scores, topk_ground_truth_labels, topk_images)):
						# Prepare the title text
						title_text = f"Top-{i+1}"
						score_text = f"Score: {score:.3f}"
						gt_text = f"GT: {gt_label.capitalize()}"
						
						# Calculate center position for this image section
						center_x = x_offset + img.width // 2
						
						# Top position for text
						top_y = 10  # Top margin
						
						# Get text dimensions using appropriate method for the PIL version
						if hasattr(title_font, 'getbbox'):
								title_bbox = title_font.getbbox(title_text)
								title_width = title_bbox[2] - title_bbox[0]
								title_height_px = title_bbox[3] - title_bbox[1]
								
								score_bbox = score_font.getbbox(score_text)
								score_width = score_bbox[2] - score_bbox[0]
								score_height_px = score_bbox[3] - score_bbox[1]
								
								# For GT text, we'll handle long text differently
								gt_bbox = gt_font.getbbox(gt_text)
								gt_width = gt_bbox[2] - gt_bbox[0]
								gt_height_px = gt_bbox[3] - gt_bbox[1]
						else:
								title_width, title_height_px = title_font.getsize(title_text)
								score_width, score_height_px = score_font.getsize(score_text)
								gt_width, gt_height_px = gt_font.getsize(gt_text)
						
						# Draw "Top-N" text centered
						draw.text(
								(center_x - title_width//2, top_y),
								title_text,
								fill="black",
								font=title_font
						)
						
						# Draw "Score: X.X" text centered
						score_y = top_y + title_height_px + 8  # Add spacing between lines
						draw.text(
								(center_x - score_width//2, score_y),
								score_text,
								fill="black",
								font=score_font
						)
						
						# For GT text, we need to handle longer text with possible line wrapping
						gt_y = score_y + score_height_px + 8  # Add more spacing before GT line
						
						# Check if GT text is too wide for the image
						max_gt_width = img.width - 20  # Leave 10px margin on each side
						
						if gt_width > max_gt_width:
								# If text is too wide, we need to wrap or truncate
								# For simplicity, we'll just make the font smaller
								gt_font_size = 14  # Reduced size for long text
								try:
										smaller_gt_font = ImageFont.truetype("arial.ttf", gt_font_size)
								except IOError:
										try:
												smaller_gt_font = ImageFont.truetype("DejaVuSans.ttf", gt_font_size)
										except IOError:
												smaller_gt_font = gt_font  # Fall back to original if no font available
								
								# Get new dimensions with smaller font
								if hasattr(smaller_gt_font, 'getbbox'):
										gt_bbox = smaller_gt_font.getbbox(gt_text)
										gt_width = gt_bbox[2] - gt_bbox[0]
								else:
										gt_width, _ = smaller_gt_font.getsize(gt_text)
								
								# If still too wide, truncate and add ellipsis
								if gt_width > max_gt_width:
										truncated = False
										while gt_width > max_gt_width and len(gt_text) > 10:  # Keep at least "GT: X..."
												gt_text = gt_text[:-1]  # Remove last character
												truncated = True
												if hasattr(smaller_gt_font, 'getbbox'):
														gt_bbox = smaller_gt_font.getbbox(gt_text + "...")
														gt_width = gt_bbox[2] - gt_bbox[0]
												else:
														gt_width, _ = smaller_gt_font.getsize(gt_text + "...")
										
										if truncated:
												gt_text += "..."  # Add ellipsis to indicate truncation
								
								# Draw GT text centered with smaller font
								draw.text(
										(center_x - gt_width//2, gt_y),
										gt_text,
										fill="black",
										font=smaller_gt_font
								)
						else:
								# If GT text fits, draw it normally centered
								draw.text(
										(center_x - gt_width//2, gt_y),
										gt_text,
										fill="black",
										font=gt_font
								)
						
						x_offset += img.width
				
				# Save the composite image
				file_name = os.path.join(
						results_dir,
						f'{dataset_name}_'
						f'Top{effective_topk}_'
						f'images_{img_hash}_'
						f'Q_{re.sub(" ", "_", query_text)}_'
						f'{strategy}_'
						f'{model_arch}_'
						f't2i_seamless.png'
				)
				
				composite.save(file_name, dpi=(dpi, dpi))
				print(f"Saved seamless visualization to: {file_name}")
		
		print(f"Total Elapsed_t: {time.time()-plt_start_time:.3f} sec".center(160, "-"))

def plot_comparison_metrics_split(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,
		finetune_strategies: list,
		results_dir: str,
		topK_values: list,
		figure_size=(6, 5.5),
		DPI: int = 250,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = [
		'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
		'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
	]
	if model_name not in finetuned_img2txt_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_img2txt_dict. Skipping...")
		print(json.dumps(finetuned_img2txt_dict, indent=4, ensure_ascii=False))
		return
	if model_name not in finetuned_txt2img_dict.keys():
		print(f"WARNING: {model_name} not found in finetuned_txt2img_dict. Skipping...")
		print(json.dumps(finetuned_txt2img_dict, indent=4, ensure_ascii=False))
		return
	
	# Validate finetune_strategies
	finetune_strategies = [s for s in finetune_strategies if s in ["full", "lora", "progressive"]][:3]  # Max 3
	if not finetune_strategies:
		print("WARNING: No valid finetune strategies provided. Skipping...")
		return
	model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
	# Define a professional color palette for fine-tuned strategies
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers
	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		for metric in metrics:
			fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
			fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
			file_path = os.path.join(results_dir, fname)
			if metric not in pretrained_dict.get(model_name, {}):
				print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
				continue
			# Get available k values across all dictionaries
			k_values = sorted(
				k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
			)
			for strategy in finetune_strategies:
				if strategy not in finetuned_dict.get(model_name, {}) or metric not in finetuned_dict.get(model_name, {}).get(strategy, {}):
					print(f"WARNING: Metric {metric} not found in finetuned_{mode.lower().replace('-', '_')}_dict for {model_name}/{strategy}")
					k_values = []  # Reset if any strategy is missing
					break
				k_values = sorted(
					set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
				)
			if not k_values:
				print(f"WARNING: No matching K values found for {metric}")
				continue
			# Plot Pre-trained (dashed line)
			pretrained_vals = [pretrained_dict[model_name][metric].get(str(k), float('nan')) for k in k_values]
			ax.plot(
				k_values,
				pretrained_vals,
				label=f"CLIP {model_name}",
				color=pretrained_colors[model_name],
				linestyle='--', 
				marker='o',
				linewidth=1.5,
				markersize=4,
				alpha=0.75,
			)
			# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
			for strategy in finetune_strategies:
				finetuned_vals = [finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for k in k_values]
				ax.plot(
					k_values,
					finetuned_vals,
					label=f"{strategy.upper()}",
					color=strategy_colors[strategy], 
					linestyle='-', 
					marker=strategy_styles[strategy],
					linewidth=2.0, 
					markersize=5,
				)
			# Find the best and worst performing strategies at key K values
			key_k_values = [1, 10, 20]
			for k in key_k_values:
				if k in k_values:
					k_idx = k_values.index(k)
					pre_val = pretrained_vals[k_idx]
					finetuned_vals_at_k = {strategy: finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for strategy in finetune_strategies}
					# Find best and worst strategies
					best_strategy = max(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
					worst_strategy = min(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
					best_val = finetuned_vals_at_k[best_strategy]
					worst_val = finetuned_vals_at_k[worst_strategy]
					# Annotate best strategy (green)
					if pre_val != 0:
						best_imp = (best_val - pre_val) / pre_val * 100
						text_color = '#016e2bff' if best_imp >= 0 else 'red'
						arrow_style = '<|-' if best_imp >= 0 else '-|>'
						
						# Place annotations with arrows
						ax.annotate(
							f"{best_imp:+.1f}%",
							xy=(k, best_val),
							xytext=(0, 30),
							textcoords='offset points',
							fontsize=8,
							fontweight='bold',
							color=text_color,
							bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
							arrowprops=dict(
								arrowstyle=arrow_style,
								color=text_color,
								shrinkA=0,
								shrinkB=3,
								alpha=0.8,
								# connectionstyle="arc3,rad=.2"
							)
						)
					# Annotate worst strategy (red)
					if pre_val != 0:
						worst_imp = (worst_val - pre_val) / pre_val * 100
						text_color = '#016e2bff' if worst_imp >= 0 else 'red'
						arrow_style = '-|>' if worst_imp >= 0 else '<|-'
						
						# Place annotations with arrows
						ax.annotate(
							f"{worst_imp:+.1f}%",
							xy=(k, worst_val),
							xytext=(0, -30),
							textcoords='offset points',
							fontsize=8,
							fontweight='bold',
							color=text_color,
							bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
							arrowprops=dict(
								arrowstyle=arrow_style,
								color=text_color,
								shrinkA=0,
								shrinkB=3,
								alpha=0.8,
								# connectionstyle="arc3,rad=.2"
							)
						)
			ax.set_title(
				f"{metric}@K", 
				fontsize=10, 
				fontweight='bold', 
				# pad=25, # Increase the padding to move the title further away from the plot
				y=1.12, # Adjust the y-position of the title
			)
			ax.set_xlabel("K", fontsize=10, fontweight='bold')
			ax.set_xticks(k_values)
			ax.grid(True, linestyle='--', alpha=0.75)
			ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
			ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
			ax.set_ylim(-0.01, 1.01)
			ax.tick_params(axis='both', labelsize=7)
			ax.legend(
				loc='best',
				frameon=False,
				fontsize=8,
				facecolor='white',
				ncol=len(finetune_strategies)+1,
			)
			# Set spine edge color to solid black
			for spine in ax.spines.values():
				spine.set_color('black')
				spine.set_linewidth(0.7)
			plt.tight_layout()
			plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
			plt.close(fig)

def plot_comparison_metrics_merged(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,
		finetune_strategies: list,
		results_dir: str,
		topK_values: list,
		figure_size=(15, 6),
		DPI: int = 300,
	):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		# Validate model_name and finetune_strategies
		finetune_strategies = [s for s in finetune_strategies if s in ["full", "lora", "progressive"]][:3]  # Max 3
		if not finetune_strategies:
				print("WARNING: No valid finetune strategies provided. Skipping...")
				return
		for mode in modes:
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				if model_name not in finetuned_dict or not all(strategy in finetuned_dict.get(model_name, {}) for strategy in finetune_strategies):
						print(f"WARNING: Some strategies for {model_name} not found in finetuned_{mode.lower().replace('-', '_')}_dict. Skipping...")
						return
		model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Green
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers

		for i, mode in enumerate(modes):
				fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
				fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_retrieval_performance_comparison_{mode.replace('-', '_')}_merged.png"
				file_path = os.path.join(results_dir, fname)
				fig.suptitle(
						f'$\\it{{{mode}}}$ Retrieval Performance Comparison\n'
						f'Pre-trained CLIP {model_name} vs. {", ".join(s.capitalize() for s in finetune_strategies)} Fine-tuning',
						fontsize=12, fontweight='bold',
				)
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				for j, metric in enumerate(metrics):
						ax = axes[j]
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						for strategy in finetune_strategies:
								k_values = sorted(
										set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
								)
						if not k_values:
								print(f"WARNING: No matching K values for {mode}, {metric}")
								continue
						# Plot Pre-trained (dashed line)
						pretrained_values = [pretrained_dict.get(model_name, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
						ax.plot(
								k_values, pretrained_values,
								label=f"Pre-trained CLIP {model_name}",
								color=pretrained_colors[model_name], marker='o', linestyle='--',
								linewidth=1.5, markersize=5, alpha=0.7,
						)
						# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
						for strategy in finetune_strategies:
								finetuned_values = [finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).get(str(k), float('nan')) for k in k_values]
								ax.plot(
										k_values, finetuned_values,
										label=f"{strategy.capitalize()} Fine-tune",
										color=strategy_colors[strategy], marker=strategy_styles[strategy], linestyle='-',
										linewidth=2.5, markersize=6,
								)
						# Find the best and worst performing strategies at key K values
						key_k_values = [1, 10, 20]
						for k in key_k_values:
								if k in k_values:
										k_idx = k_values.index(k)
										pre_val = pretrained_values[k_idx]
										finetuned_vals_at_k = {strategy: finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).get(str(k), float('nan')) for strategy in finetune_strategies}
										# Find best and worst strategies
										best_strategy = max(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
										worst_strategy = min(finetuned_vals_at_k, key=finetuned_vals_at_k.get)
										best_val = finetuned_vals_at_k[best_strategy]
										worst_val = finetuned_vals_at_k[worst_strategy]
										# Annotate best strategy (green)
										if pre_val != 0:
												best_imp = (best_val - pre_val) / pre_val * 100
												# Set color based on improvement value
												text_color = '#016e2bff' if best_imp >= 0 else 'red'
												arrow_style = '<|-' if best_imp >= 0 else '-|>'
												
												# Place annotations with arrows
												ax.annotate(
													f"{best_imp:+.1f}%",
													xy=(k, best_val),
													xytext=(0, 30),
													textcoords='offset points',
													fontsize=8,
													fontweight='bold',
													color=text_color,
													bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
													arrowprops=dict(
														arrowstyle=arrow_style,
														color=text_color,
														shrinkA=0,
														shrinkB=3,
														alpha=0.8,
														connectionstyle="arc3,rad=.2"
													)
												)
										# Annotate worst strategy (red)
										if pre_val != 0:
												worst_imp = (worst_val - pre_val) / pre_val * 100
												# Set color based on improvement value
												text_color = 'red' if worst_imp <= 0 else '#016e2bff'
												arrow_style = '-|>' if worst_imp >= 0 else '<|-'
												
												# Place annotations with arrows
												ax.annotate(
													f"{worst_imp:+.1f}%",
													xy=(k, worst_val),
													xytext=(0, -30),
													textcoords='offset points',
													fontsize=8,
													fontweight='bold',
													color=text_color,
													bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3),
													arrowprops=dict(
														arrowstyle=arrow_style,
														color=text_color,
														shrinkA=0,
														shrinkB=3,
														alpha=0.8,
														connectionstyle="arc3,rad=.2"
													)
												)
						# Axes formatting
						ax.set_xlabel('K', fontsize=11)
						ax.set_title(f'{metric}@K', fontsize=10, fontweight='bold')
						ax.grid(True, linestyle='--', alpha=0.9)
						ax.set_xticks(k_values)
						ax.set_ylim(-0.01, 1.01)
						if j == 0:
								ax.legend(fontsize=9, loc='best')
						# Set spine edge color to solid black
						for spine in ax.spines.values():
								spine.set_color('black')
								spine.set_linewidth(1.0)
				plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
				plt.close(fig)

		# Overall summary
		print(f"\n{'='*80}")
		print(f"OVERALL PERFORMANCE SUMMARY [QUANTITATIVE ANALYSIS]")
		print(f"{'='*80}")
		for mode in modes:
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				print(f"\nMode: {mode}")
				for metric in metrics:
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						for strategy in finetune_strategies:
								k_values = sorted(
										set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys())
								)
						if k_values:
								for strategy in finetune_strategies:
										improvements = [
												((finetuned_dict[model_name][strategy][metric][str(k)] - pretrained_dict[model_name][metric][str(k)]) /
												 pretrained_dict[model_name][metric][str(k)] * 100)
												for k in k_values if pretrained_dict[model_name][metric][str(k)] != 0
										]
										if improvements:
												avg_improvement = sum(improvements) / len(improvements)
												print(f"  {metric} ({strategy.capitalize()}): Average improvement across all K values: {avg_improvement:+.2f}%")
		print(f"\n{'='*80}")

def plot_all_pretrain_metrics(
		dataset_name: str,
		img2txt_metrics_dict: dict,
		txt2img_metrics_dict: dict,
		topK_values: list,
		results_dir: str,
		figure_size=(12, 5),
		DPI: int=300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	models = list(img2txt_metrics_dict.keys()) # ['RN50', 'RN101', ..., 'ViT-L/14@336px']
	
	# Use distinct colors, markers, and linestyles for each model
	colors = plt.cm.Set1.colors
	markers = ['D', 'v', 'o', 's', '^', 'p', 'h', '*', 'H'] # 9 distinct markers
	linestyles = [':', '-.', '-', '--', '-', '--', ':', '-.', '-'] # Cycle through styles
	
	# Create separate plots for each mode (Image-to-Text and Text-to-Image)
	for i, mode in enumerate(modes):
		# Determine which metrics dictionary to use based on the mode
		metrics_dict = img2txt_metrics_dict if mode == "Image-to-Text" else txt2img_metrics_dict
		
		# Create a filename for each plot
		file_name = f"{dataset_name}_{len(models)}_pretrained_clip_models_{mode.replace('-', '_').lower()}_{'_'.join(re.sub(r'[/@]', '-', m) for m in models)}.png"
		
		# Create a figure with 1x3 subplots (one for each metric)
		fig, axes = plt.subplots(1, len(metrics), figsize=figure_size, constrained_layout=True)
		# fig.suptitle(f"{dataset_name} Pre-trained CLIP - {mode} Retrieval Metrics", fontsize=11, fontweight='bold')
		fig.suptitle(f"Pre-trained CLIP {mode} Retrieval Metrics", fontsize=11, fontweight='bold')
		
		# Create a plot for each metric
		for j, metric in enumerate(metrics):
			ax = axes[j]
			legend_handles = []
			legend_labels = []
			
			# Plot data for each model
			for k, (model_name, color, marker, linestyle) in enumerate(zip(models, colors, markers, linestyles)):
				if model_name in metrics_dict:
					k_values = sorted([int(k) for k in metrics_dict[model_name][metric].keys() if int(k) in topK_values])
					values = [metrics_dict[model_name][metric][str(k)] for k in k_values]
					
					line, = ax.plot(
						k_values,
						values,
						label=model_name,
						color=color,
						marker=marker,
						linestyle=linestyle,
						linewidth=1.8,
						markersize=5,
					)
					
					legend_handles.append(line)
					legend_labels.append(model_name)
				
				# Configure the axes and labels
				ax.set_xlabel('K', fontsize=10)
				ax.set_ylabel(f'{metric}@K', fontsize=10)
				ax.set_title(f'{metric}@K', fontsize=12, fontweight="bold")
				ax.grid(True, linestyle='--', alpha=0.9)
				ax.set_xticks(topK_values)
				ax.set_xlim(min(topK_values) - 1, max(topK_values) + 1)
				
				# Set dynamic y-axis limits
				all_values = [v for m in models if m in metrics_dict for v in [metrics_dict[m][metric][str(k)] for k in k_values]]
				if all_values:
					min_val = min(all_values)
					max_val = max(all_values)
					padding = 0.05 * (max_val - min_val) if (max_val - min_val) > 0 else 0.05
					ax.set_ylim(bottom=0, top=max(1.01, max_val + padding))
			
		# Add legend at the bottom of the figure
		fig.legend(
			legend_handles,
			legend_labels,
			title="Image Encoder",
			title_fontsize=10,
			fontsize=9,
			loc='lower center',
			ncol=min(len(models), 5),  # Limit to 5 columns for readability
			bbox_to_anchor=(0.5, 0.01),
			bbox_transform=fig.transFigure,
			frameon=True,
			shadow=True,
			fancybox=True,
			edgecolor='black',
			facecolor='white',
		)
		
		# Adjust layout and save figure
		plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Make room for the legend at bottom
		plt.savefig(os.path.join(results_dir, file_name), dpi=DPI, bbox_inches='tight')
		plt.close(fig)

def visualize_samples(dataloader, dataset, num_samples=5):
		for bidx, (images, tokenized_labels, labels_indices) in enumerate(dataloader):
				print(f"Batch {bidx}, Shapes: {images.shape}, {tokenized_labels.shape}, {labels_indices.shape}")
				if bidx >= num_samples:
						break
				
				# Get the global index of the first sample in this batch
				start_idx = bidx * dataloader.batch_size
				for i in range(min(dataloader.batch_size, len(images))):
						global_idx = start_idx + i
						if global_idx >= len(dataset):
								break
						image = images[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array
						caption_idx = labels_indices[i].item()
						path = dataset.images[global_idx]
						label = dataset.labels[global_idx]
						label_int = dataset.labels_int[global_idx]
						
						print(f"Global Index: {global_idx}")
						print(f"Image {image.shape} Path: {path}")
						print(f"Label: {label}, Label Int: {label_int}, Caption Index: {caption_idx}")
						
						# Denormalize the image (adjust mean/std based on your dataset)
						mean = np.array([0.5126933455467224, 0.5045100450515747, 0.48094621300697327])
						std = np.array([0.276103675365448, 0.2733437418937683, 0.27065524458885193])
						image = image * std + mean  # Reverse normalization
						image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1]
						
						plt.figure(figsize=(10, 10))
						plt.imshow(image)
						plt.title(f"Label: {label} (Index: {caption_idx})")
						plt.axis('off')
						plt.show()

def visualize_(dataloader, batches=3, num_samples=5):
	"""
	Visualize the first 'num_samples' images of each batch in a single figure.
	Args:
			dataloader (torch.utils.data.DataLoader): Data loader containing images and captions.
			num_samples (int, optional): Number of batches to visualize. Defaults to 5.
			num_cols (int, optional): Number of columns in the visualization. Defaults to 5.
	"""
	# Get the number of batches in the dataloader
	num_batches = len(dataloader)
	# Limit the number of batches to visualize
	num_batches = min(num_batches, batches)
	# Create a figure with 'num_samples' rows and 'num_cols' columns
	fig, axes = plt.subplots(nrows=num_batches, ncols=num_samples, figsize=(20, num_batches * 2))
	# Iterate over the batches
	for bidx, (images, tokenized_labels, labels_indices) in enumerate(dataloader):
		if bidx >= num_batches:
			break
		# Iterate over the first 'num_cols' images in the batch
		for cidx in range(num_samples):
			image = images[cidx].permute(1, 2, 0).numpy()  # Convert tensor to numpy array and permute dimensions
			caption_idx = labels_indices[cidx]
			# Denormalize the image
			image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
			image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
			# Plot the image
			axes[bidx, cidx].imshow(image)
			axes[bidx, cidx].set_title(f"Batch {bidx+1}, Img {cidx+1}: {caption_idx}", fontsize=8)
			axes[bidx, cidx].axis('off')
	# Layout so plots do not overlap
	plt.tight_layout()
	plt.show()

def plot_retrieval_metrics_best_model(
		dataset_name: str,
		image_to_text_metrics: Dict[str, Dict[str, float]],
		text_to_image_metrics: Dict[str, Dict[str, float]],
		fname: str ="Retrieval_Performance_Metrics_best_model.png",
		best_model_name: str ="Best Model",
		figure_size=(11, 4),
		DPI: int=300,
	):
	metrics = list(image_to_text_metrics.keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"{dataset_name} Retrieval Performance Metrics [{best_model_name}]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | " 
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	modes = ['Image-to-Text', 'Text-to-Image']
	
	fig, axes = plt.subplots(1, len(metrics), figsize=figure_size, constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=8, fontweight='bold')
	
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []

	for i, metric in enumerate(metrics):
		ax = axes[i] if len(metrics) > 1 else axes # Handle single subplot case
		# print(f"Image-to-Text:")
		it_top_ks = list(map(int, image_to_text_metrics[metric].keys()))  # K values for Image-to-Text
		it_vals = list(image_to_text_metrics[metric].values())
		# print(metric, it_top_ks, it_vals)
		line, = ax.plot(
			it_top_ks, 
			it_vals, 
			marker='o',
			label=modes[0], 
			color='blue',
			linestyle='-',
			linewidth=1.0,
			markersize=2.0,
		)
		if modes[0] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[0])
		
		# Plotting for Text-to-Image
		# print(f"Text-to-Image:")
		ti_top_ks = list(map(int, text_to_image_metrics[metric].keys()))  # K values for Text-to-Image
		ti_vals = list(text_to_image_metrics[metric].values())
		# print(metric, ti_top_ks, ti_vals)
		line, = ax.plot(
			ti_top_ks,
			ti_vals,
			marker='s',
			label=modes[1],
			color='red',
			linestyle='-',
			linewidth=1.0,
			markersize=2.0,
		)
		if modes[1] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[1])
		
		ax.set_xlabel('K', fontsize=8)
		ax.set_ylabel(f'{metric}@K', fontsize=8)
		ax.set_title(f'{metric}@K', fontsize=9, fontweight="bold")
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Set the x-axis to only show integer values
		all_ks = sorted(set(it_top_ks + ti_top_ks))
		ax.set_xticks(all_ks)

		# Adjust y-axis to start from 0 for better visualization
		# ax.set_ylim(bottom=-0.05, top=1.05)
		all_values = it_vals + ti_vals
		min_val = min(all_values)
		max_val = max(all_values)
		padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
		ax.set_ylim(bottom=min(-0.02, min_val - padding), top=max(0.5, max_val + padding))

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=9,
		loc='upper center',
		ncol=len(modes),
		bbox_to_anchor=(0.5, 0.94),
		bbox_transform=fig.transFigure,
		frameon=True,
		shadow=True,
		fancybox=True,
		edgecolor='black',
		facecolor='white',
	)
	plt.savefig(fname, dpi=DPI, bbox_inches='tight')
	plt.close(fig)

def plot_retrieval_metrics_per_epoch(
		dataset_name: str,
		image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
		text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
		fname: str = "Retrieval_Performance_Metrics.png",
	):
	num_epochs = len(image_to_text_metrics_list)
	num_xticks = min(10, num_epochs)
	epochs = range(1, num_epochs + 1)
	selective_xticks_epochs = np.linspace(0, num_epochs, num_xticks, dtype=int)
	if num_epochs < 2:
		return
	# Derive valid K values from the metrics for each mode
	if image_to_text_metrics_list and text_to_image_metrics_list:
		# Get K values from the first epoch's metrics for each mode
		it_first_metrics = image_to_text_metrics_list[0]["mP"]  # Use "mP" as a representative metric
		ti_first_metrics = text_to_image_metrics_list[0]["mP"]  # Use "mP" as a representative metric
		it_valid_k_values = sorted([int(k) for k in it_first_metrics.keys()])  # K values for Image-to-Text
		ti_valid_k_values = sorted([int(k) for k in ti_first_metrics.keys()])  # K values for Text-to-Image
		# Print warning if K values differ significantly (optional, for debugging)
		if set(it_valid_k_values) != set(ti_valid_k_values):
			print(f"<!> Warning: K values differ between Image-to-Text ({it_valid_k_values}) and Text-to-Image ({ti_valid_k_values}).")

	modes = ["Image-to-Text", "Text-to-Image"]
	metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
	
	suptitle_text = f"{dataset_name} Retrieval Performance Metrics [per epoch]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | "
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	
	markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'H', 'x']  # Larger, distinct markers for each line
	line_styles = ['-', '--', ':', '-.', '-']  # Varied line styles for clarity
	# colors = plt.cm.tab10.colors  # Use a color map for distinct colors
	colors = plt.cm.Set1.colors
	fig, axs = plt.subplots(len(modes), len(metrics), figsize=(20, 11), constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=15, fontweight='bold')
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []
	for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
		for j, metric in enumerate(metrics):
			ax = axs[i, j]
			# Use the appropriate K values for each mode
			valid_k_values = it_valid_k_values if i == 0 else ti_valid_k_values
			all_values = []
			for k_idx, (K, color, marker, linestyle) in enumerate(zip(valid_k_values, colors, markers, line_styles)):
				values = []
				for metrics_dict in task_metrics_list:
					if metric in metrics_dict and str(K) in metrics_dict[metric]:
						values.append(metrics_dict[metric][str(K)])
					else:
						values.append(0)  # Default to 0 if K value is missing (shouldn’t happen with valid data)
				all_values.extend(values)
				line, = ax.plot(
					epochs,
					values,
					label=f'K={K}',
					color=color,
					alpha=0.9,
					linewidth=1.8,
				)
				if f'K={K}' not in legend_labels:
					legend_handles.append(line)
					legend_labels.append(f'K={K}')

			ax.set_xlabel('Epoch', fontsize=12)
			ax.set_ylabel(f'{metric}@K', fontsize=12)
			ax.set_title(f'{modes[i]} - {metric}@K', fontsize=14)
			ax.grid(True, linestyle='--', alpha=0.7)
			# ax.set_xticks(epochs)
			ax.set_xticks(selective_xticks_epochs) # Only show selected epochs
			ax.set_xlim(0, num_epochs + 1)
			# ax.set_ylim(bottom=-0.05, top=1.05)
			# Dynamic y-axis limits
			if all_values:
				min_val = min(all_values)
				max_val = max(all_values)
				padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
				# ax.set_ylim(bottom=min(-0.05, min_val - padding), top=max(1.05, max_val + padding))
				ax.set_ylim(bottom=-0.01, top=min(1.05, max_val + padding))
			else:
				ax.set_ylim(bottom=-0.01, top=1.05)
	
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=11,
		loc='upper center',
		ncol=len(legend_labels),  # Adjust number of columns based on number of K values
		bbox_to_anchor=(0.5, 0.96),
		bbox_transform=fig.transFigure,
		frameon=True,
		shadow=True,
		fancybox=True,
		edgecolor='black',
		facecolor='white',
	)
	plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.close(fig)

def plot_loss_accuracy_metrics(
		dataset_name: str,
		train_losses: List[float],
		val_losses: List[float],
		in_batch_topk_val_accuracy_i2t_list: List[float],
		in_batch_topk_val_accuracy_t2i_list: List[float],
		full_topk_val_accuracy_i2t_list: List[float] = None,
		full_topk_val_accuracy_t2i_list: List[float] = None,
		mean_reciprocal_rank_list: List[float] = None,
		cosine_similarity_list: List[float] = None,
		losses_file_path: str = "losses.png",
		in_batch_topk_val_acc_i2t_fpth: str = "in_batch_val_topk_accuracy_i2t.png",
		in_batch_topk_val_acc_t2i_fpth: str = "in_batch_val_topk_accuracy_t2i.png",
		full_topk_val_acc_i2t_fpth: str = "full_val_topk_accuracy_i2t.png",
		full_topk_val_acc_t2i_fpth: str = "full_val_topk_accuracy_t2i.png",
		mean_reciprocal_rank_file_path: str = "mean_reciprocal_rank.png",
		cosine_similarity_file_path: str = "cosine_similarity.png",
		DPI: int = 300,
		figure_size: tuple = (10, 4),
	):

	num_epochs = len(train_losses)
	if num_epochs <= 1:  # No plot if only one epoch
		return
			
	# Setup common plotting configurations
	epochs = np.arange(1, num_epochs + 1)
	
	# Create selective x-ticks for better readability
	num_xticks = min(20, num_epochs)
	selective_xticks = np.linspace(1, num_epochs, num_xticks, dtype=int)
	
	# Define a consistent color palette
	colors = {
		'train': '#1f77b4',
		'val': '#ff7f0e',
		'img2txt': '#2ca02c',
		'txt2img': '#d62728'
	}
	
	# Common plot settings function
	def setup_plot(ax, xlabel='Epoch', ylabel=None, title=None):
		ax.set_xlabel(xlabel, fontsize=12)
		if ylabel:
			ax.set_ylabel(ylabel, fontsize=12)
		if title:
			ax.set_title(title, fontsize=10, fontweight='bold')
		ax.set_xlim(0, num_epochs + 1)
		ax.set_xticks(selective_xticks)
		ax.tick_params(axis='both', labelsize=10)
		ax.grid(True, linestyle='--', alpha=0.7)
		return ax
	
	# 1. Losses plot
	fig, ax = plt.subplots(figsize=figure_size)
	ax.plot(
		epochs,
		train_losses, 
		color=colors['train'], 
		label='Training', 
		lw=1.5, 
		marker='o', 
		markersize=2,
	)
	ax.plot(
		epochs,
		val_losses,
		color=colors['val'], 
		label='Validation',
		lw=1.5, 
		marker='o', 
		markersize=2,
	)
					
	setup_plot(
		ax, ylabel='Loss', 
		title=f'{dataset_name} Learning Curve: (Loss)',
	)
	ax.legend(
		fontsize=10, 
		loc='best', 
		frameon=True, 
		fancybox=True,
		shadow=True,
		facecolor='white',
		edgecolor='black',
	)
	fig.tight_layout()
	fig.savefig(losses_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
	
	# 1. Image-to-Text Top-K[in-batch matching] Validation Accuracy plot
	if in_batch_topk_val_accuracy_i2t_list:
		topk_values = list(in_batch_topk_val_accuracy_i2t_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in in_batch_topk_val_accuracy_i2t_list]
			ax.plot(
				epochs, 
				accuracy_values, 
				label=f'Top-{k}',
				lw=1.5, 
				marker='o', 
				markersize=2, 
				color=plt.cm.tab10(i),
			)
							 
		setup_plot(
			ax, 
			ylabel='Accuracy', 
			title=f'{dataset_name} Image-to-Text Top-K [in-batch matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9, 
			loc='best',
			ncol=len(topk_values),
			frameon=True, 
			fancybox=True,
			shadow=True,
			facecolor='white',
			edgecolor='black',
		)
		fig.tight_layout()
		fig.savefig(in_batch_topk_val_acc_i2t_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	# 2. Image-to-Text Top-K[full matching] Validation Accuracy plot
	if full_topk_val_accuracy_i2t_list:
		topk_values = list(full_topk_val_accuracy_i2t_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)

		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in full_topk_val_accuracy_i2t_list]
			ax.plot(
				epochs,
				accuracy_values,
				label=f'Top-{k}',
				lw=1.5,
				marker='o',
				markersize=2,
				color=plt.cm.tab10(i),
			)

		setup_plot(
			ax,
			ylabel='Accuracy',
			title=f'{dataset_name} Image-to-Text Top-K [full matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9,
			loc='best',
			ncol=len(topk_values),
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(full_topk_val_acc_i2t_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	if full_topk_val_accuracy_t2i_list:
		topk_values = list(full_topk_val_accuracy_t2i_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in full_topk_val_accuracy_t2i_list]
			ax.plot(
				epochs,
				accuracy_values,
				label=f'Top-{k}',
				lw=1.5,
				marker='o',
				markersize=2,
				color=plt.cm.tab10(i),
			)

		setup_plot(
			ax,
			ylabel='Accuracy',
			title=f'{dataset_name} Text-to-Image Top-K [full matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9,
			loc='best',
			ncol=len(topk_values),
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(full_topk_val_acc_t2i_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)

	# 3. Text-to-Image Top-K Accuracy plot
	if in_batch_topk_val_accuracy_t2i_list:
		topk_values = list(in_batch_topk_val_accuracy_t2i_list[0].keys())
		fig, ax = plt.subplots(figsize=figure_size)
		
		for i, k in enumerate(topk_values):
			accuracy_values = [epoch_data[k] for epoch_data in in_batch_topk_val_accuracy_t2i_list]
			ax.plot(
				epochs, 
				accuracy_values, 
				label=f'Top-{k}',
				lw=1.5, 
				marker='o', 
				markersize=2, 
				color=plt.cm.tab10(i),
			)
							 
		setup_plot(
			ax, 
			ylabel='Accuracy', 
			title=f'{dataset_name} Text-to-Image Top-K [in-batch matching] Validation Accuracy'
		)
		ax.set_ylim(-0.05, 1.05)
		ax.legend(
			fontsize=9, 
			loc='best',
			ncol=len(topk_values),
			frameon=True, 
			fancybox=True, 
			shadow=True,
			edgecolor='black',
			facecolor='white',
		)
		fig.tight_layout()
		fig.savefig(in_batch_topk_val_acc_t2i_fpth, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
	
	# 4. Mean Reciprocal Rank plot (if data provided)
	if mean_reciprocal_rank_list and len(mean_reciprocal_rank_list) > 0:
		fig, ax = plt.subplots(figsize=figure_size)
		ax.plot(
			epochs, 
			mean_reciprocal_rank_list, 
			color='#9467bd', 
			label='MRR', 
			lw=1.5, 
			marker='o', 
			markersize=2,
		)
						
		setup_plot(
			ax, 
			ylabel='Mean Reciprocal Rank',
			title=f'{dataset_name} Mean Reciprocal Rank (Image-to-Text)',
		)
		
		ax.set_ylim(-0.05, 1.05)
		ax.legend(fontsize=10, loc='best', frameon=True)
		fig.tight_layout()
		fig.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
	
	# 5. Cosine Similarity plot (if data provided)
	if cosine_similarity_list and len(cosine_similarity_list) > 0:
		fig, ax = plt.subplots(figsize=figure_size)
		ax.plot(
			epochs, 
			cosine_similarity_list, 
			color='#17becf',
			label='Cosine Similarity', 
			lw=1.5, 
			marker='o', 
			markersize=2,
		)
						
		setup_plot(
			ax, 
			ylabel='Cosine Similarity',
			title=f'{dataset_name} Cosine Similarity Between Embeddings'
		)
		ax.legend(fontsize=10, loc='best')
		fig.tight_layout()
		fig.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches='tight')
		plt.close(fig)
