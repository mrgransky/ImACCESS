from utils import *

def plot_all_pretrain_metrics(
		dataset_name: str,
		img2txt_metrics_dict: dict,
		txt2img_metrics_dict: dict,
		topK_values: list,
		fname: str = "all_pretrain_retrieval_metrics.png",
	):
	"""
	Plot retrieval metrics (mP@K, mAP@K, Recall@K) for all pre-trained CLIP models in a 2x3 subplot grid.
	Rows: Image-to-Text and Text-to-Image modes.
	Columns: mP@K, mAP@K, Recall@K metrics.
	"""
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	models = list(img2txt_metrics_dict.keys())  # ['RN50', 'RN101', ..., 'ViT-L/14@336px']
	colors = plt.cm.Set1.colors  # Use a distinct color for each of the 9 models
	markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'H']  # 9 distinct markers
	linestyles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-']  # Cycle through styles
	fig, axes = plt.subplots(len(modes), len(metrics), figsize=(18, 10), constrained_layout=True)
	fig.suptitle(f"{dataset_name} Pre-trained CLIP x{len(txt2img_metrics_dict)} models Retrieval Metrics", fontsize=16, fontweight='bold')
	for i, mode in enumerate(modes):
			metrics_dict = img2txt_metrics_dict if mode == "Image-to-Text" else txt2img_metrics_dict
			for j, metric in enumerate(metrics):
					ax = axes[i, j]
					legend_handles = []
					legend_labels = []
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
											linewidth=1.5,
											markersize=6,
									)
									legend_handles.append(line)
									legend_labels.append(model_name)
					ax.set_xlabel('K', fontsize=12)
					ax.set_ylabel(f'{metric}@K', fontsize=12)
					ax.set_title(f'{mode} - {metric}@K', fontsize=14, fontweight='bold')
					ax.grid(True, linestyle='--', alpha=0.7)
					ax.set_xticks(topK_values)
					ax.set_xlim(min(topK_values) - 1, max(topK_values) + 1)
					# Dynamic y-axis limits
					all_values = [v for m in models if m in metrics_dict for v in [metrics_dict[m][metric][str(k)] for k in k_values]]
					if all_values:
							min_val = min(all_values)
							max_val = max(all_values)
							padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
							ax.set_ylim(bottom=max(-0.01, min_val - padding), top=min(1.05, max_val + padding))
	# Add legend outside the subplots
	fig.legend(
			legend_handles,
			legend_labels,
			fontsize=8,
			loc='upper center',
			ncol=len(models) // 2 + len(models) % 2,  # Adjust columns based on number of models
			bbox_to_anchor=(0.5, 0.02),
			bbox_transform=fig.transFigure,
			frameon=True,
			shadow=True,
			fancybox=True,
			edgecolor='black',
			facecolor='white',
	)
	plt.tight_layout(rect=[0, 0.05, 1, 0.95])
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f"Saved combined pretrain metrics plot to {fname}")

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
	):
	metrics = list(image_to_text_metrics.keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"{dataset_name} Retrieval Performance Metrics [{best_model_name}]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | " 
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	modes = ['Image-to-Text', 'Text-to-Image']
	
	fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4), constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=8, fontweight='bold')
	
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []

	for i, metric in enumerate(metrics):
		ax = axes[i] if len(metrics) > 1 else axes # Handle single subplot case

		print(f"Image-to-Text:")
		it_top_ks = list(map(int, image_to_text_metrics[metric].keys()))  # K values for Image-to-Text
		it_vals = list(image_to_text_metrics[metric].values())
		print(metric, it_top_ks, it_vals)
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
		print(f"Text-to-Image:")
		ti_top_ks = list(map(int, text_to_image_metrics[metric].keys()))  # K values for Text-to-Image
		ti_vals = list(text_to_image_metrics[metric].values())
		print(metric, ti_top_ks, ti_vals)
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
		# Dynamic y-axis limits
		all_values = it_vals + ti_vals
		min_val = min(all_values)
		max_val = max(all_values)
		padding = 0.02 * (max_val - min_val) if (max_val - min_val) > 0 else 0.02
		ax.set_ylim(bottom=min(-0.02, min_val - padding), top=max(0.5, max_val + padding))
		print("*"*150)
		
	
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
	plt.savefig(fname, dpi=300, bbox_inches='tight')
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
			print(f"Note: K values for Image-to-Text are limited by the number of classes (e.g., 10 for CIFAR10).")

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

def plot_loss_accuracy(
		dataset_name: str,
		train_losses: List[float],
		val_losses: List[float],
		val_acc_img2txt_list: List[float],
		val_acc_txt2img_list: List[float],
		img2txt_topk_accuracy_list,
		txt2img_topk_accuracy_list,  # Added for text-to-image top-K
		mean_reciprocal_rank_list: List[float],
		cosine_similarity_list: List[float],
		losses_file_path: str ="losses.png",
		accuracy_file_path: str ="accuracy.png",
		img2txt_topk_accuracy_file_path: str ="img2txt_topk_accuracy.png",
		txt2img_topk_accuracy_file_path: str ="txt2img_topk_accuracy.png",  # Added for text-to-image top-K
		mean_reciprocal_rank_file_path: str ="mean_reciprocal_rank.png",
		cosine_similarity_file_path: str ="cosine_similarity.png",
		DPI: int=300,  # Higher DPI for publication quality
		figure_size=(11, 5),
	):
	num_epochs = len(train_losses)
	if num_epochs <= 1:  # No plot if only one epoch
		return
	epochs = range(1, num_epochs + 1)
	# Dynamic and selective number of xticks
	num_xticks = min(10, num_epochs)
	selective_xticks_epochs = np.linspace(0, num_epochs, num_xticks, dtype=int)
	# Consistent color scheme
	colors = {
		'train': '#1f77b4', # muted blue
		'val': '#ff7f0e', # safety orange
		'img2txt': '#2ca02c', # cooked asparagus green
		'txt2img': '#d62728', # brick red
		'mean_reciprocal_rank': '#9467bd', # muted purple
		'cosine_similarity': '#8c564b', # chestnut brown
		'img2img': '#e377c2', # raspberry yogurt pink
		'txt2txt': '#7f7f7f', # middle gray
	}
	modes = ["Image-to-Text", "Text-to-Image"]

	# 1. Loss Plot
	plt.figure(figsize=figure_size)
	plt.plot(
		epochs,
		train_losses,
		color=colors['train'],
		label='Training',
		lw=1.5,
		marker='o',
		markersize=2,
	)
	plt.plot(
		epochs,
		val_losses,
		color=colors['val'],
		label='Validation',
		lw=1.5,
		marker='o',
		markersize=2,
	)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Loss', fontsize=12)
	plt.title(f'{dataset_name} Training vs. Validation Loss', fontsize=9, fontweight='bold')
	plt.legend(
		fontsize=9, 
		loc='best', 
		ncol=len(modes), 
		frameon=True, 
		edgecolor='black', 
		fancybox=True,
	)
	plt.xlim(0, num_epochs + 1)
	# plt.ylim(0, max(max(train_losses), max(val_losses)) * 1.05)
	plt.xticks(selective_xticks_epochs, fontsize=8)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(losses_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

	# 2. Top-1 Accuracy Plot
	plt.figure(figsize=figure_size)
	plt.plot(
		epochs,
		val_acc_img2txt_list,
		color=colors['img2txt'],
		label='Image-to-Text',
		lw=1.5,
		marker='o',
		markersize=2,
	)
	plt.plot(
		epochs,
		val_acc_txt2img_list,
		color=colors['txt2img'],
		label='Text-to-Image',
		lw=1.5,
		marker='o',
		markersize=2,
	)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Accuracy', fontsize=12)
	plt.title(f'{dataset_name} Zero-Shot [in-batch matching Top-1 Accuracy]', fontsize=10, fontweight='bold')
	plt.legend(
		fontsize=9,
		loc='best',
		ncol=len(modes),
	)
	plt.xlim(0, num_epochs + 1)
	# plt.ylim(-0.05, 1.05)
	plt.ylim(0, max(max(val_acc_img2txt_list), max(val_acc_txt2img_list)) * 1.05)
	plt.xticks(selective_xticks_epochs, fontsize=9)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

	# 3. Image-to-Text Top-K Accuracy Plot
	topk_values = list(img2txt_topk_accuracy_list[0].keys())  # e.g., [1, 3, 5]
	plt.figure(figsize=figure_size)
	for i, k in enumerate(topk_values):
		accuracy_values = [epoch_data[k] for epoch_data in img2txt_topk_accuracy_list]
		plt.plot(
			epochs,
			accuracy_values,
			label=f'Top-{k}',
			lw=1.5,
			marker='o',
			markersize=2,
			color=plt.cm.tab10(i), # # Distinct colors for each K value
		)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Accuracy', fontsize=12)
	plt.title(f'{dataset_name} Image-to-Text Top-K Accuracy (K={topk_values})', fontsize=10, fontweight='bold')
	plt.legend(
		fontsize=8,
		loc='best',
		ncol=len(topk_values),
		frameon=True,
		fancybox=True,
		shadow=True,
	)
	plt.xlim(0, num_epochs + 1)
	plt.ylim(-0.05, 1.05)
	plt.xticks(selective_xticks_epochs, fontsize=10)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(img2txt_topk_accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

	# 4. Text-to-Image Top-K Accuracy Plot (New)
	topk_values = list(txt2img_topk_accuracy_list[0].keys())  # e.g., [1, 3, 5]
	plt.figure(figsize=figure_size)
	for i, k in enumerate(topk_values):
		accuracy_values = [epoch_data[k] for epoch_data in txt2img_topk_accuracy_list]
		plt.plot(
			epochs,
			accuracy_values,
			label=f'Top-{k}',
			lw=1.5,
			marker='o',
			markersize=2,
			color=plt.cm.tab10(i),
		)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Accuracy', fontsize=12)
	plt.title(f'{dataset_name} Text-to-Image Top-K Accuracy (K={topk_values})', fontsize=10, fontweight='bold')
	plt.legend(
		fontsize=10,
		loc='best',
		ncol=len(topk_values),
	)
	plt.xlim(0, num_epochs + 1)
	plt.ylim(-0.05, 1.05)
	plt.xticks(selective_xticks_epochs, fontsize=10)
	plt.grid(True, linestyle='--', alpha=0.5)
	plt.tight_layout()
	plt.savefig(txt2img_topk_accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close()

	# # 5. Mean Reciprocal Rank Plot
	# plt.figure(figsize=figure_size)
	# plt.plot(
	# 	epochs,
	# 	mean_reciprocal_rank_list,
	# 	color='#9467bd',
	# 	label='MRR',
	# 	lw=1.5,
	# 	marker='o', 
	# 	markersize=2,
	# )
	# plt.xlabel('Epoch', fontsize=12)
	# plt.ylabel('Mean Reciprocal Rank', fontsize=12)
	# plt.title(f'{dataset_name} Mean Reciprocal Rank (Image-to-Text)', fontsize=14, fontweight='bold', pad=10)
	# plt.legend(fontsize=10, loc='upper left', )
	# plt.xlim(0, num_epochs + 1)
	# plt.ylim(-0.05, 1.05)
	# plt.xticks(selective_xticks_epochs, fontsize=10)
	# plt.grid(True, linestyle='--', alpha=0.5)
	# plt.tight_layout()
	# plt.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches='tight')
	# plt.close()

	# # 6. Cosine Similarity Plot
	# plt.figure(figsize=figure_size)
	# plt.plot(
	# 	epochs,
	# 	cosine_similarity_list,
	# 	color='#17becf',
	# 	label='Cosine Similarity',
	# 	lw=1.5, 
	# 	marker='o', 
	# 	markersize=2,
	# )
	# plt.xlabel('Epoch', fontsize=12)
	# plt.ylabel('Cosine Similarity', fontsize=12)
	# plt.title(f'{dataset_name} Cosine Similarity Between Embeddings', fontsize=14, fontweight='bold', pad=10)
	# plt.legend(fontsize=10, loc='upper left', )
	# plt.xlim(0, num_epochs + 1)
	# plt.xticks(selective_xticks_epochs, fontsize=10)
	# plt.grid(True, linestyle='--', alpha=0.5)
	# plt.tight_layout()
	# plt.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches='tight')
	# plt.close()