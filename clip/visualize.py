from utils import *

def plot_retrieval_metrics_best_model(
	image_to_text_metrics: Dict[str, Dict[str, float]],
	text_to_image_metrics: Dict[str, Dict[str, float]],
	fname: str ="Retrieval_Performance_Metrics_best_model.png",
	best_model_name: str ="Best Model",
	):
	metrics = list(image_to_text_metrics.keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"Retrieval Performance Metrics [{best_model_name}]: "
	for metric in metrics:
		suptitle_text += f"{metric}@K | " 
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	modes = ['Image-to-Text', 'Text-to-Image']
	
	fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4), constrained_layout=True)
	fig.suptitle(suptitle_text, fontsize=11, fontweight='bold')
	
	# Store legend handles and labels
	legend_handles = []
	legend_labels = []

	for i, metric in enumerate(metrics):
		ax = axes[i] if len(metrics) > 1 else axes # Handle single subplot case
		it_top_ks = list(map(int, image_to_text_metrics[metric].keys()))  # K values for Image-to-Text
		it_vals = list(image_to_text_metrics[metric].values())
		print("Image-to-Text: ", metric, it_top_ks, it_vals)
		# Plotting for Image-to-Text
		line, = ax.plot(
			it_top_ks, 
			it_vals, 
			marker='o', 
			label=modes[0], 
			color='blue', 
			linestyle='-', 
			linewidth=1.25,
			markersize=4,
		)
		if modes[0] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[0])
		
		# Plotting for Text-to-Image
		ti_top_ks = list(map(int, text_to_image_metrics[metric].keys()))  # K values for Text-to-Image
		ti_vals = list(text_to_image_metrics[metric].values())
		print("Text-to-Image: ", metric, ti_top_ks, it_vals)
		line, = ax.plot(
			ti_top_ks,
			ti_vals,
			marker='s',
			label=modes[1],
			color='red',
			linestyle='-',
			linewidth=1.25,
			markersize=4,
		)
		if modes[1] not in legend_labels:
			legend_handles.append(line)
			legend_labels.append(modes[1])
		
		ax.set_xlabel('K', fontsize=12)
		ax.set_ylabel(f'{metric}@K', fontsize=11)
		ax.set_title(f'{metric}@K', fontsize=12)
		ax.grid(True, linestyle='--', alpha=0.7)
		
		# Set the x-axis to only show integer values
		all_ks = sorted(set(it_top_ks + ti_top_ks))
		ax.set_xticks(all_ks)

		
		# Adjust y-axis to start from 0 for better visualization
		ax.set_ylim(bottom=-0.05, top=1.05)
	
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	fig.legend(
		legend_handles,
		legend_labels,
		fontsize=10,
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

# def plot_retrieval_metrics_per_epoch(
# 	image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
# 	text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
# 	fname: str="Retrieval_Performance_Metrics.png",
# 	):
# 	num_epochs = len(image_to_text_metrics_list)
# 	if num_epochs < 2:
# 		return

# 	valid_K_values = [K for K in topK_values if str(K) in image_to_text_metrics_list[0]["mP"]]
# 	if len(valid_K_values) < len(topK_values):
# 		print(f"<!> Warning: K values ({set(topK_values) - set(valid_K_values)}) exceed the number of classes. They will be ignored.")

# 	epochs = range(1, num_epochs + 1)
# 	modes = ["Image-to-Text", "Text-to-Image"]
# 	metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
# 	suptitle_text = f"Retrieval Performance Metrics [per epoch]: "
# 	for metric in metrics:
# 		suptitle_text += f"{metric}@K | " 
# 	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "

# 	cmap = plt.get_cmap("tab10")  # Use a colormap with at least 10 colors
# 	colors = [cmap(i) for i in range(cmap.N)]
# 	markers = ['D', 'v', 'o', 's', '^', 'P', 'X', 'd', 'H', 'h']  # Different markers for each line
# 	line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for each metric
# 	fig, axs = plt.subplots(len(modes), len(metrics), figsize=(20, 11), constrained_layout=True)
# 	fig.suptitle(suptitle_text, fontsize=15, fontweight='bold')
# 	# Store legend handles and labels
# 	legend_handles = []
# 	legend_labels = []
# 	for i, task_metrics_list in enumerate([image_to_text_metrics_list, text_to_image_metrics_list]):
# 		for j, metric in enumerate(metrics):
# 			ax = axs[i, j]
# 			for K, color, marker, linestyle in zip(valid_K_values, colors, markers, line_styles):
# 				values = []
# 				for metrics_dict in task_metrics_list:
# 					if metric in metrics_dict and str(K) in metrics_dict[metric]:
# 						values.append(metrics_dict[metric][str(K)])
# 					else:
# 						values.append(0)
# 				line, = ax.plot(
# 					epochs,
# 					values,
# 					marker=marker,
# 					markersize=6,
# 					linestyle=linestyle,
# 					label=f'K={K}',
# 					color=color, 
# 					alpha=0.8,
# 					linewidth=2.0,
# 				)
# 				# Collect handles and labels for the legend
# 				if f'K={K}' not in legend_labels:
# 					legend_handles.append(line)
# 					legend_labels.append(f'K={K}')
# 			ax.set_xlabel('Epoch', fontsize=12)
# 			ax.set_ylabel(f'{metric}@K', fontsize=12)
# 			ax.set_title(f'{modes[i]} - {metric}@K', fontsize=14)
# 			# ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
# 			ax.grid(True, linestyle='--', alpha=0.7)
# 			ax.set_xticks(epochs)
# 			ax.set_ylim(bottom=-0.05, top=1.05)
# 	fig.legend(
# 		legend_handles,
# 		legend_labels,
# 		fontsize=11,
# 		loc='upper center',
# 		ncol=len(valid_K_values),
# 		bbox_to_anchor=(0.5, 0.96),
# 		bbox_transform=fig.transFigure,
# 		frameon=True,
# 		shadow=True,
# 		fancybox=True,
# 		edgecolor='black',
# 		facecolor='white',
# 	)
# 	plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
# 	plt.savefig(fname, dpi=300, bbox_inches='tight')
# 	plt.close(fig)


def plot_retrieval_metrics_per_epoch(
	image_to_text_metrics_list: List[Dict[str, Dict[str, float]]],
	text_to_image_metrics_list: List[Dict[str, Dict[str, float]]],
	fname: str = "Retrieval_Performance_Metrics.png",
	):
	num_epochs = len(image_to_text_metrics_list)
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
	epochs = range(1, num_epochs + 1)
	modes = ["Image-to-Text", "Text-to-Image"]
	metrics = list(image_to_text_metrics_list[0].keys())  # ['mP', 'mAP', 'Recall']
	suptitle_text = f"Retrieval Performance Metrics [per epoch]: "
	for metric in metrics:
			suptitle_text += f"{metric}@K | "
	suptitle_text = suptitle_text[:-3]  # Remove trailing " | "
	cmap = plt.get_cmap("tab10")  # Use a colormap with at least 10 colors
	colors = [cmap(i) for i in range(cmap.N)]
	markers = ['D', 'v', 'o', 's', '^', 'P', 'X', 'd', 'H', 'h']  # Different markers for each line
	line_styles = ['-', '--', '-.', ':', '-']  # Different line styles for each metric
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
					for k_idx, (K, color, marker, linestyle) in enumerate(zip(valid_k_values, colors, markers, line_styles)):
							values = []
							for metrics_dict in task_metrics_list:
									if metric in metrics_dict and str(K) in metrics_dict[metric]:
											values.append(metrics_dict[metric][str(K)])
									else:
											values.append(0)  # Default to 0 if K value is missing (shouldn’t happen with valid data)
							line, = ax.plot(
									epochs,
									values,
									marker=marker,
									markersize=6,
									linestyle=linestyle,
									label=f'K={K}',
									color=color,
									alpha=0.8,
									linewidth=2.0,
							)
							# Collect handles and labels for the legend, but only for the first occurrence of each K
							if f'K={K}' not in legend_labels:
									legend_handles.append(line)
									legend_labels.append(f'K={K}')
					ax.set_xlabel('Epoch', fontsize=12)
					ax.set_ylabel(f'{metric}@K', fontsize=12)
					ax.set_title(f'{modes[i]} - {metric}@K', fontsize=14)
					ax.grid(True, linestyle='--', alpha=0.7)
					ax.set_xticks(epochs)
					ax.set_ylim(bottom=-0.05, top=1.05)
	
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
		train_losses,
		val_losses,
		val_acc_img2txt_list,
		val_acc_txt2img_list,
		img2txt_topk_accuracy_list,
		mean_reciprocal_rank_list,
		cosine_similarity_list,
		losses_file_path="losses.png",
		accuracy_file_path="accuracy.png",
		topk_accuracy_file_path="img2txt_topk_accuracy.png",
		mean_reciprocal_rank_file_path="mean_reciprocal_rank.png",
		cosine_similarity_file_path="cosine_similarity.png",
		DPI=250,
		figure_size=(11, 5),
	):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	epochs = range(1, num_epochs + 1)

	# Set xticks to be dynamically defined
	num_xticks = 10
	# Check if num_xticks is greater than num_epochs + 1
	if num_xticks > num_epochs + 1:
		num_xticks = num_epochs + 1
	xticks = np.arange(0, num_epochs + 1, (num_epochs + 1) // num_xticks)

	# Plot losses:
	plt.figure(figsize=figure_size)
	plt.plot(epochs, train_losses, color='b', label='Train', lw=1.25)
	plt.plot(epochs, val_losses, color='r', label='Validation', lw=1.25)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(os.path.splitext(os.path.basename(losses_file_path))[0], fontsize=10)
	plt.legend(ncols=2, fontsize=10, loc='best')
	plt.grid(True)
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.tight_layout()
	plt.savefig(losses_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)

	plt.figure(figsize=figure_size)
	plt.plot(epochs, val_acc_img2txt_list, label='text retrieval per image')
	plt.plot(epochs, val_acc_txt2img_list, label='image retrieval per text')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(os.path.splitext(os.path.basename(accuracy_file_path))[0], fontsize=10)
	plt.legend(title='[Top-1] Accuracy (Zero-Shot)', fontsize=8, title_fontsize=9, loc='best')
	plt.grid(True)
	plt.xticks(xticks, fontsize=7)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter to make space for the title
	plt.savefig(accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
	
	plt.figure(figsize=figure_size, constrained_layout=True)
	print(epochs)
	print(img2txt_topk_accuracy_list)

	# for k, acc in enumerate(img2txt_topk_accuracy_list):
	# 	plt.plot(epochs, acc, label=f'Top-{k+1}')
	topk_values = list(img2txt_topk_accuracy_list[0].keys()) # [1, 5, 10]
	print(topk_values)
	for k in topk_values:
		accuracy_values = [epoch_data[k] for epoch_data in img2txt_topk_accuracy_list]
		plt.plot(epochs, accuracy_values, marker='o', label=f"Top-{k}")
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title(f"Image-to-Text Top-K Accuracy (K={topk_values})", fontsize=10, fontweight='bold')
	plt.legend(ncols=len(img2txt_topk_accuracy_list), loc='best')
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.xticks(xticks, fontsize=7)
	plt.ylim([0, 1])
	plt.savefig(topk_accuracy_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
	
	plt.figure(figsize=figure_size)
	plt.plot(epochs, mean_reciprocal_rank_list,  label='Mean Reciprocal Rank')
	plt.xlabel('Epoch')
	plt.ylabel('MRR')
	plt.title("Mean Reciprocal Rank")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.ylim([0, 1])
	plt.xticks(xticks, fontsize=7)
	plt.savefig(mean_reciprocal_rank_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
		
	plt.figure(figsize=figure_size)
	plt.plot(epochs, cosine_similarity_list,  linestyle='-', color='g', label='Cosine Similarity')
	plt.xlabel('Epoch')
	plt.ylabel('Cosine Similarity')
	plt.title("Cosine Similarity Over Epochs", fontsize=10)
	plt.grid(True)
	plt.tight_layout()
	plt.legend()
	plt.xlim(0, num_epochs + 1)
	plt.xticks(xticks, fontsize=7)
	plt.savefig(cosine_similarity_file_path, dpi=DPI, bbox_inches='tight')
	plt.close(fig)
