from utils import *

def plot_comparison_metrics(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		finetune_strategy: str,  # e.g., 'LoRA'
		topK_values: list,
		fname_prefix: str="Comparison_Metrics",
		fname: str="comparison.png",
		figure_size=(12, 5),
		DPI: int=300,
	):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
		model_name_idx = all_model_architectures.index(model_name) if model_name in all_model_architectures else 0
		model_colors = plt.cm.tab10.colors
		
		# Create separate figures for each mode
		for i, mode in enumerate(modes):
				# Create a new figure for each mode
				fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
				
				# Set a descriptive title for the figure
				fig.suptitle(
						f"{dataset_name} Dataset - {mode} Retrieval Performance\n"
						f"Pre-trained CLIP {model_name} vs. {finetune_strategy.capitalize()} Fine-tuning",
						fontsize=14,
						fontweight='bold',
				)
				
				# Select the appropriate dictionaries
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				
				# Track best improvement for annotation
				best_improvements = {metric: {'value': 0, 'k': 0} for metric in metrics}
				
				for j, metric in enumerate(metrics):
						ax = axes[j]
						
						# Create lists to store values for statistical annotations
						pretrained_values = []
						finetuned_values = []
						
						# Plot pre-trained model performance
						if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
								k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
								values = [pretrained_dict[model_name][metric][str(k)] for k in k_values]
								pretrained_values = values
								
								pretrained_line, = ax.plot(
										k_values,
										values,
										label=f"Pre-trained",
										color=model_colors[model_name_idx],
										marker='o',
										linestyle='--',
										linewidth=2,
										markersize=5,
										alpha=0.7,
								)
						
						# Plot fine-tuned model performance
						if model_name in finetuned_dict and metric in finetuned_dict[model_name]:
								k_values = sorted([int(k) for k in finetuned_dict[model_name][metric].keys() if int(k) in topK_values])
								values = [finetuned_dict[model_name][metric][str(k)] for k in k_values]
								finetuned_values = values
								
								finetuned_line, = ax.plot(
										k_values,
										values,
										label=f"{finetune_strategy} Fine-tuned",
										color=model_colors[model_name_idx],
										marker='s',
										linestyle='-',
										linewidth=2,
										markersize=5
								)
								
								# Add improvement percentages at key points
								if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
										# Calculate improvement at all K values
										all_improvements = []
										for k_idx, k in enumerate(k_values):
												if str(k) in pretrained_dict[model_name][metric]:
														pretrained_val = pretrained_dict[model_name][metric][str(k)]
														finetuned_val = values[k_idx]
														improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
														all_improvements.append((k, improvement))
														
														# Track the best improvement
														if improvement > best_improvements[metric]['value']:
																best_improvements[metric]['value'] = improvement
																best_improvements[metric]['k'] = k
										
										# Annotate key improvement points (top 3 improvements)
										sorted_improvements = sorted(all_improvements, key=lambda x: abs(x[1]), reverse=True)
										for idx, (k, improvement) in enumerate(sorted_improvements[:2]):  # Annotate top 2 improvements
												k_idx = k_values.index(k)
												finetuned_val = values[k_idx]
												
												# Set color based on improvement value
												text_color = 'darkgreen' if improvement >= 0 else 'red'
												
												# Place annotations to the right with slight upward offset
												ax.annotate(
														f"{'+' if improvement >= 0 else ''}{improvement:.1f}% @K={k}",
														xy=(k, finetuned_val),
														xytext=(5, 5 + idx * 15),  # Offset to avoid overlap
														textcoords='offset points',
														fontsize=8.5,
														fontweight='bold',
														color=text_color,
														bbox=dict(
																facecolor='white',
																edgecolor='none',
																alpha=0.7,
																pad=0.3
														),
														arrowprops=dict(
																arrowstyle="->",
																color=text_color,
																shrinkA=5,
																shrinkB=5,
																alpha=0.7
														)
												)
						
						# Add statistical analysis summary if we have both pretrained and finetuned values
						if pretrained_values and finetuned_values and len(pretrained_values) == len(finetuned_values):
								# Calculate average improvement
								avg_improvement = sum([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)]) / len(pretrained_values)
								
								# Calculate maximum improvement
								max_improvement = max([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)])
								max_k = k_values[np.argmax([((f - p) / p) * 100 for p, f in zip(pretrained_values, finetuned_values)])]
								
								# Add text box with statistics
								stat_text = (
										f"Average Improvement: {avg_improvement:.1f}%\n"
										f"Maximum Improvement: {max_improvement:.1f}% @K={max_k}"
								)
								
								# Add the statistics box in upper left or right corner
								x_pos = 0.05 if avg_improvement > 0 else 0.55  # Left if positive, right if negative
								ax.text(
										x_pos, 0.95, stat_text,
										transform=ax.transAxes,
										fontsize=8,
										verticalalignment='top',
										bbox=dict(
												boxstyle='round,pad=0.5',
												facecolor='white',
												alpha=0.8,
												edgecolor='gray'
										)
								)
						
						# Configure axes
						ax.set_xlabel('K', fontsize=12)
						ax.set_ylabel(f'{metric}@K', fontsize=12)
						ax.set_title(f'{metric}@K', fontsize=14)
						ax.grid(True, linestyle='--', alpha=0.7)
						ax.set_xticks(topK_values)
						
						# Set y-axis limits based on data
						all_values = []
						if model_name in pretrained_dict and metric in pretrained_dict[model_name]:
								all_values.extend([pretrained_dict[model_name][metric][str(k)] for k in k_values if str(k) in pretrained_dict[model_name][metric]])
						if model_name in finetuned_dict and metric in finetuned_dict[model_name]:
								all_values.extend([finetuned_dict[model_name][metric][str(k)] for k in k_values if str(k) in finetuned_dict[model_name][metric]])
						
						if all_values:
								min_val = min(all_values)
								max_val = max(all_values)
								padding = 0.1 * (max_val - min_val) if max_val > min_val else 0.1
								ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.0, max_val + padding))
						
						# Add legend to first subplot only
						if j == 0:
								ax.legend(fontsize=10, loc='lower right')
				
				# Add a summary of best improvements across metrics at the bottom of the figure
				summary = "\n".join([
						f"Best {metric} improvement: {best_improvements[metric]['value']:.1f}% at K={best_improvements[metric]['k']}"
						for metric in metrics
				])
				fig.text(
						0.5, 0.01, 
						summary,
						ha='center',
						fontsize=9,
						bbox=dict(
								boxstyle='round,pad=0.5',
								facecolor='lightyellow',
								alpha=0.8,
								edgecolor='gray'
						)
				)
				
				# Save the figure for this mode
				plt.savefig(fname=f"{fname_prefix}_{mode.replace('-', '_')}.png", dpi=DPI, bbox_inches='tight')
				plt.close(fig)
				
		# Create an additional summary plot showing the relative percentage improvements
		fig, axes = plt.subplots(1, 3, figsize=figure_size, constrained_layout=True)
		fig.suptitle(
				f"{dataset_name} Dataset - Relative Improvement from Fine-tuning\n"
				f"CLIP {model_name} with {finetune_strategy.capitalize()} Strategy",
				fontsize=14,
				fontweight='bold',
		)
		
		for j, metric in enumerate(metrics):
				ax = axes[j]
				
				# Extract improvement percentages for both modes
				improvements_by_mode = {}
				
				for mode in modes:
						pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
						finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
						
						if (model_name in pretrained_dict and metric in pretrained_dict[model_name] and
								model_name in finetuned_dict and metric in finetuned_dict[model_name]):
								
								k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
								improvements = []
								
								for k in k_values:
										str_k = str(k)
										if str_k in pretrained_dict[model_name][metric] and str_k in finetuned_dict[model_name][metric]:
												pretrained_val = pretrained_dict[model_name][metric][str_k]
												finetuned_val = finetuned_dict[model_name][metric][str_k]
												improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
												improvements.append(improvement)
								
								improvements_by_mode[mode] = {
										'k_values': k_values,
										'improvements': improvements
								}
				
				# Plot improvements for each mode
				bar_width = 0.35
				for mode_idx, mode in enumerate(modes):
						if mode in improvements_by_mode:
								k_values = improvements_by_mode[mode]['k_values']
								improvements = improvements_by_mode[mode]['improvements']
								
								x_positions = np.array(range(len(k_values))) + (mode_idx - 0.5) * bar_width
								bars = ax.bar(
										x_positions,
										improvements,
										bar_width,
										label=mode,
										color=model_colors[mode_idx],
										alpha=0.7
								)
								
								# Add value labels on top of bars
								for bar_idx, bar in enumerate(bars):
										height = bar.get_height()
										align = 'center'
										va = 'bottom' if height >= 0 else 'top'
										
										ax.text(
												bar.get_x() + bar.get_width() / 2,
												height + (5 if height >= 0 else -5),
												f"{height:.1f}%",
												ha=align,
												va=va,
												fontsize=8,
												rotation=90,
												color='black'
										)
				
				# Configure axes
				ax.set_xlabel('K', fontsize=12)
				ax.set_ylabel('Relative Improvement (%)', fontsize=12)
				ax.set_title(f'{metric}@K Improvement', fontsize=14)
				ax.grid(True, linestyle='--', alpha=0.7, axis='y')
				
				# Set x-axis ticks to K values
				if any(improvements_by_mode.values()):
						# Use first available mode's K values for x-axis labels
						first_mode = next(iter(improvements_by_mode.values()))
						ax.set_xticks(range(len(first_mode['k_values'])))
						ax.set_xticklabels([f"K={k}" for k in first_mode['k_values']])
				
				# Add zero line for reference
				ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
				
				# Add legend to first subplot only
				if j == 0:
						ax.legend(fontsize=10)
		
		# Save the improvement summary figure
		plt.savefig(
			# fname=f"{fname_prefix}_Relative_Improvements.png", 
			dpi=DPI, 
			bbox_inches='tight',
		)
		plt.close(fig)

def plot_comparison_metrics_original(
		dataset_name: str,
		pretrained_img2txt_dict: dict,
		pretrained_txt2img_dict: dict,
		finetuned_img2txt_dict: dict,
		finetuned_txt2img_dict: dict,
		model_name: str,  # e.g., 'ViT-B/32'
		finetune_strategy: str,  # e.g., 'LoRA'
		topK_values: list,
		fname: str="Comparison_Metrics.png",
		figure_size=(14, 8),
		DPI: int=300,
	):
	metrics = ["mP", "mAP", "Recall"]
	modes = ["Image-to-Text", "Text-to-Image"]
	all_model_architectures = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
	model_name_idx = all_model_architectures.index(model_name)
	model_colors = plt.cm.tab10.colors
	
	# Create figure with 2x3 subplots
	fig, axes = plt.subplots(2, 3, figsize=figure_size, constrained_layout=True)
	fig.suptitle(
		f"Retrieval Performance Comparison\nPre-trained CLIP {model_name} vs. {finetune_strategy.capitalize()} Fine-tuning",
		fontsize=10,
		fontweight='bold',
	)
	
	# Plot data for each mode and metric
	for i, mode in enumerate(modes):
		# Select the appropriate dictionaries
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		print(f"mode: {mode}")
		print(f"pretrained_dict: {pretrained_dict}")
		print(f"finetuned_dict: {finetuned_dict}")
		print()
		for j, metric in enumerate(metrics):
			ax = axes[i, j]
			
			# Plot pre-trained model performance
			if model_name in pretrained_dict:
				k_values = sorted([int(k) for k in pretrained_dict[model_name][metric].keys() if int(k) in topK_values])
				values = [pretrained_dict[model_name][metric][str(k)] for k in k_values]
				
				pretrained_line, = ax.plot(
					k_values,
					values,
					label=f"Pre-trained",
					color=model_colors[model_name_idx],
					marker='o',
					linestyle='--',
					linewidth=2,
					markersize=5,
					alpha=0.55,
				)
			
			# Plot fine-tuned model performance
			if model_name in finetuned_dict:
				k_values = sorted([int(k) for k in finetuned_dict[model_name][metric].keys() if int(k) in topK_values])
				values = [finetuned_dict[model_name][metric][str(k)] for k in k_values]
				
				finetuned_line, = ax.plot(
					k_values,
					values,
					label=f"Fine-tuned",
					color=model_colors[model_name_idx],
					marker='s',
					linestyle='-',
					linewidth=2,
					markersize=5
				)
				
				# Add improvement percentages at key points
				if model_name in pretrained_dict:
					key_k_values = [1, 10, 20]  # Annotate these K values if available
					for k in key_k_values:
						if k in k_values:
							k_idx = k_values.index(k)
							pretrained_val = pretrained_dict[model_name][metric][str(k)]
							finetuned_val = values[k_idx]
							improvement = ((finetuned_val - pretrained_val) / pretrained_val) * 100
							
							# Set color based on improvement value
							text_color = 'darkgreen' if improvement >= 0 else 'red'
							
							# Place annotations to the right with slight upward offset
							ax.annotate(
								f"{'+' if improvement >= 0 else ''}{improvement:.1f}%",
								xy=(k, finetuned_val),
								xytext=(5, 5),  # Fixed offset to the right and slightly up
								textcoords='offset points',
								fontsize=8.5,
								fontweight='bold',
								color=text_color,  # Apply the color
								bbox=dict(
									facecolor='white',
									edgecolor='none',
									alpha=0.7,
									pad=0.3
								)
							)
			
			# Configure axes
			ax.set_xlabel('K', fontsize=12)
			ax.set_ylabel(f'{metric}@K', fontsize=12)
			ax.set_title(f'{mode} - {metric}@K', fontsize=14)
			ax.grid(True, linestyle='--', alpha=0.7)
			ax.set_xticks(topK_values)
			
			# Set y-axis limits based on data
			all_values = []
			if model_name in pretrained_dict:
				all_values.extend([pretrained_dict[model_name][metric][str(k)] for k in k_values])
			if model_name in finetuned_dict:
				all_values.extend([finetuned_dict[model_name][metric][str(k)] for k in k_values])
			
			if all_values:
				min_val = min(all_values)
				max_val = max(all_values)
				padding = 0.1 * (max_val - min_val) if max_val > min_val else 0.1
				ax.set_ylim(bottom=max(0, min_val - padding), top=min(1.0, max_val + padding))
				# ax.set_ylim(bottom=-0.05, top=1.05)

			# Add legend to first subplot only
			if i == 0 and j == 0:
				ax.legend(fontsize=10)
							
	plt.savefig(fname=fname, dpi=DPI, bbox_inches='tight')
	plt.close(fig)

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
	
	# 2. Image-to-Text Top-K[in-batch matching] Validation Accuracy plot
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
		ax.legend(fontsize=10, loc='best')
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
