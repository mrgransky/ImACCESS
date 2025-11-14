import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde, t
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont

import seaborn as sns
import inspect
import time
import random
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import os # For checking file existence
from itertools import combinations # For pairwise label combinations
import torch
from torch.utils.data import DataLoader, Dataset

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from typing import Tuple, Union, List, Dict, Any, Optional
Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str, 'country': str,
}

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)
sns.set_style("whitegrid")

def plot_image_to_texts_separate_horizontal_bars(
				models: dict,
				validation_loader: DataLoader,
				preprocess,
				img_path: str,
				topk: int,
				device: str,
				results_dir: str,
				dpi: int = 250,
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
		
		# IMPROVED LAYOUT CALCULATION
		# Get image dimensions for dynamic sizing
		img_width, img_height = img.size
		aspect_ratio = img_height / img_width
		
		# Number of models to display
		num_strategies = len(models)
		
		# Base the entire layout on the image aspect ratio
		img_display_width = 4  # Base width for image in inches
		img_display_height = img_display_width * aspect_ratio
		
		# Set model result panels to have identical height as the image
		# Each model panel should have a fixed width ratio relative to the image
		model_panel_width = 3.5  # Width for each model panel
		
		# Calculate total figure dimensions
		fig_width = img_display_width + (model_panel_width * num_strategies)
		fig_height = max(4, img_display_height)  # Ensure minimum height
		
		# Create figure
		fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
		
		# Create grid with precise width ratios
		# First column for image, remaining columns for models
		width_ratios = [img_display_width] + [model_panel_width] * num_strategies
		
		# Create GridSpec with exact dimensions
		gs = gridspec.GridSpec(
				1, 
				1 + num_strategies, 
				width_ratios=width_ratios,
				wspace=0.05  # Reduced whitespace between panels
		)
		
		# Subplot 1: Query Image
		ax0 = fig.add_subplot(gs[0])
		ax0.imshow(img)
		ax0.axis('off')
		
		# Add title with ground truth if available
		title_text = f"Query Image\nGT: {ground_truth_label.capitalize()}" if ground_truth_label else "Query Image"
		ax0.text(
				0.5,  # x position (center)
				-0.05,  # y position (just below the image)
				title_text,
				fontsize=10,
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
		
		# Create subplots for models - ensuring dimensions are consistent
		for model_idx in range(num_strategies):
				ax = fig.add_subplot(gs[model_idx + 1])
				axes.append(ax)
		
		# Create a list of handles for the legend
		legend_handles = []
		
		# Plot data for each model
		for model_idx, (model_name, ax) in enumerate(zip(all_strategies, axes)):
				y_pos = np.arange(topk)
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
				
				# Format axis appearance
				ax.invert_yaxis()  # Highest probs on top
				ax.set_yticks([])  # Hide y-axis ticks 
				ax.set_yticklabels([])  # Empty labels
				
				# Set consistent x-axis limits and ticks
				ax.set_xlim(0, 1)
				ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
				ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=8)
				ax.grid(True, axis='x', linestyle='--', alpha=0.5, color='#888888')
				
				# Annotate bars with labels and probabilities
				for i, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
						formatted_label = label.replace('_', ' ').title()
						ax.text(
								prob + 0.01 if prob < 0.5 else prob - 0.01,
								i,
								f"{formatted_label}\n({prob:.2f})",
								va='center',
								ha='right' if prob > 0.5 else 'left',
								fontsize=8,
								color='black',
								fontweight='bold' if prob == max(sorted_probs) else 'normal',
						)
				
				# Set border color
				for spine in ax.spines.values():
						spine.set_color('black')
		
		# Add a legend at the top of the figure
		fig.legend(
				legend_handles,
				[handle.get_label() for handle in legend_handles],
				fontsize=11,
				loc='upper center',
				ncol=len(legend_handles),
				bbox_to_anchor=(0.5, 1.02),
				bbox_transform=fig.transFigure,
				frameon=True,
				shadow=True,
				fancybox=True,
				edgecolor='black',
				facecolor='white',
		)
		
		# Add x-axis label
		fig.text(
				0.5,  # x position (center of figure)
				0.02,  # y position (near bottom of figure)
				"Probability",
				ha='center',
				va='center',
				fontsize=12,
				fontweight='bold'
		)
		
		# IMPORTANT: Instead of tight_layout which can override our settings,
		# use a more controlled approach
		fig.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95)
		
		# Save the figure
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
		dpi=200,
		scale_factor=10.0,
	):
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
			title_height = int(110 * scale_factor)  # Scale title height
			
			# First determine dimensions
			heights = [img.height for img in topk_images]
			widths = [img.width for img in topk_images]
			
			# Use the same aspect ratio for all images
			max_height = max(heights)
			
			# Scale max_height to make images larger while preserving aspect ratio
			scaled_max_height = int(max_height * scale_factor)
			
			# Resize images to have same height and apply scaling
			for i in range(len(topk_images)):
					target_height = scaled_max_height
					target_width = int(topk_images[i].width * (target_height / topk_images[i].height))
					topk_images[i] = topk_images[i].resize((target_width, target_height), Image.LANCZOS)
			
			# Update widths after resizing
			widths = [img.width for img in topk_images]
			
			# Create a composite image
			total_width = sum(widths)
			print(f"Composite dimensions: {total_width} x {scaled_max_height + title_height}")
			
			composite = Image.new(
				mode='RGB',
				size=(total_width, scaled_max_height + title_height),
				color='white',
			)
			
			# Add each image
			x_offset = 0
			for i, img in enumerate(topk_images):
					composite.paste(img, (x_offset, title_height))  # Leave space at top for text
					x_offset += img.width
			# Scale font sizes based on scale_factor
			default_font_size_title = int(28 * scale_factor)
			default_font_size_score = int(24 * scale_factor)
			default_font_size_gt = int(22 * scale_factor)
			
			try:
					title_font = ImageFont.truetype("DejaVuSansMono-Bold.ttf", default_font_size_title)
					score_font = ImageFont.truetype("DejaVuSansMono.ttf", default_font_size_score)
					gt_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_gt)
			except IOError:
					try:
							title_font = ImageFont.truetype("NimbusSans-Bold.otf", default_font_size_title)
							score_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_score)
							gt_font = ImageFont.truetype("NimbusSans-Regular.otf", default_font_size_gt)
					except IOError:
							print("Warning: Could not load any fonts. Falling back to default font.")
							try:
									# Try this approach for PIL 9.0.0+
									title_font = ImageFont.load_default().font_variant(size=default_font_size_title)
									score_font = ImageFont.load_default().font_variant(size=default_font_size_score)
									gt_font = ImageFont.load_default().font_variant(size=default_font_size_gt)
							except AttributeError:
									# Fallback for older PIL versions
									title_font = score_font = gt_font = ImageFont.load_default()
			draw = ImageDraw.Draw(composite)
			
			# Add a subtle dividing line between title area and image
			draw.line([(0, title_height-2), (total_width, title_height-2)], fill="#DDDDDD", width=int(1 * scale_factor))
			
			# Add text for each image with proper vertical alignment
			x_offset = 0
			for i, (score, gt_label, img) in enumerate(zip(topk_scores, topk_ground_truth_labels, topk_images)):
					# Prepare the title text
					title_text = f"Top-{i+1}"
					score_text = f"Score: {score:.3f}"
					gt_text = f"GT: {gt_label.capitalize()}"
					
					# Calculate center position for this image section
					center_x = x_offset + img.width // 2
					
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
					top_y = int(1 * scale_factor)
					# draw.text(
					# 		(center_x - title_width//2, top_y),
					# 		title_text,
					# 		fill="black",
					# 		font=title_font
					# )
					
					# Draw "Score: X.X" text centered
					score_y = top_y + title_height_px + int(1 * scale_factor)
					draw.text(
							(center_x - score_width//2, score_y),
							score_text,
							fill="black",
							font=score_font
					)
					
					# For GT text, we need to handle longer text with possible line wrapping
					gt_y = score_y + score_height_px + int(30 * scale_factor)
					
					# Check if GT text is too wide for the image
					max_gt_width = img.width - int(20 * scale_factor)  # Leave scaled margin
					
					if gt_width > max_gt_width:
							# If text is too wide, use a smaller font
							gt_font_size = int(14 * scale_factor)  # Scaled smaller size
							try:
									smaller_gt_font = ImageFont.truetype("DejaVuSans.ttf", gt_font_size)
							except IOError:
									try:
											smaller_gt_font = ImageFont.truetype("NimbusSans-Regular.otf", gt_font_size)
									except IOError:
											smaller_gt_font = gt_font  # Fall back to original
							
							# Get new dimensions with smaller font
							if hasattr(smaller_gt_font, 'getbbox'):
									gt_bbox = smaller_gt_font.getbbox(gt_text)
									gt_width = gt_bbox[2] - gt_bbox[0]
							else:
									gt_width, _ = smaller_gt_font.getsize(gt_text)
							
							# If still too wide, truncate and add ellipsis
							if gt_width > max_gt_width:
									truncated = False
									while gt_width > max_gt_width and len(gt_text) > 10:
											gt_text = gt_text[:-1]
											truncated = True
											if hasattr(smaller_gt_font, 'getbbox'):
													gt_bbox = smaller_gt_font.getbbox(gt_text + "...")
													gt_width = gt_bbox[2] - gt_bbox[0]
											else:
													gt_width, _ = smaller_gt_font.getsize(gt_text + "...")
									
									if truncated:
											gt_text += "..."
							
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
					f't2i.png'
			)                
			composite.save(file_name, dpi=(dpi, dpi))
			print(f"Saved scaled composite image to: {file_name}")

def plot_comparison_metrics_split_table_annotation(
				dataset_name: str,
				pretrained_img2txt_dict: dict,
				pretrained_txt2img_dict: dict,
				finetuned_img2txt_dict: dict,
				finetuned_txt2img_dict: dict,
				model_name: str,
				finetune_strategies: list,
				results_dir: str,
				topK_values: list,
				figure_size=(7, 6),
				DPI: int = 200,
		):
		metrics = ["mP", "mAP", "Recall"]
		modes = ["Image-to-Text", "Text-to-Image"]
		all_model_architectures = [
				'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
				'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
		]
		
		# Validate model exists in dictionaries
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
		
		# Define colors and styles for the different strategies
		strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
		pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
		strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers
		
		# Key K points for table annotations
		key_k_values = [1, 10, 20]
		
		# Process each mode (Image-to-Text and Text-to-Image)
		for mode in modes:
				pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
				finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
				
				# Process each metric (mP, mAP, Recall)
				for metric in metrics:
						# Create figure with adjusted size
						fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
						
						# Create filename for output
						fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
						file_path = os.path.join(results_dir, fname)
						
						# Check if metric exists in pretrained dictionary
						if metric not in pretrained_dict.get(model_name, {}):
								print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
								continue
								
						# Get available k values across all dictionaries
						k_values = sorted(
								k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {})
						)
						
						# Validate k values across all strategies
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
						
						# Store lines for legend
						lines = []
								
						# Plot Pre-trained (dashed line)
						pretrained_vals = [pretrained_dict[model_name][metric].get(str(k), float('nan')) for k in k_values]
						pretrained_line, = ax.plot(
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
						lines.append(pretrained_line)
						
						# Plot each Fine-tuned strategy (solid lines, thicker, distinct markers)
						strategy_lines = {}
						for strategy in finetune_strategies:
								finetuned_vals = [finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for k in k_values]
								line, = ax.plot(
										k_values,
										finetuned_vals,
										label=f"{strategy.upper()}",
										color=strategy_colors[strategy], 
										linestyle='-', 
										marker=strategy_styles[strategy],
										linewidth=2.0, 
										markersize=5,
								)
								lines.append(line)
								strategy_lines[strategy] = finetuned_vals
						
						# Prepare data for table annotations at key K points
						tables_data = {}
						for k in key_k_values:
								if k in k_values:
										k_idx = k_values.index(k)
										pre_val = pretrained_vals[k_idx]
										
										# Collect improvements for all strategies at this K
										improvements = {}
										for strategy in finetune_strategies:
												if k_idx < len(strategy_lines[strategy]):
														ft_val = strategy_lines[strategy][k_idx]
														if pre_val != 0:
																imp_pct = (ft_val - pre_val) / pre_val * 100
																improvements[strategy] = (imp_pct, ft_val)
										
										# Sort strategies by improvement (descending)
										sorted_strategies = sorted(improvements.items(), key=lambda x: x[1][0], reverse=True)
										
										# Store for later use
										tables_data[k] = {
												'improvements': improvements,
												'sorted_strategies': sorted_strategies,
												'best_strategy': sorted_strategies[0][0] if sorted_strategies else None,
												'worst_strategy': sorted_strategies[-1][0] if sorted_strategies else None,
												'best_val': sorted_strategies[0][1][1] if sorted_strategies else None,
												'worst_val': sorted_strategies[-1][1][1] if sorted_strategies else None,
										}
						
						# Add table annotations for each key K point
						for k, data in tables_data.items():
								if not data['sorted_strategies']:
										continue
								
								# Create table text
								table_text = f"K={k}:\n"
								
								ranking_labels = ["1)", "2)", "3)"][:len(data['sorted_strategies'])]
								if len(data['sorted_strategies']) == 2:
										ranking_labels = ["1)", "2)"]  # Only 2 strategies
								
								for (strategy, (imp, _)), rank in zip(data['sorted_strategies'], ranking_labels):
										color_hex = strategy_colors[strategy]
										table_text += f"{rank} {strategy.upper()}: {imp:+.1f}%\n"  # Add each line to the table text
									
								if k == min(k_values):  # First K point (e.g., K=1)
										# Check if there's more space above best or below worst
										best_val = data['best_val']
										worst_val = data['worst_val']
										
										# Calculate available space
										space_above = 1.0 - best_val  # Space to top of plot
										space_below = worst_val - 0.0  # Space to bottom of plot
										
										# Position based on available space
										if space_above >= 0.2 or space_above > space_below:
												# Place above the highest point
												xy = (k, best_val)
												xytext = (10, 20)  # Offset to upper right
												va = 'bottom'
										else:
												# Place below the lowest point
												xy = (k, worst_val)
												xytext = (10, -20)  # Offset to lower right
												va = 'top'
								
								elif k == max(k_values):  # Last K point (e.g., K=20)
										# Similar logic but offset to the left
										best_val = data['best_val']
										worst_val = data['worst_val']
										
										space_above = 1.0 - best_val
										space_below = worst_val - 0.0
										
										if space_above >= 0.2 or space_above > space_below:
												xy = (k, best_val)
												xytext = (-10, 20)  # Offset to upper left
												va = 'bottom'
										else:
												xy = (k, worst_val)
												xytext = (-10, -20)  # Offset to lower left
												va = 'top'
								
								else:  # Middle K points (e.g., K=10)
										# Try to position in middle of plot if possible
										mid_y = (data['best_val'] + data['worst_val']) / 2
										xy = (k, mid_y)
										xytext = (0, 30 if mid_y < 0.5 else -30)  # Above if in lower half, below if in upper half
										va = 'bottom' if mid_y < 0.5 else 'top'
								
								# Add the annotation table
								ax.annotate(
										table_text,
										xy=xy,
										xytext=xytext,
										textcoords='offset points',
										fontsize=8,
										verticalalignment=va,
										horizontalalignment='center',
										bbox=dict(
												boxstyle="round,pad=0.4",
												facecolor='white',
												edgecolor='gray',
												alpha=0.9
										),
										zorder=10  # Ensure annotation is above other elements
								)
						
						# Format the plot
						ax.set_title(
								f"{metric}@K", 
								fontsize=10, 
								fontweight='bold',
						)
						ax.set_xlabel("K", fontsize=10, fontweight='bold')
						ax.set_xticks(k_values)
						ax.grid(True, linestyle='--', alpha=0.75)
						ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
						ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
						ax.set_ylim(-0.01, 1.01)
						ax.tick_params(axis='both', labelsize=7)

						# Set spine edge color to solid black
						for spine in ax.spines.values():
								spine.set_color('black')
								spine.set_linewidth(0.7)
								
						plt.tight_layout()
						plt.savefig(file_path, dpi=DPI, bbox_inches='tight')
						plt.close(fig)

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
		figure_size=(9, 8),
		DPI: int = 300,
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
	strategy_colors = {'full': '#0058a5', 'lora': '#f58320be', 'progressive': '#cc40df'}  # Blue, Orange, Purple
	pretrained_colors = {'ViT-B/32': '#745555', 'ViT-B/16': '#9467bd', 'ViT-L/14': '#e377c2', 'ViT-L/14@336px': '#696969'}
	strategy_styles = {'full': 's', 'lora': '^', 'progressive': 'd'}  # Unique markers
	
	for mode in modes:
		pretrained_dict = pretrained_img2txt_dict if mode == "Image-to-Text" else pretrained_txt2img_dict
		finetuned_dict = finetuned_img2txt_dict if mode == "Image-to-Text" else finetuned_txt2img_dict
		for metric in metrics:
			# Create figure with slightly adjusted size for better annotation spacing
			fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
			
			# Create filename for the output
			fname = f"{dataset_name}_{'_'.join(finetune_strategies)}_finetune_vs_pretrained_CLIP_{re.sub(r'[/@]', '-', model_name)}_{mode.replace('-', '_')}_{metric}_comparison.png"
			file_path = os.path.join(results_dir, fname)
			
			# Check if metric exists in pretrained dictionary
			if metric not in pretrained_dict.get(model_name, {}):
				print(f"WARNING: Metric {metric} not found in pretrained_{mode.lower().replace('-', '_')}_dict for {model_name}")
				continue
					
			# Get available k values across all dictionaries
			k_values = sorted(k for k in topK_values if str(k) in pretrained_dict.get(model_name, {}).get(metric, {}))
			
			# Validate k values across all strategies
			for strategy in finetune_strategies:
				if strategy not in finetuned_dict.get(model_name, {}) or metric not in finetuned_dict.get(model_name, {}).get(strategy, {}):
					print(f"WARNING: Metric {metric} not found in finetuned_{mode.lower().replace('-', '_')}_dict for {model_name}/{strategy}")
					k_values = []  # Reset if any strategy is missing
					break
				k_values = sorted(set(k_values) & set(int(k) for k in finetuned_dict.get(model_name, {}).get(strategy, {}).get(metric, {}).keys()))
					
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
				linewidth=2.0,
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
					linewidth=2.5,
					markersize=5,
				)
			
			# Analyze plot data to place annotations intelligently
			key_k_values = [1, 10, 20]  # These are the key points to annotate
			annotation_positions = {}    # To store planned annotation positions
			
			# First pass: gather data about values and improvements
			for k in key_k_values:
				if k in k_values:
					k_idx = k_values.index(k)
					pre_val = pretrained_vals[k_idx]
					finetuned_vals_at_k = {strategy: finetuned_dict[model_name][strategy][metric].get(str(k), float('nan')) for strategy in finetune_strategies}
					
					# Calculate improvements
					improvements = {}
					for strategy, val in finetuned_vals_at_k.items():
						if pre_val != 0:
							imp = (val - pre_val) / pre_val * 100
							improvements[strategy] = (imp, val)
					
					# Store data for this k
					annotation_positions[k] = {
						'best': max(improvements.items(), key=lambda x: x[1][0]),
						'worst': min(improvements.items(), key=lambda x: x[1][0]),
						'all_values': [v[1] for v in improvements.values()]
					}
			
			# Second pass: determine optimal annotation placement based on plot density
			for k, data in annotation_positions.items():
				best_strategy, (best_imp, best_val) = data['best']
				worst_strategy, (worst_imp, worst_val) = data['worst']
				
				# Find y positions of all lines at this k
				all_values = data['all_values']
				all_values.sort()  # Sort for easier gap analysis
				
				# For best annotation (typically placed above)
				best_text_color = '#016e2bff' if best_imp >= 0 else 'red'
				best_arrow_style = '<|-' if best_imp >= 0 else '-|>'
				
				# For worst annotation (typically placed below)
				worst_text_color = '#016e2bff' if worst_imp >= 0 else 'red'
				worst_arrow_style = '-|>' if worst_imp >= 0 else '<|-'
				
				# Calculate the overall range and spacing between values
				if len(all_values) > 1:  # More than one strategy
					value_range = max(all_values) - min(all_values)
					avg_gap = value_range / (len(all_values) - 1) if len(all_values) > 1 else 0.1
					
					# Check if annotation positioning needs adjustment
					if value_range < 0.15:  # Values are close together
						# Use more extreme offsets
						best_offset = (5, 20)  # Further right and higher up
						worst_offset = (5, -20)  # Further right and lower down
					else:
						# Regular offsets for well-separated values
						best_offset = (0, 20)
						worst_offset = (0, -20)
				else:
					# Default offsets when there's only one strategy
					best_offset = (0, 30)
					worst_offset = (0, -30)
				
				# Place best strategy annotation with adjusted position
				ax.annotate(
					f"{best_imp:+.1f}%",
					xy=(k, best_val),
					xytext=best_offset,
					textcoords='offset points',
					fontsize=12,
					fontweight='bold',
					color=best_text_color,
					bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1),
					arrowprops=dict(
						arrowstyle=best_arrow_style,
						color=best_text_color,
						shrinkA=0,
						shrinkB=3, # 
						alpha=0.8,
					)
				)
				
				# Place worst strategy annotation with adjusted position
				# Only annotate worst if it's different from best (avoids duplication)
				if worst_strategy != best_strategy:
					ax.annotate(
						f"{worst_imp:+.1f}%",
						xy=(k, worst_val),
						xytext=worst_offset,
						textcoords='offset points',
						fontsize=12,
						fontweight='bold',
						color=worst_text_color,
						bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.1),
						arrowprops=dict(
							arrowstyle=worst_arrow_style,
							color=worst_text_color,
							shrinkA=0,
							shrinkB=3,
							alpha=0.8,
						)
					)
			
			# Format the plot
			y_offset = 1.05
			title_bottom_y = y_offset + 0.02  # Calculate position below title
			legend_gap = 0.0  # Fixed gap between title and legend
			legend_y_pos = title_bottom_y - legend_gap
			ax.set_title(
				f"{metric}@K", 
				fontsize=13, 
				fontweight='bold', 
				y=y_offset,
			)
			ax.set_xlabel("K", fontsize=11, fontweight='bold')
			ax.set_xticks(k_values)
			ax.set_xticklabels(k_values, fontsize=15)
			# ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
			# ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=10)
			ax.grid(True, linestyle='--', alpha=0.4, color='black')
			y_max = max(max(pretrained_vals), max(finetuned_vals))
			y_min = min(min(pretrained_vals), min(finetuned_vals))
			padding = (y_max - y_min) * 0.2
			print(f"{metric}@K y_min: {y_min}, y_max: {y_max} padding: {padding}")
			# ax.set_ylim(min(0, y_min - padding), max(1, y_max + padding))
			ax.set_ylim(max(-0.02, y_min - padding), min(1.02, y_max + padding))
			# ax.set_ylim(-0.01, 1.01)
			ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=15)
			
			ax.legend(
				loc='upper center',
				bbox_to_anchor=(0.5, legend_y_pos),  # Position with fixed gap below title
				frameon=False,
				fontsize=12,
				facecolor='white',
				ncol=len(finetune_strategies) + 1,
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
	
	# fig.legend(
	# 	legend_handles,
	# 	legend_labels,
	# 	fontsize=11,
	# 	loc='upper center',
	# 	ncol=len(legend_labels),  # Adjust number of columns based on number of K values
	# 	bbox_to_anchor=(0.5, 0.96),
	# 	bbox_transform=fig.transFigure,
	# 	frameon=True,
	# 	shadow=True,
	# 	fancybox=True,
	# 	edgecolor='black',
	# 	facecolor='white',
	# )
	# plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
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

def perform_multilabel_eda(
		data_path: str,
		label_column: str = 'multimodal_labels',
		n_top_labels_plot: int = 30,
		n_top_labels_co_occurrence: int = 15,
		DPI: int = 200,
	):
	"""
	Enhanced Multi-label EDA with additional research-oriented analyses.
	
	New additions for research paper:
	1. Label diversity metrics (entropy, Gini coefficient)
	2. Hierarchical clustering of labels
	3. Label imbalance analysis
	4. Temporal analysis (if date column exists)
	5. Power law distribution fitting
	6. Agreement metrics between label sources
	7. Label stability analysis
	8. Multi-view consistency metrics
	9. Statistical significance tests
	"""
	print(f">> Enhanced Multi-label EDA for {data_path} (column: {label_column})")
	eda_st = time.time()
	dataset_dir = os.path.dirname(data_path)
	output_dir = os.path.join(dataset_dir, "outputs")
	os.makedirs(output_dir, exist_ok=True)

	if not os.path.exists(data_path):
			print(f"Error: Dataset not found at '{data_path}'. Please check the path.")
			return
	try:
			df = pd.read_csv(
					filepath_or_buffer=data_path,
					on_bad_lines='skip',
					low_memory=False,
			)
			print(f"Dataset {type(df)} loaded successfully. Shape: {df.shape}\n")
	except Exception as e:
			print(f"Error loading dataset from '{data_path}': {e}")
			return

	print("--- Basic DataFrame Info ---")
	df.info()
	print("\n--- Missing Values ---")
	print(df.isnull().sum())
	print("-" * 40 + "\n")

	label_columns_to_parse = [label_column, 'textual_based_labels', 'visual_based_labels']
	processed_dfs = {}
	df_for_parsing_and_filtering = df.copy()
	
	for col in label_columns_to_parse:
			print(f"--- Parsing '{col}' column ---")
			if col not in df_for_parsing_and_filtering.columns:
					print(f"Warning: Label column '{col}' not found in the DataFrame. Skipping.\n")
					continue
			current_col_series = df_for_parsing_and_filtering[col]
			first_valid_item = current_col_series.dropna().iloc[0] if not current_col_series.dropna().empty else None
			if first_valid_item is not None and isinstance(first_valid_item, str):
					print(f"Attempting to parse string representations in '{col}' column...")
					try:
							parsed_series = current_col_series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
							df_for_parsing_and_filtering[col] = parsed_series
							print(f"Successfully parsed string representations in '{col}' column.\n")
					except (ValueError, SyntaxError) as e:
							print(
									f"Error: Could not parse some string values in '{col}' column. "
									f"Ensure they are valid string representations of lists. Error: {e}"
							)
			elif first_valid_item is not None:
					print(f"'{col}' column's first valid item is of type {type(first_valid_item)}. Assuming list-like or compatible. No string parsing attempted.\n")
			else:
					print(f"'{col}' column is empty or all NaNs. No string parsing possible.\n")
			temp_series_for_filtering = df_for_parsing_and_filtering[col].copy()
			def clean_value(val):
					if isinstance(val, (list, tuple)):
							return val
					if pd.isna(val):
							return []
					return val
			temp_series_for_filtering = temp_series_for_filtering.apply(clean_value)
			valid_entries_mask = temp_series_for_filtering.apply(lambda x: isinstance(x, (list, tuple)) and len(x) > 0)
			df_filtered_for_this_col = df[valid_entries_mask].copy()
			if len(df_filtered_for_this_col) == 0:
					print(f"No samples with valid (non-empty list) labels found in '{col}' after parsing/filtering. Skipping further analysis for this column.\n")
					continue
			print(f"For column '{col}', retained {len(df_filtered_for_this_col)} rows with non-empty label lists out of {len(df)} original rows.")
			processed_dfs[col] = df_filtered_for_this_col.copy()
			processed_dfs[col][col] = df_for_parsing_and_filtering.loc[valid_entries_mask, col]

	if label_column not in processed_dfs:
			print(f"Main label column '{label_column}' could not be processed or is empty. Exiting EDA.")
			return
	df = processed_dfs[label_column].copy()

	all_individual_labels = [label for sublist in df[label_column] for label in sublist]
	unique_labels = sorted(list(set(all_individual_labels)))
	print(f"--- Multi-label Statistics (Main Column: {label_column}) ---")
	print(f"Total number of samples with valid '{label_column}': {len(df)}")
	print(f"Total number of unique labels across the dataset (from '{label_column}'): {len(unique_labels)}")
	print(f"Example unique labels:\n{unique_labels[:50]}")

	print(f"--- Label Cardinality Statistics (Main Column: {label_column}) ---")
	df['label_cardinality'] = df[label_column].apply(len)
	print(df['label_cardinality'].describe())

	plt.figure(figsize=(10, 6))
	sns.histplot(df['label_cardinality'], bins=range(1, int(df['label_cardinality'].max()) + 2), kde=False, color='skyblue')
	plt.title(f'Distribution of Label Cardinality (Labels per Sample for "{label_column}")')
	plt.xlabel('Number of Labels')
	plt.ylabel('Number of Samples')
	plt.xticks(range(1, int(df['label_cardinality'].max()) + 1))
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_label_cardinality_distribution.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()

	print(f"--- Top 20 Most Frequent Labels (Main Column: {label_column}) ---")
	label_counts = Counter(all_individual_labels)
	label_counts_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count']).sort_values(by='Count', ascending=False)
	print(label_counts_df.head(20))
	print(f"\n--- Bottom 20 Least Frequent Labels (Main Column: {label_column}) ---")
	print(label_counts_df.tail(20))
	singleton_labels = label_counts_df[label_counts_df['Count'] == 1]
	print(f"\nNumber of labels appearing only once: {len(singleton_labels)}")
	if len(unique_labels) > 0:
			print(f"Percentage of singleton labels: {len(singleton_labels) / len(unique_labels) * 100:.2f}%")
	else:
			print("No unique labels found to calculate percentage of singletons.")
	print(f"Example singleton labels (first 10): {singleton_labels['Label'].head(10).tolist()}")
	print("-" * 40 + "\n")
	
	plt.figure(figsize=(12, 8))
	sns.barplot(x='Count', y='Label', data=label_counts_df.head(n_top_labels_plot), palette='viridis')
	plt.title(f'Top {n_top_labels_plot} Most Frequent Labels (Main Column: "{label_column}")')
	plt.xlabel('Number of Samples')
	plt.ylabel('Label')
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_top_{n_top_labels_plot}_most_frequent_labels.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()

	plt.figure(figsize=(10, 6))
	sns.histplot(label_counts_df['Count'], bins=50, kde=False, color='coral')
	plt.title(f'Distribution of All Label Frequencies (Main Column: "{label_column}")')
	plt.xlabel('Label Frequency (Number of Samples)')
	plt.ylabel('Number of Labels (Log Scale)')
	plt.yscale('log')
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_all_label_frequencies_distribution.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()

	# ============================================================
	# NEW ADDITION 1: Power Law Analysis
	# ============================================================
	print("\n" + "="*100)
	print("--- POWER LAW ANALYSIS ---")
	freq_values = label_counts_df['Count'].values
	ranks = np.arange(1, len(freq_values) + 1)
	
	# Log-log plot to visualize power law
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	
	# Linear scale
	ax1.plot(ranks, freq_values, 'o', alpha=0.5, markersize=4)
	ax1.set_xlabel('Rank')
	ax1.set_ylabel('Frequency')
	ax1.set_title('Rank-Frequency Plot (Linear Scale)')
	ax1.grid(True, alpha=0.3)
	
	# Log-log scale
	ax2.loglog(ranks, freq_values, 'o', alpha=0.5, markersize=4)
	ax2.set_xlabel('Rank (log scale)')
	ax2.set_ylabel('Frequency (log scale)')
	ax2.set_title('Rank-Frequency Plot (Log-Log Scale)')
	ax2.grid(True, alpha=0.3)
	
	# Fit power law
	log_ranks = np.log(ranks)
	log_freqs = np.log(freq_values)
	coeffs = np.polyfit(log_ranks, log_freqs, 1)
	alpha_estimate = -coeffs[0]
	
	# Plot fitted line
	fitted_freqs = np.exp(coeffs[1]) * ranks**coeffs[0]
	ax2.plot(ranks, fitted_freqs, 'r-', linewidth=2, label=f'Power law fit (α≈{alpha_estimate:.2f})')
	ax2.legend()
	
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_power_law_analysis.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()
	
	print(f"Estimated power law exponent (α): {alpha_estimate:.3f}")
	print("Note: α ≈ 2 suggests Zipf's law, typical in natural language")

	# ============================================================
	# NEW ADDITION 2: Label Diversity Metrics
	# ============================================================
	print("\n" + "="*100)
	print("--- LABEL DIVERSITY METRICS ---")
	
	# Calculate Shannon entropy
	label_probs = label_counts_df['Count'].values / label_counts_df['Count'].sum()
	shannon_entropy = scipy.stats.entropy(label_probs, base=2)
	max_entropy = np.log2(len(unique_labels))
	normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
	
	# Calculate Gini coefficient
	sorted_counts = np.sort(label_counts_df['Count'].values)
	n = len(sorted_counts)
	gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
	
	# Calculate effective number of labels (Perplexity)
	effective_labels = 2 ** shannon_entropy
	
	print(f"Shannon Entropy: {shannon_entropy:.3f} bits")
	print(f"Maximum Possible Entropy: {max_entropy:.3f} bits")
	print(f"Normalized Entropy: {normalized_entropy:.3f} (1.0 = perfectly uniform)")
	print(f"Gini Coefficient: {gini:.3f} (0 = perfect equality, 1 = perfect inequality)")
	print(f"Effective Number of Labels: {effective_labels:.1f} (perplexity measure)")
	
	# Lorenz curve for label distribution
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	
	# Lorenz curve
	cumulative_counts = np.cumsum(sorted_counts)
	cumulative_proportions = cumulative_counts / cumulative_counts[-1]
	label_proportions = np.arange(1, n+1) / n
	
	ax1.plot(label_proportions, cumulative_proportions, 'b-', linewidth=2, label='Lorenz Curve')
	ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Equality')
	ax1.fill_between(label_proportions, cumulative_proportions, label_proportions, alpha=0.3)
	ax1.set_xlabel('Cumulative Proportion of Labels')
	ax1.set_ylabel('Cumulative Proportion of Samples')
	ax1.set_title(f'Lorenz Curve (Gini Coefficient: {gini:.3f})')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	# Diversity metrics summary
	metrics_data = {
		'Metric': ['Shannon Entropy', 'Normalized Entropy', 'Gini Coefficient', 
				   'Effective Labels', 'Unique Labels'],
		'Value': [shannon_entropy, normalized_entropy, gini, effective_labels, len(unique_labels)]
	}
	metrics_df = pd.DataFrame(metrics_data)
	ax2.axis('off')
	table = ax2.table(cellText=metrics_df.values, colLabels=metrics_df.columns,
					  cellLoc='left', loc='center', colWidths=[0.6, 0.4])
	table.auto_set_font_size(False)
	table.set_fontsize(12)
	table.scale(1, 2)
	ax2.set_title('Label Diversity Summary', pad=20, fontsize=14, fontweight='bold')
	
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_diversity_metrics.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()

	# ============================================================
	# NEW ADDITION 3: Label Imbalance Analysis
	# ============================================================
	print("\n" + "="*100)
	print("--- LABEL IMBALANCE ANALYSIS ---")
	
	# Calculate imbalance ratio
	max_freq = label_counts_df['Count'].max()
	min_freq = label_counts_df['Count'].min()
	imbalance_ratio = max_freq / min_freq
	
	# Calculate mean label frequency ratio (MeanLFR)
	mean_freq = label_counts_df['Count'].mean()
	median_freq = label_counts_df['Count'].median()
	
	# Identify severely imbalanced labels (< 1% of max frequency)
	threshold = max_freq * 0.01
	rare_labels = label_counts_df[label_counts_df['Count'] < threshold]
	
	print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
	print(f"Mean Label Frequency: {mean_freq:.2f}")
	print(f"Median Label Frequency: {median_freq:.2f}")
	print(f"Number of rare labels (< 1% of max): {len(rare_labels)} ({len(rare_labels)/len(unique_labels)*100:.1f}%)")
	
	# Create imbalance visualization
	fig, axes = plt.subplots(2, 2, figsize=(16, 12))
	
	# Cumulative distribution
	ax = axes[0, 0]
	sorted_freqs = label_counts_df['Count'].sort_values(ascending=False).values
	cumsum_freqs = np.cumsum(sorted_freqs)
	cumsum_pct = cumsum_freqs / cumsum_freqs[-1] * 100
	ax.plot(np.arange(1, len(cumsum_pct)+1), cumsum_pct, linewidth=2)
	ax.axhline(y=80, color='r', linestyle='--', label='80% threshold')
	ax.set_xlabel('Number of Labels (ranked by frequency)')
	ax.set_ylabel('Cumulative Percentage of Samples')
	ax.set_title('Cumulative Label Coverage')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# Find 80-20 point
	idx_80 = np.argmax(cumsum_pct >= 80)
	pct_labels_for_80 = (idx_80 + 1) / len(unique_labels) * 100
	ax.text(0.5, 0.5, f'{pct_labels_for_80:.1f}% of labels\ncover 80% of samples', 
			transform=ax.transAxes, ha='center', va='center',
			bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
	
	# Frequency binning
	ax = axes[0, 1]

	# Define the potential bin edges
	base_bins = [1, 5, 10, 50, 100, 500]

	# Filter the bins to be strictly less than max_freq to ensure monotonicity
	dynamic_bins = [b for b in base_bins if b < max_freq]
	# Add the final bin edge to include the maximum frequency value
	dynamic_bins.append(max_freq + 1)

	# Handle the edge case where max_freq is 1
	if not dynamic_bins:
		dynamic_bins = [1, 2]

	# Create corresponding labels dynamically
	dynamic_labels = []
	for i in range(len(dynamic_bins) - 1):
		start = dynamic_bins[i]
		end = dynamic_bins[i+1] - 1
		if i == len(dynamic_bins) - 2: # This is the last bin
			label = f'{start}+'
		else:
			label = f'{start}-{end}'
		dynamic_labels.append(label)

	label_counts_df['freq_bin'] = pd.cut(label_counts_df['Count'], bins=dynamic_bins, labels=dynamic_labels, right=False)
	freq_bin_counts = label_counts_df['freq_bin'].value_counts().sort_index()

	ax.bar(range(len(freq_bin_counts)), freq_bin_counts.values, color='coral')
	ax.set_xticks(range(len(freq_bin_counts)))
	ax.set_xticklabels(freq_bin_counts.index, rotation=45)
	ax.set_xlabel('Frequency Range')
	ax.set_ylabel('Number of Labels')
	ax.set_title('Label Distribution by Frequency Bins')
	ax.grid(axis='y', alpha=0.3)
	
	# Box plot of frequencies
	ax = axes[1, 0]
	ax.boxplot([label_counts_df['Count'].values], vert=False)
	ax.set_xlabel('Label Frequency')
	ax.set_title('Box Plot of Label Frequencies')
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3)
	
	# Head vs Tail distribution
	ax = axes[1, 1]
	n_head = max(20, int(len(unique_labels) * 0.1))  # Top 10% or at least 20
	head_coverage = label_counts_df.head(n_head)['Count'].sum() / label_counts_df['Count'].sum() * 100
	tail_coverage = 100 - head_coverage
	ax.pie([head_coverage, tail_coverage], labels=[f'Top {n_head} labels', f'Remaining {len(unique_labels)-n_head} labels'],
		   autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
	ax.set_title('Sample Coverage: Head vs Tail Labels')
	
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"{label_column}_imbalance_analysis.png"),
		dpi=DPI,
		bbox_inches='tight',
	)
	plt.close()

	# Original code continues...
	print("\n" + "="*100)
	print("--- Unique Label Set Combinations ---")
	label_sets = df[label_column].apply(lambda x: tuple(sorted(x)))
	unique_label_sets = Counter(label_sets)
	unique_label_sets_df = pd.DataFrame(unique_label_sets.items(), columns=['Label Set', 'Count']).sort_values(by='Count', ascending=False)
	print(f"Total number of unique label combinations: {len(unique_label_sets)}")
	print(f"Top 10 Most Frequent Label Combinations (Main Column: {label_column}):")
	print(unique_label_sets_df.head(10))
	
	if len(unique_label_sets) > 0:
			plt.figure(figsize=(12, 8))
			top_n_combinations = unique_label_sets_df.head(min(20, len(unique_label_sets))).copy()
			top_n_combinations['Label Set String'] = top_n_combinations['Label Set'].apply(lambda x: ', '.join(x))
			sns.barplot(x='Count', y='Label Set String', data=top_n_combinations, palette='magma')
			plt.title(f'Top {len(top_n_combinations)} Most Frequent Unique Label Combinations')
			plt.xlabel('Number of Samples')
			plt.ylabel('Label Combination')
			plt.tight_layout()
			plt.savefig(
					fname=os.path.join(output_dir, f"{label_column}_top_unique_label_combinations.png"),
					dpi=DPI,
					bbox_inches='tight',
			)
			plt.close()
	
	print("="*100)
	
	# ============================================================
	# NEW ADDITION 4: Hierarchical Clustering of Labels
	# ============================================================
	print("\n--- HIERARCHICAL CLUSTERING OF LABELS (Top Labels) ---")
	if n_top_labels_co_occurrence > len(unique_labels):
		print(
			f"Warning: n_top_labels_co_occurrence ({n_top_labels_co_occurrence}) is greater than "
			f"the total unique labels ({len(unique_labels)}). Adjusting to total unique labels."
		)
		n_top_labels_co_occurrence = len(unique_labels)
	
	print(f"Top {n_top_labels_co_occurrence} labels for correlation matrix:\n{label_counts_df['Label'].head(n_top_labels_co_occurrence).tolist()}")
	
	if n_top_labels_co_occurrence >= 2:
		print(f"Binarizing labels for correlation matrix (classes: {len(unique_labels)} sparse matrix) ...")
		mlb = MultiLabelBinarizer(classes=unique_labels, sparse_output=True)
		y_binarized = mlb.fit_transform(df[label_column])
		labels_in_order = mlb.classes_
		
		top_labels_for_correlation = label_counts_df['Label'].head(n_top_labels_co_occurrence).tolist()
		top_label_indices = [list(labels_in_order).index(lab) for lab in top_labels_for_correlation]
		
		y_subset = y_binarized[:, top_label_indices].toarray()
		
		# Calculate Jaccard Similarity Matrix
		jaccard_matrix = np.zeros((n_top_labels_co_occurrence, n_top_labels_co_occurrence))
		for i in range(n_top_labels_co_occurrence):
			for j in range(n_top_labels_co_occurrence):
				if i == j:
					jaccard_matrix[i, j] = 1.0
				else:
					col_i = y_subset[:, i]
					col_j = y_subset[:, j]
					intersection = np.sum(col_i & col_j)
					union = np.sum(col_i | col_j)
					jaccard_matrix[i, j] = intersection / union if union != 0 else 0.0
		
		jaccard_df = pd.DataFrame(jaccard_matrix, index=top_labels_for_correlation, columns=top_labels_for_correlation)
		
		# Create figure with heatmap and dendrogram
		fig = plt.figure(figsize=(22, 14))
		
		# Compute hierarchical clustering
		distance_matrix = 1 - jaccard_matrix
		np.fill_diagonal(distance_matrix, 0)
		condensed_dist = scipy.spatial.distance.squareform(distance_matrix)
		linkage_matrix = scipy.cluster.hierarchy.linkage(condensed_dist, method='average')
		
		# Plot dendrogram
		ax_dendro = fig.add_subplot(2, 2, 1)
		dendrogram = scipy.cluster.hierarchy.dendrogram(
			linkage_matrix, 
			labels=top_labels_for_correlation, 
			orientation='top', 
			ax=ax_dendro, 
			leaf_font_size=10
		)
		ax_dendro.set_title('Hierarchical Clustering Dendrogram')
		ax_dendro.set_xlabel('Labels')
		ax_dendro.set_ylabel('Distance (1 - Jaccard Similarity)')
		ax_dendro.tick_params(axis='x', rotation=90)
		
		# Plot heatmap
		ax_heatmap = fig.add_subplot(2, 2, 2)
		sns.heatmap(
			jaccard_df, 
			annot=True, 
			fmt=".2f", 
			cmap='Blues', 
			linewidths=.5, 
			linecolor='gray',
			cbar_kws={'label': 'Jaccard Similarity'},
			ax=ax_heatmap
		)
		ax_heatmap.set_title(f'Jaccard Similarity Matrix')
		
		# Co-occurrence matrix (absolute counts)
		cooccurrence_matrix = np.zeros((n_top_labels_co_occurrence, n_top_labels_co_occurrence))
		for i in range(n_top_labels_co_occurrence):
			for j in range(n_top_labels_co_occurrence):
				cooccurrence_matrix[i, j] = np.sum(y_subset[:, i] & y_subset[:, j])
		
		cooccurrence_df = pd.DataFrame(cooccurrence_matrix, 
									   index=top_labels_for_correlation, 
									   columns=top_labels_for_correlation)
		
		ax_cooc = fig.add_subplot(2, 2, 3)
		sns.heatmap(
			cooccurrence_df, 
			annot=True, 
			fmt=".0f", 
			cmap='Greens', 
			linewidths=.5, 
			linecolor='gray',
			cbar_kws={'label': 'Co-occurrence Count'},
			ax=ax_cooc
		)
		ax_cooc.set_title(f'Label Co-occurrence Matrix (Absolute Counts)')
		
		# Network-style visualization of strong co-occurrences
		ax_network = fig.add_subplot(2, 2, 4)
		threshold = 0.3  # Only show edges with Jaccard > 0.3
		
		# Create adjacency list for strong connections
		strong_connections = []
		for i in range(n_top_labels_co_occurrence):
			for j in range(i+1, n_top_labels_co_occurrence):
				if jaccard_matrix[i, j] > threshold:
					strong_connections.append((i, j, jaccard_matrix[i, j]))
		
		# Simple circular layout
		angles = np.linspace(0, 2*np.pi, n_top_labels_co_occurrence, endpoint=False)
		x = np.cos(angles)
		y = np.sin(angles)
		
		# Draw edges
		for i, j, weight in strong_connections:
			ax_network.plot([x[i], x[j]], [y[i], y[j]], 'gray', alpha=weight, linewidth=weight*3)
		
		# Draw nodes
		ax_network.scatter(x, y, s=200, c='lightblue', edgecolors='black', zorder=10)
		
		# Add labels
		for idx, label in enumerate(top_labels_for_correlation):
			ax_network.text(x[idx]*1.15, y[idx]*1.15, label, 
						   ha='center', va='center', fontsize=8)
		
		ax_network.set_xlim(-1.5, 1.5)
		ax_network.set_ylim(-1.5, 1.5)
		ax_network.axis('off')
		ax_network.set_title(f'Label Co-occurrence Network (Jaccard > {threshold})')
		
		plt.tight_layout()
		plt.savefig(
			fname=os.path.join(output_dir, f"{label_column}_clustering_and_cooccurrence.png"),
			dpi=DPI,
			bbox_inches='tight',
		)
		plt.close()
	else:
		print("Not enough unique labels to display correlation analyses (need at least 2).")
	
	print("="*100)

	# ============================================================
	# NEW ADDITION 5: Multi-source Agreement Analysis
	# ============================================================
	print("\n--- MULTI-SOURCE LABEL AGREEMENT ANALYSIS ---")
	source_cols = {
		'textual_based': 'textual_based_labels',
		'visual_based': 'visual_based_labels',
		'multimodal': 'multimodal_labels'
	}
	
	unique_labels_by_source = {}
	for key, col_name in source_cols.items():
		if col_name in processed_dfs:
			current_all_labels = [label for sublist in processed_dfs[col_name][col_name] for label in sublist]
			unique_labels_by_source[key] = set(current_all_labels)
			print(f"Unique labels in '{col_name}': {len(unique_labels_by_source[key])}")
		else:
			unique_labels_by_source[key] = set()
			print(f"'{col_name}' was not processed or is empty.")
	
	text_set = unique_labels_by_source.get('textual_based', set())
	visual_set = unique_labels_by_source.get('visual_based', set())
	multimodal_set = unique_labels_by_source.get('multimodal', set())
	
	# Calculate overlaps
	all_three_overlap = text_set & visual_set & multimodal_set
	text_visual_overlap = text_set & visual_set
	text_multimodal_overlap = text_set & multimodal_set
	visual_multimodal_overlap = visual_set & multimodal_set
	
	print(f"\nCommon labels across all three sources: {len(all_three_overlap)}")
	print(f"Common labels between Textual and Visual: {len(text_visual_overlap)}")
	print(f"Common labels between Textual and Multimodal: {len(text_multimodal_overlap)}")
	print(f"Common labels between Visual and Multimodal: {len(visual_multimodal_overlap)}")
	
	unique_to_text = text_set - (visual_set | multimodal_set)
	unique_to_visual = visual_set - (text_set | multimodal_set)
	unique_to_multimodal = multimodal_set - (text_set | visual_set)
	
	print(f"\nLabels unique to textual_based_labels: {len(unique_to_text)}")
	print(f"Labels unique to visual_based_labels: {len(unique_to_visual)}")
	print(f"Labels unique to multimodal_labels: {len(unique_to_multimodal)}")
	
	# Calculate agreement metrics if we have sample-level data
	if all(col in processed_dfs for col in source_cols.values()):
		print("\n--- Sample-Level Agreement Metrics ---")
		
		# Find common samples across all three sources
		common_indices = set(processed_dfs['textual_based_labels'].index) & \
						 set(processed_dfs['visual_based_labels'].index) & \
						 set(processed_dfs['multimodal_labels'].index)
		
		if len(common_indices) > 0:
			agreement_scores = []
			perfect_agreement = 0
			partial_agreement = 0
			no_agreement = 0
			
			for idx in common_indices:
				text_labels = set(processed_dfs['textual_based_labels'].loc[idx, 'textual_based_labels'])
				visual_labels = set(processed_dfs['visual_based_labels'].loc[idx, 'visual_based_labels'])
				multi_labels = set(processed_dfs['multimodal_labels'].loc[idx, 'multimodal_labels'])
				
				# Calculate pairwise Jaccard similarities
				tv_jaccard = len(text_labels & visual_labels) / len(text_labels | visual_labels) if len(text_labels | visual_labels) > 0 else 0
				tm_jaccard = len(text_labels & multi_labels) / len(text_labels | multi_labels) if len(text_labels | multi_labels) > 0 else 0
				vm_jaccard = len(visual_labels & multi_labels) / len(visual_labels | multi_labels) if len(visual_labels | multi_labels) > 0 else 0
				
				avg_jaccard = (tv_jaccard + tm_jaccard + vm_jaccard) / 3
				agreement_scores.append(avg_jaccard)
				
				if text_labels == visual_labels == multi_labels:
					perfect_agreement += 1
				elif len(text_labels & visual_labels & multi_labels) > 0:
					partial_agreement += 1
				else:
					no_agreement += 1
			
			print(f"Samples with common labels across all sources: {len(common_indices)}")
			print(f"Perfect agreement (all labels match): {perfect_agreement} ({perfect_agreement/len(common_indices)*100:.1f}%)")
			print(f"Partial agreement (some overlap): {partial_agreement} ({partial_agreement/len(common_indices)*100:.1f}%)")
			print(f"No agreement (no overlap): {no_agreement} ({no_agreement/len(common_indices)*100:.1f}%)")
			print(f"Mean Jaccard Agreement: {np.mean(agreement_scores):.3f}")
			print(f"Median Jaccard Agreement: {np.median(agreement_scores):.3f}")
			
			# Visualize agreement distribution
			fig, axes = plt.subplots(2, 2, figsize=(16, 12))
			
			# Agreement score distribution
			ax = axes[0, 0]
			ax.hist(agreement_scores, bins=50, color='skyblue', edgecolor='black')
			ax.axvline(np.mean(agreement_scores), color='red', linestyle='--', 
					   label=f'Mean: {np.mean(agreement_scores):.3f}')
			ax.set_xlabel('Average Jaccard Agreement Score')
			ax.set_ylabel('Number of Samples')
			ax.set_title('Distribution of Multi-Source Agreement Scores')
			ax.legend()
			ax.grid(True, alpha=0.3)
			
			# Agreement categories
			ax = axes[0, 1]
			categories = ['Perfect\nAgreement', 'Partial\nAgreement', 'No\nAgreement']
			counts = [perfect_agreement, partial_agreement, no_agreement]
			colors = ['green', 'orange', 'red']
			ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
			ax.set_ylabel('Number of Samples')
			ax.set_title('Sample-Level Agreement Categories')
			for i, (cat, count) in enumerate(zip(categories, counts)):
				ax.text(i, count, f'{count}\n({count/len(common_indices)*100:.1f}%)', 
						ha='center', va='bottom')
			ax.grid(axis='y', alpha=0.3)
			
			# Venn diagram data
			ax = axes[1, 0]
			overlap_data = {
				'Category': [
					'All Three', 
					'Text & Visual only', 
					'Text & Multi only', 
					'Visual & Multi only',
					'Text only', 
					'Visual only', 
					'Multi only'
				],
				'Count': [
					len(all_three_overlap),
					len((text_set & visual_set) - multimodal_set),
					len((text_set & multimodal_set) - visual_set),
					len((visual_set & multimodal_set) - text_set),
					len(unique_to_text),
					len(unique_to_visual),
					len(unique_to_multimodal)
				]
			}
			overlap_df = pd.DataFrame(overlap_data)
			sns.barplot(x='Count', y='Category', data=overlap_df, palette='pastel', ax=ax)
			ax.set_title('Label Set Overlaps Across Sources')
			ax.set_xlabel('Number of Unique Labels')
			ax.grid(axis='x', alpha=0.3)
			
			# Source coverage comparison
			ax = axes[1, 1]
			source_coverage = {
				'Source': ['Textual', 'Visual', 'Multimodal'],
				'Unique Labels': [len(text_set), len(visual_set), len(multimodal_set)],
				'Coverage %': [
					len(text_set) / len(text_set | visual_set | multimodal_set) * 100,
					len(visual_set) / len(text_set | visual_set | multimodal_set) * 100,
					len(multimodal_set) / len(text_set | visual_set | multimodal_set) * 100
				]
			}
			coverage_df = pd.DataFrame(source_coverage)
			x_pos = np.arange(len(coverage_df))
			ax.bar(x_pos, coverage_df['Unique Labels'], alpha=0.7, edgecolor='black')
			ax.set_xticks(x_pos)
			ax.set_xticklabels(coverage_df['Source'])
			ax.set_ylabel('Number of Unique Labels')
			ax.set_title('Label Space Coverage by Source')
			for i, (labels, pct) in enumerate(zip(coverage_df['Unique Labels'], coverage_df['Coverage %'])):
				ax.text(i, labels, f'{labels}\n({pct:.1f}%)', ha='center', va='bottom')
			ax.grid(axis='y', alpha=0.3)
			
			plt.tight_layout()
			plt.savefig(
				fname=os.path.join(output_dir, f"{label_column}_agreement_analysis.png"),
				dpi=DPI,
				bbox_inches='tight',
			)
			plt.close()
	
	# ============================================================
	# NEW ADDITION 6: Temporal Analysis (if date column exists)
	# ============================================================
	if 'doc_date' in df.columns or 'date' in df.columns:
		print("\n" + "="*100)
		print("--- TEMPORAL ANALYSIS ---")
		
		date_col = 'doc_date' if 'doc_date' in df.columns else 'date'
		
		try:
			df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
			df_with_dates = df[df[date_col].notna()].copy()
			
			if len(df_with_dates) > 0:
				df_with_dates['year'] = df_with_dates[date_col].dt.year
				
				# Labels over time
				yearly_label_counts = df_with_dates.groupby('year')[label_column].apply(
					lambda x: len(set([label for sublist in x for label in sublist]))
				).reset_index()
				yearly_label_counts.columns = ['Year', 'Unique Labels']
				
				yearly_sample_counts = df_with_dates.groupby('year').size().reset_index()
				yearly_sample_counts.columns = ['Year', 'Sample Count']
				
				yearly_avg_cardinality = df_with_dates.groupby('year')['label_cardinality'].mean().reset_index()
				yearly_avg_cardinality.columns = ['Year', 'Avg Cardinality']
				
				fig, axes = plt.subplots(2, 2, figsize=(16, 12))
				
				# Unique labels over time
				ax = axes[0, 0]
				ax.plot(yearly_label_counts['Year'], yearly_label_counts['Unique Labels'], 
						marker='o', linewidth=2, markersize=8)
				ax.set_xlabel('Year')
				ax.set_ylabel('Number of Unique Labels')
				ax.set_title('Unique Label Growth Over Time')
				ax.grid(True, alpha=0.3)
				
				# Sample count over time
				ax = axes[0, 1]
				ax.bar(yearly_sample_counts['Year'], yearly_sample_counts['Sample Count'], 
					   color='coral', edgecolor='black')
				ax.set_xlabel('Year')
				ax.set_ylabel('Number of Samples')
				ax.set_title('Sample Distribution Over Time')
				ax.grid(axis='y', alpha=0.3)
				
				# Average cardinality over time
				ax = axes[1, 0]
				ax.plot(yearly_avg_cardinality['Year'], yearly_avg_cardinality['Avg Cardinality'], 
						marker='s', linewidth=2, markersize=8, color='green')
				ax.set_xlabel('Year')
				ax.set_ylabel('Average Labels per Sample')
				ax.set_title('Label Cardinality Trend Over Time')
				ax.grid(True, alpha=0.3)
				
				# Top labels by time period
				ax = axes[1, 1]
				# Split into early and late periods
				median_year = df_with_dates['year'].median()
				early_period = df_with_dates[df_with_dates['year'] <= median_year]
				late_period = df_with_dates[df_with_dates['year'] > median_year]
				
				early_labels = Counter([label for sublist in early_period[label_column] for label in sublist])
				late_labels = Counter([label for sublist in late_period[label_column] for label in sublist])
				
				top_n = 10
				early_top = set(dict(early_labels.most_common(top_n)).keys())
				late_top = set(dict(late_labels.most_common(top_n)).keys())
				
				persistent = len(early_top & late_top)
				new_labels = len(late_top - early_top)
				disappeared = len(early_top - late_top)
				
				categories = ['Persistent', 'New', 'Disappeared']
				values = [persistent, new_labels, disappeared]
				colors = ['green', 'blue', 'red']
				ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
				ax.set_ylabel('Number of Labels')
				ax.set_title(f'Top {top_n} Label Stability\n(Early: ≤{median_year:.0f} vs Late: >{median_year:.0f})')
				for i, (cat, val) in enumerate(zip(categories, values)):
					ax.text(i, val, f'{val}', ha='center', va='bottom')
				ax.grid(axis='y', alpha=0.3)
				
				plt.tight_layout()
				plt.savefig(
					fname=os.path.join(output_dir, f"{label_column}_temporal_analysis.png"),
					dpi=DPI,
					bbox_inches='tight',
				)
				plt.close()
				
				print(f"Temporal analysis completed for {len(df_with_dates)} samples with valid dates")
				print(f"Date range: {df_with_dates[date_col].min()} to {df_with_dates[date_col].max()}")
			else:
				print("No valid dates found for temporal analysis")
		except Exception as e:
			print(f"Error in temporal analysis: {e}")
	
	# ============================================================
	# Summary Statistics Table
	# ============================================================
	print("\n" + "="*100)
	print("--- COMPREHENSIVE SUMMARY STATISTICS ---")
	
	summary_stats = {
		'Metric': [
			'Total Samples',
			'Unique Labels',
			'Singleton Labels',
			'Unique Label Combinations',
			'Mean Label Cardinality',
			'Median Label Cardinality',
			'Max Label Cardinality',
			'Shannon Entropy',
			'Normalized Entropy',
			'Gini Coefficient',
			'Imbalance Ratio',
			'Power Law Exponent (α)',
			'Effective # of Labels'
		],
		'Value': [
			len(df),
			len(unique_labels),
			len(singleton_labels),
			len(unique_label_sets),
			f"{df['label_cardinality'].mean():.2f}",
			f"{df['label_cardinality'].median():.1f}",
			int(df['label_cardinality'].max()),
			f"{shannon_entropy:.3f}",
			f"{normalized_entropy:.3f}",
			f"{gini:.3f}",
			f"{imbalance_ratio:.2f}",
			f"{alpha_estimate:.3f}",
			f"{effective_labels:.1f}"
		]
	}
	
	summary_df = pd.DataFrame(summary_stats)
	print(summary_df.to_string(index=False))
	
	# Save summary to CSV
	summary_df.to_csv(
		os.path.join(output_dir, f"{label_column}_summary_statistics.csv"),
		index=False
	)
	
	print("\n" + "="*100)
	print(f"EDA Elapsed_t: {time.time()-eda_st:.3f} sec".center(100, "-"))
	print(f"All outputs saved to: {output_dir}")
	print("="*100)

def plot_label_distribution_pie_chart(
		df: pd.DataFrame = None,
		fpth: str = "label_distribution_pie_chart.png",
		figure_size: tuple = (12, 7),
		DPI: int = 200,
		dataset_name: str = "EUROPEANA_1900-01-01_1970-12-31",
	):
	# Count labels and sort by count (descending)
	label_counts = df['label'].value_counts().sort_values(ascending=False)
	labels = label_counts.index
	total_samples = label_counts.sum()
	unique_labels = len(labels)
	# Group small categories into "Other"
	threshold = 0.01  # 1% threshold
	other_count = label_counts[label_counts / total_samples < threshold].sum()
	main_counts = label_counts[label_counts / total_samples >= threshold]
	if other_count > 0:
		main_counts['Other'] = other_count
	labels = main_counts.index
	label_counts = main_counts
	# Create figure with vertical layout
	fig = plt.figure(figsize=figure_size)
	gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1])
	ax_pie = fig.add_subplot(gs[0])
	ax_legend = fig.add_subplot(gs[1])
	# Use a colorblind-friendly categorical colormap
	colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
	# Explode larger wedges
	explode = [0.1 if i < 3 else 0 for i in range(len(labels))]
	# Create pie chart
	wedges, texts, autotexts = ax_pie.pie(
			label_counts.values,
			labels=[''] * len(labels),
			colors=colors,
			autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
			startangle=0,
			explode=explode,
			wedgeprops={
					'edgecolor': 'black',
					'linewidth': 0.7,
					'alpha': 0.8,
			}
	)
	# Adjust percentage label contrast and position
	for i, autotext in enumerate(autotexts):
			if autotext.get_text():
					wedge_color = wedges[i].get_facecolor()
					luminance = 0.299 * wedge_color[0] + 0.587 * wedge_color[1] + 0.114 * wedge_color[2]
					autotext.set_color('white' if luminance < 0.5 else 'black')
					if label_counts.values[i] / total_samples < 0.1:
							autotext.set_position((autotext.get_position()[0] * 1.2, autotext.get_position()[1] * 1.2))
					autotext.set_fontsize(18)
					autotext.set_weight('bold')  # Make font bold

	# Turn off axis for legend subplot
	ax_legend.axis('off')
	# Create truncated legend
	if len(labels) > 6:
		selected_wedges = wedges[:3] + [None] + wedges[-3:]
		legend_labels_full = [
			f"{label} ({count:,}, {count/total_samples*100:.1f}%)"
			for label, count in label_counts.items()
		]
		omitted_count = len(labels) - 6
		# selected_labels = legend_labels_full[:3] + [f'... ({omitted_count} categories omitted)'] + legend_labels_full[-3:]
		selected_labels = legend_labels_full[:3] + [f'...'] + legend_labels_full[-3:]
		dummy_artist = plt.Rectangle((0, 0), 1, 1, fc='none', fill=False, edgecolor='none', linewidth=0)
		selected_wedges[3] = dummy_artist
	else:
		selected_wedges = wedges
		selected_labels = [
			f"{label} ({count:,}, {count/total_samples*100:.1f}%)"
			for label, count in label_counts.items()
		]
	# Create legend
	legend = ax_legend.legend(
		selected_wedges,
		selected_labels,
		loc='center',
		bbox_to_anchor=(0.5, 0.5),
		fontsize=16,
		title=f"Total samples: {total_samples:,} (Unique Labels: {unique_labels})",
		title_fontsize=15,
		fancybox=True,
		shadow=True,
		edgecolor='black',
		facecolor='white',
		ncol=1,
		labelspacing=1.2,
		labelcolor='black',
	)	
	for text in legend.get_texts():
		text.set_fontweight('bold')
	ax_pie.axis('equal')
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

	# Optional bar chart for top 10 categories
	plt.figure(figsize=(15, 5))
	top_n = 10
	top_counts = label_counts[:top_n]
	if len(label_counts) > top_n:
			top_counts['Other'] = label_counts[top_n:].sum()
	colors = plt.cm.tab20c(np.linspace(0, 1, len(top_counts)))
	plt.bar(top_counts.index, top_counts.values, color=colors)
	plt.yscale('log')  # Log scale for visibility of small categories
	plt.xticks(rotation=45, ha='right')
	plt.ylabel('Sample Count (Log Scale)')
	plt.title(f"Top {top_n} Label Distribution for {dataset_name} Dataset")
	plt.tight_layout()
	plt.savefig(
		fname=fpth.replace('.png', '_bar.png'), 
		dpi=DPI, 
		bbox_inches='tight',
	)
	plt.close()

def plot_grouped_bar_chart(
		merged_df: pd.DataFrame,
		DPI: int = 200,
		FIGURE_SIZE: tuple = (12, 8),
		fname: str = "grouped_bar_chart.png",
	):
	
	calling_frame = inspect.currentframe().f_back
	dfs_length = len(calling_frame.f_locals.get('dfs', []))

	dataset_unique_label_counts = merged_df.groupby('dataset')['label'].nunique()
	print(dataset_unique_label_counts)

	label_counts = merged_df['label'].value_counts()
	# print(label_counts.tail(25))

	plt.figure(figsize=FIGURE_SIZE)
	sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	new_labels = [f"{label} | ({dataset_unique_label_counts[label]})" for label in labels]
	ax.legend(handles, new_labels, loc="best", fontsize=10, title="Dataset | (Unique Label Count)")
	plt.title(f'Grouped Bar Chart for total of {label_counts.shape[0]} Labels Frequency for {dfs_length} Datasets')
	plt.xticks(fontsize=9, rotation=90, ha='right')
	plt.yticks(fontsize=9, rotation=90, va='center')
	plt.xlabel('Label')
	plt.ylabel('Frequency')
	plt.grid(axis='y', alpha=0.7, linestyle='--')
	plt.tight_layout()
	plt.savefig(
		fname=fname,
		dpi=DPI,
		bbox_inches='tight'
	)
	plt.close()

def plot_train_val_label_distribution(
		train_df: pd.DataFrame,
		val_df: pd.DataFrame,
		dataset_name: str,
		VAL_SPLIT_PCT: float,
		FIGURE_SIZE: tuple = (12, 8),
		DPI: int = 200,
		fname: str = "simple_random_split_stratified_label_distribution_train_val.png",
		label_column: str = 'label',
	):
	# Visualize label distribution in training and validation sets
	plt.figure(figsize=FIGURE_SIZE)
	train_df[label_column].value_counts().plot(kind='bar', color='blue', alpha=0.6, label=f'Train {1-VAL_SPLIT_PCT}')
	val_df[label_column].value_counts().plot(kind='bar', color='red', alpha=0.9, label=f'Validation {VAL_SPLIT_PCT}')
	plt.title(
		f'{dataset_name} Stratified Label Distribution (Total samples: {train_df.shape[0]+val_df.shape[0]})\n'
		f'Train: {train_df.shape[0]} | Validation: {val_df.shape[0]}', 
		fontsize=9, 
		fontweight='bold',
	)
	plt.xlabel('Label')
	plt.ylabel('Frequency')
	plt.yticks(rotation=90, fontsize=9, va='center')
	plt.legend(
		loc='best', 
		ncol=2, 
		frameon=True,
		fancybox=True,
		shadow=True,
		edgecolor='black',
		facecolor='white', 
		fontsize=10,
	)
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(
		fname=fname,
		dpi=DPI,
		bbox_inches='tight'
	)	
	plt.close()

def plot_year_distribution(
		df: pd.DataFrame,
		dname: str,
		fpth: str,
		BINs: int = 50,
		FIGURE_SIZE: tuple = (18, 8),
		DPI: int = 200,
	):

	# Convert 'doc_date' to datetime and handle invalid entries
	df['doc_date'] = pd.to_datetime(df['doc_date'], errors='coerce')
	
	# Extract valid dates (non-NaN)
	valid_dates = df['doc_date'].dropna()
	
	# Handle edge case: no valid dates
	if valid_dates.empty:
		plt.figure(figsize=FIGURE_SIZE)
		plt.text(0.5, 0.5, "No valid dates available for plotting", ha='center', va='center', fontsize=12)
		plt.title(f'{dname} Temporal Distribution - No Data')
		plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
		plt.close()
		return
	
	# Compute start and end dates from data
	start_date = valid_dates.min().strftime('%Y-%m-%d')
	end_date = valid_dates.max().strftime('%Y-%m-%d')
	start_year = valid_dates.min().year
	end_year = valid_dates.max().year
	print(f"start_year: {start_year} | end_year: {end_year}")
	
	# Extract the year from the 'doc_date' column (now as integer)
	df['year'] = df['doc_date'].dt.year  # This will have NaN where doc_date is NaT
	
	# Filter out None values (though dt.year gives NaN, which .dropna() handles)
	year_series = df['year'].dropna().astype(int)
	# Find the years with the highest and lowest frequencies (handle ties)
	year_counts = year_series.value_counts()
	max_hist_freq = max(year_counts.values)
	min_hist_freq = min(year_counts.values)
	print(f"max_hist_freq: {max_hist_freq} | min_hist_freq: {min_hist_freq}")
	
	# Get the years with the maximum and minimum frequencies
	max_freq = year_counts.max()
	min_freq = year_counts.min()
	max_freq_years = year_counts[year_counts == max_freq].index.tolist()
	min_freq_years = year_counts[year_counts == min_freq].index.tolist()
	
	# Calculate mean, median, and standard deviation
	mean_year = year_series.mean()
	median_year = year_series.median()
	std_year = year_series.std()
	# Calculate 95% confidence interval for the mean
	confidence_level = 0.95
	n = len(year_series)
	mean_conf_interval = stats.t.interval(confidence_level, df=n-1, loc=mean_year, scale=stats.sem(year_series))
	# Get the overall shape of the distribution
	distribution_skew = year_series.skew()
	distribution_kurtosis = year_series.kurtosis()
	skew_desc = "right-skewed" if distribution_skew > 0 else "left-skewed" if distribution_skew < 0 else "symmetric"
	kurt_desc = "heavy-tailed" if distribution_kurtosis > 0 else "light-tailed" if distribution_kurtosis < 0 else "normal-tailed"
	# Calculate percentiles
	q25, q75 = year_series.quantile([0.25, 0.75])
	# Plot KDE using scipy.stats.gaussian_kde
	plt.figure(figsize=FIGURE_SIZE)
	sns.histplot(
		year_series,
		bins=BINs,
		color="#5c6cf8",
		kde=True,
		edgecolor="white",
		alpha=0.95,
		linewidth=1.5,
		label="Temporal Distribution Histogram"
	)
	# Create the KDE object and adjust bandwidth to match Seaborn's default behavior
	kde = gaussian_kde(year_series, bw_method='scott')  # Use 'scott' or 'silverman', or a custom value
	x_range = np.linspace(start_year, end_year, 300)
	kde_values = kde(x_range)
	bin_width = (end_year - start_year) / BINs  # Approximate bin width of the histogram
	kde_scaled = kde_values * len(year_series) * bin_width  # Scale KDE to match frequency
	plt.plot(
		x_range,
		kde_scaled,
		color="#3a3a3a",
		linewidth=2.0,
		linestyle="-",
		label="Kernel Density Estimate (KDE)",
	)
	world_war_1 = [1914, 1918]
	world_war_2 = [1939, 1945]
	ww_cols = ['#fa3627', '#24f81d']
	padding = 1.05
	max_padding = 1.1
	# Add shaded regions for WWI and WWII (plot these first to ensure they are in the background)
	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		plt.axvspan(world_war_1[0], world_war_1[1], color=ww_cols[0], alpha=0.2, label='World War One')

	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		plt.axvspan(world_war_2[0], world_war_2[1], color=ww_cols[1], alpha=0.2, label='World War Two')

	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		for year in world_war_1:
			plt.axvline(x=year, color='r', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_1[0] + world_war_1[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWI',
			color=ww_cols[0],
			fontsize=12,
			fontweight="bold",
			ha="center",  # horizontal alignment
		)
	
	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		for year in world_war_2:
			plt.axvline(x=year, color=ww_cols[1], linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_2[0] + world_war_2[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWII',
			color=ww_cols[1],
			fontsize=12,
			fontweight="bold",
			ha="center", # horizontal alignment
		)

	# Add visual representations of key statistics
	plt.axvline(x=mean_year, color='#ee8206ee', linestyle='-.', lw=2.5, label=f'Mean Year: {mean_year:.1f}')
	plt.axvspan(mean_year - std_year, mean_year + std_year, color='#fdff7c', alpha=0.15, label='Mean ± 1 SD')

	valid_count = len(year_series)
	stats_text = (
		# f"Samples with valid dates: {valid_count} (~{round(valid_count / df.shape[0] * 100)}%)\n\n"
		"Frequency Statistics:\n"
		f"  Most frequent year(s): {', '.join(map(str, max_freq_years))} ({max_freq} images)\n"
		f"  Least frequent year(s): {', '.join(map(str, min_freq_years))} ({min_freq} images)\n\n"
		"Central Tendency [Year]:\n"
		f"  Median: {median_year:.0f}\n"
		f"  Mean: {mean_year:.1f}\n"
		f"     Confidence Interval (95%): [{mean_conf_interval[0]:.1f}, {mean_conf_interval[1]:.1f}]\n"
		f"  Standard deviation: {std_year:.1f}\n\n"
		"Percentiles:\n"
		f"  25th: {q25:.2f}\n"
		f"  75th: {q75:.2f}\n\n"
		"Distribution Shape:\n"
		f"  Skewness: {distribution_skew:.2f} ({skew_desc})\n"
		f"  Kurtosis: {distribution_kurtosis:.2f} ({kurt_desc})"
	)
	plt.text(
		0.01,
		0.98,
		stats_text,
		transform=plt.gca().transAxes,
		fontsize=10,
		verticalalignment='top',
		horizontalalignment='left',
		color='black',
		bbox=dict(
			boxstyle='round,pad=0.5',
			facecolor='white',
			alpha=0.9,
			edgecolor='none', 
			linewidth=0.0,
		)
	)
	plt.title(
		label=f'Temporal Distribution ({start_date} - {end_date}) Total Samples: {df.shape[0]}', fontsize=12, fontweight='bold')
	plt.xlabel('')
	plt.tick_params(axis='x', length=0, width=0, color='black', labelcolor='black', labelsize=15)
	plt.ylabel('Frequency', fontsize=15, fontweight='bold')
	plt.ylim(0, max_freq * max_padding)  # Add some padding to the y-axis
	plt.yticks(fontsize=15, rotation=90, va='center')

	plt.xlim(start_year - 2, end_year + 2)
	plt.legend(
		loc='upper left',
		bbox_to_anchor=(0.01, 0.56),
		fontsize=10,
		frameon=False,
	)
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

def plot_long_tailed_distribution(
		df: pd.DataFrame,
		fpth: str,
		top_n: int=None,  # Option to show only top N labels
		FIGURE_SIZE: tuple=(17, 9),
		DPI: int=300,
		head_threshold: int=5000,  # Labels with frequency > head_threshold
		tail_threshold: int=1000,   # Labels with frequency < tail_threshold
	):
	label_counts = df['label'].value_counts().sort_values(ascending=False)
	
	# Handle large number of labels
	if top_n and len(label_counts) > top_n:
		top_labels = label_counts.head(top_n)
		other_count = label_counts[top_n:].sum()
		top_labels = pd.concat([top_labels, pd.Series([other_count], index=['Other'])])
		label_counts = top_labels
	
	# Identify Head, Torso, and Tail segments
	head_labels = label_counts[label_counts > head_threshold].index.tolist()
	tail_labels = label_counts[label_counts < tail_threshold].index.tolist()
	torso_labels = label_counts[(label_counts >= tail_threshold) & (label_counts <= head_threshold)].index.tolist()

	segment_specs = {
		'Head': {
			'color': "#009670e4",
			'label': 'Head'.upper(),
			'opacity': 0.2,
			'fontsize': 16,
		},
		'Torso': {
			'color': "#d4ae02",
			'label': 'Torso'.upper(),
			'opacity': 0.2,
			'fontsize': 16,
		},
		'Tail': {
			'color': "#ee4747",
			'label': 'Tail'.upper(),
			'opacity': 0.2,
			'fontsize': 16,
		},
	}

	fig, ax = plt.subplots(
		figsize=FIGURE_SIZE, 
		facecolor='white', 
		constrained_layout=True,
	)
	
	bars = label_counts.plot(
		kind='bar',
		ax=ax,
		color="#193C6992",
		width=0.7,
		edgecolor='white',
		linewidth=0.8,
		alpha=0.75,
		label='Linear'.capitalize(),
		zorder=2,
	)
	
	ax.grid(False) # Remove grid lines
	
	# Create shaded regions for Head, Torso, and Tail
	all_indices = np.arange(len(label_counts))
	head_indices = [i for i, label in enumerate(label_counts.index) if label in head_labels]
	torso_indices = [i for i, label in enumerate(label_counts.index) if label in torso_labels]
	tail_indices = [i for i, label in enumerate(label_counts.index) if label in tail_labels]
	
	# Sort the indices to ensure proper shading
	head_indices.sort()
	torso_indices.sort()
	tail_indices.sort()
	
	# Add shaded areas if segments exist
	ymax = label_counts.max() * 1.11  # Set maximum y-value for shading
	
	segment_text_yoffset = 1.1 if len(head_labels) < 5 else 1.0
	segment_text_opacity = 0.7

	if head_indices:
		ax.axvspan(
			min(head_indices) - 0.4, 
			max(head_indices) + 0.4, 
			alpha=segment_specs['Head']['opacity'],
			color=segment_specs['Head']['color'],
		)
		ax.text(
			np.mean(head_indices), 
			ymax * segment_text_yoffset,
			f"{segment_specs['Head']['label']}\n({len(head_labels)} labels)",
			horizontalalignment='center',
			verticalalignment='center',
			fontsize=segment_specs['Head']['fontsize'],
			fontweight='bold',
			color=segment_specs['Head']['color'],
			zorder=5,
		)
	
	if torso_indices:
		ax.axvspan(
			min(torso_indices) - 0.4, 
			max(torso_indices) + 0.4, 
			alpha=segment_specs['Torso']['opacity'],
			color=segment_specs['Torso']['color'],
		)
		ax.text(
			np.mean(torso_indices), 
			ymax * segment_text_yoffset, 
			f"{segment_specs['Torso']['label']}\n({len(torso_labels)} labels)",
			horizontalalignment='center',
			verticalalignment='center',
			fontsize=segment_specs['Torso']['fontsize'],
			fontweight='bold',
			color=segment_specs['Torso']['color'],
			zorder=5,
		)
	
	if tail_indices:
		ax.axvspan(
			min(tail_indices) - 0.4, 
			max(tail_indices) + 0.4, 
			alpha=segment_specs['Tail']['opacity'],
			color=segment_specs['Tail']['color'],
		)
		ax.text(
			np.mean(tail_indices), 
			ymax * segment_text_yoffset,
			f"{segment_specs['Tail']['label']}\n({len(tail_labels)} labels)",
			horizontalalignment='center',
			verticalalignment='center',
			fontsize=segment_specs['Tail']['fontsize'],
			fontweight='bold',
			color=segment_specs['Tail']['color'],
			zorder=5,
		)
	
	# Hide all spines initially
	for spine in ax.spines.values():
		spine.set_visible(False)

	# Show only the left, right and bottom spines
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(True)

	# Enhance readability for large number of labels
	if len(label_counts) > 20:
		plt.xticks(rotation=90, fontsize=15, fontweight='bold')
	else:
		plt.xticks(rotation=45, fontsize=9, ha='right')
	plt.yticks(fontsize=8, rotation=90, va='center', fontweight='bold')
	
	# Add value labels on top of bars
	for i, v in enumerate(label_counts):
		text_color = segment_specs['Head']['color'] if i in head_indices else (segment_specs['Torso']['color'] if i in torso_indices else segment_specs['Tail']['color'])
		ax.text(
			i, 
			v + (v * 0.015),  # Adjust vertical position relative to bar height
			str(v), 
			ha='center',
			fontsize=9,
			fontweight='bold',
			alpha=0.85,
			color=text_color,
			rotation=75,
			zorder=5,
		)
	
	# Add a logarithmic scale option for highly imbalanced distributions
	if label_counts.max() / label_counts.min() > 50:
		ax_log = ax.twinx()
		ax_log.set_yscale('log')
		label_counts.plot(
			kind='line',
			ax=ax_log,
			color='#8a008a',
			marker='o',
			markerfacecolor='none',  				# Remove marker fill
			markeredgecolor='#8a008a',  # Set marker edge color
			markersize=2,          				# Adjust marker size
			linewidth=2.0,
			alpha=0.6,
			label='Logarithmic'.capitalize(),
			zorder=2,
		)
		ax_log.set_ylabel(
			ylabel='Log Sample Frequency', 
			color='#8a008a', 
			fontsize=15,
			fontweight='bold',
		)
		ax_log.tick_params(axis='y', colors='#8a008a')
		ax_log.spines['right'].set_visible(True)
		ax_log.spines['right'].set_color('#8a008a')
		ax_log.spines['right'].set_linewidth(1.0)
		ax_log.spines['right'].set_alpha(0.7)
		ax_log.grid(axis='y', alpha=0.3, linestyle='--', color="#696969", zorder=0)

		# Hide all spines for the logarithmic scale
		for spine in ax_log.spines.values():
			spine.set_visible(False)

	ax.set_xlabel('')
	ax.tick_params(axis='x', length=0, width=0, color='none', labelcolor='black', labelsize=12)
	ax.tick_params(axis='y', color='black', labelcolor='black', labelsize=11)
	ax.set_ylabel('Sample Frequency', fontsize=15, fontweight='bold')
	ax.set_ylim(0, label_counts.max() * 1.17)
	
	# Add basic statistics for the distribution
	imbalance_ratio = label_counts.max()/label_counts.min()
	median_label_size = label_counts.median()
	mean_label_size = label_counts.mean()
	std_label_size = label_counts.std()
	most_freq_label = label_counts.max()/df.shape[0]*100
	least_freq_label = label_counts.min()/df.shape[0]*100
	
	# Add segment statistics
	head_count = sum(label_counts[head_labels])
	torso_count = sum(label_counts[torso_labels])
	tail_count = sum(label_counts[tail_labels])
	total_samples = df.shape[0]
	
	head_percent = head_count/total_samples*100 if total_samples > 0 else 0
	torso_percent = torso_count/total_samples*100 if total_samples > 0 else 0
	tail_percent = tail_count/total_samples*100 if total_samples > 0 else 0
	
	stats_text = (
		f"Imbalance ratio: {imbalance_ratio:.1f}\n\n"
		f"Label Statistics:\n"
		f"    Median: {median_label_size:.0f}\n"
		f"    Mean: {mean_label_size:.1f}\n"
		f"    Standard deviation: {std_label_size:.1f}\n"
		f"    Most frequent: {most_freq_label:.1f}%\n"
		f"    Least frequent: {least_freq_label:.2f}%\n\n"
		f"Segment Statistics:\n"
		f"    Head: {len(head_labels)} labels, {head_count} samples ({head_percent:.1f}%)\n"
		f"    Torso: {len(torso_labels)} labels, {torso_count} samples ({torso_percent:.1f}%)\n"
		f"    Tail: {len(tail_labels)} labels, {tail_count} samples ({tail_percent:.1f}%)"
	)
	print(f"stats_text:\n{stats_text}\n")

	# Create custom legend elements
	custom_lines = [
		Line2D([0], [0], color="#00315393", lw=6),  # Linear scale
		Line2D([0], [0], color='#480264', lw=2, marker='o', markersize=3, markerfacecolor='none', markeredgecolor="#480264")  # Logarithmic scale
	]

	custom_labels = ['Linear', 'Logarithmic']

	# Create the legend with just these elements
	legend = ax.legend(
			custom_lines, 
			custom_labels, 
			loc="best",
			title='Label Distribution',
			title_fontsize=15,
			fontsize=13,
			ncol=1,
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='none',
			facecolor='white',
	)
	legend.set_zorder(100)

	plt.savefig(fpth, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	return {
		'head_labels': head_labels,
		'torso_labels': torso_labels,
		'tail_labels': tail_labels,
		'head_count': head_count,
		'torso_count': torso_count,
		'tail_count': tail_count,
	}

def plot_single_labeled_head_torso_tail_samples(
		metadata_path,
		metadata_train_path,
		metadata_val_path,
		save_path="head_torso_tail_grid.png",
		head_threshold=5000,  # Labels with frequency > 5000
		tail_threshold=1000,  # Labels with frequency < 1000
		tile_img_h=256,  # Image area height per tile (excl. title text)
		tile_w=256,  # Tile width (fixed so columns align perfectly)
		title_h=26,  # Text area height at top of each tile ("GT: ...")
		left_gutter=40,  # gutter for rotated row labels
		bg_color="#ffffff",
		scale_factor=3.5  # Parameter to scale the entire figure
	):

	print(f"Analyzing Label Distribution from {metadata_path}")

	# 1) Load metadata
	try:
		df_full = pd.read_csv(metadata_path, dtype=dtypes, low_memory=False, on_bad_lines='skip',)
		df_train = pd.read_csv(metadata_train_path, dtype=dtypes, low_memory=False, on_bad_lines='skip',)  # Not used here, but kept for parity
		df_val = pd.read_csv(metadata_val_path, dtype=dtypes, low_memory=False, on_bad_lines='skip',)
	except FileNotFoundError as e:
		print(f"Error loading metadata files: {e}")
		return None, None
	
	# 2) Head / Torso / Tail segmentation from full dataset
	label_counts_full = df_full['label'].value_counts()
	labels_in_train = set(df_train['label'].unique().tolist())
	labels_in_val = set(df_val['label'].unique().tolist())

	print(f"Total unique labels: FULL: {len(label_counts_full)} | TRAIN: {len(labels_in_train)} | VAL: {len(labels_in_val)}")


	head_labels = label_counts_full[label_counts_full > head_threshold].index.tolist()
	tail_labels = label_counts_full[label_counts_full < tail_threshold].index.tolist()
	torso_labels = label_counts_full[(label_counts_full >= tail_threshold) & (label_counts_full <= head_threshold)].index.tolist()
	# Restrict to labels present in validation set

	segments = {
		"Head": [lbl for lbl in head_labels if lbl in labels_in_val],
		"Torso": [lbl for lbl in torso_labels if lbl in labels_in_val],
		"Tail": [lbl for lbl in tail_labels if lbl in labels_in_val],
	}
	print(f"Head: {len(head_labels)} | Torso: {len(torso_labels)} | Tail: {len(tail_labels)}")

	# 3) Sample up to 3 examples per segment for the grid
	# We'll pick one image path per chosen label (if available)
	i2t_queries = {seg: [] for seg in segments}
	for segment_name, segment_labels in segments.items():
		if not segment_labels:
			continue
		labels_to_sample = random.sample(segment_labels, min(3, len(segment_labels)))
		for label in labels_to_sample:
			imgs = df_val[df_val['label'] == label]['img_path'].tolist()
			if imgs:
				i2t_queries[segment_name].append({"image_path": random.choice(imgs), "label": label})
	
	t2i_queries = [{"label": it["label"], "segment": seg} for seg, lst in i2t_queries.items() for it in lst]

	######################################
	# 4) Build composite image (3x3) grid
	######################################
	rows = ["Head", "Torso", "Tail"]
	n_cols = 3
	
	# Scale dimensions
	scaled_tile_w = int(tile_w * scale_factor)
	scaled_tile_img_h = int(tile_img_h * scale_factor)
	scaled_title_h = int(title_h * scale_factor)
	scaled_left_gutter = int(left_gutter * scale_factor)
	tile_h_total = scaled_title_h + scaled_tile_img_h
	canvas_w = scaled_left_gutter + n_cols * scaled_tile_w
	canvas_h = len(rows) * tile_h_total
	composite = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
	draw = ImageDraw.Draw(composite)
	
	def load_font(name, size):
		try:
			return ImageFont.truetype(name, int(size * scale_factor))
		except Exception:
			try:
				return ImageFont.truetype("DejaVuSans.ttf", int(size * scale_factor))
			except Exception:
				return ImageFont.load_default()
	
	title_font = load_font("DejaVuSansMono.ttf", 17)
	row_font = load_font("DejaVuSans-Bold.ttf", 15)
	segment_colors = {
		'Head': "#009670e4",
		'Torso': "#d4ae02",
		'Tail': "#ee4747",
	}
	
	# Helper to paste one tile at exact position with no gaps
	def paste_tile(img_path, label, x0, y0):
		# Make a clean tile background
		tile = Image.new("RGB", (scaled_tile_w, tile_h_total), color=bg_color)
		td = ImageDraw.Draw(tile)
		# Draw title text centered in title area
		gt_text = f"{label}"
		if hasattr(title_font, "getbbox"):
			tw, th = title_font.getbbox(gt_text)[2:]
		else:
			tw, th = title_font.getsize(gt_text)
		td.text(
			((scaled_tile_w - tw) // 2, max(0, (scaled_title_h - th) // 2)), 
			gt_text, 
			fill="#000000",
			font=title_font,
			stroke_width=2,
			stroke_fill="#000000",
		)
		# Draw a subtle background for the title area
		td.rectangle(
			[(0, 0), (scaled_tile_w, scaled_title_h)], 
			outline="#A8A8A8",
			width=int(1 * scale_factor)
		)
		# Load image, preserve aspect, fit inside scaled_tile_w x scaled_tile_img_h
		# Fallback to blank if missing/corrupted
		try:
			if img_path and os.path.exists(img_path):
				img = Image.open(img_path).convert("RGB")
			else:
				img = Image.new("RGB", (scaled_tile_w, scaled_tile_img_h), color=(230, 230, 230))
		except Exception:
			img = Image.new("RGB", (scaled_tile_w, scaled_tile_img_h), color=(230, 230, 230))
		# Resize to fit within (scaled_tile_w, scaled_tile_img_h), keeping aspect ratio
		scale = min(scaled_tile_w / img.width, scaled_tile_img_h / img.height) if img.width and img.height else 1.0
		new_w = max(1, int(img.width * scale))
		new_h = max(1, int(img.height * scale))
		img = img.resize((new_w, new_h), Image.LANCZOS)
		# Paste centered in the image area
		x_img = (scaled_tile_w - new_w) // 2
		y_img = scaled_title_h + (scaled_tile_img_h - new_h) // 2
		tile.paste(img, (x_img, y_img))
		# Paste tile onto composite at exact pixel location (no spacing)
		composite.paste(tile, (x0, y0))
	
	# Paste grid tiles
	for r, segment_name in enumerate(rows):
		samples = i2t_queries.get(segment_name, [])
		for c in range(n_cols):
			x0 = scaled_left_gutter + c * scaled_tile_w
			y0 = r * tile_h_total
			if c < len(samples):
				paste_tile(samples[c]["image_path"], samples[c]["label"], x0, y0)
			else:
				# Blank tile (still zero spacing, just empty white background)
				blank = Image.new("RGB", (scaled_tile_w, tile_h_total), color=bg_color)
				composite.paste(blank, (x0, y0))
	
	# Draw rotated row label centered vertically in this row, inside left gutter
	for r, segment_name in enumerate(rows):
		row_center_y = r * tile_h_total + tile_h_total // 2
		text = segment_name
		color = segment_colors[segment_name]
		
		# Get text dimensions
		if hasattr(row_font, "getbbox"):
			bbox = row_font.getbbox(text)
			tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
		else:
			tw, th = row_font.getsize(text)
		
		print(f"Text '{text}': width={tw}, height={th}")
		
		# Create text image with enough padding
		padding = 25
		text_img = Image.new("RGBA", (tw + padding*3, th + padding*3), (255, 255, 255, 0))
		td2 = ImageDraw.Draw(text_img)
		td2.text(
			(padding, padding), 
			text.upper(), 
			fill=color, 
			font=row_font,
			stroke_width=2,
			stroke_fill=color,
		)
		
		# Rotate the text image
		text_rot = text_img.rotate(90, expand=1)
		
		# Calculate position to center the rotated text in the gutter
		# After rotation, width becomes height and vice versa
		tx = (scaled_left_gutter - text_rot.width) // 2
		ty = row_center_y - text_rot.height // 2
		
		# Ensure we don't go negative
		tx = max(5, tx)  # Keep small margin from edge
		ty = max(0, ty)
		
		print(f"Positioning '{text}' at tx={tx}, ty={ty}, gutter_width={scaled_left_gutter}")
		
		# Add a semi-transparent background for better visibility
		bg_padding = 10
		bg_box = Image.new(
			"RGBA", 
			(text_rot.width + bg_padding*2, text_rot.height + bg_padding*2), 
			(255, 255, 255, 180)
		)
		
		# Paste background first, then text
		composite.paste(bg_box, (tx - bg_padding, ty - bg_padding), bg_box)
		composite.paste(text_rot, (tx, ty), text_rot)
	
	# Save the composite image
	composite.save(save_path.replace(".png", ".jpg"), "JPEG", quality=100, dpi=(200, 200))
	
	print(f"Saved 3x3 sample grid to: {save_path}")
	
	# Also return queries similar to before (now organized by segment)
	# Flatten i2t for backward compatibility if needed
	flat_i2t = []
	for seg, lst in i2t_queries.items():
		for it in lst:
			flat_i2t.append({"image_path": it["image_path"], "label": it["label"], "segment": seg})
		
	return flat_i2t, t2i_queries

def plot_label_distribution(
		df: pd.DataFrame,
		fpth: str,
		label_column: str,
		FIGURE_SIZE: tuple,
		DPI: int,
		top_n: int = None  # Option to show only top N labels
	):

	label_counts = df[label_column].value_counts()
	
	# Handle large number of labels
	if top_n and len(label_counts) > top_n:
		top_labels = label_counts.head(top_n)
		other_count = label_counts[top_n:].sum()
		top_labels = pd.concat([top_labels, pd.Series([other_count], index=['Other'])])
		label_counts = top_labels
	
	fig, ax = plt.subplots(
		figsize=FIGURE_SIZE, 
		facecolor='white', 
		# constrained_layout=True,
	)
	
	# Plot with better styling
	bars = label_counts.plot(
		kind='bar',
		ax=ax,
		color="green",
		width=0.8,
		edgecolor='white',
		linewidth=0.8,
		alpha=0.8,
		label='Linear Scale'.capitalize()
	)

	# Hide all spines initially
	for spine in ax.spines.values():
		spine.set_visible(False)

	# Show only the left, right and bottom spines
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(True)

	# Enhance readability for large number of labels
	if len(label_counts) > 20:
		plt.xticks(rotation=90, fontsize=11)
	else:
		plt.xticks(rotation=45, fontsize=9, ha='right')
	plt.yticks(fontsize=9, rotation=90, va='center')
	
	# Add value labels on top of bars
	for i, v in enumerate(label_counts):
		ax.text(
			i, 
			v + (v * 0.05),  # Adjust vertical position relative to bar height
			str(v), 
			ha='center',
			fontsize=8,
			fontweight='bold',
			alpha=0.8,
			color='blue',
			rotation=75,
			bbox=dict(
				facecolor='white',
				edgecolor='none',
				alpha=0.7,
				pad=0.5
			)
		)
	
	# Add a logarithmic scale option for highly imbalanced distributions
	ax_log = None
	if label_counts.max() / label_counts.min() > 50:
		ax_log = ax.twinx()
		ax_log.set_yscale('log')
		label_counts.plot(
			kind='line',
			ax=ax_log,
			color='red',
			marker='o',
			markerfacecolor='none',  # Remove marker fill
			markeredgecolor='red',   # Set marker edge color
			markersize=3,           # Optional: adjust marker size
			linewidth=2.5,
			alpha=0.9,
			label='Logarithmic'
		)
		ax_log.set_ylabel(
			ylabel='Log Sample Frequency', 
			color='red', 
			fontsize=10, 
			fontweight='bold',
		)
		ax_log.tick_params(axis='y', colors='red')
	
	# Hide all spines for the logarithmic scale
	if ax_log is not None:
		for spine in ax_log.spines.values():
			spine.set_visible(False)

	ax.set_xlabel('')
	ax.tick_params(axis='x', length=0, width=0, color='none', labelcolor='black', labelsize=12)
	ax.tick_params(axis='y', color='black', labelcolor='black', labelsize=11)
	ax.set_ylabel('Sample Frequency', fontsize=10, fontweight='bold')
	
	# Add basic statistics for the distribution
	imbalaned_ratio = label_counts.max()/label_counts.min()
	median_label_size = label_counts.median()
	mean_label_size = label_counts.mean()
	std_label_size = label_counts.std()
	most_freq_label = label_counts.max()/df.shape[0]*100
	least_freq_label = label_counts.min()/df.shape[0]*100
	stats_text = (
		f"Imbalance ratio: {imbalaned_ratio:.1f}\n\n"
		f"Label Statistics:\n"
		f"    Median: {median_label_size:.0f}\n"
		f"    Mean: {mean_label_size:.1f}\n"
		f"    Standard deviation: {std_label_size:.1f}\n"
		f"    Most frequent: {most_freq_label:.1f}%\n"
		f"    Least frequent: {least_freq_label:.2f}%"
	)
	print(f"stats_text:\n{stats_text}\n")
	plt.text(
		0.74, # horizontal position
		0.86, # vertical position
		stats_text,
		transform=ax.transAxes,
		fontsize=15,
		verticalalignment='top',
		horizontalalignment='left',
		color='black',
		bbox=dict(
			boxstyle='round,pad=0.5',
			facecolor='white',
			alpha=0.8,
			edgecolor='none', 
			linewidth=0.0,
		)
	)

	# Enhanced title and labels
	plt.title(
		f'Label Distribution (Total samples: {df.shape[0]} Unique Labels: {len(df[label_column].unique())})', 
		fontsize=15,
		fontweight='bold',
	)
	# Create a single legend
	h1, l1 = ax.get_legend_handles_labels()
	# h2, l2 = ax_log.get_legend_handles_labels()
	h2, l2 = ([], []) if ax_log is None else ax_log.get_legend_handles_labels()
	ax.legend(
		h1 + h2, 
		l1 + l2, 
		# loc='best', 
		loc='upper left',  # Changed to upper left
		bbox_to_anchor=(0.73, 0.99),  # Match horizontal position with text (0.74)
		title='Label Distribution (Scale)',
		title_fontsize=12,
		fontsize=11, 
		ncol=2,
		frameon=False, 
		# fancybox=True, 
		# shadow=True, 
		# edgecolor='black', 
		# facecolor='white'
	)

	plt.grid(axis='y', alpha=0.7, linestyle='--')
	plt.tight_layout()
	plt.savefig(fpth, dpi=DPI, bbox_inches='tight')
	plt.close()