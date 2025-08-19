import hashlib
import os
import torch
import clip
import datetime
import time
import json
import gc
import threading
import concurrent.futures
import random
import platform
import re
import argparse
import numpy as np
import pandas as pd
import gzip
import pickle
import dill
import math
import copy
import itertools
import tabulate
import inspect
import functools
import sys
import traceback
import requests
from io import BytesIO
import hashlib
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import multiprocessing
import glob
import psutil
import ast
import shutil
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Union, List, Dict, Any, Optional

from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from collections import defaultdict
import logging
import warnings
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

def round_up(num: int) -> int:
	if num == 0:
		return 0
	
	num_str = str(num)
	num_digits = len(num_str)
	
	# For single digit, round up to 10
	if num_digits == 1:
		return 10
	
	# For 3-digit numbers starting with 1, round to next hundred
	if num_digits == 3 and num_str[0] == '1':
		return 200
	
	# For numbers starting with 1 and more than 3 digits, use second-most significant
	if num_str[0] == '1' and num_digits > 3:
		base = 10 ** (num_digits - 2)
		return ((num - 1) // base + 1) * base
	else:
		# For other cases, use the most significant digit position
		base = 10 ** (num_digits - 1)
		return ((num - 1) // base + 1) * base

def cleanup_old_temp_dirs():
	
	temp_dirs = glob.glob("/tmp/pymp-*")
	for temp_dir in temp_dirs:
		try:
			shutil.rmtree(temp_dir, ignore_errors=True)
		except:
			pass
	if temp_dirs:
		print(f"Cleaned up {len(temp_dirs)} old temp directories")

def get_multi_label_head_torso_tail_samples(
		metadata_path: str,
		metadata_train_path: str,
		metadata_val_path: str,
		num_samples_per_segment: int = 5,
) -> Tuple[List[Dict], List[str]]:
		"""
		Sample from head, torso, tail distributions for multi-label datasets.
		Only used when dataset_type == 'multi_label'
		"""
		try:
				import ast
				# Load metadata
				df_val = pd.read_csv(metadata_val_path)
				
				# For multi-label, we need to count label frequencies differently
				all_labels = []
				for labels_str in df_val['multimodal_labels']:
						try:
								labels = ast.literal_eval(labels_str)
								all_labels.extend(labels)
						except:
								continue
				
				# Count label frequencies
				label_counts = pd.Series(all_labels).value_counts()
				
				# Define head, torso, tail based on frequency distribution
				total_labels = len(label_counts)
				head_threshold = int(total_labels * 0.2)  # Top 20%
				tail_threshold = int(total_labels * 0.8)   # Bottom 20%
				
				head_labels = set(label_counts.head(head_threshold).index)
				tail_labels = set(label_counts.tail(total_labels - tail_threshold).index)
				torso_labels = set(label_counts.index) - head_labels - tail_labels
				
				# Sample images and labels
				i2t_samples = []
				t2i_samples = []
				
				# Sample images with head, torso, tail labels
				for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
						segment_samples = []
						
						for _, row in df_val.iterrows():
								try:
										row_labels = set(ast.literal_eval(row['multimodal_labels']))
										if row_labels & segment_labels:  # If any intersection
												segment_samples.append({
														'image_path': row['img_path'],
														'labels': list(row_labels),
														'segment': segment_name
												})
								except:
										continue
						
						# Randomly sample from this segment
						if len(segment_samples) >= num_samples_per_segment:
								sampled = random.sample(segment_samples, num_samples_per_segment)
								i2t_samples.extend(sampled)
				
				# Sample text queries from head, torso, tail
				for segment_name, segment_labels in [("head", head_labels), ("torso", torso_labels), ("tail", tail_labels)]:
						segment_label_list = list(segment_labels)
						if len(segment_label_list) >= num_samples_per_segment:
								sampled_labels = random.sample(segment_label_list, num_samples_per_segment)
								t2i_samples.extend(sampled_labels)
				
				return i2t_samples, t2i_samples
				
		except Exception as e:
				print(f"Error in multi-label sampling: {e}")
				return [], []

def get_single_label_head_torso_tail_samples(
		metadata_path,
		metadata_train_path,
		metadata_val_path,
		num_samples_per_segment=5,
		head_threshold=5000,  # Labels with frequency > 5000
		tail_threshold=1000,  # Labels with frequency < 1000
		save_path="head_torso_tail_grid.png",
		tile_img_h=256,  # Image area height per tile (excl. title text)
		tile_w=256,  # Tile width (fixed so columns align perfectly)
		title_h=26,  # Text area height at top of each tile ("GT: ...")
		left_gutter=40,  # gutter for rotated row labels
		bg_color="#ffffff",
		scale_factor=4.0  # Parameter to scale the entire figure
	):
	print(f"Analyzing Label Distribution from {metadata_path}")
	# 1) Load metadata
	try:
			df_full = pd.read_csv(metadata_path)
			_ = pd.read_csv(metadata_train_path)  # Not used here, but kept for parity
			df_val = pd.read_csv(metadata_val_path)
	except FileNotFoundError as e:
			print(f"Error loading metadata files: {e}")
			return None, None
	
	# 2) Head / Torso / Tail segmentation from full dataset
	label_counts_full = df_full['label'].value_counts()
	head_labels = label_counts_full[label_counts_full > head_threshold].index.tolist()
	tail_labels = label_counts_full[label_counts_full < tail_threshold].index.tolist()
	torso_labels = label_counts_full[(label_counts_full >= tail_threshold) & (label_counts_full <= head_threshold)].index.tolist()
	# Restrict to labels present in validation set
	labels_in_val = set(df_val['label'].unique().tolist())
	segments = {
		"Head": [lbl for lbl in head_labels if lbl in labels_in_val],
		"Torso": [lbl for lbl in torso_labels if lbl in labels_in_val],
		"Tail": [lbl for lbl in tail_labels if lbl in labels_in_val],
	}
	
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
	
	# 4) Build composite image with PIL (true zero spacing between tiles)
	rows = ["Head", "Torso", "Tail"]
	n_cols = 3
	
	# Scale dimensions
	scaled_tile_w = int(tile_w * scale_factor)
	scaled_tile_img_h = int(tile_img_h * scale_factor)
	scaled_title_h = int(title_h * scale_factor)
	scaled_left_gutter = int(left_gutter * scale_factor)  # Proper scaling without extra multiplier
	tile_h_total = scaled_title_h + scaled_tile_img_h
	canvas_w = scaled_left_gutter + n_cols * scaled_tile_w
	canvas_h = len(rows) * tile_h_total
	composite = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
	draw = ImageDraw.Draw(composite)
	
	# Try to pick decent fonts; fall back gracefully
	def load_font(name, size):
			try:
					return ImageFont.truetype(name, int(size * scale_factor))
			except Exception:
					try:
							return ImageFont.truetype("DejaVuSans.ttf", int(size * scale_factor))
					except Exception:
							return ImageFont.load_default()
	
	title_font = load_font("DejaVuSansMono.ttf", 15)
	row_font = load_font("DejaVuSans-Bold.ttf", 15)
	segment_colors = {
		"Head": "#009670", 
		"Torso": "#d4ae02",
		"Tail": "#ee4747",
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
			td.text(((scaled_tile_w - tw) // 2, max(0, (scaled_title_h - th) // 2)), gt_text, fill=(0, 0, 0), font=title_font)
			# Draw a subtle background for the title area
			td.rectangle([(0, 0), (scaled_tile_w, scaled_title_h)], outline=(200, 200, 200), width=int(1 * scale_factor))
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
	composite.save(save_path, dpi=(300, 300))
	print(f"Saved 3x3 sample grid to: {save_path}")
	
	# Also return queries similar to before (now organized by segment)
	# Flatten i2t for backward compatibility if needed
	flat_i2t = []
	for seg, lst in i2t_queries.items():
			for it in lst:
					flat_i2t.append({"image_path": it["image_path"], "label": it["label"], "segment": seg})
	# t2i kept minimal (not used in plot)
	t2i_queries = [{"label": it["label"], "segment": seg} for seg, lst in i2t_queries.items() for it in lst]
	return flat_i2t, t2i_queries

def get_lora_params(path_string: str) -> dict:
	params = {}
	# Convention 1: lora_rank_(\d+), lora_alpha_(\d+\.\d+), lora_dropout_(\d+\.\d+)
	lora_rank_match_long = re.search(r"lora_rank_(\d+)", path_string)
	lora_alpha_match_long = re.search(r"lora_alpha_(\d+\.\d+)", path_string)
	lora_dropout_match_long = re.search(r"lora_dropout_(\d+\.\d+)", path_string)
	if lora_rank_match_long:
			params["lora_rank"] = int(lora_rank_match_long.group(1))
	if lora_alpha_match_long:
			params["lora_alpha"] = float(lora_alpha_match_long.group(1))
	if lora_dropout_match_long:
			params["lora_dropout"] = float(lora_dropout_match_long.group(1))
	# Convention 2: _lor_(\d+)_ , _loa_(\d+\.\d+)_ , _lod_(\d+\.\d+)_
	lora_rank_match_short = re.search(r"_lor_(\d+)_", path_string)
	lora_alpha_match_short = re.search(r"_loa_(\d+\.\d+)_", path_string)
	lora_dropout_match_short = re.search(r"_lod_(\d+\.\d+)_", path_string)
	if "lora_rank" not in params and lora_rank_match_short:
			params["lora_rank"] = int(lora_rank_match_short.group(1))
	if "lora_alpha" not in params and lora_alpha_match_short:
			params["lora_alpha"] = float(lora_alpha_match_short.group(1))
	if "lora_dropout" not in params and lora_dropout_match_short:
			params["lora_dropout"] = float(lora_dropout_match_short.group(1))
	return params

def get_model_hash(model: torch.nn.Module) -> str:
	hasher = hashlib.md5()
	param_sample = []
	for i, param in enumerate(model.parameters()):
		if i % 10 == 0:  # Sample every 10th parameter
			param_sample.append(param.data.cpu().numpy().mean())
	hasher.update(str(param_sample).encode())
	return hasher.hexdigest()

def get_max_samples(
		batch_size, 
		N, 
		device, 
		memory_per_sample_mb=100, 
		safety_factor=0.95, 
		verbose=False,
	):
	total_memory_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  
	memory_cap_mb = total_memory_mb * safety_factor
	# Calculate max samples based on memory constraint
	max_samples_memory = int(memory_cap_mb // memory_per_sample_mb)
	# Desired number of samples based on N and batch_size
	desired_max_samples = N * batch_size
	# Use the smaller of the two values to avoid exceeding memory
	max_samples = min(desired_max_samples, max_samples_memory)
	if verbose:
		print(f"Total GPU Memory: {total_memory_mb:.2f} MB")
		print(f"Usable Memory (after safety factor): {memory_cap_mb:.2f} MB")
		print(f"Max Samples (Memory): {max_samples_memory}")
		print(f"Desired Max Samples: {desired_max_samples}")
		print(f"Final Max Samples: {max_samples}")
	return max_samples

def compute_model_embeddings(
		strategy,
		model,
		loader,
		device,
		cache_dir,
		lora_rank=None,
		lora_alpha=None,
		lora_dropout=None
	):
	model.eval()
	embeddings = []
	paths = []
	dataset_name = getattr(loader, 'name', 'unknown_dataset')
	
	cache_file_name = (
		f"{dataset_name}_"
		f"{strategy}_"
		f"{model.__class__.__name__}_"
		f"{re.sub(r'[/@]', '_', model.name)}_"
	)
	if strategy == "lora" and lora_rank is not None and lora_alpha is not None and lora_dropout is not None:
		cache_file_name += (
			f"lora_rank_{lora_rank}_"
			f"lora_alpha_{lora_alpha}_"
			f"lora_dropout_{lora_dropout}_"
		)
	cache_file_name += "embeddings.pt"
	cache_file = os.path.join(cache_dir, cache_file_name)
	
	if os.path.exists(cache_file):
		data = torch.load(
			f=cache_file,
			map_location=device,
			mmap=True  # Memory-mapping for faster loading
		)
		return data['embeddings'], data['image_paths']
	
	for batch_idx, (images, _, _) in enumerate(tqdm(loader, desc=f"Processing {strategy}")):
		images = images.to(device, non_blocking=True)
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
			features = model.encode_image(images)
			features /= features.norm(dim=-1, keepdim=True)
		embeddings.append(features.cpu())  # Save to CPU for portability
		paths.extend([f"batch_{batch_idx}_img_{i}" for i in range(len(images))])
	
	embeddings = torch.cat(embeddings, dim=0)
	torch.save({'embeddings': embeddings, 'image_paths': paths}, cache_file)
	return embeddings.to(device), paths

def get_updated_model_name(
		original_path:str, 
		actual_epochs:int, 
		additional_info: dict=None
	) -> str:

	if not os.path.exists(original_path):
		print(f"Warning: Original model file not found at {original_path}")
		return original_path
	
	# Extract the directory and filename
	directory, filename = os.path.split(original_path)
	
	# Check if the filename already contains actual_epochs
	if f"aeps_{actual_epochs}" in filename:
		print(f"File already contains actual epochs information: {filename}")
		return original_path
	
	if "ieps_" in filename:
		pattern = r"(ieps_\d+)"
		replacement = f"\\1_aeps_{actual_epochs}"
		new_filename = re.sub(pattern, replacement, filename)
	else:
		base, ext = os.path.splitext(filename)
		new_filename = f"{base}_aeps_{actual_epochs}{ext}"
	
	# Add any additional information to the filename
	if additional_info:
		base, ext = os.path.splitext(new_filename)
		for key, value in additional_info.items():
			# Format numerical values with scientific notation if they're very small
			if isinstance(value, float) and abs(value) < 0.1:
				formatted_value = f"{value:.2e}"
			else:
				formatted_value = str(value)
			base = f"{base}_{key}_{formatted_value}"
		new_filename = f"{base}{ext}"
	
	# Create the new path
	new_path = os.path.join(directory, new_filename)
	
	# rename file
	try:
		os.rename(original_path, new_path)
		print(f"Model saved as: {new_path}")
		return new_path
	except Exception as e:
		print(f"Warning: Could not rename model file: {e}")
		try:
			# Try copying the file instead
			import shutil
			shutil.copy2(original_path, new_path)
			print(f"Model copied to: {new_path}")
			return new_path
		except Exception as e2:
			print(f"Error: Could not copy model file: {e2}")
			return original_path

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
	
	print(f"Adaptive window: {window} | Samples: {n_samples} | Labels: {n_classes}")
	return window

def get_model_directory(path):
	"""
	Extracts the model directory from a given path.
	
	The model directory is defined as the path up to the 'WW_DATASETs' directory.
	
	Parameters:
	path (str): The path to extract the model directory from.
	
	Returns:
	str: The extracted model directory.
	"""
	# Split the path into directories
	directories = path.split(os.sep)
	
	# Find the index of 'WW_DATASETs' in the directories list
	ww_datasets_index = directories.index('WW_DATASETs')
	
	# Construct the model directory by joining all directories up to 'WW_DATASETs'
	model_directory = os.sep.join(directories[:ww_datasets_index])
	model_directory = os.path.join(model_directory, "models")
	return model_directory

def get_parameters_info_original(model, mode):
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
	total_params = sum(p.numel() for p in model.parameters())
	trainable_percent = (trainable_params / total_params) * 100
	frozen_percent = (frozen_params / total_params) * 100
	print(
		f"[{model.__class__.__name__} {model.name} Parameters Statistics] Total: {total_params:,} "
		f"{mode.capitalize()} (Trainablable [Unfrozen]): {trainable_params:,} ({trainable_percent:.2f}%) "
		f"Frozen: {frozen_params:,} ({frozen_percent:.2f}%)"
		.center(170, " ")
	)

def get_parameters_info(model, mode):
		"""
		Prints parameter statistics for a CLIP model, separating image and text encoder parameters.
		
		Args:
				model: The CLIP model instance (e.g., with `visual` and `transformer` attributes).
				mode: String, typically 'train' or 'eval', used in the output message.
		"""
		# Helper function to calculate parameters for a submodule or parameter
		def count_params(item):
				if isinstance(item, torch.nn.Module):
						trainable = sum(p.numel() for p in item.parameters() if p.requires_grad)
						frozen = sum(p.numel() for p in item.parameters() if not p.requires_grad)
						total = sum(p.numel() for p in item.parameters())
				elif isinstance(item, torch.nn.Parameter):
						trainable = item.numel() if item.requires_grad else 0
						frozen = item.numel() if not item.requires_grad else 0
						total = item.numel()
				else:
						raise ValueError(f"Unsupported type in text_submodules: {type(item)}")
				return trainable, frozen, total

		# Total model parameters
		total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
		total_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
		total_params = sum(p.numel() for p in model.parameters())
		total_trainable_percent = (total_trainable / total_params) * 100 if total_params > 0 else 0
		total_frozen_percent = (total_frozen / total_params) * 100 if total_params > 0 else 0

		# Image encoder parameters (assuming 'visual' attribute)
		img_trainable, img_frozen, img_total = count_params(model.visual)
		img_trainable_percent = (img_trainable / img_total) * 100 if img_total > 0 else 0
		img_frozen_percent = (img_frozen / img_total) * 100 if img_total > 0 else 0

		# Text encoder parameters (assuming 'transformer', 'token_embedding', 'ln_final', 'text_projection')
		text_submodules = [model.transformer, model.token_embedding, model.ln_final, model.text_projection]
		text_trainable = sum(count_params(m)[0] for m in text_submodules)
		text_frozen = sum(count_params(m)[1] for m in text_submodules)
		text_total = sum(count_params(m)[2] for m in text_submodules)
		text_trainable_percent = (text_trainable / text_total) * 100 if text_total > 0 else 0
		text_frozen_percent = (text_frozen / text_total) * 100 if text_total > 0 else 0

		# Logit scale (scalar parameter)
		logit_scale_params = model.logit_scale.numel()

		# Print detailed statistics
		print(f"[{model.__class__.__name__} {model.name} Parameters Statistics]".center(160, " "))
		print(f"Image Encoder: Total: {img_total:,}  {mode.capitalize()} (Trainable [Unfrozen]): {img_trainable:,} ({img_trainable_percent:.2f}%)  Frozen: {img_frozen:,} ({img_frozen_percent:.2f}%)".center(160, " "))
		print(f"Text Encoder: Total: {text_total:,}  {mode.capitalize()} (Trainable [Unfrozen]): {text_trainable:,} ({text_trainable_percent:.2f}%)  Frozen: {text_frozen:,} ({text_frozen_percent:.2f}%)".center(160, " "))
		print(f"Logit Scale: {logit_scale_params:,}".center(160, " "))
		print(f"Total Model: Total: {total_params:,}  {mode.capitalize()} (Trainable [Unfrozen]): {total_trainable:,} ({total_trainable_percent:.2f}%)  Frozen: {total_frozen:,} ({total_frozen_percent:.2f}%)".center(160, " "))

def print_loader_info(loader, batch_size):
		loader_num_samples = len(loader.dataset)
		per_batch_samples = loader_num_samples // batch_size
		last_batch_samples = loader_num_samples % batch_size
		if last_batch_samples == 0:
				last_batch_samples = batch_size
		
		# Try multiple ways to get class information
		try:
				# Case 1: Standard PyTorch dataset
				class_names = loader.dataset.classes
		except AttributeError:
				try:
						# Case 2: Subset or wrapped dataset
						class_names = loader.dataset.dataset.classes
				except AttributeError:
						try:
								# Case 3: Our custom attribute
								class_names = loader.dataset.unique_labels
						except AttributeError:
								# Case 4: Multi-label dataset with label_dict
								if hasattr(loader.dataset, 'label_dict'):
										class_names = sorted(loader.dataset.label_dict.keys())
								else:
										class_names = ["unknown"]
		
		n_classes = len(class_names)
		total_samples_calc = per_batch_samples * batch_size + last_batch_samples
		
		# Get loader name safely
		loader_name = getattr(loader, 'name', 'UNNAMED_LOADER')
		
		print(
				f"\n{loader_name}:\n"
				f"\tWrapped in {len(loader)} batches\n"
				f"\tSamples per batch (total batches: {batch_size}): {per_batch_samples}\n"
				f"\tSamples in last batch: {last_batch_samples}\n"
				f"\tTotal samples: {loader_num_samples} (calculated: {total_samples_calc} = {per_batch_samples} x {batch_size} + {last_batch_samples})\n"
				f"\tUnique Label(s): {n_classes}\n"
		)

def log_gpu_memory(device):
	gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
	gpu_max_mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
	gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
	gpu_mem_free = torch.cuda.memory_allocated(device) / (1024 ** 2)
	gpu_mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
	gpu_mem_utilization = (gpu_mem_allocated / gpu_mem_total) * 100
	print(
		f'[GPU Memory] '
		f'Allocated: {gpu_mem_allocated:.2f} MB '
		f'Max Allocated: {gpu_max_mem_allocated:.2f} MB '
		f'Reserved: {gpu_mem_reserved:.2f} MB '
		f'Free: {gpu_mem_free:.2f} MB '
		f'Total: {gpu_mem_total:.2f} MB '
		f'Utilization: {gpu_mem_utilization:.2f} %'
	)

def plot_lrs_vs_steps(lrs, steps, fpath):
	print(f"LRs[{len(lrs)}]") # num_epochs * chuncks 10 * 348 = 3480  # len(train_data_loader) * args.num_epochs
	print(f"Steps[{len(steps)}]") # num_epochs * chuncks 10 * 348 = 3480 

	print(f"Saving learning rates vs steps in {fpath}")
	plt.figure(figsize=(10, 5))
	plt.plot(steps, lrs , color='b')
	plt.xlabel('Steps (Epochs x Chunks)')
	plt.ylabel('Learning Rates')
	plt.title(f'Learning Rate vs. Step')
	plt.grid(True)
	plt.savefig(fpath)

def get_model_details(
		model, 
		img_size=(3, 224, 224), 
		text_size=(77,), 
		batch_size=1,
	):
	print(f"Model Information Detail".center(150, "-"))
	print("Model Architecture:")
	print(model)
	print("\n" + "="*50 + "\n")
	print("Detailed Model Summary (torchinfo):")
	# Create dummy inputs
	device = next(model.parameters()).device
	dummy_image = torch.zeros((batch_size, *img_size), device=device)
	dummy_text = torch.zeros((batch_size, *text_size), dtype=torch.long, device=device)		
	# Create a dummy mask with the correct shape
	dummy_mask = torch.ones((batch_size, text_size[0]), dtype=torch.bool, device=device)
	try:
			tinfo(model, 
						input_data={
								"image": dummy_image,
								"text": dummy_text,
								"mask": dummy_mask
						},
						depth=5, 
						col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
	except Exception as e:
			print(f"Error in torchinfo: {str(e)}")
			print("Falling back to basic model information...")
	print("\n" + "="*50 + "\n")
	print("Model Parameters:")
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")
	print(f"Non-trainable parameters: {total_params - trainable_params:,}")
	print("\nParameter Details:")
	for name, param in model.named_parameters():
			if param.requires_grad:
					print(f"{name}: {param.numel():,} parameters")
	print("-"*150)

def custom_collate_fn(batch):
	# Filter out the None values from the batch
	batch = [item for item in batch if item is not None]
	# Use default collate function on the filtered batch
	return default_collate(batch)

def print_args_table(args, parser):
	args_dict = vars(args)
	table_data = []
	for key, value in args_dict.items():
		action = parser._option_string_actions.get(f'--{key}') or parser._option_string_actions.get(f'-{key}')
		if action and hasattr(action, 'type') and action.type:
			arg_type = action.type.__name__
		else:
			arg_type = type(value).__name__
		table_data.append([key, value, arg_type])
	print(tabulate.tabulate(table_data, headers=['Argument', 'Value', 'Type'], tablefmt='orgtbl'))

def save_pickle(pkl, fname:str=""):
	print(f"\nSaving {type(pkl)}\n{fname}")
	st_t = time.time()
	if isinstance(pkl, dict):
		with open(fname, mode="w") as f:
			json.dump(pkl, f)
	elif isinstance(pkl, ( pd.DataFrame, pd.Series ) ):
		pkl.to_pickle(path=fname)
	else:
		# with open(fname , mode="wb") as f:
		with gzip.open(fname , mode="wb") as f:
			dill.dump(pkl, f)
	elpt = time.time()-st_t
	fsize_dump = os.path.getsize(fname) / 1e6
	print(f"Elapsed_t: {elpt:.3f} s | {fsize_dump:.2f} MB".center(120, " "))

def load_pickle(fpath: str) -> object:
	print(f"Loading {fpath}")
	if not os.path.exists(fpath):
		raise FileNotFoundError(f"File not found: {fpath}")
	start_time = time.time()
	try:
		with open(fpath, mode='r') as f:
			pickle_obj = json.load(f)
	except Exception as exerror:
		# print(f"not a JSON file: {exerror}")
		try:
			with gzip.open(fpath, mode='rb') as f:
				pickle_obj = dill.load(f)
		except gzip.BadGzipFile as ee:
			print(f"Error BadGzipFile: {ee}")
			with open(fpath, mode='rb') as f:
				pickle_obj = dill.load(f)
		except Exception as eee:
			print(f"Error dill: {eee}")
			try:
				pickle_obj = pd.read_pickle(fpath)
			except Exception as err:
				print(f"Error pandas pkl: {err}")
				raise
	elapsed_time = time.time() - start_time
	file_size_mb = os.path.getsize(fpath) / 1e6
	print(f"Elapsed_t: {elapsed_time:.3f} s | {type(pickle_obj)} | {file_size_mb:.3f} MB".center(150, " "))
	return pickle_obj

def get_device_with_most_free_memory():
	if torch.cuda.is_available():
		# print(f"Available GPU(s)| torch = {torch.cuda.device_count()} | CuPy: {cp.cuda.runtime.getDeviceCount()}")
		max_free_memory = 0
		selected_device = 0
		for i in range(torch.cuda.device_count()):
			torch.cuda.set_device(i)
			free_memory = torch.cuda.mem_get_info()[0]
			if free_memory > max_free_memory:
				max_free_memory = free_memory
				selected_device = i
		device = torch.device(f"cuda:{selected_device}")
		print(f"Selected GPU: cuda:{selected_device} with {max_free_memory / 1024**3:.2f} GB free memory")
	else:
		device = torch.device("cpu")
		selected_device = None
		print("No GPU available ==>> using CPU")
	return device, selected_device

def format_elapsed_time(seconds):
	"""
	Convert elapsed time in seconds to DD-HH-MM-SS format.
	"""
	# Create a timedelta object from the elapsed seconds
	elapsed_time = datetime.timedelta(seconds=seconds)
	# Extract days, hours, minutes, and seconds
	days = elapsed_time.days
	hours, remainder = divmod(elapsed_time.seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	# Format the time as DD-HH-MM-SS
	return f"{days:02d}-{hours:02d}-{minutes:02d}-{seconds:02d}"

def measure_execution_time(func):
	"""
	Decorator to measure the execution time of a function.
	"""
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
			start_time = time.time()
			result = func(*args, **kwargs)  # Execute the function and store the result
			end_time = time.time()
			elapsed_time = end_time - start_time
			formatted_time = format_elapsed_time(elapsed_time)
			
			# Get the current stdout
			current_stdout = sys.stdout
			
			# Print to both log file and original stdout
			message = f"function {func.__name__} elapsed time(DD-HH-MM-SS): \033[92m{formatted_time}\033[0m"
			print(message)  # This goes to the current stdout (log file if redirected)
			
			# If stdout is redirected, also print to the original stdout (console)
			if current_stdout != sys.__stdout__:
					print(message, file=sys.__stdout__)
					
			return result
	return wrapper

def set_seeds(seed:int=42, debug:bool=False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if debug: # slows down training but ensures reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
			torch.use_deterministic_algorithms(True, warn_only=True)

def get_config(architecture: str, dropout: float=0.0) -> dict:
	configs = {
		"RN50": {
			"embed_dim": 1024,
			"image_resolution": 224,
			"vision_layers": (3, 4, 6, 3),  # (stage1, stage2, stage3, stage4)
			"vision_width": 64,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN101": {
			"embed_dim": 1024,
			"image_resolution": 224,
			"vision_layers": (3, 4, 23, 3),
			"vision_width": 64,
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x4": {
			"embed_dim": 640,
			"image_resolution": 288,
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 256,  # 4× width
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x16": {
			"embed_dim": 768,
			"image_resolution": 384,
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 1024,  # 16× width
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"RN50x64": {
			"embed_dim": 1024,
			"image_resolution": 448,
			"vision_layers": (3, 4, 6, 3),
			"vision_width": 4096,  # 64× width
			"vision_patch_size": None,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-B/32": {
			"embed_dim": 512,
			"image_resolution": 224,
			"vision_layers": 12,  # transformer layers
			"vision_width": 768,
			"vision_patch_size": 32,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-B/16": {
			"embed_dim": 512,
			"image_resolution": 224,
			"vision_layers": 12,
			"vision_width": 768,
			"vision_patch_size": 16,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 512,
			"transformer_heads": 8,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-L/14": {
			"embed_dim": 768,
			"image_resolution": 224,
			"vision_layers": 24,  # deeper transformer
			"vision_width": 1024,
			"vision_patch_size": 14,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 768,
			"transformer_heads": 12,
			"transformer_layers": 12,
			"dropout": dropout,
		},
		"ViT-L/14@336px": {
			"embed_dim": 768,
			"image_resolution": 336,  # higher resolution variant
			"vision_layers": 24,
			"vision_width": 1024,
			"vision_patch_size": 14,
			"context_length": 77,
			"vocab_size": 49408,
			"transformer_width": 768,
			"transformer_heads": 12,
			"transformer_layers": 12,
			"dropout": dropout,
		}
	}

	if architecture not in configs:
		raise ValueError(f"{architecture} not found! Available models: {list(configs.keys())}")

	return configs[architecture]
