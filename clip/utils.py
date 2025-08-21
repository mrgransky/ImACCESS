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

def translate_state_dict_keys(state_dict, key_mapping):
		"""
		Translate state dict keys to match current model structure.
		
		Args:
				state_dict: The loaded state dictionary
				key_mapping: Dict mapping old prefixes to new prefixes
				
		Returns:
				Dict with translated keys
				
		Example:
				key_mapping = {
						'clip_model.': 'clip.',
						'probe.clip_model.': 'clip.',
				}
		"""
		new_state_dict = {}
		for old_key, value in state_dict.items():
				new_key = old_key
				for old_prefix, new_prefix in key_mapping.items():
						if old_key.startswith(old_prefix):
								new_key = old_key.replace(old_prefix, new_prefix, 1)
								break
				new_state_dict[new_key] = value
		return new_state_dict

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
		head_threshold = 5000,  # Labels with frequency > 5000
		tail_threshold = 1000,  # Labels with frequency < 1000
		save_path="head_torso_tail_grid.png"
	):
	print(f"Analyzing Label Distribution from {metadata_path}")

	# 1. Load DataFrames
	try:
		df_full = pd.read_csv(metadata_path)
		df_train = pd.read_csv(metadata_train_path)
		df_val = pd.read_csv(metadata_val_path)
	except FileNotFoundError as e:
		print(f"Error loading metadata files: {e}")
		return None, None

	# Use the 'label' column for string labels as used in plotting and potentially queries
	# Use 'label_int' for analysis requiring unique integer counts if necessary,
	# but counts based on string labels from the full dataset match Figure 2.

	# 2. In-depth Analysis of Head/Torso/Tail
	label_counts_full = df_full['label'].value_counts()
	total_unique_labels_full = len(label_counts_full)
	print(f"Total unique labels in full dataset: {total_unique_labels_full}")
	print(f"Label Counts (full dataset): \n{label_counts_full.head(10)}")
	print("...")
	print(f"{label_counts_full.tail(10)}")

	head_labels = label_counts_full[label_counts_full > head_threshold].index.tolist()
	tail_labels = label_counts_full[label_counts_full < tail_threshold].index.tolist()
	torso_labels = label_counts_full[(label_counts_full >= tail_threshold) & (label_counts_full <= head_threshold)].index.tolist()
	print(f"\n--- Distribution Segments (based on full dataset frequency > {head_threshold} (Head), < {tail_threshold} (Tail)) ---")
	print(f"Head Segment ({len(head_labels)} labels): {head_labels[:min(10, len(head_labels))]}...")
	print(f"Torso Segment ({len(torso_labels)} labels): {torso_labels[:min(10, len(torso_labels))]}...")
	print(f"Tail Segment ({len(tail_labels)} labels): {tail_labels[:min(10, len(tail_labels))]}...")

	# 3. Select Samples from Validation Set
	print(f"\n--- Selecting {num_samples_per_segment} Samples from Validation Set for Each Segment ---")
	i2t_queries = []
	t2i_queries = []
	# Get labels actually present in the validation set
	labels_in_val = df_val['label'].unique().tolist()
	# Filter segment labels to include only those present in validation for sampling
	head_labels_in_val = [lbl for lbl in head_labels if lbl in labels_in_val]
	torso_labels_in_val = [lbl for lbl in torso_labels if lbl in labels_in_val]
	tail_labels_in_val = [lbl for lbl in tail_labels if lbl in labels_in_val]

	print(f"Head labels available in validation: {len(head_labels_in_val)}")
	print(f"Torso labels available in validation: {len(torso_labels_in_val)}")
	print(f"Tail labels available in validation: {len(tail_labels_in_val)}")
	
	# Check if enough labels/samples exist for sampling
	if (
		len(head_labels_in_val) < num_samples_per_segment 
		or len(torso_labels_in_val) < num_samples_per_segment
		or len(tail_labels_in_val) < num_samples_per_segment
	):
		print("\nWarning: Not enough unique labels available in validation for one or more segments to select the requested number of samples.")
		print("Adjusting sampling to available labels...")
		print("Please consider increasing the validation set size or adjusting the sampling strategy.")
		print("Continuing with available labels...")
		print(f"Head: {len(head_labels_in_val)} | Torso: {len(torso_labels_in_val)} | Tail: {len(tail_labels_in_val)}")
	
	segments = {
		'Head': head_labels_in_val, 
		'Torso': torso_labels_in_val, 
		'Tail': tail_labels_in_val
	}

	# Sample for I2T (Query Image -> Text Labels)
	print("\n--- I2T Query Samples (Image Path + GT Label) ---")
	for segment_name, segment_labels in segments.items():
		if not segment_labels:
			print(f"No {segment_name} labels in validation set. Skipping I2T sampling for this segment.")
			continue
		# Sample *labels* from the segment that are in the validation set
		labels_to_sample_from = random.sample(segment_labels, min(num_samples_per_segment, len(segment_labels)))
		print(f"\nSelected {min(num_samples_per_segment, len(segment_labels))} {segment_name} labels for I2T image sampling:")
		for label in labels_to_sample_from:
			# Get all images with this label in the validation set
			images_for_label = df_val[df_val['label'] == label]['img_path'].tolist()
			if images_for_label:
				# Sample one image path for this label
				sampled_img_path = random.choice(images_for_label)
				i2t_queries.append({'image_path': sampled_img_path, 'label': label, 'segment': segment_name})
				print(f"- Label: '{label}' ({len(images_for_label)} samples in val) -> Image: {sampled_img_path}")
			else:
				print(f"- Warning: No images found for label '{label}' in the validation set for I2T query.")

	# Sample for T2I (Query Label -> Images)
	print("\n--- T2I Query Samples (Label String) ---")
	for segment_name, segment_labels in segments.items():
		if not segment_labels:
			print(f"No {segment_name} labels in validation set. Skipping T2I sampling for this segment.")
			continue
		# Sample *label strings* from the segment that are in the validation set
		labels_to_sample = random.sample(segment_labels, min(num_samples_per_segment, len(segment_labels)))
		print(f"\nSelected {min(num_samples_per_segment, len(segment_labels))} {segment_name} labels for T2I query:")
		for label in labels_to_sample:
			# Check if the label actually exists in the validation set (should be true if sampled from segment_labels_in_val)
			# And ideally, check if there's at least one image for it in the validation set
			if label in df_val['label'].values:
				images_for_label = df_val[df_val['label'] == label]['img_path'].tolist()
				if images_for_label:
					t2i_queries.append({'label': label, 'segment': segment_name})
					print(f"- Label: '{label}' ({len(images_for_label)} samples in val)")
				else:
					print(f"- Warning: Label '{label}' found in val labels, but no images. Skipping T2I query.")
			else:
				print(f"- Warning: Label '{label}' not found in validation set for T2I query. Skipping.") # Should not happen with segment_labels_in_val

	return i2t_queries, t2i_queries

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

def get_probe_params(path_string: str) -> dict:
		params = {}
		
		# Long convention: probe_dropout_(\d+\.\d+), max_epochs_(\d+), patience_(\d+), etc.
		probe_dropout_long = re.search(r"_probe_dropout_(\d+\.\d+)", path_string)
		max_epochs_long   = re.search(r"max_epochs_(\d+)", path_string)
		patience_long     = re.search(r"patience_(\d+)", path_string)
		
		if probe_dropout_long:
				params["probe_dropout"] = float(probe_dropout_long.group(1))
		if max_epochs_long:
				params["max_epochs"] = int(max_epochs_long.group(1))
		if patience_long:
				params["patience"] = int(patience_long.group(1))
		
		# Short convention: _pdo_(\d+\.\d+)_ , _mep_(\d+)_ , _pat_(\d+)_ , etc.
		probe_dropout_short = re.search(r"_pdo_(\d+\.\d+)_", path_string)
		max_epochs_short    = re.search(r"_mep_(\d+)_", path_string)
		patience_short      = re.search(r"_pat_(\d+)_", path_string)
		min_delta_short     = re.search(r"_mdt_(\d+\.\d+e?-?\d*)_", path_string)
		cooldown_short      = re.search(r"_cdt_(\d+\.\d+e?-?\d*)_", path_string)
		val_threshold_short = re.search(r"_vt_(\d+\.\d+)_", path_string)
		stop_threshold_short= re.search(r"_st_(\d+\.\d+e?-?\d*)_", path_string)
		pos_importance_short= re.search(r"_pit_(\d+\.\d+e?-?\d*)_", path_string)
		lr_short            = re.search(r"_lr_(\d+\.\d+e?-?\d*)_", path_string)
		wd_short            = re.search(r"_wd_(\d+\.\d+e?-?\d*)_", path_string)
		bs_short            = re.search(r"_bs_(\d+)_", path_string)
		hdim_short          = re.search(r"_hdim_([A-Za-z0-9]+)_", path_string)
		
		if "probe_dropout" not in params and probe_dropout_short:
				params["probe_dropout"] = float(probe_dropout_short.group(1))
		if "max_epochs" not in params and max_epochs_short:
				params["max_epochs"] = int(max_epochs_short.group(1))
		if "patience" not in params and patience_short:
				params["patience"] = int(patience_short.group(1))
		if min_delta_short:
				params["min_delta"] = float(min_delta_short.group(1))
		if cooldown_short:
				params["cooldown"] = float(cooldown_short.group(1))
		if val_threshold_short:
				params["val_threshold"] = float(val_threshold_short.group(1))
		if stop_threshold_short:
				params["stop_threshold"] = float(stop_threshold_short.group(1))
		if pos_importance_short:
				params["pos_importance"] = float(pos_importance_short.group(1))
		if lr_short:
				params["learning_rate"] = float(lr_short.group(1))
		if wd_short:
				params["weight_decay"] = float(wd_short.group(1))
		if bs_short:
				params["batch_size"] = int(bs_short.group(1))
		if hdim_short:
				params["hidden_dim"] = None if hdim_short.group(1).lower() == "none" else int(hdim_short.group(1))
		
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

# CORRECT SOLUTION: Modify compute_model_embeddings() to handle different model types

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
    """
    Compute embeddings for different types of fine-tuned models.
    This function now properly handles:
    - Regular CLIP models (pretrained, full, progressive)
    - LoRA models 
    - Linear probe models
    """
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
            mmap=True
        )
        return data['embeddings'], data['image_paths']
    
    # Determine how to extract embeddings based on model type
    def get_image_embeddings(model, images):
        """Extract image embeddings handling different model types"""
        
        # Check if this is a linear probe model
        if hasattr(model, 'clip_model') and hasattr(model, 'probe'):
            # This is a linear probe - use the frozen CLIP encoder
            # This will show that linear probe produces same embeddings as pretrained
            return model.clip_model.encode_image(images)
            
        # Check if this is a standard CLIP model (pretrained, full, progressive, LoRA)
        elif hasattr(model, 'encode_image'):
            # This is a regular CLIP model (could be modified by LoRA, full, progressive)
            return model.encode_image(images)
            
        # Handle wrapper classes or other custom model types
        elif hasattr(model, 'visual') and hasattr(model.visual, '__call__'):
            # Fallback: try to use visual encoder directly
            return model.visual(images)
            
        else:
            raise AttributeError(
                f"Model of type {type(model)} doesn't have a recognizable image encoding method. "
                f"Expected 'encode_image' method or 'clip_model.encode_image' for probe models."
            )
    
    for batch_idx, (images, _, _) in enumerate(tqdm(loader, desc=f"Processing {strategy}")):
        images = images.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
            # Use the appropriate embedding extraction method
            features = get_image_embeddings(model, images)
            features /= features.norm(dim=-1, keepdim=True)
        embeddings.append(features.cpu())
        paths.extend([f"batch_{batch_idx}_img_{i}" for i in range(len(images))])
    
    embeddings = torch.cat(embeddings, dim=0)
    torch.save({'embeddings': embeddings, 'image_paths': paths}, cache_file)
    return embeddings.to(device), paths

def compute_model_embeddings_old(
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
