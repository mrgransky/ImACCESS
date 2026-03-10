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
import scipy
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
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

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

USER = os.getenv('USER')
CLUSTER = os.environ.get('SLURM_CLUSTER_NAME', "local")
HOST = platform.node()

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str,
}

def extract_per_k_metrics(eval_result: Dict) -> Dict:
		"""
		Extract per-K mAP and Recall from the return dict of any evaluation function.
		Works with zero_shot_multi_label, evaluate_best_model, probe, etc.

		Returns:
				{
					"i2t": {"overall": {"mAP": {k: v}, "Recall": {k: v}},
									"head":    {"mAP": {k: v}, "Recall": {k: v}},
									"rare":    {"mAP": {k: v}, "Recall": {k: v}}},
					"t2i": { ... same structure ... }
				}
		"""
		out = {}
		for direction, tier_key in [("i2t", "tiered_i2t"), ("t2i", "tiered_t2i")]:
				tiered = eval_result.get(tier_key, {})
				out[direction] = {}
				for tier in ("overall", "head", "rare"):
						tier_data = tiered.get(tier, {})
						out[direction][tier] = {
								"mAP":    {str(k): float(v) for k, v in tier_data.get("mAP",    {}).items()},
								"Recall": {str(k): float(v) for k, v in tier_data.get("Recall", {}).items()},
						}
		return out

def append_retrieval_results(
	tiered_i2t: Dict,
	tiered_t2i: Dict,
	strategy_name: str,
	results_dir: str,
	dataset_name: str,
	verbose: bool = True,
) -> str:
	per_k = extract_per_k_metrics({"tiered_i2t": tiered_i2t, "tiered_t2i": tiered_t2i})
	results_json_path = os.path.join(results_dir, f"{dataset_name}_retrieval_metrics_accumulated.json")
	accumulated = {}
	
	if os.path.exists(results_json_path):
		if verbose:
			print(f"Loading existing results from {results_json_path}")
		with open(results_json_path) as f:
			accumulated = json.load(f)
	
	accumulated[strategy_name] = per_k
	
	with open(results_json_path, "w") as f:
		json.dump(accumulated, f, indent=2)
	
	if verbose:
		collected_methods = list(accumulated.keys())
		n_methods = len(collected_methods)
		print(f"\nResults of strategy '{strategy_name}' appended to {results_json_path}")
		print(f">> {n_methods} collected method(s): {collected_methods}")
		print("="*140)
	
	return results_json_path

def compute_slope(window: List[float]) -> float:
	if len(window) < 2:
		return 0.0
	x = np.arange(len(window))
	y = np.asarray(window)
	# slope = cov(x, y) / var(x)
	var_x = np.var(x)
	return np.cov(x, y, bias=True)[0, 1] / var_x

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
				formatted_value = f"{value:.1e}"
			else:
				formatted_value = str(value)
			base = f"{base}_{key}_{formatted_value}"
		new_filename = f"{base}{ext}"
	
	# Create the new path
	new_path = os.path.join(directory, new_filename)
	
	# rename file
	try:
		os.rename(original_path, new_path)
		# print(f"Model saved as: {new_path}")
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

def get_parameters_info(model, mode):
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
	print(f"\n{model.__class__.__name__} {model.name} Parameters Statistics")
	print(f"   ├─ Fine-tuning Method: {mode.upper()}")
	print(f"   ├─ Image Encoder: Total: {img_total:,} (Trainable [Unfrozen]): {img_trainable:,} ({img_trainable_percent:.3f}%)  Frozen: {img_frozen:,} ({img_frozen_percent:.3f}%)")
	print(f"   ├─ Text Encoder: Total: {text_total:,} (Trainable [Unfrozen]): {text_trainable:,} ({text_trainable_percent:.3f}%)  Frozen: {text_frozen:,} ({text_frozen_percent:.3f}%)")
	print(f"   ├─ Logit Scale: {logit_scale_params}")
	print(f"   └─ Total: {total_params:,}  (Trainable [Unfrozen]): {total_trainable:,} ({total_trainable_percent:.3f}%)  Frozen: {total_frozen:,} ({total_frozen_percent:.3f}%)")

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

def set_seeds(seed: int = 42, debug: bool = False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	# Missing these critical settings:
	torch.backends.cudnn.deterministic = True  # Should always be True for reproducibility
	torch.backends.cudnn.benchmark = False     # Should always be False for reproducibility
	
	# Set environment variables for additional reproducibility
	os.environ['PYTHONHASHSEED'] = str(seed)
	os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA operations
	
	if debug:
		torch.use_deterministic_algorithms(True, warn_only=True)

# def cleanup_embedding_cache(
# 		dataset_name: str,
# 		cache_dir: str,
# 		finetune_strategy: str,
# 		batch_size: int,
# 		model_name: str,
# 		model_arch: str,
# 		num_workers: int,
# 	):
# 	base_name = os.path.join(
# 			cache_dir,
# 			f"{dataset_name}_"
# 			f"{finetune_strategy}_"
# 			f"bs_{batch_size}_"
# 			f"nw_{num_workers}_"
# 			f"{model_name}_"
# 			f"{re.sub(r'[/@]', '_', model_arch)}_"
# 			f"validation_embeddings"
# 	)
# 	cache_files = glob.glob(f"{base_name}.pt") + glob.glob(f"{base_name}_*.pt")
# 	if cache_files:
# 		print(f"Found {len(cache_files)} cache file(s) to clean up.")
# 		for cache_file in cache_files:
# 			try:
# 				os.remove(cache_file)
# 				print(f"Successfully removed cache file: {cache_file}")
# 			except Exception as e:
# 				print(f"Warning: Failed to remove cache file {cache_file}: {e}")
# 	else:
# 		print(f"No cache files found for {base_name}*.pt")

def get_model_hash(model: torch.nn.Module) -> str:
	"""
	Generate a hash of model parameters to detect when model weights have changed.
	This is used to determine if cached embeddings need to be recomputed.
	
	Args:
			model: The model to hash
			
	Returns:
			String hash of model parameters
	"""
	hasher = hashlib.md5()
	# Only hash a subset of parameters for efficiency on very large models
	param_sample = []
	for i, param in enumerate(model.parameters()):
			if i % 10 == 0:  # Sample every 10th parameter
					param_sample.append(param.data.cpu().numpy().mean())  # Just use the mean for speed
	
	hasher.update(str(param_sample).encode())
	return hasher.hexdigest()

def monitor_memory_usage(operation_name: str):
	if torch.cuda.is_available():
		gpu_memory = torch.cuda.memory_allocated() / 1024**3
		gpu_cached = torch.cuda.memory_reserved() / 1024**3
	else:
		gpu_memory = gpu_cached = 0
	cpu_memory = psutil.virtual_memory()
	cpu_used_gb = (cpu_memory.total - cpu_memory.available) / 1024**3
	cpu_percent = cpu_memory.percent
	if cpu_percent > 96:
		print(
			f"[{operation_name}] Memory - CPU Usage: {cpu_used_gb:.1f}GB ({cpu_percent:.1f}%), "
			f"GPU: {gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached"
		)
		print(f"WARNING: High CPU usage ({cpu_percent:.1f}%) → Clearing GPU cache...")
		return True
	return False

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
