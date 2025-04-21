import hashlib
import os
import torch
import clip
import datetime
import time
import json
import re
import argparse
import random
import numpy as np
import pandas as pd
import gzip
import pickle
import dill
import copy
import itertools
import tabulate
import inspect
import functools
import sys
import traceback
from PIL import Image
import requests
from io import BytesIO
import hashlib
from torch.optim import AdamW, SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import multiprocessing

import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.style.use('seaborn-v0_8-whitegrid')  # Modern style for sleek look

from PIL import Image
from typing import Tuple, Union, List, Dict, Any, Optional
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils._pytree')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

def get_lora_params(path_string):
	# Regular expressions to find the parameters
	lora_rank_match = re.search(r"lora_rank_(\d+)", path_string)
	lora_alpha_match = re.search(r"lora_alpha_(\d+\.\d+)", path_string)
	lora_dropout_match = re.search(r"lora_dropout_(\d+\.\d+)", path_string)
	# Extract the values if found
	if lora_rank_match and lora_alpha_match and lora_dropout_match:
		lora_rank = int(lora_rank_match.group(1))
		lora_alpha = float(lora_alpha_match.group(1))
		lora_dropout = float(lora_dropout_match.group(1))
		return {
			"lora_rank": lora_rank,
			"lora_alpha": lora_alpha,
			"lora_dropout": lora_dropout
		}
	else:
		return None  # Return None if any parameter is not found

def get_max_samples(batch_size, N, device, memory_per_sample_mb=100, safety_factor=0.95, verbose=False):
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
	):
	model.eval()
	embeddings = []
	paths = []
	dataset_name = getattr(loader, 'name', 'unknown_dataset')
	cache_file = os.path.join(
		cache_dir, 
		f"{dataset_name}_"
		f"{strategy}_"
		f"{model.__class__.__name__}_"
		f"{re.sub(r'[/@]', '_', model.name)}_"
		f"embeddings.pt"
	)

	if os.path.exists(cache_file):
		data = torch.load(
			f=cache_file, 
			map_location=device, 
			mmap=True, # Memory-mapping for faster loading
		)
		return data['embeddings'], data['image_paths']
	
	print(f"\tStrategy: {strategy}")
	for batch_idx, (images, _, _) in enumerate(tqdm(loader, desc=f"Processing {strategy}")):
		images = images.to(device, non_blocking=True)
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
			features = model.encode_image(images)
			features /= features.norm(dim=-1, keepdim=True)
		embeddings.append(features.cpu())  # Still save to CPU for portability
		paths.extend([f"batch_{batch_idx}_img_{i}" for i in range(len(images))])
	
	embeddings = torch.cat(embeddings, dim=0)
	torch.save({'embeddings': embeddings, 'image_paths': paths}, cache_file)
	return embeddings.to(device), paths

def get_updated_model_name(original_path:str, actual_epochs:int, additional_info: dict=None) -> str:
	if not os.path.exists(original_path):
		print(f"Warning: Original model file not found at {original_path}")
		return original_path
	
	# Extract the directory and filename
	directory, filename = os.path.split(original_path)
	
	# Check if the filename already contains actual_epochs
	if f"actual_epochs_{actual_epochs}" in filename:
		print(f"File already contains actual epochs information: {filename}")
		return original_path
	
	# Replace 'ieps_X' with 'ieps_X_actual_eps_Y'
	if "ieps_" in filename:
		pattern = r"(ieps_\d+)"
		replacement = f"\\1_actual_eps_{actual_epochs}"
		new_filename = re.sub(pattern, replacement, filename)
	else:
		base, ext = os.path.splitext(filename)
		new_filename = f"{base}_actual_eps_{actual_epochs}{ext}"
	
	# Add any additional information to the filename
	if additional_info:
		base, ext = os.path.splitext(new_filename)
		for key, value in additional_info.items():
			# Format numerical values with scientific notation if they're very small
			if isinstance(value, float) and abs(value) < 0.01:
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
	
	print(f"Adaptive window: {window} | Samples: {n_samples} | Classes: {n_classes}")
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
	
	try:
		class_names = loader.dataset.dataset.classes
	except:
		class_names = loader.dataset.unique_labels
	n_classes = len(class_names)

	total_samples_calc = per_batch_samples * batch_size + last_batch_samples
	print(
		f"\n{loader.name} Loader:\n"
		f"\tWrapped in {len(loader)} batches\n"
		f"\tSamples per batch (total batches: {batch_size}): {per_batch_samples}\n"
		f"\tSamples in last batch: {last_batch_samples}\n"
		f"\tTotal samples: {loader_num_samples} (calculated: {total_samples_calc} = {per_batch_samples} x {batch_size} + {last_batch_samples})\n"
		f"\tUnique classes: {n_classes}\n"
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