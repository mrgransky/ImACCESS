import re
import os
import sys
import time
import json
import textwrap
import numpy as np
import pandas as pd
import torch
import threading
import queue
import pickle
import multiprocessing
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm
import urllib.request
import urllib.parse
import argparse
import seaborn as sns
from typing import Tuple, Union, List, Dict, Any, Optional, Callable, TypedDict, Set
import certifi
import networkx as nx
import scipy
import hashlib
from torch.utils.data import Dataset, DataLoader, TensorDataset
import huggingface_hub
import io
import pprint
import itertools
import string
import math
import unicodedata
import requests
import dill
import gzip
import random
import datetime
import logging
import glob
import psutil  # For memory usage monitoring
import tabulate
import ast
import csv
import httpx
import gc
import joblib
import inspect
import warnings
import traceback
import builtins
import platform
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
import concurrent.futures
warnings.filterwarnings('ignore')

# from skimage.filters.rank import entropy
# from skimage.morphology import disk
# from skimage.measure import shannon_entropy
# from skimage.transform import resize
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, precision_recall_curve, roc_curve, auc, f1_score, hamming_loss
from sklearn.neighbors import NearestNeighbors

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError
from requests.exceptions import RequestException
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import functools

from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from tqdm import tqdm
from datetime import timedelta
from pathlib import Path
from natsort import natsorted
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
	import misc.visualize as viz  # For visualizations
except ImportError:
	try:
		import visualize as viz  # For visualizations when running from misc/ directory
	except ImportError:
		viz = None  # Fallback if visualize module is not available

Image.MAX_IMAGE_PIXELS = None  # Disable the limit completely [decompression bomb]

nltk_modules = [
	'punkt',
	'punkt_tab',
	'wordnet',
	'averaged_perceptron_tagger',
	'averaged_perceptron_tagger_eng',
	'omw-1.4',
	'stopwords',
]

# check if nltk_data exists:
try:
	nltk.data.find('tokenizers/punkt')
except LookupError:
	print("Downloading NLTK data...")
	# Download only the required components
	nltk.download(
		nltk_modules,
		quiet=False,
		raise_on_error=True,
	)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER

hf_tk: str = os.getenv("HUGGINGFACE_TOKEN")
anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY")

cache_directory = {
	"farid": "/home/farid/datasets/models",
	"alijanif": "/scratch/project_2004072/models",
	"ubuntu": "/media/volume/models",
}

os.environ["HF_HOME"] = cache_directory[USER]
os.environ["TRANSFORMERS_CACHE"] = cache_directory[USER]
os.environ["HF_HUB_CACHE"] = cache_directory[USER]
os.environ["HF_DATASETS_CACHE"] = cache_directory[USER]
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

import transformers as tfs

dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str,
}

def extract_per_k_metrics(eval_result: Dict, tier_key: str) -> Dict:
	"""
	Extract per-K mAP and Recall from a tiered-metrics dict pair stored
	under `eval_result[f"{tier_key}_i2t"]` / `eval_result[f"{tier_key}_t2i"]`.

	Works for both the ordinary run-specific tiers (`tier_key="tiered"`)
	and the R1-C shared-vocabulary tiers (`tier_key="shared_tiered"`).

	Returns None if the underlying tiered dicts are missing/None (e.g.
	shared_tiered_* when shared_protocol_path was not provided to
	evaluate_best_model).

	Returns:
			{
				"i2t": {"overall": {"mAP": {k: v}, "Recall": {k: v}},
								"head":    {"mAP": {k: v}, "Recall": {k: v}},
								"rare":    {"mAP": {k: v}, "Recall": {k: v}}},
				"t2i": { ... same structure ... }
			}
	"""
	i2t_tiered = eval_result.get(f"{tier_key}_i2t")
	t2i_tiered = eval_result.get(f"{tier_key}_t2i")

	if i2t_tiered is None or t2i_tiered is None:
		return None

	out = {}
	for direction, tiered in [("i2t", i2t_tiered), ("t2i", t2i_tiered)]:
		out[direction] = {}
		for tier in ("overall", "head", "rare"):
			tier_data = tiered.get(tier, {})
			out[direction][tier] = {
				"mAP":    {str(k): float(v) for k, v in tier_data.get("mAP",    {}).items()},
				"Recall": {str(k): float(v) for k, v in tier_data.get("Recall", {}).items()},
			}

	return out

def clean_cache(directory: str, strategy: str, verbose: bool = False):
	###########################################################################################
	# Temporary due to lack of disk space
	# Clean up any available JSON/PT files before finishing
	json_files = glob.glob(os.path.join(directory, "*.json"))
	pt_files = glob.glob(os.path.join(directory, "*.pt"))
	pth_files = glob.glob(os.path.join(directory, "*.pth"))
	cleanup_files = (
		pt_files 
		+ pth_files
		# + json_files
	)
	if cleanup_files:
		print(f"Cleaning cache in {directory}")
		for f in cleanup_files:
			if os.path.basename(f).startswith(f"{strategy}_ViT"):
				print(f">> Removing {f}")
				try:
					os.remove(f)
				except Exception as e:
					print(f"Warning: Failed to remove {f}: {e}")
	###########################################################################################

def save_tiered_retrieval_metrics(
	best_model_result: Dict,
	strategy: str,
	dataset_directory: str,
	column: str,
	seed: int,
	verbose: bool = True,
):
	output_dir = os.path.join(dataset_directory, "outputs")
	os.makedirs(output_dir, exist_ok=True)

	result_dir = os.path.join(output_dir, column, f"seed_{seed}")
	os.makedirs(result_dir, exist_ok=True)

	per_k = extract_per_k_metrics(best_model_result, tier_key="tiered")
	shared_per_k = extract_per_k_metrics(best_model_result, tier_key="shared_tiered")

	combined = {"standard": per_k}
	if shared_per_k is not None:
		combined["shared"] = shared_per_k

	retrieval_tiered_fpath = os.path.join(result_dir, f"retrieval_metrics_accumulated.json")
	retrieval_accumulated = {}
	if os.path.exists(retrieval_tiered_fpath):
		if verbose:
			print(f"Loading existing results from {retrieval_tiered_fpath}")
		with open(retrieval_tiered_fpath) as f:
			retrieval_accumulated = json.load(f)

	retrieval_accumulated[strategy] = combined

	with open(retrieval_tiered_fpath, "w") as f:
		json.dump(retrieval_accumulated, f, indent=2)

	performance_fpath = os.path.join(output_dir, f"seed_{seed}_performance.json")
	performance_accumulated = {}
	if os.path.exists(performance_fpath):
		if verbose:
			print(f"Loading existing results from {performance_fpath}")
		with open(performance_fpath) as f:
			performance_accumulated = json.load(f)

	# Ensure column key exists
	if column not in performance_accumulated:
		performance_accumulated[column] = {}

	performance_accumulated[column][strategy] = combined

	with open(performance_fpath, "w") as f:
		json.dump(performance_accumulated, f, indent=2)

	if verbose:
		# print("="*120)
		# print(strategy.upper())
		# print(json.dumps(combined, indent=2, ensure_ascii=False))

		print(f"\nRetrieval Tiered Metrics:")
		collected_retrieval_methods = list(retrieval_accumulated.keys())
		n_methods = len(collected_retrieval_methods)
		print(f"'{strategy}' strategy results appended to {retrieval_tiered_fpath}")
		print(f">> {n_methods} collected method(s): {collected_retrieval_methods}")
		if shared_per_k is None:
			print(f"[NOTE] No shared-vocabulary protocol results present for this run "
					f"(shared_protocol_path was not provided to evaluate_best_model)")
		print(f"\nPerformance Metrics:")
		collected_columns = list(performance_accumulated.keys())
		n_columns = len(collected_columns)
		print(f"'{column}' column results appended to {performance_fpath}")
		print(f">> {n_columns} collected column(s): {collected_columns}")
		print("="*120)

def check_and_cleanup_gpu_memory(
	batch_idx: int,
	reserved_threshold: int = 95,
	allocated_threshold: int = 80,
	verbose: bool = False
) -> bool:
	"""
	Monitor GPU memory usage and trigger cleanup if needed.
	
	Args:
		batch_idx: Current batch number (for logging)
		reserved_threshold: Trigger cleanup if reserved memory exceeds this % (default: 95)
		allocated_threshold: Trigger cleanup if allocated memory exceeds this % (default: 80)
		verbose: Print detailed memory stats
	
	Returns:
		True if cleanup was triggered, False otherwise
	"""
	if not torch.cuda.is_available():
		return False
	
	need_cleanup = False
	gpu_memory_stats = []
	gpu_allocated_stats = []
	
	for device_idx in range(torch.cuda.device_count()):
		mem_total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)
		mem_allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
		mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
		
		# Use reserved for "what PyTorch is holding" (includes cached free memory)
		mem_reserved_pct = (mem_reserved / mem_total) * 100 if mem_total > 0 else 0
		# Use allocated for "what we're actually using right now"
		mem_allocated_pct = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0
		
		gpu_memory_stats.append(mem_reserved_pct)
		gpu_allocated_stats.append(mem_allocated_pct)
		
		if verbose:
			print(
				f"[MEM] Batch {batch_idx} (GPU {device_idx}): {mem_reserved_pct:.2f}% reserved / "
				f"{mem_allocated_pct:.2f}% allocated — "
				f"{mem_allocated:.2f}GB alloc / {mem_reserved:.2f}GB reserved (Total: {mem_total:.1f}GB)"
			)
		
		# Only trigger cleanup if BOTH reserved AND allocated are high
		# This avoids clearing PyTorch's intentional cache when actual usage is low
		if mem_reserved_pct > reserved_threshold and mem_allocated_pct > allocated_threshold:
			need_cleanup = True
	
	if need_cleanup:
		avg_reserved = np.mean(gpu_memory_stats)
		max_reserved = max(gpu_memory_stats)

		avg_allocated = np.mean(gpu_allocated_stats)
		max_allocated = max(gpu_allocated_stats)
		
		print(
			f"\n[WARN] High memory usage detected — "
			f"Reserved: max={max_reserved:.1f}%, avg={avg_reserved:.1f}% | "
			f"Allocated: max={max_allocated:.1f}%, avg={avg_allocated:.1f}% "
			f"(thresholds: reserved>{reserved_threshold}%, allocated>{allocated_threshold}%) "
			f"=> Clearing cache..."
		)
		torch.cuda.empty_cache()  # clears all GPUs
		gc.collect()
	
	return need_cleanup

def compute_slope(window: List[float]) -> float:
	if len(window) < 2:
		return 0.0
	x = np.arange(len(window))
	y = np.asarray(window)
	# slope = cov(x, y) / var(x)
	var_x = np.var(x)
	return np.cov(x, y, bias=True)[0, 1] / var_x

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
			shutil.copy2(original_path, new_path)
			print(f"Model copied to: {new_path}")
			return new_path
		except Exception as e2:
			print(f"Error: Could not copy model file: {e2}")
			return original_path

def get_model_hash(model: torch.nn.Module, exact: bool = False) -> str:
	"""
	Hash a model's architecture AND weights for cache invalidation.
	Args:
			model: The model to hash.
			exact: If True, hash the raw bytes of every tensor (slow but 100%
						 accurate). If False, use a fast deterministic sample.
	"""
	hasher = hashlib.md5()
	state = model.state_dict()  # captures BOTH parameters and buffers

	for name, tensor in state.items():
		# --- Architecture signal: name, shape, dtype ---
		hasher.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode())
		# --- Weight signal ---
		t = tensor.detach().cpu().contiguous()
		if exact:
			# Fully accurate: hash raw bytes
			hasher.update(t.numpy().tobytes())
		else:
			# Fast but robust sample: first/last 128 elements + aggregates
			flat = t.flatten()
			n = flat.numel()
			head = flat[:128].numpy().tobytes()
			tail = flat[-128:].numpy().tobytes() if n > 128 else b""
			# Use sum + abs-sum as cheap distribution fingerprints
			agg = f"{flat.sum().item():.9g}|{flat.abs().sum().item():.9g}|{n}\n"
			hasher.update(head)
			hasher.update(tail)
			hasher.update(agg.encode())
	return hasher.hexdigest()

# Parameter-name substrings that are EXPECTED to be all-zero at initialization.
# Extend this tuple if your CLIP-Adapter zero-inits its residual output layer,
# e.g. EXPECTED_ZERO_PATTERNS += ('adapter.out_proj',)
EXPECTED_ZERO_PATTERNS = ('lora_b', 'lambda_b', 'vera_b')

# Class-name keywords used to detect injected adapter modules
ADAPTER_KEYWORDS = ('lora', 'dora', 'vera', 'ia3', 'adapter', 'tip')

GPU_BANDWIDTH_GBPS = {
		'H100': 2039, 'A100': 1555, 'V100': 900,
		'RTX 4090': 1008, 'RTX 3090': 936, 'RTX 4080': 717, 'RTX 3080': 760,
}

def _print_component_status(label, component, is_param=False):
		"""Print freeze status for a single component (module or parameter)."""
		if is_param:
				params = [component] if isinstance(component, torch.nn.Parameter) else [component]
				total = sum(p.numel() for p in params)
				train = sum(p.numel() for p in params if p.requires_grad)
		else:
				params = list(component.parameters())
				if not params:
						return
				total = sum(p.numel() for p in params)
				train = sum(p.numel() for p in params if p.requires_grad)

		frozen = total - train
		if train == total:
				icon = "🔓"
		elif train == 0:
				icon = "🔒"
		else:
				icon = "🔀"
		print(f"    {icon} {label:40s} | {total:>12,} total | {train:>12,} train | {frozen:>12,} frozen")

def get_parameters_info(model, mode, verbose=True, optimizer=None):
		"""
		Comprehensive pre-fine-tuning inspector for CLIP and its adapted variants.

		Covers: full, lora, rslora, lora_plus, dora, vera, ia3,
						clip_adapter_v/t/vt, tip_adapter, tip_adapter_f.
		(zero_shot / probe do not need to call this.)

		Args:
				model: the (possibly adapted) CLIP model.
				mode: strategy label string, e.g. 'lora_plus', 'ia3', 'tip_adapter_f'.
				verbose: extended per-layer breakdowns.
				max_listed_params: cap for the explicit trainable-parameter list.
				optimizer: optional; pass it to verify param-group setup (critical for LoRA+).

		Returns:
				dict with aggregate statistics.
		"""
		named = list(model.named_parameters())
		name_to_p = dict(named)
		mode_l = mode.lower()

		# ──────────────────────────────────────────────
		# 0. Model metadata & memory bandwidth
		# ──────────────────────────────────────────────
		if mode=="lora_plus" and optimizer is None:
			raise ValueError("LoRA+ requires passing the optimizer to verify param-group setup.")

		device = next(model.parameters()).device
		dtypes = defaultdict(int)
		for _, p in named:
			dtypes[str(p.dtype)] += 1

		print(f"\n[INSPECTION] {model.__class__.__name__} {getattr(model, 'name', '?')}")
		print(f"  Mode           : {mode}")
		print(f"  Training mode  : {model.training}  ({'model.train()' if model.training else 'model.eval()'})")
		print(f"  Device         : {device}")
		print(f"  Dtype dist     : {dict(dtypes)}")
		print(f"  Total params   : {sum(p.numel() for _, p in named):,}")
		print(f"  Trainable      : {sum(p.numel() for _, p in named if p.requires_grad):,}")
		print(f"  Frozen         : {sum(p.numel() for _, p in named if not p.requires_grad):,}")

		total_bytes = sum(p.numel() * p.element_size() for _, p in named)
		print(f"  Param memory   : {total_bytes / 1e6:.1f} MB  ({total_bytes / 1e9:.2f} GB)")

		bytes_per_step = total_bytes * 3  # rough forward+backward traffic estimate
		print(f"  Est. BW/step   : {bytes_per_step / 1e9:.2f} GB (forward+backward)")
		if torch.cuda.is_available() and str(device).startswith("cuda"):
				device_name = torch.cuda.get_device_name(device)
				for gpu, bw in GPU_BANDWIDTH_GBPS.items():
						if gpu in device_name:
								print(f"  GPU bandwidth  : ~{bw} GB/s ({device_name})")
								print(f"  Theoretical max steps/s: {bw / (bytes_per_step / 1e9):.1f}")
								break

		# ──────────────────────────────────────────────
		# 1. Model configuration
		# ──────────────────────────────────────────────
		is_vit = hasattr(model, 'visual') and hasattr(model.visual, 'transformer')

		print(f"\n  [Config]")
		cfg_attrs = [
				(model, 'context_length', 'Context length'),
				(model, 'vocab_size',     'Vocab size'),
				(model, 'embed_dim',      'Embed dim'),
		]
		if hasattr(model, 'visual'):
				cfg_attrs += [
						(model.visual, 'input_resolution', 'Image resolution'),
						(model.visual, 'output_dim',       'Vision out dim'),
				]
		for obj, attr, label in cfg_attrs:
				val = getattr(obj, attr, None)
				if val is not None:
						print(f"    {label:20s}: {val}")
		if is_vit and hasattr(model.visual, 'conv1'):
				print(f"    {'Patch size':20s}: {model.visual.conv1.kernel_size}")
		elif hasattr(model, 'visual'):
				print(f"    {'Vision backbone':20s}: ModifiedResNet")

		if 'logit_scale' in name_to_p:
				ls = name_to_p['logit_scale']
				print(f"\n  [Logit scale]")
				print(f"    raw value    : {ls.item():.6f}")
				print(f"    exp (temp)   : {ls.exp().item():.4f}")
				print(f"    requires_grad: {ls.requires_grad}")

		# ──────────────────────────────────────────────
		# 2. Prefix-based grouping (dynamic catch-all)
		# ──────────────────────────────────────────────
		def find(prefixes):
				matched = []
				for pref in prefixes:
						matched += [n for n, _ in named if n.startswith(pref)]
				# dedupe by id (LoRA params inside visual. must not be double-counted)
				seen, uniq = set(), []
				for n in matched:
						pid = id(name_to_p[n])
						if pid not in seen:
								seen.add(pid)
								uniq.append(n)
				tr = sum(name_to_p[n].numel() for n in uniq if name_to_p[n].requires_grad)
				fr = sum(name_to_p[n].numel() for n in uniq if not name_to_p[n].requires_grad)
				return tr, fr, tr + fr, uniq, seen

		visual_prefixes = ['visual.']
		core_text_prefixes = ['transformer.', 'token_embedding', 'positional_embedding', 'ln_final', 'text_projection']

		# DYNAMIC CATCH-ALL: any top-level param group that is not visual / logit_scale / core-text
		# (ia3_text_projection, lora_plus_text_projection, probe heads, adapters, ...)
		text_adapter_prefixes = sorted(set(
				n.split('.')[0] for n in name_to_p
				if not n.startswith('visual.')
				and n != 'logit_scale'
				and not any(n.startswith(p) for p in core_text_prefixes)
		))

		print(f"\n  [Prefix verification]")
		for label, prefs in [('VISUAL', visual_prefixes), ('CORE_TEXT', core_text_prefixes), ('TEXT_ADAPTERS', text_adapter_prefixes)]:
				for pref in prefs:
						hits = [n for n in name_to_p if n.startswith(pref)]
						if hits:
								n_tr = sum(1 for n in hits if name_to_p[n].requires_grad)
								n_fr = len(hits) - n_tr
								print(f"    ✓ '{pref}' -> {len(hits)} params "
											f"({n_tr} train / {n_fr} frozen), "
											f"{sum(name_to_p[n].numel() for n in hits):,} numel")
						else:
								if label != 'TEXT_ADAPTERS':
										print(f"    ✗ '{pref}' -> 0 matches")

		img_tr, img_fr, img_to, _, _ = find(visual_prefixes)
		txt_core_tr, txt_core_fr, txt_core_to, _, _ = find(core_text_prefixes)
		txt_adapter_tr, txt_adapter_fr, txt_adapter_to, text_adapter_list, _ = find(text_adapter_prefixes)

		total_tr = sum(p.numel() for _, p in named if p.requires_grad)
		total_fr = sum(p.numel() for _, p in named if not p.requires_grad)
		total_to = total_tr + total_fr
		logit = name_to_p['logit_scale'].numel() if 'logit_scale' in name_to_p else 0

		text_to = txt_core_to + txt_adapter_to
		text_tr = txt_core_tr + txt_adapter_tr

		# ──────────────────────────────────────────────
		# 3. Per-module breakdown (top-level children)
		# ──────────────────────────────────────────────
		if verbose:
			print(f"\n[Per-module breakdown]")
			for child_name, child_module in model.named_children():
				c_params = list(child_module.parameters())
				if not c_params:
					continue

				c_total = sum(p.numel() for p in c_params)
				c_train = sum(p.numel() for p in c_params if p.requires_grad)
				c_frozen = c_total - c_train
				pct = c_train / c_total * 100 if c_total > 0 else 0

				status = (
					"🔓 TRAINABLE" if c_train == c_total else
					"🔒 FROZEN" if c_train == 0 else f"🔀 PARTIAL ({pct:.3f}%)")

				print(
					f"{child_name:28s}Total: {c_total:<15,} "
					f"Trainable: {c_train:15,} Frozen: {c_frozen:<15,}{status}"
				)

		# ──────────────────────────────────────────────
		# 4. Trainable parameter list
		# ──────────────────────────────────────────────
		trainable_names = [n for n, p in named if p.requires_grad]
		print(f"\n[Trainable parameters] ({len(trainable_names)} tensors)")
		for i, n in enumerate(trainable_names):
			p = name_to_p[n]
			print(f"{i+1:5d}. {n:75s}{str(p.shape):30s}{p.numel():<10,}{str(p.dtype)}")

		# ──────────────────────────────────────────────
		# 4b. Parameter size distribution
		# ──────────────────────────────────────────────
		if verbose:
			print(f"\n[Parameter size distribution]")
			param_sizes = [p.numel() for _, p in named]
			size_buckets = {
				'tiny   (<1K)':      sum(1 for s in param_sizes if s < 1_000),
				'small  (1K-10K)':   sum(1 for s in param_sizes if 1_000 <= s < 10_000),
				'medium (10K-100K)': sum(1 for s in param_sizes if 10_000 <= s < 100_000),
				'large  (100K-1M)':  sum(1 for s in param_sizes if 100_000 <= s < 1_000_000),
				'huge   (>1M)':      sum(1 for s in param_sizes if s >= 1_000_000),
			}
			for bucket, count in size_buckets.items():
				print(f"    {bucket:20s}: {count:4d} params ({count/len(param_sizes)*100:5.1f}%)")

			largest = sorted(named, key=lambda x: x[1].numel(), reverse=True)[:100]
			print(f"\n[Top-{len(largest)} largest parameters]")
			for i, (name, param) in enumerate(largest, 1):
				print(f"{i:5d}. {name:75s}{param.numel():<25,}{tuple(param.shape)}")

		# ──────────────────────────────────────────────
		# 5. Adapter module inspection (LoRA/DoRA/VeRA/IA3/Adapter/Tip)
		# ──────────────────────────────────────────────
		adapter_modules = []
		for name, module in model.named_modules():
			cls_name = module.__class__.__name__.lower()
			if any(k in cls_name for k in ADAPTER_KEYWORDS):
				adapter_modules.append((name, module))

		if adapter_modules:
			print(f"\n[Adapter modules detected] ({len(adapter_modules)})")
			for name, mod in adapter_modules:
				n_params = sum(p.numel() for p in mod.parameters())
				n_train  = sum(p.numel() for p in mod.parameters() if p.requires_grad)
				extra = ""
				if hasattr(mod, 'r'):          extra += f" rank={mod.r}"
				if hasattr(mod, 'lora_alpha'): extra += f" alpha={mod.lora_alpha}"
				if hasattr(mod, 'lora_dropout'):
						drop = mod.lora_dropout
						extra += f" dropout={drop.p if hasattr(drop, 'p') else drop}"
				print(f"    {name:60s} | {n_params:>10,} params | {n_train:>10,} train | {mod.__class__.__name__}{extra}")
			
			# LoRA-family configuration analysis (RSLoRA-aware scaling)
			lora_ranks  = [mod.r for _, mod in adapter_modules if hasattr(mod, 'r')]
			lora_alphas = [mod.lora_alpha for _, mod in adapter_modules if hasattr(mod, 'lora_alpha')]
			if lora_ranks or lora_alphas:
				print(f"\n  [LoRA configuration analysis]")
				if lora_ranks:
					print(
						f"    Rank distribution : min={min(lora_ranks)}, max={max(lora_ranks)}, "
						f"mean={sum(lora_ranks)/len(lora_ranks):.1f}"
					)
				if lora_alphas:
					print(
						f"    Alpha distribution: min={min(lora_alphas)}, max={max(lora_alphas)}, "
						f"mean={sum(lora_alphas)/len(lora_alphas):.1f}"
					)
				if lora_ranks and lora_alphas:
					if 'rslora' in mode_l:
						scalings = [a / math.sqrt(r) for a, r in zip(lora_alphas, lora_ranks) if r > 0]
						label = "alpha/√r"
					else:
						scalings = [a / r for a, r in zip(lora_alphas, lora_ranks) if r > 0]
						label = "alpha/r"
					if scalings:
						print(
							f"    Scaling ({label:8s}): min={min(scalings):.2f}, max={max(scalings):.2f}, "
							f"mean={sum(scalings)/len(scalings):.2f}"
						)

		# ──────────────────────────────────────────────
		# 5b. Strategy-aware initialization contract checks
		# ──────────────────────────────────────────────
		print(f"\n[Strategy contract checks]")
		contract_checked = False

		# IA3: scales MUST start at 1.0 (identity), otherwise step-0 output is corrupted
		if 'ia3' in mode_l:
				contract_checked = True
				ia3_params = [(n, p) for n, p in named if 'ia3' in n.lower() and p.requires_grad]
				if ia3_params:
						n_ones = sum(1 for _, p in ia3_params if torch.allclose(p.detach(), torch.ones_like(p), atol=1e-4))
						if n_ones == len(ia3_params):
								print(f"    ✅ IA3: all {len(ia3_params)} scales initialized to 1.0 (identity preserved)")
						else:
								print(f"    ❌ IA3: only {n_ones}/{len(ia3_params)} scales are 1.0 — model output altered at step 0!")
				else:
						print(f"    ⚠️  IA3 mode but no trainable ia3 parameters found")

		# LoRA family: ΔW = B·A must be zero at init → trainable B matrices must be all-zero
		if any(k in mode_l for k in ('lora', 'rslora', 'dora', 'vera')):
				contract_checked = True
				b_mats = [p for n, p in named if 'lora_b' in n.lower() and p.requires_grad]
				if b_mats:
						n_zero_b = sum(1 for p in b_mats if (p == 0).all())
						if n_zero_b == len(b_mats):
								print(f"    ✅ LoRA family: all {len(b_mats)} trainable B-matrices are zero → ΔW=0 at step 0")
						else:
								print(f"    ❌ LoRA family: {len(b_mats) - n_zero_b}/{len(b_mats)} B-matrices NON-zero — "
											f"pretrained output shifts at step 0")
				else:
						print(f"    ℹ️  No trainable lora_B matrices (expected for VeRA with frozen random matrices)")

		# DoRA: magnitude vectors must equal column-norms of W_pretrained → strictly positive
		if 'dora' in mode_l:
				contract_checked = True
				dora_scales = [(n, p) for n, p in named if 'dora_scale' in n.lower()]
				if dora_scales:
						all_pos = all((p.detach() > 0).all() for _, p in dora_scales)
						print(f"    {'✅' if all_pos else '❌'} DoRA: {len(dora_scales)} magnitude vectors "
									f"{'all positive' if all_pos else 'NOT all positive (must equal ||W|| columns)'}")
				else:
						print(f"    ⚠️  DoRA mode but no 'dora_scale' parameters found")

		# VeRA: λb zero-init, λd small random
		if 'vera' in mode_l:
				contract_checked = True
				lb = [p for n, p in named if 'lambda_b' in n.lower() and p.requires_grad]
				ld = [p for n, p in named if 'lambda_d' in n.lower() and p.requires_grad]
				if lb:
						n_zero = sum(1 for p in lb if (p == 0).all())
						print(f"    {'✅' if n_zero == len(lb) else '⚠️ '} VeRA: {n_zero}/{len(lb)} λb vectors zero-initialized")
				if ld:
						stds = [((p.detach() - p.detach().mean()) ** 2).mean().sqrt().item() for p in ld]
						print(f"    ℹ️  VeRA: {len(ld)} λd vectors, std range [{min(stds):.4f}, {max(stds):.4f}]")

		# Tip-Adapter(-F): adapter values come from the few-shot cache → must NOT be zero
		if 'tip_adapter' in mode_l:
				contract_checked = True
				tip_params = [(n, p) for n, p in named
											if ('adapter' in n.lower() or 'tip' in n.lower()) and p.requires_grad]
				if tip_params:
						n_nonzero = sum(1 for _, p in tip_params if (p != 0).any())
						print(f"    {'✅' if n_nonzero == len(tip_params) else '⚠️ '} Tip-Adapter: "
									f"{n_nonzero}/{len(tip_params)} adapter tensors non-zero (cache-initialized)")
				else:
						print(f"    ⚠️  Tip-Adapter mode but no trainable adapter parameters found")

		# CLIP-Adapter: report zero-init status of the bottleneck (zero output layer is conventional)
		if 'clip_adapter' in mode_l:
				contract_checked = True
				ad_params = [(n, p) for n, p in named if 'adapter' in n.lower() and p.requires_grad]
				if ad_params:
						n_zero = sum(1 for _, p in ad_params if (p == 0).all())
						print(f"    ℹ️  CLIP-Adapter: {len(ad_params)} trainable adapter tensors, "
									f"{n_zero} zero-initialized {'(residual identity init — OK)' if n_zero else ''}")
				else:
						print(f"    ⚠️  CLIP-Adapter mode but no trainable adapter parameters found")

		if not contract_checked:
				print(f"    ℹ️  No family-specific contract for mode '{mode}' (full fine-tuning has no init contract)")

		# ──────────────────────────────────────────────
		# 6 & 7. Layer-level freeze maps (visual + text)
		# ──────────────────────────────────────────────
		if verbose and hasattr(model, 'visual'):
				vis = model.visual
				if is_vit:
						print(f"\n  [Visual Transformer — layer freeze map]")
						_print_component_status("conv1 (patch embed)", vis.conv1)
						if hasattr(vis, 'class_embedding'):
								_print_component_status("class_embedding", vis.class_embedding, is_param=True)
						_print_component_status("positional_embedding", vis.positional_embedding, is_param=True)
						if hasattr(vis, 'ln_pre'):
								_print_component_status("ln_pre", vis.ln_pre)
						for i, block in enumerate(vis.transformer.resblocks):
								_print_component_status(f"resblock[{i}]", block)
						if hasattr(vis, 'ln_post'):
								_print_component_status("ln_post", vis.ln_post)
						if getattr(vis, 'proj', None) is not None:
								_print_component_status("proj", vis.proj, is_param=True)
				else:
						print(f"\n  [Visual ResNet — layer freeze map]")
						for stem in ('conv1', 'conv2', 'conv3'):
								if hasattr(vis, stem):
										_print_component_status(f"stem {stem}", getattr(vis, stem))
						for lname in ('layer1', 'layer2', 'layer3', 'layer4'):
								if hasattr(vis, lname):
										_print_component_status(lname, getattr(vis, lname))
						if hasattr(vis, 'attnpool'):
								_print_component_status("attnpool", vis.attnpool)

		if verbose and hasattr(model, 'transformer'):
				print(f"\n  [Text Transformer — layer freeze map]")
				_print_component_status("token_embedding", model.token_embedding)
				_print_component_status("positional_embedding", model.positional_embedding, is_param=True)
				for i, block in enumerate(model.transformer.resblocks):
						_print_component_status(f"resblock[{i}]", block)
				_print_component_status("ln_final", model.ln_final)
				if model.text_projection is not None:
						_print_component_status("text_projection", model.text_projection, is_param=True)

		# ──────────────────────────────────────────────
		# 8 & 9. Numerical health & gradient flow
		# ──────────────────────────────────────────────
		nan_params, inf_params, expected_zero, unexpected_zero = [], [], [], []
		for n, p in named:
				if p.requires_grad and p.numel() > 0:
						if torch.isnan(p).any(): nan_params.append(n)
						if torch.isinf(p).any(): inf_params.append(n)
						if (p == 0).all():
								if any(pat in n.lower() for pat in EXPECTED_ZERO_PATTERNS):
										expected_zero.append(n)
								else:
										unexpected_zero.append(n)

		print(f"\n[Numerical health check]")
		print(f"    NaN in trainable params  : {'❌ ' + str(nan_params) if nan_params else '✅ None'}")
		print(f"    Inf in trainable params  : {'❌ ' + str(inf_params) if inf_params else '✅ None'}")
		print(f"    Expected zeros (init)    : {len(expected_zero)} params "
					f"{'✅ (LoRA-B / VeRA λb preserve pretrained output at step 0)' if expected_zero else ''}")
		if unexpected_zero:
				print(f"    ⚠️  Unexpected zero params: {len(unexpected_zero)}")
				for i, n in enumerate(unexpected_zero[:10]):
						print(f"        {i+1}. {n}")
				if len(unexpected_zero) > 10:
						print(f"        ... and {len(unexpected_zero) - 10} more")
		else:
				print(f"    Unexpected zero params   : ✅ None")

		print(f"\n  [Gradient flow check]")
		print(f"    requires_grad=True  : {len(trainable_names)} params")
		print(f"    requires_grad=False : {len(named) - len(trainable_names)} params")
		stale_grads = [n for n, p in named if not p.requires_grad and p.grad is not None]
		if stale_grads:
				print(f"    ⚠️  Frozen params with stale .grad: {stale_grads[:5]}")
		else:
				print(f"    No stale gradients on frozen params ✅")

		# ──────────────────────────────────────────────
		# 10. Activation memory estimate (ViT only, rough)
		# ──────────────────────────────────────────────
		if is_vit:
				vis_blocks = len(model.visual.transformer.resblocks)
				text_blocks = len(model.transformer.resblocks) if hasattr(model, 'transformer') else 0
				embed_dim = model.visual.output_dim if hasattr(model.visual, 'output_dim') else 768
				n_patches = (model.visual.input_resolution // model.visual.conv1.kernel_size[0]) ** 2 + 1

				text_activation_mem  = 2 * 32 * 77 * embed_dim * text_blocks * 4 / 1e6
				image_activation_mem = 2 * 32 * n_patches * embed_dim * vis_blocks * 4 / 1e6

				print(f"\n  [Activation memory estimate] (batch=32, approximate)")
				print(f"    Text encoder activations : ~{text_activation_mem:.1f} MB")
				print(f"    Image encoder activations: ~{image_activation_mem:.1f} MB")
				print(f"    Total activations        : ~{text_activation_mem + image_activation_mem:.1f} MB")
				print(f"    ⚠️  Actual memory will be higher due to intermediate states")

		# ──────────────────────────────────────────────
		# 11. Optimizer state estimate
		# ──────────────────────────────────────────────
		print(f"\n  [Optimizer state estimate] (AdamW)")
		optimizer_state_bytes = total_tr * 4 * 2  # fp32 momentum + variance
		print(f"    Trainable params     : {total_tr:,}")
		print(f"    Optimizer states     : {optimizer_state_bytes / 1e6:.1f} MB (fp32)")
		print(f"    Total training memory: {(total_bytes + optimizer_state_bytes) / 1e9:.2f} GB")

		# ──────────────────────────────────────────────
		# 11b. Optimizer inspection (optional, critical for LoRA+)
		# ──────────────────────────────────────────────
		frozen_in_opt = 0
		missing_from_opt = set()
		if optimizer is not None:
			print(f"\n  [Optimizer inspection] ({optimizer.__class__.__name__})")
			opt_param_ids = set()
			for gi, group in enumerate(optimizer.param_groups):
				n_total = sum(p.numel() for p in group['params'])
				n_train = sum(p.numel() for p in group['params'] if p.requires_grad)
				opt_param_ids.update(id(p) for p in group['params'])
				print(
					f"    group[{gi}] | lr={group['lr']:.2e} | wd={group.get('weight_decay', 0):.2e} | "
					f"{n_total:>12,} params ({n_train:,} trainable)"
				)
			if 'lora_plus' in mode_l:
				lrs = sorted(set(g['lr'] for g in optimizer.param_groups))
				if len(lrs) >= 2:
						print(f"    ✅ LoRA+ contract: distinct LR groups {lrs} (B/A ratio = {max(lrs)/min(lrs):.1f}×)")
				else:
						print(f"    ❌ LoRA+ contract VIOLATED: all param groups share lr={lrs[0]} — "
									f"LoRA+ degenerates to plain LoRA")

			frozen_in_opt = sum(1 for g in optimizer.param_groups for p in g['params'] if not p.requires_grad)

			print(f"    {'✅' if frozen_in_opt == 0 else '❌'} Frozen params inside optimizer: {frozen_in_opt}")
			model_train_ids = {id(p) for _, p in named if p.requires_grad}
			missing_from_opt = model_train_ids - opt_param_ids

			if missing_from_opt:
				print(f"    ⚠️  {len(missing_from_opt)} trainable params are NOT in the optimizer (will never update)")
			else:
				print(f"    ✅ All trainable params are covered by the optimizer")

		# ──────────────────────────────────────────────
		# 12. Summary statistics
		# ──────────────────────────────────────────────
		print(f"\n  {'─'*60}")
		print(f"  {mode.upper()} Statistics")
		print(f"  {'─'*60}")
		print(f"  Image: {img_to:>14,} total | trainable {img_tr:>12,} "
					f"({img_tr/img_to*100:.3f}%)" if img_to else f"  Image: none")
		print(f"  Text : {text_to:>14,} total | trainable {text_tr:>12,} "
					f"({text_tr/text_to*100:.3f}%)" if text_to else f"  Text : none")
		if text_adapter_list:
				print(f"         Text-side adapter/extra groups: {text_adapter_list}")
		print(f"  Logit: {logit}")
		if img_to + text_to + logit == total_to:
				print(f"  Total: {total_to:>14,} total | trainable {total_tr:>12,} -> [OK ✅]")
		else:
				diff = total_to - img_to - text_to - logit
				print(f"  Total: {total_to:>14,} total | trainable {total_tr:>12,} -> [FAIL ❌] diff={diff:,}")

		# ──────────────────────────────────────────────
		# 13. Quick health summary
		# ──────────────────────────────────────────────
		print(f"\n  {'='*80}")
		print(f"[HEALTH SUMMARY]")
		print(f"  {'='*80}")
		issues = []
		if nan_params or inf_params:
				issues.append("❌ Numerical issues detected in trainable params")
		if total_tr == 0:
				issues.append("❌ No trainable parameters")
		if img_tr == 0 and text_tr == 0:
				issues.append("⚠️  Both encoders fully frozen — verify this is intended (e.g. zero-shot/probe)")
		if total_to > 0 and 0 < total_tr and (total_tr / total_to) < 0.001:
				issues.append("⚠️  Very low trainable ratio (<0.1%)")
		if unexpected_zero:
				issues.append(f"⚠️  {len(unexpected_zero)} unexpectedly zero-initialized trainable params")
		if optimizer is not None:
				if frozen_in_opt > 0:
						issues.append(f"❌ {frozen_in_opt} frozen params registered in the optimizer")
				if missing_from_opt:
						issues.append(f"⚠️  {len(missing_from_opt)} trainable params missing from the optimizer")

		if issues:
				for issue in issues:
						print(f"  {issue}")
		else:
				print(f"  ✅ All checks passed — ready for training!")
		print(f"  {'='*80}\n")

		return {
				'total_params': total_to,
				'trainable_params': total_tr,
				'frozen_params': total_fr,
				'image_trainable': img_tr,
				'text_trainable': text_tr,
				'trainable_pct': total_tr / total_to * 100 if total_to > 0 else 0,
				'expected_zero_count': len(expected_zero),
				'unexpected_zero_count': len(unexpected_zero),
				'adapter_module_count': len(adapter_modules),
		}

def get_parameters_info_orig(model, mode):
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
	print(f"   ├─ {mode.upper()}")
	print(f"   ├─ Image Encoder: Total: {img_total:,} (Trainable [Unfrozen]): {img_trainable:,} ({img_trainable_percent:.3f}%)  Frozen: {img_frozen:,} ({img_frozen_percent:.3f}%)")
	print(f"   ├─ Text Encoder: Total: {text_total:,} (Trainable [Unfrozen]): {text_trainable:,} ({text_trainable_percent:.3f}%)  Frozen: {text_frozen:,} ({text_frozen_percent:.3f}%)")
	print(f"   ├─ Logit Scale: {logit_scale_params}")
	print(f"   └─ Total: {total_params:,}  (Trainable [Unfrozen]): {total_trainable:,} ({total_trainable_percent:.3f}%)  Frozen: {total_frozen:,} ({total_frozen_percent:.3f}%)")

def cleanup_old_temp_dirs():	
	temp_dirs = glob.glob("/tmp/pymp-*")
	for temp_dir in temp_dirs:
		try:
			shutil.rmtree(temp_dir, ignore_errors=True)
		except:
			pass
	if temp_dirs:
		print(f"Cleaned up {len(temp_dirs)} old temp directories")

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

def print_loader_info(loader):
	batch_size = loader.batch_size
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

def post_process(
	df: pd.DataFrame, 
	dataset_type: str, 
	output_dir: str, 
	is_multi_label: bool=False, 
	figure_size: tuple=(14, 8), 
	dpi: int=250, 
	historgram_bins: int=50, 
	val_split_pct: float=0.35
):
	print(f"\n--- Processing {dataset_type} dataset ---")

	dataset_dir = os.path.dirname(output_dir)
	dataset_name = os.path.basename(dataset_dir)
	print(f"output_dir: {output_dir}")
	print(f"dataset_dir: {dataset_dir}")
	print(f"dataset_name: {dataset_name}")

	if is_multi_label:
		# For multi-label, we need special handling
		plot_label_distribution_fname = os.path.join(
			output_dir, 
			f"{dataset_name}_{dataset_type}_label_distribution_{df.shape[0]}_x_{df.shape[1]}.png"
		)
		# You might want to create a special multi-label visualization here
		print(f"[WARNING] Multi-label visualization needs special handling - skipping for now")

	else:
		# Single-label visualization
		plot_label_distribution_fname = os.path.join(
			output_dir, 
			f"{dataset_name}_{dataset_type}_label_distribution_{df.shape[0]}_x_{df.shape[1]}.png"
		)
		viz.plot_label_distribution(
			df=df,
			fpth=plot_label_distribution_fname,
			FIGURE_SIZE=figure_size,
			DPI=dpi,
			label_column='label',
		)

	if not is_multi_label:
		# Single-label stratified split
		train_df, val_df = get_stratified_split(
			df=df, 
			val_split_pct=val_split_pct,
			label_col='label'
		)
		# Save train/val splits
		train_df.to_csv(os.path.join(dataset_dir, f'metadata_{dataset_type}_train.csv'), index=False)
		val_df.to_csv(os.path.join(dataset_dir, f'metadata_{dataset_type}_val.csv'), index=False)
	else:
		print(f"[WARNING] Multi-label stratified split not implemented yet!")
	
	# Train/val distribution plot
	if not is_multi_label:  # Only for single-label for now
		viz.plot_train_val_label_distribution(
			train_df=train_df,
			val_df=val_df,
			dataset_name=f"{dataset_name}_{dataset_type}",
			VAL_SPLIT_PCT=val_split_pct,
			fname=os.path.join(output_dir, f'{dataset_name}_{dataset_type}_stratified_label_distribution_train_val_{val_split_pct}_pct.png'),
			FIGURE_SIZE=figure_size,
			DPI=dpi,
		)
	
	# Year distribution plot
	viz.plot_year_distribution(
		df=df,
		dname=f"{dataset_name}_{dataset_type}",
		fpth=os.path.join(output_dir, f"{dataset_name}_{dataset_type}_year_distribution_{df.shape[0]}_samples.png"),
		BINs=historgram_bins,
	)
	print(f"{dataset_type} dataset processing complete!")

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

def get_conversation_token_breakdown(text: str, model_name: str = "bert-base-uncased") -> dict:
	tokenizer = tfs.AutoTokenizer.from_pretrained(model_name)
	
	parts = {}
	
	# Count system tokens
	if "system" in text.lower() and "user" in text.lower():
		system_part = text.split("system")[-1].split("user")[0].strip()
		parts['system'] = len(tokenizer.encode(system_part, add_special_tokens=False))
	
	# Count user tokens  
	if "user" in text.lower() and "assistant" in text.lower():
		user_part = text.split("user")[-1].split("assistant")[0].strip()
		parts['user'] = len(tokenizer.encode(user_part, add_special_tokens=False))
	
	# Count assistant tokens
	if "assistant" in text.lower():
		assistant_part = text.split("assistant")[-1].strip()
		parts['assistant'] = len(tokenizer.encode(assistant_part, add_special_tokens=False))
	
	parts['total'] = len(tokenizer.encode(text, add_special_tokens=True))
	
	return parts

def get_token_breakdown(
	inputs,  # actual inputs tensor from processor
	outputs, # generated outputs tensor
) -> dict:
	"""
	Token counting from actual model tensors.
	
	Args:
		inputs: Output from processor() containing input_ids
		outputs: Output from model.generate() 
	
	Returns:
		dict with token counts
	"""
	input_length = inputs.input_ids.shape[1]
	output_length = outputs.shape[1]
	generated_length = output_length - input_length
	
	breakdown = {
		'input_tokens': input_length,
		'generated_tokens': generated_length,
		'total_tokens': output_length,
	}
	
	print(f"[TOKEN BREAKDOWN]")
	print(f"   • Input:     {breakdown['input_tokens']}")
	print(f"   • Generated: {breakdown['generated_tokens']}")
	print(f"   • Total:     {breakdown['total_tokens']}")
	print(f"   • Ratio:     {breakdown['generated_tokens'] / breakdown['input_tokens']:.2%}")
	
	return breakdown

def debug_llm_info(model, tokenizer, device):
	print("\n=== Runtime / Environment ===")
	print(f"Python version      : {sys.version.split()[0]}")
	print(f"PyTorch version     : {torch.__version__}")
	print(f"Transformers version: {tfs.__version__}")
	print(f"CUDA available?    : {torch.cuda.is_available()}")
	if torch.cuda.is_available():
			print(f"CUDA device count  : {torch.cuda.device_count()}")
			print(f"Current CUDA device: {torch.cuda.current_device()}")
			print(f"CUDA device name   : {torch.cuda.get_device_name(0)}")
			print(f"CUDA memory (total/alloc): "
						f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB / "
						f"{torch.cuda.memory_allocated(0) // (1024**2)} MB")
	print(f"Requested device   : {device}")
	print("\n=== Model Overview ===")
	print(f"Model class        : {model.__class__.__name__}")
	# Config (pretty‑print all fields)
	print("\n--- Config ---")
	pprint.pprint(model.config.to_dict(), width=120, compact=True)
	# Parameter statistics
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("\n--- Parameter stats ---")
	print(f"Total parameters          : {total_params:,}")
	print(f"Trainable parameters      : {trainable_params:,}")
	print(f"Non‑trainable parameters  : {total_params - trainable_params:,}")
	print(f"Model in training mode? : {model.training}")
	# Device / dtype per top‑level sub‑module (helps catch mixed‑precision bugs)
	print("\n--- Sub‑module device / dtype ---")
	for name, module in model.named_children():
			# Grab the first parameter of the sub‑module (if any) to infer its device/dtype
			first_param = next(module.parameters(), None)
			if first_param is not None:
					dev = first_param.device
					dt  = first_param.dtype
					print(f"{name:30} → device: {dev}, dtype: {dt}")
			else:
					print(f"{name:30} → (no parameters)")
	# ------------------------------------------------------------------
	# 3️⃣ Tokenizer overview
	# ------------------------------------------------------------------
	print("\n=== Tokenizer Overview ===")
	print(f"Tokenizer class    : {tokenizer.__class__.__name__}")
	print(f"Fast tokenizer?   : {tokenizer.is_fast}")
	# Basic config
	print("\n--- Basic attributes ---")
	print(f"Vocab size         : {tokenizer.vocab_size}")
	print(f"Model max length   : {tokenizer.model_max_length}")
	print(f"Pad token id       : {tokenizer.pad_token_id}")
	print(f"EOS token id       : {tokenizer.eos_token_id}")
	print(f"BOS token id       : {tokenizer.bos_token_id}")
	print(f"UNK token id       : {tokenizer.unk_token_id}")
	# Show the *string* for each special token (if defined)
	specials = {
			"pad_token": tokenizer.pad_token,
			"eos_token": tokenizer.eos_token,
			"bos_token": tokenizer.bos_token,
			"unk_token": tokenizer.unk_token,
			"cls_token": getattr(tokenizer, "cls_token", None),
			"sep_token": getattr(tokenizer, "sep_token", None),
	}
	print("\n--- Special token strings ---")
	for name, token in specials.items():
			if token is not None:
					print(f"{name:12}: '{token}' (id={tokenizer.convert_tokens_to_ids(token)})")
			else:
					print(f"{name:12}: <not set>")
	# Small vocab preview (first & last 10 entries)
	if hasattr(tokenizer, "get_vocab"):
			vocab = tokenizer.get_vocab()
			vocab_items = sorted(vocab.items(), key=lambda kv: kv[1])  # sort by id
			print("\n--- Vocab preview (first & last 10) ---")
			for token, idx in vocab_items[:10]:
					print(f"{idx:5d}: {token}")
			print(" ...")
			for token, idx in vocab_items[-10:]:
					print(f"{idx:5d}: {token}")

	# ------------------------------------------------------------------
	# 4️⃣ Model capabilities
	# ------------------------------------------------------------------
	print("Model Attributes".center(150, "-"))
	print(dir(model))
	print("="*100)

	print("Tokenizer Attributes".center(150, "-"))
	print(dir(tokenizer))
	print("="*100)

def parse_tuple(s):
	try:
		# Convert the string to a tuple
		return ast.literal_eval(s)
	except (ValueError, SyntaxError):
		raise argparse.ArgumentTypeError(f"Invalid tuple format: {s}")

def clean_single_quotes(text):
		# Protect possessives and contractions first
		text = re.sub(r"(\w)'(\w)", r"\1__APOSTROPHE__\2", text)
		
		# Remove anything that looks like quotation marks
		text = re.sub(r'''\s*'\s*''', " ", text)
		text = re.sub(r"^'\s*|\s*'$", " ", text)
		
		# Remove leftover single quotes
		text = re.sub(r"'", "", text)
		
		# Restore real apostrophes
		text = text.replace("__APOSTROPHE__", "'")
		
		# Clean spaces
		return re.sub(r'\s+', ' ', text).strip()

def clean_(text:str, sw:list):
	if not text:
		return
	# print(text)
	# text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Remove special characters and digits
	# text = re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', " ", text)
	# text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', text) # remove all punctuation marks except periods and commas,
	text = re.sub(r"[^\w\s'-]", " ", text) # remove all punctuation marks, including periods and commas,
	words = nltk.tokenize.word_tokenize(text) # Tokenize the text into words
	# Filter out stopwords and words with fewer than 3 characters
	words = [
		word.lower()
		for word in words
		if len(word) >= 2
		# and word.lower() not in sw
	]
	text = ' '.join(words) # Join the words back into a string
	text = re.sub(r'\boriginal caption\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bphoto shows\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bfile record\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\boriginal field number\b', ' ', text, flags=re.IGNORECASE)
	# text = re.sub(r'\bdate taken\b', ' ', text)
	# text = re.sub(r'\bdate\b', ' ', text)
	# text = re.sub(r'\bdistrict\b', ' ', text)
	text = re.sub(r'\bobtained\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bfile record\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bcaption\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bunidentified\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bunnumbered\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\buntitled\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bfotografie\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bfotografen\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bphotograph\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bphotographer\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bphotography\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bfotoalbum\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bphoto\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\bgallery\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r"\bpart \d+\b|\bpart\b", " ", text, flags=re.IGNORECASE)
	text = re.sub(r'\bfoto\b', ' ', text, flags=re.IGNORECASE)
	text = re.sub(r'\s+', ' ', text, flags=re.IGNORECASE)
	text = text.strip() # Normalize whitespace


	if len(text) == 0:
		return None

	return text

def print_gpu_memory():
	print(
		f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, "
		f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB"
	)

def load_categories(file_path: str):
	print(f"Loading categories from {file_path}")
	try:
		with open(file_path, 'r') as file:
			categories = json.load(file)
		return categories['object_categories'], categories['scene_categories'], categories['activity_categories']
	except FileNotFoundError:
		print("File not found.")
		return [], [], []  # Return empty lists instead of None
	except json.JSONDecodeError as e:
		print(f"Invalid JSON format: {e}")
		return [], [], []  # Return empty lists instead of None
	except KeyError as e:
		print(f"Missing key in JSON: {e}")
		return [], [], []  # Return empty lists instead of None	

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

import random
import numpy as np
import torch

def set_seeds(
	seed: int = 42,
	debug: bool = False, # True = maximum reproducibility (slower)
	enable_optimizations: bool = True
):
	# Set random seeds for maximum reproducibility.
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	if debug:
		# Maximum reproducibility mode (slowest)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True, warn_only=True)
		
		# Disable TF32 for full determinism
		torch.backends.cuda.matmul.allow_tf32 = False
		torch.backends.cudnn.allow_tf32 = False
		
		print(f"✅ Seeds set to {seed} | DEBUG MODE (Maximum Reproducibility)")		
	else:
		# Normal / Performance mode
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = enable_optimizations
		
		# Allow TF32 for better performance on Ampere+ GPUs
		if enable_optimizations:
				torch.backends.cuda.matmul.allow_tf32 = True
				torch.backends.cudnn.allow_tf32 = True
				print(f"✅ Seeds set to {seed} | Performance mode (TF32 enabled)")
		else:
				torch.backends.cuda.matmul.allow_tf32 = False
				torch.backends.cudnn.allow_tf32 = False
				print(f"✅ Seeds set to {seed} | Balanced mode")
	# Optional: Print GPU status
	if torch.cuda.is_available() and debug:
			print(f"   CUDA deterministic: {torch.backends.cudnn.deterministic}")
			print(f"   CUDNN benchmark: {torch.backends.cudnn.benchmark}")

def format_elapsed_time(seconds):
	"""
	Convert elapsed time in seconds to DD-HH-MM-SS format.
	"""
	# Create a timedelta object from the elapsed seconds
	elapsed_time = timedelta(seconds=seconds)
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
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		end_time = time.time()
		elapsed_time = end_time - start_time
		formatted_time = format_elapsed_time(elapsed_time)
		print(f"Total elapsed time(DD-HH-MM-SS): \033[92m{formatted_time}\033[0m")		
		return result
	return wrapper

def get_extension(url: str="www.example.com/some_/path/to/file.jpg"):
	parsed_url = urllib.parse.urlparse(url)
	path = parsed_url.path
	_, extension = os.path.splitext(path)
	# return extension[1:].lower() # Remove the leading dot from the extension ['jpg', 'png', 'jpeg', 'txt', 'mov']
	return extension.lstrip('.').lower() # Remove the leading dot from the extension ['jpg', 'png', 'jpeg', 'txt', 'mov']

def is_valid_date(date:str="1939-12-30", start_date: str="1900-01-01", end_date:str="1950-12-31"):
	# Define the start and end dates
	start_date = pd.to_datetime(start_date)
	end_date = pd.to_datetime(end_date)
	if pd.isnull(date):
		return True  # Keep rows with None values
	try:
		date_obj = pd.to_datetime(date)
		return start_date <= date_obj <= end_date
	except ValueError:
		return False

def get_ip_info():
	"""
	Fetch and print current IP address, location, and ISP.
	"""
	try:
		response = requests.get('http://ip-api.com/json')
		data = response.json()
		ip_address = data['query']
		location = f"{data['city']}, {data['regionName']}, {data['country']}"
		isp = data['isp']
		lat, lon = data['lat'], data['lon']
		timezone = data['timezone']
		org = data['org'] # organization
		as_number = data['as']
		as_name = data.get('asname', None)
		mobile = data.get('mobile', False)
		proxy = data.get('proxy', False)
		print(f"IP Address: {ip_address} Location: {location} ISP: {isp}".center(170, "-"))
		print(f"(Latitude, Longitude): ({lat}, {lon}) Time Zone: {timezone} Organization: {org} AS Number: {as_number}, AS Name: {as_name} Mobile: {mobile}, Proxy: {proxy}")
		print("-"*170)
	except requests.exceptions.RequestException as e:
		print(f"Error: {e}")

def process_image_for_storage(
	img_path: str,
	thumbnail_size: tuple = None,  # None = no resize, (W, H) = resize to this
	verbose: bool = False
) -> bool:
	"""
	Process and optimize an image:
	- Convert to RGB and JPEG format
	- Optionally thumbnail to target size (preserving aspect ratio)
	- Apply optimization
	
	Args:
			img_path: Path to image file (will be overwritten)
			thumbnail_size: Target size (width, height) or None to keep original dimensions
			verbose: Print processing details
	
	Returns:
			True if successful, False otherwise
	"""
	if not os.path.exists(img_path):
		if verbose:
			print(f"Image file not found: {img_path}")
		return False

	try:
		original_size_bytes = os.path.getsize(img_path)
		
		with Image.open(img_path) as img:
			img = img.convert("RGB")
			original_dimensions = img.size
			
			# Thumbnail if size is specified and image is larger
			action = "\t\t => Converted to JPEG"
			if thumbnail_size is not None:
				if not isinstance(thumbnail_size, (tuple, list)) or len(thumbnail_size) != 2:
					raise ValueError(f"thumbnail_size must be a tuple of 2 integers, got: {thumbnail_size}")
				
				target_w, target_h = int(thumbnail_size[0]), int(thumbnail_size[1])
				
				if img.size[0] > target_w or img.size[1] > target_h:
					img.thumbnail((target_w, target_h), resample=Image.Resampling.LANCZOS)
					action = f"\t\t => [SUCCESS] thumbnailed: ≤{target_w}×{target_h}"
			
			# Always save as optimized JPEG
			img.save(
				fp=img_path,
				format="JPEG",
				quality=99,
				optimize=True,
				progressive=True,
			)
		
		# Verify the saved image
		with Image.open(img_path) as img:
			img.verify()
		
		if verbose:
			new_size_bytes = os.path.getsize(img_path)
			print(
				f"{action} Original: {original_dimensions} ({original_size_bytes / 1024 / 1024:.1f} MB) "
				f"→ {img.size if thumbnail_size else original_dimensions} "
				f"({new_size_bytes / 1024 / 1024:.3f} MB)"
			)
		
		return True
	except (IOError, SyntaxError, Image.DecompressionBombError) as e:
		if verbose:
			print(f"[ERROR] {img_path}: {e}")
		if os.path.exists(img_path):
			os.remove(img_path)
		return False
	except Exception as e:
		if verbose:
			print(f"[ERROR] {img_path}: {e}")

		if os.path.exists(img_path):
			os.remove(img_path)

		return False

def download_image(
	row,
	session, 
	image_dir, 
	total_rows,
	retries: int = 1,
	backoff_factor: float = 0.5,
	download_timeout: int = 10,
	thumbnail_size: tuple = None,  # None = no thumbnailing
	verbose: bool = False,
):
	"""
	Download and process an image from a URL.
	Args:
		row: DataFrame row containing 'img_url' and 'id'
		session: requests.Session object
		image_dir: Directory to save images
		total_rows: Total number of rows (for progress display)
		retries: Number of download retry attempts
		backoff_factor: Exponential backoff factor for retries
		download_timeout: Download timeout in seconds
		thumbnail_size: Target size (width, height) or None for original size
		verbose: Print detailed progress
	Returns:
		True if successful, False otherwise
	"""
	t0 = time.time()
	rIdx = row.name
	image_url = row['img_url']
	image_id = row['id']
	image_path = os.path.join(image_dir, f"{image_id}.jpg")

	headers = {
		'Content-type': 'application/json',
		'Accept': 'application/json; text/plain; */*',
		'Cache-Control': 'no-cache',
		'Connection': 'keep-alive',
		'Pragma': 'no-cache',
	}

	# Step 1: Check if image already exists
	if os.path.exists(image_path):
		try:
			with Image.open(image_path) as img:
				img.verify()
			
			# Re-process if thumbnailing settings have changed
			if not process_image_for_storage(
				img_path=image_path, 
				thumbnail_size=thumbnail_size,
				verbose=verbose
			):
				if verbose:
					print(f"Existing image {image_path} failed re-processing. Re-downloading...")
			else:
				if verbose:
					mode = "thumbnailed" if thumbnail_size else "original"
					print(f"[{rIdx:6d}/{total_rows:6d}] {image_id:<100} (Existing, {mode}) {time.time()-t0:.3f}s")
				return True							
		except (IOError, SyntaxError, Image.DecompressionBombError) as e:
			print(f"Existing image {image_path} is invalid: {e}, re-downloading...")
			os.remove(image_path)
		except Exception as e:
			print(f"Unexpected error checking {image_path}: {e}")
			os.remove(image_path)

	# Step 2: Attempt download
	attempt = 0
	while attempt < retries:
		try:
			# Try with SSL verification
			response = session.get(
				url=image_url, 
				headers=headers,
				timeout=download_timeout,
			)
			response.raise_for_status()
		except requests.exceptions.SSLError as ssl_err:
			print(f"[{rIdx:6dd}/{total_rows:6d}] SSL error. Retrying without verification: {ssl_err}")
			try:
				response = session.get(
					url=image_url,
					headers=headers,
					timeout=download_timeout, 
					verify=False,
				)
				response.raise_for_status()
			except Exception as fallback_err:
				print(f"[{rIdx:6d}/{total_rows:6d}] Retry without verification failed: {fallback_err}")
				attempt += 1
				time.sleep(backoff_factor * (2 ** attempt))
				continue
						
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx:6d}/{total_rows:6d}] {builtins.str(e):<180}retry: {attempt}/{retries}")
			time.sleep(backoff_factor * (2 ** attempt))
			continue

		# Download successful, now process the image
		try:
			with open(image_path, 'wb') as f:
				f.write(response.content)
			
			with Image.open(image_path) as img:
				img.verify()
			
			# Process and optimize the image
			if not process_image_for_storage(
				img_path=image_path, 
				thumbnail_size=thumbnail_size,
				verbose=verbose
			):
				raise ValueError(f"Failed to process image {image_id} after download.")
			
			if verbose:
				mode = f"Thumbnailed" if thumbnail_size else "Original"
				print(f"[{rIdx:6d}/{total_rows:6d}] {image_id:<100} ({mode}) {time.time()-t0:.1f}s")
			
			return True
		except (SyntaxError, Image.DecompressionBombError, ValueError) as e:
			print(f"[{rIdx:6d}/{total_rows:6d}] Downloaded image {image_id} is invalid: {e}")
			break
		except Exception as e:
			print(f"[{rIdx:6d}/{total_rows:6d}] {e}")
			attempt += 1
			time.sleep(backoff_factor * (2 ** attempt))

	# --- Step 3: Clean up if failed ---
	if os.path.exists(image_path):
		if verbose:
			print(f"\t\t => Removing broken image: {image_path}")
		os.remove(image_path)
	
	if verbose:
		print(f"[{rIdx:6d}/{total_rows:6d}] Failed downloading {image_id} after {retries} attempts.")

	return False

def get_synchronized_df_img(
	df: pd.DataFrame, 
	synched_fpath: str,
	nw: int,
	thumbnail_size: tuple = None,  # None = keep original, (W, H) = resize
	retries: int = 1,
	timeout: int = 10,
	verbose: bool = False,
):
	"""
	Download and synchronize images with DataFrame.
	
	Args:
			df: DataFrame with 'img_url' and 'id' columns
			synched_fpath: Path to save synchronized CSV
			nw: Number of worker threads
			thumbnail_size: Target size (width, height) or None to keep original dimensions
			timeout: Download timeout in seconds
	
	Returns:
			DataFrame containing only rows with successfully downloaded images
	"""
	image_dir = os.path.join(os.path.dirname(synched_fpath), "images")
	os.makedirs(image_dir, exist_ok=True)
	
	# Check if synchronized dataset already exists
	if os.path.exists(synched_fpath):
		print(f"Found existing synchronized dataset at {synched_fpath}. Loading...")
		return pd.read_csv(
			filepath_or_buffer=synched_fpath,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
	print(f"Synchronizing {df.shape[0]} images using {nw} workers...")
	
	if thumbnail_size is not None:
		print(f"Thumbnailing enabled: Images will be resized to ≤{thumbnail_size[0]}×{thumbnail_size[1]} (aspect ratio preserved)")
	else:
		print("Thumbnailing disabled: Original image dimensions will be preserved")
	
	print(f"Output directory: {image_dir}")
	
	successful_rows = []
	
	with requests.Session() as session:
		with ThreadPoolExecutor(max_workers=nw) as executor:
			futures = {
				executor.submit(
					download_image, 
					row=row, 
					session=session, 
					image_dir=image_dir, 
					total_rows=df.shape[0],
					retries=retries, 
					backoff_factor=0.5,
					download_timeout=timeout,
					thumbnail_size=thumbnail_size,
					verbose=verbose,
				): idx for idx, row in df.iterrows()
			}
			
			for future in as_completed(futures):
				original_df_idx = futures[future]
				try:
					success = future.result()
					if success:
						successful_rows.append(original_df_idx)
				except Exception as e:
					# --- IMPROVED ERROR HANDLING ---
					print(f"<!> Error for row {original_df_idx}")
					print(f"    Exception Type: {type(e).__name__}") # e.g., HTTPError, ConnectionError
					print(f"    Message: {builtins.str(e)}") # Use builtins.str to avoid recursion
					
					# Try to print the URL that failed for context
					try:
						print(f"    URL: {df.loc[original_df_idx, 'img_url']}")
					except:
						pass
					
					print("    Traceback:")
					traceback.print_exc()

	print(f"Successfully downloaded: {len(successful_rows)}/{df.shape[0]} images")
	# Create synchronized DataFrame
	synched_df = df.loc[successful_rows].copy()
	print(f"Synchronized DataFrame: {synched_df.shape}")
	# Calculate directory statistics
	actual_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
	img_dir_size_gb = sum(os.path.getsize(os.path.join(image_dir, f)) for f in actual_files) * 1e-9
	print(f"Directory: {image_dir} contains {len(actual_files)} images with total size: {img_dir_size_gb:.1f} GB")

	print(f"Saving synchronized dataset to {synched_fpath}...")
	synched_df.to_csv(synched_fpath, index=False)
	
	try:
		synched_df.to_excel(synched_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	return synched_df

def process_rgb_image(image_path: str, transform: T.Compose):
	# logging.info(f"Processing: {image_path}")
	try:
		with Image.open(image_path) as img:
			img = img.convert('RGB')
			tensor_image = transform(img)
			pixel_count = tensor_image.shape[1] * tensor_image.shape[2]
			channel_sums = tensor_image.sum(dim=[1, 2]).to(torch.float32)  # Use float32 to save memory
			channel_sums_sq = (tensor_image ** 2).sum(dim=[1, 2]).to(torch.float32)
			del tensor_image  # Explicitly free memory
			return channel_sums, channel_sums_sq, pixel_count
	except Exception as e:
		logging.error(f"Error processing {image_path}: {e}")
		return torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), 0

def get_mean_std_rgb_img_multiprocessing(
	source: Union[str, list],
	num_workers: int,
	batch_size: int,
	img_rgb_mean_fpth: str,
	img_rgb_std_fpth: str,
	TIMEOUT :int=30,
	verbose: bool = False,
) -> Tuple[List[float], List[float]]:
	if os.path.exists(img_rgb_mean_fpth) and os.path.exists(img_rgb_std_fpth):
		return load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth)

	# Validate input and prepare image paths
	if isinstance(source, str):
		image_paths = [os.path.join(source, f) for f in os.listdir(source)]
	else:
		image_paths = source

	if not image_paths:
		raise ValueError("No valid images found in the provided source.")	

	total_images = len(image_paths)

	# Dynamically adjust batch_size based on system resources
	available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
	max_batch_size = max(1, int((available_memory * 0.8) // 0.3))  # 0.3GB per batch heuristic
	num_workers = min(num_workers, os.cpu_count(), max(1, int(available_memory // 2)))  # Rough heuristic
	batch_size = min(batch_size, max_batch_size, total_images)
	
	if verbose:
		print(f"Computing mean and std for {total_images} images using {num_workers} CPUs and {batch_size} batch size...")

	# Use ThreadPoolExecutor for I/O-bound tasks (reading images from disk)
	transform = T.Compose([T.ToTensor()])
	sum_ = torch.zeros(3, dtype=torch.float64)
	sum_of_squares = torch.zeros(3, dtype=torch.float64)
	count = 0

	with ThreadPoolExecutor(max_workers=num_workers) as executor:  # Switch to threads for I/O
		futures = []
		for i in range(0, total_images, batch_size):
			batch_paths = image_paths[i:i + batch_size]
			batch_futures = [executor.submit(process_rgb_image, path, transform) for path in batch_paths]
			futures.extend(batch_futures)

		for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
			try:
				result = future.result(timeout=TIMEOUT)
				if result:
					partial_sum, partial_sum_sq, partial_count = result
					if partial_count > 0:
						sum_ += partial_sum.double()  # Accumulate in float64
						sum_of_squares += partial_sum_sq.double()
						count += partial_count
			except Exception as e:
				logging.error(f"Batch failed: {e}")
				continue
	
	if count == 0:
		raise RuntimeError("All images failed processing. Check input data.")
	
	# Compute final statistics
	mean = (sum_ / count).tolist()
	std = (torch.sqrt((sum_of_squares / count) - (sum_ / count) ** 2)).tolist()
	
	if verbose:
		print(f"Mean: {mean} | Std: {std}")
		print(f"Saving mean and std to {img_rgb_mean_fpth} and {img_rgb_std_fpth}...")

	save_pickle(mean, img_rgb_mean_fpth)
	save_pickle(std, img_rgb_std_fpth)
	
	return mean, std

def save_pickle(pkl, fname:str):
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
		
	file_size_mb = os.path.getsize(fpath) / 1e6
	
	print(f"Elapsed_t: {time.time() - start_time:.3f} s | {type(pickle_obj)} | {file_size_mb:.3f} MB".center(150, " "))
	
	return pickle_obj