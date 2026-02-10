import re
import os
import sys
import time
import json
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
from typing import Tuple, Union, List, Dict, Any, Optional, Callable, TypedDict
import certifi
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import hashlib
from torch.cuda import get_device_properties, memory_allocated
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import huggingface_hub
import io
import pprint
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
import httpx
import gc
import joblib
import inspect
import warnings
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import shannon_entropy
from skimage.transform import resize
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from bs4 import BeautifulSoup
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError
from requests.exceptions import RequestException
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from functools import cache, partial

from skmultilearn.model_selection import iterative_train_test_split, IterativeStratification
from tqdm import tqdm
from datetime import timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from natsort import natsorted
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
from sentence_transformers import SentenceTransformer

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set environment variable for memory optimization
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions (as suggested in error)

# # Verify environment variables
# print(f"HF_HOME: {os.environ['HF_HOME']}")
# print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
# print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
# print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")

import transformers as tfs
# tfs.logging.set_verbosity_info()

dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str,
}

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
		print(f"Multi-label visualization needs special handling - skipping for now")

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
		print(f"Multi-label stratified split not implemented yet!")
	
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

def monitor_memory_usage():
	"""Monitor memory usage and return True if memory is critical"""
	if torch.cuda.is_available():
		gpu_alloc = torch.cuda.memory_allocated() / 1024**3
		gpu_cached = torch.cuda.memory_reserved() / 1024**3
		gpu_percent = (gpu_alloc / (gpu_alloc + gpu_cached)) * 100 if (gpu_alloc + gpu_cached) > 0 else 0
	else:
		gpu_percent = 0
	
	cpu_mem = psutil.virtual_memory()
	cpu_percent = cpu_mem.percent
	
	if cpu_percent > 90 or gpu_percent > 90:
		print(f"Memory warning - CPU: {cpu_percent:.1f}%, GPU: {gpu_percent:.1f}%")
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
	print(f"   • Input tokens:      {breakdown['input_tokens']}")
	print(f"   • Generated tokens:  {breakdown['generated_tokens']}")
	print(f"   • Total tokens:      {breakdown['total_tokens']}")
	print(f"   • Generation ratio:  {breakdown['generated_tokens'] / breakdown['input_tokens']:.2%}")
	
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

def set_seeds(
	seed: int=42,
	debug: bool=False,
	enable_optimizations: bool=True
):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	
	if debug:  # slows down training but ensures reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		# Disable optimizations for debug mode
		torch.backends.cuda.matmul.allow_tf32 = False
		print(f"Seeds set to {seed}")
	elif enable_optimizations:  # Enable optimizations for performance
		torch.backends.cudnn.benchmark = True
		torch.backends.cuda.matmul.allow_tf32 = True
		# Additional optimizations
		torch.backends.cudnn.allow_tf32 = True
		print(f"Seeds set to {seed}")

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

def get_stratified_split(
	df: pd.DataFrame, 
	val_split_pct: float, 
	seed: int=42,
	label_col: str='label',
):
	print(f"\n>> Stratified Splitting [Single-label dataset]")

	# Count the occurrences of each label
	label_counts = df[label_col].value_counts()

	labels_to_drop = label_counts[label_counts == 1].index
	print(f"\nFound {len(labels_to_drop)} label(s) that occur(s) only once:\n{labels_to_drop.tolist()}\n=> Dropping them...\n")

	# Filter out rows with labels that appear only once
	df_filtered = df[~df[label_col].isin(labels_to_drop)]

	print(f"df_filtered: {df_filtered.shape} {df_filtered[label_col].nunique()} unique labels")

	# Check if df_filtered is not empty
	if df_filtered.empty or df_filtered[label_col].nunique() == 0:
		raise ValueError("No labels with more than one occurrence. Stratified sampling cannot be performed.")

	print(f">> Splitting dataset: train/val: {val_split_pct}...")
	train_df, val_df = train_test_split(
		df_filtered,
		test_size=val_split_pct,
		shuffle=True, 
		stratify=df_filtered[label_col],
		random_state=seed,
	)

	return train_df, val_df

def get_multi_label_stratified_split(
	csv_file: str,
	val_split_pct: float,
	label_col: str='multimodal_labels',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	print(f"\n>> Stratified Splitting [Multi-label dataset]")
	t_st = time.time()
	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	)
	df_copy = df.copy()

	# 1. Robust Label Parsing using ast.literal_eval
	print(f"Parsing '{label_col}' column...")
	if label_col not in df_copy.columns:
		raise ValueError(f"Label column '{label_col}' not found in the DataFrame.")

	def parse_label(x):
		if isinstance(x, str): # Only apply literal_eval if it's a string
			try:
				return ast.literal_eval(x)
			except (ValueError, SyntaxError) as e:
				# Raise an error if a string cannot be parsed, as it's unexpected
				raise ValueError(f"Malformed string found in '{label_col}': '{x}'. Error: {e}")
		elif isinstance(x, list): # If it's already a list, return it as is
			return x
		else:
			# Handle other unexpected types, or raise an error
			print(f"Warning: Unexpected type '{type(x)}' found in '{label_col}': {x}. Trying to convert to empty list.")
			return [] # Or raise ValueError("Unsupported type in label column")
	try:
		df_copy[label_col] = df_copy[label_col].apply(parse_label)
		print(f"Successfully processed '{label_col}' column.")
	except ValueError as e: # Catch the specific ValueError from parse_label
		raise ValueError(
			f"Error parsing multi-label column '{label_col}'. "
			f"Ensure it contains valid string representations of lists. Error: {e}"
		)
	except Exception as e: # Catch any other unexpected errors during apply
		raise ValueError(f"Unexpected error during parsing of '{label_col}': {e}")

	# --- 2. Remove rows with empty label lists (if any became empty after parsing) ---
	df_filtered = df_copy[df_copy[label_col].apply(len) > 0]
	initial_rows = len(df_copy)
	final_rows = len(df_filtered)
	if final_rows == 0:
		raise ValueError("No samples with non-empty label lists remain after parsing and initial filtering.")
	if initial_rows != final_rows:
		print(f"Removed {initial_rows - final_rows} rows with empty label lists.")
	print(f"DataFrame shape after filtering empty label lists: {df_filtered.shape}")

	# --- 3. Binarize Labels ---
	print(">> Binarizing labels...")
	mlb = MultiLabelBinarizer(sparse_output=True)
	label_matrix = mlb.fit_transform(df_filtered[label_col])
	print(f"Label matrix: {label_matrix.shape} {label_matrix.data.nbytes / 1e6:.1f} MB")

	unique_labels = mlb.classes_
	if len(unique_labels) == 0:
		raise ValueError("No unique labels found after processing. Cannot perform stratification.")
	print(f">> Found {len(unique_labels)} unique labels:\n{unique_labels.tolist()[:50]}")
	
	# --- 4. Perform Iterative Stratification ---
	print(">> Multi-label stratification using Iterative Stratification")
	X_indices = np.arange(len(df_filtered)).reshape(-1, 1)
	print(f"X_indices: {type(X_indices)} {X_indices.shape}")

	# #################################################################################################
	# print(">> iterative_train_test_split (slow)...")
	# X_train_idx, y_train_labels, X_val_idx, y_val_labels = iterative_train_test_split(
	# 	X_indices, 
	# 	label_matrix, # sparse matrix
	# 	test_size=val_split_pct,
	# 	n_jobs=-1, # Use all available CPU cores
	# )
	# print(">> Converting back to original DataFrame indices...")
	# train_original_indices = df_filtered.iloc[X_train_idx.flatten()].index.values
	# val_original_indices = df_filtered.iloc[X_val_idx.flatten()].index.values
	# #################################################################################################

	#################################################################################################
	print(f">> IterativeStratification dataset: {df_filtered.shape} [takes time for large datasets]...")
	stratifier = IterativeStratification(
		n_splits=2,
		order=1 if len(df_filtered) > int(1e5) else 2,  # Lower order = faster (default is 2)
		sample_distribution_per_fold=[val_split_pct, 1-val_split_pct],
	)
	train_indices, val_indices = next(stratifier.split(X_indices, label_matrix))
	train_original_indices = df_filtered.iloc[train_indices].index.values
	val_original_indices = df_filtered.iloc[val_indices].index.values
	#################################################################################################

	print(f"train_original_indices: {type(train_original_indices)} {train_original_indices.shape}")
	print(f"val_original_indices: {type(val_original_indices)} {val_original_indices.shape}")


	train_df = df_filtered.loc[train_original_indices].reset_index(drop=True)
	val_df = df_filtered.loc[val_original_indices].reset_index(drop=True)
	
	# --- 5. Verify Split and Print Distributions ---
	if train_df.empty or val_df.empty:
		raise ValueError("Train or validation set is empty after splitting. Adjust val_split_pct or check data.")
	print(f"\n>> Original Filtered Data: {df_filtered.shape} => Train: {train_df.shape} Validation: {val_df.shape}")

	print(f"Stratified Splitting Elapsed Time: {time.time()-t_st:.3f} sec")
	
	# Save train/val splits
	train_path = csv_file.replace('.csv', '_train.csv')
	val_path = csv_file.replace('.csv', '_val.csv')
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	print(f"Saved train/val splits to {train_path} and {val_path}")
	return train_df, val_df

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
			action = "Converted to JPEG"
			if thumbnail_size is not None:
				if not isinstance(thumbnail_size, (tuple, list)) or len(thumbnail_size) != 2:
					raise ValueError(f"thumbnail_size must be a tuple of 2 integers, got: {thumbnail_size}")
				
				target_w, target_h = int(thumbnail_size[0]), int(thumbnail_size[1])
				
				if img.size[0] > target_w or img.size[1] > target_h:
					img.thumbnail((target_w, target_h), resample=Image.Resampling.LANCZOS)
					action = f"Successfully thumbnailed to ≤{target_w}×{target_h}"
			
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
				f"{action} Original: {original_dimensions} ({original_size_bytes / 1024 / 1024:.2f} MB) "
				f"→ {img.size if thumbnail_size else original_dimensions} "
				f"({new_size_bytes / 1024 / 1024:.2f} MB)"
			)
		
		return True
	except (IOError, SyntaxError, Image.DecompressionBombError) as e:
		if verbose:
			print(f"Error processing {img_path}: {e}")
		if os.path.exists(img_path):
			os.remove(img_path)
		return False
	except Exception as e:
		if verbose:
			print(f"Unexpected error processing {img_path}: {e}")
		if os.path.exists(img_path):
			os.remove(img_path)
		return False

def download_image(
	row,
	session, 
	image_dir, 
	total_rows,
	retries: int = 2, 
	backoff_factor: float = 0.5,
	download_timeout: int = 15,
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

	# --- Step 1: Check if image already exists ---
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
					print(f"{rIdx:<10}/{total_rows:<10} {image_id:<100} (Existing, {mode}) {time.time()-t0:.1f}s")
				return True							
		except (IOError, SyntaxError, Image.DecompressionBombError) as e:
			print(f"Existing image {image_path} is invalid: {e}, re-downloading...")
			os.remove(image_path)
		except Exception as e:
			print(f"Unexpected error checking {image_path}: {e}")
			os.remove(image_path)

	# --- Step 2: Attempt download ---
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
			print(f"[{rIdx}/{total_rows}] SSL error. Retrying without verification: {ssl_err}")
			try:
				response = session.get(
					url=image_url,
					headers=headers,
					timeout=download_timeout, 
					verify=False,
				)
				response.raise_for_status()
			except Exception as fallback_err:
				print(f"[{rIdx}/{total_rows}] Retry without verification failed: {fallback_err}")
				attempt += 1
				time.sleep(backoff_factor * (2 ** attempt))
				continue
						
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx}/{total_rows}] {e} retry {attempt}/{retries}")
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
				print(f"{rIdx:<10}/{total_rows:<10} {image_id:<100} ({mode}) {time.time()-t0:.1f}s")
			
			return True
		except (SyntaxError, Image.DecompressionBombError, ValueError) as e:
			print(f"[{rIdx}/{total_rows}] Downloaded image {image_id} is invalid: {e}")
			break
		except Exception as e:
			print(f"[{rIdx}/{total_rows}] {e}")
			attempt += 1
			time.sleep(backoff_factor * (2 ** attempt))

	# --- Step 3: Clean up if failed ---
	if os.path.exists(image_path):
		if verbose:
			print(f"Removing broken image: {image_path}")
		os.remove(image_path)
	
	if verbose:
		print(f"[{rIdx}/{total_rows}] Failed downloading {image_id} after {retries} attempts.")
	return False

def get_synchronized_df_img(
	df: pd.DataFrame, 
	synched_fpath: str,
	nw: int,
	thumbnail_size: tuple = None,  # None = keep original, (W, H) = resize
	timeout: int = 30,
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
					retries=2, 
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
					print(f"Unexpected error for row {original_df_idx}: {e}")

	print(f"Successfully downloaded: {len(successful_rows)}/{df.shape[0]} images")
	# Create synchronized DataFrame
	synched_df = df.loc[successful_rows].copy()
	print(f"Synchronized DataFrame: {synched_df.shape}")
	# Calculate directory statistics
	actual_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
	img_dir_size_gb = sum(os.path.getsize(os.path.join(image_dir, f)) for f in actual_files) * 1e-9
	print(f"Directory: {image_dir}")
	print(f"\tTotal Found Files: {len(actual_files)}")
	print(f"\tTotal size: {img_dir_size_gb:.1f} GB")
	# Save synchronized dataset
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
		# Process results with timeout handling
		for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
			try:
				result = future.result(timeout=TIMEOUT)  # Increase timeout for slow I/O
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