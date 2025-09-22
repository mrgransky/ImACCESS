import json
import numpy as np
import pandas as pd
import re
import os
import sys
import time
import torch
import pickle
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
# import faiss
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import CLIPProcessor, CLIPModel, AlignProcessor, AlignModel
# from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
import transformers as tfs
tfs.logging.set_verbosity_info()

from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize, MultiLabelBinarizer
import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm
import warnings
import urllib.request
import fasttext
import argparse
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from wordcloud import WordCloud
from typing import Tuple, Union, List, Dict, Any, Optional, Callable, TypedDict
import certifi
import hdbscan
import networkx as nx
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import hashlib
from torch.cuda import get_device_properties, memory_allocated
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import urllib3
import huggingface_hub
from dataclasses import dataclass
import io
import pprint
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# from kneed import KneeLocator
# from keybert import KeyBERT
# from rake_nltk import Rake
# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
# from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings(
	"ignore",
	message=".*flash_attn.*",
	category=UserWarning,
	module="transformers"
)

# # Suppress logging warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["GRPC_VERBOSITY"] = "ERROR"
# os.environ["GLOG_minloglevel"] = "2"
# os.environ["TRANSFORMERS_QUIET"] = "1"

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.measure import shannon_entropy
from skimage.transform import resize
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

import requests
import dill
import gzip
import random
import datetime
import logging
from bs4 import BeautifulSoup
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError
from requests.exceptions import RequestException
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from functools import cache, partial
from urllib.parse import urlparse, unquote, quote_plus, urljoin
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm
from datetime import timedelta
import glob
import psutil  # For memory usage monitoring
import tabulate
import ast
import httpx
import gc
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from natsort import natsorted
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
logging.basicConfig(level=logging.INFO)
Image.MAX_IMAGE_PIXELS = None  # Disable the limit completely [decompression bomb]
warnings.filterwarnings('ignore', category=DeprecationWarning, message="invalid escape sequence")

nltk_modules = [
	'punkt',
	'punkt_tab',
	'wordnet',
	'averaged_perceptron_tagger', 
	'omw-1.4',
	'stopwords',
]
nltk.download(
	# 'all',
	nltk_modules,
	# 'stopwords',
	quiet=True,
	# raise_on_error=True,
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

dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str,
}

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

def debug_llm_info(model, tokenizer, device):
	# ------------------------------------------------------------------
	# 1️⃣ Runtime / environment
	# ------------------------------------------------------------------
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
	# ------------------------------------------------------------------
	# 2️⃣ Model overview
	# ------------------------------------------------------------------
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
		if word.lower() not in sw
		and len(word) >= 2
	]
	text = ' '.join(words) # Join the words back into a string
	text = re.sub(r'\boriginal caption\b', ' ', text)
	text = re.sub(r'\bphoto shows\b', ' ', text)
	text = re.sub(r'\bfile record\b', ' ', text)
	text = re.sub(r'\boriginal field number\b', ' ', text)
	# text = re.sub(r'\bdate taken\b', ' ', text)
	# text = re.sub(r'\bdate\b', ' ', text)
	# text = re.sub(r'\bdistrict\b', ' ', text)
	text = re.sub(r'\bobtained\b', ' ', text)
	text = re.sub(r'\bfile record\b', ' ', text)
	text = re.sub(r'\bcaption\b', ' ', text)
	text = re.sub(r'\bunidentified\b', ' ', text)
	text = re.sub(r'\bunnumbered\b', ' ', text)
	text = re.sub(r'\buntitled\b', ' ', text)
	text = re.sub(r'\bfotografie\b', ' ', text)
	text = re.sub(r'\bfotografen\b', ' ', text)
	text = re.sub(r'\bphotograph\b', ' ', text)
	text = re.sub(r'\bphotographer\b', ' ', text)
	text = re.sub(r'\bphotography\b', ' ', text)
	text = re.sub(r'\bfotoalbum\b', ' ', text)
	text = re.sub(r'\bphoto\b', ' ', text)
	text = re.sub(r'\bgallery\b', ' ', text)
	text = re.sub(r"\bpart \d+\b|\bpart\b", " ", text)
	text = re.sub(r'\bfoto\b', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
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

def is_english(
		text: str, 
		ft_model: fasttext.FastText._FastText,
		verbose: bool=False,
		min_length: int=5,
	) -> bool:
	if verbose:
		print(f"text({len(text)}): {text}")
	# Check for empty text
	if not text or len(text) < min_length:
		if verbose:
			print(f"text({len(text)}) is too short")
		return False
	# Sanitize input
	text = text.lower().replace("\n", " ").replace("\r", " ").strip()
	# Remove patterns like C.d.Lupo, S.d.Roma, etc.
	text = re.sub(r'\b[A-Z]\.d\.[A-Z][a-z]+\b', '', text)
	# Remove repeated punctuation and excess whitespace
	text = re.sub(r'\s+', ' ', text)
	text = text.strip(" .\n\r\t")

	# Short texts: rely on ASCII + stopword heuristics
	if len(text) < 20:
		ascii_chars = sum(c.isalpha() and ord(c) < 128 for c in text)
		total_chars = sum(c.isalpha() for c in text)
		if total_chars == 0 or ascii_chars / total_chars < 0.9:
			if verbose:
				print(f"text({len(text)}) is not English")
			return False
		common_words = {'the', 'and', 'of', 'to', 'in', 'is', 'was', 'for', 'with', 'on'}
		words = text.split()
		return any(word in common_words for word in words)
	
	# Long texts: fasttext is preferred
	try:
		prediction = ft_model.predict(text)[0][0]
		if verbose:
			print(f"Fasttext prediction: {prediction}")
		return prediction == '__label__en'
	except ValueError as e:
		if verbose:
			print(f"FastText error: {e}")
		return False

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
		df:pd.DataFrame, 
		val_split_pct:float, 
		seed:int=42,
		label_col: str = 'label',
	):
	print(f"Stratified Splitting [Single-label dataset]".center(150, "-"))
	# Count the occurrences of each label
	label_counts = df[label_col].value_counts()
	labels_to_drop = label_counts[label_counts == 1].index
	# Filter out rows with labels that appear only once
	df_filtered = df[~df[label_col].isin(labels_to_drop)]

	# Check if df_filtered is not empty
	if df_filtered.empty or df_filtered[label_col].nunique() == 0:
		raise ValueError("No labels with more than one occurrence. Stratified sampling cannot be performed.")

	# stratified splitting
	train_df, val_df = train_test_split(
		df_filtered,
		test_size=val_split_pct,
		shuffle=True, 
		stratify=df_filtered[label_col],
		random_state=seed,
	)
	return train_df, val_df

def get_multi_label_stratified_split(
		df: pd.DataFrame,
		val_split_pct: float,
		seed: int = 42,
		label_col: str = 'multimodal_labels',
	) -> Tuple[pd.DataFrame, pd.DataFrame]:
	print(f"Stratified Splitting [Multi-label dataset]".center(150, "-"))
	t_st = time.time()
	df_copy = df.copy()

	# --- 1. Robust Label Parsing using ast.literal_eval ---
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
	mlb = MultiLabelBinarizer()
	label_matrix = mlb.fit_transform(df_filtered[label_col])
	unique_labels = mlb.classes_
	if len(unique_labels) == 0:
		raise ValueError("No unique labels found after processing. Cannot perform stratification.")
	print(f">> Found {len(unique_labels)} unique labels:\n{unique_labels.tolist()[:10]}...") # Show first 10
	
	# --- 4. Perform Iterative Stratification ---
	print("Multi-label stratification using Iterative Stratification...")
	# X is a dummy feature matrix (can be indices or just a range)
	# y is the binarized label matrix
	X_indices = np.arange(len(df_filtered)).reshape(-1, 1)
	
	# iterative_train_test_split returns (X_train, y_train, X_val, y_val)
	X_train_idx, y_train_labels, X_val_idx, y_val_labels = iterative_train_test_split(
		X_indices, 
		label_matrix, 
		test_size=val_split_pct,
	)
	# Convert back to original DataFrame indices
	train_original_indices = df_filtered.iloc[X_train_idx.flatten()].index.values
	val_original_indices = df_filtered.iloc[X_val_idx.flatten()].index.values
	train_df = df_filtered.loc[train_original_indices].reset_index(drop=True)
	val_df = df_filtered.loc[val_original_indices].reset_index(drop=True)
	
	# --- 5. Verify Split and Print Distributions ---
	if train_df.empty or val_df.empty:
		raise ValueError("Train or validation set is empty after splitting. Adjust val_split_pct or check data.")
	print(f"\n>> Original Filtered Data: {df_filtered.shape} => Train: {train_df.shape} Validation: {val_df.shape}")

	print("\nTrain Label Distribution (Top 20):")
	train_label_counts = Counter([label for labels in train_df[label_col] for label in labels])
	train_label_df = pd.DataFrame(train_label_counts.items(), columns=['Label', 'Count']).sort_values(by='Count', ascending=False)
	print(train_label_df.head(20).to_string())
	print("\nValidation Label Distribution (Top 20):")
	val_label_counts = Counter([label for labels in val_df[label_col] for label in labels])
	val_label_df = pd.DataFrame(val_label_counts.items(), columns=['Label', 'Count']).sort_values(by='Count', ascending=False)
	print(val_label_df.head(20).to_string())
	print(f"Stratified Splitting Elapsed Time: {time.time()-t_st:.3f} sec".center(160, "-"))
	
	return train_df, val_df

def _process_image_for_storage(
		img_path: str,
		target_size: tuple,
		large_image_threshold_mb: float,
		verbose: bool=False
	) -> bool:
	if not os.path.exists(img_path):
		if verbose:
			print(f"Image file not found for processing: {img_path}")
		return False

	file_size_bytes = os.path.getsize(img_path)
	large_image_threshold_bytes = large_image_threshold_mb * 1024 * 1024

	# If image is already small, just convert to JPEG and return
	if file_size_bytes <= large_image_threshold_bytes:
		if verbose:
			print(f"\timage size: {file_size_bytes / 1024 / 1024:.2f} <= (threshold: {large_image_threshold_mb} MB) => stored unchanged.")
		return True

	if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
		if verbose:
			print(f"Invalid target_size: {target_size}. Must be a tuple/list of 2 integers.")
		raise ValueError(f"Invalid target_size: {target_size}. Must be a tuple/list of 2 integers.")

	try:
		target_size = (int(target_size[0]), int(target_size[1]))
	except (ValueError, TypeError) as e:
		if verbose:
			print(f"Invalid target_size: {target_size}. Must be a tuple/list of 2 integers. Error: {e}")
		raise ValueError(f"Invalid target_size: {target_size}. Must be a tuple/list of 2 integers.") from e

	try:
		with Image.open(img_path) as img:
			img = img.convert("RGB")
			original_size = img.size
			if file_size_bytes > large_image_threshold_bytes:
				# Only thumbnail if current dimensions are larger than target size
				if img.size[0] > target_size[0] or img.size[1] > target_size[1]:
					if verbose:
						print(
							f"\tCreating thumbnail"
							f"(Original dimensions: {original_size[0]}x{original_size[1]}, "
							f"File size: {file_size_bytes / 1024 / 1024:.2f} MB) "
							f"=> Target size: {target_size})"
						)
					img.thumbnail(target_size, resample=Image.Resampling.LANCZOS)
					action_taken = "Thumbnailed"
				else:
					if verbose:
						print(f"Large image {os.path.basename(img_path)} within target_size {target_size}. Converting to JPEG.")
					action_taken = "only JPEG-ified"
			img.save(
				fp=img_path, # Overwrite the original file
				format="JPEG",
				quality=100,
				optimize=True,
				progressive=True,
			)
			new_file_size_bytes = os.path.getsize(img_path)
			if verbose:
				print(
					f"\t{action_taken}"
					f"(New size: {img.size[0]}x{img.size[1]}, "
					f"New file size: {new_file_size_bytes / 1024 / 1024:.2f} MB)."
				)
		with Image.open(img_path) as img:
			img.verify()
		return True
	except (IOError, SyntaxError, Image.DecompressionBombError) as e:
		if verbose:
			print(f"Error processing image {img_path} for thumbnail/optimization: {e}")
		if os.path.exists(img_path):
			os.remove(img_path)
		return False
	except Exception as e:
		if verbose:
			print(f"An unexpected error occurred during image processing for {img_path}: {e}")
		if os.path.exists(img_path):
			try:
				os.remove(img_path)
			except OSError as remove_error:
				print(f"Failed to remove corrupted image {img_path}: {remove_error}")
			os.remove(img_path)
		return False

def download_image(
		row,
		session, 
		image_dir, 
		total_rows,
		retries=2, 
		backoff_factor=0.5,
		download_timeout=20,
		enable_thumbnailing: bool = False,
		thumbnail_size: tuple = (500, 500),
		large_image_threshold_mb: float = 2.0,
	):
	t0 = time.time()
	rIdx = row.name
	image_url = row['img_url']
	image_id = row['id']
	image_path = os.path.join(image_dir, f"{image_id}.jpg")

	# --- Step 1: Check if image already exists ---
	if os.path.exists(image_path):
			try:
					with Image.open(image_path) as img:
							img.verify()
					if enable_thumbnailing:
							if not _process_image_for_storage(
									img_path=image_path, 
									target_size=thumbnail_size, 
									large_image_threshold_mb=large_image_threshold_mb, 
									verbose=True
							):
									print(f"Existing image {image_path} valid but re-processing failed. Re-downloading...")
							else:
									print(f"{rIdx:<10}/ {total_rows:<10}{image_id:<150} (Skipping existing & processed) {time.time()-t0:.1f} s")
									return True
					else:
							print(f"{rIdx:<10}/ {total_rows:<10}{image_id:<150} (Skipping existing raw) {time.time()-t0:.1f} s")
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
					response = session.get(image_url, timeout=download_timeout)
					response.raise_for_status()
			except requests.exceptions.SSLError as ssl_err:
					print(f"[{rIdx}/{total_rows}] SSL error. Retrying without verification: {ssl_err}")
					try:
							response = session.get(image_url, timeout=download_timeout, verify=False)
							response.raise_for_status()
					except Exception as fallback_err:
							print(f"[{rIdx}/{total_rows}] Retry without verification failed: {fallback_err}")
							attempt += 1
							time.sleep(backoff_factor * (2 ** attempt))
							continue  # Retry loop
			except (RequestException, IOError) as e:
					attempt += 1
					print(f"<!> [{rIdx}/{total_rows}] {e}, retrying ({attempt}/{retries})")
					time.sleep(backoff_factor * (2 ** attempt))
					continue
			try:
					with open(image_path, 'wb') as f:
							f.write(response.content)
					with Image.open(image_path) as img:
							img.verify()
					if not _process_image_for_storage(
							img_path=image_path, 
							target_size=thumbnail_size, 
							large_image_threshold_mb=large_image_threshold_mb, 
							verbose=True
					):
							raise ValueError(f"Failed to process image {image_id} after download.")
					print(f"{rIdx:<10}/ {total_rows:<10}{image_id:<150}{time.time()-t0:.1f} s")
					return True
			except (SyntaxError, Image.DecompressionBombError, ValueError) as e:
					print(f"[{rIdx}/{total_rows}] Downloaded image {image_id} is invalid: {e}")
					break
			except Exception as e:
					print(f"[{rIdx}/{total_rows}] Unexpected error after download: {e}")
					attempt += 1
					time.sleep(backoff_factor * (2 ** attempt))

	# --- Step 3: Clean up if failed ---
	if os.path.exists(image_path):
			print(f"removing broken img: {image_path}")
			os.remove(image_path)
	print(f"[{rIdx}/{total_rows}] Failed downloading {image_id} after {retries} attempts.")

	return False

def get_synchronized_df_img(
		df:pd.DataFrame, 
		image_dir:str, 
		nw: int,
		thumbnail_size: tuple=(1000, 1000),
		large_image_threshold_mb: float=2.0,
		enable_thumbnailing: bool=False,
	):
	print(f"Synchronizing {df.shape[0]} images using {nw} CPUs...")
	if enable_thumbnailing:
		print(f"Image processing enabled: images > {large_image_threshold_mb} MB will be thumbnailed to {thumbnail_size}.")
	else:
		print("Image processing disabled: raw images will be downloaded as is.")

	successful_rows = [] # List to keep track of successful downloads
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
					download_timeout=20,
					enable_thumbnailing=enable_thumbnailing,
					thumbnail_size=thumbnail_size,
					large_image_threshold_mb=large_image_threshold_mb,
				): idx for idx, row in df.iterrows()
			}
			for future in as_completed(futures):
				original_df_idx = futures[future]
				try:
					success = future.result() # (True/False) from download_image
					if success:
						successful_rows.append(original_df_idx) # Keep track of successfully downloaded rows
				except Exception as e:
					print(f"Unexpected error: {e} for {original_df_idx}")
	print(f"Total successful downloads: {len(successful_rows)}/{df.shape[0]}.")

	print(f"cleaning {type(df)} {df.shape} with {len(successful_rows)} succeeded downloaded images [functional URL]...")
	df_cleaned = df.loc[successful_rows].copy() # keep only the successfully downloaded rows
	print(f"df_cleaned: {df_cleaned.shape}")

	actual_files_in_dir = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
	img_dir_size = sum(os.path.getsize(os.path.join(image_dir, f)) for f in actual_files_in_dir) * 1e-9  # GB
	print(f"{image_dir} contains {len(actual_files_in_dir)} file(s) with total size: {img_dir_size:.2f} GB")
	return df_cleaned

def get_extension(url: str="www.example.com/some_/path/to/file.jpg"):
	parsed_url = urlparse(url)
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
	) -> Tuple[List[float], List[float]]:
	
	if os.path.exists(img_rgb_mean_fpth) and os.path.exists(img_rgb_std_fpth):
		return load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth)

	# Validate input and prepare image paths
	if isinstance(source, str):
		# image_paths = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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

	print(f"Processing {total_images} images with {num_workers} workers and batch_size={batch_size}")
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
				result = future.result(timeout=30)  # Increase timeout for slow I/O
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
	
	# Save results
	save_pickle(mean, img_rgb_mean_fpth)
	save_pickle(std, img_rgb_std_fpth)
	
	return mean, std

def check_url_status(url: str, TIMEOUT:int=50) -> bool:
	try:
		response = requests.head(url, timeout=TIMEOUT)
		# Return True only if the status code is 200 (OK)
		return response.status_code == 200
	except (requests.RequestException, Exception) as e:
		print(f"Error accessing URL {url}: {e}")
		return False

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
	elapsed_time = time.time() - start_time
	file_size_mb = os.path.getsize(fpath) / 1e6
	print(f"Elapsed_t: {elapsed_time:.3f} s | {type(pickle_obj)} | {file_size_mb:.3f} MB".center(150, " "))
	return pickle_obj