import json
import numpy as np
import pandas as pd
import re
import os
import time
import torch
import pickle
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
# import faiss
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import CLIPProcessor, CLIPModel, AlignProcessor, AlignModel
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
from typing import List, Dict, Set, Tuple, Union, Callable, Optional
import hdbscan
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import hashlib
from torch.cuda import get_device_properties, memory_allocated
from torch.utils.data import Dataset, DataLoader
from kneed import KneeLocator
from keybert import KeyBERT
from rake_nltk import Rake
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Suppress logging warnings
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"


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
from tqdm import tqdm
from datetime import timedelta
import glob
import psutil  # For memory usage monitoring
import tabulate
import ast
import httpx
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

def set_seeds(seed: int = 42, debug: bool = False, enable_optimizations: bool = True):
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
    elif enable_optimizations:  # Enable optimizations for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        # Additional optimizations
        torch.backends.cudnn.allow_tf32 = True

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

def get_stratified_split(df:pd.DataFrame, val_split_pct:float, seed:int=42,):
	set_seeds(seed=seed, debug=False)
	# Count the occurrences of each label
	label_counts = df['label'].value_counts()
	labels_to_drop = label_counts[label_counts == 1].index
	# Filter out rows with labels that appear only once
	df_filtered = df[~df['label'].isin(labels_to_drop)]

	# Check if df_filtered is not empty
	if df_filtered.empty or df_filtered['label'].nunique() == 0:
		raise ValueError("No labels with more than one occurrence. Stratified sampling cannot be performed.")

	# stratified splitting
	train_df, val_df = train_test_split(
		df_filtered,
		test_size=val_split_pct,
		shuffle=True, 
		stratify=df_filtered['label'],
		random_state=seed,
	)

	return train_df, val_df

def download_image(row, session, image_dir, total_rows, retries=2, backoff_factor=0.5):
	t0 = time.time()
	rIdx = row.name
	url = row['img_url']
	img_extension = get_extension(url=url)
	image_name = row['id']
	image_path = os.path.join(image_dir, f"{image_name}.jpg")
	# image_path = os.path.join(image_dir, f"{image_name}.{img_extension}")
	if os.path.exists(image_path):
		try:
			# Verify if the existing image can be opened
			with Image.open(image_path) as img:
				img.verify()
			return True  # Image already exists and is valid, => skipping
		except (IOError, SyntaxError) as e:
			print(f"Existing image {image_path} is invalid: {e}, re-downloading...")
	attempt = 0  # Retry mechanism
	while attempt < retries:
		try:
			response = session.get(url, timeout=20)
			response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
			with open(image_path, 'wb') as f:  # Save the image to the directory
				f.write(response.content)
			# Verify if the downloaded image can be opened
			with Image.open(image_path) as img:
				img.verify()
			print(f"{rIdx:<10}/ {total_rows:<10}{image_name:<150}{time.time()-t0:.1f} s")
			return True  # Image downloaded and verified successfully
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx}/{total_rows}] Failed Downloading {url} : {e}, retrying ({attempt}/{retries})")
			time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
		except (SyntaxError, Image.DecompressionBombError) as e:
			print(f"[{rIdx}/{total_rows}] Downloaded image {image_name} is invalid: {e}")
			break  # No need to retry if the image is invalid
	if os.path.exists(image_path):
		print(f"removing broken img: {image_path}")
		os.remove(image_path)  # Remove invalid image file
	print(f"[{rIdx}/{total_rows}] Failed downloading {image_name} after {retries} attempts.")
	return False  # Indicate failed download

def get_synchronized_df_img(df, image_dir:str="path/2/img_dir", nw: int=8):
	print(f"Synchronizing merged_df(raw) & images of {df.shape[0]} records using {nw} CPUs...")
	successful_rows = [] # List to keep track of successful downloads
	with requests.Session() as session:
		with ThreadPoolExecutor(max_workers=nw) as executor:
			futures = {executor.submit(download_image, row, session, image_dir, df.shape[0]): idx for idx, row in df.iterrows()}
			for future in as_completed(futures):
				try:
					success = future.result() # Get result (True or False) from download_image
					if success:
						successful_rows.append(futures[future]) # Keep track of successfully downloaded rows
				except Exception as e:
					print(f"Unexpected error: {e}")
	print(f"cleaning {type(df)} {df.shape} with {len(successful_rows)} succeeded downloaded images [functional URL]...")
	df_cleaned = df.loc[successful_rows] # keep only the successfully downloaded rows
	print(f"Total images downloaded successfully: {len(successful_rows)} out of {df.shape[0]}")
	print(f"df_cleaned: {df_cleaned.shape}")
	img_dir_size = sum(os.path.getsize(os.path.join(image_dir, f)) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))) * 1e-9  # GB
	print(f"{image_dir} contains {len(os.listdir(image_dir))} file(s) with total size: {img_dir_size:.2f} GB")
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

# def clean_text(text):
# 		"""Clean text by removing special characters and excess whitespace"""
# 		if not isinstance(text, str):
# 				return ""
		
# 		# Apply all metadata pattern removals
# 		for pattern in METADATA_PATTERNS:
# 				text = re.sub(pattern, '', text)

# 		# Replace specific patterns often found in metadata
# 		text = re.sub(r'\[\{.*?\}\]', '', text)  # Remove JSON-like structures
# 		text = re.sub(r'http\S+', '', text)      # Remove URLs
# 		text = re.sub(r'\d+\.\d+', '', text)     # Remove floating point numbers
# 		# Remove non-alphanumeric characters but keep spaces
# 		text = re.sub(r'[^\w\s]', ' ', text)
# 		# Replace multiple spaces with a single space
# 		text = re.sub(r'\s+', ' ', text)
# 		text = text.strip().lower()

# 		return text

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