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

from torch.optim import AdamW, SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PIL import Image
from typing import Tuple, Union, List, Dict, Any
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs) # # Execute the function and store the result		
		end_time = time.time()
		elapsed_time = end_time - start_time
		formatted_time = format_elapsed_time(elapsed_time)
		print(f"function {func.__name__} elapsed time(DD-HH-MM-SS): \033[92m{formatted_time}\033[0m")		
		return result
	return wrapper

def visualize_(dataloader, num_samples=5, ):
	for batch_idx, (batch_imgs, batch_lbls) in enumerate(dataloader):
		print(batch_idx, batch_imgs.shape, batch_lbls.shape, len(batch_imgs), len(batch_lbls)) # torch.Size([32, 3, 224, 224]) torch.Size([32])
		if batch_idx >= num_samples:
			break
		
		image = batch_imgs[batch_idx].permute(1, 2, 0).numpy() # Convert tensor to numpy array and permute dimensions
		caption_idx = batch_lbls[batch_idx]
		print(image.shape, caption_idx)
		print()
			
		# # Denormalize the image
		image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
		image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
		
		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		plt.title(f"Caption {caption_idx.shape}\n{caption_idx}", fontsize=5)
		plt.axis('off')
		plt.show()

def set_seeds(seed:int=42, debug:bool=False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if debug: # slows down training but ensures reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False