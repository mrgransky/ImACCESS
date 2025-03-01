import os
import sys
HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")
########################################################################
CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
# print(sys.path)
import clip
########################################################################
import re
import math
import argparse
import random
import json
import random
import time
import torch
import dill
import gzip
import copy
import torch
import torchvision

from torch.optim import AdamW, SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Subset
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from typing import List, Set, Dict, Tuple, Union
from functools import cache
import subprocess
import traceback
import multiprocessing
import warnings
import logging
import absl.logging
import shutil
import nltk
import inspect
import hashlib
import requests
import datetime
from io import BytesIO
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import tabulate
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

def print_args_table(args, parser):
	"""
	Print a formatted table of command-line arguments.
	
	Args:
	args (argparse.Namespace): The parsed arguments.
	parser (argparse.ArgumentParser): The argument parser object.
	"""
	print("Parser")
	args_dict = vars(args)
	table_data = [
			[
					key, 
					value, 
					parser._option_string_actions.get(f'--{key}', parser._option_string_actions.get(f'-{key}')).type.__name__ 
					if parser._option_string_actions.get(f'--{key}') or parser._option_string_actions.get(f'-{key}') 
					else type(value).__name__
			] 
			for key, value in args_dict.items()
	]
	# print(tabulate.tabulate([(key, value) for key, value in args_dict.items()], headers=['Argument', 'Value'], tablefmt='orgtbl'))
	print(tabulate.tabulate(table_data, headers=['Argument', 'Value', 'Type'], tablefmt='orgtbl'))

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
		result = func(*args, **kwargs) # # Execute the function and store the result		
		end_time = time.time()
		elapsed_time = end_time - start_time
		formatted_time = format_elapsed_time(elapsed_time)
		print(f"Total elapsed time(DD-HH-MM-SS): \033[92m{formatted_time}\033[0m")		
		return result
	return wrapper

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

def visualize_samples(dataloader, num_samples=5):
		"""
		Visualize a few samples from the dataloader for debugging purposes.
		
		Args:
				dataloader (DataLoader): The dataloader to visualize samples from.
				num_samples (int): Number of samples to visualize.
		"""
		for i, batch in enumerate(dataloader):
				if i >= num_samples:
						break
				
				images = batch['image']
				captions = batch['caption']
				masks = batch['mask']
				image_filepaths = batch['image_filepath']
				
				for j in range(len(images)):
						image = images[j].permute(1, 2, 0).numpy() # Convert tensor to numpy array and permute dimensions
						caption = captions[j]
						mask = masks[j]
						filepath = image_filepaths[j]
						
						# Denormalize the image
						image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
						image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
						
						plt.figure(figsize=(8, 8))
						plt.imshow(image, cmap='gray')
						# plt.title(f"Caption: {caption}\nMask: {mask}\nFilepath: {filepath}")
						# plt.title(f"Caption: {caption.shape}\nFilepath: {filepath}")
						plt.title(f"Caption: {type(caption)} {caption.shape}\nMask: {type(mask)} {mask.shape}\nFilepath: {filepath}")
						plt.axis('off')
						plt.show()

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

def set_seeds(seed:int=42, debug:bool=False):
	print(f"Setting seeds for reproducibility")
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		if debug: # slows down training but ensures reproducibility
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False

def get_model_details(model, img_size=(3, 224, 224), text_size=(77,), batch_size=1):
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