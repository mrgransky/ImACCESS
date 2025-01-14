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
import matplotlib.pyplot as plt

from PIL import Image
from typing import Tuple, Union, List
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import timedelta

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