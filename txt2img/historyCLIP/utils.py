import os
import sys
import re
from tqdm import tqdm
import random
from collections import Counter
import json
import warnings
import random
import time
import torch
import dill
import gzip
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from PIL import Image, ImageDraw, ImageOps
from typing import List, Set, Dict, Tuple, Union
import subprocess

warnings.filterwarnings('ignore')

# Set a new limit for decompression bomb
Image.MAX_IMAGE_PIXELS = None  # Disable the limit completely
# or set a higher limit
# Image.MAX_IMAGE_PIXELS = 300000000  # Example of setting a higher limit
nw:int = multiprocessing.cpu_count() # def: 8

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
visualize: bool = False

if USER=="farid": # local laptop
	WDIR = os.path.join(HOME, "datasets")
	models_dir = os.path.join(WDIR, "trash", "models")
	# visualize = True
elif USER=="alijanif": # Puhti
	WDIR = "/scratch/project_2004072/ImACCESS"
	models_dir = os.path.join(WDIR, "trash", "models")
else: # Pouta
	WDIR = "/media/volume/ImACCESS"
	models_dir = os.path.join(WDIR, "models")
	ddir: str = os.path.join(WDIR, "myntradataset")

# Vision
emb_dim = 128 
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
n_channels = 3
vit_layers = 8
vit_heads = 4 

# Text
vocab_size = 256
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 128
text_heads = 8
text_layers = 8

################################################################################
# # Vision [requires better GPU]
# emb_dim = 256  # Increased for richer embedding
# vit_d_model = 64  # Larger dimension to handle more complex features
# n_channels = 3
# vit_layers = 12  # Increased depth for more detailed image representation
# vit_heads = 8  # More attention heads for diverse feature extraction

# # Text
# vocab_size = 512  # Increased to accommodate more complex vocabularies
# text_d_model = 128  # Increased for better text representation
# max_seq_length = 256  # Longer sequence length for detailed queries
# text_heads = 10  # More heads for better multi-aspect attention
# text_layers = 12  # Increased depth to handle complex queries

################################################################################

def save_pickle(pkl, fname:str=""):
	print(f"\nSaving {type(pkl)}\n{fname}")
	st_t = time.time()
	if isinstance(pkl, ( pd.DataFrame, pd.Series ) ):
		pkl.to_pickle(path=fname)
	else:
		# with open(fname , mode="wb") as f:
		with gzip.open(fname , mode="wb") as f:
			dill.dump(pkl, f)
	elpt = time.time()-st_t
	fsize_dump = os.stat( fname ).st_size / 1e6
	print(f"Elapsed_t: {elpt:.3f} s | {fsize_dump:.2f} MB".center(120, " "))

def load_pickle(fpath:str="unknown",):
	print(f"Checking for existence? {fpath}")
	st_t = time.time()
	try:
		with gzip.open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except gzip.BadGzipFile as ee:
		print(f"<!> {ee} gzip.open() NOT functional => traditional openning...")
		with open(fpath, mode='rb') as f:
			pkl=dill.load(f)
	except Exception as e:
		print(f"<<!>> {e} pandas read_pkl...")
		pkl = pd.read_pickle(fpath)
	elpt = time.time()-st_t
	fsize = os.stat( fpath ).st_size / 1e6
	print(f"Loaded in: {elpt:.5f} s | {type(pkl)} | {fsize:.3f} MB".center(130, " "))
	return pkl

def get_dframe(fpth: str="path/2/file.csv", img_dir: str="path/2/images"):
	print(f"Laoding  {fpth}")
	history_df = pd.read_csv(
		filepath_or_buffer=fpth,
		on_bad_lines='skip',
	)
	# Convert all text columns to lowercase (if any!)
	history_df[history_df.select_dtypes(include=['object']).columns] = history_df.select_dtypes(include=['object']).apply(lambda x: x.str.lower())
	# Check for existence of images and filter DataFrame
	history_df['image_exists'] = history_df['id'].apply(lambda x: os.path.exists(os.path.join(img_dir, f"{x}.jpg")))
	# Drop rows where the image does not exist
	filtered_df = history_df[history_df['image_exists']].drop(columns=['image_exists'])
	# df = history_df.copy() # without checking image dir
	df = filtered_df.copy()
	print(f"df: {df.shape}")
	print(df.head(10))
	print(df['query'].value_counts())
	print("#"*100)
	return df

def get_img_name_without_suffix(fpth):
	# Get the basename of the file path (removes directory)
	basename = os.path.basename(fpth)
	# Split the basename into filename and extension
	filename, extension = os.path.splitext(basename)
	return int(filename)

def plot_loss(losses, num_epochs, save_path):
	"""
	Plots the loss with respect to epoch and saves the plot.
	Parameters:
	losses (list): List of loss values for each epoch.
	num_epochs (int): Number of epochs.
	save_path (str): Path to save the plot.
	"""
	if num_epochs == 1:
		return
	print(f"Saving Loss in {save_path}")
	epochs = range(1, num_epochs + 1)		
	plt.figure(figsize=(10, 5))
	plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Loss vs. Epoch')
	plt.grid(True)
	plt.savefig(save_path)
	
def set_seeds():
	# fix random seeds
	SEED_VALUE = 42
	random.seed(SEED_VALUE)
	np.random.seed(SEED_VALUE)
	torch.manual_seed(SEED_VALUE)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(SEED_VALUE)
		torch.cuda.manual_seed_all(SEED_VALUE)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = True

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
	if encode:
		# Adding SOT and EOT tokens
		out = chr(2) + text + chr(3)
		# Truncate if length exceeds max_seq_length
		if len(out) > max_seq_length:
			out = out[:max_seq_length]
		# Add padding if needed
		out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
		# Encode the text
		out = torch.IntTensor(list(out.encode("utf-8")))
		# Create the mask
		mask = torch.ones(len(out.nonzero()))
		# Pad the mask to max_seq_length
		if len(mask) < max_seq_length:
			mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
		else:
			mask = mask.type(torch.IntTensor)
	else:
		# Decode the text
		out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
		out = "".join(out)
		mask = None
	return out, mask

def get_info(dataloader):
	tot_samples = len(dataloader.dataset)
	n_chunks = len(dataloader) # ceil(tot_samples / dataloader.batch_size)
	print(
		# f"dataloader organization has main {len(next(iter(dataloader)))} element(s)\n"
		f"Total samples:: {tot_samples} "
		f"divided into {n_chunks} chunk(s) "
		f"using batch size: {dataloader.batch_size} "
		f"in {dataloader.num_workers} CPU(s)"
	)
	# for i, data in enumerate(dataloader):
	# 	print(
	# 		f'[{i+1}/{n_chunks}] '
	# 		f'{len(data["image_filepath"])} image_filepath {type(data["image_filepath"])} '
	# 		f'{data["caption"].shape} caption {type(data["image_filepath"])}')
	# 	print(f"Batch {i+1}: {len([img for img in data['image_filepath']])} {[img for img in data['image_filepath']]}")
	# 	c = Counter(data["image_filepath"])
	# 	# print(f"{json.dumps(c, indent=2, ensure_ascii=False)}")
	# 	print("#"*100)
		# print()
		# if i == 0:  # Just show the first batch as an example
		# 	print(f"For Sample batch {i}:")
		# 	# for key in data.keys():
		# 	# 	print(f"{key}")  # Print shape of each item in the batch
		# 	# 	# print(data["caption"])
		# 	print(f'caption: {data["caption"].shape} {type(data["caption"])}')
		# 	print(f'image: {data["image"].shape} {type(data["image"])}')
		# 	print(f'image_filepath: {len(data["image_filepath"])} {data["image_filepath"][:5]} {type(data["image_filepath"])}')
		# 	break  # Exit after printing the first batch

def get_doc_description(df, col:str="colmun_name"):
	class_names = list(df[col].unique())
	captions = {idx: class_name for idx, class_name in enumerate(class_names)}
	# print(f"{len(list(captions.keys()))} Captions:\n{json.dumps(captions, indent=2, ensure_ascii=False)}")
	return captions, class_names

def custom_collate_fn(batch):
	# Filter out the None values from the batch
	batch = [item for item in batch if item is not None]
	# Use default collate function on the filtered batch
	return default_collate(batch)