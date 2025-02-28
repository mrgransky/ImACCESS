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

Image.MAX_IMAGE_PIXELS = None # Disable DecompressionBombError

# Vision
vit_d_model = 32 # vit_heads * vit_layers = vit_d_model
n_channels = 3 # must be 3 for CLIP model
vit_layers = 8
vit_heads = 4 

# Text
vocab_size = 512
text_d_model = 64 #  -->  text_heads * text_layers = text_d_model
max_seq_length = 256 # Longer sequence length for detailed queries
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

# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open('meaningless_words.txt', 'r') as file_:
	customized_meaningless_lemmas=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_lemmas)
STOPWORDS = set(STOPWORDS)

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

def clean_(text:str="this is a sample text!", sw:List=list(), check_language:bool=False):
	if not text:
		return
	# print(text)
	# text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Remove special characters and digits
	# text = re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', " ", text)
	# text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', text) # remove all punctuation marks except periods and commas,
	text = re.sub(r"[^\w\s'-]", " ", text) # remove all punctuation marks, including periods and commas,
	if check_language:
		text = remove_misspelled_(documents=text)
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

def visualize_(dataloader, num_samples=5, ):
	for batch_idx, (batch_imgs, batch_lbls, batch_lbls_int) in enumerate(dataloader):
		# torch.Size([batch_size, 3, 224, 224]), torch.Size([batch_size, 77]), torch.Size([batch_size])
		print(batch_idx, batch_imgs.shape, batch_lbls.shape, len(batch_imgs), len(batch_lbls))
		if batch_idx >= num_samples:
			break
		
		image = batch_imgs[batch_idx].permute(1, 2, 0).numpy() # Convert tensor to numpy array and permute dimensions
		caption_idx = batch_lbls_int[batch_idx]
		print(image.shape, caption_idx)
		print()
			
		# # Denormalize the image
		image = image * np.array([0.2268645167350769]) + np.array([0.6929051876068115])
		image = np.clip(image, 0, 1)  # Ensure pixel values are in [0, 1] range
		
		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		plt.title(f"Caption: {caption_idx}", fontsize=8)
		plt.axis('off')
		plt.show()

@cache
def remove_misspelled_(documents: str="This is a sample sentence."):
	# print(f"Removing misspelled word(s)".center(100, " "))	
	# Split the documents into words
	documents = documents.title()
	if not isinstance(documents, list):
		# print(f"Convert to a list of words using split() command |", end=" ")
		words = documents.split()
	else:
		words = documents	
	# print(f"Document conatins {len(words)} word(s)")
	# Remove misspelled words
	cleaned_words = []
	for word in words:
		if not (
			fi_dict.spell(word)
			or fii_dict.check(word)
			or sv_dict.check(word)
			or sv_fi_dict.check(word)
			or en_dict.check(word)
			or de_dict.check(word)
			or da_dict.check(word)
			or es_dict.check(word)
			or et_dict.check(word)
			or cs_dict.check(word)
			or fr_dict.check(word)
			# or ga_dict.check(word)
			# or hr_dict.check(word)
			# or hu_dict.check(word)
		):
			# print(f"\t\t{word} does not exist")
			pass
		else:
			cleaned_words.append(word)
	# Join the cleaned words back into a string
	cleaned_text = " ".join(cleaned_words)
	# print(f"Elapsed_t: {time.time()-t0:.3f} sec".center(100, " "))
	return cleaned_text

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

def get_dframe(fpth: str="path/2/file.csv", img_dir: str="path/2/images"):
	print(f"Creating History_df from: {fpth}", end="\t")
	history_df = pd.read_csv(
		filepath_or_buffer=fpth,
		on_bad_lines='skip',
	)
	# print(f"Raw df: {history_df.shape}")
	# history_df[history_df.select_dtypes(include=['object']).columns] = history_df.select_dtypes(include=['object']).apply(lambda x: x.str.lower()) # lowercase all cols
	# print(f"Raw df [after lower case]: {history_df.shape}")
	history_df['image_exists'] = history_df['id'].apply(lambda x: os.path.exists(os.path.join(img_dir, f"{x}.jpg"))) # Check for existence of images and filter DataFrame
	# print(f"Raw df [after image_exists]: {history_df.shape}")
	filtered_df = history_df[history_df['image_exists']].drop(columns=['image_exists']) # Drop rows where the image does not exist
	# print(f"Raw df [after droping]: {history_df.shape}")
	# print(history_df.head(20))
	# print("#"*100)
	# print(history_df.tail(20))
	# df = history_df.copy() # without checking image dir
	df = filtered_df.copy()
	print(f"=> {df.shape}")
	return df

def get_img_name_without_suffix(fpth):
	# Get the basename of the file path (removes directory)
	basename = os.path.basename(fpth)
	# Split the basename into filename and extension
	filename, extension = os.path.splitext(basename)
	return int(filename)

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

def get_train_val_metadata_df(tddir:str="path/2/train_dataset", vddir=None, split_pct=None, doc_desc:str="label", seed:bool=True):
	metadata_st = time.time()
	metadata_df = get_dframe(
		fpth=os.path.join(tddir, "metadata.csv"),
		img_dir=os.path.join(tddir, "images"),
	)
	if vddir:
		print(f"Available validation dataset: {vddir}")
		train_metadata_df_fpth:str = os.path.join(tddir, f"metadata_train_df.gz")
		val_metadata_df_fpth:str = os.path.join(vddir, f"metadata_val_df.gz")
		val_metadata_df = get_dframe(
			fpth=os.path.join(vddir, "metadata.csv"),
			img_dir=os.path.join(vddir, "images"),
		)
		train_metadata_df = metadata_df
	elif split_pct:
		print(f"spliting training dataset: {tddir} into {split_pct} validation...")
		if seed:
			set_seeds()
		train_metadata_df_fpth:str = os.path.join(tddir, f"metadata_{1-split_pct}_train_df.gz")
		val_metadata_df_fpth:str = os.path.join(tddir, f"metadata_{split_pct}_val_df.gz")
		train_metadata_df, val_metadata_df = train_test_split(
			metadata_df, 
			shuffle=True, 
			test_size=split_pct, # 0.05
			random_state=42,
		)
	else:
		print(f"No such a case!!!")
		return None, None
	save_pickle(pkl=train_metadata_df, fname=train_metadata_df_fpth)
	save_pickle(pkl=val_metadata_df, fname=val_metadata_df_fpth)
	print(f"<<< DF >>> [train] {train_metadata_df.shape} [val] {val_metadata_df.shape} Elapsed_t: {time.time()-metadata_st:.1f} sec".center(180, "-"))
	return train_metadata_df, val_metadata_df

def plot_(train_losses, val_losses, save_path, lr, wd):
	num_epochs = len(train_losses)
	if num_epochs == 1:
		return
	print(f"Saving Loss in {save_path}")
	epochs = range(1, num_epochs + 1)
	plt.figure(figsize=(11, 6))
	plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
	plt.plot(epochs, val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title(f'Training & Validation Loss vs. Epoch\nLR: {lr} wd: {wd}')
	plt.grid(True)
	plt.legend()
	plt.savefig(save_path)
	plt.close()  # Close the figure to free up memory

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

def get_args(obj):
	sig = inspect.signature(obj.__class__.__init__)
	important_args = []
	for param_name, param in sig.parameters.items():
		if param_name not in ['self', 'params', 'optimizer', 'verbose']:
			value = getattr(obj, param_name, param.default)
			print(param_name, value, param.default)
			if value != param.default:  # Only include if value is different from default
				important_args.append(f"{param_name}={value}")
	args = ",".join(important_args)
	args = re.sub(r'[^\w\-_\.]', '_', args)
	args = re.sub(r"_+", "_", args)
	return args

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

def tokenizer(text:str="sample label", encode:bool=True, mask=None, max_seq_length:int=128):
	# print(type(text), text)
	if encode: # Encode text => <class 'torch.Tensor'>
		# print(type(text), len(text), text)
		text = clean_(text=text, sw=STOPWORDS,)
		out = chr(2) + text + chr(3) # Adding SOT and EOT tokens
		if len(out) > max_seq_length:
			out = out[:max_seq_length]  # Truncate if length exceeds max_seq_length
		out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))]) # Add padding if needed
		out = torch.IntTensor(list(out.encode("utf-8"))) # Encode the text
		mask = torch.ones(len(out.nonzero()))
		if len(mask) < max_seq_length:  # Pad the mask to max_seq_length
			mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(torch.IntTensor)
		else:
			mask = mask.type(torch.IntTensor)
	else: # Decode <class 'torch.Tensor'> => text
		print(f"Decoding {type(text)} {text.shape}", end="\t")
		out = [chr(x) for x in text[1:len(mask.nonzero()) - 1]]
		out = "".join(out)
		mask = None
		print(f"=>> {out} & mask: {mask}")
	return out, mask # <class 'torch.Tensor'> torch.Size([max_seq_length]), <class 'torch.Tensor'> torch.Size([max_seq_length])

def get_info(dataloader):
	tot_samples = len(dataloader.dataset)
	n_chunks = len(dataloader) # ceil(tot_samples / dataloader.batch_size)
	print(
		# f"dataloader organization has main {len(next(iter(dataloader)))} element(s)\n"
		f"Total samples:: {tot_samples} "
		f"divided into {n_chunks} chunk(s) "
		f"| batch size: {dataloader.batch_size} "
		f"and {dataloader.num_workers} CPU(s)"
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

def get_doc_description(df, col:str="label"):
	unique_labels_list = list(df[col].unique())
	unique_labels_dict = {}
	for idx, lbl in enumerate(unique_labels_list):
		unique_labels_dict[idx] = lbl
	print(f"{len(list(unique_labels_dict.keys()))} unique_labels_dict:\n{json.dumps(unique_labels_dict, indent=2, ensure_ascii=False)}")
	key_type = type(next(iter(unique_labels_dict.keys())))
	value_type = type(next(iter(unique_labels_dict.values())))
	print(f"{len(list(unique_labels_dict.keys()))} unique_labels_dict:\n{unique_labels_dict}\nkeys: {key_type} | values: {value_type}")
	return unique_labels_dict, unique_labels_list

def custom_collate_fn(batch):
	# Filter out the None values from the batch
	batch = [item for item in batch if item is not None]
	# Use default collate function on the filtered batch
	return default_collate(batch)