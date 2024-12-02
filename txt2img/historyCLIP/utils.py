import os
import sys
import re
import math
from tqdm import tqdm
import argparse
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
from torchinfo import summary as tinfo
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from typing import List, Set, Dict, Tuple, Union
from functools import cache
import subprocess
import traceback
import multiprocessing
import logging
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from torch.utils.tensorboard import SummaryWriter
import nltk
import inspect
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

Image.MAX_IMAGE_PIXELS = None  # Disable the limit completely [decompression bomb]

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER

# if USER=="farid": # local laptop
# 	WDIR = os.path.join(HOME, "datasets")
# 	models_dir = os.path.join(WDIR, "trash", "models")
# 	# visualize = True
# elif USER=="alijanif": # Puhti
# 	WDIR = "/scratch/project_2004072/ImACCESS"
# 	models_dir = os.path.join(WDIR, "trash", "models")
# else: # Pouta
# 	WDIR = "/media/volume/ImACCESS"
# 	models_dir = os.path.join(WDIR, "models")
# 	ddir: str = os.path.join(WDIR, "myntradataset")

# if USER!="alijanif":
# 	import enchant
# 	import libvoikko
# 	fi_dict = libvoikko.Voikko(language="fi")	
# 	fii_dict = enchant.Dict("fi")
# 	sv_dict = enchant.Dict("sv_SE")
# 	sv_fi_dict = enchant.Dict("sv_FI")
# 	en_dict = enchant.Dict("en")
# 	de_dict = enchant.Dict("de")
# 	no_dict = enchant.Dict("no")
# 	da_dict = enchant.Dict("da")
# 	es_dict = enchant.Dict("es")
# 	et_dict = enchant.Dict("et")
	
# 	cs_dict = enchant.Dict("cs")
# 	cy_dict = enchant.Dict("cy")
# 	fo_dict = enchant.Dict("fo")
# 	fr_dict = enchant.Dict("fr")
# 	ga_dict = enchant.Dict("ga")
# 	hr_dict = enchant.Dict("hr")
# 	hu_dict = enchant.Dict("hu")
# 	is_dict = enchant.Dict("is")
# 	it_dict = enchant.Dict("it")
# 	lt_dict = enchant.Dict("lt")
# 	lv_dict = enchant.Dict("lv")
# 	nl_dict = enchant.Dict("nl")
# 	pl_dict = enchant.Dict("pl")
# 	sl_dict = enchant.Dict("sl")
# 	sk_dict = enchant.Dict("sk")

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

# @cache
# def clean_(text: str = "sample text", check_language:bool=False):
# 	text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Remove special characters and digits
# 	if check_language:
# 		text = remove_misspelled_(documents=text)
# 	words = nltk.tokenize.word_tokenize(text) # Tokenize the text into words
# 	# Filter out stopwords and words with fewer than 3 characters
# 	words = [word.lower() for word in words if len(word) >= 3 and word.lower() not in STOPWORDS]
# 	text = ' '.join(words) # Join the words back into a string
# 	text = re.sub(r'\boriginal caption\b', ' ', text)
# 	text = re.sub(r'\bdate taken\b', ' ', text)
# 	text = re.sub(r'\bcaption\b', ' ', text)
# 	text = re.sub(r'\bunidentified\b', ' ', text)
# 	text = re.sub(r'\bunnumbered\b', ' ', text)
# 	text = re.sub(r'\buntitled\b', ' ', text)
# 	text = re.sub(r'\bphotograph\b', ' ', text)
# 	text = re.sub(r'\bphoto\b', ' ', text)
# 	text = re.sub(r'\bdate\b', ' ', text)
# 	text = re.sub(r'\bdistrict\b', ' ', text)
# 	text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
# 	if len(text) == 0:
# 		return None
# 	return text

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
	print(f"<<< DF >>> [train] {train_metadata_df.shape} [val] {val_metadata_df.shape} Elapsed_t: {time.time()-metadata_st:.1f} sec".center(180, "-"))
	save_pickle(pkl=train_metadata_df, fname=train_metadata_df_fpth)
	save_pickle(pkl=val_metadata_df, fname=val_metadata_df_fpth)
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

def get_doc_description(df, col:str="colmun_name"):
	class_names = list(df[col].unique())
	captions = {idx: class_name for idx, class_name in enumerate(class_names)}
	# print(f"{len(list(captions.keys()))} Captions:\n{json.dumps(captions, indent=2, ensure_ascii=False)}")
	return captions, class_names

def get_mean_std_grayscale_img(dir: str="path/2/images"):
	print(f"Calculating Mean-Std for {len(os.listdir(dir))} Grayscale images (sequential approach => slow)")
	t0 = time.time()
	# Initialize variables to accumulate the sum and sum of squares
	sum_ = torch.tensor([0.0])
	sum_of_squares = torch.tensor([0.0])
	count = 0
	# Define the transform to convert images to tensors
	transform = T.Compose([
		T.Grayscale(num_output_channels=1),  # Convert to grayscale
		T.ToTensor(),  # Convert to tensor
	])
	# Iterate over all images in the dataset directory
	for filename in tqdm(os.listdir(dir)):
		if filename.endswith('.jpg') or filename.endswith('.png'):
			image_path = os.path.join(dir, filename)
			try:
				image = Image.open(image_path)
				tensor_image = transform(image)
				sum_ += tensor_image.sum()
				sum_of_squares += (tensor_image ** 2).sum()
				count += tensor_image.numel()
			except Exception as e:
				print(f"Error processing {image_path}: {e}")
	# Calculate the mean and std
	mean = sum_ / count
	std = torch.sqrt((sum_of_squares / count) - (mean ** 2))
	print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(100, " "))
	return mean.item(), std.item()

def get_mean_std_rgb_img(dir: str="path/2/images"):
	print(f"Calculating Mean-Std for {len(os.listdir(dir))} RGB images (sequential approach => slow)")
	t0 = time.time()
	# Initialize variables to accumulate the sum and sum of squares for each channel
	sum_ = torch.zeros(3)
	sum_of_squares = torch.zeros(3)
	count = 0
	# Define the transform to convert images to tensors
	transform = T.Compose([
		T.ToTensor(),  # Convert to tensor (automatically converts to RGB if not already)
	])
	# Iterate over all images in the dataset directory
	for filename in tqdm(os.listdir(dir)):
		if filename.endswith('.jpg') or filename.endswith('.png'):
			image_path = os.path.join(dir, filename)
			try:
				image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
				tensor_image = transform(image)
				sum_ += tensor_image.sum(dim=[1, 2])  # Sum over height and width dimensions
				sum_of_squares += (tensor_image ** 2).sum(dim=[1, 2])  # Sum of squares over height and width dimensions
				count += tensor_image.numel() / 3  # Total number of pixels per channel
			except Exception as e:
				print(f"Error processing {image_path}: {e}")
	# Calculate the mean and std for each channel
	mean = sum_ / count
	std = torch.sqrt((sum_of_squares / count) - (mean ** 2))
	print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(100, " "))
	return mean.tolist(), std.tolist() # return lists of length 3, corresponding to RGB channels

def process_grayscale_image(args):
		filename, dir, transform = args
		image_path = os.path.join(dir, filename)
		try:
				image = Image.open(image_path)
				tensor_image = transform(image)
				return tensor_image.sum(), (tensor_image ** 2).sum(), tensor_image.numel()
		except Exception as e:
				print(f"Error processing {image_path}: {e}")
				return 0, 0, 0

def get_mean_std_grayscale_img_multiprocessing(dir: str="path/2/images", num_workers: int=8):
		print(f"Calculating Mean-Std for {len(os.listdir(dir))} Grayscale images (multiprocessing: {num_workers} CPUs)")
		t0 = time.time()

		# Initialize variables to accumulate the sum and sum of squares
		sum_ = torch.tensor([0.0])
		sum_of_squares = torch.tensor([0.0])
		count = 0

		# Define the transform to convert images to tensors
		transform = T.Compose([
				T.Grayscale(num_output_channels=1),
				T.ToTensor(),
		])

		with ProcessPoolExecutor(max_workers=num_workers) as executor:
				futures = [executor.submit(process_grayscale_image, (filename, dir, transform)) for filename in os.listdir(dir)]
				for future in as_completed(futures):
						partial_sum, partial_sum_of_squares, partial_count = future.result()
						sum_ += partial_sum
						sum_of_squares += partial_sum_of_squares
						count += partial_count

		mean = sum_ / count
		std = torch.sqrt((sum_of_squares / count) - (mean ** 2))
		print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(100, " "))
		return mean.item(), std.item()

def process_rgb_image(args):
	filename, dir, transform = args
	image_path = os.path.join(dir, filename)
	# print(f"processing: {image_path}", end="\t")
	logging.info(f"Processing: {image_path}")
	try:
		Image.open(image_path).verify() # # Validate the image
		image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
		tensor_image = transform(image)
		return tensor_image.sum(dim=[1, 2]), (tensor_image ** 2).sum(dim=[1, 2]), tensor_image.numel() / 3
	except Exception as e:
		# print(f"Error processing {image_path}: {e}")
		logging.error(f"Error processing {image_path}: {e}")
		# traceback.print_exc()  # Print detailed traceback
		return torch.zeros(3), torch.zeros(3), 0

def get_mean_std_rgb_img_multiprocessing(dir: str="path/2/images", num_workers: int=8):
	print(f"Calculating Mean-Std « {len(os.listdir(dir))} RGB images » (multiprocessing with {num_workers} CPUs)")
	t0 = time.time()
	# Initialize variables to accumulate the sum and sum of squares for each channel
	sum_ = torch.zeros(3)
	sum_of_squares = torch.zeros(3)
	count = 0
	transform = T.Compose([
		T.ToTensor(),  # Convert to tensor (automatically converts to RGB if not already)
	])
	with ProcessPoolExecutor(max_workers=num_workers) as executor:
		futures = [executor.submit(process_rgb_image, (filename, dir, transform)) for filename in os.listdir(dir)]
		# for future in tqdm(as_completed(futures), total=len(futures)):
		for future in as_completed(futures):
			try:
				result = future.result()
				if result is not None:
					partial_sum, partial_sum_of_squares, partial_count = result
					sum_ += partial_sum
					sum_of_squares += partial_sum_of_squares
					count += partial_count
			except Exception as e:
				# print(f"Error in future result: {e}")
				logging.error(f"Error in future result: {e}")
				# traceback.print_exc()  # Print detailed traceback
	mean = sum_ / count
	std = torch.sqrt((sum_of_squares / count) - (mean ** 2))
	# print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(100, " "))
	logging.info(f"Elapsed_t: {time.time()-t0:.2f} sec")
	return mean.tolist(), std.tolist()

def custom_collate_fn(batch):
	# Filter out the None values from the batch
	batch = [item for item in batch if item is not None]
	# Use default collate function on the filtered batch
	return default_collate(batch)