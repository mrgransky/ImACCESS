import requests
import json
import time
import dill
import gzip
import pandas as pd
import numpy as np
import os
import sys
import datetime
import re
from typing import List, Dict
from natsort import natsorted
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from requests.exceptions import RequestException
import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import nltk

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

# Get the list of English stopwords
STOPWORDS = set(nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids()))

def process_rgb_image(args):
		filename, dir, transform = args
		image_path = os.path.join(dir, filename)
		try:
			Image.open(image_path).verify() # # Validate the image
			image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
			tensor_image = transform(image)
			return tensor_image.sum(dim=[1, 2]), (tensor_image ** 2).sum(dim=[1, 2]), tensor_image.numel() / 3
		except Exception as e:
			print(f"Error processing {image_path}: {e}")
			# traceback.print_exc()  # Print detailed traceback
			return torch.zeros(3), torch.zeros(3), 0

def get_mean_std_rgb_img_multiprocessing(dir: str="path/2/images", num_workers: int=8):
		print(f"Calculating Mean-Std for {len(os.listdir(dir))} RGB images (multiprocessing: {num_workers} CPUs)")
		t0 = time.time()
		# Initialize variables to accumulate the sum and sum of squares for each channel
		sum_ = torch.zeros(3)
		sum_of_squares = torch.zeros(3)
		count = 0
		# Define the transform to convert images to tensors
		transform = T.Compose([
				T.ToTensor(),  # Convert to tensor (automatically converts to RGB if not already)
		])
		with ProcessPoolExecutor(max_workers=num_workers) as executor:
				futures = [executor.submit(process_rgb_image, (filename, dir, transform)) for filename in os.listdir(dir)]
				# for future in tqdm(as_completed(futures), total=len(futures)):
				for future in as_completed(futures):
					try:
						partial_sum, partial_sum_of_squares, partial_count = future.result()
						sum_ += partial_sum
						sum_of_squares += partial_sum_of_squares
						count += partial_count
					except Exception as e:
						print(f"Error in future result: {e}")
						# traceback.print_exc()  # Print detailed traceback
		mean = sum_ / count
		std = torch.sqrt((sum_of_squares / count) - (mean ** 2))
		print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(100, " "))
		return mean.tolist(), std.tolist()

def check_url_status(url: str) -> bool:
	try:
		response = requests.head(url, timeout=50)
		# Return True only if the status code is 200 (OK)
		return response.status_code == 200
	except (requests.RequestException, Exception) as e:
		print(f"Error accessing URL {url}: {e}")
		return False

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
	print(f"Loaded in: {elpt:.3f} s | {type(pkl)} | {fsize:.3f} MB".center(130, " "))
	return pkl

def clean_(text: str = "sample text"):
	# Convert to lowercase
	text = text.lower()
	text = re.sub(r'original caption', '', text)
	# Remove special characters and digits
	text = re.sub(r'[^a-zA-Z\s]', '', text)
	# Tokenize the text into words
	words = nltk.tokenize.word_tokenize(text)
	# Filter out stopwords and words with fewer than 3 characters
	words = [word for word in words if len(word) >= 3 and word not in STOPWORDS]
	# Join the words back into a string
	text = ' '.join(words)
	# Normalize whitespace
	text = re.sub(r'\s+', ' ', text).strip()
	# TODO: some enchant cleaning for language check!
	return text