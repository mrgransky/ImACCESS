import requests
import json
import time
import dill
import gzip
import random
import pandas as pd
import numpy as np
import os
import sys
import datetime
import re
import argparse
import torch
import nltk
import multiprocessing
import shutil
import logging
from typing import List, Dict, Set, Tuple, Union
from bs4 import BeautifulSoup
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor, TimeoutError
from requests.exceptions import RequestException
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from functools import cache
from urllib.parse import urlparse, unquote, quote_plus
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import timedelta
import glob
import psutil  # For memory usage monitoring
import tabulate
from scipy.stats import gaussian_kde, t
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

logging.basicConfig(level=logging.INFO)
Image.MAX_IMAGE_PIXELS = None  # Disable the limit completely [decompression bomb]

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

if USER!="alijanif":
	import enchant
	import libvoikko
	fi_dict = libvoikko.Voikko(language="fi")	
	fii_dict = enchant.Dict("fi")
	sv_dict = enchant.Dict("sv_SE")
	sv_fi_dict = enchant.Dict("sv_FI")
	en_dict = enchant.Dict("en")
	de_dict = enchant.Dict("de")
	no_dict = enchant.Dict("no")
	da_dict = enchant.Dict("da")
	es_dict = enchant.Dict("es")
	et_dict = enchant.Dict("et")
	
	cs_dict = enchant.Dict("cs")
	cy_dict = enchant.Dict("cy")
	fo_dict = enchant.Dict("fo")
	fr_dict = enchant.Dict("fr")
	ga_dict = enchant.Dict("ga")
	hr_dict = enchant.Dict("hr")
	hu_dict = enchant.Dict("hu")
	is_dict = enchant.Dict("is")
	it_dict = enchant.Dict("it")
	lt_dict = enchant.Dict("lt")
	lv_dict = enchant.Dict("lv")
	nl_dict = enchant.Dict("nl")
	pl_dict = enchant.Dict("pl")
	sl_dict = enchant.Dict("sl")
	sk_dict = enchant.Dict("sk")

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

def get_stratified_split(
	df:pd.DataFrame,
	val_split_pct:float=0.2,
	figure_size:tuple=(11, 7),
	dpi:int=200,
	result_dir:str="result_directory",
	dname:str="DATASET_NAME",
	seed:int=42,
	):
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
	train_df.to_csv(os.path.join(result_dir, 'metadata_train.csv'), index=False)
	val_df.to_csv(os.path.join(result_dir, 'metadata_val.csv'), index=False)

	# Visualize label distribution in training and validation sets
	plt.figure(figsize=figure_size)
	train_df['label'].value_counts().plot(kind='bar', color='blue', alpha=0.6, label=f'Train {1-val_split_pct}')
	val_df['label'].value_counts().plot(kind='bar', color='red', alpha=0.9, label=f'Validation {val_split_pct}')
	plt.title(f'{dname} Stratified Sampling Label Distribution of {train_df.shape[0]} Training samples {val_df.shape[0]} Validation Samples (Total samples: {df_filtered.shape[0]})', fontsize=9)
	plt.xlabel('Label')
	plt.ylabel('Frequency')
	plt.yticks(rotation=90, fontsize=9)
	plt.legend(loc='best', ncol=2, frameon=False, fontsize=8)
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(result_dir, 'outputs', f'{dname}_stratified_random_sampling_label_distribution.png'),
		dpi=dpi,
		bbox_inches='tight'
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

def plot_year_distribution(
		df: pd.DataFrame,
		dname: str,
		fpth: str,
		BINs: int = 50,
		FIGURE_SIZE: tuple = (18, 9),
		DPI: int = 250,
	):
	# matplotlib.rcParams['font.family'] = ['Source Han Sans TW', 'sans-serif']
	# print(natsorted(matplotlib.font_manager.get_font_names()))
	# plt.rcParams["font.family"] = "DejaVu Math TeX Gyre"

	# Convert 'doc_date' to datetime and handle invalid entries
	df['doc_date'] = pd.to_datetime(df['doc_date'], errors='coerce')
	
	# Extract valid dates (non-NaN)
	valid_dates = df['doc_date'].dropna()
	
	# Handle edge case: no valid dates
	if valid_dates.empty:
		plt.figure(figsize=FIGURE_SIZE)
		plt.text(0.5, 0.5, "No valid dates available for plotting", ha='center', va='center', fontsize=12)
		plt.title(f'{dname} Year Distribution - No Data')
		plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
		plt.close()
		return
	
	# Compute start and end dates from data
	start_date = valid_dates.min().strftime('%Y-%m-%d')
	end_date = valid_dates.max().strftime('%Y-%m-%d')
	start_year = valid_dates.min().year
	end_year = valid_dates.max().year
	print(f"start_year: {start_year} | end_year: {end_year}")
	
	# Extract the year from the 'doc_date' column (now as integer)
	df['year'] = df['doc_date'].dt.year  # This will have NaN where doc_date is NaT
	
	# Filter out None values (though dt.year gives NaN, which .dropna() handles)
	year_series = df['year'].dropna().astype(int)
	# Find the years with the highest and lowest frequencies (handle ties)
	year_counts = year_series.value_counts()
	max_hist_freq = max(year_counts.values)
	min_hist_freq = min(year_counts.values)
	print(f"max_hist_freq: {max_hist_freq} | min_hist_freq: {min_hist_freq}")
	
	# Get the years with the maximum and minimum frequencies
	max_freq = year_counts.max()
	min_freq = year_counts.min()
	max_freq_years = year_counts[year_counts == max_freq].index.tolist()
	min_freq_years = year_counts[year_counts == min_freq].index.tolist()
	
	# Calculate mean, median, and standard deviation
	mean_year = year_series.mean()
	median_year = year_series.median()
	std_year = year_series.std()
	# Calculate 95% confidence interval for the mean
	confidence_level = 0.95
	n = len(year_series)
	mean_conf_interval = stats.t.interval(confidence_level, df=n-1, loc=mean_year, scale=stats.sem(year_series))
	# Get the overall shape of the distribution
	distribution_skew = year_series.skew()
	distribution_kurtosis = year_series.kurtosis()
	skew_desc = "right-skewed" if distribution_skew > 0 else "left-skewed" if distribution_skew < 0 else "symmetric"
	kurt_desc = "heavy-tailed" if distribution_kurtosis > 0 else "light-tailed" if distribution_kurtosis < 0 else "normal-tailed"
	# Calculate percentiles
	q25, q75 = year_series.quantile([0.25, 0.75])
	# Plot KDE using scipy.stats.gaussian_kde
	plt.figure(figsize=FIGURE_SIZE)
	sns.histplot(
		year_series,
		bins=BINs,
		color="skyblue",
		kde=True,
		edgecolor="white",
		alpha=0.95,
		linewidth=1.5,
		label="Year Distribution Histogram"
	)
	# Create the KDE object and adjust bandwidth to match Seaborn's default behavior
	kde = gaussian_kde(year_series, bw_method='scott')  # Use 'scott' or 'silverman', or a custom value
	x_range = np.linspace(start_year, end_year, 300)
	kde_values = kde(x_range)
	bin_width = (end_year - start_year) / BINs  # Approximate bin width of the histogram
	kde_scaled = kde_values * len(year_series) * bin_width  # Scale KDE to match frequency
	plt.plot(
		x_range,
		kde_scaled,
		color="grey", # dark gray
		linewidth=2.0,
		linestyle="-",
		label="Kernel Density Estimate (KDE)",
	)
	world_war_1 = [1914, 1918]
	world_war_2 = [1939, 1945]
	padding = 1.25
	max_padding = 1.3
	# Add shaded regions for WWI and WWII (plot these first to ensure they are in the background)
	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		plt.axvspan(world_war_1[0], world_war_1[1], color='#ff3a2d', alpha=0.2, label='World War One')

	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		plt.axvspan(world_war_2[0], world_war_2[1], color='#9aff33', alpha=0.2, label='World War Two')

	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		for year in world_war_1:
			plt.axvline(x=year, color='r', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_1[0] + world_war_1[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWI',
			color='red',
			fontsize=12,
			fontweight="bold",
			ha="center",  # horizontal alignment
		)
	
	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		for year in world_war_2:
			plt.axvline(x=year, color='g', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_2[0] + world_war_2[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWII',
			color='green',
			fontsize=12,
			fontweight="bold",
			ha="center", # horizontal alignment
		)
	# Add visual representations of key statistics
	plt.axvline(x=mean_year, color='navy', linestyle='-.', lw=1.5, label=f'Mean Year: {mean_year:.2f}')
	# plt.text(
	# 	x=mean_year-0.5, # Shift slightly to the left for better visibility
	# 	y=max_freq * padding,
	# 	rotation=90,
	# 	s='Mean',
	# 	color='navy',
	# 	fontsize=12,
	# 	fontweight="bold",
	# 	va="center",  # vertical alignment
	# 	ha="center",  # horizontal alignment
	# )

	plt.axvspan(mean_year - std_year, mean_year + std_year, color='yellow', alpha=0.16, label='Mean ± 1 SD')
	valid_count = len(year_series)
	stats_text = (
			f"Samples with valid dates: {valid_count} (~{round(valid_count / df.shape[0] * 100)}%)\n\n"
			"Frequency Statistics:\n"
			f"  Most frequent year(s): {', '.join(map(str, max_freq_years))} ({max_freq} images)\n"
			f"  Least frequent year(s): {', '.join(map(str, min_freq_years))} ({min_freq} images)\n\n"
			"Central Tendency [Year]:\n"
			f"  Mean: {mean_year:.2f}\n"
			f"  Mean 95% CI: [{mean_conf_interval[0]:.2f}, {mean_conf_interval[1]:.2f}]\n"
			f"  Median: {median_year:.2f}\n"
			f"  Standard deviation: {std_year:.2f}\n\n"
			"Percentiles:\n"
			f"  25th: {q25:.2f}\n"
			f"  75th: {q75:.2f}\n\n"
			"Distribution Shape:\n"
			f"  Skewness: {distribution_skew:.2f} ({skew_desc})\n"
			f"  Kurtosis: {distribution_kurtosis:.2f} ({kurt_desc})"
	)
	plt.text(
		0.01, 
		0.98,
		stats_text,
		transform=plt.gca().transAxes,
		ha='left',
		va='top',
		fontsize=8.5,
		color='black',
		bbox=dict(boxstyle='round,pad=0.5',facecolor='white', alpha=0.6, edgecolor='gray')
	)
	plt.title(f'{dname} Year Distribution {start_date} - {end_date} Total Images: {df.shape[0]}')
	plt.xlabel('Year')
	plt.ylabel('Frequency')
	plt.ylim(0, max_freq * max_padding)  # Add some padding to the y-axis
	plt.yticks(fontsize=9, rotation=90)
	plt.xlim(start_year - 2, end_year + 2)
	plt.legend(
		loc='center left',
		fontsize=10,
		framealpha=0.95,
		frameon=True,
		shadow=True,
		fancybox=True,
	)
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

def plot_label_distribution(
		df: pd.DataFrame,
		start_date: str, 
		end_date: str, 
		dname: str,
		fpth: str,
		figure_size: tuple = (12, 8),
		DPI: int = 250,
	):
	label_counts = df['label'].value_counts()
	plt.figure(figsize=figure_size)
	label_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dname} {start_date} - {end_date} Label Distribution (unique: {label_counts.shape[0]}) Total samples: {df.shape[0]}', fontsize=10)
	plt.xlabel('label')
	plt.ylabel('Frequency', fontsize=10)
	plt.xticks(fontsize=9)
	plt.yticks(rotation=90, fontsize=9)
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight',)

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

def extract_url_info(url):
	parsed_url = urlparse(url)
	base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/gallery" # Extract the base URL
	path_components = parsed_url.path.strip('/').split('/') # Split the path into components		
	# Extract country, main_label, and type
	country = path_components[1] if len(path_components) > 1 else None
	main_label = path_components[2] if len(path_components) > 2 else None
	type_ = path_components[3] if len(path_components) > 3 else None
	# Decode URL-encoded characters (if any)
	if main_label:
		main_label = unquote(main_label)
		main_label = re.sub(r'[^a-zA-Z\s]', ' ', main_label) # Remove special characters and digits
		main_label = re.sub(r'\s+', ' ', main_label)  # Remove extra whitespace
	if type_:
		type_ = unquote(type_)
	return {
		"base_url": base_url,
		"country": country,
		"main_label": main_label,
		"type": type_
	}

def clean_(text, sw, check_language:bool=False):
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