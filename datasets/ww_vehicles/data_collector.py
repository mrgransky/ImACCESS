import os
import csv
import json
import re
import time
import requests
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import shutil
from tqdm import tqdm

# List of URLs to scrape
URLS = [
		"https://truck-encyclopedia.com/ww2/czechoslovakia/czech-trucks.php",
		"https://truck-encyclopedia.com/ww2/france/french-trucks.php",
		"https://truck-encyclopedia.com/ww2/germany/german-trucks.php",
		"https://truck-encyclopedia.com/ww2/italy/italian-trucks.php",
		"https://truck-encyclopedia.com/ww2/japan/IJA_trucks.php",
		"https://truck-encyclopedia.com/ww2/us/us-trucks.php",
		"https://truck-encyclopedia.com/ww2/uk/british-trucks.php",
		"https://truck-encyclopedia.com/ww2/ussr/soviet-trucks.php",
		"https://truck-encyclopedia.com/ww1/trucks.php",
]

DATASET_DIRECTORY = "/home/farid/datasets/WW_DATASETs/WW_VEHICLES"
THUMBNAIL_DIRECTORY = os.path.join(DATASET_DIRECTORY, "thumbnails")
IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")
os.makedirs(DATASET_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(THUMBNAIL_DIRECTORY, exist_ok=True)

CSV_FILENAME = os.path.join(DATASET_DIRECTORY, "metadata.csv")

# Known flag images to exclude
FLAG_FILENAMES = {
		"gb_r.gif", "france_r.jpg", "russ_r.jpg", "usa_r.jpg", "italy_r.jpg",
		"jap_r.jpg", "belgium_r.jpg", "axis_r.jpg", "all_imperial_r.jpg",
		"austriahun_r.gif", "turkey_r.jpg", "allies_r.jpg", "unitedkingdom_r.jpg",
		"canada_r.jpg", "poland_r.jpg", "soviet_r.jpg", "czech_r.jpg",
		"nazi_r.jpg", "china_r.jpg", "east-germany_r.jpg", "west-germany_r.jpg",
		"uk_r.jpg", "uk_r.gif", "jp_r.jpg", "ussr_r.jpg", "germany_r.jpg",
		"challenge-coins.jpg", "italww2_r.jpg", "wehrmacht_structure_1939-1945.png",
}

SLIDE_PATTERN = re.compile(r"slide\d+\.jpg", re.IGNORECASE)

# Headers to mimic a browser request
HEADERS = {
		"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/114.0.0.0 Safari/537.36"
		)
}

def download_image(
		img_url, 
		folder,
		max_retries=3, 
		delay=3
	):
	filename = os.path.basename(urlparse(img_url).path)
	filepath = os.path.join(folder, filename)
	
	# Check if image already exists
	if os.path.exists(filepath):
		print(f"Skipping download, image already exists: {filepath}")
		return filepath
	
	# Proceed with download if image doesn't exist
	for attempt in range(1, max_retries + 1):
		try:
			response = requests.get(img_url, stream=True, timeout=10, headers=HEADERS)
			response.raise_for_status()
			with open(filepath, "wb") as f:
					for chunk in response.iter_content(1024):
							f.write(chunk)
			print(f"Downloaded {img_url} to {filepath}")
			return filepath
		except Exception as e:
			print(f"Attempt {attempt} failed to download {img_url}: {e}")
			if attempt < max_retries:
				time.sleep(delay)
			else:
				print(f"Giving up on {img_url} after {max_retries} attempts.")
				with open("failed_downloads.log", "a", encoding="utf-8") as log_file:
					log_file.write(f"{img_url} - {e}\n")
				return None

def infer_event_and_country(url):
	parsed = urlparse(url)
	parts = parsed.path.strip("/").split("/")
	event = None
	country = None
	if len(parts) >= 2:
		event = parts[0].lower()  # 'ww1' or 'ww2'
		# For WW1 URL, set country to empty string as it covers multiple countries
		if url == "https://truck-encyclopedia.com/ww1/trucks.php":
				country = ""
		else:
				country = parts[1].lower()  # e.g., 'czechoslovakia', 'france'
	return event, country

def normalize_img_id(img_url):
	parsed = urlparse(img_url)
	path = parsed.path.lower()
	return path

def clean_text(text):
	if not text:
		return ""
	text = re.sub(r'\s+', ' ', text.strip())
	return text[:200]  # Limit to 200 characters

def extract_metadata_and_images(
		html_content, 
		base_url, 
		event, 
		country, 
		downloaded_ids_global
	):
	soup = BeautifulSoup(html_content, "html.parser")
	data = []
	content_div = soup.find("div", class_="content")
	if not content_div:
		print("Main content div not found.")
		return data
	images = content_div.find_all("img")
	for img in images:
		img_src = img.get("src")
		if not img_src:
			print("Skipping image with no src attribute")
			continue
		img_url = urljoin(base_url, img_src)
		img_id = os.path.basename(urlparse(img_url).path).lower()
		# Filters
		if img_id.endswith(".gif"):
			print(f"Skipping GIF image: {img_id}")
			continue
		if img_id in FLAG_FILENAMES:
			print(f"Skipping flag image: {img_id}")
			continue
		if SLIDE_PATTERN.match(img_id):
			print(f"Skipping slide image: {img_id}")
			continue
		if img_id in downloaded_ids_global:
			print(f"Skipping duplicate image ID: {img_id}")
			continue
		# Extract label from img alt
		label = clean_text(img.get("alt", ""))
		# Extract title (prefer <a> tag's title, then img's title)
		title = ""
		parent_a = img.find_parent("a")
		if parent_a and parent_a.get("title"):
			title = clean_text(parent_a.get("title"))
			print(f"Using <a> title for {img_id}: {title}")
		elif img.get("title"):
			title = clean_text(img.get("title"))
			print(f"Using <img> title for {img_id}: {title}")
		else:
			print(f"No title found for {img_id}")
		# Extract description (prefer <em> tag after image, then parent text)
		description = ""
		next_em = img.find_next("em")
		if next_em and next_em.get_text(strip=True):
			description = clean_text(next_em.get_text(strip=True))
			print(f"Using <em> description for {img_id}: {description}")
		elif img.parent:
			parent_text = clean_text(img.parent.get_text(separator=" ", strip=True))
			if parent_text and parent_text != label and parent_text != title:
				description = parent_text
				print(f"Using parent text description for {img_id}: {description}")
		# Download image (or skip if already exists)
		local_img_path = download_image(
			img_url=img_url,
			folder=IMAGE_DIRECTORY,
		)
		if not local_img_path:
			print(f"Skipping metadata for missing image: {img_url}")
			continue
		downloaded_ids_global.add(img_id)
		
		# Define thumbnail path (to be populated later)
		thumbnail_filename = os.path.basename(local_img_path)
		thumbnail_path = os.path.join(THUMBNAIL_DIRECTORY, thumbnail_filename)

		# Combine label, title, and description
		enriched_document_description = " ".join(filter(None, [label, title, description])).strip()
		data.append({
			"id": img_id,
			"label": label,  # Empty string if no alt
			"title": title,  # Empty string if no title
			"country": country or "",  # Empty string if no country
			"description": description,  # Empty string if no description
			"img_path": local_img_path,
			"img_url": img_url,
			"enriched_document_description": enriched_document_description or "",  # Empty string if all are empty
			"event": event or "",  # Empty string if no event
			"thumbnail_path": thumbnail_path,
		})
	return data

def save_metadata(data, filename):
		"""
		Save metadata to CSV and Excel files using pandas.
		
		Args:
				data: List of dictionaries containing metadata.
				filename: Base filename for CSV (Excel will use .xlsx extension).
		"""
		if not data:
				print("No data to save.")
				return
		
		# Convert list of dicts to DataFrame
		df = pd.DataFrame(data)
		
		# Save to CSV
		csv_path = filename
		df.to_csv(csv_path, index=False, encoding='utf-8')
		print(f"Saved {len(df)} entries to {csv_path}")
		
		# Save to Excel
		excel_path = os.path.splitext(filename)[0] + '.xlsx'
		try:
				df.to_excel(excel_path, index=False, engine='openpyxl')
				print(f"Saved {len(df)} entries to {excel_path}")
		except Exception as e:
				print(f"Failed to save Excel file {excel_path}: {e}")

def create_thumbnail(
		dataset_metadata: list[dict], 
		size=(500, 500), 
		large_image_threshold_mb=2.0,
	):
	print(f"Creating thumbnails for {len(dataset_metadata)} images...")
	large_image_threshold_bytes = large_image_threshold_mb * 1024 * 1024
	for data in tqdm(dataset_metadata, desc="Creating thumbnails"):
		img_path = data.get("img_path")
		thumbnail_path = data.get("thumbnail_path")
		if not img_path or not os.path.exists(img_path):
			print(f"Skipping thumbnail for missing image: {img_path}")
			continue
		if os.path.exists(thumbnail_path):
			print(f"Thumbnail already exists: {thumbnail_path}")
			continue
		try:
			file_size_bytes = os.path.getsize(img_path)
			is_large = file_size_bytes > large_image_threshold_bytes
			with Image.open(img_path).convert("RGB") as img:
				if is_large:
					print(f"Creating thumbnail for large image ({file_size_bytes / 1024 / 1024:.2f} MB): {img_path}")
					# Create thumbnail
					img.thumbnail(size, resample=Image.Resampling.LANCZOS)
					img.save(
						fp=thumbnail_path,
						format="JPEG",
						quality=100,
						optimize=True,
						progressive=True,
					)
					print(f"Created thumbnail: {thumbnail_path}")
				else:
					shutil.copy2(img_path, thumbnail_path)
		except Exception as e:
			print(f"Failed to create thumbnail for {img_path}: {e}")

def main():
	if not os.path.exists(CSV_FILENAME):
		print(f"CSV file {CSV_FILENAME} does not exist. Creating it...")
		dataset_metadata = []
		downloaded_ids_global = set()  # Global set to track downloaded images
		for url in URLS:
			print(f"\nProcessing {url} ...")
			event, country = infer_event_and_country(url)
			if not event:
				print(f"Could not infer event from URL: {url}")
				continue
			try:
				response = requests.get(url, headers=HEADERS, timeout=10)
				response.raise_for_status()
				html_content = response.text
			except Exception as e:
				print(f"Failed to download page {url}: {e}")
				continue
			data = extract_metadata_and_images(html_content, url, event, country, downloaded_ids_global)
			dataset_metadata.extend(data)
			print(f"Extracted {len(data)} entries from {url}")
		save_metadata(dataset_metadata, CSV_FILENAME)
		print(f"\nDone. Total images downloaded and metadata saved: {len(dataset_metadata)}")
	else:
		print(f"Loading {CSV_FILENAME}...")
		df = pd.read_csv(CSV_FILENAME, encoding='utf-8')
		dataset_metadata = df.to_dict(orient='records')
		print(f"Loaded {type(df)} {df.shape} with {len(dataset_metadata)} entries from {CSV_FILENAME}")

	print(json.dumps(dataset_metadata[0], indent=2, ensure_ascii=False))

	# Creating Thumbnails
	create_thumbnail(dataset_metadata=dataset_metadata)
	
if __name__ == "__main__":
	main()