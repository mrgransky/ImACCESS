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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from requests.exceptions import RequestException, HTTPError, SSLError
import argparse

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', type=str, default="1890-01-01", help='Dataset DIR')
parser.add_argument('--end_date', type=str, default="1960-01-01", help='Dataset DIR')
parser.add_argument('--num_worker', type=int, default=8, help='Number of CPUs')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
# run in local laptop:
# $ python data_collector.py --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01
# $ nohup python data_collector.py -u --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01 >> europeana_image_download.out &

START_DATE = args.start_date
END_DATE = args.end_date
dataset_name = "europeana"
nw:int = min(args.num_worker, multiprocessing.cpu_count()) # def: 8
europeana_api_base_url: str = "https://api.europeana.eu/record/v2/search.json"
europeana_api_key: str = "plaction"
# europeana_api_key: str = "api2demo"
# europeana_api_key: str = "nLbaXYaiH"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}
os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
RESULT_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")

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

def get_data(st_date: str="1914-01-01", end_date: str="1914-01-02", query: str="world war"):
	t0 = time.time()
	query_processed = re.sub(" ", "_", query.lower())
	query_all_hits_fpth = os.path.join(RESULT_DIRECTORY, f"results_{st_date}_{end_date}_query_{query_processed}.gz")
	try:
		query_all_hits = load_pickle(fpath=query_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all docs of National Archive for Query: « {query} » ... it might take a while..")
		params = {
			'wskey': europeana_api_key,
			'qf': [
				'collection:photography', 
				'TYPE:"IMAGE"', 
				# 'contentTier:"4"', # high quality images
				'MIME_TYPE:image/jpeg',
				# 'LANGUAGE:en',
			],
			'rows': 100,
			'query': query,
			'reusability': 'open'
		}
		query_all_hits = []
		start = 1
		while True:
			loop_st = time.time()
			params["start"] = start
			response = requests.get(
				europeana_api_base_url,
				params=params,
				headers=headers,
			)
			if response.status_code == 200:
				data = response.json()
				if 'items' in data:
					# Extract the 'items' field
					hits = data['items']
					total_hits = data['totalResults']
					print(total_hits, len(hits))
					query_all_hits.extend(hits)
					# print(json.dumps(query_all_hits, indent=2, ensure_ascii=False))
					print(f"start: {start}:\tFound: {len(hits)} {type(hits)}\t{len(query_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
				if len(query_all_hits) >= total_hits:
					break
				start += params.get("rows")
			else:
				print(f"Failed to retrieve data: status_code: {response.status_code}")
				break
		if len(query_all_hits) == 0:
			return
		save_pickle(pkl=query_all_hits, fname=query_all_hits_fpth)
	print(f"Total hit(s): {len(query_all_hits)} {type(query_all_hits)} for query: « {query} » found in {time.time()-t0:.2f} sec")
	return query_all_hits

def check_url_status(url: str) -> bool:
	try:
		response = requests.head(url, timeout=50)
		# Return True only if the status code is 200 (OK)
		return response.status_code == 200
	except (requests.RequestException, Exception) as e:
		print(f"Error accessing URL {url}: {e}")
		return False

def get_dframe(query: str="query", docs: List=[Dict]):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	# Output the extracted items (metadata and descriptions of the images)
	# for item_idx, item in enumerate(items):
	# 	print(f"item: {item_idx}")
	# 	# print(list(item.keys()))
	# 	print(item.get("id"))
	# 	print(item.get("title"))
	# 	print(item.get("edmIsShownAt"))
	# 	print(item.get("edmIsShownBy"))
	# 	print(item.get("edmTimespanLabel"))
	# 	print(item.get("language"))
	# 	print(item.get("dataProvider"))
	# 	print(item.get("provider"))
	# 	print("#"*100)
	# print(json.dumps(item, indent=2))  # Pretty-print the JSON data for each item
	# You can process or save the 'items' data as needed

	data = []
	# print(type(docs), len(docs))
	for doc_idx, doc in enumerate(docs):
		# print(doc_idx, type(doc), len(doc))
		# print(list(doc.keys()))
		# print(doc.get("id"))
		# print(doc.get("title"))
		# print(doc.get("edmIsShownAt"))
		# print(doc.get("edmIsShownBy"))
		# print(doc.get("edmTimespanLabel"))
		# print(doc.get("language"))
		# print(doc.get("dataProvider"))
		# print(doc.get("provider"))
		# if doc.get("dcDescription"):
		# 	print(len(doc.get("dcDescription")), doc.get("dcDescription"))
		# print("#"*100)

		pDate = doc.get("edmTimespanLabel")[0].get("def") if doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def") else None
		image_url = doc.get("edmIsShownBy")[0]
		if image_url and (image_url.endswith('.jpg') or image_url.endswith('.png')):
			#################################################################
			# # without checking status_code [faster but broken URL]
			# first_digital_object_url = first_digital_object_url
			#################################################################
			#################################################################
			# # with checking status_code [slower but healty URL]
			# print(f"{first_digital_object_url:<140}", end=" ")
			# st_t = time.time()
			if check_url_status(image_url): # status code must be 200!
				image_url = image_url
			else:
				image_url = None
			# print(f"{time.time()-st_t:.2f} s")
			#################################################################
		else:
			image_url = None
		row = {
			'id': doc.get("id"),
			'query': query,
			'title': doc.get("title"),
			'description': doc.get("dcDescription"),
			'img_url': image_url,
			'date': pDate,
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed_t: {time.time()-df_st_time:.1f} sec")
	# print(df.head(10))
	return df

def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
	t0 = time.time()
	rIdx = row.name
	url = row['img_url']
	image_name = re.sub("/", "LBL", row['id']) # str(row['id']) + os.path.splitext(url)[1]
	image_path = os.path.join(image_dir, image_name)
	if os.path.exists(image_path): # avoid redownloading
		# print(f"File {image_name} already exists, skipping download.")
		return image_name
	attempt = 0
	while attempt < retries: # Retry mechanism
		try:
			response = session.get(url, timeout=20)
			response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
			with open(image_path, 'wb') as f:
				f.write(response.content)
			print(f"{rIdx}/{total_rows:<20} Saved {image_name:<80}in {time.time()-t0:.1f} s")
			return image_name
		except (RequestException, IOError, Exception) as e:
			attempt += 1
			print(f"{rIdx}/{total_rows:<20} Downloading {image_name}\t{e}\tRetrying ({attempt}/{retries})...")
			time.sleep(backoff_factor * (3 ** attempt))  # Exponential backoff
	print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
	return None

def get_images(df):
	print(f"Saving images of {df.shape[0]} records using {nw} CPUs...")
	os.makedirs(os.path.join(RESULT_DIRECTORY, "images"), exist_ok=True)
	IMAGE_DIR = os.path.join(RESULT_DIRECTORY, "images")
	with requests.Session() as session: # Start a session for connection reuse
		with ThreadPoolExecutor(max_workers=nw) as executor: # Use ThreadPoolExecutor for parallel downloads
			futures = [executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]) for _, row in df.iterrows()]
			for future in as_completed(futures): # Process results as they complete
				try:
					future.result()
				except Exception as e:
					print(f"Unexpected ERR: {e}")
	print(f"Total number of images downloaded: {len(os.listdir(IMAGE_DIR))}")

# def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
# 		t0 = time.time()
# 		rIdx = row.name
# 		url = row['img_url']
# 		image_name = re.sub("/", "LBL", row['id']) # str(row['id']) + os.path.splitext(url)[1]
# 		image_path = os.path.join(image_dir, image_name)
# 		if os.path.exists(image_path): # avoid redownloading
# 				# print(f"File {image_name} already exists, skipping download.")
# 				return image_name
# 		attempt = 0
# 		while attempt < retries: # Retry mechanism
# 				try:
# 						response = session.get(url, timeout=20)
# 						response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
# 						with open(image_path, 'wb') as f:
# 								f.write(response.content)
# 						print(f"{rIdx}/{total_rows:<20} Saved {image_name:<80} in {time.time()-t0:.1f} s")
# 						return image_name
# 				except HTTPError as http_err:
# 						if http_err.response.status_code == 404:
# 								print(f"{rIdx}/{total_rows:<20} 404 Not Found for {image_name}, skipping...")
# 								return None
# 						else:
# 								attempt += 1
# 								print(f"{rIdx}/{total_rows:<20} Downloading {image_name}\t{http_err}\tRetrying ({attempt}/{retries})...")
# 								time.sleep(backoff_factor * (3 ** attempt))  # Exponential backoff
# 				except (RequestException, IOError, Exception) as e:
# 						attempt += 1
# 						print(f"{rIdx}/{total_rows:<20} Downloading {image_name}\t{e}\tRetrying ({attempt}/{retries})...")
# 						time.sleep(backoff_factor * (3 ** attempt))  # Exponential backoff
# 		print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
# 		return None

# def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
# 		t0 = time.time()
# 		rIdx = row.name
# 		url = row['img_url']
# 		image_name = re.sub("/", "LBL", row['id']) # str(row['id']) + os.path.splitext(url)[1]
# 		image_path = os.path.join(image_dir, image_name)
# 		if os.path.exists(image_path): # avoid redownloading
# 				# print(f"File {image_name} already exists, skipping download.")
# 				return image_name
# 		attempt = 0
# 		while attempt < retries: # Retry mechanism
# 				try:
# 						response = session.get(url, timeout=20)
# 						response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
# 						with open(image_path, 'wb') as f:
# 								f.write(response.content)
# 						print(f"{rIdx}/{total_rows:<20} Saved {image_name:<80} in {time.time()-t0:.1f} s")
# 						return image_name
# 				except HTTPError as http_err:
# 						if http_err.response.status_code == 404:
# 								print(f"{rIdx}/{total_rows:<20} 404 Not Found for {image_name}, skipping...")
# 								return None
# 						elif http_err.response.status_code == 403:
# 								print(f"{rIdx}/{total_rows:<20} 403 Forbidden for {image_name}, skipping...")
# 								return None
# 						else:
# 								attempt += 1
# 								print(f"{rIdx}/{total_rows:<20} Downloading {image_name}\t{http_err}\tRetrying ({attempt}/{retries})...")
# 								time.sleep(backoff_factor * (3 ** attempt))  # Exponential backoff
# 				except SSLError as ssl_err:
# 						print(f"{rIdx}/{total_rows:<20} SSL Error for {image_name}, skipping...")
# 						return None
# 				except (RequestException, IOError, Exception) as e:
# 						attempt += 1
# 						print(f"{rIdx}/{total_rows:<20} Downloading {image_name}\t{e}\tRetrying ({attempt}/{retries})...")
# 						time.sleep(backoff_factor * (3 ** attempt))  # Exponential backoff
# 		print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
# 		return None

# def get_images(df):
# 		print(f"Saving images of {df.shape[0]} records using {nw} CPUs...")
# 		os.makedirs(os.path.join(RESULT_DIRECTORY, "images"), exist_ok=True)
# 		IMAGE_DIR = os.path.join(RESULT_DIRECTORY, "images")
# 		with requests.Session() as session: # Start a session for connection reuse
# 				with ThreadPoolExecutor(max_workers=nw) as executor: # Use ThreadPoolExecutor for parallel downloads
# 						futures = [executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]) for _, row in df.iterrows()]
# 						for future in as_completed(futures): # Process results as they complete
# 								try:
# 										future.result()
# 								except Exception as e:
# 										print(f"Unexpected ERR: {e}")
# 		print(f"Total number of images downloaded: {len(os.listdir(IMAGE_DIR))}")

def main():
	dfs = []
	all_query_tags = [
		"Ballistic missile",
		"flame thrower",
		"flamethrower",
		"Power Plant",
		'Nazi crime',
		"Nazi victim",
		"Constitution",
		"road construction",
		"rail construction",
		"dam construction",
		"tunnel construction",
		"Helicopter",
		"Manufacturing Plant",
		"naval aircraft factory",
		"naval air station",
		"naval air base",
		"terminal",
		"trench warfare",
		"explosion",
		"soldier",
		"Submarine",
		"allied force",
		"propaganda",
		"cemetery",
		"graveyard",
		"bayonet",
		"war bond",
		"air force base",
		"air force personnel",
		"air force station",
		"Artillery",
		"Rifle",
		"barrel",
		"Air bomb",
		"air raid",
		"flag",
		"Massacre",
		"Military Aviation",
		"evacuation",
		"Naval Vessel",
		"warship",
		"Infantry",
		"Roadbuilding",
		"Coast Guard",
		"conspiracy theory",
		"Manhattan Project",
		"Eastern Front",
		"Animal",
		"surge tank",
		"Water Tank",
		"Anti tank",
		"Anti Aircraft",
		"plane",
		"aeroplane",
		"airplane",
		"soviet union",
		"rationing",
		"Grenade",
		"cannon",
		"Navy Officer",
		"Rocket",
		"prisoner",
		"weapon",
		"Aviator",
		"Parade",
		"Aerial warfare",
		"army vehicle",
		"military vehicle",
		"Storehouse",
		"Aerial View",
		"Ambulance",
		"Destruction",
		"Army Base",
		"Army hospital",
		"Military Base",
		"Border",
		"Army Recruiting",
		"Game",
		"military leader",
		"museum",
		"board meeting",
		"nato",
		"commander",
		"Sergeant",
		"Admiral",
		"Bombing Attack",
		"Battle Monument",
		"clash",
		"strike",
		"damage",
		"leisure",
		"airport",
		"Battle of the Bulge",
		"Barn",
		"Anniversary",
		"Delegate",
		"exile",
		"Military Aviation",
		"evacuation",
		"Coast Guard",
		"Naval Vessel",
		"warship",
		"Infantry",
		"Civilian",
		"Medical aid",
		"bombardment",
		"ambassador",
		"projectile",
		"helmet",
		"Alliance",
		"Treaty of Versailles",
		"enemy territory",
		"reconnaissance",
		"nurse",
		"navy doctor",
		"military hospital",
		"Atomic Bomb",
		"embassy",
		"ship deck",
		"Red cross worker",
		"Infantry camp",
		"swimming camp",
		"fishing camp",
		"construction camp",
		"Trailer camp",
		"Nazi camp",
		"Winter camp",
		"Defence",
		"Recruitment",
		"diplomacy",
		"reservoir",
		"infrastructure",
		"public relation",
		"Association Convention",
		"ship",
		"naval hospital",
		"hospital base",
		"hospital ship",
		"hospital train",
		"migration",
		"captain",
		"summit",
		"sport",
		"Kitchen Truck",
		"Railroad Truck",
		"fire truck",
		"Line Truck",
		"gas truck",
		"Freight Truck",
		"Dump Truck",
		"Diesel truck",
		"Maintenance Truck",
		"Clinic Truck",
		"Truck Accident",
		"military truck",
		"army truck",
		"vice president",
		"Atomic Bombing",
		"Battle of the Marne",
		"anti aircraft gun",
		"anti aircraft warfare",
		"Battle of the Marne",
		"refugee",
		"president",
		"Nuremberg Trials",
		"holocaust",
		"fighter bomber",
		"Ballon gun",
		"Machine gun",
		"Mortar gun",
		"field gun",
		"gun",
		"shovel",
		"Accident",
		"Wreck",
		"Truck",
		"construction",
		"hospital",
		"Tunnel",
		# "#######################################",
		# "war strategy",
		# "vehicular",
		# "Firearm",
		# "exodus",
		# "information warfare",
		# "negotiation",
		# "Blitzkrieg",
		# "Combined arm",
		# "Pearl Harbor attack",
		# "Combat arm",
		# "surrender", # meaningless images
		# "army",
		# "world war",
		# "Bomb",
		# "WWI",
		# "WWII",
	]
	# all_query_tags = natsorted(list(set(all_query_tags)))
	all_query_tags = list(set(all_query_tags))
	print(f"{len(all_query_tags)} Query phrases are being processed, please be patient...")
	for qi, qv in enumerate(all_query_tags):
		print(f"\nQ[{qi+1}/{len(all_query_tags)}]: {qv}")
		query_all_hits = get_data(
			st_date=START_DATE,
			end_date=END_DATE,
			query=qv.lower()
		)
		if query_all_hits:
			qv_processed = re.sub(" ", "_", qv.lower())
			df_fpth = os.path.join(RESULT_DIRECTORY, f"result_df_{START_DATE}_{END_DATE}_query_{qv_processed}.gz")
			try:
				df = load_pickle(fpath=df_fpth)
			except Exception as e:
				df = get_dframe(query=qv.lower(), docs=query_all_hits)
				save_pickle(pkl=df, fname=df_fpth)
			# print(df)
			# print(df.head())
			dfs.append(df)

	print(f"Concatinating {len(dfs)} dfs...")
	# print(dfs[0])
	europeana_df_merged = pd.concat(dfs, ignore_index=True)
	replacement_dict = {
		"plane": "aircraft",
		"airplane": "aircraft",
		"aeroplane": "aircraft",
		"graveyard": "cemetery",
		"soldier": "infantry",
		"clash": "wreck",
		"game": "leisure",
		"military truck": "army truck",
		"military base": "army base",
		"military vehicle": "army base",
		"military hospital": "army hospital",
		"flame thrower": "flamethrower",
	}

	# replacement_dict = {
	# 	"plane": "military aviation",
	# 	"airplane": "military aviation",
	# 	"aeroplane": "military aviation",
	# 	"aircraft": "military aviation",
	# 	"helicopter": "military aviation",
	# 	"air force": "military aviation",
	# 	"naval warship": "navy",
	# 	"submarine": "navy",
	# 	"manufacturing plant": "infrastructure",
	# 	"barn": "infrastructure",
	# 	"construction": "infrastructure",
	# 	"road": "infrastructure",
	# 	"army base": "infrastructure",
	# 	"border": "infrastructure",
	# 	"refugee": "migration",
	# 	"evacuation": "migration",
	# 	"exodus": "migration",
	# 	"exile": "migration",
	# 	"aerial warfare": "strategy",
	# 	"air raid": "strategy",
	# 	"trench warfare": "strategy",
	# 	"battle of the bulge": "strategy",
	# 	"battle of the marne": "strategy",
	# 	"winter war": "strategy",
	# 	"blitzkrieg": "strategy",
	# 	"versailles": "international relations & treaties",
	# 	"treaty of versailles": "international relations & treaties",
	# 	"nuremberg trials": "international relations & treaties",
	# 	"game": "leisure",
	# 	"anniversary": "leisure",
	# 	"rail": "infrastructure",
	# 	"sport": "leisure",
	# 	"meeting": "diplomacy",
	# 	"negotiation": "diplomacy",
	# 	"conference": "diplomacy",
	# 	"summit": "diplomacy",
	# 	"attack": "conflict",
	# 	"clash": "conflict",
	# 	"war": "conflict",
	# 	"military strike": "conflict",
	# 	"battle": "conflict",
	# 	"propaganda": "propaganda & communication",
	# 	"public relation": "propaganda & communication",
	# 	"information warfare": "propaganda & communication",
	# 	"air bomb": "weapon",
	# 	"rifle": "weapon",
	# 	"barrel": "weapon",
	# 	"machine gun": "weapon",
	# 	"artillery": "weapon",
	# 	"tank": "weapon",
	# 	"grenade": "weapon",
	# 	"gun": "weapon",
	# 	"cannon": "weapon",
	# 	"rocket": "weapon",
	# 	"mortar": "weapon",
	# 	"firearm": "weapon",
	# 	"flamethrower": "weapon",
	# 	"bayonet": "weapon",
	# 	"tent": "camp",
	# 	"recruiting": "recruitment",
	# 	"captain": "military leader",
	# 	"army leader": "military leader",
	# 	"commander": "military leader",
	# 	"sergeant": "military leader",
	# 	"admiral": "military leader",
	# 	"explosion": "destruction",
	# 	"accident": "destruction",
	# 	"damage": "destruction",
	# 	"wreck": "destruction",
	# 	"truck":"vehicular",
	# 	"vehicle":"vehicular",
	# 	"ambulance":"vehicular",
	# 	"airport": "infrastructure",
	# 	"dam": "infrastructure",
	# 	"reservoir": "infrastructure",
	# 	"defence": "strategy",
	# }

	europeana_df_merged['query'] = europeana_df_merged['query'].replace(replacement_dict)
	europeana_df_merged = europeana_df_merged.dropna(subset=['img_url']) # drop None firstDigitalObjectUrl
	europeana_df_merged = europeana_df_merged.drop_duplicates(subset=['img_url']) # drop duplicate firstDigitalObjectUrl

	print(f"europeana_df_merged: {europeana_df_merged.shape}")
	print(europeana_df_merged.head(20))

	query_counts = europeana_df_merged['query'].value_counts()
	print(query_counts.tail(25))
	plt.figure(figsize=(20, 13))
	query_counts.plot(kind='bar', fontsize=11)
	plt.title(f'{dataset_name}: Query Frequency (total: {query_counts.shape}) {START_DATE} - {END_DATE}')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(RESULT_DIRECTORY, f"query_x_{query_counts.shape[0]}_freq.png"))

	# Save as CSV
	europeana_df_merged.to_csv(os.path.join(RESULT_DIRECTORY, "europeana.csv"), index=False)

	# Save as Excel
	try:
		europeana_df_merged.to_excel(os.path.join(RESULT_DIRECTORY, "europeana.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	get_images(df=europeana_df_merged)

def test():
	query = "bombing"
	params = {
		'wskey': 'nLbaXYaiH',  # Your API key
		'qf': [
			'collection:photography', 
			'TYPE:"IMAGE"', 
			# 'contentTier:"4"', # high quality images
			'MIME_TYPE:image/jpeg',
		],
		'rows': 100,
		'query': query,
		'reusability': 'open'
	}
	# Send a GET request to the API
	response = requests.get(europeana_api_base_url, params=params)
	# Check if the request was successful
	if response.status_code == 200:
		# Parse the JSON response
		data = response.json()	
		# Check if the 'items' field exists
		if 'items' in data:
			# Extract the 'items' field
			items = data['items']
			tot_results = data['totalResults']
			print(tot_results, len(items))
			# Output the extracted items (metadata and descriptions of the images)
			# for item_idx, item in enumerate(items):
			# 	print(f"item: {item_idx}")
			# 	# print(list(item.keys()))
			# 	print(item.get("id"))
			# 	print(item.get("title"))
			# 	print(item.get("edmIsShownAt"))
			# 	print(item.get("edmIsShownBy"))
			# 	print(item.get("edmTimespanLabel"))
			# 	print(item.get("language"))
			# 	print(item.get("dataProvider"))
			# 	print(item.get("provider"))
			# 	print("#"*100)
				# print(json.dumps(item, indent=2))  # Pretty-print the JSON data for each item
			# You can process or save the 'items' data as needed
		else:
			print("No 'items' found in the response.")
	else:
		print(f"Request failed with status code {response.status_code}")

if __name__ == '__main__':
	print(
		f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
		.center(160, " ")
	)
	START_EXECUTION_TIME = time.time()
	main()
	# test()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)