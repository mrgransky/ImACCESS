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
# $ nohup python -u data_collector.py --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01 >> europeana_image_download.out &

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
START_DATE = args.start_date
END_DATE = args.end_date
dataset_name = "europeana"
nw:int = min(args.num_worker, multiprocessing.cpu_count()) # def: 8
europeana_api_base_url: str = "https://api.europeana.eu/record/v2/search.json"
# europeana_api_key: str = "plaction"
# europeana_api_key: str = "api2demo"
europeana_api_key: str = "nLbaXYaiH"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}
os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")

os.makedirs(os.path.join(DATASET_DIRECTORY, "images"), exist_ok=True)
IMAGE_DIR = os.path.join(DATASET_DIRECTORY, "images")

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
	query_all_hits_fpth = os.path.join(DATASET_DIRECTORY, f"results_{st_date}_{end_date}_query_{query_processed}.gz")
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
					# print(total_hits, len(hits))
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
	data = []
	for doc_idx, doc in enumerate(docs):
		# print(type(doc.get("title")), doc.get("title"))
		title = doc.get("title")#.lower()
		pDate = doc.get("edmTimespanLabel")[0].get("def") if doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def") else None
		image_url = doc.get("edmIsShownBy")[0]
		if (
			image_url 
			and (image_url.endswith('.jpg') or image_url.endswith('.png'))
		):
			pass # Valid entry; no action needed here
		else:
			image_url = None
		row = {
			'id': doc.get("id"),
			'query': query,
			'title': title,
			'description': doc.get("dcDescription"),
			'img_url': image_url,
			'date': pDate,
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed_t: {time.time()-df_st_time:.1f} sec")
	return df

def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
	t0 = time.time()
	rIdx = row.name
	url = row['img_url']
	image_name = re.sub("/", "LBL", row['id']) # str(row['id']) + os.path.splitext(url)[1]
	image_path = os.path.join(image_dir, f"{image_name}.png")
	if os.path.exists(image_path):
		return True # Image already exists, => skipping
	attempt = 0  # Retry mechanism
	while attempt < retries:
		try:
			response = session.get(url, timeout=20)
			response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
			with open(image_path, 'wb') as f: # Save the image to the directory
				f.write(response.content)
			print(f"[{rIdx:<10}/ {total_rows}]{image_name:>20}{time.time() - t0:>10.1f} s")
			return True  # Image downloaded successfully
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx}/{total_rows}] Downloading « {url} » failed: {e}, retrying ({attempt}/{retries})")
			time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
	print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
	return False  # Indicate failed download

def get_synchronized_df_img(df):
	print(f"Synchronizing merged_df(raw) & images of {df.shape[0]} records using {nw} CPUs...")
	successful_rows = []  # List to keep track of successful downloads
	with requests.Session() as session:
		with ThreadPoolExecutor(max_workers=nw) as executor:
			futures = {executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]): idx for idx, row in df.iterrows()}
			for future in as_completed(futures):
				try:
					success = future.result()  # Get the result (True or False) from download_image
					if success:
						successful_rows.append(futures[future])  # Keep track of successfully downloaded rows
				except Exception as e:
					print(f"Unexpected error: {e}")
	print(f"cleaning {type(df)} {df.shape} with {len(successful_rows)} succeded downloaded images [functional URL]...")
	df_cleaned = df.loc[successful_rows] # keep only the successfully downloaded rows
	print(f"Total images downloaded successfully: {len(successful_rows)} out of {df.shape[0]}")
	print(f"df_cleaned: {df_cleaned.shape}")

	img_dir_size = sum(os.path.getsize(f) for f in os.listdir(IMAGE_DIR) if os.path.isfile(f)) #* 1e-9 # GB
	print(f"{IMAGE_DIR} contains {len(os.listdir(IMAGE_DIR))} file(s) with total size: {img_dir_size:.2f} GB")

	return df_cleaned

def main():
	all_query_tags = [
		"motor cycle",
		"ballistic missile",
		"flame thrower",
		"Red cross worker",
		"war bond",
		"Infantry camp",
		"swimming camp",
		"fishing camp",
		"construction camp",
		"Trailer camp",
		"Nazi camp",
		"Winter camp",
		"naval air station",
		"allied invasion",
		"normandy invasion",
		"naval air base",
		"Power Plant",
		"Air bomb",
		"fighter bomber",
		"Pearl Harbor attack",
		"anti tank",
		"anti aircraft",
		"Battle of the Marne",
		'Nazi crime',
		"Nazi victim",
		"Helicopter",
		"trench warfare",
		"Manufacturing Plant",
		"naval aircraft factory",
		"rail construction",
		"dam construction",
		"tunnel construction",
		"allied force",
		"air force base",
		"air force personnel",
		"air force station",
		"air raid",
		"Flag Raising",
		"conspiracy theory",
		"Manhattan Project",
		"Eastern Front",
		"surge tank",
		"Water Tank",
		"soviet union",
		"Naval Officer",
		"army vehicle",
		"Storehouse",
		"Aerial View",
		"Aerial warfare",
		"Army Base",
		"Army hospital",
		"Military Base",
		"military leader",
		"military vehicle",
		"Military Aviation",
		"board meeting",
		"Battle Monument",
		"Battle of the Bulge",
		"Naval Vessel",
		"Medical aid",
		"Coast Guard",
		"Treaty of Versailles",
		"enemy territory",
		"reconnaissance",
		"ship deck",
		"naval hospital",
		"hospital base",
		"hospital ship",
		"hospital train",
		"Army Recruiting",
		"Recruitment",
		"infrastructure",
		"military hospital",
		"Kitchen Truck",
		"Railroad Truck",
		"fire truck",
		"Line Truck",
		"Coast Guard",
		"gas truck",
		"Flying Fortress",
		"Freight Truck",
		"Dump Truck",
		"Diesel truck",
		"Maintenance Truck",
		"Clinic Truck",
		"Truck Accident",
		"military truck",
		"army truck",
		"vice president",
		"bombardment",
		"Bombing Attack",
		"Atomic Bomb",
		"Nuremberg Trials",
		"Ballon gun",
		"Machine gun",
		"Mortar gun",
		"field gun",
		"Memorial day",
		"flamethrower",
		"hunting",
		"Sailboat",
		"regatta",
		"cemetery",
		"graveyard",
		"bayonet",
		"explosion",
		"Submarine",
		"Artillery",
		"Rifle",
		"barrel",
		"Massacre",
		"evacuation",
		"aircraft",
		"soldier",
		"Infantry",
		"Animal",
		"plane", # must be before airplane, aeroplane.
		"aeroplane",
		"airplane",
		"rationing",
		"Grenade",
		"Rocket",
		"prisoner",
		"weapon",
		"Aviator",
		"Parade",
		"commander",
		"museum",
		"Sergeant",
		"Admiral",
		"Ambulance",
		"cannon",
		"ambassador",
		"projectile",
		"helmet",
		"warship",
		"clash",
		"strike",
		"damage",
		"leisure",
		"airport",
		"Barn",
		"Anniversary",
		"Delegate",
		"exile",
		"evacuation",
		"Civilian",
		"nurse",
		"doctor",
		"embassy",
		"Infantry",
		"reservoir",
		"refugee",
		"president",
		"holocaust",
		"migration",
		"Defence",
		"Border",
		"ship",
		"gun",
		"shovel",
		"Accident",
		"Wreck",
		"Truck",
		"hospital",
		"Railroad",
		"captain",
		"sport",
		"Minesweeper",
		"Ceremony",
		"Tunnel",
		"pasture",
		"farm",
	]
	# all_query_tags = natsorted(list(set(all_query_tags)))
	# all_query_tags = list(set(all_query_tags))[:5]
	if USER=="farid": # local laptop
		all_query_tags = all_query_tags#[:5]

	print(f"{len(all_query_tags)} Query phrases are being processed, please be patient...")
	dfs = []
	for qi, qv in enumerate(all_query_tags):
		print(f"\nQ[{qi+1}/{len(all_query_tags)}]: {qv}")
		query_all_hits = get_data(
			st_date=START_DATE,
			end_date=END_DATE,
			query=qv.lower()
		)
		if query_all_hits:
			qv_processed = re.sub(" ", "_", qv.lower())
			df_fpth = os.path.join(DATASET_DIRECTORY, f"result_df_{START_DATE}_{END_DATE}_query_{qv_processed}.gz")
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
	europeana_df_merged_raw = pd.concat(dfs, ignore_index=True)
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
		"military vehicle": "army vehicle",
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
	print(f"pre-processing merged {type(europeana_df_merged_raw)} {europeana_df_merged_raw.shape}")
	europeana_df_merged_raw['query'] = europeana_df_merged_raw['query'].replace(replacement_dict)
	europeana_df_merged_raw = europeana_df_merged_raw.dropna(subset=['img_url']) # drop None firstDigitalObjectUrl
	europeana_df_merged_raw = europeana_df_merged_raw.drop_duplicates(subset=['img_url']) # drop duplicate firstDigitalObjectUrl

	print(f"Processed europeana_df_merged_raw: {europeana_df_merged_raw.shape}")
	print(europeana_df_merged_raw.head(20))

	europeana_df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	try:
		europeana_df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	europeana_df = get_synchronized_df_img(df=europeana_df_merged_raw)
	query_counts = europeana_df['query'].value_counts()
	plt.figure(figsize=(21, 14))
	query_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name} Query Frequency (total: {query_counts.shape})')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(DATASET_DIRECTORY, f"query_x_{query_counts.shape[0]}_freq.png"))

	europeana_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	try:
		europeana_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

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