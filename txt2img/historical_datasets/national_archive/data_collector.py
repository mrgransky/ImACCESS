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
from requests.exceptions import RequestException
import argparse

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', type=str, default="1933-01-01", help='Dataset DIR')
parser.add_argument('--end_date', type=str, default="1933-01-02", help='Dataset DIR')
parser.add_argument('--num_worker', type=int, default=8, help='Number of CPUs')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)
# run in local laptop:
# $ python data_collector.py --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01
# $ nohup python data_collector.py -u --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01 >> na_image_download.out &

START_DATE = args.start_date
END_DATE = args.end_date
nw:int = min(args.num_worker, multiprocessing.cpu_count()) # def: 8
dataset_name = "NATIONAL_ARCHIVE"
useless_collection_terms = [
	"Cartoon Collection", 
	"Posters", 
	"Tools and Machinery",
	"Public Roads of the Past",	
]
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

def get_data(url: str="url.com", st_date: str="1914-01-01", end_date: str="1914-01-02", query: str="world war"):
	t0 = time.time()
	query_processed = re.sub(" ", "_", query.lower())
	query_all_hits_fpth = os.path.join(RESULT_DIRECTORY, f"results_{st_date}_{end_date}_query_{query_processed}.gz")
	try:
		query_all_hits = load_pickle(fpath=query_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all docs of National Archive for Query: « {query} » ... it might take a while..")
		headers = {
			'Content-type': 'application/json',
			'Accept': 'application/json; text/plain; */*',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
			'Pragma': 'no-cache',
		}
		params = {
			"limit": 100,
			"availableOnline": "true",
			"dataSource": "description",
			"endDate": end_date,
			"levelOfDescription": "item",
			"objectType": "jpg,png",
			"q": query,
			"startDate": st_date,
			"typeOfMaterials": "Photographs and other Graphic Materials",
			# "abbreviated": "true",
			# "debug": "true",
			# "datesAgg": "TRUE"
		}
		query_all_hits = []
		page = 1
		while True:
			loop_st = time.time()
			params["page"] = page
			response = requests.get(
				url,
				params=params,
				headers=headers,
			)
			if response.status_code == 200:
				data = response.json()
				hits = data.get('body').get('hits').get('hits')
				query_all_hits.extend(hits)
				total_hits = data.get('body').get("hits").get('total').get('value')
				# print(json.dumps(query_all_hits, indent=2, ensure_ascii=False))
				print(f"Page: {page}:\tFound: {len(hits)} {type(hits)}\t{len(query_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
				if len(query_all_hits) >= total_hits:
					break
				page += 1
			else:
				print("Failed to retrieve data")
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

def is_desired(collections, useless_terms):
	for term in useless_terms:
		for collection in collections:
			if term in collection:
				print(f"\t> XXXX found '{term}', => skipping! XXXX <")
				return False
	# print(f"clean collections: {collections}")
	return True

def get_dframe(query: str="query", docs: List=[Dict]):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc in docs:
		record = doc.get('_source', {}).get('record', {})
		fields = doc.get('fields', {})
		title = record.get('title') if record.get('title') != "Untitled" else None
		# print(title, "Map of" in title, "Drawing of" in title)
		pDate = record.get('productionDates')[0].get("logicalDate") if record.get('productionDates') else None
		first_digital_object_url = fields.get('firstDigitalObject', [{}])[0].get('objectUrl')
		ancesstor_collections = [f"{itm.get('title')}" for itm in record.get('ancestors')] # record.get('ancestors'): list of dict
		# print(ancesstor_collections)
		if first_digital_object_url and is_desired(ancesstor_collections, useless_collection_terms) and ("Map of" not in title or "Drawing of" not in title) and (first_digital_object_url.endswith('.jpg') or first_digital_object_url.endswith('.png')):
			#################################################################
			# # without checking status_code [faster but broken URL]
			# first_digital_object_url = first_digital_object_url
			#################################################################
			#################################################################
			# # with checking status_code [slower but healty URL]
			# print(f"{first_digital_object_url:<140}", end=" ")
			# st_t = time.time()
			if check_url_status(first_digital_object_url): # status code must be 200!
				first_digital_object_url = first_digital_object_url
			else:
				first_digital_object_url = None
			# print(f"{time.time()-st_t:.2f} s")
			#################################################################
		else:
			first_digital_object_url = None
		row = {
			'id': record.get('naId'),
			'query': query,
			'title': title,
			'description': record.get('scopeandContentNote'),
			'img_url': first_digital_object_url,
			'date': pDate,
			# 'totalDigitalObjects': fields.get('totalDigitalObjects', [0])[0],
			# 'firstDigitalObjectType': fields.get('firstDigitalObject', [{}])[0].get('objectType'),
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed_t: {time.time()-df_st_time:.1f} sec")
	return df

def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
	t0 = time.time()
	rIdx = row.name
	url = row['img_url']
	image_name = str(row['id']) + os.path.splitext(url)[1]
	image_path = os.path.join(image_dir, image_name)
	# Check if the image already exists to avoid redownloading
	if os.path.exists(image_path):
		# print(f"File {image_name} already exists, skipping download.")
		return image_name
	# Retry mechanism
	attempt = 0
	while attempt < retries:
		try:
			# Attempt to download the image
			response = session.get(url, timeout=20)
			response.raise_for_status()  # Raise an error for bad responses (e.g., 404 or 500)
			# Save the image to the directory
			with open(image_path, 'wb') as f:
				f.write(response.content)
			print(f"[{rIdx}/{total_rows}] Saved {image_name}\t\t\tin:\t{time.time()-t0:.1f} sec")
			return image_name
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx}/{total_rows}] Downloading {image_name} Failed! {e}, Retrying ({attempt}/{retries})...")
			time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
	print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
	return None

def get_images(df):
	print(f"Saving images of {df.shape[0]} records using {nw} CPUs...")
	# Create the directory if it doesn't exist
	os.makedirs(os.path.join(RESULT_DIRECTORY, "images"), exist_ok=True)
	IMAGE_DIR = os.path.join(RESULT_DIRECTORY, "images")
	# Start a session for connection reuse
	with requests.Session() as session:
		# Use ThreadPoolExecutor for parallel downloads
		with ThreadPoolExecutor(max_workers=nw) as executor:
			futures = [executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]) for _, row in df.iterrows()]
			# Process results as they complete
			for future in as_completed(futures):
				try:
					future.result()
				except Exception as e:
					print(f"An unexpected error occurred: {e}")
	print(f"Total number of images downloaded: {len(os.listdir(IMAGE_DIR))}")

def main():
	national_archive_us_URL: str = "https://catalog.archives.gov/proxy/records/search"
	dfs = []
	all_query_tags = [
		"Ballistic missile",
		"flame thrower",
		"flamethrower",
		"refugee",
		"shovel",
		"Wreck",
		"Power Plant",
		"Winter camp",
		"road construction",
		"rail construction",
		"dam construction",
		"Helicopter",
		"Manufacturing Plant",
		"naval aircraft factory",
		"naval air station",
		"naval air base",
		"terminal",
		"holocaust",
		"trench warfare",
		"explosion",
		"soldier",
		"Submarine",
		"allied force",
		"Nuremberg Trials",
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
		"Machine Gun",
		"Mortar Gun",
		"air raid",
		"flag",
		"Massacre",
		"Military Aviation",
		"evacuation",
		"Naval Vessel",
		"warship",
		"Infantry",
		"Tunnel",
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
		"Tunnel",
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
		"tunnel construction",
		"Defence",
		"Accident",
		"Ballon Gun",
		"Construction",
		"Recruitment",
		"gun",
		"diplomacy",
		"reservoir",
		"infrastructure",
		"war strategy",
		"public relation",
		"Association Convention",
		"ship",
		"naval hospital",
		"hospital base",
		"hospital ship",
		"hospital train",
		"hospital",
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
		"Truck",
		"Truck Accident",
		"military truck",
		"army truck",
		"vice president",
		"president",
		"Atomic Bombing",
		"Battle of the Marne",
		"Anti Aircraft Gun",
		"Anti aircraft warfare",
		"Battle of the Marne",
		# "#######################################",
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
	all_query_tags = natsorted(list(set(all_query_tags)))
	print(f"{len(all_query_tags)} Query phrases are being processed, please be paitient...")
	for qi, qv in enumerate(all_query_tags):
		print(f"\nQ[{qi}]: {qv}")
		query_all_hits = get_data(
			url=national_archive_us_URL,
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
	na_df_merged = pd.concat(dfs, ignore_index=True)
	replacement_dict = {
		"plane": "aircraft",
		"airplane": "aircraft",
		"aeroplane": "aircraft",
		"graveyard": "cemetery",
		"soldier": "infantry",
		"clash": "wreck",
		"game": "leisure",
		"sport": "leisure",
		"military truck": "army truck",
		"military base": "army base",
		"military vehicle": "army base",
		"military hospital": "army hospital",
		"flame thrower": "flamethrower",
		"roadbuilding": "road construction",
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

	print(f"pre-processing merged {type(na_df_merged)} {na_df_merged.shape}")
	na_df_merged['query'] = na_df_merged['query'].replace(replacement_dict)
	na_df_merged = na_df_merged.dropna(subset=['img_url']) # drop None img_url
	na_df_merged = na_df_merged.drop_duplicates(subset=['img_url']) # drop duplicate img_url

	print(f"Processed na_df_merged: {na_df_merged.shape}")
	print(na_df_merged.head(20))

	query_counts = na_df_merged['query'].value_counts()
	print(query_counts.tail(25))
	plt.figure(figsize=(20, 13))
	query_counts.plot(kind='bar', fontsize=11)
	plt.title(f'{dataset_name}: Query Frequency (total: {query_counts.shape}) {START_DATE} - {END_DATE}')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(RESULT_DIRECTORY, f"query_x_{query_counts.shape[0]}_freq.png"))

	# total_obj_counts = na_df_merged['totalDigitalObjects'].value_counts()
	# print(total_obj_counts)
	# Save as CSV
	na_df_merged.to_csv(os.path.join(RESULT_DIRECTORY, "na.csv"), index=False)

	# Save as Excel
	try:
		na_df_merged.to_excel(os.path.join(RESULT_DIRECTORY, "na.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	get_images(df=na_df_merged)

if __name__ == '__main__':
	print(
		f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
		.center(160, " ")
	)
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)