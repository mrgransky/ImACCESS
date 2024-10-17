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
HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER

na_api_base_url: str = "https://catalog.archives.gov/proxy/records/search"
START_DATE = args.start_date
END_DATE = args.end_date
nw:int = min(args.num_worker, multiprocessing.cpu_count()) # def: 8
dataset_name = "NATIONAL_ARCHIVE"
useless_collection_terms = [
	"Cartoon", 
	"Newsmap",
	"Posters", 
	"Tools and Machinery",
	"Roads of the Past",
	"Government Reports",
	"Art by",
	"Selected Passport Applications",
	"Flynn, Errol",
	"Herbert Hoover Papers",
	"Roads and Trails",
	"Approved Pension",
	"Maps of",
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

def get_data(st_date: str="1914-01-01", end_date: str="1914-01-02", query: str="world war"):
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
			"abbreviated": "true",
			"debug": "true",
			"datesAgg": "TRUE"
		}
		query_all_hits = []
		page = 1
		while True:
			loop_st = time.time()
			params["page"] = page
			response = requests.get(
				na_api_base_url,
				params=params,
				headers=headers,
			)
			if response.status_code == 200:
				data = response.json()
				hits = data.get('body').get('hits').get('hits')
				# print(len(hits), type(hits))
				# print(hits[0].keys())
				# print(json.dumps(hits[0], indent=2, ensure_ascii=False))
				query_all_hits.extend(hits)
				total_hits = data.get('body').get("hits").get('total').get('value')
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
	return True

def get_dframe(query: str="query", docs: List=[Dict]):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc in docs:
		record = doc.get('_source', {}).get('record', {})
		fields = doc.get('fields', {})
		title = record.get('title').lower() if record.get('title') != "Untitled" else None
		na_identifier = record.get('naId')
		pDate = record.get('productionDates')[0].get("logicalDate") if record.get('productionDates') else None
		first_digital_object_url = fields.get('firstDigitalObject', [{}])[0].get('objectUrl')
		ancesstor_collections = [f"{itm.get('title')}" for itm in record.get('ancestors')] # record.get('ancestors'): list of dict
		
		# Only filter URLs based on their extensions, but no need to check their status
		if first_digital_object_url and is_desired(ancesstor_collections, useless_collection_terms) and ("map of" not in title or "drawing of" not in title or "postcard" not in title) and (first_digital_object_url.endswith('.jpg') or first_digital_object_url.endswith('.png')):
			# We skip the check_url_status here for better performance
			pass
		else:
			first_digital_object_url = None

		row = {
			'id': na_identifier,
			'query': query,
			'title': title,
			'description': record.get('scopeandContentNote'),
			'img_url': first_digital_object_url,
			'date': pDate,
		}
		data.append(row)

	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")
	return df

def download_image(row, session, image_dir, total_rows, retries=5, backoff_factor=0.5):
	t0 = time.time()
	rIdx = row.name
	url = row['img_url']
	image_name = str(row['id']) + os.path.splitext(url)[1]
	image_path = os.path.join(image_dir, image_name)
	
	if os.path.exists(image_path):
		# Image already exists, consider it a successful download
		return True

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
			print(f"[{rIdx}/{total_rows}] Downloading {image_name} failed! {e}, retrying ({attempt}/{retries})...")
			time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
	print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
	return False  # Indicate failed download

def get_synchronized_df_img(df):
	print(f"Synchronizing merged_df(raw) & images of {df.shape[0]} records using {nw} CPUs...")
	os.makedirs(os.path.join(RESULT_DIRECTORY, "images"), exist_ok=True)
	IMAGE_DIR = os.path.join(RESULT_DIRECTORY, "images")
	
	successful_rows = []  # List to keep track of successful downloads

	with requests.Session() as session:
		with ThreadPoolExecutor(max_workers=nw) as executor:
			# Submit download tasks
			futures = {executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]): idx for idx, row in df.iterrows()}
			for future in as_completed(futures):
				try:
					success = future.result()  # Get the result (True or False) from download_image
					if success:
						successful_rows.append(futures[future])  # Keep track of successfully downloaded rows
				except Exception as e:
					print(f"Unexpected error: {e}")

	# Filter the DataFrame to keep only the successfully downloaded rows
	print(f"cleaning {type(df)} {df.shape} with {len(successful_rows)} succeded downloaded images [functional URL]...")
	df_cleaned = df.loc[successful_rows]
	print(f"Total images downloaded successfully: {len(successful_rows)} out of {df.shape[0]}")
	# Return the cleaned DataFrame
	print(f"df_cleaned: {df_cleaned.shape}")
	return df_cleaned

def main():	
	dfs = []
	all_query_tags = [
		"Sailboat",
		"regatta",
		"ballistic missile",
		"flame thrower",
		"flamethrower",
		"naval air station",
		"naval air base",
		"Power Plant",
		'Nazi crime',
		"Nazi victim",
		"Helicopter",
		"terminal",
		"trench warfare",
		"explosion",
		"soldier",
		"Submarine",
		"Manufacturing Plant",
		"naval aircraft factory",
		"road construction",
		"rail construction",
		"dam construction",
		"tunnel construction",
		"allied force",
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
		"air raid",
		"Flag Raising",
		"Massacre",
		"evacuation",
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
		"plane",
		"aeroplane",
		"airplane",
		"soviet union",
		"rationing",
		"Grenade",
		"cannon",
		"Naval Officer",
		"Rocket",
		"prisoner",
		"weapon",
		"Aviator",
		"Parade",
		"army vehicle",
		"Storehouse",
		"Aerial View",
		"Aerial warfare",
		"Ambulance",
		"Army Base",
		"Army hospital",
		"Military Base",
		"military leader",
		"military vehicle",
		"Military Aviation",
		"museum",
		"board meeting",
		"commander",
		"Sergeant",
		"Admiral",
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
		"evacuation",
		"Coast Guard",
		"Naval Vessel",
		"warship",
		"Infantry",
		"Civilian",
		"Medical aid",
		"ambassador",
		"projectile",
		"helmet",
		"Treaty of Versailles",
		"enemy territory",
		"reconnaissance",
		"nurse",
		"doctor",
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
		"Game",
		"Defence",
		"Border",
		"Army Recruiting",
		"Recruitment",
		"reservoir",
		"infrastructure",
		"public relation",
		"ship",
		"military hospital",
		"naval hospital",
		"hospital base",
		"hospital ship",
		"hospital train",
		"migration",
		"captain",
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
		"bombardment",
		"Bombing Attack",
		"Atomic Bomb",
		"Air bomb",
		"fighter bomber",
		"Pearl Harbor attack",
		"anti tank",
		"anti aircraft",
		"Battle of the Marne",
		"refugee",
		"president",
		"Nuremberg Trials",
		"holocaust",
		"Ballon gun",
		"Machine gun",
		"Mortar gun",
		"field gun",
		"gun",
		"shovel",
		"Accident",
		"Wreck",
		"Truck",
		"hospital",
		"Tunnel",
		"pasture",
		"farm",
		"allied invasion",
		"normandy invasion",
		"Railroad",
		"Flying Fortress",
		"Minesweeper",
		"Ceremony",
		"Memorial day",
		# "Constitution",
	]
	# all_query_tags = natsorted(list(set(all_query_tags)))
	# all_query_tags = list(set(all_query_tags))[:5]
	if USER=="farid": # local laptop
		all_query_tags = all_query_tags[:20]
	print(f"{len(all_query_tags)} Query phrases are being processed, please be paitient...")
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
			print(df.head(10))
			dfs.append(df)

	print(f"Concatinating {len(dfs)} dfs...")
	# print(dfs[0])
	na_df_merged_raw = pd.concat(dfs, ignore_index=True)
	replacement_dict = {
		"regatta": "sailboat",
		"normandy invasion": "allied invasion",
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
		"recruitment": "army recruiting",
		"farm": "pasture",
		"minesweeper": "naval vessel",
	}

	# replacement_dict = {
	# "boeing": "aircraft",
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

	print(f"pre-processing merged {type(na_df_merged_raw)} {na_df_merged_raw.shape}")
	na_df_merged_raw['query'] = na_df_merged_raw['query'].replace(replacement_dict)
	na_df_merged_raw = na_df_merged_raw.dropna(subset=['img_url']) # drop None img_url
	na_df_merged_raw = na_df_merged_raw.drop_duplicates(subset=['img_url'], keep="first", ignore_index=True) # drop duplicate img_url

	print(f"Processed na_df_merged_raw: {na_df_merged_raw.shape}")
	print(na_df_merged_raw.head(20))

	na_df_merged_raw.to_csv(os.path.join(RESULT_DIRECTORY, "na_raw.csv"), index=False)
	try:
		na_df_merged_raw.to_excel(os.path.join(RESULT_DIRECTORY, "na_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	na_df = get_synchronized_df_img(df=na_df_merged_raw)

	query_counts = na_df['query'].value_counts()
	# print(query_counts.tail(25))

	plt.figure(figsize=(20, 13))
	query_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name}: Query Frequency (total: {query_counts.shape}) {START_DATE} - {END_DATE}')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(RESULT_DIRECTORY, f"query_x_{query_counts.shape[0]}_freq.png"))

	na_df.to_csv(os.path.join(RESULT_DIRECTORY, "na.csv"), index=False)
	try:
		na_df.to_excel(os.path.join(RESULT_DIRECTORY, "na.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

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