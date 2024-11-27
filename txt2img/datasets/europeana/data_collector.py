import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from misc.utils import *

parser = argparse.ArgumentParser(description="Generate Images to Query Prompts")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', type=str, default="1890-01-01", help='Dataset DIR')
parser.add_argument('--end_date', type=str, default="1960-01-01", help='Dataset DIR')
parser.add_argument('--num_workers', type=int, default=10, help='Number of CPUs')
parser.add_argument('--img_mean_std', type=bool, default=False, help='Image mean & std')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

# sys.exit()

# run in local laptop:
# $ python data_collector.py --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01
# $ nohup python -u data_collector.py --dataset_dir $PWD --start_date 1890-01-01 --end_date 1960-01-01 > logs/europeana_image_download.out &

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
START_DATE = args.start_date
END_DATE = args.end_date

meaningless_words_fpth = os.path.join(parent_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
print(STOPWORDS, type(STOPWORDS))
dataset_name: str = "europeana".upper()
europeana_api_base_url: str = "https://api.europeana.eu/record/v2/search.json"
# europeana_api_key: str = "plaction"
europeana_api_key: str = "api2demo"
# europeana_api_key: str = "nLbaXYaiH"
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

os.makedirs(os.path.join(DATASET_DIRECTORY, "hits"), exist_ok=True)
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")

os.makedirs(os.path.join(DATASET_DIRECTORY, "outputs"), exist_ok=True)
OUTPUTs_DIR = os.path.join(DATASET_DIRECTORY, "outputs")

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

def get_data(st_date: str="1914-01-01", end_date: str="1914-01-02", label: str="world war"):
	t0 = time.time()
	label_processed = re.sub(" ", "_", label)
	label_all_hits_fpth = os.path.join(HITs_DIR, f"results_query_{label_processed}_{st_date}_{end_date}.gz")
	try:
		label_all_hits = load_pickle(fpath=label_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all docs of National Archive for label: « {label} » ... it might take a while..")
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
			'query': label,
			'reusability': 'open'
		}
		label_all_hits = []
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
					label_all_hits.extend(hits)
					# print(json.dumps(label_all_hits, indent=2, ensure_ascii=False))
					print(f"start: {start}:\tFound: {len(hits)} {type(hits)}\t{len(label_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
				if len(label_all_hits) >= total_hits:
					break
				start += params.get("rows")
			else:
				print(f"Failed to retrieve data: status_code: {response.status_code}")
				break
		if len(label_all_hits) == 0:
			return
		save_pickle(pkl=label_all_hits, fname=label_all_hits_fpth)
	print(f"Total hit(s): {len(label_all_hits)} {type(label_all_hits)} for query: « {label} » found in {time.time()-t0:.2f} sec")
	return label_all_hits

def get_dframe(label: str="query", docs: List=[Dict]):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for label: « {label} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc_idx, doc in enumerate(docs):
		# print(type(doc.get("title")), doc.get("title"))
		# title = doc.get("title")
		europeana_id = doc.get("id")
		# print(type(doc.get("title")), len(doc.get("title")))
		doc_title_list = doc.get("title") # ["title1", "title2", "title3", ...]
		doc_description_list = doc.get("dcDescription" )# ["desc1", "desc2", "desc3", ...]
		doc_title = clean_(text=' '.join(doc_title_list), sw=STOPWORDS) if doc_title_list else None
		doc_description = clean_(text=" ".join(doc_description_list), sw=STOPWORDS) if doc_description_list else None
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
			'id': europeana_id,
			'label': label,
			'title': doc_title,
			'description': doc_description,
			'img_url': image_url,
			'label_title_description': label + " " + (doc_title or '') + " " + (doc_description or ''),
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
			print(f"[{rIdx:<10}/ {total_rows}]{image_name:<50}{time.time()-t0:>50.1f} s")
			return True  # Image downloaded successfully
		except (RequestException, IOError) as e:
			attempt += 1
			print(f"[{rIdx}/{total_rows}] Failed Downloading {url} : {e}, retrying ({attempt}/{retries})")
			time.sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
	print(f"[{rIdx}/{total_rows}] Failed to download {image_name} after {retries} attempts.")
	return False  # Indicate failed download

def get_synchronized_df_img(df, nw: int=8):
	print(f"Synchronizing merged_df(raw) & images of {df.shape[0]} records using {nw} CPUs...")
	successful_rows = []  # List to keep track of successful downloads
	with requests.Session() as session:
		with ThreadPoolExecutor(max_workers=nw) as executor:
			futures = {executor.submit(download_image, row, session, IMAGE_DIR, df.shape[0]): idx for idx, row in df.iterrows()}
			for future in as_completed(futures):
				try:
					success = future.result() # Get result (True or False) from download_image
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
	with open(os.path.join(parent_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		all_label_tags = [line.strip().lower() for line in file_]
	print(type(all_label_tags), len(all_label_tags))

	# all_label_tags = natsorted(list(set(all_label_tags)))
	# all_label_tags = list(set(all_label_tags))[:5]
	# if USER=="farid": # local laptop
	# 	all_label_tags = all_label_tags#[:5]

	print(f"{len(all_label_tags)} lables are being processed for user: {USER}, please be paitient...")
	dfs = []
	for qi, qv in enumerate(all_label_tags):
		print(f"\nQ[{qi+1}/{len(all_label_tags)}]: {qv}")
		qv = clean_(text=qv, sw=STOPWORDS)
		label_all_hits = get_data(
			st_date=START_DATE,
			end_date=END_DATE,
			label=qv,
		)
		if label_all_hits:
			qv_processed = re.sub(
				pattern=" ", 
				repl="_", 
				string=qv,
			)
			df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_{START_DATE}_{END_DATE}.gz")
			try:
				df = load_pickle(fpath=df_fpth)
			except Exception as e:
				df = get_dframe(
					label=qv,
					docs=label_all_hits,
				)
				save_pickle(pkl=df, fname=df_fpth)
			# print(df)
			# print(df.head())
			dfs.append(df)

	print(f"Concatinating {len(dfs)} dfs...")
	europeana_df_merged_raw = pd.concat(dfs, ignore_index=True)
	json_file_path = os.path.join(parent_dir, 'misc', 'generalized_labels.json')
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")

	print(f"pre-processing merged {type(europeana_df_merged_raw)} {europeana_df_merged_raw.shape}")
	europeana_df_merged_raw['label'] = europeana_df_merged_raw['label'].replace(replacement_dict)
	europeana_df_merged_raw = europeana_df_merged_raw.dropna(subset=['img_url']) # drop None firstDigitalObjectUrl
	europeana_df_merged_raw = europeana_df_merged_raw.drop_duplicates(subset=['img_url']) # drop duplicate firstDigitalObjectUrl

	print(f"Processed europeana_df_merged_raw: {europeana_df_merged_raw.shape}")
	print(europeana_df_merged_raw.head(20))

	europeana_df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	try:
		europeana_df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	europeana_df = get_synchronized_df_img(df=europeana_df_merged_raw, nw=args.num_workers)
	label_counts = europeana_df['label'].value_counts()
	plt.figure(figsize=(21, 14))
	label_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name} Label Frequency (total: {label_counts.shape})')
	plt.xlabel('label')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUTs_DIR, f"all_query_labels_x_{label_counts.shape[0]}_freq.png"))

	europeana_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	try:
		europeana_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if args.img_mean_std:
		try:
			img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
		except Exception as e:
			print(f"{e}")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
				dir=os.path.join(DATASET_DIRECTORY, "images"), 
				num_workers=args.num_workers,
			)
			save_pickle(pkl=img_rgb_mean, fname=img_rgb_mean_fpth)
			save_pickle(pkl=img_rgb_std, fname=img_rgb_std_fpth)
		print(f"IMAGE Mean: {img_rgb_mean} Std: {img_rgb_std}")

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