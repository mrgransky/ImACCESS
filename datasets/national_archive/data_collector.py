import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir) # add project directory to sys.path

from misc.utils import *
from misc.visualize import *

dataset_name = "NATIONAL_ARCHIVE".upper()
parser = argparse.ArgumentParser(description=f"U.S. National Archive Dataset")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1900-01-01", help='Start Date')
parser.add_argument('--end_date', '-edt', type=str, default="1970-12-31", help='End Date')
parser.add_argument('--num_workers', '-nw', type=int, default=10, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
parser.add_argument('--val_split_pct', '-vsp', type=float, default=0.35, help='Validation Split Percentage')
parser.add_argument('--thumbnail_size', type=parse_tuple, default=None, help='Thumbnail size (width, height) in pixels')
parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print(args)
print_args_table(args=args, parser=parser)
set_seeds(seed=args.seed, debug=False)
# run in local laptop:
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 16 --img_mean_std --thumbnail_size 512,512 -v > logs/na_dataset_collection.out &

# run in Pouta:
# $ python data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -nw 12 --img_mean_std --thumbnail_size 512,512
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 -nw 40 -bs 256 --img_mean_std --thumbnail_size 512,512 > /media/volume/ImACCESS/trash/na_dl.out &

na_api_base_url: str = "https://catalog.archives.gov/proxy/records/search"
START_DATE = args.start_date
END_DATE = args.end_date
FIGURE_SIZE = (12, 9)
DPI = 350

meaningless_words_fpth = os.path.join(project_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
# print(STOPWORDS, type(STOPWORDS))
useless_collection_terms = [
	"History of Langley Field",
	"Airplanes - Instruments",
	"Awards",
	"Presentations",
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
	"Maps",
	"Camp McDowell",
	"Landing Fields",
	"Appian Way",
	"Indexes to Aerial Photography",
	"Illustrative Material Published By The Government Printing Office and other Government Agencies",
	"Field Artillery Units and Revolutionary War Artillerymen",
	"Five Civilized Tribes Section 2 (1)",
	"Reasons Why Okmulgee is an Ideal Location for Hospital",
	"Revolutionary War Pension and Bounty-Land-Warrant",
]
os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")
os.makedirs(os.path.join(DATASET_DIRECTORY, "images"), exist_ok=True)
IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")

os.makedirs(os.path.join(DATASET_DIRECTORY, "hits"), exist_ok=True)
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")

os.makedirs(os.path.join(DATASET_DIRECTORY, "outputs"), exist_ok=True)
OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

def get_doc_year(text, raw_doc_date):
	if not pd.isna(raw_doc_date):  # Check if raw_doc_date is missing (None or NaN)
		return raw_doc_date
	if text is None:  # Check if text is None
		return None
	# year_pattern = r'\b\d{4}\b'
	year_pattern = re.compile(r'\b\d{4}\b')
	match = re.search(year_pattern, text) # <re.Match object; span=(54, 58), match='1946'>
	# print(match)
	if match:
		return match.group()
	else:
		return None

def get_data(start_date: str="1914-01-01", end_date: str="1914-01-02", label: str="world war"):
	t0 = time.time()
	label_processed = re.sub(" ", "_", label)
	label_all_hits_fpth = os.path.join(HITs_DIR, f"results_query_{label_processed}_{start_date}_{end_date}.gz")
	try:
		label_all_hits = load_pickle(fpath=label_all_hits_fpth)
	except Exception as e:
		# print(f"{e}")
		print(f"Collecting all docs of National Archive for label: « {label} » ... it might take a while..")
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
			"q": label,
			"startDate": start_date,
			"typeOfMaterials": "Photographs and other Graphic Materials",
			"abbreviated": "true",
			"debug": "true",
			"datesAgg": "TRUE"
		}
		label_all_hits = []
		page = 1
		while True:
			loop_st = time.time()
			params["page"] = page
			try:
				response = requests.get(
					url=na_api_base_url,
					params=params,
					headers=headers,
					# verify=False, # Try disabling SSL verification if that's the issue
					# timeout=30, # Timeout in seconds
				)
				response.raise_for_status()
			except Exception as e:
				print(f"<!> {e}")
				break

			if response.status_code == 200:
				data = response.json()
				hits = data.get('body').get('hits').get('hits')
				# print(len(hits), type(hits))
				# print(hits[0].keys())
				# print(json.dumps(hits[0], indent=2, ensure_ascii=False))
				label_all_hits.extend(hits)
				total_hits = data.get('body').get("hits").get('total').get('value')
				print(f"Page: {page}:\tFound: {len(hits)} {type(hits)}\t{len(label_all_hits)}/{total_hits}\t{time.time()-loop_st:.1f} sec")
				if len(label_all_hits) >= total_hits:
					break
				page += 1
			else:
				print(f"Failed to retrieve data: status_code: {response.status_code}")
				break
		if len(label_all_hits) == 0:
			return
		save_pickle(pkl=label_all_hits, fname=label_all_hits_fpth)
	print(f"Total hit(s): {len(label_all_hits)} {type(label_all_hits)} for label: « {label} » found in {time.time()-t0:.2f} sec")
	return label_all_hits

def is_desired(collections, useless_terms):
	for term in useless_terms:
		for collection in collections:
			if term in collection:
				# print(f"\t> XXXX found '{term}' => skipping! XXXX <")
				return False
	return True

def get_dframe(query: str, docs: List=[Dict]) -> pd.DataFrame:
	qv_processed = re.sub(
		pattern=" ", 
		repl="_", 
		string=query,
	)
	df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_{START_DATE}_{END_DATE}.gz")

	if os.path.exists(df_fpth):
		df = load_pickle(fpath=df_fpth)
		return df	
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc in docs:
		record = doc.get('_source', {}).get('record', {})
		fields = doc.get('fields', {})
		na_identifier = record.get('naId')
		raw_doc_date = record.get('productionDates')[0].get("logicalDate") if record.get('productionDates') else None
		first_digital_object_url = fields.get('firstDigitalObject', [{}])[0].get('objectUrl')
		ancesstor_collections = [f"{itm.get('title')}" for itm in record.get('ancestors')] # record.get('ancestors'): list of dict
		doc_title = record.get('title')
		doc_description = record.get('scopeAndContentNote', None)

		useless_title_terms = [
			"wildflowers" not in doc_title.lower(), 
			"-sc-" not in doc_title.lower(),
			"notes" not in doc_title.lower(),
			"page" not in doc_title.lower(),
			"exhibit" not in doc_title.lower(),
			"ad:" not in doc_title.lower(),
			"sheets" not in doc_title.lower(),
			"report" not in doc_title.lower(),
			"map" not in doc_title.lower(),
			"portrait of" not in doc_title.lower(),
			"poster" not in doc_title.lower(),
			"drawing" not in doc_title.lower(),
			"sketch of" not in doc_title.lower(),
			"layout" not in doc_title.lower(),
			"postcard" not in doc_title.lower(),
			"table:" not in doc_title.lower(),
			"reasons why" not in doc_title.lower(),
			"we can do it!" not in doc_title.lower(),
			"traffic statistics:" not in doc_title.lower(),
			"sketch" not in doc_title.lower(),
			"painting" not in doc_title.lower(),
			"photomechanical print" not in doc_title.lower(),
		] if doc_title is not None else []

		useless_description_terms = [
			"certificate" not in doc_description.lower(),
			"drawing" not in doc_description.lower(),
			"sketch of" not in doc_description.lower(),
			"newspaper" not in doc_description.lower(),
			"sketch" not in doc_description.lower(),
			"report" not in doc_description.lower(),
			"attachment" not in doc_description.lower(),
			"illustrated family record" not in doc_description.lower(),
		] if doc_description is not None else []

		if (
			first_digital_object_url 
			and is_desired(ancesstor_collections, useless_collection_terms) 
			and all(useless_title_terms)
			and all(useless_description_terms)
			and (first_digital_object_url.endswith('.jpg') or first_digital_object_url.endswith('.png'))
		):
			pass # Valid entry; no action needed here
		else:
			first_digital_object_url = None

		doc_title = re.sub(r'\s+', ' ', doc_title).strip() if doc_title else None
		doc_description = re.sub(r'\s+', ' ', doc_description).strip() if doc_description else None

		print(f"doc_title: {doc_title}")
		print(f"doc_description: {doc_description}")
		print()

		row = {
			'id': na_identifier,
			'doc_url': f"https://catalog.archives.gov/id/{na_identifier}",
			'img_url': first_digital_object_url,
			'img_path': f"{os.path.join(IMAGE_DIRECTORY, str(na_identifier) + '.jpg')}",
			'raw_doc_date': raw_doc_date,
			'user_query': query,
			'title': doc_title,
			'description': doc_description,
		}
		data.append(row)
	df = pd.DataFrame(data)

	# extract doc_date from description:
	df['doc_date'] = df.apply(lambda row: get_doc_year(row['description'], row['raw_doc_date']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[df['doc_date'].apply(lambda x: is_valid_date(date=x, start_date=START_DATE, end_date=END_DATE))]

	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")

	if df.shape[0] == 0:
		return

	save_pickle(pkl=df, fname=df_fpth)

	return df

@measure_execution_time
def main():

	with open(os.path.join(project_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		search_labels = list(dict.fromkeys(line.strip() for line in file_))

	print(f"Total of {len(search_labels)} {type(search_labels)} lables are being processed...")

	labels_with_ZERO_result = list()
	dfs = []
	for qi, qv in enumerate(search_labels):
		print(f"\nQ[{qi+1}/{len(search_labels)}]: {qv}")
		qv = clean_(text=qv, sw=STOPWORDS)
		label_all_hits = get_data(
			start_date=START_DATE,
			end_date=END_DATE,
			label=qv,
		)
		if label_all_hits:
			df = get_dframe(
				query=qv,
				docs=label_all_hits,
			)
			if df is not None and df.shape[0]>1:
				dfs.append(df)
		else:
			labels_with_ZERO_result.append(qv)

	print(f">> {len(labels_with_ZERO_result)} labels with no results {START_DATE} - {END_DATE}\n{labels_with_ZERO_result}")

	total_searched_labels = len(dfs)
	print(f">> Concatinating {total_searched_labels} x {type(dfs[0])} dfs ...")

	concat_st = time.time()
	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f">> Concatinated dfs: {df_merged_raw.shape} Elapsed time: {time.time()-concat_st:.1f} sec")

	print(f">> Replacing labels with broad umbrella terms")
	json_file_path = os.path.join(project_dir, 'misc', 'super_labels.json')
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"{json_file_path} not found! Using empty replacement dictionary")
		replacement_dict = {}
	unq_labels = set(replacement_dict.values())
	print(f">> {len(unq_labels)} Unique Label(s):\n{unq_labels}")

	print(f">> Pre-processing merged {type(df_merged_raw)} {df_merged_raw.shape}")
	print(f">> Merging user_query to label with umbrella terms from {json_file_path}")
	df_merged_raw['label'] = df_merged_raw['user_query'].replace(replacement_dict)
	print(f">> Found {df_merged_raw['img_url'].isna().sum()} None img_url / {df_merged_raw.shape[0]} total samples")
	df_merged_raw = df_merged_raw.dropna(subset=['img_url'])
	print(f">> After dropping None img_url: {df_merged_raw.shape}")

	############################## Dropping duplicated img_url ##############################
	# df_merged_raw = df_merged_raw.drop_duplicates(subset=['img_url'], keep="first", ignore_index=True) # drop duplicate img_url
	# print(f"Processed df_merged_raw: {df_merged_raw.shape}")

	# df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	# try:
	# 	df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	# except Exception as e:
	# 	print(f"Failed to write Excel file: {e}")
	# print(f"Elapsed_t: {time.time()-concat_st:.1f} sec".center(100, "-"))

	# na_df = get_synchronized_df_img(
	# 	df=df_merged_raw,
	# 	image_dir=IMAGE_DIRECTORY,
	# 	nw=args.num_workers,
	# )
	############################## Dropping duplicated img_url ##############################

	############################## aggregating user_query to list ##############################
	print(f"\n1. Creating MULTI-LABEL version (aggregating user_query) from raw data: {df_merged_raw.shape}...")
	grouped = df_merged_raw.groupby('img_url').agg(
		{
			'id': 'first',
			'doc_url': 'first',
			'img_path': 'first',
			'user_query': lambda x: list(set(x)),  # Combine user_query into a list with unique elements
			'raw_doc_date': 'first',
			'doc_date': 'first',
			'title': 'first',
			'description': 'first',
		}
	).reset_index()
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))

	# Map user_query to labels using replacement_dict
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))
	print(f"Multi-label dataset shape: {grouped.shape}")
	############################## aggregating user_query to list ##############################

	print("\n2. Processing images (using multi-label dataset as reference)...")
	synched_fpath = os.path.join(DATASET_DIRECTORY, f"synched_x_{total_searched_labels}_searched_labels_metadata_multi_label.csv")
	multi_label_synched_df = get_synchronized_df_img(
		df=grouped,
		synched_fpath=synched_fpath,
		nw=args.num_workers,
		thumbnail_size=args.thumbnail_size,
		verbose=args.verbose,
	)
	multi_label_final_df = get_enriched_description(df=multi_label_synched_df)

	print(f"Saving full multi-label {type(multi_label_final_df)} {multi_label_final_df.shape} {list(multi_label_final_df.columns)}")
	multi_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.csv"), index=False)

	try:
		multi_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write final multi-label Excel file: {e}")

	print("\n3. Creating SINGLE-LABEL version (from successfully downloaded images)...")
	single_label_final_df = multi_label_synched_df.copy()
	single_label_columns_to_keep = [
		col 
		for col in single_label_final_df.columns 
		if col not in ['multi_labels', 'user_query']
	]
	single_label_final_df = single_label_final_df[single_label_columns_to_keep].copy()
	print(f"Single-label dataset: {type(single_label_final_df)} {single_label_final_df.shape} {list(single_label_final_df.columns)}")
	
	print("\nSaving final single-label dataset...")
	single_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_single_label.csv"), index=False)
	try:
		single_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_single_label.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write final single-label Excel file: {e}")
	
	post_process(
		df=single_label_final_df, 
		dataset_type="single_label", 
		is_multi_label=False,
		output_dir=OUTPUT_DIRECTORY,
	)
	
	post_process(
		df=multi_label_final_df, 
		dataset_type="multi_label", 
		is_multi_label=True,
		output_dir=OUTPUT_DIRECTORY,
	)

	if args.img_mean_std and os.listdir(IMAGE_DIRECTORY):
		try:
			img_rgb_mean = load_pickle(fpath=img_rgb_mean_fpth) 
			img_rgb_std = load_pickle(fpath=img_rgb_std_fpth)
		except Exception as e:
			print(f"{e}")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
				source=IMAGE_DIRECTORY, 
				num_workers=args.num_workers,
				batch_size=args.batch_size,
				img_rgb_mean_fpth=img_rgb_mean_fpth,
				img_rgb_std_fpth=img_rgb_std_fpth,
				verbose=args.verbose,
			)

	print("\nDATASET CREATION SUMMARY\n")
	print("-"*100)
	print(f"Single-label dataset: {single_label_final_df.shape}: {single_label_final_df['label'].nunique()} unique labels")
	print(list(single_label_final_df.columns))
	print(single_label_final_df['label'].value_counts().sort_values(ascending=False))
	print("-"*100)

	print(f"Multi-label dataset: {multi_label_final_df.shape}")
	print(list(multi_label_final_df.columns))
	print("-"*100)

	print(f"Unique images downloaded: {len(os.listdir(IMAGE_DIRECTORY))} files")
	print(f"Files created in {DATASET_DIRECTORY}:")
	for file in sorted(os.listdir(DATASET_DIRECTORY)):
		if file.endswith(('.csv', '.xlsx')):
			print(f"\t- {file}")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
