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
parser.add_argument('--enable_thumbnailing', action='store_true', help='Enable image thumbnailing')
parser.add_argument('--thumbnail_size', type=parse_tuple, default=(1000, 1000), help='Thumbnail size (width, height) in pixels')
parser.add_argument('--large_image_threshold_mb', type=float, default=1.0, help='Large image threshold in MB')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print_args_table(args=args, parser=parser)

# run in local laptop:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs --start_date 1933-01-01 --end_date 1933-01-10

########################## --start_date 1933-01-01 --end_date 1933-01-02 ##########################
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs --start_date 1935-01-01 --end_date 1935-01-02 --num_workers 8 --img_mean_std --enable_thumbnailing > logs/na_image_download.out &

# saving into external hard drive:
# $ nohup python -u data_collector.py -ddir /media/farid/password_WD/ImACCESS/WW_DATASETs --start_date 1935-01-01 --end_date 1935-01-02 --num_workers 8 --enable_thumbnailing --img_mean_std > logs/na_image_download.out &

########################## --start_date 1914-01-01 --end_date 1946-12-31 ##########################
# $ nohup python -u data_collector.py -ddir /media/farid/password_WD/ImACCESS/WW_DATASETs --start_date 1914-01-01 --end_date 1946-12-31 --num_workers 2 > logs/na_image_download.out &

##################################################################################################################
# run in Pouta:

# $ python data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -nw 12 --img_mean_std --enable_thumbnailing
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 -nw 12 --img_mean_std --enable_thumbnailing > /media/volume/ImACCESS/trash/na_dl.out &

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
				label_all_hits.extend(hits)
				total_hits = data.get('body').get("hits").get('total').get('value')
				print(f"Page: {page}:\tFound: {len(hits)} {type(hits)}\t{len(label_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
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

def get_dframe(label: str="label", docs: List=[Dict]) -> pd.DataFrame:
	qv_processed = re.sub(
		pattern=" ", 
		repl="_", 
		string=label,
	)
	df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_{START_DATE}_{END_DATE}.gz")
	if os.path.exists(df_fpth):
		df = load_pickle(fpath=df_fpth)
		return df	
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for label: « {label} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc in docs:
		record = doc.get('_source', {}).get('record', {})
		fields = doc.get('fields', {})
		na_identifier = record.get('naId')
		raw_doc_date = record.get('productionDates')[0].get("logicalDate") if record.get('productionDates') else None
		first_digital_object_url = fields.get('firstDigitalObject', [{}])[0].get('objectUrl')
		ancesstor_collections = [f"{itm.get('title')}" for itm in record.get('ancestors')] # record.get('ancestors'): list of dict

		# doc_title = clean_(text=record.get('title'), sw=STOPWORDS)
		# doc_description = clean_(text=record.get('scopeAndContentNote'), sw=STOPWORDS) if record.get('scopeAndContentNote') else None


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
			"traffic statistics:" not in doc_title.lower(),
			"sketch" not in doc_title,
		] if doc_title is not None else []

		useless_description_terms = [
			"certificate" not in doc_description.lower(),
			"drawing" not in doc_description.lower(),
			"sketch of" not in doc_description.lower(),
			"newspaper" not in doc_description.lower(),
			"sketch" not in doc_description.lower(),
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
		row = {
			'id': na_identifier,
			'user_query': label,
			'title': doc_title,
			'description': doc_description,
			'img_url': first_digital_object_url,
			'enriched_document_description': label + " " + (doc_title or '') + " " + (doc_description or ''),
			'raw_doc_date': raw_doc_date,
			'doc_url': f"https://catalog.archives.gov/id/{na_identifier}",
			'img_path': f"{os.path.join(IMAGE_DIRECTORY, str(na_identifier) + '.jpg')}"
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
		all_label_tags = [line.strip() for line in file_]
	print(type(all_label_tags), len(all_label_tags))

	print(f"{len(all_label_tags)} lables are being processed...")
	labels_with_ZERO_result = list()
	dfs = []
	for qi, qv in enumerate(all_label_tags):
		print(f"\nQ[{qi+1}/{len(all_label_tags)}]: {qv}")
		qv = clean_(text=qv, sw=STOPWORDS)
		label_all_hits = get_data(
			start_date=START_DATE,
			end_date=END_DATE,
			label=qv,
		)
		if label_all_hits:
			df = get_dframe(
				label=qv,
				docs=label_all_hits,
			)
			if df is not None and df.shape[0]>1:
				dfs.append(df)
		else:
			labels_with_ZERO_result.append(qv)

	print(f">> {len(labels_with_ZERO_result)} labels with no results {START_DATE} - {END_DATE}\n{labels_with_ZERO_result}")

	print(f"<!> Concatinating {len(dfs)} dfs")
	concat_st = time.time()
	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f"Concatinated dfs: {df_merged_raw.shape} Elapsed time: {time.time()-concat_st:.1f} sec")

	print(f"<!> Replacing labels with broad umbrella terms")
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
			'title': 'first',
			'description': 'first',
			'user_query': lambda x: list(set(x)),  # Combine user_query into a list with unique elements
			'enriched_document_description': 'first',
			'raw_doc_date': 'first',
			'doc_url': 'first',
			'doc_date': 'first',
			'img_path': 'first',
		}
	).reset_index()
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))

	# Map user_query to labels using replacement_dict
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))
	print(f"Multi-label dataset shape: {grouped.shape}")
	grouped.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.csv"), index=False)

	try:
		grouped.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	print("\n2. Processing images (using multi-label dataset as reference)...")
	multi_label_final_df = get_synchronized_df_img(
		df=grouped,
		image_dir=IMAGE_DIRECTORY,
		nw=args.num_workers,
		enable_thumbnailing=args.enable_thumbnailing,
		thumbnail_size=args.thumbnail_size,
		large_image_threshold_mb=args.large_image_threshold_mb,
	)
	############################## aggregating user_query to list ##############################

	# label_dirstribution_fname = os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_label_distribution_nIMGs_{na_df.shape[0]}.png")
	# plot_label_distribution(
	# 	df=na_df,
	# 	dname=dataset_name,
	# 	fpth=label_dirstribution_fname,
	# )
	
	# na_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	# try:
	# 	na_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	# except Exception as e:
	# 	print(f"Failed to write Excel file: {e}")

	# # stratified splitting:
	# train_df, val_df = get_stratified_split(df=na_df, val_split_pct=args.val_split_pct,)

	# train_df.to_csv(os.path.join(DATASET_DIRECTORY, 'metadata_train.csv'), index=False)
	# val_df.to_csv(os.path.join(DATASET_DIRECTORY, 'metadata_val.csv'), index=False)

	# plot_train_val_label_distribution(
	# 	train_df=train_df,
	# 	val_df=val_df,
	# 	dataset_name=dataset_name,
	# 	OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
	# 	VAL_SPLIT_PCT=args.val_split_pct,
	# 	fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_simple_random_split_stratified_label_distribution_train_val_{args.val_split_pct}_pct.png'),
	# 	FIGURE_SIZE=(14, 8),
	# 	DPI=DPI,
	# )

	# plot_year_distribution(
	# 	df=na_df,
	# 	dname=dataset_name,
	# 	fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_year_distribution_nIMGs_{na_df.shape[0]}.png"),
	# 	BINs=args.historgram_bin,
	# )

	# if args.img_mean_std:
	# 	try:
	# 		img_rgb_mean = load_pickle(fpath=img_rgb_mean_fpth)
	# 		img_rgb_std = load_pickle(fpath=img_rgb_std_fpth) # RGB images
	# 	except Exception as e:
	# 		print(f"<!> {e}")
	# 		img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
	# 			source=os.path.join(DATASET_DIRECTORY, "images"), 
	# 			num_workers=args.num_workers,
	# 			batch_size=args.batch_size,
	# 			img_rgb_mean_fpth=img_rgb_mean_fpth,
	# 			img_rgb_std_fpth=img_rgb_std_fpth,
	# 		)

	# 	print(f"IMAGE Mean: {img_rgb_mean} Std: {img_rgb_std}")


	print("\n3. Creating SINGLE-LABEL version (from successfully downloaded images)...")
	single_label_final_df = multi_label_final_df.copy()
	single_label_columns_to_keep = [
		col 
		for col in single_label_final_df.columns 
		if col not in ['multi_labels', 'user_query']
	]
	single_label_final_df = single_label_final_df[single_label_columns_to_keep].copy()
	
	print(f"Single-label dataset shape: {single_label_final_df.shape}")
	
	# Save single-label raw metadata (before image sync)
	single_label_raw_df = grouped[single_label_columns_to_keep].copy()
	single_label_raw_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_single_label_raw.csv"), index=False)
	try:
		single_label_raw_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_single_label_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write single-label Excel file: {e}")
	
	print("\nSaving final single-label dataset...")
	single_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_single_label.csv"), index=False)
	try:
		single_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_single_label.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write final single-label Excel file: {e}")
	
	print("Saving final multi-label dataset...")
	multi_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.csv"), index=False)
	try:
		multi_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_multi_label.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write final multi-label Excel file: {e}")
	
	def process_dataset_version(df, version_name, is_multi_label=False):
		"""Process a dataset version with visualizations and splits"""
		print(f"\n--- Processing {version_name} dataset ---")
		
		if is_multi_label:
			# For multi-label, we need special handling
			plot_label_distribution_fname = os.path.join(
				OUTPUT_DIRECTORY, 
				f"{dataset_name}_{version_name}_label_distribution_{df.shape[0]}_x_{df.shape[1]}.png"
			)
			# You might want to create a special multi-label visualization here
			print(f"Multi-label visualization needs special handling - skipping for now")
		else:
			# Single-label visualization
			plot_label_distribution_fname = os.path.join(
				OUTPUT_DIRECTORY, 
				f"{dataset_name}_{version_name}_label_distribution_{df.shape[0]}_x_{df.shape[1]}.png"
			)
			plot_label_distribution(
				df=df,
				dname=f"{dataset_name}_{version_name}",
				fpth=plot_label_distribution_fname,
			)

		if not is_multi_label:
			# Single-label stratified split
			train_df, val_df = get_stratified_split(
				df=df, 
				val_split_pct=args.val_split_pct,
				label_col='label'
			)
			# Save train/val splits
			train_df.to_csv(os.path.join(DATASET_DIRECTORY, f'metadata_{version_name}_train.csv'), index=False)
			val_df.to_csv(os.path.join(DATASET_DIRECTORY, f'metadata_{version_name}_val.csv'), index=False)
		else:
			print(f"Multi-label stratified split not implemented yet!")
		
		# Train/val distribution plot
		if not is_multi_label:  # Only for single-label for now
			plot_train_val_label_distribution(
				train_df=train_df,
				val_df=val_df,
				dataset_name=f"{dataset_name}_{version_name}",
				OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
				VAL_SPLIT_PCT=args.val_split_pct,
				fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_{version_name}_stratified_label_distribution_train_val_{args.val_split_pct}_pct.png'),
				FIGURE_SIZE=(14, 8),
				DPI=DPI,
			)
		
		# Year distribution plot
		plot_year_distribution(
			df=df,
			dname=f"{dataset_name}_{version_name}",
			fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_{version_name}_year_distribution_{df.shape[0]}_samples.png"),
			BINs=args.historgram_bin,
		)
		print(f"{version_name} dataset processing complete!")
		
	process_dataset_version(
		df=single_label_final_df, 
		version_name="single_label", 
		is_multi_label=False
	)
	
	process_dataset_version(
		df=multi_label_final_df, 
		version_name="multi_label", 
		is_multi_label=True
	)

	if args.img_mean_std:
		try:
			img_rgb_mean = load_pickle(fpath=img_rgb_mean_fpth) 
			img_rgb_std = load_pickle(fpath=img_rgb_std_fpth)
		except Exception as e:
			print(f"{e}")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
				source=os.path.join(DATASET_DIRECTORY, "images"), 
				num_workers=args.num_workers,
				batch_size=args.batch_size,
				img_rgb_mean_fpth=img_rgb_mean_fpth,
				img_rgb_std_fpth=img_rgb_std_fpth,
			)
		print(f"IMAGE Mean: {img_rgb_mean} Std: {img_rgb_std}")

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