import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
print(project_dir)
print(os.listdir(project_dir))
sys.path.insert(0, project_dir) # add project directory to sys.path
from misc.utils import *
from misc.visualize import *

# import fasttext
# FastText_Language_Identification = "lid.176.bin"
# if FastText_Language_Identification not in os.listdir():
# 	url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FastText_Language_Identification}"
# 	urllib.request.urlretrieve(url, FastText_Language_Identification)
# detector_model = fasttext.load_model(FastText_Language_Identification)

from mediapipe.tasks import python
language_detector = "language_detector.tflite"
if language_detector not in os.listdir():
	url = f"https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/{language_detector}"
	urllib.request.urlretrieve(url, language_detector)
print("Running mediapipe Language Detector on CPU...")
base_options = python.BaseOptions(model_asset_path=language_detector)
options = python.text.LanguageDetectorOptions(base_options=base_options)
detector_model = python.text.LanguageDetector.create_from_options(options)

dataset_name: str = "europeana".upper()

parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1900-01-01", help='Dataset DIR')
parser.add_argument('--end_date', '-edt', type=str, default="1970-12-31", help='Dataset DIR')
parser.add_argument('--num_workers', '-nw', type=int, default=12, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
parser.add_argument('--val_split_pct', '-vsp', type=float, default=0.35, help='Validation Split Percentage')
parser.add_argument('--enable_thumbnailing', action='store_true', help='Enable image thumbnailing')
parser.add_argument('--thumbnail_size', type=parse_tuple, default=(1000, 1000), help='Thumbnail size (width, height) in pixels')
parser.add_argument('--large_image_threshold_mb', type=float, default=1.0, help='Large image threshold in MB')
parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')

args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print_args_table(args=args, parser=parser)

# run in local laptop:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 16 -bs 128 --img_mean_std --enable_thumbnailing > logs/europeana_img_dl.out &

# run in Pouta:
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -nw 40 -bs 128 --img_mean_std --enable_thumbnailing > /media/volume/ImACCESS/trash/europeana_dl.out &

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
europeana_api_base_url: str = "https://api.europeana.eu/record/v2/search.json"
# europeana_api_key: str = "api2demo"
europeana_api_key: str = "nLbaXYaiH"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}

DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")
IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")
OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")
os.makedirs(DATASET_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(HITs_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

# def is_english(text: str, detector_model, confidence_threshold: float = 0.7) -> bool:
# 	""" using fasttext """
# 	if not text or not text.strip():
# 		return False
	
# 	try:
# 		# Clean text for better detection
# 		cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
		
# 		# Predict language
# 		predictions = detector_model.predict(cleaned_text, k=1)
# 		detected_lang = predictions[0][0].replace('__label__', '')
# 		confidence = predictions[1][0]
		
# 		# Check if English with sufficient confidence
# 		is_en = detected_lang == 'en' and confidence >= confidence_threshold
		
# 		return is_en
# 	except Exception as e:
# 		print(f"Language detection error for text '{text}'\n{e}")
# 		return False

def is_english(text: str, detector_model, confidence_threshold: float = 0.3) -> bool:
	""" using MediaPipe """
	if not text or not text.strip():
		return False
	
	try:
		# Clean text for better detection
		cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
		
		# Detect language
		detection_result = detector_model.detect(cleaned_text)
		top_detection = detection_result.detections[0]
		is_en = (top_detection.language_code == 'en' and top_detection.probability >= confidence_threshold)
		return is_en
		
	except Exception as e:
		print(f"Language detection error: {text}\n{e}")
		return False

def get_europeana_date_or_year(doc_date, doc_year):
	if doc_year is not None:
		return doc_year
	if doc_date is None:
		return None
	# Regular expression patterns for exact date and year
	date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
	year_pattern = re.compile(r'\b\d{4}\b')
	for item in doc_date:
		if 'def' in item:
			date_match = date_pattern.search(item['def'])
			if date_match:
				return date_match.group(0)
			year_match = year_pattern.search(item['def'])
			if year_match:
				return year_match.group(0)
	return None

def get_data(start_date: str="1914-01-01", end_date: str="1914-01-02", label: str="world war"):
	t0 = time.time()
	label_processed = re.sub(" ", "_", label)
	label_all_hits_fpth = os.path.join(HITs_DIR, f"results_query_{label_processed}_{start_date}_{end_date}.gz")
	try:
		label_all_hits = load_pickle(fpath=label_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all docs for label: « {label} » might take a while...")
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
			# 'reusability': 'open' # TODO: LICENCE issues(must be resolved later!)
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
		europeana_id = doc.get("id")
		doc_id = re.sub("/", "SLASH", europeana_id)
		doc_title_list = doc.get("title") # ["title1", "title2", "title3", ...]
		doc_description_list = doc.get("dcDescription" )# ["desc1", "desc2", "desc3", ...]

		doc_title = ' '.join(doc_title_list) if doc_title_list else None
		doc_description = " ".join(doc_description_list) if doc_description_list else None

		image_url = doc.get("edmIsShownBy")[0]

		pDate = doc.get("edmTimespanLabel")[0].get("def") if (doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def")) else None
		raw_doc_date = doc.get("edmTimespanLabel")
		doc_year = doc.get("year")[0] if (doc.get("year") and doc.get("year")[0]) else None
		doc_url = f"https://www.europeana.eu/en/item{europeana_id}" # doc.get("guid")
		print(f"-"*50)
		print(f'{doc.get("title")}: {[is_english(text=title, detector_model=detector_model) for title in doc.get("title") if title]}')
		for title in doc.get("title"):
			if title and is_english(text=title, detector_model=detector_model):
				title_en = title
				break
			else:
				title_en = None
		print(f"title_en: {title_en}")

		description_en = " ".join(doc.get("dcDescriptionLangAware", {}).get("en", [])) if doc.get("dcDescriptionLangAware", {}).get("en", []) else None
		print(f"description_en: {description_en}")

		# enriched_document_description = (title_en or '') + " " + (description_en or '')
		# enriched_document_description = enriched_document_description.lstrip() if len(enriched_document_description) > 1 else None
		# print(f"enriched_document_description: {enriched_document_description}")
		# print(f"-"*50)
	
		raw_enriched_document_description = " ".join(filter(None, [title_en, description_en])).strip()
		print(f"\nraw_enriched_document_description:\n{raw_enriched_document_description}\n")
		enriched_document_description = basic_clean(txt=raw_enriched_document_description)
		print(f"\nenriched_document_description:\n{enriched_document_description}\n")


		if (
			image_url 
			and (image_url.endswith('.jpg') or image_url.endswith('.jpeg'))
		):
			pass # Valid entry; no action needed here
		else:
			image_url = None
		row = {
			'doc_id': europeana_id,
			'id': doc_id,
			'user_query': label,
			'title': title_en,
			'description': description_en,
			'enriched_document_description': enriched_document_description,
			'img_url': image_url,
			"doc_url": doc_url,
			'raw_doc_date': raw_doc_date,
			'doc_year': doc_year,
			'country': doc.get("country")[0],
			'img_path': f"{os.path.join(IMAGE_DIRECTORY, str(doc_id) + '.jpg')}"
		}
		data.append(row)
	df = pd.DataFrame(data)

	# Apply the function to the 'raw_doc_date' and 'doc_year' columns
	df['doc_date'] = df.apply(lambda row: get_europeana_date_or_year(row['raw_doc_date'], row['doc_year']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[df['doc_date'].apply(lambda x: is_valid_date(date=x, start_date=START_DATE, end_date=END_DATE))]

	print()
	print(f"DF: {df.shape} {list(df.columns)}")
	print(f"Elapsed_t: {time.time()-df_st_time:.1f} sec".center(160, "-"))
	return df

@measure_execution_time
def main():
	set_seeds(seed=args.seed, debug=False)
	with open(os.path.join(project_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		all_label_tags = list(set([line.strip() for line in file_]))
	print(type(all_label_tags), len(all_label_tags))
	print(f"{len(all_label_tags)} lables are being processed...")
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
			if df is not None:
				dfs.append(df)

	print(f">> Concatinating {len(dfs)} dfs")
	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f">> Concatinated dfs: {df_merged_raw.shape}")

	print(f"<!> Replacing labels with super classes")
	json_file_path = os.path.join(project_dir, 'misc', 'super_labels.json')
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")
		replacement_dict = {}

	unq_labels = set(replacement_dict.values())
	print(f">> {len(unq_labels)} Unique Label(s):\n{unq_labels}")

	print(f">> Pre-processing raw merged {type(df_merged_raw)} {df_merged_raw.shape}")

	print(f">> Merging user_query to label with umbrella terms from {json_file_path}")
	df_merged_raw['label'] = df_merged_raw['user_query'].replace(replacement_dict)
	print(f">> Found {df_merged_raw['img_url'].isna().sum()} None img_url / {df_merged_raw.shape[0]} total samples")
	df_merged_raw = df_merged_raw.dropna(subset=['img_url'])

	# ############################# Dropping duplicated img_url ##############################
	# print("\n1. Creating SINGLE-LABEL version (dropping duplicates)...")
	# print(f"Handling img_url with duplicate...")
	# print(f"img_url duplicates before removal: {df_merged_raw['img_url'].duplicated().sum()}")
	# single_label_df = df_merged_raw.drop_duplicates(subset=['img_url'])
	# print(f"img_url duplicates after removal: {single_label_df['img_url'].duplicated().sum()}")
	# print(f"Single-label dataset shape: {single_label_df.shape}")

	# single_label_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_single_label_raw.csv"), index=False)
	# try:
	# 	single_label_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_single_label_raw.xlsx"), index=False)
	# except Exception as e:
	# 	print(f"Failed to write Excel file: {e}")

	# single_label_final_df = get_synchronized_df_img(
	# 	df=single_label_df,
	# 	image_dir=IMAGE_DIRECTORY,
	# 	nw=args.num_workers,
	# 	enable_thumbnailing=args.enable_thumbnailing,
	# 	thumbnail_size=args.thumbnail_size,
	# 	large_image_threshold_mb=args.large_image_threshold_mb,
	# )
	# ############################# Dropping duplicated img_url ##############################

	############################## aggregating user_query to list ##############################
	print(f"\n1. Creating MULTI-LABEL version (aggregating user_query) from raw data: {df_merged_raw.shape}...")
	grouped = df_merged_raw.groupby('img_url').agg(
		{
			'id': 'first',
			'doc_id': 'first',
			'title': 'first',
			'description': 'first',
			'user_query': lambda x: list(set(x)),  # Combine user_query into a list with unique elements
			'enriched_document_description': 'first',
			'raw_doc_date': 'first',
			'doc_year': 'first',
			'doc_url': 'first',
			'img_path': 'first',
			'doc_date': 'first',
			'country': 'first',
		}
	).reset_index()

	# Map user_query to labels using replacement_dict
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))
	print(f"Multi-label dataset shape: {grouped.shape}")
	grouped.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_multi_label_raw.csv"), index=False)

	try:
		grouped.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_multi_label_raw.xlsx"), index=False)
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
				fpth=plot_label_distribution_fname,
				FIGURE_SIZE=(14, 8),
				DPI=DPI,
				label_column='label',
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

def test():
	query = "RESERVOIR"
	params = {
		'wskey': 'nLbaXYaiH',  # Your API key
		'qf': [
			'collection:photography', 
			'TYPE:"IMAGE"', 
			# 'contentTier:"4"', # high quality images
			'MIME_TYPE:image/jpeg',
		],
		'rows': 100, # does not work more than 100!
		'query': query,
		# 'reusability': 'open',
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
			for item_idx, item in enumerate(items):
				print(f"\nidx: {item_idx}: {list(item.keys())}")
				print(item.get("id"))
				print(f"-"*50)
				print(f'{item.get("title")}: {[is_english(text=title, detector_model=detector_model) for title in item.get("title") if title]}')
				for title in item.get("title"):
					if title and is_english(text=title, detector_model=detector_model):
						title_en = title
						break
					else:
						title_en = None
				print(f"title_en: {title_en}")
				description_en = " ".join(item.get("dcDescriptionLangAware", {}).get("en", [])) if item.get("dcDescriptionLangAware", {}).get("en", []) else None
				print(f"description_en: {description_en}")
				enriched_document_description = (title_en or '') + " " + (description_en or '')
				enriched_document_description = enriched_document_description.lstrip() if len(enriched_document_description) > 1 else None
				print(f"enriched_document_description: {enriched_document_description}")
				print(f"-"*50)
				print(item.get("type"))
				print(item.get("edmConceptLabel"))
				print(item.get("edmConceptPrefLabelLangAware"))
				print(item.get("edmIsShownAt"))
				print(item.get("edmIsShownBy"))
				print(item.get("edmTimespanLabel"))
				print(item.get("language"))
				print(item.get("dataProvider"))
				print(item.get("provider"))
				print(item.get("country")[0])
				# print(json.dumps(item, indent=2, assure_ascii=False))
				print("#"*150)
		else:
			print("No 'items' found in the response.")
	else:
		print(f"Request failed with status code {response.status_code}")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	# test()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
