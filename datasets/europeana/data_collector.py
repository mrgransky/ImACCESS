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

# # MediaPipe Language Detector: Not good for short texts:
# from mediapipe.tasks import python
# language_detector = "language_detector.tflite"
# if language_detector not in os.listdir():
# 	print(f"Downloading {language_detector} [takes a while]...")
# 	url = f"https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/{language_detector}"
# 	urllib.request.urlretrieve(url, language_detector)
# print("Running mediapipe Language Detector on CPU...")
# base_options = python.BaseOptions(model_asset_path=language_detector)
# options = python.text.LanguageDetectorOptions(base_options=base_options)
# detector_model = python.text.LanguageDetector.create_from_options(options)

# # FastText: Good for short texts:
# import fasttext
# FastText_Language_Identification = "lid.176.bin"
# if FastText_Language_Identification not in os.listdir():
# 	print(f"Downloading {FastText_Language_Identification} [takes	a while]...")
# 	url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FastText_Language_Identification}"
# 	urllib.request.urlretrieve(url, FastText_Language_Identification)
# print("Loading FastText Language Identification Model...")
# ft_model = fasttext.load_model(FastText_Language_Identification)

dataset_name: str = "europeana".upper()
# europeana_api_key: str = "api2demo"
# europeana_api_key: str = "nLbaXYaiH"

# run in local laptop:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs -ak api2demo
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -ak api2demo -nw 16 -bs 128 --img_mean_std --enable_thumbnailing > logs/europeana_img_dl.out &

# run in Pouta:
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -ak api2demo -nw 40 -bs 128 --img_mean_std --enable_thumbnailing > /media/volume/ImACCESS/trash/europeana_dl.out &

meaningless_words_fpth = os.path.join(project_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = nltk.corpus.stopwords.words('english')
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)

europeana_api_base_url: str = "https://api.europeana.eu/record/v2/search.json"
headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}

# Install: pip install lingua-language-detector
from lingua import Language, LanguageDetectorBuilder, IsoCode639_1

print("Initializing Lingua Language Detector...")

# OPTIMIZATION 1: Restrict the languages. 
# Since this is Europeana/Western data, we only check against common European languages.
# This DRASTICALLY improves accuracy on short text.
languages_to_check = [
	IsoCode639_1.EN, # English
	IsoCode639_1.DE, # German
	IsoCode639_1.FR, # French
	IsoCode639_1.ES, # Spanish
	IsoCode639_1.IT, # Italian
	IsoCode639_1.NL, # Dutch
	IsoCode639_1.PT, # Portuguese
	IsoCode639_1.SV, # Swedish
	IsoCode639_1.FI, # Finnish
	IsoCode639_1.DA, # Danish
	IsoCode639_1.NB, # Norwegian Bokmål
	IsoCode639_1.NN, # Norwegian Nynorsk
	IsoCode639_1.PL, # Polish
	IsoCode639_1.RU, # Russian
	IsoCode639_1.HU, # Hungarian
	IsoCode639_1.CS, # Czech
	IsoCode639_1.SK, # Slovak
	IsoCode639_1.EL, # Greek
	IsoCode639_1.BG, # Bulgarian
	IsoCode639_1.RO, # Romanian
]

detector = (
	LanguageDetectorBuilder
	.from_iso_codes_639_1(*languages_to_check)
	.with_preloaded_language_models()
	.build()
)

def is_english(
	text: str,
	confidence_threshold: float = 0.1, # We can now use a safer, higher threshold
	verbose: bool = False,
) -> bool:
	if not text or not str(text).strip():
		return False
	if verbose: print(f"Checking if text is in English:\n{text}\n")
	try:
		cleaned_text = " ".join(str(text).split())
		
		# if cleaned_text.isnumeric():
		# 	return False

		results = detector.compute_language_confidence_values(cleaned_text)
		
		if verbose:
			print(f"All detected languages:")
			for res in results:
				print(f"{res.language.name:<15}{res.value}")
		
		if not results:
			return False
		
		for res in results:
			if res.language == Language.ENGLISH:
				score = res.value
								
				if verbose:
					print(f"Is English({score} > Confidence threshold: {confidence_threshold}): {score > confidence_threshold}")

				if score > confidence_threshold:
					return True

				return False
		return False
	except Exception as e:
		if verbose: print(f"Error: {e}")
		return False

# def is_english(
# 	text: str, 
# 	detector_model: fasttext.FastText._FastText=ft_model,
# 	confidence_threshold: float = 0.4,
# 	verbose: bool = False,
# ) -> bool:
# 	"""
# 		Detects if text is in English using fasttext 
# 	"""
# 	if not text or not text.strip():
# 		return False
	
# 	try:
# 		# Clean text for better detection
# 		cleaned_text = " ".join(text.split())
		
# 		# Predict language
# 		predictions = detector_model.predict(cleaned_text, k=1)
# 		if verbose:
# 			print(f"\nchecking if text is in English:")
# 			print(f"{cleaned_text}") 
# 			print(f"Predictions: {predictions}")
# 			print(f"-"*70)
# 		detected_lang = predictions[0][0].replace('__label__', '')
# 		confidence = predictions[1][0]
		
# 		# Check if English with sufficient confidence
# 		is_en = detected_lang == 'en' and confidence >= confidence_threshold
		
# 		return is_en
# 	except Exception as e:
# 		print(f"Language detection error for text: '{text}'\n{e}")
# 		return False

# def is_english(
# 		text: str, 
# 		detector_model: python.text.LanguageDetector=detector_model,
# 		confidence_threshold: float = 0.3,
# 		verbose: bool = True,
# 	) -> bool:
# 	""" Detects if text is in English using MediaPipe """
# 	if not text or not text.strip():
# 		return False
	
# 	try:
# 		# Clean text for better detection
# 		cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
		
# 		# Detect language
# 		detection_result = detector_model.detect(cleaned_text)
# 		if verbose: print(f"Detection result: {detection_result}")
# 		top_detection = detection_result.detections[0]
# 		is_en = (top_detection.language_code == 'en' and top_detection.probability >= confidence_threshold)
# 		return is_en
		
# 	except Exception as e:
# 		print(f"Language detection error: {text}\n{e}")
# 		return False

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

def get_data(europeana_api_key: str, start_date: str, end_date: str, hits_dir: str, image_dir: str, label: str):
	t0 = time.time()
	label_processed = re.sub(" ", "_", label)
	label_all_hits_fpth = os.path.join(hits_dir, f"results_query_{label_processed}_{start_date}_{end_date}.gz")
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
			try:
				response = requests.get(
					url=europeana_api_base_url,
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

def get_dframe(label: str, image_dir: str, start_date: str, end_date: str, docs: List=[Dict]):
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
		pDate = doc.get("edmTimespanLabel")[0].get("def") if (doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def")) else None

		image_url = doc.get("edmIsShownBy")[0]
		raw_doc_date = doc.get("edmTimespanLabel")
		doc_year = doc.get("year")[0] if (doc.get("year") and doc.get("year")[0]) else None
		doc_url = f"https://www.europeana.eu/en/item{europeana_id}" # doc.get("guid")

		print(f'Raw title(s): {doc.get("title")}: {[is_english(text=title) for title in doc.get("title") if title]}')
		for title in doc.get("title"):
			if title and is_english(text=title) and title.lower() not in STOPWORDS:
				title_en = title
				break
			else:
				title_en = None
		print(f"title_en: {title_en}")

		description_doc = " ".join(doc.get("dcDescriptionLangAware", {}).get("en", [])) if doc.get("dcDescriptionLangAware", {}).get("en", []) else None
		if is_english(text=description_doc, verbose=True):
			description_en = description_doc
		else:
			description_en = None
		print(f"description_en:\n{description_en}")
	
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
			'img_url': image_url,
			"doc_url": doc_url,
			'raw_doc_date': raw_doc_date,
			'doc_year': doc_year,
			'country': doc.get("country")[0],
			'img_path': f"{os.path.join(image_dir, str(doc_id) + '.jpg')}",
			'title': title_en,
			'description': description_en
		}
		data.append(row)

	df = pd.DataFrame(data)

	# Apply the function to the 'raw_doc_date' and 'doc_year' columns
	df['doc_date'] = df.apply(lambda row: get_europeana_date_or_year(row['raw_doc_date'], row['doc_year']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[df['doc_date'].apply(lambda x: is_valid_date(date=x, start_date=start_date, end_date=end_date))]

	print(f"DF: {type(df)} {df.shape} {list(df.columns)}")
	print(f"Elapsed_t: {time.time()-df_st_time:.2f} sec")

	return df

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
	parser.add_argument('--api_key', '-ak', type=str, required=True, choices=["api2demo", "nLbaXYaiH"], help='Europeana API Key (choose from valid options)')
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

	set_seeds(seed=args.seed, debug=False)

	DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{args.start_date}_{args.end_date}")
	IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")
	HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")
	OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(DATASET_DIRECTORY, exist_ok=True)
	os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
	os.makedirs(HITs_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

	img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
	img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

	with open(os.path.join(project_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		search_labels = list(dict.fromkeys(line.strip() for line in file_))

	print(f"Total of {len(search_labels)} {type(search_labels)} lables are being processed...")
	for qv in search_labels[:5]:
		print(f"Q: {qv}")

	dfs = []
	for qi, qv in enumerate(search_labels):
		print(f"\nQ[{qi+1}/{len(search_labels)}]: {qv}")
		qv = clean_(text=qv, sw=STOPWORDS)
		label_all_hits = get_data(
			europeana_api_key=args.api_key,
			start_date=args.start_date,
			end_date=args.end_date,
			hits_dir=HITs_DIR,
			image_dir=IMAGE_DIRECTORY,
			label=qv,
		)
		if label_all_hits:
			qv_processed = re.sub(
				pattern=" ", 
				repl="_", 
				string=qv,
			)
			df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_{args.start_date}_{args.end_date}.gz")
			try:
				df = load_pickle(fpath=df_fpth)
			except Exception as e:
				df = get_dframe(
					label=qv,
					docs=label_all_hits,
					image_dir=IMAGE_DIRECTORY,
					start_date=args.start_date,
					end_date=args.end_date,
				)
				save_pickle(pkl=df, fname=df_fpth)
			if df is not None:
				dfs.append(df)

	total_searched_labels = len(dfs)
	print(f">> Concatinating {total_searched_labels} x {type(dfs[0])} dfs ...")
	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f">> df_merged_raw: {df_merged_raw.shape} {type(df_merged_raw)} {list(df_merged_raw.columns)}")

	json_file_path = os.path.join(project_dir, 'misc', 'super_labels.json')
	print(f">> Loading super classes[umbrella terms]: {json_file_path}")
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")
		replacement_dict = {}

	unq_labels = set(replacement_dict.values())

	print(f">> Merging user_query to label with umbrella terms (total of {len(unq_labels)} unique labels) from {json_file_path}")
	df_merged_raw['label'] = df_merged_raw['user_query'].replace(replacement_dict)
	print(f">> Found {df_merged_raw['img_url'].isna().sum()} None img_url / {df_merged_raw.shape[0]} total samples")
	df_merged_raw = df_merged_raw.dropna(subset=['img_url'])
	print(f">> After dropping None img_url: {df_merged_raw.shape}")
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
			'img_path': 'first',
			'doc_id': 'first',
			'user_query': lambda x: list(set(x)),  # Combine user_query into a list with unique elements
			'raw_doc_date': 'first',
			'doc_year': 'first',
			'doc_url': 'first',
			'doc_date': 'first',
			'country': 'first',
			'title': 'first',
			'description': 'first',
		}
	).reset_index()

	# Map user_query to labels using replacement_dict
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))

	print(f"Multi-label dataset {type(grouped)} {grouped.shape} {list(grouped.columns)}")
	############################## aggregating user_query to list ##############################

	print("\n2. Image synchronization (using multi-label dataset as reference)...")
	synched_fpath = os.path.join(DATASET_DIRECTORY, f"synched_x_{total_searched_labels}_searched_labels_metadata_multi_label.csv")
	multi_label_synched_df = get_synchronized_df_img(
		df=grouped,
		synched_fpath=synched_fpath,
		nw=args.num_workers,
		enable_thumbnailing=args.enable_thumbnailing,
		thumbnail_size=args.thumbnail_size,
		large_image_threshold_mb=args.large_image_threshold_mb,
	)

	multi_label_final_df = get_enriched_description(df=multi_label_synched_df)

	print("Saving final multi-label dataset...")
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
				print(f'{item.get("title")}: {[is_english(text=title) for title in item.get("title") if title]}')
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
