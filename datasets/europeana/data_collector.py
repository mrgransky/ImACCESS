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
from misc.nlp_utils import get_enriched_description, validate_text_cleaning_pipeline, is_english

dataset_name: str = "europeana".upper()
# europeana_api_key: str = "api2demo"
# europeana_api_key: str = "nLbaXYaiH"

# run in local laptop:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs -ak api2demo
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -ak api2demo -nw 16 -bs 128 --img_mean_std --thumbnail_size 512,512 -v > logs/europeana_dataset_collection.out &

# run in Pouta:
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -ak api2demo -nw 40 -bs 128 --img_mean_std --thumbnail_size 512,512 -v > /media/volume/ImACCESS/trash/europeana_dataset_collection.out &

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

useless_terms = [
	"ex libris a.wiepjes",
	"ritning", "fotokuvert", 
	"photo envelope", "fragment of a letter", "reproduction of a letter",
	"poster", "stempel /", "/stamp",
	"recordings from belgian military hospital, 1917-1918",
	"ΟΙ ΝΕΚΡΟΙ ΜΑΣ", "orthophoto",
	"Machtiging tot", "Kriegserrinnerungen", "Handboek", "Textheft",
	"scheme", "schematic", "overview map", "schema", "diagram", 
	"drawing", "diploma",
	"technology history", "announcement about", "studio photo",
	"text book", "textbook", "text booklet", "handbook", "newspaper repro",
	"3D (hbim)",
]

useless_data_provider = [
	"Museum of World Culture",
	"Museum of Ethnography",
	"Count Károly Esterházy Mansion and County Museum - Pápa",
	"Toy Museum of the City of Nuremberg",
	"Wellcome Collection",
	"MAK – Museum of Applied Arts",
	"Estonian Filmarchives", # blank photos
	"Library of the Alliance Israélite Universelle",
	"Bibliotheca Hertziana, Max Planck Institute for Art History Rome. Photographic Collection",
	"Municipality of Thasos",
	"National Museum of Transylvanian History",
	"Greater Digital Library",
	"Roscheider Hof Open Air Museum",
	"The Photographic Archive of the Zentralinstitut für Kunstgeschichte",
	"Freies Deutsches Hochstift / Frankfurter Goethe-Museum",
	"Magnus-Hirschfeld-Gesellschaft",
]

useless_subjects = [
	"Manuscript", "Album", "Map", "Collection", "Food", "HIV/AIDS", "Marketplace",
	"Recreational Artifacts (hierarchy name )", "Albumen print", "Art of painting",
	'Sword', 'Art of sculpture',  "Newspaper",
	'Coin', 'Silver', 'silver (metal)', 'coins (money)', "Silver",
	# 'Daguerreotype process',
	"Postcard",
	"Book",
	'Illustration',
	# 'figures (illustrations)',
	# 'Occupation (historical period)',
	'2D Graphics',
	"Plexiglas",
	# "History",
	"Award",
	"Drawing",
	"Numismatics", 'Crop', 'Horn', 'Crown', 'Relay baton', "Revolution", "Spur",
	"Pattern", "Textile", "Cartography",
]

useless_creators = [
	"Westhoff",  # Catches both "L. Westhoff" and "L.Westhoff"
	"Reuven Rubin",
	"Takács Dezső",
	"Antonia Prodromou",
	"Stoedtner, Franz (Lichtbildverlag) (Herstellung) (Fotograf)",
]

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
		return label_all_hits
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
				f'proxy_dc_title:"{label}" OR proxy_dc_description:"{label}"',
			],
			'rows': 100,
			'query': label,
			# 'reusability': 'open' # TODO: LICENCE issues(must be resolved later!)
		}
		print(json.dumps(params, indent=2, ensure_ascii=False))
		print("-"*120)

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

def get_dframe(
	label: str, 
	image_dir: str, 
	start_date: str, 
	end_date: str, 
	docs: List=[Dict], 
	verbose: bool=False
):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for label: « {label} » might take a while...")
	df_st_time = time.time()

	data = []
	for doc_idx, doc in enumerate(docs):
		if verbose:
			print(f"\ndoc: {doc_idx} {type(doc)}\n{list(doc.keys())}\n")
			for k, v in doc.items():
				print(f"{k}: {v}")
			print()

		if doc.get("dataProvider")[0] in useless_data_provider:
			if verbose:
				print(f"IGNORING DOCUMENT from useless data provider: {doc.get('dataProvider')[0]}")
			continue

		europeana_id = doc.get("id")
		doc_id = re.sub("/", "SLASH", europeana_id)
		raw_doc_date = doc.get("edmTimespanLabel")

		# doc_title_list = doc.get("title") # ["title1", "title2", "title3", ...]
		# doc_description_list = doc.get("dcDescription" )# ["desc1", "desc2", "desc3", ...]
		# doc_title = ' '.join(doc_title_list) if doc_title_list else None
		# doc_description = " ".join(doc_description_list) if doc_description_list else None
		# pDate = doc.get("edmTimespanLabel")[0].get("def") if (doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def")) else None

		image_url = doc.get("edmIsShownBy")[0]

		# image_url_from_api = doc.get("link")

		doc_year = doc.get("year")[0] if (doc.get("year") and doc.get("year")[0]) else None
		
		# Fallback to edmTimespanLabel if year is not available
		if doc_year is None:
			raw_doc_date = doc.get("edmTimespanLabel")
			if raw_doc_date:
				doc_year = get_europeana_date_or_year(raw_doc_date, None)
		
		# Convert doc_year to int if it exists
		if doc_year:
			try:
				doc_year = int(doc_year)
			except (ValueError, TypeError):
				doc_year = None
		
		# exclude if doc_year is not in the desired range
		start_year = int(start_date.split("-")[0])
		end_year = int(end_date.split("-")[0])
		if doc_year and (doc_year < start_year or doc_year > end_year):
			if verbose:
				print(f"IGNORING DOCUMENT out of range: {doc_year} - URL: {doc.get('guid')}")
			continue

		doc_creator = doc.get("dcCreator")
		if doc_creator:
			# exclude if any of the elements from useless_creators is in doc_creator
			if any(creator in doc_creator for creator in useless_creators):
				if verbose:
					print(f"IGNORING DOCUMENT with useless creator: {doc_creator} - ID: {europeana_id}")
				continue

		doc_url = f"https://www.europeana.eu/en/item{europeana_id}" # doc.get("guid")

		# Check if document contains useless concept labels
		doc_concept_labels = doc.get("edmConceptLabel", [])
		useless_concept = next(
			(concept['def'] for concept in doc_concept_labels
			if isinstance(concept, dict) and concept.get('def') in useless_subjects),
			None
		)
		if useless_concept:
			if verbose:
				print(f"IGNORING DOCUMENT with concept: '{useless_concept}' - ID: {europeana_id}")
			continue  # ← Now correctly skips to next document!


		all_titles = doc.get("title", [])
		if any(
			term in title.lower()
			for title in all_titles if title
			for term in useless_terms
		):
			if verbose:
				matched = [(t, term) for t in all_titles if t for term in useless_terms if term in t.lower()]
				print(f"IGNORING DOCUMENT with useless term in title: {matched[0]} - ID: {europeana_id}")
			continue  # skip the document!

		# Try to get English title from dcTitleLangAware first
		title_en_list = doc.get("dcTitleLangAware", {}).get("en", None)
		if title_en_list:
			title_en = title_en_list[0] if isinstance(title_en_list, list) else title_en_list
		else:
			title_en = None
		
		# Fallback to language detection if no English title found
		if title_en is None:
			for title in doc.get("title", []):
				if not title:
					continue
				title_lower = title.lower()
				if (
					is_english(text=title, confidence_threshold=0.03, verbose=verbose)
					and not all(word in STOPWORDS for word in title_lower.split())
					and not any(term in title_lower for term in useless_terms)
				):
					title_en = title
					break

		description_doc = " ".join(doc.get("dcDescriptionLangAware", {}).get("en", [])) if doc.get("dcDescriptionLangAware", {}).get("en", []) else None
		if description_doc:
			description_lower = description_doc.lower()
			if any(term in description_lower for term in useless_terms):  # substring match
				if verbose:
					print(f"IGNORING DOCUMENT with useless term in description, ID: {europeana_id}")
				continue

			if is_english(text=description_doc, confidence_threshold=0.03, verbose=verbose):
				description_en = description_doc
			else:
				description_en = None
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

		print(json.dumps(row, indent=2, ensure_ascii=False))
		print("-"*100)

		data.append(row)

	df = pd.DataFrame(data)
	if df.empty:
		if verbose:
			print("Empty DataFrame. Returning None...")
		return None

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
	parser.add_argument('--thumbnail_size', type=parse_tuple, default=None, help='Thumbnail size (width, height) in pixels')
	parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

	args, unknown = parser.parse_known_args()
	args.dataset_dir = os.path.normpath(args.dataset_dir)
	print(args)
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

	print(f"Total of {len(search_labels)} {type(search_labels)} lables are being processed")

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
					verbose=args.verbose,
				)
				save_pickle(pkl=df, fname=df_fpth)
			if df is not None:
				dfs.append(df)

	total_searched_labels = len(dfs)
	print(f">> Concatinating {total_searched_labels} x {type(dfs[0])} dfs ...")
	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f">> df_merged_raw: {df_merged_raw.shape} {type(df_merged_raw)} {list(df_merged_raw.columns)}")

	# json_file_path = os.path.join(project_dir, 'misc', 'super_labels.json')
	json_file_path = os.path.join(project_dir, 'misc', 'canonical_labels.json')
	print(f">> Loading super classes[canonical terms]: {json_file_path}")
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")
		replacement_dict = {}

	unq_labels = list(set(replacement_dict.values()))

	print(f">> Merging user_query to label with canonical terms (total of {len(unq_labels)} unique labels) from {json_file_path}")
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

	# Map user_query to canonical labels
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))

	print(f"Multi-label dataset {type(grouped)} {grouped.shape} {list(grouped.columns)}")
	############################## aggregating user_query to list ##############################

	print("\n2. Image synchronization (using multi-label dataset as reference)...")
	synched_fpath = os.path.join(DATASET_DIRECTORY, f"synched_x_{total_searched_labels}_searched_labels_metadata_multi_label.csv")
	multi_label_synched_df = get_synchronized_df_img(
		df=grouped,
		synched_fpath=synched_fpath,
		nw=args.num_workers,
		thumbnail_size=args.thumbnail_size,
		verbose=args.verbose,
	)

	multi_label_final_df = get_enriched_description(df=multi_label_synched_df)
	validate_text_cleaning_pipeline(df=multi_label_final_df, text_column='enriched_document_description')

	print("Saving final MULTI-LABEL dataset...")
	print(f"multi_label_final_df: {type(multi_label_final_df)} {multi_label_final_df.shape} {list(multi_label_final_df.columns)}")
	multi_label_fpath = os.path.join(DATASET_DIRECTORY, "metadata_multi_label.csv")

	multi_label_final_df.to_csv(multi_label_fpath, index=False)
	try:
		multi_label_final_df.to_excel(multi_label_fpath.replace('.csv', '.xlsx'), index=False)
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
	single_label_fpath = multi_label_fpath.replace('multi_label', 'single_label')
	print(f"\nSaving final SINGLE-LABEL dataset in {single_label_fpath}...")
	single_label_final_df.to_csv(single_label_fpath, index=False)
	try:
		single_label_final_df.to_excel(single_label_fpath.replace('.csv', '.xlsx'), index=False)
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

	# if IMAGE_DIRECTORY is not empty, load it, else compute it
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

	# remove unnecessary image files which are not in the final dataset
	print("\nRemoving unnecessary image files which are not in the final dataset...")	
	# Extract actual filenames from img_path column
	valid_filenames = set()
	for img_path in single_label_final_df['img_path']:
		valid_filenames.add(os.path.basename(img_path))
	for img_path in multi_label_final_df['img_path']:
		valid_filenames.add(os.path.basename(img_path))
	print(f"Found {len(valid_filenames)} valid image files in the final dataset")

	for img_file in os.listdir(IMAGE_DIRECTORY):
		if img_file.endswith('.jpg'):
			if img_file not in valid_filenames:
				print(f"Removing unnecessary image file: {img_file}")
				os.remove(os.path.join(IMAGE_DIRECTORY, img_file))

	# confirm df size and number of images in the IMAGE_DIRECTORY are the same
	print("\nConfirming df size and number of images in the IMAGE_DIRECTORY are the same...")
	print(f"Number of images in the final dataset: {len(valid_filenames)}")
	print(f"Number of images in the IMAGE_DIRECTORY: {len(os.listdir(IMAGE_DIRECTORY))}")
	assert len(valid_filenames) == len(os.listdir(IMAGE_DIRECTORY)), "Number of images in the final dataset and in the IMAGE_DIRECTORY are not the same!"

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
				for title in item.get("title"):
					if title and is_english(text=title) and title.lower() not in STOPWORDS:
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
