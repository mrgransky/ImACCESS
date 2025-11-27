import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from misc.utils import *
import misc.visualize as viz

# local:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 --img_mean_std
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 16 -bs 128 --img_mean_std --enable_thumbnailing > logs/smu_dataset_collection.out &

# run in Pouta:
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -nw 40 -bs 256 --img_mean_std --enable_thumbnailing > /media/volume/ImACCESS/trash/smu_data_collection.out &

dataset_name = "smu".upper()
parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1900-01-01", help='Dataset DIR')
parser.add_argument('--end_date', '-edt', type=str, default="1970-12-31", help='Dataset DIR')
parser.add_argument('--num_workers', '-nw', type=int, default=12, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=512, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
parser.add_argument('--val_split_pct', '-vsp', type=float, default=0.35, help='Validation Split Percentage')
parser.add_argument('--enable_thumbnailing', action='store_true', help='Enable image thumbnailing')
parser.add_argument('--thumbnail_size', type=parse_tuple, default=(1000, 1000), help='Thumbnail size (width, height) in pixels')
parser.add_argument('--large_image_threshold_mb', type=float, default=1.0, help='Large image threshold in MB')

args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print_args_table(args=args, parser=parser)

SMU_BASE_URL:str = "https://digitalcollections.smu.edu/digital"
CUSTOMIZED_COLLECTIONS: list = [
	"apnd", "aaf", "outler", "ald", "alv", "han", "wsw", "lav", "bml", "other", "civ", "cooke", "pwl", "mbc", "eaa", "wlrd", "fjd", "gcp", "gcd", "white", "wlg", "kil", "jcc", "jhv", "mcs", "UKMeth", "mex", "ngc", "nam", "ptr", "rwy", "stn", "ryr", "rdoh", "tex", "bridhist", "haws", "wes", "wrl"
]
FORBIDDEN_DOCUMENT_FORMATS ="Letter Watercolors Paintings Copperwork Porcelain enamel Lithographs Tempera paintings Promotional materials Manuscripts Front cover Back cover Forms Documents Bibles Posters Currency Lyric poetry Real photographic postcards Newspapers Periodicals Viewbooks Book covers Postcards Pamphlets Scrapbooks Drawings Pencil works Metalwork Brasswork Correspondence Letters Timetables Ephemera Tickets Cards Maps Sketchbook Badges Slides Reproductions typed document 	Woodcuts Broadsides Relief prints Medals, Memorabilia Advertisements Matchcovers"
CUSTOMIZED_COLLECTIONS_STR: str = "!".join(CUSTOMIZED_COLLECTIONS)
FIGURE_SIZE = (12, 9)
DPI = 350

meaningless_words_fpth = os.path.join(project_dir, 'misc', 'meaningless_words.txt')
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)

headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}

DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{args.start_date}_{args.end_date}")
HITS_DIRECTORY = os.path.join(DATASET_DIRECTORY, "hits")
IMAGE_DIRECTORY = os.path.join(DATASET_DIRECTORY, "images")
OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")
os.makedirs(DATASET_DIRECTORY, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)
os.makedirs(HITS_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")
FIELD_KEY_MAP = {
	"creato": "creator",
	"datea": "iso_date",
	"decade": "decade",
	"audien": "audience",
	"covera": "coverage",
	"descri": "description",
	"notes": "notes",
	"langua": "language",
	"author": "subjects",
	"source": "source",
	"formge": "genre",
	"digita": "digital_type",
	"digiti": "digitized",
	"digitb": "display_format",
	"archiv": "archival_scan",
	"archia": "archival_image_a",
	"archib": "archival_image_b",
	"digitc": "digitization_notes",
	"digitd": "collection_name",
	"librar": "library",
	"publis": "publisher",
	"rights": "rights",
	"call": "call_number",
	"upload": "upload_file",
	"series": "series",
	"part": "collection_part",
	"place": "location",
	"keywor": "keywords",
	"physic": "physical_description",
	"boxfol": "box_folder",
	"relate": "related_resources"
}
REDUNDANT_KEYWORDS = {"World War 2", "World War II", "WW2"}

def get_doc_year(raw_doc_date):
	# if not pd.isna(raw_doc_date): # Check if raw_doc_date is missing (None or NaN)
	# 	return raw_doc_date
	# year_pattern = r'\b\d{4}\b'
	year_pattern = re.compile(r'\b\d{4}\b')
	match = re.search(year_pattern, raw_doc_date) # <re.Match object; span=(54, 58), match='1946'>
	print(match)
	if match:
		return match.group()
	else:
		return None

def get_data(start_date: str="1914-01-01", end_date: str="1914-01-02", query: str="world war"):
	t0 = time.time()
	START_DATE = re.sub("-", "", start_date) # modified date: 19140101
	END_DATE = re.sub("-", "", end_date) # modiifed date: 19140102
	query_processed = re.sub(" ", "_", query.lower())
	query_all_hits_fpth = os.path.join(HITS_DIRECTORY, f"results_{start_date}_{end_date}_query_{query_processed}.gz")
	try:
		query_all_hits = load_pickle(fpath=query_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all hits for Query: « {query} » ... might take a while..")
		query_all_hits = []
		pg = 1
		MAX_HITS_IN_ONE_PAGE = 200
		while True:
			##############################################
			# # previous functional but small collection:
			# query_url = f"{SMU_BASE_URL}/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/image!{query}!{query}!{START_DATE}-{END_DATE}/field/type!title!descri!date/mode/exact!exact!all!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			##############################################
			# query_url = f"{SMU_BASE_URL}/api/search/searchterm/image!{query}!{st_date}-{end_date}/field/type!all!date/mode/exact!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			# query_url = f"{SMU_BASE_URL}/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/{query}!{query}!{START_DATE}-{END_DATE}/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			# with selected collections and all field containing the query:
			# query_url = f"{SMU_BASE_URL}/api/search/collection/{CUSTOMIZED_COLLECTIONS_STR}/searchterm/searchterm/image!{query}!{START_DATE}-{END_DATE}/field/type!all!date/mode/exact!exact!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			# with forbiddden document formats:
			query_url = f"{SMU_BASE_URL}/api/search/searchterm/{FORBIDDEN_DOCUMENT_FORMATS}!{query}!image!{START_DATE}-{END_DATE}/field/all!all!type!date/mode/none!exact!exact!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			loop_st = time.time()
			response = requests.get(query_url)
			if response.status_code == 200:
				data = response.json()
				if 'items' in data:
					# Extract the 'items' field
					hits = data['items']
					total_hits = data['totalResults']
					# print(total_hits, len(hits))
					query_all_hits.extend(hits)
					# print(json.dumps(query_all_hits, indent=2, ensure_ascii=False))
					print(f"Page: {pg}:\tFound: {len(hits)} {type(hits)}\t{len(query_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
				if len(query_all_hits) >= total_hits:
					break
				pg += 1
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

def get_dframe(query: str, start_date:str, end_date:str, df_file_path: str):
	query_all_hits = get_data(
		start_date=args.start_date,
		end_date=args.end_date,
		query=query.lower()
	)
	if query_all_hits is None:
		return
	docs = query_all_hits
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc_idx, doc in enumerate(docs):
		doc_date = doc.get("metadataFields")[3].get("value")
		doc_type = doc.get('filetype')
		doc_link = doc.get("itemLink") # /singleitem/collection/ryr/id/2479
		doc_collection = doc.get("collectionAlias")
		doc_id = doc.get("itemId")
		doc_combined_identifier = f'{doc_collection}_{doc_id}' # agr_19
		doc_page_url = f"{SMU_BASE_URL}/collection/{doc.get('collectionAlias')}/id/{doc.get('itemId')}"
		doc_img_link = f"{SMU_BASE_URL}/api/singleitem/image/{doc_collection}/{doc_id}/default.jpg"
		doc_title = doc.get("title")
		doc_detailed_metadata = scrape_item_metadata(doc_url=doc_page_url)
		doc_description = doc_detailed_metadata.get("description", None)
		doc_raw_keywords = doc_detailed_metadata.get("keywords", None)

		doc_cleaned_keywords = None
		if doc_raw_keywords:
			keyword_list = [kw.strip() for kw in doc_raw_keywords.split(";") if kw.strip()] # Split the keywords into a list
			cleaned_keyword_list = [kw for kw in keyword_list if kw not in REDUNDANT_KEYWORDS] # Exclude redundant terms
			doc_cleaned_keywords = "; ".join(cleaned_keyword_list) if cleaned_keyword_list else None # Join the cleaned keywords back into a string
			print(f"doc_cleaned_keywords: {doc_cleaned_keywords}")

		row = {
			'id': doc_combined_identifier,
			'doc_url': doc_page_url,
			'user_query': query,
			'title': doc_title,
			'description': doc_description,
			'keywords': doc_cleaned_keywords,
			'raw_doc_date': doc_date,
			'img_path': f"{os.path.join(IMAGE_DIRECTORY, str(doc_combined_identifier) + '.jpg')}",
			'img_url': doc_img_link
		}
		data.append(row)
	df = pd.DataFrame(data)
	# Apply the function to the 'raw_doc_date' and 'doc_year' columns
	df['doc_date'] = df.apply(lambda row: get_doc_year(row['raw_doc_date']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[
		df['doc_date'].apply(
			lambda x: is_valid_date(
				date=x, 
				start_date=start_date, 
				end_date=end_date
			)
		)
	]

	print(f"DF: {type(df)} {df.shape} Elapsed_t: {time.time()-df_st_time:.1f} sec")

	save_pickle(pkl=df, fname=df_file_path)

	return df

@retry(
	reraise=True,
	stop=stop_after_attempt(3),
	wait=wait_exponential(multiplier=1, min=1, max=10),
	retry=retry_if_exception_type(httpx.RequestError)
)
def fetch_html(url: str, timeout: float = 10.0) -> str:
	"""Fetch HTML content from a URL with retry logic."""
	with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
		response = client.get(url)
		response.raise_for_status()
		return response.text

def extract_initial_state_json(html: str) -> Dict:
	"""Extract and decode JSON from the window.__INITIAL_STATE__ script tag."""
	soup = BeautifulSoup(html, 'html.parser')
	script_tag = soup.find('script', string=re.compile(r'window\.__INITIAL_STATE__'))
	if not script_tag:
			raise ValueError("Script tag with window.__INITIAL_STATE__ not found.")
	# Extract JSON.parse(...) string
	json_match = re.search(
			r'window\.__INITIAL_STATE__\s*=\s*JSON\.parse\("((?:\\.|[^"\\])*)"\);?',
			script_tag.string
	)
	if not json_match:
			raise ValueError("Unable to extract JSON.parse content from script tag.")
	encoded_json_str = json_match.group(1)
	try:
			# Decode double-escaped JSON string
			decoded_json = bytes(encoded_json_str, "utf-8").decode("unicode_escape")
			decoded_json = decoded_json.replace(r'\/', '/')  # Optional: fix slashes
			return json.loads(decoded_json)
	except json.JSONDecodeError as e:
			snippet = encoded_json_str[max(0, e.pos - 50):e.pos + 50]
			print(f"Failed to decode JSON near position {e.pos}: ...{snippet}...")
			raise e

def scrape_item_metadata(doc_url: str) -> Dict:
	try:
		html = fetch_html(doc_url)
		data = extract_initial_state_json(html)
		metadata = {}
		fields = data.get("item", {}).get("item", {}).get("fields", [])
		for field in fields:
			key = FIELD_KEY_MAP.get(field.get("key", "").strip().lower().replace(" ", "_"), "")
			value = field.get("value", "").strip()
			if key:
				metadata[key] = value
		return metadata
	except Exception as e:
		print(f"<!> [ERROR] Failed to scrape metadata from {doc_url}: {e}")
		return {}

@measure_execution_time
def main():
	with open(os.path.join(project_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		search_labels = list(dict.fromkeys(line.strip() for line in file_))
	print(f"Total of {len(search_labels)} {type(search_labels)} Query phrases are being processed, please be patient...")

	dfs = []
	for qi, qv in enumerate(search_labels):
		print(f"\nQ[{qi+1}/{len(search_labels)}]: {qv}")
		qv = clean_(text=qv, sw=STOPWORDS)
		qv_processed = re.sub(" ", "_", qv)
		df_fpth = os.path.join(HITS_DIRECTORY, f"df_query_{qv_processed}_{args.start_date}_{args.end_date}.gz")
		try:
			df = load_pickle(fpath=df_fpth)
		except Exception as e:
			print(e)
			df = get_dframe(
				query=qv,
				start_date=args.start_date,
				end_date=args.end_date,
				df_file_path=df_fpth,
			)
		if df is not None:
			dfs.append(df)

	total_searched_labels = len(dfs)
	print(f">> Concatinating {total_searched_labels} x {type(dfs[0])} dfs ...")

	df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f">> Concatinated dfs: {df_merged_raw.shape}")

	print(f">> Replacing labels with super classes")
	json_file_path = os.path.join(project_dir, 'misc', 'super_labels.json')
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")

	print(f">> Pre-processing raw merged {type(df_merged_raw)} {df_merged_raw.shape}")
	print(f">> Merging user_query to label with umbrella terms from {json_file_path}")
	df_merged_raw['label'] = df_merged_raw['user_query'].replace(replacement_dict)
	print(f">> Found {df_merged_raw['img_url'].isna().sum()} None img_url / {df_merged_raw.shape[0]} total samples")
	df_merged_raw = df_merged_raw.dropna(subset=['img_url'])
	print(f">> After dropping None img_url: {df_merged_raw.shape}")

	############################## Dropping duplicated img_url ##############################
	# print(f"img_url with duplicate: {df_merged_raw['img_url'].duplicated().sum()}")
	# num_duplicates = df_merged_raw.duplicated(subset='img_url', keep=False).sum()
	# print(f"num_duplicates ['img_url']: {num_duplicates}")
	# print(list(df_merged_raw.columns))

	# # Identify duplicates based on img_url
	# duplicates_mask = df_merged_raw['img_url'].duplicated(keep=False)

	# # Filter the DataFrame to keep only the duplicates
	# duplicate_rows = df_merged_raw[duplicates_mask]

	# # You can now select the required columns and print the result
	# result = duplicate_rows
	# print(result)

	# df_merged_raw = df_merged_raw.drop_duplicates(subset=['img_url'])

	# print(f"Processed df_merged_raw: {df_merged_raw.shape}")
	# print(df_merged_raw.head(20))

	# df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	# try:
	# 	df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	# except Exception as e:
	# 	print(f"Failed to write Excel file: {e}")
	# smu_df = get_synchronized_df_img(
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
			'raw_doc_date': 'first',
			'doc_url': 'first',
			'img_path': 'first',
			'doc_date': 'first'
		}
	).reset_index()
	print(f"Grouped: {grouped.shape}")
	
	# Map user_query to labels using replacement_dict
	grouped['label'] = grouped['user_query'].apply(lambda x: replacement_dict.get(x[0], x[0]))
	print(f"Multi-label dataset {type(grouped)} {grouped.shape} {list(grouped.columns)}")
	############################## aggregating user_query to list ##############################

	print(f"\n2. Image synchronization (using multi-label dataset as reference)...")
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
	
	print(f"Single-label dataset shape: {single_label_final_df.shape}")
		
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

	print("\nDATASET CREATION SUMMARY")
	print("-"*100)
	print(f"Single-label dataset: {single_label_final_df.shape}: {single_label_final_df['label'].nunique()} unique labels")
	print(list(single_label_final_df.columns))
	print(single_label_final_df['label'].value_counts().sort_values(ascending=False))
	print("-"*100)

	print(f"Multi-label dataset: {multi_label_final_df.shape} {list(multi_label_final_df.columns)}")
	print("-"*100)

	print(f"Unique images downloaded: {len(os.listdir(IMAGE_DIRECTORY))} files")
	print(f"Files created in {DATASET_DIRECTORY}:")
	for file in sorted(os.listdir(DATASET_DIRECTORY)):
		if file.endswith(('.csv', '.xlsx')):
			print(f"\t- {file}")

def test_on_search_(query="shovel", start_date="1900-01-01", end_date="1970-12-31"):
	START_DATE = re.sub("-", "", start_date) # modified date: 19140101
	END_DATE = re.sub("-", "", end_date) # modiifed date: 19140102
	MAX_HITS_IN_ONE_PAGE = 200
	# f"{SMU_BASE_URL}/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/president!president!18900101-19601231/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"

	# query_url = f"{SMU_BASE_URL}/api/search/collection/{CUSTOMIZED_COLLECTIONS_STR}/searchterm/image!{query}!{query}!{START_DATE}-{END_DATE}/field/type!title!descri!date/mode/exact!exact!all!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"

	# query_url = str(f"{SMU_BASE_URL}/api/search/searchterm/image!{query}!{START_DATE}-{END_DATE}/field/type!all!date/mode/exact!all!exact/conn/and!and!and/maxRecords/200/page/7")
	# without certain collections:
	# query_url = f"{SMU_BASE_URL}/api/search/collection/{CUSTOMIZED_COLLECTIONS_STR}/searchterm/{query}!{query}!{START_DATE}-{END_DATE}/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"
	# with all collections:
	# query_url = f"{SMU_BASE_URL}/api/search/searchterm/{query}!{query}!{START_DATE}-{END_DATE}/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"
	# with all collections and all field containing the query:
	# query_url = f"{SMU_BASE_URL}/api/search/searchterm/searchterm/image!{query}!{START_DATE}-{END_DATE}/field/type!all!date/mode/exact!exact!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"
	
	# with selected collections and all field containing the query:
	# query_url = f"{SMU_BASE_URL}/api/search/collection/{CUSTOMIZED_COLLECTIONS_STR}/searchterm/searchterm/Photographs!image!{query}!{START_DATE}-{END_DATE}/field/formge!type!all!date/mode/exact!exact!exact!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"
	# 
	query_url = f"{SMU_BASE_URL}/api/search/searchterm/{FORBIDDEN_DOCUMENT_FORMATS}!{query}!image!{START_DATE}-{END_DATE}/field/all!all!type!date/mode/none!exact!exact!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}"
	print(query_url)
	# Send a GET request to the API
	response = requests.get(query_url)
	print(response.status_code)
	print(type(response), response)
	print(response.headers['Content-Type']) # must be 'application/json'
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
				# print(f"item: {item_idx} {type(item)} {item.get('filetype')}")
				# print(list(item.keys()))
				if item.get('filetype') == "jp2":
					itm_collection = item.get("collectionAlias")
					itm_identifier = item.get("itemId")
					itm_link = f"{SMU_BASE_URL}/collection/{itm_collection}/id/{itm_identifier}" #item.get("itemLink")
					itm_api_link = f"{SMU_BASE_URL}/api/collections/{itm_collection}/items/{itm_identifier}/false"
					itm_img_link = f"{SMU_BASE_URL}/api/singleitem/image/{itm_collection}/{itm_identifier}/default.jpg"
					itm_title = item.get("title")
					# itm_description = 
					itm_date = item.get("metadataFields")[3].get("value")
					print(itm_title)
					print(itm_link)
					print(itm_img_link)
					print(itm_date)
					print(itm_collection)
					print("#"*100)
				else:
					pass
				# print()
				# print(item.get("id"))
				# print(item.get("title"))
				# print(item.get("edmIsShownAt"))
				# print(item.get("edmIsShownBy"))
				# print(item.get("edmTimespanLabel"))
				# print(item.get("language"))
				# print(item.get("dataProvider"))
				# print(item.get("provider"))
				# print("#"*100)
				# print(json.dumps(item, indent=2))  # Pretty-print the JSON data for each item
				# You can process or save the 'items' data as needed
		else:
			print("No 'items' found in the response.")
	else:
		print(f"Request failed with status code {response.status_code}")

def test_on_doc_page_url():
	doc_urls = [
		"https://digitalcollections.smu.edu/digital/collection/stn/id/1014",
		"https://digitalcollections.smu.edu/digital/collection/ryr/id/2239",
		"https://digitalcollections.smu.edu/digital/collection/ryr/id/3338"
	]
	for url in doc_urls:
		print(f"\nTesting metadata scraping for URL: {url}")
		metadata = scrape_item_metadata(url)
		if metadata:
			print(json.dumps(metadata, indent=2))
		else:
			print("No metadata extracted.")

if __name__ == '__main__':
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	# test_on_search_()
	# test_on_doc_page_url()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
