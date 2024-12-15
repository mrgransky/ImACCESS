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
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
# $ nohup python -u data_collector.py --dataset_dir $PWD --start_date 1900-01-01 --end_date 1970-12-31 --num_workers 8 > logs/smu_dataset_collection.out &

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
START_DATE = args.start_date
END_DATE = args.end_date

dataset_name = "SMU"
meaningless_words_fpth = os.path.join(parent_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
print(STOPWORDS, type(STOPWORDS))

headers = {
	'Content-type': 'application/json',
	'Accept': 'application/json; text/plain; */*',
	'Cache-Control': 'no-cache',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
}

os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")

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

def get_data(st_date: str="1914-01-01", end_date: str="1914-01-02", query: str="world war"):
	t0 = time.time()
	START_DATE = re.sub("-", "", args.start_date)
	END_DATE = re.sub("-", "", args.end_date)
	query_processed = re.sub(" ", "_", query.lower())
	query_all_hits_fpth = os.path.join(HITs_DIR, f"results_{st_date}_{end_date}_query_{query_processed}.gz")
	try:
		query_all_hits = load_pickle(fpath=query_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all hits for Query: « {query} » ... might take a while..")
		query_all_hits = []
		pg = 1
		MAX_HITS_IN_ONE_PAGE = 200
		while True:
			query_url = f"https://digitalcollections.smu.edu/digital/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/image!{query}!{query}!{START_DATE}-{END_DATE}/field/type!title!descri!date/mode/exact!exact!all!exact/conn/and!and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			# query_url = f"https://digitalcollections.smu.edu/digital/api/search/searchterm/image!{query}!{st_date}-{end_date}/field/type!all!date/mode/exact!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
			# query_url = f"https://digitalcollections.smu.edu/digital/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/{query}!{query}!{START_DATE}-{END_DATE}/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/{MAX_HITS_IN_ONE_PAGE}/page/{pg}"
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

def get_dframe(query: str="query", docs: List=[Dict]):
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for query: « {query} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc_idx, doc in enumerate(docs):
		# print(type(doc.get("title")), doc.get("title"))
		doc_date = doc.get("metadataFields")[3].get("value")
		doc_type = doc.get('filetype')
		doc_collection = doc.get("collectionAlias")
		doc_id = doc.get("itemId")
		doc_combined_identifier = f'{doc_collection}_{doc_id}' # agr_19
		doc_link = doc.get("itemLink") # /singleitem/collection/ryr/id/2479
		doc_url = f"https://digitalcollections.smu.edu/digital/collection/{doc.get('collectionAlias')}/id/{doc.get('itemId')}"
		doc_img_link = f"https://digitalcollections.smu.edu/digital/api/singleitem/image/{doc_collection}/{doc_id}/default.jpg"
		doc_title = clean_(text=doc.get("title"), sw=STOPWORDS)# doc.get("title")
		doc_description = doc.get("dcDescription")
		if (
			doc_type == "jp2"
			and "cover]" not in doc_title
			and doc_img_link 
			and (doc_img_link.endswith('.jpg') or doc_img_link.endswith('.png'))
		):
			pass # Valid entry; no action needed here
		else:
			doc_img_link = None
		row = {
			'id': doc_combined_identifier,
			'label': query,
			'title': doc_title,
			'description': doc.get("dcDescription"),
			'img_url': doc_img_link,
			'doc_url': doc_url,
			'label_title_description': query + " " + (doc_title or '') + " " + (doc_description or ''),
			'raw_doc_date': doc_date,
			'img_path': f"{os.path.join(args.dataset_dir, 'images', str(doc_combined_identifier) + '.jpg')}"
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(df.head(10))
	# Apply the function to the 'raw_doc_date' and 'doc_year' columns
	df['doc_date'] = df.apply(lambda row: get_doc_year(row['raw_doc_date']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[df['doc_date'].apply(lambda x: is_valid_date(date=x, start_date=START_DATE, end_date=END_DATE))]

	print(f"DF: {df.shape} {type(df)} Elapsed_t: {time.time()-df_st_time:.1f} sec")
	return df

def main():
	with open(os.path.join(parent_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		all_label_tags = [line.strip().lower() for line in file_]
	print(type(all_label_tags), len(all_label_tags))

	print(f"{len(all_label_tags)} Query phrases are being processed, please be patient...")
	dfs = []
	for qi, qv in enumerate(all_label_tags):
		print(f"\nQ[{qi+1}/{len(all_label_tags)}]: {qv}")
		query_all_hits = get_data(
			st_date=START_DATE,
			end_date=END_DATE,
			query=qv.lower()
		)
		if query_all_hits:
			qv_processed = re.sub(" ", "_", qv.lower())
			df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_{START_DATE}_{END_DATE}.gz")
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
	smu_df_merged_raw = pd.concat(dfs, ignore_index=True)
	print(f"<!> Replacing labels with super classes")
	json_file_path = os.path.join(parent_dir, 'misc', 'generalized_labels.json')
	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")

	print(f"pre-processing merged {type(smu_df_merged_raw)} {smu_df_merged_raw.shape}")
	smu_df_merged_raw['label'] = smu_df_merged_raw['label'].replace(replacement_dict)
	smu_df_merged_raw = smu_df_merged_raw.dropna(subset=['img_url']) # drop None firstDigitalObjectUrl
	smu_df_merged_raw = smu_df_merged_raw.drop_duplicates(subset=['img_url']) # drop duplicate firstDigitalObjectUrl

	print(f"Processed smu_df_merged_raw: {smu_df_merged_raw.shape}")
	print(smu_df_merged_raw.head(20))

	smu_df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	try:
		smu_df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	smu_df = get_synchronized_df_img(df=smu_df_merged_raw, image_dir=IMAGE_DIR, nw=args.num_workers)
	query_counts = smu_df['label'].value_counts()

	plt.figure(figsize=(15, 10))
	query_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name} Query Frequency (total: {query_counts.shape})')
	plt.xlabel('label')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUTs_DIR, f"query_x_{query_counts.shape[0]}_freq.png"))

	smu_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	try:
		smu_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	yr_distro_fpth = os.path.join(OUTPUTs_DIR, f"year_distribution_{dataset_name}_{START_DATE}_{END_DATE}_nIMGs_{smu_df.shape[0]}.png")
	plot_year_distribution(
		df=smu_df,
		start_date=START_DATE,
		end_date=END_DATE,
		dname=dataset_name,
		fpth=yr_distro_fpth,
		BINs=50,
	)

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
	query = "museum"
	# "https://digitalcollections.smu.edu/digital/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/president!president!18900101-19601231/field/title!descri!date/mode/any!all!exact/conn/and!and!and/maxRecords/200"
	# query_url = str(f"https://digitalcollections.smu.edu/digital/api/search/searchterm/image!{query}!{START_DATE}-{END_DATE}/field/type!all!date/mode/exact!all!exact/conn/and!and!and/maxRecords/200/page/7")
	query_url = f"https://digitalcollections.smu.edu/digital/api/search/collection/apnd!aaf!outler!ald!alv!han!wsw!lav!bml!other!civ!cooke!pwl!mbc!eaa!wlrd!fjd!gcp!gcd!white!wlg!kil!jcc!jhv!mcs!UKMeth!mex!ngc!nam!ptr!rwy!stn!ryr!rdoh!tex!bridhist!haws!wes!wrl/searchterm/{query}!{query}!{START_DATE}-{END_DATE}/field/title!descri!date/mode/any!all!exact/conn/and!and!and"
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
					itm_link = f"https://digitalcollections.smu.edu/digital/collection/{itm_collection}/id/{itm_identifier}" #item.get("itemLink")
					itm_api_link = f"https://digitalcollections.smu.edu/digital/api/collections/{itm_collection}/items/{itm_identifier}/false"
					itm_img_link = f"https://digitalcollections.smu.edu/digital/api/singleitem/image/{itm_collection}/{itm_identifier}/default.jpg"
					itm_title = item.get("title")
					# itm_description = 
					itm_date = item.get("metadataFields")[3].get("value")
					print(itm_title)
					print(itm_link)
					print(itm_img_link)
					print(itm_date)
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