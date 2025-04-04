import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from misc.utils import *
from misc.visualize import *

dataset_name: str = "europeana".upper()
parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1900-01-01", help='Dataset DIR')
parser.add_argument('--end_date', '-edt', type=str, default="1970-12-31", help='Dataset DIR')
parser.add_argument('--num_workers', '-nw', type=int, default=16, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std') # if given => True (ex. --img_mean_std)

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print_args_table(args=args, parser=parser)

# run in local laptop:
# $ python data_collector.py -ddir $HOME/datasets/WW_DATASETs
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 8 --img_mean_std > logs/europeana_img_dl.out &

# run in Pouta:
# $ python data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1960-12-31
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 -nw 40 --img_mean_std > /media/volume/ImACCESS/trash/europeana_dl.out &

START_DATE = args.start_date
END_DATE = args.end_date

meaningless_words_fpth = os.path.join(parent_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
# print(STOPWORDS, type(STOPWORDS))
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

DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")
os.makedirs(DATASET_DIRECTORY, exist_ok=True)

IMAGE_DIR = os.path.join(DATASET_DIRECTORY, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")
os.makedirs(HITs_DIR, exist_ok=True)

OUTPUTs_DIR = os.path.join(DATASET_DIRECTORY, "outputs")
os.makedirs(OUTPUTs_DIR, exist_ok=True)

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

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
		# print(type(doc.get("title")), doc.get("title"))
		# title = doc.get("title")
		europeana_id = doc.get("id")
		doc_id = re.sub("/", "SLASH", europeana_id)
		# print(type(doc.get("title")), len(doc.get("title")))
		doc_title_list = doc.get("title") # ["title1", "title2", "title3", ...]
		doc_description_list = doc.get("dcDescription" )# ["desc1", "desc2", "desc3", ...]
		doc_title = clean_(text=' '.join(doc_title_list), sw=STOPWORDS) if doc_title_list else None
		doc_description = clean_(text=" ".join(doc_description_list), sw=STOPWORDS) if doc_description_list else None
		# doc_title = ' '.join(doc_title_list) if doc_title_list else None
		# doc_description = " ".join(doc_description_list) if doc_description_list else None
		image_url = doc.get("edmIsShownBy")[0]
		# print(doc.get("edmTimespanLabel"), doc.get("year"), europeana_id, image_url, doc.get("link"))
		# print(int(doc.get("year")[0]) < 1946)

		pDate = doc.get("edmTimespanLabel")[0].get("def") if (doc.get("edmTimespanLabel") and doc.get("edmTimespanLabel")[0].get("def")) else None
		raw_doc_date = doc.get("edmTimespanLabel")
		doc_year = doc.get("year")[0] if (doc.get("year") and doc.get("year")[0]) else None
		doc_url = f"https://www.europeana.eu/en/item{europeana_id}" # doc.get("guid")
		
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
			'label': label,
			'title': doc_title,
			'description': doc_description,
			'img_url': image_url,
			'label_title_description': label + " " + (doc_title or '') + " " + (doc_description or ''),
			'raw_doc_date': raw_doc_date,
			'doc_year': doc_year,
			# 'my_date': pDate,
			"doc_url": doc_url,
			'img_path': f"{os.path.join(IMAGE_DIR, str(doc_id) + '.jpg')}"
		}
		data.append(row)
	df = pd.DataFrame(data)

	# Apply the function to the 'raw_doc_date' and 'doc_year' columns
	df['doc_date'] = df.apply(lambda row: get_europeana_date_or_year(row['raw_doc_date'], row['doc_year']), axis=1)

	# Filter the DataFrame based on the validity check
	df = df[df['doc_date'].apply(lambda x: is_valid_date(date=x, start_date=START_DATE, end_date=END_DATE))]

	# df = df.drop(['raw_doc_date', 'doc_year'], axis=1)
	print(f"DF: {df.shape} {type(df)} Elapsed_t: {time.time()-df_st_time:.1f} sec")
	return df

@measure_execution_time
def main():
	with open(os.path.join(parent_dir, 'misc', 'query_labels.txt'), 'r') as file_:
		all_label_tags = [line.strip().lower() for line in file_]
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
			# print(df)
			# print(df.head())
			dfs.append(df)

	print(f"Concatinating {len(dfs)} dfs...")
	europeana_df_merged_raw = pd.concat(dfs, ignore_index=True)
	json_file_path = os.path.join(parent_dir, 'misc', 'super_labels.json')
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

	europeana_df = get_synchronized_df_img(
		df=europeana_df_merged_raw,
		image_dir=IMAGE_DIR,
		nw=args.num_workers,
	)

	label_dirstribution_fname = os.path.join(OUTPUTs_DIR, f"{dataset_name}_label_distribution_{europeana_df.shape[0]}_x_{europeana_df.shape[1]}.png")
	plot_label_distribution(
		df=europeana_df,
		dname=dataset_name,
		fpth=label_dirstribution_fname,
		)
	
	europeana_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	get_stratified_split(
		df=europeana_df,
		val_split_pct=0.35,
		figure_size=(12, 6),
		dpi=250,
		result_dir=DATASET_DIRECTORY,
		dname=dataset_name,
	)
	try:
		europeana_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	yr_distro_fpth = os.path.join(OUTPUTs_DIR, f"{dataset_name}_year_distribution_{europeana_df.shape[0]}_samples.png")
	plot_year_distribution(
		df=europeana_df,
		dname=dataset_name,
		fpth=yr_distro_fpth,
		BINs=args.historgram_bin,
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

def test():
	query = "naval commander"
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
			# Output the extracted items (metadata and descriptions of the images)
			# for item_idx, item in enumerate(items):
				# print(f"idx: {item_idx}: {list(item.keys())}")
				# print(item.get("id"), item.get("title"), item.get("type"))
				# print(item.get("edmConceptLabel"))
				# print(item.get("edmConceptPrefLabelLangAware"))
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

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	# test()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))