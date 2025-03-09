import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from misc.utils import *

dataset_name = "SA_KUVA_WWII".upper()
parser = argparse.ArgumentParser(description=f"{dataset_name} Dataset Collector")
parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', type=str, default="1939-09-01", help='Start Date')
parser.add_argument('--end_date', type=str, default="1945-09-02", help='End Date')
parser.add_argument('--num_workers', type=int, default=10, help='Number of CPUs')
parser.add_argument('--img_mean_std', type=bool, default=False, help='Image mean & std')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()
print(args)

# run in local laptop:
# $ python data_collector.py --dataset_dir $PWD --start_date 1939-09-01 --end_date 1945-09-02
# $ nohup python -u data_collector.py --dataset_dir $PWD --start_date 1939-09-01 --end_date 1945-09-02 > logs/SA_KUVA_WW2_img_dl.out &

# run in Pouta:
# $ python data_collector.py --dataset_dir /media/volume/ImACCESS/NA_DATASETs --start_date --start_date 1939-09-01 --end_date 1945-09-02 # WW2 (with threshold)
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/NA_DATASETs --start_date --start_date 1939-09-01 --end_date 1945-09-02 --num_workers 55 --img_mean_std True > /media/volume/trash/ImACCESS/SA_KUVA_WW2_img_dl.out &

HOME: str = os.getenv('HOME') # echo $HOME
USER: str = os.getenv('USER') # echo $USER
finna_api_base_url: str = "https://api.finna.fi/v1/search"
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
IMAGE_DIR = os.path.join(DATASET_DIRECTORY, "images")

os.makedirs(os.path.join(DATASET_DIRECTORY, "hits"), exist_ok=True)
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")

os.makedirs(os.path.join(DATASET_DIRECTORY, "outputs"), exist_ok=True)
OUTPUTs_DIR = os.path.join(DATASET_DIRECTORY, "outputs")

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.pkl")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.pkl")

def get_data(st_date: str="1914-01-01", end_date: str="1914-01-02", label: str="lentokone"):
	t0 = time.time()
	label_processed = re.sub(" ", "_", label)
	label_all_hits_fpth = os.path.join(HITs_DIR, f"results_query_{label_processed}_{st_date}_{end_date}.gz")
	try:
		label_all_hits = load_pickle(fpath=label_all_hits_fpth)
	except Exception as e:
		print(f"{e}")
		print(f"Collecting all docs of National Archive for label: « {label} » ... it might take a while..")
		headers = {
			'Content-type': 'application/json',
			'Accept': 'application/json; text/plain; */*',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
			'Pragma': 'no-cache',
		}
		params = {
			'filter[]': [
				'~format_ext_str_mv:"0/Image/"',
				'~building:"1/SA-kuva/SA-kuva/"',
				'free_online_boolean:"1"',
			],
			'lookfor': label,
			'type': 'AllFields',
			'limit': 100,
		}
		label_all_hits = []
		page = 1
		while True:
			loop_st = time.time()
			params["page"] = page
			response = requests.get(
				finna_api_base_url,
				params=params,
				headers=headers,
			)
			if response.status_code == 200:
				data = response.json()
				if "records" in data:
					hits = data.get('records')
					# print(hits)
					# print(len(hits), type(hits))
					# print(hits[0].keys())
					# print(json.dumps(hits[0], indent=2, ensure_ascii=False))
					label_all_hits.extend(hits)
					total_hits = data.get('resultCount')
					print(f"Page: {page}:\tFound: {len(hits)} {type(hits)}\t{len(label_all_hits)}/{total_hits}\tin: {time.time()-loop_st:.1f} sec")
					if len(label_all_hits) >= total_hits:
						break
					page += 1
				else:
					print(f"no hits found, out!")
					break
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
				print(f"\t> XXXX found '{term}' => skipping! XXXX <")
				return False
	return True

def get_dframe(label: str="label", docs: List=[Dict]) -> pd.DataFrame:
	print(f"Analyzing {len(docs)} {type(docs)} document(s) for label: « {label} » might take a while...")
	df_st_time = time.time()
	data = []
	for doc in docs:
		# print(type(doc), list(doc.keys()))
		doc_title = doc.get("title") #clean_(text=record.get('title'), sw=STOPWORDS)
		doc_description = None
		sa_kuva_identifier = doc.get("id")
		doc_year = doc.get("year")
		img_url = f"https://www.finna.fi/Cover/Show?source=Solr&id={sa_kuva_identifier}"
		if (
			img_url 
		):
			pass # Valid entry; no action needed here
		else:
			img_url = None
		row = {
			'id': re.search(r'\d+', sa_kuva_identifier).group(), # 82080
			# 'id': sa_kuva_identifier, # sa-kuva.sa-kuva-82080
			'label': label,
			'title': doc_title,
			'description': doc_description,
			'img_url': img_url,
			'label_title_description': label + " " + (doc_title or '') + " " + (doc_description or ''),
			'doc_date': doc_year,
			'doc_url': f"https://www.finna.fi/Record/{sa_kuva_identifier}",
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")
	return df

def main():
	with open(os.path.join(parent_dir, 'misc', 'query_labels_FI.txt'), 'r') as file_:
		all_label_tags = [line.strip().lower() for line in file_]
	print(type(all_label_tags), len(all_label_tags))
	all_label_tags = all_label_tags[:136]
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
			print(df.head(10))
			dfs.append(df)

	print(f"Concatinating {len(dfs)} dfs...")
	# print(dfs[0])
	sa_kuva_df_merged_raw = pd.concat(dfs, ignore_index=True)

	json_file_path = os.path.join(parent_dir, 'misc', 'generalized_labels.json')

	if os.path.exists(json_file_path):
		with open(json_file_path, 'r') as file_:
			replacement_dict = json.load(file_)
	else:
		print(f"Error: {json_file_path} does not exist.")

	print(f"pre-processing merged {type(sa_kuva_df_merged_raw)} {sa_kuva_df_merged_raw.shape}")
	# sa_kuva_df_merged_raw['label'] = sa_kuva_df_merged_raw['label'].replace(replacement_dict) # TODO: Finnish adjustment required!
	sa_kuva_df_merged_raw = sa_kuva_df_merged_raw.dropna(subset=['img_url']) # drop None img_url
	sa_kuva_df_merged_raw = sa_kuva_df_merged_raw.drop_duplicates(subset=['img_url'], keep="first", ignore_index=True) # drop duplicate img_url

	print(f"Processed sa_kuva_df_merged_raw: {sa_kuva_df_merged_raw.shape}")
	print(sa_kuva_df_merged_raw.head(20))

	sa_kuva_df_merged_raw.to_csv(os.path.join(DATASET_DIRECTORY, "metadata_raw.csv"), index=False)
	try:
		sa_kuva_df_merged_raw.to_excel(os.path.join(DATASET_DIRECTORY, "metadata_raw.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	sa_kuva_df = get_synchronized_df_img(df=sa_kuva_df_merged_raw, image_dir=IMAGE_DIR, nw=args.num_workers)
	get_stratified_split(
		df=sa_kuva_df,
		val_split_pct=0.35,
		figure_size=(12, 6),
		dpi=250,
		result_dir=DATASET_DIRECTORY,
		dname=dataset_name,
	)

	label_counts = sa_kuva_df['label'].value_counts()
	print(label_counts.tail(25))

	plt.figure(figsize=(20, 13))
	label_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name}: Query Frequency (total: {label_counts.shape}) {START_DATE} - {END_DATE}')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUTs_DIR, f"all_query_labels_x_{label_counts.shape[0]}_freq.png"))

	sa_kuva_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	try:
		sa_kuva_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	yr_distro_fpth = os.path.join(OUTPUTs_DIR, f"year_distribution_{dataset_name}_{START_DATE}_{END_DATE}_nIMGs_{sa_kuva_df.shape[0]}.png")
	plot_year_distribution(
		df=sa_kuva_df,
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
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(dir=os.path.join(DATASET_DIRECTORY, "images"), num_workers=args.num_workers)
			save_pickle(pkl=img_rgb_mean, fname=img_rgb_mean_fpth)
			save_pickle(pkl=img_rgb_std, fname=img_rgb_std_fpth)
		print(f"RGB: Mean: {img_rgb_mean} | Std: {img_rgb_std}")

def test():
	finna_api_base_url: str = "https://api.finna.fi/v1/search"
	query = "lentokone"
	# params taken from payload of JSON URL
	params = {
		'filter[]': [
			'~format_ext_str_mv:"0/Image/"',
			'~building:"1/SA-kuva/SA-kuva/"',
			'free_online_boolean:"1"',
		],
		'lookfor': query,
		'type': 'AllFields',
		'limit': 100,
	}
	response = requests.get(finna_api_base_url, params=params)
	if response.status_code == 200: # status check: 200
		data = response.json() # Parse the JSON response
		if 'records' in data:
			hits = data['records']
			tot_hits = data['resultCount']
			hits_status = data['status']
			print(tot_hits, len(hits), hits_status)
			print(json.dumps(hits[48], indent=2))
		else:
			print("No 'records' found in the response.")
	else:
		print(f"Request failed with status code {response.status_code}")

if __name__ == '__main__':
	print(
		f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
		.center(160, " ")
	)
	START_EXECUTION_TIME = time.time()
	get_ip_info()
	main()
	# test()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)