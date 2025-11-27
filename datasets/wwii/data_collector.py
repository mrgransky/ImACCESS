import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)
from misc.utils import *
from misc.visualize import *

# how to run in local:
# $ nohup python -u data_collector.py -ddir $HOME/datasets/WW_DATASETs -nw 8 --img_mean_std > logs/wwii_image_download.out &

# run in Pouta:
# $ python data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1960-12-31
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -nw 24 --img_mean_std > /media/volume/ImACCESS/trash/wwii_data_collection.out &

dataset_name = "WWII".upper()
parser = argparse.ArgumentParser(description=f"{dataset_name} ARCHIVE data colletion")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1939-09-01", help='Start Date')
parser.add_argument('--end_date', '-edt', type=str, default="1945-09-02", help='End Date')
parser.add_argument('--num_workers', '-nw', type=int, default=8, help='Number of CPUs')
parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
parser.add_argument('--historgram_bin', '-hb', type=int, default=60, help='Histogram Bins')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
parser.add_argument('--val_split_pct', '-vsp', type=float, default=0.35, help='Validation Split Percentage')

args, unknown = parser.parse_known_args()
args.dataset_dir = os.path.normpath(args.dataset_dir)
print_args_table(args=args, parser=parser)

meaningless_words_fpth = os.path.join(project_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
# print(STOPWORDS, type(STOPWORDS))

START_DATE = args.start_date
END_DATE = args.end_date

os.makedirs(os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}"), exist_ok=True)
DATASET_DIRECTORY = os.path.join(args.dataset_dir, f"{dataset_name}_{START_DATE}_{END_DATE}")

os.makedirs(os.path.join(DATASET_DIRECTORY, "images"), exist_ok=True)
IMAGE_DIR = os.path.join(DATASET_DIRECTORY, "images")

os.makedirs(os.path.join(DATASET_DIRECTORY, "hits"), exist_ok=True)
HITs_DIR = os.path.join(DATASET_DIRECTORY, "hits")

os.makedirs(os.path.join(DATASET_DIRECTORY, "outputs"), exist_ok=True)
OUTPUT_DIRECTORY = os.path.join(DATASET_DIRECTORY, "outputs")

img_rgb_mean_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth:str = os.path.join(DATASET_DIRECTORY, "img_rgb_std.gz")

FIGURE_SIZE = (12, 9)
DPI = 250
# Define regex pattern for WWII years: 1939–1945
YEAR_PATTERN = re.compile(r'\b(19[3][9]|[1][9]4[0-5])\b')

def extract_year(text):
	match = YEAR_PATTERN.search(str(text))
	return match.group(1) if match else None

def extract_url_info(url:str)-> Dict:
	parsed_url = urllib.parse.urlparse(url)
	base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/gallery" # Extract the base URL
	path_components = parsed_url.path.strip('/').split('/') # Split the path into components		

	# Extract country, main_label, and type
	country = path_components[1] if len(path_components) > 1 else None
	main_label = path_components[2] if len(path_components) > 2 else None
	type_ = path_components[3] if len(path_components) > 3 else None

	# Decode URL-encoded characters (if any)
	if main_label:
		main_label = urllib.parse.unquote(main_label)
		main_label = re.sub(r'[^a-zA-Z\s]', ' ', main_label) # Remove special characters and digits
		main_label = re.sub(r'\s+', ' ', main_label)  # Remove extra whitespace

	if type_:
		type_ = urllib.parse.unquote(type_)

	return {
		"base_url": base_url,
		"country": country,
		"main_label": main_label,
		"type": type_
	}

def get_dframe(
		doc_idx: int,
		doc_url: str, 
		user_query: str,
	) -> pd.DataFrame:

	print(f">> Extracting DF for user_query[{doc_idx}]: « {user_query} » from {doc_url}")
	
	content_to_hash = f"{doc_url}_{START_DATE}_{END_DATE}"
	hash_digest = hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
	df_fpth = os.path.join(HITs_DIR, f"df_{hash_digest}.gz")

	if os.path.exists(df_fpth):
		df = load_pickle(fpath=df_fpth)
		return df

	doc_url_info = extract_url_info(doc_url)
	print(json.dumps(doc_url_info, indent=4, ensure_ascii=False))
	try:
		response = requests.get(doc_url)
		response.raise_for_status()
	except requests.RequestException as e:
		print(f"<!> Error sending GET request: {e}")
		return None

	df_st_time = time.time()
	try:
		response = requests.get(doc_url)
		response.raise_for_status()
		soup = BeautifulSoup(response.text, 'html.parser')
		hits = soup.find_all('img', class_='attachment-thumbnail')
		descriptions = soup.find_all("p")
		header = soup.find('h2', class_="entry-title").text
	except Exception as e:
		print(f"Failed to retrieve doc_url or parse content: {e}")
		return None

	# parts = [
	# 	header,
	# 	# (doc_url_info.get("country") or "").strip(),
	# 	(doc_url_info.get("main_label") or "").strip(),
	# 	# (doc_url_info.get("type") or "").strip(),
	# ]
	# header = " ".join(filter(None, parts))

	print(f"Doc header:\n{header}")

	# Extract caption as doc_description
	caption_element = soup.find('div', class_='entry-caption')
	if caption_element:
		doc_description = caption_element.get_text(strip=True)
		doc_description = re.sub(r'\s+', ' ', doc_description).strip()
	else:
		doc_description = ""

	if doc_description.lower() and header.lower() not in doc_description.lower():
		doc_description = header + " " + doc_description
	elif not doc_description.strip():
		doc_description = header

	print(f"\nDoc Description:\n{doc_description}\n")

	caption_map = {}
	for p in soup.find_all('p', class_='wp-caption-text gallery-caption'):
		cid = p.get('id')
		if cid:
			caption_text = p.get_text(strip=True)
			if caption_text:
				caption_map[cid] = caption_text	
	print(f"{len(caption_map)} Caption Map(s):\n{json.dumps(caption_map, indent=4, ensure_ascii=False)}")
	print(f"Found {len(hits)} Document(s) => Extracting information [might take a while]")
	data = []
	for idoc, vdoc in enumerate(hits):
		print(idoc)
		print(vdoc)
		img_tag = vdoc
		img_url = img_tag.get('data-src')
		if not img_url:
			continue
		parent_a = img_tag.find_parent('a')
		print(f"doc_url: {parent_a.get('href')}")

		try:
			doc_doc_url = parent_a.get('href')
			response = requests.get(doc_doc_url)
			response.raise_for_status()
			soup = BeautifulSoup(response.text, 'html.parser')
			header_el = soup.find('h2', class_="entry-title")
			caption_el = soup.find('div', class_='entry-caption')
			local_header = header_el.get_text(strip=True) if header_el else ""
			local_caption = caption_el.get_text(strip=True) if caption_el else ""
		except Exception as e:
			print(f"Failed to extract doc_url: {e}")
			continue
		doc_cap_x0 = local_caption if local_caption else None
		doc_header_x0 = local_header if local_header else None
		print(f"doc_header (0th try): {doc_header_x0}")
		print(f"doc_caption(0th try): {doc_cap_x0}")
		print("-"*50)

		# join doc_title_x0 and doc_cap_x0 if they're not the same
		if doc_header_x0 and doc_cap_x0 and doc_header_x0 != doc_cap_x0:
			doc_title_x0 = ". ".join(filter(None, [doc_header_x0, doc_cap_x0]))
		else:
			doc_title_x0 = doc_header_x0 or doc_cap_x0
		print(f"doc_title(0th try): {doc_title_x0}")

		doc_title_x1 = img_tag.get("alt")
		doc_title_x1 = doc_title_x1 if doc_title_x1 != "Folder Icon" else None
		print(f"doc_title(1st try): {doc_title_x1}")
		
		doc_title_x2 = None
		aria_id = img_tag.get("aria-describedby")
		if aria_id:
			caption_title = caption_map.get(aria_id)
			if caption_title:
				doc_title_x2 = caption_title

		print(f"doc_title(2nd try): {doc_title_x2}")

		# Select the most complete title or combine if they're truly different
		titles = [doc_title_x0, doc_cap_x0, doc_title_x1, doc_title_x2]
		valid_titles = [t for t in titles if t]
		
		if len(valid_titles) == 1:
			doc_title = valid_titles[0]
		elif len(valid_titles) > 1:
			# Find the longest title (most descriptive)
			doc_title = max(valid_titles, key=len)
		else:
			doc_title = None

		print(f"doc_title(final)  : {doc_title}")
		img_url = img_url.replace("_cache/", "")
		img_url = re.sub(r'-\d+x\d+\.jpg$', '.jpg', img_url) # Remove the thumbnail size from the end of the URL
		filename = os.path.basename(img_url)
		img_fpath = os.path.join(IMAGE_DIR, filename)
		specific_doc_url = urllib.parse.urljoin(doc_url, parent_a.get('href')) if parent_a and parent_a.get('href') else doc_url

		# Attempt to extract date from multiple sources
		date_sources = [
			doc_title,
			specific_doc_url,
			img_url,
			filename,
			doc_description,  # assuming it's still accessible here
		]
		
		# Try extracting year from all sources
		extracted_year = None
		for src in date_sources:
			if src:
				year = extract_year(src)
				if year:
					extracted_year = year
					break

		if not os.path.exists(img_fpath):
			try:
				img_response = requests.get(img_url)
				img_response.raise_for_status()
				with open(img_fpath, 'wb') as f:
					f.write(img_response.content)
			except Exception as e:
				print(f"Failed to download {img_url}: {e}")
				continue

		raw_enriched_document_description = ". ".join(filter(None, [doc_title, doc_description])).strip()
		print(f"\nraw_enriched_document_description:\n{raw_enriched_document_description}")

		enriched_document_description = basic_clean(txt=raw_enriched_document_description)
		print(f"\nenriched_document_description:\n{enriched_document_description}\n")


		row = {
			'id': filename,
			'date': extracted_year,
			'doc_url': specific_doc_url,
			'img_url': img_url,
			'title': doc_title,
			'description': doc_description,
			'country': doc_url_info.get("country"),
			'user_query': [user_query] if user_query else None,
			'label': user_query if user_query else None,
			'img_path': img_fpath,
			'enriched_document_description': enriched_document_description,
		}
		data.append(row)

	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")
	save_pickle(pkl=df, fname=df_fpth)
	return df

@measure_execution_time
def main():
	base_url = "https://www.worldwarphotos.info/gallery"
	URLs = { # key: url : val: user_query
		f"{base_url}/usa/pacific/biak/": None,
		f"{base_url}/usa/pacific/bougainville/": None,
		f"{base_url}/usa/pacific/gloucester/": None,
		f"{base_url}/usa/pacific/eniwetok/": None,
		f"{base_url}/usa/pacific/guadalcanal/": None,
		f"{base_url}/usa/pacific/guam/": None,
		f"{base_url}/usa/pacific/iwo-jima/": None,
		f"{base_url}/usa/pacific/iwo-jima2/": None,
		f"{base_url}/usa/pacific/kwajalein/": None,
		f"{base_url}/usa/pacific/makin/": None,
		f"{base_url}/usa/pacific/new-guinea/": None,
		f"{base_url}/usa/pacific/okinawa/": None,
		f"{base_url}/usa/pacific/peleliu/": None,
		f"{base_url}/usa/pacific/philippines/": None,
		f"{base_url}/usa/pacific/saipan/": None,
		f"{base_url}/usa/pacific/tarawa/": None,
		f"{base_url}/usa/pacific/tinian/": None,
		f"{base_url}/usa/aircrafts-2-3/a-17/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-18/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-19/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-20-havoc-boston/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-20/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a20/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-26/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/a-36/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-17/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-17-flying-fortress/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-17b/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-17g/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b17/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-17raf/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-18/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-23/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-24/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-24-liberator/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-24-bomber/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b24/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-25-mitchell/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-25/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b25/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-26-marauder/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-29/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-32-dominator/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/bt/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-106/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-109/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-46/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-47/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c47/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-54/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-69/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-73/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-76/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/c-87/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f2a/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f4f-wildcat/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f4f/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f4u-corsair/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f4u/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f6f-hellcat/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f7f/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/f8f/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/fr1/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/lodestar/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/o-38/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/o-46/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/o-47/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/o-52/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/os2u/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/ose/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-26/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-35/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-36/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-38-lightning/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-38/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-39/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-39-2/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-40-warhawk/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-40raf/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-40/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-43/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-47/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-47-thunderbolt/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p47/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-47d/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-51-mustang/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-51/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p51-raf/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-59-airacomet/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-61/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-63/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-66/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-70/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-75/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/p-80-shooting-star/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pb2y/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pb4y/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pbm/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pby/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pby5/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pq-14/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pv/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/r-4/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sb2a/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sb2c/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sb2u-vindicator/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sbc/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sbd/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sbd1/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/sc-seahawk/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/so3c/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/soc/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/tbd/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/tbf/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/tbm/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/tbu-tby/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/uc-61/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xa-21/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xa-38/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-15/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/b-19/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xb-38/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xb-39/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xb-42/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xb-43/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xf-12/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xf8b/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xfl/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-42/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-46/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-54/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-55/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-56/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-58/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xp-83/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/xpb2m/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/pbb/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/tb2f/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/yb-40/": "aircraft",
		f"{base_url}/usa/aircrafts-2-3/yfm/": "aircraft",
		f"{base_url}/usa/armoured-vehicles-2/m12/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m2-half-track/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m3_halftrack-2/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m3_scout/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m31/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m32/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m7_priest/": "armored fighting vehicle",
		f"{base_url}/usa/armoured-vehicles-2/m8_greyhound/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m1/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m10-wolverine/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m18-hellcat/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m2/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m2-medium/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m24/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m26/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m3_lee/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m3_stuart/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m3_m5/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m36-jackson/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m4_sherman/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m4-sherman-tank/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/sherman/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/sherman-tank/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m6-tank/": "armored fighting vehicle",
		f"{base_url}/usa/tanks/m8/": "armored fighting vehicle",
		f"{base_url}/usa/us-navy/": "naval forces",
		f"{base_url}/usa/vehicles/g506/": "military vehicle",
		f"{base_url}/usa/vehicles/m29/": "military vehicle",
		f"{base_url}/italy/spg2/75-18/" : "armored fighting vehicle", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
		f"{base_url}/italy/spg2/l40/" : "armored fighting vehicle", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
		f"{base_url}/france/tanks-france/" : "armored fighting vehicle", # French Tanks of World War II
		f"{base_url}/france/normandy-1944/": "normandy invasion", # Invasion of Normandy 1944 photo gallery
		f"{base_url}/japan/aircrafts/b7a/": "aircraft", #
		f"{base_url}/japan/aircrafts/d3a/": "aircraft", #
		f"{base_url}/japan/aircrafts/e13a/": "aircraft", #
		f"{base_url}/japan/aircrafts/e16a": "aircraft", #
		f"{base_url}/japan/aircrafts/m6a/": "aircraft", #
		f"{base_url}/japan/aircrafts/h8k/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-100/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-45/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-48/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-60/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-61-hien/": "aircraft", #
		f"{base_url}/japan/aircrafts/q1w/": "aircraft", #
		f"{base_url}/japan/aircrafts/l2d/": "aircraft", #
		f"{base_url}/japan/aircrafts/a5m/": "aircraft", #
		f"{base_url}/japan/aircrafts/a6m-zero/": "aircraft", #
		f"{base_url}/japan/aircrafts/g3m/": "aircraft", #
		f"{base_url}/japan/aircrafts/g4m/": "aircraft", #
		f"{base_url}/japan/aircrafts/j2m-raiden/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-21/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-46/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-57/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-67/": "aircraft", #
		f"{base_url}/japan/aircrafts/a2n/": "aircraft", #
		f"{base_url}/japan/aircrafts/b5n/": "aircraft", #
		f"{base_url}/japan/aircrafts/b6n/": "aircraft", #
		f"{base_url}/japan/aircrafts/c6n/": "aircraft", #
		f"{base_url}/japan/aircrafts/g5n/": "aircraft", #
		f"{base_url}/japan/aircrafts/g8n/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-115/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-43/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-44/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-84-hayate/": "aircraft", #
		f"{base_url}/japan/aircrafts/ki-54/": "aircraft", #
		f"{base_url}/japan/aircrafts/wrecks/": "wreck", #
		f"{base_url}/japan/aircrafts/d4y/": "aircraft", #
		f"{base_url}/japan/aircrafts/yokosuka_mxy7_ohka/": "aircraft", #
		f"{base_url}/japan/aircrafts/p1y/": "aircraft",
		f"{base_url}/japan/ijn/midget/": "naval forces",
		f"{base_url}/japan/japanese-tanks/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/cruiser-mk-iii-a13-mk-i-cruiser-mk-iv-a13-mk-ii/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/challenger/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/churchill-a22/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/comet/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/covenanter/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/a9-tank/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/cruiser-mk-ii-a10/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/crusader-tank/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/vickers/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/matilda-i-a11-tank/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/matilda-ii-a12/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/matilda-a12/": "armored fighting vehicle",
		f"{base_url}/uk/british-tanks/tetrarch/": "armored fighting vehicle",
		f"{base_url}/uk/armoured-vehicles/aec_dorchester/": "armored fighting vehicle",
		f"{base_url}/uk/armoured-vehicles/humber/": "armored fighting vehicle",
		f"{base_url}/uk/armoured-vehicles/marmon_herrington_-armoured_car/": "armored fighting vehicle",
		f"{base_url}/uk/armoured-vehicles/universal-carrier-bren-gun-carrier/": "armored fighting vehicle",
		f"{base_url}/uk/raf/aw23/": "aircraft",
		f"{base_url}/uk/raf/albacore/": "aircraft",
		f"{base_url}/uk/raf/baltimore/": "aircraft",
		f"{base_url}/uk/raf/barracuda/": "aircraft",
		f"{base_url}/uk/raf/fairey-battle/": "aircraft",
		f"{base_url}/uk/raf/beaufighter/": "aircraft",
		f"{base_url}/uk/raf/beau/": "aircraft",
		f"{base_url}/uk/raf/beaufort/": "aircraft",
		f"{base_url}/uk/raf/blenheim1/": "aircraft",
		f"{base_url}/uk/raf/blenheim/": "aircraft",
		f"{base_url}/uk/raf/brigand/": "aircraft",
		f"{base_url}/uk/raf/buckingham/": "aircraft",
		f"{base_url}/uk/raf/buckmaster/": "aircraft",
		f"{base_url}/uk/raf/defiant/": "aircraft",
		f"{base_url}/uk/raf/firebrand/": "aircraft",
		f"{base_url}/uk/raf/dh95/": "aircraft",
		f"{base_url}/uk/raf/halifax/": "aircraft",
		f"{base_url}/uk/raf/hamilcar/": "aircraft",
		f"{base_url}/uk/raf/harrow/": "aircraft",
		f"{base_url}/uk/raf/hudson/": "aircraft",
		f"{base_url}/uk/raf/hurricane/": "aircraft",
		f"{base_url}/uk/raf/hurricane2/": "aircraft",
		f"{base_url}/uk/raf/hurricane1/": "aircraft",
		f"{base_url}/uk/raf/lancaster/": "aircraft",
		f"{base_url}/uk/raf/lanc/": "aircraft",
		f"{base_url}/uk/raf/lincoln/": "aircraft",
		f"{base_url}/uk/raf/london/": "water based aircraft",
		f"{base_url}/uk/raf/lysander/": "aircraft",
		f"{base_url}/uk/raf/manchester/": "aircraft",
		f"{base_url}/uk/raf/maryland/": "aircraft",
		f"{base_url}/uk/raf/monitor/": "aircraft",
		f"{base_url}/uk/raf/mosquito/": "aircraft",
		f"{base_url}/uk/raf/mosquito2/": "aircraft",
		f"{base_url}/uk/raf/mossie/": "aircraft",
		f"{base_url}/uk/raf/roc/": "aircraft", # remove 2 flying boat
		f"{base_url}/uk/raf/seafang/": "aircraft",
		f"{base_url}/uk/raf/seafire/": "aircraft",
		f"{base_url}/uk/raf/shetland/": "aircraft",
		f"{base_url}/uk/raf/singapore/": "water based aircraft",
		f"{base_url}/uk/raf/skua/": "aircraft",
		f"{base_url}/uk/raf/spiteful/": "aircraft",
		f"{base_url}/uk/raf/spitfire/": "aircraft",
		f"{base_url}/uk/raf/spitfire2/": "aircraft",
		f"{base_url}/uk/raf/spitfire5/": "aircraft",
		f"{base_url}/uk/raf/spitfire9/": "aircraft",
		f"{base_url}/uk/raf/spit/": "aircraft",
		f"{base_url}/uk/raf/short-stirling/": "aircraft",
		f"{base_url}/uk/raf/stirling/": "aircraft",
		f"{base_url}/uk/raf/sunderland/": "aircraft",
		f"{base_url}/uk/raf/sund/": "aircraft",
		f"{base_url}/uk/raf/swordfish/": "water based aircraft",
		f"{base_url}/uk/raf/tempest/": "aircraft",
		f"{base_url}/uk/raf/tornado/": "aircraft",
		f"{base_url}/uk/raf/typhoon/": "aircraft",
		f"{base_url}/uk/raf/vickers432/": "aircraft",
		f"{base_url}/uk/raf/welkin/": "aircraft",
		f"{base_url}/uk/raf/wellington/": "aircraft",
		f"{base_url}/uk/raf/wellington1/": "aircraft",
		f"{base_url}/uk/raf/whirlwind/": "aircraft",
		f"{base_url}/uk/raf/whitley/": "aircraft",
		f"{base_url}/uk/raf/windsor/": "aircraft",
		f"{base_url}/ussr/vvs/ar-2/": "aircraft",
		f"{base_url}/ussr/vvs/i153/": "aircraft",
		f"{base_url}/ussr/vvs/il2-sturmovik/": "aircraft",
		f"{base_url}/ussr/vvs/il2/": "aircraft",
		f"{base_url}/ussr/vvs/lagg3/": "aircraft",
		f"{base_url}/ussr/vvs/li2/": "aircraft",
		f"{base_url}/ussr/vvs/mig/": "aircraft",
		f"{base_url}/ussr/vvs/pe8/": "aircraft",
		f"{base_url}/ussr/vvs/po-2/": "aircraft",
		f"{base_url}/ussr/vvs/r-10/": "aircraft",
		f"{base_url}/ussr/vvs/su-2/": "aircraft",
		f"{base_url}/ussr/armoured-vehicles-2-3/ba-10/": "armored fighting vehicle",
		f"{base_url}/ussr/armoured-vehicles-2-3/ba-20/": "armored fighting vehicle",
		f"{base_url}/ussr/armoured-vehicles-2-3/ba-27/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/isu-122/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/isu-152/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/su-100/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/su-122/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/su-152/": "armored fighting vehicle",
		f"{base_url}/ussr/spg/su-85/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/bt-2-bt-5-bt-7-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/is-2/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/kv-1/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/kv-1-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/kv-1s/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/kv-2/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-26/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-27/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-28-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-34/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-34_tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-34-85/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-35-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-37-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-38-tank/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-40/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-50/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-60/": "armored fighting vehicle",
		f"{base_url}/ussr/tanks-2/t-70/": "armored fighting vehicle",
		f"{base_url}/ussr/artillery_tractor/": "military vehicle",
		f"{base_url}/ussr/rkka/red_army/": "military personnel",
		f"{base_url}/germany/armored_vehicles/adgz/": "armored fighting vehicle",
		f"{base_url}/germany/armored_vehicles/kfz13/": "armored fighting vehicle",
		f"{base_url}/germany/armored_vehicles/sdkfz_221_222_223/": "armored fighting vehicle",
		f"{base_url}/germany/armored_vehicles/sdkfz_231_232_233/": "armored fighting vehicle",
		f"{base_url}/germany/armored_vehicles/sdkfz_247/": "armored fighting vehicle",
		f"{base_url}/germany/armored_vehicles/sdkfz_263/": "armored fighting vehicle",
		f"{base_url}/germany/kriegsmarine/": "naval forces",
		f"{base_url}/germany/german_army_soldiers/": "military personnel",
		f"{base_url}/germany/wehrmacht_trucks/bussing-nag/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/einheitsdiesel/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/faun/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/ford-lkw/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/ford-pkw/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/hanomag/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/henschel-33/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/horch_830/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/horch-901/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/krupp/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/krupp_protze_l2h_143/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/kubelwagen/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/mercedes-benz/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/opel_blitz/": "military vehicle",
		f"{base_url}/germany/wehrmacht_trucks/schwimmwagen/": "military vehicle",
		f"{base_url}/germany/aircrafts-2/ar-65/": "aircraft",
		f"{base_url}/germany/aircrafts-2/ar-66/": "aircraft",
		f"{base_url}/germany/aircrafts-2/arado_234/": "aircraft",
		f"{base_url}/germany/aircrafts-2/messerschmitt_bf_110/": "aircraft",
		f"{base_url}/germany/aircrafts-2/me_110/": "aircraft",
		f"{base_url}/germany/aircrafts-2/bf110/": "aircraft",
		f"{base_url}/germany/aircrafts-2/messerschmitt_bf109/": "aircraft",
		f"{base_url}/germany/aircrafts-2/messerschmitt-bf-109/": "aircraft",
		f"{base_url}/germany/aircrafts-2/bf109/": "aircraft",
		f"{base_url}/germany/aircrafts-2/bf_109/": "aircraft",
		f"{base_url}/germany/aircrafts-2/bv142/": "aircraft",
		f"{base_url}/germany/aircrafts-2/bv222/": "aircraft",
		f"{base_url}/germany/aircrafts-2/dornier_do_215/": "aircraft",
		f"{base_url}/germany/aircrafts-2/do217/": "aircraft",
		f"{base_url}/germany/aircrafts-2/do_335/": "aircraft",
		f"{base_url}/germany/aircrafts-2/dornier_do17/": "aircraft",
		f"{base_url}/germany/aircrafts-2/fw_189/": "aircraft",
		f"{base_url}/germany/aircrafts-2/fw190/": "aircraft",
		f"{base_url}/germany/aircrafts-2/focke_wulf_fw_190/": "aircraft",
		f"{base_url}/germany/aircrafts-2/fw190d/": "aircraft",
		f"{base_url}/germany/aircrafts-2/focke_wulf_fw200/": "aircraft",
		f"{base_url}/germany/aircrafts-2/he115/": "aircraft",
		f"{base_url}/germany/aircrafts-2/he116/": "aircraft",
		f"{base_url}/germany/aircrafts-2/heinkel_he111/": "aircraft",
		f"{base_url}/germany/aircrafts-2/he-112/": "aircraft",
		f"{base_url}/germany/aircrafts-2/he_162/": "aircraft",
		f"{base_url}/germany/aircrafts-2/he_177/": "aircraft",
		f"{base_url}/germany/aircrafts-2/hs123/": "aircraft",
		f"{base_url}/germany/aircrafts-2/hs_129/": "aircraft",
		f"{base_url}/germany/aircrafts-2/junkers-ju87-stuka/": "aircraft",
		f"{base_url}/germany/aircrafts-2/ju87/": "aircraft",
		f"{base_url}/germany/aircrafts-2/junkers_ju188/": "aircraft",
		f"{base_url}/germany/aircrafts-2/ju-290/": "aircraft",
		f"{base_url}/germany/aircrafts-2/junkers_ju_52/": "aircraft",
		f"{base_url}/germany/aircrafts-2/junkers_ju88/": "aircraft",
		f"{base_url}/germany/aircrafts-2/ju-88/": "aircraft",
		f"{base_url}/germany/aircrafts-2/ju-90/": "aircraft",
		f"{base_url}/germany/aircrafts-2/me261/": "aircraft",
		f"{base_url}/germany/aircrafts-2/me321/": "aircraft",
		f"{base_url}/germany/aircrafts-2/messerschmitt-me323-gigant/": "aircraft",
		f"{base_url}/germany/aircrafts-2/me-323-gigant/": "aircraft",
		f"{base_url}/germany/aircrafts-2/messerschmitt-me262/": "aircraft",
		f"{base_url}/germany/aircrafts-2/mistel/": "aircraft",
		f"{base_url}/germany/artillery/sturmpanzer_iii/": "artillery",
		f"{base_url}/germany/artillery/17-cm-k18/": "artillery",
		f"{base_url}/germany/artillery/flak-105/": "artillery",
		f"{base_url}/germany/artillery/flak-88/": "artillery",
		f"{base_url}/germany/artillery/flakpanzer-38/": "artillery",
		f"{base_url}/germany/artillery/grille/": "armored fighting vehicle",
		f"{base_url}/germany/artillery/hummel/": "artillery",
		f"{base_url}/germany/artillery/karl-gerat/": "artillery",
		f"{base_url}/germany/artillery/lorraine-schlepper/": "armored fighting vehicle",
		f"{base_url}/germany/artillery/pak43/": "artillery",
		f"{base_url}/germany/artillery/sig33b/": "armored fighting vehicle",
		f"{base_url}/germany/artillery/sig33-bison/": "armored fighting vehicle",
		f"{base_url}/germany/artillery/sturmpanzer_ii/": "armored fighting vehicle",
		f"{base_url}/germany/artillery/wespe/": "armored fighting vehicle",
		f"{base_url}/germany/armored-trains/": "armored fighting vehicle",
		f"{base_url}/germany/railway_gun/": "armored fighting vehicle",
		f"{base_url}/germany/units/afrika_korps/" : "military unit",
		f"{base_url}/germany/units/waffen-ss/" : "military unit",
		f"{base_url}/germany/units/grossdeutschland/" : "military unit",
		f"{base_url}/germany/units/sturmgeschutz_brigade_244/" : "military unit",
		}
	
	dfs_fname = os.path.join(HITs_DIR, f"{dataset_name}_{len(URLs)}_dfs.gz")
	
	try:
		dfs = load_pickle(fpath=dfs_fname,)
		print(f"Loaded {len(dfs)} dfs from {os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_dfs.gz')}")
	except Exception as e:
		print(f"<!> {e}")
		print(f"Scraping {len(URLs)} URLs...")
		dfs = [
			get_dframe(
				doc_idx=i, 
				doc_url=k, 
				user_query=v,
			) for i, (k, v) in enumerate(URLs.items())
		]
		dfs = [df for df in dfs if df is not None]
		save_pickle(pkl=dfs, fname=dfs_fname,)
		print(f"Saved {len(dfs)} dfs to {dfs_fname}")

	total_searched_labels = len(dfs)
	print(f"Concatinating {total_searched_labels} x {type(dfs[0])} dfs...")
	wwii_df = pd.concat(dfs, ignore_index=True)
	print(f"wwii_df {type(wwii_df)} {wwii_df.shape}, {list(wwii_df.columns)}")

	# 1: multi label:
	multi_label_synched_df = wwii_df.copy()
	multi_label_final_df = get_enriched_description(df=multi_label_synched_df)
	dfname_multi_label = "metadata_multi_label.csv"
	print(f"Saving {dfname_multi_label}...")
	multi_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, dfname_multi_label), index=False)
	try:
		multi_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, dfname_multi_label.replace('.csv', '.xlsx')), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	# 2: single label:
	# a) drop None from labels:
	print(f"Checking for None labels... {wwii_df['label'].isna().sum()} None labels / {wwii_df.shape[0]} total samples")
	single_label_final_df = wwii_df.dropna(subset=['label'])
	# b) save
	dfname_single_label = "metadata_single_label.csv"
	print(f"Saving {dfname_single_label}...")
	single_label_final_df.to_csv(os.path.join(DATASET_DIRECTORY, dfname_single_label), index=False)
	try:
		single_label_final_df.to_excel(os.path.join(DATASET_DIRECTORY, dfname_single_label.replace('.csv', '.xlsx')), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	unique_labels = single_label_final_df['label'].unique()
	print(f"{len(unique_labels)} Unique labels: {unique_labels}")

	print(single_label_final_df['label'].value_counts())

	label_dirstribution_fname = os.path.join(
		OUTPUT_DIRECTORY, 
		f"{dataset_name}_single_label_distribution_{wwii_df.shape[0]}_x_{unique_labels.shape[0]}.png"
	)
	plot_label_distribution(
		df=single_label_final_df,
		fpth=label_dirstribution_fname,
		FIGURE_SIZE=(14, 8),
		DPI=260,
		label_column='label',
	)

	# stratified splitting [single-label]:
	train_df, val_df = get_stratified_split(
		df=single_label_final_df, 
		val_split_pct=args.val_split_pct,
		label_col='label',
	)
	print(f"Train/val split for {dataset_name} dataset complete!")
	print(f"Full dataset: {wwii_df.shape} => Train: {train_df.shape} Validation: {val_df.shape}")

	train_df.to_csv(os.path.join(DATASET_DIRECTORY, dfname_single_label.replace('.csv', '_train.csv')), index=False)
	val_df.to_csv(os.path.join(DATASET_DIRECTORY, dfname_single_label.replace('.csv', '_val.csv')), index=False)

	plot_train_val_label_distribution(
		train_df=train_df,
		val_df=val_df,
		dataset_name=dataset_name,
		VAL_SPLIT_PCT=args.val_split_pct,
		fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_simple_random_split_stratified_single_label_distribution_train_val_{args.val_split_pct}_pct.png'),
		FIGURE_SIZE=(14, 8),
		DPI=DPI,
		label_column='label',
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
		print(f"RGB: Mean: {img_rgb_mean} | Std: {img_rgb_std}")

if __name__ == '__main__':
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	get_ip_info()
	main()
	print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(160, " "))
