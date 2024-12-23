import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from misc.utils import *

# how to run in local:
# $ nohup python -u data_collector.py -ddir $HOME/WS_Farid/ImACCESS/datasets/WW_DATASETs > logs/wwii_image_download.out &

# run in Pouta:
# $ python data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1960-12-31
# $ nohup python -u data_collector.py --dataset_dir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 -nw 8 --img_mean_std > /media/volume/ImACCESS/trash/wwii_data_collection.out &
# $ nohup python -u data_collector.py -ddir /media/volume/ImACCESS/WW_DATASETs -sdt 1900-01-01 -edt 1970-12-31 > /media/volume/ImACCESS/trash/wwii_data_collection.out &

parser = argparse.ArgumentParser(description="WWII Dataset")
parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset DIR')
parser.add_argument('--start_date', '-sdt', type=str, default="1939-09-01", help='Start Date')
parser.add_argument('--end_date', '-edt', type=str, default="1945-09-02", help='End Date')
parser.add_argument('--num_workers', '-nw', type=int, default=8, help='Number of CPUs')
parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')

args, unknown = parser.parse_known_args()
print(args)

meaningless_words_fpth = os.path.join(parent_dir, 'misc', 'meaningless_words.txt')
# STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
STOPWORDS = list()
with open(meaningless_words_fpth, 'r') as file_:
	customized_meaningless_words=[line.strip().lower() for line in file_]
STOPWORDS.extend(customized_meaningless_words)
STOPWORDS = set(STOPWORDS)
# print(STOPWORDS, type(STOPWORDS))

START_DATE = args.start_date
END_DATE = args.end_date
dataset_name = "WWII"

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

def get_dframe(doc_idx:int=1000, doc_url:str="www.example.com", doc_label: str="label"):
	print(f"Extracting DF for label[{doc_idx}]: {doc_label}".center(150, "-"))
	doc_url_info = extract_url_info(doc_url)
	print(f"{doc_url}".center(150, " "))
	qv_processed = re.sub(
		pattern=" ", 
		repl="_", 
		string=doc_label,
	)
	url_processed = re.sub(
		pattern=r"/|:|\.",
		repl="_",
		string=doc_url,
	)
	df_fpth = os.path.join(HITs_DIR, f"df_query_{qv_processed}_URL_{url_processed}_{START_DATE}_{END_DATE}.gz")
	if os.path.exists(df_fpth):
		df = load_pickle(fpath=df_fpth)
		return df
	df_st_time = time.time()
	qLBL_DIR = os.path.join(HITs_DIR, re.sub(" ", "_", doc_label))
	os.makedirs(qLBL_DIR, exist_ok=True)
	try:
		response = requests.get(doc_url)
		soup = BeautifulSoup(response.text, 'html.parser')
		hits = soup.find_all('img', class_='attachment-thumbnail')
		descriptions = soup.find_all("p")
		header = soup.find('h2', class_="entry-title").text
	except Exception as e:
		print(f"Failed to retrieve doc_url or parse content: {e}")
		return
	# header = re.sub(r' part \d+', '', header.lower())
	header = clean_(
		text=header,
		# sw=[],
		sw=STOPWORDS,
	)
	# print(doc_url_info)
	header = header + " " + (doc_url_info.get("country") or "") + " " + (doc_url_info.get("main_label") or "") + " " + (doc_url_info.get("type") or "")
	print(f"Doc header: {header}")
	filtered_descriptions_list = [
		t.text 
		for t in descriptions 
		if t.text
		and t.parent.get('class')==['eazyest-gallery'] 
		and t.parent.get('class')!=['textwidget']
	]
	filtered_descriptions = " ".join(filtered_descriptions_list)
	filtered_descriptions = re.sub(
		pattern=r"Bibliography:|Specifications:|Variants:",
		repl=" ",
		string=filtered_descriptions,
	)
	doc_description = re.sub(r'\s+', ' ', filtered_descriptions).strip()
	doc_description = clean_(
		text=header + " " + doc_description, 
		# sw=[],
		sw=STOPWORDS,
	)
	print(f"Doc Description: {doc_description}")
	print()

	print(f"Found {len(hits)} Document(s) => Extracting information [might take a while]")
	data = []
	for idoc, vdoc in enumerate(hits):
		img_url = vdoc.get('data-src')
		doc_title = clean_(
			text=vdoc.get("alt"), 
			# sw=[],
			sw=STOPWORDS,
		)
		# doc_title = vdoc.get("alt").lower()
		if not img_url:
			continue
		img_url = img_url.replace("_cache/", "") # Remove "_cache/" from the URL
		img_url = re.sub(r'-\d+x\d+\.jpg$', '.jpg', img_url) # Remove the thumbnail size from the end of the URL
		# print(f"[{idoc+1}/{len(hits)}] {img_url}")
		# print(doc_title)
		# print()
		filename = os.path.basename(img_url)
		img_fpath = os.path.join(qLBL_DIR, filename)
		if not os.path.exists(img_fpath):
			try:
				img_response = requests.get(img_url)
				if img_response.status_code == 200:
					# with open(os.path.join(qLBL_DIR, filename), 'wb') as f:
					with open(img_fpath, 'wb') as f:
						f.write(img_response.content)
					# print(f"Downloaded: {filename}")
				else:
					filename = None
					img_url = None
					print(f"Failed to download {img_url}. Status code: {img_response.status_code}")
			except Exception as e:
				print(f"Failed to download {img_url}: {e}")
		# else:
		# 	print(f"Skipping {img_fpath}, already exists")
		# wwii_identifier = re.sub(".jpg", "", filename)
		wwii_identifier = filename
		img_path = os.path.join(IMAGE_DIR, wwii_identifier)
		# print(f"wwii_identifier: {wwii_identifier} ==>> img_path: {img_path}")
		row = {
			'id': wwii_identifier, #wwii_identifier,
			'label': doc_label,
			'title': doc_title,
			'description': doc_description,
			'img_url': img_url,
			'label_title_description': doc_label + " " + (doc_title or '') + " " + (doc_description or ''),
			'date': None,
			'doc_url': doc_url,
			# 'img_path': f"{os.path.join(IMAGE_DIR, str(wwii_identifier) + '.jpg')}"
			'img_path': img_path,
		}
		data.append(row)
	df = pd.DataFrame(data)
	print(f"DF: {df.shape} {type(df)} Elapsed time: {time.time()-df_st_time:.1f} sec")
	save_pickle(pkl=df, fname=df_fpth)
	# print(df.head(20))
	# print("#"*100)
	# print(df.tail(20))
	# copy images of that dir to images:
	for fname in os.listdir(qLBL_DIR):
		shutil.copy(os.path.join(qLBL_DIR, fname), IMAGE_DIR)
	return df

def main():
	base_url = "https://www.worldwarphotos.info/gallery"
	URLs = { # key: url : val: doc_label
		f"{base_url}/france/normandy-1944/": "normandy invasion", # Invasion of Normandy 1944 photo gallery
		f"{base_url}/france/tanks-france/" : "armored fighting vehicle", # French Tanks of World War II
		f"{base_url}/italy/spg2/75-18/" : "armored fighting vehicle", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
		f"{base_url}/italy/spg2/l40/" : "armored fighting vehicle", # https://en.wikipedia.org/wiki/Armoured_fighting_vehicle
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
		f"{base_url}/japan/aircrafts/p1y/": "aircraft", #
		f"{base_url}/japan/ijn/midget/": "naval forces", # japanese navy
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
		f"{base_url}/germany/artillery/pak43/": "artillery", # anti-tank
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
	print(f"Scraping {len(URLs)} URLs...")
	dfs = [
		get_dframe(
			doc_idx=i, 
			doc_url=k, 
			doc_label=v,
		) for i, (k, v) in enumerate(URLs.items())
	]

	print(f"Concatinating {len(dfs)} dfs...")
	# print(dfs[0])
	merged_df = pd.concat(dfs, ignore_index=True)
	print(merged_df.shape)
	print(merged_df.describe())
	print("#"*150)
	print(merged_df.head(50))
	print("#"*150)
	print(merged_df.tail(50))

	label_counts = merged_df['label'].value_counts()
	# print(label_counts.tail(25))

	plt.figure(figsize=(16, 10))
	label_counts.plot(kind='bar', fontsize=9)
	plt.title(f'{dataset_name}: Query Frequency (total: {label_counts.shape}) {START_DATE} - {END_DATE}')
	plt.xlabel('Query')
	plt.ylabel('Frequency')
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUTs_DIR, f"all_query_labels_x_{label_counts.shape[0]}_freq.png"))

	merged_df.to_csv(os.path.join(DATASET_DIRECTORY, "metadata.csv"), index=False)
	try:
		merged_df.to_excel(os.path.join(DATASET_DIRECTORY, "metadata.xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if args.img_mean_std:
		try:
			img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
		except Exception as e:
			print(f"{e}")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(dir=os.path.join(DATASET_DIRECTORY, "images"), num_workers=args.num_workers)
			save_pickle(pkl=img_rgb_mean, fname=img_rgb_mean_fpth)
			save_pickle(pkl=img_rgb_std, fname=img_rgb_std_fpth)
		print(f"RGB: Mean: {img_rgb_mean} | Std: {img_rgb_std}")

if __name__ == "__main__":
	print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(160, " "))
	START_EXECUTION_TIME = time.time()
	main()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)