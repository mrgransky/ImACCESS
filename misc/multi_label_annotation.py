from utils import *
from visualize import perform_multilabel_eda
from fuzzywuzzy import fuzz
# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_grad_enabled(False)

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WW_VEHICLES/metadata_multi_label.csv -d "cuda:0" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WW_VEHICLES.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label.csv -d "cuda:0" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_SMU.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata_multi_label.csv -d "cuda:0" -nw 24 -tbs 8 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata_multi_label.csv -d "cuda:1" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_multi_label.csv -d "cuda:2" -nw 20 -tbs 8 -vbs 32 -vth 0.3 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata_multi_label.csv -d "cuda:3" -nw 8 -tbs 256 -vbs 16 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.txt &


# Make language detection deterministic
DetectorFactory.seed = 42

dtypes = {
	'doc_id': str, 'id': str, 'label': str, 'title': str,
	'description': str, 'img_url': str, 'enriched_document_description': str,
	'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
	'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	'user_query': str, 'country': str,
}

# Custom stopwords and metadata patterns
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union(
	{
		"bildetekst", "photo", "image", "archive", "arkivreferanse", "caption", "following", "below", "above",
		"copyright", "description", "riksarkivet", "ntbs", "ra", "pa", "bildetekst", "century", "twentieth",
		"showing", "shown", "shows", "depicts", "depicting", "pictured", "picture", "pinterest",
		"copy", "version", "view", "looking", "seen", "visible", "illustration",
		"photograph", "photography", "photo", "image", "img", "photographer",
		"sent", "received", "taken", "made", "created", "produced", "found",
		"across", "opposite", "near", "under", "over", "inside", "outside",
		"collection", "collections", "number", "abbreviation", "abbreviations",
		"folder", "box", "file", "document", "page", "index", "label", "code", "icon", "type", "unknown", "unknow",
		"folder icon", "box icon", "file icon", "document icon", "page icon", "index icon", "label icon", "code icon",
		"used", "southern", "built", "album", "album cover", "opel ma",
		"original", "information", "item", "http", "www", "jpg", "000",
		"jpeg", "png", "gif", "bmp", "tiff", "tif", "ico", "svg", "webp", "heic", "heif", "raw", "cr2", "nef", "orf", "arw", "dng", "nrw", "k25", "kdc", "rw2", "raf", "mrw", "pef", "sr2", "srf",
	}
)

FastText_Language_Identification = "lid.176.bin"
if FastText_Language_Identification not in os.listdir():
	url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FastText_Language_Identification}"
	urllib.request.urlretrieve(url, FastText_Language_Identification)

SEMANTIC_CATEGORIES = {
	'military': ['military', 'army', 'navy', 'air force', 'soldier', 'officer', 'troop', 'regiment', 'division', 'corps', 'battalion', 'brigade'],
	'political': ['government', 'parliament', 'president', 'prime minister', 'minister', 'official', 'politician', 'leader'],
	'event': ['war', 'battle', 'attack', 'invasion', 'liberation', 'occupation', 'revolution', 'protest', 'march', 'ceremony'],
	'location': ['city', 'town', 'village', 'country', 'region', 'territory', 'front', 'border', 'base', 'camp'],
	'vehicle': ['tank', 'aircraft', 'plane', 'ship', 'submarine', 'boat', 'truck', 'car', 'jeep', 'vehicle'],
	'weapon': ['gun', 'rifle', 'cannon', 'artillery', 'weapon', 'bomb', 'missile', 'ammunition'],
}

cache_directory = {
	"farid": "/home/farid/datasets/trash/models",
	"alijanif": "/scratch/project_2004072/ImACCESS/models",
	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs/models",
}

USER = os.getenv("USER")
hf_tk = os.getenv("HUGGINGFACE_TOKEN")
print(f"USER: {USER} | HUGGINGFACE_TOKEN: {hf_tk} Login to HuggingFace Hub...")
huggingface_hub.login(token=hf_tk)

def is_likely_english_term(term):
	"""Check if a term is likely English or a proper noun"""
	if not term or len(term) < 3:
		return False
					
	# Common characters in European languages but not in English
	non_english_chars = 'äöüßñéèêëìíîïçåæœø'
	
	# Check for non-English characters
	if any(char in non_english_chars for char in term.lower()):
		return False
					
	# Allow terms that might be proper nouns (start with uppercase)
	if term[0].isupper():
		return True

	# Check if term is in an English dictionary
	common_english_words = set(nltk.corpus.words.words())
	return term.lower() in common_english_words

def clean_(labels):
		cleaned = set()
		for label in labels:
				# Normalize and clean more aggressively
				label = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', label.lower())  # Remove duplicates like "Italy Italy"
				label = re.sub(r'[^a-z0-9\s\-]', '', label).strip()
				label = ' '.join(label.split())  # Normalize whitespace
				
				# More sophisticated filtering
				if (len(label) < 3 or 
						label in CUSTOM_STOPWORDS or
						label.isdigit() or
						(len(label.split()) > 3 and not any(w[0].isupper() for w in label.split()))):
						continue
				cleaned.add(label)
		return sorted(cleaned)

def filter_metadata_terms(labels):
	metadata_fragments = [
		'kunststoff',
		'bildetekst',
		'arkiv',
		'quer',
		'riksarkivet',
		'museum donated',
		'association',
		'information received',
	]
	return [label for label in labels if not any(frag in label for frag in metadata_fragments)]

def assign_semantic_categories(labels):
	"""Categorize labels into semantic groups"""
	categorized_labels = []
	for label in labels:
		for category, terms in SEMANTIC_CATEGORIES.items():
			if any(term in label for term in terms):
				if label not in categorized_labels:
					categorized_labels.append(label)
				break
	return categorized_labels

def is_year_compatible(category, year):
		"""
		Check if a label's era is compatible with the document year.
		
		Args:
						category: String label
						year: Float or int document year
		
		Returns:
						Bool indicating compatibility
		"""
		if year is None:
						return True

		era_ranges = {
						"WWI era": (1914, 1918),
						"WWII era": (1939, 1945),
						"Cold War era": (1947, 1991)
		}
		for era, (start, end) in era_ranges.items():
						if era.lower() in category.lower() and year and not (start <= year <= end):
										return False
		return True

def parse_user_query(query: Union[str, None]) -> List[str]:
		if not query or not isinstance(query, str) or not query.strip():
				return []
		# Split by common delimiters (comma, semicolon, pipe, newline)
		terms = re.split(r'[,\|;\n]', query)
		cleaned_terms = []
		english_words = set(words.words())  # For basic English validation
		for term in terms:
				term = term.strip().lower()
				# Skip short, stopword, or non-English terms
				if (len(term) < 4 or 
						term in CUSTOM_STOPWORDS or 
						not re.match(r'^[a-z\s-]+$', term) or
						term not in english_words and not term[0].isupper()):
						continue
				cleaned_terms.append(term)
		# Remove duplicates while preserving order
		return sorted(list(dict.fromkeys(cleaned_terms)))

def deduplicate_labels(labels: List[str]) -> List[str]:
	if not labels:
		return []
	deduplicated = []
	labels = sorted(labels, key=len, reverse=True)  # Process longer labels first
	for label in labels:
		label_clean = ' '.join(label.lower().strip().split())  # Normalize whitespace
		# Skip if label is too short or already covered
		if len(label_clean) < 4:
			continue
		# Split label into tokens for overlap check
		label_tokens = set(label_clean.split())
		is_duplicate = any(
			fuzz.ratio(label_clean, kept.lower()) > 90 or
			label_clean in kept.lower() or kept.lower() in label_clean or
			(len(label_tokens & set(kept.lower().split())) / len(label_tokens) > 0.3)
			for kept in deduplicated
		)
		if not is_duplicate:
			deduplicated.append(label)
	
	return sorted(deduplicated)

def preprocess_text(text: str) -> str:
	if not text or not isinstance(text, str):
		return ""
	# Normalize hyphens/underscores and whitespace
	text = re.sub(r'[-_]', ' ', text)
	text = re.sub(r'\s+', ' ', text).strip()
	
	# Split into phrases (by punctuation or conjunctions)
	phrases = re.split(r'[.,;()&-]| - ', text)
	phrases = [p.strip() for p in phrases if p.strip() and len(p.strip()) > 2]
	
	# Deduplicate phrases while preserving order
	seen = set()
	deduped_phrases = []
	for phrase in phrases:
		phrase_lower = phrase.lower()
		if phrase_lower not in seen:
			seen.add(phrase_lower)
			deduped_phrases.append(phrase)
	return ' '.join(deduped_phrases)

def custom_collate_fn(batch):
	valid_items = [(idx, img) for idx, img in batch if img is not None]
	if not valid_items:
		return [], []	
	indices, images = zip(*valid_items)
	return list(indices), list(images)

class HistoricalArchives(Dataset):
	def __init__(self, img_paths):
		self.img_paths = img_paths
			
	def __len__(self):
		return len(self.img_paths)
			
	def __getitem__(self, idx):
		try:
			img = Image.open(self.img_paths[idx]).convert("RGB")
			return idx, img
		except Exception as e:
			print(f"Error loading image {self.img_paths[idx]}: {e}")
			return idx, None

def get_visual_based_annotation(
		csv_file: str,
		vlm_model_name: str,
		batch_size: int,
		device: str,
		num_workers: int,
		verbose: bool,
		metadata_fpth: str,
		topk: int = 3,
		threshold: float = 0.1,
	):
	if verbose:
		print(f"Semi-Supervised visual-based annotation (using VLM) batch_size: {batch_size} num_workers: {num_workers}".center(160, "-"))
	
	start_time = time.time()
	
	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	texts = [f"This is a photo of {lbl}." for lbl in candidate_labels]
	# Load model and processor
	processor = AutoProcessor.from_pretrained(vlm_model_name)
	model = AutoModel.from_pretrained(
			pretrained_model_name_or_path=vlm_model_name,
			dtype=torch.float16 if torch.cuda.mem_get_info()[0] / 1024**3 < 7 else torch.float32,
			device_map=device,
			cache_dir=cache_directory[USER],
	).eval()
	model = torch.compile(model, mode="max-autotune")
	# Precompute text embeddings
	with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
		text_inputs = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
		text_embeddings = model.get_text_features(**text_inputs)
		text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
	
	df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	img_paths = df['img_path'].tolist()
	visual_based_labels = []
	visual_based_scores = []
	# DataLoader setup
	dataset = HistoricalArchives(img_paths)
	dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=num_workers,
			pin_memory=torch.cuda.is_available(),
			persistent_workers=num_workers > 1,
			prefetch_factor=2,
			drop_last=False,
			collate_fn=custom_collate_fn
	)
	print(f"Processing {len(img_paths)} images in {batch_size} batches...")
	for batch_indices, images in tqdm(dataloader, desc="Processing images"):
			if not images:
					continue
			try:
					with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
							image_inputs = processor(
									images=images,
									padding="max_length",
									max_num_patches=4096,
									return_tensors="pt",
							).to(device, non_blocking=True)
							image_embeddings = model.get_image_features(**image_inputs)
							image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
							similarities = image_embeddings @ text_embeddings.T
							for idx_in_batch, global_idx in enumerate(batch_indices[:len(images)]):
									topk_probs, topk_indices = similarities[idx_in_batch].topk(topk)
									top_labels, top_scores = [], []
									for score, label_idx in zip(topk_probs, topk_indices):
											val = score.item()
											if val >= threshold:
													top_labels.append(candidate_labels[label_idx])
													top_scores.append(round(val, 4))
									visual_based_labels.append(top_labels if top_labels else None)
									visual_based_scores.append(top_scores if top_scores else None)
			except Exception as e:
					print(f"ERROR: failed to process batch {batch_indices[0]}-{batch_indices[-1]}: {e}")
					torch.cuda.empty_cache()
					continue
	df['visual_based_labels'] = visual_based_labels
	df['visual_based_scores'] = visual_based_scores
	# Save output
	df.to_csv(metadata_fpth, index=False)
	try:
			df.to_excel(metadata_fpth.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
			print(f"Excel export failed: {e}")
	print(f"Processed {len(img_paths)} images, generated {sum(1 for x in visual_based_labels if x)} valid results")
	print(f"Visual-based annotation Elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))
	return visual_based_labels

def get_textual_based_annotation(
		csv_file: str, 
		batch_size: int,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		threshold: float = 0.35,
		topk: int = 10,
		verbose: bool = True,
	):
	t = torch.cuda.get_device_properties(device=device).total_memory
	r = torch.cuda.memory_reserved(device=device)
	a = torch.cuda.memory_allocated(device=device)
	f = r-a  # free inside reserved
	if verbose:
		print(f"Semi-Supervised textual-based annotation batch_size: {batch_size}".center(160, "-"))
		print(f"{device} Memory: Total: {t/1024**3:.2f} GB Reserved: {r/1024**3:.2f} GB Allocated: {a/1024**3:.2f} GB Free: {f/1024**3:.2f} GB".center(160, " "))
	# device = 'cpu' if t/1024**3 < 6 else device
	start_time = time.time()

	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))

	# Load model with memory optimizations
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name} in {device}")
	torch.cuda.empty_cache()
	sent_model = SentenceTransformer(
		model_name_or_path=st_model_name,
		device=device,
		trust_remote_code=True,
		cache_folder=cache_directory[USER],
	)
	sent_model.eval()
		
	# Pre-compute label embeddings once
	if verbose:
		print(
			f"Pre-computing embeddings for {len(candidate_labels)} pre-defined labels: "
			f"{device} using batch_size: {min(batch_size, len(candidate_labels))}..."
		)
		
	t0 = time.time()
	label_embs = sent_model.encode(
		sentences=candidate_labels,
		batch_size=min(batch_size, len(candidate_labels)),
		device=device,
		convert_to_tensor=True,
		normalize_embeddings=True,
		show_progress_bar=False,
		# task='retrieval', #TODO: only for ...
	)
	if verbose:
		print(f"Label embeddings: {type(label_embs)} {label_embs.shape} computed in {time.time() - t0:.2f} s".center(160, " "))
	
	if verbose:
		print(f"Loading dataframe: {csv_file}")
	df = pd.read_csv(
		filepath_or_buffer=csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	# Initialize results columns with None	
	df['textual_based_labels'] = None
	df['textual_based_scores'] = None

	# Process in chunks with memory management
	chunk_size = min(500, len(df))
	if verbose:
		print(f"Processing {len(df)} samples with {len(candidate_labels)} candidate labels in {chunk_size} chunks...")
	for chunk_start in tqdm(range(0, len(df), chunk_size), desc="Processing documents"):
		chunk_end = min(chunk_start + chunk_size, len(df))
		chunk_df = df.iloc[chunk_start:chunk_end]
		
		# Filter out empty descriptions
		valid_indices = []
		texts_to_process = []
		for idx, row in chunk_df.iterrows():
			desc = row['enriched_document_description']
			if pd.isna(desc) or not str(desc).strip():
				continue
			valid_indices.append(idx)
			texts_to_process.append(str(desc).strip())
		
		# Skip if no valid texts in this chunk
		if not texts_to_process:
			continue
		
		# Process texts in smaller batches
		text_batch_size = min(64, batch_size)
		text_embs = []
		
		for i in range(0, len(texts_to_process), text_batch_size):
			batch_texts = texts_to_process[i:i+text_batch_size]
			try:
				batch_embs = sent_model.encode(
					sentences=batch_texts,
					batch_size=text_batch_size,
					device=device,
					convert_to_tensor=True,
					normalize_embeddings=True,
					show_progress_bar=False,
					# task='retrieval',
				)
				text_embs.append(batch_embs)
				torch.cuda.empty_cache()
			except torch.cuda.OutOfMemoryError:
				text_batch_size = max(1, text_batch_size // 2)
				print(f"Reducing text batch size to {text_batch_size} due to OOM")
				continue
		
		if not text_embs:
			continue
						
		text_embs = torch.cat(text_embs)
		
		# Compute similarities in batches
		sim_batch_size = min(32, batch_size * 2)
		cosine_scores = []
		
		for i in range(0, len(text_embs), sim_batch_size):
			batch_sims = util.cos_sim(
				text_embs[i:i+sim_batch_size],
				label_embs
			)
			cosine_scores.append(batch_sims)
			torch.cuda.empty_cache()
		
		cosine_scores = torch.cat(cosine_scores)
		
		# Get top-k results with bounds checking
		topk_scores, topk_indices = torch.topk(cosine_scores, k=min(topk, len(candidate_labels)), dim=1)
				
		for i, original_idx in enumerate(valid_indices):
			try:
				labels = []
				scores = []
				for j, score in zip(topk_indices[i], topk_scores[i]):
					s = score.item()
					if s >= threshold:
						labels.append(candidate_labels[j])
						scores.append(round(s, 4))
				
				if labels:
					df.at[original_idx, 'textual_based_labels'] = labels
					df.at[original_idx, 'textual_based_scores'] = scores
				else:
					df.at[original_idx, 'textual_based_labels'] = None
					df.at[original_idx, 'textual_based_scores'] = None

			except Exception as e:
				print(f"Error processing sample {original_idx}: {str(e)[:200]}")
				df.at[original_idx, 'textual_based_labels'] = None
				df.at[original_idx, 'textual_based_scores'] = None


		del text_embs, cosine_scores, topk_scores, topk_indices
		torch.cuda.empty_cache()
	
	if verbose:
		print(f"Saving results to {metadata_fpth}...")
	df.to_csv(metadata_fpth, index=False)

	try:
		df.to_excel(metadata_fpth.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	print(f"Textual-based annotation completed in {time.time() - start_time:.2f} seconds")
	return df['textual_based_labels'].tolist()

def get_captions(
		csv_file: str, 
		batch_size: int,
		metadata_fpth: str,
		device: str,
		model_name: str,
		verbose: bool = True,
		max_length: int = 50,
	):
		t = torch.cuda.get_device_properties(device=device).total_memory
		r = torch.cuda.memory_reserved(device=device)
		a = torch.cuda.memory_allocated(device=device)
		f = r - a
		if verbose:
			print(f"Generating captions for {csv_file} | batch_size={batch_size}".center(160, "-"))
			print(
				f"{device} Memory: Total: {t/1024**3:.2f} GB | Reserved: {r/1024**3:.2f} GB "
				f"| Allocated: {a/1024**3:.2f} GB | Free: {f/1024**3:.2f} GB".center(160, " ")
			)
		start_time = time.time()

		# ---- Load model and processor ----
		if verbose:
			print(f"Loading captioning model: {model_name} on {device}")
		
		processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
		model = AutoModelForCausalLM.from_pretrained(
			model_name,
			device_map=device,
			cache_dir=cache_directory[USER],
		).eval()

		# ---- Load dataframe and dataset ----
		df = pd.read_csv(csv_file, on_bad_lines="skip", dtype=dtypes)
		img_paths = df["img_path"].tolist()
		dataset = HistoricalArchives(img_paths)   # <- same dataset class you used before
		dataloader = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=2,
			pin_memory=torch.cuda.is_available(),
			collate_fn=custom_collate_fn,
		)

		if verbose:
			print(f"Processing {len(img_paths)} images in {len(dataloader)} batches...")

		# ---- Inference loop ----
		captions = []
		for batch_indices, images in tqdm(dataloader, desc="Captioning images"):
			if not images:
				continue
			try:
				with torch.no_grad():
					with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
						inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
						generated_ids = model.generate(**inputs, max_length=max_length)
						batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
						captions.extend(batch_captions)
			except Exception as e:
				print(f"ERROR: failed to process batch {batch_indices[0]}-{batch_indices[-1]}: {e}")
				torch.cuda.empty_cache()
				for _ in batch_indices:
					captions.append(None)

		# ---- Save results ----
		df["caption"] = captions
		df.to_csv(metadata_fpth, index=False)
		try:
				df.to_excel(metadata_fpth.replace(".csv", ".xlsx"), index=False)
		except Exception as e:
				print(f"Excel export failed: {e}")

		print(f"Processed {len(img_paths)} images, generated {sum(1 for x in captions if x)} captions")
		print(f"Captioning Elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))

		return captions

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=64, help="Batch size for textual processing")
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=4, help="Batch size for vision processing")
	parser.add_argument("--text_relevance_threshold", '-trth', type=float, default=0.47, help="Relevance threshold for textual-based labels")
	parser.add_argument("--vision_relevance_threshold", '-vrth', type=float, default=0.1, help="Relevance threshold for visual-based labels")
	# parser.add_argument("--sentence_model_name", '-smn', type=str, default="google/embeddinggemma-300m", choices=["google/embeddinggemma-300m", "Qwen/Qwen3-Embedding-8B", "all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "jinaai/jina-embeddings-v3", "paraphrase-multilingual-MiniLM-L12-v2"], help="Sentence-transformer model name")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Sentence-transformer model name")
	parser.add_argument("--vlm_model_name", '-vlm', type=str, default="google/siglip2-so400m-patch16-naflex", choices=["kakaobrain/align-base", "google/siglip2-so400m-patch16-naflex"], help="Vision-Language model name")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)

	set_seeds(seed=42)
	text_output_path = args.csv_file.replace('.csv', '_textual_based_labels.csv')
	vision_output_path = args.csv_file.replace('.csv', '_visual_based_labels.csv')
	combined_output_path = args.csv_file.replace('.csv', '_multimodal.csv')
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str, 'country': str,
	}
	
	if os.path.exists(text_output_path):
		print(f"Found existing textual-based labels at {text_output_path} Loading...")
		t0 = time.time()
		text_df = pd.read_csv(
			filepath_or_buffer=text_output_path,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		textual_based_labels = []
		for label_str in text_df['textual_based_labels']:
			if pd.isna(label_str) or label_str == '[]' or not label_str:
				textual_based_labels.append([])
			else:
				try:
					labels = eval(label_str)
					textual_based_labels.append(labels if labels else [])
				except:
					textual_based_labels.append([])
		print(f"Loaded {len(textual_based_labels)} textual-based labels in {time.time() - t0:.2f} sec")
	else:
		textual_based_labels = get_textual_based_annotation(
			csv_file=args.csv_file,
			batch_size=args.text_batch_size,
			st_model_name=args.sentence_model_name,
			metadata_fpth=text_output_path,
			threshold=args.text_relevance_threshold,
			device=args.device,
			topk=3,
			verbose=True,
		)

	if os.path.exists(vision_output_path):
		print(f"Found existing visual-based labels at {vision_output_path} Loading...")
		t0 = time.time()
		vision_df = pd.read_csv(
			filepath_or_buffer=vision_output_path,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		visual_based_labels = []
		for label_str in vision_df['visual_based_labels']:
			if pd.isna(label_str) or label_str == '[]' or not label_str:
				visual_based_labels.append([])
			else:
				try:
					labels = eval(label_str)
					visual_based_labels.append(labels if labels else [])
				except:
					visual_based_labels.append([])
		
		print(f"Loaded {len(visual_based_labels)} visual-based labels in {time.time() - t0:.2f} sec")
	else:
		visual_based_labels = get_visual_based_annotation(
			csv_file=args.csv_file,
			vlm_model_name=args.vlm_model_name,
			batch_size=args.vision_batch_size,
			device=args.device,
			num_workers=args.num_workers,
			metadata_fpth=vision_output_path,
			threshold=args.vision_relevance_threshold,
			topk=3,
			verbose=True,
		)
	
	assert len(textual_based_labels) == len(visual_based_labels), "Label lists must have same length"
	


	#####################

	caption_output_path = args.csv_file.replace('.csv', '_captions.csv')

	if os.path.exists(caption_output_path):
		print(f"Found existing captions at {caption_output_path} Loading...")
		t0 = time.time()
		caption_df = pd.read_csv(
			filepath_or_buffer=caption_output_path,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		captions = []
		for cap in caption_df['caption']:
			if pd.isna(cap) or not cap:
				captions.append(None)
			else:
				captions.append(cap.strip())
		print(f"Loaded {len(captions)} captions in {time.time() - t0:.2f} sec")
	else:
		captions = get_captions(
			csv_file=args.csv_file,
			batch_size=args.vision_batch_size,  # reuse vision batch size
			metadata_fpth=caption_output_path,
			device=args.device,
			model_name="microsoft/git-large-coco",  # or args.caption_model_name if you expose new arg
			verbose=True,
		)



	######################

















	
	if os.path.exists(combined_output_path):
		print(f"Found existing combined labels at {combined_output_path} Loading...")
		print("Using existing combined labels.")
		combined_df = pd.read_csv(
			filepath_or_buffer=combined_output_path,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		combined_labels = []
		for label_str in combined_df['multimodal_labels']:
			if pd.isna(label_str) or label_str == '[]' or not label_str:
				combined_labels.append([])
			else:
				try:
					labels = eval(label_str)
					combined_labels.append(labels if labels else [])
				except:
					combined_labels.append([])				
		print(f"Loaded {len(combined_labels)} combined labels")
	else:
		print("Merging textual and visual labels".center(160, "-"))
		combined_labels = []
		empty_count = 0	
		for text_labels, vision_labels in zip(textual_based_labels, visual_based_labels):
			if not text_labels and not vision_labels:
				combined_labels.append([])
				empty_count += 1
				continue
					
			# Ensure both are lists before combining
			if not isinstance(text_labels, list):
				text_labels = []
			if not isinstance(vision_labels, list):
				vision_labels = []
					
			# Combine all labels
			all_labels = list(set(text_labels + vision_labels))
			
			# Clean and filter
			all_labels = clean_(all_labels)
			all_labels = filter_metadata_terms(all_labels)
			all_labels = deduplicate_labels(all_labels)
			
			# Add semantic categories
			categorized = assign_semantic_categories(all_labels)
			final_labels = sorted(set(all_labels + categorized))
			
			combined_labels.append(final_labels)
		
		print(f"Created {len(combined_labels)} combined labels ({empty_count} empty entries)")
	
	if not os.path.exists(combined_output_path):
		df = pd.read_csv(
			filepath_or_buffer=args.csv_file,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)

		df['textual_based_labels'] = [labels if labels else None for labels in textual_based_labels]
		df['visual_based_labels'] = [labels if labels else None for labels in visual_based_labels]
		df['multimodal_labels'] = [labels if labels else None for labels in combined_labels]

		print(f"Saving dataframe to {combined_output_path}...")
		df.to_csv(combined_output_path, index=False)
		try:
			df.to_excel(combined_output_path.replace('.csv', '.xlsx'), index=False)
		except Exception as e:
			print(f"Failed to write Excel file: {e}")

	perform_multilabel_eda(
		data_path=combined_output_path, 
		label_column='multimodal_labels',
	)

	train_df_fpth = combined_output_path.replace('.csv', '_train.csv')
	val_df_fpth = combined_output_path.replace('.csv', '_val.csv')
	try:
		train_df = pd.read_csv(
			filepath_or_buffer=train_df_fpth,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
		val_df = pd.read_csv(
			filepath_or_buffer=val_df_fpth,
			on_bad_lines='skip',
			dtype=dtypes,
			low_memory=False,
		)
	except Exception as e:
		print(f"<!> {e}")
		train_df, val_df = get_multi_label_stratified_split(
			df=df,
			val_split_pct=0.35,
			seed=42,
			label_col='multimodal_labels'
		)
		train_df.to_csv(train_df_fpth, index=False)
		val_df.to_csv(val_df_fpth, index=False)

	print("Multi-label Stratified Split Results:")
	print(f"Train set shape: {train_df.shape}")
	print(f"Validation set shape: {val_df.shape}")

if __name__ == "__main__":
	multiprocessing.set_start_method('spawn', force=True)
	main()