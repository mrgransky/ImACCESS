from utils import *
from visualize import perform_multilabel_eda
from fuzzywuzzy import fuzz
# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.set_grad_enabled(False)

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_SMU.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 24 -tbs 8 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata.csv -d "cuda:1" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata.csv -d "cuda:2" -nw 20 -tbs 8 -vbs 32 -vth 0.3 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.txt &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -d "cuda:3" -nw 8 -tbs 256 -vbs 16 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.txt &

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

def batch_filter_by_relevance(
		sent_model: SentenceTransformer,
		texts: List[str],
		all_labels_list: List[List[str]],
		threshold: float,
		batch_size: int,
		max_retries: int = 3
) -> List[List[str]]:
		results = []
		if not texts or not all_labels_list:
				return results
		
		# Initialize with conservative batch sizes
		text_batch_size = max(1, min(batch_size, 8))
		label_batch_size = max(1, min(batch_size // 2, 16))
		
		# Memory optimization flags
		torch.backends.cuda.enable_flash_sdp(True)  # Enable flash attention if available
		torch.backends.cuda.enable_mem_efficient_sdp(True)
		
		def safe_encode(items, is_labels=False):
				"""Helper function with automatic batch size reduction"""
				current_batch_size = label_batch_size if is_labels else text_batch_size
				for attempt in range(max_retries):
						try:
								return sent_model.encode(
										items,
										batch_size=current_batch_size,
										show_progress_bar=False,
										convert_to_numpy=True,
										convert_to_tensor=False,
										device='cuda' if not is_labels else 'cpu'  # Labels can often be processed on CPU
								)
						except torch.cuda.OutOfMemoryError:
								if attempt == max_retries - 1:
										raise
								current_batch_size = max(1, current_batch_size // 2)
								print(f"Reducing {'label' if is_labels else 'text'} batch size to {current_batch_size}")
								torch.cuda.empty_cache()
		
		# Process texts in optimized batches
		text_embeddings = []
		for i in tqdm(range(0, len(texts), text_batch_size), desc="Encoding texts"):
				batch = texts[i:i + text_batch_size]
				try:
						batch_emb = safe_encode(batch)
						text_embeddings.extend(batch_emb)
				except Exception as e:
						print(f"Error encoding text batch {i}-{i+text_batch_size}: {str(e)[:200]}")
						text_embeddings.extend([np.zeros(sent_model.get_sentence_embedding_dimension())] * len(batch))
		text_embeddings = np.array(text_embeddings)
		
		# Process labels with memory awareness
		for i, (text_emb, labels) in enumerate(tqdm(zip(text_embeddings, all_labels_list), total=len(texts), desc="Filtering labels")):
				if not labels:
						results.append([])
						continue
				
				try:
						# Dynamic batch size based on label count and length
						avg_label_len = sum(len(l) for l in labels) / len(labels)
						dynamic_batch_size = max(1, min(
								label_batch_size,
								int(512 / avg_label_len) if avg_label_len > 0 else label_batch_size
						))
						
						label_embeddings = safe_encode(labels, is_labels=True)
						
						# Efficient similarity calculation
						text_emb_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
						label_emb_norm = label_embeddings / (np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-8)
						similarities = np.dot(label_emb_norm, text_emb_norm)
						
						relevant_indices = np.where(similarities > threshold)[0]
						results.append([labels[idx] for idx in relevant_indices])
						
						# Periodic memory cleanup
						if i % 100 == 0:
								torch.cuda.empty_cache()
								
				except Exception as e:
						print(f"Error processing labels for text {i}: {str(e)[:200]}")
						results.append([])
		
		return results

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
	):
	if verbose:
		print(f"Semi-Supervised visual-based annotation (using VLM) batch_size: {batch_size} num_workers: {num_workers}".center(160, "-"))
	
	visual_based_annotation_start_time = time.time()
	
	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	texts = [f"This is a photo of {lbl}." for lbl in candidate_labels]
	
	gpu_name = torch.cuda.get_device_name(device)
	total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
	available_gpu_memory = torch.cuda.mem_get_info()[0] / 1024**3 # GB
	print(
		f"GPU: {gpu_name} "
		f"[Memory] Total: {total_gpu_memory:.2f} GB "
		f"Available: {available_gpu_memory:.2f} GB"
	)
	
	model = AutoModel.from_pretrained(
		pretrained_model_name_or_path=vlm_model_name,
		torch_dtype=torch.float16 if available_gpu_memory < 7 else torch.float32,
		device_map=device,
	).eval()
	
	model = torch.compile(model, mode="max-autotune")
	
	processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=vlm_model_name)
	
	# Precompute text embeddings
	with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
		text_inputs = processor(
			text=texts,
			padding="max_length",
			max_length=64,
			return_tensors="pt",
		).to(device)
		text_embeddings = model.get_text_features(**text_inputs)
		text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
	
	df = pd.read_csv(
		filepath_or_buffer=csv_file, 
		on_bad_lines='skip', 
		dtype=dtypes, 
		low_memory=False,
	)
	img_paths = df['img_path'].tolist()
	combined_labels = [[] for _ in range(len(img_paths))]
	
	# Process images using DataLoader
	dataset = HistoricalArchives(img_paths)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size, 
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=num_workers>1,
		prefetch_factor=2, # better overlap
		drop_last=False,  # Explicitly set to process all data
		collate_fn=custom_collate_fn
	)

	print(f"Processing {len(img_paths)} images in {batch_size} batches...")
	for batch_idx, (batch_indices, images) in enumerate(tqdm(dataloader, desc="Processing images")):
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
				
				for processed_idx, global_idx in enumerate(batch_indices[:len(images)]):
					topk_probs, topk_indices = similarities[processed_idx].topk(topk)
					combined_labels[global_idx] = [candidate_labels[idx] for idx in topk_indices]
			if batch_idx % 20 == 0:
				torch.cuda.empty_cache()
		except Exception as e:
			print(f"ERROR: failed to process batch {batch_indices[0]}-{batch_indices[-1]}: {e}")
			torch.cuda.empty_cache()
	
	df['visual_based_labels'] = combined_labels
	df.to_csv(metadata_fpth, index=False)
	
	print(f"Processed {len(img_paths)} images, generated {sum(1 for labels in combined_labels if labels)} valid results")
	print(f"Visual-based annotation Elapsed time: {time.time() - visual_based_annotation_start_time:.2f} sec".center(160, " "))
	return combined_labels

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		topk: int = 3,
		verbose: bool = True,
	):
	if verbose:
		print(f"Semi-Supervised textual-based annotation batch_size: {batch_size} num_workers: {num_workers}".center(160, "-"))
	
	start_time = time.time()
	
	# Load model with memory optimizations
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name}...")
	sent_model = SentenceTransformer(
		model_name_or_path=st_model_name,
		device=device,
		trust_remote_code=True
	)
	sent_model.eval()
	
	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	
	# Pre-compute label embeddings once
	if verbose:
		print(f"Pre-computing embeddings for {len(candidate_labels)} pre-defined labels...")
	t0 = time.time()
	label_embs = sent_model.encode(
		candidate_labels,
		batch_size=min(batch_size, len(candidate_labels)),
		convert_to_tensor=True,
		normalize_embeddings=True,
		show_progress_bar=False,
	)
	if verbose:
		print(f"Label embeddings: {type(label_embs)} {label_embs.shape} computed in {time.time() - t0:.2f} sec")
	
	if verbose:
		print(f"Loading dataframe: {csv_file}...")
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
	chunk_size = min(1000, len(df))
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
		text_batch_size = min(16, batch_size)
		text_embs = []
		
		for i in range(0, len(texts_to_process), text_batch_size):
			batch_texts = texts_to_process[i:i+text_batch_size]
			try:
				batch_embs = sent_model.encode(
					batch_texts,
					batch_size=text_batch_size,
					convert_to_tensor=True,
					normalize_embeddings=True,
					show_progress_bar=False
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
		
		# Store results only for valid indices
		for i, original_idx in enumerate(valid_indices):
			try:
				labels = [candidate_labels[j] for j in topk_indices[i] if j < len(candidate_labels)]
				scores = [round(s.item(), 4) for s in topk_scores[i]][:len(labels)]
				df.at[original_idx, 'textual_based_labels'] = labels
				df.at[original_idx, 'textual_based_scores'] = scores
			except Exception as e:
				print(f"Error processing sample {original_idx}: {str(e)[:200]}")
				df.at[original_idx, 'textual_based_labels'] = []
				df.at[original_idx, 'textual_based_scores'] = []
		
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

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=128, help="Batch size for textual processing")
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=4, help="Batch size for vision processing")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="all-MiniLM-L12-v2", choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "jinaai/jina-embeddings-v3", "paraphrase-multilingual-MiniLM-L12-v2"], help="Sentence-transformer model name")
	parser.add_argument("--vlm_model_name", '-vlm', type=str, default="google/siglip2-so400m-patch16-naflex", choices=["kakaobrain/align-base", "google/siglip2-so400m-patch16-naflex"], help="Vision-Language model name")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)

	set_seeds(seed=42)
	DATASET_DIRECTORY = os.path.dirname(args.csv_file)
	text_output_path = os.path.join(DATASET_DIRECTORY, "metadata_textual_based_labels.csv")
	vision_output_path = os.path.join(DATASET_DIRECTORY, "metadata_visual_based_labels.csv")
	combined_output_path = os.path.join(DATASET_DIRECTORY, "metadata_multimodal.csv")
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
			num_workers=args.num_workers,
			batch_size=args.text_batch_size,
			st_model_name=args.sentence_model_name,
			metadata_fpth=text_output_path,
			device=args.device,
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
			verbose=True,
		)
	
	assert len(textual_based_labels) == len(visual_based_labels), "Label lists must have same length"
	
	if os.path.exists(combined_output_path):
		print(f"Found existing combined labels at {combined_output_path} Loading...")
		recompute = input("Do you want to recompute the combined labels? (y/n): ").lower() == 'y'
		if not recompute:
			print("Using existing combined labels.")
			combined_df = pd.read_csv(combined_output_path)
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
			return combined_labels

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
	
	df = pd.read_csv(
		filepath_or_buffer=args.csv_file,
		on_bad_lines='skip',
		dtype=dtypes,
		low_memory=False,
	)
	
	# Convert empty lists to None for better CSV representation
	df['textual_based_labels'] = [labels if labels else None for labels in textual_based_labels]
	df['visual_based_labels'] = [labels if labels else None for labels in visual_based_labels]
	df['multimodal_labels'] = [labels if labels else None for labels in combined_labels]
	
	print(f"Saving results to {combined_output_path}...")
	df.to_csv(combined_output_path, index=False)
	try:
		df.to_excel(combined_output_path.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	perform_multilabel_eda(
		data_path=combined_output_path, 
		label_column='multimodal_labels',
	)

	train_df, val_df = get_multi_label_stratified_split(
		df=df,
		val_split_pct=0.35,
		seed=42,
		label_col='multimodal_labels'
	)
	print("\n--- Split Results ---")
	print(f"Train set shape: {train_df.shape}")
	print(f"Validation set shape: {val_df.shape}")

	train_df.to_csv(combined_output_path.replace('.csv', '_train.csv'), index=False)
	val_df.to_csv(combined_output_path.replace('.csv', '_val.csv'), index=False)

	return combined_labels

if __name__ == "__main__":
	multiprocessing.set_start_method('spawn', force=True)
	main()