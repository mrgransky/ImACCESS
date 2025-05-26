from utils import *
torch.set_grad_enabled(False)

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata.csv -d "cuda:1" -nw 8 -tbs 512 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_SMU.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 24 -tbs 512 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata.csv -d "cuda:1" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata.csv -d "cuda:2" -nw 20 -tbs 512 -vbs 32 -vth 0.3 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -d "cuda:3" -nw 20 -tbs 512 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.out &

# Make language detection deterministic
DetectorFactory.seed = 42

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
		"used", "southern", "built", "album", "album cover",
		"original", "information", "item", "http", "www", "jpg", "000",
		"jpeg", "png", "gif", "bmp", "tiff", "tif", "ico", "svg", "webp", "heic", "heif", "raw", "cr2", "nef", "orf", "arw", "dng", "nrw", "k25", "kdc", "rw2", "raf", "mrw", "pef", "sr2", "srf",
	}
)

METADATA_PATTERNS = [
	r'bildetekst \w+',        # Matches 'bildetekst german' and similar
	r'kunststoff \w+',        # Matches photography material descriptions
	r'arkivreferanse \w+',    # Archive references
	r'\w+ pa \d+',            # Reference codes like 'ra pa-1209'
	r'donated \w+',           # Donation information
	r'information received',  # Source information
	r'\w+ association',       # Organization suffixes without context
]

FastText_Language_Identification = "lid.176.bin"
if FastText_Language_Identification not in os.listdir():
	url = f"https://dl.fbaipublicfiles.com/fasttext/supervised-models/{FastText_Language_Identification}"
	urllib.request.urlretrieve(url, FastText_Language_Identification)

# Semantic categories for organization
SEMANTIC_CATEGORIES = {
	'military': ['military', 'army', 'navy', 'air force', 'soldier', 'officer', 'troop', 'regiment', 'division', 'corps', 'battalion', 'brigade'],
	'political': ['government', 'parliament', 'president', 'prime minister', 'minister', 'official', 'politician', 'leader'],
	'event': ['war', 'battle', 'attack', 'invasion', 'liberation', 'occupation', 'revolution', 'protest', 'march', 'ceremony'],
	'location': ['city', 'town', 'village', 'country', 'region', 'territory', 'front', 'border', 'base', 'camp'],
	'vehicle': ['tank', 'aircraft', 'plane', 'ship', 'submarine', 'boat', 'truck', 'car', 'jeep', 'vehicle'],
	'weapon': ['gun', 'rifle', 'cannon', 'artillery', 'weapon', 'bomb', 'missile', 'ammunition'],
}

# Entity types for NER
RELEVANT_ENTITY_TYPES = {
	'PERSON',
	'ORG',
	'GPE',
	'LOC',
	'NORP',
	'FAC',
	'EVENT',
	'WORK_OF_ART',
	'PRODUCT',
	'DATE',
	'TIME',
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

def extract_semantic_topics(
		sent_model: SentenceTransformer,
		ft_model: fasttext.FastText._FastText,
		texts: List[str],
		dataset_dir: str,
		num_workers: int,
		enable_visualizations: bool = False,
	) -> Tuple[List[List[str]], Set[str]]:
	
	vectorizer_model = CountVectorizer(
		ngram_range=(1, 3),
		stop_words=list(CUSTOM_STOPWORDS),
	)
	ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

	representation_model = {
		"KeyBERT": KeyBERTInspired(),
		"MMR": MaximalMarginalRelevance(diversity=0.5),
		# prefer nouns and adjectives
		"POS": PartOfSpeech("en_core_web_sm", pos_patterns=[[{"POS": "NOUN"}, {"POS": "ADJ"}]])
	}

	print(f"Creating BERTopic model for {len(texts)} texts...")
	topic_model = BERTopic(
		embedding_model=sent_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_model,
		min_topic_size=max(5, min(15, len(texts)//1000)),
		calculate_probabilities=True,
		nr_topics="auto",
		verbose=True,
	)
	
	topics, probs = topic_model.fit_transform(texts)
	topic_info = topic_model.get_topic_info()
	print(f"Number of topics: {len(topic_info)}")
	print(topic_info)
	print(f"Unique Topic IDs: {topic_info['Topic'].unique()}")

	filtered_topics = []
	for topic_id in topic_info['Topic'].unique():
		if topic_id == -1:  # Skip outlier topic
			continue
		topic_words = [
			word 
			for word, score in topic_model.get_topic(topic_id) 
			if is_likely_english_term(word) and score > 0.05
		]
		if topic_words:
			filtered_topics.append(topic_words[:15])
	flat_topics = set(word for topic in filtered_topics for word in topic)
	return filtered_topics, flat_topics

def clean_labels(labels):
		cleaned = set()
		# Predefined list of valid historical non-ASCII terms
		VALID_HISTORICAL_TERMS = {
				"blitzkrieg", "kübelwagen", "wehrmacht", "panzer", "luftwaffe",
				"stuka", "t-34", "afrika korps"
		}
		for label in labels:
				# Normalize the label
				label = label.lower().strip()
				# Remove non-alphanumeric characters except spaces and hyphens
				label = re.sub(r"[^a-z0-9\s\-äöüßñéèêëìíîïçåæœø]", "", label)
				# Skip short labels, stopwords, or invalid labels
				if label in CUSTOM_STOPWORDS or len(label) < 3:
						continue
				# Skip labels that are just numbers
				if label.isdigit():
						continue
				# Skip labels that start with numbers unless they're years (4 digits)
				if label[0].isdigit() and not (len(label) == 4 and label.isdigit()):
						continue
				# Allow non-ASCII labels if proper noun or in valid historical terms
				if not all(ord(char) < 128 for char in label):
						if not (label[0].isupper() or label in VALID_HISTORICAL_TERMS):
								continue
				cleaned.add(label)
		return sorted(cleaned)

def extract_named_entities(
		nlp: pipeline, 
		text: str, 
		ft_model: fasttext.FastText._FastText,
		confidence_threshold: float = 0.7
):
		if not is_english(text=text, ft_model=ft_model):
				return []
		
		try:
				ner_results = nlp(text)
				entities = []
				
				for entity in ner_results:
						# Filter by confidence score
						if entity.get("score", 0) < confidence_threshold:
								continue
								
						if entity["entity_group"] in RELEVANT_ENTITY_TYPES:
								entity_text = entity["word"].strip()
								
								# Better entity normalization
								if len(entity_text) > 2 and entity_text.isalpha():
										# Preserve proper nouns (capitalized entities)
										if entity_text[0].isupper():
												normalized = entity_text.lower()
										else:
												normalized = entity_text.lower()
										
										if (normalized not in CUSTOM_STOPWORDS and 
												not normalized.isdigit() and
												len(normalized) >= 3):
												entities.append(normalized)
				
				# Add multi-word entities (phrases that appear as complete units)
				doc_words = text.lower().split()
				for i in range(len(doc_words) - 1):
						bigram = f"{doc_words[i]} {doc_words[i+1]}"
						if (doc_words[i] not in CUSTOM_STOPWORDS and 
								doc_words[i+1] not in CUSTOM_STOPWORDS and
								len(doc_words[i]) > 2 and len(doc_words[i+1]) > 2):
								entities.append(bigram)
				
				return list(set(entities))
				
		except Exception as e:
				print(f"NER error: {e}")
				return []

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

def handle_multilingual_labels(labels):
	processed_labels = []
	
	for label in labels:
		# Skip non-ASCII labels completely
		if not all(ord(char) < 128 for char in label):
			continue
				
		words = label.split()
		
		# Single word label
		if len(words) == 1:
			if is_likely_english_term(label) or label[0].isupper():
				processed_labels.append(label)
						
		# Multi-word label
		else:
			# Keep if all words are likely English or proper nouns
			if all(is_likely_english_term(word) or word[0].isupper() for word in words):
				processed_labels.append(label)

	return processed_labels

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

def balance_label_count(
		image_labels_list,
		text_descriptions,
		sent_model,
		min_labels=1,
		max_labels=12,
		similarity_threshold=0.5
	):
	"""
	Balance the number of labels per image, prioritizing multi-word labels and ensuring semantic coherence.
	
	Args:
			image_labels_list: List of lists of labels
			text_descriptions: List of text descriptions for relevance scoring
			sent_model: SentenceTransformer model for embeddings
			min_labels: Minimum number of labels per image
			max_labels: Maximum number of labels per image
			similarity_threshold: Minimum cosine similarity for compound labels
	
	Returns:
			List of lists of balanced labels
	"""
	balanced_labels = []
	
	# Encode text descriptions once
	text_embeds = sent_model.encode(text_descriptions, show_progress_bar=False, convert_to_numpy=True)
	
	for idx, labels in tqdm(enumerate(image_labels_list), total=len(image_labels_list), desc="Label Balancing"):
			text_emb = text_embeds[idx]
			
			# Case 1: Too few labels - generate coherent compounds
			if len(labels) < min_labels:
					compound_labels = []
					if len(labels) >= 2:
							# Encode existing labels
							label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
							for i in range(len(labels)):
									for j in range(i+1, len(labels)):
											compound = f"{labels[i]} {labels[j]}"
											if 2 <= len(compound.split()) <= 3:  # 2-3 word compounds
													# Compute similarity of compound to text
													compound_emb = sent_model.encode(compound, show_progress_bar=False)
													similarity = np.dot(compound_emb, text_emb) / (
															np.linalg.norm(compound_emb) * np.linalg.norm(text_emb) + 1e-8
													)
													if similarity > similarity_threshold:
															compound_labels.append(compound)
					
					# Sort compounds by similarity
					if compound_labels:
							compound_embs = sent_model.encode(compound_labels, show_progress_bar=False, convert_to_numpy=True)
							similarities = np.dot(compound_embs, text_emb) / (
									np.linalg.norm(compound_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
							)
							compound_labels = [compound_labels[i] for i in np.argsort(similarities)[::-1]]
					
					# Add compounds to reach min_labels
					expanded_labels = labels + compound_labels[:min_labels - len(labels)]
					balanced_labels.append(expanded_labels[:min_labels] if len(expanded_labels) >= min_labels else expanded_labels)
			
			# Case 2: Too many labels - prioritize multi-word by relevance
			elif len(labels) > max_labels:
					# Encode labels and compute similarities
					label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
					similarities = np.dot(label_embs, text_emb) / (
							np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
					)
					
					# Separate multi-word and single-word labels
					multi_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' in label]
					single_word = [(label, sim) for label, sim in zip(labels, similarities) if ' ' not in label]
					
					# Sort by similarity
					multi_word.sort(key=lambda x: x[1], reverse=True)
					single_word.sort(key=lambda x: x[1], reverse=True)
					
					# Prioritize multi-word, fill with single-word if needed
					selected_labels = [label for label, _ in multi_word[:max_labels]]
					if len(selected_labels) < max_labels:
							selected_labels.extend([label for label, _ in single_word[:max_labels - len(selected_labels)]])
					
					balanced_labels.append(selected_labels)
			
			# Case 3: Good label count - keep as is
			else:
					balanced_labels.append(labels)
	
	return balanced_labels

def quick_filter_candidates(
		text: str, 
		labels: list[str], 
		sent_model: SentenceTransformer, 
		max_keep: int=30
	) -> list[str]:
	if not labels:
		print(f"<!> Empty labels list: {text}")
		return []
	text_emb = sent_model.encode(text, show_progress_bar=False)
	label_embs = sent_model.encode(labels, show_progress_bar=False)
	# Ensure label_embs is not empty (redundant but defensive)
	if label_embs.size == 0:
		print(f"<!> No embeddings generated for labels: {labels}")
		return []
	similarities = np.dot(label_embs, text_emb) / (np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8)
	sorted_labels = [label for _, label in sorted(zip(similarities, labels), reverse=True)]
	return sorted_labels[:max_keep]

def process_document_chunk(chunk_data):
	"""Process a chunk of documents in parallel"""
	start_idx, end_idx, texts, combined_labels = chunk_data
	
	# Initialize sent_model within the process
	local_model = SentenceTransformer("all-MiniLM-L6-v2")
	
	results = []
	for i in range(start_idx, end_idx):
		idx = i - start_idx  # Local index within chunk
		text = texts[idx]
		labels = combined_labels[idx]
		
		# Quick filter first
		filtered_labels = quick_filter_candidates(text, labels)
		
		if not filtered_labels:
				results.append([])
				continue
				
		# Encode text and labels
		text_emb = local_model.encode(text)
		label_embs = local_model.encode(filtered_labels)
		
		# Calculate similarities
		similarities = []
		for label_emb in label_embs:
				sim = np.dot(text_emb, label_emb) / (
						np.linalg.norm(text_emb) * np.linalg.norm(label_emb) + 1e-8
				)
				similarities.append(sim)
				
		# Filter by threshold
		threshold = 0.3
		relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
		results.append([filtered_labels[i] for i in relevant_indices])
	
	return results

def get_all_unique_user_queries(df: pd.DataFrame) -> Set[str]:
		"""Collects all unique and cleaned user_query terms from the entire DataFrame."""
		unique_queries = set()
		if 'user_query' in df.columns:
				for query in df['user_query'].fillna('').astype(str):
						if query.strip():
								# Simple cleaning for user query before adding to global pool
								cleaned_query = re.sub(r'[^\w\s\-]', '', query.lower()).strip()
								if cleaned_query and cleaned_query not in CUSTOM_STOPWORDS and len(cleaned_query) > 2:
										unique_queries.add(cleaned_query)
		return unique_queries

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

def post_process_labels(labels, text_description, sent_model, doc_year, vlm_scores, max_labels=10, similarity_threshold=0.8):
		"""
		Post-process visual labels to remove duplicates, validate temporally, and rank by relevance.
		
		Args:
				labels: List of labels
				text_description: String enriched document description
				sent_model: SentenceTransformer model
				doc_year: Float or int document year
				vlm_scores: List of VLM similarity scores
				max_labels: Maximum number of labels to retain
				similarity_threshold: Cosine similarity threshold for deduplication
		
		Returns:
				List of processed labels
		"""
		if not labels:
				return []
		
		# Validate vlm_scores length
		if len(vlm_scores) != len(labels):
				vlm_scores = vlm_scores + [0.0] * (len(labels) - len(vlm_scores)) if len(vlm_scores) < len(labels) else vlm_scores[:len(labels)]
		
		# Encode labels and text
		label_embs = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
		text_emb = sent_model.encode(text_description, show_progress_bar=False, convert_to_numpy=True)
		
		# Deduplicate based on semantic similarity
		deduplicated = []
		for i, (label, score) in enumerate(zip(labels, vlm_scores)):
				is_redundant = False
				for kept_label, kept_emb, _ in deduplicated:
						sim = np.dot(label_embs[i], kept_emb) / (np.linalg.norm(label_embs[i]) * np.linalg.norm(kept_emb) + 1e-8)
						if sim > similarity_threshold:
								is_redundant = True
								break
				if not is_redundant:
						deduplicated.append((label, label_embs[i], score))
		
		# Temporal validation
		validated = [
				(label, emb, score)
				for label, emb, score in deduplicated
				if is_year_compatible(label, doc_year)
		]
		
		# Rank by combined VLM and text similarity
		if validated:
				text_sims = np.dot(np.array([emb for _, emb, _ in validated]), text_emb) / (
						np.linalg.norm([emb for _, emb, _ in validated], axis=1) * np.linalg.norm(text_emb) + 1e-8
				)
				combined_scores = [0.6 * vlm_score + 0.4 * text_sim for vlm_score, text_sim in zip(
						[score for _, _, score in validated], text_sims
				)]
				ranked = [label for _, label in sorted(zip(combined_scores, [label for label, _, _ in validated]), reverse=True)]
				return ranked[:max_labels]
		
		return []

def process_category_batch(
		batch_paths, batch_descriptions, batch_indices, df, categories, prompt_embeds,
		category_type, sent_model, processor, model, device, verbose, base_thresholds, sub_batch_size
):
		"""
		Process a batch of images for a specific category type with adaptive thresholding and sparse metadata handling.
		
		Args:
				batch_paths: List of image file paths
				batch_descriptions: List of enriched document descriptions
				batch_indices: List of global indices for the batch
				df: DataFrame with metadata
				categories: List of category strings
				prompt_embeds: Pre-computed prompt embeddings
				category_type: String ('object', 'scene', 'era', 'activity')
				sent_model: SentenceTransformer model
				processor: ALIGN processor
				model: ALIGN model
				device: Torch device
				verbose: Bool for logging
				base_thresholds: Dict of base thresholds per category type
				sub_batch_size: Size of sub-batches for GPU memory management
		
		Returns:
				batch_results: List of lists of selected categories
				batch_scores: List of lists of VLM scores
		"""
		valid_images = []
		valid_indices = []
		valid_descriptions = []
		failed_images = []

		# Handle sparse metadata with fallback to title
		user_queries = df['user_query'].fillna('').iloc[batch_indices].tolist() if 'user_query' in df.columns else [''] * len(batch_indices)
		titles = df['title'].fillna('').iloc[batch_indices].tolist() if 'title' in df.columns else [''] * len(batch_indices)

		# Validate images and descriptions
		for i, path in enumerate(batch_paths):
			try:
				if os.path.exists(path):
					img = Image.open(path).convert('RGB')
					desc = batch_descriptions[i].strip()
					if not desc and user_queries[i].strip():
						desc = user_queries[i]
					if not desc and titles[i].strip():
						desc = titles[i]
					if len(desc.strip()) < 10:
						desc = ""  # Treat as sparse
					valid_images.append(img)
					valid_indices.append(i)
					valid_descriptions.append(desc)
				else:
					failed_images.append(path)
			except Exception as e:
				failed_images.append(path)
				if verbose:
					print(f"Error loading image {path}: {e}")

		if not valid_images:
			if verbose and failed_images:
				print(f"Failed to load {len(failed_images)} images in batch")
			return [[] for _ in range(len(batch_paths))], [[] for _ in range(len(batch_paths))]

		# Add dynamic categories from user_query
		extended_categories = categories.copy()
		new_queries = []
		for query in user_queries:
			if isinstance(query, str) and query.strip() and query not in extended_categories:
				# extended_categories.append(query)
				# new_queries.append(query)
				try:
					# Try to parse as a list
					parsed_query = ast.literal_eval(query) if query.startswith('[') and query.endswith(']') else [query]
					for q in parsed_query:
						if isinstance(q, str) and q.strip() and q not in extended_categories:
							extended_categories.append(q)
							new_queries.append(q)
				except (ValueError, SyntaxError):
					# Treat as single string if parsing fails
					if query not in extended_categories:
						extended_categories.append(query)
						new_queries.append(query)

		# Update prompt embeddings
		extended_prompt_embeds = prompt_embeds
		if new_queries:
			new_prompts = [f"a photo of {q}" for q in new_queries]
			new_embeds = sent_model.encode(new_prompts, device=device, convert_to_tensor=True, show_progress_bar=False)
			extended_prompt_embeds = torch.cat([prompt_embeds, new_embeds], dim=0)

		# Compute text similarities
		text_prompts = [f"a photo of {cat}" for cat in extended_categories]
		with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			text_embeds = sent_model.encode(valid_descriptions, device=device, convert_to_tensor=True, show_progress_bar=False)
			text_similarities = torch.mm(extended_prompt_embeds, text_embeds.T).cpu().numpy()
			prompt_norms = torch.norm(extended_prompt_embeds, dim=1, keepdim=True)
			text_norms = torch.norm(text_embeds, dim=1, keepdim=True)
			text_similarities = text_similarities / (torch.mm(prompt_norms, text_norms.T).cpu().numpy() + 1e-8)
			model.eval()
			batch_results = [[] for _ in range(len(batch_paths))]
			batch_scores = [[] for _ in range(len(batch_paths))]
			# Process sub-batches
			for sub_idx in range(0, len(valid_images), sub_batch_size):
				sub_end = min(sub_idx + sub_batch_size, len(valid_images))
				sub_images = valid_images[sub_idx:sub_end]
				sub_valid_indices = valid_indices[sub_idx:sub_end]
				inputs = processor(
					text=text_prompts,
					images=sub_images,
					padding="max_length",
					return_tensors="pt",
				).to(device)
				outputs = model(**inputs)
				image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
				vlm_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
				vlm_similarity = (100.0 * image_embeds @ vlm_embeds.T).softmax(dim=-1).cpu().numpy()
				threshold = base_thresholds[category_type]
				for i, img_idx in enumerate(range(sub_idx, sub_end)):
					batch_idx = sub_valid_indices[i]
					local_img_idx = img_idx - sub_idx
					# Entropy-based threshold adjustment
					img = sub_images[i]
					img_array = np.array(img.convert('L'))
					img_array = resize(img_array, (128, 128), anti_aliasing=True, preserve_range=True).astype(np.uint8)
					entropy = shannon_entropy(img_array)
					complexity_factor = max(0.7, 1.0 - 0.15 * (entropy / 8.0))
					adaptive_threshold = max(0.15, threshold * complexity_factor)
					is_sparse = not valid_descriptions[img_idx].strip()
					for cat_idx, cat in enumerate(extended_categories):
						# Moved text_score computation inside the loop
						text_score = text_similarities[cat_idx, img_idx]
						adjusted_threshold = adaptive_threshold
						if text_score > 0.5:
							adjusted_threshold *= 0.8
						elif text_score < 0.1:
							adjusted_threshold *= 1.2
						vlm_score = vlm_similarity[local_img_idx, cat_idx]
						combined_score = (
							0.5 * vlm_score + 0.5 * text_score if is_sparse
							else vlm_score * (0.8 + 0.2 * text_score)
						)
						if combined_score > adjusted_threshold:
							batch_results[batch_idx].append(cat)
							batch_scores[batch_idx].append(vlm_score)

		return batch_results, batch_scores

def combine_and_clean_labels(
		ner_labels: List[str], 
		keywords: List[str], 
		topic_labels: List[str], 
		user_query: str, 
		text: str, 
		sent_model: SentenceTransformer,
		relevance_threshold: float = 0.35
	):
	
	# Parse user query more robustly
	user_terms = []
	if user_query and isinstance(user_query, str):
		try:
			if user_query.startswith('[') and user_query.endswith(']'):
				parsed = ast.literal_eval(user_query)
				user_terms = [str(term).strip().lower() for term in parsed if str(term).strip()]
			else:
				# Split on common delimiters
				user_terms = [term.strip().lower() for term in re.split(r'[,;|]', user_query) if term.strip()]
		except:
			user_terms = [user_query.strip().lower()]
	
	# Combine all label sources with weights
	weighted_labels = []
	
	# Higher weight for user queries (most reliable)
	for term in user_terms:
		if term and len(term) > 2 and term not in CUSTOM_STOPWORDS:
			weighted_labels.append((term, 1.0, 'user'))
	
	# Medium weight for NER (reliable entities)
	for label in ner_labels:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.8, 'ner'))
	
	# Lower weight for keywords and topics
	for label in keywords:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.6, 'keyword'))
	
	for label in topic_labels:
		if label and len(label) > 2:
			weighted_labels.append((label, 0.4, 'topic'))
	
	if not weighted_labels:
		return []
	
	# Semantic clustering to group similar labels
	labels_only = [label for label, _, _ in weighted_labels]
	label_embeddings = sent_model.encode(labels_only, show_progress_bar=False)
	text_embedding = sent_model.encode(text, show_progress_bar=False)
	
	# Calculate relevance to original text
	relevance_scores = np.dot(label_embeddings, text_embedding) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_embedding) + 1e-8)
	
	# Filter by relevance and deduplicate semantically
	final_labels = []
	used_embeddings = []
	
	# Sort by combined score (weight * relevance)
	combined_scores = [
		(label, weight * relevance_scores[i], source) 
		for i, (label, weight, source) in enumerate(weighted_labels)
	]
	combined_scores.sort(key=lambda x: x[1], reverse=True)
	
	for label, score, source in combined_scores:
		if score < relevance_threshold:
			continue

		# Check semantic similarity with already selected labels
		label_emb = sent_model.encode(label, show_progress_bar=False)
		is_redundant = False
		
		for used_emb in used_embeddings:
			similarity = np.dot(label_emb, used_emb) / (
					np.linalg.norm(label_emb) * np.linalg.norm(used_emb) + 1e-8
			)
			if similarity > 0.85:  # High similarity threshold
				is_redundant = True
				break
		
		if not is_redundant:
			final_labels.append(label)
			used_embeddings.append(label_emb)
			
		if len(final_labels) >= 10:  # Limit number of labels
			break
	
	return final_labels

def combine_and_clean_labels_old(ner_labels, keywords, topic_labels, user_query, text, sent_model, min_threshold=0.4, max_threshold=0.7):
		"""
		Combine labels from NER, keywords, topics, and user_query, splitting concatenated phrases and ensuring coherence.
		
		Args:
				ner_labels: List of NER-extracted labels
				keywords: List of KeyBERT-extracted keywords
				topic_labels: List of topic-derived labels
				user_query: User-provided query (string or list)
				text: Original text for context
				sent_model: SentenceTransformer model
				min_threshold: Minimum similarity threshold for label retention
				max_threshold: Maximum similarity threshold for deduplication
		
		Returns:
				List of cleaned, unique labels
		"""
		# Initialize combined labels
		combined = set(ner_labels + keywords + topic_labels)
		
		# Parse user_query
		if isinstance(user_query, str) and user_query.strip():
				try:
						parsed_query = ast.literal_eval(user_query) if user_query.startswith('[') else [user_query]
						combined.update(str(q).strip().lower() for q in parsed_query if isinstance(q, str) and q.strip())
				except (ValueError, SyntaxError):
						combined.add(user_query.strip().lower())
		
		# Remove stopwords and invalid labels
		combined = [label for label in combined if label.lower() not in CUSTOM_STOPWORDS and len(label.split()) <= 2]
		
		if not combined:
				return []
		
		# Split concatenated phrases
		split_labels = []
		for label in combined:
				words = label.split()
				if len(words) > 2:
						# Split into 1-2 word phrases
						for i in range(0, len(words), 2):
								phrase = " ".join(words[i:i+2])
								if phrase and any(word in text.lower() for word in phrase.split()):
										split_labels.append(phrase)
						# Add single words if not covered
						for word in words:
								if word not in CUSTOM_STOPWORDS and word in text.lower():
										split_labels.append(word)
				else:
						split_labels.append(label)
		
		# Remove duplicates
		split_labels = list(set(split_labels))
		
		# Encode text and labels
		text_emb = sent_model.encode(text, show_progress_bar=False)
		label_embs = sent_model.encode(split_labels, show_progress_bar=False)
		
		# Filter by relevance to text
		similarities = np.dot(label_embs, text_emb) / (
				np.linalg.norm(label_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
		)
		relevant_labels = [split_labels[i] for i in range(len(split_labels)) if similarities[i] > min_threshold]
		
		if not relevant_labels:
				return []
		
		# Semantic deduplication
		label_embs = sent_model.encode(relevant_labels, show_progress_bar=False)
		deduplicated = []
		for i, (label, sim) in enumerate(zip(relevant_labels, similarities)):
				is_redundant = False
				for kept_label, _, kept_emb in deduplicated:
						sim_to_kept = np.dot(label_embs[i], kept_emb) / (
								np.linalg.norm(label_embs[i]) * np.linalg.norm(kept_emb) + 1e-8
						)
						if sim_to_kept > max_threshold or label in kept_label or kept_label in label:
								is_redundant = True
								break
				if not is_redundant and any(word in text.lower() for word in label.split()):
						deduplicated.append((label, sim, label_embs[i]))
		
		# Sort by similarity and limit to max 10 labels
		deduplicated.sort(key=lambda x: x[1], reverse=True)
		final_labels = [label for label, _, _ in deduplicated[:10]]
		
		return sorted(set(final_labels))

def batch_filter_by_relevance(
		sent_model: SentenceTransformer,
		texts: List[str],
		all_labels_list: List[List[str]],
		threshold: float,
		batch_size: int,
):
		"""Memory-efficient relevance filtering with adaptive batching"""
		
		results = []
		
		# Pre-encode all texts once
		print("Pre-encoding texts...")
		try:
				text_embeddings = sent_model.encode(
						texts, 
						batch_size=batch_size,
						show_progress_bar=False,
						convert_to_numpy=True
				)
		except torch.cuda.OutOfMemoryError:
				# Fallback to smaller batches
				text_embeddings = []
				for i in tqdm(range(0, len(texts), batch_size // 2), desc="Encoding texts (small batches)"):
						batch = texts[i:i + batch_size // 2]
						batch_emb = sent_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
						text_embeddings.extend(batch_emb)
				text_embeddings = np.array(text_embeddings)
		
		# Process labels efficiently
		for i, (text_emb, labels) in enumerate(tqdm(zip(text_embeddings, all_labels_list), total=len(texts), desc="Filtering labels")):
				if not labels:
						results.append([])
						continue
				
				# Dynamic batch size based on number of labels
				label_batch_size = min(len(labels), batch_size)
				if len(labels) > 100:  # Large label sets
						label_batch_size = batch_size // 4
				
				try:
						label_embeddings = sent_model.encode(
								labels, 
								batch_size=label_batch_size,
								show_progress_bar=False,
								convert_to_numpy=True
						)
						
						similarities = np.dot(label_embeddings, text_emb) / (
								np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8
						)
						
						relevant_indices = np.where(similarities > threshold)[0]
						results.append([labels[idx] for idx in relevant_indices])
						
				except Exception as e:
						print(f"Error processing labels for text {i}: {e}")
						results.append([])
		
		return results

def deduplicate_labels(labels):
		"""
		Deduplicate labels based on semantic similarity and substring containment.
		
		Args:
				labels: List of labels
		
		Returns:
				List of deduplicated labels
		"""
		if not labels:
				return []
		
		# Substring and semantic deduplication
		deduplicated = []
		for label in sorted(labels, key=len, reverse=True):
				if not any(
						label in kept_label or kept_label in label or label.lower() == kept_label.lower()
						for kept_label in deduplicated
				):
						deduplicated.append(label)
		
		return sorted(deduplicated)

def get_keywords(
		text: str, 
		sent_model: SentenceTransformer, 
		rake: Rake,
	):
	rake.extract_keywords_from_text(text)
	ranked_phrases = rake.get_ranked_phrases()

	kw_model = KeyBERT(model=sent_model)	
	keybert_keywords = kw_model.extract_keywords(
		text,
		keyphrase_ngram_range=(1, 3),
		stop_words=list(CUSTOM_STOPWORDS),
		top_n=15,
		use_mmr=True,  # Use Maximal Marginal Relevance for diversity
		diversity=0.7
	)
	
	# Combine and rank by relevance to text
	all_candidates = list(set(ranked_phrases[:10] + [kw[0] for kw in keybert_keywords]))
	if not all_candidates:
		print("No keywords extracted, returning empty list")
		return []

	# Filter and score candidates
	text_embedding = sent_model.encode(text, show_progress_bar=False)
	keyword_embeddings = sent_model.encode(all_candidates, show_progress_bar=False)
	if keyword_embeddings.size == 0 or text_embedding.size == 0:
		print("Empty keyword embeddings, returning empty list")
		return []
	similarities = np.dot(keyword_embeddings, text_embedding) / (np.linalg.norm(keyword_embeddings, axis=1) * np.linalg.norm(text_embedding) + 1e-8)
	
	# Select top keywords based on similarity and diversity
	selected_keywords = []
	used_words = set()
	
	for idx in np.argsort(similarities)[::-1]:
		keyword = all_candidates[idx]
		words = set(keyword.lower().split())
		
		# Skip if too much overlap with already selected keywords
		if len(words.intersection(used_words)) / len(words) < 0.5:
			selected_keywords.append(keyword)
			used_words.update(words)
	
	return selected_keywords

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		ner_model_name: str,
		verbose: bool=True,
		use_parallel: bool=False,
	):
	if verbose:
		print(f"Automatic label extraction from text data".center(160, "-"))
		print(f"Loading metadata from {csv_file}...")
	text_based_annotation_start_time = time.time()
	dataset_dir = os.path.dirname(csv_file)
	
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name}...")

	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device)
	ft_model = fasttext.load_model(FastText_Language_Identification)
	
	if verbose:
		print(f"Loading NER model: {ner_model_name}...")
	nlp = pipeline(
		task="ner", 
		model=AutoModelForTokenClassification.from_pretrained(ner_model_name),
		tokenizer=AutoTokenizer.from_pretrained(ner_model_name), 
		aggregation_strategy="simple",
		device=device,
		batch_size=batch_size,
	)
	dtypes = {
			'doc_id': str, 'id': str, 'label': str, 'title': str,
			'description': str, 'img_url': str, 'enriched_document_description': str,
			'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
			'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
			'user_query': str,
	}
	
	df = pd.read_csv(
			filepath_or_buffer=csv_file, 
			on_bad_lines='skip',
			dtype=dtypes, 
			low_memory=False,
	)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	df['content'] = df['enriched_document_description'].fillna('').astype(str)
	# Handle missing 'user_query' column
	if 'user_query' not in df.columns:
		if verbose:
			print("Warning: 'user_query' column missing in DataFrame. Using empty queries.")
		user_queries = [''] * len(df)
	else:
		user_queries = df['user_query'].fillna('').tolist()
	num_samples = df.shape[0]
	
	print(f"Filtering non-English entries for {num_samples} samples")
	t0 = time.time()
	english_mask = df['content'].apply(lambda x: is_english(text=x, ft_model=ft_model, verbose=False))
	english_indices = english_mask[english_mask].index.tolist()
	print(f"{sum(english_mask)} / {len(df)} texts are English [{sum(english_mask)/len(df)*100:.2f}%]")
	print(f"Elapsed_t: {time.time() - t0:.2f} sec")
	
	english_df = df[english_mask].reset_index(drop=True)
	english_texts = english_df['content'].tolist()
	english_queries = [user_queries[i] for i in english_indices]
	per_image_labels = [[] for _ in range(num_samples)]


	if len(english_texts) > 0:
		# Step 1: Topic Modeling
		print("Topic Modeling".center(160, "-"))
		t0 = time.time()
		topics, flat_topic_words = extract_semantic_topics(
			sent_model=sent_model,
			ft_model=ft_model,
			texts=english_texts,
			num_workers=num_workers,
			dataset_dir=dataset_dir,
		)
		print(f"{len(topics)} Topics (clusters) {type(topics)}:\n{[len(tp) for tp in topics]}")
		print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(160, "-"))
		
		# Step 2: Named Entity Recognition
		print("Extracting NER per sample...")
		t0 = time.time()
		if len(english_texts) > 1000 and use_parallel:
			chunk_size = len(english_texts) // num_workers + 1
			chunks = [(english_texts[i:i+chunk_size]) for i in range(0, len(english_texts), chunk_size)]
			print(f"Using {num_workers} processes for NER extraction...")
			with multiprocessing.Pool(processes=num_workers) as pool:
				ner_results = pool.map(process_text_chunk, [(nlp, chunk) for chunk in chunks])
			per_image_ner_labels = []
			for chunk_result in ner_results:
				per_image_ner_labels.extend(chunk_result)
		else:
			per_image_ner_labels = []
			for text in tqdm(english_texts, desc="NER Progress"):
				entities = extract_named_entities(nlp=nlp, text=text, ft_model=ft_model)
				per_image_ner_labels.append(entities)
		print(f"NER done in {time.time() - t0:.1f} sec")
		
		# Step 3: Extract keywords
		print("Extracting keywords per image using KeyBERT...")
		t0 = time.time()
		per_image_keywords = []
		rake = Rake(
			stopwords=list(CUSTOM_STOPWORDS),
			min_length=1,
			max_length=3,
			include_repeated_phrases=False
		)
		for text in tqdm(english_texts, desc="Keyword Extraction"):
			if not is_english(text, ft_model):
				per_image_keywords.append([])
				continue
			keywords = get_keywords(text, sent_model, rake)
			per_image_keywords.append(keywords)
		print(f"Keyword extraction done in {time.time() - t0:.1f} sec")

		# Step 4: Add topic labels
		print("Assigning topic labels per image...")
		t0 = time.time()
		per_image_topic_labels = []
		for text in tqdm(english_texts, desc="Topic Assignment"):
			matching_topics = [word for word in flat_topic_words if word in text.lower() and word not in CUSTOM_STOPWORDS]
			per_image_topic_labels.append(matching_topics)
		print(f"Topic assignment done in {time.time() - t0:.1f} sec")
		
		# Step 5: Combine and clean labels
		print("Combining and cleaning labels...")
		t0 = time.time()
		per_image_combined_labels = []
		for text, query, ner, keywords, topics in tqdm(zip(english_texts, english_queries, per_image_ner_labels, per_image_keywords, per_image_topic_labels), total=len(english_texts), desc="Label Combination"):
			cleaned_labels = combine_and_clean_labels(
				ner_labels=ner, 
				keywords=keywords, 
				topic_labels=topics, 
				user_query=query, 
				text=text, 
				sent_model=sent_model, 
				relevance_threshold=relevance_threshold,
				# min_threshold=0.4,
				# max_threshold=0.7,
			)
			per_image_combined_labels.append(cleaned_labels)
		print(f"Label combination and cleaning done in {time.time() - t0:.3f} sec")

		# Step 6: Filter by relevance
		print(f"Filtering labels by relevance (thresh: {relevance_threshold})...")
		t0 = time.time()
		if use_parallel:
			print("Using parallel processing for relevance filtering...")
			per_image_relevant_labels = parallel_relevance_filtering(
				texts=english_texts,
				all_labels=per_image_combined_labels,
				n_processes=num_workers,
			)
		else:
			print(f"Using batch processing for textual-based annotation ({batch_size} batches) for relevance filtering (thresh: {relevance_threshold})...")
			per_image_relevant_labels = batch_filter_by_relevance(
				sent_model=sent_model,
				texts=english_texts,
				all_labels_list=per_image_combined_labels,
				threshold=relevance_threshold,
				batch_size=batch_size,
			)
		print(f"Relevance filtering done in {time.time() - t0:.1f} sec")

		# Step 7: Post-process
		print(f"Post-processing of {len(per_image_relevant_labels)} textual labels, deduplication, and semantic categorization...")
		t0 = time.time()
		english_labels = []
		for i, relevant_labels in tqdm(enumerate(per_image_relevant_labels), total=len(per_image_relevant_labels), desc="Post-processing"):
			filtered_labels = handle_multilingual_labels(relevant_labels)
			filtered_labels = deduplicate_labels(filtered_labels)
			categorized = assign_semantic_categories(filtered_labels)
			final_labels = sorted(set(filtered_labels + categorized))
			english_labels.append(final_labels)
		print(f"Post-processing done in {time.time() - t0:.1f} sec")

		print("Balancing label counts...")
		t0 = time.time()
		english_labels = balance_label_count(
			image_labels_list=english_labels, 
			text_descriptions=english_texts, 
			sent_model=sent_model, 
			min_labels=1, 
			max_labels=12,
		)
		print(f"Label balancing done in {time.time() - t0:.3f} sec")
		
		# Transfer results
		for i, orig_idx in enumerate(english_indices):
			if i < len(english_labels):
				per_image_labels[orig_idx] = english_labels[i]
	else:
		print("No English texts found. Returning empty labels for all entries.")
	
	df['textual_based_labels'] = per_image_labels
	df.to_csv(metadata_fpth, index=False)
	
	print(f">> Generated text labels for {sum(1 for labels in per_image_labels if labels)} out of {num_samples} entries")
	print(f"Text-based annotation Elapsed time: {time.time() - text_based_annotation_start_time:.2f} sec".center(160, " "))
	
	return per_image_labels

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
		print(f"Semi-Supervised label extraction from image data (using VLM) batch_size: {batch_size}".center(160, "-"))
	
	visual_based_annotation_start_time = time.time()
	
	# Load categories
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	texts = [f"This is a photo of {lbl}." for lbl in candidate_labels]
	
	gpu_name = torch.cuda.get_device_name(device)
	total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
	available_gpu_memory = torch.cuda.mem_get_info()[0] / 1024**3 # GB
	print(f"Total GPU memory: {total_gpu_memory:.2f} GB ({gpu_name})")
	print(f"Available GPU memory: {available_gpu_memory:.2f} GB ({gpu_name})")
	
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
	
	# Load dataframe
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	}
	df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	img_paths = df['img_path'].tolist()
	combined_labels = [[] for _ in range(len(img_paths))]
	
	# Process images using DataLoader
	dataset = HistoricalArchives(img_paths)
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size, 
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=True if num_workers > 1 else False,
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
				).to(device)
				
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

def process_text_chunk(nlp, chunk):
	return [extract_named_entities(nlp=nlp, text=text) for text in chunk]

def parallel_relevance_filtering(texts, all_labels, n_processes=None):
	if n_processes is None:
		n_processes = max(1, multiprocessing.cpu_count() - 1)
	
	total_docs = len(texts)
	chunk_size = total_docs // n_processes + (1 if total_docs % n_processes else 0)
	chunks = []
	
	for i in range(0, total_docs, chunk_size):
		end_idx = min(i + chunk_size, total_docs)
		chunks.append((i, end_idx, texts[i:end_idx], all_labels[i:end_idx]))
	
	with multiprocessing.Pool(processes=n_processes) as pool:
		chunk_results = pool.map(process_document_chunk, chunks)
	
	all_results = []
	for chunk in chunk_results:
		all_results.extend(chunk)
	
	return all_results

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--use_parallel", '-parallel', action="store_true")
	parser.add_argument("--num_workers", '-nw', type=int, default=6, help="Number of workers for parallel processing")
	parser.add_argument("--relevance_threshold", '-rth', type=float, default=0.25, help="Relevance threshold for textual-based annotation")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=64)
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=8, help="Batch size for vision processing")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="all-mpnet-base-v2", choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], help="Sentence-transformer model name")
	parser.add_argument("--vlm_model_name", '-vlm', type=str, default="google/siglip2-so400m-patch16-naflex", choices=["kakaobrain/align-base", "google/siglip2-so400m-patch16-naflex"], help="Vision-Language model name")
	parser.add_argument("--ner_model_name", '-ner', type=str, default="Babelscape/wikineural-multilingual-ner", choices=["dslim/bert-large-NER", "dslim/bert-base-NER", "Babelscape/wikineural-multilingual-ner"], help="NER model name")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)

	set_seeds(seed=42)
	DATASET_DIRECTORY = os.path.dirname(args.csv_file)
	text_output_path = os.path.join(DATASET_DIRECTORY, "metadata_textual_based_labels.csv")
	vision_output_path = os.path.join(DATASET_DIRECTORY, "metadata_visual_based_labels.csv")
	combined_output_path = os.path.join(DATASET_DIRECTORY, "metadata_multimodal_labels.csv")
	
	if os.path.exists(text_output_path):
		print(f"Found existing textual-based labels at {text_output_path}. Loading...")
		text_df = pd.read_csv(text_output_path)
		
		# Handle textual labels with proper null value handling
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
		
		print(f"Loaded {len(textual_based_labels)} textual-based labels")
	else:
		textual_based_labels = get_textual_based_annotation(
			csv_file=args.csv_file,
			use_parallel=args.use_parallel,
			num_workers=args.num_workers,
			batch_size=args.text_batch_size,
			relevance_threshold=args.relevance_threshold,
			st_model_name=args.sentence_model_name,
			ner_model_name=args.ner_model_name,
			metadata_fpth=text_output_path,
			device=args.device,
			verbose=True,
		)

	if os.path.exists(vision_output_path):
		print(f"Found existing visual-based labels at {vision_output_path}. Loading...")
		vision_df = pd.read_csv(vision_output_path)
		
		# Handle visual labels with proper null value handling
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
		
		print(f"Loaded {len(visual_based_labels)} visual-based labels")
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
	
	print("Sample textual labels:", textual_based_labels[:15])
	print("Sample visual labels:", visual_based_labels[:15])
	assert len(textual_based_labels) == len(visual_based_labels), "Label lists must have same length"
	
	# Check if combined file already exists
	if os.path.exists(combined_output_path):
		print(f"Found existing combined labels at {combined_output_path}.")
		recompute = input("Do you want to recompute the combined labels? (y/n): ").lower() == 'y'
		if not recompute:
			print("Using existing combined labels.")
			combined_df = pd.read_csv(combined_output_path)
			# Handle combined labels with proper null value handling
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

	print("Merging text and vision annotations...")
	combined_labels = []
	empty_count = 0
	
	for text_labels, vision_labels in zip(textual_based_labels, visual_based_labels):
			# Check if both are empty - if so, store None for this entry
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
			all_labels = clean_labels(all_labels)
			all_labels = filter_metadata_terms(all_labels)
			all_labels = deduplicate_labels(all_labels)
			
			# Add semantic categories
			categorized = assign_semantic_categories(all_labels)
			final_labels = sorted(set(all_labels + categorized))
			
			combined_labels.append(final_labels)
	
	print(f"Created {len(combined_labels)} combined labels ({empty_count} empty entries)")
	
	# Save results
	df = pd.read_csv(args.csv_file)
	
	# Convert empty lists to None for better CSV representation
	df['textual_based_labels'] = [labels if labels else None for labels in textual_based_labels]
	df['visual_based_labels'] = [labels if labels else None for labels in visual_based_labels]
	df['multimodal_labels'] = [labels if labels else None for labels in combined_labels]
	
	# Save to CSV
	df.to_csv(combined_output_path, index=False)
	
	# Try to save as Excel
	try:
		excel_path = combined_output_path.replace('.csv', '.xlsx')
		print(f"Saving Excel file to {excel_path}...")
		df.to_excel(excel_path, index=False)
		print(f"Excel file saved successfully")
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	print("\nExample results:")
	sample_cols = ['title', 'description', 'label', 'user_query', 'img_url', 'enriched_document_description', 'textual_based_labels', 'visual_based_labels', 'multimodal_labels']
	available_cols = [col for col in sample_cols if col in df.columns]
	for i in range(min(25, len(df))):
		print(f"\nExample {i+1}:")
		for col in available_cols:
			value = df.iloc[i][col]
			if col in ['textual_based_labels', 'visual_based_labels', 'multimodal_labels']:
				print(f"{col}: {value if value else '[]'}")
			else:
				print(f"{col}: {value}")
	print(f"Combined labels saved to: {combined_output_path}")
	return combined_labels

if __name__ == "__main__":
	multiprocessing.set_start_method('spawn', force=True)
	main()