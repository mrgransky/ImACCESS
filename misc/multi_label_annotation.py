from utils import *
torch.set_grad_enabled(False)
from fuzzywuzzy import fuzz

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata.csv -d "cuda:1" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_SMU.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31/metadata.csv -d "cuda:0" -nw 24 -tbs 8 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_EUROPEANA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1930-01-01_1955-12-31/metadata.csv -d "cuda:1" -nw 16 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_NA.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata.csv -d "cuda:2" -nw 20 -tbs 8 -vbs 32 -vth 0.3 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_WWII.out &
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -d "cuda:3" -nw 20 -tbs 256 -vbs 32 -vth 0.25 -rth 0.3 > /media/volume/ImACCESS/trash/multi_label_annotation_HISTORY_X4.out &

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
		"used", "southern", "built", "album", "album cover", "opel ma",
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
	'MISC',
}

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
	min_topic_size = max(10, min(20, len(texts)//500))
	print(f"Creating BERTopic model for {len(texts)} texts => min_topic_size: {min_topic_size}")
	topic_model = BERTopic(
		embedding_model=sent_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_model,
		min_topic_size=min_topic_size,
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

def extract_named_entities(
	nlp: pipeline,
	text: str, 
	ft_model: fasttext.FastText._FastText,
	confidence_threshold: float = 0.7,
	) -> List[str]:
	if not text or not isinstance(text, str) or len(text) < 5 or not is_english(text, ft_model):
		return []
	print(text)
	try:
		ner_results = nlp(text)
		print(f"NER results ({len(ner_results)}): {ner_results}")
		entities = []
		seen_entities = set()
		for entity in ner_results:
			if entity.get("score", 0) < confidence_threshold or entity["entity_group"] not in RELEVANT_ENTITY_TYPES:
				continue
			full_entity = entity["word"].strip()
			full_entity = re.sub(r'^(The|A|An)\s+', '', full_entity, flags=re.I)
			full_entity = ' '.join(full_entity.split())
			if len(full_entity) < 3 or full_entity.lower() in CUSTOM_STOPWORDS:
				continue
			if not any(fuzz.ratio(full_entity.lower(), seen.lower()) > 95 for seen in seen_entities):
				normalized = full_entity if full_entity[0].isupper() else full_entity.lower()
				entities.append(normalized)
				seen_entities.add(full_entity.lower())
		try:
			tokens = nlp.tokenizer.tokenize(text)
			tokens = [token.lower() for token in tokens if token.lower() not in CUSTOM_STOPWORDS and len(token) > 2]
			meaningful_bigrams = [
				f"{tokens[i]} {tokens[i+1]}" 
				for i in range(len(tokens)-1) 
				if tokens[i] not in CUSTOM_STOPWORDS 
				and tokens[i+1] not in CUSTOM_STOPWORDS
				and len(tokens[i]) > 2 and len(tokens[i+1]) > 2
				and not any(f"{tokens[i]} {tokens[i+1]}".lower() in e.lower() for e in entities)
			]
			combined = list(set(entities + meaningful_bigrams))
			final_entities = []
			seen = set()
			for entity in sorted(combined, key=len, reverse=True):
				entity_lower = entity.lower()
				if not any(fuzz.ratio(entity_lower, s.lower()) > 95 for s in seen):
					final_entities.append(entity)
					seen.add(entity_lower)
			final_entities = sorted(final_entities)
			print(f"Final entities: {final_entities}")
			return final_entities
		except Exception as tokenize_error:
			print(f"Tokenization warning: {tokenize_error}")
			return sorted(list(set(entities)))
	except Exception as e:
		print(f"<!> NER error: {e} for text: {text}")
		return []

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

def balance_label_count(
		image_labels_list,
		text_descriptions,
		sent_model,
		min_labels=1,
		max_labels=12,
		similarity_threshold=0.5
	):
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

def combine_and_clean_labels(
		ner_labels: List[str],
		keywords: List[str],
		topic_labels: List[str],
		user_query: Union[str, None],
		text: str,
		sent_model: SentenceTransformer,
		doc_year: Union[float, None] = None,
		relevance_threshold: float = 0.4,
		max_labels: int = 12,
		min_label_length: int = 4,
		semantic_coherence_threshold: float = 0.9
	) -> List[str]:

	def collect_weighted_labels(
			user_terms: List[str],
			ner_labels: List[str],
			keywords: List[str],
			topic_labels: List[str]
	) -> List[Tuple[str, float, str]]:
			weighted = []
			# User terms (high priority)
			for term in user_terms:
					weighted.append((term, 1.0, 'user'))
			# NER labels (favor proper nouns)
			for label in ner_labels:
					if len(label) >= min_label_length:
							weight = 0.9 if label[0].isupper() else 0.7
							weighted.append((label, weight, 'ner'))
			# Keywords (length-based weight)
			for label in keywords:
					if len(label) >= min_label_length:
							weight = 0.6 + 0.1 * len(label.split())
							weighted.append((label, weight, 'keyword'))
			# Topic labels (lowest priority)
			for label in topic_labels:
					if len(label) >= min_label_length:
							weighted.append((label, 0.4, 'topic'))
			return weighted
	def generate_embeddings(labels: List[Tuple[str, float, str]]) -> Tuple[List[str], np.ndarray]:
			label_texts = [lbl for lbl, _, _ in labels]
			if not label_texts:
					return [], np.array([])
			embeddings = sent_model.encode(
					label_texts,
					batch_size=64,
					show_progress_bar=False,
					convert_to_numpy=True
			)
			return label_texts, embeddings
	def perform_semantic_clustering(embeddings: np.ndarray) -> np.ndarray:
			if len(embeddings) <= 2:
					return np.zeros(len(embeddings), dtype=int)
			embeddings = normalize(embeddings, norm='l2', axis=1)
			min_cluster_size = max(2, len(embeddings) // 5)
			clusterer = hdbscan.HDBSCAN(
					min_cluster_size=min_cluster_size,
					min_samples=1,
					metric='euclidean',
					cluster_selection_method='eom'
			)
			return clusterer.fit_predict(embeddings)
	def process_clusters(
		clusters: np.ndarray,
		weighted_labels: List[Tuple[str, float, str]],
		label_to_emb: Dict[str, np.ndarray],
		text_emb: np.ndarray
		) -> List[str]:
		cluster_groups = defaultdict(list)
		for idx, (label, weight, source) in enumerate(weighted_labels):
			cluster_groups[clusters[idx]].append((label, weight, source))
		
		final_labels = []
		for cluster_id, members in cluster_groups.items():
			if cluster_id == -1:
				continue
			scored = []
			for label, weight, source in members:
					source_priority = {'user': 4, 'ner': 3, 'keyword': 2, 'topic': 1}[source]
					length_factor = 1.0 + 0.1 * len(label.split())
					sim = np.dot(label_to_emb[label], text_emb) / (
							np.linalg.norm(label_to_emb[label]) * np.linalg.norm(text_emb) + 1e-8
					)
					score = weight * length_factor * source_priority * (0.5 + 0.5 * sim)
					scored.append((score, label))
			if scored:
					top_score, top_label = max(scored)
					if top_score > relevance_threshold:
							final_labels.append(top_label)
		return final_labels
	def handle_noise_labels(
			clusters: np.ndarray,
			label_texts: List[str],
			embeddings: np.ndarray,
			text_emb: np.ndarray
	) -> List[str]:
			noise_mask = (clusters == -1)
			if not noise_mask.any():
					return []
			noise_labels = [label for label, mask in zip(label_texts, noise_mask) if mask]
			noise_embs = embeddings[noise_mask]
			sims = np.dot(noise_embs, text_emb) / (
					np.linalg.norm(noise_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
			)
			top_indices = np.argsort(sims)[-2:][::-1]
			return [noise_labels[i] for i in top_indices if sims[i] > relevance_threshold]
	def apply_final_filters(
			candidates: List[str],
			label_to_emb: Dict[str, np.ndarray],
			text_emb: np.ndarray
	) -> List[str]:
			if not candidates:
					return []
			final_embs = np.array([label_to_emb[label] for label in candidates])
			final_sims = np.dot(final_embs, text_emb) / (
					np.linalg.norm(final_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
			)
			filtered = []
			used_embeddings = []
			for label, sim in sorted(zip(candidates, final_sims), key=lambda x: -x[1]):
					if sim < relevance_threshold:
							continue
					label_emb = label_to_emb[label]
					if any(np.dot(label_emb, used_emb) / (
							np.linalg.norm(label_emb) * np.linalg.norm(used_emb) + 1e-8
					) > semantic_coherence_threshold for used_emb in used_embeddings):
							continue
					if any(fuzz.ratio(label.lower(), kept.lower()) > 90 for kept in filtered):
							continue
					filtered.append(label)
					used_embeddings.append(label_emb)
					if doc_year and not is_year_compatible(label, doc_year):
							filtered.remove(label)
					if len(filtered) >= max_labels:
							break
			return sorted(filtered)
	
	# 1. Parse and clean user query
	user_terms = parse_user_query(user_query)
	# 2. Collect weighted labels with source-based priority
	weighted_labels = collect_weighted_labels(user_terms, ner_labels, keywords, topic_labels)
	if not weighted_labels:
		return []
	# 3. Generate embeddings
	label_texts, embeddings = generate_embeddings(weighted_labels)
	if not embeddings.size:
		return []
	label_to_emb = {lbl: emb for lbl, emb in zip(label_texts, embeddings)}
	text_emb = sent_model.encode(text, show_progress_bar=False)
	# 4. Cluster embeddings
	clusters = perform_semantic_clustering(embeddings)
	# 5. Process clusters
	final_labels = process_clusters(clusters, weighted_labels, label_to_emb, text_emb)
	# 6. Handle noise labels
	final_labels += handle_noise_labels(clusters, label_texts, embeddings, text_emb)
	# 7. Final filtering and deduplication
	final_labels = apply_final_filters(final_labels, label_to_emb, text_emb)
	return final_labels

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

def get_keywords(
		text: str, 
		sent_model: SentenceTransformer, 
		rake: Rake,
	):
	rake.extract_keywords_from_text(text)
	rake_phrases = [
		phrase 
		for phrase in rake.get_ranked_phrases() 
		if len(phrase.split()) <= 3 and phrase.lower() not in CUSTOM_STOPWORDS
	]

	kw_model = KeyBERT(model=sent_model)	
	keybert_keywords = kw_model.extract_keywords(
		text,
		keyphrase_ngram_range=(1, 3),
		stop_words=list(CUSTOM_STOPWORDS),
		top_n=20,
		use_mmr=True,  # Use Maximal Marginal Relevance for diversity
		diversity=0.7,
	)
	
	# Combine and rank by relevance to text
	all_candidates = list(set(rake_phrases[:15] + [kw[0] for kw in keybert_keywords]))
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
	
	# Dynamic threshold based on distribution
	threshold = np.percentile(similarities, 70)  # Keep top 30%
	filtered = [
		(cand, sim) 
		for cand, sim in zip(all_candidates, similarities) 
		if sim > threshold and is_likely_english_term(cand)
	]
	
	# Final selection with diversity
	selected = []
	used_words = set()
	for cand, sim in sorted(filtered, key=lambda x: x[1], reverse=True):
		words = set(cand.lower().split())
		overlap = len(words & used_words) / len(words)
		if overlap < 0.4:  # Allow some overlap but not too much
			selected.append(cand)
			used_words.update(words)
	return selected

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
        relevance_threshold: float,
        metadata_fpth: str,
        device: str,
        st_model_name: str,
        ner_model_name: str,
        topk: int = 3,
        verbose: bool = True,
        use_parallel: bool = False,
    ):
        # Memory optimization settings
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
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
        torch.set_grad_enabled(False)
        
        # Load categories
        CATEGORIES_FILE = "categories.json"
        object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
        candidate_labels = list(set(object_categories + scene_categories + activity_categories))
        
        # Pre-compute label embeddings once
        if verbose:
                print("Pre-computing label embeddings...")
        t0 = time.time()
        label_embs = sent_model.encode(
                candidate_labels,
                batch_size=min(32, len(candidate_labels)),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
        )
        if verbose:
                print(f"Label embeddings computed in {time.time() - t0:.2f} sec")
        
        # Load dataframe
        if verbose:
                print(f"Loading dataframe: {csv_file}...")
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
                low_memory=True
        )
        if verbose:
                print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
        
        # Initialize results columns with empty lists
        df['textual_based_labels'] = [[] for _ in range(len(df))]
        df['textual_based_scores'] = [[] for _ in range(len(df))]
        
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
                text_batch_size = min(8, batch_size)
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
                
                # Clean up memory
                del text_embs, cosine_scores, topk_scores, topk_indices
                torch.cuda.empty_cache()
        
        # Save results
        df.to_csv(metadata_fpth, index=False)
        
        try:
                df.to_excel(metadata_fpth.replace(".csv", ".xlsx"), index=False)
        except Exception as e:
                print(f"Failed to write Excel file: {e}")
        
        # Display sample results safely
        if verbose:
                print(f"Completed in {time.time() - start_time:.2f} seconds")
                print("\nSample results:")
                samples_displayed = 0
                for i in range(len(df)):
                        desc = df.iloc[i]['enriched_document_description']
                        if pd.isna(desc) or not str(desc).strip():
                                continue
                                
                        labels = df.iloc[i]['textual_based_labels']
                        if not labels:
                                continue
                                
                        print(f"\nSample {samples_displayed + 1}:")
                        print("Text:", str(desc)[:200] + ("..." if len(str(desc)) > 200 else ""))
                        print("Top Labels:", labels)
                        print("Scores:", df.iloc[i]['textual_based_scores'])
                        
                        samples_displayed += 1
                        if samples_displayed >= 5:
                                break

        return df['textual_based_labels'].tolist()

def get_textual_based_annotation_old2(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		ner_model_name: str,
		topk: int = 3,
		verbose: bool = True,
		use_parallel: bool = False,
	):
		# Memory optimization settings
		os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
		torch.backends.cuda.enable_flash_sdp(True)
		torch.backends.cuda.enable_mem_efficient_sdp(True)
		
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
		torch.set_grad_enabled(False)
		
		# Load categories
		CATEGORIES_FILE = "categories.json"
		object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
		candidate_labels = list(set(object_categories + scene_categories + activity_categories))
		
		# Pre-compute label embeddings once
		if verbose:
				print("Pre-computing label embeddings...")
		t0 = time.time()
		label_embs = sent_model.encode(
				candidate_labels,
				batch_size=min(32, len(candidate_labels)),
				convert_to_tensor=True,
				normalize_embeddings=True,
				show_progress_bar=False,
		)
		if verbose:
				print(f"Label embeddings computed in {time.time() - t0:.2f} sec")
		
		# Load dataframe
		if verbose:
				print(f"Loading dataframe: {csv_file}...")
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
				low_memory=True
		)
		if verbose:
				print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
		
		# Initialize results columns
		df['textual_based_labels'] = [[] for _ in range(len(df))]
		df['textual_based_scores'] = [[] for _ in range(len(df))]
		
		# Process in chunks with memory management
		chunk_size = min(1000, len(df))
		if verbose:
				print(f"Processing {len(df)} samples with {len(candidate_labels)} candidate labels in {chunk_size} chunks...")

		for chunk_start in tqdm(range(0, len(df), chunk_size), desc="Processing documents"):
				chunk_end = min(chunk_start + chunk_size, len(df))
				chunk_df = df.iloc[chunk_start:chunk_end]
				
				# Process texts in smaller batches
				text_batch_size = min(8, batch_size)
				text_embs = []
				
				for i in range(0, len(chunk_df), text_batch_size):
						batch_texts = chunk_df['enriched_document_description'].fillna('').iloc[i:i+text_batch_size].tolist()
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
				
				# Store results safely
				for i in range(len(chunk_df)):
						idx = chunk_start + i
						try:
								labels = [candidate_labels[j] for j in topk_indices[i] if j < len(candidate_labels)]
								scores = [round(s.item(), 4) for s in topk_scores[i]][:len(labels)]
								df.at[idx, 'textual_based_labels'] = labels
								df.at[idx, 'textual_based_scores'] = scores
						except Exception as e:
								print(f"Error processing sample {idx}: {str(e)[:200]}")
								df.at[idx, 'textual_based_labels'] = []
								df.at[idx, 'textual_based_scores'] = []
				
				# Clean up memory
				del text_embs, cosine_scores, topk_scores, topk_indices
				torch.cuda.empty_cache()
		
		# Save results
		df.to_csv(metadata_fpth, index=False)
		
		try:
				df.to_excel(metadata_fpth.replace(".csv", ".xlsx"), index=False)
		except Exception as e:
				print(f"Failed to write Excel file: {e}")
		
		# Display sample results safely
		if verbose:
				print(f"Completed in {time.time() - start_time:.2f} seconds")
				print("\nSample results:")
				samples_displayed = 0
				for i in range(len(df)):
						desc = df.iloc[i]['enriched_document_description']
						if pd.isna(desc):
								continue
								
						labels = df.iloc[i]['textual_based_labels']
						if not labels:
								continue
								
						print(f"\nSample {samples_displayed + 1}:")
						print("Text:", str(desc)[:200] + ("..." if len(str(desc)) > 200 else ""))
						print("Top Labels:", labels)
						print("Scores:", df.iloc[i]['textual_based_scores'])
						
						samples_displayed += 1
						if samples_displayed >= 5:
								break

		return df['textual_based_labels'].tolist()

def get_textual_based_annotation_old(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		relevance_threshold: float,
		metadata_fpth: str,
		device: str,
		st_model_name: str,
		ner_model_name: str,
		topk: int = 3,
		verbose: bool=True,
		use_parallel: bool=False,
	):
	if verbose:
		print(f"Semi-Supervised textual-based annotation batch_size: {batch_size} num_workers: {num_workers}".center(160, "-"))
	text_based_annotation_start_time = time.time()
	
	if verbose:
		print(f"Loading sentence-transformer model: {st_model_name}...")
	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device, trust_remote_code=True)
	# Load dataframe
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
		'user_query': str,
	}
	
	CATEGORIES_FILE = "categories.json"
	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
	
	print(f"Loading dataframe: {csv_file}...")
	df = pd.read_csv(filepath_or_buffer=csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
	if verbose:
		print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")

	# Initialize columns for results
	df['textual_based_labels'] = None
	df['textual_based_scores'] = None

	print(f"Processing {len(df)} samples with {len(candidate_labels)} candidate labels...")
	num_samples = df.shape[0]
	per_image_labels = [[] for _ in range(num_samples)]
	per_image_labels_scores = [[] for _ in range(num_samples)]
	# Process in batches to manage memory
	num_batches = (len(df) + batch_size - 1) // batch_size
	
	for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
		torch.cuda.empty_cache()
		start_idx = batch_idx * batch_size
		end_idx = min((batch_idx + 1) * batch_size, len(df))
		batch_df = df.iloc[start_idx:end_idx]
		
		# Pre-compute text embeddings for the batch
		batch_texts = batch_df['enriched_document_description'].fillna('').tolist()
		text_embs = sent_model.encode(batch_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
		
		# Compute similarities with all labels
		label_embs = sent_model.encode(candidate_labels, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
		cosine_scores = util.cos_sim(text_embs, label_embs)  # shape: (batch_size, num_labels)
		
		# Get top-k results for each text in batch
		topk_scores, topk_indices = torch.topk(cosine_scores, k=topk, dim=1)
		
		# Store results in dataframe
		for i in range(len(batch_df)):
			idx = start_idx + i
			labels = [candidate_labels[j] for j in topk_indices[i]]
			scores = [round(s.item(), 4) for s in topk_scores[i]]
			per_image_labels[idx] = labels
			per_image_labels_scores[idx] = scores
		torch.cuda.empty_cache()
	df['textual_based_labels'] = per_image_labels
	df['textual_based_scores'] = per_image_labels_scores

	# Save results
	print(f"Saving results to {metadata_fpth}...")
	df.to_csv(metadata_fpth, index=False)

	try:
		df.to_excel(metadata_fpth.replace(".csv", ".xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	print(f"Completed in {time.time() - text_based_annotation_start_time:.2f} seconds")
	
	# Show some sample results
	print("\nSample results:")
	for i in range(min(5, len(df))):
		print(f"\nSample {i+1}:")
		print("Text:", df.iloc[i]['enriched_document_description'][:200] + "...")
		print("Top Labels:", df.iloc[i]['textual_based_labels'])
		print("Scores:", df.iloc[i]['textual_based_scores'])

	return per_image_labels

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
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--relevance_threshold", '-rth', type=float, default=0.3, help="Relevance threshold for textual-based annotation")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=128, help="Batch size for textual processing")
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=4, help="Batch size for vision processing")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="jinaai/jina-embeddings-v3", choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "jinaai/jina-embeddings-v3"], help="Sentence-transformer model name")
	parser.add_argument("--vlm_model_name", '-vlm', type=str, default="google/siglip2-so400m-patch16-naflex", choices=["kakaobrain/align-base", "google/siglip2-so400m-patch16-naflex"], help="Vision-Language model name")
	parser.add_argument("--ner_model_name", '-ner', type=str, default="dslim/bert-large-NER", choices=["dslim/bert-large-NER", "dslim/bert-base-NER", "Babelscape/wikineural-multilingual-ner"], help="NER model name")
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
			all_labels = clean_(all_labels)
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