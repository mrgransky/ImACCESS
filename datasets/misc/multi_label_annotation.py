import numpy as np
import pandas as pd
import re
import os
import time
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
# import faiss
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import nltk
from tqdm import tqdm
import warnings
import urllib.request
import fasttext
import argparse

nltk.download('words', quiet=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Make language detection deterministic
DetectorFactory.seed = 42

FastText_Language_Identification = "lid.176.ftz"
if "lid.176.ftz" not in os.listdir():
	url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
	urllib.request.urlretrieve(url, "lid.176.ftz")

ft_model = fasttext.load_model("lid.176.ftz")

DATASET_DIRECTORY = {
	# "farid": "/home/farid/datasets/WW_DATASETs/HISTORY_X3",
	# "farid": "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31",
	"farid": "/home/farid/datasets/WW_DATASETs/WW_VEHICLES",
	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4",
	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs/HISTORY_X4",
	"alijani": "/lustre/sgn-data/ImACCESS/WW_DATASETs/HISTORY_X4",
}
full_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata.csv")
train_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_train.csv")
val_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_val.csv")

# Custom stopwords and metadata patterns
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union({
	"original", "bildetekst", "photo", "image", "archive", "arkivreferanse",
	"copyright", "description", "riksarkivet", "ntbs", "ra", "pa", "bildetekst",
	"left", "right", "center", "top", "bottom", "middle", "front", "back",
	"year", "month", "day", "date", "century", "decade", "era",
	"showing", "shown", "shows", "depicts", "depicting", "pictured", "picture",
	"original", "copy", "version", "view", "looking", "seen", "visible",
	"photograph", "photographer", "photography", "photo", "image", "img",
	"sent", "received", "taken", "made", "created", "produced", "found",
	"above", "below", "beside", "behind", "between", "among", "alongside",
	"across", "opposite", "near", "under", "over", "inside", "outside",
	"collection", "collections", "number",
})

METADATA_PATTERNS = [
	r'bildetekst \w+',        # Matches 'bildetekst german' and similar
	r'kunststoff \w+',        # Matches photography material descriptions
	r'arkivreferanse \w+',    # Archive references
	r'\w+ pa \d+',            # Reference codes like 'ra pa-1209'
	r'donated \w+',           # Donation information
	r'information received',  # Source information
	r'\w+ association',       # Organization suffixes without context
]

# Load models
sent_model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
nlp = pipeline(
	task="ner", 
	model=model, 
	tokenizer=tokenizer, 
	aggregation_strategy="simple",
	device="cuda:0",
	batch_size=512,
)

# Semantic categories for organization
SEMANTIC_CATEGORIES = {
	'military': ['military', 'army', 'navy', 'air force', 'soldier', 'officer', 'troop', 'regiment', 'division', 'corps', 'battalion'],
	'political': ['government', 'parliament', 'president', 'prime minister', 'minister', 'official', 'politician', 'leader'],
	'event': ['war', 'battle', 'attack', 'invasion', 'liberation', 'occupation', 'revolution', 'protest', 'march', 'ceremony'],
	'location': ['city', 'town', 'village', 'country', 'region', 'territory', 'front', 'border', 'base', 'camp'],
	'vehicle': ['tank', 'aircraft', 'plane', 'ship', 'submarine', 'boat', 'truck', 'car', 'jeep', 'vehicle'],
	'weapon': ['gun', 'rifle', 'cannon', 'artillery', 'weapon', 'bomb', 'missile', 'ammunition'],
}

# Entity types for NER
RELEVANT_ENTITY_TYPES = {
	'PERSON', 'ORG', 'GPE', 'LOC', 'NORP', 'FAC', 
	'EVENT', 'WORK_OF_ART', 'PRODUCT', 'DATE'
}

def is_english(text):
	if not text or len(text) < 5:
		return False
	if len(text) < 20:
		# Short texts: rely on ASCII + stopword heuristics
		ascii_chars = sum(c.isalpha() and ord(c) < 128 for c in text)
		total_chars = sum(c.isalpha() for c in text)
		if total_chars == 0 or ascii_chars / total_chars < 0.9:
			return False
		common_words = {'the', 'and', 'of', 'to', 'in', 'is', 'was', 'for', 'with', 'on'}
		words = text.lower().split()
		return any(word in common_words for word in words)
	# Long texts: fasttext is preferred
	return ft_model.predict(text)[0][0] == '__label__en'

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

def clean_text(text):
		"""Clean text by removing special characters and excess whitespace"""
		if not isinstance(text, str):
				return ""
		
		# Apply all metadata pattern removals
		for pattern in METADATA_PATTERNS:
				text = re.sub(pattern, '', text)

		# Replace specific patterns often found in metadata
		text = re.sub(r'\[\{.*?\}\]', '', text)  # Remove JSON-like structures
		text = re.sub(r'http\S+', '', text)      # Remove URLs
		text = re.sub(r'\d+\.\d+', '', text)     # Remove floating point numbers
		# Remove non-alphanumeric characters but keep spaces
		text = re.sub(r'[^\w\s]', ' ', text)
		# Replace multiple spaces with a single space
		text = re.sub(r'\s+', ' ', text)
		text = text.strip().lower()

		return text

def extract_phrases(text, max_words=3):
		"""Extract meaningful phrases from text"""
		words = text.lower().split()
		phrases = []
		for i in range(len(words)):
				for j in range(1, min(max_words + 1, len(words) - i + 1)):
						phrase = ' '.join(words[i:i+j])
						if all(word not in CUSTOM_STOPWORDS for word in phrase.split()):
								phrases.append(phrase)
		return phrases

def extract_semantic_topics(
		texts: list[str], 
		n_clusters: int =25, 
		top_k_words: int =10, 
		merge_threshold: float =0.65
	):
	"""Extract semantic topics using sentence embeddings and clustering with redundancy reduction"""
	# Generate embeddings
	print(f"Generating embeddings for {len(texts)} texts...")
	embeddings = sent_model.encode(texts, show_progress_bar=True)
	
	# Clustering with increased default clusters
	print(f"Clustering embeddings {embeddings.shape} into {n_clusters} topics with KMeans...")
	kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', max_iter=1000, n_init='auto')
	labels = kmeans.fit_predict(embeddings)
	
	# Plot original distribution
	topic_counts = Counter(labels)
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(topic_counts)), [topic_counts[i] for i in range(n_clusters)])
	plt.title('Document Distribution Across Topics')
	plt.xlabel('Topic ID')
	plt.ylabel('Number of Documents')
	plt.xticks(range(n_clusters))
	plt.savefig('topic_distribution.png')
	
	# Collect phrases for each cluster
	cluster_phrases = defaultdict(Counter)
	for i, label in enumerate(labels):
		# Check if the text is English before extracting phrases
		if is_english(texts[i]):
			phrases = extract_phrases(texts[i])
			# Filter out phrases less than 3 characters or containing stopwords
			valid_phrases = [
				phrase for phrase in phrases 
				if len(phrase) > 3 and 
				not any(word in CUSTOM_STOPWORDS for word in phrase.split())
			]
			cluster_phrases[label].update(valid_phrases)
	
	# Extract top phrases from each cluster
	initial_topics = []
	for label, counter in cluster_phrases.items():
		most_common = [w for w, c in counter.most_common(top_k_words * 4) if len(w.split()) <= 3 and all(ord(char) < 128 for char in w)]
		initial_topics.append(most_common[:top_k_words])
	
	# NEW: Calculate topic similarities for merging
	print("Calculating topic similarities for redundancy reduction...")
	similarity_matrix = np.zeros((len(initial_topics), len(initial_topics)))
	
	# Calculate embeddings for all topic words more efficiently
	all_words = [word for topic in initial_topics for word in topic if word]
	word_to_embedding = {}
	
	if all_words:
		word_embeddings = sent_model.encode(all_words, show_progress_bar=True)
		for i, word in enumerate(all_words):
			word_to_embedding[word] = word_embeddings[i]
	
	# Calculate average embedding for each topic
	topic_embeddings = []
	for topic in initial_topics:
		if not topic:
			# Empty topic gets zero embedding
			topic_embeddings.append(np.zeros(embeddings.shape[1]))
			continue
		
		# Average the embeddings of words in this topic
		topic_embs = [word_to_embedding[word] for word in topic if word in word_to_embedding]
		if topic_embs:
			topic_emb = np.mean(topic_embs, axis=0)
			topic_embeddings.append(topic_emb)
		else:
			topic_embeddings.append(np.zeros(embeddings.shape[1]))
	
	# Calculate similarity between each pair of topics
	for i in range(len(initial_topics)):
		for j in range(i+1, len(initial_topics)):
			sim = util.cos_sim([topic_embeddings[i]], [topic_embeddings[j]])[0][0].item()
			similarity_matrix[i, j] = sim
			similarity_matrix[j, i] = sim  # Symmetric matrix
	
	# Merge similar topics
	print(f"Merging similar topics with threshold {merge_threshold}...")
	merged_topics = []
	used_indices = set()
	
	for i in range(len(initial_topics)):
		if i in used_indices:
			continue
		
		# Start a new merged topic with all words from topic i
		merged_words = set(initial_topics[i])
		used_indices.add(i)
		
		# Find similar topics to merge
		for j in range(len(initial_topics)):
			if j in used_indices or i == j:
				continue
			
			if similarity_matrix[i, j] > merge_threshold:
				# Add all words from topic j
				merged_words.update(initial_topics[j])
				used_indices.add(j)
		
		# Limit to top words and ensure English only
		merged_topics.append([w for w in sorted(list(merged_words))[:top_k_words] if all(ord(char) < 128 for char in w)])
	
	print(f"Reduced from {len(initial_topics)} to {len(merged_topics)} topics after merging")
	
	# Plot merged topic distribution
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(merged_topics)), [len(topic) for topic in merged_topics])
	plt.title('Distribution of Terms in Merged Topics')
	plt.xlabel('Topic ID')
	plt.ylabel('Number of Terms')
	plt.xticks(range(len(merged_topics)))
	plt.savefig('merged_topic_distribution.png')
	
	# Flatten and return unique topics - FIXED LINE BELOW
	flat_topics = set(word for topic in merged_topics for word in topic)
	print(f"Extracted {len(flat_topics)} unique topic terms after merging")
	
	return merged_topics, flat_topics

def clean_labels(labels):
		"""Clean and filter candidate labels"""
		cleaned = set()
		for label in labels:
				# Normalize the label
				label = label.lower().strip()
				# Remove non-alphanumeric characters except spaces and hyphens
				label = re.sub(r"[^a-z0-9\s\-]", "", label)
				# Skip short labels and stopwords
				if label in CUSTOM_STOPWORDS or len(label) < 3:
						continue
				# Skip labels that are just numbers
				if label.isdigit():
						continue
				# Skip labels that start with numbers unless they're years (4 digits)
				if label[0].isdigit() and not (len(label) == 4 and label.isdigit()):
						continue
				# Skip non-English labels
				if not all(ord(char) < 128 for char in label):
						continue
				# Add the cleaned label
				cleaned.add(label)
		return sorted(cleaned)

def extract_named_entities(text):
	# Skip if text is not primarily English
	if not is_english(text):
		return []

	try:
		ner_results = nlp(text)
		entities = []
		for entity in ner_results:
			if entity["entity_group"] in RELEVANT_ENTITY_TYPES and entity["word"].isalpha():
				# Clean and normalize entity text
				entity_text = re.sub(r'[^\w\s\-]', '', entity["word"].lower()).strip()
				if (entity_text and len(entity_text) > 2 and 
					entity_text not in CUSTOM_STOPWORDS and 
					all(ord(char) < 128 for char in entity_text)):
					entities.append(entity_text)
		# Also extract multi-word phrases that might be significant
		tokens = [
			word.lower() for word in text.split() 
			if word.isalpha() and len(word) > 2 
			and word.lower() not in CUSTOM_STOPWORDS
			and all(ord(char) < 128 for char in word)
		]
				
		# Return unique list of entities and tokens
		return list(set(entities + tokens))
	except Exception as e:
		print(f"NER error: {e}")
		return []

def extract_keywords(text, min_count=3):
		"""Extract keywords based on TF-IDF"""
		if not is_english(text) or len(text) < 10:
				return []
				
		vectorizer = TfidfVectorizer(
				max_df=0.9,
				min_df=min_count/len([text]),
				stop_words=CUSTOM_STOPWORDS,
				ngram_range=(1, 2)
		)

		try:
				X = vectorizer.fit_transform([text])
				feature_names = vectorizer.get_feature_names_out()
				# Get scores for the first document
				scores = zip(feature_names, X.toarray()[0])
				# Sort by score in descending order
				sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
				# Return top keywords
				top_k = 20  # Get more keywords initially for further filtering
				keywords = [kw for kw, score in sorted_scores[:top_k] 
									 if score > 0.01 and all(ord(char) < 128 for char in kw)]
				return keywords
		except:
				return []

def filter_metadata_terms(labels):
		"""Filter out metadata-specific terms"""
		metadata_fragments = ['kunststoff', 'bildetekst', 'arkiv', 'quer', 'riksarkivet',
													 'museum donated', 'association', 'information received']
		
		return [label for label in labels if not any(frag in label for frag in metadata_fragments)]

def handle_multilingual_labels(labels):
		"""Process multilingual labels consistently"""
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

def deduplicate_labels(labels):
		"""Remove semantically redundant labels"""
		if not labels:
				return []
		
		# Sort labels by length (typically longer labels are more specific)
		sorted_labels = sorted(labels, key=len, reverse=True)
		deduplicated = []
		
		for label in sorted_labels:
				# Check if this label is a substring of any already-kept label
				is_redundant = False
				for kept_label in deduplicated:
						# If label is completely contained in a kept label, it's redundant
						if label in kept_label:
								is_redundant = True
								break
								
						# Check for high token overlap in multi-word labels
						if ' ' in label and ' ' in kept_label:
								label_tokens = set(label.split())
								kept_tokens = set(kept_label.split())
								
								# If 80% or more tokens overlap, consider redundant
								overlap_ratio = len(label_tokens & kept_tokens) / len(label_tokens)
								if overlap_ratio > 0.8:
										is_redundant = True
										break
				
				if not is_redundant:
						deduplicated.append(label)
		
		return deduplicated

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

def balance_label_count(image_labels_list, min_labels=3, max_labels=10):
		"""Ensure each image has a balanced number of labels"""
		balanced_labels = []
		
		for labels in image_labels_list:
				# Case 1: Too few labels - attempt to generate more
				if len(labels) < min_labels:
						# Create compound labels by combining existing ones
						compound_labels = []
						if len(labels) >= 2:
								for i in range(len(labels)):
										for j in range(i+1, len(labels)):
												compound = f"{labels[i]} {labels[j]}"
												if 3 <= len(compound.split()) <= 4:  # Reasonable phrase length
														compound_labels.append(compound)
						
						# Add compound labels to reach minimum
						expanded_labels = labels + compound_labels
						balanced_labels.append(expanded_labels[:min_labels] if len(expanded_labels) >= min_labels else expanded_labels)
								
				# Case 2: Too many labels - prioritize the most relevant ones
				elif len(labels) > max_labels:
						# Sort labels by potential relevance (length can be a simple heuristic)
						sorted_labels = sorted(labels, key=lambda x: len(x.split()), reverse=True)
						
						# Ensure we keep some single-word labels for searchability
						single_word = [label for label in labels if ' ' not in label][:max_labels//3]
						multi_word = [label for label in sorted_labels if ' ' in label][:max_labels - len(single_word)]
						
						balanced_labels.append(single_word + multi_word)
				
				# Case 3: Good label count - keep as is
				else:
						balanced_labels.append(labels)
		
		return balanced_labels

# ===== OPTIMIZED RELEVANCE FILTERING =====
def quick_filter_candidates(text, labels, max_keep=30):
		"""Quick pre-filtering using simple word overlap"""
		if not labels:
				return []
				
		text_words = set(text.lower().split())
		scores = []
		
		for label in labels:
				label_words = set(label.lower().split())
				# Calculate simple overlap score
				overlap = len(text_words.intersection(label_words))
				scores.append((label, overlap))
		
		# Sort by score and keep top candidates
		sorted_labels = [l for l, s in sorted(scores, key=lambda x: x[1], reverse=True)]
		return sorted_labels[:max_keep]

def batch_filter_by_relevance(
		texts: list,
		all_labels_list: list[list[str]],
		threshold: float=0.3,
		batch_size: int=128,
		print_every=500,
	):
	"""Process document relevance filtering in efficient batches"""
	results = []
	total = len(texts)
	
	# Process in batches to avoid memory issues
	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch_texts = texts[batch_start:batch_end]
		batch_labels_list = all_labels_list[batch_start:batch_end]
		
		print(f"Processing batch {batch_start//batch_size + 1}/{(total-1)//batch_size + 1}...")
		
		# First apply quick filtering to reduce candidates
		quick_filtered_batch = []
		for text, labels in zip(batch_texts, batch_labels_list):
			quick_filtered = quick_filter_candidates(text, labels)
			quick_filtered_batch.append(quick_filtered)
		
		# Get text embeddings for the batch (more efficient)
		text_embeddings = sent_model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size, convert_to_numpy=True)
		
		batch_results = []
		for i, (text_emb, labels) in enumerate(zip(text_embeddings, quick_filtered_batch)):
			if not labels:
				batch_results.append([])
				continue
					
			# Encode all labels at once for this document
			label_embeddings = sent_model.encode(labels, show_progress_bar=False, convert_to_numpy=True)
			
			# Calculate similarities all at once (vectorized operation)
			similarities = np.dot(label_embeddings, text_emb) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8)
			
			# Filter by threshold
			relevant_indices = np.where(similarities > threshold)[0]
			batch_results.append([labels[idx] for idx in relevant_indices])
			
			# # Progress indicator within batch
			# if (i + 1) % print_every == 0 or i + 1 == len(batch_texts):
			# 	print(f"  Processed {i+1}/{len(batch_texts)} documents in current batch")

		results.extend(batch_results)
		print(f"  Completed batch with {sum(len(labels) for labels in batch_results)} relevant labels found")
	
	return results

# ===== PARALLEL PROCESSING SUPPORT =====
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

def parallel_relevance_filtering(texts, all_labels, n_processes=None):
		"""Run relevance filtering in parallel using multiple processes"""
		if n_processes is None:
				n_processes = max(1, multiprocessing.cpu_count() - 1)
		
		total_docs = len(texts)
		chunk_size = total_docs // n_processes + (1 if total_docs % n_processes else 0)
		chunks = []
		
		# Create chunks of data
		for i in range(0, total_docs, chunk_size):
				end_idx = min(i + chunk_size, total_docs)
				chunks.append((i, end_idx, texts[i:end_idx], all_labels[i:end_idx]))
		
		# Process chunks in parallel
		with multiprocessing.Pool(processes=n_processes) as pool:
				chunk_results = pool.map(process_document_chunk, chunks)
		
		# Combine results
		all_results = []
		for chunk in chunk_results:
				all_results.extend(chunk)
		
		return all_results

def process_text_chunk(chunk):
	return [extract_named_entities(text) for text in chunk]

def get_text_based_annotation(
		csv_file: str, 
		use_parallel: bool=False, 
		num_processes: int=16,
		batch_size: int=512,
	):
	print(f"Automatic label extraction from text data".center(150, "-"))
	print(f"Loading metadata from {csv_file}...")
	
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'label_title_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	}
	
	df = pd.read_csv(
		filepath_or_buffer=csv_file, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False,
	)
	print(f"FULL Dataset {type(df)} {df.shape}")
	
	df['content'] = df['label_title_description'].fillna('')

	# Clean text
	print("Cleaning text...")
	df['clean_content'] = df['content'].apply(clean_text)
	
	# Apply enhanced language filtering
	print("Filtering non-English entries...")
	t0 = time.time()
	english_mask = df['clean_content'].apply(is_english)
	print(f"{sum(english_mask)} / {len(df)} texts are English")
	print(f"Language filter done in {time.time() - t0:.1f} sec")
	
	df = df[english_mask].reset_index(drop=True)
	clean_texts = df['clean_content'].tolist()
	
	# Step 1: Topic Modeling with redundancy reduction
	print("Performing topic modeling...")
	t0 = time.time()
	topics, flat_topic_words = extract_semantic_topics(
			texts=clean_texts, 
			n_clusters=min(20, len(clean_texts) // 100 + 5),
			top_k_words=10,
			merge_threshold=0.85 # Merge similar topics
	)
	print(f"{len(topics)} Topics(clusters) {type(topics)}:\n{topics}")
	print(f"Topic modeling done in {time.time() - t0:.1f} sec")
	
	# Step 2: Named Entity Recognition per image
	print("Extracting named entities per image...")
	t0 = time.time()
	if len(clean_texts) > 1000 and use_parallel:
		chunk_size = len(clean_texts) // num_processes + 1
		chunks = [
			(clean_texts[i:i+chunk_size]) 
			for i in range(0, len(clean_texts), chunk_size)
		]
		print(f"Using {num_processes} processes for NER extraction...")
		with multiprocessing.Pool(processes=num_processes) as pool:
			ner_results = pool.map(process_text_chunk, chunks)
				
		per_image_ner_labels = []
		for chunk_result in ner_results:
			per_image_ner_labels.extend(chunk_result)
	else:
		per_image_ner_labels = []
		for i, text in enumerate(tqdm(clean_texts, desc="NER Progress")):
			entities = extract_named_entities(text)
			per_image_ner_labels.append(entities)
					
	print(f"NER done in {time.time() - t0:.1f} sec")
	
	# Step 3: Extract keywords per image
	print("Extracting keywords per image...")
	t0 = time.time()
	per_image_keywords = [extract_keywords(text) for text in clean_texts]
	print(f"Keyword extraction done in {time.time() - t0:.1f} sec")
	
	# Step 4: Add individual topic labels
	print("Assigning topic labels per image...")
	t0 = time.time()
	per_image_topic_labels = []
	for text in clean_texts:
		# text_tokens = set(text.split())
		# Find topic words that appear in the text
		matching_topics = [word for word in flat_topic_words if word in text]
		per_image_topic_labels.append(matching_topics)
	print(f"Topic assignment done in {time.time() - t0:.1f} sec")

	# Step 5: Combine all label sources and clean
	print("Combining and cleaning labels...")
	t0 = time.time()
	per_image_combined_labels = []
	for ner, keywords, topics in zip(per_image_ner_labels, per_image_keywords, per_image_topic_labels):
		# Combine all sources
		all_labels = list(set(ner + keywords + topics))
		# Clean the labels
		cleaned_labels = clean_labels(all_labels)
		# Filter metadata terms
		cleaned_labels = filter_metadata_terms(cleaned_labels)
		per_image_combined_labels.append(cleaned_labels)
	print(f"Label combination and cleaning done in {time.time() - t0:.3f} sec")

	# Step 6: Filter by relevance and handle languages (optimized)
	print("Filtering labels by relevance...")
	t0 = time.time()
	if use_parallel:
		print("Using parallel processing for relevance filtering...")
		per_image_relevant_labels = parallel_relevance_filtering(
			texts=clean_texts,
			all_labels=per_image_combined_labels,
			n_processes=num_processes,
		)
	else:
		print("Using batch processing for relevance filtering...")
		per_image_relevant_labels = batch_filter_by_relevance(
			texts=clean_texts,
			all_labels_list=per_image_combined_labels,
			threshold=0.3,
			batch_size=batch_size,
			print_every=500,
		)
	print(f"Relevance filtering done in {time.time() - t0:.1f} sec")

	# Post-process: language handling, deduplication, etc.
	print("Post-processing labels, deduplication, and semantic categorization...")
	t0 = time.time()
	per_image_labels = []
	for i, relevant_labels in enumerate(tqdm(per_image_relevant_labels, desc="Post-processing", unit="image")):
		# Handle languages
		filtered_labels = handle_multilingual_labels(relevant_labels)
		
		# # Add user query if it exists
		if "user_query" in df.columns:
			original_label = df.iloc[i]["user_query"]
			if isinstance(original_label, str) and original_label.strip():
				original_label_clean = re.sub(r"[^a-z0-9\s\-]", "", original_label.lower().strip())
				if all(ord(char) < 128 for char in original_label_clean):
					filtered_labels.append(original_label_clean)
		
		# Remove redundancy
		filtered_labels = deduplicate_labels(filtered_labels)

		# Add semantic categories
		categorized = assign_semantic_categories(filtered_labels)
		final_labels = sorted(set(filtered_labels + categorized))
		per_image_labels.append(final_labels)
	print(f"Post-processing done in {time.time() - t0:.1f} sec")

	# Balance label counts
	print("Balancing label counts...")
	t0 = time.time()
	per_image_labels = balance_label_count(per_image_labels, min_labels=3, max_labels=12)
	print(f"Label balancing done in {time.time() - t0:.3f} sec")
	# Save the results
	df['generated_labels'] = per_image_labels
	output_path = os.path.join(os.path.dirname(csv_file), "metadata_with_labels.csv")
	df.to_csv(output_path, index=False)
	
	# Print some examples
	print("\nExample results:")
	sample_cols = ['title', 'description', 'label', 'label_title_description', 'generated_labels']
	available_cols = [col for col in sample_cols if col in df.columns]
	for i in range(min(25, len(df))):
		print(f"\nExample {i+1}:")
		for col in available_cols:
			print(f"{col}: {df.iloc[i][col]}")
	
	return per_image_labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--csv_file", type=str, default=full_meta)
	parser.add_argument("--use_parallel", action="store_true")
	parser.add_argument("--num_processes", type=int, default=16)
	parser.add_argument("--batch_size", type=int, default=512)
	args = parser.parse_args()
	multiprocessing.set_start_method('spawn', force=True)

	labels = get_text_based_annotation(
		csv_file=args.csv_file,
		use_parallel=args.use_parallel,
		num_processes=args.num_processes,
		batch_size=args.batch_size,
	)