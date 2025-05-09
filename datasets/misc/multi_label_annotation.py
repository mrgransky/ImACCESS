import numpy as np
import pandas as pd
import re
import os
import time
import torch
import torchvision
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
# import faiss
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import CLIPProcessor, CLIPModel, AlignProcessor, AlignModel

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

from typing import List
from PIL import Image

nltk.download('words', quiet=True)
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# how to run[Pouta]:
# $ nohup python -u multi_label_annotation.py -csv /media/volume/ImACCESS/WW_DATASETs/HISTORY_X4/metadata.csv -nw 50 -tbs 1024 -vbs 512 -vth 0.15 -rth 0.25 > /media/volume/ImACCESS/trash/multi_label_annotation.out &

# Make language detection deterministic
DetectorFactory.seed = 42

# Custom stopwords and metadata patterns
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union(
	{
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
		"collection", "collections", "number", "abbreviation", "abbreviations",
		"folder", "box", "file", "document", "page", "index", "label", "code", "icon", "type", "unknown", "unknow",
		"folder icon", "box icon", "file icon", "document icon", "page icon", "index icon", "label icon", "code icon",
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

FastText_Language_Identification = "lid.176.ftz"
if "lid.176.ftz" not in os.listdir():
	url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
	urllib.request.urlretrieve(url, "lid.176.ftz")

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

def is_english(text: str, ft_model: fasttext.FastText._FastText):
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
		sent_model: SentenceTransformer,
		ft_model: fasttext.FastText._FastText,
		texts: list[str],
		n_clusters: int,
		top_k_words: int,
		merge_threshold: float,
	):
	# Generate embeddings
	print(f"Generating embeddings for {len(texts)} texts...")
	embeddings = sent_model.encode(texts, show_progress_bar=True)
	
	# Clustering with increased default clusters
	print(f"Clustering embeddings {embeddings.shape} into {n_clusters} topics with KMeans...")
	kmeans = KMeans(
		n_clusters=n_clusters, 
		random_state=42, 
		init='k-means++', 
		max_iter=1000, 
		n_init='auto',
	)
	labels = kmeans.fit_predict(embeddings)
	
	# Plot original distribution
	topic_counts = Counter(labels)
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(topic_counts)), [topic_counts[i] for i in range(n_clusters)])
	plt.title('Document Distribution Across Topics')
	plt.xlabel('Topic ID')
	plt.ylabel('Number of Documents')
	plt.xticks(range(n_clusters))
	plt.savefig(f'topic_distribution_before_merging_topics_{n_clusters}_topics_{top_k_words}_words_per_topic_per_cluster.png')
	
	# Collect phrases for each cluster
	cluster_phrases = defaultdict(Counter)
	for i, label in enumerate(labels):
		# Check if the text is English before extracting phrases
		if is_english(text=texts[i], ft_model=ft_model):
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
		most_common = [
			w 
			for w, c in counter.most_common(top_k_words * 4) 
			if len(w.split()) <= 3 and all(ord(char) < 128 for char in w)
		]
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
	plt.savefig(f'merged_topic_distribution_{merge_threshold}_merge_threshold_{top_k_words}_top_k_words_per_topic_per_cluster_{n_clusters}_clusters_per_topic.png')
	
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

def extract_named_entities(nlp: pipeline, text: str, ft_model: fasttext.FastText._FastText):
	# Skip if text is not primarily English
	if not is_english(text=text, ft_model=ft_model):
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

def extract_keywords(
		text: str, 
		ft_model: fasttext.FastText._FastText, 
		min_count: int=3, 
		max_df: float=0.9, 
		min_df: float=0.01, 
		max_features: int=1000, 
		ngram_range: tuple=(1, 3), 
		top_k: int=50
	):
	if not is_english(text, ft_model) or len(text) < 5:
		return []
			
	vectorizer = TfidfVectorizer(
		# max_df=max_df,
		# min_df=min_df,
		# max_features=max_features,
		ngram_range=ngram_range,
		stop_words=CUSTOM_STOPWORDS,
	)
	try:
		X = vectorizer.fit_transform([text])
		feature_names = vectorizer.get_feature_names_out()
		# Get scores for the first document
		scores = zip(feature_names, X.toarray()[0])
		# Sort by score in descending order
		sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
		# Return top keywords
		keywords = [
			kw for kw, score in sorted_scores[:top_k] 
			if score > 0.01 and all(ord(char) < 128 for char in kw)
		]
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

				# If 75% or more tokens overlap, consider redundant
				overlap_ratio = len(label_tokens & kept_tokens) / len(label_tokens)
				if overlap_ratio > 0.75:
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
		sent_model: SentenceTransformer,
		texts: list,
		all_labels_list: list[list[str]],
		threshold: float=0.3,
		batch_size: int=128,
		print_every=10,
	):
	"""Process document relevance filtering in efficient batches"""
	results = []
	total = len(texts)
	
	# Process in batches to avoid memory issues
	for batch_start in range(0, total, batch_size):
		batch_end = min(batch_start + batch_size, total)
		batch_texts = texts[batch_start:batch_end]
		batch_labels_list = all_labels_list[batch_start:batch_end]

		
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

		if batch_start % print_every == 0 or batch_start == 0:
			print(f"batch {batch_start//batch_size + 1}/{(total-1)//batch_size + 1} with {sum(len(labels) for labels in batch_results)} relevant labels found")
	
	return results

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

def process_text_chunk(nlp, chunk):
	return [extract_named_entities(nlp=nlp, text=text) for text in chunk]

def get_textual_based_annotation(
		csv_file: str, 
		num_workers: int,
		batch_size: int,
		num_clusters: int,
		top_k_words: int,
		relevance_threshold: float,
		merge_threshold: float,
		metadata_fpth: str,
		device: str,
		use_parallel: bool=False,
	):
	print(f"Automatic label extraction from text data".center(160, "-"))
	print(f"Loading metadata from {csv_file}...")
	text_based_annotation_start_time = time.time()

	sent_model = SentenceTransformer("all-MiniLM-L6-v2")
	ft_model = fasttext.load_model("lid.176.ftz")
	tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
	model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
	nlp = pipeline(
		task="ner", 
		model=model, 
		tokenizer=tokenizer, 
		aggregation_strategy="simple",
		device=device,
		batch_size=batch_size,
	)

	# Load the full dataset
	dtypes = {
		'doc_id': str, 'id': str, 'label': str, 'title': str,
		'description': str, 'img_url': str, 'enriched_document_description': str,
		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
	}
	
	df = pd.read_csv(
		filepath_or_buffer=csv_file, 
		on_bad_lines='skip',
		dtype=dtypes, 
		low_memory=False,
	)

	print(f"FULL Dataset {type(df)} {df.shape}\n{list(df.columns)}")
	
	df['content'] = df['enriched_document_description'].fillna('')
	num_samples = len(df)
	
	# Clean text
	print("Cleaning text...")
	df['clean_content'] = df['content'].apply(clean_text)
	
	print("Filtering non-English entries...")
	t0 = time.time()
	english_mask = df['clean_content'].apply(lambda x: is_english(x, ft_model))
	english_indices = english_mask[english_mask].index.tolist()
	print(f"{sum(english_mask)} / {len(df)} texts are English")
	print(f"Language filter done in {time.time() - t0:.2f} sec")
	
	# Create a filtered dataframe with only English entries
	english_df = df[english_mask].reset_index(drop=True)
	english_texts = english_df['clean_content'].tolist()
	
	# Initialize results for all entries with empty lists
	per_image_labels = [[] for _ in range(num_samples)]
	
	if len(english_texts) > 0:
		# Step 1: Topic Modeling with redundancy reduction
		print("Topic Modeling".center(160, "-"))
		t0 = time.time()
		topics, flat_topic_words = extract_semantic_topics(
			sent_model=sent_model,
			ft_model=ft_model,
			texts=english_texts,
			n_clusters=min(num_clusters, len(english_texts) // 100 + 5),
			top_k_words=top_k_words,
			merge_threshold=merge_threshold, # Merge similar topics
		)
		print(f"{len(topics)} Topics(clusters) {type(topics)}:\n{topics}")
		print(f"Elapsed_t: {time.time()-t0:.2f} sec".center(160, "-"))
		
		# Step 2: Named Entity Recognition per image
		print("Extracting NER per sample...")
		t0 = time.time()
		if len(english_texts) > 1000 and use_parallel:
			chunk_size = len(english_texts) // num_workers + 1
			chunks = [
				(english_texts[i:i+chunk_size]) 
				for i in range(0, len(english_texts), chunk_size)
			]
			print(f"Using {num_workers} processes for NER extraction...")
			with multiprocessing.Pool(processes=num_workers) as pool:
				ner_results = pool.map(process_text_chunk, chunks)
					
			per_image_ner_labels = []
			for chunk_result in ner_results:
				per_image_ner_labels.extend(chunk_result)
		else:
			per_image_ner_labels = []
			for i, text in enumerate(tqdm(english_texts, desc="NER Progress")):
				entities = extract_named_entities(nlp=nlp, text=text, ft_model=ft_model)
				per_image_ner_labels.append(entities)
		print(f"NER done in {time.time() - t0:.1f} sec")
		
		# Step 3: Extract keywords per image
		print("Extracting keywords per image using TF-IDF...")
		t0 = time.time()
		per_image_keywords = [extract_keywords(text, ft_model) for text in english_texts]
		print(f"Keyword extraction done in {time.time() - t0:.1f} sec")
		
		# Step 4: Add individual topic labels
		print("Assigning topic labels per image...")
		t0 = time.time()
		per_image_topic_labels = []
		for text in english_texts:
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
				texts=english_texts,
				all_labels=per_image_combined_labels,
				n_processes=num_workers,
			)
		else:
			print(f"Using batch processing for relevance filtering (thresh: {relevance_threshold})...")
			per_image_relevant_labels = batch_filter_by_relevance(
				sent_model=sent_model,
				texts=english_texts,
				all_labels_list=per_image_combined_labels,
				threshold=relevance_threshold,
				batch_size=batch_size,
				print_every=500,
			)
		print(f"Relevance filtering done in {time.time() - t0:.1f} sec")
		# Post-process: language handling, deduplication, etc.
		print("Post-processing labels, deduplication, and semantic categorization...")
		t0 = time.time()
		english_labels = []
		for i, relevant_labels in enumerate(tqdm(per_image_relevant_labels, desc="Post-processing", unit="image")):
			filtered_labels = handle_multilingual_labels(relevant_labels)
			
			if "user_query" in english_df.columns:
				original_label = english_df.iloc[i]["user_query"]
				if isinstance(original_label, str) and original_label.strip():
					original_label_clean = re.sub(r"[^a-z0-9\s\-]", "", original_label.lower().strip())
					if all(ord(char) < 128 for char in original_label_clean):
						filtered_labels.append(original_label_clean)
			
			filtered_labels = deduplicate_labels(filtered_labels)
			# Add semantic categories
			categorized = assign_semantic_categories(filtered_labels)
			final_labels = sorted(set(filtered_labels + categorized))
			english_labels.append(final_labels)
		print(f"Post-processing done in {time.time() - t0:.1f} sec")
		# Balance label counts
		print("Balancing label counts...")
		t0 = time.time()
		english_labels = balance_label_count(english_labels, min_labels=3, max_labels=12)
		print(f"Label balancing done in {time.time() - t0:.3f} sec")
		
		# Transfer results to the full-sized results array using original indices
		for i, orig_idx in enumerate(english_indices):
			if i < len(english_labels):
				per_image_labels[orig_idx] = english_labels[i]
	else:
		print("No English texts found. Returning empty labels for all entries.")
	
	# Save the results in a separate column
	df['textual_based_labels'] = per_image_labels
	
	df.to_csv(metadata_fpth, index=False)
	
	print(f">> Generated text labels for {sum(1 for labels in per_image_labels if labels)} out of {num_samples} entries")
	print(f"Text-based annotation Elapsed time: {time.time() - text_based_annotation_start_time:.2f} sec".center(160, " "))
	
	return per_image_labels

def get_visual_based_annotation(
		csv_file: str,
		confidence_threshold: float,
		batch_size: int,
		device: str,
		verbose: bool,
		metadata_fpth: str,
	) -> List[List[str]]:
	print(f"Automatic label extraction from image data".center(160, "-"))
	start_time = time.time()
	
	# Load dataset
	if verbose:
		print(f"Loading metadata from {csv_file}...")
	df = pd.read_csv(csv_file, dtype={'img_path': str}, low_memory=False)
	image_paths = df['img_path'].tolist()
	if verbose:
		print(f"Found {len(image_paths)} samples to process...")
	
	if verbose:
		print("Loading VLM model for image labeling...")
	# model_name = "openai/clip-vit-large-patch14"
	# model = CLIPModel.from_pretrained(model_name).to(device)
	# processor = CLIPProcessor.from_pretrained(model_name)
	processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
	model = AlignModel.from_pretrained("kakaobrain/align-base")
	model.to(device) # Add this line to move the model to the GPU
	model.eval()

	object_categories = [
		# Tanks & Armored Vehicles (WWI-WWII)
		"tank", "light tank", "medium tank", "heavy tank", "super-heavy tank", 
		"tank destroyer", "self-propelled gun", "armored car", "half-track", 
		"armored personnel carrier", "armored train", "reconnaissance vehicle",
		"Mark IV tank", "Tiger tank", "Panther tank", "T-34 tank", "Sherman tank",
		"Churchill tank", "KV-1 tank", "Panzer IV", "Panzer III", "Stuart tank",
		"Sonderkraftfahrzeug", "Kettenkrad", "M4 Sherman", "T-34/85", "IS-2 tank",
		
		# Light & Utility Vehicles
		"jeep", "staff car", "command car", "ambulance", "motorcycle", 
		"military truck", "supply truck", "fuel truck", "artillery tractor", 
		"amphibious vehicle", "scout car", "Willys Jeep", "Kubelwagen", 
		"Dodge WC series", "Opel Blitz", "Zis truck", "weapons carrier",
			
		# Aircraft
		"military aircraft", "fighter aircraft", "bomber aircraft", "reconnaissance aircraft", 
		"dive bomber", "torpedo bomber", "transport aircraft", "seaplane", "flying boat",
		"biplane", "monoplane", "fighter-bomber", "ground attack aircraft", "night fighter",
		"Spitfire", "Messerschmitt Bf 109", "P-51 Mustang", "Focke-Wulf Fw 190", 
		"B-17 Flying Fortress", "Lancaster bomber", "Heinkel He 111", "Junkers Ju 87 Stuka",
		"Mitsubishi Zero", "Il-2 Sturmovik", "P-47 Thunderbolt", "Hurricane fighter", "helicopter",
		"aircraft engine", "aircraft propeller", "aircraft wing", "aircraft fuselage", "aircraft tail",
		"aircraft manufacturing company",
	
		# Naval Vessels
		"submarine", "U-boat", "destroyer", "cruiser", "battleship", "aircraft carrier", 
		"battlecruiser", "corvette", "frigate", "minesweeper", "torpedo boat", 
		"landing craft", "PT boat", "pocket battleship", "gunboat", "escort carrier",
		"liberty ship", "merchant vessel", "hospital ship", "troop transport",
			
		# Military Personnel
		"soldier", "infantryman", "officer", "NCO", "general", "field marshal",
		"pilot", "bomber crew", "aircraft crew", "tanker", "artilleryman", "sailor", "marine", 
		"paratrooper", "commando", "sniper", "medic", "military police", 
		"cavalry", "SS officer", "Wehrmacht soldier", "Red Army soldier", 
		"Desert Rat", "Afrika Korps soldier", "Luftwaffe personnel", "naval officer",
		
		# Weapons & Ordnance
		"rifle", "machine gun", "submachine gun", "pistol", "bayonet", "flamethrower", 
		"mortar", "artillery piece", "howitzer", "field gun", "anti-tank gun", "cannon", 
		"anti-aircraft gun", "rocket launcher", "grenade", "hand grenade", "rifle grenade",
		"landmine", "naval mine", "depth charge", "torpedo", "aerial bomb", "incendiary bomb",
		"Thompson submachine gun", "MG-42", "Karabiner 98k", "M1 Garand", "Sten gun",
		"Luger pistol", "PIAT", "Bazooka", "Panzerfaust", "88mm gun",
		
		# Military Infrastructure
		"bunker", "pillbox", "gun emplacement", "observation post", "barbed wire", 
		"trenches", "foxhole", "dugout", "fortification", "coastal defense", 
		"anti-tank obstacle", "dragon's teeth", "minefield", "floating bridge",
		"portable bridge", "military headquarters", "command post", "communications center",
		
		# Military Insignia & Symbols
		"military flag", "swastika flag", "rising sun flag", "Soviet flag", "Union Jack", 
		"American flag", "regimental colors", "military insignia", "rank insignia", 
		"unit patch", "medal", "military decoration", "Iron Cross", "Victoria Cross",
		"Medal of Honor", "military helmet", "steel helmet", "Brodie helmet",
		"Stahlhelm", "Adrian helmet", "gas mask",
		
		# Military Equipment
		"military uniform", "combat uniform", "field equipment", "backpack", "mess kit", 
		"entrenching tool", "canteen", "ammunition belt", "bandolier", "map case", 
		"binoculars", "field telephone", "radio equipment", "signal equipment",
		"parachute", "life vest", "fuel drum", "jerry can", "ration box",
		"military stretcher", "field kitchen", "anti-gas equipment"
	]

	scene_categories = [
		# European Theaters
		"Western Front", "Eastern Front", "Italian Front", "North African Front",
		"Normandy beaches", "coastline", "Soviet urban ruins",
		
		# Pacific & Asian Theaters
		"Pacific island", "jungle battlefield", "Pacific beach landing", "atoll",
		"tropical forest", "coral reef", "bamboo grove", "rice paddy",
		"Burmese jungle", "Chinese village", "Philippine beach", "volcanic island",
		"Japanese homeland", "Pacific airfield", "jungle airstrip", "coconut plantation",
		
		# Military Settings
		"prisoner of war camp", "concentration camp", "military hospital", "field hospital",
		"military cemetery", "aircraft factory", "tank factory", "shipyard",
		"military depot", "ammunition dump", "fuel depot", "supply dump",
		"military port", "embarkation point", "submarine pen", "naval dry dock",
		
		# Terrain Types
		"desert", "desert oasis", "desert dunes", "rocky desert", "forest",
		"dense forest", "winter forest", "urban area", "bombed city", "city ruins",
		"beach", "landing beach", "rocky beach", "mountain", "mountain pass",
		"field", "farm field", "snow-covered field", "ocean", "open ocean",
		"coastal waters", "river", "river crossing", "flooded river", "bridge",
		
		# Military Infrastructure
		"airfield", "temporary airstrip", "bomber base", "fighter base",
		"naval base", "submarine base", "army barracks", "training camp",
		"military headquarters", "command bunker", "coastal defense",
		"fortified line", "defensive position", "artillery position",
		
		# Military Activities
		"battlefield", "active battlefield", "battlefield aftermath",
		"military parade", "victory parade", "surrender ceremony",
		"military exercise", "amphibious landing exercise", "tank maneuvers",
		"war zone", "civilian evacuation", "occupation zone", "frontline",
		"military checkpoint", "border checkpoint", "military convoy route",
		
		# Home Front
		"war factory", "armaments factory", "aircraft assembly line",
		"vehicle assembly line", "shipyard", "munitions factory",
		"civilian air raid shelter", "bombed civilian area", "rationing center",
		"recruitment office", "propaganda poster display", "war bonds office",
		"civil defense drill", "air raid aftermath", "victory celebration"
	]

	era_categories = [
		# Pre-War & Early War
		"pre-World War I era", "World War I era", "interwar period",
		"pre-1939 military",
		"Phoney War", "Blitzkrieg era", "1939-1940 military equipment",
		
		# World War II Specific Periods
		"World War II era", "Battle of Britain era", "North African campaign",
		"Pearl Harbor era", "Midway period", "Stalingrad era", "Normandy invasion",
		"Operation Barbarossa", "Battle of the Bulge", "Italian campaign",
		"D-Day preparations", "Market Garden operation", "Fall of Berlin",
		"Island-hopping campaign", "Battle of the Atlantic", "V-E Day era",
		"Pacific War late stage", "atomic bomb era", "Japanese surrender period",
		
		# Post-War Periods
		"immediate post-war", "occupation period", "Cold War",
		"Korean War era", "1950s military", "Vietnam era", "Japanse World War era",
					
		# Military Technology Eras
		"early tank warfare", "biplane era", "early radar period", "monoplane transition",
		"early jet aircraft", "V-weapon period", "heavy bomber era", "aircraft carrier warfare",
		"submarine warfare golden age", "amphibious assault development",
		"mechanized warfare", "combined arms doctrine", "early nuclear era",
		
		# National Military Period Styles
		"Wehrmacht prime", "Soviet military buildup", "British Empire forces",
		"American war production peak", "Imperial Japanese forces",
		"Nazi Germany zenith", "Allied powers ascendancy", "Axis powers decline",
		"Red Army resurgence", "American military dominance"
	]

	activity_categories = [
		# Combat Activities
		"fighting", "tank battle", "infantry assault", "naval engagement",
		"aerial dogfight", "bombing run", "strafing run", "artillery barrage",
		"firing weapon", "machine gun firing", "mortar firing", "shelling",
		"anti-aircraft firing", "sniper activity", "flamethrower attack",
		"bayonet charge", "hand-to-hand combat", "urban combat", "house-to-house fighting",
		
		# Movement & Transportation
		"driving", "convoy movement", "tank column", "troop transport",
		"marching", "infantry advance", "tactical retreat", "military withdrawal",
		"flying", "air patrol", "reconnaissance flight", "bombing mission",
		"parachute drop", "airborne landing", "glider landing", "air resupply",
		"crossing terrain", "river crossing", "beach landing", "amphibious assault",
		"fording stream", "mountain climbing", "moving through jungle",
		"naval convoy", "fleet movement", "submarine patrol", "naval blockade",
		
		# Military Operations
		"digging trenches", "building fortifications", "laying mines", "clearing mines",
		"constructing bridge", "demolishing bridge", "breaching obstacles",
		"setting up artillery", "camouflaging position", "establishing perimeter",
		"setting up command post", "establishing field hospital", "creating airstrip",
		
		# Logistics & Support
		"loading equipment", "unloading equipment", "loading ammunition", "refueling",
		"resupplying troops", "distributing rations", "loading wounded", "evacuating casualties",
		"loading ships", "unloading landing craft", "airdrop receiving", "gathering supplies",
		"towing disabled vehicle", "vehicle recovery", "aircraft maintenance", "tank repair",
		"weapon cleaning", "equipment maintenance", "vehicle maintenance",
		
		# Military Life & Routines
		"training", "infantry drill", "weapons training", "tank crew training", "pilot training",
		"field exercise", "receiving briefing", "map reading", "radio communication",
		"standing guard", "sentry duty", "prisoner handling", "military inspection",
		"cooking field rations", 'military rations', "eating meal", "resting between battles", "writing letters",
		"medical treatment", "field surgery", "distributing supplies", "receiving orders",
		
		# Ceremonial & Administrative
		"military parade", "award ceremony", "flag raising", "surrender ceremony", 
		"prisoner processing", "military funeral", "military wedding", "religious service",
		"officer briefing", "signing documents", "military trial", "propaganda filming",
		"press conference", "VIP visit", "civilian interaction", "occupation duty",
		"war crime investigation", "reconnaissance reporting"
	]

	# Function to process image batches
	def process_batch(batch_paths, categories):
		valid_images = []
		valid_indices = []
		
		# Load images
		for i, path in enumerate(batch_paths):
			try:
				if os.path.exists(path):
					img = Image.open(path).convert('RGB')
					valid_images.append(img)
					valid_indices.append(i)
			except Exception as e:
				if verbose:
					print(f"Error loading image {path}: {e}")
		
		if not valid_images:
			return [[] for _ in range(len(batch_paths))]
				
		# Prepare text prompts
		text_prompts = [f"a photo of {cat}" for cat in categories]
		
		# Process with VLM
		with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
			# Prepare inputs
			inputs = processor(
				text=text_prompts,
				images=valid_images, 
				return_tensors="pt", 
				padding=True
			).to(device)
			
			# Get embeddings
			outputs = model(**inputs)
			
			# Normalize embeddings
			image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
			text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
			
			# Calculate similarity scores
			similarity = (100.0 * image_embeds @ text_embeds.T).softmax(dim=-1)
			
			# Create results
			batch_results = [[] for _ in range(len(batch_paths))]
			
			# Extract predictions above threshold
			for img_idx, similarities in enumerate(similarity.cpu().numpy()):
				batch_idx = valid_indices[img_idx]
				for cat_idx, score in enumerate(similarities):
					if score > confidence_threshold:
						batch_results[batch_idx].append(categories[cat_idx])

		return batch_results
	
	# Process images in batches
	all_labels = []
	
	# 1. Visual Object Detection
	if verbose:
		print("Performing visual object detection...")
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Object Detection"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, object_categories)
		all_labels.extend(batch_results)
	
	# 2. Scene Classification
	if verbose:
		print("Performing scene classification...")
	scene_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Scene Classification"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, scene_categories)
		scene_labels.extend(batch_results)
	
	# 3. Era Detection (Temporal Visual Cues)
	if verbose:
		print("Detecting temporal visual cues...")
	era_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Era Detection"):
		batch_paths = image_paths[i:i+batch_size]
		# Lower threshold for era detection (more challenging)
		batch_results = process_batch(batch_paths, era_categories)
		era_labels.extend(batch_results)
	
	# 4. Activity Recognition (Visual Relationship Detection)
	if verbose:
		print("Detecting visual relationships and activities...")
	activity_labels = []
	for i in tqdm(range(0, len(image_paths), batch_size), desc="Activity Recognition"):
		batch_paths = image_paths[i:i+batch_size]
		batch_results = process_batch(batch_paths, activity_categories)
		activity_labels.extend(batch_results)
	
	# Combine all visual annotations
	combined_labels = []
	for i in range(len(image_paths)):
		image_labels = []
		
		# Add object labels if available
		if i < len(all_labels):
			image_labels.extend(all_labels[i])
		
		# Add scene labels if available
		if i < len(scene_labels):
			image_labels.extend(scene_labels[i])
		
		# Add era labels if available
		if i < len(era_labels):
			image_labels.extend(era_labels[i])
		
		# Add activity labels if available
		if i < len(activity_labels):
			image_labels.extend(activity_labels[i])
		
		# Remove duplicates and sort
		combined_labels.append(sorted(set(image_labels)))
	
	if verbose:
		total_labels = sum(len(labels) for labels in combined_labels)
		print(f"Vision-based annotation completed in {time.time() - start_time:.2f} seconds")
		print(f"Generated {total_labels} labels for {len(image_paths)} images")
		print(f"Average labels per image: {total_labels/len(image_paths):.2f}")

	df['visual_based_labels'] = combined_labels
	df.to_csv(metadata_fpth, index=False)
	print(f"Visual-based annotation Elapsed time: {time.time() - start_time:.2f} sec".center(160, " "))

	return combined_labels

def main():
	parser = argparse.ArgumentParser(description="Multi-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--use_parallel", '-parallel', action="store_true")
	parser.add_argument("--num_workers", '-nw', type=int, default=10)
	parser.add_argument("--num_text_clusters", '-nc', type=int, default=30)
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=512)
	parser.add_argument("--top_k_words", '-tkw', type=int, default=25)
	parser.add_argument("--merge_threshold", '-mt', type=float, default=0.8)
	parser.add_argument("--vision_batch_size", '-vbs', type=int, default=16, help="Batch size for vision processing")
	parser.add_argument("--relevance_threshold", '-rth', type=float, default=0.25, help="Relevance threshold for text-based filtering")
	parser.add_argument("--vision_threshold", '-vth', type=float, default=0.20, help="Confidence threshold for VLM-based filtering")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)

	base_dir = os.path.dirname(args.csv_file)
	text_output_path = os.path.join(base_dir, "metadata_textual_based_labels.csv")
	vision_output_path = os.path.join(base_dir, "metadata_visual_based_labels.csv")
	combined_output_path = os.path.join(base_dir, "metadata_multimodal_labels.csv")
	
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
			num_clusters=args.num_text_clusters,
			top_k_words=args.top_k_words,
			merge_threshold=args.merge_threshold,
			relevance_threshold=args.relevance_threshold,
			metadata_fpth=text_output_path,
			device=args.device,
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
			batch_size=args.vision_batch_size,
			verbose=True,
			confidence_threshold=args.vision_threshold,
			device=args.device,
			metadata_fpth=vision_output_path,
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
	# Print some examples
	print("\nExample results:")
	sample_cols = ['title', 'description', 'label', 'img_url', 'textual_based_labels', 'visual_based_labels', 'multimodal_labels']
	available_cols = [col for col in sample_cols if col in df.columns]
	for i in range(min(5, len(df))):
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