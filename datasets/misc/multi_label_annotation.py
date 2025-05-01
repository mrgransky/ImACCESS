import numpy as np
import pandas as pd
import re
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import spacy
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, DetectorFactory
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DetectorFactory.seed = 42  # Make language detection deterministic
DATASET_DIRECTORY = {
	"farid": "/home/farid/datasets/WW_DATASETs/HISTORY_X3",
	# "farid": "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31",
	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4",
	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs/HISTORY_X4",
	"alijani": "/lustre/sgn-data/ImACCESS/WW_DATASETs/HISTORY_X4",
}
full_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata.csv")
train_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_train.csv")
val_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_val.csv")

# Optional: use a domain-extended stopword list
CUSTOM_STOPWORDS = ENGLISH_STOP_WORDS.union({
	"original", "bildetekst", "photo", "image", "archive", "arkivreferanse",
	"copyright", "description", "riksarkivet", "ntbs", "ra", "pa", "bildetekst",
})
# Load English language model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_labels(labels):
	cleaned = set()
	for label in labels:
		label = label.lower().strip()
		label = re.sub(r"[^a-z0-9\s\-]", "", label)
		if label in CUSTOM_STOPWORDS or len(label) < 3:
			continue
		cleaned.add(label)
	return sorted(cleaned)

def extract_semantic_topics(texts, n_clusters=15, top_k_words=5):
	embeddings = sent_model.encode(texts, show_progress_bar=True)
	kmeans = KMeans(n_clusters=n_clusters, random_state=42)
	labels = kmeans.fit_predict(embeddings)
	cluster_phrases = defaultdict(Counter)
	for i, label in enumerate(labels):
		words = [
			word.lower() for word in texts[i].split()
			if word.lower() not in CUSTOM_STOPWORDS and len(word) > 3
		]
		cluster_phrases[label].update(words)
	topics = []
	for label, counter in cluster_phrases.items():
		most_common = [w for w, _ in counter.most_common(top_k_words)]
		topics.append(most_common)
	return topics

def is_english(text):
	try:
		return detect(text) == 'en'
	except:
		return False

def clean_text(text):
	text = re.sub(r'[^\w\s]', '', text)
	text = re.sub(r'\s+', ' ', text)
	return text.strip().lower()

def extract_topics(texts, n_topics=15, n_top_words=7):
	vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
	tfidf = vectorizer.fit_transform(texts)
	nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf)
	feature_names = vectorizer.get_feature_names_out()
	topics = []
	for topic_vec in nmf.components_:
		topic_keywords = [feature_names[i] for i in topic_vec.argsort()[:-n_top_words - 1:-1]]
		topics.append(topic_keywords)
	return set(word for topic in topics for word in topic)

def extract_named_entities(text: str):
	doc = nlp(text)
	return list({ent.text.lower() for ent in doc.ents})

def generate_labels_per_image(csv_file, title_col='title', desc_col='description'):
	df = pd.read_csv(csv_file)
	titles = df[title_col].fillna('').astype(str).apply(clean_text).tolist()
	descriptions = df[desc_col].fillna('').astype(str).apply(clean_text).tolist()
	full_texts = [f"{title} {desc}" for title, desc in zip(titles, descriptions)]

	# Language filtering
	print("Filtering non-English entries...")
	t0 = time.time()
	english_mask = [is_english(text) for text in full_texts]
	print(f"{sum(english_mask)} / {len(full_texts)} texts are English")
	print(f"Language filter done in {time.time() - t0:.1f} sec")

	df = df[english_mask].reset_index(drop=True)
	full_texts = [text for i, text in enumerate(full_texts) if english_mask[i]]

	# Step 1: Named Entities per image
	print("Extracting named entities per image...")
	t0 = time.time()
	per_image_ner_labels = [extract_named_entities(text) for text in full_texts]
	print(f"NER done in {time.time() - t0:.1f} sec")

	# Step 2: Global Topic Modeling
	print("Fitting topic model...")
	t0 = time.time()
	# topics = extract_topics(full_texts, n_topics=15, n_top_words=7)
	topics = extract_semantic_topics(texts=full_texts, n_clusters=15, top_k_words=7)
	flat_topic_words = set(word for topic in topics for word in topic)
	print(f"Topic model done in {time.time() - t0:.1f} sec")

	# Step 3: Assign topic labels per image
	print("Assigning topic labels per image...")
	per_image_topic_labels = [
		list(set(text.split()) & flat_topic_words)
		for text in full_texts
	]

	# Step 4: Combine NER + Topic labels
	per_image_labels = [
		sorted(set(ner + topic))
		for ner, topic in zip(per_image_ner_labels, per_image_topic_labels)
	]

	print("Saving labels to CSV...")
	df['generated_labels'] = per_image_labels
	df.to_csv(os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_with_labels.csv"), index=False)
	print(df[['title', 'description', 'label', 'generated_labels']].head(10))
	return per_image_labels

labels = generate_labels_per_image(csv_file=full_meta)
print(len(labels))
print(labels[0]) # ['eastern front', 'wehrmacht', 'panzer division']