import numpy as np
import pandas as pd
import re
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from collections import Counter
import spacy
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

STOPLIST = {"elite", "combat", "unit", "military", "troops", "forces", "battle"}  # Expand as needed

def filter_stopwords(labels):
    return [label for label in labels if label not in STOPLIST]

# Load English language model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")
sent_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example function to clean and normalize text
def clean_text(text):
	text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
	text = re.sub(r'\s+', ' ', text)  # normalize whitespace
	return text.strip().lower()

# Function to extract named entities using spaCy
def extract_named_entities_old(texts):
	entities = set()
	for text in texts:
		doc = nlp(text)
		for ent in doc.ents:
			entities.add(ent.text.lower())
	return list(entities)

def extract_named_entities(text):
	doc = nlp(text)
	return [ent.text.lower() for ent in doc.ents if len(ent.text.strip()) > 2]

def extract_noun_phrases(text):
	doc = nlp(text)
	return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]

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

def generate_labels_per_image(csv_file, title_col='title', desc_col='description'):
    df = pd.read_csv(csv_file)
    titles = df[title_col].fillna('').astype(str).apply(clean_text).tolist()
    descriptions = df[desc_col].fillna('').astype(str).apply(clean_text).tolist()
    full_texts = [f"{t} {d}" for t, d in zip(titles, descriptions)]

    # Named Entities & Noun Phrases
    ner_labels = [extract_named_entities(text) for text in full_texts]
    noun_labels = [extract_noun_phrases(text) for text in full_texts]

    # Topic model (global keywords)
    print("Fitting topic model...")
    t0 = time.time()
    global_topic_keywords = extract_topics(full_texts)
    print(f"Topic model done in {time.time() - t0:.1f}s")

    # Topic labels per image (intersect with global topics)
    topic_labels = [list(set(text.split()) & global_topic_keywords) for text in full_texts]

    # Combine & filter
    per_image_labels = []
    for ner, noun, topic in zip(ner_labels, noun_labels, topic_labels):
        combined = set(filter_stopwords(ner + noun + topic))
        per_image_labels.append(sorted(combined))

    # Optional: Score or rank labels by frequency
    print("Scoring labels...")
    all_labels = [label for labels in per_image_labels for label in labels]
    label_freq = Counter(all_labels)
    per_image_label_scores = [
        {label: label_freq[label] for label in labels} for labels in per_image_labels
    ]

    df['generated_labels'] = per_image_labels
    df['label_scores'] = per_image_label_scores
    df.to_csv(os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_with_labels.csv"), index=False)
    return per_image_labels

labels = generate_labels_per_image(csv_file=full_meta)
print(len(labels))
print(labels[0]) # ['eastern front', 'wehrmacht', 'panzer division']