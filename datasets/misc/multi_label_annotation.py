import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy

DATASET_DIRECTORY = {
"farid": "/home/farid/datasets/WW_DATASETs/HISTORY_X3",
"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4",
"ubuntu": "/media/volume/ImACCESS/WW_DATASETs/HISTORY_X4",
"alijani": "/lustre/sgn-data/ImACCESS/WW_DATASETs/HISTORY_X4",
}
full_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata.csv")
train_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_train.csv")
val_meta = os.path.join(DATASET_DIRECTORY[os.getenv("USER")], "metadata_val.csv")

# Load English language model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Example function to clean and normalize text
def clean_text(text):
	text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
	text = re.sub(r'\s+', ' ', text)  # normalize whitespace
	return text.strip().lower()

# Function to extract named entities using spaCy
def extract_named_entities(texts):
	entities = set()
	for text in texts:
		doc = nlp(text)
		for ent in doc.ents:
			entities.add(ent.text.lower())
	return list(entities)

# Topic modeling to generate candidate labels
def extract_topics(texts, n_topics=5, n_top_words=5):
	vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
	tfidf = vectorizer.fit_transform(texts)
	nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
	feature_names = vectorizer.get_feature_names_out()
	topics = []
	for topic_idx, topic in enumerate(nmf.components_):
		topic_keywords = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
		topics.append(topic_keywords)
	return topics

def generate_labels_from_textual_metadata(csv_file, title_col=2, desc_col=3):
	df = pd.read_csv(csv_file, header=None)
	
	titles = df[title_col].fillna('').astype(str).apply(clean_text).tolist()
	descriptions = df[desc_col].fillna('').astype(str).apply(clean_text).tolist()
	
	full_texts = [f"{title} {desc}" for title, desc in zip(titles, descriptions)]
	
	# Extract labels using named entities
	print(f"Extracting labels using named entities...")
	t0 = time.time()
	named_entity_labels = extract_named_entities(full_texts)
	print(f"Elapsed_t: {time.time()-t0:.1f} sec")

	# Extract labels using topic modeling
	print(f"Extracting labels using topic modeling...")
	topics = extract_topics(full_texts, n_topics=10, n_top_words=5)
	print(f"Elapsed_t: {time.time()-t0:.1f} sec")

	topic_labels = set()
	for topic in topics:
		topic_labels.update(topic)
	combined_labels = set(named_entity_labels).union(topic_labels)

	return sorted(combined_labels)

labels = generate_labels_from_textual_metadata(csv_file=full_meta)
print(type(labels), len(labels))
print(labels)