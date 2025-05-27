import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from misc.utils import *
from sentence_transformers import SentenceTransformer, util


# def classify_text_with_labels(
# 		text: str,
# 		candidate_labels: List[str],
# 		model: SentenceTransformer,
# 		threshold: float = 0.05,
# 		top_k: int = 5,
# 		return_confidences: bool = False,
# 		verbose: bool = False
# 	) -> Union[List[str], List[Tuple[str, float]]]:

# 	if not text or not isinstance(text, str) or not candidate_labels:
# 		return []
# 	# Encode and normalize
# 	text_emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
# 	label_embs = model.encode(candidate_labels, convert_to_tensor=True, normalize_embeddings=True)
# 	print(text_emb.shape, label_embs.shape)
# 	# Compute cosine similarities
# 	cosine_scores = util.cos_sim(text_emb, label_embs)[0]  # shape: (num_labels,)
# 	# Extract top-K scores
# 	top_k = min(top_k, len(candidate_labels))
# 	top_scores, top_indices = torch.topk(cosine_scores, k=top_k)
# 	results = []

# 	for idx, score in zip(top_indices, top_scores):
# 		sim = score.item()
# 		label = candidate_labels[idx]
# 		if sim >= threshold:
# 			results.append((label, round(sim, 4)) if return_confidences else label)

# 	if verbose:
# 		print(f"\nText: {text}")
# 		print(f"Top-K Label Matches:")
# 		for label, score in zip([candidate_labels[i] for i in top_indices], top_scores):
# 			print(f"- {label:<40} Score: {score.item():.4f}")

# 	return results

# if __name__ == "__main__":
# 	csv_file = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata.csv"
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	# Load dataframe
# 	dtypes = {
# 		'doc_id': str, 'id': str, 'label': str, 'title': str,
# 		'description': str, 'img_url': str, 'enriched_document_description': str,
# 		'raw_doc_date': str, 'doc_year': float, 'doc_url': str,
# 		'img_path': str, 'doc_date': str, 'dataset': str, 'date': str,
# 		'user_query': str,
# 	}
# 	df = pd.read_csv(csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)
# 	TOPK = 10

# 	# st_model_name = "all-mpnet-base-v2"
# 	# st_model_name = "embaas/sentence-transformers-e5-large-v2"
# 	# st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# 	# st_model_name = "sentence-transformers/all-MiniLM-L12-v2"
# 	# st_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 	# st_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# 	st_model_name = "jinaai/jina-embeddings-v3"
# 	# st_model_name = "sentence-transformers/all-roberta-large-v1"

# 	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device, trust_remote_code=True)
# 	CATEGORIES_FILE = "categories.json"
# 	object_categories, scene_categories, activity_categories = load_categories(file_path=CATEGORIES_FILE)
# 	candidate_labels = list(set(object_categories + scene_categories + activity_categories))
# 	labels = candidate_labels
# 	rnd_idx = random.randint(0, len(df))
# 	print(f"obtaining text from row {rnd_idx}...")
# 	text = df.iloc[rnd_idx]['enriched_document_description']
# 	print(text)
# 	classified_labels = classify_text_with_labels(text, labels, sent_model, threshold=0.05, top_k=TOPK, verbose=False)
# 	print(classified_labels)
# 	print(f"-"*100)
# 	classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# 	result = classifier(text, candidate_labels=labels, multi_label=True)
# 	print(result.get("labels")[:TOPK])
# 	print(f"-"*100)
# 	# print(result.get("scores"))
# 	# print(f"-"*100)
# 	print(result.get("sequence"))
# 	print(df.iloc[rnd_idx]["img_path"])
# 	print(df.iloc[rnd_idx]["user_query"])

# 	if USER != "farid":
# 		classifier = pipeline("zero-shot-classification", model='google/flan-t5-xl')
# 		result = classifier(text, candidate_labels=labels, multi_label=True)
# 		print(result.get("labels")[:TOPK])

def get_textual_based_annotation(
		csv_file: str,
		sent_model: SentenceTransformer,
		top_k: int,
		batch_size: int,
	):

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
	
	print("Loading dataframe...")
	df = pd.read_csv(filepath_or_buffer=csv_file, on_bad_lines='skip', dtype=dtypes, low_memory=False)

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
		topk_scores, topk_indices = torch.topk(cosine_scores, k=top_k, dim=1)
		
		# Store results in dataframe
		for i in range(len(batch_df)):
			idx = start_idx + i
			labels = [candidate_labels[j] for j in topk_indices[i]]
			scores = [round(s.item(), 4) for s in topk_scores[i]]
			per_image_labels[idx] = labels
			per_image_labels_scores[idx] = scores

	df['textual_based_labels'] = per_image_labels
	df['textual_based_scores'] = per_image_labels_scores

	# Save results
	print(f"Saving results to {output_csv}...")
	df.to_csv(output_csv, index=False)

	try:
		df.to_excel(output_csv.replace(".csv", ".xlsx"), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	print(f"Completed in {time.time() - start_time:.2f} seconds")
	
	# Show some sample results
	print("\nSample results:")
	for i in range(min(5, len(df))):
		print(f"\nSample {i+1}:")
		print("Text:", df.iloc[i]['enriched_document_description'][:200] + "...")
		print("Top Labels:", df.iloc[i]['textual_based_labels'])
		print("Scores:", df.iloc[i]['textual_based_scores'])

	return per_image_labels

if __name__ == "__main__":
	csv_file = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata.csv"
	output_csv = "/home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_textual_based_labels.csv"
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	TOPK = 3
	BATCH_SIZE = 128
	
	# Load model and labels
	st_model_name = "jinaai/jina-embeddings-v3"
	print(f"Loading model: {st_model_name}...")
	sent_model = SentenceTransformer(model_name_or_path=st_model_name, device=device, trust_remote_code=True)
	
	# Process the entire dataframe
	start_time = time.time()	
	get_textual_based_annotation(csv_file, sent_model, top_k=TOPK, batch_size=BATCH_SIZE)
	
