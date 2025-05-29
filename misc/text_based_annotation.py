import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from misc.utils import *
from sentence_transformers import SentenceTransformer, util

def get_textual_based_annotation(
		csv_file: str,
		sent_model: SentenceTransformer,
		top_k: int,
		batch_size: int,
	):
	start_time = time.time()
	dataset_dir = os.path.dirname(csv_file)
	output_csv = os.path.join(dataset_dir, "metadata_textual_based_labels.csv")
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
	for i in range(min(50, len(df))):
		print(f"\nSample {i+1}:")
		print("Text:", df.iloc[i]['enriched_document_description'])
		print("Top Labels:", df.iloc[i]['textual_based_labels'])
		print("Scores:", df.iloc[i]['textual_based_scores'])
		print(df.iloc[i]['img_url'])

	return per_image_labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Textual-label annotation for Historical Archives Dataset")
	parser.add_argument("--csv_file", '-csv', type=str, required=True, help="Path to the metadata CSV file")
	parser.add_argument("--num_workers", '-nw', type=int, default=4, help="Number of workers for parallel processing")
	parser.add_argument("--text_batch_size", '-tbs', type=int, default=128, help="Batch size for textual processing")
	parser.add_argument("--sentence_model_name", '-smn', type=str, default="intfloat/multilingual-e5-large", choices=["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "jinaai/jina-embeddings-v3", "intfloat/multilingual-e5-large"], help="Sentence-transformer model name")
	parser.add_argument("--device", '-d', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run models on ('cuda:0' or 'cpu')")
	parser.add_argument("--topk", '-k', type=int, default=10, help="Number of top labels to return")

	args, unknown = parser.parse_known_args()
	args.device = torch.device(args.device)
	print_args_table(args=args, parser=parser)
	
	print(f"Loading model: {args.sentence_model_name}...")
	sent_model = SentenceTransformer(
		model_name_or_path=args.sentence_model_name, 
		device=args.device, 
		trust_remote_code=True,
	)
	get_textual_based_annotation(
		csv_file=args.csv_file, 
		sent_model=sent_model, 
		top_k=args.topk, 
		batch_size=args.text_batch_size,
	)
	
