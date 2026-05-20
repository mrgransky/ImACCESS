import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

print(f"sys.path: {sys.path}")

from utils import *
from clustering import *

def cluster_and_save_priors(
	receipts_jsonl: str,
	model_id: str,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	verbose: bool = True
):

	# Parse receipts JSONL to reconstruct all_sample_labels list-of-lists
	all_sample_labels = []
	print(f"[STAGE 3] Loading receipts from {receipts_jsonl}...")
	with open(receipts_jsonl, 'r') as f:
		for line in f:
			receipt = json.loads(line)
			# Fetch the raw VLM concepts we backed up under 'cot_raw_vlm' in Stage 2
			vlm_data = receipt.get("cot_raw_vlm", {})
			if isinstance(vlm_data, dict):
				text_c = vlm_data.get("text_concepts", [])
				vis_c = vlm_data.get("visual_concepts", [])
				fused_c = vlm_data.get("fused_concepts", [])
				
				# Combine all raw concepts from this sample
				sample_pool = text_c + vis_c + fused_c
				all_sample_labels.append(sample_pool)
	
	DATASET_DIRECTORY = os.path.dirname(receipts_jsonl)
	output_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(output_dir, exist_ok=True)
	
	# Dedup and Flatten unique concepts
	unique_labels = sorted(set(lbl for sample in all_sample_labels for lbl in sample if lbl))
	print(f"\n[CLUSTERING] Extracted {len(unique_labels):,} unique concepts across the dataset.")
	# Calculate raw frequencies (Global Reusability Prior)
	label_freq_dict = Counter(lbl for sample in all_sample_labels for lbl in sample if lbl)
	
	freqs_path = os.path.join(output_dir, os.path.basename(receipts_jsonl.replace(".jsonl", "_global_label_frequency.json")))
	with open(freqs_path, 'w') as f:
		json.dump(label_freq_dict, f, indent=2)
	
	# Load Sentence Transformer Model
	print(f"\n[CLUSTERING] Initializing Embedding Model {model_id}...")
	model = SentenceTransformer(
		model_id,
		device=device,
		trust_remote_code=True,
		cache_folder=cache_directory[os.getenv('USER')],
		token=os.getenv("HUGGINGFACE_TOKEN")
	).to(device)

	# Encode raw unique concepts
	print(f"[CLUSTERING] Generating embeddings for {len(unique_labels):,} labels...")
	X = model.encode(
		unique_labels,
		batch_size=1024,
		show_progress_bar=verbose,
		convert_to_numpy=True,
		normalize_embeddings=True,
		precision='float32'
	)

	# Linkage Matrix (Hierarchical)
	print(f"\n[CLUSTERING] Building Linkage Matrix (Ward Linkage, Euclidean Distance)...")
	t0 = time.time()
	if use_fastcluster:
		Z = fastcluster.linkage(X, method='ward', metric='euclidean')
	else: 
		Z = linkage(X, method='ward', metric='euclidean')
	print(f"[CLUSTERING] Linkage computation complete. Time: {time.time()-t0:.1f}s")

	# Adaptive optimal cluster search
	cluster_labels, stats = get_optimal_num_clusters(
		X=X,
		linkage_matrix=Z,
		target_intra_similarity=0.69,
		min_consolidation=3.8,
		max_consolidation=5.0,
		target_singleton_ratio=0.015,
		quality_vs_consolidation_weight=0.5,
		merge_singletons=True,
		verbose=verbose
	)
	df = pd.DataFrame({'label': unique_labels, 'cluster': cluster_labels})
	# Execute 5-Signal Canonical Selection
	print(f"\n[CLUSTERING] Executing 5-Signal Canonical Selection...")
	(
		cluster_canonicals,
		virtual_used_count,
		freq_changed_count,
		total_sim_loss,
		total_freq_gain,
		questionable_examples,
	) = assign_canonical_labels(
		df=df, 
		X=X, 
		model=model, 
		original_label_counts=label_freq_dict, 
		verbose=verbose,
	)
	print(f"[CLUSTERING] Selected canonical terms. Virtual hypernyms synthesized: {virtual_used_count}")
	
	# Map raw concepts to canonical labels
	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])
	
	# Inject Virtual Hypernyms as genuine rows into df and embeddings array
	virtual_rows, virtual_embs = [], []
	for cid, meta in cluster_canonicals.items():
		if not meta['virtual']:
			continue
		vh = meta['canonical']
		vh_emb = model.encode(
			[vh], 
			batch_size=1, 
			convert_to_numpy=True, 
			normalize_embeddings=True, 
			precision='float32'
		)[0]
		
		virtual_rows.append({'label': vh, 'cluster': cid, 'canonical': vh})
		virtual_embs.append(vh_emb)
	
	if virtual_rows:
		df = pd.concat([df, pd.DataFrame(virtual_rows)], ignore_index=True)
		X = np.vstack([X, np.array(virtual_embs)])
		print(f"[CLUSTERING] Injected {len(virtual_rows)} virtual hypernyms into the embedding space.")
	
	# Drop low-cohesion and poor representation clusters (Audit Step)
	df_clean, X_clean, removed_labels = remove_problematic_cluster_labels(
		df=df, 
		embeddings=X, 
		low_cohesion_threshold=0.50, 
		poor_canonical_threshold=0.60, 
		verbose=verbose,
	)
	
	# Re-evaluate final map
	unique_labels_clean = df_clean['label'].values
	cluster_labels_clean = df_clean['cluster'].values
	canonical_map = df_clean.groupby('cluster')['canonical'].first().to_dict()
	analyze_cluster_quality(
		embeddings=X_clean,
		labels=unique_labels_clean,
		cluster_assignments=cluster_labels_clean,
		canonical_labels=canonical_map,
		original_label_counts=label_freq_dict,
		verbose=verbose
	)
	
	# 1. Output canonical_map.json
	final_canonical_dict = df_clean.set_index('label')['canonical'].to_dict()
	map_path = os.path.join(output_dir, os.path.basename(receipts_jsonl.replace(".jsonl", "_canonical_map.json")))
	with open(map_path, 'w') as f:
		json.dump(final_canonical_dict, f, indent=2)

	# 2. Output emb_cache.pt
	# Creates a dictionary mapping raw_concept string -> L2 normalized numpy embedding array
	emb_cache = {lbl: emb for lbl, emb in zip(df_clean['label'].tolist(), X_clean)}
	emb_path = os.path.join(output_dir, os.path.basename(receipts_jsonl.replace(".jsonl", "_emb_cache.pt")))
	
	# Save using standard PyTorch serializer (supports dict(str -> np.ndarray))
	torch.save(emb_cache, emb_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Stage 3: Global Aggregation & Ontology Builder")
	parser.add_argument("--receipts_jsonl", "-r", type=str, required=True, help="Path to Stage 2 JSONL audit receipts")
	parser.add_argument("--model_id", "-m", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for canonical analysis")
	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose output")

	args = parser.parse_args()
	set_seeds(seed=42)


	# Run the main aggregation controller
	cluster_and_save_priors(
		receipts_jsonl=args.receipts_jsonl,
		model_id=args.model_id,
		verbose=args.verbose
	)