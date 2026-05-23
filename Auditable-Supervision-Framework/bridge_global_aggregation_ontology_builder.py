# import os
# import sys

# HOME, USER = os.getenv('HOME'), os.getenv('USER')
# IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

# CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
# sys.path.insert(0, CLIP_DIR)
# MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
# sys.path.insert(0, MISC_DIR)

# print(f"sys.path: {sys.path}")

# from utils import *
# from clustering import *

# def cluster_and_save_priors(
# 	input_jsonl: str,
# 	model_id: str,
# 	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
# 	verbose: bool = True
# ):
# 	# Parse receipts JSONL to reconstruct all_sample_labels list-of-lists
# 	all_sample_labels = []
# 	print(f"\n[STAGE 3] Loading receipts from {input_jsonl}")
# 	with open(input_jsonl, 'r') as f:
# 		for line in f:
# 			receipt = json.loads(line)
# 			# Fetch the raw VLM concepts we backed up under 'vlm_cot_raw' in Stage 2
# 			vlm_data = receipt.get("vlm_cot_raw", {})
# 			if isinstance(vlm_data, dict):
# 				text_c = vlm_data.get("text_concepts", [])
# 				vis_c = vlm_data.get("visual_concepts", [])
# 				fused_c = vlm_data.get("fused_concepts", [])
				
# 				# Combine all raw concepts from this sample
# 				sample_pool = text_c + vis_c + fused_c
# 				all_sample_labels.append(sample_pool)
	
# 	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
# 	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
# 	os.makedirs(outputs_dir, exist_ok=True)
	
# 	# Dedup and Flatten unique concepts
# 	unique_labels = sorted(set(lbl for sample in all_sample_labels for lbl in sample if lbl))
# 	print(f"\n[CLUSTERING] Extracted {len(unique_labels):,} unique concepts across the dataset.")
# 	# Calculate raw frequencies (Global Reusability Prior)
# 	label_freq_dict = Counter(lbl for sample in all_sample_labels for lbl in sample if lbl)
	
# 	freqs_path = os.path.join(outputs_dir, os.path.basename(input_jsonl.replace(".jsonl", "_global_label_frequency.json")))
# 	with open(freqs_path, 'w') as f:
# 		json.dump(label_freq_dict, f, indent=2)
	
# 	# Load Sentence Transformer Model
# 	print(f"\n[CLUSTERING] Initializing Embedding Model {model_id}...")
# 	model = SentenceTransformer(
# 		model_id,
# 		device=device,
# 		trust_remote_code=True,
# 		cache_folder=cache_directory[os.getenv('USER')],
# 		token=os.getenv("HUGGINGFACE_TOKEN")
# 	).to(device)

# 	# Encode raw unique concepts
# 	print(f"[CLUSTERING] Generating embeddings for {len(unique_labels):,} labels...")
# 	X = model.encode(
# 		unique_labels,
# 		batch_size=1024,
# 		show_progress_bar=verbose,
# 		convert_to_numpy=True,
# 		normalize_embeddings=True,
# 		precision='float32'
# 	)

# 	# Linkage Matrix (Hierarchical)
# 	print(f"\n[CLUSTERING] Building Linkage Matrix: {type(X)} {X.shape} (Ward Linkage, Euclidean Distance)...")
# 	t0 = time.time()
# 	if use_fastcluster:
# 		Z = fastcluster.linkage(X, method='ward', metric='euclidean')
# 	else: 
# 		Z = linkage(X, method='ward', metric='euclidean')
# 	print(f"[CLUSTERING] Linkage computation complete. Time: {time.time()-t0:.1f}s")

# 	# Adaptive optimal cluster search
# 	cluster_labels, stats = get_optimal_num_clusters(
# 		X=X,
# 		linkage_matrix=Z,
# 		target_intra_similarity=0.69,
# 		min_consolidation=3.8,
# 		max_consolidation=5.0,
# 		target_singleton_ratio=0.015,
# 		quality_vs_consolidation_weight=0.5,
# 		merge_singletons=True,
# 		verbose=verbose
# 	)
# 	df = pd.DataFrame({'label': unique_labels, 'cluster': cluster_labels})

# 	# Execute 5-Signal Canonical Selection
# 	print(f"\n[CLUSTERING] Executing 5-Signal Canonical Selection...")
# 	(
# 		cluster_canonicals,
# 		virtual_used_count,
# 		freq_changed_count,
# 		total_sim_loss,
# 		total_freq_gain,
# 		questionable_examples,
# 	) = assign_canonical_labels(
# 		df=df, 
# 		X=X, 
# 		model=model, 
# 		original_label_counts=label_freq_dict, 
# 		verbose=verbose,
# 	)
# 	print(f"[CLUSTERING] Selected canonical terms. Virtual hypernyms synthesized: {virtual_used_count}")
	
# 	# Map raw concepts to canonical labels
# 	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])
	
# 	# Inject Virtual Hypernyms as genuine rows into df and embeddings array
# 	virtual_rows, virtual_embs = [], []
# 	for cid, meta in cluster_canonicals.items():
# 		if not meta['virtual']:
# 			continue
# 		vh = meta['canonical']
# 		vh_emb = model.encode(
# 			[vh], 
# 			batch_size=1, 
# 			convert_to_numpy=True, 
# 			normalize_embeddings=True, 
# 			precision='float32'
# 		)[0]
		
# 		virtual_rows.append({'label': vh, 'cluster': cid, 'canonical': vh})
# 		virtual_embs.append(vh_emb)
	
# 	if virtual_rows:
# 		df = pd.concat([df, pd.DataFrame(virtual_rows)], ignore_index=True)
# 		X = np.vstack([X, np.array(virtual_embs)])
# 		print(f"[CLUSTERING] Injected {len(virtual_rows)} virtual hypernyms into the embedding space.")
	
# 	# Drop low-cohesion and poor representation clusters (Audit Step)
# 	df_clean, X_clean, removed_labels = remove_problematic_cluster_labels(
# 		df=df, 
# 		embeddings=X, 
# 		low_cohesion_threshold=0.50, 
# 		poor_canonical_threshold=0.60, 
# 		verbose=verbose,
# 	)
	
# 	# Re-evaluate final map
# 	unique_labels_clean = df_clean['label'].values
# 	cluster_labels_clean = df_clean['cluster'].values
# 	canonical_map = df_clean.groupby('cluster')['canonical'].first().to_dict()
# 	analyze_cluster_quality(
# 		embeddings=X_clean,
# 		labels=unique_labels_clean,
# 		cluster_assignments=cluster_labels_clean,
# 		canonical_labels=canonical_map,
# 		original_label_counts=label_freq_dict,
# 		verbose=verbose
# 	)

# 	# EXTRACT EMERGENT TARGET VOCABULARY (V)
# 	# The sorted unique canonical labels that survived the clean-up audit is V!
# 	target_vocab = sorted(list(df_clean['canonical'].unique()))
# 	print(f"\n[VOCABULARY] Emergent Target Vocabulary size: {len(target_vocab)} unique canonical classes.")
# 	print(f"  ├─ Sample classes: {target_vocab[:10]}...")

# 	# Generate embeddings for the discovered canonical classes
# 	print(f"[VOCABULARY] Pre-computing embeddings for the {len(target_vocab)} target classes...")
# 	target_embeddings = model.encode(
# 		target_vocab,
# 		batch_size=1024,
# 		convert_to_numpy=True,
# 		normalize_embeddings=True,
# 		precision='float32'
# 	)

# 	# 1. Output canonical_map.json (raw VLM concept -> clean discovered canonical)
# 	final_canonical_dict = df_clean.set_index('label')['canonical'].to_dict()
# 	map_path = os.path.join(outputs_dir, os.path.basename(input_jsonl.replace(".jsonl", "_canonical_map.json")))
# 	with open(map_path, 'w') as f:
# 		json.dump(final_canonical_dict, f, indent=2)
# 	print(f"[BRIDGE] Saved final canonical map to: {map_path}")

# 	# 2. Output target_vocabulary.json (defines the multi-hot vector dimensions for Stage 4/5)
# 	vocab_path = os.path.join(outputs_dir, os.path.basename(input_jsonl.replace(".jsonl", "_target_vocabulary.json")))
# 	with open(vocab_path, 'w') as f:
# 		json.dump(target_vocab, f, indent=2)
# 	print(f"[BRIDGE] Saved target vocabulary (V) to: {vocab_path}")

# 	# 3. Output emb_cache.pt
# 	# We map raw VLM concepts -> embedding, and target_vocabulary -> embedding 
# 	# so Stage 4 can resolve unseen concepts via fast vector lookup
# 	emb_cache = {lbl: emb for lbl, emb in zip(df_clean['label'].tolist(), X_clean)}
	
# 	# Inject the target canonical classes and their embeddings
# 	for target_class, target_emb in zip(target_vocab, target_embeddings):
# 		emb_cache[target_class] = target_emb

# 	emb_path = os.path.join(outputs_dir, os.path.basename(input_jsonl.replace(".jsonl", "_emb_cache.pt")))
# 	torch.save(emb_cache, emb_path)
# 	print(f"[CACHE] Saved emb_cache.pt with {len(emb_cache):,} total mapped embeddings to: {emb_path}")

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser(description="Global Aggregation & Ontology Builder")
# 	parser.add_argument("--jsonl_file", "-jsonl", type=str, required=True, help="Path to Stage 2 modality conflic audit JSONL file")
# 	parser.add_argument("--model_id", "-m", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for canonical analysis")
# 	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose output")

# 	args = parser.parse_args()
# 	set_seeds(seed=42)

# 	if "_modality_conflict_audit.jsonl" not in args.jsonl_file:
# 		raise ValueError(f"Input JSONL file must be a Stage 2 modality conflict audit file. Got: {args.jsonl_file}")

# 	# Run the main aggregation controller
# 	cluster_and_save_priors(
# 		input_jsonl=args.jsonl_file,
# 		model_id=args.model_id,
# 		verbose=args.verbose
# 	)

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
	input_jsonl: str,
	model_id: str,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	verbose: bool = True,
):
	"""
	The Bridge: Corpus-Level Ontology Discovery.

	Reads Stage 2 Evidence Receipts, pools raw VLM concepts per sample
	(regime-gated), runs hierarchical clustering via clustering.py, and
	emits three artefacts consumed by Stage 3 and Stage 4:

	  1. <stem>_canonical_map.json        — raw VLM concept → canonical label
	  2. <stem>_target_vocabulary.json    — the emergent vocabulary V (with metadata)
	  3. <stem>_emb_cache.pt             — {label: embedding} for fast lookup

	Intermediate checkpoints (embeddings, linkage matrix) are saved after
	each expensive step so the function is crash-safe and resumable.

	Fixes applied vs. original code
	--------------------------------
	FIX-1  Regime-gated concept pooling:
	         MISSING_MODALITY / INVALID_JSON samples are skipped entirely.
	         HARD_CONFLICT samples contribute text_concepts + visual_concepts
	         only — fused_concepts is excluded because it is either empty or
	         a hallucinated blend of two disjoint modalities.

	FIX-2  Print labels renamed from [STAGE 3] → [BRIDGE] to avoid naming
	         collision with the Micro-CGD Audit (Stage 3).

	FIX-3  emb_cache built in a single encoding pass:
	         target canonical embeddings that are already present as raw VLM
	         concept rows in X_clean are reused directly; only genuinely new
	         canonical strings (virtual hypernyms not in the raw concept set)
	         are encoded in a second pass.  This eliminates the double-encoding
	         overwrite that caused floating-point non-determinism.

	FIX-4  label_freq_dict converted from Counter → plain dict before being
	         passed to assign_canonical_labels (called inside cluster_and_save_priors
	         via clustering.py).  Counter's default-zero behaviour silently
	         disables the freq-gain safety check for unseen labels.

	FIX-5  Crash-safe checkpointing:
	         unique_label_embeddings.npy, unique_labels.npy, and
	         linkage_matrix.npy are saved immediately after their respective
	         expensive computations.  On restart the function detects these
	         files and skips recomputation.

	FIX-6  target_vocabulary.json stores a metadata wrapper dict instead of
	         a bare list, making Stage 4 loading unambiguous and self-documenting.
	"""

	print(f"\n{'='*80}")
	print(f"[BRIDGE] Corpus-Level Ontology Discovery")
	print(f"{'='*80}")
	print(f"  ├─ Input  : {input_jsonl}")
	print(f"  ├─ Model  : {model_id}")
	print(f"  └─ Device : {device}")

	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	stem = os.path.basename(input_jsonl).replace(".jsonl", "")

	# ── Checkpoint paths ──────────────────────────────────────────────────────
	# Saved immediately after each expensive step so a crash never loses work.
	ckpt_emb_path    = os.path.join(outputs_dir, f"{stem}_unique_label_embeddings.npy")
	ckpt_labels_path = os.path.join(outputs_dir, f"{stem}_unique_labels.npy")
	ckpt_Z_path      = os.path.join(outputs_dir, f"{stem}_linkage_matrix.npy")

	# ── Output paths ──────────────────────────────────────────────────────────
	freqs_path = os.path.join(outputs_dir, f"{stem}_global_label_frequency.json")
	map_path   = os.path.join(outputs_dir, f"{stem}_canonical_map.json")
	vocab_path = os.path.join(outputs_dir, f"{stem}_target_vocabulary.json")
	emb_path   = os.path.join(outputs_dir, f"{stem}_emb_cache.pt")

	# =========================================================================
	# STEP 1: REGIME-GATED CONCEPT POOLING
	# FIX-1: Gate fused_concepts by regime. HARD_CONFLICT fused concepts are
	# excluded because they are either empty (VLM detected conflict) or a
	# hallucinated blend of two disjoint modalities — including them would
	# seed the clustering with incoherent concepts and inflate the vocabulary.
	# FIX-2: All print labels use [BRIDGE] to avoid collision with [STAGE 3].
	# =========================================================================
	all_sample_labels: List[List[str]] = []
	regime_counts: dict = {}
	skipped_count = 0

	print(f"\n[BRIDGE] Loading and pooling concepts from: {input_jsonl}")
	with open(input_jsonl, 'r', encoding='utf-8') as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				receipt = json.loads(line)
			except json.JSONDecodeError as e:
				print(f"[BRIDGE][WARN] Malformed line {line_no}: {e}")
				continue

			regime   = receipt.get("regime", "UNKNOWN")
			vlm_data = receipt.get("vlm_cot_raw", {})

			# Track regime distribution for diagnostics
			regime_counts[regime] = regime_counts.get(regime, 0) + 1

			# FIX-1a: Skip samples with no usable modality signal entirely.
			# These contribute nothing to vocabulary induction and would add
			# noise to the global frequency counts.
			if regime in ("MISSING_MODALITY", "INVALID_JSON", "UNKNOWN"):
				skipped_count += 1
				continue

			if not isinstance(vlm_data, dict):
				skipped_count += 1
				continue

			text_c  = vlm_data.get("text_concepts",   []) or []
			vis_c   = vlm_data.get("visual_concepts",  []) or []
			fused_c = vlm_data.get("fused_concepts",   []) or []

			# FIX-1b: For HARD_CONFLICT, fused_concepts is unreliable.
			# The two modalities are structurally disjoint (orphan_ratio >=
			# tau_orphan), so any fused output is either empty or a blend
			# that does not correspond to a real semantic concept.
			# We still include text_c and vis_c because they are individually
			# coherent and contribute to the vocabulary.
			if regime == "HARD_CONFLICT":
				sample_pool = text_c + vis_c
			else:
				# AGREEMENT and SOFT_CONFLICT: fused_concepts is trustworthy.
				sample_pool = text_c + vis_c + fused_c

			# Drop empty strings and None values
			sample_pool = [lbl for lbl in sample_pool if lbl and isinstance(lbl, str)]

			if sample_pool:
				all_sample_labels.append(sample_pool)

	total_loaded = sum(regime_counts.values())
	print(f"[BRIDGE] Loaded {total_loaded:,} receipts | Skipped: {skipped_count:,}")
	print(f"[BRIDGE] Regime distribution:")
	for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
		print(f"  ├─ {r:<25} {cnt:>8,}  ({cnt/max(total_loaded,1)*100:.1f}%)")
	print(f"[BRIDGE] Samples contributing to vocabulary: {len(all_sample_labels):,}")

	if not all_sample_labels:
		raise ValueError(
			"[BRIDGE] No samples survived regime gating. "
			"Check that the input JSONL contains AGREEMENT / SOFT_CONFLICT / HARD_CONFLICT receipts."
		)

	# =========================================================================
	# STEP 2: GLOBAL FREQUENCY COUNTS (Reusability Prior)
	# FIX-4: Convert Counter → plain dict before saving and before passing to
	# any clustering.py function. Counter's default-zero behaviour silently
	# disables the freq-gain safety check inside assign_canonical_labels for
	# labels not seen in the corpus (Counter returns 0 for missing keys, so
	# the .get(lbl, 1) fallback in the safety ratio never triggers, causing
	# the ratio to always be 0/max(0,1)=0 and the safety check to always
	# revert to pure centroid similarity).
	# =========================================================================
	print(f"\n[BRIDGE] Computing global label frequencies...")
	label_freq_dict: dict = dict(
		Counter(lbl for sample in all_sample_labels for lbl in sample if lbl)
	)

	with open(freqs_path, 'w', encoding='utf-8') as f:
		json.dump(label_freq_dict, f, indent=2, ensure_ascii=False)
	print(f"[BRIDGE] Saved global frequencies ({len(label_freq_dict):,} labels) → {freqs_path}")

	# Dedup and sort unique concepts (after regime gating, so no HARD_CONFLICT
	# fused concepts are present)
	unique_labels: List[str] = sorted(set(
		lbl for sample in all_sample_labels for lbl in sample if lbl
	))
	print(f"[BRIDGE] Unique concepts for vocabulary induction: {len(unique_labels):,}")

	# =========================================================================
	# STEP 3: EMBEDDING (with crash-safe checkpoint)
	# FIX-5a: Save embeddings and label list immediately after encoding.
	# On restart, load from checkpoint and skip the ~10-30 min encoding step.
	# =========================================================================
	resume_emb = (
		os.path.exists(ckpt_emb_path) and
		os.path.exists(ckpt_labels_path)
	)

	if resume_emb:
		print(f"\n[BRIDGE] Resuming: loading cached embeddings from checkpoint.")
		X             = np.load(ckpt_emb_path)
		unique_labels = np.load(ckpt_labels_path, allow_pickle=True).tolist()
		print(f"[BRIDGE] Loaded embeddings {X.shape} for {len(unique_labels):,} labels.")
	else:
		print(f"\n[BRIDGE] Initialising embedding model: {model_id}")
		model = SentenceTransformer(
			model_id,
			device=device,
			trust_remote_code=True,
			cache_folder=cache_directory[os.getenv('USER')],
			token=os.getenv("HUGGINGFACE_TOKEN"),
		).to(device)

		print(f"[BRIDGE] Encoding {len(unique_labels):,} unique labels...")
		X = model.encode(
			unique_labels,
			batch_size=1024,
			show_progress_bar=verbose,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)
		print(f"[BRIDGE] Embeddings: {X.shape} {X.dtype}")

		# FIX-5a: Checkpoint immediately
		np.save(ckpt_emb_path,    X)
		np.save(ckpt_labels_path, np.array(unique_labels, dtype=object))
		print(f"[BRIDGE] Checkpointed embeddings → {ckpt_emb_path}")

	# =========================================================================
	# STEP 4: LINKAGE MATRIX (with crash-safe checkpoint)
	# FIX-5b: Save linkage matrix immediately after computation.
	# Ward linkage on 50K+ concepts can take 20-30 min; losing it to a crash
	# is unacceptable.
	# =========================================================================
	if os.path.exists(ckpt_Z_path):
		print(f"\n[BRIDGE] Resuming: loading cached linkage matrix from checkpoint.")
		Z = np.load(ckpt_Z_path)
		print(f"[BRIDGE] Loaded linkage matrix {Z.shape}.")
	else:
		print(f"\n[BRIDGE] Building linkage matrix: {X.shape} (Ward, Euclidean)...")
		t0 = time.time()
		if use_fastcluster:
			Z = fastcluster.linkage(X, method='ward', metric='euclidean')
		else:
			Z = linkage(X, method='ward', metric='euclidean')
		print(f"[BRIDGE] Linkage complete in {time.time()-t0:.1f}s. Z: {Z.shape}")

		# FIX-5b: Checkpoint immediately
		np.save(ckpt_Z_path, Z)
		print(f"[BRIDGE] Checkpointed linkage matrix → {ckpt_Z_path}")

	# =========================================================================
	# STEP 5: ADAPTIVE OPTIMAL CLUSTER SEARCH
	# Delegates entirely to clustering.py — no changes needed here.
	# =========================================================================
	print(f"\n[BRIDGE] Running adaptive optimal cluster search...")
	cluster_labels_arr, stats = get_optimal_num_clusters(
		X=X,
		linkage_matrix=Z,
		target_intra_similarity=0.69,
		min_consolidation=3.8,
		max_consolidation=5.0,
		target_singleton_ratio=0.015,
		quality_vs_consolidation_weight=0.5,
		merge_singletons=True,
		verbose=verbose,
	)
	df = pd.DataFrame({'label': unique_labels, 'cluster': cluster_labels_arr})
	print(f"[BRIDGE] Optimal k={stats['n_clusters']:,} clusters "
	      f"(consolidation {stats['consolidation_ratio']:.2f}x, "
	      f"intra_sim={stats['mean_intra_similarity']:.4f})")

	# =========================================================================
	# STEP 6: 5-SIGNAL CANONICAL SELECTION
	# Requires the embedding model — reload if we resumed from checkpoint
	# (model was not loaded in the resume branch of STEP 3).
	# =========================================================================
	if resume_emb:
		# Model was not loaded in the resume branch; load it now for canonical
		# selection (virtual hypernym encoding) and target vocab embedding.
		print(f"\n[BRIDGE] Loading embedding model for canonical selection (resume path)...")
		model = SentenceTransformer(
			model_id,
			device=device,
			trust_remote_code=True,
			cache_folder=cache_directory[os.getenv('USER')],
			token=os.getenv("HUGGINGFACE_TOKEN"),
		).to(device)

	print(f"\n[BRIDGE] Executing 5-Signal Canonical Selection...")
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
		original_label_counts=label_freq_dict,   # plain dict (FIX-4)
		verbose=verbose,
	)
	print(f"[BRIDGE] Canonical selection complete. "
	      f"Virtual hypernyms synthesised: {virtual_used_count}")

	# Map raw concepts to canonical labels
	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])

	# =========================================================================
	# STEP 7: INJECT VIRTUAL HYPERNYMS AS GENUINE ROWS
	# Virtual hypernyms must be real rows in df+X before remove_problematic_
	# cluster_labels runs, otherwise that function flags their clusters as
	# "canonical not found" and drops them.
	# =========================================================================
	virtual_rows: List[dict] = []
	virtual_embs: List[np.ndarray] = []

	for cid, meta in cluster_canonicals.items():
		if not meta['virtual']:
			continue
		vh = meta['canonical']
		vh_emb = model.encode(
			[vh],
			batch_size=1,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)[0]
		virtual_rows.append({'label': vh, 'cluster': cid, 'canonical': vh})
		virtual_embs.append(vh_emb)

	if virtual_rows:
		df = pd.concat([df, pd.DataFrame(virtual_rows)], ignore_index=True)
		X  = np.vstack([X, np.array(virtual_embs)])
		print(f"[BRIDGE] Injected {len(virtual_rows)} virtual hypernym(s) into df+X.")

	# =========================================================================
	# STEP 8: AUDIT — DROP LOW-COHESION AND POOR-CANONICAL CLUSTERS
	# =========================================================================
	df_clean, X_clean, removed_labels = remove_problematic_cluster_labels(
		df=df,
		embeddings=X,
		low_cohesion_threshold=0.50,
		poor_canonical_threshold=0.60,
		verbose=verbose,
	)
	print(f"[BRIDGE] Audit complete. "
	      f"Removed {len(removed_labels):,} labels from problematic clusters.")

	# =========================================================================
	# STEP 9: RE-EVALUATE FINAL CLUSTER QUALITY
	# =========================================================================
	unique_labels_clean   = df_clean['label'].values
	cluster_labels_clean  = df_clean['cluster'].values
	canonical_map_int     = df_clean.groupby('cluster')['canonical'].first().to_dict()

	analyze_cluster_quality(
		embeddings=X_clean,
		labels=unique_labels_clean,
		cluster_assignments=cluster_labels_clean,
		canonical_labels=canonical_map_int,
		original_label_counts=label_freq_dict,
		verbose=verbose,
	)

	# =========================================================================
	# STEP 10: EXTRACT EMERGENT TARGET VOCABULARY V
	# =========================================================================
	target_vocab: List[str] = sorted(df_clean['canonical'].unique().tolist())
	print(f"\n[BRIDGE] Emergent Target Vocabulary |V| = {len(target_vocab):,} canonical classes.")
	print(f"  ├─ Sample: {target_vocab[:10]}...")

	# =========================================================================
	# STEP 11: PRE-COMPUTE TARGET VOCABULARY EMBEDDINGS
	# Only encode canonical strings that are NOT already present as raw VLM
	# concept rows in X_clean.
	#
	# FIX-3: Build emb_cache in a single pass to avoid double-encoding.
	# If a canonical string (e.g. "soldier") also appears as a raw VLM concept,
	# its embedding is already in X_clean from the original encoding run.
	# Re-encoding it in a separate model.encode() call introduces floating-point
	# non-determinism (different batch context → slightly different output),
	# making emb_cache a mix of two encoding runs and breaking reproducibility.
	#
	# Strategy:
	#   a) Build emb_cache from all (raw concept, embedding) pairs in X_clean.
	#   b) Identify target canonicals NOT already in emb_cache (new strings only).
	#   c) Encode only those new strings in a single additional pass.
	#   d) Merge into emb_cache.
	# =========================================================================
	print(f"\n[BRIDGE] Building emb_cache (single-pass, FIX-3)...")

	# (a) Seed cache from all raw concept embeddings in X_clean
	emb_cache: dict = {
		lbl: emb
		for lbl, emb in zip(df_clean['label'].tolist(), X_clean)
	}

	# (b) Find target canonicals not yet in the cache
	#     These are exclusively virtual hypernyms whose string was synthesised
	#     and does not appear as a raw VLM concept in any sample.
	new_canonicals: List[str] = [
		tc for tc in target_vocab if tc not in emb_cache
	]

	# (c) Encode only the genuinely new strings
	if new_canonicals:
		print(f"[BRIDGE] Encoding {len(new_canonicals):,} new canonical strings "
		      f"(virtual hypernyms not in raw concept set)...")
		new_embs = model.encode(
			new_canonicals,
			batch_size=1024,
			show_progress_bar=verbose,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)
		# (d) Merge into cache
		for lbl, emb in zip(new_canonicals, new_embs):
			emb_cache[lbl] = emb
	else:
		print(f"[BRIDGE] All {len(target_vocab):,} target canonicals already in cache "
		      f"(no virtual hypernyms needed separate encoding).")

	print(f"[BRIDGE] emb_cache size: {len(emb_cache):,} entries "
	      f"(raw concepts + target canonicals, single encoding run).")

	# =========================================================================
	# STEP 12: SAVE OUTPUTS
	# =========================================================================

	# 1. canonical_map.json — raw VLM concept → canonical label
	#    Used by Stage 3 (Micro-CGD Audit) and Stage 4 (Consolidation).
	final_canonical_dict: dict = df_clean.set_index('label')['canonical'].to_dict()
	with open(map_path, 'w', encoding='utf-8') as f:
		json.dump(final_canonical_dict, f, indent=2, ensure_ascii=False)
	print(f"\n[BRIDGE] Saved canonical_map ({len(final_canonical_dict):,} entries) → {map_path}")

	# 2. target_vocabulary.json — defines the multi-hot vector dimensions for Stage 4/5.
	#    FIX-6: Wrap in a metadata dict so Stage 4 can load unambiguously and
	#    the file is self-documenting (size, source, timestamp).
	vocab_payload = {
		"vocabulary":  target_vocab,
		"size":        len(target_vocab),
		"source_jsonl": os.path.basename(input_jsonl),
		"model_id":    model_id,
		"note": (
			"Load via: vocab = json.load(f)['vocabulary']. "
			"Index i in this list corresponds to dimension i of the multi-hot vector."
		),
	}
	with open(vocab_path, 'w', encoding='utf-8') as f:
		json.dump(vocab_payload, f, indent=2, ensure_ascii=False)
	print(f"[BRIDGE] Saved target_vocabulary V ({len(target_vocab):,} classes) → {vocab_path}")

	# 3. emb_cache.pt — {label: np.ndarray} for fast vector lookup in Stage 4.
	torch.save(emb_cache, emb_path)
	print(f"[BRIDGE] Saved emb_cache.pt ({len(emb_cache):,} entries) → {emb_path}")

	print(f"\n[BRIDGE] Done. All outputs written to: {outputs_dir}")
	print(f"{'='*80}\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Bridge: Global Aggregation & Ontology Discovery")
	parser.add_argument(
		"--jsonl_file", "-jsonl", type=str, required=True,
		help="Path to Stage 2 modality conflict audit JSONL file",
	)
	parser.add_argument(
		"--model_id", "-m", type=str, default="all-MiniLM-L6-v2",
		help="SentenceTransformer model for canonical analysis",
	)
	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose output")

	args = parser.parse_args()
	set_seeds(seed=42)

	if "_modality_conflict_audit.jsonl" not in args.jsonl_file:
		raise ValueError(
			f"Input JSONL must be a Stage 2 modality conflict audit file. Got: {args.jsonl_file}"
		)

	cluster_and_save_priors(
		input_jsonl=args.jsonl_file,
		model_id=args.model_id,
		verbose=args.verbose,
	)