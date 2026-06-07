import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

# local:
# nohup python -u bridge_global_aggregation_ontology_builder.py -jsonl /home/farid/datasets/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl -v > logs/global_aggregation.log 2>&1 &

from utils import *
from clustering import *
from nlp_utils import _post_process_

def cluster_and_save_priors(
	input_jsonl: str,
	model_id: str,
	batch_size: int,
	column: str, 
	device: str,
	verbose: bool,
):
	"""
	Bridge: Corpus-Level Ontology Discovery.

	Reads Stage 2 Evidence Receipts, pools raw VLM concepts per sample
	(regime-gated), runs hierarchical clustering via clustering.py, and
	emits three artefacts consumed by Stage 3 and Stage 4:

	  1. <stem>_canonical_map.json        — raw VLM concept → canonical label
	  2. <stem>_target_vocabulary.json    — the emergent vocabulary V (with metadata)
	  3. <stem>_emb_cache.pt             — {label: embedding} for fast lookup

	Intermediate checkpoints (embeddings, linkage matrix) are saved after
	each expensive step so the function is crash-safe and resumable.
	"""
	print(f"\n[BRIDGE] Corpus-Level Ontology Discovery")
	print(f"  ├─ Input  : {input_jsonl}")
	print(f"  ├─ Model  : {model_id}")
	print(f"  ├─ Batch  : {batch_size}")
	print(f"  ├─ Column : {column}")
	print(f"  └─ Device : {device}")

	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	stem = os.path.basename(input_jsonl).replace(".jsonl", "")

	# Checkpoint paths
	# Saved immediately after each expensive step so a crash never loses work.
	ckpt_emb_path    = os.path.join(outputs_dir, f"{stem}_unique_label_embeddings.npy")
	ckpt_labels_path = os.path.join(outputs_dir, f"{stem}_unique_labels.npy")
	ckpt_Z_path      = os.path.join(outputs_dir, f"{stem}_linkage_matrix.npy")

	freqs_path = os.path.join(outputs_dir, f"{stem}_global_label_frequency.json")
	canonical_map_path = os.path.join(outputs_dir, f"{stem}_canonical_map.json")
	vocab_path = os.path.join(outputs_dir, f"{stem}_target_vocabulary.json")
	emb_path = os.path.join(outputs_dir, f"{stem}_emb_cache.pt")

	# =========================================================================
	# STEP 1: REGIME-GATED CONCEPT POOLING
	# Gate fused_concepts by regime. HARD_CONFLICT fused concepts are
	# excluded because they are either empty (VLM detected conflict) or a
	# hallucinated blend of two disjoint modalities — including them would
	# seed the clustering with incoherent concepts and inflate the vocabulary.
	# All print labels use [BRIDGE] to avoid collision with [STAGE 3].
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
			vlm_data = receipt.get(column, {})

			# Track regime distribution for diagnostics
			regime_counts[regime] = regime_counts.get(regime, 0) + 1

			# Skip samples with no usable modality signal entirely.
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

			# For HARD_CONFLICT, fused_concepts is unreliable.
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
			sample_pool = [
				lbl 
				for lbl in sample_pool 
				if lbl and isinstance(lbl, str)
			]

			if sample_pool:
				all_sample_labels.append(sample_pool)

	if verbose:
		total_loaded = sum(regime_counts.values())
		print(f"[BRIDGE] Loaded {total_loaded} receipts | Skipped: {skipped_count}")
		print(f"[BRIDGE] Regime distribution:")
		for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
			print(f"  ├─ {r:<25} {cnt:>8,}  ({cnt/max(total_loaded,1)*100:.1f}%)")
		print(f"[BRIDGE] Samples contributing to vocabulary: {len(all_sample_labels)}")
		for i, sample in enumerate(all_sample_labels):
			print(f"{i:7d} {sample}")
		print("="*185)

	if not all_sample_labels:
		raise ValueError(
			"[BRIDGE] No samples survived regime gating. "
			"Check that the input JSONL contains AGREEMENT / SOFT_CONFLICT / HARD_CONFLICT receipts."
		)

	##################################################################################################
	# Post-process labels
	all_post_processed_sample_labels = _post_process_(
		labels_list=all_sample_labels, 
		verbose=verbose, #False, # not to clutter the logs
	)
	
	# Filter out None values returned by _post_process_
	all_post_processed_sample_labels = [
		sample for sample in all_post_processed_sample_labels 
		if sample is not None
	]
	
	if verbose:
		for i, sample in enumerate(all_post_processed_sample_labels):
			print(f"{i:7d} {sample}")
	all_sample_labels = all_post_processed_sample_labels
	del all_post_processed_sample_labels
	##################################################################################################

	# return
	# STEP 2: GLOBAL FREQUENCY COUNTS (Reusability Prior)
	# Convert Counter → plain dict before saving and before passing to
	# any clustering.py function. Counter's default-zero behaviour silently
	# disables the freq-gain safety check inside assign_canonical_labels for
	# labels not seen in the corpus (Counter returns 0 for missing keys, so
	# the .get(lbl, 1) fallback in the safety ratio never triggers, causing
	# the ratio to always be 0/max(0,1)=0 and the safety check to always
	# revert to pure centroid similarity).
	print(f"\n[BRIDGE] Computing global label frequencies...")
	label_freq_dict: dict = dict(
		Counter(
			lbl 
			for sample in all_sample_labels 
			for lbl in sample 
			if lbl
		)
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
	# Save embeddings and label list immediately after encoding.
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
		print(f"\n[INIT] {model_id}")
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
			batch_size=batch_size,
			show_progress_bar=verbose,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)
		print(f"[BRIDGE] Embeddings: {type(X)} {X.shape} {X.dtype}")

		# Checkpoint immediately
		np.save(ckpt_emb_path,    X)
		np.save(ckpt_labels_path, np.array(unique_labels, dtype=object))
		print(f"[BRIDGE] Checkpointed embeddings → {ckpt_emb_path}")

	# =========================================================================
	# STEP 4: LINKAGE MATRIX (with crash-safe checkpoint)
	# Save linkage matrix immediately after computation.
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

		# Checkpoint immediately
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
		original_label_counts=label_freq_dict,
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
	print(f"[BRIDGE] Removed {len(removed_labels)} labels from problematic clusters.")

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
	print(f"[BRIDGE] Emergent Target Vocabulary |V| = {len(target_vocab)} canonical labels.")

	# =========================================================================
	# STEP 11: PRE-COMPUTE TARGET VOCABULARY EMBEDDINGS
	# Only encode canonical strings that are NOT already present as raw VLM
	# concept rows in X_clean.
	#
	# Build emb_cache in a single pass to avoid double-encoding.
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
	print(f"[BRIDGE] Building emb_cache (single-pass)")

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
		print(
			f"[BRIDGE] Encoding {len(new_canonicals)} new canonical strings "
		  f"(virtual hypernyms not in raw concept set)..."
		)
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
		print(
			f"[BRIDGE] All {len(target_vocab):,} target canonicals already in cache "
		  f"(no virtual hypernyms needed separate encoding)."
		)

	print(
		f"[BRIDGE] emb_cache size: {len(emb_cache):,} entries "
	  f"(raw concepts + target canonicals, single encoding run)."
	)

	# =========================================================================
	# STEP 12: SAVE OUTPUTS
	# =========================================================================
	# 1. canonical_map.json — raw VLM concept → canonical label
	#    Used by Stage 3 (Micro-CGD Audit) and Stage 4 (Consolidation).
	final_canonical_dict: dict = df_clean.set_index('label')['canonical'].to_dict()
	with open(canonical_map_path, 'w', encoding='utf-8') as f:
		json.dump(final_canonical_dict, f, indent=2, ensure_ascii=False)
	print(f"\n[BRIDGE] Saved canonical_map ({len(final_canonical_dict):,} entries) → {canonical_map_path}")

	# 2. target_vocabulary.json — defines the multi-hot vector dimensions for Stage 4/5.
	#    Wrap in a metadata dict so Stage 4 can load unambiguously and
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Bridge: Global Aggregation & Ontology Discovery")
	parser.add_argument("--jsonl_file", "-jsonl", type=str, required=True, help="Stage 2 modality conflict audit JSONL file",)
	parser.add_argument("--embedding_mode_id", "-emb", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="SentenceTransformer model for canonical analysis",)
	parser.add_argument("--batch_size", "-bs", type=int, default=2**10, help="Batch size for embedding")
	parser.add_argument("--column", "-col", type=str, default="mlm_cot_raw", help="Column to use for canonical analysis",)
	parser.add_argument("--device", "-dev", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use for computation",)
	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose output")

	args = parser.parse_args()
	print(args)
	set_seeds(seed=42)

	if "_modality_conflict_audit.jsonl" not in args.jsonl_file:
		raise ValueError(f"JSONL must be a Stage 2 modality conflict audit file. Got: {args.jsonl_file}")

	cluster_and_save_priors(
		input_jsonl=args.jsonl_file,
		model_id=args.embedding_mode_id,
		batch_size=args.batch_size,
		column=args.column,
		device=args.device,
		verbose=args.verbose,
	)