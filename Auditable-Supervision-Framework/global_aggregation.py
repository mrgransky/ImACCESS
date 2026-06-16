import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

# local:
# nohup python -u global_aggregation.py -jsonl /home/farid/datasets/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl -v > logs/global_aggregation.log 2>&1 &

from utils import *
from clustering import *
from nlp_utils import STOPWORDS

def filter_generic_vocabulary(
	df_clean: pd.DataFrame,
	canonical_freq_dict: Dict[str, int],
	tau_hapax: int = 2,
	tau_digit: int = 0.3,
	remove_singletons: bool = True,
	domain_blacklist: Optional[Set[str]] = None,
	verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:

	if domain_blacklist is None:
		domain_blacklist = set()
	
	def _has_digit_content(s: str) -> bool:
		s = s.strip()
		
		# Pure year (1800–2099)
		if re.match(r'^(1[89]\d{2}|20\d{2})$', s):
			return True
		
		# Pure digit string
		if s.isdigit():
			return True
		
		# Mixed: digit ratio > tau_digit among alphanumeric chars
		digits = sum(c.isdigit() for c in s)
		alphas = sum(c.isalpha() for c in s)
		if digits > 0 and (digits + alphas) > 0:
			if digits / (digits + alphas) > tau_digit:
				return True
		
		return False
	
	cluster_sizes  = df_clean.groupby('cluster').size().to_dict()
	removed_labels, keep_mask = [], []
	print(f"\n[VOCAB GATE] df_clean: {df_clean.shape}")
	for _, row in df_clean.iterrows():
		canonical    = row['canonical']
		label        = row['label']
		cluster      = row['cluster']
		freq         = canonical_freq_dict.get(canonical, 0)
		cluster_size = cluster_sizes.get(cluster, 1)
		reason = None

		if freq < tau_hapax:
			reason = f"hapax (freq={freq})"
		elif remove_singletons and cluster_size == 1:
			reason = "singleton cluster"
		elif _has_digit_content(canonical) or _has_digit_content(label):
			reason = "digit-dominated"
		elif canonical.lower() in domain_blacklist or label.lower() in domain_blacklist:
			reason = "domain blacklist"

		if reason:
			removed_labels.append(label)
			keep_mask.append(False)
			if verbose:
				print(f"{label:<35}canonical: {canonical:<35}{reason}")
		else:
			keep_mask.append(True)
	
	keep_mask    = np.array(keep_mask, dtype=bool)
	kept_indices = np.where(keep_mask)[0]
	df_filtered  = df_clean[keep_mask].reset_index(drop=True)
	
	if verbose:
		print("-"*120)
		n_before = df_clean['canonical'].nunique()
		n_after  = df_filtered['canonical'].nunique()
		print(f"\t{len(df_clean)} labels → {len(df_filtered)} (removed {len(removed_labels)})")
		print(f"\t{n_before} canonicals → {n_after} (removed {n_before - n_after})")
	
	return df_filtered, removed_labels, kept_indices

def cluster_and_save_priors(
	input_jsonl: str,
	model_id: str,
	batch_size: int,
	column: str,
	device: str,
	verbose: bool,
) -> dict:
	"""
	Bridge: Corpus-Level Ontology Discovery.
	Reads Stage 2 Evidence Receipts, pools raw VLM concepts per sample
	(regime-gated), normalises them via _post_process_, runs hierarchical
	clustering via clustering.py, and emits three artefacts consumed by
	Stage 3 and Stage 4:
			1. <stem>_canonical_map.json       — raw VLM concept → canonical label
			2. <stem>_target_vocabulary.json   — the emergent vocabulary V (with metadata)
			3. <stem>_emb_cache.pt            — {label: embedding} for fast lookup
	Intermediate checkpoints (embeddings, linkage matrix) are saved after
	each expensive step so the function is crash-safe and resumable.
	GMM Regime Induction (Step 1B):
			Each sample is characterised by a continuous conflict feature vector.
			For samples where NLI was bypassed (HARD_CONFLICT with orphan_ratio >=
			tau_orphan), asymmetry_gap is None. To avoid systematic underrepresentation
			of Hard Conflict samples in the GMM training set, we use a 2D feature
			space [set_similarity, orphan_ratio] when any sample is missing asym_gap,
			and a 3D space [set_similarity, orphan_ratio, abs(asym_gap)] otherwise.
			The feature dimensionality is recorded in the GMM payload so that Stage
			3/4 can apply the correct transform at inference time.
	Returns
	-------
	dict with keys:
			canonical_map_path, vocab_path, emb_path,
			n_vocab, n_canonical_map, n_emb_cache
	"""
	t_start = time.time()
	print(f"\n{'='*80}")
	print(f"[BRIDGE] Corpus-Level Ontology Discovery")
	print(f"{'='*80}")
	print(f"  ├─ Input   : {input_jsonl}")
	print(f"  ├─ Model   : {model_id}")
	print(f"  ├─ Batch   : {batch_size}")
	print(f"  ├─ Column  : {column}")
	print(f"  ├─ Device  : {device}")
	print(f"  └─ Verbose : {verbose}")

	DATASET_DIRECTORY = os.path.dirname(os.path.abspath(input_jsonl))
	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)
	stem = os.path.basename(input_jsonl).replace(".jsonl", "")

	ckpt_emb_path    = os.path.join(outputs_dir, f"{stem}_unique_label_embeddings.npy")
	ckpt_labels_path = os.path.join(outputs_dir, f"{stem}_unique_labels.npy")
	ckpt_Z_path      = os.path.join(outputs_dir, f"{stem}_linkage_matrix.npy")

	freqs_path         = os.path.join(outputs_dir, f"{stem}_global_label_frequency.json")
	canonical_map_path = os.path.join(outputs_dir, f"{stem}_canonical_map.json")
	vocab_path         = os.path.join(outputs_dir, f"{stem}_target_vocabulary.json")
	emb_path           = os.path.join(outputs_dir, f"{stem}_emb_cache.pt")
	gmm_path           = os.path.join(outputs_dir, f"{stem}_conflict_gmm.pkl")

	print(f"\n[BRIDGE] Output directory : {outputs_dir}")
	print(f"[BRIDGE] Checkpoint paths :")
	print(f"  ├─ Embeddings  : {ckpt_emb_path}")
	print(f"  ├─ Labels      : {ckpt_labels_path}")
	print(f"  └─ Linkage     : {ckpt_Z_path}")

	print(f"[BRIDGE] Final output paths :")
	print(f"  ├─ Frequencies : {freqs_path}")
	print(f"  ├─ Canon map   : {canonical_map_path}")
	print(f"  ├─ Vocabulary  : {vocab_path}")
	print(f"  ├─ Emb cache   : {emb_path}")
	print(f"  └─ GMM payload : {gmm_path}")

	# ══════════════════════════════════════════════════════════════════════════
	# STEP 1: REGIME-GATED CONCEPT POOLING
	# Gate fused_concepts by regime:
	#   HARD_CONFLICT  → text_c + vis_c only (fused is empty or hallucinated blend)
	#   SOFT_CONFLICT  → text_c + vis_c + fused_c (fused is trustworthy)
	#   AGREEMENT      → text_c + vis_c + fused_c (fused is trustworthy)
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 1] Regime-Gated Concept Pooling")
	print(f"{'─'*80}")
	all_sample_labels: List[List[str]] = []
	regime_counts: dict                = {}
	pool_size_by_regime: dict          = {}
	skipped_count    = 0
	malformed_count  = 0
	empty_pool_count = 0

	# GMM feature collection
	# Hard Conflict samples with orphan_ratio >= tau_orphan: NLI bypassed, so asym_gap is None. 
	# Including only the 3D subset would systematically exclude most Hard Conflict samples, 
	# biasing the GMM centroid toward the Agreement/Soft boundary.
	#
	# Strategy: collect two parallel lists.
	#   gmm_features_3d — samples where all three metrics are valid floats.
	#   gmm_features_2d — ALL samples where at least set_sim and orphan_ratio
	#                     are valid (asym_gap may be None).
	#
	# After collection we decide which space to use:
	#   • If the fraction of 3D-eligible samples is >= GMM_3D_COVERAGE_THRESHOLD
	#     (default 0.80), use 3D — the Hard Conflict underrepresentation is minor.
	#   • Otherwise fall back to 2D so that Hard Conflict samples are fully
	#     represented. The feature_dim is stored in the GMM payload so Stage 3/4
	#     applies the correct scaler at inference time.
	GMM_3D_COVERAGE_THRESHOLD = 0.80  # fraction of valid-metric samples that must be 3D-eligible
	gmm_features_3d: List[List[float]] = []  # [set_sim, orphan_ratio, abs_gap]
	gmm_features_2d: List[List[float]] = []  # [set_sim, orphan_ratio]
	with open(input_jsonl, 'r', encoding='utf-8') as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				receipt = json.loads(line)
			except json.JSONDecodeError as e:
				malformed_count += 1
				if verbose:
					print(f"[BRIDGE][WARN] Malformed JSON at line {line_no}: {e}")
				continue
			regime   = receipt.get("regime", "UNKNOWN")
			mlm_data = receipt.get(column, {})
			
			# Track regime distribution for diagnostics
			regime_counts[regime] = regime_counts.get(regime, 0) + 1
			
			# Skip samples with no usable modality signal.
			if regime not in ("AGREEMENT", "HARD_CONFLICT", "SOFT_CONFLICT", "MISSING_MODALITY"):
				skipped_count += 1
				if verbose:
					print(f"[BRIDGE][SKIP] line {line_no}: unknown regime '{regime}'")
				continue

			if not isinstance(mlm_data, dict):
				skipped_count += 1
				if verbose:
					print(f"[BRIDGE][SKIP] line {line_no}: column '{column}': {type(mlm_data).__name__} expected dict")
				continue
			
			text_c  = mlm_data.get("text_concepts",  []) or []
			vis_c   = mlm_data.get("visual_concepts", []) or []
			fused_c = mlm_data.get("fused_concepts",  []) or []

			# ── Collect continuous conflict metrics for GMM (Step 1B) ──────────
			# Only collect from regimes that have meaningful conflict metrics.
			# MISSING_MODALITY samples have no symmetric/asymmetric audit output.
			if regime in ("AGREEMENT", "HARD_CONFLICT", "SOFT_CONFLICT"):
				metrics = receipt.get("metrics", {})
				if metrics:
					set_sim      = metrics.get("set_similarity")
					orphan_ratio = metrics.get("orphan_ratio")
					asym_gap     = metrics.get("asymmetry_gap")
					# 2D: include if at least set_sim and orphan_ratio are valid.
					# This captures NLI-bypassed Hard Conflict samples (asym_gap=None).
					if set_sim is not None and orphan_ratio is not None:
						gmm_features_2d.append([float(set_sim), float(orphan_ratio)])
						# 3D: additionally require a valid asym_gap.
						if asym_gap is not None:
							gmm_features_3d.append([float(set_sim), float(orphan_ratio), abs(float(asym_gap))])
			
			# Regime-gated concept pooling
			if regime == "HARD_CONFLICT":
				# fused_concepts is unreliable: either empty (VLM detected conflict)
				# or a hallucinated blend of two disjoint modalities.
				# text_c and vis_c are individually coherent → include both.
				sample_pool = text_c + vis_c
			else:
				# AGREEMENT and SOFT_CONFLICT: fused_concepts is trustworthy.
				sample_pool = text_c + vis_c + fused_c
			
			# _post_process_: normalise raw VLM strings
			# Strip whitespace, lowercase, drop pure-digit tokens, drop empty/None.
			sample_pool = [
				lbl.strip().lower()
				for lbl in sample_pool
				if lbl and isinstance(lbl, str) and lbl.strip()
				and not lbl.strip().replace(" ", "").isdigit()
			]

			# Deduplicate within sample (preserve order via dict.fromkeys)
			sample_pool = list(dict.fromkeys(sample_pool))
			if not sample_pool:
				empty_pool_count += 1
				if verbose:
					print(
						f"[BRIDGE][SKIP] line {line_no}: regime={regime} "
						f"but pool is empty after normalisation "
						f"(text={len(text_c)}, vis={len(vis_c)}, fused={len(fused_c)})"
					)
				continue
			all_sample_labels.append(sample_pool)
			
			# Track pool size per regime for diagnostics
			if regime not in pool_size_by_regime:
				pool_size_by_regime[regime] = []
			pool_size_by_regime[regime].append(len(sample_pool))
	
	total_receipts = sum(regime_counts.values())
	print(f"\n[BRIDGE][STEP 1] Parsing complete")
	print(f"  ├─ Total lines parsed    : {total_receipts:,}")
	print(f"  ├─ Malformed JSON lines  : {malformed_count:,}")
	print(f"  ├─ Skipped (bad regime)  : {skipped_count:,}")
	print(f"  ├─ Skipped (empty pool)  : {empty_pool_count:,}")
	print(f"  ├─ Samples → vocabulary  : {len(all_sample_labels):,}")
	print(f"  ├─ GMM-eligible (2D)     : {len(gmm_features_2d):,}")
	print(f"  └─ GMM-eligible (3D)     : {len(gmm_features_3d):,}")

	print(f"\n[BRIDGE][STEP 1] Regime distribution:")
	for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
		avg_pool = (
			f"avg_pool={sum(pool_size_by_regime[r])/len(pool_size_by_regime[r]):.1f}"
			if r in pool_size_by_regime else "avg_pool=N/A"
		)
		print(
			f"  ├─ {r:<20} {cnt:>8,}  "
			f"({cnt / max(total_receipts, 1) * 100:.1f}%)  {avg_pool}"
		)

	if verbose:
		print(f"\n[BRIDGE][STEP 1] Per-sample pools (first 20):")
		for i, sample in enumerate(all_sample_labels[:20]):
			print(f"[{i:7d}] ({len(sample):3d} concepts) {sample}")
	
	if not all_sample_labels:
		raise ValueError(
			"[BRIDGE] No samples survived regime gating. "
			"Check that the input JSONL contains AGREEMENT / SOFT_CONFLICT / "
			"HARD_CONFLICT receipts and that the column name is correct."
		)

	# ══════════════════════════════════════════════════════════════════════════
	# STEP 1B: UNSUPERVISED CONFLICT REGIME INDUCTION (GMM)
	# ══════════════════════════════════════════════════════════════════════════
	# GMM induction is a logically distinct
	# step from concept pooling. Placed here as Step 1B so it runs immediately
	# after all receipts are parsed, before any embedding or clustering work.
	#
	# StandardScaler is fitted on the chosen
	# feature matrix and saved in the GMM payload. Stage 3/4 MUST apply
	# payload["scaler"].transform(f_raw) before calling gmm.predict_proba().
	#
	# BIC/AIC model selection): BIC is computed over K∈{2,…,6}.
	# K=3 is used by design (three named regimes), but a warning is emitted if
	# BIC selects a different K so the paper can address this explicitly.
	#
	# centroid separation confidence: After labelling clusters,
	# the orphan_ratio gap between the top-2 clusters is checked. A gap < 0.05
	# triggers a warning that manual inspection is recommended.
	#
	# Hard Conflict underrepresentation: Feature dimensionality
	# is chosen based on 3D coverage fraction. If >= GMM_3D_COVERAGE_THRESHOLD
	# of valid-metric samples are 3D-eligible, use 3D; otherwise fall back to 2D.
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 1B] Unsupervised Conflict Regime Induction (GMM)")
	print(f"{'─'*80}")
	_MIN_GMM_SAMPLES = 3  # absolute minimum to attempt a GMM fit
	_n_valid = len(gmm_features_2d)  # 2D is always the superset
	if _n_valid < _MIN_GMM_SAMPLES:
			print(
					f"[BRIDGE][STEP 1B][WARN] Insufficient samples ({_n_valid}) to train GMM "
					f"(minimum={_MIN_GMM_SAMPLES}). Skipping GMM generation."
			)
	else:
			# ── FIX (Issue 4): Choose feature dimensionality ──────────────────────
			_n_3d = len(gmm_features_3d)
			_coverage_3d = _n_3d / max(_n_valid, 1)
			print(
					f"[BRIDGE][STEP 1B] Feature dimensionality decision: "
					f"3D-eligible={_n_3d:,}/{_n_valid:,} ({_coverage_3d:.1%}) | "
					f"threshold={GMM_3D_COVERAGE_THRESHOLD:.0%}"
			)
			if _coverage_3d >= GMM_3D_COVERAGE_THRESHOLD:
					features_arr = np.array(gmm_features_3d)
					feature_dim  = 3
					feature_names = ["set_similarity", "orphan_ratio", "abs_asym_gap"]
					print(
							f"[BRIDGE][STEP 1B] Using 3D feature space "
							f"[set_sim, orphan_ratio, abs_gap] ({_n_3d:,} samples)."
					)
			else:
					features_arr = np.array(gmm_features_2d)
					feature_dim  = 2
					feature_names = ["set_similarity", "orphan_ratio"]
					print(
							f"[BRIDGE][STEP 1B][WARN] 3D coverage ({_coverage_3d:.1%}) below threshold "
							f"({GMM_3D_COVERAGE_THRESHOLD:.0%}). Falling back to 2D feature space "
							f"[set_sim, orphan_ratio] ({_n_valid:,} samples) to avoid systematic "
							f"underrepresentation of NLI-bypassed Hard Conflict samples."
					)

			# Fit StandardScaler
			# Features live on different scales and distributions. Without scaling,
			# set_similarity dominates the GMM covariance structure.
			scaler = StandardScaler()
			features_scaled = scaler.fit_transform(features_arr)
			print(
					f"[BRIDGE][STEP 1B] StandardScaler fitted on {features_arr.shape[0]:,} samples "
					f"(dim={feature_dim}). "
					f"Means: {scaler.mean_.round(4).tolist()} | "
					f"Stds: {scaler.scale_.round(4).tolist()}"
			)

			# BIC/AIC model selection over K∈{2,…,6}
			print(f"[BRIDGE][STEP 1B] BIC/AIC model selection over K∈{{2,…,6}}:")
			bic_scores: dict = {}
			aic_scores: dict = {}
			for k in range(2, 7):
					_g = GaussianMixture(
							n_components=k, covariance_type='full', random_state=42, max_iter=200
					)
					_g.fit(features_scaled)
					bic_scores[k] = _g.bic(features_scaled)
					aic_scores[k] = _g.aic(features_scaled)
					print(
							f"  ├─ K={k}: BIC={bic_scores[k]:>12.2f}  AIC={aic_scores[k]:>12.2f}"
					)
			optimal_k_bic = min(bic_scores, key=bic_scores.get)
			optimal_k_aic = min(aic_scores, key=aic_scores.get)
			print(
					f"  └─ BIC-optimal K={optimal_k_bic} | AIC-optimal K={optimal_k_aic} | "
					f"Design K=3 (three named conflict regimes)"
			)
			if optimal_k_bic != 3:
					print(
							f"[BRIDGE][STEP 1B][WARN] BIC selects K={optimal_k_bic}, not K=3. "
							f"Proceeding with K=3 by design (Agreement / Soft Conflict / Hard Conflict). "
							f"Report BIC curve in paper and justify K=3 via domain alignment."
					)
			else:
					print(
							f"[BRIDGE][STEP 1B] BIC confirms K=3 ✓ — "
							f"data-driven support for three conflict regimes."
					)

			# ── Fit the final K=3 GMM ─────────────────────────────────────────────
			print(
					f"[BRIDGE][STEP 1B] Fitting final 3-component GMM on "
					f"{features_arr.shape[0]:,} scaled {feature_dim}D vectors..."
			)
			gmm = GaussianMixture(
				n_components=3, 
				covariance_type='full', 
				random_state=42, 
				max_iter=200,
			)
			gmm.fit(features_scaled)
			print(
					f"[BRIDGE][STEP 1B] GMM converged={gmm.converged_} | "
					f"n_iter={gmm.n_iter_} | "
					f"log-likelihood={gmm.lower_bound_:.4f}"
			)
			# ── Deterministic cluster labelling via centroid rules ────────────────
			# Unscaled means for interpretable reporting (inverse_transform).
			means_scaled   = gmm.means_                          # shape [3, feature_dim]
			means_unscaled = scaler.inverse_transform(means_scaled)
			# Rule 1: HARD_CONFLICT → cluster with highest orphan_ratio (column 1)
			orphan_col   = 1  # index of orphan_ratio in both 2D and 3D feature vectors
			orphan_means = means_unscaled[:, orphan_col]
			sorted_by_orphan = np.argsort(orphan_means)[::-1]  # descending
			hard_cluster_idx = int(sorted_by_orphan[0])
			# ── FIX (Issue 3): Centroid separation confidence check ───────────────
			orphan_gap = float(orphan_means[sorted_by_orphan[0]] - orphan_means[sorted_by_orphan[1]])
			if orphan_gap < 0.05:
					print(
							f"[BRIDGE][STEP 1B][WARN] Hard/Soft cluster separation is weak "
							f"(orphan_ratio gap between top-2 clusters = {orphan_gap:.4f} < 0.05). "
							f"GMM may not have found a clean Hard Conflict cluster. "
							f"Manual inspection of cluster centroids is recommended before "
							f"using regime labels in the paper."
					)
			else:
					print(
							f"[BRIDGE][STEP 1B] Hard Conflict cluster separation: "
							f"orphan_ratio gap={orphan_gap:.4f} ✓"
					)
			# Rule 2: SOFT_CONFLICT → of the remaining two, highest abs_asym_gap (col 2)
			#         If feature_dim==2 (fallback), use set_similarity (col 0) inverted:
			#         lower set_sim among the two remaining → more conflict → SOFT.
			remaining = [i for i in range(3) if i != hard_cluster_idx]
			if feature_dim == 3:
					gap_col = 2  # abs_asym_gap
					soft_cluster_idx = (
							remaining[0]
							if means_unscaled[remaining[0], gap_col] > means_unscaled[remaining[1], gap_col]
							else remaining[1]
					)
					print(
							f"[BRIDGE][STEP 1B] Soft Conflict discriminator: "
							f"abs_asym_gap (col {gap_col}) — "
							f"cluster {remaining[0]}: {means_unscaled[remaining[0], gap_col]:.4f} | "
							f"cluster {remaining[1]}: {means_unscaled[remaining[1], gap_col]:.4f}"
					)
			else:
					# 2D fallback: lower set_similarity → more conflict → SOFT_CONFLICT
					sim_col = 0
					soft_cluster_idx = (
							remaining[0]
							if means_unscaled[remaining[0], sim_col] < means_unscaled[remaining[1], sim_col]
							else remaining[1]
					)
					print(
							f"[BRIDGE][STEP 1B] Soft Conflict discriminator (2D fallback): "
							f"set_similarity (col {sim_col}, lower=more conflict) — "
							f"cluster {remaining[0]}: {means_unscaled[remaining[0], sim_col]:.4f} | "
							f"cluster {remaining[1]}: {means_unscaled[remaining[1], sim_col]:.4f}"
					)
			# Rule 3: AGREEMENT → the remaining cluster
			agree_cluster_idx = [
					i for i in range(3)
					if i not in (hard_cluster_idx, soft_cluster_idx)
			][0]
			class_mapping = {
					agree_cluster_idx: "AGREEMENT",
					soft_cluster_idx:  "SOFT_CONFLICT",
					hard_cluster_idx:  "HARD_CONFLICT",
			}
			# ── Report unscaled centroids for paper / LaTeX table ─────────────────
			print(f"\n[BRIDGE][STEP 1B] Discovered Cluster Centroids (unscaled):")
			header = "  ".join(f"{n:>18}" for n in feature_names)
			print(f"  {'Regime':<20}  {header}")
			for cluster_idx, regime_name in [
					(agree_cluster_idx, "AGREEMENT"),
					(soft_cluster_idx,  "SOFT_CONFLICT"),
					(hard_cluster_idx,  "HARD_CONFLICT"),
			]:
					vals = "  ".join(f"{v:>18.4f}" for v in means_unscaled[cluster_idx])
					print(f"  {regime_name:<20}  {vals}  (cluster {cluster_idx})")
			# ── Package and serialise GMM payload ─────────────────────────────────
			# Stage 3/4 must load this payload and call:
			#   f_raw    = [[set_sim, orphan_ratio]] or [[set_sim, orphan_ratio, abs_gap]]
			#              depending on payload["feature_dim"]
			#   f_scaled = payload["scaler"].transform(f_raw)
			#   probs    = payload["gmm_model"].predict_proba(f_scaled)  # shape [1, 3]
			#   regime_probs = {payload["class_mapping"][k]: probs[0][k] for k in range(3)}
			gmm_payload = {
					"gmm_model":    gmm,
					"scaler":       scaler,
					"class_mapping": class_mapping,
					"feature_dim":  feature_dim,
					"feature_names": feature_names,
					"bic_scores":   bic_scores,
					"aic_scores":   aic_scores,
					"optimal_k_bic": optimal_k_bic,
					"optimal_k_aic": optimal_k_aic,
					"n_samples_fit": int(features_arr.shape[0]),
					"centroids_unscaled": {
							regime_name: means_unscaled[cluster_idx].tolist()
							for cluster_idx, regime_name in class_mapping.items()
					},
					"orphan_gap":   orphan_gap,
			}
			joblib.dump(gmm_payload, gmm_path)
			print(f"\n[BRIDGE][STEP 1B] GMM payload saved → {gmm_path}")
			print(
					f"[BRIDGE][STEP 1B] Payload keys: "
					f"{list(gmm_payload.keys())}"
			)
			print(
					f"[BRIDGE][STEP 1B] Usage in Stage 3/4:\n"
					f"    payload   = joblib.load(gmm_path)\n"
					f"    f_raw     = [[set_sim, orphan_ratio]]  "
					f"# or 3D if payload['feature_dim']==3\n"
					f"    f_scaled  = payload['scaler'].transform(f_raw)\n"
					f"    probs     = payload['gmm_model'].predict_proba(f_scaled)  # [1, 3]\n"
					f"    regime_probs = {{payload['class_mapping'][k]: probs[0][k] for k in range(3)}}"
			)
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 2: GLOBAL FREQUENCY COUNTS (Reusability Prior)
	# Convert Counter → plain dict before saving and before passing to clustering.py.
	# Counter's default-zero behaviour silently disables the freq-gain safety
	# check inside assign_canonical_labels for labels not seen in the corpus
	# (Counter returns 0 for missing keys, so the .get(lbl, 1) fallback in the
	# safety ratio never triggers, causing the ratio to always be 0/max(0,1)=0
	# and the safety check to always revert to pure centroid similarity).
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 2] Global Frequency Counts (Reusability Prior)")
	print(f"{'─'*80}")
	label_freq_dict: dict = dict(
			Counter(
					lbl
					for sample in all_sample_labels
					for lbl in sample
					if lbl
			)
	)
	# ── Frequency diagnostics ─────────────────────────────────────────────────
	freq_values   = sorted(label_freq_dict.values(), reverse=True)
	total_concept_occurrences = sum(freq_values)
	singleton_count = sum(1 for v in freq_values if v == 1)
	top_n = 20
	print(f"[BRIDGE][STEP 2] Unique concepts (post-normalisation) : {len(label_freq_dict):,}")
	print(f"[BRIDGE][STEP 2] Total concept occurrences            : {total_concept_occurrences:,}")
	print(f"[BRIDGE][STEP 2] Singletons (freq=1)                  : {singleton_count:,} "
				f"({singleton_count / max(len(label_freq_dict), 1) * 100:.1f}%)")
	print(f"[BRIDGE][STEP 2] Frequency stats: "
				f"max={freq_values[0] if freq_values else 0}, "
				f"median={freq_values[len(freq_values)//2] if freq_values else 0}, "
				f"mean={total_concept_occurrences/max(len(label_freq_dict),1):.2f}")
	print(f"\n[BRIDGE][STEP 2] Top-{top_n} most frequent concepts:")
	for rank, (lbl, cnt) in enumerate(
			sorted(label_freq_dict.items(), key=lambda x: -x[1])[:top_n], start=1
	):
			bar = "█" * min(int(cnt / max(freq_values[0], 1) * 30), 30)
			print(f"  {rank:3d}. {lbl:<40} {cnt:>6,}  {bar}")
	if verbose:
			print(f"\n[BRIDGE][STEP 2] Bottom-10 least frequent concepts:")
			for lbl, cnt in sorted(label_freq_dict.items(), key=lambda x: x[1])[:10]:
					print(f"  {lbl:<40} {cnt:>6,}")
	with open(freqs_path, 'w', encoding='utf-8') as f:
			json.dump(label_freq_dict, f, indent=2, ensure_ascii=False)
	print(f"\n[BRIDGE][STEP 2] Saved global frequencies → {freqs_path}")
	unique_labels: List[str] = sorted(label_freq_dict.keys())
	print(f"[BRIDGE][STEP 2] Unique concepts for vocabulary induction: {len(unique_labels):,}")
	# ── MISSING_MODALITY coverage diagnostic ──────────────────────────────────
	mm_vis_concepts: set = set()
	with open(input_jsonl, 'r', encoding='utf-8') as f:
			for line in f:
					line = line.strip()
					if not line:
							continue
					try:
							receipt = json.loads(line)
					except json.JSONDecodeError:
							continue
					if receipt.get("regime") != "MISSING_MODALITY":
							continue
					mlm_data = receipt.get(column, {})
					if not isinstance(mlm_data, dict):
							continue
					for c in mlm_data.get("visual_concepts", []) or []:
							if c and isinstance(c, str):
									mm_vis_concepts.add(c.strip().lower())
	orphaned = mm_vis_concepts - set(label_freq_dict.keys())
	print(f"[BRIDGE][DIAG] MISSING_MODALITY vis_c concepts  : {len(mm_vis_concepts):,}")
	print(f"[BRIDGE][DIAG] Not in corpus (orphaned)         : {len(orphaned):,}")
	if orphaned:
			print(f"[BRIDGE][DIAG] Orphaned sample: {sorted(orphaned)[:20]}")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 3: EMBEDDING (with crash-safe checkpoint)
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 3] Concept Embedding")
	print(f"{'─'*80}")
	resume_emb = (
			os.path.exists(ckpt_emb_path) and
			os.path.exists(ckpt_labels_path)
	)
	if resume_emb:
			print(f"[BRIDGE][STEP 3] Checkpoint found — loading cached embeddings.")
			X             = np.load(ckpt_emb_path)
			unique_labels = np.load(ckpt_labels_path, allow_pickle=True).tolist()
			print(f"[BRIDGE][STEP 3] Loaded  X={X.shape} dtype={X.dtype} "
						f"for {len(unique_labels):,} labels.")
			if len(unique_labels) != len(label_freq_dict):
					print(
							f"[BRIDGE][WARN] Checkpoint has {len(unique_labels):,} labels but "
							f"current corpus has {len(label_freq_dict):,} unique concepts. "
							f"Delete checkpoints and re-run to rebuild from scratch."
					)
			else:
					print(f"[BRIDGE][STEP 3] Checkpoint label count matches corpus ✓")
	else:
			print(f"[BRIDGE][STEP 3] No checkpoint — encoding {len(unique_labels):,} concepts.")
			print(f"[BRIDGE][STEP 3] Loading model: {model_id}")
			t_enc = time.time()
			model = SentenceTransformer(
					model_id,
					device=device,
					trust_remote_code=True,
					cache_folder=cache_directory[os.getenv('USER')],
					token=os.getenv("HUGGINGFACE_TOKEN"),
			).to(device)
			print(f"[BRIDGE][STEP 3] Model loaded in {time.time()-t_enc:.1f}s")
			print(f"[BRIDGE][STEP 3] Encoding (batch_size={batch_size})...")
			t_enc = time.time()
			X = model.encode(
					unique_labels,
					batch_size=batch_size,
					show_progress_bar=verbose,
					convert_to_numpy=True,
					normalize_embeddings=True,
					precision='float32',
			)
			enc_time = time.time() - t_enc
			print(
					f"[BRIDGE][STEP 3] Encoded {len(unique_labels):,} concepts in {enc_time:.1f}s "
					f"({len(unique_labels)/max(enc_time,1e-6):.0f} concepts/s)"
			)
			print(
					f"[BRIDGE][STEP 3] X={X.shape} dtype={X.dtype} "
					f"(min={X.min():.4f}, max={X.max():.4f}, "
					f"mean={X.mean():.4f}, std={X.std():.4f})"
			)
			norms = np.linalg.norm(X, axis=1)
			print(
					f"[BRIDGE][STEP 3] Embedding norms: "
					f"min={norms.min():.6f}, max={norms.max():.6f}, mean={norms.mean():.6f} "
					f"(expected ≈1.0 for normalised embeddings)"
			)
			np.save(ckpt_emb_path,    X)
			np.save(ckpt_labels_path, np.array(unique_labels, dtype=object))
			print(f"[BRIDGE][STEP 3] Checkpointed embeddings → {ckpt_emb_path}")
			print(f"[BRIDGE][STEP 3] Checkpointed labels     → {ckpt_labels_path}")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 4: LINKAGE MATRIX (with crash-safe checkpoint)
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 4] Hierarchical Linkage Matrix")
	print(f"{'─'*80}")
	if os.path.exists(ckpt_Z_path):
			print(f"[BRIDGE][STEP 4] Checkpoint found — loading cached linkage matrix.")
			Z = np.load(ckpt_Z_path)
			print(f"[BRIDGE][STEP 4] Loaded Z={Z.shape} dtype={Z.dtype}")
	else:
			backend = "fastcluster" if use_fastcluster else "scipy"
			print(
					f"[BRIDGE][STEP 4] Building Ward linkage matrix "
					f"(n={len(unique_labels):,}, backend={backend})..."
			)
			t_link = time.time()
			if use_fastcluster:
					Z = fastcluster.linkage(X, method='ward', metric='euclidean')
			else:
					Z = linkage(X, method='ward', metric='euclidean')
			link_time = time.time() - t_link
			print(
					f"[BRIDGE][STEP 4] Linkage complete in {link_time:.1f}s. "
					f"Z={Z.shape} dtype={Z.dtype}"
			)
			print(
					f"[BRIDGE][STEP 4] Linkage distance range: "
					f"[{Z[:,2].min():.4f}, {Z[:,2].max():.4f}]"
			)
			np.save(ckpt_Z_path, Z)
			print(f"[BRIDGE][STEP 4] Checkpointed linkage matrix → {ckpt_Z_path}")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 5: ADAPTIVE OPTIMAL CLUSTER SEARCH
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 5] Adaptive Optimal Cluster Search")
	print(f"{'─'*80}")
	print(
			f"[BRIDGE][STEP 5] Hyperparameters: "
			f"target_intra_sim=0.69, consolidation=[3.8, 5.0], "
			f"singleton_ratio=0.015, quality_weight=0.5"
	)
	t_clust = time.time()
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
	print(f"[BRIDGE][STEP 5] Cluster search complete in {time.time()-t_clust:.1f}s")
	print(
			f"[BRIDGE][STEP 5] Optimal k={stats['n_clusters']:,} clusters | "
			f"consolidation={stats['consolidation_ratio']:.2f}x | "
			f"intra_sim={stats['mean_intra_similarity']:.4f}"
	)
	df = pd.DataFrame({'label': unique_labels, 'cluster': cluster_labels_arr})
	assert len(df) == len(unique_labels) == len(X), (
			f"[BRIDGE][ASSERT] df/unique_labels/X length mismatch: "
			f"{len(df)} / {len(unique_labels)} / {len(X)}"
	)
	cluster_sizes = df['cluster'].value_counts()
	print(f"\n[BRIDGE][STEP 5] Cluster size distribution:")
	print(f"  ├─ Total clusters        : {stats['n_clusters']:,}")
	print(f"  ├─ Singleton clusters    : {(cluster_sizes == 1).sum():,}")
	print(f"  ├─ Clusters with 2-5     : {((cluster_sizes >= 2) & (cluster_sizes <= 5)).sum():,}")
	print(f"  ├─ Clusters with 6-20    : {((cluster_sizes >= 6) & (cluster_sizes <= 20)).sum():,}")
	print(f"  └─ Clusters with >20     : {(cluster_sizes > 20).sum():,}")
	if verbose:
			print(f"\n[BRIDGE][STEP 5] Top-10 largest clusters:")
			for cid, sz in cluster_sizes.head(10).items():
					members = df[df['cluster'] == cid]['label'].tolist()
					print(f"  cluster {cid:5d}: {sz:4d} members | {members[:8]}")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 6: 5-SIGNAL CANONICAL SELECTION
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 6] 5-Signal Canonical Selection")
	print(f"{'─'*80}")
	if resume_emb:
			print(f"[BRIDGE][STEP 6] Resume path: loading embedding model for canonical selection.")
			model = SentenceTransformer(
					model_id,
					device=device,
					trust_remote_code=True,
					cache_folder=cache_directory[os.getenv('USER')],
					token=os.getenv("HUGGINGFACE_TOKEN"),
			).to(device)
			print(f"[BRIDGE][STEP 6] Model loaded.")
	t_canon = time.time()
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
	print(f"\n[BRIDGE][STEP 6] Canonical selection complete in {time.time()-t_canon:.1f}s")
	print(f"  ├─ Total clusters processed  : {len(cluster_canonicals):,}")
	print(f"  ├─ Virtual hypernyms used    : {virtual_used_count:,}")
	print(f"  ├─ Freq-overridden canonicals: {freq_changed_count:,}")
	print(f"  ├─ Total similarity loss     : {sum(total_sim_loss):.4f}")
	print(f"  ├─ Total frequency gain      : {sum(total_freq_gain):,}")
	print(f"  └─ Questionable assignments  : {len(questionable_examples):,}")
	if verbose and questionable_examples:
			print(f"\n[BRIDGE][STEP 6] Questionable canonical assignments (review manually):")
			for ex in questionable_examples[:15]:
					print(f"  {ex}")
	df['canonical'] = df['cluster'].map(lambda c: cluster_canonicals[c]['canonical'])
	nan_canonical_count = df['canonical'].isna().sum()
	if nan_canonical_count > 0:
			print(
					f"[BRIDGE][WARN] {nan_canonical_count} rows have NaN canonical — "
					f"cluster_canonicals may be missing entries for some cluster IDs."
			)
			if verbose:
					print(df[df['canonical'].isna()].head(10).to_string())
	else:
			print(f"[BRIDGE][STEP 6] All {len(df):,} rows have a valid canonical ✓")
	print(f"\n[BRIDGE][STEP 6] df after canonical assignment: {df.shape}")
	if verbose:
			print(df.head(10).to_string())
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 7: INJECT VIRTUAL HYPERNYMS AS GENUINE ROWS
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 7] Virtual Hypernym Injection")
	print(f"{'─'*80}")
	virtual_rows: List[dict]       = []
	virtual_embs: List[np.ndarray] = []
	for cid, meta in cluster_canonicals.items():
			if not meta.get('virtual', False):
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
			if verbose:
					print(f"  [VH] cluster {cid:5d}: '{vh}'")
	if virtual_rows:
			df_before = df.shape
			X_before  = X.shape
			df = pd.concat([df, pd.DataFrame(virtual_rows)], ignore_index=True)
			X  = np.vstack([X, np.array(virtual_embs)])
			print(
					f"[BRIDGE][STEP 7] Injected {len(virtual_rows)} virtual hypernym(s): "
					f"df {df_before} → {df.shape} | X {X_before} → {X.shape}"
			)
			assert len(df) == len(X), (
					f"[BRIDGE][ASSERT] df/X length mismatch after VH injection: "
					f"{len(df)} / {len(X)}"
			)
	else:
			print(f"[BRIDGE][STEP 7] No virtual hypernyms to inject.")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 8: AUDIT — DROP LOW-COHESION AND POOR-CANONICAL CLUSTERS
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 8] Cluster Audit & Vocabulary Gating")
	print(f"{'─'*80}")
	print(f"[BRIDGE][STEP 8a] remove_problematic_cluster_labels: df={df.shape}, X={X.shape}")
	df_clean, X_clean, removed_labels_8a = remove_problematic_cluster_labels(
			df=df,
			embeddings=X,
			verbose=verbose,
	)
	print(
			f"[BRIDGE][STEP 8a] After audit: df_clean={df_clean.shape} | "
			f"X_clean={X_clean.shape} | removed={len(removed_labels_8a):,} labels"
	)
	if verbose and removed_labels_8a:
			print(f"[BRIDGE][STEP 8a] Removed labels (first 20): {removed_labels_8a[:20]}")
	print(f"\n[BRIDGE][STEP 8b] Building canonical frequency map...")
	canonical_freq_dict: dict = {}
	for _, row in df_clean.iterrows():
			canonical = row['canonical']
			raw_freq  = label_freq_dict.get(row['label'], 0)
			canonical_freq_dict[canonical] = canonical_freq_dict.get(canonical, 0) + raw_freq
	canon_freq_values = sorted(canonical_freq_dict.values(), reverse=True)
	print(
			f"[BRIDGE][STEP 8b] {len(canonical_freq_dict):,} canonicals | "
			f"freq range: [{min(canon_freq_values)}, {max(canon_freq_values)}] | "
			f"mean={sum(canon_freq_values)/max(len(canon_freq_values),1):.1f}"
	)
	if verbose:
			print(f"[BRIDGE][STEP 8b] Top-15 canonicals by aggregated frequency:")
			for canon, freq in sorted(canonical_freq_dict.items(), key=lambda x: -x[1])[:15]:
					print(f"  {canon:<40} {freq:>6,}")
	print(f"\n[BRIDGE][STEP 8c] filter_generic_vocabulary: df_clean={df_clean.shape}")
	df_before_gate = df_clean.shape
	df_clean, removed_labels_8c, kept_indices = filter_generic_vocabulary(
			df_clean=df_clean,
			canonical_freq_dict=canonical_freq_dict,
			domain_blacklist=STOPWORDS,
			verbose=verbose,
	)
	print(
			f"[BRIDGE][STEP 8c] Vocab gate: {df_before_gate} → {df_clean.shape} | "
			f"removed {len(removed_labels_8c):,} labels | "
			f"kept_indices: {len(kept_indices):,}"
	)
	if verbose and removed_labels_8c:
			print(f"[BRIDGE][STEP 8c] Removed labels (first 20): {removed_labels_8c[:20]}")
	blacklist_path = os.path.join(outputs_dir, f"{stem}_blacklisted_concepts.json")
	with open(blacklist_path, "w", encoding="utf-8") as f:
			json.dump(removed_labels_8c, f, indent=2, ensure_ascii=False)
	print(f"[BRIDGE][STEP 8c] Blacklist written: {len(removed_labels_8c)} entries → {blacklist_path}")
	print(f"\n[BRIDGE][STEP 8d] Building final canonical_map from df_clean...")
	final_canonical_dict: dict = df_clean.set_index('label')['canonical'].to_dict()
	valid_canonicals: set       = set(df_clean['canonical'].unique())
	invalid_mappings = {
			raw: canon
			for raw, canon in final_canonical_dict.items()
			if canon not in valid_canonicals
	}
	if invalid_mappings:
			print(
					f"[BRIDGE][WARN] {len(invalid_mappings)} mappings point to "
					f"invalid canonicals (should be 0 — check filter_generic_vocabulary):"
			)
			for raw, canon in list(invalid_mappings.items())[:10]:
					print(f"  '{raw}' → '{canon}' (NOT in valid_canonicals)")
	else:
			print(
					f"[BRIDGE][STEP 8d] All {len(final_canonical_dict):,} mappings point "
					f"to valid canonicals ✓"
			)
	print(
			f"[BRIDGE][STEP 8d] final_canonical_dict: {len(final_canonical_dict):,} mappings | "
			f"valid_canonicals: {len(valid_canonicals):,}"
	)
	if verbose:
			print(f"[BRIDGE][STEP 8d] Sample mappings (first 20):")
			for raw, canon in list(final_canonical_dict.items())[:20]:
					marker = "✓" if canon in valid_canonicals else "✗"
					print(f"  {marker} '{raw:<40}' → '{canon}'")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 9: RE-EVALUATE FINAL CLUSTER QUALITY
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 9] Final Cluster Quality Analysis")
	print(f"{'─'*80}")
	X_clean_final        = X_clean[kept_indices]
	unique_labels_clean  = df_clean['label'].values
	cluster_labels_clean = df_clean['cluster'].values
	canonical_map_int    = df_clean.groupby('cluster')['canonical'].first().to_dict()
	assert len(X_clean_final) == len(df_clean), (
			f"[BRIDGE][ASSERT] X_clean_final/df_clean length mismatch: "
			f"{len(X_clean_final)} / {len(df_clean)}"
	)
	print(
			f"[BRIDGE][STEP 9] Inputs: X_clean_final={X_clean_final.shape} | "
			f"df_clean={df_clean.shape} | "
			f"n_clusters={len(canonical_map_int):,}"
	)
	analyze_cluster_quality(
			embeddings=X_clean_final,
			labels=unique_labels_clean,
			cluster_assignments=cluster_labels_clean,
			canonical_labels=canonical_map_int,
			original_label_counts=label_freq_dict,
			verbose=verbose,
	)
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 10: EXTRACT EMERGENT TARGET VOCABULARY V
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 10] Emergent Target Vocabulary Extraction")
	print(f"{'─'*80}")
	target_vocab: List[str] = sorted(df_clean['canonical'].unique().tolist())
	print(f"[BRIDGE][STEP 10] |V| = {len(target_vocab):,} canonical labels")
	if verbose:
			print(f"[BRIDGE][STEP 10] Target vocabulary (first 40):")
			for i, v in enumerate(target_vocab[:40]):
					print(f"  {i:4d}. {v}")
			if len(target_vocab) > 40:
					print(f"  ... ({len(target_vocab) - 40} more)")
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 11: PRE-COMPUTE TARGET VOCABULARY EMBEDDINGS (emb_cache)
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 11] Pre-computing Target Vocabulary Embeddings (emb_cache)")
	print(f"{'─'*80}")
	df_labels_for_cache = df['label'].tolist()
	assert len(df_labels_for_cache) == len(X), (
			f"[BRIDGE][ASSERT] df label list / X length mismatch for emb_cache seeding: "
			f"{len(df_labels_for_cache)} / {len(X)}"
	)
	emb_cache: dict = {
			lbl: emb
			for lbl, emb in zip(df_labels_for_cache, X)
	}
	print(
			f"[BRIDGE][STEP 11] (a) Seeded emb_cache from df+X: "
			f"{len(emb_cache):,} entries (df has {len(df):,} rows, X has {len(X):,} rows)"
	)
	new_canonicals: List[str] = [tc for tc in target_vocab if tc not in emb_cache]
	already_cached = len(target_vocab) - len(new_canonicals)
	print(
			f"[BRIDGE][STEP 11] (b) Target vocab coverage: "
			f"{already_cached:,}/{len(target_vocab):,} already in cache | "
			f"{len(new_canonicals):,} need encoding"
	)
	if verbose and new_canonicals:
			print(f"[BRIDGE][STEP 11] New canonicals to encode: {new_canonicals}")
	if new_canonicals:
			print(
					f"[BRIDGE][STEP 11] (c) Encoding {len(new_canonicals):,} new canonical strings..."
			)
			t_enc2 = time.time()
			new_embs = model.encode(
					new_canonicals,
					batch_size=min(1024, len(new_canonicals)),
					show_progress_bar=verbose,
					convert_to_numpy=True,
					normalize_embeddings=True,
					precision='float32',
			)
			print(
					f"[BRIDGE][STEP 11] (c) Encoded in {time.time()-t_enc2:.1f}s | "
					f"new_embs={new_embs.shape}"
			)
			for lbl, emb in zip(new_canonicals, new_embs):
					emb_cache[lbl] = emb
			print(f"[BRIDGE][STEP 11] (d) Merged {len(new_canonicals):,} new embeddings into cache.")
	else:
			print(
					f"[BRIDGE][STEP 11] All {len(target_vocab):,} target canonicals already in cache."
			)
	missing_from_cache = [tc for tc in target_vocab if tc not in emb_cache]
	if missing_from_cache:
			print(
					f"[BRIDGE][WARN] {len(missing_from_cache)} target canonicals still missing "
					f"from emb_cache after encoding: {missing_from_cache}"
			)
	else:
			print(
					f"[BRIDGE][STEP 11] emb_cache coverage: all {len(target_vocab):,} "
					f"target canonicals present ✓"
			)
	print(
			f"[BRIDGE][STEP 11] Final emb_cache: {len(emb_cache):,} entries "
			f"(raw concepts + target canonicals, single encoding run)"
	)
	# ══════════════════════════════════════════════════════════════════════════
	# STEP 12: SAVE OUTPUTS
	# ══════════════════════════════════════════════════════════════════════════
	print(f"\n{'─'*80}")
	print(f"[BRIDGE][STEP 12] Saving Outputs")
	print(f"{'─'*80}")
	with open(canonical_map_path, 'w', encoding='utf-8') as f:
			json.dump(final_canonical_dict, f, indent=2, ensure_ascii=False)
	print(
			f"  ├─ canonical_map.json  : {len(final_canonical_dict):,} entries "
			f"→ {canonical_map_path}"
	)
	vocab_payload = {
			"vocabulary":   target_vocab,
			"size":         len(target_vocab),
			"source_jsonl": os.path.basename(input_jsonl),
			"model_id":     model_id,
			"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
			"note": (
					"Load via: vocab = json.load(f)['vocabulary']. "
					"Index i in this list corresponds to dimension i of the multi-hot vector."
			),
	}
	with open(vocab_path, 'w', encoding='utf-8') as f:
			json.dump(vocab_payload, f, indent=2, ensure_ascii=False)
	print(
			f"  ├─ target_vocabulary.json: |V|={len(target_vocab):,} labels "
			f"→ {vocab_path}"
	)
	torch.save(emb_cache, emb_path)
	print(f"  └─ emb_cache.pt        : {len(emb_cache)} entries → {emb_path}")

	total_time = time.time() - t_start
	print(f"\n{'='*80}")
	print(f"[BRIDGE] Corpus-Level Ontology Discovery COMPLETE")
	print(f"{'='*80}")
	print(f"  ├─ Total wall time          : {total_time:.1f}s ({total_time/60:.1f} min)")
	print(f"  ├─ Input receipts           : {total_receipts:,}")
	print(f"  ├─ Samples → vocabulary     : {len(all_sample_labels):,}")
	print(f"  ├─ Unique raw concepts      : {len(label_freq_dict):,}")
	print(f"  ├─ Clusters (optimal k)     : {stats['n_clusters']:,}")
	print(f"  ├─ Virtual hypernyms        : {virtual_used_count:,}")
	print(f"  ├─ |V| (target vocabulary)  : {len(target_vocab):,}")
	print(f"  ├─ canonical_map entries    : {len(final_canonical_dict):,}")
	print(f"  ├─ emb_cache entries        : {len(emb_cache):,}")
	print(f"  └─ GMM payload              : {gmm_path}")
	print(f"{'='*80}\n")

	return {
		"canonical_map_path": canonical_map_path,
		"vocab_path":         vocab_path,
		"emb_path":           emb_path,
		"gmm_path":           gmm_path,
		"n_vocab":            len(target_vocab),
		"n_canonical_map":    len(final_canonical_dict),
		"n_emb_cache":        len(emb_cache),
	}

@measure_execution_time
def main():
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

if __name__ == "__main__":
	main()