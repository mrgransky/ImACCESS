import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *

# local:
# nohup python -u stage3_4_cgd_consolidation.py -jsonl /home/farid/datasets/WW_DATASETs/SMU_1900-01-01_1970-12-31/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl -v > logs/regime_aware_consolidation.log 2>&1 &

class CGDConsolidator:
	def __init__(
			self,
			input_jsonl: str,
			alpha: float = 0.05,
			penalty_weight: float = 0.80,
			tau_density_filter: float = 0.30,
			tau_sim: float = 0.60,
			lambda_asym: float = 1.0,
			hard_conflict_w_pos: float = 0.30,
			soft_conflict_w_pos_floor: float = 0.50,
			verbose: bool = False,
	):
			"""
			Scoped to exactly ONE input_jsonl file.
			rejection_stats accumulates across all samples in this file only.
			Do NOT reuse a single CGDConsolidator instance across multiple JSONL files.
			GMM Inference Contract (Bridge Fix A + D):
					The Bridge saves a GMM payload with keys:
							gmm_model    — fitted GaussianMixture (K=3)
							scaler       — StandardScaler fitted on the same feature matrix
							class_mapping — {cluster_idx: regime_name}
							feature_dim  — 2 or 3 (chosen by Bridge based on 3D coverage fraction)
							feature_names — list of feature name strings (for logging)
							bic_scores   — {k: bic} for K∈{2,…,6}
							optimal_k_bic — int
							orphan_gap   — float (separation confidence metric)
					At inference time (consolidate_sample):
							1. Build feat_raw with shape [1, feature_dim].
								 feature_dim=2: [set_sim, orphan_ratio]
								 feature_dim=3: [set_sim, orphan_ratio, abs_asym_gap]
							2. feat_scaled = scaler.transform(feat_raw)
							3. probs = gmm_model.predict_proba(feat_scaled)[0]
							4. regime = class_mapping[argmax(probs)]
					If feature_dim=2, samples with asym_gap=None (NLI-bypassed Hard
					Conflict) are VALID inputs — do not exclude them from GMM routing.
					If feature_dim=3, samples with asym_gap=None cannot be routed by
					GMM — fall back to heuristic regime for those samples only.
			"""
			self.input_jsonl               = input_jsonl
			self.alpha                     = alpha
			self.penalty_weight            = penalty_weight
			self.tau_density_filter        = tau_density_filter
			self.tau_sim                   = tau_sim
			self.lambda_asym               = lambda_asym
			self.hard_conflict_w_pos       = hard_conflict_w_pos
			self.soft_conflict_w_pos_floor = soft_conflict_w_pos_floor
			self.verbose                   = verbose
			self.outputs_dir               = os.path.join(
				os.path.dirname(self.input_jsonl), "outputs"
			)
			# Tier-2 rejection tracking
			self.rejection_stats = {
					"no_embedding":   [],   # Concepts with no embedding
					"low_similarity": [],   # (concept, best_match, sim_score)
			}
			def _bridge_path(suffix: str) -> str:
					base = os.path.basename(self.input_jsonl).replace(".jsonl", suffix)
					return os.path.join(self.outputs_dir, base)
			canonical_map_path = _bridge_path("_canonical_map.json")
			blacklist_path     = _bridge_path("_blacklisted_concepts.json")
			freqs_path         = _bridge_path("_global_label_frequency.json")
			emb_path           = _bridge_path("_emb_cache.pt")
			target_vocab_path  = _bridge_path("_target_vocabulary.json")
			gmm_path           = _bridge_path("_conflict_gmm.pkl")
			if self.verbose:
					print(f"\n[STAGE 3 & 4] CGD Consolidator & Regime-Aware Router")
					print(f"  ├─ Canonical Map        : {canonical_map_path}")
					print(f"  ├─ Global Freqs         : {freqs_path}")
					print(f"  ├─ Embedding Cache      : {emb_path}")
					print(f"  ├─ Target Vocab         : {target_vocab_path}")
					print(f"  ├─ GMM Payload          : {gmm_path}")
					print(f"  ├─ tau_sim              : {self.tau_sim}")
					print(f"  ├─ tau_density          : {self.tau_density_filter}")
					print(f"  ├─ hard_conflict_w_pos  : {self.hard_conflict_w_pos}")
					print(f"  ├─ soft_conflict_floor  : {self.soft_conflict_w_pos_floor}")
					print(f"  └─ lambda_asym          : {self.lambda_asym}")
			# ── Canonical map ─────────────────────────────────────────────────────
			with open(canonical_map_path, 'r') as f:
					self.canonical_map: Dict[str, str] = json.load(f)
			# ── Global frequencies ────────────────────────────────────────────────
			with open(freqs_path, 'r') as f:
					self.global_freqs: Dict[str, int] = json.load(f)
			# ── Target vocabulary ─────────────────────────────────────────────────
			with open(target_vocab_path, 'r') as f:
					vocab_payload = json.load(f)
			if isinstance(vocab_payload, dict):
					self.target_vocabulary: List[str] = vocab_payload["vocabulary"]
			else:
					self.target_vocabulary = vocab_payload  # legacy bare list
			# ── GMM payload (Bridge Fix A + D) ────────────────────────────────────
			# Load scaler, feature_dim, and bic diagnostics alongside the model.
			# All four attributes are always set — None when GMM is unavailable —
			# so consolidate_sample never needs a hasattr() guard.
			if os.path.exists(gmm_path):
					try:
							gmm_payload            = joblib.load(gmm_path)
							self.gmm               = gmm_payload["gmm_model"]
							self.gmm_scaler        = gmm_payload["scaler"]          # FIX A
							self.class_mapping     = gmm_payload["class_mapping"]
							self.gmm_feature_dim   = gmm_payload.get("feature_dim", 3)  # FIX A+D
							self.gmm_feature_names = gmm_payload.get(
									"feature_names",
									["set_similarity", "orphan_ratio", "abs_asym_gap"]
									if self.gmm_feature_dim == 3
									else ["set_similarity", "orphan_ratio"],
							)
							self.regime_to_idx     = {v: k for k, v in self.class_mapping.items()}
							_optimal_k_bic         = gmm_payload.get("optimal_k_bic", "N/A")
							_orphan_gap            = gmm_payload.get("orphan_gap", float("nan"))
							_n_fit                 = gmm_payload.get("n_samples_fit", "N/A")
							_bic_scores            = gmm_payload.get("bic_scores", {})
							if self.verbose:
									print(f"\n  [GMM] Payload loaded ✓")
									print(f"    ├─ feature_dim   : {self.gmm_feature_dim}")
									print(f"    ├─ feature_names : {self.gmm_feature_names}")
									print(f"    ├─ class_mapping : {self.class_mapping}")
									print(f"    ├─ BIC-optimal K : {_optimal_k_bic}  "
												f"{'✓ matches K=3' if _optimal_k_bic == 3 else '⚠ differs from design K=3'}")
									print(f"    ├─ orphan_gap    : {_orphan_gap:.4f}  "
												f"{'✓' if isinstance(_orphan_gap, float) and _orphan_gap >= 0.05 else '⚠ weak separation'}")
									print(f"    ├─ n_samples_fit : {_n_fit}")
									if _bic_scores:
											print(f"    └─ BIC scores    : "
														f"{', '.join(f'K={k}:{v:.1f}' for k, v in sorted(_bic_scores.items()))}")
					except Exception as e:
							print(
									f"[WARN] Failed to load GMM payload from {gmm_path}: {e}. "
									f"Falling back to heuristic Stage 2 routing."
							)
							self.gmm               = None
							self.gmm_scaler        = None
							self.class_mapping     = None
							self.gmm_feature_dim   = None
							self.gmm_feature_names = None
							self.regime_to_idx     = {}
			else:
					self.gmm               = None
					self.gmm_scaler        = None
					self.class_mapping     = None
					self.gmm_feature_dim   = None
					self.gmm_feature_names = None
					self.regime_to_idx     = {}
					if self.verbose:
							print(
									f"  [GMM] pkl not found at {gmm_path}. "
									f"Falling back to heuristic Stage 2 routing."
							)
			
			# ── Blacklist ─────────────────────────────────────────────────────────
			if os.path.exists(blacklist_path):
					with open(blacklist_path, 'r') as f:
							self.blacklisted_concepts: set = set(json.load(f))
					if self.verbose:
							print(f"  [BLACKLIST] Loaded ✓ ({len(self.blacklisted_concepts):,} entries)")
			else:
					raise FileNotFoundError(
							f"[CGDConsolidator] _blacklisted_concepts.json not found at:\n"
							f"  {blacklist_path}\n"
							f"Re-run the Bridge pipeline to regenerate it.\n"
							f"The blacklist must be written by the Bridge after Step 8c vocab gate."
					)
			
			# Pre-compute normalised blacklist set ONCE — O(1) lookup per concept.
			self.blacklisted_concepts_norm: set = {
					c.lower().strip() for c in self.blacklisted_concepts
			}
			if self.verbose:
					print(
							f"  [BLACKLIST] Normalised set: {len(self.blacklisted_concepts_norm):,} entries"
					)
			
			# Cache max_freq once — avoids O(|V|) scan per sample
			self.max_freq: int = max(self.global_freqs.values()) if self.global_freqs else 1
			
			# ── Embedding cache ───────────────────────────────────────────────────
			try:
					with torch.serialization.safe_globals([np.ndarray]):
							self.emb_cache: Dict[str, np.ndarray] = torch.load(
									emb_path,
									map_location="cpu",
									weights_only=True,
							)
			except Exception as e:
					if "weights_only" in str(e):
							print(
									f"[WARN]\n{e}\n"
									f"Falling back to weights_only=False due to compatibility issues."
							)
							self.emb_cache = torch.load(
									emb_path,
									map_location="cpu",
									weights_only=False,
							)
					else:
							raise
			if self.verbose:
					print(
							f"  [EMB CACHE] {len(self.emb_cache):,} embeddings loaded | "
							f"Vocab size: {len(self.target_vocabulary):,}"
					)
	
	def _get_embedding(self, label: str) -> Optional[np.ndarray]:
			"""Fetch L2-normalized embedding from cache. Case-insensitive fallback."""
			emb = self.emb_cache.get(label)
			if emb is not None:
					return emb
			return self.emb_cache.get(label.lower().strip())
	
	def _fast_cosine_sim(self, label_a: str, label_b: str) -> float:
			"""Cosine similarity via dot product on pre-normalized vectors."""
			emb_a = self._get_embedding(label_a)
			emb_b = self._get_embedding(label_b)
			if emb_a is None or emb_b is None:
					return 0.0
			return float(np.dot(emb_a, emb_b))
	
	# Tier-0/1/2 resolution
	def _resolve_to_canonical(
		self,
		raw_concept: str,
		tau_sim_override: Optional[float] = None,
	) -> Optional[str]:
		"""
		Resolves a raw VLM concept to the discovered target vocabulary V.
		Lookup order:
				Tier 0 — Blacklist guard: drop immediately if concept is blacklisted.
									Uses pre-normalised set (self.blacklisted_concepts_norm)
									for O(1) lookup.
				Tier 1 — Direct hit in canonical_map.json (O(1) dict lookup).
				Tier 2 — Nearest-neighbour cosine search over V using emb_cache
									(O(|V|) dot products, all in-memory numpy).
				None   — Best cosine similarity < tau (concept is out-of-domain).
		Parameters
		----------
		raw_concept      : raw VLM string to resolve.
		tau_sim_override : if provided, overrides self.tau_sim for this call only.
											 Used by the robustness fallback in consolidate_sample
											 to apply a softer threshold (self.tau_sim * 0.85)
											 without mutating instance state.
		"""
		tau  = tau_sim_override if tau_sim_override is not None else self.tau_sim
		norm = raw_concept.lower().strip()
		
		# ── Tier 0: Blacklist guard ───────────────────────────────────────────
		if norm in self.blacklisted_concepts_norm:
			if self.verbose:
				print(f"  [RESOLVE] ✗ '{raw_concept}' → blacklisted → dropped")
			return None
		
		# ── Tier 1: Direct canonical map hit ─────────────────────────────────
		if raw_concept in self.canonical_map:
			return self.canonical_map[raw_concept]
		
		# Suppress Tier-2 for single short tokens — high ambiguity risk.
		if len(raw_concept.split()) == 1 and len(raw_concept) <= 5:
				if self.verbose:
						print(
								f"  [RESOLVE] ✗ '{raw_concept}' → single short token "
								f"→ Tier-2 suppressed (ambiguity risk)"
						)
				self.rejection_stats["low_similarity"].append((raw_concept, None, 0.0))
				return None
		
		# ── Tier 2: Nearest-neighbour cosine search ───────────────────────────
		emb_c = self._get_embedding(raw_concept)
		if emb_c is None:
			self.rejection_stats["no_embedding"].append(raw_concept)
			if self.verbose:
				print(f"  [RESOLVE] ✗ '{raw_concept}' → no embedding available → dropped")
			return None
		best_class: Optional[str] = None
		best_sim: float = -1.0
		
		for target_class in self.target_vocabulary:
				sim = self._fast_cosine_sim(raw_concept, target_class)
				if sim > best_sim:
						best_sim   = sim
						best_class = target_class
		
		if best_sim < tau:
			self.rejection_stats["low_similarity"].append(
				(raw_concept, best_class, best_sim)
			)
			if self.verbose:
				print(
					f"  [RESOLVE] ✗ '{raw_concept}' → best='{best_class}' "
					f"sim={best_sim:.4f} < tau={tau:.4f} → dropped"
				)
			return None
		
		if self.verbose:
			print(
				f"  [RESOLVE] ✓ '{raw_concept}' → '{best_class}' "
				f"sim={best_sim:.4f} (nearest-neighbor)"
			)
		
		return best_class
	
	# Stage 3: Micro-CGD Audit
	def audit_concept_CGD(
		self,
		concept: str,
		source_modality: str,   # "TEXT" | "VISUAL" | "FUSED"
		c_text: List[str],
		c_vis: List[str],
		regime: str,
		denser_modality: str,
		entail_V2T: float,
		entail_T2V: float,
	) -> Dict[str, float]:
		"""
		Computes continuous Coverage (C), Grounding (G), and Density (D) scores
		for a single raw concept.
		G(c)  Visual grounding:
					1.0 if concept appears verbatim in c_vis (or c_text for FUSED),
					else max cosine similarity to any visual concept.
					FUSED concepts check both modalities for verbatim match and take
					max(vis_sim, text_sim) for the cosine fallback, since a fused
					concept may be grounded in either modality.
		C(c)  Coverage / rarity: log-normalised inverse frequency.
		D(c)  Density = D_global * D_local:
					D_global : exponential frequency prior (reusability).
					D_local  : NLI-based abstraction penalty (SOFT_CONFLICT only).
										 Applied when the concept's SOURCE MODALITY is the
										 sparser one — uses modality-of-origin, not verbatim
										 membership, to avoid missing fused concepts.
		"""
		# 1. Grounding Score G(c)
		if concept in c_vis:
			g_score = 1.0
		elif source_modality == "FUSED" and concept in c_text:
			g_score = 1.0
		else:
			vis_sim = max(
				(self._fast_cosine_sim(concept, v) for v in c_vis),
				default=0.0,
			)
			text_sim = (
				max(
					(self._fast_cosine_sim(concept, t) for t in c_text),
					default=0.0,
				)
				if source_modality == "FUSED" else 0.0
			)
			g_score = max(vis_sim, text_sim)
		
		# 2. Coverage Score C(c)
		freq    = self.global_freqs.get(concept, 1) # avoid None if unavailable
		c_score = 1.0 - (math.log(1 + freq) / math.log(1 + self.max_freq))
		
		# 3. Density Score D(c) = D_global * D_local
		d_global = 1.0 - math.exp(-self.alpha * freq)
		d_local  = 1.0
		if regime == "SOFT_CONFLICT":
			evt = entail_V2T if entail_V2T is not None else 0.0
			etv = entail_T2V if entail_T2V is not None else 0.0
			if denser_modality == "VISUAL" and source_modality in ("TEXT", "FUSED"):
				d_local = 1.0 - (self.penalty_weight * evt)
			elif denser_modality == "TEXT" and source_modality in ("VISUAL", "FUSED"):
				d_local = 1.0 - (self.penalty_weight * etv)
		d_score = max(0.0, d_global * d_local)
		
		return {
			"C": round(c_score, 4),
			"G": round(g_score, 4),
			"D": round(d_score, 4),
		}
	
	# Stage 4: Regime-Aware Consolidation
	def consolidate_sample(self, receipt: Dict[str, Any], column: str) -> Dict[str, Any]:
		"""
		STAGE 4: Maps audited concepts into canonical vocabulary V 
		and applies Regime-Aware Gating to derive 
		w_pos, w_neg, positive_targets, and hard_negatives 
		for downstream BCE training.

		GMM Routing:
			- Features are scaled with self.gmm_scaler before predict_proba.
			- Feature vector dimensionality matches self.gmm_feature_dim (2 or 3).
			- For feature_dim=2, samples with asym_gap=None (NLI-bypassed Hard
				Conflict) are valid GMM inputs — they are NOT excluded from routing.
			- For feature_dim=3, samples with asym_gap=None cannot be routed —
				heuristic regime is used for those samples only.
			- A per-sample warning is emitted when GMM confidence < 0.60 (Fix C).
		
		Output:
			"heuristic_regime" — always emitted for audit trail.
			"gmm.feature_dim"  — 2 or 3, for downstream traceability.
			"gmm.regime_override" — True when GMM disagrees with Stage 2.
		"""

		sample_id        = receipt["doc_url"]
		heuristic_regime = receipt["heuristic_regime"]
		metrics          = receipt.get("metrics") or {}
		vlm_data         = receipt.get(column, {})
		evidence         = receipt.get("evidence", {})
		c_text  = vlm_data.get("text_concepts",  []) or []
		c_vis   = vlm_data.get("visual_concepts", []) or []
		c_fused = vlm_data.get("fused_concepts",  []) or []
		denser_modality = metrics.get("denser_modality", "EQUAL")
		raw_asym_gap    = metrics.get("asymmetry_gap", 0.0)
		asym_gap_abs    = abs(raw_asym_gap) if raw_asym_gap is not None else 0.0
		entail_V2T      = metrics.get("entail_V_to_T", 0.0)
		entail_T2V      = metrics.get("entail_T_to_V", 0.0)
		set_sim         = metrics.get("set_similarity", 0.0)
		orphan_ratio    = metrics.get("orphan_ratio", 0.0)

		# GMM Regime Routing:
		# The GMM guard condition no longer unconditionally requires
		# raw_asym_gap is not None. 
		# For feature_dim=2, asym_gap is not part of
		# the feature vector — NLI-bypassed Hard Conflict samples (asym_gap=None)
		# are valid 2D inputs and must not be excluded from GMM routing.
		# For feature_dim=3, asym_gap=None means the 3D vector cannot be
		# constructed — fall back to heuristic regime for that sample only.
		regime            = heuristic_regime
		gmm_confidence    = None
		gmm_probabilities = None
		_gmm_inputs_valid = (
			self.gmm is not None
			and self.gmm_scaler is not None
			and self.gmm_feature_dim is not None
			and set_sim is not None
			and orphan_ratio is not None
			and heuristic_regime != "MISSING_MODALITY"
			and (
				self.gmm_feature_dim == 2          # asym_gap not needed
				or raw_asym_gap is not None        # 3D: asym_gap required
			)
		)

		if _gmm_inputs_valid:
			# Build feature vector matching the dimensionality the GMM
			# was trained on, then apply the saved StandardScaler before
			# predict_proba. Skipping scaling causes set_similarity to dominate
			# the GMM covariance and produces unreliable regime assignments.
			if self.gmm_feature_dim == 2:
				feat_raw = np.array([[set_sim, orphan_ratio]])
			else:
				feat_raw = np.array([[set_sim, orphan_ratio, asym_gap_abs]])
			
			feat_scaled = self.gmm_scaler.transform(feat_raw)
			probs       = self.gmm.predict_proba(feat_scaled)[0] # shape [3]
			best_idx    = int(np.argmax(probs))
			regime            = self.class_mapping[best_idx]
			gmm_confidence    = float(probs[best_idx])
			gmm_probabilities = {
				self.class_mapping[k]: float(probs[k]) 
				for k in range(3)
			}
			if self.verbose:
				print(
					f"{sample_id}\n"
					f"[GMM]\n"
					f"  ├─ feat_raw: {feat_raw.tolist()}\n"
					f"  ├─ feat_scaled: {feat_scaled.round(3).tolist()}\n"
					f"  ├─ probs: {{{', '.join(f'{r}:{p:.3f}' for r, p in gmm_probabilities.items())}}}\n"
					f"  └─ regime: {regime} (conf={gmm_confidence:.4f})"
				)
			
			# Warn on low-confidence GMM assignments.
			# Mirrors the Bridge's orphan_gap < 0.05 centroid separation check
			# at the per-sample level. Useful for identifying ambiguous samples
			# in the auditable supervision matrix and for paper reporting.
			_GMM_CONF_WARN_THRESHOLD = 0.60
			if gmm_confidence < _GMM_CONF_WARN_THRESHOLD:
				if self.verbose:
					print(
						f"  [GMM][WARN] {sample_id} | low-confidence assignment: "
						f"regime={regime} conf={gmm_confidence:.4f} < "
						f"{_GMM_CONF_WARN_THRESHOLD} | "
						f"probs={gmm_probabilities}"
					)
		else:
			# Log why GMM routing was skipped for this sample.
			if self.verbose and self.gmm is not None:
				_skip_reason = (
					"MISSING_MODALITY (no conflict metrics)" if heuristic_regime == "MISSING_MODALITY"
					else f"asym_gap=None with feature_dim={self.gmm_feature_dim} (NLI-bypassed; 3D GMM cannot route)"
					if raw_asym_gap is None and self.gmm_feature_dim == 3
					else "set_sim or orphan_ratio is None"
				)
				print(
					f"  [GMM] {sample_id} | routing skipped → heuristic regime used | "
					f"reason: {_skip_reason}"
				)
		
		# ── Build concept pool with modality-of-origin tags ───────────────────
		# fused_concepts are trustworthy only for AGREEMENT and SOFT_CONFLICT.
		# HARD_CONFLICT: fused is unreliable (empty or hallucinated blend) →
		# exclude, consistent with the Bridge regime-gating logic.
		concept_pool: List[Tuple[str, str]] = []  # (concept, source_modality)
		seen: set = set()
		
		def _add(concepts: List[str], modality: str) -> None:
			for c in concepts:
				if c and c not in seen:
					concept_pool.append((c, modality))
					seen.add(c)
		
		_add(c_text, "TEXT")
		_add(c_vis,  "VISUAL")
		
		if regime in ("AGREEMENT", "SOFT_CONFLICT"):
				_add(c_fused, "FUSED")  # fused included only for trustworthy regimes
		
		if self.verbose:
			print(
				f"\n>> heuristic_regime={heuristic_regime} | "
				f"gmm_regime={regime} | "
				f"text={len(c_text)} vis={len(c_vis)} fused={len(c_fused)} | "
				f"pool={len(concept_pool)} concepts"
			)
		
		# Stage 3: CGD Audit
		# Resolve canonical ONCE per concept and audit in a single pass.
		# audited_concepts: raw_concept → {
		#     "scores": {C, G, D},
		#     "canonical": str,
		#     "source_modality": str
		# }
		audited_concepts: Dict[str, Dict[str, Any]] = {}
		for concept, source_modality in concept_pool:
			resolved = self._resolve_to_canonical(concept)
			if resolved is None:
				continue  # Out-of-domain or blacklisted — drop silently
			scores = self.audit_concept_CGD(
				concept=concept,
				source_modality=source_modality,
				c_text=c_text,
				c_vis=c_vis,
				regime=regime,
				denser_modality=denser_modality,
				entail_V2T=entail_V2T,
				entail_T2V=entail_T2V,
			)
			audited_concepts[concept] = {
				"scores":          scores,
				"canonical":       resolved,
				"source_modality": source_modality,
			}
		
		# resolved_cache records the outcome of every concept in the pool,
		# including those dropped (None). The robustness fallback must distinguish
		# between "not yet attempted" (key absent) and "attempted but dropped"
		# (key present, value None) to avoid re-resolving blacklisted concepts
		# with a softer threshold.
		resolved_cache: Dict[str, Optional[str]] = {
			c: audited_concepts[c]["canonical"] if c in audited_concepts else None
			for c, _ in concept_pool
		}
		if self.verbose:
			print(f"Audited: {len(audited_concepts)}/{len(concept_pool)} concepts resolved to vocab")
		
		# Stage 4: 
		# Regime-Aware Gated Consolidation & 
		# Gradient Scaling Coordinate Derivation
		pos_targets: set = set()
		hn_targets:  set = set()
		w_pos: float = 1.0
		w_neg: float = 0.0
		
		if regime == "AGREEMENT":
			# Both modalities agree — accept all resolved concepts as positives.
			w_pos = 1.0
			w_neg = 0.0
			for c, data in audited_concepts.items():
				pos_targets.add(data["canonical"])
		elif regime == "SOFT_CONFLICT":
				# Modalities share topic but differ in density.
				# Gate by D(c) >= tau to suppress over-abstract hypernyms.
				for c, data in audited_concepts.items():
						if data["scores"]["D"] >= self.tau_density_filter:
								pos_targets.add(data["canonical"])
				w_pos = max(
						self.soft_conflict_w_pos_floor,
						1.0 - (self.lambda_asym * asym_gap_abs),
				)
				w_neg = 0.0
		elif regime == "HARD_CONFLICT":
				# Modalities are semantically disjoint.
				# Only ORPHANED text concepts (O_text_unverified from Stage 2)
				# become hard negatives. Text concepts matched to a visual concept
				# in Stage 2 (E_strong / E_density) are NOT hard negatives — they
				# have partial visual grounding and should not be repelled.
				o_text_orphans: set = set(evidence.get("O_text_unverified", []))
				if self.verbose:
						print(
								f"  [HARD_CONFLICT] O_text_unverified={sorted(o_text_orphans)} "
								f"| evidence keys={list(evidence.keys())}"
						)
				hn_g_scores: List[float] = []
				for c, data in audited_concepts.items():
						resolved = data["canonical"]
						if data["source_modality"] == "VISUAL":
								pos_targets.add(resolved)
						elif c in o_text_orphans:
								hn_targets.add(resolved)
								hn_g_scores.append(data["scores"]["G"])
				w_pos = self.hard_conflict_w_pos
				# w_neg must be 0.0 when hn_targets is empty.
				# Without this guard, mean_hn_g=0.0 (empty list default) produces
				# w_neg=1.0 — assigning maximum repulsion weight to a non-existent
				# negative set, which is both incorrect and scientifically incoherent.
				if hn_targets:
						mean_hn_g = float(np.mean(hn_g_scores))
						w_neg     = max(0.0, 1.0 - mean_hn_g)
				else:
						w_neg = 0.0
		elif regime == "MISSING_MODALITY":
			# Only visual signal is available; accept resolved visual concepts
			# as positives with reduced confidence. No hard negatives.
			w_pos = 0.50
			w_neg = 0.0
			for c, data in audited_concepts.items():
				if data["source_modality"] == "VISUAL":
					pos_targets.add(data["canonical"])
			if self.verbose:
				print(f"  [{regime}] visual-only supervision | w_pos={w_pos}")
		else:
			# Truly unknown regime — conservative fallback.
			# This branch should never be reached in production; if it is,
			# the regime string from Stage 2 has changed without updating Stage 4.
			w_pos = 0.50
			w_neg = 0.0
			for c, data in audited_concepts.items():
				pos_targets.add(data["canonical"])
			if self.verbose:
				print(
					f"  [WARN] {sample_id} | regime='{regime}' unknown → "
					f"conservative fallback (all resolved → pos_targets, "
					f"w_pos={w_pos}, w_neg={w_neg})"
				)
		
		# Canonical collision guard:
		# A canonical cannot be both a positive target and a hard negative.
		# pos always wins — remove from hn_targets.
		canonical_collision = pos_targets & hn_targets
		if canonical_collision:
			if self.verbose:
				print(
					f"  [COLLISION] "
					f"{len(canonical_collision)} canonical(s) in both pos and hn — "
					f"removing from hn_targets: {sorted(canonical_collision)}"
				)
			hn_targets -= canonical_collision
		
		# Sanity assertions
		if regime == "HARD_CONFLICT" and len(hn_targets) == 0 and w_neg > 0:
			print(f"  <!> [BUG] HARD_CONFLICT with hn=0 but w_neg={w_neg:.4f}")
		
		for c in audited_concepts:
			if c.lower().strip() in self.blacklisted_concepts_norm:
				print(
					f"  <!> [BUG] blacklisted concept survived resolution: "
					f"'{c}' → '{audited_concepts[c]['canonical']}' "
				)
		
		# Robustness fallback:
		# If pos_targets is still empty after gating (e.g. all D(c) < tau in a
		# SOFT_CONFLICT sample with very abstract concepts), recover using raw
		# visual concepts — the most grounded signal available.
		# The density gate is intentionally bypassed here: this is a last-resort
		# path to prevent empty supervision, not a quality gate.

		# resolved_cache distinguishes two None cases:
		#   - key present, value None  → already attempted and dropped (blacklisted
		#     or below tau); do NOT retry even with a softer threshold.
		#   - key absent               → not yet attempted; safe to retry.
		if not pos_targets and c_vis:
			tau_sim_override = self.tau_sim * 0.85
			if self.verbose:
				print(
					f"  [FALLBACK] pos_targets empty after gating, recovering from "
					f"c_vis (tau_sim_override={tau_sim_override:.4f})"
				)
			
			for c in c_vis:
				if c in resolved_cache:
					# Already attempted — respect the earlier decision (may be None).
					resolved = resolved_cache[c]
				else:
					# Not in pool (e.g. c_vis was excluded for this regime) — retry.
					resolved = self._resolve_to_canonical(c, tau_sim_override=tau_sim_override)
				if resolved:
					pos_targets.add(resolved)
			
			if self.verbose:
				print(f"  >> Recovered {len(pos_targets)} pos_targets: {pos_targets} from {len(c_vis)} c_vis: {c_vis}")
		
		# flatten concepts:
		flattened_concepts = {
			concept: {
				**data["scores"],
				"canonical":       data["canonical"],
				"source_modality": data["source_modality"],
			}
			for concept, data in audited_concepts.items()
		}
		
		if self.verbose:
			print(json.dumps(flattened_concepts, indent=2))
			print(
				f">> {len(audited_concepts)} audited | "
				f"pos={len(pos_targets)} | hn={len(hn_targets)} | "
				f"w_pos={w_pos:.4f} | w_neg={w_neg:.4f}"
			)
			print("-" * 80)
		
		return {
			"doc_url":               sample_id,
			column:             vlm_data,
			"audited_concepts": flattened_concepts,
			"heuristic_regime": heuristic_regime,
			"regime":           regime,
			"positive_targets": sorted(pos_targets),
			"hard_negatives":   sorted(hn_targets),
			"w_pos":            round(w_pos, 4),
			"w_neg":            round(w_neg, 4),
			"gmm": {
				"regime_override": regime != heuristic_regime,
				"confidence":      round(gmm_confidence, 4),
				"feature_dim":     self.gmm_feature_dim,
				"probabilities":   {k: round(v, 4) for k, v in gmm_probabilities.items()},
			} if gmm_probabilities is not None else None,
		}
	
	def export_rejection_report(self, output_path: str) -> None:
			"""
			Export Tier-2 rejection statistics for vocabulary coverage analysis.
			Includes two sorted views of low-similarity rejections:
					- sorted by similarity descending (false positives just above tau_sim)
					- sorted by similarity ascending  (concepts furthest from any canonical)
			"""
			low_sim_records = [
					{"concept": c, "best_match": m, "similarity": round(s, 4)}
					for c, m, s in self.rejection_stats["low_similarity"]
			]
			report = {
					"summary": {
							"no_embedding_count":   len(self.rejection_stats["no_embedding"]),
							"low_similarity_count": len(self.rejection_stats["low_similarity"]),
							"total_rejected": (
									len(self.rejection_stats["no_embedding"]) +
									len(self.rejection_stats["low_similarity"])
							),
					},
					"no_embedding": self.rejection_stats["no_embedding"],
					"low_similarity": low_sim_records,
					"low_similarity_sorted_by_sim_desc": sorted(
							low_sim_records, key=lambda x: -x["similarity"]
					),
					"low_similarity_sorted_by_sim_asc": sorted(
							low_sim_records, key=lambda x: x["similarity"]
					),
			}
			with open(output_path, 'w', encoding='utf-8') as f:
					json.dump(report, f, indent=2, ensure_ascii=False)
			print(f"\n[REJECTION REPORT]")
			print(f"  Total rejected  : {report['summary']['total_rejected']:,}")
			print(f"  ├─ No embedding : {report['summary']['no_embedding_count']:,}")
			print(f"  └─ Low sim      : {report['summary']['low_similarity_count']:,}")
			if low_sim_records:
					sims = [r["similarity"] for r in low_sim_records]
					print(
							f"  Low-sim stats   : "
							f"min={min(sims):.4f} max={max(sims):.4f} "
							f"mean={sum(sims)/len(sims):.4f}"
					)
			print(f"  Saved to        : {output_path}")

def regime_aware_consolidation(
		input_jsonl: str,
		column: str,
		verbose: bool = False,
) -> None:
		"""
		Streams Stage 2 receipts through the CGD Consolidator (Stages 3 & 4) and
		writes the final auditable supervision matrix to .parquet / .csv / .jsonl.

		Crash-safe resume via JSONL append-and-skip.
		If a partial output JSONL already exists, already-processed sample IDs are
		loaded and skipped so a restart continues from where it left off.
		The .parquet and .csv are rebuilt from the complete JSONL at the end.

		Fix F: Verbose diagnostics now include:
				- GMM regime override rate (fraction where GMM disagreed with Stage 2)
				- Low-confidence GMM assignment count (conf < 0.60)
		These metrics are essential for the paper's methodology section.
		"""
		DATASET_DIRECTORY = os.path.dirname(input_jsonl)
		outputs_dir       = os.path.join(DATASET_DIRECTORY, "outputs")
		os.makedirs(outputs_dir, exist_ok=True)

		# Derive output paths
		stage3_path  = os.path.join(
			outputs_dir,
			os.path.basename(input_jsonl.replace(".jsonl", "_auditable_supervision_cgd.jsonl"))
		)
		parquet_path = os.path.join(
			outputs_dir,
			os.path.basename(input_jsonl.replace(".jsonl", "_auditable_supervision_matrix.parquet"))
		)
		csv_path   = parquet_path.replace(".parquet", ".csv")
		jsonl_path = parquet_path.replace(".parquet", ".jsonl")

		print(f"\n{'='*80}")
		print(f"[STAGE 3 & 4] Regime-Aware CGD Consolidation")
		print(f"{'='*80}")
		print(f"  ├─ Input JSONL  : {input_jsonl}")
		print(f"  ├─ Stage 3 out  : {stage3_path}")
		print(f"  ├─ Stage 4 JSONL: {jsonl_path}")
		print(f"  ├─ Parquet      : {parquet_path}")
		print(f"  └─ CSV          : {csv_path}")

		# ── Crash-safe resume: load already-processed IDs ─────────────────────────
		# Both Stage 4 JSONL and Stage 3 JSONL are checked so that on resume
		# neither file gets duplicate records written.
		processed_ids: set = set()
		if os.path.exists(jsonl_path):
				with open(jsonl_path, 'r', encoding="utf-8") as f_existing:
						for line in f_existing:
								line = line.strip()
								if not line:
										continue
								try:
										processed_ids.add(json.loads(line)["doc_url"])
								except Exception:
										pass
				if processed_ids:
						print(
								f"\n[STAGE 3 & 4] Resume detected: {len(processed_ids):,} samples "
								f"already processed — skipping."
						)

		stage3_processed_ids: set = set()
		if os.path.exists(stage3_path):
				with open(stage3_path, 'r', encoding="utf-8") as f_s3:
						for line in f_s3:
								line = line.strip()
								if not line:
										continue
								try:
										stage3_processed_ids.add(json.loads(line)["doc_url"])
								except Exception:
										pass

		consolidator = CGDConsolidator(input_jsonl=input_jsonl, verbose=verbose)

		print(f"\n[STAGE 3 & 4] Streaming receipts and executing stateful CGD audit...")

		rows_new: List[Dict[str, Any]] = []
		skipped, errors = 0, 0

		with (
			open(jsonl_path,  'a', encoding="utf-8") as out_f,
			open(stage3_path, 'a', encoding="utf-8") as s3_f,
			open(input_jsonl, 'r', encoding="utf-8") as in_f,
		):
			for line_no, line in enumerate(in_f, start=1):
				line = line.strip()
				if not line:
					continue
				
				try:
					receipt = json.loads(line)
				except json.JSONDecodeError as e:
					print(f"[WARN] Skipping malformed line {line_no}: {e}")
					errors += 1
					continue
				
				sample_id = receipt.get("doc_url")
				if sample_id is None:
					print(f"[WARN] Line {line_no} has no 'doc_url' field — skipping.")
					errors += 1
					continue
				if sample_id in processed_ids:
					skipped += 1
					continue

				try:
					consolidated = consolidator.consolidate_sample(receipt=receipt, column=column,)
					# Stage 3 output — guard prevents duplicate writes on crash-resume.
					if sample_id not in stage3_processed_ids:
						stage3_record = {
							"doc_url":         consolidated["doc_url"],
							"regime":     consolidated["regime"],
							"cgd_scores": consolidated["audited_concepts"],
						}
						s3_f.write(json.dumps(stage3_record, ensure_ascii=False) + "\n")
						stage3_processed_ids.add(sample_id)
					# Stage 4 output — written exactly once per sample.
					out_f.write(json.dumps(consolidated, ensure_ascii=False) + "\n")
					rows_new.append(consolidated)
				except Exception as e:
					print(f"[ERROR] {sample_id}: {e}")
					errors += 1
					continue

		# Rebuild parquet + csv from complete JSONL
		# Includes both resumed rows and newly processed rows for a consistent output.
		all_rows: List[Dict[str, Any]] = []
		with open(jsonl_path, 'r', encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					try:
						all_rows.append(json.loads(line))
					except Exception:
						pass

		df = pd.DataFrame(all_rows)
		print(f"\n[STAGE 3 & 4] DataFrame: {df.shape} | columns: {list(df.columns)}")
		print(df.info(verbose=True, memory_usage=True))

		# ── Drop samples with empty positive_targets ──────────────────────────────
		# These are genuinely out-of-vocabulary samples (all concepts failed
		# embedding lookup or similarity gating). Passing them to Stage 5 would
		# inject zero-supervision rows into BCE training.
		empty_mask = df['positive_targets'].apply(len) == 0
		empty_supervision = df[empty_mask]
		if len(empty_supervision) > 0:
				print(
						f"\n[WARN] {len(empty_supervision):,} samples have empty positive_targets "
						f"— excluded from training. IDs saved to rejection report."
				)
				empty_ids = empty_supervision["doc_url"].tolist()
				empty_ids_path = jsonl_path.replace(".jsonl", "_empty_supervision.json")
				with open(empty_ids_path, 'w', encoding='utf-8') as f_empty:
						json.dump(
								{"count": len(empty_ids), "sample_ids": empty_ids},
								f_empty, indent=2, ensure_ascii=False,
						)
				print(f"  Saved {len(empty_ids):,} empty-supervision IDs → {empty_ids_path}")
				df = df[~empty_mask].reset_index(drop=True)
				print(f"  DataFrame after filter: {df.shape}")

		# ── Serialise nested columns for Parquet ──────────────────────────────────
		# positive_targets and hard_negatives are kept as native Python lists —
		# pyarrow handles list-of-string columns natively and Stage 5 can consume
		# them without a json.loads() call.
		SKIP_SERIALISE = {"positive_targets", "hard_negatives"}
		for col in df.columns:
				if col in SKIP_SERIALISE:
						continue
				if (
						df[col].dtype == object and
						df[col].apply(lambda x: isinstance(x, (dict, list))).any()
				):
						df[col] = df[col].apply(
								lambda x: json.dumps(x, ensure_ascii=False)
								if isinstance(x, (dict, list)) else x
						)

		df.to_parquet(parquet_path, index=False, engine="pyarrow")
		df.to_csv(csv_path, index=False)

		# ── Export Tier-2 rejection report ────────────────────────────────────────
		rejection_report_path = jsonl_path.replace(".jsonl", "_tier2_rejections.json")
		consolidator.export_rejection_report(rejection_report_path)

		print(
				f"\n[STAGE 3 & 4] COMPLETE | "
				f"Total={len(all_rows):,} | New={len(rows_new):,} | "
				f"Resumed={skipped:,} | Errors={errors:,}"
		)
		print(f"  ├─ Parquet  : {parquet_path}")
		print(f"  ├─ CSV      : {csv_path}")
		print(f"  ├─ Stage 3  : {stage3_path}")
		print(f"  └─ JSONL    : {jsonl_path}")

		# ── Verbose diagnostics (Fix F) ───────────────────────────────────────────
		if verbose:
				print(f"\n{'─'*80}")
				print(f"[DIAGNOSTICS] Regime distribution (GMM-routed):")
				print(df['regime'].value_counts().to_string())

				if 'heuristic_regime' in df.columns:
						print(f"\n[DIAGNOSTICS] Heuristic regime distribution (Stage 2 raw):")
						print(df['heuristic_regime'].value_counts().to_string())

				print(f"\n[DIAGNOSTICS] Mean w_pos per regime:")
				print(df.groupby('regime')['w_pos'].mean().round(4).to_string())

				print(f"\n[DIAGNOSTICS] Mean w_neg per regime:")
				print(df.groupby('regime')['w_neg'].mean().round(4).to_string())

				print(f"\n[DIAGNOSTICS] Mean |pos_targets| per regime:")
				print(
						df.groupby('regime')['positive_targets']
						.apply(lambda x: x.apply(len).mean())
						.round(2)
						.to_string()
				)

				print(f"\n[DIAGNOSTICS] Mean |hard_negatives| per regime:")
				print(
						df.groupby('regime')['hard_negatives']
						.apply(lambda x: x.apply(len).mean())
						.round(2)
						.to_string()
				)

				# Fix F: GMM override rate and low-confidence count.
				# These are the two key metrics for the paper's methodology section:
				#   - Override rate: how often does the GMM disagree with Stage 2?
				#     A high rate suggests Stage 2 thresholds need recalibration.
				#   - Low-confidence rate: how often is the GMM uncertain?
				#     Correlates with the Bridge's orphan_gap warning.
				if 'gmm' in df.columns:
						print(f"\n[DIAGNOSTICS] GMM routing statistics (Fix F):")

						def _parse_gmm(x):
								if isinstance(x, str):
										try:
												return json.loads(x)
										except Exception:
												return None
								return x  # already dict or None

						gmm_col = df['gmm'].apply(_parse_gmm)

						# Override rate
						override_mask = gmm_col.apply(
								lambda x: bool(x.get("regime_override", False))
								if isinstance(x, dict) else False
						)
						n_overrides  = int(override_mask.sum())
						n_gmm_routed = int(gmm_col.apply(lambda x: isinstance(x, dict)).sum())
						n_total      = len(df)
						print(
								f"  ├─ Samples with GMM routing    : {n_gmm_routed:,} / {n_total:,} "
								f"({n_gmm_routed / max(n_total, 1) * 100:.1f}%)"
						)
						print(
								f"  ├─ GMM regime overrides        : {n_overrides:,} / {n_gmm_routed:,} "
								f"({n_overrides / max(n_gmm_routed, 1) * 100:.1f}% of GMM-routed samples)"
						)

						# Override breakdown: which heuristic regime was overridden to which GMM regime
						if 'heuristic_regime' in df.columns:
								override_df = df[override_mask][['heuristic_regime', 'regime']]
								if len(override_df) > 0:
										print(f"  ├─ Override breakdown (heuristic → GMM):")
										for (h_reg, g_reg), cnt in (
												override_df
												.groupby(['heuristic_regime', 'regime'])
												.size()
												.sort_values(ascending=False)
												.items()
										):
												print(f"  │    {h_reg:<20} → {g_reg:<20} : {cnt:,}")

						# Low-confidence GMM assignments
						_GMM_CONF_WARN_THRESHOLD = 0.60
						low_conf_mask = gmm_col.apply(
								lambda x: (
										isinstance(x, dict)
										and x.get("confidence") is not None
										and x["confidence"] < _GMM_CONF_WARN_THRESHOLD
								)
						)
						n_low_conf = int(low_conf_mask.sum())
						print(
								f"  ├─ Low-confidence GMM (< {_GMM_CONF_WARN_THRESHOLD}) : "
								f"{n_low_conf:,} / {n_gmm_routed:,} "
								f"({n_low_conf / max(n_gmm_routed, 1) * 100:.1f}% of GMM-routed samples)"
						)

						# Mean confidence per GMM-routed regime
						conf_by_regime = (
								df[gmm_col.apply(lambda x: isinstance(x, dict))]
								.copy()
						)
						conf_by_regime['_gmm_conf'] = gmm_col[
								gmm_col.apply(lambda x: isinstance(x, dict))
						].apply(lambda x: x.get("confidence"))
						print(f"  └─ Mean GMM confidence per regime:")
						for reg, mean_conf in (
								conf_by_regime.groupby('regime')['_gmm_conf']
								.mean()
								.round(4)
								.items()
						):
								print(f"       {reg:<20} : {mean_conf:.4f}")
				else:
						print(f"\n[DIAGNOSTICS] GMM column not found in DataFrame — "
									f"GMM routing statistics unavailable.")

				print(f"\n[DIAGNOSTICS] Numeric summary:")
				print(df.describe())
				print(df)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Stage 3 & 4: Stateful CGD Audit & Regime-Aware Consolidation")
	parser.add_argument("--jsonl_file", "-jsonl", type=str, required=True, help="Path to Stage 2 modality conflict audit JSONL file (*_modality_conflict_audit.jsonl)")
	parser.add_argument("--column", "-col", type=str, default="mlm_cot_raw", help="Column to use for canonical analysis",)
	parser.add_argument("--verbose", "-v", action='store_true', help="Print verbose diagnostics and per-regime statistics")
	args = parser.parse_args()
	print(args)

	if "_modality_conflict_audit.jsonl" not in args.jsonl_file:
		raise ValueError(
			f"Input JSONL must be a Stage 2 modality conflict audit file. "
			f"Got: {args.jsonl_file}"
		)

	regime_aware_consolidation(
		input_jsonl=args.jsonl_file,
		column=args.column,
		verbose=args.verbose
	)

if __name__ == "__main__":
	main()
