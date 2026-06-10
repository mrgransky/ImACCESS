import json
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
# nohup python -u stage3_4_cgd_consolidation.py -jsonl /home/farid/datasets/WW_DATASETs/WWII_1939-09-01_1945-09-02/metadata_multi_label_mlm_cot_modality_conflict_audit.jsonl -v > logs/regime_aware_consolidation.log 2>&1 &

class CGDConsolidator:
	def __init__(
		self,
		input_jsonl: str,
		alpha: float = 0.05,                    # Decay factor for global density D_global
		penalty_weight: float = 0.80,           # Aggressiveness of hypernym penalty in Soft Conflicts
		tau_density_filter: float = 0.30,       # Minimum D(c) score for Soft Conflict inclusion
		tau_sim: float = 0.60,                  # Minimum similarity score for Tier-2 NN resolution
		lambda_asym: float = 1.0,               # Multiplier for Soft Conflict w_pos discount
		hard_conflict_w_pos: float = 0.30,      # Ablation-exposed constant for HARD_CONFLICT
		soft_conflict_w_pos_floor: float = 0.50,# [FIX 6] Exposed floor for SOFT_CONFLICT w_pos
		verbose: bool = False,
	):
		"""
		Scoped to exactly ONE input_jsonl file.
		rejection_stats accumulates across all samples in this file only.
		Do NOT reuse a single CGDConsolidator instance across multiple JSONL files.
		"""
		self.input_jsonl                = input_jsonl
		self.alpha                      = alpha
		self.penalty_weight             = penalty_weight
		self.tau_density_filter         = tau_density_filter
		self.tau_sim                    = tau_sim
		self.lambda_asym                = lambda_asym
		self.hard_conflict_w_pos        = hard_conflict_w_pos
		self.soft_conflict_w_pos_floor  = soft_conflict_w_pos_floor  # [FIX 6]
		self.verbose                    = verbose
		self.outputs_dir                = os.path.join(os.path.dirname(self.input_jsonl), "outputs")

		# Tier-2 rejection tracking
		self.rejection_stats = {
			"no_embedding":   [],   # Concepts with no embedding
			"low_similarity": [],   # (concept, best_match, sim_score)
		}

		# ── Derive Bridge artifact paths from the input JSONL stem ──────────────
		def _bridge_path(suffix: str) -> str:
			base = os.path.basename(self.input_jsonl).replace(".jsonl", suffix)
			return os.path.join(self.outputs_dir, base)

		canonical_map_path  = _bridge_path("_canonical_map.json")
		blacklist_path      = _bridge_path("_blacklisted_concepts.json")
		freqs_path          = _bridge_path("_global_label_frequency.json")
		emb_path            = _bridge_path("_emb_cache.pt")
		target_vocab_path   = _bridge_path("_target_vocabulary.json")

		if self.verbose:
			print(f"\n[STAGE 3 & 4] CGD Consolidator & Regime-Aware Router")
			print(f"  ├─ Canonical Map        : {canonical_map_path}")
			print(f"  ├─ Global Freqs         : {freqs_path}")
			print(f"  ├─ Embedding Cache      : {emb_path}")
			print(f"  ├─ Target Vocab         : {target_vocab_path}")
			print(f"  ├─ tau_sim              : {self.tau_sim}")
			print(f"  ├─ tau_density          : {self.tau_density_filter}")
			print(f"  ├─ hard_conflict_w_pos  : {self.hard_conflict_w_pos}")
			print(f"  ├─ soft_conflict_floor  : {self.soft_conflict_w_pos_floor}")
			print(f"  └─ lambda_asym          : {self.lambda_asym}")

		with open(canonical_map_path, 'r') as f:
			self.canonical_map: Dict[str, str] = json.load(f)

		with open(freqs_path, 'r') as f:
			self.global_freqs: Dict[str, int] = json.load(f)

		# Support both the new metadata-wrapped format (post-Bridge)
		# and the legacy bare-list format so the code is backward-compatible.
		with open(target_vocab_path, 'r') as f:
			vocab_payload = json.load(f)
		if isinstance(vocab_payload, dict):
			self.target_vocabulary: List[str] = vocab_payload["vocabulary"]
		else:
			self.target_vocabulary = vocab_payload  # legacy bare list

		if os.path.exists(blacklist_path):
			with open(blacklist_path, 'r') as f:
				self.blacklisted_concepts: set = set(json.load(f))
		else:
			raise FileNotFoundError(
					f"[CGDConsolidator] _blacklisted_concepts.json not found at:\n"
					f"  {blacklist_path}\n"
					f"Re-run the Bridge pipeline to regenerate it.\n"
					f"The blacklist must be written by the Bridge after Step 8c vocab gate."
				)
		
		# Pre-compute normalised blacklist set ONCE — O(1) lookup per concept.
		# The original code rebuilt this set inside _resolve_to_canonical() on every
		# call (~770K times for 110K samples × 7 concepts), causing severe perf regression.
		self.blacklisted_concepts_norm: set = {
			c.lower().strip() for c in self.blacklisted_concepts
		}
		if self.verbose:
			print(f"  [DEBUG] blacklisted_concepts_norm sample:\n{sorted(list(self.blacklisted_concepts_norm))}")

		# Cache max_freq once — avoids O(|V|) scan per sample
		self.max_freq: int = max(self.global_freqs.values()) if self.global_freqs else 1

		# Load pre-computed L2-normalized embedding cache {label_str: np.ndarray}
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
				f"  └─ [{self.__class__.__name__}] "
				f"{len(self.emb_cache):,} embeddings | "
				f"Vocab size: {len(self.target_vocabulary):,}"
			)

	def _get_embedding(self, label: str) -> Optional[np.ndarray]:
		"""Fetch L2-normalized embedding from cache. Case-insensitive lookup."""
		emb = self.emb_cache.get(label)
		if emb is not None:
			return emb
		# Fallback: try lowercased form (VLM may return title-case)
		return self.emb_cache.get(label.lower().strip())

	def _fast_cosine_sim(self, label_a: str, label_b: str) -> float:
		"""Cosine similarity via dot product on pre-normalized vectors."""
		emb_a = self._get_embedding(label_a)
		emb_b = self._get_embedding(label_b)
		if emb_a is None or emb_b is None:
			return 0.0
		return float(np.dot(emb_a, emb_b))

	def _resolve_to_canonical(
		self,
		raw_concept: str,
		tau_sim_override: Optional[float] = None,
	) -> Optional[str]:
		"""
		Resolves a raw VLM concept to the discovered target vocabulary V.

		Lookup order:
			Tier 0 — Blacklist guard: drop immediately if concept is blacklisted.
			          Uses pre-normalised set (self.blacklisted_concepts_norm) for
			          O(1) lookup. [FIX 1]
			Tier 1 — Direct hit in canonical_map.json  (O(1) dict lookup).
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
		# [FIX 1] Hard drop before any embedding lookup.
		# self.blacklisted_concepts_norm is pre-computed in __init__ — O(1) here.
		if norm in self.blacklisted_concepts_norm:
			if self.verbose:
				print(f"  [RESOLVE] ✗ '{raw_concept}' → blacklisted → dropped")
			return None

		# ── Tier 1: Direct canonical map hit ─────────────────────────────────
		if raw_concept in self.canonical_map:
			return self.canonical_map[raw_concept]

		if len(raw_concept.split()) == 1 and len(raw_concept) <= 5:
				if self.verbose:
						print(
								f"  [RESOLVE] ✗ '{raw_concept}' → single short token "
								f"→ Tier-2 suppressed (ambiguity risk)"
						)
				self.rejection_stats["low_similarity"].append(
						(raw_concept, None, 0.0)
				)
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
				best_sim  = sim
				best_class = target_class

		if best_sim < tau:
			self.rejection_stats["low_similarity"].append((raw_concept, best_class, best_sim))
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
		source_modality: str,       # "TEXT" | "VISUAL" | "FUSED"
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
		      [FIX 6] FUSED concepts check both modalities for verbatim match
		      and take max(vis_sim, text_sim) for the cosine fallback, since a
		      fused concept may be grounded in either modality.

		C(c)  Coverage / rarity: log-normalised inverse frequency.

		D(c)  Density = D_global * D_local:
		      D_global : exponential frequency prior (reusability).
		      D_local  : NLI-based abstraction penalty (SOFT_CONFLICT only).
		                 Applied when the concept's SOURCE MODALITY is the
		                 sparser one — uses modality-of-origin, not verbatim
		                 membership, to avoid missing fused concepts.

		Parameters
		----------
		source_modality : the modality that produced this concept ("TEXT",
		                  "VISUAL", or "FUSED"). Determines which NLI direction
		                  is used for the D_local penalty in SOFT_CONFLICT, and
		                  which modalities are checked for G(c) verbatim match.
		"""
		# ── 1. Grounding Score G(c) ───────────────────────────────────────────
		# [FIX 6] FUSED concepts are grounded in either modality.
		# Verbatim check: c_vis first, then c_text for FUSED.
		# Cosine fallback: max over c_vis always; additionally max over c_text
		# for FUSED so that text-grounded fused concepts are not penalised.
		if concept in c_vis:
			g_score = 1.0
		elif source_modality == "FUSED" and concept in c_text:
			g_score = 1.0
		else:
			vis_sim  = max(
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

		# ── 2. Coverage Score C(c) ────────────────────────────────────────────
		freq    = self.global_freqs.get(concept, 1)
		c_score = 1.0 - (math.log(1 + freq) / math.log(1 + self.max_freq))

		# ── 3. Density Score D(c) = D_global * D_local ───────────────────────
		d_global = 1.0 - math.exp(-self.alpha * freq)
		d_local  = 1.0

		if regime == "SOFT_CONFLICT":
			evt = entail_V2T if entail_V2T is not None else 0.0
			etv = entail_T2V if entail_T2V is not None else 0.0
			# Penalty is based on source_modality (modality-of-origin), not
			# verbatim membership in c_text / c_vis. This correctly penalises
			# fused concepts that originate from the sparser modality even when
			# they are not verbatim in c_text/c_vis.
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
		STAGE 4: Maps audited concepts into canonical vocabulary V, applies
		Regime-Aware Gating, and derives w_pos, w_neg, positive_targets,
		and hard_negatives for downstream BCE training.
		"""
		sample_id = receipt["id"]
		regime    = receipt["regime"]
		metrics   = receipt.get("metrics") or {}
		vlm_data  = receipt.get(column, {})
		evidence  = receipt.get("evidence", {})

		c_text  = vlm_data.get("text_concepts",  []) or []
		c_vis   = vlm_data.get("visual_concepts", []) or []
		c_fused = vlm_data.get("fused_concepts",  []) or []

		denser_modality = metrics.get("denser_modality", "EQUAL")
		raw_asym_gap    = metrics.get("asymmetry_gap", 0.0)
		asym_gap_abs    = abs(raw_asym_gap) if raw_asym_gap is not None else 0.0
		entail_V2T      = metrics.get("entail_V_to_T", 0.0)
		entail_T2V      = metrics.get("entail_T_to_V", 0.0)

		# ── [FIX 2] Build concept pool with modality-of-origin tags ──────────
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
				f"\n>> [{sample_id}] regime={regime} | "
				f"text={len(c_text)} vis={len(c_vis)} fused={len(c_fused)} | "
				f"pool={len(concept_pool)} concepts"
			)

		# ── Stage 3: CGD Audit ────────────────────────────────────────────────
		# Resolve canonical ONCE per concept and audit in a single pass.
		# audited_concepts: raw_concept → {"scores": {C,G,D}, "canonical": str,
		#                                  "source_modality": str}
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

		# [FIX 4] resolved_cache records the outcome of every concept in the pool,
		# including those dropped (None). The robustness fallback must distinguish
		# between "not yet attempted" (key absent) and "attempted but dropped"
		# (key present, value None) to avoid re-resolving blacklisted concepts
		# with a softer threshold.
		resolved_cache: Dict[str, Optional[str]] = {
			c: audited_concepts[c]["canonical"] if c in audited_concepts else None
			for c, _ in concept_pool
		}

		if self.verbose:
			print(
				f"   Audited: {len(audited_concepts)}/{len(concept_pool)} concepts "
				f"resolved to vocabulary"
			)

		# ── Stage 4: Regime-Aware Gated Consolidation ────────────────────────
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
			# [FIX 6] w_pos floor is now an exposed hyperparameter for ablation.
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
			# [FIX 2] w_neg must be 0.0 when hn_targets is empty.
			# Without this guard, mean_hn_g=0.0 (empty list default) produces
			# w_neg=1.0 — assigning maximum repulsion weight to a non-existent
			# negative set, which is both incorrect and scientifically incoherent.
			if hn_targets:
				mean_hn_g = float(np.mean(hn_g_scores))
				w_neg     = max(0.0, 1.0 - mean_hn_g)
			else:
				w_neg = 0.0

		elif regime == "MISSING_MODALITY":
			# [FIX 3] Explicit branch — previously fell through to the unknown-regime
			# else clause, triggering a spurious [WARN] log on every such sample.
			# Only visual signal is available; accept resolved visual concepts as
			# positives with reduced confidence. No hard negatives.
			w_pos = 0.50
			w_neg = 0.0
			for c, data in audited_concepts.items():
				if data["source_modality"] == "VISUAL":
					pos_targets.add(data["canonical"])
			if self.verbose:
				print(
					f"  [MISSING_MODALITY] {sample_id} | "
					f"visual-only supervision | w_pos={w_pos}"
				)

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

		# In consolidate_sample, after the regime block, before the sanity assertions:
		canonical_collision = pos_targets & hn_targets
		if canonical_collision:
				if self.verbose:
						print(
								f"  [COLLISION] {sample_id} | "
								f"{len(canonical_collision)} canonical(s) in both pos and hn — "
								f"removing from hn_targets: {sorted(canonical_collision)}"
						)
				hn_targets -= canonical_collision  # pos always wins

		# ── Sanity assertions (active in all runs, not just verbose) ─────────
		# [FIX 2] Catch any future regression where w_neg > 0 with no negatives.
		if regime == "HARD_CONFLICT" and len(hn_targets) == 0 and w_neg > 0:
			print(
				f"[BUG] HARD_CONFLICT with hn=0 but w_neg={w_neg:.4f} "
				f"for {sample_id}"
			)

		# [FIX 7] Blacklist survival check uses self.blacklisted_concepts_norm
		# instead of a hardcoded geo_like set — self-maintaining as blacklist grows.
		for c in audited_concepts:
			if c.lower().strip() in self.blacklisted_concepts_norm:
				print(
					f"[BUG] blacklisted concept survived resolution: "
					f"'{c}' → '{audited_concepts[c]['canonical']}' "
					f"for {sample_id}"
				)

		# ── Robustness fallback ───────────────────────────────────────────────
		# If pos_targets is still empty after gating (e.g. all D(c) < tau in a
		# SOFT_CONFLICT sample with very abstract concepts), recover using raw
		# visual concepts — the most grounded signal available.
		# The density gate is intentionally bypassed here: this is a last-resort
		# path to prevent empty supervision, not a quality gate.
		#
		# [FIX 4] resolved_cache distinguishes two None cases:
		#   - key present, value None  → already attempted and dropped (blacklisted
		#     or below tau); do NOT retry even with a softer threshold.
		#   - key absent               → not yet attempted; safe to retry.
		# The original code used resolved_cache.get(c) for both cases, allowing
		# blacklisted concepts to be re-resolved with tau_sim * 0.85.
		if not pos_targets and c_vis:
			if self.verbose:
				print(
					f"  [FALLBACK] {sample_id} | pos_targets empty after gating — "
					f"recovering from c_vis ({len(c_vis)} concepts) with "
					f"tau_sim_override={self.tau_sim * 0.85:.4f}"
				)
			for c in c_vis:
				if c in resolved_cache:
					# Already attempted — respect the earlier decision (may be None).
					resolved = resolved_cache[c]
				else:
					# Not in pool (e.g. c_vis was excluded for this regime) — retry.
					resolved = self._resolve_to_canonical(
						c, tau_sim_override=self.tau_sim * 0.85
					)
				if resolved:
					pos_targets.add(resolved)
			if self.verbose:
				print(
					f"  [FALLBACK] Recovered {len(pos_targets)} pos_targets "
					f"from c_vis fallback."
				)

		# ── Flatten audited_concepts for output ───────────────────────────────
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
			print("-" * 120)

		return {
			"id":               sample_id,
			column:             vlm_data,
			"audited_concepts": flattened_concepts,
			"regime":           regime,
			"positive_targets": sorted(pos_targets),
			"hard_negatives":   sorted(hn_targets),
			"w_pos":            w_pos,
			"w_neg":            w_neg,
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
			# Insertion-order list (for reproducibility)
			"low_similarity": low_sim_records,
			# Sorted views for vocabulary coverage analysis
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

def regime_aware_consolidation(input_jsonl: str, column: str, verbose: bool = False) -> None:
	"""
	Streams Stage 2 receipts through the CGD Consolidator (Stages 3 & 4) and
	writes the final auditable supervision matrix to .parquet / .csv / .jsonl.

	Crash-safe resume via JSONL append-and-skip.
	If a partial output JSONL already exists, already-processed sample IDs are
	loaded and skipped so a restart continues from where it left off.
	The .parquet and .csv are rebuilt from the complete JSONL at the end.
	"""
	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
	outputs_dir       = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	# Derive output paths
	parquet_path = os.path.join(
		outputs_dir,
		os.path.basename(input_jsonl.replace(".jsonl", "_auditable_matrix.parquet"))
	)
	csv_path   = parquet_path.replace(".parquet", ".csv")
	jsonl_path = parquet_path.replace(".parquet", ".jsonl")

	# ── Load already-processed IDs for crash-safe resume ─────────────────────
	processed_ids: set = set()
	if os.path.exists(jsonl_path):
		with open(jsonl_path, 'r', encoding="utf-8") as f_existing:
			for line in f_existing:
				line = line.strip()
				if not line:
					continue
				try:
					processed_ids.add(json.loads(line)["id"])
				except Exception:
					pass
		if processed_ids:
			print(
				f"[STAGE 3 & 4] Resume detected: {len(processed_ids):,} samples "
				f"already processed — skipping."
			)

	# One CGDConsolidator instance is scoped to one input_jsonl.
	consolidator = CGDConsolidator(input_jsonl=input_jsonl, verbose=verbose)

	print(f"\n[STAGE 3 & 4] Streaming receipts and executing stateful CGD audit...")

	rows_new: List[Dict[str, Any]] = []
	skipped, errors = 0, 0

	# Open output JSONL in append mode — safe for resume
	with (
		open(jsonl_path,   'a', encoding="utf-8") as out_f,
		open(input_jsonl,  'r', encoding="utf-8") as in_f,
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

			sample_id = receipt.get("id")
			if sample_id is None:
				print(f"[WARN] Line {line_no} has no 'id' field — skipping.")
				errors += 1
				continue

			if sample_id in processed_ids:
				skipped += 1
				continue

			try:
				consolidated = consolidator.consolidate_sample(
					receipt=receipt, 
					column=column
				)
			except Exception as e:
				print(f"[ERROR] {sample_id}: {e}")
				errors += 1
				continue

			out_f.write(json.dumps(consolidated, ensure_ascii=False) + "\n")
			rows_new.append(consolidated)

	# ── Rebuild .parquet and .csv from the complete JSONL ────────────────────
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

	# ── [POST-FILTER] Drop samples with empty positive_targets ───────────────
	# These are genuinely out-of-vocabulary samples (all concepts failed
	# embedding lookup or similarity gating). Passing them to Stage 5 would
	# inject zero-supervision rows into BCE training.
	empty_mask = df['positive_targets'].apply(len) == 0
	empty_supervision = df[empty_mask]
	if len(empty_supervision) > 0:
			print(
					f"\n[WARN] {len(empty_supervision)} samples have empty positive_targets "
					f"— excluded from training. IDs saved to rejection report."
			)
			empty_ids = empty_supervision['id'].tolist()
			print(f"  Affected IDs: {empty_ids}")
			# Persist excluded IDs alongside the Tier-2 rejection report
			empty_ids_path = jsonl_path.replace(".jsonl", "_empty_supervision.json")
			with open(empty_ids_path, 'w', encoding='utf-8') as f_empty:
					json.dump(
							{"count": len(empty_ids), "sample_ids": empty_ids},
							f_empty, indent=2, ensure_ascii=False,
					)
			df = df[~empty_mask].reset_index(drop=True)
			print(f"  DataFrame after filter: {df.shape}")


	# [FIX 9] Serialise nested dict/list columns to JSON strings for Parquet
	# compatibility, but SKIP positive_targets and hard_negatives so Stage 5
	# can consume them as native Python lists without a json.loads() call.
	# pyarrow handles list-of-string columns natively.
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

	# Export Tier-2 rejection report
	rejection_report_path = jsonl_path.replace(".jsonl", "_tier2_rejections.json")
	consolidator.export_rejection_report(rejection_report_path)

	print(
		f"\n[STAGE 3 & 4] COMPLETE | "
		f"Total={len(all_rows):,} | New={len(rows_new):,} | "
		f"Resumed={skipped:,} | Errors={errors:,}"
	)
	print(f"  ├─ Parquet : {parquet_path}")
	print(f"  ├─ CSV     : {csv_path}")
	print(f"  └─ JSONL   : {jsonl_path}")

	if verbose:
		print(df.info(verbose=True, memory_usage=True))
		print("\n[DIAGNOSTICS] Regime distribution:")
		print(df['regime'].value_counts())
		print("\n[DIAGNOSTICS] Mean w_pos per Regime:")
		print(df.groupby('regime')['w_pos'].mean().round(4))
		print("\n[DIAGNOSTICS] Mean w_neg per Regime:")
		print(df.groupby('regime')['w_neg'].mean().round(4))
		print("\n[DIAGNOSTICS] Mean |pos_targets| per Regime:")
		print(
			df.groupby('regime')['positive_targets']
			.apply(lambda x: x.apply(len).mean())
			.round(2)
		)
		print("\n[DIAGNOSTICS] Mean |hard_negatives| per Regime:")
		print(
			df.groupby('regime')['hard_negatives']
			.apply(lambda x: x.apply(len).mean())
			.round(2)
		)
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
