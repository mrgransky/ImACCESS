import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)
MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *

class CGDConsolidator:
	def __init__(
		self,
		input_jsonl: str,
		alpha: float = 0.05,            # Decay factor for global density D_global
		penalty_weight: float = 0.80,   # Aggressiveness of hypernym penalty in Soft Conflicts
		tau_density_filter: float = 0.30, # Minimum D(c) score for Soft Conflict inclusion
		lambda_asym: float = 1.0,       # Multiplier for Soft Conflict w_pos discount
		verbose: bool = False,
	):
		self.input_jsonl = input_jsonl
		self.alpha = alpha
		self.penalty_weight = penalty_weight
		self.tau_density_filter = tau_density_filter
		self.lambda_asym = lambda_asym
		self.verbose = verbose
		self.outputs_dir = os.path.join(os.path.dirname(self.input_jsonl), "outputs")

		# ── Derive Bridge artifact paths from the input JSONL stem ──────────────
		def _bridge_path(suffix: str) -> str:
			base = os.path.basename(self.input_jsonl).replace(".jsonl", suffix)
			return os.path.join(self.outputs_dir, base)

		map_path          = _bridge_path("_canonical_map.json")
		freqs_path        = _bridge_path("_global_label_frequency.json")
		emb_path          = _bridge_path("_emb_cache.pt")
		target_vocab_path = _bridge_path("_target_vocabulary.json")

		if self.verbose:
			print(f"\n[STAGE 3 & 4] CGD Consolidator & Regime-Aware Router")
			print(f"  ├─ Canonical Map   : {map_path}")
			print(f"  ├─ Global Freqs    : {freqs_path}")
			print(f"  ├─ Embedding Cache : {emb_path}")
			print(f"  ├─ Target Vocab    : {target_vocab_path}")

		with open(map_path, 'r') as f:
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

		# Cache max_freq once — avoids O(|V|) scan per sample
		self.max_freq: int = max(self.global_freqs.values()) if self.global_freqs else 1

		# Load pre-computed L2-normalized embedding cache {label_str: np.ndarray}
		try:
			with torch.serialization.safe_globals([np.ndarray]):
				self.emb_cache: Dict[str, np.ndarray] = torch.load(
					emb_path, 
					map_location="cpu", 
					weights_only=True
				)
		except Exception as e:
			if "weights_only" in str(e):
				print(f"[WARN]\n{e}\nFalling back to weights_only=False due to compatibility issues.")
				self.emb_cache = torch.load(
					emb_path, 
					map_location="cpu", 
					weights_only=False
				)
			else:
				raise

		if self.verbose:
			print(f"  └─ [{self.__class__.__name__}] {len(self.emb_cache)} embeddings | Vocab size: {len(self.target_vocabulary)}")

	def _get_embedding(self, label: str) -> Optional[np.ndarray]:
		"""Safely fetch L2-normalized embedding from cache."""
		return self.emb_cache.get(label, None)

	def _fast_cosine_sim(self, label_a: str, label_b: str) -> float:
		"""Instant cosine similarity via dot product on pre-normalized vectors."""
		emb_a = self._get_embedding(label_a)
		emb_b = self._get_embedding(label_b)
		if emb_a is None or emb_b is None:
			return 0.0
		return float(np.dot(emb_a, emb_b))

	def _resolve_to_canonical(self, raw_concept: str) -> Optional[str]:
		"""
		Resolves a raw VLM concept to the discovered target vocabulary V.

		Lookup order:
		  1. Direct hit in canonical_map.json  (O(1) dict lookup)
		  2. Fallback: nearest-neighbour cosine search over V using emb_cache
		     (O(|V|) dot products, all in-memory numpy — fast enough for |V|~1K)
		  3. Returns None if best cosine similarity < 0.50 (concept is out-of-domain)
		"""
		# Tier 1: direct map hit
		if raw_concept in self.canonical_map:
			return self.canonical_map[raw_concept]

		# Tier 2: fallback nearest-neighbour search
		emb_c = self._get_embedding(raw_concept)
		if emb_c is None:
			return None

		best_class: Optional[str] = None
		best_sim: float = -1.0
		for target_class in self.target_vocabulary:
			sim = self._fast_cosine_sim(raw_concept, target_class)
			if sim > best_sim:
				best_sim = sim
				best_class = target_class

		return best_class if best_sim >= 0.50 else None

	# Stage 3: Micro-CGD Audit
	def audit_concept_CGD(
		self,
		concept: str,
		c_text: List[str],
		c_vis: List[str],
		regime: str,
		denser_modality: str,
		entail_V_to_T: float,
		entail_T_to_V: float,
	) -> Dict[str, float]:
		"""
		Computes continuous Coverage (C), Grounding (G), and Density (D) scores
		for a single raw concept.

		  G(c) — Visual grounding: 1.0 if concept appears verbatim in c_vis,
		          else max cosine similarity to any visual concept.
		  C(c) — Coverage / rarity: log-normalised inverse frequency.
		  D(c) — Density = D_global * D_local:
		            D_global: exponential frequency prior (reusability).
		            D_local : NLI-based abstraction penalty (SOFT_CONFLICT only).
		"""
		# 1. Grounding Score G(c)
		if concept in c_vis:
			g_score = 1.0
		else:
			g_score = max((self._fast_cosine_sim(concept, v) for v in c_vis), default=0.0)

		# 2. Coverage Score C(c)
		freq = self.global_freqs.get(concept, 1)
		c_score = 1.0 - (math.log(1 + freq) / math.log(1 + self.max_freq))

		# 3. Density Score D(c) = D_global * D_local
		d_global = 1.0 - math.exp(-self.alpha * freq)

		d_local = 1.0
		if regime == "SOFT_CONFLICT":
			# Safe unpack: Stage 2 short-circuit receipts default these to 0.0
			evt = entail_V_to_T if entail_V_to_T is not None else 0.0
			etv = entail_T_to_V if entail_T_to_V is not None else 0.0
			# Penalise the sparser modality's concepts proportionally to how
			# strongly the denser modality entails them (hypernym penalty).
			if denser_modality == "VISUAL" and concept in c_text:
				d_local = 1.0 - (self.penalty_weight * evt)
			elif denser_modality == "TEXT" and concept in c_vis:
				d_local = 1.0 - (self.penalty_weight * etv)

		d_score = max(0.0, d_global * d_local)

		return {"G": g_score, "C": c_score, "D": d_score}

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

		c_text = vlm_data.get("text_concepts", [])
		c_vis  = vlm_data.get("visual_concepts", [])

		denser_modality = metrics.get("denser_modality", "EQUAL")

		# metrics["asymmetry_gap"] is the SIGNED gap from Stage 2 (V_entails_T - T_entails_V).
		# We need the magnitude here for w_pos discounting.
		raw_asym_gap    = metrics.get("asymmetry_gap", 0.0)
		asym_gap_abs    = abs(raw_asym_gap) if raw_asym_gap is not None else 0.0

		entail_V_to_T = metrics.get("entail_V_to_T", 0.0)
		entail_T_to_V = metrics.get("entail_T_to_V", 0.0)

		# Stage 3: CGD Audit
		# Resolve canonical ONCE per concept and store alongside scores.
		# Eliminates second O(|V|) cosine scan that original code triggered
		# inside every regime block.
		all_concepts: List[str] = list(set(c_text + c_vis))

		# audited_concepts: raw_concept -> {"scores": {G,C,D}, "canonical": str}
		audited_concepts: Dict[str, Dict[str, Any]] = {}
		for c in all_concepts:
			resolved_c = self._resolve_to_canonical(c)
			
			if resolved_c is None:
				if self.verbose:
					print(f"[WARNING] Dropping out-of-domain concept: {c}")
				continue  # Out-of-domain concept — drop silently
			
			scores = self.audit_concept_CGD(
				c,
				c_text,
				c_vis,
				regime,
				denser_modality,
				entail_V_to_T,
				entail_T_to_V,
			)
			audited_concepts[c] = {"scores": scores, "canonical": resolved_c}

		# Stage 4: Regime-Aware Gated Consolidation
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
			# w_pos discounted proportionally to the density asymmetry magnitude.
			w_pos = max(0.5, 1.0 - (self.lambda_asym * asym_gap_abs))
			w_neg = 0.0
		elif regime == "HARD_CONFLICT":
			# Modalities are semantically disjoint.
			# Only ORPHANED text concepts (O_text_unverified from Stage 2)
			# become hard negatives. Text concepts that were matched to a visual
			# concept in Stage 2 (E_strong / E_density) are NOT hard negatives —
			# they have partial visual grounding and should not be repelled.
			o_text_orphans: set = set(evidence.get("O_text_unverified", []))

			hn_g_scores: List[float] = []
			for c, data in audited_concepts.items():
				resolved = data["canonical"]
				if c in c_vis:
					pos_targets.add(resolved)
				elif c in o_text_orphans:
					hn_targets.add(resolved)
					hn_g_scores.append(data["scores"]["G"])

			# w_neg = 1.0 - mean_G(hard_negatives) per blueprint spec.
			# Repulsion is strongest for completely ungrounded text concepts (G≈0)
			# and softer for concepts with residual visual similarity (G>0).
			mean_hn_g = float(np.mean(hn_g_scores)) if hn_g_scores else 0.0
			w_pos = 0.30
			w_neg = max(0.0, 1.0 - mean_hn_g)
		else:
			w_pos = 0.50
			w_neg = 0.0
			for c, data in audited_concepts.items():
				pos_targets.add(data["canonical"])
			if self.verbose:
				print(
					f"[WARN] {sample_id:<85}{regime:<20} conservative fallback "
					f"(all resolved concepts → pos_targets, w_pos={w_pos}, w_neg={w_neg})"
				)

		# Robustness fallback
		# If pos_targets is still empty after gating (e.g. all D(c) < tau in a
		# SOFT_CONFLICT sample with very abstract concepts), recover using raw
		# visual concepts — the most grounded signal available.
		if not pos_targets and c_vis:
			for c in c_vis:
				resolved = self._resolve_to_canonical(c)
				if resolved:
					pos_targets.add(resolved)

		# Build audit_trail for interpretability / paper diagnostics
		# Expose only the CGD scores (not the canonical string) to keep the
		# audit_trail schema identical to the original for downstream readers.
		audit_trail: Dict[str, Dict[str, float]] = {
			c: data["scores"] 
			for c, data in audited_concepts.items()
		}

		return {
			"id":               sample_id,
			column: 						vlm_data,
			"regime":           regime,
			"audit_trail":      audit_trail,
			"positive_targets": sorted(pos_targets),
			"hard_negatives":   sorted(hn_targets),
			"w_pos":            w_pos,
			"w_neg":            w_neg,
		}

def run(input_jsonl: str, column: str, verbose: bool = False) -> None:
	"""
	Streams Stage 2 receipts through the CGD Consolidator (Stages 3 & 4) and
	writes the final auditable supervision matrix to .parquet / .csv / .jsonl.

	Crash-safe resume via JSONL append-and-skip.
	If a partial output JSONL already exists, already-processed sample IDs are
	loaded and skipped so a restart continues from where it left off.
	The .parquet and .csv are rebuilt from the complete JSONL at the end.
	"""
	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	# Derive output paths
	parquet_path = os.path.join(
		outputs_dir,
		os.path.basename(input_jsonl.replace(".jsonl", "_auditable_matrix.parquet"))
	)
	csv_path   = parquet_path.replace(".parquet", ".csv")
	jsonl_path = parquet_path.replace(".parquet", ".jsonl")

	# Resume: load already-processed IDs
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
				f"already processed. Skipping."
			)

	consolidator = CGDConsolidator(input_jsonl=input_jsonl, verbose=verbose)

	print(f"\n[STAGE 3 & 4] Streaming receipts and executing stateful CGD audit...")

	rows_new: List[Dict[str, Any]] = []
	skipped = 0
	errors  = 0

	# Open output JSONL in append mode — safe for resume
	with open(jsonl_path, 'a', encoding="utf-8") as out_f, open(input_jsonl, 'r', encoding="utf-8") as in_f:
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

			# Resume skip
			if sample_id in processed_ids:
				skipped += 1
				continue

			try:
				consolidated = consolidator.consolidate_sample(receipt=receipt, column=column)
			except Exception as e:
				print(f"[ERROR] {sample_id}': {e}")
				errors += 1
				continue

			out_f.write(json.dumps(consolidated, ensure_ascii=False) + "\n")
			rows_new.append(consolidated)

	print(f"[DONE] New: {len(rows_new)} | Resumed: {skipped} | Errors: {errors}")

	# Rebuild .parquet and .csv from the complete JSONL
	# (Includes both resumed rows and newly processed rows for a consistent output)
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
	df.to_parquet(parquet_path, index=False)
	df.to_csv(csv_path, index=False)

	if verbose:
		print(f"\n[STAGE 4 COMPLETE] Saved final supervision matrix ({len(df):,} rows) to:")
		print(f"  ├─ Parquet : {parquet_path}")
		print(f"  ├─ CSV     : {csv_path}")
		print(f"  └─ JSONL   : {jsonl_path}")
		print(df.info(verbose=True, memory_usage=True))
		print("\n[DIAGNOSTICS] Regime distribution:")
		print(df['regime'].value_counts())
		print("\n[DIAGNOSTICS] Mean w_pos per Regime:")
		print(df.groupby('regime')['w_pos'].mean().round(4))
		print("\n[DIAGNOSTICS] Mean w_neg per Regime:")
		print(df.groupby('regime')['w_neg'].mean().round(4))
		print("\n[DIAGNOSTICS] Mean |pos_targets| per Regime:")
		print(df.groupby('regime')['positive_targets'].apply(lambda x: x.apply(len).mean()).round(2))
		print("\n[DIAGNOSTICS] Mean |hard_negatives| per Regime:")
		print(df.groupby('regime')['hard_negatives'].apply(lambda x: x.apply(len).mean()).round(2))
		print(df.describe())
		print(df)

if __name__ == "__main__":
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

	run(
		input_jsonl=args.jsonl_file,
		column=args.column,
		verbose=args.verbose
	)