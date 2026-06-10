import os
import sys

HOME, USER = os.getenv('HOME'), os.getenv('USER')
IMACCESS_PROJECT_WORKSPACE = os.path.join(HOME, "WS_Farid", "ImACCESS")

CLIP_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "clip")
sys.path.insert(0, CLIP_DIR)

MISC_DIR = os.path.join(IMACCESS_PROJECT_WORKSPACE, "misc")
sys.path.insert(0, MISC_DIR)

from utils import *

def is_empty_concepts(concepts: Optional[Dict[str, Any]]) -> bool:
	if not concepts or not isinstance(concepts, dict):
		return True
	return (
		not concepts.get("text_concepts") and
		not concepts.get("visual_concepts") and
		not concepts.get("fused_concepts")
	)

class ConflictQuantifier:
	def __init__(
		self,
		sym_model_id: str,
		nli_model_id: str,
		device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
		tau_match: float = 0.85, # exact semantic equivalence
		tau_soft: float = 0.55, # related/hierarchical concepts
		tau_orphan: float = 0.60,
		tau_asym: float = 0.25,
		tau_fast_fail: float = 0.40,
		batch_size: int = 2**10,
		verbose: bool = False
	):
		self.device = device
		self.tau_match = tau_match
		self.tau_soft = tau_soft
		self.tau_orphan = tau_orphan
		self.tau_asym = tau_asym
		self.tau_fast_fail = tau_fast_fail
		self.batch_size = batch_size
		self.verbose = verbose
				
		# Load Symmetric Embedder (Cosine Similarity)
		self.sym_model = SentenceTransformer(
			sym_model_id,
			model_kwargs={"attn_implementation": self.get_attention(), "dtype": self.get_dtype()} if "Qwen" in sym_model_id else {},
			device=self.device,
			trust_remote_code=True,
			cache_folder=cache_directory[os.getenv('USER')],
			token=os.getenv("HUGGINGFACE_TOKEN"),
			tokenizer_kwargs={"padding_side": "left"},
		)
		
		# Load Asymmetric NLI Model (Cross Encoder)
		self.nli_model = CrossEncoder(
			nli_model_id, 
			device=self.device,
			trust_remote_code=True,
			cache_folder=cache_directory[os.getenv('USER')],
			token=os.getenv("HUGGINGFACE_TOKEN"),
			tokenizer_kwargs={"padding_side": "left"},
		)
		
		# Robust, config-aware NLI Entailment Index Resolution.
		# Iterate over label2id keys to catch any casing variant (e.g. "ENTAILMENT", "Entailment").
		# Raise a hard error rather than silently defaulting to a wrong index.
		self.entail_idx = None
		if hasattr(self.nli_model.config, 'label2id'):
			label2id = self.nli_model.config.label2id
			for label_name, idx in label2id.items():
				if 'entail' in label_name.lower():
					self.entail_idx = int(idx)
					if self.verbose:
						print(f"\t[NLI] Entailment index resolved: '{label_name}' → {self.entail_idx}")
					break

			if self.entail_idx is None:
				raise ValueError(
					f"[FATAL] NLI model '{nli_model_id}' label2id does not contain an "
					f"'entailment' label. Found: {list(label2id.keys())}. "
					f"Cannot safely resolve entailment index."
				)
		else:
			# No label2id exposed — fall back to index 1 with an explicit warning.
			# This is a known safe default for DeBERTa-v3 MNLI models.
			self.entail_idx = 1
			print(
				f"[WARN] NLI model '{nli_model_id}' does not expose label2id. "
				f"Defaulting entail_idx=1. Verify this is correct for your model."
			)
	
	def get_attention(self):
		if not torch.cuda.is_available():
			return "eager"
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			try:
				import flash_attn
				if self.verbose:
					print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
				return "flash_attention_2"
			except ImportError:
				if self.verbose:
					print(f"[WARN] Flash Attention 2 not installed (pip install flash-attn)")
		if compute_cap >= 7.0 and torch.__version__ >= "2.0.0":
			if self.verbose:
				print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
			return "sdpa"
		if self.verbose:
			print(f"[INFO] Using eager attention (compute {compute_cap})")
		return "eager"

	def get_dtype(self):
		dtype = torch.float32
		if torch.cuda.is_available():
			dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
		return dtype

	def compute_asymmetry_gap(
		self,
		text_concepts: List[str],
		vis_concepts: List[str],
	) -> Dict[str, Any]:
		"""
		Computes Semantic Information Asymmetry using NLI Entailment.

		Returns None for gap (not 0.0) when inputs are empty so the caller
		can distinguish "no mutual matches found" from "gap measured as zero".
		Adds 'computed_on' to record how many concept pairs were evaluated,
		which is essential for per-sample auditability in the Evidence Receipt.

		Returns:
			dict with keys: V_entails_T, T_entails_V, gap (float | None), computed_on (int)
		"""
		if not text_concepts or not vis_concepts:
			# Return None, not 0.0. A gap of 0.0 looks like AGREEMENT to the router.
			# The caller must handle None explicitly and route to HARD_CONFLICT.
			return {
				"V_entails_T": None,
				"T_entails_V": None,
				"gap": None,
				"computed_on": 0,
			}

		# V → T: Does the visual concept entail (subsume) the text concept?
		# High V→T score means visual is more specific (hyponym) than text.
		v_to_t_pairs = [[v, t] for v in vis_concepts for t in text_concepts]

		# T → V: Does the text concept entail (subsume) the visual concept?
		# High T→V score means text is more specific (hyponym) than visual.
		t_to_v_pairs = [[t, v] for t in text_concepts for v in vis_concepts]

		# Predict entailment probabilities (softmax over [contradiction, entailment, neutral])
		v2t_preds = self.nli_model.predict(v_to_t_pairs, apply_softmax=True)
		t2v_preds = self.nli_model.predict(t_to_v_pairs, apply_softmax=True)

		# Extract entailment column and reshape to [n_vis × n_text] and [n_text × n_vis]
		v2t_entail_probs = v2t_preds[:, self.entail_idx].reshape(len(vis_concepts), len(text_concepts))
		t2v_entail_probs = t2v_preds[:, self.entail_idx].reshape(len(text_concepts), len(vis_concepts))

		# Average-of-max: for each source concept, take its best entailment score,
		# then average across all source concepts. This is robust to list-length asymmetry.
		v2t_mean = float(v2t_entail_probs.max(axis=1).mean())
		t2v_mean = float(t2v_entail_probs.max(axis=1).mean())

		# gap > 0 → visual is denser / more specific than text (visual hyponym) example: "a dog" vs "a dog with a collar"
		# gap < 0 → text is denser / more specific than visual (text hyponym) example: "a dog with a collar" vs "a dog"
		gap = v2t_mean - t2v_mean

		return {
			"V_entails_T": round(v2t_mean, 4),
			"T_entails_V": round(t2v_mean, 4),
			"gap": round(gap, 4),
			"computed_on": len(v_to_t_pairs),
		}

	def process_sample(self, sample_id: str, column: str, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Computes the Evidence Receipt and routes the sample to a Conflict Regime.

		Routing authority: Stage 2 is the sole authority on regime assignment.
		Stage 1 outputs (including fused_concepts) are evidence, never routing gates.

		Regime hierarchy (evaluated in order):
		  INVALID_JSON      → vlm_json is not a dict
		  MISSING_MODALITY  → c_text or c_vis is empty (separate from conflict)
		  HARD_CONFLICT     → orphan_ratio >= tau_orphan  (structural disjointness)
		  HARD_CONFLICT     → no mutual matches survived  (NLI gap undefined)
		  SOFT_CONFLICT     → |asym.gap| >= tau_asym      (density mismatch)
		  AGREEMENT         → all signals within bounds
		"""

		# STEP 0: INPUT VALIDATION
		if not isinstance(vlm_json, dict):
			print(f"[ERROR] Invalid JSON for id: {sample_id}")
			return {
				"id": sample_id,
				"regime": "INVALID_JSON",
				"failure_mode": "vlm_json is not a dict",
				"metrics": None,
				"evidence": None,
				"advisory": None,
				"action": "Discard. Cannot process non-dict input.",
			}

		c_text  = vlm_json.get("text_concepts", [])
		c_vis   = vlm_json.get("visual_concepts", [])
		c_fused = vlm_json.get("fused_concepts", [])

		# Record VLM fusion signal as a soft advisory flag, NOT a routing gate.
		# Stage 1's fused_concepts=[] is informative but Stage 2 must verify independently.
		vlm_fusion_empty = isinstance(c_fused, list) and len(c_fused) == 0

		# MISSING_MODALITY is its own failure mode, not a HARD_CONFLICT variant.
		# Conflating them would pollute corpus-level conflict statistics.
		if not c_text or not c_vis:
			regime = "MISSING_MODALITY"
			missing = (
				"Missing Text & Visual" if not c_text and not c_vis else
				"Missing Visual" if not c_vis else
				"Missing Text"
			)
			return {
				"id": sample_id,
				column: vlm_json,
				"regime": regime,
				"failure_mode": missing,
				# Use None, not 0.0/1.0. Fake values pollute corpus-level metric distributions.
				"metrics": {
					"set_similarity": None,
					"orphan_ratio": None,
					"asymmetry_gap": None,
					"entail_V_to_T": None,
					"entail_T_to_V": None,
					"denser_modality": None,
					"nli_bypassed": None,
					"nli_computed_on": None,
				},
				"evidence": {
					"E_strong_pairs": [],
					"E_density_pairs": [],
					"O_text_unverified": c_text,
					"O_vis_unmentioned": c_vis,
				},
				"advisory": {
					"vlm_fusion_empty": vlm_fusion_empty,
					"centroid_sim_low": None,
				},
				"action": f"Abstain from cross-modal routing. {missing}.",
			}

		# STEP 1: SYMMETRIC AUDIT
		emb_t = self.sym_model.encode(
			c_text, 
			batch_size=self.batch_size,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)
		emb_v = self.sym_model.encode(
			c_vis,
			batch_size=self.batch_size,
			convert_to_numpy=True,
			normalize_embeddings=True,
			precision='float32',
		)

		# Centroid similarity: 
		# cheap global coherence signal.
		# advisory only: it must NOT gate the regime alone.
		# few unrelated concepts drag the centroid down even when most pairs align.
		set_sim = float(1 - scipy.spatial.distance.cosine(emb_t.mean(axis=0), emb_v.mean(axis=0)))
		centroid_sim_low = set_sim < self.tau_fast_fail

		# Full pairwise similarity matrix: shape [|c_text| × |c_vis|]
		sim_matrix = 1 - scipy.spatial.distance.cdist(emb_t, emb_v, metric="cosine")

		# True bidirectional (mutual) best-match.
		# A pair (t_i, v_j) is accepted only if t_i's best match is v_j AND v_j's best match is t_i.
		# One-sided greedy matching inflates matched_v and deflates orphan_ratio,
		# causing the router to systematically under-report conflict.
		t2v_best = {i: int(np.argmax(sim_matrix[i]))    for i in range(len(c_text))}
		v2t_best = {j: int(np.argmax(sim_matrix[:, j])) for j in range(len(c_vis))}

		matched_t: set = set()
		matched_v: set = set()
		e_strong:  List[Dict] = []
		e_density: List[Dict] = []

		for i, t in enumerate(c_text):
			j = t2v_best[i]
			# Mutual confirmation gate: only accept if the match is reciprocal
			if v2t_best[j] == i:
				sim = float(sim_matrix[i][j])
				if sim >= self.tau_match:
					e_strong.append({"text": t, "vis": c_vis[j], "sim": round(sim, 4)})
					matched_t.add(i)
					matched_v.add(j)
				elif sim >= self.tau_soft:
					e_density.append({"text": t, "vis": c_vis[j], "sim": round(sim, 4)})
					matched_t.add(i)
					matched_v.add(j)

		o_text = [c_text[i] for i in range(len(c_text)) if i not in matched_t]
		o_vis  = [c_vis[j]  for j in range(len(c_vis))  if j not in matched_v]
		# ratio of unverified hallucinated or uncaptioned concepts
		orphan_ratio = (len(o_text) + len(o_vis)) / max(1, len(c_text) + len(c_vis))

		# STEP 2: DETERMINISTIC HARD CONFLICT GATE (orphan_ratio only)
		# set_sim is demoted to advisory. orphan_ratio is the sole structural gate.
		# This makes the routing decision reproducible and independently verifiable.
		if orphan_ratio >= self.tau_orphan:
			regime = "HARD_CONFLICT"
			return self._build_full_receipt(
				sample_id=sample_id,
				column=column,
				vlm_json=vlm_json,
				regime=regime,
				failure_mode=None,
				set_sim=set_sim,
				orphan_ratio=orphan_ratio,
				asym_metrics={"V_entails_T": None, "T_entails_V": None, "gap": None, "computed_on": 0},
				nli_bypassed=True,
				e_strong=e_strong,
				e_density=e_density,
				o_text=o_text,
				o_vis=o_vis,
				vlm_fusion_empty=vlm_fusion_empty,
				centroid_sim_low=centroid_sim_low,
				action=(
					f"Structural disjointness "
					f"(orphan_ratio={orphan_ratio:.2f} >= tau_orphan={self.tau_orphan}) NLI bypassed."
				),
			)

		# STEP 3: ASYMMETRIC AUDIT (NLI)
		# Compute Semantic Asymmetry only on mutually matched concepts.
		# This isolates genuine density asymmetry from unrelated orphan noise.
		matched_text_concepts = [c_text[i] for i in sorted(matched_t)]
		matched_vis_concepts  = [c_vis[j]  for j in sorted(matched_v)]
		asym_metrics = self.compute_asymmetry_gap(matched_text_concepts, matched_vis_concepts)

		# Handle None gap explicitly. An empty mutual match set means no semantic
		# overlap survived bidirectional filtering → treat as HARD_CONFLICT, data-driven.
		if asym_metrics["gap"] is None:
			regime = "HARD_CONFLICT"
			return self._build_full_receipt(
				sample_id=sample_id,
				column=column,
				vlm_json=vlm_json,
				regime=regime,
				failure_mode="NO_MUTUAL_MATCHES",
				set_sim=set_sim,
				orphan_ratio=orphan_ratio,
				asym_metrics=asym_metrics,
				nli_bypassed=True,
				e_strong=e_strong,
				e_density=e_density,
				o_text=o_text,
				o_vis=o_vis,
				vlm_fusion_empty=vlm_fusion_empty,
				centroid_sim_low=centroid_sim_low,
				action=(
					"No mutually matched concept pairs survived bidirectional filtering. "
					f"NLI asymmetry gap: {asym_metrics['gap']}."
				),
			)

		# STEP 4: REGIME ROUTER (AGGREEMENT | SOFT_CONFLICT | HARD_CONFLICT)
		# Specificity Gap = Entail_V2T − Entail_T2V
		# Gap ≈ 0	: Both entailments are semantically equal (Agreement).
		# Gap > 0	: Visual entails Text, but Text does not entail Visual. (Visual is denser/hyponym).
		# Gap < 0	: Text entails Visual, but Visual does not entail Text. (Text is denser/hyponym).
		abs_gap = abs(asym_metrics["gap"])
		if abs_gap >= self.tau_asym:
			denser = "VISUAL" if asym_metrics["gap"] > 0 else "TEXT"
			regime = "SOFT_CONFLICT"
			action = (
				f"NLI Density mismatch: "
				f"|gap|={abs_gap:.4f} >= tau_asym={self.tau_asym}. "
				f"denser: {denser} "
				f"(gap={asym_metrics['gap']:.4f})."
			)
		else:
			regime = "AGREEMENT"
			action = (
				f"Modalities structurally & semantically aligned. "
				f"orphan_ratio={orphan_ratio:.3f} < tau_orphan={self.tau_orphan}. "
				f"NLI asymmetry |gap|={abs_gap:.4f}."
			)

		return self._build_full_receipt(
			sample_id=sample_id,
			column=column,
			vlm_json=vlm_json,
			regime=regime,
			failure_mode=None,
			set_sim=set_sim,
			orphan_ratio=orphan_ratio,
			asym_metrics=asym_metrics,
			nli_bypassed=False,
			e_strong=e_strong,
			e_density=e_density,
			o_text=o_text,
			o_vis=o_vis,
			vlm_fusion_empty=vlm_fusion_empty,
			centroid_sim_low=centroid_sim_low,
			action=action,
		)

	def _build_full_receipt(
		self,
		sample_id: str,
		column: str,
		vlm_json: Dict[str, Any],
		regime: str,
		failure_mode: Optional[str],
		set_sim: float,
		orphan_ratio: float,
		asym_metrics: Dict[str, Any],
		nli_bypassed: bool,
		e_strong: List[Dict],
		e_density: List[Dict],
		o_text: List[str],
		o_vis: List[str],
		vlm_fusion_empty: bool,
		centroid_sim_low: bool,
		action: str,
	) -> Dict[str, Any]:
		"""
		Constructs the canonical Evidence Receipt dict.

		Metrics are either real measurements or explicit None, 
		making the receipt safe for corpus-level statistical analysis.

		The 'advisory' block carries soft signals that informed but did not determine
		the regime. This separation is essential for ablation studies.
		"""
		gap = asym_metrics.get("gap")
		denser_modality = (
			None        if gap is None else
			"VISUAL"    if gap > 0     else
			"TEXT"      if gap < 0     else
			"EQUAL"
		)
		return {
			"id": sample_id,
			column: vlm_json,
			"evidence": {
				"E_strong_pairs":    e_strong,
				"E_density_pairs":   e_density,
				"O_text_unverified": o_text,
				"O_vis_unmentioned": o_vis,
			},
			"regime": regime,
			"failure_mode": failure_mode,
			"metrics": {
				"set_similarity":   round(set_sim, 4),
				"orphan_ratio":     round(orphan_ratio, 4),
				"asymmetry_gap":    gap,
				"entail_V_to_T":    asym_metrics.get("V_entails_T"),
				"entail_T_to_V":    asym_metrics.get("T_entails_V"),
				"denser_modality":  denser_modality,
				"nli_bypassed":     nli_bypassed,
				"nli_computed_on":  asym_metrics.get("computed_on", 0),
			},
			# Advisory block separates soft signals from routing decisions.
			# vlm_fusion_empty and centroid_sim_low are recorded here for ablation
			# but must never appear in the routing logic above.
			"advisory": {
				"vlm_fusion_empty": vlm_fusion_empty,
				"centroid_sim_low": centroid_sim_low,
			},
			"action": action,
		}

def modality_conflict_audit(
	input_jsonl: str,
	sym_model_id: str,
	asym_model_id: str,
	batch_size: int,
	column: str,
	verbose: bool = False
):
	if verbose:
		print(f"\n[STAGE 2] Modality Conflict Audit")
		print(f"  ├─ Input  : {input_jsonl}")
		print(f"  ├─ Symmetric Embedding Model  : {sym_model_id}")
		print(f"  ├─ Asymmetric Embedding Model : {asym_model_id}")
		print(f"  ├─ Batch Size : {batch_size}")
		print(f"  └─ Column : {column}")

	outputs_dir = os.path.join(os.path.dirname(input_jsonl), "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	records = []
	skipped_load = 0
	with open(input_jsonl, "r", encoding="utf-8") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
				sid = rec.get("id")
				concepts = rec.get(column)
				if sid is None or concepts is None:
					skipped_load += 1
					continue
				records.append((sid, concepts))
			except json.JSONDecodeError as e:
				print(f"[WARN] Skipping malformed line {line_no:05d}: {e}")
				skipped_load += 1
	if verbose:
		print(f"\n[LOADED] {len(records)} {type(records)} records ({skipped_load} skipped during load).")

	# RESUME LOGIC: collect already-processed IDs
	output_jsonl = input_jsonl.replace(".jsonl", "_modality_conflict_audit.jsonl")
	done_ids: set = set()
	if os.path.exists(output_jsonl):
		with open(output_jsonl, "r", encoding="utf-8") as f_done:
			for line in f_done:
				try:
					done_ids.add(json.loads(line.strip())["id"])
				except Exception:
					pass
		if done_ids:
			print(f"[STAGE 2] Resume detected: {len(done_ids):,} samples already processed. Skipping.")

	pending = [(sid, data) for sid, data in records if sid not in done_ids]
	print(f"\n[STAGE 2] Pending {type(pending)} {len(pending)} samples to process.")

	if not pending:
		print("[STAGE 2] Nothing to do. All records already processed.")
	else:
		quantifier = ConflictQuantifier(
			sym_model_id=sym_model_id,
			nli_model_id=asym_model_id,
			batch_size=batch_size,
			verbose=verbose,
		)

		skipped_empty = 0
		errors = 0
		# Stream-write each receipt immediately after processing.
		# f.flush() after every write ensures the line is on disk before the next sample.
		with open(output_jsonl, "a", encoding="utf-8") as f_out:
			for sample_id, vlm_data in tqdm(pending, desc="[STAGE 2] Auditing", ncols=120):
				if is_empty_concepts(vlm_data):
					skipped_empty += 1
					if verbose:
						print(f"  [SKIP] Empty concepts: {sample_id}")
					continue
				try:
					receipt = quantifier.process_sample(sample_id=sample_id, column=column, vlm_json=vlm_data)
					f_out.write(json.dumps(receipt) + "\n")
					f_out.flush()
				except Exception as e:
					errors += 1
					print(f"  [ERROR] quantifier failed for {sample_id}: {e}")
		print(f"[STAGE 2] Done. Skipped (empty): {skipped_empty} | Errors: {errors}")

	if verbose:
		print(f"\n[STAGE 2] Saving Evidence Receipts to: {output_jsonl}")
	all_receipts = []
	with open(output_jsonl, "r", encoding="utf-8") as f_read:
		for line in f_read:
			try:
				all_receipts.append(json.loads(line.strip()))
			except Exception:
				pass
	df_receipts = pd.DataFrame(all_receipts)

	if verbose:
		print("\nDATASET HEALTH DIAGNOSTIC:")
		print(df_receipts['regime'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

	txt_file = os.path.join(outputs_dir, "modality_conflict_stats.txt")
	with open(txt_file, "w", encoding="utf-8") as f_txt:
		regime_stats = df_receipts['regime'].value_counts(normalize=True)
		f_txt.write("MODALITY CONFLICT REGIME DISTRIBUTION\n")
		f_txt.write("=" * 50 + "\n\n")

		for regime_name, frac in regime_stats.items():
			f_txt.write(f"{regime_name:<25}{frac:.4f}  ({frac*100:.1f}%)\n")

		f_txt.write("\n" + "=" * 50 + "\n")
		f_txt.write(f"Total processed : {len(df_receipts):,}\n")
		f_txt.write(f"Input records   : {len(records):,}\n")
	print(f"\n[STAGE 2] Stats written to: {txt_file}")

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="MLM-instruct-based keyword annotation")
	parser.add_argument("--jsonl_file", '-jsonl', type=str, required=True, help="MLM CoT JSONL")
	parser.add_argument("--sym_emb_model", "-sym", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Sentence embedding model (symmetrical embedding)")
	parser.add_argument("--asym_nli_model", "-asym", type=str, default="cross-encoder/nli-deberta-v3-large", help="NLI model (asymmetrical embedding)")
	parser.add_argument("--batch_size", "-bs", type=int, default=2**10, help="Batch size for embedding")
	parser.add_argument("--column", "-col", type=str, default="mlm_cot_raw", help="Column to use for canonical analysis",)
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	args = parser.parse_args()
	print(args)
	set_seeds(seed=42)

	if not args.jsonl_file.endswith(".jsonl"):
		raise ValueError(f"Input file must be a JSONL file, Got: {args.jsonl_file}")
	
	modality_conflict_audit(
		input_jsonl=args.jsonl_file, 
		column=args.column,
		sym_model_id=args.sym_emb_model,
		asym_model_id=args.asym_nli_model,
		batch_size=args.batch_size,
		verbose=args.verbose
	)

if __name__ == "__main__":
	main()