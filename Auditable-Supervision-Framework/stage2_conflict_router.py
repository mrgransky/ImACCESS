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

def is_empty_concepts(concepts: Optional[Dict[str, Any]]) -> bool:
	if not concepts or not isinstance(concepts, dict):
		return True
	return (
		not concepts.get("text_concepts") and
		not concepts.get("visual_concepts") and
		not concepts.get("fused_concepts")
	)

def safe_parse_vlm_str(vlm_data_str: str) -> Optional[Dict]:
	if not isinstance(vlm_data_str, str):
		print(f"vlm_data_str is not a string: {vlm_data_str}")
		return vlm_data_str
	
	# Normalize smart/curly quotes to ASCII equivalents BEFORE any JSON parsing
	vlm_data_str = (
		vlm_data_str
		.replace('\u201c', '\u2018')  # " → ' (left double → left single)
		.replace('\u201d', '\u2019')  # " → ' (right double → right single)
		.replace('\u2018', "'")       # ' → '
		.replace('\u2019', "'")       # ' → '
	)
	
	# Try JSON first (clean path)
	try:
		return json.loads(vlm_data_str)
	except json.JSONDecodeError as json_e:
		print(f"[ERROR] JSON parsing failed: {json_e} - Attempting fallback: {type(vlm_data_str)} {vlm_data_str}")
		pass
	
	# Fallback: ast.literal_eval for Python-style dicts
	try:
		return ast.literal_eval(vlm_data_str)
	except Exception as e:
		print(f"[ERROR] ast.literal_eval failed: {e}")
		pass
	
	return None

class ConflictQuantifier:
	def __init__(
		self,
		sym_model_id: str = 'all-MiniLM-L6-v2',
		nli_model_id: str = 'cross-encoder/nli-deberta-v3-large',
		device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
		tau_match: float = 0.85,
		tau_soft: float = 0.55,
		tau_orphan: float = 0.60,
		tau_asym: float = 0.25,
		tau_fast_fail: float = 0.40,
		verbose: bool = False
	):
		self.device = device
		self.tau_match = tau_match
		self.tau_soft = tau_soft
		self.tau_orphan = tau_orphan
		self.tau_asym = tau_asym
		self.tau_fast_fail = tau_fast_fail
		self.verbose = verbose
		
		if self.verbose:
			print(f"\n{'='*80}")
			print(f"[STAGE 2] INIT: Modality Conflict Quantifier")
			print(f"{'='*80}")
			print(f"  ├─ Symmetric Model : {sym_model_id}")
			print(f"  ├─ NLI Model       : {nli_model_id}")
			print(f"  ├─ Device          : {self.device}")
			print(f"  └─ Thresholds      : Match={tau_match}, Soft={tau_soft}, Orphan={tau_orphan}, Asym={tau_asym}, FastFail={tau_fast_fail}")
			print(f"{'='*80}\n")
		
		# Load Symmetric Embedder (Cosine Similarity)
		self.sym_model = SentenceTransformer(
			sym_model_id, 
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
		
		# Determine Entailment Index (usually 1 for MNLI-trained CrossEncoders)
		self.entail_idx = 1 
		if hasattr(self.nli_model.config, 'label2id') and 'entailment' in self.nli_model.config.label2id:
			self.entail_idx = self.nli_model.config.label2id['entailment']
	
	def compute_asymmetry_gap(self, text_concepts: List[str], vis_concepts: List[str]) -> Dict[str, float]:
		"""
		Computes Semantic Information Asymmetry using NLI Entailment.
		"""
		if not text_concepts or not vis_concepts:
			return {"V_entails_T": 0.0, "T_entails_V": 0.0, "gap": 0.0}
		
		# V -> T (Visual entails Text. Visual is Denser/Hyponym)
		v_to_t_pairs = [[v, t] for v in vis_concepts for t in text_concepts]
		
		# T -> V (Text entails Visual. Text is Denser/Hyponym)
		t_to_v_pairs = [[t, v] for t in text_concepts for v in vis_concepts]
		
		# Predict Entailment (Softmax probabilities)
		v2t_preds = self.nli_model.predict(v_to_t_pairs, apply_softmax=True)
		t2v_preds = self.nli_model.predict(t_to_v_pairs, apply_softmax=True)
		
		# Extract entailment probabilities
		v2t_entail_probs = v2t_preds[:, self.entail_idx].reshape(len(vis_concepts), len(text_concepts))
		t2v_entail_probs = t2v_preds[:, self.entail_idx].reshape(len(text_concepts), len(vis_concepts))
		
		# Average Maximum Entailment
		avg_v_to_t = float(v2t_entail_probs.max(axis=1).mean())
		avg_t_to_v = float(t2v_entail_probs.max(axis=1).mean())
		
		gap = avg_v_to_t - avg_t_to_v

		return {
			"V_entails_T": avg_v_to_t,
			"T_entails_V": avg_t_to_v,
			"gap": gap
		}

	def process_sample(self, sample_id: str, vlm_json: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Computes the Evidence Receipt and routes the sample to a Conflict Regime.
		"""
		# Ensure VLM JSON is valid
		if not isinstance(vlm_json, dict):
			print(f"[ERROR] Invalid JSON for id: {sample_id}")
			return {"id": sample_id, "regime": "HARD_CONFLICT", "error": "Invalid JSON"}
		
		c_text = vlm_json.get("text_concepts", [])
		c_vis = vlm_json.get("visual_concepts", [])
		c_fused = vlm_json.get("fused_concepts", [])
				
		# 1. VLM-Driven Hard Conflict Short-Circuit
		if isinstance(c_fused, list) and len(c_fused) == 0:
			return self._build_receipt(
				sample_id, 
				"HARD_CONFLICT", 
				c_text, 
				c_vis, 
				"VLM short-circuit triggered ([])"
			)
		
		if not c_text or not c_vis:
			return self._build_receipt(
				sample_id, 
				"HARD_CONFLICT", 
				c_text, 
				c_vis, 
				"Missing Modality"
			)

		# 2. TIER 1: SYMMETRIC GATE (FAST COMPUTE)
		emb_t = self.sym_model.encode(c_text, convert_to_numpy=True, normalize_embeddings=True)
		emb_v = self.sym_model.encode(c_vis, convert_to_numpy=True, normalize_embeddings=True)

		# Mean-pooled set similarity (Shape: 1x1)
		set_sim = float(1 - scipy.spatial.distance.cosine(emb_t.mean(axis=0), emb_v.mean(axis=0)))

		# FAST SHORT-CIRCUIT: If sets are completely disjoint, skip NLI!
		if set_sim < self.tau_fast_fail:
			return self._build_receipt(
				sample_id, 
				"HARD_CONFLICT", 
				c_text, 
				c_vis, 
				f"Fast fail (Set Sim={set_sim})"
			)

		# 3. Symmetric Matrix & Orphan Extraction
		sim_matrix = 1 - scipy.spatial.distance.cdist(emb_t, emb_v, metric="cosine")
		matched_t, matched_v = set(), set()
		e_strong, e_density = [], []

		for i, t in enumerate(c_text):
			best_v_idx = int(np.argmax(sim_matrix[i]))
			best_sim = float(sim_matrix[i][best_v_idx])
			v = c_vis[best_v_idx]
			if best_sim >= self.tau_match:
				e_strong.append({"text": t, "vis": v, "sim": best_sim})
				matched_t.add(i)
				matched_v.add(best_v_idx)
			elif best_sim >= self.tau_soft:
				e_density.append({"text": t, "vis": v, "sim": best_sim})
				matched_t.add(i)
				matched_v.add(best_v_idx)
		
		o_text = [c_text[i] for i in range(len(c_text)) if i not in matched_t]
		o_vis = [c_vis[j] for j in range(len(c_vis)) if j not in matched_v]
		orphan_ratio = (len(o_text) + len(o_vis)) / max(1, len(c_text) + len(c_vis))

		# 4. TIER 2: Asymmetric Audit (NLI Density Verification)
		asym_metrics = self.compute_asymmetry_gap(c_text, c_vis)
		gap = abs(asym_metrics["gap"])
		
		# 5. Deterministic Regime Router
		regime = "AGREEMENT"
		action = "Standard mapping."
		
		# Condition 1: Hard Conflict (Too many unverified concepts)
		if orphan_ratio >= self.tau_orphan:
			regime = "HARD_CONFLICT"
			action = f"High orphan ratio ({orphan_ratio}). Modalities disjoint."
		# Condition 2: Soft Conflict (Topic matches, but density mismatches)
		elif gap >= self.tau_asym:
			regime = "SOFT_CONFLICT"
			denser = "VISUAL" if asym_metrics["gap"] > 0 else "TEXT"
			action = f"Density mismatch confirmed by NLI. {denser} is denser. Gap: {gap}"

		return {
			"id": sample_id,
			"vlm_cot_raw": vlm_json,
			"regime": regime,
			"metrics": {
				"set_similarity": set_sim,
				"orphan_ratio": orphan_ratio,
				"asymmetry_gap": asym_metrics["gap"],
				"entail_V_to_T": asym_metrics["V_entails_T"],
				"entail_T_to_V": asym_metrics["T_entails_V"],
				"denser_modality": "VISUAL" if asym_metrics["gap"] > 0 else "TEXT" if asym_metrics["gap"] < 0 else "EQUAL"
			},
			"evidence": {
				"E_strong_pairs": e_strong,
				"E_density_pairs": e_density,
				"O_text_unverified": o_text,
				"O_vis_unmentioned": o_vis
			},
			"action": action
		}

	def _build_receipt(self, sample_id, regime, c_text, c_vis, action):
		"""Helper for short-circuit conditions. Updated to include set_similarity."""
		return {
			"id": sample_id,
			"vlm_cot_raw": {
				"text_concepts": c_text,
				"visual_concepts": c_vis,
				"fused_concepts": []
			},
			"regime": regime,
			"metrics": {
				"set_similarity": 0.0, # Defaulting to 0 for short-circuits
				"orphan_ratio": 1.0, 
				"asymmetry_gap": 0.0,
				"entail_V_to_T": 0.0,
				"entail_T_to_V": 0.0,
				"denser_modality": "EQUAL"
			},
			"evidence": {
				"E_strong_pairs": [], "E_density_pairs": [],
				"O_text_unverified": c_text, "O_vis_unmentioned": c_vis
			},
			"action": action
		}

def modality_conflict_audit(input_jsonl: str, column: str, verbose: bool = False,):
	# Modality Conflict Quantifier over the dataset.
	print(f"[STAGE 2] Loading Stage 1 outputs from {input_jsonl}")
	records = []
	skipped = 0
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
					skipped += 1
					continue
				records.append((sid, concepts))
			except json.JSONDecodeError as e:
				print(f"[WARN] Skipping malformed line {line_no:.4d} {e}")
				print(f"{line}")
				skipped += 1

	print(f"[STAGE 2] Loaded {len(records)} records ({skipped} skipped).")
	quantifier = ConflictQuantifier(verbose=verbose)
	
	receipts = []
	for sample_id, vlm_data in tqdm(records, desc="Auditing"):
		if is_empty_concepts(vlm_data):
			if verbose:
				print(f"[SKIP] Empty concepts for {sample_id}")
			continue
		
		try:
			receipt = quantifier.process_sample(sample_id, vlm_data)
			receipts.append(receipt)
		except Exception as e:
			print(f"[ERROR] quantifier failed for {sample_id}: {e}")

	# Save Receipts to JSONL
	output_jsonl = input_jsonl.replace(".jsonl", "_modality_conflict_audit.jsonl")
	print(f"[STAGE 2] Saving Evidence Receipts to {output_jsonl}")
	with open(output_jsonl, 'w') as f:
		for r in receipts:
			f.write(json.dumps(r) + '\n')
					
	# Quick Diagnostic
	df_receipts = pd.DataFrame(receipts)
	print("\n[STAGE 2] DATASET HEALTH DIAGNOSTIC:")
	print(df_receipts['regime'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="VLLM-instruct-based keyword annotation for Historical Dataset")
	parser.add_argument("--jsonl_file", '-jsonl', type=str, required=True, help="Path to the VLM CoT")
	parser.add_argument("--verbose", '-v', action='store_true', help="Verbose output")
	args = parser.parse_args()
	set_seeds(seed=42)

	if not args.jsonl_file.endswith(".jsonl"):
		raise ValueError(f"Input file must be a JSONL file, Got: {args.jsonl_file}")
	
	modality_conflict_audit(input_jsonl=args.jsonl_file, column="vlm_cot_raw", verbose=args.verbose)