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

class CGDConsolidator:
	def __init__(
		self,
		input_jsonl: str,
		alpha: float = 0.05,       # Decay factor for global density
		penalty_weight: float = 0.80, # How aggressively to penalize hypernyms in Soft Conflicts
		tau_density_filter: float = 0.30, # Minimum density score for soft conflict targets
		lambda_asym: float = 1.0,  # Multiplier for soft-conflict positive weight discount
		verbose: bool = False
	):
		self.input_jsonl = input_jsonl
		self.alpha = alpha
		self.penalty_weight = penalty_weight
		self.tau_density_filter = tau_density_filter
		self.lambda_asym = lambda_asym
		self.verbose = verbose
		self.outputs_dir = os.path.join(os.path.dirname(self.input_jsonl), "outputs")
		
		map_path = os.path.join(self.outputs_dir, os.path.basename(self.input_jsonl.replace(".jsonl", "_canonical_map.json")))
		freqs_path = os.path.join(self.outputs_dir, os.path.basename(self.input_jsonl.replace(".jsonl", "_global_label_frequency.json")))
		emb_path = os.path.join(self.outputs_dir, os.path.basename(self.input_jsonl.replace(".jsonl", "_emb_cache.pt")))
		
		if self.verbose:
			print(f"\n{'='*80}\n[STAGE 3 & 4] INIT: CGD Consolidator & Regime-Aware Router\n{'='*80}")
			print(f"  ├─ Canonical Map   : {map_path}")
			print(f"  ├─ Global Freqs    : {freqs_path}")
			print(f"  ├─ Embedding Cache : {emb_path}")
		
		with open(map_path, 'r') as f:
			self.canonical_map = json.load(f)
		
		with open(freqs_path, 'r') as f:
			self.global_freqs = json.load(f)
			
		# PERFORMANCE OPTIMIZATION: Cache max_freq once in init to avoid redundant O(Vocab) loops
		self.max_freq = max(self.global_freqs.values()) if self.global_freqs else 1
		
		# Load pre-computed normalized embeddings dictionary {label_str: np.ndarray}
		try:
			with torch.serialization.safe_globals([np.ndarray]):
				self.emb_cache = torch.load(emb_path, map_location="cpu", weights_only=True)
		except Exception as e:
			if "weights_only" in str(e):
				print("[WARN] Falling back to weights_only=False due to compatibility issues.")
				self.emb_cache = torch.load(emb_path, map_location="cpu", weights_only=False)
			else:
				raise e
		
		print(f"  └─ Loaded {len(self.emb_cache)} embeddings")
	
	def _get_embedding(self, label: str) -> Optional[np.ndarray]:
		"""Safely fetch normalized embedding from cache."""
		return self.emb_cache.get(label, None)
	
	def _fast_cosine_sim(self, label_a: str, label_b: str) -> float:
		"""Instant L2-normalized cosine similarity via dot product."""
		emb_a = self._get_embedding(label_a)
		emb_b = self._get_embedding(label_b)
		if emb_a is None or emb_b is None:
			return 0.0
		return float(np.dot(emb_a, emb_b))
	
	def audit_concept_CGD(
		self, 
		concept: str, 
		c_text: List[str], 
		c_vis: List[str], 
		regime: str, 
		denser_modality: str,
		entail_V_to_T: float,
		entail_T_to_V: float
	) -> Dict[str, float]:
		# STAGE 3: Calculates continuous C, G, D scores for an individual concept.
		
		# 1. Grounding Score G(c) (Spatial verification to visual pixels)
		if concept in c_vis:
			g_score = 1.0
		else:
			g_score = max([self._fast_cosine_sim(concept, v) for v in c_vis]) if c_vis else 0.0
		
		# 2. Coverage Score C(c) (Dataset Rarity / Information Content)
		freq = self.global_freqs.get(concept, 1)
		c_score = 1.0 - (math.log(1 + freq) / math.log(1 + self.max_freq))
		
		# 3. Density Score D(c) = D_global * D_local
		# Global Reusability prior (exponential frequency scaling)
		d_global = 1.0 - math.exp(-self.alpha * freq)
		
		# Local Abstraction Penalty (Symmetric NLI Entailment scaling)
		d_local = 1.0
		if regime == "SOFT_CONFLICT":
			if denser_modality == "VISUAL" and concept in c_text:
				# Text is broad hypernym; penalize based on V -> T entailment
				d_local = 1.0 - (self.penalty_weight * entail_V_to_T)
			elif denser_modality == "TEXT" and concept in c_vis:
				# Visual is broad hypernym; penalize based on T -> V entailment
				d_local = 1.0 - (self.penalty_weight * entail_T_to_V)
		
		d_score = max(0.0, d_global * d_local)
		
		return {"G": g_score, "C": c_score, "D": d_score}
	
	def consolidate_sample(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
		"""
		STAGE 4: Maps audited concepts into canonical V, applies Regime Gating, 
		and derives w_pos, w_neg, positive_targets, and hard_negatives.
		"""
		print(receipt.keys())
		sample_id = receipt["id"]
		regime = receipt["regime"]
		metrics = receipt["metrics"]
		vlm_data = receipt.get("vlm_cot_raw", {})
		c_text = vlm_data.get("text_concepts", [])
		c_vis = vlm_data.get("visual_concepts", [])
		
		denser_modality = metrics.get("denser_modality", "EQUAL")
		asym_gap = abs(metrics.get("asymmetry_gap", 0.0))
		
		# Extract directional NLI metrics for Stage 3 continuous density penalty
		entail_V_to_T = metrics.get("entail_V_to_T", 0.0)
		entail_T_to_V = metrics.get("entail_T_to_V", 0.0)
		
		# Perform the Stage 3 CGD audit for every proposed concept
		all_concepts = list(set(c_text + c_vis))
		audited_concepts = {}
		for c in all_concepts:
			if c not in self.canonical_map:
				continue
			audited_concepts[c] = self.audit_concept_CGD(
				c, c_text, c_vis, regime, denser_modality, entail_V_to_T, entail_T_to_V
			)

		# Output Target Lists
		pos_targets = set()
		hn_targets = set()
		w_pos = 1.0
		w_neg = 0.0

		# Stage 4 Regime-Aware Routing Gating
		if regime == "AGREEMENT":
			for c, scores in audited_concepts.items():
				pos_targets.add(self.canonical_map[c])
			w_pos = 1.0
			w_neg = 0.0
			
		elif regime == "SOFT_CONFLICT":
			for c, scores in audited_concepts.items():
				if scores["D"] >= self.tau_density_filter:
					pos_targets.add(self.canonical_map[c])
			
			w_pos = max(0.5, 1.0 - (self.lambda_asym * asym_gap))
			w_neg = 0.0
			
		elif regime == "HARD_CONFLICT":
			for c, scores in audited_concepts.items():
				if c in c_vis:
					pos_targets.add(self.canonical_map[c])
				elif c in c_text:
					hn_targets.add(self.canonical_map[c])
			w_pos = 0.30
			w_neg = 1.0  # Safe, explicit, maximum repulsion for ungrounded noise
		
		# Robustness fallback: If pos_targets ends up completely empty, assign fallback visual concepts
		if not pos_targets and c_vis:
			for c in c_vis:
				if c in self.canonical_map:
					pos_targets.add(self.canonical_map[c])
		
		return {
			"id": sample_id,
			"regime": regime,
			"positive_targets": sorted(list(pos_targets)),
			"hard_negatives": sorted(list(hn_targets)),
			"w_pos": float(w_pos),
			"w_neg": float(w_neg),
			"audit_trail": audited_concepts
		}

def run_stateful_map_pipeline(input_jsonl: str, verbose: bool = False):
	DATASET_DIRECTORY = os.path.dirname(input_jsonl)
	outputs_dir = os.path.join(DATASET_DIRECTORY, "outputs")
	os.makedirs(outputs_dir, exist_ok=True)

	consolidator = CGDConsolidator(input_jsonl=input_jsonl, verbose=verbose)
	rows = []
	print(f"\n[STAGE 3 & 4] Parsing receipts and executing stateful audit...")
	
	with open(input_jsonl, 'r') as f:
		for line in f:
			receipt = json.loads(line)
			consolidated = consolidator.consolidate_sample(receipt)
			rows.append(consolidated)
	
	df = pd.DataFrame(rows)
	
	parquet_path = os.path.join(outputs_dir, os.path.basename(input_jsonl.replace(".jsonl", "_auditable_matrix.parquet")))
	csv_path = parquet_path.replace(".parquet", ".csv")
	jsonl_path = parquet_path.replace(".parquet", ".jsonl")

	df.to_parquet(parquet_path, index=False)
	df.to_csv(csv_path, index=False)

	with open(jsonl_path, 'w') as f:
		for row in rows:  
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	if verbose:
		print(f"\n[STAGE 4 COMPLETE] Saved final supervision matrix ({len(df):,} rows) to:")
		print(f"  - Parquet: {parquet_path}")
		print(f"  - CSV: {csv_path}")
		print(f"  - JSONL: {jsonl_path}")
		print(df.info(verbose=True, memory_usage=True))
		print("\n[DIAGNOSTICS] Mean Positive Weight (w_pos) per Regime:")
		print(df.groupby('regime')['w_pos'].mean())
		print("\n[DIAGNOSTICS] Mean Negative Weight (w_neg) per Regime:")
		print(df.groupby('regime')['w_neg'].mean())
		print(df.head(5))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Stage 3 & 4: Stateful CGD Audit & Regime Routing")
	parser.add_argument("--jsonl_file", "-jsonl", type=str, required=True, help="Path to Stage 2 modality conflic audit JSONL file")
	parser.add_argument("--verbose", "-v", action='store_true', help="Verbose diagnostics")
	args = parser.parse_args()

	if "_modality_conflict_audit.jsonl" not in args.jsonl_file:
		raise ValueError(f"Input JSONL file must be a Stage 2 modality conflict audit file. Got: {args.jsonl_file}")
	
	run_stateful_map_pipeline(input_jsonl=args.jsonl_file, verbose=args.verbose)