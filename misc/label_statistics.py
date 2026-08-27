from tabnanny import verbose

from utils import *
import visualize as viz
import clip

def _infer_performance_schema(perf: Dict[str, Any]) -> str:
		"""
		Returns one of:
			- "nested_by_source_then_strategy"  (your performance.json)
			- "per_k_single_source_blob"        (already-selected strategy dict containing i2t/t2i)
			- "flat_metrics_by_source"          (old style: {source: {metric: value}})
		"""
		if not isinstance(perf, dict) or len(perf) == 0:
				return "flat_metrics_by_source"

		# If it already looks like per-k results (contains i2t/t2i at top level)
		if "i2t" in perf and "t2i" in perf:
				return "per_k_single_source_blob"

		# If top-level keys look like sources, and second-level looks like strategies
		# (heuristic: for one source, values are dicts and contain "dora"/"zero_shot"/etc OR contain dicts with i2t/t2i)
		any_source = next(iter(perf.values()))
		if isinstance(any_source, dict):
				# nested if: source -> strategy -> i2t/t2i ...
				any_strategy = next(iter(any_source.values())) if len(any_source) else None
				if isinstance(any_strategy, dict) and ("i2t" in any_strategy or "t2i" in any_strategy):
						return "nested_by_source_then_strategy"

		return "flat_metrics_by_source"

def _flatten_performance_json(
		performance_json: Dict[str, Any],
		strategy: Optional[str],
		k: Union[int, str] = 10,
		include_map: bool = True,
		include_recall: bool = False,
		tier_alias: Optional[Dict[str, str]] = None,
		reference_source: str = "multimodal_canonical_labels",
		verbose: bool = False,
) -> pd.DataFrame:
		"""
		Convert performance.json into a flat DataFrame with one row per source.

		Strategy selection (when strategy=None):
				Auto-selects the best strategy by i2t+t2i mAP@k overall on `reference_source`.
				The SAME strategy is then applied to ALL sources for a fair comparison.
				reference_source should be your proposed method (default: multimodal_canonical_labels).

		Output columns (if include_map=True):
				i2t_map{k}_overall, i2t_map{k}_head, i2t_map{k}_tail
				t2i_map{k}_overall, t2i_map{k}_head, t2i_map{k}_tail
		plus Recall analogs if include_recall=True.
		"""
		if tier_alias is None:
				tier_alias = {"rare": "tail"}

		k_str = str(k)

		# ------------------------------------------------------------------ #
		# Step 1: Resolve which strategy to use (once, from reference_source) #
		# ------------------------------------------------------------------ #
		if strategy is not None:
				chosen_strategy = strategy
				if verbose:
						print(f"[perf] Using provided strategy='{chosen_strategy}' for all sources")
		else:
				chosen_strategy = _select_best_strategy(
						performance_json=performance_json,
						reference_source=reference_source,
						k_str=k_str,
						verbose=verbose,
				)
				if verbose:
						print(
								f"[perf] Auto-selected strategy='{chosen_strategy}' "
								f"(best on '{reference_source}' at k={k_str}, i2t+t2i mAP overall)"
						)

		# ------------------------------------------------------------------ #
		# Step 2: Extract metrics for each source using the chosen strategy   #
		# ------------------------------------------------------------------ #
		rows: List[Dict[str, Any]] = []
		for source, source_blob in performance_json.items():
				if not isinstance(source_blob, dict):
						continue

				schema = _infer_performance_schema({source: source_blob})
				if schema != "nested_by_source_then_strategy":
						continue

				if chosen_strategy not in source_blob:
						if verbose:
								print(f"[perf] strategy='{chosen_strategy}' missing for source='{source}' -> NaNs")
						rows.append({"source": source, "strategy": chosen_strategy})
						continue

				per_k = source_blob[chosen_strategy]

				def get_metric(direction: str, tier: str, metric_name: str) -> Optional[float]:
						try:
								d = per_k[direction][tier][metric_name]
								if k_str in d:
										return float(d[k_str])
								if int(k_str) in d:
										return float(d[int(k_str)])
						except Exception:
								return None
						return None

				out: Dict[str, Any] = {"source": source, "strategy": chosen_strategy}
				for direction in ["i2t", "t2i"]:
						for tier in ["overall", "head", "rare"]:
								tier_out = tier_alias.get(tier, tier)
								if include_map:
										out[f"{direction}_map{k_str}_{tier_out}"] = get_metric(direction, tier, "mAP")
								if include_recall:
										out[f"{direction}_recall{k_str}_{tier_out}"] = get_metric(direction, tier, "Recall")

				rows.append(out)

		return pd.DataFrame(rows)

def _select_best_strategy(
		performance_json: Dict[str, Any],
		reference_source: str,
		k_str: str,
		verbose: bool = False,
) -> str:
		"""
		From performance_json[reference_source], pick the strategy with the highest
		combined (i2t + t2i) mAP@k overall score.
		Falls back to a deterministic preferred list if scores are unavailable.
		"""
		PREFERRED_FALLBACK = ["dora", "lora_plus", "rslora", "lora", "vera", "ia3", "full", "probe", "zero_shot"]

		source_blob = performance_json.get(reference_source)
		if source_blob is None:
				if verbose:
						print(
								f"[perf] reference_source='{reference_source}' not found in performance.json. "
								f"Falling back to preferred list."
						)
				available = list(next(iter(performance_json.values()), {}).keys())
				for s in PREFERRED_FALLBACK:
						if s in available:
								return s
				return sorted(available)[0] if available else "dora"

		best_strategy = None
		best_score = -1.0
		score_table: Dict[str, float] = {}

		for strat, per_k in source_blob.items():
				try:
						i2t_score = float(per_k["i2t"]["overall"]["mAP"].get(k_str, -1))
						t2i_score = float(per_k["t2i"]["overall"]["mAP"].get(k_str, -1))
						combined = i2t_score + t2i_score
				except Exception:
						combined = -1.0
				score_table[strat] = combined
				if combined > best_score:
						best_score = combined
						best_strategy = strat

		if verbose:
				print(f"[perf] Strategy scores on '{reference_source}' at k={k_str} (i2t+t2i mAP overall):")
				for s, sc in sorted(score_table.items(), key=lambda x: -x[1]):
						marker = " ← SELECTED" if s == best_strategy else ""
						print(f"       {s:20s}: {sc:.4f}{marker}")

		if best_strategy is None:
				# Fallback
				available = list(source_blob.keys())
				for s in PREFERRED_FALLBACK:
						if s in available:
								return s
				return sorted(available)[0]

		return best_strategy

def _parse_label_cell(val: Any) -> List[str]:
	"""
	Robustly parse a dataframe cell that should contain a list of labels.
	Supports: list[str], string representation of list, NaN/None/""/"[]".
	Returns a list (possibly empty). Never raises.
	"""
	if isinstance(val, list):
		return [x for x in val if isinstance(x, str) and x != ""]
	
	if val is None or (isinstance(val, float) and pd.isna(val)):
		return []
	if isinstance(val, str):
		if val.strip() in ("", "[]"):
			return []
		try:
			parsed = ast.literal_eval(val)
			if isinstance(parsed, list):
				return [x for x in parsed if isinstance(x, str) and x != ""]
			return []
		except Exception:
			return []
	
	return []

def _shannon_entropy(counts: Counter, base: float = 2.0) -> float:
	"""
	Shannon entropy of a discrete distribution defined by counts.
	Returns 0.0 for empty counts.
	"""
	total = sum(counts.values())
	if total <= 0:
		return 0.0
	log = math.log
	ent = 0.0
	for c in counts.values():
		p = c / total
		if p > 0:
			ent -= p * (log(p) / log(base))
	
	return ent

def auto_calibrate_semantic_threshold(
		model: SentenceTransformer,
		verbose: bool = False
) -> Tuple[float, Dict]:

	# Extract model name/identifier
	model_name = None

	if verbose:
		print("\n[DEBUG] model attributes")
		print(f"{type(model)}")
		print(f"Has model_card_data: {hasattr(model, 'model_card_data')}")

		if hasattr(model, 'model_card_data') and model.model_card_data:
			print(f"  model_card_data.model_id: {model.model_card_data.model_id}")

		print(f"Has _model_card_data: {hasattr(model, '_model_card_data')}")
		print(f"Has model_name: {hasattr(model, 'model_name')}")
		print(f"Has name_or_path: {hasattr(model, 'name_or_path')}")
		print(f"\nmodel[0] type: {type(model[0])}")
		print(f"Has auto_model: {hasattr(model[0], 'auto_model')}")

		if hasattr(model[0], 'auto_model'):
			print(f"  auto_model type: {type(model[0].auto_model)}")
			print(f"  Has config: {hasattr(model[0].auto_model, 'config')}")
			if hasattr(model[0].auto_model, 'config'):
				cfg = model[0].auto_model.config
				print(f"  config._name_or_path: {getattr(cfg, '_name_or_path', 'NOT FOUND')}")
				print(f"  config.name_or_path: {getattr(cfg, 'name_or_path', 'NOT FOUND')}")
				print(f"  All config attrs: {[a for a in dir(cfg) if not a.startswith('_')]}")

		print("="*100)
	
	# Method 1: Check model_card_data (Public API)
	if hasattr(model, 'model_card_data') and model.model_card_data:
			# Sometimes model_card_data exists but model_id is None
			m_id = getattr(model.model_card_data, 'model_id', None)
			if m_id:
					model_name = m_id
					
	# Method 2: Check private _model_card_data (Internal API)
	if not model_name:
			if hasattr(model, '_model_card_data') and model._model_card_data:
					m_id = getattr(model._model_card_data, 'model_id', None)
					if m_id:
							model_name = m_id

	# Method 3: Check direct attributes on the SentenceTransformer object
	if not model_name:
			for attr in ['name_or_path', 'model_name_or_path']:
					if hasattr(model, attr):
							val = getattr(model, attr)
							if val:
									model_name = val
									break

	# Method 4: Fallback - Extract from the underlying Transformer's Config
	if not model_name:
			try:
					# SentenceTransformers models are usually a list of modules. 
					# The first module [0] is typically the Transformer.
					if len(model) > 0:
							first_module = model[0]
							
							# Check if it has an auto_model (HuggingFace model)
							if hasattr(first_module, 'auto_model'):
									hf_model = first_module.auto_model
									
									# Check if it has a config
									if hasattr(hf_model, 'config'):
											cfg = hf_model.config
											
											# Try standard HuggingFace config attributes
											if hasattr(cfg, '_name_or_path') and cfg._name_or_path:
													model_name = cfg._name_or_path
											elif hasattr(cfg, 'name_or_path') and cfg.name_or_path:
													model_name = cfg.name_or_path
			except Exception as e:
					if verbose:
							print(f"Fallback extraction error: {e}")
	if not model_name or model_name == "None":
		model_name = "unknown"

	if verbose:
		print(f"AUTOMATIC THRESHOLD CALIBRATION | embedding model: {model_name}\n")
	
	# TEST PAIRS - Diverse and challenging	

	# CATEGORY 1: Direct Synonyms (MUST match)
	direct_synonyms = [
		("soldier", "infantry"),
		("aircraft", "airplane"),
		("military", "army"),
		("vehicle", "car"),
		("building", "structure"),
		("weapon", "gun"),
		("uniform", "clothing"),
		("commandant", "commander"),
		("pilot", "aviator"),
		("ship", "vessel"),
		("artillery", "weapon"),
		("airfield", "airstrip"),
		("commander", "captain"),
		("admiral", "commander"),
		("rifle", "shotgun"),
	]
	
	# CATEGORY 2: Related Concepts (SHOULD match - semantic field overlap)
	related_concepts = [
		("soldier", "military base"),
		("aircraft", "pilot"),
		("tank", "armored vehicle"),
		("photograph", "camera"),
		("portrait", "face"),
		("landscape", "scenery"),
		("group", "crowd"),
		("officer", "military"),
		("uniform", "soldier"),
		("propeller", "aircraft"),
		("howitzer", "artillery"),
		("mortar", "cannon"),
		("ship", "navy"),
		("M1 Garand", "rifle"),
		("M4 Sherman", "tank"),
	]
	
	# CATEGORY 3: Distant Relations (BORDERLINE - could go either way)
	distant_relations = [
		("soldier", "uniform"),      # Related but different semantic types
		("aircraft", "propeller"),   # Part-whole relationship
		("building", "city"),        # Part-whole
		("photograph", "image"),     # Generic-specific
		("pilot", "uniform"),        # Associated but different
		("vehicle", "road"),         # Associated context
		("weapon", "military"),      # Associated domain
		("ship", "ocean"),           # Associated context
		("portrait", "photography"), # Type-of relationship
		("landscape", "nature"),     # Type-of relationship
		("officer", "sherlock"),     # Distant association
	]
	
	# CATEGORY 4: Unrelated Concepts (MUST NOT match)
	unrelated_concepts = [
		("soldier", "aircraft"),
		("building", "weapon"),
		("uniform", "landscape"),
		("pilot", "tank"),
		("photograph", "vehicle"),
		("officer", "ship"),
		("infantry", "scenery"),
		("army", "portrait"),
		("airplane", "clothing"),
		("structure", "gun"),
		("commander", "crowd"),
		("aviator", "armored vehicle"),
		("vessel", "image"),
		("car", "face"),
		("military", "camera"),
		("volcano", "shopping"),
	]
	
	# CATEGORY 5: Confusables (MUST NOT match - different but similar domain)
	confusables = [
		("soldier", "sailor"),       # Both military but different
		("aircraft", "helicopter"),  # Both aerial but different specificity
		("tank", "truck"),           # Both vehicles but very different
		("rifle", "pistol"),         # Both weapons but different
		("captain", "general"),      # Both ranks but different
		("fighter", "bomber"),       # Both aircraft types but different
		("navy", "army"),            # Both military branches but different
		("portrait", "landscape"),   # Both photo types but opposite
		("pilot", "driver"),         # Both operators but different
		("ship", "submarine"),       # Both naval but different
	]
	
	# COMPUTE SIMILARITIES FOR ALL CATEGORIES	
	def compute_similarities(pairs, category_name):
		"""Compute similarities and return scores with diagnostics."""
		scores = []
		details = []
		
		for w1, w2 in pairs:
			emb1 = model.encode(w1, convert_to_tensor=False)
			emb2 = model.encode(w2, convert_to_tensor=False)
			sim = float(1 - scipy.spatial.distance.cosine(emb1, emb2))
			scores.append(sim)
			details.append((w1, w2, sim))
		
		return {
			'scores': scores,
			'mean': float(np.mean(scores)),
			'std': float(np.std(scores)),
			'min': float(np.min(scores)),
			'max': float(np.max(scores)),
			'details': details,
			'category': category_name,
		}

	cat1_results = compute_similarities(pairs=direct_synonyms, category_name="Direct Synonyms")
	cat2_results = compute_similarities(pairs=related_concepts, category_name="Related Concepts")
	cat3_results = compute_similarities(pairs=distant_relations, category_name="Distant Relations")
	cat4_results = compute_similarities(pairs=unrelated_concepts, category_name="Unrelated Concepts")
	cat5_results = compute_similarities(pairs=confusables, category_name="Confusables")
	
	# DETAILED CATEGORY ANALYSIS
	if verbose:
		print("CATEGORY ANALYSIS")
		for cat_result in [cat1_results, cat2_results, cat3_results, cat4_results, cat5_results]:
			print(f"{cat_result['category']} (n={len(cat_result['scores'])}):")
			print(f"  Mean±Std: {cat_result['mean']:.4f}±{cat_result['std']:.4f}")
			print(f"  Range: [{cat_result['min']:.4f}, {cat_result['max']:.4f}]")
			
			# Show top-3 and bottom-3 examples
			sorted_details = sorted(cat_result['details'], key=lambda x: x[2], reverse=True)

			print(f"  Highest similarities:")
			for w1, w2, sim in sorted_details[:3]:
				print(f"    {w1:35} <-> {w2:35}{sim:.4f}")
			
			print(f"  Lowest similarities:")
			for w1, w2, sim in sorted_details[-3:]:
				print(f"    {w1:35} <-> {w2:35}{sim:.4f}")

			print()
	
	# OVERLAP ANALYSIS - Check for distribution overlap
	# Combine "should match" categories
	should_match_scores = cat1_results['scores'] + cat2_results['scores']
	should_match_mean = np.mean(should_match_scores)
	should_match_std = np.std(should_match_scores)
	
	# Combine "should NOT match" categories
	should_not_match_scores = cat4_results['scores'] + cat5_results['scores']
	should_not_match_mean = np.mean(should_not_match_scores)
	should_not_match_std = np.std(should_not_match_scores)
	
	# Calculate separation
	gap = should_match_mean - should_not_match_mean
	overlap_start = should_not_match_mean + should_not_match_std
	overlap_end = should_match_mean - should_match_std
	overlap_zone = max(0, overlap_start - overlap_end)
	
	if verbose:
		print("\nDISTRIBUTION OVERLAP ANALYSIS\n")
		print(f"Should MATCH (Synonyms + Related):")
		print(f"  Mean: {should_match_mean:.4f}")
		print(f"  Std:  {should_match_std:.4f}")
		print(f"  Range: [{np.min(should_match_scores):.4f}, {np.max(should_match_scores):.4f}]")
		
		print(f"\nShould NOT match (Unrelated + Confusables):")
		print(f"  Mean: {should_not_match_mean:.4f}")
		print(f"  Std:  {should_not_match_std:.4f}")
		print(f"  Range: [{np.min(should_not_match_scores):.4f}, {np.max(should_not_match_scores):.4f}]")
		
		print(f"\nSeparation Analysis:")
		print(f"  Gap between means: {gap:.4f}")
		print(f"  Overlap zone (±1σ): {overlap_zone:.4f}")
		
		if overlap_zone < 0.05:
			print(f"  ✅ Excellent separation (minimal overlap)")
		elif overlap_zone < 0.15:
			print(f"  ✅ Good separation")
		elif overlap_zone < 0.25:
			print(f"  ⚠️  Moderate separation")
		else:
			print(f"  ❌ Poor separation (significant overlap)")
	
	# THRESHOLD OPTIMIZATION - Test multiple strategies
	threshold_candidates = {
		'midpoint': (should_match_mean + should_not_match_mean) / 2,
		'mean_minus_1std': should_match_mean - should_match_std,
		'mean_minus_0.5std': should_match_mean - 0.5 * should_match_std,
		'optimal_f1': None,  # Will calculate below
	}
	
	# Find threshold that maximizes F1 on combined "should match" vs "should not match"
	range_ths = np.arange(0.15, 0.95, 0.01)
	best_f1 = 0
	best_th = 0.5
	if verbose:
		print(f"\nFinding optimal threshold ({len(range_ths)}) for F1 score on combined 'should match' vs 'should not...")
	for th in range_ths:
		# True positives: should_match scores >= threshold
		tp = sum(1 for s in should_match_scores if s >= th)

		# False positives: should_not_match scores >= threshold
		fp = sum(1 for s in should_not_match_scores if s >= th)

		# False negatives: should_match scores < threshold
		fn = sum(1 for s in should_match_scores if s < th)
		
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
		
		if f1 > best_f1:
			best_f1 = f1
			best_th = th
	
	threshold_candidates['optimal_f1'] = best_th
	
	if verbose:
		print("\nTHRESHOLD CANDIDATES\n")
		for name, thresh in threshold_candidates.items():
			if thresh is not None:
				print(f"{name:<25}{thresh:.4f}")
	
	# PERFORMANCE EVALUATION - Test each candidate threshold
	if verbose:
		print("\nPERFORMANCE AT CANDIDATE THRESHOLDS\n")
		
		for name, threshold in threshold_candidates.items():
			if threshold is None:
				continue
			
			# Evaluate on each category separately
			results_by_cat = {}
			
			for cat_name, cat_result in [
				("Direct Synonyms", cat1_results),
				("Related Concepts", cat2_results),
				("Distant Relations", cat3_results),
				("Unrelated", cat4_results),
				("Confusables", cat5_results),
			]:
				matches = sum(1 for s in cat_result['scores'] if s >= threshold)
				total = len(cat_result['scores'])
				pct = 100 * matches / total if total > 0 else 0
				results_by_cat[cat_name] = (matches, total, pct)
			
			print(f"\tTh: {threshold:.4f} ({name})")
			print(f"\tDirect Synonyms:   {results_by_cat['Direct Synonyms'][0]:2}/{results_by_cat['Direct Synonyms'][1]:2} matched ({results_by_cat['Direct Synonyms'][2]:5.1f}%) {'✅' if results_by_cat['Direct Synonyms'][2] >= 80 else '⚠️' if results_by_cat['Direct Synonyms'][2] >= 60 else '❌'}")
			print(f"\tRelated Concepts:  {results_by_cat['Related Concepts'][0]:2}/{results_by_cat['Related Concepts'][1]:2} matched ({results_by_cat['Related Concepts'][2]:5.1f}%) {'✅' if results_by_cat['Related Concepts'][2] >= 70 else '⚠️' if results_by_cat['Related Concepts'][2] >= 50 else '❌'}")
			print(f"\tDistant Relations: {results_by_cat['Distant Relations'][0]:2}/{results_by_cat['Distant Relations'][1]:2} matched ({results_by_cat['Distant Relations'][2]:5.1f}%) (ambiguous)")
			print(f"\tUnrelated:         {results_by_cat['Unrelated'][0]:2}/{results_by_cat['Unrelated'][1]:2} matched ({results_by_cat['Unrelated'][2]:5.1f}%) {'✅' if results_by_cat['Unrelated'][2] <= 20 else '⚠️' if results_by_cat['Unrelated'][2] <= 40 else '❌'}")
			print(f"\tConfusables:       {results_by_cat['Confusables'][0]:2}/{results_by_cat['Confusables'][1]:2} matched ({results_by_cat['Confusables'][2]:5.1f}%) {'✅' if results_by_cat['Confusables'][2] <= 30 else '⚠️' if results_by_cat['Confusables'][2] <= 50 else '❌'}")
			
			# Overall precision/recall
			tp = sum(1 for s in should_match_scores if s >= threshold)
			fp = sum(1 for s in should_not_match_scores if s >= threshold)
			fn = sum(1 for s in should_match_scores if s < threshold)
			tn = sum(1 for s in should_not_match_scores if s < threshold)
			
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
			accuracy = (tp + tn) / (tp + fp + fn + tn)
			
			print(f"\t\t→ Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}\n")
	
	# IDENTIFY PROBLEMATIC PAIRS - Find borderline cases	
	if verbose:
		print("\nPROBLEMATIC PAIRS (Near Decision Boundary)\n")
		
		recommended_threshold = threshold_candidates['optimal_f1']
		margin = 0.05
		
		print(f"Pairs near threshold {recommended_threshold:.4f} (±{margin}):")
		
		all_pairs_with_expected = []
		all_pairs_with_expected.extend([(w1, w2, sim, "SHOULD_MATCH") for w1, w2, sim in cat1_results['details']])
		all_pairs_with_expected.extend([(w1, w2, sim, "SHOULD_MATCH") for w1, w2, sim in cat2_results['details']])
		all_pairs_with_expected.extend([(w1, w2, sim, "AMBIGUOUS") for w1, w2, sim in cat3_results['details']])
		all_pairs_with_expected.extend([(w1, w2, sim, "SHOULD_NOT_MATCH") for w1, w2, sim in cat4_results['details']])
		all_pairs_with_expected.extend([(w1, w2, sim, "SHOULD_NOT_MATCH") for w1, w2, sim in cat5_results['details']])
		
		borderline_pairs = [
			(w1, w2, sim, expected) 
			for w1, w2, sim, expected in all_pairs_with_expected
			if abs(sim - recommended_threshold) < margin
		]
		
		borderline_pairs.sort(key=lambda x: abs(x[2] - recommended_threshold))
		
		if borderline_pairs:
			print(f"  Found {len(borderline_pairs)} borderline pairs:")
			for w1, w2, sim, expected in borderline_pairs[:10]:  # Show top 10
				decision = "MATCH" if sim >= recommended_threshold else "NO_MATCH"
				correct = "✅" if (decision == "MATCH" and "SHOULD_MATCH" in expected) or \
												(decision == "NO_MATCH" and "SHOULD_NOT_MATCH" in expected) else \
								 "⚠️" if expected == "AMBIGUOUS" else "❌"
				print(f"    {w1:20} <-> {w2:20}: {sim:.4f} → {decision:10} (expected: {expected:20}) {correct}")
		else:
			print(f"  ✅ No borderline pairs found (excellent separation!)")
	
	# FINAL RECOMMENDATION
	recommended_threshold = threshold_candidates['optimal_f1']
	if verbose:
		print("\nRECOMMENDATION\n")
		print(f"Optimal threshold: {recommended_threshold:.4f}")
		print(f"  Based on: Maximum F1 score on synonym/related vs unrelated/confusable pairs")
		print(f"  F1 score: {best_f1:.3f}")
		print(f"  Gap between distributions: {gap:.4f}")
		
		if best_f1 >= 0.90:
			print(f"  ✅ Excellent discriminative power")
		elif best_f1 >= 0.80:
			print(f"  ✅ Good discriminative power")
		elif best_f1 >= 0.70:
			print(f"  ⚠️  Moderate discriminative power")
		else:
			print(f"  ❌ Poor discriminative power - consider different model")
		print("-"*100)
	
	diagnostics = {
		'categories': {
			'direct_synonyms': cat1_results,
			'related_concepts': cat2_results,
			'distant_relations': cat3_results,
			'unrelated': cat4_results,
			'confusables': cat5_results,
		},
		'separation': {
			'should_match_mean': should_match_mean,
			'should_not_match_mean': should_not_match_mean,
			'gap': gap,
			'overlap_zone': overlap_zone,
		},
		'thresholds': threshold_candidates,
		'best_f1': best_f1,
		'borderline_pairs': borderline_pairs if verbose else None,
	}

	if verbose:
		print(f"Diagnostics for model {model_name}:")
		print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
	
	return recommended_threshold, diagnostics

def _precompute_label_embeddings(
	all_labels: List[str], 
	model: SentenceTransformer,
	batch_size: int = 512,
	verbose: bool = False
) -> Dict[str, np.ndarray]:
	"""
	Pre-compute embeddings for all unique labels to avoid redundant computation.
	
	Args:
			all_labels: List of all labels (may contain duplicates)
			model: SentenceTransformer model
			verbose: Print progress
	
	Returns:
			Dictionary mapping label -> embedding vector
	"""
	unique_labels = sorted(list(set(all_labels)))
	
	if verbose:
		print(f"\n[EMBEDDING] Pre-computing for {len(unique_labels):,} unique labels (batch_size={batch_size})")
	
	# Batch encode all unique labels
	embeddings = model.encode(
		unique_labels, 
		convert_to_tensor=False,
		batch_size=batch_size,
		normalize_embeddings=True,
		show_progress_bar=verbose,
	)
	
	# Create lookup dictionary
	emb_cache = {label: emb for label, emb in zip(unique_labels, embeddings)}
	
	if verbose:
		print(f"[DONE] Embeddings cached for {len(emb_cache):,} labels | Embedding: {embeddings[0].shape}")
	
	return emb_cache

def _make_normalized_getter(emb_cache: Dict[str, np.ndarray]):
		"""
		Lazy, memoized L2-normalized embedding lookup.

		Each label's vector is normalized once on first use (zero-vectors are
		left unchanged via guard), so per-sample similarity computation is a
		single matrix product of unit vectors.
		"""
		memo: Dict[str, np.ndarray] = {}

		def get(label: str) -> np.ndarray:
				e = memo.get(label)
				if e is None:
						v = np.asarray(emb_cache[label], dtype=float)
						nrm = np.linalg.norm(v)
						e = v / nrm if nrm > 0 else v
						memo[label] = e
				return e

		return get

def _cosine_sim_matrix(
		a_labels: List[str],
		b_labels: List[str],
		get_emb,
) -> np.ndarray:
		"""Vectorized |A| x |B| cosine similarity, clipped to [-1, 1]."""
		A = np.stack([get_emb(l) for l in a_labels])
		B = np.stack([get_emb(l) for l in b_labels])
		return np.clip(A @ B.T, -1.0, 1.0)

def _sample_semantic_scores(
	sim: np.ndarray,
	threshold: float,
	card_priority: Optional[float] = None,
) -> Tuple[float, float, List[Tuple[int, int]], np.ndarray]:
	"""
	Per-sample scores. Returns (j_match, j_upper, matched_pairs, feasible).
	j_match (rigorous):
			One-to-one maximum-cardinality matching over edges with sim >= threshold
			(similarity used only as tie-breaker). Intersection = |M|, so all
			Jaccard axioms hold: |M| <= min(|A|, |B|); J = 1 iff a perfect one-to-one
			correspondence covers both sets; reduces to exact set-theoretic Jaccard
			when only identical strings exceed the threshold.
	j_upper (upper bound):
			The previous thresholded bidirectional-coverage overlap, retained ONLY
			as a provable upper bound (j_match <= j_upper always) for interval
			reporting. Not a Jaccard index (intersection can exceed min(|A|,|B|)).
	Cardinality-first guarantee:
			A k-edge matching must always outweigh any (k-1)-edge matching.
			Worst case requires C > k(1-thr) - 1 for all k <= min(m, n), so
			C = min(m, n) + 1 is safe for ANY threshold and ANY set size.
			(A fixed C=10 is only safe up to ~36 edges at thr=0.7.)
	"""

	m, n = sim.shape
	feasible = sim >= threshold
	if card_priority is None:
		card_priority = float(min(m, n)) + 1.0

	# Padded LAP: zero blocks = "leave node unmatched"; -1e9 forbids
	# below-threshold pairs; feasible edges get card_priority + sim so the
	# solver maximizes cardinality first, similarity second.
	W = np.zeros((m + n, m + n))
	W[:m, :n] = np.where(feasible, card_priority + sim, -1e9)
	row_ind, col_ind = scipy.optimize.linear_sum_assignment(W, maximize=True)

	matched_pairs = [
		(int(i), int(j)) for i, j in zip(row_ind, col_ind)
		if i < m and j < n and feasible[i, j]
	]
	matched = len(matched_pairs)
	union_m = m + n - matched
	j_match = matched / union_m if union_m > 0 else 0.0

	# Upper bound (previous heuristic, renamed)
	m_ab = int(np.sum(feasible.any(axis=1)))   # A-labels with >= 1 match
	m_ba = int(np.sum(feasible.any(axis=0)))   # B-labels with >= 1 match

	I = (m_ab + m_ba) / 2.0
	union_u = m + n - I
	j_upper = I / union_u if union_u > 0 else 0.0

	return j_match, j_upper, matched_pairs, feasible

def _trace_sample(
	idx: int,
	a_labels: List[str],
	b_labels: List[str],
	sim: np.ndarray,
	feasible: np.ndarray,
	matched_pairs: List[Tuple[int, int]],
	j_match: float,
	j_upper: float,
	threshold: float,
) -> None:
	"""Detailed per-sample debug trace (verbose mode only)."""
	print(f"\n[trace] sample #{idx}: A={a_labels} B={b_labels}")
	with np.printoptions(precision=3, suppress=True, linewidth=120):
		print(f"cosine sim matrix:\n{sim}")
		print(f"feasible mask (sim >= {threshold}):\n{feasible.astype(int)}")

	edges = [
		(a_labels[i], b_labels[j], float(sim[i, j]))
		for i in range(sim.shape[0]) for j in range(sim.shape[1])
		if feasible[i, j]
	]
	print(f"feasible edges : {edges}")
	print(f"1-to-1 matching: {[(a_labels[i], b_labels[j]) for i, j in matched_pairs]}")
	print(f"J_match={j_match:.3f} | J_upper={j_upper:.3f}")
	print("."*40)

def _semantic_jaccard_cached(
	sets_a: List[Set[str]],
	sets_b: List[Set[str]],
	emb_cache: Dict[str, np.ndarray],
	threshold: float = 0.7,
	verbose: bool = False,
	max_traced_samples: int = 10,
	return_per_sample: bool = False,
) -> Dict:
	"""
	Semantic label agreement via thresholded cosine matching.
	Per sample, two scores are computed:
		- j_match : rigorous one-to-one matching Jaccard (lower bound)
		- j_upper : thresholded bidirectional coverage overlap (upper bound)
	Denominator accounting (no silent skips):
		- both label sets empty          -> excluded, counted in n_both_empty
		- exactly one set empty          -> excluded from PRIMARY mean (identical
			denominator to the original function; backward compatible), counted in
			n_one_empty; the zero-policy mean (counting them as full disagreement)
			is ALWAYS reported as a sensitivity analysis
		- labels exist but none embeddable -> excluded, counted in n_missing_emb
	Returns dict with keys:
		'jaccard_matching', 'jaccard_coverage_upper'          (primary, skip policy)
		'jaccard_matching_zero_policy',
		'jaccard_coverage_upper_zero_policy'                  (sensitivity)
		'median_matching', 'threshold',
		'n_total', 'n_scored', 'n_both_empty', 'n_one_empty',
		'n_missing_emb', 'n_many_to_many'
		(+ 'per_sample_matching' / 'per_sample_upper' if return_per_sample)
	"""

	if len(sets_a) != len(sets_b):
		raise ValueError(f"Input length mismatch: {len(sets_a)} vs {len(sets_b)}")

	if not 0.0 < threshold <= 1.0:
		raise ValueError(f"threshold must be in (0, 1], got {threshold}")

	n_total = len(sets_a)
	get_emb = _make_normalized_getter(emb_cache)
	if verbose:
		dim = np.asarray(next(iter(emb_cache.values()))).shape if emb_cache else None
		print("=" * 78)
		print(f"[sem-jaccard]")
		print(f"  ├─ n_samples={n_total}")
		print(f"  ├─ threshold={threshold}")
		print(f"  ├─ card_priority=adaptive(min(m,n)+1)")
		print(f"  ├─ emb_cache: {len(emb_cache)} labels dim={dim}")
		uniq_a: Set[str] = set()
		uniq_b: Set[str] = set()

		for a in sets_a:
			uniq_a.update(a)

		for b in sets_b:
			uniq_b.update(b)

		miss_a = uniq_a - emb_cache.keys()
		miss_b = uniq_b - emb_cache.keys()

		print(f"  ├─ coverage A: {len(uniq_a) - len(miss_a)}/{len(uniq_a)} labels embeddable, missing {sorted(miss_a)[:5]}")
		print(f"  └─ coverage B: {len(uniq_b) - len(miss_b)}/{len(uniq_b)} labels embeddable, missing: {sorted(miss_b)[:5]}")

	# Main loop                                                          #
	scores_match: List[float] = []
	scores_upper: List[float] = []
	n_both_empty = n_one_empty = n_missing_emb = n_many_to_many = 0
	traced = 0
	missing_examples: List[str] = []
	step = max(1, n_total // 10)
	for idx, (raw_a, raw_b) in enumerate(zip(sets_a, sets_b)):
		a, b = set(raw_a), set(raw_b)
		if not a and not b: # uninformative: exclude
			n_both_empty += 1
			continue
		if not a or not b: # tracked; excluded from primary
			n_one_empty += 1
			continue

		a_labels = sorted(l for l in a if l in emb_cache)
		b_labels = sorted(l for l in b if l in emb_cache)
		if len(missing_examples) < 10:
			for l in (a | b) - emb_cache.keys():
				if l not in missing_examples:
					missing_examples.append(l)
				if len(missing_examples) >= 10:
					break

		if not a_labels or not b_labels:          # cannot score: report, don't absorb
			n_missing_emb += 1
			continue

		sim = _cosine_sim_matrix(a_labels, b_labels, get_emb)
		j_m, j_u, matched_pairs, feasible = _sample_semantic_scores(sim, threshold)

		if j_u - j_m > 1e-9:                      # many-to-many case: bounds diverge
			n_many_to_many += 1

		scores_match.append(j_m)
		scores_upper.append(j_u)
		if verbose and traced < max_traced_samples:
			traced += 1
			_trace_sample(
				idx, 
				a_labels, 
				b_labels, 
				sim, 
				feasible,
				matched_pairs, 
				j_m, 
				j_u, 
				threshold
			)
		if verbose and (idx + 1) % step == 0:
			print(
				f"[sem-jaccard] {idx + 1:7d}/{n_total} "
				f"scored: {len(scores_match):<10}one_empty: {n_one_empty:<10}"
				f"missing_emb: {n_missing_emb}"
			)

	# Aggregation: BOTH denominators reported; no silent default         #
	n_scored = len(scores_match)
	sum_m = float(np.sum(scores_match)) if n_scored else 0.0
	sum_u = float(np.sum(scores_upper)) if n_scored else 0.0
	n_zero = n_scored + n_one_empty
	result = {
		# primary: same skip rules as the original function (backward compatible)
		'jaccard_matching': sum_m / n_scored if n_scored else 0.0,
		'jaccard_coverage_upper': sum_u / n_scored if n_scored else 0.0,
		# sensitivity analysis: one-empty samples counted as full disagreement
		'jaccard_matching_zero_policy': sum_m / n_zero if n_zero else 0.0,
		'jaccard_coverage_upper_zero_policy': sum_u / n_zero if n_zero else 0.0,
		'median_matching': float(np.median(scores_match)) if n_scored else 0.0,
		'threshold': float(threshold),
		'n_total': n_total,
		'n_scored': n_scored,
		'n_both_empty': n_both_empty,
		'n_one_empty': n_one_empty,
		'n_missing_emb': n_missing_emb,
		'n_many_to_many': n_many_to_many,
	}

	if return_per_sample:
		result['per_sample_matching'] = scores_match
		result['per_sample_upper'] = scores_upper

	if verbose:
		print("-" * 80)
		print(f"[sem-jaccard] SUMMARY | threshold={threshold}")
		print(
			f"  denominator accounting: total={n_total} scored={n_scored} "
			f"both_empty={n_both_empty} one_empty={n_one_empty} "
			f"missing_emb={n_missing_emb}"
		)
		if missing_examples:
			print(f"  example labels without embedding: {missing_examples}")
		if n_scored:
			print(
				f"  many-to-many samples (J_upper > J_match): {n_many_to_many} "
				f"({100.0 * n_many_to_many / n_scored:.1f}% of scored)"
			)
		print(f"  primary (skip policy; backward-compatible denominator):")
		print(
			f"    J_match = {result['jaccard_matching']:.4f} "
			f"(median {result['median_matching']:.4f})"
		)
		print(f"    J_upper = {result['jaccard_coverage_upper']:.4f}")
		print(f"  sensitivity (one-empty counted as 0 disagreement):")
		print(
			f"    J_match = {result['jaccard_matching_zero_policy']:.4f} | "
			f"J_upper = {result['jaccard_coverage_upper_zero_policy']:.4f}"
		)
		print(
			f"  reported agreement interval: "
			f"[{result['jaccard_matching']:.4f}, {result['jaccard_coverage_upper']:.4f}]"
		)
		print("=" * 80)

	return result

@torch.no_grad()
def compute_clip_visual_grounding(
	df: pd.DataFrame,
	sources: List[str],
	norm_stats: Dict[str, List[float]],
	device: str,
	architecture: str = "ViT-B/32",
	batch_size: int = 64,
	label_chunk_size: int = 512,
	max_failure_rate: float = 0.1,
	verbose: bool = False,
) -> Dict[str, Dict]:
	"""
	Independent replacement for AXIS 2. Never compares one label source's
	text against another's — scores each source against the actual image
	pixels via frozen CLIP, so no source can be a reference for another.
	Grounding is a percentile rank, not raw cosine similarity: for each
	(image, label) pair actually assigned by a source, we ask what
	fraction of ALL images in the corpus this label matches worse than
	its true image. This cancels out a label's baseline CLIP similarity
	(abstract vs. concrete vocabulary sits at different baseline levels
	regardless of grounding quality), which a raw mean would conflate
	with actual grounding.
	Images that fail to load are excluded entirely from both the ranking
	pool and the scored pairs — never included as zero-vectors, since a
	zero vector has a specific, non-neutral dot product with every label
	and would silently bias every label's rank. If more than
	`max_failure_rate` of images fail to load, this raises rather than
	silently reporting numbers computed over an unverified subset.
	Similarity/rank computation is chunked over the label dimension, so
	the full [n_valid_images, n_unique_labels] matrix is never held more
	than once at a time — peak memory is O(n_valid_images) plus one
	chunk, not three full-size matrices simultaneously.
	Returns a dict mapping source column -> {'visual_grounding_clip': float, 'n_scored_pairs': int}
	"""
	image_path_col = "img_path"
	assert image_path_col in df.columns, f"Column {image_path_col} not found in DataFrame: available columns are {df.columns}"
	print(f">> CLIP Model Architecture: {architecture}...")
	try:
		model_config = clip.get_config(architecture=architecture)
		if verbose:
			print(json.dumps(model_config, indent=4, ensure_ascii=False))
		model, _ = clip.load(name=architecture, device=device)
		preprocess = clip.get_preprocess(
			norm_stats=norm_stats,
			input_resolution=model_config["image_resolution"],
		)
	except Exception as e:
			print(f"[Warning] Custom CLIP preprocess failed ({e}). Falling back to standard OpenAI CLIP preprocess.")
			import clip as openai_clip
			model, preprocess = openai_clip.load(name=architecture, device=device)
	model.name = architecture
	model_name = model.__class__.__name__
	print(f"[LOADED] {model_name} {model.name} ({device})")
	model.eval()

	# 1. Collect all unique labels across all provided sources
	all_labels = sorted({l for col in sources for cell in df[col] for l in _parse_label_cell(cell)})
	label_to_idx = {l: i for i, l in enumerate(all_labels)}
	n_labels = len(all_labels)

	# 2. Encode all labels
	print(f"Encoding {n_labels} unique labels (batch_size={batch_size})")
	t0 = time.time()
	text_embeds = []
	for i in range(0, n_labels, batch_size):
		toks = clip.tokenize(all_labels[i:i + batch_size]).to(device)
		e = torch.nn.functional.normalize(model.encode_text(toks), dim=-1)
		text_embeds.append(e.cpu())
	text_embeds = torch.cat(text_embeds, dim=0)  # [n_labels, dim]
	print(f"[ELAPSED] {time.time() - t0:.1f}sec for {n_labels} labels: {text_embeds.shape}")

	# 3. Encode all images. Track which rows actually produced a real
	#    embedding — a failed row stays untouched rather than becoming
	#    a zero-vector "image" that participates in every label's rank.
	n_rows = len(df)
	image_embeds = torch.zeros(n_rows, text_embeds.shape[1])
	valid_mask = torch.zeros(n_rows, dtype=torch.bool)
	print(f"Encoding {n_rows} images (batch size={batch_size})")
	t0 = time.time()
	for start in range(0, n_rows, batch_size):
		paths = df[image_path_col].iloc[start:start + batch_size].tolist()
		imgs, local_valid = [], []
		for local_idx, p in enumerate(paths):
			try:
				if os.path.exists(p):
					imgs.append(preprocess(Image.open(p).convert("RGB")))
					local_valid.append(local_idx)
			except Exception as e:
				print(f"Error loading {p}: {e}")
		if not imgs:
			continue

		imgs = torch.stack(imgs).to(device)
		e = torch.nn.functional.normalize(model.encode_image(imgs), dim=-1)

		for batch_pos, local_idx in enumerate(local_valid):
			row = start + local_idx
			image_embeds[row] = e[batch_pos].cpu()
			valid_mask[row] = True

	n_valid = int(valid_mask.sum().item())
	n_failed = n_rows - n_valid
	failure_rate = n_failed / n_rows if n_rows else 0.0

	print(f"Image loading: {n_valid}/{n_rows} succeeded ({n_failed} failed, {failure_rate*100:.2f}%)")
	print(f"[ELAPSED: {time.time() - t0:.1f}sec for {n_rows} images]")
	if failure_rate > max_failure_rate:
		raise RuntimeError(
			f"{n_failed} images failed to load: "
			f"{failure_rate*100:.2f}% > max_failure_rate={max_failure_rate*100:.2f}%. " 
			f"Check image paths against "
			f"this machine's filesystem before trusting grounding scores. "
			f"(Common cause: absolute paths recorded on a different machine than this is running on.)"
		)

	if n_valid < 2:
		raise RuntimeError(f"Only {n_valid} valid images — cannot compute a meaningful percentile rank.")

	valid_idx = torch.where(valid_mask)[0]
	image_embeds_valid = image_embeds[valid_idx]  # [n_valid, dim] — real embeddings only

	# 4. Build (valid_row_position, label_idx) pairs per source, ONCE,
	#    skipping any row whose image failed to load, then sort by label
	#    index so they can be swept through in lockstep with the label
	#    chunks below in a single linear pass.
	row_to_valid_pos = {int(row.item()): pos for pos, row in enumerate(valid_idx)}
	source_pairs = {col: [] for col in sources}
	for col in sources:
		for row_idx, cell in enumerate(df[col]):
			if row_idx not in row_to_valid_pos:
				continue  # image for this row failed to load — never scored

			idxs = [label_to_idx[l] for l in _parse_label_cell(cell) if l in label_to_idx]

			if idxs:
				pos = row_to_valid_pos[row_idx]
				source_pairs[col].extend((pos, label_idx) for label_idx in idxs)

		source_pairs[col].sort(key=lambda pr: pr[1])

	# 5. Rank/percentile computation, chunked over the label dimension.
	print(
		f"Computing similarity + percentile ranks in chunks of {label_chunk_size} labels "
		f"({n_valid} valid images x {n_labels} labels)..."
	)
	sum_percentile = {col: 0.0 for col in sources}
	cursor = {col: 0 for col in sources}

	for chunk_start in range(0, n_labels, label_chunk_size):
		chunk_end = min(chunk_start + label_chunk_size, n_labels)
		text_chunk = text_embeds[chunk_start:chunk_end]                   # [C, dim]
		sim_chunk = image_embeds_valid @ text_chunk.T                     # [n_valid, C]
		ranks_chunk = sim_chunk.argsort(dim=0).argsort(dim=0).float()     # [n_valid, C]
		pct_chunk = ranks_chunk / (n_valid - 1)                           # [n_valid, C]
		for col in sources:
			pairs = source_pairs[col]
			j = cursor[col]
			in_chunk = []
			while j < len(pairs) and pairs[j][1] < chunk_end:
				pos, lbl = pairs[j]
				in_chunk.append((pos, lbl - chunk_start))
				j += 1
			cursor[col] = j
			if in_chunk:
				pos_t = torch.tensor([p for p, _ in in_chunk], dtype=torch.long)
				lbl_t = torch.tensor([l for _, l in in_chunk], dtype=torch.long)
				sum_percentile[col] += pct_chunk[pos_t, lbl_t].sum().item()
		if verbose:
			print(f"  labels [{chunk_start}:{chunk_end}) done")

	results = {}

	print(f"\n[VISUAL GROUNDING] {model_name} {model.name}")

	for col in sources:
		n_scored = len(source_pairs[col])
		mean_percentile = sum_percentile[col] / n_scored if n_scored else 0.0
		results[col] = {
			"visual_grounding_clip": float(mean_percentile),
			"n_scored_pairs": n_scored,
		}
		print(f"{col:<30} G={mean_percentile:.4f} over {n_scored} pairs")

	del image_embeds, image_embeds_valid, text_embeds
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return results

def get_embedding_model(embedding_model_id: str, device: str="cuda", verbose: bool=False):
	if verbose:
		print(f"STEP 2: Loading embedding model {embedding_model_id} on {device}")

	dtype = torch.float32
	if torch.cuda.is_available():
		dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
	if verbose:
		print(f"[INFO] {embedding_model_id} Dtype selection: {dtype}")
	def _optimal_attn_impl() -> str:
		if not torch.cuda.is_available():
			return "eager"
		major, minor = torch.cuda.get_device_capability()
		compute_cap = major + minor / 10
		if compute_cap >= 8.0:
			try:
				import flash_attn
				if verbose:
					print(f"[INFO] Flash Attention 2 available (compute {compute_cap})")
				return "flash_attention_2"
			except ImportError:
				if verbose:
					print(f"[WARN] Flash Attention 2 not installed (pip install flash-attn)")
		if compute_cap >= 7.0 and torch.__version__ >= "2.0.0":
			if verbose:
				print(f"[INFO] Using SDPA attention (compute {compute_cap}, PyTorch {torch.__version__})")
			return "sdpa"
		if verbose:
			print(f"[INFO] Using eager attention (compute {compute_cap})")
		return "eager"

	attn_impl = _optimal_attn_impl()

	if verbose:
		print(f"[INFO] {embedding_model_id} with {attn_impl} attention")

	model_kwargs = {}
	if "Qwen" in embedding_model_id:
		model_kwargs = {
			"attn_implementation": attn_impl,
			"torch_dtype": dtype,
		}

	model = SentenceTransformer(
		model_name_or_path=embedding_model_id,
		trust_remote_code=True,
		cache_folder=cache_directory.get(os.getenv('USER'), None),
		model_kwargs=model_kwargs,
		token=os.getenv("HUGGINGFACE_TOKEN"),
		tokenizer_kwargs={"padding_side": "left"},
	).to(device)

	if verbose:
		total_params = sum(p.numel() for p in model.parameters())
		print(f"[LOADED] {embedding_model_id} with {total_params:,} parameters")

	return model

def get_cgd_taxonomy_supervision(
	df: pd.DataFrame,
	output_directory: str,
	embedding_model_id: str,
	architecture: str,
	device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
	norm_stats: Dict[str, List[float]]=None,
	anchor_column: str = "vlm_canonical_labels",
	semantic_threshold: Optional[float] = None,
	base: float = 2.0,
	normalize: str = "L2",
	verbose: bool = False,
) -> pd.DataFrame:
	"""
	Compute Coverage-Grounding-Density taxonomy for multi-source supervision.
	
	THREE AXES:
	
	1. SEMANTIC COVERAGE (Axis 1):
		 - Measures vocabulary richness via perplexity = 2^H (entropy)
		 - Higher = more diverse/expressive vocabulary
		 - Example: 500 means "effective vocabulary of 500 concepts"
	
	2. VISUAL GROUNDING (Axis 2):
		 - Measures semantic overlap with VLM-generated labels (visual anchor)
		 - Uses embedding similarity, NOT string matching
		 - Higher = better captures what's visually present in images
		 - Example: 0.65 means "65% semantic agreement with visual content"
	
	3. STATISTICAL DENSITY (Axis 3):
		 - Measures label concentration/reusability
		 - Formula: (avg occurrences per label) × (1 - singleton_rate)
		 - Higher = labels appear frequently, lower = sparse/rare labels
		 - Example: 100 means labels appear ~100 times on average
	
	Args:
			df: DataFrame with label columns
			output_directory: Where to save visualizations
			sources: List of label columns to analyze (default: LLM/VLM/Multimodal)
			anchor_column: Reference column for visual grounding (default: VLM)
			semantic_threshold: Cosine similarity threshold for label equivalence (auto-tuned if None)
			base: Logarithm base for entropy (2.0 = bits, math.e = nats)
			normalize: Normalization method ('L2', 'minmax', 'zscore', 'none')
			verbose: Print detailed progress
	
	Returns:
			DataFrame with raw and normalized scores for each source
	"""
	
	# Setup and Validation
	if verbose:
		print("\n[CGD TAXONOMY] Coverage-Grounding-Density Analysis")
		print(df.info(verbose=verbose, memory_usage="deep"))
		# print(df.head(5).to_string(index=False))
	
	if norm_stats is None:
		norm_stats = {"mean": [0.52, 0.50, 0.48], "std": [0.27, 0.27, 0.26]}

	# Use default sources if not specified
	sources = ["llm_canonical_labels", "vlm_canonical_labels", "multimodal_canonical_labels"]
	for i, s in enumerate(sources):
		if s not in df.columns:
			print(f"\n[WARNING] Missing required columns: {s}\nAvailable columns: {list(df.columns)} => skipping")
			return
	
	if verbose:
		print(f"\nConfiguration:")
		print(f"  df: {df.shape}")
		print(f"  Embedding model: {embedding_model_id}")
		print(f"  {architecture} normalization stats: {norm_stats}")
		print(f"  Sources to analyze: {sources}")
		print(f"  Visual anchor: {anchor_column}")
		print(f"  Entropy base: {base} ({'bits' if base == 2.0 else 'nats' if base == math.e else 'units'})")
		print(f"  Normalization: {normalize}")
	
	# Validate columns exist
	required_cols = [anchor_column] + sources
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	
	# STEP 1: Parse Label Sets and Collect All Unique Labels
	if verbose:
		print("\nSTEP 1: Parsing label sets")
	
	parsed_sets: Dict[str, List[Set[str]]] = {}
	all_labels_for_embedding = []
	
	for col in required_cols:
		# Parse each cell into a set of labels
		parsed_sets[col] = [set(_parse_label_cell(v)) for v in df[col].tolist()]
		
		# Collect all labels from this column
		col_labels = [label for label_set in parsed_sets[col] for label in label_set]
		all_labels_for_embedding.extend(col_labels)
		
		if verbose:
			total_labels = len(col_labels)
			avg_labels = total_labels / len(df) if len(df) > 0 else 0
			unique_labels = len(set(col_labels))

			print(f"{col}:")
			print(f"  ├─ Total labels: {total_labels:,}")
			print(f"  ├─ Unique labels: {unique_labels:,}")
			print(f"  └─ Avg per sample: {avg_labels:.2f}\n")

	# STEP 2: Pre-compute Embeddings for Semantic Grounding
	model = get_embedding_model(
		embedding_model_id=embedding_model_id, 
		device=device, 
		verbose=verbose
	)
	
	# Auto-calibrate threshold if not provided
	if semantic_threshold is None:
		semantic_threshold, _ = auto_calibrate_semantic_threshold(model=model, verbose=verbose)
	else:
		if verbose:
			print(f">> Using provided semantic threshold: {semantic_threshold}")

	# STEP 3: Compute Three Axes (CGD) for Each Source
	if verbose:
		print(f"\nSTEP 3: CGD metrics (sources: {sources})")

	# Pre-compute embeddings for ALL unique labels across all sources
	emb_cache = _precompute_label_embeddings(
		all_labels=all_labels_for_embedding,
		model=model,
		verbose=verbose,
	)

	clip_grounding = compute_clip_visual_grounding(
		df=df,
		sources=sources,
		norm_stats=norm_stats,
		architecture=architecture,
		device=device,
		verbose=verbose,
	)

	# anchor embeddings for cross-modal concordance:
	anchor_sets = parsed_sets[anchor_column]
	results = []
	for source_idx, source_col in enumerate(sources, 1):
		if verbose:
			print(f"\n[{source_idx}/{len(sources)}] Analyzing: {source_col}")
			print("-" * 80)
		
		# Collect label statistics for this source
		all_labels_in_source = []
		for label_set in parsed_sets[source_col]:
			all_labels_in_source.extend(list(label_set))

		counts = Counter(all_labels_in_source)
		total_occurrences = sum(counts.values())

		unique_labels = len(counts)

		singletons = sum(1 for count in counts.values() if count == 1)
		singleton_rate = (singletons / unique_labels) if unique_labels > 0 else 0.0
		
		if verbose:
			print(f"Label Statistics:")
			print(f"\tTotal occurrences: {total_occurrences:,}")
			print(f"\tUnique labels: {unique_labels:,}")
			print(f"\tSingletons: {singletons:,} ({singleton_rate*100:.1f}%)")
			
			# top-3 most frequent
			if counts:
				top_3 = counts.most_common(3)
				print(f"\tTop-3 labels:")
				for rank, (label, count) in enumerate(top_3, 1):
					pct = (count / total_occurrences * 100) if total_occurrences > 0 else 0
					print(f"\t\t{rank:<5}{label:<30}{count:<7}({pct:.2f}%)")
		
		# AXIS 1: SEMANTIC COVERAGE (Vocabulary Richness)
		H = _shannon_entropy(counts, base=base)
		perplexity = (base ** H) if H > 0 else 1.0
		semantic_coverage = perplexity
		if verbose:
			print(f"\n  📚 AXIS 1 - Semantic Coverage (C):")
			print(f"    Shannon entropy: {H:.3f} {'bits' if base == 2.0 else 'units'}")
			print(f"    Perplexity: {perplexity:.1f}")
			print(f"    Interpretation: Effective vocabulary of ~{int(perplexity)} concepts")
		
		# AXIS 2: VISUAL GROUNDING (via CLIP)
		if verbose:
			print(f"\n  👁️  AXIS 2 - Visual Grounding (G)")

		visual_grounding = clip_grounding[source_col]["visual_grounding_clip"]
		# descriptive measure of inter-source agreement: Cross-Modal Concordance
		vlm_concordance = _semantic_jaccard_cached(
			sets_a=parsed_sets[source_col],
			sets_b=anchor_sets,
			emb_cache=emb_cache,
			threshold=semantic_threshold,
			verbose=verbose,
		)["jaccard_matching"]

		if verbose:
			print(f"    Interpretation: {visual_grounding*100:.1f}% semantic agreement")
			print(f"    Cross-Modal Concordance (vs VLM): {vlm_concordance:.4f}")


		# # AXIS 2: VISUAL GROUNDING (Semantic Overlap with VLM)
		# if verbose:
		# 	print(f"\n  👁️  AXIS 2 - Visual Grounding (G): Semantic overlap with VLM [th: {semantic_threshold:.4f}]")

		# if source_col == anchor_column:
		# 	# Self-comparison = perfect grounding
		# 	visual_grounding = 1.0
		# else:
		# 	# Compute SEMANTIC Jaccard (not string matching!)
		# 	visual_grounding = _semantic_jaccard_cached(
		# 		sets_a=parsed_sets[source_col],
		# 		sets_b=anchor_sets,
		# 		emb_cache=emb_cache,
		# 		threshold=semantic_threshold,
		# 		verbose=verbose
		# 	)["jaccard_matching"]
		# if verbose:
		# 	print(f"    Semantic overlap with {anchor_column}: {visual_grounding:.4f} ({visual_grounding*100:.1f}%)")
		# 	print(f"    Interpretation: {visual_grounding*100:.1f}% semantic agreement with the VLM anchor: {anchor_column}")
		
		# AXIS 3: STATISTICAL DENSITY (Label Concentration)
		avg_occurrences_per_label = (total_occurrences / unique_labels) if unique_labels > 0 else 0.0
		statistical_density = avg_occurrences_per_label * (1.0 - singleton_rate)
		
		if verbose:
			print(f"\n  📊 AXIS 3 - Statistical Density (D):")
			print(f"    Avg occurrences per label: {avg_occurrences_per_label:.2f}")
			print(f"    Non-singleton rate: {(1.0 - singleton_rate)*100:.1f}%")
			print(f"    Statistical density: {statistical_density:.2f}")
			
			# Interpretation
			if statistical_density < 2.0:
				interpretation = "LOW (very sparse, many rare labels)"
			elif statistical_density < 5.0:
				interpretation = "MODERATE (balanced distribution)"
			elif statistical_density < 20.0:
				interpretation = "HIGH (concentrated, frequently reused labels)"
			else:
				interpretation = "VERY HIGH (highly concentrated vocabulary)"
			print(f"    Interpretation: {interpretation}")
		
		# Store results for this source
		results.append({
			"source": source_col,
			"unique_labels": int(unique_labels),
			"total_occurrences": int(total_occurrences),
			"singletons": int(singletons),
			"singleton_rate": float(singleton_rate),
			"semantic_coverage_raw": float(semantic_coverage),
			"visual_grounding_raw": float(visual_grounding),
			"statistical_density_raw": float(statistical_density),
		})
	
	# STEP 4: Create DataFrame and Normalize Scores
	scores_df = pd.DataFrame(results)
	if verbose:
		print(f"\n{'='*80}")
		print("STEP 4: Raw scores (before normalization)")
		print("="*80)
		print(scores_df.to_string(index=False))
	
	# Define axis columns
	axis_cols = [
		"semantic_coverage_raw",
		"visual_grounding_raw",
		"statistical_density_raw"
	]
	
	# Apply normalization
	if verbose:
		print(f"\n{'='*80}")
		print(f"STEP 5: Applying {normalize} normalization")
		print("="*80)
	
	for col in axis_cols:
		values = scores_df[col].to_numpy(dtype=float)
		norm_col_name = col.replace("_raw", f"_{normalize}_norm")
		
		if normalize == "minmax":
				vmin, vmax = np.min(values), np.max(values)
				diff = vmax - vmin
				normalized = (values - vmin) / diff if diff > 1e-12 else np.full_like(values, 0.5)
				
				if verbose:
						print(f"\n  {col}:")
						print(f"    Range: [{vmin:.4f}, {vmax:.4f}]")
						print(f"    Normalized: {normalized}")
		
		elif normalize == "zscore":
				mean, std = np.mean(values), np.std(values)
				normalized = (values - mean) / std if std > 1e-12 else np.full_like(values, 0.5)
				
				if verbose:
						print(f"\n  {col}:")
						print(f"    Mean: {mean:.4f}, Std: {std:.4f}")
						print(f"    Normalized: {normalized}")
		
		elif normalize == "L2":
				norm = np.linalg.norm(values)
				normalized = values / norm if norm > 1e-12 else np.full_like(values, 0.0)
				
				if verbose:
						print(f"\n  {col}:")
						print(f"    L2 norm: {norm:.4f}")
						print(f"    Normalized: {normalized}")
		
		else:  # none or unknown
				vmin, vmax = float(np.min(values)), float(np.max(values))
				normalized = (values - vmin) / (vmax - vmin) if (vmax - vmin) > 1e-12 else np.full_like(values, 0.5)
				
				if verbose:
						print(f"\n  {col}: Using minmax for visualization")
		
		scores_df[norm_col_name] = normalized
	
	# STEP 5: Visualizations
	viz_dir = os.path.join(output_directory, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	if verbose:
		print(f"\n{'='*80}")
		print(f"STEP 6: Generating radar plots: {viz_dir}")
		print("="*80)
			
	# Plot 1: Raw scores
	viz.plot_taxonomy_radar(
		scores_df,
		value_cols=axis_cols,
		title="Supervision Taxonomy (Raw Scores)",
		output_path=os.path.join(viz_dir, "taxonomy_radar_raw.png")
	)
			
	# Plot 2: Normalized scores
	norm_cols = [col.replace("_raw", f"_{normalize}_norm") for col in axis_cols]
	
	viz.plot_taxonomy_radar(
		scores_df,
		value_cols=norm_cols,
		title=f"Supervision Taxonomy ({normalize} Normalized)",
		output_path=os.path.join(viz_dir, f"taxonomy_radar_{normalize}_normalized.png")
	)
					
	if verbose:
		print("\nFINAL RESULTS\n")
		print(scores_df)
	
	return scores_df

def entropy_vs_performance(
	df: pd.DataFrame,
	performance: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
	label_columns: Optional[List[str]] = None,
	base: float = 2.0,
	verbose: bool = False,
	perf_strategy: Optional[str] = None,
	perf_k: Union[int, str] = 10,
	perf_reference_source: str = "multimodal_canonical_labels",
	perf_include_recall: bool = False,
	perf_tier_alias: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
	"""
	Compute label-distribution entropy/stats per supervision source (label column),
	and optionally merge with retrieval performance.
	Supported performance formats:
		1) DataFrame with 'source' column (already flat)
		2) Old dict: {source: {metric: value, ...}} (flat)
		3) performance.json dict: {source: {strategy: {i2t/t2i -> tier -> metric -> k -> value}}}
	When using performance.json, set perf_strategy (recommended) and perf_k (default=10).
	"""

	if verbose:
		print("\n[ENTROPY VS PERFORMANCE]")
		print(f"DF: {df.shape}")
		print(f"Entropy base: {base} ({'bits' if base == 2.0 else 'nats' if base == math.e else 'units'})")
		print(f"reference_source: {perf_reference_source}")
		print(f"strategy: {perf_strategy}")
		print(f"include_recall: {perf_include_recall}")
		print(f"k: {perf_k}")

	if label_columns is None:
		label_columns = [
			"llm_based_labels",
			"vlm_based_labels",
			"multimodal_labels",
			"llm_canonical_labels",
			"vlm_canonical_labels",
			"multimodal_canonical_labels",
		]
		if verbose:
			print(f"Default label columns: {len(label_columns)} sources: {label_columns}")
	
	rows: List[Dict[str, Any]] = []
	for idx, col in enumerate(label_columns, 1):
		if col not in df.columns:
			if verbose:
				print(f"\n[{idx}/{len(label_columns)}] '{col}' not found, skipping")
			continue
		if verbose:
			print(f"\n[{idx}/{len(label_columns)}] Processing: {col}")
			print("-" * 80)
		
		all_labels: List[str] = []
		for v in df[col].tolist():
			all_labels.extend(_parse_label_cell(v))
		
		counts = Counter(all_labels)

		total_occ = sum(counts.values())

		unique = len(counts)

		num_singletons = sum(1 for _, c in counts.items() if c == 1)

		singleton_rate = (num_singletons / unique) if unique > 0 else 0.0
		
		H = _shannon_entropy(counts, base=base)

		H_max = math.log(unique, base) if unique > 1 else 0.0

		H_norm = (H / H_max) if H_max > 0 else 0.0

		perplexity = (base ** H) if H > 0 else 1.0

		rows.append(
			{
				"source": col,
				"total_occurrences": int(total_occ),
				"unique_labels": int(unique),
				"singletons": int(num_singletons),
				"singleton_rate": float(singleton_rate),
				f"entropy_{'bits' if base == 2.0 else 'units'}": float(H),
				"entropy_max": float(H_max),
				"entropy_normalized": float(H_norm),
				"perplexity": float(perplexity),
				"effective_num_labels": float(perplexity),
			}
		)

	stats_df = pd.DataFrame(rows).sort_values("source").reset_index(drop=True)
	
	if performance is None:
		if verbose:
			print("\n✓ No performance data provided, returning entropy statistics only")
			print("\nSUMMARY STATISTICS\n")
			print(stats_df)
		
		return stats_df
	
	if verbose:
		print("\nMERGING WITH PERFORMANCE METRICS\n")
		print(df.shape, list(df.columns))
		print(df.info(verbose=True, memory_usage=True))
		print()
	
	# Build perf_df depending on input type
	if isinstance(performance, pd.DataFrame):
		perf_df = performance.copy()
		if "source" not in perf_df.columns:
			raise ValueError("If performance is a DataFrame, it must contain a 'source' column.")
	elif isinstance(performance, dict):
		schema = _infer_performance_schema(performance)
		if schema == "nested_by_source_then_strategy":
			perf_df = _flatten_performance_json(
				performance_json=performance,
				strategy=perf_strategy,
				k=perf_k,
				include_map=True,
				reference_source=perf_reference_source,
				include_recall=perf_include_recall,
				tier_alias=perf_tier_alias,
				verbose=verbose,
			)
		elif schema == "flat_metrics_by_source":
			# Old-style dict: {source: {metric: value}}
			perf_df = (
				pd.DataFrame.from_dict(performance, orient="index")
				.reset_index()
				.rename(columns={"index": "source"})
			)
		else:
			raise ValueError(
				f"Unsupported dict schema for performance. Expected either performance.json "
				f"(source->strategy->i2t/t2i...) or flat {source:{metric:...}}."
			)
	else:
		raise TypeError("performance must be a DataFrame, dict, or None.")
	
	merged = stats_df.merge(perf_df, on="source", how="left")
	if verbose:
		print(f"  perf_df shape: {perf_df.shape}")
		print(f"  merged shape: {merged.shape}")
		missing = merged[merged.filter(regex=r"^(i2t|t2i)_").isna().all(axis=1)]
		if len(missing) > 0:
			print(f"\n  ⚠️  {len(missing)} sources missing performance data:")
			for src in missing["source"].tolist():
				print(f"    - {src}")
		print("\nFINAL MERGED RESULTS\n")
		print(merged)

		x = merged["perplexity"].astype(float)
		y = merged["t2i_map10_overall"].astype(float)

		ok = x.notna() & y.notna()
		rho, pval = scipy.stats.spearmanr(x[ok], y[ok])

		if verbose:
			print(f"\nSpearman rho(perplexity, t2i_map@10 overall) = {rho:.3f}, p = {pval:.3g}")
			print("-" * 100)

	return merged

def compute_entropy_vs_performance(
	df: pd.DataFrame,
	performance: Optional[Union[pd.DataFrame, Dict[str, Dict[str, float]]]] = None,
	base: float = 2.0,
	label_columns: Optional[List[str]] = None,
	verbose: bool = False,
) -> pd.DataFrame:
	"""
	Compute label-distribution entropy (and related stats) per supervision source,
	and optionally merge with retrieval performance numbers.
	Entropy here is the Shannon entropy of the *marginal label distribution*:
			H = -sum_i p_i log p_i, where p_i = freq(label_i) / total_occurrences.
	Parameters
	----------
	df : pd.DataFrame
		Must contain the specified label columns.
	performance : DataFrame or dict (optional)
		If dict: {source_name: {"i2t_map10_overall": ..., "t2i_map10_overall": ..., ...}}
		If DataFrame: must contain a column 'source' plus any metric columns you want.
	label_columns : list[str] (optional)
		Defaults to your 6 label sources.
	base : float
		Log base for entropy. base=2 => bits. base=math.e => nats.
	
	Returns
	-------
	pd.DataFrame
		One row per label source with entropy/statistics, merged with performance if provided.
	"""
	if verbose:
		print("\n[ENTROPY VS PERFORMANCE]")
		print(f"{type(df)} {df.shape}")
		print(f"Entropy base: {base} ({'bits' if base == 2.0 else 'nats' if base == math.e else 'units'})")
	
	if label_columns is None:
		label_columns = [
			"llm_based_labels",
			"vlm_based_labels",
			"multimodal_labels",
			"llm_canonical_labels",
			"vlm_canonical_labels",
			"multimodal_canonical_labels",
		]
		if verbose:
			print(f"[DEFAULT] {len(label_columns)} sources: {label_columns}")
	else:
		if verbose:
			print(f"Custom label columns: {label_columns}")

	rows: List[Dict[str, Any]] = []
	for idx, col in enumerate(label_columns, 1):
		if col not in df.columns:
			if verbose:
				print(f"\n[{idx}/{len(label_columns)}] '{col}' not found, skipping")
			continue
		if verbose:
				print(f"\n[{idx}/{len(label_columns)}] Processing: {col}")
				print("-" * 80)

		# Flatten all labels for marginal distribution
		all_labels: List[str] = []
		for v in df[col].tolist():
			all_labels.extend(_parse_label_cell(v))

		counts = Counter(all_labels)
		total_occ = sum(counts.values())
		unique = len(counts)

		if verbose:
			print(f"  Total label occurrences: {total_occ}")
			print(f"  Unique labels: {unique}")
			print(f"  Average labels per sample: {total_occ/len(df):.2f}")

		# Singletons (unique labels appearing exactly once in the entire dataset)
		num_singletons = sum(1 for _, c in counts.items() if c == 1)
		singleton_rate = (num_singletons / unique) if unique > 0 else 0.0

		if verbose:
			print(f"  Singletons: {num_singletons}/{unique} ({singleton_rate*100:.2f}%)")
				
			# Show top-K most frequent labels
			K: int = 7
			top_labels = counts.most_common(K)
			print(f"  Top-{len(top_labels)} labels:")
			for rank, (label, count) in enumerate(top_labels, 1):
				freq_pct = (count / total_occ * 100) if total_occ > 0 else 0
				print(f"    {rank} {label:<35}{count:6d} ({freq_pct:.2f}%)")

		H = _shannon_entropy(counts, base=base)
		H_max = math.log(unique, base) if unique > 1 else 0.0  # max entropy if uniform over K
		H_norm = (H / H_max) if H_max > 0 else 0.0

		# Interpretable transforms
		perplexity = (base ** H) if H > 0 else 1.0  # "effective" support size in label space
		eff_num_labels = perplexity  # same notion under this definition

		if verbose:
			print(f"  Entropy (H): {H:.3f} {'bits' if base == 2.0 else 'units'}")
			print(f"  Max entropy (H_max): {H_max:.3f} (uniform distribution)")
			print(f"  Normalized entropy: {H_norm:.3f} (0=concentrated, 1=uniform)")
			print(f"  Perplexity: {perplexity:.1f} (effective vocabulary size)")
			print(f"  Effective # labels: {eff_num_labels:.1f}")
			
			# Interpretation
			if H_norm < 0.5:
				print(f"  📊 Distribution: HIGHLY CONCENTRATED (few dominant labels)")
			elif H_norm < 0.8:
				print(f"  📊 Distribution: MODERATELY DIVERSE")
			else:
				print(f"  📊 Distribution: HIGHLY UNIFORM (well-balanced)")

		rows.append(
			{
				"source": col,
				"total_occurrences": int(total_occ),
				"unique_labels": int(unique),
				"singletons": int(num_singletons),
				"singleton_rate": float(singleton_rate),
				f"entropy_{'bits' if base == 2.0 else 'units'}": float(H),
				"entropy_max": float(H_max),
				"entropy_normalized": float(H_norm),
				"perplexity": float(perplexity),
				"effective_num_labels": float(eff_num_labels),
			}
		)

	stats_df = pd.DataFrame(rows).sort_values("source").reset_index(drop=True)

	if verbose:
		print("\nSUMMARY STATISTICS\n")
		print(stats_df)

	# Merge performance if provided
	if performance is None:
		if verbose:
			print("\n✓ No performance data provided, returning entropy statistics only")
		return stats_df

	if verbose:
		print("\n" + "="*80)
		print("MERGING WITH PERFORMANCE METRICS")
		print("="*80)

	if isinstance(performance, dict):
		if verbose:
			print(f"  Performance data type: dict with {len(performance)} sources")
		perf_df = (
			pd.DataFrame.from_dict(performance, orient="index")
			.reset_index()
			.rename(columns={"index": "source"})
		)
	else:
		if verbose:
			print(f"  Performance data type: DataFrame with {len(performance)} rows")
		perf_df = performance.copy()
		if "source" not in perf_df.columns:
			raise ValueError("If performance is a DataFrame, it must contain a 'source' column.")

	merged = stats_df.merge(perf_df, on="source", how="left")
	if verbose:
		print(f"  Merged shape: {merged.shape}")
		print(f"  Columns: {list(merged.columns)}")
		
		# Check for missing performance data
		missing_perf = merged[merged.iloc[:, len(stats_df.columns):].isna().all(axis=1)]
		if len(missing_perf) > 0:
			print(f"\n  ⚠️  {len(missing_perf)} sources missing performance data:")
			for src in missing_perf['source'].tolist():
				print(f"    - {src}")
		else:
			print(f"\n  ✓ All sources have performance data")
		
		print("\n" + "="*80)
		print("FINAL MERGED RESULTS")
		print("="*80)
		print(merged)

	return merged

def get_singletons(df: pd.DataFrame, output_dir: str):
	print("="*100)
	print(f"[SINGETONS] {df.shape}")
	print(df.info(verbose=True, memory_usage="deep"))

	COLUMNs = [
		'llm_based_labels', 'vlm_based_labels', 'multimodal_labels',
		'llm_canonical_labels', 'vlm_canonical_labels', 'multimodal_canonical_labels',
	]

	for i, col in enumerate(COLUMNs):
		print(f"\n{i} {col}")
		if col not in df.columns:
			print(f"<!> {col} not found in dataframe columns!")
			continue

		label_list: List[List[str]] = df[col].tolist()

		print(f"  ├─ {len(label_list)} {type(label_list)} {label_list[:5]}")
		labels = list()

		for i, lbl in enumerate(label_list):
			if isinstance(lbl, list):
				# It is a valid list of labels, proceed
				pass
			elif pd.isna(lbl):
				# print(f"<!> {col} containing {labels} => skipping!")
				continue
			elif isinstance(lbl, str):
				try:
					lbl = ast.literal_eval(lbl)  # Parse string representation of list
				except Exception as e:
					print(f"<!> {col} containing {lbl} => skipping! {e}")
					continue
			else:
				print(f"<!> containing {type(lbl)} {lbl} => skipping!")
				continue

			labels.extend(lbl)

		print(f"  ├─ Total parsed labels: {len(labels)} {type(labels)}")

		unique_labels = sorted(list(set(labels)))
		print(f"  ├─ unique labels: {len(unique_labels)} {type(unique_labels)}")

		# Count frequencies
		label_counts = Counter(labels)
		label_counts_df = pd.DataFrame(
			label_counts.items(), 
			columns=['Label', 'Count']
		).sort_values(by='Count', ascending=False)
		
		label_singletons = label_counts_df[label_counts_df['Count'] == 1]['Label'].tolist()

		print(f"  └─ Singleton {len(label_singletons)}/{len(unique_labels)} ({len(label_singletons) / len(unique_labels) * 100:.2f}%):")

		print(f"{type(label_singletons)} {label_singletons[:25]}")

		label_counts_df.to_csv(os.path.join(output_dir, f"{col}_x_{len(unique_labels)}_unique_labels.csv"), index=False)

	print("="*100)
