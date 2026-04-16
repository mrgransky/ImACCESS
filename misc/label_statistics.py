from utils import *


def _mean_jaccard(sets_a: List[set], sets_b: List[set]) -> float:
    """Mean Jaccard over rows where union is non-empty."""
    if len(sets_a) != len(sets_b):
        raise ValueError("Jaccard inputs must have the same number of rows.")
    vals = []
    for a, b in zip(sets_a, sets_b):
        if not a and not b:
            continue
        u = a | b
        if len(u) == 0:
            continue
        vals.append(len(a & b) / len(u))
    return float(np.mean(vals)) if vals else 0.0

def get_taxonomy_supervison(
    df: pd.DataFrame,
    sources: Optional[List[str]] = None,
    anchor_vlm_col: str = "vlm_canonical_labels",
    base: float = 2.0,
    title: str = "Supervision Taxonomy Radar (normalized)",
    normalize: str = "minmax",
    fill_alpha: float = 0.12,
    line_width: float = 2.0,
    figsize: Tuple[int, int] = (7, 7),
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Compute and plot a 3-axis radar chart for:
      (1) Semantic coverage
      (2) Visual grounding
      (3) Statistical density

    Definitions (per supervision source):
      - Semantic coverage: perplexity = base ** H, where H is Shannon entropy of the marginal label distribution.
        (Interpretation: 'effective vocabulary size'.)
      - Visual grounding: mean Jaccard(source_labels, vlm_canonical_labels) across samples.
        (Interpretation: fraction of a source that is consistent with image-grounded concepts.)
      - Statistical density: (avg occurrences per label) * (1 - singleton_rate)
        where avg occurrences per label = total_occurrences / unique_labels.

    Normalization:
      - minmax: scales each axis across the provided sources to [0, 1].
      - none: returns raw values but plots them after minmax anyway (radar needs comparable scale).

    Returns
    -------
    scores_df : pd.DataFrame
        Raw and normalized axis scores per source.
    fig, ax : matplotlib figure/axes
    """
    if sources is None:
        sources = ["llm_canonical_labels", "vlm_canonical_labels", "multimodal_canonical_labels"]

    missing = [c for c in ([anchor_vlm_col] + sources) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Pre-parse per-row sets (needed for Jaccard grounding)
    parsed_sets: Dict[str, List[set]] = {}
    for col in set([anchor_vlm_col] + sources):
        parsed_sets[col] = [set(_parse_label_cell(v)) for v in df[col].tolist()]

    anchor_sets = parsed_sets[anchor_vlm_col]

    rows = []
    for col in sources:
        # Marginal distribution stats
        all_labels: List[str] = []
        for s in parsed_sets[col]:
            all_labels.extend(list(s))  # set -> unique per row; avoids duplicates inside a sample

        counts = Counter(all_labels)
        total_occ = sum(counts.values())
        unique = len(counts)

        singletons = sum(1 for _, c in counts.items() if c == 1)
        singleton_rate = (singletons / unique) if unique > 0 else 0.0

        H = _shannon_entropy(counts, base=base)
        perplexity = (base ** H) if H > 0 else 1.0  # effective vocabulary size

        # Axis 1: semantic coverage proxy
        semantic_coverage = perplexity

        # Axis 2: visual grounding proxy (agreement with VLM canonical)
        visual_grounding = _mean_jaccard(parsed_sets[col], anchor_sets) if col != anchor_vlm_col else 1.0

        # Axis 3: statistical density proxy
        avg_occ_per_label = (total_occ / unique) if unique > 0 else 0.0
        statistical_density = avg_occ_per_label * (1.0 - singleton_rate)

        rows.append(
            {
                "source": col,
                "unique_labels": int(unique),
                "total_occurrences": int(total_occ),
                "singletons": int(singletons),
                "singleton_rate": float(singleton_rate),
                "semantic_coverage_raw": float(semantic_coverage),
                "visual_grounding_raw": float(visual_grounding),
                "statistical_density_raw": float(statistical_density),
            }
        )

    scores_df = pd.DataFrame(rows)

    # Normalize to [0,1] per axis across the chosen sources
    axis_cols = ["semantic_coverage_raw", "visual_grounding_raw", "statistical_density_raw"]
    for c in axis_cols:
        v = scores_df[c].to_numpy(dtype=float)
        if normalize == "minmax":
            vmin, vmax = float(np.min(v)), float(np.max(v))
            if abs(vmax - vmin) < 1e-12:
                scores_df[c.replace("_raw", "_norm")] = 0.5  # all equal => neutral
            else:
                scores_df[c.replace("_raw", "_norm")] = (v - vmin) / (vmax - vmin)
        else:
            # still create *_norm for plotting convenience
            vmin, vmax = float(np.min(v)), float(np.max(v))
            scores_df[c.replace("_raw", "_norm")] = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    # Radar plot
    categories = ["Semantic coverage", "Visual grounding", "Statistical density"]
    norm_cols = ["semantic_coverage_norm", "visual_grounding_norm", "statistical_density_norm"]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, y=1.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"])
    ax.set_ylim(0.0, 1.0)

    for _, r in scores_df.iterrows():
        vals = [float(r[c]) for c in norm_cols]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=line_width, label=r["source"])
        ax.fill(angles, vals, alpha=fill_alpha)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()

    return scores_df, fig, ax

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

def compute_entropy_vs_performance(
		df: pd.DataFrame,
		performance: Optional[Union[pd.DataFrame, Dict[str, Dict[str, float]]]] = None,
		label_columns: Optional[List[str]] = None,
		base: float = 2.0,
		verbose: bool = False
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
				print("\n" + "="*80)
				print("COMPUTING ENTROPY VS PERFORMANCE ANALYSIS")
				print("="*80)
				print(f"Dataset size: {len(df):,} samples")
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
						print(f"Using default label columns: {len(label_columns)} sources")
		else:
				if verbose:
						print(f"Using custom label columns: {label_columns}")

		rows: List[Dict[str, Any]] = []

		for idx, col in enumerate(label_columns, 1):
				if col not in df.columns:
						if verbose:
								print(f"\n[{idx}/{len(label_columns)}] ⚠️  Column '{col}' not found, skipping")
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
						print(f"  Total label occurrences: {total_occ:,}")
						print(f"  Unique labels: {unique:,}")
						print(f"  Average labels per sample: {total_occ/len(df):.2f}")

				# Singletons (unique labels appearing exactly once in the entire dataset)
				num_singletons = sum(1 for _, c in counts.items() if c == 1)
				singleton_rate = (num_singletons / unique) if unique > 0 else 0.0

				if verbose:
						print(f"  Singletons: {num_singletons:,}/{unique:,} ({singleton_rate*100:.1f}%)")
						
						# Show top-5 most frequent labels
						top_labels = counts.most_common(5)
						print(f"  Top-5 labels:")
						for rank, (label, count) in enumerate(top_labels, 1):
								freq_pct = (count / total_occ * 100) if total_occ > 0 else 0
								print(f"    {rank}. '{label}': {count:,} ({freq_pct:.2f}%)")

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
				print("\n" + "="*80)
				print("SUMMARY STATISTICS")
				print("="*80)
				print(stats_df.to_string(index=False))

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
				print(merged.to_string(index=False))

		return merged

def compute_label_agreement_and_singletons(df: pd.DataFrame):
	print(f"Computing label agreement and singletons for {len(df)}")
	print(f"{list(df.columns)}")
	COLUMNs = [
		'llm_based_labels', 'vlm_based_labels', 'multimodal_labels',
		'llm_canonical_labels', 'vlm_canonical_labels', 'multimodal_canonical_labels',
	]
	# cols = df.columns[-6:].tolist()
	# print(cols)

	parsed_cols = {}
	for col in COLUMNs:
		if col not in df.columns:
			print(f"Column {col} not found in the dataframe")
			continue
		label_list_raw = df[col].tolist()
		all_labels = []
		parsed_rows = []
		for val in label_list_raw:
				# Handle various data formats (list, string-list, or NaN)
				if isinstance(val, list):
						row_lbls = val
				elif pd.isna(val) or val == "" or val == "[]":
						row_lbls = []
				elif isinstance(val, str):
						try:
								row_lbls = ast.literal_eval(val)
						except:
								row_lbls = []
				else:
						row_lbls = []
				
				all_labels.extend(row_lbls)
				parsed_rows.append(set(row_lbls)) # Store as sets for agreement logic
		
		parsed_cols[col] = parsed_rows
		unique_labels = sorted(list(set(all_labels)))
		label_counts = Counter(all_labels)
		label_singletons = [l for l, count in label_counts.items() if count == 1]
		
		print(f"\n[{col.upper()}]")
		print(f"Total Labels: {len(all_labels)}")
		print(f"Unique Labels: {len(unique_labels)}")
		print(f"Singletons: {len(label_singletons)}/{len(unique_labels)} "
					f"({len(label_singletons)/len(unique_labels)*100 if len(unique_labels)>0 else 0:.2f}%)")
		print("-" * 50)
	
	# --- Agreement Analysis (The "Grounding" Metric) ---
	# Check if there are LLM and VLM canonical columns
	l_can = [c for c in df.columns if 'LLM_CANONICAL' in c.upper()]
	v_can = [c for c in df.columns if 'VLM_CANONICAL' in c.upper()]

	if l_can and v_can:
		llm_sets = parsed_cols[l_can[0]]
		vlm_sets = parsed_cols[v_can[0]]
		
		jaccard_scores = []
		at_least_one_intersect = 0
		exact_matches = 0
		
		for l_set, v_set in zip(llm_sets, vlm_sets):
				intersection = l_set.intersection(v_set)
				union = l_set.union(v_set)
				
				if len(union) > 0:
						jaccard_scores.append(len(intersection) / len(union))
						if len(intersection) > 0:
								at_least_one_intersect += 1
				
				if l_set == v_set and len(l_set) > 0:
						exact_matches += 1
		print("\n" + "!" * 20 + " CROSS-MODAL AGREEMENT ANALYSIS " + "!" * 20)
		print(f"System: {l_can[0]} <-> {v_can[0]}")
		avg_j = sum(jaccard_scores)/len(jaccard_scores) if jaccard_scores else 0
		print(f"Mean Jaccard Index: {avg_j:.4f}")
		print(f"Partial Agreement (>=1 shared label): {at_least_one_intersect/len(llm_sets)*100:.2f}%")
		print(f"Exact Match Rate: {exact_matches/len(llm_sets)*100:.2f}%")
		print("!" * 72)
	else:
		print(f"l_can: {l_can}, v_can: {v_can} => no agreement could be computed!")

def get_singleton_in_uniques(df: pd.DataFrame):
	print(f">> Getting singleton statistics for {len(df)} samples")
	COLUMNs = [
		'llm_based_labels', 'vlm_based_labels', 'multimodal_labels',
		'llm_canonical_labels', 'vlm_canonical_labels', 'multimodal_canonical_labels',
	]
	# cols = df.columns[-6:].tolist()
	# print(cols)

	for i, col in enumerate(COLUMNs):
		if col not in df.columns:
			print(f"<!> {col} not found in dataframe columns!")
			continue

		label_list: List[List[str]] = df[col].tolist()

		print(f"\n[{col.upper()}] {len(label_list)} {type(label_list)} labels: {label_list[:3]}")
		labels = list()

		for i, lbl in enumerate(label_list):
			# print(i, type(lbl), lbl)

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

		print(f"{len(labels)} labels")
		unique_labels = sorted(list(set(labels)))
		print(f"unique_labels: {type(unique_labels)} {len(unique_labels)} {unique_labels[:10]}")

		# Count frequencies
		label_counts = Counter(labels)
		label_counts_df = pd.DataFrame(
			label_counts.items(), 
			columns=['Label', 'Count']
		).sort_values(by='Count', ascending=False)
		
		# Singleton analysis
		label_singletons = label_counts_df[label_counts_df['Count'] == 1]['Label'].tolist()
		print(f"[{col.upper()}] Singleton {type(label_singletons)}: {len(label_singletons)}/{len(unique_labels)} ({len(label_singletons) / len(unique_labels) * 100:.2f}%):")
		print(label_singletons[:25])
		print("="*100)

