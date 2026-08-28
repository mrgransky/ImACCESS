from utils import *

def get_single_label_stratified_split(
	csv_file: str,
	val_split_pct: float, 
	seed: int=42,
	label_col: str='label',
	verbose: bool=True,
):
	df = pd.read_csv(csv_file)
	if verbose:
		print(f"\n[SINGLE-LABBEL] Stratified Splitting")
		print(df.info(verbose=True, memory_usage="deep"))

	# Count the occurrences of each label
	label_counts = df[label_col].value_counts()

	labels_to_drop = label_counts[label_counts == 1].index
	if verbose:
		print(f"\n>> Dropping {len(labels_to_drop)} label(s) that occur(s) only once:\n{labels_to_drop.tolist()}\n")

	# Filter out rows with labels that appear only once
	df_filtered = df[~df[label_col].isin(labels_to_drop)]

	if verbose:
		print(f"df_filtered: {df_filtered.shape} {df_filtered[label_col].nunique()} unique labels")

	# Check if df_filtered is not empty
	if df_filtered.empty or df_filtered[label_col].nunique() == 0:
		raise ValueError("No labels with more than one occurrence. Stratified sampling cannot be performed.")

	if verbose:
		print(f">> Splitting dataset: train/val: {val_split_pct}...")
	
	train_df, val_df = train_test_split(
		df_filtered,
		test_size=val_split_pct,
		shuffle=True, 
		stratify=df_filtered[label_col],
		random_state=seed,
	)

	if verbose:
		print("\nLabels per dataset in train split:")
		print(train_df[label_col].value_counts())
		print('-'*120)
		print("\nLabels per dataset in val split:")
		print(val_df[label_col].value_counts())
		print('-'*120)

	train_fpath = csv_file.replace('.csv', '_train.csv')
	print(f"[SAVING] {train_fpath}")
	train_df.to_csv(train_fpath, index=False)

	val_fpath = csv_file.replace('.csv', '_val.csv')
	print(f"[SAVING] {val_fpath}")
	val_df.to_csv(val_fpath, index=False)
	print('-'*100)

	return train_df, val_df

def get_multi_label_stratified_split(
	df: pd.DataFrame,
	csv_file: str,
	val_split_pct: float,
	label_col: str,
	min_label_frequency: int = None,        # None triggers principled auto-threshold
	min_val_label_count: int = 2,           # min expected occurrences in val set
	stratification_order: int = 2,          # 1=fast/independent, 2=co-occurrence aware
	verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Split a multi-label dataset into stratified train/val sets using IterativeStratification.
	Args:
			df:                    Input DataFrame.
			csv_file:              Output CSV path stem (produces _train.csv / _val.csv).
			val_split_pct:         Fraction of data to place in validation set (e.g. 0.35).
			label_col:             Column containing lists of string labels.
			min_label_frequency:   Minimum global label count to be retained. If None (default),
														 derived from min_val_label_count and val_split_pct — this is
														 the recommended, reproducible setting. Pass an int to override
														 manually (e.g. for ablations comparing thresholds).
			min_val_label_count:   Minimum expected occurrences of a label in the val split.
														 Only used when min_label_frequency is None.
			stratification_order:  IterativeStratification order.
														 1 = independent per-label (faster, weaker co-occurrence fidelity).
														 2 = pairwise co-occurrence (recommended up to ~a few hundred k rows).
	Returns:
			(train_df, val_df)
	"""

	# ── Resolve frequency threshold ──────────────────────────────────────────
	if min_label_frequency is None:
			min_label_frequency = math.ceil(min_val_label_count / val_split_pct)
			_freq_source = (
					f"auto: ceil(min_val_label_count={min_val_label_count} / "
					f"val_split_pct={val_split_pct}) = {min_label_frequency}"
			)
	else:
			_freq_source = f"manual override = {min_label_frequency}"

	if verbose:
		print(f"\n[MULTI-LABEL STRATIFIED SPLIT]")
		print(f"  ├─ val_split_pct        : {val_split_pct}")
		print(f"  ├─ stratification_order : {stratification_order}")
		print(f"  ├─ min_label_frequency  : {min_label_frequency}  ({_freq_source})")
		print(f"  ├─ label_col: {label_col}")
		print(f"  └─ df: {df.shape} {df.columns.tolist()}")

	# ── STEP 1: Robust label parsing ─────────────────────────────────────────
	print(f"\n[1/6] Parsing '{label_col}' column...")
	if label_col not in df.columns:
		raise ValueError(f"Label column '{label_col}' not found in the DataFrame.")

	def parse_label(x):
		if isinstance(x, str):
			try:
				return ast.literal_eval(x)
			except (ValueError, SyntaxError) as e:
				raise ValueError(f"Malformed string in '{label_col}': '{x}'. Error: {e}")
		elif isinstance(x, list):
			return x
		else:
			print(f"   Warning: Unexpected type '{type(x)}' for value '{x}'. Treating as empty list.")
			return []

	df_copy = df.copy()
	n_input_rows = len(df_copy)

	try:
		df_copy[label_col] = df_copy[label_col].apply(parse_label)
		print(f"   ✓ Parsed '{label_col}' column ({n_input_rows} input rows).")
	except ValueError as e:
		raise ValueError(f"Error parsing multi-label column '{label_col}': {e}")

	# ── STEP 2: Drop rows with empty label lists ──────────────────────────────
	print(f"\n[2/6] Removing samples with empty labels...")
	df_filtered = df_copy[df_copy[label_col].apply(len) > 0].copy()  # .copy(): avoid SettingWithCopyWarning
	n_removed_empty = n_input_rows - len(df_filtered)
	print(f"   Removed {n_removed_empty} rows with empty label lists ({n_removed_empty / n_input_rows * 100:.2f}% of input).")
	if len(df_filtered) == 0:
			raise ValueError("No samples with non-empty label lists remain after parsing.")
	print(f"   DataFrame shape: {df_filtered.shape}")

	# ── STEP 3: Filter rare labels ────────────────────────────────────────────
	print(f"\n[3/6] Filtering rare labels (min_label_frequency={min_label_frequency})...")
	all_labels_flat = [l for labels in df_filtered[label_col] for l in labels]
	label_counts = Counter(all_labels_flat)
	initial_unique = len(label_counts)
	print(f"   Total label occurrences (non-unique): {len(all_labels_flat)}")
	print(f"   Total unique labels before filtering: {initial_unique}")
	rare_labels = {l for l, c in label_counts.items() if c < min_label_frequency}
	kept_labels = set(label_counts.keys()) - rare_labels
	print(f"   Rare labels (< {min_label_frequency}): {len(rare_labels)} ({len(rare_labels) / initial_unique * 100:.1f}%)")
	print(f"   Labels to keep: {len(kept_labels)} ({len(kept_labels) / initial_unique * 100:.1f}%)")

	if rare_labels:
		rare_freq_dist = Counter(label_counts[l] for l in rare_labels)
		print(f"   Rare label frequency distribution:")
		for freq in sorted(rare_freq_dist.keys()):
			labels_at_freq = [l for l, c in label_counts.items() if c == freq]
			print(
				f"      freq={freq}: {rare_freq_dist[freq]} labels: "
				f"{labels_at_freq[:20]}{'  ...' if len(labels_at_freq) > 20 else ''}"
			)
		# x2-style flat example listing, useful for quick eyeballing in logs
		rare_examples = sorted(rare_labels)[:20]
		print(
			f"   Rare labels being removed (examples): {rare_examples}"
			f"{'  ...' if len(rare_labels) > 20 else ''}"
		)

	df_filtered[label_col] = df_filtered[label_col].apply(
		lambda llist: [l for l in llist if l not in rare_labels]
	)

	n_before = len(df_filtered)
	df_filtered = df_filtered[df_filtered[label_col].apply(len) > 0].copy()  # .copy(): re-slice, avoid warning
	n_after = len(df_filtered)
	print(f"   Samples after rare-label filtering: {n_after} "
				f"(removed {n_before - n_after} that became label-empty, "
				f"{(n_before - n_after) / n_before * 100:.2f}%)")
	if n_after == 0:
			raise ValueError(
					"No samples remain after filtering rare labels. "
					"Try lowering min_label_frequency or min_val_label_count."
			)
	final_unique = len({l for labels in df_filtered[label_col] for l in labels})
	print(f"   Final unique labels: {final_unique}")
	print(f"   Net sample retention: {n_after}/{n_input_rows} ({n_after / n_input_rows * 100:.1f}% of original input)")

	# ── STEP 4: Binarise label matrix ─────────────────────────────────────────
	print(f"\n[4/6] Binarizing label matrix ({n_after} samples x {final_unique} labels)...")
	mlb = MultiLabelBinarizer(sparse_output=True)
	label_matrix = mlb.fit_transform(df_filtered[label_col])
	unique_labels = mlb.classes_
	density = label_matrix.count_nonzero() / (label_matrix.shape[0] * label_matrix.shape[1])
	print(f"   Shape       : {label_matrix.shape}")
	print(f"   dtype       : {label_matrix.dtype}")
	print(f"   Density     : {density * 100:.3f}%")
	print(f"   Non-zeros   : {label_matrix.count_nonzero()}")
	print(f"   Data size   : {label_matrix.data.nbytes / 1e6:.3f} MB")
	if len(unique_labels) == 0:
			raise ValueError("No unique labels after processing. Cannot stratify.")
	print(f"   Sample labels: {unique_labels.tolist()[:20]}"
				f"{'  ...' if len(unique_labels) > 20 else ''}")

	# ── STEP 5: Iterative stratification ─────────────────────────────────────
	print(f"\n[5/6] Iterative stratification (order={stratification_order}, "
				f"n={n_after}, val_split_pct={val_split_pct})...")
	X_indices = np.arange(n_after).reshape(-1, 1)
	try:
			stratifier = IterativeStratification(
					n_splits=2,
					order=stratification_order,
					sample_distribution_per_fold=[val_split_pct, 1.0 - val_split_pct],
			)
			t_strat = time.time()
			train_indices, val_indices = next(stratifier.split(X_indices, label_matrix))
			print(f"   ✓ Stratification completed in {time.time() - t_strat:.1f}s")
	except Exception as e:
			print(f"   ❌ Stratification failed: {e}")
			print(f"   Hint: some labels may still have too few samples.")
			print(f"   Try raising min_label_frequency (current: {min_label_frequency}) "
						f"or switching stratification_order (current: {stratification_order}).")
			raise
	train_df = df_filtered.iloc[train_indices].reset_index(drop=True)
	val_df = df_filtered.iloc[val_indices].reset_index(drop=True)
	if train_df.empty or val_df.empty:
			raise ValueError("Train or validation set is empty after splitting.")
	print(f"   Train indices: {len(train_indices)} | Val indices: {len(val_indices)}")

	# ── STEP 6: Post-split label coverage audit ───────────────────────────────
	print(f"\n[6/6] Post-split label coverage audit...")
	train_label_set = {l for labels in train_df[label_col] for l in labels}
	val_label_set = {l for labels in val_df[label_col] for l in labels}
	train_only = train_label_set - val_label_set
	val_only = val_label_set - train_label_set
	both = train_label_set & val_label_set
	print(f"   Labels in both splits : {len(both)} ({len(both) / final_unique * 100:.1f}% of kept labels)")
	print(f"   Labels only in train  : {len(train_only)} ({len(train_only) / final_unique * 100:.1f}%)")
	print(f"   Labels only in val    : {len(val_only)} ({len(val_only) / final_unique * 100:.1f}%)")

	if train_only:
		print(f"Train-only samples: {sorted(train_only)[:20]}")

	if val_only:
		print(f"Val-only samples: {sorted(val_only)[:20]}")

	if not train_only and not val_only:
		print("[OK] All labels present in both splits.")

	print(f"\n[Multi-label stratified split] SUMMARY")
	print(f"   Original : {df_filtered.shape}")
	print(f"   Train    : {train_df.shape}  ({len(train_df) / n_after * 100:.1f}%)")
	print(f"   Val      : {val_df.shape}  ({len(val_df) / n_after * 100:.1f}%)")
	train_path = csv_file.replace('.csv', '_train.csv')
	val_path = csv_file.replace('.csv', '_val.csv')
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)

	cols = ['llm_canonical_labels', 'vlm_canonical_labels', 'multimodal_canonical_labels']
	vocabs = {}
	for split_path in [train_path, val_path]:
		print(f"{split_path}")
		df = pd.read_csv(split_path)
		for col in cols:
			labels = set()
			for v in df[col].dropna():
				labels.update(ast.literal_eval(v))
			vocabs[(split_path, col)] = labels
			print(f"{col:30}{len(labels):5d} unique labels")
		print()

	intersection_train = vocabs[(train_path, cols[0])] & vocabs[(train_path, cols[1])] & vocabs[(train_path, cols[2])]
	intersection_val = vocabs[(val_path, cols[0])] & vocabs[(val_path, cols[1])] & vocabs[(val_path, cols[2])]

	print(f"Train-split intersection across all three: {len(intersection_train)} labels")
	print(f"Val-split intersection across all three: {len(intersection_val)} labels")
	print(f"Overlap (labels in both): {len(intersection_train & intersection_val)}")

	return train_df, val_df

def diagnose_tier_masks(
	class_names: List[str],
	freq,
	head_mask,
	rare_mask,
	active_mask=None,
	tag: str = "TIER DIAGNOSTIC",
	max_rows: int = 60,
) -> Dict[str, Any]:
	"""
	Print and validate per-label tier assignments.
	"""
	f = torch.as_tensor(freq, dtype=torch.float32).flatten().cpu()
	h = torch.as_tensor(head_mask, dtype=torch.bool).flatten().cpu()
	r = torch.as_tensor(rare_mask, dtype=torch.bool).flatten().cpu()

	if active_mask is None:
		a = torch.ones(len(class_names), dtype=torch.bool)
	else:
		a = torch.as_tensor(active_mask, dtype=torch.bool).flatten().cpu()

	lengths = {
		"class_names": len(class_names),
		"freq": len(f),
		"head_mask": len(h),
		"rare_mask": len(r),
		"active_mask": len(a),
	}

	if len(set(lengths.values())) != 1:
		raise ValueError(
			f"[{tag}] Length mismatch among class names, frequencies, and masks: "
			f"{lengths}"
		)

	overlap = h & r
	tiered = h | r
	untiered_active = a & ~tiered
	tiered_inactive = tiered & ~a

	print(f"\n{'=' * 78}")
	print(f"[{tag}]")
	print(f"{'=' * 78}")
	print(f"  ├─ Classes             : {len(class_names):,}")
	print(f"  ├─ Active              : {int(a.sum().item()):,}")
	print(f"  ├─ Head                : {int(h.sum().item()):,}")
	print(f"  ├─ Rare                : {int(r.sum().item()):,}")
	print(f"  ├─ Head ∩ Rare         : {int(overlap.sum().item()):,}")
	print(f"  ├─ Active body         : {int(untiered_active.sum().item()):,}")
	print(f"  └─ Tiered but inactive : {int(tiered_inactive.sum().item()):,}")

	print(f"\n  [Per-label assignments]")
	print(f"  {'#':>4s}  {'label':<32s} {'freq':>10s}  {'active':>7s}  {'head':>6s}  {'rare':>6s}")

	order = torch.argsort(f, descending=True).tolist()

	for rank, index in enumerate(order[:max_rows]):
		print(
			f"  {rank:>4d}  "
			f"{str(class_names[index]):<32s} "
			f"{f[index].item():>10.1f}  "
			f"{str(bool(a[index])):>7s}  "
			f"{str(bool(h[index])):>6s}  "
			f"{str(bool(r[index])):>6s}"
		)

	if len(class_names) > max_rows:
		print(
			f"  ... {len(class_names) - max_rows:,} additional labels omitted "
			f"(max_rows={max_rows})"
		)

	if bool(overlap.any()):
		overlap_labels = [
			class_names[index]
			for index in torch.nonzero(overlap).flatten().tolist()
		]

		print("\n  [ERROR] Head/rare overlap detected:")
		for label in overlap_labels:
			index = class_names.index(label)
			print(f"    ├─ {label!r}: frequency={f[index].item():.1f}")

	if bool(tiered_inactive.any()):
		inactive_labels = [
			class_names[index]
			for index in torch.nonzero(tiered_inactive).flatten().tolist()
		]

		print("\n  [WARNING] Tiered labels marked inactive:")
		print(f"    {inactive_labels[:20]}")

	return {
		"n_classes": len(class_names),
		"n_active": int(a.sum().item()),
		"n_head": int(h.sum().item()),
		"n_rare": int(r.sum().item()),
		"n_overlap": int(overlap.sum().item()),
		"overlap_labels": [
			class_names[index]
			for index in torch.nonzero(overlap).flatten().tolist()
		],
		"n_active_body": int(untiered_active.sum().item()),
		"n_tiered_inactive": int(tiered_inactive.sum().item()),
	}

def build_shared_eval_protocol(
	train_df: pd.DataFrame,
	output_dir: str,
	llm_col: str = "llm_canonical_labels",
	vlm_col: str = "vlm_canonical_labels",
	multimodal_col: str = "multimodal_canonical_labels",
	pareto_threshold: float = 0.8,
	rare_percentile: float = 0.2,
	verbose: bool = False,
) -> dict:
	"""
	Build a fixed shared-vocabulary tier specification for R1-C evaluation.

	The label vocabulary is the INTERSECTION of labels observed in all three
	training regimes (LLM, VLM, multimodal). Tier definitions are INSPIRED BY
	compute_loss_masks() (Pareto-mass head / bottom-quantile rare) but, unlike
	compute_loss_masks(), head and rare are derived from a SINGLE frequency
	ordering so they are DISJOINT BY CONSTRUCTION:
		- head: smallest rank-prefix (by frequency, descending) covering
		        `pareto_threshold` of total shared-vocab occurrence mass;
		- rare: bottom `rare_percentile` of active classes, drawn only from
		        classes the head tier did NOT already claim.

	On small/flat shared vocabularies, an independent Pareto rule and an
	independent quantile rule (the original two-predicate design) can claim
	the same classes — e.g. 5/7 head and 4/7 rare on a 7-class intersection,
	which is incoherent (head ∩ rare should be empty for a tiered protocol).
	This function forces disjointness explicitly and asserts it before
	writing anything to disk, rather than letting the consumer discover the
	collision at evaluation time.

	Reference frequencies come from MULTIMODAL training annotations only.
	Rarity is defined on TRAINING frequency by construction — using val
	frequency would measure a sampling artifact of the val split rather
	than the model's training exposure, which is precisely the long-tail
	property the paper claims to measure. val_df therefore has no role in
	*defining* tiers; it only enters downstream when samples are scored
	against these fixed tiers (see evaluate_shared_protocol()).
	"""
	protocol_path = os.path.join(output_dir, "shared_eval_protocol.json")

	if verbose:
		print(f"\n{'='*70}")
		print(f"[SHARED EVAL PROTOCOL] Building fixed cross-run tier specification")
		print(f"{'='*70}")
		print(f"  ├─ train_df shape          : {train_df.shape}")
		print(f"  ├─ llm_col                 : {llm_col!r}")
		print(f"  ├─ vlm_col                 : {vlm_col!r}")
		print(f"  ├─ multimodal_col (ref)    : {multimodal_col!r}")
		print(f"  ├─ pareto_threshold        : {pareto_threshold}")
		print(f"  ├─ rare_percentile         : {rare_percentile}")
		print(f"  └─ output_dir              : {output_dir}")

	def parse_labels(value):
		if isinstance(value, str):
			try:
				parsed = ast.literal_eval(value)
			except (ValueError, SyntaxError):
				raise ValueError(f"Malformed label list encountered: {value!r}")
			if not isinstance(parsed, list):
				raise ValueError(f"Expected a list of labels, received: {type(parsed)}")
			return parsed
		if isinstance(value, list):
			return value
		if pd.isna(value):
			return []
		raise ValueError(f"Unsupported label value type: {type(value)}")

	required_columns = [llm_col, vlm_col, multimodal_col]
	missing_columns = [
		column
		for column in required_columns
		if column not in train_df.columns
	]
	if missing_columns:
		raise ValueError(f"Missing required training-label columns: {missing_columns}")

	parsed_train_df = train_df.copy()
	for column in required_columns:
		parsed_train_df[column] = parsed_train_df[column].apply(parse_labels)

	llm_counts = Counter(
		label
		for labels in parsed_train_df[llm_col]
		for label in labels
	)
	vlm_counts = Counter(
		label
		for labels in parsed_train_df[vlm_col]
		for label in labels
	)
	multimodal_counts = Counter(
		label
		for labels in parsed_train_df[multimodal_col]
		for label in labels
	)

	shared_class_names = sorted(
		set(llm_counts)
		& set(vlm_counts)
		& set(multimodal_counts)
	)

	if verbose:
		llm_vocab   = set(llm_counts)
		vlm_vocab   = set(vlm_counts)
		mm_vocab    = set(multimodal_counts)
		union_vocab = llm_vocab | vlm_vocab | mm_vocab
		print(f"\n  [Vocabulary intersection]")
		print(f"  ├─ LLM vocabulary          : {len(llm_vocab):,}")
		print(f"  ├─ VLM vocabulary          : {len(vlm_vocab):,}")
		print(f"  ├─ Multimodal vocabulary   : {len(mm_vocab):,}")
		print(f"  ├─ Union (any regime)      : {len(union_vocab):,}")
		print(f"  ├─ Shared (all 3 regimes)  : {len(shared_class_names):,}")
		if union_vocab:
			print(f"  ├─ Intersection / union    : {len(shared_class_names)/len(union_vocab):.1%}")
		if llm_vocab:
			print(f"  ├─ LLM-only (dropped)      : {len(llm_vocab - set(shared_class_names)):,}")
		if vlm_vocab:
			print(f"  ├─ VLM-only (dropped)      : {len(vlm_vocab - set(shared_class_names)):,}")
		if mm_vocab:
			print(f"  └─ MM-only  (dropped)      : {len(mm_vocab - set(shared_class_names)):,}")

	if not shared_class_names:
		raise ValueError("training-label intersection across LLM, VLM, & multimodal supervision is empty.")

	# Reference freq: multimodal training counts, restricted to the shared intersection.
	shared_train_freq = torch.tensor(
		[multimodal_counts[label] for label in shared_class_names],
		dtype=torch.float32,
	)

	# NOTE: active_mask is a NO-OP BY CONSTRUCTION. shared_class_names is the
	# intersection of the three Counters' keys, and a Counter only holds keys
	# seen >= 1 time, so multimodal_counts[label] >= 1 for every shared label.
	# We keep it purely to stay structurally parallel to compute_loss_masks()
	# (which does need it, because it operates over the full num_classes space
	# including inactive classes). Here it is always all-True.
	active_mask = (shared_train_freq > 0)
	if verbose and not bool(active_mask.all()):
		# Should be unreachable; if it ever fires, an upstream invariant broke.
		print(f"  [WARNING] active_mask unexpectedly has "
			  f"{int((~active_mask).sum().item())} inactive shared class(es) — "
			  f"this violates the intersection invariant.")

	n_active = int(active_mask.sum().item())

	# ── Head tier — Pareto cumulative-frequency prefix ──────────────────────
	sorted_freq, sorted_idx = torch.sort(shared_train_freq, descending=True)
	cumulative_freq = sorted_freq.cumsum(0)
	total_mass = cumulative_freq[-1]

	# `>=` (not the old `<=` then `+1`): the previous formulation could take
	# one class more than needed once the running sum already crosses the
	# budget, inflating the head tier beyond the stated pareto_threshold.
	reached = torch.nonzero(cumulative_freq >= total_mass * pareto_threshold).flatten()
	pareto_cutoff_raw = int(reached[0].item()) + 1 if len(reached) else n_active

	if verbose:
		print(f"\n  [Reference frequency distribution (multimodal, shared vocab)]")
		print(f"  ├─ freq [min, max]         : [{shared_train_freq.min():.0f}, {shared_train_freq.max():.0f}]")
		print(f"  ├─ freq mean / std         : {shared_train_freq.mean():.1f} / {shared_train_freq.std():.1f}")
		print(f"  ├─ freq median             : {shared_train_freq.median():.1f}")
		print(f"  ├─ classes with freq == 1  : {(shared_train_freq == 1).sum().item():,}")
		print(f"  ├─ classes with freq <= 5  : {(shared_train_freq <= 5).sum().item():,}")
		print(f"  └─ classes with freq > 10  : {(shared_train_freq > 10).sum().item():,}")
		print(f"\n  [Head-tier Pareto computation]")
		print(f"  ├─ total occurrence mass   : {total_mass.item():.0f}")
		print(f"  ├─ mass budget ({pareto_threshold:.0%})       : {(total_mass * pareto_threshold).item():.1f}")
		print(f"  └─ raw Pareto cutoff       : {pareto_cutoff_raw} class(es) (before disjointness clamp)")

	# ── Rare tier — bottom rare_percentile of ACTIVE classes, taken ONLY from
	#    classes the head tier did NOT already claim (identical guard to
	#    compute_loss_masks() for the degenerate n<=1 case). This is the fix:
	#    deriving rare from "what's left after head" makes the two tiers
	#    disjoint BY CONSTRUCTION instead of by coincidence of vocab scale.
	if n_active > 1:
		# n_rare_target = max(1, int(round(rare_percentile * n_active)))
		# more defensible for a “bottom rare_percentile*100%” specification 
		# because it ensures the tier contains at least the requested proportion:
		n_rare_target = max(1, math.ceil(rare_percentile * n_active)) 

		# Never let the head tier swallow so much of the vocabulary that no
		# room is left for a rare tier — this is exactly the 5-of-7 collision
		# case observed on the small dataset.
		max_head = max(1, n_active - n_rare_target)
		pareto_cutoff = pareto_cutoff_raw
		clamped = pareto_cutoff_raw > max_head
		if clamped:
			pareto_cutoff = max_head
			if verbose:
				print(
					f"\n  [TIER ADJUST] Pareto head would cover {pareto_cutoff_raw}/{n_active} "
					f"active classes, leaving fewer than {n_rare_target} for the rare tier. "
					f"Clamping head to {max_head}. This shared vocabulary ({n_active} classes) "
					f"is too small/flat for a {pareto_threshold:.0%} mass budget and a bottom-"
					f"{rare_percentile:.0%} quantile to describe disjoint classes."
				)

		head_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
		head_mask[sorted_idx[:pareto_cutoff]] = True

		# Rare candidates = active, non-head classes, ordered ascending by
		# frequency (sorted_idx is descending, so the remainder just needs
		# flipping) — guarantees rare picks the LOWEST-frequency leftovers.
		tail_idx = sorted_idx[pareto_cutoff:].flip(0)
		n_rare = min(n_rare_target, len(tail_idx))

		rare_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
		rare_mask[tail_idx[:n_rare]] = True

		rare_frequency_threshold = (
			shared_train_freq[rare_mask].max()
			if bool(rare_mask.any())
			else torch.tensor(float("nan"))
		)
	else:
		# Degenerate intersection: a single (or zero) active class cannot
		# define a meaningful quantile boundary. Mirror compute_loss_masks()
		# and emit an empty rare tier rather than an ill-defined one.
		pareto_cutoff = pareto_cutoff_raw
		clamped = False
		head_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
		head_mask[sorted_idx[:pareto_cutoff]] = True
		rare_frequency_threshold = torch.tensor(float("nan"))
		rare_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
		if verbose:
			print(f"\n  [WARNING] n_active={n_active} — rare-tier quantile is "
				  f"degenerate/undefined; rare_mask forced to EMPTY.")

	# ── Hard invariant, enforced HERE at construction time, not discovered
	#    later inside evaluate_shared_protocol() ─────────────────────────────
	overlap = head_mask & rare_mask

	if bool(overlap.any()):
		overlap_labels = [shared_class_names[i] for i in torch.nonzero(overlap).flatten().tolist()]
		raise AssertionError(
			f"head/rare overlap survived disjoint construction — this should be "
			f"unreachable: {overlap_labels}"
		)

	if not bool(head_mask.any()):
		raise AssertionError("Disjoint construction produced an empty head tier — check pareto_threshold.")

	n_head = int(head_mask.sum().item())
	n_rare = int(rare_mask.sum().item())
	body_n = n_active - n_head - n_rare

	if verbose:
		print(f"\n  [Tier assignment — disjoint by construction]")
		print(f"  ├─ Shared classes          : {len(shared_class_names):,}")
		print(f"  ├─ Active classes          : {n_active:,}")
		print(f"  ├─ Head (Pareto {pareto_threshold:.0%})       : {n_head:,}"
			  f"  (cutoff idx = {pareto_cutoff}{', clamped' if clamped else ''})")
		print(f"  ├─ Rare (bottom {rare_percentile:.0%})       : {n_rare:,}")
		print(f"  ├─ Head ∩ Rare             : {int(overlap.sum().item())}  (must be 0)")
		if not torch.isnan(rare_frequency_threshold):
			print(f"  ├─ Rare freq threshold     : {rare_frequency_threshold.item():.1f}")
		else:
			print(f"  ├─ Rare freq threshold     : n/a (degenerate)")
		print(f"  ├─ Body (neither)          : {body_n:,}")

		# Per-label tier table — makes the exact membership auditable at a glance
		print(f"\n  [Per-label tier table] (sorted by frequency, descending)")
		print(f"  {'#':>3s}  {'label':<28s} {'freq':>8s}  {'head':>5s} {'rare':>5s}")
		order = torch.argsort(shared_train_freq, descending=True).tolist()
		max_rows = 60
		for rank, i in enumerate(order[:max_rows]):
			print(
				f"  {rank:>3d}  {shared_class_names[i]:<28s} {shared_train_freq[i].item():>8.0f}  "
				f"{str(bool(head_mask[i])):>5s} {str(bool(rare_mask[i])):>5s}"
			)
		if len(shared_class_names) > max_rows:
			print(f"  ... {len(shared_class_names) - max_rows} more labels suppressed (max_rows={max_rows})")

		# Explicit small-N caveats — the exact things R1-F would flag if unremarked
		if 0 < n_rare < 10:
			print(f"\n  ⚠  Rare tier has only {n_rare} class(es). Metrics over this tier "
				  f"are HIGH-VARIANCE / small-sample; report n_rare prominently and "
				  f"consider variance estimates in the writeup (R1-F).")
		if n_rare == 0:
			print(f"\n  ⚠  Rare tier is EMPTY. Any downstream shared-protocol rare-tier "
				  f"metric will be undefined for this run.")
		if n_head < 10:
			print(f"  ⚠  Head tier has only {n_head} class(es) — also small-N.")
		if len(shared_class_names) < 20:
			print(f"  ⚠  Shared vocabulary itself has only {len(shared_class_names)} class(es) — "
				  f"tiered metrics on this protocol will be noisy regardless of tier split; "
				  f"consider reporting overall shared-vocab retrieval alongside tiers.")

		diagnose_tier_masks(
			class_names=shared_class_names,
			freq=shared_train_freq,
			head_mask=head_mask,
			rare_mask=rare_mask,
			active_mask=active_mask,
			tag="SHARED EVAL PROTOCOL — BUILDER",
			max_rows=60,
		)

	shared_protocol = {
		"protocol_name": "shared_intersection_mm_reference_v2_disjoint",
		"reference_label_column": multimodal_col,
		"pareto_threshold": pareto_threshold,
		"rare_percentile": rare_percentile,
		"shared_class_names": shared_class_names,
		"shared_train_freq": shared_train_freq.tolist(),
		"active_mask": active_mask.tolist(),
		"head_mask": head_mask.tolist(),
		"rare_mask": rare_mask.tolist(),
		# explicit name lists — removes all index-alignment ambiguity downstream
		"head_labels": [c for c, m in zip(shared_class_names, head_mask.tolist()) if m],
		"rare_labels": [c for c, m in zip(shared_class_names, rare_mask.tolist()) if m],
		"n_classes": len(shared_class_names),
		"n_head_classes": n_head,
		"n_rare_classes": n_rare,
		"rare_frequency_threshold": (
			rare_frequency_threshold.item()
			if not torch.isnan(rare_frequency_threshold)
			else None
		),
		"tiers_disjoint": True,
		"head_pareto_clamped": clamped,
	}

	with open(protocol_path, "w", encoding="utf-8") as file:
		json.dump(shared_protocol, file, indent=2)

	if verbose:
		print(f"\n  └─ Saved protocol          : {protocol_path}")
		print(f"{'='*70}\n")

	return shared_protocol