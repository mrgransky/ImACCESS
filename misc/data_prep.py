from utils import *

def get_single_label_stratified_split(
	df: pd.DataFrame, 
	val_split_pct: float, 
	seed: int=42,
	label_col: str='label',
	verbose: bool=True,
):
	if verbose:
		print(f"\n>> Stratified Splitting [Single-label dataset]")

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

	return train_df, val_df

def get_multi_label_stratified_split(
	df: pd.DataFrame,
	csv_file: str,
	val_split_pct: float,
	label_col: str,
	min_label_frequency: int = None,        # None triggers principled auto-threshold
	min_val_label_count: int = 2,           # min expected occurrences in val set
	stratification_order: int = 2,          # 1=fast/independent, 2=co-occurrence aware
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
	training regimes (LLM, VLM, multimodal). Tier definitions reproduce
	compute_loss_masks() exactly so that the evaluation notion of head/rare
	matches the training-time notion of rarity:
		- head: smallest set of classes covering `pareto_threshold` of total
		        occurrences (Pareto cumulative-frequency cutoff);
		- rare: active classes at or below the `rare_percentile` quantile of
		        positive frequencies.

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
		# how much of each regime survives into the shared vocab
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

	# ── Head tier — Pareto cumulative-frequency cutoff (identical to compute_loss_masks) ──
	sorted_freq, sorted_idx = torch.sort(shared_train_freq, descending=True)
	cumulative_freq = sorted_freq.cumsum(0)
	pareto_cutoff = int((cumulative_freq <= cumulative_freq[-1] * pareto_threshold).sum().item()) + 1

	head_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
	head_mask[sorted_idx[:pareto_cutoff]] = True

	# ── Rare tier — bottom rare_percentile of ACTIVE classes (identical guard to compute_loss_masks) ──
	active_freq = shared_train_freq[active_mask]
	if active_freq.numel() > 1:
		rare_frequency_threshold = torch.quantile(active_freq, rare_percentile)
		rare_mask = (shared_train_freq <= rare_frequency_threshold) & active_mask
	else:
		# Degenerate intersection: a single (or zero) active class cannot define
		# a meaningful quantile boundary. Mirror compute_loss_masks() and emit
		# an empty rare tier rather than an ill-defined one.
		rare_frequency_threshold = torch.tensor(float("nan"))
		rare_mask = torch.zeros(len(shared_class_names), dtype=torch.bool)
		if verbose:
			print(f"\n  [WARNING] active_freq has only {active_freq.numel()} class(es) — "
				  f"rare-tier quantile is degenerate/undefined; rare_mask forced to EMPTY.")

	# ── Small-N surfacing — these numbers should appear in the response-to-reviewers letter (R1-F) ──
	n_head = int(head_mask.sum().item())
	n_rare = int(rare_mask.sum().item())
	if verbose:
		print(f"\n  [Reference frequency distribution (multimodal, shared vocab)]")
		print(f"  ├─ freq [min, max]         : [{shared_train_freq.min():.0f}, {shared_train_freq.max():.0f}]")
		print(f"  ├─ freq mean / std         : {shared_train_freq.mean():.1f} / {shared_train_freq.std():.1f}")
		print(f"  ├─ freq median             : {shared_train_freq.median():.1f}")
		print(f"  ├─ classes with freq == 1  : {(shared_train_freq == 1).sum().item():,}")
		print(f"  ├─ classes with freq <= 5  : {(shared_train_freq <= 5).sum().item():,}")
		print(f"  └─ classes with freq > 10  : {(shared_train_freq > 10).sum().item():,}")

		print(f"\n  [Tier assignment]")
		print(f"  ├─ Shared classes          : {len(shared_class_names):,}")
		print(f"  ├─ Head (Pareto {pareto_threshold:.0%})       : {n_head:,}  (cutoff idx = {pareto_cutoff})")
		print(f"  ├─ Rare (bottom {rare_percentile:.0%})      : {n_rare:,}")
		if not torch.isnan(rare_frequency_threshold):
			print(f"  ├─ Rare freq threshold     : {rare_frequency_threshold.item():.1f}")
		else:
			print(f"  ├─ Rare freq threshold     : n/a (degenerate)")
		# neither head nor rare == "body"
		body_n = len(shared_class_names) - n_head - n_rare
		print(f"  ├─ Body (neither)          : {body_n:,}")

		# Explicit small-N caveats — the exact things R1-F would flag if unremarked
		if n_rare > 0 and n_rare < 10:
			print(f"\n  ⚠  Rare tier has only {n_rare} class(es). Metrics over this tier "
				  f"are HIGH-VARIANCE / small-sample; report n_rare prominently and "
				  f"consider variance estimates in the writeup (R1-F).")
		if n_head < 10:
			print(f"  ⚠  Head tier has only {n_head} class(es) — also small-N.")

	shared_protocol = {
		"protocol_name": "shared_intersection_mm_reference_v1",
		"reference_label_column": multimodal_col,
		"pareto_threshold": pareto_threshold,
		"rare_percentile": rare_percentile,
		"shared_class_names": shared_class_names,
		"shared_train_freq": shared_train_freq.tolist(),
		"active_mask": active_mask.tolist(),
		"head_mask": head_mask.tolist(),
		"rare_mask": rare_mask.tolist(),
		"n_classes": len(shared_class_names),
		"n_head_classes": n_head,
		"n_rare_classes": n_rare,
		"rare_frequency_threshold": (
			rare_frequency_threshold.item()
			if not torch.isnan(rare_frequency_threshold)
			else None
		),
	}

	with open(protocol_path, "w", encoding="utf-8") as file:
		json.dump(shared_protocol, file, indent=2)

	if verbose:
		print(f"  └─ Saved protocol          : {protocol_path}")
		print(f"{'='*70}\n")

	return shared_protocol
