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

def get_multi_label_stratified_split_x1(
	df: pd.DataFrame,
	csv_file: str,
	val_split_pct: float,
	label_col: str = 'multimodal_labels',
	min_label_frequency: int = None,        # None triggers principled auto-threshold
	min_val_label_count: int = 2,           # min expected occurrences in val set
	stratification_order: int = 2,          # default to order=2 for better quality
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Split a multi-label dataset into stratified train/val sets using IterativeStratification.
	Args:
			df:                    Input DataFrame.
			csv_file:              Output CSV path stem (will produce _train.csv / _val.csv).
			val_split_pct:         Fraction of data to place in validation set (e.g. 0.35).
			label_col:             Column containing lists of string labels.
			min_label_frequency:   Minimum global label count to be retained. If None, derived
														 from min_val_label_count and val_split_pct (recommended).
			min_val_label_count:   Minimum expected occurrences of a label in the val split.
														 Only used when min_label_frequency is None.
			stratification_order:  IterativeStratification order.
														 1 = independent per-label (faster, lower quality).
														 2 = pairwise co-occurrence (slower, recommended for <500k rows).
	"""

	# Auto-set min_label_frequency if not provided, using a principled frequency threshold
	if min_label_frequency is None:
		min_label_frequency = math.ceil(min_val_label_count / val_split_pct)
		_freq_source = (
			f"auto (ceil({min_val_label_count} / {val_split_pct}) "
			f"= {min_label_frequency})"
		)
	else:
		_freq_source = f"manual ({min_label_frequency})"
	
	print("-" * 150)
	print(
		f"[MULTI-LABEL STRATIFIED SPLIT] | val={val_split_pct} | "
		f"order={stratification_order} | min_label_freq={min_label_frequency} [{_freq_source}]"
	)
	t_st = time.time()
	# ── STEP 1: Robust label parsing ─────────────────────────────────────────
	print(f"\n[1/6] Parsing '{label_col}' column...")
	if label_col not in df.columns:
			raise ValueError(f"Label column '{label_col}' not found in the DataFrame.")
	def parse_label(x):
			if isinstance(x, str):
					try:
							return ast.literal_eval(x)
					except (ValueError, SyntaxError) as e:
							raise ValueError(
									f"Malformed string in '{label_col}': '{x}'. Error: {e}"
							)
			elif isinstance(x, list):
					return x
			else:
					print(
							f"   Warning: Unexpected type '{type(x)}' for value '{x}'. "
							f"Treating as empty list."
					)
					return []
	df_copy = df.copy()
	try:
			df_copy[label_col] = df_copy[label_col].apply(parse_label)
			print(f"   ✓ Parsed '{label_col}' column.")
	except ValueError as e:
			raise ValueError(f"Error parsing multi-label column '{label_col}': {e}")

	# ── STEP 2: Drop rows with empty label lists ──────────────────────────────
	print(f"\n[2/6] Removing samples with empty labels...")
	
	# .copy() to avoid SettingWithCopyWarning on downstream mutations
	df_filtered = df_copy[df_copy[label_col].apply(len) > 0].copy()
	n_removed_empty = len(df_copy) - len(df_filtered)
	
	if n_removed_empty:
		print(f"   Removed {n_removed_empty} rows with empty label lists.")
	if len(df_filtered) == 0:
		raise ValueError("No samples with non-empty label lists remain after parsing.")
	
	print(f"   DataFrame shape: {df_filtered.shape}")

	# ── STEP 3: Filter rare labels
	print(
			f"\n[3/6] Filtering rare labels "
			f"(min_label_frequency={min_label_frequency})..."
	)
	all_labels_flat = [l for labels in df_filtered[label_col] for l in labels]
	label_counts = Counter(all_labels_flat)
	initial_unique = len(label_counts)
	print(f"   Total unique labels before filtering: {initial_unique}")
	rare_labels = {l for l, c in label_counts.items() if c < min_label_frequency}
	kept_labels = set(label_counts.keys()) - rare_labels
	print(
			f"   Rare labels (< {min_label_frequency}): {len(rare_labels)} "
			f"({len(rare_labels) / initial_unique * 100:.1f}%)"
	)
	print(
			f"   Labels to keep: {len(kept_labels)} "
			f"({len(kept_labels) / initial_unique * 100:.1f}%)"
	)
	if rare_labels:
			rare_freq_dist = Counter(label_counts[l] for l in rare_labels)
			print(f"   Rare label frequency distribution:")
			for freq in sorted(rare_freq_dist.keys()):
					labels_at_freq = [l for l, c in label_counts.items() if c == freq]
					print(
							f"      freq={freq}: {rare_freq_dist[freq]} labels: "
							f"{labels_at_freq[:20]}"
							f"{'  ...' if len(labels_at_freq) > 20 else ''}"
					)
	
	# assign via loc to avoid chained-assignment warning
	df_filtered[label_col] = df_filtered[label_col].apply(
		lambda llist: [l for l in llist if l not in rare_labels]
	)
	n_before = len(df_filtered)
	df_filtered = df_filtered[df_filtered[label_col].apply(len) > 0].copy()
	n_after = len(df_filtered)
	
	print(
		f"   Samples after rare-label filtering: {n_after} "
		f"(removed {n_before - n_after} that became label-empty)"
	)
	if n_after == 0:
		raise ValueError(
			"No samples remain after filtering rare labels. "
			"Try lowering min_label_frequency or min_val_label_count."
		)
	final_unique = len({l for labels in df_filtered[label_col] for l in labels})
	print(f"   Final unique labels: {final_unique}")

	# ── STEP 4: Binarise label matrix
	print(f"\n[4/6] Binarizing label matrix ({df_filtered.shape[0]} samples)...")
	mlb = MultiLabelBinarizer(sparse_output=True)
	label_matrix = mlb.fit_transform(df_filtered[label_col])
	unique_labels = mlb.classes_
	density = label_matrix.count_nonzero() / (label_matrix.shape[0] * label_matrix.shape[1])
	print(
			f"   Shape: {label_matrix.shape} | dtype: {label_matrix.dtype} | "
			f"Density: {density * 100:.3f}% | "
			f"Non-zeros: {label_matrix.count_nonzero()} | "
			f"Data size: {label_matrix.data.nbytes / 1e6:.3f} MB"
	)
	if len(unique_labels) == 0:
			raise ValueError("No unique labels after processing. Cannot stratify.")
	print(f"   Sample labels: {unique_labels.tolist()[:20]}")

	# ── STEP 5: Iterative stratification
	# expose stratification_order as a parameter so callers can
	# compare order=1 (fast) vs order=2 (better label-pair coverage).
	print(f"\n[5/6] Iterative stratification (order={stratification_order}, n={len(df_filtered)})")
	X_indices = np.arange(len(df_filtered)).reshape(-1, 1)
	try:
		stratifier = IterativeStratification(
			n_splits=2,
			order=stratification_order,
			sample_distribution_per_fold=[val_split_pct, 1.0 - val_split_pct],
		)
		train_indices, val_indices = next(stratifier.split(X_indices, label_matrix))
	except Exception as e:
		print(f"   ❌ Stratification failed: {e}")
		print(
			f"   Hint: some labels may still have too few samples. "
			f"Try raising min_label_frequency or switching to order=1."
		)
		raise
	train_df = df_filtered.iloc[train_indices].reset_index(drop=True)
	val_df   = df_filtered.iloc[val_indices].reset_index(drop=True)
	if train_df.empty or val_df.empty:
		raise ValueError("Train or validation set is empty after splitting.")
	
	# ── STEP 6: Post-split label coverage audit ───────────────────────────────
	# verify every retained label appears in both splits
	print(f"\n[6/6] Post-split label coverage audit...")
	train_label_set = {l for labels in train_df[label_col] for l in labels}
	val_label_set   = {l for labels in val_df[label_col]   for l in labels}
	train_only = train_label_set - val_label_set
	val_only   = val_label_set   - train_label_set
	both       = train_label_set & val_label_set
	
	print(f"   Labels in both splits : {len(both)}")
	print(f"   Labels only in train  : {len(train_only)}")
	print(f"   Labels only in val    : {len(val_only)}")
	
	if train_only:
		print(f"   ⚠ Train-only label examples: {sorted(train_only)[:20]}")
	if val_only:
		print(f"   ⚠ Val-only label examples  : {sorted(val_only)[:20]}")
	if len(train_only) == 0 and len(val_only) == 0:
		print("   ✓ All labels present in both splits.")
	
	print(f"\nSPLIT SUMMARY")
	print(f"   Original : {df_filtered.shape}")
	print(f"   Train    : {train_df.shape} ({len(train_df) / len(df_filtered) * 100:.1f}%)")
	print(f"   Val      : {val_df.shape} ({len(val_df) / len(df_filtered) * 100:.1f}%)")
	
	train_path = csv_file.replace('.csv', '_train.csv')
	val_path   = csv_file.replace('.csv', '_val.csv')
	
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path,   index=False)

	print(f"\n   Total elapsed: {time.time() - t_st:.1f} s")
	print("-" * 100)

	return train_df, val_df

def get_multi_label_stratified_split_x2(
	df: pd.DataFrame,
	csv_file: str,
	val_split_pct: float,
	label_col: str,
	min_val_label_count: int = 2, # minimum number of samples per label in validation set
) -> Tuple[pd.DataFrame, pd.DataFrame]:

	print(f"-"*100)
	print(f"[MULTI-LABEL STRATIFIED SPLIT] {val_split_pct} train/val split")

	t_st = time.time()
	df_copy = df.copy()

	# Check if the label column exists
	if label_col not in df_copy.columns:
		raise ValueError(f"Label column '{label_col}' not found in the DataFrame.")

	# Get min_label_frequency
	min_label_frequency = math.ceil(min_val_label_count / val_split_pct)
	print(f">> Min label frequency: {min_label_frequency} (= ceil({min_val_label_count}/{val_split_pct})")

	# 1. Robust Label Parsing
	print(f"\n[1/5] Parsing '{label_col}' column...")
	if label_col not in df_copy.columns:
		raise ValueError(f"Label column '{label_col}' not found in the DataFrame.")

	def parse_label(x):
		if isinstance(x, str):
			try:
				return ast.literal_eval(x)
			except (ValueError, SyntaxError) as e:
				raise ValueError(f"Malformed string found in '{label_col}': '{x}'. Error: {e}")
		elif isinstance(x, list):
			return x
		else:
			print(f"Warning: Unexpected type '{type(x)}' found in '{label_col}': {x}. Trying to convert to empty list.")
			return []
	
	try:
		df_copy[label_col] = df_copy[label_col].apply(parse_label)
		print(f"   ✓ Successfully processed '{label_col}' column.")
	except ValueError as e:
		raise ValueError(f"Error parsing multi-label column '{label_col}'. Error: {e}")
	
	# 2. Remove rows with empty label lists
	print(f"\n[2/5] Removing samples with empty labels...")
	df_filtered = df_copy[df_copy[label_col].apply(len) > 0].copy()
	initial_rows = len(df_copy)
	final_rows = len(df_filtered)
	
	if final_rows == 0:
		raise ValueError("No samples with non-empty label lists remain after parsing.")
	
	if initial_rows != final_rows:
		print(f"   Removed {initial_rows - final_rows} rows with empty label lists.")
	
	print(f"df_filtered: {df_filtered.shape}")

	print(f"\n[3/5] Filtering rare labels (min frequency: {min_label_frequency})...")
	# Count label frequencies
	all_labels = []
	for label_list in df_filtered[label_col]:
		all_labels.extend(label_list)
	
	label_counts = Counter(all_labels)
	initial_unique_labels = len(label_counts)
	print(f"\tTotal unique labels before filtering: {initial_unique_labels}")
	
	# Identify rare labels
	rare_labels = {
		label 
		for label, count in label_counts.items() 
		if count < min_label_frequency
	}

	kept_labels = set(label_counts.keys()) - rare_labels
	
	print(f"\tRare labels (< {min_label_frequency}): {len(rare_labels)} ({len(rare_labels)/initial_unique_labels*100:.1f}%)")
	print(f"\tLabels to keep: {len(kept_labels)} ({len(kept_labels)/initial_unique_labels*100:.1f}%)")
	
	if rare_labels:
		# Show frequency distribution of rare labels
		rare_freq_dist = Counter([label_counts[label] for label in rare_labels])
		print(f"\tRare label frequency/Occurences distribution:")
		for freq in sorted(rare_freq_dist.keys()):
			count = rare_freq_dist[freq]
			# Get labels with this frequency
			labels_with_freq = [label for label, lbl_count in label_counts.items() if lbl_count == freq]
			print(f"\t\tfreq={freq}: {count} labels: {labels_with_freq[:20]}{' ...' if len(labels_with_freq) > 20 else ''}")
		
		# Show examples
		rare_examples = sorted(rare_labels)[:20]
		print(f"\trare labels being removed: {rare_examples}")
	
	# Filter out rare labels from each sample
	def remove_rare_labels(label_list):
		return [label for label in label_list if label not in rare_labels]
	
	df_filtered[label_col] = df_filtered[label_col].apply(remove_rare_labels)
	
	# Remove samples that became empty after rare label removal
	samples_before = len(df_filtered)
	df_filtered = df_filtered[df_filtered[label_col].apply(len) > 0]

	samples_after = len(df_filtered)
	
	if samples_after == 0:
		raise ValueError("No samples remain after filtering rare labels. Try lowering min_label_frequency.")
	
	print(f"Samples after filtering: {samples_after} (removed {samples_before - samples_after})")
	
	# Verify final label count
	final_labels = set([l for labels in df_filtered[label_col] for l in labels])
	print(f"Final unique labels: {len(final_labels)}")

	print(f"\n[4/5] Binarizing ({df_filtered[label_col].shape} labels using MultiLabelBinarizer")
	mlb = MultiLabelBinarizer(sparse_output=True)
	label_matrix = mlb.fit_transform(df_filtered[label_col])
	print(f"Label matrix: {type(label_matrix)} {label_matrix.shape} | {label_matrix.dtype} | Density: {label_matrix.count_nonzero() / np.prod(label_matrix.shape) * 100:.1f}% | Non-zeroes: {label_matrix.count_nonzero()} | Data size: ({label_matrix.data.nbytes / 1e6:.3f} MB)")
	unique_labels = mlb.classes_
	if len(unique_labels) == 0:
		raise ValueError("No unique labels found after processing. Cannot perform stratification.")
	
	print(f"{len(unique_labels)} unique labels:\n{unique_labels.tolist()[:20]}")
	
	# Stratify the data
	stratification_order = 1 if len(df_filtered) > int(2e5) else 2 # 1 for large datasets, 2 for small datasets
	print(f"\n[5/5] Iterative stratification (order={stratification_order} len(df_filtered): {len(df_filtered)} val: {val_split_pct})")
	X_indices = np.arange(len(df_filtered)).reshape(-1, 1)
	
	try:
		stratifier = IterativeStratification(
			n_splits=2,
			order=stratification_order,
			sample_distribution_per_fold=[val_split_pct, 1-val_split_pct],
		)
		train_indices, val_indices = next(stratifier.split(X_indices, label_matrix))
	except Exception as e:
		print(f"   ❌ Stratification failed: {e}")
		print(f"   This may indicate labels with insufficient samples.")
		print(f"   Try increasing min_label_frequency (current: {min_label_frequency})")
		raise e
	
	train_original_indices = df_filtered.iloc[train_indices].index.values
	val_original_indices = df_filtered.iloc[val_indices].index.values
	train_df = df_filtered.loc[train_original_indices].reset_index(drop=True)
	val_df = df_filtered.loc[val_original_indices].reset_index(drop=True)
	
	# Verify Split
	if train_df.empty or val_df.empty:
		raise ValueError("Train or validation set is empty after splitting.")
	
	print(f"\n[6/6] Post-split label coverage audit...")
	train_labels = set(l for labels in train_df[label_col] for l in labels)
	val_labels   = set(l for labels in val_df[label_col]   for l in labels)
	train_only = train_labels - val_labels
	val_only   = val_labels - train_labels
	common_labels = train_labels & val_labels
	print(f"{len(common_labels)} common labels in both train & val")
	print(f"{len(train_only)} train_only labels: {train_only}")
	print(f"{len(val_only)} val_only labels: {val_only}")

	print(f"\nSPLIT SUMMARY")
	print(f"   Original: {df_filtered.shape}")
	print(f"   Train:    {train_df.shape} ({len(train_df)/len(df_filtered)*100:.1f}%)")
	print(f"   Val:      {val_df.shape} ({len(val_df)/len(df_filtered)*100:.1f}%)")
	
	# print(f"\n>> TRAIN SET")
	# print(f"   Samples: {len(train_df)}")
	# print(f"   Unique label combinations: {train_df[label_col].apply(tuple).nunique()}")
	# sample_size = min(1000, len(train_df))
	# print(f"   Label distribution (sample of {sample_size}):")
	# print(train_df[label_col].sample(sample_size).apply(tuple).value_counts().head(10))
	
	# print(f"\n>> VAL SET")
	# print(f"   Samples: {len(val_df)}")
	# print(f"   Unique label combinations: {val_df[label_col].apply(tuple).nunique()}")
	# sample_size = min(1000, len(val_df))
	# print(f"   Label distribution (sample of {sample_size}):")
	# print(val_df[label_col].sample(sample_size).apply(tuple).value_counts().head(10))
	
	train_path = csv_file.replace('.csv', '_train.csv')
	val_path = csv_file.replace('.csv', '_val.csv')
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	
	print(f"\n[MULTI-LABEL STRATIFIED SPLIT] DONE in {time.time()-t_st:.1f} sec")
	print(f"-"*100)

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
	print("-" * 150)
	print(f"[MULTI-LABEL STRATIFIED SPLIT]")
	print(f"  ├─ val_split_pct        : {val_split_pct}")
	print(f"  ├─ stratification_order : {stratification_order}")
	print(f"  ├─ min_label_frequency  : {min_label_frequency}  ({_freq_source})")
	print(f"  ├─ label_col: {label_col}")
	print(f"  └─ df: {df.shape} {df.columns.tolist()}")

	t_st = time.time()

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
	print(f"   Rare labels (< {min_label_frequency})  : {len(rare_labels)} ({len(rare_labels) / initial_unique * 100:.1f}%)")
	print(f"   Labels to keep                  : {len(kept_labels)} ({len(kept_labels) / initial_unique * 100:.1f}%)")
	if rare_labels:
			rare_freq_dist = Counter(label_counts[l] for l in rare_labels)
			print(f"   Rare label frequency distribution:")
			for freq in sorted(rare_freq_dist.keys()):
					labels_at_freq = [l for l, c in label_counts.items() if c == freq]
					print(f"      freq={freq}: {rare_freq_dist[freq]} labels: "
								f"{labels_at_freq[:20]}{'  ...' if len(labels_at_freq) > 20 else ''}")
			# x2-style flat example listing, useful for quick eyeballing in logs
			rare_examples = sorted(rare_labels)[:20]
			print(f"   Rare labels being removed (examples): {rare_examples}"
						f"{'  ...' if len(rare_labels) > 20 else ''}")
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
			print(f"   ⚠ Train-only label examples: {sorted(train_only)[:20]}"
						f"{'  ...' if len(train_only) > 20 else ''}")
	if val_only:
			print(f"   ⚠ Val-only label examples  : {sorted(val_only)[:20]}"
						f"{'  ...' if len(val_only) > 20 else ''}")
	if not train_only and not val_only:
			print("   ✓ All labels present in both splits.")

	# ── Summary ───────────────────────────────────────────────────────────────
	print(f"\n>> SPLIT SUMMARY")
	print(f"   Original : {df_filtered.shape}")
	print(f"   Train    : {train_df.shape}  ({len(train_df) / n_after * 100:.1f}%)")
	print(f"   Val      : {val_df.shape}  ({len(val_df) / n_after * 100:.1f}%)")
	train_path = csv_file.replace('.csv', '_train.csv')
	val_path = csv_file.replace('.csv', '_val.csv')
	train_df.to_csv(train_path, index=False)
	val_df.to_csv(val_path, index=False)
	print(f"   Saved → {train_path}")
	print(f"   Saved → {val_path}")
	print(f"\n   Total elapsed: {time.time() - t_st:.1f}s")
	print("-" * 100)

	return train_df, val_df