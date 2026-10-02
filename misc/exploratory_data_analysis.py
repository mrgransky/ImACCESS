from utils import *
import visualize as viz

def get_top_labels_per_source(
	processed_dfs: dict,
	output_dir: str,
	top_n: int=100,
	DPI: int=200
):
	print(f"\n>> TOP-{top_n} MOST FREQUENT LABELS PER-SOURCE: {list(processed_dfs.keys())}")
	
	all_label_counts = {}
	# PART 1: Individual Source Analysis
	for col, lst in processed_dfs.items():
		source_labels = list()

		for labels in lst:
			# Handle NaN and non-list values
			if isinstance(labels, list):
				# It is a valid list of labels, proceed
				pass
			elif pd.isna(labels):
				# print(f"<!> {col} containing {labels} => skipping!")
				continue
			elif isinstance(labels, str):
				try:
					labels = ast.literal_eval(labels)  # Parse string representation of list
				except Exception as e:
					print(f"<!> {col} containing {labels} => skipping! {e}")
					continue
			else:
				print(f"<!> {col} containing {type(labels)} {labels} => skipping!")
				continue

			source_labels.extend(labels)

		print(f"\n{col} {type(source_labels)} {len(source_labels)} samples")
		source_unique = sorted(list(set(source_labels)))
		print(f"Total unique: {type(source_unique)} {len(source_unique)}")

		# Count frequencies
		source_counts = Counter(source_labels)
		source_counts_df = pd.DataFrame(
			source_counts.items(), 
			columns=['Label', 'Count']
		).sort_values(by='Count', ascending=False)
		
		# Store for later comparison
		all_label_counts[col] = source_counts_df
		
		# Singleton analysis
		source_singletons = source_counts_df[source_counts_df['Count'] == 1]['Label'].tolist()
		print(
			f"[SINGLETONS] {type(source_singletons)}: "
			f"{len(source_singletons)}/{len(source_unique)} "
			f"({len(source_singletons) / len(source_unique) * 100:.2f}%):"
		)
		print(source_singletons[:25])

		# Create individual visualization for this source
		plt.figure(figsize=(14, 12))
		plot_data = source_counts_df.head(top_n)
		sns.barplot(x='Count', y='Label', data=plot_data, palette='viridis')
		# plt.title(
		# 	f'Top-{top_n} Most Frequent Labels ({col})', 
		# 	fontsize=11, 
		# 	weight='bold'
		# )
		plt.xlabel('Samples', fontsize=10)
		plt.ylabel('Label', fontsize=10)
		plt.tight_layout()
		plt.savefig(
			fname=os.path.join(output_dir, f"top_{top_n}_most_frequent_labels_{col}.png"),
			dpi=DPI,
			bbox_inches='tight',
		)
		plt.close()
	
	# Create side-by-side comparison plot
	n_sources = len(all_label_counts)
	fig, axes = plt.subplots(1, n_sources, figsize=(8*n_sources, 11))
	
	if n_sources == 1:
		axes = [axes]
	
	for idx, (source_name, counts_df) in enumerate(all_label_counts.items()):
		ax = axes[idx]
		plot_data = counts_df.head(top_n)
		
		# Create horizontal bar plot
		y_pos = np.arange(len(plot_data))
		ax.barh(y_pos, plot_data['Count'].values, color="#002d4d", alpha=0.8)
		ax.set_yticks(y_pos)
		ax.set_yticklabels(plot_data['Label'].values, fontsize=7)
		ax.invert_yaxis()  # Top label at top
		ax.set_xlabel('Frequency', fontsize=11)
		ax.set_title(
			f'{source_name} (Top-{len(plot_data)} Most Frequent Labels) [Total: {len(counts_df)}]',
			fontsize=10,
			weight='bold'
		)
		ax.grid(axis='x', alpha=0.5)
	
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"top_{top_n}_comparative_labels_x{n_sources}_sources_{'_'.join(list(processed_dfs.keys()))}.png"),
		dpi=DPI,
		bbox_inches='tight'
	)
	plt.close()
	
	# Get top-N from each source
	print(f"\n>> Top-{top_n} Label Agreement Between Sources")
	llm_top_n = set(
		all_label_counts[
			next(k for k in processed_dfs.keys() if k.startswith("llm"))
		].head(top_n)['Label'].values
	)
	vlm_top_n = set(
		all_label_counts[
			next(k for k in processed_dfs.keys() if k.startswith("vlm"))
		].head(top_n)['Label'].values
	)
	
	agreement = llm_top_n & vlm_top_n
	llm_unique_top = llm_top_n - vlm_top_n
	vlm_unique_top = vlm_top_n - llm_top_n
	
	print(f"  Agreed by both: {len(agreement)} labels")
	print(f"  Only in LLM top-{top_n}: {len(llm_unique_top)} labels")
	print(f"  Only in VLM top-{top_n}: {len(vlm_unique_top)} labels")
	print(f"  Agreement rate: {len(agreement)/top_n*100:.1f}%")
		
	# Create agreement visualization
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
	
	# Pie chart of agreement
	agreement_data = [len(agreement), len(llm_unique_top), len(vlm_unique_top)]
	agreement_labels = [
		f'Both ({len(agreement)})',
		f'LLM-only ({len(llm_unique_top)})',
		f'VLM-only ({len(vlm_unique_top)})'
	]
	colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
	
	ax1.pie(agreement_data, labels=agreement_labels, autopct='%1.1f%%', colors=colors, startangle=90)
	ax1.set_title(f'Agreement Among Top-{top_n} Labels', fontsize=12, weight='bold')
	
	# Venn-style bar chart
	categories = ['Agreed', 'LLM-only', 'VLM-only']
	ax2.bar(categories, agreement_data, color=colors, alpha=0.8, edgecolor='black')
	ax2.set_ylabel('Number of Labels', fontsize=11)
	ax2.set_title(f'Top-{top_n} Label Distribution', fontsize=12, weight='bold')
	ax2.grid(axis='y', alpha=0.3)
	for i, (cat, val) in enumerate(zip(categories, agreement_data)):
		ax2.text(i, val, str(val), ha='center', va='bottom', fontsize=11, weight='bold')
	
	plt.tight_layout()
	plt.savefig(
		fname=os.path.join(output_dir, f"top_{top_n}_labels_agreement_{'_'.join(list(processed_dfs.keys()))}.png"),
		dpi=DPI,
		bbox_inches='tight'
	)
	plt.close()
		
	return all_label_counts

def multilabel_eda(
	df: pd.DataFrame,
	label_column: str,
	output_dir: str,
	top_n: int = 100,
	n_top_labels_co_occurrence: int = 50,
	DPI: int = 200,
	verbose: bool = False,
):
	print(f"\nMulti-label EDA {type(df)} {df.shape} (column: {label_column})")
	eda_st = time.time()
	dataset_dir = os.path.dirname(output_dir)
	dataset_name = os.path.basename(dataset_dir)
	viz_dir = os.path.join(output_dir, "viz")
	os.makedirs(viz_dir, exist_ok=True)
	print(f"{dataset_name}: {type(df)} {df.shape}\n{list(df.columns)}")
	print(df.info(verbose=True, memory_usage="deep"))
	all_individual_labels = []
	for labels in df[label_column].tolist():
			if isinstance(labels, str):
					labels = ast.literal_eval(labels)
			all_individual_labels.extend(labels)
	unique_labels = sorted(list(set(all_individual_labels)))
	label_cardinality = df[label_column].apply(len)
	if verbose:
			print(f"\nLabel Cardinality ({label_column})")
			print(f"  ├─ {type(df)} {df.shape}")
			print(f"  ├─ unique labels: {len(unique_labels)}")
			print(f"  └─ {unique_labels[:10]}")
			print(label_cardinality.describe())

	# 1. Raw labels agreement
	processed_dfs_raw_labels = {
			"llm_based_labels": df["llm_based_labels"].tolist(),
			"vlm_based_labels": df["vlm_based_labels"].tolist(),
			"multimodal_labels": df["multimodal_labels"].tolist(),
	}
	all_label_counts = get_top_labels_per_source(
			processed_dfs=processed_dfs_raw_labels,
			top_n=top_n,
			output_dir=viz_dir,
			DPI=DPI,
	)
	viz.plot_multi_source_agreement(
			processed_dfs=processed_dfs_raw_labels,
			output_dir=viz_dir,
			DPI=DPI,
	)

	# 2. Canonical labels agreement
	processed_dfs_canonical_labels = {
			"llm_canonical_labels": df["llm_canonical_labels"].tolist(),
			"vlm_canonical_labels": df["vlm_canonical_labels"].tolist(),
			"multimodal_canonical_labels": df["multimodal_canonical_labels"].tolist(),
	}
	all_label_counts = get_top_labels_per_source(
			processed_dfs=processed_dfs_canonical_labels,
			top_n=top_n,
			output_dir=viz_dir,
			DPI=DPI,
	)
	viz.plot_multi_source_agreement(
			processed_dfs=processed_dfs_canonical_labels,
			output_dir=viz_dir,
			DPI=DPI,
	)

	# 3. Power Law Analysis
	print(f"\n[POWER LAW ANALYSIS] {label_column}")
	freq_values = all_label_counts[label_column]["Count"].values
	ranks = np.arange(1, len(freq_values) + 1)
	coeffs = np.polyfit(np.log(ranks), np.log(freq_values), 1)
	alpha_estimate = -coeffs[0]
	print(f"Estimated power law exponent (α): {alpha_estimate:.3f}")
	print("Note: α ≈ 2 suggests Zipf's law. α > 1 suggests a power law distribution.")
	viz.plot_power_law_analysis(
			ranks=ranks,
			freq_values=freq_values,
			coeffs=coeffs,
			alpha_estimate=alpha_estimate,
			label_column=label_column,
			output_dir=viz_dir,
			dpi=DPI,
	)

	# 4. Diversity Metrics & Lorenz Curve
	print("\n>> LABEL DIVERSITY METRICS")
	label_probs = freq_values / freq_values.sum()
	shannon_entropy = scipy.stats.entropy(label_probs, base=2)
	max_entropy = np.log2(len(unique_labels))
	normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
	sorted_counts = np.sort(freq_values)
	n = len(sorted_counts)
	gini = (2 * np.sum(np.arange(1, n + 1) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
	effective_labels = 2 ** shannon_entropy
	print(f"Shannon Entropy: {shannon_entropy:.3f} bits")
	print(f"Maximum Possible Entropy: {max_entropy:.3f} bits")
	print(f"Normalized Entropy: {normalized_entropy:.3f} (1.0 = perfectly uniform)")
	print(f"Gini Coefficient: {gini:.3f} (0 = perfect equality, 1 = perfect inequality)")
	print(f"Effective Number of Labels: {effective_labels:.1f} (perplexity measure: 2^H)")
	viz.plot_comparative_lorenz_curve(
			sorted_counts=sorted_counts,
			gini=gini,
			dataset_name=dataset_name,
			label_column=label_column,
			output_dir=viz_dir,
			dpi=DPI,
	)

	# 5. Label Imbalance Analysis
	print("\n>> LABEL IMBALANCE ANALYSIS")
	counts_df = all_label_counts[label_column]
	max_freq = counts_df["Count"].max()
	min_freq = counts_df["Count"].min()
	imbalance_ratio = max_freq / min_freq
	rare_labels = counts_df[counts_df["Count"] < (max_freq * 0.01)]
	print(f"Imbalance Ratio (Max/Min): {imbalance_ratio:.3f}")
	print(f"Label Frequency: mean: {counts_df['Count'].mean():.2f} | median: {counts_df['Count'].median():.2f} | max: {max_freq:.2f} | min: {min_freq:.2f}")
	print(f"Number of rare labels (< 1% of max): {len(rare_labels)} ({len(rare_labels)/len(unique_labels)*100:.1f}%)")
	viz.plot_imbalance_analysis(
			counts_df=counts_df,
			unique_labels_count=len(unique_labels),
			label_column=label_column,
			output_dir=viz_dir,
			dpi=DPI,
	)

	# 6. Unique Label Combinations
	print("\n>> Unique Label Set Combinations")
	def parse_and_create_tuple(x):
			if isinstance(x, str):
					try:
							x = ast.literal_eval(x)
					except Exception:
							return tuple()
			return tuple(sorted(x)) if isinstance(x, list) and len(x) > 0 else tuple()
	label_sets = df[label_column].apply(parse_and_create_tuple)
	unique_label_sets = Counter(label_sets)
	print(f"Total number of unique label combinations: {len(unique_label_sets)}/{len(df)} ({len(unique_label_sets)/len(df)*100:.2f}%)")
	for label_set, count in unique_label_sets.most_common(10):
			print(f"{count:4d}x | {len(label_set):2d} labels | {', '.join(label_set)}")
	unique_label_sets_df = pd.DataFrame(
			unique_label_sets.items(), columns=["Label Set", "Count"]
	).sort_values(by="Count", ascending=False)
	viz.plot_unique_label_combinations(
			unique_label_sets_df=unique_label_sets_df,
			label_column=label_column,
			output_dir=viz_dir,
			dpi=DPI,
	)

	# 7. Hierarchical Clustering / Correlation Network
	print("\n>> HIERARCHICAL CLUSTERING OF LABELS (Top Labels)")
	if n_top_labels_co_occurrence > len(unique_labels):
		print(f"[WARNING] n_top_labels_co_occurrence adjusted to total unique labels ({len(unique_labels)}).")
		n_top_labels_co_occurrence = len(unique_labels)

	if n_top_labels_co_occurrence >= 2:
		top_labels_for_correlation = counts_df["Label"].head(n_top_labels_co_occurrence).tolist()
		mlb = MultiLabelBinarizer(classes=unique_labels, sparse_output=True)
		def parse_labels(x):
			if isinstance(x, str):
				return ast.literal_eval(x)
			return x if isinstance(x, list) else []

		y_binarized = mlb.fit_transform(df[label_column].apply(parse_labels))
		labels_in_order = list(mlb.classes_)
		top_label_indices = [labels_in_order.index(lab) for lab in top_labels_for_correlation]
		y_subset = y_binarized[:, top_label_indices].toarray()

		jaccard_matrix = 1 - pairwise_distances(y_subset.T, metric="jaccard")

		jaccard_df = pd.DataFrame(
			jaccard_matrix,
			index=top_labels_for_correlation,
			columns=top_labels_for_correlation,
		)
		viz.plot_jaccard_heatmap(
			jaccard_df=jaccard_df,
			label_column=label_column,
			output_dir=viz_dir,
			dpi=DPI,
		)

		viz.plot_label_cooccurrence_network(
			jaccard_matrix=jaccard_matrix,
			labels=top_labels_for_correlation,
			label_column=label_column,
			output_dir=viz_dir,
			threshold=0.01,
			dpi=DPI,
		)
	else:
		print("Not enough unique labels to display correlation analyses (need at least 2).")

	# 8. Comparative Radar Chart & Summary
	summary_stats_dict = {
		"Dataset Name": dataset_name,
		"Total Samples": len(df),
		"Unique Labels": len(unique_labels),
		"Unique Label Combinations": len(unique_label_sets),
		"Mean Label Cardinality": label_cardinality.mean(),
		"Median Label Cardinality": label_cardinality.median(),
		"Max Label Cardinality": int(label_cardinality.max()),
		"Shannon Entropy": shannon_entropy,
		"Normalized Entropy": normalized_entropy,
		"Gini Coefficient": gini,
		"Imbalance Ratio": imbalance_ratio,
		"Power Law Exponent (α)": alpha_estimate,
		"Effective # of Labels": effective_labels,
		"freqs": freq_values,
	}

	viz.plot_comparative_radar_chart(
		summary_stats_dict=summary_stats_dict,
		label_col=label_column,
		output_dir=viz_dir,
	)

	print("\nCOMPREHENSIVE SUMMARY STATISTICS")

	# Automatically format floats to .3f and ignore 'freqs'
	table_data = []
	for metric, val in summary_stats_dict.items():
		if metric == "freqs":
			continue
		formatted_val = f"{val:.3f}" if isinstance(val, (float, np.floating)) else str(val)
		table_data.append({"Metric": metric, "Value": formatted_val})

	summary_df = pd.DataFrame(table_data)

	print(summary_df.to_string(index=False))

	print(f"\n[EDA TOTAL ELAPSED TIME] {time.time()-eda_st:.1f} sec")
	print("=" * 100)