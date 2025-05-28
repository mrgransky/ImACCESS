import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde, t
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import inspect

import ast # To safely evaluate the string representation of lists
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import os # For checking file existence
from itertools import combinations # For pairwise label combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast # To safely evaluate the string representation of lists
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import os # For checking file existence
from itertools import combinations # For pairwise label combinations

def perform_multilabel_eda(
    data_path: str,
    label_column: str = 'multimodal_labels',
    n_top_labels_plot: int = 30,
    n_top_labels_co_occurrence: int = 15,
    DPI: int = 200,
) -> None:
    """
    Performs comprehensive exploratory data analysis on a multi-label dataset,
    focusing on label distribution, cardinality, co-occurrence, unique label sets,
    and comparison of different label sources. All generated figures are saved
    to an 'outputs' directory within the dataset's parent directory and are not displayed.

    Args:
        data_path (str): The path to the CSV file containing the dataset.
        label_column (str, optional): The name of the column containing multi-labels.
                                       This column is expected to contain string
                                       representations of Python lists (e.g., "['tag1', 'tag2']").
                                       Defaults to 'multimodal_labels'.
        n_top_labels_plot (int, optional): The number of top most frequent labels
                                           to display in the bar plot. Defaults to 30.
        n_top_labels_co_occurrence (int, optional): The number of top labels
                                                  to include in the co-occurrence heatmap.
                                                  Defaults to 15.
        DPI (int, optional): Dots per inch for saved figures. Higher DPI means better quality images.
                             Defaults to 200.
    Returns:
        None: Prints statistical summaries and saves plots.
    """

    dataset_dir = os.path.dirname(data_path)
    output_dir = os.path.join(dataset_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True) # Ensure outputs directory exists

    # Set display options for pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 100)

    # Set plotting style
    sns.set_style("whitegrid")

    print(f"--- Starting Multi-label EDA for '{data_path}' ---")
    print(f"Focusing on label column: '{label_column}'\n")

    # --- 1. Load Data ---
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at '{data_path}'. Please check the path.")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}\n")
    except Exception as e:
        print(f"Error loading dataset from '{data_path}': {e}")
        return

    # --- 2. Basic Data Information ---
    print("--- Basic DataFrame Info ---")
    df.info()
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- First 5 rows of the DataFrame (After initial load) ---")
    print(df.head())
    print("-" * 40 + "\n")

    # --- 3. Preprocessing Label Columns ---
    label_columns_to_parse = [label_column, 'textual_based_labels', 'visual_based_labels']
    processed_dfs = {} # Store processed dataframes for comparison

    for col in label_columns_to_parse:
        print(f"--- Parsing '{col}' column ---")
        if col not in df.columns:
            print(f"Warning: Label column '{col}' not found in the DataFrame. Skipping.\n")
            continue

        if not df.empty and isinstance(df[col].iloc[0], str):
            try:
                df[col] = df[col].apply(ast.literal_eval)
                print(f"Successfully parsed '{col}' column from string to list.\n")
            except (ValueError, SyntaxError) as e:
                print(f"Error: Could not parse '{col}' column. "
                      f"Ensure it contains valid string representations of lists. Error: {e}")
                continue # For comparison, we can still proceed with other columns even if one fails
        elif not df.empty:
            print(f"'{col}' column is already in list format or not a string. No parsing needed.\n")
        else:
            print(f"DataFrame is empty, cannot proceed with '{col}' label parsing.\n")
            continue

        # Filter out rows with empty label lists for the current column
        initial_len_col = len(df)
        df_filtered_col = df[df[col].apply(len) > 0].copy()
        if len(df_filtered_col) == 0:
            print(f"No samples with valid labels found in '{col}' after parsing/filtering. Skipping further analysis for this column.\n")
            continue
        if initial_len_col != len(df_filtered_col):
            print(f"Removed {initial_len_col - len(df_filtered_col)} rows with empty label lists for column '{col}'.")
            # Note: We are not modifying the original df here,
            # but creating a filtered df for specific column analysis.
        processed_dfs[col] = df_filtered_col # Store the filtered dataframe for this column

    # Use the main label_column's filtered DataFrame for overall stats
    if label_column not in processed_dfs:
        print(f"Main label column '{label_column}' could not be processed or is empty. Exiting EDA.")
        return

    df = processed_dfs[label_column].copy() # Ensure df is the filtered one for the main label column


    # --- 4. Multi-label Statistics (for main label_column) ---
    all_individual_labels = [label for sublist in df[label_column] for label in sublist]
    unique_labels = sorted(list(set(all_individual_labels)))

    print("--- Multi-label Statistics (Main Column: '{label_column}') ---")
    print(f"Total number of samples with valid '{label_column}': {len(df)}")
    print(f"Total number of unique labels across the dataset (from '{label_column}'): {len(unique_labels)}")
    print(f"Example unique labels (first 10): {unique_labels[:10]}")
    print("-" * 40 + "\n")

    # --- 5. Label Cardinality (Number of labels per sample) ---
    df['label_cardinality'] = df[label_column].apply(len)
    print("--- Label Cardinality Statistics (Main Column: '{label_column}') ---")
    print(df['label_cardinality'].describe())
    print("-" * 40 + "\n")

    plt.figure(figsize=(10, 6))
    sns.histplot(df['label_cardinality'], bins=range(1, int(df['label_cardinality'].max()) + 2), kde=False, color='skyblue')
    plt.title(f'Distribution of Label Cardinality (Labels per Sample for "{label_column}")')
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Samples')
    plt.xticks(range(1, int(df['label_cardinality'].max()) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        fname=os.path.join(output_dir, f"{label_column}_label_cardinality_distribution.png"),
        dpi=DPI,
        bbox_inches='tight',
    )
    plt.close() # Close the plot to free memory and prevent display

    # --- 6. Label Frequency (Distribution of each label) ---
    label_counts = Counter(all_individual_labels)
    label_counts_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count']).sort_values(by='Count', ascending=False)

    print("--- Top 20 Most Frequent Labels (Main Column: '{label_column}') ---")
    print(label_counts_df.head(20))
    print("\n--- Bottom 20 Least Frequent Labels (Main Column: '{label_column}') ---")
    print(label_counts_df.tail(20))

    singleton_labels = label_counts_df[label_counts_df['Count'] == 1]
    print(f"\nNumber of labels appearing only once: {len(singleton_labels)}")
    if len(unique_labels) > 0:
        print(f"Percentage of singleton labels: {len(singleton_labels) / len(unique_labels) * 100:.2f}%")
    else:
        print("No unique labels found to calculate percentage of singletons.")
    print(f"Example singleton labels (first 10): {singleton_labels['Label'].head(10).tolist()}")
    print("-" * 40 + "\n")

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Count', y='Label', data=label_counts_df.head(n_top_labels_plot), palette='viridis')
    plt.title(f'Top {n_top_labels_plot} Most Frequent Labels (Main Column: "{label_column}")')
    plt.xlabel('Number of Samples')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig(
        fname=os.path.join(output_dir, f"{label_column}_top_{n_top_labels_plot}_most_frequent_labels.png"),
        dpi=DPI,
        bbox_inches='tight',
    )
    plt.close() # Close the plot

    # New Visualization: Full Distribution of Label Frequencies
    plt.figure(figsize=(10, 6))
    sns.histplot(label_counts_df['Count'], bins=50, kde=False, color='coral')
    plt.title(f'Distribution of All Label Frequencies (Main Column: "{label_column}")')
    plt.xlabel('Label Frequency (Number of Samples)')
    plt.ylabel('Number of Labels (Log Scale)')
    plt.yscale('log') # Use log scale to better visualize the long tail
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(
        fname=os.path.join(output_dir, f"{label_column}_all_label_frequencies_distribution.png"),
        dpi=DPI,
        bbox_inches='tight',
    )
    plt.close() # Close the plot

    # --- 7. Unique Label Combinations ---
    print("--- Unique Label Set Combinations ---")
    # Convert lists to tuples so they are hashable and can be counted
    label_sets = df[label_column].apply(lambda x: tuple(sorted(x)))
    unique_label_sets = Counter(label_sets)
    unique_label_sets_df = pd.DataFrame(unique_label_sets.items(), columns=['Label Set', 'Count']).sort_values(by='Count', ascending=False)

    print(f"Total number of unique label combinations: {len(unique_label_sets)}")
    print(f"Top 10 Most Frequent Label Combinations (from '{label_column}'):")
    print(unique_label_sets_df.head(10))

    if len(unique_label_sets) > 0:
        plt.figure(figsize=(12, 8))
        # Plot top N unique combinations or all if less than N
        top_n_combinations = unique_label_sets_df.head(min(20, len(unique_label_sets)))

        # FIX: Convert tuple Label Sets to readable strings for plotting
        top_n_combinations['Label Set String'] = top_n_combinations['Label Set'].apply(lambda x: ', '.join(x))

        sns.barplot(x='Count', y='Label Set String', data=top_n_combinations, palette='magma')
        plt.title(f'Top {len(top_n_combinations)} Most Frequent Unique Label Combinations')
        plt.xlabel('Number of Samples')
        plt.ylabel('Label Combination')
        plt.tight_layout()
        plt.savefig(
            fname=os.path.join(output_dir, f"{label_column}_top_unique_label_combinations.png"),
            dpi=DPI,
            bbox_inches='tight',
        )
        plt.close() # Close the plot
    print("-" * 40 + "\n")

    # --- 8. Label Correlation Matrix (Jaccard Similarity) ---
    print(f"--- Label Correlation Matrix (Jaccard Similarity) for Top {n_top_labels_co_occurrence} Labels ---")

    if n_top_labels_co_occurrence > len(unique_labels):
        print(f"Warning: n_top_labels_co_occurrence ({n_top_labels_co_occurrence}) is greater than "
              f"the total unique labels ({len(unique_labels)}). Adjusting to total unique labels.")
        n_top_labels_co_occurrence = len(unique_labels)

    if n_top_labels_co_occurrence >= 2: # Correlation makes sense for at least 2 labels
        mlb = MultiLabelBinarizer(classes=unique_labels)
        y_binarized = mlb.fit_transform(df[label_column])
        labels_in_order = mlb.classes_

        # Get indices of top N labels for the subset
        top_labels_for_correlation = label_counts_df['Label'].head(n_top_labels_co_occurrence).tolist()
        top_label_indices = [list(labels_in_order).index(lab) for lab in top_labels_for_correlation]

        # Calculate Jaccard Similarity Matrix
        jaccard_matrix = np.zeros((n_top_labels_co_occurrence, n_top_labels_co_occurrence))
        for i, lab1_idx in enumerate(top_label_indices):
            for j, lab2_idx in enumerate(top_label_indices):
                # Self-similarity (diagonal)
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                else:
                    intersection = np.sum(y_binarized[:, lab1_idx] & y_binarized[:, lab2_idx])
                    union = np.sum(y_binarized[:, lab1_idx] | y_binarized[:, lab2_idx])
                    jaccard_matrix[i, j] = intersection / union if union != 0 else 0.0

        jaccard_df = pd.DataFrame(jaccard_matrix, index=top_labels_for_correlation, columns=top_labels_for_correlation)

        plt.figure(figsize=(15, 12))
        sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap='Blues', linewidths=.5, linecolor='gray',
                    cbar_kws={'label': 'Jaccard Similarity'})
        plt.title(f'Jaccard Similarity Matrix of Top {n_top_labels_co_occurrence} Labels')
        plt.xlabel('Labels')
        plt.ylabel('Labels')
        plt.tight_layout()
        plt.savefig(
            fname=os.path.join(output_dir, f"{label_column}_jaccard_similarity_matrix_top_{n_top_labels_co_occurrence}_labels.png"),
            dpi=DPI,
            bbox_inches='tight',
        )
        plt.close() # Close the plot
    else:
        print("Not enough unique labels to display Jaccard similarity matrix (need at least 2).")
    print("-" * 40 + "\n")


    # --- 9. Comparison of Label Sources ---
    print("--- Comparison of Label Sources: textual_based_labels vs. visual_based_labels vs. multimodal_labels ---")

    source_cols = {
        'textual_based': 'textual_based_labels',
        'visual_based': 'visual_based_labels',
        'multimodal': 'multimodal_labels'
    }

    unique_labels_by_source = {}
    for key, col_name in source_cols.items():
        if col_name in processed_dfs:
            current_all_labels = [label for sublist in processed_dfs[col_name][col_name] for label in sublist]
            unique_labels_by_source[key] = set(current_all_labels)
            print(f"Unique labels in '{col_name}': {len(unique_labels_by_source[key])}")
        else:
            unique_labels_by_source[key] = set()
            print(f"'{col_name}' was not processed or is empty.")

    # Calculate overlaps
    text_set = unique_labels_by_source.get('textual_based', set())
    visual_set = unique_labels_by_source.get('visual_based', set())
    multimodal_set = unique_labels_by_source.get('multimodal', set())

    # Overall overlap
    all_three_overlap = text_set & visual_set & multimodal_set
    text_visual_overlap = text_set & visual_set
    text_multimodal_overlap = text_set & multimodal_set
    visual_multimodal_overlap = visual_set & multimodal_set

    print(f"\nCommon labels across all three sources: {len(all_three_overlap)}")
    print(f"Common labels between Textual and Visual: {len(text_visual_overlap)}")
    print(f"Common labels between Textual and Multimodal: {len(text_multimodal_overlap)}")
    print(f"Common labels between Visual and Multimodal: {len(visual_multimodal_overlap)}")

    # Labels unique to each source (relative to the others)
    unique_to_text = text_set - (visual_set | multimodal_set)
    unique_to_visual = visual_set - (text_set | multimodal_set)
    unique_to_multimodal = multimodal_set - (text_set | visual_set)

    print(f"\nLabels unique to textual_based_labels: {len(unique_to_text)}")
    print(f"Labels unique to visual_based_labels: {len(unique_to_visual)}")
    print(f"Labels unique to multimodal_labels: {len(unique_to_multimodal)}")

    # Visualization: Overlap using a bar chart
    overlap_data = {
        'Category': ['All Three', 'Textual & Visual', 'Textual & Multimodal', 'Visual & Multimodal',
                     'Unique Textual', 'Unique Visual', 'Unique Multimodal'],
        'Count': [len(all_three_overlap), len(text_visual_overlap), len(text_multimodal_overlap),
                  len(visual_multimodal_overlap), len(unique_to_text), len(unique_to_visual), len(unique_to_multimodal)]
    }
    overlap_df = pd.DataFrame(overlap_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Count', y='Category', data=overlap_df, palette='pastel')
    plt.title('Overlap and Uniqueness of Labels from Different Sources')
    plt.xlabel('Number of Unique Labels')
    plt.ylabel('Label Set')
    plt.tight_layout()
    plt.savefig(
        fname=os.path.join(output_dir, "label_source_overlap_uniqueness.png"),
        dpi=DPI,
        bbox_inches='tight',
    )
    plt.close() # Close the plot
    print("-" * 40 + "\n")


    # --- 10. Temporal Analysis (doc_date) ---
    print("--- Temporal Analysis (doc_date) ---")
    # Ensure original df is used for temporal analysis before any filtering related to labels
    original_df = pd.read_csv(data_path) # Reload to get original state for date columns

    if 'doc_date' in original_df.columns or 'raw_doc_date' in original_df.columns:
        original_df['parsed_date'] = pd.NaT # Initialize as NaT (Not a Time)

        # Prefer 'raw_doc_date' if available and looks like a full date string
        if 'raw_doc_date' in original_df.columns:
            # Attempt to parse as full date, coerce errors
            original_df['parsed_date'] = pd.to_datetime(original_df['raw_doc_date'], errors='coerce')

        # If 'raw_doc_date' failed or wasn't present, try 'doc_date' as year float
        # Only try if 'parsed_date' is still NaT or if doc_date is better
        if original_df['parsed_date'].isnull().all() and 'doc_date' in original_df.columns:
             # Convert float years (e.g., 1903.0) to int, then to string for '%Y' format
            original_df['parsed_date'] = original_df['doc_date'].apply(
                lambda x: pd.to_datetime(str(int(x)), format='%Y', errors='coerce') if pd.notnull(x) else pd.NaT
            )

        df_valid_dates = original_df.dropna(subset=['parsed_date']).copy()

        # Re-parse labels for temporal analysis if needed, ensuring they're lists
        for col in label_columns_to_parse:
            if col in df_valid_dates.columns and isinstance(df_valid_dates[col].iloc[0], str):
                df_valid_dates[col] = df_valid_dates[col].apply(ast.literal_eval)

        if not df_valid_dates.empty:
            df_valid_dates['year'] = df_valid_dates['parsed_date'].dt.year
            print(f"Date range: {df_valid_dates['year'].min()} - {df_valid_dates['year'].max()}")

            plt.figure(figsize=(12, 6))
            sns.histplot(df_valid_dates['year'], bins=range(int(df_valid_dates['year'].min()), int(df_valid_dates['year'].max()) + 2), kde=False, color='lightgreen')
            plt.title('Distribution of Documents by Year')
            plt.xlabel('Year')
            plt.ylabel('Number of Documents')
            plt.tight_layout()
            plt.savefig(
                fname=os.path.join(output_dir, "document_year_distribution.png"),
                dpi=DPI,
                bbox_inches='tight',
            )
            plt.close() # Close the plot

            # Optional: Label frequency over time (e.g., top N labels)
            # This is more complex and might require resampling or pivoting,
            # but a simple visual can be average cardinality over time.
            if label_column in df_valid_dates.columns:
                df_valid_dates['label_cardinality'] = df_valid_dates[label_column].apply(len)
                avg_cardinality_by_year = df_valid_dates.groupby('year')['label_cardinality'].mean().reset_index()

                if not avg_cardinality_by_year.empty:
                    plt.figure(figsize=(12, 6))
                    sns.lineplot(x='year', y='label_cardinality', data=avg_cardinality_by_year, marker='o', color='purple')
                    plt.title(f'Average Label Cardinality by Year (for "{label_column}")')
                    plt.xlabel('Year')
                    plt.ylabel('Average Number of Labels per Document')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(
                        fname=os.path.join(output_dir, f"{label_column}_avg_cardinality_by_year.png"),
                        dpi=DPI,
                        bbox_inches='tight',
                    )
                    plt.close() # Close the plot
            else:
                 print(f"'{label_column}' column not available in date-filtered data for cardinality over time.")
        else:
            print("No valid dates found for temporal analysis after parsing.")
    else:
        print("Neither 'doc_date' nor 'raw_doc_date' column found. Skipping temporal analysis.")
    print("-" * 40 + "\n")

    print("--- EDA Complete ---")

def plot_label_distribution_pie_chart(
		df: pd.DataFrame = None,
		fpth: str = "label_distribution_pie_chart.png",
		figure_size: tuple = (12, 7),
		DPI: int = 200,
		dataset_name: str = "EUROPEANA_1900-01-01_1970-12-31",
	):
	# Count labels and sort by count (descending)
	label_counts = df['label'].value_counts().sort_values(ascending=False)
	labels = label_counts.index
	total_samples = label_counts.sum()
	unique_labels = len(labels)
	# Group small categories into "Other"
	threshold = 0.01  # 1% threshold
	other_count = label_counts[label_counts / total_samples < threshold].sum()
	main_counts = label_counts[label_counts / total_samples >= threshold]
	if other_count > 0:
		main_counts['Other'] = other_count
	labels = main_counts.index
	label_counts = main_counts
	# Create figure with vertical layout
	fig = plt.figure(figsize=figure_size)
	gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1])
	ax_pie = fig.add_subplot(gs[0])
	ax_legend = fig.add_subplot(gs[1])
	# Use a colorblind-friendly categorical colormap
	colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
	# Explode larger wedges
	explode = [0.1 if i < 3 else 0 for i in range(len(labels))]
	# Create pie chart
	wedges, texts, autotexts = ax_pie.pie(
			label_counts.values,
			labels=[''] * len(labels),
			colors=colors,
			autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
			startangle=0,
			explode=explode,
			wedgeprops={
					'edgecolor': 'black',
					'linewidth': 0.7,
					'alpha': 0.8,
			}
	)
	# Adjust percentage label contrast and position
	for i, autotext in enumerate(autotexts):
			if autotext.get_text():
					wedge_color = wedges[i].get_facecolor()
					luminance = 0.299 * wedge_color[0] + 0.587 * wedge_color[1] + 0.114 * wedge_color[2]
					autotext.set_color('white' if luminance < 0.5 else 'black')
					if label_counts.values[i] / total_samples < 0.1:
							autotext.set_position((autotext.get_position()[0] * 1.2, autotext.get_position()[1] * 1.2))
					autotext.set_fontsize(18)
					autotext.set_weight('bold')  # Make font bold

	# Turn off axis for legend subplot
	ax_legend.axis('off')
	# Create truncated legend
	if len(labels) > 6:
		selected_wedges = wedges[:3] + [None] + wedges[-3:]
		legend_labels_full = [
			f"{label} ({count:,}, {count/total_samples*100:.1f}%)"
			for label, count in label_counts.items()
		]
		omitted_count = len(labels) - 6
		# selected_labels = legend_labels_full[:3] + [f'... ({omitted_count} categories omitted)'] + legend_labels_full[-3:]
		selected_labels = legend_labels_full[:3] + [f'...'] + legend_labels_full[-3:]
		dummy_artist = plt.Rectangle((0, 0), 1, 1, fc='none', fill=False, edgecolor='none', linewidth=0)
		selected_wedges[3] = dummy_artist
	else:
		selected_wedges = wedges
		selected_labels = [
			f"{label} ({count:,}, {count/total_samples*100:.1f}%)"
			for label, count in label_counts.items()
		]
	# Create legend
	legend = ax_legend.legend(
		selected_wedges,
		selected_labels,
		loc='center',
		bbox_to_anchor=(0.5, 0.5),
		fontsize=16,
		title=f"Total samples: {total_samples:,} (Unique Labels: {unique_labels})",
		title_fontsize=15,
		fancybox=True,
		shadow=True,
		edgecolor='black',
		facecolor='white',
		ncol=1,
		labelspacing=1.2,
		labelcolor='black',
	)	
	for text in legend.get_texts():
		text.set_fontweight('bold')
	ax_pie.axis('equal')
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

	# Optional bar chart for top 10 categories
	plt.figure(figsize=(15, 5))
	top_n = 10
	top_counts = label_counts[:top_n]
	if len(label_counts) > top_n:
			top_counts['Other'] = label_counts[top_n:].sum()
	colors = plt.cm.tab20c(np.linspace(0, 1, len(top_counts)))
	plt.bar(top_counts.index, top_counts.values, color=colors)
	plt.yscale('log')  # Log scale for visibility of small categories
	plt.xticks(rotation=45, ha='right')
	plt.ylabel('Sample Count (Log Scale)')
	plt.title(f"Top {top_n} Label Distribution for {dataset_name} Dataset")
	plt.tight_layout()
	plt.savefig(
		fname=fpth.replace('.png', '_bar.png'), 
		dpi=DPI, 
		bbox_inches='tight',
	)
	plt.close()

def plot_grouped_bar_chart(
		merged_df: pd.DataFrame,
		DPI: int = 200,
		FIGURE_SIZE: tuple = (12, 8),
		fname: str = "grouped_bar_chart.png",
	):
	
	calling_frame = inspect.currentframe().f_back
	dfs_length = len(calling_frame.f_locals.get('dfs', []))

	dataset_unique_label_counts = merged_df.groupby('dataset')['label'].nunique()
	print(dataset_unique_label_counts)

	label_counts = merged_df['label'].value_counts()
	# print(label_counts.tail(25))

	plt.figure(figsize=FIGURE_SIZE)
	sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
	ax = plt.gca()
	handles, labels = ax.get_legend_handles_labels()
	new_labels = [f"{label} | ({dataset_unique_label_counts[label]})" for label in labels]
	ax.legend(handles, new_labels, loc="best", fontsize=10, title="Dataset | (Unique Label Count)")
	plt.title(f'Grouped Bar Chart for total of {label_counts.shape[0]} Labels Frequency for {dfs_length} Datasets')
	plt.xticks(fontsize=9, rotation=90, ha='right')
	plt.yticks(fontsize=9, rotation=90, va='center')
	plt.xlabel('Label')
	plt.ylabel('Frequency')
	plt.grid(axis='y', alpha=0.7, linestyle='--')
	plt.tight_layout()
	plt.savefig(
		fname=fname,
		dpi=DPI,
		bbox_inches='tight'
	)
	plt.close()

def plot_train_val_label_distribution(
		train_df: pd.DataFrame,
		val_df: pd.DataFrame,
		dataset_name: str,
		OUTPUT_DIRECTORY: str,
		DPI: int = 200,
		FIGURE_SIZE: tuple = (12, 8),
		VAL_SPLIT_PCT: float = 0.2,
		fname: str = "simple_random_split_stratified_label_distribution_train_val.png",
	):
	# Visualize label distribution in training and validation sets
	plt.figure(figsize=FIGURE_SIZE)
	train_df['label'].value_counts().plot(kind='bar', color='blue', alpha=0.6, label=f'Train {1-VAL_SPLIT_PCT}')
	val_df['label'].value_counts().plot(kind='bar', color='red', alpha=0.9, label=f'Validation {VAL_SPLIT_PCT}')
	plt.title(
		f'{dataset_name} Stratified Label Distribution (Total samples: {train_df.shape[0]+val_df.shape[0]})\n'
		f'Train: {train_df.shape[0]} | Validation: {val_df.shape[0]}', 
		fontsize=9, 
		fontweight='bold',
	)
	plt.xlabel('Label')
	plt.ylabel('Frequency')
	plt.yticks(rotation=90, fontsize=9, va='center')
	plt.legend(
		loc='best', 
		ncol=2, 
		frameon=True,
		fancybox=True,
		shadow=True,
		edgecolor='black',
		facecolor='white', 
		fontsize=10,
	)
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.tight_layout()
	plt.savefig(
		fname=fname,
		dpi=DPI,
		bbox_inches='tight'
	)	
	plt.close()

def plot_year_distribution(
		df: pd.DataFrame,
		dname: str,
		fpth: str,
		BINs: int = 50,
		FIGURE_SIZE: tuple = (18, 8),
		DPI: int = 200,
	):
	# matplotlib.rcParams['font.family'] = ['Source Han Sans TW', 'sans-serif']
	# print(natsorted(matplotlib.font_manager.get_font_names()))
	# plt.rcParams["font.family"] = "DejaVu Math TeX Gyre"

	# Convert 'doc_date' to datetime and handle invalid entries
	df['doc_date'] = pd.to_datetime(df['doc_date'], errors='coerce')
	
	# Extract valid dates (non-NaN)
	valid_dates = df['doc_date'].dropna()
	
	# Handle edge case: no valid dates
	if valid_dates.empty:
		plt.figure(figsize=FIGURE_SIZE)
		plt.text(0.5, 0.5, "No valid dates available for plotting", ha='center', va='center', fontsize=12)
		plt.title(f'{dname} Temporal Distribution - No Data')
		plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
		plt.close()
		return
	
	# Compute start and end dates from data
	start_date = valid_dates.min().strftime('%Y-%m-%d')
	end_date = valid_dates.max().strftime('%Y-%m-%d')
	start_year = valid_dates.min().year
	end_year = valid_dates.max().year
	print(f"start_year: {start_year} | end_year: {end_year}")
	
	# Extract the year from the 'doc_date' column (now as integer)
	df['year'] = df['doc_date'].dt.year  # This will have NaN where doc_date is NaT
	
	# Filter out None values (though dt.year gives NaN, which .dropna() handles)
	year_series = df['year'].dropna().astype(int)
	# Find the years with the highest and lowest frequencies (handle ties)
	year_counts = year_series.value_counts()
	max_hist_freq = max(year_counts.values)
	min_hist_freq = min(year_counts.values)
	print(f"max_hist_freq: {max_hist_freq} | min_hist_freq: {min_hist_freq}")
	
	# Get the years with the maximum and minimum frequencies
	max_freq = year_counts.max()
	min_freq = year_counts.min()
	max_freq_years = year_counts[year_counts == max_freq].index.tolist()
	min_freq_years = year_counts[year_counts == min_freq].index.tolist()
	
	# Calculate mean, median, and standard deviation
	mean_year = year_series.mean()
	median_year = year_series.median()
	std_year = year_series.std()
	# Calculate 95% confidence interval for the mean
	confidence_level = 0.95
	n = len(year_series)
	mean_conf_interval = stats.t.interval(confidence_level, df=n-1, loc=mean_year, scale=stats.sem(year_series))
	# Get the overall shape of the distribution
	distribution_skew = year_series.skew()
	distribution_kurtosis = year_series.kurtosis()
	skew_desc = "right-skewed" if distribution_skew > 0 else "left-skewed" if distribution_skew < 0 else "symmetric"
	kurt_desc = "heavy-tailed" if distribution_kurtosis > 0 else "light-tailed" if distribution_kurtosis < 0 else "normal-tailed"
	# Calculate percentiles
	q25, q75 = year_series.quantile([0.25, 0.75])
	# Plot KDE using scipy.stats.gaussian_kde
	plt.figure(figsize=FIGURE_SIZE)
	sns.histplot(
		year_series,
		bins=BINs,
		color="#5c6cf8",
		kde=True,
		edgecolor="white",
		alpha=0.95,
		linewidth=1.5,
		label="Temporal Distribution Histogram"
	)
	# Create the KDE object and adjust bandwidth to match Seaborn's default behavior
	kde = gaussian_kde(year_series, bw_method='scott')  # Use 'scott' or 'silverman', or a custom value
	x_range = np.linspace(start_year, end_year, 300)
	kde_values = kde(x_range)
	bin_width = (end_year - start_year) / BINs  # Approximate bin width of the histogram
	kde_scaled = kde_values * len(year_series) * bin_width  # Scale KDE to match frequency
	plt.plot(
		x_range,
		kde_scaled,
		color="#3a3a3a",
		linewidth=2.0,
		linestyle="-",
		label="Kernel Density Estimate (KDE)",
	)
	world_war_1 = [1914, 1918]
	world_war_2 = [1939, 1945]
	ww_cols = ['#fa3627', '#24f81d']
	padding = 1.05
	max_padding = 1.1
	# Add shaded regions for WWI and WWII (plot these first to ensure they are in the background)
	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		plt.axvspan(world_war_1[0], world_war_1[1], color=ww_cols[0], alpha=0.2, label='World War One')

	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		plt.axvspan(world_war_2[0], world_war_2[1], color=ww_cols[1], alpha=0.2, label='World War Two')

	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		for year in world_war_1:
			plt.axvline(x=year, color='r', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_1[0] + world_war_1[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWI',
			color=ww_cols[0],
			fontsize=12,
			fontweight="bold",
			ha="center",  # horizontal alignment
		)
	
	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		for year in world_war_2:
			plt.axvline(x=year, color=ww_cols[1], linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_2[0] + world_war_2[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWII',
			color=ww_cols[1],
			fontsize=12,
			fontweight="bold",
			ha="center", # horizontal alignment
		)

	# Add visual representations of key statistics
	plt.axvline(x=mean_year, color='#ee8206ee', linestyle='-.', lw=2.5, label=f'Mean Year: {mean_year:.1f}')
	plt.axvspan(mean_year - std_year, mean_year + std_year, color='#fdff7c', alpha=0.15, label='Mean ± 1 SD')

	valid_count = len(year_series)
	stats_text = (
		# f"Samples with valid dates: {valid_count} (~{round(valid_count / df.shape[0] * 100)}%)\n\n"
		"Frequency Statistics:\n"
		f"  Most frequent year(s): {', '.join(map(str, max_freq_years))} ({max_freq} images)\n"
		f"  Least frequent year(s): {', '.join(map(str, min_freq_years))} ({min_freq} images)\n\n"
		"Central Tendency [Year]:\n"
		f"  Median: {median_year:.0f}\n"
		f"  Mean: {mean_year:.1f}\n"
		f"     Confidence Interval (95%): [{mean_conf_interval[0]:.1f}, {mean_conf_interval[1]:.1f}]\n"
		f"  Standard deviation: {std_year:.1f}\n\n"
		"Percentiles:\n"
		f"  25th: {q25:.2f}\n"
		f"  75th: {q75:.2f}\n\n"
		"Distribution Shape:\n"
		f"  Skewness: {distribution_skew:.2f} ({skew_desc})\n"
		f"  Kurtosis: {distribution_kurtosis:.2f} ({kurt_desc})"
	)
	plt.text(
		0.01,
		0.98,
		stats_text,
		transform=plt.gca().transAxes,
		fontsize=10,
		verticalalignment='top',
		horizontalalignment='left',
		color='black',
		bbox=dict(
			boxstyle='round,pad=0.5',
			facecolor='white',
			alpha=0.9,
			edgecolor='none', 
			linewidth=0.0,
		)
	)
	plt.title(
		label=f'Temporal Distribution ({start_date} - {end_date}) Total Samples: {df.shape[0]}', fontsize=12, fontweight='bold')
	plt.xlabel('')
	plt.tick_params(axis='x', length=0, width=0, color='black', labelcolor='black', labelsize=15)
	plt.ylabel('Frequency', fontsize=15, fontweight='bold')
	plt.ylim(0, max_freq * max_padding)  # Add some padding to the y-axis
	plt.yticks(fontsize=15, rotation=90, va='center')

	plt.xlim(start_year - 2, end_year + 2)
	plt.legend(
		loc='upper left',
		bbox_to_anchor=(0.01, 0.56),
		fontsize=10,
		frameon=False,
	)
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

def create_distribution_plot_with_long_tail_analysis(
		df: pd.DataFrame,
		fpth: str,
		FIGURE_SIZE: tuple = (14, 9),
		DPI: int = 200,
		top_n: int = None,  # Option to show only top N labels
		head_threshold: int = 5000,  # Labels with frequency > head_threshold
		tail_threshold: int = 1000,   # Labels with frequency < tail_threshold
	):
	label_counts = df['label'].value_counts().sort_values(ascending=False)
	
	# Handle large number of labels
	if top_n and len(label_counts) > top_n:
			top_labels = label_counts.head(top_n)
			other_count = label_counts[top_n:].sum()
			top_labels = pd.concat([top_labels, pd.Series([other_count], index=['Other'])])
			label_counts = top_labels
	
	# Identify Head, Torso, and Tail segments
	head_labels = label_counts[label_counts > head_threshold].index.tolist()
	tail_labels = label_counts[label_counts < tail_threshold].index.tolist()
	torso_labels = label_counts[(label_counts >= tail_threshold) & (label_counts <= head_threshold)].index.tolist()
	segment_colors = {
		'Head': '#00a87ee5',
		'Torso': '#b99700',
		'Tail': '#eb3e3e',
	}
	# Create figure and primary axis
	fig, ax = plt.subplots(
		figsize=FIGURE_SIZE, 
		facecolor='white', 
	)
	
	# Plot with better styling
	bars = label_counts.plot(
		kind='bar',
		ax=ax,
		color="#00315393",
		width=0.7,
		edgecolor='white',
		linewidth=0.8,
		alpha=0.65,
		label='Linear Scale'.capitalize(),
		zorder=2,
	)
	
	# Create shaded regions for Head, Torso, and Tail
	all_indices = np.arange(len(label_counts))
	head_indices = [i for i, label in enumerate(label_counts.index) if label in head_labels]
	torso_indices = [i for i, label in enumerate(label_counts.index) if label in torso_labels]
	tail_indices = [i for i, label in enumerate(label_counts.index) if label in tail_labels]
	
	# Sort the indices to ensure proper shading
	head_indices.sort()
	torso_indices.sort()
	tail_indices.sort()
	
	# Add shaded areas if segments exist
	ymax = label_counts.max() * 1.1  # Set maximum y-value for shading
	
	segment_opacity = 0.2
	segment_text_yoffset = 1.045 if len(head_labels) < 5 else 1.0
	segment_text_opacity = 0.7
	if head_indices:
		ax.axvspan(
			min(head_indices) - 0.4, 
			max(head_indices) + 0.4, 
			alpha=segment_opacity, 
			color=segment_colors['Head'], 
			# label='Head'.upper()
		)
		ax.text(
			np.mean(head_indices), 
			ymax * segment_text_yoffset,
			f"HEAD\n({len(head_labels)} labels)",
			horizontalalignment='center',
			verticalalignment='center',
			fontsize=12,
			fontweight='bold',
			color=segment_colors['Head'],
			bbox=dict(facecolor='white', alpha=segment_text_opacity, edgecolor='none', pad=2),
			zorder=5,
		)
	
	if torso_indices:
		ax.axvspan(
			min(torso_indices) - 0.4, 
			max(torso_indices) + 0.4, 
			alpha=segment_opacity, 
			color=segment_colors['Torso'], 
			# label='Torso'.upper(),
		)
		ax.text(
			np.mean(torso_indices), 
			ymax * segment_text_yoffset, 
			f"TORSO\n({len(torso_labels)} labels)",
			horizontalalignment='center',
			verticalalignment='center',
			fontsize=12,
			fontweight='bold',
			color=segment_colors['Torso'],
			bbox=dict(facecolor='white', alpha=segment_text_opacity, edgecolor='none', pad=2),
			zorder=5,
		)
	
	if tail_indices:
			ax.axvspan(
				min(tail_indices) - 0.4, 
				max(tail_indices) + 0.4, 
				alpha=segment_opacity, 
				color=segment_colors['Tail'], 
				# label='Tail'.upper()
			)
			ax.text(
				np.mean(tail_indices), 
				ymax * segment_text_yoffset,
				f"TAIL\n({len(tail_labels)} labels)",
				horizontalalignment='center',
				verticalalignment='center',
				fontsize=12,
				fontweight='bold',
				color=segment_colors['Tail'],
				bbox=dict(
					facecolor='white', 
					alpha=0.5, 
					edgecolor='none', 
					pad=2,
				),
				zorder=5,
			)
	
	# Hide all spines initially
	for spine in ax.spines.values():
		spine.set_visible(False)

	# Show only the left, right and bottom spines
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(True)

	# Enhance readability for large number of labels
	if len(label_counts) > 20:
		plt.xticks(rotation=90, fontsize=11)
	else:
		plt.xticks(rotation=45, fontsize=9, ha='right')
	plt.yticks(fontsize=9, rotation=90, va='center')
	
	# Add value labels on top of bars
	for i, v in enumerate(label_counts):
		text_color = segment_colors['Head'] if i in head_indices else (segment_colors['Torso'] if i in torso_indices else segment_colors['Tail'])
		ax.text(
			i, 
			v + (v * 0.04),  # Adjust vertical position relative to bar height
			str(v), 
			ha='center',
			fontsize=8,
			fontweight='bold',
			alpha=0.8,
			color=text_color,
			rotation=75,
			bbox=dict(
				facecolor='white',
				edgecolor='none',
				alpha=0.78,
				pad=0.5
			),
			zorder=5,
		)
	
	# Add a logarithmic scale option for highly imbalanced distributions
	if label_counts.max() / label_counts.min() > 50:
		ax_log = ax.twinx()
		ax_log.set_yscale('log')
		label_counts.plot(
			kind='line',
			ax=ax_log,
			color='#8a008a',
			marker='o',
			markerfacecolor='none',  # Remove marker fill
			markeredgecolor='#8a008a',   # Set marker edge color
			markersize=3,           # Optional: adjust marker size
			linewidth=2.5,
			alpha=0.9,
			label='Logarithmic Scale'.capitalize(),
			zorder=3,
		)
		ax_log.set_ylabel(
				ylabel='Log Sample Frequency', 
				color='#8a008a', 
				fontsize=10, 
				fontweight='bold',
		)
		ax_log.tick_params(axis='y', colors='#8a008a')
		ax_log.spines['right'].set_visible(True)
		ax_log.spines['right'].set_color('#8a008a')
		ax_log.spines['right'].set_linewidth(1.0)
		ax_log.spines['right'].set_alpha(0.7)
		ax_log.grid(axis='y', alpha=0.3, linestyle='--', color='#727272', zorder=0)

		# Hide all spines for the logarithmic scale
		for spine in ax_log.spines.values():
			spine.set_visible(False)
	ax.set_xlabel('')
	ax.tick_params(axis='x', length=0, width=0, color='none', labelcolor='black', labelsize=12)
	ax.tick_params(axis='y', color='black', labelcolor='black', labelsize=11)
	ax.set_ylabel('Sample Frequency', fontsize=10, fontweight='bold')
	# ax.set_ylim(0, label_counts.max() * 1.1)
	
	# Add basic statistics for the distribution
	imbalance_ratio = label_counts.max()/label_counts.min()
	median_label_size = label_counts.median()
	mean_label_size = label_counts.mean()
	std_label_size = label_counts.std()
	most_freq_label = label_counts.max()/df.shape[0]*100
	least_freq_label = label_counts.min()/df.shape[0]*100
	
	# Add segment statistics
	head_count = sum(label_counts[head_labels])
	torso_count = sum(label_counts[torso_labels])
	tail_count = sum(label_counts[tail_labels])
	total_samples = df.shape[0]
	
	head_percent = head_count/total_samples*100 if total_samples > 0 else 0
	torso_percent = torso_count/total_samples*100 if total_samples > 0 else 0
	tail_percent = tail_count/total_samples*100 if total_samples > 0 else 0
	
	stats_text = (
			f"Imbalance ratio: {imbalance_ratio:.1f}\n\n"
			f"Label Statistics:\n"
			f"    Median: {median_label_size:.0f}\n"
			f"    Mean: {mean_label_size:.1f}\n"
			f"    Standard deviation: {std_label_size:.1f}\n"
			f"    Most frequent: {most_freq_label:.1f}%\n"
			f"    Least frequent: {least_freq_label:.2f}%\n\n"
			f"Segment Statistics:\n"
			f"    Head: {len(head_labels)} labels, {head_count} samples ({head_percent:.1f}%)\n"
			f"    Torso: {len(torso_labels)} labels, {torso_count} samples ({torso_percent:.1f}%)\n"
			f"    Tail: {len(tail_labels)} labels, {tail_count} samples ({tail_percent:.1f}%)"
	)
	print(f"stats_text:\n{stats_text}\n")

	# Create custom legend elements
	custom_lines = [
		Line2D([0], [0], color="#00315393", lw=4),  # Linear scale
		Line2D([0], [0], color='#8a008a', lw=2, marker='o', markersize=3, markerfacecolor='none', markeredgecolor='#8a008a')  # Logarithmic scale
	]

	custom_labels = ['Linear', 'Logarithmic']

	# Create the legend with just these elements
	legend = ax.legend(
			custom_lines, 
			custom_labels, 
			loc="best",
			title='Label Distribution',
			title_fontsize=14,
			fontsize=12,
			ncol=1,
			frameon=True,
			fancybox=True,
			shadow=True,
			edgecolor='black',
			facecolor='white',
	)
	legend.set_zorder(100)

	plt.tight_layout()
	plt.savefig(fpth, dpi=DPI, bbox_inches='tight')
	plt.close()
	
	return {
		'head_labels': head_labels,
		'torso_labels': torso_labels,
		'tail_labels': tail_labels,
		'head_count': head_count,
		'torso_count': torso_count,
		'tail_count': tail_count,
	}

def plot_label_distribution(
		df: pd.DataFrame,
		dname: str,
		fpth: str,
		FIGURE_SIZE: tuple = (14, 8),
		DPI: int = 200,
		top_n: int = None  # Option to show only top N labels
	):

	label_counts = df['label'].value_counts()
	
	# Handle large number of labels
	if top_n and len(label_counts) > top_n:
		top_labels = label_counts.head(top_n)
		other_count = label_counts[top_n:].sum()
		top_labels = pd.concat([top_labels, pd.Series([other_count], index=['Other'])])
		label_counts = top_labels
	
	fig, ax = plt.subplots(
		figsize=FIGURE_SIZE, 
		facecolor='white', 
		# constrained_layout=True,
	)
	
	# Plot with better styling
	bars = label_counts.plot(
		kind='bar',
		ax=ax,
		color="green",
		width=0.8,
		edgecolor='white',
		linewidth=0.8,
		alpha=0.8,
		label='Linear Scale'.capitalize()
	)

	# Hide all spines initially
	for spine in ax.spines.values():
		spine.set_visible(False)

	# Show only the left, right and bottom spines
	ax.spines['bottom'].set_visible(True)
	ax.spines['left'].set_visible(True)
	ax.spines['right'].set_visible(True)

	# Enhance readability for large number of labels
	if len(label_counts) > 20:
		plt.xticks(rotation=90, fontsize=11)
	else:
		plt.xticks(rotation=45, fontsize=9, ha='right')
	plt.yticks(fontsize=9, rotation=90, va='center')
	
	# Add value labels on top of bars
	for i, v in enumerate(label_counts):
		ax.text(
			i, 
			v + (v * 0.05),  # Adjust vertical position relative to bar height
			str(v), 
			ha='center',
			fontsize=8,
			fontweight='bold',
			alpha=0.8,
			color='blue',
			rotation=75,
			bbox=dict(
				facecolor='white',
				edgecolor='none',
				alpha=0.7,
				pad=0.5
			)
		)
	
	# Add a logarithmic scale option for highly imbalanced distributions
	ax_log = None
	if label_counts.max() / label_counts.min() > 50:
		ax_log = ax.twinx()
		ax_log.set_yscale('log')
		label_counts.plot(
			kind='line',
			ax=ax_log,
			color='red',
			marker='o',
			markerfacecolor='none',  # Remove marker fill
			markeredgecolor='red',   # Set marker edge color
			markersize=3,           # Optional: adjust marker size
			linewidth=2.5,
			alpha=0.9,
			label='Logarithmic'
		)
		ax_log.set_ylabel(
			ylabel='Log Sample Frequency', 
			color='red', 
			fontsize=10, 
			fontweight='bold',
		)
		ax_log.tick_params(axis='y', colors='red')
	
	# Hide all spines for the logarithmic scale
	if ax_log is not None:
		for spine in ax_log.spines.values():
			spine.set_visible(False)

	ax.set_xlabel('')
	ax.tick_params(axis='x', length=0, width=0, color='none', labelcolor='black', labelsize=12)
	ax.tick_params(axis='y', color='black', labelcolor='black', labelsize=11)
	ax.set_ylabel('Sample Frequency', fontsize=10, fontweight='bold')
	
	# Add basic statistics for the distribution
	imbalaned_ratio = label_counts.max()/label_counts.min()
	median_label_size = label_counts.median()
	mean_label_size = label_counts.mean()
	std_label_size = label_counts.std()
	most_freq_label = label_counts.max()/df.shape[0]*100
	least_freq_label = label_counts.min()/df.shape[0]*100
	stats_text = (
		f"Imbalance ratio: {imbalaned_ratio:.1f}\n\n"
		f"Label Statistics:\n"
		f"    Median: {median_label_size:.0f}\n"
		f"    Mean: {mean_label_size:.1f}\n"
		f"    Standard deviation: {std_label_size:.1f}\n"
		f"    Most frequent: {most_freq_label:.1f}%\n"
		f"    Least frequent: {least_freq_label:.2f}%"
	)
	print(f"stats_text:\n{stats_text}\n")
	plt.text(
		0.74, # horizontal position
		0.86, # vertical position
		stats_text,
		transform=ax.transAxes,
		fontsize=15,
		verticalalignment='top',
		horizontalalignment='left',
		color='black',
		bbox=dict(
			boxstyle='round,pad=0.5',
			facecolor='white',
			alpha=0.8,
			edgecolor='none', 
			linewidth=0.0,
		)
	)

	# Enhanced title and labels
	plt.title(
		f'Label Distribution (Total samples: {df.shape[0]} Unique Labels: {len(df["label"].unique())})', 
		fontsize=15,
		fontweight='bold',
	)
	# Create a single legend
	h1, l1 = ax.get_legend_handles_labels()
	# h2, l2 = ax_log.get_legend_handles_labels()
	h2, l2 = ([], []) if ax_log is None else ax_log.get_legend_handles_labels()
	ax.legend(
		h1 + h2, 
		l1 + l2, 
		# loc='best', 
		loc='upper left',  # Changed to upper left
		bbox_to_anchor=(0.73, 0.99),  # Match horizontal position with text (0.74)
		title='Label Distribution (Scale)',
		title_fontsize=12,
		fontsize=11, 
		ncol=2,
		frameon=False, 
		# fancybox=True, 
		# shadow=True, 
		# edgecolor='black', 
		# facecolor='white'
	)

	plt.grid(axis='y', alpha=0.7, linestyle='--')
	plt.tight_layout()
	plt.savefig(fpth, dpi=DPI, bbox_inches='tight')
	plt.close()