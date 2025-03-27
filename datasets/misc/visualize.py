import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import gaussian_kde, t
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import inspect

def plot_grouped_bar_chart(
		merged_df: pd.DataFrame,
		dataset_name: str,
		OUTPUT_DIRECTORY: str,
		DPI: int = 250,
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
		DPI: int = 250,
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
		FIGURE_SIZE: tuple = (18, 9),
		DPI: int = 250,
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
		color="skyblue",
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
		color="grey", # dark gray
		linewidth=2.0,
		linestyle="-",
		label="Kernel Density Estimate (KDE)",
	)
	world_war_1 = [1914, 1918]
	world_war_2 = [1939, 1945]
	padding = 1.25
	max_padding = 1.3
	# Add shaded regions for WWI and WWII (plot these first to ensure they are in the background)
	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		plt.axvspan(world_war_1[0], world_war_1[1], color='#ff3a2d', alpha=0.2, label='World War One')

	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		plt.axvspan(world_war_2[0], world_war_2[1], color='#9aff33', alpha=0.2, label='World War Two')

	if start_year <= world_war_1[0] and world_war_1[1] <= end_year:
		for year in world_war_1:
			plt.axvline(x=year, color='r', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_1[0] + world_war_1[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWI',
			color='red',
			fontsize=12,
			fontweight="bold",
			ha="center",  # horizontal alignment
		)
	
	if start_year <= world_war_2[0] and world_war_2[1] <= end_year:
		for year in world_war_2:
			plt.axvline(x=year, color='g', linestyle='--', lw=2.5)
		plt.text(
			x=(world_war_2[0] + world_war_2[1]) / 2,  # float division for precise centering
			y=max_freq * padding,
			s='WWII',
			color='green',
			fontsize=12,
			fontweight="bold",
			ha="center", # horizontal alignment
		)

	# Add visual representations of key statistics
	plt.axvline(x=mean_year, color='navy', linestyle='-.', lw=1.5, label=f'Mean Year: {mean_year:.2f}')
	plt.axvspan(mean_year - std_year, mean_year + std_year, color='yellow', alpha=0.16, label='Mean ± 1 SD')

	valid_count = len(year_series)
	stats_text = (
			f"Samples with valid dates: {valid_count} (~{round(valid_count / df.shape[0] * 100)}%)\n\n"
			"Frequency Statistics:\n"
			f"  Most frequent year(s): {', '.join(map(str, max_freq_years))} ({max_freq} images)\n"
			f"  Least frequent year(s): {', '.join(map(str, min_freq_years))} ({min_freq} images)\n\n"
			"Central Tendency [Year]:\n"
			f"  Mean: {mean_year:.2f}\n"
			f"  Mean 95% CI: [{mean_conf_interval[0]:.2f}, {mean_conf_interval[1]:.2f}]\n"
			f"  Median: {median_year:.2f}\n"
			f"  Standard deviation: {std_year:.2f}\n\n"
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
		ha='left',
		va='top',
		fontsize=10.0,
		color='black',
		bbox=dict(boxstyle='round,pad=0.5',facecolor='white', alpha=0.8, edgecolor='gray')
	)
	plt.title(f'{dname} Temporal Distribution ({start_date} - {end_date}) Total Samples: {df.shape[0]}', fontsize=10, fontweight='bold')
	plt.xlabel('Year')
	plt.ylabel('Frequency')
	plt.ylim(0, max_freq * max_padding)  # Add some padding to the y-axis
	plt.yticks(fontsize=10, rotation=90, va='center')

	plt.xlim(start_year - 2, end_year + 2)
	plt.legend(
		loc='center left',
		fontsize=10,
		framealpha=0.95,
		frameon=True,
		shadow=True,
		fancybox=True,
	)
	plt.tight_layout()
	plt.savefig(fname=fpth, dpi=DPI, bbox_inches='tight')
	plt.close()

def plot_label_distribution(
		df: pd.DataFrame,
		dname: str,
		fpth: str,
		FIGURE_SIZE: tuple = (12, 7),
		DPI: int = 300,
		top_n: int = None  # Option to show only top N labels
	):

	label_counts = df['label'].value_counts()
	
	# Handle large number of labels
	if top_n and len(label_counts) > top_n:
		top_labels = label_counts.head(top_n)
		other_count = label_counts[top_n:].sum()
		top_labels = pd.concat([top_labels, pd.Series([other_count], index=['Other'])])
		label_counts = top_labels
	
	fig, ax = plt.subplots(figsize=FIGURE_SIZE)
	
	# Plot with better styling
	bars = label_counts.plot(
		kind='bar',
		ax=ax,
		color="green",
		width=0.8,
		edgecolor='white',
		linewidth=0.8,
		alpha=0.8,
		label='Linear'
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
			v + (v * 0.03),  # Adjust vertical position relative to bar height
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
			markersize=5,           # Optional: adjust marker size
			linewidth=1.5,
			alpha=0.8,
			label='Logarithmic'
		)
		ax_log.set_ylabel('Log Frequency', color='red', fontsize=9, fontweight='bold')
		ax_log.tick_params(axis='y', colors='red')
	
	# Hide all spines for the logarithmic scale
	for spine in ax_log.spines.values():
		spine.set_visible(False)

	ax.set_xlabel('Label', fontsize=10)
	ax.set_ylabel('Frequency', fontsize=10)
	
	# Add basic statistics for the distribution
	imbalaned_ratio = label_counts.max()/label_counts.min()
	median_label_size = label_counts.median()
	mean_label_size = label_counts.mean()
	std_label_size = label_counts.std()
	most_freq_label = label_counts.max()/df.shape[0]*100
	least_freq_label = label_counts.min()/df.shape[0]*100
	stats_text = (
		f"Dataset Imbalance ratio: {imbalaned_ratio:.2f}\n\n"
		f"Label Statistics:\n"
		f"    Median: {median_label_size:.0f}\n"
		f"    Mean: {mean_label_size:.2f}\n"
		f"    Standard deviation: {std_label_size:.2f}\n"
		f"    Most frequent: {most_freq_label:.1f}%\n"
		f"    Least frequent: {least_freq_label:.2f}%"
	)
	print(f"stats_text:\n{stats_text}\n")
	plt.text(
		0.865, # horizontal position
		0.87, # vertical position
		stats_text,
		transform=ax.transAxes,
		fontsize=7,
		verticalalignment='top',
		horizontalalignment='left',
		color='black',
		bbox=dict(boxstyle='round,pad=0.5',facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.8)
	)

	# Enhanced title and labels
	plt.title(
		f'{dname} Label Distribution (Total samples: {df.shape[0]} Unique Labels: {len(df["label"].unique())})', 
		fontsize=11, 
		fontweight='bold',
	)
	# Create a single legend
	h1, l1 = ax.get_legend_handles_labels()
	h2, l2 = ax_log.get_legend_handles_labels()
	ax.legend(
		h1 + h2, 
		l1 + l2, 
		loc='best', 
		title='Label Distribution (Scale)',
		title_fontsize=12,
		fontsize=9, 
		ncol=2,
		frameon=True, 
		fancybox=True, 
		shadow=True, 
		edgecolor='black', 
		facecolor='white'
	)

	plt.grid(axis='y', alpha=0.7, linestyle='--')
	plt.tight_layout()
	plt.savefig(fpth, dpi=DPI, bbox_inches='tight')
	plt.close()
