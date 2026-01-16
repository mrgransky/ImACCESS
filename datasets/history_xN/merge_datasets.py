import os
from pickle import NONE
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from misc.utils import *
from misc.visualize import *
from misc.preprocess_text import get_enriched_description

# how to run [local]:
# $ python merge_datasets.py -ddir /home/farid/datasets/WW_DATASETs
# $ nohup python -u merge_datasets.py -ddir /home/farid/datasets/WW_DATASETs --target_chunk_mb 8 -v > logs/history_xN_merged_datasets.out &

# run in Pouta:
# $ nohup python -u merge_datasets.py -ddir /media/volume/ImACCESS/dataset/WW_DATASETs --img_mean_std --target_chunk_mb 8 -v > /media/volume/ImACCESS/trash/history_xN_merged_datasets.out &

def get_dataset(ddir: str):
	# Patterns to match dataset directories
	DATASET_PATTERNS = [
		"NATIONAL_ARCHIVE*",
		"EUROPEANA*",
		"SMU*",
		"WWII*",
	]

	datasets = []
	for pattern in DATASET_PATTERNS:
		matches = glob.glob(os.path.join(ddir, pattern))
		datasets.extend(matches)
	print(f"Found {len(datasets)} dataset(s):")

	print("\nAVAILABLE DATASET SUMMARY")
	print("-"*100)
	for i, dataset in enumerate(datasets):
		print(f"Dataset[{i}]: {dataset}")
		for file in sorted(os.listdir(dataset)):
			if file.endswith(('.csv')):
				print(f"\t {file}")
	
	return datasets

def merge_datasets(
	ddir: str, 
	val_split_pct: float=0.35, 
	seed: int=42, 
	head_threshold: int=5000, 
	tail_threshold: int=1000, 
	bins: int=60,
	num_workers: int=16,
	batch_size: int=64,
	target_chunk_mb: int=None,
	img_mean_std: bool=False,
	verbose: bool=False,
):
	datasets = get_dataset(ddir=ddir)

	# Create output directories
	dataset_name = f"history_x{len(datasets)}".upper()
	HISTORY_XN_DIRECTORY = os.path.join(ddir, dataset_name)
	OUTPUT_DIRECTORY = os.path.join(HISTORY_XN_DIRECTORY, "outputs")
	os.makedirs(HISTORY_XN_DIRECTORY, exist_ok=True)
	os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

	# 1. Load and merge single-label dataframes
	if verbose:
		print("\nLoading and merging single-label datasets...")
	single_label_dfs = []
	for i, dataset_path in enumerate(datasets):
		if verbose:
			print(f"Reading Dataset[{i}]: {dataset_path}")
	
		dataset = os.path.basename(dataset_path)
		single_label_path = os.path.join(dataset_path, 'metadata_single_label.csv')
	
		if not os.path.exists(single_label_path):
			print(f"  Warning: {single_label_path} not found, skipping.")
			continue
	
		try:
			df = pd.read_csv(filepath_or_buffer=single_label_path, on_bad_lines='skip', low_memory=False, dtype=dtypes,)
			df['dataset'] = dataset
			single_label_dfs.append(df)
		except Exception as e:
			print(f"  Error loading {single_label_path}: {e}")
	
		if verbose:
			print(f"\tLoaded {type(df)} {df.shape} from {single_label_path}")
	
	if not single_label_dfs:
		raise RuntimeError("No valid datasets found. Exiting.")
	
	if verbose:
		print(f"Merging {len(single_label_dfs)} dataset(s) into '{dataset_name}'...")
	
	merged_single_label_df = pd.concat(single_label_dfs, ignore_index=True)
	print(f"Merged single-label df {type(merged_single_label_df)} {merged_single_label_df.shape} {list(merged_single_label_df.columns)}")
	num_unique_labels = merged_single_label_df['label'].nunique()
	print(f"Unique labels in merged dataset: {num_unique_labels}")
	for label, count in merged_single_label_df['label'].value_counts().items():
		print(f"  - {label}: {count}")
	
	print(f"merged_single_label_df: {type(merged_single_label_df)} {merged_single_label_df.shape}\n{list(merged_single_label_df.columns)}")
	# print(merged_single_label_df.head())

	merged_single_label_df_fpath = os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label.csv')
	print(f"Saving merged single-label dataset to: {merged_single_label_df_fpath}")
	merged_single_label_df.to_csv(merged_single_label_df_fpath, index=False)

	# 2. Load and merge multi-label dataframes
	if verbose:
		print("\nLoading and merging multi-label datasets...")
	multi_label_dfs = []
	for i, dataset_path in enumerate(datasets):
		print(f"Reading Dataset[{i}]: {dataset_path}")
		dataset = os.path.basename(dataset_path)
		multi_label_path = os.path.join(dataset_path, 'metadata_multi_label.csv')
		if not os.path.exists(multi_label_path):
			print(f"  Warning: {multi_label_path} not found, skipping.")
			continue
		try:
			df = pd.read_csv(filepath_or_buffer=multi_label_path, on_bad_lines='skip', low_memory=False, dtype=dtypes,)
			df['dataset'] = dataset
			multi_label_dfs.append(df)
		except Exception as e:
			print(f"  Error loading {multi_label_path}: {e}")
		print(f"\tLoaded {type(df)} {df.shape} from {multi_label_path}")
	if not multi_label_dfs:
		raise RuntimeError("No valid datasets found. Exiting.")

	print(f"Merging {len(multi_label_dfs)} dataset(s) into '{dataset_name}'...")
	merged_multi_label_df = pd.concat(multi_label_dfs, ignore_index=True)	

	# 2.1 Inspect merged multi-label dataframe
	print(f"merged_multi_label_df: {type(merged_multi_label_df)} {merged_multi_label_df.shape}\n{list(merged_multi_label_df.columns)}")
	# print(merged_multi_label_df.head())
	
	# 2.2 Add enriched_document_description column to merged_multi_label_df
	merged_multi_label_df = get_enriched_description(df=merged_multi_label_df, check_english=True, verbose=verbose)

	if "enriched_document_description" not in merged_multi_label_df.columns:
		raise ValueError("enriched_document_description column not found in merged_multi_label_df")

	# 2.3 Save merged multi-label dataframe
	merged_multi_label_df_fpath = merged_single_label_df_fpath.replace('single', 'multi')
	print(f"Saving merged multi-label dataset to: {merged_multi_label_df_fpath}")
	merged_multi_label_df.to_csv(merged_multi_label_df_fpath, index=False)

	# 2.4 Chunk merged multi-label dataframe
	if target_chunk_mb is not None:
		print(f"Chunking merged multi-label dataset to {target_chunk_mb} MB...")
		# Calculate optimal chunk size
		estimated_mb_per_row = merged_multi_label_df.memory_usage(deep=True).sum() / (1024**2) / len(merged_multi_label_df)
		
		optimal_chunk_size = max(1000, min(50000, int(target_chunk_mb / estimated_mb_per_row)))
		total_num_chunks = (len(merged_multi_label_df) + optimal_chunk_size - 1) // optimal_chunk_size
		print(f"Saving {len(merged_multi_label_df)} rows in {total_num_chunks} chunks of ~{optimal_chunk_size} rows each (estimated_mb_per_row: {estimated_mb_per_row:.3f} MB)...")
		# Save in chunks
		for i in range(total_num_chunks):
			start_idx = i * optimal_chunk_size
			end_idx = min((i + 1) * optimal_chunk_size, len(merged_multi_label_df))
			chunk_df = merged_multi_label_df.iloc[start_idx:end_idx]
			chunk_df_fpth = merged_multi_label_df_fpath.replace('.csv', f'_chunk_{i}.csv')
			chunk_df.to_csv(chunk_df_fpth, index=False)
			if verbose:
				print(f"  Saved chunk {i+1}/{total_num_chunks}: {chunk_df.shape[0]} rows -> {chunk_df_fpth}")
	
	try:
		merged_single_label_df.to_excel(merged_single_label_df_fpath.replace('.csv', '.xlsx'), index=False)
		merged_multi_label_df.to_excel(merged_multi_label_df_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")
	
	if verbose:
		print("\nGenerating label distribution plots...")
	plot_label_distribution(
		df=merged_single_label_df,
		fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_single_label_{num_unique_labels}_labels_dist.png"),
		FIGURE_SIZE=(15, 8),
		DPI=300,
		label_column='label',
	)
	plot_label_distribution_pie_chart(
		df=merged_single_label_df,
		fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_single_label_pie_chart_{merged_single_label_df.shape[0]}_samples.png"),
		figure_size=(7, 11),
		DPI=300,
	)
	plot_grouped_bar_chart(
		merged_df=merged_single_label_df,
		FIGURE_SIZE=(15, 8),
		DPI=300,
		fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_grouped_bar_chart_{merged_single_label_df.shape[0]}_samples_{num_unique_labels}_labels.png")
	)

	# Stratified train/val split
	if verbose:
		print("Stratified Splitting".center(150, "-"))
	single_label_train_df, single_label_val_df = train_test_split(
		merged_single_label_df,
		test_size=val_split_pct,
		shuffle=True,
		stratify=merged_single_label_df['label'],
		random_state=seed
	)
	single_label_train_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_train.csv'), index=False)
	single_label_val_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_val.csv'), index=False)
	
	if verbose:
		print("Labels per dataset in train split:")
		print(single_label_train_df.groupby('dataset')['label'].nunique())
		print("Labels per dataset in val split:")
		print(single_label_val_df.groupby('dataset')['label'].nunique())
	
	plot_train_val_label_distribution(
		train_df=single_label_train_df,
		val_df=single_label_val_df,
		dataset_name=dataset_name,
		VAL_SPLIT_PCT=val_split_pct,
		fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_train_val_label_dist_{merged_single_label_df.shape[0]}_samples_{num_unique_labels}_labels.png"),
		FIGURE_SIZE=(15, 8),
		DPI=250,
	)

	plot_year_distribution(
		df=merged_single_label_df,
		dname=dataset_name,
		fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_year_dist_{merged_single_label_df.shape[0]}_samples.png"),
		BINs=bins,
	)

	plot_long_tailed_distribution(
		df=merged_single_label_df,
		fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_long_tailed_dist_{merged_single_label_df.shape[0]}_samples_{num_unique_labels}_labels.png"),
		head_threshold=head_threshold,
		tail_threshold=tail_threshold,
	)

	plot_single_labeled_head_torso_tail_samples(
		metadata_path=os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label.csv'),
		metadata_train_path=os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_train.csv'),
		metadata_val_path=os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_val.csv'),
		save_path=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_head_torso_tail_samples_{merged_single_label_df.shape[0]}_samples_{num_unique_labels}_labels.png"),
		head_threshold=head_threshold,
		tail_threshold=tail_threshold,
	)

	if img_mean_std:
		all_image_paths = merged_single_label_df['img_path'].tolist()
		img_rgb_mean_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_mean.gz")
		img_rgb_std_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_std.gz")
		try:
			img_rgb_mean, img_rgb_std = load_pickle(fpath=img_rgb_mean_fpth), load_pickle(fpath=img_rgb_std_fpth) # RGB images
		except Exception as e:
			print(f"<!> {e}")
			num_workers = min(num_workers, multiprocessing.cpu_count())
			if verbose:
				print(f"Computing RGB mean and std across all images with {num_workers} workers (this may take a while)...")
			img_rgb_mean, img_rgb_std = get_mean_std_rgb_img_multiprocessing(
				source=all_image_paths,
				num_workers=num_workers,
				batch_size=batch_size,
				img_rgb_mean_fpth=img_rgb_mean_fpth,
				img_rgb_std_fpth=img_rgb_std_fpth,
				verbose=verbose,
			)

	if verbose:
		print(f"Files created in {HISTORY_XN_DIRECTORY}:")
		for file in sorted(os.listdir(HISTORY_XN_DIRECTORY)):
			if file.endswith(('.csv', '.xlsx')):
				print(f"\t{file}")

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description="Merge multiple WW datasets into a single consolidated dataset with train/val splits and visualizations.")
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Dataset root directory')
	parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
	parser.add_argument('--bins', type=int, default=60, help='Number of bins for year distribution histogram (default: 60)')
	parser.add_argument('--val_split_pct', type=float, default=0.35, help='Validation split percentage (default: 0.35)')
	parser.add_argument('--head_threshold', type=int, default=5000, help='Threshold for head class in long-tail analysis (default: 5000)')
	parser.add_argument('--tail_threshold', type=int, default=1000, help='Threshold for tail class in long-tail analysis (default: 1000)')
	parser.add_argument('--num_workers', '-nw', type=int, default=4, help='Number of workers for image stats computation (default: min(16, cpu_count))')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size for computing image statistics (default: 64)')
	parser.add_argument('--target_chunk_mb', type=int, default=None, help='Target chunk size in MB (None = no chunking)')
	parser.add_argument('--img_mean_std', action='store_true', help='calculate image mean & std')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')

	args = parser.parse_args()
	args.dataset_dir = os.path.normpath(args.dataset_dir)

	if args.verbose:
		print_args_table(args=args, parser=parser)
		print(args)
	
	set_seeds(seed=args.seed)

	merge_datasets(
		ddir=args.dataset_dir, 
		val_split_pct=args.val_split_pct, 
		seed=args.seed, 
		head_threshold=args.head_threshold, 
		tail_threshold=args.tail_threshold, 
		bins=args.bins, 
		num_workers=args.num_workers, 
		batch_size=args.batch_size,
		target_chunk_mb=args.target_chunk_mb,
		img_mean_std=args.img_mean_std,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()