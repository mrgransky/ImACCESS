from utils import *
import visualize as viz
from nlp_utils import _post_process_
from clustering import get_canonical_labels_parallel, cluster

# how to run:
# Puhti/Mahti:
# srun -J cpu --account=project_2004072 --partition=large --time=00-13:45:00 --mem=164G --ntasks=1 --cpus-per-task=40 --pty /bin/bash -i
# $ python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4/ -nw 8 -v

# new dataset:
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4 -emb "Qwen/Qwen3-Embedding-8B" -nw 40 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_h4.txt &

# old dataset:
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4 -emb "Qwen/Qwen3-Embedding-8B" -nw 20 -v > /scratch/project_2004072/ImACCESS/trash/logs/_interactive_multimodal_annotation_h4.txt &

# Global variable for worker processes
canonical_labels_global = None

def init_worker_canonical(canonical_dict):
	global canonical_labels_global
	canonical_labels_global = canonical_dict

def parallel_canonical_mapping(labels_str):
	if isinstance(labels_str, str):
		try:
			labels = ast.literal_eval(labels_str)
		except (ValueError, SyntaxError):
			return []
	elif labels_str is None or (isinstance(labels_str, float) and math.isnan(labels_str)):
		return []
	elif isinstance(labels_str, list):
		labels = labels_str
	else:
		return []

	# # Map to canonical labels using global dict
	# return [canonical_labels_global.get(label, label) for label in labels]

	# Map to canonical labels, SKIPPING labels not in dict
	# (these are labels that were removed as problematic)
	canonical_labels_ = []
	for label in labels:
		if label in canonical_labels_global:
			canonical_labels_.append(canonical_labels_global[label])
		# else: label was removed as problematic, skip it
	
	return canonical_labels_

def merge_csv_files(
	dataset_dir: str,
	num_workers: int,
	embedding_model_id: str,
	nc: int = None,
	verbose: bool = False
):
	OUTPUT_DIR = os.path.join(dataset_dir, "outputs")
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	output_fpath = os.path.join(dataset_dir, "metadata_multi_label_multimodal.csv")
	# Get a list of all CSV files in the input directory
	csv_files = glob.glob(os.path.join(dataset_dir, 'metadata_multi_label_chunk_*_multimodal.csv'))
	# sort the list of CSV files based on the chunk number
	csv_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-2]))

	if verbose:
		print(f"Found {len(csv_files)} CSV files in {dataset_dir}")
		for i, file in enumerate(csv_files):
			print(f"\t{i:02d}: {file}")
		print(f"\n>> Merging {len(csv_files)} CSV files to {output_fpath}")

	dfs = []
	for file in csv_files:
		temp_df = pd.read_csv(
			filepath_or_buffer=file, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False
		)
		dfs.append(temp_df)
	df = pd.concat(dfs, ignore_index=True)
	print(f">> Merged {type(df)} from {len(csv_files)} CSV files: {df.shape}\n{list(df.columns)}")


	multimodal_labels = df['multimodal_labels'].tolist()
	multimodal_labels = _post_process_(labels_list=multimodal_labels, verbose=False)

	# get_canonical_labels_parallel(
	# 	labels=multimodal_labels,
	# 	model_id=embedding_model_id,
	# 	label_source="multimodal",
	# 	output_dir=OUTPUT_DIR,
	# 	num_workers=num_workers,
	# 	nc=nc,
	# 	verbose=verbose,
	# )

	clustered_df = cluster(
		labels=multimodal_labels,
		model_id=embedding_model_id,
		batch_size=4096,
		nc=nc,
		clusters_fname=os.path.join(OUTPUT_DIR, os.path.basename(output_fpath).replace(".csv", "_clusters.csv")),
		verbose=verbose,
	)

	# canonical label available in clustered_df "canonical" column: [label] -> canonical_label]
	# mapping each label of multimodal_labels to its canonical label:
	# desired example:
	# multimodal_labels						-> multimodal_canonical_labels (new column)
	# [label_1, label_2, label_3] -> [canonical_label_1, canonical_label_2, canonical_label_3]
	canonical_labels = clustered_df.set_index('label')['canonical'].to_dict()
	print(f">> canonical_labels: {type(canonical_labels)} {len(canonical_labels)}")

	# Parallel mapping
	chunksize = max(1, len(df) // (num_workers * 4))  # 4 chunks per worker
	print(f"Mapping {len(df)} samples to their corresponding canonical labels")
	print(f"num_workers: {num_workers} chunksize: {chunksize}")
	with multiprocessing.Pool(
		processes=num_workers,
		initializer=init_worker_canonical, # Called ONCE per worker
		initargs=(canonical_labels,) # Sent ONCE per worker
	) as pool:
		df['multimodal_canonical_labels'] = pool.map(
			parallel_canonical_mapping,
			multimodal_labels,
			chunksize=chunksize
		)

	# Filter out samples with no valid canonical labels
	before_count = len(df)
	df = df[df['multimodal_canonical_labels'].apply(len) > 0].copy()
	after_count = len(df)
	
	if verbose:
		print(f"\n>> Canonical mapping complete:")
		print(f"   Samples before: {before_count:,}")
		print(f"   Samples after: {after_count:,}")
		if before_count != after_count:
			removed = before_count - after_count
			print(f"   Removed {removed:,} samples with no valid labels ({removed/before_count*100:.2f}%)")
		
		# Show some statistics
		label_counts = df['multimodal_canonical_labels'].apply(len)
		print(f"\n   Labels per sample:")
		print(f"     Mean: {label_counts.mean():.2f}")
		print(f"     Median: {label_counts.median():.0f}")
		print(f"     Min: {label_counts.min()}")
		print(f"     Max: {label_counts.max()}")

	# DEDUPLICATION: Remove duplicate canonical labels
	if verbose:
		print(f"\n>> Deduplicating canonical labels...")
		
		# Count documents with duplicates BEFORE
		duplicate_count = sum(
			1 for labels in df['multimodal_canonical_labels'] 
			if len(labels) != len(set(labels))
		)
		print(f"   Documents with duplicates: {duplicate_count:,} ({duplicate_count/len(df)*100:.1f}%)")
	
	# Deduplicate while preserving order
	df['multimodal_canonical_labels'] = df['multimodal_canonical_labels'].apply(lambda labels: list(dict.fromkeys(labels)))
	
	if verbose:
		print(f"   ✓ Deduplication complete")
		
		# Verify no duplicates remain
		duplicate_count_after = sum(
			1 for labels in df['multimodal_canonical_labels'] 
			if len(labels) != len(set(labels))
		)
		assert duplicate_count_after == 0, "Duplicates still present after deduplication!"
		print(f"   ✓ Verified: 0 duplicates remaining")
	
		print(f"\n>> canonical_multimodal_labels: {len(df['multimodal_canonical_labels'])}")
		print(df['multimodal_canonical_labels'].head(15).tolist())

	df.to_csv(output_fpath, index=False)

	try:
		df.to_excel(output_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if verbose:
		print(f"Saved merged CSV file to {output_fpath}")

	viz.multilabel_eda(
		df=df,
		output_dir=OUTPUT_DIR,
		label_column='multimodal_canonical_labels'
	)

	get_multi_label_stratified_split(
		df=df,
		csv_file=output_fpath,
		val_split_pct=0.35,
		label_col='multimodal_canonical_labels',
		min_label_frequency=5,
	)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--num_workers', '-nw', type=int, required=True, help='Number of workers for parallel processing')
	parser.add_argument('--embedding_model_id', '-emb', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='HuggingFace model ID')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	parser.add_argument('--num_clusters', '-nc', type=int, default=None, help='Number of clusters')
	args = parser.parse_args()
	args.dataset_dir = os.path.normpath(args.dataset_dir)
	args.num_workers = min(args.num_workers, multiprocessing.cpu_count())
	set_seeds(seed=42)

	if args.verbose:
		print(args)
		print_args_table(args=args, parser=parser)

	merge_csv_files(
		dataset_dir=args.dataset_dir, 
		num_workers=args.num_workers,
		embedding_model_id=args.embedding_model_id,
		nc=args.num_clusters,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()