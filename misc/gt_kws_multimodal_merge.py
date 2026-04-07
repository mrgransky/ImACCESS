from utils import *
import visualize as viz
from nlp_utils import _post_process_
from clustering import get_canonical_labels_with_parallel_mapping, cluster

# how to run:
# local:
# $ python -u gt_kws_multimodal_merge.py -ddir /home/farid/datasets/WW_DATASETs/HISTORY_X4/ -nw 8 -emb "Qwen/Qwen3-Embedding-0.6B" -v
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /home/farid/datasets/WW_DATASETs/HISTORY_X4 -nw 8 -emb "Qwen/Qwen3-Embedding-0.6B" -v > logs/multimodal_merge.log 2>&1 &

# Puhti/Mahti:
# srun -J cpu --account=project_2004072 --partition=large --time=00-13:00:00 --mem=96G --ntasks=1 --cpus-per-task=40 --pty /bin/bash -i
# $ python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4/ -nw 8 -emb "Qwen/Qwen3-Embedding-8B" -v

# new dataset:
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4 -emb "Qwen/Qwen3-Embedding-8B" -nw 8 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_h4.txt &

# old dataset:
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4 -emb "intfloat/e5-mistral-7b-instruct" -nw 20 -v > /scratch/project_2004072/ImACCESS/trash/logs/_interactive_multimodal_annotation_h4.txt &

def merge_csv_files(
	dataset_dir: str,
	num_workers: int,
	batch_size: int,
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
		print(f"\nFound {len(csv_files)} CSV files in {dataset_dir}")
		print(f"\tMerging {len(csv_files)} CSVs => {output_fpath}")

	dfs = list()
	for i, file_ in enumerate(csv_files):
		print(f"{i:02d} {file_}", end="\t")
		temp_df = pd.read_csv(
			filepath_or_buffer=file_, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False,
			usecols=[
				'doc_url', 
				'img_path', 
				'title', 
				'description', 
				'llm_based_labels', 
				'vlm_based_labels', 
				'multimodal_labels',
			]
		)
		print(temp_df.shape, list(temp_df.columns))
		dfs.append(temp_df)

	df = pd.concat(dfs, ignore_index=True)

	if verbose:
		print(f">> Merged {type(df)} from {len(csv_files)} CSV files: {df.shape}\n{list(df.columns)}")
		print(df.info(verbose=True, memory_usage='deep'))

	if verbose:
		print(f"Post-processing LLM-based labels...")
	llm_based_labels = _post_process_(labels_list=df['llm_based_labels'].tolist(), verbose=False)
	df['llm_based_labels'] = llm_based_labels  # Update the DataFrame column

	if verbose:
		print(f"Post-processing VLM-based labels...")
	vlm_based_labels = _post_process_(labels_list=df['vlm_based_labels'].tolist(), verbose=False)
	df['vlm_based_labels'] = vlm_based_labels  # Update the DataFrame column

	if verbose:
		print(f"Post-processing Multimodal labels...")
	multimodal_labels = _post_process_(labels_list=df['multimodal_labels'].tolist(), verbose=False)
	df['multimodal_labels'] = multimodal_labels  # Update the DataFrame column

	llm_canonical_labels, _ = get_canonical_labels_with_parallel_mapping(
		labels=llm_based_labels,
		model_id=embedding_model_id,
		label_source="llm",
		output_dir=OUTPUT_DIR,
		batch_size=batch_size,
		num_workers=num_workers,
		nc=nc,
		verbose=verbose,
	)

	vlm_canonical_labels, _ = get_canonical_labels_with_parallel_mapping(
		labels=vlm_based_labels,
		model_id=embedding_model_id,
		label_source="vlm",
		output_dir=OUTPUT_DIR,
		batch_size=batch_size,
		num_workers=num_workers,
		nc=nc,
		verbose=verbose,
	)

	multimodal_canonical_labels, _ = get_canonical_labels_with_parallel_mapping(
		labels=multimodal_labels,
		model_id=embedding_model_id,
		label_source="multimodal",
		output_dir=OUTPUT_DIR,
		batch_size=batch_size,
		num_workers=num_workers,
		nc=nc,
		verbose=verbose,
	)

	# check length of each before setting into column:
	if verbose:
		print("="*60)
		print(f"Canonical labels lengths:")
		print(f"LLM         {type(llm_canonical_labels)} {len(llm_canonical_labels)}")
		print(f"VLM         {type(vlm_canonical_labels)} {len(vlm_canonical_labels)}")
		print(f"Multimodal: {type(multimodal_canonical_labels)} {len(multimodal_canonical_labels)}")
		print("="*60)

	df['llm_canonical_labels'] = llm_canonical_labels
	df['vlm_canonical_labels'] = vlm_canonical_labels
	df['multimodal_canonical_labels'] = multimodal_canonical_labels

	if verbose:
		print(f"[FILTERING] samples with no valid canonical labels")

	before_count = len(df)
	df = df[df['multimodal_canonical_labels'].apply(lambda x: len(x) if x is not None else 0) > 0].copy()
	after_count = len(df)
	
	if verbose:
		print(f"\n[DONE] Canonical mapping:")
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

	# Deduplicate canonical labels safely
	if verbose:
		print(f"\n>> Deduplicating canonical labels...")

	# Use a helper to handle None values
	def safe_dedup(labels):
		if labels is None:
			return None
		return list(dict.fromkeys(labels))

	df['llm_canonical_labels'] = df['llm_canonical_labels'].apply(safe_dedup)
	df['vlm_canonical_labels'] = df['vlm_canonical_labels'].apply(safe_dedup)
	df['multimodal_canonical_labels'] = df['multimodal_canonical_labels'].apply(safe_dedup)

	# Update assertions to handle the empty lists we just created
	assert sum(1 for labels in df['llm_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0
	assert sum(1 for labels in df['vlm_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0
	assert sum(1 for labels in df['multimodal_canonical_labels'] if labels is not None and len(labels) != len(set(labels))) == 0

	if verbose:
		print(f"   ✓ Deduplication complete")
		llm_duplicate_count = sum(
			1 for labels in df['llm_canonical_labels']
			if labels is not None and len(labels) != len(set(labels))
		)
		vlm_duplicate_count = sum(
			1 for labels in df['vlm_canonical_labels']
			if labels is not None and len(labels) != len(set(labels))
		)
		multimodal_duplicate_count = sum(
			1 for labels in df['multimodal_canonical_labels']
			if labels is not None and len(labels) != len(set(labels))
		)

		print(f"[LLM] Documents with duplicates: {llm_duplicate_count:,} ({llm_duplicate_count/len(df)*100:.1f}%)")
		print(f"[VLM] Documents with duplicates: {vlm_duplicate_count:,} ({vlm_duplicate_count/len(df)*100:.1f}%)")
		print(f"[Multimodal] Documents with duplicates: {multimodal_duplicate_count:,} ({multimodal_duplicate_count/len(df)*100:.1f}%)")
		print(f"   ✓ Verified: 0 duplicates remaining")
	
	if verbose:
		print(f"Saving {type(df)} {df.shape}\n{list(df.columns)}")

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
	parser.add_argument('--batch_size', '-bs', type=int, default=512, help='Batch size')
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
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		embedding_model_id=args.embedding_model_id,
		nc=args.num_clusters,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()
