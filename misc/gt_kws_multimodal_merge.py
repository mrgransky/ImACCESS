from utils import *
import visualize as viz
from nlp_utils import _post_process_
from clustering import cluster

# how to run:
# Puhti/Mahti:
# srun -J interactive_cpu --account=project_2014707 --partition=large --time=00-23:45:00 --mem=256G --ntasks=1 --cpus-per-task=40 --pty /bin/bash -i
# $ python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4/ -nw 40 -v
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4 -m "Qwen/Qwen3-Embedding-8B" -nw 40 -nc 2500 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_h4.txt &

# Global variable for worker processes
canonical_labels_global = None

def init_worker_canonical(canonical_dict):
	"""Initialize each worker process with the canonical labels dictionary"""
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
	# Map to canonical labels using global dict
	return [canonical_labels_global.get(label, label) for label in labels]

def merge_csv_files(
	dataset_dir: str,
	num_workers: int,
	model_id: str,
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

	# post_process multimodal labels:
	multimodal_labels = _post_process_(
		labels_list=df['multimodal_labels'].tolist(), 
		# verbose=verbose,
	)

	clustered_df = cluster(
		labels=multimodal_labels,
		model_id=model_id,
		batch_size=2048,
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
	print("First 10 canonical labels (label -> canonical_label):")
	samples = {k:v for i, (k, v) in enumerate(canonical_labels.items()) if i < 10}
	print(json.dumps(samples, indent=2, ensure_ascii=False))

	# ========== Parallel mapping ==========
	chunksize = max(1, len(df) // (num_workers * 4))  # 4 chunks per worker
	print(f"Mapping {len(df)} multimodal labels to canonical labels using {num_workers} cores with chunks: {chunksize}")
	t_start = time.time()
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
	elapsed = time.time() - t_start

	if verbose:
		print(f"Mapping completed in {elapsed:.4f}s ({len(df)/elapsed:.1f} rows/sec)")
		print(f">> canonical_multimodal_labels: {len(df['multimodal_canonical_labels'])}")
		print(df['multimodal_canonical_labels'].head(15).tolist())
		print(f"\n>> Saving {type(df)} {df.shape} to {output_fpath}\n{list(df.columns)}")
		# Check if "discharge" and "hospital discharge" are in the SAME cluster
		discharge_cluster = df[df['label'] == 'discharge']['cluster'].iloc[0]
		hospital_discharge_cluster = df[df['label'] == 'hospital discharge']['cluster'].iloc[0]

		print(f"discharge cluster: {discharge_cluster}")
		print(f"hospital discharge cluster: {hospital_discharge_cluster}")
		print(f"Same cluster? {discharge_cluster == hospital_discharge_cluster}")

		if discharge_cluster == hospital_discharge_cluster:
				print("❌ THEY ARE STILL TOGETHER!")
				canonical = df[df['cluster'] == discharge_cluster]['canonical'].iloc[0]
				print(f"Canonical: {canonical}")

	df.to_csv(output_fpath, index=False)

	try:
		df.to_excel(output_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if verbose:
		print(f"Saved merged CSV file to {output_fpath}")

	viz.perform_multilabel_eda(
		data_path=output_fpath,
		label_column='multimodal_canonical_labels'
	)

	train_df, val_df = get_multi_label_stratified_split(
		csv_file=output_fpath,
		val_split_pct=0.35,
		label_col='multimodal_canonical_labels'
	)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--num_workers', '-nw', type=int, required=True, help='Number of workers for parallel processing')
	parser.add_argument('--model_id', '-m', type=str, default="sentence-transformers/all-MiniLM-L6-v2", help='HuggingFace model ID')
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
		model_id=args.model_id,
		nc=args.num_clusters,
		verbose=args.verbose,
	)

if __name__ == "__main__":
	main()