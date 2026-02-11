from utils import *
import visualize as viz
from clustering import cluster
import ast

# how to run:
# Puhti/Mahti:
# srun -J interactive_cpu --account=project_2014707 --partition=large --time=00-23:45:00 --mem=128G --ntasks=1 --cpus-per-task=20 --pty /bin/bash -i
# $ python gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4 -v
# $ nohup python -u gt_kws_multimodal_merge.py -ddir /scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4 -v > /scratch/project_2004072/ImACCESS/trash/logs/interactive_multimodal_annotation_h4.txt &

def merge_csv_files(
	dataset_dir: str,
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
		print(f"Found {len(csv_files)} CSV files to merge:")
		for i, file in enumerate(csv_files):
			print(f"\t{i:02d}: {file}")
		print(f"\n>> Merging {len(csv_files)} CSV files to {output_fpath}")

	df = pd.DataFrame()

	# Iterate over the CSV files and concatenate them
	for file in csv_files:
		temp_df = pd.read_csv(
			filepath_or_buffer=file, 
			on_bad_lines='skip', 
			dtype=dtypes, 
			low_memory=False,
		)
		df = pd.concat([df, temp_df], ignore_index=True)

	clustered_df = cluster(
		labels=df['multimodal_labels'].tolist(),
		# model_id="Qwen/Qwen3-Embedding-8B" if os.getenv('USER') == "alijanif" else "Qwen/Qwen3-Embedding-0.6B",
		model_id="all-MiniLM-L6-v2",
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
	# print only first 10 canonical labels
	print("First 10 canonical labels (label -> canonical_label):")
	print(f"{'label':<30} -> {'canonical_label':<30}")
	for k, v in canonical_labels.items()[:10]:
		print(f"{k:<30} -> {v:<30}")

	canonical_multimodal_labels = []
	multimodal_labels = df['multimodal_labels'].tolist()
	print(f">> Mapping {len(multimodal_labels)} multimodal labels to canonical labels...")
	print(f"multimodal_labels: {type(multimodal_labels)} {len(multimodal_labels)}")
	print(multimodal_labels[:15])

	for labels in tqdm(multimodal_labels, desc="Canonical labels"):
		# Parse string representation to actual list
		if isinstance(labels, str):
			try:
				labels = ast.literal_eval(labels)
			except (ValueError, SyntaxError):
				labels = []
		elif pd.isna(labels):
			labels = []
		elif not isinstance(labels, list):
			labels = []
		
		# Map to canonical labels
		canonical_labels_ = [canonical_labels.get(label, label) for label in labels]
		canonical_multimodal_labels.append(canonical_labels_)

	df['multimodal_canonical_labels'] = canonical_multimodal_labels

	if verbose:
		print(df["multimodal_canonical_labels"].value_counts())
		print(f"Saving {type(df)} {df.shape} {list(df.columns)} to {output_csv}")

	if verbose:
		print(f"Saving {type(df)} {df.shape} to {output_fpath}")
	df.to_csv(output_fpath, index=False)

	try:
		df.to_excel(output_fpath.replace('.csv', '.xlsx'), index=False)
	except Exception as e:
		print(f"Failed to write Excel file: {e}")

	if verbose:
		print(f"Saved merged CSV file to {output_fpath}")

	viz.perform_multilabel_eda(
		data_path=output_fpath,
		label_column='multimodal_labels'
	)

	train_df, val_df = get_multi_label_stratified_split(
		csv_file=output_fpath,
		val_split_pct=0.35,
		label_col='multimodal_labels'
	)

@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	parser.add_argument('--num_clusters', '-nc', type=int, default=None, help='Number of clusters')
	args = parser.parse_args()
	set_seeds(seed=42)

	args.dataset_dir = os.path.normpath(args.dataset_dir)
	if args.verbose:
		print(args)
		print_args_table(args=args, parser=parser)

	merge_csv_files(dataset_dir=args.dataset_dir, verbose=args.verbose, nc=args.num_clusters)

if __name__ == "__main__":
	main()
