from utils import *
import visualize as viz

def merge_csv_files(dataset_dir, verbose: bool = False):
	output_fpath = os.path.join(dataset_dir, "metadata_multi_label_multimodal.csv")
	# Get a list of all CSV files in the input directory
	csv_files = glob.glob(os.path.join(dataset_dir, 'metadata_multi_label_chunk_*_multimodal.csv'))
	# sort the list of CSV files based on the chunk number
	csv_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-2]))

	if verbose:
		print(f"Found {len(csv_files)} CSV files to merge:")
		for i, file in enumerate(csv_files):
			print(f"\t{i+1:02d}: {file}")
		print(f"Merging CSV files in {dataset_dir} to {output_fpath}...")

	# Initialize an empty DataFrame
	df = pd.DataFrame()

	# Iterate over the CSV files and concatenate them
	for file in csv_files:
		temp_df = pd.read_csv(filepath_or_buffer=file, on_bad_lines='skip', dtype=dtypes, low_memory=False,)
		df = pd.concat([df, temp_df], ignore_index=True)

	if verbose:
		print(f"Saving {type(df)} {df.shape} to {output_fpath}...")

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
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	args = parser.parse_args()
	args.dataset_dir = os.path.normpath(args.dataset_dir)
	print(args)
	merge_csv_files(dataset_dir=args.dataset_dir, verbose=args.verbose)