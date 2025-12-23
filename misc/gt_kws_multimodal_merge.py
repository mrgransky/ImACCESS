from utils import *
import visualize as viz

def merge_csv_files(dataset_dir, verbose: bool = False):
	output_fpath = os.path.join(dataset_dir, "metadata_multi_label_multimodal.csv")
	if verbose:
		for file in glob.glob(os.path.join(dataset_dir, 'metadata_multi_label_chunk_*_multimodal.csv')):
			print(f"  {file}")
		print(f"Merging CSV files in {dataset_dir} to {output_fpath}...")

	# Get a list of all CSV files in the input directory
	csv_files = glob.glob(os.path.join(dataset_dir, 'metadata_multi_label_chunk_*_multimodal.csv'))
	if verbose:
		print(f"Found {len(csv_files)} CSV files to merge:")

	# Initialize an empty DataFrame
	df = pd.DataFrame()

	# Iterate over the CSV files and concatenate them
	for file in csv_files:
		temp_df = pd.read_csv(file)
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
		data_path=output_csv,
		label_column='multimodal_labels'
	)

	train_df, val_df = get_multi_label_stratified_split(
		csv_file=output_csv,
		val_split_pct=0.35,
		label_col='multimodal_labels'
	)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	args.dataset_dir = os.path.normpath(args.dataset_dir)
	args = parser.parse_args()
	print(args)
	merge_csv_files(dataset_dir=args.dataset_dir, verbose=args.verbose)