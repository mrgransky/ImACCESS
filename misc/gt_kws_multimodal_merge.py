from utils import *
import visualize as viz

# how to run:
# Puhti/Mahti:
# srun -J interactive_cpu --account=project_2009043 --partition=large --time=00-05:15:00 --mem=128G --ntasks=1 --cpus-per-task=20 --pty /bin/bash -i
# $ python gt_kws_multimodal_merge.py --dataset_dir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4 -v

def merge_csv_files(dataset_dir, verbose: bool = False):
	output_fpath = os.path.join(dataset_dir, "metadata_multi_label_multimodal.csv")
	# Get a list of all CSV files in the input directory
	csv_files = glob.glob(os.path.join(dataset_dir, 'metadata_multi_label_chunk_*_multimodal.csv'))
	# sort the list of CSV files based on the chunk number
	csv_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-2]))

	if verbose:
		print(f"Found {len(csv_files)} CSV files to merge:")
		for i, file in enumerate(csv_files):
			print(f"\t{i:02d}: {file}")
		print(f"Merging {len(csv_files)} CSV files to {output_fpath}...")

	# Initialize an empty DataFrame
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
	
@measure_execution_time
def main():
	parser = argparse.ArgumentParser(description='Merge CSV files')
	parser.add_argument('--dataset_dir', '-ddir', type=str, required=True, help='Directory containing CSV files')
	parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
	args = parser.parse_args()
	set_seeds(seed=42)

	args.dataset_dir = os.path.normpath(args.dataset_dir)
	if args.verbose:
		print(args)
		print_args_table(args=args, parser=parser)

	merge_csv_files(dataset_dir=args.dataset_dir, verbose=args.verbose)

if __name__ == "__main__":
	main()