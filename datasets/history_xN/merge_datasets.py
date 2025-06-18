# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)
# from misc.utils import *
# from misc.visualize import *

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(parent_dir)
sys.path.insert(0, project_dir)

from misc.utils import *
from misc.visualize import *


# how to run:
# $ python merge_datasets.py
# $ nohup python -u merge_datasets.py > history_xN_merged_datasets.out &

# run in pouta:
# $ nohup python -u merge_datasets.py > /media/volume/ImACCESS/trash/history_xN_merged_datasets.out &

# run in puhti:
# $ nohup python -u merge_datasets.py > /scratch/project_2004072/ImACCESS/trash/logs/history_xN_merged_datasets.out &

USER = os.getenv("USER")
BINs = 60
VAL_SPLIT_PCT = 0.35
DATASET_DIRECTORY = {
	"farid": "/home/farid/datasets/WW_DATASETs",
	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs",
	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs",
	"alijani": "/lustre/sgn-data/ImACCESS/WW_DATASETs",
}

# DATASETS = [
#       os.path.join(DATASET_DIRECTORY.get(USER), "NATIONAL_ARCHIVE_1900-01-01_1970-12-31"),
#       os.path.join(DATASET_DIRECTORY.get(USER), "EUROPEANA_1900-01-01_1970-12-31"),
#       os.path.join(DATASET_DIRECTORY.get(USER), "SMU_1900-01-01_1970-12-31"),
#       os.path.join(DATASET_DIRECTORY.get(USER), "WWII_1939-09-01_1945-09-02"),
# ]

# Patterns to match dataset directories
DATASET_PATTERNS = [
	"NATIONAL_ARCHIVE*",
	"EUROPEANA*",
	"SMU*",
	"WWII*",
]

# Find all matching dataset directories
DATASETS = []
for pattern in DATASET_PATTERNS:
	DATASETS.extend(glob.glob(os.path.join(DATASET_DIRECTORY.get(USER), pattern)))
print(len(DATASETS), DATASETS)

print("\nAVAILABLE DATASET SUMMARY")
print("-"*100)
for dataset in DATASETS:
	print(f">> {dataset}")
	for file in sorted(os.listdir(dataset)):
		if file.endswith(('.csv')):
			print(f"\t- {file}")

dataset_name = f"history_x{len(DATASETS)}".upper()
HISTORY_XN_DIRECTORY = os.path.join(DATASET_DIRECTORY.get(USER), dataset_name)
OUTPUT_DIRECTORY = os.path.join(HISTORY_XN_DIRECTORY, "outputs")
os.makedirs(HISTORY_XN_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

single_label_dfs = []
for i, dataset_path in enumerate(DATASETS):
	print(f"Reading Dataset[{i}]: {dataset_path}")
	try:
		dataset = os.path.basename(dataset_path)
		single_label_dataset_metadata_fpath = os.path.join(dataset_path, 'metadata_single_label.csv')
		multi_label_dataset_metadata_fpath = os.path.join(dataset_path, 'metadata_multi_label.csv')
		single_label_df = pd.read_csv(single_label_dataset_metadata_fpath)
		single_label_df['dataset'] = dataset
		single_label_dfs.append(single_label_df)
	except Exception as e:
		print(f"{e}")

print(f"merging {len(single_label_dfs)} [single-label] dataframe(s) to create {dataset_name} dataset...")
merged_single_label_df = pd.concat(single_label_dfs, ignore_index=True)
print(list(merged_single_label_df.columns), merged_single_label_df.shape)

num_unique_labels_single_label = merged_single_label_df['label'].nunique()
print(f"Total number of unique labels [single-label]: {num_unique_labels_single_label}")

print(f"Saving {len(single_label_dfs)} merged single-label dataset to {HISTORY_XN_DIRECTORY}")
merged_single_label_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label.csv'), index=False)
try:
	merged_single_label_df.to_excel(os.path.join(HISTORY_XN_DIRECTORY, "metadata_single_label.xlsx"), index=False)
except Exception as e:
	print(f"Failed to write Excel file: {e}")

plot_label_distribution(
	df=merged_single_label_df,
	dname=dataset_name,
	fpth=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_single_label_dataset_{num_unique_labels_single_label}_labels_distribution.png"),
	FIGURE_SIZE=(15,8),
	DPI=400,
	label_column='label',
)

plot_label_distribution_pie_chart(
	df=merged_single_label_df,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_single_label_dataset_{num_unique_labels_single_label}_labels_distribution_pie_chart_{merged_single_label_df.shape[0]}_samples.png'),
	figure_size=(7, 11),
	DPI=400,
)

plot_grouped_bar_chart(
	merged_df=merged_single_label_df,
	FIGURE_SIZE=(15,8),
	DPI=400,
	fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_single_label_dataset_labels_x_{num_unique_labels_single_label}_freq_x_{len(single_label_dfs)}_datasets_grouped_bar_chart.png")
)

print(f"Stratified Splitting".center(150, "-"))
single_label_train_df, single_label_val_df = train_test_split(
	merged_single_label_df,
	test_size=VAL_SPLIT_PCT,
	shuffle=True,
	stratify=merged_single_label_df['label'],
	random_state=42
)

single_label_train_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_train.csv'), index=False)
single_label_val_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_single_label_val.csv'), index=False)

print(single_label_train_df.groupby('dataset')['label'].nunique())
print(single_label_val_df.groupby('dataset')['label'].nunique())

plot_train_val_label_distribution(
	train_df=single_label_train_df,
	val_df=single_label_val_df,
	dataset_name=dataset_name,
	OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
	VAL_SPLIT_PCT=VAL_SPLIT_PCT,
	fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_single_label_dataset_simple_random_split_stratified_label_distribution_train_val.png'),
	FIGURE_SIZE=(15,8),
	DPI=400,
)

plot_year_distribution(
	df=merged_single_label_df,
	dname=dataset_name,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_year_distribution_{merged_single_label_df.shape[0]}_samples.png'),
	BINs=BINs,
)

create_distribution_plot_with_long_tail_analysis(
	df=merged_single_label_df,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_single_label_dataset_{num_unique_labels_single_label}_labels_long_tailed_distribution.png'),
	FIGURE_SIZE=(15,8),
	DPI=400,
)

img_rgb_mean_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_std.gz")
all_image_paths = merged_single_label_df['img_path'].tolist()
print(f">> Computing mean and std for {len(all_image_paths)} images...")
mean, std = get_mean_std_rgb_img_multiprocessing(
	source=all_image_paths,
	num_workers=4,
	batch_size=8,
	img_rgb_mean_fpth=img_rgb_mean_fpth,
	img_rgb_std_fpth=img_rgb_std_fpth,
)
print(f"Mean: {mean}, Std: {std}")