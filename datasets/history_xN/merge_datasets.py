import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
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
FIGURE_SIZE = (12, 9)
DPI = 350
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
dataset_name = f"history_x{len(DATASETS)}".upper()

HISTORY_XN_DIRECTORY = os.path.join(DATASET_DIRECTORY.get(USER), dataset_name)
OUTPUT_DIRECTORY = os.path.join(HISTORY_XN_DIRECTORY, "outputs")
os.makedirs(HISTORY_XN_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

dfs = []
for i, dataset_path in enumerate(DATASETS):
	print(f"Reading Dataset[{i}]: {dataset_path}")
	try:
		dataset = os.path.basename(dataset_path)
		metadata_file = os.path.join(dataset_path, 'metadata.csv')
		df = pd.read_csv(metadata_file)
		df['dataset'] = dataset
		dfs.append(df)
	except Exception as e:
		print(f"{e}")

print(f"merging {len(dfs)} dataframe(s) to create {dataset_name} dataset...")
merged_df = pd.concat(dfs, ignore_index=True)
print(list(merged_df.columns), merged_df.shape)
print(merged_df.head(10))
merged_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata.csv'), index=False)
all_image_paths = merged_df['img_path'].tolist()
print(f"Total number of images: {len(all_image_paths)}")

plot_label_distribution(
	df=merged_df,
	dname=dataset_name,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_all_labels_distribution.png'),
)

plot_label_distribution_pie_chart(
	df=merged_df,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_all_labels_distribution_pie_chart_{merged_df.shape[0]}_samples.png'),
	figure_size=(7, 11),
	DPI=DPI,
)
plot_grouped_bar_chart(
	merged_df=merged_df,
	DPI=DPI,
	FIGURE_SIZE=(16, 8),
	fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_labels_x_{merged_df['label'].value_counts().shape[0]}_freq_x_{len(dfs)}_datasets_grouped_bar_chart.png")
)

print(f"Stratified Splitting".center(150, "-"))
train_df, val_df = train_test_split(
	merged_df,
	test_size=VAL_SPLIT_PCT,
	shuffle=True,
	stratify=merged_df['label'],
	random_state=42
)

train_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_train.csv'), index=False)
val_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_val.csv'), index=False)


print(train_df.groupby('dataset')['label'].nunique())
print(val_df.groupby('dataset')['label'].nunique())

plot_train_val_label_distribution(
	train_df=train_df,
	val_df=val_df,
	dataset_name=dataset_name,
	OUTPUT_DIRECTORY=OUTPUT_DIRECTORY,
	VAL_SPLIT_PCT=VAL_SPLIT_PCT,
	fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_simple_random_split_stratified_label_distribution_train_val.png'),
	FIGURE_SIZE=(14, 8),
	DPI=DPI,
)

plot_year_distribution(
	df=merged_df,
	dname=dataset_name,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_year_distribution_{merged_df.shape[0]}_samples.png'),
	BINs=BINs,
	FIGURE_SIZE=(18, 10),
	DPI=DPI,
)

# img_rgb_mean_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_mean.gz")
# img_rgb_std_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_std.gz")
# mean, std = get_mean_std_rgb_img_multiprocessing(
# 	source=all_image_paths,
# 	num_workers=8,
# 	batch_size=16,
# 	img_rgb_mean_fpth=img_rgb_mean_fpth,
# 	img_rgb_std_fpth=img_rgb_std_fpth,
# )
# print(f"Mean: {mean}, Std: {std}")