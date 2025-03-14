import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from misc.utils import *

# how to run:
# $ python merge_datasets.py
# $ nohup python -u merge_datasets.py > history_xN_merged_datasets.out &

# run in puhti:
# $ nohup python -u merge_datasets.py > /scratch/project_2004072/ImACCESS/trash/logs/history_xN_merged_datasets.out &

USER = os.getenv("USER")
FIGURE_SIZE = (13, 7)
DPI = 250
BINs = 60

VAL_SPLIT_PCT = 0.35
DATASET_DIRECTORY = {
	"farid": "/home/farid/datasets/WW_DATASETs",
	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs",
	"ubuntu": "/media/volume/ImACCESS/WW_DATASETs",
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
label_counts = merged_df['label'].value_counts()
all_image_paths = merged_df['img_path'].tolist()
print(f"Total number of images: {len(all_image_paths)}")
# print(label_counts.tail(25))

# Visualize label distribution
plt.figure(figsize=FIGURE_SIZE)
label_counts.plot(kind='bar', fontsize=9)
plt.title(f'Label Frequency (total: {label_counts.shape[0]}) total IMGs: {merged_df.shape[0]}')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.yticks(rotation=90, fontsize=9,)
plt.tight_layout()
plt.savefig(
	fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_all_query_labels_x_{label_counts.shape[0]}_freq.png"),
	dpi=DPI,
	bbox_inches='tight'
)

dataset_unique_label_counts = merged_df.groupby('dataset')['label'].nunique()
print(dataset_unique_label_counts)
plt.figure(figsize=FIGURE_SIZE)
sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
ax = plt.gca()
ax.tick_params(axis='x', rotation=90, labelsize=9)  # Rotate the x-axis tick labels
ax.tick_params(axis='y', rotation=90, labelsize=9)
handles, labels = ax.get_legend_handles_labels()
new_labels = [f"{label} | ({dataset_unique_label_counts[label]})" for label in labels]
ax.legend(handles, new_labels, loc="best", fontsize=8, title="Dataset | (Unique Label Count)")
plt.title(f'Grouped Bar Chart for total of {label_counts.shape[0]} Labels Frequency for {len(dfs)} Datasets')
plt.tight_layout()
plt.savefig(
	fname=os.path.join(OUTPUT_DIRECTORY, f"{dataset_name}_labels_x_{label_counts.shape[0]}_freq_x_{len(dfs)}.png"),
	dpi=DPI,
	bbox_inches='tight'
)

# stratified splitting
train_df, val_df = train_test_split(
	merged_df,
	test_size=VAL_SPLIT_PCT,
	shuffle=True,
	stratify=merged_df['label'],
	random_state=42
)

train_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_train.csv'), index=False)
val_df.to_csv(os.path.join(HISTORY_XN_DIRECTORY, 'metadata_val.csv'), index=False)

# Visualize label distribution in training and validation sets
plt.figure(figsize=FIGURE_SIZE)
train_df['label'].value_counts().plot(kind='bar', color='blue', alpha=0.6, label=f'Train {1-VAL_SPLIT_PCT}')
val_df['label'].value_counts().plot(kind='bar', color='red', alpha=0.9, label=f'Validation {VAL_SPLIT_PCT}')
plt.title(f'Stratified Label Distribution of {train_df.shape[0]} Training samples {val_df.shape[0]} & Validation Samples (Total: {merged_df.shape[0]})', fontsize=9)
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.yticks(rotation=90, fontsize=9,)
plt.legend(loc='best', ncol=2, frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(
	fname=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_simple_random_split_stratified_label_distribution_train_val.png'),
	dpi=DPI,
	bbox_inches='tight'
)
plt.close()

plot_year_distribution(
	df=merged_df,
	dname=dataset_name,
	fpth=os.path.join(OUTPUT_DIRECTORY, f'{dataset_name}_year_distribution_{merged_df.shape[0]}_samples.png'),
	BINs=BINs,
)
img_rgb_mean_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_mean.gz")
img_rgb_std_fpth = os.path.join(HISTORY_XN_DIRECTORY, "img_rgb_std.gz")

mean, std = get_mean_std_rgb_img_multiprocessing(
	source=all_image_paths,
	num_workers=8,
	batch_size=16,
	img_rgb_mean_fpth=img_rgb_mean_fpth,
	img_rgb_std_fpth=img_rgb_std_fpth,
)
print(f"Mean: {mean}, Std: {std}")