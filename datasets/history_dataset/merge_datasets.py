import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

USER = os.getenv("USER")

DATASET_DIRECTORY = {
	"farid": "/home/farid/WS_Farid/ImACCESS/datasets/WW_DATASETs",
	"alijanif": "/scratch/project_2004072/ImACCESS/WW_DATASETs/",
}

DATASETS = [
	os.path.join(DATASET_DIRECTORY.get(USER), "NATIONAL_ARCHIVE_1900-01-01_1970-12-31"),
	os.path.join(DATASET_DIRECTORY.get(USER), "EUROPEANA_1900-01-01_1970-12-31"),
	os.path.join(DATASET_DIRECTORY.get(USER), "SMU_1900-01-01_1970-12-31"),
	os.path.join(DATASET_DIRECTORY.get(USER), "WWII_1939-09-01_1945-09-02"),
]

def get_dataset_name(path):
	return os.path.basename(path)

dfs = []
for i, dataset_path in enumerate(DATASETS):
	print(f"Reading Dataset[{i}]: {dataset_path}")
	try:
		dataset_name = get_dataset_name(dataset_path)
		metadata_file = os.path.join(dataset_path, 'metadata.csv')
		df = pd.read_csv(metadata_file)	
		df['dataset'] = dataset_name	
		dfs.append(df)
	except Exception as e:
		print(f"{e}")

merged_df = pd.concat(dfs, ignore_index=True)
print(merged_df.shape)
print(merged_df.head(20))
merged_df.to_csv('metadata.csv', index=False)
label_counts = merged_df['label'].value_counts()
print(label_counts.tail(25))

plt.figure(figsize=(18, 10))
label_counts.plot(kind='bar', fontsize=9)
plt.title(f'Label Frequency (total: {label_counts.shape[0]}) total IMGs: {merged_df.shape[0]}')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"all_query_labels_x_{label_counts.shape[0]}_freq.png")

dataset_unique_label_counts = merged_df.groupby('dataset')['label'].nunique()
print(dataset_unique_label_counts)

plt.figure(figsize=(22, 8))
sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
ax = plt.gca()
ax.tick_params(axis='x', rotation=90, labelsize=9)  # Rotate the x-axis tick labels
ax.tick_params(axis='y', rotation=90, labelsize=9)

# Add dataset label counts to the legend
handles, labels = ax.get_legend_handles_labels()
new_labels = [f"{label} | ({dataset_unique_label_counts[label]})" for label in labels]
ax.legend(handles, new_labels, loc="best", fontsize=10, title="Dataset | (Unique Label Count)")
plt.title(f'Grouped Bar Chart for total of {label_counts.shape[0]} Labels Frequency for {len(dfs)} Datasets')
plt.tight_layout()
plt.savefig(f"labels_x_{label_counts.shape[0]}_freq_x_{len(dfs)}.png")

# Perform stratified splitting
VAL_SPLIT_PCT = 0.35
train_df, val_df = train_test_split(
	merged_df,
	test_size=VAL_SPLIT_PCT,
	stratify=merged_df['label'], 
	random_state=42
)

# Verify label distribution in training set
print("Training Set Label Distribution:")
print(train_df['label'].value_counts(normalize=True))

# Verify label distribution in validation set
print("Validation Set Label Distribution:")
print(val_df['label'].value_counts(normalize=True))

# Save the training set
train_df.to_csv('train_metadata.csv', index=False)

# Save the validation set
val_df.to_csv('val_metadata.csv', index=False)

# Visualize label distribution in training and validation sets
plt.figure(figsize=(18, 6))
train_df['label'].value_counts().plot(kind='bar', color='blue', alpha=0.6, label=f'Train {1-VAL_SPLIT_PCT}')
val_df['label'].value_counts().plot(kind='bar', color='orange', alpha=0.6, label=f'Validation {VAL_SPLIT_PCT}')
plt.title(f'Stratified Label Distribution of {train_df.shape[0]} Training samples {val_df.shape[0]} Validation Samples (Total samples: {merged_df.shape[0]})', fontsize=9)
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.legend(loc='best', ncol=2, frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig('stratified_label_distribution_train_val.png')