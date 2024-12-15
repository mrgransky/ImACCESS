import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DATASETS = [
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31",
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31",
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31",
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02",
]

# Function to get the dataset name from the path
def get_dataset_name(path):
	return os.path.basename(path)

# Initialize an empty list to store the DataFrames
dfs = []

# Loop through each dataset path
for dataset_path in DATASETS:
	# Get the dataset name
	dataset_name = get_dataset_name(dataset_path)
	
	# Read the metadata.csv file from the dataset path
	metadata_file = os.path.join(dataset_path, 'metadata.csv')
	df = pd.read_csv(metadata_file)
	
	# Add a new column to indicate the origin of each row
	df['dataset'] = dataset_name
	
	# Append the DataFrame to the list
	dfs.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)
print(merged_df.shape)
print(merged_df.head(20))
print(merged_df.describe())
# Save the merged DataFrame to a new CSV file in the current directory
merged_df.to_csv('metadata.csv', index=False)
label_counts = merged_df['label'].value_counts()
print(label_counts.tail(25))
plt.figure(figsize=(18, 10))
label_counts.plot(kind='bar', fontsize=9)
plt.title(f'Label Frequency (total: {label_counts.shape})total IMGs: {merged_df.shape[0]}')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f"all_query_labels_x_{label_counts.shape[0]}_freq.png")

# Plotting
plt.figure(figsize=(20, 10))
sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
# Adjust x-axis ticks
ax = plt.gca()
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10))  # Adjust the number of ticks
ax.tick_params(axis='x', rotation=90)  # Rotate the x-axis tick labels
ax.tick_params(axis='y', rotation=90, labelsize=5)  # Rotate the x-axis tick labels

plt.title('Grouped Bar Chart of Labels by Dataset')
plt.legend()
plt.tight_layout()
plt.savefig("label_freq.png")
plt.show()