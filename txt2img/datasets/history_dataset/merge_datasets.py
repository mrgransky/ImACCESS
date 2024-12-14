import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASETS = [
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31",
	"/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31",
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

# Plotting
plt.figure(figsize=(18, 10))
sns.countplot(x="label", hue="dataset", data=merged_df, palette="bright")
plt.title("Distribution of Labels by Dataset")
plt.xlabel("Label")
plt.ylabel("Count")
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("label_freq.png")
plt.show()