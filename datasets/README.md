# HistoryCLIP

This repository contains the implementation and data collection pipeline for HistoryCLIP, a vision-language model trained on historical image datasets.

---

## How to Obtain History-X4 Dataset?

The History-X4 dataset is a large-scale collection of historical images from multiple archives, including:
- **Europeana** (1900-1970)
- **National Archive**
- **SA-Kuva** (Finnish Wartime Photograph Archive)
- **SMU** (Southern Methodist University Digital Collections)
- **WWII Images**
- **WW Vehicles**

### Prerequisites

- Python 3.9+
- Required Python packages (see `requirements.txt`)
- At least 100GB of free disk space
- Stable internet connection

### Step-by-Step Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/mrgransky/ImACCESS.git
cd ImACCESS
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment

Make sure you have the following directory structure:
```
ImACCESS/
├── datasets/
│   ├── europeana/
│   ├── national_archive/
│   ├── sa_kuva/
│   ├── smu/
│   ├── wwii/
│   └── ww_vehicles/
├── misc/
│   ├── query_labels.txt
│   ├── super_labels.json
│   └── meaningless_words.txt
└── ...
```

#### 4. Run the Data Collection Pipeline

Make the collection script executable:
```bash
chmod +x datasets/run_all_data_collectors.sh
```

Run all data collectors (default output: `~/datasets/WW_DATASETs`):
```bash
cd datasets
./run_all_data_collectors.sh
```

Or specify a custom output directory:
```bash
./run_all_data_collectors.sh -d /path/to/your/dataset/directory
```

For help and options:
```bash
./run_all_data_collectors.sh -h
```

#### 5. Monitor Progress

Each dataset collector will:
- Download images from respective archives
- Process and validate images
- Create metadata CSV files
- Generate both single-label and multi-label versions
- Create train/validation splits
- Calculate image statistics (mean/std)
- Generate visualization plots

Logs are saved in each dataset's `logs/` directory:
```
datasets/
├── europeana/logs/
├── national_archive/logs/
├── sa_kuva/logs/
└── ...
```

#### 6. Merge Datasets (Optional)

After individual collection, merge all datasets:
```bash
cd datasets/history_xN
python merge_datasets.py -ddir ~/datasets/WW_DATASETs
```

Or use the interactive option when prompted by the main collection script.

---

## Dataset Structure

After collection, each dataset will have the following structure:

```
DATASET_NAME_START_DATE_END_DATE/
├── images/                          # Downloaded images
├── hits/                            # Raw API responses
├── outputs/                         # Visualizations and analysis
├── metadata_multi_label_raw.csv     # Before image processing (multi-label)
├── metadata_multi_label.csv         # After image processing (multi-label)
├── metadata_multi_label_train.csv   # Training split (multi-label)
├── metadata_multi_label_val.csv     # Validation split (multi-label)
├── metadata_single_label_raw.csv    # Before image processing (single-label)
├── metadata_single_label.csv        # After image processing (single-label)
├── metadata_single_label_train.csv  # Training split (single-label)
├── metadata_single_label_val.csv    # Validation split (single-label)
├── img_rgb_mean.gz                  # Image normalization stats
└── img_rgb_std.gz                   # Image normalization stats
```

## Configuration Options

### Common Arguments for Data Collectors

- `-ddir, --dataset_dir`: Output directory for datasets (required)
- `-sdt, --start_date`: Start date for filtering (default: 1900-01-01)
- `-edt, --end_date`: End date for filtering (default: 1970-12-31)
- `-nw, --num_workers`: Number of parallel workers (default: 12)
- `-bs, --batch_size`: Batch size for processing (default: 128)
- `-vsp, --val_split_pct`: Validation split percentage (default: 0.35)
- `--img_mean_std`: Calculate image mean and standard deviation
- `--enable_thumbnailing`: Enable image thumbnailing for large files
- `--thumbnail_size`: Thumbnail dimensions (default: 1000x1000)
- `--large_image_threshold_mb`: Threshold for thumbnailing in MB (default: 1.0)

### Example: Custom Europeana Collection

```bash
python datasets/europeana/data_collector.py \
    -ddir ~/my_datasets \
    -sdt 1914-01-01 \
    -edt 1918-12-31 \
    -nw 20 \
    --enable_thumbnailing \
    --thumbnail_size 800 800
```

---

## Running on HPC Clusters

For running on High-Performance Computing clusters (Mahti, Puhti, Narvi), use the provided SLURM batch scripts:

### Puhti (CSC, Finland)
```bash
cd datasets/europeana
sbatch puhti_sbatch_europeana_dataset_collector.sh
```

### Mahti (CSC, Finland)
```bash
cd datasets/europeana
sbatch mahti_sbatch_europeana_dataset_collector.sh
```

### Narvi (Tampere University)
```bash
cd datasets/europeana
sbatch narvi_sbatch_europeana_dataset_collector.sh
```

---

## Troubleshooting

### Common Issues

**1. SSL Certificate Errors**
```bash
# Solution: The script automatically retries without SSL verification
# Check logs for details
```

**2. Out of Memory**
```bash
# Reduce number of workers
./run_all_data_collectors.sh -d ~/datasets -nw 4
```

**3. Disk Space Issues**
```bash
# Enable thumbnailing to reduce space
# Edit the script to add: --enable_thumbnailing --large_image_threshold_mb 0.5
```

**4. Network Timeouts**
```bash
# The script has automatic retry logic
# Failed downloads are logged and can be retried
```

---

## Dataset Statistics

Expected dataset sizes (approximate):

| Dataset | Images | Size | Time Period |
|---------|--------|------|-------------|
| Europeana | 50K-100K | 30-50 GB | 1900-1970 |
| National Archive | 20K-50K | 15-30 GB | Various |
| SA-Kuva | 30K-60K | 20-40 GB | 1939-1945 |
| SMU | 10K-30K | 10-20 GB | Various |
| WWII | 15K-40K | 10-25 GB | 1939-1945 |
| WW Vehicles | 5K-15K | 5-10 GB | Various |
| **Total** | **130K-295K** | **90-175 GB** | **1900-1970** |

---

## Citation

If you use this dataset or code, please cite the following paper:

```bibtex
@InProceedings{historyCLIP,
author="Alijani, Farid and Late, Elina and Kumpulainen, Sanna",
title="HistoryCLIP: Adaptive Multi-modal Retrieval of Imbalanced Long-Tailed Archival Data",
booktitle="Linking Theory and Practice of Digital Libraries",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="245--262",
isbn="978-3-032-05409-8",
doi="https://doi.org/10.1007/978-3-032-05409-8_15"
}
```

---

## License

This work is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [farid.alijani@tuni.fi]

---

## Acknowledgments

This project is supported by:
- Tampere University
- Research Council of Finland, grant number 351247. 
- The authors wish to acknowledge CSC – IT Center for Science, Finland, for generous computational resources. 

Data sources:
- Europeana Collections
- U.S. National Archives
- Finnish Heritage Agency (SA-Kuva)
- Southern Methodist University
- WWII Images
- WW Vehicles