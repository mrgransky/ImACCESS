#!/bin/bash

#SBATCH --job-name=finetune_historyCLIP_finetune_strategy_x_dataset_x_model_architecture_x_with_dropout
#SBATCH --output=/lustre/sgn-data/ImACCESS/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24  # Match the 24 CPUs available on skylake nodes with Tesla V100 32GB
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --constraint=gpumem_32
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-59 # 3 strategies × 5 datasets × 4 model architectures = 60 tasks
#SBATCH --time=07-00:00:00

set -e
set -u
set -o pipefail
user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

# Define constants
FINETUNE_STRATEGIES=("full" "lora" "progressive")
DATASETS=(
/lustre/sgn-data/ImACCESS/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
/lustre/sgn-data/ImACCESS/ImACCESS/WW_DATASETs/HISTORY_X4
/lustre/sgn-data/ImACCESS/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
/lustre/sgn-data/ImACCESS/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
/lustre/sgn-data/ImACCESS/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)
MODEL_ARCHITECTURES=("ViT-B/32" "ViT-B/16" "ViT-L/14" "ViT-L/14@336px")

NUM_DATASETS=${#DATASETS[@]} # Number of datasets
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]} # Number of fine-tune strategies
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]} # Number of model architectures

# Calculate indices from SLURM_ARRAY_TASK_ID
# We're now using a 3D array: strategy × dataset × architecture
total_datasets_x_architectures=$((NUM_DATASETS * NUM_ARCHITECTURES))
strategy_index=$((SLURM_ARRAY_TASK_ID / total_datasets_x_architectures))
remainder=$((SLURM_ARRAY_TASK_ID % total_datasets_x_architectures))
dataset_index=$((remainder / NUM_ARCHITECTURES))
architecture_index=$((remainder % NUM_ARCHITECTURES))

# Note: We can't change SLURM memory allocation dynamically after job submission
# The memory optimization is handled through reduced batch sizes for large models

# Validate indices
if [ $dataset_index -ge ${#DATASETS[@]} ] || 
	 [ $strategy_index -ge ${#FINETUNE_STRATEGIES[@]} ] ||
	 [ $architecture_index -ge ${#MODEL_ARCHITECTURES[@]} ]; then
	echo "Error: Invalid dataset, strategy, or architecture index"
	exit 1
fi

# Hyperparameter configuration
INIT_LRS=(1e-5 1e-5 1e-5 5e-5 8e-6)
INIT_WDS=(1e-2 1e-2 1e-2 1e-2 1e-2)
DROPOUTS=(0.1 0.1 0.05 0.05 0.05)
EPOCHS=(50 50 150 150 150)
LORA_RANKS=(4 4 8 8 8)
LORA_ALPHAS=(16 16 16 16 16)
LORA_DROPOUTS=(0.0 0.0 0.0 0.0 0.0) # TODO: Lora dropout must be 0.05 [original paper]
BATCH_SIZES=(64 32 64 64 64)
PRINT_FREQUENCIES=(250 250 50 50 10)
SAMPLINGS=("kfold_stratified" "stratified_random")

# Set dropout based on strategy
# Only full and progressive can have nonzero dropouts, lora must have zero dropouts
if [ "${FINETUNE_STRATEGIES[$strategy_index]}" = "lora" ]; then
	DROPOUT=0.0
else
	DROPOUT="${DROPOUTS[$dataset_index]}" # Use the original dropout for full and progressive
fi

# Debugging output
echo "=== CONFIGURATION ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "DATASET_INDEX: $dataset_index"
echo "DATASET: ${DATASETS[$dataset_index]}"
echo "STRATEGY_INDEX: $strategy_index"
echo "FINETUNE_STRATEGY: ${FINETUNE_STRATEGIES[$strategy_index]}"
echo "ARCHITECTURE_INDEX: $architecture_index"
echo "MODEL_ARCHITECTURE: ${MODEL_ARCHITECTURES[$architecture_index]}"
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "INITIAL LEARNING RATE: ${INIT_LRS[$dataset_index]}"
echo "INITIAL WEIGHT DECAY: ${INIT_WDS[$dataset_index]}"
echo "DROPOUT: ${DROPOUT}"
echo "DEFAULT BATCH SIZE: ${BATCH_SIZES[$dataset_index]}"

# Dynamically adjust batch size based on model architecture and dataset
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"

# For larger models (ViT-L/14 and ViT-L/14@336px), reduce batch size
if [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"ViT-L"* ]]; then
		# Further reduce batch size for HISTORY_X4 dataset due to its size
		if [[ "${DATASETS[$dataset_index]}" == *"HISTORY_X4"* ]]; then
				ADJUSTED_BATCH_SIZE=8  # Very conservative batch size for large model + large dataset
		else
				ADJUSTED_BATCH_SIZE=16 # Reduced batch size for large models with other datasets
		fi
fi

# Further batch size reduction for the largest model with 336px resolution
if [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"336px"* ]]; then
		# Even smaller batch size for 336px resolution
		if [[ "${DATASETS[$dataset_index]}" == *"HISTORY_X4"* ]]; then
				ADJUSTED_BATCH_SIZE=4  # Extremely conservative for largest model + largest dataset
		else
				ADJUSTED_BATCH_SIZE=8  # Very conservative for largest model with other datasets
		fi
fi

echo "ADJUSTED_BATCH_SIZE: ${ADJUSTED_BATCH_SIZE}"

# Run training command
python -u history_clip_trainer.py \
	--dataset_dir "${DATASETS[$dataset_index]}" \
	--epochs "${EPOCHS[$dataset_index]}" \
	--num_workers "$SLURM_CPUS_PER_TASK" \
	--print_every "${PRINT_FREQUENCIES[$dataset_index]}" \
	--batch_size "${ADJUSTED_BATCH_SIZE}" \
	--learning_rate "${INIT_LRS[$dataset_index]}" \
	--weight_decay "${INIT_WDS[$dataset_index]}" \
	--mode "finetune" \
	--finetune_strategy "${FINETUNE_STRATEGIES[$strategy_index]}" \
	--lora_rank "${LORA_RANKS[$dataset_index]}" \
	--lora_alpha "${LORA_ALPHAS[$dataset_index]}" \
	--lora_dropout "${LORA_DROPOUTS[$dataset_index]}" \
	--sampling "${SAMPLINGS[1]}" \
	--dropout "${DROPOUT}" \
	--model_architecture "${MODEL_ARCHITECTURES[$architecture_index]}"

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"