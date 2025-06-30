#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=historyX4_single_label_finetune_strategy_x_arch # adjust job name!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=373G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=8 # adjust job name!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#SBATCH --time=03-00:00:00

set -euo pipefail

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

SAMPLINGS=(
	"kfold_stratified" 
	"stratified_random"
)

DATASETS=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

DATASET_TYPE=(
	"single_label"
	"multi_label"
)

FINETUNE_STRATEGIES=(
	"full" 				# 0-3, 12-15, 24-27, 36-39, 48-51
	"lora" 				# 4-7, 16-19, 28-31, 40-43, 52-55
	"progressive" # 8-11, 20-23, 32-35, 44-47, 56-59
)

MODEL_ARCHITECTURES=(
	"ViT-L/14@336px" 	# 0, 4, 8, 	12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56
	"ViT-L/14"				# 1, 5, 9, 	13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57
	"ViT-B/32"				# 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58
	"ViT-B/16"				# 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59
)

NUM_DATASETS=${#DATASETS[@]} # Number of datasets
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]} # Number of fine-tune strategies
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]} # Number of model architectures

# dataset × strategy × architecture
### 0-11:  dataset[0] with all strategy×architecture [H4]
### 12-23: dataset[1] with all strategy×architecture [NA]
### 24-35: dataset[2] with all strategy×architecture [EU]
### 36-47: dataset[3] with all strategy×architecture [WWII]
### 48-59: dataset[4] with all strategy×architecture [SMU]
dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_ARCHITECTURES)))
remainder=$((SLURM_ARRAY_TASK_ID % (NUM_STRATEGIES * NUM_ARCHITECTURES)))
strategy_index=$((remainder / NUM_ARCHITECTURES))
architecture_index=$((remainder % NUM_ARCHITECTURES))

# Validate indices
if [ $dataset_index -ge ${#DATASETS[@]} ] || 
	 [ $strategy_index -ge ${#FINETUNE_STRATEGIES[@]} ] ||
	 [ $architecture_index -ge ${#MODEL_ARCHITECTURES[@]} ]; then
	echo "Error: Invalid dataset, strategy, or architecture index"
	exit 1
fi

INIT_LRS=(5.0e-06 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
INIT_WDS=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
DROPOUTS=(0.1 0.1 0.05 0.05 0.05)
EPOCHS=(90 100 150 150 150)
LORA_RANKS=(64 64 64 64 64)
LORA_ALPHAS=(128.0 128.0 128.0 128.0 128.0) # 2x rank
LORA_DROPOUTS=(0.1 0.1 0.05 0.05 0.05)
BATCH_SIZES=(512 64 64 64 64)
PRINT_FREQUENCIES=(1000 1000 50 50 10)
INIT_EARLY_STOPPING_MIN_EPOCHS=(11 25 17 17 12)  # History_X4, National Archive, Europeana, WWII, SMU
EARLY_STOPPING_PATIENCE=(5 5 5 5 5)  # History_X4, National Archive, Europeana, WWII, SMU
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)  # History_X4, National Archive, Europeana, WWII, SMU
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)  # History_X4, National Archive, Europeana, WWII, SMU
CACHE_SIZES=(1024 512 1000 1000 1000)  # History_X4, National Archive, Europeana, WWII, SMU

# Adjust early stopping minimum epochs based on strategy
strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
initial_early_stopping_minimum_epochs="${INIT_EARLY_STOPPING_MIN_EPOCHS[$dataset_index]}"
case $strategy in
	"full")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 5))  # Lower for Full
		;;
	"lora")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 5))  # Higher for LoRA
		;;
	"progressive")
		EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs          # Base for Progressive
		;;
esac
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 5 ? 5 : EARLY_STOPPING_MIN_EPOCHS))  # Ensure minimum of 5

# Set dropout based on strategy
# Only full and progressive can have nonzero dropouts, lora must have zero dropouts
if [ "${FINETUNE_STRATEGIES[$strategy_index]}" = "lora" ]; then
	DROPOUT=0.0
else
	DROPOUT="${DROPOUTS[$dataset_index]}" # Use the original dropout for full and progressive
fi

# Dynamically adjust batch size based on dataset and model architecture
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"
if [[ "${DATASETS[$dataset_index]}" == *"HISTORY_X4"* ]]; then
	if [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"ViT-L"* ]]; then
		ADJUSTED_BATCH_SIZE=32
	else
		ADJUSTED_BATCH_SIZE=128
	fi
elif [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"ViT-L"* ]]; then
	ADJUSTED_BATCH_SIZE=64
fi

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
echo "EARLY_STOPPING_MIN_EPOCHS: ${EARLY_STOPPING_MIN_EPOCHS}"
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: ${ADJUSTED_BATCH_SIZE}"

echo ">> Starting history_clip_trainer.py for (${DATASET_TYPE[1]}) dataset[$SLURM_ARRAY_TASK_ID]: ${DATASETS[$dataset_index]}"

python -u history_clip_trainer.py \
	--dataset_dir "${DATASETS[$dataset_index]}" \
	--dataset_type "${DATASET_TYPE[0]}" \
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
	--minimum_epochs "${EARLY_STOPPING_MIN_EPOCHS}" \
	--patience "${EARLY_STOPPING_PATIENCE[$dataset_index]}" \
	--model_architecture "${MODEL_ARCHITECTURES[$architecture_index]}" \
	--minimum_delta "${EARLY_STOPPING_MIN_DELTA[$dataset_index]}" \
	--cumulative_delta "${EARLY_STOPPING_CUMULATIVE_DELTA[$dataset_index]}" \
	# --cache_size "${CACHE_SIZES[$dataset_index]}" \


done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"