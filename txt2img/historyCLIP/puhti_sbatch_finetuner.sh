#!/bin/bash
#SBATCH --account=project_2009043
#SBATCH --job-name=finetune_historyCLIP_with_dropout_strategy_x_dataset_x
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=51G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-14 # 3 strategies × 5 datasets = 15 tasks
#SBATCH --time=03-00:00:00
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
/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)
NUM_DATASETS=${#DATASETS[@]} # Number of datasets
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]} # Number of fine-tune strategies
# Calculate dataset and strategy indices from SLURM_ARRAY_TASK_ID
dataset_index=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))
strategy_index=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))
# Validate indices
if [ $dataset_index -ge ${#DATASETS[@]} ] || [ $strategy_index -ge ${#FINETUNE_STRATEGIES[@]} ]; then
echo "Error: Invalid dataset or strategy index"
exit 1
fi
# Hyperparameter configuration
INIT_LRS=(1e-4 1e-4 1e-4 1e-5 2e-5)
WEIGHT_DECAYS=(1e-1 1e-1 1e-1 1e-1 1e-1)
DROPOUTS=(0.1 0.1 0.1 0.2 0.2)
EPOCHS=(50 50 150 150 150)
LORA_RANKS=(4 4 16 16 16)
LORA_ALPHAS=(16 16 16 16 16)
LORA_DROPOUTS=(0.0 0.0 0.0 0.0 0.0) # TODO: Lora dropout must be 0.05 [original paper]
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
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "INIT_LR: ${INIT_LRS[$dataset_index]}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAYS[$dataset_index]}"
echo "DROPOUT: ${DROPOUT}"

# Run training command
python -u history_clip_trainer.py \
	--dataset_dir "${DATASETS[$dataset_index]}" \
	--epochs "${EPOCHS[$dataset_index]}" \
	--num_workers "$SLURM_CPUS_PER_TASK" \
	--print_every 250 \
	--batch_size 256 \
	--learning_rate "${INIT_LRS[$dataset_index]}" \
	--weight_decay "${WEIGHT_DECAYS[$dataset_index]}" \
	--mode "finetune" \
	--finetune_strategy "${FINETUNE_STRATEGIES[$strategy_index]}" \
	--lora_rank "${LORA_RANKS[$dataset_index]}" \
	--lora_alpha "${LORA_ALPHAS[$dataset_index]}" \
	--lora_dropout "${LORA_DROPOUTS[$dataset_index]}" \
	--sampling "${SAMPLINGS[1]}" \
	--dropout "${DROPOUT}" \
	--model_architecture "ViT-B/32"

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"