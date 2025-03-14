#!/bin/bash

# SLURM directives
#SBATCH --account=project_2009043
#SBATCH --job-name=historyCLIP_%A_%a
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=40
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-14  # 3 modes x 5 datasets = 15 tasks
#SBATCH --time=03-00:00:00

set -e
set -u
set -o pipefail

# Define modes and datasets
MODES=("train" "finetune" "pretrain")
DATASETS=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

# Calculate mode and dataset indices from SLURM_ARRAY_TASK_ID
NUM_DATASETS=${#DATASETS[@]}
MODE_IDX=$((SLURM_ARRAY_TASK_ID / NUM_DATASETS))  # Integer division
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % NUM_DATASETS))  # Modulo

# Ensure array task ID is within bounds
if [ $SLURM_ARRAY_TASK_ID -ge $((NUM_DATASETS * ${#MODES[@]})) ]; then
	echo "Error: SLURM_ARRAY_TASK_ID out of bounds"
	exit 1
fi

# Override SLURM_CPUS_PER_TASK (since #SBATCH can't be dynamic)
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 1))  # Reserve 1 CPU for overhead

# Common parameters
INIT_LRS=(5e-4 1e-4 1e-4 5e-4 1e-4)
WEIGHT_DECAYS=(1e-3 1e-3 1e-3 1e-3 1e-2)
DROPOUTS=(0.0 0.0 0.0 0.0 0.0)
EPOCHS=(50 50 150 150 150)
SAMPLINGS=("kfold_stratified" "stratified_random")

# Logging
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
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

# Debugging output
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "MODE: ${MODES[$MODE_IDX]} (Index: $MODE_IDX)"
echo "DATASET: ${DATASETS[$DATASET_IDX]} (Index: $DATASET_IDX)"
echo "EPOCHS: ${EPOCHS[$DATASET_IDX]}"
echo "INIT_LR: ${INIT_LRS[$DATASET_IDX]}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAYS[$DATASET_IDX]}"
echo "DROPOUT: ${DROPOUTS[$DATASET_IDX]}"
echo "NUM_WORKERS: $NUM_WORKERS"

# Run the Python script
python -u history_clip_trainer.py \
	--dataset_dir "${DATASETS[$DATASET_IDX]}" \
	--epochs "${EPOCHS[$DATASET_IDX]}" \
	--num_workers "$NUM_WORKERS" \
	--print_every 250 \
	--batch_size 128 \
	--learning_rate "${INIT_LRS[$DATASET_IDX]}" \
	--weight_decay "${WEIGHT_DECAYS[$DATASET_IDX]}" \
	--mode "${MODES[$MODE_IDX]}" \
	--sampling "${SAMPLINGS[1]}" \
	--dropout "${DROPOUTS[$DATASET_IDX]}" \
	--model_architecture "ViT-B/32"

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"