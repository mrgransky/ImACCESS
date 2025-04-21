#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=inference
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --time=01-00:00:00
#SBATCH --array=0 # HISTORY_X4
#####SBATCH --array=0-19 # 4 architectures x 5 datasets = 20 jobs
#SBATCH --gres=gpu:v100:1

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

DATASETS=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

# Map dataset directories to their names for checkpoint paths
declare -A DATASET_NAMES
DATASET_NAMES["/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4"]="HISTORY_X4"
DATASET_NAMES["/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31"]="NATIONAL_ARCHIVE_1900-01-01_1970-12-31"
DATASET_NAMES["/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31"]="EUROPEANA_1900-01-01_1970-12-31"
DATASET_NAMES["/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02"]="WWII_1939-09-01_1945-09-02"
DATASET_NAMES["/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31"]="SMU_1900-01-01_1970-12-31"

INIT_LRS=(1.0e-5 1.0e-5 1.0e-5 1.0e-5 1.0e-5)
INIT_WDS=(1.0e-2 1.0e-2 1.0e-2 1.0e-2 1.0e-2)
DROPOUTS=(0.15 0.1 0.05 0.05 0.05)
EPOCHS=(110 100 150 150 150)
LORA_RANKS=(64 64 64 64 64)
LORA_ALPHAS=(128.0 128.0 128.0 128.0 128.0) # 2x rank
LORA_DROPOUTS=(0.05 0.05 0.05 0.05 0.05)
BATCH_SIZES=(64 64 64 64 64)
SAMPLINGS=("kfold_stratified" "stratified_random")
MODEL_ARCHITECTURES=(
	"ViT-L/14@336px"
	"ViT-L/14"
	"ViT-B/32"
	"ViT-B/16"
)

# Calculate indices
dataset_index=$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))
architecture_index=$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))

# Validate indices
if [ $dataset_index -ge ${#DATASETS[@]} ] || [ $architecture_index -ge ${#MODEL_ARCHITECTURES[@]} ]; then
	echo "Error: Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
	exit 1
fi

# Dynamically adjust batch size based on model architecture and dataset
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"
LR="${INIT_LRS[$dataset_index]}"
WD="${INIT_WDS[$dataset_index]}"
dropout="${DROPOUTS[$dataset_index]}"
lora_rank="${LORA_RANKS[$dataset_index]}"
lora_alpha="${LORA_ALPHAS[$dataset_index]}"
lora_dropout="${LORA_DROPOUTS[$dataset_index]}"

# For larger models (ViT-L/14 and ViT-L/14@336px), reduce batch size
if [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"ViT-L"* ]]; then
	if [[ "${DATASETS[$dataset_index]}" == *"HISTORY_X4"* ]]; then
		ADJUSTED_BATCH_SIZE=16  # Very conservative batch size for large model + large dataset
	else
		ADJUSTED_BATCH_SIZE=32  # Reduced batch size for large models with other datasets
	fi
fi
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} ADJUSTED: ${ADJUSTED_BATCH_SIZE}"

# Convert model architecture to filename format (e.g., ViT-B/16 -> ViT-B-16)
model_arch_filename=${MODEL_ARCHITECTURES[$architecture_index]//[@\/]/-}
dataset_dir=${DATASETS[$dataset_index]}
dataset_name=${DATASET_NAMES[$dataset_dir]}
bs="${ADJUSTED_BATCH_SIZE[$dataset_index]}"

# Define checkpoint paths
results_dir="${dataset_dir}/results"
full_checkpoint="${results_dir}/${dataset_name}_full_finetune_CLIP_${model_arch_filename}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_actual_epochs_21_dropout_${dropout}_lr_${LR}_wd_${WD}_bs_${bs}_best_model.pth"
lora_checkpoint="${results_dir}/${dataset_name}_lora_finetune_CLIP_${model_arch_filename}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_actual_epochs_31_lr_${LR}_wd_${WD}_lora_rank_${lora_rank}_lora_alpha_${lora_alpha}_lora_dropout_${lora_dropout}_bs_${bs}_best_model.pth"
progressive_checkpoint="${results_dir}/${dataset_name}_progressive_unfreeze_finetune_CLIP_${model_arch_filename}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_100_dropout_${dropout}_init_lr_${LR}_init_wd_${WD}_bs_${bs}_best_model.pth"


echo "Starting history_clip_inference.py for task $SLURM_ARRAY_TASK_ID"
python -u history_clip_inference.py \
	--dataset_dir "${DATASETS[$dataset_index]}" \
	--num_workers "$SLURM_CPUS_PER_TASK" \
	--batch_size "${ADJUSTED_BATCH_SIZE}" \
	--model_architecture "${MODEL_ARCHITECTURES[$architecture_index]}" \
	--sampling "${SAMPLINGS[1]}" \
	--full_checkpoint "${full_checkpoint}" \
	--lora_checkpoint "${lora_checkpoint}" \
	--progressive_checkpoint "${progressive_checkpoint}" \
	--query_image "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg" \
	--query_label "cemetery" \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"