#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=quantitative_qualitative_evaluation_historyCLIP_dataset_x
#####SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a.out # only for testing different img & lbl
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --time=0-01:00:00
#SBATCH --array=0,5,10,15 # History_X4
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

INIT_LRS=(1.0e-05 1.0e-05 1.0e-05 1.0e-05 1.0e-05)
INIT_WDS=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
DROPOUTS=(0.1 0.1 0.05 0.05 0.05)
EPOCHS=(100 100 150 150 150)
LORA_RANKS=(64 64 64 64 64)
LORA_ALPHAS=(128.0 128.0 128.0 128.0 128.0)
LORA_DROPOUTS=(0.05 0.05 0.05 0.05 0.05)
BATCH_SIZES=(64 64 64 64 64)
SAMPLINGS=("kfold_stratified" "stratified_random")
MODEL_ARCHITECTURES=(
	"ViT-L/14@336px" 	# 0-4
	"ViT-L/14" 				# 5-9
	"ViT-B/32" 				# 10-14
	"ViT-B/16" 				# 15-19
)

# Hardcoded actual epochs for HISTORY_X4 per architecture
declare -A FULL_EPOCHS
FULL_EPOCHS["ViT-L-14-336px"]="20"
FULL_EPOCHS["ViT-L-14"]="20"
FULL_EPOCHS["ViT-B-32"]="21"
FULL_EPOCHS["ViT-B-16"]="21"

declare -A LORA_EPOCHS
LORA_EPOCHS["ViT-L-14-336px"]="26"
LORA_EPOCHS["ViT-L-14"]="26"
LORA_EPOCHS["ViT-B-32"]="27"
LORA_EPOCHS["ViT-B-16"]="27"

declare -A PROGRESSIVE_EPOCHS
PROGRESSIVE_EPOCHS["ViT-L-14-336px"]="100"
PROGRESSIVE_EPOCHS["ViT-L-14"]="100"
PROGRESSIVE_EPOCHS["ViT-B-32"]="100"
PROGRESSIVE_EPOCHS["ViT-B-16"]="100"

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
DROPOUT="${DROPOUTS[$dataset_index]}"
LORA_RANK="${LORA_RANKS[$dataset_index]}"
LORA_ALPHA="${LORA_ALPHAS[$dataset_index]}"
LORA_DROPOUT="${LORA_DROPOUTS[$dataset_index]}"
INIT_EPOCHS="${EPOCHS[$dataset_index]}"

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

# Get actual epochs for HISTORY_X4 or default for other datasets
if [[ "${dataset_name}" == "HISTORY_X4" ]]; then
		FULL_ACTUAL_EPOCHS=${FULL_EPOCHS[$model_arch_filename]}
		LORA_ACTUAL_EPOCHS=${LORA_EPOCHS[$model_arch_filename]}
		PROGRESSIVE_ACTUAL_EPOCHS=${PROGRESSIVE_EPOCHS[$model_arch_filename]}
else
		# Default epochs for other datasets
		FULL_ACTUAL_EPOCHS=21
		LORA_ACTUAL_EPOCHS=27
		PROGRESSIVE_ACTUAL_EPOCHS=100
fi

# Define checkpoint filenames dynamically
full_checkpoints=(
		"${dataset_name}_full_finetune_CLIP_${MODEL_ARCHITECTURES[0]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_20_dropout_${DROPOUT}_lr_${LR}_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_full_finetune_CLIP_${MODEL_ARCHITECTURES[1]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_20_dropout_${DROPOUT}_lr_${LR}_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_full_finetune_CLIP_${MODEL_ARCHITECTURES[2]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_21_dropout_${DROPOUT}_lr_${LR}_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_full_finetune_CLIP_${MODEL_ARCHITECTURES[3]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_21_dropout_${DROPOUT}_lr_${LR}_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
)
lora_checkpoints=(
		"${dataset_name}_lora_finetune_CLIP_${MODEL_ARCHITECTURES[0]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_26_lr_${LR}_wd_${WD}_lora_rank_${LORA_RANK}_lora_alpha_${LORA_ALPHA}_lora_dropout_${LORA_DROPOUT}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_lora_finetune_CLIP_${MODEL_ARCHITECTURES[1]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_26_lr_${LR}_wd_${WD}_lora_rank_${LORA_RANK}_lora_alpha_${LORA_ALPHA}_lora_dropout_${LORA_DROPOUT}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_lora_finetune_CLIP_${MODEL_ARCHITECTURES[2]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_27_lr_${LR}_wd_${WD}_lora_rank_${LORA_RANK}_lora_alpha_${LORA_ALPHA}_lora_dropout_${LORA_DROPOUT}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_lora_finetune_CLIP_${MODEL_ARCHITECTURES[3]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_actual_epochs_27_lr_${LR}_wd_${WD}_lora_rank_${LORA_RANK}_lora_alpha_${LORA_ALPHA}_lora_dropout_${LORA_DROPOUT}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
)
progressive_checkpoints=(
		"${dataset_name}_progressive_unfreeze_finetune_CLIP_${MODEL_ARCHITECTURES[0]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_dropout_${DROPOUT}_init_lr_${LR}_init_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_progressive_unfreeze_finetune_CLIP_${MODEL_ARCHITECTURES[1]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_dropout_${DROPOUT}_init_lr_${LR}_init_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_progressive_unfreeze_finetune_CLIP_${MODEL_ARCHITECTURES[2]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_dropout_${DROPOUT}_init_lr_${LR}_init_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
		"${dataset_name}_progressive_unfreeze_finetune_CLIP_${MODEL_ARCHITECTURES[3]//[@\/]/-}_AdamW_OneCycleLR_CrossEntropyLoss_GradScaler_init_epochs_${INIT_EPOCHS}_dropout_${DROPOUT}_init_lr_${LR}_init_wd_${WD}_bs_${ADJUSTED_BATCH_SIZE}_best_model.pth"
)

# Define checkpoint paths
results_dir="${dataset_dir}/results"
full_checkpoint="${results_dir}/${full_checkpoints[$architecture_index]}"
lora_checkpoint="${results_dir}/${lora_checkpoints[$architecture_index]}"
progressive_checkpoint="${results_dir}/${progressive_checkpoints[$architecture_index]}"

# Check if checkpoints exist
for checkpoint in "$full_checkpoint" "$lora_checkpoint" "$progressive_checkpoint"; do
		if [ ! -f "$checkpoint" ]; then
				echo "Warning: Checkpoint $checkpoint does not exist"
		fi
done

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
		--lora_rank "${LORA_RANK}" \
		--lora_alpha "${LORA_ALPHA}" \
		--lora_dropout "${LORA_DROPOUT}" \
		--topK 3 \
		# --query_image "https://pbs.twimg.com/media/GowwFwkbQAAaMs-?format=jpg" \ # validaton set
		# --query_label "cemetery" \ # validation set

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"