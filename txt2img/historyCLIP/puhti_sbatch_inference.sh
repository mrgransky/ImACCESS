#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=inference
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --array=0 # History X4
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

BATCH_SIZES=(64 64 64 64 64)
SAMPLINGS=("kfold_stratified" "stratified_random")
MODEL_ARCHITECTURES=(
	"ViT-L/14@336px"
	"ViT-L/14"
	"ViT-B/32"
	"ViT-B/16"
)

if [ $SLURM_ARRAY_TASK_ID -ge ${#DATASETS[@]} ]; then
	echo "Error: SLURM_ARRAY_TASK_ID out of bounds"
	exit 1
fi

# Dynamically adjust batch size based on model architecture and dataset
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"

# For larger models (ViT-L/14 and ViT-L/14@336px), reduce batch size
if [[ "${MODEL_ARCHITECTURES[$architecture_index]}" == *"ViT-L"* ]]; then
		# Further reduce batch size for HISTORY_X4 dataset due to its size
		if [[ "${DATASETS[$dataset_index]}" == *"HISTORY_X4"* ]]; then
				ADJUSTED_BATCH_SIZE=16  # Very conservative batch size for large model + large dataset
		else
				ADJUSTED_BATCH_SIZE=32 # Reduced batch size for large models with other datasets
		fi
fi
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} ADJUSTED: ${ADJUSTED_BATCH_SIZE}"

echo "Starting history_clip_inference.py for task $SLURM_ARRAY_TASK_ID"
python -u history_clip_inference.py \
	--dataset_dir ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
	--num_workers "$SLURM_CPUS_PER_TASK" \
	--batch_size "${ADJUSTED_BATCH_SIZE}" \
	--model_architecture "${MODEL_ARCHITECTURES[$architecture_index]}"\
	--topK $prec \
	--sampling "${SAMPLINGS[1]}" \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"