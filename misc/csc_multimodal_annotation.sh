#!/bin/bash

#SBATCH --account=project_2014707
#SBATCH --job-name=multimodal_annotation
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72  # 72 per GH200, 288 for 4x
#SBATCH --mem=0             # whole node on GH200, don't request 64G
#SBATCH --partition=gpularge
#SBATCH --time=01-12:00:00
#SBATCH --gres=gpu:gh200:4
#SBATCH --array=0

set -euo pipefail

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $SLURM_GPUS_ON_NODE Partition: $SLURM_JOB_PARTITION"
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

# Extract GPU count more simply (format: "gpu:type:count")
NUM_GPUS="${SLURM_GPUS_ON_NODE##*:}"  # Get everything after the last colon

# Validate it's a number, fallback to 1
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
	echo "Warning: Could not parse GPU count from GRES, using 1"
	NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPUs, selecting model configuration"

# Select model configuration based on GPU count
if [ "$NUM_GPUS" -gt 1 ]; then
	echo "LARGE models (multi-GPU configuration)"
	MLM_MODEL="Qwen/Qwen3.6-35B-A3B"
	LLM_MAX_GENERATED_TOKENS=128
	VLM_MAX_GENERATED_TOKENS=96
else
	echo "SMALL models (single-GPU configuration)"
	MLM_MODEL="Qwen/Qwen3.5-9B"
	LLM_MAX_GENERATED_TOKENS=256
	VLM_MAX_GENERATED_TOKENS=96
fi

DATASET_DIRECTORY="/scratch/project_2004072/ImACCESS/WW_DATASETs"
DATASETS=(
	${DATASET_DIRECTORY}/HISTORY_X4
	${DATASET_DIRECTORY}/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	${DATASET_DIRECTORY}/EUROPEANA_1900-01-01_1970-12-31
	${DATASET_DIRECTORY}/WWII_1939-09-01_1945-09-02
	${DATASET_DIRECTORY}/SMU_1900-01-01_1970-12-31
)
CSV_FILE=${DATASETS[$SLURM_ARRAY_TASK_ID]}/metadata_multi_label.csv
LLM_BATCH_SIZES=(20 6 18 18 28)
VLM_BATCH_SIZES=(24 8 8 8 8)

echo "Running Multimodal Annotation on $CSV_FILE using $MLM_MODEL"
echo "[LLM] batch sizes: ${LLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} max generated tokens: $LLM_MAX_GENERATED_TOKENS"
echo "[VLM] batch sizes: ${VLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} max generated tokens: $VLM_MAX_GENERATED_TOKENS"

python -u gt_kws_multimodal.py \
	--csv_file $CSV_FILE \
	--num_workers $SLURM_CPUS_PER_TASK \
	--llm_model_id $MLM_MODEL \
	--llm_batch_size ${LLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} \
	--llm_max_generated_tks $LLM_MAX_GENERATED_TOKENS \
	--vlm_model_id $MLM_MODEL \
	--vlm_batch_size ${VLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} \
	--vlm_max_generated_tks $VLM_MAX_GENERATED_TOKENS \
	--max_keywords 3 \
	--verbose \
	# --llm_use_quantization \
	# --vlm_use_quantization \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"
