#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=mm_annot
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --array=0-4
#SBATCH --gres=gpu:v100:4,nvme:100

set -euo pipefail

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $SLURM_GPUS_ON_NODE, Partition: $SLURM_JOB_PARTITION"
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

# SMALL MODELS:
LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
BASE_LLM_BATCH_SIZES=(4 4 8 8 8)
BASE_VLM_BATCH_SIZES=(4 4 6 6 6)
MAX_GENERATED_TOKENS=256

# # LARGE MODELS:
# LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# VLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"
# Base batch sizes (per GPU)
# BASE_LLM_BATCH_SIZES=(4 4 8 8 8)
# BASE_VLM_BATCH_SIZES=(4 4 8 8 8)
# MAX_GENERATED_TOKENS=128

# Extract GPU count more simply (format: "gpu:type:count")
NUM_GPUS="${SLURM_GPUS_ON_NODE##*:}"  # Get everything after the last colon

# Validate it's a number, fallback to 1
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
	echo "Warning: Could not parse GPU count from GRES, using 1"
	NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPUs, scaling batch sizes accordingly"

# Scale batch sizes by number of GPUs
LLM_BATCH_SIZES=()
VLM_BATCH_SIZES=()
for i in "${!BASE_LLM_BATCH_SIZES[@]}"; do
	LLM_BATCH_SIZES[$i]=$((BASE_LLM_BATCH_SIZES[i] * NUM_GPUS))
	VLM_BATCH_SIZES[$i]=$((BASE_VLM_BATCH_SIZES[i] * NUM_GPUS))
done

echo "Scaled LLM batch sizes: ${LLM_BATCH_SIZES[@]}"
echo "Scaled VLM batch sizes: ${VLM_BATCH_SIZES[@]}"

DATASET_DIRECTORY="/scratch/project_2004072/ImACCESS/WW_DATASETs"
DATASETS=(
	${DATASET_DIRECTORY}/HISTORY_X4
	${DATASET_DIRECTORY}/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	${DATASET_DIRECTORY}/EUROPEANA_1900-01-01_1970-12-31
	${DATASET_DIRECTORY}/WWII_1939-09-01_1945-09-02
	${DATASET_DIRECTORY}/SMU_1900-01-01_1970-12-31
)
CSV_FILE=${DATASETS[$SLURM_ARRAY_TASK_ID]}/metadata_multi_label.csv

echo "Running Multimodal Annotation on $CSV_FILE using $LLM_MODEL and $VLM_MODEL"

python -u gt_kws_multimodal.py \
  --csv_file $CSV_FILE \
  --num_workers $SLURM_CPUS_PER_TASK \
  --llm_batch_size ${LLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} \
  --vlm_batch_size ${VLM_BATCH_SIZES[$SLURM_ARRAY_TASK_ID]} \
  --llm_model_id $LLM_MODEL \
  --vlm_model_id $VLM_MODEL \
  --max_generated_tks $MAX_GENERATED_TOKENS \
  --max_keywords 5 \
  --verbose \
  # --use_llm_quantization \
  # --use_vlm_quantization \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"