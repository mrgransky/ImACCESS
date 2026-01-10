#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=chunked_mm_annot
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --array=0-21
#SBATCH --gres=gpu:v100:4,nvme:100

set -euo pipefail

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "Local scratch: $LOCAL_SCRATCH"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "GPU(s): $SLURM_GPUS_ON_NODE, Partition: $SLURM_JOB_PARTITION"
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

# Determine number of GPUs from GRES allocation
GPU_GRES="${SLURM_GPUS_ON_NODE%%,*}"  # Extract "gpu:v100:4" part
NUM_GPUS="${GPU_GRES##*:}"        		# Extract "4" from gpu part

# Validate GPU count
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
	echo "Warning: Could not parse GPU count from '$SLURM_GPUS_ON_NODE', defaulting to 1"
	NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPU(s) for job: $SLURM_JOB_ID with array task: $SLURM_ARRAY_TASK_ID"

# Select model configuration based on GPU count
if [ "$NUM_GPUS" -gt 1 ]; then
	echo "LARGE models (multi-GPU configuration)"
	LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
	LLM_BATCH_SIZE=16
	LLM_MAX_GEN_TKs=128
	# LLM_QUANTIZATION="--use_llm_quantization"  # Enable
	VLM_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
	VLM_BATCH_SIZE=12
	VLM_MAX_GEN_TKs=64
	# VLM_QUANTIZATION="--use_vlm_quantization"  # Enable
else
	echo "SMALL models (single-GPU configuration)"
	LLM_MODEL="Qwen/Qwen3-4B-Instruct-2507"
	LLM_BATCH_SIZE=32
	LLM_MAX_GEN_TKs=256
	# LLM_QUANTIZATION="--use_llm_quantization"  # Enable
	VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
	VLM_BATCH_SIZE=24
	VLM_MAX_GEN_TKs=64
	# VLM_QUANTIZATION="--use_vlm_quantization"  # Enable
fi

echo "LLM Model: $LLM_MODEL (batch size: $LLM_BATCH_SIZE) max generated tokens: $LLM_MAX_GEN_TKs"
echo "VLM Model: $VLM_MODEL (batch size: $VLM_BATCH_SIZE) max generated tokens: $VLM_MAX_GEN_TKs"

DATASET_DIRECTORY="/scratch/project_2004072/ImACCESS/WW_DATASETs"
CSV_FILE=${DATASET_DIRECTORY}/HISTORY_X4/metadata_multi_label_chunk_$SLURM_ARRAY_TASK_ID.csv
echo "Running (chunked) Multimodal Annotation on $CSV_FILE"

python -u gt_kws_multimodal.py \
	--csv_file $CSV_FILE \
	--llm_model_id $LLM_MODEL \
	--llm_batch_size $LLM_BATCH_SIZE \
	--llm_max_generated_tks $LLM_MAX_GEN_TKs \
	--vlm_model_id $VLM_MODEL \
	--vlm_batch_size $VLM_BATCH_SIZE \
	--vlm_max_generated_tks $VLM_MAX_GEN_TKs \
	--num_workers $SLURM_CPUS_PER_TASK \
	--max_keywords 5 \
	--verbose \
	# $LLM_QUANTIZATION \
	# $VLM_QUANTIZATION \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"