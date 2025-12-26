#!/bin/bash

#SBATCH --account=project_2014707
#SBATCH --job-name=dataset_collection
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --array=0-3
#SBATCH --partition=small
#SBATCH --time=0-08:00:00

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
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda env from tykky module..."

DATASET_DIR="/scratch/project_2004072/ImACCESS/WW_DATASETs"

DATASETS=(
	"smu"
	"europeana"
	"national_archive"
	"wwii"
)

CURRENT_DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "Current dataset: $CURRENT_DATASET"

cd "$CURRENT_DATASET"
echo "Changed to directory: $PWD"

# Build the python command with conditional API key
PYTHON_CMD="python -u data_collector.py \
--dataset_dir $DATASET_DIR \
--num_workers $SLURM_CPUS_PER_TASK \
--batch_size 256 \
--historgram_bin 60 \
--img_mean_std \
--thumbnail_size 800,800 \
--verbose"

# Add API key only for Europeana dataset
if [ "$CURRENT_DATASET" = "europeana" ]; then
	PYTHON_CMD="$PYTHON_CMD --api_key nLbaXYaiH"
	echo "Adding Europeana API key"
fi

echo "Executing: $PYTHON_CMD"
eval $PYTHON_CMD

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"