#!/bin/bash
#SBATCH --account=project_2004072
#SBATCH --job-name=test
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=test
#SBATCH --time=00-00:15:00

set -euo pipefail

user=$(whoami)
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: $(date)"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME"
echo "Using $SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "Node: $SLURM_NODELIST"
echo "GPU(s): ${SLURM_GPUS_ON_NODE:-0}, Partition: $SLURM_JOB_PARTITION"
echo "TMPDIR (local storage): $TMPDIR"
echo "${stars// /*}"

# Show storage info
echo "Local storage information:"
df -h "$TMPDIR"
du -sh "$TMPDIR" 2>/dev/null || echo "TMPDIR empty or inaccessible"

echo "${stars// /*}"

# Set cache locations to TMPDIR (fast local storage)
export HF_HOME="$TMPDIR/hf_cache"
export TRANSFORMERS_CACHE="$TMPDIR/hf_cache"
export HF_DATASETS_CACHE="$TMPDIR/hf_cache"

echo "Cache locations set:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "${stars// /*}"

# Your Python script here
# python your_script.py --args