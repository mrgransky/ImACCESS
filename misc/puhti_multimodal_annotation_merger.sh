#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=mm_annot_merge
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --gres=gpu:v100:1,nvme:100

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

DATASET_DIR="/scratch/project_2004072/ImACCESS/_WW_DATASETs/HISTORY_X4"

TEXT_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"

python -u gt_kws_multimodal_merge.py \
	--dataset_dir $DATASET_DIR \
	--num_workers $SLURM_CPUS_PER_TASK \
	--model_id $TEXT_EMBEDDING_MODEL \
	--verbose

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"