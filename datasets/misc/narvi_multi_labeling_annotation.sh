#!/bin/bash

#SBATCH --job-name=multi_label_annotation_textual_visual
#SBATCH --output=/lustre/sgn-data/ImACCESS/trash/logs/%x_%a_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --time=07-00:00:00

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE MB"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"

echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME) conda env from Anaconda..."

source activate py39

DATASETS=(
	/lustre/sgn-data/ImACCESS/WW_DATASETs/HISTORY_X4
	/lustre/sgn-data/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/lustre/sgn-data/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/lustre/sgn-data/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/lustre/sgn-data/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

python -u multi_label_annotation.py \
  --csv_file ${DATASETS[$SLURM_ARRAY_TASK_ID]}/metadata.csv \
  --num_workers $SLURM_CPUS_PER_TASK \
  --text_batch_size 1024 \
  --vision_batch_size 512 \
  --relevance_threshold 0.25 \
  --vision_threshold 0.20 \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"