#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=finetune_historyCLIP_dataset_x
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-4
#SBATCH --time=03-00:00:00

set -e
set -u
set -o pipefail

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
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"
NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 1))  # reserve 1 CPU for the main process and other overheads
INIT_LRS=(1e-5 1e-5 1e-5 1e-5 1e-5)
WEIGHT_DECAYS=(1e-2 1e-2 1e-2 1e-2 1e-2)
DROPOUTS=(0.0 0.0 0.0 0.0 0.0)
EPOCHS=(60 60 150 150 150)
MODES=(train finetune pretrain)
SAMPLINGS=("kfold_stratified" "stratified_random")
DATASETS=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

if [ $SLURM_ARRAY_TASK_ID -ge ${#DATASETS[@]} ]; then
	echo "Error: SLURM_ARRAY_TASK_ID out of bounds"
	exit 1
fi
# Debugging output
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "DATASET: ${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "EPOCHS: ${EPOCHS[$SLURM_ARRAY_TASK_ID]}"
echo "INIT_LR: ${INIT_LRS[$SLURM_ARRAY_TASK_ID]}"
echo "WEIGHT_DECAY: ${WEIGHT_DECAYS[$SLURM_ARRAY_TASK_ID]}"
echo "DROPOUT: ${DROPOUTS[$SLURM_ARRAY_TASK_ID]}"

python -u history_clip_trainer.py \
	--dataset_dir ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
	--epochs ${EPOCHS[$SLURM_ARRAY_TASK_ID]} \
	--num_workers $NUM_WORKERS \
	--print_every 250 \
	--batch_size 128 \
	--learning_rate ${INIT_LRS[$SLURM_ARRAY_TASK_ID]} \
	--weight_decay ${WEIGHT_DECAYS[$SLURM_ARRAY_TASK_ID]} \
	--mode ${MODES[1]} \
	--sampling ${SAMPLINGS[1]} \
	--dropout ${DROPOUTS[$SLURM_ARRAY_TASK_ID]} \
	--model_architecture "ViT-B/32" \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"