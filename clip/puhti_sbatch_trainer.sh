#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=cifar10X
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --array=0-1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03-00:00:00

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
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"
echo "$SLURM_SUBMIT_HOST conda env from tykky module..."
MODES=(train finetune)
datasets=(cifar100 cifar10)

for dset in "${datasets[@]}" 
do
  echo "Dataset: $dset : MODES[$SLURM_ARRAY_TASK_ID]: ${MODES[$SLURM_ARRAY_TASK_ID]}"
  # python -u trainer.py \
	# --dataset $dset \
	# --num_epochs 256 \
	# --num_workers 40 \
	# --print_every 100 \
	# --batch_size 256 \
	# --learning_rate 1e-4 \
	# --mode ${MODES[$SLURM_ARRAY_TASK_ID]} \
	# --weight_decay 1e-3 \
	# --model_name "ViT-B/32" \

done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"