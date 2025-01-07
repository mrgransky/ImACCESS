#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=cifar10X
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=51G
#SBATCH --partition=gpu
#SBATCH --array=0
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
datasets=(cifar100 cifar10)
MODES=(train finetune)
INIT_LRS=(1e-5 5e-3)
num_workers=$((SLURM_CPUS_PER_TASK - 1))  # reserve 1 CPU for the main process and other overheads

for dset in "${datasets[@]}" 
	do
		echo "Dataset: $dset MODES[$SLURM_ARRAY_TASK_ID]: ${MODES[$SLURM_ARRAY_TASK_ID]} with $num_workers workers"
		python -u trainer.py \
		--dataset $dset \
		--num_epochs 256 \
		--num_workers $num_workers \
		--print_every 100 \
		--batch_size 256 \
		--learning_rate ${INIT_LRS[$SLURM_ARRAY_TASK_ID]} \
		--weight_decay 1e-3 \
		--mode ${MODES[$SLURM_ARRAY_TASK_ID]} \
		--model_name "ViT-B/32" \

	done
done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"