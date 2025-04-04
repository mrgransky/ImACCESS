#!/bin/bash

#SBATCH --job-name=europeana_dataset_collection
#SBATCH --output=/lustre/sgn-data/ImACCESS/trash/logs/%x_%a_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=9G
#SBATCH --partition=amd
#SBATCH --time=00-10:00:00

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

ddir="/lustre/sgn-data/ImACCESS/WW_DATASETs"
st_dt="1900-01-01"
end_dt="1970-12-31"
num_workers=$((SLURM_CPUS_PER_TASK - 1))

python -u data_collector.py \
	--dataset_dir $ddir \
	--start_date  $st_dt \
	--end_date  $end_dt \
	--num_worker $num_workers \
	--batch_size 256 \
	--historgram_bin 60 \
	--img_mean_std \

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"