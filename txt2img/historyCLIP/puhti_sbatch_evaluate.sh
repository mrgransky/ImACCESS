#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=preck_dataset_x
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=12G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --array=0-3
#SBATCH --gres=gpu:v100:1

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
echo "$SLURM_SUBMIT_HOST conda env from tykky module..."

DATASETS=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
)

sampling_strategies=("simple_random" "kfold_stratified")

if [ $SLURM_ARRAY_TASK_ID -ge ${#DATASETS[@]} ]; then
	echo "Error: SLURM_ARRAY_TASK_ID out of bounds"
	exit 1
fi

for strategy in "${sampling_strategies[@]}"; do
	for prec in 1 5 10 15 20; do
		echo "Evaluation metrics@K=$prec for Dataset[$SLURM_ARRAY_TASK_ID]: ${DATASETS[$SLURM_ARRAY_TASK_ID]} with sampling strategy: $strategy"
		python -u history_clip_evaluate.py \
			--dataset_dir ${DATASETS[$SLURM_ARRAY_TASK_ID]} \
			--batch_size 512 \
			--model_name "ViT-B/32" \
			--device "cuda:0" \
			--kfold 3 \
			--topK $prec \
			--sampling "$strategy" \

	done
done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"