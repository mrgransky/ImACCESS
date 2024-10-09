#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=NA_dataset_collection
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=longrun
#SBATCH --time=14-00:00:00

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
files=(/scratch/project_2004072/Nationalbiblioteket/NLF_DATASET/*.gz) # newly scraped NLF data (17.04.2024)
ddir="/scratch/project_2004072/Nationalbiblioteket/dataframes_x732" # x58: 30-80 & 725-731
# maxNumFeatures=$(awk -v x="1.9e+6" 'BEGIN {printf("%d\n",x)}') # adjust values 2.2e+6
maxNumFeatures=-1
LEMMATIZER="stanza"
# LEMMATIZER="trankit"

echo "Query[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"

# for mx in 1.0 0.9 0.8 0.7 0.6 0.5
for mx in 1.0
do
	# for mn in 10 15 30 # 1 3 5 already done (only for max 1.0)
	for mn in 1
	do
		# ddir="/lustre/sgn-data/Nationalbiblioteket/dfXY_${mx}_max_df_${mn}_min_df" #### must be adjusted ####
		echo "max doc_freq $mx | min doc_freq $mn | maxNumFeat: $maxNumFeatures | outDIR $ddir"
		python -u user_token.py \
						--inputDF ${files[$SLURM_ARRAY_TASK_ID]} \
						--outDIR $ddir \
						--lmMethod $LEMMATIZER \
						--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
						--maxDocFreq $mx \
						--minDocFreq $mn \
						--maxNumFeat $maxNumFeatures \

	done
done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"