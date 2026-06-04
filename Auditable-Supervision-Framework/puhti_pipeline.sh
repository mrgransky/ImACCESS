#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=H4_mlm
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=192G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --gres=gpu:v100:4,nvme:100
####SBATCH --begin=04:00:00

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

DATASET_DIR="/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4"
# DATASET_DIR="/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31"
CSV_FILE=$DATASET_DIR/metadata_multi_label.csv
JSONL_COT_FILE="${CSV_FILE%.csv}_mlm_cot.jsonl"
JSONL_MODALITY_CONFLICT_FILE="${JSONL_COT_FILE%.jsonl}_modality_conflict_audit.jsonl"

VLM_MODEL="Qwen/Qwen3.6-35B-A3B"
SYMMETRICAL_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
# SYMMETRICAL_EMBEDDING_MODEL="nvidia/llama-embed-nemotron-8b"
# SYMMETRICAL_EMBEDDING_MODEL="Octen/Octen-Embedding-8B"
ASYMMETRICAL_EMBEDDING_MODEL="cross-encoder/nli-deberta-v3-large"

# stage 1: VLM with CoT:
python -u stage1_mlm_cot.py -csv $CSV_FILE -vlm $VLM_MODEL -bs 28 -mgt 128 -v

# stage 2: Modality Conflict Quantification
python -u stage2_modality_conflict.py -jsonl $JSONL_COT_FILE -sym $SYMMETRICAL_EMBEDDING_MODEL -asym $ASYMMETRICAL_EMBEDDING_MODEL -v

# Bridge Global Aggregation:
python -u bridge_global_aggregation_ontology_builder.py -jsonl $JSONL_MODALITY_CONFLICT_FILE -m $SYMMETRICAL_EMBEDDING_MODEL -v

# stage 3 & 4: Regime-Aware Consolidation
python -u stage3_4_cgd_consolidation.py -jsonl $JSONL_MODALITY_CONFLICT_FILE -v


done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"