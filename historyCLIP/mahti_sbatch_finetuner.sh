#!/bin/bash

#SBATCH --account=project_2014707
#SBATCH --job-name=h4_multi_label_dataset_
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=33G
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1
#SBATCH --array=252
#SBATCH --time=0-00:15:00

set -euo pipefail

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
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

LABEL_TYPE="${LABEL_TYPE:-multi}"

SAMPLINGS=(
  "kfold_stratified" 
  "stratified_random"
)

BASE_DATASET_DIRECTORY=(
  /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
  /scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
  /scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
  /scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
  /scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

SINGLE_LABEL_FILE="metadata_single_label.csv"
MULTI_LABEL_FILE="metadata_multi_label_multimodal.csv"

SINGLE_LABEL_CSVS=()
MULTI_LABEL_CSVS=()
for dir in "${BASE_DATASET_DIRECTORY[@]}"; do
  SINGLE_LABEL_CSVS+=("${dir}/${SINGLE_LABEL_FILE}")
  MULTI_LABEL_CSVS+=("${dir}/${MULTI_LABEL_FILE}")
done
##############################################################################
# 										# H4			# NA				# EU				# WWII			# SMU
FINETUNE_STRATEGIES=(
  "full"              # 00-03, 	# 52-55, 		# 104-107, 	# 156-159, 	# 208-211
  "lora"              # 04-07, 	# 56-59, 		# 108-111, 	# 160-163, 	# 212-215
  "progressive"       # 08-11, 	# 60-63, 		# 112-115, 	# 164-167, 	# 216-219
  "probe"             # 12-15, 	# 64-67, 		# 116-119, 	# 168-171, 	# 220-223
  "lora_plus"         # 16-19, 	# 68-71, 		# 120-123, 	# 172-175, 	# 224-227
  "dora"              # 20-23, 	# 72-75, 		# 124-127, 	# 176-179, 	# 228-231
  "vera"              # 24-27, 	# 76-79, 		# 128-131, 	# 180-183, 	# 232-235
  "ia3"               # 28-31, 	# 80-83, 		# 132-135, 	# 184-187, 	# 236-239
  "clip_adapter_v"    # 32-35, 	# 84-87, 		# 136-139, 	# 188-191, 	# 240-243
  "clip_adapter_t"    # 36-39, 	# 88-91, 		# 140-143, 	# 192-195, 	# 244-247
  "clip_adapter_vt"   # 40-43, 	# 92-95, 		# 144-147, 	# 196-199, 	# 248-251
  "tip_adapter"       # 44-47, 	# 96-99, 		# 148-151, 	# 200-203, 	# 252-255
  "tip_adapter_f"     # 48-51, 	# 100-103, 	# 152-155, 	# 204-207, 	# 256-259
)
##############################################################################

MODEL_ARCHITECTURES=(
  "ViT-L/14@336px"
  "ViT-L/14"
  "ViT-B/32"
  "ViT-B/16"
)

NUM_DATASETS=${#BASE_DATASET_DIRECTORY[@]}
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]}
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]}

dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_ARCHITECTURES)))
remainder=$((SLURM_ARRAY_TASK_ID % (NUM_STRATEGIES * NUM_ARCHITECTURES)))
strategy_index=$((remainder / NUM_ARCHITECTURES))
architecture_index=$((remainder % NUM_ARCHITECTURES))

if [ $dataset_index -ge $NUM_DATASETS ] || \
   [ $strategy_index -ge $NUM_STRATEGIES ] || \
   [ $architecture_index -ge $NUM_ARCHITECTURES ]; then
  echo "Error: Invalid dataset, strategy, or architecture index" >&2
  exit 1
fi

case "$LABEL_TYPE" in
  single) METADATA_CSV="${SINGLE_LABEL_CSVS[$dataset_index]}" ;;
  multi)  METADATA_CSV="${MULTI_LABEL_CSVS[$dataset_index]}" ;;
  *)
    echo "Error: LABEL_TYPE must be 'single' or 'multi', got '$LABEL_TYPE'" >&2
    exit 1
    ;;
esac

INIT_LRS=(5.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
INIT_WDS=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
DROPOUTS=(0.0 0.1 0.05 0.05 0.05)
EPOCHS=(100 100 150 150 150)

LORA_RANKS=(32 64 64 64 64)
LORA_ALPHAS=(64.0 128.0 128.0 128.0 128.0)
LORA_DROPOUTS=(0.15 0.1 0.05 0.05 0.05)

LORA_PLUS_LAMBDAS=(16.0 16.0 16.0 16.0 16.0)

PROBE_DROPOUTS=(0.1 0.1 0.05 0.05 0.05)

MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
TOTAL_NUM_PHASES=(8 4 4 4 4)

BATCH_SIZES=(512 64 64 64 64)
PRINT_FREQUENCIES=(1000 1000 50 50 25)

EARLY_STOPPING_INIT_MIN_EPOCHS=(10 25 17 17 12)
EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

VOLATILITY_THRESHOLDS=(5.0 15.0 15.0 15.0 15.0)
SLOPE_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)
PAIRWISE_IMP_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)

strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
architecture="${MODEL_ARCHITECTURES[$architecture_index]}"

ADAPTER_METHOD=""
if [[ "$strategy" == *"adapter"* ]]; then
  ADAPTER_METHOD="$strategy"
  strategy="adapter"
fi

initial_early_stopping_minimum_epochs="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
case $strategy in
  "full")
    EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 3))
    ;;
  "lora"|"lora_plus"|"dora"|"vera")
    EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 3))
    ;;
  "probe")
    EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 4))
    ;;
  "progressive")
    EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs
    ;;
  "ia3"|"adapter")
    EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 2))
    ;;
esac
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
   [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || \
   [ "$strategy" = "ia3" ] || [ "$strategy" = "probe" ] || \
   [ "$strategy" = "adapter" ]; then
  DROPOUT=0.0
else
  DROPOUT="${DROPOUTS[$dataset_index]}"
fi

case $strategy in
  "full"|"lora"|"lora_plus"|"dora"|"vera")
    ADJUSTED_BATCH_SIZE=48
    ;;
  "progressive")
    case $architecture in
      "ViT-L/14@336px") ADJUSTED_BATCH_SIZE=64 ;;
      "ViT-L/14")       ADJUSTED_BATCH_SIZE=128 ;;
      "ViT-B/32"|"ViT-B/16") ADJUSTED_BATCH_SIZE=256 ;;
    esac
    ;;
  "probe"|"ia3"|"adapter")
    case $architecture in
      "ViT-L/14@336px") ADJUSTED_BATCH_SIZE=256 ;;
      "ViT-L/14")       ADJUSTED_BATCH_SIZE=512 ;;
      "ViT-B/32"|"ViT-B/16") ADJUSTED_BATCH_SIZE=1024 ;;
    esac
    ;;
esac

echo "=== CONFIGURATION ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LABEL_TYPE: $LABEL_TYPE"
echo "DATASET_INDEX: $dataset_index"
echo "CSV_FILE: $METADATA_CSV"
echo "STRATEGY_INDEX: $strategy_index"
echo "FINETUNE_STRATEGY: $strategy"
if [ -n "$ADAPTER_METHOD" ]; then
  echo "ADAPTER_METHOD: $ADAPTER_METHOD"
fi
echo "ARCHITECTURE_INDEX: $architecture_index"
echo "MODEL_ARCHITECTURE: $architecture"
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "INITIAL LEARNING RATE: ${INIT_LRS[$dataset_index]}"
echo "INITIAL WEIGHT DECAY: ${INIT_WDS[$dataset_index]}"
echo "DROPOUT: $DROPOUT"
echo "EARLY_STOPPING_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: $ADJUSTED_BATCH_SIZE"
echo "====================="

echo ">> Starting trainer.py for dataset[$SLURM_ARRAY_TASK_ID]: $METADATA_CSV"

CMD="python -u trainer.py \
  --metadata_csv \"$METADATA_CSV\" \
  --model_architecture \"$architecture\" \
  --mode \"finetune\" \
  --finetune_strategy \"$strategy\" \
  --epochs \"${EPOCHS[$dataset_index]}\" \
  --num_workers \"$SLURM_CPUS_PER_TASK\" \
  --batch_size \"$ADJUSTED_BATCH_SIZE\" \
  --dropout \"$DROPOUT\" \
  --learning_rate \"${INIT_LRS[$dataset_index]}\" \
  --weight_decay \"${INIT_WDS[$dataset_index]}\" \
  --minimum_epochs \"$EARLY_STOPPING_MIN_EPOCHS\" \
  --patience \"${EARLY_STOPPING_PATIENCE[$dataset_index]}\" \
  --minimum_delta \"${EARLY_STOPPING_MIN_DELTA[$dataset_index]}\" \
  --cumulative_delta \"${EARLY_STOPPING_CUMULATIVE_DELTA[$dataset_index]}\" \
  --volatility_threshold \"${VOLATILITY_THRESHOLDS[$dataset_index]}\" \
  --slope_threshold \"${SLOPE_THRESHOLDS[$dataset_index]}\" \
  --pairwise_imp_threshold \"${PAIRWISE_IMP_THRESHOLDS[$dataset_index]}\" \
  --lora_rank \"${LORA_RANKS[$dataset_index]}\" \
  --lora_alpha \"${LORA_ALPHAS[$dataset_index]}\" \
  --lora_dropout \"${LORA_DROPOUTS[$dataset_index]}\" \
  --probe_dropout \"${PROBE_DROPOUTS[$dataset_index]}\" \
  --min_phases_before_stopping \"${MIN_PHASES_BEFORE_STOPPING[$dataset_index]}\" \
  --min_epochs_per_phase \"${MIN_EPOCHS_PER_PHASE[$dataset_index]}\" \
  --total_num_phases \"${TOTAL_NUM_PHASES[$dataset_index]}\" \
  --print_every \"${PRINT_FREQUENCIES[$dataset_index]}\" \
  --sampling \"${SAMPLINGS[1]}\""

if [ "$strategy" = "lora_plus" ]; then
  CMD="$CMD --lora_plus_lambda \"${LORA_PLUS_LAMBDAS[$dataset_index]}\""
fi

if [ -n "$ADAPTER_METHOD" ]; then
  CMD="$CMD --adapter_method \"$ADAPTER_METHOD\""
fi

eval $CMD

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"