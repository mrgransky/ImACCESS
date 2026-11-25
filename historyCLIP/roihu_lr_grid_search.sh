#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=lr_grid_42
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=204G
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=1-12:00:00
##SBATCH --array=0-59
#SBATCH --array=34

##############################################################################
# EQUAL-BUDGET LEARNING RATE GRID SEARCH (PILOT CALIBRATION)
# ──────────────────────────────────────────────────────────────────────────
# Runs a fair, shared 6-point log-spaced LR sweep across all 10 trainable
# strategies on ONE representative calibration setting:
#   - Dataset      : HISTORY_X4 (union of NA + EU + WWII + SMU)
#   - Architecture : ViT-L/14@336px
#   - Column       : multimodal_canonical_labels
#   - Seed         : 42
#
# Everything produced by this sweep (checkpoints, viz plots, run_metrics.json,
# summary tables, sensitivity plot) is stored in a COMPLETELY ISOLATED root:
#   ${DATASET_DIR}/lr_grid_search/${ARCH_TAG}/${COLUMN}/seed_${SEED}/
# so production outputs/ and retrieval_metrics_accumulated.json are untouched.
#
# ARRAY INDEXING SCHEME: 10 strategies × 6 LRs = 60 tasks (--array=0-59)
#   strategy_index = SLURM_ARRAY_TASK_ID / 6   (0..9)
#   lr_index       = SLURM_ARRAY_TASK_ID % 6   (0..5)
#
#   strategy 0  probe          :  0 –  5
#   strategy 1  full           :  6 – 11
#   strategy 2  lora           : 12 – 17
#   strategy 3  lora_plus      : 18 – 23
#   strategy 4  rslora         : 24 – 29
#   strategy 5  dora           : 30 – 35
#   strategy 6  vera           : 36 – 41
#   strategy 7  ia3            : 42 – 47
#   strategy 8  tip_adapter_f  : 48 – 53
#   strategy 9  clip_adapter_v : 54 – 59
#
# HOW TO RUN:
#   sbatch roihu_lr_grid_search.sh
#
# AFTER COMPLETION (runs in ~2s on login node):
#   python summarize_lr_grid_search.py \
#     --sweep_dir /scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4/lr_grid_search/ViT-L-14-336px/multimodal_canonical_labels/seed_42
##############################################################################

set -euo pipefail
user="$(whoami)"
stars=$(printf '%*s' 100 '')
txt="$user began LR Grid Search Slurm job: $(date)"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID | JOB: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "NODE: $SLURM_JOB_NODELIST | CPUS: $SLURM_CPUS_PER_TASK | MEM: $SLURM_MEM_PER_NODE"
echo "${stars// /*}"

##############################################################################
# FIXED CALIBRATION CONFIGURATION (overridable via environment variables)
##############################################################################
DATASET_DIR="${DATASET_DIR:-/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4}"
METADATA_CSV="${DATASET_DIR}/metadata_multi_label_multimodal.csv"
ARCHITECTURE="${ARCHITECTURE:-ViT-L/14@336px}"
COLUMN="${COLUMN:-multimodal_canonical_labels}"
SEED="${SEED:-42}"

# Match exact HISTORY_X4 controlled training settings from roihu_finetuner.sh
BATCH_SIZE=16
EPOCHS=100
WEIGHT_DECAY="1.0e-02"
PRINT_EVERY=1000
EARLY_STOPPING_MIN_EPOCHS=15
EARLY_STOPPING_PATIENCE=3
EARLY_STOPPING_MIN_DELTA="1e-4"
EARLY_STOPPING_CUMULATIVE_DELTA="5e-3"
VOLATILITY_THRESHOLD="5.0"
SLOPE_THRESHOLD="1e-4"
PAIRWISE_IMP_THRESHOLD="1e-4"

# Strategy-specific non-LR hyperparameters (identical to roihu_finetuner.sh for H4)
LORA_RANK=16
LORA_ALPHA="32.0"
LORA_DROPOUT="0.0"
RSLORA_ALPHA="8.0"
VERA_RANK=256
LORA_PLUS_LAMBDA="4.0"
ADAPTER_BOTTLENECK_DIM=256
TIP_INITIAL_BETA="5.5"
TIP_INITIAL_ALPHA="1.0"
TIP_SUPPORT_SHOTS=16

##############################################################################
# 2D SEARCH GRID: 10 TRAINABLE STRATEGIES × 6 LOG-SPACED LEARNING RATES
##############################################################################
TRAINABLE_STRATEGIES=(
	"probe"          # 0  : tasks  0– 5
	"full"           # 1  : tasks  6–11
	"lora"           # 2  : tasks 12–17
	"lora_plus"      # 3  : tasks 18–23
	"rslora"         # 4  : tasks 24–29
	"dora"           # 5  : tasks 30–35
	"vera"           # 6  : tasks 36–41
	"ia3"            # 7  : tasks 42–47
	"tip_adapter_f"  # 8  : tasks 48–53
	"clip_adapter_v" # 9  : tasks 54–59
)

LR_GRID=(
	"5.0e-06"        # lr 0
	"2.0e-05"        # lr 1
	"1.0e-04"        # lr 2
	"5.0e-04"        # lr 3
	"1.0e-03"        # lr 4
	"5.0e-03"        # lr 5
)

NUM_STRATEGIES=${#TRAINABLE_STRATEGIES[@]}
NUM_LRS=${#LR_GRID[@]}

strategy_index=$((SLURM_ARRAY_TASK_ID / NUM_LRS))
lr_index=$((SLURM_ARRAY_TASK_ID % NUM_LRS))

if [ "$strategy_index" -ge "$NUM_STRATEGIES" ] || [ "$lr_index" -ge "$NUM_LRS" ]; then
	echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of bounds (expected 0-$((NUM_STRATEGIES * NUM_LRS - 1)))" >&2
	exit 1
fi

strategy_orig="${TRAINABLE_STRATEGIES[$strategy_index]}"
LEARNING_RATE="${LR_GRID[$lr_index]}"

# Remap strategy name to trainer.py CLI interface
strategy="$strategy_orig"
ADAPTER_METHOD=""
BASELINE_METHOD=""

if [[ "$strategy" == *"adapter"* ]]; then
	ADAPTER_METHOD="$strategy"
	strategy="adapter"
elif [ "$strategy" = "probe" ]; then
	BASELINE_METHOD="probe"
	strategy="baseline"
fi

EFFECTIVE_LORA_RANK="$LORA_RANK"
EFFECTIVE_LORA_ALPHA="$LORA_ALPHA"
if [ "$strategy" = "rslora" ]; then
	EFFECTIVE_LORA_ALPHA="$RSLORA_ALPHA"
elif [ "$strategy" = "vera" ]; then
	EFFECTIVE_LORA_RANK="$VERA_RANK"
fi

##############################################################################
# DEDICATED ISOLATED OUTPUT DIRECTORY (ZERO TOUCH ON PRODUCTION outputs/)
##############################################################################
ARCH_TAG=$(echo "$ARCHITECTURE" | sed 's|[/@]|-|g')
SWEEP_ROOT="${DATASET_DIR}/lr_grid_search/${ARCH_TAG}/${COLUMN}/seed_${SEED}"
RUN_OUT_DIR="${SWEEP_ROOT}/${strategy_orig}/lr_${LEARNING_RATE}"
mkdir -p "$RUN_OUT_DIR"

echo "=== LR GRID SEARCH CONFIGURATION ==="
echo "TASK_ID             : $SLURM_ARRAY_TASK_ID ($((strategy_index+1))/$NUM_STRATEGIES × $((lr_index+1))/$NUM_LRS)"
echo "DATASET_DIR         : $DATASET_DIR"
echo "ARCHITECTURE        : $ARCHITECTURE ($ARCH_TAG)"
echo "COLUMN              : $COLUMN"
echo "SEED                : $SEED"
echo "STRATEGY            : $strategy_orig (cli: $strategy)"
[ -n "$ADAPTER_METHOD"  ] && echo "ADAPTER_METHOD      : $ADAPTER_METHOD"
[ -n "$BASELINE_METHOD" ] && echo "BASELINE_METHOD     : $BASELINE_METHOD"
echo "LEARNING_RATE       : $LEARNING_RATE"
echo "WEIGHT_DECAY        : $WEIGHT_DECAY"
echo "BATCH_SIZE          : $BATCH_SIZE"
echo "ISOLATED OUTPUT_DIR : $RUN_OUT_DIR"
echo "===================================="

CMD="python -u trainer.py \
	--metadata_csv              \"$METADATA_CSV\" \
	--model_architecture        \"$ARCHITECTURE\" \
	--strategy                  \"$strategy\" \
	--column                    \"$COLUMN\" \
	--epochs                    \"$EPOCHS\" \
	--num_workers               \"$SLURM_CPUS_PER_TASK\" \
	--batch_size                \"$BATCH_SIZE\" \
	--learning_rate             \"$LEARNING_RATE\" \
	--weight_decay              \"$WEIGHT_DECAY\" \
	--minimum_epochs            \"$EARLY_STOPPING_MIN_EPOCHS\" \
	--patience                  \"$EARLY_STOPPING_PATIENCE\" \
	--minimum_delta             \"$EARLY_STOPPING_MIN_DELTA\" \
	--cumulative_delta          \"$EARLY_STOPPING_CUMULATIVE_DELTA\" \
	--volatility_threshold      \"$VOLATILITY_THRESHOLD\" \
	--slope_threshold           \"$SLOPE_THRESHOLD\" \
	--pairwise_imp_threshold    \"$PAIRWISE_IMP_THRESHOLD\" \
	--print_every               \"$PRINT_EVERY\" \
	--sampling                  \"stratified_random\" \
	--seed                      \"$SEED\" \
	--output_dir                \"$RUN_OUT_DIR\" \
	--verbose"

if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
   [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || \
   [ "$strategy" = "rslora" ]; then
	CMD="$CMD --lora_rank \"$EFFECTIVE_LORA_RANK\""
	CMD="$CMD --lora_alpha \"$EFFECTIVE_LORA_ALPHA\""
	CMD="$CMD --lora_dropout \"$LORA_DROPOUT\""
fi

if [ "$strategy" = "lora_plus" ]; then
	CMD="$CMD --lora_plus_lambda \"$LORA_PLUS_LAMBDA\""
fi

if [ -n "$ADAPTER_METHOD" ]; then
	CMD="$CMD --adapter_method \"$ADAPTER_METHOD\""
	if [[ "$ADAPTER_METHOD" == clip_adapter* ]]; then
		CMD="$CMD --bottleneck_dim \"$ADAPTER_BOTTLENECK_DIM\""
	elif [[ "$ADAPTER_METHOD" == tip_adapter* ]]; then
		CMD="$CMD --initial_beta \"$TIP_INITIAL_BETA\""
		CMD="$CMD --initial_alpha \"$TIP_INITIAL_ALPHA\""
		CMD="$CMD --support_shots \"$TIP_SUPPORT_SHOTS\""
	fi
fi

[ -n "$BASELINE_METHOD" ] && CMD="$CMD --baseline_method \"$BASELINE_METHOD\""

eval $CMD

done_txt="$user finished LR Grid Search task $SLURM_ARRAY_TASK_ID ($strategy_orig @ lr=$LEARNING_RATE): $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
