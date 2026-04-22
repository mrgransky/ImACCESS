#!/bin/bash

#SBATCH --account=project_2014707
#SBATCH --job-name=ft_h4_multi_label
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03-00:00:00
##############################################################################
# ARRAY INDEXING SCHEME
# ──────────────────────────────────────────────────────────────────────────
# Dimension order:  dataset × architecture × strategy × column
# Total jobs:       5 × 4 × 14 × 3 = 840
#
# Block sizes (how many IDs each "outer" dimension occupies):
#   Per-dataset block      = 4 × 14 × 3 = 168  IDs
#   Per-architecture block =     14 × 3 =  42  IDs
#   Per-strategy block     =          3 =   3  IDs
#
# ── Dataset blocks ──────────────────────────────────────────────────────
#   dataset 0  HISTORY_X4                        :   0 – 167
#   dataset 1  NATIONAL_ARCHIVE_1900-01-01_...   : 168 – 335
#   dataset 2  EUROPEANA_1900-01-01_...          : 336 – 503
#   dataset 3  WWII_1939-09-01_...               : 504 – 671
#   dataset 4  SMU_1900-01-01_...                : 672 – 839
#
# ── Architecture blocks (within each dataset block) ─────────────────────
#   arch 0  ViT-L/14@336px  : dataset_offset +  0 – 41
#   arch 1  ViT-L/14        : dataset_offset + 42 – 83
#   arch 2  ViT-B/32        : dataset_offset + 84 – 125
#   arch 3  ViT-B/16        : dataset_offset + 126 – 167
#
# ── Strategy blocks (within each architecture block) ────────────────────
#   strategy  0  zero_shot       : arch_offset +  0 –  2
#   strategy  1  probe           : arch_offset +  3 –  5
#   strategy  2  full            : arch_offset +  6 –  8
#   strategy  3  lora            : arch_offset +  9 – 11
#   strategy  4  lora_plus       : arch_offset + 12 – 14
#   strategy  5  rslora          : arch_offset + 15 – 17
#   strategy  6  dora            : arch_offset + 18 – 20
#   strategy  7  vera            : arch_offset + 21 – 23
#   strategy  8  ia3             : arch_offset + 24 – 26
#   strategy  9  tip_adapter_f   : arch_offset + 27 – 29
#   strategy 10  clip_adapter_v  : arch_offset + 30 – 32
#   strategy 11  clip_adapter_vt : arch_offset + 33 – 35
#   strategy 12  clip_adapter_t  : arch_offset + 36 – 38
#   strategy 13  tip_adapter     : arch_offset + 39 – 41
#
# ── Column offsets (within each strategy block of 3) ────────────────────
#   +0  llm_canonical_labels
#   +1  vlm_canonical_labels
#   +2  multimodal_canonical_labels
#
# ── COMMON SUBSET EXAMPLES ──────────────────────────────────────────────
#   All 840 jobs                                    : --array=0-839
#   H4 only (all archs, strategies, cols)           : --array=0-167
#   H4 + ViT-L/14@336px (all strategies, all cols) : --array=0-41
#   H4 + ViT-L/14@336px + first 11 strats, all cols: --array=0-32   ✅
#   H4 + ViT-L/14@336px + all strats + llm only    : --array=0-41:3
#   H4 + ViT-L/14@336px + all strats + vlm only    : --array=1-41:3
#   H4 + ViT-L/14@336px + all strats + multimodal  : --array=2-41:3
#   H4 + ViT-L/14@336px + lora_plus + all cols     : --array=12-14
#   H4 + ViT-L/14@336px + lora_plus + multimodal   : --array=14
##############################################################################
#SBATCH --array=0-32

set -euo pipefail

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

##############################################################################
# LABEL TYPE
# Can be overridden at submission time via:
#   LABEL_TYPE=single sbatch mahti_sbatch_finetuner.sh
# Defaults to "multi" if not set.
##############################################################################
LABEL_TYPE="${LABEL_TYPE:-multi}"

##############################################################################
# SAMPLING STRATEGIES
#   Index 0: kfold_stratified   — used for cross-validation runs
#   Index 1: stratified_random  — used for standard train/val/test splits
##############################################################################
SAMPLINGS=(
	"kfold_stratified"
	"stratified_random"
)

##############################################################################
# DATASETS
# Five historical image datasets, each with a corresponding single-label
# and multi-label metadata CSV file.
##############################################################################
BASE_DATASET_DIRECTORY=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4                       # dataset 0
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31  # dataset 1
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31         # dataset 2
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02             # dataset 3
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31              # dataset 4
)

SINGLE_LABEL_FILE="metadata_single_label.csv"
MULTI_LABEL_FILE="metadata_multi_label_multimodal.csv"

# Build full CSV paths for each dataset
SINGLE_LABEL_CSVS=()
MULTI_LABEL_CSVS=()
for dir in "${BASE_DATASET_DIRECTORY[@]}"; do
	SINGLE_LABEL_CSVS+=("${dir}/${SINGLE_LABEL_FILE}")
	MULTI_LABEL_CSVS+=("${dir}/${MULTI_LABEL_FILE}")
done

##############################################################################
# FINE-TUNING STRATEGIES
# ──────────────────────────────────────────────────────────────────────────
# Strategies are grouped by type:
#   Baselines  : zero_shot, probe
#   Full FT    : full
#   LoRA family: lora, lora_plus, rslora, dora, vera, ia3
#   Adapters   : tip_adapter_f, clip_adapter_v, clip_adapter_vt,
#                clip_adapter_t, tip_adapter
#
# Adapter strategies are detected at runtime via the "*adapter*" pattern
# and remapped to strategy="adapter" + ADAPTER_METHOD=<original_name>.
#
# Baseline strategies (zero_shot, probe) are remapped to
# strategy="baseline" + BASELINE_METHOD=<original_name>.
##############################################################################
FINETUNE_STRATEGIES=(
	"zero_shot"        # strategy  0  │ baseline group
	"probe"            # strategy  1  │ baseline group
	"full"             # strategy  2  │ full fine-tuning
	"lora"             # strategy  3  │ LoRA family
	"lora_plus"        # strategy  4  │ LoRA family (requires --lora_plus_lambda)
	"rslora"           # strategy  5  │ LoRA family
	"dora"             # strategy  6  │ LoRA family
	"vera"             # strategy  7  │ LoRA family
	"ia3"              # strategy  8  │ LoRA family
	"tip_adapter_f"    # strategy  9  │ adapter group → adapter_method=tip_adapter_f
	"clip_adapter_v"   # strategy 10  │ adapter group → adapter_method=clip_adapter_v
	"clip_adapter_vt"  # strategy 11  │ adapter group → adapter_method=clip_adapter_vt
	"clip_adapter_t"   # strategy 12  │ adapter group → adapter_method=clip_adapter_t
	"tip_adapter"      # strategy 13  │ adapter group → adapter_method=tip_adapter
)

##############################################################################
# MODEL ARCHITECTURES
##############################################################################
MODEL_ARCHITECTURES=(
	"ViT-L/14@336px"   # arch 0  — largest, highest resolution
	"ViT-L/14"         # arch 1
	"ViT-B/32"         # arch 2
	"ViT-B/16"         # arch 3
)

##############################################################################
# LABEL COLUMNS
# These are the three canonical label sources produced by different models.
##############################################################################
COLUMNS=(
	"llm_canonical_labels"         # col 0  — labels from a large language model
	"vlm_canonical_labels"         # col 1  — labels from a vision-language model
	"multimodal_canonical_labels"  # col 2  — labels fused from both modalities
)

##############################################################################
# DIMENSION SIZES (derived automatically — do not hardcode)
##############################################################################
NUM_DATASETS=${#BASE_DATASET_DIRECTORY[@]}    # 5
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]}     # 14
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]}  # 4
NUM_COLUMNS=${#COLUMNS[@]}                    # 3

##############################################################################
# INDEX DECODING
# ──────────────────────────────────────────────────────────────────────────
# Dimension order: dataset × architecture × strategy × column
#
# Formula:
#   dataset_index      = ID  ÷ (NUM_ARCH × NUM_STRAT × NUM_COL)
#   architecture_index = (ID % (NUM_ARCH × NUM_STRAT × NUM_COL)) ÷ (NUM_STRAT × NUM_COL)
#   strategy_index     = (ID % (NUM_STRAT × NUM_COL)) ÷ NUM_COL
#   column_index       = ID % NUM_COL
##############################################################################
dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_ARCHITECTURES * NUM_STRATEGIES * NUM_COLUMNS)))
remainder=$((SLURM_ARRAY_TASK_ID % (NUM_ARCHITECTURES * NUM_STRATEGIES * NUM_COLUMNS)))
architecture_index=$((remainder / (NUM_STRATEGIES * NUM_COLUMNS)))
remainder2=$((remainder % (NUM_STRATEGIES * NUM_COLUMNS)))
strategy_index=$((remainder2 / NUM_COLUMNS))
column_index=$((remainder2 % NUM_COLUMNS))

# Bounds check — catches misconfigured --array ranges immediately
if [ $dataset_index      -ge $NUM_DATASETS      ] || \
   [ $architecture_index -ge $NUM_ARCHITECTURES ] || \
   [ $strategy_index     -ge $NUM_STRATEGIES    ] || \
   [ $column_index       -ge $NUM_COLUMNS       ]; then
	echo "Error: Index out of bounds for TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
	echo "  dataset_index=$dataset_index      (valid: 0–$((NUM_DATASETS-1)))"      >&2
	echo "  architecture_index=$architecture_index (valid: 0–$((NUM_ARCHITECTURES-1)))" >&2
	echo "  strategy_index=$strategy_index    (valid: 0–$((NUM_STRATEGIES-1)))"    >&2
	echo "  column_index=$column_index        (valid: 0–$((NUM_COLUMNS-1)))"       >&2
	exit 1
fi

##############################################################################
# RESOLVE METADATA CSV
##############################################################################
case "$LABEL_TYPE" in
	single) METADATA_CSV="${SINGLE_LABEL_CSVS[$dataset_index]}" ;;
	multi)  METADATA_CSV="${MULTI_LABEL_CSVS[$dataset_index]}"  ;;
	*)
		echo "Error: LABEL_TYPE must be 'single' or 'multi', got '$LABEL_TYPE'" >&2
		exit 1
		;;
esac

##############################################################################
# PER-DATASET HYPERPARAMETERS
# All arrays are indexed by dataset_index (0=H4, 1=NA, 2=EU, 3=WWII, 4=SMU).
# Tune these values per dataset based on dataset size and class distribution.
##############################################################################

# Batch sizes — H4 is smaller so uses 16; larger datasets use 64
BATCH_SIZES=(16 64 64 64 64)

# How often (in steps) to print training progress
PRINT_FREQUENCIES=(500 1000 50 50 25)

# Learning rates — three groups based on fine-tuning strategy type:
#   FULL_FT       : full model fine-tuning (all parameters updated)
#   LINEAR_PROBE  : only the classification head is trained
#   PEFT_ALL      : all parameter-efficient methods (LoRA family + adapters)
LR_FULL_FT=(5.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_LINEAR_PROBE=(1.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_PEFT_ALL=(1.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)

WEIGHT_DECAY=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
EPOCHS=(100 100 150 150 150)

# ── LoRA family hyperparameters ──────────────────────────────────────────
# Used by: lora, lora_plus, rslora, dora, vera
#   LORA_RANK    : intrinsic rank of the low-rank decomposition
#   LORA_ALPHA   : scaling factor (typically 2× rank)
#   LORA_DROPOUT : dropout applied inside LoRA layers
#   LORA_PLUS_LAMBDA : ratio of learning rates (B matrix / A matrix) for LoRA+
LORA_RANKS=(16 64 64 64 64)
LORA_ALPHAS=(32.0 128.0 128.0 128.0 128.0)
LORA_DROPOUTS=(0.05 0.1 0.05 0.05 0.05)
LORA_PLUS_LAMBDAS=(4.0 16.0 16.0 16.0 16.0)  # only used by lora_plus

# ── Progressive fine-tuning parameters ──────────────────────────────────
# Controls the layer-by-layer unfreezing schedule used in "full" strategy
MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
TOTAL_NUM_PHASES=(8 4 4 4 4)

# ── Early stopping parameters ────────────────────────────────────────────
# INIT_MIN_EPOCHS : minimum epochs before early stopping can trigger
# PATIENCE        : how many checks without improvement before stopping
# MIN_DELTA       : minimum absolute improvement to count as progress
# CUMULATIVE_DELTA: minimum cumulative improvement over patience window
EARLY_STOPPING_INIT_MIN_EPOCHS=(15 25 17 17 12)
EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

# ── Stability / convergence thresholds ──────────────────────────────────
# VOLATILITY_THRESHOLD  : max allowed loss variance before flagging instability
# SLOPE_THRESHOLD       : minimum loss slope to consider training still improving
# PAIRWISE_IMP_THRESHOLD: minimum pairwise epoch-to-epoch improvement
VOLATILITY_THRESHOLDS=(5.0 15.0 15.0 15.0 15.0)
SLOPE_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)
PAIRWISE_IMP_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)

##############################################################################
# RESOLVE STRATEGY, ARCHITECTURE, AND COLUMN FOR THIS JOB
##############################################################################
strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
architecture="${MODEL_ARCHITECTURES[$architecture_index]}"
column="${COLUMNS[$column_index]}"

# ── Adapter detection ────────────────────────────────────────────────────
# If the strategy name contains "adapter", extract it as the adapter method
# and pass strategy="adapter" to trainer.py, with --adapter_method set separately.
# This keeps trainer.py's strategy interface clean and consistent.
ADAPTER_METHOD=""
if [[ "$strategy" == *"adapter"* ]]; then
	ADAPTER_METHOD="$strategy"   # e.g. "tip_adapter_f", "clip_adapter_v"
	strategy="adapter"
fi

# ── Baseline detection ───────────────────────────────────────────────────
# zero_shot and probe are passed as strategy="baseline" with --baseline_method
# set to the original name, so trainer.py can handle both under one branch.
BASELINE_METHOD=""
if [[ "$strategy" == "zero_shot" ]] || [[ "$strategy" == "probe" ]]; then
	BASELINE_METHOD="$strategy"  # e.g. "zero_shot" or "probe"
	strategy="baseline"
fi

##############################################################################
# LEARNING RATE SELECTION
# Chosen based on the resolved strategy group:
#   full     → LR_FULL_FT
#   probe    → LR_LINEAR_PROBE
#   all else → LR_PEFT_ALL  (LoRA family + adapters + zero_shot)
##############################################################################
if [ "$strategy" = "full" ]; then
	LEARNING_RATE="${LR_FULL_FT[$dataset_index]}"
elif [ "$BASELINE_METHOD" = "probe" ]; then
	LEARNING_RATE="${LR_LINEAR_PROBE[$dataset_index]}"
else
	# Covers: lora, lora_plus, rslora, dora, vera, ia3, adapter, baseline(zero_shot)
	LEARNING_RATE="${LR_PEFT_ALL[$dataset_index]}"
fi

##############################################################################
# EARLY STOPPING MINIMUM EPOCHS
# Enforce a floor of 3 epochs regardless of dataset setting, to avoid
# stopping before any meaningful training has occurred.
##############################################################################
EARLY_STOPPING_MIN_EPOCHS="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

# Batch size (currently 1:1 with default, reserved for future per-arch scaling)
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"

##############################################################################
# CONFIGURATION SUMMARY (printed to the SLURM log for traceability)
##############################################################################
echo "=== CONFIGURATION ==="
echo "SLURM_ARRAY_TASK_ID : $SLURM_ARRAY_TASK_ID"
echo "LABEL_TYPE          : $LABEL_TYPE"
echo "DATASET[$dataset_index]      : ${BASE_DATASET_DIRECTORY[$dataset_index]}"
echo "CSV_FILE            : $METADATA_CSV"
echo "STRATEGY[$strategy_index]    : $strategy"
[ -n "$ADAPTER_METHOD"  ] && echo "ADAPTER_METHOD      : $ADAPTER_METHOD"
[ -n "$BASELINE_METHOD" ] && echo "BASELINE_METHOD     : $BASELINE_METHOD"
echo "ARCHITECTURE[$architecture_index] : $architecture"
echo "COLUMN[$column_index]       : $column"
echo "EPOCHS              : ${EPOCHS[$dataset_index]}"
echo "LEARNING_RATE       : $LEARNING_RATE"
echo "WEIGHT_DECAY        : ${WEIGHT_DECAY[$dataset_index]}"
echo "EARLY_STOP_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
echo "BATCH_SIZE [default/adjusted]: ${BATCH_SIZES[$dataset_index]} / $ADJUSTED_BATCH_SIZE"
# Print LoRA-specific params only when relevant
if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
   [ "$strategy" = "dora" ] || [ "$strategy" = "vera"      ] || \
   [ "$strategy" = "rslora" ]; then
	echo "LORA_RANK           : ${LORA_RANKS[$dataset_index]}"
	echo "LORA_ALPHA          : ${LORA_ALPHAS[$dataset_index]}"
	echo "LORA_DROPOUT        : ${LORA_DROPOUTS[$dataset_index]}"
	[ "$strategy" = "lora_plus" ] && \
		echo "LORA_PLUS_LAMBDA    : ${LORA_PLUS_LAMBDAS[$dataset_index]}"
fi
echo "====================="

##############################################################################
# BUILD TRAINER COMMAND
# All arguments common to every strategy are set here.
# Strategy-specific arguments are appended conditionally below.
##############################################################################
CMD="python -u trainer.py \
	--metadata_csv              \"$METADATA_CSV\" \
	--model_architecture        \"$architecture\" \
	--strategy                  \"$strategy\" \
	--column                    \"$column\" \
	--epochs                    \"${EPOCHS[$dataset_index]}\" \
	--num_workers               \"$SLURM_CPUS_PER_TASK\" \
	--batch_size                \"$ADJUSTED_BATCH_SIZE\" \
	--learning_rate             \"$LEARNING_RATE\" \
	--weight_decay              \"${WEIGHT_DECAY[$dataset_index]}\" \
	--minimum_epochs            \"$EARLY_STOPPING_MIN_EPOCHS\" \
	--patience                  \"${EARLY_STOPPING_PATIENCE[$dataset_index]}\" \
	--minimum_delta             \"${EARLY_STOPPING_MIN_DELTA[$dataset_index]}\" \
	--cumulative_delta          \"${EARLY_STOPPING_CUMULATIVE_DELTA[$dataset_index]}\" \
	--volatility_threshold      \"${VOLATILITY_THRESHOLDS[$dataset_index]}\" \
	--slope_threshold           \"${SLOPE_THRESHOLDS[$dataset_index]}\" \
	--pairwise_imp_threshold    \"${PAIRWISE_IMP_THRESHOLDS[$dataset_index]}\" \
	--min_phases_before_stopping \"${MIN_PHASES_BEFORE_STOPPING[$dataset_index]}\" \
	--min_epochs_per_phase      \"${MIN_EPOCHS_PER_PHASE[$dataset_index]}\" \
	--total_num_phases          \"${TOTAL_NUM_PHASES[$dataset_index]}\" \
	--print_every               \"${PRINT_FREQUENCIES[$dataset_index]}\" \
	--sampling                  \"${SAMPLINGS[1]}\" \
	--verbose"

# ── LoRA family: append shared rank/alpha/dropout args ───────────────────
# Applied to: lora, lora_plus, rslora, dora, vera
# NOT applied to: ia3 (uses a different parameterisation with no rank/alpha)
if [ "$strategy" = "lora"      ] || [ "$strategy" = "lora_plus" ] || \
   [ "$strategy" = "dora"      ] || [ "$strategy" = "vera"      ] || \
   [ "$strategy" = "rslora"    ]; then
	CMD="$CMD --lora_rank    \"${LORA_RANKS[$dataset_index]}\""
	CMD="$CMD --lora_alpha   \"${LORA_ALPHAS[$dataset_index]}\""
	CMD="$CMD --lora_dropout \"${LORA_DROPOUTS[$dataset_index]}\""
fi

# ── LoRA+: additionally pass the B/A learning-rate ratio ─────────────────
if [ "$strategy" = "lora_plus" ]; then
	CMD="$CMD --lora_plus_lambda \"${LORA_PLUS_LAMBDAS[$dataset_index]}\""
fi

# ── Adapter: pass the specific adapter variant ───────────────────────────
# ADAPTER_METHOD is one of: tip_adapter_f, clip_adapter_v,
#                            clip_adapter_vt, clip_adapter_t, tip_adapter
[ -n "$ADAPTER_METHOD"  ] && CMD="$CMD --adapter_method  \"$ADAPTER_METHOD\""

# ── Baseline: pass zero_shot or probe as the baseline method ─────────────
[ -n "$BASELINE_METHOD" ] && CMD="$CMD --baseline_method \"$BASELINE_METHOD\""

##############################################################################
# EXECUTE
##############################################################################
eval $CMD

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"