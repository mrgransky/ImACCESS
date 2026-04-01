#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=ft_h4_multi_label
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=192G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-131:12
#SBATCH --time=03-00:00:00

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
# DIMENSIONS:
#   dataset(5) × strategy(14) × architecture(4) × column(3) = 840 jobs
#
# Per-dataset block  = 14 × 4 × 3 = 168 jobs
# Per-strategy block =      4 × 3 =  12 jobs
# Per-arch block     =          3 =   3 jobs
#
# dataset 0 (H4):   0–167
# dataset 1 (NA):   168–335
# dataset 2 (EU):   336–503
# dataset 3 (WWII): 504–671
# dataset 4 (SMU):  672–839
#
# Within each dataset block (offset = dataset_index * 168):
#   strategy 0  (zero_shot):       +0–11
#   strategy 1  (probe):           +12–23
#   strategy 2  (full):            +24–35
#   strategy 3  (lora):            +36–47
#   strategy 4  (lora_plus):       +48–59
#   strategy 5  (rslora):          +60–71
#   strategy 6  (dora):            +72–83
#   strategy 7  (vera):            +84–95
#   strategy 8  (ia3):             +96–107
#   strategy 9  (tip_adapter_f):   +108–119
#   strategy 10 (clip_adapter_v):  +120–131
#   strategy 11 (clip_adapter_vt): +132–143
#   strategy 12 (clip_adapter_t):  +144–155
#   strategy 13 (tip_adapter):     +156–167
#
# Within each strategy block (offset = strategy_index * 12):
#   arch 0 (ViT-L/14@336px): +0, +1, +2   (cols: llm, vlm, multimodal)
#   arch 1 (ViT-L/14):       +3, +4, +5
#   arch 2 (ViT-B/32):       +6, +7, +8
#   arch 3 (ViT-B/16):       +9, +10, +11
##############################################################################

##############################################################################
#                        # H4        # NA        # EU        # WWII      # SMU
FINETUNE_STRATEGIES=(
	"zero_shot"      	#    00–11       168–179     336–347     504–515     672–683
	"probe"          	#    12–23       180–191     348–359     516–527     684–695
	"full"           	#    24–35       192–203     360–371     528–539     696–707
	"lora"           	#    36–47       204–215     372–383     540–551     708–719
	"lora_plus"      	#    48–59       216–227     384–395     552–563     720–731
	"rslora"         	#    60–71       228–239     396–407     564–575     732–743
	"dora"           	#    72–83       240–251     408–419     576–587     744–755
	"vera"           	#    84–95       252–263     420–431     588–599     756–767
	"ia3"            	#    96–107      264–275     432–443     600–611     768–779
	"tip_adapter_f"  	#    108–119     276–287     444–455     612–623     780–791
	"clip_adapter_v" 	#:   120–131     288–299     456–467     624–635     792–803
	"clip_adapter_vt"	#:   132–143     300–311     468–479     636–647     804–815
	"clip_adapter_t" 	#:   144–155     312–323     480–491     648–659     816–827
	"tip_adapter"    	#:   156–167     324–335     492–503     660–671     828–839
)
##############################################################################

MODEL_ARCHITECTURES=(
	"ViT-L/14@336px"   # arch 0
	"ViT-L/14"         # arch 1
	"ViT-B/32"         # arch 2
	"ViT-B/16"         # arch 3
)

COLUMNS=(
	"llm_canonical_labels"         # col 0
	"vlm_canonical_labels"         # col 1
	"multimodal_canonical_labels"  # col 2
)

NUM_DATASETS=${#BASE_DATASET_DIRECTORY[@]}   # 5
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]}    # 14
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]} # 4
NUM_COLUMNS=${#COLUMNS[@]}                   # 3

# INDEXING: dataset × strategy × architecture × column
dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_ARCHITECTURES * NUM_COLUMNS)))
remainder=$((SLURM_ARRAY_TASK_ID % (NUM_STRATEGIES * NUM_ARCHITECTURES * NUM_COLUMNS)))
strategy_index=$((remainder / (NUM_ARCHITECTURES * NUM_COLUMNS)))
remainder2=$((remainder % (NUM_ARCHITECTURES * NUM_COLUMNS)))
architecture_index=$((remainder2 / NUM_COLUMNS))
column_index=$((remainder2 % NUM_COLUMNS))

if [ $dataset_index -ge $NUM_DATASETS ] || \
	 [ $strategy_index -ge $NUM_STRATEGIES ] || \
	 [ $architecture_index -ge $NUM_ARCHITECTURES ] || \
	 [ $column_index -ge $NUM_COLUMNS ]; then
	echo "Error: Index out of bounds" >&2
	echo "  dataset_index=$dataset_index (max $((NUM_DATASETS-1)))" >&2
	echo "  strategy_index=$strategy_index (max $((NUM_STRATEGIES-1)))" >&2
	echo "  architecture_index=$architecture_index (max $((NUM_ARCHITECTURES-1)))" >&2
	echo "  column_index=$column_index (max $((NUM_COLUMNS-1)))" >&2
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

BATCH_SIZES=(16 64 64 64 64)
PRINT_FREQUENCIES=(500 1000 50 50 25)

LR_FULL_FT=(5.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_LINEAR_PROBE=(1.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_PEFT_ALL=(1.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)

WEIGHT_DECAY=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
EPOCHS=(100 100 150 150 150)

LORA_RANKS=(16 64 64 64 64)
LORA_ALPHAS=(32.0 128.0 128.0 128.0 128.0)
LORA_DROPOUTS=(0.05 0.1 0.05 0.05 0.05)
LORA_PLUS_LAMBDAS=(4.0 16.0 16.0 16.0 16.0)

MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
TOTAL_NUM_PHASES=(8 4 4 4 4)

EARLY_STOPPING_INIT_MIN_EPOCHS=(15 25 17 17 12)
EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

VOLATILITY_THRESHOLDS=(5.0 15.0 15.0 15.0 15.0)
SLOPE_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)
PAIRWISE_IMP_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)

strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
architecture="${MODEL_ARCHITECTURES[$architecture_index]}"
column="${COLUMNS[$column_index]}"

ADAPTER_METHOD=""
if [[ "$strategy" == *"adapter"* ]]; then
	ADAPTER_METHOD="$strategy"
	strategy="adapter"
fi

BASELINE_METHOD=""
if [[ "$strategy" == "zero_shot" ]] || [[ "$strategy" == "probe" ]]; then
	BASELINE_METHOD="$strategy"
	strategy="baseline"
fi

if [ "$strategy" = "full" ]; then
	LEARNING_RATE="${LR_FULL_FT[$dataset_index]}"
elif [ "$BASELINE_METHOD" = "probe" ]; then
	LEARNING_RATE="${LR_LINEAR_PROBE[$dataset_index]}"
else
	LEARNING_RATE="${LR_PEFT_ALL[$dataset_index]}"
fi

initial_early_stopping_minimum_epochs="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"

echo "=== CONFIGURATION ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LABEL_TYPE: $LABEL_TYPE"
echo "DATASET[$dataset_index]: ${BASE_DATASET_DIRECTORY[$dataset_index]}"
echo "CSV_FILE: $METADATA_CSV"
echo "STRATEGY[$strategy_index]: $strategy"
[ -n "$ADAPTER_METHOD" ]  && echo "ADAPTER_METHOD: $ADAPTER_METHOD"
[ -n "$BASELINE_METHOD" ] && echo "BASELINE_METHOD: $BASELINE_METHOD"
echo "ARCHITECTURE[$architecture_index]: $architecture"
echo "COLUMN[$column_index]: $column"
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "LR: $LEARNING_RATE"
echo "WD: ${WEIGHT_DECAY[$dataset_index]}"
echo "EARLY_STOPPING_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: $ADJUSTED_BATCH_SIZE"
if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
	 [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || [ "$strategy" = "rslora" ]; then
	echo "LORA_RANK: ${LORA_RANKS[$dataset_index]}"
	echo "LORA_ALPHA: ${LORA_ALPHAS[$dataset_index]}"
	echo "LORA_DROPOUT: ${LORA_DROPOUTS[$dataset_index]}"
	[ "$strategy" = "lora_plus" ] && echo "LORA_PLUS_LAMBDA: ${LORA_PLUS_LAMBDAS[$dataset_index]}"
fi
echo "====================="

CMD="python -u trainer.py \
	--metadata_csv \"$METADATA_CSV\" \
	--model_architecture \"$architecture\" \
	--strategy \"$strategy\" \
	--column \"$column\" \
	--epochs \"${EPOCHS[$dataset_index]}\" \
	--num_workers \"$SLURM_CPUS_PER_TASK\" \
	--batch_size \"$ADJUSTED_BATCH_SIZE\" \
	--learning_rate \"$LEARNING_RATE\" \
	--weight_decay \"${WEIGHT_DECAY[$dataset_index]}\" \
	--minimum_epochs \"$EARLY_STOPPING_MIN_EPOCHS\" \
	--patience \"${EARLY_STOPPING_PATIENCE[$dataset_index]}\" \
	--minimum_delta \"${EARLY_STOPPING_MIN_DELTA[$dataset_index]}\" \
	--cumulative_delta \"${EARLY_STOPPING_CUMULATIVE_DELTA[$dataset_index]}\" \
	--volatility_threshold \"${VOLATILITY_THRESHOLDS[$dataset_index]}\" \
	--slope_threshold \"${SLOPE_THRESHOLDS[$dataset_index]}\" \
	--pairwise_imp_threshold \"${PAIRWISE_IMP_THRESHOLDS[$dataset_index]}\" \
	--min_phases_before_stopping \"${MIN_PHASES_BEFORE_STOPPING[$dataset_index]}\" \
	--min_epochs_per_phase \"${MIN_EPOCHS_PER_PHASE[$dataset_index]}\" \
	--total_num_phases \"${TOTAL_NUM_PHASES[$dataset_index]}\" \
	--print_every \"${PRINT_FREQUENCIES[$dataset_index]}\" \
	--sampling \"${SAMPLINGS[1]}\" \
	--verbose"

if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
	 [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || [ "$strategy" = "rslora" ]; then
	CMD="$CMD --lora_rank \"${LORA_RANKS[$dataset_index]}\""
	CMD="$CMD --lora_alpha \"${LORA_ALPHAS[$dataset_index]}\""
	CMD="$CMD --lora_dropout \"${LORA_DROPOUTS[$dataset_index]}\""
fi

if [ "$strategy" = "lora_plus" ]; then
	CMD="$CMD --lora_plus_lambda \"${LORA_PLUS_LAMBDAS[$dataset_index]}\""
fi

[ -n "$ADAPTER_METHOD" ]  && CMD="$CMD --adapter_method \"$ADAPTER_METHOD\""
[ -n "$BASELINE_METHOD" ] && CMD="$CMD --baseline_method \"$BASELINE_METHOD\""

eval $CMD

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"