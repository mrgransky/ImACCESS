#!/bin/bash

#SBATCH --account=project_2009043
#SBATCH --job-name=ft_h4_multi_label
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=164G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
####SBATCH --array=0-40:4
###SBATCH --array=4,8
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
# 										# H4			# NA			# EU				# WWII			# SMU
FINETUNE_STRATEGIES=(
	"zero_shot"					# 00-03,  56-59,  	112-115,  	168-171,  	224-227
	"probe"							# 04-07, 	60-63,  	116-119,  	172-175,  	228-231
	"full" 							# 08-11, 	64-67,  	120-123,  	176-179,  	232-235
	"lora"							# 12-15, 	68-71,  	124-127,  	180-183,  	236-239
	"lora_plus"					# 16-19, 	72-75,  	128-131,  	184-187,  	240-243
	"rslora"						# 20-23, 	76-79,  	132-135,  	188-191,  	244-247
	"dora"							# 24-27, 	80-83,  	136-139,  	192-195,  	248-251
	"vera"							# 28-31, 	84-87,  	140-143,  	196-199,  	252-255
	"ia3"								# 32-35, 	88-91,  	144-147,  	200-203,  	256-259
	"tip_adapter_f"			# 36-39, 	92-95,  	148-151,  	204-207,  	260-263
	"clip_adapter_v"		# 40-43, 	96-99,  	152-155,  	208-211,  	264-267
	"clip_adapter_vt"		# 44-47, 	100-103, 	156-159,  	212-215,  	268-271
	"clip_adapter_t"		# 48-51, 	104-107, 	160-163,  	216-219,  	272-275
	"tip_adapter"				# 52-55, 	108-111, 	164-167,  	220-223,  	276-279
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

BATCH_SIZES=(16 64 64 64 64)
PRINT_FREQUENCIES=(1000 1000 50 50 25)

# Learning rates by method group
# LR_FULL_FT=(1.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_FULL_FT=(5.0e-05 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_LINEAR_PROBE=(1.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
LR_PEFT_ALL=(1.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)  # LoRA, LoRA+, DoRA, VeRA, IA³, Adapters

WEIGHT_DECAY=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
EPOCHS=(85 100 150 150 150)

# LoRA family parameters
LORA_RANKS=(16 64 64 64 64)
LORA_ALPHAS=(32.0 128.0 128.0 128.0 128.0)
LORA_DROPOUTS=(0.05 0.1 0.05 0.05 0.05)
LORA_PLUS_LAMBDAS=(4.0 16.0 16.0 16.0 16.0)

# Progressive fine-tuning parameters
MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
TOTAL_NUM_PHASES=(8 4 4 4 4)

# Early stopping parameters
EARLY_STOPPING_INIT_MIN_EPOCHS=(15 25 17 17 12)
EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

# Stability thresholds
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

BASELINE_METHOD=""
if [[ "$strategy" == "zero_shot" ]] || [[ "$strategy" == "probe" ]]; then
	BASELINE_METHOD="$strategy"
	strategy="baseline"
fi

# Determine learning rate based on strategy
if [ "$strategy" = "full" ]; then
		LEARNING_RATE="${LR_FULL_FT[$dataset_index]}"
elif [ "$BASELINE_METHOD" = "probe" ]; then
		LEARNING_RATE="${LR_LINEAR_PROBE[$dataset_index]}"
else
		# All PEFT methods: LoRA, LoRA+, DoRA, VeRA, IA³, Adapters
		LEARNING_RATE="${LR_PEFT_ALL[$dataset_index]}"
fi

# Early stopping minimum epochs
initial_early_stopping_minimum_epochs="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

# Batch size
ADJUSTED_BATCH_SIZE="${BATCH_SIZES[$dataset_index]}"

echo "=== CONFIGURATION ==="
echo "LABEL_TYPE: $LABEL_TYPE"
echo "CSV_FILE[$dataset_index]: $METADATA_CSV"
echo "STRATEGY[$strategy_index]: $strategy"
if [ -n "$ADAPTER_METHOD" ]; then
	echo "ADAPTER_METHOD: $ADAPTER_METHOD"
fi
if [ -n "$BASELINE_METHOD" ]; then
	echo "BASELINE_METHOD: $BASELINE_METHOD"
fi
echo "ARCHITECTURE[$architecture_index]: $architecture"
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "LR: $LEARNING_RATE"
echo "WD: ${WEIGHT_DECAY[$dataset_index]}"
echo "EARLY_STOPPING_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: $ADJUSTED_BATCH_SIZE"
# Show LoRA parameters only for applicable methods
if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
   [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || [ "$strategy" = "rslora" ]; then
    echo "LORA_RANK: ${LORA_RANKS[$dataset_index]}"
    echo "LORA_ALPHA: ${LORA_ALPHAS[$dataset_index]}"
    echo "LORA_DROPOUT: ${LORA_DROPOUTS[$dataset_index]}"
    if [ "$strategy" = "lora_plus" ]; then
        echo "LORA_PLUS_LAMBDA: ${LORA_PLUS_LAMBDAS[$dataset_index]}"
    fi
fi
echo "====================="

CMD="python -u trainer.py \
	--metadata_csv \"$METADATA_CSV\" \
	--model_architecture \"$architecture\" \
	--strategy \"$strategy\" \
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

# Add LoRA parameters only for LoRA, LoRA+, DoRA, VeRA, and RSLora
if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || \
	 [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || [ "$strategy" = "rslora" ]; then
	CMD="$CMD --lora_rank \"${LORA_RANKS[$dataset_index]}\""
	CMD="$CMD --lora_alpha \"${LORA_ALPHAS[$dataset_index]}\""
	CMD="$CMD --lora_dropout \"${LORA_DROPOUTS[$dataset_index]}\""
fi

if [ "$strategy" = "lora_plus" ]; then
	CMD="$CMD --lora_plus_lambda \"${LORA_PLUS_LAMBDAS[$dataset_index]}\""
fi

if [ -n "$ADAPTER_METHOD" ]; then
	CMD="$CMD --adapter_method \"$ADAPTER_METHOD\""
fi

if [ -n "$BASELINE_METHOD" ]; then
	CMD="$CMD --baseline_method \"$BASELINE_METHOD\""
fi

eval $CMD

done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"