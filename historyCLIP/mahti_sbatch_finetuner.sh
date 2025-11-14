# #!/bin/bash

# #SBATCH --account=project_2014707
# #SBATCH --job-name=h4_multi_label_dataset_
# #SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
# #SBATCH --mail-user=farid.alijani@gmail.com
# #SBATCH --mail-type=END,FAIL
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=64
# #SBATCH --mem=333G
# #SBATCH --partition=gpusmall
# #SBATCH --gres=gpu:a100:1
# #SBATCH --array=4
# #SBATCH --time=1-12:00:00

# set -euo pipefail

# user="`whoami`"
# stars=$(printf '%*s' 100 '')
# txt="$user began Slurm job: `date`"
# ch="#"
# echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
# echo "${stars// /*}"
# echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
# echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
# echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
# echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
# echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
# echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
# echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
# echo "${stars// /*}"

# # LABEL TYPE CONFIGURATION
# # LABEL_TYPE can be set from environment or sbatch: single | multi
# # Usage: sbatch --export=LABEL_TYPE=single mahti_sbatch_finetuner.sh
# LABEL_TYPE="${LABEL_TYPE:-multi}"  # default to multi-label

# # DATASET CONFIGURATION
# SAMPLINGS=(
# 	"kfold_stratified" 
# 	"stratified_random"
# )

# # Define base directories for datasets
# BASE_DATASET_DIRECTORY=(
# 	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
# 	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
# 	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
# 	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
# 	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
# )

# # Define metadata file names
# SINGLE_LABEL_FILE="metadata_single_label.csv"
# MULTI_LABEL_FILE="metadata_multi_label_multimodal.csv"

# # Build CSV file arrays dynamically
# SINGLE_LABEL_CSVS=()
# MULTI_LABEL_CSVS=()

# for dir in "${BASE_DATASET_DIRECTORY[@]}"; do
# 	SINGLE_LABEL_CSVS+=("${dir}/${SINGLE_LABEL_FILE}")
# 	MULTI_LABEL_CSVS+=("${dir}/${MULTI_LABEL_FILE}")
# done

# # MODEL CONFIGURATION
# FINETUNE_STRATEGIES=(
# 	"full" 					# 0-3, 16-19, 32-35, 48-51, 64-67
# 	"lora" 					# 4-7, 20-23, 36-39, 52-55, 68-71
# 	"progressive" 	# 8-11, 24-27, 40-43, 56-59, 72-75
# 	"probe"					# 12-15, 28-31, 44-47, 60-63, 76-79
# )

# MODEL_ARCHITECTURES=(
# 	"ViT-L/14@336px" 	# 0, 4, 8, 12,  16, 20, 24, 28,  32, 36, 40, 44,  48, 52, 56, 60,  64, 68, 72, 76
# 	"ViT-L/14"				# 1, 5, 9, 13,  17, 21, 25, 29,  33, 37, 41, 45,  49, 53, 57, 61,  65, 69, 73, 77
# 	"ViT-B/32"				# 2, 6, 10, 14, 18, 22, 26, 30,  34, 38, 42, 46,  50, 54, 58, 62,  66, 70, 74, 78
# 	"ViT-B/16"				# 3, 7, 11, 15, 19, 23, 27, 31,  35, 39, 43, 47,  51, 55, 59, 63,  67, 71, 75, 79
# )

# # ARRAY JOB INDEX CALCULATION
# NUM_DATASETS=${#BASE_DATASET_DIRECTORY[@]} # Number of datasets
# NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]} # Number of fine-tune strategies
# NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]} # Number of model architectures

# # dataset × strategy × architecture
# ### 0-15:  dataset[0] with all strategy×architecture [H4]
# ### 16-31: dataset[1] with all strategy×architecture [NA]
# ### 32-47: dataset[2] with all strategy×architecture [EU]
# ### 48-63: dataset[3] with all strategy×architecture [WWII]
# ### 64-79: dataset[4] with all strategy×architecture [SMU]
# dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_ARCHITECTURES)))
# remainder=$((SLURM_ARRAY_TASK_ID % (NUM_STRATEGIES * NUM_ARCHITECTURES)))
# strategy_index=$((remainder / NUM_ARCHITECTURES))
# architecture_index=$((remainder % NUM_ARCHITECTURES))

# # Validate indices
# if [ $dataset_index -ge $NUM_DATASETS ] || 
# 	 [ $strategy_index -ge $NUM_STRATEGIES ] ||
# 	 [ $architecture_index -ge $NUM_ARCHITECTURES ]; then
# 	echo "Error: Invalid dataset, strategy, or architecture index" >&2
# 	exit 1
# fi

# # SELECT LABEL TYPE AND METADATA CSV
# case "$LABEL_TYPE" in
# 	single)
# 		METADATA_CSV="${SINGLE_LABEL_CSVS[$dataset_index]}"
# 		;;
# 	multi)
# 		METADATA_CSV="${MULTI_LABEL_CSVS[$dataset_index]}"
# 		;;
# 	*)
# 		echo "Error: LABEL_TYPE must be 'single' or 'multi', got '$LABEL_TYPE'" >&2
# 		exit 1
# 		;;
# esac

# # HYPERPARAMETERS (per dataset)
# INIT_LRS=(5.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
# INIT_WDS=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
# DROPOUTS=(0.0 0.1 0.05 0.05 0.05)
# EPOCHS=(100 100 150 150 150)

# # LoRA parameters
# LORA_RANKS=(32 64 64 64 64)
# LORA_ALPHAS=(64.0 128.0 128.0 128.0 128.0) # 2x rank
# LORA_DROPOUTS=(0.15 0.1 0.05 0.05 0.05)

# # Linear probe parameters
# PROBE_DROPOUTS=(0.1 0.1 0.05 0.05 0.05)

# # Progressive finetuning parameters
# MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
# MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
# TOTAL_NUM_PHASES=(8 4 4 4 4)

# # Training parameters
# BATCH_SIZES=(512 64 64 64 64)
# PRINT_FREQUENCIES=(1000 1000 50 50 25)

# # Early stopping parameters
# EARLY_STOPPING_INIT_MIN_EPOCHS=(10 25 17 17 12)  # H4, NA, EU, WWII, SMU
# EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
# EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
# EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

# # Monitoring thresholds
# VOLATILITY_THRESHOLDS=(5.0 15.0 15.0 15.0 15.0)
# SLOPE_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)
# PAIRWISE_IMP_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)

# # Cache sizes
# CACHE_SIZES=(1024 512 1000 1000 1000)

# # STRATEGY-SPECIFIC ADJUSTMENTS
# strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
# architecture="${MODEL_ARCHITECTURES[$architecture_index]}"

# # Adjust early stopping minimum epochs based on strategy
# initial_early_stopping_minimum_epochs="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
# case $strategy in
# 	"full")
# 		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 3))  # Lower for Full
# 		;;
# 	"lora")
# 		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 3))  # Higher for LoRA
# 		;;
# 	"probe")
# 		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 4))  # Even lower for Linear Probe (fastest convergence)
# 		;;
# 	"progressive")
# 		EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs          # Original for Progressive
# 		;;
# esac
# # Ensure minimum of 3 epochs
# EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

# # Set dropout based on strategy
# # Only full and progressive can have nonzero dropouts, lora and probe must have zero dropouts
# if [ "$strategy" = "lora" ] || [ "$strategy" = "probe" ]; then
# 	DROPOUT=0.0
# else
# 	DROPOUT="${DROPOUTS[$dataset_index]}" # Use the original dropout for full and progressive
# fi

# # Determine batch size based on strategy and architecture
# case $strategy in
# 	"full"|"lora")
# 		ADJUSTED_BATCH_SIZE=48
# 		;;
# 	"progressive")
# 		# Progressive: Memory efficient, can use larger batches
# 		case $architecture in
# 			"ViT-L/14@336px")
# 				ADJUSTED_BATCH_SIZE=64  # Conservative for largest model
# 				;;
# 			"ViT-L/14")
# 				ADJUSTED_BATCH_SIZE=128  # Higher batch size
# 				;;
# 			"ViT-B/32"|"ViT-B/16")
# 				ADJUSTED_BATCH_SIZE=256 # Large batches for smaller models
# 				;;
# 		esac
# 		;;
# 	"probe")
# 		# Linear probe: Most memory efficient (only trains classifier)
# 		case $architecture in
# 			"ViT-L/14@336px")
# 				ADJUSTED_BATCH_SIZE=256 # Large batches work well
# 				;;
# 			"ViT-L/14")
# 				ADJUSTED_BATCH_SIZE=512 # Very large batches
# 				;;
# 			"ViT-B/32"|"ViT-B/16")
# 				ADJUSTED_BATCH_SIZE=1024 # Maximum efficiency for smaller models
# 				;;
# 		esac
# 		;;
# esac

# # CONFIGURATION SUMMARY
# echo "=== CONFIGURATION ==="
# echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
# echo "LABEL_TYPE: $LABEL_TYPE"
# echo "DATASET_INDEX: $dataset_index"
# echo "CSV_FILE: $METADATA_CSV"
# echo "STRATEGY_INDEX: $strategy_index"
# echo "FINETUNE_STRATEGY: $strategy"
# echo "ARCHITECTURE_INDEX: $architecture_index"
# echo "MODEL_ARCHITECTURE: $architecture"
# echo "EPOCHS: ${EPOCHS[$dataset_index]}"
# echo "INITIAL LEARNING RATE: ${INIT_LRS[$dataset_index]}"
# echo "INITIAL WEIGHT DECAY: ${INIT_WDS[$dataset_index]}"
# echo "DROPOUT: $DROPOUT"
# echo "EARLY_STOPPING_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
# echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: $ADJUSTED_BATCH_SIZE"
# echo "====================="

# # TRAINING EXECUTION
# echo ">> Starting trainer.py for dataset[$SLURM_ARRAY_TASK_ID]: $METADATA_CSV"
# python -u trainer.py \
# 	--metadata_csv "$METADATA_CSV" \
# 	--model_architecture "$architecture" \
# 	--mode "finetune" \
# 	--finetune_strategy "$strategy" \
# 	--epochs "${EPOCHS[$dataset_index]}" \
# 	--num_workers "$SLURM_CPUS_PER_TASK" \
# 	--batch_size "$ADJUSTED_BATCH_SIZE" \
# 	--dropout "$DROPOUT" \
# 	--learning_rate "${INIT_LRS[$dataset_index]}" \
# 	--weight_decay "${INIT_WDS[$dataset_index]}" \
# 	--minimum_epochs "$EARLY_STOPPING_MIN_EPOCHS" \
# 	--patience "${EARLY_STOPPING_PATIENCE[$dataset_index]}" \
# 	--minimum_delta "${EARLY_STOPPING_MIN_DELTA[$dataset_index]}" \
# 	--cumulative_delta "${EARLY_STOPPING_CUMULATIVE_DELTA[$dataset_index]}" \
# 	--volatility_threshold "${VOLATILITY_THRESHOLDS[$dataset_index]}" \
# 	--slope_threshold "${SLOPE_THRESHOLDS[$dataset_index]}" \
# 	--pairwise_imp_threshold "${PAIRWISE_IMP_THRESHOLDS[$dataset_index]}" \
# 	--lora_rank "${LORA_RANKS[$dataset_index]}" \
# 	--lora_alpha "${LORA_ALPHAS[$dataset_index]}" \
# 	--lora_dropout "${LORA_DROPOUTS[$dataset_index]}" \
# 	--probe_dropout "${PROBE_DROPOUTS[$dataset_index]}" \
# 	--min_phases_before_stopping "${MIN_PHASES_BEFORE_STOPPING[$dataset_index]}" \
# 	--min_epochs_per_phase "${MIN_EPOCHS_PER_PHASE[$dataset_index]}" \
# 	--total_num_phases "${TOTAL_NUM_PHASES[$dataset_index]}" \
# 	--print_every "${PRINT_FREQUENCIES[$dataset_index]}" \
# 	--sampling "${SAMPLINGS[1]}"
# 	# --cache_size "${CACHE_SIZES[$dataset_index]}"

# # JOB COMPLETION
# done_txt="$user finished Slurm job: $(date)"
# echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"


#!/bin/bash

#SBATCH --account=project_2014707
#SBATCH --job-name=h4_multi_label_dataset_
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=333G
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1
#SBATCH --array=4
#SBATCH --time=1-12:00:00

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

# LABEL TYPE CONFIGURATION
# LABEL_TYPE can be set from environment or sbatch: single | multi
# Usage: sbatch --export=LABEL_TYPE=single mahti_sbatch_finetuner.sh
LABEL_TYPE="${LABEL_TYPE:-multi}"  # default to multi-label

# DATASET CONFIGURATION
SAMPLINGS=(
	"kfold_stratified" 
	"stratified_random"
)

# Define base directories for datasets
BASE_DATASET_DIRECTORY=(
	/scratch/project_2004072/ImACCESS/WW_DATASETs/HISTORY_X4
	/scratch/project_2004072/ImACCESS/WW_DATASETs/NATIONAL_ARCHIVE_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/EUROPEANA_1900-01-01_1970-12-31
	/scratch/project_2004072/ImACCESS/WW_DATASETs/WWII_1939-09-01_1945-09-02
	/scratch/project_2004072/ImACCESS/WW_DATASETs/SMU_1900-01-01_1970-12-31
)

# Define metadata file names
SINGLE_LABEL_FILE="metadata_single_label.csv"
MULTI_LABEL_FILE="metadata_multi_label_multimodal.csv"

# Build CSV file arrays dynamically
SINGLE_LABEL_CSVS=()
MULTI_LABEL_CSVS=()

for dir in "${BASE_DATASET_DIRECTORY[@]}"; do
	SINGLE_LABEL_CSVS+=("${dir}/${SINGLE_LABEL_FILE}")
	MULTI_LABEL_CSVS+=("${dir}/${MULTI_LABEL_FILE}")
done

# MODEL CONFIGURATION
FINETUNE_STRATEGIES=(
	"full" 					# 0-3, 36-39, 72-75, 108-111, 144-147
	"lora" 					# 4-7, 40-43, 76-79, 112-115, 148-151
	"progressive" 	# 8-11, 44-47, 80-83, 116-119, 152-155
	"probe"					# 12-15, 48-51, 84-87, 120-123, 156-159
	"lora_plus"			# 16-19, 52-55, 88-91, 124-127, 160-163
	"dora"					# 20-23, 56-59, 92-95, 128-131, 164-167
	"vera"					# 24-27, 60-63, 96-99, 132-135, 168-171
	"ia3"						# 28-31, 64-67, 100-103, 136-139, 172-175
	"adapter"				# 32-35, 68-71, 104-107, 140-143, 176-179
)

MODEL_ARCHITECTURES=(
	"ViT-L/14@336px" 	# 0, 4, 8, 12, 16, 20, 24, 28, 32,  36, 40, 44, 48, 52, 56, 60, 64, 68,  72, 76, 80, 84, 88, 92, 96, 100, 104,  108, 112, 116, 120, 124, 128, 132, 136, 140,  144, 148, 152, 156, 160, 164, 168, 172, 176
	"ViT-L/14"				# 1, 5, 9, 13, 17, 21, 25, 29, 33,  37, 41, 45, 49, 53, 57, 61, 65, 69,  73, 77, 81, 85, 89, 93, 97, 101, 105,  109, 113, 117, 121, 125, 129, 133, 137, 141,  145, 149, 153, 157, 161, 165, 169, 173, 177
	"ViT-B/32"				# 2, 6, 10, 14, 18, 22, 26, 30, 34,  38, 42, 46, 50, 54, 58, 62, 66, 70,  74, 78, 82, 86, 90, 94, 98, 102, 106,  110, 114, 118, 122, 126, 130, 134, 138, 142,  146, 150, 154, 158, 162, 166, 170, 174, 178
	"ViT-B/16"				# 3, 7, 11, 15, 19, 23, 27, 31, 35,  39, 43, 47, 51, 55, 59, 63, 67, 71,  75, 79, 83, 87, 91, 95, 99, 103, 107,  111, 115, 119, 123, 127, 131, 135, 139, 143,  147, 151, 155, 159, 163, 167, 171, 175, 179
)

# ARRAY JOB INDEX CALCULATION
NUM_DATASETS=${#BASE_DATASET_DIRECTORY[@]} # Number of datasets
NUM_STRATEGIES=${#FINETUNE_STRATEGIES[@]} # Number of fine-tune strategies
NUM_ARCHITECTURES=${#MODEL_ARCHITECTURES[@]} # Number of model architectures

# dataset × strategy × architecture
### 0-35:   dataset[0] with all strategy×architecture [H4]
### 36-71:  dataset[1] with all strategy×architecture [NA]
### 72-107: dataset[2] with all strategy×architecture [EU]
### 108-143: dataset[3] with all strategy×architecture [WWII]
### 144-179: dataset[4] with all strategy×architecture [SMU]
dataset_index=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES * NUM_ARCHITECTURES)))
remainder=$((SLURM_ARRAY_TASK_ID % (NUM_STRATEGIES * NUM_ARCHITECTURES)))
strategy_index=$((remainder / NUM_ARCHITECTURES))
architecture_index=$((remainder % NUM_ARCHITECTURES))

# Validate indices
if [ $dataset_index -ge $NUM_DATASETS ] || 
	 [ $strategy_index -ge $NUM_STRATEGIES ] ||
	 [ $architecture_index -ge $NUM_ARCHITECTURES ]; then
	echo "Error: Invalid dataset, strategy, or architecture index" >&2
	exit 1
fi

# SELECT LABEL TYPE AND METADATA CSV
case "$LABEL_TYPE" in
	single)
		METADATA_CSV="${SINGLE_LABEL_CSVS[$dataset_index]}"
		;;
	multi)
		METADATA_CSV="${MULTI_LABEL_CSVS[$dataset_index]}"
		;;
	*)
		echo "Error: LABEL_TYPE must be 'single' or 'multi', got '$LABEL_TYPE'" >&2
		exit 1
		;;
esac

# HYPERPARAMETERS (per dataset)
INIT_LRS=(5.0e-04 5.0e-06 5.0e-06 5.0e-06 5.0e-06)
INIT_WDS=(1.0e-02 1.0e-02 1.0e-02 1.0e-02 1.0e-02)
DROPOUTS=(0.0 0.1 0.05 0.05 0.05)
EPOCHS=(100 100 150 150 150)

# LoRA parameters
LORA_RANKS=(32 64 64 64 64)
LORA_ALPHAS=(64.0 128.0 128.0 128.0 128.0) # 2x rank
LORA_DROPOUTS=(0.15 0.1 0.05 0.05 0.05)

# LoRA+ parameters
LORA_PLUS_LAMBDAS=(16.0 16.0 16.0 16.0 16.0) # Lambda multiplier for LoRA+

# Adapter parameters
ADAPTER_METHODS=("clip_adapter_vt" "clip_adapter_vt" "clip_adapter_vt" "clip_adapter_vt" "clip_adapter_vt") # H4, NA, EU, WWII, SMU

# Linear probe parameters
PROBE_DROPOUTS=(0.1 0.1 0.05 0.05 0.05)

# Progressive finetuning parameters
MIN_PHASES_BEFORE_STOPPING=(3 3 3 3 3)
MIN_EPOCHS_PER_PHASE=(5 5 5 5 5)
TOTAL_NUM_PHASES=(8 4 4 4 4)

# Training parameters
BATCH_SIZES=(512 64 64 64 64)
PRINT_FREQUENCIES=(1000 1000 50 50 25)

# Early stopping parameters
EARLY_STOPPING_INIT_MIN_EPOCHS=(10 25 17 17 12)  # H4, NA, EU, WWII, SMU
EARLY_STOPPING_PATIENCE=(3 5 5 5 5)
EARLY_STOPPING_MIN_DELTA=(1e-4 1e-4 1e-4 1e-4 1e-4)
EARLY_STOPPING_CUMULATIVE_DELTA=(5e-3 5e-3 5e-3 5e-3 5e-3)

# Monitoring thresholds
VOLATILITY_THRESHOLDS=(5.0 15.0 15.0 15.0 15.0)
SLOPE_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)
PAIRWISE_IMP_THRESHOLDS=(1e-4 1e-4 1e-4 1e-4 1e-4)

# Cache sizes
CACHE_SIZES=(1024 512 1000 1000 1000)

# STRATEGY-SPECIFIC ADJUSTMENTS
strategy="${FINETUNE_STRATEGIES[$strategy_index]}"
architecture="${MODEL_ARCHITECTURES[$architecture_index]}"

# Adjust early stopping minimum epochs based on strategy
initial_early_stopping_minimum_epochs="${EARLY_STOPPING_INIT_MIN_EPOCHS[$dataset_index]}"
case $strategy in
	"full")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 3))  # Lower for Full
		;;
	"lora"|"lora_plus"|"dora"|"vera")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 3))  # Higher for LoRA-based methods
		;;
	"probe")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs - 4))  # Even lower for Linear Probe (fastest convergence)
		;;
	"progressive")
		EARLY_STOPPING_MIN_EPOCHS=$initial_early_stopping_minimum_epochs          # Original for Progressive
		;;
	"ia3"|"adapter")
		EARLY_STOPPING_MIN_EPOCHS=$((initial_early_stopping_minimum_epochs + 2))  # Slightly higher for IA3 and Adapter
		;;
esac
# Ensure minimum of 3 epochs
EARLY_STOPPING_MIN_EPOCHS=$((EARLY_STOPPING_MIN_EPOCHS < 3 ? 3 : EARLY_STOPPING_MIN_EPOCHS))

# Set dropout based on strategy
# Only full and progressive can have nonzero dropouts, parameter-efficient methods must have zero dropouts
if [ "$strategy" = "lora" ] || [ "$strategy" = "lora_plus" ] || [ "$strategy" = "dora" ] || [ "$strategy" = "vera" ] || [ "$strategy" = "ia3" ] || [ "$strategy" = "probe" ] || [ "$strategy" = "adapter" ]; then
	DROPOUT=0.0
else
	DROPOUT="${DROPOUTS[$dataset_index]}" # Use the original dropout for full and progressive
fi

# Determine batch size based on strategy and architecture
case $strategy in
	"full"|"lora"|"lora_plus"|"dora"|"vera")
		ADJUSTED_BATCH_SIZE=48
		;;
	"progressive")
		# Progressive: Memory efficient, can use larger batches
		case $architecture in
			"ViT-L/14@336px")
				ADJUSTED_BATCH_SIZE=64  # Conservative for largest model
				;;
			"ViT-L/14")
				ADJUSTED_BATCH_SIZE=128  # Higher batch size
				;;
			"ViT-B/32"|"ViT-B/16")
				ADJUSTED_BATCH_SIZE=256 # Large batches for smaller models
				;;
		esac
		;;
	"probe"|"ia3"|"adapter")
		# Linear probe, IA3, Adapter: Most memory efficient (only trains small subset of parameters)
		case $architecture in
			"ViT-L/14@336px")
				ADJUSTED_BATCH_SIZE=256 # Large batches work well
				;;
			"ViT-L/14")
				ADJUSTED_BATCH_SIZE=512 # Very large batches
				;;
			"ViT-B/32"|"ViT-B/16")
				ADJUSTED_BATCH_SIZE=1024 # Maximum efficiency for smaller models
				;;
		esac
		;;
esac

# CONFIGURATION SUMMARY
echo "=== CONFIGURATION ==="
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "LABEL_TYPE: $LABEL_TYPE"
echo "DATASET_INDEX: $dataset_index"
echo "CSV_FILE: $METADATA_CSV"
echo "STRATEGY_INDEX: $strategy_index"
echo "FINETUNE_STRATEGY: $strategy"
echo "ARCHITECTURE_INDEX: $architecture_index"
echo "MODEL_ARCHITECTURE: $architecture"
echo "EPOCHS: ${EPOCHS[$dataset_index]}"
echo "INITIAL LEARNING RATE: ${INIT_LRS[$dataset_index]}"
echo "INITIAL WEIGHT DECAY: ${INIT_WDS[$dataset_index]}"
echo "DROPOUT: $DROPOUT"
echo "EARLY_STOPPING_MIN_EPOCHS: $EARLY_STOPPING_MIN_EPOCHS"
echo "BATCH SIZE: [DEFAULT]: ${BATCH_SIZES[$dataset_index]} [ADJUSTED]: $ADJUSTED_BATCH_SIZE"
echo "====================="

# TRAINING EXECUTION
echo ">> Starting trainer.py for dataset[$SLURM_ARRAY_TASK_ID]: $METADATA_CSV"

# Build the base command
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

# Add strategy-specific parameters
if [ "$strategy" = "lora_plus" ]; then
	CMD="$CMD --lora_plus_lambda \"${LORA_PLUS_LAMBDAS[$dataset_index]}\""
fi

if [ "$strategy" = "adapter" ]; then
	CMD="$CMD --adapter_method \"${ADAPTER_METHODS[$dataset_index]}\""
fi

# Execute the command
eval $CMD
	# --cache_size "${CACHE_SIZES[$dataset_index]}"

# JOB COMPLETION
done_txt="$user finished Slurm job: $(date)"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"