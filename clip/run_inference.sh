#!/bin/bash

## run [Pouta] server:
## $ nohup bash run_inference.sh > /dev/null 2>&1 &
## $ nohup bash run_inference.sh > /media/volume/ImACCESS/trash/run_inference_prec_at_k.out 2>&1 &

## run [local] server:
## $ nohup bash run_inference.sh > prec_at_k.out 2>&1 &

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error and exit immediately.
set -o pipefail # If any command in a pipeline fails, the entire pipeline will fail.

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

source $(conda info --base)/bin/activate py39
LOGS_DIRECTORY="logs"
mkdir -p $LOGS_DIRECTORY
topk_values=(1 5 10 15 20)
DATASET="imagenet"
# batch_size=1024

# Function to get GPU with most available memory
get_max_memory_gpu() {
	local device=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
								 sort -k2 -n | tail -n 1 | cut -d ',' -f 1)
	if [[ -z "$device" ]]; then
			echo "No CUDA devices found. Please check your drivers or hardware."
			exit 1
	fi
	echo "$device"
}

get_batch_size() {
  local device_id=$1
  local memory_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $device_id)

  # Error check nvidia-smi
  if [[ -z "$memory_free" || ! "$memory_free" =~ ^[0-9]+$ ]]; then
    echo "Error: Could not retrieve free memory for GPU $device_id"
    exit 1  # Or handle differently (e.g., default batch size, skip run)
  fi

  # # Memory is in MiB by default. Convert to bytes (1 MiB = 1024 * 1024 bytes)
  # local memory_free_bytes=$((memory_free * 1024 * 1024))

  # # Model parameter count (replace with actual parameter count in bytes)
  # local model_param_size=100000000 # Example: 100 million parameters * 4 bytes/param
  # # Assuming each weight is 4 bytes

  # # Rough calculation:  (Free memory / (Memory per weight * num weights))
  # local batch_size=$((memory_free_bytes / (4 * model_param_size) ))

  # # Adjust batch size based on memory requirements per image
  # # Note: the 2 below is completely a guesstimate, this needs to be measured
  # batch_size=$((batch_size / 2)) # 2 bytes per image

	local batch_size=$((memory_free / 12)) # Adjust the divisor based on your model's memory requirements
  
	# Ensure batch size is at least 1.  But maybe 32 or 64 is better.
  if [[ $batch_size -lt 128 ]]; then
    batch_size=128
  fi

  echo "$batch_size"
}

# # Loop through topK values and run inference.py sequentially:
# for k in "${topk_values[@]}"
# do
# 	echo "Running inference.py with topK=${K} and dataset=${DATASET}"
# 	python -u inference.py -d "${DATASET}" -k $k
# 	echo "Finished running inference with topK = $k"
# 	echo "----------------------------------------"
# 	sleep 1 # Add a short delay between runs
# done
# echo "All inference runs completed"

# Loop through topK values and run inference.py in the background(Parallel):
for k in "${topk_values[@]}"
do
	device_id=$(get_max_memory_gpu) # Get device with max memory
	device_id_with_max_memory="cuda:${device_id}"
	
	# Calculate batch size dynamically based on the selected GPU's available memory
	batch_size=$(get_batch_size $device_id)
	echo "Starting inference.py with topK=${k} and dataset=${DATASET} batch_size=$batch_size in background ${device_id_with_max_memory} : $(date)"
	python -u inference.py -d "${DATASET}" -bs $batch_size -k "$k" --device $device_id_with_max_memory > "${LOGS_DIRECTORY}/inference_topK_${k}.log" 2>&1 &
	echo "==>> PID $!"
	sleep 10 # Add a short delay between runs
done

wait # Wait for all background processes to finish
echo "All inference runs have been initiated in parallel."