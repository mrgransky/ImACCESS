#!/bin/bash

## run [local] server:
## $ nohup bash run_inference.sh > prec_at_k.out 2>&1 &

## run [Pouta] server:
## $ nohup bash run_inference.sh > /dev/null 2>&1 &
## $ nohup bash run_inference.sh > /media/volume/ImACCESS/trash/prec_at_k.out 2>&1 &

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
# topk_values=(20 15 10 5 1)
topk_values=(1)
DATASET="imagenet"
# batch_size=1024
number_of_workers=40
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
		local total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $device_id)

		# Error checking
		if [[ -z "$memory_free" || ! "$memory_free" =~ ^[0-9]+$ ]]; then
				echo "Error: Could not retrieve free memory for GPU $device_id"
				exit 1
		fi
		# Reserve some memory for CUDA runtime and other overhead (e.g., 1GB = 1024)
		local reserved_memory=1024
		local usable_memory=$((memory_free - reserved_memory))

		# Approximate memory per sample (in MB) - adjust these values based on your model
		local memory_per_sample=16  # Example: 16MB per sample

		# Calculate maximum possible batch size based on available memory
		local max_batch_size=$((usable_memory / memory_per_sample))

		# Define batch size constraints
		local min_batch_size=128
		local max_allowed_batch_size=2048  # Upper limit to prevent excessive memory usage

		# Apply constraints and round to nearest power of 2
		local final_batch_size
		if ((max_batch_size < min_batch_size)); then
				final_batch_size=$min_batch_size
		elif ((max_batch_size > max_allowed_batch_size)); then
				final_batch_size=$max_allowed_batch_size
		else
				# Round to nearest power of 2
				local power=1
				while ((power * 2 <= max_batch_size)); do
						power=$((power * 2))
				done
				final_batch_size=$power
		fi

		# # Add logging lines here
		# echo "GPU $device_id: Free Memory: ${memory_free}MB, Usable: ${usable_memory}MB"
		# echo "Calculated batch size: $final_batch_size (rounded from $max_batch_size)"

		# Return the final batch size
		echo "$final_batch_size"
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
	device_with_max_mem="cuda:${device_id}"
	
	# Calculate batch size dynamically based on the selected GPU's available memory
	batch_size=$(get_batch_size $device_id)
	echo "Starting inference.py with topK=${k} and dataset=${DATASET} batch_size=$batch_size in background ${device_with_max_mem} : $(date)"
	python -u inference.py -d "${DATASET}" -nw $number_of_workers -bs $batch_size -k "$k" --device $device_with_max_mem > "${LOGS_DIRECTORY}/inference_topK_${k}.log" 2>&1 &
	echo "==>> PID $!"
	sleep 10 # Add a short delay between runs
done

wait # Wait for all background processes to finish
echo "All inference runs have been initiated in parallel."