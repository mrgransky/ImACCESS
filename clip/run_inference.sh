#!/bin/bash

## run using command:
## $ nohup bash run_inference.sh > /dev/null 2>&1 &
## $ nohup bash run_inference.sh > /media/volume/ImACCESS/trash/run_inference_logs.out 2>&1 &

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error and exit immediately.
set -o pipefail # If any command in a pipeline fails, the entire pipeline will fail.

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"

source $(conda info --base)/bin/activate py39

topk_values=(1 5 10 15 20)
DATASET="imagenet"
batch_size=2048
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
	echo "Starting inference.py with topK=${k} and dataset=${DATASET} in the background"
	python -u inference.py -d "${DATASET}" -bs $batch_size -k "$k" > "inference_topK_${k}.log" 2>&1 &
	echo "Started inference with topK = ${k} with PID $!"
done

wait # Wait for all background processes to finish
echo "All inference runs have been initiated in parallel."