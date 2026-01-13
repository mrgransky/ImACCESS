#!/bin/bash

# how to run:
# nohup bash get_history_xN.sh > logs/all_data_collectors.out &

# Script to run all dataset collectors for ImACCESS project
# Author: ImACCESS Team, Tampere University
# Date: 2024-2025

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
	echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
	echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
	echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
	echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to run a data collector
run_collector() {
	local dataset_name=$1
	local collector_path=$2
	local args=$3
	
	print_status "Starting collection for: ${dataset_name}"
	
	if [ ! -f "$collector_path" ]; then
		print_error "Collector not found: $collector_path"
		return 1
	fi
	
	# Create logs directory if it doesn't exist
	local log_dir=$(dirname "$collector_path")/logs
	mkdir -p "$log_dir"
	
	local log_file="$log_dir/${dataset_name}_$(date +'%Y%m%d_%H%M%S').log"
	
	print_status "Running: python $collector_path $args"
	print_status "Log file: $log_file"
	
	if python -u "$collector_path" $args 2>&1 | tee "$log_file"; then
		print_success "Completed: ${dataset_name}"
		return 0
	else
		print_error "Failed: ${dataset_name}"
		return 1
	fi
}

# Main execution
main() {
	print_status "=================================================="
	print_status "ImACCESS Dataset Collection Pipeline"
	print_status "Tampere University, Finland"
	print_status "=================================================="
	
	# Get the base directory (where this script is located)
	SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	BASE_DIR="${SCRIPT_DIR}"
	
	# Default dataset directory (can be overridden with -d flag)
	if [[ "$(hostname)" == "gpu-vm" ]] || [[ "$USER" == "ubuntu" ]]; then
		DATASET_DIR="/media/volume/ImACCESS/datasets/WW_DATASETs"
	else
		DATASET_DIR="${HOME}/datasets/WW_DATASETs"
	fi

	# Parse command line arguments
	while getopts "d:h" opt; do
		case $opt in
			d)
				DATASET_DIR="$OPTARG"
				;;
			h)
				echo "Usage: $0 [-d DATASET_DIR] [-h]"
				echo "  -d DATASET_DIR  Specify the output directory for datasets (default: ~/datasets/WW_DATASETs or /media/volume/ImACCESS/datasets/WW_DATASETs)"
				echo "  -h              Show this help message"
				exit 0
				;;
			\?)
				print_error "Invalid option: -$OPTARG"
				exit 1
				;;
		esac
	done
	
	print_status "Output directory: ${DATASET_DIR}"
	mkdir -p "$DATASET_DIR"
	
	# Array to track results
	declare -a RESULTS
	
	# Collection configuration
	# Format: "dataset_name|collector_path|arguments"
	COLLECTORS=(
		"SMU|${BASE_DIR}/smu/data_collector.py|-ddir ${DATASET_DIR} -nw 16 --img_mean_std --thumbnail_size 512,512 --verbose"
		"WWII|${BASE_DIR}/wwii/data_collector.py|-ddir ${DATASET_DIR} -nw 8 --img_mean_std --thumbnail_size 512,512 --verbose"
		"National_Archive|${BASE_DIR}/national_archive/data_collector.py|-ddir ${DATASET_DIR} -nw 4 --img_mean_std --thumbnail_size 512,512 --verbose"
		"Europeana|${BASE_DIR}/europeana/data_collector.py|-ddir ${DATASET_DIR} -nw 8 --img_mean_std --thumbnail_size 512,512 --api_key api2demo --verbose"
		# "SA_Kuva|${BASE_DIR}/sa_kuva/data_collector.py|-ddir ${DATASET_DIR} -nw 12 --img_mean_std --thumbnail_size 512,512"
		# "WW_Vehicles|${BASE_DIR}/ww_vehicles/data_collector.py|-ddir ${DATASET_DIR} -nw 12 --img_mean_std --thumbnail_size 512,512"
	)
	
	print_status "Total datasets to collect: ${#COLLECTORS[@]}"
	echo ""
	
	# Run each collector
	for collector_config in "${COLLECTORS[@]}"; do
		IFS='|' read -r name path args <<< "$collector_config"
		
		echo ""
		print_status "=================================================="
		
		if run_collector "$name" "$path" "$args"; then
			RESULTS+=("${GREEN}✓${NC} $name: SUCCESS")
		else
			RESULTS+=("${RED}✗${NC} $name: FAILED")
		fi
		
		echo ""
	done
	
	echo ""
	print_status "=================================================="
	print_status "COLLECTION SUMMARY"
	print_status "=================================================="
	
	for result in "${RESULTS[@]}"; do
		echo -e "  $result"
	done
	
	echo ""
	print_status "All dataset collection tasks completed!"
	print_status "Output directory: ${DATASET_DIR}"

	print_status "Running dataset merger => History_xN ..."

	python -u "${BASE_DIR}/history_xN/merge_datasets.py" -ddir "${DATASET_DIR}"	--img_mean_std --target_chunk_mb 13 -nw 4 -v
	print_success "Pipeline execution completed!"
}

# Run main function
main "$@"