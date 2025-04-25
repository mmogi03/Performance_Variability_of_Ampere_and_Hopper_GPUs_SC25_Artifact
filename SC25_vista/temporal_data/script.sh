#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_dir> <tmp_dir>"
    exit 1
fi

node_id=$SLURMD_NODENAME
echo "NODE ID: $node_id"

gpu_id="$SLURM_LOCALID"
gpu_uuid=$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader,nounits)
echo "NODE: $SLURMD_NODENAME | GPU ID: $gpu_id | UUID: $gpu_uuid"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

script_dir=$1
tmp_dir=$2

cd "${script_dir}/temporal_data"
formats=("BF16" "FP16" "FP32" "FP64" "FP8")

# Folder mappings for operations
small_op="matrix-scalar"
medium_op="matrix-vector"
heavy_op="matrix-matrix"
heavycuda_op="matrix-matrix-cuda"

# Temporal combinations
full_combinations=("heavy-heavy" "heavycuda-heavy" "heavy-heavycuda" "heavycuda-heavycuda" "heavy-medium" "heavycuda-medium" "heavy-small" "heavycuda-small" "medium-heavy" "medium-heavycuda" "medium-medium" "medium-small" "small-heavy" "small-heavycuda" "small-medium" "small-small")
reduced_combinations=("heavy-heavy" "heavycuda-heavy" "heavy-heavycuda" "heavycuda-heavycuda" "heavy-medium" "heavycuda-medium" "medium-heavy" "medium-heavycuda" "medium-medium")
fp8_combinations=("heavy-heavy")

# Function to determine the operation type
operation_type() {
    local op=$1
    case $op in
        heavy) echo "$heavy_op" ;;
        heavycuda) echo "$heavycuda_op" ;;
        medium) echo "$medium_op" ;;
        small) echo "$small_op" ;;
        *) echo "Unknown operation type" ;;
    esac
}

# Function to run matrix operations with GPU logging using dumpGpuPower
run_operations() {
    local format=$1
    local operation=$2
    local node_dir=$3

    # Extract operations from folder name (e.g. heavy-medium)
    IFS="-" read -r op1 op2 <<< "$operation"

    # Determine the operation types
    op1_type=$(operation_type "$op1")
    op2_type=$(operation_type "$op2")

    # Run the first operation
    echo "Running $op1_type operation for $format in $node_dir"
    log_file_op1="${node_dir}/${gpu_id}_${op1_type}_dumpGpuPower_1.csv"
    echo "Starting dumpGpuPower logging to $log_file_op1 for $op1_type"
    (
      "../shared/dumpGpuPower" -d 0 -r 0.25 -n -1 > "${log_file_op1}"
    ) &
    logging_pid_op1=$!

    if [ "$op1_type" = "matrix-scalar" ]; then
        "../${op1_type}/${format}/src/main" 1073741824 100 20 0 >> "${node_dir}/${gpu_id}.csv"
    elif [ "$op1_type" = "matrix-vector" ]; then
        "../${op1_type}/${format}/src/main" 32768 100 20 0 >> "${node_dir}/${gpu_id}.csv"
    elif [ "$op1_type" = "matrix-matrix-cuda" ]; then
        "../matrix-matrix/${format}/src/main" 32768 100 20 0 0 >> "${node_dir}/${gpu_id}.csv"
    elif [ "$op1_type" = "matrix-matrix" ]; then
        "../${op1_type}/${format}/src/main" 32768 100 20 0 1 >> "${node_dir}/${gpu_id}.csv"
    fi

    kill $logging_pid_op1
    wait $logging_pid_op1 2>/dev/null
    echo "Stopped dumpGpuPower logging for $op1_type (PID: $logging_pid_op1)"

    # Run the second operation (if any)
    if [[ -n "$op2" ]]; then
        echo "Running $op2_type operation for $format in $node_dir"
        log_file_op2="${node_dir}/${gpu_id}_${op2_type}_dumpGpuPower_2.csv"
        echo "Starting dumpGpuPower logging to $log_file_op2 for $op2_type"
        (
          "../shared/dumpGpuPower" -d 0 -r 0.25 -n -1 > "${log_file_op2}"
        ) &
        logging_pid_op2=$!

        if [ "$op2_type" = "matrix-scalar" ]; then
            "../${op2_type}/${format}/src/main" 1073741824 100 20 0 >> "${node_dir}/${gpu_id}.csv"
        elif [ "$op2_type" = "matrix-vector" ]; then
            "../${op2_type}/${format}/src/main" 32768 100 20 0 >> "${node_dir}/${gpu_id}.csv"
        elif [ "$op2_type" = "matrix-matrix-cuda" ]; then
            "../matrix-matrix/${format}/src/main" 32768 100 20 0 0 >> "${node_dir}/${gpu_id}.csv"
        elif [ "$op2_type" = "matrix-matrix" ]; then
            "../${op2_type}/${format}/src/main" 32768 100 20 0 1 >> "${node_dir}/${gpu_id}.csv"
        fi

        kill $logging_pid_op2
        wait $logging_pid_op2 2>/dev/null
        echo "Stopped dumpGpuPower logging for $op2_type (PID: $logging_pid_op2)"
    fi
}

# Loop through each format
for format in "${formats[@]}"; do
    format_dir="${format}"

    if [[ "$format" == "FP32" || "$format" == "FP64" ]]; then
        combinations=("${full_combinations[@]}")
    elif [[ "$format" == "FP8" ]]; then
        combinations=("${fp8_combinations[@]}")
    else
        combinations=("${reduced_combinations[@]}")
    fi

    # Loop through each combination of operations using an index
    for ((i=0; i<${#combinations[@]}; i++)); do
        operation="${combinations[$i]}"
        op_dir="${format_dir}/${operation}"
        # Create a GPU-specific folder inside a node-level folder in tmp:
        node_dir="${tmp_dir}/temporal_data-output/${node_id}/${gpu_id}/vista_${op_dir}"
        if [ ! -d "$node_dir" ]; then
            mkdir -p "$node_dir"
            echo "Created directory: $node_dir"
        fi

        # Run the operations and log GPU metrics for each run separately
        run_operations "$format" "$operation" "$node_dir"
        echo "Output written to: ${node_dir}/${gpu_id}.csv"

        # Sleep 30 seconds if not the last experiment
        if [ $i -lt $((${#combinations[@]} - 1)) ]; then
            echo "Sleeping for 30 seconds before next experiment..."
            sleep 30
        fi
    done
done

# Group all GPU data for this node in one folder in the destination
dest_node_dir="${script_dir}/vista-output/temporal_data-output/${node_id}"
dest_gpu_dir="${dest_node_dir}/${gpu_id}"
if [ -d "$dest_gpu_dir" ]; then
    echo "Destination directory $dest_gpu_dir exists. Removing it..."
    rm -rf "$dest_gpu_dir"
fi
mkdir -p "$dest_gpu_dir"
cp -r "$tmp_dir/temporal_data-output/${node_id}/${gpu_id}" "$dest_gpu_dir"
echo "Copied node $node_id, GPU $gpu_id results to $dest_gpu_dir"
