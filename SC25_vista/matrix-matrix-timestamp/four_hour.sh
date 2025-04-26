#!/bin/bash

# Check if both parameters are provided
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

cd "${script_dir}/matrix-matrix-timestamp"
formats=("BF16" "FP64")

# Loop through each format
for format in "${formats[@]}"; do
    src_dir="${format}/src"
    data_dir="${format}/data"

    # Create a GPU-specific folder inside a node-level folder in tmp:
    node_dir="${tmp_dir}/matrix-matrix-timestamp-output/${node_id}/${gpu_id}/vista_${data_dir}"
    if [ ! -d "$node_dir" ]; then
        mkdir -p "$node_dir"
        echo "Created directory: $node_dir"
    fi

    # Start dumpGpuPower logging in the background
    gpu_log_file="${node_dir}/gpu_power_${gpu_id}.csv"
    echo "Starting dumpGpuPower logging to $gpu_log_file"
    (
      "../shared/dumpGpuPower" -d 0 -r 0.25 -n -1 > "${gpu_log_file}"
    ) &
    logging_pid=$!

    output_file="${node_dir}/${gpu_id}.csv"
    
    echo "MAIN: ${src_dir}/main"
    echo "OUTPUT: ${output_file}"

    # Call main with different parameters based on the format
    if [ "$format" = "BF16" ]; then
        # Parameters for BF16, adjust these values to run as long as needed (default for BF16 is to run for 4 hours):
        "${src_dir}/main" 32768 130909 0 0 1 > "$output_file"
    elif [ "$format" = "FP64" ]; then
        # Parameters for FP64, adjust these values to run as long as needed (default for FP64 is to run for 4 hours):
        "${src_dir}/main" 32768 9000 0 0 1 > "$output_file"
    else
        echo "Unknown format: $format"
    fi
    echo "Output written to: $output_file"
    
    # Terminate the background dumpGpuPower logging process
    kill $logging_pid
    wait $logging_pid 2>/dev/null
    echo "Stopped dumpGpuPower logging (PID: $logging_pid)"
done

# Group all GPU data for this node
dest_node_dir="${script_dir}/vista-output/matrix-matrix-timestamp-output/${node_id}"
dest_gpu_dir="${dest_node_dir}/${gpu_id}"
if [ -d "$dest_gpu_dir" ]; then
    echo "Destination directory $dest_gpu_dir exists. Removing it..."
    rm -rf "$dest_gpu_dir"
fi
mkdir -p "$dest_gpu_dir"
cp -r "$tmp_dir/matrix-matrix-timestamp-output/${node_id}/${gpu_id}" "$dest_gpu_dir"
echo "Copied node $node_id, GPU $gpu_id results to $dest_gpu_dir"
