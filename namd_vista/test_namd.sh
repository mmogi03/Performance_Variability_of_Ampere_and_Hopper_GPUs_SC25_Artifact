#!/bin/bash

# Check if both parameters are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <script_dir> <tmp_dir>"
    exit 1
fi

node_id=$SLURMD_NODENAME
gpu_id=$SLURM_LOCALID

echo "NODE ID: $node_id"
echo "GPU ID: $gpu_id"

script_dir=$1
tmp_dir=$2

# Create a GPU-specific folder inside a node-level folder in tmp:
node_dir="${tmp_dir}/namd3-output/${node_id}/${gpu_id}"
if [ ! -d "$node_dir" ]; then
    mkdir -p "$node_dir"
    echo "Created directory: $node_dir"
fi

# Start dumpGpuPower logging in the background (only if the executable exists)
gpu_log_file="${node_dir}/gpu_power_${gpu_id}.csv"
dump_gpu_power_exec="${script_dir}/shared/dumpGpuPower"

# Debugging: Check if dumpGpuPower exists and is executable
if [ -x "$dump_gpu_power_exec" ]; then
    echo "Starting dumpGpuPower logging to $gpu_log_file"
    (
      "$dump_gpu_power_exec" -d 0 -r 0.25 -n -1 > "$gpu_log_file"
    ) &
    logging_pid=$!
else
    echo "Warning: dumpGpuPower not found or not executable. Skipping GPU power logging."
    logging_pid=0
fi

# Define output file for NAMD3
output_file="${node_dir}/${gpu_id}.log"

# Ensure OUTPUT_FOLDER is set
if [ -z "$OUTPUT_FOLDER" ]; then
    echo "Error: OUTPUT_FOLDER is not set."
    exit 1
fi

# Debugging: List directory contents before running NAMD3
echo "Listing contents of OUTPUT_FOLDER before running NAMD3:"
ls -lR $OUTPUT_FOLDER

# Check if the NAMD3 configuration file exists
config_file="$OUTPUT_FOLDER/2x2x2stmv.namd"
if [ ! -f "$config_file" ]; then
    echo "Error: NAMD3 configuration file not found at $config_file"
    exit 1
fi

echo "Running NAMD3 simulation..."

# --- Time the NAMD3 Execution ---
start_time=$(date +%s.%N)

/opt/apps/nvidia24/openmpi5/namd-gpu/3.0/bin/namd3 +setcpuaffinity +idlepoll +p8 +devices 0 $OUTPUT_FOLDER/2x2x2stmv.namd > "$output_file"

end_time=$(date +%s.%N)

# Calculate elapsed time
elapsed_time=$(echo "$end_time - $start_time" | bc)

echo "NAMD3 output written to: $output_file"
echo "Execution time: $elapsed_time seconds"

# Terminate the background dumpGpuPower logging process if it was started
if [ "$logging_pid" -ne 0 ]; then
    kill $logging_pid 2>/dev/null
    wait $logging_pid 2>/dev/null
fi

# --- Final Copy: Group all GPU data for this node ---
dest_node_dir="${script_dir}/vista-output/namd3-output/${node_id}"
dest_gpu_dir="${dest_node_dir}/${gpu_id}"

# Remove old destination directory if it exists
if [ -d "$dest_gpu_dir" ]; then
    echo "Destination directory $dest_gpu_dir exists. Removing it..."
    rm -rf "$dest_gpu_dir"
fi

# Create destination directory
mkdir -p "$dest_gpu_dir"

# Copy results from temporary location back to the original location
cp -r "$node_dir" "$dest_gpu_dir"

# Write execution time to a CSV file
runtime_csv="${dest_node_dir}/namd3_runtime.csv"
echo "$node_id,$gpu_id,$elapsed_time" >> "$runtime_csv"

echo "Copied node $node_id, GPU $gpu_id results to $dest_gpu_dir"
echo "Execution time logged in $runtime_csv"
