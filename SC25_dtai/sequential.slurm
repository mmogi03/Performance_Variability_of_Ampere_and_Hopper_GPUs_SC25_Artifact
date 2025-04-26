#!/bin/bash -l

#SBATCH --nodes=5
#SBATCH --job-name=deltaAI_test
#SBATCH --mail-type=all
#SBATCH --mail-user=hgz5@scarletmail.rutgers.edu
#SBATCH --output=outputs/dtai_output.out
#SBATCH --error=outputs/dtai_errors.out
#SBATCH --account=bdrw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00

# Debug: Print allocated nodes
NODELIST=$SLURM_NODELIST
echo "Node(s) allocated to the job: $NODELIST"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_NODELIST"
echo "SLURM Task Count: $SLURM_NTASKS"

# Stage executable and data
echo "Changing directory to /work/hdd/bcrc/$USER"
cd /work/hdd/bcrc/$USER

# Directory to check
TARGET_DIR="$OUTPUT_FOLDER"

echo "Checking if $TARGET_DIR exists..."
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR exists. Removing it..."
    rm -rf "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR does not exist yet."
fi

echo "Creating directory: $TARGET_DIR"
mkdir $TARGET_DIR

# Debug: Copy operations
echo "Copying necessary files to $OUTPUT_FOLDER..."
cp -r /projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/matrix-matrix $OUTPUT_FOLDER && echo "Copied matrix-matrix"
cp -r /projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/matrix-scalar $OUTPUT_FOLDER && echo "Copied matrix-scalar"
cp -r /projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/matrix-vector $OUTPUT_FOLDER && echo "Copied matrix-vector"
cp -r /projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/temporal_data $OUTPUT_FOLDER && echo "Copied temporal_data"
cp -r /projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/shared $OUTPUT_FOLDER && echo "Copied shared folder"

# Load necessary modules
echo "Loading required modules..."
module load cudnn && echo "Loaded cudnn"
module load cuda && echo "Loaded cuda"
# module load cdtools

# Debug: Check module list
echo "Current Modules:"
module list

# Set environment variables
echo "Setting environment variables..."
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
echo "LD_LIBRARY_PATH set to $LD_LIBRARY_PATH"
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include
echo "INCLUDE set to $INCLUDE"

# Compile source code
echo "Compiling source code..."
make -C $OUTPUT_FOLDER/matrix-matrix clean && make -C $OUTPUT_FOLDER/matrix-matrix && echo "Compiled matrix-matrix"
make -C $OUTPUT_FOLDER/matrix-vector clean && make -C $OUTPUT_FOLDER/matrix-vector && echo "Compiled matrix-vector"
make -C $OUTPUT_FOLDER/matrix-scalar clean && make -C $OUTPUT_FOLDER/matrix-scalar && echo "Compiled matrix-scalar"
make -C $OUTPUT_FOLDER/shared clean && make -C $OUTPUT_FOLDER/shared && echo "Compiled shared folder"

# Create output directories
echo "Creating output directories..."
mkdir -p $OUTPUT_FOLDER/dtai-output/matrix-matrix-output
mkdir -p $OUTPUT_FOLDER/dtai-output/matrix-matrix-cuda-output
mkdir -p $OUTPUT_FOLDER/dtai-output/matrix-scalar-output
mkdir -p $OUTPUT_FOLDER/dtai-output/matrix-vector-output
mkdir -p $OUTPUT_FOLDER/dtai-output/temporal_data-output
echo "Output directories created."

# Stage executables and files from scratch to /tmp for I/O
echo "Staging executables and files to /tmp/$OUTPUT_FOLDER..."
mkdir -p /tmp/$OUTPUT_FOLDER/matrix-matrix-output
mkdir -p /tmp/$OUTPUT_FOLDER/matrix-scalar-output
mkdir -p /tmp/$OUTPUT_FOLDER/matrix-vector-output
mkdir -p /tmp/$OUTPUT_FOLDER/temporal_data-output
mkdir -p /tmp/$OUTPUT_FOLDER/matrix-matrix-cuda-output
echo "Staging complete."

wait

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
echo "CUDA_VISIBLE_DEVICES set to $CUDA_VISIBLE_DEVICES"
exec $*
chmod -R +x /work/hdd/bcrc/hzaky/sequential_1/

# Running SPATIAL
echo 'Running Spatial Analysis...'
echo 'Running matrix-matrix'

for gpu_id in {0..3}; do
    echo "Running on GPU $gpu_id"
    srun --ntasks=5 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_id; /projects/bcrc/$USER/$OUTPUT_FOLDER/matrix-matrix/script2.sh /projects/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER"
    wait
done
echo "Finished matrix-matrix!"
sleep 60
echo 'Running matrix-matrix-cuda'
for gpu_id in {0..3}; do
    echo "Running on GPU $gpu_id"
    srun --ntasks=5 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_id;  /projects/bcrc/$USER/$OUTPUT_FOLDER/matrix-matrix/script_cuda2.sh /projects/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER"
    wait
done
echo "Finished matrix-matrix-cuda!"
sleep 60
echo 'Running matrix-vector'
for gpu_id in {0..3}; do
    echo "Running on GPU $gpu_id"
    srun --ntasks=5 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_id; /projects/bcrc/$USER/$OUTPUT_FOLDER/matrix-vector/script2.sh /projects/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER"
    wait
done
echo "Finished matrix-vector!"
sleep 60
echo 'Running matrix-scalar'
for gpu_id in {0..3}; do
    echo "Running on GPU $gpu_id"
    srun --ntasks=5 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_id; /projects/bcrc/$USER/$OUTPUT_FOLDER/matrix-scalar/script2.sh /projects/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER"
    wait
done
echo "Finished matrix-scalar!"
echo 'Finished Spatial Analysis!'
sleep 60

# # Running TEMPORAL
# echo 'Running Temporal Analysis...'
# for gpu_id in {0..3}; do
#     echo "Running on GPU $gpu_id"
#     srun --ntasks=5 --ntasks-per-node=1 --gpus-per-task=1 --gpu-bind=map_gpu:$gpu_id /bin/bash -c "export CUDA_VISIBLE_DEVICES=$gpu_id; /work/hdd/bcrc/$USER/$OUTPUT_FOLDER/temporal_data/script2.sh /work/hdd/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER"
#     wait
# done
# echo 'Finished Temporal Analysis!'

# Copy results back
echo "Copying results back to permanent storage..."
DESTINATION_DIR="/projects/bcrc/$USER/$REPO_FOLDER/GPU-Bottleneck-Deep-Learning-at-Scale/variations/SC25_dtai/dtai_outputs/$OUTPUT_FOLDER"
echo "Checking if $DESTINATION_DIR exists..."
if [ -d "$DESTINATION_DIR" ]; then
    echo "Directory $DESTINATION_DIR exists. Ignoring creation..."
else
    echo "Directory $DESTINATION_DIR does not exist yet. Creating new directory..."
    mkdir -p $DESTINATION_DIR
fi

cp -r $OUTPUT_FOLDER/dtai-output $DESTINATION_DIR && echo "Results copied successfully!"
exit
