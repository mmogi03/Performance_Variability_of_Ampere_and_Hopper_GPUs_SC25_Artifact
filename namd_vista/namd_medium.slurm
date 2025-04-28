#!/bin/bash

#SBATCH -N 5
#SBATCH -n 5            # Total number of mpi tasks
#SBATCH --job-name=vista_namd_slowest
#SBATCH --mail-type=all
#SBATCH --mail-user=<EMAIL_ADDRESS>
#SBATCH -o outputs/O_SC25_Vista_NAMD_Experiment_MEDIUM_%j.out
#SBATCH -e outputs/E_SC25_Vista_NAMD_Experiment_MEDIUM_%j.out
#SBATCH -A <ACCOUNT>
#SBATCH -p gh
#SBATCH --time=00:30:00
#SBATCH --nodelist=c636-041,c640-061,c621-002,c635-081


NODELIST=$SLURM_NODELIST
echo "Node(s) allocated to the job: $NODELIST"

# Stage executable and data
cd $WORK

echo "OUTPUT_FOLDER is set to: $OUTPUT_FOLDER"

# Ensure OUTPUT_FOLDER is set
if [ -z "$OUTPUT_FOLDER" ]; then
    echo "ERROR: OUTPUT_FOLDER is not set."
    exit 1
fi

TARGET_DIR="$OUTPUT_FOLDER"

# Check if the directory exists and remove it if necessary
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR exists. Removing it..."
    rm -rf "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR does not exist yet."
fi
    
mkdir -p "$TARGET_DIR"

cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/shared $OUTPUT_FOLDER
cp $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/namd_vista/namd.sh $OUTPUT_FOLDER/
cp $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/namd_vista/2x2x2stmv.namd $OUTPUT_FOLDER/

# Load necessary modules
module load nvidia/24.9
module load cuda
module use /opt/apps/nvidia24/openmpi5/modulefiles
module use /opt/apps/nvidia24/modulefiles
module use /opt/apps/modulefiles
module load namd-gpu/3.0
which namd3

# NOTE: The below exports are necessary to be compatible with Makefiles in Vista
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include

echo "Current Modules:"
module list

# Compile dumpGpuPower using the Makefile in the shared folder
echo "Checking if dumpGpuPower exists before compilation..."
ls -l $OUTPUT_FOLDER/shared

make -C $OUTPUT_FOLDER/shared

if [ ! -f "$OUTPUT_FOLDER/shared/dumpGpuPower" ]; then
    echo "ERROR: dumpGpuPower was not compiled successfully!"
    exit 1
fi

chmod +x $OUTPUT_FOLDER/shared/dumpGpuPower
echo "dumpGpuPower is now executable."

# Debugging: List directory contents
ls -lR $OUTPUT_FOLDER

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Ensure namd.sh exists
if [ ! -f "$WORK/$OUTPUT_FOLDER/namd.sh" ]; then
    echo "ERROR: namd.sh not found in $WORK/$OUTPUT_FOLDER"
    exit 1
fi

# Running namd.sh
echo 'Running namd.sh...'
srun --ntasks=4 --ntasks-per-node=1 /bin/bash $WORK/$OUTPUT_FOLDER/namd.sh $WORK/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished namd.sh!"

exit
