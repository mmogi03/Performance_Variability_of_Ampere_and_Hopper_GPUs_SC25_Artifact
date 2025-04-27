#!/bin/bash -l

#SBATCH -N 2
#SBATCH --job-name=SC25_Vista_FourHour_NEW
#SBATCH --mail-type=all
#SBATCH --mail-user=mm3201@scarletmail.rutgers.edu
#SBATCH -o outputs/O_SC25_Vista_FourHour_%j.out
#SBATCH -e outputs/E_SC25_Vista_FourHour_%j.err
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH --time=09:00:00
#SBATCH --ntasks-per-node=1

NODELIST=$SLURM_NODELIST
echo "Node(s) allocated to the job: $NODELIST"

# stage executable and data
cd $SCRATCH

# Directory to check
TARGET_DIR="$OUTPUT_FOLDER"

# Check if the folder exists and remove it
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR exists. Removing it..."
    rm -rf "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR does not exist yet."
fi
    
mkdir $TARGET_DIR

cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/matrix-matrix-fourhour $OUTPUT_FOLDER

# Copy the shared folder (which contains dumpGpuPower.cpp and its Makefile)
cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/shared $OUTPUT_FOLDER

# load necessary modules
module load nvidia/24.9
module load cuda
# module load cdtools

# NOTE: the below exports are necessary to be compatible with Makefiles in Vista
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include

echo "Current Modules:"
module list

# Clean and make the source code for the three matrix operation folders
make -C $OUTPUT_FOLDER/matrix-matrix-fourhour clean
make -C $OUTPUT_FOLDER/matrix-matrix-fourhour

# Clean and compile dumpGpuPower using the Makefile in the shared folder
make -C $OUTPUT_FOLDER/shared clean
make -C $OUTPUT_FOLDER/shared

# Create output directories in scratch after running scripts
mkdir $OUTPUT_FOLDER/vista-output
mkdir $OUTPUT_FOLDER/vista-output/matrix-matrix-fourhour-output

# Make directories in /tmp for I/O
mkdir /tmp/$OUTPUT_FOLDER
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-fourhour-output

wait

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*

# Running SPATIAL
echo 'Running Spatial Analysis...'
echo 'Running matrix-matrix-fourhour'
srun  --ntasks=2 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-matrix-fourhour/four_hour.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix-fourhour!"

exit