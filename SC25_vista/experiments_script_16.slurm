#!/bin/bash -l

#SBATCH -N 16
#SBATCH --job-name=SC25_Vista_Experiment
#SBATCH --mail-type=all
#SBATCH --mail-user=<EMAIL_ADDRESS>
#SBATCH -o outputs/O_SC25_Vista_Experiment_%j.out
#SBATCH -e outputs/E_SC25_Vista_Experiment_%j.err
#SBATCH -A <ACCOUNT>
#SBATCH -p gh
#SBATCH --time=06:00:00
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

cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/matrix-matrix $OUTPUT_FOLDER
cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/matrix-scalar $OUTPUT_FOLDER
cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/matrix-vector $OUTPUT_FOLDER
cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/temporal_data $OUTPUT_FOLDER

# Copy the shared folder (which contains dumpGpuPower.cpp and its Makefile)
cp -r $WORK/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_vista/shared $OUTPUT_FOLDER

# load necessary modules
module load nvidia/24.9
module load cuda

# NOTE: the below exports are necessary to be compatible with Makefiles in Vista
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include

echo "Current Modules:"
module list

# Clean and make the source code for the three matrix operation folders
make -C $OUTPUT_FOLDER/matrix-matrix clean
make -C $OUTPUT_FOLDER/matrix-matrix
make -C $OUTPUT_FOLDER/matrix-vector clean
make -C $OUTPUT_FOLDER/matrix-vector
make -C $OUTPUT_FOLDER/matrix-scalar clean
make -C $OUTPUT_FOLDER/matrix-scalar

# Clean and compile dumpGpuPower using the Makefile in the shared folder
make -C $OUTPUT_FOLDER/shared clean
make -C $OUTPUT_FOLDER/shared

# Create output directories in scratch
mkdir $OUTPUT_FOLDER/vista-output
mkdir $OUTPUT_FOLDER/vista-output/matrix-matrix-output
mkdir $OUTPUT_FOLDER/vista-output/matrix-matrix-cuda-output
mkdir $OUTPUT_FOLDER/vista-output/matrix-scalar-output
mkdir $OUTPUT_FOLDER/vista-output/matrix-vector-output
mkdir $OUTPUT_FOLDER/vista-output/temporal_data-output

# Create output directories /tmp for I/O
mkdir /tmp/$OUTPUT_FOLDER
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-cuda-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-scalar-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-vector-output
mkdir /tmp/$OUTPUT_FOLDER/temporal_data-output

wait

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*

# Running SPATIAL
echo 'Running Spatial Analysis...'
echo 'Running matrix-matrix'
srun  --ntasks=16 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-matrix/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix!"
sleep 60

echo 'Running matrix-matrix-cuda'
srun  --ntasks=16 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-matrix/script_cuda.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix-cuda!"
sleep 60

echo 'Running matrix-vector'
srun  --ntasks=16 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-vector/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-vector!"
sleep 60

echo 'Running matrix-scalar'
srun  --ntasks=16 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-scalar/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-scalar!"

echo 'Finished Spatial Analysis!'
sleep 60

# running TEMPORAL
echo 'Running Temporal Analysis...'
srun  --ntasks=16 --ntasks-per-node=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/temporal_data/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait

echo 'Finished Temporal Analysis!'

exit