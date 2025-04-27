#!/bin/bash -l

#SBATCH -N 2
#SBATCH -C gpu&hbm40g
#SBATCH --job-name=SC25_Perlmutter_Experiment
#SBATCH --mail-type=all
#SBATCH --mail-user=mm3201@scarletmail.rutgers.edu
#SBATCH -o outputs/O_SC25_Perlmutter_Experiment_%j.out
#SBATCH -e outputs/E_SC25_Perlmutter_Experiment_%j.err
#SBATCH --account=m4431_g
#SBATCH -q regular
#SBATCH --time=07:00:00
#SBATCH --ntasks-per-node=4

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

cp -r $HOME/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_perlmutter/matrix-matrix $OUTPUT_FOLDER
cp -r $HOME/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_perlmutter/matrix-scalar $OUTPUT_FOLDER
cp -r $HOME/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_perlmutter/matrix-vector $OUTPUT_FOLDER
# cp -r $HOME/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_perlmutter/temporal_data $OUTPUT_FOLDER

# Copy the shared folder (which contains dumpGpuPower.cpp and its Makefile)
cp -r $HOME/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_perlmutter/shared $OUTPUT_FOLDER

# load necessary modules
module load cudatoolkit
module load cudnn

# NOTE: the below exports are necessary to be compatible with Makefiles in Perlmutter
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64
export INCLUDE=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/include

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

# Create output directories in scratch after running scripts
mkdir $OUTPUT_FOLDER/perlmutter-output
mkdir $OUTPUT_FOLDER/perlmutter-output/matrix-matrix-output
mkdir $OUTPUT_FOLDER/perlmutter-output/matrix-matrix-cuda-output
mkdir $OUTPUT_FOLDER/perlmutter-output/matrix-scalar-output
mkdir $OUTPUT_FOLDER/perlmutter-output/matrix-vector-output
# mkdir $OUTPUT_FOLDER/perlmutter-output/temporal_data-output

# Stage executables and files from scratch to /tmp for I/O
mkdir /tmp/$OUTPUT_FOLDER
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-cuda-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-scalar-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-vector-output
# mkdir /tmp/$OUTPUT_FOLDER/temporal_data-output

wait

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*

# Running SPATIAL
echo 'Running Spatial Analysis...'
echo 'Running matrix-matrix'
srun  --ntasks=8 --gpus-per-task=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-matrix/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix!"
sleep 60

echo 'Running matrix-matrix-cuda'
srun  --ntasks=8 --gpus-per-task=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-matrix/script_cuda.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix-cuda!"
sleep 60

echo 'Running matrix-vector'
srun  --ntasks=8 --gpus-per-task=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-vector/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-vector!"
sleep 60

echo 'Running matrix-scalar'
srun  --ntasks=8 --gpus-per-task=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/matrix-scalar/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-scalar!"

echo 'Finished Spatial Analysis!'
sleep 60

# running TEMPORAL
# echo 'Running Temporal Analysis...'
# srun  --ntasks=256 --gpus-per-task=1 /bin/bash $SCRATCH/$OUTPUT_FOLDER/temporal_data/script.sh $SCRATCH/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
# wait

# echo 'Finished Temporal Analysis!'

exit