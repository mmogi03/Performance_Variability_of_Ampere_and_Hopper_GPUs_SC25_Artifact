#!/bin/bash -l

#SBATCH --nodes=28
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
#SBATCH --time=02:00:00
#SBATCH --exclude=gh003,gh004,gh006,gh007,gh008,gh009,gh010,gh011,gh012,gh020,gh021,gh022,gh023,gh024,gh025,gh026,gh027,gh028,gh030,gh032,gh033,gh035,gh036,gh037,gh038,gh039,gh041,gh044,gh047,gh049,gh050,gh051,gh053,gh054,gh055,gh057,gh058,gh060,gh063,gh064,gh065,gh066,gh067,gh068,gh069,gh071,gh073,gh074,gh075,gh076,gh077,gh078,gh081,gh082,gh083,gh087,gh088,gh089,gh090,gh091,gh092,gh093,gh097,gh103,gh104,gh105,gh106,gh107,gh110,gh111,gh112,gh113,gh116,gh118,gh120,gh121,gh122,gh123,gh126,gh127,gh128,gh130,gh131,gh132,gh135,gh137,gh139,gh142,gh143,gh146,gh147,gh148,gh149,gh150


NODELIST=$SLURM_NODELIST
echo "Node(s) allocated to the job: $NODELIST"

# stage executable and data
cd /work/hdd/bcrc/$USER

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

cp -r /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/matrix-matrix $OUTPUT_FOLDER
cp -r /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/matrix-scalar $OUTPUT_FOLDER
cp -r /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/matrix-vector $OUTPUT_FOLDER
cp -r /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/temporal_data $OUTPUT_FOLDER

# Copy the shared folder (which contains dumpGpuPower.cpp and its Makefile)
cp -r /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/shared $OUTPUT_FOLDER

# load necessary modules
module load cudnn
module load cuda
# module load cdtools

# NOTE: the below exports are necessary to be compatible with Makefiles in dtai
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include

echo "Current Modules:"
module list

# make the source code for the three matrix operation folders
make -C $OUTPUT_FOLDER/matrix-matrix
make -C $OUTPUT_FOLDER/matrix-vector
make -C $OUTPUT_FOLDER/matrix-scalar

# Compile dumpGpuPower using the Makefile in the shared folder
make -C $OUTPUT_FOLDER/variations/SC25_dtai/shared

# Create output directories in scratch after running scripts
mkdir $OUTPUT_FOLDER/dtai-output
mkdir $OUTPUT_FOLDER/dtai-output/matrix-matrix-output
mkdir $OUTPUT_FOLDER/dtai-output/matrix-scalar-output
mkdir $OUTPUT_FOLDER/dtai-output/matrix-vector-output
mkdir $OUTPUT_FOLDER/dtai-output/temporal_data-output

# Stage executables and files from scratch to /tmp for I/O
mkdir /tmp/$OUTPUT_FOLDER
mkdir /tmp/$OUTPUT_FOLDER/matrix-matrix-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-scalar-output
mkdir /tmp/$OUTPUT_FOLDER/matrix-vector-output
mkdir /tmp/$OUTPUT_FOLDER/temporal_data-output

wait

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*

# Running SPATIAL
echo 'Running Spatial Analysis...'
echo 'Running matrix-matrix'
srun  --ntasks=112 --ntasks-per-node=4 /bin/bash /work/hdd/bcrc/$USER/$OUTPUT_FOLDER/matrix-matrix/script.sh /work/hdd/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-matrix!"

echo 'Running matrix-vector'
srun  --ntasks=112 --ntasks-per-node=4 /bin/bash /work/hdd/bcrc/$USER/$OUTPUT_FOLDER/matrix-vector/script.sh /work/hdd/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-vector!"

echo 'Running matrix-scalar'
srun  --ntasks=112 --ntasks-per-node=4 /bin/bash /work/hdd/bcrc/$USER/$OUTPUT_FOLDER/matrix-scalar/script.sh /work/hdd/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
wait
echo "Finished matrix-scalar!"

echo 'Finished Spatial Analysis!'
sleep 60

# # running TEMPORAL
# echo 'Running Temporal Analysis...'
# srun  --ntasks=112 --ntasks-per-node=4 /bin/bash /work/hdd/bcrc/$USER/$OUTPUT_FOLDER/temporal_data/script.sh /work/hdd/bcrc/$USER/$OUTPUT_FOLDER /tmp/$OUTPUT_FOLDER
# wait

# echo 'Finished Temporal Analysis!'

# # Directory to check
TARGET_DIR="/projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/dtai_output/$OUTPUT_FOLDER"

# Check if the folder exists and remove it
if [ -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR exists. Removing it..."
    rm -rf "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR does not exist yet."
fi

mkdir $TARGET_DIR

cp -r $OUTPUT_FOLDER/dtai-output /projects/bcrc/$USER/$REPO_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/dtai_outputs/$OUTPUT_FOLDER

exit