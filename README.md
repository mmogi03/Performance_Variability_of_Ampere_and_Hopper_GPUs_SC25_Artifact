# Experiments Ran in "Performance Variability of Ampere and Hopper GPUs" Paper
This artifact contains code to reproduce the exeriments performed in "Performance Variability of Ampere and Hopper GPUs". The repository is organized into directories for each experiment. See [Experiments](#experiments) below for a summary of where the contributions are located in each directory, and [Compile and Run](#compile-and-run) for instructions on how to reproduce our experiments.

## Table of Contents

- [Experiments](#experiments)
- [Compile and Run](#compile-and-run)
- [Related Documentation](#related-documentation)

## Experiments

There were 8 experiments we ran in our paper:
- **Micro-benchmarks on NERSC Perlmutter (A100s)**: a series of micro-benchmarks we wrote for NVIDIA's A100s on NERSC Perlmutter (4 GPUs per node) that utilize NVIDIA's cuBLAS library to perform matrix-matrix, matrix-vector, and vector-scalar operations using varying precisions (BF16, FP16, FP32, FP64).
- **Micro-benchmarks on TACC Vista (GH200s)**: a series of micro-benchmarks we wrote for NVIDIA's GH200s on TACC Vista (1 GPU per node) that utilize NVIDIA's cuBLAS library to perform matrix-matrix, matrix-vector, and vector-scalar operations using varying precisions (BF16, FP16, FP32, FP64, FP8).
- **Micro-benchmarks on TACC Vista (GH200s)**: a series of micro-benchmarks we wrote for NVIDIA's GH200s on NCSA DeltaAI (4 GPUs per node) that utilize NVIDIA's cuBLAS library to perform matrix-matrix, matrix-vector, and vector-scalar operations using varying precisions (BF16, FP16, FP32, FP64, FP8).
- **Timestamp for Iteration Mapping on TACC Vista (GH200s)**: an extension of the micro-benchmarks we wrote for mapping the three phases of a benchmarked kernel to precise timestamps (specifically on NVIDIA's GH200s on TACC Vista running matrix-matrix tensor operations with FP32 precision).
- **NAMD Benchmarks**: FILL THIS IN
- **GPT-4.8B Training Benchmarks**: FILL THIS IN
- **Dedicated Micro-benchmarks on NCSA DeltaAI (GH200s)**: an extension of the micro-benchmarks we wrote, except it runs on 1 GPU at a time on NVIDIA's GH200s on NCSA DeltaAI.
- **Temporal Benchmark on TACC Vista (GH200s)**: an extension of the micro-benchmarks we wrote, running matrix-matrix tensor core operations with BF16 and FP64 precision on NVIDIA's GH200s on TACC Vista for four hours.

For each experiment, there is an associated directory in this repository. `SC25_perlmutter`, `SC25_vista`, and `SC25_dtai` contain the code for the micro-benchmarking experiments (note that `SC25_vista` also contains code for running the timestamp and temporal benchmarking experiments, and `SC25_dtai` also contains code for dedicated micro-benchmarks). `namd_vista` contains the code for running the NAMD benchmarks. `gpt_vista` contains the code for benchmarking the training of GPT-4.8B.

### Compile and Run
To run each of our applications, we provide SLURM scripts that load all necessary modules and compile all necessary library code in each directory. Note that the SLURM scripts must be submitted via `sbatch` on the respective supercomputing machine (Perlmutter, Vista, or DeltaAI), as each machine has specific file systems. Directions to run each application can be found in each applications's `README.md` file (in their corresponding directories).

### Steps to Reproduce Experiments
1. Login to a **compute node** with at least one GPU. All the steps that follow should be run on the compute node.
2. Clone this artifact repository on the compute node. If working on NERSC Perlmutter, ensure the repository is cloned into a new folder in the `$HOME` directory. If working on TACC Vista, ensure the repository is cloned into a new folder in the `$WORK` directory. If working on NCSA DeltaAI, ensure the repository is cloned into a new folder in the `/projects` directory. To run the scripts, the repository must be cloned inside another folder within the respective file systems (e.g., clone inside `$WORK/MY_FOLDER` on Vista). Use the below commands to clone the repository:
    ```
    cd $WORK/MY_FOLDER/
    git clone https://github.com/mmogi03/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact.git
    ```
3. After cloning the repository, navigate to the repository folder:
    ```
    cd Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/
    ```
4. Navigate to the corresponding subdirectory for the specific experiment (e.g., to run the micro-benchmarks on Vista, `EXPERIMENT_FOLDER` would be `SC25_vista`):
    ```
    cd EXPERIMENT_FOLDER/
    ```
5. Follow the instructions in the `README.md` file in the corresponding subdirectory experiment folder.

## Related Documentation
For this artifact, we used a Linux environment to develop and test each application. The artifact depends on the following software packages:
  - **CUDA Toolkit** (version 12.2 for DeltaAI, version 12.4 for Perlmutter, version 12.5 for Vista)
    URL: https://developer.nvidia.com/cuda-toolkit-archive
    
  - **cuBLAS library** (bundled with CUDA Toolkit)
    URL: https://docs.nvidia.com/cuda/cublas/
    
  - **cuRAND library** (bundled with CUDA Toolkit)
    URL: https://docs.nvidia.com/cuda/curand/
    
  - **NVIDIA Management Library (NVML)** (bundled with NVIDIA drivers)
    URL: https://developer.nvidia.com/nvidia-management-library-nvml
    
  - **NAMD-GPU** (version 3.0.1) for molecular dynamics experiments
    URL: https://www.ks.uiuc.edu/Research/namd/
    
  - **OpenMPI** (version 5.0.x)
    URL: https://www.open-mpi.org/
    
  - **cuDNN**
    URL: https://developer.nvidia.com/cudnn