# Microbenchmarks, Dedicated GPU Experiment on NCSA DeltaAI (NVIDIA's GH200s)

For NCSA DeltaAI, our analysis consists of two experiments: micro-benchmarking and dedicated gpu experiment.

## Microbenchmarks

### Experiment Overview and Directory Structure

The micro-benchmark experiment utilizes cuBLAS level 1, 2, and 3 API calls to perform matrix-matrix (both on Tensor Cores and CUDA Cores), matrix-vector (CUDA Cores), and vector-scalar (CUDA Cores) kernel operations with different data types (BF16, FP16, FP32, FP64, and FP8). On DeltaAI, we ran it as a 4-GPU application (per node) using NVIDIA's GH200 superchips and allowed the application to run to completion.

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Microbenchmarks](#compile-and-run-microbenchmarks). Below is an overview of this directory.
```
├── matrix-matrix
    ├── BF16/src/main.cu
    ├── FP8/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP8/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script_cuda.sh: script used to run the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
    ├── script_cuda2.sh: script used to run the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores with dedicated superchips
    ├── script.sh: script used to run the matrix-matrix kernels using BF16, FP8, FP16, FP32, and FP64 precisions on Tensor cores
    ├── script.sh: script used to run the matrix-matrix kernels using BF16, FP8, FP16, FP32, and FP64 precisions on Tensor cores with dedicated superchips
├── matrix-scalar:
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `FP32/src/main.cu` and `FP64/src/main.cu`
    ├── script.sh: script used to run the vector-scalar kernels using FP32 and FP64 precisions on CUDA cores
    ├── script2.sh: script used to run the vector-scalar kernels using FP32 and FP64 precisions on CUDA cores with dedicated superchips
├── matrix-vector: 
    ├── BF16/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu 
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script.sh: script used to run the matrix-vector kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
    ├── script.sh: script used to run the matrix-vector kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores with dedicated superchips
├── shared:
    ├── dumpGPuPower.cpp: launches a profiler collecting GPU telemetry in the background
    ├── Makefile: make binaries for `dumpGpuPower.cpp`
├── microbenchmarks.slurm: script that loads all modules, compiles all necessary library code, and runs the micro-benchmark workloads in sequence
```

### Compile and Run Microbenchmarks

To run the full micro-benchmark experiments, submit the SLURM script `microbenchmarks.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/lib64,INCLUDE=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/include microbenchmarks.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on DeltaAI, the output folder will be stored in /projects).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that by default, each kernel runs for 20 warm-up iterations and records the elapsed time for 100 iterations. For matrix-matrix operations, the size of the matrices used is 32768x32768. For matrix-vector operations, the size of the matrices used is 32768x32768 and the size of the vectors used is 32768x1. For vector-scalar operations, the size of the vectors used is 1073741824x1.

### Outputs

Once the job finishes, on DeltaAI, the folder containing all the data will be stored under `/projects/MY_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/dtai_outputs/OUTPUT_FOLDER`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-cuda-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-matrix-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP8
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-scalar-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-vector-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on DeltaAI there are 4 GPUs per node so the ID is 0, 1, 2, or 3).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Dedicated GPU Experiment

### Experiment Overview and Directory Structure

The dedicated GPU experiment utilizes cuBLAS level 1, 2, and 3 API calls to perform matrix-matrix (both on Tensor Cores and CUDA Cores), matrix-vector (CUDA Cores), and vector-scalar (CUDA Cores) kernel operations with different data types (BF16, FP16, FP32, FP64, and FP8) with only 1 superchip active at a time. On DeltaAI, we ran it as a 4-GPU application (per node) using NVIDIA's GH200 superchips and allowed the application to run to completion.

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Microbenchmarks](#compile-and-run-microbenchmarks). Below is an overview of this directory.
```
├── matrix-matrix
    ├── BF16/src/main.cu
    ├── FP8/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP8/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script_cuda.sh: script used to run the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
    ├── script.sh: script used to run the matrix-matrix kernels using BF16, FP8, FP16, FP32, and FP64 precisions on Tensor cores
├── matrix-scalar:
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `FP32/src/main.cu` and `FP64/src/main.cu`
    ├── script.sh: script used to run the vector-scalar kernels using FP32 and FP64 precisions on CUDA cores
├── matrix-vector: 
    ├── BF16/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu 
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script.sh: script used to run the matrix-vector kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
├── shared:
    ├── dumpGPuPower.cpp: launches a profiler collecting GPU telemetry in the background
    ├── Makefile: make binaries for `dumpGpuPower.cpp`
├── microbenchmarks.slurm: script that loads all modules, compiles all necessary library code, and runs the micro-benchmark workloads in sequence
```

### Compile and Run Microbenchmarks

To run the full micro-benchmark experiments, submit the SLURM script `dedicated.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/lib64,INCLUDE=/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/include dedicated.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on DeltaAI, the output folder will be stored in /projects).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that by default, each kernel runs for 20 warm-up iterations and records the elapsed time for 100 iterations. For matrix-matrix operations, the size of the matrices used is 32768x32768. For matrix-vector operations, the size of the matrices used is 32768x32768 and the size of the vectors used is 32768x1. For vector-scalar operations, the size of the vectors used is 1073741824x1.

### Outputs

Once the job finishes, on DeltaAI, the folder containing all the data will be stored under `/projects/MY_FOLDER/Performance_Variability_of_Ampere_and_Hopper_GPUs_SC25_Artifact/SC25_dtai/dtai_outputs/OUTPUT_FOLDER`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-cuda-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-matrix-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP8
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-scalar-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-vector-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── dtai_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── dtai_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on DeltaAI there are 4 GPUs per node so the ID is 0, 1, 2, or 3).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Prerequisites

* Machine with an NVIDIA GH200 Superchip
* Updated GPU drivers installed (if not, please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
* The scripts for compiling and launching the experiments assume the Hopper Architecture. The Makefiles all use `SM_80` for this analysis, which is compatible with the Ampere and Hopper architectures.

## Post-Processing

Note that on DeltaAI, `/projects` is a permanent file system and is not subject to purging, Allowing the user to perform data analysis directly from this file system.