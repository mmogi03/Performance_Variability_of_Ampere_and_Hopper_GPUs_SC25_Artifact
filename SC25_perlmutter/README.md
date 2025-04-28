# Micro-benchmarks on NERSC Perlmutter (NVIDIA's A100s)

For NERSC Perlmutter, our analysis consists of a micro-benchmarking experiment.

## Micro-benchmarks

### Experiment Overview and Directory Structure

The micro-benchmark experiment utilizes cuBLAS level 1, 2, and 3 API calls to perform matrix-matrix (both on Tensor Cores and CUDA Cores), matrix-vector (CUDA Cores), and vector-scalar (CUDA Cores) kernel operations with different data types (BF16, FP16, FP32, and FP64). On Perlmutter, we ran it as a 4-GPU application (per node) using NVIDIA's A100 GPUs and allowed the application to run to completion.

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Micro-benchmarks](#compile-and-run-microbenchmarks). Below is an overview of this directory.
```
├── matrix-matrix
    ├── BF16/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script_cuda.sh: script used to run the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
    ├── script.sh: script used to run the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on Tensor cores
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

### Compile and Run Micro-benchmarks

To run the full micro-benchmark experiments, submit the SLURM script `microbenchmarks.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64,INCLUDE=/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/include microbenchmarks.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on Perlmutter, the output folder will be stored in $SCRATCH).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that by default, each kernel runs for 20 warm-up iterations and records the elapsed time for 100 iterations. For matrix-matrix operations, the size of the matrices used is 32768x32768. For matrix-vector operations, the size of the matrices used is 32768x32768 and the size of the vectors used is 32768x1. For vector-scalar operations, the size of the vectors used is 1073741824x1.

### Outputs

Once the job finishes, on Perlmutter, the folder containing all the data will be stored under `$SCRATCH/OUTPUT_FOLDER/perlmutter-output`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-cuda-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── perlmutter_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-matrix-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── perlmutter_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-scalar-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── perlmutter_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-vector-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── perlmutter_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── perlmutter_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on Perlmutter there are 4 GPUs per node so the ID is 0, 1, 2, or 3).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Prerequisites

* Machine with an NVIDIA A100 GPU
* Updated GPU drivers installed (if not, please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
* The scripts for compiling and launching the experiments assume the Ampere Architecture. The Makefiles all use `SM_80` for this analysis, which is compatible with the Ampere and Hopper architectures.

## Post-Processing

Note that on Perlmutter, `$SCRATCH` is a temporary file system and is subject to purging. The user should manually move the entire `OUTPUT_FOLDER` back into permanent storage, such as Community (`$CFS`) or Archive (HPSS `hsi`), before performing any data analysis.