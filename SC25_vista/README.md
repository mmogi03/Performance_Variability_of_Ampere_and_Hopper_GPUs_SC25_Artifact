# Microbenchmarks, Timestamp Experiment, and Temporal Analysis on TACC Vista (NVIDIA's GH200s)

For TACC Vista, our analysis consists of three experiments: micro-benchmarking, timestamp experiment, and temporal analysis.

## Microbenchmarks

### Experiment Overview and Directory Structure

The micro-benchmark experiment utilizes cuBLAS level 1, 2, and 3 API calls to perform matrix-matrix (both on Tensor Cores and CUDA Cores), matrix-vector (CUDA Cores), and vector-scalar (CUDA Cores) kernel operations with different data types (BF16, FP16, FP32, FP64, and FP8). On Vista, we ran it as a single-GPU application (per node) using NVIDIA's GH200 Superchips and allowed the application to run to completion.

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

To run the full micro-benchmark experiments, submit the SLURM script `microbenchmarks.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64,INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include microbenchmarks.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on Vista, the output folder will be stored in $SCRATCH).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that by default, each kernel runs for 20 warm-up iterations and records the elapsed time for 100 iterations. For matrix-matrix operations, the size of the matrices used is 32768x32768. For matrix-vector operations, the size of the matrices used is 32768x32768 and the size of the vectors used is 32768x1. For vector-scalar operations, the size of the vectors used is 1073741824x1.

### Outputs

Once the job finishes, on Vista, the folder containing all the data will be stored under `$SCRATCH/OUTPUT_FOLDER/vista-output`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-cuda-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-matrix-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP8
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-scalar-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-vector-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on Vista there is only 1 GPU per node so the ID is 0).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Timestamp Experiment

The timestamp experiment extends the micro-benchmark application code to output a high-resolution timestamp before and after the initialization/allocation of data structures as well as each iteration both during the warm-up and recorded phases. In post-processing, this allows us to "stitch" together and map each iteration's timing to the GPU telemtry throughout the kernel's entire execution path (e.g., temperature, power, core clock frequency, and total energy consumption).

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Timestamps](#compile-and-run-timestamps). Below is an overview of this directory.
```
├── matrix-matrix-timestamp
    ├── BF16/src/main.cu
    ├── FP8/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP8/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script_cuda.sh: script used to run the timestamp versions of the matrix-matrix kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
    ├── script.sh: script used to run the timestamp versions of the matrix-matrix kernels using BF16, FP8, FP16, FP32, and FP64 precisions on Tensor cores
├── matrix-scalar-timestamp:
    ├── FP32/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `FP32/src/main.cu` and `FP64/src/main.cu`
    ├── script.sh: script used to run the timestamp versions of the vector-scalar kernels using FP32 and FP64 precisions on CUDA cores
├── matrix-vector-timestamp: 
    ├── BF16/src/main.cu
    ├── FP16/src/main.cu
    ├── FP32/src/main.cu 
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu`, `FP16/src/main.cu`, `FP32/src/main.cu`, and `FP64/src/main.cu`
    ├── script.sh: script used to run the timestamp versions of the matrix-vector kernels using BF16, FP16, FP32, and FP64 precisions on CUDA cores
├── shared:
    ├── dumpGPuPower.cpp: launches a profiler collecting GPU telemetry in the background
    ├── Makefile: make binaries for `dumpGpuPower.cpp`
├── timestamps.slurm: script that loads all modules, compiles all necessary library code, and runs the timestamp versions of the micro-benchmark workloads in sequence
```

### Compile and Run Timestamps

To run the full timestamp experiments, submit the SLURM script `timestamps.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64,INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include timestamps.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on Vista, the output folder will be stored in $SCRATCH).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that everything runs the same as the micro-benchmark experiments, with the addition of the timestamp outputs.

### Outputs

Once the job finishes, on Vista, the folder containing all the data will be stored under `$SCRATCH/OUTPUT_FOLDER/vista-output`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-timestamp-cuda-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-matrix-timestamp-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP8
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-scalar-timestamp-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
├── matrix-vector-timestamp-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP32
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on Vista there is only 1 GPU per node so the ID is 0).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, `start_ts`, `stop_ts`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Temporal Analysis

The temporal analysis application extends the timestamp version of the micro-benchmark application code to run the matrix-matrix kernels (using Tensor Cores) on the BF16 and FP64 precisions for 4 hours each (measured as the cumulative elapsed iteration time to avoid any inter-iteration phases). This is to observe any correlation between dynamic voltage adjustment and performance variability as kernels are executed at high temperatures for extended periods of time.

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Temporal Analysis](#compile-and-run-temporal-analysis). Below is an overview of this directory.
```
├── matrix-matrix-fourhour
    ├── BF16/src/main.cu
    ├── FP64/src/main.cu
    ├── Makefile: make binaries for `BF16/src/main.cu` and `FP64/src/main.cu`
    ├── four_hour.sh: script used to run the temporal analysis of the matrix-matrix kernels using BF16 and FP64 precisions on Tensor cores
├── shared:
    ├── dumpGPuPower.cpp: launches a profiler collecting GPU telemetry in the background
    ├── Makefile: make binaries for `dumpGpuPower.cpp`
├── temporal.slurm: script that loads all modules, compiles all necessary library code, and runs the temporal analysis of the matrix-matrix kernels using BF16 and FP64 precisions on Tensor cores
```

### Compile and Run Temporal Analysis

To run the full timestamp experiments, submit the SLURM script `temporal.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=run_A,REPO_FOLDER=MY_FOLDER,LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64,INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include temporal.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on Vista, the output folder will be stored in $SCRATCH).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

Note that everything runs the same as the timestamp version of the micro-benchmark experiments, except that the BF16 and FP64 kernels are constrained to only run for 4 hours each.

### Outputs

Once the job finishes, on Vista, the folder containing all the data will be stored under `$SCRATCH/OUTPUT_FOLDER/vista-output`. The directory structure of this folder is summarized below:
```
├── matrix-matrix-fourhour-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── vista_BF16
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
            ├── vista_FP64
                ├── <GPU_ID>.csv
                ├── gpu_power_<GPU_ID>.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on Vista there is only 1 GPU per node so the ID is 0).
- `<GPU_ID>.csv` contains the per-iteration elapsed run-time of the kernel performing the workload with the specified precision. The columns are `size`, `iteration`, `start_ts`, `stop_ts`, and `time(ms)`.
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.

## Prerequisites

* Machine with an NVIDIA GH200 Superchip
* Updated GPU drivers installed (if not, please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
* The scripts for compiling and launching the experiments assume the Hopper Architecture. The Makefiles all use `SM_80` for this analysis, which is compatible with the Ampere and Hopper architectures.

## Post-Processing

Note that on Vista, `$SCRATCH` is a temporary file system and is subject to purging. The user should manually move the entire `OUTPUT_FOLDER` back into permanent storage, such as Work (`$WORK`), before performing any data analysis.