# NAMD 3 Molecular-Dynamics Benchmark on TACC Vista (NVIDIA GH200)

This document explains how to compile, launch, and collect results for the 2×2×2 STMV NAMD 3 benchmark on the TACC Vista GPU testbed.

## Microbenchmarks

### Experiment Overview and Directory Structure

The run exercises a single GH200 GPU per node for a short NPT simulation

For compiling and launching the full experiment on NVIDIA GPUs, please see section [Compile and Run Microbenchmarks](#compile-and-run-microbenchmarks). Below is an overview of this directory.
```
namd_vista/
├── shared/                     
    ├── dumpGPuPower.cpp: launches a profiler collecting GPU telemetry in the background
    ├── Makefile: make binaries for `dumpGpuPower.cpp`
├── 2x2x2stmv.namd: 6000-step NAMD input
├── namd_fastest.slurm: script that loads all modules, compiles all necessary library code, and runs the namd benchmarks on the 5 designated fastest nodes
├── namd_medium.slurm: script that loads all modules, compiles all necessary library code, and runs the namd benchmarks on the 5 designated medium nodes
├── namd_slowest.slurm: script that loads all modules, compiles all necessary library code, and runs the namd benchmarks on the 5 designated slowest nodes
└── namd.sh: script used to run the namd simulation
```

### Compile and Run Microbenchmarks

To run the namd experiments, submit the SLURM script `namd_fastest.slurm` via `sbatch` as shown below:
```
sbatch --export=OUTPUT_FOLDER=NAMD_fastest,REPO_FOLDER=Fortuna,LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/compilers/lib:/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64:/home1/apps/nvidia/Linux_aarch64/24.7/comm_libs/nvshmem/lib:/home1/apps/nvidia/Linux_aarch64/24.7/comm_libs/nccl/lib,INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include namd_fastest.slurm
```
where
- `OUTPUT_FOLDER` specifies the output directory for benchmark logs (note that on Vista, the output folder will be stored in $SCRATCH).
- `REPO_FOLDER` is the folder name of where the GitHub repository was cloned into (e.g., `MY_FOLDER`).
- `LD_LIBRARY_PATH` and `INCLUDE` are environment variables required to locate the NVIDIA HPC SDK math libraries.

### Outputs

Once the job finishes, on Vista, the folder containing all the data will be stored under `$SCRATCH/OUTPUT_FOLDER/vista-output`. The directory structure of this folder is summarized below:
```
├── /namd_vista/data/NAMD_fastest/vista-output/namd3-output
    ├── <NODE_ID>
        ├── <GPU_ID>/<GPU_ID>
            ├── <GPU_ID>.log
            ├── gpu_power_<GPU_ID>.csv
        ├── namd3_runtime.csv
```
where
- `<GPU_ID>` is the local rank of the GPU on the node (e.g., on Vista there is only 1 GPU per node so the ID is 0).
- `<GPU_ID>.log` contains the logs of running the NAMD script
- `gpu_power_<GPU_ID>.csv` contains the GPU telemetry recorded by the profiler throughout the kernel's execution. The columns are `sample`, `power(W)`, `gpu_util(%)`, `core_clock(MHz)`, `mem_clock(MHz)`, `timestamp_ns`, `temp(C)`, and `energy(mJ)`.
- `namd3_runtime.csv` contains the runtime of the NAMD script. Since NAMD is only ran once on the input file, there is only 1 line of runtime.

## Prerequisites

* Machine with an NVIDIA GH200 Superchip
* Updated GPU drivers installed (if not, please see https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
* The scripts for compiling and launching the experiments assume the Hopper Architecture. The Makefiles all use `SM_80` for this analysis, which is compatible with the Ampere and Hopper architectures.

## Post-Processing

Note that on Vista, `$SCRATCH` is a temporary file system and is subject to purging. The user should manually move the entire `OUTPUT_FOLDER` back into permanent storage, such as Work (`$WORK`), before performing any data analysis.