#!/bin/bash -l

#SBATCH -N 2
#SBATCH --job-name=TPDS_VISTA_COMBINED
#SBATCH --mail-type=all
#SBATCH --mail-user=hgz5@scarletmail.rutgers.edu
#SBATCH -o outputs/O_TPDS_VISTA_COMBINED__%j.out
#SBATCH -e outputs/E_TPDS_VISTA_COMBINED__%j.err
#SBATCH -A CCR24047
#SBATCH -p gh
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1

set -euo pipefail

echo "==== Combined job start ===="
echo "JobID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "NNODES: ${SLURM_NNODES}"

# ---------------------------
# Inputs / overridable env vars
# ---------------------------
REPO_FOLDER="${REPO_FOLDER:-${SLURM_SUBMIT_DIR}}"

CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-90a}"

# Optional: tweak cooldown globally (wrappers honor these)
export COOLDOWN_SLEEP_SEC="${COOLDOWN_SLEEP_SEC:-5}"
export COOLDOWN_MAX_ITERS="${COOLDOWN_MAX_ITERS:-60}"
export TEMP_HEADROOM_C="${TEMP_HEADROOM_C:-1}"
export POWER_SAMPLE_RATE="${POWER_SAMPLE_RATE:-0.25}"
export POWER_SAMPLES="${POWER_SAMPLES:--1}"

# Unique run id used for outputs in repo
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="VISTA_COMBINED_${SLURM_JOB_ID}_${TS}"

echo "REPO_FOLDER=${REPO_FOLDER}"
echo "RUN_ID=${RUN_ID}"

# ---------------------------
# Stage repo to SCRATCH
# ---------------------------
cd "${SCRATCH}"
SCRATCH_RUN="${SCRATCH}/${RUN_ID}"

safe_rm() {
  local p="$1"
  if [[ -d "${p}" ]]; then
    case "${p}" in
      "${SCRATCH}"/VISTA_COMBINED_*)
        echo "Removing existing SCRATCH_RUN: ${p}"
        rm -rf "${p}"
        ;;
      *)
        echo "Refusing to rm -rf unexpected path: ${p}"
        exit 1
        ;;
    esac
  fi
}

safe_rm "${SCRATCH_RUN}"
mkdir -p "${SCRATCH_RUN}"


echo "Staging required folders into: ${SCRATCH_RUN}"
cp -r "${REPO_FOLDER}/shared" "${SCRATCH_RUN}/"
cp -r "${REPO_FOLDER}/utils"  "${SCRATCH_RUN}/"

mkdir -p "${SCRATCH_RUN}/exp_cublas"
cp -r "${REPO_FOLDER}/exp_cublas/matrix-matrix" "${SCRATCH_RUN}/exp_cublas/"
cp -r "${REPO_FOLDER}/exp_cublas/scripts"       "${SCRATCH_RUN}/exp_cublas/"

mkdir -p "${SCRATCH_RUN}/exp_cublas_cuda_11.8"
cp -r "${REPO_FOLDER}/exp_cublas_cuda_11.8/matrix-matrix" "${SCRATCH_RUN}/exp_cublas_cuda_11.8/"
cp -r "${REPO_FOLDER}/exp_cublas_cuda_11.8/scripts"       "${SCRATCH_RUN}/exp_cublas_cuda_11.8/"

mkdir -p "${SCRATCH_RUN}/exp_cublas_cuda_12.4"
cp -r "${REPO_FOLDER}/exp_cublas_cuda_12.4/matrix-matrix" "${SCRATCH_RUN}/exp_cublas_cuda_12.4/"
cp -r "${REPO_FOLDER}/exp_cublas_cuda_12.4/scripts"       "${SCRATCH_RUN}/exp_cublas_cuda_12.4/"

mkdir -p "${SCRATCH_RUN}/exp_stream"
cp -r "${REPO_FOLDER}/exp_stream/stream-gpu-linux-aarch64-gpu" "${SCRATCH_RUN}/exp_stream/"

mkdir -p "${SCRATCH_RUN}/node_info/per_node"

echo "Staged contents:"
ls -la "${SCRATCH_RUN}"

# Node-local TMP base (each task also uses this)
TMP_BASE="/tmp/${USER}/${RUN_ID}"
echo "TMP_BASE=${TMP_BASE}"

# ---------------------------
# Helpers
# ---------------------------
append_node_info_persistent() {
  local merged_csv="$1"
  local persistent_dir="${REPO_FOLDER}/node_info"
  local persistent_csv="${persistent_dir}/node_info.csv"

  mkdir -p "${persistent_dir}"

  if [[ ! -f "${persistent_csv}" ]]; then
    echo "Creating persistent node_info file: ${persistent_csv}"
    cp -f "${merged_csv}" "${persistent_csv}"
  else
    echo "Appending to persistent node_info file: ${persistent_csv}"
    tail -n +2 "${merged_csv}" >> "${persistent_csv}"
  fi
}

copy_dir_to_repo() {
  local src="$1"
  local dst="$2"
  echo "Copying: ${src} -> ${dst}"
  rm -rf "${dst}" 2>/dev/null || true
  mkdir -p "${dst}"
  cp -r "${src}/." "${dst}/"
}

# ---------------------------
# 0) Collect node/GPU identity first (srun)
# ---------------------------
echo "==== Step 0: Collect node/GPU identity ===="

module purge || true
module load nvidia/24.7 || true
module load cuda/12.5 || true

echo "Modules:"
module list || true

echo "Running srun collect_node_info..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/utils/collect_node_info.sh" "${SCRATCH_RUN}" "${TMP_BASE}"

echo "Merging per-node node_info CSV rows..."
MERGED_NODE_INFO="${SCRATCH_RUN}/node_info/node_info_${RUN_ID}.csv"
{
  echo "NodeID,GpuLocalID,GpuUUID,GpuName,GpuSerial,PciBusId,VbiosVersion,DriverVersion,JobID,Datetime"
  cat "${SCRATCH_RUN}/node_info/per_node/"*.csv
} > "${MERGED_NODE_INFO}"

echo "Merged node_info: ${MERGED_NODE_INFO}"
mkdir -p "${REPO_FOLDER}/node_info"
cp -f "${MERGED_NODE_INFO}" "${REPO_FOLDER}/node_info/node_info_${RUN_ID}.csv"
append_node_info_persistent "${MERGED_NODE_INFO}"

# ---------------------------
# 1) Experiment A: cuBLAS matrix-matrix BF16 only
# ---------------------------
echo "==== Step 1: cuBLAS matrix-matrix BF16-only ===="

module purge
module load nvidia/24.7
module load cuda/12.5

echo "Modules:"
module list

make -C "${SCRATCH_RUN}/shared" clean
make -C "${SCRATCH_RUN}/shared"


echo "Building exp_cublas/matrix-matrix..."
make -C "${SCRATCH_RUN}/exp_cublas/matrix-matrix" clean
make -C "${SCRATCH_RUN}/exp_cublas/matrix-matrix"

CUDA11_EXP_DIR="${SCRATCH_RUN}/exp_cublas/matrix-matrix"
cd "${CUDA11_EXP_DIR}"

rm -rf build 2>/dev/null || true
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build . -j

mkdir -p "${SCRATCH_RUN}/exp_cublas/vista-output/matrix-matrix-output"

echo "Running srun BF16-only wrapper..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/exp_cublas/scripts/run_mm_bf16_only.sh" "${SCRATCH_RUN}" "${TMP_BASE}"

echo "cuBLAS BF16-only done."

CUBLAS_REPO_OUT="${REPO_FOLDER}/exp_cublas/outputs/${RUN_ID}"
mkdir -p "${CUBLAS_REPO_OUT}"
copy_dir_to_repo "${SCRATCH_RUN}/exp_cublas/vista-output" "${CUBLAS_REPO_OUT}/vista-output"

cat > "${CUBLAS_REPO_OUT}/run_manifest.txt" <<EOF
RUN_ID=${RUN_ID}
JOB_ID=${SLURM_JOB_ID}
NODES=${SLURM_NODELIST}
DATETIME=${TS}
EXPERIMENT=cuBLAS matrix-matrix BF16 only
EOF

# ---------------------------
# 3) Experiment C: cuBLAS matrix-matrix BF16 only with cuda version 11.8
# ---------------------------
echo "==== Step 3: cuBLAS matrix-matrix BF16-cuda118 ===="

module purge
module load nvidia/24.7
module load cuda/11.8

echo "Modules:"
module list


CUDA11_EXP_DIR="${SCRATCH_RUN}/exp_cublas_cuda_11.8/matrix-matrix"
cd "${CUDA11_EXP_DIR}"

rm -rf build 2>/dev/null || true
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . -j

mkdir -p "${SCRATCH_RUN}/exp_cublas_cuda_11.8/vista-output/matrix-matrix-output"

echo "Running srun BF16-cuda11 wrapper..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/exp_cublas_cuda_11.8/scripts/run_mm_bf16_cuda118.sh" "${SCRATCH_RUN}" "${TMP_BASE}"

echo "cuBLAS BF16-cuda118 done."

CUBLAS_REPO_OUT="${REPO_FOLDER}/exp_cublas_cuda_11.8/outputs/${RUN_ID}"
mkdir -p "${CUBLAS_REPO_OUT}"
copy_dir_to_repo "${SCRATCH_RUN}/exp_cublas_cuda_11.8/vista-output" "${CUBLAS_REPO_OUT}/vista-output"

cat > "${CUBLAS_REPO_OUT}/run_manifest.txt" <<EOF
RUN_ID=${RUN_ID}
JOB_ID=${SLURM_JOB_ID}
NODES=${SLURM_NODELIST}
DATETIME=${TS}
EXPERIMENT=cuBLAS matrix-matrix BF16 cuda11
EOF

# ---------------------------
# 4) Experiment D: cuBLAS matrix-matrix BF16 only with cuda version 12.4
# ---------------------------
echo "==== Step 4: cuBLAS matrix-matrix BF16-cuda124 ===="

module purge
module load nvidia/24.7
module load cuda/12.4

echo "Modules:"
module list

CUDA124_EXP_DIR="${SCRATCH_RUN}/exp_cublas_cuda_12.4/matrix-matrix"
cd "${CUDA124_EXP_DIR}"

rm -rf build 2>/dev/null || true
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build . -j

mkdir -p "${SCRATCH_RUN}/exp_cublas_cuda_12.4/vista-output/matrix-matrix-output"

echo "Running srun BF16-cuda124 wrapper..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/exp_cublas_cuda_12.4/scripts/run_mm_bf16_cuda124.sh" "${SCRATCH_RUN}" "${TMP_BASE}"

echo "cuBLAS BF16-cuda124 done."

CUBLAS_REPO_OUT="${REPO_FOLDER}/exp_cublas_cuda_12.4/outputs/${RUN_ID}"
mkdir -p "${CUBLAS_REPO_OUT}"
copy_dir_to_repo "${SCRATCH_RUN}/exp_cublas_cuda_12.4/vista-output" "${CUBLAS_REPO_OUT}/vista-output"

cat > "${CUBLAS_REPO_OUT}/run_manifest.txt" <<EOF
RUN_ID=${RUN_ID}
JOB_ID=${SLURM_JOB_ID}
NODES=${SLURM_NODELIST}
DATETIME=${TS}
EXPERIMENT=cuBLAS matrix-matrix BF16 cuda124
EOF

# ---------------------------
# 5) Experiment E: STREAM Bandwidth Benchmark
# ---------------------------
echo "==== Step 5: STREAM Bandwidth Benchmark ===="

module purge
module load gcc/13.2.0 cuda/12.8
module load openmpi/5.0.5

echo "Modules:"
module list

THREADS=72                 # Adjust based on system memory capacity
export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=spread

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"

mkdir -p "${SCRATCH_RUN}/exp_stream/vista-output/stream-output"

echo "Running srun STREAM GPU wrapper..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/exp_stream/stream-gpu-linux-aarch64-gpu/stream-gpu-test.sh" "${SCRATCH_RUN}" "${TMP_BASE}"

echo "cuBLAS BF16-only done."

CUBLAS_REPO_OUT="${REPO_FOLDER}/exp_stream/outputs/${RUN_ID}"
mkdir -p "${CUBLAS_REPO_OUT}"
copy_dir_to_repo "${SCRATCH_RUN}/exp_stream/vista-output" "${CUBLAS_REPO_OUT}/vista-output"

cat > "${CUBLAS_REPO_OUT}/run_manifest.txt" <<EOF
RUN_ID=${RUN_ID}
JOB_ID=${SLURM_JOB_ID}
NODES=${SLURM_NODELIST}
DATETIME=${TS}
EXPERIMENT=STREAM
EOF

echo "==== Combined job complete ===="
echo "SCRATCH_RUN=${SCRATCH_RUN}"
echo "CUBLAS_OUT=${CUBLAS_REPO_OUT}"
echo "NODE_INFO_RUN=${REPO_FOLDER}/node_info/node_info_${RUN_ID}.csv"
echo "NODE_INFO_PERSIST=${REPO_FOLDER}/node_info/node_info.csv"