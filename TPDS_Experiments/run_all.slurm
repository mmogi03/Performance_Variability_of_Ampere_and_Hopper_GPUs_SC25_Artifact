#!/bin/bash -l

#SBATCH -N 1
#SBATCH --job-name=TPDS_VISTA_COMBINED
#SBATCH --mail-type=all
#SBATCH --mail-user=<email>
#SBATCH -o outputs/O_TPDS_VISTA_COMBINED__%j.out
#SBATCH -e outputs/E_TPDS_VISTA_COMBINED__%j.err
#SBATCH -A <account>
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
: "${CUTLASS_DIR:?Must set CUTLASS_DIR (path to cutlass repo)}"

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
echo "CUTLASS_DIR=${CUTLASS_DIR}"
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

mkdir -p "${SCRATCH_RUN}/exp_cutlass_tma"
cp -r "${REPO_FOLDER}/exp_cutlass_tma/CMakeLists.txt" "${SCRATCH_RUN}/exp_cutlass_tma/"
cp -r "${REPO_FOLDER}/exp_cutlass_tma/src"            "${SCRATCH_RUN}/exp_cutlass_tma/"
cp -r "${REPO_FOLDER}/exp_cutlass_tma/scripts"        "${SCRATCH_RUN}/exp_cutlass_tma/"

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

# Required for Vista Makefiles (keep exactly as you had)
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64
export INCLUDE=/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include

echo "Modules:"
module list

echo "Building shared/dumpGpuPower..."
make -C "${SCRATCH_RUN}/shared" clean
make -C "${SCRATCH_RUN}/shared"

echo "Building exp_cublas/matrix-matrix..."
make -C "${SCRATCH_RUN}/exp_cublas/matrix-matrix" clean
make -C "${SCRATCH_RUN}/exp_cublas/matrix-matrix"

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
# 2) Experiment B: CUTLASS TMA vs non-TMA minimal BF16-u01
# ---------------------------
echo "==== Step 2: CUTLASS TMA vs non-TMA (BF16 u01 only) ===="

module purge
module load cmake/4.1.1
module load nvidia/24.7
module load cuda/12.5

echo "Modules:"
module list

echo "Building shared/dumpGpuPower (again under CUTLASS toolchain)..."
make -C "${SCRATCH_RUN}/shared" clean
make -C "${SCRATCH_RUN}/shared"

echo "Configuring/building CUTLASS benchmarks..."
CUTLASS_EXP_DIR="${SCRATCH_RUN}/exp_cutlass_tma"
cd "${CUTLASS_EXP_DIR}"

rm -rf build 2>/dev/null || true
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
  -DCUTLASS_DIR="${CUTLASS_DIR}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"

cmake --build . -j

echo "Built binaries:"
ls -la "${CUTLASS_EXP_DIR}/build"

mkdir -p "${SCRATCH_RUN}/exp_cutlass_tma/results"

echo "Running srun CUTLASS minimal BF16-u01 suite..."
srun --export=ALL --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  /bin/bash "${SCRATCH_RUN}/exp_cutlass_tma/scripts/run_cutlass_suite_bf16_u01.sh" \
  "${SCRATCH_RUN}" "${TMP_BASE}"

echo "CUTLASS minimal suite done."

CUTLASS_REPO_OUT="${REPO_FOLDER}/exp_cutlass_tma/outputs/${RUN_ID}"
mkdir -p "${CUTLASS_REPO_OUT}"
copy_dir_to_repo "${SCRATCH_RUN}/exp_cutlass_tma/results" "${CUTLASS_REPO_OUT}/results"

cat > "${CUTLASS_REPO_OUT}/run_manifest.txt" <<EOF
RUN_ID=${RUN_ID}
JOB_ID=${SLURM_JOB_ID}
NODES=${SLURM_NODELIST}
DATETIME=${TS}
EXPERIMENT=CUTLASS non-TMA BF16 u01 + TMA BF16 u01 (repeats=100 warmup=20)
EOF

echo "==== Combined job complete ===="
echo "SCRATCH_RUN=${SCRATCH_RUN}"
echo "CUBLAS_OUT=${CUBLAS_REPO_OUT}"
echo "CUTLASS_OUT=${CUTLASS_REPO_OUT}"
echo "NODE_INFO_RUN=${REPO_FOLDER}/node_info/node_info_${RUN_ID}.csv"
echo "NODE_INFO_PERSIST=${REPO_FOLDER}/node_info/node_info.csv"