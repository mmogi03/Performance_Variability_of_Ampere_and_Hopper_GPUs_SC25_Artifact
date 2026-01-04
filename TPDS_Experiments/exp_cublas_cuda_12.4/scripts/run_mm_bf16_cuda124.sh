#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_mm_bf16_only.sh <SCRATCH_RUN_ROOT> <TMP_BASE>

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <SCRATCH_RUN_ROOT> <TMP_BASE>"
  exit 1
fi

SCRATCH_RUN_ROOT="$1"
TMP_BASE="$2"

EXP_ROOT="${SCRATCH_RUN_ROOT}/exp_cublas_cuda_12.4"
MM_DIR="${EXP_ROOT}/matrix-matrix"
BIN="${MM_DIR}/BF16/src/main"
PROFILER="${SCRATCH_RUN_ROOT}/shared/dumpGpuPower"

node_id="${SLURMD_NODENAME:-$(hostname)}"
gpu_id="${SLURM_LOCALID:-0}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# shellcheck source=/dev/null
source "${SCRATCH_RUN_ROOT}/utils/gpu_info_utils.sh"
# shellcheck source=/dev/null
source "${SCRATCH_RUN_ROOT}/utils/cooldown_utils.sh"

uuid="$(gpu_uuid "${gpu_id}")"
name="$(gpu_name "${gpu_id}")"

echo "NODE: ${node_id} | GPU_LOCAL_ID: ${gpu_id} | UUID: ${uuid} | NAME: ${name}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: Missing/non-executable BF16 binary: ${BIN}"
  exit 1
fi
if [[ ! -x "${PROFILER}" ]]; then
  echo "ERROR: Missing/non-executable profiler: ${PROFILER}"
  exit 1
fi

baseline="$(get_gpu_temp_c "${gpu_id}")"
if [[ -z "${baseline}" ]]; then
  echo "ERROR: Could not read baseline GPU temperature."
  exit 1
fi
echo "Baseline locked once: ${baseline}C"

format="BF16"
data_dir="${format}/data"
node_dir="${TMP_BASE}/matrix-matrix-output/${node_id}/${gpu_id}/vista_${data_dir}"
mkdir -p "${node_dir}"

POWER_SAMPLE_RATE="${POWER_SAMPLE_RATE:-0.25}"
POWER_SAMPLES="${POWER_SAMPLES:--1}"

gpu_log_file="${node_dir}/gpu_power_${gpu_id}.csv"
echo "Starting dumpGpuPower -> ${gpu_log_file}"
(
  "${PROFILER}" -d "${gpu_id}" -r "${POWER_SAMPLE_RATE}" -n "${POWER_SAMPLES}" > "${gpu_log_file}"
) &
prof_pid="$!"

out_csv="${node_dir}/${gpu_id}.csv"
echo "Running ${BIN} -> ${out_csv}"
set +e
"${BIN}" 32768 100 20 0 1 > "${out_csv}"
rc=$?
set -e

kill "${prof_pid}" 2>/dev/null || true
wait "${prof_pid}" 2>/dev/null || true
echo "Stopped dumpGpuPower (PID: ${prof_pid})"

if [[ ${rc} -ne 0 ]]; then
  echo "ERROR: BF16 benchmark exited non-zero (rc=${rc})"
  exit "${rc}"
fi

dest_node_dir="${EXP_ROOT}/vista-output/matrix-matrix-output/${node_id}"
dest_gpu_dir="${dest_node_dir}/${gpu_id}"

rm -rf "${dest_gpu_dir}" 2>/dev/null || true
mkdir -p "${dest_gpu_dir}"

cp -r "${TMP_BASE}/matrix-matrix-output/${node_id}/${gpu_id}" "${dest_gpu_dir}"
echo "Copied results to: ${dest_gpu_dir}/${gpu_id}"

# Cooldown at end so next experiment starts cooled
cooldown_to_baseline "${gpu_id}" "${baseline}" "${TEMP_HEADROOM_C:-1}" "${COOLDOWN_SLEEP_SEC:-5}" "${COOLDOWN_MAX_ITERS:-60}"
