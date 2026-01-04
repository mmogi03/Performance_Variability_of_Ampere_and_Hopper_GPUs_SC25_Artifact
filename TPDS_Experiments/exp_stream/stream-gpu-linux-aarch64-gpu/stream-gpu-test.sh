#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_stream_gpu.sh <SCRATCH_RUN_ROOT> <TMP_BASE> [--n <elements>] [--d <device>] [--t <test_type>] [--dt <datatype>]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <SCRATCH_RUN_ROOT> <TMP_BASE> [--n <elements>] [--d <device>] [--t <test_type>] [--dt <datatype>]"
  exit 1
fi

SCRATCH_RUN_ROOT="$1"
TMP_BASE="$2"
shift 2

# ---------------------------
# Parse optional arguments
# ---------------------------
NUMBER_OF_ELEMENTS=""
DEVICE=""
TEST_TYPE=""
EXEC_NAME="stream_test_fp32"

SCRIPT_DIR=$( cd -- "$( dirname -- "$( readlink -f "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )
STREAM_EXEC="$SCRIPT_DIR/$EXEC_NAME"

# ---------------------------
# Node/GPU assignment
# ---------------------------
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

if [[ ! -x "${STREAM_EXEC}" ]]; then
    echo "ERROR: Missing/non-executable STREAM binary: ${STREAM_EXEC}"
    exit 1
fi

baseline="$(get_gpu_temp_c "${gpu_id}")"
if [[ -z "${baseline}" ]]; then
  echo "ERROR: Could not read baseline GPU temperature."
  exit 1
fi
echo "Baseline locked once: ${baseline}C"

# ---------------------------
# Output directories
# ---------------------------
node_dir="${TMP_BASE}/stream-output/${node_id}/${gpu_id}"
mkdir -p "${node_dir}"

POWER_SAMPLE_RATE="${POWER_SAMPLE_RATE:-0.25}"
POWER_SAMPLES="${POWER_SAMPLES:--1}"

gpu_log_file="${node_dir}/gpu_power_${gpu_id}.csv"
echo "Starting dumpGpuPower -> ${gpu_log_file}"
(
  "${SCRATCH_RUN_ROOT}/shared/dumpGpuPower" -d "${gpu_id}" -r "${POWER_SAMPLE_RATE}" -n "${POWER_SAMPLES}" > "${gpu_log_file}"
) &
prof_pid="$!"

# ---------------------------
# Run STREAM binary
# ---------------------------
out_csv="${node_dir}/${gpu_id}.csv"
echo "Running ${STREAM_EXEC} -> ${out_csv}"
set +e
"${STREAM_EXEC}" $DEVICE $NUMBER_OF_ELEMENTS $TEST_TYPE > "${out_csv}"
rc=$?
set -e

# Stop profiler
kill "${prof_pid}" 2>/dev/null || true
wait "${prof_pid}" 2>/dev/null || true
echo "Stopped dumpGpuPower (PID: ${prof_pid})"

if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: STREAM benchmark exited non-zero (rc=${rc})"
    exit "${rc}"
fi

# ---------------------------
# Copy results back to repo
# ---------------------------
dest_node_dir="${SCRATCH_RUN_ROOT}/exp_stream/vista-output/stream-output/${node_id}"
dest_gpu_dir="${dest_node_dir}/${gpu_id}"

rm -rf "${dest_gpu_dir}" 2>/dev/null || true
mkdir -p "${dest_gpu_dir}"

cp -r "${node_dir}/." "${dest_gpu_dir}"
echo "Copied results to: ${dest_gpu_dir}"

# ---------------------------
# Cooldown GPU
# ---------------------------
cooldown_to_baseline "${gpu_id}" "${baseline}" "${TEMP_HEADROOM_C:-1}" "${COOLDOWN_SLEEP_SEC:-5}" "${COOLDOWN_MAX_ITERS:-60}"
