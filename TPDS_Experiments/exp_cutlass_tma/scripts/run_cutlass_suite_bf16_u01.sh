#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_cutlass_suite_bf16_u01.sh <SCRATCH_RUN_ROOT> <TMP_RUN_ROOT>
#
# Expects:
#   <SCRATCH_RUN_ROOT>/shared/dumpGpuPower built
#   <SCRATCH_RUN_ROOT>/exp_cutlass_tma/build/{wgmma_sm90_bench,wgmma_tma_sm90_bench} built

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <SCRATCH_RUN_ROOT> <TMP_RUN_ROOT>"
  exit 1
fi

SCRATCH_RUN_ROOT="$1"
TMP_RUN_ROOT="$2"

EXP_DIR="${SCRATCH_RUN_ROOT}/exp_cutlass_tma"
BIN_NON_TMA="${EXP_DIR}/build/wgmma_sm90_bench"
BIN_TMA="${EXP_DIR}/build/wgmma_tma_sm90_bench"
PROFILER="${SCRATCH_RUN_ROOT}/shared/dumpGpuPower"

node_id="${SLURMD_NODENAME:-$(hostname)}"
gpu_id="${SLURM_LOCALID:-0}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# shellcheck source=/dev/null
source "${SCRATCH_RUN_ROOT}/utils/gpu_info_utils.sh"
# shellcheck source=/dev/null
source "${SCRATCH_RUN_ROOT}/utils/cooldown_utils.sh"

uuid="$(gpu_uuid "${gpu_id}")"
gname="$(gpu_name "${gpu_id}")"

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[${ts}][${node_id}][gpu${gpu_id}] $*"
}

if [[ ! -x "${BIN_NON_TMA}" ]]; then
  log "ERROR: Missing/non-executable: ${BIN_NON_TMA}"
  exit 1
fi
if [[ ! -x "${BIN_TMA}" ]]; then
  log "ERROR: Missing/non-executable: ${BIN_TMA}"
  exit 1
fi
if [[ ! -x "${PROFILER}" ]]; then
  log "ERROR: Missing/non-executable: ${PROFILER}"
  exit 1
fi

log "Starting CUTLASS minimal suite"
log "GPU: id=${gpu_id} uuid=${uuid} name=${gname}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

baseline="$(get_gpu_temp_c "${gpu_id}")"
if [[ -z "${baseline}" ]]; then
  log "ERROR: Could not read baseline GPU temperature."
  exit 1
fi
log "Baseline locked once: ${baseline}C"

POWER_SAMPLE_RATE="${POWER_SAMPLE_RATE:-0.25}"
POWER_SAMPLES="${POWER_SAMPLES:--1}"
prof_pid=""

start_profiler() {
  local out_file="$1"
  log "Starting dumpGpuPower -> ${out_file}"
  (
    "${PROFILER}" -d "${gpu_id}" -r "${POWER_SAMPLE_RATE}" -n "${POWER_SAMPLES}" > "${out_file}"
  ) &
  prof_pid="$!"
}

stop_profiler() {
  if [[ -n "${prof_pid}" ]]; then
    log "Stopping dumpGpuPower (pid=${prof_pid})"
    kill -2 "${prof_pid}" 2>/dev/null || true
    wait "${prof_pid}" 2>/dev/null || true
    prof_pid=""
  fi
}

cleanup() { stop_profiler || true; }
trap cleanup EXIT

run_case() {
  local flavor="$1"   # non_tma | tma
  local bin="$2"

  local m="32768"
  local n="32768"
  local k="32768"
  local repeats="100"
  local warmup="20"
  local device="${gpu_id}"
  local dtype="bf16"
  local transA="N"
  local transB="T"
  local init="u01"

  local out_dir="${TMP_RUN_ROOT}/results/${node_id}/${flavor}/BF16/U01"
  mkdir -p "${out_dir}"

  local bench_csv="${out_dir}/bench_gpu${gpu_id}.csv"
  local power_csv="${out_dir}/gpu_power_gpu${gpu_id}.csv"

  log "------------------------------------------------------------"
  log "RUN: flavor=${flavor} dtype=${dtype} init=${init} trans=${transA}${transB}"
  log "Dims: m=${m} n=${n} k=${k} repeats=${repeats} warmup=${warmup} device=${device}"
  log "Baseline temp (locked): ${baseline}C"
  log "Output: ${bench_csv}"
  log "Power:  ${power_csv}"

  start_profiler "${power_csv}"

  local -a cmd
  cmd=( "${bin}"
        --m "${m}" --n "${n}" --k "${k}"
        --repeats "${repeats}" --warmup "${warmup}"
        --device "${device}"
        --dtype "${dtype}"
        --transA "${transA}" --transB "${transB}"
        --init "${init}"
        --check
        --check-samples "4096"
        --check-abs "1e-2f"
        --check-rel "1e-1f"
        --check-seed "1234"
      )

  log "CMD: ${cmd[*]}"

  set +e
  "${cmd[@]}" > "${bench_csv}"
  rc=$?
  set -e

  stop_profiler

  if [[ ${rc} -ne 0 ]]; then
    log "ERROR: benchmark exited non-zero (rc=${rc})"
    return ${rc}
  fi

  log "DONE: wrote ${bench_csv} and ${power_csv}"
  return 0
}

run_case "non_tma" "${BIN_NON_TMA}"
cooldown_to_baseline "${gpu_id}" "${baseline}" "${TEMP_HEADROOM_C:-1}" "${COOLDOWN_SLEEP_SEC:-5}" "${COOLDOWN_MAX_ITERS:-60}"

run_case "tma" "${BIN_TMA}"

cooldown_to_baseline "${gpu_id}" "${baseline}" "${TEMP_HEADROOM_C:-1}" "${COOLDOWN_SLEEP_SEC:-5}" "${COOLDOWN_MAX_ITERS:-60}"

DEST_DIR="${SCRATCH_RUN_ROOT}/exp_cutlass_tma/results/${node_id}"
SRC_DIR="${TMP_RUN_ROOT}/results/${node_id}"

log "Copying /tmp -> ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${SRC_DIR}/" "${DEST_DIR}/"
else
  cp -r "${SRC_DIR}/." "${DEST_DIR}/"
fi

log "CUTLASS minimal suite complete."
