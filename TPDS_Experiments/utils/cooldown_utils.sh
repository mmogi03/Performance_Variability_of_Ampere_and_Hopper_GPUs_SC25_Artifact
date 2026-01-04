#!/usr/bin/env bash
set -euo pipefail

cooldown_log() {
  local msg="$*"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[cooldown][${ts}] ${msg}"
}

get_gpu_temp_c() {
  local gpu_id="$1"
  local t=""
  t="$(nvidia-smi -i "${gpu_id}" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
      | tr -d '[:space:]' || true)"
  if [[ "${t}" =~ ^[0-9]+$ ]]; then
    echo "${t}"
  else
    echo ""
  fi
}

cooldown_to_baseline() {
  local gpu_id="$1"
  local baseline="$2"
  local headroom="${3:-1}"
  local sleep_sec="${4:-5}"
  local max_iters="${5:-60}"

  if [[ -z "${baseline}" || ! "${baseline}" =~ ^[0-9]+$ ]]; then
    cooldown_log "invalid baseline ('${baseline}'); skipping cooldown."
    return 0
  fi

  cooldown_log "baseline=${baseline}C headroom=+${headroom}C sleep=${sleep_sec}s max_iters=${max_iters}"
  local it=0
  while true; do
    local cur
    cur="$(get_gpu_temp_c "${gpu_id}")"
    if [[ -z "${cur}" ]]; then
      cooldown_log "could not read temperature; stopping cooldown."
      break
    fi

    if (( cur <= baseline + headroom )); then
      cooldown_log "done: cur=${cur}C <= ${baseline}+${headroom}"
      break
    fi

    ((it++)) || true
    if (( it >= max_iters )); then
      cooldown_log "reached max iterations: cur=${cur}C (stopping)."
      break
    fi

    sleep "${sleep_sec}"
  done
}
