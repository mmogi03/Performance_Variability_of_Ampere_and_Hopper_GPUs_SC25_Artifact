#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   collect_node_info.sh <SCRATCH_RUN_ROOT> <TMP_BASE>

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <SCRATCH_RUN_ROOT> <TMP_BASE>"
  exit 1
fi

SCRATCH_RUN_ROOT="$1"
TMP_BASE="$2"

node_id="${SLURMD_NODENAME:-$(hostname)}"
gpu_id="${SLURM_LOCALID:-0}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"

# shellcheck source=/dev/null
source "${SCRATCH_RUN_ROOT}/utils/gpu_info_utils.sh"

uuid="$(gpu_uuid "${gpu_id}")"
name="$(gpu_name "${gpu_id}")"
serial="$(gpu_serial "${gpu_id}")"
pci_bus="$(gpu_pci_bus_id "${gpu_id}")"
vbios="$(gpu_vbios "${gpu_id}")"
driver="$(gpu_driver "${gpu_id}")"

job_id="${SLURM_JOB_ID:-}"
ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"

echo "NODE_INFO: node=${node_id} gpu_id=${gpu_id} uuid=${uuid} name=${name}"

tmp_dir="${TMP_BASE}/node_info"
mkdir -p "${tmp_dir}"
tmp_file="${tmp_dir}/${node_id}_gpu${gpu_id}.csv"

printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
  "${node_id}" "${gpu_id}" "${uuid}" "${name}" "${serial}" "${pci_bus}" "${vbios}" "${driver}" "${job_id}" "${ts}" \
  > "${tmp_file}"

out_dir="${SCRATCH_RUN_ROOT}/node_info/per_node"
mkdir -p "${out_dir}"
cp -f "${tmp_file}" "${out_dir}/"
