#!/usr/bin/env bash
set -euo pipefail

gpu_query_field() {
  local gpu_id="$1"
  local field="$2"
  nvidia-smi -i "${gpu_id}" --query-gpu="${field}" --format=csv,noheader,nounits 2>/dev/null || echo ""
}

gpu_uuid()      { gpu_query_field "$1" "uuid" | tr -d '[:space:]'; }
gpu_name()      { gpu_query_field "$1" "name" | sed 's/[[:space:]]*$//'; }
gpu_serial()    { gpu_query_field "$1" "serial" | tr -d '[:space:]'; }
gpu_pci_bus_id(){ gpu_query_field "$1" "pci.bus_id" | tr -d '[:space:]'; }
gpu_vbios()     { gpu_query_field "$1" "vbios_version" | tr -d '[:space:]'; }
gpu_driver()    { gpu_query_field "$1" "driver_version" | tr -d '[:space:]'; }
