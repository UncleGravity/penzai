#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
period_ns="${1:-3.333}"
out_dir="${2:-$repo/.zig-cache/fpga-verify/vivado-ooc/dot4}"

command -v vivado >/dev/null 2>&1 || {
  echo "ERROR: vivado is not available on PATH" >&2
  exit 1
}

rm -rf "$out_dir"
exec vivado -mode batch \
  -log "$out_dir/vivado.log" \
  -journal "$out_dir/vivado.jou" \
  -source "$repo/fpga/verify/qor/vivado_ooc/projection/route_dot.tcl" \
  -tclargs "$period_ns" "$out_dir" "$repo"
