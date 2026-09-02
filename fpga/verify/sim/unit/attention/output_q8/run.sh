#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
OBJ="$repo/.zig-cache/penzai-flash-output-q8-obj"
rm -rf "$OBJ"
mkdir -p "$OBJ"

verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb --Mdir "$OBJ" \
  "$repo/fpga/verify/sim/unit/attention/output_q8/tb.sv" \
  "$repo/fpga/rtl/attention/flash_output_q8.v" \
  "$repo/fpga/rtl/vector/q8_pack4.v" \
  "$repo/fpga/rtl/vector/q8_block.v" \
  -o sim
"$OBJ/sim"
