#!/usr/bin/env bash
set -euo pipefail
verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
OBJ="$repo/.zig-cache/penzai-small-read-mux-obj"
rm -rf "$OBJ"
verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb --Mdir "$OBJ" \
  "$repo/fpga/rtl/io/small_read_mux.v" "$repo/fpga/verify/sim/unit/io/small_read_mux/tb.sv"
"$OBJ/Vtb"
