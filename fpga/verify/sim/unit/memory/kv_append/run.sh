#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
obj="$repo/.zig-cache/penzai-kv-append-obj"
rm -rf "$obj"
mkdir -p "$obj"
verilator --binary --timing -Wall -Wno-fatal --top-module tb \
  --Mdir "$obj" \
  "$repo/fpga/verify/sim/unit/memory/kv_append/tb.sv" \
  "$repo/fpga/rtl/attention/kv_append8.v" -o sim
"$obj/sim"
