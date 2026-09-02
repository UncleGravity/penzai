#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
out="$repo/.zig-cache/attention-query-gather"
rm -rf "$out"
mkdir -p "$out"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb --Mdir "$out/obj" -o "$out/sim" \
  "$repo/fpga/verify/sim/unit/attention/query_gather/tb.sv" \
  "$repo/fpga/rtl/attention/flash_query_gather.v"

"$out/sim"
