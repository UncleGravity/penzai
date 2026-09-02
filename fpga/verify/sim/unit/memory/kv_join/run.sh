#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
out="$repo/.zig-cache/kv-join"
rm -rf "$out"
mkdir -p "$out"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb --Mdir "$out/obj" -o "$out/sim" \
  "$repo/fpga/verify/sim/unit/memory/kv_join/tb.sv" \
  "$repo/fpga/rtl/attention/kv_join8.v"

"$out/sim"
