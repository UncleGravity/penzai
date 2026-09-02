#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
OBJ="$repo/.zig-cache/penzai-embedding-service-obj"
rm -rf "$OBJ"
mkdir -p "$OBJ"

verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb --Mdir "$OBJ" \
  "$repo/fpga/verify/sim/unit/vector/embedding_service/tb.sv" \
  "$repo/fpga/rtl/vector/embedding_service.v" \
  "$repo/fpga/rtl/vector/embedding_decode.v" \
  "$repo/fpga/rtl/vector/embedding_store4.v" \
  "$repo/fpga/rtl/lib/cvt.v" \
  -o sim
"$OBJ/sim"
