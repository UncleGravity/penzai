#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
cd "$repo"

rm -rf .zig-cache/penzai-embedding-obj
verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb \
  --Mdir .zig-cache/penzai-embedding-obj \
  fpga/rtl/lib/cvt.v \
  fpga/rtl/vector/embedding_decode.v \
  fpga/verify/sim/unit/vector/embedding_decode/tb.sv
.zig-cache/penzai-embedding-obj/Vtb
