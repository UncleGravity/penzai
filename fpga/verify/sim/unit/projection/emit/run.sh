#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
cd "$repo"
rm -rf .zig-cache/penzai-emit-obj
verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb \
  --Mdir .zig-cache/penzai-emit-obj \
  fpga/rtl/projection/gemm.v \
  fpga/rtl/lib/fma.v \
  fpga/rtl/projection/emit_stream.v \
  fpga/verify/sim/unit/projection/emit/tb.sv
.zig-cache/penzai-emit-obj/Vtb
