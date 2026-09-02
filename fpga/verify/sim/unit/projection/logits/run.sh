#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
cd "$repo"
rm -rf .zig-cache/penzai-logits-obj
verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb \
  --Mdir .zig-cache/penzai-logits-obj \
  fpga/rtl/projection/logits_sink.v \
  fpga/verify/sim/unit/projection/logits/tb.sv
.zig-cache/penzai-logits-obj/Vtb
