#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
BUILD="${TMPDIR:-/tmp}/penzai-engine-rope4-verilator"
rm -rf "$BUILD"

verilator --binary --timing -Wall -Wno-fatal \
  --top-module tb \
  -I"$repo/fpga/rtl/lib" \
  "$repo/fpga/rtl/lib/fmul.v" \
  "$repo/fpga/rtl/lib/fadd.v" \
  "$repo/fpga/rtl/vector/rope4.v" \
  "$repo/fpga/verify/sim/unit/vector/rope/tb.sv" \
  --Mdir "$BUILD"

"$BUILD/Vtb"
