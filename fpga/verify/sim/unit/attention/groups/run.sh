#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
out="$repo/.zig-cache/attention-groups"
rm -rf "$out"
mkdir -p "$out"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --top-module tb --Mdir "$out/obj" -o "$out/sim" \
  -I"$repo/fpga/rtl" \
  -I"$repo/fpga/rtl/lib" \
  -I"$repo/fpga/rtl/attention" \
  "$repo/fpga/verify/sim/unit/attention/groups/tb.sv" \
  "$repo/fpga/rtl/attention/flash_groups8.v" \
  "$repo/fpga/rtl/attention/flash_kernel.v" \
  "$repo/fpga/rtl/attention/fp_dot.v" \
  "$repo/fpga/rtl/attention/fp_axpy8.v" \
  "$repo/fpga/rtl/attention/flash_softmax.v" \
  "$repo/fpga/rtl/lib/exp.v" \
  "$repo/fpga/rtl/lib/interp.v" \
  "$repo/fpga/rtl/lib/fadd.v" \
  "$repo/fpga/rtl/lib/fmul.v" \
  "$repo/fpga/rtl/lib/cvt.v" \
  "$repo/fpga/rtl/lib/reduce.v" \
  "$repo/fpga/rtl/lib/recip.v"

"$out/sim"
