#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-projection-sink-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  -I"$repo/fpga/rtl/lib" \
  --Mdir "$mdir" --top-module projection_sink_tb \
  "$repo/fpga/verify/sim/unit/projection/sink/tb.sv" \
  "$repo/fpga/rtl/projection/sink.v" \
  "$repo/fpga/rtl/vector/shared_q8.v" \
  "$repo/fpga/rtl/vector/rms_reduce4.v" \
  "$repo/fpga/rtl/vector/norm_apply4.v" \
  "$repo/fpga/rtl/vector/q8_pack4.v" \
  "$repo/fpga/rtl/vector/residual4.v" \
  "$repo/fpga/rtl/vector/rope4.v" \
  "$repo/fpga/rtl/vector/f32_to_f16.v" \
  "$repo/fpga/rtl/vector/q8_block.v" \
  "$repo/fpga/rtl/vector/swiglu.v" \
  "$repo/fpga/rtl/vector/rms_inverse.v" \
  "$repo/fpga/rtl/lib/fmul.v" \
  "$repo/fpga/rtl/lib/fadd.v"
"$mdir/Vprojection_sink_tb"
