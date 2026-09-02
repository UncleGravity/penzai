#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-vector-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --Mdir "$mdir" --top-module vector_service_tb \
  "$repo/fpga/verify/sim/unit/vector/service/tb.sv" \
  "$repo/fpga/rtl/vector/vector_service.v" \
  "$repo/fpga/rtl/vector/shared_q8.v" \
  "$repo/fpga/rtl/vector/rms_reduce4.v" \
  "$repo/fpga/rtl/vector/norm_apply4.v" \
  "$repo/fpga/rtl/vector/q8_pack4.v" \
  "$repo/fpga/rtl/vector/q8_block.v" \
  "$repo/fpga/rtl/vector/rms_inverse.v" \
  "$repo/fpga/rtl/lib/fmul.v" \
  "$repo/fpga/rtl/lib/fadd.v"
"$mdir/Vvector_service_tb"
