#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-rms-reduce-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY \
  --Mdir "$mdir" --top-module rms_reduce4_tb \
  "$repo/fpga/verify/sim/unit/vector/rms_reduce/tb.sv" \
  "$repo/fpga/rtl/vector/rms_reduce4.v" \
  "$repo/fpga/rtl/vector/rms_inverse.v" \
  "$repo/fpga/rtl/lib/fmul.v" \
  "$repo/fpga/rtl/lib/fadd.v"
"$mdir/Vrms_reduce4_tb"
