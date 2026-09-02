#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-shared-q8-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --Mdir "$mdir" --top-module shared_q8_tb \
  "$repo/fpga/verify/sim/unit/vector/shared_q8/tb.sv" \
  "$repo/fpga/rtl/vector/shared_q8.v" \
  "$repo/fpga/rtl/vector/q8_pack4.v" \
  "$repo/fpga/rtl/vector/q8_block.v"
"$mdir/Vshared_q8_tb"
