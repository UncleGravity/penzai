#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-engine-pl-top-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  -Wno-BLKSEQ -Wno-PINMISSING \
  --top-module penzai_top_tb --Mdir "$mdir" \
  -I"$repo/fpga/regmap" \
  "$repo/fpga/verify/sim/integration/top/tb.sv" \
  "$repo/fpga/verify/support/datapath_stub.sv" \
  "$repo/fpga/rtl/engine/metrics.v" \
  "$repo/fpga/rtl/io/penzai_top.v"

"$mdir/Vpenzai_top_tb"
