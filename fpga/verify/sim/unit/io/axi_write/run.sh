#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
OBJ="${TMPDIR:-/tmp}/penzai-engine-axi-write-obj"
rm -rf "$OBJ"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-PROCASSINIT -Wno-BLKSEQ \
  --Mdir "$OBJ" --top-module tb \
  "$repo/fpga/rtl/io/axi_write128.v" \
  "$repo/fpga/verify/sim/unit/io/axi_write/tb.sv"
"$OBJ/Vtb"
