#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
obj="${TMPDIR:-/tmp}/penzai-engine-quad-arbiter-obj"
rm -rf "$obj"

verilator --binary --timing --assert -Wall -Wno-fatal \
  --Mdir "$obj" --top-module tb \
  "$repo/fpga/rtl/io/quad_read_arbiter.v" \
  "$repo/fpga/verify/sim/unit/io/quad_read_arbiter/tb.sv"
"$obj/Vtb"
