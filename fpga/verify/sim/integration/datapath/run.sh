#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-engine-datapath-verilator"

# shellcheck source=../../../verify/production_sources.sh
source "$repo/fpga/verify/production_sources.sh"
penzai_load_production_rtl "$repo"
datapath_sources=()
for source in "${PENZAI_RTL_SOURCES[@]}"; do
  [[ "$source" == "$repo/fpga/rtl/io/penzai_top.v" ]] ||
    datapath_sources+=("$source")
done

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY -Wno-WIDTHEXPAND \
  -Wno-WIDTHTRUNC -Wno-BLKSEQ \
  --top-module engine_datapath_tb --Mdir "$mdir" \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  -I"$repo/fpga/rtl/lib" \
  -I"$repo/fpga/rtl/attention" \
  "$repo/fpga/verify/sim/integration/datapath/tb.sv" \
  "${datapath_sources[@]}"

"$mdir/Vengine_datapath_tb"
