#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
# shellcheck source=../../../verify/production_sources.sh
source "$repo/fpga/verify/production_sources.sh"
penzai_load_production_rtl "$repo"

[[ -f "$repo/fpga/regmap/engine_regs.vh" ]] || {
  echo "ERROR: missing generated register contract; run 'zig build regmap'" >&2
  exit 1
}

verilator --lint-only --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY -Wno-WIDTHEXPAND \
  -Wno-WIDTHTRUNC -Wno-BLKSEQ \
  -DPENZAI_XILINX_DSP48E2 \
  --top-module penzai_top \
  -I"$repo/fpga/regmap" \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  -I"$repo/fpga/rtl/lib" \
  -I"$repo/fpga/rtl/attention" \
  "$repo/fpga/verify/support/dsp48e2_model.sv" \
  "${PENZAI_RTL_SOURCES[@]}"
