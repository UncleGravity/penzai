#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-projection-service-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY -Wno-WIDTHEXPAND \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --Mdir "$mdir" --top-module projection_service_tb \
  "$repo/fpga/verify/sim/unit/projection/service/tb.sv" \
  "$repo/fpga/rtl/projection/projection_service.v" \
  "$repo/fpga/rtl/projection/emit_stream.v" \
  "$repo/fpga/rtl/projection/logits_sink.v" \
  "$repo/fpga/rtl/projection/gemm.v"
"$mdir/Vprojection_service_tb"
