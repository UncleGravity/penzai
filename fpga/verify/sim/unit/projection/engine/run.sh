#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
tmp="${TMPDIR:-/tmp}"
behavioral_mdir="$tmp/penzai-gemm-verilator"
dsp_mdir="$tmp/penzai-gemm-dsp-verilator"
digit_mdir="$tmp/penzai-digit-verilator"

common_sources=(
  "$repo/fpga/verify/sim/unit/projection/engine/tb.sv"
  "$repo/fpga/rtl/projection/engine.v"
  "$repo/fpga/rtl/projection/digit_accum.v"
  "$repo/fpga/rtl/projection/dot4.v"
  "$repo/fpga/rtl/projection/ternary_select.v"
  "$repo/fpga/rtl/projection/gemm.v"
  "$repo/fpga/rtl/lib/fma.v"
)
verilator_flags=(
  --binary --timing --assert -Wall -Wno-fatal
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL
  -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-BLKSEQ
)

rm -rf "$behavioral_mdir"
verilator "${verilator_flags[@]}" --top-module projection_engine_tb \
  --Mdir "$behavioral_mdir" \
  "${common_sources[@]}"
"$behavioral_mdir/Vprojection_engine_tb"

rm -rf "$dsp_mdir"
verilator "${verilator_flags[@]}" --top-module projection_engine_tb \
  -DPENZAI_XILINX_DSP48E2 \
  --Mdir "$dsp_mdir" \
  "$repo/fpga/verify/support/dsp48e2_model.sv" \
  "${common_sources[@]}"
"$dsp_mdir/Vprojection_engine_tb"

rm -rf "$digit_mdir"
verilator "${verilator_flags[@]}" \
  --top-module digit_accum_tb --Mdir "$digit_mdir" \
  "$repo/fpga/verify/sim/unit/projection/engine/tb_digit.sv" \
  "$repo/fpga/rtl/projection/digit_accum.v" \
  "$repo/fpga/rtl/projection/dot4.v"
"$digit_mdir/Vdigit_accum_tb"

echo "PASS projection engine simulations"
