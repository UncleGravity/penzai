#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
out="$repo/.zig-cache/fpga-verify/yosys/production-map"
mkdir -p "$out"

# shellcheck source=../../../verify/production_sources.sh
source "$repo/fpga/verify/production_sources.sh"
penzai_load_production_rtl "$repo"

yosys -q -l "$out/yosys.log" -p "
  read_verilog -lib +/xilinx/cells_sim.v;
  read_verilog -lib +/xilinx/cells_xtra.v;
  read_verilog -DPENZAI_XILINX_DSP48E2 \
    -I$repo/fpga/regmap \
    -I$repo/fpga/rtl/engine -I$repo/fpga/rtl/vector \
    -I$repo/fpga/rtl/lib \
    -I$repo/fpga/rtl/attention \
    ${PENZAI_RTL_SOURCES[*]};
  hierarchy -check -top penzai_top;
  synth_xilinx -family xcup -top penzai_top -noiopad;
  flatten;
  select -assert-count 695 t:DSP48E2;
  select -assert-count 8 t:URAM288;
  tee -o $out/stat.txt stat;
  check;
"
