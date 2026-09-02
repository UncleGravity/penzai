#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"

# Portable full-engine structural check.
yosys -q -p "
read_verilog \
  $repo/fpga/rtl/projection/dot4.v \
  $repo/fpga/rtl/projection/digit_accum.v \
  $repo/fpga/rtl/projection/engine.v \
  $repo/fpga/rtl/projection/ternary_select.v \
  $repo/fpga/rtl/projection/gemm.v \
  $repo/fpga/rtl/lib/fma.v;
hierarchy -check -top projection_engine;
proc;
check;
"

# The explicit implementation contains one DSP per dot row.
yosys -q -p "
read_verilog -lib +/xilinx/cells_sim.v;
read_verilog -lib +/xilinx/cells_xtra.v;
read_verilog -DPENZAI_XILINX_DSP48E2 \
  $repo/fpga/rtl/projection/dot4.v;
synth_xilinx -family xcu -top dot4;
flatten;
select -assert-count 512 t:DSP48E2;
check;
"

# The serialized post-dot path adds two scale DSPs per output row.
yosys -q -p "
read_verilog -lib +/xilinx/cells_sim.v;
read_verilog -lib +/xilinx/cells_xtra.v;
read_verilog -DPENZAI_XILINX_DSP48E2 \
  $repo/fpga/rtl/projection/dot4.v \
  $repo/fpga/rtl/projection/digit_accum.v \
  $repo/fpga/rtl/projection/engine.v \
  $repo/fpga/rtl/projection/ternary_select.v \
  $repo/fpga/rtl/projection/gemm.v \
  $repo/fpga/rtl/lib/fma.v;
synth_xilinx -family xcu -top projection_engine;
flatten;
select -assert-count 544 t:DSP48E2;
check;
"

echo "PASS projection structural checks"
