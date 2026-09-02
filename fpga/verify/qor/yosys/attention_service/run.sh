#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
yosys -Q -p "
  read_verilog -sv -I$repo/fpga/rtl -I$repo/fpga/rtl/lib -I$repo/fpga/rtl/attention \
    $repo/fpga/rtl/attention/attention_service.v \
    $repo/fpga/rtl/attention/flash_query_gather.v \
    $repo/fpga/rtl/attention/kv_join8.v \
    $repo/fpga/rtl/attention/flash_groups8.v \
    $repo/fpga/rtl/attention/flash_output_q8.v \
    $repo/fpga/rtl/vector/q8_pack4.v \
    $repo/fpga/rtl/vector/q8_block.v \
    $repo/fpga/rtl/attention/flash_kernel.v \
    $repo/fpga/rtl/attention/fp_dot.v \
    $repo/fpga/rtl/attention/fp_axpy8.v \
    $repo/fpga/rtl/attention/flash_softmax.v \
    $repo/fpga/rtl/lib/exp.v \
    $repo/fpga/rtl/lib/interp.v \
    $repo/fpga/rtl/lib/fadd.v \
    $repo/fpga/rtl/lib/fmul.v \
    $repo/fpga/rtl/lib/cvt.v \
    $repo/fpga/rtl/lib/reduce.v \
    $repo/fpga/rtl/lib/recip.v;
  hierarchy -check -top  attention_service;
  synth_xilinx -family xcup -top  attention_service -noiopad;
  check;
  stat -tech xilinx;
"
