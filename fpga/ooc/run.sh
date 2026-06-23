#!/usr/bin/env bash
# run.sh - plan-7 phase-2 DSP-inference derisk. OOC-synths the candidate fixed-point
# rowblock (mac_array) vs the current fp32 rowblock (rowblock_ooc) on the Vivado VM,
# same part + same clock, and prints DSP / LUT / CARRY8 / FF / Fmax for each.
#
#   ./run.sh [period_ns]     # default 3.333 (the f300 goal)
#
# Reuses the combined-v1 VM config (VM=...). No bitstream; ~1-2 min/run.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
source ../bitstreams/combined-v1/config.env
: "${VM:?config.env must set VM (Windows Vivado host)}"
VM_OOC="${VM_OOC:-penzai-ooc}"
PERIOD="${1:-3.333}"
FP=../rtl/fp
MM=../rtl/matmul

echo "== sync OOC probe -> $VM:$VM_OOC =="
ssh "$VM" "if not exist $VM_OOC mkdir $VM_OOC"
scp ooc_synth.tcl ooc.bat fp_fixed_mac.v mac_array.v rowblock_ooc.v "$VM:$VM_OOC/"
scp "$FP/fp32_add_pipe.v" "$FP/fp32_mul_pipe.v" "$FP/fp16_to_fp32.v" "$FP/int_to_fp32.v" "$VM:$VM_OOC/"
scp "$MM/matmul_reducer.v" "$MM/matmul_rowblock.v" "$VM:$VM_OOC/"

echo "== [1/2] candidate: mac_array (fixed-point accumulate) @ ${PERIOD}ns =="
ssh "$VM" "cd $VM_OOC && ooc.bat mac_array $PERIOD cand fp_fixed_mac.v mac_array.v" 2>&1 | tee out/cand.log | grep -E '^RESULT|^ERROR|CRITICAL WARNING' || true

echo "== [2/2] baseline: rowblock_ooc (current fp32) @ ${PERIOD}ns =="
ssh "$VM" "cd $VM_OOC && ooc.bat rowblock_ooc $PERIOD base int_to_fp32.v fp16_to_fp32.v fp32_mul_pipe.v fp32_add_pipe.v matmul_reducer.v matmul_rowblock.v rowblock_ooc.v" 2>&1 | tee out/base.log | grep -E '^RESULT|^ERROR|CRITICAL WARNING' || true

echo "== fetch reports -> out/ =="
scp "$VM:$VM_OOC/cand_util.rpt" "$VM:$VM_OOC/cand_timing.rpt" \
    "$VM:$VM_OOC/base_util.rpt" "$VM:$VM_OOC/base_timing.rpt" out/ 2>/dev/null || true

echo "== SUMMARY (period=${PERIOD}ns, part xck26) =="
grep -hE '^RESULT' out/cand.log out/base.log 2>/dev/null || echo "(no RESULT lines -- check out/*.log)"
echo "== run.sh DONE =="
