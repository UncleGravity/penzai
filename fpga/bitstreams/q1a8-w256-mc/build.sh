#!/usr/bin/env bash
# build.sh - drive the Vivado bitstream build on the Windows VM.
#
# Syncs the v8 RTL + generated regmap header + the TCL/BAT to the VM,
# runs Vivado (build.bat) over ssh, and fetches the .bit/.bit.bin back to ./out.
# Run from this directory after `cp config.env.example config.env` and editing it.
#
#   ./build.sh                         # uses VARIANT from config.env
#   ./build.sh w512-p4-f125-wc250      # explicit variant
#
# matmul_top `include`s matmul_regs.vh; regenerate it first if the regmap
# changed:  (cd ../../.. && zig build regmap)

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${1:-${VARIANT:-w512-p4-f125-wc250}}"
BIT_PREFIX="penzai-q1a8-mc"
RTL_MATMUL="../../rtl/matmul"
RTL_FP="../../rtl/fp"

# v8 multi-column RTL set. matmul_regs.vh must exist — generate with
# `zig build regmap`.
RTL_FILES=(
  "$RTL_MATMUL/matmul_top.v"
  "$RTL_MATMUL/matmul_kernel.v"
  "$RTL_MATMUL/matmul_rowblock.v"
  "$RTL_MATMUL/matmul_reducer.v"
  "$RTL_FP/fp32_add_pipe.v"
  "$RTL_FP/fp32_mul_pipe.v"
  "$RTL_FP/fp16_to_fp32.v"
  "$RTL_FP/int_to_fp32.v"
  "$RTL_MATMUL/matmul_regs.vh"
)
for f in "${RTL_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run 'zig build regmap' for matmul_regs.vh)" >&2; exit 1; }
done

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
# Windows cmd `mkdir` (no -p); ignore "already exists".
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR" || true
ssh "$VM" "if not exist $VM_DIR\\rtl mkdir $VM_DIR\\rtl" || true
scp tcl/build.tcl build.bat "$VM:$VM_DIR/"
scp "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

echo "== Vivado build variant=$VARIANT on $VM =="
ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"

echo "== fetch outputs -> ./out =="
mkdir -p out
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" \
    "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" out/
ls -la "out/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
