#!/usr/bin/env bash
# build.sh - drive the Vivado flash bitstream build on the Windows VM.
#
# Syncs the flash RTL + generated regmap header + LUT header + address map + the
# TCL/BAT to the VM, runs Vivado (build.bat) over ssh, and fetches the .bit/.bit.bin
# to ./out. Run from this directory after `cp config.env.example config.env`.
#
#   ./build.sh           # uses VARIANT from config.env
#   ./build.sh f100      # explicit variant (f<MHz>)
#
# flash_top `include`s flash_regs.vh and fp_exp/fp_recip `include` flash_luts.vh;
# regenerate flash_regs.vh first if the regmap changed:  (cd ../../.. && zig build regmap)

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${1:-${VARIANT:-f100}}"
BIT_PREFIX="penzai-flash-v1"
RTL_FLASH="../../rtl/flash_attn"
RTL_FP="../../rtl/fp"

RTL_FILES=(
  "$RTL_FLASH/flash_top.v"
  "$RTL_FLASH/flash_kernel.v"
  "$RTL_FLASH/fp_dot.v"
  "$RTL_FLASH/flash_softmax.v"
  "$RTL_FLASH/fp_exp.v"
  "$RTL_FLASH/fp_recip.v"
  "$RTL_FLASH/fp_interp.v"
  "$RTL_FP/fp32_add_pipe.v"
  "$RTL_FP/fp32_mul_pipe.v"
  "$RTL_FP/fp16_to_fp32.v"
  "$RTL_FP/int_to_fp32.v"
  "$RTL_FLASH/flash_regs.vh"
  "$RTL_FLASH/flash_luts.vh"
)
for f in "${RTL_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run 'zig build regmap' for flash_regs.vh)" >&2; exit 1; }
done

ADDR_MAP="tcl/address_map.tcl"
[[ -f "$ADDR_MAP" ]] || { echo "ERROR: missing $ADDR_MAP (run 'zig build regmap')" >&2; exit 1; }

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR" || true
ssh "$VM" "if not exist $VM_DIR\\rtl mkdir $VM_DIR\\rtl" || true
scp tcl/build.tcl "$ADDR_MAP" build.bat "$VM:$VM_DIR/"
scp "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

echo "== Vivado build variant=$VARIANT on $VM =="
ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"

echo "== fetch outputs -> ./out =="
mkdir -p out
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" \
    "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" out/
ls -la "out/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
