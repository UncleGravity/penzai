#!/usr/bin/env bash
# build.sh - drive the Vivado COMBINED (matmul + flash) bitstream build on the VM.
#
# Syncs both RTL sets (matmul gemm + flash, both on the shared numeric/ leaves) + both generated
# register headers + the flash LUTs + both generated address maps, runs Vivado, fetches
# the .bit/.bit.bin to ./out. Run after `cp config.env.example config.env`.
#
#   ./build.sh                       # uses VARIANT from config.env
#   ./build.sh w512-p4-f200-wc300    # explicit; f = shared kernel clock (both ops), wc = matmul weight clock
#
# Regenerate the generated inputs first if the regmaps changed:
#   (cd ../../.. && zig build regmap)   # writes fpga/regmap/{matmul,flash} contract files

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${1:-${VARIANT:-w512-p4-f200-wc300}}"
BIT_PREFIX="penzai-combined-v1"
RTL_ROOT="../../rtl"
RTL_FLASH="../../rtl/flash_attn"
RTL_NUMERIC="../../rtl/numeric"
RTL_SEQ="../../rtl/seq"
REGMAP_ROOT="../../regmap"

# Union of both RTL sets. GEMM and flash compose the shared numeric/ leaves; generated
# register contracts come from regmap/. This matches the deployable cosim dependencies.
RTL_FILES=(
  # Plan-7 fixed-point GEMM path.
  "$RTL_ROOT/decode_top.v"
  "$RTL_ROOT/gemm_kernel.v"
  "$RTL_ROOT/gemm.v"
  "$REGMAP_ROOT/matmul_regs.vh"
  # Migrated flash kernel; these pipeline compositions use numeric/ leaves.
  "$RTL_FLASH/flash_top.v"
  "$RTL_FLASH/flash_kernel.v"
  "$RTL_FLASH/fp_dot.v"
  "$RTL_FLASH/fp_axpy8.v"
  "$RTL_FLASH/flash_softmax.v"
  "$REGMAP_ROOT/flash_regs.vh"
  "$RTL_FLASH/flash_luts.vh"
  # numeric/ leaf library: fma (matmul) + the float leaves shared by flash. cvt carries the
  # f16/f32/bf16 conversions (incl. the bf16 p·V seam); reduce composes fadd; exp/recip
  # compose interp; fmt.vh is the format+latency contract every leaf includes.
  "$RTL_NUMERIC/fma.v"
  "$RTL_NUMERIC/cvt.v"
  "$RTL_NUMERIC/fmul.v"
  "$RTL_NUMERIC/fadd.v"
  "$RTL_NUMERIC/reduce.v"
  "$RTL_NUMERIC/exp.v"
  "$RTL_NUMERIC/interp.v"
  "$RTL_NUMERIC/recip.v"
  "$RTL_NUMERIC/fmt.vh"
  # seq.v = the on-PL command executor (batches the per-op PS dispatch, v2.1). seq_top =
  # control slave + command BRAM + cosim-green seq_core + the AXI-Lite replay master.
  "$RTL_SEQ/seq_top.v"
  "$RTL_SEQ/seq_core.v"
  "$RTL_SEQ/seq_reg_master.v"
)
for f in "${RTL_FILES[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run 'zig build regmap' for the *_regs.vh)" >&2; exit 1; }
done

# Both generated address maps (build.tcl sources renamed copies). One source: regmap/.
MATMUL_ADDR="$REGMAP_ROOT/matmul_address_map.tcl"
FLASH_ADDR="$REGMAP_ROOT/flash_address_map.tcl"
for f in "$MATMUL_ADDR" "$FLASH_ADDR"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run 'zig build regmap')" >&2; exit 1; }
done

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR" || true
# Clean rtl/ before sync so a changed RTL_FILES list leaves no stale modules for
# build.tcl's `glob ./rtl/*.v` to pick up.
ssh "$VM" "if exist $VM_DIR\\rtl rmdir /s /q $VM_DIR\\rtl" || true
ssh "$VM" "mkdir $VM_DIR\\rtl" || true
scp tcl/build.tcl build.bat "$VM:$VM_DIR/"
scp "$MATMUL_ADDR" "$VM:$VM_DIR/matmul_address_map.tcl"
scp "$FLASH_ADDR"  "$VM:$VM_DIR/flash_address_map.tcl"
scp "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

echo "== Vivado build variant=$VARIANT on $VM =="
set +e
ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"
build_status=$?
set -e

echo "== fetch reports -> ./out =="
mkdir -p out
# Vivado writes these before build.tcl's timing gate. Always retrieve them, including
# failed builds; otherwise the information needed to diagnose a near miss is stranded
# on the VM because `set -e` exits before this block.
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT"_*.rpt out/ 2>/dev/null || true

if (( build_status != 0 )); then
  echo "ERROR: Vivado build failed (status $build_status); reports, if produced, are in ./out" >&2
  exit "$build_status"
fi

echo "== fetch bitstream -> ./out =="
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" \
    "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" out/
ls -la "out/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
