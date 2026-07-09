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
#   (cd ../../.. && zig build regmap)   # writes matmul_regs.vh, flash_regs.vh, both address_map.tcl

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${1:-${VARIANT:-w512-p4-f200-wc300}}"
BIT_PREFIX="penzai-combined-v1"
RTL_ROOT="../../rtl"
RTL_MATMUL="../../rtl/matmul"
RTL_FLASH="../../rtl/flash_attn"
RTL_FP="../../rtl/fp"
RTL_NUMERIC="../../rtl/numeric"
RTL_SEQ="../../rtl/seq"

# Union of both RTL sets — matmul (gemm) and flash now BOTH compose the numeric/ leaf
# library; rtl/fp is fully retired from the build (no synthesizable consumer left). This
# set is exactly decode_top's + flash_top's cosim deps (build.zig), so it's proven to elaborate.
RTL_FILES=(
  # matmul = the plan-7 fixed-point gemm path (decode_top replaces matmul_top; the legacy
  # matmul_{top,kernel,rowblock,reducer}.v stay on disk for the cosims but are not built).
  "$RTL_ROOT/decode_top.v"
  "$RTL_ROOT/gemm_kernel.v"
  "$RTL_ROOT/gemm.v"
  "$RTL_MATMUL/matmul_regs.vh"
  # flash = the migrated kernel; fp_dot/fp_axpy8/flash_softmax are numeric compositions now
  # (the old fp_addtree/fp_exp/fp_recip/fp_interp are dead — reduce/exp/recip replace them).
  "$RTL_FLASH/flash_top.v"
  "$RTL_FLASH/flash_kernel.v"
  "$RTL_FLASH/fp_dot.v"
  "$RTL_FLASH/fp_axpy8.v"
  "$RTL_FLASH/flash_softmax.v"
  "$RTL_FLASH/flash_regs.vh"
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

# Both generated address maps (build.tcl sources them, renamed). One source: the regmaps.
MATMUL_ADDR="../q1a8-w256-mc/tcl/address_map.tcl"
FLASH_ADDR="../flash-v1/tcl/address_map.tcl"
for f in "$MATMUL_ADDR" "$FLASH_ADDR"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (run 'zig build regmap')" >&2; exit 1; }
done

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR" || true
# Clean rtl/ before sync so a changed RTL_FILES list (matmul_* -> gemm) leaves no stale
# modules for build.tcl's `glob ./rtl/*.v` to pick up (e.g. a dead matmul_top.v).
ssh "$VM" "if exist $VM_DIR\\rtl rmdir /s /q $VM_DIR\\rtl" || true
ssh "$VM" "mkdir $VM_DIR\\rtl" || true
scp tcl/build.tcl build.bat "$VM:$VM_DIR/"
scp "$MATMUL_ADDR" "$VM:$VM_DIR/matmul_address_map.tcl"
scp "$FLASH_ADDR"  "$VM:$VM_DIR/flash_address_map.tcl"
scp "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

# Floorplan pblock (f300 congestion fix — docs/plan-f300-pblock.md). On by default; build.tcl
# applies it impl-only when present. USE_PBLOCK=0 removes any stale copy to A/B an unconstrained
# build. Removed first so the choice is never stale.
ssh "$VM" "if exist $VM_DIR\\pblock.xdc del $VM_DIR\\pblock.xdc" || true
if [[ "${USE_PBLOCK:-1}" == "1" ]]; then
  scp pblock.xdc "$VM:$VM_DIR/"
  echo "   + pblock.xdc synced (floorplan; rebuild with USE_PBLOCK=0 to compare unconstrained)"
fi

echo "== Vivado build variant=$VARIANT on $VM =="
ssh "$VM" "cd $VM_DIR && build.bat $VARIANT"

echo "== fetch outputs -> ./out =="
mkdir -p out
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit.bin" \
    "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT.bit" out/
# Fetch the resource/timing reports too (the combined-fit answer).
scp "$VM:$VM_DIR/out/$BIT_PREFIX-$VARIANT"_*.rpt out/ 2>/dev/null || true
ls -la "out/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
