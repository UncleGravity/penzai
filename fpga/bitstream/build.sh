#!/usr/bin/env bash
# build.sh - drive the Vivado COMBINED (matmul + flash) bitstream build on the VM.
#
# Syncs the production RTL tree + generated register contracts and address maps, runs
# Vivado, and fetches the .bit/.bit.bin to ./out. Run after
# `cp config.env.example config.env`.
#
#   ./build.sh                       # uses VARIANT from config.env
#   ./build.sh w512-p4-f250          # explicit shared kernel clock
#
# Regenerate the generated inputs first if the regmaps changed:
#   (cd ../.. && zig build regmap)   # writes fpga/regmap/{matmul,flash} contract files

set -euo pipefail
cd "$(dirname "$0")"

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${1:-${VARIANT:-w512-p4-f300}}"
BIT_PREFIX="penzai-combined-v1"
RTL_ROOT="../rtl"
REGMAP_ROOT="../regmap"

# Discover the production RTL recursively so adding a module under rtl/ does not require
# maintaining a second source manifest here. Generated contracts live in regmap/.
RTL_FILES=()
while IFS= read -r f; do
  RTL_FILES+=("$f")
done < <(find "$RTL_ROOT" -type f \( -name '*.v' -o -name '*.vh' \) -print | sort)
RTL_FILES+=(
  "$REGMAP_ROOT/matmul_regs.vh"
  "$REGMAP_ROOT/flash_regs.vh"
)

# scp flattens the source tree on the VM. Reject duplicate basenames instead of letting
# one source silently overwrite another.
duplicate_basenames="$({ for f in "${RTL_FILES[@]}"; do basename "$f"; done; } | sort | uniq -d)"
[[ -z "$duplicate_basenames" ]] || {
  echo "ERROR: duplicate RTL basenames cannot be flattened into VM rtl/:" >&2
  echo "$duplicate_basenames" >&2
  exit 1
}
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
scp build.tcl build.bat "$VM:$VM_DIR/"
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
