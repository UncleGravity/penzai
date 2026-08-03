#!/usr/bin/env bash
# build.sh - drive the Vivado COMBINED (matmul + flash) bitstream build on the VM.
#
# Syncs the production RTL tree + generated register contracts and address maps, runs
# Vivado, and fetches the .bit/.bit.bin to ./out. Run after
# `cp config.env.example config.env`.
#
#   ./build.sh                              # clean build; uses VARIANT from config.env
#   ./build.sh w512-p4-f250                 # clean build at an explicit clock
#   ./build.sh --incremental                # reuse the last timing-clean checkpoint
#   ./build.sh w512-p4-f250 --incremental   # explicit clock + checkpoint reuse
#
# Regenerate the generated inputs first if the regmaps changed:
#   (cd ../.. && zig build regmap)   # writes fpga/regmap/{matmul,flash} contract files

set -euo pipefail
cd "$(dirname "$0")"

normalize_text_dir() {
  local dir="$1" file tmp
  for file in "$dir"/*; do
    [[ -f "$file" ]] || continue
    case "$file" in
      *.bif|*.log|*.rpt|*.tsv|*.txt|*.xdc)
        tmp="${file}.lf.$$"
        LC_ALL=C tr -d '\r' < "$file" > "$tmp"
        mv "$tmp" "$file"
        ;;
    esac
  done
}

write_source_manifest() {
  local file digest
  printf 'path\tsha256\n'
  for file in "${HASH_FILES[@]}"; do
    digest="$(shasum -a 256 "$file" | awk '{print $1}')"
    printf '%s\t%s\n' "$file" "$digest"
  done
}

write_run_status() {
  local result="$1" host_status="$2"
  printf 'key\tvalue\ndriver_exit_code\t%s\nhost_status\t%s\n' \
    "$build_status" "$host_status" > "$LOCAL_RUN_DIR/driver_status.tsv"
  {
    printf 'FPGA_RUN %s\n' "$result"
    printf 'driver_exit_code=%s host_status=%s\n' "$build_status" "$host_status"
    if [[ -f "$LOCAL_RUN_DIR/vivado_summary.txt" ]]; then
      printf '\n'
      cat "$LOCAL_RUN_DIR/vivado_summary.txt"
    fi
  } > "$LOCAL_RUN_DIR/summary.txt"
}

[[ -f config.env ]] || { echo "ERROR: missing config.env; copy config.env.example first" >&2; exit 1; }
set -a; source config.env; set +a

: "${VM:?config.env must set VM (Windows Vivado host)}"
: "${VM_DIR:?config.env must set VM_DIR (build dir on the VM)}"
VARIANT="${VARIANT:-w512-p4-f300}"
INCREMENTAL="${INCREMENTAL:-0}"
for arg in "$@"; do
  case "$arg" in
    --incremental) INCREMENTAL=1 ;;
    w512-p4-f[0-9]*) VARIANT="$arg" ;;
    *) echo "ERROR: unknown argument '$arg'" >&2; exit 1 ;;
  esac
done
[[ "$VARIANT" =~ ^w512-p4-f[0-9]+$ ]] || {
  echo "ERROR: invalid variant '$VARIANT'; expected w512-p4-f<MHz>" >&2
  exit 1
}
case "$INCREMENTAL" in
  0) BUILD_MODE=clean ;;
  1) BUILD_MODE=incremental ;;
  *) echo "ERROR: INCREMENTAL must be 0 or 1" >&2; exit 1 ;;
esac
BIT_PREFIX="penzai-combined-v1"
REPO_ROOT="../.."
RTL_ROOT="../rtl"
REGMAP_ROOT="../regmap"
METRICS_TCL="../tools/metrics.tcl"

# Discover the production RTL recursively so adding a module under rtl/ does not require
# maintaining a second source manifest here. Generated contracts live in regmap/.
RTL_FILES=()
while IFS= read -r f; do
  RTL_FILES+=("$f")
done < <(find "$RTL_ROOT" -type f \( -name '*.v' -o -name '*.vh' \) -print | LC_ALL=C sort)
RTL_FILES+=(
  "$REGMAP_ROOT/matmul_regs.vh"
  "$REGMAP_ROOT/flash_regs.vh"
)

# scp flattens the source tree on the VM. Reject duplicate basenames instead of letting
# one source silently overwrite another.
duplicate_basenames="$({ for f in "${RTL_FILES[@]}"; do basename "$f"; done; } | LC_ALL=C sort | uniq -d)"
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

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
PROVENANCE_PATHS=(
  fpga/rtl
  fpga/regmap/matmul_regs.vh
  fpga/regmap/flash_regs.vh
  fpga/regmap/matmul_address_map.tcl
  fpga/regmap/flash_address_map.tcl
  fpga/bitstream/build.sh
  fpga/bitstream/build.bat
  fpga/bitstream/build.tcl
  fpga/tools/metrics.tcl
)
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "${PROVENANCE_PATHS[@]}")" ]]; then
  GIT_DIRTY=1
  DIRTY_TAG=-dirty
else
  GIT_DIRTY=0
  DIRTY_TAG=""
fi
HASH_FILES=(build.sh build.tcl build.bat "$METRICS_TCL" "$MATMUL_ADDR" "$FLASH_ADDR" "${RTL_FILES[@]}")
SOURCE_HASH="$(write_source_manifest | shasum -a 256 | awk '{print $1}')"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT}${DIRTY_TAG}-${VARIANT}-${BUILD_MODE}"
LOCAL_RUN_DIR="out/runs/$RUN_ID"
mkdir -p "$LOCAL_RUN_DIR"
write_source_manifest > "$LOCAL_RUN_DIR/source_files.tsv"

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
ssh "$VM" "if not exist $VM_DIR mkdir $VM_DIR" || true
# Clean rtl/ before sync so a changed RTL_FILES list leaves no stale modules for
# build.tcl's `glob ./rtl/*.v` to pick up.
ssh "$VM" "if exist $VM_DIR\\rtl rmdir /s /q $VM_DIR\\rtl" || true
ssh "$VM" "mkdir $VM_DIR\\rtl" || true
scp build.tcl build.bat "$METRICS_TCL" "$VM:$VM_DIR/"
scp "$MATMUL_ADDR" "$VM:$VM_DIR/matmul_address_map.tcl"
scp "$FLASH_ADDR"  "$VM:$VM_DIR/flash_address_map.tcl"
scp "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

echo "== Vivado build run=$RUN_ID variant=$VARIANT mode=$BUILD_MODE on $VM =="
set +e
ssh "$VM" "cd $VM_DIR && build.bat $VARIANT $BUILD_MODE $RUN_ID $GIT_COMMIT $GIT_DIRTY $SOURCE_HASH"
build_status=$?
set -e

echo "== fetch run artifacts -> $LOCAL_RUN_DIR =="
# Fetch partial runs too: manifests, metrics, and early reports are useful when a gate fails.
scp "$VM:$VM_DIR/out/runs/$RUN_ID/*" "$LOCAL_RUN_DIR/" 2>/dev/null || true
normalize_text_dir "$LOCAL_RUN_DIR"
if [[ -f "$LOCAL_RUN_DIR/summary.txt" ]]; then
  mv "$LOCAL_RUN_DIR/summary.txt" "$LOCAL_RUN_DIR/vivado_summary.txt"
fi

if (( build_status != 0 )); then
  write_run_status FAIL remote_driver_failed
  echo "ERROR: Vivado build failed (status $build_status); artifacts are in $LOCAL_RUN_DIR" >&2
  exit "$build_status"
fi

echo "== promote deployable bitstream -> ./out =="
for ext in bit bit.bin; do
  src="$LOCAL_RUN_DIR/$BIT_PREFIX-$VARIANT.$ext"
  if [[ ! -f "$src" ]]; then
    write_run_status FAIL missing_artifact
    echo "ERROR: successful build did not fetch $src" >&2
    exit 1
  fi
done
for ext in bit bit.bin; do
  src="$LOCAL_RUN_DIR/$BIT_PREFIX-$VARIANT.$ext"
  cp "$src" "out/$BIT_PREFIX-$VARIANT.$ext"
done
write_run_status PASS complete
ln -sfn "runs/$RUN_ID" out/latest
ls -la "out/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Run artifacts: $LOCAL_RUN_DIR"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
