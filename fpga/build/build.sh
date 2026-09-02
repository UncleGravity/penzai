#!/usr/bin/env bash
# build.sh - drive the production Penzai Vivado bitstream build on the VM.
#
# Syncs the closed production RTL set and generated register artifacts, runs
# Vivado, and fetches the .bit/.bit.bin under .zig-cache/fpga-build. Run after
# `cp config.env.example config.env`.
#
#   ./build.sh                              # clean build; uses VARIANT from config.env
#   ./build.sh f225                         # clean build at an explicit clock
#   ./build.sh --incremental                # reuse the last timing-clean checkpoint
#   ./build.sh f225 --incremental           # explicit clock + checkpoint reuse
#
# Regenerate the generated inputs first if the regmaps changed:
#   (cd ../.. && zig build regmap)

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
[[ "$VM_DIR" =~ ^[A-Za-z0-9._/:-]+$ ]] || {
  echo "ERROR: VM_DIR contains unsupported characters: $VM_DIR" >&2
  exit 1
}
VARIANT="${VARIANT:-f225}"
INCREMENTAL="${INCREMENTAL:-0}"
QUALIFIED_VARIANT=f225

SSH_ARGS=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
if [[ -n "${PENZAI_SSH_IDENTITY:-}" ]]; then
  [[ -r "$PENZAI_SSH_IDENTITY" ]] || {
    echo "ERROR: PENZAI_SSH_IDENTITY is not readable: $PENZAI_SSH_IDENTITY" >&2
    exit 1
  }
  SSH_ARGS+=(-o IdentityAgent=none -o IdentitiesOnly=yes -i "$PENZAI_SSH_IDENTITY")
fi
for arg in "$@"; do
  case "$arg" in
    --incremental) INCREMENTAL=1 ;;
    f[0-9]*) VARIANT="$arg" ;;
    *) echo "ERROR: unknown argument '$arg'" >&2; exit 1 ;;
  esac
done
[[ "$VARIANT" =~ ^f[0-9]+$ ]] || {
  echo "ERROR: invalid variant '$VARIANT'; expected f<MHz>" >&2
  exit 1
}
case "$INCREMENTAL" in
  0) BUILD_MODE=clean ;;
  1) BUILD_MODE=incremental ;;
  *) echo "ERROR: INCREMENTAL must be 0 or 1" >&2; exit 1 ;;
esac
BIT_PREFIX="penzai"
REPO_ROOT="$(cd ../.. && pwd)"
RTL_ROOT="../rtl"
REGMAP_ROOT="../regmap"
METRICS_TCL="metrics.tcl"
RTL_MANIFEST="sources.f"
LOCAL_OUT_ROOT="${PENZAI_FPGA_OUT:-$REPO_ROOT/.zig-cache/fpga-build}"

# The root is deliberately closed: only modules named by the production source
# manifest can enter the Vivado project.
RTL_FILES=()
while IFS= read -r relative; do
  relative="${relative%%#*}"
  relative="${relative#"${relative%%[![:space:]]*}"}"
  relative="${relative%"${relative##*[![:space:]]}"}"
  [[ -n "$relative" ]] || continue
  case "/$relative/" in
    */../*|*/./*)
      echo "ERROR: non-canonical production RTL path: $relative" >&2
      exit 1
      ;;
  esac
  if [[ "$relative" == /* || "$relative" =~ [[:space:]] ]]; then
    echo "ERROR: invalid production RTL path: $relative" >&2
    exit 1
  fi
  case "$relative" in
    *.v|*.sv|*.vh) ;;
    *) echo "ERROR: unsupported production RTL extension: $relative" >&2; exit 1 ;;
  esac
  RTL_FILES+=("$RTL_ROOT/$relative")
done < "$RTL_MANIFEST"
RTL_FILES+=(
  "$REGMAP_ROOT/engine_regs.vh"
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
EXPECTED_RTL_HASHES="$({
  for f in "${RTL_FILES[@]}"; do
    printf '%s\t%s\n' "$(basename "$f")" "$(shasum -a 256 "$f" | awk '{print $1}')"
  done
} | LC_ALL=C sort)"

ENGINE_ADDR="$REGMAP_ROOT/engine_address_map.tcl"
[[ -f "$ENGINE_ADDR" ]] || {
  echo "ERROR: missing $ENGINE_ADDR (run 'zig build regmap')" >&2
  exit 1
}

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
PROVENANCE_PATHS=(
  fpga/rtl
  fpga/regmap/engine_regs.vh
  fpga/regmap/engine_address_map.tcl
  fpga/build/build.sh
  fpga/build/build.bat
  fpga/build/build.tcl
  fpga/build/sources.f
  fpga/build/metrics.tcl
)
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- "${PROVENANCE_PATHS[@]}")" ]]; then
  GIT_DIRTY=1
  DIRTY_TAG=-dirty
else
  GIT_DIRTY=0
  DIRTY_TAG=""
fi
HASH_FILES=(build.sh build.tcl build.bat "$RTL_MANIFEST" "$METRICS_TCL" \
  "$ENGINE_ADDR" "${RTL_FILES[@]}")
SOURCE_HASH="$(write_source_manifest | shasum -a 256 | awk '{print $1}')"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT}${DIRTY_TAG}-${VARIANT}-${BUILD_MODE}"
LOCAL_RUN_DIR="$LOCAL_OUT_ROOT/runs/$RUN_ID"
mkdir -p "$LOCAL_RUN_DIR"
write_source_manifest > "$LOCAL_RUN_DIR/source_files.tsv"

echo "== sync FPGA inputs -> $VM:$VM_DIR =="
ssh "${SSH_ARGS[@]}" "$VM" "if not exist \"$VM_DIR\" mkdir \"$VM_DIR\""
# Clean rtl/ before sync so a changed RTL_FILES list cannot leave stale modules
# for the remote closure checks to admit.
ssh "${SSH_ARGS[@]}" "$VM" "if exist \"$VM_DIR\\rtl\" rmdir /s /q \"$VM_DIR\\rtl\""
ssh "${SSH_ARGS[@]}" "$VM" "if exist \"$VM_DIR\\rtl\" (echo ERROR: could not reset $VM_DIR\\rtl 1>&2 & exit /b 1)"
ssh "${SSH_ARGS[@]}" "$VM" "mkdir \"$VM_DIR\\rtl\""
scp "${SSH_ARGS[@]}" build.tcl build.bat "$METRICS_TCL" "$RTL_MANIFEST" "$VM:$VM_DIR/"
scp "${SSH_ARGS[@]}" "$ENGINE_ADDR" "$VM:$VM_DIR/engine_address_map.tcl"
scp "${SSH_ARGS[@]}" "${RTL_FILES[@]}" "$VM:$VM_DIR/rtl/"

echo "== verify remote RTL closure =="
REMOTE_RTL_HASHES="$(
  ssh "${SSH_ARGS[@]}" "$VM" \
    "powershell.exe -NoProfile -NonInteractive -Command \"Get-ChildItem '$VM_DIR/rtl' -File | ForEach-Object { Write-Output (\$_.Name + [char]9 + (Get-FileHash -Algorithm SHA256 \$_.FullName).Hash.ToLowerInvariant()) }\"" |
    LC_ALL=C tr -d '\r' |
    LC_ALL=C sort
)"
if [[ "$REMOTE_RTL_HASHES" != "$EXPECTED_RTL_HASHES" ]]; then
  echo "ERROR: remote RTL names or hashes differ from the closed local input set" >&2
  diff -u \
    <(printf '%s\n' "$EXPECTED_RTL_HASHES") \
    <(printf '%s\n' "$REMOTE_RTL_HASHES") >&2 || true
  exit 1
fi
echo "Verified ${#RTL_FILES[@]} remote RTL files"

echo "== Vivado build run=$RUN_ID variant=$VARIANT mode=$BUILD_MODE on $VM =="
set +e
ssh "${SSH_ARGS[@]}" "$VM" "cd $VM_DIR && build.bat $VARIANT $BUILD_MODE $RUN_ID $GIT_COMMIT $GIT_DIRTY $SOURCE_HASH"
build_status=$?
set -e

echo "== fetch run artifacts -> $LOCAL_RUN_DIR =="
# Fetch partial runs too: manifests, metrics, and early reports are useful when a gate fails.
scp "${SSH_ARGS[@]}" "$VM:$VM_DIR/out/runs/$RUN_ID/*" "$LOCAL_RUN_DIR/" 2>/dev/null || true
normalize_text_dir "$LOCAL_RUN_DIR"
if [[ -f "$LOCAL_RUN_DIR/summary.txt" ]]; then
  mv "$LOCAL_RUN_DIR/summary.txt" "$LOCAL_RUN_DIR/vivado_summary.txt"
fi

if (( build_status != 0 )); then
  write_run_status FAIL remote_driver_failed
  echo "ERROR: Vivado build failed (status $build_status); artifacts are in $LOCAL_RUN_DIR" >&2
  exit "$build_status"
fi

echo "== validate fetched bitstream artifacts =="
for ext in bit bit.bin; do
  src="$LOCAL_RUN_DIR/$BIT_PREFIX-$VARIANT.$ext"
  if [[ ! -f "$src" ]]; then
    write_run_status FAIL missing_artifact
    echo "ERROR: successful build did not fetch $src" >&2
    exit 1
  fi
done
if [[ "$VARIANT" != "$QUALIFIED_VARIANT" || "$BUILD_MODE" != clean ]]; then
  write_run_status PASS complete_unpromoted
  echo "Run completed but was not promoted: production requires a clean $QUALIFIED_VARIANT build"
  echo "Run artifacts: $LOCAL_RUN_DIR"
  exit 0
fi
echo "== promote deployable bitstream -> $LOCAL_OUT_ROOT =="
for ext in bit bit.bin; do
  src="$LOCAL_RUN_DIR/$BIT_PREFIX-$VARIANT.$ext"
  cp "$src" "$LOCAL_OUT_ROOT/$BIT_PREFIX-$VARIANT.$ext"
done
write_run_status PASS complete
ln -sfn "runs/$RUN_ID" "$LOCAL_OUT_ROOT/latest"
ls -la "$LOCAL_OUT_ROOT/$BIT_PREFIX-$VARIANT.bit.bin"
echo "Run artifacts: $LOCAL_RUN_DIR"
echo "Done. Deploy with: ./deploy.sh $VARIANT"
