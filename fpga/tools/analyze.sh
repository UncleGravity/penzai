#!/usr/bin/env bash
# Run routed-checkpoint analysis on the Vivado VM and fetch a run-scoped bundle.

set -euo pipefail
cd "$(dirname "$0")"

normalize_text_dir() {
  local dir="$1" file tmp
  for file in "$dir"/*; do
    [[ -f "$file" ]] || continue
    case "$file" in
      *.log|*.rpt|*.tsv|*.txt)
        tmp="${file}.lf.$$"
        LC_ALL=C tr -d '\r' < "$file" > "$tmp"
        mv "$tmp" "$file"
        ;;
    esac
  done
}

usage() {
  echo "usage: ./analyze.sh summary|deep|check gemm-acc [w512-p4-f<MHz>]" >&2
  exit 2
}

[[ -f ../bitstream/config.env ]] || {
  echo "ERROR: missing ../bitstream/config.env" >&2
  exit 1
}
set -a; source ../bitstream/config.env; set +a
: "${VM:?config.env must set VM}"
: "${VM_DIR:?config.env must set VM_DIR}"

MODE="${1:-summary}"
shift $(( $# > 0 ? 1 : 0 ))
if [[ "$MODE" == check ]]; then
  [[ "${1:-}" == gemm-acc ]] || usage
  MODE=gemm-acc
  shift
fi
case "$MODE" in summary|deep|gemm-acc) ;; *) usage ;; esac

VARIANT="${1:-${VARIANT:-w512-p4-f300}}"
[[ "$VARIANT" =~ ^w512-p4-f[0-9]+$ ]] || usage
[[ $# -le 1 ]] || usage

BIT_PREFIX=penzai-combined-v1
CHECKPOINT="cache/checkpoints/$BIT_PREFIX-$VARIANT-routed.dcp"
GIT_COMMIT="$(git -C ../.. rev-parse --short=12 HEAD)"
ANALYSIS_SHA256="$({ shasum -a 256 analyze.sh analyze.bat summarize.tcl metrics.tcl report.tcl report_gemm_acc.tcl; } \
  | shasum -a 256 | awk '{print $1}')"
ANALYSIS_ID="$(date -u +%Y%m%dT%H%M%SZ)-${GIT_COMMIT}-${MODE}-${VARIANT}"
REMOTE_OUTPUT="analysis/$ANALYSIS_ID"
LOCAL_PARENT="../bitstream/out/analysis"
LOCAL_OUTPUT="$LOCAL_PARENT/$ANALYSIS_ID"

echo "== sync analysis tools -> $VM:$VM_DIR/tools =="
ssh "$VM" "if not exist $VM_DIR\tools mkdir $VM_DIR\tools" || true
scp analyze.bat summarize.tcl metrics.tcl report.tcl report_gemm_acc.tcl "$VM:$VM_DIR/tools/"

echo "== analyze mode=$MODE checkpoint=$CHECKPOINT =="
set +e
ssh "$VM" "cd $VM_DIR && tools\analyze.bat $MODE $CHECKPOINT $REMOTE_OUTPUT"
analysis_status=$?
set -e

mkdir -p "$LOCAL_PARENT"
scp -r "$VM:$VM_DIR/$REMOTE_OUTPUT" "$LOCAL_PARENT/" 2>/dev/null || true
if [[ ! -d "$LOCAL_OUTPUT" ]]; then
  echo "ERROR: analysis completed with status $analysis_status but no output was fetched" >&2
  exit 1
fi
normalize_text_dir "$LOCAL_OUTPUT"
printf 'key\tvalue\ndriver_exit_code\t%s\ngit_commit\t%s\nanalysis_sha256\t%s\n' \
  "$analysis_status" "$GIT_COMMIT" "$ANALYSIS_SHA256" \
  > "$LOCAL_OUTPUT/driver_status.tsv"
ln -sfn "$ANALYSIS_ID" "$LOCAL_PARENT/latest"

if (( analysis_status != 0 )); then
  echo "ERROR: routed analysis failed (status $analysis_status); partial output: $LOCAL_OUTPUT" >&2
  exit "$analysis_status"
fi
echo "Analysis artifacts: $LOCAL_OUTPUT"
