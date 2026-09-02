#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
tool_dir="$verify_dir/qor/vivado"
config="$repo/fpga/build/config.env"

usage() {
  echo "usage: $0 summary|deep [f<MHz>]" >&2
  exit 2
}

[[ -f "$config" ]] || {
  echo "ERROR: missing $config" >&2
  exit 1
}
set -a
source "$config"
set +a
: "${VM:?config.env must set VM}"
: "${VM_DIR:?config.env must set VM_DIR}"

mode="${1:-summary}"
case "$mode" in summary|deep) ;; *) usage ;; esac
variant="${2:-${VARIANT:-f225}}"
[[ "$variant" =~ ^f[0-9]+$ && $# -le 2 ]] || usage

checkpoint="cache/checkpoints/penzai-$variant-routed.dcp"
git_commit="$(git -C "$repo" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
analysis_sha256="$({
  shasum -a 256 \
    "$tool_dir/analyze.sh" \
    "$tool_dir/analyze.bat" \
    "$tool_dir/summarize.tcl" \
    "$repo/fpga/build/metrics.tcl" \
    "$tool_dir/report.tcl"
} | shasum -a 256 | awk '{print $1}')"
analysis_id="$(date -u +%Y%m%dT%H%M%SZ)-${git_commit}-${mode}-${variant}"
remote_output="analysis/$analysis_id"
local_parent="$repo/.zig-cache/fpga-verify/analysis"
local_output="$local_parent/$analysis_id"

echo "== sync analysis tools -> $VM:$VM_DIR/tools =="
ssh "$VM" "if not exist $VM_DIR\tools mkdir $VM_DIR\tools" || true
scp \
  "$tool_dir/analyze.bat" \
  "$tool_dir/summarize.tcl" \
  "$repo/fpga/build/metrics.tcl" \
  "$tool_dir/report.tcl" \
  "$VM:$VM_DIR/tools/"

echo "== analyze mode=$mode checkpoint=$checkpoint =="
set +e
ssh "$VM" "cd $VM_DIR && tools\analyze.bat $mode $checkpoint $remote_output"
status=$?
set -e

mkdir -p "$local_parent"
scp -r "$VM:$VM_DIR/$remote_output" "$local_parent/" 2>/dev/null || true
[[ -d "$local_output" ]] || {
  echo "ERROR: analysis exited $status and produced no output" >&2
  exit 1
}

for file in "$local_output"/*; do
  [[ -f "$file" ]] || continue
  case "$file" in
    *.log|*.rpt|*.tsv|*.txt)
      tmp="$file.lf.$$"
      LC_ALL=C tr -d '\r' < "$file" > "$tmp"
      mv "$tmp" "$file"
      ;;
  esac
done
printf 'key\tvalue\ndriver_exit_code\t%s\ngit_commit\t%s\nanalysis_sha256\t%s\n' \
  "$status" "$git_commit" "$analysis_sha256" \
  > "$local_output/driver_status.tsv"
ln -sfn "$analysis_id" "$local_parent/latest"

(( status == 0 )) || {
  echo "ERROR: routed analysis failed; partial output: $local_output" >&2
  exit "$status"
}
echo "Analysis artifacts: $local_output"
