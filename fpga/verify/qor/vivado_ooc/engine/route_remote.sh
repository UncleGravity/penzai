#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
qor_dir="$repo/fpga/verify/qor/vivado_ooc/engine"
config="$repo/fpga/build/config.env"
period_ns="${1:-3.333}"
out_dir="${2:-$repo/.zig-cache/fpga-verify/vivado-ooc/engine-remote}"
manifest="$qor_dir/sources.f"

test -f "$config" || {
  echo "ERROR: missing $config" >&2
  exit 1
}
set -a
source "$config"
set +a
: "${VM:?config.env must set VM}"
VM_OOC="${VM_OOC:-penzai-ooc}"

ssh_args=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10)
if [[ -n "${PENZAI_SSH_IDENTITY:-}" ]]; then
  test -r "$PENZAI_SSH_IDENTITY" || {
    echo "ERROR: unreadable SSH identity: $PENZAI_SSH_IDENTITY" >&2
    exit 1
  }
  ssh_args+=(-o IdentityAgent=none -o IdentitiesOnly=yes \
             -i "$PENZAI_SSH_IDENTITY")
fi

stage="$(mktemp -d "${TMPDIR:-/tmp}/penzai-route.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$out_dir"
cp "$qor_dir/route.tcl" "$qor_dir/route.bat" "$stage/"
: > "$stage/ooc_sources.f"
printf 'sha256\tbytes\tpath\n' > "$out_dir/resolved_sources.tsv"
for driver in route.tcl route.bat; do
  file="$qor_dir/$driver"
  printf '%s\t%s\t%s\n' \
    "$(shasum -a 256 "$file" | awk '{print $1}')" \
    "$(wc -c < "$file" | tr -d ' ')" \
    "fpga/verify/qor/vivado_ooc/engine/$driver" \
    >> "$out_dir/resolved_sources.tsv"
done

basenames=""
while IFS= read -r source; do
  case "$source" in ''|'#'*) continue ;; esac
  file="$repo/$source"
  test -f "$file" || { echo "ERROR: missing source: $source" >&2; exit 1; }
  base="${source##*/}"
  [[ " $basenames " != *" $base "* ]] || {
    echo "ERROR: duplicate flattened source basename: $base" >&2
    exit 1
  }
  basenames+=" $base"
  cp "$file" "$stage/$base"
  printf '%s\n' "$base" >> "$stage/ooc_sources.f"
  printf '%s\t%s\t%s\n' \
    "$(shasum -a 256 "$file" | awk '{print $1}')" \
    "$(wc -c < "$file" | tr -d ' ')" "$source" \
    >> "$out_dir/resolved_sources.tsv"
done < "$manifest"

run_id="penzai-$(date -u +%Y%m%dT%H%M%SZ)-$$"
remote_dir="$VM_OOC/$run_id"
remote_dir_win="$VM_OOC\\$run_id"

echo "==> remote engine OOC route: $VM:$remote_dir period=${period_ns}ns"
ssh "${ssh_args[@]}" "$VM" "if not exist $VM_OOC mkdir $VM_OOC"
ssh "${ssh_args[@]}" "$VM" \
  "if not exist $remote_dir_win mkdir $remote_dir_win"
scp "${ssh_args[@]}" "$stage"/* "$VM:$remote_dir/"

set +e
ssh "${ssh_args[@]}" "$VM" \
  "cd $remote_dir && route.bat . $period_ns result ooc_sources.f" \
  > "$out_dir/driver.log" 2>&1
status=$?
set -e

scp "${ssh_args[@]}" "$VM:$remote_dir/result/*" "$out_dir/" 2>/dev/null || true
scp "${ssh_args[@]}" "$VM:$remote_dir/vivado.log" "$out_dir/" 2>/dev/null || true
for file in "$out_dir"/*.log "$out_dir"/*.rpt "$out_dir"/*.tsv; do
  [[ -f "$file" ]] || continue
  tmp="$file.lf.$$"
  LC_ALL=C tr -d '\r' < "$file" > "$tmp"
  mv "$tmp" "$file"
done

printf 'remote\t%s\nremote_dir\t%s\nperiod_ns\t%s\nexit_code\t%s\n' \
  "$VM" "$remote_dir" "$period_ns" "$status" > "$out_dir/driver_status.tsv"
if (( status != 0 )); then
  tail -80 "$out_dir/driver.log" >&2 || true
  exit "$status"
fi

cat "$out_dir/metrics.tsv"
