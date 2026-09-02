#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
period_ns="${1:-3.333}"
out_dir="${2:-$repo/.zig-cache/fpga-verify/vivado-ooc/engine}"
manifest="$repo/fpga/verify/qor/vivado_ooc/engine/sources.f"

command -v vivado >/dev/null 2>&1 || {
  echo "ERROR: vivado is not available on PATH" >&2
  exit 127
}

mkdir -p "$out_dir"
resolved="$out_dir/resolved_sources.tsv"
printf 'sha256\tbytes\tpath\n' > "$resolved"
driver="$repo/fpga/verify/qor/vivado_ooc/engine/route.tcl"
printf '%s\t%s\t%s\n' \
  "$(shasum -a 256 "$driver" | awk '{print $1}')" \
  "$(wc -c < "$driver" | tr -d ' ')" \
  "fpga/verify/qor/vivado_ooc/engine/route.tcl" >> "$resolved"
while IFS= read -r source; do
  case "$source" in ''|'#'*) continue ;; esac
  file="$repo/$source"
  test -f "$file"
  hash="$(shasum -a 256 "$file" | awk '{print $1}')"
  bytes="$(wc -c < "$file" | tr -d ' ')"
  printf '%s\t%s\t%s\n' "$hash" "$bytes" "$source" >> "$resolved"
done < "$manifest"

exec vivado -mode batch \
  -log "$out_dir/vivado.log" \
  -journal "$out_dir/vivado.jou" \
  -source "$driver" \
  -tclargs "$repo" "$period_ns" "$out_dir" "$manifest"
