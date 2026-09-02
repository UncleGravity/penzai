#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
# shellcheck source=production_sources.sh
source "$repo/fpga/verify/production_sources.sh"
penzai_load_production_rtl "$repo"

manifest_paths="$(mktemp)"
referenced_paths="$(mktemp)"
trap 'rm -f "$manifest_paths" "$referenced_paths"' EXIT

{
  for path in "${PENZAI_RTL_SOURCES[@]}" "${PENZAI_RTL_HEADERS[@]}"; do
    printf '%s\n' "${path#"$repo/"}"
  done
} | LC_ALL=C sort -u > "$manifest_paths"

find "$repo/fpga/verify" -type f \
  \( -name '*.sh' -o -name '*.tcl' -o -name '*.f' -o -name '*.sby' \) \
  -print0 |
  xargs -0 grep -Eho 'fpga/rtl/[A-Za-z0-9_./-]+\.(v|sv|vh)' |
  LC_ALL=C sort -u > "$referenced_paths"

unexpected="$(comm -13 "$manifest_paths" "$referenced_paths")"
[[ -z "$unexpected" ]] || {
  echo "ERROR: current-engine verification references RTL outside the production manifest:" >&2
  echo "$unexpected" >&2
  exit 1
}

for script in \
  fpga/verify/lint/run.sh \
  fpga/verify/sim/integration/datapath/run.sh \
  fpga/verify/qor/yosys/production/run.sh; do
  grep -q 'penzai_load_production_rtl' "$repo/$script" || {
    echo "ERROR: authoritative verification script does not load the production manifest: $script" >&2
    exit 1
  }
done

[[ "${#PENZAI_RTL_SOURCES[@]}" -eq 55 ]] || {
  echo "ERROR: production source count changed: ${#PENZAI_RTL_SOURCES[@]} (expected 55)" >&2
  exit 1
}
[[ "${#PENZAI_RTL_HEADERS[@]}" -eq 4 ]] || {
  echo "ERROR: production header count changed: ${#PENZAI_RTL_HEADERS[@]} (expected 4)" >&2
  exit 1
}

printf 'PASS production RTL closure: %d modules, %d headers\n' \
  "${#PENZAI_RTL_SOURCES[@]}" "${#PENZAI_RTL_HEADERS[@]}"
