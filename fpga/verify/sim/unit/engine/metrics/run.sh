#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-engine-metrics-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-BLKSEQ \
  --top-module engine_metrics_tb --Mdir "$mdir" \
  "$repo/fpga/rtl/engine/metrics.v" \
  "$repo/fpga/verify/sim/unit/engine/metrics/tb.sv"
"$mdir/Vengine_metrics_tb"
