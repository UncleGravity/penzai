#!/usr/bin/env bash
set -euo pipefail

verify_dir="$(cd "$(dirname "$0")"; while [[ ! -f production_sources.sh ]]; do cd ..; done; pwd)"
repo="$(cd "$verify_dir/../.." && pwd)"
mdir="${TMPDIR:-/tmp}/penzai-engine-engine-verilator"

rm -rf "$mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --Mdir "$mdir" \
  --top-module engine_core_tb \
  "$repo/fpga/verify/sim/integration/engine/tb.sv" \
  "$repo/fpga/rtl/engine/core.v" \
  "$repo/fpga/rtl/engine/model_spec_store.v" \
  "$repo/fpga/rtl/engine/arenas.v" \
  "$repo/fpga/verify/support/leaf_stub.v"
"$mdir/Vengine_core_tb"

verilator --lint-only -Wall \
  -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PINMISSING \
  -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --top-module engine_ooc \
  "$repo/fpga/verify/qor/vivado_ooc/engine/engine_ooc.v" \
  "$repo/fpga/rtl/engine/core.v" \
  "$repo/fpga/rtl/engine/model_spec_store.v" \
  "$repo/fpga/rtl/engine/arenas.v" \
  "$repo/fpga/verify/support/leaf_stub.v"

ooc_mdir="${TMPDIR:-/tmp}/penzai-engine-engine-ooc-verilator"
rm -rf "$ooc_mdir"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-DECLFILENAME -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -Wno-PINMISSING -I"$repo/fpga/rtl/engine" -I"$repo/fpga/rtl/vector" \
  --Mdir "$ooc_mdir" \
  --top-module engine_ooc_tb \
  "$repo/fpga/verify/sim/integration/engine/ooc_tb.sv" \
  "$repo/fpga/verify/qor/vivado_ooc/engine/engine_ooc.v" \
  "$repo/fpga/rtl/engine/core.v" \
  "$repo/fpga/rtl/engine/model_spec_store.v" \
  "$repo/fpga/rtl/engine/arenas.v" \
  "$repo/fpga/verify/support/leaf_stub.v"
"$ooc_mdir/Vengine_ooc_tb"
