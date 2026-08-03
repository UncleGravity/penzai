# Bugs found by formal verification

The formal pilot and follow-on ternary implementation found three production
RTL control bugs, one format-contract bug, and five verification or physical
implementation methodology bugs. The control bugs came from using writable
inputs as live execution state after a run had already started.

## 1. `RUN_COUNT` changed the length of an active run

**Location:** `fpga/rtl/seq/seq_core.v`

`seq_core` used the live `desc_count` input in `S_ADV` to decide whether the
current descriptor was the last one. `seq_top` leaves `RUN_COUNT` writable
while the executor is busy, so software could change the length of a run after
it had started.

The formal counterexample started a one-entry run and then changed
`desc_count` to two. The core advanced to descriptor 1 and executed an entry
that was not part of the accepted run. Rewriting the count downward could also
end a run early.

**Fix:** `seq_core` now captures `desc_count` into `run_count` when it accepts
`go`. `S_ADV` compares against this snapshot for the lifetime of the run.

**Regression coverage:**

- `fpga/formal/seq_core_formal.sv` permits arbitrary changes to `desc_count`
  while busy and proves descriptor bounds against the accepted count.
- Scenario 6 in `fpga/sim/seq_core/tb.zig` changes the count from two to one
  during execution and verifies that both original writes still occur.

## 2. `RUN_START` redirected descriptor fetches during an active run

**Location:** `fpga/rtl/seq/seq_top.v`

`seq_top` calculated the command BRAM address from the live `run_start`
register plus `desc_idx`. Because the control register remains writable while
busy, a `RUN_START` write could redirect later descriptor fetches into another
resident command segment.

The bounded `seq_top` proof reached the failure in 10 steps through the real
control AXI-Lite interface: it accepted a run, rewrote `RUN_START` while busy,
and observed `rd_idx` no longer match the segment selected by the accepted
`go`.

**Fix:** `seq_top` now captures `run_start` into `active_start` when it accepts
`go`. Command BRAM addressing uses `active_start + desc_idx` until the run is
reset or aborted. Software can therefore prepare the next `RUN_START` value
without changing the active run.

**Regression coverage:**

- `fpga/formal/seq_top_properties.vh` proves that every descriptor request uses
  the start index captured for the active run.
- The `seq_top` cover task demonstrates that rewriting `RUN_START` while busy
  is reachable through the control AXI-Lite interface.
- Scenario 4 in `fpga/sim/seq_top/tb.zig` rewrites `RUN_START` during a stalled
  `WAIT` and verifies that the next descriptor still comes from the original
  segment.

## Property-harness corrections

Two initial `seq_top` failures were corrections to the formal model, not RTL
bugs:

- The control read slave asserts `ARREADY` and `RVALID` together when it
  captures a read; the first harness assertion incorrectly required
  `RVALID` to be low in that cycle.
- `ABORT` intentionally resets the downstream AXI master mid-transaction. AXI
  stability properties now exclude that documented reset boundary while
  separately checking the resulting reset state.

## 3. GEMM dimensions changed an active run

**Location:** `fpga/rtl/gemm_kernel.v`

`gemm_kernel` used the live `num_q1_blocks`, `num_rowblocks`, and `num_cols`
inputs throughout its load, issue, precompute, and emit states. The AXI-Lite
registers driving those inputs remain writable while the kernel is busy.

The first GEMM bounded trace changed `NUM_COLS` after an accepted start. By
step 3, both the last-column decision and result-buffer length reflected the
new value rather than the accepted run.

**Fix:** the kernel now snapshots all run dimensions and `emin` with an
accepted start. Loop termination, addressing, output sizing, and arithmetic
use those snapshots until completion.

**Regression coverage:** `fpga/formal/gemm_kernel_formal.sv` permits arbitrary
live input mutation and proves the captured configuration, address bounds,
beat bounds, output backpressure stability, and final `TLAST` accounting.

## 4. The proposed group-128 ternary layout discarded a real group-64 scale

**Location:** `shared/layout.zig`, `fpga/rtl/gemm_kernel.v`

The initial upstream Q2_0 packer tried to combine two 64-weight source groups
into one 128-weight resident group with a single scale. That is lossless only
when both f16 scales match. Synthetic tests used matching pairs, but the merged
upstream format gives each group its own scale.

Inspection of the actual `_g64.gguf` found 178 unequal pairs among 13,436,752
pairs. Every mismatch was small, but rejecting them made the model unloadable
and choosing either scale would have changed model weights.

**Fix:** the resident format retains both f16 scales and the upstream two-bit
codes, reordered to match the GEMM issue sequence. Four rows use exactly nine
128-bit AXIS beats per port: one dual-scale beat followed by two code beats for
each 32-weight sub-block. The RTL selects the low scale for sub-blocks 0-1 and
the high scale for sub-blocks 2-3 without buffering or decoding a full block.

**Regression coverage:**

- Layout tests preserve deliberately unequal half scales and still reject the
  unused Q2_0 code `3`.
- PS matmul and `get_rows` tests exercise both scales.
- Kernel and `decode_top` cosim use unequal scales and remain bit-exact.
- The actual upstream group-64 model loads through the penzai backend.

## 5. The OOC summary misreported LUT and DSP utilization

**Location:** `fpga/ooc/ooc_synth.tcl`

The first OOC script filtered `PRIMITIVE_GROUP == LUT`, which is not how Vivado
2025.2 classifies LUT primitives. Replacing that with broad `REF_NAME` patterns
still counted synthesis macros: the summary printed 39,803 LUTs and 288 DSPs
while Vivado's utilization table reported 37,673 CLB LUTs and 32 DSPs.

**Fix:** capture `report_utilization -return_string`, write that exact text to
the detailed report, and extract the one-line summary from its authoritative
`CLB LUTs`, `DSPs`, `CARRY8`, and `Register as Flip Flop` rows.

## 6. The OOC kernel probe optimized the ternary front end away

**Location:** `fpga/ooc/gemm_kernel_ooc.v`

The full-kernel probe tied `weight_fmt` to binary. Vivado therefore removed the
ternary receiver and selector, so its area and timing numbers could not measure
the dual-format kernel even though the production build contained that logic.

**Fix:** expose `weight_fmt` as an unconstrained OOC input. The kernel snapshots
it at start just as it does in production, retaining both format paths during
synthesis. Full routed implementation remains the sign-off timing gate.

## 7. Expanded ternary controls spent avoidable registers and LUTs

**Location:** `fpga/rtl/gemm_kernel.v`, `fpga/rtl/gemm_ternary_select.v`

The first issue-ordered implementation captured the first two-bit code beat,
then expanded both beats into registered `{sign, nonzero}` controls. It also
gated the sign bit with the zero mask even though the GEMM already ignores sign
when `nonzero=0`. This was functionally correct, but retained 512 unnecessary
registers and roughly one avoidable selector LUT per weight lane. The first
clean combined f300 route missed timing at `-0.074 ns` on a 99%-occupied device.

**Fix:** retain both code beats in their raw two-bit form and select them only
when issuing the current sub-block. `sign` is the code high bit and `nonzero` is
the inverse low bit; reserved code `3` is harmless because its nonzero mask is
clear. OOC synthesis dropped by 535 LUTs and 510 FFs with unchanged `+0.619 ns`
WNS. The next independent clean f300 route passed with zero failing endpoints,
although its reported WNS rounded to `0.000 ns`, so routing margin remains a
project-level constraint rather than a solved capacity problem.

**Regression coverage:** selector formal proof, kernel control proof, bit-exact
GEMM and `decode_top` ternary cosims, corrected dual-format OOC synthesis, and a
clean timing-gated combined bitstream build.

## 8. Ternary runtime verification appeared to hang in the PS oracle

**Location:** `device/ps/matmul_q1a8.zig`, `shared/layout.zig`

`PENZAI_PL_VERIFY=1` recomputes every PL result with the PS reference kernel.
The first ternary oracle decoded each individual weight through
`packedTernaryWeight`, which repeated full resident-buffer length validation
and layout address calculation 32 times per Q8 inner product. A full Bonsai
graph therefore stopped producing visible progress after prefill and looked
like a device or RTL deadlock, even though non-verifying inference was healthy.

**Fix:** add `packedTernaryWeightCodes`, the ternary equivalent of the existing
binary `packedWeightBits` accessor. It reads and validates all 32 two-bit codes
as one `u64`; the oracle then decodes that local value in its inner loop.

**Regression coverage:** the Q2 resident-layout test checks every code returned
by both the scalar and bulk accessors, and the ternary oracle test exercises
minus, zero, and plus selectors through the bulk path.

## 9. Runtime flash verification mislabeled intentional approximation error

**Location:** `device/runtime.zig`

The runtime verifier compared the full-f32 PS attention implementation against
the PL result, whose committed datapath uses BF16 for `p * V` and LUT-based exp
and reciprocal. It normalized error by `max(abs(reference), 1e-6)`, so expected
milliscale differences near zero produced large relative errors and hundreds of
misleading mismatch lines during an otherwise healthy binary model run. The
near-zero denominator also made `max_rel` unsuitable for assessing the large
absolute differences seen at larger activation magnitudes.

**Fix:** describe runtime verification as an approximation smoke test and use
the same robust normalization as the flash cosim:
`abs(error) / max(abs(reference), 1)`. Values above 2% are now reported as
approximation outliers. Bit-faithful structural correctness remains gated by
cosim against the hardware-modeled `flash_ref`, not by the full-f32 PS result.

**Regression coverage:** the full flash-kernel cosim passes all configurations
with RTL-to-hardware-model normalized error below `1.5e-5`; its measured
BF16-to-full-f32 baseline is `0.18%` to `0.60%` for the committed test cases.
