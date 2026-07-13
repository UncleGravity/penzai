# Bugs found by formal verification

The formal pilot found two production RTL bugs in the sequencer. Both came
from using writable control inputs as live execution state after a run had
already started.

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

- `formal/rtl/seq_core_formal.sv` permits arbitrary changes to `desc_count`
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

- `formal/rtl/seq_top_properties.vh` proves that every descriptor request uses
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
