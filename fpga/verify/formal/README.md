# Formal verification

These SymbiYosys suites target RTL admitted by the production source manifest.
Formal checks complement simulation and routed implementation; they do not replace
numeric reference tests or timing sign-off.

## Commands

Run the default, development-time proof set:

```sh
nix develop -c zig build formal
```

Run the default set plus deeper induction, liveness, and cover tasks:

```sh
nix develop -c zig build formal-all
```

Generated proof artifacts belong under `.zig-cache/fpga-verify/formal/` and must
not be written beside these sources.

## Current suites

- `attention/flash_kernel` checks controller accounting, ordering, barriers,
  framing, bounded completion, and slot mapping with arithmetic leaves abstracted.
- `math/ternary_select` exhaustively checks every two-bit weight selector code.
- `math/rms_inverse` checks inverse-RMS control, framing, failures, abort, and
  restart with latency-faithful arithmetic stubs.
- `math/swiglu` checks stream credits, backpressure, output stability, abort,
  and restart with latency-faithful arithmetic stubs.
- `engine/metrics` checks recorder lifecycle, exact controller/stage accounting,
  saturation, overflow reporting, and frozen snapshot behavior.

The Flash controller harness uses a reduced command shape, and its slot-map
harness covers the four-token/sixteen-head and two-token/thirty-two-head layouts.
It does not yet elaborate the production eight-token/eight-head layout or its
arena-native query ordering. That production-configuration harness is required
before the attention proof can be considered complete for the deployed engine.

## Next proofs

Priorities are the AXI readers and writers, shared-reader arbitration, command
lifecycle, clear/restart behavior, ownership, and the projection queues.
