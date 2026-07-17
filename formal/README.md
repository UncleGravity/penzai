# Formal verification

Run the fast formal suite inside the development shell:

```sh
zig build formal
```

The same suite is exposed as the `formal-control` flake check. SymbiYosys writes
proof logs and cover traces below `.zig-cache/sby/`.

## `seq_reg_master`

`formal/rtl/seq_reg_master_formal.sv` drives arbitrary request traffic and
arbitrary AXI-Lite backpressure. It assumes the documented req/gnt contract:

- `req`, `we`, `addr`, and `wdata` remain stable until `gnt`;
- `req` is low for one cycle after `gnt`, re-arming the adapter.

ABC PDR proves the safety properties over all reachable states. The proof
checks reset behavior, AXI VALID stability, independent AW/W acceptance,
read/write exclusion, request payload preservation, response-to-grant
correspondence, one-cycle grants, and read-data capture. A separate Boolector
cover job produces traces for completed reads, completed writes, and both AW/W
stall orders.

The pilot does not prove eventual completion because AXI peers may stall
forever. A future liveness job can add an explicit bounded-response assumption.
It also does not assign semantics to AXI error responses; the current RTL
intentionally ignores `BRESP` and `RRESP`.

## `seq_core`

`formal/rtl/seq_core_formal.sv` checks the executor at the production address
width with reduced count and timeout parameters. The reduced count still covers
zero, maximum-count, rollover, and live-input mutation behavior without making
PDR enumerate the full 16-bit counter state space. It allows arbitrary
descriptor contents, arbitrary response latency, and control-plane rewrites
while a run is active. Grants are constrained only to the documented one-cycle
req/gnt protocol.

ABC PDR proves descriptor bounds and ordering, mutual exclusion of the two
request ports, request stability until grant or watchdog termination, sticky
completion/error levels, and bounded error indices. A 40-cycle Boolector BMC
provides source-level diagnostic traces, and cover traces exercise empty and
normal completion, writes, waits, poll timeout, and watchdog timeout.

The first proof found that `seq_core` observed `desc_count` live in `S_ADV`.
Rewriting `RUN_COUNT` through `seq_top` could therefore lengthen or shorten an
in-flight run. The core now snapshots the count with the accepted `go`; the
formal bound and a Verilator regression preserve that behavior.

## `seq_top`

`formal/rtl/seq_top_formal.sv` drives the complete control AXI-Lite slave and
register-master response interface. It proves control response stability,
downstream request stability, reset/abort behavior, and that every descriptor
request addresses the command segment selected when the run was accepted.
Cover traces exercise control writes during an active run, normal and error
completion, descriptor fetch, and abort.

The first bounded trace found that `seq_top` used the writable `RUN_START`
register directly for BRAM addressing. Rewriting it while busy redirected later
descriptor fetches into a different resident command segment. `seq_top` now
snapshots the start index with the accepted `go`; both the formal invariant and
an end-to-end Verilator regression preserve that behavior.

## `gemm_kernel`

`formal/rtl/gemm_kernel_formal.sv` drives arbitrary configuration changes and
AXIS backpressure around a reduced GEMM instance. ABC PDR proves that an
accepted run uses snapshotted dimensions, format, and exponent floor; internal
indices remain in range; output data is stable while stalled; and terminal
`TLAST` accounting agrees with the accepted shape. BMC provides bounded
diagnostics, while cover reaches binary and ternary control paths.

The first trace found that writable dimensions remained live after start.
`gemm_kernel` now snapshots the complete run configuration.

## `gemm_ternary_select32`

`formal/rtl/gemm_ternary_select_formal.sv` exhaustively proves the combinational
two-bit selector for all 32 lanes. Codes `0`, `1`, and `2` map to minus, zero,
and plus controls respectively; reserved code `3` has `nonzero=0` and is
therefore safely disabled. Cover points pin uniform words for all four codes.

The software packer independently rejects source code `3`; the leaf proof
establishes that every accepted upstream code maps to the same selector values
as the software oracle.
