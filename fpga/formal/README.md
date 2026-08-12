# FPGA formal verification

This directory contains the SymbiYosys harnesses and properties for
`fpga/rtl/`. Formal checks complement RTL cosim; they do not replace it.

## Run

Run the complete suite from the repository root:

```sh
nix develop -c zig build formal
```

Run one target while developing a property or fixing a failure:

```sh
nix develop -c zig build formal-seq-reg-master
nix develop -c zig build formal-seq-core
nix develop -c zig build formal-seq-top
nix develop -c zig build formal-gemm-kernel
nix develop -c zig build formal-gemm-ternary-selector
nix develop -c zig build formal-flash-kernel
```

Proof logs and cover traces are written below `.zig-cache/sby/`. The aggregate
suite is also exposed as the `formal-control` flake check.

## Coverage

| Target | Contract checked |
|---|---|
| `seq_reg_master` | AXI-Lite handshake, backpressure, payload stability, and response capture |
| `seq_core` | Descriptor ordering and bounds, request stability, completion, and errors |
| `seq_top` | Control responses, command addressing, active-run snapshots, and abort behavior |
| `gemm_kernel` | Run-configuration snapshots, index bounds, AXIS stalls, and terminal `TLAST` |
| `gemm_ternary_select` | Exhaustive mapping of all two-bit ternary selector codes |
| `flash_kernel` | Run snapshots, adaptive 64-slot query/head mapping, tag and address bounds, two-stage BRAM-read alignment, sparse-mask advancement, II=1 softmax ordering, once-per-KV stream accounting, KV writeback barriers, output stalls, and terminal `TLAST` |

Each `.sby` file declares its proof, bounded-model, and cover tasks. Files named
`*_formal.sv` provide the harness and environment assumptions. Files named
`*_properties.vh` are included by production RTL only when `FORMAL` is defined.

Formal establishes the properties encoded by those harnesses. Arithmetic
agreement remains a cosim responsibility, isolated mapping belongs to OOC
synthesis, and timing closure belongs to the routed bitstream build.

The flash-kernel target replaces floating-point leaves with ordered fixed-latency
stubs and proves the real controller with a two-query tile at reduced dimensions.
Two KV positions are sufficient for the control proof: arbitrary masks exercise
every finite/all-masked transition pair, including recurrence and skip re-entry.
Active configuration inputs remain arbitrary after start, exercising the kernel's
run snapshot. Input streams are kept continuously valid and output backpressure is
bounded explicitly for liveness; independent input bubbles and numeric agreement
are covered by the flash-kernel RTL cosim. The directed cover uses one all-masked
query beside a finite causal-prefix query, an all-masked KV position, overlapping
softmax updates, the final AXPY barrier, and an output stall. The unbounded proof
and BMC leave mask choices and output stalls arbitrary within those assumptions.

Flash bounded liveness is a separate exhaustive BMC task. The harness launches one
run immediately after reset, keeps all input streams valid, and permits at most two
consecutive output-stall cycles. Under those explicit fairness assumptions every
mask assignment must finish in fewer than 256 run cycles. The liveness task checks
272 steps, beyond the reset/start prefix and the full watchdog interval, while the
PDR and the short structural BMC omit only that watchdog; the dedicated liveness
task enables it and checks the full interval against a reduced cone containing the
real controller and its progress assumptions, without treating it as an inductive
invariant. The structural assertions remain exclusively in the PDR/BMC tasks.

Seven late aggregate properties use a complementary exhaustive `completion` task.
The local tag, ordering, and barrier invariants remain in PDR; the completion task
checks AXPY commit ordering, terminal pipeline totals and readiness, and the final
V-stream count across the full 272-step interval. It enables the same watchdog and
fairness assumptions as liveness. Since the only active run completes in fewer than
256 cycles, and explicit assertions close both its reset boundary and permanent
post-run idle suffix, this bounded check is exhaustive for the supported run.

The complete controller proof retains its two-query/two-head schedule and uses the
minimum faithful 32-slot narrow-layout instance (the second query occupies slots
16 and 17). A separate bounded/unbounded slot-map harness instantiates the exact
production 64-slot kernel, nondeterministically accepts either maximum command
shape (four queries by 16 heads or two queries by 32 heads), and proves the
production `slotIndex` mapping and derived scratch addresses are in range and
injective. This covers both layouts, all 64 physical slots, and the wide-mode
two-query limit without putting the full 64-slot state arrays in the controller
PDR cone. A separate exact-production SMT induction task proves the wide Q/K/V/acc
and scalar register delays while preserving symbolic memories as arrays; the
controller PDR proves the matching valid and mode latency.

See `BUGS.md` for production and verification issues found by the suite.
