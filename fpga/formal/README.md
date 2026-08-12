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
| `flash_kernel` | Run snapshots, ordered pipeline tags, address bounds, KV writeback barriers, output stalls, and terminal `TLAST` |

Each `.sby` file declares its proof, bounded-model, and cover tasks. Files named
`*_formal.sv` provide the harness and environment assumptions. Files named
`*_properties.vh` are included by production RTL only when `FORMAL` is defined.

Formal establishes the properties encoded by those harnesses. Arithmetic
agreement remains a cosim responsibility, isolated mapping belongs to OOC
synthesis, and timing closure belongs to the routed bitstream build.

The flash-kernel target replaces floating-point leaves with ordered fixed-latency
stubs and proves the real controller at reduced dimensions. Active configuration
inputs remain arbitrary after start, exercising the kernel's run snapshot. Input
streams are kept continuously valid to bound the state space; independent input
bubbles and numeric agreement are covered by the flash-kernel RTL cosim.
The cover task directs one processed KV, one masked KV, and one output stall; the
unbounded proof and BMC leave those choices arbitrary.

See `BUGS.md` for production and verification issues found by the suite.
