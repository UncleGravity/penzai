# penzai — design plan

A clean-slate Zig rewrite of an FPGA LLM accelerator stack for the **KR260**
(Kria K26 SOM, Zynq UltraScale+ MPSoC, quad Cortex-A53 PS + FPGA PL, 4 GB DDR4).
One language, one build, the network demoted from "the architecture" to one
transport option. W1A8 (1-bit) now, W1.58A8 (ternary) later. First derisked on a
PYNQ-Z1 reference. `docs/plan-long.md` is the full design; `docs/zig.md` is the
implementation contract.

**Binding constraint: memory bandwidth, not compute.** Weights stream from DDR
once per token. Measured KR260 read ceiling is ~12.1 GB/s (four 128-bit HP lanes
at 300 MHz; `experiments/kr260-xrt-ddr-bandwidth-multiport/`), so a ~236 MB model
caps near ~51 tok/s. Consequence: **weights live resident in board DDR after one
upload**; per-token traffic is only commands, activations, profiling spans, and
the sampled token.

## What it is

- `penzai` — the host binary. Links llama.cpp as a **library**, registers the
  ggml backend **in-process** (no `.so`, no `dlopen`, no env vars), and drives the
  llama C API. This is the binary you run.
- `penzaid` — the board daemon on the KR260. Owns `device/`: XRT memory, MMIO,
  DMA, PS kernels, PL matmul, and profiling collection. Never runs llama/ggml.
- llama owns GGUF, tokenizer, sampler, graph-per-architecture (multi-arch is
  free). penzai owns the process, the backend, the device link, and all UX.

## File tree

```
build.zig  build.zig.zon          # penzai native, penzaid board target, tests

shared/                           # ── BOTH binaries; comptime; no OS/sockets/ggml/llama ──
  protocol/
    transport.zig                 #   byte-pipe shape (read/write exact bytes; I/O at edges)
    framing.zig                   #   message envelope + length validation
    wire.zig                      #   schema + command buffer + explicit codecs (THE contract)
  q1a8.zig                        #   1-bit weight pack/merge layout  (q158.zig later)
  trace.zig                       #   comptime-gated diagnostics sink
  profiling.zig                   #   span records + counters + merge

host/                             # ── runs wherever llama.cpp runs ──
  main.zig                        #   main(init), --device, profiling flags
  run.zig                         #   decode loop: tokenize → llama_decode → sample → emit
  llama.zig                       #   wrapper over build-translated llama/ggml C headers
  backend.zig                     #   ggml vtables, in-process; remote tensor residency
  lower.zig                       #   ggml node → wire command (see ggml integration)
  census.zig                      #   graph_compute op-surface diagnostic
  link.zig                        #   generic Link: framing+wire over any transport
  transport/{fake,tcp,usb}.zig    #   client byte-pipes (fake = host-side no-hardware device)
  prof_report.zig                 #   JSONL rollup + Chrome-trace export

device/                           # ── runs on the KR260; ZERO ggml/llama ──
  main.zig                        #   penzaid entry: bring up fabric, serve
  server.zig                      #   transport loop + framing + dispatch
  runtime.zig                     #   composition root + op dispatch switch (only)
  transport/{tcp,usb}.zig         #   server byte-pipes (usb deferred)
  mem/{slab,xrt,fake}.zig         #   slab interface + KR260 XRT (real) + bytearray (test)
  ps/{matmul_q1a8,rmsnorm,rope,softmax,activations,elemwise,rows,q1a8_merge}.zig  # ARM kernels
  pl/{matmul,mmio,dma,fpga}.zig                   #   fabric driver + HAL

fpga/                             # ── gateware: Verilog/Vivado/Verilator; NOT zig build ──
  rtl/
    decode/{decode_w1,decode_w158}.v   #   ONLY datatype-aware RTL → {zero,sign} control
    core/{mac_array,reducer,accumulator,output_scale}.v   #   datatype-neutral, shared
    kernel_top.v                       #   decoder(s) + core; AXI-Lite regs; weight_fmt mode
  bitstreams/<name>/{build.sh, tcl/{build.tcl,timing.xdc}, out/*.bit,*.hwh}
  sim/<dut>/tb.zig                     #   Verilator driven from Zig (`zig build test-rtl`)
  regmap/q1a8.regmap                   #   single source → Verilog localparams + mmio.zig

test/{golden,kernels,alloc,fullstack}.zig   # host-side, no hardware
```

Deliberate omissions: no graph scheduler module, no JSON codec. `ps/` modules are
named by responsibility, not a catch-all "glue."

## Architecture decisions

1. **Organize by execution location.** Top-level dir = the machine. `shared/`
   (both), `host/` (llama side), `device/` (board), `fpga/` (gateware).
2. **The wire schema is the host↔device contract**, not a Zig vtable. Host
   produces command buffers; device consumes them and never imports ggml/llama.
3. **One binary format, no JSON anywhere.** HELLO, profiling, and trace are binary
   records too. Fixed-width fields, explicit encode/decode, specified endianness;
   never native layout or pointer-cast parsing. Device collects, host formats.
4. **Single source of truth**: `wire.zig` (shared, comptime) and a generated
   `regmap` for the RTL↔driver register contract. No tri-lingual drift.
5. **llama in-process**, no `.so`/env/rpath dance.
6. **Simplest correct mechanism first.** In-order execution, synchronous RPC, one
   serialization format are the defaults; overlap, async, and dependency graphs
   are earned by profiles, not assumed.

## ggml integration (the hard part)

The in-process ggml backend is the subtlest subsystem. These are forced by ggml's
scheduler/memory model (established in `experiments/llama/`); violating them gives
silent wrong results or load-time aborts:

- **Strict `supports_buft`** — accept only the penzai buffer type, or ggml places
  ops on tensors with no remote handle.
- **Residency forces the op surface.** Resident weights require full-layer
  offload, which puts the KV cache + attention tensors on the device too. So KV
  ops — notably `SET_ROWS` — gate `llama_init_from_model`; you can't pick "matmul
  only" (that would copy weights per call and defeat the accelerator).
- **`supports_op` is not the truth** — validate the op worklist against nodes seen
  at `graph_compute` (`census.zig`).
- **Metadata ops are descriptors, not commands** (`NONE/RESHAPE/VIEW/PERMUTE/
  TRANSPOSE`); `CONT`/`CPY` are the materialization points.
- **Matmul operands are contiguous** — the command carries byte ranges + dims, no
  strides; non-contiguous matmuls are rejected so ggml keeps them off device.
- **The backend object is pinned** — vtables point back into it, so create it once
  at a stable address and pass by pointer only. Host pointers = bookkeeping,
  remote handles = storage; never encode one as the other.

## Transports

- One `transport.zig` byte-pipe interface; framing + wire sit on top.
- `fake` (host-side runtime for full-stack tests), `tcp` (real KR260 target),
  `usb` (deferred). Adding a transport = two byte-pipe files + a `--device` case.
- Topology is a flag: `penzai run -m model.gguf --device fake|tcp:host:port`.
  Hardware = `penzaid` on the board + `penzai --device tcp:board:port` on the host.

## Device internals (keep it simple)

- **Each op carries its operand handles; no dependency edges, no scheduler.** The
  host emits ggml's topological order and the device executes in order — in-order
  single-pass is a correct schedule. Inter-op overlap is added only on profile
  evidence.
- **Direct `switch` on op tag**, no runtime kernel registry (closed op set).
- **`runtime` = composition root + op switch, only.** `server` owns transport and
  framing; resource lifecycles live in `mem`/`pl`; kernels in `ps`/`pl`.
- **PL runs only the bandwidth-bound weight matmul; PS runs everything else
  resident** (attention, softmax, rmsnorm, rope, elementwise, rows). Host CPU
  fallback is a bring-up convenience only — over TCP it round-trips state per
  token — so attention lives on the PS in steady state.
- **Slab interface** (xrt/fake) is a justified seam. A multi-extent allocator is
  **conditional**: PYNQ-Z1 needed it (CMA fragments ~32 MB, weights ~230 MB);
  build it only if KR260/XRT proves the same need.

## FPGA / datatypes

- Datatype affects **only the weight decoder** (unpack + per-weight apply);
  accumulate → reduce → scale → fp is neutral and shared.
- Internal interface = ternary `{zero, sign}` control; binary is the degenerate
  case. Keep packing **dense per datatype** (bandwidth is the bottleneck); the
  matmul command is **weight-format-tagged** (`.w1a8`/`.w158a8`).
- The PL matmul **double-buffers weight-tile DMA under compute** so the array
  never stalls on DDR — the one overlap the architecture assumes, local to the
  kernel/RTL, not a graph scheduler.
- One bitstream + `weight_fmt` register if it fits; else one per datatype.
- Adding a datatype = one `decode_*.v` + one `shared/q*.zig` packer + a
  `WeightFormat` value + one `sim/` tb + a regmap entry.

## Numerics & testing

- **Integer matmul is bit-exact** — `==` against `ps/matmul_q1a8.zig` (the same ref
  drives the Verilator RTL test, validating software, gateware, and packing).
  Float glue ops (rmsnorm/softmax/rope/silu) use tolerance.
- Tiers: T0 pure units · T1 in-process (`--device fake`; validate the lowerer and
  census at `graph_compute`, not `supports_op`) · T2 golden (per-op vectors +
  full-model logits vs llama) · T3 hardware (minimal; differential real-vs-ref,
  bitstream smoke, XRT/bandwidth probes).
- Generate golden vectors from the existing Python impl to de-risk the port.

## Build

- `zig build` → penzai + penzaid (+ `test`, fast, no hardware). Pin one exact Zig
  version (0.16.x); design entry points around `std.process.Init` and pass
  `std.Io` to code that performs I/O.
- Verify the KR260 aarch64 triple, libc, CPU features, and minimum kernel version
  in `build.zig`; do not treat the board target as an implicit default.
- `zig build test-rtl` → Verilator over `fpga/rtl/`. `test-golden`/`test-board`/
  `bench` for the heavier tiers.
- **Vivado bitstreams are out-of-band**: `fpga/bitstreams/<name>/build.sh`; `.bit`/
  `.hwh` tracked as artifacts (LFS). No `dlopen`/env-var requirement for the llama
  backend; use build-system C translation for llama/ggml headers.

## Profiling & trace

- **Device collects fixed-size binary spans** (near-free on the A53), ships them in
  the RPC response; the host formats (JSONL + live `--prof`). "No spans" is a
  well-formed empty response section, so gating profiling off does not fork the
  wire format.
- Per-domain clocks; correlate at the RPC boundary (`overhead = round_trip −
  device_total`). PL compute/stall counters (array utilization) are first-class —
  the evidence for whether the PL prefetch keeps the array fed and whether any
  inter-op overlap is worth adding.
- Mechanism lives in `shared/`; content is inline scoped calls; both comptime-gated.

## Don't add yet

A graph scheduler, dependency edges on the wire, a JSON codec, or an extent
allocator — until profiling or the real memory backend demands them.
