# penzai — design plan

A clean-slate rewrite of the PYNQ-Z1 LLM accelerator stack in Zig. One language,
one build, the network demoted from "the architecture" to one transport option.
Targets W1A8 (1-bit) now, W1.58A8 (ternary) later.

## What it is

- `penzai` — the host binary. Links llama.cpp as a **library**, registers the
  ggml backend **in-process** (no `.so`, no `dlopen`, no env vars), and drives the
  llama C API for the model frontend. This is the binary you run.
- `penzaid` — the board runtime daemon. Runs on the PYNQ-Z1 and owns `device/`:
  CMA, MMIO, DMA, PS kernels, PL kernels, scheduling, and profiling collection.
- llama owns: GGUF parse, tokenizer, sampler, graph-per-architecture (→ multi-arch
  support is free). penzai owns: the process, the backend, the device link, all UX.
  The board never runs llama or ggml.

## File tree

```
build.zig  build.zig.zon          # two artifacts: penzai (native), penzaid (arm cross-compile)

shared/                           # ── compiled into BOTH binaries; comptime; no OS/ggml ──
  protocol/
    transport.zig                 #   byte-pipe interface (writeAll/readExact)
    framing.zig                   #   message delimiting (envelope)
    wire.zig                      #   message schema + command buffer (THE contract; grows)
  q1a8.zig                        #   1-bit weight pack/merge layout  (q158.zig later)
  trace.zig                       #   comptime-gated diagnostics (custom std.log sink)
  profiling.zig                   #   Span record + ring buffer + clock + merge

host/                             # ── runs wherever llama.cpp runs ──
  main.zig                        #   `penzai run|prof`: args, --device, profiling flags
  run.zig                         #   decode loop: tokenize → llama_decode → sample → emit
  llama.zig                       #   thin @cImport("llama.h") wrapper
  backend.zig                     #   ggml vtables; registered in-process
  lower.zig                       #   ggml node → wire command (table-driven; op registry)
  link.zig                        #   generic Link: framing+wire over any transport
  transport/{fake,tcp,usb}.zig    #   client byte-pipes (fake = host-side no-hardware device)
  prof_report.zig                 #   `penzai prof`: JSONL rollup + Chrome-trace export

device/                           # ── runs on the Zynq board; ZERO ggml/llama ──
  main.zig                        #   penzaid entry: bring up fabric, serve
  server.zig                      #   transport loop + framing + dispatch (generic)
  runtime.zig                     #   composition root: owns allocator+kernels+fabric; routes
  schedule.zig                    #   single-in-flight async pipeline (follows explicit deps)
  transport/{tcp,usb}.zig         #   server byte-pipes
  mem/{slab,cma,fake}.zig         #   slab interface + CMA (real) + bytearray (test)
  ps/{rmsnorm,rope,softmax,glue,q1a8_merge}.zig   #   ARM kernels (pure fns)
  pl/{matmul,matmul_ref,mmio,dma,fpga}.zig        #   fabric driver + scalar ref + HAL

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

## Architecture decisions

1. **Organize by execution location.** Top-level dir = the machine. `shared/` (both),
   `host/` (llama side), `device/` (board), `fpga/` (gateware). The path tells you
   where the code runs.
2. **The wire schema is the host↔device contract**, not a Zig vtable. Host produces
   command buffers; device consumes them. `device/` never imports ggml/llama, so it
   is frontend-agnostic.
3. **Binary wire format**, not JSON. The device collects fixed-size records; the host
   formats. JSON only for HELLO/trace.
4. **Single source of truth schema** in `wire.zig` (comptime, shared) and a generated
   `regmap` for the RTL↔driver register contract. No tri-lingual drift.
5. **llama in-process**, no `.so`/env/rpath dance.

## Transports

- One `transport.zig` byte-pipe interface; framing + wire sit on top, transport-agnostic.
- Impls: `fake` (host-side fake device runtime for full-stack tests), `tcp`, `usb`
  (libusb host / FunctionFS gadget device). Adding a real board transport = two files
  + a `--device` case; nothing above the transport line changes. USB is noop for now.
  TCP is the real hardware target.
- Topology is a flag: `penzai run --device fake|tcp:host:port|usb:VID:PID`.
  Hardware deployment = `penzaid` on the board plus `penzai --device tcp:board:port`
  on the host.

## Device internals (keep it simple)

- **Explicit dependencies in the command buffer.** Host owns the graph, so each op
  declares its input/output handles. `schedule.zig` *follows* deps — no on-device
  inference, no reliance on ggml arena invariants.
- **Direct `switch` on op tag**, no kernel registry (closed op set).
- **Slab interface** (cma/fake) and **matmul/matmul_ref** are the justified seams:
  both exist to run the whole runtime on a laptop and to give a bit-exact oracle.
- **server/runtime split exists for tests and deployment** — `server` owns transport
  and framing; `runtime` is directly testable with fake memory/reference kernels.
  Keep `runtime` thin (compose + route).
- Allocator does multi-slab extents (CMA fragments ~32 MB; weights ~230 MB).

## FPGA / datatypes

- Datatype affects **only the weight decoder** (unpack + per-weight apply). Everything
  downstream (accumulate → reduce → scale → fp) is neutral and shared.
- Internal interface = ternary `{zero, sign}` control; binary is the degenerate case.
- Keep packing **dense per datatype** (bandwidth is the bottleneck); unify the
  backend, not the wire format.
- One bitstream + `weight_fmt` mode register if it fits; the factoring makes a
  per-datatype bitstream a one-line fallback. Datatype is per-model, never hot-path.
- Adding a datatype = one `decode_*.v` + one `shared/q*.zig` packer + a `WeightFormat`
  param + one `sim/` tb + a regmap entry. Nothing else moves.

## Numerics & testing

- **Integer matmul is bit-exact** — test `==` against `matmul_ref.zig` (and the same
  ref drives the Verilator RTL test, so it validates software, gateware, *and* the
  packing contract). Float glue ops (rmsnorm/softmax/rope/silu) use tolerance.
- Test tiers by what they need: T0 pure units (colocated) · T1 in-process integration
  (`--device fake`, no hardware) · T2 golden (per-op vectors + full-model logits vs
  llama) · T3 hardware (differential real-vs-ref, bitstream smoke, bandwidth). T3 is
  minimal because T0–T2 cover all logic.
- Generate golden vectors from the existing Python impl to de-risk the port.

## Build

- `zig build` → penzai + penzaid (+ `test`, fast, no hardware).
- `zig build test-rtl` → Verilator (free/fast/local) over `fpga/rtl/`, driven by Zig.
- `zig build test-golden` / `test-board` / `bench` for the heavier tiers.
- **Vivado bitstreams are out-of-band**: `fpga/bitstreams/<name>/build.sh`
  (`vivado -mode batch`), `.bit`/`.hwh` tracked as artifacts (LFS). Free/fast tools
  go in `zig build`; proprietary/slow tools produce tracked artifacts.

## Profiling & trace

- **Device collects fixed-size binary spans** (near-free on the A9), ships them in the
  RPC response; the host formats/aggregates (JSONL + live `--prof` table).
- Clocks stamped per-domain; correlate at the RPC boundary
  (`overhead = round_trip − device_total`). PL hardware counters (compute vs stall
  cycles → array utilization) are first-class.
- Mechanism lives in `shared/`; content is inline scoped calls; both comptime-gated.
```
