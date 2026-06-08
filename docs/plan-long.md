# penzai — full design document

`penzai` is a clean-slate Zig rewrite of the PYNQ-Z1 LLM accelerator stack.
This document describes the architecture. `docs/zig.md` is the implementation
contract for Zig versioning, APIs, ownership, errors, and style.

## 1. Goals and constraints

Build an LLM inference appliance for the PYNQ-Z1: Zynq-7020, dual Cortex-A9 PS,
FPGA PL, 512 MB DDR3. The first accelerated path is W1A8
(`{-1,+1}` weights, `int8` activations); W1.58A8 (`{-1,0,+1}` weights) follows.
The target model family is Bonsai-style transformers, about 1.7B params,
1.1 bits/weight, and roughly 236 MB resident.

The hard limit is memory bandwidth. Usable DDR bandwidth is about 2.1 GB/s, so
a 236 MB model has a rough ceiling of `2.1e9 / 236e6 ~= 9 tok/s`, assuming the
weight stream is touched once per token and overlapped with compute. Therefore
weights live resident in board DDR after one upload; per-token traffic is only a
command buffer, activations, profiling spans, and the sampled token.

The rewrite removes accidental complexity from the prototype: JSON-over-TCP
schemas duplicated across C++/Python/framing code, a ggml backend loaded through
`dlopen` and env vars, Python board code calling C through `ctypes`, and a large
graph matcher. The target shape is one language, one build, a binary contract,
and a transport boundary that is not the architecture.

## 2. Guiding principles

1. Organize by execution location: `shared/`, `host/`, `device/`, `fpga/`.
2. The host/device contract is `shared/protocol/wire.zig`, not a vtable.
3. Keep computation pure and effects thin: encoders, kernels, and schedulers are
   testable without hardware; syscalls and MMIO sit at the edge.
4. Abstractions need a real second implementation, a test seam, or a contract.
5. Cross-boundary facts have one source of truth: wire schema, register map,
   target contract, and quant layout.
6. Specialize for this board, this model family, and a closed device op set.

## 3. The three inversions

**penzai hosts llama.** `penzai` is the binary you run. It links llama.cpp as a
library, imports the build-translated llama/ggml headers through Zig wrappers,
registers the ggml backend in-process, and drives the llama C API. llama still
owns GGUF, tokenization, sampling, and per-architecture graph construction. We
own the process, CLI, device link, profiling, and UX.

**The network is one transport.** The architecture is the command-buffer schema.
TCP, USB, and fake mode are byte-pipe implementations below framing and wire.
Nothing above the transport line changes when the link changes.

**Binary wire, device collects, host formats.** The A9 handles fixed-size binary
records and explicit encode/decode. JSON is reserved for optional human-facing
handshake or trace metadata; profiling is collected as binary spans and rendered
on the host.

## 4. File tree

```
build.zig  build.zig.zon          # penzai native, penzaid board target, tests

shared/                           # both binaries; no OS, sockets, ggml, llama
  protocol/
    transport.zig                 # byte-pipe shape: read/write exact bytes
    framing.zig                   # message envelope and length validation
    wire.zig                      # schema, command buffer, explicit codecs
  q1a8.zig                        # W1A8 pack/merge layout; q158.zig later
  trace.zig                       # comptime-gated diagnostics sink
  profiling.zig                   # fixed span records, counters, merge helpers

host/                             # workstation side
  main.zig                        # main(init), args, --device, profiling flags
  run.zig                         # llama_decode loop and sampling
  llama.zig                       # safe wrapper over build-translated C headers
  backend.zig                     # ggml backend vtables, registered in-process
  lower.zig                       # ggml node -> wire command metadata table
  link.zig                        # Link(Transport): framing + wire + counters
  transport/{fake,tcp,usb}.zig    # client byte-pipes
  prof_report.zig                 # JSONL rollup and Chrome-trace export

device/                           # PYNQ-Z1 side; never imports ggml/llama
  main.zig                        # penzaid entry, fabric setup, serve loop
  server.zig                      # transport loop, framing, decode, dispatch
  runtime.zig                     # composition root and op routing
  schedule.zig                    # single-in-flight overlap from explicit deps
  transport/{tcp,usb}.zig         # server byte-pipes
  mem/{slab,cma,fake}.zig         # extent allocator, real CMA, test memory
  ps/{matmul_q1a8,rmsnorm,rope,softmax,glue,q1a8_merge}.zig
  pl/{matmul,mmio,dma,fpga}.zig   # fabric driver and HAL

fpga/                             # gateware, sims, and bitstream artifacts
  rtl/decode/{decode_w1,decode_w158}.v
  rtl/core/{mac_array,reducer,accumulator,output_scale}.v
  rtl/kernel_top.v
  bitstreams/<name>/{build.sh,tcl/{build.tcl,timing.xdc},out/*.bit,*.hwh}
  sim/<dut>/tb.zig                # Verilator driven from Zig
  regmap/q1a8.regmap              # source for Verilog params and mmio.zig

test/{golden,kernels,alloc,fullstack}.zig
```

Dependency rules are strict: `shared/` imports only portable `std` APIs and uses
fixed-width values for protocols; `device/` depends on wire, not ggml details;
ggml/llama coupling stays in `host/{backend,lower,llama}.zig`; hardware effects
stay in `device/pl/*` and `device/mem/cma.zig`.

## 5. Protocol stack

`transport.zig` defines the byte-pipe shape: write all bytes, read exactly N
bytes, close. Concrete transports live at the application edge and receive
`std.Io` from `std.process.Init`; reusable modules do not create I/O backends.

`framing.zig` turns a byte stream into messages: magic, version, metadata length,
payload length, then bytes. It validates lengths, versions, and truncation before
wire decode.

`wire.zig` defines RPC tags, tensor handles, shapes, command buffers, profiling
fields, protocol versions, and feature bits. Treat incoming bytes as hostile:
fixed-width integers only, explicit enum backing types, explicit endianness,
named narrow error sets, and explicit encode/decode helpers. Do not pointer-cast
untrusted bytes, rely on native layout, or leak `usize`/pointers into the wire.

Command buffers carry explicit input/output handles and dependencies. The device
scheduler follows those declarations instead of reconstructing graph shape from
ggml arenas or tensor byte ranges.

## 6. Quant layout

`shared/q1a8.zig` owns the packed W1A8 weight layout and the per-column
activation-merge contract. Host packing happens once at model upload; the board
PS runs `q1a8_merge` on the hot path; the FPGA decoder consumes the same bytes.
When ternary lands, `shared/q158.zig` sits beside it. Each datatype keeps dense
packing because bandwidth, not decoder logic, is the scarce resource.

## 7. Host: `penzai`

Typical flow:

```
main.zig:  pub fn main(init: std.process.Init) !void
           parse args and --device from init
           link = Link.open(io, parseDevice(args))
           registerBackend(link)

run.zig:   model = llama.loadModel(path)
           // backend uploads weights to the device once
           ctx = llama.newContext(model)
           while (!done) {
               llama.decode(ctx, batch)   // ggml -> backend -> wire -> device
               tok = sampler.sample(ctx)
               emit(llama.tokenToPiece(tok))
               profiling.tick()
           }
```

`llama.zig` wraps C-owned resources with Zig `init`/`deinit`, converts pointer
nullability and integer widths at the boundary, and prevents raw C pointers from
spreading through host logic. New C interop comes from build-system header
translation, not new `@cImport` blocks.

`backend.zig` is the ggml ABI surface. `lower.zig` maps supported ggml node
patterns to wire commands through a small metadata table and owns unsupported-op
diagnostics. `link.zig` is generic over the transport and owns framing, wire,
upload/download, reconnect policy, and byte counters. `prof_report.zig` formats
saved JSONL and Chrome-trace output.

## 8. Transports

`fake` runs the same runtime on the host with fake memory and scalar kernels,
through the real framing/wire path. Use it for full-stack tests and development
without a board.

`tcp` is the real hardware target: `penzai` connects from the host and `penzaid`
listens on the board.

`usb` is deferred. When real, it is libusb bulk transfers on the host and a
FunctionFS gadget on the device. Until then it is a stub, not speculative logic.

Topology is a flag:
`penzai run -m model.gguf --device fake|tcp:host:port|usb:VID:PID`. Because
weights stay resident and per-token traffic is tiny, steady-state decode
throughput is gated by DDR and PL/PS overlap, not the board transport.

## 9. Device: `penzaid`

The device imports zero ggml/llama. `main.zig` accepts `std.process.Init`, builds
allocators and I/O at the boundary, brings up fabric, then serves. `server.zig`
owns transport, framing, wire decode, and response encoding. `runtime.zig` owns
the allocator, slabs, kernels, fabric handles, and dispatch routing. Keep it
thin; logic belongs in `schedule`, `mem`, `ps`, and `pl`.

Dispatch is an exhaustive `switch` on the wire op tag. The device op set is
closed, so a runtime kernel registry only adds indirection.

`schedule.zig` implements the single-in-flight pipeline. A PL matmul can issue
DMA and return pending; independent PS ops continue; the pending op completes
when a dependency needs its output or another fabric op needs the PL. Explicit
command-buffer dependencies make this mechanical.

`mem/` exposes slab reads/writes/clears and an extent allocator. `cma` uses
udmabuf/CMA on the board; `fake` is a bytearray for tests. Extents are required
because CMA may fragment around 32 MB while weights can be around 230 MB. Keep
extent complexity inside memory and DMA code.

`ps/` kernels are pure functions over slices and handles. `matmul_q1a8` is both
CPU fallback and bit-exact oracle. `pl/` separates pure descriptor/register
encoders from the small effect shims that use mmap, ioctls, sysfs, volatile MMIO,
DMA, and cache maintenance. Physical addresses, virtual addresses, and handles
must stay distinct.

Avoid general allocation and string formatting in the steady-state device path.
Use preallocated buffers, arenas for bounded lifetimes, narrow public errors,
and `error.Canceled` propagation where I/O can cancel.

## 10. FPGA and datatypes

Datatype-specific logic is only weight decode and per-weight apply. The decoder
emits `{zero, sign}` control; binary never asserts `zero`; ternary can. The MAC
array, reducer, accumulator, output scaling, AXI, DMA, and scheduling are shared.

Keep packing dense per datatype. Encoding binary weights as ternary-shaped data
would double the binary weight stream and directly reduce tok/s; decoders are
cheap compared with DDR bandwidth.

Prefer one bitstream with a `weight_fmt` register if timing and area fit. If the
Zynq-7020 is tight, use one bitstream per datatype; the decoder/core split makes
that a parameter choice, not a rewrite.

`fpga/regmap/q1a8.regmap` generates both Verilog register offsets and Zig MMIO
types. This prevents driver/RTL drift. Adding a datatype means: one decoder RTL
file, one `shared/q*.zig` packer, one `WeightFormat` value, one simulation, and
one regmap mode entry.

## 11. Numerics

W1A8 weights are trained parameters, not a lossy runtime approximation. The
integer matmul `sum(w*a)` is exact and order-independent for W1A8 and ternary;
compare raw `i32` accumulators with `==` against `ps/matmul_q1a8.zig`.

Scaling after accumulation is float math. RMSNorm, softmax, RoPE, SiLU/SwiGLU,
and other glue ops use documented tolerances because libm implementations and
reduction orders differ. Activation quantization is a separate bit-exact unit:
pin scale formula and rounding mode, because one bucket mismatch propagates
exactly through matmul.

Full-model comparison to llama.cpp is a fuzzy integration check, not a precision
gate. Per-op matmul correctness comes from the integer reference and the same
packed bytes used by the FPGA decoder.

## 12. Testing

T0 pure units: colocated `test` blocks for wire/framing roundtrips and malformed
input, q1a8 pack/unpack/merge, fake slab allocation, scalar matmul, float kernels
with tolerances, and HAL encoders.

T1 in-process integration: full device runtime over `mem/fake` and scalar
kernels; full host-to-fake-device path through real framing and wire.

T2 golden: per-op vectors and small-model logits against llama.cpp. Generate the
initial fixtures from the existing Python implementation to de-risk the port.

T3 hardware: real-vs-reference matmul, bitstream-load smoke, MMIO checks, CMA and
bandwidth probes. Keep T3 small because T0-T2 cover most logic.

Use one oracle many ways: `ps/matmul_q1a8.zig` validates software, drives the
Verilator RTL testbench, and pins the packing contract. Use `std.testing.io` for
I/O tests, `std.testing.allocator` for allocation tests, allocation-failure
checks where useful, fuzzing for byte parsers, and timeouts for protocol or
transport tests that can hang.

Where an interface has fake and real implementations, run the same behavior spec
against both: fake in normal CI, real hardware in T3.

## 13. Profiling and trace

Device profiling appends fixed-size binary span records and counters, then ships
them in RPC responses. The host merges, formats, writes JSON Lines, renders the
rolling `--prof` table, and can export Chrome trace. The A9 does not format
strings or JSON on the hot path.

Span calls are inline and comptime-gated, coarse enough to matter: per op and
phase, with bytes streamed and PL counters. Use local clock domains and correlate
at the RPC boundary: `software_overhead = round_trip - device_total`. PL
compute/stall counters give array utilization and identify DMA starvation.
Always-on counters report tok/s, bytes, and achieved GB/s; `-Dprofile` enables
the heavier per-op span tree.

`profiling.zig` owns structured timing records and merge helpers. `trace.zig`
owns human diagnostics over a gated log sink. Stateful diagnostics, such as byte
counters or unsupported-op census, live with the module that owns the state.

## 14. Build system

Pin one exact Zig version and keep `build.zig`, CI, and local tooling on it. If
the repo targets Zig 0.16.x, design new code around `std.Io` and
`std.process.Init`.

Supported targets are an explicit contract. Verify the PYNQ-Z1 Linux triple,
libc, CPU features, and minimum kernel version in `build.zig`; do not bake in
`arm-linux-gnueabihf` until that check is done. Keep endian, alignment, and
page-size assumptions explicit.

`zig build` builds `penzai`, `penzaid`, and fast hardware-free tests. `zig build
test-rtl` runs Verilator over `fpga/rtl/` from Zig testbenches. `test-golden`,
`test-board`, and `bench` cover heavier numerical, hardware, and performance
suites.

Vivado bitstreams are out-of-band:
`fpga/bitstreams/<name>/build.sh` runs Vivado batch scripts, and `.bit`/`.hwh`
artifacts are tracked, likely with LFS. Zig and Verilator belong in the normal
build; proprietary or slow Vivado runs produce artifacts the Zig side consumes.

Build-system C translation owns llama/ggml headers. Keep package fingerprints in
`build.zig.zon` current, do not commit `zig-pkg/` unless deliberately vendoring,
and use `--fork=[path]` for temporary dependency overrides.

## 15. Deployment topologies

| Command | Where | Transport | Notes |
|---|---|---|---|
| `penzaid` + `penzai run -m model.gguf --device tcp:board:port` | board + host | tcp | real PYNQ-Z1 |
| `penzai run -m model.gguf --device usb:VID:PID` | board + host | usb | future |
| `penzai run -m model.gguf --device fake` | host only | RAM byte-pipe | full-stack tests |

`penzaid` is the only board-side process. llama.cpp and ggml remain on the host.

## 16. Contracts

1. Wire schema: `shared/protocol/wire.zig`, explicit codecs, protocol versions,
   fixed-width fields, and tests for invalid bytes.
2. Register map: `fpga/regmap/q1a8.regmap`, generating Verilog params and Zig
   MMIO types.
3. Quant layout: `shared/q1a8.zig` and later `q158.zig`, checked by host
   pack/unpack tests, PS reference kernels, and Verilator.
4. Target contract: `build.zig` records supported triples, libc, CPU, kernel, and
   build options.

## 17. Risks and open decisions

- USB is deferred; implement libusb/FunctionFS only when needed.
- One bitstream vs. one per datatype depends on real timing and area.
- `schedule.zig` can fold into `runtime.zig` if explicit deps make it tiny.
- `runtime.zig` can become a god object; keep it as composition and routing.
- llama.cpp stays host-side for tokenizer, sampler, GGUF, and graph support.
- Performance work follows profiling: remove Python/ctypes overhead, then tune
  fabric clock, HP port usage, DMA overlap, and bandwidth.

## 18. Quick reference

- Add an op: lowerer metadata row, wire tag/record/codecs, device kernel, tests.
- Add a datatype: decoder RTL, `shared/q*.zig`, `WeightFormat`, sim, regmap mode.
- Add a transport: host byte-pipe, device byte-pipe, `--device` parser case.
- Add a bitstream: `fpga/bitstreams/<name>/` with `build.sh` and `tcl/`; track
  `.bit`/`.hwh`.
