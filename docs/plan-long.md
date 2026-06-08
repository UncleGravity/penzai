# penzai — full design document

`penzai` is a clean-slate Zig rewrite of an FPGA LLM accelerator stack for the
AMD/Xilinx **KR260**: Kria K26 SOM, Zynq UltraScale+ MPSoC, quad Cortex-A53 PS,
FPGA PL, and 4 GB DDR4. `docs/zig.md` is the implementation contract for Zig
versioning, APIs, ownership, errors, and style.

The architecture was first derisked on a PYNQ-Z1 reference design. The shape is
the same on KR260; device specs and ceilings differ. The KR260 DDR read ceiling
is now measured in `experiments/kr260-xrt-ddr-bandwidth-multiport/`.

## 1. Goals and constraints

Build an LLM inference appliance. The first accelerated path is W1A8
(`{-1,+1}` weights, `int8` activations); W1.58A8 (`{-1,0,+1}` weights) follows.
The target model family is Bonsai-style transformers, about 1.7B params,
~1.1 bits/weight, and roughly 236 MB resident.

The binding constraint is **memory bandwidth, not compute**. A weight-streaming
matmul touches the resident model once per token, so the rough ceiling is
`usable_DDR_read_bytes_per_s / resident_model_bytes`. On KR260, four 128-bit HP
lanes at 300 MHz sustain **~12.1 GB/s aggregate read** from a DDR XRT BO into PL
(11535 MiB/s, 64.5% of the 19.2 GB/s offered bandwidth). For a ~236 MB model,
that is `12.1e9 / 236e6 ~= 51 tok/s`. The PYNQ-Z1 reference was ~2.1 GB/s, or
~9 tok/s. This ceiling assumes the PL array can consume the budget (~286
MAC/cycle); that is an FPGA-side requirement (§10), not a given. In both cases the
consequence is the same: weights live resident in board DDR after one upload;
per-token traffic is only commands, activations, profiling spans, and the sampled
token.

Two corollaries matter everywhere:

- Compute concurrency that does not reduce or hide DDR traffic will not move
  tok/s.
- Computation that leaves the board forces state over the host link. Over TCP
  that is a per-token round trip, so the cost model includes where each op runs,
  not just whether weights are resident.

The rewrite removes prototype complexity: JSON-over-TCP schemas duplicated
across C++/Python/framing, a ggml backend loaded through `dlopen` and env vars,
Python board code calling C through `ctypes`, and a large graph matcher. The
target is one language, one build, one binary contract, and transports below the
architecture.

## 2. Guiding principles

1. Organize by execution location: `shared/`, `host/`, `device/`, `fpga/`.
2. The host/device contract is `shared/protocol/wire.zig`, not a vtable.
3. Keep computation pure and effects thin; test encoders, kernels, lowering,
   and HAL encoders without hardware.
4. An abstraction needs a real second implementation, a test seam, or a contract.
5. Cross-boundary facts have one source of truth: wire schema, regmap, target
   contract, quant layout, and ggml integration contract.
6. Specialize for this board, model family, and closed device op set.
7. Start with the simplest correct mechanism. In-order execution, synchronous
   RPC, and one serialization format are defaults; overlap, async pipelines, and
   dependency graphs are earned by profiles.

## 3. The three inversions

**penzai hosts llama.** `penzai` is the binary you run. It links llama.cpp as a
library, imports build-translated llama/ggml headers through Zig wrappers,
registers the ggml backend in-process, and drives the llama C API. llama owns
GGUF, tokenization, sampling, and graph construction. We own the process, CLI,
device link, profiling, and UX.

**The network is one transport.** The architecture is the command-buffer schema.
TCP, USB, and fake mode are byte-pipes below framing and wire.

**Binary wire, device collects, host formats.** The board speaks exactly one
binary format: fixed-size records, explicit encode/decode, specified endianness.
HELLO, profiling, and trace are binary records too; no JSON parser or string
formatter belongs on the device.

## 4. File tree

```
build.zig  build.zig.zon          # penzai native, penzaid board target, tests

shared/                           # both binaries; no OS, sockets, ggml, llama
  protocol/
    transport.zig                 # byte-pipe shape: read/write exact bytes
    framing.zig                   # envelope and length validation
    wire.zig                      # schema, command buffer, explicit codecs
  q1a8.zig                        # W1A8 pack/merge layout; q158.zig later
  trace.zig                       # comptime-gated diagnostics sink
  profiling.zig                   # spans, counters, merge helpers

host/
  main.zig                        # main(init), args, --device, profiling
  run.zig                         # llama_decode loop and sampling
  llama.zig                       # wrapper over build-translated C headers
  backend.zig                     # ggml backend vtables, remote residency
  lower.zig                       # ggml node -> wire command (see §8)
  census.zig                      # graph_compute op-surface diagnostic
  link.zig                        # Link(Transport): framing, wire, counters
  transport/{fake,tcp,usb}.zig    # client byte-pipes
  prof_report.zig                 # JSONL rollup, Chrome-trace export

device/                           # KR260 side; never imports ggml/llama
  main.zig                        # penzaid entry, fabric setup, serve loop
  server.zig                      # transport, framing, decode, dispatch
  runtime.zig                     # composition root + op dispatch switch only
  transport/{tcp,usb}.zig         # server byte-pipes; usb deferred
  mem/{slab,xrt,fake}.zig         # slab interface, KR260 XRT, test memory
  ps/{matmul_q1a8,rmsnorm,rope,softmax,activations,elemwise,rows,q1a8_merge}.zig
  pl/{matmul,mmio,dma,fpga}.zig   # fabric driver and HAL

fpga/
  rtl/decode/{decode_w1,decode_w158}.v
  rtl/core/{mac_array,reducer,accumulator,output_scale}.v
  rtl/kernel_top.v
  bitstreams/<name>/{build.sh,tcl/{build.tcl,timing.xdc},out/*.bit,*.hwh}
  sim/<dut>/tb.zig                # Verilator driven from Zig
  regmap/q1a8.regmap              # source for Verilog params and mmio.zig

test/{golden,kernels,alloc,fullstack}.zig
```

Deliberate omissions: no graph scheduler module and no JSON codec. `ps/` modules
are named by responsibility (`activations`, `elemwise`, `rows`), not a catch-all
"glue."

Dependency rules: `shared/` imports only portable `std` APIs and fixed-width
protocol types; `device/` depends on wire, not ggml; ggml/llama coupling stays
in `host/{backend,lower,llama,census}.zig`; hardware effects stay in
`device/pl/*` and `device/mem/xrt.zig`.

## 5. Protocol stack

`transport.zig` defines byte pipes: write all bytes, read exactly N, close.
Concrete transports live at the application edge and receive `std.Io` from
`std.process.Init`; reusable modules do not create I/O backends.

`framing.zig` turns a byte stream into messages: magic, version, metadata length,
payload length, then bytes. It validates lengths, versions, and truncation before
wire decode.

`wire.zig` defines RPC tags, tensor handles, shapes, command buffers, profiling
fields, protocol versions, and feature bits. Treat incoming bytes as hostile:
fixed-width integers, explicit enum backing types, explicit endianness, narrow
errors, explicit encode/decode. Do not pointer-cast untrusted bytes, rely on
native layout, or leak `usize`/pointers into the wire.

Today each op carries operand handles and parameters, not dependency edges. The
host emits ggml topological order and the device executes in order; dependency
edges are deferred until §9's profile gate is met. The matmul command is
weight-format-tagged (`.w1a8` now, `.w158a8` later), so a datatype adds a packer
and kernel path, not a wire redesign.

## 6. Quant layout

`shared/q1a8.zig` owns the packed W1A8 weight layout and per-column
activation-merge contract. Host packing happens once at upload; the board PS
runs `q1a8_merge`; the FPGA decoder consumes the same bytes. Ternary adds
`shared/q158.zig` and a `WeightFormat` value. Each datatype keeps dense packing
because bandwidth, not decoder logic, is scarce.

## 7. Host: penzai

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

`llama.zig` wraps C-owned resources with Zig `init`/`deinit`, converts
nullability and integer widths at the boundary, and keeps raw C pointers out of
host logic. New C interop comes from build-system header translation.

`backend.zig` is the ggml ABI surface and owns remote tensor residency.
`lower.zig` turns ggml nodes into wire commands. `census.zig` reports the actual
op surface. `link.zig` owns framing, wire, upload/download, reconnect policy,
and byte counters over any transport. The ggml-facing code is the subtlest part
of the system; §8 is its contract.

## 8. The ggml integration contract

The host registers an in-process ggml backend. The behaviors below are forced by
ggml's scheduler and memory model, and were established in `experiments/llama/`.
Violating them usually produces silent wrong results or load-time aborts, not a
clean error.

- **Strict buffer ownership.** `supports_buft` accepts only the penzai buffer
  type. Accepting host buffers lets ggml place ops on tensors with no remote
  handle.

- **Residency forces the op surface.** Resident weights require full-layer
  offload, which places KV cache and attention tensors in penzai buffers too.
  ggml then requires this backend to run every op on those tensors. You cannot
  pick "matmul only": KV-cache ops, notably `SET_ROWS`, gate
  `llama_init_from_model`. Matmul-only offload copies weights per call and
  defeats the accelerator.

- **`supports_op` is not the lowering source of truth.** Nodes reach
  `graph_compute` that were never support-probed. Validate lowering and the op
  worklist against observed `graph_compute` nodes. `census.zig` records reached
  ops, support status, operand bindings, and byte traffic.

- **Metadata ops are descriptors, not commands.** `NONE`, `RESHAPE`, `VIEW`,
  `PERMUTE`, and `TRANSPOSE` update handle, offset, dtype, shape, and strides.
  `CONT`/`CPY` and unsupported-stride boundaries are the materialization points.

- **Matmul operands are contiguous.** The matmul command carries byte ranges and
  dimensions, not strides. Non-contiguous matmuls are rejected by the support
  predicate so ggml keeps them off device or materializes first.

- **Backend vtable objects are pinned resources.** Vtables point back into the
  object, so create it once at a stable address and pass by pointer only. Host
  pointers are ggml bookkeeping; remote handles are storage identity. Never
  encode one as the other.

## 9. Device: penzaid

The device imports zero ggml/llama. `main.zig` accepts `std.process.Init`, builds
allocators and I/O at the boundary, brings up fabric, then serves. `server.zig`
owns transport, framing, wire decode, and response encoding.

`runtime.zig` is the composition root and op dispatch switch only. It wires the
allocator, slabs, kernels, and fabric handles, then routes each wire op through
an exhaustive `switch`. Resource lifecycles belong in `mem/` and `pl/`; kernel
logic belongs in `ps/` and `pl/`; framing belongs in `server`. A runtime kernel
registry only adds indirection for a closed op set.

Execution is in-order and single-pass. The host emits a topological order, so
the device needs no graph reconstruction.

There is deliberately no graph scheduler. The earlier single-in-flight idea
(issue PL matmul, run independent PS ops, complete lazily from dependency edges)
is deferred because:

1. Decode batch 1 is mostly a serial chain; PS ops sit between matmuls, while
   independent matmuls serialize on the single PL resource.
2. In the bandwidth-bound steady state, PL weight DMA and PS activation traffic
   contend for the same saturated DDR bus, so overlapping them is unlikely to buy
   wall-clock. (While the array is still compute-bound (§10) the bus has slack,
   but reason 1 and "profile first" still defer a scheduler.)
3. Dependency edges and async completion state would enter the wire/runtime
   contract before a synchronous baseline proves they are needed.

The overlap that matters is local to the PL matmul: prefetch the next weight
tile's DMA under current tile compute so the array does not stall on DDR. Add
inter-op PL/PS overlap only if profiling shows significant, hideable PS time.

`mem/` exposes slab reads/writes/clears/alloc with XRT BOs on KR260 and a fake
bytearray for tests. v1 grabs one large contiguous BO (~768 MiB measured
reliable) at daemon start and suballocates inside it (bump for weights, arenas
for scratch). XRT fragments, so per-tensor BOs are avoided and multi-extent
tensors are unnecessary. On daemon restart host remote handles are invalid; the
host reconnects and re-uploads board-resident state.

The PL runs only bandwidth-bound weight matmul. The PS runs resident non-matmul
ops: attention, softmax, rmsnorm, rope, elementwise, row ops. Host CPU fallback
is a bring-up convenience only; over TCP it round-trips tensor state per token.
Attention therefore lives on the PS in steady state, either directly or after
`CONT`-then-contiguous-matmul.

`ps/` kernels are pure functions over slices and handles; `matmul_q1a8` is both
CPU fallback and oracle. `pl/` separates pure descriptor/register encoders from
small effect shims using mmap, ioctls, sysfs, volatile MMIO, DMA, and cache
maintenance. Physical addresses, virtual addresses, and handles stay distinct.
Avoid general allocation and string formatting in the steady-state path.

## 10. FPGA and datatypes

Datatype-specific logic is only weight decode and per-weight apply. The decoder
emits `{zero, sign}`; binary never asserts `zero`, ternary can. The MAC array,
reducer, accumulator, output scaling, AXI, DMA, and scheduling are shared.

The matmul kernel overlaps weight-tile DMA with compute by double-buffering.
This is the one overlap assumed by the architecture, and it lives inside the
kernel/RTL, not a graph scheduler.

The §1 ceiling holds only if the array consumes the DDR budget: ~12.1 GB/s of
weight stream is ~286 MAC/cycle at 300 MHz. The ported PYNQ-Z1 kernel reaches
~40.6 (compute-bound, ~6 tok/s); cosim widening reached 163.8. Widening the
datapath toward ~286, then feeding it from multiple HP weight DMAs, is the open
FPGA workstream — until then array width, not bandwidth, is the bottleneck. See
`experiments/kr260-q1a8-matmul-bringup`.

Keep packing dense per datatype. Encoding binary weights as ternary-shaped data
would double the binary weight stream. Prefer one bitstream with a `weight_fmt`
register if timing and area fit; otherwise use one bitstream per datatype.

`fpga/regmap/q1a8.regmap` generates Verilog offsets and Zig MMIO types. Adding a
datatype means: decoder RTL, `shared/q*.zig`, `WeightFormat`, simulation, regmap
mode.

## 11. Numerics

W1A8 weights are trained parameters, not a runtime approximation. Integer matmul
`sum(w*a)` is exact and order-independent for W1A8 and ternary; compare raw
`i32` accumulators with `==` against `ps/matmul_q1a8.zig`.

Scaling after accumulation is float math. RMSNorm, softmax, RoPE, SiLU/SwiGLU,
and other glue ops use documented tolerances. Activation quantization is a
separate bit-exact unit: pin scale formula and rounding mode, because one bucket
mismatch propagates exactly through matmul.

Full-model comparison to llama.cpp is a fuzzy integration check. Per-op matmul
correctness comes from the integer reference and the FPGA decoder's packed bytes.

## 12. Testing

T0 pure units: wire/framing roundtrips and malformed input, q1a8
pack/unpack/merge, fake slab allocation, scalar matmul, float kernels, HAL
encoders.

T1 in-process integration: full runtime over `mem/fake` and scalar kernels;
full host-to-fake-device path through real framing/wire. Validate lowerer and
census against nodes observed at `graph_compute`, not `supports_op` probes.

T2 golden: per-op vectors and small-model logits against llama.cpp. Generate
initial fixtures from the existing Python implementation.

T3 hardware: real-vs-reference matmul, bitstream-load smoke, MMIO checks, XRT
and bandwidth probes. Keep T3 small because T0-T2 cover most logic.

Use `ps/matmul_q1a8.zig` as the oracle for software, Verilator RTL, and packing.
Use `std.testing.io`, `std.testing.allocator`, allocation-failure checks where
useful, fuzzing for byte parsers, and timeouts for protocol/transport tests.
Run the same behavior spec against fake and real implementations.

## 13. Profiling and trace

Device profiling appends fixed-size binary spans and counters, then ships them
in RPC responses. The host merges, formats JSON Lines, renders `--prof`, and can
export Chrome trace. "No spans" is a well-formed empty response section, so
profiling gates do not fork the wire format.

Span calls are inline and comptime-gated: per op and phase, with bytes streamed
and PL counters. Correlate local clocks at the RPC boundary:
`software_overhead = round_trip - device_total`. PL compute/stall counters show
whether §10's prefetch keeps the array fed and are the evidence required before
considering §9 inter-op overlap. Always-on counters report tok/s, bytes, and
achieved GB/s; `-Dprofile` enables the per-op span tree.

`profiling.zig` owns structured timing. `trace.zig` owns human diagnostics.
Stateful diagnostics, such as byte counters and census data, live with their
owning modules.

## 14. Build system

Pin one Zig version. The repo targets Zig 0.16.x; new code uses `std.Io` and
`std.process.Init`. Verify the KR260 aarch64 Linux triple, libc, CPU features,
and minimum kernel version in `build.zig`; keep endian, alignment, and page-size
assumptions explicit.

`zig build` builds `penzai`, `penzaid`, and fast hardware-free tests.
`zig build test-rtl` runs Verilator from Zig testbenches. `test-golden`,
`test-board`, and `bench` cover numerical, hardware, and performance suites. Do
not add env-var or `dlopen` requirements for the llama backend; use build-system
C translation for llama/ggml headers.

Vivado bitstreams are out-of-band:
`fpga/bitstreams/<name>/build.sh` runs batch scripts, and `.bit`/`.hwh` artifacts
are tracked, likely with LFS. Zig and Verilator belong in the normal build;
Vivado produces artifacts the Zig side consumes. Keep `build.zig.zon`
fingerprints current; do not commit `zig-pkg/` unless deliberately vendoring.

## 15. Deployment topologies

| Command | Where | Transport | Notes |
|---|---|---|---|
| `penzaid` + `penzai run -m model.gguf --device tcp:board:port` | board + host | tcp | real KR260 |
| `penzai run -m model.gguf --device usb:VID:PID` | board + host | usb | future |
| `penzai run -m model.gguf --device fake` | host only | RAM byte-pipe | full-stack tests |

`penzaid` is the only board process. llama.cpp and ggml remain on the host. A
correct `tcp:` deployment keeps required ops resident on the board; host fallback
is for bring-up and fake mode.

## 16. Contracts

1. **Wire schema**: `shared/protocol/wire.zig` — explicit codecs, versions,
   fixed-width fields, invalid-byte tests, operand handles, no dependency edges
   yet, weight-format-tagged matmul.
2. **ggml integration** (§8): strict `supports_buft`; residency-forced op
   surface; validate at `graph_compute`; metadata-as-descriptors; contiguous
   matmul operands; pinned backend objects; pointer/handle separation.
3. **Register map**: `fpga/regmap/q1a8.regmap`, generating Verilog params and
   Zig MMIO types.
4. **Quant layout**: `shared/q1a8.zig` and later `q158.zig`, checked by host
   pack/unpack, PS reference kernels, and Verilator.
5. **Target contract**: `build.zig` records triples, libc, CPU, kernel, and
   build options.

## 17. Risks and open decisions

- **Inter-op PL/PS overlap and dependency edges**: deferred until profiling shows
  significant, hideable PS time.
- **Multi-extent allocator**: conditional on KR260/XRT fragmentation; keep the
  slab seam regardless.
- **F16 attention path**: PS kernel directly vs `CONT`-then-contiguous-matmul;
  decide during op bring-up. Host fallback is bring-up only.
- **PL array width is the current bottleneck.** On silicon the ported kernel is
  compute-bound at ~40.6 MAC/cycle (~6 tok/s) vs the ~286 MAC/cycle DDR budget
  (~51 tok/s ceiling). Widening the array toward that budget is the open FPGA
  workstream (`kr260-q1a8-matmul-bringup`); ~51 tok/s is the target the widening
  chases, not today's number.
- **One bitstream vs one per datatype**: depends on timing and area.
- **USB transport**: deferred until needed.
- **llama.cpp stays host-side** for tokenizer, sampler, GGUF, and graph support.
- **PYNQ-Z1 is aspirational, not current.** The architecture is open to it: the
  fixed-width pointer-agnostic wire survives the 32-bit ARMv7 move, `device/` is
  board-agnostic, the bandwidth thesis is parametric, the slab seam takes a
  `mem/cma.zig`, and the conditional extent allocator is exactly its ~32 MB CMA
  case. Real work is contained to gateware (a per-board bitstream: smaller 7020
  fabric, 64-bit HP ports) and a second `build.zig` target. The discipline that
  keeps it free: XRT/aarch64 assumptions stay inside `mem/`, `pl/`, and
  `build.zig`, never above the slab/transport line. Caveat: 512 MB DDR3 is tight
  for a 236 MB model. Build nothing for it now (YAGNI).

## 18. Quick reference

- **Add an op**: census real `graph_compute` nodes first; then lowerer row,
  fixed-width wire record, device kernel in the closed `switch`, tests.
- **Add a datatype**: decoder RTL, `shared/q*.zig`, `WeightFormat`, sim, regmap
  mode.
- **Add a transport**: host byte-pipe, device byte-pipe, `--device` parser case.
- **Add a bitstream**: `fpga/bitstreams/<name>/` with `build.sh` and `tcl/`;
  track `.bit`/`.hwh`.
- **Do not add yet**: graph scheduler, dependency edges, JSON codec, or extent
  allocator without profiling or real-backend evidence.
