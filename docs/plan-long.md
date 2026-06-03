# penzai — full design document

A clean-slate rewrite of the PYNQ-Z1 LLM accelerator stack in Zig. This document is
self-contained: it captures the goals, the architecture, every major decision and the
reasoning behind it, and enough concrete detail to start a new repository and build.

---

## 1. Goals and constraints

**What we are building.** An LLM inference appliance for the PYNQ-Z1 board
(Xilinx Zynq-7020: a dual-core ARM Cortex-A9 "PS" + FPGA fabric "PL", with 512 MB
DDR3). The accelerated kernel is a quantized matmul: **W1A8** today (1-bit binary
weights `{-1,+1}`, 8-bit `int8` activations), **W1.58A8** (ternary `{-1,0,+1}`) later.
The target model family is Bonsai-style transformers (~1.7B params, ~1.1 bits/weight,
≈236 MB resident).

**The dominant constraint: memory bandwidth.** Decode is weight-streaming bound. The
board's usable DDR bandwidth is ~2.1 GB/s, so the arithmetic ceiling for a 236 MB model
is roughly `2.1e9 / 236e6 ≈ 9 tok/s` — and only if every weight byte is touched exactly
once per token, overlapped with compute. Everything in this design serves that prime
directive: **weights live resident in board DDR (uploaded once), and per-token traffic is
tiny (command buffer + activations + sampled token).** This single fact is what makes
transport choice, serialization format, and most other decisions low-stakes for steady
-state throughput — they only ever carry control traffic, never the weight stream.

**Why a rewrite.** The prototype works but its center of gravity is the wire protocol:
a JSON-over-TCP schema duplicated by hand across C++ (`ops.h`), Python (`ops.py`), and a
framing module, with the host side living as a `dlopen`-ed ggml backend `.so` injected
into upstream `llama-cli` via environment variables, and the board daemon written in
Python on the PYNQ platform with a C hot path reached through `ctypes`. The result is
three languages, two build systems, an env-var/rpath dance, and a ~1100-line graph
"lowering" pattern-matcher. None of that is essential complexity. The rewrite collapses
it to one language, one build, and a network boundary that is optional and isolated.

---

## 2. Guiding principles

1. **Organize by execution location.** The top-level directory tells you which machine
   the code runs on. This is the axis a reader most needs at a glance in a distributed
   embedded system.
2. **The contract is the wire schema, not a vtable.** Host and device are separate
   binaries (often separate machines). Their interface is the binary command-buffer
   format. The device never imports ggml or llama — it is frontend-agnostic.
3. **Separate computation from effect, everywhere.** Pure functions and pure encoders are
   testable on a laptop; I/O, syscalls, and hardware pokes are thin shims at the edges.
   Applied recursively, this is what makes ~everything host-testable.
4. **Abstractions must earn their keep.** A seam is justified only if it has ≥2 real
   implementations, is a test seam that makes the layer above runnable without hardware,
   or is a contract boundary that prevents drift. "We might need it" is not a reason.
5. **Single source of truth for every cross-language contract.** The wire schema, the
   register map, and the weight packing layout each exist once and are consumed by both
   sides, with a test asserting agreement.
6. **Specialize hard.** One model family, one board, a closed op set. Resist building a
   generic accelerator framework or a mini-ggml.

---

## 3. The three inversions

The rewrite is three deliberate inversions of the prototype's assumptions.

**Inversion 1 — penzai hosts llama, not the reverse.** Today upstream `llama-cli` is the
entry point and our code is a guest `.so` it `dlopen`s. Instead, `penzai` is *our* binary;
it links llama.cpp as a **library**, registers the ggml backend with a direct in-process
function call (no `dlopen`, no `GGML_BACKEND_PATH`, no `LD_LIBRARY_PATH`/rpath juggling),
and drives the llama C API (`llama_model_load_from_file`, `llama_decode`,
`llama_sampler_*`, `llama_batch`) for the decode loop. llama still owns GGUF parsing, the
tokenizer, sampling, and graph construction per architecture — so multi-architecture
support stays free — but we own the process, which is what lets `--device`, the live
profiling readout, and trace scopes be first-class CLI flags instead of env vars smuggled
through a plugin seam. The per-layer compute path is unchanged: llama builds the ggml
graph → the ggml scheduler hands ops to our backend → we lower them to command buffers →
the device executes. We did not reimplement the forward pass.

**Inversion 2 — the network is one optional transport, not the architecture.** The
prototype is organized around feeding a TCP socket. Here the architecture is the command
-buffer schema; a transport is just *how the bytes get to the device*, and one transport
(`inproc`) moves no bytes over a wire at all. The whole comms stack is swappable below an
unchanging schema.

**Inversion 3 — binary wire format, device-collects/host-formats.** JSON encode/decode and
string formatting per `RUN_GRAPH` on a 650 MHz A9 is pure overhead. The device deals only
in fixed-size binary records; the host (which has spare cycles) does all formatting and
aggregation. JSON survives only for the HELLO handshake and human-readable trace.

---

## 4. File tree

```
build.zig  build.zig.zon          # two artifacts: penzai (native), penzaid (arm cross-compile)

shared/                           # ── compiled into BOTH binaries; comptime; no OS/ggml ──
  protocol/
    transport.zig                 #   byte-pipe interface (writeAll/readExact)
    framing.zig                   #   message delimiting (the envelope)
    wire.zig                      #   message schema + command buffer (THE contract; grows)
  q1a8.zig                        #   1-bit weight pack/merge layout  (q158.zig added later)
  trace.zig                       #   comptime-gated diagnostics (custom std.log sink)
  profiling.zig                   #   Span record + ring buffer + clock abstraction + merge

host/                             # ── runs wherever llama.cpp runs (Mac dev, or the A9) ──
  main.zig                        #   `penzai run|prof`: args, --device, profiling flags
  run.zig                         #   decode loop: tokenize → llama_decode → sample → emit
  llama.zig                       #   thin @cImport("llama.h") wrapper
  backend.zig                     #   ggml vtables (reg/device/buffer); registered in-process
  lower.zig                       #   ggml node → wire command (table-driven; the op registry)
  link.zig                        #   generic Link: framing+wire over any transport
  transport/{inproc,tcp,usb}.zig  #   client byte-pipes (inproc = RAM loopback for tests)
  prof_report.zig                 #   `penzai prof`: JSONL rollup + Chrome-trace export

device/                           # ── runs on the Zynq board; imports ZERO ggml/llama ──
  main.zig                        #   penzaid entry: bring up fabric, serve
  server.zig                      #   transport loop + framing + dispatch (generic over Transport)
  runtime.zig                     #   composition root: owns allocator+kernels+fabric; routes
  schedule.zig                    #   single-in-flight async pipeline (follows explicit deps)
  transport/{tcp,usb}.zig         #   server byte-pipes
  mem/
    slab.zig                      #     interface: write/read/clear over byte ranges
    cma.zig                       #     real: udmabuf/CMA contiguous buffers (board only)
    fake.zig                      #     test: bytearray (runs the whole device on a laptop)
  ps/{rmsnorm,rope,softmax,glue,q1a8_merge}.zig   #   ARM Cortex-A9 kernels (pure fns)
  pl/
    matmul.zig                    #     real W1A8 fabric driver (DMA descriptors + control)
    matmul_ref.zig                #     scalar reference (runs with no fabric; the oracle)
    mmio.zig dma.zig fpga.zig     #     HAL: register map, DMA engine, bitstream load

fpga/                             # ── gateware: Verilog/Vivado/Verilator; NOT zig build ──
  rtl/
    decode/{decode_w1,decode_w158}.v   #   ONLY datatype-aware RTL → {zero,sign} control
    core/{mac_array,reducer,accumulator,output_scale}.v   #   datatype-neutral, shared
    kernel_top.v                       #   decoder(s) + core; AXI-Lite regs; weight_fmt mode
  bitstreams/<name>/{build.sh, tcl/{build.tcl,timing.xdc}, out/*.bit,*.hwh}
  sim/<dut>/tb.zig                     #   Verilator driven from Zig (`zig build test-rtl`)
  regmap/q1a8.regmap                   #   single source → Verilog localparams + mmio.zig

test/{golden,kernels,alloc,fullstack}.zig   # host-side cross-module tests, no hardware
```

**The dependency rules that keep this legible:**

- `shared/` imports nothing but `std`. No OS, no ggml, no llama. That is what lets it
  compile unchanged into both targets.
- `device/` never imports ggml or llama. Its only contract is `shared/protocol/wire.zig`.
  A future non-ggml frontend would just emit the same command buffers.
- ggml/llama coupling is confined to `host/{backend,lower,llama}.zig`.
- The only hardware-bound files are `device/pl/{mmio,dma,fpga}.zig` and `device/mem/cma.zig`.
  Everything else — including the entire device runtime, via `mem/fake.zig` and
  `pl/matmul_ref.zig` — runs and is tested on a development machine.

---

## 5. The protocol stack (`shared/protocol/`)

Three layers, conceptually stacked but with **no imports between them** — they are
independent contracts composed at the edges (`host/link.zig`, `device/server.zig`).

**`transport.zig` — the byte-pipe.** A minimal interface: `writeAll([]const u8)` and
`readExact([]u8)` (loop until done, the same shape as the prototype's `send_all`/`recv_all`),
plus `close`. Content-agnostic; it moves bytes and knows nothing about messages.

**`framing.zig` — the envelope.** Turns a byte stream into discrete messages. A small
fixed header (magic, version, metadata length, payload length) followed by the metadata
bytes and the payload bytes. It deals only in opaque `[]u8`; it does not know what a
message means. This layer changes essentially never (only if you alter delimiting, add a
checksum, etc.).

**`wire.zig` — the schema.** What the bytes *mean*: the RPC envelope (op tag, request id),
the command buffer (an array of typed op records), tensor handles/offsets/shapes, and the
profiling response field. This is the contract that grows as you add ops or datatypes.
It is pure structs/enums with no dependencies. Idiomatic Zig uses `extern struct` records
that both ends `@bitCast`/`memcpy` — no parser, no JSON.

Why two files for framing and wire: different change cadence. Framing is frozen after day
one; wire grows over the project's life. Keeping them apart means the thing that never
changes and the thing that changes constantly do not share a file. The analogy is HTTP
chunked-transfer-encoding (framing) vs. the JSON body (wire): independent concerns that
meet only at "here are the bytes."

**Command buffer with explicit dependencies.** Because the host owns the graph (via
`lower.zig`), each op record declares its input and output handles explicitly. This is a
deliberate move that simplifies the device: `schedule.zig` *follows* declared dependencies
rather than reconstructing them on-device from tensor byte-intervals (which is what the
Python prototype does, leaning on "ggml's compute-arena guarantee" for safety). A few
bytes of metadata in the command buffer delete the scariest logic on the board.

---

## 6. Quant layout (`shared/q1a8.zig`)

The on-DDR packed weight format and the per-column activation-merge layout, designed for
memcpy throughput on the A9 and for the fabric's consumption pattern. Packing is done once
at model-upload time on the host; the per-column merge runs on the board PS hot path
(`device/ps/q1a8_merge.zig`). The layout is a **cross-language contract** shared by the
host packer, the device merge kernel, and the FPGA decoder — see §10. When ternary lands,
`q158.zig` is added beside it; the two formats keep distinct packings to preserve density
(see §9).

---

## 7. Host (`host/`) — the penzai binary

`penzai` is the binary you run. Flow of `penzai run model.gguf --device tcp:host:port`:

```
main.zig:  parse args, --device, profiling flags
           link = Link.open(parseDevice(args))           // selects a transport
           registerBackend(link)                          // in-process ggml backend
run.zig:   model = llama.loadModel("model.gguf")          // llama parses GGUF, any arch
           //  ↳ weights stream to the device here, once, via backend buffer set_tensor
           ctx = llama.newContext(model)
           batch = llama.tokenize(ctx, prompt)
           while (!done) {
               llama.decode(ctx, batch)   // ggml graph → scheduler → backend → device
               tok = sampler.sample(ctx)
               emit(llama.tokenToPiece(tok))
               profiling.tick()           // updates the live --prof readout
           }
```

- **`llama.zig`** — a thin wrapper over `@cImport("llama.h")`. From Zig this is a direct
  include; no bindings.
- **`backend.zig`** — the ggml backend vtables (registry/device/buffer interfaces).
  Registered in-process via a direct call, not `dlopen`. This is the only ggml-ABI surface.
- **`lower.zig`** — the op registry. Given a ggml graph node, emit the matching wire
  command. **Table-driven**: a lookup table from op pattern → emitter, exactly the right
  shape for extensibility. The supported op set is greppable in one place. Adding an op
  (or an architecture whose ops are already supported) touches this table and nothing else
  on the host. This file also owns the "unsupported op census" diagnostic — it is lowering
  state, not trace infrastructure.
- **`link.zig`** — the host's protocol-aware handle to the device. It composes
  framing + wire over a chosen `Transport` and exposes `call(cmdbuf) → response`, plus
  `upload`/`download`, plus reconnect-on-drop. It is **fully generic over the transport** —
  no transport special-casing lives here. It also owns the cumulative upload/download byte
  counters (I/O state belongs to the I/O layer).
- **`transport/`** — the client byte-pipes. `tcp` (socket connect), `usb` (libusb;
  deferred, see §8), `inproc` (a RAM loopback used for tests).
- **`prof_report.zig`** — `penzai prof`: roll a JSONL profiling run into a per-op table,
  and export Chrome-trace for a visual timeline.

---

## 8. Transports

**One interface, impls below it.** `transport.zig` (the byte-pipe) is the interface;
`framing.zig` and `wire.zig` sit on top and are transport-agnostic by construction. The
implementations differ per side because a client connects/opens and a server
listens/accepts: client transports live in `host/transport/`, server transports in
`device/transport/`. The generic compositions are `host/link.zig` and `device/server.zig`.

**The three transports:**

- **`inproc` — RAM loopback, for tests.** A thread-safe in-memory pipe; the device
  `server.zig` loop runs on a background thread reading/writing it. It is a *real*
  transport (it moves bytes through memory), so `link.zig` and `server.zig` stay completely
  generic and inproc requires no special-casing anywhere. The payoff: tests exercise the
  genuine `framing` + `wire` + dispatch code path in one process with no hardware. (A
  single-threaded loopback would deadlock — client `readExact` waits for a response the
  server loop has not produced — so the background thread is intrinsic, and it faithfully
  models the concurrent client↔server interaction.)
- **`tcp` — the real target.** Socket client (host) and listen/accept server (device). This
  is the working host↔board link for development and for board deployments where penzai
  runs off-board.
- **`usb` — deferred / no-op for now.** Stubbed so the seam exists, but not implemented.
  When it lands it is libusb bulk transfers on the host (`@cImport` libusb) and a USB
  FunctionFS gadget on the device (read/write an fd, like a socket). Adding it is two files
  plus a `--device` case; nothing above the transport line changes. This is the one seam
  whose justification is future, not present — built the day USB is real, not on spec.

**Topology is a flag, not an architecture.**
`penzai run --device inproc | tcp:host:port | usb:VID:PID`. On-board production uses
`inproc`: penzai links the device runtime directly and there is no daemon and no socket.
Because weights are resident and per-token traffic is tiny, the choice of transport never
gates decode throughput — it only ever carries control traffic.

---

## 9. Device (`device/`) — the penzaid runtime

The device runtime is plain Zig with the hardware quarantined behind two twinned
interfaces. Data flow:

```
server.zig  read frame → framing.decode → wire.decode → runtime.dispatch(cmdbuf)
runtime.zig route by op tag (a switch) → schedule or direct kernel call
schedule.zig follow explicit deps; overlap PL DMA with PS compute (single in-flight)
ps/*.zig     pure kernels over mem handles      pl/matmul.zig  fabric matmul
            → wire.encode(result + profiling spans) → framing → server writes
```

**`server.zig` vs `runtime.zig` — split for a load-bearing reason.** `server` is the
transport loop + framing + dispatch; `runtime` owns the allocator, kernels, and fabric
handle and executes commands. They are separate because the **inproc/co-located path
bypasses the serve loop** — penzai calls `runtime.dispatch` directly — so `runtime` must be
usable without `server`. Keep `runtime` thin: compose the pieces and route dispatch; the
real logic lives in `schedule`/`mem`/`ps`/`pl`.

**`schedule.zig` — the single-in-flight async pipeline.** The one piece of hard-won
performance design. A matmul that issues its DMA returns a *pending* handle; subsequent ops
that do not touch the in-flight result run on the ARM while the PL streams, and the pending
op is completed the moment an op needs its output (or another matmul needs the fabric).
This overlap of weight streaming with glue compute is mandatory for a bandwidth-bound
design. Because dependencies are now **explicit in the command buffer** (§5), this module
follows declared deps instead of inferring them from byte-intervals — the dangerous part of
the prototype is gone. If it ends up small enough, it folds into `runtime.zig`.

**Dispatch is a `switch`, not a registry.** The op set is closed and known, so a `switch`
on the wire op tag that calls the kernel function directly beats a dynamic kernel registry
(which is indirection with no second impl, no test seam, no contract — it fails the rubric).

**`mem/` — slab + extent allocator.** A `slab` is a contiguous board-resident byte range
with `write`/`read`/`clear`. Two impls: `cma` (real PYNQ/udmabuf contiguous buffer) and
`fake` (a bytearray). The fake impl is **load-bearing**: it is what lets the entire device
runtime run and be tested on a laptop. The allocator above the slabs handles **multi-slab
extents**: CMA fragments at runtime (the largest contiguous buffer is ~32 MB) while a weight
buffer can be ~230 MB, so a large tensor spans multiple extents. PS kernels walk extents
transparently via `read`/`write`; PL DMA asks for the extent list and programs one
descriptor per extent. The extent complexity is forced by a hard hardware constraint, not
gold-plating — but it is the one place extents are visible, and it should stay that way.

**`ps/` — ARM kernels.** `rmsnorm`, `rope`, `softmax`, `glue` (add/mul/scale/silu/swiglu),
and `q1a8_merge` (per-column activation merge into packed weights). Pure functions over
slices — trivially unit-testable.

**`pl/` — fabric driver + reference + HAL.**
- `matmul.zig` — the real W1A8 driver: build DMA descriptors, kick the fabric, read back.
- `matmul_ref.zig` — a scalar reference doing the same integer accumulation. Two roles:
  it lets the runtime run with no fabric (laptop tests), and it is the bit-exact oracle
  for the real path and for the RTL. Same function signature as `matmul.zig`, selected at
  comptime/runtime — no vtable.
- `mmio.zig`/`dma.zig`/`fpga.zig` — the HAL. Each is split into a **pure encoder** (build
  the descriptor list, encode the register bits, parse/validate the bitstream header) and a
  **thin effect shim** (mmap + write, ioctl, write to `fpga_manager` sysfs). The pure half
  is host-testable against a mock register page; only the final poke needs the board. Most
  HAL bugs are wrong-offset/wrong-bit, caught on the laptop.

---

## 10. FPGA / gateware (`fpga/`) and multi-datatype

**The whole design in one sentence:** the datatype affects only the front of the datapath
(weight unpack + per-weight apply); everything downstream is neutral and shared.

| Stage | 1-bit `{-1,+1}` | 1.58-bit `{-1,0,+1}` | Shared? |
|---|---|---|---|
| weight unpack | 1 bit → sign | trits (5/byte) → {zero,sign} | differs |
| per-weight apply | `±a` | `±a or 0` | differs (superset) |
| accumulate (int32) | Σ | Σ | identical |
| reduce / scale / int→fp32 | — | — | identical |
| activations (int8), AXI, DMA, streaming | — | — | identical |

**The one seam.** A per-datatype **decoder** emits, per weight, a 2-bit sign-control
`{zero, sign}` (a `{-1,0,+1}` selector). The neutral MAC array does
`acc += zero ? 0 : (sign ? -a : a)`. Binary is the degenerate case — its decoder never
asserts `zero`. The array, reducer, accumulator, and output-scale stage speak sign-control
and **never grow** when a datatype is added. So the internal datapath is ternary-shaped, and
each datatype's decoder maps its packed bytes onto that control.

**Keep packing dense per datatype.** Do not unify the wire format. You are
bandwidth-bound; forcing binary weights into a ternary packing (1 → 2 bits/weight) would
double the binary model's weight stream and directly cost tok/s. Decoders are cheap fabric;
bandwidth is not. Unify the backend, not the format.

**Bitstream strategy.** Prefer one bitstream with a `weight_fmt` mode register selecting the
decoder, if both fit (Zynq-7020 is small, but decoders are tiny next to the array). Datatype
is fixed per model and reconfig is hundreds of ms, so it is never on the hot path — setting a
mode register at load is enough. If place-and-route is tight, the same factoring makes a
per-datatype bitstream a one-line parameter on `kernel_top`. Do not pre-commit; let timing
and area decide.

**Tree.** `rtl/decode/` holds the only datatype-aware modules; `rtl/core/` holds the neutral
backend; `kernel_top.v` wires the chosen decoder(s) to the core and exposes the AXI-Lite
control/register port plus the weight/acts/result AXIS streams. Per-bitstream Vivado projects
live in `bitstreams/<name>/` with their own `build.sh` (kept per-folder; a root launcher can
link to them later). Verilator testbenches live in `sim/<dut>/` (see §12). The block design
mirrors the working prototype: PS7 + two AXI DMAs (one for the weight + result streams, one
for the activation stream) sharing an HP port, driving `kernel_top`.

**The register-map contract.** The RTL defines registers at offsets; `device/pl/mmio.zig`
must agree exactly, and `device/pl/dma.zig` must match the stream topology. This is a
cross-toolchain contract — the gateware analogue of `wire.zig`. Make it a **single source of
truth**: `fpga/regmap/q1a8.regmap` generates both the Verilog `localparam` offsets and the
Zig register struct, with a test asserting they match. Most fabric-bringup bugs are
driver↔RTL offset mismatches; a generated regmap kills the whole class.

**Adding a datatype** = one `fpga/rtl/decode/decode_<fmt>.v` + one `shared/q<fmt>.zig` packer
+ extend the `WeightFormat` param in `matmul.zig`/`matmul_ref.zig` + one `fpga/sim/<fmt>/`
testbench + a regmap mode value. The core array, reducer, accumulator, output stage, DMA,
AXI, and host runtime do not move.

---

## 11. Numerics

A W1A8 model is **not** a lossy approximation of an F32 original — it was trained
quantization-aware, so the binary weights *are* the parameters. Divergence between two
implementations does not come from quantization; it comes from arithmetic domain:

- **The matmul core — `Σ(w·a)`, `w ∈ {-1,+1}`, `a ∈ int8` — is exact and order-independent.**
  Integer accumulation is lossless (an int32 accumulator trivially holds `127 × ~2048`), so
  it does not matter whether `matmul_ref` sums left-to-right and the fabric sums in a tree:
  identical integers out. This is the strongest invariant in the system — test it `==`,
  never with tolerance. The same holds for ternary.
- **Scales applied after** (`β_w · scale_a · accumulator`) are float multiplies — deterministic
  but with normal float rounding. (The per-group scales are also where the ~1.1 bits/weight
  comes from: 1 bit of weight + a little scale overhead.)
- **Float glue ops** (rmsnorm's `rsqrt`, softmax's `exp`, rope's `sin`/`cos`, silu) are the
  only genuinely not-bit-exact part across different implementations, because of differing
  transcendental implementations and reduction orders. Use tolerance here.
- **Activation quantization (F32 → int8)** is deterministic *if* you pin the scale formula and
  rounding mode to the reference — but a single rounding mismatch flips an int8 bucket and
  then propagates exactly through the integer matmul. Test it as its own bit-exact unit; it
  is the dangerous boundary.

**Practical kernel-design tip:** expose the raw int32 accumulator before scaling, so the
matmul's exact part can be tested `==` and the float scaling tested separately within ε. If a
kernel only exposes the final scaled float, the bit-exact guarantee is thrown away.

**On comparing to llama.cpp:** bit-exact against llama is impossible for the float ops, and
possibly for the matmul too (if its kernel dequantizes to float and runs an F32 GEMM instead
of integer accumulation). So the per-op matmul oracle is *your own integer reference*
(`matmul_ref.zig` / numpy), compared `==`; the full-model logits-vs-llama check is a fuzzy
integration smoke (top-k agreement or logits within ε), not a precision gate.

---

## 12. Testing

The architecture did the hard part: hardware is quarantined behind thin twinned interfaces
(`mem/fake`↔`cma`, `pl/matmul_ref`↔`matmul`, `inproc`↔`tcp`), so the entire software stack
runs on a laptop. The test pyramid maps onto distance-from-hardware — the same axis as the
file tree.

**Tiers by what they need to run:**

- **T0 — pure units** (colocated `test {}` blocks; milliseconds; every save): kernels,
  `matmul_ref`, wire/framing roundtrip, allocator over fake slabs, q1a8 layout, the HAL
  encoders.
- **T1 — in-process integration** (sub-second; no hardware): the full device runtime over
  `fake` + `matmul_ref`; the full stack `host → inproc → device`. Exercises the real
  framing/wire/schedule code via the loopback transport.
- **T2 — golden / numerical** (needs a small model fixture; no hardware): per-op golden
  vectors (fast, *localizing* — they tell you which kernel broke) plus full-model logits vs
  upstream llama.cpp (integration/lowering coverage).
- **T3 — hardware-in-the-loop** (board only; minimal): differential real-vs-ref matmul
  (bit-exact `==` on the integer path), bitstream-load smoke, a CMA/bandwidth probe, MMIO
  against the real fabric. T3 is minimal *because* T0–T2 already proved all the logic.

**Two patterns do most of the work:**

- **One spec, many impls (differential).** Write a behavior spec once, parameterize over
  impls at comptime: the transport spec over `inproc`/`tcp`; the device spec over
  `fake`/`cma` and `matmul_ref`/`matmul`. Cheap impls run always; hardware impls run the same
  assertions in T3. The reference implementation *is* the executable spec.
- **One oracle for software, gateware, and the wire layout.** `matmul_ref.zig` is the
  bit-exact integer reference. It checks the Zig software kernel, it drives the Verilator RTL
  testbench, and — because it decodes the same packed bytes the RTL decoder does — it
  validates the host↔RTL packing contract for free.

**RTL simulation: Verilator, driven from Zig.** Keep Verilator (free, fast, license-free,
CI-friendly) — not commercial sims, not pure-SV/UVM testbenches. The testbenches are Zig
(`fpga/sim/<dut>/tb.zig`): a `zig build test-rtl` step verilates the DUT, compiles the
generated `Vtop` together with the Zig testbench, and runs it, checking against
`matmul_ref.zig`. This unifies language and build and gives the single-oracle property
directly. (Migrating from the prototype's cocotb tests can be incremental; in the meantime
feed cocotb and the Zig tests the same golden fixtures so they share one reference.)

**De-risk the port:** generate the golden vectors from the existing Python implementation.
The old `board/kernels/*.py` and `q1a8_layout.py` already encode correct behavior; running
them once to emit fixtures gives every Zig kernel a pass/fail gate on day one.

**Zig testing mechanics:** `std.testing.allocator` everywhere (fails on leaks);
`checkAllAllocationFailures` to fuzz the OOM paths (the link and runtime must degrade
gracefully under the CMA-fragmentation reality); `expectApproxEqAbs` for float kernels with
honest tolerances; built-in fuzzing (`zig build test --fuzz`) pointed at the byte-parsing
surfaces (`framing.zig`, the `server.zig` command-buffer decode) where the bar is robustness
— malformed/truncated/hostile input must return a clean error, never crash or read OOB.

**Pitfalls:** do not make quantized golden tests bit-exact against llama (only the integer
path against your own reference is `==`); do not mock the protocol in integration tests (route
through real wire/framing via inproc); do not let any logic test reach for hardware — if it
does, the hardware boundary leaked.

---

## 13. Profiling and trace

Profiling is the primary instrument for the tok/s goal, not a debugging luxury. Every
question that matters — at the bandwidth ceiling or wasting it? is the overhead transport,
scheduling, or marshalling? is the systolic array computing or starving? — is a profiling
question, so the design is shaped by them.

**The core idea: device collects, host formats.** The device appends fixed-size binary
**span** records during execution (a counter read and a store — near-free on the A9), ships
them in the RPC response payload, and the host (with spare cycles) deserializes, merges,
formats, and aggregates. Co-located (`inproc`), the "response" is just the returned buffer —
same path, no serialization. The A9 never formats a string on the hot path.

**Call-site primitive:** a `defer`-based span, comptime-gated, with counters:

```zig
var s = profiling.span(.matmul);  defer s.end();
s.bytes(weight_nbytes);           // host derives achieved GB/s
s.pl(cycles, stalls);             // PL hardware counters
```

Without `-Dprofile`, spans compile to nothing. Keep spans coarse (per-op, per-phase), never
per-instruction.

**Clock domains — stamp locally, correlate at the boundary.** Each side stamps its own
monotonic clock (and labels the domain); the PL exposes a free-running cycle counter read via
MMIO. Do not try to globally sync clocks. The device's total returns inside the response, so
the host computes `software_overhead = round_trip − device_total` — exactly the overhead
decomposition the project cares about — with no clock sync, only one honest subtraction at the
one boundary both endpoints witnessed.

**PL hardware counters are first-class.** For a systolic, bandwidth-bound design the single
most valuable number is **array utilization = compute_cycles / total_cycles**, plus
stall-waiting-on-DMA cycles, read straight from fabric counter registers. If the array is
starving, that is the bottleneck; if it is saturated and still slow, it is genuinely
bandwidth (clock/HP work). These ride along as span counters.

**Two tiers.** Tier 0 (always on, ~free): a few atomic counters → tok/s, bytes streamed,
achieved GB/s. Cheap enough for production; it is the headline KPI. Tier 1 (`-Dprofile`): the
full per-op span tree with phase breakdown and PL cycle counts, for investigation.

**Output.** The host writes JSON Lines (one self-contained record per token; append-only,
streamable). `penzai run --prof` shows a rolling per-op table (op, µs, %, GB/s, PL util,
tok/s) for live iteration; `penzai prof run.jsonl` rolls up a saved run; a small converter
emits Chrome-trace (`chrome://tracing`) for a visual timeline that shows whether DMA-out of
one op overlaps the matmul of the next.

**Where it lives (and the trace counterpart).** Mechanism is in `shared/` (`profiling.zig`
holds the `Span` record, ring buffer, clock abstraction, merge; `trace.zig` is a comptime gate
over a custom `std.log` sink). Content is distributed: spans/trace lines are inline scoped
calls at the sites that own the data (`schedule`, `pl/matmul`, `link`). Stateful diagnostics
(byte counters, unsupported-op census) live with their owners, not in the trace file. Trace is
human diagnostics; profiling is structured timing; they share the sink but not the content.

---

## 14. Build system

- **`zig build`** → `penzai` (native) + `penzaid` (`-Dtarget=arm-linux-gnueabihf`), plus
  `zig build test` (T0+T1; fast; no hardware) as the inner loop.
- **`zig build test-rtl`** → Verilator over `fpga/rtl/`, driven by the Zig testbenches.
  Verilator is free/fast/local, so it belongs *in* the build.
- **`zig build test-golden` / `test-board` / `bench`** → the heavier tiers (T2 / T3 /
  performance thresholds via `profiling.zig`).
- **Vivado bitstreams are out-of-band.** `fpga/bitstreams/<name>/build.sh` runs
  `vivado -mode batch -source tcl/build.tcl` (kept per-folder; a root launcher can link to it
  later). The `.bit`/`.hwh` outputs are **tracked artifacts** (git-LFS): rebuilds are slow and
  remote, and the board needs them to run. `out/vivado/` scratch and `sim_build/` are
  gitignored.

The rule: free/fast/local tools (Zig, Verilator) integrate into `zig build`;
proprietary/slow tools (Vivado) produce tracked artifacts that the Zig side consumes.

Cross-compilation to the board is `zig build -Dtarget=arm-linux-gnueabihf` — no separate
toolchain, no sysroot juggling, and `zig cc` compiles any retained C in the same invocation.

---

## 15. Deployment topologies

One codebase, one schema, topology chosen by a flag:

| Command | Where | Transport | Notes |
|---|---|---|---|
| `penzai run model.gguf --device inproc` | on the board | none (direct call) | production: no daemon, no socket |
| `penzaid` + `penzai run --device tcp:board:port` | board + dev machine | tcp | development against real hardware |
| `penzai run --device inproc` (with `mem/fake`+`matmul_ref`) | laptop | loopback | full-stack test, no hardware |

`penzaid` is the same `device/` code wrapped in `server.zig`; `inproc` links it directly.

---

## 16. Contracts (single sources of truth)

Three cross-boundary contracts, each defined once with a test asserting agreement:

1. **Wire schema** — `shared/protocol/wire.zig`, comptime-shared by host and device. Adding
   an op edits this and nothing else in the protocol.
2. **Register map** — `fpga/regmap/q1a8.regmap`, generating Verilog `localparam` offsets and
   the Zig `mmio.zig` struct. Kills driver↔RTL offset drift.
3. **Weight packing layout** — `shared/q1a8.zig` (and `q158.zig`), matched by the FPGA
   decoder and `matmul_ref`. The Verilator-driven test and the host pack/unpack roundtrip
   together pin it.

---

## 17. Risks and open decisions

- **USB transport** is the one abstraction justified only by a future need. It is stubbed,
  not built; implement it the day it is real (libusb host + FunctionFS gadget). TCP is the
  working target; inproc covers tests.
- **One bitstream vs. two** for the two datatypes depends on Zynq-7020 place-and-route fit.
  The RTL factoring (decoder seam + neutral core) keeps both cheap, so the decision can wait
  for real timing/area numbers.
- **`schedule.zig` size.** With explicit dependencies it may be small enough to fold into
  `runtime.zig`; keep it separate only if the overlap logic stays meaty.
- **`runtime.zig` god-object risk.** It is a composition root by design; keep logic in
  `schedule`/`mem`/`ps`/`pl` and let `runtime` only compose and route, or it rots.
- **Tokenizer/sampler** stay in llama.cpp. If a future need arises to drop llama, the device
  is already frontend-agnostic — only the host frontend would change — but that is explicitly
  out of scope; multi-architecture support is the reason llama stays.
- **Performance path** (from the bandwidth analysis): closing the gap to ~5–8 tok/s needs the
  native daemon (this rewrite removes the Python/ctypes overhead), then fabric clock and a
  second HP port for DDR bandwidth. Profiling (§13) is the instrument that tells you which of
  these is binding at any moment.

---

## 18. Quick reference — "how do I add X?"

- **An op:** add the pattern→emitter row in `host/lower.zig`, the op tag + record in
  `wire.zig`, and the kernel in `device/ps/` or `device/pl/`. Three edits, one per location.
- **A datatype:** `fpga/rtl/decode/decode_<fmt>.v` + `shared/q<fmt>.zig` + a `WeightFormat`
  param in `matmul.zig`/`matmul_ref.zig` + a `fpga/sim/<fmt>/` testbench + a regmap mode value.
- **A transport:** `host/transport/<name>.zig` + `device/transport/<name>.zig` + a `--device`
  case. Nothing above the transport line changes.
- **A bitstream:** a folder under `fpga/bitstreams/<name>/` with `build.sh` + `tcl/`; build
  out-of-band; track the `.bit`/`.hwh`.
```
