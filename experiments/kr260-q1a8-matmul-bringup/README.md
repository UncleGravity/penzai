# kr260-q1a8-matmul-bringup

Derisk the one untouched hardware risk: an actual **Q1A8 matmul on the KR260
fabric**, bit-exact vs reference, fed at DDR bandwidth. Every other piece (XRT
BO, AXI DMA, MMIO, bitstream build/load, HP ports) is already proven by the
bandwidth/loopback experiments and is reused as the skeleton — the only new
variable here is the matmul kernel itself.

Self-contained: dedicated `flake.nix` (zig 0.16 + verilator + dtc), no coupling
to the main tree. The ported RTL in `fpga/rtl/q1a8/` is the old PYNQ-Z1
kernel **verbatim** (known-good DUT); the clean decode/core refactor from the
plan is a later step, not part of derisking.

## Milestones (each gates the next)

- **M0 — reference oracle** ✅ `src/matmul_ref.zig`. Scalar Q1A8. Exposes raw
  int32 sub-sums (the `==` gate) separately from the f32-scaled output (the ε
  gate), per plan §11.
- **M1 — packing contract** ✅ `src/pack.zig` + `src/q1a8.zig`. Q8_0 act
  quantization, AXIS weight/act layouts, result unpack. Roundtrip-tested.
- **M2 — Verilator sim vs ref** ✅ `fpga/sim/q1a8_kernel/{tb.zig,shim.cpp}`,
  `zig build test-rtl`. Zig drives the verilated `q1a8_kernel` (C ABI shim over
  `--lib-create` static lib), feeds M1-packed bytes, checks results vs
  `matmul_ref` within ε. Passing shapes: 8×128, 8×512, 24×256, and the
  Bonsai-representative **64×2048** — all `max_rel=0.0000`. Retires the datapath
  + packing-contract logic risk before any Vivado build.
- **M3 — bitstream** 🔧 scaffolded, runs on the Vivado VM + board. `fpga/build.tcl`
  forks the multiport block design down to: `q1a8_kernel_top` fed by DMA0/HP0
  (weights MM2S + results S2MM) and DMA1/HP1 (acts MM2S), 64-bit datapath, clk_wiz
  at `w64-f<MHz>` (default **100**). **Timing finding:** w64-f200 failed route with
  WNS −2.267 ns → unpipelined fp32 reducer fmax ≈ **137 MHz** (matches the old Z1's
  ~143 MHz). Build at 100 MHz for correctness; pushing the clock is M5/M6 work
  (pipeline the reducer, or restore the old multicycle-accumulator constraint).
  `kernel_top` lints clean. Smoke: the runner reads ID(`0xB05A2000`)/VERSION first.
  **Data-path finding:** first board run computed garbage (control plane fine,
  cycle counts exact, but results showed `[data][0][data][0]` and weights were
  read corrupted). Root cause: the AXI DMA forced a **128-bit** memory/stream
  width for the 40-bit-addr/burst-16 config, mismatching the 64-bit kernel — it
  zero-padded writes and skipped every other 8 bytes on reads. Fix: run the DMAs
  at 128-bit (proven) and bridge to the kernel with `axis_dwidth_converter`
  (128↔64), with `assert_config` guarding the width.
- **M4 — hardware differential** ✅ **Q1A8 matmul runs correctly on the KR260
  fabric.** All 4 shapes (incl. Bonsai-rep 64×2048) match `matmul_ref` within ε,
  cycle counts and MAC/cycle identical to the cosim (40.6 on 64×2048). Raw result
  bytes confirm the numerics: got vs expected differ only in the low byte (RTL
  round-toward-zero vs reference round-to-nearest) — plan §11's "integer exact,
  fp scale within ε" validated on silicon. `src/board.zig` + `src/{config,mmio}.zig`,
  reusing `xrt.zig`/`dma.zig`. Run: `zig build all -Dvariant=w64-f100`.
- **M5 — throughput/util** 🔜 (on hardware). The compute-bound finding is already
  hardware-confirmed (40.6 MAC/cycle = cosim). Remaining: real wall-clock GB/s, a
  STALL_CYCLES counter, and the clock ceiling (push 100 → ~137 MHz).
- **M6 — widen the array** 🔧 cosim prototype done (`q1a8_kernel_wide.v`,
  `zig build test-rtl-wide`). Widened the weight stream to ROWS×32 bits (256) so a
  full subblock for all rows loads in **one beat**, issued at 1 subblock/cycle into
  the *unchanged* `q1a8_rowblock` core. Result: **40.6 → 163.8 MAC/cycle (4.0×)**
  at realistic M (256×2048), all bit-exact-within-ε; WISSUE is now 63% of cycles
  (was 15%). Next levers: kill WSCALE overhead (15% — pack 2 q1blocks' scales per
  256-bit beat or pipeline) → ~190; then the hard ceiling is **DDR delivery** — the
  cosim has no bandwidth model, so reaching ~286 needs multi-HP weight DMAs feeding
  the 256-bit stream (the real hardware step). See the budget below.

## Sim finding (from M2 cosim + multiport DDR result)

The four-port DDR fixture (`kr260-xrt-ddr-bandwidth-multiport`) delivers
**12.1 GB/s** read = ~40 B/cycle at 300 MHz. The weight stream is 144 B per
(rowblock × Q1-block): 128 B of 1-bit weights + 16 B of fp16 per-row scales
(~11% overhead). So the bandwidth-bound budget the array must consume is:

```
12.1 GB/s × 128/144 / 300 MHz ≈ 286 weight-bits/cycle = ~286 MACs/cycle
```

Activations (int8 + fp16 scales) and result writes are <~1% of decode DDR
traffic — batch-1 decode reads the K-element act vector once and reuses it across
all M rows — so weights set the budget.

The cosim feeds with no stalls, so its `busy_cycles` is the kernel's intrinsic
(compute-bound) throughput. The ported 8-lane kernel tops out at:

```
rows=64 blocks=16 (Bonsai-rep): 40.6 MAC/cycle   vs ~286 budget  -> ~7x too slow
```

So on KR260 **array width, not DDR bandwidth, is the bottleneck**: ~6 tok/s
compute-bound vs ~51 tok/s bandwidth-bound for a 236 MB model. M6 = widen the
datapath (more lanes / parallel rowblocks / wider issue) toward ~286 MACs/cycle,
iterating in cosim against both correctness and `MAC/cycle`.

## Exit criteria

A Bonsai-representative matmul runs on the KR260 PL matching `matmul_ref`
(integer path `==`), **and** the widened array reaches a `bits/cycle` close to
the ~322 DDR budget (bandwidth-bound, not compute-starved).

## Layout

```
src/q1a8.zig        layout constants (single source)
src/matmul_ref.zig  M0 oracle
src/pack.zig        M1 wire packer
src/main.zig        laptop self-test (zig build run-selftest)
src/board.zig       M4 board runner (aarch64)
src/{config,mmio}.zig   board addresses + kernel AXI-Lite driver
src/{xrt,dma}.zig   reused from the bandwidth experiments
fpga/rtl/q1a8/      ported RTL (verbatim, the DUT)
fpga/regmap/        AXI-Lite register contract
fpga/build.tcl      M3 Vivado block design (kernel + 2 DMAs)
fpga/sim/q1a8_kernel/  M2 cosim: tb.zig (driver) + shim.cpp (C ABI over DUT)
overlay/            device-tree overlay for the XRT app
```

## Run

```
# Laptop (no board):
zig build test          # M0 + M1 unit tests
zig build run-selftest  # pack roundtrip + reference
nix develop -c zig build test-rtl   # M2 Verilator cosim
zig build board         # cross-compile the aarch64 board binary

# Hardware (M3/M4): set up config.env first (cp config.env.example config.env)
zig build all -Dvariant=w64-f100    # bitstream (Vivado VM) -> deploy -> run on board
zig build bitstream -Dvariant=w64-f100   # individual steps for debugging
zig build deploy    -Dvariant=w64-f100
zig build run                            # copy + run the board binary
```
