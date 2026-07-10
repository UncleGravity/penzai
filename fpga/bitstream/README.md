# penzai combined bitstream (matmul + flash, both on PL)

**One bitstream that serves decode end-to-end on PL** — the fixed-point four-port GEMM
kernel and kv-major v3 flash kernel in a single design. Flash v3 operates directly on
the native KV layout, performs GQA in hardware, and supports 32 query heads.

The two ops run **sequentially** in the graph, so their DDR feeds are kept independent
rather than time-shared — each keeps exactly the topology it was tuned/validated with:

| | kernel | clock | DMAs | PS ports |
|---|---|---|---|---|
| **GEMM** | `kernel_mm` (`decode_top`) | `fclk` | dma_w0–3 + dma_a | **HP0–3** (GP2–5) |
| **flash** (kv-major v3) | `kernel_fa` (`flash_top`) | `fclk` | dma_q/k/v/mask/o | **HPC0–1** (GP0–1, coherent) |

Both kernels and all data movement run on the single **`fclk`** domain.
Flash gets the otherwise-free coherent HPC ports (matmul owns all four HP ports for its
512-bit weight bandwidth). Address maps don't collide by design — matmul `0xA00x_0000`,
flash `0xA01x_0000` (the flash regmap was allocated in `0xA01x` for exactly this).

## Build & deploy
```sh
cp config.env.example config.env       # edit VM / BOARD (or reuse the committed one)
(cd ../.. && zig build regmap)          # refresh generated contracts under fpga/regmap/
./build.sh                              # default variant: w512-p4-f300
./deploy.sh
# serve the daemon telling it BOTH ops are on PL:
(cd ../../.. && nix run .#deploy-penzaid)
PENZAI_PL_OPS=all nix run .#serve-penzaid
# PL init must report BOTH:  "pl: q1a8 kernel ready"  AND  "pl: flash kernel ready"
```

`PENZAI_PL_OPS=all` is required — it makes the daemon probe both kernels (each present in
this bitstream). With the default (`matmul` only) flash would stay on PS; with `flash`
only matmul would. (The runtime already supports `all`; no daemon code change.)

## Fit and timing

Cosim cannot establish whether GEMM and flash fit and close timing together on the XCK26.
The routed build remains the gate:

- `build.tcl` writes **`out/<bit>_utilization_synth.rpt`** right after synthesis (before
  the long impl) — **check DSP / LUT / BRAM there first.** If synthesis over-maps, that
  report says what's over before you wait for place-and-route.
- It then runs impl and **refuses to write a bitstream unless timing closes** (WNS ≥ 0),
  before producing an invalid artifact.

If it doesn't fit or close at `f200`:
- **Timing** — drop `f` (e.g. `w512-p4-f250`) if a future change causes the combined
  design to miss timing.
- **Resources** — the levers are matmul width (a narrower array) or sharing flash's fp
  units; bring the utilization report back and we'll pick.

## Validate on silicon
1. **Correctness:** `PENZAI_PL_OPS=all PENZAI_PL_VERIFY=1 nix run .#serve-penzaid`, then a
   run — expect **no** `pl verify` mismatch lines for either op.
2. **Perf:** `--prof` over `tcp:` — now **both** `matmul_q1a8` and `flash_attn_f32` show PL
   numbers (`MAC/cyc`, real `flash_ms_tok ~13`), the scoreboard `variant` shows a real
   clock (not `f0`), and decode should land ~132 ms/tok. This is the first run where the
   *whole* decode is on PL.

Older standalone flash artifacts report `VERSION < 3` and are rejected by the current
driver. The former matmul-only and flash-only build trees have been retired; this is the
only production bitstream flow.
