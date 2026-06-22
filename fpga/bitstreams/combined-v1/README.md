# penzai combined bitstream (matmul + flash, both on PL)

**One bitstream that serves decode end-to-end on PL** — the q1a8 four-port matmul kernel
*and* the kv-major v2 flash kernel in a single design. This is the integration that turns
the two proven-separately kernels into a real system win: on the matmul-only bitstream
flash runs on PS at 30.9 ms/tok; here it runs on PL at ~13.5 ms/tok, so decode drops
from ~149 to ~132 ms/tok (≈ 6.7 → 7.6 tok/s).

The two ops run **sequentially** in the graph, so their DDR feeds are kept independent
rather than time-shared — each keeps exactly the topology it was tuned/validated with:

| | kernel | clock | DMAs | PS ports |
|---|---|---|---|---|
| **matmul** (unchanged from `q1a8-w256-mc`) | `kernel_mm` (`matmul_top`) | `wclk` feed / `fclk` ctrl | dma_w0–3 + dma_a | **HP0–3** (GP2–5) |
| **flash** (unchanged from `flash-v1`, v2 kernel) | `kernel_fa` (`flash_top`) | `fclk` | dma_q/k/v/mask/o | **HPC0–1** (GP0–1, coherent) |

Both kernels run on the **shared `fclk`**; `wclk` is the matmul weight-feed clock only.
Flash gets the otherwise-free coherent HPC ports (matmul owns all four HP ports for its
512-bit weight bandwidth). Address maps don't collide by design — matmul `0xA00x_0000`,
flash `0xA01x_0000` (the flash regmap was allocated in `0xA01x` for exactly this).

## Build & deploy
```sh
cp config.env.example config.env       # edit VM / BOARD (or reuse the committed one)
(cd ../../.. && zig build regmap)       # refresh matmul_regs.vh, flash_regs.vh, both address_map.tcl
./build.sh                              # variant w512-p4-f200-wc300 (start here — f200 is flash-validated)
./deploy.sh
# serve the daemon telling it BOTH ops are on PL:
(cd ../../.. && nix run .#deploy-penzaid)
PENZAI_PL_OPS=all nix run .#serve-penzaid
# PL init must report BOTH:  "pl: q1a8 kernel ready"  AND  "pl: flash kernel ready"
```

`PENZAI_PL_OPS=all` is required — it makes the daemon probe both kernels (each present in
this bitstream). With the default (`matmul` only) flash would stay on PS; with `flash`
only matmul would. (The runtime already supports `all`; no daemon code change.)

## The open question this build answers: does it FIT?

This is the one thing cosim can't tell us — whether matmul's wide array **and** flash's
datapath fit and close timing **together** on the XCK26. The build is the test:

- `build.tcl` writes **`out/<bit>_utilization_synth.rpt`** right after synthesis (before
  the long impl) — **check DSP / LUT / BRAM there first.** If synthesis over-maps, that
  report says what's over before you wait for place-and-route.
- It then runs impl and **refuses to write a bitstream unless timing closes** (WNS ≥ 0),
  same as the standalone builds.

If it doesn't fit or close at `f200`:
- **Timing** — drop `f` (e.g. `w512-p4-f150-wc300`); flash closes ≥ f200 but the combined
  routing is denser, so a lower `f` may be needed first. `wc` can also come down.
- **Resources** — the levers are matmul width (a narrower array) or sharing flash's fp
  units; bring the utilization report back and we'll pick.

## Validate on silicon
1. **Correctness:** `PENZAI_PL_OPS=all PENZAI_PL_VERIFY=1 nix run .#serve-penzaid`, then a
   run — expect **no** `pl verify` mismatch lines for either op.
2. **Perf:** `--prof` over `tcp:` — now **both** `matmul_q1a8` and `flash_attn_f32` show PL
   numbers (`MAC/cyc`, real `flash_ms_tok ~13`), the scoreboard `variant` shows a real
   clock (not `f0`), and decode should land ~132 ms/tok. This is the first run where the
   *whole* decode is on PL.

## Deprecating the old bitstreams
`q1a8-w256-mc` (matmul-only) and `flash-v1` (flash-only) stay until this is proven out
fully on silicon — they remain the fallback and the per-op timing references. Once the
combined build is validated, they can be retired.
