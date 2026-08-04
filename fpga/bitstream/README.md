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
./build.sh --incremental                # development build; reuse last clean route
./deploy.sh
# serve the daemon telling it BOTH ops are on PL:
(cd ../../.. && nix run .#deploy-penzaid)
PENZAI_PL_OPS=all nix run .#serve-penzaid
# PL init must report BOTH:  "pl: q1a8 kernel ready"  AND  "pl: flash kernel ready"
# Query the receipt plus live engine registers through the daemon:
nix run .#penzai -- capabilities --device tcp:kria:29092
```

`PENZAI_PL_OPS=all` makes the daemon probe both kernels. It is a selection flag,
not proof of what is loaded. `penzai capabilities` is authoritative: it returns the
hash-verified deployment receipt and only advertises engines whose live ID, version,
dimensions, and clock registers passed initialization.

## Fit and timing

Cosim cannot establish whether GEMM and flash fit and close timing together on the XCK26.
The routed build remains the gate:

- `build.tcl` writes `utilization_synth.rpt` immediately after synthesis, before the
  long implementation phase.
- It refuses to write a bitstream unless setup and hold timing close, the design is
  fully routed, and `check_timing` finds no clockless or unconstrained internal endpoints.
- It records methodology findings, utilization, near-critical path counts, phase times,
  source identity, and build configuration in the run bundle.

Builds use all eight Vivado worker threads and a persistent VM-side `cache/` for generated
AMD IP. Every timing-clean build refreshes a routed checkpoint for its variant.
`--incremental` uses that checkpoint to accelerate localized RTL changes; a missing
checkpoint falls back to clean implementation. Use the default mode for an independent
full build.

Each invocation writes an immutable ignored bundle under `out/runs/<run-id>/`:

```text
summary.txt             concise whole-run pass/fail result
vivado_summary.txt      routed gates and key metrics
manifest.tsv            Git/source/Vivado/part/directive identity
source_files.tsv        ordered path and SHA-256 input manifest
metrics.tsv             timing, constraints, utilization, and phase durations
*_routed.rpt            detailed reports for diagnosis
<bit>.bit(.bin)         artifacts produced by this exact run
driver_status.tsv       host-side Vivado/bootgen exit status
```

On success, only the `.bit` and `.bit.bin` are promoted to the stable paths directly
under `out/`, preserving the existing deploy interface. `out/latest` points to the full
bundle. Failed and partial runs remain inspectable but cannot replace a deployable image.

`deploy.sh` resolves one exact successful bundle (normally `out/latest`), verifies its
bitstream against the promoted file and again after transfer, and installs
`deployment_receipt.tsv` beside the firmware. Set `PENZAI_BITSTREAM_RUN_ID` to deploy a
specific retained run. The daemon reads this receipt by default; it can be overridden
with `PENZAI_BITSTREAM_RECEIPT` or `--bitstream-receipt` for test environments.

Use `../tools/analyze.sh summary` to reapply the production metrics and gates to the
retained routed checkpoint, or `../tools/analyze.sh deep` for detailed congestion, path,
fanout, QoR, and clock reports. These tools inspect a checkpoint; they do not produce a
new bitstream.

If it doesn't fit or close at `f200`:
- **Timing** — drop `f` (e.g. `w512-p4-f250`) if a future change causes the combined
  design to miss timing.
- **Resources** — the levers are matmul width (a narrower array) or sharing flash's fp
  units; bring the utilization report back and we'll pick.

## Validate on silicon
1. **Runtime correctness test:** `PENZAI_PL_OPS=all PENZAI_PL_VERIFY=1 nix run .#serve-penzaid`,
   then a run. Expect no matmul mismatch or flash approximation-outlier lines. Flash
   uses a 2% normalized comparison against the higher-precision PS implementation;
   the bit-faithful structural gate is `zig build test-rtl-flash-kernel`.
2. **Perf:** `--prof` over `tcp:` — now **both** `matmul_q1a8` and `flash_attn_f32` show PL
   numbers (`MAC/cyc`, real `flash_ms_tok ~13`), the scoreboard `variant` shows a real
   clock (not `f0`), and decode should land ~132 ms/tok. This is the first run where the
   *whole* decode is on PL.

Older standalone flash artifacts report `VERSION < 3` and are rejected by the current
driver. The former matmul-only and flash-only build trees have been retired; this is the
only production bitstream flow.
