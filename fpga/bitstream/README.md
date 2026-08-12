# penzai combined bitstream (matmul + adaptive flash, both on PL)

**One bitstream for compressed GEMM and supported flash operations on PL** — the
fixed-point four-port GEMM kernel and adaptive query-blocked flash kernel in one
design. Flash engine `0xF1A54A01`, version 1, operates directly on native KV layout,
performs GQA in hardware, and exposes 64 physical query-head slots. Decode uses one
query. Multi-token calls use four-query tiles at up to 16 heads and two-query tiles
at 17-32 heads, so Bonsai retains a four-query physical tile without allocating a
128-slot state bank.

The two ops run **sequentially** in the graph, so their DDR feeds are kept independent
rather than time-shared — each keeps exactly the topology it was tuned/validated with:

| | kernel | clock | DMAs | PS ports |
|---|---|---|---|---|
| **GEMM** | `kernel_mm` (`decode_top`) | `fclk` | dma_w0–3 + dma_a | **HP0–3** (GP2–5) |
| **flash** (adaptive query-blocked) | `kernel_fa` (`flash_top`) | `fclk` | dma_q/k/v/mask/o | **HPC0–1** (GP0–1, coherent) |

Both kernels and all data movement run on the single **`fclk`** domain.
Flash gets the otherwise-free coherent HPC ports (matmul owns all four HP ports for its
512-bit weight bandwidth). Address maps don't collide by design — matmul `0xA00x_0000`,
flash `0xA01x_0000` (the flash regmap was allocated in `0xA01x` for exactly this).

## Build & deploy
```sh
cp config.env.example config.env       # edit VM / BOARD (or reuse the committed one)
(cd ../.. && zig build regmap)          # refresh generated contracts under fpga/regmap/
./build.sh w512-p4-f285                 # current adaptive P2c release point
./build.sh                              # f300 development target; P2c does not yet close
./build.sh --incremental                # development build; reuse last clean route
PENZAI_BITSTREAM_RUN_ID=20260812T155038Z-3ef082b0fe4a-w512-p4-f285-clean \
  ./deploy.sh w512-p4-f285
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
- Placement uses a setup-only 75 ps guardband, removes it before routing, and
  verifies the constraint delta and nominal restoration. The build refuses to
  write or promote a bitstream below the 25 ps setup release floor. It records
  50 ps as the headroom target and warns when that target is missed; hold must
  remain nonnegative and is not artificially tightened.
- It records methodology findings, utilization, near-critical path counts, phase times,
  source identity, and build configuration in the run bundle.

A released clean P2b timing run using the guardband plus pre-route
`AggressiveExplore` reached setup/hold of +0.033/+0.007 ns, with 12 setup
paths below the 50 ps target. It uses 80,108 LUTs, 94,587 FFs, 48.5 BRAMs, four
URAMs, and 92 DSPs; 98.25% of CLBs are occupied. The remaining short tail spans
the disabled sequencer, flash, and several unrelated GEMM structures, so 50 ps
is not treated as a single-path architectural boundary. Exact-policy run
`20260812T062923Z-b829dee03903-dirty-w512-p4-f300-clean` clears the 25 ps floor,
was promoted and deployed, and passed receipt/capability, Q1/Q2 logits, and
profile smoke checks. Resolve the existing clock methodology warnings and derive
any stricter production margin from an explicit timing budget.

The first P2c design used 128 query-head slots so every 32-head shape could retain
a four-query tile. Although flash OOC passed at +0.215 ns, clean combined run
`20260812T112623Z-95b751ff9ac3-w512-p4-f300-clean` failed at -0.035 ns setup WNS
with 124 failing setup endpoints, 69.5 BRAM tiles, and 98.91% CLB occupancy. No
bitstream from that run was deployed. The adaptive 64-slot revision reduces the
flash OOC footprint from 33 to 19 BRAM tiles while retaining +0.215 ns WNS.

The adaptive clean f300 run
`20260812T144901Z-bbeac0c04eff-w512-p4-f300-clean` also failed closed, at
-0.088 ns setup WNS with 231 failing setup endpoints and 560 paths below 50 ps;
it was not promoted. Clean f285 run
`20260812T155038Z-3ef082b0fe4a-w512-p4-f285-clean` passes setup/hold at
+0.036/+0.010 ns with five setup paths below 50 ps. It uses 80,837 LUTs, 95,229
FFs, 55.5 BRAM tiles, four URAMs, and 92 DSPs at 98.61% CLB occupancy. The exact
image was promoted and deployed with verified source, manifest, transfer, and
bitstream hashes. It clears the 25 ps release floor but misses the 50 ps headroom
target, so f285 is the current qualification point and f300 remains unclosed.

Builds use all eight Vivado worker threads and a persistent VM-side `cache/` for generated
AMD IP. Every release-gate-passing build refreshes a routed checkpoint for its variant.
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
bitstream against the promoted file and again after transfer, explicitly unloads the
occupied XRT_FLAT slot, and verifies that the replacement app acquired the slot. The
local and board `deployment_receipt.tsv` files are promoted only after those checks
pass. Set `PENZAI_BITSTREAM_RUN_ID` to deploy a specific retained run. The daemon reads
this receipt by default; it can be overridden with `PENZAI_BITSTREAM_RECEIPT` or
`--bitstream-receipt` for test environments.

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
   then a run. Expect no matmul mismatch. Investigate flash approximation-outlier
   lines, but remember that this is a smoke comparison: flash uses a 2% normalized
   threshold against the higher-precision PS implementation. The bit-faithful
   structural gate is `zig build test-rtl`, and the model-level logit comparison is
   the numerical release gate. The aggregate RTL target covers binary and ternary
   GEMM plus the flash kernel and wrapper, and also passes as the Nix
   `checks.rtl-cosim` check.
2. **Identity:** query `penzai capabilities` after every deployment and require the
   run ID, source hash, bitstream hash, ABI, clock, engine versions, dimensions, and
   formats to match the selected run bundle. Adaptive P2c requires capability schema
   2 (552 bytes), wire ABI 13, profile ABI 6, flash ID `0xF1A54A01`, flash version 1,
   and `flash.query_slots=64`. A stale daemon or bitstream must fail closed rather
   than reinterpret the engine contract.
3. **Profile invariants:** `--prof` must report PL execution for both GEMM and
   single-query flash plus supported multi-token flash, closed accounting, exact
   requested token counts, and the expected DMA beat counts with no unexplained
   stalls. Within a query tile, K/V beat counts are independent of query count.
   Compare device counters, not VPN-sensitive wall or transport time.

The initial f285 multi-token smoke truthfully reported a PS fallback. The driver
had modeled Q as head-major, while the supported GGML prefill tensor is packed
`[token][head][dimension]`: `q_nb1` is the token stride and `q_nb2` is the head
stride. Commit `0dc91c0` corrected that exact eligibility predicate and added an
observed Bonsai-stride test; the bitstream was unchanged.

The repaired bounded board gates pass for Q1 and Q2. The 32-token (`p32`) logit
comparisons had zero token mismatches and maximum absolute differences of
0.0982/0.1396. Artifact
`20260812T174257Z-p2c-repaired-p128` reports 224 `pl/staged` prefill calls, 896
kernel runs, exact beats, zero K/V/O stalls, closed accounting, and 24.05 cycles
per processed query-head/KV update. Artifact
`20260812T174506Z-p2c-repaired-c512` reports 896 staged prefill calls and 3,584
kernel runs at 21.222 cycles per processed update, again with exact beats and zero
stalls. Q1/Q2 prefill fell from 160.300/185.434 s on the P2b f300 PS path to
79.132/104.541 s on P2c f285, reductions of 50.63%/43.62%. Decode remains
`pl/direct` at 33.315 cycles per
update and 96.093/119.008 ms/token of device time.

Artifact `20260812T175007Z-p2c-repaired-c2048` completes the scale check. Its
3,584 prefill calls and 14,336 kernel runs all use `pl/staged`, at 20.494 cycles
per processed query-head/KV update with exact pairs/beats, zero stalls, and closed
accounting. Q1/Q2 prefill is 425.994/813.215 s, 76.96%/57.79% below baseline
artifact `20260804T211805Z-baseline` at 1848.895/1926.439 s. Decode remains direct
at 32.944 cycles per update, 180.560/203.465 ms/token device, and
193.748/221.127 ms/token steady wall. Relative to P0, device time falls
66.14%/63.33% and the flash scoreboard falls from about 459.1 to 115.7 ms/token.
Q1 cycles per update are only 0.21% above P2b f300; the 3.74% device-time increase
is consistent with f285. Backend downloads remain 5.8/11.6 GiB, so graph and
section-boundary traffic now dominates prefill wall and transport.

For scale only, the final P2b context-512 characterization measured 33.224 cycles
per valid query-head/KV update, 27.016 ms/token in the flash kernel, and
93.715/115.429 ms/token of Q1/Q2 device time. The flash scoreboard includes wrapper
work and reported 30.2 ms/token. At context 2048, Q1 measured 32.874 cycles/update,
102.138 ms/token in the flash kernel, and 174.055 ms/token of device time, with
zero K/V/O stalls. Treat these as artifact- and workload-specific reference values,
not hard-coded pass thresholds.

Older standalone and pre-adaptive flash artifacts have a different engine identity
or slot contract and are rejected by the current driver. The former matmul-only and
flash-only build trees have been retired; this is the supported combined-bitstream
flow.
