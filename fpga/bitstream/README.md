# penzai combined bitstream (GEMM/FFN + adaptive flash on PL)

**One bitstream for compressed GEMM, the named FFN section, and supported flash
operations on PL** - the fixed-point four-port GEMM kernel and adaptive
query-blocked flash kernel in one design. Flash engine `0xF1A54A01`, version 1,
operates directly on native KV layout,
performs GQA in hardware, and exposes 64 physical query-head slots. Decode uses one
query. Multi-token calls use four-query tiles at up to 16 heads and two-query tiles
at 17-32 heads, so Bonsai retains a four-query physical tile without allocating a
128-slot state bank.

P2d advanced GEMM engine `0xB05A2000` to version 13 and wire ABI 14. In addition
to the existing packed-Q8 primitive input, it can quantize raw F32
activations into the resident Q8 store and reuse a validated resident epoch for an
adjacent second projection. The fixed grouped command structurally accepts any
adjacent compatible pair; in the validated Qwen/Bonsai graphs, only the 28 FFN
gate/up pairs per graph match and Q/K/V do not.

The deployed P2e image advances GEMM to version 14 without changing wire ABI 14.
It implements 512 KiB of four-bank F32 scratch in exactly 16 URAM288s and adds an
opt-in atomic DDR+scratch tee plus canonical drain. Normal behavior remains the
P2d DDR path. `PENZAI_PL_SCRATCH_VERIFY=1` limits grouped tiles to four queries,
tees Qwen projection 0 (`UP`) to `X1` and projection 1 (`GATE`) to `X0`, then
byte-compares both drains against their authoritative DDR results. This is a
diagnostic substrate, not yet a named or scratch-only section command.

The deployed P2f image advances wire ABI to 15 and GEMM to version 15. Command
tag 19 identifies a version-1 named FFN command serialized in 172 bytes. The PS performs
weighted RMSNorm, PL retains UP/GATE in X1/X0 through a 15-cycle II=1 PWL SwiGLU,
canonical Q8 requantization, and DOWN, then the PS performs the residual add from
a private result. This is the first executable section and closes the P2 contract
and substrate; optimizing its product-path performance is P3, while a named
attention section remains P4.

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
./build.sh w512-p4-f285                 # current bounded P2f qualification point
./build.sh                              # f300 development target; combined route unclosed
./build.sh --incremental                # development build; reuse last clean route
PENZAI_BITSTREAM_RUN_ID=20260813T112328Z-6e8fae0eb637-w512-p4-f285-clean \
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
  verifies the constraint delta and nominal restoration. Final setup and hold
  slack must be nonnegative; no additional margin threshold is imposed.
- It records methodology findings, utilization, near-critical path counts, phase times,
  source identity, and build configuration in the run bundle.

A released clean P2b timing run using the guardband plus pre-route
`AggressiveExplore` reached setup/hold of +0.033/+0.007 ns, with 12 setup
paths below 50 ps. It uses 80,108 LUTs, 94,587 FFs, 48.5 BRAMs, four
URAMs, and 92 DSPs; 98.25% of CLBs are occupied. The remaining short tail spans
the disabled sequencer, flash, and several unrelated GEMM structures, so 50 ps
is not treated as a single-path architectural boundary. Run
`20260812T062923Z-b829dee03903-dirty-w512-p4-f300-clean` passes timing,
was promoted and deployed, and passed receipt/capability, Q1/Q2 logits, and
profile smoke checks. Resolve critical TIMING-2/TIMING-4 plus the five TIMING-28
and one ULMTCS-1 warning before deriving a stricter production margin from an
explicit timing budget.

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
bitstream hashes. It passes timing with five setup paths below 50 ps; it established
P2c's qualification point, while f300 remains unclosed.

### P2d bounded qualification status

P2d RTL validation passes for both binary and ternary weights. The first
projection raw-loads 128 physical 64-bit activation beats in the focused test and
the second projection reuses resident Q8 with zero activation beats. Invalid epoch
or shape reuse contracts consume no activation beats, reject before the kernel
consumes weights or emits results, and preserve the previous resident record. Raw
framing, non-finite, scale, or arithmetic aborts invalidate resident state before
kernel weight/result consumption. DMA movers may already be armed on either path.
The exact Q8 leaf agrees with the software oracle across 1,032 finite-domain
blocks. Aggregate RTL cosim and focused formal reuse/abort checks pass.

The first clean f285 build,
`20260812T212725Z-f9e1ca83f8ae-w512-p4-f285-clean`, routed fully but failed closed
at -0.215/+0.010 ns setup/hold. It had 142 failing setup paths and 580 below 50 ps,
used 81,284 LUTs, 96,220 FFs, 55.5 BRAM tiles, four URAMs, and 95 DSPs, and reached
99.32% CLB occupancy. It was not promoted or deployed. Commit `547d87b` repairs
the identified ingress-control and GEMM exponent paths with nested counters, a
registered scalar boundary, compact output emission, and `FE_LAT` exponent retiming.
The repair removes 228 state bits and one DSP, preserves the focused hashes and
cycles, and expands nested-boundary/framing coverage.

Full-decode integration OOC run
`20260812T223110Z-547d87b12094-full-decode` passes the 3.333 ns constraint at
+0.245/+0.039 ns setup/hold with 40,141 LUTs, 44,912 FFs, 775 CARRY8s, 34 DSPs,
two BRAM tiles, and four URAMs. The failed route's targeted path families are
absent from the current probe. The 3.333 ns production artifacts pass; the probe
driver exited afterward when its read-only, non-production second-period report
script attempted to mutate the open design. That later report failure does not
affect the production source or result.

Replacement clean f285 run
`20260812T224303Z-547d87b12094-w512-p4-f285-clean` uses clean commit
`547d87b12094` and source bundle
`1cfc1e173ba0ae1d06d1fceb1d3fa83ec29535f760a7a7f91a6b5e0458078249`.
It is fully routed with all structural timing counts zero and exact restoration of
the 75 ps placement guardband. Setup/hold pass at +0.043/+0.010 ns with no negative
paths; 4/55/622 setup paths are below 50/100/200 ps. Routed utilization is 81,887
LUTs, 95,699 FFs, 1,408 CARRY8s, 94 DSPs, 55.5 BRAM tiles, and four URAMs, with
14,519/14,640 CLBs used (99.17%). Methodology reports critical TIMING-2/TIMING-4
plus five TIMING-28 and one ULMTCS-1 warning.

| Gate | P2d result |
|---|---|
| First clean `w512-p4-f285` route | failed closed; run above was not promoted |
| Replacement clean `w512-p4-f285` route | pass; +0.043/+0.010 ns |
| Promotion and deployment receipt | pass; bit SHA `9ab576cad24eb3c77d6b55200d5e9a08d92f0197625f76d479c13fc6fa82a70f` |
| Capability identity | pass; schema 2, wire 14, profile 6, GEMM v13 at 284,997,152 Hz, flash v1/64 slots |
| Q1 `p32` logits | pass; max abs 0.098204, zero token mismatches |
| Q2 `p32` logits | pass; step diffs 0.089185/0.131115, exact argmax, zero token mismatches, `check=ok` |
| Grouped/profile invariants | pass; all group paths PL, exact calls/runs/beats, closed accounting |
| Device-time regression check | pass; all four single-repeat P2c comparisons are lower |

Board artifact `20260812T235644Z-characterize-9bb6d3eb522f` is complete and
run-validated with all six expected samples, `accounting=ok`, and identical start
and end capabilities. Its exact workload structure is:

| Phase | Graphs | Commands | Group2 | Primitive matmul |
|---|---:|---:|---:|---:|
| p128 prefill | 15 | 4,002 | 217 | 1,114 |
| p128 decode | 1 | 509 | 28 | 141 |
| c0 prefill | 1 | 509 | 28 | 141 |
| c0 decode | 64 | 32,576 | 1,792 | 9,024 |
| c512 prefill | 63 | 15,978 | 865 | 4,450 |
| c512 decode | 64 | 32,576 | 1,792 | 9,024 |

Every group runs on PL: `pl/staged` for 16 columns and `pl/direct` for one-column
tails/decode, with no fallback. Each group raw-loads A once and reuses it with zero
activation beats. The c512 decode `6144x1x2048` bucket records 1,792 calls,
3,584 runs, A/R beats of 1,835,008/11,010,048, and Q1/Q2 W beats of
110,100,480/198,180,864; host quantize/pack time is zero. P128 records a
216-call/864-run/3,538,944-A-beat staged main bucket plus one direct two-run,
1,024-beat tail. C512 records 864/3,456/14,155,776 for the staged main bucket plus
the same direct tail.

Attention remains unmodified on its expected staged/direct PL paths, with zero
K/V/O stalls and at most 0.01% cycle delta from P2c. Approximate-flash verification
advisories cover 12 smoke calls and 40 of 24,576 values, with maximum absolute and
normalized differences of 0.5542/0.0467. These are expected diagnostics from the
unchanged approximate flash path: Q1/Q2 model logits pass, and there are no GEMM,
group, Q8, DMA, kernel, or request errors.

The single-repeat device-time check shows no regression; all four observations are
lower than P2c. P128 Q1/Q2 moves
73.282->73.028 ms/token (-0.35%) and 96.143->94.659 (-1.54%); c512 moves
96.093->94.773 (-1.37%) and 119.008->117.700 (-1.10%). The new c0 anchor is
65.716/88.615 ms/token. Raw PL quantization does raise grouped kernel cycles over
two primitive calls by 25.98%/21.87% for Q1/Q2 staged columns-16 and
18.28%/8.96% for c512 decode. Command removal and zero grouped host quantize/pack
time are consistent with offsetting that cost; one repeat does not establish a
repeatable speedup.

P128 prefill wall is 47.259/91.355 s and c512 is 224.114/411.642 s. This run
observed a transport-rate slowdown without changing download volume. P128 Q1/Q2
changed from
8.1 s at 40.5 MiB/s and 15.3 s at 43.0 MiB/s to 36.5 s at 9.0 MiB/s and
79.9 s at 8.2 MiB/s. C512 Q1/Q2 changed from 35.0 s at 41.6 MiB/s and
58.5 s at 49.7 MiB/s to 176.5 s at 8.3 MiB/s and 362.4 s at 8.0 MiB/s. That
degraded transport is not a P2d accelerator regression.

Qualification teardown stopped the daemon and left no buffer objects or serving
processes. The exact deployed application remains loaded in XRT slot 0 (slot
0 before and after replacement).

P2d closed at this bounded f285 qualification. At that checkpoint P2 remained
open for scratch-backed outputs and the named FFN/attention section commands.

### P2e diagnostic scratch qualification

The scratch implementation is four physical `16384 x 64` banks: 65,536 words,
512 KiB, and exactly 16 URAM288s. The leaf cosim checks 65,616 input beats,
16,404 assembled groups, and every physical word, including mapping, bounds,
framing, backpressure, collisions, abort, and retention. Integrated binary and
ternary `decode_top` tests tee and drain X1/X0 exactly, retain the first role while
writing the second, reject bad shapes before consuming W/A/R, reject stale roles,
and suppress output after drain abort. The aggregate RTL suite passes 41/41 steps.
Map formal closes 23 PDR assertions plus 34 BMC assertions and five covers at depth
32; storage formal closes 24 assertions through depth 64 and reaches its terminal
cover at step 45.

Standalone OOC run `20260813T032516Z-c856d82731dd` passes 3.333 ns at +0.220 ns
WNS with 99 LUTs, 447 FFs, five CARRY8s, zero DSPs, exactly 16 URAMs, and zero
BRAM/LUTRAM. Full-decode integration run
`20260813T040643Z-c856d82731dd-dirty-p2e-groupreg-full-decode` passes at
+0.042/+0.060 ns setup/hold with 40,406 LUTs, 45,522 FFs, 783 CARRY8s, 34 DSPs,
two BRAM tiles, and exactly 20 URAMs.

The original clean combined run,
`20260813T042026Z-1e0b9a350d84-w512-p4-f285-clean`, is fully routed and meets
timing at +0.007/+0.008 ns setup/hold with zero TNS/THS. It has zero clockless
or unconstrained internal endpoints, and exact restoration of the 75 ps placement
guardband. It was initially rejected only because the build still imposed a 25 ps
project floor. Commit `15f3ec3` removed that policy and retained the actual release
rule: final setup and hold slack must both be nonnegative.

No-reroute run
`20260813T060347Z-1e0b9a350d84-w512-p4-f285-routed-finalize` finalized the exact
same routed checkpoint under the corrected rule. Provenance is explicit:

| Item | SHA-256 |
|---|---|
| Origin routed DCP | `2877ac5a84beb335b99820cc6aae14d94a7b440eef34d56bed74174230845d05` |
| Origin manifest | `fc75587fc2eb95ce3cf4ccb789076c89db8e397c195320240764200c03975a2d` |
| Source bundle | `0dac679c47f31932e401e9766205061a50fd703fef8940e8349ce3ff2b2247ff` |
| Final manifest | `b2b6c1d2d100ba8f4b7bc927ba5eb7840278a878f0ac77cda6cb156656438e17` |
| `.bit` | `3d7a95d08ad58dcf6ebefa7e9b1d1cdf6e660d0fed220fd1abb89bc2e701b04e` |
| `.bit.bin` | `ca630e47e7a47fea67b745b754d9f1ccff5d7e0868c1871c86b46e6d2326baed` |

Routed utilization is 82,145 LUTs, 96,755 FFs, 1,416 CARRY8s, 55.5 BRAM tiles,
20 URAMs, and 94 DSPs. CLB occupancy is 14,594/14,640 (99.69%). There are 127
setup paths below 50 ps; that count is diagnostic input for a future timing-
optimization pass, not an additional release threshold. Methodology retains
TIMING-2/TIMING-4, three TIMING-28 warnings, and ULMTCS-1.

The exact image was promoted and deployed. Scratch-on and scratch-off capabilities
are byte-identical at schema 2, wire ABI 14, profile ABI 6, GEMM v14 at
284,997,152 Hz, and flash v1 with 64 slots. Both modes pass the same `p32` logits:
Q1 step differences 0.098204/0.084137 and Q2 0.089185/0.131115, exact argmax
25/25 and 220/220, zero token mismatches, and `check=ok`. The enabled run completes
the full X1 and X0 drains without a byte mismatch.

Profile artifact `20260813T062032Z-characterize-8f96db0c8124` is complete and
run-validated for all four p128/c512 Q1/Q2 samples. P128 device time is
71.948/95.031 ms/token and c512 is 94.820/117.732 ms/token. Against P2d, p128 Q1
is lower, p128 Q2 is +0.39%, and c512 Q1/Q2 are +0.05%/+0.03%; all pass the
no-regression gate. P128/c512 prefill wall is 16.291/24.796 s and
70.387/92.653 s; unchanged downloads sustain
49.6/44.9 and 52.0/59.5 MiB/s. The graph, command, group, primitive, attention,
and beat structures remain the P2d values and accounting closes. These single
observations establish no regression, not a speedup.

Unprofiled artifact `20260813T062620Z-regression-7f1de76f4fab` is complete and
run-validated across six c0 samples. Its three-repeat steady-decode medians are
87.922 ms/token for Q1 and 109.635 ms/token for Q2. P2e therefore closes the
diagnostic scratch substrate. The following P2f image consumes it in the first
named section.

### P2f named FFN qualification

Wire ABI 15 freezes `ffn_section` tag 19, contract version 1 and flags zero. Its
172-byte command names residual, norm weight, up/gate/down weights, destination,
dimensions, token count, epsilon, and Q1/Q2 format; a one-command buffer is 176
bytes including its four-byte count. The strict lowerer requires the exact
Qwen/Bonsai dataflow, shapes/formats, use and alias safety, complete resident
bindings, and six pairwise-disjoint external ranges. A mismatch stays on the
legacy commands before section execution. The pinned one-graph census changes
from 508 commands, 28 group2, and 141 primitive matmuls to 396 commands, 28 FFN,
zero group2, and 113 primitive matmuls, exactly 112 commands removed.

The v15 executable path tiles at up to four tokens. Weighted RMSNorm and residual
add remain on the PS. PL canonical-Q8 ingress feeds UP to scratch-only X1, reuses
that Q8 record for GATE into X0, applies the 1,024-segment `[-16,16]` PWL SwiGLU,
requantizes through the canonical ingress, and runs DOWN into a private result.
Only after a successful tile does the PS add the residual, publish `dst` once,
and synchronize it. Unsupported pre-start execution may use the whole named PS
oracle; after PL starts a hardware or DMA error is a backend failure with no
retry. R and X2 remain contract roles but are not populated by this executable
subset.

`test-rtl` passes 46/46 steps; `lint-rtl test-rtl` together pass 48/48. The
normal/pinned Zig suites pass 265/265 and 292/292. The 15-cycle II=1 SwiGLU cosim covers 5,143 scalars and all
1,024 ROM entries. Normalized PS error is `6.097758e-5`; canonical Q8 drift is
2/32,768 bytes with maximum delta one and no F16 scale-code delta. SwiGLU and
internal Q8 ingress have unbounded PDR proofs; decode-FFN has a depth-400 BMC and
cover, not an unbounded proof.

Standalone SwiGLU OOC run `20260813T082045Z-48c8be5cebaf` passes at +0.336 ns
WNS with 771 LUTs, 784 FFs, 17 CARRY8s, four DSPs, and 2.5 BRAM tiles. Final
integrated OOC run
`20260813T110854Z-90a2dc70c5e8-dirty-p2f-romreq-full-decode` passes at
+0.198/+0.037 ns setup/hold with 41,580 LUTs, 47,235 FFs, 808 CARRY8s, 38 DSPs,
4.5 BRAM tiles, 20 URAMs, and 94 LUTRAMs. Its production-source and complete-probe
bundle SHA-256 values are
`c0bb8d16aceef6343103f7ac66b0c570fbb7a4200972ef07ea6483a6e03050e0` and
`55024b877f0e3c118d380684914dd229761c7ca5b5439dab60e03d9ecd15fb81`.

The first clean P2f route,
`20260813T094435Z-90a2dc70c5e8-w512-p4-f285-clean`, was fully routed but failed
-0.081/+0.007 ns and was not promoted. Commit
`6e8fae0eb637589cc0d31c0d14e2a53603830c2b` pipelined the scratch, Q8, and
SwiGLU boundaries. The replacement authority is clean combined run
`20260813T112328Z-6e8fae0eb637-w512-p4-f285-clean`:

| Gate | P2f result |
|---|---|
| Timing | pass; +0.015/+0.010 ns setup/hold, no negative paths, 32 setup paths below 50 ps |
| Routing/constraints | pass; 161,294/161,294 routable nets, no clockless or unconstrained internal endpoints, exact guardband restoration |
| Utilization | 83,463 LUTs, 98,323 FFs, 1,441 CARRY8s, 58 BRAM tiles, 20 URAMs, 98 DSPs |
| Methodology | TIMING-2/TIMING-4, five TIMING-28, and ULMTCS-1 remain |
| Source bundle | `a4536e2a73b10fb6528019b9b2f0b0b09b659584932da85cda06826c61bf555c` |
| Manifest | `032b2349380a5b03eee6f9870ece88e87d24ea6cdb3fff557fd55eda582dab26` |
| Raw `.bit` | `39b89b68f50393f528a95eeee5595ca1182c7a4dc40d27c0e6067a5e0c602283` |
| Deployed `.bit.bin` | `b8f983b8065a9eeb6eb850dc6d296f613e72f5323a48e733b6260a853c522904` |

Both artifacts were promoted; the exact `.bit.bin` was deployed with a matching
receipt.
Consolidated board evidence is `/tmp/p2f-qualification-summary.json`, SHA-256
`a64175d3379fa545da904da98df57c9899a244b8a74c5bbbf3fc472aa1d1e66a`.
Start/end capabilities match at schema 2, wire 15, profile 6, GEMM v15 at
284,997,152 Hz, and flash v1/64 slots. Q1/Q2 `p32` logits pass at maximum absolute
differences 0.102924/0.107559, exact 25/25 and 220/220 argmax, zero token
mismatches, and `check=ok`; no fallback or execution error occurs. Each Q1/Q2
`p32` prefill and decode profile phase reports one graph, 397 commands, 28 FFN,
zero group2, and 113 primitive matmuls. The extra command versus the pinned census
is argmax. All FFN buckets execute on PL and close their run and W/A/R beat
formulas.

Characterization artifact
`p2f-characterize-20260813T112328Z-6e8fae0eb637-w512-p4-f285-clean` is complete
and run-validated with fingerprint
`20928b3c636dbe0504900a9c3adb40033bd7820d2069a640ba667d8e8e5bcf33`.
Its single-repeat profiled device times improve from P2e by 4.59-7.53%: p128
Q1/Q2 is 66.533089/89.390916 ms/token and c512 is
89.438339/112.333491 ms/token. This is not a blanket speedup claim. Complete,
run-validated six-sample artifact
`p2f-regression-20260813T112328Z-6e8fae0eb637-w512-p4-f285-clean`, fingerprint
`0e61f6f63bd558e5692a4428cc2ac2b04958c747c1e9d030fd1f2bd47950f35e`,
instead records unprofiled c0 medians of 90.364345/112.856650 ms/token for Q1/Q2:
regressions of 2.7773%/2.9388% that pass the +15% guard. P2f closes P2 and
establishes the P3 baseline; P3 remains open for repeatable product-path gain.

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
   the numerical release gate. The local P2f aggregate covers binary and ternary
   GEMM, exact Q8 ingress and grouped reuse, scratch, named FFN/SwiGLU execution,
   and the flash kernel/wrapper; all 46 `test-rtl` steps pass, and the combined
   `lint-rtl test-rtl` invocation passes 48/48. A current P2f smoke
   must report 28 `ffn_section` commands on PL, zero group2 commands, exact
   run/beat accounting, and no fallback or execution error.
2. **Identity:** query `penzai capabilities` after every deployment and require the
   run ID, source hash, bitstream hash, ABI, clock, engine versions, dimensions, and
   formats to match the selected run bundle. Adaptive P2c requires capability schema
   2 (552 bytes), wire ABI 13, profile ABI 6, flash ID `0xF1A54A01`, flash version 1,
   and `flash.query_slots=64`. A stale daemon or bitstream must fail closed rather
   than reinterpret the engine contract. P2d additionally requires wire ABI 14,
   matmul ID `0xB05A2000`, and matmul version 13. P2e retains wire ABI 14 and
   requires matmul version 14. P2f requires wire ABI 15, matmul version 15, and
   command tag 19 with contract version 1/flags zero. The current deployment reads
   back schema 2, wire ABI 15, profile ABI 6, matmul v15 at 284,997,152 Hz, and
   flash v1 with 64 query slots.
3. **Profile invariants:** `--prof` must report PL execution for both GEMM and
   single-query flash plus supported multi-token flash, closed accounting, exact
   requested token counts, and the expected DMA beat counts with no unexplained
   stalls. Named FFN profiling must report one aggregate wall time plus exact
   logical gate/up and down bucket accounting; `swiglu_ns=0` means it is fused
   into the down wait, not independently free. Within a query tile, K/V beat
   counts are independent of query count. Compare device counters, not
   VPN-sensitive wall or transport time.

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
