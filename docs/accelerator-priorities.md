# Accelerator Priorities

Status date: 2026-08-12

This is the short, ranked tracker for turning Penzai from an op-at-a-time FPGA
backend into a robust binary and ternary transformer accelerator. Detailed design
notes remain in `docs/inference-architecture-roadmap.md`.

## Architecture decision

Keep the current compressed-weight GEMM topology. Do not replace it with a
conventional systolic array unless measurements after fusion show that spatial
column throughput is the remaining limit.

Preserve:

- four-port compressed-weight streaming;
- output-stationary, fixed-point GEMM accumulation;
- one GEMM core shared by all transformer projections;
- binary/ternary differences localized to weight decode;
- KV-major online attention with FP32 softmax state and accumulation;
- PS kernels as correctness fallbacks.

Change the shell around those kernels. The destination is two named layer sections,
FFN and attention, connected through banked on-chip scratch memory and started with
section-level commands. Q8_0 should be an internal PL activation type. DDR should
hold weights, historical KV, residual boundaries, and outputs that genuinely must
leave the section, not every intermediate tensor.

Use one online attention engine for both modes. Decode runs it with one query;
prefill runs it with a small query tile and reuses K/V across those queries. Do not
create independent prefill and decode attention datapaths.

## Current measured baseline

The P0 characterization has 20 profiled runs per model through 512-token prompts
and decode contexts 0 and 512. All 280 samples passed accounting and requested-token
count checks. One context-2048 feasibility run per model is also complete.

| Workload | Q1 | Q2 | Main signal |
|---|---:|---:|---|
| Prefill 128 | 22.89 s | 29.28 s | PS attention and boundary costs are both material |
| Prefill 512 | 163.43 s | 189.27 s | 63 graphs, about 20,400 commands |
| Decode context 0 | 106.2 ms/token | 125.8 ms/token | Device is about 84/105 ms/token |
| Decode context 512 | 210.8 ms/token | 260.6 ms/token | Flash is about 119 ms/token |
| Decode context 2048 | 581.6 ms/token | 604.5 ms/token | Device is 533.3/554.9 ms; flash about 459 ms |

Context-2048 prefill took 30.8/32.1 minutes for Q1/Q2. Each issued 255 graphs
and 81,550 commands, with 5.8/11.6 GiB of backend downloads. At that context,
attention is 86%/83% of decode device time. The effective KV rate is only about
0.5 GB/s, far below the weight-stream bandwidth, so the P0 attention schedule was
compute/control limited rather than DDR limited.

At the P0 baseline, prefill GEMMs ran in PL but multi-token attention fell back to
PS. P2c now executes supported primitive multi-token flash operations on PL in
adaptive query tiles. Its released f285 image and bounded Q1/Q2 board gates are
described below, including the completed context-2048 scale check.

Backend downloads at the baseline grow from roughly 1.4/2.8 GiB for Q1/Q2 at a
512-token prompt. Those are graph-boundary transfers, not GEMM-kernel DMA counters,
and the primitive P2c bridge does not remove them. Section-level boundary removal
therefore remains necessary. Old transport tail
measurements taken through mixed SSH tunnels are not signoff data, but device-clock
measurements are stable and trustworthy.

The canonical unprofiled regression artifact is
`20260805T001719Z-regression-632dcf611699` (fingerprint prefix
`632dcf611699`). It completed and validated all 30 samples:

| Workload | Q1 median | Q2 median |
|---|---:|---:|
| Prefill 128 | 22.52 s | 27.76 s |
| Decode context 0 | 99.0 ms/token | 118.4 ms/token |
| Decode context 512 | 205.3 ms/token | 248.6 ms/token |

Comparing overlapping medians puts aggregate-profiling overhead at roughly 2-7%,
depending on workload. Product throughput therefore comes from unprofiled regression;
profiled measurements are compared only against profiled measurements.

## Ranked work

| Rank | Track | Why now | Completion gate |
|---:|---|---|---|
| 0 | Measurement foundation (complete) | Later work needs a fixed A/B contract | Profiled characterization, unprofiled regression, immutable build identity |
| 1 | Bounded waste removal (complete) | Low-risk short-context savings and benchmark validation | Independent A/B for each change with identical numerical output |
| 2 | Section substrate and unified PL attention | Boundary traffic breaks prefill; attention dominates long-context decode | Banked scratch, PL Q8 ingress, tiled prefill on PL, faster 512/2K decode |
| 3 | FFN section pipeline | Contained proof of the P2 scratch and command contracts | Named FFN command retains intermediates on chip and beats the op path |
| 4 | Attention section pipeline | Eliminates QKV/RoPE/KV/residual materialization around the P2 engine | Named attention command with correct same-token KV visibility |
| 5 | Ternary weight and KV memory efficiency | Raises the Q2 bandwidth roof and enables larger contexts/models | Lossless format adapter, routed decoder, quality and bandwidth A/B |
| 6 | Control, DMA, and transport consolidation | Useful after section commands reduce operation count | Fewer movers/control targets and lower wall residual without device regression |
| 7 | Spatial GEMM scaling | Expensive and unsupported by present bottlenecks | Only start if fused prefill or batched decode is compute-bound |

### P0: truthful measurement and build identity

- [x] Report flash execution backend, actual mask-clamped KV extent, cycles, stalls,
  measured beats, and wrapper segments. Do not infer PL execution from the op name.
- [x] Bucket matmul by format and shape, and split quantize, pack, sync-to, setup,
  kernel, sync-from, and result-layout time.
- [x] Rename the present `TTFT` field to `first_decode_step`; report user TTFT as
  prefill wall plus first decode step.
- [x] Add a bitstream capability handshake containing source/build identity, ABI,
  enabled engines, frequency, dimensions, and formats.
- [x] Establish repeated Q1/Q2 prefill and decode characterization through context
  512, plus a context-2048 feasibility measurement.
- [x] Measure profiler overhead against an unprofiled run.
- [x] Replace the large p95 matrix with small immutable characterization and
  regression artifacts that report median and range.
- [x] Retain successful bitstreams with a versioned source/build manifest,
  timing/utilization reports, and a hash-verified deployment receipt.

P0 is complete. Context 4096 remains an optional heap, bounds, and feasibility
check rather than a repeated performance benchmark. Numerical quality, formal,
cosim, OOC, and routed-build evidence remain continuous gates on later priorities.

### P1: bounded waste removal

- [x] Remove full padded-vocabulary work before argmax. Penzai's terminal greedy
  sampler now connects ARGMAX directly to the real logits for the proven
  single-sequence topology and retains PAD as a strict fallback.
- [x] Fuse RMSNorm and gamma multiplication in the PS fallback/reference path;
  its PL implementation belongs in the P2 section substrate.
- [x] Remove avoidable final-row result padding and copies.
- [x] Keep the current sequencer disabled; its measured A/B regressed decode.

P1a removed PAD from every characterized decode graph while retaining one ARGMAX
and a four-byte token download. Generated Q1 and Q2 output remained byte-identical
to the canonical P0 regression, and accounting closed in every sample:

| P1a metric | P0 | P1a | Change |
|---|---:|---:|---:|
| Q1 profiled device | 83.35 ms/token | 77.66 ms/token | -5.68 ms/token |
| Q2 profiled device | 104.87 ms/token | 99.49 ms/token | -5.38 ms/token |
| Q1 unprofiled steady | 99.00 ms/token | 95.95 ms/token | -3.05 ms/token |
| Q2 unprofiled steady | 118.42 ms/token | 118.88 ms/token | +0.45 ms/token |

The Q2 unprofiled wall result is within short-run variance, so P1a claims the
measured device-work reduction rather than a Q2 product-throughput improvement.
The sampler uses llama.cpp's experimental backend-sampling API, but topology
changes fail closed to the existing PAD path. The characterization and regression
artifacts are `20260805T031434Z-characterize-290c86efece2` and
`20260805T032338Z-regression-923832fe70a8`.

P1b fused each supported adjacent RMSNorm-gamma pair into one PS command. Decode
graphs fell from 650 to 537 commands: the 113 RMSNorm commands remain and the 113
gamma MUL commands disappear. Accounting closed in all samples, and profiled
device time improved consistently:

| P1b metric | P1a | P1b | Change |
|---|---:|---:|---:|
| Q1 profiled device | 77.66 ms/token | 74.68 ms/token | -2.98 ms/token |
| Q2 profiled device | 99.49 ms/token | 96.52 ms/token | -2.97 ms/token |
| Q1 unprofiled steady | 95.95 ms/token | 92.29 ms/token | -3.66 ms/token |
| Q2 unprofiled steady | 118.88 ms/token | 114.69 ms/token | -4.19 ms/token |

The unprofiled ranges overlap, so P1b claims the repeatable device-work reduction,
not a product-throughput gain. Q1/Q2 generated text remained byte-identical to P1a;
CPU logit comparisons had zero token mismatches and maximum absolute errors of
0.151340/0.136081. The characterization and regression artifacts are
`20260805T050508Z-characterize-c2598349015c` and
`20260805T052006Z-regression-b3f1211a2b6c`.

P1c made the single-column GEMM result length explicit in the v12 kernel contract.
The `151669x1x2048` vocabulary projection now emits exact final `TKEEP`/`TLAST`
and writes directly to its logical output. Its Q1/Q2 result-layout time fell from
3.2/3.3 ms to zero, while generated text and CPU logit comparisons remained
identical to P1b:

| P1c metric | P1b | P1c | Change |
|---|---:|---:|---:|
| Q1 profiled device | 74.68 ms/token | 71.46 ms/token | -3.22 ms/token |
| Q2 profiled device | 96.52 ms/token | 93.16 ms/token | -3.36 ms/token |
| Q1 unprofiled steady | 92.29 ms/token | 88.60 ms/token | -3.69 ms/token |
| Q2 unprofiled steady | 114.69 ms/token | 107.29 ms/token | -7.40 ms/token |

The profiled device ranges do not overlap. The unprofiled ranges still overlap,
so P1c claims the repeatable device-work reduction rather than a product-throughput
gain. Accounting closed in every profiled sample. The characterization and
regression artifacts are `20260805T111905Z-characterize-48513b8ea6c1` and
`20260805T112859Z-regression-21f7adb4ccf7`.

P1 is closed. Its changes remove about 12 ms/token of measured short-context
device work but cannot materially change the 533 ms context-2048 device time. Do
not add more primitive fusion, cross-graph activation caches, or PS-side command
extensions. Begin P2 with the explicit section/scratch contract and a
cycle-per-query-head/KV-update baseline for the unified attention engine.

### P2: section substrate and unified PL attention

- [x] Define the internal section/scratch contract: bank ownership, tile lifetime,
  layouts, bounds, and the external descriptor ownership boundary. Do not
  introduce a generic IR.
- [ ] Freeze concrete versioned FFN/attention descriptors with the first executable
  section, including DDR ranges, strides, RoPE, normalization, cache, and weight
  semantics.
- [x] Implement an exact PL FP32-to-Q8_0 quantizer matching the canonical host
  quantizer for every supported finite block.
- [x] Store and reuse the adjacent gate/up Q8 activation in GEMM's existing
  activation memories, guarded by an explicit epoch and shape.
- [ ] Reuse grouped Q/K/V activations inside the named attention section; do not
  force the observed graph ordering into a more general primitive matcher.
- [ ] Write GEMM results into banked scratch in the next consumer's layout.
- [ ] Make prefill token/row layout conversion a scratchpad addressing problem,
  rather than a PS transpose and DDR round trip.
- [x] Extend the existing online attention recurrence with a query-tile axis.
  Decode uses one query. The 64-slot implementation tiles four queries for up to
  16 heads and two queries for 17-32 heads, reusing K/V within each tile.
- [x] Pipeline single-query QK dot, online softmax update, and `p*V` accumulation
  across heads within each KV position, with a hard recurrence barrier between KV
  positions.
- [x] Make the softmax issue path truly II=1 and add alignment and burst tests for
  consecutive independent query/head updates.
- [x] Preserve FP32 softmax and output-accumulator state in the unified engine.
- [ ] Guarantee that newly appended K/V rows are visible to the same causal named
  attention section.
- [ ] Require strict shape, layout, RoPE, normalization, mask, and KV-cache matches;
  retain the current PS/op path as the correctness fallback.
- [ ] Qualify the PL quantizer with full-model board logits/perplexity and its
  saturation/status counters before extending it beyond the bounded group.

P2 must produce one attention engine and ABI, not separate prefill and decode
kernels. Completion requires PL execution for supported multi-token prefill,
lower backend downloads and command counts, faster 128/512-token prefill, and
faster attention at decode contexts 512 and 2048. Query tile size and any spatial
math replication are selected from OOC/routed timing and measured cycles per
query-head/KV update.

P2a froze contract v1 in `docs/p2-section-contract.md` and
`shared/section.zig`: a four-token tile, four fixed F32 scratch roles, native GEMM
Q8 storage with explicit epoch reuse, local new-K/V storage, KV-outer unified
attention, and strict pre-execution fallback. It also made cycles per valid and
processed query-head/KV update first-class profile and benchmark-artifact metrics
without changing the profile or execution ABI. Its 16-head decode baseline was
about 143-149 cycles per query-head/KV update; combined K/V starvation at context
2048 was only about 2.1%, confirming that compute/control scheduling, not DDR
width, was the immediate limit.

P2b replaces the sequential head loop with a head-streamed, within-KV schedule.
Tagged dot, score, softmax, and AXPY work overlap across heads, while a hard
barrier prevents the next KV position from starting until every accumulator
writeback for the current position has retired. Hardened cosim is bit-exact across
production, non-power-of-two, multi-token, bubbled-input, and backpressured-output
cases. Formal checks cover tag/order/address/GQA mapping, configuration snapshots,
the KV barrier, framing, stream accounting, and the current softmax issue spacing.

The context-512 board A/B isolates the scheduler improvement:

| Metric | Sequential | P2b | Change |
|---|---:|---:|---:|
| Cycles/valid query-head/KV update | 142.57 | 33.224 | 4.29x throughput |
| Flash kernel | 115.93 ms/token | 27.016 ms/token | -76.7% |
| Q1 device | 182.652 ms/token | 93.715 ms/token | -48.7% |
| Q2 device | 204.366 ms/token | 115.429 ms/token | -43.5% |

The post-repair context-2048 Q1 run sustained 9.126 million valid updates/s
(32.874 cycles/update). Flash fell from 451.20 to 102.14 ms/token and total device
time from 533.28 to 174.06 ms/token, improvements of 77.4% and 67.4%. Both final
board runs closed accounting, produced exact token counts, and reported zero K/V/O
stalls. The context-512 artifact is
`20260812T034043Z-characterize-dfc29ec25d66`.

The AXPY repair is bit-identical and splits the routed two-DSP multiply path. Its
hardened cosim, formal, OOC, and board gates pass. The registered-boundary OOC
result is 60 DSPs, 22,109 LUTs, 18,889 FFs, and +0.291 ns WNS at 3.333 ns
(328.7 MHz estimated Fmax). The clean combined route passes setup and hold at
+0.033/+0.007 ns after guarded placement and pre-route physical optimization.
Run `20260812T062923Z-b829dee03903-dirty-w512-p4-f300-clean` clears the 25 ps
development release floor but remains below the 50 ps headroom target. The
remaining 12-path tail spans the disabled sequencer, flash, and unrelated GEMM
structures, so no single narrow RTL repair is justified. The exact run was
promoted and deployed with a hash-verified receipt; Q1/Q2 logits checks had zero
token mismatches, and characterization artifact
`20260812T072252Z-characterize-ed0a92df72da` passed identity, exact-token,
accounting, PL-backend, and zero K/V/O-stall checks. P2b is closed. Activation
boundaries and P2 as a whole remain open; P2c status follows.

P2c implements the measured attention bottleneck. The softmax issue path is II=1,
and the primitive `flash_attn_f32` bridge gathers bounded Q/mask tiles, walks K/V
outermost, streams each K/V tile once, and scatters the output back to the existing
GGML layout. Decode still uses the direct single-query path. A 64-slot physical
state contract keeps Bonsai's 16-head tile at four queries while limiting 17-32
heads to two queries; the bitstream advertises this as flash engine
`0xF1A54A01`, version 1, `QUERY_SLOTS=64`, through capability schema 2. Unknown
IDs, versions, slot counts, shapes, strides, or masks fail closed to the PS path.

The hardened kernel and wrapper cosims preserve the exact query-one hashes and
cover four-query Bonsai tiles, two-query 32-head tiles, partial tiles, sparse and
all-masked rows, input bubbles, output backpressure, beat counts, and framing.
The aggregate RTL suite passes 26/26 targets; all 247 Zig tests pass. Compositional
formal checks cover the adaptive slot map at both maximum shapes, scheduler tags
and ordering, KV barriers, stream accounting, completion, liveness, and the added
BRAM-read pipeline boundaries.

The first fixed 128-slot tile-four implementation passed cosim, formal, and OOC
at +0.215 ns WNS, but used 33 BRAM tiles in flash alone. Its clean combined f300
route failed at -0.035 ns WNS with 124 failing setup endpoints, 69.5 total BRAM
tiles, and 98.91% CLB occupancy; it was not deployed. The adaptive 64-slot design
reduces the flash OOC result to 19 BRAM tiles, 22,657 LUTs, and 19,216 FFs while
retaining 60 DSPs and +0.215 ns WNS (320.7 MHz estimated Fmax).

The adaptive clean f300 route also failed closed: run
`20260812T144901Z-bbeac0c04eff-w512-p4-f300-clean` reached -0.088 ns WNS with
231 failing setup endpoints and 560 paths below 50 ps. No artifact was promoted.
One deliberate clock qualification then produced clean f285 run
`20260812T155038Z-3ef082b0fe4a-w512-p4-f285-clean`. It passes setup/hold at
+0.036/+0.010 ns, has five setup paths below 50 ps, and uses 80,837 LUTs, 95,229
FFs, 55.5 BRAM tiles, four URAMs, and 92 DSPs at 98.61% CLB occupancy. It clears
the 25 ps release floor but not the 50 ps headroom target, so the margin is valid
but thin. The exact image was promoted and deployed with a hash-verified receipt;
live capabilities report 284,997,152 Hz, engine `0xF1A54A01` version 1, and 64
query slots.

The first board prefill smoke truthfully fell back to PS and exposed an integration
error rather than hiding it: the eligibility predicate had reversed the two Q
strides. The supported GGML tensor is packed `[token][head][dimension]`, with
`q_nb1 = n_heads * head_row_bytes` and `q_nb2 = head_row_bytes`. Commit `0dc91c0`
corrects the predicate and tests the observed Bonsai strides; the RTL and deployed
bitstream did not change.

After that repair, bounded Q1/Q2 gates passed. The 32-token (`p32`) logit
comparisons had zero token mismatches and maximum absolute differences of
0.0982/0.1396. Artifact
`20260812T174257Z-p2c-repaired-p128` reports multi-token prefill as `pl/staged`
for all 224 calls and 896 kernel runs, 24.05 cycles per processed query-head/KV
update, zero K/V/O stalls, closed accounting, and 18.397/25.606 s Q1/Q2 prefill.
Against the same f285 image before the Q-layout eligibility repair, those walls
are down 24.51%/21.86%; flash command time fell from about 6.4 s to 0.691 s.
Artifact `20260812T174506Z-p2c-repaired-c512` reports all 896 prefill calls and
3,584 kernel runs on the same PL path at 21.222 cycles per processed update with
zero stalls. Q1/Q2 prefill fell from the P2b 160.300/185.434 s to
79.132/104.541 s, reductions of 50.63%/43.62%. Decode remains `pl/direct` at
33.315 cycles per update and
96.093/119.008 ms/token of device time. The small device-time increase from P2b's
93.715/115.429 ms/token is consistent with the qualified 285 MHz clock; the cycle
schedule is essentially unchanged.

The final scale artifact, `20260812T175007Z-p2c-repaired-c2048`, keeps every
multi-token attention call on `pl/staged`: 3,584 calls, 14,336 kernel runs, and
19,292,640,695/19,292,626,575 Q1/Q2 cycles. It sustains 20.494 cycles per processed
query-head/KV update with exact pairs and beats, zero K/V/O stalls, and closed
accounting. Q1/Q2 prefill is 425.994/813.215 s, 76.96%/57.79% below artifact
`20260804T211805Z-baseline` at 1848.895/1926.439 s. Decode remains `pl/direct` at
32.944 cycles per update, 180.560/203.465 ms/token of device time, and
193.748/221.127 ms/token steady wall. Against P0, decode device time falls
66.14%/63.33% and the flash scoreboard falls from about 459.1 to 115.7 ms/token.
Against P2b Q1 at f300, device time is 3.74% higher but cycles per update are only
0.21% higher, consistent with the f285 qualification clock. P2c is closed.

P2c is deliberately a primitive-op bridge. The context-2048 run still downloads
5.8/11.6 GiB for Q1/Q2 and retains the surrounding graph and command boundaries;
that traffic now dominates prefill wall and transport. It also does not yet provide
same-section K/V visibility. P2d now supplies PL Q8 ingress and bounded gate/up
activation reuse; banked-scratch GEMM output layouts and named section commands
remain. P2 as a whole stays open until those boundaries exist.

P2d adds wire ABI 14's fixed-arity `matmul_q1a8_group2`. The matcher is structural,
not semantic: it can group any adjacent pair with one identical F32 activation
range, matching shapes and weight format, safe compute flags, complete bindings,
disjoint destinations, no output views, and no intervening compute operation. In
the validated Qwen/Bonsai graphs, exactly 28 gate/up pairs satisfy that contract;
Q/K/V do not. Anything else retains the two primitive matmuls. All ranges are
validated before either projection can start; a pre-v13 bitstream executes two
validated primitive PL operations, with the shared-quantization PS implementation
as the final fallback.

GEMM v13 receives each column tile as raw FP32, quantizes it once with exact
canonical FP32 division/multiplication and round-to-nearest-even behavior, and
latches the native Q8 bytes plus F16 scales under `{epoch, K, columns}`. The second
projection must match that state and consumes zero activation beats. Framing,
non-finite, scale, or arithmetic faults abort raw ingress and invalidate resident
state before the kernel consumes weights or emits results. A bad reuse epoch or
shape also rejects before kernel weight/result consumption, but preserves the
previous resident record. DMA movers may already be armed on either error path.
Fake-device full-model Q1 and Q2 runs reduced one-token prefill and decode graphs
from 537 to 509 commands and reported exactly 28 grouped operations per graph.
Inspection identifies those operations as gate/up; this validates structural graph
recognition and accounting, not semantic matching or board throughput.

The exact quantizer cosim covers 1,032 blocks, binary and ternary integration
cosims cover raw-load/reuse behavior, and focused formal tasks cover invalid reuse
and activation abort. Standard Zig tests pass 255/255, pinned llama-enabled tests
278/278, and the aggregate RTL suite passes 36/36 build steps. The initial 3.333 ns
isolated probes passed at +0.192 ns WNS for the 792-LUT/908-FF/2-DSP quantizer leaf
and +0.619 ns for the 38,544-LUT/42,935-FF/32-DSP GEMM v13 core.

Those isolated results did not predict the first combined route. Clean f285 run
`20260812T212725Z-f9e1ca83f8ae-w512-p4-f285-clean` routed fully but failed release
at -0.215/+0.010 ns setup/hold, with 142 failing setup paths and 580 below the
50 ps headroom target. It used 81,284 LUTs, 96,220 FFs, 55.5 BRAM tiles, four
URAMs, and 95 DSPs at 99.32% CLB occupancy. No image from that run was promoted.

Timing-repair commit `547d87b` replaces the flat ingress block-count multiply and
wide counters with nested column/Q1/sub-block state, registers the scalar boundary,
uses a compact 64-bit emitter, and retimes the GEMM `FE_LAT` exponent addition.
It removes 228 state bits and one DSP while preserving the established binary and
ternary hashes and cycle counts; the expanded cosim also crosses multiple column
and Q1 boundaries and checks missing terminal framing. Current-source full-decode
OOC run `20260812T223110Z-547d87b12094-full-decode` passes the 3.333 ns constraint
at +0.245/+0.039 ns setup/hold with 40,141 LUTs, 44,912 FFs, 775 CARRY8s,
34 DSPs, two BRAM tiles, and four URAMs. The failed route's ingress bookkeeping
and exponent-add path families are absent from its reported critical paths. The
3.333 ns production artifacts pass; the probe driver exited later when its read-
only, non-production second-period report script attempted to mutate the open
design. That report failure does not affect the production source or result.

Replacement clean f285 run
`20260812T224303Z-547d87b12094-w512-p4-f285-clean` is a clean commit build with
source bundle
`1cfc1e173ba0ae1d06d1fceb1d3fa83ec29535f760a7a7f91a6b5e0458078249`.
It is fully routed with clean structural constraint counts and exact restoration of
the 75 ps placement guardband. Setup/hold pass at +0.043/+0.010 ns with no negative
paths; 4/55/622 setup paths are below 50/100/200 ps. The 25 ps release floor is met,
but the 50 ps headroom target is missed by 7 ps. Routed use is 81,887 LUTs, 95,699
FFs, 1,408 CARRY8s, 94 DSPs, 55.5 BRAM tiles, and four URAMs, with
14,519/14,640 CLBs occupied (99.17%). Methodology still reports two critical
findings, TIMING-2 and TIMING-4, plus six warnings: five TIMING-28 and one ULMTCS-1.

The image was promoted and deployed with bitstream SHA-256
`9ab576cad24eb3c77d6b55200d5e9a08d92f0197625f76d479c13fc6fa82a70f`.
Live capabilities report schema 2, wire ABI 14, profile ABI 6, GEMM
`0xB05A2000` v13 at 284,997,152 Hz, and flash v1 with 64 query slots. Q1 `p32`
logits pass with 0.098204 maximum absolute error and zero token mismatches. Q2
`p32` also passes: its two step differences are 0.089185/0.131115, maximum
absolute error is 0.131115, both argmax results are exact, token mismatches are
zero, and `check=ok`.

Final board artifact `20260812T235644Z-characterize-9bb6d3eb522f` contains all
six expected Q1/Q2 p128, c0, and c512 samples. It is complete and run-validated,
every sample closes accounting, and start/end capability responses are identical.
The full-model structure is exact: p128 prefill emits 15 graphs, 4,002 commands,
217 grouped and 1,114 primitive matmuls, followed by one 509-command decode graph
with 28/141 grouped/primitive matmuls. The c0 decode emits 64 graphs, 32,576
commands, and 1,792/9,024 grouped/primitive matmuls, after its own one-graph,
509-command, 28/141 grouped/primitive prefill. The c512 prefill emits 63 graphs,
15,978 commands, and 865/4,450 grouped/primitive matmuls; its decode matches c0
exactly.

Every group executes on PL: 16-column groups use `pl/staged`, one-column tails and
decode use `pl/direct`, and no group falls back. Each group raw-loads its activation
once and reuses it with zero activation beats. The c512 decode
`6144x1x2048` bucket records 1,792 calls, 3,584 kernel runs, 1,835,008 A beats,
11,010,048 R beats, and 110,100,480/198,180,864 Q1/Q2 W beats, with zero host
quantize/pack time. P128's staged main bucket records 216 calls, 864 runs, and
3,538,944 A beats plus one direct two-run/1,024-beat tail. C512 records 864 staged
calls, 3,456 runs, and 14,155,776 A beats plus the same direct tail.

Attention remains the exact P2c implementation and uses the expected staged/direct
PL paths with zero K/V/O stalls; matched cycle deltas are at most 0.01%. The
approximate-flash verify advisory fired across 12 smoke calls for 40 of 24,576
values, with maximum absolute/normalized differences of 0.5542/0.0467. This is an
expected diagnostic from the unchanged approximate flash path, not a P2d failure:
model logits pass, and there are no GEMM, group, Q8, DMA, kernel, or request errors.

The single-repeat device-time check shows no regression; all four matched P2c
observations are lower. P128 Q1/Q2 changes
73.282->73.028 ms/token (-0.35%) and 96.143->94.659 (-1.54%); c512 changes
96.093->94.773 (-1.37%) and 119.008->117.700 (-1.10%). The new c0 anchor is
65.716/88.615 ms/token. This occurs even though raw PL quantization raises the
grouped kernel's cycles versus two primitive gate/up calls by 25.98%/21.87% for
Q1/Q2 staged columns-16 and 18.28%/8.96% for c512 decode. Fewer commands and zero
grouped host quantize/pack work are consistent with offsetting that local cost;
this one-repeat gate does not establish a repeatable speedup.

P128 prefill wall is 47.259/91.355 s and c512 is 224.114/411.642 s. The same Q1
p128 328.9 MiB download changed from 8.1 s at 40.5 MiB/s to 36.5 s at 9.0 MiB/s;
Q2's 655.9 MiB changed from 15.3 s at 43.0 MiB/s to 79.9 s at 8.2 MiB/s. At c512,
Q1's 1.4 GiB changed from 35.0 s at 41.6 MiB/s to 176.5 s at 8.3 MiB/s, and Q2's
2.8 GiB from 58.5 s at 49.7 MiB/s to 362.4 s at 8.0 MiB/s. The data volume is
unchanged; this network/transport regression is not attributed to P2d.

P2d is closed at this bounded f285 structural, numerical, and device-regression
qualification. P2 as a whole remains open: grouped outputs still return through
DDR and no named FFN or attention section command exists.

The grouped command is atomic only at the scheduling and preflight boundary. A
hardware error after the first destination is written does not roll it back, and
the runtime returns an error rather than retrying. Armed DMA channels receive
best-effort resets on every error path, but a DMA timeout whose reset never clears
still requires daemon or bitstream recovery. With P2d closed, focus moves to writing
gate/up results directly into their consumer's banked-scratch layout, then using
that substrate for the first named FFN section.

### P3: FFN section

The first named pipeline should be FFN because it avoids attention and KV-cache
ordering while exercising the new scratch and Q8 contracts:

`residual -> norm*gamma -> Q8 -> gate/up GEMM -> SwiGLU -> Q8 -> down GEMM -> residual`

Keep gate/up outputs in separate scratch banks so SwiGLU can consume them without
DDR materialization. The section lowerer must require recognized shapes, strides,
activation, normalization parameters, and resident weight formats; otherwise it
must use the existing op path. This is an integration proof for P2, not a reason
to postpone the P2 attention engine.

### P4: attention section

`residual -> norm*gamma -> Q8 -> Q/K/V GEMM -> RoPE -> KV append -> tiled attention -> output GEMM -> residual`

- Preserve FP32 attention dot/softmax state, output accumulation, and residuals.
- BF16 `p*V` with FP32 accumulation remains a reasonable implementation split.
- Guarantee that the newly appended K/V entry is visible to the same causal
  attention section.
- Schedule weight and historical-KV DDR traffic explicitly before attempting overlap.
- Keep banked scratch between stages; do not build one device-wide combinational
  ready/valid chain.

### P5: ternary and KV memory efficiency

The current resident ternary layout uses 576 bytes for 16 x 128 weights, or 2.25
physical bits per weight including scale/padding structure. The measured kernel
streams about 484 MB/token and has an approximately 23 token/s bandwidth roof.

- [ ] Define the exact supported ternary scale semantics independently of GGML names.
- [ ] Evaluate denser trit packing only after P2-P4 make weight traffic dominant.
- [ ] Add a lossless source-to-resident adapter and exhaustive selector/decoder tests.
- [ ] Add f16 or quantized KV options for large-model/long-context operation.

The practical heap is about 1.5 GiB. A binary Bonsai-8B file is roughly 1.15 GB and
a 4K f16 KV cache is roughly another 576 MiB before compute buffers, so memory work
is mandatory for that operating point regardless of available PL logic.

### P6: control, DMA, and transport

The current combined design has ten AXI DMA engines and a 13-target control
SmartConnect. Do not immediately rewrite it: first reduce hundreds of operation
commands to a few section commands.

- Consolidate activation/result/KV movement behind a layer controller.
- Retain four weight streams unless measured bandwidth says otherwise.
- Reconsider a slower control clock only after simplifying the control topology;
  doing it now would reintroduce CDC complexity that the single-clock build removed.
- Re-measure TCP encoding, transfer, and response overhead after the command count falls.

### P7: spatial scaling, deferred

Do not build a conventional systolic array now. Decode still has to read every weight,
and prefill first needs to eliminate activation/result overhead. If fused, tiled
prefill or routine batched/speculative decode later becomes compute-bound, evaluate
two or four spatial columns before a larger mesh.

## Continuous verification gates

These are requirements on every priority, not a separate final cleanup phase:

- `zig build test` and all formal-control targets pass;
- one aggregate RTL cosim target runs in Nix/CI for binary and ternary paths;
- changed kernels pass OOC for resource/Fmax feedback;
- the combined routed build clears the 25 ps development floor, while 50 ps remains
  the tracked headroom target;
- Q1 and Q2 model-level logits/perplexity gates pass at short and long contexts;
- a board smoke run verifies bitstream identity, capabilities, DMA/cache behavior,
  prefill, and decode.

The aggregate `zig build test-rtl` target now covers binary and ternary GEMM,
the exact Q8 quantizer and grouped activation path, plus the hardened flash kernel
and wrapper cosims. The committed P2d suite passes all 36 build steps locally.

Formal verification should target control snapshots, bounds, handshakes, TLAST/TKEEP,
DMA safety, and liveness under explicit fairness assumptions. Approximate floating
point behavior belongs in cosim and model-level quality tests.

## Current timing notes

Historical accumulator reset and broad-enable problems were real and the fixes should
remain. They are not currently independent roadmap projects:

- accumulator data reset was removed and enable decoding was localized;
- carry-save banks removed the wide carry-propagate recurrence;
- the read-column pair is registered before its final add;
- the DSP multiply output is already registered before the 104-bit shift result.

The qualified P2d build is the repaired clean f285 run described above. It reports
+0.043 ns setup and +0.010 ns hold slack, with four setup paths below the 50 ps
target; P2c remains its direct performance comparison baseline. The fixed and
adaptive P2c designs both failed clean f300 routing despite passing OOC, and the
first P2d f285 design similarly failed before repair. This is direct evidence that
combined routing, not OOC Fmax, remains the release authority. Methodology still
reports critical TIMING-2/TIMING-4 and five TIMING-28 plus one ULMTCS-1 warning.
Therefore:

- keep 25 ps as the hard development release floor and 50 ps as the headroom target;
- treat f285 as P2d's bounded qualification point, not proof of f300 closure;
- resolve the methodology findings before defining a stricter production budget;
- add per-port starvation counters before adding elastic weight FIFOs;
- treat timing locality and reproducible headroom as build requirements for every phase.

## General BitNet support gate

Bonsai validates a Qwen3-shaped graph using Q1_0 or this fork's group-64 Q2_0. It does
not establish general BitNet compatibility. A model is supported only when all of the
following are explicitly accepted:

- weight code and scale semantics;
- graph operations and activation family;
- shapes, strides, head counts, head dimensions, and context limits;
- RoPE, normalization, mask, and KV-cache semantics;
- memory capacity and numerical-quality gates.

Add a compatibility-only load/census mode that reports rejected nodes and parameters.
Official BitNet formats or squared-ReLU/sub-layer-normalized graphs require explicit
format and section-lowering work; they must not be treated as Q2_0 aliases.
