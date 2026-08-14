# Penzai inference architecture roadmap

Status: 2026-08-13. This document records the performance and offload architecture
for the KR260 implementation. Its early measurements start from the historical
deployed f300 bitstream and one Bonsai-1.7B run; P0-P2 results that supersede that
baseline are summarized in `docs/accelerator-priorities.md`.

The P0-P2 delivery text below is retained as design history. The forward P3-P8
ordering is current and matches `docs/accelerator-priorities.md`, which remains the
short authoritative tracker.

## Executive decision

The accelerator does not need a wider GEMM array or a replacement matrix engine.
During decode, the GEMM kernel streams 256.3 MiB of packed weights per token at
12.69 GB/s and reaches 93.1% array utilization. Its approximately 21.1 ms/token
kernel time is the current physical floor. The other approximately 60 ms/token
of device time is mostly repeated activation quantization, tensor layout
conversion, and standalone fp32 vector passes on the Cortex-A53.

The destination architecture is therefore:

1. Keep the existing output-stationary GEMM and native-layout online attention.
2. Make Q8 quantized activations an internal PL data type, not a PS-generated DMA
   staging format.
3. Replace the generic GEMM/scratch FFN schedule with a fixed streaming
   superkernel, and execute attention as a strict named section.
4. Keep the residual banked on chip across both sublayers and ultimately through
   the fixed 28-layer walk. Keep weights and historical KV in DDR, streaming each
   only when the model operation requires it.
5. Replace repeated DMA setup with a persistent layer controller after the named
   sections and resident execution have fixed the real movement schedule.
6. Treat DSP/BRAM/URAM mapping and removal of superseded LUT-heavy logic as a
   requirement in every architectural increment.
7. Optimize dense exact attention first. Introduce KV compression or approximate
   sparsity only when real context lengths make KV traffic comparable to weights.
8. Preserve one bitstream and one downstream datapath for binary and ternary
   weights; only packing and the GEMM front end depend on weight format.

At the present binary weight density and measured bandwidth, the absolute
weight-stream roof is about 47 tokens/s. A realistic first destination is
26-30 device tokens/s after removing PS passes, with wall throughput depending
on the transport work described below. The implemented issue-ordered Q2_0 layout
uses 36 resident bytes per 128 weights versus binary's 20, including scales and
alignment. That 1.8x traffic ratio puts the ternary weight-stream roof near
26 tokens/s before later fusion work. Every proposed fusion must improve both
formats without assuming binary's higher roof.

## Measurement baseline

### Reproduction contract

The baseline is:

- Model: `Bonsai-1.7B-Q1_0`, 256.8 MiB uploaded resident representation.
- Device: KR260, XRT heap, combined matmul + flash bitstream.
- Bitstream: `w512-p4-f300`, kernel version 10, sequencer disabled.
- Prompt: `"hello"` through the chat template, producing 13 prefill tokens.
- Generation: 25 greedy tokens with backend sampling and aggregate profiling.
- Correctness verification: disabled for the performance run.

The measured headline is:

| Metric | Value |
|---|---:|
| Decode wall | 90.2 ms/token, 11.08 tokens/s |
| Decode device | 81.2 ms/token |
| Decode transport | 7.6 ms/token |
| Other host residual | about 1.4 ms/token |
| Prefill wall/device | 1.1 s / 1.0 s for 13 tokens |
| First decode step | 89.0 ms |

The last row is not user-visible time to first token by itself. The current
profile labels the first post-prefill decode step as TTFT. User-visible TTFT is
approximately prefill wall time plus that first step, about 1.2 seconds in this
run. The profiler should report both quantities with unambiguous names.

### Decode budget reconstructed

The stable decode `pl seg` windows report approximately 67 us quantization,
4.9 us sync-to-device, 2.7 us setup, 109.7 us wait, 5.4 us sync-from-device,
and 16.7 us result copy per matmul call. There are 197 matmuls/token. The kernel
counters account for about 107 us of the wait.

| Component | ms/token | Basis |
|---|---:|---|
| GEMM kernel busy | 21.1 | 107 us x 197 |
| Activation quantization | 13.2 | 67 us x 197 |
| GEMM result layout copy | 3.3 | 16.7 us x 197 |
| GEMM cache sync | 2.0 | (4.9 + 5.4) us x 197 |
| GEMM setup and wait overhead | 1.1 | setup plus wait minus kernel |
| Flash attention | 7.4 | aggregate op profile |
| RMSNorm + gamma multiply | 12.4 | 6.3 + 6.1 |
| SwiGLU | 7.0 | aggregate op profile |
| Sampling pad + argmax | 6.3 | 4.1 + 2.2 |
| Residual adds | 3.3 | aggregate op profile |
| RoPE | 2.4 | aggregate op profile |
| KV cache row writes | 0.9 | aggregate op profile |
| Remaining ops/runtime | about 0.8 | device total minus above |
| **Device total** | **81.2** | measured |

The `copy=16.7 us/call` average does not imply that all 197 decode results are
copied. The driver already DMAs directly for one-column outputs whose row count
is divisible by 16. The likely remaining cost is one large, partial-final-row
projection such as the tied LM head. Add per-shape segment attribution before
changing RTL, then make the last partial row DMA-safe with a logical row count
and correct final-beat `TKEEP`.

The first `pl seg` line in the supplied server log is not a decode measurement.
Its 838 us quantization and 1.335 ms copy averages cover the preceding multi-token
prefill calls before the 200-call reporting window crosses into decode. Phase and
shape must be attached to future segment reports so these windows cannot be
misread.

### GEMM roofline

Decode counters report 104,976,000 64-byte weight beats over 25 tokens:

```text
weight bytes/token = 104,976,000 * 64 / 25 = 268,738,560 B = 256.3 MiB
kernel floor        = 268,738,560 / 12.69e9  = 21.2 ms/token
weight-only roof    = 1 / 0.0212             = 47.2 tokens/s
```

Weight stalls are 6.9%. Eliminating every stall would improve the kernel only by
about 7%; widening the array cannot remove the 256.3 MiB/token read. GEMM work
should target its input/output boundaries and ternary selector seam, not more MACs.

Prefill is different: 13 columns reuse each weight beat, the array reaches 99.9%
utilization, and kernel-busy time is only about 157 ms of the 599 ms matmul total.
The current prefill matmul boundary is approximately:

| Component | Approximate time/prompt |
|---|---:|
| GEMM kernel busy | 155-157 ms |
| Activation quantization | 165 ms |
| Result gather/transposition | 263 ms |
| Sync, setup, and other wrapper work | 10-20 ms |

The result stream is `[rowblock][column][row]`, while ggml expects column-major
output. Decode normally avoids this transpose because it has one column; prefill
cannot. Removing the prefill transpose is at least as important as PL-side
quantization for TTFT.

### Attention facts

The flash op's profile row reports `n_kv=256`, but that is the padded command
shape. The PL driver scans the causal mask, replaces the kernel's extent with the
real finite prefix, and shortens all K/V/mask DMAs. Consequently:

- The reported 3,853.8 MiB/s effective decode bandwidth is an operand-span
  estimate, not measured flash DMA bandwidth.
- The increasing 42.7k, 59.0k, and 75.2k cycle windows reflect the growing real
  context even though the report continues to print 256.
- Zero K/V/O stalls show that the current short-context kernel is compute/control
  bound, not DDR bound.
- The RTL still waits through several numeric operations in a sequential FSM;
  sparse attention cannot fix that fixed per-head/per-KV schedule.

At this model shape, one KV position costs:

```text
per layer = 2 (K,V) * 8 KV heads * 128 values * 2 B = 4,096 B
28 layers = 114,688 B per context position per generated token
```

KV traffic equals the current 268.7 MB weight stream at a context of roughly
2,344 tokens. The present benchmark covers positions about 13-37, where dense KV
traffic is small. Long-context compression is important, but not the current
11 tokens/s limiter.

### Implementation capacity

The historical routed f300 baseline used 65.47% LUT, 38.76% FF, 33.68% BRAM,
6.25% URAM, and 7.37% DSP. P2's section substrate and unified attention engine
changed the limiting resource: the qualified P2f image uses 83,463 LUTs, 98,323
FFs, 58 BRAM tiles, 20 URAMs, and 98 DSPs, with roughly 99.6% of CLBs occupied.
It passes f285 at +0.015/+0.010 ns setup/hold.

The design is now CLB/routing constrained while substantial DSP and block-memory
capacity remains. Prefer BRAM/URAM queues and state plus DSP-shaped arithmetic,
and remove each superseded LUT-heavy path. Use leaf and integrated OOC for local
feedback, with a clean combined route at coherent architectural milestones rather
than every RTL edit. An ASIC-style shared Booth datapath or a 52k-LUT ternary
lookup engine does not match this resource balance.

## Target execution model

The unit of optimization is a transformer sublayer, not an individual ggml node
and not a general device-side graph scheduler.

### Attention sublayer

```text
residual/input
  -> RMSNorm * gamma -> one Q8 activation
  -> grouped Q/K/V projections
  -> Q/K per-head norm + RoPE
  -> K/V append
  -> dense online attention
  -> Q8 quantization -> output projection
  -> residual add
```

Q/K/V share one normalized activation and therefore one quantization. The Q/K
normalizations and RoPE should join the attention ingress rather than remain
standalone fp32 DRAM passes. For Bonsai GQA, the natural streaming unit is one KV
head and its two query heads, not an arbitrary single query head.

### FFN sublayer

```text
residual/input
  -> PL RMSNorm * gamma -> one Q8 activation
  -> paired UP/GATE output blocks
  -> immediate SwiGLU and exact Q8
  -> ping-pong DOWN-input RAM -> down projection
  -> PL residual add -> resident residual
```

The FFN should be one fixed streaming machine, not three generic GEMMs separated
by complete F32 scratch tensors. Pair UP/GATE work in output-row chunks, consume
each pair through SwiGLU immediately, and write only compact canonical Q8 blocks
for DOWN. Ping-pong buffering decouples the exact quantizer from the DOWN reader.
The largest Bonsai intermediate is small relative to available BRAM/URAM; use that
capacity to reduce LUT muxing and DDR traffic.

### Internal contracts

- External graph tensors remain the ggml-required f32/f16 formats.
- The PL-internal activation contract is Q8_0: 32 int8 values plus one f16 scale.
- Weight format is independently tagged as binary or ternary.
- Fused operations are a short, explicit wire enum with concrete tensor ranges;
  do not introduce a generic device IR.
- The existing PS kernels remain the golden oracles and fallback paths.

## Delivery plan

Each phase has a measurement gate. Savings from later phases replace earlier
stepping-stone savings; they must not be added twice.

### Phase 0: make the profile truthful

Do this before another performance change.

1. Move matmul segment counters into the structured profile. Split by phase and
   at least `{rows, cols, k}` shape; report kernel busy separately from wait.
2. Extend flash statistics with actual `kv_hi`, actual DMA bytes, kernel cycles,
   and K/V/O stalls. Do not derive bandwidth from padded tensor spans.
3. Report run-graph command bytes, preload bytes, preload application time,
   command decode time, execution time, and response encode time. The current
   7.6 ms/token `transport` bucket cannot distinguish TCP from device-side preload
   and protocol work.
4. Rename the current 89 ms metric to `first_decode_step`; add user TTFT as
   `prefill wall + first decode step`.
5. Establish profiled and unprofiled A/B baselines. Aggregate profiling reads the
   clock around every one of 652 commands and must not silently become the product
   throughput number.
6. Use a fixed benchmark set: 13-token prompt/25-token decode, 64-token prefill,
   and decode at several real KV lengths. Run at least three repetitions and
   report median plus range.

Gate: the structured totals reconcile to device time within 1%, flash reports the
same real extent as its DMA lengths, and profiled/unprofiled overhead is known.

### Phase 1: bounded wins on the current architecture

These changes reduce work without inventing a new datapath.

1. **Fold sampling pad into argmax.** Recognize the `pad -> argmax` graph pattern,
   read only the real logits rows, and synthesize any required dummy-row result.
   Preserve ggml's last-maximum tie rule. Expected ceiling: 6.3 ms/token.
2. **Group projections that share activations.** Introduce explicit grouped Q/K/V
   and gate/up commands. Quantize and pack their shared f32 input once, then launch
   the existing GEMM kernel for each weight/destination. This avoids up to 84 of
   197 quantizations/token, approximately 5.6 ms/token and about 70 ms for the
   measured prefill. Do not use a cross-graph cache keyed only by a reused tensor
   address.
3. **Fuse RMSNorm and gamma on the PS.** One reduction pass plus one normalized
   multiply/write pass replaces RMSNorm output followed by a separate multiply.
   This is both an immediate A53 win and the oracle for the later PL prologue.
   Measure rather than assuming the full 6.1 ms multiply disappears.
4. **Make partial final rows DMA-direct.** Attribute the remaining copy to shapes,
   pass the logical row count to the kernel, set final-beat `TKEEP`, and DMA exactly
   `rows * 4` bytes for one-column output. Expected ceiling: 3.3 ms/token if the
   shape attribution confirms the LM-head hypothesis.
5. Keep the sequencer disabled. Current register setup is about 0.5 ms/token in
   total; batching it cannot repay added machinery at this phase.

Gate: identical greedy tokens and acceptable full-logit error; each change gets an
independent on/off A/B. A reasonable Phase-1 device target is 60-65 ms/token, not
the sum of every optimistic ceiling.

### Phase 2: rebuild the GEMM boundaries

1. **Raw-f32 activation ingress.** DMA the resident f32 source directly to a PL
   Q8_0 quantizer. For each 32-value block, reproduce finite-value absmax, f16 scale
   rounding, and round-to-nearest-even int8 conversion. Buffer one block or one
   vector as required; packed activations then enter the unchanged GEMM core.
2. Treat exact agreement with `shared/layout.zig` as the first goal. If an
   approximate reciprocal is required, model its rounding in the reference and
   gate it with full-model logits and perplexity, not sampled-token equality alone.
3. Preserve grouped projection reuse: norm/quant once, consume the same packed
   activation for all Q/K/V or gate/up projections.
4. **Column-major prefill output.** Accumulate rowblock results into a
   column-major tile buffer, preferably URAM, and emit the completed group directly
   to the destination. For hot shapes whose rows are divisible by 16, this removes
   the PS transpose entirely. Retain a fallback or valid-lane metadata for generic
   partial rows.
5. Keep `COLS_MAX=8` initially. A 13-token prompt already streams weights only
   about 1.8 times, and doubling the accumulator columns worsens the current
   read-mux/timing problem. Revisit only after the direct-output build is measured.

Gate: matmul cosim against the canonical quantizer/output layout; no regression in
decode weight bandwidth; routed WNS nonnegative; prefill result-copy time near zero.
The direct Phase-2 prize is the remaining 13.2 ms/token quantization class and
roughly 428 ms of measured prefill quantization plus transpose, less the portions
already removed by Phase 1.

### Ternary vertical: land after the GEMM boundary is stable

Ternary is a format vertical, not a second accelerator:

1. Retain both upstream group-64 scales and reorder the two-bit Q2_0 codes into
   the GEMM issue sequence during upload.
2. Stream one dual-scale beat and two code beats per 32-weight sub-block; select
   `{nonzero, sign}` directly without buffering or decoding a full block.
3. Reuse the Q8 ingress, fixed-point accumulator, output path, flash kernel, and
   all layer fusions unchanged.
4. Maintain separate per-format counters and roofline reporting.

Do not import table-lookup arithmetic solely because it is ternary. The current
FPGA is LUT/timing constrained and DSP/BRAM rich; direct zero/sign selection is a
better fit unless a same-device synthesis A/B proves otherwise.

Gate: packed-layout round trip, bit-exact ternary dot products against the software
oracle, routed single bitstream, and a measured physical roof that agrees with the
actual packed ternary bytes including scales and alignment.

### P3: streaming FFN and local hard-block rebalance

P2f proved a strict named FFN command, exact failure boundary, scratch contract,
and full-model correctness. Its profiled device observations improved, but its
unprofiled c0 medians regressed by about 2.8%. P3 replaces the remaining generic
GEMM-plus-scratch schedule in measured increments:

1. **P3a, fixed-cost attribution and removal (complete).** Split
   matching/preflight, PS norm, synchronization, setup, kernel, scratch feed,
   gather, residual, and publication time. Remove avoidable host matching/runtime
   work and bypass DOWN gathering for naturally direct one-token layouts. Gate
   with pinned census, numerical tests, exact fallback behavior, and a focused
   same-image A/B; product wall performance remains a later P3 gate. This step
   should require no Vivado build when RTL is unchanged.
   Commit `cbb621b` implements graph-fact caching and direct canonical one-token
   publication. The same-image summary has SHA-256
   `e92232efc1d74bcad42f279ed78e25d679feda8f7801796648f8fd2327cbe55e`:
   Q1/Q2 DOWN layout totals fall from 78.1/77.9 ms to zero and device time falls
   0.42/0.54 ms/token. A host-scheduling-inclusive matcher proxy falls
   1.26/1.12 ms/token directionally; this is not a direct matcher timer or a wall
   performance claim. Quality and accounting remain exact.
2. **P3b, DOWN feeder repair (complete).** Prefetch scratch, insert ping-pong
   block buffers, and pipeline or replicate exact Q8 block conversion enough to
   keep DOWN fed. Add producer-empty and consumer-stall counters. Gate with
   hardened cosim/formal, exact Q8 behavior, utilization counters, integrated
   OOC, a clean combined route, and repeated board A/B.
   Commit `813fd87` passes integrated OOC at +0.190/+0.040 ns and clean route
   `20260813T200433Z-813fd87ce299-w512-p4-f285-clean` at +0.008/+0.010 ns. The
   deployed-bit SHA-256 is
   `9329e56b858505828bccf048d50e74443eac73cbf1dcfcea33db349c93901ebe`.
   Board Q1 changes are DOWN cycles -10.064%, DOWN MAC/cycle +11.190%, FFN
   -3.802%, device -1.652%, and unprofiled -2.758%; Q2 changes are -7.017%,
   +7.546%, -2.494%, -1.190%, and -0.752% in the same order. All quality and
   accounting gates pass.
3. **P3c, streaming superkernel (complete).** Retain the complete F32 UP result in
   X1 while streaming GATE through the packer and pairer, SwiGLU, exact Q8, and the
   compact Q8 buffer bank 0 for DOWN replay. This removes the complete F32 GATE
   tensor and post-hoc X0/X1 rescan without claiming ping-pong integration. Gate
   with per-layer differential tests, exact stream accounting, resource deltas,
   integrated OOC, a clean combined route, and bounded same-software board A/B.
   Commits `6957b1b` and `ee12c8a` integrate and time-harden this v16 path.
   Production-strategy OOC run
   `20260814T041041Z-ee12c8a2826e-p3c-prod-strategy-full-decode` passes nominal
   setup/hold at +0.086/+0.057 ns at 3.333 ns and guarded setup at +0.011 ns, with
   24 URAMs, 38 DSPs, 10 BRAM36s, two BRAM18s, and 94 LUTRAMs. Clean f285 run
   `20260814T043858Z-ee12c8a2826e-w512-p4-f285-clean` passes at +0.021/+0.010 ns;
   its `bit.bin` SHA-256 is
   `1a4d8455a1217ed28da58e6d250e24ddc6b6f3566e2952d24ec952ffd45a1179`.
   Exact logits, capabilities, and accounting pass. Profiled x3 Q1
   wall/device/FFN/DOWN cycles/DOWN MAC per cycle changes are
   -8.016%/-7.320%/-17.391%/-53.312%/+114.188%; Q2 changes are
   -6.401%/-5.812%/-12.276%/-35.965%/+56.165%. Primary unprofiled Q1 x3 was a
   near-tie at +0.218%; its predeclared order-balanced schema-4 follow-up passes
   every gate at C1/B2 -4.169%, C4/B3 -1.545%, and pooled C10/B10 -3.428%.
   Ranges overlap, so Q1 is directional only. Q2 unprofiled x3 changes -5.508%
   with disjoint ranges. The consolidated summary
   `/tmp/p3c-board-qualification-20260814T053441Z/qualification-summary.json`
   has SHA-256 `b8ac70157f85658349fbcc7c793e9be4b477b85d14ac1d981285a90d6679ae4f`;
   its 163-entry evidence ledger has SHA-256
   `5569dd0cc15941de566c34c19233710d73e217f4a73b1388b3a76da73c21b9f9`
   and passes an independent full rehash. v16 DOWN's 960 activation beats per
   FFN command, or 1,720,320 per qualification decode, are local Q8-to-GEMM
   replay beats rather than external traffic or a regression from P3b resident reuse.
4. **P3d, PL norm and residual (active).** Add weighted RMSNorm-to-Q8 and residual
   addition in PL, populate the resident residual role, and keep the PS
   implementation as oracle and strict pre-start fallback. Map reduction state and
   queues to BRAM/URAM and suitable arithmetic to DSPs, removing replaced LUT/PS
   machinery.
   Gate with arithmetic cosim, formal control, full-model quality, resources, and
   a clean combined route.
5. **P3e, prefill scale and qualification.** Evaluate an eight-token tile against
   the current four-token tile using the existing eight-column GEMM capacity.
   Retain four if routing or throughput does not repay added scratch. Remove
   superseded debug datapaths and run the complete Q1/Q2 characterization and
   regression suite on the final clean image.

P3a-P3c are complete, P3d is active, and P3e has not started. P3 closes only when
repeated same-image unprofiled Q1 and Q2 runs beat the legacy operation path. The
strict matcher, fail-before-start fallback, one-command profile, and full
numerical/accounting contract remain unchanged.

### P4: named attention section

P2 already supplied the pipelined KV-major engine and adaptive query-blocked
prefill. P4 closes the boundaries around that engine:

1. Freeze and strictly match the concrete attention descriptor, including layout,
   normalization, RoPE, mask, cache extent, strides, and alias/range safety.
2. Produce one normalized Q8 activation for Q/K/V; attach Q/K normalization and
   RoPE to attention ingress rather than materializing them in DDR.
3. Append new K/V locally and guarantee that those rows are visible to the same
   causal section before invoking the existing tiled attention schedule.
4. Quantize attention output for the output projection and produce the residual
   without publishing intermediate Q/K/V or attention tensors.
5. Schedule weight and historical-KV traffic around the shared DDR controller and
   prove the schedule using actual bytes, extent, and starvation counters.

Gate: strict fallback tests, per-layer differential coverage, full logits/perplexity,
same-token KV visibility, exact counters, non-negative clean-route timing, and a
repeated prefill/decode A/B.

### P5: resident transformer execution

1. Introduce two banked residual roles and alternate ownership across attention
   and FFN without an intermediate DDR residual.
2. Qualify one complete transformer block against the standalone section oracles.
3. Add a fixed descriptor/layer table and walk all 28 layers while the residual
   remains resident. Weights and historical KV continue to stream from DDR.
4. Retain standalone named attention and FFN commands for differential debugging,
   recovery, and strict fallback before resident execution begins.

Gate: block and full-layer-walk equivalence, exact layer/traffic counters, reduced
DDR residual traffic, full-model quality, clean routing, and lower unprofiled wall
time on the same image.

### P6: persistent controller and DMA consolidation

After P5 fixes the actual movement schedule, replace the ten-DMA, 13-target generic
shell rather than adding another controller beside it:

1. Keep four persistent weight readers unless counters justify a different split.
2. Use one fixed layer controller and one sequential activation/result mover.
3. Consolidate K/V movement only if starvation counters prove that it will not
   throttle attention.
4. Remove superseded movers, control targets, and duplicated generic Q8/FP units;
   re-measure command bytes, transport, response work, CLBs, and routing pressure.
5. Consider a separate control clock only after topology simplification. CDC alone
   is not a token-throughput feature.

Gate: fewer routed movers/targets, recovered CLB capacity, unchanged error and
stream semantics, and improved unprofiled wall time without device regression.

### P7: ternary and KV memory efficiency

1. Evaluate denser exact ternary packing once P3-P6 make weight bytes the measured
   limit; retain a lossless source-to-resident adapter and exhaustive decoder tests.
2. Measure int8 or other explicitly qualified K/V storage first. Keep access
   contiguous and preserve dense exact attention as the short-context oracle.
3. Consider block-sparse or predictive top-K attention only after reporting
   candidate ratio, gather bandwidth, perplexity, and downstream task quality.

The leading-one predictor in VitaLLM remains a research option, not an assumption
in the dense execution plan.

### P8: spatial scaling, conditional

Do not replace the compressed-weight shared GEMM with a conventional systolic
array. If fused prefill, continuous batching, or speculative verification later
becomes compute-bound, broadcast weights into two or four spatial compute columns
and measure the same-device throughput/resource trade. Single-token decode cannot
escape its full weight stream by adding MACs.

Gate: counters demonstrate a compute rather than DDR/control bottleneck, and a
same-image A/B repays the additional area without degrading the single-request path.

### Vivado iteration ladder

Use increasingly expensive evidence only as the design becomes ready for it:

| Tier | Evidence | Purpose |
|---:|---|---|
| 0 | Focused Zig/Verilator/numerical tests and focused formal | Every relevant edit; functional, arithmetic, handshake, bounds, and ownership checks |
| 1 | Leaf OOC | Local inference, resource, and Fmax feedback for a changed RTL block |
| 2 | Integrated decode OOC | Boundary changes across GEMM, scratch, Q8, section, attention, or control logic |
| 3 | Synthesis/place/congestion probe or incremental combined route | Early exploratory fit/congestion feedback for a coherent batch |
| 4 | Clean combined route | Deployable milestone or material top-level, clock, constraint, memory, DMA, regmap, or control change |
| 5 | Board qualification | End-to-end correctness and performance claim for the routed milestone |

Host matchers, codecs without RTL changes, profiling, docs, harnesses, and RTL
that still fails Tier 0 do not justify a combined route. Synthesis-only/place-only
hooks, durable source-hash-addressed remote jobs, identical-build reuse, checkpoint
reuse, and a build lock should shorten feedback. Incremental routing is exploratory;
only a clean route supports release.

Timing acceptance is deliberately sign-only: a fully routed, structurally
constrained build passes when both WNS and WHS are non-negative. Positive slack is
a pass no matter how close it is to zero. Earning additional margin is a separate
timing-optimization pass, not a reason to reject or repeat a functional milestone.

## Ideas deliberately rejected for the current path

- **Wider GEMM or more arrays:** decode is already weight-bandwidth-bound.
- **Shared BoothFlex-style arithmetic:** valuable for ASIC area, mismatched to an
  FPGA with 7% DSP use and tight LUT routing.
- **Ternary lookup-table GEMM:** keep dense grouped ternary storage, but require a
  same-device PPA win before replacing the smaller direct reduction.
- **Immediate sparse attention:** current real KV traffic is far below weight
  traffic and the flash FSM is compute-bound.
- **Head-level overlap without DDR accounting:** matmul and flash use different PS
  ports but still share the DDR controller. Overlap is a measured scheduling option,
  not assumed free bandwidth.
- **A general fusion IR:** the model has two high-value repeated sublayers. Name
  them, version them, and retain the flat runtime switch.

## Architecture lessons incorporated

The external designs reinforce three choices, adapted to this implementation:

- TeLLMe's strongest applicable idea is causal query-blocked prefill with online
  softmax and small resident query state. Penzai should combine that schedule with
  its already-verified native KV layout rather than copy TeLLMe's complete engine.
- TeLLMe and VitaLLM both validate treating normalization/absmax as a reduction
  barrier and passing a quantized vector plus scale into projection hardware. In
  Penzai the contract remains blockwise Q8_0 rather than one scale per vector.
- VitaLLM's head pipeline motivates grouped GQA-head streaming, but only after the
  simpler shared-activation and layer-local fusions land.

References:

- VitaLLM: https://arxiv.org/abs/2605.00320
- TeLLMe: https://arxiv.org/abs/2504.16266

## Completion criteria

The roadmap is complete when:

1. Binary and issue-ordered two-bit ternary run from one fully routed image with
   non-negative setup and hold slack.
2. The streaming FFN and named attention sections remove repeated PS activation
   quantization and intermediate sublayer materialization.
3. A fixed layer walk retains the residual through all 28 layers, and its persistent
   controller has retired the superseded generic DMA/control shell.
4. Decode profiling shows weight streaming as the largest remaining short-context
   component, and the binary design sustains at least 26 device tokens/s on the
   fixed benchmark or a measured hardware limit explains the miss.
5. User-visible 13-token TTFT is reported correctly and is materially below the
   historical approximately 1.2 seconds; 64-token prefill is a standard regression.
6. Dense attention scales against real, not padded, KV length; query-blocked PL
   prefill is faster than the PS oracle.
7. Long-context compression or sparsity is enabled only with quality and actual
   memory-traffic evidence.
8. Every claimed win has an independent A/B, a numerical gate, hardware counters,
   and the Vivado evidence appropriate to its iteration tier; deployed milestones
   include a clean routed timing report.
