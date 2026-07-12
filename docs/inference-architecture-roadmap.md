# Penzai inference architecture roadmap

Status: 2026-07-10. This document is the performance and offload plan for the
current KR260 implementation. It starts from the deployed f300 bitstream, the
current Zig/RTL data paths, and one measured Bonsai-1.7B run. Estimates below are
explicitly marked; everything else is either measured in that run or derived
from hardware counters.

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
3. Execute the attention and FFN sublayers as a small number of named pipelines
   that retain short-lived intermediates on-chip.
4. Keep weights and the KV cache resident in DDR, streaming each only when the
   model operation requires it.
5. Optimize dense exact attention first. Introduce KV compression or approximate
   sparsity only when real context lengths make KV traffic comparable to weights.
6. Preserve one bitstream and one downstream datapath for binary and ternary
   weights; only packing and the GEMM front end depend on weight format.

At the present binary weight density and measured bandwidth, the absolute
weight-stream roof is about 47 tokens/s. A realistic first destination is
26-30 device tokens/s after removing PS passes, with wall throughput depending
on the transport work described below. Five-trits-per-byte ternary carries five
weight payloads where binary carries eight. Ignoring block scales and alignment,
that reduces the payload-only roof to about 29.5 tokens/s. The physical roof will
be somewhat higher, roughly 29-34 tokens/s, because scales and padding are already
part of the measured binary stream; the final packed layout must provide the exact
number. Every proposed fusion must improve both formats without assuming binary's
higher roof.

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
should target its input/output boundaries and ternary decode seam, not more MACs.

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

The routed f300 build uses 65.47% LUT, 38.76% FF, 33.68% BRAM, 6.25% URAM, and
7.37% DSP. Timing passes by only 0.003 ns. The design has abundant DSP, BRAM, and
URAM but little timing margin and localized LUT-routing pressure. Prefer on-chip
buffers and DSP-shaped arithmetic; require a clean routed build for every RTL
increment. An ASIC-style shared Booth datapath or a 52k-LUT ternary lookup engine
does not match this resource balance.

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
  -> RMSNorm * gamma -> one Q8 activation
  -> grouped gate/up projections
  -> SwiGLU
  -> Q8 quantization -> down projection
  -> residual add
```

The gate vector can be buffered while the up projection streams, then the SwiGLU
output can feed the down-projection quantizer without an fp32 DDR round trip.
The largest Bonsai intermediate is small relative to available BRAM/URAM.

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

1. Pack five trits per byte in resident DDR.
2. Unpack in the GEMM weight front end to `{zero, sign}` controls.
3. Reuse the Q8 ingress, fixed-point accumulator, output path, flash kernel, and
   all layer fusions unchanged.
4. Maintain separate per-format counters and roofline reporting.

Do not import table-lookup arithmetic solely because it is ternary. The current
FPGA is LUT/timing constrained and DSP/BRAM rich; direct zero/sign selection is a
better fit unless a same-device synthesis A/B proves otherwise.

Gate: packed-layout round trip, bit-exact ternary dot products against the software
oracle, routed single bitstream, and a measured physical roof that agrees with the
actual packed ternary bytes including scales and alignment.

### Phase 3: named layer-local pipelines

Build two closed operations, in this order.

1. **FFN pipeline.** Fuse normalized Q8 production, gate/up launch, on-chip
   gate buffering, SwiGLU, down-input quantization, and down projection. This is
   simpler than attention and removes the measured 7.0 ms/token SwiGLU pass plus
   intermediate writes and another quantization boundary.
2. **Attention pipeline.** Fuse normalized Q8 production with grouped Q/K/V;
   attach Q/K normalization and RoPE to attention ingress; append K/V; run flash;
   quantize its output for the output projection. Add residual production where
   doing so does not destroy a value needed by the next sublayer.
3. Use ready/valid FIFOs and explicit small BRAM/URAM buffers between stages.
   Do not require both large weight and KV streams to peak concurrently; schedule
   around the shared DDR controller and prove overlap with stall counters.
4. Keep intermediate f32 materialization as a debug/fallback mode until the
   fused path passes full-model numerical gates.

Gate: per-layer differential tests, full logits/perplexity, routed timing, and
end-to-end counters showing that standalone RMSNorm/mul/SwiGLU/RoPE/set_rows calls
fall rather than merely moving time to an opaque fused bucket.

A defensible binary destination after Phase 3 is approximately 33-38 ms/token
device time: about 21 ms weights, the current 7 ms attention pending its own
optimization, and 5-10 ms of remaining control/data movement. This corresponds
to roughly 26-30 device tokens/s. Wall throughput will be lower until Phase 5.

### Phase 4: attention throughput and prefill

1. Remove the per-call mask scan duplication. All 28 layer calls in a graph share
   the same causal extent; compute it once per graph or carry an explicit validated
   `kv_hi` into each command.
2. Pipeline the existing dense exact kernel before considering sparsity. Interleave
   independent heads through dot, score, softmax, and AXPY pipelines instead of
   waiting for each numeric result in a sequential handshake. The target is a
   measured feed-bound kernel, not merely zero AXIS stalls around a compute-bound
   FSM.
3. Add query-blocked prefill. Hold `p` query tokens and their online-softmax state,
   stream each native-layout K/V position once for the block, and suppress causally
   invalid query/KV pairs. Start with `p=4`; do not replicate arithmetic merely to
   match a paper's diagram.
4. Keep decode and prefill schedules behind the same external op and numerical
   oracle. Decode remains KV-major with one query; prefill adds a query-tile axis.

Gate: actual cycles versus real `kv_hi`, attention error versus the PS oracle,
decode latency across context lengths, and prefill wall improvement. Moving the
measured 102 ms prefill attention to PL is useful only if K/V are reused across
the query block; a token-outer PL loop would just move the bottleneck.

### Long-context track: compression before approximation

Only promote this track when benchmarks cover contexts near or above 2k tokens.

1. Measure int8 K/V storage and attention first. It approximately halves KV
   traffic while preserving contiguous access and admits the same numerical gates
   as other precision changes.
2. Consider block-sparse or predictive top-K attention only after reporting
   candidate ratio, index-gather bandwidth, perplexity, and downstream task
   quality. A top-K selector that turns one contiguous DMA into many random reads
   can lose despite reducing bytes.
3. Keep exact dense attention available for short contexts and as the quality
   oracle.

The leading-one predictor in VitaLLM is a research option for this track, not a
near-term optimization for the current 13-37-token workload.

### Phase 5: control plane and transport

Revisit control batching only after named layer operations exist.

1. A resident program should launch a small number of attention/FFN operations
   per layer, with dynamic token position and KV extent patches. Replaying hundreds
   of today's tiny operations is not the target.
2. Re-measure the 7.6 ms/token transport bucket using Phase-0 decomposition. The
   request already contains one graph RPC/token; likely candidates include command
   serialization, approximately 10 KiB of per-token preloads, device preload writes,
   profiling response work, and TCP framing.
3. Reduce command bytes by lowering fused operations before designing a new wire
   transport. Preserve the current request/response correctness model until data
   shows it is the limiter.
4. Split the control clock only if it creates robust timing margin for the fused
   build. The current f300 design passes by 3 ps, so margin matters, but a clock
   split is not itself a token-throughput feature.

Gate: unprofiled wall time, device/transport decomposition, commands/token, wire
bytes/token, and a timing-clean bitstream. The sequencer is successful only if it
reduces end-to-end wall time, not if it reduces AXI-Lite writes in isolation.

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

1. Binary and five-trits-per-byte ternary run from one timing-clean bitstream.
2. Decode profiling shows weight streaming as the largest remaining short-context
   component, with no repeated PS activation quantization or standalone FFN passes.
3. The binary design sustains at least 26 device tokens/s on the fixed benchmark,
   or a measured hardware limit explains the miss.
4. User-visible 13-token TTFT is reported correctly and is materially below the
   current approximately 1.2 seconds; 64-token prefill is a standard regression.
5. Dense attention scales against real, not padded, KV length; query-blocked PL
   prefill is faster than the PS oracle.
6. Long-context compression or sparsity is enabled only with quality and actual
   memory-traffic evidence.
7. Every claimed win has an independent A/B, a numerical gate, hardware counters,
   and a routed timing report.
