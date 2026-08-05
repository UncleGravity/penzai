# Accelerator Priorities

Status date: 2026-08-04

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
0.5 GB/s, far below the weight-stream bandwidth, so the current attention schedule
is compute/control limited rather than DDR limited.

Prefill GEMMs already run in PL, but multi-token attention falls back to PS.
Backend downloads grow from roughly 1.4/2.8 GiB for Q1/Q2 at a 512-token prompt;
these are graph-boundary transfers, not GEMM kernel DMA counters. The measurements
therefore require both section-level boundary removal and an earlier unified PL
attention redesign. Old transport tail measurements taken through mixed SSH tunnels
are not signoff data, but device-clock measurements are stable and trustworthy.

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
| 1 | Bounded waste removal | Low-risk short-context savings and benchmark validation | Independent A/B for each change with identical numerical output |
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
cycle-per-query-KV-pair baseline for the unified attention engine.

### P2: section substrate and unified PL attention

- [ ] Define an explicit section/scratch contract: bank ownership, tile lifetime,
  layouts, bounds, and concrete tensor descriptors. Do not introduce a generic IR.
- [ ] Implement a PL FP32-to-Q8_0 quantizer matching the canonical host quantizer.
- [ ] Store and reuse grouped Q/K/V and gate/up Q8 activations on chip.
- [ ] Write GEMM results into banked scratch in the next consumer's layout.
- [ ] Make prefill token/row layout conversion a scratchpad addressing problem,
  rather than a PS transpose and DDR round trip.
- [ ] Extend the existing online attention recurrence with a query-tile axis.
  Decode uses one query; prefill initially tiles 4-8 queries and reuses K/V.
- [ ] Pipeline QK dot, online softmax update, and `p*V` accumulation to reduce
  cycles per valid query-KV pair for single-query decode as well as prefill.
- [ ] Preserve FP32 softmax and output-accumulator state, and guarantee that newly
  appended K/V rows are visible to the same causal attention operation.
- [ ] Require strict shape, layout, RoPE, normalization, mask, and KV-cache matches;
  retain the current PS/op path as the correctness fallback.
- [ ] Add saturation counters and full-model logits/perplexity gates before replacing
  the host quantizer.

P2 must produce one attention engine and ABI, not separate prefill and decode
kernels. Completion requires PL execution for supported multi-token prefill,
lower backend downloads and command counts, faster 128/512-token prefill, and
faster attention at decode contexts 512 and 2048. Query tile size and any spatial
math replication are selected from OOC/routed timing and measured cycles per pair.

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
- the combined routed build passes with deliberate timing headroom, not rounded-zero WNS;
- Q1 and Q2 model-level logits/perplexity gates pass at short and long contexts;
- a board smoke run verifies bitstream identity, capabilities, DMA/cache behavior,
  prefill, and decode.

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

The latest recorded routed build reports 0.000 ns WNS, 329 setup paths below 50 ps,
two critical methodology warnings, 79,406 LUTs, and a dirty source manifest. Its top
reported setup paths include control SmartConnect to a DMA register, a GEMM activation
reduction path, and flash control. Therefore:

- rerun a clean routed build before acting on older path rankings;
- fix the current report's path owners, not historical path names;
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
