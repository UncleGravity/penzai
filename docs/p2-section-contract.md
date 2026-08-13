# P2 Section And Scratch Contract

Status: P2a contract v1 with board-qualified P2c adaptive attention at f285; P2d
grouped activation reuse is closed at the bounded f285 structural, numerical, and
device-regression qualification, 2026-08-12

This document fixes the boundary for P2 implementation. P2a did not add a wire
operation, and P2b/P2c changed the internal attention schedule and primitive bridge
without adding a named section ABI. P2d adds one bounded grouped-GEMM operation;
it is not a named section ABI. Later RTL revisions must preserve this ownership,
layout, numerical, fallback, and measurement contract.

## Scope

P2 turns the existing GEMM and attention kernels into reusable engines behind a
fixed layer-local substrate. It is not a generic graph compiler, scratch allocator,
or device-side GGML interpreter.

The host will eventually issue two explicit section commands. Their external
descriptors name DDR tensors and model semantics. The PL controller alone owns
scratch addresses and stage sequencing.

V1 has these caps:

| Item | Cap |
|---|---:|
| Logical query tile | 4 tokens |
| Physical query-head slots | 64 |
| Effective tile, up to 16 heads | 4 tokens |
| Effective tile, 17-32 heads | 2 tokens |
| Model dimension | 4096 |
| FFN dimension | 12288 |
| Head dimension | 128 |
| Query heads | 32 |
| KV heads | 8 |
| Context | 8192 |

A longer prompt is processed as several tiles; scratch never scales with the full
prompt. The bitstream advertises 64 query-head slots. The implementation uses a
16-head stride for shapes with up to 16 heads and a 32-head stride above that, so
the logical four-query cap fits both the Bonsai shape and the supported 32-head
shape without allocating 128 physical slots. A descriptor outside the advertised
caps uses the existing primitive path.

The executable form of the address and capacity rules lives in
`shared/section.zig`.

## Scratch Ownership

Only one atomic section owns scratch in V1. Scratch is invalid before command
start and after command completion. It is not a `TensorRange`, is not visible to
the host, and never persists by pointer identity across commands.

Four named F32 roles provide 512 KiB of logical storage:

| Role | Capacity | Attention | FFN |
|---|---:|---|---|
| `R` | `4 x 4096` | Input residual | Input residual |
| `X0` | `4 x 12288` | Q/postprocessed Q | Gate, then in-place SwiGLU |
| `X1` | `4 x 12288` | K temporary, then output projection | Up projection |
| `X2` | `4 x 4096` | V temporary, then attention output | Down projection |

Each role is four physical 64-bit banks. For an even F32 row:

```text
group = row / 8
bank  = (row % 8) / 2
addr  = token * groups_per_role + group
word  = {odd row, even row}
```

The current GEMM emits two F32 rows per beat in
`[rowblock][token][row-pair]` order. The scratch writer uses the mapping above,
so that stream becomes a token-addressable 256-bit group without a transpose
copy or DDR round trip. Consumers use address generators over this layout rather
than requiring a token-major materialization.

The Q8 role reuses GEMM's existing `acts_mem` and `act_scale_mem`:

```text
address = q8_subblock * 8 + token
data    = 32 x i8
scale   = one f16
```

The padded 40-byte DMA stream remains the legacy packed ingress format, not an
internal scratch type. GEMM v13 exposes three explicit activation modes:

- `packed_load` retains the existing host-quantized primitive path;
- `raw_f32_load` accepts two FP32 values per 64-bit beat and writes the exact
  canonical Q8 record into GEMM's existing activation memories;
- `reuse` consumes that record without another activation DMA or quantization.

Reuse is valid only for a latched `{epoch, K, token_count}`; it is never inferred
from a host address. An epoch or shape mismatch consumes no activation beats and
rejects before the kernel consumes weights or emits results, while preserving the
previous resident record. DMA movers may already be armed, so this is a kernel
consumption guarantee rather than a claim of no surrounding bus activity. The P2d
group assigns a fresh epoch to each column tile, loads raw FP32 for the first
projection, and reuses the same native-Q8 record for the second projection.

For every valid finite 32-value block, the PL quantizer matches
`shared/layout.zig` exactly:

```text
scale        = binary32(amax / 127)
stored_scale = binary16_rne(scale)
inv          = binary32(1 / scale)
q[i]         = clamp_i8(round_even(binary32(value[i] * inv)))
```

The quantized bytes use the unrounded FP32 scale even when the stored F16 scale is
subnormal. Zero blocks are valid. Non-finite inputs, malformed framing, an infinite
stored scale, or unsupported FP32 arithmetic set a diagnostic status, invalidate
resident state, and abort before the kernel consumes weights or emits results.

Attention additionally owns small dedicated storage:

- `K_NEW[token][kv_head][dimension]` in F16;
- `V_NEW[token][kv_head][dimension]` in F16;
- FP32 `m` and `l` state indexed by `(query, head)`;
- an optional staged F16 mask tile for the generic primitive path.

K/V projections are rounded to F16 before entering `K_NEW` and `V_NEW`. The same
values are used by attention and committed to the persistent cache. X2 is the
only attention output-accumulator store: after V has been copied into `V_NEW`, X2
is reinitialized as FP32 accumulator/output state and remains owned by attention
until the output projection consumes it.

Reset control, ownership, epoch, and valid state. Do not reset scratch data whose
valid bit is clear.

## Section Schedules

Attention owns the roles in this order:

```text
DDR input -> R
R -> norm/gamma -> Q8
Q8 + Wq -> X0 -> Q norm + RoPE in place
Q8 + Wk -> X1 -> K norm + RoPE -> K_NEW + DDR append
Q8 + Wv -> X2 -> V_NEW + DDR append
X2 becomes attention accumulator/output
X2 -> Q8
Q8 + Wo -> X1
R + X1 -> DDR output
```

FFN owns them in this order:

```text
DDR input -> R
R -> norm/gamma -> Q8
Q8 + Wgate -> X0
Q8 + Wup   -> X1
SwiGLU(X0, X1) -> X0 in place
X0 -> Q8
Q8 + Wdown -> X2
R + X2 -> DDR output
```

Weights, historical/persistent KV, section input/output residuals, positions, and
parameter vectors remain in DDR. Temporary projections, Q8 activations, SwiGLU
values, and attention state do not.

## Bounded Gate/Up Bridge

Wire ABI 14 adds the fixed-arity `matmul_q1a8_group2` operation. The lowerer does
not inspect semantic tensor roles. It structurally groups any two adjacent compute
matmuls with the identical F32 activation range, equal shape and weight format,
complete resident bindings, disjoint outputs, no output views, no unsafe tensor
flags, and no intervening compute operation. In the validated Qwen/Bonsai graphs,
only the 28 gate/up pairs per graph meet that contract; Q/K/V do not. A failed
condition leaves both primitive matmuls unchanged.

The device validates every activation, weight, and destination range before either
projection starts. GEMM v13 then uses `raw_f32_load` followed by `reuse`. An older
kernel executes the already validated pair as two primitive PL operations when
both are eligible, then uses the shared-quantization PS oracle if PL is unavailable.
The grouped command therefore has a strict pre-execution fallback and does not
silently broaden graph eligibility.

This bridge does not yet write gate/up results into section scratch: both results
still cross the primitive DDR boundary before SwiGLU. It also does not group Q/K/V;
the observed graph ordering and lifetimes belong in the future named attention
section rather than a more general primitive matcher. P2d proves the canonical
PL-Q8 ingress and resident-reuse contract needed by the FFN section while reducing
each 28-layer one-token graph from 537 to 509 commands. Fake-device full-model Q1
and Q2 runs each emit exactly 28 grouped commands in both prefill and decode, and
inspection identifies them as the gate/up pairs. Those runs validate structural
lowering, command accounting, and both weight formats; the matcher itself does not
identify tensor semantics, and the runs are not hardware performance measurements.

The pre-execution checks provide fail-before-write behavior for malformed handles
and unsupported requests, but a started group is not transactional. If the first
projection has committed before a later kernel or DMA failure, its destination is
not rolled back and the runtime returns a backend error without retrying the pair.
Driver error cleanup issues best-effort resets to every armed mover. Bounded kernel
and DMA polls prevent an infinite software wait, but a DMA timeout combined with a
reset that never clears has no in-command recovery guarantee; daemon or bitstream
recovery remains the containment boundary.

## Unified Attention

Decode and prefill use one core and one numerical oracle:

```text
query_count = 1       decode
query_count = 1..4    prefill tile with up to 16 heads
query_count = 1..2    prefill tile with 17-32 heads
```

The core holds independent state for every `(query, head)`, walks KV outermost,
loads each K/V position once, and applies it to every valid query and mapped GQA
head. Query head `h` uses KV head `h / head_ratio`.

The recurrence remains:

```text
m'   = max(m, score)
corr = exp(m - m')
p    = exp(score - m')
l'   = l*corr + p
acc' = acc*corr + p*V
out  = acc/l
```

Dot products, `m`, `l`, correction, accumulation, and residuals retain the current
FP32 semantics. The current BF16 `p*V` multiply with FP32 accumulation remains the
hardware-modeled implementation. Any precision change is a separate quality-gated
decision.

There are two providers behind this core:

- The generic `flash_attn_f32` path accepts staged arbitrary F16 mask values,
  including finite additive biases and interior `-inf` holes.
- The named attention section accepts only a recognized causal-prefix graph and
  generates per-query bounds from validated positions and cache extent.

The section path reads newly produced rows from `K_NEW`/`V_NEW`, so every query can
see its own key/value and earlier rows in the same tile without relying on a DDR
read-after-write. Cache writes must finish before the section reports completion.

P2b pipelines the existing single-query decode path across heads within one KV
position. P2c extends the tags with a query index, makes the composed softmax path
II=1, and issues independent `(query, head)` work in consecutive cycles. The
controller retains a hard KV barrier: it does not advance to the next position
until every AXPY accumulator writeback for that position has retired. This
preserves the `m`, `l`, and output-accumulator recurrence while reusing each K/V
load across all finite queries in the resident tile.

The production engine contains 64 physical query-head state slots. A multiplier-
free map uses a 16-head stride for `n_heads <= 16` and a 32-head stride otherwise.
The driver derives the legal tile size from the advertised slot count and rejects
any shape that would alias a slot. Bonsai therefore retains four-query tiles;
32-head models use two-query tiles. Decode keeps the existing direct single-query
DMA path and exact output behavior.

P2c also provides a bounded bridge for the generic `flash_attn_f32` operation. It
copies a Q tile from the supported packed `[token][head][dimension]` layout into
query-major stream order, transposes and analyzes the mask once, streams K/V
directly once per tile, and scatters output back into the existing strided tensor.
For Q, the exact accepted strides are `q_nb1 = n_heads * head_row_bytes` and
`q_nb2 = head_row_bytes`; padded or transposed views fail closed. Partial, sparse,
and all-masked tiles are supported. An unknown engine ID/version/slot count or
unsupported shape, stride, destination, range, or mask layout falls back before PL
execution.

That bridge accelerates a primitive operation; it is not the named attention
section described above. Q, mask, and output still cross the primitive boundary,
and the surrounding GGML graphs, commands, downloads, projections, RoPE, and KV
append behavior are unchanged. Same-section visibility of newly produced K/V
remains a requirement for the future named attention section.

## External Descriptor Boundary

The future wire ABI will contain two concrete versioned operations, with no
scratch addresses or arbitrary stage list. The following lists semantic fields,
not frozen wire structs:

```text
ffn_section_v1 {
  input, output, norm_gamma,
  gate_weights, up_weights, down_weights,
  model_dim, ffn_dim, token_count,
  norm_eps, weight_format
}

attention_section_v1 {
  input, output, norm_gamma,
  q_weights, k_weights, v_weights, output_weights,
  q_norm_gamma, k_norm_gamma,
  positions, k_cache, v_cache,
  model_dim, head_dim_q, head_dim_v, n_heads, n_head_kv,
  token_count, cache_start, cache_capacity,
  norm_eps, qk_norm_eps, scale, rope_parameters, weight_format
}
```

The device validates the complete descriptor before changing scratch or KV. The
host lowerer emits it only for exact recognized shapes, strides, normalization,
RoPE, mask, cache, and resident weight semantics. A failed match falls back before
section execution begins; a failure after start is an execution error, not a
partial retry.

P2a deliberately did not assign section wire tags or serialize these fields. Wire
ABI 14's fixed `matmul_q1a8_group2` is a bounded primitive operation and does not
freeze either section descriptor. Concrete range, stride, RoPE, and weight-format
types are frozen with the first executable section so that ABI describes an
implemented controller rather than a speculative one.

## Measurement Contract

Existing profile fields retain their meaning:

- `valid_qkv_pairs` is query tokens times mask entries other than F16 `-inf`;
- `processed_qkv_pairs` is the query/KV space actually walked;
- neither count includes query heads.

Engine work is reported as:

```text
valid query-head/KV updates = valid_qkv_pairs * n_heads
cycles per valid update     = kernel cycles / valid updates
cycles per processed update = kernel cycles / processed updates
```

The profiler emits these derived values for humans and one integer-only
`attention_result` record per exact backend/path/shape bucket for artifacts.
Zero-valid-pair cases report `n/a`, not zero efficiency. Compare cycles/update
only with matching head dimensions, precision, clock, and query-shape mix.

The P2a 16-head Bonsai decode baselines were:

| Decode range | Cycles/query-KV position | Cycles/query-head/KV update |
|---|---:|---:|
| Context 0, positions 2-65 | 2389.7 | 149.4 |
| Context 512, positions 513-576 | 2281.1 | 142.6 |
| Context 2048, positions 2049-2112 | 2323.6 | 145.2 |

At context 2048, combined K/V starvation is about 2.1% of cycles. The limit was
therefore the sequential compute/control schedule, not the observed DMA
feed. VPN latency changes wall and transport time but not these device counters.

The context-512 P2b board A/B measures the same work before and after the new
schedule:

| Metric | P2a sequential | P2b head-streamed |
|---|---:|---:|
| Cycles/valid query-head/KV update | 142.57 | 33.224 |
| Flash kernel | 115.93 ms/token | 27.016 ms/token |
| Q1 device | 182.652 ms/token | 93.715 ms/token |
| Q2 device | 204.366 ms/token | 115.429 ms/token |

This is a 4.29x increase in valid update throughput and cuts the flash kernel by
76.7%. The final post-AXPY-repair artifact is
`20260812T034043Z-characterize-dfc29ec25d66`; accounting and exact-token checks
pass and K/V/O starvation is zero for both formats.

At context 2048, the final Q1 run measured 32.874 cycles per valid update
(9.126 million updates/s), 102.138 ms/token of flash kernel time, and
174.055 ms/token of device time. Relative to the sequential baseline, valid-update
throughput improved 4.42x, flash time fell 77.4%, and device time fell 67.4%.
Accounting and token-count checks pass and K/V/O starvation remains zero.

The complete current flash kernel was also measured through registered OOC
boundaries at the production 3.333 ns target:

| Probe revision and target | DSP | LUT | FF | BRAM tiles | WNS | Estimated Fmax |
|---|---:|---:|---:|---:|---:|---:|
| P2a diagnostic, 3.333 ns | 60 | 21,623 | 18,276 | - | -0.157 ns | 286.5 MHz |
| P2a temporary baseline, 3.600 ns | 60 | 21,623 | 18,276 | - | +0.110 ns | 286.5 MHz |
| P2b plus AXPY repair, 3.333 ns | 60 | 22,109 | 18,889 | - | +0.291 ns | 328.7 MHz |
| P2c fixed 128 slots, 3.333 ns | 60 | 23,489 | 19,315 | 33 | +0.215 ns | 320.7 MHz |
| P2c adaptive 64 slots, 3.333 ns | 60 | 22,657 | 19,216 | 19 | +0.215 ns | 320.7 MHz |

This is a synthesis/OOC diagnostic, not routed signoff. P2b both restores the
3.333 ns registry target and reduces cycles per update without adding DSPs. The
small AXPY pipeline repair splits the two-DSP multiply path seen in the first P2b
combined route and preserves all cosim hashes. Its cosim, formal, OOC, and board
gates pass.

The released P2b clean combined timing run reached setup and hold at +0.033/+0.007 ns
after guarded placement and pre-route `AggressiveExplore`. It uses 80,108 LUTs,
94,587 FFs, 48.5 BRAMs, four URAMs, and 92 DSPs, with 98.25% of CLBs occupied.
The AXPY cascade is absent. Twelve setup paths remain below the 50 ps target, but
they span the disabled sequencer, flash, and unrelated GEMM structures rather
than one repairable path family. Exact-policy run
`20260812T062923Z-b829dee03903-dirty-w512-p4-f300-clean` clears the 25 ps
development release floor, was promoted and deployed with matching hashes, and
passed bounded Q1/Q2 logits and profile smoke. The smoke artifact is
`20260812T072252Z-characterize-ed0a92df72da`; it has identical start/end
capabilities, closed accounting, PL execution for GEMM and attention in both
phases, exact token counts, and zero K/V/O stalls. The 50 ps value remains a
target rather than a hard architectural boundary.

The first P2c implementation kept four queries for every supported head count,
which made Q and accumulator storage 128 slots wide. It passed OOC but its clean
combined f300 run `20260812T112623Z-95b751ff9ac3-w512-p4-f300-clean` failed at
-0.035 ns setup WNS with 124 failing setup endpoints, 69.5 total BRAM tiles, and
98.91% CLB occupancy. It produced no deployable bitstream. The adaptive 64-slot
revision removes 14 flash BRAM tiles while retaining the four-query Bonsai tile.

Adaptive clean f300 run
`20260812T144901Z-bbeac0c04eff-w512-p4-f300-clean` also failed closed, at
-0.088 ns WNS with 231 failing setup endpoints and 560 paths below the 50 ps
target. It was not promoted. Clean f285 run
`20260812T155038Z-3ef082b0fe4a-w512-p4-f285-clean` passes at +0.036/+0.010 ns
setup/hold, with five paths below 50 ps. It uses 80,837 LUTs, 95,229 FFs, 55.5
BRAM tiles, four URAMs, and 92 DSPs at 98.61% CLB occupancy. The exact artifact
was promoted and deployed with matching receipt and bitstream hashes. Live
capabilities report 284,997,152 Hz, engine `0xF1A54A01` version 1, and 64 query
slots. The run clears the 25 ps release floor but misses the 50 ps headroom target;
it was P2c's qualification point, not evidence that f300 closes.

The first multi-token board smoke fell back to PS because the driver predicate had
the Q stride axes reversed. That failure was visible in the truthful backend field.
Commit `0dc91c0` aligned the predicate with the observed token-major layout and
added a concrete Bonsai-stride regression test; no RTL or bitstream change was
required.

The repaired bounded board gates pass for both formats. Q1/Q2 32-token (`p32`)
logit checks had zero token mismatches and maximum absolute differences of
0.0982/0.1396. Artifact
`20260812T174257Z-p2c-repaired-p128` records `pl/staged` prefill for 224 calls and
896 kernel runs, 24.05 cycles per processed query-head/KV update, zero K/V/O stalls,
closed accounting, and 18.397/25.606 s Q1/Q2 prefill. Artifact
`20260812T170744Z-characterize-32c539cee6c8` provides the same-f285 PS fallback
control: repaired prefill wall is lower by 24.51%/21.86%, and flash command time
fell from about 6.4 s to 0.691 s. Artifact
`20260812T174506Z-p2c-repaired-c512` records 896 PL prefill calls and 3,584 kernel
runs at 21.222 cycles per processed update, with exact counters and zero stalls.
Its Q1/Q2 prefill wall is 79.132/104.541 s, down 50.63%/43.62% from the P2b
160.300/185.434 s. Decode remains on the direct path at 33.315 cycles per update,
96.093/119.008 ms/token of device time, and 108.446/137.450 ms/token steady wall.
The slight device-time increase from P2b is consistent with f285; cycle efficiency
is unchanged.

Scale artifact `20260812T175007Z-p2c-repaired-c2048` completes the long-context
gate. All 3,584 prefill calls and 14,336 kernel runs use `pl/staged`, totaling
19,292,640,695/19,292,626,575 Q1/Q2 cycles at 20.494 cycles per processed
query-head/KV update. Pair and beat counts are exact, K/V/O stalls are zero, and
accounting closes. Q1/Q2 prefill wall is 425.994/813.215 s, 76.96%/57.79% below
artifact `20260804T211805Z-baseline` at 1848.895/1926.439 s. Decode remains
`pl/direct` at 32.944 cycles per update, 180.560/203.465 ms/token of device time,
and 193.748/221.127 ms/token steady wall. Relative to P0, decode device time falls
66.14%/63.33% and the flash scoreboard falls from about 459.1 to 115.7 ms/token.
Relative to P2b Q1 at f300, cycles per update increase only 0.21% while device time
increases 3.74%, consistent with f285. The unchanged 5.8/11.6 GiB Q1/Q2 backend
downloads now expose the section-boundary work that P2c intentionally does not
remove.

P2d's implementation verification covers the new activation boundary independently
of product performance. The canonical quantizer cosim matches all quant bytes and
F16 scale bits across 1,032 directed and randomized blocks, including RNE ties,
subnormal stored scales, bubbles, backpressure, framing, and error status. Binary
and ternary GEMM integration cosims exercise raw load followed by zero-activation-
beat reuse; focused formal tasks prove invalid reuse and raw-ingress abort both
fail closed. Standard Zig tests pass 255/255, pinned llama-enabled tests pass
278/278, and the aggregate RTL suite passes 36/36 build steps.

The initial registered OOC probes passed the 3.333 ns target:

| Initial P2d probe | DSP | LUT | FF | CARRY8 | WNS | Estimated Fmax |
|---|---:|---:|---:|---:|---:|---:|
| Exact Q8 quantizer leaf | 2 | 792 | 908 | 22 | +0.192 ns | 318.4 MHz |
| GEMM v13 core | 32 | 38,544 | 42,935 | 719 | +0.619 ns | 368.5 MHz |

The corresponding run IDs are
`20260812T210745Z-0c77a2d69ab0-q8subnormal` and
`20260812T211434Z-0c77a2d69ab0`.

Those isolated synthesis diagnostics were not release evidence. The first clean
combined f285 attempt,
`20260812T212725Z-f9e1ca83f8ae-w512-p4-f285-clean`, routed fully but failed closed
at -0.215/+0.010 ns setup/hold. It had 142 failing setup paths and 580 below 50 ps,
used 81,284 LUTs, 96,220 FFs, 55.5 BRAM tiles, four URAMs, and 95 DSPs, and reached
99.32% CLB occupancy. It was not promoted or deployed.

Commit `547d87b` repairs the exposed integrated paths by replacing the flat ingress
block-count multiply and wide counters with nested column/Q1/sub-block state,
registering the input scalar boundary, compacting output emission to one 64-bit
register, and retiming the GEMM `FE_LAT` exponent addition. This removes 228 state
bits and one DSP. Existing binary and ternary raw-load/reuse hashes and cycle
counts are unchanged, and the expanded cosim now crosses multiple column and Q1
boundaries and rejects a missing final `TLAST`.

The current-source full-decode integration probe passes at 3.333 ns:

| Current P2d probe | DSP | LUT | FF | CARRY8 | BRAM | URAM | Setup/hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full decode | 34 | 40,141 | 44,912 | 775 | 2 | 4 | +0.245/+0.039 ns |

Its artifact is `20260812T223110Z-547d87b12094-full-decode`. The failed route's
ingress bookkeeping/multiply and GEMM exponent-add path families do not appear in
the current probe's reported critical paths. Its 3.333 ns production artifacts
pass. The probe driver exited afterward when a read-only, non-production second-
period report script attempted to mutate the open design; that later report error
does not affect the production source or result.

Replacement clean f285 run
`20260812T224303Z-547d87b12094-w512-p4-f285-clean` uses commit `547d87b12094`
and source bundle
`1cfc1e173ba0ae1d06d1fceb1d3fa83ec29535f760a7a7f91a6b5e0458078249`.
It is fully routed, all structural timing counts are zero, and removal of the
75 ps placement guardband restores exactly 75 ps. Setup/hold pass at
+0.043/+0.010 ns with no violations; 4/55/622 setup paths are below
50/100/200 ps. The release floor is met, while the 50 ps headroom target is missed
by 7 ps. Routed utilization is 81,887 LUTs, 95,699 FFs, 1,408 CARRY8s, 94 DSPs,
55.5 BRAM tiles, and four URAMs. CLB use is 14,519/14,640 (99.17%). Methodology
reports two critical findings, TIMING-2 and TIMING-4, and six warnings: five
TIMING-28 and one ULMTCS-1.

That exact image was promoted and deployed with bitstream SHA-256
`9ab576cad24eb3c77d6b55200d5e9a08d92f0197625f76d479c13fc6fa82a70f`.
Live capabilities pass at schema 2, wire ABI 14, profile ABI 6, GEMM ID
`0xB05A2000` v13 at 284,997,152 Hz, and flash v1 with 64 query slots. Q1 `p32`
logits pass with maximum absolute error 0.098204 and zero token mismatches. Q2
`p32` passes with per-step differences 0.089185/0.131115, maximum absolute error
0.131115, exact argmax at both steps, zero token mismatches, and `check=ok`.

Final board artifact `20260812T235644Z-characterize-9bb6d3eb522f` contains all
six expected samples and reports `complete=true`, `run_validated=true`, and
`accounting=ok` throughout. Its capability responses are byte-identical at start
and end. Exact graph and operation counts are:

| Workload phase | Graphs | Commands | Group2 | Primitive matmul |
|---|---:|---:|---:|---:|
| p128 prefill | 15 | 4,002 | 217 | 1,114 |
| p128 decode | 1 | 509 | 28 | 141 |
| c0 prefill | 1 | 509 | 28 | 141 |
| c0 decode | 64 | 32,576 | 1,792 | 9,024 |
| c512 prefill | 63 | 15,978 | 865 | 4,450 |
| c512 decode | 64 | 32,576 | 1,792 | 9,024 |

All grouped operations execute on PL with no fallback. Sixteen-column groups use
`pl/staged`; one-column tails and decode use `pl/direct`. Each first projection
raw-loads the activation and each second projection reuses it with zero activation
beats. The group buckets close exactly:

| Group bucket | Calls | Runs | A beats | R beats | W beats |
|---|---:|---:|---:|---:|---:|
| p128 staged main | 216 | 864 | 3,538,944 | - | - |
| p128 direct tail | 1 | 2 | 1,024 | - | - |
| c512 staged main | 864 | 3,456 | 14,155,776 | - | - |
| c512 direct tail | 1 | 2 | 1,024 | - | - |
| c512 decode `6144x1x2048` | 1,792 | 3,584 | 1,835,008 | 11,010,048 | 110,100,480 Q1 / 198,180,864 Q2 |

The decode bucket reports zero host quantize/pack time. Attention is unmodified
from P2c, stays on its exact expected staged/direct PL paths, has zero K/V/O stalls,
and differs by at most 0.01% in matched cycle comparisons. The known approximate-
flash verification advisory appears on 12 smoke calls: 40 of 24,576 values, maximum
absolute difference 0.5542, and maximum normalized difference 0.0467. Model logits
still pass, and the artifact contains no GEMM, grouped-command, Q8, DMA, kernel, or
request errors. The advisory therefore remains diagnostic evidence for the
unchanged approximate attention path rather than a P2d failure.

The single-repeat matched decode check shows no device regression; all four
observations are lower than P2c:

| Context bucket | Q1 P2c -> P2d | Q2 P2c -> P2d |
|---|---:|---:|
| p128 | 73.282 -> 73.028 ms/token (-0.35%) | 96.143 -> 94.659 ms/token (-1.54%) |
| c512 | 96.093 -> 94.773 ms/token (-1.37%) | 119.008 -> 117.700 ms/token (-1.10%) |
| c0 P2d anchor | 65.716 ms/token | 88.615 ms/token |

The grouped kernel itself is not faster than two primitive projections: raw PL
quantization increases grouped cycles by 25.98%/21.87% for Q1/Q2 staged columns-16
and 18.28%/8.96% for c512 decode. Removing one command per pair and the grouped
host quantize/pack stage is consistent with offsetting that cost, but this
one-repeat gate does not establish a repeatable speedup.

P128 prefill wall is 47.259/91.355 s for Q1/Q2 and c512 is 224.114/411.642 s.
The download comparison explains the wall result:

| Workload/format | Bytes | P2c download | P2d measurement |
|---|---:|---:|---:|
| p128 Q1 | 328.9 MiB | 8.1 s at 40.5 MiB/s | 36.5 s at 9.0 MiB/s |
| p128 Q2 | 655.9 MiB | 15.3 s at 43.0 MiB/s | 79.9 s at 8.2 MiB/s |
| c512 Q1 | 1.4 GiB | 35.0 s at 41.6 MiB/s | 176.5 s at 8.3 MiB/s |
| c512 Q2 | 2.8 GiB | 58.5 s at 49.7 MiB/s | 362.4 s at 8.0 MiB/s |

The volume is unchanged, so this wall regression is a degraded-network transport
measurement and is not attributed to the P2d accelerator change.

These results close P2d only for the bounded Qwen/Bonsai shapes, Q1/Q2 formats,
f285 image, and structural/numerical/device gates above. They do not close P2:
group results still cross DDR, and named FFN and attention section commands remain
future work.

## Implementation Gates

P2d preserves the two attention numerical gates established by P2b/P2c:

1. RTL versus `flash_ref` in hardware-approximation mode.
2. PL versus the PS/full-model oracle at the established tolerance.

The hardened flash-kernel and wrapper cosims preserve the exact query-one hashes
and cover production and non-power-of-two shapes, four-query Bonsai tiles,
two-query 32-head tiles, partial tiles, randomized stream bubbles, output
backpressure, stable stalled output, exact beat counts, one terminal `TLAST`, and
valid `TKEEP`. K/V beat counts are independent of query count within one tile.

The formal suite uses the real controller with fixed-latency arithmetic stubs. Its
compositional tasks prove scheduler tags and ordering, adaptive slot injectivity
and bounds at both production maxima, two-stage BRAM-read alignment, sparse-mask
advancement, II=1 softmax ordering, once-per-KV stream accounting, the KV barrier,
completion, and bounded liveness. All nine wired tasks pass against the same
production RTL used by cosim and OOC.

The aggregate `zig build test-rtl` target combines those tests with binary and
ternary GEMM, exact Q8 quantizer, and grouped raw-load/reuse cosim. It passes
locally; the committed P2d aggregate contains 36 build steps.

Use the full-kernel OOC probe for resource and isolated-Fmax feedback and the
combined routed build for timing signoff. P2c's bounded implementation, route,
identity, numerical, and profile gates are complete at f285. Do not infer combined
deployability from P2d's passing OOC results: both P2c f300 implementations and
the first P2d f285 implementation failed their clean combined routes despite
passing isolated probes. Repaired P2d now passes its clean f285 route, deployment,
live identity, Q1/Q2 logit gates, exact structural accounting, and device-regression
check.
P2d is closed at that bounded qualification. P2 remains open; the next work is
scratch-backed GEMM output layouts followed by the first named FFN section.
