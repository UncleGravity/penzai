# P2 Section And Scratch Contract

Status: P2a contract v1 with routed, deployed, and board-qualified P2c adaptive
attention at f285, 2026-08-12

This document fixes the boundary for P2 implementation. It does not add a wire
operation. P2b and P2c change the internal attention schedule and primitive bridge
without adding a named section ABI. Later RTL revisions must preserve this
ownership, layout, numerical, fallback, and measurement contract.

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

The padded 40-byte DMA stream is an ingress format, not an internal scratch type.
GEMM gains explicit `load_acts` and `reuse_acts` modes. Reuse is valid only for a
latched `{epoch, K, token_count}`; it is never inferred from a host address.

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

P2a deliberately does not assign wire tags or serialize these fields. Concrete
range, stride, RoPE, and weight-format types are frozen with the first executable
section so the ABI describes an implemented controller rather than a speculative
one.

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
it is the current qualification point, not evidence that f300 closes.

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

## Implementation Gates

P2b and P2c preserve two distinct numerical gates:

1. RTL versus `flash_ref` in hardware-approximation mode.
2. PL versus the PS/full-model oracle at the established tolerance.

The hardened flash-kernel and wrapper cosims preserve the exact query-one hashes
and cover production and non-power-of-two shapes, four-query Bonsai tiles,
two-query 32-head tiles, partial tiles, randomized stream bubbles, output
backpressure, stable stalled output, exact beat counts, one terminal `TLAST`, and
valid `TKEEP`. K/V beat counts are independent of query count within one tile.
The aggregate RTL suite passes 26/26 targets, alongside all 247 Zig tests.

The formal suite uses the real controller with fixed-latency arithmetic stubs. Its
compositional tasks prove scheduler tags and ordering, adaptive slot injectivity
and bounds at both production maxima, two-stage BRAM-read alignment, sparse-mask
advancement, II=1 softmax ordering, once-per-KV stream accounting, the KV barrier,
completion, and bounded liveness. All nine wired tasks pass against the same
production RTL used by cosim and OOC.

The aggregate `zig build test-rtl` target combines those tests with binary and
ternary GEMM cosim and passes both locally and through the Nix
`checks.rtl-cosim` check.

Use the full-kernel OOC probe for resource and isolated-Fmax feedback and the
combined routed build for timing signoff. P2c's bounded implementation, route,
identity, numerical, and profile gates are complete at f285. Do not infer f300
deployability from the passing OOC result: both P2c f300 implementations failed
their clean combined routes. The next P2 work is PL Q8 ingress and grouped
activation reuse, scratch-backed GEMM output layouts, and then the first named FFN
section.
