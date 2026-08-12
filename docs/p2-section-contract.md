# P2 Section And Scratch Contract

Status: P2a contract v1 with qualified P2b decode scheduler, 2026-08-12

This document fixes the boundary for P2 implementation. It does not add a wire
operation. P2b changes the internal attention schedule without changing that
ABI, and later RTL revisions must preserve this ownership, layout, numerical,
fallback, and measurement contract.

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
| Query tile | 4 tokens |
| Model dimension | 4096 |
| FFN dimension | 12288 |
| Head dimension | 128 |
| Query heads | 32 |
| KV heads | 8 |
| Context | 8192 |

A longer prompt is processed as several tiles; scratch never scales with the full
prompt. The bitstream must advertise these caps. A descriptor outside them uses
the existing primitive path.

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
query_count = 2..4    prefill tile
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
position. It snapshots accepted configuration, tags dot/tree/score/softmax/AXPY
work by head, and uses a per-head score queue so those stages can overlap. The
controller retains a hard KV barrier: it does not advance to the next position
until every AXPY accumulator writeback for the current position has retired. This
preserves the `m`, `l`, and output-accumulator recurrence without serializing all
heads through the complete datapath.

The current composed softmax path is not a general II=1 pipeline. P2b spaces its
issues by eight cycles, above the proven five-cycle minimum; production head-128
scores arrive 16 cycles apart, so this does not limit single-query decode. A truly
II=1 softmax issue path, with explicit tag-alignment and consecutive-burst tests,
is a prerequisite for query-blocked attention where independent query/head work
can arrive in bursts.

The RTL's `n_tokens > 1` mode is still token-outer and rereads all K/V for every
query. It is not the contract's KV-outer query-blocked algorithm and must not be
enabled as the prefill hot path.

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

| Probe revision and target | DSP | LUT | FF | WNS | Estimated Fmax |
|---|---:|---:|---:|---:|---:|
| P2a diagnostic, 3.333 ns | 60 | 21,623 | 18,276 | -0.157 ns | 286.5 MHz |
| P2a temporary baseline, 3.600 ns | 60 | 21,623 | 18,276 | +0.110 ns | 286.5 MHz |
| P2b plus AXPY repair, 3.333 ns | 60 | 22,109 | 18,889 | +0.291 ns | 328.7 MHz |

This is a synthesis/OOC diagnostic, not routed signoff. P2b both restores the
3.333 ns registry target and reduces cycles per update without adding DSPs. The
small AXPY pipeline repair splits the two-DSP multiply path seen in the first P2b
combined route and preserves all cosim hashes. Its cosim, formal, OOC, and board
gates pass.

The released clean combined timing run reached setup and hold at +0.033/+0.007 ns
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

## Implementation Gates

P2b beats the single-query baseline while preserving two distinct numerical gates:

1. RTL versus `flash_ref` in hardware-approximation mode.
2. PL versus the PS/full-model oracle at the established tolerance.

The hardened flash-kernel and wrapper cosims now cover exact output hashes,
production and non-power-of-two shapes, multiple tokens, randomized stream
bubbles, output backpressure, stable stalled output, exact beat counts, one
terminal `TLAST`, and valid `TKEEP`. The flash control proof covers tag ordering,
addresses, GQA mapping, the KV barrier, accepted-configuration snapshots,
framing, stream accounting, and minimum softmax issue spacing with arithmetic
leaves stubbed.

The aggregate `zig build test-rtl` target combines those tests with binary and
ternary GEMM cosim and passes both locally and through the Nix
`checks.rtl-cosim` check.

Query blocking must add a four-query causal case and consecutive independent
softmax bursts after the path is truly II=1. K/V beat counts must then be
independent of query count.

Use the full-kernel OOC probe for resource and isolated-Fmax feedback and the
combined routed build for timing signoff. P2c query-blocked PL prefill attention
is the next implementation slice: first make softmax issue truly II=1, then add
the four-query KV-outer schedule while preserving the query-one hashes and
barriers. The current context-2048 prefill spends 1544.2 of 1725.8 device seconds
in PS attention; that is 91.7% of summed profiled op-device time. It therefore
precedes Q8 ingress, scratch-backed GEMM boundaries, and the FFN section.
