# llama.cpp sampling offloading (backend sampling)

What we know about doing token sampling on the device instead of downloading full
logits to the host every decode step, and what penzai actually ships.

## TL;DR

- llama.cpp has a "backend sampling" feature: the sampler chain becomes part of the
  model compute graph, so sampling runs on the backend (our FPGA device) and, in the
  best case, only the sampled token id is transferred to the host.
- **Stock greedy backend sampling does NOT reduce transport on its own.** The greedy
  sampler leaves `data.logits` non-null, so `build_sampling` records it and llama.cpp
  copies the full-vocab logits back every decode as `sampled_logits` — same bytes,
  different tensor name. This is true at our pinned revision *and* on current upstream
  master (not a pin-bump fix).
- Penzai uses a terminal greedy sampler implemented through llama.cpp's public
  `llama_sampler_i` interface. It clears the consumed logits output and, for the
  exact single-row sampling topology, connects `ggml_argmax` to the real logits
  before the unused PAD/view can become reachable.
- On the P0 bitstream, removing PAD reduced profiled device time by 5.68 ms/token for
  Q1 and 5.38 ms/token for Q2. Decode still returns only the 4-byte sampled token,
  and greedy output is byte-identical to the canonical P0 regression.

Pinned llama.cpp: `github:ggml-org/llama.cpp/e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`.
All file:line references below are approximate because this document spans the
original investigation and the later P1a implementation.

## What the feature is

The feature commit (`d3dce4e0`, "sampling : add support for backend sampling") frames
it as: sampler ops become part of `build_graph`, and

> "the backend sampler chain **might** select/sample a token directly **in which case
> only the sampled token needs to be transferred** ... It is also possible for the
> backend samplers to perform filtering of the logits ... in which case only the
> filtered logits or probabilities need to be transferred back."

Note the conditional. "Only the sampled token" is a *capability*, not a guarantee —
it holds only when the terminal sampler leaves nothing else behind to transfer.

API (in `include/llama.h`):
- `llama_context_params.samplers : llama_sampler_seq_config *` + `.n_samplers : size_t`.
- `llama_sampler_seq_config { llama_seq_id seq_id; llama_sampler * sampler; }` — binds a
  chain to a sequence. The sampler **must** be a `llama_sampler_chain`
  (`llama_sampler_chain_init`), set on `ctx_params` **before** `llama_init_from_model`
  so the sampling graph is known at graph/output reserve.
- `llama_sampler_sample(smpl, ctx, -1)` returns the backend-sampled token immediately if
  one was produced (skips CPU samplers).

## The mechanism (how outputs get transferred)

`src/llama-graph.cpp::build_sampling()` (≈:2811):

1. `ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);` — appends one
   dummy row. Comment: "this trick makes the graph static, regardless of which samplers
   are activated." (See **The PAD tax** below.)
2. For each sampler, build `data = { .logits = view(logits_t, row), .probs=null,
   .sampled=null, .candidates=null }` and call `sampler->iface->backend_apply(...)`.
3. Record **every non-null field** as a graph output — four independent `if`s (not
   `else if`, no suppression):
   ```
   if (data.sampled    != nullptr) res->t_sampled[seq]        = data.sampled;
   if (data.probs      != nullptr) res->t_sampled_probs[seq]  = data.probs;
   if (data.logits     != nullptr) res->t_sampled_logits[seq] = data.logits;   // :2873
   if (data.candidates != nullptr) res->t_candidates[seq]     = data.candidates;
   ```

`src/llama-context.cpp`:
- `needs_raw_logits()` (:1516) returns false when *every* output sequence has a backend
  sampler → the normal raw-logits copy is skipped.
- Backend outputs are copied **independently** (:1813): if `t_sampled_logits` is
  non-empty, `copy_tensor_async_floats` copies `ggml_nbytes(tensor)` per row.
- `copy_tensor_async_floats` (:1454) early-returns only if `!dst.has_data()` — i.e. if
  the host destination buffer wasn't allocated.
- `has_sampling = !sampling.samplers.empty()` (:1915). When true, the host allocates
  `sampling.logits/probs/candidates` at **full** `n_vocab * n_outputs_max` (:1973–1997);
  otherwise they're null (:1999). So the `has_data()` gate is **always open** once any
  backend sampler exists — it does not save greedy.

For penzai's backend the "copy to host" is `ggml_backend_tensor_get_async`, which is a
TCP download (`bufGetTensor` in `host/backend.zig`).

## The critical finding: which samplers actually reduce transport

A sampler reduces transport only if, after the chain, the recorded tensors are *small*.
What each stock `*_backend_apply` leaves in `data` (`src/llama-sampler.cpp`):

| sampler | effect on `data.logits` | reduces transport? |
|---|---|---|
| `greedy` (:984) | sets `data.sampled = argmax`, **leaves logits full-vocab** | ❌ full logits copied as `sampled_logits` |
| `dist` (:1144) | sets `sampled` + **`probs`** (:1197/1198), leaves logits | ❌ *worse* — logits **and** probs |
| `temp`, `temp_ext`, `logit_bias` | scale/add in place, full width | ❌ full width |
| `top_p`, `min_p` | mask, full width (static graph can't shrink) | ❌ full width |
| **`top_k`** (:1302) | **reshapes `data.logits` to `k`** | ✅ only `k` values |

**No stock sampler nulls `data.logits`.** Greedy keeps the full-vocab row, so the
"only the sampled token" promise is unreachable for greedy as written — confirmed
identical on upstream `master`, so a pin bump does not help. The only stock path that
shrinks transport is a `top_k`-led chain (e.g. `top_k(k); temp; dist`), which transfers
~`k` logits + `k` probs + `k` candidates + 1 token instead of the full vocab.

## The fix penzai ships (greedy)

P1a removed the downstream llama.cpp patch. `host/llama/chat.cpp` now owns a narrow
terminal greedy sampler using the public, experimental `llama_sampler_i` API:

1. CPU `apply`, `accept`, `reset`, and backend capability probing delegate to
   llama.cpp's stock greedy sampler; cloning creates the same stateless wrapper.
2. Backend `apply` emits the same `ggml_argmax`, sets `data.sampled`, and clears
   `data.logits`. `build_sampling` therefore records only the token result.
3. Before emitting ARGMAX, it recognizes only the exact zero-offset
   `VIEW(PAD(real_logits))` produced for one contiguous F32 logits row. The PAD must add
   exactly one right-side row, have no existing graph consumer, and no other sampler
   output may already be present. In that case ARGMAX reads `real_logits` directly.
4. Any shape, stride, offset, padding, graph-consumer, or sampler-output mismatch uses
   the supplied padded view. Generic `GGML_OP_PAD` support remains the fallback.

The sampler is selected only by Penzai's existing `--backend-sampling` path, whose
context has `n_seq_max = 1` and one terminal sampler bound to sequence 0. Normal host
greedy and logits-check modes continue using the stock sampler.

Static graph reuse remains valid: the sampler is installed before context creation,
the supported topology is identical on every one-output decode graph, and the emitted
ARGMAX keeps the real logits as an ordinary graph dependency. The matcher carries no
tensor pointers or state across graph builds. If a llama.cpp pin changes the topology,
the strict matcher fails closed and retains PAD rather than changing semantics.

## The PAD tax

`build_sampling` always emits `ggml_pad(res->t_logits, 0,1,0,0)` (a static-graph dummy
row) before the argmax. If the device does **not** support `GGML_OP_PAD`, ggml's
scheduler runs the pad on CPU, which means it downloads the full logits to feed it —
defeating the whole point. So the device **must** support both `PAD` and `ARGMAX` or the
logits leak to the host anyway.

The pad's dummy row is **never read** in Penzai's supported single-sequence path, so the
copy was pure overhead (~4 ms/token over the 606 KiB vocabulary row on the ARM PS).
The terminal sampler now makes PAD unreachable in that path without changing
llama.cpp. Unsupported sampling graphs still execute the existing PAD fallback.

`ggml_pad` op_params layout (relevant to lowering): `[lp0,rp0,lp1,rp1,lp2,rp2,lp3,rp3,
circular]`; the CPU ref writes src at index `(i - lp)`, so the data is front-aligned
(src bytes then a zero region) only when every **left** pad is 0 and not circular.
`host/lower.zig::supportsPad` enforces exactly that; the kernel is then "copy src, zero
the tail".

## What penzai implements

- **Flag:** `--backend-sampling` (`host/main.zig` → `host/run.zig` → `host/llama.zig`),
  greedy only. Builds the chain bound to `seq_id = 0` via `ctx_params.samplers` before
  context init. **Not** enabled in `penzai logits` mode (that path needs full logits).
- **Device ops:**
  - `device/ps/select.zig` — `argmax`. Single `@Vector(8, f32)` pass; tie-break is
    **last-max-wins** to match `ggml_vec_argmax_f32` (verified bit-identical to CPU
    greedy on the real 151669-vocab model). `ggml_argmax` requires a matrix src and
    returns 1-D i32 of length `ne[1]`.
  - `device/ps/pad.zig` — `padZeroTailBytes` remains for unsupported/fallback graphs.
  - Lowered in `host/lower.zig` (`supportsArgmax`/`supportsPad`), wire tags in
    `shared/protocol/wire.zig` (`argmax=16`, `pad=17`), dispatch in
    `device/runtime.zig`, accounting in `device/profile.zig`.
- ARGMAX/PAD originally entered the wire protocol in version 10; the current protocol
  is version 12. P1a changes only host graph construction and does not require a daemon
  or bitstream redeploy.

## Performance: the SIMD lesson and measured results

pad+argmax run on the **slow ARM PS** over the *full* 151k-vocab row, so kernel quality
matters. The first board run was a **net regression**: the argmax kernel used a
`std.mem.readInt`-per-element scalar loop that ran at 34 MiB/s — 12× slower than
`@memcpy`'s 423 MiB/s over the same buffer — costing **16.9 ms/tok**. Rewriting it as a
vectorized `@Vector` pass dropped it to **2.2 ms/tok**. Lesson: any full-vocab PS kernel
must be SIMD/`@memcpy`-class, not a scalar `readInt` loop.

Bonsai-1.7B-Q1_0, `tcp:kria`, `--max-tokens 100` (49 generated):

| metric | baseline (no flag) | backend, scalar argmax | backend, SIMD argmax |
|---|---|---|---|
| decode download | 28.3 MiB (592 KiB/tok) | 196 B | 196 B |
| `argmax` ms/tok | – | 16.9 | 2.2 |
| `pad` ms/tok | – | 4.1 | 4.1 |
| `device_ms` | 128.0 | 148.9 | 134.2 |
| `transport_ms` | 25.7 | 13.5 | 12.7 |
| **decode ms/tok** | 155.6 | 164.1 | **147.7** |
| **tok/s** | 6.43 | 6.08 | **6.77** |

Output is byte-identical across all three (greedy is deterministic). The fake device
(`--device fake`) reproduces the **byte** reduction exactly (592 KiB/tok → 4 B/tok) since
fake vs tcp differ only in timing — useful for local correctness/transport checks, not
for timing.

## P1a result and the ceiling

- Fake full-stack Q1 and Q2 graphs contain ARGMAX and no PAD for the supported path;
  each graph downloads 4 bytes and produces the same tokens as host greedy sampling.
- The three-repeat profiled board artifact is
  `20260805T031434Z-characterize-290c86efece2`. Every Q1/Q2 sample validated with
  64 ARGMAX operations, zero PAD operations, 256 decode-download bytes (4 bytes per
  token), and closed accounting. Compared with the 20-repeat P0 profiled baseline,
  median device time fell from 83.35 to 77.66 ms/token for Q1 and from 104.87 to
  99.49 ms/token for Q2. The 5.68/5.38 ms reductions include the removed 4.1 ms PAD
  plus small surrounding command/accounting savings.
- The five-repeat unprofiled artifact is
  `20260805T032338Z-regression-923832fe70a8`. Q1 steady decode improved from 99.00
  to 95.95 ms/token. Q2 measured 118.88 versus 118.42 ms/token in the canonical P0
  artifact, a 0.45 ms regression despite the clear profiled device reduction; its
  decode-wall median was effectively flat (118.59 versus 118.77 ms/token). Treat Q2
  product throughput as unresolved run variance, not as a claimed P1a speedup.
- Generated Q1 and Q2 text is byte-identical across the profiled artifact, the new
  unprofiled artifact, and the canonical P0 regression. The same deployed P0 daemon,
  wire ABI, and bitstream were used; P1a required no board redeploy.
- **Ceiling:** the remaining ARGMAX costs about 2.2 ms/token, but moving it into the PL
  GEMM epilogue is outside P1a. The larger remaining device budget is in projection,
  attention, and elementwise work tracked by later priorities.

## Future: real (non-greedy) sampling

A `top_k`-led chain is the only stock config that both reduces transport *and* gives
real sampling. To support `top_k(k); temp; dist` on-device you'd add (same
lower→wire→runtime→PS chain as argmax/pad): `top_k` (the only width-narrowing op),
**i32** `get_rows` (`host/lower.zig` only does f32 today), `scale`, and `dist`'s
`softmax`/`cumsum`/threshold plus RNG fed via `backend_set_input`. Then transport is
~`k` floats + `k` candidates + token. That path needs its own explicit terminal-output
contract; the P1a sampler intentionally remains greedy-only.

## How to run / verify

```
# build + unit tests (llama-free)
zig build test

# unmodified pinned llama.cpp + host sampler
nix build .#penzai

# local correctness + byte proof (fake device, real model)
nix run .#penzai -- run --device fake -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --raw-prompt --prompt "Once upon a time" --max-tokens 24 --backend-sampling   # vs without

# real board (P1a changes host code only; keep the deployed P0 daemon/bitstream)
nix run .#penzai -- run -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --device tcp:kria:29092 --prompt "hello" --max-tokens 100 --prof --backend-sampling
```

Confirmation signals: `download decode` collapses to a few bytes; `argmax` appears and
`pad` is absent from the `ops · decode` table; generated text matches the non-flag run.

See also `docs/handoff-backend-sampling-transport.md` (the original investigation/plan).
