# Bonsai Q1_0 real-model census

Run date: 2026-06-05.

Model:

```text
/Users/angel/Documents/asic/tt-tpu/models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf
```

The existing `op-census` and `binding-lower-dryrun` experiments were rebuilt
with `-Dmodel` pointed at the real Bonsai Q1_0 GGUF instead of the pinned tiny
fixture.

Both experiments still use the hardcoded prompt `Hello`, with `n_ctx=64`,
`n_batch=16`, `n_ubatch=16`, and Flash Attention disabled. The support probes
therefore include batch-16 shapes, but the evaluated graph is the short-prompt
decode path.

## op-census

Result: pass.

```text
summary: unique_patterns=66, support_calls=6776, eval_nodes=1126,
penzai_nodes=1014, graph_compute_calls=1014, accepted_ops=4574,
accepted_mul_mat=1519, overflow=0
```

Observed executed op surface:

- Metadata and tensor state: `NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `CONT`.
- Resident model / embedding ops: `GET_ROWS`, `MUL_MAT`.
- F32 glue: `MUL`, `ADD`, `GLU`, `RMS_NORM`, `ROPE`, `SOFT_MAX`, `SET_ROWS`.
- Attention-cache path: F16 KV-cache views/permutations, F16 x F32 attention
  `MUL_MAT`, and F16 cache writes through `SET_ROWS`.
- Weight matmul path: Q1_0 x F32 `MUL_MAT` for attention projections, FFN
  projections, token embedding/output shapes, and support-probed batch-16
  variants.

Important scheduler detail: several metadata patterns reached evaluation with
`support=0`, matching the earlier tiny-model finding that `supports_op` probes
are not a complete lowering source of truth. `lower.zig` should validate against
nodes observed at `graph_compute`, not only support probes.

## binding-lower-dryrun

Result: pass.

```text
penzai counters: init_backend=1, supports_op=5270, accepted_mul_mat=1519,
accepted_metadata=6, accepted_none=6, rejected=3745, offload_op=1176,
offloaded_mul_mat=1176, graph_compute=253, graph_nodes=365,
graph_metadata=112, metadata_bound=112, graph_mul_mat=253,
graph_unexpected=0, dryrun_commands=253, dryrun_missing_bindings=0,
init_tensor=1095, normal_bindings=675, view_bindings=420,
binding_overflow=0, alloc_buffer=2, non_host_alloc=2, set_tensor=422,
get_tensor=253, remote_alloc_bytes=53518560,
remote_upload_bytes=273294240, remote_download_bytes=3588564
logits max_abs_diff=0.000000
```

Implications:

- Strict penzai-only `supports_buft` still works on the real Bonsai model.
- Remote descriptor binding covers normal tensors and views without overflow.
- Metadata remains descriptor-only; no dry-run commands were missing handles.
- The first real `lower.zig` can start from complete descriptor binding plus
  explicit materialization for `CONT`/unsupported-stride boundaries.
- The device op roadmap should include Q1_0 matmul, F16 attention matmul or a
  CPU fallback boundary, `SET_ROWS`, `GET_ROWS`, `ROPE`, `SOFT_MAX`, `RMS_NORM`,
  `MUL`, `ADD`, and `GLU` if the goal is to keep full Bonsai layer regions
  resident.
