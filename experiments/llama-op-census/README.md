# llama op census

Runs a real `llama_decode` and records the ggml op patterns llama.cpp emits for
the pinned tiny GGUF fixture.

This experiment exists to inform `host/lower.zig`. It answers what op/type/shape
patterns the first lowering table actually has to support, instead of guessing
from memory or copying the old backend blindly.

Run:

```sh
nix run .#e2e
```

The census records three perspectives:

- `support` calls: ops llama's scheduler asked the panzai device about.
- `eval` calls: actual executed graph nodes observed via the scheduler eval
  callback.
- `panzai` calls: nodes that reached the Zig panzai backend's `graph_compute`.

The backend still delegates math to ggml CPU. This experiment is about graph
shape and scheduler placement, not accelerator kernels.

## Current findings

`nix run .#e2e` passes with the pinned tiny model. The run observed:

- `77` unique op/type/shape patterns.
- `1136` scheduler `supports_op` calls.
- `186` executed graph nodes via the scheduler eval callback.
- `176` nodes that actually reached the panzai backend `graph_compute`.
- `277` accepted `MUL_MAT` support checks.
- `0` census overflow.

The important detail is that `supports_op` is not a complete source of truth for
lowering. Some nodes reached backend execution with no support-call record in
this run, including reshape/view/permute glue and one RMS norm shape. The real
`lower.zig` table should therefore be validated against `graph_compute` nodes,
not only scheduler capability probes.

The first tiny-model op surface includes at least: `GET_ROWS`, `MUL`, `MUL_MAT`,
`SET_ROWS`, `RESHAPE`, `VIEW`, `PERMUTE`, `ROPE`, `SOFT_MAX`, `CONT`, `ADD`,
`GLU`, and `RMS_NORM`, plus `NONE` tensors for model weights and KV buffers seen
during support checks.
