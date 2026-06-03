# llama partial offload

End-to-end proof of how llama.cpp behaves when the Zig panzai backend advertises
matmul plus metadata support, while selecting only matmul as real offloaded
work.

This experiment is intentionally stricter than `llama-backend-e2e`: the custom
backend advertises support for `GGML_OP_MUL_MAT` and no-cost metadata ops
(`NONE`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`), but its `offload_op`
callback returns true only for `GGML_OP_MUL_MAT`. It still delegates math to ggml
CPU once a graph split reaches `graph_compute`; the question here is scheduler
placement, not accelerator implementation.

Run:

```sh
nix run .#e2e
```

The test compares one decode against CPU-only and fails if:

- llama does not initialize/use the Zig backend.
- no `MUL_MAT` ops are accepted for panzai.
- panzai receives anything other than `MUL_MAT` or metadata ops in
  `graph_compute`.
- logits differ from the CPU-only baseline.

This is the experiment that decides whether the first `lower.zig` can start with
accelerator-worthy ops only, instead of owning every llama.cpp glue op on day
one.

## Current findings

Strict `MUL_MAT`-only offload is not the same as strict `MUL_MAT`-only backend
execution. In this llama.cpp scheduler path, `offload_op` is installed but was
not called at all; `supports_op` is the active placement signal. With
`supports_op` accepting `GGML_OP_MUL_MAT` plus metadata, the graph splits
assigned to panzai still include view-like metadata nodes around attention
matmuls.

The observed extra nodes are `RESHAPE` and `PERMUTE`, both of which are metadata
or layout glue rather than accelerator math. That means the first `lower.zig`
can still focus on matmul as the only real kernel, but it must classify metadata
nodes and absorb them through tensor views, offsets, shapes, and strides instead
of emitting device commands for them.

This matches the original `pynqz1` shape: metadata ops are accepted for scheduler
placement, then skipped during lowering. The callback difference is that
`offload_op` exists for parity with the old backend API, but this exact
`llama_decode` path did not exercise it.

## Decision for panzai

Metadata ops should not become wire/device commands. `lower.zig` should treat
them as tensor descriptor changes:

- `NONE`, `VIEW`, and `RESHAPE`: update/bind descriptor state only.
- `PERMUTE` and `TRANSPOSE`: descriptor-only when the downstream kernel accepts
  the resulting strides/layout.
- `CONT`, `CPY`, or an unsupported-stride boundary: real materialization command.

Real wire commands should represent work the device must execute. Metadata
effects belong in tensor descriptors: handle, offset, dtype, shape, strides, and
layout.

Passing run:

- `init_backend=1`
- `supports_op=1736`
- `accepted_mul_mat=277`
- `accepted_metadata=498`
- `accepted_none=348`
- `rejected=961`
- `offload_op=0`
- `offloaded_mul_mat=0`
- `offload_rejected=0`
- `offload_metadata_rejected=0`
- `graph_compute=36`
- `graph_nodes=66`
- `graph_mul_mat=46`
- `graph_metadata=20`
- `graph_reshape=15`
- `graph_permute=5`
- `graph_view=0`
- `graph_transpose=0`
- `graph_unexpected=0`
- `logits max_abs_diff=0.000000`
