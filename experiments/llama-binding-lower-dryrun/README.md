# llama binding lower dryrun

End-to-end proof that panzai can model ggml metadata ops as remote tensor
descriptors, then dry-run lower real work into command records.

This experiment ports the key old `pynqz1` idea into Zig:

- the backend exposes a non-host buffer type;
- `supports_buft` accepts only that panzai buffer type, not host buffers;
- each ggml buffer allocation gets one fake remote handle;
- `init_tensor` binds every ggml tensor to `(handle, offset, nbytes)`;
- views reuse the source handle and add `view_offs`;
- `graph_compute` skips metadata ops and emits dry-run command descriptors only
  for real work.

The dry-run command encoder currently emits `MUL_MAT` descriptors only. It still
delegates math to ggml CPU over shadow bytes so the test can compare logits
against CPU-only.

Run:

```sh
nix run .#e2e
```

The test fails if:

- llama does not allocate/use the non-host panzai buffer;
- `init_tensor` does not bind normal tensors and view tensors;
- panzai graph splits contain unsupported compute ops;
- a dry-run `MUL_MAT` command is missing source or destination bindings;
- logits differ from CPU-only.

## Relation to old pynqz1

The original backend did this in `host/backend/src/buffer.cpp` and
`host/backend/src/lowering.cpp`: metadata ops were accepted for scheduler
placement, represented through remote bindings and view offsets, skipped during
lowering, and never sent as board graph ops.

This experiment proves the same shape before we build `shared/wire.zig` or real
device kernels.

## Current findings

The first version accidentally accepted host buffers in `supports_buft`, inherited
from the earlier remote-buffer smoke test. That let the scheduler assign matmuls
whose inputs were still host/CPU tensors, so dry-run lowering saw
`dryrun_missing_bindings=46`.

Matching the old `pynqz1` policy fixed it: panzai supports only its own remote
buffer type. With that change, ggml materializes the needed tensors into panzai
buffers, every backend matmul has complete source/destination bindings, and
metadata nodes remain descriptor-only.

Passing run:

- `init_backend=1`
- `supports_op=1112`
- `accepted_mul_mat=277`
- `accepted_metadata=6`
- `accepted_none=6`
- `rejected=829`
- `offload_op=210`
- `offloaded_mul_mat=210`
- `graph_compute=46`
- `graph_nodes=66`
- `graph_metadata=20`
- `metadata_bound=20`
- `graph_mul_mat=46`
- `graph_unexpected=0`
- `dryrun_commands=46`
- `dryrun_missing_bindings=0`
- `init_tensor=198`
- `normal_bindings=123`
- `view_bindings=75`
- `binding_overflow=0`
- `non_host_alloc=2`
- `remote_upload_bytes=1396656`
- `remote_download_bytes=214112`
- `logits max_abs_diff=0.000000`

Design rule: a real panzai backend should not claim host buffer support unless it
can truly lower commands against host-resident tensors. For the planned remote
runtime, `supports_buft` should be strict, and CPU/host boundaries should appear
as explicit scheduler copies/materializations into panzai-owned descriptors.
