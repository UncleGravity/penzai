# llama remote buffer e2e

End-to-end proof that llama.cpp can use a Zig-implemented ggml backend with a
non-host buffer type during `llama_decode`.

This intentionally does not implement accelerator math yet. The custom backend
advertises itself as a GPU-like device, accepts scheduler ops, stores tensors in
a fake remote buffer (`is_host=false`), and delegates the assigned graph split to
ggml's CPU backend over shadow bytes. That keeps the experiment focused on the
host/device memory seam:

```
GGUF load -> llama tensor upload -> non-host panzai buffer -> scheduler -> Zig backend -> logits
```

Run:

```sh
nix run .#e2e
```

The flake pins both llama.cpp and a tiny GGUF fixture. The test compares one
decode against CPU-only and fails if llama did not initialize/use the Zig backend,
did not upload/download through the fake remote buffer, or if logits drift beyond
a small tolerance.

## Findings

- llama.cpp can load tensors into a Zig backend buffer with `is_host=false` and
  still complete `llama_decode` through the scheduler.
- ggml still requires `get_base` to return a stable non-null address for buffer
  allocation bookkeeping; this experiment uses shadow bytes behind fake remote
  handles.
- Non-host preallocated weight tensors are checked with `GGML_OP_NONE`, so the
  backend must accept `GGML_OP_NONE` for tensors it owns or scheduler reserve
  aborts.
- Passing run: `non_host_alloc=2`, `remote_upload_bytes=131328`,
  `remote_download_bytes=11520`, `graph_compute=11`,
  `logits max_abs_diff=0.000000`.
