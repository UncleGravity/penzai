# llama backend e2e

End-to-end proof that llama.cpp can use a Zig-implemented ggml backend during
`llama_decode`.

This intentionally does not implement accelerator math yet. The custom backend
advertises itself as a GPU-like device, accepts scheduler ops, and delegates the
assigned graph split to ggml's CPU backend. That keeps the experiment focused on
the integration seam:

```
GGUF load -> llama graph construction -> llama scheduler -> Zig ggml backend -> logits
```

Run:

```sh
nix run .#e2e
```

The flake pins both llama.cpp and a tiny GGUF fixture. The test compares one
decode against CPU-only and fails if llama did not initialize/use the Zig backend
or if logits drift beyond a small tolerance.

## Findings

- llama.cpp can use a Zig-implemented ggml backend during real `llama_decode`
  without `.so`, `dlopen`, env vars, or global backend discovery.
- Passing the device through `llama_model_params.devices` is enough for llama to
  initialize the backend, include it in the scheduler, and assign graph splits.
- A CPU-delegating Zig backend is sufficient to prove the integration seam before
  implementing accelerator math.
- The CPU-only and mixed-backend runs must use matching graph modes. With Flash
  Attention left on `AUTO`, llama enabled it for CPU-only and disabled it for the
  mixed backend, causing small logit drift; pinning both to disabled produced
  exact logits.
- Passing run: `init_backend=1`, `supports_op=1136`, `accepted_mul_mat=277`,
  `graph_compute=11`, `logits max_abs_diff=0.000000`.
