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
