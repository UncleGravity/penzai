# llama experiments

Shared llama.cpp backend experiments for penzai.

Run one experiment from this directory:

```sh
nix run .#backend-e2e
nix run .#remote-buffer-e2e
nix run .#partial-offload
nix run .#op-census
nix run .#binding-lower-dryrun
```

The flake pins llama.cpp and the tiny GGUF fixture once. The Zig build shares the
C translation layer and builds one executable per experiment source file.

Per-experiment notes live in `docs/`.
