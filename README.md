# penzai

1-Bit / 1.58-Bit LLM inference accelerator on FPGA (KR260)

## TLDR
```sh
# Terminal 1
nix run .#deploy-penzaid # build and upload daemon
nix run .#serve-penzaid # run remote daemon

# Terminal 2
nix run .#penzai -- run \
  -m /path/to/model.gguf \
  --device tcp:kria:29092 \
  --prompt "Write one sentence about FPGAs." \
  --max-tokens 64
```

Local dev:

```sh
nix develop
zig build test
zig build all             # host + KR260 daemon + native daemon
```

## Profiling

```sh
# 
nix run .#penzai -- run \
  -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --device tcp:kria:29092 \
  --prompt "hello" \
  --prof # stats table
```
