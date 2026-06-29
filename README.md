# penzai

1-Bit / 1.58-Bit LLM inference accelerator on FPGA (KR260)

## TLDR
```sh
# Terminal 1
nix run .#deploy-penzaid # build and upload daemon
nix run .#serve-penzaid # run remote daemon
# OR run local daemon (for testing)
nix run .#penzaid-native -- serve --device tcp:127.0.0.1:29097 --mem fake --heap-mib 768

# Terminal 2 (choose 1 from below)

## Custom CLI
nix run .#penzai -- run \
  -m /path/to/model.gguf \
  --device tcp:kria:29092 \
  --prompt "Write one sentence about FPGAs." \
  --max-tokens 64

## OR llama.cpp
PENZAI_HOST=kria PENZAI_PORT=29092 \
  nix run .#llama-cli-penzai -- \
    --device penzai \
    -ngl 999 \
    --no-op-offload -fa on \
    -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
    -p "hello" -n 64 -no-cnv
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
