# penzai

LLM inference on FPGA (KR260)

## TLDR
```sh
# Terminal 1
nix run .#deploy-penzaid # build and upload daemon
nix run .#serve-penzaid # run remote daemon
ssh ubuntu@kria "pkill -f '/tmp/penzai/penzaid serve'" # kill when done

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
```

## Profiling

```sh
# 
nix run .#penzai -- run \
  -m ./models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --device tcp:kria:29092 \
  --prompt "hello" \
  --prof # pretty table
  # OR view as json
  # --prof=json 2>/dev/null | tail -1 | jq 
```
