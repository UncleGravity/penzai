# Penzai

1-Bit / 1.58-Bit LLM inference accelerator on FPGA (KR260)

## Load The Bitstream

```bash
# Check if it's loaded
ssh ubuntu@kria 'sudo xmutil listapps'

# If `penzai` does not show `0->0`, load it:
ssh ubuntu@kria 'sudo xmutil loadapp penzai'
```

## Run a model
```sh
# Terminal 1
nix run .#deploy-penzaid # build and upload daemon
nix run .#serve-penzaid # run remote daemon
# OR run local daemon (for testing)
nix run .#penzaid-native -- serve --device tcp:127.0.0.1:29097 --mem fake

# Terminal 2 (choose 1 from below)

## Custom CLI
nix run .#penzai -- run \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --prompt "Write one sentence about FPGAs." \
  --max-tokens 64

## OR llama.cpp
nix run .#llama-cli-penzai -- \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  -no-cnv
```

## Model support

| Model | Parameters | Weight encoding | HuggingFace |
| --- | ---: | --- | --- |
| `Bonsai-1.7B` | 1.7B | `Q1_0` | [prism-ml/Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) |
| `Ternary-Bonsai-1.7B` | 1.7B | `Q2_0` (group 64) | [prism-ml/Ternary-Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) |
| `Bonsai-4B` | 4B | `Q1_0` | [prism-ml/Bonsai-4B-gguf](https://huggingface.co/prism-ml/Bonsai-4B-gguf) |
| `Ternary-Bonsai-4B` | 4B | `Q2_0` (group 64) | [prism-ml/Ternary-Bonsai-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf) |
| `Bonsai-8B` | 8B | `Q1_0` | [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) |
| `Ternary-Bonsai-8B` | 8B | `Q2_0` (group 64) | [prism-ml/Ternary-Bonsai-8B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf) |

- [Architecture](docs/architecture.md)
- [CLI and llama integration](docs/cli.md)
- [KR260 deployment](docs/deployment.md)
- [Metrics](docs/metrics.md)
- [Verification and qualification](docs/verification.md)

```sh
nix develop -c zig build test
nix develop -c zig build verify-rtl
nix flake check --no-build
```
