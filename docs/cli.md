# CLI

The host CLI accepts one TCP device form: `tcp:HOST:PORT`. Its default is
`tcp:127.0.0.1:29092`.

## Commands

```text
penzai run -m MODEL.gguf [OPTIONS]
penzai serve -m MODEL.gguf [OPTIONS]
penzai benchmark inference -m MODEL.gguf [OPTIONS]
penzai benchmark hardware -m MODEL.gguf [OPTIONS]
penzai verify logits -m MODEL.gguf [OPTIONS]
penzai inspect device [--device tcp:HOST:PORT]
penzai help
```

`run` generates one greedy sequence. `serve` replaces the CLI process with the
patched `llama-server` and fixes executor parallelism at one. `benchmark
inference` reports end-to-end and aggregate device timing. `benchmark hardware`
also prints every production recorder counter for prefill and decode. `verify
logits` compares the FPGA winner and winning logit with llama.cpp's CPU path.

Common inference options:

| Option | Default | Meaning |
| --- | --- | --- |
| `-m`, `--model` | required | GGUF model path |
| `--device` | `tcp:127.0.0.1:29092` | daemon endpoint |
| `--prompt` | `Hello` | input text |
| `--prompt-tokens` | tokenized length | repeat the prompt tokens to an exact test length |
| `--max-tokens` | `16` | maximum generated or verified steps |
| `--context` | `4096` | llama context size |
| `--batch` | `32` | logical prompt batch size |
| `--ubatch` | `16` | physical prompt batch size |
| `--metrics` | command-specific | `none`, `summary`, or `full` |
| `--raw-prompt` | off | bypass the model chat template |
| `--think` | off | enable template thinking mode |
| `--exact-tokens` | off | do not stop at end-of-generation tokens |

`run` also accepts `--token-ids`. `verify logits` accepts `--tolerance`, which
defaults to `0.25`. `serve` accepts the HTTP listener options `--host` and
`--port`; they default to `127.0.0.1:8080`. Its context, batch, and microbatch
defaults are `4096`, `32`, and `16`, and it accepts `--parallel 1` only.
Serving is fixed to metrics `none`; `--metrics summary` and `full` are rejected
until llama-server has an observable metrics endpoint.

`benchmark inference` and `benchmark hardware` raise `--metrics none` to
`summary`. `--metrics full` is reserved for a future versioned diagnostic-bank
contract and is rejected by current software.

## llama.cpp

The Nix wrappers select the dynamic backend and required executor flags. They
use context, batch, and microbatch defaults of `4096`, `32`, and `16`; later
command-line arguments may override those sizes:

```sh
nix run .#llama-cli-penzai -- \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  -p "hello" -n 64 -no-cnv

nix run .#llama-server-penzai -- \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --host 127.0.0.1 --port 8080
```

The packaged `penzai serve` launches that same server:

```sh
nix run .#penzai -- serve \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --host 127.0.0.1 --port 8080
```

These commands expect `nix run .#serve-penzaid` to remain attached in another
terminal. That helper owns the SSH tunnel for the default localhost endpoint.

The executor supports one causal text sequence, greedy sampling, and one winner
logit. Context shifting, prompt/KV reuse, state save and restore, embeddings,
encoders, speculative decoding, multimodal input, adapters, control vectors,
and backend samplers are rejected.

The 4096-token default fits the 1500 MiB development heap for 1.7B Q1/Q2 and
4B Q1. The larger 4B Q2 and 8B Q1 resident images require an explicit smaller
`--context` value; the qualified 8B Q1 configuration uses 513 tokens.
