# Architecture

Penzai runs a whole-token inference engine in the KR260 FPGA. The host owns
GGUF loading, tokenization, prompt formatting, and user-facing APIs; the board
owns the resident model image, KV state, scheduling, and token execution.

## Execution path

```text
penzai run / penzai serve / llama-cli-penzai / llama-server-penzai
                              |
                       patched llama.cpp
                              |
                    penzai.inference.v1
                              |
                  libggml-penzai backend
                   |                  |
            model packing       typed TCP client
                                      |
                         framed request/response wire
                                      |
                                   penzaid
                         |             |             |
                     XRT heap    model/session    MMIO driver
                                   manager            |
                                               resident FPGA engine
```

`penzai run` embeds the pinned llama.cpp library. `penzai serve` launches the
patched `llama-server`; the llama CLI and server wrappers load the same backend
as a dynamic GGML plugin. Every route therefore meets at the same
`penzai.inference.v1` contract.

The extension exposes six operations: load and unload a model,
open and close a session, reset a session, and execute a token tile. It is
deliberately limited to one causal, greedy sequence. The current engine accepts
up to eight tokens per tile: multi-token tiles perform prefill and one-token
tiles perform decode.

## Model and session lifecycle

1. The backend connects to `penzaid`, reads its capabilities, and rejects an
   incompatible wire, engine interface, or weight-format contract before
   provisioning data. The daemon validates the model-layout contract during
   installation.
2. llama.cpp supplies semantic GGUF tensor roles through
   `penzai.inference.v1`. The backend validates the supported Bonsai shape and
   Q1_0 or Q2_0 format, packs the tensors and RoPE table into the engine's fixed
   image layout, then allocates and uploads that image once. Format acceptance
   does not override the board's heap limit: the 8B Q2 image is valid for the
   engine but too large for the qualified KR260 memory configuration.
3. `penzaid` publishes the image in its resident model table. Opening a context
   allocates a board-side KV arena and creates an epoch-tracked session.
4. Each execute request carries token IDs and the expected committed position.
   The daemon acquires the session, runs the FPGA, and advances the visible KV
   watermark only after successful completion. A logits request returns the
   winning token and its logit.
5. Closing the context releases its KV arena. Unloading the model removes the
   resident image and releases its XRT allocation.

Weights cross TCP only while the model is provisioned. During inference the
host sends token tiles and receives compact results; the FPGA reads resident
weights and KV state directly from DDR.

## Responsibilities

| Area | Owner |
| --- | --- |
| CLI, prompt formatting, tokenization, HTTP serving | `host/cli`, `host/llama` |
| llama executor registration and lifecycle | `host/backend` |
| GGUF validation, image planning, and packing | `host/engine` |
| Typed RPC client and TCP transport | `host/link` |
| Cross-process contracts and metrics schemas | `shared` |
| Request dispatch, allocation, and transactional sessions | `device/daemon`, `device/mem`, `device/engine` |
| Datapath, register interface, build, and proof suites | `fpga/rtl`, `fpga/regmap`, `fpga/build`, `fpga/verify` |

## Versioned contracts

These versions protect different boundaries and must not be treated as one
global version:

| Boundary | Current version | Source |
| --- | ---: | --- |
| llama whole-token extension | ABI 1 (`penzai.inference.v1`) | `patches/llama-inference-v1.patch`, `host/llama/inference.h` |
| frame envelope | 1 | `shared/protocol/framing.zig` |
| host-daemon wire | 18 | `shared/protocol/wire.zig` |
| capability report | 5 | `shared/capabilities.zig` |
| resident model layout | 5 | `shared/engine/model_spec.zig` |
| execute-tile record | 2 | `shared/engine/command.zig` |
| inference metrics | 1 | `shared/engine/metrics.zig` |
| FPGA register interface | `0x00010007` | `fpga/regmap/engine.regmap` |

The register interface also fixes layout hash `0xC255C7A52FC14A79`. The daemon
checks the FPGA identity at startup, the backend checks the daemon capability
report at connection time, and model installation repeats the interface and
layout checks. `penzai inspect device` exposes this identity together with the
hash-verified bitstream deployment receipt.

Metrics use the same execution path. `none` skips indexed recorder reads.
Production advertises `summary`, which reports aggregate host and device timing
plus all 40 metrics in the lean recorder, including stage cycles and weight,
Q8, AXI, and pipeline stall accounting. `full` is reserved for a future
versioned diagnostic-bank contract; current software does not advertise it, so
an explicit `full` request fails closed.
