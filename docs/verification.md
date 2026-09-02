# Verification

Verification is layered. Deterministic host checks establish software and
protocol behavior without a board. RTL checks establish source closure and
logic properties. Only a routed bitstream exercised on the KR260 can qualify a
production image.

## Software checks

Run the Zig contract, host, daemon, memory, and register-map tests:

```sh
nix develop -c zig build test
```

This includes metrics encoding and driver snapshot tests against a mock MMIO
window.

Build the packaged host and device programs:

```sh
nix build .#penzai .#penzaid
```

Exercise the real patched `llama-cli` and `llama-server` binaries through the
dynamic Penzai backend:

```sh
nix build ".#checks.$(nix eval --impure --raw --expr builtins.currentSystem).llama-inference"
```

That smoke check generates a tiny GGUF and uses a deterministic executor. It
proves backend discovery, model and session lifecycle, tiled prefill, monotonic
commits, greedy CLI generation, HTTP completion, rejection of unsupported
server requests, and the `penzai serve` launcher's explicit context plumbing.
It does not exercise TCP, XRT, DDR, the FPGA datapath, real Bonsai
weights, numerical accuracy, timing, or power.

Evaluate every flake output without building it, or run all checks exported for
the current system:

```sh
nix flake check --no-build
nix flake check
```

## RTL checks

The default RTL gate validates the closed production source manifest, lints the
complete top, runs all registered simulations, checks the production Yosys map
and resource invariants, and runs the fast formal proofs:

```sh
nix develop -c zig build verify-rtl
```

The components can be run independently:

```sh
nix develop -c zig build lint-rtl
nix develop -c zig build test-rtl
nix develop -c zig build synth-rtl
nix develop -c zig build formal
```

Run the extended structural maps and deeper proofs before changing a shared
datapath or control contract:

```sh
nix develop -c zig build synth-rtl-all
nix develop -c zig build formal-all
```

Every entry in `fpga/verify/suites.zig` is also available as
`zig build verify-<suite>`, for example:

```sh
nix develop -c zig build verify-engine-metrics
nix develop -c zig build verify-datapath
nix develop -c zig build verify-formal-attention-kernel
```

These checks cover RTL behavior and structure, including simulation and formal
properties for the metrics recorder. They do not establish routed timing or
board-level numerical correctness.

## Bitstream qualification

Regenerate the shared register artifacts, run the default RTL gate, then create
a clean qualified-frequency build:

```sh
nix develop -c zig build regmap
nix develop -c zig build verify-rtl
cd fpga/build
cp config.env.example config.env
# Set BOARD, BOARD_TMP, VM, and VM_DIR in config.env.
./build.sh f225
./deploy.sh f225
```

`build.sh` promotes a bitstream only from a completed Vivado run bundle.
`deploy.sh` verifies that bundle and bitstream hashes, loads the image, and
installs the receipt read by `penzaid`.

Deploy and start the daemon in separate host invocations:

```sh
nix run .#deploy-penzaid
# In a second terminal; this remains attached until Ctrl-C.
nix run .#serve-penzaid
```

Confirm the running hardware identity before testing inference:

```sh
nix run .#penzai -- inspect device
```

The report must show a loaded receipt, a verified bitstream hash, wire ABI 18,
metrics schema 1, engine ID `0xB05A4000`, engine interface `0x00010007`, and
summary metrics support without the diagnostic full-bank capability.

## Model qualification

Stop the attached 1500 MiB development daemon with Ctrl-C, then restart it with
the exact largest heap used by this qualification matrix:

```sh
PENZAI_HEAP_MIB=1392 nix run .#serve-penzaid
```

Compare the FPGA winner and winning logit against llama.cpp's CPU path for each
of the five board-runnable model and weight combinations:

```sh
for model in \
  models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  models/Bonsai-4B/Bonsai-4B-Q1_0.gguf \
  models/Bonsai-8B/Bonsai-8B-Q1_0.gguf \
  models/Bonsai-Ternary-1.7B/Ternary-Bonsai-1.7B-Q2_0_g64.gguf \
  models/Bonsai-Ternary-4B/Ternary-Bonsai-4B-Q2_0_g64.gguf
do
  nix run .#penzai -- verify logits \
    -m "$model" \
    --prompt "hello" --max-tokens 8 --exact-tokens --tolerance 0.25
done
```

Exercise the qualified 8B Q1 context configuration separately, still using the
1392 MiB heap, with a 505-token prefill and eight verified winner checks:

```sh
nix run .#penzai -- verify logits \
  -m models/Bonsai-8B/Bonsai-8B-Q1_0.gguf \
  --prompt "hello" --prompt-tokens 505 \
  --context 513 --max-tokens 8 --exact-tokens --tolerance 0.25
```

The 8B Q2_g64 artifact is engine- and file-format-compatible, but its planned
resident image is 2.338 GB and exceeds the qualified KR260 heap. Loading it is
a negative capacity check: allocation must fail before model publication, and
no execute request may reach the engine. It is not a sixth board-runnable
accuracy case.

Finish qualification with both stable performance views on the intended model,
prompt length, and generation length:

```sh
nix run .#penzai -- benchmark inference \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --prompt "hello" --max-tokens 64

nix run .#penzai -- benchmark hardware \
  -m models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf \
  --prompt "hello" --max-tokens 64
```

`benchmark hardware` uses `summary`: it reads all 40 metrics in the lean
production recorder. `full` is reserved for a separately versioned diagnostic
bank and must be rejected by the current capability check.

Record the inspection identity with benchmark results. A production promotion
requires the local software and RTL gates, a clean timing-passing `f225` build,
verified deployment identity, zero winner mismatches, winner-logit differences
within tolerance, and no engine fault or counter-overflow indication during the
benchmark run.
