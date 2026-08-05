# P0 benchmark harness

P0 is a fixed Q1/Q2 performance harness for architecture decisions. It has two
canonical suites and deliberately does not serve as a general benchmark system.

## Characterize

`characterize` enables aggregate profiling and runs both models with this fixed
matrix:

| Workload | Repetitions |
|---|---:|
| Prefill 128 | 3 |
| Prefill 512 | 3 |
| Decode context 0 | 3 |
| Decode context 512 | 3 |
| Decode context 2048 | 1 |

Run it with:

```sh
nix run .#p0-benchmark -- characterize --device tcp:kria:29092
```

Context 4096 is a feasibility case for heap and bounds validation. It is not part
of the normal suite and must be requested explicitly:

```sh
nix run .#p0-benchmark -- characterize --case decode-c4096 --device tcp:kria:29092
```

## Regression

`regression` disables profiling and runs both models five times for prefill 128,
decode context 0, and decode context 512:

```sh
nix run .#p0-benchmark -- regression --device tcp:kria:29092
```

Use repeatable `--case NAME`, plus `--repeats`, `--batch`, and `--ubatch`, only
for focused diagnostics. Every supplied override is recorded in the immutable
artifact identity. A focused case replaces the suite matrix; it does not add to
it.

## Artifacts And Resume

Each ignored `runs/<UTC>-<suite>-<fingerprint>/` directory contains:

- `manifest.json`: one immutable identity object and its deterministic SHA-256
  fingerprint. The identity contains the schema and suite versions, exact cases
  and repetitions, batch settings, profiling mode, device endpoint, runner and
  executable hashes, model paths and hashes, overrides, and the complete starting
  device capability response.
- `samples.jsonl`: append-only accepted samples.
- `raw/*.txt`: the complete report for each attempted sample that reached the
  runner.
- `summary.json`: median/min/max latency summaries, completion status, and the
  completed, expected, and missing sample keys. It is refreshed after each
  accepted sample and on clean interruption. `complete` becomes true only after
  all samples and the ending identity checks pass.
- `capabilities-start.txt` and, when available, `capabilities-end.txt`: the raw
  daemon responses used to establish that the deployed system stayed fixed.

Resume requires the original suite and every original override:

```sh
nix run .#p0-benchmark -- characterize \
  --resume benchmarks/p0/runs/<run> \
  --device tcp:kria:29092
```

The runner reconstructs the identity and requires an exact identity and
fingerprint match. Changes to the runner, executable, models, workload, settings,
endpoint, or starting device capabilities reject resume. Old artifacts are never
migrated or rewritten.

## Results And Correctness

The summary aggregates only lower-is-better latency measurements: prefill wall,
compute and output TTFT, first decode step, steady decode, and decode
wall/device/transport/residual time per token. Each group reports the conventional
median and min/max range. Displayed throughput is derived as
`1000 / median steady latency`; it is not aggregated independently.

The harness validates exact requested prompt and generated token counts. That
proves workload completion, not numerical correctness. Profiled runs must also
close their accounting, while regression runs must report that profiling was off.
In an unprofiled machine result, device, transport, and residual sub-buckets are
zero sentinels meaning unavailable; use wall and steady-decode metrics for regression.

Logits and perplexity tests remain the model-level numerical gates. RTL tests,
formal verification, and cosimulation remain separate hardware correctness suites.
