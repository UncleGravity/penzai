# Metrics

Metrics are selected for runs and benchmarks with `--metrics
none|summary|full`. The selection changes recorder readout and transport only;
it does not change the compiled datapath counters.

| Level | Production behavior |
| --- | --- |
| `none` | validate the frozen recorder envelope and acknowledge it without indexed counter reads |
| `summary` | read and return the 40-counter lean production bank |
| `full` | reserved until a versioned diagnostic-bank payload exists; reject |

`run` defaults to `none`. `serve` is fixed to `none` because llama-server has no
current consumer for backend snapshots; server metrics require a future logging
or HTTP endpoint. Both benchmark commands default to at least `summary`.

## Output

Every completed `penzai run` or benchmark emits a stable `benchmark_result`
record. Instrumented runs also emit `metrics_summary schema=1` with prompt and
generation counts, prefill tiles, decode executions, host wall time, device
time, first-token time, output first-token time, and total engine cycles.

`benchmark hardware` additionally emits one `hardware_metric schema=1` record
per metric for each `prefill` and `decode` phase:

```text
hardware_metric schema=1 phase=prefill id=33 name=projection_drain_cycles value=123 overflow=false
```

The stable IDs cover:

- total and control cycles
- cycles and call counts for embedding, normalization, QKV/RoPE, KV append,
  attention, residual projections, feed-forward work, final normalization, and
  the LM head
- projection weight beats, source starvation, consumer blocking, selector
  fullness and high-water mark, Q8 request/response waits, drain, and bank waits
- weight and history AXI reads, weight-port gaps and skew, and KV AXI writes

The exact ID-to-name mapping is defined in `shared/engine/metrics.zig` and is
part of metrics schema 1.

## Recorder contract

The production recorder exposes schema `0x00010000`, capability word
`0x00280B1D`, 11 stages, and 40 stable metrics. Each execution freezes one
snapshot tagged with its command. The driver validates schema, capabilities,
tag, status, and outcome before acknowledging it. With instrumentation enabled,
it also reads every indexed metric and checks overflow state. Total cycles is
u64, stage-call metrics 13 through 23 are saturating u8, selector high-water
metric 28 is u3, and the remaining metrics are saturating u32.

The daemon advertises metrics schema 1 and summary support in `inspect device`.
`full` is reserved until a versioned diagnostic-bank payload and selector are
defined; current software does not advertise it, even if an experimental image
sets the reserved hardware capability bit. Any schema, count, tag, outcome, or
capability mismatch fails the request instead of returning partial measurements.

Host wall time includes software, TCP, and output work. Device time converts
engine cycles using the clock reported by the daemon. Keep both: their
difference is the fastest way to separate datapath limits from host or transport
overhead. Qualification should also reject counter overflow.
