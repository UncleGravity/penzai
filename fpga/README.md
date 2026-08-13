# FPGA development and analysis

The FPGA workflow is a ladder. Each rung answers a different question; no
single report directory is expected to do all of them.

| Rung | Command | Question answered |
|---|---|---|
| RTL lint | `zig build lint-rtl` | Does the deployable RTL elaborate cleanly? |
| Formal | `zig build formal` | Do the control and datapath invariants hold for all explored states? |
| Cosim | `zig build test-rtl-*` | Does RTL agree with the software oracle on representative data? |
| OOC synthesis | `fpga/ooc/run.sh <probe>` | How does one block map, and what is its isolated timing margin? |
| Routed build | `fpga/bitstream/build.sh` | Does the complete design fit, route, and pass timing? |
| Checkpoint analysis | `fpga/tools/analyze.sh <mode>` | Why did a routed design get its timing and utilization result? |

Formal harnesses and properties live in `fpga/formal/`. Verilator harnesses
live in `fpga/sim/`; synthesis harnesses and their declarative registry live in
`fpga/ooc/`.

## Bitstream gates

The production bitstream build refuses to generate a deployable artifact unless
all of these hold after routing:

- setup WNS is non-negative;
- hold WHS is non-negative;
- `report_route_status` says the design is fully routed;
- `check_timing` reports no clockless or unconstrained internal endpoints.

Methodology violations and near-critical path counts are recorded in every run.
They remain diagnostic and do not replace Vivado's setup and hold result. A
future timing-optimization pass may use those reports to earn additional slack,
but normal bitstream generation does not impose an extra margin threshold.

## Artifact ownership

Production evidence is immutable and ignored by Git:

```text
fpga/bitstream/out/
  penzai-combined-v1-<variant>.bit       # canonical deploy input
  penzai-combined-v1-<variant>.bit.bin   # canonical deploy input
  latest -> runs/<run-id>
  runs/<run-id>/                         # manifest, metrics, reports, bitstream
  analysis/<analysis-id>/                # checkpoint diagnostic bundles

fpga/ooc/out/
  runs/<run-id>/                         # probe manifests, metrics, reports, logs
```

The canonical `.bit` files are promoted only after every routed gate and
`bootgen` pass. VM-side `cache/` contains accelerators such as generated IP and
routed reference checkpoints; it is not evidence and may be deleted without
losing a recorded result.

Start with `summary.txt`, then `metrics.tsv` and `manifest.tsv`. Open the full
reports only when a metric needs explanation.
