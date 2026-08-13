# FPGA routed analysis tools

These tools inspect the timing-clean checkpoint retained by the production bitstream flow.
They do not replace the routed gates that run during every build.

```sh
cd fpga/tools
./analyze.sh summary                 # production metrics and gates
./analyze.sh deep                    # broad timing/congestion/QoR diagnostic bundle
./analyze.sh check gemm-acc          # targeted accumulator CE locality check
./analyze.sh summary w512-p4-f250    # choose another retained variant
```

Results are fetched into `fpga/bitstream/out/analysis/<run-id>/`; `latest` points to the
most recent local bundle. `summary` uses `metrics.tcl`, the same implementation used by
the production build, and fails if any routed timing or structural gate fails. `deep` runs `report.tcl`
and captures broad diagnostics without imposing new gates. Design-specific structural
checks remain separate and are invoked only when their invariant is relevant.

The normal escalation order is:

1. Read the build run's `summary.txt`, `metrics.tsv`, and `manifest.tsv`.
2. Run `summary` when checking an older retained checkpoint or validating the gate code.
3. Run `deep` only when timing, congestion, utilization, clocks, or QoR need explanation.
4. Run `check gemm-acc` when accumulator clock-enable locality is the suspected invariant.

`metrics.tcl` records setup and hold slack, near-critical path counts, route status,
constraint coverage, methodology severities, flat/hierarchical utilization, and build
phase durations. The `.rpt` files are supporting evidence; TSV files are the stable
comparison surface.
