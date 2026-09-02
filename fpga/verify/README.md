# FPGA verification

`fpga/build/sources.f` is the authoritative production RTL
closure. Simulation, lint, and synthesis scripts must either consume that
manifest or be checked against it by `check_sources.sh`.

`suites.zig` is the central verification registry used by `build.zig`. It
separates fast default gates from expensive extended checks:

```sh
zig build lint-rtl
zig build test-rtl
zig build synth-rtl
zig build formal
zig build verify-rtl

zig build synth-rtl-all
zig build formal-all
```

Each suite also has a focused `zig build verify-<suite>` step. Verification
outputs belong under `.zig-cache/fpga-verify/`; they are not source artifacts.

`formal/` contains only harnesses and proofs for RTL in the production source
manifest. See `formal/README.md` for coverage and known gaps.
