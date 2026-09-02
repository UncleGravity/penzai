# FPGA quality-of-results checks

`yosys/` contains registered structural gates. The production map must retain
exactly 695 DSP48E2 cells and eight URAM288 cells; focused projection,
attention, and vector maps guard their local structure.

`vivado_ooc/` contains optional routed checks for the engine and projection
boundaries. Their outputs belong under `.zig-cache/fpga-verify/vivado-ooc/`.

`vivado/` analyzes a timing-clean checkpoint retained by the production build.
Use `./analyze.sh summary [f<MHz>]` for the standard timing, utilization, route,
and methodology bundle, or `./analyze.sh deep [f<MHz>]` for diagnostics. Results
are stored under `.zig-cache/fpga-verify/analysis/`.
