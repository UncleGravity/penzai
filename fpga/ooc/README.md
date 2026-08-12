# FPGA out-of-context synthesis probes

**What it is:** `synth_design -mode out_of_context` on **one module** on the Vivado VM,
giving **real DSP / LUT / CARRY8 / FF counts and a synth-level Fmax** — the
middle rung between Verilator cosim (correctness, seconds, no Vivado) and the full ~30-min
bitstream build (routed timing + congestion). Use it to answer "did this map onto DSP/CARRY8?
how fast is this path in isolation?" without paying for place-and-route.

> ### The important limitation
> **OOC ≠ the combined build.** OOC times one module's internal reg→reg paths with the I/O
> false-pathed and *no congestion*. The real build adds interconnect/DMA paths and routing
> congestion that OOC cannot see. We got burned twice: the 104-bit matmul OOC closed f300 in
> isolation but the **combined f250 build failed at −0.370ns**. So: OOC is a *resource + isolated-Fmax*
> probe. For a go/no-go on a clock, you still need the routed combined build and must **read its
> worst path**.

Small leaves usually finish in 1–2 minutes. The full GEMM kernel takes a few
minutes because Vivado also optimizes the wide accumulator bank and complete
dual-format control path. `flash-kernel` similarly covers the complete attention
controller, BRAMs, and composed numeric pipelines; use it as the before/after P2
resource and isolated-Fmax probe. P2b restored its registry period from the
temporary P2a 3.600 ns baseline to the 3.333 ns production target.

P2c is a concrete example of the OOC limitation. The fixed 128-slot query-blocked
kernel passed OOC at `+0.215 ns` with 60 DSPs, 23,489 LUTs, 19,315 FFs, and 33 BRAM
tiles, but its clean combined f300 route failed at `-0.035 ns` with 124 failing
setup endpoints, 69.5 total BRAM tiles, and 98.91% CLB occupancy. The adaptive
64-slot implementation retains four-query tiles for up to 16 heads, uses two-query
tiles above 16 heads, and reduces flash OOC to 22,657 LUTs, 19,216 FFs, and 19
BRAM tiles. It retains 60 DSPs and the same `+0.215 ns` WNS (320.7 MHz estimated
Fmax). The adaptive result is retained in
`out/runs/20260812T133852Z-95b751ff9ac3`.

That smaller kernel still did not close in the combined design at f300. Adaptive
run `20260812T144901Z-bbeac0c04eff-w512-p4-f300-clean` failed at `-0.088 ns`
setup WNS with 231 failing endpoints and 560 paths below 50 ps. Clean f285 run
`20260812T155038Z-3ef082b0fe4a-w512-p4-f285-clean` then passed at
`+0.036/+0.010 ns` setup/hold with five paths below 50 ps. It uses 80,837 LUTs,
95,229 FFs, 55.5 BRAM tiles, four URAMs, and 92 DSPs at 98.61% CLB occupancy,
and it was promoted, deployed, and board-qualified. It clears the 25 ps release
floor but remains below the 50 ps headroom target. This is why combined routing,
not OOC Fmax, remains authoritative; the OOC result did not establish f300
deployability. Board qualification now extends through Q1/Q2 context 2048; the Q1
attention schedule remains essentially cycle-neutral relative to P2b despite the
lower release clock.

---

## Run probes

```bash
cd fpga/ooc
./run.sh list
./run.sh gemm-rb
./run.sh all
```

`run.sh` reads `probes.tsv`, validates and syncs the declared sources, runs Vivado,
and fetches complete results to `out/runs/<run-id>/`. The run-level `summary.tsv`
is the comparison table; each probe directory contains its detailed timing and
utilization reports, metrics, status, and driver log. `out/runs/latest` points at
the most recent bundle, including failed runs.

`wns_ns >= 0` means the module closes its declared period in isolation.
`fmax_mhz = 1000 / (period - WNS)` is the achieved synth-level frequency. A probe
fails when there is no setup path or WNS is negative.

Reference numbers (xck26 @ 3.333ns, current carry-save matmul): `gemm_rb_ooc` 398.9 MHz,
`gemm_emit_ooc` 386.8 MHz, and the issue-ordered dual-format `gemm_kernel_ooc` 368.5 MHz
(`+0.619 ns`, 38,439 LUTs, 42,796 FFs, 32 DSPs). The current adaptive P2c
`flash_kernel_ooc` is 320.7 MHz (`+0.215 ns`, 22,657 LUTs, 19,216 FFs, 19 BRAM
tiles, and 60 DSPs). All clear f300 in isolation, but see the caveat.

## Writing a new probe

The harness exists to make the *internal* logic the timed path. **Register every input and every
output** of the module under test, so the synthesizer sees real `reg → logic → reg` paths;
`ooc_synth.tcl` false-paths the top-level I/O so unconstrained ports don't pollute the report.
Minimal shape (mirror `gemm_rb_ooc.v` / `gemm_emit_ooc.v`):

```verilog
module my_ooc ( input wire clk, input wire rst_n, input wire [...] in, output reg [...] out_q );
    reg [...] in_q;
    always @(posedge clk) in_q <= in;          // register the inputs
    wire [...] y;
    dut_under_test u (.clk(clk), .rst_n(rst_n), .x(in_q), .y(y));
    always @(posedge clk) out_q <= y;          // register the outputs
endmodule
```

Add one tab-separated row to `probes.tsv`:

```text
name    top    period_ns    space-separated source files
```

Paths are relative to `fpga/ooc/`. Declare the harness and every RTL dependency;
headers may be listed and are copied without being compiled as standalone sources.
The runner rejects missing sources and duplicate flattened basenames.

## The scripts (what's what)

- `run.sh` — registry reader, VM sync/driver, result collector, and run summarizer.
- `probes.tsv` — the source-controlled probe name/top/period/source registry.
- `ooc_synth.tcl` — `<top> <period_ns> <out_prefix> <rtl...>`. `read_verilog` each file,
  `synth_design -mode out_of_context -part xck26-sfvc784-2LV-c -top <top>`, create the clock,
  **false-path all I/O**, write `<pfx>_{util,timing}.rpt`, and print the machine-readable `RESULT`
  line.
- `ooc.bat` — VM-side shim that sources Vivado settings and runs `ooc_synth.tcl`.
- `check_acc_fanout.tcl` — structural regression check for the lane-local GEMM accumulator
  clock-enable topology. Run from an initialized Vivado shell with
  `vivado -mode batch -source check_acc_fanout.tcl -tclargs <rtl-dir> <report-dir>`.
