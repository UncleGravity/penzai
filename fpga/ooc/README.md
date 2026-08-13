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
and it was promoted, deployed, and board-qualified. It passes timing with five
setup paths below 50 ps. This is why combined routing,
not OOC Fmax, remains authoritative; the OOC result did not establish f300
deployability. Board qualification now extends through Q1/Q2 context 2048; the Q1
attention schedule remains essentially cycle-neutral relative to P2b despite the
lower release clock.

P2d adds exact raw-F32-to-Q8 ingress and resident Q8 activation reuse to GEMM
version 13. Its initial isolated probes closed 3.333 ns: the quantizer leaf reached
`+0.192 ns` (318.4 MHz estimated Fmax) with 792 LUTs, 908 FFs, 22 CARRY8s, and two
DSPs in `out/runs/20260812T210745Z-0c77a2d69ab0-q8subnormal/q8-quantizer`; the
complete GEMM reached `+0.619 ns` (368.5 MHz) with 38,544 LUTs, 42,935 FFs,
719 CARRY8s, and 32 DSPs in `out/runs/20260812T211434Z-0c77a2d69ab0`.

The first clean combined f285 route exposed logic those probes did not compose.
Run `20260812T212725Z-f9e1ca83f8ae-w512-p4-f285-clean` routed fully but failed at
`-0.215/+0.010 ns` setup/hold, with 142 failing setup paths and 580 below 50 ps.
It used 81,284 LUTs, 96,220 FFs, 55.5 BRAM tiles, four URAMs, and 95 DSPs at
99.32% CLB occupancy. Its ingress bookkeeping/multiply and GEMM exponent-add
families became the bounded repair targets; no image from this run was promoted.

Commit `547d87b` replaces the flat total-block multiply and wide ingress counters
with nested column/Q1/sub-block state, registers the scalar boundary, emits through
one 64-bit register, and retimes the `FE_LAT` exponent addition. It removes 228
state bits and one DSP without changing the established binary or ternary hashes
or cycles, and expands the nested-boundary and missing-`TLAST` cosim coverage.
Current-source full-decode integration OOC run
`out/runs/20260812T223110Z-547d87b12094-full-decode` passes at 3.333 ns with
`+0.245/+0.039 ns` setup/hold, 40,141 LUTs, 44,912 FFs, 775 CARRY8s, 34 DSPs,
two BRAM tiles, and four URAMs. The failed route's targeted path families are
absent from this probe's reported critical paths. The 3.333 ns production
artifacts pass; the probe driver exited afterward because its read-only,
non-production second-period report script attempted to mutate the open design.
That later report error does not affect the production source or result.

Binary and ternary RTL integration tests still raw-load and quantize the first
grouped projection, reuse the resident activation for the second with zero
activation beats, and reject stale epochs or incompatible shapes before the kernel
consumes weights or emits results while preserving the prior resident record. Raw
framing, non-finite, scale, and arithmetic aborts instead invalidate resident state
before kernel weight/result consumption. Movers may already be armed on either
path. The quantizer tests retain exact finite-domain agreement across 1,032 Q8
blocks. Even the full-decode probe remains resource and isolated-timing evidence;
the combined route is authoritative.

That authority is supplied by replacement clean f285 run
`20260812T224303Z-547d87b12094-w512-p4-f285-clean`, which uses source bundle
`1cfc1e173ba0ae1d06d1fceb1d3fa83ec29535f760a7a7f91a6b5e0458078249`.
It is fully routed and passes at +0.043/+0.010 ns setup/hold with no negative paths
and 4/55/622 paths below 50/100/200 ps. Routed use is 81,887 LUTs, 95,699 FFs,
1,408 CARRY8s, 94 DSPs, 55.5 BRAM tiles, and four URAMs at 99.17% CLB occupancy.
Structural timing counts are clean and the 75 ps guardband restoration is exact;
methodology reports critical TIMING-2/TIMING-4 plus five TIMING-28 and one ULMTCS-1
warning.

The promoted image was deployed with SHA-256
`9ab576cad24eb3c77d6b55200d5e9a08d92f0197625f76d479c13fc6fa82a70f`.
Live schema 2/wire 14/profile 6 capabilities report GEMM `0xB05A2000` v13 at
284,997,152 Hz and flash v1 with 64 query slots. Q1 `p32` logits pass at 0.098204
maximum absolute error with zero token mismatches. Q2 `p32` also passes with
0.089185/0.131115 step differences, 0.131115 maximum absolute error, exact argmax,
zero token mismatches, and `check=ok`.

Final artifact `20260812T235644Z-characterize-9bb6d3eb522f` is complete and
run-validated across all six samples, closes accounting, and has identical start/end
capabilities. Every grouped operation stays on PL (`pl/staged` for 16 columns and
`pl/direct` for one column), raw-loads A once, reuses with zero activation beats,
and has no fallback. The c512 decode `6144x1x2048` bucket closes at 1,792 calls,
3,584 runs, 1,835,008 A beats, 11,010,048 R beats, and
110,100,480/198,180,864 Q1/Q2 W beats with zero host quantize/pack time. P128's
staged main/tail split is 216 calls and 864 runs with 3,538,944 A beats, then one
direct call/two runs/1,024 beats; c512 uses 864 calls and 3,456 runs with
14,155,776 A beats plus the same tail.

Attention is unmodified, stays on its expected staged/direct PL paths with zero
K/V/O stalls, and changes matched cycles by at most 0.01%. Its expected approximate-
flash advisory covers 12 smoke calls and 40/24,576 values at maximum
absolute/normalized differences 0.5542/0.0467. Model logits pass, with no GEMM,
group, Q8, DMA, kernel, or request errors.

The single-repeat P2c-to-P2d decode check shows no device regression; all four
observations are lower by 0.35%/1.54% for p128 Q1/Q2 and 1.37%/1.10% for c512,
with a new c0 anchor of 65.716/88.615 ms/token. The grouped kernel still pays for
raw PL quantization: cycles rise 25.98%/21.87% for Q1/Q2 staged columns-16 and
18.28%/8.96% for c512 decode versus two primitive calls. Command removal and zero
grouped host quantize/pack time are consistent with offsetting that cost; this
one-repeat gate does not establish a repeatable speedup.

P128 and c512 prefill walls of 47.259/91.355 s and 224.114/411.642 s reflect a
download-rate drop from roughly 40.5-49.7 MiB/s to 8.0-9.0 MiB/s at unchanged
328.9 MiB-2.8 GiB volumes; that degraded transport is not attributed to P2d.
These board results close P2d at the bounded f285 qualification. They do not close
the larger P2 section and scratch contract.

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

Reference numbers (xck26 @ 3.333ns, current carry-save matmul): `gemm_rb_ooc`
398.9 MHz and `gemm_emit_ooc` 386.8 MHz. Before timing repair, P2d's issue-ordered
dual-format `gemm_kernel_ooc` reached 368.5 MHz (`+0.619 ns`, 38,544 LUTs,
42,935 FFs, 32 DSPs), while the separate exact-Q8 leaf reached 318.4 MHz
(`+0.192 ns`, 792 LUTs, 908 FFs, 22 CARRY8s, two DSPs). The repaired full-decode
integration probe closes at `+0.245/+0.039 ns` setup/hold with 40,141 LUTs,
44,912 FFs, 775 CARRY8s, 34 DSPs, two BRAM tiles, and four URAMs. The current
adaptive P2c `flash_kernel_ooc` is 320.7 MHz (`+0.215 ns`, 22,657 LUTs, 19,216
FFs, 19 BRAM tiles, and 60 DSPs). All clear f300 in isolation, but see the caveat.

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
