# fpga/ooc — out-of-context synthesis probe (the fast timing/area loop)

**What it is:** `synth_design -mode out_of_context` on **one module** on the Vivado VM
(~1–2 min), giving **real DSP / LUT / CARRY8 / FF counts and a synth-level Fmax** — the
middle rung between Verilator cosim (correctness, seconds, no Vivado) and the full ~30-min
bitstream build (routed timing + congestion). Use it to answer "did this map onto DSP/CARRY8?
how fast is this path in isolation?" without paying for place-and-route.

> ### ⚠️ The one caveat (this is the lesson we keep relearning)
> **OOC ≠ the combined build.** OOC times one module's internal reg→reg paths with the I/O
> false-pathed and *no congestion*. The real build adds interconnect/DMA paths and routing
> congestion that OOC cannot see. We got burned twice: the 104-bit matmul OOC closed f300 in
> isolation but the **combined f250 build failed at −0.370ns**. So: OOC is a *resource + isolated-Fmax*
> probe. For a go/no-go on a clock, you still need the routed combined build and must **read its
> worst path** (`…_timing_summary_routed.rpt`, grep `Slack`).

---

## Run an OOC (the general workflow — copy-paste)

```bash
cd fpga/ooc
source ../bitstreams/combined-v1/config.env        # sets $VM (the Windows Vivado VM)
VM_OOC=penzai-ooc                                   # the OOC working dir on the VM

# 1. sync the RTL the probe needs (the harness + every module it instantiates)
scp <files...> "$VM:$VM_OOC/"

# 2. synth + report. period_ns: 3.333 = f300, 4.0 = f250. pfx names the report files.
ssh "$VM" "cd $VM_OOC && ooc.bat <top> <period_ns> <pfx> <files...>"
#   → prints one line:
#   RESULT top=<top> dsp=N lut=N carry8=N ff=N wns_ns=<slack> fmax_mhz=<f> (period=<p>ns)
```

**Reading the RESULT line:** `wns_ns > 0` ⇒ the module closes that period in isolation;
`fmax_mhz = 1000/(period − wns)` is the achieved synth-level frequency; `dsp/lut/carry8/ff` are
the real mapped resource counts. Full breakdown lands on the VM in `$VM_OOC/<pfx>_util.rpt`
(resources) and `<pfx>_timing.rpt` (top-5 worst paths) — fetch with
`scp "$VM:$VM_OOC/<pfx>_timing.rpt" .` if you need the path detail.

Current `$VM` = `10.211.55.3`; the VM already has the gemm/flash RTL synced from prior runs, so
re-sync only the files you changed.

## The gemm probes (so you don't re-derive the file lists)

```bash
cd fpga/ooc
source ../bitstreams/combined-v1/config.env
scp ../rtl/gemm.v ../rtl/numeric/fma.v ../rtl/gemm_kernel.v gemm_rb_ooc.v gemm_emit_ooc.v "$VM:penzai-ooc/"

# the throughput accumulate path (the f250/f300 matmul limiter to watch)
ssh "$VM" "cd penzai-ooc && ooc.bat gemm_rb_ooc     3.333 gemmrb   gemm.v fma.v gemm_rb_ooc.v"
# the per-output fixed→fp32 emit (LZD + barrel shift)
ssh "$VM" "cd penzai-ooc && ooc.bat gemm_emit_ooc   3.333 gemmemit gemm.v gemm_emit_ooc.v"
# the full kernel (FSM + banked rowblock + pipelined emit + result buffer)
ssh "$VM" "cd penzai-ooc && ooc.bat gemm_kernel_ooc 3.333 gemmk    gemm.v fma.v gemm_kernel.v gemm_kernel_ooc.v"
```

Reference numbers (xck26 @ 3.333ns, current carry-save matmul): `gemm_rb_ooc` 398.9 / `gemm_emit_ooc`
386.8 / `gemm_kernel_ooc` 375.8 MHz — all clear f300 in isolation (but see the caveat).

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

Then add it to the sync + `ooc.bat` invocation above with the files it instantiates.

## The scripts (what's what)

- `ooc_synth.tcl` — `<top> <period_ns> <out_prefix> <rtl...>`. `read_verilog` each file,
  `synth_design -mode out_of_context -part xck26-sfvc784-2LV-c -top <top>`, create the clock,
  **false-path all I/O**, write `<pfx>_{util,timing}.rpt`, and print the machine-readable `RESULT`
  line (counts via `get_cells` filters; WNS via `get_timing_paths`).
- `ooc.bat` — VM-side shim: sources Vivado settings, runs `vivado -mode batch -source ooc_synth.tcl
  -tclargs %*`.
- `run.sh` — a *specific* two-top A/B (the plan-7 phase-2 derisk below), not the general path. For an
  arbitrary module use the `scp` + `ssh ooc.bat` recipe above; `run.sh` is just a saved comparison.

---

## Appendix — why this exists (the plan-7 phase-2 derisk)

The probe was built to derisk the fixed-point-accumulate bet before the gemm rebuild. `mac_array`
(candidate: cheap multiply → barrel shift → single-cycle 72-bit CARRY8 accumulate) vs `rowblock_ooc`
(the old fp32 `matmul_rowblock`), same part/clock, @ 3.333 ns:

| metric        | baseline fp32 | candidate fixed-pt | delta |
|---------------|--------------:|-------------------:|-------|
| CLB LUTs      | 54,148        | 4,294              | −92%  |
| CLB Registers | 39,654        | 2,663              | −93%  |
| CARRY8        | 1,536         | 144 (= 16×⌈72/8⌉)  | −91%  |
| DSP48E2       | 64            | 32 (= 16×2)        | −50%  |
| WNS @3.333ns  | +0.442 ns     | +0.893 ns          | both close |
| Fmax (OOC)    | 346 MHz       | 410 MHz            | +64   |

It retired all three mapping risks (multiply→DSP and halved, wide accumulate→CARRY8 with no LUT
blow-up, single-cycle accumulate closes f300) — and the win that actually unblocked the design was
the freed ~50k LUT, **not** the 410 MHz (which the congested build never sees). `fp_fixed_mac.v` /
`mac_array.v` / `rowblock_ooc.v` are throwaway probes, not synthesized into any bitstream.
