# fpga/ooc — out-of-context synthesis probe

A "Level 2" feedback loop between Verilator cosim (correctness, seconds, no Vivado)
and the full ~30-min bitstream build (timing closure + routed CLB). This runs
`synth_design -mode out_of_context` on a single module on the Vivado VM (~1-2 min),
giving **real DSP / LUT / CARRY8 / FF counts and a synth-level Fmax** — enough to
derisk a *resource-mapping* bet (did fp32 re-map onto DSPs/CARRY8?) before paying for
place-and-route. It does **not** give routed CLB% or congestion — that's still the
real build (plan-7 phase 4).

## Run

    ./run.sh [period_ns]        # default 3.333 (the f300 goal); reuses combined-v1/config.env VM

Synthesizes two tops at the same part/clock and prints a one-line RESULT for each;
full reports land in `out/{cand,base}_{util,timing}.rpt`.

## Plan-7 phase-2 derisk (the reason this exists)

`mac_array` (candidate fixed-point datapath: cheap multiply → barrel shift → single-cycle
72-bit CARRY8 accumulate) vs `rowblock_ooc` (the current fp32 `matmul_rowblock`,
ROWS=16/COLS_MAX=8/ACCUM_DEPTH=5). Result @ 3.333 ns, part xck26-sfvc784-2LV-c:

| metric        | baseline fp32 | candidate fixed-pt | delta |
|---------------|--------------:|-------------------:|-------|
| CLB LUTs      | 54,148        | 4,294              | −92%  |
| CLB Registers | 39,654        | 2,663              | −93%  |
| CARRY8        | 1,536         | 144 (= 16×⌈72/8⌉)  | −91%  |
| DSP48E2       | 64            | 32 (= 16×2)        | −50%  |
| WNS @3.333ns  | +0.442 ns     | +0.893 ns          | both close |
| Fmax (OOC)    | 346 MHz       | 410 MHz            | +64   |

All three phase-2 risks retired: multiply → DSP (and halved), wide accumulate →
CARRY8 (no LUT blow-up), single-cycle 72-bit accumulate closes f300 with margin.

**Caveats:** the candidate models the *decode arithmetic core* (16 single accumulators),
not a complete rowblock — a finished `gemm` adds emit-normalize, the prefill COLS_MAX
column accumulators, and a control FSM, so expect a complete rowblock ~3–4× smaller
than baseline (still past plan-7's 50% target), not 12.6×. And OOC timing ≠ the
congested build: the win that unblocks f300 is the freed ~50k LUT, not the 410 MHz.

## Files

- `fp_fixed_mac.v` — candidate per-row lane (plan-7 Part 1 recipe). Throwaway probe, not synthesized into any bitstream.
- `mac_array.v` — ROWS=16 lanes of the above.
- `rowblock_ooc.v` — OOC wrapper pinning the current `matmul_rowblock` to the shipping decode config.
- `ooc_synth.tcl` — parameterized: `<top> <period_ns> <out_prefix> <rtl...>`; false-paths I/O so only internal reg→reg paths are timed.
- `ooc.bat` — VM-side Vivado-settings shim (mirrors `bitstreams/combined-v1/build.bat`).
- `run.sh` — syncs RTL to the VM, runs both tops, fetches reports.
