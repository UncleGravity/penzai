// Shared numeric formats and pipeline latencies.
//
// Every leaf's mantissa width and pipeline latency live here as localparams; consumers
// DERIVE their delay-pipe depths and metadata-pipe lengths from these instead of
// hand-counting. The `MUL_LAT 2->3` change that rippled through six files
// (D1/D2/LAT/sub_sum_s6/guard_m2) is the fragility this kills: edit a latency here and
// every dependent depth recomputes — and the cosim proves the result unchanged.
//
// NO include guard, on purpose: Verilog `define`s are global across a compilation, so a
// guard would let only the FIRST module that `include`s this see the localparams. This
// is a per-module constants header — each `include re-emits them in that module's scope.

// ---- formats: stored mantissa bits (excl. the hidden 1); exponent stays fp32-wide (8b) ----
localparam integer FMT_FP32_MANT = 23;
localparam integer FMT_BF16_MANT = 7; // element-wise path; here so the menu is single-sourced

// ---- leaf pipeline latency (valid_in -> valid_out), the contract on each fp seam.
//      These mirror the leaf register structure; the pipeline cosims guard that they
//      stay in sync (a wrong value mis-times the derived pipes below and fails results). ----
localparam integer FP32_MUL_LATENCY = 3; // fmul: in-reg + product-reg + out-reg
localparam integer FP32_ADD_LATENCY = 4; // fadd: 4 registered stages
