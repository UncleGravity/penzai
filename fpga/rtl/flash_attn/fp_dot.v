// fp_dot - one beat of a Q·K dot product: sum over LANES of f32 Q × widened-f16 K.
//
//   sum = Σ_{i<LANES} q[i] · fp32(k[i])
//
// Pure feed-forward: widen → multiply → pipelined adder tree, composed from numeric/
// leaves (cvt_f16_f32 + fmul + reduce) at fp32. NO accumulation across beats — the
// kernel sums the per-beat results into its per-head dot accumulator, where the
// n_heads-deep interleave hides the fp-add recurrence (the same pool that hides
// m/l/acc). That keeps recurrence-hiding in one place instead of duplicating
// matmul_rowblock's ACCUM_DEPTH machinery inside the leaf.
//
// LANES = 8 (a 128-bit K beat / 256-bit Q beat). Latency valid_in→valid_out:
// (FP32_MUL_LATENCY+1) + log2(LANES)·FP32_ADD_LATENCY = 4 + 12 = 16 — the dot's fmul runs
// MUL_PIPE=1 (the fp32 2-DSP multiply split for f300). VALUE-identical to the rtl/fp
// version (the extra stage is pure pipelining): the numeric leaves ≡ the fp32_*_pipe leaves
// and reduce(N=8) ≡ the hand-rolled add tree (same (0,1)(2,3)… order — the cosim is the gate).

`default_nettype none

module fp_dot (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] q,   // 8 × f32
    input  wire [127:0] k,   // 8 × f16
    output wire         valid_out,
    output wire [31:0]  sum
);
    `include "fmt.vh"
    localparam integer LANES = 8;

    // Widen each f16 K lane to f32 (combinational).
    wire [31:0] kf [0:LANES-1];
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_widen
            cvt_f16_f32 u_w (.in(k[i*16 +: 16]), .out(kf[i]));
        end
    endgenerate

    // Multiply lanes (all share valid_in, so they stay synchronized) and pack the
    // products straight into the reduce input bus.
    wire [LANES-1:0]    prod_v;
    wire [LANES*32-1:0] prod_bus;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_mul
            // MUL_PIPE=1: the fp32 24×24 mantissa multiply is a 2-DSP cascade; the extra
            // stage splits it (one DSP/cycle) — the f300 flash-dot fix. The kernel is
            // self-timed off the dot valid, so the +1 latency needs no caller change.
            fmul #(.MANT_W(FMT_FP32_MANT), .MUL_PIPE(1)) u_m (
                .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(q[i*32 +: 32]), .b(kf[i]),
                .valid_out(prod_v[i]), .out(prod_bus[i*32 +: 32])
            );
        end
    endgenerate

    // Pipelined fp32 adder tree 8 → 1 (numeric/reduce, same (0,1)(2,3)… pairing order).
    reduce #(.MANT_W(FMT_FP32_MANT), .N(LANES)) u_tree (
        .clk(clk), .rst_n(rst_n), .valid_in(prod_v[0]),
        .in(prod_bus), .valid_out(valid_out), .sum(sum)
    );

    // Lanes are synchronized; prod_v[0] drives the tree, the rest are redundant.
    wire _unused = &{1'b0, prod_v[LANES-1:1]};
endmodule
