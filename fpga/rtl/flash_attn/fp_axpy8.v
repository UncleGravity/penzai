// fp_axpy8 - 8-lane fused accumulator update: out[i] = acc[i]*s1 + p*V[i].
//
//   axpy step:  s1 = corr, p = weight, v = a V row  -> acc[i]*corr + p*V[i]
//   scale-only: p = 0                               -> acc[i]*s1 (e.g. emit ·1/l)
//
// Feed-forward (no accumulation), 8 lanes in parallel, composed from numeric/ leaves.
// Latency FP32_MUL_LATENCY + FP32_ADD_LATENCY = 3 + 4 = 7 (fmul is 3 cycles at any
// MANT_W; the cvt seams are combinational). The kernel streams one acc beat (8
// elements) per cycle through this and writes the result back. Self-timed: the adder
// follows the mul's valid.
//
// Mixed precision (plan-attention-migration §2, industry-standard recipe):
//   u_m1  acc·s1  (accumulator rescale / emit scale)  fp32  — keeps the accumulator exact
//   u_m2  p·V     (the new convex-combination term)   bf16  — operands->bf16, fmul#(7),
//                                                            product widened back to f32
//   u_a   t1 + t2 (the accumulate)                    fp32
// P·V is a convex combination (softmax weights sum to 1), so bf16 is benign; the QK dot,
// m/l and softmax all stay fp32. The bf16 path is gated bit-faithfully vs
// flash_ref.bf16MulPV (numeric/fmul #(7) modeled exactly).

`default_nettype none

module fp_axpy8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] acc,   // 8 × f32
    input  wire [127:0] v,     // 8 × f16
    input  wire [31:0]  s1,    // corr (axpy) or 1/l (emit)
    input  wire [31:0]  p,     // weight (axpy); 0 for scale-only
    output wire         valid_out,
    output wire [255:0] out    // 8 × f32
);
    `include "fmt.vh"
    wire [7:0] add_v;

    // p -> bf16 once, broadcast to all lanes (the convex-combination operand).
    wire [15:0] p_b;
    cvt_f32_bf16 u_pb (.in(p), .out(p_b));

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane
            wire [31:0] vf, t1, t2;
            wire [15:0] v_b, t2_b;
            wire        m1v, m2v;
            cvt_f16_f32 u_w (.in(v[i*16 +: 16]), .out(vf));
            // acc·s1 (accumulator rescale / emit scale) — fp32.
            fmul #(.MANT_W(FMT_FP32_MANT)) u_m1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(acc[i*32 +: 32]), .b(s1), .valid_out(m1v), .out(t1));
            // p·V — bf16 operands, fmul#(7), widen the product back to f32 for the fp32 add.
            cvt_f32_bf16 u_vb (.in(vf), .out(v_b));
            fmul #(.MANT_W(FMT_BF16_MANT)) u_m2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(p_b), .b(v_b), .valid_out(m2v), .out(t2_b)); // m2v unused (synced to m1v)
            cvt_bf16_f32 u_t2w (.in(t2_b), .out(t2));
            fadd #(.MANT_W(FMT_FP32_MANT)) u_a (.clk(clk), .rst_n(rst_n), .valid_in(m1v),
                .a(t1), .b(t2), .valid_out(add_v[i]), .out(out[i*32 +: 32]));
        end
    endgenerate

    assign valid_out = add_v[0]; // lanes synchronized
    wire _unused = &{1'b0, add_v[7:1]};
endmodule
