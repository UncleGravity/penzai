// fp_axpy8_fp32 - THROWAWAY OOC baseline: the all-fp32 axpy (pre-bf16, the Phase-1c
// version), for the area A/B against the shipping bf16 fp_axpy8. p·V here is fmul#(23)
// (fp32) instead of the bf16 fmul#(7)+cvt seam. Not synthesized into any bitstream.

`default_nettype none

module fp_axpy8_fp32 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] acc,
    input  wire [127:0] v,
    input  wire [31:0]  s1,
    input  wire [31:0]  p,
    output wire         valid_out,
    output wire [255:0] out
);
    `include "fmt.vh"
    wire [7:0] add_v;
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane
            wire [31:0] vf, t1, t2;
            wire        m1v, m2v;
            cvt_f16_f32 u_w (.in(v[i*16 +: 16]), .out(vf));
            fmul #(.MANT_W(FMT_FP32_MANT)) u_m1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(acc[i*32 +: 32]), .b(s1), .valid_out(m1v), .out(t1));
            fmul #(.MANT_W(FMT_FP32_MANT)) u_m2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(p), .b(vf), .valid_out(m2v), .out(t2));
            fadd #(.MANT_W(FMT_FP32_MANT)) u_a (.clk(clk), .rst_n(rst_n), .valid_in(m1v),
                .a(t1), .b(t2), .valid_out(add_v[i]), .out(out[i*32 +: 32]));
        end
    endgenerate
    assign valid_out = add_v[0];
    wire _unused = &{1'b0, add_v[7:1]};
endmodule
