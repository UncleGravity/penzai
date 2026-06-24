// numeric_diff_top - cosim-only harness: each numeric/ leaf instantiated next to the
// proven rtl/fp leaf it replaces, sharing one stimulus, exposing both outputs so the tb
// can assert NEW === OLD bit-for-bit. This is the gate that makes "retire the old"
// safe (plan-numeric-leaves.md). Extend with fmul/cvt/reduce pairs as they land.

`default_nettype none

module numeric_diff_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [15:0] c16,         // cvt_f16_f32 input
    input  wire [13:0] cint,        // cvt_i2f input (signed int14)
    input  wire [511:0] rin,        // reduce input (16 × f32)
    input  wire [31:0] ex_x,        // exp input
    input  wire [31:0] rc_l,        // recip input

    // fadd#(FP32) vs fp32_add_pipe
    output wire        fadd_new_valid,
    output wire [31:0] fadd_new_out,
    output wire        fadd_old_valid,
    output wire [31:0] fadd_old_out,

    // fmul#(FP32) vs fp32_mul_pipe
    output wire        fmul_new_valid,
    output wire [31:0] fmul_new_out,
    output wire        fmul_old_valid,
    output wire [31:0] fmul_old_out,

    // cvt_f16_f32 vs fp16_to_fp32 ; cvt_i2f vs int_to_fp32 (combinational)
    output wire [31:0] cvt_f16_new_out,
    output wire [31:0] cvt_f16_old_out,
    output wire [31:0] cvt_i2f_new_out,
    output wire [31:0] cvt_i2f_old_out,

    // reduce#(FP32,16) vs fp_addtree
    output wire        reduce_new_valid,
    output wire [31:0] reduce_new_out,
    output wire        reduce_old_valid,
    output wire [31:0] reduce_old_out,

    // exp vs fp_exp ; recip vs fp_recip (each exercises interp vs fp_interp internally)
    output wire        exp_new_valid,
    output wire [31:0] exp_new_out,
    output wire        exp_old_valid,
    output wire [31:0] exp_old_out,
    output wire        recip_new_valid,
    output wire [31:0] recip_new_out,
    output wire        recip_old_valid,
    output wire [31:0] recip_old_out,

    // cvt bf16 seams (new — no rtl/fp predecessor; the tb checks vs the exact bit-slice def)
    output wire [15:0] cvt_bf16_narrow_out,  // cvt_f32_bf16(a)
    output wire [31:0] cvt_bf16_widen_out    // cvt_bf16_f32(c16)
);
    `include "fmt.vh"

    fadd #(.MANT_W(FMT_FP32_MANT)) u_fadd_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(a), .b(b), .valid_out(fadd_new_valid), .out(fadd_new_out)
    );
    fp32_add_pipe u_fadd_old (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(a), .b(b), .valid_out(fadd_old_valid), .out(fadd_old_out)
    );

    fmul #(.MANT_W(FMT_FP32_MANT)) u_fmul_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(a), .b(b), .valid_out(fmul_new_valid), .out(fmul_new_out)
    );
    fp32_mul_pipe u_fmul_old (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(a), .b(b), .valid_out(fmul_old_valid), .out(fmul_old_out)
    );

    // cvt (combinational): f16->f32 widen, signed int14->f32
    cvt_f16_f32       u_cvtf16_new (.in(c16), .out(cvt_f16_new_out));
    fp16_to_fp32      u_cvtf16_old (.in(c16), .out(cvt_f16_old_out));
    cvt_i2f #(.WIDTH(14)) u_cvti2f_new (.in(cint), .out(cvt_i2f_new_out));
    int_to_fp32 #(.WIDTH(14)) u_cvti2f_old (.in(cint), .out(cvt_i2f_old_out));

    // reduce#(FP32,16) vs fp_addtree (both compose their respective fp add)
    reduce #(.MANT_W(FMT_FP32_MANT), .N(16)) u_reduce_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in(rin), .valid_out(reduce_new_valid), .sum(reduce_new_out)
    );
    fp_addtree u_reduce_old (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .in(rin), .valid_out(reduce_old_valid), .sum(reduce_old_out)
    );

    // exp / recip (each composes interp internally → gates interp vs fp_interp too)
    exp u_exp_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x(ex_x), .valid_out(exp_new_valid), .y(exp_new_out)
    );
    fp_exp u_exp_old (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x(ex_x), .valid_out(exp_old_valid), .y(exp_old_out)
    );
    recip u_recip_new (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .l(rc_l), .valid_out(recip_new_valid), .y(recip_new_out)
    );
    fp_recip u_recip_old (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .l(rc_l), .valid_out(recip_old_valid), .y(recip_old_out)
    );

    // bf16 narrow/widen — reuse the a (f32) and c16 (16-bit) stimulus.
    cvt_f32_bf16 u_bf16_narrow (.in(a),   .out(cvt_bf16_narrow_out));
    cvt_bf16_f32 u_bf16_widen  (.in(c16), .out(cvt_bf16_widen_out));
endmodule
