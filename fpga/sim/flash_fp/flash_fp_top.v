// Cosim-only wrapper for numeric interp/exp/recip and the flash dot composition.

`default_nettype none

module flash_fp_top (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [31:0]  x,
    input  wire [31:0]  l,
    input  wire [31:0]  interp_lo,
    input  wire [31:0]  interp_hi,
    input  wire [7:0]   interp_t,
    input  wire [31:0]  interp_meta,
    input  wire [255:0] dq,
    input  wire [127:0] dk,
    output wire         interp_valid,
    output wire [31:0]  interp_frac,
    output wire [31:0]  interp_meta_out,
    output wire         exp_valid,
    output wire [31:0]  exp_y,
    output wire         recip_valid,
    output wire [31:0]  recip_y,
    output wire         dot_valid,
    output wire [31:0]  dot_sum
);
    interp #(.META_W(32)) u_interp (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .lo(interp_lo), .hi(interp_hi), .t(interp_t), .meta(interp_meta),
        .valid_out(interp_valid), .frac(interp_frac), .meta_out(interp_meta_out)
    );
    exp u_exp (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x(x), .valid_out(exp_valid), .y(exp_y)
    );
    recip u_recip (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .l(l), .valid_out(recip_valid), .y(recip_y)
    );
    fp_dot u_dot (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .q(dq), .k(dk), .valid_out(dot_valid), .sum(dot_sum)
    );
endmodule
