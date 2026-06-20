// flash_fp_top - cosim harness wrapping the two flash fp leaves so one Verilator
// model exercises both. Same valid_in launches an exp(x) and a 1/l in lockstep;
// each leaf raises its own valid_out (different latencies) when its result lands.
// Not a deployed module — purely the micro-cosim DUT.

`default_nettype none

module flash_fp_top (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [31:0]  x,
    input  wire [31:0]  l,
    input  wire [255:0] dq,    // 8 × f32 for fp_dot
    input  wire [127:0] dk,    // 8 × f16 for fp_dot
    output wire         exp_valid,
    output wire [31:0]  exp_y,
    output wire         recip_valid,
    output wire [31:0]  recip_y,
    output wire         dot_valid,
    output wire [31:0]  dot_sum
);
    fp_exp u_exp (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .x(x), .valid_out(exp_valid), .y(exp_y)
    );
    fp_recip u_recip (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .l(l), .valid_out(recip_valid), .y(recip_y)
    );
    fp_dot u_dot (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .q(dq), .k(dk), .valid_out(dot_valid), .sum(dot_sum)
    );
endmodule
