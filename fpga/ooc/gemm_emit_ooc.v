// gemm_emit_ooc - OOC timing probe for gemm_emit (the per-output fixed→fp32 normalize).
//
// gemm_emit (104-bit leading-zero-detect + 104-bit barrel shift + negate) sits on the kernel's
// readout path. The combinational form OOC'd at 198 MHz (WNS −1.7ns, fails f300) — so it is
// now PIPELINED over 3 stages (negate | LZD | shift+compose). Register acc+emin in and f32
// out so each internal stage is a real reg→reg path the OOC times (ooc_synth false-paths top
// I/O). The wider accumulator (full-f16-range fixed window) lengthens the LZD/shift — this
// is the probe that says whether 104b still closes f300.

`default_nettype none

module gemm_emit_ooc (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [103:0] acc,
    input  wire signed [7:0] emin,
    output reg  [31:0] f32_q
);
    reg [103:0]      acc_q;
    reg signed [7:0] emin_q;
    always @(posedge clk) begin
        acc_q  <= acc;
        emin_q <= emin;
    end

    wire [31:0] f32;
    wire        vo;
    gemm_emit #(.ACC_W(104), .EXP_W(8)) u (
        .clk(clk), .rst_n(rst_n), .valid_in(1'b1),
        .acc($signed(acc_q)), .emin(emin_q), .valid_out(vo), .f32(f32)
    );

    always @(posedge clk) f32_q <= f32;
endmodule
