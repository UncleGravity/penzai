// mac_array - ROWS lanes of fp_fixed_mac, the candidate phase-2 rowblock datapath.
//
// One fp_fixed_mac per output row (decode/single_col: one accumulator per row).
// Each lane reads a distinct slice of the flat input buses so the synth tool can't
// fold the 16 lanes into one. OOC synth of this gives the candidate's DSP / LUT /
// CARRY8 / FF / Fmax, to compare against rowblock_ooc (the thing it replaces).

`default_nettype none

module mac_array #(
    parameter integer ROWS     = 16,
    parameter integer WS_SIG_W = 11,
    parameter integer AS_SIG_W = 11,
    parameter integer S_W      = 14,
    parameter integer EXP_W    = 5,
    parameter integer ACC_W    = 72
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       clear,
    input  wire                       valid_in,
    input  wire [ROWS*WS_SIG_W-1:0]   ws_sig_flat,
    input  wire [ROWS*AS_SIG_W-1:0]   as_sig_flat,
    input  wire [ROWS*EXP_W-1:0]      ws_exp_flat,
    input  wire [ROWS*EXP_W-1:0]      as_exp_flat,
    input  wire [ROWS*S_W-1:0]        s_sum_flat,
    output wire [ROWS*ACC_W-1:0]      acc_flat
);
    genvar r;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_lane
            wire signed [ACC_W-1:0] acc_r;
            fp_fixed_mac #(
                .WS_SIG_W(WS_SIG_W), .AS_SIG_W(AS_SIG_W),
                .S_W(S_W), .EXP_W(EXP_W), .ACC_W(ACC_W)
            ) u_lane (
                .clk(clk), .rst_n(rst_n), .clear(clear), .valid_in(valid_in),
                .ws_sig(ws_sig_flat[r*WS_SIG_W +: WS_SIG_W]),
                .as_sig(as_sig_flat[r*AS_SIG_W +: AS_SIG_W]),
                .ws_exp(ws_exp_flat[r*EXP_W +: EXP_W]),
                .as_exp(as_exp_flat[r*EXP_W +: EXP_W]),
                .s_sum(s_sum_flat[r*S_W +: S_W]),
                .acc(acc_r)
            );
            assign acc_flat[r*ACC_W +: ACC_W] = acc_r;
        end
    endgenerate
endmodule
