// rowblock_ooc - OOC synth wrapper for the CURRENT matmul_rowblock (the baseline
// the phase-2 fixed-point rewrite replaces). Pinned to the shipping decode config
// (ROWS=16, COLS_MAX=8, ACCUM_DEPTH=5 per the combined-v1 f250 build) so the area /
// timing numbers are apples-to-apples with mac_array (ROWS=16).
//
// Pure pass-through wrapper: just fixes the params and re-exports the ports so
// synth_design -top rowblock_ooc reproduces one rowblock's worth of logic.

`default_nettype none

module rowblock_ooc (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire               valid_in,
    input  wire               last_in,
    input  wire               single_col,
    input  wire [2:0]         col_idx,            // CW = clog2(COLS_MAX=8) = 3
    input  wire [16*32-1:0]   weight_bits_flat,
    input  wire [16*16-1:0]   weight_scales_flat,
    input  wire [255:0]       acts_packed,
    input  wire [15:0]        act_scale,
    input  wire [2:0]         read_col,
    output wire               done,
    output wire [16*32-1:0]   results_flat
);
    wire done_w;
    matmul_rowblock #(
        .ROWS(16), .COLS_MAX(8), .ACCUM_DEPTH(5)
    ) u (
        .clk(clk), .rst_n(rst_n), .start(start),
        .valid_in(valid_in), .last_in(last_in), .single_col(single_col),
        .col_idx(col_idx),
        .weight_bits_flat(weight_bits_flat),
        .weight_scales_flat(weight_scales_flat),
        .acts_packed(acts_packed), .act_scale(act_scale),
        .read_col(read_col),
        .done(done_w),
        .results_flat(results_flat)
    );
    assign done = done_w;
endmodule
