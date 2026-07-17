// gemm_rb_ooc - OOC timing probe for gemm_rowblock (pipelined front-end + COLS_MAX bank).
//
// The risk: gemm_rowblock computes Σ±a (32-element signed add tree) + f16 decompose
// into fma's input register (now PIPELINED FE_LAT=3), and — since the COLS_MAX bank —
// a per-row saturating accumulate that reads/writes acc[row][wcol] (a COLS_MAX:1 read mux
// + 1:COLS_MAX write demux per row). Both paths must close f300. To time them, register
// ALL inputs so `input_reg → front-end/mulshift/bank` is a real reg→reg path (ooc_synth.tcl
// false-paths the top I/O, so unregistered logic would vanish from the report). Output
// registered for symmetry.
//
// RE-RUN after any datapath change (the bank added accumulators + the column mux vs the
// decode-only version that OOC'd at 399 MHz) — cosim cannot see timing.

`default_nettype none

module gemm_rb_ooc #(
    parameter integer COLS_MAX = 8,
    parameter integer CW       = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        valid_in,
    input  wire [CW-1:0]      col_idx,
    input  wire signed [7:0]  emin,
    input  wire [16*32-1:0]   weight_bits_flat,
    input  wire [16*16-1:0]   weight_scales_flat,
    input  wire [255:0]       acts_packed,
    input  wire [15:0]        act_scale,
    input  wire [CW-1:0]      read_col,
    output wire [16*104-1:0]   acc_flat
);
    reg        clear_q, valid_q;
    reg [CW-1:0]      col_q, rdcol_q;
    reg signed [7:0]  emin_q;
    reg [16*32-1:0]   wb_q;
    reg [16*16-1:0]   ws_q;
    reg [255:0]       acts_q;
    reg [15:0]        as_q;
    always @(posedge clk) begin
        clear_q <= clear; valid_q <= valid_in; emin_q <= emin;
        col_q <= col_idx; rdcol_q <= read_col;
        wb_q <= weight_bits_flat; ws_q <= weight_scales_flat;
        acts_q <= acts_packed; as_q <= act_scale;
    end

    wire [16*104-1:0] acc_w;
    gemm_rowblock #(.ROWS(16), .COLS_MAX(COLS_MAX)) u_rb (
        .clk(clk), .rst_n(rst_n), .clear(clear_q), .valid_in(valid_q),
        .col_idx(col_q), .emin(emin_q),
        .weight_bits_flat(wb_q), .weight_nonzero_flat({16*32{1'b1}}), .weight_scales_flat(ws_q),
        .acts_packed(acts_q), .act_scale(as_q),
        .read_col(rdcol_q), .acc_flat(acc_w)
    );

    reg [16*104-1:0] acc_q;
    always @(posedge clk) acc_q <= acc_w;
    assign acc_flat = acc_q;
endmodule
