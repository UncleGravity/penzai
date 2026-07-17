// gemm_top - cosim-only wrapper for gemm_rowblock (increment 1: the decode datapath core).
//
// Pins ROWS, muxes one lane's 104-bit accumulator out as four 32-bit ports (read_row
// selects the lane — same VlWide-avoidance trick as fma_top), and exposes a standalone
// gemm_f16_decompose so the tb can sweep all 65536 f16 patterns against matmul_ref.decompose.
// Gated bit-exact vs matmul_ref.windowedRow (exact-in-window).

`default_nettype none

module gemm_top #(
    parameter integer ROWS     = 16,
    parameter integer COLS_MAX = 8,
    parameter integer ACC_W    = 104,
    parameter integer CW       = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX)
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        valid_in,
    input  wire [CW-1:0]           col_idx,      // accumulator column this issue targets
    input  wire signed [7:0]       emin,
    input  wire [ROWS*32-1:0]      weight_bits_flat,
    input  wire [ROWS*16-1:0]      weight_scales_flat,
    input  wire [255:0]            acts_packed,
    input  wire [15:0]             act_scale,
    input  wire [CW-1:0]           read_col,     // which column acc_flat exposes
    input  wire [4:0]              read_row,     // which lane's acc to expose
    output wire [31:0]             acc0,
    output wire [31:0]             acc1,
    output wire [31:0]             acc2,
    output wire [31:0]             acc3,
    // standalone decompose for the exhaustive f16 sweep
    input  wire [15:0]             dbg_f16,
    output wire [11:0]             dbg_sig,
    output wire [7:0]              dbg_e,
    // directed emit sweep: stream arbitrary acc + emin through the (pipelined) emit.
    // The decode emit is gated end-to-end by test-rtl-gemm-kernel; this keeps msb/sign
    // edge coverage the realistic-window kernel cases don't reach.
    input  wire                    dbg_emit_vin,
    input  wire [31:0]             dbg_acc0,
    input  wire [31:0]             dbg_acc1,
    input  wire [31:0]             dbg_acc2,
    input  wire [31:0]             dbg_acc3,
    input  wire signed [7:0]       dbg_emin,
    output wire                    dbg_emit_vout,
    output wire [31:0]             dbg_emit_f32
);
    wire [ROWS*ACC_W-1:0] acc_flat;
    gemm_rowblock #(.ROWS(ROWS), .COLS_MAX(COLS_MAX), .ACC_W(ACC_W)) u_rb (
        .clk(clk), .rst_n(rst_n), .clear(clear), .valid_in(valid_in),
        .col_idx(col_idx), .emin(emin),
        .weight_bits_flat(weight_bits_flat),
        .weight_nonzero_flat({ROWS*32{1'b1}}),
        .weight_scales_flat(weight_scales_flat),
        .acts_packed(acts_packed),
        .act_scale(act_scale),
        .read_col(read_col),
        .acc_flat(acc_flat)
    );

    wire [ACC_W-1:0] acc_sel = acc_flat[read_row*ACC_W +: ACC_W];
    assign acc0 = acc_sel[31:0];
    assign acc1 = acc_sel[63:32];
    assign acc2 = acc_sel[95:64];
    assign acc3 = {24'd0, acc_sel[103:96]};

    // directed sweep instance: {dbg_acc3[7:0], dbg_acc2, dbg_acc1, dbg_acc0} is a 104-bit acc.
    wire [ACC_W-1:0] dbg_acc = {dbg_acc3[ACC_W-97:0], dbg_acc2, dbg_acc1, dbg_acc0};
    gemm_emit #(.ACC_W(ACC_W), .EXP_W(8)) u_emit_dbg (
        .clk(clk), .rst_n(rst_n), .valid_in(dbg_emit_vin),
        .acc($signed(dbg_acc)), .emin(dbg_emin),
        .valid_out(dbg_emit_vout), .f32(dbg_emit_f32)
    );

    gemm_f16_decompose #(.SIG_W(12), .EXP_W(8)) u_dbg (
        .f16(dbg_f16), .sig(dbg_sig), .e(dbg_e)
    );
endmodule
