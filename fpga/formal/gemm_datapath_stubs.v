`default_nettype none

module gemm_rowblock #(
    parameter integer ROWS = 2,
    parameter integer COLS_MAX = 2,
    parameter integer ACC_W = 16,
    parameter integer CW = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX)
) (
    input wire clk, input wire rst_n, input wire clear, input wire valid_in,
    input wire [CW-1:0] col_idx, input wire signed [7:0] emin,
    input wire [ROWS*32-1:0] weight_bits_flat,
    input wire [ROWS*32-1:0] weight_nonzero_flat,
    input wire [ROWS*16-1:0] weight_scales_flat,
    input wire [255:0] acts_packed, input wire [15:0] act_scale,
    input wire [CW-1:0] read_col,
    output wire [ROWS*ACC_W-1:0] acc_flat
);
    (* anyseq *) reg [ROWS*ACC_W-1:0] arbitrary_acc;
    assign acc_flat = arbitrary_acc;
    wire _unused = &{1'b0, clk, rst_n, clear, valid_in, col_idx, emin,
        weight_bits_flat, weight_nonzero_flat, weight_scales_flat, acts_packed, act_scale, read_col};
endmodule

module gemm_emit #(
    parameter integer ACC_W = 16,
    parameter integer EXP_W = 8
) (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire signed [ACC_W-1:0] acc,
    input wire signed [EXP_W-1:0] emin,
    output reg valid_out, output reg [31:0] f32
);
    reg [2:0] valid_pipe;
    (* anyseq *) reg [31:0] arbitrary_f32;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_pipe <= 3'b000;
            valid_out <= 1'b0;
            f32 <= 32'd0;
        end else begin
            valid_pipe <= {valid_pipe[1:0], valid_in};
            valid_out <= valid_pipe[2];
            f32 <= arbitrary_f32;
        end
    end
    wire _unused = &{1'b0, acc, emin};
endmodule

`default_nettype wire
