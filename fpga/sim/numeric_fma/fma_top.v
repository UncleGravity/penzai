// fma_top - cosim-only wrapper for numeric/fma: pins the gemm params and splits the
// 104-bit accumulator into four 32-bit ports so the tb reads it without VlWide plumbing
// (still exercises the full ACC_W=104 datapath). Gated vs matmul_ref.windowedRow.

`default_nettype none

module fma_top (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        valid_in,
    input  wire signed [11:0] ws_sig,
    input  wire signed [11:0] as_sig,
    input  wire signed [7:0]  p_exp,
    input  wire signed [7:0]  emin,
    input  wire signed [13:0] s_sum,
    output wire [31:0] acc0,
    output wire [31:0] acc1,
    output wire [31:0] acc2,
    output wire [31:0] acc3
);
    wire signed [103:0] acc_w;
    fma #(.SIG_W(12), .S_W(14), .EXP_W(8), .ACC_W(104), .SHAMT_W(7)) u (
        .clk(clk), .rst_n(rst_n), .clear(clear), .valid_in(valid_in),
        .ws_sig(ws_sig), .as_sig(as_sig), .p_exp(p_exp), .emin(emin), .s_sum(s_sum),
        .acc(acc_w)
    );
    assign acc0 = acc_w[31:0];
    assign acc1 = acc_w[63:32];
    assign acc2 = acc_w[95:64];
    assign acc3 = {24'd0, acc_w[103:96]};
endmodule
