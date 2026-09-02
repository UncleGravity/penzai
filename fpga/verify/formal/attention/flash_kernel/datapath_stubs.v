// Control-only models for flash_kernel formal verification. Arithmetic values are
// deliberately unconstrained; each leaf preserves the production contract that
// accepted operations complete in order after a fixed, feed-forward latency.

`default_nettype none

module flash_formal_pipe #(
    parameter integer LATENCY = 2
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire valid_out
);
    reg [LATENCY-1:0] valid_q;
    always @(posedge clk) begin
        if (!rst_n)
            valid_q <= {LATENCY{1'b0}};
        else
            valid_q <= {valid_q[LATENCY-2:0], valid_in};
    end
    assign valid_out = valid_q[LATENCY-1];
endmodule

module fp_dot (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [255:0] q, input wire [127:0] k,
    output wire valid_out, output wire [31:0] sum
);
    flash_formal_pipe #(.LATENCY(3)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_sum;
    assign sum = {32{arbitrary_sum}};
    wire _unused = &{1'b0, q, k};
endmodule

module reduce #(
    parameter integer MANT_W = 23,
    parameter integer N = 16
) (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [N*(MANT_W+9)-1:0] in,
    output wire valid_out, output wire [MANT_W+8:0] sum
);
    flash_formal_pipe #(.LATENCY(2)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_sum;
    assign sum = {(MANT_W+9){arbitrary_sum}};
    wire _unused = &{1'b0, in};
endmodule

module fmul #(
    parameter integer MANT_W = 23,
    parameter integer MUL_PIPE = 0
) (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [MANT_W+8:0] a, input wire [MANT_W+8:0] b,
    output wire valid_out, output wire [MANT_W+8:0] out
);
    flash_formal_pipe #(.LATENCY(2)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_out;
    assign out = {(MANT_W+9){arbitrary_out}};
    wire _unused = &{1'b0, a, b};
endmodule

module fadd #(
    parameter integer MANT_W = 23
) (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [MANT_W+8:0] a, input wire [MANT_W+8:0] b,
    output wire valid_out, output wire [MANT_W+8:0] out
);
    flash_formal_pipe #(.LATENCY(2)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_out;
    assign out = {(MANT_W+9){arbitrary_out}};
    wire _unused = &{1'b0, a, b};
endmodule

module flash_softmax (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [31:0] m_in, input wire [31:0] l_in, input wire [31:0] score,
    output wire valid_out, output wire [31:0] m_out,
    output wire [31:0] l_out, output wire [31:0] p,
    output wire [31:0] corr, output wire grew
);
    // Accept a tuple every cycle and preserve completion order. Four independent
    // query/head slots can therefore be in flight in the reduced controller proof. The
    // arithmetic pipeline's exact production latency is a leaf-cosim contract;
    // the controller proof needs only a fixed, feed-forward latency.
    flash_formal_pipe #(.LATENCY(12)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_out;
    assign {m_out, l_out, p, corr, grew} = {129{arbitrary_out}};
    wire _unused = &{1'b0, m_in, l_in, score};
endmodule

module recip (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [31:0] l, output wire valid_out, output wire [31:0] y
);
    flash_formal_pipe #(.LATENCY(2)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_y;
    assign y = {32{arbitrary_y}};
    wire _unused = &{1'b0, l};
endmodule

module fp_axpy8 (
    input wire clk, input wire rst_n, input wire valid_in,
    input wire [255:0] acc, input wire [127:0] v,
    input wire [31:0] s1, input wire [31:0] p,
    output wire valid_out, output wire [255:0] out
);
    flash_formal_pipe #(.LATENCY(3)) u_v (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in), .valid_out(valid_out));
    (* anyseq *) reg arbitrary_out;
    assign out = {256{arbitrary_out}};
    wire _unused = &{1'b0, acc, v, s1, p};
endmodule

module cvt_f16_f32 (
    input wire [15:0] in,
    output wire [31:0] out
);
    assign out = {in, 16'd0};
endmodule

`default_nettype wire
