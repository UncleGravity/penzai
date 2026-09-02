// Valid-only arithmetic abstractions for the swiglu stream-control proof.
// Data is unconstrained; each leaf retains its production fixed latency and reset.

`default_nettype none

module swiglu_harness_pipe #(
    parameter integer WIDTH = 32,
    parameter integer LATENCY = 1
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    output wire             valid_out,
    output wire [WIDTH-1:0] out
);
    reg [LATENCY-1:0] valid_pipe;
    (* anyseq *) reg [WIDTH-1:0] arbitrary_out;

    always @(posedge clk) begin
        if (!rst_n)
            valid_pipe <= {LATENCY{1'b0}};
        else
            valid_pipe <= (valid_pipe << 1) | valid_in;
    end

    assign valid_out = valid_pipe[LATENCY-1];
    assign out = arbitrary_out;
endmodule

module fmul #(
    parameter integer MANT_W = 23,
    parameter integer MUL_PIPE = 0
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire [MANT_W+8:0] a,
    input  wire [MANT_W+8:0] b,
    output wire              valid_out,
    output wire [MANT_W+8:0] out
);
    swiglu_harness_pipe #(
        .WIDTH(MANT_W + 9),
        .LATENCY(3 + MUL_PIPE)
    ) u_pipe (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .valid_out(valid_out), .out(out)
    );
    wire _unused = &{1'b0, a, b};
endmodule

module fadd #(
    parameter integer MANT_W = 23
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire [MANT_W+8:0] a,
    input  wire [MANT_W+8:0] b,
    output wire              valid_out,
    output wire [MANT_W+8:0] out
);
    swiglu_harness_pipe #(
        .WIDTH(MANT_W + 9),
        .LATENCY(4)
    ) u_pipe (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .valid_out(valid_out), .out(out)
    );
    wire _unused = &{1'b0, a, b};
endmodule

`default_nettype wire
