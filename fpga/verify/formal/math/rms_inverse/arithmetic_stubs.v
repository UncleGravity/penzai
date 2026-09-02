// Latency-faithful arithmetic abstractions for inverse-RMS control proofs.
// Numeric values are checked bit-exactly in the native cosim; formal keeps the
// production valid timing and a positive-normal payload so lifecycle paths stay
// nonvacuous without expanding two FP multipliers into the SMT cone.

`default_nettype none

module rms_inverse_harness_pipe #(
    parameter integer LATENCY = 1
) (
    input  wire clk,
    input  wire rst_n,
    input  wire valid_in,
    output wire valid_out,
    output wire [31:0] out
);
    reg [LATENCY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (!rst_n)
            valid_pipe <= {LATENCY{1'b0}};
        else
            valid_pipe <= (valid_pipe << 1) | valid_in;
    end
    assign valid_out = valid_pipe[LATENCY-1];
    assign out = 32'h3f80_0000;
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
    wire [31:0] pipe_out;
    rms_inverse_harness_pipe #(.LATENCY(3 + MUL_PIPE)) u_pipe (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .valid_out(valid_out), .out(pipe_out)
    );
    assign out = pipe_out[MANT_W+8:0];
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
    wire [31:0] pipe_out;
    rms_inverse_harness_pipe #(.LATENCY(4)) u_pipe (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .valid_out(valid_out), .out(pipe_out)
    );
    assign out = pipe_out[MANT_W+8:0];
    wire _unused = &{1'b0, a, b};
endmodule

`default_nettype wire
