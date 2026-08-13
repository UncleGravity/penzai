// Registered-I/O OOC wrapper for the P2f scalar section SwiGLU leaf.

`default_nettype none

module section_swiglu_ooc (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        abort,
    input  wire        in_valid,
    input  wire [31:0] in_gate,
    input  wire [31:0] in_up,
    input  wire        in_last,
    input  wire        out_ready,
    output reg         in_ready_q,
    output reg         out_valid_q,
    output reg  [31:0] out_data_q,
    output reg         out_last_q,
    output reg  [1:0]  out_status_q
);
    reg abort_q;
    reg in_valid_q;
    reg [31:0] in_gate_q;
    reg [31:0] in_up_q;
    reg in_last_q;
    reg out_ready_q;
    always @(posedge clk) begin
        abort_q <= abort;
        in_valid_q <= in_valid;
        in_gate_q <= in_gate;
        in_up_q <= in_up;
        in_last_q <= in_last;
        out_ready_q <= out_ready;
    end

    wire in_ready_w;
    wire out_valid_w;
    wire [31:0] out_data_w;
    wire out_last_w;
    wire [1:0] out_status_w;
    section_swiglu u_swiglu (
        .clk(clk), .rst_n(rst_n), .abort(abort_q),
        .in_valid(in_valid_q), .in_ready(in_ready_w),
        .in_gate(in_gate_q), .in_up(in_up_q), .in_last(in_last_q),
        .out_valid(out_valid_w), .out_ready(out_ready_q),
        .out_data(out_data_w), .out_last(out_last_w), .out_status(out_status_w)
    );

    always @(posedge clk) begin
        in_ready_q <= in_ready_w;
        out_valid_q <= out_valid_w;
        out_data_q <= out_data_w;
        out_last_q <= out_last_w;
        out_status_q <= out_status_w;
    end
endmodule

`default_nettype wire
