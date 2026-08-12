// OOC resource/timing probe for the exact-RNE FP32 -> Q8_0 leaf.
// All external signals are registered so synthesis times only internal reg paths.

`default_nettype none

module q8_quantizer_ooc (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    input  wire [31:0]  in_data,
    input  wire         in_last,
    input  wire         out_ready,
    output reg          in_ready_q,
    output reg          out_valid_q,
    output reg  [255:0] out_quants_q,
    output reg  [15:0]  out_scale_q,
    output reg  [3:0]   out_status_q
);
    reg in_valid_q;
    reg [31:0] in_data_q;
    reg in_last_q;
    reg out_ready_q;
    always @(posedge clk) begin
        in_valid_q <= in_valid;
        in_data_q <= in_data;
        in_last_q <= in_last;
        out_ready_q <= out_ready;
    end

    wire in_ready;
    wire out_valid;
    wire [255:0] out_quants;
    wire [15:0] out_scale;
    wire [3:0] out_status;
    q8_quantizer u_quantizer (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid_q), .in_ready(in_ready),
        .in_data(in_data_q), .in_last(in_last_q),
        .out_valid(out_valid), .out_ready(out_ready_q),
        .out_quants(out_quants), .out_scale(out_scale), .out_status(out_status)
    );

    always @(posedge clk) begin
        in_ready_q <= in_ready;
        out_valid_q <= out_valid;
        out_quants_q <= out_quants;
        out_scale_q <= out_scale;
        out_status_q <= out_status;
    end
endmodule

`default_nettype wire
