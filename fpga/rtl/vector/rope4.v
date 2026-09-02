// Four-token FP32 RoPE pair pipeline.
//
// Each accepted record rotates one complex pair for each physical token lane.
// Cos/sin coefficients come from the immutable model RoPE table and may differ
// by token position. The controller serializes heads and pair indices; this
// leaf remains an II=1 arithmetic pipe with no transaction metadata.

`default_nettype none

module rope4 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          in_valid,
    input  wire [255:0]  in_data,
    input  wire [255:0]  in_coeff,
    output wire          out_valid,
    output wire [255:0]  out_data
);
    wire [3:0] mul_valid;
    wire [3:0] add0_valid;
    wire [3:0] add1_valid;

    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_lane
            wire [31:0] x0 = in_data[lane*64 +: 32];
            wire [31:0] x1 = in_data[lane*64 + 32 +: 32];
            wire [31:0] cos_value = in_coeff[lane*64 +: 32];
            wire [31:0] sin_value = in_coeff[lane*64 + 32 +: 32];
            wire [31:0] x0_cos;
            wire [31:0] x1_sin;
            wire [31:0] x0_sin;
            wire [31:0] x1_cos;
            wire mul0_valid;
            wire mul1_valid;
            wire mul2_valid;
            wire mul3_valid;

            fmul u_x0_cos (
                .clk(clk), .rst_n(rst_n), .valid_in(in_valid),
                .a(x0), .b(cos_value), .valid_out(mul0_valid), .out(x0_cos)
            );
            fmul u_x1_sin (
                .clk(clk), .rst_n(rst_n), .valid_in(in_valid),
                .a(x1), .b(sin_value), .valid_out(mul1_valid), .out(x1_sin)
            );
            fmul u_x0_sin (
                .clk(clk), .rst_n(rst_n), .valid_in(in_valid),
                .a(x0), .b(sin_value), .valid_out(mul2_valid), .out(x0_sin)
            );
            fmul u_x1_cos (
                .clk(clk), .rst_n(rst_n), .valid_in(in_valid),
                .a(x1), .b(cos_value), .valid_out(mul3_valid), .out(x1_cos)
            );

            assign mul_valid[lane] = mul0_valid;

            fadd u_y0 (
                .clk(clk), .rst_n(rst_n), .valid_in(mul0_valid),
                .a(x0_cos), .b({~x1_sin[31], x1_sin[30:0]}),
                .valid_out(add0_valid[lane]),
                .out(out_data[lane*64 +: 32])
            );
            fadd u_y1 (
                .clk(clk), .rst_n(rst_n), .valid_in(mul0_valid),
                .a(x0_sin), .b(x1_cos),
                .valid_out(add1_valid[lane]),
                .out(out_data[lane*64 + 32 +: 32])
            );

`ifndef SYNTHESIS
            always @(posedge clk) begin
                if (rst_n && ((mul1_valid != mul0_valid) ||
                              (mul2_valid != mul0_valid) ||
                              (mul3_valid != mul0_valid)))
                    $fatal(1, " rope4 multiplier valid skew");
                if (rst_n && (add1_valid[lane] != add0_valid[lane]))
                    $fatal(1, " rope4 adder valid skew");
            end
`endif
        end
    endgenerate

    assign out_valid = add0_valid[0];

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && ((mul_valid != {4{mul_valid[0]}}) ||
                      (add0_valid != {4{add0_valid[0]}})))
            $fatal(1, " rope4 lane valid skew");
    end
`endif
endmodule

`default_nettype wire
