// fp_axpy8 - 8-lane fused accumulator update: out[i] = acc[i]*s1 + p*f32(v[i]).
//
//   axpy step:  s1 = corr, p = weight, v = a V row  -> acc[i]*corr + p*V[i]
//   scale-only: p = 0                               -> acc[i]*s1 (e.g. emit ·1/l)
//
// Feed-forward (no accumulation), 8 lanes in parallel, latency MUL_LAT + ADD_LAT =
// 2 + 4 = 6. The kernel streams one acc beat (8 elements) per cycle through this and
// writes the result back — replacing the v1 one-element-at-a-time walk.

`default_nettype none

module fp_axpy8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] acc,   // 8 × f32
    input  wire [127:0] v,     // 8 × f16
    input  wire [31:0]  s1,    // corr (axpy) or 1/l (emit)
    input  wire [31:0]  p,     // weight (axpy); 0 for scale-only
    output wire         valid_out,
    output wire [255:0] out    // 8 × f32
);
    wire [7:0] add_v;
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lane
            wire [31:0] vf, t1, t2;
            wire        m1v, m2v;
            fp16_to_fp32 u_w (.in(v[i*16 +: 16]), .out(vf));
            fp32_mul_pipe u_m1 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(acc[i*32 +: 32]), .b(s1), .valid_out(m1v), .out(t1));
            fp32_mul_pipe u_m2 (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(p), .b(vf), .valid_out(m2v), .out(t2)); // m2v unused (synced to m1v)
            fp32_add_pipe u_a (.clk(clk), .rst_n(rst_n), .valid_in(m1v),
                .a(t1), .b(t2), .valid_out(add_v[i]), .out(out[i*32 +: 32]));
        end
    endgenerate

    assign valid_out = add_v[0]; // lanes synchronized
    wire _unused = &{1'b0, add_v[7:1]};
endmodule
