// fp_addtree - feed-forward fp32 sum of 16 lanes: sum = Σ_{i<16} in[i].
//
//   16 → 8 → 4 → 2 → 1, four levels of pipelined fp32 add, NO recurrence.
//
// The kv-major flash_kernel streams a head's per-beat fp_dot partials into a 16-deep
// buffer (head_dim_q/8 ≤ 16 beats) and sums them here in one shot — replacing v1's
// per-beat sequential accumulate (the ~320-cyc/kv dot). Unused lanes (head_dim_q < 128)
// are zero-padded by the caller; fp32 `+0` is exact, so the pad is free.
//
// Latency valid_in → valid_out: 4·ADD_LAT = 4·4 = 16.

`default_nettype none

module fp_addtree (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [511:0] in,        // 16 × f32, lane i = in[i*32 +: 32]
    output wire         valid_out,
    output wire [31:0]  sum
);
    wire [31:0] l0 [0:15];
    wire [31:0] l1 [0:7];
    wire [31:0] l2 [0:3];
    wire [31:0] l3 [0:1];
    wire [7:0]  v1;
    wire [3:0]  v2;
    wire [1:0]  v3;

    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : g_in
            assign l0[i] = in[i*32 +: 32];
        end
        // level 1: 16 → 8
        for (i = 0; i < 8; i = i + 1) begin : g_l1
            fp32_add_pipe u (.clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(l0[2*i]), .b(l0[2*i+1]), .valid_out(v1[i]), .out(l1[i]));
        end
        // level 2: 8 → 4
        for (i = 0; i < 4; i = i + 1) begin : g_l2
            fp32_add_pipe u (.clk(clk), .rst_n(rst_n), .valid_in(v1[2*i]),
                .a(l1[2*i]), .b(l1[2*i+1]), .valid_out(v2[i]), .out(l2[i]));
        end
        // level 3: 4 → 2
        for (i = 0; i < 2; i = i + 1) begin : g_l3
            fp32_add_pipe u (.clk(clk), .rst_n(rst_n), .valid_in(v2[2*i]),
                .a(l2[2*i]), .b(l2[2*i+1]), .valid_out(v3[i]), .out(l3[i]));
        end
    endgenerate
    // level 4: 2 → 1
    fp32_add_pipe u_l4 (.clk(clk), .rst_n(rst_n), .valid_in(v3[0]),
        .a(l3[0]), .b(l3[1]), .valid_out(valid_out), .out(sum));

    // lanes are synchronized; one representative valid per level drives the next.
    wire _unused = &{1'b0, v1[7:1], v2[3:1], v3[1]};
endmodule
