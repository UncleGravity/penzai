// fp_dot - one beat of a Q·K dot product: sum over LANES of f32 Q × widened-f16 K.
//
//   sum = Σ_{i<LANES} q[i] · fp32(k[i])
//
// Pure feed-forward: widen → multiply → pipelined fp32 adder tree. NO accumulation
// across beats — the kernel sums the per-beat results into its per-head dot
// accumulator, where the n_heads-deep interleave hides the fp-add recurrence (the
// same pool that hides m/l/acc). That keeps recurrence-hiding in one place instead
// of duplicating matmul_rowblock's ACCUM_DEPTH machinery inside the leaf.
//
// LANES = 8 (a 128-bit K beat / 256-bit Q beat). Latency valid_in→valid_out:
// MUL_LAT + 3·ADD_LAT = 2 + 12 = 14.

`default_nettype none

module fp_dot (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] q,   // 8 × f32
    input  wire [127:0] k,   // 8 × f16
    output wire         valid_out,
    output wire [31:0]  sum
);
    localparam integer LANES = 8;

    // Widen each f16 K lane to f32 (combinational).
    wire [31:0] kf [0:LANES-1];
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_widen
            fp16_to_fp32 u_w (.in(k[i*16 +: 16]), .out(kf[i]));
        end
    endgenerate

    // Multiply lanes (all share valid_in, so they stay synchronized).
    wire [31:0]      prod [0:LANES-1];
    wire [LANES-1:0] prod_v;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_mul
            fp32_mul_pipe u_m (
                .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
                .a(q[i*32 +: 32]), .b(kf[i]),
                .valid_out(prod_v[i]), .out(prod[i])
            );
        end
    endgenerate

    // Adder tree 8 → 4 → 2 → 1. Each level's valid drives the next; lanes are
    // synchronized so one representative valid per pair suffices.
    wire [31:0] a0, a1, a2, a3;
    wire        av0, av1, av2, av3;
    fp32_add_pipe u_a0 (.clk(clk), .rst_n(rst_n), .valid_in(prod_v[0]), .a(prod[0]), .b(prod[1]), .valid_out(av0), .out(a0));
    fp32_add_pipe u_a1 (.clk(clk), .rst_n(rst_n), .valid_in(prod_v[2]), .a(prod[2]), .b(prod[3]), .valid_out(av1), .out(a1));
    fp32_add_pipe u_a2 (.clk(clk), .rst_n(rst_n), .valid_in(prod_v[4]), .a(prod[4]), .b(prod[5]), .valid_out(av2), .out(a2));
    fp32_add_pipe u_a3 (.clk(clk), .rst_n(rst_n), .valid_in(prod_v[6]), .a(prod[6]), .b(prod[7]), .valid_out(av3), .out(a3));

    wire [31:0] b0, b1;
    wire        bv0, bv1;
    fp32_add_pipe u_b0 (.clk(clk), .rst_n(rst_n), .valid_in(av0), .a(a0), .b(a1), .valid_out(bv0), .out(b0));
    fp32_add_pipe u_b1 (.clk(clk), .rst_n(rst_n), .valid_in(av2), .a(a2), .b(a3), .valid_out(bv1), .out(b1));

    fp32_add_pipe u_sum (.clk(clk), .rst_n(rst_n), .valid_in(bv0), .a(b0), .b(b1), .valid_out(valid_out), .out(sum));

    // Synchronized redundant valids (one representative per level is used).
    wire _unused = &{1'b0, prod_v[1], prod_v[3], prod_v[5], prod_v[7], av1, av3, bv1};
endmodule
