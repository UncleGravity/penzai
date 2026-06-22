// fp_exp - hardware exp for softmax: y ~= exp(x) for x <= 0, output in (0,1].
//
// Range-reduce on a = |x| (>= 0) so the floor is on a positive number:
//   a    = |x| * log2e                     (fp32 multiply)
//   ai   = floor(a) , af = a - ai          (fp32 -> Q8.16 fixed; af in [0,1))
//   2^-af ~= lut+lerp(af)                  (fp_interp over FLASH_EXP_LUT)
//   y    = 2^-af * 2^-ai = ldexp(.,-ai)    (subtract ai from the exponent; underflow -> 0)
// Guards: x >= 0 -> 1.0 ; x <= -87 -> 0. Modeled exactly by flash_ref.softmaxExp.

`default_nettype none

module fp_exp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] x,
    output reg         valid_out,
    output reg  [31:0] y
);
    `include "flash_luts.vh"

    localparam [31:0] LOG2E   = 32'h3FB8AA3B; // 1.4426950
    localparam [31:0] ONE_F32 = 32'h3F800000;
    localparam [31:0] X87     = 32'h42AE0000; // 87.0

    // ---- stage 0: guards + |x| ----
    wire        sgn  = x[31];
    wire [30:0] xabs = x[30:0];
    wire [31:0] ax   = {1'b0, xabs};
    wire g_one_0  = (~sgn) | (xabs == 31'd0);          // x >= 0 (or +-0) -> exp = 1
    wire g_zero_0 = sgn & (ax >= X87);                 // x <= -87 -> 0

    // ---- a = |x| * log2e (MUL_LAT = 3) ----
    wire        a_v;
    wire [31:0] a_fp;
    fp32_mul_pipe u_log2e (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(ax), .b(LOG2E), .valid_out(a_v), .out(a_fp)
    );
    // guards delayed to meet a_v (fp32_mul_pipe latency 3, was 2).
    reg g_one_m0, g_one_m1, g_one_m2, g_zero_m0, g_zero_m1, g_zero_m2;
    always @(posedge clk) begin
        g_one_m0  <= g_one_0;  g_one_m1  <= g_one_m0;  g_one_m2  <= g_one_m1;
        g_zero_m0 <= g_zero_0; g_zero_m1 <= g_zero_m0; g_zero_m2 <= g_zero_m1;
    end

    // ---- fp32 -> Q8.16 fixed (a_fp >= 0; live range exp <= 133 -> right shift) ----
    wire [7:0]  e_a = a_fp[30:23];
    wire [23:0] M_a = {1'b1, a_fp[22:0]};
    wire [7:0]  shr = (e_a <= 8'd133) ? (8'd134 - e_a) : 8'd0;
    wire [23:0] fx  = (e_a >= 8'd134) ? 24'hFFFFFF : (M_a >> shr);
    wire [7:0]  ai_c  = fx[23:16];
    wire [7:0]  idx_c = fx[15:8];
    wire [7:0]  t_c   = fx[7:0];

    reg        v_fx, g1_fx, g0_fx;
    reg [7:0]  ai_fx, idx_fx, t_fx;
    always @(posedge clk) begin
        if (!rst_n) v_fx <= 1'b0;
        else        v_fx <= a_v;
        g1_fx <= g_one_m2; g0_fx <= g_zero_m2;
        ai_fx <= ai_c; idx_fx <= idx_c; t_fx <= t_c;
    end

    // ---- LUT read (one registered stage) ----
    reg        v_lut, g1_lut, g0_lut;
    reg [7:0]  ai_lut, t_lut;
    reg [31:0] lo_lut, hi_lut;
    always @(posedge clk) begin
        if (!rst_n) v_lut <= 1'b0;
        else        v_lut <= v_fx;
        g1_lut <= g1_fx; g0_lut <= g0_fx; ai_lut <= ai_fx; t_lut <= t_fx;
        lo_lut <= FLASH_EXP_LUT[idx_fx*32 +: 32];
        hi_lut <= FLASH_EXP_LUT[(idx_fx + 8'd1)*32 +: 32];
    end

    // ---- lerp (meta carries the guards + ai through to the ldexp) ----
    wire        i_v;
    wire [31:0] frac;
    wire [9:0]  meta_o;
    fp_interp #(.META_W(10)) u_interp (
        .clk(clk), .rst_n(rst_n), .valid_in(v_lut),
        .lo(lo_lut), .hi(hi_lut), .t(t_lut),
        .meta({g1_lut, g0_lut, ai_lut}),
        .valid_out(i_v), .frac(frac), .meta_out(meta_o)
    );

    // ---- ldexp(frac, -ai) + guard mux ----
    wire        g1  = meta_o[9];
    wire        g0  = meta_o[8];
    wire [7:0]  aio = meta_o[7:0];
    wire [7:0]  ef  = frac[30:23];
    wire signed [9:0] ne = $signed({2'b00, ef}) - $signed({2'b00, aio});
    wire [31:0] y_calc = (ne <= 10'sd0) ? 32'd0 : {1'b0, ne[7:0], frac[22:0]};
    wire [31:0] y_mux  = g1 ? ONE_F32 : (g0 ? 32'd0 : y_calc);

    always @(posedge clk) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= i_v;
        y <= y_mux;
    end

endmodule
