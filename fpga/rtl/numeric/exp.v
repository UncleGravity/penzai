// numeric/exp - hardware exp for softmax: y ~= exp(x) for x <= 0, output in (0,1].
// Composes fmul + numeric/interp with guard delay derived from fmt.vh. Effective
// valid_in -> valid_out pipeline depth is 19 stages (mul3 + fx1 + lut1 + interp13 +
// out1). Migration was gated bit-identical against the former standalone exp leaf.
//
//   a = |x|·log2e ; ai=floor(a), af=a-ai ; 2^-af = lut+lerp(af) ; y = ldexp(2^-af, -ai)
//   guards: x >= 0 -> 1.0 ; x <= -87 -> 0.

`default_nettype none

module exp (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] x,
    output reg         valid_out,
    output reg  [31:0] y
);
    `include "fmt.vh"
    `include "flash_luts.vh"

    localparam [31:0] LOG2E   = 32'h3FB8AA3B;
    localparam [31:0] ONE_F32 = 32'h3F800000;
    localparam [31:0] X87     = 32'h42AE0000;

    // ---- stage 0: guards + |x| ----
    wire        sgn  = x[31];
    wire [30:0] xabs = x[30:0];
    wire [31:0] ax   = {1'b0, xabs};
    wire g_one_0  = (~sgn) | (xabs == 31'd0);
    wire g_zero_0 = sgn & (ax >= X87);

    // ---- a = |x|·log2e ----
    wire        a_v;
    wire [31:0] a_fp;
    fmul #(.MANT_W(FMT_FP32_MANT)) u_log2e (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(ax), .b(LOG2E), .valid_out(a_v), .out(a_fp)
    );
    // guards delayed FP32_MUL_LATENCY to meet a_v.
    reg [FP32_MUL_LATENCY-1:0] g_one_sr, g_zero_sr;
    always @(posedge clk) begin
        g_one_sr  <= {g_one_sr[FP32_MUL_LATENCY-2:0],  g_one_0};
        g_zero_sr <= {g_zero_sr[FP32_MUL_LATENCY-2:0], g_zero_0};
    end
    wire g_one_d  = g_one_sr[FP32_MUL_LATENCY-1];
    wire g_zero_d = g_zero_sr[FP32_MUL_LATENCY-1];

    // ---- fp32 -> Q8.16 fixed ----
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
        g1_fx <= g_one_d; g0_fx <= g_zero_d;
        ai_fx <= ai_c; idx_fx <= idx_c; t_fx <= t_c;
    end

    // ---- LUT read ----
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

    // ---- lerp ----
    wire        i_v;
    wire [31:0] frac;
    wire [9:0]  meta_o;
    interp #(.META_W(10)) u_interp (
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
