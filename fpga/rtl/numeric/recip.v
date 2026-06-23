// numeric/recip - hardware reciprocal for the softmax denominator: y ~= 1/l for l >= 1.
// Port of fp_recip onto numeric/interp (its only fp-leaf dependency). Bit-identical to
// fp_recip (gated by the differential cosim).
//
//   1/sig = lut+lerp(mantissa) ; y = ldexp(1/sig, -(exp-127)).

`default_nettype none

module recip (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] l,
    output reg         valid_out,
    output reg  [31:0] y
);
    `include "flash_luts.vh"

    // ---- stage 0: split l ----
    wire        lsgn = l[31];
    wire [7:0]  le   = l[30:23];
    wire [22:0] lm   = l[22:0];
    wire g0_0 = lsgn | (l[30:0] == 31'd0);
    wire [7:0] idx_c   = lm[22:15];
    wire [7:0] t_c     = lm[14:7];
    wire [7:0] shift_c = le - 8'd127;

    reg        v_s, g0_s;
    reg [7:0]  idx_s, t_s, shift_s;
    always @(posedge clk) begin
        if (!rst_n) v_s <= 1'b0;
        else        v_s <= valid_in;
        g0_s <= g0_0; idx_s <= idx_c; t_s <= t_c; shift_s <= shift_c;
    end

    // ---- LUT read ----
    reg        v_lut, g0_lut;
    reg [7:0]  t_lut, shift_lut;
    reg [31:0] lo_lut, hi_lut;
    always @(posedge clk) begin
        if (!rst_n) v_lut <= 1'b0;
        else        v_lut <= v_s;
        g0_lut <= g0_s; t_lut <= t_s; shift_lut <= shift_s;
        lo_lut <= FLASH_RECIP_LUT[idx_s*32 +: 32];
        hi_lut <= FLASH_RECIP_LUT[(idx_s + 8'd1)*32 +: 32];
    end

    // ---- lerp ----
    wire        i_v;
    wire [31:0] frac;
    wire [8:0]  meta_o;
    interp #(.META_W(9)) u_interp (
        .clk(clk), .rst_n(rst_n), .valid_in(v_lut),
        .lo(lo_lut), .hi(hi_lut), .t(t_lut),
        .meta({g0_lut, shift_lut}),
        .valid_out(i_v), .frac(frac), .meta_out(meta_o)
    );

    // ---- ldexp(frac, -e2) + guard ----
    wire        g0 = meta_o[8];
    wire [7:0]  sh = meta_o[7:0];
    wire [7:0]  ef = frac[30:23];
    wire signed [9:0] ne = $signed({2'b00, ef}) - $signed({2'b00, sh});
    wire [31:0] y_calc = (ne <= 10'sd0) ? 32'd0 : {1'b0, ne[7:0], frac[22:0]};
    wire [31:0] y_mux  = g0 ? 32'd0 : y_calc;

    always @(posedge clk) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= i_v;
        y <= y_mux;
    end
endmodule
