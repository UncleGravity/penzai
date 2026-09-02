// Combinational, truncating numeric conversions.
//
//   cvt_f16_f32 : f16 -> fp32 widen (attention K/V/mask)
//   cvt_i2f     : signed int(WIDTH) -> fp32 (interpolation input)
//   cvt_f32_bf16/cvt_bf16_f32 : bf16 narrow/widen

`default_nettype none

// f16 -> fp32. Subnormals flush to signed zero; Inf/NaN are not expected.
module cvt_f16_f32 (
    input  wire [15:0] in,
    output wire [31:0] out
);
    wire       sign    = in[15];
    wire [4:0] exp_in  = in[14:10];
    wire [9:0] mant_in = in[9:0];
    wire       is_zero_sub = (exp_in == 5'd0);
    wire [7:0]  exp_out  = {3'd0, exp_in} + 8'd112; // rebias 15 -> 127
    wire [22:0] mant_out = {mant_in, 13'd0};
    assign out = is_zero_sub ? {sign, 31'd0} : {sign, exp_out, mant_out};
endmodule

// signed int(WIDTH) -> fp32, truncating toward zero. WIDTH<=24 keeps the
// magnitude exact in the fp32 mantissa.
module cvt_i2f #(
    parameter integer WIDTH = 14
) (
    input  wire signed [WIDTH-1:0] in,
    output wire        [31:0]      out
);
    localparam integer MAG_W = WIDTH - 1;

    wire                    sign   = in[WIDTH-1];
    wire signed [WIDTH-1:0] neg_in = -in;
    wire [MAG_W-1:0]        mag    = sign ? neg_in[MAG_W-1:0] : in[MAG_W-1:0];

    reg [4:0] msb_pos;
    integer i;
    always @(*) begin
        msb_pos = 5'd0;
        for (i = 0; i < MAG_W; i = i + 1)
            if (mag[i]) msb_pos = i[4:0];
    end

    wire [MAG_W-1:0] mag_norm = mag << ((MAG_W - 1) - {27'd0, msb_pos});
    wire [22:0]      mantissa = {mag_norm[MAG_W-2:0], {(23 - (MAG_W-1)){1'b0}}};
    wire [7:0]       exponent = 8'd127 + {3'd0, msb_pos};
    wire             is_zero  = (mag == {MAG_W{1'b0}});

    assign out = is_zero ? 32'd0 : {sign, exponent, mantissa};
endmodule

// f32 -> bf16 narrow: bf16 is the top 16 bits of an f32 — { sign, exp[8], mant[22:16] }.
// Truncating (drop the low 16 mantissa bits) to match the leaves' house style and keep
// bit-exactness vs a truncating reference (no round-to-nearest-even).
module cvt_f32_bf16 (
    input  wire [31:0] in,
    output wire [15:0] out
);
    assign out = in[31:16];
endmodule

// bf16 -> f32 widen: zero-extend the mantissa (bf16 occupies the high 16 bits of an f32).
// Exact / lossless: { bf16, 16'd0 }.
module cvt_bf16_f32 (
    input  wire [15:0] in,
    output wire [31:0] out
);
    assign out = {in, 16'd0};
endmodule
