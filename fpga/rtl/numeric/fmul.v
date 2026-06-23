// fmul - MANT_W-parameterized truncating float multiply.
//
// fp32 (MANT_W=23) and bf16 (MANT_W=7) are the SAME module, different param; the
// exponent stays 8-bit (fp32 range) for both. Truncating, same semantics as the
// rtl/fp leaf it replaces. The (MANT_W+1)² mantissa product is forced into a DSP
// (use_dsp) and runs register-to-register, as in fp32_mul_pipe (the f250 fix).
//
// At MANT_W=23 this is bit-identical to fp32_mul_pipe (differential cosim is the gate).
// Latency valid_in -> valid_out: 3. Format: { sign[1], exp[8], mant[MANT_W] }.

`default_nettype none

module fmul #(
    parameter integer MANT_W = 23                 // FMT_FP32_MANT; 7 for bf16
) (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                valid_in,
    input  wire [MANT_W+8:0]   a,
    input  wire [MANT_W+8:0]   b,
    output reg                 valid_out,
    output reg  [MANT_W+8:0]   out
);
    localparam integer OUT_W = MANT_W + 9;
    localparam integer SIG_W = MANT_W + 1;        // significand incl. hidden 1
    localparam integer PW    = 2 * SIG_W;         // product width

    wire              sa = a[OUT_W-1];
    wire [7:0]        ea = a[OUT_W-2 -: 8];
    wire [MANT_W-1:0] ma = a[MANT_W-1:0];
    wire              sb = b[OUT_W-1];
    wire [7:0]        eb = b[OUT_W-2 -: 8];
    wire [MANT_W-1:0] mb = b[MANT_W-1:0];
    wire        is_zero = (ea == 8'd0) || (eb == 8'd0);
    wire        sr = sa ^ sb;
    wire signed [9:0] er_pre = $signed({2'b00, ea}) + $signed({2'b00, eb}) - 10'sd127;

    // ---- input-register stage: capture decoded operands BEFORE the multiply so the
    // mantissa product runs register-to-register (operands pack into the DSP A/B regs). ----
    reg              valid_i;
    reg              zero_i;
    reg              sign_i;
    reg signed [9:0] exp_pre_i;
    reg [SIG_W-1:0]  mant_a_i, mant_b_i;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_i <= 1'b0; zero_i <= 1'b0; sign_i <= 1'b0; exp_pre_i <= 10'sd0;
            mant_a_i <= 0; mant_b_i <= 0;
        end else begin
            valid_i   <= valid_in;
            zero_i    <= is_zero;
            sign_i    <= sr;
            exp_pre_i <= er_pre;
            mant_a_i  <= {1'b1, ma};
            mant_b_i  <= {1'b1, mb};
        end
    end

    // ---- product stage (DSP) ----
    reg              valid_s0;
    reg              zero_s0;
    reg              sign_s0;
    reg signed [9:0] exp_pre_s0;
    (* use_dsp = "yes" *) reg [PW-1:0] prod_s0;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0 <= 1'b0; zero_s0 <= 1'b0; sign_s0 <= 1'b0; exp_pre_s0 <= 10'sd0; prod_s0 <= 0;
        end else begin
            valid_s0   <= valid_i;
            zero_s0    <= zero_i;
            sign_s0    <= sign_i;
            exp_pre_s0 <= exp_pre_i;
            prod_s0    <= mant_a_i * mant_b_i;
        end
    end

    // ---- normalize (combinational) ----
    wire renorm_s0 = prod_s0[PW-1];
    wire [MANT_W-1:0] mant_s0 = renorm_s0 ? prod_s0[PW-2 -: MANT_W] : prod_s0[PW-3 -: MANT_W];
    wire signed [9:0] exp_s0 = exp_pre_s0 + (renorm_s0 ? 10'sd1 : 10'sd0);
    wire underflow_s0 = (exp_s0 <= 10'sd0);
    wire overflow_s0  = (exp_s0 >= 10'sd255);
    wire [OUT_W-1:0] out_comb_s0 =
        zero_s0      ? {OUT_W{1'b0}} :
        underflow_s0 ? {sign_s0, {(OUT_W-1){1'b0}}} :
        overflow_s0  ? {sign_s0, 8'hFE, {MANT_W{1'b1}}} :
                       {sign_s0, exp_s0[7:0], mant_s0};

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            out       <= 0;
        end else begin
            valid_out <= valid_s0;
            out       <= out_comb_s0;
        end
    end
endmodule
