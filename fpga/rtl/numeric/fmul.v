// fmul - MANT_W-parameterized truncating float multiply.
//
// fp32 (MANT_W=23) and bf16 (MANT_W=7) are the SAME module, different param; the
// exponent stays 8-bit (fp32 range) for both. Truncating, same semantics as the
// rtl/fp leaf it replaces. The (MANT_W+1)² mantissa product is forced into a DSP
// (use_dsp) and runs register-to-register, as in fp32_mul_pipe (the f250 fix).
//
// At MANT_W=23 this is bit-identical to fp32_mul_pipe (differential cosim is the gate).
// Latency valid_in -> valid_out: 3 + MUL_PIPE. Format: { sign[1], exp[8], mant[MANT_W] }.
//
// MUL_PIPE adds extra register stage(s) on the mantissa product. At fp32 (MANT_W=23) the
// (24×24)=48-bit product exceeds one DSP48E2 (27×18), so Vivado infers a TWO-DSP cascade;
// at MUL_PIPE=0 both DSPs evaluate combinationally in one cycle (the f300 flash-dot wall:
// 2.9ns of multiply+ALU+cascade-ALU). MUL_PIPE=1 gives retiming a register to push into
// the cascade (one DSP per cycle), halving the path. Value is UNCHANGED (the extra stage
// is pure pipelining) — the cosim is the bit-exact gate. Only the wide-multiply consumers
// that wall f300 opt in (fp_dot); everything else stays MUL_PIPE=0 (latency 3, untouched).

`default_nettype none

module fmul #(
    parameter integer MANT_W   = 23,              // FMT_FP32_MANT; 7 for bf16
    parameter integer MUL_PIPE = 0                // extra mantissa-product pipeline stages
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

    // ---- product stage (DSP) + optional MUL_PIPE cascade-split registers ----
    // index 0 = raw mantissa product; indices 1..MUL_PIPE are extra pipeline registers
    // Vivado retimes into the multi-DSP cascade (fp32 → 2 DSPs). Metadata shifts alongside,
    // so the normalize below is unchanged. MUL_PIPE=0 ≡ the original single product reg.
    (* use_dsp = "yes" *) reg [PW-1:0] prod_sr  [0:MUL_PIPE];
    reg              valid_sr [0:MUL_PIPE];
    reg              zero_sr  [0:MUL_PIPE];
    reg              sign_sr  [0:MUL_PIPE];
    reg signed [9:0] exp_sr   [0:MUL_PIPE];
    integer p;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (p = 0; p <= MUL_PIPE; p = p + 1) begin
                prod_sr[p] <= 0; valid_sr[p] <= 1'b0; zero_sr[p] <= 1'b0;
                sign_sr[p] <= 1'b0; exp_sr[p] <= 10'sd0;
            end
        end else begin
            prod_sr[0]  <= mant_a_i * mant_b_i;
            valid_sr[0] <= valid_i;
            zero_sr[0]  <= zero_i;
            sign_sr[0]  <= sign_i;
            exp_sr[0]   <= exp_pre_i;
            for (p = 1; p <= MUL_PIPE; p = p + 1) begin
                prod_sr[p]  <= prod_sr[p-1];
                valid_sr[p] <= valid_sr[p-1];
                zero_sr[p]  <= zero_sr[p-1];
                sign_sr[p]  <= sign_sr[p-1];
                exp_sr[p]   <= exp_sr[p-1];
            end
        end
    end
    // tail of the multiply pipeline feeds the (unchanged) normalize.
    wire [PW-1:0]     prod_s0    = prod_sr[MUL_PIPE];
    wire              valid_s0   = valid_sr[MUL_PIPE];
    wire              zero_s0    = zero_sr[MUL_PIPE];
    wire              sign_s0    = sign_sr[MUL_PIPE];
    wire signed [9:0] exp_pre_s0 = exp_sr[MUL_PIPE];

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
