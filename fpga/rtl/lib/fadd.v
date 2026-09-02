// fadd - MANT_W-parameterized truncating IEEE-style float addition.
//
// fp32 (MANT_W=23) and bf16 (MANT_W=7) use the same module with different
// parameters. The exponent stays 8-bit for both formats. Arithmetic truncates
// instead of rounding.
//
// Format on the wire:
//   { sign[1], exp[8], mant[MANT_W] }   width = MANT_W + 9.

`default_nettype none

module fadd #(
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
    localparam integer OUT_W = MANT_W + 9;        // sign + 8 exp + mantissa
    localparam integer SIG_W = MANT_W + 1;        // significand incl. hidden 1
    localparam integer SUM_W = MANT_W + 2;        // sum incl. carry
    localparam integer LPW   = $clog2(SUM_W);     // lead-pos / shift-amount width
    localparam [7:0]        ALIGN_MAX = SIG_W[7:0]; // align past this -> 0
    localparam [LPW-1:0]    MANT_W_LP = MANT_W[LPW-1:0];

    // ---- decode ----
    wire              sa = a[OUT_W-1];
    wire [7:0]        ea = a[OUT_W-2 -: 8];
    wire [MANT_W-1:0] ma = a[MANT_W-1:0];
    wire              sb = b[OUT_W-1];
    wire [7:0]        eb = b[OUT_W-2 -: 8];
    wire [MANT_W-1:0] mb = b[MANT_W-1:0];
    wire              a_ge_b = (ea >= eb);

    // ---- stage 0: decode + select larger-exponent operand ----
    reg              valid_s0;
    reg [OUT_W-1:0]  a_s0, b_s0;
    reg              a_zero_s0, b_zero_s0;
    reg [7:0]        exp_big_s0, exp_diff_s0;
    reg [SIG_W-1:0]  mant_big_s0, mant_small_s0;
    reg              sign_big_s0, sign_small_s0;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0 <= 1'b0; a_s0 <= 0; b_s0 <= 0; a_zero_s0 <= 1'b0; b_zero_s0 <= 1'b0;
            exp_big_s0 <= 0; exp_diff_s0 <= 0; mant_big_s0 <= 0; mant_small_s0 <= 0;
            sign_big_s0 <= 1'b0; sign_small_s0 <= 1'b0;
        end else begin
            valid_s0      <= valid_in;
            a_s0          <= a;
            b_s0          <= b;
            a_zero_s0     <= (ea == 8'd0);
            b_zero_s0     <= (eb == 8'd0);
            exp_big_s0    <= a_ge_b ? ea : eb;
            exp_diff_s0   <= a_ge_b ? (ea - eb) : (eb - ea);
            mant_big_s0   <= a_ge_b ? {1'b1, ma} : {1'b1, mb};
            mant_small_s0 <= a_ge_b ? {1'b1, mb} : {1'b1, ma};
            sign_big_s0   <= a_ge_b ? sa : sb;
            sign_small_s0 <= a_ge_b ? sb : sa;
        end
    end

    // ---- stage 1: align the smaller mantissa ----
    wire [SIG_W-1:0] mant_small_aligned_s0 = (exp_diff_s0 > ALIGN_MAX)
        ? {SIG_W{1'b0}}
        : (mant_small_s0 >> exp_diff_s0[LPW-1:0]);
    reg              valid_s1;
    reg [OUT_W-1:0]  a_s1, b_s1;
    reg              a_zero_s1, b_zero_s1;
    reg [7:0]        exp_big_s1;
    reg [SIG_W-1:0]  mant_big_s1, mant_small_aligned_s1;
    reg              sign_big_s1, sign_small_s1;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s1 <= 1'b0; a_s1 <= 0; b_s1 <= 0; a_zero_s1 <= 1'b0; b_zero_s1 <= 1'b0;
            exp_big_s1 <= 0; mant_big_s1 <= 0; mant_small_aligned_s1 <= 0;
            sign_big_s1 <= 1'b0; sign_small_s1 <= 1'b0;
        end else begin
            valid_s1              <= valid_s0;
            a_s1                  <= a_s0;
            b_s1                  <= b_s0;
            a_zero_s1             <= a_zero_s0;
            b_zero_s1             <= b_zero_s0;
            exp_big_s1            <= exp_big_s0;
            mant_big_s1           <= mant_big_s0;
            mant_small_aligned_s1 <= mant_small_aligned_s0;
            sign_big_s1           <= sign_big_s0;
            sign_small_s1         <= sign_small_s0;
        end
    end

    // ---- stage 2: resolve magnitude tie, add/subtract ----
    wire             small_bigger_s1 = (mant_small_aligned_s1 > mant_big_s1);
    wire [SIG_W-1:0] m1_s1 = small_bigger_s1 ? mant_small_aligned_s1 : mant_big_s1;
    wire [SIG_W-1:0] m2_s1 = small_bigger_s1 ? mant_big_s1 : mant_small_aligned_s1;
    wire             result_sign_s1 = small_bigger_s1 ? sign_small_s1 : sign_big_s1;
    wire             same_sign_s1 = (sign_big_s1 == sign_small_s1);
    wire [SUM_W-1:0] mant_sum_s1 = same_sign_s1
        ? ({1'b0, m1_s1} + {1'b0, m2_s1})
        : ({1'b0, m1_s1} - {1'b0, m2_s1});
    reg              valid_s2;
    reg [OUT_W-1:0]  a_s2, b_s2;
    reg              a_zero_s2, b_zero_s2;
    reg              result_sign_s2;
    reg [7:0]        exp_big_s2;
    reg [SUM_W-1:0]  mant_sum_s2;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s2 <= 1'b0; a_s2 <= 0; b_s2 <= 0; a_zero_s2 <= 1'b0; b_zero_s2 <= 1'b0;
            result_sign_s2 <= 1'b0; exp_big_s2 <= 0; mant_sum_s2 <= 0;
        end else begin
            valid_s2       <= valid_s1;
            a_s2           <= a_s1;
            b_s2           <= b_s1;
            a_zero_s2      <= a_zero_s1;
            b_zero_s2      <= b_zero_s1;
            result_sign_s2 <= result_sign_s1;
            exp_big_s2     <= exp_big_s1;
            mant_sum_s2    <= mant_sum_s1;
        end
    end

    // ---- stage 3: leading-one detect (index of the highest set bit of mant_sum) ----
    // Tree-structured to avoid a linear, SUM_W-deep priority-mux path while computing
    // the identical value:
    //   parallel-prefix OR fills every bit at/below the MSB; one AND isolates the MSB as a
    //   one-hot; an OR-tree encodes that one-hot to its index (all-zero in -> 0 out, matching
    //   the scan). The
    //   five fixed shifts cover any SUM_W <= 32 (here SUM_W = MANT_W + 2 <= 25).
    function automatic [LPW-1:0] lead_one(input [SUM_W-1:0] v);
        reg [SUM_W-1:0] f, oh;
        integer j;
        begin
            f  = v | (v >> 1);
            f  = f | (f >> 2);
            f  = f | (f >> 4);
            f  = f | (f >> 8);
            f  = f | (f >> 16);
            oh = f & ~(f >> 1);                  // highest set bit, one-hot
            lead_one = {LPW{1'b0}};
            for (j = 0; j < SUM_W; j = j + 1)
                if (oh[j]) lead_one = lead_one | j[LPW-1:0];  // one-hot -> index (OR-tree)
        end
    endfunction
    wire [LPW-1:0] lead_pos_s2 = lead_one(mant_sum_s2);

    // Normalize controls are precomputed in stage 3 to shorten the lead_pos-to-output
    // path. They are derived from the stage-2 values, identical to
    // computing them in stage 4 from the registered copies (exp_big_s3==exp_big_s2 etc.), so
    // bit-identical; the differential cosim is the gate.
    wire sum_zero_s2    = (mant_sum_s2 == {SUM_W{1'b0}});
    wire shift_right_s2 = (lead_pos_s2 > MANT_W_LP);
    wire [LPW-1:0] right_amount_s2 = shift_right_s2 ? (lead_pos_s2 - MANT_W_LP) : {LPW{1'b0}};
    wire [LPW-1:0] left_amount_s2  = (lead_pos_s2 < MANT_W_LP) ? (MANT_W_LP - lead_pos_s2) : {LPW{1'b0}};
    wire signed [9:0] exp_signed_s2 = shift_right_s2
        ? ($signed({2'b00, exp_big_s2}) + $signed({{(10-LPW){1'b0}}, right_amount_s2}))
        : ($signed({2'b00, exp_big_s2}) - $signed({{(10-LPW){1'b0}}, left_amount_s2}));
    wire underflow_s2 = sum_zero_s2 || (exp_signed_s2 <= 10'sd0);
    wire overflow_s2  = (exp_signed_s2 >= 10'sd255);

    reg              valid_s3;
    reg [OUT_W-1:0]  a_s3, b_s3;
    reg              a_zero_s3, b_zero_s3;
    reg              result_sign_s3;
    reg [SUM_W-1:0]  mant_sum_s3;
    reg              shift_right_s3;
    reg [LPW-1:0]    right_amount_s3, left_amount_s3;
    reg [7:0]        exp_out_s3;
    reg              underflow_s3, overflow_s3;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s3 <= 1'b0; a_s3 <= 0; b_s3 <= 0; a_zero_s3 <= 1'b0; b_zero_s3 <= 1'b0;
            result_sign_s3 <= 1'b0; mant_sum_s3 <= 0; shift_right_s3 <= 1'b0;
            right_amount_s3 <= 0; left_amount_s3 <= 0; exp_out_s3 <= 0;
            underflow_s3 <= 1'b0; overflow_s3 <= 1'b0;
        end else begin
            valid_s3        <= valid_s2;
            a_s3            <= a_s2;
            b_s3            <= b_s2;
            a_zero_s3       <= a_zero_s2;
            b_zero_s3       <= b_zero_s2;
            result_sign_s3  <= result_sign_s2;
            mant_sum_s3     <= mant_sum_s2;
            shift_right_s3  <= shift_right_s2;
            right_amount_s3 <= right_amount_s2;
            left_amount_s3  <= left_amount_s2;
            exp_out_s3      <= exp_signed_s2[7:0];
            underflow_s3    <= underflow_s2;
            overflow_s3     <= overflow_s2;
        end
    end

    // ---- stage 4: barrel-shift the sum into place + assemble (controls precomputed) ----
    wire [SUM_W-1:0] mant_norm_s3 = shift_right_s3
        ? (mant_sum_s3 >> right_amount_s3)
        : (mant_sum_s3 << left_amount_s3);
    wire [OUT_W-1:0] out_comb_s3 = a_zero_s3 ? b_s3
        : b_zero_s3 ? a_s3
        : underflow_s3 ? {result_sign_s3, {(OUT_W-1){1'b0}}}
        : overflow_s3  ? {result_sign_s3, 8'hFE, {MANT_W{1'b1}}}
                       : {result_sign_s3, exp_out_s3, mant_norm_s3[MANT_W-1:0]};

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            out       <= 0;
        end else begin
            valid_out <= valid_s3;
            out       <= out_comb_s3;
        end
    end
endmodule
