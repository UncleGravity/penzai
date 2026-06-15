// fp32_add_pipe - one-register-cut version of fp32_add.
//
// Same truncating fp32 semantics as fp32_add.v, but split across two cycles so
// replicated ROWS=16 rowblocks do not place the whole align/add/normalize path
// between one source register and one destination register.

`default_nettype none

module fp32_add_pipe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         valid_out,
    output reg  [31:0] out
);
    // -- Decode / align / magnitude add ----------------------------------
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] ma = a[22:0];
    wire        sb = b[31];
    wire [7:0]  eb = b[30:23];
    wire [22:0] mb = b[22:0];

    wire a_zero = (ea == 8'd0);
    wire b_zero = (eb == 8'd0);

    wire        a_ge_b     = (ea >= eb);
    wire [7:0]  exp_big    = a_ge_b ? ea : eb;
    wire [7:0]  exp_small  = a_ge_b ? eb : ea;
    wire [23:0] mant_big   = a_ge_b ? {1'b1, ma} : {1'b1, mb};
    wire [23:0] mant_small = a_ge_b ? {1'b1, mb} : {1'b1, ma};
    wire        sign_big   = a_ge_b ? sa : sb;
    wire        sign_small = a_ge_b ? sb : sa;

    wire [7:0]  exp_diff = exp_big - exp_small;
    wire [23:0] mant_small_aligned = (exp_diff > 8'd24)
                                       ? 24'd0
                                       : (mant_small >> exp_diff[4:0]);

    wire same_exp          = (exp_diff == 8'd0);
    wire small_mant_bigger = same_exp && (mant_small_aligned > mant_big);

    wire [23:0] m1 = small_mant_bigger ? mant_small_aligned : mant_big;
    wire [23:0] m2 = small_mant_bigger ? mant_big : mant_small_aligned;
    wire result_sign = small_mant_bigger ? sign_small : sign_big;

    wire same_sign = (sign_big == sign_small);
    wire [24:0] mant_sum = same_sign
        ? ({1'b0, m1} + {1'b0, m2})
        : ({1'b0, m1} - {1'b0, m2});

    // Stage 0: cut after exponent alignment and magnitude add/subtract.
    reg        valid_s0;
    reg [31:0] a_s0;
    reg [31:0] b_s0;
    reg        a_zero_s0;
    reg        b_zero_s0;
    reg        result_sign_s0;
    reg [7:0]  exp_big_s0;
    reg [24:0] mant_sum_s0;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0       <= 1'b0;
            a_s0           <= 32'd0;
            b_s0           <= 32'd0;
            a_zero_s0      <= 1'b0;
            b_zero_s0      <= 1'b0;
            result_sign_s0 <= 1'b0;
            exp_big_s0     <= 8'd0;
            mant_sum_s0    <= 25'd0;
        end else begin
            valid_s0       <= valid_in;
            a_s0           <= a;
            b_s0           <= b;
            a_zero_s0      <= a_zero;
            b_zero_s0      <= b_zero;
            result_sign_s0 <= result_sign;
            exp_big_s0     <= exp_big;
            mant_sum_s0    <= mant_sum;
        end
    end

    // -- Normalize / assemble --------------------------------------------
    wire sum_zero = (mant_sum_s0 == 25'd0);

    reg [4:0] lead_pos;
    integer ii;
    always @(*) begin
        lead_pos = 5'd0;
        for (ii = 0; ii <= 24; ii = ii + 1) begin
            if (mant_sum_s0[ii]) lead_pos = ii[4:0];
        end
    end

    wire       shift_right  = (lead_pos > 5'd23);
    wire [4:0] right_amount = shift_right ? (lead_pos - 5'd23) : 5'd0;
    wire [4:0] left_amount  = (lead_pos < 5'd23) ? (5'd23 - lead_pos) : 5'd0;

    wire [24:0] mant_normalized = shift_right
        ? (mant_sum_s0 >> right_amount)
        : (mant_sum_s0 << left_amount);

    wire signed [9:0] exp_signed = shift_right
        ? ($signed({2'b00, exp_big_s0}) + $signed({5'd0, right_amount}))
        : ($signed({2'b00, exp_big_s0}) - $signed({5'd0, left_amount}));

    wire underflow = sum_zero || (exp_signed <= 10'sd0);
    wire overflow  = (exp_signed >= 10'sd255);

    wire [31:0] out_comb = a_zero_s0 ? b_s0
                         : b_zero_s0 ? a_s0
                         : underflow ? {result_sign_s0, 31'd0}
                         : overflow  ? {result_sign_s0, 8'hFE, 23'h7FFFFF}
                                     : {result_sign_s0, exp_signed[7:0], mant_normalized[22:0]};

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            out       <= 32'd0;
        end else begin
            valid_out <= valid_s0;
            out       <= out_comb;
        end
    end

endmodule
