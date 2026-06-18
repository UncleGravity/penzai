// fp32_add_pipe - pipelined truncating IEEE 754 fp32 addition.
//
// Same semantics as fp32_add.v. Latency from input valid observed by this module
// to valid_out is four cycles; callers that register valid before this module
// generally need ADD_LAT=5 for their metadata pipe.

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
    // Stage 0: decode and select larger-exponent operand.
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] ma = a[22:0];
    wire        sb = b[31];
    wire [7:0]  eb = b[30:23];
    wire [22:0] mb = b[22:0];
    wire        a_ge_b = (ea >= eb);

    reg        valid_s0;
    reg [31:0] a_s0;
    reg [31:0] b_s0;
    reg        a_zero_s0;
    reg        b_zero_s0;
    reg [7:0]  exp_big_s0;
    reg [7:0]  exp_diff_s0;
    reg [23:0] mant_big_s0;
    reg [23:0] mant_small_s0;
    reg        sign_big_s0;
    reg        sign_small_s0;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0      <= 1'b0;
            a_s0          <= 32'd0;
            b_s0          <= 32'd0;
            a_zero_s0     <= 1'b0;
            b_zero_s0     <= 1'b0;
            exp_big_s0    <= 8'd0;
            exp_diff_s0   <= 8'd0;
            mant_big_s0   <= 24'd0;
            mant_small_s0 <= 24'd0;
            sign_big_s0   <= 1'b0;
            sign_small_s0 <= 1'b0;
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

    // Stage 1: align the smaller mantissa.
    reg        valid_s1;
    reg [31:0] a_s1;
    reg [31:0] b_s1;
    reg        a_zero_s1;
    reg        b_zero_s1;
    reg [7:0]  exp_big_s1;
    reg [23:0] mant_big_s1;
    reg [23:0] mant_small_aligned_s1;
    reg        sign_big_s1;
    reg        sign_small_s1;

    wire [23:0] mant_small_aligned_s0 = (exp_diff_s0 > 8'd24)
        ? 24'd0
        : (mant_small_s0 >> exp_diff_s0[4:0]);

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s1               <= 1'b0;
            a_s1                   <= 32'd0;
            b_s1                   <= 32'd0;
            a_zero_s1              <= 1'b0;
            b_zero_s1              <= 1'b0;
            exp_big_s1             <= 8'd0;
            mant_big_s1            <= 24'd0;
            mant_small_aligned_s1  <= 24'd0;
            sign_big_s1            <= 1'b0;
            sign_small_s1          <= 1'b0;
        end else begin
            valid_s1               <= valid_s0;
            a_s1                   <= a_s0;
            b_s1                   <= b_s0;
            a_zero_s1              <= a_zero_s0;
            b_zero_s1              <= b_zero_s0;
            exp_big_s1             <= exp_big_s0;
            mant_big_s1            <= mant_big_s0;
            mant_small_aligned_s1  <= mant_small_aligned_s0;
            sign_big_s1            <= sign_big_s0;
            sign_small_s1          <= sign_small_s0;
        end
    end

    // Stage 2: resolve magnitude tie and add/subtract mantissas.
    wire small_mant_bigger_s1 = (mant_small_aligned_s1 > mant_big_s1);
    wire [23:0] m1_s1 = small_mant_bigger_s1 ? mant_small_aligned_s1 : mant_big_s1;
    wire [23:0] m2_s1 = small_mant_bigger_s1 ? mant_big_s1 : mant_small_aligned_s1;
    wire result_sign_s1 = small_mant_bigger_s1 ? sign_small_s1 : sign_big_s1;
    wire same_sign_s1 = (sign_big_s1 == sign_small_s1);
    wire [24:0] mant_sum_s1 = same_sign_s1
        ? ({1'b0, m1_s1} + {1'b0, m2_s1})
        : ({1'b0, m1_s1} - {1'b0, m2_s1});

    reg        valid_s2;
    reg [31:0] a_s2;
    reg [31:0] b_s2;
    reg        a_zero_s2;
    reg        b_zero_s2;
    reg        result_sign_s2;
    reg [7:0]  exp_big_s2;
    reg [24:0] mant_sum_s2;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s2       <= 1'b0;
            a_s2           <= 32'd0;
            b_s2           <= 32'd0;
            a_zero_s2      <= 1'b0;
            b_zero_s2      <= 1'b0;
            result_sign_s2 <= 1'b0;
            exp_big_s2     <= 8'd0;
            mant_sum_s2    <= 25'd0;
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

    // Stage 3: priority encode the mantissa.
    reg [4:0] lead_pos_s2;
    integer ii;
    always @(*) begin
        lead_pos_s2 = 5'd0;
        for (ii = 0; ii <= 24; ii = ii + 1) begin
            if (mant_sum_s2[ii]) lead_pos_s2 = ii[4:0];
        end
    end

    reg        valid_s3;
    reg [31:0] a_s3;
    reg [31:0] b_s3;
    reg        a_zero_s3;
    reg        b_zero_s3;
    reg        result_sign_s3;
    reg [7:0]  exp_big_s3;
    reg [24:0] mant_sum_s3;
    reg [4:0]  lead_pos_s3;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s3       <= 1'b0;
            a_s3           <= 32'd0;
            b_s3           <= 32'd0;
            a_zero_s3      <= 1'b0;
            b_zero_s3      <= 1'b0;
            result_sign_s3 <= 1'b0;
            exp_big_s3     <= 8'd0;
            mant_sum_s3    <= 25'd0;
            lead_pos_s3    <= 5'd0;
        end else begin
            valid_s3       <= valid_s2;
            a_s3           <= a_s2;
            b_s3           <= b_s2;
            a_zero_s3      <= a_zero_s2;
            b_zero_s3      <= b_zero_s2;
            result_sign_s3 <= result_sign_s2;
            exp_big_s3     <= exp_big_s2;
            mant_sum_s3    <= mant_sum_s2;
            lead_pos_s3    <= lead_pos_s2;
        end
    end

    // Stage 4: normalize and assemble.
    wire sum_zero_s3 = (mant_sum_s3 == 25'd0);
    wire shift_right_s3 = (lead_pos_s3 > 5'd23);
    wire [4:0] right_amount_s3 = shift_right_s3 ? (lead_pos_s3 - 5'd23) : 5'd0;
    wire [4:0] left_amount_s3  = (lead_pos_s3 < 5'd23) ? (5'd23 - lead_pos_s3) : 5'd0;
    wire [24:0] mant_normalized_s3 = shift_right_s3
        ? (mant_sum_s3 >> right_amount_s3)
        : (mant_sum_s3 << left_amount_s3);
    wire signed [9:0] exp_signed_s3 = shift_right_s3
        ? ($signed({2'b00, exp_big_s3}) + $signed({5'd0, right_amount_s3}))
        : ($signed({2'b00, exp_big_s3}) - $signed({5'd0, left_amount_s3}));
    wire underflow_s3 = sum_zero_s3 || (exp_signed_s3 <= 10'sd0);
    wire overflow_s3  = (exp_signed_s3 >= 10'sd255);
    wire [31:0] out_comb_s3 = a_zero_s3 ? b_s3
        : b_zero_s3 ? a_s3
        : underflow_s3 ? {result_sign_s3, 31'd0}
        : overflow_s3  ? {result_sign_s3, 8'hFE, 23'h7FFFFF}
                       : {result_sign_s3, exp_signed_s3[7:0], mant_normalized_s3[22:0]};

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            out       <= 32'd0;
        end else begin
            valid_out <= valid_s3;
            out       <= out_comb_s3;
        end
    end
endmodule
