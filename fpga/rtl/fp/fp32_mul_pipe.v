// fp32_mul_pipe - pipelined version of fp32_mul.
//
// Same truncating/saturating semantics as fp32_mul.v. Latency from valid_in to
// valid_out is two cycles.

`default_nettype none

module fp32_mul_pipe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         valid_out,
    output reg  [31:0] out
);
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] ma = a[22:0];
    wire        sb = b[31];
    wire [7:0]  eb = b[30:23];
    wire [22:0] mb = b[22:0];
    wire        is_zero = (ea == 8'd0) || (eb == 8'd0);
    wire        sr = sa ^ sb;
    wire signed [9:0] er_pre = $signed({2'b00, ea})
                             + $signed({2'b00, eb})
                             - 10'sd127;

    reg        valid_s0;
    reg        zero_s0;
    reg        sign_s0;
    reg signed [9:0] exp_pre_s0;
    (* use_dsp = "yes" *) reg [47:0] prod_s0;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0   <= 1'b0;
            zero_s0    <= 1'b0;
            sign_s0    <= 1'b0;
            exp_pre_s0 <= 10'sd0;
            prod_s0    <= 48'd0;
        end else begin
            valid_s0   <= valid_in;
            zero_s0    <= is_zero;
            sign_s0    <= sr;
            exp_pre_s0 <= er_pre;
            prod_s0    <= {1'b1, ma} * {1'b1, mb};
        end
    end

    wire renorm_s0 = prod_s0[47];
    wire [22:0] mant_s0 = renorm_s0 ? prod_s0[46:24] : prod_s0[45:23];
    wire signed [9:0] exp_s0 = exp_pre_s0 + (renorm_s0 ? 10'sd1 : 10'sd0);
    wire underflow_s0 = (exp_s0 <= 10'sd0);
    wire overflow_s0  = (exp_s0 >= 10'sd255);
    wire [31:0] out_comb_s0 =
        zero_s0      ? 32'd0 :
        underflow_s0 ? {sign_s0, 31'd0} :
        overflow_s0  ? {sign_s0, 8'hFE, 23'h7FFFFF} :
                       {sign_s0, exp_s0[7:0], mant_s0};

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            out       <= 32'd0;
        end else begin
            valid_out <= valid_s0;
            out       <= out_comb_s0;
        end
    end
endmodule
