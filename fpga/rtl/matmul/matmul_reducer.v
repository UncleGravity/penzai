// matmul reducer - pipelined for the multi-column ROWS=16 kernel.
//
// Produces contribution = (fp32) weight_scale * act_scale *
// Sigma_i (b_i ? +a_i : -a_i). Math is the same as the unpipelined reducer; the path is
// explicitly pipelined for higher fclk.
//
// Latency:    MATMUL_REDUCER_LATENCY (numeric/fmt.vh) = MATMUL_REDUCER_STAGES + 2*FP32_MUL_LATENCY.
// Throughput: 1 sub-block / cycle

`default_nettype none

module matmul_reducer (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,

    input  wire [31:0]  weight_bits,
    input  wire [255:0] acts_packed,
    input  wire [15:0]  weight_scale,
    input  wire [15:0]  act_scale,

    output wire         valid_out,
    output wire [31:0]  contribution
);
    `include "fmt.vh"

    function automatic signed [13:0] sext_act;
        input [7:0] a;
        sext_act = $signed({{6{a[7]}}, a});
    endfunction

    wire [31:0] weight_scale_f32;
    wire [31:0] act_scale_f32;
    fp16_to_fp32 u_ws (.in(weight_scale), .out(weight_scale_f32));
    fp16_to_fp32 u_as (.in(act_scale),    .out(act_scale_f32));

    reg         valid_s0;
    reg [31:0]  weight_bits_s0;
    reg [255:0] acts_packed_s0;
    reg [31:0]  weight_scale_f32_s0;
    reg [31:0]  act_scale_f32_s0;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s0            <= 1'b0;
            weight_bits_s0      <= 32'd0;
            acts_packed_s0      <= 256'd0;
            weight_scale_f32_s0 <= 32'd0;
            act_scale_f32_s0    <= 32'd0;
        end else begin
            valid_s0            <= valid_in;
            weight_bits_s0      <= weight_bits;
            acts_packed_s0      <= acts_packed;
            weight_scale_f32_s0 <= weight_scale_f32;
            act_scale_f32_s0    <= act_scale_f32;
        end
    end

    integer group;
    integer elem;
    integer bit_index;
    reg signed [13:0] partial_sum [0:7];
    always @(*) begin
        for (group = 0; group < 8; group = group + 1) begin
            partial_sum[group] = 14'sd0;
            for (elem = 0; elem < 4; elem = elem + 1) begin
                bit_index = group * 4 + elem;
                partial_sum[group] = partial_sum[group] + (weight_bits_s0[bit_index] ?
                     sext_act(acts_packed_s0[bit_index*8 +: 8]) :
                    -sext_act(acts_packed_s0[bit_index*8 +: 8]));
            end
        end
    end

    reg         valid_s1;
    reg [31:0]  weight_scale_f32_s1;
    reg [31:0]  act_scale_f32_s1;
    reg signed [13:0] partial_sum_s1 [0:7];

    integer ps;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s1            <= 1'b0;
            weight_scale_f32_s1 <= 32'd0;
            act_scale_f32_s1    <= 32'd0;
            for (ps = 0; ps < 8; ps = ps + 1) partial_sum_s1[ps] <= 14'sd0;
        end else begin
            valid_s1            <= valid_s0;
            weight_scale_f32_s1 <= weight_scale_f32_s0;
            act_scale_f32_s1    <= act_scale_f32_s0;
            for (ps = 0; ps < 8; ps = ps + 1) partial_sum_s1[ps] <= partial_sum[ps];
        end
    end

    reg         valid_s2;
    reg [31:0]  weight_scale_f32_s2;
    reg [31:0]  act_scale_f32_s2;
    reg signed [13:0] sum_l1_s2 [0:3];

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s2            <= 1'b0;
            weight_scale_f32_s2 <= 32'd0;
            act_scale_f32_s2    <= 32'd0;
            for (ps = 0; ps < 4; ps = ps + 1) sum_l1_s2[ps] <= 14'sd0;
        end else begin
            valid_s2            <= valid_s1;
            weight_scale_f32_s2 <= weight_scale_f32_s1;
            act_scale_f32_s2    <= act_scale_f32_s1;
            sum_l1_s2[0]        <= partial_sum_s1[0] + partial_sum_s1[1];
            sum_l1_s2[1]        <= partial_sum_s1[2] + partial_sum_s1[3];
            sum_l1_s2[2]        <= partial_sum_s1[4] + partial_sum_s1[5];
            sum_l1_s2[3]        <= partial_sum_s1[6] + partial_sum_s1[7];
        end
    end

    wire signed [13:0] sum_l2_0 = sum_l1_s2[0] + sum_l1_s2[1];
    wire signed [13:0] sum_l2_1 = sum_l1_s2[2] + sum_l1_s2[3];

    reg         valid_s3;
    reg [31:0]  weight_scale_f32_s3;
    reg [31:0]  act_scale_f32_s3;
    reg signed [13:0] sub_sum_s3;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s3            <= 1'b0;
            weight_scale_f32_s3 <= 32'd0;
            act_scale_f32_s3    <= 32'd0;
            sub_sum_s3          <= 14'sd0;
        end else begin
            valid_s3            <= valid_s2;
            weight_scale_f32_s3 <= weight_scale_f32_s2;
            act_scale_f32_s3    <= act_scale_f32_s2;
            sub_sum_s3          <= sum_l2_0 + sum_l2_1;
        end
    end

    wire [31:0] combined_f32;
    wire        combined_valid;
    fp32_mul_pipe u_combine (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_s3),
        .a(weight_scale_f32_s3),
        .b(act_scale_f32_s3),
        .valid_out(combined_valid),
        .out(combined_f32)
    );

    wire [31:0] sub_sum_f32;
    int_to_fp32 #(.WIDTH(14)) u_int (.in(sub_sum_s3), .out(sub_sum_f32));

    // Delay the int->fp32 sub-sum by FP32_MUL_LATENCY to meet combined_f32 at u_contrib.
    // Depth tracks the leaf latency in numeric/fmt.vh — no hand-counted pipe to ripple.
    reg [31:0] sub_sum_f32_dly [0:FP32_MUL_LATENCY-1];
    integer d;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (d = 0; d < FP32_MUL_LATENCY; d = d + 1) sub_sum_f32_dly[d] <= 32'd0;
        end else begin
            sub_sum_f32_dly[0] <= sub_sum_f32;
            for (d = 1; d < FP32_MUL_LATENCY; d = d + 1)
                sub_sum_f32_dly[d] <= sub_sum_f32_dly[d-1];
        end
    end

    fp32_mul_pipe u_contrib (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(combined_valid),
        .a(combined_f32),
        .b(sub_sum_f32_dly[FP32_MUL_LATENCY-1]),
        .valid_out(valid_out),
        .out(contribution)
    );
endmodule
