// Serialized exact finite binary32 adder for the section residual boundary.
//
// The datapath keeps the 24-bit significand plus guard, round, and sticky bits.
// Alignment and normalization share one five-cycle 16/8/4/2/1 shift-jam path.
// Every finite operand, including subnormals and signed zero, is added with IEEE
// round-to-nearest-even semantics. Non-finite operands are outside the section
// contract and return deterministic +0 with status. A finite overflow returns
// signed infinity with status. Raw operands are captured before decode; an
// accepted request reaches RESULT after 15 clocks. One request may be
// outstanding; the result holds under backpressure and abort discards every
// stage.

`default_nettype none

module section_residual_add_rne (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        abort_run,

    output wire        busy,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire [31:0] s_a,
    input  wire [31:0] s_b,

    output wire        result_valid,
    input  wire        result_ready,
    output wire [31:0] result_data,
    // bit 0 NONFINITE_INPUT, bit 1 OVERFLOW.
    output wire [1:0]  result_status
);
    localparam [1:0] STATUS_NONFINITE = 2'b01;
    localparam [1:0] STATUS_OVERFLOW  = 2'b10;

    localparam [3:0] ST_IDLE    = 4'd0;
    localparam [3:0] ST_CAPTURE = 4'd1;
    localparam [3:0] ST_ALIGN16 = 4'd2;
    localparam [3:0] ST_ALIGN8  = 4'd3;
    localparam [3:0] ST_ALIGN4  = 4'd4;
    localparam [3:0] ST_ALIGN2  = 4'd5;
    localparam [3:0] ST_ALIGN1  = 4'd6;
    localparam [3:0] ST_ADD     = 4'd7;
    localparam [3:0] ST_NORM16  = 4'd8;
    localparam [3:0] ST_NORM8   = 4'd9;
    localparam [3:0] ST_NORM4   = 4'd10;
    localparam [3:0] ST_NORM2   = 4'd11;
    localparam [3:0] ST_NORM1   = 4'd12;
    localparam [3:0] ST_ROUND   = 4'd13;
    localparam [3:0] ST_FINAL   = 4'd14;
    localparam [3:0] ST_RESULT  = 4'd15;

    reg [3:0] state_q;

    reg [31:0] operand_a_q;
    reg [31:0] operand_b_q;

    reg        sign_q;
    reg        same_sign_q;
    reg        nonfinite_q;
    reg        special_q;
    reg [31:0] special_data_q;

    reg [8:0]  exponent_q;
    reg [4:0]  align_distance_q;
    reg [26:0] big_ext_q;
    reg [26:0] small_ext_q;
    reg [27:0] magnitude_q;

    reg [24:0] rounded_q;

    reg [31:0] result_data_q;
    reg [1:0]  result_status_q;

    // Capture the raw operands before any decode, ordering, or exponent
    // distance logic. This boundary keeps the caller's control decode out of
    // the adder's magnitude-ordering path.
    wire [7:0] exp_a = operand_a_q[30:23];
    wire [7:0] exp_b = operand_b_q[30:23];
    wire [22:0] frac_a = operand_a_q[22:0];
    wire [22:0] frac_b = operand_b_q[22:0];
    wire a_nonfinite = exp_a == 8'hff;
    wire b_nonfinite = exp_b == 8'hff;
    wire a_zero = (exp_a == 8'd0) && (frac_a == 23'd0);
    wire b_zero = (exp_b == 8'd0) && (frac_b == 23'd0);
    wire [7:0] effective_a = exp_a == 8'd0 ? 8'd1 : exp_a;
    wire [7:0] effective_b = exp_b == 8'd0 ? 8'd1 : exp_b;
    wire [23:0] significand_a = {exp_a != 8'd0, frac_a};
    wire [23:0] significand_b = {exp_b != 8'd0, frac_b};
    wire a_mag_ge_b = (effective_a > effective_b) ||
                      ((effective_a == effective_b) &&
                       (significand_a >= significand_b));
    wire [7:0] selected_exp_big = a_mag_ge_b ? effective_a : effective_b;
    wire [7:0] selected_exp_small = a_mag_ge_b ? effective_b : effective_a;
    wire [7:0] selected_exp_diff = selected_exp_big - selected_exp_small;
    wire [23:0] selected_sig_big = a_mag_ge_b ?
                                   significand_a : significand_b;
    wire [23:0] selected_sig_small = a_mag_ge_b ?
                                     significand_b : significand_a;

    assign busy = state_q != ST_IDLE;
    assign s_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_data = result_data_q;
    assign result_status = result_status_q;

    wire [23:0] round_main = magnitude_q[26:3];
    wire round_increment = magnitude_q[2] &&
                           (magnitude_q[1] || magnitude_q[0] ||
                            magnitude_q[3]);

    wire exponent_gt16 = (|exponent_q[8:5]) ||
                         (exponent_q[4] && (|exponent_q[3:0]));
    wire exponent_gt8 = (|exponent_q[8:4]) ||
                        (exponent_q[3] && (|exponent_q[2:0]));
    wire exponent_gt4 = (|exponent_q[8:3]) ||
                        (exponent_q[2] && (|exponent_q[1:0]));
    wire exponent_gt2 = (|exponent_q[8:2]) ||
                        (exponent_q[1] && exponent_q[0]);
    wire exponent_gt1 = |exponent_q[8:1];

    wire norm_left16 = !magnitude_q[27] &&
                       (magnitude_q[26:11] == 16'd0) && exponent_gt16;
    wire norm_left8 = (magnitude_q[26:19] == 8'd0) && exponent_gt8;
    wire norm_left4 = (magnitude_q[26:23] == 4'd0) && exponent_gt4;
    wire norm_left2 = (magnitude_q[26:25] == 2'd0) && exponent_gt2;
    wire norm_left1 = !magnitude_q[26] && exponent_gt1;

    // All arithmetic states share one 28-bit carry chain. Subtraction uses the
    // conventional xor-plus-carry form so ADD does not infer parallel add and
    // subtract units. Exponent updates, rounding, and final exponent carry use
    // the same chain in later cycles.
    reg [27:0] alu_a;
    reg [27:0] alu_b;
    reg        alu_cin;

    always @* begin
        alu_a = 28'd0;
        alu_b = 28'd0;
        alu_cin = 1'b0;
        case (state_q)
            ST_ADD: begin
                alu_a = {1'b0, big_ext_q};
                alu_b = {1'b0, small_ext_q} ^
                        {28{!same_sign_q}};
                alu_cin = !same_sign_q;
            end
            ST_NORM16: begin
                alu_a = {19'd0, exponent_q};
                if (magnitude_q[27]) begin
                    alu_cin = 1'b1;
                end else begin
                    alu_b = ~{19'd0, 9'd16};
                    alu_cin = 1'b1;
                end
            end
            ST_NORM8: begin
                alu_a = {19'd0, exponent_q};
                alu_b = ~{19'd0, 9'd8};
                alu_cin = 1'b1;
            end
            ST_NORM4: begin
                alu_a = {19'd0, exponent_q};
                alu_b = ~{19'd0, 9'd4};
                alu_cin = 1'b1;
            end
            ST_NORM2: begin
                alu_a = {19'd0, exponent_q};
                alu_b = ~{19'd0, 9'd2};
                alu_cin = 1'b1;
            end
            ST_NORM1: begin
                alu_a = {19'd0, exponent_q};
                alu_b = ~{19'd0, 9'd1};
                alu_cin = 1'b1;
            end
            ST_ROUND: begin
                alu_a = {4'd0, round_main};
                alu_cin = round_increment;
            end
            ST_FINAL: begin
                alu_a = {19'd0, exponent_q};
                alu_cin = rounded_q[24];
            end
            default: begin
                alu_a = 28'd0;
                alu_b = 28'd0;
                alu_cin = 1'b0;
            end
        endcase
    end

    wire [27:0] alu_sum = alu_a + alu_b + alu_cin;
    wire [8:0] final_exp = alu_sum[8:0];
    wire [23:0] final_sig = rounded_q[24] ?
                            rounded_q[24:1] : rounded_q[23:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            result_data_q <= 32'd0;
            result_status_q <= 2'd0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            result_data_q <= 32'd0;
            result_status_q <= 2'd0;
        end else begin
            case (state_q)
                ST_IDLE: if (s_valid) begin
                    operand_a_q <= s_a;
                    operand_b_q <= s_b;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                    state_q <= ST_CAPTURE;
                end

                ST_CAPTURE: begin
                    sign_q <= a_mag_ge_b ? operand_a_q[31] :
                                              operand_b_q[31];
                    same_sign_q <= operand_a_q[31] == operand_b_q[31];
                    nonfinite_q <= a_nonfinite || b_nonfinite;
                    special_q <= a_zero || b_zero;
                    special_data_q <= (a_zero && b_zero) ?
                        {operand_a_q[31] & operand_b_q[31], 31'd0} :
                        (a_zero ? operand_b_q : operand_a_q);
                    exponent_q <= {1'b0, selected_exp_big};
                    align_distance_q <= ((|selected_exp_diff[7:5]) ||
                        (selected_exp_diff[4] && selected_exp_diff[3] &&
                         (selected_exp_diff[2] ||
                          (selected_exp_diff[1] && selected_exp_diff[0])))) ?
                                        5'd27 : selected_exp_diff[4:0];
                    big_ext_q <= {selected_sig_big, 3'b000};
                    small_ext_q <= {selected_sig_small, 3'b000};
                    state_q <= ST_ALIGN16;
                end

                ST_ALIGN16: begin
                    if (align_distance_q[4])
                        small_ext_q <= {
                            16'd0, small_ext_q[26:17],
                            (|small_ext_q[16:0])
                        };
                    state_q <= ST_ALIGN8;
                end
                ST_ALIGN8: begin
                    if (align_distance_q[3])
                        small_ext_q <= {
                            8'd0, small_ext_q[26:9],
                            (|small_ext_q[8:0])
                        };
                    state_q <= ST_ALIGN4;
                end
                ST_ALIGN4: begin
                    if (align_distance_q[2])
                        small_ext_q <= {
                            4'd0, small_ext_q[26:5],
                            (|small_ext_q[4:0])
                        };
                    state_q <= ST_ALIGN2;
                end
                ST_ALIGN2: begin
                    if (align_distance_q[1])
                        small_ext_q <= {
                            2'd0, small_ext_q[26:3],
                            (|small_ext_q[2:0])
                        };
                    state_q <= ST_ALIGN1;
                end
                ST_ALIGN1: begin
                    if (align_distance_q[0])
                        small_ext_q <= {
                            1'b0, small_ext_q[26:2],
                            (|small_ext_q[1:0])
                        };
                    state_q <= ST_ADD;
                end

                ST_ADD: begin
                    magnitude_q <= alu_sum;
                    state_q <= ST_NORM16;
                end

                ST_NORM16: begin
                    if (magnitude_q[27]) begin
                        magnitude_q <= {
                            1'b0, magnitude_q[27:2],
                            (|magnitude_q[1:0])
                        };
                        exponent_q <= alu_sum[8:0];
                    end else if (norm_left16) begin
                        magnitude_q <= {magnitude_q[11:0], 16'd0};
                        exponent_q <= alu_sum[8:0];
                    end
                    state_q <= ST_NORM8;
                end
                ST_NORM8: begin
                    if (norm_left8) begin
                        magnitude_q <= {magnitude_q[19:0], 8'd0};
                        exponent_q <= alu_sum[8:0];
                    end
                    state_q <= ST_NORM4;
                end
                ST_NORM4: begin
                    if (norm_left4) begin
                        magnitude_q <= {magnitude_q[23:0], 4'd0};
                        exponent_q <= alu_sum[8:0];
                    end
                    state_q <= ST_NORM2;
                end
                ST_NORM2: begin
                    if (norm_left2) begin
                        magnitude_q <= {magnitude_q[25:0], 2'd0};
                        exponent_q <= alu_sum[8:0];
                    end
                    state_q <= ST_NORM1;
                end
                ST_NORM1: begin
                    if (norm_left1) begin
                        magnitude_q <= {magnitude_q[26:0], 1'b0};
                        exponent_q <= alu_sum[8:0];
                    end
                    state_q <= ST_ROUND;
                end

                ST_ROUND: begin
                    rounded_q <= alu_sum[24:0];
                    state_q <= ST_FINAL;
                end

                ST_FINAL: begin
                    if (nonfinite_q) begin
                        result_data_q <= 32'd0;
                        result_status_q <= STATUS_NONFINITE;
                    end else if (special_q) begin
                        result_data_q <= special_data_q;
                        result_status_q <= 2'd0;
                    end else if (rounded_q == 25'd0) begin
                        // Exact cancellation is +0 in round-to-nearest mode.
                        result_data_q <= 32'd0;
                        result_status_q <= 2'd0;
                    end else if (rounded_q[24] || rounded_q[23]) begin
                        if (final_exp[8] || (&final_exp[7:0])) begin
                            result_data_q <= {sign_q, 8'hff, 23'd0};
                            result_status_q <= STATUS_OVERFLOW;
                        end else begin
                            result_data_q <= {
                                sign_q,
                                final_exp[7:0],
                                final_sig[22:0]
                            };
                            result_status_q <= 2'd0;
                        end
                    end else begin
                        result_data_q <= {sign_q, 8'd0, rounded_q[22:0]};
                        result_status_q <= 2'd0;
                    end
                    state_q <= ST_RESULT;
                end

                ST_RESULT: if (result_ready) begin
                    state_q <= ST_IDLE;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                end

                default: begin
                    state_q <= ST_IDLE;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
