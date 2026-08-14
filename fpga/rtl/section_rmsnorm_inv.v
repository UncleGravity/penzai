// Section-local inverse-RMS scalar for the P3d fixed reduction.
//
// The input identity is:
//   mean = (sum_sq / rows) * 2**(2 * (max_exp - 144))
//
// Power-of-two rows make the division an exponent adjustment. The mean rounds
// once to binary32, epsilon is added with the shared truncating fadd, and two
// Newton steps refine the classic affine inverse-square-root seed using one
// shared fmul instance. Results are deterministic hardware approximations, not
// an IEEE sqrt claim. Input records remain tentative until final completion.

`default_nettype none

module section_rmsnorm_inv (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         cfg_valid,
    output wire         cfg_ready,
    input  wire [13:0]  cfg_rows,
    input  wire [2:0]   cfg_tokens,
    input  wire [31:0]  cfg_eps,

    input  wire         abort_run,
    output wire         busy,
    output reg          done,
    output reg          error,
    // bit 0 BAD_CFG, 1 FRAME, 2 ARITHMETIC, 3 INTERNAL.
    output reg  [3:0]   status,

    input  wire         s_valid,
    output wire         s_ready,
    input  wire [1:0]   s_token,
    input  wire [7:0]   s_max_exp,
    input  wire [47:0]  s_sum_sq,
    input  wire [13:0]  s_rows,
    input  wire         s_final,

    output wire         result_valid,
    input  wire         result_ready,
    output wire [1:0]   result_token,
    output wire [31:0]  result_inv_rms,
    output wire         result_final
`ifdef FORMAL
    , output wire [3:0] formal_state
`endif
);
    localparam [3:0] STATUS_BAD_CFG    = 4'b0001;
    localparam [3:0] STATUS_FRAME      = 4'b0010;
    localparam [3:0] STATUS_ARITHMETIC = 4'b0100;
    localparam [3:0] STATUS_INTERNAL   = 4'b1000;

    localparam [3:0] ST_IDLE         = 4'd0;
    localparam [3:0] ST_INPUT        = 4'd1;
    localparam [3:0] ST_MEAN_SCAN    = 4'd2;
    localparam [3:0] ST_MEAN_SHIFT   = 4'd3;
    localparam [3:0] ST_MEAN_FINAL   = 4'd4;
    localparam [3:0] ST_ADD_ISSUE    = 4'd5;
    localparam [3:0] ST_ADD_WAIT     = 4'd6;
    localparam [3:0] ST_SQ_ISSUE     = 4'd7;
    localparam [3:0] ST_SQ_WAIT      = 4'd8;
    localparam [3:0] ST_SCALE_ISSUE  = 4'd9;
    localparam [3:0] ST_SCALE_WAIT   = 4'd10;
    localparam [3:0] ST_CORR_ISSUE   = 4'd11;
    localparam [3:0] ST_CORR_WAIT    = 4'd12;
    localparam [3:0] ST_REFINE_ISSUE = 4'd13;
    localparam [3:0] ST_REFINE_WAIT  = 4'd14;
    localparam [3:0] ST_RESULT       = 4'd15;

    localparam [31:0] FP32_ONE_POINT_FIVE = 32'h3fc0_0000;
    localparam [31:0] INV_SQRT_MAGIC = 32'h5f37_59df;

    reg [3:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [31:0] run_eps_q;
    reg [3:0] run_row_shift_q;
    reg [1:0] token_q;
    reg [1:0] record_token_q;
    reg record_final_q;
    reg [47:0] mean_sum_q;
    reg [7:0] mean_max_exp_q;
    reg [5:0] mean_msb_q;
    reg [5:0] mean_shift_q;
    reg [23:0] mean_sig_base_q;
    reg mean_round_up_q;
    reg signed [11:0] mean_biased_pre_q;
    reg [31:0] mean_q;
    reg [31:0] half_adjusted_q;
    reg [31:0] estimate_q;
    reg [31:0] square_q;
    reg [31:0] scaled_q;
    reg [31:0] correction_q;
    reg [31:0] result_q;
    reg iteration_q;
    reg [3:0] wait_age_q;

    function automatic [5:0] lead_one48(input [47:0] value);
        integer bit_index;
        begin
            lead_one48 = 6'd0;
            for (bit_index = 0; bit_index < 48; bit_index = bit_index + 1)
                if (value[bit_index])
                    lead_one48 = bit_index[5:0];
        end
    endfunction

    function automatic [3:0] log2_rows(input [13:0] value);
        integer bit_index;
        begin
            log2_rows = 4'd0;
            for (bit_index = 0; bit_index < 14; bit_index = bit_index + 1)
                if (value[bit_index])
                    log2_rows = bit_index[3:0];
        end
    endfunction

    wire cfg_rows_power_two = (cfg_rows != 14'd0) &&
                              ((cfg_rows & (cfg_rows - 1'b1)) == 14'd0);
    wire cfg_eps_ok = !cfg_eps[31] &&
                      (cfg_eps[30:23] != 8'd0) &&
                      (cfg_eps[30:23] != 8'hff);
    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) &&
                        cfg_rows_power_two &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4) && cfg_eps_ok;
    wire cfg_accept = cfg_valid && cfg_ready;

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;
    assign s_ready = rst_n && !abort_run && (state_q == ST_INPUT);
    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_token = record_token_q;
    assign result_inv_rms = result_q;
    assign result_final = record_final_q;
`ifdef FORMAL
    assign formal_state = state_q;
`endif

    wire s_accept = s_valid && s_ready;
    wire expected_final = ({1'b0, token_q} + 3'd1) == run_tokens_q;
    wire record_frame_bad = (s_token != token_q) ||
                            (s_rows != run_rows_q) ||
                            (s_final != expected_final) ||
                            (s_max_exp == 8'hff) ||
                            ((s_sum_sq == 48'd0) != (s_max_exp == 8'd0));

    // Exact positive integer-to-binary32 RNE for the fixed mean identity. The
    // scan, variable shift/round decision, and final carry are separate scalar
    // stages so none of them sits on the section control-enable path.
    wire [5:0] mean_scan_msb = lead_one48(mean_sum_q);
    wire [5:0] mean_scan_shift = mean_scan_msb > 6'd23 ?
                                 mean_scan_msb - 6'd23 : 6'd0;
    wire signed [11:0] mean_scan_msb_signed = {6'd0, mean_scan_msb};
    wire signed [11:0] mean_scan_exp_signed = {4'd0, mean_max_exp_q};
    wire signed [11:0] mean_scan_row_signed = {8'd0, run_row_shift_q};
    wire signed [11:0] mean_scan_biased =
        mean_scan_msb_signed +
        ((mean_scan_exp_signed - 12'sd144) <<< 1) -
        mean_scan_row_signed + 12'sd127;

    wire [63:0] mean_sum_ext = {16'd0, mean_sum_q};
    wire [63:0] mean_shifted = mean_msb_q <= 6'd23 ?
                               (mean_sum_ext << (6'd23 - mean_msb_q)) :
                               (mean_sum_ext >> mean_shift_q);
    wire [47:0] mean_remainder_mask = mean_shift_q == 0 ? 48'd0 :
                                      ((48'd1 << mean_shift_q) - 1'b1);
    wire [47:0] mean_remainder = mean_sum_q & mean_remainder_mask;
    wire [47:0] mean_halfway = mean_shift_q == 0 ? 48'd0 :
                                (48'd1 << (mean_shift_q - 1'b1));
    wire mean_shift_round_up = mean_shift_q != 0 &&
                               ((mean_remainder > mean_halfway) ||
                                ((mean_remainder == mean_halfway) &&
                                 mean_shifted[0]));

    wire [24:0] mean_sig_rounded = {1'b0, mean_sig_base_q} +
                                    mean_round_up_q;
    wire mean_renormalize = mean_sig_rounded[24];
    wire [23:0] mean_sig_normalized = mean_renormalize ?
                                      mean_sig_rounded[24:1] :
                                      mean_sig_rounded[23:0];
    wire signed [11:0] mean_biased = mean_biased_pre_q +
                                     (mean_renormalize ? 12'sd1 : 12'sd0);
    wire mean_overflow = (mean_sum_q != 0) &&
                         (mean_biased >= 12'sd255);
    wire [31:0] mean_bits = (mean_sum_q == 0) || (mean_biased <= 0) ?
                            32'd0 :
                            {1'b0, mean_biased[7:0],
                             mean_sig_normalized[22:0]};

    wire add_issue = (state_q == ST_ADD_ISSUE) ||
                     (state_q == ST_CORR_ISSUE);
    wire [31:0] add_a = state_q == ST_ADD_ISSUE ? mean_q :
                        FP32_ONE_POINT_FIVE;
    wire [31:0] add_b = state_q == ST_ADD_ISSUE ? run_eps_q :
                        {1'b1, scaled_q[30:0]};
    wire add_out_valid;
    wire [31:0] add_out;
    fadd #(.MANT_W(23)) u_add (
        .clk(clk), .rst_n(rst_n), .valid_in(add_issue),
        .a(add_a), .b(add_b), .valid_out(add_out_valid), .out(add_out)
    );

    wire mul_issue = (state_q == ST_SQ_ISSUE) ||
                     (state_q == ST_SCALE_ISSUE) ||
                     (state_q == ST_REFINE_ISSUE);
    wire [31:0] mul_a = state_q == ST_SQ_ISSUE ? estimate_q :
                        state_q == ST_SCALE_ISSUE ? half_adjusted_q :
                        estimate_q;
    wire [31:0] mul_b = state_q == ST_SQ_ISSUE ? estimate_q :
                        state_q == ST_SCALE_ISSUE ? square_q :
                        correction_q;
    wire mul_out_valid;
    wire [31:0] mul_out;
    fmul #(.MANT_W(23)) u_mul (
        .clk(clk), .rst_n(rst_n), .valid_in(mul_issue),
        .a(mul_a), .b(mul_b), .valid_out(mul_out_valid), .out(mul_out)
    );

    function automatic positive_normal(input [31:0] value);
        positive_normal = !value[31] &&
                          (value[30:23] != 8'd0) &&
                          (value[30:23] != 8'hff);
    endfunction

    task automatic fail_run(input [3:0] failure);
        begin
            state_q <= ST_IDLE;
            done <= 1'b1;
            error <= 1'b1;
            status <= status | failure;
            result_q <= 32'd0;
            wait_age_q <= 4'd0;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_eps_q <= 32'd0;
            run_row_shift_q <= 4'd0;
            token_q <= 2'd0;
            record_token_q <= 2'd0;
            record_final_q <= 1'b0;
            mean_sum_q <= 48'd0;
            mean_max_exp_q <= 8'd0;
            mean_msb_q <= 6'd0;
            mean_shift_q <= 6'd0;
            mean_sig_base_q <= 24'd0;
            mean_round_up_q <= 1'b0;
            mean_biased_pre_q <= 12'sd0;
            mean_q <= 32'd0;
            half_adjusted_q <= 32'd0;
            estimate_q <= 32'd0;
            square_q <= 32'd0;
            scaled_q <= 32'd0;
            correction_q <= 32'd0;
            result_q <= 32'd0;
            iteration_q <= 1'b0;
            wait_age_q <= 4'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            token_q <= 2'd0;
            record_token_q <= 2'd0;
            record_final_q <= 1'b0;
            mean_sum_q <= 48'd0;
            mean_max_exp_q <= 8'd0;
            mean_msb_q <= 6'd0;
            mean_shift_q <= 6'd0;
            mean_sig_base_q <= 24'd0;
            mean_round_up_q <= 1'b0;
            mean_biased_pre_q <= 12'sd0;
            result_q <= 32'd0;
            iteration_q <= 1'b0;
            wait_age_q <= 4'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else begin
            done <= 1'b0;
            case (state_q)
                ST_IDLE: if (cfg_accept) begin
                    error <= 1'b0;
                    status <= 4'd0;
                    token_q <= 2'd0;
                    record_token_q <= 2'd0;
                    record_final_q <= 1'b0;
                    result_q <= 32'd0;
                    iteration_q <= 1'b0;
                    wait_age_q <= 4'd0;
                    if (!cfg_shape_ok) begin
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_BAD_CFG;
                    end else begin
                        run_rows_q <= cfg_rows;
                        run_tokens_q <= cfg_tokens;
                        run_eps_q <= cfg_eps;
                        run_row_shift_q <= log2_rows(cfg_rows);
                        state_q <= ST_INPUT;
                    end
                end

                ST_INPUT: if (s_accept) begin
                    if (record_frame_bad) begin
                        fail_run(STATUS_FRAME);
                    end else begin
                        record_token_q <= s_token;
                        record_final_q <= s_final;
                        mean_sum_q <= s_sum_sq;
                        mean_max_exp_q <= s_max_exp;
                        state_q <= ST_MEAN_SCAN;
                    end
                end

                ST_MEAN_SCAN: begin
                    mean_msb_q <= mean_scan_msb;
                    mean_shift_q <= mean_scan_shift;
                    mean_biased_pre_q <= mean_scan_biased;
                    state_q <= ST_MEAN_SHIFT;
                end

                ST_MEAN_SHIFT: begin
                    mean_sig_base_q <= mean_shifted[23:0];
                    mean_round_up_q <= mean_shift_round_up;
                    state_q <= ST_MEAN_FINAL;
                end

                ST_MEAN_FINAL: begin
                    if (mean_overflow) begin
                        fail_run(STATUS_ARITHMETIC);
                    end else begin
                        mean_q <= mean_bits;
                        state_q <= ST_ADD_ISSUE;
                    end
                end

                ST_ADD_ISSUE: begin
                    wait_age_q <= 4'd0;
                    state_q <= ST_ADD_WAIT;
                end
                ST_ADD_WAIT: begin
                    if (add_out_valid && wait_age_q >= 4'd3) begin
                        if (!positive_normal(add_out) ||
                            add_out[30:23] < 8'd2 ||
                            add_out[30:23] > 8'd251) begin
                            fail_run(STATUS_ARITHMETIC);
                        end else begin
                            half_adjusted_q <= add_out - 32'h0080_0000;
                            estimate_q <= INV_SQRT_MAGIC - (add_out >> 1);
                            iteration_q <= 1'b0;
                            state_q <= ST_SQ_ISSUE;
                        end
                    end else if (wait_age_q == 4'd15) begin
                        fail_run(STATUS_INTERNAL);
                    end else begin
                        wait_age_q <= wait_age_q + 1'b1;
                    end
                end

                ST_SQ_ISSUE: begin
                    wait_age_q <= 4'd0;
                    state_q <= ST_SQ_WAIT;
                end
                ST_SQ_WAIT: begin
                    if (mul_out_valid && wait_age_q >= 4'd2) begin
                        if (!positive_normal(mul_out))
                            fail_run(STATUS_ARITHMETIC);
                        else begin
                            square_q <= mul_out;
                            state_q <= ST_SCALE_ISSUE;
                        end
                    end else if (wait_age_q == 4'd15) begin
                        fail_run(STATUS_INTERNAL);
                    end else begin
                        wait_age_q <= wait_age_q + 1'b1;
                    end
                end

                ST_SCALE_ISSUE: begin
                    wait_age_q <= 4'd0;
                    state_q <= ST_SCALE_WAIT;
                end
                ST_SCALE_WAIT: begin
                    if (mul_out_valid && wait_age_q >= 4'd2) begin
                        if (!positive_normal(mul_out))
                            fail_run(STATUS_ARITHMETIC);
                        else begin
                            scaled_q <= mul_out;
                            state_q <= ST_CORR_ISSUE;
                        end
                    end else if (wait_age_q == 4'd15) begin
                        fail_run(STATUS_INTERNAL);
                    end else begin
                        wait_age_q <= wait_age_q + 1'b1;
                    end
                end

                ST_CORR_ISSUE: begin
                    wait_age_q <= 4'd0;
                    state_q <= ST_CORR_WAIT;
                end
                ST_CORR_WAIT: begin
                    if (add_out_valid && wait_age_q >= 4'd3) begin
                        if (!positive_normal(add_out))
                            fail_run(STATUS_ARITHMETIC);
                        else begin
                            correction_q <= add_out;
                            state_q <= ST_REFINE_ISSUE;
                        end
                    end else if (wait_age_q == 4'd15) begin
                        fail_run(STATUS_INTERNAL);
                    end else begin
                        wait_age_q <= wait_age_q + 1'b1;
                    end
                end

                ST_REFINE_ISSUE: begin
                    wait_age_q <= 4'd0;
                    state_q <= ST_REFINE_WAIT;
                end
                ST_REFINE_WAIT: begin
                    if (mul_out_valid && wait_age_q >= 4'd2) begin
                        if (!positive_normal(mul_out)) begin
                            fail_run(STATUS_ARITHMETIC);
                        end else if (!iteration_q) begin
                            estimate_q <= mul_out;
                            iteration_q <= 1'b1;
                            state_q <= ST_SQ_ISSUE;
                        end else begin
                            result_q <= mul_out;
                            state_q <= ST_RESULT;
                        end
                    end else if (wait_age_q == 4'd15) begin
                        fail_run(STATUS_INTERNAL);
                    end else begin
                        wait_age_q <= wait_age_q + 1'b1;
                    end
                end

                ST_RESULT: if (result_ready) begin
                    if (record_final_q) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else begin
                        token_q <= token_q + 1'b1;
                        state_q <= ST_INPUT;
                    end
                end

                default: fail_run(STATUS_INTERNAL);
            endcase
        end
    end

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(result_valid && error));
            assert(!(s_ready && result_valid));
            if (busy) begin
                assert(run_rows_q >= 14'd8 && run_rows_q <= 14'd4096);
                assert((run_rows_q & (run_rows_q - 1'b1)) == 0);
                assert(run_tokens_q >= 3'd1 && run_tokens_q <= 3'd4);
                assert(token_q < run_tokens_q);
            end
            if (result_valid) begin
                assert(result_token == token_q);
                assert(result_inv_rms[31] == 1'b0);
                assert(result_inv_rms[30:23] != 8'd0);
                assert(result_inv_rms[30:23] != 8'hff);
                assert(result_final == expected_final);
            end
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_inv_rms == $past(result_inv_rms));
            assert(result_final == $past(result_final));
        end
        if (f_past_valid && rst_n && $past(rst_n && abort_run)) begin
            assert(!busy);
            assert(!result_valid);
            assert(!error);
            assert(status == 4'd0);
        end
    end
`endif

endmodule

`default_nettype wire
