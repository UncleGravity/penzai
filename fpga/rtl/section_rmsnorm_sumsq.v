// Section-local RMSNorm sum-of-squares reduction.
//
// One configured maximum FP32 exponent per token fixes the accumulator scale:
//
//   d = E - e
//   q = d >= 18 ? 0 : ({1'b1, mantissa} >> (6 + d))
//   S = sum(q * q)
//
// S is exact in 48 bits for at most 4096 rows.  The represented mean is
// (S / rows) * 2**(2 * (E - 144)).  This is a deliberately truncated diagnostic
// reduction, not the PS RMSNorm arithmetic.  Subnormals contribute zero and set
// a nonfatal warning which integration must reject before architectural publish.
//
// Result records are tentative until the final record handshakes and done pulses.
// A fatal error in a later token cannot retract an earlier accepted record.

`default_nettype none

module section_rmsnorm_sumsq (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    // Token zero occupies bits 7:0; token three occupies bits 31:24.
    input  wire [31:0]   cfg_max_exp,

    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 BAD_CFG, 1 NONFINITE, 2 MAX_MISMATCH, 3 FRAME,
    // bit 4 SCRATCH, 5 INTERNAL, 6 SUBNORMAL_WARNING.
    output reg  [6:0]    status,

    // Token-major scratch groups. Lane zero is data[31:0].
    input  wire [255:0]  s_group_data,
    input  wire          s_group_error,
    input  wire          s_group_valid,
    output wire          s_group_ready,
    input  wire          s_group_last,

    output wire          result_valid,
    input  wire          result_ready,
    output wire [1:0]    result_token,
    output wire [7:0]    result_max_exp,
    output reg  [47:0]   result_sum_sq,
    output wire [13:0]   result_rows,
    output reg           result_subnormal_warning,
    output wire          result_final
);
    localparam [6:0] STATUS_BAD_CFG            = 7'b0000001;
    localparam [6:0] STATUS_NONFINITE          = 7'b0000010;
    localparam [6:0] STATUS_MAX_MISMATCH       = 7'b0000100;
    localparam [6:0] STATUS_FRAME              = 7'b0001000;
    localparam [6:0] STATUS_SCRATCH            = 7'b0010000;
    localparam [6:0] STATUS_INTERNAL           = 7'b0100000;
    localparam [6:0] STATUS_SUBNORMAL_WARNING  = 7'b1000000;

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_INPUT  = 3'd1;
    localparam [2:0] ST_LANES  = 3'd2;
    localparam [2:0] ST_DRAIN  = 3'd3;
    localparam [2:0] ST_RESULT = 3'd4;

    reg [2:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [31:0] run_max_exp_q;
    reg [9:0] run_groups_q;
    reg [1:0] token_q;
    reg [9:0] group_q;
    reg [2:0] lane_q;
    reg [255:0] group_data_q;
    (* use_dsp = "yes" *) reg [47:0] sum_q;
    reg token_subnormal_q;
    reg saw_max_exp_q;
    reg [17:0] quant_q;
    reg quant_valid_q;
    reg [35:0] product_q;
    reg product_valid_q;
    reg [7:0] token_max_exp;

    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) &&
                        (cfg_rows[2:0] == 3'b000) &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4);
    wire cfg_exp_ok = ((cfg_tokens < 3'd1) || (cfg_max_exp[7:0] != 8'hff)) &&
                      ((cfg_tokens < 3'd2) || (cfg_max_exp[15:8] != 8'hff)) &&
                      ((cfg_tokens < 3'd3) || (cfg_max_exp[23:16] != 8'hff)) &&
                      ((cfg_tokens < 3'd4) || (cfg_max_exp[31:24] != 8'hff));
    wire cfg_accept = cfg_valid && cfg_ready;

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;
    assign s_group_ready = rst_n && !abort_run && (state_q == ST_INPUT);
    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_token = token_q;
    assign result_max_exp = token_max_exp;
    assign result_rows = run_rows_q;
    assign result_final = ({1'b0, token_q} + 3'd1) == run_tokens_q;

    wire group_accept = s_group_valid && s_group_ready;
    wire expected_group_last = (({1'b0, token_q} + 3'd1) == run_tokens_q) &&
                               (group_q + 1'b1 == run_groups_q);
    wire group_frame_bad = s_group_last != expected_group_last;

    wire [31:0] lane_data = group_data_q[lane_q * 32 +: 32];
    wire [7:0] lane_exp = lane_data[30:23];
    wire [22:0] lane_mantissa = lane_data[22:0];
    wire lane_zero = lane_exp == 8'd0 && lane_mantissa == 23'd0;
    wire lane_subnormal = lane_exp == 8'd0 && lane_mantissa != 23'd0;
    wire lane_nonfinite = lane_exp == 8'hff;

    always @* begin
        case (token_q)
            2'd0: token_max_exp = run_max_exp_q[7:0];
            2'd1: token_max_exp = run_max_exp_q[15:8];
            2'd2: token_max_exp = run_max_exp_q[23:16];
            default: token_max_exp = run_max_exp_q[31:24];
        endcase
    end

    // E comes from the preceding scanner. Values above it fail immediately;
    // token completion also requires that a nonzero E was observed exactly.
    wire lane_max_mismatch = !lane_zero && !lane_subnormal && !lane_nonfinite &&
                             ((token_max_exp == 8'd0) ||
                              (lane_exp > token_max_exp));
    wire [7:0] lane_delta = token_max_exp - lane_exp;
    wire [4:0] lane_shift = 5'd6 + lane_delta[4:0];
    wire [23:0] lane_significand = {1'b1, lane_mantissa};
    wire [23:0] lane_shifted = lane_delta >= 8'd18 ? 24'd0 :
                               (lane_significand >> lane_shift);
    wire [17:0] lane_quant = (lane_zero || lane_subnormal) ? 18'd0 :
                             lane_shifted[17:0];

    // Lane k is decoded while lane k-1 is squared. The product register is
    // deliberately unconditional and resetless so Vivado can absorb it into the
    // DSP PREG; product_valid_q makes stale values unobservable. Two drain cycles
    // retire the last product. Full-range unsigned 18x18 uses two DSP48E2s.
    (* use_dsp = "yes" *) wire [35:0] quant_product = quant_q * quant_q;
    wire [47:0] mac_sum = sum_q + {12'd0, product_q};
    always @(posedge clk)
        product_q <= quant_product;
`ifdef FORMAL
    // The legal geometry proves the bit discarded by production's canonical
    // 48-bit MAC is unreachable; keep this observer out of synthesized logic.
    wire [48:0] formal_sum_ext = {1'b0, sum_q} + {13'd0, product_q};
`endif

    task automatic fail_run(input [6:0] failure);
        begin
            state_q <= ST_IDLE;
            done <= 1'b1;
            error <= 1'b1;
            status <= status | failure;
            sum_q <= 48'd0;
            token_subnormal_q <= 1'b0;
            saw_max_exp_q <= 1'b0;
            quant_q <= 18'd0;
            quant_valid_q <= 1'b0;
            product_valid_q <= 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_max_exp_q <= 32'd0;
            run_groups_q <= 10'd0;
            token_q <= 2'd0;
            group_q <= 10'd0;
            lane_q <= 3'd0;
            group_data_q <= 256'd0;
            sum_q <= 48'd0;
            token_subnormal_q <= 1'b0;
            saw_max_exp_q <= 1'b0;
            quant_q <= 18'd0;
            quant_valid_q <= 1'b0;
            product_valid_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
            result_sum_sq <= 48'd0;
            result_subnormal_warning <= 1'b0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            token_q <= 2'd0;
            group_q <= 10'd0;
            lane_q <= 3'd0;
            sum_q <= 48'd0;
            token_subnormal_q <= 1'b0;
            saw_max_exp_q <= 1'b0;
            quant_q <= 18'd0;
            quant_valid_q <= 1'b0;
            product_valid_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
            result_sum_sq <= 48'd0;
            result_subnormal_warning <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state_q)
                ST_IDLE: if (cfg_accept) begin
                    error <= 1'b0;
                    status <= 7'd0;
                    token_q <= 2'd0;
                    group_q <= 10'd0;
                    lane_q <= 3'd0;
                    sum_q <= 48'd0;
                    token_subnormal_q <= 1'b0;
                    saw_max_exp_q <= 1'b0;
                    quant_q <= 18'd0;
                    quant_valid_q <= 1'b0;
                    product_valid_q <= 1'b0;
                    result_sum_sq <= 48'd0;
                    result_subnormal_warning <= 1'b0;
                    if (!cfg_shape_ok || !cfg_exp_ok) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_BAD_CFG;
                    end else begin
                        run_rows_q <= cfg_rows;
                        run_tokens_q <= cfg_tokens;
                        run_max_exp_q <= cfg_max_exp;
                        run_groups_q <= cfg_rows[12:3];
                        state_q <= ST_INPUT;
                    end
                end

                ST_INPUT: if (group_accept) begin
                    if (s_group_error) begin
                        fail_run(STATUS_SCRATCH);
                    end else if (group_frame_bad) begin
                        fail_run(STATUS_FRAME);
                    end else begin
                        group_data_q <= s_group_data;
                        lane_q <= 3'd0;
                        quant_valid_q <= 1'b0;
                        product_valid_q <= 1'b0;
                        state_q <= ST_LANES;
                    end
                end

                ST_LANES: begin
                    if (lane_nonfinite) begin
                        fail_run(STATUS_NONFINITE);
                    end else if (lane_max_mismatch) begin
                        fail_run(STATUS_MAX_MISMATCH);
                    end else begin
                        if (product_valid_q)
                            sum_q <= mac_sum;
                        product_valid_q <= quant_valid_q;
                        quant_q <= lane_quant;
                        quant_valid_q <= 1'b1;
                        if (lane_subnormal) begin
                            token_subnormal_q <= 1'b1;
                            status <= status | STATUS_SUBNORMAL_WARNING;
                        end
                        if (!lane_zero && !lane_subnormal &&
                            (lane_exp == token_max_exp))
                            saw_max_exp_q <= 1'b1;
                        if (lane_q == 3'd7)
                            state_q <= ST_DRAIN;
                        else
                            lane_q <= lane_q + 1'b1;
                    end
                end

                ST_DRAIN: begin
                    quant_valid_q <= 1'b0;
                    product_valid_q <= quant_valid_q;
                    if (product_valid_q)
                        sum_q <= mac_sum;
                    if (!quant_valid_q) begin
                        if (!product_valid_q) begin
                            fail_run(STATUS_INTERNAL);
                        end else if (group_q + 1'b1 != run_groups_q) begin
                            group_q <= group_q + 1'b1;
                            state_q <= ST_INPUT;
                        end else if ((token_max_exp != 8'd0) &&
                                     !saw_max_exp_q) begin
                            fail_run(STATUS_MAX_MISMATCH);
                        end else begin
                            result_sum_sq <= mac_sum;
                            result_subnormal_warning <= token_subnormal_q;
                            state_q <= ST_RESULT;
                        end
                    end
                end

                ST_RESULT: if (result_ready) begin
                    if (result_final) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else begin
                        token_q <= token_q + 1'b1;
                        group_q <= 10'd0;
                        lane_q <= 3'd0;
                        sum_q <= 48'd0;
                        token_subnormal_q <= 1'b0;
                        saw_max_exp_q <= 1'b0;
                        quant_q <= 18'd0;
                        quant_valid_q <= 1'b0;
                        product_valid_q <= 1'b0;
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
            assert(!(s_group_ready && result_valid));
            if (product_valid_q &&
                ((state_q == ST_LANES) || (state_q == ST_DRAIN)))
                assert(!formal_sum_ext[48]);
            if (result_valid) begin
                assert(result_token < run_tokens_q);
                assert(result_rows == run_rows_q);
            end
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_max_exp == $past(result_max_exp));
            assert(result_sum_sq == $past(result_sum_sq));
            assert(result_rows == $past(result_rows));
            assert(result_subnormal_warning ==
                   $past(result_subnormal_warning));
            assert(result_final == $past(result_final));
        end
        if (f_past_valid && rst_n && $past(rst_n && abort_run)) begin
            assert(!busy);
            assert(!result_valid);
            assert(!error);
            assert(status == 7'd0);
        end
    end
`endif

endmodule

`default_nettype wire
