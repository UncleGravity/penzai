// Four-lane RMSNorm reduction and inverse service.
//
// One command covers one physical four-token wave. The resident source is read
// twice: an exponent-max pass followed by an exact fixed-point sum-of-squares
// pass. Four accumulator banks per token remove the loop-carried II hazard.
// D=2560 is converted to the inverse leaf's power-of-two identity by computing
// round_down(sum * 8 / 5) and presenting 4096 rows.

`default_nettype none

module rms_reduce4 (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [12:0]   cfg_rows,
    input  wire [3:0]    cfg_lane_mask,
    input  wire [31:0]   cfg_epsilon,
    input  wire          abort_run,
    output wire          busy,

    output wire          src_req_valid,
    input  wire          src_req_ready,
    output wire [11:0]   src_req_addr,
    input  wire          src_rsp_valid,
    output wire          src_rsp_ready,
    input  wire [127:0]  src_rsp_data,
    input  wire          src_rsp_error,

    output wire          result_valid,
    input  wire          result_ready,
    output wire          result_error,
    output wire [7:0]    result_status,
    output wire [127:0]  result_inv_rms
);
    localparam [4:0] ST_IDLE       = 5'd0;
    localparam [4:0] ST_MAX        = 5'd1;
    localparam [4:0] ST_SUM        = 5'd2;
    localparam [4:0] ST_SUM_DRAIN  = 5'd3;
    localparam [4:0] ST_MERGE_01   = 5'd4;
    localparam [4:0] ST_MERGE_ALL  = 5'd5;
    localparam [4:0] ST_DIV_INIT   = 5'd6;
    localparam [4:0] ST_DIV_RUN    = 5'd7;
    localparam [4:0] ST_DIV_DONE   = 5'd8;
    localparam [4:0] ST_INV_CFG    = 5'd9;
    localparam [4:0] ST_INV_IO     = 5'd10;
    localparam [4:0] ST_RESULT     = 5'd11;
    localparam [4:0] ST_CLEANUP    = 5'd12;
    localparam [4:0] ST_ERROR_DRAIN = 5'd13;

    localparam [7:0] STATUS_BAD_CFG    = 8'h01;
    localparam [7:0] STATUS_SOURCE     = 8'h02;
    localparam [7:0] STATUS_NONFINITE  = 8'h04;
    localparam [7:0] STATUS_SUBNORMAL  = 8'h08;
    localparam [7:0] STATUS_OVERFLOW   = 8'h10;
    localparam [7:0] STATUS_INVERSE    = 8'h20;
    localparam [7:0] STATUS_INTERNAL   = 8'h40;

    reg [4:0] state_q;
    reg [12:0] rows_q;
    reg [3:0] lane_mask_q;
    reg [2:0] lane_count_q;
    reg [31:0] epsilon_q;
    reg [13:0] issue_count_q;
    reg [13:0] rsp_count_q;
    reg [7:0] max_exp_q [0:3];
    // Snapshot the completed exponent pass beside the square pipeline.  The
    // running maxima are physically tied to the source-response island and
    // otherwise create a long routed path into the four square DSP inputs.
    (* keep = "true" *) reg [7:0] sum_exp_q [0:3];

    reg q_valid_q;
    reg q_last_q;
    reg [1:0] q_bank_q;
    reg [17:0] quant_q [0:3];
    reg square_valid_q;
    reg square_last_q;
    // Each lane gets a preserved copy of the bank selector so the two control
    // bits do not drive every bit of all sixteen accumulator banks.
    (* keep = "true", dont_touch = "true" *)
        reg [1:0] square_bank_lane_q [0:3];
    (* use_dsp = "yes" *) reg [35:0] square_q [0:3];
    reg accum_valid_q;
    reg accum_last_q;
    reg [3:0] accum_bank_oh_q;
    reg [47:0] accum_base_q [0:3];
    reg [47:0] accum_square_q [0:3];
    reg sum_update_valid_q;
    reg sum_update_last_q;
    reg [3:0] sum_update_bank_oh_q;
    (* use_dsp = "yes" *) reg [47:0] sum_update_q [0:3];
    reg [47:0] sum_bank_q [0:15];
    reg [48:0] merge01_q [0:3];
    reg [48:0] merge23_q [0:3];
    reg [49:0] merged_q [0:3];
    reg [47:0] inverse_sum_q [0:3];

    reg [50:0] div_numerator_q [0:3];
    reg [50:0] div_quotient_q [0:3];
    reg [2:0] div_remainder_q [0:3];
    reg [5:0] div_bit_q;

    reg [2:0] inv_send_count_q;
    reg [2:0] inv_capture_count_q;
    reg [31:0] inv_result_q [0:3];
    reg result_error_q;
    reg [7:0] result_status_q;

    function automatic [2:0] mask_count(input [3:0] mask);
        begin
            mask_count = {2'd0, mask[0]} + {2'd0, mask[1]} +
                         {2'd0, mask[2]} + {2'd0, mask[3]};
        end
    endfunction

    function automatic [1:0] nth_active_lane(
        input [3:0] mask,
        input [1:0] nth
    );
        integer scan;
        reg [2:0] seen;
        begin
            nth_active_lane = 2'd0;
            seen = 3'd0;
            for (scan = 0; scan < 4; scan = scan + 1) begin
                if (mask[scan]) begin
                    if (seen[1:0] == nth)
                        nth_active_lane = scan[1:0];
                    seen = seen + 1'b1;
                end
            end
        end
    endfunction

    function automatic [17:0] quantize(
        input [31:0] value,
        input [7:0] maximum_exp
    );
        reg [7:0] delta;
        reg [4:0] shift;
        reg [23:0] significand;
        reg [23:0] shifted;
        begin
            if (value[30:23] == 8'd0) begin
                quantize = 18'd0;
            end else begin
                delta = maximum_exp - value[30:23];
                shift = 5'd6 + delta[4:0];
                significand = {1'b1, value[22:0]};
                shifted = significand >> shift;
                quantize = delta >= 8'd18 ? 18'd0 : shifted[17:0];
            end
        end
    endfunction

    wire cfg_rows_ok = (cfg_rows == 13'd128) ||
                       (cfg_rows == 13'd2048) ||
                       (cfg_rows == 13'd2560) ||
                       (cfg_rows == 13'd4096);
    wire cfg_eps_ok = !cfg_epsilon[31] &&
                      (cfg_epsilon[30:23] != 8'd0) &&
                      (cfg_epsilon[30:23] != 8'hff);
    wire cfg_ok = cfg_rows_ok && (cfg_lane_mask != 4'd0) && cfg_eps_ok;
    wire cfg_fire = cfg_valid && cfg_ready;

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;
    assign src_req_addr = issue_count_q[11:0];

    wire lane0_nonfinite = lane_mask_q[0] &&
                           (src_rsp_data[30:23] == 8'hff);
    wire lane1_nonfinite = lane_mask_q[1] &&
                           (src_rsp_data[62:55] == 8'hff);
    wire lane2_nonfinite = lane_mask_q[2] &&
                           (src_rsp_data[94:87] == 8'hff);
    wire lane3_nonfinite = lane_mask_q[3] &&
                           (src_rsp_data[126:119] == 8'hff);
    wire rsp_nonfinite = lane0_nonfinite || lane1_nonfinite ||
                         lane2_nonfinite || lane3_nonfinite;
    wire lane0_subnormal = lane_mask_q[0] &&
        (src_rsp_data[30:23] == 8'd0) && (src_rsp_data[22:0] != 23'd0);
    wire lane1_subnormal = lane_mask_q[1] &&
        (src_rsp_data[62:55] == 8'd0) && (src_rsp_data[54:32] != 23'd0);
    wire lane2_subnormal = lane_mask_q[2] &&
        (src_rsp_data[94:87] == 8'd0) && (src_rsp_data[86:64] != 23'd0);
    wire lane3_subnormal = lane_mask_q[3] &&
        (src_rsp_data[126:119] == 8'd0) &&
        (src_rsp_data[118:96] != 23'd0);
    wire rsp_subnormal = lane0_subnormal || lane1_subnormal ||
                         lane2_subnormal || lane3_subnormal;
    wire rsp_bad = src_rsp_valid &&
                   (src_rsp_error || rsp_nonfinite || rsp_subnormal);

    assign src_req_valid = rst_n && !abort_run &&
        ((state_q == ST_MAX) || (state_q == ST_SUM)) &&
        (issue_count_q < {1'b0, rows_q}) && !rsp_bad;
    assign src_rsp_ready = rst_n &&
        ((state_q == ST_MAX) || (state_q == ST_SUM) ||
         (state_q == ST_CLEANUP) || (state_q == ST_ERROR_DRAIN));
    wire src_req_fire = src_req_valid && src_req_ready;
    wire src_rsp_fire = src_rsp_valid && src_rsp_ready;

    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_error = result_error_q;
    assign result_status = result_status_q;
    assign result_inv_rms = {inv_result_q[3], inv_result_q[2],
                             inv_result_q[1], inv_result_q[0]};

    wire [7:0] rsp_exp0 = src_rsp_data[30:23];
    wire [7:0] rsp_exp1 = src_rsp_data[62:55];
    wire [7:0] rsp_exp2 = src_rsp_data[94:87];
    wire [7:0] rsp_exp3 = src_rsp_data[126:119];

    wire inv_cfg_ready;
    wire inv_busy;
    wire inv_done;
    wire inv_error;
    wire [3:0] inv_status;
    wire inv_s_ready;
    wire inv_result_valid;
    wire [1:0] inv_result_token;
    wire [31:0] inv_result_value;
    wire inv_result_final;
    wire [13:0] inverse_rows = rows_q == 13'd2560 ?
                               14'd4096 : {1'b0, rows_q};
    wire inv_cfg_valid = state_q == ST_INV_CFG;
    wire inv_s_valid = (state_q == ST_INV_IO) &&
                       (inv_send_count_q < lane_count_q);
    wire inv_s_fire = inv_s_valid && inv_s_ready;
    wire inv_result_ready = state_q == ST_INV_IO;
    wire inv_result_fire = inv_result_valid && inv_result_ready;

    wire [3:0] div_shift0 = {div_remainder_q[0],
                             div_numerator_q[0][div_bit_q]};
    wire [3:0] div_shift1 = {div_remainder_q[1],
                             div_numerator_q[1][div_bit_q]};
    wire [3:0] div_shift2 = {div_remainder_q[2],
                             div_numerator_q[2][div_bit_q]};
    wire [3:0] div_shift3 = {div_remainder_q[3],
                             div_numerator_q[3][div_bit_q]};
    wire [3:0] div_shift [0:3];
    wire [3:0] div_reduced [0:3];
    assign div_shift[0] = div_shift0;
    assign div_shift[1] = div_shift1;
    assign div_shift[2] = div_shift2;
    assign div_shift[3] = div_shift3;
    assign div_reduced[0] = div_shift0 >= 4'd5 ? div_shift0 - 4'd5 : div_shift0;
    assign div_reduced[1] = div_shift1 >= 4'd5 ? div_shift1 - 4'd5 : div_shift1;
    assign div_reduced[2] = div_shift2 >= 4'd5 ? div_shift2 - 4'd5 : div_shift2;
    assign div_reduced[3] = div_shift3 >= 4'd5 ? div_shift3 - 4'd5 : div_shift3;

    wire [1:0] inv_source_lane =
        nth_active_lane(lane_mask_q, inv_send_count_q[1:0]);
    wire [1:0] inv_result_lane =
        nth_active_lane(lane_mask_q, inv_result_token);
    wire [7:0] selected_max_exp = max_exp_q[inv_source_lane];
    wire [47:0] selected_inverse_sum = inverse_sum_q[inv_source_lane];

    rms_inverse u_inverse (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(inv_cfg_valid), .cfg_ready(inv_cfg_ready),
        .cfg_rows(inverse_rows), .cfg_tokens(lane_count_q),
        .cfg_eps(epsilon_q), .abort_run(abort_run),
        .busy(inv_busy), .done(inv_done), .error(inv_error),
        .status(inv_status), .s_valid(inv_s_valid), .s_ready(inv_s_ready),
        .s_token(inv_send_count_q[1:0]), .s_max_exp(selected_max_exp),
        .s_sum_sq(selected_inverse_sum), .s_rows(inverse_rows),
        .s_final((inv_send_count_q + 1'b1) == lane_count_q),
        .result_valid(inv_result_valid), .result_ready(inv_result_ready),
        .result_token(inv_result_token),
        .result_inv_rms(inv_result_value), .result_final(inv_result_final)
    );

    integer lane;
    integer bank;
    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            rows_q <= 13'd0;
            lane_mask_q <= 4'd0;
            lane_count_q <= 3'd0;
            epsilon_q <= 32'd0;
            issue_count_q <= 14'd0;
            rsp_count_q <= 14'd0;
            q_valid_q <= 1'b0;
            q_last_q <= 1'b0;
            q_bank_q <= 2'd0;
            square_valid_q <= 1'b0;
            square_last_q <= 1'b0;
            accum_valid_q <= 1'b0;
            accum_last_q <= 1'b0;
            accum_bank_oh_q <= 4'd0;
            sum_update_valid_q <= 1'b0;
            sum_update_last_q <= 1'b0;
            sum_update_bank_oh_q <= 4'd0;
            div_bit_q <= 6'd0;
            inv_send_count_q <= 3'd0;
            inv_capture_count_q <= 3'd0;
            result_error_q <= 1'b0;
            result_status_q <= 8'd0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                max_exp_q[lane] <= 8'd0;
                sum_exp_q[lane] <= 8'd0;
                quant_q[lane] <= 18'd0;
                square_q[lane] <= 36'd0;
                square_bank_lane_q[lane] <= 2'd0;
                accum_base_q[lane] <= 48'd0;
                accum_square_q[lane] <= 48'd0;
                sum_update_q[lane] <= 48'd0;
                merge01_q[lane] <= 49'd0;
                merge23_q[lane] <= 49'd0;
                merged_q[lane] <= 50'd0;
                inverse_sum_q[lane] <= 48'd0;
                div_numerator_q[lane] <= 51'd0;
                div_quotient_q[lane] <= 51'd0;
                div_remainder_q[lane] <= 3'd0;
                inv_result_q[lane] <= 32'd0;
            end
            for (bank = 0; bank < 16; bank = bank + 1)
                sum_bank_q[bank] <= 48'd0;
        end else if (abort_run) begin
            q_valid_q <= 1'b0;
            square_valid_q <= 1'b0;
            accum_valid_q <= 1'b0;
            sum_update_valid_q <= 1'b0;
            result_error_q <= 1'b0;
            result_status_q <= 8'd0;
            // A previously issued response may handshake on the abort edge.
            // Count it before deciding whether cleanup still has work.
            if (src_rsp_fire)
                rsp_count_q <= rsp_count_q + 1'b1;
            if ((issue_count_q == rsp_count_q) ||
                (src_rsp_fire &&
                 (issue_count_q == (rsp_count_q + 1'b1))) ||
                (state_q == ST_IDLE) || (state_q == ST_RESULT))
                state_q <= ST_IDLE;
            else
                state_q <= ST_CLEANUP;
        end else begin
            // Pipeline valids drain naturally after the final sum response.
            q_valid_q <= 1'b0;
            square_valid_q <= q_valid_q;
            square_last_q <= q_last_q;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                square_bank_lane_q[lane] <= q_bank_q;
                square_q[lane] <= quant_q[lane] * quant_q[lane];
            end

            // The four accumulator banks are revisited only every fourth
            // response.  Use those free cycles to split select, DSP add, and
            // bank writeback into distinct stages while retaining II=1.
            accum_valid_q <= square_valid_q;
            accum_last_q <= square_last_q;
            accum_bank_oh_q <= 4'b0001 << square_bank_lane_q[0];
            for (lane = 0; lane < 4; lane = lane + 1) begin
                case (square_bank_lane_q[lane])
                    2'd0: accum_base_q[lane] <= sum_bank_q[lane*4];
                    2'd1: accum_base_q[lane] <= sum_bank_q[lane*4 + 1];
                    2'd2: accum_base_q[lane] <= sum_bank_q[lane*4 + 2];
                    default:
                        accum_base_q[lane] <= sum_bank_q[lane*4 + 3];
                endcase
                accum_square_q[lane] <= {12'd0, square_q[lane]};
            end

            sum_update_valid_q <= accum_valid_q;
            sum_update_last_q <= accum_last_q;
            sum_update_bank_oh_q <= accum_bank_oh_q;
            for (lane = 0; lane < 4; lane = lane + 1)
                sum_update_q[lane] <= accum_base_q[lane] +
                                      accum_square_q[lane];

            if (src_req_fire)
                issue_count_q <= issue_count_q + 1'b1;
            if (src_rsp_fire)
                rsp_count_q <= rsp_count_q + 1'b1;

            if (sum_update_valid_q) begin
                for (lane = 0; lane < 4; lane = lane + 1) begin
                    if (sum_update_bank_oh_q[0])
                        sum_bank_q[lane*4] <= sum_update_q[lane];
                    if (sum_update_bank_oh_q[1])
                        sum_bank_q[lane*4 + 1] <= sum_update_q[lane];
                    if (sum_update_bank_oh_q[2])
                        sum_bank_q[lane*4 + 2] <= sum_update_q[lane];
                    if (sum_update_bank_oh_q[3])
                        sum_bank_q[lane*4 + 3] <= sum_update_q[lane];
                end
            end

            case (state_q)
                ST_IDLE: if (cfg_fire) begin
                    rows_q <= cfg_rows;
                    lane_mask_q <= cfg_lane_mask;
                    lane_count_q <= mask_count(cfg_lane_mask);
                    epsilon_q <= cfg_epsilon;
                    issue_count_q <= 14'd0;
                    rsp_count_q <= 14'd0;
                    result_error_q <= !cfg_ok;
                    result_status_q <= cfg_ok ? 8'd0 : STATUS_BAD_CFG;
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        max_exp_q[lane] <= 8'd0;
                        sum_exp_q[lane] <= 8'd0;
                        inv_result_q[lane] <= 32'd0;
                    end
                    state_q <= cfg_ok ? ST_MAX : ST_RESULT;
                end

                ST_MAX: if (src_rsp_fire) begin
                    if (src_rsp_error || rsp_nonfinite || rsp_subnormal) begin
                        result_error_q <= 1'b1;
                        result_status_q <= src_rsp_error ? STATUS_SOURCE :
                            (rsp_nonfinite ? STATUS_NONFINITE :
                                             STATUS_SUBNORMAL);
                        q_valid_q <= 1'b0;
                        square_valid_q <= 1'b0;
                        state_q <= ST_ERROR_DRAIN;
                    end else begin
                        if (lane_mask_q[0] && (rsp_exp0 > max_exp_q[0]))
                            max_exp_q[0] <= rsp_exp0;
                        if (lane_mask_q[1] && (rsp_exp1 > max_exp_q[1]))
                            max_exp_q[1] <= rsp_exp1;
                        if (lane_mask_q[2] && (rsp_exp2 > max_exp_q[2]))
                            max_exp_q[2] <= rsp_exp2;
                        if (lane_mask_q[3] && (rsp_exp3 > max_exp_q[3]))
                            max_exp_q[3] <= rsp_exp3;
                        if ((rsp_count_q + 1'b1) == {1'b0, rows_q}) begin
                            issue_count_q <= 14'd0;
                            rsp_count_q <= 14'd0;
                            sum_exp_q[0] <= lane_mask_q[0] &&
                                (rsp_exp0 > max_exp_q[0]) ?
                                rsp_exp0 : max_exp_q[0];
                            sum_exp_q[1] <= lane_mask_q[1] &&
                                (rsp_exp1 > max_exp_q[1]) ?
                                rsp_exp1 : max_exp_q[1];
                            sum_exp_q[2] <= lane_mask_q[2] &&
                                (rsp_exp2 > max_exp_q[2]) ?
                                rsp_exp2 : max_exp_q[2];
                            sum_exp_q[3] <= lane_mask_q[3] &&
                                (rsp_exp3 > max_exp_q[3]) ?
                                rsp_exp3 : max_exp_q[3];
                            for (bank = 0; bank < 16; bank = bank + 1)
                                sum_bank_q[bank] <= 48'd0;
                            state_q <= ST_SUM;
                        end
                    end
                end

                ST_SUM: if (src_rsp_fire) begin
                    if (src_rsp_error || rsp_nonfinite || rsp_subnormal) begin
                        result_error_q <= 1'b1;
                        result_status_q <= src_rsp_error ? STATUS_SOURCE :
                            (rsp_nonfinite ? STATUS_NONFINITE :
                                             STATUS_SUBNORMAL);
                        q_valid_q <= 1'b0;
                        square_valid_q <= 1'b0;
                        state_q <= ST_ERROR_DRAIN;
                    end else begin
                        q_valid_q <= 1'b1;
                        q_last_q <= (rsp_count_q + 1'b1) ==
                                    {1'b0, rows_q};
                        q_bank_q <= rsp_count_q[1:0];
                        quant_q[0] <= lane_mask_q[0] ?
                            quantize(src_rsp_data[31:0], sum_exp_q[0]) : 18'd0;
                        quant_q[1] <= lane_mask_q[1] ?
                            quantize(src_rsp_data[63:32], sum_exp_q[1]) : 18'd0;
                        quant_q[2] <= lane_mask_q[2] ?
                            quantize(src_rsp_data[95:64], sum_exp_q[2]) : 18'd0;
                        quant_q[3] <= lane_mask_q[3] ?
                            quantize(src_rsp_data[127:96], sum_exp_q[3]) : 18'd0;
                        if ((rsp_count_q + 1'b1) == {1'b0, rows_q})
                            state_q <= ST_SUM_DRAIN;
                    end
                end

                ST_SUM_DRAIN: if (sum_update_valid_q && sum_update_last_q)
                    state_q <= ST_MERGE_01;

                ST_MERGE_01: begin
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        merge01_q[lane] <=
                            {1'b0, sum_bank_q[lane*4]} +
                            {1'b0, sum_bank_q[lane*4 + 1]};
                        merge23_q[lane] <=
                            {1'b0, sum_bank_q[lane*4 + 2]} +
                            {1'b0, sum_bank_q[lane*4 + 3]};
                    end
                    state_q <= ST_MERGE_ALL;
                end

                ST_MERGE_ALL: begin
                    for (lane = 0; lane < 4; lane = lane + 1)
                        merged_q[lane] <= {1'b0, merge01_q[lane]} +
                                          {1'b0, merge23_q[lane]};
                    state_q <= rows_q == 13'd2560 ?
                               ST_DIV_INIT : ST_INV_CFG;
                    if (rows_q != 13'd2560) begin
                        for (lane = 0; lane < 4; lane = lane + 1)
                            inverse_sum_q[lane] <=
                                merge01_q[lane][47:0] +
                                merge23_q[lane][47:0];
                    end
                end

                ST_DIV_INIT: begin
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        div_numerator_q[lane] <= {merged_q[lane][47:0], 3'b000};
                        div_quotient_q[lane] <= 51'd0;
                        div_remainder_q[lane] <= 3'd0;
                    end
                    div_bit_q <= 6'd50;
                    state_q <= ST_DIV_RUN;
                end

                ST_DIV_RUN: begin
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        div_remainder_q[lane] <= div_reduced[lane][2:0];
                        div_quotient_q[lane][div_bit_q] <=
                            div_shift[lane] >= 4'd5;
                    end
                    if (div_bit_q == 6'd0)
                        state_q <= ST_DIV_DONE;
                    else
                        div_bit_q <= div_bit_q - 1'b1;
                end

                ST_DIV_DONE: begin
                    for (lane = 0; lane < 4; lane = lane + 1)
                        inverse_sum_q[lane] <= div_quotient_q[lane][47:0];
                    if ((|div_quotient_q[0][50:48]) ||
                        (|div_quotient_q[1][50:48]) ||
                        (|div_quotient_q[2][50:48]) ||
                        (|div_quotient_q[3][50:48]) ||
                        (|merged_q[0][49:48]) || (|merged_q[1][49:48]) ||
                        (|merged_q[2][49:48]) || (|merged_q[3][49:48])) begin
                        result_error_q <= 1'b1;
                        result_status_q <= STATUS_OVERFLOW;
                        state_q <= ST_RESULT;
                    end else begin
                        state_q <= ST_INV_CFG;
                    end
                end

                ST_INV_CFG: if (inv_cfg_ready) begin
                    inv_send_count_q <= 3'd0;
                    inv_capture_count_q <= 3'd0;
                    state_q <= ST_INV_IO;
                end

                ST_INV_IO: begin
                    if (inv_s_fire)
                        inv_send_count_q <= inv_send_count_q + 1'b1;
                    if (inv_result_fire) begin
                        inv_result_q[inv_result_lane] <= inv_result_value;
                        inv_capture_count_q <= inv_capture_count_q + 1'b1;
                        if (inv_result_final) begin
                            if ((inv_capture_count_q + 1'b1) != lane_count_q) begin
                                result_error_q <= 1'b1;
                                result_status_q <= STATUS_INTERNAL;
                            end
                            state_q <= ST_RESULT;
                        end
                    end
                    if (inv_error) begin
                        result_error_q <= 1'b1;
                        result_status_q <= STATUS_INVERSE | {4'd0, inv_status};
                        state_q <= ST_RESULT;
                    end
                end

                ST_RESULT: if (result_valid && result_ready) begin
                    state_q <= ST_IDLE;
                    result_error_q <= 1'b0;
                    result_status_q <= 8'd0;
                end

                ST_CLEANUP: begin
                    q_valid_q <= 1'b0;
                    square_valid_q <= 1'b0;
                    accum_valid_q <= 1'b0;
                    sum_update_valid_q <= 1'b0;
                    if (issue_count_q == rsp_count_q)
                        state_q <= ST_IDLE;
                end

                ST_ERROR_DRAIN: begin
                    q_valid_q <= 1'b0;
                    square_valid_q <= 1'b0;
                    accum_valid_q <= 1'b0;
                    sum_update_valid_q <= 1'b0;
                    if ((issue_count_q == rsp_count_q) ||
                        (src_rsp_fire &&
                         (issue_count_q == (rsp_count_q + 1'b1))))
                        state_q <= ST_RESULT;
                end

                default: begin
                    result_error_q <= 1'b1;
                    result_status_q <= STATUS_INTERNAL;
                    state_q <= ST_RESULT;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && src_rsp_valid && !src_rsp_ready &&
            ((state_q == ST_MAX) || (state_q == ST_SUM)))
            $fatal(1, " rms_reduce4 refused an active source response");
        if (rst_n && result_valid && !result_error &&
            (inv_capture_count_q != lane_count_q))
            $fatal(1, " rms_reduce4 published an incomplete inverse vector");
        if (rst_n && result_valid && result_error &&
            ((result_status == STATUS_SOURCE) ||
             (result_status == STATUS_NONFINITE) ||
             (result_status == STATUS_SUBNORMAL)) &&
            (issue_count_q != rsp_count_q))
            $fatal(1, " rms_reduce4 published before source drain");
    end
`endif
endmodule

`default_nettype wire
