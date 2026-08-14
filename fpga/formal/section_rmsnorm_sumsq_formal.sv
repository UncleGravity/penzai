`default_nettype none

// Reduced full-token proof boundary: rows=8 gives one group per token while
// retaining all eight serialized lanes and the production four-token terminal
// width. Arithmetic is checked against an independent fixed-integer function.
module section_rmsnorm_sumsq_formal(input wire clk);
    localparam [3:0] SC_CLEAN       = 4'd0;
    localparam [3:0] SC_EARLY_LAST  = 4'd1;
    localparam [3:0] SC_LATE_LAST   = 4'd2;
    localparam [3:0] SC_SCRATCH     = 4'd3;
    localparam [3:0] SC_NONFINITE   = 4'd4;
    localparam [3:0] SC_MAX_LOW     = 4'd5;
    localparam [3:0] SC_MAX_HIGH    = 4'd6;
    localparam [3:0] SC_SUBNORMAL   = 4'd7;
    localparam [3:0] SC_ABORT       = 4'd8;

    localparam [6:0] ST_BAD_CFG      = 7'h01;
    localparam [6:0] ST_NONFINITE    = 7'h02;
    localparam [6:0] ST_MAX_MISMATCH = 7'h04;
    localparam [6:0] ST_FRAME        = 7'h08;
    localparam [6:0] ST_SCRATCH      = 7'h10;
    localparam [6:0] ST_SUBNORMAL    = 7'h40;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg input_allow;
    (* anyseq *) reg result_ready_any;
    (* anyseq *) reg abort_request;
    (* anyconst *) reg [3:0] scenario;
    (* anyconst *) reg [1:0] abort_phase;

`ifdef FORMAL_COVER
    wire [3:0] active_scenario = scenario;
`elsif FORMAL_FAULTS
    wire [3:0] active_scenario = scenario;
`else
    wire [3:0] active_scenario = SC_CLEAN;
`endif

    reg cfg_pending_q;
    reg [2:0] sent_groups_q;
    reg [2:0] accepted_results_q;
    reg aborted_q;
    reg restarted_q;
    reg [3:0] group_age_q;
    reg f_past_valid = 1'b0;

    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [31:0] cfg_max_exp =
        (active_scenario == SC_MAX_LOW)  ? 32'h7e7e_7e7e :
        (active_scenario == SC_MAX_HIGH) ? 32'h7f7f_807f :
                                           32'h7f7f_7f7f;

    wire busy;
    wire done;
    wire error;
    wire [6:0] status;
    wire group_ready;
    wire result_valid;
    wire [1:0] result_token;
    wire [7:0] result_max_exp;
    wire [47:0] result_sum_sq;
    wire [13:0] result_rows;
    wire result_subnormal_warning;
    wire result_final;

    function automatic [7:0] configured_exp(input [1:0] token);
        case (token)
            2'd0: configured_exp = cfg_max_exp[7:0];
            2'd1: configured_exp = cfg_max_exp[15:8];
            2'd2: configured_exp = cfg_max_exp[23:16];
            default: configured_exp = cfg_max_exp[31:24];
        endcase
    endfunction

    function automatic [31:0] healthy_lane(
        input [1:0] token,
        input [2:0] lane
    );
        reg [7:0] exponent;
        reg [22:0] mantissa;
        begin
            exponent = (lane == 0) ? 8'd127 : 8'd127 - {5'd0, lane};
            mantissa = {token, lane, 18'h2a155};
            healthy_lane = {token[0], exponent, mantissa};
        end
    endfunction

    function automatic [255:0] scenario_group(
        input [3:0] which,
        input [1:0] token
    );
        reg [255:0] value;
        integer lane;
        begin
            value = 256'd0;
            for (lane = 0; lane < 8; lane = lane + 1)
                value[lane*32 +: 32] = healthy_lane(token, lane[2:0]);
            if ((which == SC_NONFINITE) && (token == 0))
                value[2*32 +: 32] = 32'h7f80_0000;
            if ((which == SC_SUBNORMAL) && (token == 0))
                value[2*32 +: 32] = 32'h0000_0001;
            scenario_group = value;
        end
    endfunction

    function automatic [47:0] fixed_group_sum(
        input [255:0] data,
        input [7:0] max_exp
    );
        reg [47:0] total;
        reg [31:0] bits;
        reg [7:0] exponent;
        reg [7:0] delta;
        reg [23:0] significand;
        reg [17:0] quant;
        reg [35:0] product;
        integer lane;
        begin
            total = 48'd0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                bits = data[lane*32 +: 32];
                exponent = bits[30:23];
                if (exponent == 0) begin
                    quant = 18'd0;
                end else begin
                    delta = max_exp - exponent;
                    significand = {1'b1, bits[22:0]};
                    quant = (delta >= 18) ? 18'd0 :
                            significand >> (6 + delta);
                end
                product = quant * quant;
                total = total + {12'd0, product};
            end
            fixed_group_sum = total;
        end
    endfunction

    localparam [47:0] CLEAN_SUM_0 = fixed_group_sum(
        scenario_group(SC_CLEAN, 2'd0), 8'd127
    );
    localparam [47:0] CLEAN_SUM_1 = fixed_group_sum(
        scenario_group(SC_CLEAN, 2'd1), 8'd127
    );
    localparam [47:0] CLEAN_SUM_2 = fixed_group_sum(
        scenario_group(SC_CLEAN, 2'd2), 8'd127
    );
    localparam [47:0] CLEAN_SUM_3 = fixed_group_sum(
        scenario_group(SC_CLEAN, 2'd3), 8'd127
    );
    localparam [47:0] SUBNORMAL_SUM_0 = fixed_group_sum(
        scenario_group(SC_SUBNORMAL, 2'd0), 8'd127
    );

    function automatic [47:0] clean_expected_sum(input [1:0] token);
        case (token)
            2'd0: clean_expected_sum = CLEAN_SUM_0;
            2'd1: clean_expected_sum = CLEAN_SUM_1;
            2'd2: clean_expected_sum = CLEAN_SUM_2;
            default: clean_expected_sum = CLEAN_SUM_3;
        endcase
    endfunction

    function automatic [47:0] scenario_expected_sum(
        input [3:0] which,
        input [1:0] token
    );
        if ((which == SC_SUBNORMAL) && (token == 0))
            scenario_expected_sum = SUBNORMAL_SUM_0;
        else
            scenario_expected_sum = clean_expected_sum(token);
    endfunction

    wire [255:0] group_data = scenario_group(active_scenario,
                                             sent_groups_q[1:0]);
`ifdef FORMAL_DATA
    wire group_valid = busy && (sent_groups_q < 4);
    wire result_ready = 1'b1;
`elsif FORMAL_FAULTS
    wire group_valid = busy && (sent_groups_q < 4);
    wire result_ready = 1'b1;
`elsif FORMAL_COVER
    wire group_valid = busy && (sent_groups_q < 4);
    wire result_ready = 1'b1;
`else
    wire group_valid = busy && (sent_groups_q < 4) && input_allow;
    wire result_ready = result_ready_any;
`endif
    wire group_last = (active_scenario == SC_EARLY_LAST) ?
                      (sent_groups_q == 0) :
                      (active_scenario == SC_LATE_LAST) ? 1'b0 :
                      (sent_groups_q == 3);
    wire group_error = (active_scenario == SC_SCRATCH) &&
                       (sent_groups_q == 0);

    wire cover_abort_point;
    wire abort_run;

    wire cfg_fire = cfg_valid && cfg_ready;
    wire group_fire = group_valid && group_ready;
    wire result_fire = result_valid && result_ready;

    section_rmsnorm_sumsq dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(14'd8), .cfg_tokens(3'd4),
        .cfg_max_exp(cfg_max_exp), .abort_run(abort_run),
        .busy(busy), .done(done), .error(error), .status(status),
        .s_group_data(group_data), .s_group_error(group_error),
        .s_group_valid(group_valid), .s_group_ready(group_ready),
        .s_group_last(group_last),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_max_exp(result_max_exp),
        .result_sum_sq(result_sum_sq), .result_rows(result_rows),
        .result_subnormal_warning(result_subnormal_warning),
        .result_final(result_final)
    );

    // Abort covers input, lane issue, drain, and stalled-result ownership. The
    // group-age tracker derives each phase from public handshakes. Age one is
    // the first lane-issue cycle, ages nine and ten are the two drain cycles,
    // and age eleven owns the result before its handshake.
    assign cover_abort_point =
        (abort_phase == 0) ? (busy && !cfg_pending_q &&
                              sent_groups_q == 0 &&
                              group_age_q == 0) :
        (abort_phase == 1) ? (busy && group_age_q == 4) :
        (abort_phase == 2) ? (busy && group_age_q == 9) :
                             (busy && group_age_q == 11);
`ifdef FORMAL_COVER
    assign abort_run = (active_scenario == SC_ABORT) && !aborted_q &&
                       cover_abort_point;
`elsif FORMAL_BMC
    assign abort_run = abort_request && busy && !aborted_q;
`else
    assign abort_run = 1'b0;
`endif

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);
`ifdef FORMAL_COVER
        assume(scenario <= SC_ABORT);
`elsif FORMAL_FAULTS
        assume(scenario >= SC_EARLY_LAST && scenario <= SC_SUBNORMAL);
`else
        assume(scenario == SC_CLEAN);
`endif

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            sent_groups_q <= 3'd0;
            accepted_results_q <= 3'd0;
            aborted_q <= 1'b0;
            restarted_q <= 1'b0;
            group_age_q <= 4'd0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                sent_groups_q <= 3'd0;
                accepted_results_q <= 3'd0;
                group_age_q <= 4'd0;
                if (aborted_q) restarted_q <= 1'b1;
            end
            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                sent_groups_q <= 3'd0;
                accepted_results_q <= 3'd0;
                aborted_q <= 1'b1;
                group_age_q <= 4'd0;
            end else begin
                if (group_fire) begin
                    sent_groups_q <= sent_groups_q + 1'b1;
                    group_age_q <= 4'd1;
                end else if (result_fire || group_ready) begin
                    group_age_q <= 4'd0;
                end else if ((group_age_q != 0) && busy && !result_valid) begin
                    group_age_q <= group_age_q + 1'b1;
                end
                if (result_fire)
                    accepted_results_q <= accepted_results_q + 1'b1;
            end
        end

        if (rst_n) begin
            assert(accepted_results_q <= 4);
            assert(!status[5]);
            if (result_valid) begin
                assert(result_token == accepted_results_q[1:0]);
                assert(result_max_exp == configured_exp(result_token));
                assert(result_sum_sq == scenario_expected_sum(
                    active_scenario, result_token
                ));
                assert(result_rows == 8);
                assert(result_subnormal_warning ==
                       ((active_scenario == SC_SUBNORMAL) &&
                        (result_token == 0)));
                assert(result_final == (result_token == 3));
            end
            if (done) begin
                if ((active_scenario == SC_CLEAN) ||
                    (active_scenario == SC_SUBNORMAL) ||
                    (active_scenario == SC_ABORT)) begin
                    assert(!error);
                end else begin
                    assert(error);
                end
            end
            if (done && !error) begin
                assert(accepted_results_q == 4);
                assert(status == ((active_scenario == SC_SUBNORMAL) ?
                                  ST_SUBNORMAL : 7'd0));
            end
            if (done && error) begin
                assert(!result_valid);
                if ((active_scenario == SC_EARLY_LAST) ||
                    (active_scenario == SC_LATE_LAST))
                    assert(status == ST_FRAME);
                if (active_scenario == SC_SCRATCH)
                    assert(status == ST_SCRATCH);
                if (active_scenario == SC_NONFINITE)
                    assert(status == ST_NONFINITE);
                if ((active_scenario == SC_MAX_LOW) ||
                    (active_scenario == SC_MAX_HIGH))
                    assert(status == ST_MAX_MISMATCH);

                if ((active_scenario == SC_EARLY_LAST) ||
                    (active_scenario == SC_SCRATCH) ||
                    (active_scenario == SC_NONFINITE) ||
                    (active_scenario == SC_MAX_LOW))
                    assert(accepted_results_q == 0);
                if (active_scenario == SC_MAX_HIGH)
                    assert(accepted_results_q == 1);
                if (active_scenario == SC_LATE_LAST)
                    assert(accepted_results_q == 3);
            end
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && !abort_run && group_valid && !group_ready)) begin
            assume(group_valid);
            assume(group_data == $past(group_data));
            assume(group_last == $past(group_last));
            assume(group_error == $past(group_error));
        end

        cover(rst_n && (active_scenario == SC_CLEAN) && done && !error &&
              accepted_results_q == 4);
        cover(rst_n && (active_scenario == SC_EARLY_LAST) && done && error);
        cover(rst_n && (active_scenario == SC_LATE_LAST) && done && error &&
              accepted_results_q == 3);
        cover(rst_n && (active_scenario == SC_SCRATCH) && done && error);
        cover(rst_n && (active_scenario == SC_NONFINITE) && done && error);
        cover(rst_n && (active_scenario == SC_MAX_LOW) && done && error);
        cover(rst_n && (active_scenario == SC_MAX_HIGH) && done && error &&
              accepted_results_q >= 1);
        cover(rst_n && (active_scenario == SC_SUBNORMAL) && done && !error &&
              status == ST_SUBNORMAL);
        cover(rst_n && (active_scenario == SC_ABORT) &&
              (abort_phase == 0) && aborted_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_ABORT) &&
              (abort_phase == 1) && aborted_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_ABORT) &&
              (abort_phase == 2) && aborted_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_ABORT) &&
              (abort_phase == 3) && aborted_q && restarted_q && done && !error);
    end
endmodule

`default_nettype wire
