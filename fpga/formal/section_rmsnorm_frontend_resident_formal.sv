`default_nettype none

// Real-leaf resident replay proof at rows=8, tokens=2. Each token is one
// scratch group. The environment retains exactly one untagged response and,
// after abort, returns it to the frontend's drain state before allowing restart.
module section_rmsnorm_frontend_resident_formal(input wire clk);
    localparam [2:0] SC_CLEAN       = 3'd0;
    localparam [2:0] SC_MAX_FAULT   = 3'd1;
    localparam [2:0] SC_SUM_FAULT   = 3'd2;
    localparam [2:0] SC_ABORT_MAX   = 3'd3;
    localparam [2:0] SC_ABORT_SUM   = 3'd4;

    localparam       PHASE_MAX      = 1'b0;
    localparam       PHASE_SUM      = 1'b1;

    localparam [6:0] STATUS_MAXEXP  = 7'h04;
    localparam [6:0] STATUS_SCRATCH = 7'h10;
    localparam [47:0] EXPECTED_SUM  = 48'h0020_0000_0000;

`ifdef FORMAL_RESIDENT_MAX_FAULT
    localparam [2:0] ACTIVE_SCENARIO = SC_MAX_FAULT;
`elsif FORMAL_RESIDENT_SUM_FAULT
    localparam [2:0] ACTIVE_SCENARIO = SC_SUM_FAULT;
`elsif FORMAL_RESIDENT_ABORT_MAX
    localparam [2:0] ACTIVE_SCENARIO = SC_ABORT_MAX;
`elsif FORMAL_RESIDENT_ABORT_SUM
    localparam [2:0] ACTIVE_SCENARIO = SC_ABORT_SUM;
`else
    localparam [2:0] ACTIVE_SCENARIO = SC_CLEAN;
`endif

    (* anyseq *) reg rst_n;
    reg f_past_valid = 1'b0;

    reg cfg_pending_q;
    reg restart_needed_q;
    reg restarted_q;
    reg aborted_q;
    reg response_pending_q;
    reg response_phase_q;
    reg [1:0] response_token_q;
    reg [2:0] response_age_q;

    reg [2:0] max_requests_q;
    reg [2:0] max_responses_q;
    reg [2:0] max_results_q;
    reg [2:0] sum_requests_q;
    reg [2:0] sum_responses_q;
    reg [2:0] sum_results_q;

    reg saw_expected_fault_q;
    reg saw_abort_owner_q;
    reg saw_drain_response_q;
    reg saw_max_terminal_response_q;
    reg saw_max_pass_completion_q;
    reg saw_sum_terminal_response_q;
    reg saw_sum_terminal_result_q;
    reg saw_clean_completion_q;
    reg configured_resident_q;

    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire busy;
    wire done;
    wire error;
    wire [6:0] status;

    wire s_axis_tready;
    wire r_wr_valid;
    wire [1:0] r_wr_bank;
    wire [13:0] r_wr_address;
    wire [63:0] r_wr_data;

    wire rd_req_valid;
    wire rd_req_ready = !response_pending_q;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_valid = response_pending_q && (response_age_q >= 3'd3);
    wire rd_rsp_ready;
    wire [255:0] rd_rsp_data = response_token_q == 0 ?
                               {8{32'h3f80_0000}} :
                               {8{32'h3f00_0000}};
    wire rd_rsp_error = !restarted_q && response_pending_q &&
        (((ACTIVE_SCENARIO == SC_MAX_FAULT) &&
          (response_phase_q == PHASE_MAX)) ||
         ((ACTIVE_SCENARIO == SC_SUM_FAULT) &&
          (response_phase_q == PHASE_SUM)));

    wire result_valid;
    wire [1:0] result_token;
    wire [7:0] result_max_exp;
    wire [47:0] result_sum_sq;
    wire [13:0] result_rows;
    wire result_final;
    wire formal_maxexp_done;
    wire formal_maxexp_error;
    wire formal_maxexp_result_fire;
    wire [3:0] formal_max_records_after;
    wire formal_sum_done;
    wire formal_sum_error;
    wire formal_sum_result_fire;
    wire [3:0] formal_sum_records_after;
    wire formal_replay_complete_after;
    wire formal_replay_outstanding_after;

    wire cfg_fire = cfg_valid && cfg_ready;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire result_fire = result_valid;

    wire request_phase = max_requests_q < 3'd2 ? PHASE_MAX : PHASE_SUM;
    wire [3:0] max_responses_after = {1'b0, max_responses_q} +
        (response_fire && (response_phase_q == PHASE_MAX));
    wire [3:0] sum_responses_after = {1'b0, sum_responses_q} +
        (response_fire && (response_phase_q == PHASE_SUM));
    wire [3:0] sum_results_after = {1'b0, sum_results_q} + result_fire;

    wire abort_phase_selected =
        ((ACTIVE_SCENARIO == SC_ABORT_MAX) &&
         (response_phase_q == PHASE_MAX)) ||
        ((ACTIVE_SCENARIO == SC_ABORT_SUM) &&
         (response_phase_q == PHASE_SUM));
    wire abort_run = !restarted_q && !aborted_q &&
                     response_pending_q && (response_age_q == 3'd1) &&
                     abort_phase_selected;

    section_rmsnorm_frontend dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(14'd8), .cfg_tokens(3'd2), .cfg_resident(1'b1),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_axis_tdata(64'h3f80_0000_3f80_0000),
        .s_axis_tkeep(8'hff), .s_axis_tvalid(1'b1),
        .s_axis_tready(s_axis_tready), .s_axis_tlast(1'b0),
        .r_wr_valid(r_wr_valid), .r_wr_ready(1'b1),
        .r_wr_error(1'b0), .r_wr_bank(r_wr_bank),
        .r_wr_address(r_wr_address), .r_wr_data(r_wr_data),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_token(rd_req_token), .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error),
        .result_valid(result_valid), .result_ready(1'b1),
        .result_token(result_token), .result_max_exp(result_max_exp),
        .result_sum_sq(result_sum_sq), .result_rows(result_rows),
        .result_final(result_final),
        .formal_maxexp_done(formal_maxexp_done),
        .formal_maxexp_error(formal_maxexp_error),
        .formal_maxexp_result_fire(formal_maxexp_result_fire),
        .formal_max_records_after(formal_max_records_after),
        .formal_sum_done(formal_sum_done),
        .formal_sum_error(formal_sum_error),
        .formal_sum_result_fire(formal_sum_result_fire),
        .formal_sum_records_after(formal_sum_records_after),
        .formal_replay_complete_after(formal_replay_complete_after),
        .formal_replay_outstanding_after(formal_replay_outstanding_after)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            restart_needed_q <= 1'b0;
            restarted_q <= 1'b0;
            aborted_q <= 1'b0;
            response_pending_q <= 1'b0;
            response_phase_q <= PHASE_MAX;
            response_token_q <= 2'd0;
            response_age_q <= 3'd0;
            max_requests_q <= 3'd0;
            max_responses_q <= 3'd0;
            max_results_q <= 3'd0;
            sum_requests_q <= 3'd0;
            sum_responses_q <= 3'd0;
            sum_results_q <= 3'd0;
            saw_expected_fault_q <= 1'b0;
            saw_abort_owner_q <= 1'b0;
            saw_drain_response_q <= 1'b0;
            saw_max_terminal_response_q <= 1'b0;
            saw_max_pass_completion_q <= 1'b0;
            saw_sum_terminal_response_q <= 1'b0;
            saw_sum_terminal_result_q <= 1'b0;
            saw_clean_completion_q <= 1'b0;
            configured_resident_q <= 1'b0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                configured_resident_q <= 1'b1;
                max_requests_q <= 3'd0;
                max_responses_q <= 3'd0;
                max_results_q <= 3'd0;
                sum_requests_q <= 3'd0;
                sum_responses_q <= 3'd0;
                sum_results_q <= 3'd0;
                if (restart_needed_q) begin
                    restarted_q <= 1'b1;
                    restart_needed_q <= 1'b0;
                end
            end

            if (request_fire) begin
                response_pending_q <= 1'b1;
                response_phase_q <= request_phase;
                response_token_q <= rd_req_token[1:0];
                response_age_q <= 3'd0;
                if (request_phase == PHASE_MAX)
                    max_requests_q <= max_requests_q + 1'b1;
                else begin
                    sum_requests_q <= sum_requests_q + 1'b1;
                    if (sum_requests_q == 0)
                        saw_max_pass_completion_q <= 1'b1;
                end
            end else if (response_pending_q && !response_fire &&
                         response_age_q != 3'd7) begin
                response_age_q <= response_age_q + 1'b1;
            end

            if (response_fire) begin
                response_pending_q <= 1'b0;
                response_age_q <= 3'd0;
                if (response_phase_q == PHASE_MAX)
                    max_responses_q <= max_responses_q + 1'b1;
                else
                    sum_responses_q <= sum_responses_q + 1'b1;
                if (aborted_q && !restarted_q)
                    saw_drain_response_q <= 1'b1;
            end

            if (result_fire)
                sum_results_q <= sum_results_q + 1'b1;
            if (formal_maxexp_result_fire)
                max_results_q <= max_results_q + 1'b1;

            if (response_fire && (response_phase_q == PHASE_MAX) &&
                (response_token_q == 2'd1))
                saw_max_terminal_response_q <= 1'b1;
            if (result_fire && result_final)
                saw_sum_terminal_result_q <= 1'b1;
            if (response_fire && (response_phase_q == PHASE_SUM) &&
                (response_token_q == 2'd1))
                saw_sum_terminal_response_q <= 1'b1;

            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                restart_needed_q <= 1'b1;
                aborted_q <= 1'b1;
                saw_abort_owner_q <= response_pending_q && !rd_rsp_valid;
            end

            if (done && error) begin
                cfg_pending_q <= 1'b1;
                restart_needed_q <= 1'b1;
                saw_expected_fault_q <= 1'b1;
            end

            if (done && !error) begin
                saw_clean_completion_q <= 1'b1;
            end
        end

        if (rst_n) begin
            assert(!(done && busy));
            assert(!(result_valid && error));
            assert(max_requests_q <= 3'd2);
            assert(max_responses_q <= max_requests_q);
            assert(max_results_q <= 3'd2);
            assert(sum_requests_q <= 3'd2);
            assert(sum_responses_q <= sum_requests_q);
            assert(sum_results_q <= 3'd2);

            // A resident configuration never exposes the external load/write
            // channels, including while an owned read is being drained.
            if (configured_resident_q) begin
                assert(!s_axis_tready);
                assert(!r_wr_valid);
            end

            if (rd_req_valid) begin
                assert(!response_pending_q);
                assert(rd_req_group == 11'd0);
                if (request_phase == PHASE_MAX) begin
                    assert(rd_req_token == max_requests_q);
                    assert(sum_requests_q == 0);
                end else begin
                    assert(rd_req_token == sum_requests_q);
                    assert(max_requests_q == 2);
                    assert(max_responses_q == 2);
                end
            end

            if (response_pending_q) begin
                assert(response_token_q < 2);
                if (response_phase_q == PHASE_MAX)
                    assert(max_responses_q < max_requests_q);
                else
                    assert(sum_responses_q < sum_requests_q);
            end

            if (result_valid) begin
                assert(result_token == sum_results_q[1:0]);
                assert(result_max_exp ==
                       (result_token == 0 ? 8'd127 : 8'd126));
                assert(result_sum_sq == EXPECTED_SUM);
                assert(result_rows == 14'd8);
                assert(result_final == (sum_results_q == 3'd1));
            end

            // Inclusive counts remain exact if a terminal response, child
            // completion, or final exported result shares the accepting clock.
            if (response_fire && (response_phase_q == PHASE_MAX) &&
                (response_token_q == 2'd1)) begin
                assert(max_responses_after == 4'd2);
            end
            if (request_fire && (request_phase == PHASE_SUM) &&
                (sum_requests_q == 0)) begin
                assert(max_responses_after == 4'd2);
                assert(!response_pending_q);
            end
            if (response_fire && (response_phase_q == PHASE_SUM) &&
                (response_token_q == 2'd1)) begin
                assert(sum_responses_after == 4'd2);
            end
            if (result_fire && result_final) begin
                assert(sum_results_after == 4'd2);
            end
            if (formal_maxexp_result_fire) begin
                assert(formal_max_records_after ==
                       ({1'b0, max_results_q} + 1'b1));
            end
            if (formal_maxexp_done && !formal_maxexp_error) begin
                assert(formal_max_records_after == 4'd2);
                assert(({1'b0, max_results_q} +
                        formal_maxexp_result_fire) == 4'd2);
                assert(formal_replay_complete_after);
                assert(!formal_replay_outstanding_after);
            end
            if (formal_sum_result_fire) begin
                assert(formal_sum_records_after == sum_results_after);
            end
            if (formal_sum_done && !formal_sum_error) begin
                assert(formal_sum_records_after == 4'd2);
                assert(sum_results_after == 4'd2);
                assert(formal_replay_complete_after);
                assert(!formal_replay_outstanding_after);
            end

            if (abort_run) begin
                assert(response_pending_q);
                assert(!rd_rsp_valid);
            end
            if (aborted_q && !restarted_q) begin
                assert(!done);
                assert(!error);
                assert(status == 7'd0);
                if (response_pending_q)
                    assert(!cfg_ready);
            end

            if (done && error) begin
                assert(!restarted_q);
                assert(sum_results_q == 0);
                if (ACTIVE_SCENARIO == SC_MAX_FAULT) begin
                    assert((status & STATUS_MAXEXP) != 0);
                    assert((status & STATUS_SCRATCH) != 0);
                    assert(max_requests_q == 1);
                    assert(max_responses_q == 1);
                    assert(sum_requests_q == 0);
                    assert(sum_responses_q == 0);
                end else if (ACTIVE_SCENARIO == SC_SUM_FAULT) begin
                    assert((status & STATUS_SCRATCH) != 0);
                    assert(max_requests_q == 2);
                    assert(max_responses_q == 2);
                    assert(sum_requests_q == 1);
                    assert(sum_responses_q == 1);
                end else begin
                    assert(1'b0);
                end
            end

            if (done && !error) begin
                assert(status == 0);
                assert(max_requests_q == 2);
                assert(max_responses_after == 2);
                assert(max_results_q == 2);
                assert(sum_requests_q == 2);
                assert(sum_responses_after == 2);
                assert(sum_results_after == 2);
                if (ACTIVE_SCENARIO != SC_CLEAN)
                    assert(restarted_q);
            end
        end

`ifdef FORMAL_RESIDENT_CLEAN
        cover(rst_n && done && !error &&
              saw_max_terminal_response_q && saw_max_pass_completion_q &&
              saw_sum_terminal_response_q && saw_sum_terminal_result_q);
`elsif FORMAL_RESIDENT_MAX_FAULT
        cover(rst_n && done && !error && restarted_q &&
              saw_expected_fault_q);
`elsif FORMAL_RESIDENT_SUM_FAULT
        cover(rst_n && done && !error && restarted_q &&
              saw_expected_fault_q);
`elsif FORMAL_RESIDENT_ABORT_MAX
        cover(rst_n && done && !error && restarted_q &&
              saw_abort_owner_q && saw_drain_response_q);
`elsif FORMAL_RESIDENT_ABORT_SUM
        cover(rst_n && done && !error && restarted_q &&
              saw_abort_owner_q && saw_drain_response_q);
`endif
    end

    wire _unused = &{1'b0, r_wr_bank, r_wr_address, r_wr_data,
                     saw_clean_completion_q};
endmodule

`default_nettype wire
