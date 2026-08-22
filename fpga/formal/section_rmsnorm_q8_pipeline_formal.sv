`default_nettype none

// Composition proof for the complete scratch-backed RMSNorm-to-Q8 boundary.
// The real reducer, weighted-source, Q8 ingress, arbiter, and lifecycle wrapper
// remain in the cone. Independently checked arithmetic leaves use their bounded
// control stubs so faults, retained reads, abort cleanup, and restart stay
// visible without reproving the numeric datapaths here.
module section_rmsnorm_q8_pipeline_formal(input wire clk);
    localparam [3:0] SC_STARTUP      = 4'd0;
    localparam [3:0] SC_CLEAN        = 4'd1;
    localparam [3:0] SC_REDUCE_FAULT = 4'd2;
    localparam [3:0] SC_SOURCE_FAULT = 4'd3;
    localparam [3:0] SC_ABORT_REDUCE = 4'd4;
    localparam [3:0] SC_ABORT_SOURCE = 4'd5;
    localparam [3:0] SC_ABORT_OUTPUT = 4'd6;
    localparam [3:0] SC_CFG_REJECT   = 4'd7;
    localparam [3:0] SC_ORPHAN_FAULT = 4'd8;

    localparam [2:0] DR_GAMMA_CFG  = 3'd0;
    localparam [2:0] DR_GAMMA_DATA = 3'd1;
    localparam [2:0] DR_GAMMA_WAIT = 3'd2;
    localparam [2:0] DR_RUN_CFG     = 3'd3;
    localparam [2:0] DR_RUN         = 3'd4;
    localparam [2:0] DR_HOLD_ERROR  = 3'd5;
    localparam [2:0] DR_FINISHED    = 3'd6;

    localparam [1:0] PIPE_IDLE = 2'd0;
    localparam [1:0] PIPE_RUN = 2'd1;
    localparam [1:0] PIPE_CLEANUP = 2'd2;
    localparam [1:0] PIPE_START = 2'd3;
    localparam [1:0] OWNER_NONE = 2'd0;
    localparam [1:0] OWNER_REDUCE = 2'd1;
    localparam [1:0] OWNER_SOURCE = 2'd2;

    localparam [13:0] RUN_ROWS = 14'd128;
    localparam [2:0] RUN_TOKENS = 3'd1;
    localparam [31:0] RUN_EPS = 32'h3586_37bd;
    localparam [31:0] GAMMA_SCALAR = 32'h3f00_0000;
    localparam [29:0] STATUS_REDUCE_SCRATCH = 30'h0000_0020;
    localparam [29:0] STATUS_SOURCE_SCRATCH = 30'h0001_0000;
    localparam [29:0] STATUS_SOURCE_GAMMA = 30'h0000_4000;
    localparam [29:0] STATUS_INTERNAL = 30'h2000_0000;
    localparam [7:0] EXPECT_REDUCE_REQUESTS = 8'd1;
    localparam [7:0] EXPECT_SOURCE_REQUESTS = 8'd16;
    localparam [7:0] EXPECT_RESPONSES = 8'd17;
    localparam [5:0] EXPECT_OUTPUT_BEATS = 6'd20;
    localparam [2:0] EXPECT_RECORDS = 3'd4;

`ifdef FORMAL_SC_STARTUP
    wire [3:0] scenario = SC_STARTUP;
`elsif FORMAL_SC_CLEAN
    wire [3:0] scenario = SC_CLEAN;
`elsif FORMAL_SC_REDUCE_FAULT
    wire [3:0] scenario = SC_REDUCE_FAULT;
`elsif FORMAL_SC_SOURCE_FAULT
    wire [3:0] scenario = SC_SOURCE_FAULT;
`elsif FORMAL_SC_ABORT_REDUCE
    wire [3:0] scenario = SC_ABORT_REDUCE;
`elsif FORMAL_SC_ABORT_SOURCE
    wire [3:0] scenario = SC_ABORT_SOURCE;
`elsif FORMAL_SC_ABORT_OUTPUT
    wire [3:0] scenario = SC_ABORT_OUTPUT;
`elsif FORMAL_SC_CFG_REJECT
    wire [3:0] scenario = SC_CFG_REJECT;
`else
    wire [3:0] scenario = SC_ORPHAN_FAULT;
`endif

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;
    reg [12:0] cycle_q = 13'd0;
    reg [2:0] driver_q = DR_GAMMA_CFG;
    reg [6:0] gamma_word_q = 7'd0;
    reg [2:0] cfg_count_q = 3'd0;
    reg [2:0] gamma_load_count_q = 3'd0;
    reg [2:0] clean_done_count_q = 3'd0;
    reg [2:0] error_hold_q = 3'd0;

    reg [1:0] env_owner_q = OWNER_NONE;
    reg [2:0] env_owner_token_q = 3'd0;
    reg [10:0] env_owner_group_q = 11'd0;
    reg [3:0] response_age_q = 4'd0;

    reg [7:0] reduce_request_count_q = 8'd0;
    reg [7:0] source_request_count_q = 8'd0;
    reg [7:0] response_count_q = 8'd0;
    reg [5:0] output_beat_count_q = 6'd0;
    reg [2:0] output_record_count_q = 3'd0;
    reg [2:0] expected_output_beat_q = 3'd0;
    reg [1:0] expected_output_token_q = 2'd0;
    reg [8:0] expected_output_block_q = 9'd0;

    reg saw_atomic_start_q = 1'b0;
    reg saw_reduce_result_q = 1'b0;
    reg saw_reduce_done_q = 1'b0;
    reg saw_final_output_q = 1'b0;
    reg saw_fault_q = 1'b0;
    reg saw_status_before_abort_q = 1'b0;
    reg saw_abort_q = 1'b0;
    reg saw_owned_abort_q = 1'b0;
    reg saw_owned_cleanup_q = 1'b0;
    reg saw_abort_response_q = 1'b0;
    reg saw_output_stall_q = 1'b0;
    reg saw_restart_q = 1'b0;
    reg saw_cfg_reject_q = 1'b0;
    reg saw_orphan_inject_q = 1'b0;
    reg saw_orphan_gamma_invalid_q = 1'b0;
    reg [29:0] captured_status_q = 30'd0;

    wire first_attempt = cfg_count_q == 3'd1;
    wire fault_scenario = (scenario == SC_REDUCE_FAULT) ||
                          (scenario == SC_SOURCE_FAULT);
    wire abort_scenario = (scenario == SC_ABORT_REDUCE) ||
                          (scenario == SC_ABORT_SOURCE) ||
                          (scenario == SC_ABORT_OUTPUT);
    wire cfg_reject_scenario = scenario == SC_CFG_REJECT;
    wire orphan_fault_scenario = scenario == SC_ORPHAN_FAULT;
    wire restart_scenario = fault_scenario || abort_scenario ||
                            cfg_reject_scenario || orphan_fault_scenario;

    function automatic [31:0] response_scalar(
        input [2:0] token,
        input [10:0] group_index,
        input [2:0] lane
    );
        response_scalar = 32'h4000_0000 |
                          {15'd0, token, group_index[6:0], lane};
    endfunction

    function automatic [255:0] response_group(
        input [2:0] token,
        input [10:0] group_index
    );
        integer lane;
        begin
            for (lane = 0; lane < 8; lane = lane + 1)
                response_group[lane*32 +: 32] =
                    response_scalar(token, group_index, lane[2:0]);
        end
    endfunction

    wire gamma_cfg_valid = rst_n && driver_q == DR_GAMMA_CFG;
    wire gamma_cfg_ready;
    wire gamma_tvalid = rst_n && driver_q == DR_GAMMA_DATA;
    wire gamma_tready;
    wire gamma_tlast = gamma_word_q == 7'd63;
    wire gamma_busy;
    wire gamma_done;
    wire gamma_error;
    wire [3:0] gamma_status;
    wire gamma_valid;
    wire gamma_cfg_fire = gamma_cfg_valid && gamma_cfg_ready;
    wire gamma_fire = gamma_tvalid && gamma_tready;

    wire cfg_valid = rst_n && driver_q == DR_RUN_CFG;
    wire cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;
    wire busy;
    wire done;
    wire error;
    wire [29:0] status;
    wire r_wr_valid;
    wire [1:0] r_wr_bank;
    wire [13:0] r_wr_address;
    wire [63:0] r_wr_data;

    wire rd_req_valid;
    wire rd_req_ready = rst_n && env_owner_q == OWNER_NONE;
    wire [1:0] rd_req_role;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire orphan_inject_now;
    wire owned_rd_rsp_valid = rst_n && env_owner_q != OWNER_NONE &&
                              response_age_q >= 4'd4;
    wire rd_rsp_valid = owned_rd_rsp_valid || orphan_inject_now;
    wire rd_rsp_ready;
    wire reduce_fault_response = first_attempt &&
        scenario == SC_REDUCE_FAULT && env_owner_q == OWNER_REDUCE &&
        env_owner_token_q == 3'd0 && env_owner_group_q == 11'd0;
    wire source_fault_response = first_attempt &&
        scenario == SC_SOURCE_FAULT && env_owner_q == OWNER_SOURCE &&
        env_owner_token_q == 3'd0 && env_owner_group_q == 11'd0;
    wire rd_rsp_error = reduce_fault_response || source_fault_response;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;

    wire [63:0] residual_tdata = 64'h0123_4567_89ab_cdef;
    wire [7:0] residual_tkeep = 8'hff;
    wire residual_tvalid = orphan_inject_now;
    wire residual_tready;
    wire residual_tlast = 1'b1;
    wire residual_fire = residual_tvalid && residual_tready;

    wire [63:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tready = !((scenario == SC_ABORT_OUTPUT) && first_attempt &&
                           !saw_abort_q && cycle_q[1:0] == 2'b01);
    wire m_axis_tlast;
    wire [1:0] m_axis_token;
    wire [8:0] m_axis_block;
    wire output_fire = m_axis_tvalid && m_axis_tready;

    wire [1:0] formal_state;
    wire formal_cfg_fire;
    wire formal_reduce_busy;
    wire formal_reduce_done;
    wire formal_reduce_error;
    wire [12:0] formal_reduce_status;
    wire formal_source_busy;
    wire formal_source_done;
    wire formal_source_error;
    wire [15:0] formal_source_status;
    wire formal_reduce_result_fire;
    wire [1:0] formal_scratch_owner;
    wire formal_reduce_rd_req_fire;
    wire formal_source_rd_req_fire;
    wire formal_rd_rsp_fire;
    wire formal_run_window;
    wire formal_traffic_enable;
    wire formal_internal_fault;
    wire formal_reduce_rd_req_valid_raw;
    wire formal_source_rd_req_valid_raw;
    wire formal_cross_abort;
    wire formal_scalar_cross_abort;
    wire formal_cleanup_abort_issued;
    wire formal_reduce_done_seen;
    wire formal_fault_latched;
    wire [29:0] formal_latched_status;
    wire formal_output_fire;
    wire formal_final_output_fire;
    wire formal_source_internal;
    wire formal_source_completion_mismatch;

    wire abort_reduce_now = first_attempt && !saw_abort_q &&
                            scenario == SC_ABORT_REDUCE &&
                            env_owner_q == OWNER_REDUCE &&
                            response_age_q == 4'd1;
    wire abort_source_now = first_attempt && !saw_abort_q &&
                            scenario == SC_ABORT_SOURCE &&
                            env_owner_q == OWNER_SOURCE &&
                            response_age_q == 4'd1;
    wire abort_output_now = first_attempt && !saw_abort_q &&
                            scenario == SC_ABORT_OUTPUT &&
                            saw_output_stall_q;
    wire abort_run = rst_n &&
                     (abort_reduce_now || abort_source_now || abort_output_now);

    section_rmsnorm_q8_pipeline #(
        .MIN_ROWS(RUN_ROWS),
        .MAX_ROWS(RUN_ROWS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .abort_run(abort_run),
        .gamma_cfg_valid(gamma_cfg_valid),
        .gamma_cfg_ready(gamma_cfg_ready), .gamma_cfg_rows(RUN_ROWS),
        .gamma_tdata({GAMMA_SCALAR, GAMMA_SCALAR}),
        .gamma_tkeep(8'hff), .gamma_tvalid(gamma_tvalid),
        .gamma_tready(gamma_tready), .gamma_tlast(gamma_tlast),
        .gamma_busy(gamma_busy), .gamma_done(gamma_done),
        .gamma_error(gamma_error), .gamma_status(gamma_status),
        .gamma_valid(gamma_valid), .cfg_valid(cfg_valid),
        .cfg_ready(cfg_ready), .cfg_rows(RUN_ROWS),
        .cfg_tokens(RUN_TOKENS), .cfg_eps(RUN_EPS),
        .cfg_resident(scenario == SC_CLEAN), .busy(busy),
        .done(done), .error(error), .status(status),
        .s_axis_tdata(residual_tdata), .s_axis_tkeep(residual_tkeep),
        .s_axis_tvalid(residual_tvalid),
        .s_axis_tready(residual_tready), .s_axis_tlast(residual_tlast),
        .r_wr_valid(r_wr_valid), .r_wr_ready(1'b1), .r_wr_error(1'b0),
        .r_wr_bank(r_wr_bank), .r_wr_address(r_wr_address),
        .r_wr_data(r_wr_data),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group), .rd_rsp_valid(rd_rsp_valid),
        .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(response_group(env_owner_token_q,
                                    env_owner_group_q)),
        .rd_rsp_error(rd_rsp_error), .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast), .m_axis_token(m_axis_token),
        .m_axis_block(m_axis_block), .formal_state(formal_state),
        .formal_cfg_fire(formal_cfg_fire),
        .formal_reduce_busy(formal_reduce_busy),
        .formal_reduce_done(formal_reduce_done),
        .formal_reduce_error(formal_reduce_error),
        .formal_reduce_status(formal_reduce_status),
        .formal_source_busy(formal_source_busy),
        .formal_source_done(formal_source_done),
        .formal_source_error(formal_source_error),
        .formal_source_status(formal_source_status),
        .formal_reduce_result_fire(formal_reduce_result_fire),
        .formal_scratch_owner(formal_scratch_owner),
        .formal_reduce_rd_req_fire(formal_reduce_rd_req_fire),
        .formal_source_rd_req_fire(formal_source_rd_req_fire),
        .formal_rd_rsp_fire(formal_rd_rsp_fire),
        .formal_run_window(formal_run_window),
        .formal_traffic_enable(formal_traffic_enable),
        .formal_internal_fault(formal_internal_fault),
        .formal_reduce_rd_req_valid_raw(
            formal_reduce_rd_req_valid_raw),
        .formal_source_rd_req_valid_raw(
            formal_source_rd_req_valid_raw),
        .formal_cross_abort(formal_cross_abort),
        .formal_scalar_cross_abort(formal_scalar_cross_abort),
        .formal_cleanup_abort_issued(formal_cleanup_abort_issued),
        .formal_reduce_done_seen(formal_reduce_done_seen),
        .formal_fault_latched(formal_fault_latched),
        .formal_latched_status(formal_latched_status),
        .formal_output_fire(formal_output_fire),
        .formal_final_output_fire(formal_final_output_fire),
        .formal_source_internal(formal_source_internal),
        .formal_source_completion_mismatch(
            formal_source_completion_mismatch)
    );

    wire effective_scalar_abort = abort_run || formal_cross_abort ||
                                  formal_scalar_cross_abort;

    // Make the internal-fault quarantine nonvacuous: the orphan arrives in
    // the exact RUN cycle where the reducer's first unowned request is pending.
    assign orphan_inject_now = rst_n && orphan_fault_scenario &&
        first_attempt && !saw_orphan_inject_q &&
        (formal_state == PIPE_RUN) &&
        (formal_scratch_owner == OWNER_NONE) && rd_req_ready &&
        formal_reduce_rd_req_valid_raw &&
        !formal_source_rd_req_valid_raw;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            cycle_q <= 13'd0;
            driver_q <= cfg_reject_scenario ? DR_RUN_CFG : DR_GAMMA_CFG;
            gamma_word_q <= 7'd0;
            cfg_count_q <= 3'd0;
            gamma_load_count_q <= 3'd0;
            clean_done_count_q <= 3'd0;
            error_hold_q <= 3'd0;
            env_owner_q <= OWNER_NONE;
            env_owner_token_q <= 3'd0;
            env_owner_group_q <= 11'd0;
            response_age_q <= 4'd0;
            reduce_request_count_q <= 8'd0;
            source_request_count_q <= 8'd0;
            response_count_q <= 8'd0;
            output_beat_count_q <= 6'd0;
            output_record_count_q <= 3'd0;
            expected_output_beat_q <= 3'd0;
            expected_output_token_q <= 2'd0;
            expected_output_block_q <= 9'd0;
            saw_atomic_start_q <= 1'b0;
            saw_reduce_result_q <= 1'b0;
            saw_reduce_done_q <= 1'b0;
            saw_final_output_q <= 1'b0;
            saw_fault_q <= 1'b0;
            saw_status_before_abort_q <= 1'b0;
            saw_abort_q <= 1'b0;
            saw_owned_abort_q <= 1'b0;
            saw_owned_cleanup_q <= 1'b0;
            saw_abort_response_q <= 1'b0;
            saw_output_stall_q <= 1'b0;
            saw_restart_q <= 1'b0;
            saw_cfg_reject_q <= 1'b0;
            saw_orphan_inject_q <= 1'b0;
            saw_orphan_gamma_invalid_q <= 1'b0;
            captured_status_q <= 30'd0;
        end else begin
            cycle_q <= cycle_q + 1'b1;

            if (gamma_cfg_fire) begin
                driver_q <= DR_GAMMA_DATA;
                gamma_word_q <= 7'd0;
                gamma_load_count_q <= gamma_load_count_q + 1'b1;
            end
            if (gamma_fire) begin
                if (gamma_tlast)
                    driver_q <= DR_GAMMA_WAIT;
                else
                    gamma_word_q <= gamma_word_q + 1'b1;
            end
            if (driver_q == DR_GAMMA_WAIT && gamma_done && !gamma_error)
                driver_q <= DR_RUN_CFG;

            if (cfg_fire) begin
                driver_q <= DR_RUN;
                cfg_count_q <= cfg_count_q + 1'b1;
                reduce_request_count_q <= 8'd0;
                source_request_count_q <= 8'd0;
                response_count_q <= 8'd0;
                output_beat_count_q <= 6'd0;
                output_record_count_q <= 3'd0;
                expected_output_beat_q <= 3'd0;
                expected_output_token_q <= 2'd0;
                expected_output_block_q <= 9'd0;
                saw_reduce_result_q <= 1'b0;
                saw_reduce_done_q <= 1'b0;
                saw_final_output_q <= 1'b0;
                if (cfg_count_q != 0)
                    saw_restart_q <= 1'b1;
            end

            if (formal_cfg_fire)
                saw_atomic_start_q <= 1'b1;
            if (orphan_inject_now)
                saw_orphan_inject_q <= 1'b1;
            if (saw_orphan_inject_q && !gamma_valid)
                saw_orphan_gamma_invalid_q <= 1'b1;
            if (formal_reduce_result_fire)
                saw_reduce_result_q <= 1'b1;
            if (formal_reduce_done && !formal_reduce_error)
                saw_reduce_done_q <= 1'b1;

            if (formal_reduce_rd_req_fire)
                reduce_request_count_q <= reduce_request_count_q + 1'b1;
            if (formal_source_rd_req_fire)
                source_request_count_q <= source_request_count_q + 1'b1;

            if (request_fire) begin
                env_owner_q <= formal_reduce_rd_req_fire ?
                               OWNER_REDUCE : OWNER_SOURCE;
                env_owner_token_q <= rd_req_token;
                env_owner_group_q <= rd_req_group;
                response_age_q <= 4'd0;
            end else if (env_owner_q != OWNER_NONE && !response_fire) begin
                if (response_age_q != 4'hf)
                    response_age_q <= response_age_q + 1'b1;
            end
            if (response_fire) begin
                env_owner_q <= OWNER_NONE;
                response_age_q <= 4'd0;
                response_count_q <= response_count_q + 1'b1;
                if (saw_abort_q)
                    saw_abort_response_q <= 1'b1;
            end

            if (output_fire) begin
                output_beat_count_q <= output_beat_count_q + 1'b1;
                if (m_axis_tlast) begin
                    output_record_count_q <= output_record_count_q + 1'b1;
                    expected_output_beat_q <= 3'd0;
                    if (m_axis_block == 9'd3)
                        saw_final_output_q <= 1'b1;
                    else
                        expected_output_block_q <=
                            expected_output_block_q + 1'b1;
                end else
                    expected_output_beat_q <= expected_output_beat_q + 1'b1;
            end
            if (m_axis_tvalid && !m_axis_tready)
                saw_output_stall_q <= 1'b1;

            if (formal_fault_latched) begin
                saw_fault_q <= 1'b1;
                captured_status_q <= formal_latched_status;
            end
            if (formal_cross_abort && formal_fault_latched &&
                formal_latched_status != 30'd0)
                saw_status_before_abort_q <= 1'b1;

            if (abort_run) begin
                saw_abort_q <= 1'b1;
                saw_owned_abort_q <= env_owner_q != OWNER_NONE;
                driver_q <= DR_GAMMA_CFG;
            end
            if (saw_abort_q && formal_state == PIPE_CLEANUP &&
                env_owner_q != OWNER_NONE && rd_rsp_ready)
                saw_owned_cleanup_q <= 1'b1;

            if (done && error) begin
                if (cfg_reject_scenario && first_attempt)
                    saw_cfg_reject_q <= 1'b1;
                driver_q <= DR_HOLD_ERROR;
                error_hold_q <= 3'd0;
            end else if (driver_q == DR_HOLD_ERROR) begin
                if (error_hold_q == 3'd2)
                    driver_q <= DR_GAMMA_CFG;
                else
                    error_hold_q <= error_hold_q + 1'b1;
            end else if (done && !error) begin
                clean_done_count_q <= clean_done_count_q + 1'b1;
                driver_q <= DR_FINISHED;
            end
        end

        if (rst_n) begin
            assert(formal_state == PIPE_IDLE || formal_state == PIPE_START ||
                   formal_state == PIPE_RUN || formal_state == PIPE_CLEANUP);
            assert(formal_scratch_owner <= OWNER_SOURCE);
            assert(formal_cfg_fire == cfg_fire);
            assert(formal_scratch_owner == env_owner_q);
            assert(formal_output_fire == output_fire);
            assert(formal_rd_rsp_fire == response_fire);
            assert(!(formal_reduce_rd_req_fire &&
                     formal_source_rd_req_fire));
            assert(request_fire == (formal_reduce_rd_req_fire ||
                                    formal_source_rd_req_fire));
            assert(!(done && busy));
            assert(formal_source_status[15] == formal_source_internal);
            if (formal_state != PIPE_IDLE) begin
                assert(!gamma_cfg_ready);
                assert(!gamma_tready);
            end
            if (gamma_cfg_fire || gamma_fire)
                assert(formal_state == PIPE_IDLE);
            if (formal_source_completion_mismatch) begin
                assert(formal_source_internal);
                assert(formal_source_error);
            end
            assert(!(m_axis_tvalid && error));
            assert(reduce_request_count_q <= EXPECT_REDUCE_REQUESTS);
            assert(source_request_count_q <= EXPECT_SOURCE_REQUESTS);
            assert(response_count_q <= EXPECT_RESPONSES);
            assert(output_beat_count_q <= EXPECT_OUTPUT_BEATS);
            assert(output_record_count_q <= EXPECT_RECORDS);

            if (env_owner_q != OWNER_NONE) begin
                assert(!rd_req_valid);
                if (formal_state == PIPE_CLEANUP)
                    assert(rd_rsp_ready);
            end
            if (env_owner_q == OWNER_NONE)
                assert(rd_rsp_ready ==
                       (rd_rsp_valid || effective_scalar_abort));
            if (rd_rsp_ready)
                assert(env_owner_q != OWNER_NONE || orphan_inject_now ||
                       effective_scalar_abort);
            if (effective_scalar_abort)
                assert(rd_rsp_ready);
            if (effective_scalar_abort && rd_rsp_valid)
                assert(response_fire && formal_rd_rsp_fire);
            if (formal_reduce_rd_req_fire)
                assert(env_owner_q == OWNER_NONE && rd_req_role == 2'd0);
            if (formal_source_rd_req_fire)
                assert(env_owner_q == OWNER_NONE && rd_req_role == 2'd0);

            if (f_past_valid &&
                $past(rst_n && formal_reduce_rd_req_fire)) begin
                assert(formal_scratch_owner == OWNER_REDUCE);
            end
            if (f_past_valid &&
                $past(rst_n && formal_source_rd_req_fire)) begin
                assert(formal_scratch_owner == OWNER_SOURCE);
            end
            if (f_past_valid && $past(rst_n &&
                formal_scratch_owner != OWNER_NONE &&
                !formal_rd_rsp_fire)) begin
                assert(formal_scratch_owner ==
                       $past(formal_scratch_owner));
            end

            if (orphan_inject_now) begin
                assert(first_attempt && gamma_valid);
                assert(formal_state == PIPE_RUN);
                assert(formal_scratch_owner == OWNER_NONE);
                assert(formal_run_window && formal_internal_fault);
                assert(!formal_reduce_error && !formal_source_error);
                assert(formal_reduce_rd_req_valid_raw &&
                       !formal_source_rd_req_valid_raw && rd_req_ready);
                assert(rd_rsp_valid && rd_rsp_ready &&
                       formal_rd_rsp_fire);
                assert(!formal_traffic_enable);
                assert(!rd_req_valid && !request_fire);
                assert(!formal_reduce_rd_req_fire &&
                       !formal_source_rd_req_fire);
                assert(residual_tvalid && !residual_tready &&
                       !residual_fire);
                assert(!r_wr_valid);
                assert(!formal_reduce_result_fire);
                assert(!m_axis_tvalid && !formal_output_fire);
            end
            if (f_past_valid &&
                $past(rst_n && orphan_inject_now)) begin
                assert(formal_state == PIPE_CLEANUP);
                assert(formal_fault_latched && error);
                assert(formal_latched_status == STATUS_INTERNAL);
                assert(status == STATUS_INTERNAL);
                assert(formal_cross_abort);
                assert(!formal_cleanup_abort_issued);
                assert(formal_scratch_owner == OWNER_NONE);
            end
            if (orphan_fault_scenario && first_attempt &&
                saw_orphan_inject_q && formal_cleanup_abort_issued) begin
                assert(!formal_cross_abort);
                assert(!gamma_valid);
            end

            if (f_past_valid && $past(rst_n && cfg_fire)) begin
                assert(formal_state == PIPE_START);
                assert(busy);
                if (!formal_reduce_error && !formal_source_error) begin
                    assert(formal_reduce_busy);
                    assert(formal_source_busy);
                end
                assert(!formal_cross_abort);
                assert(formal_scratch_owner == OWNER_NONE);
                if (cfg_reject_scenario && cfg_count_q == 3'd1)
                    assert(!gamma_valid);
            end
            if (f_past_valid &&
                $past(rst_n && formal_state == PIPE_START &&
                      !abort_run && !formal_reduce_error &&
                      !formal_source_error)) begin
                assert(formal_state == PIPE_RUN);
            end
            if (formal_state == PIPE_START) begin
                assert(busy);
                if (!formal_reduce_error && !formal_source_error) begin
                    assert(formal_reduce_busy);
                    assert(formal_source_busy);
                end
                assert(formal_scratch_owner == OWNER_NONE);
                assert(!rd_req_valid);
                assert(!r_wr_valid);
                assert(!formal_reduce_result_fire);
                assert(!m_axis_tvalid);
            end

            if (formal_reduce_result_fire) begin
                assert(formal_state == PIPE_RUN);
                assert(formal_reduce_busy);
                assert(formal_source_busy);
                assert(!formal_cross_abort);
            end
            if (formal_reduce_done_seen) begin
                assert(saw_reduce_done_q || formal_reduce_done);
                assert(!formal_reduce_error);
            end

            if (m_axis_tvalid) begin
                assert(formal_state == PIPE_RUN);
                assert(!abort_run);
                assert(!formal_cross_abort);
                assert(!formal_fault_latched);
                assert(!formal_reduce_error);
                assert(!formal_source_error);
                assert(m_axis_token == expected_output_token_q);
                assert(m_axis_block == expected_output_block_q);
                assert(m_axis_tlast == (expected_output_beat_q == 3'd4));
            end
            if (formal_state == PIPE_CLEANUP || formal_cross_abort ||
                formal_fault_latched || abort_run ||
                formal_reduce_error || formal_source_error) begin
                assert(!m_axis_tvalid);
            end
            assert(formal_final_output_fire ==
                   (output_fire && m_axis_tlast &&
                    m_axis_token == 2'd0 && m_axis_block == 9'd3));

            if (formal_fault_latched) begin
                assert(formal_latched_status != 30'd0);
                assert(error);
                assert(status == formal_latched_status);
                assert(formal_state == PIPE_CLEANUP || (done && !busy));
                if (formal_state == PIPE_CLEANUP &&
                    !formal_cleanup_abort_issued)
                    assert(formal_cross_abort);
            end
            if (formal_cross_abort && formal_fault_latched)
                assert(formal_latched_status == captured_status_q ||
                       !saw_fault_q);
            if (cfg_reject_scenario && first_attempt &&
                formal_cross_abort && formal_fault_latched) begin
                assert(formal_latched_status == STATUS_SOURCE_GAMMA);
                assert(formal_source_status == 16'h0002);
                assert(formal_reduce_busy);
            end
            if (orphan_fault_scenario && first_attempt &&
                formal_fault_latched) begin
                assert(formal_latched_status == STATUS_INTERNAL);
                assert(status == STATUS_INTERNAL);
                assert(!formal_source_status[15]);
            end
            if (formal_fault_latched && formal_latched_status[28])
                assert(formal_source_status[15]);
            if (formal_cleanup_abort_issued && formal_fault_latched) begin
                assert(!formal_cross_abort);
                assert(formal_latched_status != 30'd0);
            end
            if (f_past_valid && formal_cleanup_abort_issued &&
                !$past(formal_cleanup_abort_issued)) begin
                assert($past(formal_cross_abort) || $past(abort_run));
            end

            if (abort_run) begin
                assert(!m_axis_tvalid);
                assert(!done);
            end
            if (f_past_valid && $past(rst_n && abort_run)) begin
                assert(formal_state == PIPE_CLEANUP);
                assert(!formal_fault_latched);
                assert(formal_latched_status == 30'd0);
                assert(formal_cleanup_abort_issued);
                assert(!formal_cross_abort);
                assert(!done && !error && status == 30'd0);
            end
            if (saw_abort_q && cfg_count_q == 3'd1) begin
                assert(!done);
                assert(!error);
                assert(status == 30'd0);
                assert(!m_axis_tvalid);
            end
            if (cfg_reject_scenario && first_attempt &&
                !saw_cfg_reject_q) begin
                assert(reduce_request_count_q == 8'd0);
                assert(source_request_count_q == 8'd0);
                assert(response_count_q == 8'd0);
                assert(output_beat_count_q == 6'd0);
                assert(!rd_req_valid);
                assert(!r_wr_valid);
                assert(!formal_reduce_result_fire);
                assert(!m_axis_tvalid);
            end

            if (done && error) begin
                assert(!busy);
                assert(saw_status_before_abort_q);
                if (scenario == SC_REDUCE_FAULT)
                    assert(status == STATUS_REDUCE_SCRATCH);
                else if (scenario == SC_SOURCE_FAULT)
                    assert(status == STATUS_SOURCE_SCRATCH);
                else if (scenario == SC_CFG_REJECT) begin
                    assert(status == STATUS_SOURCE_GAMMA);
                    assert(reduce_request_count_q == 8'd0);
                    assert(source_request_count_q == 8'd0);
                    assert(response_count_q == 8'd0);
                    assert(output_beat_count_q == 6'd0);
                end
                else if (scenario == SC_ORPHAN_FAULT) begin
                    assert(status == STATUS_INTERNAL);
                    assert(saw_orphan_inject_q);
                    assert(saw_orphan_gamma_invalid_q);
                    assert(reduce_request_count_q == 8'd0);
                    assert(source_request_count_q == 8'd0);
                    assert(response_count_q == 8'd1);
                    assert(output_beat_count_q == 6'd0);
                end
                else
                    assert(1'b0);
            end
            if (done && !error) begin
                assert(!busy && status == 30'd0);
                assert(saw_reduce_result_q);
                assert(saw_reduce_done_q);
                assert(saw_final_output_q);
                assert(reduce_request_count_q == EXPECT_REDUCE_REQUESTS);
                assert(source_request_count_q == EXPECT_SOURCE_REQUESTS);
                assert(response_count_q == EXPECT_RESPONSES);
                assert(output_beat_count_q == EXPECT_OUTPUT_BEATS);
                assert(output_record_count_q == EXPECT_RECORDS);
            end
            if (f_past_valid && $past(rst_n && done))
                assert(!done);

            if (f_past_valid && !abort_run &&
                $past(rst_n && !abort_run && m_axis_tvalid &&
                      !m_axis_tready && !formal_reduce_error &&
                      !formal_source_error)) begin
                if (!formal_reduce_error && !formal_source_error &&
                    !formal_cross_abort) begin
                    assert(m_axis_tvalid);
                    assert(m_axis_tdata == $past(m_axis_tdata));
                    assert(m_axis_tlast == $past(m_axis_tlast));
                    assert(m_axis_token == $past(m_axis_token));
                    assert(m_axis_block == $past(m_axis_block));
                end
            end

            if (scenario == SC_CLEAN || scenario == SC_STARTUP)
                assert(!saw_abort_q && !saw_fault_q);
            if (restart_scenario && cfg_count_q >= 3'd2)
                assert(gamma_load_count_q >=
                       (cfg_reject_scenario ? 3'd1 : 3'd2) && saw_restart_q);
        end

`ifdef FORMAL_COVER
`ifdef FORMAL_SC_STARTUP
        cover(rst_n && saw_atomic_start_q && formal_state == PIPE_START &&
              formal_reduce_busy && formal_source_busy);
`elsif FORMAL_SC_CLEAN
        cover(rst_n && driver_q == DR_FINISHED &&
              clean_done_count_q == 3'd1 && saw_final_output_q);
`elsif FORMAL_SC_REDUCE_FAULT
        cover(rst_n && driver_q == DR_FINISHED && saw_fault_q &&
              saw_status_before_abort_q && saw_restart_q &&
              clean_done_count_q == 3'd1);
`elsif FORMAL_SC_SOURCE_FAULT
        cover(rst_n && driver_q == DR_FINISHED && saw_fault_q &&
              saw_status_before_abort_q && saw_restart_q &&
              clean_done_count_q == 3'd1);
`elsif FORMAL_SC_ABORT_REDUCE
        cover(rst_n && driver_q == DR_FINISHED && saw_owned_abort_q &&
              saw_owned_cleanup_q && saw_abort_response_q && saw_restart_q &&
              clean_done_count_q == 3'd1);
`elsif FORMAL_SC_ABORT_SOURCE
        cover(rst_n && driver_q == DR_FINISHED && saw_owned_abort_q &&
              saw_owned_cleanup_q && saw_abort_response_q && saw_restart_q &&
              clean_done_count_q == 3'd1);
`elsif FORMAL_SC_ABORT_OUTPUT
        cover(rst_n && driver_q == DR_FINISHED && saw_output_stall_q &&
              saw_abort_q && saw_restart_q && clean_done_count_q == 3'd1);
`elsif FORMAL_SC_CFG_REJECT
        cover(rst_n && driver_q == DR_FINISHED && saw_cfg_reject_q &&
              saw_status_before_abort_q && saw_restart_q &&
              gamma_load_count_q == 3'd1 && clean_done_count_q == 3'd1);
`else
        cover(rst_n && driver_q == DR_FINISHED &&
              saw_orphan_inject_q && saw_status_before_abort_q &&
              saw_orphan_gamma_invalid_q && saw_restart_q &&
              gamma_load_count_q == 3'd2 && clean_done_count_q == 3'd1);
`endif
`endif
    end

    wire _unused = &{1'b0, gamma_busy, gamma_status, gamma_valid,
                     formal_reduce_status, formal_source_status,
                     captured_status_q, expected_output_token_q,
                     residual_tdata, residual_tkeep, residual_tlast};
endmodule

`default_nettype wire
