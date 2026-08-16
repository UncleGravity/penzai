`default_nettype none

module section_rmsnorm_q8_source_formal(input wire clk);
    localparam [3:0] SC_STARTUP       = 4'd0;
    localparam [3:0] SC_CLEAN1        = 4'd1;
    localparam [3:0] SC_CLEAN2        = 4'd2;
    localparam [3:0] SC_ELASTIC       = 4'd3;
    localparam [3:0] SC_REUSE         = 4'd4;
    localparam [3:0] SC_SOURCE_FAULT  = 4'd5;
    localparam [3:0] SC_Q8_FAULT      = 4'd6;
    localparam [3:0] SC_RESTART_SOURCE = 4'd7;
    localparam [3:0] SC_RESTART_Q8    = 4'd8;
    localparam [3:0] SC_ABORT_OWNER   = 4'd9;
    localparam [3:0] SC_ABORT_DRAIN   = 4'd10;
    localparam [3:0] SC_ABORT_PRIORITY = 4'd11;

    localparam [2:0] DR_GAMMA_CFG  = 3'd0;
    localparam [2:0] DR_GAMMA_DATA = 3'd1;
    localparam [2:0] DR_GAMMA_WAIT = 3'd2;
    localparam [2:0] DR_RUN_CFG     = 3'd3;
    localparam [2:0] DR_RUN         = 3'd4;
    localparam [2:0] DR_HOLD_ERROR  = 3'd5;
    localparam [2:0] DR_FINISHED    = 3'd6;

    localparam [1:0] WR_IDLE = 2'd0, WR_RUN = 2'd1;
    localparam [1:0] WR_Q8_DRAIN = 2'd2, WR_CLEANUP = 2'd3;
    localparam [3:0] WS_IDLE = 4'd0, WS_INV = 4'd2;
    localparam [3:0] Q8_IDLE = 4'd0, Q8_EMIT = 4'd5;
    localparam [3:0] Q8_ERROR = 4'd7, Q8_INTERNAL = 4'd8;

    localparam [13:0] RUN_ROWS = 14'd128;
    localparam [31:0] GAMMA_SCALAR = 32'h3f00_0000;
    localparam [31:0] QUANT_FAULT_WORD = 32'hdead_c0de;
    localparam [15:0] STATUS_SCRATCH = 16'h0008;
    localparam [15:0] STATUS_Q8 = 16'h0400;

`ifdef FORMAL_SC_STARTUP
    wire [3:0] scenario = SC_STARTUP;
`elsif FORMAL_SC_CLEAN1
    wire [3:0] scenario = SC_CLEAN1;
`elsif FORMAL_SC_CLEAN2
    wire [3:0] scenario = SC_CLEAN2;
`elsif FORMAL_SC_ELASTIC
    wire [3:0] scenario = SC_ELASTIC;
`elsif FORMAL_SC_REUSE
    wire [3:0] scenario = SC_REUSE;
`elsif FORMAL_SC_SOURCE_FAULT
    wire [3:0] scenario = SC_SOURCE_FAULT;
`elsif FORMAL_SC_Q8_FAULT
    wire [3:0] scenario = SC_Q8_FAULT;
`elsif FORMAL_SC_RESTART_SOURCE
    wire [3:0] scenario = SC_RESTART_SOURCE;
`elsif FORMAL_SC_RESTART_Q8
    wire [3:0] scenario = SC_RESTART_Q8;
`elsif FORMAL_SC_ABORT_OWNER
    wire [3:0] scenario = SC_ABORT_OWNER;
`elsif FORMAL_SC_ABORT_DRAIN
    wire [3:0] scenario = SC_ABORT_DRAIN;
`else
    wire [3:0] scenario = SC_ABORT_PRIORITY;
`endif

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;
    reg [12:0] cycle_q = 13'd0;
    reg [2:0] driver_q = DR_GAMMA_CFG;
    reg [6:0] gamma_word_q = 7'd0;
    reg [1:0] inverse_count_q = 2'd0;
    reg [2:0] cfg_count_q = 3'd0;
    reg [2:0] gamma_load_count_q = 3'd0;
    reg [2:0] clean_done_count_q = 3'd0;
    reg [2:0] hold_age_q = 3'd0;

    reg owner_q = 1'b0;
    reg response_offer_q = 1'b0;
    reg [3:0] response_age_q = 4'd0;
    reg [2:0] owner_token_q = 3'd0;
    reg [10:0] owner_group_q = 11'd0;

    reg [2:0] expected_req_token_q = 3'd0;
    reg [4:0] expected_req_group_q = 5'd0;
    reg [9:0] request_count_q = 10'd0;
    reg [9:0] response_count_q = 10'd0;
    reg [9:0] source_scalar_count_q = 10'd0;
    reg [9:0] quant_scalar_count_q = 10'd0;
    reg [8:0] record_count_q = 9'd0;
    reg [12:0] output_beat_count_q = 13'd0;
    reg [8:0] tlast_count_q = 9'd0;

    reg saw_atomic_start_q = 1'b0;
    reg saw_source_done_q = 1'b0;
    reg saw_q8_drain_q = 1'b0;
    reg saw_output_stall_q = 1'b0;
    reg saw_request_stall_q = 1'b0;
    reg saw_fault_q = 1'b0;
    reg saw_source_fault_q = 1'b0;
    reg saw_q8_fault_q = 1'b0;
    reg saw_abort_q = 1'b0;
    reg saw_abort_owned_q = 1'b0;
    reg saw_abort_owner_drain_q = 1'b0;
    reg saw_abort_response_q = 1'b0;
    reg saw_abort_drain_state_q = 1'b0;
    reg saw_abort_priority_q = 1'b0;
    reg saw_restart_q = 1'b0;
    reg [15:0] sticky_status_q = 16'd0;

    wire [2:0] run_tokens = scenario == SC_CLEAN2 ? 3'd2 : 3'd1;
    wire [9:0] expected_scalars = run_tokens == 3'd2 ? 10'd256 : 10'd128;
    wire [9:0] expected_requests = run_tokens == 3'd2 ? 10'd32 : 10'd16;
    wire [8:0] expected_records = run_tokens == 3'd2 ? 9'd8 : 9'd4;
    wire [12:0] expected_beats = run_tokens == 3'd2 ? 13'd40 : 13'd20;
    wire first_attempt = cfg_count_q == 3'd1;
    wire source_fault_scenario = (scenario == SC_SOURCE_FAULT) ||
                                 (scenario == SC_RESTART_SOURCE);
    wire q8_fault_scenario = (scenario == SC_Q8_FAULT) ||
                             (scenario == SC_RESTART_Q8) ||
                             (scenario == SC_ABORT_PRIORITY);
    wire restart_fault_scenario = (scenario == SC_RESTART_SOURCE) ||
                                  (scenario == SC_RESTART_Q8);
    wire restart_abort_scenario = (scenario == SC_ABORT_OWNER) ||
                                  (scenario == SC_ABORT_DRAIN) ||
                                  (scenario == SC_ABORT_PRIORITY);

    function automatic [31:0] inverse_scalar(input [1:0] token);
        inverse_scalar = 32'h3f80_0100 | {30'd0, token};
    endfunction

    function automatic [31:0] swap16(input [31:0] value);
        swap16 = {value[15:0], value[31:16]};
    endfunction

    function automatic [31:0] residual_scalar(
        input [2:0] token,
        input [10:0] group_index,
        input [2:0] lane
    );
        residual_scalar = 32'h4000_0000 |
                          {15'd0, token, group_index[6:0], lane};
    endfunction

    function automatic [31:0] selected_residual(
        input [2:0] token,
        input [10:0] group_index,
        input [2:0] lane
    );
        begin
            if (first_attempt && q8_fault_scenario && token == 3'd0 &&
                group_index == 11'd0 && lane == 3'd0)
                selected_residual = QUANT_FAULT_WORD ^
                                    swap16(inverse_scalar(token[1:0])) ^
                                    swap16(GAMMA_SCALAR);
            else
                selected_residual = residual_scalar(token, group_index, lane);
        end
    endfunction

    function automatic [255:0] response_group(
        input [2:0] token,
        input [10:0] group_index
    );
        integer lane;
        begin
            for (lane = 0; lane < 8; lane = lane + 1)
                response_group[lane*32 +: 32] =
                    selected_residual(token, group_index, lane[2:0]);
        end
    endfunction

    function automatic [63:0] record_beat(
        input [2:0] beat,
        input [8:0] record_index
    );
        begin
            case (beat)
                3'd0: record_beat = 64'h1100_0000_0000_0000 |
                                      {55'd0, record_index};
                3'd1: record_beat = 64'h2200_0000_0000_0000 |
                                      {55'd0, record_index};
                3'd2: record_beat = 64'h3300_0000_0000_0000 |
                                      {55'd0, record_index};
                default: record_beat = 64'h4400_0000_0000_0000 |
                                       {55'd0, record_index};
            endcase
        end
    endfunction

    function automatic [63:0] native_data(
        input [2:0] beat,
        input [8:0] record_index
    );
        begin
            if (beat == 3'd4)
                native_data = {48'd0, (16'h5000 | {7'd0, record_index})};
            else
                native_data = record_beat(beat, record_index);
        end
    endfunction

    wire gamma_cfg_valid = rst_n && driver_q == DR_GAMMA_CFG;
    wire gamma_cfg_ready;
    wire gamma_tvalid = rst_n && driver_q == DR_GAMMA_DATA;
    wire gamma_tready;
    wire [63:0] gamma_tdata = {GAMMA_SCALAR, GAMMA_SCALAR};
    wire gamma_tlast = gamma_word_q == 7'd63;
    wire gamma_busy, gamma_done, gamma_error, gamma_valid;
    wire [3:0] gamma_status;
    wire gamma_cfg_fire = gamma_cfg_valid && gamma_cfg_ready;
    wire gamma_fire = gamma_tvalid && gamma_tready;

    wire cfg_valid = rst_n && driver_q == DR_RUN_CFG;
    wire cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;
    wire busy, done, error;
    wire [15:0] status;

    wire inv_valid = rst_n && driver_q == DR_RUN &&
                     formal_weighted_state == WS_INV;
    wire inv_ready;
    wire [1:0] inv_token = inverse_count_q;
    wire [31:0] inv_rms = inverse_scalar(inverse_count_q);
    wire inv_final = {1'b0, inverse_count_q} + 1'b1 == run_tokens;
    wire inv_fire = inv_valid && inv_ready;

    wire rd_req_valid;
    wire rd_req_ready = rst_n && !owner_q &&
        (scenario != SC_ELASTIC || cycle_q[1:0] != 2'b00);
    wire [1:0] rd_req_role;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_valid = owner_q && response_offer_q;
    wire rd_rsp_ready;
    wire [255:0] rd_rsp_data = response_group(owner_token_q, owner_group_q);
    wire rd_rsp_error = first_attempt && source_fault_scenario &&
                        owner_token_q == 3'd0 && owner_group_q == 11'd0;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;

    wire [63:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tready = scenario == SC_ELASTIC ?
                         (cycle_q[2:0] != 3'b011) : 1'b1;
    wire m_axis_tlast;
    wire [1:0] m_axis_token;
    wire [8:0] m_axis_block;
    wire output_fire = m_axis_tvalid && m_axis_tready;

    wire [1:0] formal_state;
    wire formal_cfg_fire;
    wire [3:0] formal_weighted_state;
    wire formal_weighted_read_owned;
    wire [1:0] formal_weighted_mul_phase;
    wire formal_weighted_scalar_fire;
    wire formal_q8_start, formal_q8_abort;
    wire [3:0] formal_q8_state;
    wire formal_q8_activation_fault, formal_q8_record_done;
    wire [5:0] formal_q8_scalar_index;
    wire [2:0] formal_q8_emit_index;
    wire formal_q8_staged_valid, formal_q8_quant_input_fire;
    wire formal_q8_quant_output_valid;
    wire [2:0] formal_beat_index;
    wire [1:0] formal_token;
    wire [8:0] formal_block;
    wire formal_source_done_seen, formal_fault_latched;
    wire formal_cleanup_abort_issued;
    wire [15:0] formal_latched_status;

    wire abort_owner_now = scenario == SC_ABORT_OWNER && first_attempt &&
                           owner_q && response_age_q == 4'd1;
    wire abort_drain_now = scenario == SC_ABORT_DRAIN && first_attempt &&
                           formal_state == WR_Q8_DRAIN;
    wire abort_priority_now = scenario == SC_ABORT_PRIORITY && first_attempt &&
                              formal_q8_state == Q8_ERROR;
    wire abort_run = rst_n && !saw_abort_q &&
                     (abort_owner_now || abort_drain_now || abort_priority_now);

    section_rmsnorm_q8_source #(
        .MIN_ROWS(RUN_ROWS), .MAX_ROWS(RUN_ROWS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .abort_run(abort_run),
        .gamma_cfg_valid(gamma_cfg_valid),
        .gamma_cfg_ready(gamma_cfg_ready), .gamma_cfg_rows(RUN_ROWS),
        .gamma_tdata(gamma_tdata), .gamma_tkeep(8'hff),
        .gamma_tvalid(gamma_tvalid), .gamma_tready(gamma_tready),
        .gamma_tlast(gamma_tlast), .gamma_busy(gamma_busy),
        .gamma_done(gamma_done), .gamma_error(gamma_error),
        .gamma_status(gamma_status), .gamma_valid(gamma_valid),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_rows(RUN_ROWS),
        .cfg_tokens(run_tokens), .busy(busy), .done(done), .error(error),
        .status(status), .inv_valid(inv_valid), .inv_ready(inv_ready),
        .inv_token(inv_token), .inv_rms(inv_rms), .inv_final(inv_final),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group), .rd_rsp_valid(rd_rsp_valid),
        .rd_rsp_ready(rd_rsp_ready), .rd_rsp_data(rd_rsp_data),
        .rd_rsp_error(rd_rsp_error), .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast), .m_axis_token(m_axis_token),
        .m_axis_block(m_axis_block), .formal_state(formal_state),
        .formal_cfg_fire(formal_cfg_fire),
        .formal_weighted_state(formal_weighted_state),
        .formal_weighted_read_owned(formal_weighted_read_owned),
        .formal_weighted_mul_phase(formal_weighted_mul_phase),
        .formal_weighted_scalar_fire(formal_weighted_scalar_fire),
        .formal_q8_start(formal_q8_start),
        .formal_q8_abort(formal_q8_abort),
        .formal_q8_state(formal_q8_state),
        .formal_q8_activation_fault(formal_q8_activation_fault),
        .formal_q8_record_done(formal_q8_record_done),
        .formal_q8_scalar_index(formal_q8_scalar_index),
        .formal_q8_emit_index(formal_q8_emit_index),
        .formal_q8_staged_valid(formal_q8_staged_valid),
        .formal_q8_quant_input_fire(formal_q8_quant_input_fire),
        .formal_q8_quant_output_valid(formal_q8_quant_output_valid),
        .formal_beat_index(formal_beat_index),
        .formal_token(formal_token), .formal_block(formal_block),
        .formal_source_done_seen(formal_source_done_seen),
        .formal_fault_latched(formal_fault_latched),
        .formal_cleanup_abort_issued(formal_cleanup_abort_issued),
        .formal_latched_status(formal_latched_status)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!rst_n) begin
            cycle_q <= 13'd0;
            driver_q <= DR_GAMMA_CFG;
            gamma_word_q <= 7'd0;
            inverse_count_q <= 2'd0;
            cfg_count_q <= 3'd0;
            gamma_load_count_q <= 3'd0;
            clean_done_count_q <= 3'd0;
            hold_age_q <= 3'd0;
            owner_q <= 1'b0;
            response_offer_q <= 1'b0;
            response_age_q <= 4'd0;
            expected_req_token_q <= 3'd0;
            expected_req_group_q <= 5'd0;
            request_count_q <= 10'd0;
            response_count_q <= 10'd0;
            source_scalar_count_q <= 10'd0;
            quant_scalar_count_q <= 10'd0;
            record_count_q <= 9'd0;
            output_beat_count_q <= 13'd0;
            tlast_count_q <= 9'd0;
            saw_atomic_start_q <= 1'b0;
            saw_source_done_q <= 1'b0;
            saw_q8_drain_q <= 1'b0;
            saw_output_stall_q <= 1'b0;
            saw_request_stall_q <= 1'b0;
            saw_fault_q <= 1'b0;
            saw_source_fault_q <= 1'b0;
            saw_q8_fault_q <= 1'b0;
            saw_abort_q <= 1'b0;
            saw_abort_owned_q <= 1'b0;
            saw_abort_owner_drain_q <= 1'b0;
            saw_abort_response_q <= 1'b0;
            saw_abort_drain_state_q <= 1'b0;
            saw_abort_priority_q <= 1'b0;
            saw_restart_q <= 1'b0;
            sticky_status_q <= 16'd0;
        end else begin
            cycle_q <= cycle_q + 1'b1;

            if (gamma_cfg_fire) begin
                driver_q <= DR_GAMMA_DATA;
                gamma_word_q <= 7'd0;
                gamma_load_count_q <= gamma_load_count_q + 1'b1;
            end
            if (gamma_fire) begin
                if (gamma_word_q == 7'd63)
                    driver_q <= DR_GAMMA_WAIT;
                else
                    gamma_word_q <= gamma_word_q + 1'b1;
            end
            if (driver_q == DR_GAMMA_WAIT && gamma_done && !gamma_error)
                driver_q <= DR_RUN_CFG;

            if (cfg_fire) begin
                driver_q <= DR_RUN;
                cfg_count_q <= cfg_count_q + 1'b1;
                inverse_count_q <= 2'd0;
                expected_req_token_q <= 3'd0;
                expected_req_group_q <= 5'd0;
                request_count_q <= 10'd0;
                response_count_q <= 10'd0;
                source_scalar_count_q <= 10'd0;
                quant_scalar_count_q <= 10'd0;
                record_count_q <= 9'd0;
                output_beat_count_q <= 13'd0;
                tlast_count_q <= 9'd0;
                saw_source_done_q <= 1'b0;
                saw_q8_drain_q <= 1'b0;
                saw_output_stall_q <= 1'b0;
                saw_request_stall_q <= 1'b0;
                if (cfg_count_q != 0)
                    saw_restart_q <= 1'b1;
            end
            if (inv_fire)
                inverse_count_q <= inverse_count_q + 1'b1;

            if (request_fire) begin
                owner_q <= 1'b1;
                response_offer_q <= 1'b0;
                response_age_q <= 4'd0;
                owner_token_q <= rd_req_token;
                owner_group_q <= rd_req_group;
                request_count_q <= request_count_q + 1'b1;
                if (expected_req_group_q == 5'd15) begin
                    expected_req_group_q <= 5'd0;
                    expected_req_token_q <= expected_req_token_q + 1'b1;
                end else begin
                    expected_req_group_q <= expected_req_group_q + 1'b1;
                end
            end else if (owner_q && !response_fire) begin
                if (response_age_q != 4'hf)
                    response_age_q <= response_age_q + 1'b1;
                if (!response_offer_q &&
                    response_age_q >= (scenario == SC_ELASTIC ? 4'd4 : 4'd2))
                    response_offer_q <= 1'b1;
            end
            if (response_fire) begin
                owner_q <= 1'b0;
                response_offer_q <= 1'b0;
                response_age_q <= 4'd0;
                response_count_q <= response_count_q + 1'b1;
                if (saw_abort_q)
                    saw_abort_response_q <= 1'b1;
            end

            if (formal_weighted_scalar_fire)
                source_scalar_count_q <= source_scalar_count_q + 1'b1;
            if (formal_q8_quant_input_fire)
                quant_scalar_count_q <= quant_scalar_count_q + 1'b1;
            if (output_fire) begin
                output_beat_count_q <= output_beat_count_q + 1'b1;
                if (m_axis_tlast)
                    tlast_count_q <= tlast_count_q + 1'b1;
                if (formal_beat_index == 3'd4)
                    record_count_q <= record_count_q + 1'b1;
            end

            if (formal_q8_start)
                saw_atomic_start_q <= 1'b1;
            if (formal_source_done_seen)
                saw_source_done_q <= 1'b1;
            if (formal_state == WR_Q8_DRAIN)
                saw_q8_drain_q <= 1'b1;
            if (m_axis_tvalid && !m_axis_tready)
                saw_output_stall_q <= 1'b1;
            if (rd_req_valid && !rd_req_ready)
                saw_request_stall_q <= 1'b1;

            if (formal_q8_activation_fault)
                saw_q8_fault_q <= 1'b1;
            if (formal_fault_latched) begin
                saw_fault_q <= 1'b1;
                sticky_status_q <= formal_latched_status;
                if (formal_latched_status == STATUS_SCRATCH)
                    saw_source_fault_q <= 1'b1;
                if (formal_latched_status == STATUS_Q8)
                    saw_q8_fault_q <= 1'b1;
            end

            if (abort_run) begin
                saw_abort_q <= 1'b1;
                driver_q <= DR_GAMMA_CFG;
                if (owner_q)
                    saw_abort_owned_q <= 1'b1;
                if (formal_state == WR_Q8_DRAIN)
                    saw_abort_drain_state_q <= 1'b1;
                if (formal_q8_state == Q8_ERROR)
                    saw_abort_priority_q <= 1'b1;
            end
            if (saw_abort_q && owner_q && formal_state == WR_CLEANUP)
                saw_abort_owner_drain_q <= 1'b1;

            if (done && error) begin
                saw_fault_q <= 1'b1;
                sticky_status_q <= status;
                hold_age_q <= 3'd0;
                driver_q <= DR_HOLD_ERROR;
            end else if (done && !error) begin
                clean_done_count_q <= clean_done_count_q + 1'b1;
                if (scenario == SC_REUSE && cfg_count_q == 3'd1)
                    driver_q <= DR_RUN_CFG;
                else
                    driver_q <= DR_FINISHED;
            end else if (driver_q == DR_HOLD_ERROR) begin
                if (hold_age_q == 3'd3) begin
                    if (restart_fault_scenario)
                        driver_q <= DR_GAMMA_CFG;
                    else
                        driver_q <= DR_FINISHED;
                end else begin
                    hold_age_q <= hold_age_q + 1'b1;
                end
            end
        end

        if (rst_n) begin
            assert(formal_state <= WR_CLEANUP);
            assert(formal_weighted_state <= 4'd9);
            assert(formal_q8_state <= Q8_INTERNAL);
            assert(formal_cfg_fire == cfg_fire);
            assert(formal_q8_start == formal_cfg_fire);
            assert(formal_weighted_read_owned == owner_q);
            assert(!(rd_req_valid && owner_q));
            if (rd_rsp_ready)
                assert(owner_q);

            if (rd_req_valid) begin
                assert(rd_req_role == 2'd0);
                assert(rd_req_token == expected_req_token_q);
                assert(rd_req_group == {6'd0, expected_req_group_q});
            end
            if (f_past_valid && $past(rst_n && rd_req_valid && !rd_req_ready)) begin
                assert(rd_req_valid);
                assert(rd_req_role == $past(rd_req_role));
                assert(rd_req_token == $past(rd_req_token));
                assert(rd_req_group == $past(rd_req_group));
            end
            if (f_past_valid && $past(rst_n && rd_rsp_valid && !rd_rsp_ready)) begin
                assert(rd_rsp_valid);
                assert(rd_rsp_data == $past(rd_rsp_data));
                assert(rd_rsp_error == $past(rd_rsp_error));
            end

            if (f_past_valid && $past(rst_n && cfg_fire)) begin
                assert(formal_state == WR_RUN);
                assert(formal_weighted_state == WS_INV);
                assert(formal_q8_state == Q8_INTERNAL);
                assert(busy);
            end
            if (formal_source_done_seen) begin
                assert(formal_state == WR_Q8_DRAIN);
                assert(source_scalar_count_q == expected_scalars);
                assert(!done && busy);
            end
            if (formal_state == WR_Q8_DRAIN) begin
                assert(formal_source_done_seen);
                assert(saw_source_done_q || formal_source_done_seen);
            end

            if ((formal_state == WR_RUN || formal_state == WR_Q8_DRAIN) &&
                !formal_q8_abort && !formal_q8_activation_fault) begin
                assert(source_scalar_count_q == quant_scalar_count_q +
                       (formal_q8_staged_valid ? 1'b1 : 1'b0));
                assert(formal_q8_scalar_index ==
                       {1'b0, quant_scalar_count_q[4:0]});
                assert(record_count_q <= quant_scalar_count_q[9:5]);
            end

            if (m_axis_tvalid) begin
                assert(formal_state == WR_RUN ||
                       formal_state == WR_Q8_DRAIN);
                assert(formal_q8_state == Q8_EMIT);
                assert(formal_beat_index == formal_q8_emit_index);
                assert(formal_token == m_axis_token);
                assert(formal_block == m_axis_block);
                assert(m_axis_token == record_count_q[3:2]);
                assert(m_axis_block == {7'd0, record_count_q[1:0]});
                assert(m_axis_tlast == (formal_beat_index == 3'd4));
                assert(m_axis_tdata == native_data(formal_beat_index,
                                                   record_count_q));
            end
            assert(formal_q8_record_done ==
                   (output_fire && formal_beat_index == 3'd4));
            if (f_past_valid && !abort_run &&
                $past(rst_n && m_axis_tvalid && !m_axis_tready && !abort_run)) begin
                assert(m_axis_tvalid);
                assert(m_axis_tdata == $past(m_axis_tdata));
                assert(m_axis_tlast == $past(m_axis_tlast));
                assert(m_axis_token == $past(m_axis_token));
                assert(m_axis_block == $past(m_axis_block));
            end

            if (formal_q8_activation_fault && !abort_run) begin
                assert(!m_axis_tvalid);
                assert(formal_q8_state == Q8_ERROR);
            end
            if (formal_fault_latched) begin
                assert(formal_latched_status != 16'd0);
                assert(error);
                assert(status == formal_latched_status);
                assert(formal_state == WR_CLEANUP || (done && !busy));
                if (formal_state == WR_CLEANUP)
                    assert(formal_q8_abort);
            end
            if (formal_cleanup_abort_issued && formal_fault_latched) begin
                assert(formal_q8_abort);
                assert(formal_latched_status == sticky_status_q);
            end
            if (f_past_valid && formal_cleanup_abort_issued &&
                !$past(formal_cleanup_abort_issued) && formal_fault_latched) begin
                assert($past(formal_fault_latched));
                assert($past(formal_latched_status) != 16'd0);
            end

            if (abort_run) begin
                assert(formal_q8_abort);
                assert(!m_axis_tvalid && !done);
            end
            if (f_past_valid && $past(rst_n && abort_run)) begin
                assert(formal_state == WR_CLEANUP);
                assert(!formal_fault_latched);
                assert(formal_latched_status == 16'd0);
                assert(!done && !error && status == 16'd0);
                assert(!gamma_valid);
            end

            if (done && !error) begin
                assert(!busy && status == 16'd0);
                assert(saw_source_done_q && saw_q8_drain_q);
                assert(source_scalar_count_q == expected_scalars);
                assert(quant_scalar_count_q == expected_scalars);
                assert(request_count_q == expected_requests);
                assert(response_count_q == expected_requests);
                assert(record_count_q == expected_records);
                assert(output_beat_count_q == expected_beats);
                assert(tlast_count_q == expected_records);
            end
            if (done && error) begin
                assert(!busy && status != 16'd0);
                if (source_fault_scenario && first_attempt) begin
                    assert(status == STATUS_SCRATCH);
                    assert(source_scalar_count_q == 10'd0);
                    assert(record_count_q == 9'd0);
                end
                if (q8_fault_scenario && first_attempt &&
                    scenario != SC_ABORT_PRIORITY) begin
                    assert(status == STATUS_Q8);
                    assert(source_scalar_count_q == 10'd32);
                    assert(quant_scalar_count_q == 10'd32);
                    assert(record_count_q == 9'd0);
                end
            end
            if (error)
                assert(status != 16'd0);
            if (f_past_valid && $past(rst_n && error) &&
                !$past(abort_run) && !abort_run &&
                !$past(cfg_fire) && !cfg_fire) begin
                assert(error);
                assert(status == $past(status));
            end
            if (f_past_valid && $past(rst_n && done))
                assert(!done);

            if (scenario == SC_REUSE && cfg_count_q >= 3'd2)
                assert(gamma_load_count_q == 3'd1);
            if (restart_fault_scenario && cfg_count_q >= 3'd2)
                assert(gamma_load_count_q == 3'd2);
            if (restart_abort_scenario && cfg_count_q >= 3'd2)
                assert(gamma_load_count_q == 3'd2);
        end

`ifdef FORMAL_COVER
`ifdef FORMAL_SC_STARTUP
        cover(rst_n && saw_atomic_start_q && formal_state == WR_RUN &&
              formal_weighted_state == WS_INV && formal_q8_state == Q8_INTERNAL);
`elsif FORMAL_SC_CLEAN1
        cover(rst_n && driver_q == DR_FINISHED && clean_done_count_q == 3'd1);
`elsif FORMAL_SC_CLEAN2
        cover(rst_n && driver_q == DR_FINISHED && clean_done_count_q == 3'd1 &&
              record_count_q == 9'd8);
`elsif FORMAL_SC_ELASTIC
        cover(rst_n && driver_q == DR_FINISHED && clean_done_count_q == 3'd1 &&
              saw_output_stall_q && saw_request_stall_q);
`elsif FORMAL_SC_REUSE
        cover(rst_n && driver_q == DR_FINISHED && clean_done_count_q == 3'd2 &&
              cfg_count_q == 3'd2 && gamma_load_count_q == 3'd1);
`elsif FORMAL_SC_SOURCE_FAULT
        cover(rst_n && driver_q == DR_FINISHED && saw_source_fault_q &&
              status == STATUS_SCRATCH);
`elsif FORMAL_SC_Q8_FAULT
        cover(rst_n && driver_q == DR_FINISHED && saw_q8_fault_q &&
              status == STATUS_Q8);
`elsif FORMAL_SC_RESTART_SOURCE
        cover(rst_n && driver_q == DR_FINISHED && saw_source_fault_q &&
              saw_restart_q && clean_done_count_q == 3'd1);
`elsif FORMAL_SC_RESTART_Q8
        cover(rst_n && driver_q == DR_FINISHED && saw_q8_fault_q &&
              saw_restart_q && clean_done_count_q == 3'd1);
`elsif FORMAL_SC_ABORT_OWNER
        cover(rst_n && driver_q == DR_FINISHED && saw_abort_owned_q &&
              saw_abort_owner_drain_q && saw_abort_response_q && saw_restart_q &&
              clean_done_count_q == 3'd1);
`elsif FORMAL_SC_ABORT_DRAIN
        cover(rst_n && driver_q == DR_FINISHED && saw_abort_drain_state_q &&
              saw_restart_q && clean_done_count_q == 3'd1);
`else
        cover(rst_n && driver_q == DR_FINISHED && saw_abort_priority_q &&
              saw_restart_q && clean_done_count_q == 3'd1 && !error);
`endif
`endif
    end

    wire _unused = &{1'b0, gamma_busy, gamma_status,
                     formal_weighted_mul_phase,
                     formal_q8_quant_output_valid, WS_IDLE, Q8_IDLE};
endmodule

`default_nettype wire
