`default_nettype none

module section_rmsnorm_weighted_source_formal(input wire clk);
    localparam [3:0] ST_IDLE = 0, ST_GAMMA = 1, ST_INV = 2, ST_REQ = 3;
    localparam [3:0] ST_WAIT_RSP = 4, ST_MUL1_REQ = 5, ST_MUL1_WAIT = 6;
    localparam [3:0] ST_MUL2_REQ = 7, ST_MUL2_WAIT = 8, ST_CLEANUP = 9;

    localparam [3:0] SC_CLEAN = 0, SC_REUSE = 1, SC_GAMMA_BAD = 2;
    localparam [3:0] SC_GAMMA_FRAME = 3, SC_GAMMA_NONFINITE = 4;
    localparam [3:0] SC_MISMATCH = 5, SC_RUN_BAD = 6;
    localparam [3:0] SC_INV_ORDER = 7, SC_INV_VALUE = 8;
    localparam [3:0] SC_SCRATCH = 9, SC_MUL1_EARLY = 10;
    localparam [3:0] SC_MUL1_LATE = 11, SC_MUL2_LATE = 12;
    localparam [3:0] SC_ABORT = 13, SC_PRIORITY = 14;
    localparam [3:0] SC_BAD_REPLACE = 15;

    localparam [3:0] GAMMA_BAD_CFG = 4'h1, GAMMA_FRAME = 4'h2;
    localparam [3:0] GAMMA_NONFINITE = 4'h4;
    localparam [8:0] STATUS_BAD_CFG = 9'h001, STATUS_GAMMA = 9'h002;
    localparam [8:0] STATUS_INV_FRAME = 9'h004, STATUS_SCRATCH = 9'h008;
    localparam [8:0] STATUS_MUL1_NONFINITE = 9'h010;
    localparam [8:0] STATUS_MUL2_OVERFLOW = 9'h080;

    (* anyconst *) reg [3:0] scenario_any;
    (* anyconst *) reg [2:0] abort_phase_any;
    (* anyseq *) reg request_ready_any;
    (* anyseq *) reg scalar_ready_any;
    (* anyseq *) reg response_launch_any;

`ifdef FORMAL_CLEAN
    wire [3:0] scenario = SC_CLEAN;
`elsif FORMAL_REUSE
    wire [3:0] scenario = SC_REUSE;
`elsif FORMAL_RESTART_SCRATCH
    wire [3:0] scenario = SC_SCRATCH;
`elsif FORMAL_RESTART_MUL1
    wire [3:0] scenario = SC_MUL1_LATE;
`elsif FORMAL_RESTART_MUL2
    wire [3:0] scenario = SC_MUL2_LATE;
`elsif FORMAL_RESTART_ABORT
    wire [3:0] scenario = SC_ABORT;
`elsif FORMAL_RESTART
    wire [3:0] scenario = SC_GAMMA_FRAME;
`else
    wire [3:0] scenario = scenario_any;
`endif

`ifdef FORMAL_ABORT_PHASE_0
    wire [2:0] abort_phase = 0;
`elsif FORMAL_ABORT_PHASE_1
    wire [2:0] abort_phase = 1;
`elsif FORMAL_ABORT_PHASE_2
    wire [2:0] abort_phase = 2;
`elsif FORMAL_ABORT_PHASE_3
    wire [2:0] abort_phase = 3;
`elsif FORMAL_ABORT_PHASE_4
    wire [2:0] abort_phase = 4;
`elsif FORMAL_ABORT_PHASE_5
    wire [2:0] abort_phase = 5;
`elsif FORMAL_ABORT_PHASE_6
    wire [2:0] abort_phase = 6;
`else
    wire [2:0] abort_phase = abort_phase_any;
`endif

`ifdef FORMAL_STRUCTURAL
    wire request_allow = request_ready_any;
    wire scalar_allow = scalar_ready_any;
    wire response_launch = response_launch_any;
`else
    wire request_allow = 1'b1;
    wire scalar_allow = 1'b1;
    wire response_launch = 1'b1;
`endif

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;
    reg want_gamma_q = 1'b1;
    reg want_run_q = 1'b0;
    reg recovery_q = 1'b0;
    reg recovery_pending_q = 1'b0;
    reg [2:0] sticky_age_q = 0;
    reg [1:0] run_ordinal_q = 0;
    reg completed_q = 1'b0;
    reg fault_seen_q = 1'b0;
    reg rejected_run_q = 1'b0;
    reg abort_seen_q = 1'b0;
    reg restarted_q = 1'b0;
    reg saw_reuse_q = 1'b0;
    reg [2:0] abort_phase_seen_q = 0;

    reg [5:0] gamma_word_q = 0;
    reg [2:0] gamma_epoch_q = 0;
    reg [2:0] loaded_epoch_q = 0;
    reg [13:0] loaded_rows_q = 0;
    reg model_gamma_valid_q = 0;
    reg [2:0] inverse_count_q = 0;

    reg owner_q = 0;
    reg response_offer_q = 0;
    reg [3:0] response_age_q = 0;
    reg [2:0] owner_token_q = 0;
    reg [10:0] owner_group_q = 0;
    reg [8:0] request_count_q = 0;
    reg [8:0] response_count_q = 0;
    reg [7:0] scalar_count_q = 0;
    reg [2:0] expected_req_token_q = 0;
    reg [10:0] expected_req_group_q = 0;
    reg [31:0] expected_mul1_q = 0;
    reg [31:0] expected_mul2_q = 0;
    reg saw_copy_release_q = 0;
    reg saw_scalar_stall_q = 0;
    reg saw_abort_scalar_stall_q = 0;
    reg saw_abort_drain_q = 0;
    reg saw_abort_response_fire_q = 0;
    reg saw_late_prefix_q = 0;
    reg replacement_q = 0;
    reg saw_prior_seal_q = 0;

    wire initial_attempt = !recovery_q;
    wire [13:0] gamma_rows = initial_attempt && scenario == SC_GAMMA_BAD ?
                             14'd16 : 14'd32;
    wire [13:0] run_rows = initial_attempt && scenario == SC_MISMATCH ?
                           14'd64 :
                           (initial_attempt && scenario == SC_RUN_BAD ?
                            14'd16 : 14'd32);
    wire [2:0] run_tokens = (!recovery_q && scenario == SC_CLEAN) ?
                            3'd2 : 3'd1;
    wire [5:0] gamma_words = gamma_rows[6:1];

    function automatic [31:0] gamma_scalar(
        input [2:0] epoch, input [6:0] row
    );
        gamma_scalar = 32'h3f00_0000 | {18'd0, epoch, row[6:0]};
    endfunction
    function automatic [31:0] inverse_scalar(
        input [2:0] token, input [1:0] epoch
    );
        inverse_scalar = 32'h3f80_0100 | {27'd0, epoch, token};
    endfunction
    function automatic [31:0] residual_scalar(
        input [2:0] token, input [10:0] group_index, input [2:0] lane
    );
        residual_scalar = 32'h4000_0000 |
                          {15'd0, token, group_index[6:0], lane};
    endfunction
    function automatic [31:0] mul_model(
        input [31:0] a, input [31:0] b
    );
        mul_model = a ^ {b[15:0], b[31:16]} ^ 32'h6d2b_79f5;
    endfunction
    function automatic [255:0] response_group(
        input [2:0] token, input [10:0] group_index
    );
        integer lane;
        begin
            for (lane = 0; lane < 8; lane = lane + 1)
                response_group[lane*32 +: 32] =
                    residual_scalar(token, group_index, lane[2:0]);
        end
    endfunction

    wire [6:0] gamma_row_lo = {gamma_word_q, 1'b0};
    wire [31:0] gamma_lo_base = gamma_scalar(gamma_epoch_q, gamma_row_lo);
    wire [31:0] gamma_hi_base = gamma_scalar(
        gamma_epoch_q, gamma_row_lo + 1'b1);
    wire mul2_fault_word = initial_attempt && scenario == SC_MUL2_LATE &&
                           gamma_word_q == 0;
    wire [31:0] gamma_lo = gamma_lo_base;
    wire [31:0] gamma_hi = mul2_fault_word ? 32'h7f7f_ffff : gamma_hi_base;
    wire bad_replacement_word = initial_attempt &&
                                scenario == SC_BAD_REPLACE && replacement_q;
    wire gamma_cfg_valid = rst_n && want_gamma_q;
    wire gamma_cfg_ready;
    wire gamma_tvalid;
    wire gamma_tready;
    wire gamma_tlast = initial_attempt && scenario == SC_GAMMA_FRAME ?
                       (gamma_word_q == 0) :
                       (gamma_word_q + 1'b1 == gamma_words);
    wire [7:0] gamma_tkeep = initial_attempt &&
        (scenario == SC_GAMMA_FRAME || bad_replacement_word) ?
        8'h0f : 8'hff;
    wire [63:0] gamma_tdata = initial_attempt &&
                              scenario == SC_GAMMA_NONFINITE &&
                              gamma_word_q == 0 ?
                              {gamma_hi, 32'h7f80_0000} :
                              {gamma_hi, gamma_lo};
    wire gamma_busy, gamma_done, gamma_error, gamma_valid;
    wire [3:0] gamma_status;
    assign gamma_tvalid = rst_n && gamma_busy;

    wire cfg_valid = rst_n && (want_run_q ||
        (initial_attempt && scenario == SC_PRIORITY && want_gamma_q));
    wire cfg_ready, busy, done, error;
    wire [8:0] status;
    wire inv_valid = rst_n && formal_state == ST_INV;
    wire inv_ready;
    wire [1:0] inv_token = initial_attempt && scenario == SC_INV_ORDER ?
                           inverse_count_q[1:0] + 1'b1 :
                           inverse_count_q[1:0];
    wire [1:0] inverse_epoch = recovery_q ? 2'd2 : run_ordinal_q;
    wire [31:0] inv_rms = initial_attempt && scenario == SC_INV_VALUE ?
                          32'd0 : inverse_scalar(inverse_count_q,
                                                 inverse_epoch);
    wire inv_final = inverse_count_q + 1'b1 == run_tokens;

    wire rd_req_valid, rd_req_ready;
    wire [1:0] rd_req_role;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_valid = owner_q && response_offer_q;
    wire rd_rsp_ready;
    wire [255:0] rd_rsp_base = response_group(owner_token_q, owner_group_q);
    // Inject diagnosed MUL1 operands into the actual retained scratch response,
    // not only into the expected-data scoreboard.
    wire [255:0] rd_rsp_data = initial_attempt &&
        scenario == SC_MUL1_EARLY && owner_token_q == 0 &&
        owner_group_q == 0 ? {rd_rsp_base[255:32], 32'h7f80_0000} :
        (initial_attempt && scenario == SC_MUL1_LATE &&
         owner_token_q == 0 && owner_group_q == 0 ?
         {rd_rsp_base[255:64], 32'h7f80_0000, rd_rsp_base[31:0]} :
         rd_rsp_base);
    wire rd_rsp_error = initial_attempt && scenario == SC_SCRATCH;
    assign rd_req_ready = request_allow && !owner_q;

    wire scalar_valid;
    wire scalar_ready = initial_attempt && scenario == SC_ABORT &&
                        abort_phase == 3'd5 ? 1'b0 : scalar_allow;
    wire [31:0] scalar_data;
    wire scalar_last;
    wire [1:0] scalar_status;

    wire [3:0] formal_state;
    wire [13:0] formal_gamma_rows;
    wire [2:0] formal_scale_count;
    wire formal_read_owned, formal_rsp_buffered;
    wire [1:0] formal_run_token;
    wire [10:0] formal_run_group;
    wire [2:0] formal_run_lane;
    wire [1:0] formal_mul_phase;
    wire formal_mul_s_fire;
    wire [31:0] formal_mul_s_a, formal_mul_s_b;
    wire formal_mul_result_fire;
    wire [1:0] formal_mul_result_status;
    wire [31:0] formal_mul_result_data;

    wire gamma_cfg_fire = gamma_cfg_valid && gamma_cfg_ready;
    wire gamma_fire = gamma_tvalid && gamma_tready;
    wire cfg_fire = cfg_valid && cfg_ready;
    wire inv_fire = inv_valid && inv_ready;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire scalar_fire = scalar_valid && scalar_ready;

    wire abort_gamma = gamma_busy && gamma_word_q == 2;
    wire abort_inverse = formal_state == ST_INV && inverse_count_q == 0;
    wire abort_owner = owner_q && response_age_q == 1;
    wire abort_mul1 = formal_state == ST_MUL1_WAIT;
    wire abort_mul2 = formal_state == ST_MUL2_WAIT &&
                      !saw_scalar_stall_q;
    wire abort_scalar = saw_scalar_stall_q;
    wire abort_response = owner_q && response_offer_q;
    wire selected_abort = abort_phase == 0 ? abort_gamma :
                          abort_phase == 1 ? abort_inverse :
                          abort_phase == 2 ? abort_owner :
                          abort_phase == 3 ? abort_mul1 :
                          abort_phase == 4 ? abort_mul2 :
                          abort_phase == 5 ? abort_scalar :
                                                 abort_response;
    wire abort_run = rst_n && initial_attempt && scenario == SC_ABORT &&
                     !abort_seen_q && selected_abort;

    wire [3:0] expected_gamma_fault =
        scenario == SC_GAMMA_BAD ? GAMMA_BAD_CFG :
        (scenario == SC_GAMMA_FRAME ? GAMMA_FRAME :
        (scenario == SC_GAMMA_NONFINITE ? GAMMA_NONFINITE :
         (scenario == SC_BAD_REPLACE ? GAMMA_FRAME : 4'd0)));
    wire [8:0] expected_run_fault =
        scenario == SC_MISMATCH ? STATUS_GAMMA :
        (scenario == SC_RUN_BAD ? STATUS_BAD_CFG :
         ((scenario == SC_INV_ORDER || scenario == SC_INV_VALUE) ?
          STATUS_INV_FRAME :
          (scenario == SC_SCRATCH ? STATUS_SCRATCH :
           ((scenario == SC_MUL1_EARLY || scenario == SC_MUL1_LATE) ?
            STATUS_MUL1_NONFINITE :
            (scenario == SC_MUL2_LATE ? STATUS_MUL2_OVERFLOW : 9'd0)))));

    section_rmsnorm_weighted_source #(.MIN_ROWS(32), .MAX_ROWS(64)) dut (
        .clk(clk), .rst_n(rst_n), .abort_run(abort_run),
        .gamma_cfg_valid(gamma_cfg_valid), .gamma_cfg_ready(gamma_cfg_ready),
        .gamma_cfg_rows(gamma_rows), .gamma_tdata(gamma_tdata),
        .gamma_tkeep(gamma_tkeep), .gamma_tvalid(gamma_tvalid),
        .gamma_tready(gamma_tready), .gamma_tlast(gamma_tlast),
        .gamma_busy(gamma_busy), .gamma_done(gamma_done),
        .gamma_error(gamma_error), .gamma_status(gamma_status),
        .gamma_valid(gamma_valid), .cfg_valid(cfg_valid),
        .cfg_ready(cfg_ready), .cfg_rows(run_rows),
        .cfg_tokens(run_tokens), .busy(busy), .done(done),
        .error(error), .status(status), .inv_valid(inv_valid),
        .inv_ready(inv_ready), .inv_token(inv_token), .inv_rms(inv_rms),
        .inv_final(inv_final), .rd_req_valid(rd_req_valid),
        .rd_req_ready(rd_req_ready), .rd_req_role(rd_req_role),
        .rd_req_token(rd_req_token), .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error),
        .scalar_valid(scalar_valid), .scalar_ready(scalar_ready),
        .scalar_data(scalar_data), .scalar_last(scalar_last),
        .scalar_status(scalar_status), .formal_state(formal_state),
        .formal_gamma_rows(formal_gamma_rows),
        .formal_scale_count(formal_scale_count),
        .formal_read_owned(formal_read_owned),
        .formal_rsp_buffered(formal_rsp_buffered),
        .formal_run_token(formal_run_token),
        .formal_run_group(formal_run_group),
        .formal_run_lane(formal_run_lane),
        .formal_mul_phase(formal_mul_phase),
        .formal_mul_s_fire(formal_mul_s_fire),
        .formal_mul_s_a(formal_mul_s_a), .formal_mul_s_b(formal_mul_s_b),
        .formal_mul_result_fire(formal_mul_result_fire),
        .formal_mul_result_status(formal_mul_result_status),
        .formal_mul_result_data(formal_mul_result_data)
    );

    wire [10:0] last_group = (run_rows >> 3) - 1'b1;
    wire [6:0] active_row = {formal_run_group[3:0], formal_run_lane};
    wire [31:0] expected_gamma =
        (initial_attempt && scenario == SC_MUL2_LATE && active_row == 1) ?
        32'h7f7f_ffff : gamma_scalar(loaded_epoch_q, active_row);
    wire [31:0] base_residual = residual_scalar(
        {1'b0, formal_run_token}, formal_run_group, formal_run_lane);
    wire mul1_fault_now = initial_attempt &&
        ((scenario == SC_MUL1_EARLY && active_row == 0) ||
         (scenario == SC_MUL1_LATE && active_row == 1));
    wire [31:0] expected_residual = mul1_fault_now ?
                                    32'h7f80_0000 : base_residual;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        assume(scenario <= SC_BAD_REPLACE);
        if (scenario == SC_ABORT)
            assume(abort_phase <= 6);
`ifdef FORMAL_FAULTS
        assume(scenario >= SC_GAMMA_BAD && scenario <= SC_BAD_REPLACE);
`endif
`ifdef FORMAL_FAULT_COVER
        assume(scenario >= SC_GAMMA_BAD && scenario <= SC_BAD_REPLACE);
`endif

        if (!rst_n) begin
            want_gamma_q <= 1'b1;
            want_run_q <= 1'b0;
            recovery_q <= 1'b0;
            recovery_pending_q <= 1'b0;
            sticky_age_q <= 0;
            run_ordinal_q <= 0;
            completed_q <= 0;
            fault_seen_q <= 0;
            rejected_run_q <= 0;
            abort_seen_q <= 0;
            restarted_q <= 0;
            saw_reuse_q <= 0;
            gamma_word_q <= 0;
            gamma_epoch_q <= 0;
            loaded_epoch_q <= 0;
            loaded_rows_q <= 0;
            model_gamma_valid_q <= 0;
            inverse_count_q <= 0;
            owner_q <= 0;
            response_offer_q <= 0;
            response_age_q <= 0;
            request_count_q <= 0;
            response_count_q <= 0;
            scalar_count_q <= 0;
            expected_req_token_q <= 0;
            expected_req_group_q <= 0;
            expected_mul1_q <= 0;
            expected_mul2_q <= 0;
            saw_copy_release_q <= 0;
            saw_scalar_stall_q <= 0;
            saw_abort_scalar_stall_q <= 0;
            saw_abort_drain_q <= 0;
            saw_abort_response_fire_q <= 0;
            saw_late_prefix_q <= 0;
            replacement_q <= 0;
            saw_prior_seal_q <= 0;
        end else begin
            if (gamma_cfg_fire) begin
                want_gamma_q <= 0;
                gamma_word_q <= 0;
                gamma_epoch_q <= gamma_epoch_q + 1'b1;
                model_gamma_valid_q <= 0;
            end
            if (gamma_fire) begin
                gamma_word_q <= gamma_word_q + 1'b1;
                if (gamma_word_q + 1'b1 == gamma_words && gamma_tlast &&
                    gamma_tkeep == 8'hff &&
                    gamma_tdata[30:23] != 8'hff &&
                    gamma_tdata[62:55] != 8'hff) begin
                    model_gamma_valid_q <= 1;
                    loaded_epoch_q <= gamma_epoch_q;
                    loaded_rows_q <= gamma_rows;
                end
            end
            if (gamma_done && !gamma_error) begin
                if (initial_attempt && scenario == SC_BAD_REPLACE &&
                    !replacement_q) begin
                    want_run_q <= 0;
                    want_gamma_q <= 1;
                    replacement_q <= 1;
                    saw_prior_seal_q <= 1;
                end else begin
                    want_run_q <= 1;
                end
            end

            if (cfg_fire) begin
                want_run_q <= 0;
                inverse_count_q <= 0;
                request_count_q <= 0;
                response_count_q <= 0;
                scalar_count_q <= 0;
                expected_req_token_q <= 0;
                expected_req_group_q <= 0;
                saw_scalar_stall_q <= 0;
                if (recovery_q)
                    restarted_q <= 1;
                if ((!model_gamma_valid_q) || loaded_rows_q != run_rows ||
                    run_rows < 32 || run_rows > 64)
                    model_gamma_valid_q <= 0;
            end
            if (inv_fire)
                inverse_count_q <= inverse_count_q + 1'b1;

            if (request_fire) begin
                owner_q <= 1;
                response_offer_q <= 0;
                response_age_q <= 0;
                owner_token_q <= rd_req_token;
                owner_group_q <= rd_req_group;
                request_count_q <= request_count_q + 1'b1;
                if (expected_req_group_q == last_group) begin
                    expected_req_group_q <= 0;
                    expected_req_token_q <= expected_req_token_q + 1'b1;
                end else begin
                    expected_req_group_q <= expected_req_group_q + 1'b1;
                end
            end else if (owner_q && !response_fire) begin
                if (response_age_q != 15)
                    response_age_q <= response_age_q + 1'b1;
                if (!response_offer_q && response_age_q >= 2 &&
                    response_launch)
                    response_offer_q <= 1;
            end
            if (response_fire) begin
                owner_q <= 0;
                response_offer_q <= 0;
                response_age_q <= 0;
                response_count_q <= response_count_q + 1'b1;
            end
            if (scalar_fire) begin
                scalar_count_q <= scalar_count_q + 1'b1;
                saw_late_prefix_q <= 1;
            end
            if (scalar_valid && !scalar_ready)
                saw_scalar_stall_q <= 1;

            if (formal_mul_s_fire && formal_mul_phase == 1)
                expected_mul1_q <= mul_model(expected_residual,
                                             inverse_scalar(
                                                 {1'b0, formal_run_token},
                                                 inverse_epoch));
            if (formal_mul_s_fire && formal_mul_phase == 2)
                expected_mul2_q <= mul_model(expected_mul1_q,
                                             expected_gamma);

            if ((gamma_done && gamma_error) || (done && error)) begin
                fault_seen_q <= 1;
                sticky_age_q <= 0;
                model_gamma_valid_q <= 0;
`ifdef FORMAL_RESTART
                if (initial_attempt && gamma_done && gamma_error) begin
                    want_run_q <= 1;
                    recovery_pending_q <= 0;
                end else begin
                    if (initial_attempt && done && error)
                        rejected_run_q <= 1;
                    recovery_pending_q <= 1;
                end
`else
                recovery_pending_q <= 1;
`endif
            end
            if (error && !fault_seen_q)
                model_gamma_valid_q <= 0;
            if (abort_run) begin
                abort_seen_q <= 1;
                abort_phase_seen_q <= abort_phase;
                if (abort_phase == 5 && saw_scalar_stall_q)
                    saw_abort_scalar_stall_q <= 1;
                recovery_pending_q <= 1;
                sticky_age_q <= 0;
                model_gamma_valid_q <= 0;
                if (response_fire)
                    saw_abort_response_fire_q <= 1;
            end
            if (recovery_pending_q) begin
                if (sticky_age_q != 3)
                    sticky_age_q <= sticky_age_q + 1'b1;
                else if (!busy && !gamma_busy && formal_state == ST_IDLE) begin
                    recovery_pending_q <= 0;
                    recovery_q <= 1;
                    want_gamma_q <= 1;
                end
            end
            if (abort_seen_q && !restarted_q && formal_state == ST_CLEANUP &&
                owner_q)
                saw_abort_drain_q <= 1;

            if (done && !error) begin
                if (!recovery_q && scenario == SC_REUSE &&
                    run_ordinal_q == 0) begin
                    run_ordinal_q <= 1;
                    want_run_q <= 1;
                    saw_reuse_q <= 1;
                end else begin
                    completed_q <= 1;
                end
            end
            if (f_past_valid && $past(response_fire && !rd_rsp_error) &&
                formal_rsp_buffered && !formal_read_owned)
                saw_copy_release_q <= 1;
        end

        if (rst_n) begin
            assert(!(done && busy));
            assert(!(gamma_done && gamma_busy));
            assert(!(scalar_valid && error));
            assert(gamma_valid == model_gamma_valid_q ||
                   (error && !gamma_valid));
            assert(formal_read_owned == owner_q);
            assert(!(rd_req_valid && owner_q));
            if (rd_rsp_ready)
                assert(owner_q);
            if (rd_req_valid) begin
                assert(rd_req_role == 0);
                assert(rd_req_token == expected_req_token_q);
                assert(rd_req_group == expected_req_group_q);
            end
            if (formal_mul_s_fire && formal_mul_phase == 1) begin
                assert(formal_mul_s_a == expected_residual);
                assert(formal_mul_s_b == inverse_scalar(
                    {1'b0, formal_run_token}, inverse_epoch));
            end
            if (formal_mul_s_fire && formal_mul_phase == 2) begin
                assert(formal_mul_s_a == expected_mul1_q);
                assert(formal_mul_s_b == expected_gamma);
            end
            if (formal_mul_result_fire && formal_mul_phase == 1) begin
                assert(formal_mul_result_data == expected_mul1_q);
                assert(formal_mul_result_status ==
                    (mul1_fault_now ? 2'd1 : 2'd0));
            end
            if (formal_mul_result_fire && formal_mul_phase == 2) begin
                assert(formal_mul_result_data == expected_mul2_q);
                assert(formal_mul_result_status ==
                    (initial_attempt && scenario == SC_MUL2_LATE &&
                     active_row == 1 ? 2'd2 : 2'd0));
            end
            if (scalar_valid) begin
                assert(formal_mul_phase == 2);
                assert(formal_mul_result_status == 0);
                assert(scalar_data == expected_mul2_q);
                assert(scalar_status == 0);
                assert(scalar_last == (formal_run_lane == 7 &&
                                       formal_run_group[1:0] == 3));
            end
            if (formal_state == ST_GAMMA)
                assert(!gamma_valid);
            if (abort_seen_q && recovery_pending_q) begin
                assert(!gamma_valid && !scalar_valid);
                assert(!done && !error && status == 0);
            end
            if (fault_seen_q && recovery_pending_q)
                assert(!gamma_valid && !scalar_valid);
            if (done && !error) begin
                assert(status == 0);
                assert(scalar_count_q == run_rows * run_tokens);
                assert(request_count_q == (run_rows >> 3) * run_tokens);
                assert(response_count_q == request_count_q);
            end
            if (done && error)
                assert(!gamma_valid && !scalar_valid);
            if (initial_attempt && done && error) begin
                if (scenario == SC_MUL1_LATE || scenario == SC_MUL2_LATE)
                    assert(scalar_count_q == 1);
                else if (scenario >= SC_MISMATCH &&
                         scenario <= SC_MUL1_EARLY)
                    assert(scalar_count_q == 0);
            end
            if (gamma_done && gamma_error)
                assert(!gamma_valid);
            if (initial_attempt && gamma_done && gamma_error &&
                expected_gamma_fault != 0)
                assert(gamma_status == expected_gamma_fault);
            if (initial_attempt && done && error &&
                expected_run_fault != 0)
                assert(status == expected_run_fault);
`ifdef FORMAL_RESTART
            if (initial_attempt && gamma_done && gamma_error) begin
                assert(gamma_status == GAMMA_FRAME);
                assert(!gamma_valid);
            end
            if (initial_attempt && cfg_fire) begin
                assert(fault_seen_q);
                assert(!gamma_valid);
            end
            if (initial_attempt && done && error) begin
                assert(status == STATUS_GAMMA);
                assert(inverse_count_q == 0);
                assert(request_count_q == 0);
                assert(response_count_q == 0);
                assert(scalar_count_q == 0);
            end
            if (recovery_pending_q || recovery_q)
                assert(rejected_run_q);
            if (recovery_q && gamma_done && !gamma_error) begin
                assert(gamma_word_q == gamma_words);
                assert(model_gamma_valid_q);
                assert(loaded_epoch_q == gamma_epoch_q);
            end
            if (completed_q) begin
                assert(fault_seen_q && rejected_run_q && restarted_q);
                assert(gamma_epoch_q == 2 && loaded_epoch_q == 2);
            end
`endif
            if (fault_seen_q && recovery_pending_q && !abort_run) begin
                if (error)
                    assert(status != 0);
                if (gamma_error)
                    assert(gamma_status != 0);
            end
            if (f_past_valid &&
                $past(rst_n && gamma_error && !abort_run &&
                      !gamma_cfg_fire)) begin
                assert(gamma_error);
                assert(gamma_status == $past(gamma_status));
            end
            if (f_past_valid &&
                $past(rst_n && error && !abort_run && !cfg_fire)) begin
                assert(error);
                assert(status == $past(status));
            end
            if (gamma_cfg_valid && cfg_valid && formal_state == ST_IDLE &&
                !abort_run) begin
                assert(gamma_cfg_fire);
                assert(!cfg_fire);
            end
            if (initial_attempt && scenario == SC_BAD_REPLACE &&
                replacement_q && gamma_cfg_fire) begin
                assert(saw_prior_seal_q);
                assert(gamma_valid);
            end
            if (!abort_run && f_past_valid &&
                $past(scalar_valid && !scalar_ready && !abort_run)) begin
                assert(scalar_valid);
                assert(scalar_data == $past(scalar_data));
                assert(scalar_last == $past(scalar_last));
            end
        end

`ifdef FORMAL_COVER
`ifdef FORMAL_CLEAN
        cover(rst_n && scenario == SC_CLEAN && completed_q &&
              saw_copy_release_q);
`elsif FORMAL_REUSE
        cover(rst_n && scenario == SC_REUSE && completed_q && saw_reuse_q);
`elsif FORMAL_FAULT_COVER
        cover(rst_n && scenario == SC_GAMMA_BAD && gamma_done && gamma_error &&
              gamma_status == GAMMA_BAD_CFG);
        cover(rst_n && scenario == SC_GAMMA_FRAME && gamma_done &&
              gamma_error && gamma_status == GAMMA_FRAME);
        cover(rst_n && scenario == SC_GAMMA_NONFINITE && gamma_done &&
              gamma_error && gamma_status == GAMMA_NONFINITE);
        cover(rst_n && scenario == SC_BAD_REPLACE && saw_prior_seal_q &&
              gamma_done && gamma_error && gamma_status == GAMMA_FRAME);
        cover(rst_n && scenario == SC_MISMATCH && done && error &&
              status == STATUS_GAMMA);
        cover(rst_n && scenario == SC_RUN_BAD && done && error &&
              status == STATUS_BAD_CFG);
        cover(rst_n && scenario == SC_INV_ORDER && done && error &&
              status == STATUS_INV_FRAME);
        cover(rst_n && scenario == SC_INV_VALUE && done && error &&
              status == STATUS_INV_FRAME);
        cover(rst_n && scenario == SC_SCRATCH && done && error &&
              status == STATUS_SCRATCH);
        cover(rst_n && scenario == SC_MUL1_EARLY && done && error &&
              status == STATUS_MUL1_NONFINITE && scalar_count_q == 0);
        cover(rst_n && scenario == SC_MUL1_LATE && done && error &&
              status == STATUS_MUL1_NONFINITE && scalar_count_q == 1);
        cover(rst_n && scenario == SC_MUL2_LATE && done && error &&
              status == STATUS_MUL2_OVERFLOW && scalar_count_q == 1);
        cover(rst_n && scenario == SC_PRIORITY && gamma_cfg_fire && cfg_valid &&
              !cfg_fire);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 0 &&
              abort_seen_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 1 &&
              abort_seen_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 2 &&
              abort_seen_q && saw_abort_drain_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 3 &&
              abort_seen_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 4 &&
              abort_seen_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 5 &&
              abort_seen_q && saw_scalar_stall_q);
        cover(rst_n && scenario == SC_ABORT && abort_phase == 6 &&
              abort_seen_q && saw_abort_response_fire_q);
`elsif FORMAL_RESTART
        cover(rst_n && scenario == SC_GAMMA_FRAME && fault_seen_q &&
              rejected_run_q && recovery_q && gamma_epoch_q == 2 &&
              loaded_epoch_q == 2 && model_gamma_valid_q && restarted_q &&
              completed_q);
`elsif FORMAL_RESTART_SCRATCH
        cover(rst_n && scenario == SC_SCRATCH && restarted_q && completed_q);
`elsif FORMAL_RESTART_MUL1
        cover(rst_n && scenario == SC_MUL1_LATE && saw_late_prefix_q &&
              fault_seen_q && restarted_q && completed_q);
`elsif FORMAL_RESTART_MUL2
        cover(rst_n && scenario == SC_MUL2_LATE && saw_late_prefix_q &&
              fault_seen_q && restarted_q && completed_q);
`elsif FORMAL_RESTART_ABORT
        cover(rst_n && scenario == SC_ABORT && abort_seen_q && restarted_q &&
              completed_q && abort_phase_seen_q == abort_phase);
`ifdef FORMAL_ABORT_PHASE_0
        cover(rst_n && scenario == SC_ABORT && abort_phase == 0 &&
              abort_seen_q && restarted_q && completed_q);
`elsif FORMAL_ABORT_PHASE_1
        cover(rst_n && scenario == SC_ABORT && abort_phase == 1 &&
              abort_seen_q && restarted_q && completed_q);
`elsif FORMAL_ABORT_PHASE_2
        cover(rst_n && scenario == SC_ABORT && abort_phase == 2 &&
              abort_seen_q && saw_abort_drain_q && restarted_q && completed_q);
`elsif FORMAL_ABORT_PHASE_3
        cover(rst_n && scenario == SC_ABORT && abort_phase == 3 &&
              abort_seen_q && restarted_q && completed_q);
`elsif FORMAL_ABORT_PHASE_4
        cover(rst_n && scenario == SC_ABORT && abort_phase == 4 &&
              abort_seen_q && restarted_q && completed_q);
`elsif FORMAL_ABORT_PHASE_5
        cover(rst_n && scenario == SC_ABORT && abort_phase == 5 &&
              abort_seen_q && saw_abort_scalar_stall_q && restarted_q &&
              completed_q);
`elsif FORMAL_ABORT_PHASE_6
        cover(rst_n && scenario == SC_ABORT && abort_phase == 6 &&
              abort_seen_q && saw_abort_response_fire_q && restarted_q &&
              completed_q);
`endif
`endif
`endif
    end
endmodule

`default_nettype wire
