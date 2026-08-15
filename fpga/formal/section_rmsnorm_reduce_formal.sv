`default_nettype none

// Composition proof for the buffered P3d RMSNorm scalar boundary. Child stubs
// preserve their independently proven lifecycle contracts and make retained
// scratch ownership visible to the wrapper's cross-abort cleanup controller.
module section_rmsnorm_reduce_formal(input wire clk);
    localparam [2:0] SC_CLEAN       = 3'd0;
    localparam [2:0] SC_BAD_CFG     = 3'd1;
    localparam [2:0] SC_FRONT_FAULT = 3'd2;
    localparam [2:0] SC_INV_FAULT   = 3'd3;
    localparam [2:0] SC_ABORT       = 3'd4;

    localparam [1:0] DUT_ST_IDLE    = 2'd0;
    localparam [1:0] DUT_ST_RUN     = 2'd1;
    localparam [1:0] DUT_ST_CLEANUP = 2'd2;
    localparam [1:0] DUT_ST_EMIT    = 2'd3;

    localparam [12:0] STATUS_BAD_CFG       = 13'h001;
    localparam [12:0] STATUS_FRONT_SCRATCH = 13'h020;
    localparam [12:0] STATUS_INV_ARITH     = 13'h400;
    localparam [31:0] CLEAN_EPS = 32'h3586_37bd;
    localparam [31:0] FAULT_EPS = 32'h3586_37be;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg result_ready_any;
    (* anyconst *) reg [2:0] scenario;

`ifdef FORMAL_BMC
    wire [2:0] active_scenario = SC_CLEAN;
`else
    wire [2:0] active_scenario = scenario;
`endif

`ifdef FORMAL_FAULTS
    wire result_ready = 1'b1;
`else
    wire result_ready = result_ready_any;
`endif

    reg f_past_valid = 1'b0;
    reg cfg_pending_q;
    reg aborted_q;
    reg faulted_q;
    reg restarted_q;
    reg [2:0] output_count_q;
    reg [2:0] request_count_q;
    reg response_pending_q;
    reg [2:0] response_ordinal_q;
    reg [3:0] response_age_q;
    reg saw_inv_cleanup_owner_q;
    reg saw_front_cross_abort_q;
    reg saw_abort_owner_q;
    reg saw_abort_drain_q;
    reg saw_output_stall_q;

    wire initial_run = !aborted_q && !faulted_q && !restarted_q;
    wire recovery_cfg = !initial_run;
    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [13:0] cfg_rows = !recovery_cfg &&
        (active_scenario == SC_BAD_CFG) ? 14'd7 : 14'd8;
    wire [2:0] cfg_tokens = 3'd2;
    wire [31:0] cfg_eps = !recovery_cfg &&
        (active_scenario == SC_INV_FAULT) ? FAULT_EPS : CLEAN_EPS;

    wire busy;
    wire done;
    wire error;
    wire [12:0] status;
    wire result_valid;
    wire [1:0] result_token;
    wire [31:0] result_inv_rms;
    wire result_final;

    wire rd_req_valid;
    wire rd_req_ready = !response_pending_q;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_valid = response_pending_q && (response_age_q >= 4'd6);
    wire rd_rsp_ready;
    wire rd_rsp_error = initial_run &&
        (active_scenario == SC_FRONT_FAULT) &&
        (response_ordinal_q == 3'd1);

    wire [1:0] formal_state;
    wire [2:0] formal_bridge_count;
    wire [2:0] formal_capture_count;
    wire formal_front_busy;
    wire formal_inv_busy;

    wire cfg_fire = cfg_valid && cfg_ready;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire result_fire = result_valid && result_ready;
    wire abort_point = initial_run && !aborted_q &&
                       (active_scenario == SC_ABORT) &&
                       response_pending_q &&
                       (response_ordinal_q == 3'd1) &&
                       (response_age_q == 4'd2);
`ifdef FORMAL_BMC
    wire abort_run = 1'b0;
`else
    wire abort_run = abort_point;
`endif

    section_rmsnorm_reduce dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens), .cfg_eps(cfg_eps),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_axis_tdata(64'd0), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(1'b0), .s_axis_tready(), .s_axis_tlast(1'b0),
        .r_wr_valid(), .r_wr_ready(1'b1), .r_wr_error(1'b0),
        .r_wr_bank(), .r_wr_address(), .r_wr_data(),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_token(rd_req_token), .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data({8{32'h3f80_0000}}),
        .rd_rsp_error(rd_rsp_error),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_inv_rms(result_inv_rms),
        .result_final(result_final),
        .formal_state(formal_state),
        .formal_bridge_count(formal_bridge_count),
        .formal_capture_count(formal_capture_count),
        .formal_front_busy(formal_front_busy),
        .formal_inv_busy(formal_inv_busy)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

`ifdef FORMAL_FAULTS
        assume(scenario >= SC_BAD_CFG && scenario <= SC_ABORT);
`elsif FORMAL_COVER
        assume(scenario <= SC_ABORT);
`else
        assume(scenario == SC_CLEAN);
`endif

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            aborted_q <= 1'b0;
            faulted_q <= 1'b0;
            restarted_q <= 1'b0;
            output_count_q <= 3'd0;
            request_count_q <= 3'd0;
            response_pending_q <= 1'b0;
            response_ordinal_q <= 3'd0;
            response_age_q <= 4'd0;
            saw_inv_cleanup_owner_q <= 1'b0;
            saw_front_cross_abort_q <= 1'b0;
            saw_abort_owner_q <= 1'b0;
            saw_abort_drain_q <= 1'b0;
            saw_output_stall_q <= 1'b0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                output_count_q <= 3'd0;
                request_count_q <= 3'd0;
                if (aborted_q || faulted_q)
                    restarted_q <= 1'b1;
            end

            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                aborted_q <= 1'b1;
                output_count_q <= 3'd0;
                saw_abort_owner_q <= response_pending_q;
            end

            if (done && error && initial_run) begin
                cfg_pending_q <= 1'b1;
                faulted_q <= 1'b1;
                output_count_q <= 3'd0;
            end

            if (result_fire)
                output_count_q <= output_count_q + 1'b1;
            if (result_valid && !result_ready)
                saw_output_stall_q <= 1'b1;

            if (request_fire) begin
                response_pending_q <= 1'b1;
                response_ordinal_q <= request_count_q;
                response_age_q <= 4'd0;
                request_count_q <= request_count_q + 1'b1;
            end else if (response_pending_q && !response_fire) begin
                if (response_age_q != 4'hf)
                    response_age_q <= response_age_q + 1'b1;
            end
            if (response_fire) begin
                response_pending_q <= 1'b0;
                response_age_q <= 4'd0;
            end

            if (initial_run && (active_scenario == SC_INV_FAULT) &&
                (formal_state == DUT_ST_CLEANUP) && response_pending_q) begin
                saw_inv_cleanup_owner_q <= 1'b1;
            end
            if (initial_run && (active_scenario == SC_FRONT_FAULT) &&
                (formal_state == DUT_ST_CLEANUP) && formal_inv_busy) begin
                saw_front_cross_abort_q <= 1'b1;
            end
            if (aborted_q && !restarted_q &&
                (formal_state == DUT_ST_CLEANUP) && response_pending_q) begin
                saw_abort_drain_q <= 1'b1;
            end
        end

        if (rst_n) begin
            assert(output_count_q <= 3'd2);
            assert(request_count_q <= 3'd2);
            assert(!(done && busy));
            assert(!(result_valid && error));

            if (f_past_valid &&
                $past(rst_n && cfg_fire && (cfg_rows == 14'd8))) begin
                assert(busy);
                assert(formal_state == DUT_ST_RUN);
                assert(formal_front_busy);
                assert(formal_inv_busy);
            end

            if (f_past_valid &&
                $past(rst_n && cfg_fire && (cfg_rows == 14'd7))) begin
                assert(!busy);
                assert(done && error);
                assert(status == STATUS_BAD_CFG);
                assert(!formal_front_busy);
                assert(!formal_inv_busy);
                assert(!rd_req_valid);
                assert(!result_valid);
            end

            if (initial_run && (active_scenario == SC_BAD_CFG)) begin
                assert(!formal_front_busy);
                assert(!formal_inv_busy);
                assert(!rd_req_valid);
                assert(!result_valid);
                assert(request_count_q == 0);
            end

            if (result_valid) begin
                assert(formal_state == DUT_ST_EMIT);
                assert(!formal_front_busy);
                assert(!formal_inv_busy);
                assert(formal_bridge_count == 3'd2);
                assert(formal_capture_count == 3'd2);
                assert(status == 0);
                assert(result_token == output_count_q[1:0]);
                assert(result_inv_rms ==
                       (32'h3f80_0000 + {30'd0, output_count_q[1:0]}));
                assert(result_final == (output_count_q == 3'd1));
            end

            if ((formal_state == DUT_ST_RUN) &&
                (formal_capture_count != 0)) begin
                assert(!result_valid);
            end
            if (formal_state == DUT_ST_CLEANUP) begin
                assert(busy);
                assert(!result_valid);
            end

            if (initial_run && (active_scenario == SC_INV_FAULT) &&
                (formal_state == DUT_ST_CLEANUP) && response_pending_q) begin
                assert(formal_front_busy);
                assert(rd_rsp_ready);
                assert(error);
                assert(status == STATUS_INV_ARITH);
            end

            if (initial_run && (active_scenario == SC_FRONT_FAULT) &&
                (formal_state == DUT_ST_CLEANUP) && formal_inv_busy) begin
                assert(error);
                assert(status == STATUS_FRONT_SCRATCH);
            end

            if (aborted_q && !restarted_q) begin
                assert(!done);
                assert(!error);
                assert(status == 0);
                assert(!result_valid);
                if (response_pending_q) begin
                    assert(busy);
                    assert(formal_front_busy);
                    assert(rd_rsp_ready);
                end
            end

            if (done && error) begin
                assert(initial_run);
                assert(output_count_q == 0);
                assert(!formal_front_busy);
                assert(!formal_inv_busy);
                case (active_scenario)
                    SC_BAD_CFG: assert(status == STATUS_BAD_CFG);
                    SC_FRONT_FAULT:
                        assert(status == STATUS_FRONT_SCRATCH);
                    SC_INV_FAULT: assert(status == STATUS_INV_ARITH);
                    default: assert(1'b0);
                endcase
            end

            if (done && !error) begin
                assert((active_scenario == SC_CLEAN) || restarted_q);
                assert(status == 0);
                assert(output_count_q == 3'd2);
                assert(!formal_front_busy);
                assert(!formal_inv_busy);
            end
        end

        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_inv_rms == $past(result_inv_rms));
            assert(result_final == $past(result_final));
        end

`ifdef FORMAL_COVER
        cover(rst_n && (active_scenario == SC_CLEAN) &&
              done && !error && !restarted_q);
        cover(rst_n && (active_scenario == SC_CLEAN) &&
              saw_output_stall_q && result_fire);
        cover(rst_n && (active_scenario == SC_BAD_CFG) &&
              faulted_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_FRONT_FAULT) &&
              saw_front_cross_abort_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_INV_FAULT) &&
              saw_inv_cleanup_owner_q && restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_ABORT) &&
              saw_abort_owner_q && saw_abort_drain_q &&
              restarted_q && done && !error);
`endif
    end
endmodule

`default_nettype wire
