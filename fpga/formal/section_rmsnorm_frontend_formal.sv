`default_nettype none

// Real-leaf composition proof at rows=8, tokens=2. Each token occupies one
// scratch group, keeping the bounded model small while retaining token stride,
// untagged read ownership, tentative results, and abort/drain/restart behavior.
module section_rmsnorm_frontend_formal(input wire clk);
    localparam [2:0] SC_CLEAN     = 3'd0;
    localparam [2:0] SC_EARLY     = 3'd1;
    localparam [2:0] SC_SINK      = 3'd2;
    localparam [2:0] SC_NONFINITE = 3'd3;
    localparam [2:0] SC_SUBNORMAL = 3'd4;
    localparam [2:0] SC_READ_ERR  = 3'd5;
    localparam [2:0] SC_BAD_CFG   = 3'd6;
    localparam [2:0] SC_ABORT     = 3'd7;

    localparam [6:0] ST_BAD_CFG   = 7'h01;
    localparam [6:0] ST_LOADER    = 7'h02;
    localparam [6:0] ST_MAXEXP    = 7'h04;
    localparam [6:0] ST_SCRATCH   = 7'h10;
    localparam [6:0] ST_SUBNORMAL = 7'h20;
    localparam [47:0] EXPECTED_SUM = 48'h0020_0000_0000;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg source_allow_any;
    (* anyseq *) reg write_allow_any;
    (* anyseq *) reg request_allow_any;
    (* anyseq *) reg response_allow_any;
    (* anyseq *) reg result_ready_any;
    (* anyconst *) reg [2:0] scenario;

`ifdef FORMAL_BMC
    wire [2:0] active_scenario = SC_CLEAN;
    wire source_allow = source_allow_any;
    wire write_allow = write_allow_any;
    wire request_allow = request_allow_any;
    wire response_allow = response_allow_any;
    wire result_ready = result_ready_any;
`else
    wire [2:0] active_scenario = scenario;
    wire source_allow = 1'b1;
    wire write_allow = 1'b1;
    wire request_allow = 1'b1;
    wire response_allow = 1'b1;
    wire result_ready = 1'b1;
`endif

    reg f_past_valid = 1'b0;
    reg cfg_pending_q;
    reg [3:0] sent_words_q;
    reg [3:0] accepted_writes_q;
    reg [2:0] accepted_requests_q;
    reg [2:0] accepted_results_q;
    reg response_pending_q;
    reg [1:0] response_token_q;
    reg [2:0] response_age_q;
    reg aborted_q;
    reg restarted_q;
    reg saw_abort_owner_q;
    reg saw_drain_q;

    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [13:0] cfg_rows = ((active_scenario == SC_BAD_CFG) &&
                            !restarted_q) ? 14'd7 : 14'd8;
    wire [2:0] cfg_tokens = 3'd2;
    wire busy;
    wire done;
    wire error;
    wire [6:0] status;

    function automatic [31:0] lane_value(
        input [2:0] which,
        input [4:0] ordinal,
        input restarted
    );
        reg [31:0] value;
        begin
            value = ordinal < 5'd8 ? 32'h3f80_0000 : 32'h3f00_0000;
            if (!restarted && (which == SC_NONFINITE) && ordinal == 5'd1)
                value = 32'h7f80_0000;
            if (!restarted && (which == SC_SUBNORMAL) && ordinal == 5'd1)
                value = 32'h0000_0001;
            lane_value = value;
        end
    endfunction

    function automatic [63:0] word_value(
        input [2:0] which,
        input [3:0] word,
        input restarted
    );
        begin
            word_value = {
                lane_value(which, {word, 1'b1}, restarted),
                lane_value(which, {word, 1'b0}, restarted)
            };
        end
    endfunction

    function automatic [255:0] group_value(input [1:0] token);
        reg [31:0] lane;
        begin
            lane = token == 0 ? 32'h3f80_0000 : 32'h3f00_0000;
            group_value = {8{lane}};
        end
    endfunction

    wire input_valid = busy && sent_words_q < 4'd8 && source_allow;
    wire [63:0] input_data = word_value(
        active_scenario, sent_words_q, restarted_q
    );
    wire input_last = !restarted_q && (active_scenario == SC_EARLY) ?
                      sent_words_q == 0 : sent_words_q == 7;
    wire input_ready;

    wire r_wr_valid;
    wire r_wr_ready = write_allow;
    wire r_wr_error = !restarted_q && (active_scenario == SC_SINK) &&
                      sent_words_q == 1;
    wire [1:0] r_wr_bank;
    wire [13:0] r_wr_address;
    wire [63:0] r_wr_data;

    wire rd_req_valid;
    wire rd_req_ready = request_allow && !response_pending_q;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire rd_rsp_valid = response_pending_q && response_age_q >= 3 &&
                        response_allow;
    wire rd_rsp_ready;
    wire [255:0] rd_rsp_data = group_value(response_token_q);
    wire rd_rsp_error = !restarted_q && (active_scenario == SC_READ_ERR);

    wire result_valid;
    wire [1:0] result_token;
    wire [7:0] result_max_exp;
    wire [47:0] result_sum_sq;
    wire [13:0] result_rows;
    wire result_final;

    wire cfg_fire = cfg_valid && cfg_ready;
    wire input_fire = input_valid && input_ready;
    wire write_fire = r_wr_valid && r_wr_ready;
    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire result_fire = result_valid && result_ready;

    wire abort_point = response_pending_q && response_age_q == 1;
`ifdef FORMAL_COVER
    wire abort_run = !restarted_q && !aborted_q &&
                     (active_scenario == SC_ABORT) && abort_point;
`else
    wire abort_run = 1'b0;
`endif

    section_rmsnorm_frontend dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_axis_tdata(input_data), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(input_valid), .s_axis_tready(input_ready),
        .s_axis_tlast(input_last),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_error(r_wr_error), .r_wr_bank(r_wr_bank),
        .r_wr_address(r_wr_address), .r_wr_data(r_wr_data),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_token(rd_req_token), .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_max_exp(result_max_exp),
        .result_sum_sq(result_sum_sq), .result_rows(result_rows),
        .result_final(result_final)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

`ifdef FORMAL_FAULTS
        assume(scenario >= SC_EARLY && scenario <= SC_BAD_CFG);
`elsif FORMAL_COVER
        assume(scenario <= SC_ABORT);
`else
        assume(scenario == SC_CLEAN);
`endif

        if (f_past_valid && rst_n && busy &&
            $past(rst_n && busy && !abort_run && input_valid && !input_ready))
            assume(source_allow);
        if (f_past_valid && rst_n && response_pending_q &&
            $past(rst_n && response_pending_q && rd_rsp_valid &&
                  !rd_rsp_ready))
            assume(response_allow);

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            sent_words_q <= 4'd0;
            accepted_writes_q <= 4'd0;
            accepted_requests_q <= 3'd0;
            accepted_results_q <= 3'd0;
            response_pending_q <= 1'b0;
            response_token_q <= 2'd0;
            response_age_q <= 3'd0;
            aborted_q <= 1'b0;
            restarted_q <= 1'b0;
            saw_abort_owner_q <= 1'b0;
            saw_drain_q <= 1'b0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                sent_words_q <= 4'd0;
                accepted_writes_q <= 4'd0;
                accepted_requests_q <= 3'd0;
                accepted_results_q <= 3'd0;
                if (aborted_q)
                    restarted_q <= 1'b1;
            end
            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                sent_words_q <= 4'd0;
                accepted_writes_q <= 4'd0;
                accepted_requests_q <= 3'd0;
                accepted_results_q <= 3'd0;
                aborted_q <= 1'b1;
                saw_abort_owner_q <= response_pending_q;
            end else begin
                if (input_fire)
                    sent_words_q <= sent_words_q + 1'b1;
                if (write_fire)
                    accepted_writes_q <= accepted_writes_q + 1'b1;
                if (request_fire)
                    accepted_requests_q <= accepted_requests_q + 1'b1;
                if (result_fire)
                    accepted_results_q <= accepted_results_q + 1'b1;
            end

            if (request_fire) begin
                response_pending_q <= 1'b1;
                response_token_q <= rd_req_token[1:0];
                response_age_q <= 3'd0;
            end else if (response_pending_q && !response_fire) begin
                if (response_age_q != 3'd7)
                    response_age_q <= response_age_q + 1'b1;
            end
            if (response_fire) begin
                response_pending_q <= 1'b0;
                response_age_q <= 3'd0;
            end
            if (aborted_q && !restarted_q && busy)
                saw_drain_q <= 1'b1;
        end

        if (rst_n) begin
            assert(input_fire == write_fire);
            assert(sent_words_q == accepted_writes_q);
            assert(sent_words_q <= 8);
            assert(accepted_requests_q <= 2);
            assert(accepted_results_q <= 2);
            assert(!(done && busy));
            assert(!(result_valid && error));

            if (r_wr_valid) begin
                assert(r_wr_bank == sent_words_q[1:0]);
                assert(r_wr_address == (sent_words_q < 4 ? 14'd0 : 14'd512));
                assert(r_wr_data == input_data);
            end
            if (rd_req_valid) begin
                assert(rd_req_group == 0);
                assert(rd_req_token == accepted_requests_q);
                assert(!response_pending_q);
                assert(accepted_writes_q == 8);
            end
            if (response_pending_q)
                assert(response_token_q < 2);
            if (result_valid) begin
                assert(result_token == accepted_results_q[1:0]);
                assert(result_max_exp == (result_token == 0 ? 8'd127 : 8'd126));
                assert(result_sum_sq == EXPECTED_SUM);
                assert(result_rows == 8);
                assert(result_final == (accepted_results_q == 1));
            end

            if (done && !error) begin
                assert(restarted_q || active_scenario == SC_CLEAN);
                assert(status == 0);
                assert(sent_words_q == 8);
                assert(accepted_writes_q == 8);
                assert(accepted_requests_q == 2);
                assert(accepted_results_q == 2);
            end
            if (done && error) begin
                assert(!restarted_q);
                assert(accepted_results_q == 0);
                case (active_scenario)
                    SC_EARLY, SC_SINK: begin
                        assert((status & ST_LOADER) != 0);
                    end
                    SC_NONFINITE: begin
                        assert((status & ST_MAXEXP) != 0);
                    end
                    SC_SUBNORMAL: begin
                        assert((status & ST_SUBNORMAL) != 0);
                    end
                    SC_READ_ERR: begin
                        assert((status & ST_SCRATCH) != 0);
                    end
                    SC_BAD_CFG: begin
                        assert(status == ST_BAD_CFG);
                    end
                    default: begin
                        assert(1'b0);
                    end
                endcase
            end
            if (aborted_q && !restarted_q) begin
                assert(!done);
                assert(!error);
                assert(status == 0);
            end
        end

        cover(rst_n && done && !error && !restarted_q &&
              active_scenario == SC_CLEAN);
        cover(rst_n && done && error && active_scenario == SC_EARLY);
        cover(rst_n && done && error && active_scenario == SC_SINK);
        cover(rst_n && done && error && active_scenario == SC_NONFINITE);
        cover(rst_n && done && error && active_scenario == SC_SUBNORMAL);
        cover(rst_n && done && error && active_scenario == SC_READ_ERR);
        cover(rst_n && done && error && active_scenario == SC_BAD_CFG);
        cover(rst_n && restarted_q && done && !error &&
              saw_abort_owner_q && saw_drain_q);
    end
endmodule

`default_nettype wire
