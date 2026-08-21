`default_nettype none

// Bounded control proof for the scratch-backed residual-add boundary. The
// representative rows=16 shape exercises one read per pair, both R groups, all
// four banks, the native pair wrap, final publication, and the residency seal
// in a compact state space. Production shape coverage remains real-RTL cosim
// over all 24 legal rows/tokens combinations.
module section_residual_add_formal(input wire clk);
    localparam [2:0] SC_CLEAN       = 3'd0;
    localparam [2:0] SC_READ_FAULT  = 3'd1;
    localparam [2:0] SC_WRITE_FAULT = 3'd2;
    localparam [2:0] SC_FRAME_FAULT = 3'd3;
    localparam [2:0] SC_ARITH_FAULT = 3'd4;
    localparam [2:0] SC_ABORT_READ  = 3'd5;
    localparam [2:0] SC_ORPHAN      = 3'd6;

`ifdef FORMAL_SC_READ_FAULT
    localparam [2:0] SCENARIO = SC_READ_FAULT;
`elsif FORMAL_SC_WRITE_FAULT
    localparam [2:0] SCENARIO = SC_WRITE_FAULT;
`elsif FORMAL_SC_FRAME_FAULT
    localparam [2:0] SCENARIO = SC_FRAME_FAULT;
`elsif FORMAL_SC_ARITH_FAULT
    localparam [2:0] SCENARIO = SC_ARITH_FAULT;
`elsif FORMAL_SC_ABORT_READ
    localparam [2:0] SCENARIO = SC_ABORT_READ;
`elsif FORMAL_SC_ORPHAN
    localparam [2:0] SCENARIO = SC_ORPHAN;
`else
    localparam [2:0] SCENARIO = SC_CLEAN;
`endif

    localparam [31:0] ARITH_FAULT_WORD = 32'h7f80_0000;
    localparam [6:0] STATUS_FRAME       = 7'h02;
    localparam [6:0] STATUS_READ        = 7'h04;
    localparam [6:0] STATUS_WRITE       = 7'h08;
    localparam [6:0] STATUS_ARITH       = 7'h10;
    localparam [6:0] STATUS_INTERNAL    = 7'h40;

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;

    (* anyseq *) reg f_req_ready;
    (* anyseq *) reg f_write_ready;
    (* anyseq *) reg f_output_ready;
`ifdef FORMAL_PROGRESS
    wire request_ready = 1'b1;
    wire write_ready = 1'b1;
    wire output_ready = 1'b1;
`elsif FORMAL_COVER
    wire request_ready = 1'b1;
    wire write_ready = 1'b1;
    wire output_ready = 1'b1;
`else
    wire request_ready = f_req_ready;
    wire write_ready = f_write_ready;
    wire output_ready = f_output_ready;
`endif

    reg cfg_pending_q;
    reg restart_phase_q;
    reg restarted_q;
    reg saw_initial_event_q;
    reg saw_abort_retained_q;
    reg saw_orphan_quarantine_q;
    reg abort_seen_q;
    reg orphan_seen_q;

    reg [3:0] sent_words_q;
    reg [3:0] requests_q;
    reg [3:0] responses_q;
    reg [3:0] write_attempts_q;
    reg [3:0] committed_writes_q;
    reg [3:0] outputs_q;

    reg         env_read_pending_q;
    reg         env_rsp_valid_q;
    reg         env_rsp_error_q;

    wire initial_run = !restart_phase_q;
    wire cfg_valid = rst_n && cfg_pending_q;
    wire cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;
    wire busy;
    wire done;
    wire error;
    wire [6:0] status;

    wire abort_run = rst_n && initial_run &&
                     (SCENARIO == SC_ABORT_READ) &&
                     env_read_pending_q && !env_rsp_valid_q && !abort_seen_q;

    function automatic [63:0] word_value(input [3:0] ordinal);
        begin
            word_value = {
                24'h3f1234, ordinal, 4'h5,
                24'h3e5678, ordinal, 4'ha
            };
        end
    endfunction

    wire input_valid = rst_n && busy && !abort_run &&
                       (sent_words_q < 4'd8);
    wire [63:0] normal_input_data = word_value(sent_words_q);
    wire [63:0] input_data = initial_run &&
        (SCENARIO == SC_ARITH_FAULT) && (sent_words_q == 4'd0) ?
        {normal_input_data[63:32], ARITH_FAULT_WORD} : normal_input_data;
    wire input_last = initial_run && (SCENARIO == SC_FRAME_FAULT) ?
                      (sent_words_q == 4'd0) : (sent_words_q == 4'd7);
    wire input_ready;
    wire input_fire = input_valid && input_ready;

    wire rd_req_valid;
    wire [1:0] rd_req_role;
    wire [2:0] rd_req_token;
    wire [10:0] rd_req_group;
    wire request_fire = rd_req_valid && request_ready;

    // The orphan is injected raw in ST_REQ, before any legitimate request can
    // handshake. This is deliberately independent of rd_req_valid/ready.
    wire orphan_now = rst_n && initial_run && (SCENARIO == SC_ORPHAN) &&
                      busy && (requests_q == 4'd0) &&
                      !env_read_pending_q && !orphan_seen_q;
    wire rd_rsp_valid = orphan_now || env_rsp_valid_q;
    wire rd_rsp_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire [255:0] rd_rsp_data = {
        64'h0000_0000_0000_0000,
        64'h0000_0000_0000_0000,
        64'h0000_0000_0000_0000,
        64'h0000_0000_0000_0000
    };
    wire rd_rsp_error = env_rsp_error_q;

    wire r_wr_valid;
    wire [1:0] r_wr_bank;
    wire [13:0] r_wr_address;
    wire [63:0] r_wr_data;
    wire r_wr_error = initial_run && (SCENARIO == SC_WRITE_FAULT) &&
                      (write_attempts_q == 4'd0);
    wire write_fire = r_wr_valid && write_ready;

    wire [63:0] output_data;
    wire [7:0] output_keep;
    wire output_valid;
    wire output_last;
    wire output_fire = output_valid && output_ready;

    section_residual_add #(
        .MIN_ROWS(14'd8),
        .MAX_ROWS(14'd4096)
    ) dut (
        .clk(clk), .rst_n(rst_n), .abort_run(abort_run),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(14'd16), .cfg_tokens(3'd1),
        .busy(busy), .done(done), .error(error), .status(status),
        .s_axis_tdata(input_data), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(input_valid), .s_axis_tready(input_ready),
        .s_axis_tlast(input_last),
        .rd_req_valid(rd_req_valid), .rd_req_ready(request_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error),
        .r_wr_valid(r_wr_valid), .r_wr_ready(write_ready),
        .r_wr_error(r_wr_error), .r_wr_bank(r_wr_bank),
        .r_wr_address(r_wr_address), .r_wr_data(r_wr_data),
        .m_axis_tdata(output_data), .m_axis_tkeep(output_keep),
        .m_axis_tvalid(output_valid), .m_axis_tready(output_ready),
        .m_axis_tlast(output_last)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            restart_phase_q <= 1'b0;
            restarted_q <= 1'b0;
            saw_initial_event_q <= 1'b0;
            saw_abort_retained_q <= 1'b0;
            saw_orphan_quarantine_q <= 1'b0;
            abort_seen_q <= 1'b0;
            orphan_seen_q <= 1'b0;
            sent_words_q <= 4'd0;
            requests_q <= 4'd0;
            responses_q <= 4'd0;
            write_attempts_q <= 4'd0;
            committed_writes_q <= 4'd0;
            outputs_q <= 4'd0;
            env_read_pending_q <= 1'b0;
            env_rsp_valid_q <= 1'b0;
            env_rsp_error_q <= 1'b0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                sent_words_q <= 4'd0;
                requests_q <= 4'd0;
                responses_q <= 4'd0;
                write_attempts_q <= 4'd0;
                committed_writes_q <= 4'd0;
                outputs_q <= 4'd0;
                if (restart_phase_q)
                    restarted_q <= 1'b1;
                assert(!env_read_pending_q);
                assert(!env_rsp_valid_q);
            end

            if (request_fire) begin
                assert(!env_read_pending_q);
                env_read_pending_q <= 1'b1;
                requests_q <= requests_q + 1'b1;
            end

            if (env_read_pending_q && !env_rsp_valid_q &&
                !((SCENARIO == SC_ABORT_READ) && initial_run &&
                  !abort_seen_q)) begin
                env_rsp_valid_q <= 1'b1;
                env_rsp_error_q <= initial_run &&
                    (SCENARIO == SC_READ_FAULT) &&
                    (responses_q == 4'd0);
            end else if (env_rsp_valid_q && rd_rsp_ready) begin
                env_rsp_valid_q <= 1'b0;
                env_rsp_error_q <= 1'b0;
                env_read_pending_q <= 1'b0;
                // Do not count the retained response that merely drains an
                // aborted run. Responses from the configured restart are
                // part of the new run and must be counted normally.
                if (!restart_phase_q || restarted_q)
                    responses_q <= responses_q + 1'b1;
            end

            if (orphan_now && rd_rsp_ready) begin
                orphan_seen_q <= 1'b1;
                saw_orphan_quarantine_q <= 1'b1;
            end

            if (input_fire)
                sent_words_q <= sent_words_q + 1'b1;
            if (write_fire) begin
                write_attempts_q <= write_attempts_q + 1'b1;
                if (!r_wr_error)
                    committed_writes_q <= committed_writes_q + 1'b1;
            end
            if (output_fire)
                outputs_q <= outputs_q + 1'b1;

            if (abort_run) begin
                abort_seen_q <= 1'b1;
                saw_abort_retained_q <= env_read_pending_q;
                saw_initial_event_q <= 1'b1;
                restart_phase_q <= 1'b1;
                cfg_pending_q <= 1'b1;
                sent_words_q <= 4'd0;
                requests_q <= 4'd0;
                responses_q <= 4'd0;
                write_attempts_q <= 4'd0;
                committed_writes_q <= 4'd0;
                outputs_q <= 4'd0;
            end

            if (done && error && initial_run) begin
                saw_initial_event_q <= 1'b1;
                restart_phase_q <= 1'b1;
                cfg_pending_q <= 1'b1;
            end

            assert(sent_words_q <= 4'd8);
            assert(requests_q <= 4'd8);
            assert(responses_q <= requests_q);
            assert(write_attempts_q <= sent_words_q);
            assert(committed_writes_q <= write_attempts_q);
            assert(outputs_q <= committed_writes_q);
            assert(!(done && busy));

            if (rd_req_valid) begin
                assert(rd_req_role == 2'd0);
                assert(rd_req_token == 3'd0);
                assert(rd_req_group == {10'd0, requests_q[2]});
                assert(requests_q < 4'd8);
                assert(!rd_rsp_valid);
            end

            if (r_wr_valid) begin
                assert(write_attempts_q < 4'd8);
                assert(r_wr_bank == write_attempts_q[1:0]);
                assert(r_wr_address == {13'd0, write_attempts_q[2]});
                assert(r_wr_data == word_value(write_attempts_q));
            end

            if (output_valid) begin
                assert(outputs_q < committed_writes_q);
                assert(output_keep == 8'hff);
                assert(output_data == word_value(outputs_q));
                assert(output_last == (outputs_q == 4'd7));
            end

            if (done && !error) begin
                assert(status == 7'd0);
                assert(sent_words_q == 4'd8);
                assert(requests_q == 4'd8);
                assert(responses_q == 4'd8);
                assert(write_attempts_q == 4'd8);
                assert(committed_writes_q == 4'd8);
                assert(outputs_q == 4'd8);
                if (SCENARIO != SC_CLEAN)
                    assert(restarted_q && saw_initial_event_q);
            end

            if (done && error && initial_run) begin
                case (SCENARIO)
                    SC_READ_FAULT: begin
                        assert(status == STATUS_READ);
                        assert(requests_q == 4'd1);
                        assert(sent_words_q == 4'd0);
                    end
                    SC_WRITE_FAULT: begin
                        assert(status == STATUS_WRITE);
                        assert(write_attempts_q == 4'd1);
                        assert(committed_writes_q == 4'd0);
                        assert(outputs_q == 4'd0);
                    end
                    SC_FRAME_FAULT: begin
                        assert(status == STATUS_FRAME);
                        assert(sent_words_q == 4'd1);
                        assert(write_attempts_q == 4'd0);
                    end
                    SC_ARITH_FAULT: begin
                        assert(status == STATUS_ARITH);
                        assert(sent_words_q == 4'd1);
                        assert(write_attempts_q == 4'd0);
                    end
                    SC_ORPHAN: begin
                        assert(status == STATUS_INTERNAL);
                        assert(saw_orphan_quarantine_q);
                        assert(requests_q == 4'd0);
                        assert(sent_words_q == 4'd0);
                        assert(write_attempts_q == 4'd0);
                        assert(outputs_q == 4'd0);
                    end
                    default: assert(1'b0);
                endcase
            end

            if (orphan_now) begin
                assert(rd_rsp_ready);
                assert(!rd_req_valid);
                assert(!input_ready);
                assert(!r_wr_valid);
                assert(!output_valid);
            end

            if (saw_abort_retained_q && env_read_pending_q) begin
                assert(busy);
                assert(!cfg_ready);
            end

            if (abort_run) begin
                assert(env_read_pending_q);
                assert(!rd_req_valid);
                assert(!input_ready);
                assert(!r_wr_valid);
                assert(!output_valid);
            end
        end

        if (f_past_valid && rst_n &&
            $past(rst_n && rd_req_valid && !request_ready && !abort_run)) begin
            assert(rd_req_valid);
            assert(rd_req_role == $past(rd_req_role));
            assert(rd_req_token == $past(rd_req_token));
            assert(rd_req_group == $past(rd_req_group));
        end

        if (f_past_valid && rst_n &&
            $past(rst_n && r_wr_valid && !write_ready && !abort_run)) begin
            assert(r_wr_valid);
            assert(r_wr_bank == $past(r_wr_bank));
            assert(r_wr_address == $past(r_wr_address));
            assert(r_wr_data == $past(r_wr_data));
        end

        if (f_past_valid && rst_n &&
            $past(rst_n && output_valid && !output_ready && !abort_run)) begin
            assert(output_valid);
            assert(output_data == $past(output_data));
            assert(output_keep == $past(output_keep));
            assert(output_last == $past(output_last));
        end

`ifdef FORMAL_COVER
        if (SCENARIO == SC_CLEAN)
            cover(rst_n && done && !error);
        else
            cover(rst_n && saw_initial_event_q && restarted_q &&
                  done && !error);
        if (SCENARIO == SC_ABORT_READ)
            cover(rst_n && saw_abort_retained_q && restarted_q &&
                  done && !error);
        if (SCENARIO == SC_ORPHAN)
            cover(rst_n && saw_orphan_quarantine_q && restarted_q &&
                  done && !error);
`endif
    end
endmodule

`default_nettype wire
