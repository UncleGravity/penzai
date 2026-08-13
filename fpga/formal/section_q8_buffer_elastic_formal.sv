`default_nettype none

// Prove replay elasticity independently of the storage lifecycle. An unsealed
// bank intentionally returns the canonical five-beat error record, which keeps
// payload RAM contents outside the proof cone while ready remains unconstrained.
module section_q8_buffer_elastic_formal(input wire clk);
    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;

    reg request_issued = 1'b0;
    reg response_done = 1'b0;
    reg [2:0] accepted_beats = 3'd0;
    (* anyseq *) reg output_ready;

    wire rd_req_valid = !request_issued;
    wire rd_req_ready;
    wire rd_issue_valid;
    wire [11:0] rd_issue_address;

    wire [63:0] out_data;
    wire out_valid;
    wire out_last;
    wire out_error;
    wire out_bank;
    wire [1:0] out_token;
    wire [8:0] out_block;
    wire out_accept = out_valid && output_ready;

    wire cfg_ready;
    wire seal_ready;
    wire seal_done;
    wire seal_error;
    wire [1:0] bank_clearing;
    wire [1:0] bank_active;
    wire [1:0] bank_valid;
    wire [1:0] bank_error;
    wire [10:0] bank0_record_count;
    wire [10:0] bank1_record_count;
    wire stream_ready;
    wire cap_record_done;
    wire cap_record_error;
    wire cap_commit_valid;
    wire [11:0] cap_commit_address;
    wire formal_cap_busy;
    wire [2:0] formal_cap_index;
    wire formal_rd_pending;
    wire [2:0] formal_emit_index;

    section_q8_buffer dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(1'b0), .cfg_ready(cfg_ready), .cfg_bank(1'b0),
        .cfg_tokens(3'd0), .cfg_blocks(9'd0),
        .seal_valid(1'b0), .seal_ready(seal_ready), .seal_bank(1'b0),
        .seal_done(seal_done), .seal_error(seal_error),
        .abort_valid(1'b0), .abort_bank(1'b0),
        .bank_clearing(bank_clearing), .bank_active(bank_active),
        .bank_valid(bank_valid), .bank_error(bank_error),
        .bank0_record_count(bank0_record_count),
        .bank1_record_count(bank1_record_count),
        .s_axis_tdata(64'd0), .s_axis_tvalid(1'b0),
        .s_axis_tready(stream_ready), .s_axis_tlast(1'b0),
        .s_axis_bank(1'b0), .s_axis_token(2'd0), .s_axis_block(9'd0),
        .cap_record_done(cap_record_done),
        .cap_record_error(cap_record_error),
        .cap_commit_valid(cap_commit_valid),
        .cap_commit_address(cap_commit_address),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_bank(1'b0), .rd_req_token(2'd0), .rd_req_block(9'd0),
        .rd_issue_valid(rd_issue_valid),
        .rd_issue_address(rd_issue_address),
        .m_axis_tdata(out_data), .m_axis_tvalid(out_valid),
        .m_axis_tready(output_ready), .m_axis_tlast(out_last),
        .m_axis_error(out_error), .m_axis_bank(out_bank),
        .m_axis_token(out_token), .m_axis_block(out_block),
        .formal_cap_busy(formal_cap_busy),
        .formal_cap_index(formal_cap_index),
        .formal_rd_pending(formal_rd_pending),
        .formal_emit_index(formal_emit_index)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            request_issued <= 1'b0;
            response_done <= 1'b0;
            accepted_beats <= 3'd0;
        end else begin
            if (rd_req_valid && rd_req_ready)
                request_issued <= 1'b1;

            if (out_accept) begin
                if (out_last) begin
                    response_done <= 1'b1;
                end else begin
                    accepted_beats <= accepted_beats + 1'b1;
                end
            end

            assert(!bank_clearing && !bank_active && !bank_valid && !bank_error);
            assert(bank0_record_count == 11'd0);
            assert(bank1_record_count == 11'd0);
            assert(!formal_cap_busy && !cap_commit_valid);

            if (rd_issue_valid)
                assert(rd_issue_address == 12'd0);
            if (out_valid) begin
                assert(request_issued);
                assert(out_error);
                assert(out_data == 64'd0);
                assert(!out_bank && out_token == 2'd0 && out_block == 9'd0);
                assert(formal_emit_index == accepted_beats);
                assert(out_last == (accepted_beats == 3'd4));
            end
            if (response_done)
                assert(!out_valid && accepted_beats == 3'd4);
        end

        if (f_past_valid && rst_n &&
            $past(rst_n && out_valid && !output_ready)) begin
            assert(out_valid);
            assert(out_data == $past(out_data));
            assert(out_last == $past(out_last));
            assert(out_error == $past(out_error));
            assert(out_bank == $past(out_bank));
            assert(out_token == $past(out_token));
            assert(out_block == $past(out_block));
            assert(formal_emit_index == $past(formal_emit_index));
        end

        cover(rst_n && response_done);
    end

    wire _unused = &{1'b0, cfg_ready, seal_ready, seal_done, seal_error,
                     stream_ready, cap_record_done, cap_record_error,
                     cap_commit_address, formal_cap_index, formal_rd_pending};
endmodule

`default_nettype wire
