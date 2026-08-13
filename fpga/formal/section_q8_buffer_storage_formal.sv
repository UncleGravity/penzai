`default_nettype none

// Reduced-storage proof of the production lifecycle and stream control. The
// frozen address arithmetic remains production width, while FORMAL_REDUCED_STORAGE
// limits RAM depth and duplicate clearing to the four exercised records.
module section_q8_buffer_storage_formal(input wire clk);
    localparam [4:0] PH_CFG_GOOD       = 5'd0;
    localparam [4:0] PH_WAIT_GOOD      = 5'd1;
    localparam [4:0] PH_WRITE_GOOD     = 5'd2;
    localparam [4:0] PH_SEAL_GOOD      = 5'd3;
    localparam [4:0] PH_READ_GOOD      = 5'd4;
    localparam [4:0] PH_CFG_ABORT      = 5'd5;
    localparam [4:0] PH_WAIT_ABORT     = 5'd6;
    localparam [4:0] PH_WRITE_ABORT    = 5'd7;
    localparam [4:0] PH_FIRE_ABORT     = 5'd8;
    localparam [4:0] PH_CHECK_ABORT    = 5'd9;
    localparam [4:0] PH_CFG_DUP        = 5'd10;
    localparam [4:0] PH_WAIT_DUP       = 5'd11;
    localparam [4:0] PH_WRITE_DUP_A    = 5'd12;
    localparam [4:0] PH_WRITE_DUP_B    = 5'd13;
    localparam [4:0] PH_SEAL_DUP       = 5'd14;
    localparam [4:0] PH_DONE           = 5'd15;

    function automatic [63:0] record_data(
        input [1:0] record_index,
        input [2:0] beat_index
    );
        if (beat_index == 3'd4)
            record_data = {48'd0, 8'ha5, 4'd0, record_index, beat_index[1:0]};
        else
            record_data = {8'h51, record_index, beat_index, 51'h123456789abcd};
    endfunction

    wire output_ready = 1'b1;

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;
    reg [4:0] phase = PH_CFG_GOOD;
    reg [2:0] beat = 3'd0;
    reg [2:0] record = 3'd0;
    reg request_sent = 1'b0;

    wire cfg_valid = (phase == PH_CFG_GOOD) ||
                     (phase == PH_CFG_ABORT) ||
                     (phase == PH_CFG_DUP);
    wire cfg_ready;
    wire [2:0] cfg_tokens = (phase == PH_CFG_DUP) ? 3'd1 : 3'd2;
    wire [8:0] cfg_blocks = (phase == PH_CFG_DUP) ? 9'd1 : 9'd2;

    wire seal_valid = (phase == PH_SEAL_GOOD) || (phase == PH_SEAL_DUP);
    wire seal_ready;
    wire seal_done;
    wire seal_error;
    wire abort_valid = phase == PH_FIRE_ABORT;

    wire writing_good = phase == PH_WRITE_GOOD;
    wire writing_abort = phase == PH_WRITE_ABORT;
    wire writing_dup_a = phase == PH_WRITE_DUP_A;
    wire writing_dup_b = phase == PH_WRITE_DUP_B;
    wire stream_valid = writing_good || writing_abort ||
                        writing_dup_a || writing_dup_b;
    wire stream_ready;
    wire [1:0] stream_token = writing_good ? {1'b0, record[0]} : 2'd0;
    wire [8:0] stream_block = writing_good ? {8'd0, record[1]} : 9'd0;
    wire [63:0] stream_data = record_data(
        writing_good ? record[1:0] : 2'd0, beat);
    wire stream_last = beat == 3'd4;
    wire stream_accept = stream_valid && stream_ready;
    wire cap_record_done;
    wire cap_record_error;
    wire cap_commit_valid;
    wire [11:0] cap_commit_address;

    wire reading_good = phase == PH_READ_GOOD;
    wire rd_req_valid = reading_good && !request_sent;
    wire rd_req_ready;
    // Read order is token-major: record[1] is token, record[0] is block.
    wire [1:0] rd_req_token = {1'b0, record[1]};
    wire [8:0] rd_req_block = {8'd0, record[0]};
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

    wire [1:0] bank_clearing;
    wire [1:0] bank_active;
    wire [1:0] bank_valid;
    wire [1:0] bank_error;
    wire [10:0] bank0_record_count;
    wire [10:0] bank1_record_count_unused;
    wire formal_cap_busy;
    wire [2:0] formal_cap_index;
    wire formal_rd_pending;
    wire [2:0] formal_emit_index;

    section_q8_buffer dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready), .cfg_bank(1'b0),
        .cfg_tokens(cfg_tokens), .cfg_blocks(cfg_blocks),
        .seal_valid(seal_valid), .seal_ready(seal_ready),
        .seal_bank(1'b0), .seal_done(seal_done), .seal_error(seal_error),
        .abort_valid(abort_valid), .abort_bank(1'b0),
        .bank_clearing(bank_clearing), .bank_active(bank_active),
        .bank_valid(bank_valid), .bank_error(bank_error),
        .bank0_record_count(bank0_record_count),
        .bank1_record_count(bank1_record_count_unused),
        .s_axis_tdata(stream_data), .s_axis_tvalid(stream_valid),
        .s_axis_tready(stream_ready), .s_axis_tlast(stream_last),
        .s_axis_bank(1'b0), .s_axis_token(stream_token),
        .s_axis_block(stream_block),
        .cap_record_done(cap_record_done),
        .cap_record_error(cap_record_error),
        .cap_commit_valid(cap_commit_valid),
        .cap_commit_address(cap_commit_address),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_bank(1'b0), .rd_req_token(rd_req_token),
        .rd_req_block(rd_req_block), .rd_issue_valid(rd_issue_valid),
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

    wire [2:0] expected_record = {1'b0, record[0], record[1]};

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            phase <= PH_CFG_GOOD;
            beat <= 3'd0;
            record <= 3'd0;
            request_sent <= 1'b0;
        end else begin
            case (phase)
                PH_CFG_GOOD: if (cfg_valid && cfg_ready) begin
                    phase <= PH_WAIT_GOOD;
                    record <= 3'd0;
                    beat <= 3'd0;
                end
                PH_WAIT_GOOD: if (bank_active[0])
                    phase <= PH_WRITE_GOOD;
                PH_WRITE_GOOD: if (stream_accept) begin
                    if (beat == 3'd4) begin
                        beat <= 3'd0;
                        if (record == 3'd3) begin
                            phase <= PH_SEAL_GOOD;
                            record <= 3'd0;
                        end else begin
                            record <= record + 1'b1;
                        end
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end
                PH_SEAL_GOOD: if (seal_valid && seal_ready) begin
                    phase <= PH_READ_GOOD;
                    record <= 3'd0;
                    beat <= 3'd0;
                    request_sent <= 1'b0;
                end
                PH_READ_GOOD: begin
                    if (rd_req_valid && rd_req_ready)
                        request_sent <= 1'b1;
                    if (out_accept) begin
                        if (beat == 3'd4) begin
                            beat <= 3'd0;
                            request_sent <= 1'b0;
                            if (record == 3'd3) begin
                                phase <= PH_CFG_ABORT;
                                record <= 3'd0;
                            end else begin
                                record <= record + 1'b1;
                            end
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                PH_CFG_ABORT: if (cfg_valid && cfg_ready) begin
                    phase <= PH_WAIT_ABORT;
                    beat <= 3'd0;
                end
                PH_WAIT_ABORT: if (bank_active[0])
                    phase <= PH_WRITE_ABORT;
                PH_WRITE_ABORT: if (stream_accept) begin
                    if (beat == 3'd1) begin
                        phase <= PH_FIRE_ABORT;
                        beat <= 3'd0;
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end
                PH_FIRE_ABORT: phase <= PH_CHECK_ABORT;
                PH_CHECK_ABORT: phase <= PH_CFG_DUP;
                PH_CFG_DUP: if (cfg_valid && cfg_ready) begin
                    phase <= PH_WAIT_DUP;
                    beat <= 3'd0;
                end
                PH_WAIT_DUP: if (bank_active[0])
                    phase <= PH_WRITE_DUP_A;
                PH_WRITE_DUP_A: if (stream_accept) begin
                    if (beat == 3'd4) begin
                        phase <= PH_WRITE_DUP_B;
                        beat <= 3'd0;
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end
                PH_WRITE_DUP_B: if (stream_accept) begin
                    if (beat == 3'd4) begin
                        phase <= PH_SEAL_DUP;
                        beat <= 3'd0;
                    end else begin
                        beat <= beat + 1'b1;
                    end
                end
                PH_SEAL_DUP: if (seal_valid && seal_ready)
                    phase <= PH_DONE;
                default: ;
            endcase
        end

        if (rst_n) begin
            assert(bank_clearing[1:0] != 2'b11);
            assert(!bank_active[1] && !bank_valid[1]);
            if (stream_valid)
                assert(stream_ready);
            if (cap_commit_valid) begin
                assert(cap_commit_address ==
                       {1'b0, stream_block, 2'b00} +
                       {10'd0, stream_token});
                assert(beat == 3'd4);
            end

            if (phase == PH_READ_GOOD && out_valid) begin
                assert(!out_error && !out_bank);
                assert(out_token == rd_req_token);
                assert(out_block == rd_req_block);
                assert(formal_emit_index == beat);
                assert(out_last == (beat == 3'd4));
                assert(out_data == record_data(expected_record[1:0], beat));
            end

            if (phase == PH_CHECK_ABORT) begin
                assert(!formal_cap_busy);
                assert(!bank_active[0] && !bank_valid[0]);
                assert(bank_error[0]);
                assert(bank0_record_count == 11'd0);
            end

            if (phase == PH_SEAL_DUP) begin
                assert(bank_error[0]);
                assert(bank0_record_count == 11'd1);
            end
            if (phase == PH_DONE) begin
                assert(!bank_active[0] && !bank_valid[0]);
                assert(bank_error[0]);
                assert(bank0_record_count == 11'd1);
                if ($past(phase == PH_SEAL_DUP))
                    assert(seal_done && seal_error);
            end
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
        end

        cover(rst_n && phase == PH_DONE);
    end

    wire _unused = &{1'b0, cap_record_done, cap_record_error,
                     rd_issue_valid, rd_issue_address, formal_cap_index,
                     formal_rd_pending, bank1_record_count_unused};
endmodule

`default_nettype wire
