`default_nettype none

// Safety and bounded ordering proof boundary for section_gate_packer. The data
// task completes a maximum-token one-block run and proves every packed bit/tag.
// The control task admits arbitrary source/output stalls, framing faults, and
// aborts. The cover task cancels a stalled output, restarts, and completes.
module section_gate_packer_formal(input wire clk);
    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;

`ifdef FORMAL_DATA
    localparam [2:0] RUN_TOKENS = 3'd4;
    localparam [6:0] RUN_BEATS = 7'd64;
    localparam [4:0] RUN_GROUPS = 5'd16;
`else
    localparam [2:0] RUN_TOKENS = 3'd2;
    localparam [6:0] RUN_BEATS = 7'd32;
    localparam [4:0] RUN_GROUPS = 5'd8;
`endif

    (* anyseq *) reg f_input_offer;
    (* anyseq *) reg f_keep_fault_choice;
    (* anyseq *) reg f_frame_fault_choice;
    (* anyseq *) reg f_output_ready;
    (* anyseq *) reg f_abort_choice;

    reg want_start = 1'b1;
    reg [6:0] accepted_beats = 7'd0;
    reg [4:0] accepted_groups = 5'd0;
    reg source_valid_q = 1'b0;
    reg source_keep_fault_q = 1'b0;
    reg source_frame_fault_q = 1'b0;
    reg expected_failure_q = 1'b0;
    reg saw_input_stall = 1'b0;
    reg saw_output_stall = 1'b0;
    reg saw_frame_fault = 1'b0;
    reg saw_keep_fault = 1'b0;
    reg saw_abort = 1'b0;
    reg saw_restart = 1'b0;

`ifdef FORMAL_DATA
    wire input_offer = 1'b1;
    wire keep_fault_choice = 1'b0;
    wire frame_fault_choice = 1'b0;
    wire output_ready = !m_axis_tvalid || saw_output_stall;
    wire abort_choice = 1'b0;
`elsif FORMAL_COVER
    wire input_offer = 1'b1;
    wire keep_fault_choice = 1'b0;
    wire frame_fault_choice = 1'b0;
    wire output_ready = !m_axis_tvalid || saw_output_stall;
    // Abort one cycle after observing the first stalled result. Basing this on
    // registered history avoids feeding abort back through m_axis_tvalid.
    wire abort_choice = !saw_abort && saw_output_stall;
`else
    wire input_offer = f_input_offer;
    wire keep_fault_choice = f_keep_fault_choice;
    wire frame_fault_choice = f_frame_fault_choice;
    wire output_ready = f_output_ready;
    wire abort_choice = f_abort_choice;
`endif

    function automatic [63:0] beat_value(input [6:0] ordinal);
        begin
            beat_value = {25'h1000000, ordinal, 25'h0800000, ordinal};
        end
    endfunction

    function automatic [255:0] group_value(input [4:0] ordinal);
        reg [6:0] first;
        begin
            first = {ordinal, 2'b00};
            group_value = {
                beat_value(first + 7'd3), beat_value(first + 7'd2),
                beat_value(first + 7'd1), beat_value(first)
            };
        end
    endfunction

    wire start_valid = want_start;
    wire start_ready;
    wire busy;
    wire done;
    wire error;
    wire abort_run = rst_n && busy && abort_choice;

    wire s_axis_tvalid = busy && source_valid_q;
    wire s_axis_tready;
    wire [63:0] s_axis_tdata = beat_value(accepted_beats);
    wire [7:0] s_axis_tkeep = source_keep_fault_q ? 8'hfe : 8'hff;
    wire physical_last = accepted_beats + 1'b1 == RUN_BEATS;
    wire s_axis_tlast = physical_last ^ source_frame_fault_q;

    wire [255:0] m_axis_tdata;
    wire m_axis_tvalid;
    wire m_axis_tlast;
    wire [1:0] m_axis_token;
    wire [8:0] m_axis_block;
    wire [1:0] m_axis_group;

    wire start_fire = start_valid && start_ready;
    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire input_bad = input_fire &&
                     (source_keep_fault_q || source_frame_fault_q);
    wire output_fire = m_axis_tvalid && output_ready;

    wire [5:0] expected_rowblock = accepted_groups / (RUN_TOKENS * 2);
    wire [2:0] expected_token = (accepted_groups >> 1) % RUN_TOKENS;
    wire [1:0] expected_group = {expected_rowblock[0], accepted_groups[0]};

    section_gate_packer dut (
        .clk(clk), .rst_n(rst_n),
        .start_valid(start_valid), .start_ready(start_ready),
        .start_tokens(RUN_TOKENS), .start_blocks(9'd1),
        .abort_run(abort_run), .busy(busy), .done(done), .error(error),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(output_ready), .m_axis_tlast(m_axis_tlast),
        .m_axis_token(m_axis_token), .m_axis_block(m_axis_block),
        .m_axis_group(m_axis_group)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!rst_n) begin
            want_start <= 1'b1;
            accepted_beats <= 7'd0;
            accepted_groups <= 5'd0;
            source_valid_q <= 1'b0;
            source_keep_fault_q <= 1'b0;
            source_frame_fault_q <= 1'b0;
            expected_failure_q <= 1'b0;
            saw_input_stall <= 1'b0;
            saw_output_stall <= 1'b0;
            saw_frame_fault <= 1'b0;
            saw_keep_fault <= 1'b0;
            saw_abort <= 1'b0;
            saw_restart <= 1'b0;
        end else begin
            if (start_fire) begin
                want_start <= 1'b0;
                accepted_beats <= 7'd0;
                accepted_groups <= 5'd0;
                source_valid_q <= 1'b0;
                source_keep_fault_q <= 1'b0;
                source_frame_fault_q <= 1'b0;
                expected_failure_q <= 1'b0;
                if (saw_abort)
                    saw_restart <= 1'b1;
            end

            if (done)
                want_start <= 1'b1;

            if (!busy) begin
                source_valid_q <= 1'b0;
            end else if (input_fire) begin
                source_valid_q <= 1'b0;
            end else if (!source_valid_q && (accepted_beats < RUN_BEATS) &&
                         input_offer) begin
                source_valid_q <= 1'b1;
                source_keep_fault_q <= keep_fault_choice;
                source_frame_fault_q <= frame_fault_choice;
            end

            if (input_fire && !input_bad)
                accepted_beats <= accepted_beats + 1'b1;
            if (output_fire)
                accepted_groups <= accepted_groups + 1'b1;

            if (input_bad) begin
                expected_failure_q <= 1'b1;
                if (source_keep_fault_q)
                    saw_keep_fault <= 1'b1;
                if (source_frame_fault_q)
                    saw_frame_fault <= 1'b1;
            end
            if (abort_run) begin
                expected_failure_q <= 1'b1;
                saw_abort <= 1'b1;
            end

            if (s_axis_tvalid && !s_axis_tready)
                saw_input_stall <= 1'b1;
            if (m_axis_tvalid && !output_ready)
                saw_output_stall <= 1'b1;

            assert(accepted_beats <= RUN_BEATS);
            assert(accepted_groups <= RUN_GROUPS);
            assert({accepted_groups, 2'b00} <= accepted_beats);

            if (s_axis_tvalid) begin
                assert(accepted_beats < RUN_BEATS);
                assert(s_axis_tdata == beat_value(accepted_beats));
                assert(s_axis_tkeep ==
                       (source_keep_fault_q ? 8'hfe : 8'hff));
                assert(s_axis_tlast ==
                       (physical_last ^ source_frame_fault_q));
            end

            if (m_axis_tvalid) begin
                assert(accepted_groups < RUN_GROUPS);
                assert(m_axis_token == expected_token[1:0]);
                assert(m_axis_block == {4'd0, expected_rowblock[5:1]});
                assert(m_axis_group == expected_group);
                assert(m_axis_tlast == (expected_group == 2'd3));
`ifdef FORMAL_DATA
                assert(m_axis_tdata == group_value(accepted_groups));
`endif
            end

            if (done && !error) begin
                assert(!expected_failure_q);
                assert(accepted_beats == RUN_BEATS);
                assert(accepted_groups == RUN_GROUPS);
            end
            if (done && error)
                assert(expected_failure_q);
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && busy && s_axis_tvalid && !s_axis_tready &&
                  !abort_run)) begin
            assert(s_axis_tvalid);
            assert(s_axis_tdata == $past(s_axis_tdata));
            assert(s_axis_tkeep == $past(s_axis_tkeep));
            assert(s_axis_tlast == $past(s_axis_tlast));
        end

        if (f_past_valid && $past(rst_n && (input_bad || abort_run))) begin
            assert(!busy);
            assert(done && error);
            assert(!s_axis_tready);
            assert(!m_axis_tvalid);
        end

`ifdef FORMAL_COVER
        cover(rst_n && saw_abort && saw_restart && done && !error &&
              saw_output_stall);
`endif
`ifdef FORMAL_DATA_COVER
        cover(rst_n && done && !error &&
              accepted_beats == RUN_BEATS &&
              accepted_groups == RUN_GROUPS &&
              saw_output_stall);
`endif
    end
endmodule

`default_nettype wire
