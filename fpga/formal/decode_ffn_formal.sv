`default_nettype none

module decode_ffn_formal(input wire clk);
`ifdef FORMAL_COVER_CLEAN
    wire abort_case = 1'b0;
`elsif FORMAL_COVER_ABORT
    wire abort_case = 1'b1;
`else
    (* anyconst *) reg abort_case;
`endif
    reg rst_n = 1'b0;

    localparam [7:0] REG_CTRL = 8'h08;
    localparam [7:0] REG_NUM_Q1 = 8'h10;
    localparam [7:0] REG_NUM_RB = 8'h14;
    localparam [7:0] REG_NUM_COLS = 8'h38;
    localparam [7:0] REG_WEIGHT_FMT = 8'h44;
    localparam [7:0] REG_NUM_ROWS = 8'h48;
    localparam [7:0] REG_ACT_MODE = 8'h4c;
    localparam [7:0] REG_ACT_EPOCH = 8'h50;
    localparam [7:0] REG_SCRATCH_MODE = 8'h68;
    localparam [7:0] REG_SCRATCH_ROLE = 8'h6c;
    localparam [7:0] REG_SCRATCH_ROWS = 8'h70;
    localparam [7:0] REG_SCRATCH_TOKENS = 8'h74;
    localparam [7:0] REG_SCRATCH_CTRL = 8'h78;

    // Each write occupies a four-cycle slot. The two valid cycles match the
    // wrapper's registered AW/W acceptance and commit boundary. Wait commands
    // advance from observable lifecycle states rather than guessed latency.
    reg [5:0] command_q;
    reg [1:0] slot_phase_q;
    reg did_abort_q;
    reg [7:0] write_addr;
    reg [31:0] write_data;
    reg write_enable;

    always @* begin
        write_addr = 8'd0;
        write_data = 32'd0;
        write_enable = 1'b1;
        case (command_q)
            6'd0:  begin write_addr = REG_SCRATCH_MODE; write_data = 0; end
            6'd1:  begin write_addr = REG_SCRATCH_ROWS; write_data = 128; end
            6'd2:  begin write_addr = REG_SCRATCH_TOKENS; write_data = 1; end
            6'd3:  begin write_addr = REG_SCRATCH_CTRL; write_data = 4; end
            6'd4:  begin write_addr = REG_SCRATCH_MODE; write_data = 3; end
            6'd5:  begin write_addr = REG_SCRATCH_ROLE; write_data = 2; end
            6'd6:  begin write_addr = REG_NUM_Q1; write_data = 1; end
            6'd7:  begin write_addr = REG_NUM_RB; write_data = 8; end
            6'd8:  begin write_addr = REG_NUM_ROWS; write_data = 128; end
            6'd9:  begin write_addr = REG_NUM_COLS; write_data = 1; end
            6'd10: begin write_addr = REG_WEIGHT_FMT; write_data = 1; end
            6'd11: begin write_addr = REG_ACT_MODE; write_data = 2; end
            6'd12: begin write_addr = REG_ACT_EPOCH; write_data = 1; end
            6'd13: begin write_addr = REG_CTRL; write_data = 1; end
            6'd15: begin write_addr = REG_SCRATCH_ROLE; write_data = 1; end
            6'd16: begin write_addr = REG_ACT_MODE; write_data = 1; end
            6'd17: begin write_addr = REG_CTRL; write_data = 1; end
            6'd19: begin write_addr = REG_SCRATCH_MODE; write_data = 0; end
            6'd20: begin write_addr = REG_ACT_MODE; write_data = 3; end
            6'd21: begin write_addr = REG_ACT_EPOCH; write_data = 2; end
            6'd22: begin write_addr = REG_CTRL; write_data = 1; end
            6'd30: begin write_addr = REG_SCRATCH_CTRL; write_data = 2; end
            6'd32: begin write_addr = REG_SCRATCH_MODE; write_data = 0; end
            6'd33: begin write_addr = REG_SCRATCH_ROWS; write_data = 128; end
            6'd34: begin write_addr = REG_SCRATCH_TOKENS; write_data = 1; end
            6'd35: begin write_addr = REG_SCRATCH_CTRL; write_data = 4; end
            default: write_enable = 1'b0;
        endcase
    end

    wire axi_valid = rst_n && write_enable && (slot_phase_q < 2);
    wire [1:0] bresp;
    wire bvalid;
    wire awready, wready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire arready, rvalid;
    wire [127:0] weight_word = 128'd0;
    wire w0_ready, w1_ready, w2_ready, w3_ready;
    wire acts_ready;
    wire [63:0] result_data;
    wire [7:0] result_keep;
    wire result_valid, result_last;
    wire [2:0] formal_ffn_phase;
    wire formal_ffn_gate_ready;
    wire [1:0] formal_scratch_rd_owner;
    wire formal_scratch_rd_rsp_valid, formal_scratch_rd_rsp_ready;
    wire formal_section_active, formal_section_done;
    wire formal_abort_cleanup;
    wire [6:0] formal_scratch_error;
    wire formal_capture_fire, formal_capture_record_done;
    wire [10:0] formal_bank0_record_count;
    wire formal_replay_fire, formal_replay_complete;
    wire formal_down_kernel_done, formal_kernel_done;
    wire formal_abort_strobe, formal_ffn_fault;
    wire formal_section_begin_ok, formal_up_start, formal_gate_start;
    wire formal_gate_drain_ready, formal_seal_done;
    wire formal_down_start, formal_down_complete_ready;
    wire formal_producer_done;
    wire [1:0] formal_capture_token, formal_replay_token;
    wire [8:0] formal_capture_block, formal_replay_block;
    wire [2:0] formal_capture_beat, formal_replay_beat;
    wire formal_legacy_q8_cfg_pending, formal_legacy_q8_cfg_fire;
    wire formal_legacy_q8_cfg_ready;
`ifdef FORMAL_COVER_CLEAN
    wire formal_inject_abort = 1'b0;
`elsif FORMAL_COVER_ABORT
    wire formal_inject_abort = 1'b0;
`else
    (* anyseq *) reg formal_inject_abort;
    reg formal_injected_abort_q = 1'b0;
    always @(posedge clk) begin
        if (!rst_n)
            formal_injected_abort_q <= 1'b0;
        else if (formal_inject_abort)
            formal_injected_abort_q <= 1'b1;
    end
    always @* begin
        assume(!formal_inject_abort ||
               (formal_legacy_q8_cfg_pending &&
                !formal_injected_abort_q));
    end
`endif

    decode_top dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .s_axi_awaddr(write_addr), .s_axi_awprot(3'd0),
        .s_axi_awvalid(axi_valid), .s_axi_awready(awready),
        .s_axi_wdata(write_data), .s_axi_wstrb(4'hf),
        .s_axi_wvalid(axi_valid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(8'd0), .s_axi_arprot(3'd0),
        .s_axi_arvalid(1'b0), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
        .s_axis_w0_tdata(weight_word), .s_axis_w0_tkeep(16'hffff),
        .s_axis_w0_tvalid(1'b1), .s_axis_w0_tready(w0_ready),
        .s_axis_w0_tlast(1'b0),
        .s_axis_w1_tdata(weight_word), .s_axis_w1_tkeep(16'hffff),
        .s_axis_w1_tvalid(1'b1), .s_axis_w1_tready(w1_ready),
        .s_axis_w1_tlast(1'b0),
        .s_axis_w2_tdata(weight_word), .s_axis_w2_tkeep(16'hffff),
        .s_axis_w2_tvalid(1'b1), .s_axis_w2_tready(w2_ready),
        .s_axis_w2_tlast(1'b0),
        .s_axis_w3_tdata(weight_word), .s_axis_w3_tkeep(16'hffff),
        .s_axis_w3_tvalid(1'b1), .s_axis_w3_tready(w3_ready),
        .s_axis_w3_tlast(1'b0),
        .s_axis_acts_tdata(64'd0), .s_axis_acts_tkeep(8'hff),
        .s_axis_acts_tvalid(1'b1), .s_axis_acts_tready(acts_ready),
        .s_axis_acts_tlast(1'b0),
        .m_axis_tdata(result_data), .m_axis_tkeep(result_keep),
        .m_axis_tvalid(result_valid), .m_axis_tready(1'b1),
        .m_axis_tlast(result_last),
        .formal_inject_abort(formal_inject_abort),
        .formal_ffn_phase(formal_ffn_phase),
        .formal_ffn_gate_ready(formal_ffn_gate_ready),
        .formal_scratch_rd_owner(formal_scratch_rd_owner),
        .formal_scratch_rd_rsp_valid(formal_scratch_rd_rsp_valid),
        .formal_scratch_rd_rsp_ready(formal_scratch_rd_rsp_ready),
        .formal_section_active(formal_section_active),
        .formal_section_done(formal_section_done),
        .formal_abort_cleanup(formal_abort_cleanup),
        .formal_scratch_error(formal_scratch_error),
        .formal_capture_fire(formal_capture_fire),
        .formal_capture_record_done(formal_capture_record_done),
        .formal_bank0_record_count(formal_bank0_record_count),
        .formal_replay_fire(formal_replay_fire),
        .formal_replay_complete(formal_replay_complete),
        .formal_down_kernel_done(formal_down_kernel_done),
        .formal_kernel_done(formal_kernel_done),
        .formal_abort_strobe(formal_abort_strobe),
        .formal_ffn_fault(formal_ffn_fault),
        .formal_section_begin_ok(formal_section_begin_ok),
        .formal_up_start(formal_up_start),
        .formal_gate_start(formal_gate_start),
        .formal_gate_drain_ready(formal_gate_drain_ready),
        .formal_seal_done(formal_seal_done),
        .formal_down_start(formal_down_start),
        .formal_down_complete_ready(formal_down_complete_ready),
        .formal_producer_done(formal_producer_done),
        .formal_capture_token(formal_capture_token),
        .formal_capture_block(formal_capture_block),
        .formal_capture_beat(formal_capture_beat),
        .formal_replay_token(formal_replay_token),
        .formal_replay_block(formal_replay_block),
        .formal_replay_beat(formal_replay_beat),
        .formal_legacy_q8_cfg_pending(formal_legacy_q8_cfg_pending),
        .formal_legacy_q8_cfg_fire(formal_legacy_q8_cfg_fire),
        .formal_legacy_q8_cfg_ready(formal_legacy_q8_cfg_ready)
    );

    // Drive one clean section. If abort_case is selected, interrupt the first
    // GATE run only after PAIRER owns a physical scratch request, wait for the
    // orphan response and cleanup mask, then run the same clean section again.
    always @(posedge clk) begin
        rst_n <= 1'b1;
        if (!rst_n) begin
            command_q <= 6'd0;
            slot_phase_q <= 2'd0;
            did_abort_q <= 1'b0;
        end else begin
            case (command_q)
                6'd14: begin
                    slot_phase_q <= 2'd0;
                    if (formal_ffn_gate_ready)
                        command_q <= 6'd15;
                end
                6'd18: begin
                    slot_phase_q <= 2'd0;
                    if (abort_case && !did_abort_q) begin
                        if (formal_scratch_rd_owner == 2'd2)
                            command_q <= 6'd30;
                    end else if (formal_ffn_phase == 3'd6) begin
                        command_q <= 6'd19;
                    end
                end
                6'd23: begin
                    slot_phase_q <= 2'd0;
                    command_q <= 6'd23;
                end
                6'd31: begin
                    slot_phase_q <= 2'd0;
                    if (!formal_section_active && !formal_abort_cleanup &&
                        (formal_scratch_rd_owner == 2'd0) &&
                        !formal_scratch_rd_rsp_valid)
                        command_q <= 6'd32;
                end
                default: begin
                    if (slot_phase_q == 2'd3) begin
                        slot_phase_q <= 2'd0;
                        if (command_q == 6'd30) begin
                            command_q <= 6'd31;
                            did_abort_q <= 1'b1;
                        end else if (command_q == 6'd35) begin
                            command_q <= 6'd4;
                        end else begin
                            command_q <= command_q + 1'b1;
                        end
                    end else begin
                        slot_phase_q <= slot_phase_q + 1'b1;
                    end
                end
            endcase
        end
    end

    reg f_past_valid = 1'b0;
    reg prev_rst_n_q = 1'b0;
    reg [4:0] capture_beats_q;
    reg [2:0] capture_records_q;
    reg [2:0] capture_expect_beat_q;
    reg [2:0] capture_expect_block_q;
    reg [4:0] replay_beats_q;
    reg [2:0] replay_expect_beat_q;
    reg [2:0] replay_expect_block_q;
    reg [2:0] clean_milestone_q;
    reg saw_abort_owner_q;
    reg saw_abort_hold_q;
    reg saw_abort_drained_q;
    reg saw_restart_begin_q;
    reg [1:0] prev_owner_q;
    reg [2:0] prev_phase_q;
    reg prev_rsp_fire_q;
    reg prev_section_active_q;
    reg prev_abort_cleanup_q;
    reg prev_any_error_q;
    reg prev_replay_complete_q;
    reg prev_down_kernel_done_q;
    reg prev_kernel_done_q;
    reg prev_abort_strobe_q;
    reg prev_fault_q;
    reg prev_section_begin_ok_q;
    reg prev_up_start_q;
    reg prev_gate_start_q;
    reg prev_gate_drain_ready_q;
    reg prev_seal_done_q;
    reg prev_down_start_q;
    reg prev_down_complete_ready_q;
    reg prev_legacy_q8_cfg_pending_q;
    reg saw_legacy_q8_cfg_q;

    wire capture_fire = formal_capture_fire;
    wire replay_fire = formal_replay_fire;
    wire scratch_rsp_fire = formal_scratch_rd_rsp_valid &&
                            formal_scratch_rd_rsp_ready;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        prev_rst_n_q <= rst_n;
        prev_owner_q <= formal_scratch_rd_owner;
        prev_phase_q <= formal_ffn_phase;
        prev_rsp_fire_q <= scratch_rsp_fire;
        prev_section_active_q <= formal_section_active;
        prev_abort_cleanup_q <= formal_abort_cleanup;
        prev_any_error_q <= |formal_scratch_error;
        prev_replay_complete_q <= formal_replay_complete;
        prev_down_kernel_done_q <= formal_down_kernel_done;
        prev_kernel_done_q <= formal_kernel_done;
        prev_abort_strobe_q <= formal_abort_strobe;
        prev_fault_q <= formal_ffn_fault;
        prev_section_begin_ok_q <= formal_section_begin_ok;
        prev_up_start_q <= formal_up_start;
        prev_gate_start_q <= formal_gate_start;
        prev_gate_drain_ready_q <= formal_gate_drain_ready;
        prev_seal_done_q <= formal_seal_done;
        prev_down_start_q <= formal_down_start;
        prev_down_complete_ready_q <= formal_down_complete_ready;
        prev_legacy_q8_cfg_pending_q <= formal_legacy_q8_cfg_pending;
        if (!rst_n) begin
            capture_beats_q <= 5'd0;
            capture_records_q <= 3'd0;
            capture_expect_beat_q <= 3'd0;
            capture_expect_block_q <= 3'd0;
            replay_beats_q <= 5'd0;
            replay_expect_beat_q <= 3'd0;
            replay_expect_block_q <= 3'd0;
            clean_milestone_q <= 3'd0;
            saw_abort_owner_q <= 1'b0;
            saw_abort_hold_q <= 1'b0;
            saw_abort_drained_q <= 1'b0;
            saw_restart_begin_q <= 1'b0;
            saw_legacy_q8_cfg_q <= 1'b0;
        end else begin
            if (formal_section_begin_ok) begin
                clean_milestone_q <= 3'd1;
                capture_beats_q <= 5'd0;
                replay_beats_q <= 5'd0;
                assert(!formal_legacy_q8_cfg_pending);
                assert(!formal_legacy_q8_cfg_fire);
                if (saw_abort_drained_q)
                    saw_restart_begin_q <= 1'b1;
            end
            if (formal_legacy_q8_cfg_fire)
                saw_legacy_q8_cfg_q <= 1'b1;
            if (formal_legacy_q8_cfg_fire)
                assert(formal_legacy_q8_cfg_ready);
            if (formal_inject_abort) begin
                assert(formal_legacy_q8_cfg_pending);
                assert(formal_abort_strobe);
                assert(!formal_legacy_q8_cfg_fire);
            end

            if (formal_legacy_q8_cfg_pending) begin
                assert(formal_section_active);
                if (formal_abort_strobe || formal_abort_cleanup)
                    assert(!formal_legacy_q8_cfg_fire);
                else
                    assert(formal_legacy_q8_cfg_fire);
            end else begin
                assert(!formal_legacy_q8_cfg_fire);
            end
            if (f_past_valid && prev_rst_n_q &&
                prev_section_begin_ok_q) begin
                assert(formal_legacy_q8_cfg_pending);
                if (!formal_abort_strobe)
                    assert(formal_legacy_q8_cfg_fire);
            end
            if (f_past_valid && prev_rst_n_q &&
                prev_legacy_q8_cfg_pending_q &&
                !prev_section_begin_ok_q)
                assert(!formal_legacy_q8_cfg_pending);

            if ((clean_milestone_q == 3'd1) && formal_ffn_gate_ready)
                clean_milestone_q <= 3'd2;
            if ((clean_milestone_q == 3'd2) && capture_fire &&
                (formal_capture_beat == 3'd4))
                clean_milestone_q <= 3'd3;
            if ((clean_milestone_q == 3'd3) &&
                (formal_ffn_phase == 3'd5))
                clean_milestone_q <= 3'd4;
            if ((clean_milestone_q == 3'd4) &&
                (formal_ffn_phase == 3'd6) && formal_producer_done)
                clean_milestone_q <= 3'd5;
            if ((clean_milestone_q == 3'd5) && replay_fire &&
                (formal_replay_beat == 3'd4))
                clean_milestone_q <= 3'd6;
            if ((clean_milestone_q == 3'd6) &&
                formal_section_done && !formal_section_active)
                clean_milestone_q <= 3'd7;

            if (formal_gate_start) begin
                capture_beats_q <= 5'd0;
                capture_records_q <= 3'd0;
                capture_expect_beat_q <= 3'd0;
                capture_expect_block_q <= 3'd0;
            end else if (capture_fire) begin
                assert(capture_beats_q < 5'd20);
                assert(formal_capture_token == 2'd0);
                assert(formal_capture_beat == capture_expect_beat_q);
                assert(formal_capture_block == capture_expect_block_q);
                capture_beats_q <= capture_beats_q + 1'b1;
                if (capture_expect_beat_q == 3'd4) begin
                    capture_expect_beat_q <= 3'd0;
                    capture_expect_block_q <= capture_expect_block_q + 1'b1;
                end else begin
                    capture_expect_beat_q <= capture_expect_beat_q + 1'b1;
                end
            end
            if (formal_capture_record_done) begin
                assert(capture_records_q < 3'd4);
                capture_records_q <= capture_records_q + 1'b1;
            end

            if (formal_down_start) begin
                replay_beats_q <= 5'd0;
                replay_expect_beat_q <= 3'd0;
                replay_expect_block_q <= 3'd0;
            end else if (replay_fire) begin
                assert(replay_beats_q < 5'd20);
                assert(formal_replay_token == 2'd0);
                assert(formal_replay_beat == replay_expect_beat_q);
                assert(formal_replay_block == replay_expect_block_q);
                replay_beats_q <= replay_beats_q + 1'b1;
                if (replay_expect_beat_q == 3'd4) begin
                    replay_expect_beat_q <= 3'd0;
                    replay_expect_block_q <= replay_expect_block_q + 1'b1;
                end else begin
                    replay_expect_beat_q <= replay_expect_beat_q + 1'b1;
                end
            end

            if (formal_producer_done && !(|formal_scratch_error)) begin
                assert(capture_beats_q == 5'd20);
                assert(capture_records_q == 3'd4);
                assert(formal_bank0_record_count == 11'd4);
            end
            if (formal_section_done && !formal_section_active &&
                !(|formal_scratch_error))
                assert(replay_beats_q == 5'd20);

            if (abort_case && !did_abort_q &&
                (formal_scratch_rd_owner == 2'd2))
                saw_abort_owner_q <= 1'b1;
            if (saw_abort_owner_q && formal_abort_cleanup &&
                formal_section_active &&
                (formal_scratch_rd_owner == 2'd2))
                saw_abort_hold_q <= 1'b1;
            if (saw_abort_hold_q && !formal_abort_cleanup &&
                !formal_section_active &&
                (formal_scratch_rd_owner == 2'd0) &&
                !formal_scratch_rd_rsp_valid)
                saw_abort_drained_q <= 1'b1;

            if (f_past_valid && prev_rst_n_q &&
                (prev_owner_q != 2'd0) &&
                !prev_rsp_fire_q)
                assert(formal_scratch_rd_owner == prev_owner_q);

            if (f_past_valid && prev_rst_n_q && prev_section_active_q &&
                !formal_section_active &&
                !prev_abort_cleanup_q && !prev_any_error_q) begin
                assert(prev_phase_q == 3'd7);
                assert(prev_replay_complete_q);
                assert(prev_down_kernel_done_q || prev_kernel_done_q);
            end

            if (f_past_valid && prev_rst_n_q && !prev_abort_strobe_q &&
                !prev_abort_cleanup_q && !prev_fault_q) begin
                case (prev_phase_q)
                    3'd0: assert((formal_ffn_phase == 3'd0) ||
                                 (formal_ffn_phase == 3'd1));
                    3'd1: assert((formal_ffn_phase == 3'd1) ||
                                 (formal_ffn_phase == 3'd2));
                    3'd2: assert((formal_ffn_phase == 3'd2) ||
                                 (formal_ffn_phase == 3'd3));
                    3'd3: assert((formal_ffn_phase == 3'd3) ||
                                 (formal_ffn_phase == 3'd4));
                    3'd4: assert((formal_ffn_phase == 3'd4) ||
                                 (formal_ffn_phase == 3'd5));
                    3'd5: assert((formal_ffn_phase == 3'd5) ||
                                 (formal_ffn_phase == 3'd6));
                    3'd6: assert((formal_ffn_phase == 3'd6) ||
                                 (formal_ffn_phase == 3'd7));
                    3'd7: assert((formal_ffn_phase == 3'd7) ||
                                 (formal_ffn_phase == 3'd0));
                endcase

                if (formal_ffn_phase != prev_phase_q) begin
                    case (formal_ffn_phase)
                        3'd0: assert((prev_phase_q == 3'd7) &&
                                     prev_down_complete_ready_q);
                        3'd1: assert((prev_phase_q == 3'd0) &&
                                     prev_section_begin_ok_q);
                        3'd2: assert((prev_phase_q == 3'd1) &&
                                     prev_up_start_q);
                        3'd3: assert((prev_phase_q == 3'd2) &&
                                     prev_kernel_done_q);
                        3'd4: assert((prev_phase_q == 3'd3) &&
                                     prev_gate_start_q);
                        3'd5: assert((prev_phase_q == 3'd4) &&
                                     prev_gate_drain_ready_q);
                        3'd6: assert((prev_phase_q == 3'd5) &&
                                     prev_seal_done_q);
                        3'd7: assert((prev_phase_q == 3'd6) &&
                                     prev_down_start_q);
                    endcase
                end
            end
        end

`ifdef FORMAL_COVER_CLEAN
        cover(rst_n && saw_legacy_q8_cfg_q &&
              (clean_milestone_q == 3'd7));
`elsif FORMAL_COVER_ABORT
        cover(rst_n && abort_case && saw_abort_owner_q && saw_abort_hold_q &&
              saw_abort_drained_q && saw_restart_begin_q &&
              saw_legacy_q8_cfg_q &&
              (clean_milestone_q == 3'd7));
`endif
    end

    wire _unused = &{1'b0, bresp, bvalid, awready, wready, rdata, rresp,
                     arready, rvalid, w0_ready, w1_ready, w2_ready, w3_ready,
                     acts_ready, result_data, result_keep, result_valid,
                     result_last, command_q, slot_phase_q};
endmodule

`default_nettype wire
