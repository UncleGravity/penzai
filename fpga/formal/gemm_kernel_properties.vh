    reg f_gemm_past_valid = 1'b0;
    reg f_run_active = 1'b0;
    reg [15:0] f_q1_blocks = 16'd0;
    reg [15:0] f_rowblocks = 16'd0;
    reg [31:0] f_rows = 32'd0;
    reg [15:0] f_cols = 16'd0;
    reg [31:0] f_output_beats = 32'd0;
    reg [31:0] f_output_bytes = 32'd0;
    reg [1:0] f_weight_fmt = 2'd1;
    reg [1:0] f_act_mode = ACT_PACKED_LOAD;
    reg [31:0] f_pc_present_count = 32'd0;
    reg [31:0] f_pc_collect_count = 32'd0;

    wire [31:0] f_effective_rows = (f_rows == 0) ? f_rowblocks * ROWS : f_rows;
    wire [31:0] f_final_rows = f_effective_rows - ((f_rowblocks - 1'b1) * ROWS);
    wire [31:0] f_final_beats = (f_final_rows + 1'b1) >> 1;
    wire [31:0] f_expected_beats = ((f_rowblocks - 1'b1) * f_cols * EMIT_BEATS) +
        (f_cols * f_final_beats);
    wire [31:0] f_expected_bytes = f_effective_rows * f_cols * 4;
    wire [3:0] f_transfer_bytes = (m_axis_tkeep == 8'h0F) ? 4'd4 : 4'd8;

    always @(posedge clk) begin
        f_gemm_past_valid <= 1'b1;

        if (!rst_n) begin
            f_run_active <= 1'b0;
            f_q1_blocks <= 16'd0;
            f_rowblocks <= 16'd0;
            f_rows <= 32'd0;
            f_cols <= 16'd0;
            f_output_beats <= 32'd0;
            f_output_bytes <= 32'd0;
            f_weight_fmt <= 2'd1;
            f_act_mode <= ACT_PACKED_LOAD;
            f_pc_present_count <= 32'd0;
            f_pc_collect_count <= 32'd0;
        end else begin
            if (start_pulse) begin
                f_run_active <= 1'b1;
                f_q1_blocks <= num_q1_blocks;
                f_rowblocks <= num_rowblocks;
                f_rows <= num_rows;
                f_cols <= num_cols;
                f_output_beats <= 32'd0;
                f_output_bytes <= 32'd0;
                f_weight_fmt <= weight_fmt;
                f_act_mode <= act_mode;
                f_pc_present_count <= 32'd0;
                f_pc_collect_count <= 32'd0;
            end else if (kernel_done) begin
                f_run_active <= 1'b0;
            end
            if (pc_walk_start) begin
                f_pc_present_count <= 32'd0;
                f_pc_collect_count <= 32'd0;
            end else begin
                if (pc_presenting)
                    f_pc_present_count <= f_pc_present_count + 32'd1;
                if (emit_collect_fire)
                    f_pc_collect_count <= f_pc_collect_count + 32'd1;
            end
            if (m_axis_tvalid && m_axis_tready) begin
                f_output_beats <= f_output_beats + 32'd1;
                f_output_bytes <= f_output_bytes + f_transfer_bytes;
            end
        end

        if (f_gemm_past_valid && !$past(rst_n)) begin
            assert(!busy_q && !kernel_done);
            assert(!m_axis_tvalid);
        end

        if (rst_n) begin
            assert(state <= ST_ERROR);
            assert(busy_q == (state != ST_IDLE));
            if (busy_q) begin
                assert(f_run_active);
                assert(run_num_q1_blocks == f_q1_blocks);
                assert(run_num_rowblocks == f_rowblocks);
                assert(run_num_rows == f_rows);
                assert(run_num_cols == f_cols);
                assert(run_weight_fmt == f_weight_fmt);
                assert(run_act_mode == f_act_mode);
                assert(last_q1 == (q1_idx_wide + 16'd1 == f_q1_blocks));
                assert(last_col == (col + 16'd1 == f_cols));
                assert(n_total == f_cols * EMIT_BEATS[15:0]);
            end

            if (state == ST_ERROR) begin
                assert(activation_error);
                assert(!s_axis_tready && !s_axis_acts_tready && !m_axis_tvalid);
            end
            if (f_run_active && f_act_mode == ACT_REUSE && !activation_error) begin
                assert(!s_axis_acts_tready);
            end

            if (state == ST_LOAD_ACTS || state == ST_LOAD_ASCALE) begin
                assert(acts_load_q1 < f_q1_blocks);
                assert(acts_load_col < f_cols);
                assert(acts_load_sub < 4);
                assert(acts_load_addr < ACT_DEPTH);
            end
            if (state == ST_WISSUE) begin
                assert(q1_idx_wide < f_q1_blocks);
                assert(col < f_cols);
                assert(issue_addr < ACT_DEPTH);
            end
            if (state == ST_PRECOMPUTE) begin
                assert(pc_col <= f_cols);
                assert(wr_idx <= f_cols * EMIT_BEATS[15:0]);
            end
            if (pc_walk_active_q) begin
                assert(state == ST_PRECOMPUTE);
                assert(pc_col < f_cols);
                assert(pc_beat < EMIT_BEATS);
                assert((pc_col * EMIT_BEATS) + pc_beat ==
                       f_pc_present_count);
            end
            assert(f_pc_present_count <= f_cols * EMIT_BEATS);
            assert(f_pc_collect_count <= f_pc_present_count);
            if (state == ST_PRECOMPUTE)
                assert(wr_idx == f_pc_collect_count[WRW-1:0]);
            assert(emit_vo == emit_vo_hi);
            if (activation_abort)
                assert(!emit_collect_fire);
            if (f_gemm_past_valid &&
                $past(rst_n && pc_walk_start && !activation_abort)) begin
                assert(pc_walk_active_q);
                assert(pc_col == 16'd0);
                assert(pc_beat == {EBW{1'b0}});
            end
            if (state == ST_EMIT) begin
                assert(emit_col < f_cols);
                assert(rowblock_remaining > 0 && rowblock_remaining <= f_rowblocks);
                assert(emit_beat <= active_emit_last);
                assert(active_emit_last < EMIT_BEATS);
                assert(rd_idx < BUF_DEPTH);
            end

            if (m_axis_tvalid) begin
                assert(!activation_error);
                assert(m_axis_tkeep == 8'hFF || m_axis_tkeep == 8'h0F);
                assert((m_axis_tkeep == 8'h0F) ==
                    ((rowblock_remaining == 16'd1) && (emit_beat == active_emit_last) && f_rows[0]));
                assert(m_axis_tlast == ((rowblock_remaining == 16'd1) &&
                    (emit_col + 16'd1 == f_cols) && (emit_beat == active_emit_last)));
            end

            assert(f_output_beats <= f_expected_beats);
            assert(f_output_bytes <= f_expected_bytes);
            if (m_axis_tlast) begin
                assert(m_axis_tvalid);
                assert(f_output_beats + 32'd1 == f_expected_beats);
                assert(f_output_bytes + f_transfer_bytes == f_expected_bytes);
            end
            if (kernel_done) begin
                if (activation_error) begin
                    assert(f_output_beats == 0);
                    assert(f_output_bytes == 0);
                end else begin
                    assert(f_output_beats == f_expected_beats);
                    assert(f_output_bytes == f_expected_bytes);
                end
            end

            if (f_gemm_past_valid && $past(rst_n) &&
                $past(m_axis_tvalid && !m_axis_tready)) begin
                assert(m_axis_tvalid);
                assert(m_axis_tdata == $past(m_axis_tdata));
                assert(m_axis_tkeep == $past(m_axis_tkeep));
                assert(m_axis_tlast == $past(m_axis_tlast));
            end
            if (f_gemm_past_valid && $past(rst_n) &&
                $past(activation_abort && !start_pulse)) begin
                assert(!activation_valid);
                assert(activation_error);
            end
            if (f_gemm_past_valid && $past(rst_n && busy_q &&
                                           state == ST_ERROR)) begin
                assert(kernel_done);
                assert(!busy_q);
            end
        end

`ifndef GEMM_FORMAL_DISABLE_GENERIC_COVERS
        cover(rst_n && state == ST_EMIT && !m_axis_tready);
        cover(rst_n && f_run_active && f_rows == 0 && m_axis_tlast);
        cover(rst_n && m_axis_tvalid && m_axis_tready && m_axis_tkeep == 8'h0F && m_axis_tlast);
        cover(rst_n && kernel_done);
        cover(rst_n && pc_presenting && pc_col == 16'd0 &&
              pc_beat == {EBW{1'b0}});
        cover(rst_n && pc_presenting && pc_col + 16'd1 == f_cols &&
              pc_beat == EMIT_LAST[EBW-1:0]);
        cover(rst_n && busy_q &&
              (num_q1_blocks != f_q1_blocks || num_rowblocks != f_rowblocks || num_cols != f_cols));
`endif
    end
