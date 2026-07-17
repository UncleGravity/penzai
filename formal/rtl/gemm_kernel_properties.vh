    reg f_gemm_past_valid = 1'b0;
    reg f_run_active = 1'b0;
    reg [15:0] f_q1_blocks = 16'd0;
    reg [15:0] f_rowblocks = 16'd0;
    reg [15:0] f_cols = 16'd0;
    reg [31:0] f_output_beats = 32'd0;
    reg [1:0] f_weight_fmt = 2'd1;

    always @(posedge clk) begin
        f_gemm_past_valid <= 1'b1;

        if (!rst_n) begin
            f_run_active <= 1'b0;
            f_q1_blocks <= 16'd0;
            f_rowblocks <= 16'd0;
            f_cols <= 16'd0;
            f_output_beats <= 32'd0;
            f_weight_fmt <= 2'd1;
        end else begin
            if (start_pulse) begin
                f_run_active <= 1'b1;
                f_q1_blocks <= num_q1_blocks;
                f_rowblocks <= num_rowblocks;
                f_cols <= num_cols;
                f_output_beats <= 32'd0;
                f_weight_fmt <= weight_fmt;
            end else if (kernel_done) begin
                f_run_active <= 1'b0;
            end
            if (m_axis_tvalid && m_axis_tready)
                f_output_beats <= f_output_beats + 32'd1;
        end

        if (f_gemm_past_valid && !$past(rst_n)) begin
            assert(!busy_q && !kernel_done);
            assert(!m_axis_tvalid);
        end

        if (rst_n) begin
            assert(state <= ST_TCODE1);
            assert(busy_q == (state != ST_IDLE));
            if (busy_q) begin
                assert(f_run_active);
                assert(run_num_q1_blocks == f_q1_blocks);
                assert(run_num_rowblocks == f_rowblocks);
                assert(run_num_cols == f_cols);
                assert(run_weight_fmt == f_weight_fmt);
                assert(last_q1 == (q1_idx_wide + 16'd1 == f_q1_blocks));
                assert(last_col == (col + 16'd1 == f_cols));
                assert(n_total == f_cols * EMIT_BEATS[15:0]);
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
            if (state == ST_EMIT) begin
                assert(emit_col < f_cols);
                assert(rowblock_remaining > 0 && rowblock_remaining <= f_rowblocks);
                assert(rd_idx < BUF_DEPTH);
            end

            assert(f_output_beats <= f_rowblocks * f_cols * EMIT_BEATS[15:0]);
            if (m_axis_tlast) begin
                assert(m_axis_tvalid);
                assert(f_output_beats + 32'd1 == f_rowblocks * f_cols * EMIT_BEATS[15:0]);
            end
            if (kernel_done)
                assert(f_output_beats == f_rowblocks * f_cols * EMIT_BEATS[15:0]);

            if (f_gemm_past_valid && $past(rst_n) &&
                $past(m_axis_tvalid && !m_axis_tready)) begin
                assert(m_axis_tvalid);
                assert(m_axis_tdata == $past(m_axis_tdata));
                assert(m_axis_tkeep == $past(m_axis_tkeep));
                assert(m_axis_tlast == $past(m_axis_tlast));
            end
        end

        cover(rst_n && state == ST_EMIT && !m_axis_tready);
        cover(rst_n && kernel_done);
        cover(rst_n && busy_q &&
              (num_q1_blocks != f_q1_blocks || num_rowblocks != f_rowblocks || num_cols != f_cols));
    end
