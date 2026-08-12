    // Control-only scoreboards. Arithmetic values are abstracted by the formal
    // harness, but every request and completion must retain its ordered tag.
    reg f_flash_past_valid = 1'b0;
    reg f_pair_active = 1'b0;
    reg f_dot_req_done = 1'b0;
    reg f_dot_cap_done = 1'b0;
    reg f_ax_req_done = 1'b0;
    reg f_ax_wb_done = 1'b0;
    reg [15:0] f_dot_req_head = 0, f_dot_req_kvh = 0;
    reg [15:0] f_dot_req_ratio = 0, f_dot_req_beat = 0;
    reg [15:0] f_dot_cap_head = 0, f_dot_cap_beat = 0;
    reg [15:0] f_score_count = 0, f_tree_count = 0;
    reg [15:0] f_soft_launch_count = 0, f_soft_issue_count = 0;
    reg [15:0] f_soft_commit_count = 0;
    reg [15:0] f_ax_req_head = 0, f_ax_req_kvh = 0;
    reg [15:0] f_ax_req_ratio = 0, f_ax_req_beat = 0;
    reg [15:0] f_ax_wb_head = 0, f_ax_wb_beat = 0;
    reg [31:0] f_dot_issue_count = 0, f_dot_capture_count = 0;
    reg [31:0] f_ax_issue_count = 0, f_ax_commit_count = 0;
    reg [MAX_HEADS-1:0] f_score_ready = 0;
    reg [3:0] f_soft_issue_history = 0;
    integer f_h;

    wire f_soft_launch = (state == S_PAIR) && (soft_hold == 0) &&
                         score_ready[soft_req_head[HW-1:0]];

    always @(posedge clk) begin
        f_flash_past_valid <= 1'b1;

        if (!rst_n) begin
            f_pair_active <= 1'b0;
            f_dot_req_done <= 1'b0;
            f_dot_cap_done <= 1'b0;
            f_ax_req_done <= 1'b0;
            f_ax_wb_done <= 1'b0;
            f_dot_req_head <= 0; f_dot_req_kvh <= 0;
            f_dot_req_ratio <= 0; f_dot_req_beat <= 0;
            f_dot_cap_head <= 0; f_dot_cap_beat <= 0;
            f_score_count <= 0; f_tree_count <= 0;
            f_soft_launch_count <= 0; f_soft_issue_count <= 0;
            f_soft_commit_count <= 0;
            f_ax_req_head <= 0; f_ax_req_kvh <= 0;
            f_ax_req_ratio <= 0; f_ax_req_beat <= 0;
            f_ax_wb_head <= 0; f_ax_wb_beat <= 0;
            f_dot_issue_count <= 0; f_dot_capture_count <= 0;
            f_ax_issue_count <= 0; f_ax_commit_count <= 0;
            f_score_ready <= 0;
            f_soft_issue_history <= 0;
        end else begin
            f_soft_issue_history <= {f_soft_issue_history[2:0], soft_fire};
            // The last K beat initializes all production and shadow tags.
            if ((state == S_KLOAD) && k_tvalid &&
                (ld_b + 16'd1 == qbeats) &&
                (ld_a + 16'd1 == n_head_kv_r)) begin
                f_pair_active <= 1'b1;
                f_dot_req_done <= 1'b0;
                f_dot_cap_done <= 1'b0;
                f_ax_req_done <= 1'b0;
                f_ax_wb_done <= 1'b0;
                f_dot_req_head <= 0; f_dot_req_kvh <= 0;
                f_dot_req_ratio <= 0; f_dot_req_beat <= 0;
                f_dot_cap_head <= 0; f_dot_cap_beat <= 0;
                f_score_count <= 0; f_tree_count <= 0;
                f_soft_launch_count <= 0; f_soft_issue_count <= 0;
                f_soft_commit_count <= 0;
                f_ax_req_head <= 0; f_ax_req_kvh <= 0;
                f_ax_req_ratio <= 0; f_ax_req_beat <= 0;
                f_ax_wb_head <= 0; f_ax_wb_beat <= 0;
                f_dot_issue_count <= 0; f_dot_capture_count <= 0;
                f_ax_issue_count <= 0; f_ax_commit_count <= 0;
                f_score_ready <= 0;
            end

            if (dot_fire) begin
                f_dot_issue_count <= f_dot_issue_count + 1'b1;
                if (f_dot_req_beat + 16'd1 == qbeats) begin
                    f_dot_req_beat <= 0;
                    if (f_dot_req_head + 16'd1 == n_heads_r)
                        f_dot_req_done <= 1'b1;
                    else begin
                        f_dot_req_head <= f_dot_req_head + 1'b1;
                        if (f_dot_req_ratio + 16'd1 == head_ratio_r) begin
                            f_dot_req_ratio <= 0;
                            f_dot_req_kvh <= f_dot_req_kvh + 1'b1;
                        end else f_dot_req_ratio <= f_dot_req_ratio + 1'b1;
                    end
                end else f_dot_req_beat <= f_dot_req_beat + 1'b1;
            end

            if (dot_v) begin
                f_dot_capture_count <= f_dot_capture_count + 1'b1;
                if (f_dot_cap_beat + 16'd1 == qbeats) begin
                    f_dot_cap_beat <= 0;
                    if (f_dot_cap_head + 16'd1 == n_heads_r)
                        f_dot_cap_done <= 1'b1;
                    else f_dot_cap_head <= f_dot_cap_head + 1'b1;
                end else f_dot_cap_beat <= f_dot_cap_beat + 1'b1;
            end

            if (tree_fire) f_tree_count <= f_tree_count + 1'b1;
            if (sadd_v) begin
                f_score_count <= f_score_count + 1'b1;
                f_score_ready[score_head[HW-1:0]] <= 1'b1;
            end
            if (f_soft_launch) begin
                f_soft_launch_count <= f_soft_launch_count + 1'b1;
                f_score_ready[soft_req_head[HW-1:0]] <= 1'b0;
            end
            if (soft_fire) f_soft_issue_count <= f_soft_issue_count + 1'b1;
            if (soft_v) f_soft_commit_count <= f_soft_commit_count + 1'b1;

            if (axpy_fire) begin
                f_ax_issue_count <= f_ax_issue_count + 1'b1;
                if (f_ax_req_beat + 16'd1 == vbeats) begin
                    f_ax_req_beat <= 0;
                    if (f_ax_req_head + 16'd1 == n_heads_r)
                        f_ax_req_done <= 1'b1;
                    else begin
                        f_ax_req_head <= f_ax_req_head + 1'b1;
                        if (f_ax_req_ratio + 16'd1 == head_ratio_r) begin
                            f_ax_req_ratio <= 0;
                            f_ax_req_kvh <= f_ax_req_kvh + 1'b1;
                        end else f_ax_req_ratio <= f_ax_req_ratio + 1'b1;
                    end
                end else f_ax_req_beat <= f_ax_req_beat + 1'b1;
            end

            if ((state == S_PAIR) && ax_v) begin
                f_ax_commit_count <= f_ax_commit_count + 1'b1;
                if (f_ax_wb_beat + 16'd1 == vbeats) begin
                    f_ax_wb_beat <= 0;
                    if (f_ax_wb_head + 16'd1 == n_heads_r)
                        f_ax_wb_done <= 1'b1;
                    else f_ax_wb_head <= f_ax_wb_head + 1'b1;
                end else f_ax_wb_beat <= f_ax_wb_beat + 1'b1;
            end

            if (state == S_KVNEXT)
                f_pair_active <= 1'b0;
        end

        if (f_flash_past_valid && !$past(rst_n)) begin
            assert(state == S_IDLE);
            assert(!busy && !done);
        end

        if (rst_n) begin
            assert(state == S_IDLE || state == S_QLOAD || state == S_TINIT ||
                state == S_MASK || state == S_KLOAD || state == S_PAIR ||
                state == S_KSKIP || state == S_VSKIP || state == S_KVNEXT ||
                state == S_FRECF || state == S_FRECW || state == S_FEMITF ||
                state == S_FEMITW || state == S_FEMITE || state == S_TNEXT ||
                state == S_DONE);
            assert(busy == (state != S_IDLE));
            assert(score_ready == f_score_ready);

            if (f_flash_past_valid && $past(rst_n) &&
                $past(state == S_IDLE && start)) begin
                assert(head_dim_q_r == $past(head_dim_q));
                assert(head_dim_v_r == $past(head_dim_v));
                assert(n_heads_r == $past(n_heads));
                assert(n_head_kv_r == $past(n_head_kv));
                assert(head_ratio_r == $past(head_ratio));
                assert(n_kv_r == $past(n_kv));
                assert(n_tokens_r == $past(n_tokens));
                assert(scale_r == $past(scale));
            end

            // Once accepted, the active configuration is immutable even if the
            // wrapper-facing inputs change while the kernel is busy.
            if (f_flash_past_valid && $past(rst_n) && $past(busy)) begin
                assert(head_dim_q_r == $past(head_dim_q_r));
                assert(head_dim_v_r == $past(head_dim_v_r));
                assert(n_heads_r == $past(n_heads_r));
                assert(n_head_kv_r == $past(n_head_kv_r));
                assert(head_ratio_r == $past(head_ratio_r));
                assert(n_kv_r == $past(n_kv_r));
                assert(n_tokens_r == $past(n_tokens_r));
                assert(scale_r == $past(scale_r));
            end

            if (busy) begin
                assert(qbeats > 0 && qbeats <= MAXB);
                assert(vbeats > 0 && vbeats <= MAXB);
                assert(n_heads_r > 0 && n_heads_r <= MAX_HEADS);
                assert(n_head_kv_r > 0 && n_head_kv_r <= MAX_HEAD_KV);
                assert(head_ratio_r > 0);
                assert(n_kv_r > 0 && n_tokens_r > 0);
            end

            if ((state == S_QLOAD) && q_tvalid) begin
                assert(ld_a < n_heads_r && ld_b < qbeats);
                assert(q_ld_addr == ld_a * MAXB + ld_b);
            end
            if (state == S_TINIT) begin
                assert(ld_a < n_heads_r && ld_b < vbeats);
                assert(q_ld_addr == ld_a * MAXB + ld_b);
            end
            if ((state == S_KLOAD) && k_tvalid) begin
                assert(ld_a < n_head_kv_r && ld_b < qbeats);
                assert(kv_ld_addr == ld_a * MAXB + ld_b);
            end
            if ((state == S_PAIR) && v_tvalid && v_tready) begin
                assert(ld_a < n_head_kv_r && ld_b < vbeats);
                assert(kv_ld_addr == ld_a * MAXB + ld_b);
            end

            if (dot_fire) begin
                assert(f_pair_active && !f_dot_req_done);
                assert(dot_req_head == f_dot_req_head);
                assert(dot_req_kvh == f_dot_req_kvh);
                assert(dot_req_ratio == f_dot_req_ratio);
                assert(dot_req_beat == f_dot_req_beat);
                assert(dot_req_head < n_heads_r);
                assert(dot_req_kvh < n_head_kv_r);
                assert(dot_req_ratio < head_ratio_r);
                assert(dot_req_beat < qbeats);
                assert(dot_req_head == dot_req_kvh * head_ratio_r + dot_req_ratio);
                assert(q_dot_addr < MAX_HEADS * MAXB);
                assert(k_dot_addr < MAX_HEAD_KV * MAXB);
                assert(q_dot_addr == dot_req_head * MAXB + dot_req_beat);
                assert(k_dot_addr == dot_req_kvh * MAXB + dot_req_beat);
            end

            if (dot_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(f_dot_capture_count < f_dot_issue_count);
                assert(dot_cap_head == f_dot_cap_head);
                assert(dot_cap_beat == f_dot_cap_beat);
                assert(dot_cap_head < n_heads_r);
                assert(dot_cap_beat < qbeats);
            end

            if (tree_fire) begin
                assert(state == S_PAIR && f_pair_active);
                assert(f_tree_count < n_heads_r);
                assert(f_tree_count < f_dot_cap_head +
                    ((f_dot_cap_beat != 0 || f_dot_cap_done) ? 16'd1 : 16'd0));
                if (dot_v)
                    assert(dot_cap_beat == 0);
            end

            if (sadd_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(score_head == f_score_count);
                assert(f_score_count < n_heads_r);
                assert(f_score_count < f_tree_count);
                assert(!f_score_ready[score_head[HW-1:0]]);
            end
            if (f_soft_launch) begin
                assert(f_pair_active);
                assert(soft_req_head == f_soft_launch_count);
                assert(f_soft_launch_count < f_score_count);
                assert(f_score_ready[soft_req_head[HW-1:0]]);
            end
            if (soft_fire) begin
                assert(state == S_PAIR && f_pair_active);
                assert($past(f_soft_launch));
                assert(f_soft_issue_count < f_soft_launch_count);
                // flash_softmax consumes its input tuple across four composition
                // seams. A fifth-cycle issue is safe; the controller currently
                // leaves eight clocks between accepted tuples.
                assert(f_soft_issue_history == 0);
            end
            if (soft_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(soft_wb_head == f_soft_commit_count);
                assert(f_soft_commit_count < f_soft_issue_count);
                assert(soft_wb_head < n_heads_r);
            end

            if (axpy_fire) begin
                assert(f_pair_active && v_load_done && !f_ax_req_done);
                assert(pc_ready[ax_req_head[HW-1:0]]);
                assert(ax_req_head == f_ax_req_head);
                assert(ax_req_kvh == f_ax_req_kvh);
                assert(ax_req_ratio == f_ax_req_ratio);
                assert(ax_req_beat == f_ax_req_beat);
                assert(ax_req_head < n_heads_r);
                assert(ax_req_kvh < n_head_kv_r);
                assert(ax_req_ratio < head_ratio_r);
                assert(ax_req_beat < vbeats);
                assert(ax_req_head == ax_req_kvh * head_ratio_r + ax_req_ratio);
                assert(acc_ax_rd < MAX_HEADS * MAXB);
                assert(v_ax_addr < MAX_HEAD_KV * MAXB);
                assert(acc_ax_rd == ax_req_head * MAXB + ax_req_beat);
                assert(v_ax_addr == ax_req_kvh * MAXB + ax_req_beat);
            end

            if ((state == S_PAIR) && ax_v) begin
                assert(f_pair_active);
                assert(f_ax_commit_count < f_ax_issue_count);
                assert(ax_wb_head == f_ax_wb_head);
                assert(ax_wb_beat == f_ax_wb_beat);
                assert(ax_wb_head < n_heads_r);
                assert(ax_wb_beat < vbeats);
                assert(acc_ax_wr < MAX_HEADS * MAXB);
                assert(acc_ax_wr == ax_wb_head * MAXB + ax_wb_beat);
            end

            if (emit_fire) begin
                assert(head_i < n_heads_r && bi < vbeats);
                assert(acc_em_rd == head_i * MAXB + bi);
            end

            // The first KVNEXT cycle is after the final registered accumulator
            // writeback. No score or AXPY work may remain outstanding there.
            if (state == S_KVNEXT && f_pair_active) begin
                assert(dot_req_done && f_dot_req_done && f_dot_cap_done);
                assert(f_tree_count == n_heads_r);
                assert(f_score_count == n_heads_r);
                assert(f_soft_launch_count == n_heads_r);
                assert(f_soft_issue_count == n_heads_r);
                assert(f_soft_commit_count == n_heads_r);
                assert(f_score_ready == 0);
                assert(v_load_done);
                assert(ax_req_done && f_ax_req_done && f_ax_wb_done);
                assert(f_dot_issue_count == f_dot_capture_count);
                assert(f_ax_issue_count == f_ax_commit_count);
                for (f_h = 0; f_h < MAX_HEADS; f_h = f_h + 1)
                    if (f_h < n_heads_r) assert(pc_ready[f_h]);
            end

            if (f_flash_past_valid && $past(rst_n) &&
                $past(state == S_PAIR) && state == S_KVNEXT) begin
                assert($past(ax_v));
                assert($past(ax_wb_head + 16'd1 == n_heads_r));
                assert($past(ax_wb_beat + 16'd1 == vbeats));
            end
        end

        cover(rst_n && state == S_PAIR && tree_fire && dot_v);
        cover(rst_n && f_soft_issue_count == 2 && f_soft_commit_count == 0);
        cover(rst_n && state == S_KVNEXT && f_pair_active);
    end
