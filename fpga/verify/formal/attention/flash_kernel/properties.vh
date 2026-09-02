`ifdef FORMAL_PIPELINE_VALUES
    // The wide BRAM/fabric data and scalar delays are direct one-cycle register
    // contracts. Prove them with the SMT engine, which keeps symbolic memories as
    // arrays instead of bit-blasting their contents into the controller PDR cone.
    reg f_pipe_past_valid = 1'b0;
    always @(posedge clk) begin
        f_pipe_past_valid <= 1'b1;
        if (f_pipe_past_valid && $past(rst_n)) begin
            assert(q_rd_q == $past(q_rd));
            assert(k_rd_q == $past(k_rd));
            assert(v_rd_q == $past(v_rd));
            assert(acc_rd_q == $past(acc_rd));
            assert(ax_s1_q == $past(ax_s1));
            assert(ax_s1_qq == $past(ax_s1_q));
            assert(ax_p_q == $past(ax_p));
            assert(ax_p_qq == $past(ax_p_q));
        end
    end
`elsif FORMAL_ADAPTIVE_SLOTS
    // Isolated proof of the production adaptive slot function. The harness holds
    // the controller in QLOAD after accepting a legal command, leaving PDR with
    // only the snapshot and mapping cone rather than a 32-head datapath schedule.
    (* anyconst *) reg [15:0] f_slot_tok_a;
    (* anyconst *) reg [15:0] f_slot_tok_b;
    (* anyconst *) reg [15:0] f_slot_head_a;
    (* anyconst *) reg [15:0] f_slot_head_b;
    (* anyconst *) reg [15:0] f_slot_beat_a;
    (* anyconst *) reg [15:0] f_slot_beat_b;

    wire [SW-1:0] f_slot_a = slotIndex(f_slot_tok_a, f_slot_head_a);
    wire [SW-1:0] f_slot_b = slotIndex(f_slot_tok_b, f_slot_head_b);
    wire [AW-1:0] f_addr_a = {f_slot_a, f_slot_beat_a[LMAXB-1:0]};
    wire [AW-1:0] f_addr_b = {f_slot_b, f_slot_beat_b[LMAXB-1:0]};
    reg f_slot_past_valid = 1'b0;

    always @(posedge clk) begin
        f_slot_past_valid <= 1'b1;

        if (f_slot_past_valid && !$past(rst_n)) begin
            assert(state == S_IDLE);
            assert(!busy && !done);
        end

        if (rst_n && busy) begin
            assume(f_slot_tok_a < n_tokens_r);
            assume(f_slot_tok_b < n_tokens_r);
            assume(f_slot_head_a < n_heads_r);
            assume(f_slot_head_b < n_heads_r);
            assume(f_slot_beat_a < MAXB);
            assume(f_slot_beat_b < MAXB);

            assert(wide_heads_r == (n_heads_r > 16));
            assert((!wide_heads_r && n_heads_r <= 16 && n_tokens_r <= 4) ||
                   (wide_heads_r && n_heads_r <= 32 && n_tokens_r <= 2));
            assert(f_slot_a < MAX_SLOTS && f_slot_b < MAX_SLOTS);
            assert(f_addr_a < MAX_SLOTS * MAXB);
            assert(f_addr_b < MAX_SLOTS * MAXB);

            if (wide_heads_r) begin
                assert(f_slot_a == {f_slot_tok_a[0], f_slot_head_a[4:0]});
                assert(f_slot_b == {f_slot_tok_b[0], f_slot_head_b[4:0]});
            end else begin
                assert(f_slot_a == {f_slot_tok_a[1:0], f_slot_head_a[3:0]});
                assert(f_slot_b == {f_slot_tok_b[1:0], f_slot_head_b[3:0]});
            end

            if (f_slot_tok_a != f_slot_tok_b || f_slot_head_a != f_slot_head_b)
                assert(f_slot_a != f_slot_b);
            if (f_slot_tok_a != f_slot_tok_b || f_slot_head_a != f_slot_head_b ||
                f_slot_beat_a != f_slot_beat_b)
                assert(f_addr_a != f_addr_b);
        end

        cover(rst_n && busy && wide_heads_r && n_tokens_r == 2 &&
              f_slot_tok_a == 1 && f_slot_head_a == 31 &&
              f_slot_beat_a + 1 == MAXB && f_slot_a + 1 == MAX_SLOTS &&
              f_addr_a + 1 == MAX_SLOTS * MAXB);
        cover(rst_n && busy && !wide_heads_r && n_tokens_r == 4 &&
              f_slot_tok_a == 3 && f_slot_head_a == 15 &&
              f_slot_beat_a + 1 == MAXB && f_slot_a + 1 == MAX_SLOTS &&
              f_addr_a + 1 == MAX_SLOTS * MAXB);
    end
`else
`ifndef FORMAL_LIVENESS_ONLY
    // Control-only scoreboards for the KV-outer, query-blocked scheduler.
    // Floating-point values are abstracted by the harness; every ordered request
    // and completion still has to retain its (query, head, beat) identity.
    reg f_flash_past_valid = 1'b0;
    reg f_pair_active = 1'b0;
    reg [MAX_TOKENS-1:0] f_pair_mask = 0;
    reg [MAX_TOKENS-1:0] f_ever_finite = 0;

    reg [15:0] f_dot_req_tok = 0, f_dot_req_head = 0;
    reg [15:0] f_dot_req_kvh = 0, f_dot_req_ratio = 0, f_dot_req_beat = 0;
    reg [15:0] f_dot_cap_tok = 0, f_dot_cap_head = 0, f_dot_cap_beat = 0;
    reg [15:0] f_tree_tok = 0, f_tree_head = 0;
    reg [15:0] f_score_tok = 0, f_score_head = 0;
    reg [15:0] f_soft_launch_tok = 0, f_soft_launch_head = 0;
    reg [15:0] f_soft_wb_tok = 0, f_soft_wb_head = 0;
    reg [15:0] f_ax_req_tok = 0, f_ax_req_head = 0;
    reg [15:0] f_ax_req_kvh = 0, f_ax_req_ratio = 0, f_ax_req_beat = 0;
    reg [15:0] f_ax_wb_tok = 0, f_ax_wb_head = 0, f_ax_wb_beat = 0;

    // The reduced proof can issue at most N_TOKENS*N_HEADS*MAXB = 16
    // operations of any kind per KV. Keep scoreboard widths proportional to that
    // bound so PDR does not carry hundreds of unreachable counter bits.
    reg [7:0] f_dot_issue_count = 0, f_dot_capture_count = 0;
    reg [7:0] f_tree_count = 0, f_score_count = 0;
    reg [7:0] f_soft_launch_count = 0, f_soft_issue_count = 0;
    reg [7:0] f_soft_commit_count = 0;
    reg [7:0] f_ax_issue_count = 0, f_ax_commit_count = 0;
    reg [NSLOTS-1:0] f_score_ready = 0, f_pc_ready = 0;

    function automatic [15:0] fCountValid;
        input [MAX_TOKENS-1:0] valid;
        input [15:0] limit;
        integer t;
        begin
            fCountValid = 0;
            for (t = 0; t < MAX_TOKENS; t = t + 1)
                if (t < limit && valid[t]) fCountValid = fCountValid + 1'b1;
        end
    endfunction

    wire f_last_k_load = (state == S_KLOAD) && k_tvalid &&
                         (ld_b + 16'd1 == qbeats) &&
                         (ld_a + 16'd1 == n_head_kv_r);
    wire f_soft_launch = (state == S_PAIR) && score_ready[soft_req_slot];
    wire [15:0] f_valid_queries = fCountValid(f_pair_mask, n_tokens_r);
    wire [15:0] f_valid_slots = f_valid_queries * n_heads_r;

    always @(posedge clk) begin
        f_flash_past_valid <= 1'b1;

        if (!rst_n) begin
            f_pair_active <= 1'b0;
            f_pair_mask <= 0;
            f_ever_finite <= 0;
            f_dot_req_tok <= 0; f_dot_req_head <= 0;
            f_dot_req_kvh <= 0; f_dot_req_ratio <= 0; f_dot_req_beat <= 0;
            f_dot_cap_tok <= 0; f_dot_cap_head <= 0; f_dot_cap_beat <= 0;
            f_tree_tok <= 0; f_tree_head <= 0;
            f_score_tok <= 0; f_score_head <= 0;
            f_soft_launch_tok <= 0; f_soft_launch_head <= 0;
            f_soft_wb_tok <= 0; f_soft_wb_head <= 0;
            f_ax_req_tok <= 0; f_ax_req_head <= 0;
            f_ax_req_kvh <= 0; f_ax_req_ratio <= 0; f_ax_req_beat <= 0;
            f_ax_wb_tok <= 0; f_ax_wb_head <= 0; f_ax_wb_beat <= 0;
            f_dot_issue_count <= 0; f_dot_capture_count <= 0;
            f_tree_count <= 0; f_score_count <= 0;
            f_soft_launch_count <= 0; f_soft_issue_count <= 0;
            f_soft_commit_count <= 0;
            f_ax_issue_count <= 0; f_ax_commit_count <= 0;
            f_score_ready <= 0; f_pc_ready <= 0;
        end else begin
            if (state == S_IDLE && start)
                f_ever_finite <= 0;
            if (state == S_MASK && mask_tvalid && mask_tdata != F16_NEG_INF)
                f_ever_finite[ld_tok[TW-1:0]] <= 1'b1;

            // The final K beat freezes this KV's sparse query set and initializes
            // every ordered shadow tag at its first finite query.
            if (f_last_k_load) begin
                f_pair_active <= (first_valid_tok != n_tokens_r);
                f_pair_mask <= mask_finite;
                f_dot_req_tok <= first_valid_tok; f_dot_req_head <= 0;
                f_dot_req_kvh <= 0; f_dot_req_ratio <= 0; f_dot_req_beat <= 0;
                f_dot_cap_tok <= first_valid_tok; f_dot_cap_head <= 0; f_dot_cap_beat <= 0;
                f_tree_tok <= first_valid_tok; f_tree_head <= 0;
                f_score_tok <= first_valid_tok; f_score_head <= 0;
                f_soft_launch_tok <= first_valid_tok; f_soft_launch_head <= 0;
                f_soft_wb_tok <= first_valid_tok; f_soft_wb_head <= 0;
                f_ax_req_tok <= first_valid_tok; f_ax_req_head <= 0;
                f_ax_req_kvh <= 0; f_ax_req_ratio <= 0; f_ax_req_beat <= 0;
                f_ax_wb_tok <= first_valid_tok; f_ax_wb_head <= 0; f_ax_wb_beat <= 0;
                f_dot_issue_count <= 0; f_dot_capture_count <= 0;
                f_tree_count <= 0; f_score_count <= 0;
                f_soft_launch_count <= 0; f_soft_issue_count <= 0;
                f_soft_commit_count <= 0;
                f_ax_issue_count <= 0; f_ax_commit_count <= 0;
                // Production only consumes these scoreboards in S_PAIR and resets
                // them when a finite pair starts. An all-masked KV enters VSKIP
                // without touching the now-unused bits.
                if (first_valid_tok != n_tokens_r) begin
                    f_score_ready <= 0;
                    f_pc_ready <= 0;
                end
            end

            if (dot_fire) begin
                f_dot_issue_count <= f_dot_issue_count + 1'b1;
                if (f_dot_req_beat + 16'd1 == qbeats) begin
                    f_dot_req_beat <= 0;
                    if (f_dot_req_head + 16'd1 == n_heads_r) begin
                        if (nextValidToken(f_dot_req_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                            f_dot_req_tok <= nextValidToken(f_dot_req_tok, f_pair_mask, n_tokens_r);
                            f_dot_req_head <= 0; f_dot_req_kvh <= 0; f_dot_req_ratio <= 0;
                        end
                    end else begin
                        f_dot_req_head <= f_dot_req_head + 1'b1;
                        if (f_dot_req_ratio + 16'd1 == head_ratio_r) begin
                            f_dot_req_ratio <= 0; f_dot_req_kvh <= f_dot_req_kvh + 1'b1;
                        end else f_dot_req_ratio <= f_dot_req_ratio + 1'b1;
                    end
                end else f_dot_req_beat <= f_dot_req_beat + 1'b1;
            end

            if (dot_v) begin
                f_dot_capture_count <= f_dot_capture_count + 1'b1;
                if (f_dot_cap_beat + 16'd1 == qbeats) begin
                    f_dot_cap_beat <= 0;
                    if (f_dot_cap_head + 16'd1 == n_heads_r) begin
                        if (nextValidToken(f_dot_cap_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                            f_dot_cap_tok <= nextValidToken(f_dot_cap_tok, f_pair_mask, n_tokens_r);
                            f_dot_cap_head <= 0;
                        end
                    end else f_dot_cap_head <= f_dot_cap_head + 1'b1;
                end else f_dot_cap_beat <= f_dot_cap_beat + 1'b1;
            end

            if (tree_v) f_tree_count <= f_tree_count + 1'b1;
            if (smul_v) begin
                if (f_tree_head + 16'd1 == n_heads_r) begin
                    if (nextValidToken(f_tree_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                        f_tree_tok <= nextValidToken(f_tree_tok, f_pair_mask, n_tokens_r);
                        f_tree_head <= 0;
                    end
                end else f_tree_head <= f_tree_head + 1'b1;
            end

            if (sadd_v) begin
                f_score_count <= f_score_count + 1'b1;
                f_score_ready[score_slot] <= 1'b1;
                if (f_score_head + 16'd1 == n_heads_r) begin
                    if (nextValidToken(f_score_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                        f_score_tok <= nextValidToken(f_score_tok, f_pair_mask, n_tokens_r);
                        f_score_head <= 0;
                    end
                end else f_score_head <= f_score_head + 1'b1;
            end

            if (f_soft_launch) begin
                f_soft_launch_count <= f_soft_launch_count + 1'b1;
                f_score_ready[soft_req_slot] <= 1'b0;
                if (f_soft_launch_head + 16'd1 == n_heads_r) begin
                    if (nextValidToken(f_soft_launch_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                        f_soft_launch_tok <= nextValidToken(f_soft_launch_tok, f_pair_mask, n_tokens_r);
                        f_soft_launch_head <= 0;
                    end
                end else f_soft_launch_head <= f_soft_launch_head + 1'b1;
            end
            if (soft_fire) f_soft_issue_count <= f_soft_issue_count + 1'b1;

            if (soft_v) begin
                f_soft_commit_count <= f_soft_commit_count + 1'b1;
                f_pc_ready[soft_wb_slot] <= 1'b1;
                if (f_soft_wb_head + 16'd1 == n_heads_r) begin
                    if (nextValidToken(f_soft_wb_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                        f_soft_wb_tok <= nextValidToken(f_soft_wb_tok, f_pair_mask, n_tokens_r);
                        f_soft_wb_head <= 0;
                    end
                end else f_soft_wb_head <= f_soft_wb_head + 1'b1;
            end

            if (axpy_fire) begin
                f_ax_issue_count <= f_ax_issue_count + 1'b1;
                if (f_ax_req_beat + 16'd1 == vbeats) begin
                    f_ax_req_beat <= 0;
                    if (f_ax_req_head + 16'd1 == n_heads_r) begin
                        if (nextValidToken(f_ax_req_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                            f_ax_req_tok <= nextValidToken(f_ax_req_tok, f_pair_mask, n_tokens_r);
                            f_ax_req_head <= 0; f_ax_req_kvh <= 0; f_ax_req_ratio <= 0;
                        end
                    end else begin
                        f_ax_req_head <= f_ax_req_head + 1'b1;
                        if (f_ax_req_ratio + 16'd1 == head_ratio_r) begin
                            f_ax_req_ratio <= 0; f_ax_req_kvh <= f_ax_req_kvh + 1'b1;
                        end else f_ax_req_ratio <= f_ax_req_ratio + 1'b1;
                    end
                end else f_ax_req_beat <= f_ax_req_beat + 1'b1;
            end

            if (state == S_PAIR && ax_v) begin
                f_ax_commit_count <= f_ax_commit_count + 1'b1;
                if (f_ax_wb_beat + 16'd1 == vbeats) begin
                    f_ax_wb_beat <= 0;
                    if (f_ax_wb_head + 16'd1 == n_heads_r) begin
                        if (nextValidToken(f_ax_wb_tok, f_pair_mask, n_tokens_r) != n_tokens_r) begin
                            f_ax_wb_tok <= nextValidToken(f_ax_wb_tok, f_pair_mask, n_tokens_r);
                            f_ax_wb_head <= 0;
                        end
                    end else f_ax_wb_head <= f_ax_wb_head + 1'b1;
                end else f_ax_wb_beat <= f_ax_wb_beat + 1'b1;
            end

            if (state == S_KVNEXT) f_pair_active <= 1'b0;
        end

`ifndef FORMAL_COMPLETION_ONLY
        if (f_flash_past_valid && !$past(rst_n)) begin
            assert(state == S_IDLE);
            assert(!busy && !done && !o_tvalid);
        end

        if (rst_n) begin
            assert(state == S_IDLE || state == S_QLOAD || state == S_TINIT ||
                state == S_MASK || state == S_KLOAD || state == S_PAIR ||
                state == S_VSKIP || state == S_KVNEXT || state == S_FRECF ||
                state == S_FRECW || state == S_FEMITF || state == S_FEMITW ||
                state == S_FEMITE || state == S_DONE);
            assert(busy == (state != S_IDLE));
            assert(score_ready == f_score_ready);
            assert(pc_ready == f_pc_ready);

            if (f_flash_past_valid && $past(rst_n) && $past(state == S_IDLE && start)) begin
                assert(head_dim_q_r == $past(head_dim_q));
                assert(head_dim_v_r == $past(head_dim_v));
                assert(n_heads_r == $past(n_heads));
                assert(n_head_kv_r == $past(n_head_kv));
                assert(head_ratio_r == $past(head_ratio));
                assert(n_kv_r == $past(n_kv));
                assert(n_tokens_r == $past(n_tokens));
                assert(scale_r == $past(scale));
                assert(wide_heads_r == ($past(n_heads) > 16));
            end
            if (f_flash_past_valid && $past(rst_n) && $past(busy)) begin
                assert(head_dim_q_r == $past(head_dim_q_r));
                assert(head_dim_v_r == $past(head_dim_v_r));
                assert(n_heads_r == $past(n_heads_r));
                assert(n_head_kv_r == $past(n_head_kv_r));
                assert(head_ratio_r == $past(head_ratio_r));
                assert(n_kv_r == $past(n_kv_r));
                assert(n_tokens_r == $past(n_tokens_r));
                assert(scale_r == $past(scale_r));
                assert(wide_heads_r == $past(wide_heads_r));
            end

            // Scheduler valid and mode controls cross the same two boundaries as
            // the BRAM data. Wide data/scalar delay equalities are proved in the
            // isolated structural harness above to keep this PDR cone tractable.
            if (f_flash_past_valid && $past(rst_n)) begin
                assert(dot_fire_q == $past(dot_fire));
                assert(dot_fire_qq == $past(dot_fire_q));
                assert(ax_fire_q == $past(ax_fire));
                assert(ax_fire_qq == $past(ax_fire_q));
                assert(ax_axpy_q == $past(axpy_fire));
                assert(ax_axpy_qq == $past(ax_axpy_q));
            end

            if (busy) begin
                assert(qbeats > 0 && qbeats <= MAXB);
                assert(vbeats > 0 && vbeats <= MAXB);
                assert(n_heads_r > 0 && n_heads_r <= MAX_HEADS);
                assert(n_head_kv_r > 0 && n_head_kv_r <= MAX_HEAD_KV);
                assert(n_tokens_r > 0 && n_tokens_r <= MAX_TOKENS);
                assert(n_kv_r > 0 && head_ratio_r > 0);
                assert(n_heads_r == n_head_kv_r * head_ratio_r);
                assert((!wide_heads_r && n_heads_r <= 16 && n_tokens_r <= 4) ||
                       (wide_heads_r && n_heads_r <= 32 && n_tokens_r <= 2));
            end

            if (state == S_QLOAD && q_tvalid) begin
                assert(ld_tok < n_tokens_r && ld_a < n_heads_r && ld_b < qbeats);
                assert(load_slot == slotIndex(ld_tok, ld_a));
                assert(load_slot < MAX_SLOTS);
                assert(q_ld_addr == slotIndex(ld_tok, ld_a) * MAXB + ld_b);
            end
            if (state == S_TINIT) begin
                assert(ld_tok < n_tokens_r && ld_a < n_heads_r && ld_b < vbeats);
                assert(init_slot == slotIndex(ld_tok, ld_a));
                assert(init_slot < MAX_SLOTS);
                assert(q_ld_addr == slotIndex(ld_tok, ld_a) * MAXB + ld_b);
            end
            if (state == S_MASK) assert(ld_tok < n_tokens_r);
            if (state == S_KLOAD && k_tvalid) begin
                assert(ld_a < n_head_kv_r && ld_b < qbeats);
                assert(kv_ld_addr == ld_a * MAXB + ld_b);
            end
            if ((state == S_PAIR || state == S_VSKIP) && v_tvalid && v_tready) begin
                assert(ld_a < n_head_kv_r && ld_b < vbeats);
                assert(kv_ld_addr == ld_a * MAXB + ld_b);
            end

            if (dot_fire) begin
                assert(f_pair_active && mask_finite[dot_req_tok[TW-1:0]]);
                assert(dot_req_tok == f_dot_req_tok && dot_req_head == f_dot_req_head);
                assert(dot_req_kvh == f_dot_req_kvh && dot_req_ratio == f_dot_req_ratio);
                assert(dot_req_beat == f_dot_req_beat);
                assert(dot_req_tok < n_tokens_r && dot_req_head < n_heads_r);
                assert(dot_req_kvh < n_head_kv_r && dot_req_ratio < head_ratio_r);
                assert(dot_req_head == dot_req_kvh * head_ratio_r + dot_req_ratio);
                assert(dot_req_beat < qbeats);
                assert(dot_req_slot == slotIndex(dot_req_tok, dot_req_head));
                assert(dot_req_slot < MAX_SLOTS);
                assert(q_dot_addr == slotIndex(dot_req_tok, dot_req_head) * MAXB + dot_req_beat);
                assert(k_dot_addr == dot_req_kvh * MAXB + dot_req_beat);
            end
            if (dot_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(f_dot_capture_count < f_dot_issue_count);
                assert(dot_cap_tok == f_dot_cap_tok && dot_cap_head == f_dot_cap_head);
                assert(dot_cap_beat == f_dot_cap_beat);
                assert(mask_finite[dot_cap_tok[TW-1:0]]);
            end
            if (tree_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(f_tree_count < f_valid_slots);
            end
            if (smul_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(score_add_tok == f_tree_tok && score_add_head == f_tree_head);
                assert(mask_finite[score_add_tok[TW-1:0]]);
                assert(f_score_count < f_tree_count);
            end
            if (sadd_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(score_tok == f_score_tok && score_head == f_score_head);
                assert(mask_finite[score_tok[TW-1:0]]);
                assert(score_slot == slotIndex(score_tok, score_head));
                assert(score_slot < MAX_SLOTS);
                assert(f_score_count < f_tree_count);
                assert(!f_score_ready[score_slot]);
            end
            if (f_soft_launch) begin
                assert(f_pair_active && f_score_ready[soft_req_slot]);
                assert(soft_req_tok == f_soft_launch_tok && soft_req_head == f_soft_launch_head);
                assert(mask_finite[soft_req_tok[TW-1:0]]);
                assert(soft_req_slot == slotIndex(soft_req_tok, soft_req_head));
                assert(soft_req_slot < MAX_SLOTS);
                assert(f_soft_launch_count < f_score_count);
            end
            if (soft_fire) begin
                assert(state == S_PAIR && f_pair_active);
                assert($past(f_soft_launch));
                assert(f_soft_issue_count < f_soft_launch_count);
            end
            if (f_flash_past_valid && $past(rst_n) && $past(f_soft_launch))
                assert(soft_fire);
            if (soft_v) begin
                assert(state == S_PAIR && f_pair_active);
                assert(soft_wb_tok == f_soft_wb_tok && soft_wb_head == f_soft_wb_head);
                assert(mask_finite[soft_wb_tok[TW-1:0]]);
                assert(soft_wb_slot == slotIndex(soft_wb_tok, soft_wb_head));
                assert(soft_wb_slot < MAX_SLOTS);
                assert(f_soft_commit_count < f_soft_issue_count);
            end

            if (axpy_fire) begin
                assert(f_pair_active && v_load_done && f_pc_ready[ax_req_slot]);
                assert(ax_req_tok == f_ax_req_tok && ax_req_head == f_ax_req_head);
                assert(ax_req_kvh == f_ax_req_kvh && ax_req_ratio == f_ax_req_ratio);
                assert(ax_req_beat == f_ax_req_beat);
                assert(mask_finite[ax_req_tok[TW-1:0]]);
                assert(ax_req_head == ax_req_kvh * head_ratio_r + ax_req_ratio);
                assert(ax_req_slot == slotIndex(ax_req_tok, ax_req_head));
                assert(ax_req_slot < MAX_SLOTS);
                assert(acc_ax_rd == slotIndex(ax_req_tok, ax_req_head) * MAXB + ax_req_beat);
                assert(v_ax_addr == ax_req_kvh * MAXB + ax_req_beat);
            end
            if (state == S_PAIR && ax_v) begin
                assert(ax_wb_tok == f_ax_wb_tok && ax_wb_head == f_ax_wb_head);
                assert(ax_wb_beat == f_ax_wb_beat);
                assert(mask_finite[ax_wb_tok[TW-1:0]]);
                assert(ax_wb_slot == slotIndex(ax_wb_tok, ax_wb_head));
                assert(ax_wb_slot < MAX_SLOTS);
                assert(acc_ax_wr == slotIndex(ax_wb_tok, ax_wb_head) * MAXB + ax_wb_beat);
            end

            // KVNEXT is the recurrence barrier. A processed KV retires all valid
            // query/head slots; an all-masked KV reaches it only after consuming V
            // and launches no arithmetic work.
            if (state == S_KVNEXT && f_pair_active) begin
                assert(f_dot_issue_count == f_valid_slots * qbeats);
                assert(f_dot_capture_count == f_dot_issue_count);
                assert(f_soft_launch_count == f_valid_slots);
                // Completion counters, rather than retained ready bits, are the
                // architectural proof that every valid slot retired.
            end
            if (state == S_VSKIP) begin
                assert(first_valid_tok == n_tokens_r && mask_finite == 0);
                assert(!dot_fire && !tree_fire && !soft_fire && !axpy_fire);
            end
            if (f_flash_past_valid && $past(rst_n) &&
                $past(state == S_PAIR) && state == S_KVNEXT) begin
                assert($past(ax_v));
                assert($past(ax_wb_head + 16'd1 == n_heads_r));
                assert($past(ax_wb_beat + 16'd1 == vbeats));
                assert($past(ax_wb_next_tok == n_tokens_r));
            end
            if (f_flash_past_valid && $past(rst_n) &&
                $past(state == S_VSKIP) && state == S_KVNEXT) begin
                assert($past(v_tvalid && v_tready));
                assert($past(ld_a + 16'd1 == n_head_kv_r));
                assert($past(ld_b + 16'd1 == vbeats));
            end

            if (emit_fire) begin
                assert(tok_i < n_tokens_r && head_i < n_heads_r && bi < vbeats);
                assert(emit_slot == slotIndex(tok_i, head_i));
                assert(emit_slot < MAX_SLOTS);
                assert(acc_em_rd == slotIndex(tok_i, head_i) * MAXB + bi);
            end
            // Exact all-masked zero data is an arithmetic/cosim contract. Formal
            // proves the row never launches score/softmax/accumulate work and the
            // external harness still requires its complete framed output packet.
        end

        cover(rst_n && f_soft_issue_count >= 2 && f_soft_commit_count == 0);
        cover(rst_n && state == S_KVNEXT && f_pair_active);
        cover(rst_n && state == S_VSKIP && mask_finite == 0);
`else
        // These late aggregate checks are consequences of the local ordering and
        // tag invariants above. Check them over the complete bounded run instead
        // of forcing PDR to rediscover the full scheduler recurrence.
        if (f_flash_past_valid && !$past(rst_n))
            assert(state == S_IDLE && !busy && !done && !o_tvalid);
        if (f_flash_past_valid && $past(rst_n) && $past(done))
            assert(state == S_IDLE && !busy && !done && !o_tvalid);
        if (f_flash_past_valid && $past(rst_n) &&
            $past(state == S_IDLE && !start && !done))
            assert(state == S_IDLE && !busy && !done && !o_tvalid);
        if (rst_n && state == S_PAIR && ax_v)
            assert(f_pair_active && f_ax_commit_count < f_ax_issue_count);
        if (rst_n && state == S_KVNEXT && f_pair_active) begin
            assert(f_tree_count == f_valid_slots && f_score_count == f_valid_slots);
            assert(f_soft_issue_count == f_valid_slots && f_soft_commit_count == f_valid_slots);
            assert(f_ax_issue_count == f_valid_slots * vbeats);
            assert(f_ax_commit_count == f_ax_issue_count);
            assert(score_ready == 0 && v_load_done && ax_req_done);
        end
`endif
    end
`endif
`endif
