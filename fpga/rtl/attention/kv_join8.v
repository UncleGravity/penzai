// Join committed KV history with the current tile-8 request's resident NewKV.
//
// History arrives already filtered to the requested KV-head group. K and V
// retain independent cursors because FlashAttention may backpressure them
// independently. Local records are scalar-banked across four token lanes, so a
// shared arena reader assembles eight fp16 scalars into each 128-bit beat.

`default_nettype none

module kv_join8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         abort_run,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [16:0]  cmd_history_len,
    input  wire [3:0]   cmd_token_count,
    input  wire [3:0]   cmd_kv_head_base,
    input  wire [2:0]   cmd_kv_head_count,

    // Mover streams contain exactly history_len * kv_head_count * 16 beats,
    // ordered position, group-local head, beat. last/error are checked on
    // every accepted record.
    input  wire [127:0] hist_k_data,
    input  wire         hist_k_valid,
    output wire         hist_k_ready,
    input  wire         hist_k_last,
    input  wire         hist_k_error,
    input  wire [127:0] hist_v_data,
    input  wire         hist_v_valid,
    output wire         hist_v_ready,
    input  wire         hist_v_last,
    input  wire         hist_v_error,

    // One scalar NewKV arena read service. Address is
    // {kind (K=0/V=1), global_kv_head[2:0], dim[6:0]}; wave and the selected
    // 16-bit lane encode the current tile token.
    output wire         newkv_rd_req_valid,
    input  wire         newkv_rd_req_ready,
    output wire         newkv_rd_req_wave,
    output wire [10:0]  newkv_rd_req_addr,
    input  wire         newkv_rd_rsp_valid,
    output wire         newkv_rd_rsp_ready,
    input  wire [63:0]  newkv_rd_rsp_data,
    input  wire         newkv_rd_rsp_error,

    output wire [127:0] k_data,
    output wire         k_valid,
    input  wire         k_ready,
    output wire         k_last,
    output wire [127:0] v_data,
    output wire         v_valid,
    input  wire         v_ready,
    output wire         v_last,

    output wire         busy,
    output wire         done_valid,
    input  wire         done_ready,
    output wire         done_error,
    output wire [7:0]   done_status
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_RUN   = 3'd1;
    localparam [2:0] ST_FAIL  = 3'd2;
    localparam [2:0] ST_DONE  = 3'd3;
    localparam [2:0] ST_ABORT = 3'd4;

    localparam [7:0] STATUS_OK       = 8'h00;
    localparam [7:0] STATUS_BAD_CMD  = 8'h01;
    localparam [7:0] STATUS_K_STREAM = 8'h02;
    localparam [7:0] STATUS_V_STREAM = 8'h03;
    localparam [7:0] STATUS_ARENA    = 8'h04;

    reg [2:0] state_q;
    reg [16:0] history_len_q;
    reg [3:0] token_count_q;
    reg [2:0] kv_head_base_q;
    reg [2:0] kv_head_count_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    reg k_history_q;
    reg v_history_q;
    reg [16:0] k_hist_pos_q;
    reg [16:0] v_hist_pos_q;
    reg [2:0] k_hist_head_q;
    reg [2:0] v_hist_head_q;
    reg [3:0] k_hist_beat_q;
    reg [3:0] v_hist_beat_q;

    reg [2:0] k_token_q;
    reg [2:0] v_token_q;
    reg [2:0] k_local_head_q;
    reg [2:0] v_local_head_q;
    reg [3:0] k_local_beat_q;
    reg [3:0] v_local_beat_q;
    reg [2:0] k_scalar_q;
    reg [2:0] v_scalar_q;
    reg k_generated_q;
    reg v_generated_q;
    reg k_complete_q;
    reg v_complete_q;

    reg [127:0] k_out_data_q;
    reg [127:0] v_out_data_q;
    reg k_out_valid_q;
    reg v_out_valid_q;
    reg k_out_last_q;
    reg v_out_last_q;

    reg arena_outstanding_q;
    reg arena_owner_q;
    reg [1:0] arena_lane_q;
    reg arena_round_robin_q;

    wire [17:0] command_total =
        {1'b0, cmd_history_len} + {14'd0, cmd_token_count};
    wire [4:0] command_head_end =
        {1'b0, cmd_kv_head_base} + {2'd0, cmd_kv_head_count};
    wire command_tokens_ok = (cmd_token_count >= 4'd1) &&
                             (cmd_token_count <= 4'd8);
    wire command_heads_ok =
        ((cmd_kv_head_count == 3'd2) && !cmd_kv_head_base[0] &&
         (command_head_end <= 5'd8)) ||
        ((cmd_kv_head_count == 3'd4) &&
         (cmd_kv_head_base[1:0] == 2'd0) &&
         (command_head_end <= 5'd8));
    wire command_context_ok = command_total <= 18'd65536;
    wire command_ok = command_tokens_ok && command_heads_ok &&
                      command_context_ok;

    wire k_out_fire = k_out_valid_q && k_ready;
    wire v_out_fire = v_out_valid_q && v_ready;
    wire k_out_capacity = !k_out_valid_q || k_ready;
    wire v_out_capacity = !v_out_valid_q || v_ready;

    wire k_hist_expected_last =
        (k_hist_pos_q + 17'd1 == history_len_q) &&
        (k_hist_head_q + 3'd1 == kv_head_count_q) &&
        (k_hist_beat_q == 4'd15);
    wire v_hist_expected_last =
        (v_hist_pos_q + 17'd1 == history_len_q) &&
        (v_hist_head_q + 3'd1 == kv_head_count_q) &&
        (v_hist_beat_q == 4'd15);
    wire k_hist_fire = hist_k_valid && hist_k_ready;
    wire v_hist_fire = hist_v_valid && hist_v_ready;
    wire k_hist_bad = k_hist_fire &&
                      (hist_k_error || (hist_k_last != k_hist_expected_last));
    wire v_hist_bad = v_hist_fire &&
                      (hist_v_error || (hist_v_last != v_hist_expected_last));

    wire k_local_want = (state_q == ST_RUN) && !clear && !abort_run &&
                        !k_history_q && !k_generated_q &&
                        ((k_scalar_q != 3'd0) || k_out_capacity);
    wire v_local_want = (state_q == ST_RUN) && !clear && !abort_run &&
                        !v_history_q && !v_generated_q &&
                        ((v_scalar_q != 3'd0) || v_out_capacity);
    wire choose_v = v_local_want &&
                    (!k_local_want || !arena_round_robin_q);
    wire choose_k = k_local_want && !choose_v;
    wire arena_issue_want = !arena_outstanding_q &&
                            (choose_k || choose_v);
    wire arena_req_fire = newkv_rd_req_valid && newkv_rd_req_ready;
    wire arena_rsp_fire = newkv_rd_rsp_valid && newkv_rd_rsp_ready;

    wire [2:0] issue_token = choose_v ? v_token_q : k_token_q;
    wire [2:0] issue_local_head =
        choose_v ? v_local_head_q : k_local_head_q;
    wire [3:0] issue_beat = choose_v ? v_local_beat_q : k_local_beat_q;
    wire [2:0] issue_scalar = choose_v ? v_scalar_q : k_scalar_q;
    // Accepted commands guarantee the sum is in 0..7.
    wire [2:0] issue_global_head =
        kv_head_base_q[2:0] + issue_local_head;

    wire [15:0] arena_scalar =
        arena_lane_q == 2'd0 ? newkv_rd_rsp_data[15:0] :
        arena_lane_q == 2'd1 ? newkv_rd_rsp_data[31:16] :
        arena_lane_q == 2'd2 ? newkv_rd_rsp_data[47:32] :
                               newkv_rd_rsp_data[63:48];
    wire arena_response_is_last_scalar = arena_owner_q ?
        (v_scalar_q == 3'd7) : (k_scalar_q == 3'd7);
    wire k_local_record_last =
                               ({1'b0, k_token_q} + 4'd1 == token_count_q) &&
                               (k_local_head_q + 3'd1 == kv_head_count_q) &&
                               (k_local_beat_q == 4'd15);
    wire v_local_record_last =
                               ({1'b0, v_token_q} + 4'd1 == token_count_q) &&
                               (v_local_head_q + 3'd1 == kv_head_count_q) &&
                               (v_local_beat_q == 4'd15);

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign hist_k_ready = (state_q == ST_RUN) && !clear && !abort_run &&
                          k_history_q && k_out_capacity;
    assign hist_v_ready = (state_q == ST_RUN) && !clear && !abort_run &&
                          v_history_q && v_out_capacity;

    assign newkv_rd_req_valid = arena_issue_want;
    assign newkv_rd_req_wave = issue_token[2];
    assign newkv_rd_req_addr = {
        choose_v, issue_global_head[2:0], issue_beat, issue_scalar
    };
    assign newkv_rd_rsp_ready = arena_outstanding_q &&
        ((state_q == ST_RUN) || (state_q == ST_FAIL) ||
         (state_q == ST_ABORT));

    assign k_data = k_out_data_q;
    assign k_valid = k_out_valid_q;
    assign k_last = k_out_last_q;
    assign v_data = v_out_data_q;
    assign v_valid = v_out_valid_q;
    assign v_last = v_out_last_q;

    assign busy = (state_q == ST_RUN) || (state_q == ST_FAIL) ||
                  (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            history_len_q <= 17'd0;
            token_count_q <= 4'd0;
            kv_head_base_q <= 3'd0;
            kv_head_count_q <= 3'd0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
            k_history_q <= 1'b0;
            v_history_q <= 1'b0;
            k_hist_pos_q <= 17'd0;
            v_hist_pos_q <= 17'd0;
            k_hist_head_q <= 3'd0;
            v_hist_head_q <= 3'd0;
            k_hist_beat_q <= 4'd0;
            v_hist_beat_q <= 4'd0;
            k_token_q <= 3'd0;
            v_token_q <= 3'd0;
            k_local_head_q <= 3'd0;
            v_local_head_q <= 3'd0;
            k_local_beat_q <= 4'd0;
            v_local_beat_q <= 4'd0;
            k_scalar_q <= 3'd0;
            v_scalar_q <= 3'd0;
            k_generated_q <= 1'b0;
            v_generated_q <= 1'b0;
            k_complete_q <= 1'b0;
            v_complete_q <= 1'b0;
            k_out_data_q <= 128'd0;
            v_out_data_q <= 128'd0;
            k_out_valid_q <= 1'b0;
            v_out_valid_q <= 1'b0;
            k_out_last_q <= 1'b0;
            v_out_last_q <= 1'b0;
            arena_outstanding_q <= 1'b0;
            arena_owner_q <= 1'b0;
            arena_lane_q <= 2'd0;
            arena_round_robin_q <= 1'b0;
        end else begin
            if (k_out_fire)
                k_out_valid_q <= 1'b0;
            if (v_out_fire)
                v_out_valid_q <= 1'b0;

            if (arena_req_fire) begin
                arena_outstanding_q <= 1'b1;
                arena_owner_q <= choose_v;
                arena_lane_q <= issue_token[1:0];
                arena_round_robin_q <= choose_v;
            end
            if (arena_rsp_fire)
                arena_outstanding_q <= 1'b0;

            if (clear) begin
                // The parent simultaneously clears the resident NewKV arena,
                // so any accepted response is cancelled rather than drained.
                arena_outstanding_q <= 1'b0;
                state_q <= ST_IDLE;
                k_out_valid_q <= 1'b0;
                v_out_valid_q <= 1'b0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
            end else if (abort_run) begin
                // A local abort leaves the arena alive and drains its one
                // accepted response before making the next command ready.
                state_q <= ST_ABORT;
                k_out_valid_q <= 1'b0;
                v_out_valid_q <= 1'b0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
            end else begin
                case (state_q)
                    ST_IDLE: if (cmd_valid) begin
                        history_len_q <= cmd_history_len;
                        token_count_q <= cmd_token_count;
                        kv_head_base_q <= cmd_kv_head_base[2:0];
                        kv_head_count_q <= cmd_kv_head_count;
                        done_error_q <= !command_ok;
                        done_status_q <= command_ok ? STATUS_OK :
                                                      STATUS_BAD_CMD;
                        k_history_q <= cmd_history_len != 17'd0;
                        v_history_q <= cmd_history_len != 17'd0;
                        k_hist_pos_q <= 17'd0;
                        v_hist_pos_q <= 17'd0;
                        k_hist_head_q <= 3'd0;
                        v_hist_head_q <= 3'd0;
                        k_hist_beat_q <= 4'd0;
                        v_hist_beat_q <= 4'd0;
                        k_token_q <= 3'd0;
                        v_token_q <= 3'd0;
                        k_local_head_q <= 3'd0;
                        v_local_head_q <= 3'd0;
                        k_local_beat_q <= 4'd0;
                        v_local_beat_q <= 4'd0;
                        k_scalar_q <= 3'd0;
                        v_scalar_q <= 3'd0;
                        k_generated_q <= 1'b0;
                        v_generated_q <= 1'b0;
                        k_complete_q <= 1'b0;
                        v_complete_q <= 1'b0;
                        k_out_valid_q <= 1'b0;
                        v_out_valid_q <= 1'b0;
                        k_out_last_q <= 1'b0;
                        v_out_last_q <= 1'b0;
                        arena_round_robin_q <= 1'b0;
                        state_q <= command_ok ? ST_RUN : ST_DONE;
                    end

                    ST_RUN: begin
                        if (k_hist_fire && !k_hist_bad) begin
                            k_out_data_q <= hist_k_data;
                            k_out_valid_q <= 1'b1;
                            k_out_last_q <= 1'b0;
                            if (k_hist_expected_last) begin
                                k_history_q <= 1'b0;
                            end else if (k_hist_beat_q == 4'd15) begin
                                k_hist_beat_q <= 4'd0;
                                if (k_hist_head_q + 3'd1 == kv_head_count_q) begin
                                    k_hist_head_q <= 3'd0;
                                    k_hist_pos_q <= k_hist_pos_q + 17'd1;
                                end else begin
                                    k_hist_head_q <= k_hist_head_q + 3'd1;
                                end
                            end else begin
                                k_hist_beat_q <= k_hist_beat_q + 4'd1;
                            end
                        end
                        if (v_hist_fire && !v_hist_bad) begin
                            v_out_data_q <= hist_v_data;
                            v_out_valid_q <= 1'b1;
                            v_out_last_q <= 1'b0;
                            if (v_hist_expected_last) begin
                                v_history_q <= 1'b0;
                            end else if (v_hist_beat_q == 4'd15) begin
                                v_hist_beat_q <= 4'd0;
                                if (v_hist_head_q + 3'd1 == kv_head_count_q) begin
                                    v_hist_head_q <= 3'd0;
                                    v_hist_pos_q <= v_hist_pos_q + 17'd1;
                                end else begin
                                    v_hist_head_q <= v_hist_head_q + 3'd1;
                                end
                            end else begin
                                v_hist_beat_q <= v_hist_beat_q + 4'd1;
                            end
                        end

                        if (arena_rsp_fire && !newkv_rd_rsp_error) begin
                            if (!arena_owner_q) begin
                                k_out_data_q <= {
                                    arena_scalar, k_out_data_q[127:16]
                                };
                                if (arena_response_is_last_scalar) begin
                                    k_out_valid_q <= 1'b1;
                                    k_out_last_q <= k_local_record_last;
                                    k_scalar_q <= 3'd0;
                                    if (k_local_record_last) begin
                                        k_generated_q <= 1'b1;
                                    end else if (k_local_beat_q == 4'd15) begin
                                        k_local_beat_q <= 4'd0;
                                        if (k_local_head_q + 3'd1 ==
                                            kv_head_count_q) begin
                                            k_local_head_q <= 3'd0;
                                            k_token_q <= k_token_q + 3'd1;
                                        end else begin
                                            k_local_head_q <=
                                                k_local_head_q + 3'd1;
                                        end
                                    end else begin
                                        k_local_beat_q <= k_local_beat_q + 4'd1;
                                    end
                                end else begin
                                    k_scalar_q <= k_scalar_q + 3'd1;
                                end
                            end else begin
                                v_out_data_q <= {
                                    arena_scalar, v_out_data_q[127:16]
                                };
                                if (arena_response_is_last_scalar) begin
                                    v_out_valid_q <= 1'b1;
                                    v_out_last_q <= v_local_record_last;
                                    v_scalar_q <= 3'd0;
                                    if (v_local_record_last) begin
                                        v_generated_q <= 1'b1;
                                    end else if (v_local_beat_q == 4'd15) begin
                                        v_local_beat_q <= 4'd0;
                                        if (v_local_head_q + 3'd1 ==
                                            kv_head_count_q) begin
                                            v_local_head_q <= 3'd0;
                                            v_token_q <= v_token_q + 3'd1;
                                        end else begin
                                            v_local_head_q <=
                                                v_local_head_q + 3'd1;
                                        end
                                    end else begin
                                        v_local_beat_q <= v_local_beat_q + 4'd1;
                                    end
                                end else begin
                                    v_scalar_q <= v_scalar_q + 3'd1;
                                end
                            end
                        end

                        if (k_out_fire && k_out_last_q)
                            k_complete_q <= 1'b1;
                        if (v_out_fire && v_out_last_q)
                            v_complete_q <= 1'b1;

                        if (k_hist_bad || v_hist_bad ||
                            (arena_rsp_fire && newkv_rd_rsp_error)) begin
                            done_error_q <= 1'b1;
                            done_status_q <= k_hist_bad ? STATUS_K_STREAM :
                                v_hist_bad ? STATUS_V_STREAM : STATUS_ARENA;
                            k_out_valid_q <= 1'b0;
                            v_out_valid_q <= 1'b0;
                            state_q <= ST_FAIL;
                        end else if ((k_complete_q ||
                                      (k_out_fire && k_out_last_q)) &&
                                     (v_complete_q ||
                                      (v_out_fire && v_out_last_q))) begin
                            done_error_q <= 1'b0;
                            done_status_q <= STATUS_OK;
                            state_q <= ST_DONE;
                        end
                    end

                    ST_FAIL: begin
                        k_out_valid_q <= 1'b0;
                        v_out_valid_q <= 1'b0;
                        if (!arena_outstanding_q || arena_rsp_fire)
                            state_q <= ST_DONE;
                    end

                    ST_DONE: if (done_ready)
                        state_q <= ST_IDLE;

                    ST_ABORT: begin
                        k_out_valid_q <= 1'b0;
                        v_out_valid_q <= 1'b0;
                        if (!clear && !abort_run &&
                            (!arena_outstanding_q || arena_rsp_fire))
                            state_q <= ST_IDLE;
                    end

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end
endmodule

`default_nettype wire
