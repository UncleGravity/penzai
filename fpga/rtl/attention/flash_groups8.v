// Tile-8 grouped FlashAttention service.
//
// The numeric core owns eight query heads at a time. Bonsai model_specs therefore
// execute either two groups of {8 Q, 4 KV} heads or four groups of {8 Q, 2 KV}
// heads. Groups cover disjoint KV heads, so every historical K/V byte is
// consumed exactly once across the complete command. The upstream KV joiner
// concatenates committed history with the current tile's local NewKV records.

`default_nettype none

module flash_groups8 #(
    parameter integer HEAD_DIM = 128
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [3:0]   cmd_token_count,
    input  wire [5:0]   cmd_q_heads,
    input  wire [16:0]  cmd_history_len,
    input  wire [31:0]  cmd_scale,

    // One request per disjoint head group. The stream source must present Q in
    // token/head/beat order and K/V in position/KV-head/beat order.
    output wire         group_req_valid,
    input  wire         group_req_ready,
    output wire [1:0]   group_req_index,
    output wire [5:0]   group_req_q_head_base,
    output wire [3:0]   group_req_kv_head_base,
    output wire [2:0]   group_req_kv_heads,
    output wire [16:0]  group_req_total_kv,

    input  wire [255:0] q_tdata,
    input  wire         q_tvalid,
    output wire         q_tready,
    input  wire [127:0] k_tdata,
    input  wire         k_tvalid,
    output wire         k_tready,
    input  wire [127:0] v_tdata,
    input  wire         v_tvalid,
    output wire         v_tready,

    output wire [255:0] out_data,
    output wire         out_valid,
    input  wire         out_ready,
    output wire [2:0]   out_token,
    output wire [5:0]   out_head,
    output wire [3:0]   out_beat,
    output wire         out_group_last,
    output wire         out_last,

    output wire         busy,
    output wire         done_valid,
    input  wire         done_ready,
    output wire         done_error,
    output wire [7:0]   done_status
);
    localparam integer HEAD_BEATS = HEAD_DIM / 8;
    localparam [3:0] LAST_BEAT = HEAD_BEATS[3:0] - 4'd1;
    localparam [2:0] S_IDLE  = 3'd0;
    localparam [2:0] S_GROUP = 3'd1;
    localparam [2:0] S_START = 3'd2;
    localparam [2:0] S_RUN   = 3'd3;
    localparam [2:0] S_DONE  = 3'd4;

    reg [2:0] state_q;
    reg [3:0] token_count_q;
    reg [16:0] history_len_q;
    reg [16:0] total_kv_q;
    reg [31:0] scale_q;
    reg [1:0] group_q;
    reg [1:0] group_last_q;
    reg [2:0] group_kv_heads_q;
    reg [3:0] group_head_ratio_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    reg [16:0] mask_kv_q;
    reg [2:0] mask_token_q;
    reg [2:0] out_token_q;
    reg [2:0] out_head_q;
    reg [3:0] out_beat_q;

    wire [17:0] incoming_total =
        {1'b0, cmd_history_len} + {14'd0, cmd_token_count};
    wire command_tokens_ok = (cmd_token_count >= 4'd1) &&
                             (cmd_token_count <= 4'd8);
    wire command_heads_ok = (cmd_q_heads == 6'd16) ||
                            (cmd_q_heads == 6'd32);
    wire command_context_ok = incoming_total <= 18'd65536;

    wire core_rst_n = rst_n && !clear;
    wire core_start = state_q == S_START;
    wire core_busy;
    wire core_done;
    wire [255:0] core_o_data;
    wire core_o_valid;
    wire core_o_last;
    wire [31:0] core_o_keep;
    wire core_mask_ready;

    wire mask_is_history = mask_kv_q < history_len_q;
    wire [16:0] mask_local_index = mask_kv_q - history_len_q;
    wire mask_is_finite = mask_is_history ||
                          (mask_local_index <= {14'd0, mask_token_q});
    wire [15:0] mask_data = mask_is_finite ? 16'h0000 : 16'hfc00;
    wire mask_valid = state_q == S_RUN;
    wire mask_fire = mask_valid && core_mask_ready;

    wire core_o_fire = core_o_valid && out_ready;
    wire last_group = group_q == group_last_q;

    assign cmd_ready = state_q == S_IDLE;
    assign group_req_valid = state_q == S_GROUP;
    assign group_req_index = group_q;
    assign group_req_q_head_base = {1'b0, group_q, 3'b000};
    assign group_req_kv_head_base =
        (group_kv_heads_q == 3'd4) ? {group_q, 2'b00} :
                                     {1'b0, group_q, 1'b0};
    assign group_req_kv_heads = group_kv_heads_q;
    assign group_req_total_kv = total_kv_q;

    assign q_tready = core_busy ? core_q_ready : 1'b0;
    assign k_tready = core_busy ? core_k_ready : 1'b0;
    assign v_tready = core_busy ? core_v_ready : 1'b0;

    assign out_data = core_o_data;
    assign out_valid = (state_q == S_RUN) && core_o_valid;
    assign out_token = out_token_q;
    assign out_head = group_req_q_head_base + {3'd0, out_head_q};
    assign out_beat = out_beat_q;
    assign out_group_last = core_o_last;
    assign out_last = core_o_last && last_group;

    assign busy = (state_q != S_IDLE) && (state_q != S_DONE);
    assign done_valid = state_q == S_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    wire core_q_ready;
    wire core_k_ready;
    wire core_v_ready;

    flash_kernel #(
        .HEAD_DIM_MAX(HEAD_DIM),
        .MAX_HEADS(8),
        .MAX_HEAD_KV(8),
        .MAX_TOKENS(8),
        .MAX_SLOTS(64),
        .LANES(8),
        .TILE8_HEAD8_LAYOUT(1),
        .TILE8_LANE4_Q_ORDER(1)
    ) u_flash (
        .clk(clk), .rst_n(core_rst_n),
        .start(core_start),
        .head_dim_q(HEAD_DIM[15:0]), .head_dim_v(HEAD_DIM[15:0]),
        .n_heads(16'd8),
        .n_head_kv({13'd0, group_kv_heads_q}),
        .head_ratio({12'd0, group_head_ratio_q}),
        .n_kv(total_kv_q),
        .n_tokens({12'd0, token_count_q}),
        .scale(scale_q),
        .busy(core_busy), .done(core_done),
        .q_tdata(q_tdata), .q_tvalid(q_tvalid), .q_tready(core_q_ready),
        .q_tlast(1'b0), .q_tkeep(32'hffffffff),
        .k_tdata(k_tdata), .k_tvalid(k_tvalid), .k_tready(core_k_ready),
        .k_tlast(1'b0), .k_tkeep(16'hffff),
        .v_tdata(v_tdata), .v_tvalid(v_tvalid), .v_tready(core_v_ready),
        .v_tlast(1'b0), .v_tkeep(16'hffff),
        .mask_tdata(mask_data), .mask_tvalid(mask_valid),
        .mask_tready(core_mask_ready), .mask_tlast(1'b0), .mask_tkeep(2'b11),
        .o_tdata(core_o_data), .o_tvalid(core_o_valid), .o_tready(out_ready),
        .o_tlast(core_o_last), .o_tkeep(core_o_keep)
    );

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q <= S_IDLE;
            token_count_q <= 4'd0;
            history_len_q <= 17'd0;
            total_kv_q <= 17'd0;
            scale_q <= 32'd0;
            group_q <= 2'd0;
            group_last_q <= 2'd0;
            group_kv_heads_q <= 3'd0;
            group_head_ratio_q <= 4'd0;
            done_error_q <= 1'b0;
            done_status_q <= 8'd0;
            mask_kv_q <= 17'd0;
            mask_token_q <= 3'd0;
            out_token_q <= 3'd0;
            out_head_q <= 3'd0;
            out_beat_q <= 4'd0;
        end else begin
            case (state_q)
                S_IDLE: if (cmd_valid) begin
                    done_error_q <= 1'b0;
                    done_status_q <= 8'd0;
                    if (!command_tokens_ok) begin
                        done_error_q <= 1'b1;
                        done_status_q <= 8'h01;
                        state_q <= S_DONE;
                    end else if (!command_heads_ok) begin
                        done_error_q <= 1'b1;
                        done_status_q <= 8'h02;
                        state_q <= S_DONE;
                    end else if (!command_context_ok) begin
                        done_error_q <= 1'b1;
                        done_status_q <= 8'h03;
                        state_q <= S_DONE;
                    end else begin
                        token_count_q <= cmd_token_count;
                        history_len_q <= cmd_history_len;
                        total_kv_q <= incoming_total[16:0];
                        scale_q <= cmd_scale;
                        group_q <= 2'd0;
                        if (cmd_q_heads == 6'd16) begin
                            group_last_q <= 2'd1;
                            group_kv_heads_q <= 3'd4;
                            group_head_ratio_q <= 4'd2;
                        end else begin
                            group_last_q <= 2'd3;
                            group_kv_heads_q <= 3'd2;
                            group_head_ratio_q <= 4'd4;
                        end
                        state_q <= S_GROUP;
                    end
                end

                S_GROUP: if (group_req_ready) begin
                    mask_kv_q <= 17'd0;
                    mask_token_q <= 3'd0;
                    out_token_q <= 3'd0;
                    out_head_q <= 3'd0;
                    out_beat_q <= 4'd0;
                    state_q <= S_START;
                end

                S_START: state_q <= S_RUN;

                S_RUN: begin
                    if (mask_fire) begin
                        if ({1'b0, mask_token_q} + 4'd1 == token_count_q) begin
                            mask_token_q <= 3'd0;
                            if (mask_kv_q + 17'd1 != total_kv_q)
                                mask_kv_q <= mask_kv_q + 17'd1;
                        end else begin
                            mask_token_q <= mask_token_q + 3'd1;
                        end
                    end
                    if (core_o_fire) begin
                        if (out_beat_q == LAST_BEAT) begin
                            out_beat_q <= 4'd0;
                            if (out_head_q == 3'd7) begin
                                out_head_q <= 3'd0;
                                if ({1'b0, out_token_q} + 4'd1 != token_count_q)
                                    out_token_q <= out_token_q + 3'd1;
                            end else begin
                                out_head_q <= out_head_q + 3'd1;
                            end
                        end else begin
                            out_beat_q <= out_beat_q + 4'd1;
                        end
                    end
                    if (core_done) begin
                        if (last_group) begin
                            state_q <= S_DONE;
                        end else begin
                            group_q <= group_q + 2'd1;
                            state_q <= S_GROUP;
                        end
                    end
                end

                S_DONE: if (done_ready)
                    state_q <= S_IDLE;

                default: state_q <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && core_o_fire) begin
            if (core_o_keep != 32'hffffffff)
                $fatal(1, "flash_groups8 partial Flash output");
            if (core_o_last != ((out_beat_q == LAST_BEAT) &&
                               (out_head_q == 3'd7) &&
                               ({1'b0, out_token_q} + 4'd1 == token_count_q)))
                $fatal(1, "flash_groups8 output tag skew");
        end
    end
`endif

endmodule

`default_nettype wire
