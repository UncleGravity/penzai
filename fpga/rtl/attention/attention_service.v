`default_nettype none

// One resident attention transaction. The service walks every disjoint
// eight-query-head group, reads each historical K/V byte once, joins the
// current tile's NewKV, runs FlashAttention, and writes the result directly
// into the resident Q8 arena used by the O projection.
module attention_service (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,

    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [3:0]    cmd_token_count,
    input  wire [7:0]    cmd_token_mask,
    input  wire [5:0]    cmd_q_heads,
    input  wire [16:0]   cmd_history_len,
    input  wire [31:0]   cmd_scale,
    input  wire [63:0]   cmd_layer_kv_base,

    // Resident FP32 Query arena.
    output wire          query_rd_req_valid,
    input  wire          query_rd_req_ready,
    output wire          query_rd_req_wave,
    output wire [11:0]   query_rd_req_addr,
    input  wire          query_rd_rsp_valid,
    output wire          query_rd_rsp_ready,
    input  wire [127:0]  query_rd_rsp_data,

    // Resident fp16 NewKV arena.
    output wire          newkv_rd_req_valid,
    input  wire          newkv_rd_req_ready,
    output wire          newkv_rd_req_wave,
    output wire [10:0]   newkv_rd_req_addr,
    input  wire          newkv_rd_rsp_valid,
    output wire          newkv_rd_rsp_ready,
    input  wire [63:0]   newkv_rd_rsp_data,

    // Resident Q8 arena, consumed next by the O projection.
    output wire          q8_wr_valid,
    input  wire          q8_wr_ready,
    output wire          q8_wr_wave,
    output wire [8:0]    q8_wr_addr,
    output wire [3:0]    q8_wr_lane_mask,
    output wire [1087:0] q8_wr_data,

    // Client of the token engine's sole shared Q8 leaf. Ownership is held for
    // each attention wave until leaf_q8_out_last is accepted.
    output wire          leaf_q8_cfg_valid,
    input  wire          leaf_q8_cfg_ready,
    output wire [14:0]   leaf_q8_cfg_rows,
    output wire [3:0]    leaf_q8_cfg_lane_mask,
    output wire          leaf_q8_abort,
    input  wire          leaf_q8_busy,
    output wire          leaf_q8_in_valid,
    input  wire          leaf_q8_in_ready,
    output wire [127:0]  leaf_q8_in_data,
    input  wire          leaf_q8_out_valid,
    output wire          leaf_q8_out_ready,
    input  wire [8:0]    leaf_q8_out_block,
    input  wire [1087:0] leaf_q8_out_data,
    input  wire [7:0]    leaf_q8_out_status,
    input  wire          leaf_q8_out_last,

    // Independent committed-K history mover.
    output wire          hist_k_cmd_valid,
    input  wire          hist_k_cmd_ready,
    output wire [63:0]   hist_k_cmd_addr,
    output wire [31:0]   hist_k_cmd_segment_beats,
    output wire [31:0]   hist_k_cmd_stride_bytes,
    output wire [16:0]   hist_k_cmd_repeats,
    output wire          hist_k_abort,
    input  wire [127:0]  hist_k_data,
    input  wire          hist_k_valid,
    output wire          hist_k_ready,
    input  wire          hist_k_last,
    input  wire          hist_k_error,
    input  wire          hist_k_busy,
    input  wire          hist_k_done_valid,
    output wire          hist_k_done_ready,
    input  wire          hist_k_done_error,
    input  wire [7:0]    hist_k_done_status,

    // Independent committed-V history mover.
    output wire          hist_v_cmd_valid,
    input  wire          hist_v_cmd_ready,
    output wire [63:0]   hist_v_cmd_addr,
    output wire [31:0]   hist_v_cmd_segment_beats,
    output wire [31:0]   hist_v_cmd_stride_bytes,
    output wire [16:0]   hist_v_cmd_repeats,
    output wire          hist_v_abort,
    input  wire [127:0]  hist_v_data,
    input  wire          hist_v_valid,
    output wire          hist_v_ready,
    input  wire          hist_v_last,
    input  wire          hist_v_error,
    input  wire          hist_v_busy,
    input  wire          hist_v_done_valid,
    output wire          hist_v_done_ready,
    input  wire          hist_v_done_error,
    input  wire [7:0]    hist_v_done_status,

    output wire          busy,
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [7:0]    done_status
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_RUN   = 3'd1;
    localparam [2:0] ST_FAIL  = 3'd2;
    localparam [2:0] ST_ABORT = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    localparam [7:0] STATUS_BAD_CMD = 8'h01;
    localparam [7:0] STATUS_QUERY   = 8'h10;
    localparam [7:0] STATUS_KV_JOIN = 8'h20;
    localparam [7:0] STATUS_K_MOVER = 8'h30;
    localparam [7:0] STATUS_V_MOVER = 8'h40;
    localparam [7:0] STATUS_FLASH   = 8'h50;
    localparam [7:0] STATUS_OUTPUT  = 8'h60;

    reg [2:0] state_q;
    reg [7:0] status_q;
    reg [16:0] history_len_q;
    reg [63:0] layer_kv_base_q;
    reg q_started_q;
    reg join_started_q;
    reg k_mover_started_q;
    reg v_mover_started_q;
    reg flash_done_seen_q;
    reg output_done_seen_q;
    reg local_abort_q;
    reg [4:0] query_outstanding_q;

    function automatic [7:0] prefix_mask8(input [3:0] count);
        begin
            case (count)
                4'd1: prefix_mask8 = 8'h01;
                4'd2: prefix_mask8 = 8'h03;
                4'd3: prefix_mask8 = 8'h07;
                4'd4: prefix_mask8 = 8'h0f;
                4'd5: prefix_mask8 = 8'h1f;
                4'd6: prefix_mask8 = 8'h3f;
                4'd7: prefix_mask8 = 8'h7f;
                4'd8: prefix_mask8 = 8'hff;
                default: prefix_mask8 = 8'h00;
            endcase
        end
    endfunction

    wire [17:0] command_total =
        {1'b0, cmd_history_len} + {14'd0, cmd_token_count};
    wire command_ok = (cmd_token_count >= 4'd1) &&
                      (cmd_token_count <= 4'd8) &&
                      (cmd_token_mask == prefix_mask8(cmd_token_count)) &&
                      ((cmd_q_heads == 6'd16) ||
                       (cmd_q_heads == 6'd32)) &&
                      (command_total <= 18'd65536) &&
                      (cmd_layer_kv_base[3:0] == 4'd0);

    wire flash_cmd_ready;
    wire flash_busy;
    wire flash_done_valid;
    wire flash_done_error;
    wire [7:0] flash_done_status;
    wire output_cfg_ready;
    wire output_busy;
    wire output_done_valid;
    wire output_done_error;
    wire [7:0] output_done_status;
    wire command_children_ready = flash_cmd_ready && output_cfg_ready;

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE) &&
                       (!command_ok || command_children_ready);
    wire command_fire = cmd_valid && cmd_ready;
    wire child_command_fire = command_fire && command_ok;

    wire child_abort = clear || abort_run || local_abort_q;
    assign hist_k_abort = child_abort;
    assign hist_v_abort = child_abort;

    wire group_req_valid;
    wire group_req_ready;
    wire [1:0] group_req_index;
    wire [5:0] group_req_q_head_base;
    wire [3:0] group_req_kv_head_base;
    wire [2:0] group_req_kv_heads;
    wire [16:0] group_req_total_kv;

    wire [255:0] flash_q_data;
    wire flash_q_valid;
    wire flash_q_ready;
    wire [127:0] flash_k_data;
    wire flash_k_valid;
    wire flash_k_ready;
    wire [127:0] flash_v_data;
    wire flash_v_valid;
    wire flash_v_ready;
    wire [255:0] flash_out_data;
    wire flash_out_valid;
    wire flash_out_ready;
    wire [2:0] flash_out_token;
    wire [5:0] flash_out_head;
    wire [3:0] flash_out_beat;
    wire flash_out_group_last;
    wire flash_out_last;

    flash_groups8 u_flash_groups (
        .clk(clk), .rst_n(rst_n), .clear(child_abort),
        .cmd_valid(child_command_fire), .cmd_ready(flash_cmd_ready),
        .cmd_token_count(cmd_token_count), .cmd_q_heads(cmd_q_heads),
        .cmd_history_len(cmd_history_len), .cmd_scale(cmd_scale),
        .group_req_valid(group_req_valid),
        .group_req_ready(group_req_ready),
        .group_req_index(group_req_index),
        .group_req_q_head_base(group_req_q_head_base),
        .group_req_kv_head_base(group_req_kv_head_base),
        .group_req_kv_heads(group_req_kv_heads),
        .group_req_total_kv(group_req_total_kv),
        .q_tdata(flash_q_data), .q_tvalid(flash_q_valid),
        .q_tready(flash_q_ready),
        .k_tdata(flash_k_data), .k_tvalid(flash_k_valid),
        .k_tready(flash_k_ready),
        .v_tdata(flash_v_data), .v_tvalid(flash_v_valid),
        .v_tready(flash_v_ready),
        .out_data(flash_out_data), .out_valid(flash_out_valid),
        .out_ready(flash_out_ready), .out_token(flash_out_token),
        .out_head(flash_out_head), .out_beat(flash_out_beat),
        .out_group_last(flash_out_group_last),
        .out_last(flash_out_last),
        .busy(flash_busy), .done_valid(flash_done_valid), .done_ready(1'b1),
        .done_error(flash_done_error), .done_status(flash_done_status)
    );

     flash_output_q8 u_output (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .abort_run(abort_run || local_abort_q),
        .cfg_valid(child_command_fire), .cfg_ready(output_cfg_ready),
        .cfg_token_count(cmd_token_count),
        .cfg_token_mask(cmd_token_mask), .cfg_q_heads(cmd_q_heads),
        .in_data(flash_out_data), .in_valid(flash_out_valid),
        .in_ready(flash_out_ready), .in_token(flash_out_token),
        .in_head(flash_out_head), .in_beat(flash_out_beat),
        .in_group_last(flash_out_group_last), .in_last(flash_out_last),
        .q8_wr_valid(q8_wr_valid), .q8_wr_ready(q8_wr_ready),
        .q8_wr_wave(q8_wr_wave), .q8_wr_addr(q8_wr_addr),
        .q8_wr_lane_mask(q8_wr_lane_mask), .q8_wr_data(q8_wr_data),
        .leaf_q8_cfg_valid(leaf_q8_cfg_valid),
        .leaf_q8_cfg_ready(leaf_q8_cfg_ready),
        .leaf_q8_cfg_rows(leaf_q8_cfg_rows),
        .leaf_q8_cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .leaf_q8_abort(leaf_q8_abort), .leaf_q8_busy(leaf_q8_busy),
        .leaf_q8_in_valid(leaf_q8_in_valid),
        .leaf_q8_in_ready(leaf_q8_in_ready),
        .leaf_q8_in_data(leaf_q8_in_data),
        .leaf_q8_out_valid(leaf_q8_out_valid),
        .leaf_q8_out_ready(leaf_q8_out_ready),
        .leaf_q8_out_block(leaf_q8_out_block),
        .leaf_q8_out_data(leaf_q8_out_data),
        .leaf_q8_out_status(leaf_q8_out_status),
        .leaf_q8_out_last(leaf_q8_out_last),
        .busy(output_busy), .done_valid(output_done_valid), .done_ready(1'b1),
        .done_error(output_done_error),
        .done_status(output_done_status)
    );

    wire q_start_ready;
    wire q_done_valid;
    wire q_done_error;
    wire q_busy;
    wire q_req_valid_i;
    wire q_req_ready_i;
    wire q_rsp_valid_i;
    wire q_rsp_ready_i;

    flash_query_gather u_q_gather (
        .clk(clk), .rst_n(rst_n), .clear(child_abort),
        .start_valid(q_start_valid), .start_ready(q_start_ready),
        .start_q_head_base(group_req_q_head_base),
        .start_token_count(cmd_token_count_q),
        .query_rd_req_valid(q_req_valid_i),
        .query_rd_req_ready(q_req_ready_i),
        .query_rd_req_wave(query_rd_req_wave),
        .query_rd_req_addr(query_rd_req_addr),
        .query_rd_rsp_valid(q_rsp_valid_i),
        .query_rd_rsp_ready(q_rsp_ready_i),
        .query_rd_rsp_data(query_rd_rsp_data),
        .q_tdata(flash_q_data), .q_tvalid(flash_q_valid),
        .q_tready(flash_q_ready), .busy(q_busy),
        .done_valid(q_done_valid), .done_ready(1'b1),
        .done_error(q_done_error)
    );

    // The resident Query arena has one buffered response. Count it outside
    // the gatherer so an abort can consume a stale response before restart.
    wire query_run = state_q == ST_RUN;
    assign query_rd_req_valid = q_req_valid_i && query_run && !child_abort;
    assign q_req_ready_i = query_rd_req_ready && query_run && !child_abort;
    assign q_rsp_valid_i = query_rd_rsp_valid && query_run && !child_abort;
    assign query_rd_rsp_ready = query_run ? q_rsp_ready_i :
                                ((state_q == ST_FAIL) ||
                                 (state_q == ST_ABORT));
    wire query_req_fire = query_rd_req_valid && query_rd_req_ready;
    wire query_rsp_fire = query_rd_rsp_valid && query_rd_rsp_ready;

    wire join_cmd_ready;
    wire join_done_valid;
    wire join_done_error;
    wire [7:0] join_done_status;
    wire join_busy;
    wire join_hist_k_ready;
    wire join_hist_v_ready;
    wire join_k_last;
    wire join_v_last;

     kv_join8 u_kv_join (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .abort_run(abort_run || local_abort_q),
        .cmd_valid(join_start_valid), .cmd_ready(join_cmd_ready),
        .cmd_history_len(history_len_q),
        .cmd_token_count(cmd_token_count_q),
        .cmd_kv_head_base(group_req_kv_head_base),
        .cmd_kv_head_count(group_req_kv_heads),
        .hist_k_data(hist_k_data),
        .hist_k_valid(hist_k_valid && query_run),
        .hist_k_ready(join_hist_k_ready), .hist_k_last(hist_k_last),
        .hist_k_error(hist_k_error),
        .hist_v_data(hist_v_data),
        .hist_v_valid(hist_v_valid && query_run),
        .hist_v_ready(join_hist_v_ready), .hist_v_last(hist_v_last),
        .hist_v_error(hist_v_error),
        .newkv_rd_req_valid(newkv_rd_req_valid),
        .newkv_rd_req_ready(newkv_rd_req_ready),
        .newkv_rd_req_wave(newkv_rd_req_wave),
        .newkv_rd_req_addr(newkv_rd_req_addr),
        .newkv_rd_rsp_valid(newkv_rd_rsp_valid),
        .newkv_rd_rsp_ready(newkv_rd_rsp_ready),
        .newkv_rd_rsp_data(newkv_rd_rsp_data),
        .newkv_rd_rsp_error(1'b0),
        .k_data(flash_k_data), .k_valid(flash_k_valid),
        .k_ready(flash_k_ready), .k_last(join_k_last),
        .v_data(flash_v_data), .v_valid(flash_v_valid),
        .v_ready(flash_v_ready), .v_last(join_v_last),
        .busy(join_busy), .done_valid(join_done_valid),
        .done_ready(1'b1), .done_error(join_done_error),
        .done_status(join_done_status)
    );

    assign hist_k_ready = join_hist_k_ready && query_run;
    assign hist_v_ready = join_hist_v_ready && query_run;
    assign hist_k_done_ready = 1'b1;
    assign hist_v_done_ready = 1'b1;

    reg [3:0] cmd_token_count_q;
    // Do not start the next Flash group while the output adapter is
    // transposing/quantizing the previous group.
    wire group_launch_enable = query_run && group_req_valid &&
                               flash_out_ready && !child_abort;
    wire q_start_valid = group_launch_enable && !q_started_q;
    wire join_start_valid = group_launch_enable && !join_started_q;
    wire k_mover_start_valid = group_launch_enable &&
                               (history_len_q != 17'd0) &&
                               !k_mover_started_q;
    wire v_mover_start_valid = group_launch_enable &&
                               (history_len_q != 17'd0) &&
                               !v_mover_started_q;
    wire q_start_fire = q_start_valid && q_start_ready;
    wire join_start_fire = join_start_valid && join_cmd_ready;
    wire k_mover_start_fire = k_mover_start_valid && hist_k_cmd_ready;
    wire v_mover_start_fire = v_mover_start_valid && hist_v_cmd_ready;
    wire q_group_started = q_started_q || q_start_fire;
    wire join_group_started = join_started_q || join_start_fire;
    wire k_group_started = (history_len_q == 17'd0) ||
                           k_mover_started_q || k_mover_start_fire;
    wire v_group_started = (history_len_q == 17'd0) ||
                           v_mover_started_q || v_mover_start_fire;
    assign group_req_ready = group_launch_enable && q_group_started &&
                             join_group_started && k_group_started &&
                             v_group_started;

    assign hist_k_cmd_valid = k_mover_start_valid;
    assign hist_v_cmd_valid = v_mover_start_valid;
    wire [63:0] group_head_byte_offset =
        {60'd0, group_req_kv_head_base} << 8;
    assign hist_k_cmd_addr = layer_kv_base_q + group_head_byte_offset;
    assign hist_v_cmd_addr = layer_kv_base_q + 64'd2048 +
                             group_head_byte_offset;
    // Each KV head is 256 bytes, or sixteen 128-bit mover beats.
    assign hist_k_cmd_segment_beats = {25'd0, group_req_kv_heads, 4'b0000};
    assign hist_v_cmd_segment_beats = {25'd0, group_req_kv_heads, 4'b0000};
    assign hist_k_cmd_stride_bytes = 32'd4096;
    assign hist_v_cmd_stride_bytes = 32'd4096;
    assign hist_k_cmd_repeats = history_len_q;
    assign hist_v_cmd_repeats = history_len_q;

    wire q_error_event = q_done_valid && q_done_error;
    wire join_error_event = join_done_valid && join_done_error;
    wire k_mover_error_event = hist_k_done_valid && hist_k_done_error;
    wire v_mover_error_event = hist_v_done_valid && hist_v_done_error;
    wire flash_error_event = flash_done_valid && flash_done_error;
    wire output_error_event = output_done_valid && output_done_error;
    wire child_error_event = q_error_event || join_error_event ||
                             k_mover_error_event || v_mover_error_event ||
                             flash_error_event || output_error_event;
    wire [7:0] child_error_status = q_error_event ?
        (STATUS_QUERY | 8'h01) : join_error_event ?
        (STATUS_KV_JOIN | {4'd0, join_done_status[3:0]}) :
        k_mover_error_event ?
        (STATUS_K_MOVER | {4'd0, hist_k_done_status[3:0]}) :
        v_mover_error_event ?
        (STATUS_V_MOVER | {4'd0, hist_v_done_status[3:0]}) :
        flash_error_event ?
        (STATUS_FLASH | {4'd0, flash_done_status[3:0]}) :
        (STATUS_OUTPUT | {4'd0, output_done_status[3:0]});

    wire children_idle = (query_outstanding_q == 5'd0) &&
                         !hist_k_busy && !hist_v_busy &&
                         !q_busy && !join_busy &&
                         !flash_busy && !output_busy &&
                         q_start_ready && join_cmd_ready &&
                         flash_cmd_ready && output_cfg_ready;
    wire command_complete =
        (flash_done_seen_q || (flash_done_valid && !flash_done_error)) &&
        (output_done_seen_q || (output_done_valid && !output_done_error)) &&
        children_idle;

    assign busy = (state_q == ST_RUN) || (state_q == ST_FAIL) ||
                  (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = status_q != 8'd0;
    assign done_status = status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            status_q <= 8'd0;
            history_len_q <= 17'd0;
            layer_kv_base_q <= 64'd0;
            cmd_token_count_q <= 4'd0;
            q_started_q <= 1'b0;
            join_started_q <= 1'b0;
            k_mover_started_q <= 1'b0;
            v_mover_started_q <= 1'b0;
            flash_done_seen_q <= 1'b0;
            output_done_seen_q <= 1'b0;
            local_abort_q <= 1'b0;
            query_outstanding_q <= 5'd0;
        end else begin
            local_abort_q <= 1'b0;

            // Global clear also clears the resident Query arena response-valid
            // pipeline, so its outstanding credit is cancelled here. A local
            // abort leaves the arena intact and must drain the accepted read.
            if (clear) begin
                query_outstanding_q <= 5'd0;
            end else begin
                case ({query_req_fire, query_rsp_fire})
                    2'b10: query_outstanding_q <= query_outstanding_q + 5'd1;
                    2'b01: query_outstanding_q <= query_outstanding_q - 5'd1;
                    default: query_outstanding_q <= query_outstanding_q;
                endcase
            end

            if (clear || abort_run) begin
                status_q <= 8'd0;
                q_started_q <= 1'b0;
                join_started_q <= 1'b0;
                k_mover_started_q <= 1'b0;
                v_mover_started_q <= 1'b0;
                flash_done_seen_q <= 1'b0;
                output_done_seen_q <= 1'b0;
                state_q <= children_idle ? ST_IDLE : ST_ABORT;
            end else begin
                case (state_q)
                    ST_IDLE: if (command_fire) begin
                        status_q <= command_ok ? 8'd0 : STATUS_BAD_CMD;
                        history_len_q <= cmd_history_len;
                        layer_kv_base_q <= cmd_layer_kv_base;
                        cmd_token_count_q <= cmd_token_count;
                        q_started_q <= 1'b0;
                        join_started_q <= 1'b0;
                        k_mover_started_q <= 1'b0;
                        v_mover_started_q <= 1'b0;
                        flash_done_seen_q <= 1'b0;
                        output_done_seen_q <= 1'b0;
                        state_q <= command_ok ? ST_RUN : ST_DONE;
                    end

                    ST_RUN: begin
                        if (q_start_fire)
                            q_started_q <= 1'b1;
                        if (join_start_fire)
                            join_started_q <= 1'b1;
                        if (k_mover_start_fire)
                            k_mover_started_q <= 1'b1;
                        if (v_mover_start_fire)
                            v_mover_started_q <= 1'b1;
                        if (group_req_valid && group_req_ready) begin
                            q_started_q <= 1'b0;
                            join_started_q <= 1'b0;
                            k_mover_started_q <= 1'b0;
                            v_mover_started_q <= 1'b0;
                        end
                        if (flash_done_valid && !flash_done_error)
                            flash_done_seen_q <= 1'b1;
                        if (output_done_valid && !output_done_error)
                            output_done_seen_q <= 1'b1;

                        if (child_error_event) begin
                            status_q <= child_error_status;
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (command_complete) begin
                            status_q <= 8'd0;
                            state_q <= ST_DONE;
                        end
                    end

                    ST_FAIL: if (children_idle)
                        state_q <= ST_DONE;

                    ST_ABORT: if (children_idle)
                        state_q <= ST_IDLE;

                    ST_DONE: if (done_ready)
                        state_q <= ST_IDLE;

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && query_rsp_fire && (query_outstanding_q == 5'd0) &&
            !query_req_fire)
            $fatal(1, " attention_service stale Query response count");
        if (rst_n && group_req_valid && group_req_ready &&
            (group_req_total_kv !=
             history_len_q + {13'd0, cmd_token_count_q}))
            $fatal(1, " attention_service group context skew");
        if (rst_n && group_req_valid &&
            (group_req_q_head_base != {1'b0, group_req_index, 3'b000}))
            $fatal(1, " attention_service group head skew");
        if (rst_n && flash_k_valid && flash_k_ready &&
            join_k_last && !flash_out_last && (state_q != ST_RUN))
            $fatal(1, " attention_service K terminal outside run");
        if (rst_n && flash_v_valid && flash_v_ready &&
            join_v_last && !flash_out_last && (state_q != ST_RUN))
            $fatal(1, " attention_service V terminal outside run");
        if (rst_n && query_outstanding_q > 5'd8)
            $fatal(1, " attention_service Query outstanding overflow");
    end
`endif
endmodule

`default_nettype wire
