`default_nettype none

// Transpose one Flash 8-head group from token/head/beat order into the
// coordinate-major four-lane stream consumed by one shared Q8 packer. A single
// group buffer is reused for wave 0 and wave 1; Flash is backpressured while
// each wave is quantized, so no full attention tensor reaches DDR.
module flash_output_q8 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [3:0]    cfg_token_count,
    input  wire [7:0]    cfg_token_mask,
    input  wire [5:0]    cfg_q_heads,

    input  wire [255:0]  in_data,
    input  wire          in_valid,
    output wire          in_ready,
    input  wire [2:0]    in_token,
    input  wire [5:0]    in_head,
    input  wire [3:0]    in_beat,
    input  wire          in_group_last,
    input  wire          in_last,

    output wire          q8_wr_valid,
    input  wire          q8_wr_ready,
    output wire          q8_wr_wave,
    output wire [8:0]    q8_wr_addr,
    output wire [3:0]    q8_wr_lane_mask,
    output wire [1087:0] q8_wr_data,

    // Ownership lasts from cfg acceptance through the final output handshake.
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

    output wire          busy,
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [7:0]    done_status
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_INGEST = 3'd1;
    localparam [2:0] ST_QCFG   = 3'd2;
    localparam [2:0] ST_QRUN   = 3'd3;
    localparam [2:0] ST_FAIL   = 3'd4;
    localparam [2:0] ST_DONE   = 3'd5;

    localparam [7:0] STATUS_BAD_CFG = 8'h01;
    localparam [7:0] STATUS_ORDER   = 8'h02;
    localparam [7:0] STATUS_FRAME   = 8'h04;
    localparam [7:0] STATUS_Q8      = 8'h08;

    reg [2:0] state_q;
    reg [3:0] token_count_q;
    reg [7:0] token_mask_q;
    reg [2:0] group_count_q;
    reg [2:0] group_q;
    reg [2:0] token_q;
    reg [2:0] head_q;
    reg [3:0] beat_q;
    reg wave_q;
    reg ingest_wr_valid_q;
    reg [1:0] ingest_wr_lane_q;
    reg [6:0] ingest_wr_addr_q;
    reg [255:0] ingest_wr_data_q;
    reg [6:0] drain_word_q;
    reg [2:0] drain_scalar_q;
    reg drain_pending_q;
    reg drain_mem_valid_q;
    reg [6:0] drain_mem_word_q;
    reg drain_valid_q;
    reg [6:0] drain_hold_word_q;
    reg drain_input_complete_q;
    reg [255:0] drain_lane0_q;
    reg [255:0] drain_lane1_q;
    reg [255:0] drain_lane2_q;
    reg [255:0] drain_lane3_q;
    reg [255:0] drain_hold_lane0_q;
    reg [255:0] drain_hold_lane1_q;
    reg [255:0] drain_hold_lane2_q;
    reg [255:0] drain_hold_lane3_q;
    reg error_q;
    reg [7:0] status_q;

    // One group is 8 heads * 16 beats. Each beat already holds eight FP32
    // scalars, so a single wide write and read maps cleanly to block RAM.
    (* ram_style = "block" *) reg [255:0] lane0_mem [0:127];
    (* ram_style = "block" *) reg [255:0] lane1_mem [0:127];
    (* ram_style = "block" *) reg [255:0] lane2_mem [0:127];
    (* ram_style = "block" *) reg [255:0] lane3_mem [0:127];

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

    wire cfg_ok = (cfg_token_count >= 4'd1) &&
                  (cfg_token_count <= 4'd8) &&
                  (cfg_token_mask == prefix_mask8(cfg_token_count)) &&
                  ((cfg_q_heads == 6'd16) || (cfg_q_heads == 6'd32));
    wire [5:0] expected_head = {group_q, 3'b000} + {3'd0, head_q};
    wire expected_group_last =
        ({1'b0, token_q} + 4'd1 == token_count_q) &&
        (head_q == 3'd7) && (beat_q == 4'd15);
    wire expected_last = expected_group_last &&
                         ({1'b0, group_q} + 4'd1 ==
                          {1'b0, group_count_q});
    wire input_ok = (in_token == token_q) &&
                    (in_head == expected_head) &&
                    (in_beat == beat_q);
    wire input_frame_ok = (in_group_last == expected_group_last) &&
                          (in_last == expected_last);
    wire in_fire = in_valid && in_ready;
    wire wave_boundary = (head_q == 3'd7) && (beat_q == 4'd15) &&
                         ((token_q[1:0] == 2'd3) ||
                          ({1'b0, token_q} + 4'd1 == token_count_q));
    wire [6:0] ingest_addr = {head_q, beat_q};

    assign cfg_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign in_ready = rst_n && !clear && !abort_run &&
                      (state_q == ST_INGEST);

    wire [3:0] active_lane_mask = wave_q ? token_mask_q[7:4] :
                                                  token_mask_q[3:0];
    wire q8_out_fire = leaf_q8_out_valid && leaf_q8_out_ready;

    wire [127:0] drain_data = {
        drain_hold_lane3_q[drain_scalar_q*32 +: 32],
        drain_hold_lane2_q[drain_scalar_q*32 +: 32],
        drain_hold_lane1_q[drain_scalar_q*32 +: 32],
        drain_hold_lane0_q[drain_scalar_q*32 +: 32]
    };
    wire drain_fire = drain_valid_q && leaf_q8_in_ready;
    wire drain_hold_release = drain_fire && (drain_scalar_q == 3'd7);
    wire drain_hold_slot_available = !drain_valid_q || drain_hold_release;
    wire drain_mem_to_hold = drain_mem_valid_q &&
                             drain_hold_slot_available;
    wire drain_mem_slot_available = !drain_mem_valid_q || drain_mem_to_hold;
    wire drain_read_issue = drain_pending_q && drain_mem_slot_available;

    assign leaf_q8_cfg_valid = rst_n && !clear && !abort_run &&
                               (state_q == ST_QCFG);
    assign leaf_q8_cfg_rows = 15'd1024;
    assign leaf_q8_cfg_lane_mask = active_lane_mask;
    assign leaf_q8_abort = clear || abort_run || (state_q == ST_FAIL);
    assign leaf_q8_in_valid = (state_q == ST_QRUN) && drain_valid_q;
    assign leaf_q8_in_data = drain_data;
    assign leaf_q8_out_ready = (state_q == ST_QRUN) && q8_wr_ready;

    assign q8_wr_valid = (state_q == ST_QRUN) && leaf_q8_out_valid;
    assign q8_wr_wave = wave_q;
    assign q8_wr_addr = {group_q, 5'b0} + leaf_q8_out_block;
    assign q8_wr_lane_mask = active_lane_mask;
    assign q8_wr_data = leaf_q8_out_data;

    assign busy = (state_q != ST_IDLE) && (state_q != ST_DONE);
    assign done_valid = state_q == ST_DONE;
    assign done_error = error_q;
    assign done_status = status_q;

    // RAM payload ports are isolated from state and clear. An accepted input is
    // written from a local intent register on the following edge; an in-flight
    // intent may retire on the clear edge, but no intent survives the clear.
    always @(posedge clk) begin
        if (ingest_wr_valid_q) begin
            case (ingest_wr_lane_q)
                2'd0: lane0_mem[ingest_wr_addr_q] <= ingest_wr_data_q;
                2'd1: lane1_mem[ingest_wr_addr_q] <= ingest_wr_data_q;
                2'd2: lane2_mem[ingest_wr_addr_q] <= ingest_wr_data_q;
                default: lane3_mem[ingest_wr_addr_q] <= ingest_wr_data_q;
            endcase
        end
        if (drain_read_issue) begin
            drain_lane0_q <= lane0_mem[drain_word_q];
            drain_lane1_q <= lane1_mem[drain_word_q];
            drain_lane2_q <= lane2_mem[drain_word_q];
            drain_lane3_q <= lane3_mem[drain_word_q];
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear || abort_run) begin
            state_q <= ST_IDLE;
            token_count_q <= 4'd0;
            token_mask_q <= 8'd0;
            group_count_q <= 3'd0;
            group_q <= 3'd0;
            token_q <= 3'd0;
            head_q <= 3'd0;
            beat_q <= 4'd0;
            wave_q <= 1'b0;
            ingest_wr_valid_q <= 1'b0;
            drain_word_q <= 7'd0;
            drain_scalar_q <= 3'd0;
            drain_pending_q <= 1'b0;
            drain_mem_valid_q <= 1'b0;
            drain_valid_q <= 1'b0;
            drain_input_complete_q <= 1'b0;
            error_q <= 1'b0;
            status_q <= 8'd0;
        end else begin
            ingest_wr_valid_q <= 1'b0;

            if (drain_read_issue) begin
                drain_pending_q <= 1'b0;
                drain_mem_valid_q <= 1'b1;
                drain_mem_word_q <= drain_word_q;
            end else if (drain_mem_to_hold) begin
                drain_mem_valid_q <= 1'b0;
            end
            if (drain_mem_to_hold) begin
                drain_hold_lane0_q <= drain_lane0_q;
                drain_hold_lane1_q <= drain_lane1_q;
                drain_hold_lane2_q <= drain_lane2_q;
                drain_hold_lane3_q <= drain_lane3_q;
                drain_hold_word_q <= drain_mem_word_q;
                drain_valid_q <= 1'b1;
                if (drain_mem_word_q != 7'd127) begin
                    drain_word_q <= drain_mem_word_q + 1'b1;
                    drain_pending_q <= 1'b1;
                end
            end else if (drain_hold_release) begin
                drain_valid_q <= 1'b0;
            end

            case (state_q)
                ST_IDLE: if (cfg_valid) begin
                    token_count_q <= cfg_token_count;
                    token_mask_q <= cfg_token_mask;
                    group_count_q <= (cfg_q_heads == 6'd16) ? 3'd2 : 3'd4;
                    group_q <= 3'd0;
                    token_q <= 3'd0;
                    head_q <= 3'd0;
                    beat_q <= 4'd0;
                    wave_q <= 1'b0;
                    error_q <= !cfg_ok;
                    status_q <= cfg_ok ? 8'd0 : STATUS_BAD_CFG;
                    state_q <= cfg_ok ? ST_INGEST : ST_DONE;
                end

                ST_INGEST: if (in_fire) begin
                    if (!input_ok || !input_frame_ok) begin
                        error_q <= 1'b1;
                        status_q <= !input_ok ? STATUS_ORDER : STATUS_FRAME;
                        state_q <= ST_FAIL;
                    end else begin
                        ingest_wr_valid_q <= 1'b1;
                        ingest_wr_lane_q <= token_q[1:0];
                        ingest_wr_addr_q <= ingest_addr;
                        ingest_wr_data_q <= in_data;
                        if (wave_boundary) begin
                            wave_q <= token_q[2];
                            drain_word_q <= 7'd0;
                            drain_scalar_q <= 3'd0;
                            drain_pending_q <= 1'b0;
                            drain_mem_valid_q <= 1'b0;
                            drain_valid_q <= 1'b0;
                            drain_input_complete_q <= 1'b0;
                            state_q <= ST_QCFG;
                        end else if (beat_q != 4'd15) begin
                            beat_q <= beat_q + 1'b1;
                        end else if (head_q != 3'd7) begin
                            beat_q <= 4'd0;
                            head_q <= head_q + 1'b1;
                        end else begin
                            beat_q <= 4'd0;
                            head_q <= 3'd0;
                            token_q <= token_q + 1'b1;
                        end
                    end
                end

                ST_QCFG: if (leaf_q8_cfg_ready) begin
                    drain_word_q <= 7'd0;
                    drain_scalar_q <= 3'd0;
                    drain_pending_q <= 1'b1;
                    drain_mem_valid_q <= 1'b0;
                    drain_valid_q <= 1'b0;
                    drain_input_complete_q <= 1'b0;
                    state_q <= ST_QRUN;
                end

                ST_QRUN: begin
                    if (drain_fire) begin
                        if (drain_scalar_q != 3'd7) begin
                            drain_scalar_q <= drain_scalar_q + 1'b1;
                        end else begin
                            drain_scalar_q <= 3'd0;
                            if (drain_hold_word_q == 7'd127)
                                drain_input_complete_q <= 1'b1;
                        end
                    end

                    if (q8_out_fire && (leaf_q8_out_status != 8'd0)) begin
                        error_q <= 1'b1;
                        status_q <= STATUS_Q8;
                        state_q <= ST_FAIL;
                    end else if (q8_out_fire && leaf_q8_out_last) begin
                        if (!drain_input_complete_q) begin
                            error_q <= 1'b1;
                            status_q <= STATUS_Q8;
                            state_q <= ST_FAIL;
                        end else if (!wave_q && (token_count_q > 4'd4)) begin
                            token_q <= 3'd4;
                            head_q <= 3'd0;
                            beat_q <= 4'd0;
                            state_q <= ST_INGEST;
                        end else if ({1'b0, group_q} + 4'd1 <
                                     {1'b0, group_count_q}) begin
                            group_q <= group_q + 1'b1;
                            token_q <= 3'd0;
                            head_q <= 3'd0;
                            beat_q <= 4'd0;
                            wave_q <= 1'b0;
                            state_q <= ST_INGEST;
                        end else begin
                            state_q <= ST_DONE;
                        end
                    end
                end

                ST_FAIL: if (!leaf_q8_busy)
                    state_q <= ST_DONE;

                ST_DONE: if (done_ready)
                    state_q <= ST_IDLE;

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && !abort_run && q8_wr_valid && !q8_wr_ready) begin
            assert($stable(q8_wr_wave));
            assert($stable(q8_wr_addr));
            assert($stable(q8_wr_lane_mask));
            assert($stable(q8_wr_data));
        end
    end
`endif
endmodule

`default_nettype wire
