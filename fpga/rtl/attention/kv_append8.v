`default_nettype none

// Publish one resident tile-8 NewKV tile into the layer-major DDR KV image.
// Each token owns one 4 KiB record: 2 KiB K followed by 2 KiB V, ordered
// head then 128-wide dimension. The external writer command therefore uses
// one 256-beat segment, a 4 KiB stride, and one repeat per active token.
module kv_append8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         abort_run,

    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire [63:0]  cmd_layer_kv_base,
    input  wire [16:0]  cmd_position_base,
    input  wire [3:0]   cmd_token_count,
    input  wire [7:0]   cmd_token_mask,

    output wire         newkv_rd_req_valid,
    input  wire         newkv_rd_req_ready,
    output wire         newkv_rd_req_wave,
    output wire [10:0]  newkv_rd_req_addr,
    input  wire         newkv_rd_rsp_valid,
    output wire         newkv_rd_rsp_ready,
    input  wire [63:0]  newkv_rd_rsp_data,
    input  wire         newkv_rd_rsp_error,

    output wire         wr_cmd_valid,
    input  wire         wr_cmd_ready,
    output wire [63:0]  wr_cmd_addr,
    output wire [31:0]  wr_cmd_segment_beats,
    output wire [31:0]  wr_cmd_stride_bytes,
    output wire [16:0]  wr_cmd_repeats,

    output wire [127:0] wr_data,
    output wire         wr_valid,
    input  wire         wr_ready,
    output wire         wr_last,
    output wire         wr_error,
    input  wire         wr_busy,
    input  wire         wr_done_valid,
    output wire         wr_done_ready,
    input  wire         wr_done_error,
    input  wire [7:0]   wr_done_status,

    output wire         busy,
    output wire         done_valid,
    input  wire         done_ready,
    output wire         done_error,
    output wire [7:0]   done_status
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_WCMD   = 3'd1;
    localparam [2:0] ST_RUN    = 3'd2;
    localparam [2:0] ST_WAIT   = 3'd3;
    localparam [2:0] ST_ABORT  = 3'd4;
    localparam [2:0] ST_DONE   = 3'd5;

    localparam [7:0] STATUS_BAD_CMD = 8'h01;
    localparam [7:0] STATUS_ARENA   = 8'h02;

    reg [2:0] state_q;
    reg [63:0] layer_kv_base_q;
    reg [16:0] position_base_q;
    reg [3:0] token_count_q;
    reg [2:0] token_q;
    reg kind_q;
    reg [2:0] head_q;
    reg [3:0] beat_q;
    reg [2:0] scalar_q;
    reg arena_outstanding_q;
    reg [1:0] arena_lane_q;
    reg [127:0] out_data_q;
    reg out_valid_q;
    reg out_last_q;
    reg out_error_q;
    reg error_q;
    reg [7:0] status_q;

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

    wire [17:0] position_end =
        {1'b0, cmd_position_base} + {14'd0, cmd_token_count};
    wire [63:0] position_offset = {35'd0, cmd_position_base, 12'd0};
    wire command_ok = (cmd_token_count >= 4'd1) &&
                      (cmd_token_count <= 4'd8) &&
                      (cmd_token_mask == prefix_mask8(cmd_token_count)) &&
                      (cmd_layer_kv_base[11:0] == 12'd0) &&
                      (position_end <= 18'd65536) &&
                      ((cmd_layer_kv_base + position_offset) >=
                       cmd_layer_kv_base);

    wire arena_req_fire = newkv_rd_req_valid && newkv_rd_req_ready;
    wire arena_rsp_fire = newkv_rd_rsp_valid && newkv_rd_rsp_ready;
    wire out_fire = wr_valid && wr_ready;
    wire final_beat = ({1'b0, token_q} + 4'd1 == token_count_q) &&
                      kind_q && (head_q == 3'd7) && (beat_q == 4'd15);
    wire [15:0] arena_scalar =
        (arena_lane_q == 2'd0) ? newkv_rd_rsp_data[15:0] :
        (arena_lane_q == 2'd1) ? newkv_rd_rsp_data[31:16] :
        (arena_lane_q == 2'd2) ? newkv_rd_rsp_data[47:32] :
                                 newkv_rd_rsp_data[63:48];

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign newkv_rd_req_valid = (state_q == ST_RUN) && !abort_run &&
                                !arena_outstanding_q && !out_valid_q;
    assign newkv_rd_req_wave = token_q[2];
    assign newkv_rd_req_addr = {kind_q, head_q, beat_q, scalar_q};
    assign newkv_rd_rsp_ready = arena_outstanding_q &&
        ((state_q == ST_RUN) || (state_q == ST_ABORT));

    assign wr_cmd_valid = state_q == ST_WCMD;
    assign wr_cmd_addr = layer_kv_base_q +
                         {35'd0, position_base_q, 12'd0};
    assign wr_cmd_segment_beats = 32'd256;
    assign wr_cmd_stride_bytes = 32'd4096;
    assign wr_cmd_repeats = {13'd0, token_count_q};
    assign wr_data = out_data_q;
    assign wr_valid = (state_q == ST_RUN) && out_valid_q;
    assign wr_last = out_last_q;
    assign wr_error = out_error_q;
    assign wr_done_ready = (state_q == ST_WAIT) ||
                           (state_q == ST_ABORT);

    assign busy = (state_q != ST_IDLE) && (state_q != ST_DONE);
    assign done_valid = state_q == ST_DONE;
    assign done_error = error_q;
    assign done_status = status_q;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q <= ST_IDLE;
            layer_kv_base_q <= 64'd0;
            position_base_q <= 17'd0;
            token_count_q <= 4'd0;
            token_q <= 3'd0;
            kind_q <= 1'b0;
            head_q <= 3'd0;
            beat_q <= 4'd0;
            scalar_q <= 3'd0;
            arena_outstanding_q <= 1'b0;
            arena_lane_q <= 2'd0;
            out_data_q <= 128'd0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_error_q <= 1'b0;
            error_q <= 1'b0;
            status_q <= 8'd0;
        end else begin
            if ((state_q == ST_RUN) && abort_run) begin
                out_valid_q <= 1'b0;
                state_q <= ST_ABORT;
            end else case (state_q)
                ST_IDLE: if (cmd_valid) begin
                    layer_kv_base_q <= cmd_layer_kv_base;
                    position_base_q <= cmd_position_base;
                    token_count_q <= cmd_token_count;
                    token_q <= 3'd0;
                    kind_q <= 1'b0;
                    head_q <= 3'd0;
                    beat_q <= 4'd0;
                    scalar_q <= 3'd0;
                    arena_outstanding_q <= 1'b0;
                    out_valid_q <= 1'b0;
                    out_error_q <= 1'b0;
                    error_q <= !command_ok;
                    status_q <= command_ok ? 8'd0 : STATUS_BAD_CMD;
                    state_q <= command_ok ? ST_WCMD : ST_DONE;
                end

                ST_WCMD: begin
                    if (abort_run)
                        state_q <= ST_IDLE;
                    else if (wr_cmd_ready)
                        state_q <= ST_RUN;
                end

                ST_RUN: begin
                    if (arena_req_fire) begin
                        arena_outstanding_q <= 1'b1;
                        arena_lane_q <= token_q[1:0];
                    end
                    if (arena_rsp_fire) begin
                        arena_outstanding_q <= 1'b0;
                        out_data_q[scalar_q*16 +: 16] <= arena_scalar;
                        if (newkv_rd_rsp_error) begin
                            out_data_q <= 128'd0;
                            out_error_q <= 1'b1;
                            out_last_q <= final_beat;
                            out_valid_q <= 1'b1;
                            error_q <= 1'b1;
                            status_q <= STATUS_ARENA;
                        end else if (scalar_q == 3'd7) begin
                            out_last_q <= final_beat;
                            out_valid_q <= 1'b1;
                            scalar_q <= 3'd0;
                        end else begin
                            scalar_q <= scalar_q + 1'b1;
                        end
                    end
                    if (out_fire) begin
                        out_valid_q <= 1'b0;
                        if (out_error_q || out_last_q) begin
                            state_q <= ST_WAIT;
                        end else if (beat_q != 4'd15) begin
                            beat_q <= beat_q + 1'b1;
                        end else if (head_q != 3'd7) begin
                            beat_q <= 4'd0;
                            head_q <= head_q + 1'b1;
                        end else if (!kind_q) begin
                            beat_q <= 4'd0;
                            head_q <= 3'd0;
                            kind_q <= 1'b1;
                        end else begin
                            beat_q <= 4'd0;
                            head_q <= 3'd0;
                            kind_q <= 1'b0;
                            token_q <= token_q + 1'b1;
                        end
                    end
                end

                ST_WAIT: if (wr_done_valid) begin
                    if (wr_done_error && !error_q) begin
                        error_q <= 1'b1;
                        status_q <= 8'h80 | wr_done_status;
                    end
                    state_q <= ST_DONE;
                end

                ST_ABORT: begin
                    if (arena_rsp_fire)
                        arena_outstanding_q <= 1'b0;
                    if (!arena_outstanding_q && !wr_busy)
                        state_q <= ST_IDLE;
                end

                ST_DONE: if (done_ready)
                    state_q <= ST_IDLE;

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && $past(rst_n && !clear) &&
            $past(wr_valid && !wr_ready)) begin
            assert(wr_valid);
            assert($stable(wr_data));
            assert($stable(wr_last));
            assert($stable(wr_error));
        end
        if (rst_n && !clear && arena_rsp_fire && !arena_outstanding_q)
            $fatal(1, " kv_append8 response without request");
    end
`endif
endmodule

`default_nettype wire
