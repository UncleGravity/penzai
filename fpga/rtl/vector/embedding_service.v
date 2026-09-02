`default_nettype none

// Token IDs -> resident FP32 R. The random packed-table reader is hidden behind
// the same quad-reader client contract used by projection weights and RoPE.
module embedding_service (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,

    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [63:0]   cmd_table_addr,
    input  wire [5:0]    cmd_q1_blocks,
    input  wire [12:0]   cmd_hidden_dim,
    input  wire [17:0]   cmd_vocab_rows,
    input  wire [1:0]    cmd_weight_fmt,
    input  wire [3:0]    cmd_token_count,
    input  wire [7:0]    cmd_token_mask,
    input  wire [255:0]  cmd_token_ids,

    output wire          read_cmd_valid,
    input  wire          read_cmd_ready,
    output wire [63:0]   read_cmd_base_addr,
    output wire [31:0]   read_cmd_port_beats,
    output wire [3:0]    read_cmd_port_mask,
    output wire          read_abort,
    input  wire [511:0]  read_data,
    input  wire          read_valid,
    output wire          read_ready,
    input  wire          read_last,
    input  wire          read_error,
    input  wire          read_busy,
    input  wire          read_done_valid,
    output wire          read_done_ready,
    input  wire          read_done_error,
    input  wire [7:0]    read_done_status,

    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    output wire          r_wr_wave,
    output wire [11:0]   r_wr_addr,
    output wire [3:0]    r_wr_lane_mask,
    output wire [127:0]  r_wr_data,

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

    localparam [2:0] RD_IDLE   = 3'd0;
    localparam [2:0] RD_CMD    = 3'd1;
    localparam [2:0] RD_STREAM = 3'd2;
    localparam [2:0] RD_WAIT   = 3'd3;
    localparam [2:0] RD_ABORT  = 3'd4;

    localparam [7:0] STATUS_READER = 8'h20;

    reg [2:0] state_q;
    reg [2:0] read_state_q;
    reg [63:0] read_addr_q;
    reg [7:0] done_status_q;
    reg decoder_done_q;
    reg local_abort_q;

    wire decoder_start_ready;
    wire decoder_busy;
    wire decoder_done;
    wire decoder_error;
    wire [7:0] decoder_status;
    wire decoder_mem_req_valid;
    wire decoder_mem_req_ready;
    wire [63:0] decoder_mem_req_addr;
    wire decoder_mem_rsp_valid;
    wire decoder_mem_rsp_ready;
    wire [127:0] decoder_mem_rsp_data;
    wire decoder_mem_rsp_error;
    wire decoder_out_valid;
    wire decoder_out_ready;
    wire [2:0] decoder_out_token;
    wire [11:0] decoder_out_index;
    wire [127:0] decoder_out_data;
    wire decoder_out_last;

    wire store_cfg_ready;
    wire store_busy;
    wire store_done_valid;
    wire store_done_error;
    wire [7:0] store_done_status;

    wire command_shape_ok = (cmd_hidden_dim != 13'd0) &&
                            (cmd_hidden_dim[6:0] == 7'd0) &&
                            ({7'd0, cmd_q1_blocks} << 7) == cmd_hidden_dim;
    wire command_fire = cmd_valid && cmd_ready;
    wire local_clear = clear || local_abort_q;

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE) &&
                       (read_state_q == RD_IDLE) &&
                       decoder_start_ready && store_cfg_ready;

     embedding_decode u_decode (
        .clk(clk), .rst_n(rst_n), .clear(local_clear || abort_run),
        .start_valid(command_fire), .start_ready(decoder_start_ready),
        .table_addr(cmd_table_addr), .q1_blocks(cmd_q1_blocks),
        .vocab_rows(cmd_vocab_rows), .weight_fmt(cmd_weight_fmt),
        .token_count(cmd_token_count), .token_mask(cmd_token_mask),
        .token_ids(cmd_token_ids),
        .mem_req_valid(decoder_mem_req_valid),
        .mem_req_ready(decoder_mem_req_ready),
        .mem_req_addr(decoder_mem_req_addr),
        .mem_rsp_valid(decoder_mem_rsp_valid),
        .mem_rsp_ready(decoder_mem_rsp_ready),
        .mem_rsp_data(decoder_mem_rsp_data),
        .mem_rsp_error(decoder_mem_rsp_error),
        .out_valid(decoder_out_valid), .out_ready(decoder_out_ready),
        .out_token(decoder_out_token), .out_index(decoder_out_index),
        .out_data(decoder_out_data), .out_last(decoder_out_last),
        .busy(decoder_busy), .done(decoder_done),
        .error(decoder_error), .status(decoder_status)
    );

     embedding_store4 u_store (
        .clk(clk), .rst_n(rst_n), .abort_run(local_clear || abort_run),
        .cfg_valid(command_fire), .cfg_ready(store_cfg_ready),
        .cfg_rows(cmd_hidden_dim), .cfg_token_count(cmd_token_count),
        .cfg_token_mask(cmd_token_mask),
        .in_valid(decoder_out_valid), .in_ready(decoder_out_ready),
        .in_token(decoder_out_token), .in_index(decoder_out_index),
        .in_data(decoder_out_data), .in_last(decoder_out_last),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_wave(r_wr_wave), .r_wr_addr(r_wr_addr),
        .r_wr_lane_mask(r_wr_lane_mask), .r_wr_data(r_wr_data),
        .busy(store_busy), .done_valid(store_done_valid),
        .done_ready(1'b1), .done_error(store_done_error),
        .done_status(store_done_status)
    );

    // Each decoder request is one flat 128-bit read on quad-reader port zero.
    // Capture the decoder address before arbitration so embedding counters and
    // address arithmetic cannot feed the shared weight-reader control cone.
    wire decoder_request_capture = (read_state_q == RD_IDLE) &&
                                   (state_q == ST_RUN) &&
                                   decoder_mem_req_valid &&
                                   !clear && !abort_run;
    assign read_cmd_valid = (read_state_q == RD_CMD) &&
                            (state_q == ST_RUN) &&
                            !clear && !abort_run;
    // The decoder owns an outstanding response only after the registered
    // command actually leaves this service. A clear may therefore discard an
    // unissued RD_CMD entry without creating an undrainable decoder response.
    assign decoder_mem_req_ready = read_cmd_valid && read_cmd_ready;
    assign read_cmd_base_addr = read_addr_q;
    assign read_cmd_port_beats = 32'd1;
    assign read_cmd_port_mask = 4'h1;
    assign read_abort = abort_run || local_abort_q ||
                        (read_state_q == RD_ABORT);
    assign decoder_mem_rsp_valid = (read_state_q == RD_STREAM) && read_valid;
    assign decoder_mem_rsp_data = read_data[127:0];
    assign decoder_mem_rsp_error = read_error;
    assign read_ready = (read_state_q == RD_STREAM) &&
                        decoder_mem_rsp_ready;
    assign read_done_ready = read_state_q == RD_WAIT;

    assign busy = (state_q == ST_RUN) || (state_q == ST_FAIL) ||
                  (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_status_q != 8'd0;
    assign done_status = done_status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            read_state_q <= RD_IDLE;
            read_addr_q <= 64'd0;
            done_status_q <= 8'd0;
            decoder_done_q <= 1'b0;
            local_abort_q <= 1'b0;
        end else begin
            local_abort_q <= 1'b0;

            if (clear || abort_run) begin
                state_q <= read_busy ? ST_ABORT : ST_IDLE;
                read_state_q <= read_busy ? RD_ABORT : RD_IDLE;
                done_status_q <= 8'd0;
                decoder_done_q <= 1'b0;
            end else begin
                case (read_state_q)
                    RD_IDLE: if (decoder_request_capture) begin
                        read_addr_q <= decoder_mem_req_addr;
                        read_state_q <= RD_CMD;
                    end
                    RD_CMD: if (read_cmd_valid && read_cmd_ready)
                        read_state_q <= RD_STREAM;
                    RD_STREAM: if (read_valid && read_ready && read_last)
                        read_state_q <= RD_WAIT;
                    RD_WAIT: if (read_done_valid) begin
                        if (read_done_error && (state_q == ST_RUN)) begin
                            done_status_q <= read_done_status == 8'd0 ?
                                             STATUS_READER : read_done_status;
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end
                        read_state_q <= RD_IDLE;
                    end
                    RD_ABORT: if (!read_busy && read_cmd_ready)
                        read_state_q <= RD_IDLE;
                    default: read_state_q <= RD_IDLE;
                endcase

                case (state_q)
                    ST_IDLE: if (command_fire) begin
                        decoder_done_q <= 1'b0;
                        done_status_q <= command_shape_ok ? 8'd0 : 8'h01;
                        if (command_shape_ok)
                            state_q <= ST_RUN;
                        else begin
                            local_abort_q <= 1'b1;
                            state_q <= ST_DONE;
                        end
                    end

                    ST_RUN: begin
                        if (decoder_done) begin
                            decoder_done_q <= !decoder_error;
                            if (decoder_error) begin
                                done_status_q <= decoder_status;
                                local_abort_q <= 1'b1;
                                state_q <= ST_FAIL;
                            end
                        end
                        if (store_done_valid) begin
                            if (store_done_error) begin
                                done_status_q <= store_done_status;
                                local_abort_q <= 1'b1;
                                state_q <= ST_FAIL;
                            end else if (decoder_done_q ||
                                         (decoder_done && !decoder_error)) begin
                                state_q <= ST_DONE;
                            end
                        end
                    end

                    ST_FAIL: if ((read_state_q == RD_IDLE) && !read_busy)
                        state_q <= ST_DONE;

                    ST_ABORT: if ((read_state_q == RD_IDLE) && !read_busy)
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
        if (rst_n && command_fire && !command_shape_ok)
            assert(!decoder_start_ready || store_cfg_ready);
    end
`endif
endmodule

`default_nettype wire
