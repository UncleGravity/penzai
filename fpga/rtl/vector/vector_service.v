// Full-vector RMSNorm service for the whole-token engine.
//
// Attention, FFN, and final norms share this one four-lane datapath. Q/K head norm and
// RoPE belong to projection_sink because fused QKV values never enter an
// arena before normalization. Projection residuals likewise stay in that sink.
// This service is a client of, but does not instantiate, the shared Q8 packer.

`default_nettype none
`include "vector_defs.vh"

module vector_service (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,

    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [3:0]    cmd_op,
    input  wire [7:0]    cmd_token_mask,
    input  wire [7:0]    cmd_hidden_blocks,
    input  wire [63:0]   cmd_gamma_addr,
    input  wire [31:0]   cmd_epsilon,
    input  wire          abort_run,

    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [15:0]   done_status,
    output wire [31:0]   done_cycles,
    output wire          busy,
    output wire [3:0]    debug_state,

    output wire          gamma_req_valid,
    input  wire          gamma_req_ready,
    output wire [63:0]   gamma_req_addr,
    output wire [10:0]   gamma_req_words,
    input  wire          gamma_rsp_valid,
    output wire          gamma_rsp_ready,
    input  wire [127:0]  gamma_rsp_data,
    input  wire          gamma_rsp_last,
    input  wire          gamma_rsp_error,

    output wire          r_rd_req_valid,
    input  wire          r_rd_req_ready,
    output wire          r_rd_req_wave,
    output wire [11:0]   r_rd_req_addr,
    input  wire          r_rd_rsp_valid,
    output wire          r_rd_rsp_ready,
    input  wire [127:0]  r_rd_rsp_data,
    input  wire          r_rd_rsp_error,

    output wire          q8_wr_valid,
    input  wire          q8_wr_ready,
    output wire          q8_wr_wave,
    output wire [8:0]    q8_wr_addr,
    output wire [3:0]    q8_wr_lane_mask,
    output wire [1087:0] q8_wr_data,

    // Ownership lasts from cfg acceptance through the out_last handshake.
    output wire          leaf_q8_cfg_valid,
    input  wire          leaf_q8_cfg_ready,
    output wire [14:0]   leaf_q8_cfg_rows,
    output wire [3:0]    leaf_q8_cfg_lane_mask,
    input  wire          leaf_q8_busy,
    output wire          leaf_q8_in_valid,
    input  wire          leaf_q8_in_ready,
    output wire [127:0]  leaf_q8_in_data,
    input  wire          leaf_q8_out_valid,
    output wire          leaf_q8_out_ready,
    input  wire [8:0]    leaf_q8_out_block,
    input  wire [1087:0] leaf_q8_out_data,
    input  wire [7:0]    leaf_q8_out_status,
    input  wire          leaf_q8_out_last
);
    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_RESET       = 4'd1;
    localparam [3:0] ST_GAMMA_REQ   = 4'd2;
    localparam [3:0] ST_GAMMA_LOAD  = 4'd3;
    localparam [3:0] ST_REDUCE_CFG  = 4'd4;
    localparam [3:0] ST_REDUCE_WAIT = 4'd5;
    localparam [3:0] ST_Q8_CFG      = 4'd6;
    localparam [3:0] ST_APPLY_CFG   = 4'd7;
    localparam [3:0] ST_STREAM      = 4'd8;
    localparam [3:0] ST_DONE        = 4'd9;
    localparam [3:0] ST_ABORT       = 4'd10;

    reg [3:0] state_q;
    reg [3:0] op_q;
    reg [7:0] token_mask_q;
    reg [12:0] rows_q;
    reg [63:0] gamma_addr_q;
    reg [31:0] epsilon_q;
    reg [31:0] cycle_q;
    reg done_error_q;
    reg [15:0] status_q;
    reg wave_q;
    reg [127:0] inv_rms_q;

    (* ram_style = "block" *) reg [127:0] gamma_mem [0:1023];
    reg [10:0] gamma_count_q;
    reg gamma_active_q;
    reg gamma_stream_error_q;

    reg [13:0] read_issue_count_q;
    reg [13:0] read_rsp_count_q;
    reg [14:0] r_outstanding_q;
    reg gamma_read_valid_q;
    reg [127:0] gamma_read_data_q;
    reg [127:0] gamma_read_word_q;
    reg [1:0] gamma_read_select_q;
    reg norm_stage_valid_q;
    reg [127:0] norm_stage_data_q;
    reg [127:0] norm_stage_gamma_q;

    function automatic [3:0] wave_mask(
        input [7:0] mask,
        input wave
    );
        wave_mask = wave ? mask[7:4] : mask[3:0];
    endfunction

    function automatic gamma_finite(input [127:0] value);
        gamma_finite = (value[30:23] != 8'hff) &&
                       (value[62:55] != 8'hff) &&
                       (value[94:87] != 8'hff) &&
                       (value[126:119] != 8'hff);
    endfunction

    function automatic [31:0] gamma_scalar(
        input [127:0] word,
        input [1:0] select
    );
        begin
            case (select)
                2'd0: gamma_scalar = word[31:0];
                2'd1: gamma_scalar = word[63:32];
                2'd2: gamma_scalar = word[95:64];
                default: gamma_scalar = word[127:96];
            endcase
        end
    endfunction

    wire cmd_op_ok = (cmd_op == `VECTOR_OP_ATTN_NORM) ||
                     (cmd_op == `VECTOR_OP_FFN_NORM) ||
                     (cmd_op == `VECTOR_OP_FINAL_NORM);
    wire cmd_d_ok = (cmd_hidden_blocks == 8'd64) ||
                    (cmd_hidden_blocks == 8'd80) ||
                    (cmd_hidden_blocks == 8'd128);
    wire cmd_eps_ok = !cmd_epsilon[31] &&
                      (cmd_epsilon[30:23] != 8'd0) &&
                      (cmd_epsilon[30:23] != 8'hff);
    wire cmd_gamma_ok = (cmd_gamma_addr != 64'd0) &&
                        (cmd_gamma_addr[5:0] == 6'd0);
    wire cmd_ok = cmd_op_ok && cmd_d_ok && (cmd_token_mask != 8'd0) &&
                  cmd_eps_ok && cmd_gamma_ok;

    wire [3:0] current_lane_mask = wave_mask(token_mask_q, wave_q);
    wire has_next_wave = !wave_q && (|token_mask_q[7:4]);
    wire [10:0] gamma_words = rows_q[12:2];

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    wire cmd_fire = cmd_valid && cmd_ready;
    assign busy = state_q != ST_IDLE;
    assign debug_state = state_q;
    assign done_valid = rst_n && !clear && !abort_run &&
                        (state_q == ST_DONE);
    assign done_error = done_error_q;
    assign done_status = status_q;
    assign done_cycles = cycle_q;

    assign gamma_req_valid = !clear && !abort_run &&
                             (state_q == ST_GAMMA_REQ);
    assign gamma_req_addr = gamma_addr_q;
    assign gamma_req_words = gamma_words;
    wire gamma_req_fire = gamma_req_valid && gamma_req_ready;
    assign gamma_rsp_ready = !clear && ((state_q == ST_GAMMA_LOAD) ||
                                       (state_q == ST_ABORT));
    wire gamma_rsp_fire = gamma_rsp_valid && gamma_rsp_ready;
    wire gamma_expected_last =
        (gamma_count_q + 1'b1) == gamma_words;

    wire child_abort = clear || abort_run || (state_q == ST_RESET);
    wire reduce_cfg_ready;
    wire reduce_busy;
    wire reduce_src_req_valid;
    wire reduce_src_req_ready;
    wire [11:0] reduce_src_req_addr;
    wire reduce_src_rsp_valid;
    wire reduce_src_rsp_ready;
    wire [127:0] reduce_src_rsp_data;
    wire reduce_src_rsp_error;
    wire reduce_result_valid;
    wire reduce_result_error;
    wire [7:0] reduce_result_status;
    wire [127:0] reduce_result_inv;

     rms_reduce4 u_reduce (
        .clk(clk), .rst_n(rst_n && !clear),
        .cfg_valid(!clear && (state_q == ST_REDUCE_CFG)),
        .cfg_ready(reduce_cfg_ready), .cfg_rows(rows_q),
        .cfg_lane_mask(current_lane_mask), .cfg_epsilon(epsilon_q),
        .abort_run(child_abort), .busy(reduce_busy),
        .src_req_valid(reduce_src_req_valid),
        .src_req_ready(reduce_src_req_ready),
        .src_req_addr(reduce_src_req_addr),
        .src_rsp_valid(reduce_src_rsp_valid),
        .src_rsp_ready(reduce_src_rsp_ready),
        .src_rsp_data(reduce_src_rsp_data),
        .src_rsp_error(reduce_src_rsp_error),
        .result_valid(reduce_result_valid),
        .result_ready(!clear && (state_q == ST_REDUCE_WAIT)),
        .result_error(reduce_result_error),
        .result_status(reduce_result_status),
        .result_inv_rms(reduce_result_inv)
    );

    wire apply_cfg_ready;
    wire apply_busy;
    wire apply_in_ready;
    wire apply_out_valid;
    wire apply_out_ready;
    wire [127:0] apply_out_data;
    wire apply_out_last;

     norm_apply4 u_apply (
        .clk(clk), .rst_n(rst_n && !clear),
        .cfg_valid(!clear && (state_q == ST_APPLY_CFG)),
        .cfg_ready(apply_cfg_ready), .cfg_rows(rows_q),
        .cfg_lane_mask(current_lane_mask), .cfg_inv_rms(inv_rms_q),
        .abort_run(child_abort), .busy(apply_busy),
        .in_valid(norm_stage_valid_q), .in_ready(apply_in_ready),
        .in_data(norm_stage_data_q), .in_gamma(norm_stage_gamma_q),
        .out_valid(apply_out_valid), .out_ready(apply_out_ready),
        .out_data(apply_out_data), .out_last(apply_out_last)
    );
    wire apply_in_fire = norm_stage_valid_q && apply_in_ready;

    assign leaf_q8_cfg_valid = rst_n && !clear && !abort_run &&
                               (state_q == ST_Q8_CFG);
    assign leaf_q8_cfg_rows = {2'd0, rows_q};
    assign leaf_q8_cfg_lane_mask = current_lane_mask;
    wire leaf_q8_cfg_fire = leaf_q8_cfg_valid && leaf_q8_cfg_ready;
    assign leaf_q8_in_valid = !clear && (state_q == ST_STREAM) &&
                              apply_out_valid;
    assign leaf_q8_in_data = apply_out_data;
    assign apply_out_ready = !clear && (state_q == ST_STREAM) &&
                             leaf_q8_in_ready;
    assign leaf_q8_out_ready = !clear && (state_q == ST_STREAM) &&
                               q8_wr_ready;
    wire leaf_q8_out_fire = leaf_q8_out_valid && leaf_q8_out_ready;

    assign q8_wr_valid = !clear && (state_q == ST_STREAM) &&
                         leaf_q8_out_valid;
    assign q8_wr_wave = wave_q;
    assign q8_wr_addr = leaf_q8_out_block;
    assign q8_wr_lane_mask = current_lane_mask;
    assign q8_wr_data = leaf_q8_out_data;

    wire reduce_owns_r = (state_q == ST_REDUCE_CFG) ||
                         (state_q == ST_REDUCE_WAIT);
    wire reduce_drains_r = reduce_owns_r ||
                           ((state_q == ST_ABORT) && reduce_busy);
    wire norm_stage_capacity = !norm_stage_valid_q || apply_in_ready;
    wire gamma_to_norm_fire = gamma_read_valid_q && norm_stage_capacity;
    wire norm_rsp_capacity = !gamma_read_valid_q || gamma_to_norm_fire;
    wire stream_read_want = !clear && !abort_run &&
                            (state_q == ST_STREAM) &&
                            (read_issue_count_q < {1'b0, rows_q}) &&
                            norm_rsp_capacity;

    assign r_rd_req_valid = !clear &&
                            (reduce_owns_r ? reduce_src_req_valid :
                                             stream_read_want);
    assign r_rd_req_wave = wave_q;
    assign r_rd_req_addr = reduce_owns_r ? reduce_src_req_addr :
                           read_issue_count_q[11:0];
    assign reduce_src_req_ready = !clear && reduce_owns_r && r_rd_req_ready;
    assign reduce_src_rsp_valid = !clear && reduce_drains_r && r_rd_rsp_valid;
    assign reduce_src_rsp_data = r_rd_rsp_data;
    assign reduce_src_rsp_error = r_rd_rsp_error;
    assign r_rd_rsp_ready = !clear && (state_q == ST_ABORT ?
                            (reduce_busy ? reduce_src_rsp_ready : 1'b1) :
                            reduce_owns_r ? reduce_src_rsp_ready :
                            state_q == ST_STREAM ? norm_rsp_capacity : 1'b0);

    wire r_req_fire = r_rd_req_valid && r_rd_req_ready;
    wire r_rsp_fire = r_rd_rsp_valid && r_rd_rsp_ready;
    wire stream_req_fire = stream_read_want && r_rd_req_ready;
    wire stream_rsp_fire = (state_q == ST_STREAM) && r_rsp_fire;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q <= ST_IDLE;
            op_q <= 4'd0;
            token_mask_q <= 8'd0;
            rows_q <= 13'd0;
            gamma_addr_q <= 64'd0;
            epsilon_q <= 32'd0;
            cycle_q <= 32'd0;
            done_error_q <= 1'b0;
            status_q <= 16'd0;
            wave_q <= 1'b0;
            inv_rms_q <= 128'd0;
            gamma_count_q <= 11'd0;
            gamma_active_q <= 1'b0;
            gamma_stream_error_q <= 1'b0;
            read_issue_count_q <= 14'd0;
            read_rsp_count_q <= 14'd0;
            r_outstanding_q <= 15'd0;
            gamma_read_valid_q <= 1'b0;
            gamma_read_data_q <= 128'd0;
            gamma_read_word_q <= 128'd0;
            gamma_read_select_q <= 2'd0;
            norm_stage_valid_q <= 1'b0;
            norm_stage_data_q <= 128'd0;
            norm_stage_gamma_q <= 128'd0;
        end else begin
            if (state_q != ST_IDLE && state_q != ST_DONE)
                cycle_q <= cycle_q + 1'b1;

            case ({r_req_fire, r_rsp_fire})
                2'b10: r_outstanding_q <= r_outstanding_q + 1'b1;
                2'b01: r_outstanding_q <= r_outstanding_q - 1'b1;
                default: begin end
            endcase

            if (abort_run && state_q != ST_IDLE) begin
                state_q <= ST_ABORT;
                gamma_read_valid_q <= 1'b0;
                norm_stage_valid_q <= 1'b0;
                done_error_q <= 1'b0;
                status_q <= 16'd0;
                if (gamma_rsp_fire && gamma_rsp_last)
                    gamma_active_q <= 1'b0;
            end else begin
                case (state_q)
                    ST_IDLE: if (cmd_fire) begin
                        op_q <= cmd_op;
                        token_mask_q <= cmd_token_mask;
                        rows_q <= {cmd_hidden_blocks, 5'd0};
                        gamma_addr_q <= cmd_gamma_addr;
                        epsilon_q <= cmd_epsilon;
                        cycle_q <= 32'd0;
                        done_error_q <= !cmd_ok;
                        status_q <= cmd_ok ? 16'd0 :
                                    `VECTOR_STATUS_BAD_CMD;
                        wave_q <= |cmd_token_mask[3:0] ? 1'b0 : 1'b1;
                        state_q <= cmd_ok ? ST_RESET : ST_DONE;
                    end

                    ST_RESET: state_q <= ST_GAMMA_REQ;

                    ST_GAMMA_REQ: if (gamma_req_fire) begin
                        gamma_count_q <= 11'd0;
                        gamma_active_q <= 1'b1;
                        gamma_stream_error_q <= 1'b0;
                        state_q <= ST_GAMMA_LOAD;
                    end

                    ST_GAMMA_LOAD: if (gamma_rsp_fire) begin
                        if (gamma_count_q < gamma_words)
                            gamma_mem[gamma_count_q[9:0]] <= gamma_rsp_data;
                        if (gamma_rsp_error || !gamma_finite(gamma_rsp_data) ||
                            (gamma_rsp_last != gamma_expected_last)) begin
                            gamma_stream_error_q <= 1'b1;
                            status_q <= status_q |
                                        `VECTOR_STATUS_GAMMA_STREAM;
                        end
                        if (gamma_rsp_last) begin
                            gamma_active_q <= 1'b0;
                            if (gamma_stream_error_q || gamma_rsp_error ||
                                !gamma_finite(gamma_rsp_data) ||
                                !gamma_expected_last) begin
                                done_error_q <= 1'b1;
                                state_q <= ST_DONE;
                            end else begin
                                state_q <= ST_REDUCE_CFG;
                            end
                        end else if (gamma_count_q < gamma_words) begin
                            gamma_count_q <= gamma_count_q + 1'b1;
                        end
                    end

                    ST_REDUCE_CFG: if (reduce_cfg_ready)
                        state_q <= ST_REDUCE_WAIT;

                    ST_REDUCE_WAIT: if (reduce_result_valid) begin
                        if (reduce_result_error) begin
                            done_error_q <= 1'b1;
                            status_q <= status_q | `VECTOR_STATUS_REDUCE |
                                        {8'd0, reduce_result_status};
                            state_q <= ST_DONE;
                        end else begin
                            inv_rms_q <= reduce_result_inv;
                            state_q <= ST_Q8_CFG;
                        end
                    end

                    ST_Q8_CFG: if (leaf_q8_cfg_fire)
                        state_q <= ST_APPLY_CFG;

                    ST_APPLY_CFG: if (apply_cfg_ready) begin
                        read_issue_count_q <= 14'd0;
                        read_rsp_count_q <= 14'd0;
                        gamma_read_valid_q <= 1'b0;
                        norm_stage_valid_q <= 1'b0;
                        state_q <= ST_STREAM;
                    end

                    ST_STREAM: begin
                        if (stream_req_fire)
                            read_issue_count_q <= read_issue_count_q + 1'b1;
                        if (gamma_to_norm_fire) begin
                            norm_stage_valid_q <= 1'b1;
                            norm_stage_data_q <= gamma_read_data_q;
                            norm_stage_gamma_q <= {4{gamma_scalar(
                                gamma_read_word_q,
                                gamma_read_select_q)}};
                        end else if (apply_in_fire) begin
                            norm_stage_valid_q <= 1'b0;
                        end
                        if (stream_rsp_fire) begin
                            read_rsp_count_q <= read_rsp_count_q + 1'b1;
                            gamma_read_valid_q <= 1'b1;
                            gamma_read_data_q <= r_rd_rsp_data;
                            gamma_read_word_q <=
                                gamma_mem[read_rsp_count_q[11:2]];
                            gamma_read_select_q <= read_rsp_count_q[1:0];
                            if (r_rd_rsp_error) begin
                                status_q <= status_q |
                                            `VECTOR_STATUS_SOURCE;
                                done_error_q <= 1'b1;
                            end
                        end else if (gamma_to_norm_fire) begin
                            gamma_read_valid_q <= 1'b0;
                        end
                        if (leaf_q8_out_fire &&
                            (leaf_q8_out_status != 8'd0)) begin
                            status_q <= status_q | `VECTOR_STATUS_Q8 |
                                        {8'd0, leaf_q8_out_status};
                            done_error_q <= 1'b1;
                        end
                        if (leaf_q8_out_fire && leaf_q8_out_last) begin
                            if (has_next_wave) begin
                                wave_q <= 1'b1;
                                state_q <= ST_REDUCE_CFG;
                            end else begin
                                state_q <= ST_DONE;
                            end
                        end
                    end

                    ST_DONE: if (done_valid && done_ready) begin
                        state_q <= ST_IDLE;
                        done_error_q <= 1'b0;
                        status_q <= 16'd0;
                    end

                    ST_ABORT: begin
                        if (gamma_rsp_fire && gamma_rsp_last)
                            gamma_active_q <= 1'b0;
                        if (!gamma_active_q && !gamma_rsp_valid &&
                            (r_outstanding_q == 15'd0) &&
                            !reduce_busy && !apply_busy && !leaf_q8_busy)
                            state_q <= ST_IDLE;
                    end

                    default: begin
                        done_error_q <= 1'b1;
                        status_q <= status_q | `VECTOR_STATUS_INTERNAL;
                        state_q <= ST_DONE;
                    end
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && done_valid && !done_error &&
            (status_q != 16'd0))
            $fatal(1, " vector_service status/error mismatch");
        if (rst_n && !clear && leaf_q8_out_valid &&
            (state_q != ST_STREAM))
            $fatal(1, " vector_service unowned Q8 output");
        if (rst_n && !clear && r_rsp_fire &&
            (r_outstanding_q == 15'd0) &&
            !r_req_fire)
            $fatal(1, " vector_service unexpected R response");
    end
`endif
endmodule

`default_nettype wire
