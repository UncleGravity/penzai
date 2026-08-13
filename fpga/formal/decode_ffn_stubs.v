`default_nettype none

module q8_ingress (
    input wire clk, input wire rst_n, input wire start, input wire abort,
    input wire raw_mode, input wire internal_mode,
    input wire [15:0] num_q1_blocks, input wire [15:0] num_cols,
    input wire [63:0] s_axis_tdata, input wire s_axis_tvalid,
    output wire s_axis_tready, input wire s_axis_tlast,
    input wire [31:0] internal_data, input wire internal_last,
    input wire [1:0] internal_status, input wire internal_valid,
    output wire internal_ready, output wire internal_record_done,
    output wire [63:0] m_axis_tdata, output wire m_axis_tvalid,
    input wire m_axis_tready, output wire activation_abort,
    output wire [5:0] quantizer_status
);
    reg internal_run;
    reg [5:0] scalar_count;
    reg [2:0] emit_count;
    reg [6:0] quant_wait;
    reg emit_active;
    reg error_q;

    assign s_axis_tready = !abort && raw_mode && !internal_run && !error_q;
    assign internal_ready = !abort && internal_run && (quant_wait == 0) &&
                            !emit_active && !error_q;
    assign m_axis_tdata = 64'd0;
    assign m_axis_tvalid = !abort && internal_run && emit_active && !error_q;
    assign internal_record_done = m_axis_tvalid && m_axis_tready &&
                                  (emit_count == 3'd4);
    assign activation_abort = !abort && error_q;
    assign quantizer_status = (!abort && error_q) ? 6'b000100 : 6'd0;

    always @(posedge clk) begin
        if (!rst_n) begin
            internal_run <= 1'b0;
            scalar_count <= 6'd0;
            emit_count <= 3'd0;
            quant_wait <= 7'd0;
            emit_active <= 1'b0;
            error_q <= 1'b0;
        end else if (abort) begin
            internal_run <= 1'b0;
            scalar_count <= 6'd0;
            emit_count <= 3'd0;
            quant_wait <= 7'd0;
            emit_active <= 1'b0;
            error_q <= 1'b0;
        end else if (start) begin
            internal_run <= internal_mode;
            scalar_count <= 6'd0;
            emit_count <= 3'd0;
            quant_wait <= 7'd0;
            emit_active <= 1'b0;
            error_q <= 1'b0;
        end else begin
            if (internal_valid && internal_ready) begin
                if ((internal_status != 0) ||
                    (internal_last != (scalar_count == 6'd31))) begin
                    error_q <= 1'b1;
                end else if (scalar_count == 6'd31) begin
                    scalar_count <= 6'd0;
                    emit_count <= 3'd0;
                    // Model the canonical quantizer's nontrivial block latency so
                    // the controller proof exercises multiple queued blocks.
                    quant_wait <= 7'd64;
                end else begin
                    scalar_count <= scalar_count + 1'b1;
                end
            end
            if (quant_wait != 0) begin
                quant_wait <= quant_wait - 1'b1;
                if (quant_wait == 1)
                    emit_active <= 1'b1;
            end
            if (m_axis_tvalid && m_axis_tready) begin
                if (emit_count == 3'd4) begin
                    emit_count <= 3'd0;
                    emit_active <= 1'b0;
                end else begin
                    emit_count <= emit_count + 1'b1;
                end
            end
        end
    end

    wire _unused = &{1'b0, num_q1_blocks, num_cols, s_axis_tdata,
                     s_axis_tvalid, s_axis_tlast, internal_data};
endmodule

module section_swiglu (
    input wire clk, input wire rst_n, input wire abort,
    input wire in_valid, output wire in_ready,
    input wire [31:0] in_gate, input wire [31:0] in_up,
    input wire in_last, output wire out_valid, input wire out_ready,
    output wire [31:0] out_data, output wire out_last,
    output wire [1:0] out_status
);
    reg [6:0] queued;
    reg [5:0] input_lane;
    reg [5:0] output_lane;
    wire input_fire = in_valid && in_ready;
    wire output_fire = out_valid && out_ready;

    assign in_ready = rst_n && !abort && (queued < 7'd64);
    assign out_valid = rst_n && !abort && (queued != 0);
    assign out_data = in_gate;
    assign out_last = output_lane == 6'd31;
    assign out_status = 2'd0;

    always @(posedge clk) begin
        if (!rst_n || abort) begin
            queued <= 7'd0;
            input_lane <= 6'd0;
            output_lane <= 6'd0;
        end else begin
            case ({input_fire, output_fire})
                2'b10: queued <= queued + 1'b1;
                2'b01: queued <= queued - 1'b1;
                default: queued <= queued;
            endcase
            if (input_fire) begin
                assert(in_last == (input_lane == 6'd31));
                input_lane <= (input_lane == 6'd31) ?
                              6'd0 : input_lane + 1'b1;
            end
            if (output_fire)
                output_lane <= (output_lane == 6'd31) ?
                               6'd0 : output_lane + 1'b1;
        end
    end
    wire _unused = &{1'b0, clk, in_up};
endmodule

module section_f32_scratch (
    input wire clk, input wire rst_n,
    input wire wr_cfg_valid, output wire wr_cfg_ready,
    input wire [1:0] wr_cfg_role, input wire [13:0] wr_cfg_rows,
    input wire [2:0] wr_cfg_tokens, input wire wr_abort,
    output wire wr_busy, output reg wr_done, output reg wr_error,
    input wire [63:0] s_axis_tdata, input wire [7:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready,
    input wire s_axis_tlast, output wire wr_commit_valid,
    output wire [1:0] wr_commit_bank, output wire [13:0] wr_commit_address,
    input wire rd_req_valid, output wire rd_req_ready,
    input wire [1:0] rd_req_role, input wire [2:0] rd_req_token,
    input wire [10:0] rd_req_group, output wire rd_issue_valid,
    output wire [13:0] rd_issue_address, output wire rd_rsp_valid,
    input wire rd_rsp_ready, output wire [255:0] rd_rsp_data,
    output reg rd_rsp_error
);
    reg wr_busy_q;
    reg rd_valid_q;
    assign wr_cfg_ready = rst_n && !wr_busy_q && !wr_abort;
    assign wr_busy = wr_busy_q;
    assign s_axis_tready = wr_busy_q && !wr_abort;
    assign wr_commit_valid = 1'b0;
    assign wr_commit_bank = 2'd0;
    assign wr_commit_address = 14'd0;
    assign rd_req_ready = rst_n && (!rd_valid_q || rd_rsp_ready);
    assign rd_issue_valid = rd_req_valid && rd_req_ready;
    assign rd_issue_address = {3'd0, rd_req_group};
    assign rd_rsp_valid = rd_valid_q;
    assign rd_rsp_data = {8{32'h3f80_0000}};

    always @(posedge clk) begin
        if (!rst_n) begin
            wr_busy_q <= 1'b0;
            wr_done <= 1'b0;
            wr_error <= 1'b0;
            rd_valid_q <= 1'b0;
            rd_rsp_error <= 1'b0;
        end else begin
            wr_done <= 1'b0;
            if (wr_cfg_valid && wr_cfg_ready) begin
                wr_busy_q <= 1'b1;
                wr_error <= 1'b0;
            end else if (wr_abort && wr_busy_q) begin
                wr_busy_q <= 1'b0;
                wr_done <= 1'b1;
                wr_error <= 1'b1;
            end else if (wr_busy_q) begin
                wr_busy_q <= 1'b0;
                wr_done <= 1'b1;
            end
            if (rd_rsp_ready) rd_valid_q <= 1'b0;
            if (rd_req_valid && rd_req_ready) begin
                rd_valid_q <= 1'b1;
                rd_rsp_error <= 1'b0;
            end
        end
    end

    wire _unused = &{1'b0, wr_cfg_role, wr_cfg_rows, wr_cfg_tokens,
                     s_axis_tdata, s_axis_tkeep, s_axis_tvalid,
                     s_axis_tlast, rd_req_role, rd_req_token};
endmodule

module gemm_kernel #(
    parameter integer ROWS = 16,
    parameter integer COLS_MAX = 8,
    parameter integer MAX_SUB_INDEX = 64,
    parameter integer ACC_W = 104
) (
    input wire clk, input wire rst_n, input wire start_kernel,
    input wire [15:0] num_q1_blocks, input wire [15:0] num_rowblocks,
    input wire [31:0] num_rows, input wire [15:0] num_cols,
    input wire [1:0] weight_fmt, input wire [1:0] act_mode,
    input wire [31:0] act_epoch, input wire activation_abort,
    input wire signed [7:0] emin, output reg kernel_done,
    output reg activation_error, output reg activation_valid,
    output reg [31:0] loaded_act_epoch,
    output reg [15:0] loaded_act_q1_blocks,
    output reg [15:0] loaded_act_cols, output wire busy,
    input wire [ROWS*32-1:0] s_axis_tdata, input wire s_axis_tvalid,
    output wire s_axis_tready, input wire [63:0] s_axis_acts_tdata,
    input wire s_axis_acts_tvalid, output wire s_axis_acts_tready,
    output wire [63:0] m_axis_tdata, output wire m_axis_tvalid,
    input wire m_axis_tready, output wire m_axis_tlast,
    output wire [7:0] m_axis_tkeep, output wire [3:0] dbg_state
);
    reg busy_q;
    reg [8:0] wait_count;
    assign busy = busy_q;
    assign s_axis_tready = busy_q;
    assign s_axis_acts_tready = busy_q;
    assign m_axis_tdata = 64'd0;
    assign m_axis_tvalid = 1'b0;
    assign m_axis_tlast = 1'b0;
    assign m_axis_tkeep = 8'hff;
    assign dbg_state = busy_q ? 4'd1 : 4'd0;

    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            wait_count <= 9'd0;
            kernel_done <= 1'b0;
            activation_error <= 1'b0;
            activation_valid <= 1'b0;
            loaded_act_epoch <= 32'd0;
            loaded_act_q1_blocks <= 16'd0;
            loaded_act_cols <= 16'd0;
        end else begin
            kernel_done <= 1'b0;
            if (start_kernel && !busy_q) begin
                busy_q <= 1'b1;
                // Internal activation production owns completion ordering. Keep
                // this control stub busy beyond the proof horizon so it cannot
                // retire before the buffered Q8 source has delivered its records.
                wait_count <= (act_epoch == 32'd2) ? 9'd500 : 9'd3;
                activation_error <= 1'b0;
                if (act_mode == 2'd2) begin
                    activation_valid <= 1'b1;
                    loaded_act_epoch <= act_epoch;
                    loaded_act_q1_blocks <= num_q1_blocks;
                    loaded_act_cols <= num_cols;
                end
            end else if (activation_abort && busy_q) begin
                busy_q <= 1'b0;
                kernel_done <= 1'b1;
                activation_error <= 1'b1;
                activation_valid <= 1'b0;
            end else if (busy_q) begin
                if (wait_count == 0) begin
                    busy_q <= 1'b0;
                    kernel_done <= 1'b1;
                end else begin
                    wait_count <= wait_count - 1'b1;
                end
            end
        end
    end

    wire _unused = &{1'b0, num_rowblocks, num_rows, weight_fmt, emin,
                     s_axis_tdata, s_axis_tvalid, s_axis_acts_tdata,
                     s_axis_acts_tvalid, m_axis_tready, COLS_MAX,
                     MAX_SUB_INDEX, ACC_W};
endmodule

`default_nettype wire
