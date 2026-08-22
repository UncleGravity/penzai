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

    wire raw_run = raw_mode && !internal_run;

    assign s_axis_tready = !abort && raw_run && !error_q && m_axis_tready;
    assign internal_ready = !abort && internal_run && (quant_wait == 0) &&
                            !emit_active && !error_q;
    assign m_axis_tdata = 64'd0;
    assign m_axis_tvalid = !abort && !error_q &&
                           ((internal_run && emit_active) ||
                            (raw_run && s_axis_tvalid));
    assign internal_record_done = m_axis_tvalid && m_axis_tready &&
                                  (emit_count == 3'd4);
    assign activation_abort = !start && !abort && error_q;
    assign quantizer_status = (!start && !abort && error_q) ?
                              6'b000100 : 6'd0;

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
                    // Retain a nonzero quantizer boundary without dominating the
                    // bounded integration proof depth.
                    quant_wait <= 7'd2;
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
                     s_axis_tlast, internal_data};
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
    output wire s_axis_tready_core,
    input wire s_axis_tlast, output wire wr_commit_valid,
    output wire [1:0] wr_commit_bank, output wire [13:0] wr_commit_address,
    input wire r_wr_abort, input wire r_wr_valid, output wire r_wr_ready,
    input wire [1:0] r_wr_bank, input wire [13:0] r_wr_address,
    input wire [63:0] r_wr_data, output wire r_wr_error,
    input wire rd_req_valid, output wire rd_req_ready,
    output wire rd_admission_idle,
    output wire rd_quiescent,
    input wire [1:0] rd_req_role, input wire [2:0] rd_req_token,
    input wire [10:0] rd_req_group, output wire rd_issue_valid,
    output wire [13:0] rd_issue_address, output wire rd_rsp_valid,
    input wire rd_rsp_ready, output wire [255:0] rd_rsp_data,
    output reg rd_rsp_error
);
    reg wr_busy_q;
    reg rd_valid_q;
    reg rd_pending_q;
    reg [2:0] rd_delay_q;
    assign wr_cfg_ready = rst_n && !wr_busy_q && !wr_abort;
    assign wr_busy = wr_busy_q;
    assign s_axis_tready_core = rst_n && wr_busy_q;
    assign s_axis_tready = s_axis_tready_core && !wr_abort;
    assign wr_commit_valid = 1'b0;
    assign wr_commit_bank = 2'd0;
    assign wr_commit_address = 14'd0;
    assign r_wr_ready = rst_n && !r_wr_abort;
    assign r_wr_error = 1'b0;
    assign rd_req_ready = rst_n && !rd_pending_q &&
                          (!rd_valid_q || rd_rsp_ready);
    assign rd_admission_idle = rst_n && !rd_pending_q;
    assign rd_quiescent = rst_n && !rd_pending_q && !rd_valid_q;
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
            rd_pending_q <= 1'b0;
            rd_delay_q <= 3'd0;
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
                rd_pending_q <= 1'b1;
                rd_delay_q <= 3'd5;
                rd_rsp_error <= 1'b0;
            end else if (rd_pending_q) begin
                if (rd_delay_q == 1) begin
                    rd_pending_q <= 1'b0;
                    rd_valid_q <= 1'b1;
                end else begin
                    rd_delay_q <= rd_delay_q - 1'b1;
                end
            end
        end
    end

    wire _unused = &{1'b0, wr_cfg_role, wr_cfg_rows, wr_cfg_tokens,
                     s_axis_tdata, s_axis_tkeep, s_axis_tvalid,
                     s_axis_tlast, r_wr_abort, r_wr_valid, r_wr_bank, r_wr_address,
                     r_wr_data, rd_req_role, rd_req_token};
endmodule

// Control-oriented abstraction of the P3d weighted RMSNorm wrapper.  The
// integration proof keeps the real top-level beat accounting and scratch/Q8
// routing, while abstracting the numerical reduction and multiply pipelines.
module section_rmsnorm_scalar_pipeline (
    input wire clk, input wire rst_n, input wire abort_run,
    input wire gamma_cfg_valid, output wire gamma_cfg_ready,
    input wire [13:0] gamma_cfg_rows,
    input wire [63:0] gamma_tdata, input wire [7:0] gamma_tkeep,
    input wire gamma_tvalid, output wire gamma_tready,
    input wire gamma_tlast, output wire gamma_busy,
    output reg gamma_done, output reg gamma_error,
    output reg [3:0] gamma_status, output reg gamma_valid,
    input wire cfg_valid, output wire cfg_ready,
    input wire [13:0] cfg_rows, input wire [2:0] cfg_tokens,
    input wire [31:0] cfg_eps, input wire cfg_resident,
    output wire busy, output reg done, output reg error,
    output reg [22:0] status,
    input wire [63:0] s_axis_tdata, input wire [7:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready,
    input wire s_axis_tlast,
    output wire r_wr_valid, input wire r_wr_ready,
    input wire r_wr_error, output wire [1:0] r_wr_bank,
    output wire [13:0] r_wr_address, output wire [63:0] r_wr_data,
    output wire rd_req_valid, input wire rd_req_ready,
    output wire [1:0] rd_req_role, output wire [2:0] rd_req_token,
    output wire [10:0] rd_req_group, input wire rd_rsp_valid,
    output wire rd_rsp_ready, input wire [255:0] rd_rsp_data,
    input wire rd_rsp_error, output wire [31:0] scalar_data,
    output wire scalar_valid, input wire scalar_ready,
    output wire scalar_last, output wire [1:0] scalar_status
);
    localparam [2:0] ST_IDLE = 3'd0;
    localparam [2:0] ST_LOAD = 3'd1;
    localparam [2:0] ST_REQ  = 3'd2;
    localparam [2:0] ST_RSP  = 3'd3;
    localparam [2:0] ST_OUT  = 3'd4;
    localparam [2:0] ST_CLEANUP = 3'd5;

    reg gamma_busy_q;
    reg [13:0] gamma_words_q;
    reg [13:0] gamma_expected_q;
    reg [2:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg run_resident_q;
    reg [14:0] pair_q;
    reg [14:0] scalar_q;
    reg read_owned_q;

    wire [14:0] scalar_total = run_rows_q * run_tokens_q;
    wire [14:0] pair_total = scalar_total >> 1;
    wire [14:0] group_total = scalar_total >> 3;
    wire cfg_shape_ok = (cfg_rows >= 14'd128) &&
                        (cfg_rows <= 14'd4096) &&
                        ((cfg_rows & (cfg_rows - 1'b1)) == 0) &&
                        (cfg_tokens != 0) && (cfg_tokens <= 4) &&
                        !cfg_eps[31] && (cfg_eps[30:23] != 0) &&
                        (cfg_eps[30:23] != 8'hff) && gamma_valid;

    assign gamma_cfg_ready = rst_n && !abort_run && !gamma_busy_q &&
                             (state_q == ST_IDLE);
    assign gamma_tready = gamma_busy_q && !abort_run;
    assign gamma_busy = gamma_busy_q;
    assign cfg_ready = rst_n && !abort_run && !gamma_busy_q &&
                       (state_q == ST_IDLE) && !read_owned_q &&
                       !rd_rsp_valid;
    assign busy = state_q != ST_IDLE;

    assign s_axis_tready = (state_q == ST_LOAD) && !abort_run;
    assign r_wr_valid = s_axis_tvalid && s_axis_tready;
    assign r_wr_bank = pair_q[1:0];
    assign r_wr_address = pair_q[13:0];
    assign r_wr_data = s_axis_tdata;

    assign rd_req_valid = (state_q == ST_REQ) && !read_owned_q &&
                          !rd_rsp_valid && !abort_run;
    assign rd_req_role = 2'd0;
    assign rd_req_token = 3'd0;
    assign rd_req_group = scalar_q[13:3];
    assign rd_rsp_ready = rst_n &&
                          (read_owned_q || (rd_rsp_valid && !busy));
    assign scalar_data = 32'd0;
    assign scalar_valid = (state_q == ST_OUT) && !abort_run;
    assign scalar_last = scalar_q[4:0] == 5'd31;
    assign scalar_status = 2'd0;

    always @(posedge clk) begin
        if (!rst_n) begin
            gamma_busy_q <= 1'b0;
            gamma_words_q <= 14'd0;
            gamma_expected_q <= 14'd0;
            gamma_done <= 1'b0;
            gamma_error <= 1'b0;
            gamma_status <= 4'd0;
            gamma_valid <= 1'b0;
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_resident_q <= 1'b0;
            pair_q <= 15'd0;
            scalar_q <= 15'd0;
            read_owned_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 23'd0;
        end else begin
            gamma_done <= 1'b0;
            done <= 1'b0;

            if (abort_run) begin
                gamma_busy_q <= 1'b0;
                gamma_valid <= 1'b0;
                if (state_q != ST_IDLE)
                    state_q <= read_owned_q ? ST_CLEANUP : ST_IDLE;
                error <= 1'b0;
                status <= 23'd0;
            end else begin
                if (gamma_cfg_valid && gamma_cfg_ready) begin
                    gamma_busy_q <= 1'b1;
                    gamma_words_q <= 14'd0;
                    gamma_expected_q <= gamma_cfg_rows >> 1;
                    gamma_error <= 1'b0;
                    gamma_status <= 4'd0;
                    gamma_valid <= 1'b0;
                end
                if (gamma_tvalid && gamma_tready) begin
                    gamma_words_q <= gamma_words_q + 1'b1;
                    if ((gamma_tkeep != 8'hff) ||
                        (gamma_tlast !=
                         (gamma_words_q + 1'b1 == gamma_expected_q))) begin
                        gamma_error <= 1'b1;
                        gamma_status <= 4'b0010;
                    end
                    if (gamma_tlast ||
                        (gamma_words_q + 1'b1 == gamma_expected_q)) begin
                        gamma_busy_q <= 1'b0;
                        gamma_done <= 1'b1;
                        gamma_valid <= (gamma_tkeep == 8'hff) && gamma_tlast &&
                            (gamma_words_q + 1'b1 == gamma_expected_q);
                    end
                end

                if (cfg_valid && cfg_ready) begin
                    run_rows_q <= cfg_rows;
                    run_tokens_q <= cfg_tokens;
                    run_resident_q <= cfg_resident;
                    pair_q <= 15'd0;
                    scalar_q <= 15'd0;
                    error <= !cfg_shape_ok;
                    status <= cfg_shape_ok ? 23'd0 : 23'h40_0000;
                    state_q <= cfg_shape_ok ?
                               (cfg_resident ? ST_REQ : ST_LOAD) : ST_IDLE;
                    if (!cfg_shape_ok)
                        done <= 1'b1;
                end else if (state_q == ST_LOAD && s_axis_tvalid &&
                             s_axis_tready) begin
                    if (r_wr_error || (s_axis_tkeep != 8'hff) ||
                        (s_axis_tlast != (pair_q + 1'b1 == pair_total))) begin
                        error <= 1'b1;
                        status <= 23'h40_0000;
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else if (r_wr_ready) begin
                        pair_q <= pair_q + 1'b1;
                        if (pair_q + 1'b1 == pair_total) begin
                            scalar_q <= 15'd0;
                            state_q <= ST_REQ;
                        end
                    end
                end else if (state_q == ST_REQ && rd_req_valid &&
                             rd_req_ready) begin
                    read_owned_q <= 1'b1;
                    state_q <= ST_RSP;
                end else if (state_q == ST_RSP && rd_rsp_valid &&
                             rd_rsp_ready) begin
                    read_owned_q <= 1'b0;
                    if (rd_rsp_error) begin
                        error <= 1'b1;
                        status <= 23'h40_0000;
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else begin
                        state_q <= ST_OUT;
                    end
                end else if (state_q == ST_OUT && scalar_valid &&
                             scalar_ready) begin
                    scalar_q <= scalar_q + 1'b1;
                    if (scalar_q + 1'b1 == scalar_total) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end
                end else if (state_q == ST_CLEANUP && rd_rsp_valid &&
                             rd_rsp_ready) begin
                    read_owned_q <= 1'b0;
                    state_q <= ST_IDLE;
                end
            end
        end
    end

    wire _unused = &{1'b0, gamma_tdata, rd_rsp_data, group_total,
                     run_resident_q, r_wr_valid};
endmodule

// Control-oriented residual abstraction.  One representative read preserves
// retained scratch ownership; every result still observes write-before-output.
module section_residual_add (
    input wire clk, input wire rst_n, input wire abort_run,
    input wire cfg_valid, output wire cfg_ready,
    input wire [13:0] cfg_rows, input wire [2:0] cfg_tokens,
    output wire busy, output reg done, output reg error,
    output reg [6:0] status,
    input wire [63:0] s_axis_tdata, input wire [7:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready,
    output wire s_axis_tready_core,
    input wire s_axis_tlast,
    output wire rd_req_valid, input wire rd_req_ready,
    output wire [1:0] rd_req_role, output wire [2:0] rd_req_token,
    output wire [10:0] rd_req_group, input wire rd_rsp_valid,
    output wire rd_rsp_ready, input wire [255:0] rd_rsp_data,
    input wire rd_rsp_error, output wire r_wr_valid,
    input wire r_wr_ready, input wire r_wr_error,
    output wire [1:0] r_wr_bank, output wire [13:0] r_wr_address,
    output wire [63:0] r_wr_data, output wire [63:0] m_axis_tdata,
    output wire [7:0] m_axis_tkeep, output wire m_axis_tvalid,
    input wire m_axis_tready, output wire m_axis_tlast
);
    localparam [2:0] ST_IDLE = 3'd0;
    localparam [2:0] ST_REQ = 3'd1;
    localparam [2:0] ST_RSP = 3'd2;
    localparam [2:0] ST_INPUT = 3'd3;
    localparam [2:0] ST_WRITE = 3'd4;
    localparam [2:0] ST_OUTPUT = 3'd5;
    localparam [2:0] ST_CLEANUP = 3'd6;
    reg [2:0] state_q;
    reg [14:0] beat_q;
    reg [14:0] beat_total_q;
    reg read_owned_q;
    reg [63:0] word_q;
    reg last_q;

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE) &&
                       !read_owned_q && !rd_rsp_valid;
    assign busy = state_q != ST_IDLE;
    assign rd_req_valid = (state_q == ST_REQ) && !read_owned_q &&
                          !rd_rsp_valid && !abort_run;
    assign rd_req_role = 2'd0;
    assign rd_req_token = 3'd0;
    assign rd_req_group = beat_q[13:3];
    assign rd_rsp_ready = rst_n &&
                          (read_owned_q || (rd_rsp_valid && !busy));
    assign s_axis_tready_core = rst_n && (state_q == ST_INPUT);
    assign s_axis_tready = s_axis_tready_core && !abort_run;
    assign r_wr_valid = (state_q == ST_WRITE) && !abort_run;
    assign r_wr_bank = beat_q[1:0];
    assign r_wr_address = beat_q[13:0];
    assign r_wr_data = word_q;
    assign m_axis_tdata = word_q;
    assign m_axis_tkeep = 8'hff;
    assign m_axis_tvalid = (state_q == ST_OUTPUT) && !abort_run;
    assign m_axis_tlast = last_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            beat_q <= 15'd0;
            beat_total_q <= 15'd0;
            read_owned_q <= 1'b0;
            word_q <= 64'd0;
            last_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
        end else begin
            done <= 1'b0;
            if (abort_run) begin
                state_q <= read_owned_q ? ST_CLEANUP : ST_IDLE;
                error <= 1'b0;
                status <= 7'd0;
            end else if (cfg_valid && cfg_ready) begin
                beat_q <= 15'd0;
                beat_total_q <= (cfg_rows * cfg_tokens) >> 1;
                error <= 1'b0;
                status <= 7'd0;
                state_q <= ST_REQ;
            end else if (state_q == ST_REQ && rd_req_valid && rd_req_ready) begin
                read_owned_q <= 1'b1;
                state_q <= ST_RSP;
            end else if (state_q == ST_RSP && rd_rsp_valid && rd_rsp_ready) begin
                read_owned_q <= 1'b0;
                if (rd_rsp_error) begin
                    error <= 1'b1;
                    status <= 7'b000_0100;
                    done <= 1'b1;
                    state_q <= ST_IDLE;
                end else begin
                    state_q <= ST_INPUT;
                end
            end else if (state_q == ST_INPUT && s_axis_tvalid &&
                         s_axis_tready) begin
                word_q <= s_axis_tdata;
                last_q <= s_axis_tlast;
                if ((s_axis_tkeep != 8'hff) ||
                    (s_axis_tlast != (beat_q + 1'b1 == beat_total_q))) begin
                    error <= 1'b1;
                    status <= 7'b000_0010;
                    done <= 1'b1;
                    state_q <= ST_IDLE;
                end else begin
                    state_q <= ST_WRITE;
                end
            end else if (state_q == ST_WRITE && r_wr_valid && r_wr_ready) begin
                if (r_wr_error) begin
                    error <= 1'b1;
                    status <= 7'b000_1000;
                    done <= 1'b1;
                    state_q <= ST_IDLE;
                end else begin
                    state_q <= ST_OUTPUT;
                end
            end else if (state_q == ST_OUTPUT && m_axis_tvalid &&
                         m_axis_tready) begin
                beat_q <= beat_q + 1'b1;
                if (beat_q + 1'b1 == beat_total_q) begin
                    done <= 1'b1;
                    state_q <= ST_IDLE;
                end else begin
                    state_q <= ST_INPUT;
                end
            end else if (state_q == ST_CLEANUP && rd_rsp_valid &&
                         rd_rsp_ready) begin
                read_owned_q <= 1'b0;
                state_q <= ST_IDLE;
            end
        end
    end

    wire _unused = &{1'b0, rd_rsp_data};
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
    reg [1:0] mode_q;
    reg [8:0] wait_count;
    reg [15:0] result_beats_left_q;
    reg [15:0] act_beats_left_q;
    assign busy = busy_q;
    assign s_axis_tready = busy_q;
    assign s_axis_acts_tready = busy_q &&
                                ((mode_q == 2'd0) ||
                                 (mode_q == 2'd2)) &&
                                (act_beats_left_q != 0);
    assign m_axis_tdata = 64'd0;
    assign m_axis_tvalid = busy_q && (mode_q == 2'd1) &&
                           (result_beats_left_q != 0);
    assign m_axis_tlast = m_axis_tvalid && (result_beats_left_q == 1);
    assign m_axis_tkeep = 8'hff;
    assign dbg_state = busy_q ? 4'd1 : 4'd0;

    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            mode_q <= 2'd0;
            wait_count <= 9'd0;
            result_beats_left_q <= 16'd0;
            act_beats_left_q <= 16'd0;
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
                mode_q <= act_mode;
                wait_count <= 9'd3;
                result_beats_left_q <= (act_mode == 2'd1) ?
                    (num_rowblocks * num_cols * 16'd8) : 16'd0;
                act_beats_left_q <= ((act_mode == 2'd0) ||
                                     (act_mode == 2'd2)) ?
                    (num_q1_blocks * num_cols * 16'd20) : 16'd0;
                activation_error <= 1'b0;
                if (act_mode != 2'd1)
                    activation_valid <= 1'b0;
            end else if (activation_abort && busy_q) begin
                busy_q <= 1'b0;
                kernel_done <= 1'b1;
                activation_error <= 1'b1;
                activation_valid <= 1'b0;
                result_beats_left_q <= 16'd0;
                act_beats_left_q <= 16'd0;
            end else if (busy_q) begin
                if (m_axis_tvalid && m_axis_tready) begin
                    result_beats_left_q <= result_beats_left_q - 1'b1;
                    if (result_beats_left_q == 1) begin
                        busy_q <= 1'b0;
                        kernel_done <= 1'b1;
                    end
                end else if (s_axis_acts_tvalid && s_axis_acts_tready) begin
                    act_beats_left_q <= act_beats_left_q - 1'b1;
                    if (act_beats_left_q == 1) begin
                        activation_valid <= 1'b1;
                        loaded_act_epoch <= act_epoch;
                        loaded_act_q1_blocks <= num_q1_blocks;
                        loaded_act_cols <= num_cols;
                        if (mode_q == 2'd0) begin
                            mode_q <= 2'd1;
                            result_beats_left_q <=
                                num_rowblocks * num_cols * 16'd8;
                        end else begin
                            busy_q <= 1'b0;
                            kernel_done <= 1'b1;
                        end
                    end
                end
            end else if (activation_abort) begin
                // Match the real kernel when a registered abort follows a
                // final output beat that retired busy_q in the prior cycle.
                activation_error <= 1'b1;
                activation_valid <= 1'b0;
            end
        end
    end

    wire _unused = &{1'b0, num_rowblocks, num_rows, weight_fmt, emin,
                     s_axis_tdata, s_axis_tvalid, s_axis_acts_tdata,
                     s_axis_acts_tvalid, m_axis_tready, COLS_MAX,
                     MAX_SUB_INDEX, ACC_W};
endmodule

`default_nettype wire
