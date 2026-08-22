// Scratch-backed P3d residual addition boundary.
//
// DOWN arrives in the native GEMM result order
// [16-row block][token][row pair]. Each pair rereads its resident R group and
// retains only the selected 64-bit bank word; one exact binary32 RNE adder is
// time-multiplexed over the two lanes. Each completed pair is committed through
// the direct R write port before the same word is emitted in unchanged order.
//
// A successful done pulse is the R residency seal point: every word for the
// accepted (cfg_rows, cfg_tokens) shape has committed and the final output beat
// has been accepted. The integrating controller may mark that R shape valid on
// done && !error. Abort or error leaves R tentative, because prior direct writes
// are intentionally not transactional; global invalidation remains top-level
// scratch-ownership policy.

`default_nettype none

module section_residual_add #(
    parameter [13:0] MIN_ROWS = 14'd128,
    parameter [13:0] MAX_ROWS = 14'd4096
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          abort_run,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 bad cfg, bit 1 frame, bits 2/3 scratch read/write,
    // bits 5:4 exact-adder raw status, bit 6 lifecycle/internal.
    output reg  [6:0]    status,

    // Directly compatible with the DOWN gemm_kernel result stream.
    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    output wire          s_axis_tready_core,
    input  wire          s_axis_tlast,

    // Untagged shared-scratch group read. The requested role is always R.
    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [1:0]    rd_req_role,
    output wire [2:0]    rd_req_token,
    output wire [10:0]   rd_req_group,
    input  wire          rd_rsp_valid,
    output wire          rd_rsp_ready,
    input  wire [255:0]  rd_rsp_data,
    input  wire          rd_rsp_error,

    // Direct R write using section_f32_scratch's physical bank/address map.
    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    input  wire          r_wr_error,
    output wire [1:0]    r_wr_bank,
    output wire [13:0]   r_wr_address,
    output wire [63:0]   r_wr_data,

    // Successfully committed sums in unchanged native DOWN order.
    output wire [63:0]   m_axis_tdata,
    output wire [7:0]    m_axis_tkeep,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast
);
    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_REQ         = 4'd1;
    localparam [3:0] ST_WAIT_RSP    = 4'd2;
    localparam [3:0] ST_INPUT       = 4'd3;
    localparam [3:0] ST_ADD_LO_REQ  = 4'd4;
    localparam [3:0] ST_ADD_LO_WAIT = 4'd5;
    localparam [3:0] ST_ADD_HI_REQ  = 4'd6;
    localparam [3:0] ST_ADD_HI_WAIT = 4'd7;
    localparam [3:0] ST_WRITE       = 4'd8;
    localparam [3:0] ST_OUTPUT      = 4'd9;
    localparam [3:0] ST_CLEANUP     = 4'd10;

    localparam [6:0] STATUS_BAD_CFG       = 7'b000_0001;
    localparam [6:0] STATUS_FRAME         = 7'b000_0010;
    localparam [6:0] STATUS_SCRATCH_READ  = 7'b000_0100;
    localparam [6:0] STATUS_SCRATCH_WRITE = 7'b000_1000;
    localparam [6:0] STATUS_INTERNAL      = 7'b100_0000;

    reg [3:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0]  run_tokens_q;
    reg [8:0]  run_rowblocks_q;
    reg        run_half_final_q;
    reg [8:0]  rowblock_q;
    reg [1:0]  token_q;
    reg [2:0]  pair_q;

    reg         read_owned_q;
    reg [63:0]  down_word_q;
    reg [63:0]  residual_word_q;
    reg [31:0]  sum_lo_q;
    reg [31:0]  sum_hi_q;

    reg         report_error_q;
    reg [6:0]   latched_status_q;

    wire wrapper_idle = state_q == ST_IDLE;
    assign busy = !wrapper_idle;

    wire cfg_rows_power_two = (cfg_rows != 14'd0) &&
                              ((cfg_rows & (cfg_rows - 1'b1)) == 14'd0);
    wire cfg_shape_ok = (cfg_rows >= MIN_ROWS) &&
                        (cfg_rows <= MAX_ROWS) &&
                        (cfg_rows[2:0] == 3'b000) &&
                        cfg_rows_power_two &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4);

    wire add_busy;
    wire add_s_ready;
    wire add_result_valid;
    wire [31:0] add_result_data;
    wire [1:0] add_result_status;

    // Idle orphan responses are drained before another run can acquire the
    // untagged port. Likewise, the exact adder must have completed cleanup.
    assign cfg_ready = rst_n && !abort_run && wrapper_idle &&
                       !read_owned_q && !rd_rsp_valid && !add_busy;
    wire cfg_fire = cfg_valid && cfg_ready;

    wire final_rowblock = rowblock_q + 1'b1 == run_rowblocks_q;
    wire [2:0] active_last_pair =
        (final_rowblock && run_half_final_q) ? 3'd3 : 3'd7;
    wire final_pair = final_rowblock &&
                      ({1'b0, token_q} + 1'b1 == run_tokens_q) &&
                      (pair_q == active_last_pair);
    wire expected_last = final_pair;
    wire input_frame_bad = (s_axis_tkeep != 8'hff) ||
                           (s_axis_tlast != expected_last);

    // Raw orphan presence is independent of any downstream ready signal. It
    // fail-closes all tentative traffic in the detection cycle without a
    // combinational ready/valid loop.
    wire orphan_rsp_present = rd_rsp_valid && !read_owned_q;
    wire unexpected_rsp = orphan_rsp_present && !wrapper_idle;
    wire traffic_enable = rst_n && !abort_run && !unexpected_rsp;

    assign s_axis_tready_core = rst_n && !unexpected_rsp &&
                                (state_q == ST_INPUT);
    assign s_axis_tready = s_axis_tready_core && !abort_run;
    wire input_fire = s_axis_tvalid && s_axis_tready;

    wire [10:0] current_group =
        {1'b0, rowblock_q, 1'b0} + {{10{1'b0}}, pair_q[2]};
    assign rd_req_valid = traffic_enable && (state_q == ST_REQ) &&
                          !read_owned_q && !rd_rsp_valid;
    assign rd_req_role = 2'd0;
    assign rd_req_token = {1'b0, token_q};
    assign rd_req_group = current_group;
    wire rd_req_fire = rd_req_valid && rd_req_ready;

    // Owned responses are consumed in WAIT_RSP or cleanup. Any unowned response
    // is drained as an orphan; while busy it is also a lifecycle fault.
    assign rd_rsp_ready = rst_n &&
        (read_owned_q ?
            ((state_q == ST_WAIT_RSP) || (state_q == ST_CLEANUP) || abort_run) :
            rd_rsp_valid);
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;

    function automatic [63:0] select_bank_word(
        input [255:0] group_data,
        input [1:0] bank
    );
        begin
            case (bank)
                2'd0: select_bank_word = group_data[63:0];
                2'd1: select_bank_word = group_data[127:64];
                2'd2: select_bank_word = group_data[191:128];
                default: select_bank_word = group_data[255:192];
            endcase
        end
    endfunction

    wire add_low_phase = (state_q == ST_ADD_LO_REQ) ||
                         (state_q == ST_ADD_LO_WAIT);
    wire add_s_valid = traffic_enable &&
                       ((state_q == ST_ADD_LO_REQ) ||
                        (state_q == ST_ADD_HI_REQ));
    wire [31:0] add_s_a = add_low_phase ? residual_word_q[31:0] :
                                          residual_word_q[63:32];
    wire [31:0] add_s_b = add_low_phase ? down_word_q[31:0] :
                                          down_word_q[63:32];
    wire add_s_fire = add_s_valid && add_s_ready;
    wire add_result_ready = traffic_enable &&
                            ((state_q == ST_ADD_LO_WAIT) ||
                             (state_q == ST_ADD_HI_WAIT));
    wire add_result_fire = add_result_valid && add_result_ready;

    section_residual_add_rne u_add (
        .clk(clk),
        .rst_n(rst_n),
        .abort_run(abort_run || (state_q == ST_CLEANUP)),
        .busy(add_busy),
        .s_valid(add_s_valid),
        .s_ready(add_s_ready),
        .s_a(add_s_a),
        .s_b(add_s_b),
        .result_valid(add_result_valid),
        .result_ready(add_result_ready),
        .result_data(add_result_data),
        .result_status(add_result_status)
    );

    assign r_wr_valid = traffic_enable && (state_q == ST_WRITE);
    assign r_wr_bank = pair_q[1:0];
    assign r_wr_address = {2'b00, token_q, 9'b0} +
                          {3'b000, current_group};
    assign r_wr_data = {sum_hi_q, sum_lo_q};
    wire r_wr_fire = r_wr_valid && r_wr_ready;

    assign m_axis_tdata = {sum_hi_q, sum_lo_q};
    assign m_axis_tkeep = 8'hff;
    assign m_axis_tvalid = traffic_enable && (state_q == ST_OUTPUT);
    assign m_axis_tlast = final_pair;
    wire output_fire = m_axis_tvalid && m_axis_tready;

    wire cleanup_complete = !read_owned_q && !rd_rsp_valid && !add_busy;

    always @(posedge clk) begin
        if (!rst_n) begin
            read_owned_q <= 1'b0;
        end else if (rd_req_fire) begin
            read_owned_q <= 1'b1;
        end else if (rd_rsp_fire) begin
            read_owned_q <= 1'b0;
        end
    end

    task automatic fail_run(input [6:0] failure_status);
        begin
            report_error_q <= 1'b1;
            latched_status_q <= (failure_status != 7'd0) ?
                                failure_status : STATUS_INTERNAL;
            state_q <= ST_CLEANUP;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_rowblocks_q <= 9'd0;
            run_half_final_q <= 1'b0;
            rowblock_q <= 9'd0;
            token_q <= 2'd0;
            pair_q <= 3'd0;
            down_word_q <= 64'd0;
            residual_word_q <= 64'd0;
            sum_lo_q <= 32'd0;
            sum_hi_q <= 32'd0;
            report_error_q <= 1'b0;
            latched_status_q <= 7'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
        end else begin
            done <= 1'b0;

            if (abort_run) begin
                // Abort cancels the tentative R epoch. Drain a read already
                // accepted by shared scratch before exposing cfg_ready again.
                report_error_q <= 1'b0;
                latched_status_q <= 7'd0;
                error <= 1'b0;
                status <= 7'd0;
                if (wrapper_idle && !read_owned_q && !rd_rsp_valid)
                    state_q <= ST_IDLE;
                else
                    state_q <= ST_CLEANUP;
            end else if (unexpected_rsp) begin
                fail_run(STATUS_INTERNAL);
            end else begin
                case (state_q)
                    ST_IDLE: if (cfg_fire) begin
                        error <= !cfg_shape_ok;
                        status <= cfg_shape_ok ? 7'd0 : STATUS_BAD_CFG;
                        done <= !cfg_shape_ok;
                        report_error_q <= 1'b0;
                        latched_status_q <= 7'd0;
                        rowblock_q <= 9'd0;
                        token_q <= 2'd0;
                        pair_q <= 3'd0;
                        sum_lo_q <= 32'd0;
                        sum_hi_q <= 32'd0;
                        if (cfg_shape_ok) begin
                            run_rows_q <= cfg_rows;
                            run_tokens_q <= cfg_tokens;
                            run_rowblocks_q <= cfg_rows[12:4] +
                                               {{8{1'b0}}, cfg_rows[3]};
                            run_half_final_q <= cfg_rows[3];
                            state_q <= ST_REQ;
                        end
                    end

                    ST_REQ: if (rd_req_fire)
                        state_q <= ST_WAIT_RSP;

                    ST_WAIT_RSP: if (rd_rsp_fire) begin
                        if (rd_rsp_error) begin
                            fail_run(STATUS_SCRATCH_READ);
                        end else begin
                            residual_word_q <= select_bank_word(
                                rd_rsp_data, pair_q[1:0]
                            );
                            state_q <= ST_INPUT;
                        end
                    end

                    ST_INPUT: if (input_fire) begin
                        if (input_frame_bad) begin
                            fail_run(STATUS_FRAME);
                        end else begin
                            down_word_q <= s_axis_tdata;
                            state_q <= ST_ADD_LO_REQ;
                        end
                    end

                    ST_ADD_LO_REQ: if (add_s_fire)
                        state_q <= ST_ADD_LO_WAIT;

                    ST_ADD_LO_WAIT: if (add_result_fire) begin
                        if (add_result_status != 2'd0) begin
                            fail_run({1'b0, add_result_status, 4'd0});
                        end else begin
                            sum_lo_q <= add_result_data;
                            state_q <= ST_ADD_HI_REQ;
                        end
                    end

                    ST_ADD_HI_REQ: if (add_s_fire)
                        state_q <= ST_ADD_HI_WAIT;

                    ST_ADD_HI_WAIT: if (add_result_fire) begin
                        if (add_result_status != 2'd0) begin
                            fail_run({1'b0, add_result_status, 4'd0});
                        end else begin
                            sum_hi_q <= add_result_data;
                            state_q <= ST_WRITE;
                        end
                    end

                    ST_WRITE: if (r_wr_fire) begin
                        if (r_wr_error)
                            fail_run(STATUS_SCRATCH_WRITE);
                        else
                            state_q <= ST_OUTPUT;
                    end

                    ST_OUTPUT: if (output_fire) begin
                        if (final_pair) begin
                            // This pulse seals exactly run_rows_q x run_tokens_q
                            // resident R values for the integrating controller.
                            state_q <= ST_IDLE;
                            done <= 1'b1;
                            error <= 1'b0;
                            status <= 7'd0;
                        end else if (pair_q == active_last_pair) begin
                            pair_q <= 3'd0;
                            if ({1'b0, token_q} + 1'b1 == run_tokens_q) begin
                                token_q <= 2'd0;
                                rowblock_q <= rowblock_q + 1'b1;
                            end else begin
                                token_q <= token_q + 1'b1;
                            end
                            state_q <= ST_REQ;
                        end else begin
                            pair_q <= pair_q + 1'b1;
                            state_q <= ST_REQ;
                        end
                    end

                    ST_CLEANUP: if (cleanup_complete) begin
                        state_q <= ST_IDLE;
                        if (report_error_q) begin
                            done <= 1'b1;
                            error <= 1'b1;
                            status <= latched_status_q;
                        end else begin
                            error <= 1'b0;
                            status <= 7'd0;
                        end
                        report_error_q <= 1'b0;
                        latched_status_q <= 7'd0;
                    end

                    default: fail_run(STATUS_INTERNAL);
                endcase
            end
        end
    end

    // Retain the latched shape through completion for formal/lint visibility.
    wire _unused_run_rows = &{1'b0, run_rows_q};

`ifdef FORMAL
    always @* begin
        assert(s_axis_tready == (s_axis_tready_core && !abort_run));
        if (abort_run)
            assert(!s_axis_tready && !input_fire);
    end
`endif
endmodule

`default_nettype wire
