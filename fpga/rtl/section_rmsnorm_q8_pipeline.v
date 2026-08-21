// Complete scratch-backed weighted RMSNorm to native Q8_0 pipeline.
//
// Reduction loads the residual into R scratch and computes one inverse-RMS
// scalar per token. The weighted source consumes those scalars, rereads R, and
// publishes canonical Q8 records. This wrapper owns the untagged scratch read
// response across both phases and closes both child lifecycles as one command.

`default_nettype none

module section_rmsnorm_q8_pipeline #(
    parameter [13:0] MIN_ROWS = 14'd128,
    parameter [13:0] MAX_ROWS = 14'd4096
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          abort_run,

    input  wire          gamma_cfg_valid,
    output wire          gamma_cfg_ready,
    input  wire [13:0]   gamma_cfg_rows,
    input  wire [63:0]   gamma_tdata,
    input  wire [7:0]    gamma_tkeep,
    input  wire          gamma_tvalid,
    output wire          gamma_tready,
    input  wire          gamma_tlast,
    output wire          gamma_busy,
    output wire          gamma_done,
    output wire          gamma_error,
    output wire [3:0]    gamma_status,
    output wire          gamma_valid,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    input  wire [31:0]   cfg_eps,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bits 12:0 reduction raw status, bits 28:13 Q8-source raw status,
    // bit 29 wrapper lifecycle/protocol status.
    output reg  [29:0]   status,

    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    input  wire          r_wr_error,
    output wire [1:0]    r_wr_bank,
    output wire [13:0]   r_wr_address,
    output wire [63:0]   r_wr_data,

    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [1:0]    rd_req_role,
    output wire [2:0]    rd_req_token,
    output wire [10:0]   rd_req_group,
    input  wire          rd_rsp_valid,
    output wire          rd_rsp_ready,
    input  wire [255:0]  rd_rsp_data,
    input  wire          rd_rsp_error,

    output wire [63:0]   m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,
    output wire [1:0]    m_axis_token,
    output wire [8:0]    m_axis_block
`ifdef FORMAL
    , output wire [1:0]  formal_state
    , output wire        formal_cfg_fire
    , output wire        formal_reduce_busy
    , output wire        formal_reduce_done
    , output wire        formal_reduce_error
    , output wire [12:0] formal_reduce_status
    , output wire        formal_source_busy
    , output wire        formal_source_done
    , output wire        formal_source_error
    , output wire [15:0] formal_source_status
    , output wire        formal_reduce_result_fire
    , output wire [1:0]  formal_scratch_owner
    , output wire        formal_reduce_rd_req_fire
    , output wire        formal_source_rd_req_fire
    , output wire        formal_rd_rsp_fire
    , output wire        formal_run_window
    , output wire        formal_traffic_enable
    , output wire        formal_internal_fault
    , output wire        formal_reduce_rd_req_valid_raw
    , output wire        formal_source_rd_req_valid_raw
    , output wire        formal_cross_abort
    , output wire        formal_cleanup_abort_issued
    , output wire        formal_reduce_done_seen
    , output wire        formal_fault_latched
    , output wire [29:0] formal_latched_status
    , output wire        formal_output_fire
    , output wire        formal_final_output_fire
`endif
);
    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_RUN     = 2'd1;
    localparam [1:0] ST_CLEANUP = 2'd2;
    localparam [1:0] ST_START   = 2'd3;

    localparam [1:0] RD_NONE   = 2'd0;
    localparam [1:0] RD_REDUCE = 2'd1;
    localparam [1:0] RD_SOURCE = 2'd2;

    localparam [29:0] STATUS_INTERNAL = 30'h2000_0000;

    reg [1:0]  state_q;
    reg [1:0]  scratch_owner_q;
    reg [2:0]  run_tokens_q;
    reg [8:0]  last_block_q;
    reg        reduce_done_seen_q;
    reg        fault_latched_q;
    reg        report_error_q;
    reg        cross_abort_q;
    reg        cleanup_abort_issued_q;
    reg [29:0] latched_status_q;

    wire wrapper_idle = state_q == ST_IDLE;
    wire child_abort = abort_run || cross_abort_q;

    wire reduce_cfg_ready;
    wire reduce_busy;
    wire reduce_done;
    wire reduce_error;
    wire [12:0] reduce_status;
    wire reduce_s_axis_tready;
    wire reduce_r_wr_valid;
    wire reduce_r_wr_ready;
    wire [1:0] reduce_r_wr_bank;
    wire [13:0] reduce_r_wr_address;
    wire [63:0] reduce_r_wr_data;
    wire reduce_rd_req_valid;
    wire reduce_rd_req_ready;
    wire [2:0] reduce_rd_req_token;
    wire [10:0] reduce_rd_req_group;
    wire reduce_rd_rsp_valid;
    wire reduce_rd_rsp_ready;
    wire reduce_result_valid;
    wire reduce_result_ready;
    wire [1:0] reduce_result_token;
    wire [31:0] reduce_result_inv_rms;
    wire reduce_result_final;

    wire source_gamma_cfg_ready;
    wire source_gamma_tready;
    wire source_cfg_ready;
    wire source_busy;
    wire source_done;
    wire source_error;
    wire [15:0] source_status;
    wire source_inv_ready;
    wire source_rd_req_valid;
    wire source_rd_req_ready;
    wire [1:0] source_rd_req_role;
    wire [2:0] source_rd_req_token;
    wire [10:0] source_rd_req_group;
    wire source_rd_rsp_valid;
    wire source_rd_rsp_ready;
    wire [63:0] source_m_axis_tdata;
    wire source_m_axis_tvalid;
    wire source_m_axis_tready;
    wire source_m_axis_tlast;
    wire [1:0] source_m_axis_token;
    wire [8:0] source_m_axis_block;

    assign cfg_ready = rst_n && !abort_run && wrapper_idle &&
                       reduce_cfg_ready && source_cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;

    assign gamma_cfg_ready = wrapper_idle && source_gamma_cfg_ready;
    assign gamma_tready = wrapper_idle && source_gamma_tready;

    wire child_fault_now = reduce_error || source_error;
    wire run_window = rst_n && !abort_run && (state_q == ST_RUN) &&
                      !cross_abort_q;
    wire internal_fault_now;
    wire traffic_enable = run_window && !child_fault_now &&
                          !internal_fault_now;

    assign s_axis_tready = traffic_enable && reduce_s_axis_tready;
    assign reduce_r_wr_ready = traffic_enable && r_wr_ready;
    assign r_wr_valid = traffic_enable && reduce_r_wr_valid;
    assign r_wr_bank = reduce_r_wr_bank;
    assign r_wr_address = reduce_r_wr_address;
    assign r_wr_data = reduce_r_wr_data;

    assign reduce_result_ready = traffic_enable && source_inv_ready;
    wire reduce_result_fire = reduce_result_valid && reduce_result_ready;

    section_rmsnorm_reduce u_reduce (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_valid(cfg_fire),
        .cfg_ready(reduce_cfg_ready),
        .cfg_rows(cfg_rows),
        .cfg_tokens(cfg_tokens),
        .cfg_eps(cfg_eps),
        .abort_run(child_abort),
        .busy(reduce_busy),
        .done(reduce_done),
        .error(reduce_error),
        .status(reduce_status),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid && traffic_enable),
        .s_axis_tready(reduce_s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .r_wr_valid(reduce_r_wr_valid),
        .r_wr_ready(reduce_r_wr_ready),
        .r_wr_error(r_wr_error),
        .r_wr_bank(reduce_r_wr_bank),
        .r_wr_address(reduce_r_wr_address),
        .r_wr_data(reduce_r_wr_data),
        .rd_req_valid(reduce_rd_req_valid),
        .rd_req_ready(reduce_rd_req_ready),
        .rd_req_token(reduce_rd_req_token),
        .rd_req_group(reduce_rd_req_group),
        .rd_rsp_valid(reduce_rd_rsp_valid),
        .rd_rsp_ready(reduce_rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data),
        .rd_rsp_error(rd_rsp_error),
        .result_valid(reduce_result_valid),
        .result_ready(reduce_result_ready),
        .result_token(reduce_result_token),
        .result_inv_rms(reduce_result_inv_rms),
        .result_final(reduce_result_final)
    );

    section_rmsnorm_q8_source #(
        .MIN_ROWS(MIN_ROWS),
        .MAX_ROWS(MAX_ROWS)
    ) u_source (
        .clk(clk),
        .rst_n(rst_n),
        .abort_run(child_abort),
        .gamma_cfg_valid(gamma_cfg_valid && wrapper_idle),
        .gamma_cfg_ready(source_gamma_cfg_ready),
        .gamma_cfg_rows(gamma_cfg_rows),
        .gamma_tdata(gamma_tdata),
        .gamma_tkeep(gamma_tkeep),
        .gamma_tvalid(gamma_tvalid && wrapper_idle),
        .gamma_tready(source_gamma_tready),
        .gamma_tlast(gamma_tlast),
        .gamma_busy(gamma_busy),
        .gamma_done(gamma_done),
        .gamma_error(gamma_error),
        .gamma_status(gamma_status),
        .gamma_valid(gamma_valid),
        .cfg_valid(cfg_fire),
        .cfg_ready(source_cfg_ready),
        .cfg_rows(cfg_rows),
        .cfg_tokens(cfg_tokens),
        .busy(source_busy),
        .done(source_done),
        .error(source_error),
        .status(source_status),
        .inv_valid(reduce_result_valid && traffic_enable),
        .inv_ready(source_inv_ready),
        .inv_token(reduce_result_token),
        .inv_rms(reduce_result_inv_rms),
        .inv_final(reduce_result_final),
        .rd_req_valid(source_rd_req_valid),
        .rd_req_ready(source_rd_req_ready),
        .rd_req_role(source_rd_req_role),
        .rd_req_token(source_rd_req_token),
        .rd_req_group(source_rd_req_group),
        .rd_rsp_valid(source_rd_rsp_valid),
        .rd_rsp_ready(source_rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data),
        .rd_rsp_error(rd_rsp_error),
        .m_axis_tdata(source_m_axis_tdata),
        .m_axis_tvalid(source_m_axis_tvalid),
        .m_axis_tready(source_m_axis_tready),
        .m_axis_tlast(source_m_axis_tlast),
        .m_axis_token(source_m_axis_token),
        .m_axis_block(source_m_axis_block)
    );

    // Scratch responses are untagged. Retain the accepted requester until its
    // response handshakes, including through abort and cross-fault cleanup.
    wire can_consider_read = run_window &&
                             (scratch_owner_q == RD_NONE);
    wire scratch_request_conflict = can_consider_read &&
                                    reduce_rd_req_valid &&
                                    source_rd_req_valid;
    wire orphan_response = run_window &&
                           (scratch_owner_q == RD_NONE) && rd_rsp_valid;
    wire source_role_bad = run_window && source_rd_req_valid &&
                           (source_rd_req_role != 2'd0);
    assign internal_fault_now = scratch_request_conflict ||
                                orphan_response || source_role_bad;

    wire can_issue_read = traffic_enable && (scratch_owner_q == RD_NONE);
    wire select_reduce = can_issue_read && reduce_rd_req_valid &&
                         !source_rd_req_valid;
    wire select_source = can_issue_read && source_rd_req_valid &&
                         !reduce_rd_req_valid;

    assign rd_req_valid = select_reduce || select_source;
    assign rd_req_role = select_reduce ? 2'd0 : source_rd_req_role;
    assign rd_req_token = select_reduce ? reduce_rd_req_token :
                          source_rd_req_token;
    assign rd_req_group = select_reduce ? reduce_rd_req_group :
                          source_rd_req_group;
    assign reduce_rd_req_ready = select_reduce && rd_req_ready;
    assign source_rd_req_ready = select_source && rd_req_ready;
    wire rd_req_fire = rd_req_valid && rd_req_ready;

    assign reduce_rd_rsp_valid = rd_rsp_valid &&
                                 (scratch_owner_q == RD_REDUCE);
    assign source_rd_rsp_valid = rd_rsp_valid &&
                                 (scratch_owner_q == RD_SOURCE);
    assign rd_rsp_ready =
        (scratch_owner_q == RD_REDUCE) ? reduce_rd_rsp_ready :
        (scratch_owner_q == RD_SOURCE) ? source_rd_rsp_ready :
        rd_rsp_valid;
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            scratch_owner_q <= RD_NONE;
        end else if (rd_req_fire) begin
            scratch_owner_q <= select_reduce ? RD_REDUCE : RD_SOURCE;
        end else if (rd_rsp_fire) begin
            scratch_owner_q <= RD_NONE;
        end
    end

    wire output_enable = traffic_enable;
    assign source_m_axis_tready = m_axis_tready && output_enable;
    assign m_axis_tdata = source_m_axis_tdata;
    assign m_axis_tvalid = source_m_axis_tvalid && output_enable;
    assign m_axis_tlast = source_m_axis_tlast && output_enable;
    assign m_axis_token = source_m_axis_token;
    assign m_axis_block = source_m_axis_block;
    wire output_fire = source_m_axis_tvalid && source_m_axis_tready;
    wire final_output_fire = output_fire && source_m_axis_tlast &&
                             ({1'b0, source_m_axis_token} + 3'd1 ==
                              run_tokens_q) &&
                             (source_m_axis_block == last_block_q);

    assign busy = state_q != ST_IDLE;
    wire [29:0] child_status_now = {1'b0, source_status, reduce_status};
    wire [29:0] diagnosed_child_status =
        (child_status_now != 30'd0) ? child_status_now : STATUS_INTERNAL;
    wire [29:0] fault_status_now =
        (child_fault_now ? diagnosed_child_status : 30'd0) |
        (internal_fault_now ? STATUS_INTERNAL : 30'd0);
    wire cleanup_idle = !reduce_busy && !source_busy &&
                        (scratch_owner_q == RD_NONE);

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_tokens_q <= 3'd0;
            last_block_q <= 9'd0;
            reduce_done_seen_q <= 1'b0;
            fault_latched_q <= 1'b0;
            report_error_q <= 1'b0;
            cross_abort_q <= 1'b0;
            cleanup_abort_issued_q <= 1'b0;
            latched_status_q <= 30'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 30'd0;
        end else begin
            done <= 1'b0;

            if (abort_run) begin
                run_tokens_q <= 3'd0;
                last_block_q <= 9'd0;
                reduce_done_seen_q <= 1'b0;
                fault_latched_q <= 1'b0;
                report_error_q <= 1'b0;
                cleanup_abort_issued_q <= 1'b0;
                latched_status_q <= 30'd0;
                error <= 1'b0;
                status <= 30'd0;
                if (state_q == ST_IDLE) begin
                    state_q <= ST_IDLE;
                    cross_abort_q <= 1'b0;
                end else begin
                    state_q <= ST_CLEANUP;
                    // abort_run itself is the child abort edge. Do not schedule
                    // a redundant registered pulse after the caller releases it.
                    cross_abort_q <= 1'b0;
                    cleanup_abort_issued_q <= 1'b1;
                end
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        run_tokens_q <= 3'd0;
                        last_block_q <= 9'd0;
                        reduce_done_seen_q <= 1'b0;
                        fault_latched_q <= 1'b0;
                        report_error_q <= 1'b0;
                        cross_abort_q <= 1'b0;
                        cleanup_abort_issued_q <= 1'b0;
                        latched_status_q <= 30'd0;
                        if (cfg_fire) begin
                            // Child configuration errors are registered on this
                            // edge. Keep all tentative traffic closed until the
                            // following cycle has validated both children.
                            state_q <= ST_START;
                            run_tokens_q <= cfg_tokens;
                            last_block_q <= cfg_rows[13:5] - 1'b1;
                            error <= 1'b0;
                            status <= 30'd0;
                        end
                    end

                    ST_START: begin
                        if (child_fault_now || internal_fault_now) begin
                            state_q <= ST_CLEANUP;
                            fault_latched_q <= 1'b1;
                            report_error_q <= 1'b1;
                            cross_abort_q <= 1'b1;
                            cleanup_abort_issued_q <= 1'b0;
                            latched_status_q <= latched_status_q |
                                                fault_status_now;
                            error <= 1'b1;
                            status <= latched_status_q |
                                      fault_status_now;
                        end else begin
                            state_q <= ST_RUN;
                        end
                    end

                    ST_RUN: begin
                        if (child_fault_now || internal_fault_now) begin
                            state_q <= ST_CLEANUP;
                            fault_latched_q <= 1'b1;
                            report_error_q <= 1'b1;
                            cross_abort_q <= 1'b1;
                            cleanup_abort_issued_q <= 1'b0;
                            latched_status_q <= latched_status_q |
                                                fault_status_now;
                            error <= 1'b1;
                            status <= latched_status_q |
                                      fault_status_now;
                        end else begin
                            if (reduce_done && !reduce_error)
                                reduce_done_seen_q <= 1'b1;

                            if (source_done && !source_error) begin
                                if ((reduce_done_seen_q ||
                                     (reduce_done && !reduce_error)) &&
                                    !reduce_busy && !source_busy &&
                                    (scratch_owner_q == RD_NONE)) begin
                                    state_q <= ST_IDLE;
                                    reduce_done_seen_q <= 1'b0;
                                    done <= 1'b1;
                                    error <= 1'b0;
                                    status <= 30'd0;
                                end else begin
                                    state_q <= ST_CLEANUP;
                                    fault_latched_q <= 1'b1;
                                    report_error_q <= 1'b1;
                                    cross_abort_q <= 1'b1;
                                    cleanup_abort_issued_q <= 1'b0;
                                    latched_status_q <= latched_status_q |
                                                        STATUS_INTERNAL;
                                    error <= 1'b1;
                                    status <= latched_status_q |
                                              STATUS_INTERNAL;
                                end
                            end
                        end
                    end

                    ST_CLEANUP: begin
                        if (!cleanup_abort_issued_q) begin
                            // cross_abort_q was asserted on entry. Drop it only
                            // after both children have observed one abort edge;
                            // holding abort high would pin their cleanup states.
                            cross_abort_q <= 1'b0;
                            cleanup_abort_issued_q <= 1'b1;
                        end else begin
                            cross_abort_q <= 1'b0;
                            if (cleanup_idle) begin
                                state_q <= ST_IDLE;
                                run_tokens_q <= 3'd0;
                                last_block_q <= 9'd0;
                                reduce_done_seen_q <= 1'b0;
                                cleanup_abort_issued_q <= 1'b0;
                                if (report_error_q) begin
                                    done <= 1'b1;
                                    error <= 1'b1;
                                    status <= (latched_status_q != 30'd0) ?
                                              latched_status_q : STATUS_INTERNAL;
                                end else begin
                                    error <= 1'b0;
                                    status <= 30'd0;
                                end
                                report_error_q <= 1'b0;
                            end
                        end
                    end

                    default: begin
                        state_q <= ST_CLEANUP;
                        fault_latched_q <= 1'b1;
                        report_error_q <= 1'b1;
                        cross_abort_q <= 1'b1;
                        cleanup_abort_issued_q <= 1'b0;
                        latched_status_q <= latched_status_q |
                                            STATUS_INTERNAL;
                        error <= 1'b1;
                        status <= latched_status_q | STATUS_INTERNAL;
                    end
                endcase
            end
        end
    end

`ifdef VERILATOR
    wire [3:0] debug_state /* verilator public_flat_rd */ =
        {2'b00, state_q};
    wire [1:0] debug_read_owner /* verilator public_flat_rd */ =
        scratch_owner_q;
    wire [3:0] debug_reduce_state /* verilator public_flat_rd */ =
        {2'b00, u_reduce.state_q};
    wire [3:0] debug_reduce_frontend_state /* verilator public_flat_rd */ =
        {1'b0, u_reduce.u_frontend.state_q};
    wire [3:0] debug_reduce_inverse_state /* verilator public_flat_rd */ =
        u_reduce.u_inverse.state_q;
    wire [3:0] debug_source_state /* verilator public_flat_rd */ =
        {2'b00, u_source.state_q};
    wire [3:0] debug_weighted_state /* verilator public_flat_rd */ =
        u_source.u_weighted.state_q;
    wire [3:0] debug_q8_state /* verilator public_flat_rd */ =
        u_source.u_q8.state;
    wire [3:0] debug_q8_quantizer_state /* verilator public_flat_rd */ =
        u_source.u_q8.u_quantizer.state;
    wire [2:0] debug_q8_emit_index /* verilator public_flat_rd */ =
        u_source.u_q8.emit_index;
    wire debug_q8_record_done /* verilator public_flat_rd */ =
        u_source.q8_record_done;
    wire debug_output_fire /* verilator public_flat_rd */ = output_fire;
    wire debug_final_output_fire /* verilator public_flat_rd */ =
        final_output_fire;
    wire debug_reduce_busy /* verilator public_flat_rd */ = reduce_busy;
    wire debug_reduce_done /* verilator public_flat_rd */ = reduce_done;
    wire debug_reduce_error /* verilator public_flat_rd */ = reduce_error;
    wire debug_source_busy /* verilator public_flat_rd */ = source_busy;
    wire debug_source_done /* verilator public_flat_rd */ = source_done;
    wire debug_source_error /* verilator public_flat_rd */ = source_error;
    wire debug_child_fault /* verilator public_flat_rd */ =
        child_fault_now || internal_fault_now;
    wire debug_child_abort /* verilator public_flat_rd */ = child_abort;
`endif

`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_cfg_fire = cfg_fire;
    assign formal_reduce_busy = reduce_busy;
    assign formal_reduce_done = reduce_done;
    assign formal_reduce_error = reduce_error;
    assign formal_reduce_status = reduce_status;
    assign formal_source_busy = source_busy;
    assign formal_source_done = source_done;
    assign formal_source_error = source_error;
    assign formal_source_status = source_status;
    assign formal_reduce_result_fire = reduce_result_fire;
    assign formal_scratch_owner = scratch_owner_q;
    assign formal_reduce_rd_req_fire = reduce_rd_req_valid &&
                                       reduce_rd_req_ready;
    assign formal_source_rd_req_fire = source_rd_req_valid &&
                                       source_rd_req_ready;
    assign formal_rd_rsp_fire = rd_rsp_fire;
    assign formal_run_window = run_window;
    assign formal_traffic_enable = traffic_enable;
    assign formal_internal_fault = internal_fault_now;
    assign formal_reduce_rd_req_valid_raw = reduce_rd_req_valid;
    assign formal_source_rd_req_valid_raw = source_rd_req_valid;
    assign formal_cross_abort = cross_abort_q;
    assign formal_cleanup_abort_issued = cleanup_abort_issued_q;
    assign formal_reduce_done_seen = reduce_done_seen_q;
    assign formal_fault_latched = fault_latched_q;
    assign formal_latched_status = latched_status_q;
    assign formal_output_fire = output_fire;
    assign formal_final_output_fire = final_output_fire;

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(done && busy));
            assert(scratch_owner_q != 2'd3);
            assert(!(reduce_rd_rsp_valid && source_rd_rsp_valid));
            assert(!(reduce_rd_req_ready && source_rd_req_ready));
            if (m_axis_tvalid)
                assert(state_q == ST_RUN && !child_fault_now &&
                       !internal_fault_now);
            if (internal_fault_now) begin
                assert(!s_axis_tready);
                assert(!r_wr_valid);
                assert(!reduce_result_fire);
                assert(!rd_req_valid);
                assert(!m_axis_tvalid);
            end
            if (scratch_owner_q == RD_REDUCE)
                assert(!source_rd_rsp_valid);
            if (scratch_owner_q == RD_SOURCE)
                assert(!reduce_rd_rsp_valid);
        end
        if (f_past_valid && rst_n && !abort_run && !child_fault_now &&
            !internal_fault_now &&
            $past(rst_n && !abort_run && m_axis_tvalid &&
                  !m_axis_tready)) begin
            assert(m_axis_tvalid);
            assert(m_axis_tdata == $past(m_axis_tdata));
            assert(m_axis_tlast == $past(m_axis_tlast));
            assert(m_axis_token == $past(m_axis_token));
            assert(m_axis_block == $past(m_axis_block));
        end
    end
`endif

endmodule

`default_nettype wire
