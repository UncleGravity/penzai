// Standalone scratch-backed weighted RMSNorm to native Q8_0 pipeline.
//
// Production integration consumes section_rmsnorm_scalar_pipeline directly so
// the decode top can time-share its existing q8_ingress. This compatibility
// wrapper retains the standalone 30-bit contract and debug/formal surface by
// pairing that scalar pipeline with one private Q8 adapter.

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
    input  wire          cfg_resident,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bits 12:0 reduction raw status, bits 21:13 weighted raw status,
    // bits 27:22 Q8 raw status, bit 28 legacy source-internal status,
    // bit 29 scalar/full-pipeline lifecycle and ownership status.
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
    , output wire        formal_scalar_cross_abort
    , output wire        formal_cleanup_abort_issued
    , output wire        formal_reduce_done_seen
    , output wire        formal_fault_latched
    , output wire [29:0] formal_latched_status
    , output wire        formal_output_fire
    , output wire        formal_final_output_fire
    , output wire        formal_source_internal
    , output wire        formal_source_completion_mismatch
`endif
);
    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_RUN     = 2'd1;
    localparam [1:0] ST_CLEANUP = 2'd2;
    localparam [1:0] ST_START   = 2'd3;

    localparam [1:0] SRC_IDLE     = 2'd0;
    localparam [1:0] SRC_RUN      = 2'd1;
    localparam [1:0] SRC_Q8_DRAIN = 2'd2;
    localparam [1:0] SRC_CLEANUP  = 2'd3;

    localparam [29:0] STATUS_Q8_INTERNAL = 30'h1000_0000;
    localparam [29:0] STATUS_INTERNAL = 30'h2000_0000;

    reg [1:0] state_q;
    reg [1:0] source_state_q;
    reg [2:0] beat_index_q;
    reg [1:0] token_q;
    reg [8:0] block_q;
    reg [8:0] last_block_q;
    reg [2:0] run_tokens_q;
    reg scratch_owned_q;
    reg scalar_done_seen_q;
    reg source_done_q /* verilator public_flat_rw */;
    reg fault_latched_q;
    reg report_error_q;
    reg cross_abort_q;
    reg cleanup_abort_issued_q;
    reg [29:0] latched_status_q;

    wire wrapper_idle = state_q == ST_IDLE;
    wire child_abort = abort_run || cross_abort_q;

    wire scalar_gamma_cfg_ready;
    wire scalar_gamma_tready;
    wire scalar_cfg_ready;
    wire scalar_busy;
    wire scalar_done;
    wire scalar_error;
    wire [22:0] scalar_pipeline_status;
    wire [31:0] weighted_scalar_data;
    wire weighted_scalar_valid;
    wire weighted_scalar_ready;
    wire weighted_scalar_last;
    wire [1:0] weighted_scalar_status;

`ifdef FORMAL
    wire [1:0] scalar_formal_state;
    wire scalar_formal_cfg_fire;
    wire scalar_formal_reduce_busy;
    wire scalar_formal_reduce_done;
    wire scalar_formal_reduce_error;
    wire [12:0] scalar_formal_reduce_status;
    wire scalar_formal_source_busy;
    wire scalar_formal_source_done;
    wire scalar_formal_source_error;
    wire [8:0] scalar_formal_source_status;
    wire scalar_formal_reduce_result_fire;
    wire [1:0] scalar_formal_scratch_owner;
    wire scalar_formal_reduce_rd_req_fire;
    wire scalar_formal_source_rd_req_fire;
    wire scalar_formal_rd_rsp_fire;
    wire scalar_formal_run_window;
    wire scalar_formal_traffic_enable;
    wire scalar_formal_internal_fault;
    wire scalar_formal_reduce_rd_req_valid_raw;
    wire scalar_formal_source_rd_req_valid_raw;
    wire scalar_formal_cross_abort;
    wire scalar_formal_cleanup_abort_issued;
    wire scalar_formal_reduce_done_seen;
    wire scalar_formal_source_done_seen;
    wire scalar_formal_final_output_seen;
    wire scalar_formal_fault_latched;
    wire [22:0] scalar_formal_latched_status;
    wire scalar_formal_output_fire;
    wire scalar_formal_final_output_fire;
    wire [3:0] q8_formal_state;
    wire [5:0] q8_formal_scalar_index;
    wire [2:0] q8_formal_emit_index;
    wire q8_formal_staged_valid;
    wire q8_formal_quant_input_fire;
    wire q8_formal_quant_output_valid;
`endif

    assign gamma_cfg_ready = wrapper_idle && scalar_gamma_cfg_ready;
    assign gamma_tready = wrapper_idle && scalar_gamma_tready;
    assign cfg_ready = wrapper_idle && scalar_cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;

    section_rmsnorm_scalar_pipeline #(
        .MIN_ROWS(MIN_ROWS), .MAX_ROWS(MAX_ROWS)
    ) u_scalar (
        .clk(clk), .rst_n(rst_n), .abort_run(child_abort),
        .gamma_cfg_valid(gamma_cfg_valid && wrapper_idle),
        .gamma_cfg_ready(scalar_gamma_cfg_ready),
        .gamma_cfg_rows(gamma_cfg_rows), .gamma_tdata(gamma_tdata),
        .gamma_tkeep(gamma_tkeep),
        .gamma_tvalid(gamma_tvalid && wrapper_idle),
        .gamma_tready(scalar_gamma_tready), .gamma_tlast(gamma_tlast),
        .gamma_busy(gamma_busy), .gamma_done(gamma_done),
        .gamma_error(gamma_error), .gamma_status(gamma_status),
        .gamma_valid(gamma_valid), .cfg_valid(cfg_fire),
        .cfg_ready(scalar_cfg_ready), .cfg_rows(cfg_rows),
        .cfg_tokens(cfg_tokens), .cfg_eps(cfg_eps),
        .cfg_resident(cfg_resident), .busy(scalar_busy),
        .done(scalar_done), .error(scalar_error),
        .status(scalar_pipeline_status), .s_axis_tdata(s_axis_tdata),
        .s_axis_tkeep(s_axis_tkeep), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_error(r_wr_error), .r_wr_bank(r_wr_bank),
        .r_wr_address(r_wr_address), .r_wr_data(r_wr_data),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group), .rd_rsp_valid(rd_rsp_valid),
        .rd_rsp_ready(rd_rsp_ready), .rd_rsp_data(rd_rsp_data),
        .rd_rsp_error(rd_rsp_error), .scalar_data(weighted_scalar_data),
        .scalar_valid(weighted_scalar_valid),
        .scalar_ready(weighted_scalar_ready),
        .scalar_last(weighted_scalar_last),
        .scalar_status(weighted_scalar_status)
`ifdef FORMAL
        , .formal_state(scalar_formal_state)
        , .formal_cfg_fire(scalar_formal_cfg_fire)
        , .formal_reduce_busy(scalar_formal_reduce_busy)
        , .formal_reduce_done(scalar_formal_reduce_done)
        , .formal_reduce_error(scalar_formal_reduce_error)
        , .formal_reduce_status(scalar_formal_reduce_status)
        , .formal_source_busy(scalar_formal_source_busy)
        , .formal_source_done(scalar_formal_source_done)
        , .formal_source_error(scalar_formal_source_error)
        , .formal_source_status(scalar_formal_source_status)
        , .formal_reduce_result_fire(scalar_formal_reduce_result_fire)
        , .formal_scratch_owner(scalar_formal_scratch_owner)
        , .formal_reduce_rd_req_fire(scalar_formal_reduce_rd_req_fire)
        , .formal_source_rd_req_fire(scalar_formal_source_rd_req_fire)
        , .formal_rd_rsp_fire(scalar_formal_rd_rsp_fire)
        , .formal_run_window(scalar_formal_run_window)
        , .formal_traffic_enable(scalar_formal_traffic_enable)
        , .formal_internal_fault(scalar_formal_internal_fault)
        , .formal_reduce_rd_req_valid_raw(
            scalar_formal_reduce_rd_req_valid_raw)
        , .formal_source_rd_req_valid_raw(
            scalar_formal_source_rd_req_valid_raw)
        , .formal_cross_abort(scalar_formal_cross_abort)
        , .formal_cleanup_abort_issued(
            scalar_formal_cleanup_abort_issued)
        , .formal_reduce_done_seen(scalar_formal_reduce_done_seen)
        , .formal_source_done_seen(scalar_formal_source_done_seen)
        , .formal_final_output_seen(scalar_formal_final_output_seen)
        , .formal_fault_latched(scalar_formal_fault_latched)
        , .formal_latched_status(scalar_formal_latched_status)
        , .formal_output_fire(scalar_formal_output_fire)
        , .formal_final_output_fire(scalar_formal_final_output_fire)
`endif
    );

    wire [63:0] q8_m_axis_tdata;
    wire q8_m_axis_tvalid;
    wire q8_m_axis_tready;
    wire q8_activation_abort;
    wire [5:0] q8_status;
    wire q8_record_done;
    wire unused_q8_raw_ready;
    wire [15:0] q8_num_blocks = {2'b00, cfg_rows} >> 7;
    wire [15:0] q8_num_cols = {13'd0, cfg_tokens};

    q8_ingress u_q8 (
        .clk(clk), .rst_n(rst_n), .start(cfg_fire), .abort(child_abort),
        .raw_mode(1'b1), .internal_mode(1'b1),
        .num_q1_blocks(q8_num_blocks), .num_cols(q8_num_cols),
        .s_axis_tdata(64'd0), .s_axis_tvalid(1'b0),
        .s_axis_tready(unused_q8_raw_ready), .s_axis_tlast(1'b0),
        .internal_data(weighted_scalar_data),
        .internal_last(weighted_scalar_last),
        .internal_status(weighted_scalar_status),
        .internal_valid(weighted_scalar_valid),
        .internal_ready(weighted_scalar_ready),
        .internal_record_done(q8_record_done),
        .m_axis_tdata(q8_m_axis_tdata),
        .m_axis_tvalid(q8_m_axis_tvalid),
        .m_axis_tready(q8_m_axis_tready),
        .activation_abort(q8_activation_abort),
        .quantizer_status(q8_status)
`ifdef FORMAL
        , .formal_state(q8_formal_state)
        , .formal_scalar_index(q8_formal_scalar_index)
        , .formal_emit_index(q8_formal_emit_index)
        , .formal_staged_valid(q8_formal_staged_valid)
        , .formal_quant_input_fire(q8_formal_quant_input_fire)
        , .formal_quant_output_valid(q8_formal_quant_output_valid)
`endif
    );

    wire run_window = rst_n && !abort_run && (state_q == ST_RUN) &&
                      !cross_abort_q;
    wire rd_req_fire = rd_req_valid && rd_req_ready;
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;
    wire outer_orphan_response = run_window && !scratch_owned_q &&
                                 rd_rsp_valid;
    wire child_fault_now = scalar_error || q8_activation_abort ||
                           outer_orphan_response;
    wire output_enable = run_window && !child_fault_now;
    assign q8_m_axis_tready = m_axis_tready && output_enable;
    assign m_axis_tdata = q8_m_axis_tdata;
    assign m_axis_tvalid = q8_m_axis_tvalid && output_enable;
    assign m_axis_tlast = output_enable && (beat_index_q == 3'd4);
    assign m_axis_token = token_q;
    assign m_axis_block = block_q;
    wire output_fire = q8_m_axis_tvalid && q8_m_axis_tready;
    wire record_fire = output_fire && (beat_index_q == 3'd4);
    wire q8_record_mismatch = run_window &&
                              (q8_record_done != record_fire);
    wire final_output_fire = record_fire &&
                             ({1'b0, token_q} + 3'd1 == run_tokens_q) &&
                             (block_q == last_block_q);
    wire scalar_clean_complete =
        (scalar_done_seen_q || (scalar_done && !scalar_error)) &&
        !scalar_busy;
    wire source_completion_mismatch = run_window && source_done_q &&
                                      !scalar_clean_complete;
    // Preserve the legacy q8-source bit-15 meaning even after the refactor:
    // final Q8 publication must not retire before its clean scalar producer.
    wire source_internal_active = latched_status_q[28] ||
                                  q8_record_mismatch ||
                                  source_completion_mismatch;
    wire source_error = q8_activation_abort || source_internal_active ||
        (scalar_error && (scalar_pipeline_status[21:13] != 9'd0));
    wire source_busy = source_state_q != SRC_IDLE;
    wire [15:0] source_status = {
        source_internal_active, q8_status, scalar_pipeline_status[21:13]
    };
    wire reduce_error = scalar_error &&
                        (scalar_pipeline_status[12:0] != 13'd0);

    assign busy = state_q != ST_IDLE;
    wire [29:0] child_status_now = {
        scalar_pipeline_status[22], source_internal_active, q8_status,
        scalar_pipeline_status[21:0]
    };
    wire [29:0] diagnosed_child_status =
        (child_status_now != 30'd0) ? child_status_now : STATUS_INTERNAL;
    wire [29:0] fault_status_now =
        (child_fault_now ? diagnosed_child_status : 30'd0) |
        ((q8_record_mismatch || source_completion_mismatch) ?
         STATUS_Q8_INTERNAL : 30'd0) |
        (outer_orphan_response ? STATUS_INTERNAL : 30'd0);
    wire cleanup_idle = !scalar_busy;

    always @(posedge clk) begin
        if (!rst_n)
            scratch_owned_q <= 1'b0;
        else if (rd_req_fire)
            scratch_owned_q <= 1'b1;
        else if (rd_rsp_fire)
            scratch_owned_q <= 1'b0;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            source_state_q <= SRC_IDLE;
            beat_index_q <= 3'd0;
            token_q <= 2'd0;
            block_q <= 9'd0;
            last_block_q <= 9'd0;
            run_tokens_q <= 3'd0;
            scalar_done_seen_q <= 1'b0;
            source_done_q <= 1'b0;
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
            source_done_q <= 1'b0;

            if (abort_run) begin
                source_state_q <= SRC_CLEANUP;
                beat_index_q <= 3'd0;
                token_q <= 2'd0;
                block_q <= 9'd0;
                last_block_q <= 9'd0;
                run_tokens_q <= 3'd0;
                scalar_done_seen_q <= 1'b0;
                fault_latched_q <= 1'b0;
                report_error_q <= 1'b0;
                cleanup_abort_issued_q <= 1'b0;
                latched_status_q <= 30'd0;
                error <= 1'b0;
                status <= 30'd0;
                if (state_q == ST_IDLE) begin
                    state_q <= ST_IDLE;
                    source_state_q <= SRC_IDLE;
                    cross_abort_q <= 1'b0;
                end else begin
                    state_q <= ST_CLEANUP;
                    cross_abort_q <= 1'b0;
                    cleanup_abort_issued_q <= 1'b1;
                end
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        source_state_q <= SRC_IDLE;
                        beat_index_q <= 3'd0;
                        token_q <= 2'd0;
                        block_q <= 9'd0;
                        last_block_q <= 9'd0;
                        run_tokens_q <= 3'd0;
                        scalar_done_seen_q <= 1'b0;
                        fault_latched_q <= 1'b0;
                        report_error_q <= 1'b0;
                        cross_abort_q <= 1'b0;
                        cleanup_abort_issued_q <= 1'b0;
                        latched_status_q <= 30'd0;
                        if (cfg_fire) begin
                            state_q <= ST_START;
                            source_state_q <= SRC_RUN;
                            run_tokens_q <= cfg_tokens;
                            last_block_q <= cfg_rows[13:5] - 1'b1;
                            error <= 1'b0;
                            status <= 30'd0;
                        end
                    end

                    ST_START: begin
                        if (child_fault_now || q8_record_mismatch ||
                            source_completion_mismatch) begin
                            state_q <= ST_CLEANUP;
                            source_state_q <= SRC_CLEANUP;
                            fault_latched_q <= 1'b1;
                            report_error_q <= 1'b1;
                            cross_abort_q <= 1'b1;
                            cleanup_abort_issued_q <= 1'b0;
                            latched_status_q <= latched_status_q |
                                                fault_status_now;
                            error <= 1'b1;
                            status <= latched_status_q | fault_status_now;
                        end else begin
                            state_q <= ST_RUN;
                        end
                    end

                    ST_RUN: begin
                        if (output_fire) begin
                            if (beat_index_q == 3'd4) begin
                                beat_index_q <= 3'd0;
                                if (final_output_fire) begin
                                    source_state_q <= SRC_IDLE;
                                    source_done_q <= 1'b1;
                                end else if (block_q == last_block_q) begin
                                    block_q <= 9'd0;
                                    token_q <= token_q + 1'b1;
                                end else begin
                                    block_q <= block_q + 1'b1;
                                end
                            end else begin
                                beat_index_q <= beat_index_q + 1'b1;
                            end
                        end
                        if (scalar_done && !scalar_error) begin
                            scalar_done_seen_q <= 1'b1;
                            if (!final_output_fire)
                                source_state_q <= SRC_Q8_DRAIN;
                        end

                        if (child_fault_now || q8_record_mismatch ||
                            source_completion_mismatch) begin
                            state_q <= ST_CLEANUP;
                            source_state_q <= SRC_CLEANUP;
                            fault_latched_q <= 1'b1;
                            report_error_q <= 1'b1;
                            cross_abort_q <= 1'b1;
                            cleanup_abort_issued_q <= 1'b0;
                            latched_status_q <= latched_status_q |
                                                fault_status_now;
                            error <= 1'b1;
                            status <= latched_status_q | fault_status_now;
                        end else if (source_done_q) begin
                            if ((scalar_done_seen_q ||
                                 (scalar_done && !scalar_error)) &&
                                !scalar_busy) begin
                                state_q <= ST_IDLE;
                                scalar_done_seen_q <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b0;
                                status <= 30'd0;
                            end else begin
                                state_q <= ST_CLEANUP;
                                source_state_q <= SRC_CLEANUP;
                                fault_latched_q <= 1'b1;
                                report_error_q <= 1'b1;
                                cross_abort_q <= 1'b1;
                                cleanup_abort_issued_q <= 1'b0;
                                latched_status_q <= latched_status_q |
                                                    STATUS_Q8_INTERNAL;
                                error <= 1'b1;
                                status <= latched_status_q |
                                          STATUS_Q8_INTERNAL;
                            end
                        end
                    end

                    ST_CLEANUP: begin
                        if (!cleanup_abort_issued_q) begin
                            cross_abort_q <= 1'b0;
                            cleanup_abort_issued_q <= 1'b1;
                        end else begin
                            cross_abort_q <= 1'b0;
                            if (cleanup_idle) begin
                                state_q <= ST_IDLE;
                                source_state_q <= SRC_IDLE;
                                beat_index_q <= 3'd0;
                                token_q <= 2'd0;
                                block_q <= 9'd0;
                                last_block_q <= 9'd0;
                                run_tokens_q <= 3'd0;
                                scalar_done_seen_q <= 1'b0;
                                cleanup_abort_issued_q <= 1'b0;
                                if (report_error_q) begin
                                    done <= 1'b1;
                                    error <= 1'b1;
                                    status <= (latched_status_q != 30'd0) ?
                                              latched_status_q :
                                              STATUS_INTERNAL;
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
                        source_state_q <= SRC_CLEANUP;
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
        u_scalar.scratch_owner_q;
    wire [3:0] debug_reduce_state /* verilator public_flat_rd */ =
        {2'b00, u_scalar.u_reduce.state_q};
    wire [3:0] debug_reduce_frontend_state /* verilator public_flat_rd */ =
        {1'b0, u_scalar.u_reduce.u_frontend.state_q};
    wire [3:0] debug_reduce_inverse_state /* verilator public_flat_rd */ =
        u_scalar.u_reduce.u_inverse.state_q;
    wire [3:0] debug_source_state /* verilator public_flat_rd */ =
        {2'b00, source_state_q};
    wire [3:0] debug_weighted_state /* verilator public_flat_rd */ =
        u_scalar.u_source.state_q;
    wire [3:0] debug_q8_state /* verilator public_flat_rd */ = u_q8.state;
    wire [3:0] debug_q8_quantizer_state /* verilator public_flat_rd */ =
        u_q8.u_quantizer.state;
    wire [2:0] debug_q8_emit_index /* verilator public_flat_rd */ =
        u_q8.emit_index;
    wire debug_q8_record_done /* verilator public_flat_rd */ =
        q8_record_done;
    wire debug_output_fire /* verilator public_flat_rd */ = output_fire;
    wire debug_final_output_fire /* verilator public_flat_rd */ =
        final_output_fire;
    wire debug_reduce_busy /* verilator public_flat_rd */ =
        u_scalar.u_reduce.busy;
    wire debug_reduce_done /* verilator public_flat_rd */ =
        u_scalar.u_reduce.done;
    wire debug_reduce_error /* verilator public_flat_rd */ = reduce_error;
    wire debug_source_busy /* verilator public_flat_rd */ = source_busy;
    wire debug_source_done /* verilator public_flat_rd */ = source_done_q;
    wire debug_source_error /* verilator public_flat_rd */ = source_error;
    wire [15:0] debug_source_status /* verilator public_flat_rd */ =
        source_status;
    wire debug_child_fault /* verilator public_flat_rd */ =
        child_fault_now || q8_record_mismatch || source_completion_mismatch;
    wire debug_child_abort /* verilator public_flat_rd */ = child_abort;
`endif

`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_cfg_fire = cfg_fire;
    assign formal_reduce_busy = scalar_formal_reduce_busy;
    assign formal_reduce_done = scalar_formal_reduce_done;
    assign formal_reduce_error = scalar_formal_reduce_error;
    assign formal_reduce_status = scalar_formal_reduce_status;
    assign formal_source_busy = source_busy;
    assign formal_source_done = source_done_q;
    assign formal_source_error = source_error;
    assign formal_source_status = source_status;
    assign formal_reduce_result_fire = scalar_formal_reduce_result_fire;
    assign formal_scratch_owner = scalar_formal_scratch_owner;
    assign formal_reduce_rd_req_fire = scalar_formal_reduce_rd_req_fire;
    assign formal_source_rd_req_fire = scalar_formal_source_rd_req_fire;
    assign formal_rd_rsp_fire = scalar_formal_rd_rsp_fire;
    assign formal_run_window = run_window;
    assign formal_traffic_enable = output_enable;
    assign formal_internal_fault = scalar_formal_internal_fault ||
                                   q8_record_mismatch ||
                                   source_completion_mismatch ||
                                   outer_orphan_response;
    assign formal_reduce_rd_req_valid_raw =
        scalar_formal_reduce_rd_req_valid_raw;
    assign formal_source_rd_req_valid_raw =
        scalar_formal_source_rd_req_valid_raw;
    assign formal_cross_abort = cross_abort_q;
    assign formal_scalar_cross_abort = scalar_formal_cross_abort;
    assign formal_cleanup_abort_issued = cleanup_abort_issued_q;
    assign formal_reduce_done_seen = scalar_formal_reduce_done_seen;
    assign formal_fault_latched = fault_latched_q;
    assign formal_latched_status = latched_status_q;
    assign formal_output_fire = output_fire;
    assign formal_final_output_fire = final_output_fire;
    assign formal_source_internal = source_internal_active;
    assign formal_source_completion_mismatch = source_completion_mismatch;

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(done && busy));
            assert(formal_source_status[15] == source_internal_active);
            if (source_completion_mismatch) begin
                assert(formal_source_status[15]);
                assert(fault_status_now[28]);
            end
            if (scalar_pipeline_status[22])
                assert(child_status_now[29]);
            if (m_axis_tvalid)
                assert(state_q == ST_RUN && !child_fault_now &&
                       !q8_record_mismatch);
            assert(!q8_record_mismatch);
        end
        if (f_past_valid && rst_n && !abort_run && !child_fault_now &&
            !q8_record_mismatch &&
            $past(rst_n && !abort_run && m_axis_tvalid &&
                  !m_axis_tready)) begin
            assert(m_axis_tvalid);
            assert(m_axis_tdata == $past(m_axis_tdata));
            assert(m_axis_tlast == $past(m_axis_tlast));
            assert(m_axis_token == $past(m_axis_token));
            assert(m_axis_block == $past(m_axis_block));
        end
        if (f_past_valid && rst_n &&
            $past(rst_n && source_completion_mismatch &&
                  !scalar_pipeline_status[22] && !outer_orphan_response)) begin
            assert(fault_latched_q && error);
            assert(latched_status_q[28]);
            assert(!latched_status_q[29]);
        end
    end
`endif

endmodule

`default_nettype wire
