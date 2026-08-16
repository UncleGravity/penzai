// Scratch-backed weighted RMSNorm source with native Q8_0 publication.
//
// The weighted source and the canonical Q8 ingress start atomically. Weighted
// completion only says that the final scalar is staged at the Q8 boundary;
// clean public completion waits for the final record's fifth beat handshake.
// Published records are tentative until that clean completion pulse.

`default_nettype none

module section_rmsnorm_q8_source #(
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
    output wire          busy,
    output reg           done,
    output reg           error,
    // bits 8:0 weighted raw status, bits 14:9 Q8 raw status,
    // bit 15 wrapper lifecycle/protocol status.
    output reg  [15:0]   status,

    input  wire          inv_valid,
    output wire          inv_ready,
    input  wire [1:0]    inv_token,
    input  wire [31:0]   inv_rms,
    input  wire          inv_final,

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
    , output wire [3:0]  formal_weighted_state
    , output wire        formal_weighted_read_owned
    , output wire [1:0]  formal_weighted_mul_phase
    , output wire        formal_weighted_scalar_fire
    , output wire        formal_q8_start
    , output wire        formal_q8_abort
    , output wire [3:0]  formal_q8_state
    , output wire        formal_q8_activation_fault
    , output wire        formal_q8_record_done
    , output wire [5:0]  formal_q8_scalar_index
    , output wire [2:0]  formal_q8_emit_index
    , output wire        formal_q8_staged_valid
    , output wire        formal_q8_quant_input_fire
    , output wire        formal_q8_quant_output_valid
    , output wire [2:0]  formal_beat_index
    , output wire [1:0]  formal_token
    , output wire [8:0]  formal_block
    , output wire        formal_source_done_seen
    , output wire        formal_fault_latched
    , output wire        formal_cleanup_abort_issued
    , output wire [15:0] formal_latched_status
`endif
);
    localparam [1:0] ST_IDLE     = 2'd0;
    localparam [1:0] ST_RUN      = 2'd1;
    localparam [1:0] ST_Q8_DRAIN = 2'd2;
    localparam [1:0] ST_CLEANUP  = 2'd3;

    localparam [15:0] STATUS_INTERNAL = 16'h8000;

    reg [1:0]  state_q;
    reg [2:0]  beat_index_q;
    reg [1:0]  token_q;
    reg [8:0]  block_q;
    reg [8:0]  last_block_q;
    reg [2:0]  run_tokens_q;
    reg        source_done_seen_q;
    reg        fault_latched_q;
    reg        report_error_q;
    reg        cross_abort_q;
    reg        cleanup_abort_issued_q;
    reg [15:0] latched_status_q;

    wire wrapper_idle = state_q == ST_IDLE;
    wire output_state = (state_q == ST_RUN) ||
                        (state_q == ST_Q8_DRAIN);
    wire child_abort = abort_run || cross_abort_q;

    wire weighted_gamma_cfg_ready;
    wire weighted_gamma_tready;
    wire weighted_cfg_ready;
    wire weighted_busy;
    wire weighted_done;
    wire weighted_error;
    wire [8:0] weighted_status;
    wire weighted_scalar_valid;
    wire weighted_scalar_ready;
    wire [31:0] weighted_scalar_data;
    wire weighted_scalar_last;
    wire [1:0] weighted_scalar_status;

    wire weighted_gamma_cfg_valid = gamma_cfg_valid && wrapper_idle;
    wire weighted_gamma_tvalid = gamma_tvalid && wrapper_idle;
    wire weighted_cfg_valid = cfg_valid && wrapper_idle;

    assign gamma_cfg_ready = wrapper_idle && weighted_gamma_cfg_ready;
    assign gamma_tready = wrapper_idle && weighted_gamma_tready;
    assign cfg_ready = wrapper_idle && weighted_cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;

`ifdef FORMAL
    wire [13:0] weighted_formal_gamma_rows;
    wire [2:0] weighted_formal_scale_count;
    wire weighted_formal_rsp_buffered;
    wire [1:0] weighted_formal_run_token;
    wire [10:0] weighted_formal_run_group;
    wire [2:0] weighted_formal_run_lane;
    wire weighted_formal_mul_s_fire;
    wire [31:0] weighted_formal_mul_s_a;
    wire [31:0] weighted_formal_mul_s_b;
    wire weighted_formal_mul_result_fire;
    wire [1:0] weighted_formal_mul_result_status;
    wire [31:0] weighted_formal_mul_result_data;
    wire [3:0] q8_formal_state;
    wire [5:0] q8_formal_scalar_index;
    wire [2:0] q8_formal_emit_index;
    wire q8_formal_staged_valid;
    wire q8_formal_quant_input_fire;
    wire q8_formal_quant_output_valid;
`endif

    section_rmsnorm_weighted_source #(
        .MIN_ROWS(MIN_ROWS),
        .MAX_ROWS(MAX_ROWS)
    ) u_weighted (
        .clk(clk),
        .rst_n(rst_n),
        .abort_run(child_abort),
        .gamma_cfg_valid(weighted_gamma_cfg_valid),
        .gamma_cfg_ready(weighted_gamma_cfg_ready),
        .gamma_cfg_rows(gamma_cfg_rows),
        .gamma_tdata(gamma_tdata),
        .gamma_tkeep(gamma_tkeep),
        .gamma_tvalid(weighted_gamma_tvalid),
        .gamma_tready(weighted_gamma_tready),
        .gamma_tlast(gamma_tlast),
        .gamma_busy(gamma_busy),
        .gamma_done(gamma_done),
        .gamma_error(gamma_error),
        .gamma_status(gamma_status),
        .gamma_valid(gamma_valid),
        .cfg_valid(weighted_cfg_valid),
        .cfg_ready(weighted_cfg_ready),
        .cfg_rows(cfg_rows),
        .cfg_tokens(cfg_tokens),
        .busy(weighted_busy),
        .done(weighted_done),
        .error(weighted_error),
        .status(weighted_status),
        .inv_valid(inv_valid),
        .inv_ready(inv_ready),
        .inv_token(inv_token),
        .inv_rms(inv_rms),
        .inv_final(inv_final),
        .rd_req_valid(rd_req_valid),
        .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role),
        .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid),
        .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data),
        .rd_rsp_error(rd_rsp_error),
        .scalar_valid(weighted_scalar_valid),
        .scalar_ready(weighted_scalar_ready),
        .scalar_data(weighted_scalar_data),
        .scalar_last(weighted_scalar_last),
        .scalar_status(weighted_scalar_status)
`ifdef FORMAL
        , .formal_state(formal_weighted_state)
        , .formal_gamma_rows(weighted_formal_gamma_rows)
        , .formal_scale_count(weighted_formal_scale_count)
        , .formal_read_owned(formal_weighted_read_owned)
        , .formal_rsp_buffered(weighted_formal_rsp_buffered)
        , .formal_run_token(weighted_formal_run_token)
        , .formal_run_group(weighted_formal_run_group)
        , .formal_run_lane(weighted_formal_run_lane)
        , .formal_mul_phase(formal_weighted_mul_phase)
        , .formal_mul_s_fire(weighted_formal_mul_s_fire)
        , .formal_mul_s_a(weighted_formal_mul_s_a)
        , .formal_mul_s_b(weighted_formal_mul_s_b)
        , .formal_mul_result_fire(weighted_formal_mul_result_fire)
        , .formal_mul_result_status(weighted_formal_mul_result_status)
        , .formal_mul_result_data(weighted_formal_mul_result_data)
`endif
    );

    wire [63:0] q8_m_axis_tdata;
    wire q8_m_axis_tvalid;
    wire q8_m_axis_tready;
    wire q8_activation_abort;
    wire [5:0] q8_status;
    wire q8_record_done;
    wire unused_q8_raw_ready;
    wire q8_start = cfg_fire;
    wire [15:0] q8_num_blocks = {2'b00, cfg_rows} >> 7;
    wire [15:0] q8_num_cols = {13'd0, cfg_tokens};

    wire child_fault_now = weighted_error || q8_activation_abort;
    wire output_enable = rst_n && output_state && !abort_run &&
                         !cross_abort_q && !child_fault_now;
    assign q8_m_axis_tready = m_axis_tready && output_enable;
    wire output_fire = q8_m_axis_tvalid && q8_m_axis_tready;

    q8_ingress u_q8 (
        .clk(clk),
        .rst_n(rst_n),
        .start(q8_start),
        .abort(child_abort),
        .raw_mode(1'b1),
        .internal_mode(1'b1),
        .num_q1_blocks(q8_num_blocks),
        .num_cols(q8_num_cols),
        .s_axis_tdata(64'd0),
        .s_axis_tvalid(1'b0),
        .s_axis_tready(unused_q8_raw_ready),
        .s_axis_tlast(1'b0),
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

    assign m_axis_tdata = q8_m_axis_tdata;
    assign m_axis_tvalid = q8_m_axis_tvalid && output_enable;
    assign m_axis_tlast = output_enable && (beat_index_q == 3'd4);
    assign m_axis_token = token_q;
    assign m_axis_block = block_q;
    assign busy = state_q != ST_IDLE;

    wire final_token = ({1'b0, token_q} + 3'd1) == run_tokens_q;
    wire final_block = block_q == last_block_q;
    wire final_record_fire = output_fire && (beat_index_q == 3'd4) &&
                             final_token && final_block;
    wire source_clean_done = weighted_done && !weighted_error &&
                             (weighted_status == 9'd0);
    wire [15:0] child_status_now = {1'b0, q8_status, weighted_status};
    wire [15:0] diagnosed_status_now =
        (child_status_now != 16'd0) ? child_status_now : STATUS_INTERNAL;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            beat_index_q <= 3'd0;
            token_q <= 2'd0;
            block_q <= 9'd0;
            last_block_q <= 9'd0;
            run_tokens_q <= 3'd0;
            source_done_seen_q <= 1'b0;
            fault_latched_q <= 1'b0;
            report_error_q <= 1'b0;
            cross_abort_q <= 1'b0;
            cleanup_abort_issued_q <= 1'b0;
            latched_status_q <= 16'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 16'd0;
        end else begin
            done <= 1'b0;

            if (abort_run) begin
                // External abort is silent but must reset both children and keep
                // ownership until the weighted source drains its scratch read.
                state_q <= ST_CLEANUP;
                beat_index_q <= 3'd0;
                token_q <= 2'd0;
                block_q <= 9'd0;
                source_done_seen_q <= 1'b0;
                fault_latched_q <= 1'b0;
                report_error_q <= 1'b0;
                cross_abort_q <= 1'b1;
                cleanup_abort_issued_q <= 1'b0;
                latched_status_q <= 16'd0;
                error <= 1'b0;
                status <= 16'd0;
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        cross_abort_q <= 1'b0;
                        cleanup_abort_issued_q <= 1'b0;
                        source_done_seen_q <= 1'b0;
                        fault_latched_q <= 1'b0;
                        report_error_q <= 1'b0;
                        if (cfg_fire) begin
                            state_q <= ST_RUN;
                            beat_index_q <= 3'd0;
                            token_q <= 2'd0;
                            block_q <= 9'd0;
                            last_block_q <= cfg_rows[13:5] - 1'b1;
                            run_tokens_q <= cfg_tokens;
                            latched_status_q <= 16'd0;
                            error <= 1'b0;
                            status <= 16'd0;
                        end
                    end

                    ST_RUN, ST_Q8_DRAIN: begin
                        if (child_fault_now) begin
                            state_q <= ST_CLEANUP;
                            fault_latched_q <= 1'b1;
                            report_error_q <= 1'b1;
                            cross_abort_q <= 1'b1;
                            cleanup_abort_issued_q <= 1'b0;
                            latched_status_q <= latched_status_q |
                                                diagnosed_status_now;
                            error <= 1'b1;
                            status <= latched_status_q |
                                      diagnosed_status_now;
                        end else begin
                            if (source_clean_done) begin
                                source_done_seen_q <= 1'b1;
                                state_q <= ST_Q8_DRAIN;
                            end

                            if (output_fire) begin
                                if (beat_index_q == 3'd4) begin
                                    beat_index_q <= 3'd0;
                                    if (!final_block) begin
                                        block_q <= block_q + 1'b1;
                                    end else if (!final_token) begin
                                        block_q <= 9'd0;
                                        token_q <= token_q + 1'b1;
                                    end
                                end else begin
                                    beat_index_q <= beat_index_q + 1'b1;
                                end
                            end

                            if (final_record_fire) begin
                                if (source_done_seen_q || source_clean_done) begin
                                    state_q <= ST_IDLE;
                                    source_done_seen_q <= 1'b0;
                                    done <= 1'b1;
                                    error <= 1'b0;
                                    status <= 16'd0;
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
                        cross_abort_q <= 1'b1;
                        if (!cleanup_abort_issued_q) begin
                            // This edge is the registered abort observed by both
                            // children. Never retire cleanup on the same edge.
                            cleanup_abort_issued_q <= 1'b1;
                        end else if (!weighted_busy) begin
                            state_q <= ST_IDLE;
                            beat_index_q <= 3'd0;
                            token_q <= 2'd0;
                            block_q <= 9'd0;
                            source_done_seen_q <= 1'b0;
                            cross_abort_q <= 1'b0;
                            cleanup_abort_issued_q <= 1'b0;
                            if (report_error_q) begin
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= (latched_status_q != 16'd0) ?
                                          latched_status_q : STATUS_INTERNAL;
                            end else begin
                                error <= 1'b0;
                                status <= 16'd0;
                            end
                            report_error_q <= 1'b0;
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
    // Stable, shim-only observation aliases. They are absent from synthesis and
    // do not enlarge the public RTL interface.
    wire [1:0] debug_state /* verilator public_flat_rd */ = state_q;
    wire [3:0] debug_weighted_state /* verilator public_flat_rd */ =
        u_weighted.state_q;
    wire [3:0] debug_q8_state /* verilator public_flat_rd */ = u_q8.state;
    wire [3:0] debug_q8_quantizer_state /* verilator public_flat_rd */ =
        u_q8.u_quantizer.state;
    wire [2:0] debug_q8_emit_index /* verilator public_flat_rd */ =
        u_q8.emit_index;
    wire debug_weighted_scalar_fire /* verilator public_flat_rd */ =
        weighted_scalar_valid && weighted_scalar_ready;
    wire debug_weighted_done /* verilator public_flat_rd */ = weighted_done;
    wire debug_q8_record_done /* verilator public_flat_rd */ = q8_record_done;
    wire debug_q8_activation_abort /* verilator public_flat_rd */ =
        q8_activation_abort;
    wire [5:0] debug_q8_status /* verilator public_flat_rd */ = q8_status;
`endif

`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_cfg_fire = cfg_fire;
    assign formal_weighted_scalar_fire = weighted_scalar_valid &&
                                          weighted_scalar_ready;
    assign formal_q8_start = q8_start;
    assign formal_q8_abort = child_abort;
    assign formal_q8_state = q8_formal_state;
    assign formal_q8_activation_fault = q8_activation_abort;
    assign formal_q8_record_done = q8_record_done;
    assign formal_q8_scalar_index = q8_formal_scalar_index;
    assign formal_q8_emit_index = q8_formal_emit_index;
    assign formal_q8_staged_valid = q8_formal_staged_valid;
    assign formal_q8_quant_input_fire = q8_formal_quant_input_fire;
    assign formal_q8_quant_output_valid = q8_formal_quant_output_valid;
    assign formal_beat_index = beat_index_q;
    assign formal_token = token_q;
    assign formal_block = block_q;
    assign formal_source_done_seen = source_done_seen_q;
    assign formal_fault_latched = fault_latched_q;
    assign formal_cleanup_abort_issued = cleanup_abort_issued_q;
    assign formal_latched_status = latched_status_q;
`endif
endmodule

`default_nettype wire
