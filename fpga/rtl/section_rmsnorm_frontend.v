// Section-local RMSNorm reduction frontend.
//
// A token-major F32 stream is written into R scratch while its maximum exponent
// is scanned. Once every token exponent is committed, this controller replays R
// groups through the fixed sum-of-squares leaf and exports one tentative record
// per token. Subnormals are deliberately fail-closed at this boundary: the leaf
// warning is not architecturally publishable by the later v17 controller.

`default_nettype none

module section_rmsnorm_frontend (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,

    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 BAD_CFG, 1 LOADER, 2 MAXEXP, 3 SUMSQ,
    // bit 4 SCRATCH, 5 SUBNORMAL, 6 INTERNAL.
    output reg  [6:0]    status,

    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    // Direct token-major R writes.
    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    input  wire          r_wr_error,
    output wire [1:0]    r_wr_bank,
    output wire [13:0]   r_wr_address,
    output wire [63:0]   r_wr_data,

    // One untagged R scratch request may be retained at a time.
    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [2:0]    rd_req_token,
    output wire [10:0]   rd_req_group,
    input  wire          rd_rsp_valid,
    output wire          rd_rsp_ready,
    input  wire [255:0]  rd_rsp_data,
    input  wire          rd_rsp_error,

    output wire          result_valid,
    input  wire          result_ready,
    output wire [1:0]    result_token,
    output wire [7:0]    result_max_exp,
    output wire [47:0]   result_sum_sq,
    output wire [13:0]   result_rows,
    output wire          result_final
);
    localparam [6:0] STATUS_BAD_CFG   = 7'b0000001;
    localparam [6:0] STATUS_LOADER    = 7'b0000010;
    localparam [6:0] STATUS_MAXEXP    = 7'b0000100;
    localparam [6:0] STATUS_SUMSQ     = 7'b0001000;
    localparam [6:0] STATUS_SCRATCH   = 7'b0010000;
    localparam [6:0] STATUS_SUBNORMAL = 7'b0100000;
    localparam [6:0] STATUS_INTERNAL  = 7'b1000000;

    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_LOAD_START = 3'd1;
    localparam [2:0] ST_LOAD       = 3'd2;
    localparam [2:0] ST_SUM_START  = 3'd3;
    localparam [2:0] ST_SUM        = 3'd4;
    localparam [2:0] ST_CLEANUP    = 3'd5;
    localparam [2:0] ST_DRAIN      = 3'd6;

    reg [2:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [9:0] run_groups_q;
    reg [31:0] max_exp_q;
    reg [2:0] max_records_q;
    reg max_subnormal_q;
    reg loader_done_q;
    reg maxexp_done_q;
    reg [1:0] replay_token_q;
    reg [9:0] replay_group_q;
    reg replay_outstanding_q;
    reg replay_complete_q;
    reg [2:0] sum_records_q;
    reg abort_pulse_q;
    reg drain_report_error_q;

    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) &&
                        (cfg_rows[2:0] == 3'b000) &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4);
    wire cfg_accept = cfg_valid && cfg_ready;
    wire pipeline_abort = abort_run || abort_pulse_q;

    assign cfg_ready = rst_n && !abort_run && !abort_pulse_q &&
                       (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;

    // ---- Residual load and exponent scan ----

    wire loader_cfg_ready;
    wire maxexp_cfg_ready;
    wire load_cfg_valid = (state_q == ST_LOAD_START) &&
                          loader_cfg_ready && maxexp_cfg_ready;
    wire load_cfg_fire = load_cfg_valid;
    wire loader_busy;
    wire loader_done;
    wire loader_error;
    wire [3:0] loader_status;
    wire loader_group_valid;
    wire loader_group_ready;
    wire [255:0] loader_group_data;
    wire loader_group_last;

    section_rmsnorm_loader u_loader (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(load_cfg_valid), .cfg_ready(loader_cfg_ready),
        .cfg_rows(run_rows_q), .cfg_tokens(run_tokens_q),
        .abort_run(pipeline_abort),
        .busy(loader_busy), .done(loader_done),
        .error(loader_error), .status(loader_status),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .wr_valid(r_wr_valid), .wr_ready(r_wr_ready), .wr_error(r_wr_error),
        .wr_bank(r_wr_bank), .wr_address(r_wr_address), .wr_data(r_wr_data),
        .group_valid(loader_group_valid), .group_ready(loader_group_ready),
        .group_data(loader_group_data), .group_last(loader_group_last)
    );

    wire maxexp_busy;
    wire maxexp_done;
    wire maxexp_error;
    wire [5:0] maxexp_status;
    wire maxexp_result_valid;
    wire [1:0] maxexp_result_token;
    wire [7:0] maxexp_result_value;
    wire [13:0] maxexp_result_rows;
    wire maxexp_result_subnormal;
    wire maxexp_result_final;
    wire maxexp_result_ready = (state_q == ST_LOAD) && !pipeline_abort;
    wire maxexp_result_fire = maxexp_result_valid && maxexp_result_ready;
    wire maxexp_result_bad = maxexp_result_fire &&
                             ((maxexp_result_token != max_records_q[1:0]) ||
                              (maxexp_result_rows != run_rows_q) ||
                              (maxexp_result_final !=
                               (max_records_q + 1'b1 == run_tokens_q)));

    section_rmsnorm_maxexp u_maxexp (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(load_cfg_valid), .cfg_ready(maxexp_cfg_ready),
        .cfg_rows(run_rows_q), .cfg_tokens(run_tokens_q),
        .abort_run(pipeline_abort),
        .busy(maxexp_busy), .done(maxexp_done),
        .error(maxexp_error), .status(maxexp_status),
        .s_group_data(loader_group_data), .s_group_error(1'b0),
        .s_group_valid(loader_group_valid),
        .s_group_ready(loader_group_ready),
        .s_group_last(loader_group_last),
        .result_valid(maxexp_result_valid),
        .result_ready(maxexp_result_ready),
        .result_token(maxexp_result_token),
        .result_max_exp(maxexp_result_value),
        .result_rows(maxexp_result_rows),
        .result_subnormal_warning(maxexp_result_subnormal),
        .result_final(maxexp_result_final)
    );

    // ---- Token-major scratch replay and fixed sumsq ----

    wire sum_cfg_ready;
    wire sum_cfg_valid = (state_q == ST_SUM_START) && sum_cfg_ready;
    wire sum_cfg_fire = sum_cfg_valid;
    wire sum_busy;
    wire sum_done;
    wire sum_error;
    wire [6:0] sum_status;
    wire sum_group_ready;
    wire sum_result_valid;
    wire [1:0] sum_result_token;
    wire [7:0] sum_result_max_exp;
    wire [47:0] sum_result_sum_sq;
    wire [13:0] sum_result_rows;
    wire sum_result_subnormal;
    wire sum_result_final;
    wire sum_result_ready = (state_q == ST_SUM) && !pipeline_abort &&
                            result_ready;
    wire sum_result_fire = sum_result_valid && sum_result_ready;
    wire sum_result_bad = sum_result_fire &&
                          ((sum_result_token != sum_records_q[1:0]) ||
                           (sum_result_rows != run_rows_q) ||
                           (sum_result_final !=
                            (sum_records_q + 1'b1 == run_tokens_q)) ||
                           (sum_result_max_exp !=
                            max_exp_q[sum_result_token * 8 +: 8]));

    assign rd_req_valid = rst_n && !pipeline_abort && (state_q == ST_SUM) &&
                          !replay_outstanding_q && !replay_complete_q &&
                          sum_group_ready;
    assign rd_req_token = {1'b0, replay_token_q};
    assign rd_req_group = {1'b0, replay_group_q};
    wire rd_req_fire = rd_req_valid && rd_req_ready;

    wire sum_group_valid = rst_n && !pipeline_abort && (state_q == ST_SUM) &&
                           replay_outstanding_q && rd_rsp_valid;
    wire sum_group_last = ({1'b0, replay_token_q} + 3'd1 == run_tokens_q) &&
                          (replay_group_q + 1'b1 == run_groups_q);
    assign rd_rsp_ready = rst_n && replay_outstanding_q &&
                          (abort_run || (state_q == ST_DRAIN) ||
                           ((state_q == ST_SUM) && sum_group_ready));
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;

    section_rmsnorm_sumsq u_sumsq (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(sum_cfg_valid), .cfg_ready(sum_cfg_ready),
        .cfg_rows(run_rows_q), .cfg_tokens(run_tokens_q),
        .cfg_max_exp(max_exp_q), .abort_run(pipeline_abort),
        .busy(sum_busy), .done(sum_done),
        .error(sum_error), .status(sum_status),
        .s_group_data(rd_rsp_data), .s_group_error(rd_rsp_error),
        .s_group_valid(sum_group_valid), .s_group_ready(sum_group_ready),
        .s_group_last(sum_group_last),
        .result_valid(sum_result_valid), .result_ready(sum_result_ready),
        .result_token(sum_result_token),
        .result_max_exp(sum_result_max_exp),
        .result_sum_sq(sum_result_sum_sq),
        .result_rows(sum_result_rows),
        .result_subnormal_warning(sum_result_subnormal),
        .result_final(sum_result_final)
    );

    assign result_valid = rst_n && !pipeline_abort &&
                          (state_q == ST_SUM) && sum_result_valid;
    assign result_token = sum_result_token;
    assign result_max_exp = sum_result_max_exp;
    assign result_sum_sq = sum_result_sum_sq;
    assign result_rows = sum_result_rows;
    assign result_final = sum_result_final;

    task automatic fail_frontend(input [6:0] failure);
        begin
            error <= 1'b1;
            status <= status | failure;
            abort_pulse_q <= 1'b1;
            if (replay_outstanding_q && !rd_rsp_fire) begin
                state_q <= ST_DRAIN;
                drain_report_error_q <= 1'b1;
            end else begin
                state_q <= ST_CLEANUP;
                replay_outstanding_q <= 1'b0;
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_groups_q <= 10'd0;
            max_exp_q <= 32'd0;
            max_records_q <= 3'd0;
            max_subnormal_q <= 1'b0;
            loader_done_q <= 1'b0;
            maxexp_done_q <= 1'b0;
            replay_token_q <= 2'd0;
            replay_group_q <= 10'd0;
            replay_outstanding_q <= 1'b0;
            replay_complete_q <= 1'b0;
            sum_records_q <= 3'd0;
            abort_pulse_q <= 1'b0;
            drain_report_error_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
        end else if (abort_run) begin
            abort_pulse_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
            max_records_q <= 3'd0;
            max_subnormal_q <= 1'b0;
            loader_done_q <= 1'b0;
            maxexp_done_q <= 1'b0;
            replay_complete_q <= 1'b0;
            sum_records_q <= 3'd0;
            if (replay_outstanding_q && !rd_rsp_fire) begin
                state_q <= ST_DRAIN;
                drain_report_error_q <= 1'b0;
            end else begin
                state_q <= ST_IDLE;
                replay_outstanding_q <= 1'b0;
                drain_report_error_q <= 1'b0;
            end
        end else begin
            done <= 1'b0;
            abort_pulse_q <= 1'b0;

            case (state_q)
                ST_IDLE: if (cfg_accept) begin
                    error <= 1'b0;
                    status <= 7'd0;
                    max_exp_q <= 32'd0;
                    max_records_q <= 3'd0;
                    max_subnormal_q <= 1'b0;
                    loader_done_q <= 1'b0;
                    maxexp_done_q <= 1'b0;
                    replay_token_q <= 2'd0;
                    replay_group_q <= 10'd0;
                    replay_outstanding_q <= 1'b0;
                    replay_complete_q <= 1'b0;
                    sum_records_q <= 3'd0;
                    drain_report_error_q <= 1'b0;
                    if (!cfg_shape_ok) begin
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_BAD_CFG;
                    end else begin
                        run_rows_q <= cfg_rows;
                        run_tokens_q <= cfg_tokens;
                        run_groups_q <= cfg_rows[12:3];
                        state_q <= ST_LOAD_START;
                    end
                end

                ST_LOAD_START: if (load_cfg_fire)
                    state_q <= ST_LOAD;

                ST_LOAD: begin
                    if (maxexp_result_fire) begin
                        case (maxexp_result_token)
                            2'd0: max_exp_q[7:0] <= maxexp_result_value;
                            2'd1: max_exp_q[15:8] <= maxexp_result_value;
                            2'd2: max_exp_q[23:16] <= maxexp_result_value;
                            default: max_exp_q[31:24] <= maxexp_result_value;
                        endcase
                        max_records_q <= max_records_q + 1'b1;
                        if (maxexp_result_subnormal)
                            max_subnormal_q <= 1'b1;
                    end
                    if (loader_done)
                        loader_done_q <= 1'b1;
                    if (maxexp_done)
                        maxexp_done_q <= 1'b1;

                    if (loader_error) begin
                        fail_frontend(STATUS_LOADER);
                    end else if (maxexp_error) begin
                        fail_frontend(STATUS_MAXEXP |
                                      (maxexp_status[3] ? STATUS_SCRATCH : 7'd0));
                    end else if (maxexp_result_bad) begin
                        fail_frontend(STATUS_INTERNAL);
                    end else if ((loader_done_q || loader_done) &&
                                 (maxexp_done_q || maxexp_done)) begin
                        if ((max_records_q != run_tokens_q) ||
                            max_subnormal_q || maxexp_status[5]) begin
                            fail_frontend((max_subnormal_q || maxexp_status[5]) ?
                                          STATUS_SUBNORMAL : STATUS_INTERNAL);
                        end else begin
                            state_q <= ST_SUM_START;
                            replay_token_q <= 2'd0;
                            replay_group_q <= 10'd0;
                            replay_outstanding_q <= 1'b0;
                            replay_complete_q <= 1'b0;
                            sum_records_q <= 3'd0;
                        end
                    end
                end

                ST_SUM_START: if (sum_cfg_fire)
                    state_q <= ST_SUM;

                ST_SUM: begin
                    if (rd_req_fire)
                        replay_outstanding_q <= 1'b1;
                    if (rd_rsp_fire) begin
                        replay_outstanding_q <= 1'b0;
                        if (sum_group_last) begin
                            replay_complete_q <= 1'b1;
                        end else if (replay_group_q + 1'b1 == run_groups_q) begin
                            replay_group_q <= 10'd0;
                            replay_token_q <= replay_token_q + 1'b1;
                        end else begin
                            replay_group_q <= replay_group_q + 1'b1;
                        end
                    end
                    if (sum_result_fire)
                        sum_records_q <= sum_records_q + 1'b1;

                    if (sum_error) begin
                        fail_frontend((sum_status[4] ? STATUS_SCRATCH :
                                       STATUS_SUMSQ) |
                                      (sum_status[6] ? STATUS_SUBNORMAL : 7'd0));
                    end else if (sum_result_bad) begin
                        fail_frontend(STATUS_INTERNAL);
                    end else if (sum_done) begin
                        if (!replay_complete_q || replay_outstanding_q ||
                            (sum_records_q != run_tokens_q) || sum_status[6]) begin
                            fail_frontend(sum_status[6] ? STATUS_SUBNORMAL :
                                          STATUS_INTERNAL);
                        end else begin
                            state_q <= ST_IDLE;
                            done <= 1'b1;
                        end
                    end
                end

                ST_CLEANUP: begin
                    state_q <= ST_IDLE;
                    done <= 1'b1;
                end

                ST_DRAIN: if (rd_rsp_fire) begin
                    replay_outstanding_q <= 1'b0;
                    state_q <= ST_IDLE;
                    if (drain_report_error_q)
                        done <= 1'b1;
                    drain_report_error_q <= 1'b0;
                end

                default: fail_frontend(STATUS_INTERNAL);
            endcase
        end
    end

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(result_valid && error));
            assert(!(rd_req_valid && replay_outstanding_q));
            if (replay_outstanding_q)
                assert(state_q == ST_SUM || state_q == ST_DRAIN);
            if (result_valid) begin
                assert(state_q == ST_SUM);
                assert(result_token < run_tokens_q);
                assert(result_rows == run_rows_q);
            end
        end
        if (f_past_valid && rst_n && !pipeline_abort &&
            $past(rst_n && !pipeline_abort && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_max_exp == $past(result_max_exp));
            assert(result_sum_sq == $past(result_sum_sq));
            assert(result_rows == $past(result_rows));
            assert(result_final == $past(result_final));
        end
        if (f_past_valid && rst_n &&
            $past(rst_n && (state_q == ST_DRAIN) && !rd_rsp_valid))
            assert(state_q == ST_DRAIN);
    end
`endif

    wire _unused = &{1'b0, loader_busy, loader_status, maxexp_busy,
                     sum_busy, sum_result_subnormal};
endmodule

`default_nettype wire
