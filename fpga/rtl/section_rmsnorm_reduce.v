// Composed P3d RMSNorm reduction and inverse-RMS scalar table.
//
// The frontend owns residual loading and scratch replay. Its tentative fixed
// reduction records feed the inverse leaf directly. Inverse records are retained
// locally and become visible only after both child lifecycles complete cleanly.
// A cross-child fault aborts both leaves and keeps this boundary busy until the
// frontend drains any retained untagged scratch response.

`default_nettype none

module section_rmsnorm_reduce (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    input  wire [31:0]   cfg_eps,
    input  wire          cfg_resident,

    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 wrapper BAD_CFG, bits 7:1 raw frontend status,
    // bits 11:8 raw inverse status, bit 12 wrapper INTERNAL.
    output reg  [12:0]   status,

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
    output wire [2:0]    rd_req_token,
    output wire [10:0]   rd_req_group,
    input  wire          rd_rsp_valid,
    output wire          rd_rsp_ready,
    input  wire [255:0]  rd_rsp_data,
    input  wire          rd_rsp_error,

    output wire          result_valid,
    input  wire          result_ready,
    output wire [1:0]    result_token,
    output wire [31:0]   result_inv_rms,
    output wire          result_final
`ifdef FORMAL
    , output wire [1:0]  formal_state,
    output wire [2:0]    formal_bridge_count,
    output wire [2:0]    formal_capture_count,
    output wire          formal_front_busy,
    output wire          formal_inv_busy
`endif
);
    localparam [12:0] STATUS_BAD_CFG  = 13'h0001;
    localparam [12:0] STATUS_INTERNAL = 13'h1000;

    localparam [1:0] ST_IDLE    = 2'd0;
    localparam [1:0] ST_RUN     = 2'd1;
    localparam [1:0] ST_CLEANUP = 2'd2;
    localparam [1:0] ST_EMIT    = 2'd3;

    reg [1:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [2:0] bridge_count_q;
    reg [2:0] capture_count_q;
    reg [2:0] emit_count_q;
    reg front_done_seen_q;
    reg inv_done_seen_q;
    reg abort_pulse_q;
    reg cleanup_report_error_q;
    reg [31:0] scalar_q [0:3];

    wire cfg_rows_power_two = (cfg_rows != 14'd0) &&
                              ((cfg_rows & (cfg_rows - 1'b1)) == 14'd0);
    wire cfg_eps_ok = !cfg_eps[31] &&
                      (cfg_eps[30:23] != 8'd0) &&
                      (cfg_eps[30:23] != 8'hff);
    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) && cfg_rows_power_two &&
                        (cfg_tokens != 3'd0) && (cfg_tokens <= 3'd4) &&
                        cfg_eps_ok;

    wire front_cfg_ready;
    wire inv_cfg_ready;
    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE) &&
                       front_cfg_ready && inv_cfg_ready;
    wire cfg_accept = cfg_valid && cfg_ready;
    wire child_cfg_valid = cfg_accept && cfg_shape_ok;
    wire child_abort = abort_run || abort_pulse_q;

    assign busy = state_q != ST_IDLE;
    assign result_valid = rst_n && !abort_run && (state_q == ST_EMIT);
    assign result_token = emit_count_q[1:0];
    assign result_inv_rms = scalar_q[emit_count_q[1:0]];
    assign result_final = ({1'b0, emit_count_q[1:0]} + 3'd1) ==
                          run_tokens_q;
    wire result_fire = result_valid && result_ready;

    wire front_busy;
    wire front_done;
    wire front_error;
    wire [6:0] front_status;
    wire front_result_valid;
    wire front_result_ready;
    wire [1:0] front_result_token;
    wire [7:0] front_result_max_exp;
    wire [47:0] front_result_sum_sq;
    wire [13:0] front_result_rows;
    wire front_result_final;

    wire inv_busy;
    wire inv_done;
    wire inv_error;
    wire [3:0] inv_status;
    wire inv_input_ready;
    wire inv_result_valid;
    wire inv_result_ready;
    wire [1:0] inv_result_token;
    wire [31:0] inv_result_value;
    wire inv_result_final;

    wire bridge_active = rst_n && !child_abort && (state_q == ST_RUN);
    wire inv_input_valid = bridge_active && front_result_valid;
    assign front_result_ready = bridge_active && inv_input_ready;
    wire bridge_fire = inv_input_valid && inv_input_ready;
    wire bridge_expected_final = ({1'b0, bridge_count_q[1:0]} + 3'd1) ==
                                 run_tokens_q;
    wire bridge_bad = bridge_fire &&
                      ((bridge_count_q >= run_tokens_q) ||
                       (front_result_token != bridge_count_q[1:0]) ||
                       (front_result_rows != run_rows_q) ||
                       (front_result_final != bridge_expected_final));

    assign inv_result_ready = bridge_active &&
                              (capture_count_q < run_tokens_q);
    wire inv_result_fire = inv_result_valid && inv_result_ready;
    wire capture_expected_final =
        ({1'b0, capture_count_q[1:0]} + 3'd1) == run_tokens_q;
    wire capture_bad = inv_result_fire &&
                       ((capture_count_q >= run_tokens_q) ||
                        (inv_result_token != capture_count_q[1:0]) ||
                        (inv_result_final != capture_expected_final));

    section_rmsnorm_frontend u_frontend (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(child_cfg_valid), .cfg_ready(front_cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens),
        .cfg_resident(cfg_resident),
        .abort_run(child_abort), .busy(front_busy), .done(front_done),
        .error(front_error), .status(front_status),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_error(r_wr_error), .r_wr_bank(r_wr_bank),
        .r_wr_address(r_wr_address), .r_wr_data(r_wr_data),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_token(rd_req_token), .rd_req_group(rd_req_group),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error),
        .result_valid(front_result_valid),
        .result_ready(front_result_ready),
        .result_token(front_result_token),
        .result_max_exp(front_result_max_exp),
        .result_sum_sq(front_result_sum_sq),
        .result_rows(front_result_rows),
        .result_final(front_result_final)
    );

    section_rmsnorm_inv u_inverse (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(child_cfg_valid), .cfg_ready(inv_cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens), .cfg_eps(cfg_eps),
        .abort_run(child_abort), .busy(inv_busy), .done(inv_done),
        .error(inv_error), .status(inv_status),
        .s_valid(inv_input_valid), .s_ready(inv_input_ready),
        .s_token(front_result_token),
        .s_max_exp(front_result_max_exp),
        .s_sum_sq(front_result_sum_sq), .s_rows(front_result_rows),
        .s_final(front_result_final),
        .result_valid(inv_result_valid), .result_ready(inv_result_ready),
        .result_token(inv_result_token),
        .result_inv_rms(inv_result_value),
        .result_final(inv_result_final)
    );

    wire [12:0] front_failure = {1'b0, 4'b0000, front_status, 1'b0};
    wire [12:0] inv_failure = {1'b0, inv_status, 7'b0000000, 1'b0};
    wire front_done_success = front_done && !front_error;
    wire inv_done_success = inv_done && !inv_error;

    task automatic fail_composed(input [12:0] failure);
        begin
            error <= 1'b1;
            status <= status | failure;
            abort_pulse_q <= 1'b1;
            cleanup_report_error_q <= 1'b1;
            state_q <= ST_CLEANUP;
            emit_count_q <= 3'd0;
        end
    endtask

    integer scalar_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            bridge_count_q <= 3'd0;
            capture_count_q <= 3'd0;
            emit_count_q <= 3'd0;
            front_done_seen_q <= 1'b0;
            inv_done_seen_q <= 1'b0;
            abort_pulse_q <= 1'b0;
            cleanup_report_error_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 13'd0;
            for (scalar_index = 0; scalar_index < 4;
                 scalar_index = scalar_index + 1)
                scalar_q[scalar_index] <= 32'd0;
        end else if (abort_run) begin
            done <= 1'b0;
            error <= 1'b0;
            status <= 13'd0;
            bridge_count_q <= 3'd0;
            capture_count_q <= 3'd0;
            emit_count_q <= 3'd0;
            front_done_seen_q <= 1'b0;
            inv_done_seen_q <= 1'b0;
            abort_pulse_q <= 1'b0;
            cleanup_report_error_q <= 1'b0;
            if (state_q == ST_IDLE)
                state_q <= ST_IDLE;
            else
                state_q <= ST_CLEANUP;
        end else begin
            done <= 1'b0;
            abort_pulse_q <= 1'b0;

            case (state_q)
                ST_IDLE: if (cfg_accept) begin
                    error <= 1'b0;
                    status <= 13'd0;
                    bridge_count_q <= 3'd0;
                    capture_count_q <= 3'd0;
                    emit_count_q <= 3'd0;
                    front_done_seen_q <= 1'b0;
                    inv_done_seen_q <= 1'b0;
                    cleanup_report_error_q <= 1'b0;
                    for (scalar_index = 0; scalar_index < 4;
                         scalar_index = scalar_index + 1)
                        scalar_q[scalar_index] <= 32'd0;
                    if (!cfg_shape_ok) begin
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_BAD_CFG;
                    end else begin
                        run_rows_q <= cfg_rows;
                        run_tokens_q <= cfg_tokens;
                        state_q <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (bridge_fire)
                        bridge_count_q <= bridge_count_q + 1'b1;
                    if (inv_result_fire) begin
                        scalar_q[inv_result_token] <= inv_result_value;
                        capture_count_q <= capture_count_q + 1'b1;
                    end
                    if (front_done_success)
                        front_done_seen_q <= 1'b1;
                    if (inv_done_success)
                        inv_done_seen_q <= 1'b1;

                    if (front_error || inv_error) begin
                        fail_composed(front_failure | inv_failure |
                            (((front_error && front_status == 0) ||
                              (inv_error && inv_status == 0)) ?
                             STATUS_INTERNAL : 13'd0));
                    end else if (bridge_bad || capture_bad) begin
                        fail_composed(STATUS_INTERNAL);
                    end else if (front_done_success &&
                                 (bridge_count_q != run_tokens_q)) begin
                        fail_composed(STATUS_INTERNAL);
                    end else if (inv_done_success &&
                                 (capture_count_q != run_tokens_q)) begin
                        fail_composed(STATUS_INTERNAL);
                    end else if ((front_done_seen_q || front_done_success) &&
                                 (inv_done_seen_q || inv_done_success)) begin
                        if ((bridge_count_q != run_tokens_q) ||
                            (capture_count_q != run_tokens_q)) begin
                            fail_composed(STATUS_INTERNAL);
                        end else begin
                            state_q <= ST_EMIT;
                            emit_count_q <= 3'd0;
                        end
                    end
                end

                ST_CLEANUP: if (!front_busy && !inv_busy) begin
                    state_q <= ST_IDLE;
                    bridge_count_q <= 3'd0;
                    capture_count_q <= 3'd0;
                    emit_count_q <= 3'd0;
                    front_done_seen_q <= 1'b0;
                    inv_done_seen_q <= 1'b0;
                    if (cleanup_report_error_q)
                        done <= 1'b1;
                    else begin
                        error <= 1'b0;
                        status <= 13'd0;
                    end
                    cleanup_report_error_q <= 1'b0;
                end

                ST_EMIT: if (result_fire) begin
                    if (result_final) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else begin
                        emit_count_q <= emit_count_q + 1'b1;
                    end
                end

                default: fail_composed(STATUS_INTERNAL);
            endcase
        end
    end

`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_bridge_count = bridge_count_q;
    assign formal_capture_count = capture_count_q;
    assign formal_front_busy = front_busy;
    assign formal_inv_busy = inv_busy;

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(done && busy));
            assert(!(result_valid && error));
            assert(bridge_count_q <= run_tokens_q || state_q == ST_IDLE);
            assert(capture_count_q <= run_tokens_q || state_q == ST_IDLE);
            if (result_valid) begin
                assert(state_q == ST_EMIT);
                assert(emit_count_q < run_tokens_q);
                assert(front_done_seen_q);
                assert(inv_done_seen_q);
                assert(status == 0);
            end
            if (state_q == ST_CLEANUP)
                assert(!result_valid);
            assert(child_cfg_valid ==
                   (cfg_valid && cfg_ready && cfg_shape_ok));
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_inv_rms == $past(result_inv_rms));
            assert(result_final == $past(result_final));
        end
    end
`endif

endmodule

`default_nettype wire
