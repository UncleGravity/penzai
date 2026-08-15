// Scratch-backed weighted RMSNorm scalar source.
//
// Gamma is sealed in four 512x64 BRAM banks before a run starts. A run first
// buffers one inverse-RMS scalar per token, then replays token-major R scratch
// groups. One exact FP32 multiplier is time-multiplexed in the PS-compatible
// order RNE(RNE(x * inv_rms) * gamma). Only healthy scalars are published;
// every output remains tentative until the surrounding section closes cleanly.

`default_nettype none

module section_rmsnorm_weighted_source #(
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
    output reg           gamma_done,
    output reg           gamma_error,
    // bit 0 BAD_CFG, bit 1 FRAME, bit 2 NONFINITE, bit 3 INTERNAL.
    output reg  [3:0]    gamma_status,
    output wire          gamma_valid,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 BAD_CFG, bit 1 GAMMA, bit 2 INV_FRAME, bit 3 SCRATCH,
    // bits 5:4 MUL1 raw status, bits 7:6 MUL2 raw status, bit 8 INTERNAL.
    output reg  [8:0]    status,

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

    output wire          scalar_valid,
    input  wire          scalar_ready,
    output wire [31:0]   scalar_data,
    output wire          scalar_last,
    output wire [1:0]    scalar_status
`ifdef FORMAL
    , output wire [3:0]  formal_state
    , output wire [13:0] formal_gamma_rows
    , output wire [2:0]  formal_scale_count
    , output wire        formal_read_owned
    , output wire        formal_rsp_buffered
    , output wire [1:0]  formal_run_token
    , output wire [10:0] formal_run_group
    , output wire [2:0]  formal_run_lane
    , output wire [1:0]  formal_mul_phase
    , output wire        formal_mul_s_fire
    , output wire [31:0] formal_mul_s_a
    , output wire [31:0] formal_mul_s_b
    , output wire        formal_mul_result_fire
    , output wire [1:0]  formal_mul_result_status
    , output wire [31:0] formal_mul_result_data
`endif
);
    localparam [3:0] GAMMA_BAD_CFG   = 4'b0001;
    localparam [3:0] GAMMA_FRAME     = 4'b0010;
    localparam [3:0] GAMMA_NONFINITE = 4'b0100;
    localparam [3:0] GAMMA_INTERNAL  = 4'b1000;

    localparam [8:0] STATUS_BAD_CFG   = 9'h001;
    localparam [8:0] STATUS_GAMMA     = 9'h002;
    localparam [8:0] STATUS_INV_FRAME = 9'h004;
    localparam [8:0] STATUS_SCRATCH   = 9'h008;
    localparam [8:0] STATUS_INTERNAL  = 9'h100;
    localparam [13:0] HARD_MIN_ROWS  = 14'd32;
    localparam [13:0] HARD_MAX_ROWS  = 14'd4096;

    localparam [3:0] ST_IDLE      = 4'd0;
    localparam [3:0] ST_GAMMA     = 4'd1;
    localparam [3:0] ST_INV       = 4'd2;
    localparam [3:0] ST_REQ       = 4'd3;
    localparam [3:0] ST_WAIT_RSP  = 4'd4;
    localparam [3:0] ST_MUL1_REQ  = 4'd5;
    localparam [3:0] ST_MUL1_WAIT = 4'd6;
    localparam [3:0] ST_MUL2_REQ  = 4'd7;
    localparam [3:0] ST_MUL2_WAIT = 4'd8;
    localparam [3:0] ST_CLEANUP   = 4'd9;

    reg [3:0] state_q;

    reg        gamma_valid_q;
    reg [13:0] gamma_rows_q;
    reg [10:0] gamma_word_q;

    reg [13:0] run_rows_q;
    reg [2:0]  run_tokens_q;
    reg [2:0]  scale_count_q;
    reg [31:0] inv_table_q [0:3];
    reg [1:0]  run_token_q;
    reg [10:0] run_group_q;
    reg [2:0]  run_lane_q;

    reg         read_owned_q;
    reg [255:0] rsp_group_q;
    reg [255:0] gamma_group_q;
    reg [31:0]  mul1_result_q;
    reg         cleanup_report_error_q;
    reg         mul_abort_q;

    (* ram_style = "block" *) reg [63:0] gamma_bank0 [0:511];
    (* ram_style = "block" *) reg [63:0] gamma_bank1 [0:511];
    (* ram_style = "block" *) reg [63:0] gamma_bank2 [0:511];
    (* ram_style = "block" *) reg [63:0] gamma_bank3 [0:511];

    wire gamma_rows_power_two = (gamma_cfg_rows != 14'd0) &&
        ((gamma_cfg_rows & (gamma_cfg_rows - 1'b1)) == 14'd0);
    wire gamma_shape_ok = (gamma_cfg_rows >= MIN_ROWS) &&
        (gamma_cfg_rows >= HARD_MIN_ROWS) &&
        (gamma_cfg_rows <= MAX_ROWS) &&
        (gamma_cfg_rows <= HARD_MAX_ROWS) && gamma_rows_power_two;
    wire run_rows_power_two = (cfg_rows != 14'd0) &&
        ((cfg_rows & (cfg_rows - 1'b1)) == 14'd0);
    wire run_shape_ok = (cfg_rows >= MIN_ROWS) &&
        (cfg_rows >= HARD_MIN_ROWS) && (cfg_rows <= MAX_ROWS) &&
        (cfg_rows <= HARD_MAX_ROWS) && run_rows_power_two &&
        (cfg_tokens != 3'd0) && (cfg_tokens <= 3'd4);

    assign gamma_cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    // Gamma replacement has priority if both independent request channels are
    // asserted together, so exactly one public handshake can occur.
    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE) &&
                       !gamma_cfg_valid;
    wire gamma_cfg_fire = gamma_cfg_valid && gamma_cfg_ready;
    wire cfg_fire = cfg_valid && cfg_ready;

    assign gamma_tready = rst_n && !abort_run && (state_q == ST_GAMMA);
    wire gamma_fire = gamma_tvalid && gamma_tready;
    wire [11:0] gamma_last_word = gamma_rows_q[12:1] - 12'd1;
    wire gamma_expected_last = {1'b0, gamma_word_q} == gamma_last_word;
    wire gamma_frame_ok = (gamma_tkeep == 8'hff) &&
                          (gamma_tlast == gamma_expected_last);
    wire gamma_finite = (gamma_tdata[30:23] != 8'hff) &&
                        (gamma_tdata[62:55] != 8'hff);
    wire gamma_input_ok = gamma_frame_ok && gamma_finite;

    assign gamma_busy = state_q == ST_GAMMA;
    assign gamma_valid = gamma_valid_q;
    assign busy = (state_q >= ST_INV) && (state_q <= ST_CLEANUP);

    assign inv_ready = rst_n && !abort_run && (state_q == ST_INV);
    wire inv_fire = inv_valid && inv_ready;
    wire inv_expected_final = ({1'b0, scale_count_q[1:0]} + 3'd1) ==
                              run_tokens_q;
    wire inv_value_ok = !inv_rms[31] && (inv_rms[30:23] != 8'd0) &&
                        (inv_rms[30:23] != 8'hff);
    wire inv_frame_ok = (scale_count_q < run_tokens_q) &&
                        (inv_token == scale_count_q[1:0]) &&
                        (inv_final == inv_expected_final) && inv_value_ok;

    assign rd_req_valid = rst_n && !abort_run && (state_q == ST_REQ);
    assign rd_req_role = 2'd0;
    assign rd_req_token = {1'b0, run_token_q};
    assign rd_req_group = run_group_q;
    wire rd_req_fire = rd_req_valid && rd_req_ready;

    assign rd_rsp_ready = rst_n && read_owned_q &&
        ((state_q == ST_WAIT_RSP) || (state_q == ST_CLEANUP) || abort_run);
    wire rd_rsp_fire = rd_rsp_valid && rd_rsp_ready;

    function automatic [31:0] lane32(
        input [255:0] value,
        input [2:0] lane
    );
        begin
            case (lane)
                3'd0: lane32 = value[31:0];
                3'd1: lane32 = value[63:32];
                3'd2: lane32 = value[95:64];
                3'd3: lane32 = value[127:96];
                3'd4: lane32 = value[159:128];
                3'd5: lane32 = value[191:160];
                3'd6: lane32 = value[223:192];
                default: lane32 = value[255:224];
            endcase
        end
    endfunction

    wire [31:0] residual_lane = lane32(rsp_group_q, run_lane_q);
    wire [31:0] gamma_lane = lane32(gamma_group_q, run_lane_q);

    wire mul_busy;
    wire mul_s_ready;
    wire mul_s_valid = !abort_run &&
        ((state_q == ST_MUL1_REQ) || (state_q == ST_MUL2_REQ));
    wire [31:0] mul_s_a = state_q == ST_MUL1_REQ ?
                          residual_lane : mul1_result_q;
    wire [31:0] mul_s_b = state_q == ST_MUL1_REQ ?
                          inv_table_q[run_token_q] : gamma_lane;
    wire mul_s_fire = mul_s_valid && mul_s_ready;
    wire mul_result_valid;
    wire [31:0] mul_result_data;
    wire [1:0] mul_result_status;
    // A healthy second result is the held scalar output itself. A diagnosed
    // result is consumed immediately and never becomes public data.
    wire mul_result_ready = !abort_run &&
        ((state_q == ST_MUL1_WAIT) ||
         ((state_q == ST_MUL2_WAIT) &&
          ((mul_result_status != 2'd0) || scalar_ready)));
    wire mul_result_fire = mul_result_valid && mul_result_ready;
    wire mul_abort = abort_run || mul_abort_q;

    section_rmsnorm_mul_rne u_mul (
        .clk(clk), .rst_n(rst_n), .abort_run(mul_abort),
        .busy(mul_busy), .s_valid(mul_s_valid), .s_ready(mul_s_ready),
        .s_a(mul_s_a), .s_b(mul_s_b),
        .result_valid(mul_result_valid),
        .result_ready(mul_result_ready),
        .result_data(mul_result_data),
        .result_status(mul_result_status)
    );

    assign scalar_valid = rst_n && !abort_run &&
                          (state_q == ST_MUL2_WAIT) && mul_result_valid &&
                          (mul_result_status == 2'd0);
    assign scalar_data = mul_result_data;
    assign scalar_last = (run_group_q[1:0] == 2'd3) &&
                         (run_lane_q == 3'd7);
    assign scalar_status = 2'd0;
    wire scalar_fire = scalar_valid && scalar_ready;

    wire [13:0] run_group_count_ext = run_rows_q >> 3;
    wire [10:0] run_last_group = run_group_count_ext[10:0] - 1'b1;
    wire run_final_group = run_group_q == run_last_group;
    wire run_final_token = ({1'b0, run_token_q} + 3'd1) == run_tokens_q;

    task automatic fail_run(input [8:0] failure_status);
        begin
            error <= 1'b1;
            status <= status | failure_status;
            gamma_valid_q <= 1'b0;
            cleanup_report_error_q <= 1'b1;
            mul_abort_q <= 1'b1;
            state_q <= ST_CLEANUP;
        end
    endtask

    integer table_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            gamma_valid_q <= 1'b0;
            gamma_rows_q <= 14'd0;
            gamma_word_q <= 11'd0;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            scale_count_q <= 3'd0;
            run_token_q <= 2'd0;
            run_group_q <= 11'd0;
            run_lane_q <= 3'd0;
            read_owned_q <= 1'b0;
            rsp_group_q <= 256'd0;
            gamma_group_q <= 256'd0;
            mul1_result_q <= 32'd0;
            cleanup_report_error_q <= 1'b0;
            mul_abort_q <= 1'b0;
            gamma_done <= 1'b0;
            gamma_error <= 1'b0;
            gamma_status <= 4'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 9'd0;
            for (table_index = 0; table_index < 4;
                 table_index = table_index + 1)
                inv_table_q[table_index] <= 32'd0;
        end else if (abort_run) begin
            gamma_valid_q <= 1'b0;
            gamma_done <= 1'b0;
            gamma_error <= 1'b0;
            gamma_status <= 4'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 9'd0;
            scale_count_q <= 3'd0;
            run_token_q <= 2'd0;
            run_group_q <= 11'd0;
            run_lane_q <= 3'd0;
            cleanup_report_error_q <= 1'b0;
            mul_abort_q <= 1'b0;
            if (read_owned_q && !rd_rsp_fire) begin
                state_q <= ST_CLEANUP;
            end else if (mul_busy) begin
                read_owned_q <= 1'b0;
                state_q <= ST_CLEANUP;
            end else begin
                read_owned_q <= 1'b0;
                state_q <= ST_IDLE;
            end
            if (rd_rsp_fire)
                read_owned_q <= 1'b0;
        end else begin
            gamma_done <= 1'b0;
            done <= 1'b0;
            mul_abort_q <= 1'b0;

            case (state_q)
                ST_IDLE: begin
                    read_owned_q <= 1'b0;
                    cleanup_report_error_q <= 1'b0;
                    if (gamma_cfg_fire) begin
                        gamma_valid_q <= 1'b0;
                        gamma_error <= 1'b0;
                        gamma_status <= 4'd0;
                        gamma_word_q <= 11'd0;
                        if (!gamma_shape_ok) begin
                            gamma_done <= 1'b1;
                            gamma_error <= 1'b1;
                            gamma_status <= GAMMA_BAD_CFG;
                        end else begin
                            gamma_rows_q <= gamma_cfg_rows;
                            state_q <= ST_GAMMA;
                        end
                    end else if (cfg_fire) begin
                        error <= 1'b0;
                        status <= 9'd0;
                        scale_count_q <= 3'd0;
                        run_token_q <= 2'd0;
                        run_group_q <= 11'd0;
                        run_lane_q <= 3'd0;
                        if (!run_shape_ok) begin
                            gamma_valid_q <= 1'b0;
                            done <= 1'b1;
                            error <= 1'b1;
                            status <= STATUS_BAD_CFG;
                        end else if (!gamma_valid_q ||
                                     (gamma_rows_q != cfg_rows)) begin
                            gamma_valid_q <= 1'b0;
                            done <= 1'b1;
                            error <= 1'b1;
                            status <= STATUS_GAMMA;
                        end else begin
                            run_rows_q <= cfg_rows;
                            run_tokens_q <= cfg_tokens;
                            state_q <= ST_INV;
                        end
                    end
                end

                ST_GAMMA: if (gamma_fire) begin
                    if (!gamma_input_ok) begin
                        gamma_valid_q <= 1'b0;
                        gamma_done <= 1'b1;
                        gamma_error <= 1'b1;
                        gamma_status <= (!gamma_frame_ok ? GAMMA_FRAME : 4'd0) |
                                        (!gamma_finite ? GAMMA_NONFINITE : 4'd0);
                        state_q <= ST_IDLE;
                    end else begin
                        case (gamma_word_q[1:0])
                            2'd0: gamma_bank0[gamma_word_q[10:2]] <= gamma_tdata;
                            2'd1: gamma_bank1[gamma_word_q[10:2]] <= gamma_tdata;
                            2'd2: gamma_bank2[gamma_word_q[10:2]] <= gamma_tdata;
                            default:
                                gamma_bank3[gamma_word_q[10:2]] <= gamma_tdata;
                        endcase
                        if (gamma_expected_last) begin
                            gamma_valid_q <= 1'b1;
                            gamma_done <= 1'b1;
                            gamma_error <= 1'b0;
                            gamma_status <= 4'd0;
                            state_q <= ST_IDLE;
                        end else begin
                            gamma_word_q <= gamma_word_q + 1'b1;
                        end
                    end
                end

                ST_INV: if (inv_fire) begin
                    if (!inv_frame_ok) begin
                        fail_run(STATUS_INV_FRAME);
                    end else begin
                        inv_table_q[inv_token] <= inv_rms;
                        scale_count_q <= scale_count_q + 1'b1;
                        if (inv_expected_final) begin
                            run_token_q <= 2'd0;
                            run_group_q <= 11'd0;
                            run_lane_q <= 3'd0;
                            state_q <= ST_REQ;
                        end
                    end
                end

                ST_REQ: if (rd_req_fire) begin
                    read_owned_q <= 1'b1;
                    gamma_group_q[63:0] <= gamma_bank0[run_group_q[8:0]];
                    gamma_group_q[127:64] <= gamma_bank1[run_group_q[8:0]];
                    gamma_group_q[191:128] <= gamma_bank2[run_group_q[8:0]];
                    gamma_group_q[255:192] <= gamma_bank3[run_group_q[8:0]];
                    state_q <= ST_WAIT_RSP;
                end

                ST_WAIT_RSP: if (rd_rsp_fire) begin
                    read_owned_q <= 1'b0;
                    if (rd_rsp_error) begin
                        fail_run(STATUS_SCRATCH);
                    end else begin
                        rsp_group_q <= rd_rsp_data;
                        run_lane_q <= 3'd0;
                        state_q <= ST_MUL1_REQ;
                    end
                end

                ST_MUL1_REQ: if (mul_s_fire)
                    state_q <= ST_MUL1_WAIT;

                ST_MUL1_WAIT: if (mul_result_fire) begin
                    if (mul_result_status != 2'd0) begin
                        fail_run({3'd0, mul_result_status, 4'd0});
                    end else begin
                        mul1_result_q <= mul_result_data;
                        state_q <= ST_MUL2_REQ;
                    end
                end

                ST_MUL2_REQ: if (mul_s_fire)
                    state_q <= ST_MUL2_WAIT;

                ST_MUL2_WAIT: if (mul_result_fire) begin
                    if (mul_result_status != 2'd0) begin
                        fail_run({1'd0, mul_result_status, 6'd0});
                    end else if (run_lane_q != 3'd7) begin
                        run_lane_q <= run_lane_q + 1'b1;
                        state_q <= ST_MUL1_REQ;
                    end else if (!run_final_group) begin
                        run_lane_q <= 3'd0;
                        run_group_q <= run_group_q + 1'b1;
                        state_q <= ST_REQ;
                    end else if (!run_final_token) begin
                        run_lane_q <= 3'd0;
                        run_group_q <= 11'd0;
                        run_token_q <= run_token_q + 1'b1;
                        state_q <= ST_REQ;
                    end else begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                        error <= 1'b0;
                        status <= 9'd0;
                    end
                end

                ST_CLEANUP: begin
                    if (rd_rsp_fire)
                        read_owned_q <= 1'b0;
                    if ((!read_owned_q || rd_rsp_fire) && !mul_busy) begin
                        state_q <= ST_IDLE;
                        read_owned_q <= 1'b0;
                        scale_count_q <= 3'd0;
                        run_token_q <= 2'd0;
                        run_group_q <= 11'd0;
                        run_lane_q <= 3'd0;
                        if (cleanup_report_error_q)
                            done <= 1'b1;
                        else begin
                            error <= 1'b0;
                            status <= 9'd0;
                        end
                        cleanup_report_error_q <= 1'b0;
                    end
                end

                default: begin
                    gamma_valid_q <= 1'b0;
                    fail_run(STATUS_INTERNAL);
                end
            endcase
        end
    end

`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_gamma_rows = gamma_rows_q;
    assign formal_scale_count = scale_count_q;
    assign formal_read_owned = read_owned_q;
    assign formal_rsp_buffered = (state_q >= ST_MUL1_REQ) &&
                                 (state_q <= ST_MUL2_WAIT);
    assign formal_run_token = run_token_q;
    assign formal_run_group = run_group_q;
    assign formal_run_lane = run_lane_q;
    assign formal_mul_phase = ((state_q == ST_MUL1_REQ) ||
                               (state_q == ST_MUL1_WAIT)) ? 2'd1 :
                              (((state_q == ST_MUL2_REQ) ||
                                (state_q == ST_MUL2_WAIT)) ? 2'd2 : 2'd0);
    assign formal_mul_s_fire = mul_s_fire;
    assign formal_mul_s_a = mul_s_a;
    assign formal_mul_s_b = mul_s_b;
    assign formal_mul_result_fire = mul_result_fire;
    assign formal_mul_result_status = mul_result_status;
    assign formal_mul_result_data = mul_result_data;

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(done && busy));
            assert(!(gamma_done && gamma_busy));
            assert(!(scalar_valid && error));
            assert(!(read_owned_q && (state_q != ST_WAIT_RSP) &&
                     (state_q != ST_CLEANUP)));
            assert(scale_count_q <= run_tokens_q || state_q == ST_IDLE ||
                   state_q == ST_GAMMA);
            if (scalar_valid) begin
                assert(scalar_status == 0);
                assert(run_token_q < run_tokens_q);
                assert(run_group_q <= run_last_group);
            end
            if (!abort_run && f_past_valid &&
                $past(rst_n && !abort_run && scalar_valid && !scalar_ready)) begin
                assert(scalar_valid);
                assert(scalar_data == $past(scalar_data));
                assert(scalar_last == $past(scalar_last));
                assert(scalar_status == $past(scalar_status));
            end
        end
    end
`endif
endmodule

`default_nettype wire
