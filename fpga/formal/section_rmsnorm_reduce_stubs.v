// Bounded child contracts for the composed RMSNorm reduction proof. The
// production leaves have separate arithmetic and protocol proofs; these stubs
// retain their observable lifecycle, tentative stream, cross-abort, and
// untagged scratch-response ownership behavior.

`default_nettype none

module section_rmsnorm_frontend (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    input  wire          cfg_resident,
    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    output reg  [6:0]    status,
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
    output wire [7:0]    result_max_exp,
    output wire [47:0]   result_sum_sq,
    output wire [13:0]   result_rows,
    output wire          result_final
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_REQ    = 3'd1;
    localparam [2:0] ST_WAIT   = 3'd2;
    localparam [2:0] ST_RESULT = 3'd3;
    localparam [2:0] ST_DRAIN  = 3'd4;
    localparam [6:0] STATUS_SCRATCH = 7'h10;

    reg [2:0] state_q;
    reg busy_q;
    reg owner_q;
    reg [13:0] rows_q;
    reg [2:0] tokens_q;
    reg [1:0] token_q;

    assign cfg_ready = rst_n && !abort_run && !busy_q;
    assign busy = busy_q;
    assign s_axis_tready = 1'b0;
    assign r_wr_valid = 1'b0;
    assign r_wr_bank = 2'd0;
    assign r_wr_address = 14'd0;
    assign r_wr_data = 64'd0;

    assign rd_req_valid = rst_n && !abort_run && busy_q &&
                          (state_q == ST_REQ);
    assign rd_req_token = {1'b0, token_q};
    assign rd_req_group = 11'd0;
    assign rd_rsp_ready = rst_n && busy_q && owner_q &&
                          ((state_q == ST_WAIT) ||
                           (state_q == ST_DRAIN));

    assign result_valid = rst_n && !abort_run && busy_q &&
                          (state_q == ST_RESULT);
    assign result_token = token_q;
    assign result_max_exp = 8'd127 + {6'd0, token_q};
    assign result_sum_sq = 48'h0020_0000_0000;
    assign result_rows = rows_q;
    assign result_final = ({1'b0, token_q} + 3'd1) == tokens_q;

    wire request_fire = rd_req_valid && rd_req_ready;
    wire response_fire = rd_rsp_valid && rd_rsp_ready;
    wire result_fire = result_valid && result_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            busy_q <= 1'b0;
            owner_q <= 1'b0;
            rows_q <= 14'd0;
            tokens_q <= 3'd0;
            token_q <= 2'd0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
        end else if (abort_run) begin
            done <= 1'b0;
            error <= 1'b0;
            status <= 7'd0;
            if (busy_q && owner_q && !response_fire) begin
                state_q <= ST_DRAIN;
            end else begin
                state_q <= ST_IDLE;
                busy_q <= 1'b0;
                owner_q <= 1'b0;
                token_q <= 2'd0;
            end
        end else begin
            done <= 1'b0;
            error <= 1'b0;
            case (state_q)
                ST_IDLE: if (cfg_valid && cfg_ready) begin
                    state_q <= ST_REQ;
                    busy_q <= 1'b1;
                    owner_q <= 1'b0;
                    rows_q <= cfg_rows;
                    tokens_q <= cfg_tokens;
                    token_q <= 2'd0;
                    status <= 7'd0;
                end

                ST_REQ: if (request_fire) begin
                    state_q <= ST_WAIT;
                    owner_q <= 1'b1;
                end

                ST_WAIT: if (response_fire) begin
                    owner_q <= 1'b0;
                    if (rd_rsp_error) begin
                        state_q <= ST_IDLE;
                        busy_q <= 1'b0;
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_SCRATCH;
                    end else begin
                        state_q <= ST_RESULT;
                    end
                end

                ST_RESULT: if (result_fire) begin
                    if (result_final) begin
                        state_q <= ST_IDLE;
                        busy_q <= 1'b0;
                        done <= 1'b1;
                        status <= 7'd0;
                    end else begin
                        token_q <= token_q + 1'b1;
                        state_q <= ST_REQ;
                    end
                end

                ST_DRAIN: if (response_fire) begin
                    state_q <= ST_IDLE;
                    busy_q <= 1'b0;
                    owner_q <= 1'b0;
                    token_q <= 2'd0;
                end

                default: begin
                    state_q <= ST_IDLE;
                    busy_q <= 1'b0;
                    owner_q <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                    status <= 7'h40;
                end
            endcase
        end
    end

`ifdef FORMAL
    always @(posedge clk) if (rst_n) begin
        assert(owner_q == ((state_q == ST_WAIT) ||
                           (state_q == ST_DRAIN)));
        assert(!(done && busy_q));
        if (state_q == ST_DRAIN) begin
            assert(busy_q);
            assert(rd_rsp_ready);
            assert(!result_valid);
        end
    end
`endif

    wire _unused = &{1'b0, cfg_resident, s_axis_tdata, s_axis_tkeep,
                     s_axis_tvalid,
                     s_axis_tlast, r_wr_ready, r_wr_error, rd_rsp_data};
endmodule

module section_rmsnorm_inv (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,
    input  wire [31:0]   cfg_eps,
    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    output reg  [3:0]    status,
    input  wire          s_valid,
    output wire          s_ready,
    input  wire [1:0]    s_token,
    input  wire [7:0]    s_max_exp,
    input  wire [47:0]   s_sum_sq,
    input  wire [13:0]   s_rows,
    input  wire          s_final,
    output wire          result_valid,
    input  wire          result_ready,
    output wire [1:0]    result_token,
    output wire [31:0]   result_inv_rms,
    output wire          result_final
);
    localparam [1:0] ST_IDLE   = 2'd0;
    localparam [1:0] ST_INPUT  = 2'd1;
    localparam [1:0] ST_WAIT   = 2'd2;
    localparam [1:0] ST_RESULT = 2'd3;
    localparam [31:0] FAULT_EPS = 32'h3586_37be;
    localparam [3:0] STATUS_ARITHMETIC = 4'h4;

    reg [1:0] state_q;
    reg busy_q;
    reg [2:0] tokens_q;
    reg [1:0] token_q;
    reg final_q;
    reg [1:0] wait_q;
    reg fault_mode_q;

    assign cfg_ready = rst_n && !abort_run && !busy_q;
    assign busy = busy_q;
    assign s_ready = rst_n && !abort_run && busy_q &&
                     (state_q == ST_INPUT);
    assign result_valid = rst_n && !abort_run && busy_q &&
                          (state_q == ST_RESULT);
    assign result_token = token_q;
    assign result_inv_rms = 32'h3f80_0000 + {30'd0, token_q};
    assign result_final = final_q;

    wire input_fire = s_valid && s_ready;
    wire result_fire = result_valid && result_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            busy_q <= 1'b0;
            tokens_q <= 3'd0;
            token_q <= 2'd0;
            final_q <= 1'b0;
            wait_q <= 2'd0;
            fault_mode_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            busy_q <= 1'b0;
            token_q <= 2'd0;
            final_q <= 1'b0;
            wait_q <= 2'd0;
            fault_mode_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else begin
            done <= 1'b0;
            error <= 1'b0;
            case (state_q)
                ST_IDLE: if (cfg_valid && cfg_ready) begin
                    state_q <= ST_INPUT;
                    busy_q <= 1'b1;
                    tokens_q <= cfg_tokens;
                    token_q <= 2'd0;
                    final_q <= 1'b0;
                    wait_q <= 2'd0;
                    fault_mode_q <= cfg_eps == FAULT_EPS;
                    status <= 4'd0;
                end

                ST_INPUT: if (input_fire) begin
                    state_q <= ST_WAIT;
                    token_q <= s_token;
                    final_q <= s_final;
                    wait_q <= 2'd2;
                end

                ST_WAIT: if (wait_q > 1) begin
                    wait_q <= wait_q - 1'b1;
                end else if (fault_mode_q && (token_q == 0)) begin
                    state_q <= ST_IDLE;
                    busy_q <= 1'b0;
                    wait_q <= 2'd0;
                    done <= 1'b1;
                    error <= 1'b1;
                    status <= STATUS_ARITHMETIC;
                end else begin
                    state_q <= ST_RESULT;
                    wait_q <= 2'd0;
                end

                ST_RESULT: if (result_fire) begin
                    if (final_q) begin
                        state_q <= ST_IDLE;
                        busy_q <= 1'b0;
                        done <= 1'b1;
                        status <= 4'd0;
                    end else begin
                        state_q <= ST_INPUT;
                    end
                end

                default: begin
                    state_q <= ST_IDLE;
                    busy_q <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                    status <= 4'h8;
                end
            endcase
        end
    end

`ifdef FORMAL
    always @(posedge clk) if (rst_n) begin
        assert(!(done && busy_q));
        if (result_valid)
            assert(result_final == (({1'b0, result_token} + 3'd1) ==
                                    tokens_q));
    end
`endif

    wire _unused = &{1'b0, cfg_rows, s_max_exp, s_sum_sq, s_rows};
endmodule

`default_nettype wire
