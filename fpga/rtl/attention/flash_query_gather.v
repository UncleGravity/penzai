// Gather the resident four-lane query arena into Flash's 8-wide FP32 Q stream.
//
// The arena returns one scalar dimension for four physical token lanes. Eight
// ordered reads build four 256-bit query beats, which are then emitted token
// inner. A second wave handles tokens 4..7. This exactly matches the
// TILE8_LANE4_Q_ORDER specialization without a transpose RAM.

`default_nettype none

module flash_query_gather (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,

    input  wire         start_valid,
    output wire         start_ready,
    input  wire [5:0]   start_q_head_base,
    input  wire [3:0]   start_token_count,

    output wire         query_rd_req_valid,
    input  wire         query_rd_req_ready,
    output wire         query_rd_req_wave,
    output wire [11:0]  query_rd_req_addr,
    input  wire         query_rd_rsp_valid,
    output wire         query_rd_rsp_ready,
    input  wire [127:0] query_rd_rsp_data,

    output wire [255:0] q_tdata,
    output wire         q_tvalid,
    input  wire         q_tready,

    output wire         busy,
    output wire         done_valid,
    input  wire         done_ready,
    output wire         done_error
);
    localparam [1:0] S_IDLE   = 2'd0;
    localparam [1:0] S_GATHER = 2'd1;
    localparam [1:0] S_EMIT   = 2'd2;
    localparam [1:0] S_DONE   = 2'd3;

    reg [1:0] state_q;
    reg [5:0] q_head_base_q;
    reg [3:0] token_count_q;
    reg [2:0] head_q;
    reg [3:0] beat_q;
    reg wave_q;
    reg [3:0] issue_count_q;
    reg [3:0] capture_count_q;
    reg [1:0] emit_lane_q;
    reg done_error_q;
    reg [255:0] lane_buf [0:3];

    wire [5:0] global_head = q_head_base_q + {3'd0, head_q};
    wire [11:0] scalar_base =
        {global_head[4:0], 7'b0} + {5'd0, beat_q, 3'b000};
    wire [3:0] wave_lane_count = wave_q ? (token_count_q - 4'd4) :
        ((token_count_q > 4'd4) ? 4'd4 : token_count_q);

    assign start_ready = state_q == S_IDLE;
    assign query_rd_req_valid = (state_q == S_GATHER) &&
                                (issue_count_q < 4'd8);
    assign query_rd_req_wave = wave_q;
    assign query_rd_req_addr = scalar_base + {9'd0, issue_count_q[2:0]};
    assign query_rd_rsp_ready = state_q == S_GATHER;

    assign q_tvalid = state_q == S_EMIT;
    assign q_tdata = (emit_lane_q == 2'd0) ? lane_buf[0] :
                     (emit_lane_q == 2'd1) ? lane_buf[1] :
                     (emit_lane_q == 2'd2) ? lane_buf[2] : lane_buf[3];

    assign busy = (state_q == S_GATHER) || (state_q == S_EMIT);
    assign done_valid = state_q == S_DONE;
    assign done_error = done_error_q;

    wire query_req_fire = query_rd_req_valid && query_rd_req_ready;
    wire query_rsp_fire = query_rd_rsp_valid && query_rd_rsp_ready;
    wire q_fire = q_tvalid && q_tready;

    integer lane;
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q <= S_IDLE;
            q_head_base_q <= 6'd0;
            token_count_q <= 4'd0;
            head_q <= 3'd0;
            beat_q <= 4'd0;
            wave_q <= 1'b0;
            issue_count_q <= 4'd0;
            capture_count_q <= 4'd0;
            emit_lane_q <= 2'd0;
            done_error_q <= 1'b0;
        end else begin
            case (state_q)
                S_IDLE: if (start_valid) begin
                    done_error_q <= 1'b0;
                    if ((start_token_count == 4'd0) ||
                        (start_token_count > 4'd8) ||
                        (start_q_head_base > 6'd24) ||
                        (start_q_head_base[2:0] != 3'd0)) begin
                        done_error_q <= 1'b1;
                        state_q <= S_DONE;
                    end else begin
                        q_head_base_q <= start_q_head_base;
                        token_count_q <= start_token_count;
                        head_q <= 3'd0;
                        beat_q <= 4'd0;
                        wave_q <= 1'b0;
                        issue_count_q <= 4'd0;
                        capture_count_q <= 4'd0;
                        emit_lane_q <= 2'd0;
                        state_q <= S_GATHER;
                    end
                end

                S_GATHER: begin
                    if (query_req_fire)
                        issue_count_q <= issue_count_q + 4'd1;
                    if (query_rsp_fire) begin
                        for (lane = 0; lane < 4; lane = lane + 1)
                            lane_buf[lane] <= {
                                query_rd_rsp_data[lane*32 +: 32],
                                lane_buf[lane][255:32]
                            };
                        if (capture_count_q == 4'd7) begin
                            emit_lane_q <= 2'd0;
                            state_q <= S_EMIT;
                        end else begin
                            capture_count_q <= capture_count_q + 4'd1;
                        end
                    end
                end

                S_EMIT: if (q_fire) begin
                    if ({2'd0, emit_lane_q} + 4'd1 == wave_lane_count) begin
                        emit_lane_q <= 2'd0;
                        issue_count_q <= 4'd0;
                        capture_count_q <= 4'd0;
                        if (!wave_q && (token_count_q > 4'd4)) begin
                            wave_q <= 1'b1;
                            state_q <= S_GATHER;
                        end else begin
                            wave_q <= 1'b0;
                            if (beat_q == 4'd15) begin
                                beat_q <= 4'd0;
                                if (head_q == 3'd7) begin
                                    state_q <= S_DONE;
                                end else begin
                                    head_q <= head_q + 3'd1;
                                    state_q <= S_GATHER;
                                end
                            end else begin
                                beat_q <= beat_q + 4'd1;
                                state_q <= S_GATHER;
                            end
                        end
                    end else begin
                        emit_lane_q <= emit_lane_q + 2'd1;
                    end
                end

                S_DONE: if (done_ready)
                    state_q <= S_IDLE;

                default: state_q <= S_IDLE;
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && query_rsp_fire &&
            (capture_count_q >= issue_count_q))
            $fatal(1, "flash_query_gather response without request");
    end
`endif
endmodule

`default_nettype wire
