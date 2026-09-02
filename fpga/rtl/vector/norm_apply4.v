// Four-lane streamed RMSNorm apply pipe: y = x * gamma * inv_rms.
// A credit-counted output FIFO absorbs downstream stalls around the fixed
// latency arithmetic leaves. One accepted four-lane record can enter every cycle.

`default_nettype none

module norm_apply4 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [12:0]   cfg_rows,
    input  wire [3:0]    cfg_lane_mask,
    input  wire [127:0]  cfg_inv_rms,
    input  wire          abort_run,
    output wire          busy,

    input  wire          in_valid,
    output wire          in_ready,
    input  wire [127:0]  in_data,
    input  wire [127:0]  in_gamma,

    output wire          out_valid,
    input  wire          out_ready,
    output wire [127:0]  out_data,
    output wire          out_last
);
    reg active_q;
    reg [12:0] rows_q;
    reg [3:0] lane_mask_q;
    reg [127:0] inv_rms_q;
    reg [13:0] input_count_q;
    reg [13:0] output_count_q;
    reg [4:0] inflight_q;

    reg [127:0] fifo_q [0:15];
    reg [3:0] fifo_wr_q;
    reg [3:0] fifo_rd_q;
    reg [4:0] fifo_count_q;

    wire arithmetic_rst_n = rst_n && !abort_run;
    wire [3:0] mul1_valid;
    wire [3:0] mul2_valid;
    wire [127:0] mul1_data;
    wire [127:0] mul2_data;

    assign cfg_ready = rst_n && !abort_run && !active_q;
    assign busy = active_q;
    wire cfg_fire = cfg_valid && cfg_ready;

    wire [5:0] occupied = {1'b0, fifo_count_q} + {1'b0, inflight_q};
    assign in_ready = active_q && !abort_run &&
                      (input_count_q < {1'b0, rows_q}) &&
                      (occupied < 6'd16);
    wire in_fire = in_valid && in_ready;

    assign out_valid = active_q && !abort_run && (fifo_count_q != 5'd0);
    assign out_data = fifo_q[fifo_rd_q];
    assign out_last = out_valid &&
                      ((output_count_q + 1'b1) == {1'b0, rows_q});
    wire out_fire = out_valid && out_ready;
    wire pipe_fire = mul2_valid[0];

    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_lane
            wire [31:0] lane_x = lane_mask_q[lane] ?
                in_data[lane*32 +: 32] : 32'd0;
            wire [31:0] lane_gamma = lane_mask_q[lane] ?
                in_gamma[lane*32 +: 32] : 32'd0;
            wire [31:0] lane_inv = inv_rms_q[lane*32 +: 32];

            fmul u_gamma (
                .clk(clk), .rst_n(arithmetic_rst_n), .valid_in(in_fire),
                .a(lane_x), .b(lane_gamma),
                .valid_out(mul1_valid[lane]),
                .out(mul1_data[lane*32 +: 32])
            );
            fmul u_inverse (
                .clk(clk), .rst_n(arithmetic_rst_n),
                .valid_in(mul1_valid[lane]),
                .a(mul1_data[lane*32 +: 32]), .b(lane_inv),
                .valid_out(mul2_valid[lane]),
                .out(mul2_data[lane*32 +: 32])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            active_q <= 1'b0;
            rows_q <= 13'd0;
            lane_mask_q <= 4'd0;
            inv_rms_q <= 128'd0;
            input_count_q <= 14'd0;
            output_count_q <= 14'd0;
            inflight_q <= 5'd0;
            fifo_wr_q <= 4'd0;
            fifo_rd_q <= 4'd0;
            fifo_count_q <= 5'd0;
        end else begin
            if (cfg_fire) begin
                active_q <= 1'b1;
                rows_q <= cfg_rows;
                lane_mask_q <= cfg_lane_mask;
                inv_rms_q <= cfg_inv_rms;
                input_count_q <= 14'd0;
                output_count_q <= 14'd0;
                inflight_q <= 5'd0;
                fifo_wr_q <= 4'd0;
                fifo_rd_q <= 4'd0;
                fifo_count_q <= 5'd0;
            end

            if (in_fire) begin
                input_count_q <= input_count_q + 1'b1;
                inflight_q <= inflight_q + 1'b1;
            end
            if (pipe_fire) begin
                fifo_q[fifo_wr_q] <= mul2_data;
                fifo_wr_q <= fifo_wr_q + 1'b1;
                inflight_q <= inflight_q - 1'b1;
            end
            if (out_fire) begin
                fifo_rd_q <= fifo_rd_q + 1'b1;
                output_count_q <= output_count_q + 1'b1;
                if (out_last)
                    active_q <= 1'b0;
            end

            case ({pipe_fire, out_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: begin end
            endcase

            // Both events can occur together; preserve the net inflight change.
            if (in_fire && pipe_fire)
                inflight_q <= inflight_q;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && (mul2_valid != {4{mul2_valid[0]}}))
            $fatal(1, " norm_apply4 lane valid skew");
        if (rst_n && pipe_fire && (fifo_count_q == 5'd16) && !out_fire)
            $fatal(1, " norm_apply4 output FIFO overflow");
        if (rst_n && out_fire && out_last &&
            (input_count_q != {1'b0, rows_q}))
            $fatal(1, " norm_apply4 retired before exact input count");
    end
`endif
endmodule

`default_nettype wire
