// Credit-counted four-lane residual adder. Four independent FP32 adds share one
// ready/valid record boundary and sustain one accepted record per cycle.

`default_nettype none

module residual4 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [12:0]   cfg_rows,
    input  wire [3:0]    cfg_lane_mask,
    input  wire          abort_run,
    output wire          busy,

    input  wire          in_valid,
    output wire          in_ready,
    input  wire [127:0]  in_residual,
    input  wire [127:0]  in_delta,

    output wire          out_valid,
    input  wire          out_ready,
    output wire [127:0]  out_data,
    output wire          out_last
);
    reg active_q;
    reg [12:0] rows_q;
    reg [3:0] lane_mask_q;
    reg [13:0] input_count_q;
    reg [13:0] output_count_q;
    reg [4:0] inflight_q;
    reg [127:0] fifo_q [0:15];
    reg [3:0] fifo_wr_q;
    reg [3:0] fifo_rd_q;
    reg [4:0] fifo_count_q;

    wire arithmetic_rst_n = rst_n && !abort_run;
    wire [3:0] add_valid;
    wire [127:0] add_data;
    wire add_fire = add_valid[0];

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

    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : g_lane
            wire [31:0] a = lane_mask_q[lane] ?
                in_residual[lane*32 +: 32] : 32'd0;
            wire [31:0] b = lane_mask_q[lane] ?
                in_delta[lane*32 +: 32] : 32'd0;
            fadd u_add (
                .clk(clk), .rst_n(arithmetic_rst_n), .valid_in(in_fire),
                .a(a), .b(b), .valid_out(add_valid[lane]),
                .out(add_data[lane*32 +: 32])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            active_q <= 1'b0;
            rows_q <= 13'd0;
            lane_mask_q <= 4'd0;
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
            if (add_fire) begin
                fifo_q[fifo_wr_q] <= add_data;
                fifo_wr_q <= fifo_wr_q + 1'b1;
                inflight_q <= inflight_q - 1'b1;
            end
            if (out_fire) begin
                fifo_rd_q <= fifo_rd_q + 1'b1;
                output_count_q <= output_count_q + 1'b1;
                if (out_last)
                    active_q <= 1'b0;
            end
            case ({add_fire, out_fire})
                2'b10: fifo_count_q <= fifo_count_q + 1'b1;
                2'b01: fifo_count_q <= fifo_count_q - 1'b1;
                default: begin end
            endcase
            if (in_fire && add_fire)
                inflight_q <= inflight_q;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && (add_valid != {4{add_valid[0]}}))
            $fatal(1, " residual4 lane valid skew");
        if (rst_n && add_fire && (fifo_count_q == 5'd16) && !out_fire)
            $fatal(1, " residual4 output FIFO overflow");
    end
`endif
endmodule

`default_nettype wire
