// Four-lane transpose/serialization adapter around one proven scalar-to-Q8 block.
//
// The input is one hidden coordinate for four token lanes per transfer. A
// 32-coordinate tile is buffered, then active token lanes are serialized into
// one shared q8_block. The output remains a lockstep four-lane arena record.

`default_nettype none

module q8_pack4 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [14:0]   cfg_rows,
    input  wire [3:0]    cfg_lane_mask,
    input  wire          abort_run,
    output wire          busy,

    input  wire          in_valid,
    output wire          in_ready,
    input  wire [127:0]  in_data,

    output wire          out_valid,
    input  wire          out_ready,
    output wire [8:0]    out_block,
    output wire [1087:0] out_data,
    output wire [7:0]    out_status,
    output wire          out_last
);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_LOAD   = 3'd1;
    localparam [2:0] ST_FEED   = 3'd2;
    localparam [2:0] ST_WAIT   = 3'd3;
    localparam [2:0] ST_OUTPUT = 3'd4;

    reg [2:0] state_q;
    reg [14:0] rows_q;
    reg [3:0] lane_mask_q;
    reg [15:0] input_count_q;
    reg [9:0] output_count_q;
    reg [4:0] load_index_q;
    reg [4:0] feed_index_q;
    reg [1:0] current_lane_q;
    reg [1087:0] record_q;
    reg [7:0] status_q;

    (* ram_style = "distributed" *) reg [31:0] lane0_mem [0:31];
    (* ram_style = "distributed" *) reg [31:0] lane1_mem [0:31];
    (* ram_style = "distributed" *) reg [31:0] lane2_mem [0:31];
    (* ram_style = "distributed" *) reg [31:0] lane3_mem [0:31];

    function automatic [1:0] first_lane(input [3:0] mask);
        begin
            if (mask[0]) first_lane = 2'd0;
            else if (mask[1]) first_lane = 2'd1;
            else if (mask[2]) first_lane = 2'd2;
            else first_lane = 2'd3;
        end
    endfunction

    function automatic has_next_lane(
        input [3:0] mask,
        input [1:0] lane
    );
        begin
            case (lane)
                2'd0: has_next_lane = |mask[3:1];
                2'd1: has_next_lane = |mask[3:2];
                2'd2: has_next_lane = mask[3];
                default: has_next_lane = 1'b0;
            endcase
        end
    endfunction

    function automatic [1:0] next_lane(
        input [3:0] mask,
        input [1:0] lane
    );
        begin
            case (lane)
                2'd0: begin
                    if (mask[1]) next_lane = 2'd1;
                    else if (mask[2]) next_lane = 2'd2;
                    else next_lane = 2'd3;
                end
                2'd1: next_lane = mask[2] ? 2'd2 : 2'd3;
                default: next_lane = 2'd3;
            endcase
        end
    endfunction

    wire cfg_ok = (cfg_rows != 15'd0) && (cfg_rows[4:0] == 5'd0) &&
                  (cfg_rows <= 15'd16384) && (cfg_lane_mask != 4'd0);
    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;
    wire cfg_fire = cfg_valid && cfg_ready;

    assign in_ready = rst_n && !abort_run && (state_q == ST_LOAD);
    wire in_fire = in_valid && in_ready;

    wire numeric_rst_n = rst_n && !abort_run;
    wire q8_in_ready;
    reg [31:0] selected_input;
    always @(*) begin
        case (current_lane_q)
            2'd0: selected_input = lane0_mem[feed_index_q];
            2'd1: selected_input = lane1_mem[feed_index_q];
            2'd2: selected_input = lane2_mem[feed_index_q];
            default: selected_input = lane3_mem[feed_index_q];
        endcase
    end

    wire q8_in_valid = (state_q == ST_FEED);
    wire q8_in_fire = q8_in_valid && q8_in_ready;
    wire q8_leaf_valid;
    wire [255:0] q8_leaf_quants;
    wire [15:0] q8_leaf_scale;
    wire [1:0] q8_leaf_status;
    wire q8_leaf_fire = q8_leaf_valid && (state_q == ST_WAIT);

    q8_block u_q8 (
        .clk(clk), .rst_n(numeric_rst_n),
        .in_valid(q8_in_valid), .in_ready(q8_in_ready),
        .in_data(selected_input),
        .out_valid(q8_leaf_valid), .out_ready(state_q == ST_WAIT),
        .out_quants(q8_leaf_quants), .out_scale(q8_leaf_scale),
        .out_status(q8_leaf_status)
    );

    assign out_valid = rst_n && !abort_run && (state_q == ST_OUTPUT);
    wire out_fire = out_valid && out_ready;
    assign out_block = output_count_q[8:0];
    assign out_data = record_q;
    assign out_status = status_q;
    assign out_last = out_valid &&
                      (({5'd0, output_count_q} + 15'd1) ==
                       (rows_q >> 5));

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            state_q <= ST_IDLE;
            rows_q <= 15'd0;
            lane_mask_q <= 4'd0;
            input_count_q <= 16'd0;
            output_count_q <= 10'd0;
            load_index_q <= 5'd0;
            feed_index_q <= 5'd0;
            current_lane_q <= 2'd0;
            status_q <= 8'd0;
        end else begin
            case (state_q)
                ST_IDLE: if (cfg_fire) begin
                    rows_q <= cfg_rows;
                    lane_mask_q <= cfg_lane_mask;
                    input_count_q <= 16'd0;
                    output_count_q <= 10'd0;
                    load_index_q <= 5'd0;
                    feed_index_q <= 5'd0;
                    current_lane_q <= first_lane(cfg_lane_mask);
                    status_q <= 8'd0;
                    state_q <= cfg_ok ? ST_LOAD : ST_IDLE;
                end

                ST_LOAD: if (in_fire) begin
                    lane0_mem[load_index_q] <= in_data[31:0];
                    lane1_mem[load_index_q] <= in_data[63:32];
                    lane2_mem[load_index_q] <= in_data[95:64];
                    lane3_mem[load_index_q] <= in_data[127:96];
                    input_count_q <= input_count_q + 1'b1;
                    if (load_index_q == 5'd31) begin
                        load_index_q <= 5'd0;
                        feed_index_q <= 5'd0;
                        current_lane_q <= first_lane(lane_mask_q);
                        state_q <= ST_FEED;
                    end else begin
                        load_index_q <= load_index_q + 1'b1;
                    end
                end

                ST_FEED: if (q8_in_fire) begin
                    if (feed_index_q == 5'd31) begin
                        feed_index_q <= 5'd0;
                        state_q <= ST_WAIT;
                    end else begin
                        feed_index_q <= feed_index_q + 1'b1;
                    end
                end

                ST_WAIT: if (q8_leaf_fire) begin
                    status_q <= status_q |
                        ({6'd0, q8_leaf_status} << (current_lane_q * 2));
                    if (has_next_lane(lane_mask_q, current_lane_q)) begin
                        current_lane_q <= next_lane(lane_mask_q,
                                                    current_lane_q);
                        feed_index_q <= 5'd0;
                        state_q <= ST_FEED;
                    end else begin
                        state_q <= ST_OUTPUT;
                    end
                end

                ST_OUTPUT: if (out_fire) begin
                    if (out_last) begin
                        state_q <= ST_IDLE;
                    end else begin
                        output_count_q <= output_count_q + 1'b1;
                        load_index_q <= 5'd0;
                        current_lane_q <= first_lane(lane_mask_q);
                        status_q <= 8'd0;
                        state_q <= ST_LOAD;
                    end
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end

    // The record payload is visible only in ST_OUTPUT. Keeping it outside the
    // abort/reset tree avoids broadcasting one control net to 1088 data-bit
    // loads. A new command and every non-final block initialize the complete
    // payload before any later ST_OUTPUT can expose it.
    always @(posedge clk) begin
        if (cfg_fire || (out_fire && !out_last)) begin
            record_q <= 1088'd0;
        end else if (q8_leaf_fire) begin
            case (current_lane_q)
                2'd0: record_q[271:0] <=
                    {q8_leaf_scale, q8_leaf_quants};
                2'd1: record_q[543:272] <=
                    {q8_leaf_scale, q8_leaf_quants};
                2'd2: record_q[815:544] <=
                    {q8_leaf_scale, q8_leaf_quants};
                default: record_q[1087:816] <=
                    {q8_leaf_scale, q8_leaf_quants};
            endcase
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && cfg_fire && !cfg_ok)
            $fatal(1, " q8_pack4 bad configuration");
        if (rst_n && out_fire && out_last &&
            (input_count_q != {1'b0, rows_q}))
            $fatal(1, " q8_pack4 retired before exact input count");
        if (rst_n && q8_leaf_valid && (state_q != ST_WAIT))
            $fatal(1, " q8_pack4 unexpected leaf output");
    end
`endif
endmodule

`default_nettype wire
