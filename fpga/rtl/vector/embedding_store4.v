`default_nettype none

// Transpose embedding decoder records ({one token, four adjacent FP32 values})
// into the resident-R four-lane arena ({one coordinate, up to four token lanes}).
// Embedding is startup-only, so a four-cycle serializer is preferable to a
// second wide write port on the URAM arena.
module embedding_store4 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          abort_run,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [12:0]   cfg_rows,
    input  wire [3:0]    cfg_token_count,
    input  wire [7:0]    cfg_token_mask,

    input  wire          in_valid,
    output wire          in_ready,
    input  wire [2:0]    in_token,
    input  wire [11:0]   in_index,
    input  wire [127:0]  in_data,
    input  wire          in_last,

    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    output wire          r_wr_wave,
    output wire [11:0]   r_wr_addr,
    output wire [3:0]    r_wr_lane_mask,
    output wire [127:0]  r_wr_data,

    output wire          busy,
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [7:0]    done_status
);
    localparam [1:0] ST_IDLE = 2'd0;
    localparam [1:0] ST_RUN  = 2'd1;
    localparam [1:0] ST_DONE = 2'd2;

    localparam [7:0] STATUS_OK          = 8'h00;
    localparam [7:0] STATUS_BAD_COMMAND = 8'h01;
    localparam [7:0] STATUS_FRAMING     = 8'h02;

    reg [1:0] state_q;
    reg [12:0] rows_q;
    reg [3:0] token_count_q;
    reg [7:0] token_mask_q;
    reg [2:0] expected_token_q;
    reg [11:0] expected_index_q;
    reg held_valid_q;
    reg [2:0] held_token_q;
    reg [11:0] held_index_q;
    reg [127:0] held_data_q;
    reg held_last_q;
    reg [1:0] scalar_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    function automatic [7:0] prefix_mask(input [3:0] count);
        begin
            case (count)
                4'd1: prefix_mask = 8'h01;
                4'd2: prefix_mask = 8'h03;
                4'd3: prefix_mask = 8'h07;
                4'd4: prefix_mask = 8'h0f;
                4'd5: prefix_mask = 8'h1f;
                4'd6: prefix_mask = 8'h3f;
                4'd7: prefix_mask = 8'h7f;
                4'd8: prefix_mask = 8'hff;
                default: prefix_mask = 8'h00;
            endcase
        end
    endfunction

    wire cfg_ok = (cfg_rows != 13'd0) && (cfg_rows <= 13'd4096) &&
                  (cfg_rows[1:0] == 2'd0) &&
                  (cfg_token_count >= 4'd1) &&
                  (cfg_token_count <= 4'd8) &&
                  (cfg_token_mask == prefix_mask(cfg_token_count));
    wire input_expected_last =
        ({1'b0, expected_token_q} + 4'd1 == token_count_q) &&
        ({1'b0, expected_index_q} + 13'd4 == rows_q);
    wire input_framing_ok = token_mask_q[in_token] &&
                            (in_token == expected_token_q) &&
                            (in_index == expected_index_q) &&
                            ({1'b0, in_index} + 13'd4 <= rows_q) &&
                            (in_last == input_expected_last);
    wire in_fire = in_valid && in_ready;
    wire r_wr_fire = r_wr_valid && r_wr_ready;
    wire [31:0] selected_scalar =
        held_data_q[scalar_q*32 +: 32];

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign in_ready = rst_n && !abort_run && (state_q == ST_RUN) &&
                      !held_valid_q;
    assign r_wr_valid = rst_n && !abort_run && (state_q == ST_RUN) &&
                        held_valid_q;
    assign r_wr_wave = held_token_q[2];
    assign r_wr_addr = held_index_q + {10'd0, scalar_q};
    assign r_wr_lane_mask = 4'b0001 << held_token_q[1:0];
    assign r_wr_data = {4{selected_scalar}};
    assign busy = state_q == ST_RUN;
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            rows_q <= 13'd0;
            token_count_q <= 4'd0;
            token_mask_q <= 8'd0;
            expected_token_q <= 3'd0;
            expected_index_q <= 12'd0;
            held_valid_q <= 1'b0;
            held_token_q <= 3'd0;
            held_index_q <= 12'd0;
            held_data_q <= 128'd0;
            held_last_q <= 1'b0;
            scalar_q <= 2'd0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            held_valid_q <= 1'b0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else begin
            case (state_q)
                ST_IDLE: if (cfg_valid) begin
                    rows_q <= cfg_rows;
                    token_count_q <= cfg_token_count;
                    token_mask_q <= cfg_token_mask;
                    expected_token_q <= 3'd0;
                    expected_index_q <= 12'd0;
                    held_valid_q <= 1'b0;
                    scalar_q <= 2'd0;
                    done_error_q <= !cfg_ok;
                    done_status_q <= cfg_ok ? STATUS_OK :
                                              STATUS_BAD_COMMAND;
                    state_q <= cfg_ok ? ST_RUN : ST_DONE;
                end

                ST_RUN: begin
                    if (in_fire) begin
                        if (!input_framing_ok) begin
                            held_valid_q <= 1'b0;
                            done_error_q <= 1'b1;
                            done_status_q <= STATUS_FRAMING;
                            state_q <= ST_DONE;
                        end else begin
                            held_valid_q <= 1'b1;
                            held_token_q <= in_token;
                            held_index_q <= in_index;
                            held_data_q <= in_data;
                            held_last_q <= in_last;
                            scalar_q <= 2'd0;
                        end
                    end

                    if (r_wr_fire) begin
                        if (scalar_q != 2'd3) begin
                            scalar_q <= scalar_q + 1'b1;
                        end else begin
                            held_valid_q <= 1'b0;
                            scalar_q <= 2'd0;
                            if (held_last_q) begin
                                state_q <= ST_DONE;
                            end else if ({1'b0, expected_index_q} + 13'd4 ==
                                         rows_q) begin
                                expected_token_q <= expected_token_q + 1'b1;
                                expected_index_q <= 12'd0;
                            end else begin
                                expected_index_q <= expected_index_q + 12'd4;
                            end
                        end
                    end
                end

                ST_DONE: if (done_ready)
                    state_q <= ST_IDLE;

                default: state_q <= ST_IDLE;
            endcase
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && r_wr_valid && !r_wr_ready) begin
            assert($stable(r_wr_wave));
            assert($stable(r_wr_addr));
            assert($stable(r_wr_lane_mask));
            assert($stable(r_wr_data));
        end
    end
`endif
endmodule

`default_nettype wire
