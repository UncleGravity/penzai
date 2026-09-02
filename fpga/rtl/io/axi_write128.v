`default_nettype none

// Persistent 128-bit AXI4 writer for committed KV records and optional
// logits.  Only one burst is outstanding, which bounds abort quarantine to
// one AW/W/B transaction.  A command describes `repeats` equal-sized,
// strided segments; every segment is split at 256 beats and 4 KiB.
module axi_write128 #(
    parameter integer ADDR_W = 40
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  abort_run,

    input  wire                  cmd_valid,
    output wire                  cmd_ready,
    input  wire [63:0]           cmd_addr,
    input  wire [31:0]           cmd_segment_beats,
    input  wire [31:0]           cmd_stride_bytes,
    input  wire [16:0]           cmd_repeats,

    input  wire [127:0]          in_data,
    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire                  in_last,
    input  wire                  in_error,

    output wire                  busy,
    output wire                  done_valid,
    input  wire                  done_ready,
    output wire                  done_error,
    output wire [7:0]            done_status,

    output wire [ADDR_W-1:0]     m_axi_awaddr,
    output wire [7:0]            m_axi_awlen,
    output wire [2:0]            m_axi_awsize,
    output wire [1:0]            m_axi_awburst,
    output wire                  m_axi_awvalid,
    input  wire                  m_axi_awready,
    output wire [127:0]          m_axi_wdata,
    output wire [15:0]           m_axi_wstrb,
    output wire                  m_axi_wlast,
    output wire                  m_axi_wvalid,
    input  wire                  m_axi_wready,
    input  wire [1:0]            m_axi_bresp,
    input  wire                  m_axi_bvalid,
    output wire                  m_axi_bready
);
    localparam [2:0] ST_IDLE     = 3'd0;
    localparam [2:0] ST_AW       = 3'd1;
    localparam [2:0] ST_W        = 3'd2;
    localparam [2:0] ST_B        = 3'd3;
    localparam [2:0] ST_DONE     = 3'd4;
    localparam [2:0] ST_DRAIN_AW = 3'd5;
    localparam [2:0] ST_DRAIN_W  = 3'd6;
    localparam [2:0] ST_DRAIN_B  = 3'd7;

    localparam [7:0] STATUS_OK          = 8'h00;
    localparam [7:0] STATUS_BAD_CMD     = 8'h01;
    localparam [7:0] STATUS_AXI_RESP    = 8'h02;
    localparam [7:0] STATUS_INPUT_LAST  = 8'h03;
    localparam [7:0] STATUS_INPUT_ERROR = 8'h04;

    reg [2:0] state_q;
    reg [ADDR_W-1:0] repeat_addr_q;
    reg [ADDR_W-1:0] next_aw_addr_q;
    reg [31:0] segment_beats_q;
    reg [31:0] segment_remaining_q;
    reg [31:0] stride_bytes_q;
    reg [16:0] repeats_q;
    reg [16:0] repeat_index_q;
    reg [8:0] burst_beats_q;
    reg [8:0] burst_load_left_q;

    // A one-record elastic register makes W stable across downstream stalls
    // and across an asynchronous abort request.
    reg [127:0] wbuf_data_q;
    reg [15:0] wbuf_strb_q;
    reg wbuf_last_q;
    reg wbuf_valid_q;

    reg drain_to_done_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    wire cmd_addr_fits = (cmd_addr >> ADDR_W) == 64'd0;
    wire cmd_ok = cmd_addr_fits && (cmd_addr[3:0] == 4'd0) &&
                  (cmd_segment_beats != 32'd0) &&
                  (cmd_repeats != 17'd0) &&
                  ((cmd_repeats == 17'd1) ||
                   (cmd_stride_bytes[3:0] == 4'd0));

    wire [8:0] boundary_beats =
        9'd256 - {1'b0, next_aw_addr_q[11:4]};
    wire [8:0] remaining_cap =
        segment_remaining_q > 32'd256 ? 9'd256 :
        segment_remaining_q[8:0];
    wire [8:0] issue_beats =
        remaining_cap < boundary_beats ? remaining_cap : boundary_beats;
    wire [ADDR_W-1:0] stride_ext =
        {{(ADDR_W-32){1'b0}}, stride_bytes_q};
    wire [ADDR_W-1:0] burst_bytes =
        ({{(ADDR_W-9){1'b0}}, burst_beats_q} << 4);

    wire aw_state = (state_q == ST_AW) || (state_q == ST_DRAIN_AW);
    wire w_state = (state_q == ST_W) || (state_q == ST_DRAIN_W);
    wire aw_fire = m_axi_awvalid && m_axi_awready;
    wire w_fire = m_axi_wvalid && m_axi_wready;
    wire b_fire = m_axi_bvalid && m_axi_bready;

    wire wbuf_capacity = !wbuf_valid_q || m_axi_wready;
    wire normal_load = (state_q == ST_W) && !clear && !abort_run &&
                       (burst_load_left_q != 9'd0) && wbuf_capacity &&
                       in_valid;
    wire drain_load = (state_q == ST_DRAIN_W) &&
                      (burst_load_left_q != 9'd0) && wbuf_capacity;
    wire expected_input_last =
        (repeat_index_q + 17'd1 == repeats_q) &&
        (segment_remaining_q == 32'd1);
    wire input_last_error = in_last != expected_input_last;
    wire input_fault = in_error || input_last_error;

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign in_ready = (state_q == ST_W) && !clear && !abort_run &&
                      (burst_load_left_q != 9'd0) && wbuf_capacity;
    assign busy = (state_q != ST_IDLE) && (state_q != ST_DONE);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    assign m_axi_awaddr = next_aw_addr_q;
    assign m_axi_awlen = issue_beats[7:0] - 8'd1;
    assign m_axi_awsize = 3'd4;
    assign m_axi_awburst = 2'b01;
    // Once AWVALID is visible it remains asserted, including during abort,
    // until the slave accepts it.  ST_DRAIN_AW then writes a null burst.
    assign m_axi_awvalid = aw_state;

    assign m_axi_wdata = wbuf_data_q;
    assign m_axi_wstrb = wbuf_strb_q;
    assign m_axi_wlast = wbuf_last_q;
    assign m_axi_wvalid = w_state && wbuf_valid_q;
    assign m_axi_bready = (state_q == ST_B) ||
                          (state_q == ST_DRAIN_B);

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            repeat_addr_q <= {ADDR_W{1'b0}};
            next_aw_addr_q <= {ADDR_W{1'b0}};
            segment_beats_q <= 32'd0;
            segment_remaining_q <= 32'd0;
            stride_bytes_q <= 32'd0;
            repeats_q <= 17'd0;
            repeat_index_q <= 17'd0;
            burst_beats_q <= 9'd0;
            burst_load_left_q <= 9'd0;
            wbuf_data_q <= 128'd0;
            wbuf_strb_q <= 16'd0;
            wbuf_last_q <= 1'b0;
            wbuf_valid_q <= 1'b0;
            drain_to_done_q <= 1'b0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else begin
            if (w_fire)
                wbuf_valid_q <= 1'b0;

            case (state_q)
                ST_IDLE: if (cmd_valid) begin
                    repeat_addr_q <= cmd_addr[ADDR_W-1:0];
                    next_aw_addr_q <= cmd_addr[ADDR_W-1:0];
                    segment_beats_q <= cmd_segment_beats;
                    segment_remaining_q <= cmd_segment_beats;
                    stride_bytes_q <= cmd_stride_bytes;
                    repeats_q <= cmd_repeats;
                    repeat_index_q <= 17'd0;
                    burst_beats_q <= 9'd0;
                    burst_load_left_q <= 9'd0;
                    wbuf_valid_q <= 1'b0;
                    drain_to_done_q <= 1'b0;
                    done_error_q <= !cmd_ok;
                    done_status_q <= cmd_ok ? STATUS_OK : STATUS_BAD_CMD;
                    state_q <= cmd_ok ? ST_AW : ST_DONE;
                end

                ST_AW: if (aw_fire) begin
                    burst_beats_q <= issue_beats;
                    burst_load_left_q <= issue_beats;
                    state_q <= ST_W;
                end

                ST_DRAIN_AW: if (aw_fire) begin
                    burst_beats_q <= issue_beats;
                    burst_load_left_q <= issue_beats;
                    state_q <= ST_DRAIN_W;
                end

                ST_W: if (w_fire && wbuf_last_q)
                    state_q <= ST_B;

                ST_B: if (b_fire) begin
                    if (m_axi_bresp != 2'b00) begin
                        done_error_q <= 1'b1;
                        done_status_q <= STATUS_AXI_RESP;
                        state_q <= ST_DONE;
                    end else if (segment_remaining_q == 32'd0) begin
                        if (repeat_index_q + 17'd1 == repeats_q) begin
                            state_q <= ST_DONE;
                        end else begin
                            repeat_index_q <= repeat_index_q + 17'd1;
                            repeat_addr_q <= repeat_addr_q + stride_ext;
                            next_aw_addr_q <= repeat_addr_q + stride_ext;
                            segment_remaining_q <= segment_beats_q;
                            state_q <= ST_AW;
                        end
                    end else begin
                        next_aw_addr_q <= next_aw_addr_q + burst_bytes;
                        state_q <= ST_AW;
                    end
                end

                ST_DRAIN_W: if (w_fire && wbuf_last_q)
                    state_q <= ST_DRAIN_B;

                ST_DRAIN_B: if (b_fire) begin
                    if (drain_to_done_q) begin
                        if ((m_axi_bresp != 2'b00) && !done_error_q) begin
                            done_error_q <= 1'b1;
                            done_status_q <= STATUS_AXI_RESP;
                        end
                        state_q <= ST_DONE;
                    end else begin
                        state_q <= ST_IDLE;
                    end
                end

                ST_DONE: if (done_ready)
                    state_q <= ST_IDLE;

                default: state_q <= ST_IDLE;
            endcase

            // Normal input occupies the elastic W record.  A bad record is
            // suppressed with WSTRB=0, then the rest of this accepted burst
            // is completed with null writes before reporting the fault.
            if (normal_load) begin
                wbuf_data_q <= in_data;
                wbuf_strb_q <= input_fault ? 16'h0000 : 16'hffff;
                wbuf_last_q <= burst_load_left_q == 9'd1;
                wbuf_valid_q <= 1'b1;
                burst_load_left_q <= burst_load_left_q - 9'd1;
                segment_remaining_q <= segment_remaining_q - 32'd1;
                if (input_fault) begin
                    drain_to_done_q <= 1'b1;
                    done_error_q <= 1'b1;
                    done_status_q <= input_last_error ? STATUS_INPUT_LAST :
                                                        STATUS_INPUT_ERROR;
                    state_q <= ST_DRAIN_W;
                end
            end

            // Quarantine never depends on the abandoned upstream stream.
            // It synthesizes every not-yet-buffered beat of the live burst.
            if (drain_load) begin
                wbuf_data_q <= 128'd0;
                wbuf_strb_q <= 16'h0000;
                wbuf_last_q <= burst_load_left_q == 9'd1;
                wbuf_valid_q <= 1'b1;
                burst_load_left_q <= burst_load_left_q - 9'd1;
            end

            // External cancellation has final priority.  AXI records already
            // visible remain stable; all subsequent W records are null.  No
            // completion survives clear/abort, so restart observes IDLE.
            if (clear || abort_run) begin
                drain_to_done_q <= 1'b0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
                case (state_q)
                    ST_IDLE: begin
                        wbuf_valid_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end
                    ST_AW: begin
                        if (aw_fire) begin
                            burst_beats_q <= issue_beats;
                            burst_load_left_q <= issue_beats;
                            state_q <= ST_DRAIN_W;
                        end else begin
                            state_q <= ST_DRAIN_AW;
                        end
                    end
                    ST_DRAIN_AW: begin
                        if (aw_fire) begin
                            burst_beats_q <= issue_beats;
                            burst_load_left_q <= issue_beats;
                            state_q <= ST_DRAIN_W;
                        end
                    end
                    ST_W, ST_DRAIN_W: begin
                        if (w_fire && wbuf_last_q)
                            state_q <= ST_DRAIN_B;
                        else
                            state_q <= ST_DRAIN_W;
                    end
                    ST_B, ST_DRAIN_B: begin
                        if (b_fire)
                            state_q <= ST_IDLE;
                        else
                            state_q <= ST_DRAIN_B;
                    end
                    ST_DONE: begin
                        wbuf_valid_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end
                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && m_axi_awvalid) begin
            assert(issue_beats >= 9'd1);
            assert(issue_beats <= 9'd256);
            assert(({1'b0, next_aw_addr_q[11:4]} + issue_beats) <=
                   9'd256);
        end
        if (rst_n && $past(rst_n) &&
            $past(m_axi_awvalid && !m_axi_awready)) begin
            assert(m_axi_awvalid);
            assert($stable(m_axi_awaddr));
            assert($stable(m_axi_awlen));
        end
        if (rst_n && $past(rst_n) &&
            $past(m_axi_wvalid && !m_axi_wready)) begin
            assert(m_axi_wvalid);
            assert($stable(m_axi_wdata));
            assert($stable(m_axi_wstrb));
            assert($stable(m_axi_wlast));
        end
    end
`endif
endmodule

`default_nettype wire
