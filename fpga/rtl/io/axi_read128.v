`default_nettype none

// Persistent 128-bit AXI4 reader used by both packed-weight streams and
// strided committed-KV reads.  It intentionally allows only one outstanding
// burst: long transfers still amortize AR latency, while abort can drain one
// bounded transaction without retaining stale data for the next command.
module axi_read128 #(
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

    output wire [127:0]          out_data,
    output wire                  out_valid,
    input  wire                  out_ready,
    output wire                  out_last,
    output wire                  out_error,

    output wire                  busy,
    output wire                  done_valid,
    input  wire                  done_ready,
    output wire                  done_error,
    output wire [7:0]            done_status,

    output wire [ADDR_W-1:0]     m_axi_araddr,
    output wire [7:0]            m_axi_arlen,
    output wire [2:0]            m_axi_arsize,
    output wire [1:0]            m_axi_arburst,
    output wire                  m_axi_arvalid,
    input  wire                  m_axi_arready,
    input  wire [127:0]          m_axi_rdata,
    input  wire [1:0]            m_axi_rresp,
    input  wire                  m_axi_rlast,
    input  wire                  m_axi_rvalid,
    output wire                  m_axi_rready
);
    localparam [3:0] ST_IDLE      = 4'd0;
    localparam [3:0] ST_AR        = 4'd1;
    localparam [3:0] ST_R         = 4'd2;
    localparam [3:0] ST_FINISH    = 4'd3;
    localparam [3:0] ST_DONE      = 4'd4;
    localparam [3:0] ST_DRAIN     = 4'd5;
    localparam [3:0] ST_ERR_DRAIN = 4'd6;
    localparam [3:0] ST_DRAIN_AR  = 4'd7;
    localparam [3:0] ST_PREP      = 4'd8;

    localparam [7:0] STATUS_OK       = 8'h00;
    localparam [7:0] STATUS_BAD_CMD  = 8'h01;
    localparam [7:0] STATUS_AXI_RESP = 8'h02;
    localparam [7:0] STATUS_AXI_LAST = 8'h03;

    reg [3:0] state_q;
    reg [ADDR_W-1:0] cmd_addr_stage_q;
    reg [31:0] cmd_segment_beats_stage_q;
    reg [31:0] cmd_stride_bytes_stage_q;
    reg [16:0] cmd_repeats_stage_q;
    reg cmd_ok_stage_q;
    reg [ADDR_W-1:0] repeat_addr_q;
    reg [ADDR_W-1:0] next_ar_addr_q;
    reg [31:0] segment_beats_q;
    reg [31:0] segment_remaining_q;
    reg [31:0] stride_bytes_q;
    reg [16:0] repeats_q;
    reg [16:0] repeat_index_q;
    reg [8:0] issue_beats_q;
    reg [8:0] burst_beats_q;
    reg [8:0] burst_left_q;

    reg [127:0] out_data_q;
    reg out_valid_q;
    reg out_last_q;
    reg out_error_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    wire cmd_addr_fits = (cmd_addr >> ADDR_W) == 64'd0;
    wire cmd_ok = cmd_addr_fits && (cmd_addr[3:0] == 4'd0) &&
                  (cmd_segment_beats != 32'd0) &&
                  (cmd_repeats != 17'd0) &&
                  ((cmd_repeats == 17'd1) ||
                   (cmd_stride_bytes[3:0] == 4'd0));

    // AXI4 bursts cannot cross a 4 KiB boundary. Register this calculation
    // before exposing ARVALID so neither the remaining counter nor a boundary
    // subtract drives the PS-facing ARLEN pins.
    function automatic [8:0] burst_issue_beats;
        input [31:0] remaining;
        input [ADDR_W-1:0] addr;
        reg [8:0] boundary;
        reg [8:0] capped;
        begin
            boundary = 9'd256 - {1'b0, addr[11:4]};
            capped = remaining > 32'd256 ? 9'd256 : remaining[8:0];
            burst_issue_beats = capped < boundary ? capped : boundary;
        end
    endfunction

    wire [ADDR_W-1:0] stride_ext =
        {{(ADDR_W-32){1'b0}}, stride_bytes_q};
    wire [ADDR_W-1:0] burst_bytes =
        ({{(ADDR_W-9){1'b0}}, burst_beats_q} << 4);
    wire [ADDR_W-1:0] next_segment_addr = next_ar_addr_q + burst_bytes;
    wire [ADDR_W-1:0] next_repeat_addr = repeat_addr_q + stride_ext;

    wire ar_state = (state_q == ST_AR) || (state_q == ST_DRAIN_AR);
    wire ar_fire = m_axi_arvalid && m_axi_arready;
    wire out_fire = out_valid_q && out_ready;
    wire out_capacity = !out_valid_q || out_ready;
    // RREADY is never asserted before AR ownership is established. In
    // particular, a canceled stalled AR may see a zero-latency RVALID in its
    // eventual handshake cycle; ST_DRAIN consumes it on the following cycle.
    wire discarding = (state_q == ST_DRAIN) ||
                      (state_q == ST_ERR_DRAIN) ||
                      (((clear || abort_run) && (state_q == ST_R)));
    wire r_fire = m_axi_rvalid && m_axi_rready;
    wire expected_rlast = burst_left_q == 9'd1;
    wire final_input_beat =
        (repeat_index_q + 17'd1 == repeats_q) &&
        (segment_remaining_q == 32'd1);
    wire beat_resp_error = m_axi_rresp != 2'b00;
    wire beat_last_error = m_axi_rlast != expected_rlast;

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    wire cmd_fire = cmd_valid && cmd_ready;
    assign out_data = out_data_q;
    assign out_valid = out_valid_q;
    assign out_last = out_last_q;
    assign out_error = out_error_q;
    assign busy = (state_q != ST_IDLE) && (state_q != ST_DONE);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    assign m_axi_araddr = next_ar_addr_q;
    assign m_axi_arlen = issue_beats_q[7:0] - 8'd1;
    assign m_axi_arsize = 3'd4;
    assign m_axi_arburst = 2'b01;
    // AXI VALID cannot be withdrawn after it is visible. Cancellation before
    // AR acceptance holds the request, then quarantines the accepted burst.
    assign m_axi_arvalid = ar_state;
    assign m_axi_rready = ((state_q == ST_R) && out_capacity) ||
                          discarding;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            cmd_addr_stage_q <= {ADDR_W{1'b0}};
            cmd_segment_beats_stage_q <= 32'd0;
            cmd_stride_bytes_stage_q <= 32'd0;
            cmd_repeats_stage_q <= 17'd0;
            cmd_ok_stage_q <= 1'b0;
            repeat_addr_q <= {ADDR_W{1'b0}};
            next_ar_addr_q <= {ADDR_W{1'b0}};
            segment_beats_q <= 32'd0;
            segment_remaining_q <= 32'd0;
            stride_bytes_q <= 32'd0;
            repeats_q <= 17'd0;
            repeat_index_q <= 17'd0;
            issue_beats_q <= 9'd0;
            burst_beats_q <= 9'd0;
            burst_left_q <= 9'd0;
            out_data_q <= 128'd0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_error_q <= 1'b0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else begin
            if (out_fire)
                out_valid_q <= 1'b0;

            if (clear || abort_run) begin
                out_valid_q <= 1'b0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
                if ((state_q == ST_AR) || (state_q == ST_DRAIN_AR)) begin
                    state_q <= ar_fire ? ST_DRAIN : ST_DRAIN_AR;
                    if (ar_fire) begin
                        burst_beats_q <= issue_beats_q;
                        burst_left_q <= issue_beats_q;
                    end
                end else if ((state_q == ST_R) || (state_q == ST_DRAIN) ||
                    (state_q == ST_ERR_DRAIN)) begin
                    state_q <= ST_DRAIN;
                    if (r_fire && m_axi_rlast)
                        state_q <= ST_IDLE;
                end else begin
                    state_q <= ST_IDLE;
                end
            end else begin
                case (state_q)
                    ST_IDLE: begin
                        // Sample the command payload independently of VALID.
                        // Only state_q consumes cmd_fire, keeping the upstream
                        // producer's reset/control cone off the wide active
                        // reader register enables.
                        cmd_addr_stage_q <= cmd_addr[ADDR_W-1:0];
                        cmd_segment_beats_stage_q <= cmd_segment_beats;
                        cmd_stride_bytes_stage_q <= cmd_stride_bytes;
                        cmd_repeats_stage_q <= cmd_repeats;
                        cmd_ok_stage_q <= cmd_ok;
                        if (cmd_fire)
                            state_q <= ST_PREP;
                    end

                    ST_PREP: begin
                        repeat_addr_q <= cmd_addr_stage_q;
                        next_ar_addr_q <= cmd_addr_stage_q;
                        segment_beats_q <= cmd_segment_beats_stage_q;
                        segment_remaining_q <= cmd_segment_beats_stage_q;
                        stride_bytes_q <= cmd_stride_bytes_stage_q;
                        repeats_q <= cmd_repeats_stage_q;
                        repeat_index_q <= 17'd0;
                        issue_beats_q <= burst_issue_beats(
                            cmd_segment_beats_stage_q, cmd_addr_stage_q);
                        done_error_q <= !cmd_ok_stage_q;
                        done_status_q <= cmd_ok_stage_q ? STATUS_OK :
                                                          STATUS_BAD_CMD;
                        out_valid_q <= 1'b0;
                        out_last_q <= 1'b0;
                        out_error_q <= 1'b0;
                        state_q <= cmd_ok_stage_q ? ST_AR : ST_DONE;
                    end

                    ST_AR: if (ar_fire) begin
                        burst_beats_q <= issue_beats_q;
                        burst_left_q <= issue_beats_q;
                        state_q <= ST_R;
                    end

                    ST_DRAIN_AR: if (ar_fire) begin
                        burst_beats_q <= issue_beats_q;
                        burst_left_q <= issue_beats_q;
                        state_q <= ST_DRAIN;
                    end

                    ST_R: if (r_fire) begin
                        out_data_q <= m_axi_rdata;
                        out_valid_q <= 1'b1;
                        out_last_q <= final_input_beat ||
                                      (m_axi_rlast && !expected_rlast);
                        out_error_q <= beat_resp_error || beat_last_error;

                        if (beat_resp_error) begin
                            done_error_q <= 1'b1;
                            if (done_status_q == STATUS_OK)
                                done_status_q <= STATUS_AXI_RESP;
                        end
                        if (beat_last_error) begin
                            done_error_q <= 1'b1;
                            done_status_q <= STATUS_AXI_LAST;
                        end

                        segment_remaining_q <= segment_remaining_q - 32'd1;
                        burst_left_q <= burst_left_q - 9'd1;

                        if (m_axi_rlast && !expected_rlast) begin
                            // The interconnect ended a burst early.  Publish
                            // the bad terminal beat and stop this command.
                            state_q <= ST_FINISH;
                        end else if (!m_axi_rlast && expected_rlast) begin
                            // The requested burst count is complete but RLAST
                            // is late.  Publish the expected terminal record,
                            // then quarantine and drain until the bus closes.
                            out_last_q <= 1'b1;
                            state_q <= ST_ERR_DRAIN;
                        end else if (expected_rlast) begin
                            if (segment_remaining_q == 32'd1) begin
                                if (repeat_index_q + 17'd1 == repeats_q) begin
                                    state_q <= ST_FINISH;
                                end else begin
                                    repeat_index_q <= repeat_index_q + 17'd1;
                                    repeat_addr_q <= next_repeat_addr;
                                    next_ar_addr_q <= next_repeat_addr;
                                    segment_remaining_q <= segment_beats_q;
                                    issue_beats_q <= burst_issue_beats(
                                        segment_beats_q, next_repeat_addr);
                                    state_q <= ST_AR;
                                end
                            end else begin
                                next_ar_addr_q <= next_segment_addr;
                                issue_beats_q <= burst_issue_beats(
                                    segment_remaining_q - 32'd1,
                                    next_segment_addr);
                                state_q <= ST_AR;
                            end
                        end
                    end

                    ST_FINISH: if (!out_valid_q || out_fire)
                        state_q <= ST_DONE;

                    ST_DONE: if (done_ready)
                        state_q <= ST_IDLE;

                    ST_DRAIN: if (r_fire && m_axi_rlast)
                        state_q <= ST_IDLE;

                    ST_ERR_DRAIN: if (r_fire && m_axi_rlast)
                        state_q <= ST_FINISH;

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && m_axi_arvalid) begin
            assert(issue_beats_q >= 9'd1);
            assert(issue_beats_q <= 9'd256);
            assert(({1'b0, next_ar_addr_q[11:4]} + issue_beats_q) <=
                   9'd256);
        end
        if (rst_n && $past(rst_n) &&
            $past(m_axi_arvalid && !m_axi_arready)) begin
            assert(m_axi_arvalid);
            assert($stable(m_axi_araddr));
            assert($stable(m_axi_arlen));
        end
        if (rst_n && out_valid && !out_ready) begin
            assert($stable(out_data));
            assert($stable(out_last));
            assert($stable(out_error));
        end
    end
`endif
endmodule

`default_nettype wire
