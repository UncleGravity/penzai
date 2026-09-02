`default_nettype none

// Four independent 128-bit AXI readers present one lockstep 512-bit packed
// weight stream.  Resident matrices are port-major, so port N starts at
// base + N*port_beats*16.  The model/model_spec controller supplies the exact
// per-port beat count; this service deliberately contains no shape multiplier.
module weight_quad128 #(
    parameter integer ADDR_W = 40
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,
    input  wire                  abort_run,

    input  wire                  cmd_valid,
    output wire                  cmd_ready,
    input  wire [63:0]           cmd_base_addr,
    input  wire [31:0]           cmd_port_beats,
    input  wire [3:0]            cmd_port_mask,

    output wire [511:0]          weight_data,
    output wire                  weight_valid,
    input  wire                  weight_ready,
    output wire                  weight_last,
    output wire                  weight_error,

    output wire                  busy,
    output wire                  done_valid,
    input  wire                  done_ready,
    output wire                  done_error,
    output wire [7:0]            done_status,

    output wire [4*ADDR_W-1:0]   m_axi_araddr,
    output wire [31:0]           m_axi_arlen,
    output wire [11:0]           m_axi_arsize,
    output wire [7:0]            m_axi_arburst,
    output wire [3:0]            m_axi_arvalid,
    input  wire [3:0]            m_axi_arready,
    input  wire [511:0]          m_axi_rdata,
    input  wire [7:0]            m_axi_rresp,
    input  wire [3:0]            m_axi_rlast,
    input  wire [3:0]            m_axi_rvalid,
    output wire [3:0]            m_axi_rready,

    output reg  [2:0]            metrics_axi_r_beats,
    output reg  [2:0]            metrics_axi_r_gap_ports,
    output reg                   metrics_zip_skew
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_START = 3'd1;
    localparam [2:0] ST_RUN   = 3'd2;
    localparam [2:0] ST_ABORT = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    localparam [7:0] STATUS_OK          = 8'h00;
    localparam [7:0] STATUS_BAD_COMMAND = 8'h01;
    localparam [7:0] STATUS_PORT_ERROR  = 8'h02;
    localparam [7:0] STATUS_PORT_LAST   = 8'h03;

    reg [2:0] state_q;
    reg [63:0] base_addr_q;
    reg [31:0] port_beats_q;
    reg [3:0] port_mask_q;
    reg [3:0] reader_started_q;
    reg [3:0] reader_done_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    wire command_ok = (cmd_base_addr != 64'd0) &&
                      (cmd_base_addr[3:0] == 4'd0) &&
                      (cmd_port_beats != 32'd0) &&
                      (cmd_port_mask != 4'd0);
    wire [63:0] port_bytes = {28'd0, port_beats_q, 4'd0};

    wire [3:0] reader_cmd_valid;
    wire [3:0] reader_cmd_ready;
    wire [255:0] reader_cmd_addr;
    wire [511:0] reader_out_data;
    wire [3:0] reader_out_valid;
    wire [3:0] reader_out_ready;
    wire [3:0] reader_out_last;
    wire [3:0] reader_out_error;
    wire [3:0] reader_busy;
    wire [3:0] reader_done_valid;
    wire [3:0] reader_done_ready;
    wire [3:0] reader_done_error;
    wire [31:0] reader_done_status;

    wire [3:0] effective_output_valid = reader_out_valid | ~port_mask_q;
    wire [3:0] effective_output_last = reader_out_last | ~port_mask_q;
    wire all_output_valid = &effective_output_valid;
    wire zip_fire = all_output_valid && weight_ready;
    wire [3:0] started_after = reader_started_q |
                               (reader_cmd_valid & reader_cmd_ready);
    wire [3:0] done_after = reader_done_q | reader_done_valid;
    wire reader_last_mismatch =
        (|(reader_out_last & port_mask_q)) && !(&effective_output_last);
    wire [7:0] first_reader_status = reader_done_error[0] ?
        reader_done_status[7:0] : reader_done_error[1] ?
        reader_done_status[15:8] : reader_done_error[2] ?
        reader_done_status[23:16] : reader_done_status[31:24];

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign weight_data = {
        port_mask_q[3] ? reader_out_data[511:384] : 128'd0,
        port_mask_q[2] ? reader_out_data[383:256] : 128'd0,
        port_mask_q[1] ? reader_out_data[255:128] : 128'd0,
        port_mask_q[0] ? reader_out_data[127:0] : 128'd0
    };
    assign weight_valid = (state_q == ST_RUN) && all_output_valid;
    assign weight_last = &effective_output_last;
    assign weight_error = (|(reader_out_error & port_mask_q)) ||
                          reader_last_mismatch;
    assign reader_out_ready = port_mask_q &
                              {4{(state_q == ST_RUN) && zip_fire}};
    assign reader_done_ready = port_mask_q & {4{state_q == ST_RUN}};
    assign busy = (state_q == ST_START) || (state_q == ST_RUN) ||
                  (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    assign reader_cmd_valid = (state_q == ST_START) ?
                              (~reader_started_q & port_mask_q) : 4'd0;
    assign reader_cmd_addr[0*64 +: 64] = base_addr_q;
    assign reader_cmd_addr[1*64 +: 64] = base_addr_q + port_bytes;
    assign reader_cmd_addr[2*64 +: 64] = base_addr_q +
                                         (port_bytes << 1);
    assign reader_cmd_addr[3*64 +: 64] = base_addr_q +
                                         port_bytes + (port_bytes << 1);

    function automatic [2:0] popcount4(input [3:0] mask);
        begin
            popcount4 = {2'd0, mask[0]} + {2'd0, mask[1]} +
                        {2'd0, mask[2]} + {2'd0, mask[3]};
        end
    endfunction

    // These are observation-only registers beside the physical reader island.
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            metrics_axi_r_beats <= 3'd0;
            metrics_axi_r_gap_ports <= 3'd0;
            metrics_zip_skew <= 1'b0;
        end else begin
            metrics_axi_r_beats <= popcount4(m_axi_rvalid & m_axi_rready);
            metrics_axi_r_gap_ports <= popcount4(m_axi_rready &
                                                  ~m_axi_rvalid);
            metrics_zip_skew <= (state_q == ST_RUN) &&
                (|(reader_out_valid & port_mask_q)) && !all_output_valid;
        end
    end

    genvar port;
    generate
        for (port = 0; port < 4; port = port + 1) begin : g_reader
             axi_read128 #(.ADDR_W(ADDR_W)) u_reader (
                .clk(clk), .rst_n(rst_n), .clear(clear),
                .abort_run(abort_run || (state_q == ST_ABORT)),
                .cmd_valid(reader_cmd_valid[port]),
                .cmd_ready(reader_cmd_ready[port]),
                .cmd_addr(reader_cmd_addr[port*64 +: 64]),
                .cmd_segment_beats(port_beats_q),
                .cmd_stride_bytes(32'd0), .cmd_repeats(17'd1),
                .out_data(reader_out_data[port*128 +: 128]),
                .out_valid(reader_out_valid[port]),
                .out_ready(reader_out_ready[port]),
                .out_last(reader_out_last[port]),
                .out_error(reader_out_error[port]),
                .busy(reader_busy[port]),
                .done_valid(reader_done_valid[port]),
                .done_ready(reader_done_ready[port]),
                .done_error(reader_done_error[port]),
                .done_status(reader_done_status[port*8 +: 8]),
                .m_axi_araddr(m_axi_araddr[port*ADDR_W +: ADDR_W]),
                .m_axi_arlen(m_axi_arlen[port*8 +: 8]),
                .m_axi_arsize(m_axi_arsize[port*3 +: 3]),
                .m_axi_arburst(m_axi_arburst[port*2 +: 2]),
                .m_axi_arvalid(m_axi_arvalid[port]),
                .m_axi_arready(m_axi_arready[port]),
                .m_axi_rdata(m_axi_rdata[port*128 +: 128]),
                .m_axi_rresp(m_axi_rresp[port*2 +: 2]),
                .m_axi_rlast(m_axi_rlast[port]),
                .m_axi_rvalid(m_axi_rvalid[port]),
                .m_axi_rready(m_axi_rready[port])
            );
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            base_addr_q <= 64'd0;
            port_beats_q <= 32'd0;
            port_mask_q <= 4'd0;
            reader_started_q <= 4'd0;
            reader_done_q <= 4'd0;
            done_error_q <= 1'b0;
            done_status_q <= STATUS_OK;
        end else begin
            if (clear || abort_run) begin
                reader_started_q <= 4'd0;
                reader_done_q <= 4'd0;
                done_error_q <= 1'b0;
                done_status_q <= STATUS_OK;
                state_q <= (|reader_busy) ? ST_ABORT : ST_IDLE;
            end else begin
                case (state_q)
                    ST_IDLE: if (cmd_valid) begin
                        base_addr_q <= cmd_base_addr;
                        port_beats_q <= cmd_port_beats;
                        port_mask_q <= cmd_port_mask;
                        reader_started_q <= ~cmd_port_mask;
                        reader_done_q <= ~cmd_port_mask;
                        done_error_q <= !command_ok;
                        done_status_q <= command_ok ? STATUS_OK :
                                                      STATUS_BAD_COMMAND;
                        state_q <= command_ok ? ST_START : ST_DONE;
                    end

                    ST_START: begin
                        reader_started_q <= started_after;
                        if (&started_after)
                            state_q <= ST_RUN;
                    end

                    ST_RUN: begin
                        reader_done_q <= done_after;
                        if (weight_valid && weight_ready && weight_error) begin
                            done_error_q <= 1'b1;
                            done_status_q <= reader_last_mismatch ?
                                             STATUS_PORT_LAST :
                                             STATUS_PORT_ERROR;
                        end
                        if (|reader_done_error) begin
                            done_error_q <= 1'b1;
                            if (done_status_q == STATUS_OK)
                                done_status_q <= first_reader_status == 8'd0 ?
                                                 STATUS_PORT_ERROR :
                                                 first_reader_status;
                        end
                        if (&done_after)
                            state_q <= ST_DONE;
                    end

                    // Child cmd_ready is intentionally suppressed while this
                    // state holds abort high.  Reader busy dropping proves the
                    // bounded AXI response has drained and is sufficient.
                    ST_ABORT: if (!(|reader_busy))
                        state_q <= ST_IDLE;

                    ST_DONE: if (done_ready)
                        state_q <= ST_IDLE;

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && weight_valid && !weight_ready) begin
            assert($stable(weight_data));
            assert($stable(weight_last));
            assert($stable(weight_error));
        end
        if (rst_n && (state_q == ST_RUN) && (|reader_out_ready))
            assert(&reader_out_ready);
    end
`endif
endmodule

`default_nettype wire
