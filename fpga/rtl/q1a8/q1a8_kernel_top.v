// q1a8_kernel_top - AXI-Lite top for the dual-stream Q1A8 kernel (v5).
//
// One host command drives a full matmul column: PS sets NUM_Q1_BLOCKS and
// NUM_ROWBLOCKS, then strobes CTRL.start. The kernel reads acts (once,
// shared across rowblocks) via S_AXIS_ACTS, then walks rowblocks via
// S_AXIS for the weight stream. Results stream out the M_AXIS master.
//
// Per rowblock: 4 beats of 64-bit data (lane-major, 2 fp32/beat). Total
// result burst = NUM_ROWBLOCKS * 32 bytes.
//
// v5 adds the performance counter bank (W/A/R_STALL, W/A/R_BEATS) over v4.
// The register map is the single source fpga/regmap/q1a8.regmap, included here
// as the generated q1a8_regs.vh — do not duplicate offsets in this file.

`default_nettype none

module q1a8_kernel_top (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    // No fixed FREQ_HZ: the Zynq PLL can't hit every requested FCLK exactly
    // (e.g. a 150 MHz request yields 142.857 MHz), and a pinned value here
    // makes validate_bd_design fail with a FREQ_HZ mismatch against the
    // propagated DMA/FCLK clock. Leaving it out lets Vivado propagate the
    // actual FCLK_CLK0 frequency through all four associated interfaces.
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS:S_AXIS_ACTS:M_AXIS, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire         s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [7:0]   s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0]   s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire         s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire         s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]  s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire         s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire         s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]   s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire         s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire         s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [7:0]   s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire [2:0]   s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire         s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire         s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0]  s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]   s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire         s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire         s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  wire [63:0]  s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input  wire [7:0]   s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input  wire         s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire         s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  wire         s_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_ACTS TDATA" *)
    input  wire [63:0]  s_axis_acts_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_ACTS TKEEP" *)
    input  wire [7:0]   s_axis_acts_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_ACTS TVALID" *)
    input  wire         s_axis_acts_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_ACTS TREADY" *)
    output wire         s_axis_acts_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_ACTS TLAST" *)
    input  wire         s_axis_acts_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [63:0]  m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [7:0]   m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  wire         m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire         m_axis_tlast
);
    localparam integer ROWS = 8;

    // Register offsets and RO reset values, generated from fpga/regmap/q1a8.regmap
    // by `zig build regmap`. ID/VERSION/ROWS resets and every offset live here so
    // the RTL decode and the Zig MMIO driver never drift.
    `include "q1a8_regs.vh"

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    reg [15:0] num_q1_blocks_q;
    reg [15:0] num_rowblocks_q;
    reg        start_strobe;
    reg        done_latched;
    reg [31:0] cycle_count_q;

    wire kernel_busy;
    wire kernel_done;

    q1a8_kernel #(.ROWS(ROWS)) u_kernel (
        .clk(clk),
        .rst_n(rst_n),
        .start_kernel(start_strobe),
        .num_q1_blocks(num_q1_blocks_q),
        .num_rowblocks(num_rowblocks_q),
        .kernel_done(kernel_done),
        .busy(kernel_busy),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_acts_tdata(s_axis_acts_tdata),
        .s_axis_acts_tvalid(s_axis_acts_tvalid),
        .s_axis_acts_tready(s_axis_acts_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tkeep(m_axis_tkeep)
    );

    always @(posedge clk) begin
        if (!rst_n)            done_latched <= 1'b0;
        else if (start_strobe) done_latched <= 1'b0;
        else if (kernel_done)  done_latched <= 1'b1;
    end

    always @(posedge clk) begin
        if (!rst_n)            cycle_count_q <= 32'd0;
        else if (start_strobe) cycle_count_q <= 32'd0;
        else if (kernel_busy)  cycle_count_q <= cycle_count_q + 32'd1;
    end

    // Performance counter bank (latched per run, like CYCLES; counts only while
    // the kernel is busy). Beats: AXIS transfers actually moved. Stalls: the
    // kernel was ready for an input beat but the stream had no data (starved), or
    // had an output beat the sink was not ready for (backpressure). On silicon,
    // array utilization = (CYCLES - max(stall)) / CYCLES.
    reg [31:0] w_stall_q, a_stall_q, r_stall_q;
    reg [31:0] w_beats_q, a_beats_q, r_beats_q;
    always @(posedge clk) begin
        if (!rst_n || start_strobe) begin
            w_stall_q <= 32'd0; a_stall_q <= 32'd0; r_stall_q <= 32'd0;
            w_beats_q <= 32'd0; a_beats_q <= 32'd0; r_beats_q <= 32'd0;
        end else if (kernel_busy) begin
            if (s_axis_tvalid && s_axis_tready)           w_beats_q <= w_beats_q + 32'd1;
            if (s_axis_acts_tvalid && s_axis_acts_tready) a_beats_q <= a_beats_q + 32'd1;
            if (m_axis_tvalid && m_axis_tready)           r_beats_q <= r_beats_q + 32'd1;
            if (s_axis_tready && !s_axis_tvalid)           w_stall_q <= w_stall_q + 32'd1;
            if (s_axis_acts_tready && !s_axis_acts_tvalid) a_stall_q <= a_stall_q + 32'd1;
            if (m_axis_tvalid && !m_axis_tready)           r_stall_q <= r_stall_q + 32'd1;
        end
    end

    reg awready_q, wready_q, bvalid_q;
    reg [7:0] awaddr_q;

    wire write_accept =
        !awready_q && !wready_q && s_axi_awvalid && s_axi_wvalid && !bvalid_q;
    wire write_commit = awready_q && wready_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            awready_q       <= 1'b0;
            wready_q        <= 1'b0;
            bvalid_q        <= 1'b0;
            awaddr_q        <= 8'd0;
            num_q1_blocks_q <= 16'd0;
            num_rowblocks_q <= 16'd0;
            start_strobe    <= 1'b0;
        end else begin
            start_strobe <= 1'b0;

            awready_q <= write_accept;
            wready_q  <= write_accept;
            if (write_accept) awaddr_q <= s_axi_awaddr;

            if (write_commit)                  bvalid_q <= 1'b1;
            else if (bvalid_q && s_axi_bready) bvalid_q <= 1'b0;

            if (write_commit) begin
                case (awaddr_q[5:0])
                    Q1A8_OFF_CTRL[5:0]: begin
                        if (s_axi_wstrb[0] && s_axi_wdata[0])
                            start_strobe <= 1'b1;
                    end
                    Q1A8_OFF_NUM_Q1_BLOCKS[5:0]: begin
                        if (s_axi_wstrb[0]) num_q1_blocks_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_q1_blocks_q[15:8] <= s_axi_wdata[15:8];
                    end
                    Q1A8_OFF_NUM_ROWBLOCKS[5:0]: begin
                        if (s_axi_wstrb[0]) num_rowblocks_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_rowblocks_q[15:8] <= s_axi_wdata[15:8];
                    end
                    default: ;
                endcase
            end
        end
    end

    reg        arready_q, rvalid_q;
    reg [31:0] rdata_q;

    wire read_accept = !arready_q && s_axi_arvalid && !rvalid_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            arready_q <= 1'b0;
            rvalid_q  <= 1'b0;
            rdata_q   <= 32'd0;
        end else begin
            arready_q <= read_accept;
            if (read_accept) begin
                rvalid_q <= 1'b1;
                case (s_axi_araddr[5:0])
                    Q1A8_OFF_ID[5:0]:            rdata_q <= Q1A8_RST_ID;
                    Q1A8_OFF_VERSION[5:0]:       rdata_q <= Q1A8_RST_VERSION;
                    Q1A8_OFF_STATUS[5:0]:        rdata_q <= {30'd0, done_latched, kernel_busy};
                    Q1A8_OFF_NUM_Q1_BLOCKS[5:0]: rdata_q <= {16'd0, num_q1_blocks_q};
                    Q1A8_OFF_NUM_ROWBLOCKS[5:0]: rdata_q <= {16'd0, num_rowblocks_q};
                    Q1A8_OFF_CYCLES[5:0]:        rdata_q <= cycle_count_q;
                    Q1A8_OFF_ROWS[5:0]:          rdata_q <= Q1A8_RST_ROWS;
                    Q1A8_OFF_W_STALL[5:0]:       rdata_q <= w_stall_q;
                    Q1A8_OFF_A_STALL[5:0]:       rdata_q <= a_stall_q;
                    Q1A8_OFF_R_STALL[5:0]:       rdata_q <= r_stall_q;
                    Q1A8_OFF_W_BEATS[5:0]:       rdata_q <= w_beats_q;
                    Q1A8_OFF_A_BEATS[5:0]:       rdata_q <= a_beats_q;
                    Q1A8_OFF_R_BEATS[5:0]:       rdata_q <= r_beats_q;
                    default: rdata_q <= 32'd0;
                endcase
            end else if (rvalid_q && s_axi_rready) begin
                rvalid_q <= 1'b0;
            end
        end
    end

    assign s_axi_awready = awready_q;
    assign s_axi_wready  = wready_q;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = bvalid_q;
    assign s_axi_arready = arready_q;
    assign s_axi_rdata   = rdata_q;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = rvalid_q;

    wire _unused = &{
        1'b0,
        s_axi_awprot,
        s_axi_arprot,
        s_axis_tkeep,
        s_axis_tlast,
        s_axis_acts_tkeep,
        s_axis_acts_tlast
    };

endmodule
