// decode_top - AXI-Lite top for the fixed-point gemm kernel (plan-fpga-7.md §gemm.v / decode_top).
//
// Owns the deployed AXI-Lite contract, four-port weight zip, 64-bit acts/result streams,
// counter bank, and the fixed-point gemm_kernel datapath.
// The fixed-point window floor is a baked-in constant (EMIN_FLOOR — the f16-format floor, not a
// per-model knob: a contribution exponent is e_ws+e_as ∈ [-48,+10] for ANY f16 model, and the
// 104-bit accumulator covers that whole range exactly). So there is NO EMIN register and NO
// calibration; the wire layout and host driver share the generated regmap contract.
//
`default_nettype none

module decode_top #(
    // Set by the Vivado BD build from the propagated clk_wiz output frequency (one
    // checked-in regmap serves every fclk variant).
    parameter [31:0] CLK_HZ = 32'd0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS_W0:S_AXIS_W1:S_AXIS_W2:S_AXIS_W3:S_AXIS_ACTS:M_AXIS, ASSOCIATED_RESET s_axi_aresetn" *)
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

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W0 TDATA" *)
    input  wire [127:0] s_axis_w0_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W0 TKEEP" *)
    input  wire [15:0]  s_axis_w0_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W0 TVALID" *)
    input  wire         s_axis_w0_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W0 TREADY" *)
    output wire         s_axis_w0_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W0 TLAST" *)
    input  wire         s_axis_w0_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W1 TDATA" *)
    input  wire [127:0] s_axis_w1_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W1 TKEEP" *)
    input  wire [15:0]  s_axis_w1_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W1 TVALID" *)
    input  wire         s_axis_w1_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W1 TREADY" *)
    output wire         s_axis_w1_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W1 TLAST" *)
    input  wire         s_axis_w1_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W2 TDATA" *)
    input  wire [127:0] s_axis_w2_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W2 TKEEP" *)
    input  wire [15:0]  s_axis_w2_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W2 TVALID" *)
    input  wire         s_axis_w2_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W2 TREADY" *)
    output wire         s_axis_w2_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W2 TLAST" *)
    input  wire         s_axis_w2_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W3 TDATA" *)
    input  wire [127:0] s_axis_w3_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W3 TKEEP" *)
    input  wire [15:0]  s_axis_w3_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W3 TVALID" *)
    input  wire         s_axis_w3_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W3 TREADY" *)
    output wire         s_axis_w3_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_W3 TLAST" *)
    input  wire         s_axis_w3_tlast,

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
    localparam integer ROWS = 16;

    // Register offsets and RO reset values, generated from fpga/regmap/matmul.regmap.
    `include "matmul_regs.vh"

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    reg [15:0] num_q1_blocks_q;
    reg [15:0] num_rowblocks_q;
    reg [31:0] num_rows_q;
    reg [15:0] num_cols_q;
    reg [1:0]  weight_fmt_q;
    reg [1:0]  act_mode_q;
    reg [31:0] act_epoch_q;
    reg        start_strobe;

    // Fixed-point window floor: a constant of the f16 format (min contribution exponent
    // e_ws+e_as = -24 + -24), NOT a runtime register. Wired straight to the kernel; the
    // 104-bit accumulator (gemm_kernel ACC_W) covers the full f16 range with no calibration.
    localparam signed [7:0] EMIN_FLOOR = -8'sd48;
    reg        done_latched;
    reg [31:0] cycle_count_q;

    wire kernel_busy;
    wire kernel_done;
    wire activation_error;
    wire activation_valid;
    wire [31:0] loaded_act_epoch;
    wire [15:0] loaded_act_q1_blocks;
    wire [15:0] loaded_act_cols;
    wire [3:0] quantizer_status;
    wire q8_activation_abort;
    wire weight_tready;
    wire weight_tvalid = s_axis_w0_tvalid && s_axis_w1_tvalid &&
                         s_axis_w2_tvalid && s_axis_w3_tvalid;
    wire [ROWS*32-1:0] weight_tdata = {
        s_axis_w3_tdata,
        s_axis_w2_tdata,
        s_axis_w1_tdata,
        s_axis_w0_tdata
    };

    assign s_axis_w0_tready = weight_tready && s_axis_w1_tvalid && s_axis_w2_tvalid && s_axis_w3_tvalid;
    assign s_axis_w1_tready = weight_tready && s_axis_w0_tvalid && s_axis_w2_tvalid && s_axis_w3_tvalid;
    assign s_axis_w2_tready = weight_tready && s_axis_w0_tvalid && s_axis_w1_tvalid && s_axis_w3_tvalid;
    assign s_axis_w3_tready = weight_tready && s_axis_w0_tvalid && s_axis_w1_tvalid && s_axis_w2_tvalid;

    // COLS_MAX comes from the generated header (caps.cols_max), the same source the host's
    // mc_cols_max reads — never a literal. MAX_SUB_INDEX=512 sizes the acts BRAM for K up to
    // 512/q8_subblocks = 128 q1-blocks = 16384, the exact limit of the 104-bit accumulator
    // window. Must match the host's layout.max_sub_index.
    wire raw_activation_mode = act_mode_q == 2'd2;
    wire [63:0] native_acts_tdata;
    wire native_acts_tvalid;
    wire native_acts_tready;
    wire raw_acts_tready;
    wire kernel_acts_tready;

    q8_ingress u_q8_ingress (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_strobe),
        .raw_mode(raw_activation_mode),
        .num_q1_blocks(num_q1_blocks_q),
        .num_cols(num_cols_q),
        .s_axis_tdata(s_axis_acts_tdata),
        .s_axis_tvalid(s_axis_acts_tvalid),
        .s_axis_tready(raw_acts_tready),
        .s_axis_tlast(s_axis_acts_tlast),
        .m_axis_tdata(native_acts_tdata),
        .m_axis_tvalid(native_acts_tvalid),
        .m_axis_tready(native_acts_tready),
        .activation_abort(q8_activation_abort),
        .quantizer_status(quantizer_status)
    );

    wire [63:0] kernel_acts_tdata = raw_activation_mode ? native_acts_tdata : s_axis_acts_tdata;
    wire kernel_acts_tvalid = raw_activation_mode ? native_acts_tvalid : s_axis_acts_tvalid;
    assign native_acts_tready = raw_activation_mode && kernel_acts_tready;
    assign s_axis_acts_tready = raw_activation_mode ? raw_acts_tready : kernel_acts_tready;

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(MATMUL_COLS_MAX), .MAX_SUB_INDEX(512)) u_kernel (
        .clk(clk),
        .rst_n(rst_n),
        .start_kernel(start_strobe),
        .num_q1_blocks(num_q1_blocks_q),
        .num_rowblocks(num_rowblocks_q),
        .num_rows(num_rows_q),
        .num_cols(num_cols_q),
        .weight_fmt(weight_fmt_q),
        .act_mode(act_mode_q),
        .act_epoch(act_epoch_q),
        .activation_abort(q8_activation_abort),
        .emin(EMIN_FLOOR),
        .kernel_done(kernel_done),
        .activation_error(activation_error),
        .activation_valid(activation_valid),
        .loaded_act_epoch(loaded_act_epoch),
        .loaded_act_q1_blocks(loaded_act_q1_blocks),
        .loaded_act_cols(loaded_act_cols),
        .busy(kernel_busy),
        .s_axis_tdata(weight_tdata),
        .s_axis_tvalid(weight_tvalid),
        .s_axis_tready(weight_tready),
        .s_axis_acts_tdata(kernel_acts_tdata),
        .s_axis_acts_tvalid(kernel_acts_tvalid),
        .s_axis_acts_tready(kernel_acts_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .m_axis_tkeep(m_axis_tkeep),
        .dbg_state()
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

    // Performance counter bank (latched per run; counts only while busy). Beats: AXIS
    // transfers moved. Stalls: kernel ready for an input beat but starved, or had an output
    // beat the sink wasn't ready for. util = (CYCLES - max stall) / CYCLES.
    reg [31:0] w_stall_q, a_stall_q, r_stall_q;
    reg [31:0] w_beats_q, a_beats_q, r_beats_q;
    always @(posedge clk) begin
        if (!rst_n || start_strobe) begin
            w_stall_q <= 32'd0; a_stall_q <= 32'd0; r_stall_q <= 32'd0;
            w_beats_q <= 32'd0; a_beats_q <= 32'd0; r_beats_q <= 32'd0;
        end else if (kernel_busy) begin
            if (weight_tvalid && weight_tready)           w_beats_q <= w_beats_q + 32'd1;
            if (s_axis_acts_tvalid && s_axis_acts_tready) a_beats_q <= a_beats_q + 32'd1;
            if (m_axis_tvalid && m_axis_tready)           r_beats_q <= r_beats_q + 32'd1;
            if (weight_tready && !weight_tvalid)           w_stall_q <= w_stall_q + 32'd1;
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
            num_rows_q      <= 32'd0;
            num_cols_q      <= 16'd0;
            weight_fmt_q    <= 2'd1;
            act_mode_q      <= 2'd0;
            act_epoch_q     <= 32'd0;
            start_strobe    <= 1'b0;
        end else begin
            start_strobe <= 1'b0;

            awready_q <= write_accept;
            wready_q  <= write_accept;
            if (write_accept) awaddr_q <= s_axi_awaddr;

            if (write_commit)                  bvalid_q <= 1'b1;
            else if (bvalid_q && s_axi_bready) bvalid_q <= 1'b0;

            if (write_commit) begin
                case (awaddr_q[7:0])
                    MATMUL_OFF_CTRL[7:0]: begin
                        if (s_axi_wstrb[0] && s_axi_wdata[0])
                            start_strobe <= 1'b1;
                    end
                    MATMUL_OFF_NUM_Q1_BLOCKS[7:0]: begin
                        if (s_axi_wstrb[0]) num_q1_blocks_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_q1_blocks_q[15:8] <= s_axi_wdata[15:8];
                    end
                    MATMUL_OFF_NUM_ROWBLOCKS[7:0]: begin
                        if (s_axi_wstrb[0]) num_rowblocks_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_rowblocks_q[15:8] <= s_axi_wdata[15:8];
                    end
                    MATMUL_OFF_NUM_COLS[7:0]: begin
                        if (s_axi_wstrb[0]) num_cols_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_cols_q[15:8] <= s_axi_wdata[15:8];
                    end
                    MATMUL_OFF_NUM_ROWS[7:0]: begin
                        if (s_axi_wstrb[0]) num_rows_q[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_rows_q[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) num_rows_q[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) num_rows_q[31:24] <= s_axi_wdata[31:24];
                    end
                    MATMUL_OFF_WEIGHT_FMT[7:0]: begin
                        if (s_axi_wstrb[0]) weight_fmt_q <= s_axi_wdata[1:0];
                    end
                    MATMUL_OFF_ACT_MODE[7:0]: begin
                        if (s_axi_wstrb[0]) act_mode_q <= s_axi_wdata[1:0];
                    end
                    MATMUL_OFF_ACT_EPOCH[7:0]: begin
                        if (s_axi_wstrb[0]) act_epoch_q[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) act_epoch_q[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) act_epoch_q[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) act_epoch_q[31:24] <= s_axi_wdata[31:24];
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
                case (s_axi_araddr[7:0])
                    MATMUL_OFF_ID[7:0]:            rdata_q <= MATMUL_RST_ID;
                    MATMUL_OFF_VERSION[7:0]:       rdata_q <= MATMUL_RST_VERSION;
                    MATMUL_OFF_STATUS[7:0]:        rdata_q <= {30'd0, done_latched, kernel_busy};
                    MATMUL_OFF_NUM_Q1_BLOCKS[7:0]: rdata_q <= {16'd0, num_q1_blocks_q};
                    MATMUL_OFF_NUM_ROWBLOCKS[7:0]: rdata_q <= {16'd0, num_rowblocks_q};
                    MATMUL_OFF_NUM_COLS[7:0]:      rdata_q <= {16'd0, num_cols_q};
                    MATMUL_OFF_NUM_ROWS[7:0]:      rdata_q <= num_rows_q;
                    MATMUL_OFF_WEIGHT_FMT[7:0]:    rdata_q <= {30'd0, weight_fmt_q};
                    MATMUL_OFF_ACT_MODE[7:0]:      rdata_q <= {30'd0, act_mode_q};
                    MATMUL_OFF_ACT_EPOCH[7:0]:     rdata_q <= act_epoch_q;
                    MATMUL_OFF_ACT_STATE[7:0]:     rdata_q <= {30'd0, activation_error, activation_valid};
                    MATMUL_OFF_LOADED_EPOCH[7:0]:  rdata_q <= loaded_act_epoch;
                    MATMUL_OFF_LOADED_Q1_BLOCKS[7:0]: rdata_q <= {16'd0, loaded_act_q1_blocks};
                    MATMUL_OFF_LOADED_COLS[7:0]:   rdata_q <= {16'd0, loaded_act_cols};
                    MATMUL_OFF_QUANT_STATUS[7:0]:  rdata_q <= {28'd0, quantizer_status};
                    MATMUL_OFF_CYCLES[7:0]:        rdata_q <= cycle_count_q;
                    MATMUL_OFF_ROWS[7:0]:          rdata_q <= MATMUL_RST_ROWS;
                    MATMUL_OFF_CLK_HZ[7:0]:        rdata_q <= CLK_HZ;
                    MATMUL_OFF_W_STALL[7:0]:       rdata_q <= w_stall_q;
                    MATMUL_OFF_A_STALL[7:0]:       rdata_q <= a_stall_q;
                    MATMUL_OFF_R_STALL[7:0]:       rdata_q <= r_stall_q;
                    MATMUL_OFF_W_BEATS[7:0]:       rdata_q <= w_beats_q;
                    MATMUL_OFF_A_BEATS[7:0]:       rdata_q <= a_beats_q;
                    MATMUL_OFF_R_BEATS[7:0]:       rdata_q <= r_beats_q;
                    MATMUL_OFF_WEIGHT_PORTS[7:0]:  rdata_q <= MATMUL_RST_WEIGHT_PORTS;
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
        s_axis_w0_tkeep,
        s_axis_w0_tlast,
        s_axis_w1_tkeep,
        s_axis_w1_tlast,
        s_axis_w2_tkeep,
        s_axis_w2_tlast,
        s_axis_w3_tkeep,
        s_axis_w3_tlast,
        s_axis_acts_tkeep,
        1'b0
    };

endmodule
