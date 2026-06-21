// flash_top - AXI-Lite + DMA wrapper for flash_kernel.
//
// The host writes the shape config (HEAD_DIM_Q/V, N_HEADS, N_KV, N_TOKENS, SCALE)
// then strobes CTRL.start. Q/K/V/mask stream in from DMA, O streams out to DMA. A
// per-run counter bank records Q/K/V/O beats + K/V/O stalls. Mirrors matmul_top; the
// register decode comes from the generated flash_regs.vh so RTL and the Zig driver
// never drift.

`default_nettype none

module flash_top #(
    parameter [31:0]  CLK_HZ       = 32'd0,
    parameter integer HEAD_DIM_MAX = 128
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS_Q:S_AXIS_K:S_AXIS_V:S_AXIS_MASK:M_AXIS_O, ASSOCIATED_RESET s_axi_aresetn" *)
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

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_Q TDATA" *)
    input  wire [255:0] s_axis_q_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_Q TKEEP" *)
    input  wire [31:0]  s_axis_q_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_Q TLAST" *)
    input  wire         s_axis_q_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_Q TVALID" *)
    input  wire         s_axis_q_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_Q TREADY" *)
    output wire         s_axis_q_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_K TDATA" *)
    input  wire [127:0] s_axis_k_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_K TKEEP" *)
    input  wire [15:0]  s_axis_k_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_K TLAST" *)
    input  wire         s_axis_k_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_K TVALID" *)
    input  wire         s_axis_k_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_K TREADY" *)
    output wire         s_axis_k_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_V TDATA" *)
    input  wire [127:0] s_axis_v_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_V TKEEP" *)
    input  wire [15:0]  s_axis_v_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_V TLAST" *)
    input  wire         s_axis_v_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_V TVALID" *)
    input  wire         s_axis_v_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_V TREADY" *)
    output wire         s_axis_v_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_MASK TDATA" *)
    input  wire [15:0]  s_axis_mask_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_MASK TKEEP" *)
    input  wire [1:0]   s_axis_mask_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_MASK TLAST" *)
    input  wire         s_axis_mask_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_MASK TVALID" *)
    input  wire         s_axis_mask_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_MASK TREADY" *)
    output wire         s_axis_mask_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_O TDATA" *)
    output wire [255:0] m_axis_o_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_O TKEEP" *)
    output wire [31:0]  m_axis_o_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_O TLAST" *)
    output wire         m_axis_o_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_O TVALID" *)
    output wire         m_axis_o_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_O TREADY" *)
    input  wire         m_axis_o_tready
);
    `include "flash_regs.vh"

    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    reg [15:0] head_dim_q_q, head_dim_v_q, n_heads_q, n_head_kv_q, head_ratio_q, n_kv_q, n_tokens_q;
    reg [31:0] scale_q;
    reg        start_strobe, done_latched;
    reg [31:0] cycle_count_q;

    wire kernel_busy, kernel_done;
    wire q_tready_w, k_tready_w, v_tready_w, mask_tready_w, o_tvalid_w, o_tlast_w;
    wire [31:0] o_tkeep_w;

    flash_kernel #(.HEAD_DIM_MAX(HEAD_DIM_MAX), .LANES(FLASH_LANES)) u_kernel (
        .clk(clk), .rst_n(rst_n),
        .start(start_strobe),
        .head_dim_q(head_dim_q_q), .head_dim_v(head_dim_v_q),
        .n_heads(n_heads_q), .n_head_kv(n_head_kv_q), .head_ratio(head_ratio_q),
        .n_kv(n_kv_q), .n_tokens(n_tokens_q), .scale(scale_q),
        .busy(kernel_busy), .done(kernel_done),
        .q_tdata(s_axis_q_tdata),       .q_tvalid(s_axis_q_tvalid),       .q_tready(q_tready_w),
        .k_tdata(s_axis_k_tdata),       .k_tvalid(s_axis_k_tvalid),       .k_tready(k_tready_w),
        .v_tdata(s_axis_v_tdata),       .v_tvalid(s_axis_v_tvalid),       .v_tready(v_tready_w),
        .mask_tdata(s_axis_mask_tdata), .mask_tvalid(s_axis_mask_tvalid), .mask_tready(mask_tready_w),
        .o_tdata(m_axis_o_tdata),       .o_tvalid(o_tvalid_w),            .o_tready(m_axis_o_tready),
        .q_tlast(s_axis_q_tlast),       .q_tkeep(s_axis_q_tkeep),
        .k_tlast(s_axis_k_tlast),       .k_tkeep(s_axis_k_tkeep),
        .v_tlast(s_axis_v_tlast),       .v_tkeep(s_axis_v_tkeep),
        .mask_tlast(s_axis_mask_tlast), .mask_tkeep(s_axis_mask_tkeep),
        .o_tlast(o_tlast_w),            .o_tkeep(o_tkeep_w)
    );

    assign s_axis_q_tready    = q_tready_w;
    assign s_axis_k_tready    = k_tready_w;
    assign s_axis_v_tready    = v_tready_w;
    assign s_axis_mask_tready = mask_tready_w;
    assign m_axis_o_tvalid    = o_tvalid_w;
    assign m_axis_o_tlast     = o_tlast_w;
    assign m_axis_o_tkeep     = o_tkeep_w;

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

    // Counter bank (latched per run, cleared on start). Beats = AXIS transfers moved;
    // stalls = busy cycles a stream was needed but absent (K/V) or backpressured (O).
    reg [31:0] q_beats_q, k_beats_q, k_stall_q, v_beats_q, v_stall_q, o_beats_q, o_stall_q;
    always @(posedge clk) begin
        if (!rst_n || start_strobe) begin
            q_beats_q <= 0; k_beats_q <= 0; k_stall_q <= 0;
            v_beats_q <= 0; v_stall_q <= 0; o_beats_q <= 0; o_stall_q <= 0;
        end else if (kernel_busy) begin
            if (s_axis_q_tvalid    && q_tready_w)    q_beats_q <= q_beats_q + 32'd1;
            if (s_axis_k_tvalid    && k_tready_w)    k_beats_q <= k_beats_q + 32'd1;
            if (s_axis_v_tvalid    && v_tready_w)    v_beats_q <= v_beats_q + 32'd1;
            if (o_tvalid_w         && m_axis_o_tready) o_beats_q <= o_beats_q + 32'd1;
            if (k_tready_w && !s_axis_k_tvalid)      k_stall_q <= k_stall_q + 32'd1;
            if (v_tready_w && !s_axis_v_tvalid)      v_stall_q <= v_stall_q + 32'd1;
            if (o_tvalid_w && !m_axis_o_tready)      o_stall_q <= o_stall_q + 32'd1;
        end
    end

    // ---- AXI-Lite write ----
    reg awready_q, wready_q, bvalid_q;
    reg [7:0] awaddr_q;
    wire write_accept = !awready_q && !wready_q && s_axi_awvalid && s_axi_wvalid && !bvalid_q;
    wire write_commit = awready_q && wready_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            awready_q <= 1'b0; wready_q <= 1'b0; bvalid_q <= 1'b0; awaddr_q <= 8'd0;
            head_dim_q_q <= 0; head_dim_v_q <= 0; n_heads_q <= 0; n_head_kv_q <= 0; head_ratio_q <= 0;
            n_kv_q <= 0; n_tokens_q <= 0; scale_q <= 32'd0; start_strobe <= 1'b0;
        end else begin
            start_strobe <= 1'b0;
            awready_q <= write_accept;
            wready_q  <= write_accept;
            if (write_accept) awaddr_q <= s_axi_awaddr;
            if (write_commit)                  bvalid_q <= 1'b1;
            else if (bvalid_q && s_axi_bready) bvalid_q <= 1'b0;

            if (write_commit) begin
                case (awaddr_q[7:0])
                    FLASH_OFF_CTRL[7:0]:       if (s_axi_wstrb[0] && s_axi_wdata[0]) start_strobe <= 1'b1;
                    FLASH_OFF_HEAD_DIM_Q[7:0]: begin if (s_axi_wstrb[0]) head_dim_q_q[7:0] <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) head_dim_q_q[15:8] <= s_axi_wdata[15:8]; end
                    FLASH_OFF_HEAD_DIM_V[7:0]: begin if (s_axi_wstrb[0]) head_dim_v_q[7:0] <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) head_dim_v_q[15:8] <= s_axi_wdata[15:8]; end
                    FLASH_OFF_N_HEADS[7:0]:    begin if (s_axi_wstrb[0]) n_heads_q[7:0]    <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) n_heads_q[15:8]    <= s_axi_wdata[15:8]; end
                    FLASH_OFF_N_HEAD_KV[7:0]:  begin if (s_axi_wstrb[0]) n_head_kv_q[7:0]  <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) n_head_kv_q[15:8]  <= s_axi_wdata[15:8]; end
                    FLASH_OFF_HEAD_RATIO[7:0]: begin if (s_axi_wstrb[0]) head_ratio_q[7:0] <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) head_ratio_q[15:8] <= s_axi_wdata[15:8]; end
                    FLASH_OFF_N_KV[7:0]:       begin if (s_axi_wstrb[0]) n_kv_q[7:0]       <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) n_kv_q[15:8]       <= s_axi_wdata[15:8]; end
                    FLASH_OFF_N_TOKENS[7:0]:   begin if (s_axi_wstrb[0]) n_tokens_q[7:0]   <= s_axi_wdata[7:0]; if (s_axi_wstrb[1]) n_tokens_q[15:8]   <= s_axi_wdata[15:8]; end
                    FLASH_OFF_SCALE[7:0]: begin
                        if (s_axi_wstrb[0]) scale_q[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) scale_q[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) scale_q[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) scale_q[31:24] <= s_axi_wdata[31:24];
                    end
                    default: ;
                endcase
            end
        end
    end

    // ---- AXI-Lite read ----
    reg        arready_q, rvalid_q;
    reg [31:0] rdata_q;
    wire read_accept = !arready_q && s_axi_arvalid && !rvalid_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            arready_q <= 1'b0; rvalid_q <= 1'b0; rdata_q <= 32'd0;
        end else begin
            arready_q <= read_accept;
            if (read_accept) begin
                rvalid_q <= 1'b1;
                case (s_axi_araddr[7:0])
                    FLASH_OFF_ID[7:0]:         rdata_q <= FLASH_RST_ID;
                    FLASH_OFF_VERSION[7:0]:    rdata_q <= FLASH_RST_VERSION;
                    FLASH_OFF_STATUS[7:0]:     rdata_q <= {30'd0, done_latched, kernel_busy};
                    FLASH_OFF_HEAD_DIM_Q[7:0]: rdata_q <= {16'd0, head_dim_q_q};
                    FLASH_OFF_HEAD_DIM_V[7:0]: rdata_q <= {16'd0, head_dim_v_q};
                    FLASH_OFF_N_HEADS[7:0]:    rdata_q <= {16'd0, n_heads_q};
                    FLASH_OFF_N_HEAD_KV[7:0]:  rdata_q <= {16'd0, n_head_kv_q};
                    FLASH_OFF_HEAD_RATIO[7:0]: rdata_q <= {16'd0, head_ratio_q};
                    FLASH_OFF_N_KV[7:0]:       rdata_q <= {16'd0, n_kv_q};
                    FLASH_OFF_N_TOKENS[7:0]:   rdata_q <= {16'd0, n_tokens_q};
                    FLASH_OFF_SCALE[7:0]:      rdata_q <= scale_q;
                    FLASH_OFF_CYCLES[7:0]:     rdata_q <= cycle_count_q;
                    FLASH_OFF_CLK_HZ[7:0]:     rdata_q <= CLK_HZ;
                    FLASH_OFF_LANES[7:0]:      rdata_q <= FLASH_RST_LANES;
                    FLASH_OFF_Q_BEATS[7:0]:    rdata_q <= q_beats_q;
                    FLASH_OFF_K_BEATS[7:0]:    rdata_q <= k_beats_q;
                    FLASH_OFF_K_STALL[7:0]:    rdata_q <= k_stall_q;
                    FLASH_OFF_V_BEATS[7:0]:    rdata_q <= v_beats_q;
                    FLASH_OFF_V_STALL[7:0]:    rdata_q <= v_stall_q;
                    FLASH_OFF_O_BEATS[7:0]:    rdata_q <= o_beats_q;
                    FLASH_OFF_O_STALL[7:0]:    rdata_q <= o_stall_q;
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

    wire _unused = &{1'b0, s_axi_awprot, s_axi_arprot};
endmodule
