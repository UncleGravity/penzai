// seq_top - the seq.v descriptor executor, BD-ready. Wires the cosim-green seq_core (the
// {WRITE|WAIT|END} FSM) to its three bus edges:
//   S_AXI       AXI-Lite SLAVE  - the PS programs DESC_BASE/COUNT, strobes go, polls STATUS
//   M_AXI_REG   AXI-Lite MASTER - the replay into sc_ctrl (seq_reg_master)
//   M_AXI_DESC  AXI4   read MASTER - descriptor fetch from DRAM (seq_desc_reader)
// One descriptor run = PS writes the regs + go; seq_core walks the entries (fetch via M_AXI_DESC,
// replay/poll via M_AXI_REG), no PS in the inner loop; PS polls STATUS.done once.
// Cosim-gated end-to-end (test-rtl-seq-top): control slave + core + both masters together.

`default_nettype none

module seq_top #(
    parameter integer DESC_ADDR_W = 40,   // M_AXI_DESC (DRAM) address width
    parameter integer REG_ADDR_W  = 32,   // M_AXI_REG (sc_ctrl) address width
    parameter integer COUNT_W     = 16,
    parameter integer POLL_TIMEOUT = 2000000  // > a real op (~100us ~= 30k cyc @300MHz)
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI_REG:M_AXI_DESC, ASSOCIATED_RESET rst_n" *)
    input  wire                    clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                    rst_n,

    // ---- S_AXI: control AXI-Lite slave (PS-facing) ----
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [7:0]              s_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                    s_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output reg                     s_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]             s_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]              s_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                    s_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output reg                     s_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output reg  [1:0]              s_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output reg                     s_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                    s_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [7:0]              s_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire                    s_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output reg                     s_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg  [31:0]             s_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg  [1:0]              s_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg                     s_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire                    s_rready,

    // ---- M_AXI_REG: AXI-Lite replay master (-> sc_ctrl) ----
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG AWADDR" *)
    output wire [REG_ADDR_W-1:0]   reg_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG AWVALID" *)
    output wire                    reg_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG AWREADY" *)
    input  wire                    reg_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG WDATA" *)
    output wire [31:0]             reg_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG WSTRB" *)
    output wire [3:0]              reg_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG WVALID" *)
    output wire                    reg_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG WREADY" *)
    input  wire                    reg_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG BRESP" *)
    input  wire [1:0]              reg_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG BVALID" *)
    input  wire                    reg_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG BREADY" *)
    output wire                    reg_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG ARADDR" *)
    output wire [REG_ADDR_W-1:0]   reg_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG ARVALID" *)
    output wire                    reg_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG ARREADY" *)
    input  wire                    reg_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG RDATA" *)
    input  wire [31:0]             reg_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG RRESP" *)
    input  wire [1:0]              reg_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG RVALID" *)
    input  wire                    reg_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_REG RREADY" *)
    output wire                    reg_rready,

    // ---- M_AXI_DESC: AXI4 read master (descriptor fetch from DRAM) ----
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARADDR" *)
    output wire [DESC_ADDR_W-1:0]  desc_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARLEN" *)
    output wire [7:0]              desc_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARSIZE" *)
    output wire [2:0]              desc_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARBURST" *)
    output wire [1:0]              desc_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARVALID" *)
    output wire                    desc_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC ARREADY" *)
    input  wire                    desc_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC RDATA" *)
    input  wire [127:0]            desc_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC RRESP" *)
    input  wire [1:0]              desc_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC RLAST" *)
    input  wire                    desc_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC RVALID" *)
    input  wire                    desc_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_DESC RREADY" *)
    output wire                    desc_rready
);
    // ---- control registers (S_AXI) ----
    localparam [7:0] OFF_DESC_BASE_LO = 8'h00,  // RW: descriptor region base [31:0]
                     OFF_DESC_BASE_HI = 8'h04,  // RW: descriptor region base [DESC_ADDR_W-1:32]
                     OFF_DESC_COUNT   = 8'h08,  // RW: number of entries in the run
                     OFF_CTRL         = 8'h0C,  // W : bit0 = go strobe
                     OFF_STATUS       = 8'h10,  // RO: {.., err_timeout[2], done[1], busy[0]}
                     OFF_ERR_INDEX    = 8'h14;  // RO: entry index that timed out

    reg [31:0]        desc_base_lo, desc_base_hi;
    reg [COUNT_W-1:0] desc_count_q;
    reg               go_strobe;

    wire                  busy, done, err_timeout;
    wire [COUNT_W-1:0]    err_index;

    // ---- S_AXI slave (decode_top-style: paired aw/w accept, single-flight) ----
    reg awr, wwr;
    reg [7:0] awaddr_q;
    wire write_accept = !awr && !wwr && s_awvalid && s_wvalid && !s_bvalid;
    wire write_commit = awr && wwr;
    always @(posedge clk) begin
        if (!rst_n) begin
            awr <= 1'b0; wwr <= 1'b0; awaddr_q <= 8'd0;
            s_awready <= 1'b0; s_wready <= 1'b0; s_bvalid <= 1'b0; s_bresp <= 2'b00;
            desc_base_lo <= 32'd0; desc_base_hi <= 32'd0; desc_count_q <= {COUNT_W{1'b0}};
            go_strobe <= 1'b0;
        end else begin
            go_strobe <= 1'b0; // 1-cycle strobe
            awr <= write_accept; wwr <= write_accept;
            s_awready <= write_accept; s_wready <= write_accept;
            if (write_accept) awaddr_q <= s_awaddr;
            if (write_commit) s_bvalid <= 1'b1; else if (s_bvalid && s_bready) s_bvalid <= 1'b0;
            if (write_commit) begin
                case (awaddr_q)
                    OFF_DESC_BASE_LO: desc_base_lo <= s_wdata;
                    OFF_DESC_BASE_HI: desc_base_hi <= s_wdata;
                    OFF_DESC_COUNT:   desc_count_q <= s_wdata[COUNT_W-1:0];
                    OFF_CTRL:         if (s_wdata[0]) go_strobe <= 1'b1;
                    default: ;
                endcase
            end
        end
    end

    reg arr;
    always @(posedge clk) begin
        if (!rst_n) begin
            arr <= 1'b0; s_arready <= 1'b0; s_rvalid <= 1'b0; s_rdata <= 32'd0; s_rresp <= 2'b00;
        end else begin
            s_arready <= !arr && s_arvalid && !s_rvalid;
            if (!arr && s_arvalid && !s_rvalid) begin
                arr <= 1'b1; s_rvalid <= 1'b1;
                case (s_araddr)
                    OFF_DESC_BASE_LO: s_rdata <= desc_base_lo;
                    OFF_DESC_BASE_HI: s_rdata <= desc_base_hi;
                    OFF_DESC_COUNT:   s_rdata <= {{(32-COUNT_W){1'b0}}, desc_count_q};
                    OFF_STATUS:       s_rdata <= {29'd0, err_timeout, done, busy};
                    OFF_ERR_INDEX:    s_rdata <= {{(32-COUNT_W){1'b0}}, err_index};
                    default:          s_rdata <= 32'd0;
                endcase
            end else if (s_rvalid && s_rready) begin
                s_rvalid <= 1'b0; arr <= 1'b0;
            end
        end
    end

    // ---- core <-> adapters ----
    wire                  desc_req, desc_gnt;
    wire [COUNT_W-1:0]    desc_idx;
    wire [127:0]          desc_data;
    wire                  reg_req, reg_we, reg_gnt;
    wire [REG_ADDR_W-1:0] reg_addr;
    wire [31:0]           reg_wdata_core, reg_rdata_core;

    seq_core #(.ADDR_W(REG_ADDR_W), .COUNT_W(COUNT_W), .POLL_TIMEOUT(POLL_TIMEOUT)) u_core (
        .clk(clk), .rst_n(rst_n),
        .go(go_strobe), .desc_count(desc_count_q),
        .busy(busy), .done(done), .err_timeout(err_timeout), .err_index(err_index),
        .desc_req(desc_req), .desc_idx(desc_idx), .desc_gnt(desc_gnt), .desc_data(desc_data),
        .reg_req(reg_req), .reg_we(reg_we), .reg_addr(reg_addr), .reg_wdata(reg_wdata_core),
        .reg_gnt(reg_gnt), .reg_rdata(reg_rdata_core)
    );

    seq_desc_reader #(.ADDR_W(DESC_ADDR_W), .COUNT_W(COUNT_W)) u_desc (
        .clk(clk), .rst_n(rst_n),
        .desc_base({desc_base_hi[DESC_ADDR_W-32-1:0], desc_base_lo}),
        .desc_req(desc_req), .desc_idx(desc_idx), .desc_gnt(desc_gnt), .desc_data(desc_data),
        .m_araddr(desc_araddr), .m_arlen(desc_arlen), .m_arsize(desc_arsize), .m_arburst(desc_arburst),
        .m_arvalid(desc_arvalid), .m_arready(desc_arready),
        .m_rdata(desc_rdata), .m_rresp(desc_rresp), .m_rlast(desc_rlast), .m_rvalid(desc_rvalid), .m_rready(desc_rready)
    );

    seq_reg_master #(.ADDR_W(REG_ADDR_W)) u_reg (
        .clk(clk), .rst_n(rst_n),
        .req(reg_req), .we(reg_we), .addr(reg_addr), .wdata(reg_wdata_core),
        .gnt(reg_gnt), .rdata(reg_rdata_core),
        .m_awaddr(reg_awaddr), .m_awvalid(reg_awvalid), .m_awready(reg_awready),
        .m_wdata(reg_wdata), .m_wstrb(reg_wstrb), .m_wvalid(reg_wvalid), .m_wready(reg_wready),
        .m_bresp(reg_bresp), .m_bvalid(reg_bvalid), .m_bready(reg_bready),
        .m_araddr(reg_araddr), .m_arvalid(reg_arvalid), .m_arready(reg_arready),
        .m_rdata(reg_rdata), .m_rresp(reg_rresp), .m_rvalid(reg_rvalid), .m_rready(reg_rready)
    );

    wire _unused = &{1'b0, s_wstrb};
endmodule
