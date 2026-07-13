// seq_top - the seq.v command executor, BD-ready (docs/plan-seq-impl-v2.md, v2.1). Wires the
// cosim-green seq_core ({WRITE|WAIT} FSM) to its two bus edges:
//   S_AXI       AXI-Lite SLAVE  - PS loads the command BRAM (CMD window), programs
//                                 RUN_START/RUN_COUNT, strobes go/abort, polls STATUS
//   M_AXI_REG   AXI-Lite MASTER - the replay into sc_ctrl (seq_reg_master)
// The command lives IN this module's BRAM, PS-written through the control slave: no DRAM, no
// AXI4 read master, no cache coherency — the v1 brick class is unrepresentable. One run = PS
// loads entries once, writes RUN_START/RUN_COUNT + go, polls STATUS.done once. RUN_START lets
// many runs (segments) stay resident and be kicked by index — the resident-program endgame.
//
// ABORT is a synchronous reset of the executor core, the BRAM read adapter, and the AXI-Lite
// master: always reclaimable from the PS. Caveat (documented, accepted): resetting the master
// mid-handshake violates AXI if the slave is alive-but-slow; the abort path exists for slaves
// that are dead, where the bus is already lost. The control slave itself never resets on abort.
//
// No CMD read-back: the host keeps its own copy of every stream it loads; delivery is proven
// once by the bring-up dry-run gate. Fewer ports, simpler BRAM.

`default_nettype none

module seq_top #(
    parameter integer REG_ADDR_W       = 32,       // M_AXI_REG (sc_ctrl) address width
    parameter integer COUNT_W          = 16,
    parameter integer POLL_TIMEOUT     = 2000000,  // > a real op (~100us ~= 30k cyc @300MHz)
    parameter integer WATCHDOG_TIMEOUT = 8000000,  // ~27ms @300MHz without a single gnt = dead bus
    // log2(command BRAM entries). 11 -> 2048 entries; with the CMD window at 0x8000 this fills
    // the 64 KiB AXI window exactly. Widening it means widening the BD window + S_ADDR_W too.
    parameter integer CMD_DEPTH_LOG2   = 11
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI_REG, ASSOCIATED_RESET rst_n" *)
    input  wire                    clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                    rst_n,

    // ---- S_AXI: control AXI-Lite slave (PS-facing) ----
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [15:0]             s_awaddr,
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
    input  wire [15:0]             s_araddr,
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
    output wire                    reg_rready
);
    localparam integer CMD_DEPTH = 1 << CMD_DEPTH_LOG2;

    // ---- control registers (S_AXI) — host mirror in device/pl/seq.zig `ctrl` ----
    localparam [15:0] OFF_RUN_START = 16'h00,  // RW: entry index the run begins at
                      OFF_RUN_COUNT = 16'h04,  // RW: entries to execute
                      OFF_CTRL      = 16'h08,  // W : bit0 = go, bit1 = abort
                      OFF_STATUS    = 16'h0C,  // RO: {err_watchdog[3], err_timeout[2], done[1], busy[0]}
                      OFF_ERR_INDEX = 16'h10;  // RO: entry index at the fault

    reg [COUNT_W-1:0] run_start, run_count;
    reg               go_strobe, abort_strobe;

    wire               busy, done, err_timeout, err_watchdog;
    wire [COUNT_W-1:0] err_index;

    // Abort = synchronous reset of everything downstream of the control slave. The slave
    // itself stays alive (the PS must always be able to poll/abort again).
    wire core_rst_n = rst_n && !abort_strobe;

    // ---- command BRAM: CMD_DEPTH x 128b as four 32b lanes (lane = entry word) ----
    // Write port: S_AXI CMD window (addr[15]=1, entry idx = addr[14:4], lane = addr[3:2]).
    // Read port: the executor, at run_start + desc_idx. Simple dual-port, no read-back.
    reg [31:0] cmd_w0 [0:CMD_DEPTH-1];  // tag
    reg [31:0] cmd_w1 [0:CMD_DEPTH-1];  // addr
    reg [31:0] cmd_w2 [0:CMD_DEPTH-1];  // a
    reg [31:0] cmd_w3 [0:CMD_DEPTH-1];  // b

    // ---- S_AXI write path (decode_top-style: paired aw/w accept, single-flight) ----
    reg awr, wwr;
    reg [15:0] awaddr_q;
    wire write_accept = !awr && !wwr && s_awvalid && s_wvalid && !s_bvalid;
    wire write_commit = awr && wwr;
    wire        cmd_sel  = awaddr_q[15];
    wire [CMD_DEPTH_LOG2-1:0] cmd_widx = awaddr_q[4 +: CMD_DEPTH_LOG2];
    wire [1:0]  cmd_lane = awaddr_q[3:2];

    always @(posedge clk) begin
        if (!rst_n) begin
            awr <= 1'b0; wwr <= 1'b0; awaddr_q <= 16'd0;
            s_awready <= 1'b0; s_wready <= 1'b0; s_bvalid <= 1'b0; s_bresp <= 2'b00;
            run_start <= {COUNT_W{1'b0}}; run_count <= {COUNT_W{1'b0}};
            go_strobe <= 1'b0; abort_strobe <= 1'b0;
        end else begin
            go_strobe <= 1'b0; abort_strobe <= 1'b0; // 1-cycle strobes
            awr <= write_accept; wwr <= write_accept;
            s_awready <= write_accept; s_wready <= write_accept;
            if (write_accept) awaddr_q <= s_awaddr;
            if (write_commit) s_bvalid <= 1'b1; else if (s_bvalid && s_bready) s_bvalid <= 1'b0;
            if (write_commit) begin
                if (cmd_sel) begin
                    case (cmd_lane)
                        2'd0: cmd_w0[cmd_widx] <= s_wdata;
                        2'd1: cmd_w1[cmd_widx] <= s_wdata;
                        2'd2: cmd_w2[cmd_widx] <= s_wdata;
                        2'd3: cmd_w3[cmd_widx] <= s_wdata;
                    endcase
                end else begin
                    case (awaddr_q)
                        OFF_RUN_START: run_start <= s_wdata[COUNT_W-1:0];
                        OFF_RUN_COUNT: run_count <= s_wdata[COUNT_W-1:0];
                        OFF_CTRL: begin
                            if (s_wdata[0]) go_strobe <= 1'b1;
                            if (s_wdata[1]) abort_strobe <= 1'b1;
                        end
                        default: ;
                    endcase
                end
            end
        end
    end

    // ---- S_AXI read path (control regs only; CMD region reads as 0) ----
    reg arr;
    always @(posedge clk) begin
        if (!rst_n) begin
            arr <= 1'b0; s_arready <= 1'b0; s_rvalid <= 1'b0; s_rdata <= 32'd0; s_rresp <= 2'b00;
        end else begin
            s_arready <= !arr && s_arvalid && !s_rvalid;
            if (!arr && s_arvalid && !s_rvalid) begin
                arr <= 1'b1; s_rvalid <= 1'b1;
                case (s_araddr)
                    OFF_RUN_START: s_rdata <= {{(32-COUNT_W){1'b0}}, run_start};
                    OFF_RUN_COUNT: s_rdata <= {{(32-COUNT_W){1'b0}}, run_count};
                    OFF_STATUS:    s_rdata <= {28'd0, err_watchdog, err_timeout, done, busy};
                    OFF_ERR_INDEX: s_rdata <= {{(32-COUNT_W){1'b0}}, err_index};
                    default:       s_rdata <= 32'd0;
                endcase
            end else if (s_rvalid && s_rready) begin
                s_rvalid <= 1'b0; arr <= 1'b0;
            end
        end
    end

    // ---- executor core + BRAM read adapter + AXI-Lite master ----
    wire                  desc_req, desc_gnt_w;
    wire [COUNT_W-1:0]    desc_idx;
    wire                  reg_req, reg_we, reg_gnt;
    wire [REG_ADDR_W-1:0] reg_addr;
    wire [31:0]           reg_wdata_core, reg_rdata_core;

    // Snapshot the segment base with an accepted run. Software may prepare the next run's
    // control registers while this one is active without redirecting in-flight fetches.
    reg [COUNT_W-1:0] active_start;

    // BRAM sync read at active_start + desc_idx; gnt two cycles after req (read, then present).
    // Same one-pulse-per-req `armed` handshake as seq_reg_master (req held until gnt, dropped
    // a cycle later).
    wire [CMD_DEPTH_LOG2-1:0] rd_idx =
        active_start[CMD_DEPTH_LOG2-1:0] + desc_idx[CMD_DEPTH_LOG2-1:0];
    reg        rd_armed, rd_pending, desc_gnt_q;
    reg [31:0] dq0, dq1, dq2, dq3;
    always @(posedge clk) begin
        if (!core_rst_n) begin
            active_start <= {COUNT_W{1'b0}};
            rd_armed <= 1'b1; rd_pending <= 1'b0; desc_gnt_q <= 1'b0;
            dq0 <= 32'd0; dq1 <= 32'd0; dq2 <= 32'd0; dq3 <= 32'd0;
        end else begin
            if (go_strobe && !busy) active_start <= run_start;
            desc_gnt_q <= rd_pending;   // data registered last cycle, grant it now
            rd_pending <= 1'b0;
            if (!desc_req) rd_armed <= 1'b1;
            else if (rd_armed) begin
                rd_armed <= 1'b0; rd_pending <= 1'b1;
            end
            dq0 <= cmd_w0[rd_idx]; dq1 <= cmd_w1[rd_idx];
            dq2 <= cmd_w2[rd_idx]; dq3 <= cmd_w3[rd_idx];
        end
    end
    assign desc_gnt_w = desc_gnt_q;

    seq_core #(
        .ADDR_W(REG_ADDR_W), .COUNT_W(COUNT_W),
        .POLL_TIMEOUT(POLL_TIMEOUT), .WATCHDOG_TIMEOUT(WATCHDOG_TIMEOUT)
    ) u_core (
        .clk(clk), .rst_n(core_rst_n),
        .go(go_strobe), .desc_count(run_count),
        .busy(busy), .done(done), .err_timeout(err_timeout), .err_watchdog(err_watchdog),
        .err_index(err_index),
        .desc_req(desc_req), .desc_idx(desc_idx), .desc_gnt(desc_gnt_w),
        .desc_data({dq3, dq2, dq1, dq0}),
        .reg_req(reg_req), .reg_we(reg_we), .reg_addr(reg_addr), .reg_wdata(reg_wdata_core),
        .reg_gnt(reg_gnt), .reg_rdata(reg_rdata_core)
    );

    seq_reg_master #(.ADDR_W(REG_ADDR_W)) u_reg (
        .clk(clk), .rst_n(core_rst_n),
        .req(reg_req), .we(reg_we), .addr(reg_addr), .wdata(reg_wdata_core),
        .gnt(reg_gnt), .rdata(reg_rdata_core),
        .m_awaddr(reg_awaddr), .m_awvalid(reg_awvalid), .m_awready(reg_awready),
        .m_wdata(reg_wdata), .m_wstrb(reg_wstrb), .m_wvalid(reg_wvalid), .m_wready(reg_wready),
        .m_bresp(reg_bresp), .m_bvalid(reg_bvalid), .m_bready(reg_bready),
        .m_araddr(reg_araddr), .m_arvalid(reg_arvalid), .m_arready(reg_arready),
        .m_rdata(reg_rdata), .m_rresp(reg_rresp), .m_rvalid(reg_rvalid), .m_rready(reg_rready)
    );

`ifdef FORMAL
`include "seq_top_properties.vh"
`endif

    wire _unused = &{1'b0, s_wstrb, desc_idx};
endmodule
