// seq_reg_master - adapts seq_core's register req/gnt port to an AXI-Lite MASTER (the replay
// path: seq.v drives sc_ctrl, reaching every DMA + kernel the PS reaches today). One outstanding
// transaction at a time (the executor is strictly sequential), so no IDs / no reordering:
//   req & we  -> AXI-Lite write (AW + W, then B)  -> gnt
//   req & !we -> AXI-Lite read  (AR, then R)      -> gnt + rdata
// gnt is a 1-cycle pulse when the transaction retires, exactly the handshake seq_core expects.
// Cosim-gated (test-rtl-seq-reg-master) vs a modeled AXI-Lite slave.

`default_nettype none

module seq_reg_master #(
    parameter integer ADDR_W = 32
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // seq_core register port (req held until gnt; gnt is a 1-cycle pulse)
    input  wire                 req,
    input  wire                 we,
    input  wire [ADDR_W-1:0]    addr,
    input  wire [31:0]          wdata,
    output reg                  gnt,
    output reg  [31:0]          rdata,

    // AXI-Lite master
    output reg  [ADDR_W-1:0]    m_awaddr,
    output reg                  m_awvalid,
    input  wire                 m_awready,
    output reg  [31:0]          m_wdata,
    output reg  [3:0]           m_wstrb,
    output reg                  m_wvalid,
    input  wire                 m_wready,
    input  wire [1:0]           m_bresp,
    input  wire                 m_bvalid,
    output reg                  m_bready,
    output reg  [ADDR_W-1:0]    m_araddr,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    input  wire [31:0]          m_rdata,
    input  wire [1:0]           m_rresp,
    input  wire                 m_rvalid,
    output reg                  m_rready
);
    localparam [2:0] S_IDLE = 3'd0, S_AWW = 3'd1, S_B = 3'd2, S_AR = 3'd3, S_R = 3'd4;
    reg [2:0] state;
    reg       aw_done, w_done;
    // req is held until gnt and dropped one cycle LATER by seq_core (both registered), so on return
    // to IDLE req is briefly still high. `armed` (re-armed only once req has gone low) makes one req
    // pulse start exactly one transaction.
    reg       armed;

    wire aw_hs = m_awvalid && m_awready;
    wire w_hs  = m_wvalid && m_wready;

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; gnt <= 1'b0; rdata <= 32'd0;
            m_awaddr <= 0; m_awvalid <= 1'b0; m_wdata <= 32'd0; m_wstrb <= 4'd0; m_wvalid <= 1'b0;
            m_bready <= 1'b0; m_araddr <= 0; m_arvalid <= 1'b0; m_rready <= 1'b0;
            aw_done <= 1'b0; w_done <= 1'b0; armed <= 1'b1;
        end else begin
            gnt <= 1'b0; // 1-cycle pulse, default low
            case (state)
                S_IDLE: begin
                    if (!req) armed <= 1'b1;       // re-arm once the consumer drops req
                    if (req && armed) begin
                        armed <= 1'b0;
                        if (we) begin
                            m_awaddr <= addr; m_awvalid <= 1'b1;
                            m_wdata  <= wdata; m_wstrb <= 4'hF; m_wvalid <= 1'b1;
                            aw_done <= 1'b0; w_done <= 1'b0; state <= S_AWW;
                        end else begin
                            m_araddr <= addr; m_arvalid <= 1'b1; state <= S_AR;
                        end
                    end
                end

                // Write address + write data are independent channels; drop each as it handshakes,
                // then issue B once both have.
                S_AWW: begin
                    if (aw_hs) m_awvalid <= 1'b0;
                    if (w_hs)  m_wvalid  <= 1'b0;
                    if (aw_hs) aw_done <= 1'b1;
                    if (w_hs)  w_done  <= 1'b1;
                    if ((aw_done || aw_hs) && (w_done || w_hs)) begin
                        m_bready <= 1'b1; state <= S_B;
                    end
                end
                S_B: if (m_bvalid && m_bready) begin
                    m_bready <= 1'b0; gnt <= 1'b1; state <= S_IDLE;
                end

                S_AR: if (m_arvalid && m_arready) begin
                    m_arvalid <= 1'b0; m_rready <= 1'b1; state <= S_R;
                end
                S_R: if (m_rvalid && m_rready) begin
                    m_rready <= 1'b0; rdata <= m_rdata; gnt <= 1'b1; state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused = &{1'b0, m_bresp, m_rresp};
endmodule
