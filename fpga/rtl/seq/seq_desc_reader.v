// seq_desc_reader - adapts seq_core's descriptor req/gnt port to an AXI4 read MASTER: fetch the
// 128-bit entry at desc_base + desc_idx*16 from DRAM (the seq.v descriptor region). Single beat,
// single outstanding (the executor walks entries strictly in order), so arlen=0 / no IDs. The PS
// writes desc_base into the control slave; seq_core drives desc_idx and waits desc_gnt.
// Cosim-gated (test-rtl-seq-desc-reader) vs a modeled AXI4 read slave (a descriptor memory).

`default_nettype none

module seq_desc_reader #(
    parameter integer ADDR_W  = 40,   // HP-port DRAM address width
    parameter integer COUNT_W = 16
) (
    input  wire                 clk,
    input  wire                 rst_n,

    input  wire [ADDR_W-1:0]    desc_base,   // descriptor region base (from the control slave)

    // seq_core descriptor port
    input  wire                 desc_req,
    input  wire [COUNT_W-1:0]   desc_idx,
    output reg                  desc_gnt,
    output reg  [127:0]         desc_data,

    // AXI4 read master
    output reg  [ADDR_W-1:0]    m_araddr,
    output reg  [7:0]           m_arlen,
    output reg  [2:0]           m_arsize,
    output reg  [1:0]           m_arburst,
    output reg                  m_arvalid,
    input  wire                 m_arready,
    input  wire [127:0]         m_rdata,
    input  wire [1:0]           m_rresp,
    input  wire                 m_rlast,
    input  wire                 m_rvalid,
    output reg                  m_rready
);
    localparam [1:0] S_IDLE = 2'd0, S_AR = 2'd1, S_R = 2'd2;
    reg [1:0] state;
    // seq_core holds desc_req until gnt and drops it a cycle later, so it's briefly still high on
    // return to IDLE; `armed` (re-armed once req is low) makes one req pulse fetch exactly one entry.
    reg       armed;

    // entry offset = desc_idx * 16 (zero-extended to the address width).
    wire [ADDR_W-1:0] entry_off = {{(ADDR_W - COUNT_W - 4){1'b0}}, desc_idx, 4'b0000};

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; desc_gnt <= 1'b0; desc_data <= 128'd0;
            m_araddr <= 0; m_arlen <= 8'd0; m_arsize <= 3'd0; m_arburst <= 2'b01;
            m_arvalid <= 1'b0; m_rready <= 1'b0; armed <= 1'b1;
        end else begin
            desc_gnt <= 1'b0; // 1-cycle pulse
            case (state)
                S_IDLE: begin
                    if (!desc_req) armed <= 1'b1;   // re-arm once the consumer drops req
                    if (desc_req && armed) begin
                        armed <= 1'b0;
                        m_araddr  <= desc_base + entry_off;
                        m_arlen   <= 8'd0;          // 1 beat
                        m_arsize  <= 3'd4;          // 16 bytes / beat (128-bit)
                        m_arburst <= 2'b01;         // INCR
                        m_arvalid <= 1'b1;
                        state     <= S_AR;
                    end
                end
                S_AR: if (m_arvalid && m_arready) begin
                    m_arvalid <= 1'b0; m_rready <= 1'b1; state <= S_R;
                end
                S_R: if (m_rvalid && m_rready) begin
                    desc_data <= m_rdata; m_rready <= 1'b0; desc_gnt <= 1'b1; state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused = &{1'b0, m_rresp, m_rlast};
endmodule
