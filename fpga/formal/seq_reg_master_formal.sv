`default_nettype none

module seq_reg_master_formal (
    input wire clk
);
    localparam integer ADDR_W = 32;

    (* anyseq *) reg                  rst_n;
    (* anyseq *) reg                  req;
    (* anyseq *) reg                  we;
    (* anyseq *) reg [ADDR_W-1:0]     addr;
    (* anyseq *) reg [31:0]           wdata;
    (* anyseq *) reg                  m_awready;
    (* anyseq *) reg                  m_wready;
    (* anyseq *) reg [1:0]            m_bresp;
    (* anyseq *) reg                  m_bvalid;
    (* anyseq *) reg                  m_arready;
    (* anyseq *) reg [31:0]           m_rdata;
    (* anyseq *) reg [1:0]            m_rresp;
    (* anyseq *) reg                  m_rvalid;

    wire                 gnt;
    wire [31:0]          rdata;
    wire [ADDR_W-1:0]    m_awaddr;
    wire                 m_awvalid;
    wire [31:0]          m_wdata;
    wire [3:0]           m_wstrb;
    wire                 m_wvalid;
    wire                 m_bready;
    wire [ADDR_W-1:0]    m_araddr;
    wire                 m_arvalid;
    wire                 m_rready;

    seq_reg_master #(.ADDR_W(ADDR_W)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .req(req),
        .we(we),
        .addr(addr),
        .wdata(wdata),
        .gnt(gnt),
        .rdata(rdata),
        .m_awaddr(m_awaddr),
        .m_awvalid(m_awvalid),
        .m_awready(m_awready),
        .m_wdata(m_wdata),
        .m_wstrb(m_wstrb),
        .m_wvalid(m_wvalid),
        .m_wready(m_wready),
        .m_bresp(m_bresp),
        .m_bvalid(m_bvalid),
        .m_bready(m_bready),
        .m_araddr(m_araddr),
        .m_arvalid(m_arvalid),
        .m_arready(m_arready),
        .m_rdata(m_rdata),
        .m_rresp(m_rresp),
        .m_rvalid(m_rvalid),
        .m_rready(m_rready)
    );

    wire aw_hs = m_awvalid && m_awready;
    wire w_hs  = m_wvalid && m_wready;
    wire b_hs  = m_bvalid && m_bready;
    wire ar_hs = m_arvalid && m_arready;
    wire r_hs  = m_rvalid && m_rready;

    reg f_past_valid = 1'b0;
    reg aw_seen = 1'b0;
    reg w_seen = 1'b0;
    reg ar_seen = 1'b0;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        // Start from reset, then analyze arbitrary legal traffic after reset.
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        // seq_core's req/gnt contract: hold the request and its payload until
        // grant, then present a low req cycle so the adapter can re-arm.
        if (f_past_valid && $past(rst_n) && $past(req && !gnt)) begin
            assume(req);
            assume(we == $past(we));
            assume(addr == $past(addr));
            assume(wdata == $past(wdata));
        end
        if (f_past_valid && $past(gnt))
            assume(!req);

        // AXI sources must hold VALID and payload stable while backpressured.
        if (f_past_valid && $past(rst_n) && $past(m_awvalid && !m_awready)) begin
            assert(m_awvalid);
            assert(m_awaddr == $past(m_awaddr));
        end
        if (f_past_valid && $past(rst_n) && $past(m_wvalid && !m_wready)) begin
            assert(m_wvalid);
            assert(m_wdata == $past(m_wdata));
            assert(m_wstrb == $past(m_wstrb));
        end
        if (f_past_valid && $past(rst_n) && $past(m_arvalid && !m_arready)) begin
            assert(m_arvalid);
            assert(m_araddr == $past(m_araddr));
        end

        // Model the accepted halves of each outstanding transaction. This is
        // monitor state only; it does not constrain independent AW/W stalls.
        if (!rst_n) begin
            aw_seen <= 1'b0;
            w_seen <= 1'b0;
            ar_seen <= 1'b0;
        end else begin
            if (aw_hs) begin
                assert(!aw_seen);
                aw_seen <= 1'b1;
            end
            if (w_hs) begin
                assert(!w_seen);
                w_seen <= 1'b1;
            end
            if (ar_hs) begin
                assert(!ar_seen);
                ar_seen <= 1'b1;
            end
            if (b_hs) begin
                assert(aw_seen && w_seen);
                aw_seen <= 1'b0;
                w_seen <= 1'b0;
            end
            if (r_hs) begin
                assert(ar_seen);
                ar_seen <= 1'b0;
            end
        end

        if (rst_n) begin
            // Reachable channel shape, expressed only through the public AXI
            // interface so the proof does not depend on the FSM encoding.
            assert(!(m_arvalid && m_rready));
            assert(!(m_bready && (m_awvalid || m_wvalid)));
            assert(ar_seen == m_rready);
            assert(aw_seen == (!m_awvalid && (m_wvalid || m_bready)));
            assert(w_seen == (!m_wvalid && (m_awvalid || m_bready)));

            // The transaction presented to AXI is the held req-side command.
            if (m_awvalid) begin
                assert(req && we);
                assert(m_awaddr == addr);
            end
            if (m_wvalid) begin
                assert(req && we);
                assert(m_wdata == wdata);
                assert(m_wstrb == 4'hf);
            end
            if (m_bready)
                assert(req && we);
            if (m_arvalid) begin
                assert(req && !we);
                assert(m_araddr == addr);
            end
            if (m_rready)
                assert(req && !we);

            // Read and write channel families are never active together.
            assert(!((m_awvalid || m_wvalid || m_bready) &&
                     (m_arvalid || m_rready)));

            // Responses are accepted only after their request channels retire.
            if (m_bready)
                assert(aw_seen && w_seen);
            if (m_rready)
                assert(ar_seen);

            // Every grant corresponds to exactly the preceding response
            // handshake; it is never speculative and never stretches.
            if (f_past_valid && $past(rst_n)) begin
                assert(gnt == $past(b_hs || r_hs));
                if ($past(gnt))
                    assert(!gnt);
                if (rdata != $past(rdata))
                    assert($past(r_hs));
            end
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!gnt);
            assert(!m_awvalid && !m_wvalid && !m_bready);
            assert(!m_arvalid && !m_rready);
        end

        // Reachability checks: both operations can complete, and AW/W can
        // independently stall without confusing the adapter.
        cover(rst_n && gnt && $past(b_hs));
        cover(rst_n && gnt && $past(r_hs));
        cover(rst_n && aw_seen && !w_seen);
        cover(rst_n && !aw_seen && w_seen);
    end
endmodule

`default_nettype wire
