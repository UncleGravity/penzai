`default_nettype none

module seq_top_formal (
    input wire clk
);
    localparam integer REG_ADDR_W = 32;
    localparam integer COUNT_W = 4;
    localparam integer CMD_DEPTH_LOG2 = 2;

    (* anyseq *) reg                  rst_n;
    (* anyseq *) reg [15:0]           s_awaddr;
    (* anyseq *) reg                  s_awvalid;
    (* anyseq *) reg [31:0]           s_wdata;
    (* anyseq *) reg [3:0]            s_wstrb;
    (* anyseq *) reg                  s_wvalid;
    (* anyseq *) reg                  s_bready;
    (* anyseq *) reg [15:0]           s_araddr;
    (* anyseq *) reg                  s_arvalid;
    (* anyseq *) reg                  s_rready;
    (* anyseq *) reg                  reg_awready;
    (* anyseq *) reg                  reg_wready;
    (* anyseq *) reg [1:0]            reg_bresp;
    (* anyseq *) reg                  reg_bvalid;
    (* anyseq *) reg                  reg_arready;
    (* anyseq *) reg [31:0]           reg_rdata;
    (* anyseq *) reg [1:0]            reg_rresp;
    (* anyseq *) reg                  reg_rvalid;

    wire                 s_awready;
    wire                 s_wready;
    wire [1:0]           s_bresp;
    wire                 s_bvalid;
    wire                 s_arready;
    wire [31:0]          s_rdata;
    wire [1:0]           s_rresp;
    wire                 s_rvalid;
    wire [REG_ADDR_W-1:0] reg_awaddr;
    wire                 reg_awvalid;
    wire [31:0]          reg_wdata;
    wire [3:0]           reg_wstrb;
    wire                 reg_wvalid;
    wire                 reg_bready;
    wire [REG_ADDR_W-1:0] reg_araddr;
    wire                 reg_arvalid;
    wire                 reg_rready;

    seq_top #(
        .REG_ADDR_W(REG_ADDR_W),
        .COUNT_W(COUNT_W),
        .POLL_TIMEOUT(2),
        .WATCHDOG_TIMEOUT(4),
        .CMD_DEPTH_LOG2(CMD_DEPTH_LOG2)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_awaddr(s_awaddr), .s_awvalid(s_awvalid), .s_awready(s_awready),
        .s_wdata(s_wdata), .s_wstrb(s_wstrb), .s_wvalid(s_wvalid), .s_wready(s_wready),
        .s_bresp(s_bresp), .s_bvalid(s_bvalid), .s_bready(s_bready),
        .s_araddr(s_araddr), .s_arvalid(s_arvalid), .s_arready(s_arready),
        .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rvalid(s_rvalid), .s_rready(s_rready),
        .reg_awaddr(reg_awaddr), .reg_awvalid(reg_awvalid), .reg_awready(reg_awready),
        .reg_wdata(reg_wdata), .reg_wstrb(reg_wstrb), .reg_wvalid(reg_wvalid), .reg_wready(reg_wready),
        .reg_bresp(reg_bresp), .reg_bvalid(reg_bvalid), .reg_bready(reg_bready),
        .reg_araddr(reg_araddr), .reg_arvalid(reg_arvalid), .reg_arready(reg_arready),
        .reg_rdata(reg_rdata), .reg_rresp(reg_rresp), .reg_rvalid(reg_rvalid), .reg_rready(reg_rready)
    );

    reg f_past_valid = 1'b0;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        // Legal AXI-Lite sources hold VALID and payload while backpressured.
        if (f_past_valid && $past(rst_n) && $past(s_awvalid && !s_awready)) begin
            assume(s_awvalid);
            assume(s_awaddr == $past(s_awaddr));
        end
        if (f_past_valid && $past(rst_n) && $past(s_wvalid && !s_wready)) begin
            assume(s_wvalid);
            assume(s_wdata == $past(s_wdata));
            assume(s_wstrb == $past(s_wstrb));
        end
        if (f_past_valid && $past(rst_n) && $past(s_arvalid && !s_arready)) begin
            assume(s_arvalid);
            assume(s_araddr == $past(s_araddr));
        end
        if (f_past_valid && $past(rst_n) && $past(reg_bvalid && !reg_bready)) begin
            assume(reg_bvalid);
            assume(reg_bresp == $past(reg_bresp));
        end
        if (f_past_valid && $past(rst_n) && $past(reg_rvalid && !reg_rready)) begin
            assume(reg_rvalid);
            assume(reg_rdata == $past(reg_rdata));
            assume(reg_rresp == $past(reg_rresp));
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!s_awready && !s_wready && !s_bvalid);
            assert(!s_arready && !s_rvalid);
            assert(!reg_awvalid && !reg_wvalid && !reg_bready);
            assert(!reg_arvalid && !reg_rready);
        end

        if (rst_n) begin
            assert(s_awready == s_wready);
            if (s_awready)
                assert(s_awvalid && s_wvalid && !s_bvalid);
            if (s_arready)
                assert(s_arvalid && s_rvalid);

            if (f_past_valid && $past(rst_n) && $past(s_bvalid && !s_bready)) begin
                assert(s_bvalid);
                assert(s_bresp == $past(s_bresp));
            end
            if (f_past_valid && $past(rst_n) && $past(s_rvalid && !s_rready)) begin
                assert(s_rvalid);
                assert(s_rdata == $past(s_rdata));
                assert(s_rresp == $past(s_rresp));
            end

        end
    end
endmodule

`default_nettype wire
