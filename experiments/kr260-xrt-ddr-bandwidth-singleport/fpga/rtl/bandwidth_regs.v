module bandwidth_regs (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ACLK CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXIS:S_AXIS, ASSOCIATED_RESET aresetn" *)
    input  wire         aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ARESETN RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, DATA_WIDTH 32, ADDR_WIDTH 6, PROTOCOL AXI4LITE, READ_WRITE_MODE READ_WRITE" *)
    input  wire [5:0]   s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0]   s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire         s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output reg          s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]  s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire         s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output reg          s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output reg  [1:0]   s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output reg          s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire         s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [5:0]   s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire [2:0]   s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire         s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output reg          s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg  [31:0]  s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg  [1:0]   s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg          s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire         s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [127:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [15:0]  m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire         m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  wire         m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire         m_axis_tlast,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  wire [127:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input  wire [15:0]  s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input  wire         s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire         s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  wire         s_axis_tlast
);
    localparam [3:0] REG_CONTROL       = 4'h0;
    localparam [3:0] REG_STATUS        = 4'h1;
    localparam [3:0] REG_LENGTH_LO     = 4'h2;
    localparam [3:0] REG_LENGTH_HI     = 4'h3;
    localparam [3:0] REG_SEED          = 4'h4;
    localparam [3:0] REG_GEN_CYC_LO    = 4'h5;
    localparam [3:0] REG_GEN_CYC_HI    = 4'h6;
    localparam [3:0] REG_CHK_CYC_LO    = 4'h7;
    localparam [3:0] REG_CHK_CYC_HI    = 4'h8;
    localparam [3:0] REG_BYTES_LO      = 4'h9;
    localparam [3:0] REG_BYTES_HI      = 4'ha;
    localparam [3:0] REG_FIRST_ERR_LO  = 4'hb;
    localparam [3:0] REG_FIRST_ERR_HI  = 4'hc;
    localparam [3:0] REG_EXPECTED      = 4'hd;
    localparam [3:0] REG_ACTUAL        = 4'he;
    localparam [3:0] REG_BASE_LO       = 4'hf;

    reg [63:0] length_bytes;
    reg [31:0] base_index_lo;
    reg [7:0]  seed;
    reg        gen_start;
    reg        check_start;
    reg [31:0] read_data;

    wire gen_busy;
    wire gen_done;
    wire [63:0] gen_cycles;

    wire check_busy;
    wire check_done;
    wire check_error_seen;
    wire [63:0] check_first_error_index;
    wire [7:0] check_expected;
    wire [7:0] check_actual;
    wire [63:0] check_bytes_checked;
    wire [63:0] check_cycles;

    wire [31:0] status_word = {
        27'd0,
        check_error_seen,
        check_done,
        check_busy,
        gen_done,
        gen_busy
    };

    always @* begin
        case (s_axi_araddr[5:2])
            REG_STATUS:       read_data = status_word;
            REG_LENGTH_LO:    read_data = length_bytes[31:0];
            REG_LENGTH_HI:    read_data = length_bytes[63:32];
            REG_SEED:         read_data = {24'd0, seed};
            REG_GEN_CYC_LO:   read_data = gen_cycles[31:0];
            REG_GEN_CYC_HI:   read_data = gen_cycles[63:32];
            REG_CHK_CYC_LO:   read_data = check_cycles[31:0];
            REG_CHK_CYC_HI:   read_data = check_cycles[63:32];
            REG_BYTES_LO:     read_data = check_bytes_checked[31:0];
            REG_BYTES_HI:     read_data = check_bytes_checked[63:32];
            REG_FIRST_ERR_LO: read_data = check_first_error_index[31:0];
            REG_FIRST_ERR_HI: read_data = check_first_error_index[63:32];
            REG_EXPECTED:     read_data = {24'd0, check_expected};
            REG_ACTUAL:       read_data = {24'd0, check_actual};
            REG_BASE_LO:      read_data = base_index_lo;
            default:          read_data = 32'd0;
        endcase
    end

    wire write_accept = !s_axi_bvalid && s_axi_awvalid && s_axi_wvalid;
    wire read_accept = !s_axi_rvalid && s_axi_arvalid;

    always @(posedge aclk) begin
        if (!aresetn) begin
            length_bytes <= 64'd0;
            base_index_lo <= 32'd0;
            seed <= 8'd1;
            gen_start <= 1'b0;
            check_start <= 1'b0;
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_bvalid <= 1'b0;
            s_axi_arready <= 1'b0;
            s_axi_rdata <= 32'd0;
            s_axi_rresp <= 2'b00;
            s_axi_rvalid <= 1'b0;
        end else begin
            gen_start <= 1'b0;
            check_start <= 1'b0;

            s_axi_awready <= write_accept;
            s_axi_wready <= write_accept;
            if (write_accept) begin
                case (s_axi_awaddr[5:2])
                    REG_CONTROL: begin
                        if (s_axi_wdata[0]) gen_start <= 1'b1;
                        if (s_axi_wdata[1]) check_start <= 1'b1;
                    end
                    REG_LENGTH_LO: length_bytes[31:0] <= s_axi_wdata;
                    REG_LENGTH_HI: length_bytes[63:32] <= s_axi_wdata;
                    REG_BASE_LO:   base_index_lo <= s_axi_wdata;
                    REG_SEED:      seed <= s_axi_wdata[7:0];
                    default: begin end
                endcase
                s_axi_bresp <= 2'b00;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            s_axi_arready <= read_accept;
            if (read_accept) begin
                s_axi_rdata <= read_data;
                s_axi_rresp <= 2'b00;
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    axis_pattern_gen gen (
        .aclk(aclk),
        .aresetn(aresetn),
        .start(gen_start),
        .length_bytes(length_bytes),
        .base_index({32'd0, base_index_lo}),
        .seed(seed),
        .busy(gen_busy),
        .done(gen_done),
        .cycles(gen_cycles),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast)
    );

    axis_pattern_check check (
        .aclk(aclk),
        .aresetn(aresetn),
        .start(check_start),
        .length_bytes(length_bytes),
        .base_index({32'd0, base_index_lo}),
        .seed(seed),
        .busy(check_busy),
        .done(check_done),
        .error_seen(check_error_seen),
        .first_error_index(check_first_error_index),
        .expected(check_expected),
        .actual(check_actual),
        .bytes_checked(check_bytes_checked),
        .cycles(check_cycles),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast)
    );

    wire unused_axi = |s_axi_awaddr[1:0] | |s_axi_araddr[1:0] |
        |s_axi_awprot | |s_axi_arprot | |s_axi_wstrb;
endmodule
