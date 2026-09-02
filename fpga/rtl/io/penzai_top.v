`default_nettype none

// Production shell for the Penzai datapath. All control is serialized
// through one AXI-Lite slave; the datapath's seven memory clients remain native
// AXI4 masters and share this clock/reset domain.
module penzai_top #(
    parameter integer ADDR_W = 40,
    parameter [31:0] CLK_HZ = 32'd0
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:M_AXI_W0:M_AXI_W1:M_AXI_W2:M_AXI_W3:M_AXI_HIST_K:M_AXI_HIST_V:M_AXI_KV, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire                 s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                 s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [11:0]          s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0]           s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                 s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire                 s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0]          s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]           s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                 s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire                 s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]           s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire                 s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                 s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [11:0]          s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire [2:0]           s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire                 s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire                 s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0]          s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]           s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire                 s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire                 s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_W0, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_w0_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARLEN" *)
    output wire [7:0]           m_axi_w0_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARSIZE" *)
    output wire [2:0]           m_axi_w0_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARBURST" *)
    output wire [1:0]           m_axi_w0_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARVALID" *)
    output wire                 m_axi_w0_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 ARREADY" *)
    input  wire                 m_axi_w0_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 RDATA" *)
    input  wire [127:0]         m_axi_w0_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 RRESP" *)
    input  wire [1:0]           m_axi_w0_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 RLAST" *)
    input  wire                 m_axi_w0_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 RVALID" *)
    input  wire                 m_axi_w0_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W0 RREADY" *)
    output wire                 m_axi_w0_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_W1, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_w1_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARLEN" *)
    output wire [7:0]           m_axi_w1_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARSIZE" *)
    output wire [2:0]           m_axi_w1_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARBURST" *)
    output wire [1:0]           m_axi_w1_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARVALID" *)
    output wire                 m_axi_w1_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 ARREADY" *)
    input  wire                 m_axi_w1_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 RDATA" *)
    input  wire [127:0]         m_axi_w1_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 RRESP" *)
    input  wire [1:0]           m_axi_w1_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 RLAST" *)
    input  wire                 m_axi_w1_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 RVALID" *)
    input  wire                 m_axi_w1_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W1 RREADY" *)
    output wire                 m_axi_w1_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_W2, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_w2_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARLEN" *)
    output wire [7:0]           m_axi_w2_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARSIZE" *)
    output wire [2:0]           m_axi_w2_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARBURST" *)
    output wire [1:0]           m_axi_w2_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARVALID" *)
    output wire                 m_axi_w2_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 ARREADY" *)
    input  wire                 m_axi_w2_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 RDATA" *)
    input  wire [127:0]         m_axi_w2_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 RRESP" *)
    input  wire [1:0]           m_axi_w2_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 RLAST" *)
    input  wire                 m_axi_w2_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 RVALID" *)
    input  wire                 m_axi_w2_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W2 RREADY" *)
    output wire                 m_axi_w2_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_W3, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_w3_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARLEN" *)
    output wire [7:0]           m_axi_w3_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARSIZE" *)
    output wire [2:0]           m_axi_w3_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARBURST" *)
    output wire [1:0]           m_axi_w3_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARVALID" *)
    output wire                 m_axi_w3_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 ARREADY" *)
    input  wire                 m_axi_w3_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 RDATA" *)
    input  wire [127:0]         m_axi_w3_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 RRESP" *)
    input  wire [1:0]           m_axi_w3_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 RLAST" *)
    input  wire                 m_axi_w3_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 RVALID" *)
    input  wire                 m_axi_w3_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_W3 RREADY" *)
    output wire                 m_axi_w3_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_HIST_K, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_hist_k_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARLEN" *)
    output wire [7:0]           m_axi_hist_k_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARSIZE" *)
    output wire [2:0]           m_axi_hist_k_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARBURST" *)
    output wire [1:0]           m_axi_hist_k_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARVALID" *)
    output wire                 m_axi_hist_k_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K ARREADY" *)
    input  wire                 m_axi_hist_k_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K RDATA" *)
    input  wire [127:0]         m_axi_hist_k_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K RRESP" *)
    input  wire [1:0]           m_axi_hist_k_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K RLAST" *)
    input  wire                 m_axi_hist_k_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K RVALID" *)
    input  wire                 m_axi_hist_k_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_K RREADY" *)
    output wire                 m_axi_hist_k_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_HIST_V, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE READ_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_hist_v_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARLEN" *)
    output wire [7:0]           m_axi_hist_v_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARSIZE" *)
    output wire [2:0]           m_axi_hist_v_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARBURST" *)
    output wire [1:0]           m_axi_hist_v_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARVALID" *)
    output wire                 m_axi_hist_v_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V ARREADY" *)
    input  wire                 m_axi_hist_v_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V RDATA" *)
    input  wire [127:0]         m_axi_hist_v_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V RRESP" *)
    input  wire [1:0]           m_axi_hist_v_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V RLAST" *)
    input  wire                 m_axi_hist_v_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V RVALID" *)
    input  wire                 m_axi_hist_v_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_HIST_V RREADY" *)
    output wire                 m_axi_hist_v_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI_KV, PROTOCOL AXI4, DATA_WIDTH 128, ADDR_WIDTH 40, READ_WRITE_MODE WRITE_ONLY" *)
    output wire [ADDR_W-1:0]    m_axi_kv_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWLEN" *)
    output wire [7:0]           m_axi_kv_awlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWSIZE" *)
    output wire [2:0]           m_axi_kv_awsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWBURST" *)
    output wire [1:0]           m_axi_kv_awburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWVALID" *)
    output wire                 m_axi_kv_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV AWREADY" *)
    input  wire                 m_axi_kv_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV WDATA" *)
    output wire [127:0]         m_axi_kv_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV WSTRB" *)
    output wire [15:0]          m_axi_kv_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV WLAST" *)
    output wire                 m_axi_kv_wlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV WVALID" *)
    output wire                 m_axi_kv_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV WREADY" *)
    input  wire                 m_axi_kv_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV BRESP" *)
    input  wire [1:0]           m_axi_kv_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV BVALID" *)
    input  wire                 m_axi_kv_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI_KV BREADY" *)
    output wire                 m_axi_kv_bready
);
    `include "engine_regs.vh"

    localparam [2:0] MODEL_SPEC_OP_NONE  = 3'd0;
    localparam [2:0] MODEL_SPEC_OP_CLEAR = 3'd1;
    localparam [2:0] MODEL_SPEC_OP_BEGIN = 3'd2;
    localparam [2:0] MODEL_SPEC_OP_LAYER = 3'd3;
    localparam [2:0] MODEL_SPEC_OP_SEAL  = 3'd4;

    localparam [7:0] FRONTEND_COLLISION   = 8'h01;
    localparam [7:0] FRONTEND_MODEL_SPEC_OP  = 8'h02;
    localparam [7:0] FRONTEND_EXECUTE     = 8'h03;
    localparam [7:0] FRONTEND_CLEAR       = 8'h04;
    localparam [7:0] FRONTEND_EVENT_FULL  = 8'h05;
    localparam [7:0] FRONTEND_LOGITS      = 8'h06;
    localparam [7:0] FRONTEND_RESERVED    = 8'h07;

    wire clk = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    function automatic [31:0] merge_wstrb;
        input [31:0] old_value;
        input [31:0] new_value;
        input [3:0] byte_enable;
        integer byte_index;
        begin
            merge_wstrb = old_value;
            for (byte_index = 0; byte_index < 4;
                 byte_index = byte_index + 1)
                if (byte_enable[byte_index])
                    merge_wstrb[byte_index*8 +: 8] =
                        new_value[byte_index*8 +: 8];
        end
    endfunction

    function automatic address_fits;
        input [63:0] address;
        integer address_bit;
        begin
            address_fits = 1'b1;
            for (address_bit = ADDR_W; address_bit < 64;
                 address_bit = address_bit + 1)
                if (address[address_bit])
                    address_fits = 1'b0;
        end
    endfunction

    // Independently capture AXI-Lite AW and W. Neither channel relies on the
    // other arriving in the same cycle, and both payloads remain stable until
    // one ordered write response is created.
    reg aw_pending_q;
    reg [11:0] awaddr_q;
    reg w_pending_q;
    reg [31:0] wdata_q;
    reg [3:0] wstrb_q;
    reg bvalid_q;
    wire aw_fire = s_axi_awvalid && s_axi_awready;
    wire w_fire = s_axi_wvalid && s_axi_wready;
    wire write_commit = aw_pending_q && w_pending_q && !bvalid_q;
    wire [31:0] write_masked = wdata_q &
        {{8{wstrb_q[3]}}, {8{wstrb_q[2]}},
         {8{wstrb_q[1]}}, {8{wstrb_q[0]}}};

    assign s_axi_awready = rst_n && !aw_pending_q && !bvalid_q;
    assign s_axi_wready = rst_n && !w_pending_q && !bvalid_q;
    assign s_axi_bresp = 2'b00;
    assign s_axi_bvalid = bvalid_q;

    // Staging registers are freely writable. MODEL_SPEC_BEGIN, MODEL_SPEC_LAYER,
    // and EXECUTE snapshot them into immutable command registers.
    reg [31:0] model_spec_id_stage_q;
    reg [31:0] model_spec_hash_lo_stage_q;
    reg [31:0] model_spec_hash_hi_stage_q;
    reg [31:0] model_spec_layer_count_stage_q;
    reg [31:0] model_spec_hidden_blocks_stage_q;
    reg [31:0] model_spec_ffn_blocks_stage_q;
    reg [31:0] model_spec_q_heads_stage_q;
    reg [31:0] model_spec_kv_heads_stage_q;
    reg [31:0] model_spec_head_dim_stage_q;
    reg [31:0] model_spec_weight_fmt_stage_q;
    reg [31:0] model_spec_context_limit_stage_q;
    reg [31:0] model_spec_vocab_rows_stage_q;
    reg [31:0] model_spec_embed_addr_lo_stage_q;
    reg [31:0] model_spec_embed_addr_hi_stage_q;
    reg [31:0] model_spec_lm_addr_lo_stage_q;
    reg [31:0] model_spec_lm_addr_hi_stage_q;
    reg [31:0] model_spec_final_addr_lo_stage_q;
    reg [31:0] model_spec_final_addr_hi_stage_q;
    reg [31:0] model_spec_rope_addr_lo_stage_q;
    reg [31:0] model_spec_rope_addr_hi_stage_q;
    reg [31:0] model_spec_layer_index_stage_q;
    reg [31:0] model_spec_layer_word_stage_q;
    reg [31:0] model_spec_layer_data_lo_stage_q;
    reg [31:0] model_spec_layer_data_hi_stage_q;

    reg [31:0] cmd_kv_capacity_stage_q;
    reg [31:0] cmd_tag_stage_q;
    reg [31:0] cmd_model_spec_id_stage_q;
    reg [31:0] cmd_model_spec_hash_lo_stage_q;
    reg [31:0] cmd_model_spec_hash_hi_stage_q;
    reg [31:0] cmd_shape_stage_q;
    reg [31:0] cmd_position_stage_q;
    reg [31:0] cmd_kv_base_lo_stage_q;
    reg [31:0] cmd_kv_base_hi_stage_q;
    reg [31:0] cmd_token_stage_q [0:7];

    // Frozen command records presented to the core.
    reg [2:0] model_spec_op_q;
    reg [31:0] begin_id_q;
    reg [63:0] begin_hash_q;
    reg [5:0] begin_layer_count_q;
    reg [7:0] begin_hidden_blocks_q;
    reg [9:0] begin_ffn_blocks_q;
    reg [5:0] begin_q_heads_q;
    reg [3:0] begin_kv_heads_q;
    reg [7:0] begin_head_dim_q;
    reg [1:0] begin_weight_fmt_q;
    reg [16:0] begin_context_limit_q;
    reg [17:0] begin_vocab_rows_q;
    reg [63:0] begin_embed_addr_q;
    reg [63:0] begin_lm_addr_q;
    reg [63:0] begin_final_addr_q;
    reg [63:0] begin_rope_addr_q;
    reg [5:0] layer_index_q;
    reg [2:0] layer_word_q;
    reg [63:0] layer_data_q;

    reg exec_pending_q;
    reg [31:0] exec_tag_q;
    reg [31:0] exec_model_spec_id_q;
    reg [63:0] exec_model_spec_hash_q;
    reg [3:0] exec_token_count_q;
    reg [7:0] exec_lane_mask_q;
    reg [255:0] exec_token_ids_q;
    reg [16:0] exec_position_q;
    reg [63:0] exec_kv_base_q;
    reg [16:0] exec_kv_capacity_q;
    reg exec_emit_logits_q;

    reg run_clear_q;
    reg [6:0] metrics_index_q;

    // One-entry architectural event mailboxes. A new EXEC is rejected until
    // every event produced by the previous command has been acknowledged.
    reg commit_pending_q;
    reg [31:0] commit_tag_q;
    reg [31:0] commit_model_spec_id_q;
    reg [63:0] commit_model_spec_hash_q;
    reg [3:0] commit_token_count_q;
    reg [16:0] commit_kv_length_q;
    reg commit_logits_valid_q;
    reg error_pending_q;
    reg [31:0] error_tag_q;
    reg [15:0] error_code_q;
    reg [7:0] error_detail_q;
    reg [5:0] error_layer_q;
    reg [4:0] error_stage_q;
    reg result_pending_q;
    reg [17:0] result_token_q;
    reg [31:0] result_logit_q;
    reg result_error_q;
    reg [7:0] result_status_q;
    reg model_spec_error_pending_q;
    reg [7:0] model_spec_error_code_q;
    reg [5:0] model_spec_error_layer_q;
    reg [2:0] model_spec_error_word_q;
    reg frontend_error_pending_q;
    reg [7:0] frontend_error_q;

    // Core control/status boundary.
    wire core_model_spec_clear_ready;
    wire core_model_spec_begin_ready;
    wire core_model_spec_layer_ready;
    wire core_model_spec_seal_ready;
    wire core_model_spec_error_valid;
    wire [7:0] core_model_spec_error_code;
    wire [5:0] core_model_spec_error_layer;
    wire [2:0] core_model_spec_error_word;
    wire core_model_spec_loading;
    wire core_model_spec_sealed;
    wire [31:0] core_interface_version;
    wire [63:0] core_layout_hash;
    wire [31:0] core_active_model_spec_id;
    wire [63:0] core_active_model_spec_hash;
    wire core_cmd_ready;
    wire core_commit_valid;
    wire [31:0] core_commit_tag;
    wire [31:0] core_commit_model_spec_id;
    wire [63:0] core_commit_model_spec_hash;
    wire [3:0] core_commit_token_count;
    wire [16:0] core_commit_kv_length;
    wire core_commit_logits_valid;
    wire core_error_valid;
    wire [31:0] core_error_tag;
    wire [15:0] core_error_code;
    wire [7:0] core_error_detail;
    wire [5:0] core_error_layer;
    wire [4:0] core_error_stage;
    wire core_logits_valid;
    wire [17:0] core_logits_row;
    wire [31:0] core_logits_data;
    wire core_logits_last;
    wire core_result_valid;
    wire [17:0] core_result_token;
    wire [31:0] core_result_logit;
    wire core_result_error;
    wire [7:0] core_result_status;
    wire core_busy;
    wire core_clear_done;
    wire [5:0] core_debug_layer;
    wire [4:0] core_debug_stage;
    wire core_metrics_stage_active;
    wire [4:0] core_metrics_stage;
    wire core_trace_valid;
    wire [5:0] core_trace_layer;
    wire [4:0] core_trace_stage;
    wire core_protocol_error;
    wire [12:0] core_metrics_projection_probe;
    wire [2:0] core_metrics_weight_axi_r_beats;
    wire [2:0] core_metrics_weight_axi_r_gap_ports;
    wire core_metrics_weight_zip_skew;
    wire [1:0] core_metrics_history_axi_r_beats;
    wire core_metrics_kv_axi_w_beat;

    wire [31:0] metrics_schema;
    wire [31:0] metrics_capabilities;
    wire [31:0] metrics_status;
    wire [31:0] metrics_tag;
    wire [31:0] metrics_data_lo;
    wire [31:0] metrics_data_hi;
    wire [31:0] metrics_overflow0;
    wire [31:0] metrics_overflow1;
    wire [31:0] metrics_overflow2;
    wire [31:0] metrics_overflow3;
    wire [63:0] metrics_total_cycles;
    wire metrics_recording;
    wire metrics_sealing;
    wire metrics_snapshot_valid;

    wire core_model_spec_clear_valid = model_spec_op_q == MODEL_SPEC_OP_CLEAR;
    wire core_model_spec_begin_valid = model_spec_op_q == MODEL_SPEC_OP_BEGIN;
    wire core_model_spec_layer_valid = model_spec_op_q == MODEL_SPEC_OP_LAYER;
    wire core_model_spec_seal_valid = model_spec_op_q == MODEL_SPEC_OP_SEAL;
    wire model_spec_op_fire =
        (core_model_spec_clear_valid && core_model_spec_clear_ready) ||
        (core_model_spec_begin_valid && core_model_spec_begin_ready) ||
        (core_model_spec_layer_valid && core_model_spec_layer_ready) ||
        (core_model_spec_seal_valid && core_model_spec_seal_ready);
    wire core_cmd_valid = exec_pending_q;
    wire core_cmd_fire = core_cmd_valid && core_cmd_ready;
    wire core_commit_ready = !commit_pending_q;
    wire core_error_ready = !error_pending_q;
    wire core_result_ready = !result_pending_q;
    wire core_model_spec_error_ready = !model_spec_error_pending_q;
    wire event_pending = commit_pending_q || error_pending_q ||
                         result_pending_q || model_spec_error_pending_q ||
                         frontend_error_pending_q || metrics_snapshot_valid;
    wire execute_available = !exec_pending_q && !core_busy && !run_clear_q &&
                             (model_spec_op_q == MODEL_SPEC_OP_NONE) &&
                             !event_pending && !metrics_sealing;
    wire begin_stage_reserved_ok =
        (model_spec_layer_count_stage_q[31:6] == 26'd0) &&
        (model_spec_hidden_blocks_stage_q[31:8] == 24'd0) &&
        (model_spec_ffn_blocks_stage_q[31:10] == 22'd0) &&
        (model_spec_q_heads_stage_q[31:6] == 26'd0) &&
        (model_spec_kv_heads_stage_q[31:4] == 28'd0) &&
        (model_spec_head_dim_stage_q[31:8] == 24'd0) &&
        (model_spec_weight_fmt_stage_q[31:2] == 30'd0) &&
        (model_spec_context_limit_stage_q[31:17] == 15'd0) &&
        (model_spec_vocab_rows_stage_q[31:18] == 14'd0);
    wire begin_stage_address_ok =
        address_fits({model_spec_embed_addr_hi_stage_q,
                      model_spec_embed_addr_lo_stage_q}) &&
        address_fits({model_spec_lm_addr_hi_stage_q,
                      model_spec_lm_addr_lo_stage_q}) &&
        address_fits({model_spec_final_addr_hi_stage_q,
                      model_spec_final_addr_lo_stage_q}) &&
        address_fits({model_spec_rope_addr_hi_stage_q,
                      model_spec_rope_addr_lo_stage_q});
    wire layer_stage_reserved_ok =
        (model_spec_layer_index_stage_q[31:6] == 26'd0) &&
        (model_spec_layer_word_stage_q[31:3] == 29'd0);
    wire layer_stage_address_ok =
        address_fits({model_spec_layer_data_hi_stage_q,
                      model_spec_layer_data_lo_stage_q});
    wire execute_stage_reserved_ok =
        (cmd_kv_capacity_stage_q[31:17] == 15'd0) &&
        (cmd_shape_stage_q[31:17] == 15'd0) &&
        (cmd_shape_stage_q[7:4] == 4'd0) &&
        (cmd_position_stage_q[31:17] == 15'd0);
    wire execute_stage_address_ok =
        address_fits({cmd_kv_base_hi_stage_q, cmd_kv_base_lo_stage_q});

    wire core_commit_fire = core_commit_valid && core_commit_ready;
    wire core_error_fire = core_error_valid && core_error_ready;
    wire core_result_fire = core_result_valid && core_result_ready;
    wire core_model_spec_error_fire = core_model_spec_error_valid &&
                                   core_model_spec_error_ready;

    wire ctrl_write = write_commit && (awaddr_q == ENGINE_REG_OFF_CTRL);
    wire [5:0] ctrl_cmd_bits = write_masked[5:0];
    wire ctrl_reserved_bad = write_masked[31:12] != 20'd0;
    wire ctrl_cmd_collision = (ctrl_cmd_bits != 6'd0) &&
        ((ctrl_cmd_bits & (ctrl_cmd_bits - 6'd1)) != 6'd0);
    wire ctrl_ack_commit = ctrl_write && !ctrl_reserved_bad && write_masked[6];
    wire ctrl_ack_error = ctrl_write && !ctrl_reserved_bad && write_masked[7];
    wire ctrl_ack_result = ctrl_write && !ctrl_reserved_bad && write_masked[8];
    wire ctrl_ack_model_spec_error = ctrl_write && !ctrl_reserved_bad &&
                                  write_masked[9];
    wire ctrl_ack_frontend_error = ctrl_write && !ctrl_reserved_bad &&
                                   write_masked[10];
    wire ctrl_ack_metrics = ctrl_write && !ctrl_reserved_bad &&
                            write_masked[11];
    wire ctrl_run_clear_accept = ctrl_write && !ctrl_reserved_bad &&
        !ctrl_cmd_collision && (ctrl_cmd_bits == 6'b100000) &&
        !run_clear_q && (model_spec_op_q == MODEL_SPEC_OP_NONE);
    wire metrics_finish = core_commit_fire || core_error_fire ||
                          (ctrl_run_clear_accept && metrics_recording);
    wire [1:0] metrics_finish_outcome = core_error_fire ? 2'd2 :
        core_commit_fire ? 2'd1 : 2'd3;
    wire begin_snapshot_attempt = ctrl_write && !ctrl_reserved_bad &&
        !ctrl_cmd_collision && (ctrl_cmd_bits == 6'b000010) &&
        begin_stage_reserved_ok && begin_stage_address_ok;
    wire execute_snapshot_attempt = ctrl_write && !ctrl_reserved_bad &&
        !ctrl_cmd_collision && (ctrl_cmd_bits == 6'b010000) &&
        execute_stage_reserved_ok && execute_stage_address_ok;

    wire [31:0] status_word = {
        13'd0,
        metrics_snapshot_valid,
        core_model_spec_seal_ready,
        core_model_spec_layer_ready,
        core_model_spec_begin_ready,
        core_model_spec_clear_ready,
        core_cmd_ready,
        frontend_error_pending_q,
        core_protocol_error,
        model_spec_error_pending_q,
        result_pending_q,
        error_pending_q,
        commit_pending_q,
        exec_pending_q,
        model_spec_op_q != MODEL_SPEC_OP_NONE,
        core_model_spec_sealed,
        core_model_spec_loading,
        core_clear_done,
        run_clear_q,
        core_busy
    };
    wire [31:0] debug_word = {19'd0, core_debug_stage, 2'd0,
                              core_debug_layer};
    wire [31:0] commit_info_word = {
        10'd0, commit_logits_valid_q, commit_kv_length_q,
        commit_token_count_q
    };
    wire [31:0] error_code_detail_word = {
        8'd0, error_detail_q, error_code_q
    };
    wire [31:0] error_location_word = {
        19'd0, error_stage_q, 2'd0, error_layer_q
    };
    wire [31:0] result_token_status_word = {
        5'd0, result_error_q, result_status_q, result_token_q
    };
    wire [31:0] model_spec_error_word = {
        13'd0, model_spec_error_word_q, 2'd0,
        model_spec_error_layer_q, model_spec_error_code_q
    };

    reg rvalid_q;
    reg [31:0] rdata_q;
    assign s_axi_arready = rst_n && !rvalid_q;
    assign s_axi_rdata = rdata_q;
    assign s_axi_rresp = 2'b00;
    assign s_axi_rvalid = rvalid_q;

    // Packed internal memory interfaces. The public master numbering is the
    // same as the projection engine's low-to-high 128-bit lane numbering.
    wire [4*ADDR_W-1:0] core_weight_araddr;
    wire [31:0] core_weight_arlen;
    wire [11:0] core_weight_arsize;
    wire [7:0] core_weight_arburst;
    wire [3:0] core_weight_arvalid;
    wire [3:0] core_weight_rready;

    // AXI-Lite writes and architectural mailbox capture.
    integer token_index;
    always @(posedge clk) begin
        if (!rst_n) begin
            aw_pending_q <= 1'b0;
            awaddr_q <= 12'd0;
            w_pending_q <= 1'b0;
            wdata_q <= 32'd0;
            wstrb_q <= 4'd0;
            bvalid_q <= 1'b0;

            model_spec_id_stage_q <= ENGINE_REG_RST_MODEL_SPEC_ID;
            model_spec_hash_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_HASH_LO;
            model_spec_hash_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_HASH_HI;
            model_spec_layer_count_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LAYER_COUNT;
            model_spec_hidden_blocks_stage_q <= ENGINE_REG_RST_MODEL_SPEC_HIDDEN_BLOCKS;
            model_spec_ffn_blocks_stage_q <= ENGINE_REG_RST_MODEL_SPEC_FFN_BLOCKS;
            model_spec_q_heads_stage_q <= ENGINE_REG_RST_MODEL_SPEC_Q_HEADS;
            model_spec_kv_heads_stage_q <= ENGINE_REG_RST_MODEL_SPEC_KV_HEADS;
            model_spec_head_dim_stage_q <= ENGINE_REG_RST_MODEL_SPEC_HEAD_DIM;
            model_spec_weight_fmt_stage_q <= ENGINE_REG_RST_MODEL_SPEC_WEIGHT_FMT;
            model_spec_context_limit_stage_q <= ENGINE_REG_RST_MODEL_SPEC_CONTEXT_LIMIT;
            model_spec_vocab_rows_stage_q <= ENGINE_REG_RST_MODEL_SPEC_VOCAB_ROWS;
            model_spec_embed_addr_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_EMBED_ADDR_LO;
            model_spec_embed_addr_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_EMBED_ADDR_HI;
            model_spec_lm_addr_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LM_ADDR_LO;
            model_spec_lm_addr_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LM_ADDR_HI;
            model_spec_final_addr_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_FINAL_ADDR_LO;
            model_spec_final_addr_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_FINAL_ADDR_HI;
            model_spec_rope_addr_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_ROPE_ADDR_LO;
            model_spec_rope_addr_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_ROPE_ADDR_HI;
            model_spec_layer_index_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LAYER_INDEX;
            model_spec_layer_word_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LAYER_WORD;
            model_spec_layer_data_lo_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LAYER_DATA_LO;
            model_spec_layer_data_hi_stage_q <= ENGINE_REG_RST_MODEL_SPEC_LAYER_DATA_HI;

            cmd_kv_capacity_stage_q <= ENGINE_REG_RST_CMD_KV_CAPACITY;
            cmd_tag_stage_q <= ENGINE_REG_RST_CMD_TAG;
            cmd_model_spec_id_stage_q <= ENGINE_REG_RST_CMD_MODEL_SPEC_ID;
            cmd_model_spec_hash_lo_stage_q <= ENGINE_REG_RST_CMD_MODEL_SPEC_HASH_LO;
            cmd_model_spec_hash_hi_stage_q <= ENGINE_REG_RST_CMD_MODEL_SPEC_HASH_HI;
            cmd_shape_stage_q <= ENGINE_REG_RST_CMD_SHAPE;
            cmd_position_stage_q <= ENGINE_REG_RST_CMD_POSITION_BASE;
            cmd_kv_base_lo_stage_q <= ENGINE_REG_RST_CMD_KV_BASE_LO;
            cmd_kv_base_hi_stage_q <= ENGINE_REG_RST_CMD_KV_BASE_HI;
            for (token_index = 0; token_index < 8;
                 token_index = token_index + 1)
                cmd_token_stage_q[token_index] <= 32'd0;

            model_spec_op_q <= MODEL_SPEC_OP_NONE;
            begin_id_q <= 32'd0;
            begin_hash_q <= 64'd0;
            begin_layer_count_q <= 6'd0;
            begin_hidden_blocks_q <= 8'd0;
            begin_ffn_blocks_q <= 10'd0;
            begin_q_heads_q <= 6'd0;
            begin_kv_heads_q <= 4'd0;
            begin_head_dim_q <= 8'd0;
            begin_weight_fmt_q <= 2'd0;
            begin_context_limit_q <= 17'd0;
            begin_vocab_rows_q <= 18'd0;
            begin_embed_addr_q <= 64'd0;
            begin_lm_addr_q <= 64'd0;
            begin_final_addr_q <= 64'd0;
            begin_rope_addr_q <= 64'd0;
            layer_index_q <= 6'd0;
            layer_word_q <= 3'd0;
            layer_data_q <= 64'd0;

            exec_pending_q <= 1'b0;
            exec_tag_q <= 32'd0;
            exec_model_spec_id_q <= 32'd0;
            exec_model_spec_hash_q <= 64'd0;
            exec_token_count_q <= 4'd0;
            exec_lane_mask_q <= 8'd0;
            exec_token_ids_q <= 256'd0;
            exec_position_q <= 17'd0;
            exec_kv_base_q <= 64'd0;
            exec_kv_capacity_q <= 17'd0;
            exec_emit_logits_q <= 1'b0;

            run_clear_q <= 1'b0;
            metrics_index_q <= 7'd0;

            commit_pending_q <= 1'b0;
            commit_tag_q <= 32'd0;
            commit_model_spec_id_q <= 32'd0;
            commit_model_spec_hash_q <= 64'd0;
            commit_token_count_q <= 4'd0;
            commit_kv_length_q <= 17'd0;
            commit_logits_valid_q <= 1'b0;
            error_pending_q <= 1'b0;
            error_tag_q <= 32'd0;
            error_code_q <= 16'd0;
            error_detail_q <= 8'd0;
            error_layer_q <= 6'd0;
            error_stage_q <= 5'd0;
            result_pending_q <= 1'b0;
            result_token_q <= 18'd0;
            result_logit_q <= 32'd0;
            result_error_q <= 1'b0;
            result_status_q <= 8'd0;
            model_spec_error_pending_q <= 1'b0;
            model_spec_error_code_q <= 8'd0;
            model_spec_error_layer_q <= 6'd0;
            model_spec_error_word_q <= 3'd0;
            frontend_error_pending_q <= 1'b0;
            frontend_error_q <= 8'd0;
        end else begin
            if (aw_fire) begin
                aw_pending_q <= 1'b1;
                awaddr_q <= s_axi_awaddr;
            end
            if (w_fire) begin
                w_pending_q <= 1'b1;
                wdata_q <= s_axi_wdata;
                wstrb_q <= s_axi_wstrb;
            end
            if (write_commit) begin
                aw_pending_q <= 1'b0;
                w_pending_q <= 1'b0;
                bvalid_q <= 1'b1;
            end else if (bvalid_q && s_axi_bready) begin
                bvalid_q <= 1'b0;
            end

            if (model_spec_op_fire)
                model_spec_op_q <= MODEL_SPEC_OP_NONE;
            if (core_cmd_fire) begin
                exec_pending_q <= 1'b0;
            end
            if (run_clear_q && core_clear_done)
                run_clear_q <= 1'b0;

            if (ctrl_ack_commit)
                commit_pending_q <= 1'b0;
            if (ctrl_ack_error)
                error_pending_q <= 1'b0;
            if (ctrl_ack_result)
                result_pending_q <= 1'b0;
            if (ctrl_ack_model_spec_error)
                model_spec_error_pending_q <= 1'b0;
            if (ctrl_ack_frontend_error) begin
                frontend_error_pending_q <= 1'b0;
                frontend_error_q <= 8'd0;
            end

            if (core_commit_fire) begin
                commit_pending_q <= 1'b1;
                commit_tag_q <= core_commit_tag;
                commit_model_spec_id_q <= core_commit_model_spec_id;
                commit_model_spec_hash_q <= core_commit_model_spec_hash;
                commit_token_count_q <= core_commit_token_count;
                commit_kv_length_q <= core_commit_kv_length;
                commit_logits_valid_q <= core_commit_logits_valid;
            end
            if (core_error_fire) begin
                error_pending_q <= 1'b1;
                error_tag_q <= core_error_tag;
                error_code_q <= core_error_code;
                error_detail_q <= core_error_detail;
                error_layer_q <= core_error_layer;
                error_stage_q <= core_error_stage;
            end
            if (core_result_fire) begin
                result_pending_q <= 1'b1;
                result_token_q <= core_result_token;
                result_logit_q <= core_result_logit;
                result_error_q <= core_result_error;
                result_status_q <= core_result_status;
            end
            if (core_model_spec_error_fire) begin
                model_spec_error_pending_q <= 1'b1;
                model_spec_error_code_q <= core_model_spec_error_code;
                model_spec_error_layer_q <= core_model_spec_error_layer;
                model_spec_error_word_q <= core_model_spec_error_word;
            end
            if (core_logits_valid && !frontend_error_pending_q) begin
                frontend_error_pending_q <= 1'b1;
                frontend_error_q <= FRONTEND_LOGITS;
            end

            if (write_commit) begin
                case (awaddr_q)
                    ENGINE_REG_OFF_MODEL_SPEC_ID:
                        model_spec_id_stage_q <= merge_wstrb(
                            model_spec_id_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_HASH_LO:
                        model_spec_hash_lo_stage_q <= merge_wstrb(
                            model_spec_hash_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_HASH_HI:
                        model_spec_hash_hi_stage_q <= merge_wstrb(
                            model_spec_hash_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_COUNT:
                        model_spec_layer_count_stage_q <= merge_wstrb(
                            model_spec_layer_count_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_HIDDEN_BLOCKS:
                        model_spec_hidden_blocks_stage_q <= merge_wstrb(
                            model_spec_hidden_blocks_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_FFN_BLOCKS:
                        model_spec_ffn_blocks_stage_q <= merge_wstrb(
                            model_spec_ffn_blocks_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_Q_HEADS:
                        model_spec_q_heads_stage_q <= merge_wstrb(
                            model_spec_q_heads_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_KV_HEADS:
                        model_spec_kv_heads_stage_q <= merge_wstrb(
                            model_spec_kv_heads_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_HEAD_DIM:
                        model_spec_head_dim_stage_q <= merge_wstrb(
                            model_spec_head_dim_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_WEIGHT_FMT:
                        model_spec_weight_fmt_stage_q <= merge_wstrb(
                            model_spec_weight_fmt_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_CONTEXT_LIMIT:
                        model_spec_context_limit_stage_q <= merge_wstrb(
                            model_spec_context_limit_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_VOCAB_ROWS:
                        model_spec_vocab_rows_stage_q <= merge_wstrb(
                            model_spec_vocab_rows_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_EMBED_ADDR_LO:
                        model_spec_embed_addr_lo_stage_q <= merge_wstrb(
                            model_spec_embed_addr_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_EMBED_ADDR_HI:
                        model_spec_embed_addr_hi_stage_q <= merge_wstrb(
                            model_spec_embed_addr_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LM_ADDR_LO:
                        model_spec_lm_addr_lo_stage_q <= merge_wstrb(
                            model_spec_lm_addr_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LM_ADDR_HI:
                        model_spec_lm_addr_hi_stage_q <= merge_wstrb(
                            model_spec_lm_addr_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_FINAL_ADDR_LO:
                        model_spec_final_addr_lo_stage_q <= merge_wstrb(
                            model_spec_final_addr_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_FINAL_ADDR_HI:
                        model_spec_final_addr_hi_stage_q <= merge_wstrb(
                            model_spec_final_addr_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_ROPE_ADDR_LO:
                        model_spec_rope_addr_lo_stage_q <= merge_wstrb(
                            model_spec_rope_addr_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_ROPE_ADDR_HI:
                        model_spec_rope_addr_hi_stage_q <= merge_wstrb(
                            model_spec_rope_addr_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_INDEX:
                        model_spec_layer_index_stage_q <= merge_wstrb(
                            model_spec_layer_index_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_WORD:
                        model_spec_layer_word_stage_q <= merge_wstrb(
                            model_spec_layer_word_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_DATA_LO:
                        model_spec_layer_data_lo_stage_q <= merge_wstrb(
                            model_spec_layer_data_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_DATA_HI:
                        model_spec_layer_data_hi_stage_q <= merge_wstrb(
                            model_spec_layer_data_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_KV_CAPACITY:
                        cmd_kv_capacity_stage_q <= merge_wstrb(
                            cmd_kv_capacity_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TAG:
                        cmd_tag_stage_q <= merge_wstrb(
                            cmd_tag_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_ID:
                        cmd_model_spec_id_stage_q <= merge_wstrb(
                            cmd_model_spec_id_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_HASH_LO:
                        cmd_model_spec_hash_lo_stage_q <= merge_wstrb(
                            cmd_model_spec_hash_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_HASH_HI:
                        cmd_model_spec_hash_hi_stage_q <= merge_wstrb(
                            cmd_model_spec_hash_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_SHAPE:
                        cmd_shape_stage_q <= merge_wstrb(
                            cmd_shape_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_POSITION_BASE:
                        cmd_position_stage_q <= merge_wstrb(
                            cmd_position_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_KV_BASE_LO:
                        cmd_kv_base_lo_stage_q <= merge_wstrb(
                            cmd_kv_base_lo_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_KV_BASE_HI:
                        cmd_kv_base_hi_stage_q <= merge_wstrb(
                            cmd_kv_base_hi_stage_q, wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN0:
                        cmd_token_stage_q[0] <= merge_wstrb(
                            cmd_token_stage_q[0], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN1:
                        cmd_token_stage_q[1] <= merge_wstrb(
                            cmd_token_stage_q[1], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN2:
                        cmd_token_stage_q[2] <= merge_wstrb(
                            cmd_token_stage_q[2], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN3:
                        cmd_token_stage_q[3] <= merge_wstrb(
                            cmd_token_stage_q[3], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN4:
                        cmd_token_stage_q[4] <= merge_wstrb(
                            cmd_token_stage_q[4], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN5:
                        cmd_token_stage_q[5] <= merge_wstrb(
                            cmd_token_stage_q[5], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN6:
                        cmd_token_stage_q[6] <= merge_wstrb(
                            cmd_token_stage_q[6], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_CMD_TOKEN7:
                        cmd_token_stage_q[7] <= merge_wstrb(
                            cmd_token_stage_q[7], wdata_q, wstrb_q);
                    ENGINE_REG_OFF_METRICS_INDEX:
                        if (wstrb_q[0])
                            metrics_index_q <= wdata_q[6:0];
                    default: ;
                endcase
            end

            // Snapshot well-formed command payloads independently from command
            // admission.  The one-entry operation mailboxes below still apply
            // every busy/ready/event check before enqueueing a command.  A
            // rejected attempt can therefore update only an inactive shadow,
            // keeping wide capture enables off the core readiness cones.
            if (begin_snapshot_attempt) begin
                begin_id_q <= model_spec_id_stage_q;
                begin_hash_q <= {model_spec_hash_hi_stage_q,
                                 model_spec_hash_lo_stage_q};
                begin_layer_count_q <= model_spec_layer_count_stage_q[5:0];
                begin_hidden_blocks_q <= model_spec_hidden_blocks_stage_q[7:0];
                begin_ffn_blocks_q <= model_spec_ffn_blocks_stage_q[9:0];
                begin_q_heads_q <= model_spec_q_heads_stage_q[5:0];
                begin_kv_heads_q <= model_spec_kv_heads_stage_q[3:0];
                begin_head_dim_q <= model_spec_head_dim_stage_q[7:0];
                begin_weight_fmt_q <= model_spec_weight_fmt_stage_q[1:0];
                begin_context_limit_q <= model_spec_context_limit_stage_q[16:0];
                begin_vocab_rows_q <= model_spec_vocab_rows_stage_q[17:0];
                begin_embed_addr_q <= {model_spec_embed_addr_hi_stage_q,
                                       model_spec_embed_addr_lo_stage_q};
                begin_lm_addr_q <= {model_spec_lm_addr_hi_stage_q,
                                    model_spec_lm_addr_lo_stage_q};
                begin_final_addr_q <= {model_spec_final_addr_hi_stage_q,
                                       model_spec_final_addr_lo_stage_q};
                begin_rope_addr_q <= {model_spec_rope_addr_hi_stage_q,
                                      model_spec_rope_addr_lo_stage_q};
            end

            if (execute_snapshot_attempt) begin
                exec_tag_q <= cmd_tag_stage_q;
                exec_model_spec_id_q <= cmd_model_spec_id_stage_q;
                exec_model_spec_hash_q <= {cmd_model_spec_hash_hi_stage_q,
                                        cmd_model_spec_hash_lo_stage_q};
                exec_token_count_q <= cmd_shape_stage_q[3:0];
                exec_lane_mask_q <= cmd_shape_stage_q[15:8];
                exec_token_ids_q <= {
                    cmd_token_stage_q[7], cmd_token_stage_q[6],
                    cmd_token_stage_q[5], cmd_token_stage_q[4],
                    cmd_token_stage_q[3], cmd_token_stage_q[2],
                    cmd_token_stage_q[1], cmd_token_stage_q[0]
                };
                exec_position_q <= cmd_position_stage_q[16:0];
                exec_kv_base_q <= {cmd_kv_base_hi_stage_q,
                                   cmd_kv_base_lo_stage_q};
                exec_kv_capacity_q <= cmd_kv_capacity_stage_q[16:0];
                // This enables FINAL_NORM + LM greedy result. Full-logit
                // publication remains independently disabled below.
                exec_emit_logits_q <= cmd_shape_stage_q[16];
            end

            if (ctrl_write) begin
                if (ctrl_reserved_bad) begin
                    if (!frontend_error_pending_q) begin
                        frontend_error_pending_q <= 1'b1;
                        frontend_error_q <= FRONTEND_RESERVED;
                    end
                end else if (ctrl_cmd_collision) begin
                    if (!frontend_error_pending_q ||
                        ctrl_ack_frontend_error) begin
                        frontend_error_pending_q <= 1'b1;
                        frontend_error_q <= FRONTEND_COLLISION;
                    end
                end else begin
                    case (ctrl_cmd_bits)
                        6'b000001: begin
                            if ((model_spec_op_q == MODEL_SPEC_OP_NONE) &&
                                !exec_pending_q && !core_busy &&
                                !run_clear_q && !event_pending &&
                                core_model_spec_clear_ready)
                                model_spec_op_q <= MODEL_SPEC_OP_CLEAR;
                            else if (!frontend_error_pending_q ||
                                     ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_MODEL_SPEC_OP;
                            end
                        end
                        6'b000010: begin
                            if ((model_spec_op_q == MODEL_SPEC_OP_NONE) &&
                                !exec_pending_q && !core_busy &&
                                !run_clear_q && !event_pending &&
                                begin_stage_reserved_ok &&
                                begin_stage_address_ok &&
                                core_model_spec_begin_ready) begin
                                model_spec_op_q <= MODEL_SPEC_OP_BEGIN;
                            end else if (!frontend_error_pending_q ||
                                         ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_MODEL_SPEC_OP;
                            end
                        end
                        6'b000100: begin
                            if ((model_spec_op_q == MODEL_SPEC_OP_NONE) &&
                                !exec_pending_q && !core_busy &&
                                !run_clear_q && !event_pending &&
                                layer_stage_reserved_ok &&
                                layer_stage_address_ok &&
                                core_model_spec_layer_ready) begin
                                model_spec_op_q <= MODEL_SPEC_OP_LAYER;
                                layer_index_q <= model_spec_layer_index_stage_q[5:0];
                                layer_word_q <= model_spec_layer_word_stage_q[2:0];
                                layer_data_q <= {model_spec_layer_data_hi_stage_q,
                                                 model_spec_layer_data_lo_stage_q};
                            end else if (!frontend_error_pending_q ||
                                         ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_MODEL_SPEC_OP;
                            end
                        end
                        6'b001000: begin
                            if ((model_spec_op_q == MODEL_SPEC_OP_NONE) &&
                                !exec_pending_q && !core_busy &&
                                !run_clear_q && !event_pending &&
                                core_model_spec_seal_ready)
                                model_spec_op_q <= MODEL_SPEC_OP_SEAL;
                            else if (!frontend_error_pending_q ||
                                     ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_MODEL_SPEC_OP;
                            end
                        end
                        6'b010000: begin
                            if (execute_available && core_cmd_ready &&
                                execute_stage_reserved_ok &&
                                execute_stage_address_ok) begin
                                exec_pending_q <= 1'b1;
                            end else if (!frontend_error_pending_q ||
                                         ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_EXECUTE;
                            end
                        end
                        6'b100000: begin
                            if (!run_clear_q &&
                                (model_spec_op_q == MODEL_SPEC_OP_NONE)) begin
                                run_clear_q <= 1'b1;
                                exec_pending_q <= 1'b0;
                            end else if (!frontend_error_pending_q ||
                                         ctrl_ack_frontend_error) begin
                                frontend_error_pending_q <= 1'b1;
                                frontend_error_q <= FRONTEND_CLEAR;
                            end
                        end
                        default: ;
                    endcase
                end
            end
        end
    end

    // The read response is retained verbatim under R-channel backpressure.
    always @(posedge clk) begin
        if (!rst_n) begin
            rvalid_q <= 1'b0;
            rdata_q <= 32'd0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                rvalid_q <= 1'b1;
                case (s_axi_araddr)
                    ENGINE_REG_OFF_ID: rdata_q <= ENGINE_REG_RST_ID;
                    ENGINE_REG_OFF_VERSION: rdata_q <= ENGINE_REG_RST_VERSION;
                    ENGINE_REG_OFF_LAYOUT_HASH_LO:
                        rdata_q <= ENGINE_REG_RST_LAYOUT_HASH_LO;
                    ENGINE_REG_OFF_LAYOUT_HASH_HI:
                        rdata_q <= ENGINE_REG_RST_LAYOUT_HASH_HI;
                    ENGINE_REG_OFF_STATUS: rdata_q <= status_word;
                    ENGINE_REG_OFF_CYCLES_LO:
                        rdata_q <= metrics_total_cycles[31:0];
                    ENGINE_REG_OFF_CYCLES_HI:
                        rdata_q <= metrics_total_cycles[63:32];
                    ENGINE_REG_OFF_CLK_HZ: rdata_q <= CLK_HZ;
                    ENGINE_REG_OFF_ACTIVE_MODEL_SPEC_ID:
                        rdata_q <= core_active_model_spec_id;
                    ENGINE_REG_OFF_ACTIVE_MODEL_SPEC_HASH_LO:
                        rdata_q <= core_active_model_spec_hash[31:0];
                    ENGINE_REG_OFF_ACTIVE_MODEL_SPEC_HASH_HI:
                        rdata_q <= core_active_model_spec_hash[63:32];
                    ENGINE_REG_OFF_DEBUG: rdata_q <= debug_word;
                    ENGINE_REG_OFF_FRONTEND_ERROR:
                        rdata_q <= {24'd0, frontend_error_q};
                    ENGINE_REG_OFF_AXI_MASTERS:
                        rdata_q <= ENGINE_REG_RST_AXI_MASTERS;
                    ENGINE_REG_OFF_MODEL_SPEC_ID: rdata_q <= model_spec_id_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_HASH_LO:
                        rdata_q <= model_spec_hash_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_HASH_HI:
                        rdata_q <= model_spec_hash_hi_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_COUNT:
                        rdata_q <= model_spec_layer_count_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_HIDDEN_BLOCKS:
                        rdata_q <= model_spec_hidden_blocks_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_FFN_BLOCKS:
                        rdata_q <= model_spec_ffn_blocks_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_Q_HEADS:
                        rdata_q <= model_spec_q_heads_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_KV_HEADS:
                        rdata_q <= model_spec_kv_heads_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_HEAD_DIM:
                        rdata_q <= model_spec_head_dim_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_WEIGHT_FMT:
                        rdata_q <= model_spec_weight_fmt_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_CONTEXT_LIMIT:
                        rdata_q <= model_spec_context_limit_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_VOCAB_ROWS:
                        rdata_q <= model_spec_vocab_rows_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_EMBED_ADDR_LO:
                        rdata_q <= model_spec_embed_addr_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_EMBED_ADDR_HI:
                        rdata_q <= model_spec_embed_addr_hi_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LM_ADDR_LO:
                        rdata_q <= model_spec_lm_addr_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LM_ADDR_HI:
                        rdata_q <= model_spec_lm_addr_hi_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_FINAL_ADDR_LO:
                        rdata_q <= model_spec_final_addr_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_FINAL_ADDR_HI:
                        rdata_q <= model_spec_final_addr_hi_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_ROPE_ADDR_LO:
                        rdata_q <= model_spec_rope_addr_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_ROPE_ADDR_HI:
                        rdata_q <= model_spec_rope_addr_hi_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_INDEX:
                        rdata_q <= model_spec_layer_index_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_WORD:
                        rdata_q <= model_spec_layer_word_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_DATA_LO:
                        rdata_q <= model_spec_layer_data_lo_stage_q;
                    ENGINE_REG_OFF_MODEL_SPEC_LAYER_DATA_HI:
                        rdata_q <= model_spec_layer_data_hi_stage_q;
                    ENGINE_REG_OFF_CMD_KV_CAPACITY:
                        rdata_q <= cmd_kv_capacity_stage_q;
                    ENGINE_REG_OFF_CMD_TAG: rdata_q <= cmd_tag_stage_q;
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_ID:
                        rdata_q <= cmd_model_spec_id_stage_q;
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_HASH_LO:
                        rdata_q <= cmd_model_spec_hash_lo_stage_q;
                    ENGINE_REG_OFF_CMD_MODEL_SPEC_HASH_HI:
                        rdata_q <= cmd_model_spec_hash_hi_stage_q;
                    ENGINE_REG_OFF_CMD_SHAPE: rdata_q <= cmd_shape_stage_q;
                    ENGINE_REG_OFF_CMD_POSITION_BASE:
                        rdata_q <= cmd_position_stage_q;
                    ENGINE_REG_OFF_CMD_KV_BASE_LO:
                        rdata_q <= cmd_kv_base_lo_stage_q;
                    ENGINE_REG_OFF_CMD_KV_BASE_HI:
                        rdata_q <= cmd_kv_base_hi_stage_q;
                    ENGINE_REG_OFF_CMD_TOKEN0: rdata_q <= cmd_token_stage_q[0];
                    ENGINE_REG_OFF_CMD_TOKEN1: rdata_q <= cmd_token_stage_q[1];
                    ENGINE_REG_OFF_CMD_TOKEN2: rdata_q <= cmd_token_stage_q[2];
                    ENGINE_REG_OFF_CMD_TOKEN3: rdata_q <= cmd_token_stage_q[3];
                    ENGINE_REG_OFF_CMD_TOKEN4: rdata_q <= cmd_token_stage_q[4];
                    ENGINE_REG_OFF_CMD_TOKEN5: rdata_q <= cmd_token_stage_q[5];
                    ENGINE_REG_OFF_CMD_TOKEN6: rdata_q <= cmd_token_stage_q[6];
                    ENGINE_REG_OFF_CMD_TOKEN7: rdata_q <= cmd_token_stage_q[7];
                    ENGINE_REG_OFF_COMMIT_TAG: rdata_q <= commit_tag_q;
                    ENGINE_REG_OFF_COMMIT_MODEL_SPEC_ID:
                        rdata_q <= commit_model_spec_id_q;
                    ENGINE_REG_OFF_COMMIT_MODEL_SPEC_HASH_LO:
                        rdata_q <= commit_model_spec_hash_q[31:0];
                    ENGINE_REG_OFF_COMMIT_MODEL_SPEC_HASH_HI:
                        rdata_q <= commit_model_spec_hash_q[63:32];
                    ENGINE_REG_OFF_COMMIT_INFO: rdata_q <= commit_info_word;
                    ENGINE_REG_OFF_ERROR_TAG: rdata_q <= error_tag_q;
                    ENGINE_REG_OFF_ERROR_CODE_DETAIL:
                        rdata_q <= error_code_detail_word;
                    ENGINE_REG_OFF_ERROR_LOCATION:
                        rdata_q <= error_location_word;
                    ENGINE_REG_OFF_RESULT_TOKEN_STATUS:
                        rdata_q <= result_token_status_word;
                    ENGINE_REG_OFF_RESULT_LOGIT: rdata_q <= result_logit_q;
                    ENGINE_REG_OFF_MODEL_SPEC_ERROR:
                        rdata_q <= model_spec_error_word;
                    ENGINE_REG_OFF_METRICS_SCHEMA:
                        rdata_q <= metrics_schema;
                    ENGINE_REG_OFF_METRICS_CAPABILITIES:
                        rdata_q <= metrics_capabilities;
                    ENGINE_REG_OFF_METRICS_STATUS:
                        rdata_q <= metrics_status;
                    ENGINE_REG_OFF_METRICS_TAG:
                        rdata_q <= metrics_tag;
                    ENGINE_REG_OFF_METRICS_INDEX:
                        rdata_q <= {25'd0, metrics_index_q};
                    ENGINE_REG_OFF_METRICS_DATA_LO:
                        rdata_q <= metrics_data_lo;
                    ENGINE_REG_OFF_METRICS_DATA_HI:
                        rdata_q <= metrics_data_hi;
                    ENGINE_REG_OFF_METRICS_OVERFLOW0:
                        rdata_q <= metrics_overflow0;
                    ENGINE_REG_OFF_METRICS_OVERFLOW1:
                        rdata_q <= metrics_overflow1;
                    ENGINE_REG_OFF_METRICS_OVERFLOW2:
                        rdata_q <= metrics_overflow2;
                    ENGINE_REG_OFF_METRICS_OVERFLOW3:
                        rdata_q <= metrics_overflow3;
                    default: rdata_q <= 32'd0;
                endcase
            end else if (rvalid_q && s_axi_rready) begin
                rvalid_q <= 1'b0;
            end
        end
    end

    engine_metrics u_metrics (
        .clk(clk),
        .rst_n(rst_n),
        .start(core_cmd_fire),
        .start_tag(exec_tag_q),
        .finish(metrics_finish),
        .finish_outcome(metrics_finish_outcome),
        .acknowledge(ctrl_ack_metrics),
        .core_stage_active(core_metrics_stage_active),
        .core_stage(core_metrics_stage),
        .core_stage_call(core_trace_valid),
        .core_stage_call_id(core_trace_stage),
        .projection_probe(core_metrics_projection_probe),
        .weight_axi_r_beats(core_metrics_weight_axi_r_beats),
        .weight_axi_r_gap_ports(core_metrics_weight_axi_r_gap_ports),
        .weight_zip_skew(core_metrics_weight_zip_skew),
        .history_axi_r_beats(core_metrics_history_axi_r_beats),
        .kv_axi_w_beat(core_metrics_kv_axi_w_beat),
        .read_index(metrics_index_q),
        .read_data_lo(metrics_data_lo),
        .read_data_hi(metrics_data_hi),
        .schema(metrics_schema),
        .capabilities(metrics_capabilities),
        .status(metrics_status),
        .snapshot_tag(metrics_tag),
        .overflow0(metrics_overflow0),
        .overflow1(metrics_overflow1),
        .overflow2(metrics_overflow2),
        .overflow3(metrics_overflow3),
        .total_cycles(metrics_total_cycles),
        .recording(metrics_recording),
        .sealing(metrics_sealing),
        .snapshot_valid(metrics_snapshot_valid)
    );

    engine_datapath #(
        .ADDR_W(ADDR_W),
        .EMIT_FULL_LOGITS(0)
    ) u_datapath (
        .clk(clk),
        .rst_n(rst_n),
        .run_clear(run_clear_q),
        .clear_done(core_clear_done),

        .model_spec_clear_valid(core_model_spec_clear_valid),
        .model_spec_clear_ready(core_model_spec_clear_ready),
        .model_spec_begin_valid(core_model_spec_begin_valid),
        .model_spec_begin_ready(core_model_spec_begin_ready),
        .model_spec_begin_id(begin_id_q),
        .model_spec_begin_hash(begin_hash_q),
        .model_spec_begin_layer_count(begin_layer_count_q),
        .model_spec_begin_hidden_blocks(begin_hidden_blocks_q),
        .model_spec_begin_ffn_blocks(begin_ffn_blocks_q),
        .model_spec_begin_q_heads(begin_q_heads_q),
        .model_spec_begin_kv_heads(begin_kv_heads_q),
        .model_spec_begin_head_dim(begin_head_dim_q),
        .model_spec_begin_weight_fmt(begin_weight_fmt_q),
        .model_spec_begin_context_limit(begin_context_limit_q),
        .model_spec_begin_vocab_rows(begin_vocab_rows_q),
        .model_spec_begin_embed_addr(begin_embed_addr_q),
        .model_spec_begin_lm_head_addr(begin_lm_addr_q),
        .model_spec_begin_final_norm_addr(begin_final_addr_q),
        .model_spec_begin_rope_table_addr(begin_rope_addr_q),
        .model_spec_layer_wr_valid(core_model_spec_layer_valid),
        .model_spec_layer_wr_ready(core_model_spec_layer_ready),
        .model_spec_layer_wr_layer(layer_index_q),
        .model_spec_layer_wr_word(layer_word_q),
        .model_spec_layer_wr_data(layer_data_q),
        .model_spec_seal_valid(core_model_spec_seal_valid),
        .model_spec_seal_ready(core_model_spec_seal_ready),
        .model_spec_cfg_error_valid(core_model_spec_error_valid),
        .model_spec_cfg_error_ready(core_model_spec_error_ready),
        .model_spec_cfg_error_code(core_model_spec_error_code),
        .model_spec_cfg_error_layer(core_model_spec_error_layer),
        .model_spec_cfg_error_word(core_model_spec_error_word),
        .model_spec_loading(core_model_spec_loading),
        .model_spec_sealed(core_model_spec_sealed),
        .interface_version(core_interface_version),
        .layer_layout_hash(core_layout_hash),
        .active_model_spec_id(core_active_model_spec_id),
        .active_model_spec_hash(core_active_model_spec_hash),

        .cmd_valid(core_cmd_valid),
        .cmd_ready(core_cmd_ready),
        .cmd_tag(exec_tag_q),
        .cmd_model_spec_id(exec_model_spec_id_q),
        .cmd_model_spec_hash(exec_model_spec_hash_q),
        .cmd_token_count(exec_token_count_q),
        .cmd_lane_mask(exec_lane_mask_q),
        .cmd_token_ids(exec_token_ids_q),
        .cmd_position_base(exec_position_q),
        .cmd_kv_base(exec_kv_base_q),
        .cmd_kv_capacity(exec_kv_capacity_q),
        .cmd_emit_logits(exec_emit_logits_q),

        .commit_valid(core_commit_valid),
        .commit_ready(core_commit_ready),
        .commit_tag(core_commit_tag),
        .commit_model_spec_id(core_commit_model_spec_id),
        .commit_model_spec_hash(core_commit_model_spec_hash),
        .commit_token_count(core_commit_token_count),
        .commit_kv_length(core_commit_kv_length),
        .commit_logits_valid(core_commit_logits_valid),
        .error_valid(core_error_valid),
        .error_ready(core_error_ready),
        .error_tag(core_error_tag),
        .error_code(core_error_code),
        .error_detail(core_error_detail),
        .error_layer(core_error_layer),
        .error_stage(core_error_stage),
        .logits_valid(core_logits_valid),
        .logits_ready(1'b1),
        .logits_row(core_logits_row),
        .logits_data(core_logits_data),
        .logits_last(core_logits_last),
        .result_valid(core_result_valid),
        .result_ready(core_result_ready),
        .result_token(core_result_token),
        .result_logit(core_result_logit),
        .result_error(core_result_error),
        .result_status(core_result_status),
        .busy(core_busy),
        .debug_layer(core_debug_layer),
        .debug_stage(core_debug_stage),
        .metrics_stage_active(core_metrics_stage_active),
        .metrics_stage(core_metrics_stage),
        .metrics_projection_probe(core_metrics_projection_probe),
        .metrics_weight_axi_r_beats(core_metrics_weight_axi_r_beats),
        .metrics_weight_axi_r_gap_ports(
            core_metrics_weight_axi_r_gap_ports),
        .metrics_weight_zip_skew(core_metrics_weight_zip_skew),
        .metrics_history_axi_r_beats(core_metrics_history_axi_r_beats),
        .metrics_kv_axi_w_beat(core_metrics_kv_axi_w_beat),
        .trace_valid(core_trace_valid),
        .trace_layer(core_trace_layer),
        .trace_stage(core_trace_stage),
        .protocol_error(core_protocol_error),

        .weight_axi_araddr(core_weight_araddr),
        .weight_axi_arlen(core_weight_arlen),
        .weight_axi_arsize(core_weight_arsize),
        .weight_axi_arburst(core_weight_arburst),
        .weight_axi_arvalid(core_weight_arvalid),
        .weight_axi_arready({m_axi_w3_arready, m_axi_w2_arready,
                             m_axi_w1_arready, m_axi_w0_arready}),
        .weight_axi_rdata({m_axi_w3_rdata, m_axi_w2_rdata,
                           m_axi_w1_rdata, m_axi_w0_rdata}),
        .weight_axi_rresp({m_axi_w3_rresp, m_axi_w2_rresp,
                           m_axi_w1_rresp, m_axi_w0_rresp}),
        .weight_axi_rlast({m_axi_w3_rlast, m_axi_w2_rlast,
                           m_axi_w1_rlast, m_axi_w0_rlast}),
        .weight_axi_rvalid({m_axi_w3_rvalid, m_axi_w2_rvalid,
                            m_axi_w1_rvalid, m_axi_w0_rvalid}),
        .weight_axi_rready(core_weight_rready),

        .hist_k_axi_araddr(m_axi_hist_k_araddr),
        .hist_k_axi_arlen(m_axi_hist_k_arlen),
        .hist_k_axi_arsize(m_axi_hist_k_arsize),
        .hist_k_axi_arburst(m_axi_hist_k_arburst),
        .hist_k_axi_arvalid(m_axi_hist_k_arvalid),
        .hist_k_axi_arready(m_axi_hist_k_arready),
        .hist_k_axi_rdata(m_axi_hist_k_rdata),
        .hist_k_axi_rresp(m_axi_hist_k_rresp),
        .hist_k_axi_rlast(m_axi_hist_k_rlast),
        .hist_k_axi_rvalid(m_axi_hist_k_rvalid),
        .hist_k_axi_rready(m_axi_hist_k_rready),
        .hist_v_axi_araddr(m_axi_hist_v_araddr),
        .hist_v_axi_arlen(m_axi_hist_v_arlen),
        .hist_v_axi_arsize(m_axi_hist_v_arsize),
        .hist_v_axi_arburst(m_axi_hist_v_arburst),
        .hist_v_axi_arvalid(m_axi_hist_v_arvalid),
        .hist_v_axi_arready(m_axi_hist_v_arready),
        .hist_v_axi_rdata(m_axi_hist_v_rdata),
        .hist_v_axi_rresp(m_axi_hist_v_rresp),
        .hist_v_axi_rlast(m_axi_hist_v_rlast),
        .hist_v_axi_rvalid(m_axi_hist_v_rvalid),
        .hist_v_axi_rready(m_axi_hist_v_rready),
        .kv_axi_awaddr(m_axi_kv_awaddr),
        .kv_axi_awlen(m_axi_kv_awlen),
        .kv_axi_awsize(m_axi_kv_awsize),
        .kv_axi_awburst(m_axi_kv_awburst),
        .kv_axi_awvalid(m_axi_kv_awvalid),
        .kv_axi_awready(m_axi_kv_awready),
        .kv_axi_wdata(m_axi_kv_wdata),
        .kv_axi_wstrb(m_axi_kv_wstrb),
        .kv_axi_wlast(m_axi_kv_wlast),
        .kv_axi_wvalid(m_axi_kv_wvalid),
        .kv_axi_wready(m_axi_kv_wready),
        .kv_axi_bresp(m_axi_kv_bresp),
        .kv_axi_bvalid(m_axi_kv_bvalid),
        .kv_axi_bready(m_axi_kv_bready)
    );

    assign m_axi_w0_araddr = core_weight_araddr[0*ADDR_W +: ADDR_W];
    assign m_axi_w1_araddr = core_weight_araddr[1*ADDR_W +: ADDR_W];
    assign m_axi_w2_araddr = core_weight_araddr[2*ADDR_W +: ADDR_W];
    assign m_axi_w3_araddr = core_weight_araddr[3*ADDR_W +: ADDR_W];
    assign {m_axi_w3_arlen, m_axi_w2_arlen,
            m_axi_w1_arlen, m_axi_w0_arlen} = core_weight_arlen;
    assign {m_axi_w3_arsize, m_axi_w2_arsize,
            m_axi_w1_arsize, m_axi_w0_arsize} = core_weight_arsize;
    assign {m_axi_w3_arburst, m_axi_w2_arburst,
            m_axi_w1_arburst, m_axi_w0_arburst} = core_weight_arburst;
    assign {m_axi_w3_arvalid, m_axi_w2_arvalid,
            m_axi_w1_arvalid, m_axi_w0_arvalid} = core_weight_arvalid;
    assign {m_axi_w3_rready, m_axi_w2_rready,
            m_axi_w1_rready, m_axi_w0_rready} = core_weight_rready;

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n) begin
            assert(core_interface_version == ENGINE_REG_RST_VERSION);
            assert(core_layout_hash ==
                   {ENGINE_REG_RST_LAYOUT_HASH_HI, ENGINE_REG_RST_LAYOUT_HASH_LO});
            assert(!core_logits_valid);
        end
    end
`endif

    wire [5:0] unused_control_inputs = {s_axi_awprot, s_axi_arprot};
    wire [63:0] unused_observation = {
        core_trace_valid, core_trace_layer, core_trace_stage,
        core_logits_row, core_logits_data, core_logits_last, 1'b0
    };

endmodule

`default_nettype wire
