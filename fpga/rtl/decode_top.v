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
`ifdef VERILATOR
    , input wire         sim_inject_q8_numeric_error
`endif
`ifdef FORMAL
    , output wire [2:0]  formal_ffn_phase
    , output wire        formal_ffn_gate_ready
    , output wire [1:0]  formal_scratch_rd_owner
    , output wire        formal_scratch_rd_rsp_valid
    , output wire        formal_scratch_rd_rsp_ready
    , output wire        formal_section_active
    , output wire        formal_section_done
    , output wire        formal_abort_cleanup
    , output wire [6:0]  formal_scratch_error
    , output wire        formal_capture_fire
    , output wire        formal_capture_record_done
    , output wire [10:0] formal_bank0_record_count
    , output wire        formal_replay_fire
    , output wire        formal_replay_complete
    , output wire        formal_down_kernel_done
    , output wire        formal_kernel_done
    , output wire        formal_abort_strobe
    , output wire        formal_ffn_fault
    , output wire        formal_section_begin_ok
    , output wire        formal_up_start
    , output wire        formal_gate_start
    , output wire        formal_gate_drain_ready
    , output wire        formal_seal_done
    , output wire        formal_down_start
    , output wire        formal_down_complete_ready
    , output wire        formal_producer_done
    , output wire [1:0]  formal_capture_token
    , output wire [8:0]  formal_capture_block
    , output wire [2:0]  formal_capture_beat
    , output wire [1:0]  formal_replay_token
    , output wire [8:0]  formal_replay_block
    , output wire [2:0]  formal_replay_beat
`endif
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
    reg        kernel_start_q;
    reg        scratch_tee_start_q;
    reg        scratch_only_start_q;
    reg        scratch_consumer_start_q;
    reg [1:0]  scratch_mode_q;
    reg [1:0]  scratch_role_q;
    reg [13:0] scratch_rows_q;
    reg [2:0]  scratch_tokens_q;
    reg        scratch_drain_start_strobe;
    reg        scratch_abort_strobe;
    reg        scratch_section_begin_strobe;
    reg        scratch_tee_run_q;
    reg        scratch_only_run_q;
    reg [1:0]  scratch_write_role_q;
    reg [13:0] scratch_write_rows_q;
    reg [2:0]  scratch_write_tokens_q;
    reg        scratch_writer_done_q;
    reg [6:0]  scratch_error_q;
    reg [3:0]  scratch_valid_q;
    reg [13:0] scratch_valid_rows_q [0:3];
    reg [2:0]  scratch_valid_tokens_q [0:3];
    integer scratch_role_i;

    reg        scratch_drain_busy_q;
    reg        scratch_drain_done_q;
    reg [1:0]  scratch_drain_role_q;
    reg [2:0]  scratch_drain_tokens_q;
    reg [2:0]  scratch_drain_token_q;
    reg [10:0] scratch_drain_group_q;
    reg [1:0]  scratch_drain_bank_q;
    reg        scratch_drain_have_group_q;
    reg        scratch_drain_group_valid_q;
    reg [255:0] scratch_drain_group_data_q;
    reg        scratch_drain_emit_valid_q;
    reg [63:0] scratch_drain_emit_data_q;
    reg [10:0] scratch_drain_groups_q;

    localparam [3:0] CONSUMER_IDLE      = 4'd0;
    localparam [3:0] CONSUMER_REQ_GATE  = 4'd1;
    localparam [3:0] CONSUMER_WAIT_GATE = 4'd2;
    localparam [3:0] CONSUMER_REQ_UP    = 4'd3;
    localparam [3:0] CONSUMER_WAIT_UP   = 4'd4;
    localparam [3:0] CONSUMER_ISSUE     = 4'd5;
    localparam [3:0] CONSUMER_DRAIN     = 4'd6;
    reg        scratch_section_active_q;
    reg        scratch_section_done_q;
    reg        scratch_consumer_busy_q;
    reg        scratch_consumer_done_q;
    reg [3:0]  scratch_consumer_state_q;
    reg [2:0]  scratch_consumer_token_q;
    reg [10:0] scratch_consumer_group_q;
    reg [10:0] scratch_consumer_groups_q;
    reg [2:0]  scratch_consumer_lane_q;
    reg [15:0] scratch_consumer_blocks_q;
    reg [15:0] scratch_consumer_records_q;
    reg [15:0] scratch_consumer_total_blocks_q;
    reg [255:0] scratch_consumer_gate_q;
    reg [255:0] scratch_consumer_up_q;

    // V16 streaming FFN lifecycle. The existing section registers remain the
    // software contract; these snapshots and counters own one accepted section.
    reg [13:0] ffn_rows_q;
    reg [2:0]  ffn_tokens_q;
    reg [8:0]  ffn_blocks_q;
    reg        ffn_gate_start_q;
    reg        ffn_down_start_q;
    reg        ffn_gate_run_q;
    reg        ffn_producer_busy_q;
    reg        ffn_producer_done_q;
    reg [1:0]  ffn_capture_token_q;
    reg [8:0]  ffn_capture_block_q;
    reg        ffn_seal_pending_q;
    reg        ffn_replay_active_q;
    reg        ffn_replay_inflight_q;
    reg [1:0]  ffn_replay_token_q;
    reg [8:0]  ffn_replay_block_q;
    reg [2:0]  ffn_capture_beat_q;
    reg [2:0]  ffn_replay_beat_q;
    reg        ffn_capture_complete_q;
    reg        ffn_replay_complete_q;
    reg        ffn_gate_kernel_done_q;
    reg        ffn_gate_packer_done_q;
    reg        ffn_gate_pairer_done_q;
    reg        ffn_down_kernel_done_q;

    localparam [2:0] FFN_IDLE      = 3'd0;
    localparam [2:0] FFN_WAIT_UP   = 3'd1;
    localparam [2:0] FFN_UP_RUN    = 3'd2;
    localparam [2:0] FFN_WAIT_GATE = 3'd3;
    localparam [2:0] FFN_GATE_RUN  = 3'd4;
    localparam [2:0] FFN_SEAL      = 3'd5;
    localparam [2:0] FFN_WAIT_DOWN = 3'd6;
    localparam [2:0] FFN_DOWN_RUN  = 3'd7;
    reg [2:0] ffn_phase_q;
    reg       ffn_fault_q;
    reg       ffn_abort_cleanup_q;

    localparam [1:0] SCRATCH_RD_NONE   = 2'd0;
    localparam [1:0] SCRATCH_RD_DRAIN  = 2'd1;
    localparam [1:0] SCRATCH_RD_PAIRER = 2'd2;
    reg [1:0] scratch_rd_owner_q;

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
    wire [5:0] quantizer_status;
    wire q8_activation_abort;
    wire q8_ingress_abort;
    wire kernel_start;
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
    wire internal_activation_mode = act_mode_q == 2'd3;
    wire raw_activation_mode = (act_mode_q == 2'd2) || internal_activation_mode;
    wire [63:0] native_acts_tdata;
    wire native_acts_tvalid;
    wire native_acts_tready;
    wire raw_acts_tready;
    wire kernel_acts_tready;
    wire swiglu_in_valid;
    wire swiglu_in_ready;
    wire [31:0] swiglu_in_gate;
    wire [31:0] swiglu_in_up;
    wire swiglu_in_last;
    wire swiglu_out_valid;
    wire swiglu_out_ready;
    wire q8_internal_record_done;
    wire [31:0] swiglu_out_data;
    wire swiglu_out_last;
    wire [1:0] swiglu_out_status;

    wire section_abort_now = scratch_abort_strobe;
    wire qualified_kernel_start = kernel_start_q && !section_abort_now &&
                                  !ffn_fault_q;
    wire q8_ingress_start = ffn_gate_start_q ||
                            (qualified_kernel_start && raw_activation_mode &&
                             !ffn_down_start_q);
    wire q8_ingress_internal_mode = ffn_gate_start_q;
    wire [15:0] q8_ingress_blocks = ffn_gate_start_q ?
                                     {9'd0, ffn_rows_q[13:7]} :
                                     num_q1_blocks_q;
    wire [15:0] q8_ingress_cols = ffn_gate_start_q ?
                                   {13'd0, ffn_tokens_q} : num_cols_q;

    q8_ingress u_q8_ingress (
        .clk(clk),
        .rst_n(rst_n),
        .start(q8_ingress_start),
        .abort(q8_ingress_abort),
        .raw_mode(raw_activation_mode || ffn_gate_start_q),
        .internal_mode(q8_ingress_internal_mode),
        .num_q1_blocks(q8_ingress_blocks),
        .num_cols(q8_ingress_cols),
        .s_axis_tdata(s_axis_acts_tdata),
        .s_axis_tvalid(s_axis_acts_tvalid),
        .s_axis_tready(raw_acts_tready),
        .s_axis_tlast(s_axis_acts_tlast),
        .internal_data(swiglu_out_data),
        .internal_last(swiglu_out_last),
        .internal_status(swiglu_out_status),
        .internal_valid(swiglu_out_valid),
        .internal_ready(swiglu_out_ready),
        .internal_record_done(q8_internal_record_done),
        .m_axis_tdata(native_acts_tdata),
        .m_axis_tvalid(native_acts_tvalid),
        .m_axis_tready(native_acts_tready),
        .activation_abort(q8_activation_abort),
        .quantizer_status(quantizer_status)
    );

    wire [63:0] q8_buffer_m_axis_tdata;
    wire q8_buffer_m_axis_tvalid;
    wire q8_buffer_m_axis_tready;
    wire q8_buffer_m_axis_tlast;
    wire q8_buffer_m_axis_error;
    wire q8_buffer_m_axis_bank;
    wire [1:0] q8_buffer_m_axis_token;
    wire [8:0] q8_buffer_m_axis_block;
    wire q8_buffer_replay_selected = ffn_replay_active_q ||
                                     ffn_replay_inflight_q;

    wire q8_buffer_replay_tag_ok = !q8_buffer_m_axis_bank &&
                                   (q8_buffer_m_axis_token ==
                                    ffn_replay_token_q) &&
                                   (q8_buffer_m_axis_block ==
                                    ffn_replay_block_q) &&
                                   (q8_buffer_m_axis_tlast ==
                                    (ffn_replay_beat_q == 3'd4));
    wire q8_buffer_replay_healthy = !q8_buffer_m_axis_error &&
                                     q8_buffer_replay_tag_ok;

    wire [63:0] kernel_acts_tdata = q8_buffer_replay_selected ?
                                    q8_buffer_m_axis_tdata :
                                    (raw_activation_mode ?
                                     native_acts_tdata : s_axis_acts_tdata);
    wire kernel_acts_tvalid = q8_buffer_replay_selected ?
                              (q8_buffer_m_axis_tvalid &&
                               q8_buffer_replay_healthy) :
                              (raw_activation_mode ?
                               native_acts_tvalid : s_axis_acts_tvalid);
    assign native_acts_tready = ffn_producer_busy_q ? q8_buffer_s_axis_tready :
                                (raw_activation_mode && !q8_buffer_replay_selected ?
                                 kernel_acts_tready : 1'b0);
    assign q8_buffer_m_axis_tready = q8_buffer_replay_selected &&
                                     q8_buffer_replay_healthy &&
                                     kernel_acts_tready;
    assign s_axis_acts_tready = internal_activation_mode ? 1'b0 :
                                (raw_activation_mode ? raw_acts_tready :
                                 kernel_acts_tready);

    wire [63:0] kernel_m_axis_tdata;
    wire        kernel_m_axis_tvalid;
    wire        kernel_m_axis_tready;
    wire        kernel_m_axis_tlast;
    wire [7:0]  kernel_m_axis_tkeep;

    localparam [1:0] SCRATCH_MODE_DDR   = 2'd0;
    localparam [1:0] SCRATCH_MODE_TEE   = 2'd1;
    localparam [1:0] SCRATCH_MODE_DRAIN = 2'd2;
    localparam [1:0] SCRATCH_MODE_ONLY  = 2'd3;

    // Tee preflight is deliberately stricter than the leaf.  The current staged
    // result plan materializes complete 16-row GEMM rowblocks, so accepting an
    // 8-row tail here would make the writer and DDR sink disagree on framing.
    wire scratch_tee_shape_ok = (scratch_mode_q == SCRATCH_MODE_TEE) &&
                                ((scratch_role_q == 2'd1) || (scratch_role_q == 2'd2)) &&
                                (scratch_rows_q != 14'd0) &&
                                (scratch_rows_q <= 14'd12288) &&
                                (scratch_rows_q[3:0] == 4'd0) &&
                                (scratch_tokens_q != 3'd0) &&
                                (scratch_tokens_q <= 3'd4) &&
                                (num_rows_q == {18'd0, scratch_rows_q}) &&
                                (num_cols_q == {13'd0, scratch_tokens_q}) &&
                                (num_rowblocks_q == {6'd0, scratch_rows_q[13:4]});
    wire scratch_wr_cfg_ready;
    wire scratch_wr_busy;
    wire scratch_wr_done;
    wire scratch_wr_error;
    wire scratch_wr_tready;
    wire scratch_wr_commit_valid;
    wire [1:0] scratch_wr_commit_bank;
    wire [13:0] scratch_wr_commit_address;
    wire scratch_rd_req_valid;
    wire scratch_rd_req_ready;
    wire [1:0] scratch_rd_req_role;
    wire [2:0] scratch_rd_req_token;
    wire [10:0] scratch_rd_req_group;
    wire scratch_rd_issue_valid;
    wire [13:0] scratch_rd_issue_address;
    wire scratch_rd_rsp_valid;
    wire scratch_rd_rsp_ready;
    wire [255:0] scratch_rd_rsp_data;
    wire scratch_rd_rsp_error;

    wire gate_packer_start_ready;
    wire gate_packer_busy;
    wire gate_packer_done;
    wire gate_packer_error;
    wire ffn_pairer_start_ready;
    wire ffn_pairer_busy;
    wire ffn_pairer_done;
    wire ffn_pairer_error;
    wire q8_buffer_cfg_ready;
    wire q8_buffer_seal_ready;
    wire q8_buffer_seal_done;
    wire q8_buffer_seal_error;
    wire [1:0] q8_buffer_bank_clearing;
    wire [1:0] q8_buffer_bank_active;
    wire [1:0] q8_buffer_bank_valid;
    wire [1:0] q8_buffer_bank_error;
    wire q8_buffer_s_axis_tready;
    wire q8_buffer_cap_record_done;
    wire q8_buffer_cap_record_error;
    wire [10:0] q8_buffer_bank0_record_count;

    wire scratch_idle = !kernel_busy && !kernel_start_q &&
                        !scratch_tee_run_q && !scratch_only_run_q &&
                        !scratch_wr_busy && !scratch_drain_busy_q &&
                        !scratch_consumer_busy_q && !ffn_gate_run_q &&
                        !ffn_producer_busy_q && !ffn_replay_active_q &&
                        !ffn_replay_inflight_q && !scratch_rd_rsp_valid &&
                        (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                        scratch_rd_req_ready;
    wire scratch_consumer_metadata_ok = scratch_valid_q[1] && scratch_valid_q[2] &&
                                        (scratch_valid_rows_q[1] == scratch_rows_q) &&
                                        (scratch_valid_rows_q[2] == scratch_rows_q) &&
                                        (scratch_valid_tokens_q[1] == scratch_tokens_q) &&
                                        (scratch_valid_tokens_q[2] == scratch_tokens_q);
    wire scratch_consumer_shape_ok = internal_activation_mode &&
                                     (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                     scratch_section_active_q &&
                                     scratch_consumer_metadata_ok &&
                                     (scratch_rows_q != 14'd0) &&
                                     (scratch_rows_q <= 14'd12288) &&
                                     (scratch_rows_q[6:0] == 7'd0) &&
                                     (scratch_tokens_q != 3'd0) &&
                                     (scratch_tokens_q <= 3'd4) &&
                                     (num_q1_blocks_q == {9'd0, scratch_rows_q[13:7]}) &&
                                     (num_cols_q == {13'd0, scratch_tokens_q});
    wire ffn_gate_ready = (ffn_phase_q == FFN_WAIT_GATE) &&
                          scratch_valid_q[2] &&
                          (scratch_valid_rows_q[2] == ffn_rows_q) &&
                          (scratch_valid_tokens_q[2] == ffn_tokens_q) &&
                          q8_buffer_bank_active[0] &&
                          !q8_buffer_bank_valid[0] &&
                          !q8_buffer_bank_error[0] &&
                          !ffn_producer_busy_q && !ffn_producer_done_q;
    wire ffn_up_candidate = (ffn_phase_q == FFN_WAIT_UP) &&
                            scratch_section_active_q &&
                            (scratch_mode_q == SCRATCH_MODE_ONLY) &&
                            (scratch_role_q == 2'd2);
    wire ffn_up_shape_ok = ffn_up_candidate &&
                           (act_mode_q == 2'd2) &&
                           (scratch_rows_q == ffn_rows_q) &&
                           (scratch_tokens_q == ffn_tokens_q) &&
                           (num_rows_q == {18'd0, ffn_rows_q}) &&
                           (num_cols_q == {13'd0, ffn_tokens_q}) &&
                           (num_rowblocks_q == {6'd0, ffn_rows_q[13:4]});
    wire ffn_up_preflight_ok = ffn_up_shape_ok && scratch_wr_cfg_ready &&
                               scratch_idle;

    wire ffn_gate_candidate = (ffn_phase_q == FFN_WAIT_GATE) &&
                              scratch_section_active_q &&
                              (scratch_mode_q == SCRATCH_MODE_ONLY) &&
                              (scratch_role_q == 2'd1);
    wire ffn_gate_shape_ok = ffn_gate_candidate && ffn_gate_ready &&
                             (act_mode_q == 2'd1) &&
                             (scratch_rows_q == ffn_rows_q) &&
                             (scratch_tokens_q == ffn_tokens_q) &&
                             (num_rows_q == {18'd0, ffn_rows_q}) &&
                             (num_cols_q == {13'd0, ffn_tokens_q}) &&
                             (num_rowblocks_q == {6'd0, ffn_rows_q[13:4]});
    wire ffn_gate_preflight_ok = ffn_gate_shape_ok && scratch_idle &&
                                 gate_packer_start_ready &&
                                 ffn_pairer_start_ready;

    wire ffn_down_candidate = (ffn_phase_q == FFN_WAIT_DOWN) &&
                              internal_activation_mode &&
                              scratch_section_active_q &&
                              (scratch_mode_q == SCRATCH_MODE_DDR);
    wire ffn_down_shape_ok = ffn_down_candidate && ffn_producer_done_q &&
                             q8_buffer_bank_valid[0] &&
                             !q8_buffer_bank_error[0] &&
                             (scratch_rows_q == ffn_rows_q) &&
                             (scratch_tokens_q == ffn_tokens_q) &&
                             (num_q1_blocks_q == {9'd0, ffn_rows_q[13:7]}) &&
                             (num_cols_q == {13'd0, ffn_tokens_q});
    wire ffn_down_preflight_ok = ffn_down_shape_ok && scratch_idle;

    wire scratch_ddr_preflight_ok = (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                    scratch_idle &&
                                    (ffn_down_candidate ? ffn_down_shape_ok :
                                     (!scratch_section_active_q &&
                                      !internal_activation_mode));
    wire scratch_tee_preflight_ok = scratch_tee_shape_ok && scratch_wr_cfg_ready &&
                                    scratch_idle && !scratch_section_active_q;
    wire scratch_launch_ok = scratch_ddr_preflight_ok ||
                             scratch_tee_preflight_ok ||
                             ffn_up_preflight_ok ||
                             ffn_gate_preflight_ok ||
                             ffn_down_preflight_ok;
    wire scratch_tee_start = scratch_tee_start_q;
    wire scratch_only_start = scratch_only_start_q;
    wire scratch_consumer_start = scratch_consumer_start_q;
    assign kernel_start = qualified_kernel_start;
    wire scratch_start_rejected = start_strobe && !scratch_launch_ok;
    wire scratch_section_shape_ok = (scratch_rows_q != 14'd0) &&
                                    (scratch_rows_q <= 14'd12288) &&
                                    (scratch_rows_q[6:0] == 7'd0) &&
                                    (scratch_tokens_q != 3'd0) &&
                                    (scratch_tokens_q <= 3'd4);
    wire scratch_section_begin_ok = scratch_section_begin_strobe &&
                                    !scratch_abort_strobe &&
                                    (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                    scratch_section_shape_ok &&
                                    q8_buffer_cfg_ready && scratch_idle &&
                                    !scratch_section_active_q;
    wire scratch_section_begin_bad = scratch_section_begin_strobe &&
                                     !scratch_abort_strobe &&
                                     !scratch_section_begin_ok;

    // Complete shape/ownership preflight before presenting a launch to GEMM.
    // The accepted mode pulses move together one cycle later, breaking the
    // scratch-dimension compare cone before the kernel's high-fanout start CEs.
    always @(posedge clk) begin
        if (!rst_n) begin
            kernel_start_q           <= 1'b0;
            scratch_tee_start_q      <= 1'b0;
            scratch_only_start_q     <= 1'b0;
            scratch_consumer_start_q <= 1'b0;
            ffn_gate_start_q         <= 1'b0;
            ffn_down_start_q         <= 1'b0;
        end else begin
            kernel_start_q <= start_strobe && scratch_launch_ok &&
                              !scratch_abort_strobe && !ffn_fault_q;
            scratch_tee_start_q <= start_strobe && scratch_tee_preflight_ok &&
                                   !scratch_abort_strobe && !ffn_fault_q;
            scratch_only_start_q <= start_strobe && ffn_up_preflight_ok &&
                                    !scratch_abort_strobe && !ffn_fault_q;
            scratch_consumer_start_q <= 1'b0;
            ffn_gate_start_q <= start_strobe && ffn_gate_preflight_ok &&
                                !scratch_abort_strobe && !ffn_fault_q;
            ffn_down_start_q <= start_strobe && ffn_down_preflight_ok &&
                                !scratch_abort_strobe && !ffn_fault_q;
        end
    end

    // Atomic fork: neither downstream observes TVALID unless the other is
    // ready, and the kernel advances only when both accept on the same edge.
    wire scratch_tee_active = scratch_tee_run_q;
    wire scratch_only_active = scratch_only_run_q;
    wire scratch_writer_active = scratch_tee_active || scratch_only_active;
    wire scratch_sink_valid = scratch_writer_active && kernel_m_axis_tvalid &&
                              (scratch_only_active || m_axis_tready);
    wire ddr_kernel_valid = kernel_m_axis_tvalid && !scratch_only_active &&
                            !ffn_gate_run_q &&
                            (!scratch_tee_active || scratch_wr_tready);
    wire gate_packer_s_axis_tready;
    assign kernel_m_axis_tready = ffn_gate_run_q ? gate_packer_s_axis_tready :
                                  (scratch_only_active ? scratch_wr_tready :
                                  (scratch_tee_active ?
                                   (m_axis_tready && scratch_wr_tready) :
                                   (scratch_drain_busy_q ? 1'b0 : m_axis_tready)));

    wire scratch_writer_abort = scratch_abort_strobe ||
                                (scratch_writer_active &&
                                 (q8_activation_abort || activation_error ||
                                  (kernel_done && scratch_wr_busy)));

    wire [255:0] gate_packer_m_axis_tdata;
    wire gate_packer_m_axis_tvalid;
    wire gate_packer_m_axis_tready;
    wire gate_packer_m_axis_tlast;
    wire [1:0] gate_packer_m_axis_token;
    wire [8:0] gate_packer_m_axis_block;
    wire [1:0] gate_packer_m_axis_group;
    wire ffn_pipeline_abort = section_abort_now;

    section_gate_packer u_section_gate_packer (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(ffn_gate_start_q),
        .start_ready(gate_packer_start_ready),
        .start_tokens(ffn_tokens_q),
        .start_blocks(ffn_blocks_q),
        .abort_run(ffn_pipeline_abort),
        .busy(gate_packer_busy),
        .done(gate_packer_done),
        .error(gate_packer_error),
        .s_axis_tdata(kernel_m_axis_tdata),
        .s_axis_tkeep(kernel_m_axis_tkeep),
        .s_axis_tvalid(ffn_gate_run_q && kernel_m_axis_tvalid),
        .s_axis_tready(gate_packer_s_axis_tready),
        .s_axis_tlast(kernel_m_axis_tlast),
        .m_axis_tdata(gate_packer_m_axis_tdata),
        .m_axis_tvalid(gate_packer_m_axis_tvalid),
        .m_axis_tready(gate_packer_m_axis_tready),
        .m_axis_tlast(gate_packer_m_axis_tlast),
        .m_axis_token(gate_packer_m_axis_token),
        .m_axis_block(gate_packer_m_axis_block),
        .m_axis_group(gate_packer_m_axis_group)
    );

    wire ffn_pairer_rd_req_valid;
    wire ffn_pairer_rd_req_ready;
    wire [1:0] ffn_pairer_rd_req_role;
    wire [2:0] ffn_pairer_rd_req_token;
    wire [10:0] ffn_pairer_rd_req_group;
    wire ffn_pairer_rd_rsp_valid;
    wire ffn_pairer_rd_rsp_ready;
    wire ffn_pairer_out_valid;
    wire ffn_pairer_out_ready;
    wire [31:0] ffn_pairer_out_gate;
    wire [31:0] ffn_pairer_out_up;
    wire ffn_pairer_out_last;

    section_ffn_pairer u_section_ffn_pairer (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(ffn_gate_start_q),
        .start_ready(ffn_pairer_start_ready),
        .start_tokens(ffn_tokens_q),
        .start_blocks(ffn_blocks_q),
        .abort_run(ffn_pipeline_abort),
        .busy(ffn_pairer_busy),
        .done(ffn_pairer_done),
        .error(ffn_pairer_error),
        .s_axis_tdata(gate_packer_m_axis_tdata),
        .s_axis_tvalid(gate_packer_m_axis_tvalid),
        .s_axis_tready(gate_packer_m_axis_tready),
        .s_axis_tlast(gate_packer_m_axis_tlast),
        .s_axis_token(gate_packer_m_axis_token),
        .s_axis_block(gate_packer_m_axis_block),
        .s_axis_group(gate_packer_m_axis_group),
        .rd_req_valid(ffn_pairer_rd_req_valid),
        .rd_req_ready(ffn_pairer_rd_req_ready),
        .rd_req_role(ffn_pairer_rd_req_role),
        .rd_req_token(ffn_pairer_rd_req_token),
        .rd_req_group(ffn_pairer_rd_req_group),
        .rd_rsp_valid(ffn_pairer_rd_rsp_valid),
        .rd_rsp_ready(ffn_pairer_rd_rsp_ready),
        .rd_rsp_data(scratch_rd_rsp_data),
        .rd_rsp_error(scratch_rd_rsp_error),
        .out_valid(ffn_pairer_out_valid),
        .out_ready(ffn_pairer_out_ready),
        .out_gate(ffn_pairer_out_gate),
        .out_up(ffn_pairer_out_up),
        .out_last(ffn_pairer_out_last)
    );

    section_f32_scratch u_section_scratch (
        .clk(clk),
        .rst_n(rst_n),
        .wr_cfg_valid(scratch_tee_start || scratch_only_start),
        .wr_cfg_ready(scratch_wr_cfg_ready),
        .wr_cfg_role(scratch_role_q),
        .wr_cfg_rows(scratch_rows_q),
        .wr_cfg_tokens(scratch_tokens_q),
        .wr_abort(scratch_writer_abort),
        .wr_busy(scratch_wr_busy),
        .wr_done(scratch_wr_done),
        .wr_error(scratch_wr_error),
        .s_axis_tdata(kernel_m_axis_tdata),
        .s_axis_tkeep(kernel_m_axis_tkeep),
        .s_axis_tvalid(scratch_sink_valid),
        .s_axis_tready(scratch_wr_tready),
        .s_axis_tlast(kernel_m_axis_tlast),
        .wr_commit_valid(scratch_wr_commit_valid),
        .wr_commit_bank(scratch_wr_commit_bank),
        .wr_commit_address(scratch_wr_commit_address),
        .r_wr_valid(1'b0),
        .r_wr_ready(),
        .r_wr_bank(2'd0),
        .r_wr_address(14'd0),
        .r_wr_data(64'd0),
        .r_wr_error(),
        .rd_req_valid(scratch_rd_req_valid),
        .rd_req_ready(scratch_rd_req_ready),
        .rd_req_role(scratch_rd_req_role),
        .rd_req_token(scratch_rd_req_token),
        .rd_req_group(scratch_rd_req_group),
        .rd_issue_valid(scratch_rd_issue_valid),
        .rd_issue_address(scratch_rd_issue_address),
        .rd_rsp_valid(scratch_rd_rsp_valid),
        .rd_rsp_ready(scratch_rd_rsp_ready),
        .rd_rsp_data(scratch_rd_rsp_data),
        .rd_rsp_error(scratch_rd_rsp_error)
    );

    // Canonical drain walks [token][group][bank0..3].  One 256-bit memory read
    // is copied into a no-reset group register, then a local 64-bit register
    // serializes its banks.  The two boundaries keep the four-deep URAM cascade
    // and bank selection on separate cycles and guarantee stable AXIS data.
    wire scratch_drain_shape_ok = (scratch_mode_q == SCRATCH_MODE_DRAIN) &&
                                  (scratch_rows_q != 14'd0) &&
                                  (scratch_rows_q[3:0] == 4'd0) &&
                                  (scratch_rows_q <= ((scratch_role_q == 2'd1 || scratch_role_q == 2'd2) ?
                                                     14'd12288 : 14'd4096)) &&
                                  (scratch_tokens_q != 3'd0) &&
                                  (scratch_tokens_q <= 3'd4);
    wire scratch_role_metadata_ok = scratch_valid_q[scratch_role_q] &&
                                    (scratch_valid_rows_q[scratch_role_q] == scratch_rows_q) &&
                                    (scratch_valid_tokens_q[scratch_role_q] == scratch_tokens_q);
    wire scratch_drain_start_ok = scratch_drain_start_strobe &&
                                  !scratch_abort_strobe &&
                                  scratch_drain_shape_ok && scratch_role_metadata_ok &&
                                  scratch_idle && !scratch_section_active_q;
    wire scratch_drain_start_bad = scratch_drain_start_strobe &&
                                   !scratch_abort_strobe &&
                                   !scratch_drain_start_ok;

    // Scratch returns are untagged. Latch the accepted request owner until its
    // response handshakes so an aborting pairer cannot leak a stale UP group into
    // a later diagnostic drain.
    wire scratch_drain_req_valid = scratch_drain_busy_q &&
                                   !scratch_drain_have_group_q;
    // Retained declarations keep the legacy controller structurally quiescent
    // in this v16 image; no request from it participates in arbitration.
    wire scratch_consumer_req_gate = scratch_consumer_busy_q &&
                                     (scratch_consumer_state_q == CONSUMER_REQ_GATE);
    wire scratch_consumer_req_up = scratch_consumer_busy_q &&
                                   (scratch_consumer_state_q == CONSUMER_REQ_UP);
    wire scratch_consumer_req_valid = scratch_consumer_req_gate ||
                                      scratch_consumer_req_up;
    wire scratch_consumer_wait_rsp = scratch_consumer_busy_q &&
                                     ((scratch_consumer_state_q == CONSUMER_WAIT_GATE) ||
                                      (scratch_consumer_state_q == CONSUMER_WAIT_UP));
    wire scratch_consumer_record_accounting_ok =
        (scratch_consumer_records_q < scratch_consumer_blocks_q) &&
        (scratch_consumer_records_q < scratch_consumer_total_blocks_q);
    wire scratch_select_pairer = (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                                  ffn_pairer_rd_req_valid;
    wire scratch_select_drain = (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                                !ffn_pairer_rd_req_valid &&
                                scratch_drain_req_valid;
    assign scratch_rd_req_valid = scratch_select_pairer || scratch_select_drain;
    assign scratch_rd_req_role = scratch_select_pairer ? ffn_pairer_rd_req_role :
                                 scratch_drain_role_q;
    assign scratch_rd_req_token = scratch_select_pairer ? ffn_pairer_rd_req_token :
                                  scratch_drain_token_q;
    assign scratch_rd_req_group = scratch_select_pairer ? ffn_pairer_rd_req_group :
                                  scratch_drain_group_q;
    assign ffn_pairer_rd_req_ready = scratch_select_pairer &&
                                     scratch_rd_req_ready;

    wire scratch_drain_rsp_valid = scratch_rd_rsp_valid &&
                                   (scratch_rd_owner_q == SCRATCH_RD_DRAIN);
    assign ffn_pairer_rd_rsp_valid = scratch_rd_rsp_valid &&
                                     (scratch_rd_owner_q == SCRATCH_RD_PAIRER);
    wire scratch_drain_rsp_ready = !scratch_drain_busy_q ? 1'b1 :
                                   (scratch_drain_have_group_q &&
                                    (scratch_rd_rsp_error ||
                                     (scratch_drain_emit_valid_q &&
                                      (scratch_drain_bank_q == 2'd3) &&
                                      m_axis_tready)));
    assign scratch_rd_rsp_ready =
        (scratch_rd_owner_q == SCRATCH_RD_PAIRER) ? ffn_pairer_rd_rsp_ready :
        (scratch_rd_owner_q == SCRATCH_RD_DRAIN) ? scratch_drain_rsp_ready :
        scratch_rd_rsp_valid;

    wire scratch_rd_req_accept = scratch_rd_req_valid && scratch_rd_req_ready;
    wire scratch_rd_rsp_accept = scratch_rd_rsp_valid && scratch_rd_rsp_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            scratch_rd_owner_q <= SCRATCH_RD_NONE;
        end else if (scratch_rd_req_accept) begin
            scratch_rd_owner_q <= scratch_select_pairer ?
                                  SCRATCH_RD_PAIRER : SCRATCH_RD_DRAIN;
        end else if (scratch_rd_rsp_accept) begin
            scratch_rd_owner_q <= SCRATCH_RD_NONE;
        end
    end

    assign swiglu_in_valid = ffn_pairer_out_valid;
    assign swiglu_in_gate = ffn_pairer_out_gate;
    assign swiglu_in_up = ffn_pairer_out_up;
    assign swiglu_in_last = ffn_pairer_out_last;
    assign ffn_pairer_out_ready = swiglu_in_ready;

    // Section abort is a lifecycle boundary for every stage that can retain
    // work. A traversal failure also kills any scalar, quantization, or native
    // record still owned by the section before software starts another run.
    assign q8_ingress_abort = section_abort_now;

    section_swiglu u_section_swiglu (
        .clk(clk),
        .rst_n(rst_n),
        .abort(ffn_pipeline_abort),
        .in_valid(swiglu_in_valid),
        .in_ready(swiglu_in_ready),
        .in_gate(swiglu_in_gate),
        .in_up(swiglu_in_up),
        .in_last(swiglu_in_last),
        .out_valid(swiglu_out_valid),
        .out_ready(swiglu_out_ready),
        .out_data(swiglu_out_data),
        .out_last(swiglu_out_last),
        .out_status(swiglu_out_status)
    );

    wire q8_buffer_cfg_valid = scratch_section_begin_ok;
    wire q8_buffer_seal_valid = ffn_seal_pending_q;
    wire q8_buffer_abort_valid = section_abort_now;
    wire q8_buffer_rd_req_valid = ffn_replay_active_q &&
                                  !ffn_replay_inflight_q;
    wire q8_buffer_rd_req_ready;

    section_q8_buffer u_section_q8_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_valid(q8_buffer_cfg_valid),
        .cfg_ready(q8_buffer_cfg_ready),
        .cfg_bank(1'b0),
        .cfg_tokens(scratch_tokens_q),
        .cfg_blocks(scratch_rows_q[13:5]),
        .seal_valid(q8_buffer_seal_valid),
        .seal_ready(q8_buffer_seal_ready),
        .seal_bank(1'b0),
        .seal_done(q8_buffer_seal_done),
        .seal_error(q8_buffer_seal_error),
        .abort_valid(q8_buffer_abort_valid),
        .abort_bank(1'b0),
        .bank_clearing(q8_buffer_bank_clearing),
        .bank_active(q8_buffer_bank_active),
        .bank_valid(q8_buffer_bank_valid),
        .bank_error(q8_buffer_bank_error),
        .bank0_record_count(q8_buffer_bank0_record_count),
        .bank1_record_count(),
        .s_axis_tdata(native_acts_tdata),
        .s_axis_tvalid(ffn_producer_busy_q && native_acts_tvalid),
        .s_axis_tready(q8_buffer_s_axis_tready),
        .s_axis_tlast(ffn_capture_beat_q == 3'd4),
        .s_axis_bank(1'b0),
        .s_axis_token(ffn_capture_token_q),
        .s_axis_block(ffn_capture_block_q),
        .cap_record_done(q8_buffer_cap_record_done),
        .cap_record_error(q8_buffer_cap_record_error),
        .cap_commit_valid(),
        .cap_commit_address(),
        .rd_req_valid(q8_buffer_rd_req_valid),
        .rd_req_ready(q8_buffer_rd_req_ready),
        .rd_req_bank(1'b0),
        .rd_req_token(ffn_replay_token_q),
        .rd_req_block(ffn_replay_block_q),
        .rd_issue_valid(),
        .rd_issue_address(),
        .m_axis_tdata(q8_buffer_m_axis_tdata),
        .m_axis_tvalid(q8_buffer_m_axis_tvalid),
        .m_axis_tready(q8_buffer_m_axis_tready),
        .m_axis_tlast(q8_buffer_m_axis_tlast),
        .m_axis_error(q8_buffer_m_axis_error),
        .m_axis_bank(q8_buffer_m_axis_bank),
        .m_axis_token(q8_buffer_m_axis_token),
        .m_axis_block(q8_buffer_m_axis_block)
    );

    wire scratch_drain_tvalid = scratch_drain_busy_q && !scratch_abort_strobe &&
                                scratch_drain_have_group_q &&
                                scratch_drain_emit_valid_q;
    wire scratch_drain_tlast = scratch_drain_tvalid &&
                               (scratch_drain_bank_q == 2'd3) &&
                               (scratch_drain_group_q + 1'b1 == scratch_drain_groups_q) &&
                               (scratch_drain_token_q + 1'b1 == scratch_drain_tokens_q);

    assign m_axis_tdata  = scratch_drain_busy_q ?
                           scratch_drain_emit_data_q : kernel_m_axis_tdata;
    assign m_axis_tvalid = scratch_drain_busy_q ?
                           scratch_drain_tvalid : ddr_kernel_valid;
    assign m_axis_tlast  = scratch_drain_busy_q ?
                           scratch_drain_tlast : kernel_m_axis_tlast;
    assign m_axis_tkeep  = scratch_drain_busy_q ?
                           8'hff : kernel_m_axis_tkeep;

    wire q8_capture_fire = ffn_producer_busy_q && native_acts_tvalid &&
                           q8_buffer_s_axis_tready;
    wire q8_replay_fire = q8_buffer_m_axis_tvalid &&
                          q8_buffer_m_axis_tready &&
                          q8_buffer_replay_healthy;
    wire q8_replay_fault = q8_buffer_replay_selected &&
                           q8_buffer_m_axis_tvalid &&
                           !q8_buffer_replay_healthy;
    wire up_run_fault = (ffn_phase_q == FFN_UP_RUN) &&
                        (scratch_wr_error || activation_error ||
                         (kernel_done &&
                          !(scratch_valid_q[2] ||
                            (scratch_wr_done && scratch_writer_active &&
                             (scratch_write_role_q == 2'd2) &&
                             !scratch_wr_error))));
    wire q8_numeric_fault = ffn_producer_busy_q &&
                            (q8_activation_abort
`ifdef VERILATOR
                             || sim_inject_q8_numeric_error
`endif
                            );
    wire producer_leaf_fault = (ffn_phase_q == FFN_GATE_RUN) &&
                               (gate_packer_error || ffn_pairer_error ||
                                q8_numeric_fault ||
                                q8_buffer_cap_record_error ||
                                q8_buffer_bank_error[0] || activation_error);
    wire section_fault_event = scratch_section_active_q &&
                               !ffn_abort_cleanup_q &&
                               (up_run_fault || producer_leaf_fault ||
                                ((ffn_phase_q == FFN_DOWN_RUN) &&
                                 activation_error) ||
                                q8_replay_fault ||
                                ((ffn_phase_q == FFN_SEAL) &&
                                 q8_buffer_seal_error));
    wire kernel_section_abort = scratch_abort_strobe || ffn_fault_q ||
                                section_fault_event;

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(MATMUL_COLS_MAX), .MAX_SUB_INDEX(512)) u_kernel (
        .clk(clk),
        .rst_n(rst_n),
        .start_kernel(kernel_start),
        .num_q1_blocks(num_q1_blocks_q),
        .num_rowblocks(num_rowblocks_q),
        .num_rows(num_rows_q),
        .num_cols(num_cols_q),
        .weight_fmt(weight_fmt_q),
        .act_mode(ffn_down_start_q ? 2'd0 : act_mode_q),
        .act_epoch(act_epoch_q),
        .activation_abort(kernel_section_abort || q8_activation_abort ||
                          q8_ingress_abort ||
                          (scratch_writer_active && scratch_abort_strobe) ||
                          (scratch_consumer_busy_q && scratch_abort_strobe)),
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
        .m_axis_tdata(kernel_m_axis_tdata),
        .m_axis_tvalid(kernel_m_axis_tvalid),
        .m_axis_tready(kernel_m_axis_tready),
        .m_axis_tlast(kernel_m_axis_tlast),
        .m_axis_tkeep(kernel_m_axis_tkeep),
        .dbg_state()
    );

    // Scratch lifecycle and ownership.  Configuration registers remain writable,
    // but every accepted operation runs exclusively from these snapshots.
    always @(posedge clk) begin
        if (!rst_n) begin
            scratch_tee_run_q          <= 1'b0;
            scratch_only_run_q         <= 1'b0;
            scratch_write_role_q       <= 2'd0;
            scratch_write_rows_q       <= 14'd0;
            scratch_write_tokens_q     <= 3'd0;
            scratch_writer_done_q      <= 1'b0;
            scratch_error_q            <= 7'd0;
            scratch_valid_q            <= 4'd0;
            scratch_section_active_q   <= 1'b0;
            scratch_section_done_q     <= 1'b0;
            scratch_consumer_busy_q    <= 1'b0;
            scratch_consumer_done_q    <= 1'b0;
            scratch_consumer_state_q   <= CONSUMER_IDLE;
            scratch_consumer_token_q   <= 3'd0;
            scratch_consumer_group_q   <= 11'd0;
            scratch_consumer_groups_q  <= 11'd0;
            scratch_consumer_lane_q    <= 3'd0;
            scratch_consumer_blocks_q  <= 16'd0;
            scratch_consumer_records_q <= 16'd0;
            scratch_consumer_total_blocks_q <= 16'd0;
            ffn_rows_q                 <= 14'd0;
            ffn_tokens_q               <= 3'd0;
            ffn_blocks_q               <= 9'd0;
            ffn_phase_q                <= FFN_IDLE;
            ffn_fault_q                <= 1'b0;
            ffn_abort_cleanup_q        <= 1'b0;
            ffn_gate_run_q             <= 1'b0;
            ffn_producer_busy_q        <= 1'b0;
            ffn_producer_done_q        <= 1'b0;
            ffn_capture_token_q        <= 2'd0;
            ffn_capture_block_q        <= 9'd0;
            ffn_capture_beat_q         <= 3'd0;
            ffn_capture_complete_q     <= 1'b0;
            ffn_gate_kernel_done_q     <= 1'b0;
            ffn_gate_packer_done_q     <= 1'b0;
            ffn_gate_pairer_done_q     <= 1'b0;
            ffn_down_kernel_done_q     <= 1'b0;
            ffn_seal_pending_q         <= 1'b0;
            ffn_replay_active_q        <= 1'b0;
            ffn_replay_inflight_q      <= 1'b0;
            ffn_replay_token_q         <= 2'd0;
            ffn_replay_block_q         <= 9'd0;
            ffn_replay_beat_q          <= 3'd0;
            ffn_replay_complete_q      <= 1'b0;
            scratch_drain_busy_q       <= 1'b0;
            scratch_drain_done_q       <= 1'b0;
            scratch_drain_role_q       <= 2'd0;
            scratch_drain_tokens_q     <= 3'd0;
            scratch_drain_token_q      <= 3'd0;
            scratch_drain_group_q      <= 11'd0;
            scratch_drain_bank_q       <= 2'd0;
            scratch_drain_have_group_q <= 1'b0;
            scratch_drain_group_valid_q <= 1'b0;
            scratch_drain_emit_valid_q <= 1'b0;
            scratch_drain_groups_q     <= 11'd0;
            for (scratch_role_i = 0; scratch_role_i < 4; scratch_role_i = scratch_role_i + 1) begin
                scratch_valid_rows_q[scratch_role_i]   <= 14'd0;
                scratch_valid_tokens_q[scratch_role_i] <= 3'd0;
            end
        end else begin
            if (scratch_section_begin_ok) begin
                scratch_section_active_q <= 1'b1;
                scratch_section_done_q   <= 1'b0;
                scratch_consumer_done_q  <= 1'b0;
                scratch_writer_done_q    <= 1'b0;
                scratch_drain_done_q     <= 1'b0;
                scratch_error_q          <= 7'd0;
                scratch_valid_q          <= 4'd0;
                ffn_rows_q               <= scratch_rows_q;
                ffn_tokens_q             <= scratch_tokens_q;
                ffn_blocks_q             <= scratch_rows_q[13:5];
                ffn_phase_q              <= FFN_WAIT_UP;
                ffn_fault_q              <= 1'b0;
                ffn_abort_cleanup_q      <= 1'b0;
                ffn_gate_run_q           <= 1'b0;
                ffn_producer_busy_q      <= 1'b0;
                ffn_producer_done_q      <= 1'b0;
                ffn_capture_complete_q   <= 1'b0;
                ffn_seal_pending_q       <= 1'b0;
                ffn_replay_active_q      <= 1'b0;
                ffn_replay_inflight_q    <= 1'b0;
                ffn_replay_complete_q    <= 1'b0;
                ffn_down_kernel_done_q    <= 1'b0;
            end else if (scratch_section_begin_bad) begin
                scratch_section_done_q <= 1'b1;
                scratch_error_q[0] <= 1'b1;
            end

            if (scratch_tee_start || scratch_only_start) begin
                scratch_tee_run_q      <= scratch_tee_start;
                scratch_only_run_q     <= scratch_only_start;
                scratch_write_role_q   <= scratch_role_q;
                scratch_write_rows_q   <= scratch_rows_q;
                scratch_write_tokens_q <= scratch_tokens_q;
                scratch_writer_done_q  <= 1'b0;
                if (scratch_tee_start)
                    scratch_error_q    <= 7'd0;
                scratch_valid_q[scratch_role_q] <= 1'b0;
                if (scratch_only_start && (ffn_phase_q == FFN_WAIT_UP))
                    ffn_phase_q <= FFN_UP_RUN;
            end else if (scratch_start_rejected) begin
                // Includes invalid/busy tee, DRAIN/unknown-mode CTRL.START, and
                // a DDR start attempted while either scratch direction is active.
                scratch_error_q[0] <= 1'b1;
                if ((scratch_mode_q == SCRATCH_MODE_TEE) ||
                    (scratch_mode_q == SCRATCH_MODE_ONLY)) begin
                    scratch_writer_done_q <= 1'b0;
                    scratch_valid_q[scratch_role_q] <= 1'b0;
                end
            end

            if (scratch_writer_active && scratch_wr_error) begin
                scratch_error_q[1] <= 1'b1;
                scratch_valid_q[scratch_write_role_q] <= 1'b0;
            end
            if (scratch_wr_done) begin
                scratch_writer_done_q <= 1'b1;
                if (scratch_writer_active && !scratch_wr_error && !scratch_error_q[2]) begin
                    scratch_valid_q[scratch_write_role_q] <= 1'b1;
                    scratch_valid_rows_q[scratch_write_role_q] <= scratch_write_rows_q;
                    scratch_valid_tokens_q[scratch_write_role_q] <= scratch_write_tokens_q;
                end else begin
                    scratch_valid_q[scratch_write_role_q] <= 1'b0;
                    scratch_error_q[1] <= 1'b1;
                end
            end
            if (kernel_done) begin
                scratch_tee_run_q <= 1'b0;
                scratch_only_run_q <= 1'b0;
            end

            // Streaming section phase and exact capture/replay accounting.
            if (ffn_gate_start_q) begin
                ffn_phase_q              <= FFN_GATE_RUN;
                ffn_gate_run_q           <= 1'b1;
                ffn_producer_busy_q      <= 1'b1;
                ffn_producer_done_q      <= 1'b0;
                ffn_capture_token_q      <= 2'd0;
                ffn_capture_block_q      <= 9'd0;
                ffn_capture_beat_q       <= 3'd0;
                ffn_capture_complete_q   <= 1'b0;
                ffn_gate_kernel_done_q   <= 1'b0;
                ffn_gate_packer_done_q   <= 1'b0;
                ffn_gate_pairer_done_q   <= 1'b0;
                ffn_seal_pending_q       <= 1'b0;
            end

            if ((ffn_phase_q == FFN_UP_RUN) && kernel_done) begin
                if (activation_error ||
                    !(scratch_valid_q[2] ||
                      (scratch_wr_done && scratch_writer_active &&
                       (scratch_write_role_q == 2'd2) &&
                       !scratch_wr_error))) begin
                    ffn_fault_q <= 1'b1;
                    scratch_error_q[5] <= 1'b1;
                end else begin
                    ffn_phase_q <= FFN_WAIT_GATE;
                end
            end

            if (ffn_phase_q == FFN_GATE_RUN) begin
                if (kernel_done) begin
                    ffn_gate_kernel_done_q <= 1'b1;
                    ffn_gate_run_q <= 1'b0;
                end
                if (gate_packer_done)
                    ffn_gate_packer_done_q <= 1'b1;
                if (ffn_pairer_done)
                    ffn_gate_pairer_done_q <= 1'b1;

                if (q8_capture_fire) begin
                    if ((ffn_capture_beat_q == 3'd4) !=
                        q8_internal_record_done) begin
                        ffn_fault_q <= 1'b1;
                        scratch_error_q[5] <= 1'b1;
                    end
                    if (ffn_capture_beat_q == 3'd4) begin
                        ffn_capture_beat_q <= 3'd0;
                        if ((ffn_capture_block_q + 1'b1 == ffn_blocks_q) &&
                            ({1'b0, ffn_capture_token_q} + 1'b1 ==
                             ffn_tokens_q)) begin
                            ffn_capture_complete_q <= 1'b1;
                        end else if ({1'b0, ffn_capture_token_q} + 1'b1 ==
                                     ffn_tokens_q) begin
                            ffn_capture_token_q <= 2'd0;
                            ffn_capture_block_q <=
                                ffn_capture_block_q + 1'b1;
                        end else begin
                            ffn_capture_token_q <=
                                ffn_capture_token_q + 1'b1;
                        end
                    end else begin
                        ffn_capture_beat_q <= ffn_capture_beat_q + 1'b1;
                    end
                end

                if ((ffn_gate_kernel_done_q || kernel_done) &&
                    (ffn_gate_packer_done_q || gate_packer_done) &&
                    (ffn_gate_pairer_done_q || ffn_pairer_done) &&
                    ffn_capture_complete_q && !section_fault_event &&
                    !ffn_fault_q) begin
                    ffn_phase_q <= FFN_SEAL;
                    ffn_seal_pending_q <= 1'b1;
                    ffn_gate_run_q <= 1'b0;
                end
            end

            if ((ffn_phase_q == FFN_SEAL) &&
                ffn_seal_pending_q && q8_buffer_seal_ready)
                ffn_seal_pending_q <= 1'b0;

            if ((ffn_phase_q == FFN_SEAL) && q8_buffer_seal_done) begin
                if (q8_buffer_seal_error || !q8_buffer_bank_valid[0]) begin
                    ffn_fault_q <= 1'b1;
                    scratch_error_q[5] <= 1'b1;
                end else begin
                    ffn_phase_q <= FFN_WAIT_DOWN;
                    ffn_producer_busy_q <= 1'b0;
                    ffn_producer_done_q <= 1'b1;
                end
            end

            if (ffn_down_start_q) begin
                ffn_phase_q             <= FFN_DOWN_RUN;
                ffn_replay_active_q     <= 1'b1;
                ffn_replay_inflight_q   <= 1'b0;
                ffn_replay_token_q      <= 2'd0;
                ffn_replay_block_q      <= 9'd0;
                ffn_replay_beat_q       <= 3'd0;
                ffn_replay_complete_q   <= 1'b0;
                ffn_down_kernel_done_q  <= 1'b0;
            end

            if ((ffn_phase_q == FFN_DOWN_RUN) && kernel_done)
                ffn_down_kernel_done_q <= 1'b1;

            if ((ffn_phase_q == FFN_DOWN_RUN) && q8_buffer_rd_req_valid &&
                q8_buffer_rd_req_ready)
                ffn_replay_inflight_q <= 1'b1;

            if ((ffn_phase_q == FFN_DOWN_RUN) && q8_replay_fire) begin
                if (ffn_replay_beat_q == 3'd4) begin
                    ffn_replay_beat_q <= 3'd0;
                    ffn_replay_inflight_q <= 1'b0;
                    if ((ffn_replay_block_q + 1'b1 == ffn_blocks_q) &&
                        ({1'b0, ffn_replay_token_q} + 1'b1 ==
                         ffn_tokens_q)) begin
                        ffn_replay_active_q <= 1'b0;
                        ffn_replay_complete_q <= 1'b1;
                    end else if (ffn_replay_block_q + 1'b1 ==
                                 ffn_blocks_q) begin
                        ffn_replay_block_q <= 9'd0;
                        ffn_replay_token_q <= ffn_replay_token_q + 1'b1;
                    end else begin
                        ffn_replay_block_q <= ffn_replay_block_q + 1'b1;
                    end
                end else begin
                    ffn_replay_beat_q <= ffn_replay_beat_q + 1'b1;
                end
            end

            if ((ffn_phase_q == FFN_DOWN_RUN) &&
                (ffn_down_kernel_done_q || kernel_done) &&
                ffn_replay_complete_q && !activation_error &&
                !ffn_fault_q) begin
                ffn_phase_q <= FFN_IDLE;
                scratch_section_active_q <= 1'b0;
                scratch_section_done_q <= 1'b1;
                scratch_valid_q <= 4'd0;
            end

            if (section_fault_event) begin
                ffn_fault_q <= 1'b1;
                scratch_error_q[5] <= 1'b1;
                if (q8_numeric_fault)
                    scratch_error_q[6] <= 1'b1;
                ffn_seal_pending_q <= 1'b0;
            end

            if (scratch_drain_start_ok) begin
                scratch_drain_busy_q       <= 1'b1;
                scratch_drain_done_q       <= 1'b0;
                scratch_drain_role_q       <= scratch_role_q;
                scratch_drain_tokens_q     <= scratch_tokens_q;
                scratch_drain_token_q      <= 3'd0;
                scratch_drain_group_q      <= 11'd0;
                scratch_drain_bank_q       <= 2'd0;
                scratch_drain_have_group_q <= 1'b0;
                scratch_drain_group_valid_q <= 1'b0;
                scratch_drain_emit_valid_q <= 1'b0;
                scratch_drain_groups_q     <= scratch_rows_q[13:3];
                scratch_error_q            <= 7'd0;
            end else if (scratch_drain_start_bad) begin
                scratch_drain_done_q <= 1'b1;
                if (!scratch_drain_shape_ok || !scratch_idle)
                    scratch_error_q[0] <= 1'b1;
                if (scratch_drain_shape_ok && !scratch_role_metadata_ok)
                    scratch_error_q[4] <= 1'b1;
            end

            if (scratch_consumer_start) begin
                scratch_consumer_busy_q   <= 1'b1;
                scratch_consumer_done_q   <= 1'b0;
                scratch_consumer_state_q  <= CONSUMER_REQ_GATE;
                scratch_consumer_token_q  <= 3'd0;
                scratch_consumer_group_q  <= 11'd0;
                scratch_consumer_groups_q <= scratch_rows_q[13:3];
                scratch_consumer_lane_q   <= 3'd0;
                scratch_consumer_blocks_q <= 16'd0;
                scratch_consumer_records_q <= 16'd0;
                // Keep both operands explicitly widened.  The maximum section
                // consumes (12288 / 32) * 4 = 1536 native Q8 records; an
                // operand-sized multiply would silently truncate that result.
                scratch_consumer_total_blocks_q <=
                    {7'd0, scratch_rows_q[13:5]} * {13'd0, scratch_tokens_q};
            end else if (scratch_consumer_busy_q && !scratch_abort_strobe &&
                         !q8_activation_abort) begin
                // Input blocks may run ahead into the SwiGLU BRAM FIFO. Retire
                // them only after all five canonical Q8 record beats are accepted.
                if (q8_internal_record_done) begin
                    if (scratch_consumer_record_accounting_ok) begin
                        scratch_consumer_records_q <=
                            scratch_consumer_records_q + 1'b1;
                    end else begin
                        scratch_consumer_busy_q  <= 1'b0;
                        scratch_consumer_done_q  <= 1'b1;
                        scratch_consumer_state_q <= CONSUMER_IDLE;
                        scratch_error_q[5] <= 1'b1;
                    end
                end

                if (!q8_internal_record_done ||
                    scratch_consumer_record_accounting_ok) begin
                    case (scratch_consumer_state_q)
                    CONSUMER_REQ_GATE: if (scratch_rd_req_ready)
                        scratch_consumer_state_q <= CONSUMER_WAIT_GATE;

                    CONSUMER_WAIT_GATE: if (scratch_rd_rsp_valid) begin
                        if (scratch_rd_rsp_error) begin
                            scratch_consumer_busy_q  <= 1'b0;
                            scratch_consumer_done_q  <= 1'b1;
                            scratch_consumer_state_q <= CONSUMER_IDLE;
                            scratch_error_q[5]       <= 1'b1;
                        end else begin
                            scratch_consumer_gate_q  <= scratch_rd_rsp_data;
                            scratch_consumer_state_q <= CONSUMER_REQ_UP;
                        end
                    end

                    CONSUMER_REQ_UP: if (scratch_rd_req_ready)
                        scratch_consumer_state_q <= CONSUMER_WAIT_UP;

                    CONSUMER_WAIT_UP: if (scratch_rd_rsp_valid) begin
                        if (scratch_rd_rsp_error) begin
                            scratch_consumer_busy_q  <= 1'b0;
                            scratch_consumer_done_q  <= 1'b1;
                            scratch_consumer_state_q <= CONSUMER_IDLE;
                            scratch_error_q[5]       <= 1'b1;
                        end else begin
                            scratch_consumer_up_q    <= scratch_rd_rsp_data;
                            scratch_consumer_lane_q  <= 3'd0;
                            scratch_consumer_state_q <= CONSUMER_ISSUE;
                        end
                    end

                    CONSUMER_ISSUE: if (swiglu_in_ready) begin
                        if (scratch_consumer_lane_q == 3'd7) begin
                            scratch_consumer_lane_q <= 3'd0;
                            if (scratch_consumer_group_q[1:0] == 2'd3)
                                scratch_consumer_blocks_q <= scratch_consumer_blocks_q + 1'b1;

                            if ((scratch_consumer_group_q + 1'b1 ==
                                 scratch_consumer_groups_q) &&
                                (scratch_consumer_token_q + 1'b1 ==
                                 scratch_tokens_q)) begin
                                scratch_consumer_state_q <= CONSUMER_DRAIN;
                            end else if (scratch_consumer_group_q + 1'b1 ==
                                         scratch_consumer_groups_q) begin
                                scratch_consumer_group_q <= 11'd0;
                                scratch_consumer_token_q <=
                                    scratch_consumer_token_q + 1'b1;
                                scratch_consumer_state_q <= CONSUMER_REQ_GATE;
                            end else begin
                                scratch_consumer_group_q <= scratch_consumer_group_q + 1'b1;
                                scratch_consumer_state_q <= CONSUMER_REQ_GATE;
                            end
                        end else begin
                            scratch_consumer_lane_q <= scratch_consumer_lane_q + 1'b1;
                        end
                    end

                    CONSUMER_DRAIN: if (q8_internal_record_done) begin
                        if (scratch_consumer_records_q + 1'b1 ==
                            scratch_consumer_total_blocks_q) begin
                            scratch_consumer_busy_q  <= 1'b0;
                            scratch_consumer_done_q  <= 1'b1;
                            scratch_consumer_state_q <= CONSUMER_IDLE;
                            if (scratch_consumer_blocks_q != scratch_consumer_total_blocks_q)
                                scratch_error_q[5] <= 1'b1;
                        end
                    end

                    default: begin
                        scratch_consumer_busy_q  <= 1'b0;
                        scratch_consumer_done_q  <= 1'b1;
                        scratch_consumer_state_q <= CONSUMER_IDLE;
                        scratch_error_q[5]       <= 1'b1;
                    end
                    endcase
                end
            end

            if (internal_activation_mode && q8_activation_abort) begin
                scratch_consumer_busy_q  <= 1'b0;
                scratch_consumer_done_q  <= 1'b1;
                scratch_consumer_state_q <= CONSUMER_IDLE;
                scratch_error_q[6]       <= 1'b1;
                scratch_valid_q          <= 4'd0;
            end

            if (scratch_drain_busy_q && !scratch_abort_strobe) begin
                if (!scratch_drain_have_group_q && scratch_rd_req_valid && scratch_rd_req_ready) begin
                    scratch_drain_have_group_q <= 1'b1;
                    scratch_drain_bank_q <= 2'd0;
                    scratch_drain_group_valid_q <= 1'b0;
                    scratch_drain_emit_valid_q <= 1'b0;
                end

                if (scratch_drain_have_group_q && scratch_drain_rsp_valid) begin
                    if (scratch_rd_rsp_error) begin
                        scratch_drain_busy_q       <= 1'b0;
                        scratch_drain_done_q       <= 1'b1;
                        scratch_drain_have_group_q <= 1'b0;
                        scratch_drain_group_valid_q <= 1'b0;
                        scratch_drain_emit_valid_q <= 1'b0;
                        scratch_error_q[3]         <= 1'b1;
                    end else if (!scratch_drain_group_valid_q) begin
                        scratch_drain_group_data_q  <= scratch_rd_rsp_data;
                        scratch_drain_group_valid_q <= 1'b1;
                    end else if (!scratch_drain_emit_valid_q) begin
                        scratch_drain_emit_data_q  <= scratch_drain_group_data_q[63:0];
                        scratch_drain_emit_valid_q <= 1'b1;
                    end else if (m_axis_tready) begin
                        if (scratch_drain_bank_q == 2'd3) begin
                            scratch_drain_bank_q       <= 2'd0;
                            scratch_drain_have_group_q <= 1'b0;
                            scratch_drain_group_valid_q <= 1'b0;
                            scratch_drain_emit_valid_q <= 1'b0;
                            if ((scratch_drain_group_q + 1'b1 == scratch_drain_groups_q) &&
                                (scratch_drain_token_q + 1'b1 == scratch_drain_tokens_q)) begin
                                scratch_drain_busy_q <= 1'b0;
                                scratch_drain_done_q <= 1'b1;
                            end else if (scratch_drain_group_q + 1'b1 == scratch_drain_groups_q) begin
                                scratch_drain_group_q <= 11'd0;
                                scratch_drain_token_q <= scratch_drain_token_q + 1'b1;
                            end else begin
                                scratch_drain_group_q <= scratch_drain_group_q + 1'b1;
                            end
                        end else begin
                            scratch_drain_bank_q <= scratch_drain_bank_q + 1'b1;
                            case (scratch_drain_bank_q)
                                2'd0: scratch_drain_emit_data_q <= scratch_drain_group_data_q[127:64];
                                2'd1: scratch_drain_emit_data_q <= scratch_drain_group_data_q[191:128];
                                default: scratch_drain_emit_data_q <= scratch_drain_group_data_q[255:192];
                            endcase
                        end
                    end
                end
            end

            if (scratch_abort_strobe) begin
                ffn_abort_cleanup_q <= scratch_section_active_q;
                ffn_fault_q <= 1'b0;
                ffn_gate_run_q <= 1'b0;
                ffn_producer_done_q <= 1'b0;
                ffn_seal_pending_q <= 1'b0;
                ffn_replay_active_q <= 1'b0;
                ffn_replay_inflight_q <= 1'b0;
                scratch_tee_run_q <= 1'b0;
                scratch_only_run_q <= 1'b0;
                if (scratch_writer_active || scratch_wr_busy) begin
                    scratch_valid_q[scratch_write_role_q] <= 1'b0;
                    scratch_error_q[2] <= 1'b1;
                end
                if (scratch_drain_busy_q) begin
                    scratch_drain_busy_q       <= 1'b0;
                    scratch_drain_done_q       <= 1'b1;
                    scratch_drain_have_group_q <= 1'b0;
                    scratch_drain_group_valid_q <= 1'b0;
                    scratch_drain_emit_valid_q <= 1'b0;
                    scratch_error_q[2]         <= 1'b1;
                end
                if (scratch_consumer_busy_q || scratch_section_active_q) begin
                    scratch_consumer_busy_q  <= 1'b0;
                    scratch_consumer_done_q  <= 1'b1;
                    scratch_consumer_state_q <= CONSUMER_IDLE;
                    scratch_section_done_q   <= 1'b1;
                    scratch_valid_q          <= 4'd0;
                    scratch_error_q[2]       <= 1'b1;
                end
            end

            if (ffn_abort_cleanup_q && !kernel_busy && !gate_packer_busy &&
                !ffn_pairer_busy &&
                (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                !scratch_rd_rsp_valid) begin
                ffn_abort_cleanup_q <= 1'b0;
                ffn_phase_q <= FFN_IDLE;
                ffn_producer_busy_q <= 1'b0;
                ffn_producer_done_q <= 1'b0;
                scratch_section_active_q <= 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n)            done_latched <= 1'b0;
        else if (start_strobe && !kernel_busy)
            done_latched <= scratch_start_rejected;
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
    wire perf_a_valid = q8_buffer_replay_selected ?
                        kernel_acts_tvalid : s_axis_acts_tvalid;
    wire perf_a_ready = q8_buffer_replay_selected ?
                        kernel_acts_tready : s_axis_acts_tready;
    always @(posedge clk) begin
        if (!rst_n || start_strobe) begin
            w_stall_q <= 32'd0; a_stall_q <= 32'd0; r_stall_q <= 32'd0;
            w_beats_q <= 32'd0; a_beats_q <= 32'd0; r_beats_q <= 32'd0;
        end else if (kernel_busy) begin
            if (weight_tvalid && weight_tready)           w_beats_q <= w_beats_q + 32'd1;
            if (perf_a_valid && perf_a_ready) a_beats_q <= a_beats_q + 32'd1;
            if (m_axis_tvalid && m_axis_tready)           r_beats_q <= r_beats_q + 32'd1;
            if (weight_tready && !weight_tvalid)           w_stall_q <= w_stall_q + 32'd1;
            if (perf_a_ready && !perf_a_valid) a_stall_q <= a_stall_q + 32'd1;
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
            scratch_mode_q  <= SCRATCH_MODE_DDR;
            scratch_role_q  <= 2'd0;
            scratch_rows_q  <= 14'd0;
            scratch_tokens_q <= 3'd0;
            scratch_drain_start_strobe <= 1'b0;
            scratch_abort_strobe <= 1'b0;
            scratch_section_begin_strobe <= 1'b0;
        end else begin
            start_strobe <= 1'b0;
            scratch_drain_start_strobe <= 1'b0;
            scratch_abort_strobe <= 1'b0;
            scratch_section_begin_strobe <= 1'b0;

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
                    MATMUL_OFF_SCRATCH_MODE[7:0]: begin
                        if (s_axi_wstrb[0]) scratch_mode_q <= s_axi_wdata[1:0];
                    end
                    MATMUL_OFF_SCRATCH_ROLE[7:0]: begin
                        if (s_axi_wstrb[0]) scratch_role_q <= s_axi_wdata[1:0];
                    end
                    MATMUL_OFF_SCRATCH_ROWS[7:0]: begin
                        if (s_axi_wstrb[0]) scratch_rows_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) scratch_rows_q[13:8] <= s_axi_wdata[13:8];
                    end
                    MATMUL_OFF_SCRATCH_TOKENS[7:0]: begin
                        if (s_axi_wstrb[0]) scratch_tokens_q <= s_axi_wdata[2:0];
                    end
                    MATMUL_OFF_SCRATCH_CTRL[7:0]: begin
                        if (s_axi_wstrb[0]) begin
                            scratch_drain_start_strobe <= s_axi_wdata[0];
                            scratch_abort_strobe <= s_axi_wdata[1];
                            scratch_section_begin_strobe <= s_axi_wdata[2];
                        end
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
                    MATMUL_OFF_QUANT_STATUS[7:0]:  rdata_q <= {26'd0, quantizer_status};
                    MATMUL_OFF_SCRATCH_MODE[7:0]:  rdata_q <= {30'd0, scratch_mode_q};
                    MATMUL_OFF_SCRATCH_ROLE[7:0]:  rdata_q <= {30'd0, scratch_role_q};
                    MATMUL_OFF_SCRATCH_ROWS[7:0]:  rdata_q <= {18'd0, scratch_rows_q};
                    MATMUL_OFF_SCRATCH_TOKENS[7:0]: rdata_q <= {29'd0, scratch_tokens_q};
                    MATMUL_OFF_SCRATCH_STATUS[7:0]: rdata_q <= {
                        18'd0, ffn_gate_ready,
                        scratch_section_done_q, scratch_section_active_q,
                        ffn_producer_done_q, ffn_producer_busy_q,
                        scratch_valid_q, |scratch_error_q,
                        scratch_drain_done_q,
                        (scratch_drain_busy_q ||
                         (scratch_rd_owner_q == SCRATCH_RD_DRAIN)),
                        scratch_writer_done_q, scratch_wr_busy
                    };
                    MATMUL_OFF_SCRATCH_ERROR[7:0]: rdata_q <= {25'd0, scratch_error_q};
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

`ifdef FORMAL
    assign formal_ffn_phase = ffn_phase_q;
    assign formal_ffn_gate_ready = ffn_gate_ready;
    assign formal_scratch_rd_owner = scratch_rd_owner_q;
    assign formal_scratch_rd_rsp_valid = scratch_rd_rsp_valid;
    assign formal_scratch_rd_rsp_ready = scratch_rd_rsp_ready;
    assign formal_section_active = scratch_section_active_q;
    assign formal_section_done = scratch_section_done_q;
    assign formal_abort_cleanup = ffn_abort_cleanup_q;
    assign formal_scratch_error = scratch_error_q;
    assign formal_capture_fire = q8_capture_fire;
    assign formal_capture_record_done = q8_buffer_cap_record_done;
    assign formal_bank0_record_count = q8_buffer_bank0_record_count;
    assign formal_replay_fire = q8_replay_fire;
    assign formal_replay_complete = ffn_replay_complete_q;
    assign formal_down_kernel_done = ffn_down_kernel_done_q;
    assign formal_kernel_done = kernel_done;
    assign formal_abort_strobe = scratch_abort_strobe;
    assign formal_ffn_fault = ffn_fault_q;
    assign formal_section_begin_ok = scratch_section_begin_ok;
    assign formal_up_start = scratch_only_start &&
                             (ffn_phase_q == FFN_WAIT_UP);
    assign formal_gate_start = ffn_gate_start_q;
    assign formal_gate_drain_ready =
        (ffn_gate_kernel_done_q || kernel_done) &&
        (ffn_gate_packer_done_q || gate_packer_done) &&
        (ffn_gate_pairer_done_q || ffn_pairer_done) &&
        ffn_capture_complete_q && !section_fault_event && !ffn_fault_q;
    assign formal_seal_done = q8_buffer_seal_done &&
                              !q8_buffer_seal_error;
    assign formal_down_start = ffn_down_start_q;
    assign formal_down_complete_ready =
        (ffn_down_kernel_done_q || kernel_done) &&
        ffn_replay_complete_q && !activation_error && !ffn_fault_q;
    assign formal_producer_done = ffn_producer_done_q;
    assign formal_capture_token = ffn_capture_token_q;
    assign formal_capture_block = ffn_capture_block_q;
    assign formal_capture_beat = ffn_capture_beat_q;
    assign formal_replay_token = ffn_replay_token_q;
    assign formal_replay_block = ffn_replay_block_q;
    assign formal_replay_beat = ffn_replay_beat_q;

    reg        f_decode_past_valid = 1'b0;

    always @(posedge clk) begin
        f_decode_past_valid <= 1'b1;

        if (rst_n) begin
            assert(ffn_phase_q <= FFN_DOWN_RUN);
            assert(ffn_capture_beat_q <= 3'd4);
            assert(ffn_replay_beat_q <= 3'd4);
            assert(ffn_capture_block_q <
                   (ffn_blocks_q == 0 ? 9'd1 : ffn_blocks_q));
            assert(ffn_replay_block_q <
                   (ffn_blocks_q == 0 ? 9'd1 : ffn_blocks_q));
            assert({1'b0, ffn_capture_token_q} <
                   (ffn_tokens_q == 0 ? 3'd1 : ffn_tokens_q));
            assert({1'b0, ffn_replay_token_q} <
                   (ffn_tokens_q == 0 ? 3'd1 : ffn_tokens_q));

            if (scratch_rd_owner_q == SCRATCH_RD_PAIRER)
                assert(!scratch_drain_rsp_valid);
            if (scratch_rd_owner_q == SCRATCH_RD_DRAIN)
                assert(!ffn_pairer_rd_rsp_valid);
            if (ffn_pairer_rd_rsp_valid)
                assert(scratch_rd_owner_q == SCRATCH_RD_PAIRER);
            if (scratch_drain_rsp_valid)
                assert(scratch_rd_owner_q == SCRATCH_RD_DRAIN);

            if (scratch_only_run_q) begin
                assert(scratch_section_active_q);
                assert(!m_axis_tvalid);
            end
            if (internal_activation_mode)
                assert(!s_axis_acts_tready);
            if (scratch_section_active_q &&
                (scratch_mode_q == SCRATCH_MODE_ONLY))
                assert(scratch_launch_ok ==
                       (ffn_up_preflight_ok || ffn_gate_preflight_ok));
            if (!scratch_section_active_q && internal_activation_mode &&
                (scratch_mode_q == SCRATCH_MODE_DDR))
                assert(!scratch_ddr_preflight_ok);

            if (q8_ingress_abort) begin
                assert(!raw_acts_tready);
                assert(!native_acts_tvalid);
                assert(!q8_internal_record_done);
                assert(!q8_activation_abort);
                assert(quantizer_status == 6'd0);
            end

            if (ffn_producer_busy_q) begin
                assert(scratch_section_active_q);
                assert((ffn_phase_q == FFN_GATE_RUN) ||
                       (ffn_phase_q == FFN_SEAL) || ffn_abort_cleanup_q);
            end

            if (f_decode_past_valid &&
                $past(rst_n && scratch_section_begin_ok)) begin
                assert(scratch_section_active_q);
                assert(!scratch_section_done_q);
                assert(scratch_valid_q == 4'd0);
                assert(scratch_error_q == 7'd0);
                assert(ffn_phase_q == FFN_WAIT_UP);
            end

            if (f_decode_past_valid &&
                $past(rst_n && scratch_abort_strobe &&
                      scratch_section_active_q)) begin
                assert(ffn_abort_cleanup_q || !scratch_section_active_q);
                assert(scratch_consumer_done_q && scratch_section_done_q);
                assert(scratch_valid_q == 4'd0);
                assert(scratch_error_q[2]);
            end

            assert(!(scratch_abort_strobe &&
                     (scratch_section_begin_ok || scratch_section_begin_bad ||
                      scratch_drain_start_ok || scratch_drain_start_bad)));
            if (f_decode_past_valid &&
                $past(rst_n && scratch_abort_strobe &&
                      scratch_section_begin_strobe &&
                      !scratch_section_active_q)) begin
                assert(!scratch_section_active_q);
                assert(ffn_phase_q == FFN_IDLE);
                assert(!ffn_abort_cleanup_q);
            end
            if (f_decode_past_valid &&
                $past(rst_n && scratch_abort_strobe &&
                      scratch_drain_start_strobe &&
                      !scratch_drain_busy_q))
                assert(!scratch_drain_busy_q);
            if (f_decode_past_valid &&
                $past(rst_n && section_fault_event && q8_numeric_fault))
                assert(scratch_error_q[6]);

            if (q8_replay_fault) begin
                assert(!kernel_acts_tvalid);
                assert(!q8_buffer_m_axis_tready);
            end
            if (ffn_producer_done_q) begin
                assert(q8_buffer_bank_valid[0] || |scratch_error_q);
                assert(!ffn_producer_busy_q);
            end
            if (scratch_section_done_q && !scratch_section_active_q &&
                !(|scratch_error_q)) begin
                assert(!ffn_producer_busy_q);
                assert(scratch_rd_owner_q == SCRATCH_RD_NONE);
                assert(!scratch_rd_rsp_valid);
            end
        end

`ifndef FORMAL_DECODE_INTEGRATION
        cover(rst_n && ffn_phase_q == FFN_WAIT_GATE && ffn_gate_ready);
        cover(rst_n && ffn_phase_q == FFN_GATE_RUN &&
              ffn_capture_beat_q == 3'd4);
        cover(rst_n && ffn_phase_q == FFN_SEAL);
        cover(rst_n && ffn_phase_q == FFN_WAIT_DOWN &&
              ffn_producer_done_q);
        cover(rst_n && ffn_phase_q == FFN_DOWN_RUN &&
              ffn_replay_beat_q == 3'd4);
        cover(rst_n && scratch_section_done_q &&
              !scratch_section_active_q && ffn_phase_q == FFN_IDLE);
        cover(rst_n && ffn_abort_cleanup_q &&
              scratch_rd_owner_q == SCRATCH_RD_PAIRER);
`endif
    end
`endif

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
