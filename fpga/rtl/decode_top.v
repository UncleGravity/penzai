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
    , input wire         sim_inject_p3d_scratch_error
    , input wire         sim_inject_residual_numeric_error
    , input wire         sim_inject_down_activation_error
    , input wire         sim_hold_p3d_rd_rsp
    , input wire         sim_force_scratch_abort_strobe
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
    , output wire        formal_p3d_active
    , output wire        formal_p3d_cleanup
    , output wire        formal_p3d_kill
    , output wire [1:0]  formal_p3d_rd_owner
    , output wire [1:0]  formal_q8_owner
    , output wire        formal_p3d_r_load_complete
    , output wire        formal_p3d_norm_sealed
    , output wire        formal_p3d_residual_started
    , output wire        formal_p3d_begin_ok
    , output wire        formal_p3d_fault
    , output wire        formal_p3d_clean_complete
    , output wire        formal_gamma_busy
    , output wire        formal_gamma_valid
    , output wire        formal_rms_rd_req
    , output wire        formal_residual_rd_req
    , output wire        formal_rms_r_wr_valid
    , output wire        formal_residual_r_wr_valid
    , output wire        formal_scratch_r_wr_valid
    , output wire        formal_kernel_output_valid
    , output wire        formal_residual_output_valid
    , output wire        formal_r_valid
    , output wire        formal_norm_error
    , output wire        formal_norm_controller_error
    , output wire        formal_residual_error
    , output wire        formal_shared_activation_idle
    , output wire        formal_compute_acts_tvalid
    , output wire        formal_q8_ingress_start
    , output wire        formal_kernel_start
    , output wire [5:0]  formal_quant_status
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
    reg        scratch_section_resident_strobe;
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

    // V17 P3d lifecycle. MODEL_ROWS==0 deliberately selects the retained v16
    // section contract; nonzero MODEL_ROWS selects the weighted-RMSNorm and
    // exact-residual boundary added in v17.
    reg [13:0] model_rows_q;
    reg [31:0] norm_eps_q;
    reg        norm_load_gamma_strobe;
    reg [13:0] p3d_model_rows_q;
    reg [2:0]  p3d_tokens_q;
    reg [31:0] p3d_eps_q;
    reg        p3d_resident_q;
    reg        p3d_leaf_start_q;
    reg        p3d_active_q;
    reg        p3d_cleanup_q;
    reg        p3d_cleanup_is_abort_q;
    reg        p3d_kill_q;
    reg        p3d_r_load_complete_q;
    reg [13:0] p3d_r_write_count_q;
    reg [13:0] p3d_r_write_expected_q;
    reg [9:0]  p3d_q8_record_count_q;
    reg [9:0]  p3d_q8_record_expected_q;
    reg        p3d_norm_leaf_done_q;
    reg        p3d_norm_q8_done_q;
    reg        p3d_norm_sealed_q;
    reg        p3d_up_kernel_done_q;
    reg        p3d_residual_started_q;
    reg [1:0]  p3d_scratch_error_q;
    reg [5:0]  p3d_quant_status_q;

    reg        gamma_done_q;
    reg        gamma_error_q;
    reg [13:0] gamma_load_rows_q;
    reg [13:0] gamma_sealed_rows_q;
    reg        norm_done_q;
    reg        norm_error_q;
    reg        residual_done_q;
    reg        residual_error_q;
    reg [22:0] norm_scalar_error_q;
    reg [3:0]  norm_gamma_error_q;
    reg        norm_controller_error_q;
    reg [6:0]  residual_error_detail_q;

    localparam [1:0] SCRATCH_RD_NONE   = 2'd0;
    localparam [1:0] SCRATCH_RD_DRAIN  = 2'd1;
    localparam [1:0] SCRATCH_RD_PAIRER = 2'd2;
    localparam [1:0] SCRATCH_RD_P3D    = 2'd3;
    reg [1:0] scratch_rd_owner_q;

    localparam [1:0] P3D_RD_NONE     = 2'd0;
    localparam [1:0] P3D_RD_RMS      = 2'd1;
    localparam [1:0] P3D_RD_RESIDUAL = 2'd2;
    reg [1:0] p3d_rd_owner_q;

    localparam [1:0] Q8_OWNER_NONE = 2'd0;
    localparam [1:0] Q8_OWNER_RMS  = 2'd1;
    localparam [1:0] Q8_OWNER_GATE = 2'd2;
    reg [1:0] q8_owner_q;

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
    wire kernel_start /* verilator public_flat_rd */;
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

    wire rms_gamma_cfg_ready;
    wire rms_gamma_tready /* verilator public_flat_rd */;
    wire rms_gamma_busy /* verilator public_flat_rd */;
    wire rms_gamma_done;
    wire rms_gamma_error;
    wire [3:0] rms_gamma_status;
    wire rms_gamma_valid;
    wire rms_cfg_ready;
    wire rms_busy /* verilator public_flat_rd */;
    wire rms_done;
    wire rms_error;
    wire [22:0] rms_status;
    wire rms_s_axis_tready;
    wire rms_r_wr_valid;
    wire rms_r_wr_ready;
    wire [1:0] rms_r_wr_bank;
    wire [13:0] rms_r_wr_address;
    wire [63:0] rms_r_wr_data;
    wire rms_rd_req_valid;
    wire rms_rd_req_ready;
    wire [1:0] rms_rd_req_role;
    wire [2:0] rms_rd_req_token;
    wire [10:0] rms_rd_req_group;
    wire rms_rd_rsp_valid;
    wire rms_rd_rsp_ready;
    wire [31:0] rms_scalar_data;
    wire rms_scalar_valid;
    wire rms_scalar_ready;
    wire rms_scalar_last;
    wire [1:0] rms_scalar_status;

    wire residual_cfg_ready;
    wire residual_busy;
    wire residual_done;
    wire residual_error;
    wire [6:0] residual_status;
    wire residual_s_axis_tready;
    wire residual_rd_req_valid;
    wire residual_rd_req_ready;
    wire [1:0] residual_rd_req_role;
    wire [2:0] residual_rd_req_token;
    wire [10:0] residual_rd_req_group;
    wire residual_rd_rsp_valid;
    wire residual_rd_rsp_ready;
    wire residual_r_wr_valid;
    wire residual_r_wr_ready;
    wire [1:0] residual_r_wr_bank;
    wire [13:0] residual_r_wr_address;
    wire [63:0] residual_r_wr_data;
    wire [63:0] residual_m_axis_tdata;
    wire [7:0] residual_m_axis_tkeep;
    wire residual_m_axis_tvalid;
    wire residual_m_axis_tready;
    wire residual_m_axis_tlast;

    wire norm_global_idle;
    wire gamma_load_accept;
    wire p3d_section_begin_ok /* verilator public_flat_rd */;
    wire p3d_fault_event;
    // Gamma loading exclusively owns S_AXIS_ACTS.  Every compute launch and
    // direct activation ingress uses this one predicate, so a busy loader can
    // neither share a beat nor admit work which will consume a later beat.
    wire shared_activation_idle = !rms_gamma_busy;
    wire compute_s_axis_acts_tvalid /* verilator public_flat_rd */ =
        s_axis_acts_tvalid && shared_activation_idle;

    wire section_abort_now = scratch_abort_strobe
`ifdef VERILATOR
                             || sim_force_scratch_abort_strobe
`endif
                             ;
    wire p3d_abort_now = section_abort_now || p3d_kill_q;
    wire p3d_leaf_start /* verilator public_flat_rd */ =
                          p3d_leaf_start_q && p3d_active_q &&
                          !p3d_cleanup_q && !p3d_abort_now;
    wire qualified_kernel_start = kernel_start_q && !section_abort_now &&
                                  !ffn_fault_q && !p3d_kill_q &&
                                  shared_activation_idle;
    wire q8_ingress_start /* verilator public_flat_rd */ =
                            shared_activation_idle &&
                            (p3d_leaf_start || ffn_gate_start_q ||
                             (qualified_kernel_start && raw_activation_mode &&
                              !ffn_down_start_q && !p3d_active_q));
    wire q8_ingress_internal_mode = p3d_leaf_start || ffn_gate_start_q;
    wire [15:0] q8_ingress_blocks = p3d_leaf_start ?
                                     {9'd0, p3d_model_rows_q[13:7]} :
                                     (ffn_gate_start_q ?
                                      {9'd0, ffn_rows_q[13:7]} :
                                      num_q1_blocks_q);
    wire [15:0] q8_ingress_cols = p3d_leaf_start ?
                                   {13'd0, p3d_tokens_q} :
                                   (ffn_gate_start_q ?
                                    {13'd0, ffn_tokens_q} : num_cols_q);
    wire [31:0] q8_internal_data = (q8_owner_q == Q8_OWNER_RMS) ?
                                    rms_scalar_data : swiglu_out_data;
    wire q8_internal_last = (q8_owner_q == Q8_OWNER_RMS) ?
                            rms_scalar_last : swiglu_out_last;
    wire [1:0] q8_internal_status = (q8_owner_q == Q8_OWNER_RMS) ?
                                     rms_scalar_status : swiglu_out_status;
    wire q8_internal_valid = (q8_owner_q == Q8_OWNER_RMS) ?
                             rms_scalar_valid :
                             ((q8_owner_q == Q8_OWNER_GATE) ?
                              swiglu_out_valid : 1'b0);
    wire q8_internal_ready;

    q8_ingress u_q8_ingress (
        .clk(clk),
        .rst_n(rst_n),
        .start(q8_ingress_start),
        .abort(q8_ingress_abort),
        .raw_mode(raw_activation_mode || q8_ingress_internal_mode),
        .internal_mode(q8_ingress_internal_mode),
        .num_q1_blocks(q8_ingress_blocks),
        .num_cols(q8_ingress_cols),
        .s_axis_tdata(s_axis_acts_tdata),
        .s_axis_tvalid(compute_s_axis_acts_tvalid),
        .s_axis_tready(raw_acts_tready),
        .s_axis_tlast(s_axis_acts_tlast),
        .internal_data(q8_internal_data),
        .internal_last(q8_internal_last),
        .internal_status(q8_internal_status),
        .internal_valid(q8_internal_valid),
        .internal_ready(q8_internal_ready),
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
                               native_acts_tvalid :
                               compute_s_axis_acts_tvalid);
    assign native_acts_tready = (q8_owner_q == Q8_OWNER_GATE) ?
                                q8_buffer_s_axis_tready :
                                ((q8_owner_q == Q8_OWNER_RMS) ?
                                 kernel_acts_tready :
                                 (raw_activation_mode &&
                                  !q8_buffer_replay_selected ?
                                  kernel_acts_tready : 1'b0));
    assign q8_buffer_m_axis_tready = q8_buffer_replay_selected &&
                                     q8_buffer_replay_healthy &&
                                     kernel_acts_tready;
    wire p3d_external_residual_ingress = p3d_active_q &&
                                          !p3d_resident_q &&
                                          !p3d_r_load_complete_q &&
                                          !p3d_cleanup_q &&
                                          shared_activation_idle;
    assign s_axis_acts_tready = rms_gamma_busy ? rms_gamma_tready :
                                (p3d_external_residual_ingress ?
                                 rms_s_axis_tready :
                                 (p3d_active_q ? 1'b0 :
                                  (internal_activation_mode ? 1'b0 :
                                   (raw_activation_mode ? raw_acts_tready :
                                    kernel_acts_tready))));

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
    wire scratch_rd_quiescent;
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
                        scratch_rd_quiescent;
    assign norm_global_idle = scratch_idle &&
                              (ffn_phase_q == FFN_IDLE) &&
                              !scratch_section_active_q &&
                              !ffn_abort_cleanup_q &&
                              !gate_packer_busy && !ffn_pairer_busy &&
                              shared_activation_idle && !rms_busy &&
                              !residual_busy && !p3d_active_q &&
                              !p3d_cleanup_q && !p3d_kill_q &&
                              !(|q8_buffer_bank_clearing) &&
                              (q8_owner_q == Q8_OWNER_NONE);
    assign gamma_load_accept = norm_load_gamma_strobe &&
                               !scratch_abort_strobe && norm_global_idle &&
                               rms_gamma_cfg_ready;
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
                           (num_rowblocks_q == {6'd0, ffn_rows_q[13:4]}) &&
                           (!p3d_active_q ||
                            (num_q1_blocks_q ==
                             {9'd0, p3d_model_rows_q[13:7]}));
    wire p3d_up_resources_ok = !kernel_busy && !kernel_start_q &&
                               !scratch_tee_run_q && !scratch_only_run_q &&
                               !scratch_wr_busy && !scratch_drain_busy_q &&
                               !scratch_consumer_busy_q && !ffn_gate_run_q &&
                               !ffn_replay_active_q && !ffn_replay_inflight_q &&
                               !gate_packer_busy && !ffn_pairer_busy &&
                               !residual_busy && !p3d_cleanup_q &&
                               !p3d_kill_q && p3d_r_load_complete_q;
    wire ffn_up_preflight_ok = ffn_up_shape_ok && scratch_wr_cfg_ready &&
                               (p3d_active_q ? p3d_up_resources_ok :
                                scratch_idle);

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
                                 ffn_pairer_start_ready &&
                                 (!p3d_active_q ||
                                  (p3d_norm_sealed_q &&
                                   (q8_owner_q == Q8_OWNER_NONE)));

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
                             (num_cols_q == {13'd0, ffn_tokens_q}) &&
                             (!p3d_active_q ||
                              ((num_rows_q == {18'd0, p3d_model_rows_q}) &&
                               (num_rowblocks_q ==
                                {6'd0, p3d_model_rows_q[13:4]})));
    wire ffn_down_preflight_ok = ffn_down_shape_ok && scratch_idle &&
                                 (!p3d_active_q || residual_cfg_ready);

    wire scratch_ddr_preflight_ok = (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                    scratch_idle &&
                                    (ffn_down_candidate ?
                                     (ffn_down_shape_ok &&
                                      (!p3d_active_q || residual_cfg_ready)) :
                                     (!scratch_section_active_q &&
                                      !internal_activation_mode));
    wire scratch_tee_preflight_ok = scratch_tee_shape_ok && scratch_wr_cfg_ready &&
                                    scratch_idle && !scratch_section_active_q;
    wire scratch_launch_ok = shared_activation_idle &&
                             (scratch_ddr_preflight_ok ||
                              scratch_tee_preflight_ok ||
                              ffn_up_preflight_ok ||
                              ffn_gate_preflight_ok ||
                              ffn_down_preflight_ok);
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
    wire p3d_model_rows_power_two = (model_rows_q != 14'd0) &&
        ((model_rows_q & (model_rows_q - 1'b1)) == 14'd0);
    wire p3d_model_rows_ok = (model_rows_q >= 14'd128) &&
                             (model_rows_q <= 14'd4096) &&
                             p3d_model_rows_power_two;
    reg [14:0] p3d_cfg_scalar_count;
    always @* begin
        case (scratch_tokens_q)
            3'd1: p3d_cfg_scalar_count = {1'b0, model_rows_q};
            3'd2: p3d_cfg_scalar_count = {model_rows_q, 1'b0};
            3'd3: p3d_cfg_scalar_count =
                {1'b0, model_rows_q} + {model_rows_q, 1'b0};
            3'd4: p3d_cfg_scalar_count = {model_rows_q[12:0], 2'b00};
            default: p3d_cfg_scalar_count = 15'd0;
        endcase
    end
    wire p3d_eps_ok = !norm_eps_q[31] &&
                      (norm_eps_q[30:23] != 8'd0) &&
                      (norm_eps_q[30:23] != 8'hff);
    wire p3d_resident_metadata_ok = scratch_valid_q[0] &&
                                    (scratch_valid_rows_q[0] == model_rows_q) &&
                                    (scratch_valid_tokens_q[0] ==
                                     scratch_tokens_q);
    wire p3d_section_request = scratch_section_begin_strobe &&
                               (model_rows_q != 14'd0);
    wire p3d_section_begin_bad = p3d_section_request &&
                                 !scratch_abort_strobe &&
                                 !p3d_section_begin_ok;
    assign p3d_section_begin_ok = p3d_section_request &&
                                  !scratch_abort_strobe &&
                                  shared_activation_idle &&
                                  (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                  scratch_section_shape_ok &&
                                  p3d_model_rows_ok && p3d_eps_ok &&
                                  rms_gamma_valid &&
                                  (gamma_sealed_rows_q == model_rows_q) &&
                                  rms_cfg_ready && q8_buffer_cfg_ready &&
                                  norm_global_idle &&
                                  (!scratch_section_resident_strobe ||
                                   p3d_resident_metadata_ok);
    wire legacy_section_begin_ok = scratch_section_begin_strobe &&
                                   (model_rows_q == 14'd0) &&
                                   !scratch_abort_strobe &&
                                   shared_activation_idle &&
                                   (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                   scratch_section_shape_ok &&
                                   q8_buffer_cfg_ready && scratch_idle &&
                                   !scratch_section_active_q;
    wire scratch_section_begin_ok = p3d_section_begin_ok ||
                                    legacy_section_begin_ok;
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
            p3d_leaf_start_q         <= 1'b0;
        end else begin
            kernel_start_q <= start_strobe && scratch_launch_ok &&
                              !scratch_abort_strobe && !ffn_fault_q;
            scratch_tee_start_q <= start_strobe && scratch_tee_preflight_ok &&
                                   shared_activation_idle &&
                                   !scratch_abort_strobe && !ffn_fault_q;
            scratch_only_start_q <= start_strobe && ffn_up_preflight_ok &&
                                    shared_activation_idle &&
                                    !scratch_abort_strobe && !ffn_fault_q;
            scratch_consumer_start_q <= 1'b0;
            ffn_gate_start_q <= start_strobe && ffn_gate_preflight_ok &&
                                shared_activation_idle &&
                                !scratch_abort_strobe && !ffn_fault_q;
            ffn_down_start_q <= start_strobe && ffn_down_preflight_ok &&
                                shared_activation_idle &&
                                !scratch_abort_strobe && !ffn_fault_q;
            p3d_leaf_start_q <= p3d_section_begin_ok;
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
                            !(p3d_active_q && p3d_residual_started_q) &&
                            (!scratch_tee_active || scratch_wr_tready);
    wire gate_packer_s_axis_tready;
    assign kernel_m_axis_tready = (p3d_active_q &&
                                  p3d_residual_started_q) ?
                                  (residual_s_axis_tready && !p3d_kill_q &&
                                   !p3d_fault_event) :
                                  (ffn_gate_run_q ? gate_packer_s_axis_tready :
                                  (scratch_only_active ? scratch_wr_tready :
                                  (scratch_tee_active ?
                                   (m_axis_tready && scratch_wr_tready) :
                                   (scratch_drain_busy_q ? 1'b0 :
                                    m_axis_tready))));

    wire scratch_writer_abort = scratch_abort_strobe || p3d_kill_q ||
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
    wire ffn_pipeline_abort = section_abort_now || p3d_kill_q;

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

    wire rms_write_phase = p3d_active_q && !p3d_resident_q &&
                           !p3d_r_load_complete_q && !p3d_cleanup_q;
    wire residual_write_phase = p3d_active_q && p3d_residual_started_q &&
                                !p3d_cleanup_q;
    wire p3d_write_conflict = rms_r_wr_valid && residual_r_wr_valid;
    wire scratch_r_wr_valid = (rms_write_phase && rms_r_wr_valid) ||
                              (residual_write_phase && residual_r_wr_valid);
    wire scratch_r_wr_ready;
    wire scratch_r_wr_error;
    wire [1:0] scratch_r_wr_bank = rms_write_phase ? rms_r_wr_bank :
                                    residual_r_wr_bank;
    wire [13:0] scratch_r_wr_address = rms_write_phase ?
                                        rms_r_wr_address :
                                        residual_r_wr_address;
    wire [63:0] scratch_r_wr_data = rms_write_phase ? rms_r_wr_data :
                                      residual_r_wr_data;
    assign rms_r_wr_ready = rms_write_phase && !residual_r_wr_valid &&
                            scratch_r_wr_ready;
    assign residual_r_wr_ready = residual_write_phase && !rms_r_wr_valid &&
                                 scratch_r_wr_ready;

    section_rmsnorm_scalar_pipeline u_section_rmsnorm_scalar_pipeline (
        .clk(clk),
        .rst_n(rst_n),
        .abort_run(p3d_abort_now),
        .gamma_cfg_valid(gamma_load_accept),
        .gamma_cfg_ready(rms_gamma_cfg_ready),
        .gamma_cfg_rows(model_rows_q),
        .gamma_tdata(s_axis_acts_tdata),
        .gamma_tkeep(s_axis_acts_tkeep),
        .gamma_tvalid(s_axis_acts_tvalid && rms_gamma_busy),
        .gamma_tready(rms_gamma_tready),
        .gamma_tlast(s_axis_acts_tlast),
        .gamma_busy(rms_gamma_busy),
        .gamma_done(rms_gamma_done),
        .gamma_error(rms_gamma_error),
        .gamma_status(rms_gamma_status),
        .gamma_valid(rms_gamma_valid),
        .cfg_valid(p3d_leaf_start),
        .cfg_ready(rms_cfg_ready),
        .cfg_rows(p3d_model_rows_q),
        .cfg_tokens(p3d_tokens_q),
        .cfg_eps(p3d_eps_q),
        .cfg_resident(p3d_resident_q),
        .busy(rms_busy),
        .done(rms_done),
        .error(rms_error),
        .status(rms_status),
        .s_axis_tdata(s_axis_acts_tdata),
        .s_axis_tkeep(s_axis_acts_tkeep),
        .s_axis_tvalid(s_axis_acts_tvalid &&
                       p3d_external_residual_ingress),
        .s_axis_tready(rms_s_axis_tready),
        .s_axis_tlast(s_axis_acts_tlast),
        .r_wr_valid(rms_r_wr_valid),
        .r_wr_ready(rms_r_wr_ready),
        .r_wr_error(scratch_r_wr_error),
        .r_wr_bank(rms_r_wr_bank),
        .r_wr_address(rms_r_wr_address),
        .r_wr_data(rms_r_wr_data),
        .rd_req_valid(rms_rd_req_valid),
        .rd_req_ready(rms_rd_req_ready),
        .rd_req_role(rms_rd_req_role),
        .rd_req_token(rms_rd_req_token),
        .rd_req_group(rms_rd_req_group),
        .rd_rsp_valid(rms_rd_rsp_valid),
        .rd_rsp_ready(rms_rd_rsp_ready),
        .rd_rsp_data(scratch_rd_rsp_data),
        .rd_rsp_error(scratch_rd_rsp_error
`ifdef VERILATOR
                      || sim_inject_p3d_scratch_error
`endif
                     ),
        .scalar_data(rms_scalar_data),
        .scalar_valid(rms_scalar_valid),
        .scalar_ready(rms_scalar_ready),
        .scalar_last(rms_scalar_last),
        .scalar_status(rms_scalar_status)
    );

    wire residual_cfg_valid = p3d_active_q && ffn_down_start_q;
    section_residual_add u_section_residual_add (
        .clk(clk),
        .rst_n(rst_n),
        .abort_run(p3d_abort_now),
        .cfg_valid(residual_cfg_valid),
        .cfg_ready(residual_cfg_ready),
        .cfg_rows(p3d_model_rows_q),
        .cfg_tokens(p3d_tokens_q),
        .busy(residual_busy),
        .done(residual_done),
        .error(residual_error),
        .status(residual_status),
        .s_axis_tdata(
`ifdef VERILATOR
            sim_inject_residual_numeric_error ?
                64'h7fc0_0000_7fc0_0000 :
`endif
            kernel_m_axis_tdata),
        .s_axis_tkeep(kernel_m_axis_tkeep),
        .s_axis_tvalid(p3d_active_q && p3d_residual_started_q &&
                       !p3d_kill_q && !p3d_fault_event &&
                       kernel_m_axis_tvalid),
        .s_axis_tready(residual_s_axis_tready),
        .s_axis_tlast(kernel_m_axis_tlast),
        .rd_req_valid(residual_rd_req_valid),
        .rd_req_ready(residual_rd_req_ready),
        .rd_req_role(residual_rd_req_role),
        .rd_req_token(residual_rd_req_token),
        .rd_req_group(residual_rd_req_group),
        .rd_rsp_valid(residual_rd_rsp_valid),
        .rd_rsp_ready(residual_rd_rsp_ready),
        .rd_rsp_data(scratch_rd_rsp_data),
        .rd_rsp_error(scratch_rd_rsp_error
`ifdef VERILATOR
                      || sim_inject_p3d_scratch_error
`endif
                     ),
        .r_wr_valid(residual_r_wr_valid),
        .r_wr_ready(residual_r_wr_ready),
        .r_wr_error(scratch_r_wr_error),
        .r_wr_bank(residual_r_wr_bank),
        .r_wr_address(residual_r_wr_address),
        .r_wr_data(residual_r_wr_data),
        .m_axis_tdata(residual_m_axis_tdata),
        .m_axis_tkeep(residual_m_axis_tkeep),
        .m_axis_tvalid(residual_m_axis_tvalid),
        .m_axis_tready(residual_m_axis_tready),
        .m_axis_tlast(residual_m_axis_tlast)
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
        .r_wr_valid(scratch_r_wr_valid),
        .r_wr_ready(scratch_r_wr_ready),
        .r_wr_bank(scratch_r_wr_bank),
        .r_wr_address(scratch_r_wr_address),
        .r_wr_data(scratch_r_wr_data),
        .r_wr_error(scratch_r_wr_error),
        .rd_req_valid(scratch_rd_req_valid),
        .rd_req_ready(scratch_rd_req_ready),
        .rd_quiescent(scratch_rd_quiescent),
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
                                  scratch_idle && !scratch_section_active_q &&
                                  shared_activation_idle;
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
    wire p3d_read_conflict = rms_rd_req_valid && residual_rd_req_valid;
    wire p3d_select_rms = (p3d_rd_owner_q == P3D_RD_NONE) &&
                          rms_rd_req_valid && !residual_rd_req_valid;
    wire p3d_select_residual = (p3d_rd_owner_q == P3D_RD_NONE) &&
                               residual_rd_req_valid && !rms_rd_req_valid;
    wire p3d_rd_req_valid = p3d_select_rms || p3d_select_residual;
    wire scratch_select_p3d = (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                              p3d_rd_req_valid;
    wire scratch_select_pairer = (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                                  !p3d_rd_req_valid &&
                                  ffn_pairer_rd_req_valid;
    wire scratch_select_drain = (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                                !p3d_rd_req_valid && !ffn_pairer_rd_req_valid &&
                                scratch_drain_req_valid;
    assign scratch_rd_req_valid = scratch_select_p3d || scratch_select_pairer ||
                                  scratch_select_drain;
    assign scratch_rd_req_role = scratch_select_p3d ?
                                 (p3d_select_rms ? rms_rd_req_role :
                                  residual_rd_req_role) :
                                 (scratch_select_pairer ? ffn_pairer_rd_req_role :
                                  scratch_drain_role_q);
    assign scratch_rd_req_token = scratch_select_p3d ?
                                  (p3d_select_rms ? rms_rd_req_token :
                                   residual_rd_req_token) :
                                  (scratch_select_pairer ?
                                   ffn_pairer_rd_req_token :
                                   scratch_drain_token_q);
    assign scratch_rd_req_group = scratch_select_p3d ?
                                  (p3d_select_rms ? rms_rd_req_group :
                                   residual_rd_req_group) :
                                  (scratch_select_pairer ?
                                   ffn_pairer_rd_req_group :
                                   scratch_drain_group_q);
    assign rms_rd_req_ready = scratch_select_p3d && p3d_select_rms &&
                              scratch_rd_req_ready;
    assign residual_rd_req_ready = scratch_select_p3d &&
                                   p3d_select_residual &&
                                   scratch_rd_req_ready;
    assign ffn_pairer_rd_req_ready = scratch_select_pairer &&
                                     scratch_rd_req_ready;

    wire scratch_drain_rsp_valid = scratch_rd_rsp_valid &&
                                   (scratch_rd_owner_q == SCRATCH_RD_DRAIN);
    assign ffn_pairer_rd_rsp_valid = scratch_rd_rsp_valid &&
                                     (scratch_rd_owner_q == SCRATCH_RD_PAIRER);
    assign rms_rd_rsp_valid = scratch_rd_rsp_valid &&
                                   (scratch_rd_owner_q == SCRATCH_RD_P3D) &&
                              (p3d_rd_owner_q == P3D_RD_RMS)
`ifdef VERILATOR
                              && !sim_hold_p3d_rd_rsp
`endif
                              ;
    assign residual_rd_rsp_valid = scratch_rd_rsp_valid &&
                                   (scratch_rd_owner_q == SCRATCH_RD_P3D) &&
                                   (p3d_rd_owner_q == P3D_RD_RESIDUAL)
`ifdef VERILATOR
                                   && !sim_hold_p3d_rd_rsp
`endif
                                   ;
    wire scratch_drain_rsp_ready = !scratch_drain_busy_q ? 1'b1 :
                                   (scratch_drain_have_group_q &&
                                    (scratch_rd_rsp_error ||
                                     (scratch_drain_emit_valid_q &&
                                      (scratch_drain_bank_q == 2'd3) &&
                                      m_axis_tready)));
    assign scratch_rd_rsp_ready =
        (scratch_rd_owner_q == SCRATCH_RD_PAIRER) ? ffn_pairer_rd_rsp_ready :
        (scratch_rd_owner_q == SCRATCH_RD_DRAIN) ? scratch_drain_rsp_ready :
        (scratch_rd_owner_q == SCRATCH_RD_P3D) ?
            (
`ifdef VERILATOR
             sim_hold_p3d_rd_rsp ? 1'b0 :
`endif
             (p3d_rd_owner_q == P3D_RD_RMS) ? rms_rd_rsp_ready :
             (p3d_rd_owner_q == P3D_RD_RESIDUAL) ?
                residual_rd_rsp_ready : scratch_rd_rsp_valid) :
        scratch_rd_rsp_valid;

    wire scratch_rd_req_accept = scratch_rd_req_valid && scratch_rd_req_ready;
    wire scratch_rd_rsp_accept = scratch_rd_rsp_valid && scratch_rd_rsp_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            scratch_rd_owner_q <= SCRATCH_RD_NONE;
            p3d_rd_owner_q <= P3D_RD_NONE;
        end else if (scratch_rd_req_accept) begin
            scratch_rd_owner_q <= scratch_select_p3d ? SCRATCH_RD_P3D :
                                  (scratch_select_pairer ?
                                   SCRATCH_RD_PAIRER : SCRATCH_RD_DRAIN);
            if (scratch_select_p3d)
                p3d_rd_owner_q <= p3d_select_rms ? P3D_RD_RMS :
                                  P3D_RD_RESIDUAL;
        end else if (scratch_rd_rsp_accept) begin
            scratch_rd_owner_q <= SCRATCH_RD_NONE;
            if (scratch_rd_owner_q == SCRATCH_RD_P3D)
                p3d_rd_owner_q <= P3D_RD_NONE;
        end
    end

    assign swiglu_in_valid = ffn_pairer_out_valid;
    assign swiglu_in_gate = ffn_pairer_out_gate;
    assign swiglu_in_up = ffn_pairer_out_up;
    assign swiglu_in_last = ffn_pairer_out_last;
    assign ffn_pairer_out_ready = swiglu_in_ready;
    assign rms_scalar_ready = (q8_owner_q == Q8_OWNER_RMS) ?
                              q8_internal_ready : 1'b0;
    assign swiglu_out_ready = (q8_owner_q == Q8_OWNER_GATE) ?
                              q8_internal_ready : 1'b0;

    // Section abort is a lifecycle boundary for every stage that can retain
    // work. A traversal failure also kills any scalar, quantization, or native
    // record still owned by the section before software starts another run.
    assign q8_ingress_abort = section_abort_now || p3d_kill_q;

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

    wire q8_buffer_cfg_p3d = p3d_leaf_start;
    wire q8_buffer_cfg_valid = legacy_section_begin_ok || q8_buffer_cfg_p3d;
    wire [2:0] q8_buffer_cfg_tokens = q8_buffer_cfg_p3d ?
                                      ffn_tokens_q : scratch_tokens_q;
    wire [8:0] q8_buffer_cfg_blocks = q8_buffer_cfg_p3d ?
                                      ffn_blocks_q : scratch_rows_q[13:5];
    wire q8_buffer_seal_valid = ffn_seal_pending_q;
    wire q8_buffer_abort_valid = section_abort_now || p3d_kill_q;
    wire q8_buffer_rd_req_valid = ffn_replay_active_q &&
                                  !ffn_replay_inflight_q;
    wire q8_buffer_rd_req_ready;
    wire q8_buffer_capture_valid = (q8_owner_q == Q8_OWNER_GATE) &&
                                   ffn_producer_busy_q &&
                                   native_acts_tvalid;

    section_q8_buffer u_section_q8_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_valid(q8_buffer_cfg_valid),
        .cfg_ready(q8_buffer_cfg_ready),
        .cfg_bank(1'b0),
        .cfg_tokens(q8_buffer_cfg_tokens),
        .cfg_blocks(q8_buffer_cfg_blocks),
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
        .s_axis_tvalid(q8_buffer_capture_valid),
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

    wire p3d_residual_output_selected = p3d_active_q &&
                                        p3d_residual_started_q;
    assign residual_m_axis_tready = p3d_residual_output_selected &&
                                    !p3d_kill_q && !p3d_fault_event &&
                                    m_axis_tready;
    assign m_axis_tdata  = p3d_residual_output_selected ?
                           residual_m_axis_tdata :
                           (scratch_drain_busy_q ? scratch_drain_emit_data_q :
                            kernel_m_axis_tdata);
    assign m_axis_tvalid = p3d_residual_output_selected ?
                           (residual_m_axis_tvalid && !p3d_kill_q &&
                            !p3d_fault_event) :
                           (scratch_drain_busy_q ? scratch_drain_tvalid :
                            ddr_kernel_valid);
    assign m_axis_tlast  = p3d_residual_output_selected ?
                           residual_m_axis_tlast :
                           (scratch_drain_busy_q ? scratch_drain_tlast :
                            kernel_m_axis_tlast);
    assign m_axis_tkeep  = p3d_residual_output_selected ?
                           residual_m_axis_tkeep :
                           (scratch_drain_busy_q ? 8'hff :
                            kernel_m_axis_tkeep);

    wire q8_capture_fire /* verilator public_flat_rd */ =
        ffn_producer_busy_q && native_acts_tvalid &&
        q8_buffer_s_axis_tready;
    wire rms_q8_record_fire = (q8_owner_q == Q8_OWNER_RMS) &&
                              q8_internal_record_done;
    wire rms_q8_final_record_fire = rms_q8_record_fire &&
        (p3d_q8_record_count_q + 1'b1 == p3d_q8_record_expected_q);
    wire rms_r_write_fire = rms_r_wr_valid && rms_r_wr_ready;
    wire rms_r_final_write_fire = rms_r_write_fire &&
        (p3d_r_write_count_q + 1'b1 == p3d_r_write_expected_q);
    wire p3d_norm_seal_event = p3d_active_q && !p3d_cleanup_q &&
        !p3d_norm_sealed_q &&
        (p3d_norm_leaf_done_q || (rms_done && !rms_error)) &&
        (p3d_norm_q8_done_q || rms_q8_final_record_fire) &&
        !rms_error && !q8_fault_live;
    wire p3d_accounting_fault =
        (rms_r_write_fire &&
         (p3d_r_write_count_q >= p3d_r_write_expected_q)) ||
        (rms_q8_record_fire &&
         (p3d_q8_record_count_q >= p3d_q8_record_expected_q));
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
    wire q8_fault_live = q8_activation_abort
`ifdef VERILATOR
                         || sim_inject_q8_numeric_error
`endif
                        ;
    wire [5:0] q8_fault_status_now = (quantizer_status != 6'd0) ?
                                           quantizer_status :
                                           (q8_fault_live ? 6'b00_1000 : 6'd0);
    wire p3d_q8_fault_now = p3d_active_q &&
                             (q8_owner_q != Q8_OWNER_NONE) &&
                             q8_fault_live;
    wire q8_numeric_fault = ffn_producer_busy_q && q8_fault_live;
    wire producer_leaf_fault = (ffn_phase_q == FFN_GATE_RUN) &&
                               (gate_packer_error || ffn_pairer_error ||
                                q8_numeric_fault ||
                                q8_buffer_cap_record_error ||
                                q8_buffer_bank_error[0] || activation_error);
    wire ffn_fault_event = up_run_fault || producer_leaf_fault ||
                           ((ffn_phase_q == FFN_DOWN_RUN) &&
                            activation_error) || q8_replay_fault ||
                           ((ffn_phase_q == FFN_SEAL) &&
                            q8_buffer_seal_error);
    assign p3d_fault_event = p3d_active_q && !p3d_cleanup_q &&
                             (rms_error || residual_error ||
                              p3d_q8_fault_now ||
                              p3d_read_conflict || p3d_write_conflict ||
                              p3d_accounting_fault ||
                              (scratch_r_wr_error && scratch_r_wr_valid) ||
                              ffn_fault_event);
    wire section_fault_event = scratch_section_active_q &&
                               !ffn_abort_cleanup_q &&
                               (p3d_active_q ? p3d_fault_event :
                                ffn_fault_event);
    wire kernel_section_abort = scratch_abort_strobe || ffn_fault_q ||
                                (p3d_active_q ? p3d_kill_q :
                                 section_fault_event);
    wire sim_down_activation_abort =
`ifdef VERILATOR
        sim_inject_down_activation_error && p3d_active_q &&
        p3d_residual_started_q;
`else
        1'b0;
`endif

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
                          sim_down_activation_abort ||
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

    wire p3d_cleanup_resources_idle = !rms_busy && !residual_busy &&
                                       !kernel_busy && !gate_packer_busy &&
                                       !ffn_pairer_busy && !scratch_wr_busy &&
                                       !scratch_tee_run_q &&
                                       !scratch_only_run_q &&
                                       !scratch_drain_busy_q &&
                                       !scratch_consumer_busy_q &&
                                       (scratch_rd_owner_q ==
                                        SCRATCH_RD_NONE) &&
                                       (p3d_rd_owner_q == P3D_RD_NONE) &&
                                       !scratch_rd_rsp_valid &&
                                       (q8_owner_q == Q8_OWNER_NONE);
    wire p3d_cleanup_retire = p3d_cleanup_q &&
                              p3d_cleanup_resources_idle;
    wire p3d_clean_complete = (ffn_phase_q == FFN_DOWN_RUN) &&
                               p3d_active_q && residual_done &&
                               !residual_error &&
                               (ffn_down_kernel_done_q || kernel_done) &&
                               ffn_replay_complete_q && !activation_error &&
                               !ffn_fault_q && !p3d_fault_event;

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
                end else if (!p3d_active_q || p3d_norm_sealed_q ||
                             p3d_norm_seal_event) begin
                    ffn_phase_q <= FFN_WAIT_GATE;
                end
            end

            if (p3d_active_q && (ffn_phase_q == FFN_UP_RUN) &&
                p3d_up_kernel_done_q && p3d_norm_seal_event)
                ffn_phase_q <= FFN_WAIT_GATE;

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

            if ((ffn_phase_q == FFN_DOWN_RUN) && !p3d_active_q &&
                (ffn_down_kernel_done_q || kernel_done) &&
                ffn_replay_complete_q && !activation_error &&
                !ffn_fault_q) begin
                ffn_phase_q <= FFN_IDLE;
                scratch_section_active_q <= 1'b0;
                scratch_section_done_q <= 1'b1;
                scratch_valid_q <= 4'd0;
            end


            if (p3d_clean_complete) begin
                ffn_phase_q <= FFN_IDLE;
                scratch_section_active_q <= 1'b0;
                scratch_section_done_q <= 1'b1;
                scratch_valid_q <= 4'b0001;
                scratch_valid_rows_q[0] <= p3d_model_rows_q;
                scratch_valid_tokens_q[0] <= p3d_tokens_q;
            end

            if (section_fault_event) begin
                ffn_fault_q <= 1'b1;
                scratch_error_q[5] <= 1'b1;
                if (q8_numeric_fault)
                    scratch_error_q[6] <= 1'b1;
                ffn_seal_pending_q <= 1'b0;
            end

            if (p3d_kill_q) begin
                ffn_gate_run_q <= 1'b0;
                ffn_seal_pending_q <= 1'b0;
                ffn_replay_active_q <= 1'b0;
                ffn_replay_inflight_q <= 1'b0;
                scratch_tee_run_q <= 1'b0;
                scratch_only_run_q <= 1'b0;
            end

            if (p3d_cleanup_retire) begin
                ffn_phase_q <= FFN_IDLE;
                ffn_fault_q <= 1'b0;
                ffn_gate_run_q <= 1'b0;
                ffn_producer_busy_q <= 1'b0;
                ffn_producer_done_q <= 1'b0;
                ffn_seal_pending_q <= 1'b0;
                ffn_replay_active_q <= 1'b0;
                ffn_replay_inflight_q <= 1'b0;
                scratch_section_active_q <= 1'b0;
                scratch_section_done_q <= p3d_cleanup_is_abort_q ?
                                          1'b0 : 1'b1;
                scratch_valid_q <= 4'd0;
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
                if (!scratch_drain_shape_ok || !scratch_idle ||
                    !shared_activation_idle)
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
                    scratch_section_done_q   <= p3d_active_q ? 1'b0 : 1'b1;
                    scratch_valid_q          <= 4'd0;
                    scratch_error_q[2]       <= 1'b1;
                end
            end

            if (ffn_abort_cleanup_q && !kernel_busy && !gate_packer_busy &&
                !ffn_pairer_busy &&
                (scratch_rd_owner_q == SCRATCH_RD_NONE) &&
                !scratch_rd_rsp_valid && !p3d_active_q &&
                !p3d_cleanup_q) begin
                ffn_abort_cleanup_q <= 1'b0;
                ffn_phase_q <= FFN_IDLE;
                ffn_producer_busy_q <= 1'b0;
                ffn_producer_done_q <= 1'b0;
                scratch_section_active_q <= 1'b0;
            end
        end
    end

    wire gate_q8_final_record_fire =
        (q8_owner_q == Q8_OWNER_GATE) && q8_internal_record_done &&
        (ffn_capture_block_q + 1'b1 == ffn_blocks_q) &&
        ({1'b0, ffn_capture_token_q} + 1'b1 == ffn_tokens_q);
    wire gamma_load_rejected = norm_load_gamma_strobe &&
                               !scratch_abort_strobe &&
                               !gamma_load_accept;
    wire p3d_norm_fault_now = rms_error ||
        ((q8_owner_q == Q8_OWNER_RMS) && p3d_q8_fault_now) ||
        p3d_accounting_fault ||
        (p3d_read_conflict && !p3d_residual_started_q) ||
        (p3d_write_conflict && !p3d_residual_started_q) ||
        (scratch_r_wr_error && rms_write_phase);
    wire p3d_residual_fault_now = residual_error ||
        (p3d_read_conflict && p3d_residual_started_q) ||
        (p3d_write_conflict && p3d_residual_started_q) ||
        (scratch_r_wr_error && residual_write_phase);

    // P3d owns lifecycle/status and the registered cross-abort. Combinational
    // fault detection only closes ingress/output in the diagnosis cycle; every
    // leaf observes the registered kill on the following edge and remains killed
    // until all retained scratch ownership has drained.
    always @(posedge clk) begin
        if (!rst_n) begin
            p3d_model_rows_q <= 14'd0;
            p3d_tokens_q <= 3'd0;
            p3d_eps_q <= 32'd0;
            p3d_resident_q <= 1'b0;
            p3d_active_q <= 1'b0;
            p3d_cleanup_q <= 1'b0;
            p3d_cleanup_is_abort_q <= 1'b0;
            p3d_kill_q <= 1'b0;
            p3d_r_load_complete_q <= 1'b0;
            p3d_r_write_count_q <= 14'd0;
            p3d_r_write_expected_q <= 14'd0;
            p3d_q8_record_count_q <= 10'd0;
            p3d_q8_record_expected_q <= 10'd0;
            p3d_norm_leaf_done_q <= 1'b0;
            p3d_norm_q8_done_q <= 1'b0;
            p3d_norm_sealed_q <= 1'b0;
            p3d_up_kernel_done_q <= 1'b0;
            p3d_residual_started_q <= 1'b0;
            p3d_scratch_error_q <= 2'd0;
            p3d_quant_status_q <= 6'd0;
            q8_owner_q <= Q8_OWNER_NONE;
            gamma_done_q <= 1'b0;
            gamma_error_q <= 1'b0;
            gamma_load_rows_q <= 14'd0;
            gamma_sealed_rows_q <= 14'd0;
            norm_done_q <= 1'b0;
            norm_error_q <= 1'b0;
            residual_done_q <= 1'b0;
            residual_error_q <= 1'b0;
            norm_scalar_error_q <= 23'd0;
            norm_gamma_error_q <= 4'd0;
            norm_controller_error_q <= 1'b0;
            residual_error_detail_q <= 7'd0;
        end else begin
            // Cross-abort is a registered pulse. The independent cleanup bit
            // remains asserted until every leaf and retained scratch owner is
            // idle; holding abort high would prevent the leaves from retiring
            // their own cleanup states.
            p3d_kill_q <= 1'b0;
            if (gamma_load_accept || scratch_section_begin_ok ||
                (start_strobe && scratch_launch_ok))
                p3d_quant_status_q <= 6'd0;
            if (p3d_active_q && (q8_owner_q != Q8_OWNER_NONE) &&
                (quantizer_status != 6'd0))
                p3d_quant_status_q <= quantizer_status;
            if (p3d_q8_fault_now)
                p3d_quant_status_q <= q8_fault_status_now;
            if (gamma_load_accept) begin
                gamma_done_q <= 1'b0;
                gamma_error_q <= 1'b0;
                gamma_load_rows_q <= model_rows_q;
                gamma_sealed_rows_q <= 14'd0;
                norm_gamma_error_q <= 4'd0;
                norm_controller_error_q <= 1'b0;
            end else if (gamma_load_rejected) begin
                gamma_done_q <= 1'b1;
                gamma_error_q <= 1'b1;
                norm_controller_error_q <= 1'b1;
            end

            if (rms_gamma_done) begin
                gamma_done_q <= 1'b1;
                gamma_error_q <= rms_gamma_error;
                norm_gamma_error_q <= rms_gamma_status;
                if (!rms_gamma_error && rms_gamma_valid)
                    gamma_sealed_rows_q <= gamma_load_rows_q;
                else
                    gamma_sealed_rows_q <= 14'd0;
            end

            if (p3d_section_begin_bad) begin
                norm_done_q <= 1'b1;
                norm_error_q <= 1'b1;
                norm_controller_error_q <= 1'b1;
            end

            if (p3d_section_begin_ok) begin
                p3d_model_rows_q <= model_rows_q;
                p3d_tokens_q <= scratch_tokens_q;
                p3d_eps_q <= norm_eps_q;
                p3d_resident_q <= scratch_section_resident_strobe;
                p3d_active_q <= 1'b1;
                p3d_cleanup_q <= 1'b0;
                p3d_cleanup_is_abort_q <= 1'b0;
                p3d_kill_q <= 1'b0;
                p3d_r_load_complete_q <=
                    scratch_section_resident_strobe;
                p3d_r_write_count_q <= 14'd0;
                p3d_r_write_expected_q <= p3d_cfg_scalar_count[14:1];
                p3d_q8_record_count_q <= 10'd0;
                p3d_q8_record_expected_q <= p3d_cfg_scalar_count[14:5];
                p3d_norm_leaf_done_q <= 1'b0;
                p3d_norm_q8_done_q <= 1'b0;
                p3d_norm_sealed_q <= 1'b0;
                p3d_up_kernel_done_q <= 1'b0;
                p3d_residual_started_q <= 1'b0;
                p3d_scratch_error_q <= 2'd0;
                q8_owner_q <= Q8_OWNER_RMS;
                norm_done_q <= 1'b0;
                norm_error_q <= 1'b0;
                residual_done_q <= 1'b0;
                residual_error_q <= 1'b0;
                norm_scalar_error_q <= 23'd0;
                norm_controller_error_q <= 1'b0;
                residual_error_detail_q <= 7'd0;
            end else if (p3d_active_q && !p3d_cleanup_q) begin
                if (rms_r_write_fire) begin
                    p3d_r_write_count_q <= p3d_r_write_count_q + 1'b1;
                    if (rms_r_final_write_fire)
                        p3d_r_load_complete_q <= 1'b1;
                end
                if (rms_q8_record_fire) begin
                    p3d_q8_record_count_q <= p3d_q8_record_count_q + 1'b1;
                    if (rms_q8_final_record_fire) begin
                        p3d_norm_q8_done_q <= 1'b1;
                        q8_owner_q <= Q8_OWNER_NONE;
                    end
                end
                if (rms_done && !rms_error)
                    p3d_norm_leaf_done_q <= 1'b1;
                if (p3d_norm_seal_event) begin
                    p3d_norm_sealed_q <= 1'b1;
                    norm_done_q <= 1'b1;
                    norm_error_q <= 1'b0;
                    norm_scalar_error_q <= 23'd0;
                    q8_owner_q <= Q8_OWNER_NONE;
                end
                if ((ffn_phase_q == FFN_UP_RUN) && kernel_done)
                    p3d_up_kernel_done_q <= 1'b1;
                if (ffn_gate_start_q)
                    q8_owner_q <= Q8_OWNER_GATE;
                if (gate_q8_final_record_fire)
                    q8_owner_q <= Q8_OWNER_NONE;
                if (ffn_down_start_q)
                    p3d_residual_started_q <= 1'b1;
                if (p3d_clean_complete) begin
                    p3d_active_q <= 1'b0;
                    p3d_residual_started_q <= 1'b0;
                    residual_done_q <= 1'b1;
                    residual_error_q <= 1'b0;
                    residual_error_detail_q <= 7'd0;
                    q8_owner_q <= Q8_OWNER_NONE;
                end
            end else if (!p3d_active_q) begin
                if (ffn_gate_start_q)
                    q8_owner_q <= Q8_OWNER_GATE;
                if (gate_q8_final_record_fire)
                    q8_owner_q <= Q8_OWNER_NONE;
            end

            if (p3d_fault_event) begin
                p3d_cleanup_q <= 1'b1;
                p3d_cleanup_is_abort_q <= 1'b0;
                p3d_kill_q <= 1'b1;
                gamma_sealed_rows_q <= 14'd0;
                if (!p3d_norm_sealed_q) begin
                    norm_error_q <= 1'b1;
                    if (p3d_norm_fault_now) begin
                        norm_scalar_error_q <= (rms_status != 23'd0) ?
                                               rms_status : 23'h40_0000;
                        p3d_scratch_error_q[0] <= 1'b1;
                    end else begin
                        norm_controller_error_q <= 1'b1;
                    end
                end else if (!p3d_residual_started_q) begin
                    // The weighted norm itself sealed, but a later UP/GATE/SEAL
                    // failure still needs a terminal v17 attribution. There is
                    // no separate FFN status bank, so classify it as a section
                    // controller failure while preserving NORM_DONE.
                    norm_error_q <= 1'b1;
                    norm_controller_error_q <= 1'b1;
                end
                if (p3d_residual_started_q) begin
                    residual_error_q <= 1'b1;
                    residual_error_detail_q <=
                        (p3d_residual_fault_now &&
                         (residual_status != 7'd0)) ?
                            residual_status : 7'h40;
                    p3d_scratch_error_q[1] <= 1'b1;
                end
                if (p3d_read_conflict || p3d_write_conflict ||
                    p3d_accounting_fault)
                    norm_controller_error_q <= 1'b1;
            end

            if (p3d_kill_q)
                q8_owner_q <= Q8_OWNER_NONE;

            if (p3d_cleanup_retire) begin
                p3d_active_q <= 1'b0;
                p3d_cleanup_q <= 1'b0;
                p3d_kill_q <= 1'b0;
                p3d_r_load_complete_q <= 1'b0;
                p3d_norm_sealed_q <= 1'b0;
                p3d_residual_started_q <= 1'b0;
                q8_owner_q <= Q8_OWNER_NONE;
                if (!p3d_cleanup_is_abort_q) begin
                    if (norm_error_q)
                        norm_done_q <= 1'b1;
                    if (residual_error_q)
                        residual_done_q <= 1'b1;
                end
            end

            if (scratch_abort_strobe) begin
                if (p3d_active_q) begin
                    p3d_cleanup_q <= 1'b1;
                    p3d_cleanup_is_abort_q <= 1'b1;
                    p3d_kill_q <= 1'b1;
                end else begin
                    p3d_cleanup_q <= 1'b0;
                    p3d_cleanup_is_abort_q <= 1'b0;
                    p3d_kill_q <= 1'b0;
                end
                p3d_r_load_complete_q <= 1'b0;
                p3d_norm_sealed_q <= 1'b0;
                p3d_residual_started_q <= 1'b0;
                p3d_scratch_error_q <= 2'd0;
                p3d_quant_status_q <= 6'd0;
                q8_owner_q <= Q8_OWNER_NONE;
                gamma_done_q <= 1'b0;
                gamma_error_q <= 1'b0;
                gamma_sealed_rows_q <= 14'd0;
                norm_done_q <= 1'b0;
                norm_error_q <= 1'b0;
                residual_done_q <= 1'b0;
                residual_error_q <= 1'b0;
                norm_scalar_error_q <= 23'd0;
                norm_gamma_error_q <= 4'd0;
                norm_controller_error_q <= 1'b0;
                residual_error_detail_q <= 7'd0;
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
            scratch_section_resident_strobe <= 1'b0;
            model_rows_q <= 14'd0;
            norm_eps_q <= 32'd0;
            norm_load_gamma_strobe <= 1'b0;
        end else begin
            start_strobe <= 1'b0;
            scratch_drain_start_strobe <= 1'b0;
            scratch_abort_strobe <= 1'b0;
            scratch_section_begin_strobe <= 1'b0;
            scratch_section_resident_strobe <= 1'b0;
            norm_load_gamma_strobe <= 1'b0;

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
                            scratch_section_resident_strobe <=
                                s_axi_wdata[2] && s_axi_wdata[3];
                        end
                    end
                    MATMUL_OFF_MODEL_ROWS[7:0]: begin
                        if (s_axi_wstrb[0]) model_rows_q[7:0] <=
                            s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) model_rows_q[13:8] <=
                            s_axi_wdata[13:8];
                    end
                    MATMUL_OFF_NORM_EPS[7:0]: begin
                        if (s_axi_wstrb[0]) norm_eps_q[7:0] <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) norm_eps_q[15:8] <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) norm_eps_q[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) norm_eps_q[31:24] <= s_axi_wdata[31:24];
                    end
                    MATMUL_OFF_NORM_CTRL[7:0]: begin
                        if (s_axi_wstrb[0] && s_axi_wdata[0])
                            norm_load_gamma_strobe <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    reg        arready_q, rvalid_q;
    reg [31:0] rdata_q;

    wire read_accept = !arready_q && s_axi_arvalid && !rvalid_q;
    wire norm_busy_live = p3d_active_q && !norm_done_q;
    wire residual_busy_live = p3d_active_q &&
                              (p3d_residual_started_q || residual_busy);
    wire any_scratch_error = (|scratch_error_q) ||
                             (|p3d_scratch_error_q);
    wire [5:0] quant_status_value = (p3d_quant_status_q != 6'd0) ?
                                     p3d_quant_status_q : quantizer_status;
    wire [31:0] norm_status_value = {
        21'd0,
        norm_global_idle,
        residual_error_q,
        residual_done_q,
        residual_busy_live,
        norm_error_q,
        norm_done_q,
        norm_busy_live,
        rms_gamma_valid,
        gamma_error_q,
        gamma_done_q,
        rms_gamma_busy
    };
    wire [31:0] norm_error_value = {
        4'd0,
        norm_controller_error_q,
        norm_gamma_error_q,
        norm_scalar_error_q
    };

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
                    MATMUL_OFF_QUANT_STATUS[7:0]:  rdata_q <= {26'd0, quant_status_value};
                    MATMUL_OFF_SCRATCH_MODE[7:0]:  rdata_q <= {30'd0, scratch_mode_q};
                    MATMUL_OFF_SCRATCH_ROLE[7:0]:  rdata_q <= {30'd0, scratch_role_q};
                    MATMUL_OFF_SCRATCH_ROWS[7:0]:  rdata_q <= {18'd0, scratch_rows_q};
                    MATMUL_OFF_SCRATCH_TOKENS[7:0]: rdata_q <= {29'd0, scratch_tokens_q};
                    MATMUL_OFF_SCRATCH_STATUS[7:0]: rdata_q <= {
                        18'd0, ffn_gate_ready,
                        scratch_section_done_q, scratch_section_active_q,
                        ffn_producer_done_q, ffn_producer_busy_q,
                        scratch_valid_q, any_scratch_error,
                        scratch_drain_done_q,
                        (scratch_drain_busy_q ||
                         (scratch_rd_owner_q == SCRATCH_RD_DRAIN)),
                        scratch_writer_done_q, scratch_wr_busy
                    };
                    MATMUL_OFF_SCRATCH_ERROR[7:0]: rdata_q <=
                        {23'd0, p3d_scratch_error_q, scratch_error_q};
                    MATMUL_OFF_MODEL_ROWS[7:0]:    rdata_q <= {18'd0, model_rows_q};
                    MATMUL_OFF_NORM_EPS[7:0]:      rdata_q <= norm_eps_q;
                    MATMUL_OFF_NORM_STATUS[7:0]:   rdata_q <= norm_status_value;
                    MATMUL_OFF_NORM_ERROR[7:0]:    rdata_q <= norm_error_value;
                    MATMUL_OFF_RESIDUAL_ERROR[7:0]: rdata_q <=
                        {25'd0, residual_error_detail_q};
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
    assign formal_p3d_active = p3d_active_q;
    assign formal_p3d_cleanup = p3d_cleanup_q;
    assign formal_p3d_kill = p3d_kill_q;
    assign formal_p3d_rd_owner = p3d_rd_owner_q;
    assign formal_q8_owner = q8_owner_q;
    assign formal_p3d_r_load_complete = p3d_r_load_complete_q;
    assign formal_p3d_norm_sealed = p3d_norm_sealed_q;
    assign formal_p3d_residual_started = p3d_residual_started_q;
    assign formal_p3d_begin_ok = p3d_section_begin_ok;
    assign formal_p3d_fault = p3d_fault_event;
    assign formal_p3d_clean_complete = p3d_clean_complete;
    assign formal_gamma_busy = rms_gamma_busy;
    assign formal_gamma_valid = rms_gamma_valid;
    assign formal_rms_rd_req = rms_rd_req_valid;
    assign formal_residual_rd_req = residual_rd_req_valid;
    assign formal_rms_r_wr_valid = rms_r_wr_valid;
    assign formal_residual_r_wr_valid = residual_r_wr_valid;
    assign formal_scratch_r_wr_valid = scratch_r_wr_valid;
    assign formal_kernel_output_valid = kernel_m_axis_tvalid;
    assign formal_residual_output_valid = residual_m_axis_tvalid;
    assign formal_r_valid = scratch_valid_q[0];
    assign formal_norm_error = norm_error_q;
    assign formal_norm_controller_error = norm_controller_error_q;
    assign formal_residual_error = residual_error_q;
    assign formal_shared_activation_idle = shared_activation_idle;
    assign formal_compute_acts_tvalid = compute_s_axis_acts_tvalid;
    assign formal_q8_ingress_start = q8_ingress_start;
    assign formal_kernel_start = kernel_start;
    assign formal_quant_status = quant_status_value;

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

            if (rms_gamma_busy) begin
                assert(!shared_activation_idle);
                assert(!scratch_launch_ok);
                assert(!scratch_section_begin_ok);
                assert(!scratch_drain_start_ok);
                assert(!qualified_kernel_start);
                assert(!kernel_start_q);
                assert(!scratch_tee_start_q);
                assert(!scratch_only_start_q);
                assert(!ffn_gate_start_q);
                assert(!ffn_down_start_q);
                assert(!q8_ingress_start);
                assert(!compute_s_axis_acts_tvalid);
                assert(!p3d_external_residual_ingress);
                assert(s_axis_acts_tready == rms_gamma_tready);
            end

            if (scratch_rd_owner_q == SCRATCH_RD_PAIRER)
                assert(!scratch_drain_rsp_valid);
            if (scratch_rd_owner_q == SCRATCH_RD_DRAIN)
                assert(!ffn_pairer_rd_rsp_valid);
            if (ffn_pairer_rd_rsp_valid)
                assert(scratch_rd_owner_q == SCRATCH_RD_PAIRER);
            if (scratch_drain_rsp_valid)
                assert(scratch_rd_owner_q == SCRATCH_RD_DRAIN);
            if (scratch_rd_quiescent) begin
                assert(!scratch_rd_rsp_valid);
                assert(scratch_rd_req_ready);
            end
            if (scratch_idle)
                assert(scratch_rd_quiescent);

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
            if (p3d_cleanup_q && (p3d_quant_status_q != 6'd0))
                assert(quant_status_value == p3d_quant_status_q);
            if (f_decode_past_valid &&
                $past(rst_n && p3d_q8_fault_now &&
                      !scratch_abort_strobe))
                assert(p3d_quant_status_q != 6'd0);
            if (f_decode_past_valid &&
                $past(rst_n && p3d_kill_q &&
                      (p3d_quant_status_q != 6'd0) &&
                      !scratch_abort_strobe))
                assert(p3d_quant_status_q == $past(p3d_quant_status_q));

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
                assert(scratch_consumer_done_q);
                if ($past(p3d_active_q)) begin
                    assert(p3d_cleanup_q);
                    assert(!scratch_section_done_q);
                end else begin
                    assert(scratch_section_done_q);
                end
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

            // P3d acceptance only snapshots controller state. The leaves launch
            // from that snapshot one cycle later, unless an intervening abort
            // quarantines the pending launch.
            if (f_decode_past_valid && $past(rst_n)) begin
                assert(p3d_leaf_start_q == $past(p3d_section_begin_ok));
                if ($past(p3d_section_begin_ok)) begin
                    assert(p3d_model_rows_q == $past(model_rows_q));
                    assert(p3d_tokens_q == $past(scratch_tokens_q));
                    assert(p3d_eps_q == $past(norm_eps_q));
                    assert(p3d_resident_q ==
                           $past(scratch_section_resident_strobe));
                    assert(ffn_tokens_q == $past(scratch_tokens_q));
                    assert(ffn_blocks_q == $past(scratch_rows_q[13:5]));
                end
            end
            if (p3d_section_begin_ok) begin
                assert(!p3d_leaf_start);
                assert(!q8_buffer_cfg_p3d);
            end
            assert(!(p3d_leaf_start && legacy_section_begin_ok));
            assert(!(p3d_leaf_start && ffn_gate_start_q));
            if (p3d_leaf_start) begin
                assert(p3d_active_q && !p3d_cleanup_q && !p3d_abort_now);
                assert(shared_activation_idle);
                assert(rms_cfg_ready && q8_buffer_cfg_ready);
                assert(q8_owner_q == Q8_OWNER_RMS);
                assert(!p3d_q8_fault_now);
                assert(!q8_fault_live && (quantizer_status == 6'd0));
                assert(q8_ingress_start && q8_ingress_internal_mode);
                assert(q8_ingress_blocks ==
                       {9'd0, p3d_model_rows_q[13:7]});
                assert(q8_ingress_cols == {13'd0, p3d_tokens_q});
                assert(q8_buffer_cfg_p3d && q8_buffer_cfg_valid);
                assert(q8_buffer_cfg_tokens == ffn_tokens_q);
                assert(q8_buffer_cfg_blocks == ffn_blocks_q);
            end
            if (q8_ingress_start) begin
                assert(!q8_activation_abort);
                assert(quantizer_status == 6'd0);
            end
            if (legacy_section_begin_ok) begin
                assert(q8_buffer_cfg_valid && !q8_buffer_cfg_p3d);
                assert(q8_buffer_cfg_tokens == scratch_tokens_q);
                assert(q8_buffer_cfg_blocks == scratch_rows_q[13:5]);
            end
            if (scratch_section_begin_ok && (model_rows_q == 14'd0)) begin
                assert(legacy_section_begin_ok);
                assert(!p3d_section_begin_ok);
                assert(!p3d_leaf_start_q && !p3d_leaf_start);
            end
            if (f_decode_past_valid &&
                $past(rst_n && legacy_section_begin_ok)) begin
                assert(!p3d_leaf_start_q);
                assert(!p3d_leaf_start);
            end
            if (p3d_leaf_start_q && p3d_abort_now) begin
                assert(!p3d_leaf_start);
                assert(!q8_buffer_cfg_p3d);
                assert(!q8_ingress_start);
            end

            if (p3d_abort_now && p3d_residual_output_selected) begin
                assert(!m_axis_tvalid);
                assert(!kernel_m_axis_tready);
                assert(!scratch_r_wr_valid);
            end

            // P3d scratch reads are untagged at both arbitration levels.  Each
            // accepted child remains the sole subowner until the physical
            // response drains, including across fault/abort cleanup.
            assert(!(rms_rd_req_valid && residual_rd_req_valid));
            assert(!(rms_r_wr_valid && residual_r_wr_valid));
            if (scratch_rd_owner_q == SCRATCH_RD_P3D)
                assert(p3d_rd_owner_q != P3D_RD_NONE);
            if (p3d_rd_owner_q != P3D_RD_NONE)
                assert(scratch_rd_owner_q == SCRATCH_RD_P3D);
            if (rms_rd_rsp_valid)
                assert((scratch_rd_owner_q == SCRATCH_RD_P3D) &&
                       (p3d_rd_owner_q == P3D_RD_RMS));
            if (residual_rd_rsp_valid)
                assert((scratch_rd_owner_q == SCRATCH_RD_P3D) &&
                       (p3d_rd_owner_q == P3D_RD_RESIDUAL));
            if (f_decode_past_valid &&
                $past(rst_n &&
                      (scratch_rd_owner_q == SCRATCH_RD_P3D) &&
                      !(scratch_rd_rsp_valid && scratch_rd_rsp_ready))) begin
                assert(scratch_rd_owner_q == SCRATCH_RD_P3D);
                assert(p3d_rd_owner_q == $past(p3d_rd_owner_q));
            end

            // Direct R writes are phase-qualified.  External RMSNorm is the
            // only pre-UP writer; residual addition is the only DOWN writer.
            if (scratch_r_wr_valid) begin
                assert(rms_write_phase ^ residual_write_phase);
                if (rms_write_phase) begin
                    assert(rms_r_wr_valid);
                    assert(!residual_r_wr_valid);
                end else begin
                    assert(residual_r_wr_valid);
                    assert(!rms_r_wr_valid);
                end
            end
            if (p3d_active_q && scratch_only_start)
                assert(p3d_r_load_complete_q);
            if (p3d_active_q && ffn_gate_start_q) begin
                assert(p3d_norm_sealed_q);
                assert(q8_owner_q == Q8_OWNER_NONE);
            end

            // The shared quantizer has one retained destination. RMS records
            // feed GEMM activation RAM directly; GATE records feed Q8 capture.
            if (q8_owner_q == Q8_OWNER_RMS) begin
                assert(!q8_buffer_capture_valid);
                assert(!swiglu_out_ready);
                assert(native_acts_tready == kernel_acts_tready);
            end
            if (q8_owner_q == Q8_OWNER_GATE) begin
                assert(!rms_scalar_ready);
                assert(native_acts_tready == q8_buffer_s_axis_tready);
            end

            // R is tentative for the whole section and becomes resident only
            // after the residual leaf has committed and handed off its final
            // output beat. Raw DOWN data is never visible at M_AXIS in P3d.
            if (p3d_active_q)
                assert(!scratch_valid_q[0]);
            if (p3d_cleanup_q)
                assert(!scratch_valid_q[0]);
            if (p3d_residual_output_selected) begin
                assert(!ddr_kernel_valid);
                assert(m_axis_tvalid ==
                       (residual_m_axis_tvalid && !p3d_kill_q &&
                        !p3d_fault_event));
                if (m_axis_tvalid) begin
                    assert(m_axis_tdata == residual_m_axis_tdata);
                    assert(m_axis_tkeep == residual_m_axis_tkeep);
                    assert(m_axis_tlast == residual_m_axis_tlast);
                end
            end
            if (p3d_fault_event && p3d_residual_output_selected) begin
                assert(!m_axis_tvalid);
                assert(!residual_m_axis_tready);
                assert(!kernel_m_axis_tready);
            end
            if (p3d_clean_complete) begin
                assert(!p3d_fault_event);
                assert(residual_done && !residual_error);
            end
            if (f_decode_past_valid &&
                $past(rst_n && p3d_clean_complete)) begin
                assert(!p3d_active_q);
                assert(scratch_valid_q[0]);
                assert(scratch_valid_rows_q[0] == $past(p3d_model_rows_q));
                assert(scratch_valid_tokens_q[0] == $past(p3d_tokens_q));
            end
            if (p3d_cleanup_retire)
                assert(p3d_cleanup_resources_idle && !m_axis_tvalid);

            // Every accepted failed section has a visible terminal attribution:
            // norm/controller before residual starts, residual thereafter.
            if (f_decode_past_valid &&
                $past(rst_n && p3d_fault_event &&
                      p3d_norm_sealed_q && !p3d_residual_started_q)) begin
                assert(norm_done_q);
                assert(norm_error_q);
                assert(norm_controller_error_q);
            end
            if (f_decode_past_valid &&
                $past(rst_n && p3d_fault_event &&
                      p3d_residual_started_q))
                assert(residual_error_q);
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
