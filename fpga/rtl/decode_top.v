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

    q8_ingress u_q8_ingress (
        .clk(clk),
        .rst_n(rst_n),
        .start(kernel_start),
        .raw_mode(raw_activation_mode),
        .internal_mode(internal_activation_mode),
        .num_q1_blocks(num_q1_blocks_q),
        .num_cols(num_cols_q),
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

    wire [63:0] kernel_acts_tdata = raw_activation_mode ? native_acts_tdata : s_axis_acts_tdata;
    wire kernel_acts_tvalid = raw_activation_mode ? native_acts_tvalid : s_axis_acts_tvalid;
    assign native_acts_tready = raw_activation_mode && kernel_acts_tready;
    assign s_axis_acts_tready = internal_activation_mode ? 1'b0 :
                                (raw_activation_mode ? raw_acts_tready : kernel_acts_tready);

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
    wire scratch_only_order_ok = ((scratch_role_q == 2'd2) &&
                                  !scratch_valid_q[2] && !scratch_valid_q[1]) ||
                                 ((scratch_role_q == 2'd1) &&
                                  scratch_valid_q[2] && !scratch_valid_q[1]);
    wire scratch_only_shape_ok = (scratch_mode_q == SCRATCH_MODE_ONLY) &&
                                 scratch_section_active_q && scratch_only_order_ok &&
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

    wire scratch_idle = !kernel_busy && !kernel_start_q &&
                        !scratch_tee_run_q && !scratch_only_run_q &&
                        !scratch_wr_busy && !scratch_drain_busy_q &&
                        !scratch_consumer_busy_q && !scratch_rd_rsp_valid &&
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
    wire scratch_ddr_preflight_ok = (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                    scratch_idle &&
                                    (internal_activation_mode ? scratch_consumer_shape_ok :
                                     !scratch_section_active_q);
    wire scratch_tee_preflight_ok = scratch_tee_shape_ok && scratch_wr_cfg_ready &&
                                    scratch_idle && !scratch_section_active_q;
    wire scratch_only_preflight_ok = scratch_only_shape_ok && scratch_wr_cfg_ready && scratch_idle;
    wire scratch_launch_ok = scratch_ddr_preflight_ok ||
                             scratch_tee_preflight_ok ||
                             scratch_only_preflight_ok;
    wire scratch_tee_start = scratch_tee_start_q;
    wire scratch_only_start = scratch_only_start_q;
    wire scratch_consumer_start = scratch_consumer_start_q;
    assign kernel_start = kernel_start_q;
    wire scratch_start_rejected = start_strobe && !scratch_launch_ok;
    wire scratch_section_begin_ok = scratch_section_begin_strobe &&
                                    (scratch_mode_q == SCRATCH_MODE_DDR) &&
                                    scratch_idle && !scratch_section_active_q;
    wire scratch_section_begin_bad = scratch_section_begin_strobe &&
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
        end else begin
            kernel_start_q <= start_strobe && scratch_launch_ok;
            scratch_tee_start_q <= start_strobe && scratch_tee_preflight_ok;
            scratch_only_start_q <= start_strobe && scratch_only_preflight_ok;
            scratch_consumer_start_q <= start_strobe &&
                                        scratch_ddr_preflight_ok &&
                                        internal_activation_mode;
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
                            (!scratch_tee_active || scratch_wr_tready);
    assign kernel_m_axis_tready = scratch_only_active ? scratch_wr_tready :
                                  (scratch_tee_active ?
                                   (m_axis_tready && scratch_wr_tready) :
                                   (scratch_drain_busy_q ? 1'b0 : m_axis_tready));

    wire scratch_writer_abort = scratch_abort_strobe ||
                                (scratch_writer_active &&
                                 (q8_activation_abort || activation_error ||
                                  (kernel_done && scratch_wr_busy)));

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
                                  scratch_drain_shape_ok && scratch_role_metadata_ok &&
                                  scratch_idle && !scratch_section_active_q;
    wire scratch_drain_start_bad = scratch_drain_start_strobe && !scratch_drain_start_ok;

    wire scratch_consumer_req_gate = scratch_consumer_busy_q &&
                                     (scratch_consumer_state_q == CONSUMER_REQ_GATE);
    wire scratch_consumer_req_up = scratch_consumer_busy_q &&
                                   (scratch_consumer_state_q == CONSUMER_REQ_UP);
    wire scratch_consumer_req_valid = scratch_consumer_req_gate || scratch_consumer_req_up;
    assign scratch_rd_req_valid = scratch_consumer_req_valid ||
                                  (scratch_drain_busy_q && !scratch_drain_have_group_q);
    assign scratch_rd_req_role  = scratch_consumer_req_gate ? 2'd1 :
                                  (scratch_consumer_req_up ? 2'd2 : scratch_drain_role_q);
    assign scratch_rd_req_token = scratch_consumer_req_valid ?
                                  scratch_consumer_token_q : scratch_drain_token_q;
    assign scratch_rd_req_group = scratch_consumer_req_valid ?
                                  scratch_consumer_group_q : scratch_drain_group_q;
    // Consume an orphaned response while idle (possible after abort), and consume
    // an errored response without exposing its zero payload on M_AXIS.
    wire scratch_consumer_wait_rsp = scratch_consumer_busy_q &&
                                     ((scratch_consumer_state_q == CONSUMER_WAIT_GATE) ||
                                      (scratch_consumer_state_q == CONSUMER_WAIT_UP));
    wire scratch_consumer_record_accounting_ok =
        (scratch_consumer_records_q < scratch_consumer_blocks_q) &&
        (scratch_consumer_records_q < scratch_consumer_total_blocks_q);
    assign scratch_rd_rsp_ready = scratch_consumer_wait_rsp ||
                                  ((!scratch_consumer_busy_q && !scratch_drain_busy_q) &&
                                   scratch_rd_rsp_valid) ||
                                  (scratch_drain_have_group_q &&
                                   (scratch_rd_rsp_error ||
                                    (scratch_drain_emit_valid_q &&
                                     (scratch_drain_bank_q == 2'd3) &&
                                     m_axis_tready)));

    assign swiglu_in_valid = scratch_consumer_busy_q &&
                             (scratch_consumer_state_q == CONSUMER_ISSUE);
    assign swiglu_in_gate = scratch_consumer_gate_q[
                             {scratch_consumer_lane_q, 5'b00000} +: 32];
    assign swiglu_in_up = scratch_consumer_up_q[
                           {scratch_consumer_lane_q, 5'b00000} +: 32];
    assign swiglu_in_last = (scratch_consumer_group_q[1:0] == 2'd3) &&
                            (scratch_consumer_lane_q == 3'd7);

    section_swiglu u_section_swiglu (
        .clk(clk),
        .rst_n(rst_n),
        .abort(scratch_abort_strobe || q8_activation_abort || scratch_error_q[5]),
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

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(MATMUL_COLS_MAX), .MAX_SUB_INDEX(512)) u_kernel (
        .clk(clk),
        .rst_n(rst_n),
        .start_kernel(kernel_start),
        .num_q1_blocks(num_q1_blocks_q),
        .num_rowblocks(num_rowblocks_q),
        .num_rows(num_rows_q),
        .num_cols(num_cols_q),
        .weight_fmt(weight_fmt_q),
        .act_mode(internal_activation_mode ? 2'd2 : act_mode_q),
        .act_epoch(act_epoch_q),
        .activation_abort(q8_activation_abort ||
                          (scratch_writer_active && scratch_abort_strobe) ||
                          (scratch_consumer_busy_q && scratch_abort_strobe) ||
                          (internal_activation_mode && |scratch_error_q[6:5])),
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
                if (internal_activation_mode && scratch_section_active_q) begin
                    scratch_section_active_q <= 1'b0;
                    scratch_section_done_q <= 1'b1;
                    scratch_valid_q <= 4'd0;
                end
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

                if (scratch_drain_have_group_q && scratch_rd_rsp_valid) begin
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
                    scratch_section_active_q <= 1'b0;
                    scratch_section_done_q   <= 1'b1;
                    scratch_valid_q          <= 4'd0;
                    scratch_error_q[2]       <= 1'b1;
                end
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
    always @(posedge clk) begin
        if (!rst_n || start_strobe) begin
            w_stall_q <= 32'd0; a_stall_q <= 32'd0; r_stall_q <= 32'd0;
            w_beats_q <= 32'd0; a_beats_q <= 32'd0; r_beats_q <= 32'd0;
        end else if (kernel_busy) begin
            if (weight_tvalid && weight_tready)           w_beats_q <= w_beats_q + 32'd1;
            if (s_axis_acts_tvalid && s_axis_acts_tready) a_beats_q <= a_beats_q + 32'd1;
            if (m_axis_tvalid && m_axis_tready)           r_beats_q <= r_beats_q + 32'd1;
            if (weight_tready && !weight_tvalid)           w_stall_q <= w_stall_q + 32'd1;
            if (s_axis_acts_tready && !s_axis_acts_tvalid) a_stall_q <= a_stall_q + 32'd1;
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
                        19'd0, scratch_section_done_q, scratch_section_active_q,
                        scratch_consumer_done_q, scratch_consumer_busy_q,
                        scratch_valid_q, |scratch_error_q,
                        scratch_drain_done_q, scratch_drain_busy_q,
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
    reg        f_decode_past_valid = 1'b0;
    reg [3:0]  f_reads_per_block = 4'd0;
    reg [5:0]  f_scalars_per_block = 6'd0;

    wire f_swiglu_input_fire = swiglu_in_valid && swiglu_in_ready;
    wire f_swiglu_block_last = f_swiglu_input_fire &&
                               (scratch_consumer_lane_q == 3'd7) &&
                               (scratch_consumer_group_q[1:0] == 2'd3);

    always @(posedge clk) begin
        f_decode_past_valid <= 1'b1;
        if (!rst_n || scratch_consumer_start || scratch_abort_strobe) begin
            f_reads_per_block <= 4'd0;
            f_scalars_per_block <= 6'd0;
        end else if (scratch_consumer_busy_q) begin
            if (scratch_rd_req_valid && scratch_rd_req_ready)
                f_reads_per_block <= f_reads_per_block + 1'b1;
            if (f_swiglu_input_fire) begin
                if (f_swiglu_block_last) begin
                    assert(f_reads_per_block == 4'd8);
                    assert(f_scalars_per_block == 6'd31);
                    f_reads_per_block <= 4'd0;
                    f_scalars_per_block <= 6'd0;
                end else begin
                    f_scalars_per_block <= f_scalars_per_block + 1'b1;
                end
            end
            if (q8_internal_record_done)
                assert(scratch_consumer_records_q <
                       scratch_consumer_blocks_q);
        end

        if (rst_n) begin
            assert(scratch_consumer_state_q <= CONSUMER_DRAIN);
            assert(scratch_consumer_records_q <= scratch_consumer_blocks_q);
            assert(scratch_consumer_blocks_q <= scratch_consumer_total_blocks_q);
            assert(scratch_consumer_blocks_q <=
                   scratch_consumer_records_q + 16'd3);
            assert(scratch_consumer_total_blocks_q <= 16'd1536);
            assert(f_reads_per_block <= 4'd8);
            assert(f_scalars_per_block <= 6'd31);

            if (scratch_only_run_q) begin
                assert(scratch_section_active_q);
                assert(!m_axis_tvalid);
            end
            if (internal_activation_mode)
                assert(!s_axis_acts_tready);

            if (scratch_section_active_q && scratch_valid_q[1])
                assert(scratch_valid_q[2]);

            if (scratch_consumer_busy_q) begin
                assert(scratch_section_active_q);
                assert(internal_activation_mode);
                assert(scratch_valid_q[1] && scratch_valid_q[2]);
                assert(!scratch_drain_busy_q && !scratch_writer_active);
                assert(scratch_consumer_group_q < scratch_consumer_groups_q);
                assert(scratch_consumer_token_q < scratch_tokens_q);
                assert(scratch_consumer_lane_q <= 3'd7);
            end

            if (scratch_consumer_req_gate)
                assert(scratch_rd_req_valid && scratch_rd_req_role == 2'd1);
            if (scratch_consumer_req_up)
                assert(scratch_rd_req_valid && scratch_rd_req_role == 2'd2);
            if (scratch_consumer_state_q == CONSUMER_DRAIN) begin
                assert(!scratch_rd_req_valid);
                assert(!swiglu_in_valid);
            end

            if (scratch_consumer_start) begin
                assert(scratch_consumer_shape_ok);
                assert(({7'd0, scratch_rows_q[13:5]} *
                        {13'd0, scratch_tokens_q}) <= 16'd1536);
            end

            if (f_decode_past_valid && $past(rst_n && scratch_consumer_start)) begin
                assert(scratch_consumer_busy_q);
                assert(scratch_consumer_state_q == CONSUMER_REQ_GATE);
                assert(scratch_consumer_total_blocks_q ==
                       ({7'd0, $past(scratch_rows_q[13:5])} *
                        {13'd0, $past(scratch_tokens_q)}));
            end

            if (f_decode_past_valid &&
                $past(rst_n && scratch_section_begin_ok)) begin
                assert(scratch_section_active_q);
                assert(!scratch_section_done_q);
                assert(scratch_valid_q == 4'd0);
                assert(scratch_error_q == 7'd0);
            end

            if (f_decode_past_valid &&
                $past(rst_n && scratch_abort_strobe &&
                      (scratch_consumer_busy_q || scratch_section_active_q))) begin
                assert(!scratch_consumer_busy_q && !scratch_section_active_q);
                assert(scratch_consumer_done_q && scratch_section_done_q);
                assert(scratch_valid_q == 4'd0);
                assert(scratch_error_q[2]);
            end

            if (f_decode_past_valid &&
                $past(rst_n && kernel_done && internal_activation_mode &&
                      scratch_section_active_q)) begin
                assert(!scratch_section_active_q && scratch_section_done_q);
                assert(scratch_valid_q == 4'd0);
            end

            if (f_decode_past_valid &&
                $past(rst_n && internal_activation_mode && q8_activation_abort)) begin
                assert(!scratch_consumer_busy_q && scratch_consumer_done_q);
                assert(scratch_valid_q == 4'd0);
                assert(scratch_error_q[6]);
            end
        end

        cover(rst_n && scratch_section_active_q &&
              scratch_valid_q[2] && !scratch_valid_q[1]);
        cover(rst_n && scratch_section_active_q &&
              scratch_valid_q[2] && scratch_valid_q[1]);
        cover(rst_n && scratch_consumer_busy_q &&
              scratch_consumer_state_q == CONSUMER_DRAIN);
        cover(rst_n && scratch_consumer_busy_q &&
              (scratch_consumer_blocks_q >=
               scratch_consumer_records_q + 16'd2));
        cover(rst_n && scratch_section_done_q && !scratch_section_active_q &&
              scratch_consumer_done_q && scratch_valid_q == 4'd0);
        cover(rst_n && scratch_error_q[2] && scratch_section_done_q &&
              !scratch_section_active_q && !scratch_consumer_busy_q &&
              scratch_valid_q == 4'd0);
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
