`default_nettype none

// Integrated v17 P3d driver.  Numerical leaves are abstracted in
// decode_ffn_stubs.v; the real decode_top controller, shared Q8 ingress,
// scratch arbitration, Q8 buffer, phase machine, and output mux remain intact.
module decode_p3d_formal(input wire clk);
    reg rst_n = 1'b0;

    localparam [7:0] REG_CTRL = 8'h08;
    localparam [7:0] REG_NUM_Q1 = 8'h10;
    localparam [7:0] REG_NUM_RB = 8'h14;
    localparam [7:0] REG_NUM_COLS = 8'h38;
    localparam [7:0] REG_WEIGHT_FMT = 8'h44;
    localparam [7:0] REG_NUM_ROWS = 8'h48;
    localparam [7:0] REG_ACT_MODE = 8'h4c;
    localparam [7:0] REG_ACT_EPOCH = 8'h50;
    localparam [7:0] REG_SCRATCH_MODE = 8'h68;
    localparam [7:0] REG_SCRATCH_ROLE = 8'h6c;
    localparam [7:0] REG_SCRATCH_ROWS = 8'h70;
    localparam [7:0] REG_SCRATCH_TOKENS = 8'h74;
    localparam [7:0] REG_SCRATCH_CTRL = 8'h78;
    localparam [7:0] REG_MODEL_ROWS = 8'h84;
    localparam [7:0] REG_NORM_EPS = 8'h88;
    localparam [7:0] REG_NORM_CTRL = 8'h8c;

    localparam [5:0] CMD_MODEL = 6'd0;
    localparam [5:0] CMD_EPS = 6'd1;
    localparam [5:0] CMD_GAMMA = 6'd2;
    localparam [5:0] WAIT_GAMMA = 6'd3;
    localparam [5:0] CMD_MODE_DDR = 6'd4;
    localparam [5:0] CMD_ROWS = 6'd5;
    localparam [5:0] CMD_TOKENS = 6'd6;
    localparam [5:0] CMD_BEGIN = 6'd7;
    localparam [5:0] WAIT_R = 6'd8;
    localparam [5:0] CMD_MODE_ONLY = 6'd9;
    localparam [5:0] CMD_ROLE_UP = 6'd10;
    localparam [5:0] CMD_Q1 = 6'd11;
    localparam [5:0] CMD_RB = 6'd12;
    localparam [5:0] CMD_NUM_ROWS = 6'd13;
    localparam [5:0] CMD_COLS = 6'd14;
    localparam [5:0] CMD_FMT = 6'd15;
    localparam [5:0] CMD_ACT_UP = 6'd16;
    localparam [5:0] CMD_EPOCH_UP = 6'd17;
    localparam [5:0] CMD_START_UP = 6'd18;
    localparam [5:0] WAIT_GATE = 6'd19;
    localparam [5:0] CMD_ROLE_GATE = 6'd20;
    localparam [5:0] CMD_ACT_GATE = 6'd21;
    localparam [5:0] CMD_START_GATE = 6'd22;
    localparam [5:0] WAIT_DOWN = 6'd23;
    localparam [5:0] CMD_DOWN_DDR = 6'd24;
    localparam [5:0] CMD_ACT_DOWN = 6'd25;
    localparam [5:0] CMD_EPOCH_DOWN = 6'd26;
    localparam [5:0] CMD_START_DOWN = 6'd27;
    localparam [5:0] WAIT_DONE = 6'd28;
    localparam [5:0] WAIT_ABORT_OWNER = 6'd29;
    localparam [5:0] CMD_ABORT = 6'd30;
    localparam [5:0] WAIT_ABORT_DONE = 6'd31;

    reg [5:0] command_q;
    reg [1:0] slot_phase_q;
    reg [7:0] write_addr;
    reg [31:0] write_data;
    reg write_enable;

    always @* begin
        write_addr = 8'd0;
        write_data = 32'd0;
        write_enable = 1'b1;
        case (command_q)
            CMD_MODEL: begin write_addr = REG_MODEL_ROWS; write_data = 128; end
            CMD_EPS: begin write_addr = REG_NORM_EPS;
                           write_data = 32'h3586_37bd; end
            CMD_GAMMA: begin write_addr = REG_NORM_CTRL; write_data = 1; end
            CMD_MODE_DDR: begin write_addr = REG_SCRATCH_MODE;
                                write_data = 0; end
            CMD_ROWS: begin write_addr = REG_SCRATCH_ROWS; write_data = 128; end
            CMD_TOKENS: begin write_addr = REG_SCRATCH_TOKENS;
                              write_data = 1; end
            CMD_BEGIN: begin write_addr = REG_SCRATCH_CTRL; write_data = 4; end
            CMD_MODE_ONLY: begin write_addr = REG_SCRATCH_MODE;
                                 write_data = 3; end
            CMD_ROLE_UP: begin write_addr = REG_SCRATCH_ROLE;
                               write_data = 2; end
            CMD_Q1: begin write_addr = REG_NUM_Q1; write_data = 1; end
            CMD_RB: begin write_addr = REG_NUM_RB; write_data = 8; end
            CMD_NUM_ROWS: begin write_addr = REG_NUM_ROWS; write_data = 128; end
            CMD_COLS: begin write_addr = REG_NUM_COLS; write_data = 1; end
            CMD_FMT: begin write_addr = REG_WEIGHT_FMT; write_data = 1; end
            CMD_ACT_UP: begin write_addr = REG_ACT_MODE; write_data = 2; end
            CMD_EPOCH_UP: begin write_addr = REG_ACT_EPOCH; write_data = 1; end
            CMD_START_UP: begin write_addr = REG_CTRL; write_data = 1; end
            CMD_ROLE_GATE: begin write_addr = REG_SCRATCH_ROLE;
                                 write_data = 1; end
            CMD_ACT_GATE: begin write_addr = REG_ACT_MODE; write_data = 1; end
            CMD_START_GATE: begin write_addr = REG_CTRL; write_data = 1; end
            CMD_DOWN_DDR: begin write_addr = REG_SCRATCH_MODE;
                                write_data = 0; end
            CMD_ACT_DOWN: begin write_addr = REG_ACT_MODE; write_data = 3; end
            CMD_EPOCH_DOWN: begin write_addr = REG_ACT_EPOCH;
                                  write_data = 2; end
            CMD_START_DOWN: begin write_addr = REG_CTRL; write_data = 1; end
            CMD_ABORT: begin write_addr = REG_SCRATCH_CTRL; write_data = 2; end
            default: write_enable = 1'b0;
        endcase
    end

    wire axi_valid = rst_n && write_enable && (slot_phase_q < 2);
    wire [1:0] bresp;
    wire bvalid;
    wire awready, wready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire arready, rvalid;
    wire [127:0] weight_word = 128'd0;
    wire w0_ready, w1_ready, w2_ready, w3_ready;
    wire acts_ready;
    wire [63:0] result_data;
    wire [7:0] result_keep;
    wire result_valid, result_last;

    wire [2:0] formal_ffn_phase;
    wire formal_ffn_gate_ready;
    wire [1:0] formal_scratch_rd_owner;
    wire formal_scratch_rd_rsp_valid, formal_scratch_rd_rsp_ready;
    wire formal_section_active, formal_section_done;
    wire formal_abort_cleanup;
    wire formal_producer_done;
    wire formal_up_start, formal_gate_start, formal_down_start;
    wire formal_p3d_active, formal_p3d_cleanup, formal_p3d_kill;
    wire [1:0] formal_p3d_rd_owner, formal_q8_owner;
    wire formal_p3d_r_load_complete, formal_p3d_norm_sealed;
    wire formal_p3d_residual_started, formal_p3d_begin_ok;
    wire formal_p3d_fault, formal_p3d_clean_complete;
    wire formal_gamma_busy, formal_gamma_valid;
    wire formal_rms_rd_req, formal_residual_rd_req;
    wire formal_rms_r_wr_valid, formal_residual_r_wr_valid;
    wire formal_scratch_r_wr_valid;
    wire formal_kernel_output_valid, formal_residual_output_valid;
    wire formal_r_valid;
    wire formal_norm_error, formal_norm_controller_error;
    wire formal_residual_error;
    wire formal_shared_activation_idle, formal_compute_acts_tvalid;
    wire formal_q8_ingress_start, formal_kernel_start;
    wire [5:0] formal_quant_status;

    reg [6:0] gamma_beat_q;
    reg [6:0] residual_beat_q;
    wire acts_last = formal_gamma_busy ? (gamma_beat_q == 7'd63) :
                     (formal_p3d_active &&
                      !formal_p3d_r_load_complete &&
                      (residual_beat_q == 7'd63));

    decode_top dut (
        .s_axi_aclk(clk), .s_axi_aresetn(rst_n),
        .s_axi_awaddr(write_addr), .s_axi_awprot(3'd0),
        .s_axi_awvalid(axi_valid), .s_axi_awready(awready),
        .s_axi_wdata(write_data), .s_axi_wstrb(4'hf),
        .s_axi_wvalid(axi_valid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(1'b1),
        .s_axi_araddr(8'd0), .s_axi_arprot(3'd0),
        .s_axi_arvalid(1'b0), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(1'b1),
        .s_axis_w0_tdata(weight_word), .s_axis_w0_tkeep(16'hffff),
        .s_axis_w0_tvalid(1'b1), .s_axis_w0_tready(w0_ready),
        .s_axis_w0_tlast(1'b0),
        .s_axis_w1_tdata(weight_word), .s_axis_w1_tkeep(16'hffff),
        .s_axis_w1_tvalid(1'b1), .s_axis_w1_tready(w1_ready),
        .s_axis_w1_tlast(1'b0),
        .s_axis_w2_tdata(weight_word), .s_axis_w2_tkeep(16'hffff),
        .s_axis_w2_tvalid(1'b1), .s_axis_w2_tready(w2_ready),
        .s_axis_w2_tlast(1'b0),
        .s_axis_w3_tdata(weight_word), .s_axis_w3_tkeep(16'hffff),
        .s_axis_w3_tvalid(1'b1), .s_axis_w3_tready(w3_ready),
        .s_axis_w3_tlast(1'b0),
        .s_axis_acts_tdata(64'd0), .s_axis_acts_tkeep(8'hff),
        .s_axis_acts_tvalid(1'b1), .s_axis_acts_tready(acts_ready),
        .s_axis_acts_tlast(acts_last),
        .m_axis_tdata(result_data), .m_axis_tkeep(result_keep),
        .m_axis_tvalid(result_valid), .m_axis_tready(1'b1),
        .m_axis_tlast(result_last),
        .formal_ffn_phase(formal_ffn_phase),
        .formal_ffn_gate_ready(formal_ffn_gate_ready),
        .formal_scratch_rd_owner(formal_scratch_rd_owner),
        .formal_scratch_rd_rsp_valid(formal_scratch_rd_rsp_valid),
        .formal_scratch_rd_rsp_ready(formal_scratch_rd_rsp_ready),
        .formal_section_active(formal_section_active),
        .formal_section_done(formal_section_done),
        .formal_abort_cleanup(formal_abort_cleanup),
        .formal_up_start(formal_up_start),
        .formal_gate_start(formal_gate_start),
        .formal_down_start(formal_down_start),
        .formal_producer_done(formal_producer_done),
        .formal_p3d_active(formal_p3d_active),
        .formal_p3d_cleanup(formal_p3d_cleanup),
        .formal_p3d_kill(formal_p3d_kill),
        .formal_p3d_rd_owner(formal_p3d_rd_owner),
        .formal_q8_owner(formal_q8_owner),
        .formal_p3d_r_load_complete(formal_p3d_r_load_complete),
        .formal_p3d_norm_sealed(formal_p3d_norm_sealed),
        .formal_p3d_residual_started(formal_p3d_residual_started),
        .formal_p3d_begin_ok(formal_p3d_begin_ok),
        .formal_p3d_fault(formal_p3d_fault),
        .formal_p3d_clean_complete(formal_p3d_clean_complete),
        .formal_gamma_busy(formal_gamma_busy),
        .formal_gamma_valid(formal_gamma_valid),
        .formal_rms_rd_req(formal_rms_rd_req),
        .formal_residual_rd_req(formal_residual_rd_req),
        .formal_rms_r_wr_valid(formal_rms_r_wr_valid),
        .formal_residual_r_wr_valid(formal_residual_r_wr_valid),
        .formal_scratch_r_wr_valid(formal_scratch_r_wr_valid),
        .formal_kernel_output_valid(formal_kernel_output_valid),
        .formal_residual_output_valid(formal_residual_output_valid),
        .formal_r_valid(formal_r_valid),
        .formal_norm_error(formal_norm_error),
        .formal_norm_controller_error(formal_norm_controller_error),
        .formal_residual_error(formal_residual_error),
        .formal_shared_activation_idle(formal_shared_activation_idle),
        .formal_compute_acts_tvalid(formal_compute_acts_tvalid),
        .formal_q8_ingress_start(formal_q8_ingress_start),
        .formal_kernel_start(formal_kernel_start),
        .formal_quant_status(formal_quant_status)
    );

    always @(posedge clk) begin
        rst_n <= 1'b1;
        if (!rst_n) begin
            command_q <= CMD_MODEL;
            slot_phase_q <= 2'd0;
            gamma_beat_q <= 7'd0;
            residual_beat_q <= 7'd0;
        end else begin
            if (!formal_gamma_busy)
                gamma_beat_q <= 7'd0;
            else if (acts_ready)
                gamma_beat_q <= gamma_beat_q + 1'b1;
            if (formal_p3d_begin_ok)
                residual_beat_q <= 7'd0;
            else if (formal_p3d_active &&
                     !formal_p3d_r_load_complete && acts_ready)
                residual_beat_q <= residual_beat_q + 1'b1;

            case (command_q)
                WAIT_GAMMA: begin
                    slot_phase_q <= 2'd0;
                    if (formal_gamma_valid)
                        command_q <= CMD_MODE_DDR;
                end
                WAIT_R: begin
                    slot_phase_q <= 2'd0;
                    if (formal_p3d_r_load_complete) begin
`ifdef FORMAL_P3D_ABORT
                        command_q <= WAIT_ABORT_OWNER;
`else
                        command_q <= CMD_MODE_ONLY;
`endif
                    end
                end
                WAIT_GATE: begin
                    slot_phase_q <= 2'd0;
                    if (formal_ffn_gate_ready && formal_p3d_norm_sealed)
                        command_q <= CMD_ROLE_GATE;
                end
                WAIT_DOWN: begin
                    slot_phase_q <= 2'd0;
                    if ((formal_ffn_phase == 3'd6) && formal_producer_done)
                        command_q <= CMD_DOWN_DDR;
                end
                WAIT_DONE: begin
                    slot_phase_q <= 2'd0;
                    command_q <= WAIT_DONE;
                end
                WAIT_ABORT_OWNER: begin
                    slot_phase_q <= 2'd0;
                    if ((formal_scratch_rd_owner == 2'd3) &&
                        (formal_p3d_rd_owner == 2'd1))
                        command_q <= CMD_ABORT;
                end
                WAIT_ABORT_DONE: begin
                    slot_phase_q <= 2'd0;
                    command_q <= WAIT_ABORT_DONE;
                end
                default: begin
                    if (slot_phase_q == 2'd3) begin
                        slot_phase_q <= 2'd0;
                        if (command_q == CMD_START_UP)
                            command_q <= WAIT_GATE;
                        else if (command_q == CMD_START_GATE)
                            command_q <= WAIT_DOWN;
                        else if (command_q == CMD_START_DOWN)
                            command_q <= WAIT_DONE;
                        else if (command_q == CMD_ABORT)
                            command_q <= WAIT_ABORT_DONE;
                        else
                            command_q <= command_q + 1'b1;
                    end else begin
                        slot_phase_q <= slot_phase_q + 1'b1;
                    end
                end
            endcase
        end
    end

    reg f_past_valid = 1'b0;
    reg saw_begin_q;
    reg saw_r_barrier_q;
    reg saw_rms_owner_q;
    reg saw_norm_seal_q;
    reg saw_gate_q8_q;
    reg saw_residual_start_q;
    reg saw_residual_output_q;
    reg saw_abort_cleanup_owner_q;
    reg saw_gamma_exclusion_q;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!rst_n) begin
            saw_begin_q <= 1'b0;
            saw_r_barrier_q <= 1'b0;
            saw_rms_owner_q <= 1'b0;
            saw_norm_seal_q <= 1'b0;
            saw_gate_q8_q <= 1'b0;
            saw_residual_start_q <= 1'b0;
            saw_residual_output_q <= 1'b0;
            saw_abort_cleanup_owner_q <= 1'b0;
            saw_gamma_exclusion_q <= 1'b0;
        end else begin
            if (formal_p3d_begin_ok)
                saw_begin_q <= 1'b1;
            if (formal_p3d_r_load_complete)
                saw_r_barrier_q <= 1'b1;
            if ((formal_scratch_rd_owner == 2'd3) &&
                (formal_p3d_rd_owner == 2'd1))
                saw_rms_owner_q <= 1'b1;
            if (formal_p3d_norm_sealed)
                saw_norm_seal_q <= 1'b1;
            if (formal_q8_owner == 2'd2)
                saw_gate_q8_q <= 1'b1;
            if (formal_p3d_residual_started)
                saw_residual_start_q <= 1'b1;
            if (formal_p3d_residual_started && result_valid &&
                formal_residual_output_valid)
                saw_residual_output_q <= 1'b1;
            if (formal_p3d_cleanup &&
                (formal_scratch_rd_owner == 2'd3) &&
                (formal_p3d_rd_owner == 2'd1))
                saw_abort_cleanup_owner_q <= 1'b1;
            if (formal_gamma_busy && acts_ready &&
                !formal_shared_activation_idle &&
                !formal_compute_acts_tvalid &&
                !formal_q8_ingress_start && !formal_kernel_start)
                saw_gamma_exclusion_q <= 1'b1;

            assert(!(formal_rms_rd_req && formal_residual_rd_req));
            assert(!(formal_rms_r_wr_valid && formal_residual_r_wr_valid));
            if (formal_gamma_busy) begin
                assert(!formal_shared_activation_idle);
                assert(!formal_compute_acts_tvalid);
                assert(!formal_q8_ingress_start);
                assert(!formal_kernel_start);
            end
            if (formal_up_start && formal_p3d_active)
                assert(formal_p3d_r_load_complete);
            if (formal_gate_start && formal_p3d_active)
                assert(formal_p3d_norm_sealed);
            if (formal_p3d_residual_started && result_valid)
                assert(formal_residual_output_valid);
            if (formal_p3d_cleanup)
                assert(!formal_r_valid);
            if (f_past_valid &&
                $past(rst_n && formal_p3d_fault &&
                      formal_p3d_norm_sealed &&
                      !formal_p3d_residual_started)) begin
                assert(formal_norm_error);
                assert(formal_norm_controller_error);
            end
            if (f_past_valid &&
                $past(rst_n && formal_p3d_fault &&
                      formal_p3d_residual_started))
                assert(formal_residual_error);
        end

`ifdef FORMAL_P3D_ABORT
        cover(rst_n && saw_begin_q && saw_r_barrier_q && saw_rms_owner_q &&
              saw_abort_cleanup_owner_q && saw_gamma_exclusion_q &&
              !formal_p3d_active &&
              !formal_p3d_cleanup &&
              (formal_scratch_rd_owner == 2'd0));
`else
        cover(rst_n && saw_begin_q && saw_r_barrier_q && saw_rms_owner_q &&
              saw_norm_seal_q && saw_gate_q8_q && saw_residual_start_q &&
              saw_gamma_exclusion_q &&
              saw_residual_output_q && formal_section_done &&
              !formal_section_active && formal_r_valid);
`endif
    end

    wire _unused = &{1'b0, bresp, bvalid, awready, wready, rdata, rresp,
                     arready, rvalid, w0_ready, w1_ready, w2_ready, w3_ready,
                     result_data, result_keep, result_last, formal_p3d_kill,
                     formal_abort_cleanup, formal_p3d_clean_complete,
                     formal_scratch_r_wr_valid,
                     formal_kernel_output_valid, formal_quant_status,
                     command_q, slot_phase_q};
endmodule

`default_nettype wire
