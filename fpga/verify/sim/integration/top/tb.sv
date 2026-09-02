`timescale 1ns/1ps
`default_nettype none

module penzai_top_tb;
    localparam integer ADDR_W = 40;

    logic s_axi_aclk = 1'b0;
    logic s_axi_aresetn = 1'b0;
    always #2 s_axi_aclk = ~s_axi_aclk;

    logic [11:0] s_axi_awaddr = 12'd0;
    logic [2:0] s_axi_awprot = 3'd0;
    logic s_axi_awvalid = 1'b0;
    wire s_axi_awready;
    logic [31:0] s_axi_wdata = 32'd0;
    logic [3:0] s_axi_wstrb = 4'd0;
    logic s_axi_wvalid = 1'b0;
    wire s_axi_wready;
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    logic s_axi_bready = 1'b1;
    logic [11:0] s_axi_araddr = 12'd0;
    logic [2:0] s_axi_arprot = 3'd0;
    logic s_axi_arvalid = 1'b0;
    wire s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0] s_axi_rresp;
    wire s_axi_rvalid;
    logic s_axi_rready = 1'b1;

    wire [ADDR_W-1:0] m_axi_w0_araddr;
    wire [7:0] m_axi_w0_arlen;
    wire [2:0] m_axi_w0_arsize;
    wire [1:0] m_axi_w0_arburst;
    wire m_axi_w0_arvalid;
    logic m_axi_w0_arready = 1'b0;
    logic [127:0] m_axi_w0_rdata = 128'd0;
    logic [1:0] m_axi_w0_rresp = 2'd0;
    logic m_axi_w0_rlast = 1'b0;
    logic m_axi_w0_rvalid = 1'b0;
    wire m_axi_w0_rready;
    wire [ADDR_W-1:0] m_axi_w1_araddr;
    wire [7:0] m_axi_w1_arlen;
    wire [2:0] m_axi_w1_arsize;
    wire [1:0] m_axi_w1_arburst;
    wire m_axi_w1_arvalid;
    logic m_axi_w1_arready = 1'b0;
    logic [127:0] m_axi_w1_rdata = 128'd0;
    logic [1:0] m_axi_w1_rresp = 2'd0;
    logic m_axi_w1_rlast = 1'b0;
    logic m_axi_w1_rvalid = 1'b0;
    wire m_axi_w1_rready;
    wire [ADDR_W-1:0] m_axi_w2_araddr;
    wire [7:0] m_axi_w2_arlen;
    wire [2:0] m_axi_w2_arsize;
    wire [1:0] m_axi_w2_arburst;
    wire m_axi_w2_arvalid;
    logic m_axi_w2_arready = 1'b0;
    logic [127:0] m_axi_w2_rdata = 128'd0;
    logic [1:0] m_axi_w2_rresp = 2'd0;
    logic m_axi_w2_rlast = 1'b0;
    logic m_axi_w2_rvalid = 1'b0;
    wire m_axi_w2_rready;
    wire [ADDR_W-1:0] m_axi_w3_araddr;
    wire [7:0] m_axi_w3_arlen;
    wire [2:0] m_axi_w3_arsize;
    wire [1:0] m_axi_w3_arburst;
    wire m_axi_w3_arvalid;
    logic m_axi_w3_arready = 1'b0;
    logic [127:0] m_axi_w3_rdata = 128'd0;
    logic [1:0] m_axi_w3_rresp = 2'd0;
    logic m_axi_w3_rlast = 1'b0;
    logic m_axi_w3_rvalid = 1'b0;
    wire m_axi_w3_rready;

    wire [ADDR_W-1:0] m_axi_hist_k_araddr;
    wire [7:0] m_axi_hist_k_arlen;
    wire [2:0] m_axi_hist_k_arsize;
    wire [1:0] m_axi_hist_k_arburst;
    wire m_axi_hist_k_arvalid;
    logic m_axi_hist_k_arready = 1'b0;
    logic [127:0] m_axi_hist_k_rdata = 128'd0;
    logic [1:0] m_axi_hist_k_rresp = 2'd0;
    logic m_axi_hist_k_rlast = 1'b0;
    logic m_axi_hist_k_rvalid = 1'b0;
    wire m_axi_hist_k_rready;
    wire [ADDR_W-1:0] m_axi_hist_v_araddr;
    wire [7:0] m_axi_hist_v_arlen;
    wire [2:0] m_axi_hist_v_arsize;
    wire [1:0] m_axi_hist_v_arburst;
    wire m_axi_hist_v_arvalid;
    logic m_axi_hist_v_arready = 1'b0;
    logic [127:0] m_axi_hist_v_rdata = 128'd0;
    logic [1:0] m_axi_hist_v_rresp = 2'd0;
    logic m_axi_hist_v_rlast = 1'b0;
    logic m_axi_hist_v_rvalid = 1'b0;
    wire m_axi_hist_v_rready;

    wire [ADDR_W-1:0] m_axi_kv_awaddr;
    wire [7:0] m_axi_kv_awlen;
    wire [2:0] m_axi_kv_awsize;
    wire [1:0] m_axi_kv_awburst;
    wire m_axi_kv_awvalid;
    logic m_axi_kv_awready = 1'b0;
    wire [127:0] m_axi_kv_wdata;
    wire [15:0] m_axi_kv_wstrb;
    wire m_axi_kv_wlast;
    wire m_axi_kv_wvalid;
    logic m_axi_kv_wready = 1'b0;
    logic [1:0] m_axi_kv_bresp = 2'd0;
    logic m_axi_kv_bvalid = 1'b0;
    wire m_axi_kv_bready;

    penzai_top #(
        .ADDR_W(ADDR_W),
        .CLK_HZ(32'd300_000_000)
    ) dut (.*);

    task automatic send_aw(input logic [11:0] address);
        begin
            @(negedge s_axi_aclk);
            s_axi_awaddr = address;
            s_axi_awvalid = 1'b1;
            while (!s_axi_awready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axi_awvalid = 1'b0;
        end
    endtask

    task automatic send_w(input logic [31:0] data,
                          input logic [3:0] strobe);
        begin
            @(negedge s_axi_aclk);
            s_axi_wdata = data;
            s_axi_wstrb = strobe;
            s_axi_wvalid = 1'b1;
            while (!s_axi_wready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axi_wvalid = 1'b0;
        end
    endtask

    task automatic write_reg(input logic [11:0] address,
                             input logic [31:0] data,
                             input logic [3:0] strobe,
                             input integer order);
        begin
            case (order)
                0: fork
                    send_aw(address);
                    send_w(data, strobe);
                join
                1: begin
                    send_aw(address);
                    repeat (2) @(negedge s_axi_aclk);
                    send_w(data, strobe);
                end
                default: begin
                    send_w(data, strobe);
                    repeat (2) @(negedge s_axi_aclk);
                    send_aw(address);
                end
            endcase
            while (!s_axi_bvalid) @(negedge s_axi_aclk);
            assert(s_axi_bresp == 2'b00);
            @(negedge s_axi_aclk);
        end
    endtask

    task automatic read_reg(input logic [11:0] address,
                            output logic [31:0] data);
        begin
            @(negedge s_axi_aclk);
            s_axi_araddr = address;
            s_axi_arvalid = 1'b1;
            while (!s_axi_arready) @(negedge s_axi_aclk);
            @(negedge s_axi_aclk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) @(negedge s_axi_aclk);
            assert(s_axi_rresp == 2'b00);
            data = s_axi_rdata;
            @(negedge s_axi_aclk);
        end
    endtask

    task automatic wait_status_bit(input integer bit_index);
        logic [31:0] value;
        integer polls;
        begin
            value = 32'd0;
            polls = 0;
            while (!value[bit_index] && (polls < 100)) begin
                read_reg(12'h014, value);
                polls = polls + 1;
            end
            assert(value[bit_index])
                else $fatal(1, "status bit %0d timeout", bit_index);
        end
    endtask

    logic [31:0] value;
    logic [31:0] held_rdata;
    logic [31:0] completed_cycles;
    logic [31:0] metric_control_cycles;
    logic [31:0] metric_down_cycles;
    integer held_clear_cycles;

    initial begin
        repeat (5) @(posedge s_axi_aclk);
        s_axi_aresetn = 1'b1;
        repeat (3) @(posedge s_axi_aclk);

        read_reg(12'h000, value);
        assert(value == 32'hb05a_4000);
        read_reg(12'h004, value);
        assert(value == 32'h0001_0007);
        read_reg(12'h008, value);
        assert(value == 32'h2fc1_4a79);
        read_reg(12'h00c, value);
        assert(value == 32'hc255_c7a5);
        read_reg(12'h020, value);
        assert(value == 32'd300_000_000);
        read_reg(12'h038, value);
        assert(value == 32'd7);
        read_reg(12'h160, value);
        assert(value == 32'h0001_0000);
        read_reg(12'h164, value);
        assert(value == 32'h0028_0b1d);
        read_reg(12'h168, value);
        assert(value == 32'h0000_0008);

        // AW and W are independently accepted; byte strobes merge in place.
        write_reg(12'h040, 32'h1122_3344, 4'hf, 1);
        write_reg(12'h040, 32'haabb_ccdd, 4'b0101, 2);
        read_reg(12'h040, value);
        assert(value == 32'h11bb_33dd);

        // Multiple command bits are rejected atomically and reported sticky.
        write_reg(12'h010, 32'h0000_0012, 4'hf, 0);
        read_reg(12'h014, value);
        assert(value[12]);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h01);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        read_reg(12'h014, value);
        assert(!value[12]);

        write_reg(12'h010, 32'h8000_0000, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h07);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);

        // Narrowed model_spec fields and layer selectors fail closed.
        write_reg(12'h04c, 32'h1000_001c, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0002, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h02);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h04c, 32'd28, 4'hf, 0);

        // BEGIN snapshots all 64-bit staging pairs before issuing to the core.
        write_reg(12'h040, 32'h0000_0007, 4'hf, 0);
        write_reg(12'h044, 32'h89ab_cdef, 4'hf, 0);
        write_reg(12'h048, 32'h0123_4567, 4'hf, 0);
        write_reg(12'h074, 32'h0000_0100, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0002, 4'hf, 0);
        assert(dut.model_spec_op_q == 3'd0);
        assert(dut.begin_id_q == 32'd0);
        assert(dut.begin_hash_q == 64'd0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h02);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h074, 32'd0, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0002, 4'hf, 0);
        assert(dut.begin_id_q == 32'h0000_0007);
        assert(dut.begin_hash_q == 64'h0123_4567_89ab_cdef);
        write_reg(12'h040, 32'h0000_0008, 4'hf, 0);
        write_reg(12'h044, 32'hdead_beef, 4'hf, 0);
        assert(dut.begin_hash_q == 64'h0123_4567_89ab_cdef);
        wait_status_bit(3);

        // A well-formed BEGIN while the model_spec store is not ready updates
        // only the inactive shadow. It is rejected without enqueueing or
        // changing the active model_spec; a later retry snapshots current data.
        write_reg(12'h010, 32'h0000_0002, 4'hf, 0);
        assert(dut.model_spec_op_q == 3'd0);
        assert(dut.begin_id_q == 32'h0000_0008);
        assert(dut.begin_hash_q == 64'h0123_4567_dead_beef);
        assert(dut.core_active_model_spec_id == 32'h0000_0007);
        assert(dut.core_active_model_spec_hash == 64'h0123_4567_89ab_cdef);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h02);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);

        write_reg(12'h010, 32'h0000_0001, 4'hf, 0);
        repeat (4) @(posedge s_axi_aclk);
        assert(!dut.core_model_spec_loading);
        assert(dut.model_spec_op_q == 3'd0);
        write_reg(12'h040, 32'h0000_0007, 4'hf, 0);
        write_reg(12'h044, 32'h89ab_cdef, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0002, 4'hf, 0);
        assert(dut.begin_id_q == 32'h0000_0007);
        assert(dut.begin_hash_q == 64'h0123_4567_89ab_cdef);
        wait_status_bit(3);

        write_reg(12'h090, 32'h0000_0041, 4'hf, 0);
        write_reg(12'h094, 32'h0000_0008, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0004, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h02);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);

        write_reg(12'h090, 32'd3, 4'hf, 0);
        write_reg(12'h094, 32'd5, 4'hf, 0);
        write_reg(12'h09c, 32'h0000_0100, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0004, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h02);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);

        // A bad descriptor response is serialized into the model_spec mailbox.
        write_reg(12'h090, 32'd3, 4'hf, 0);
        write_reg(12'h094, 32'd5, 4'hf, 0);
        write_reg(12'h098, 32'hcdef_0000, 4'hf, 0);
        write_reg(12'h09c, 32'h0000_00ab, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0004, 4'hf, 0);
        assert(dut.layer_index_q == 6'd3);
        assert(dut.layer_word_q == 3'd5);
        assert(dut.layer_data_q == 64'h0000_00ab_cdef_0000);
        wait_status_bit(10);
        read_reg(12'h150, value);
        assert(value == {13'd0, 3'd5, 2'd0, 6'd3, 8'ha5});
        write_reg(12'h010, 32'h0000_0200, 4'hf, 0);

        write_reg(12'h098, 32'h5566_7788, 4'hf, 0);
        write_reg(12'h09c, 32'h0000_0034, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0004, 4'hf, 0);
        assert(dut.layer_data_q == 64'h0000_0034_5566_7788);
        write_reg(12'h010, 32'h0000_0008, 4'hf, 0);
        wait_status_bit(4);

        // Reserved shape and position bits cannot alias valid core commands.
        write_reg(12'h0c0, 32'hbad0_bad0, 4'hf, 0);
        write_reg(12'h0d0, 32'h8001_0101, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        assert(!dut.exec_pending_q);
        assert(dut.exec_tag_q == 32'd0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h0d0, 32'h0001_0101, 4'hf, 0);
        write_reg(12'h0d4, 32'h0002_0000, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h0d4, 32'd0, 4'hf, 0);
        write_reg(12'h0b0, 32'h0002_0000, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h0b0, 32'd64, 4'hf, 0);
        write_reg(12'h0dc, 32'h0000_0100, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        write_reg(12'h0dc, 32'd0, 4'hf, 0);

        // A valid EXEC snapshots its command, preserves LM greedy enable, and
        // captures result and commit even when staging is changed afterward.
        write_reg(12'h0c0, 32'h1234_5678, 4'hf, 0);
        write_reg(12'h0c4, 32'h0000_0007, 4'hf, 0);
        write_reg(12'h0c8, 32'h89ab_cdef, 4'hf, 0);
        write_reg(12'h0cc, 32'h0123_4567, 4'hf, 0);
        write_reg(12'h0d4, 32'd9, 4'hf, 0);
        write_reg(12'h0d8, 32'h1000_0000, 4'hf, 0);
        write_reg(12'h0dc, 32'd0, 4'hf, 0);
        write_reg(12'h0e0, 32'h0000_0042, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        assert(dut.exec_tag_q == 32'h1234_5678);
        assert(dut.exec_model_spec_hash_q == 64'h0123_4567_89ab_cdef);
        assert(dut.exec_kv_capacity_q == 17'd64);
        assert(dut.exec_emit_logits_q);
        write_reg(12'h0c0, 32'hfeed_face, 4'hf, 0);
        write_reg(12'h0b0, 32'd96, 4'hf, 0);
        assert(dut.exec_tag_q == 32'h1234_5678);
        assert(dut.exec_kv_capacity_q == 17'd64);
        assert(dut.core_busy);

        // A valid-shaped EXEC attempt while the core is busy is rejected and
        // cannot enqueue a second command. Only its inactive shadow changes;
        // the already accepted core transaction remains the original one.
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        assert(!dut.exec_pending_q);
        assert(dut.exec_tag_q == 32'hfeed_face);
        assert(dut.exec_kv_capacity_q == 17'd96);
        assert(dut.u_datapath.accepted_tag_q == 32'h1234_5678);
        assert(dut.u_datapath.accepted_kv_capacity_q == 17'd64);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0400, 4'hf, 0);
        wait_status_bit(9);
        wait_status_bit(7);
        wait_status_bit(18);
        read_reg(12'h140, value);
        assert(value == {5'd0, 1'b0, 8'h5a, 18'h2aaaa});
        read_reg(12'h144, value);
        assert(value == 32'h3f80_0000);
        read_reg(12'h100, value);
        assert(value == 32'h1234_5678);
        read_reg(12'h110, value);
        assert(value == {10'd0, 1'b1, 17'd10, 4'd1});
        read_reg(12'h018, value);
        assert(value != 32'd0);
        completed_cycles = value;
        read_reg(12'h168, value);
        assert(value[0] && value[3] && (value[6:5] == 2'd1));
        read_reg(12'h16c, value);
        assert(value == 32'h1234_5678);
        write_reg(12'h170, 32'd0, 4'hf, 0);
        read_reg(12'h174, value);
        assert(value == completed_cycles);
        read_reg(12'h178, value);
        assert(value == 32'd0);
        write_reg(12'h170, 32'd1, 4'hf, 0);
        read_reg(12'h174, metric_control_cycles);
        write_reg(12'h170, 32'd10, 4'hf, 0);
        read_reg(12'h174, metric_down_cycles);
        assert(metric_control_cycles + metric_down_cycles ==
               completed_cycles);

        // No new EXEC enters while an architectural event is unacknowledged.
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        read_reg(12'h034, value);
        assert(value[7:0] == 8'h03);
        write_reg(12'h010, 32'h0000_0d40, 4'hf, 0);
        read_reg(12'h014, value);
        assert(!value[18] && !value[12] && !value[9] && !value[7]);

        // Runtime error fields are captured independently from commit/result.
        write_reg(12'h0c0, 32'h8000_005a, 4'hf, 0);
        write_reg(12'h0b0, 32'd80, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        assert(dut.exec_tag_q == 32'h8000_005a);
        assert(dut.exec_kv_capacity_q == 17'd80);
        assert(dut.u_datapath.accepted_tag_q == 32'h8000_005a);
        assert(dut.u_datapath.accepted_kv_capacity_q == 17'd80);
        wait_status_bit(8);
        wait_status_bit(18);
        read_reg(12'h120, value);
        assert(value == 32'h8000_005a);
        read_reg(12'h124, value);
        assert(value == 32'h0056_1234);
        read_reg(12'h128, value);
        assert(value == {19'd0, 5'd11, 2'd0, 6'd7});
        read_reg(12'h168, value);
        assert(value[0] && (value[6:5] == 2'd2));
        read_reg(12'h16c, value);
        assert(value == 32'h8000_005a);
        write_reg(12'h010, 32'h0000_0880, 4'hf, 0);

        // RUN_CLEAR cancels a live command and remains asserted until every
        // modeled mover reports idle, then permits a clean restart.
        write_reg(12'h0c0, 32'h0000_0099, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0020, 4'hf, 0);
        held_clear_cycles = 0;
        while (dut.run_clear_q) begin
            held_clear_cycles = held_clear_cycles + 1;
            @(posedge s_axi_aclk);
        end
        assert(held_clear_cycles >= 4);
        wait_status_bit(18);
        read_reg(12'h014, value);
        assert(value[18] && !value[1] && value[2] && !value[0]);
        read_reg(12'h168, value);
        assert(value[0] && (value[6:5] == 2'd3));
        read_reg(12'h16c, value);
        assert(value == 32'h0000_0099);
        write_reg(12'h010, 32'h0000_0800, 4'hf, 0);

        write_reg(12'h0c0, 32'h0000_00aa, 4'hf, 0);
        write_reg(12'h010, 32'h0000_0010, 4'hf, 0);
        wait_status_bit(9);
        wait_status_bit(7);
        wait_status_bit(18);
        write_reg(12'h010, 32'h0000_0940, 4'hf, 0);

        // RVALID/data stay stable until RREADY, independent of AR changes.
        s_axi_rready = 1'b0;
        @(negedge s_axi_aclk);
        s_axi_araddr = 12'h000;
        s_axi_arvalid = 1'b1;
        while (!s_axi_arready) @(negedge s_axi_aclk);
        @(negedge s_axi_aclk);
        s_axi_arvalid = 1'b0;
        while (!s_axi_rvalid) @(negedge s_axi_aclk);
        held_rdata = s_axi_rdata;
        s_axi_araddr = 12'h034;
        repeat (4) begin
            @(posedge s_axi_aclk);
            assert(s_axi_rvalid);
            assert(s_axi_rdata == held_rdata);
        end
        assert(held_rdata == 32'hb05a_4000);
        s_axi_rready = 1'b1;
        @(posedge s_axi_aclk);

        assert(m_axi_w0_arsize == 3'd4 && m_axi_w3_arsize == 3'd4);
        assert(m_axi_w0_arburst == 2'b01 &&
               m_axi_hist_k_arburst == 2'b01 &&
               m_axi_hist_v_arburst == 2'b01 &&
               m_axi_kv_awburst == 2'b01);

        $display("PASS penzai_top AXI-Lite wrapper");
        $finish;
    end

    initial begin
        #2_000_000;
        $fatal(1, "timeout");
    end
endmodule

`default_nettype wire
