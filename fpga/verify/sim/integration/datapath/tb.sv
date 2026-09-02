`timescale 1ns/1ps

`include "engine_defs.vh"

module engine_datapath_tb;
    localparam integer ADDR_W = 40;
    localparam [31:0] MODEL_SPEC_ID = `MODEL_BONSAI_1_7B;
    localparam [63:0] MODEL_SPEC_HASH = `MODEL_LAYOUT_HASH;

    logic clk = 1'b0;
    always #1 clk = ~clk;

    logic rst_n = 1'b0;
    logic run_clear = 1'b0;
    wire clear_done;

    logic model_spec_clear_valid = 1'b0;
    wire model_spec_clear_ready;
    logic model_spec_begin_valid = 1'b0;
    wire model_spec_begin_ready;
    logic [31:0] model_spec_begin_id = MODEL_SPEC_ID;
    logic [63:0] model_spec_begin_hash = MODEL_SPEC_HASH;
    logic [5:0] model_spec_begin_layer_count = 6'd28;
    logic [7:0] model_spec_begin_hidden_blocks = 8'd64;
    logic [9:0] model_spec_begin_ffn_blocks = 10'd192;
    logic [5:0] model_spec_begin_q_heads = 6'd16;
    logic [3:0] model_spec_begin_kv_heads = 4'd8;
    logic [7:0] model_spec_begin_head_dim = 8'd128;
    logic [1:0] model_spec_begin_weight_fmt = 2'd1;
    logic [16:0] model_spec_begin_context_limit = 17'd32768;
    logic [17:0] model_spec_begin_vocab_rows = 18'd151669;
    logic [63:0] model_spec_begin_embed_addr = 64'h0010_0000;
    logic [63:0] model_spec_begin_lm_head_addr = 64'h0010_0000;
    logic [63:0] model_spec_begin_final_norm_addr = 64'h0030_0000;
    logic [63:0] model_spec_begin_rope_table_addr = 64'h0040_0000;
    logic model_spec_layer_wr_valid = 1'b0;
    wire model_spec_layer_wr_ready;
    logic [5:0] model_spec_layer_wr_layer = 6'd0;
    logic [2:0] model_spec_layer_wr_word = 3'd0;
    logic [63:0] model_spec_layer_wr_data = 64'd0;
    logic model_spec_seal_valid = 1'b0;
    wire model_spec_seal_ready;
    wire model_spec_cfg_error_valid;
    logic model_spec_cfg_error_ready = 1'b1;
    wire [7:0] model_spec_cfg_error_code;
    wire [5:0] model_spec_cfg_error_layer;
    wire [2:0] model_spec_cfg_error_word;
    wire model_spec_loading;
    wire model_spec_sealed;
    wire [31:0] interface_version;
    wire [63:0] layer_layout_hash;
    wire [31:0] active_model_spec_id;
    wire [63:0] active_model_spec_hash;

    logic cmd_valid = 1'b0;
    wire cmd_ready;
    logic [31:0] cmd_tag = 32'd0;
    logic [31:0] cmd_model_spec_id = MODEL_SPEC_ID;
    logic [63:0] cmd_model_spec_hash = MODEL_SPEC_HASH;
    logic [3:0] cmd_token_count = 4'd1;
    logic [7:0] cmd_lane_mask = 8'h01;
    logic [255:0] cmd_token_ids = 256'd0;
    logic [16:0] cmd_position_base = 17'd0;
    logic [63:0] cmd_kv_base = 64'h1000_0000;
    logic [16:0] cmd_kv_capacity = 17'd8;
    logic cmd_emit_logits = 1'b0;

    wire commit_valid;
    logic commit_ready = 1'b1;
    wire [31:0] commit_tag;
    wire [31:0] commit_model_spec_id;
    wire [63:0] commit_model_spec_hash;
    wire [3:0] commit_token_count;
    wire [16:0] commit_kv_length;
    wire commit_logits_valid;
    wire error_valid;
    logic error_ready = 1'b1;
    wire [31:0] error_tag;
    wire [15:0] error_code;
    wire [7:0] error_detail;
    wire [5:0] error_layer;
    wire [4:0] error_stage;
    wire logits_valid;
    logic logits_ready = 1'b1;
    wire [17:0] logits_row;
    wire [31:0] logits_data;
    wire logits_last;
    wire result_valid;
    logic result_ready = 1'b1;
    wire [17:0] result_token;
    wire [31:0] result_logit;
    wire result_error;
    wire [7:0] result_status;
    wire busy;
    wire [5:0] debug_layer;
    wire [4:0] debug_stage;
    wire trace_valid;
    wire [5:0] trace_layer;
    wire [4:0] trace_stage;
    wire protocol_error;
    wire metrics_stage_active;
    wire [4:0] metrics_stage;
    wire [12:0] metrics_projection_probe;
    wire [2:0] metrics_weight_axi_r_beats;
    wire [2:0] metrics_weight_axi_r_gap_ports;
    wire metrics_weight_zip_skew;
    wire [1:0] metrics_history_axi_r_beats;
    wire metrics_kv_axi_w_beat;

    wire [4*ADDR_W-1:0] weight_axi_araddr;
    wire [31:0] weight_axi_arlen;
    wire [11:0] weight_axi_arsize;
    wire [7:0] weight_axi_arburst;
    wire [3:0] weight_axi_arvalid;
    logic [3:0] weight_axi_arready;
    logic [511:0] weight_axi_rdata;
    logic [7:0] weight_axi_rresp;
    logic [3:0] weight_axi_rlast;
    logic [3:0] weight_axi_rvalid;
    wire [3:0] weight_axi_rready;

    wire [ADDR_W-1:0] hist_k_axi_araddr;
    wire [7:0] hist_k_axi_arlen;
    wire [2:0] hist_k_axi_arsize;
    wire [1:0] hist_k_axi_arburst;
    wire hist_k_axi_arvalid;
    logic hist_k_axi_arready;
    logic [127:0] hist_k_axi_rdata;
    logic [1:0] hist_k_axi_rresp;
    logic hist_k_axi_rlast;
    logic hist_k_axi_rvalid;
    wire hist_k_axi_rready;
    wire [ADDR_W-1:0] hist_v_axi_araddr;
    wire [7:0] hist_v_axi_arlen;
    wire [2:0] hist_v_axi_arsize;
    wire [1:0] hist_v_axi_arburst;
    wire hist_v_axi_arvalid;
    logic hist_v_axi_arready;
    logic [127:0] hist_v_axi_rdata;
    logic [1:0] hist_v_axi_rresp;
    logic hist_v_axi_rlast;
    logic hist_v_axi_rvalid;
    wire hist_v_axi_rready;

    wire [ADDR_W-1:0] kv_axi_awaddr;
    wire [7:0] kv_axi_awlen;
    wire [2:0] kv_axi_awsize;
    wire [1:0] kv_axi_awburst;
    wire kv_axi_awvalid;
    logic kv_axi_awready;
    wire [127:0] kv_axi_wdata;
    wire [15:0] kv_axi_wstrb;
    wire kv_axi_wlast;
    wire kv_axi_wvalid;
    logic kv_axi_wready;
    logic [1:0] kv_axi_bresp;
    logic kv_axi_bvalid;
    wire kv_axi_bready;

    engine_datapath #(.ADDR_W(ADDR_W)) dut (.*);

    logic [31:0] lfsr_q = 32'h7a31_4d29;
    integer cycle_count = 0;
    integer weight_left [0:3];
    logic [3:0] weight_active = 4'd0;
    integer hist_k_left = 0;
    integer hist_v_left = 0;
    logic hist_k_active = 1'b0;
    logic hist_v_active = 1'b0;
    integer kv_write_left = 0;
    logic kv_write_active = 1'b0;
    integer kv_write_beats = 0;
    logic result_seen = 1'b0;
    logic [17:0] captured_result_token = 18'd0;
    logic [31:0] captured_result_logit = 32'd0;

    integer port_index;
    always_comb begin
        weight_axi_arready = ~weight_active;
        weight_axi_rdata = 512'd0;
        weight_axi_rresp = 8'd0;
        weight_axi_rlast = 4'd0;
        weight_axi_rvalid = 4'd0;
        for (port_index = 0; port_index < 4; port_index = port_index + 1) begin
            weight_axi_rvalid[port_index] = weight_active[port_index] &&
                (lfsr_q[port_index] || lfsr_q[port_index + 8]);
            weight_axi_rlast[port_index] = weight_active[port_index] &&
                                           (weight_left[port_index] == 1);
        end

        hist_k_axi_arready = !hist_k_active;
        hist_k_axi_rdata = 128'd0;
        hist_k_axi_rresp = 2'd0;
        hist_k_axi_rvalid = hist_k_active && (lfsr_q[12] || lfsr_q[4]);
        hist_k_axi_rlast = hist_k_active && (hist_k_left == 1);
        hist_v_axi_arready = !hist_v_active;
        hist_v_axi_rdata = 128'd0;
        hist_v_axi_rresp = 2'd0;
        hist_v_axi_rvalid = hist_v_active && (lfsr_q[13] || lfsr_q[5]);
        hist_v_axi_rlast = hist_v_active && (hist_v_left == 1);

        kv_axi_awready = !kv_write_active && !kv_axi_bvalid;
        kv_axi_wready = kv_write_active && (lfsr_q[14] || lfsr_q[6]);
        kv_axi_bresp = 2'd0;
    end

    integer lane;
    always @(posedge clk) begin
        if (!rst_n) begin
            lfsr_q <= 32'h7a31_4d29;
            cycle_count <= 0;
            weight_active <= 4'd0;
            hist_k_active <= 1'b0;
            hist_v_active <= 1'b0;
            kv_write_active <= 1'b0;
            kv_write_beats <= 0;
            kv_axi_bvalid <= 1'b0;
            for (lane = 0; lane < 4; lane = lane + 1)
                weight_left[lane] <= 0;
        end else begin
            lfsr_q <= {lfsr_q[30:0],
                       lfsr_q[31] ^ lfsr_q[21] ^ lfsr_q[1] ^ lfsr_q[0]};
            cycle_count <= cycle_count + 1;

            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (weight_axi_arvalid[lane] && weight_axi_arready[lane]) begin
                    weight_active[lane] <= 1'b1;
                    weight_left[lane] <= weight_axi_arlen[lane*8 +: 8] + 1;
                    assert(weight_axi_arsize[lane*3 +: 3] == 3'd4);
                    assert(weight_axi_arburst[lane*2 +: 2] == 2'b01);
                end
                if (weight_axi_rvalid[lane] && weight_axi_rready[lane]) begin
                    assert(weight_left[lane] > 0);
                    assert(weight_axi_rlast[lane] ==
                           (weight_left[lane] == 1));
                    if (weight_left[lane] == 1) begin
                        weight_left[lane] <= 0;
                        weight_active[lane] <= 1'b0;
                    end else begin
                        weight_left[lane] <= weight_left[lane] - 1;
                    end
                end
            end

            if (hist_k_axi_arvalid && hist_k_axi_arready) begin
                hist_k_active <= 1'b1;
                hist_k_left <= hist_k_axi_arlen + 1;
                assert(hist_k_axi_arsize == 3'd4);
                assert(hist_k_axi_arburst == 2'b01);
            end
            if (hist_k_axi_rvalid && hist_k_axi_rready) begin
                assert(hist_k_axi_rlast == (hist_k_left == 1));
                if (hist_k_left == 1) begin
                    hist_k_active <= 1'b0;
                    hist_k_left <= 0;
                end else begin
                    hist_k_left <= hist_k_left - 1;
                end
            end

            if (hist_v_axi_arvalid && hist_v_axi_arready) begin
                hist_v_active <= 1'b1;
                hist_v_left <= hist_v_axi_arlen + 1;
                assert(hist_v_axi_arsize == 3'd4);
                assert(hist_v_axi_arburst == 2'b01);
            end
            if (hist_v_axi_rvalid && hist_v_axi_rready) begin
                assert(hist_v_axi_rlast == (hist_v_left == 1));
                if (hist_v_left == 1) begin
                    hist_v_active <= 1'b0;
                    hist_v_left <= 0;
                end else begin
                    hist_v_left <= hist_v_left - 1;
                end
            end

            if (kv_axi_awvalid && kv_axi_awready) begin
                kv_write_active <= 1'b1;
                kv_write_left <= kv_axi_awlen + 1;
                assert(kv_axi_awsize == 3'd4);
                assert(kv_axi_awburst == 2'b01);
            end
            if (kv_axi_wvalid && kv_axi_wready) begin
                assert(kv_axi_wstrb == 16'hffff);
                assert(kv_axi_wlast == (kv_write_left == 1));
                kv_write_beats <= kv_write_beats + 1;
                if (kv_write_left == 1) begin
                    kv_write_active <= 1'b0;
                    kv_write_left <= 0;
                    kv_axi_bvalid <= 1'b1;
                end else begin
                    kv_write_left <= kv_write_left - 1;
                end
            end
            if (kv_axi_bvalid && kv_axi_bready)
                kv_axi_bvalid <= 1'b0;

            if (result_valid && result_ready) begin
                result_seen <= 1'b1;
                captured_result_token <= result_token;
                captured_result_logit <= result_logit;
                assert(!result_error);
                assert(result_status == 8'd0);
            end
        end
    end

    task automatic send_model_spec_begin;
        begin
            @(negedge clk);
            model_spec_begin_valid = 1'b1;
            while (!model_spec_begin_ready) @(negedge clk);
            @(negedge clk);
            model_spec_begin_valid = 1'b0;
        end
    endtask

    task automatic send_layer_word(
        input [5:0] layer_index,
        input [2:0] word_index
    );
        begin
            @(negedge clk);
            model_spec_layer_wr_layer = layer_index;
            model_spec_layer_wr_word = word_index;
            model_spec_layer_wr_data = 64'h0100_0000 +
                                    ({58'd0, layer_index} << 24) +
                                    ({61'd0, word_index} << 20);
            model_spec_layer_wr_valid = 1'b1;
            while (!model_spec_layer_wr_ready) @(negedge clk);
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b0;
        end
    endtask

    task automatic seal_model_spec;
        begin
            @(negedge clk);
            model_spec_seal_valid = 1'b1;
            while (!model_spec_seal_ready) @(negedge clk);
            @(negedge clk);
            model_spec_seal_valid = 1'b0;
            wait(model_spec_sealed);
        end
    endtask

    task automatic issue_tile(input [31:0] tag, input [16:0] position);
        begin
            @(negedge clk);
            cmd_tag = tag;
            cmd_position_base = position;
            cmd_token_ids = 256'd0;
            cmd_token_ids[31:0] = position;
            cmd_valid = 1'b1;
            while (!cmd_ready) @(negedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic wait_for_commit(
        input [31:0] expected_tag,
        input [16:0] expected_length,
        input expected_logits
    );
        integer deadline;
        begin
            deadline = cycle_count + 20_000_000;
            while (!commit_valid && !error_valid &&
                   (cycle_count < deadline)) @(posedge clk);
            assert(!error_valid)
                else $fatal(1, "tile %h failed code=%h detail=%h layer=%0d stage=%0d",
                            expected_tag, error_code, error_detail,
                            error_layer, error_stage);
            assert(commit_valid) else $fatal(1, "tile timeout");
            assert(commit_tag == expected_tag);
            assert(commit_model_spec_id == MODEL_SPEC_ID);
            assert(commit_model_spec_hash == MODEL_SPEC_HASH);
            assert(commit_token_count == 4'd1);
            assert(commit_kv_length == expected_length);
            assert(commit_logits_valid == expected_logits);
            @(posedge clk);
        end
    endtask

    integer trace_count = 0;
    integer q8_req_total = 0;
    integer q8_req_qkv = 0;
    integer q8_req_o = 0;
    integer q8_req_gate = 0;
    integer q8_req_down = 0;
    integer q8_req_lm = 0;
    logic check_trace = 1'b0;
    function automatic [4:0] expected_stage(input integer index);
        begin
            if (index < 27) begin
                case (index % 9)
                    0: expected_stage = 5'd0;
                    1: expected_stage = 5'd1;
                    2: expected_stage = 5'd2;
                    3: expected_stage = 5'd4;
                    4: expected_stage = 5'd5;
                    5: expected_stage = 5'd6;
                    6: expected_stage = 5'd8;
                    7: expected_stage = 5'd9;
                    default: expected_stage = 5'd11;
                endcase
            end else if (index == 27) begin
                expected_stage = 5'd13;
            end else begin
                expected_stage = 5'd14;
            end
        end
    endfunction

    always @(posedge clk) begin
        if (rst_n && dut.arena_q8_rd_req_valid &&
            dut.arena_q8_rd_req_ready) begin
            q8_req_total <= q8_req_total + 1;
            case (debug_stage)
                `ENGINE_STAGE_QKV_ROPE: q8_req_qkv <= q8_req_qkv + 1;
                `ENGINE_STAGE_O_PROJ_RESID: q8_req_o <= q8_req_o + 1;
                `ENGINE_STAGE_GATE_UP_SWIGLU_Q8: q8_req_gate <= q8_req_gate + 1;
                `ENGINE_STAGE_DOWN_RESID: q8_req_down <= q8_req_down + 1;
                `ENGINE_STAGE_LM_HEAD: q8_req_lm <= q8_req_lm + 1;
                default: $fatal(1, "Q8 read outside projection stage %0d",
                                debug_stage);
            endcase
        end
        if (rst_n && check_trace && trace_valid) begin
            assert(trace_layer == 6'd0);
            assert(trace_stage == expected_stage(trace_count))
                else $fatal(1, "trace[%0d]=%0d expected=%0d",
                            trace_count, trace_stage,
                            expected_stage(trace_count));
            trace_count <= trace_count + 1;
        end
        if (rst_n && protocol_error)
            $fatal(1, "datapath protocol_error");
    end

    integer layer_index;
    integer word_index;
    integer clear_deadline;
    initial begin
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        repeat (4) @(posedge clk);

        send_model_spec_begin();
        for (layer_index = 0; layer_index < 28; layer_index = layer_index + 1)
            for (word_index = 0; word_index < 8; word_index = word_index + 1)
                send_layer_word(layer_index[5:0], word_index[2:0]);
        seal_model_spec();
        assert(interface_version == `ENGINE_INTERFACE_VERSION);
        assert(layer_layout_hash == MODEL_SPEC_HASH);

        // This is the arithmetic/ownership smoke test, not the controller's
        // full-layer-count test. Model-spec admission and all 28 address records
        // above are canonical; one layer keeps the real datapath regression
        // bounded while engine_core/tb.sv walks all model layers.
        force dut.u_engine.p_layer_count = 6'd1;

        // Cancel with both a held arena response and packed AXI traffic live,
        // then require bounded AXI quarantine before accepting a restart.
        issue_tile(32'h0000_0001, 17'd0);
        wait(trace_valid && trace_stage == 5'd2);
        clear_deadline = cycle_count + 1_000_000;
        while (!((|weight_active) && dut.arena_q8_rd_rsp_valid &&
                 (dut.u_projection.q8_outstanding_q != 0)) &&
               (cycle_count < clear_deadline)) @(negedge clk);
        assert((|weight_active) && dut.arena_q8_rd_rsp_valid &&
               (dut.u_projection.q8_outstanding_q != 0))
            else $fatal(1, "did not reach simultaneous arena/AXI clear point");
        run_clear = 1'b1;
        #1ps;
        assert(!clear_done && !cmd_ready)
            else $fatal(1, "clear distribution did not quarantine commands");
        @(negedge clk);
        clear_deadline = cycle_count + 10000;
        while (dut.arena_q8_rd_rsp_valid &&
               (cycle_count < clear_deadline)) @(posedge clk);
        assert(!dut.arena_q8_rd_rsp_valid)
            else $fatal(1, "distributed clear did not cancel arena response");
        while (!clear_done && (cycle_count < clear_deadline)) @(posedge clk);
        if (!clear_done)
            $display("clear state eng=%b go=%b fo=%b vd=%b emb=%b proj=%b attn=%b app=%b cluster=%b rope=%b quad=%b hk=%b hv=%b wr=%b pstate=%0d cowner=%0d qstate=%0d",
                     busy, dut.gemm_owner_q, dut.flash_owner_q,
                     dut.vector_dispatch_busy, dut.embed_busy,
                     dut.projection_busy, dut.attention_busy,
                     dut.append_busy, dut.cluster_busy, dut.rope_busy,
                     dut.quad_busy, dut.hist_k_busy, dut.hist_v_busy,
                     dut.kv_writer_busy, dut.u_projection.state_q,
                     dut.u_vector_sink_cluster.command_owner_q,
                     dut.u_weight_reader.state_q);
        assert(clear_done) else $fatal(1, "clear did not drain");
        assert(!commit_valid && !error_valid);
        @(negedge clk);
        run_clear = 1'b0;
        #1ps;
        assert(!clear_done && !cmd_ready)
            else $fatal(1, "restart escaped before clear islands settled");
        clear_deadline = cycle_count + 100;
        while (!clear_done && (cycle_count < clear_deadline)) @(posedge clk);
        assert(clear_done && cmd_ready)
            else $fatal(1, "clear islands did not release cleanly");

        check_trace = 1'b1;
        issue_tile(32'h0000_0002, 17'd0);
        wait_for_commit(32'h0000_0002, 17'd1, 1'b0);
        issue_tile(32'h0000_0003, 17'd1);
        wait_for_commit(32'h0000_0003, 17'd2, 1'b0);
        cmd_emit_logits = 1'b1;
        issue_tile(32'h0000_0004, 17'd2);
        wait_for_commit(32'h0000_0004, 17'd3, 1'b1);

        assert(trace_count == 29);
        assert(kv_write_beats == 768)
            else $fatal(1, "KV beats=%0d expected=768", kv_write_beats);
        assert(result_seen);
        assert(captured_result_token == 18'd0);
        assert(captured_result_logit == 32'd0);
        assert(!logits_valid);
        $display("PASS engine_datapath cycles=%0d trace=%0d kv_beats=%0d q8_req=%0d qkv=%0d o=%0d gate=%0d down=%0d lm=%0d",
                 cycle_count, trace_count, kv_write_beats, q8_req_total,
                 q8_req_qkv, q8_req_o, q8_req_gate, q8_req_down, q8_req_lm);
        $finish;
    end
endmodule
