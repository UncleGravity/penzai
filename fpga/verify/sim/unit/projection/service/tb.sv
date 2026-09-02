`timescale 1ns/1ps
`default_nettype none

`include "engine_defs.vh"

module projection_service_tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;

    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [2:0] cmd_op = `PROJECTION_OP_O;
    reg [3:0] cmd_token_count = 4'd1;
    reg [7:0] cmd_token_mask = 8'h01;
    reg [63:0] cmd_addr0 = 64'h1000;
    reg [63:0] cmd_addr1 = 64'h2000;
    reg [63:0] cmd_addr2 = 64'h3000;
    reg [63:0] cmd_addr3 = 64'h4000;
    reg [7:0] cmd_hidden_blocks = 8'd64;
    reg [9:0] cmd_ffn_blocks = 10'd192;
    reg [5:0] cmd_q_heads = 6'd16;
    reg [3:0] cmd_kv_heads = 4'd8;
    reg [7:0] cmd_head_dim = 8'd128;
    reg [16:0] cmd_position_base = 17'd0;
    reg [31:0] cmd_epsilon = 32'h3586_37bd;
    reg [17:0] cmd_vocab_rows = 18'd16;
    reg [1:0] cmd_weight_fmt = 2'd1;
    reg cmd_emit_full_logits = 1'b0;

    wire read_cmd_valid;
    wire read_cmd_ready;
    wire [63:0] read_cmd_base_addr;
    wire [31:0] read_cmd_port_beats;
    wire [3:0] read_cmd_port_mask;
    wire read_abort;
    reg [511:0] read_data = 512'd0;
    wire read_valid;
    wire read_ready;
    wire read_last;
    reg read_error = 1'b0;
    wire read_busy;
    wire read_done_valid;
    wire read_done_ready;
    reg read_done_error = 1'b0;
    reg [7:0] read_done_status = 8'd0;

    wire q8_rd_req_valid;
    wire q8_rd_req_ready;
    wire q8_rd_req_wave;
    wire [8:0] q8_rd_req_addr;
    wire q8_rd_rsp_valid;
    wire q8_rd_rsp_ready;
    reg [1087:0] q8_rd_rsp_data = 1088'd0;

    wire sink_cfg_valid;
    wire sink_cfg_ready;
    wire [1:0] sink_cfg_mode;
    wire [7:0] sink_cfg_token_mask;
    wire [12:0] sink_cfg_hidden_dim;
    wire [14:0] sink_cfg_ffn_dim;
    wire [5:0] sink_cfg_q_heads;
    wire [3:0] sink_cfg_kv_heads;
    wire [16:0] sink_cfg_position_base;
    wire [31:0] sink_cfg_epsilon;
    wire [63:0] sink_cfg_q_gamma_addr;
    wire [63:0] sink_cfg_k_gamma_addr;
    wire [63:0] sink_cfg_rope_addr;
    wire sink_abort_run;
    wire sink_projection_armed;
    wire sink_proj_valid;
    reg sink_proj_ready = 1'b1;
    wire [2:0] sink_proj_token;
    wire [17:0] sink_proj_row;
    wire [31:0] sink_proj_data_f32;
    wire sink_proj_last;
    reg sink_done_valid = 1'b0;
    wire sink_done_ready;
    reg sink_done_error = 1'b0;
    reg [15:0] sink_done_status = 16'd0;

    wire logits_valid;
    reg logits_ready = 1'b1;
    wire [17:0] logits_row;
    wire [31:0] logits_data;
    wire logits_last;
    wire result_valid;
    reg result_ready = 1'b0;
    wire [17:0] result_token;
    wire [31:0] result_logit;
    wire result_error;
    wire [7:0] result_status;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b0;
    wire done_error;
    wire [7:0] done_status;
    wire [15:0] derived_k;
    wire [17:0] derived_m;
    wire [15:0] derived_rowblocks;
    wire [31:0] accepted_weight_beats;
    wire [31:0] activation_wave_issues;
    wire [12:0] metrics_probe;

     projection_service dut (.*);

    // Minimal shared-reader model. It preserves command/done ownership and
    // streams exactly the requested number of lockstep beats.
    reg reader_enable = 1'b0;
    reg reader_stream_q = 1'b0;
    reg reader_done_q = 1'b0;
    reg [31:0] reader_left_q = 32'd0;
    assign read_cmd_ready = reader_enable && !reader_stream_q &&
                            !reader_done_q;
    assign read_valid = reader_stream_q;
    assign read_last = reader_stream_q && (reader_left_q == 32'd1);
    assign read_busy = reader_stream_q;
    assign read_done_valid = reader_done_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            reader_stream_q <= 1'b0;
            reader_done_q <= 1'b0;
            reader_left_q <= 32'd0;
        end else if (read_abort) begin
            reader_stream_q <= 1'b0;
            reader_done_q <= 1'b0;
            reader_left_q <= 32'd0;
        end else begin
            if (read_cmd_valid && read_cmd_ready) begin
                reader_stream_q <= 1'b1;
                reader_left_q <= read_cmd_port_beats;
            end
            if (read_valid && read_ready) begin
                if (reader_left_q == 32'd1) begin
                    reader_stream_q <= 1'b0;
                    reader_done_q <= 1'b1;
                    reader_left_q <= 32'd0;
                end else begin
                    reader_left_q <= reader_left_q - 1'b1;
                end
            end
            if (reader_done_q && read_done_ready)
                reader_done_q <= 1'b0;
        end
    end

    // Two-entry resident arena model with a deliberate four-cycle delay. This
    // matches the production memory-output plus held-response elasticity and
    // allows both tile-8 waves to be accepted before the first response returns.
    reg [2:0] q8_delay_q = 3'd0;
    reg [1:0] q8_pending_q = 2'd0;
    wire q8_rsp_valid_q = (q8_pending_q != 0) && (q8_delay_q == 0);
    wire q8_req_fire = q8_rd_req_valid && q8_rd_req_ready;
    wire q8_rsp_fire = q8_rd_rsp_valid && q8_rd_rsp_ready;
    assign q8_rd_req_ready = q8_pending_q < 2;
    assign q8_rd_rsp_valid = q8_rsp_valid_q;
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            q8_delay_q <= 3'd0;
            q8_pending_q <= 2'd0;
        end else begin
            if (q8_req_fire && (q8_pending_q == 0))
                q8_delay_q <= 3'd4;
            else if (q8_delay_q != 0)
                q8_delay_q <= q8_delay_q - 1'b1;
            case ({q8_req_fire, q8_rsp_fire})
                2'b10: q8_pending_q <= q8_pending_q + 1'b1;
                2'b01: q8_pending_q <= q8_pending_q - 1'b1;
                default: ;
            endcase
        end
    end

    // Semantic sink model. QKV arming is held by the test to prove that the
    // long projection reader cannot starve gamma/RoPE preparation.
    reg sink_active_q = 1'b0;
    reg sink_cfg_enable = 1'b1;
    reg sink_arm_enable = 1'b0;
    reg [1:0] sink_mode_q = 2'd0;
    integer sink_cfg_fire_count = 0;
    integer logits_start_fire_count = 0;
    integer read_cmd_fire_count = 0;
    integer cfg_count_before;
    integer logits_count_before;
    integer read_count_before;
    integer full_logit_count;
    assign sink_cfg_ready = sink_cfg_enable && !sink_active_q;
    assign sink_projection_armed = sink_active_q &&
        ((sink_mode_q != 2'd0) || sink_arm_enable);
    always @(posedge clk) begin
        if (!rst_n || sink_abort_run) begin
            sink_active_q <= 1'b0;
            sink_mode_q <= 2'd0;
        end else if (sink_cfg_valid && sink_cfg_ready) begin
            sink_active_q <= 1'b1;
            sink_mode_q <= sink_cfg_mode;
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            sink_cfg_fire_count <= 0;
            logits_start_fire_count <= 0;
            read_cmd_fire_count <= 0;
            full_logit_count <= 0;
        end else begin
            if (sink_cfg_valid && sink_cfg_ready)
                sink_cfg_fire_count <= sink_cfg_fire_count + 1;
            if (dut.logits_start_valid && dut.logits_start_ready)
                logits_start_fire_count <= logits_start_fire_count + 1;
            if (read_cmd_valid && read_cmd_ready)
                read_cmd_fire_count <= read_cmd_fire_count + 1;
            if (logits_valid && logits_ready)
                full_logit_count <= full_logit_count + 1;
        end
    end

    task automatic set_common;
        begin
            cmd_valid = 1'b0;
            cmd_token_count = 4'd1;
            cmd_token_mask = 8'h01;
            cmd_addr0 = 64'h1000;
            cmd_addr1 = 64'h2000;
            cmd_addr2 = 64'h3000;
            cmd_addr3 = 64'h4000;
            cmd_hidden_blocks = 8'd64;
            cmd_ffn_blocks = 10'd192;
            cmd_q_heads = 6'd16;
            cmd_kv_heads = 4'd8;
            cmd_head_dim = 8'd128;
            cmd_position_base = 17'd0;
            cmd_epsilon = 32'h3586_37bd;
            cmd_vocab_rows = 18'd16;
            cmd_weight_fmt = 2'd1;
            cmd_emit_full_logits = 1'b0;
        end
    endtask

    task automatic issue_command;
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            #0.1;
            while (!cmd_ready) begin
                @(negedge clk);
                #0.1;
            end
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic pulse_abort;
        begin
            @(negedge clk);
            abort_run = 1'b1;
            @(negedge clk);
            abort_run = 1'b0;
            while (!cmd_ready) @(negedge clk);
        end
    endtask

    task automatic pulse_clear;
        begin
            @(negedge clk);
            clear = 1'b1;
            @(negedge clk);
            clear = 1'b0;
            while (!cmd_ready) @(negedge clk);
        end
    endtask

    task automatic check_shape;
        input [2:0] op;
        input [7:0] hidden_blocks;
        input [9:0] ffn_blocks;
        input [5:0] q_heads;
        input [1:0] fmt;
        input [15:0] expected_k;
        input [17:0] expected_m;
        input [15:0] expected_rowblocks;
        input [31:0] expected_beats;
        integer cfg_before;
        integer logits_before;
        begin
            set_common();
            cmd_op = op;
            cmd_hidden_blocks = hidden_blocks;
            cmd_ffn_blocks = ffn_blocks;
            cmd_q_heads = q_heads;
            cmd_vocab_rows = `MODEL_BONSAI_VOCAB_ROWS;
            cmd_weight_fmt = fmt;
            reader_enable = 1'b0;
            sink_cfg_enable = 1'b1;
            sink_arm_enable = 1'b1;
            cfg_before = sink_cfg_fire_count;
            logits_before = logits_start_fire_count;
            issue_command();
            if (read_cmd_valid)
                $fatal(1, "projection reader skipped PREP op=%0d fmt=%0d",
                       op, fmt);
            while (!read_cmd_valid) @(negedge clk);
            if (op == `PROJECTION_OP_LM_HEAD) begin
                if ((logits_start_fire_count != logits_before + 1) ||
                    (sink_cfg_fire_count != cfg_before))
                    $fatal(1, "LM child configured wrong number of times");
            end else if ((sink_cfg_fire_count != cfg_before + 1) ||
                         (logits_start_fire_count != logits_before)) begin
                $fatal(1, "semantic child configured wrong number of times");
            end
            if ((derived_k != expected_k) || (derived_m != expected_m) ||
                (derived_rowblocks != expected_rowblocks) ||
                (read_cmd_port_beats != expected_beats))
                $fatal(1,
                    "shape mismatch op=%0d fmt=%0d got k=%0d m=%0d rb=%0d beats=%0d expected k=%0d m=%0d rb=%0d beats=%0d",
                    op, fmt, derived_k, derived_m, derived_rowblocks,
                    read_cmd_port_beats, expected_k, expected_m,
                    expected_rowblocks, expected_beats);
            pulse_clear();
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        // Every supported model_spec/operation pair exercises both packed-weight
        // formats. LM uses the non-multiple-of-16 Bonsai vocabulary tail.
        check_shape(`PROJECTION_OP_QKV, 8'd64, 10'd192, 6'd16, 2'd1,
                    16'd2048, 18'd4096, 16'd256, 32'd20480);
        check_shape(`PROJECTION_OP_O, 8'd64, 10'd192, 6'd16, 2'd1,
                    16'd2048, 18'd2048, 16'd128, 32'd10240);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd64, 10'd192, 6'd16, 2'd1,
                    16'd2048, 18'd12288, 16'd768, 32'd61440);
        check_shape(`PROJECTION_OP_DOWN, 8'd64, 10'd192, 6'd16, 2'd1,
                    16'd6144, 18'd2048, 16'd128, 32'd30720);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd64, 10'd192, 6'd16, 2'd1,
                    16'd2048, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd758400);
        check_shape(`PROJECTION_OP_QKV, 8'd64, 10'd192, 6'd16, 2'd2,
                    16'd2048, 18'd4096, 16'd256, 32'd36864);
        check_shape(`PROJECTION_OP_O, 8'd64, 10'd192, 6'd16, 2'd2,
                    16'd2048, 18'd2048, 16'd128, 32'd18432);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd64, 10'd192, 6'd16, 2'd2,
                    16'd2048, 18'd12288, 16'd768, 32'd110592);
        check_shape(`PROJECTION_OP_DOWN, 8'd64, 10'd192, 6'd16, 2'd2,
                    16'd6144, 18'd2048, 16'd128, 32'd55296);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd64, 10'd192, 6'd16, 2'd2,
                    16'd2048, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd1365120);

        check_shape(`PROJECTION_OP_QKV, 8'd80, 10'd304, 6'd32, 2'd1,
                    16'd2560, 18'd6144, 16'd384, 32'd38400);
        check_shape(`PROJECTION_OP_O, 8'd80, 10'd304, 6'd32, 2'd1,
                    16'd4096, 18'd2560, 16'd160, 32'd25600);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd80, 10'd304, 6'd32, 2'd1,
                    16'd2560, 18'd19456, 16'd1216, 32'd121600);
        check_shape(`PROJECTION_OP_DOWN, 8'd80, 10'd304, 6'd32, 2'd1,
                    16'd9728, 18'd2560, 16'd160, 32'd60800);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd80, 10'd304, 6'd32, 2'd1,
                    16'd2560, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd948000);
        check_shape(`PROJECTION_OP_QKV, 8'd80, 10'd304, 6'd32, 2'd2,
                    16'd2560, 18'd6144, 16'd384, 32'd69120);
        check_shape(`PROJECTION_OP_O, 8'd80, 10'd304, 6'd32, 2'd2,
                    16'd4096, 18'd2560, 16'd160, 32'd46080);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd80, 10'd304, 6'd32, 2'd2,
                    16'd2560, 18'd19456, 16'd1216, 32'd218880);
        check_shape(`PROJECTION_OP_DOWN, 8'd80, 10'd304, 6'd32, 2'd2,
                    16'd9728, 18'd2560, 16'd160, 32'd109440);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd80, 10'd304, 6'd32, 2'd2,
                    16'd2560, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd1706400);

        check_shape(`PROJECTION_OP_QKV, 8'd128, 10'd384, 6'd32, 2'd1,
                    16'd4096, 18'd6144, 16'd384, 32'd61440);
        check_shape(`PROJECTION_OP_O, 8'd128, 10'd384, 6'd32, 2'd1,
                    16'd4096, 18'd4096, 16'd256, 32'd40960);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd128, 10'd384, 6'd32, 2'd1,
                    16'd4096, 18'd24576, 16'd1536, 32'd245760);
        check_shape(`PROJECTION_OP_DOWN, 8'd128, 10'd384, 6'd32, 2'd1,
                    16'd12288, 18'd4096, 16'd256, 32'd122880);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd128, 10'd384, 6'd32, 2'd1,
                    16'd4096, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd1516800);
        check_shape(`PROJECTION_OP_QKV, 8'd128, 10'd384, 6'd32, 2'd2,
                    16'd4096, 18'd6144, 16'd384, 32'd110592);
        check_shape(`PROJECTION_OP_O, 8'd128, 10'd384, 6'd32, 2'd2,
                    16'd4096, 18'd4096, 16'd256, 32'd73728);
        check_shape(`PROJECTION_OP_GATE_UP, 8'd128, 10'd384, 6'd32, 2'd2,
                    16'd4096, 18'd24576, 16'd1536, 32'd442368);
        check_shape(`PROJECTION_OP_DOWN, 8'd128, 10'd384, 6'd32, 2'd2,
                    16'd12288, 18'd4096, 16'd256, 32'd221184);
        check_shape(`PROJECTION_OP_LM_HEAD, 8'd128, 10'd384, 6'd32, 2'd2,
                    16'd4096, `MODEL_BONSAI_VOCAB_ROWS, 16'd9480,
                    32'd2730240);
        $display("projection service supported shape matrix PASS");

        // Command acceptance is local. A semantic child may backpressure its
        // configuration indefinitely while every captured field remains stable
        // and later raw commands cannot alter the owned transaction.
        set_common();
        cmd_op = `PROJECTION_OP_QKV;
        cmd_token_count = 4'd3;
        cmd_token_mask = 8'h07;
        cmd_addr0 = 64'h8000;
        cmd_addr1 = 64'h9000;
        cmd_addr2 = 64'ha000;
        cmd_addr3 = 64'hb000;
        cmd_hidden_blocks = 8'd80;
        cmd_ffn_blocks = 10'd304;
        cmd_q_heads = 6'd32;
        cmd_position_base = 17'd123;
        cmd_epsilon = 32'h3a83_126f;
        cmd_weight_fmt = 2'd2;
        sink_cfg_enable = 1'b0;
        sink_arm_enable = 1'b1;
        reader_enable = 1'b0;
        cfg_count_before = sink_cfg_fire_count;
        issue_command();

        cmd_op = `PROJECTION_OP_LM_HEAD;
        cmd_token_count = 4'd8;
        cmd_token_mask = 8'h80;
        cmd_addr0 = 64'h1000;
        cmd_addr1 = 64'h2000;
        cmd_addr2 = 64'h3000;
        cmd_addr3 = 64'h4000;
        cmd_hidden_blocks = 8'd64;
        cmd_ffn_blocks = 10'd192;
        cmd_q_heads = 6'd16;
        cmd_position_base = 17'd9;
        cmd_epsilon = 32'h3586_37bd;
        cmd_vocab_rows = 18'd3;
        cmd_weight_fmt = 2'd1;
        cmd_emit_full_logits = 1'b1;

        while (!sink_cfg_valid) @(negedge clk);
        repeat (6) begin
            @(negedge clk);
            if (!sink_cfg_valid || sink_cfg_ready || read_cmd_valid)
                $fatal(1, "registered semantic config did not hold");
            if ((sink_cfg_mode != 2'd0) ||
                (sink_cfg_token_mask != 8'h07) ||
                (sink_cfg_hidden_dim != 13'd2560) ||
                (sink_cfg_ffn_dim != 15'd9728) ||
                (sink_cfg_q_heads != 6'd32) ||
                (sink_cfg_kv_heads != 4'd8) ||
                (sink_cfg_position_base != 17'd123) ||
                (sink_cfg_epsilon != 32'h3a83_126f) ||
                (sink_cfg_q_gamma_addr != 64'h9000) ||
                (sink_cfg_k_gamma_addr != 64'ha000) ||
                (sink_cfg_rope_addr != 64'hb000))
                $fatal(1, "semantic config changed under backpressure");
        end
        if (sink_cfg_fire_count != cfg_count_before)
            $fatal(1, "backpressured semantic config fired early");
        sink_cfg_enable = 1'b1;
        while (sink_cfg_fire_count == cfg_count_before) @(negedge clk);
        while (!read_cmd_valid) @(negedge clk);
        if ((sink_cfg_fire_count != cfg_count_before + 1) ||
            (derived_k != 16'd2560) || (derived_m != 18'd6144) ||
            (derived_rowblocks != 16'd384) ||
            (read_cmd_port_beats != 32'd69120))
            $fatal(1, "registered semantic config/shape mismatch");
        pulse_clear();
        $display("projection service registered child config hold PASS");

        // Abort a command while its captured configuration is pending. No
        // child or reader transaction may leak, and the next command starts
        // from an exact fresh configuration.
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_token_count = 4'd4;
        cmd_token_mask = 8'h0f;
        cmd_hidden_blocks = 8'd128;
        cmd_ffn_blocks = 10'd384;
        cmd_q_heads = 6'd32;
        sink_cfg_enable = 1'b0;
        reader_enable = 1'b0;
        cfg_count_before = sink_cfg_fire_count;
        read_count_before = read_cmd_fire_count;
        issue_command();
        while (!sink_cfg_valid) @(negedge clk);
        pulse_abort();
        if (sink_cfg_valid || read_cmd_valid ||
            (sink_cfg_fire_count != cfg_count_before) ||
            (read_cmd_fire_count != read_count_before))
            $fatal(1, "abort leaked pending child configuration");

        sink_cfg_enable = 1'b1;
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_token_count = 4'd4;
        cmd_token_mask = 8'h0f;
        cmd_hidden_blocks = 8'd128;
        cmd_ffn_blocks = 10'd384;
        cmd_q_heads = 6'd32;
        reader_enable = 1'b0;
        issue_command();
        while (!read_cmd_valid) @(negedge clk);
        if ((sink_cfg_fire_count != cfg_count_before + 1) ||
            (derived_k != 16'd4096) || (derived_m != 18'd4096) ||
            (derived_rowblocks != 16'd256) ||
            (read_cmd_port_beats != 32'd40960))
            $fatal(1, "semantic restart after config abort mismatch");
        pulse_clear();
        $display("projection service config abort/exact restart PASS");

        // Caller-supplied unsupported dimensions are rejected before any leaf.
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_hidden_blocks = 8'd65;
        sink_cfg_enable = 1'b0;
        cfg_count_before = sink_cfg_fire_count;
        logits_count_before = logits_start_fire_count;
        issue_command();
        if (!done_valid || !done_error || (done_status != 8'h01))
            $fatal(1, "bad shape was not rejected locally");
        if (sink_cfg_valid || dut.logits_start_valid ||
            (sink_cfg_fire_count != cfg_count_before) ||
            (logits_start_fire_count != logits_count_before))
            $fatal(1, "bad shape reached a child config port");
        sink_cfg_enable = 1'b1;
        @(negedge clk); done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;

        // A gamma/RoPE preparation failure completes with an error instead of
        // waiting forever for projection_armed.
        set_common();
        cmd_op = `PROJECTION_OP_QKV;
        sink_arm_enable = 1'b0;
        reader_enable = 1'b0;
        issue_command();
        @(negedge clk);
        sink_done_error = 1'b1;
        sink_done_status = 16'h0200;
        sink_done_valid = 1'b1;
        while (!sink_done_ready) @(negedge clk);
        reader_enable = 1'b1;
        @(negedge clk);
        sink_done_valid = 1'b0;
        sink_done_error = 1'b0;
        sink_done_status = 16'd0;
        while (!done_valid) @(negedge clk);
        if (!done_error || (done_status != 8'h3a) || read_cmd_valid)
            $fatal(1, "sink preparation error did not retire cleanly");
        done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;

        // Semantic tiles are physical prefixes; LM selects only count-1.
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_token_count = 4'd2;
        cmd_token_mask = 8'h05;
        issue_command();
        if (!done_valid || !done_error || (done_status != 8'h01))
            $fatal(1, "sparse semantic mask was not rejected");
        @(negedge clk); done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;

        set_common();
        cmd_op = `PROJECTION_OP_LM_HEAD;
        cmd_token_count = 4'd8;
        cmd_token_mask = 8'h01;
        issue_command();
        if (!done_valid || !done_error || (done_status != 8'h01))
            $fatal(1, "LM mask outside count-1 was not rejected");
        @(negedge clk); done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;

        // 4B has D=2560 but 32 query heads: O consumes 4096 attention rows.
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_hidden_blocks = 8'd80;
        cmd_ffn_blocks = 10'd304;
        cmd_q_heads = 6'd32;
        reader_enable = 1'b0;
        issue_command();
        while (!read_cmd_valid) @(negedge clk);
        if ((derived_k != 16'd4096) || (derived_m != 18'd2560) ||
            (derived_rowblocks != 16'd160) ||
            (read_cmd_port_beats != 32'd25600) ||
            (read_cmd_port_mask != 4'hf))
            $fatal(1, "4B O shape/reader derivation mismatch");
        reader_enable = 1'b1;
        while (!(q8_rd_req_valid && q8_rd_req_ready)) @(negedge clk);
        // Clear invalidates the accepted activation response in both the
        // arena and the service's matching outstanding counter.
        pulse_clear();
        if (q8_rsp_valid_q || (q8_delay_q != 0))
            $fatal(1, "clear retained stale Q8 response");
        $display("projection service 4B O + Q8 clear/restart PASS");

        // Tile-8 has two active four-lane waves. The arena accepts both back-to-back and
        // the service must track depth two until the delayed responses drain.
        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_token_count = 4'd8;
        cmd_token_mask = 8'hff;
        reader_enable = 1'b1;
        issue_command();
        while (dut.q8_outstanding_q != 3'd2) @(negedge clk);
        if ((q8_pending_q != 2'd2) ||
            (activation_wave_issues != 32'd2))
            $fatal(1, "tile-8 Q8 waves were not accepted back-to-back");
        pulse_abort();
        if ((q8_pending_q != 0) || (dut.q8_outstanding_q != 0))
            $fatal(1, "tile-8 Q8 abort did not drain outstanding waves");

        set_common();
        cmd_op = `PROJECTION_OP_O;
        cmd_token_count = 4'd8;
        cmd_token_mask = 8'hff;
        reader_enable = 1'b1;
        issue_command();
        while (dut.q8_outstanding_q != 3'd2) @(negedge clk);
        pulse_clear();
        if ((q8_pending_q != 0) || (dut.q8_outstanding_q != 0))
            $fatal(1, "tile-8 Q8 clear retained outstanding waves");
        $display("projection service tile-8 Q8 depth-two abort/clear PASS");

        // DOWN consumes the disjoint SwiGLU partition. At the largest model_spec
        // the first activation block is 128, never normalized-input block 0.
        set_common();
        cmd_op = `PROJECTION_OP_DOWN;
        cmd_hidden_blocks = 8'd128;
        cmd_ffn_blocks = 10'd384;
        cmd_q_heads = 6'd32;
        reader_enable = 1'b0;
        issue_command();
        while (!read_cmd_valid) @(negedge clk);
        reader_enable = 1'b1;
        while (!(q8_rd_req_valid && q8_rd_req_ready)) @(negedge clk);
        if (q8_rd_req_addr != 9'd128)
            $fatal(1, "DOWN first activation did not use Q8 block 128");
        pulse_clear();
        $display("projection service DOWN Q8 base128 PASS");

        // Final-position single-token decode is legal and QKV preparation owns the
        // reader only after the semantic sink reports gamma/RoPE armed.
        set_common();
        cmd_op = `PROJECTION_OP_QKV;
        cmd_position_base = 17'd65535;
        sink_arm_enable = 1'b0;
        reader_enable = 1'b0;
        issue_command();
        repeat (8) begin
            @(negedge clk);
            if (read_cmd_valid)
                $fatal(1, "QKV reader launched before projection_armed");
        end
        sink_arm_enable = 1'b1;
        while (!read_cmd_valid) @(negedge clk);
        if ((derived_k != 16'd2048) || (derived_m != 18'd4096) ||
            (derived_rowblocks != 16'd256) ||
            (read_cmd_port_beats != 32'd20480))
            $fatal(1, "QKV derived shape mismatch");
        pulse_abort();
        $display("projection service final-position single-token + arm gate PASS");

        // LM completion is not published until the greedy result is accepted.
        set_common();
        cmd_op = `PROJECTION_OP_LM_HEAD;
        cmd_token_count = 4'd8;
        cmd_token_mask = 8'h80;
        reader_enable = 1'b1;
        result_ready = 1'b0;
        issue_command();
        while (!result_valid) @(negedge clk);
        if (result_error || (result_token != 18'd0) ||
            (result_logit != 32'd0))
            $fatal(1, "LM result mismatch");
        repeat (10) begin
            @(negedge clk);
            if (done_valid)
                $fatal(1, "LM completed past held result");
        end
        result_ready = 1'b1;
        @(negedge clk);
        result_ready = 1'b0;
        while (!done_valid) @(negedge clk);
        if (done_error || (accepted_weight_beats != 32'd80))
            $fatal(1, "LM service completion/count mismatch");
        done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;
        $display("projection service held LM result/drain PASS");

        // Full-logit backpressure, tie-breaking, and a second LM launch prove
        // that registered vocab/format controls cannot retain stale ownership.
        set_common();
        cmd_op = `PROJECTION_OP_LM_HEAD;
        cmd_token_count = 4'd4;
        cmd_token_mask = 8'h08;
        cmd_vocab_rows = 18'd3;
        cmd_weight_fmt = 2'd2;
        cmd_emit_full_logits = 1'b1;
        reader_enable = 1'b1;
        logits_ready = 1'b0;
        result_ready = 1'b0;
        read_count_before = full_logit_count;
        logits_count_before = logits_start_fire_count;
        issue_command();
        cmd_vocab_rows = 18'd17;
        cmd_weight_fmt = 2'd1;
        cmd_emit_full_logits = 1'b0;
        while (!logits_valid) @(negedge clk);
        repeat (5) begin
            @(negedge clk);
            if (!logits_valid || (logits_row != 18'd0) ||
                (logits_data != 32'd0) || logits_last || result_valid)
                $fatal(1, "full-logit output changed under backpressure");
        end
        if ((logits_start_fire_count != logits_count_before + 1) ||
            (full_logit_count != read_count_before))
            $fatal(1, "registered LM config/start mismatch");
        logits_ready = 1'b1;
        while (!result_valid) @(negedge clk);
        if (result_error || (result_token != 18'd0) ||
            (result_logit != 32'd0) ||
            (full_logit_count != read_count_before + 3) ||
            (dut.logits_accepted != 32'd3) ||
            (accepted_weight_beats != 32'd144))
            $fatal(1, "full-logit tie/result mismatch");
        result_ready = 1'b1;
        @(negedge clk);
        result_ready = 1'b0;
        while (!done_valid) @(negedge clk);
        if (done_error)
            $fatal(1, "full-logit LM command failed");
        done_ready = 1'b1;
        @(negedge clk);
        done_ready = 1'b0;

        set_common();
        cmd_op = `PROJECTION_OP_LM_HEAD;
        cmd_token_count = 4'd2;
        cmd_token_mask = 8'h02;
        cmd_vocab_rows = 18'd2;
        cmd_weight_fmt = 2'd1;
        cmd_emit_full_logits = 1'b0;
        reader_enable = 1'b1;
        result_ready = 1'b0;
        read_count_before = full_logit_count;
        issue_command();
        while (!result_valid) @(negedge clk);
        if (result_error || (result_token != 18'd0) ||
            (result_logit != 32'd0) ||
            (full_logit_count != read_count_before) ||
            (dut.logits_accepted != 32'd2) ||
            (accepted_weight_beats != 32'd80))
            $fatal(1, "LM exact restart result mismatch");
        result_ready = 1'b1;
        @(negedge clk);
        result_ready = 1'b0;
        while (!done_valid) @(negedge clk);
        if (done_error)
            $fatal(1, "LM exact restart failed");
        done_ready = 1'b1;
        @(negedge clk);
        done_ready = 1'b0;
        $display("projection service LM tie/full/restart PASS");

        $display(" projection_service_tb PASS");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "projection service timeout");
    end
endmodule

// Fast deterministic engine surrogate for service-control tests. The real
// projection engine has independent exact numeric/backpressure regression.
module projection_engine (
    input wire clk, input wire rst_n, input wire clear,
    input wire start, input wire [15:0] model_spec_k,
    input wire [17:0] model_spec_m, input wire [15:0] model_spec_rowblocks,
    input wire [1:0] weight_fmt, input wire signed [7:0] emin,
    input wire [7:0] token_mask, output reg busy, output reg done,
    output reg error, input wire [511:0] w_data, input wire w_valid,
    output wire w_ready, output wire act_req_valid,
    input wire act_req_ready, output wire [8:0] act_req_addr,
    output wire act_req_wave, input wire act_rsp_valid,
    output wire act_rsp_ready, input wire [1087:0] act_rsp_data,
    output wire signed [103:0] out_acc,
    output wire signed [7:0] out_emin, output wire [2:0] out_token,
    output wire [17:0] out_row, output wire out_last,
    output wire out_valid, input wire out_ready,
    output reg [31:0] weight_beat_count,
    output reg [31:0] wave_issue_count,
    output wire metrics_selector_full,
    output wire [2:0] metrics_selector_level,
    output wire metrics_drain,
    output wire metrics_bank_wait
);
    reg emit_q;
    reg [17:0] row_q;
    reg [17:0] m_q;
    reg [31:0] expected_beats_q;
    reg [1:0] act_requested_q;
    reg [1:0] act_received_q;
    integer scan;
    reg [2:0] first_token_q;

    assign w_ready = busy && !emit_q;
    wire [1:0] active_wave_mask = {(|token_mask[7:4]),
                                   (|token_mask[3:0])};
    wire request_wave0 = active_wave_mask[0] && !act_requested_q[0];
    wire request_wave1 = active_wave_mask[1] && !act_requested_q[1];
    wire selected_request_wave = !request_wave0 && request_wave1;
    wire response_wave0 = act_requested_q[0] && !act_received_q[0];
    wire response_wave1 = act_requested_q[1] && !act_received_q[1];
    wire selected_response_wave = !response_wave0 && response_wave1;
    assign act_req_valid = busy && (request_wave0 || request_wave1);
    assign act_req_addr = 9'd0;
    assign act_req_wave = selected_request_wave;
    assign act_rsp_ready = busy && (response_wave0 || response_wave1);
    assign out_valid = busy && emit_q;
    assign out_acc = 104'sd0;
    assign out_emin = emin;
    assign out_token = first_token_q;
    assign out_row = row_q;
    assign out_last = out_valid && (row_q + 1'b1 == m_q);
    assign metrics_selector_full = 1'b0;
    assign metrics_selector_level = 3'd0;
    assign metrics_drain = 1'b0;
    assign metrics_bank_wait = 1'b0;
    wire weight_fire = w_valid && w_ready;
    wire output_fire = out_valid && out_ready;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            emit_q <= 1'b0;
            row_q <= 18'd0;
            m_q <= 18'd0;
            expected_beats_q <= 32'd0;
            weight_beat_count <= 32'd0;
            wave_issue_count <= 32'd0;
            act_requested_q <= 2'b00;
            act_received_q <= 2'b00;
            first_token_q <= 3'd0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                emit_q <= 1'b0;
                row_q <= 18'd0;
                m_q <= model_spec_m;
                expected_beats_q <= model_spec_rowblocks *
                    {16'd0, (model_spec_k >> 7)} *
                    (weight_fmt == 2'd1 ? 5 : 9);
                weight_beat_count <= 32'd0;
                wave_issue_count <= 32'd0;
                act_requested_q <= 2'b00;
                act_received_q <= 2'b00;
                first_token_q <= 3'd0;
                for (scan = 7; scan >= 0; scan = scan - 1)
                    if (token_mask[scan]) first_token_q <= scan[2:0];
            end
            if (act_req_valid && act_req_ready) begin
                act_requested_q[selected_request_wave] <= 1'b1;
                wave_issue_count <= wave_issue_count + 1'b1;
            end
            if (act_rsp_valid && act_rsp_ready)
                act_received_q[selected_response_wave] <= 1'b1;
            if (weight_fire) begin
                weight_beat_count <= weight_beat_count + 1'b1;
                if (weight_beat_count + 1'b1 == expected_beats_q)
                    emit_q <= 1'b1;
            end
            if (output_fire) begin
                if (out_last) begin
                    emit_q <= 1'b0;
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    row_q <= row_q + 1'b1;
                end
            end
        end
    end
    wire unused = &{1'b0, w_data, act_rsp_data};
endmodule

`default_nettype wire
