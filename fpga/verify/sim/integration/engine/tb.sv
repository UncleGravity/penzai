`timescale 1ns/1ps
`default_nettype none

`include "engine_defs.vh"

module engine_core_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg run_clear = 1'b0;

    reg model_spec_clear_valid = 1'b0;
    wire model_spec_clear_ready;
    reg model_spec_begin_valid = 1'b0;
    wire model_spec_begin_ready;
    reg [31:0] model_spec_begin_id = `MODEL_BONSAI_1_7B;
    reg [63:0] model_spec_begin_hash = 64'h6d2a_8f41_c703_190b;
    reg [5:0] model_spec_begin_layer_count = 6'd28;
    reg [7:0] model_spec_begin_hidden_blocks = 8'd64;
    reg [9:0] model_spec_begin_ffn_blocks = 10'd192;
    reg [5:0] model_spec_begin_q_heads = 6'd16;
    reg [3:0] model_spec_begin_kv_heads = 4'd8;
    reg [7:0] model_spec_begin_head_dim = 8'd128;
    reg [1:0] model_spec_begin_weight_fmt = 2'd1;
    reg [16:0] model_spec_begin_context_limit = 17'd32768;
    reg [17:0] model_spec_begin_vocab_rows = `MODEL_BONSAI_VOCAB_ROWS;
    reg [63:0] model_spec_begin_embed_addr = 64'h0000_0001_0000_0000;
    reg [63:0] model_spec_begin_final_norm_addr = 64'h0000_0001_0100_0000;
    reg [63:0] model_spec_begin_lm_head_addr = 64'h0000_0001_0000_0000;
    reg [63:0] model_spec_begin_rope_table_addr = 64'h0000_0001_0400_0000;
    reg model_spec_layer_wr_valid = 1'b0;
    wire model_spec_layer_wr_ready;
    reg [5:0] model_spec_layer_wr_layer = 6'd0;
    reg [2:0] model_spec_layer_wr_word = 3'd0;
    reg [63:0] model_spec_layer_wr_data = 64'd0;
    reg model_spec_seal_valid = 1'b0;
    wire model_spec_seal_ready;
    wire model_spec_cfg_error_valid;
    reg model_spec_cfg_error_ready = 1'b0;
    wire [7:0] model_spec_cfg_error_code;
    wire [5:0] model_spec_cfg_error_layer;
    wire [2:0] model_spec_cfg_error_word;
    wire model_spec_loading;
    wire model_spec_sealed;
    wire [31:0] interface_version;
    wire [63:0] layer_layout_hash;
    wire [31:0] active_model_spec_id;
    wire [63:0] active_model_spec_hash;

    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [31:0] cmd_tag = 32'd0;
    reg [31:0] cmd_model_spec_id = `MODEL_BONSAI_1_7B;
    reg [63:0] cmd_model_spec_hash = 64'h6d2a_8f41_c703_190b;
    reg [3:0] cmd_token_count = 4'd1;
    reg [7:0] cmd_lane_mask = 8'h01;
    reg [255:0] cmd_token_ids = 256'd0;
    reg [16:0] cmd_position_base = 17'd0;
    reg [63:0] cmd_kv_base = 64'h0000_0002_0000_0000;
    reg [16:0] cmd_kv_capacity = 17'd128;
    reg cmd_emit_logits = 1'b1;

    wire commit_valid;
    reg commit_ready = 1'b0;
    wire [31:0] commit_tag;
    wire [31:0] commit_model_spec_id;
    wire [63:0] commit_model_spec_hash;
    wire [3:0] commit_token_count;
    wire [16:0] commit_kv_length;
    wire commit_logits_valid;
    wire error_valid;
    reg error_ready = 1'b0;
    wire [31:0] error_tag;
    wire [15:0] error_code;
    wire [7:0] error_detail;
    wire [5:0] error_layer;
    wire [4:0] error_stage;
    wire busy;
    wire [5:0] debug_layer;
    wire [4:0] debug_stage;
    wire metrics_stage_active;
    wire [4:0] metrics_stage;
    wire trace_valid;
    wire [5:0] trace_layer;
    wire [4:0] trace_stage;

    reg arena_r_wr_valid = 1'b0;
    wire arena_r_wr_ready;
    reg arena_r_wr_wave = 1'b0;
    reg [11:0] arena_r_wr_addr = 12'd0;
    reg [3:0] arena_r_wr_lane_mask = 4'd0;
    reg [127:0] arena_r_wr_data = 128'd0;
    reg arena_r_rd_req_valid = 1'b0;
    wire arena_r_rd_req_ready;
    reg arena_r_rd_req_wave = 1'b0;
    reg [11:0] arena_r_rd_req_addr = 12'd0;
    wire arena_r_rd_rsp_valid;
    reg arena_r_rd_rsp_ready = 1'b1;
    wire [127:0] arena_r_rd_rsp_data;

    reg arena_query_wr_valid = 1'b0;
    wire arena_query_wr_ready;
    reg arena_query_wr_wave = 1'b0;
    reg [11:0] arena_query_wr_addr = 12'd0;
    reg [3:0] arena_query_wr_lane_mask = 4'd0;
    reg [127:0] arena_query_wr_data = 128'd0;
    reg arena_query_rd_req_valid = 1'b0;
    wire arena_query_rd_req_ready;
    reg arena_query_rd_req_wave = 1'b0;
    reg [11:0] arena_query_rd_req_addr = 12'd0;
    wire arena_query_rd_rsp_valid;
    reg arena_query_rd_rsp_ready = 1'b1;
    wire [127:0] arena_query_rd_rsp_data;

    reg arena_q8_wr_valid = 1'b0;
    wire arena_q8_wr_ready;
    reg arena_q8_wr_wave = 1'b0;
    reg [8:0] arena_q8_wr_addr = 9'd0;
    reg [3:0] arena_q8_wr_lane_mask = 4'd0;
    reg [1087:0] arena_q8_wr_data = 1088'd0;
    reg arena_q8_rd_req_valid = 1'b0;
    wire arena_q8_rd_req_ready;
    reg arena_q8_rd_req_wave = 1'b0;
    reg [8:0] arena_q8_rd_req_addr = 9'd0;
    wire arena_q8_rd_rsp_valid;
    reg arena_q8_rd_rsp_ready = 1'b1;
    wire [1087:0] arena_q8_rd_rsp_data;

    reg arena_newkv_wr_valid = 1'b0;
    wire arena_newkv_wr_ready;
    reg arena_newkv_wr_wave = 1'b0;
    reg [10:0] arena_newkv_wr_addr = 11'd0;
    reg [3:0] arena_newkv_wr_lane_mask = 4'd0;
    reg [63:0] arena_newkv_wr_data = 64'd0;
    reg arena_newkv_rd_req_valid = 1'b0;
    wire arena_newkv_rd_req_ready;
    reg arena_newkv_rd_req_wave = 1'b0;
    reg [10:0] arena_newkv_rd_req_addr = 11'd0;
    wire arena_newkv_rd_rsp_valid;
    reg arena_newkv_rd_rsp_ready = 1'b1;
    wire [63:0] arena_newkv_rd_rsp_data;

    wire gemm_req_valid;
    wire gemm_req_ready;
    wire [2:0] gemm_req_op;
    wire [5:0] gemm_req_layer;
    wire [3:0] gemm_req_token_count;
    wire [7:0] gemm_req_lane_mask;
    wire [63:0] gemm_req_addr0;
    wire [63:0] gemm_req_addr1;
    wire [63:0] gemm_req_addr2;
    wire [63:0] gemm_req_addr3;
    wire [7:0] gemm_req_hidden_blocks;
    wire [9:0] gemm_req_ffn_blocks;
    wire [5:0] gemm_req_q_heads;
    wire [3:0] gemm_req_kv_heads;
    wire [7:0] gemm_req_head_dim;
    wire [16:0] gemm_req_position_base;
    wire [31:0] gemm_req_epsilon;
    wire [17:0] gemm_req_vocab_rows;
    wire [1:0] gemm_req_weight_fmt;
    wire gemm_done_valid;
    wire gemm_done_ready;
    wire gemm_done_error;
    wire [7:0] gemm_done_status;

    wire flash_req_valid;
    wire flash_req_ready;
    wire [5:0] flash_req_layer;
    wire [3:0] flash_req_token_count;
    wire [7:0] flash_req_lane_mask;
    wire [5:0] flash_req_q_heads;
    wire [3:0] flash_req_kv_heads;
    wire [7:0] flash_req_head_dim;
    wire [2:0] flash_req_head_group_count;
    wire [3:0] flash_req_group_q_heads;
    wire [3:0] flash_req_group_kv_heads;
    wire flash_req_kv_single_pass;
    wire [16:0] flash_req_position_base;
    wire [63:0] flash_req_kv_base;
    wire flash_done_valid;
    wire flash_done_ready;
    wire flash_done_error;
    wire [7:0] flash_done_status;

    wire vector_req_valid;
    wire vector_req_ready;
    wire [3:0] vector_req_op;
    wire [5:0] vector_req_layer;
    wire [3:0] vector_req_token_count;
    wire [7:0] vector_req_lane_mask;
    wire [255:0] vector_req_token_ids;
    wire [63:0] vector_req_addr0;
    wire [63:0] vector_req_addr1;
    wire [63:0] vector_req_addr2;
    wire [7:0] vector_req_hidden_blocks;
    wire [9:0] vector_req_ffn_blocks;
    wire [17:0] vector_req_vocab_rows;
    wire [1:0] vector_req_weight_fmt;
    wire [16:0] vector_req_position_base;
    wire [63:0] vector_req_kv_base;
    wire vector_done_valid;
    wire vector_done_ready;
    wire vector_done_error;
    wire [7:0] vector_done_status;

    reg gemm_fail_enable = 1'b0;
    reg flash_fail_enable = 1'b0;
    reg vector_fail_enable = 1'b0;
    reg [4:0] fail_stage = 5'd0;
    reg [7:0] fail_status = 8'ha5;

    integer cycle = 0;
    integer trace_count = 0;
    integer flash_kv_checks = 0;
    integer append_kv_checks = 0;
    reg trace_check = 1'b0;
    reg [5:0] expected_layer = 6'd0;
    reg [4:0] expected_stage = `ENGINE_STAGE_EMBED;

    always #1 clk = ~clk;

    always @(posedge clk) begin
        if (rst_n)
            cycle <= cycle + 1;
        if (rst_n && (cycle == 100000))
            $fatal(1, "test timeout r=%b/%b/%b q=%b/%b/%b q8=%b/%b/%b kv=%b/%b/%b state=%0d",
                   arena_r_rd_req_valid, arena_r_rd_req_ready,
                   arena_r_rd_rsp_valid, arena_query_rd_req_valid,
                   arena_query_rd_req_ready, arena_query_rd_rsp_valid,
                   arena_q8_rd_req_valid, arena_q8_rd_req_ready,
                   arena_q8_rd_rsp_valid, arena_newkv_rd_req_valid,
                   arena_newkv_rd_req_ready, arena_newkv_rd_rsp_valid,
                   dut.state_q);
        if (rst_n && trace_valid && trace_check) begin
            if ((trace_layer !== expected_layer) ||
                (trace_stage !== expected_stage))
                $fatal(1, "trace mismatch index=%0d got L%0d/S%0d expected L%0d/S%0d",
                       trace_count, trace_layer, trace_stage,
                       expected_layer, expected_stage);
            trace_count <= trace_count + 1;
            if (expected_stage == `ENGINE_STAGE_EMBED) begin
                expected_layer <= 6'd0;
                expected_stage <= `ENGINE_STAGE_ATTN_NORM;
            end else begin
                case (expected_stage)
                    `ENGINE_STAGE_ATTN_NORM:
                        expected_stage <= `ENGINE_STAGE_QKV_ROPE;
                    `ENGINE_STAGE_QKV_ROPE:
                        expected_stage <= `ENGINE_STAGE_KV_APPEND;
                    `ENGINE_STAGE_KV_APPEND:
                        expected_stage <= `ENGINE_STAGE_ATTENTION;
                    `ENGINE_STAGE_ATTENTION:
                        expected_stage <= `ENGINE_STAGE_O_PROJ_RESID;
                    `ENGINE_STAGE_O_PROJ_RESID:
                        expected_stage <= `ENGINE_STAGE_FFN_NORM;
                    `ENGINE_STAGE_FFN_NORM:
                        expected_stage <= `ENGINE_STAGE_GATE_UP_SWIGLU_Q8;
                    `ENGINE_STAGE_GATE_UP_SWIGLU_Q8:
                        expected_stage <= `ENGINE_STAGE_DOWN_RESID;
                    `ENGINE_STAGE_DOWN_RESID: begin
                        if (expected_layer + 1'b1 <
                            model_spec_begin_layer_count) begin
                            expected_layer <= expected_layer + 1'b1;
                            expected_stage <= `ENGINE_STAGE_ATTN_NORM;
                        end else begin
                            expected_stage <= `ENGINE_STAGE_FINAL_NORM;
                        end
                    end
                    `ENGINE_STAGE_FINAL_NORM:
                        expected_stage <= `ENGINE_STAGE_LM_HEAD;
                    default: ;
                endcase
            end
        end
        if (rst_n && gemm_req_valid && gemm_req_ready &&
            (gemm_req_op == `PROJECTION_OP_QKV) &&
            (gemm_req_addr3 != model_spec_begin_rope_table_addr))
            $fatal(1, "QKV sink did not receive the sealed RoPE table");
        if (rst_n && flash_req_valid && flash_req_ready) begin
            if (flash_req_kv_base != cmd_kv_base +
                (flash_req_layer * cmd_kv_capacity * 4096))
                $fatal(1, "Flash layer KV base mismatch layer=%0d got=%h",
                       flash_req_layer, flash_req_kv_base);
            flash_kv_checks <= flash_kv_checks + 1;
        end
        if (rst_n && vector_req_valid && vector_req_ready &&
            (vector_req_op == `VECTOR_OP_KV_APPEND)) begin
            if (vector_req_addr0 != cmd_kv_base +
                (vector_req_layer * cmd_kv_capacity * 4096))
                $fatal(1, "append layer KV base mismatch layer=%0d got=%h",
                       vector_req_layer, vector_req_addr0);
            append_kv_checks <= append_kv_checks + 1;
        end
    end

    engine_core dut (.*);

    leaf_stub #(.LATENCY(2)) gemm_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(gemm_req_valid), .req_ready(gemm_req_ready),
        .req_stage(debug_stage), .fail_enable(gemm_fail_enable),
        .fail_stage(fail_stage), .fail_status(fail_status),
        .done_valid(gemm_done_valid), .done_ready(gemm_done_ready),
        .done_error(gemm_done_error), .done_status(gemm_done_status)
    );
    leaf_stub #(.LATENCY(3)) flash_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(flash_req_valid), .req_ready(flash_req_ready),
        .req_stage(debug_stage), .fail_enable(flash_fail_enable),
        .fail_stage(fail_stage), .fail_status(fail_status),
        .done_valid(flash_done_valid), .done_ready(flash_done_ready),
        .done_error(flash_done_error), .done_status(flash_done_status)
    );
    leaf_stub #(.LATENCY(1)) vector_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(vector_req_valid), .req_ready(vector_req_ready),
        .req_stage(debug_stage), .fail_enable(vector_fail_enable),
        .fail_stage(fail_stage), .fail_status(fail_status),
        .done_valid(vector_done_valid), .done_ready(vector_done_ready),
        .done_error(vector_done_error), .done_status(vector_done_status)
    );

    task automatic load_model_spec(input integer layers, input integer words_per_layer);
        integer layer;
        integer word_index;
        begin
            model_spec_begin_layer_count = layers[5:0];
            @(negedge clk);
            model_spec_begin_valid = 1'b1;
            while (!model_spec_begin_ready)
                @(negedge clk);
            @(negedge clk);
            model_spec_begin_valid = 1'b0;
            if (!model_spec_loading)
                $fatal(1, "model_spec did not enter loading state");

            for (layer = 0; layer < layers; layer = layer + 1) begin
                for (word_index = 0; word_index < words_per_layer;
                     word_index = word_index + 1) begin
                    @(negedge clk);
                    model_spec_layer_wr_valid = 1'b1;
                    model_spec_layer_wr_layer = layer[5:0];
                    model_spec_layer_wr_word = word_index[2:0];
                    model_spec_layer_wr_data = 64'h0000_0003_0000_0000 +
                                            layer * 64'h0000_0000_0100_0000 +
                                            word_index * 64'h0000_0000_0001_0000;
                    while (!model_spec_layer_wr_ready)
                        @(negedge clk);
                end
            end
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b0;
        end
    endtask

    task automatic seal_model_spec;
        begin
            @(negedge clk);
            model_spec_seal_valid = 1'b1;
            while (!model_spec_seal_ready)
                @(negedge clk);
            @(negedge clk);
            model_spec_seal_valid = 1'b0;
        end
    endtask

    task automatic clear_model_spec;
        begin
            @(negedge clk);
            model_spec_clear_valid = 1'b1;
            while (!model_spec_clear_ready)
                @(negedge clk);
            @(negedge clk);
            model_spec_clear_valid = 1'b0;
            if (model_spec_sealed || model_spec_loading)
                $fatal(1, "model_spec clear did not invalidate state");
        end
    endtask

    task automatic begin_model_spec_only;
        begin
            @(negedge clk);
            model_spec_begin_valid = 1'b1;
            while (!model_spec_begin_ready)
                @(negedge clk);
            @(negedge clk);
            model_spec_begin_valid = 1'b0;
            if (!model_spec_loading || model_spec_sealed || model_spec_cfg_error_valid)
                $fatal(1, "canonical model_spec header was not accepted");
        end
    endtask

    task automatic expect_bad_model_spec_header;
        integer header_timeout;
        begin
            @(negedge clk);
            model_spec_begin_valid = 1'b1;
            while (!model_spec_begin_ready)
                @(negedge clk);
            @(negedge clk);
            model_spec_begin_valid = 1'b0;
            header_timeout = 0;
            while (!model_spec_cfg_error_valid && header_timeout < 20) begin
                @(negedge clk);
                header_timeout = header_timeout + 1;
            end
            if (!model_spec_cfg_error_valid ||
                (model_spec_cfg_error_code != `MODEL_SPEC_ERROR_BAD_HEADER) ||
                model_spec_loading || model_spec_sealed)
                $fatal(1, "noncanonical model_spec header was not quarantined");
            @(negedge clk);
            model_spec_cfg_error_ready = 1'b1;
            @(negedge clk);
            model_spec_cfg_error_ready = 1'b0;
        end
    endtask

    task automatic exercise_layer_write_staging;
        integer layer_timeout;
        reg [63:0] staged_value;
        reg [63:0] retry_value;
        begin
            staged_value = 64'h0000_1234_5678_9a00;
            retry_value = 64'h0000_0004_0000_0080;

            // The accepted payload, not live wrapper pins, owns the RAM write.
            begin_model_spec_only();
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b1;
            model_spec_layer_wr_layer = 6'd3;
            model_spec_layer_wr_word = 3'd5;
            model_spec_layer_wr_data = staged_value;
            if (!model_spec_layer_wr_ready)
                $fatal(1, "model_spec layer write was not accepted");
            @(posedge clk);
            #0.1;
            if (model_spec_layer_wr_ready)
                $fatal(1, "model_spec layer staging did not apply backpressure");
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b0;
            model_spec_layer_wr_layer = 6'd9;
            model_spec_layer_wr_word = 3'd1;
            model_spec_layer_wr_data = 64'hffff_ffff_ffff_ffff;
            @(posedge clk);
            #0.1;
            if (dut.u_model_spec.layer_table[9'd29] != staged_value)
                $fatal(1, "model_spec RAM write followed live payload pins");
            clear_model_spec();

            // A bad staged word quarantines its captured coordinates. Retry
            // after acknowledgment and clear must write a clean replacement.
            begin_model_spec_only();
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b1;
            model_spec_layer_wr_layer = 6'd2;
            model_spec_layer_wr_word = 3'd4;
            model_spec_layer_wr_data = 64'h0000_0004_0000_0041;
            if (!model_spec_layer_wr_ready)
                $fatal(1, "bad model_spec layer write was not accepted");
            @(posedge clk);
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b0;
            model_spec_layer_wr_layer = 6'd7;
            model_spec_layer_wr_word = 3'd0;
            model_spec_layer_wr_data = 64'h0000_0007_0000_0000;
            layer_timeout = 0;
            while (!model_spec_cfg_error_valid && layer_timeout < 20) begin
                @(negedge clk);
                layer_timeout = layer_timeout + 1;
            end
            if (!model_spec_cfg_error_valid ||
                (model_spec_cfg_error_code != `MODEL_SPEC_ERROR_BAD_LAYER_WORD) ||
                (model_spec_cfg_error_layer != 6'd2) ||
                (model_spec_cfg_error_word != 3'd4) || model_spec_loading)
                $fatal(1, "staged bad layer word was not quarantined exactly");
            @(negedge clk);
            model_spec_cfg_error_ready = 1'b1;
            @(negedge clk);
            model_spec_cfg_error_ready = 1'b0;
            clear_model_spec();

            begin_model_spec_only();
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b1;
            model_spec_layer_wr_layer = 6'd2;
            model_spec_layer_wr_word = 3'd4;
            model_spec_layer_wr_data = retry_value;
            while (!model_spec_layer_wr_ready)
                @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            model_spec_layer_wr_valid = 1'b0;
            model_spec_layer_wr_data = 64'd0;
            @(posedge clk);
            #0.1;
            if (dut.u_model_spec.layer_table[9'd20] != retry_value)
                $fatal(1, "model_spec layer retry did not commit captured payload");
            clear_model_spec();
        end
    endtask

    task automatic submit_command;
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            while (!cmd_ready)
                @(negedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic consume_error;
        begin
            @(negedge clk);
            error_ready = 1'b1;
            @(negedge clk);
            error_ready = 1'b0;
        end
    endtask

    integer timeout;
    initial begin
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        if ((interface_version != `ENGINE_INTERFACE_VERSION) ||
            (layer_layout_hash != `MODEL_LAYOUT_HASH))
            $fatal(1, "interface identity mismatch");

        exercise_layer_write_staging();

        // Every arena preserves lane isolation across both tile-8 waves.
        @(negedge clk);
        arena_r_wr_valid = 1'b1;
        arena_r_wr_wave = 1'b0;
        arena_r_wr_addr = 12'd19;
        arena_r_wr_lane_mask = 4'h1;
        arena_r_wr_data[31:0] = 32'h1122_3344;
        @(negedge clk);
        arena_r_wr_wave = 1'b1;
        arena_r_wr_lane_mask = 4'h8;
        arena_r_wr_data[127:96] = 32'haabb_ccdd;
        @(negedge clk);
        arena_r_wr_valid = 1'b0;
        arena_r_rd_rsp_ready = 1'b0;
        arena_r_rd_req_valid = 1'b1;
        arena_r_rd_req_wave = 1'b0;
        arena_r_rd_req_addr = 12'd19;
        #1ps;
        while (!arena_r_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_r_rd_req_valid = 1'b0;
        while (!arena_r_rd_rsp_valid) @(negedge clk);
        if (arena_r_rd_rsp_data[31:0] != 32'h1122_3344)
            $fatal(1, "resident R wave-0 banking mismatch");

        // The extra URAM output register holds a complete response while a
        // second request occupies the pending memory slot.
        @(negedge clk);
        arena_r_rd_req_valid = 1'b1;
        arena_r_rd_req_wave = 1'b1;
        #1ps;
        while (!arena_r_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_r_rd_req_valid = 1'b0;
        repeat (3) begin
            if (!arena_r_rd_rsp_valid ||
                (arena_r_rd_rsp_data[31:0] != 32'h1122_3344) ||
                arena_r_rd_req_ready)
                $fatal(1, "resident R response changed while stalled");
            @(negedge clk);
        end
        arena_r_rd_rsp_ready = 1'b1;
        @(negedge clk);
        if (!arena_r_rd_rsp_valid)
            $fatal(1, "resident R pending response was lost");
        if (arena_r_rd_rsp_data[127:96] != 32'haabb_ccdd)
            $fatal(1, "resident R wave-1 banking mismatch");
        @(negedge clk);

        // Global clear cancels a read between URAM issue and response publish.
        arena_r_rd_rsp_ready = 1'b0;
        arena_r_rd_req_valid = 1'b1;
        arena_r_rd_req_wave = 1'b0;
        #1ps;
        while (!arena_r_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_r_rd_req_valid = 1'b0;
        run_clear = 1'b1;
        @(negedge clk);
        run_clear = 1'b0;
        repeat (3) @(negedge clk);
        if (arena_r_rd_rsp_valid)
            $fatal(1, "resident R clear retained pipelined response");
        arena_r_rd_rsp_ready = 1'b1;
        $display("arena R pipeline PASS cycle=%0d", cycle);

        @(negedge clk);
        arena_query_wr_valid = 1'b1;
        arena_query_wr_wave = 1'b0;
        arena_query_wr_addr = 12'd20;
        arena_query_wr_lane_mask = 4'h2;
        arena_query_wr_data[63:32] = 32'h1357_9bdf;
        @(negedge clk);
        arena_query_wr_wave = 1'b1;
        arena_query_wr_lane_mask = 4'h4;
        arena_query_wr_data[95:64] = 32'h2468_ace0;
        @(negedge clk);
        arena_query_wr_valid = 1'b0;
        arena_query_rd_req_valid = 1'b1;
        arena_query_rd_req_wave = 1'b0;
        arena_query_rd_req_addr = 12'd20;
        #1ps;
        while (!arena_query_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_query_rd_req_valid = 1'b0;
        while (!arena_query_rd_rsp_valid) @(negedge clk);
        if (arena_query_rd_rsp_data[63:32] != 32'h1357_9bdf)
            $fatal(1, "query wave-0 banking mismatch");
        @(negedge clk);
        arena_query_rd_req_valid = 1'b1;
        arena_query_rd_req_wave = 1'b1;
        #1ps;
        while (!arena_query_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_query_rd_req_valid = 1'b0;
        while (!arena_query_rd_rsp_valid) @(negedge clk);
        if (arena_query_rd_rsp_data[95:64] != 32'h2468_ace0)
            $fatal(1, "query wave-1 banking mismatch");
        $display("arena Query pipeline PASS cycle=%0d", cycle);

        @(negedge clk);
        arena_q8_wr_valid = 1'b1;
        arena_q8_wr_wave = 1'b1;
        arena_q8_wr_addr = 9'd511;
        arena_q8_wr_lane_mask = 4'h4;
        arena_q8_wr_data[815:544] = {16'h3c00, 256'hcafe};
        @(negedge clk);
        arena_q8_wr_valid = 1'b0;
        arena_q8_rd_rsp_ready = 1'b0;
        arena_q8_rd_req_valid = 1'b1;
        arena_q8_rd_req_wave = 1'b1;
        arena_q8_rd_req_addr = 9'd511;
        #1ps;
        while (!arena_q8_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_q8_rd_req_valid = 1'b0;
        while (!arena_q8_rd_rsp_valid) @(negedge clk);
        if (arena_q8_rd_rsp_data[815:544] != {16'h3c00, 256'hcafe})
            $fatal(1, "Q8 token banking mismatch");
        arena_q8_rd_req_valid = 1'b1;
        #1ps;
        if (!arena_q8_rd_req_ready)
            $fatal(1, "Q8 memory slot did not accept behind held response");
        @(negedge clk);
        arena_q8_rd_req_valid = 1'b0;
        repeat (3) begin
            if (!arena_q8_rd_rsp_valid ||
                (arena_q8_rd_rsp_data[815:544] != {16'h3c00, 256'hcafe}) ||
                arena_q8_rd_req_ready)
                $fatal(1, "Q8 held response changed under backpressure");
            @(negedge clk);
        end
        run_clear = 1'b1;
        @(negedge clk);
        run_clear = 1'b0;
        repeat (2) @(negedge clk);
        if (arena_q8_rd_rsp_valid)
            $fatal(1, "Q8 clear retained held response");
        arena_q8_rd_rsp_ready = 1'b1;
        arena_q8_rd_req_valid = 1'b1;
        #1ps;
        while (!arena_q8_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_q8_rd_req_valid = 1'b0;
        while (!arena_q8_rd_rsp_valid) @(negedge clk);
        if (arena_q8_rd_rsp_data[815:544] != {16'h3c00, 256'hcafe})
            $fatal(1, "Q8 clear/restart response mismatch");
        $display("arena Q8 stall/clear/restart PASS cycle=%0d", cycle);

        @(negedge clk);
        arena_newkv_wr_valid = 1'b1;
        arena_newkv_wr_wave = 1'b1;
        arena_newkv_wr_addr = {1'b1, 3'd7, 7'd127};
        arena_newkv_wr_lane_mask = 4'h8;
        arena_newkv_wr_data[63:48] = 16'h3555;
        @(negedge clk);
        arena_newkv_wr_valid = 1'b0;
        arena_newkv_rd_rsp_ready = 1'b0;
        arena_newkv_rd_req_valid = 1'b1;
        arena_newkv_rd_req_wave = 1'b1;
        arena_newkv_rd_req_addr = {1'b1, 3'd7, 7'd127};
        #1ps;
        while (!arena_newkv_rd_req_ready) @(negedge clk);
        @(negedge clk);
        arena_newkv_rd_req_valid = 1'b0;
        while (!arena_newkv_rd_rsp_valid) @(negedge clk);
        if (arena_newkv_rd_rsp_data[63:48] != 16'h3555)
            $fatal(1, "NewKV token banking mismatch");
        repeat (3) begin
            if (!arena_newkv_rd_rsp_valid ||
                (arena_newkv_rd_rsp_data[63:48] != 16'h3555) ||
                arena_newkv_rd_req_ready)
                $fatal(1, "NewKV direct response changed under backpressure");
            @(negedge clk);
        end
        arena_newkv_rd_rsp_ready = 1'b1;
        @(negedge clk);
        $display("arena NewKV pipeline PASS cycle=%0d", cycle);

        // RoPE constants are an immutable host-supplied table, not RTL literals.
        model_spec_begin_rope_table_addr = 64'd0;
        @(negedge clk);
        model_spec_begin_valid = 1'b1;
        while (!model_spec_begin_ready)
            @(negedge clk);
        @(negedge clk);
        model_spec_begin_valid = 1'b0;
        timeout = 0;
        while (!model_spec_cfg_error_valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!model_spec_cfg_error_valid ||
            (model_spec_cfg_error_code != `MODEL_SPEC_ERROR_BAD_HEADER))
            $fatal(1, "missing RoPE table was not rejected");
        @(negedge clk);
        model_spec_cfg_error_ready = 1'b1;
        @(negedge clk);
        model_spec_cfg_error_ready = 1'b0;
        model_spec_begin_rope_table_addr = 64'h0000_0001_0400_0000;

        // Model-spec IDs name complete, immutable tuples. Reject every available
        // geometry/context/vocabulary field independently and reject a known
        // ID carrying another model_spec's otherwise-supported dimensions.
        model_spec_begin_id = 32'd99;
        expect_bad_model_spec_header();
        model_spec_begin_id = `MODEL_BONSAI_4B;
        expect_bad_model_spec_header();
        model_spec_begin_id = `MODEL_BONSAI_1_7B;

        model_spec_begin_layer_count = 6'd36;
        expect_bad_model_spec_header();
        model_spec_begin_layer_count = 6'd28;
        model_spec_begin_hidden_blocks = 8'd80;
        expect_bad_model_spec_header();
        model_spec_begin_hidden_blocks = 8'd64;
        model_spec_begin_ffn_blocks = 10'd304;
        expect_bad_model_spec_header();
        model_spec_begin_ffn_blocks = 10'd192;
        model_spec_begin_q_heads = 6'd32;
        expect_bad_model_spec_header();
        model_spec_begin_q_heads = 6'd16;
        model_spec_begin_kv_heads = 4'd7;
        expect_bad_model_spec_header();
        model_spec_begin_kv_heads = 4'd8;
        model_spec_begin_head_dim = 8'd64;
        expect_bad_model_spec_header();
        model_spec_begin_head_dim = 8'd128;
        model_spec_begin_context_limit = 17'd65536;
        expect_bad_model_spec_header();
        model_spec_begin_context_limit = 17'd32768;
        model_spec_begin_vocab_rows = `MODEL_BONSAI_VOCAB_ROWS - 1'b1;
        expect_bad_model_spec_header();
        model_spec_begin_vocab_rows = `MODEL_BONSAI_VOCAB_ROWS;

        // The canonical 4B tuple is accepted as one unit.
        model_spec_begin_id = `MODEL_BONSAI_4B;
        model_spec_begin_layer_count = 6'd36;
        model_spec_begin_hidden_blocks = 8'd80;
        model_spec_begin_ffn_blocks = 10'd304;
        model_spec_begin_q_heads = 6'd32;
        begin_model_spec_only();
        clear_model_spec();

        model_spec_begin_id = `MODEL_BONSAI_1_7B;
        model_spec_begin_layer_count = 6'd28;
        model_spec_begin_hidden_blocks = 8'd64;
        model_spec_begin_ffn_blocks = 10'd192;
        model_spec_begin_q_heads = 6'd16;
        load_model_spec(28, 8);
        seal_model_spec();
        if (!model_spec_sealed || model_spec_loading)
            $fatal(1, "tied-embedding model_spec was not sealed");
        if (model_spec_layer_wr_ready || model_spec_begin_ready)
            $fatal(1, "sealed model_spec remained mutable");

        // Reject a stale/tail-incoherent command without issuing any leaf work.
        cmd_tag = 32'hbad0_0001;
        cmd_token_count = 4'd3;
        cmd_lane_mask = 8'h0f;
        submit_command();
        timeout = 0;
        while (!error_valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!error_valid || error_code != `ENGINE_ERROR_LANE_MASK || commit_valid)
            $fatal(1, "invalid lane mask was not rejected atomically");
        consume_error();

        // KV records are 4 KiB and every layer reuses that exact stride.
        cmd_tag = 32'hbad0_0002;
        cmd_token_count = 4'd1;
        cmd_lane_mask = 8'h01;
        cmd_kv_base = 64'h0000_0002_0000_0040;
        submit_command();
        timeout = 0;
        while (!error_valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!error_valid || error_code != `ENGINE_ERROR_KV_BASE || commit_valid)
            $fatal(1, "unaligned KV arena was not rejected atomically");
        consume_error();
        cmd_kv_base = 64'h0000_0002_0000_0000;

        // KV capacity is the allocated per-layer stride, not the model_spec's
        // maximum context. Zero, oversized, and overrun commands fail closed.
        cmd_kv_capacity = 17'd0;
        submit_command();
        wait(error_valid);
        if (error_code != `ENGINE_ERROR_CONTEXT || commit_valid)
            $fatal(1, "zero KV capacity was not rejected");
        consume_error();
        cmd_kv_capacity = 17'd32769;
        submit_command();
        wait(error_valid);
        if (error_code != `ENGINE_ERROR_CONTEXT || commit_valid)
            $fatal(1, "capacity above model_spec context was not rejected");
        consume_error();
        cmd_kv_capacity = 17'd127;
        cmd_token_count = 4'd8;
        cmd_lane_mask = 8'hff;
        cmd_position_base = 17'd120;
        submit_command();
        wait(error_valid);
        if (error_code != `ENGINE_ERROR_CONTEXT || commit_valid)
            $fatal(1, "tile beyond allocated KV capacity was not rejected");
        consume_error();
        cmd_kv_capacity = 17'd128;
        cmd_token_count = 4'd1;
        cmd_lane_mask = 8'h01;
        cmd_position_base = 17'd0;

        // Runtime clear cancels only the active tile. The immutable model_spec
        // remains sealed and can be reused by the next command.
        cmd_tag = 32'hc1ea_0001;
        submit_command();
        repeat (2) @(negedge clk);
        run_clear = 1'b1;
        @(negedge clk);
        run_clear = 1'b0;
        repeat (5) @(negedge clk);
        if (busy || commit_valid || error_valid || !model_spec_sealed)
            $fatal(1, "runtime clear corrupted model_spec or published work");

        // Tile-8, 28 layers, logits: 1 + 28*8 + 2 = 227 service requests.
        cmd_tag = 32'h600d_0008;
        cmd_token_count = 4'd8;
        cmd_lane_mask = 8'hff;
        cmd_position_base = 17'd100;
        // A non-power-of-two compact allocation catches accidental use of the
        // model_spec context limit or a rounded per-layer stride.
        cmd_kv_capacity = 17'd127;
        cmd_emit_logits = 1'b1;
        trace_count = 0;
        flash_kv_checks = 0;
        append_kv_checks = 0;
        expected_layer = 6'd0;
        expected_stage = `ENGINE_STAGE_EMBED;
        trace_check = 1'b1;
        submit_command();
        timeout = 0;
        while (!commit_valid && timeout < 10000) begin
            @(negedge clk); timeout = timeout + 1;
        end
        trace_check = 1'b0;
        if (!commit_valid)
            $fatal(1, "valid EXEC_TILE did not commit");
        if ((trace_count != 227) || (commit_tag != 32'h600d_0008) ||
            (commit_token_count != 4'd8) || (commit_kv_length != 17'd108) ||
            !commit_logits_valid)
            $fatal(1, "EXEC_TILE commit metadata mismatch");
        if ((flash_kv_checks != 28) || (append_kv_checks != 28))
            $fatal(1, "compact KV address coverage flash=%0d append=%0d",
                   flash_kv_checks, append_kv_checks);
        if ((flash_req_group_q_heads != 4'd8) ||
            (flash_req_head_group_count != 3'd2) ||
            (flash_req_group_kv_heads != 4'd4) ||
            !flash_req_kv_single_pass)
            $fatal(1, "Flash eight-token head-group contract mismatch");
        if (model_spec_clear_ready)
            $fatal(1, "model_spec was mutable before commit acceptance");
        @(negedge clk);
        commit_ready = 1'b1;
        @(negedge clk);
        commit_ready = 1'b0;
        if (busy)
            $fatal(1, "engine stayed busy after commit acceptance");

        // Decode is the same path with one active lane, not another controller.
        cmd_tag = 32'hdec0_0001;
        cmd_token_count = 4'd1;
        cmd_lane_mask = 8'h01;
        cmd_position_base = 17'd108;
        trace_count = 0;
        expected_layer = 6'd0;
        expected_stage = `ENGINE_STAGE_EMBED;
        trace_check = 1'b1;
        submit_command();
        timeout = 0;
        while (!commit_valid && timeout < 10000) begin
            @(negedge clk); timeout = timeout + 1;
        end
        trace_check = 1'b0;
        if (!commit_valid || (trace_count != 227) ||
            (commit_tag != 32'hdec0_0001) ||
            (commit_token_count != 4'd1) || (commit_kv_length != 17'd109))
            $fatal(1, "single-token decode did not reuse the tile-8 execution path");
        @(negedge clk);
        commit_ready = 1'b1;
        @(negedge clk);
        commit_ready = 1'b0;

        // A missing layer word cannot produce a sealed model_spec.
        clear_model_spec();
        load_model_spec(28, 7);
        seal_model_spec();
        timeout = 0;
        while (!model_spec_cfg_error_valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!model_spec_cfg_error_valid ||
            model_spec_cfg_error_code != `MODEL_SPEC_ERROR_INCOMPLETE || model_spec_sealed)
            $fatal(1, "incomplete model_spec was not quarantined");
        @(negedge clk);
        model_spec_cfg_error_ready = 1'b1;
        @(negedge clk);
        model_spec_cfg_error_ready = 1'b0;
        clear_model_spec();

        // Canonical 8B has independent embedding, LM-head, and output norm
        // addresses. Address tying remains a host/model-image policy.
        model_spec_begin_id = `MODEL_BONSAI_8B;
        cmd_model_spec_id = `MODEL_BONSAI_8B;
        model_spec_begin_layer_count = 6'd36;
        model_spec_begin_hidden_blocks = 8'd128;
        model_spec_begin_ffn_blocks = 10'd384;
        model_spec_begin_q_heads = 6'd32;
        model_spec_begin_context_limit = 17'd65536;
        cmd_kv_capacity = 17'd65536;
        model_spec_begin_lm_head_addr = 64'h0000_0001_0200_0000;
        model_spec_begin_final_norm_addr = 64'h0000_0001_0300_0000;
        model_spec_begin_rope_table_addr = 64'h0000_0001_0500_0000;
        load_model_spec(36, 8);
        seal_model_spec();
        if (!model_spec_sealed || model_spec_loading)
            $fatal(1, "distinct-header 8B-like model_spec was not sealed");

        // The last legal tile-8 request reaches the 17-bit 65536 KV watermark.
        cmd_tag = 32'h8b00_fff8;
        cmd_token_count = 4'd8;
        cmd_lane_mask = 8'hff;
        cmd_position_base = 17'd65528;
        cmd_emit_logits = 1'b0;
        submit_command();
        timeout = 0;
        while (!commit_valid && timeout < 10000) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!commit_valid || (commit_kv_length != 17'd65536) ||
            (commit_token_count != 4'd8) || commit_logits_valid)
            $fatal(1, "final 8B context tile did not commit at 65536");
        @(negedge clk);
        commit_ready = 1'b1;
        @(negedge clk);
        commit_ready = 1'b0;

        // One token beyond the sealed context limit is rejected atomically.
        cmd_tag = 32'h8b01_fff9;
        cmd_position_base = 17'd65529;
        submit_command();
        timeout = 0;
        while (!error_valid && timeout < 20) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!error_valid || (error_code != `ENGINE_ERROR_CONTEXT) || commit_valid)
            $fatal(1, "65537 context end was not rejected atomically");
        consume_error();

        // A leaf failure returns an error and never publishes speculative KV.
        gemm_fail_enable = 1'b1;
        fail_stage = `ENGINE_STAGE_QKV_ROPE;
        cmd_tag = 32'hfa17_0003;
        cmd_token_count = 4'd1;
        cmd_lane_mask = 8'h01;
        cmd_position_base = 17'd7;
        submit_command();
        timeout = 0;
        while (!error_valid && timeout < 300) begin
            @(negedge clk); timeout = timeout + 1;
        end
        if (!error_valid || (error_code != `ENGINE_ERROR_LEAF) ||
            (error_stage != `ENGINE_STAGE_QKV_ROPE) ||
            (error_layer != 6'd0) || (error_detail != 8'ha5) || commit_valid)
            $fatal(1, "leaf fault escaped the transaction boundary");
        consume_error();
        gemm_fail_enable = 1'b0;

        $display("engine_core_tb PASS cycles=%0d trace=%0d", cycle, trace_count);
        $finish;
    end
endmodule

`default_nettype wire
