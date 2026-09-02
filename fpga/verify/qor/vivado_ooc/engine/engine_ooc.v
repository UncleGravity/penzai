`default_nettype none

`include "engine_defs.vh"

// Registered out-of-context shell. It self-loads the canonical 1.7B model_spec,
// touches
// every arena bank, executes one tile-8 request through local leaf stubs, and folds all
// terminal state into a registered result. No production GEMM RTL is required.
module engine_ooc (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start_valid,
    output wire        start_ready,
    input  wire [31:0] seed,
    output wire        result_valid,
    input  wire        result_ready,
    output wire        result_error,
    output wire [31:0] result_signature
);
    localparam [3:0] H_IDLE        = 4'd0;
    localparam [3:0] H_CLEAR       = 4'd1;
    localparam [3:0] H_BEGIN       = 4'd2;
    localparam [3:0] H_LAYER       = 4'd3;
    localparam [3:0] H_SEAL        = 4'd4;
    localparam [3:0] H_ARENA_WRITE = 4'd5;
    localparam [3:0] H_ARENA_READ  = 4'd6;
    localparam [3:0] H_ARENA_WAIT  = 4'd7;
    localparam [3:0] H_COMMAND     = 4'd8;
    localparam [3:0] H_EXECUTE     = 4'd9;
    localparam [3:0] H_RESULT      = 4'd10;
    localparam [3:0] H_ARENA_FOLD  = 4'd11;

    reg [3:0] state_q;
    reg [31:0] seed_q;
    reg [5:0] load_layer_q;
    reg [2:0] load_word_q;
    reg result_error_q;
    reg [31:0] result_signature_q;
    (* DONT_TOUCH = "yes" *) reg [127:0] arena_r_capture_q;
    (* DONT_TOUCH = "yes" *) reg [127:0] arena_query_capture_q;
    (* DONT_TOUCH = "yes" *) reg [1087:0] arena_q8_capture_q;
    (* DONT_TOUCH = "yes" *) reg [63:0] arena_newkv_capture_q;

    wire model_spec_clear_ready;
    wire model_spec_begin_ready;
    wire model_spec_layer_wr_ready;
    wire model_spec_seal_ready;
    wire model_spec_cfg_error_valid;
    wire model_spec_loading;
    wire model_spec_sealed;
    wire [31:0] active_model_spec_id;
    wire [63:0] active_model_spec_hash;
    wire [31:0] interface_version;
    wire [63:0] layer_layout_hash;

    wire cmd_ready;
    wire commit_valid;
    wire [31:0] commit_tag;
    wire [16:0] commit_kv_length;
    wire error_valid;
    wire [15:0] error_code;
    wire [7:0] error_detail;
    wire busy;
    wire [4:0] debug_stage;

    wire arena_r_rd_rsp_valid;
    wire arena_r_rd_req_ready;
    wire [127:0] arena_r_rd_rsp_data;
    wire arena_query_rd_rsp_valid;
    wire arena_query_rd_req_ready;
    wire [127:0] arena_query_rd_rsp_data;
    wire arena_q8_rd_rsp_valid;
    wire arena_q8_rd_req_ready;
    wire [1087:0] arena_q8_rd_rsp_data;
    wire arena_newkv_rd_rsp_valid;
    wire arena_newkv_rd_req_ready;
    wire [63:0] arena_newkv_rd_rsp_data;

    wire gemm_req_valid;
    wire gemm_req_ready;
    wire gemm_done_valid;
    wire gemm_done_ready;
    wire gemm_done_error;
    wire [7:0] gemm_done_status;
    wire flash_req_valid;
    wire flash_req_ready;
    wire flash_done_valid;
    wire flash_done_ready;
    wire flash_done_error;
    wire [7:0] flash_done_status;
    wire vector_req_valid;
    wire vector_req_ready;
    wire vector_done_valid;
    wire vector_done_ready;
    wire vector_done_error;
    wire [7:0] vector_done_status;

    wire [63:0] model_spec_hash_value = {seed_q, ~seed_q};
    wire [63:0] layer_word_value =
        64'h0000_0003_0000_0000 +
        {34'd0, load_layer_q, 24'd0} +
        {45'd0, load_word_q, 16'd0} +
        {48'd0, seed_q[15:6], 6'd0};
    wire all_arena_responses = arena_r_rd_rsp_valid &&
                               arena_query_rd_rsp_valid &&
                               arena_q8_rd_rsp_valid &&
                               arena_newkv_rd_rsp_valid;

    assign start_ready = state_q == H_IDLE;
    assign result_valid = state_q == H_RESULT;
    assign result_error = result_error_q;
    assign result_signature = result_signature_q;

    (* DONT_TOUCH = "yes" *) engine_core dut (
        .clk(clk), .rst_n(rst_n), .run_clear(1'b0),
        .model_spec_clear_valid(state_q == H_CLEAR),
        .model_spec_clear_ready(model_spec_clear_ready),
        .model_spec_begin_valid(state_q == H_BEGIN),
        .model_spec_begin_ready(model_spec_begin_ready),
        .model_spec_begin_id(`MODEL_BONSAI_1_7B),
        .model_spec_begin_hash(model_spec_hash_value),
        .model_spec_begin_layer_count(6'd28),
        .model_spec_begin_hidden_blocks(8'd64),
        .model_spec_begin_ffn_blocks(10'd192),
        .model_spec_begin_q_heads(6'd16),
        .model_spec_begin_kv_heads(4'd8),
        .model_spec_begin_head_dim(8'd128),
        .model_spec_begin_weight_fmt(seed_q[1] ? 2'd2 : 2'd1),
        .model_spec_begin_context_limit(17'd32768),
        .model_spec_begin_vocab_rows(`MODEL_BONSAI_VOCAB_ROWS),
        .model_spec_begin_embed_addr(64'h0000_0001_0000_0000 +
                                  {40'd0, seed_q[23:6], 6'd0}),
        .model_spec_begin_final_norm_addr(64'h0000_0001_0100_0000 +
                                       {40'd0, seed_q[23:6], 6'd0}),
        .model_spec_begin_lm_head_addr(64'h0000_0001_0200_0000 +
                                    {40'd0, seed_q[23:6], 6'd0}),
        .model_spec_begin_rope_table_addr(64'h0000_0001_0300_0000 +
                                       {40'd0, seed_q[23:6], 6'd0}),
        .model_spec_layer_wr_valid(state_q == H_LAYER),
        .model_spec_layer_wr_ready(model_spec_layer_wr_ready),
        .model_spec_layer_wr_layer(load_layer_q),
        .model_spec_layer_wr_word(load_word_q),
        .model_spec_layer_wr_data(layer_word_value),
        .model_spec_seal_valid(state_q == H_SEAL),
        .model_spec_seal_ready(model_spec_seal_ready),
        .model_spec_cfg_error_valid(model_spec_cfg_error_valid),
        .model_spec_cfg_error_ready(1'b1),
        .model_spec_loading(model_spec_loading), .model_spec_sealed(model_spec_sealed),
        .interface_version(interface_version),
        .layer_layout_hash(layer_layout_hash),
        .active_model_spec_id(active_model_spec_id),
        .active_model_spec_hash(active_model_spec_hash),

        .cmd_valid(state_q == H_COMMAND), .cmd_ready(cmd_ready),
        .cmd_tag(seed_q ^ 32'h4558_4543),
        .cmd_model_spec_id(`MODEL_BONSAI_1_7B),
        .cmd_model_spec_hash(model_spec_hash_value),
        .cmd_token_count(4'd8), .cmd_lane_mask(8'hff),
        .cmd_token_ids({8{seed_q}}),
        .cmd_position_base({6'd0, seed_q[10:0]}),
        .cmd_kv_base(64'h0000_0002_0000_0000 +
                     {40'd0, seed_q[23:12], 12'd0}),
        .cmd_kv_capacity(17'd32768),
        .cmd_emit_logits(1'b1),
        .commit_valid(commit_valid), .commit_ready(1'b1),
        .commit_tag(commit_tag), .commit_kv_length(commit_kv_length),
        .error_valid(error_valid), .error_ready(1'b1),
        .error_code(error_code), .error_detail(error_detail),
        .busy(busy), .debug_stage(debug_stage),

        .arena_r_wr_valid(state_q == H_ARENA_WRITE),
        .arena_r_wr_wave(seed_q[2]),
        .arena_r_wr_addr(seed_q[11:0]), .arena_r_wr_lane_mask(4'hf),
        .arena_r_wr_data({4{seed_q}}),
        .arena_r_rd_req_valid(state_q == H_ARENA_READ),
        .arena_r_rd_req_ready(arena_r_rd_req_ready),
        .arena_r_rd_req_wave(seed_q[2]),
        .arena_r_rd_req_addr(seed_q[11:0]),
        .arena_r_rd_rsp_valid(arena_r_rd_rsp_valid),
        .arena_r_rd_rsp_ready(all_arena_responses),
        .arena_r_rd_rsp_data(arena_r_rd_rsp_data),
        .arena_query_wr_valid(state_q == H_ARENA_WRITE),
        .arena_query_wr_wave(seed_q[2]),
        .arena_query_wr_addr(seed_q[11:0] ^ 12'h5a5),
        .arena_query_wr_lane_mask(4'hf),
        .arena_query_wr_data({4{~seed_q}}),
        .arena_query_rd_req_valid(state_q == H_ARENA_READ),
        .arena_query_rd_req_ready(arena_query_rd_req_ready),
        .arena_query_rd_req_wave(seed_q[2]),
        .arena_query_rd_req_addr(seed_q[11:0] ^ 12'h5a5),
        .arena_query_rd_rsp_valid(arena_query_rd_rsp_valid),
        .arena_query_rd_rsp_ready(all_arena_responses),
        .arena_query_rd_rsp_data(arena_query_rd_rsp_data),
        .arena_q8_wr_valid(state_q == H_ARENA_WRITE),
        .arena_q8_wr_wave(seed_q[2]),
        .arena_q8_wr_addr(seed_q[8:0]), .arena_q8_wr_lane_mask(4'hf),
        .arena_q8_wr_data({34{seed_q}}),
        .arena_q8_rd_req_valid(state_q == H_ARENA_READ),
        .arena_q8_rd_req_ready(arena_q8_rd_req_ready),
        .arena_q8_rd_req_wave(seed_q[2]),
        .arena_q8_rd_req_addr(seed_q[8:0]),
        .arena_q8_rd_rsp_valid(arena_q8_rd_rsp_valid),
        .arena_q8_rd_rsp_ready(all_arena_responses),
        .arena_q8_rd_rsp_data(arena_q8_rd_rsp_data),
        .arena_newkv_wr_valid(state_q == H_ARENA_WRITE),
        .arena_newkv_wr_wave(seed_q[2]),
        .arena_newkv_wr_addr(seed_q[10:0]),
        .arena_newkv_wr_lane_mask(4'hf),
        .arena_newkv_wr_data({2{seed_q}}),
        .arena_newkv_rd_req_valid(state_q == H_ARENA_READ),
        .arena_newkv_rd_req_ready(arena_newkv_rd_req_ready),
        .arena_newkv_rd_req_wave(seed_q[2]),
        .arena_newkv_rd_req_addr(seed_q[10:0]),
        .arena_newkv_rd_rsp_valid(arena_newkv_rd_rsp_valid),
        .arena_newkv_rd_rsp_ready(all_arena_responses),
        .arena_newkv_rd_rsp_data(arena_newkv_rd_rsp_data),

        .gemm_req_valid(gemm_req_valid), .gemm_req_ready(gemm_req_ready),
        .gemm_done_valid(gemm_done_valid), .gemm_done_ready(gemm_done_ready),
        .gemm_done_error(gemm_done_error),
        .gemm_done_status(gemm_done_status),
        .flash_req_valid(flash_req_valid), .flash_req_ready(flash_req_ready),
        .flash_done_valid(flash_done_valid),
        .flash_done_ready(flash_done_ready),
        .flash_done_error(flash_done_error),
        .flash_done_status(flash_done_status),
        .vector_req_valid(vector_req_valid),
        .vector_req_ready(vector_req_ready),
        .vector_done_valid(vector_done_valid),
        .vector_done_ready(vector_done_ready),
        .vector_done_error(vector_done_error),
        .vector_done_status(vector_done_status)
    );

    leaf_stub #(.LATENCY(3)) u_gemm_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(gemm_req_valid), .req_ready(gemm_req_ready),
        .req_stage(debug_stage), .fail_enable(1'b0), .fail_stage(5'd0),
        .fail_status(8'd0), .done_valid(gemm_done_valid),
        .done_ready(gemm_done_ready), .done_error(gemm_done_error),
        .done_status(gemm_done_status)
    );
    leaf_stub #(.LATENCY(5)) u_flash_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(flash_req_valid), .req_ready(flash_req_ready),
        .req_stage(debug_stage), .fail_enable(1'b0), .fail_stage(5'd0),
        .fail_status(8'd0), .done_valid(flash_done_valid),
        .done_ready(flash_done_ready), .done_error(flash_done_error),
        .done_status(flash_done_status)
    );
    leaf_stub #(.LATENCY(2)) u_vector_stub (
        .clk(clk), .rst_n(rst_n),
        .req_valid(vector_req_valid), .req_ready(vector_req_ready),
        .req_stage(debug_stage), .fail_enable(1'b0), .fail_stage(5'd0),
        .fail_status(8'd0), .done_valid(vector_done_valid),
        .done_ready(vector_done_ready), .done_error(vector_done_error),
        .done_status(vector_done_status)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= H_IDLE;
            seed_q <= 32'd0;
            load_layer_q <= 6'd0;
            load_word_q <= 3'd0;
            result_error_q <= 1'b0;
            result_signature_q <= 32'd0;
            arena_r_capture_q <= 128'd0;
            arena_query_capture_q <= 128'd0;
            arena_q8_capture_q <= 1088'd0;
            arena_newkv_capture_q <= 64'd0;
        end else begin
            case (state_q)
                H_IDLE: begin
                    if (start_valid && start_ready) begin
                        seed_q <= seed;
                        load_layer_q <= 6'd0;
                        load_word_q <= 3'd0;
                        result_error_q <= 1'b0;
                        state_q <= model_spec_sealed ? H_CLEAR : H_BEGIN;
                    end
                end
                H_CLEAR: begin
                    if (model_spec_clear_ready)
                        state_q <= H_BEGIN;
                end
                H_BEGIN: begin
                    if (model_spec_begin_ready)
                        state_q <= H_LAYER;
                end
                H_LAYER: begin
                    if (model_spec_layer_wr_ready) begin
                        if (load_word_q == 3'd7) begin
                            load_word_q <= 3'd0;
                            if (load_layer_q == 6'd27)
                                state_q <= H_SEAL;
                            else
                                load_layer_q <= load_layer_q + 1'b1;
                        end else begin
                            load_word_q <= load_word_q + 1'b1;
                        end
                    end
                end
                H_SEAL: begin
                    if (model_spec_seal_ready)
                        state_q <= H_ARENA_WRITE;
                end
                H_ARENA_WRITE: state_q <= H_ARENA_READ;
                H_ARENA_READ: begin
                    if (arena_r_rd_req_ready && arena_query_rd_req_ready &&
                        arena_q8_rd_req_ready && arena_newkv_rd_req_ready)
                        state_q <= H_ARENA_WAIT;
                end
                H_ARENA_WAIT: begin
                    if (all_arena_responses) begin
                        arena_r_capture_q <= arena_r_rd_rsp_data;
                        arena_query_capture_q <= arena_query_rd_rsp_data;
                        arena_q8_capture_q <= arena_q8_rd_rsp_data;
                        arena_newkv_capture_q <= arena_newkv_rd_rsp_data;
                        state_q <= H_ARENA_FOLD;
                    end
                end
                H_ARENA_FOLD: begin
                    result_signature_q <= seed_q ^ arena_r_capture_q[31:0] ^
                        arena_query_capture_q[63:32] ^
                        arena_q8_capture_q[815:784] ^
                        arena_newkv_capture_q[63:32] ^ interface_version ^
                        layer_layout_hash[31:0] ^ layer_layout_hash[63:32] ^
                        {30'd0, model_spec_loading, model_spec_cfg_error_valid};
                    state_q <= H_COMMAND;
                end
                H_COMMAND: begin
                    if (cmd_ready)
                        state_q <= H_EXECUTE;
                end
                H_EXECUTE: begin
                    if (commit_valid) begin
                        result_signature_q <= result_signature_q ^ commit_tag ^
                                              {15'd0, commit_kv_length};
                        result_error_q <= 1'b0;
                        state_q <= H_RESULT;
                    end else if (error_valid) begin
                        result_signature_q <= result_signature_q ^
                                              {error_code, error_detail, 8'd0};
                        result_error_q <= 1'b1;
                        state_q <= H_RESULT;
                    end
                end
                H_RESULT: begin
                    if (result_valid && result_ready)
                        state_q <= H_IDLE;
                end
                default: state_q <= H_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
