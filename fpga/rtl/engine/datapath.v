`default_nettype none

`include "engine_defs.vh"

// Whole-token datapath. There is one EXEC_TILE controller,
// one set of resident arenas, one projection fabric, and one shared Q8 leaf.
// External wrappers only adapt this contract to AXI-Lite, RPC, or platform
// clock/reset plumbing.
module engine_datapath #(
    parameter integer ADDR_W = 40,
    parameter integer EMIT_FULL_LOGITS = 0
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          run_clear,
    output wire          clear_done,

    input  wire          model_spec_clear_valid,
    output wire          model_spec_clear_ready,
    input  wire          model_spec_begin_valid,
    output wire          model_spec_begin_ready,
    input  wire [31:0]   model_spec_begin_id,
    input  wire [63:0]   model_spec_begin_hash,
    input  wire [5:0]    model_spec_begin_layer_count,
    input  wire [7:0]    model_spec_begin_hidden_blocks,
    input  wire [9:0]    model_spec_begin_ffn_blocks,
    input  wire [5:0]    model_spec_begin_q_heads,
    input  wire [3:0]    model_spec_begin_kv_heads,
    input  wire [7:0]    model_spec_begin_head_dim,
    input  wire [1:0]    model_spec_begin_weight_fmt,
    input  wire [16:0]   model_spec_begin_context_limit,
    input  wire [17:0]   model_spec_begin_vocab_rows,
    input  wire [63:0]   model_spec_begin_embed_addr,
    input  wire [63:0]   model_spec_begin_lm_head_addr,
    input  wire [63:0]   model_spec_begin_final_norm_addr,
    input  wire [63:0]   model_spec_begin_rope_table_addr,
    input  wire          model_spec_layer_wr_valid,
    output wire          model_spec_layer_wr_ready,
    input  wire [5:0]    model_spec_layer_wr_layer,
    input  wire [2:0]    model_spec_layer_wr_word,
    input  wire [63:0]   model_spec_layer_wr_data,
    input  wire          model_spec_seal_valid,
    output wire          model_spec_seal_ready,
    output wire          model_spec_cfg_error_valid,
    input  wire          model_spec_cfg_error_ready,
    output wire [7:0]    model_spec_cfg_error_code,
    output wire [5:0]    model_spec_cfg_error_layer,
    output wire [2:0]    model_spec_cfg_error_word,
    output wire          model_spec_loading,
    output wire          model_spec_sealed,
    output wire [31:0]   interface_version,
    output wire [63:0]   layer_layout_hash,
    output wire [31:0]   active_model_spec_id,
    output wire [63:0]   active_model_spec_hash,

    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [31:0]   cmd_tag,
    input  wire [31:0]   cmd_model_spec_id,
    input  wire [63:0]   cmd_model_spec_hash,
    input  wire [3:0]    cmd_token_count,
    input  wire [7:0]    cmd_lane_mask,
    input  wire [255:0]  cmd_token_ids,
    input  wire [16:0]   cmd_position_base,
    input  wire [63:0]   cmd_kv_base,
    input  wire [16:0]   cmd_kv_capacity,
    input  wire          cmd_emit_logits,

    output wire          commit_valid,
    input  wire          commit_ready,
    output wire [31:0]   commit_tag,
    output wire [31:0]   commit_model_spec_id,
    output wire [63:0]   commit_model_spec_hash,
    output wire [3:0]    commit_token_count,
    output wire [16:0]   commit_kv_length,
    output wire          commit_logits_valid,

    output wire          error_valid,
    input  wire          error_ready,
    output wire [31:0]   error_tag,
    output wire [15:0]   error_code,
    output wire [7:0]    error_detail,
    output wire [5:0]    error_layer,
    output wire [4:0]    error_stage,

    output wire          logits_valid,
    input  wire          logits_ready,
    output wire [17:0]   logits_row,
    output wire [31:0]   logits_data,
    output wire          logits_last,
    output wire          result_valid,
    input  wire          result_ready,
    output wire [17:0]   result_token,
    output wire [31:0]   result_logit,
    output wire          result_error,
    output wire [7:0]    result_status,

    output wire          busy,
    output wire [5:0]    debug_layer,
    output wire [4:0]    debug_stage,
    output wire          trace_valid,
    output wire [5:0]    trace_layer,
    output wire [4:0]    trace_stage,
    output wire          protocol_error,

    output wire          metrics_stage_active,
    output wire [4:0]    metrics_stage,
    output wire [12:0]   metrics_projection_probe,
    output wire [2:0]    metrics_weight_axi_r_beats,
    output wire [2:0]    metrics_weight_axi_r_gap_ports,
    output wire          metrics_weight_zip_skew,
    output reg  [1:0]    metrics_history_axi_r_beats,
    output reg           metrics_kv_axi_w_beat,

    // Four lockstep packed-weight/table readers.
    output wire [4*ADDR_W-1:0] weight_axi_araddr,
    output wire [31:0]         weight_axi_arlen,
    output wire [11:0]         weight_axi_arsize,
    output wire [7:0]          weight_axi_arburst,
    output wire [3:0]          weight_axi_arvalid,
    input  wire [3:0]          weight_axi_arready,
    input  wire [511:0]        weight_axi_rdata,
    input  wire [7:0]          weight_axi_rresp,
    input  wire [3:0]          weight_axi_rlast,
    input  wire [3:0]          weight_axi_rvalid,
    output wire [3:0]          weight_axi_rready,

    // Independent committed-K history reader.
    output wire [ADDR_W-1:0] hist_k_axi_araddr,
    output wire [7:0]        hist_k_axi_arlen,
    output wire [2:0]        hist_k_axi_arsize,
    output wire [1:0]        hist_k_axi_arburst,
    output wire              hist_k_axi_arvalid,
    input  wire              hist_k_axi_arready,
    input  wire [127:0]      hist_k_axi_rdata,
    input  wire [1:0]        hist_k_axi_rresp,
    input  wire              hist_k_axi_rlast,
    input  wire              hist_k_axi_rvalid,
    output wire              hist_k_axi_rready,

    // Independent committed-V history reader.
    output wire [ADDR_W-1:0] hist_v_axi_araddr,
    output wire [7:0]        hist_v_axi_arlen,
    output wire [2:0]        hist_v_axi_arsize,
    output wire [1:0]        hist_v_axi_arburst,
    output wire              hist_v_axi_arvalid,
    input  wire              hist_v_axi_arready,
    input  wire [127:0]      hist_v_axi_rdata,
    input  wire [1:0]        hist_v_axi_rresp,
    input  wire              hist_v_axi_rlast,
    input  wire              hist_v_axi_rvalid,
    output wire              hist_v_axi_rready,

    // Committed-KV writer.
    output wire [ADDR_W-1:0] kv_axi_awaddr,
    output wire [7:0]        kv_axi_awlen,
    output wire [2:0]        kv_axi_awsize,
    output wire [1:0]        kv_axi_awburst,
    output wire              kv_axi_awvalid,
    input  wire              kv_axi_awready,
    output wire [127:0]      kv_axi_wdata,
    output wire [15:0]       kv_axi_wstrb,
    output wire              kv_axi_wlast,
    output wire              kv_axi_wvalid,
    input  wire              kv_axi_wready,
    input  wire [1:0]        kv_axi_bresp,
    input  wire              kv_axi_bvalid,
    output wire              kv_axi_bready
);
    localparam [1:0] VOWNER_NONE   = 2'd0;
    localparam [1:0] VOWNER_EMBED  = 2'd1;
    localparam [1:0] VOWNER_NORM   = 2'd2;
    localparam [1:0] VOWNER_APPEND = 2'd3;

    // RUN_CLEAR is an infrequent control-plane event. Preserve one registered
    // copy per physical island so the MMIO bit never becomes a global CE/D
    // control tree across the full token engine.
    (* keep = "true", dont_touch = "true" *) reg clear_engine_q;
    (* keep = "true", dont_touch = "true" *) reg clear_owner_q;
    (* keep = "true", dont_touch = "true" *) reg clear_vector_q;
    (* keep = "true", dont_touch = "true" *) reg clear_embedding_q;
    (* keep = "true", dont_touch = "true" *) reg clear_cluster_q;
    (* keep = "true", dont_touch = "true" *) reg clear_projection_q;
    (* keep = "true", dont_touch = "true" *) reg clear_append_q;
    (* keep = "true", dont_touch = "true" *) reg clear_kv_writer_q;
    (* keep = "true", dont_touch = "true" *) reg clear_attention_q;
    (* keep = "true", dont_touch = "true" *) reg clear_history_q;
    (* keep = "true", dont_touch = "true" *) reg clear_small_q;
    (* keep = "true", dont_touch = "true" *) reg clear_rope_q;
    (* keep = "true", dont_touch = "true" *) reg clear_arbiter_q;
    (* keep = "true", dont_touch = "true" *) reg clear_weight_q;
    (* keep = "true", dont_touch = "true" *) reg clear_seen_q;
    (* keep = "true", dont_touch = "true" *) reg runtime_quiescent_q;
    (* keep = "true", dont_touch = "true" *) reg clear_done_q;

    wire clear_event = run_clear && !clear_seen_q;
    wire clear_pulse_active = clear_engine_q || clear_owner_q ||
        clear_vector_q || clear_embedding_q || clear_cluster_q ||
        clear_projection_q || clear_append_q || clear_kv_writer_q ||
        clear_attention_q || clear_history_q || clear_small_q ||
        clear_rope_q || clear_arbiter_q || clear_weight_q;
    wire clear_distribution_settled = !clear_pulse_active &&
        (run_clear ? clear_seen_q : !clear_seen_q);

    always @(posedge clk) begin
        if (!rst_n) begin
            clear_engine_q <= 1'b0;
            clear_owner_q <= 1'b0;
            clear_vector_q <= 1'b0;
            clear_embedding_q <= 1'b0;
            clear_cluster_q <= 1'b0;
            clear_projection_q <= 1'b0;
            clear_append_q <= 1'b0;
            clear_kv_writer_q <= 1'b0;
            clear_attention_q <= 1'b0;
            clear_history_q <= 1'b0;
            clear_small_q <= 1'b0;
            clear_rope_q <= 1'b0;
            clear_arbiter_q <= 1'b0;
            clear_weight_q <= 1'b0;
            clear_seen_q <= 1'b0;
        end else begin
            clear_engine_q <= clear_event;
            clear_owner_q <= clear_event;
            clear_vector_q <= clear_event;
            clear_embedding_q <= clear_event;
            clear_cluster_q <= clear_event;
            clear_projection_q <= clear_event;
            clear_append_q <= clear_event;
            clear_kv_writer_q <= clear_event;
            clear_attention_q <= clear_event;
            clear_history_q <= clear_event;
            clear_small_q <= clear_event;
            clear_rope_q <= clear_event;
            clear_arbiter_q <= clear_event;
            clear_weight_q <= clear_event;
            clear_seen_q <= run_clear;
        end
    end

    wire engine_cmd_ready;
    wire runtime_ready = clear_done && !run_clear;
    assign cmd_ready = engine_cmd_ready && runtime_ready;

    // Controller leaf contracts.
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
    wire [16:0] vector_req_position_base;
    wire [63:0] vector_req_kv_base;
    wire [17:0] vector_req_vocab_rows;
    wire [1:0] vector_req_weight_fmt;
    wire vector_done_valid;
    wire vector_done_ready;
    wire vector_done_error;
    wire [7:0] vector_done_status;

    // Resident arena boundary owned by the engine instance.
    wire arena_r_wr_valid;
    wire arena_r_wr_ready;
    wire arena_r_wr_wave;
    wire [11:0] arena_r_wr_addr;
    wire [3:0] arena_r_wr_lane_mask;
    wire [127:0] arena_r_wr_data;
    wire arena_r_rd_req_valid;
    wire arena_r_rd_req_ready;
    wire arena_r_rd_req_wave;
    wire [11:0] arena_r_rd_req_addr;
    wire arena_r_rd_rsp_valid;
    wire arena_r_rd_rsp_ready;
    wire [127:0] arena_r_rd_rsp_data;
    wire arena_query_wr_valid;
    wire arena_query_wr_ready;
    wire arena_query_wr_wave;
    wire [11:0] arena_query_wr_addr;
    wire [3:0] arena_query_wr_lane_mask;
    wire [127:0] arena_query_wr_data;
    wire arena_query_rd_req_valid;
    wire arena_query_rd_req_ready;
    wire arena_query_rd_req_wave;
    wire [11:0] arena_query_rd_req_addr;
    wire arena_query_rd_rsp_valid;
    wire arena_query_rd_rsp_ready;
    wire [127:0] arena_query_rd_rsp_data;
    wire arena_q8_wr_valid;
    wire arena_q8_wr_ready;
    wire arena_q8_wr_wave;
    wire [8:0] arena_q8_wr_addr;
    wire [3:0] arena_q8_wr_lane_mask;
    wire [1087:0] arena_q8_wr_data;
    wire arena_q8_rd_req_valid;
    wire arena_q8_rd_req_ready;
    wire arena_q8_rd_req_wave;
    wire [8:0] arena_q8_rd_req_addr;
    wire arena_q8_rd_rsp_valid;
    wire arena_q8_rd_rsp_ready;
    wire [1087:0] arena_q8_rd_rsp_data;
    wire arena_newkv_wr_valid;
    wire arena_newkv_wr_ready;
    wire arena_newkv_wr_wave;
    wire [10:0] arena_newkv_wr_addr;
    wire [3:0] arena_newkv_wr_lane_mask;
    wire [63:0] arena_newkv_wr_data;
    wire arena_newkv_rd_req_valid;
    wire arena_newkv_rd_req_ready;
    wire arena_newkv_rd_req_wave;
    wire [10:0] arena_newkv_rd_req_addr;
    wire arena_newkv_rd_rsp_valid;
    wire arena_newkv_rd_rsp_ready;
    wire [63:0] arena_newkv_rd_rsp_data;

    engine_core u_engine (
        .clk(clk), .rst_n(rst_n), .run_clear(clear_engine_q),
        .model_spec_clear_valid(model_spec_clear_valid),
        .model_spec_clear_ready(model_spec_clear_ready),
        .model_spec_begin_valid(model_spec_begin_valid),
        .model_spec_begin_ready(model_spec_begin_ready),
        .model_spec_begin_id(model_spec_begin_id),
        .model_spec_begin_hash(model_spec_begin_hash),
        .model_spec_begin_layer_count(model_spec_begin_layer_count),
        .model_spec_begin_hidden_blocks(model_spec_begin_hidden_blocks),
        .model_spec_begin_ffn_blocks(model_spec_begin_ffn_blocks),
        .model_spec_begin_q_heads(model_spec_begin_q_heads),
        .model_spec_begin_kv_heads(model_spec_begin_kv_heads),
        .model_spec_begin_head_dim(model_spec_begin_head_dim),
        .model_spec_begin_weight_fmt(model_spec_begin_weight_fmt),
        .model_spec_begin_context_limit(model_spec_begin_context_limit),
        .model_spec_begin_vocab_rows(model_spec_begin_vocab_rows),
        .model_spec_begin_embed_addr(model_spec_begin_embed_addr),
        .model_spec_begin_lm_head_addr(model_spec_begin_lm_head_addr),
        .model_spec_begin_final_norm_addr(model_spec_begin_final_norm_addr),
        .model_spec_begin_rope_table_addr(model_spec_begin_rope_table_addr),
        .model_spec_layer_wr_valid(model_spec_layer_wr_valid),
        .model_spec_layer_wr_ready(model_spec_layer_wr_ready),
        .model_spec_layer_wr_layer(model_spec_layer_wr_layer),
        .model_spec_layer_wr_word(model_spec_layer_wr_word),
        .model_spec_layer_wr_data(model_spec_layer_wr_data),
        .model_spec_seal_valid(model_spec_seal_valid),
        .model_spec_seal_ready(model_spec_seal_ready),
        .model_spec_cfg_error_valid(model_spec_cfg_error_valid),
        .model_spec_cfg_error_ready(model_spec_cfg_error_ready),
        .model_spec_cfg_error_code(model_spec_cfg_error_code),
        .model_spec_cfg_error_layer(model_spec_cfg_error_layer),
        .model_spec_cfg_error_word(model_spec_cfg_error_word),
        .model_spec_loading(model_spec_loading), .model_spec_sealed(model_spec_sealed),
        .interface_version(interface_version),
        .layer_layout_hash(layer_layout_hash),
        .active_model_spec_id(active_model_spec_id),
        .active_model_spec_hash(active_model_spec_hash),
        .cmd_valid(cmd_valid && runtime_ready), .cmd_ready(engine_cmd_ready),
        .cmd_tag(cmd_tag), .cmd_model_spec_id(cmd_model_spec_id),
        .cmd_model_spec_hash(cmd_model_spec_hash),
        .cmd_token_count(cmd_token_count), .cmd_lane_mask(cmd_lane_mask),
        .cmd_token_ids(cmd_token_ids), .cmd_position_base(cmd_position_base),
        .cmd_kv_base(cmd_kv_base), .cmd_kv_capacity(cmd_kv_capacity),
        .cmd_emit_logits(cmd_emit_logits),
        .commit_valid(commit_valid), .commit_ready(commit_ready),
        .commit_tag(commit_tag), .commit_model_spec_id(commit_model_spec_id),
        .commit_model_spec_hash(commit_model_spec_hash),
        .commit_token_count(commit_token_count),
        .commit_kv_length(commit_kv_length),
        .commit_logits_valid(commit_logits_valid),
        .error_valid(error_valid), .error_ready(error_ready),
        .error_tag(error_tag), .error_code(error_code),
        .error_detail(error_detail), .error_layer(error_layer),
        .error_stage(error_stage), .busy(busy),
        .debug_layer(debug_layer), .debug_stage(debug_stage),
        .metrics_stage_active(metrics_stage_active),
        .metrics_stage(metrics_stage),
        .trace_valid(trace_valid), .trace_layer(trace_layer),
        .trace_stage(trace_stage),
        .arena_r_wr_valid(arena_r_wr_valid),
        .arena_r_wr_ready(arena_r_wr_ready),
        .arena_r_wr_wave(arena_r_wr_wave), .arena_r_wr_addr(arena_r_wr_addr),
        .arena_r_wr_lane_mask(arena_r_wr_lane_mask),
        .arena_r_wr_data(arena_r_wr_data),
        .arena_r_rd_req_valid(arena_r_rd_req_valid),
        .arena_r_rd_req_ready(arena_r_rd_req_ready),
        .arena_r_rd_req_wave(arena_r_rd_req_wave),
        .arena_r_rd_req_addr(arena_r_rd_req_addr),
        .arena_r_rd_rsp_valid(arena_r_rd_rsp_valid),
        .arena_r_rd_rsp_ready(arena_r_rd_rsp_ready),
        .arena_r_rd_rsp_data(arena_r_rd_rsp_data),
        .arena_query_wr_valid(arena_query_wr_valid),
        .arena_query_wr_ready(arena_query_wr_ready),
        .arena_query_wr_wave(arena_query_wr_wave),
        .arena_query_wr_addr(arena_query_wr_addr),
        .arena_query_wr_lane_mask(arena_query_wr_lane_mask),
        .arena_query_wr_data(arena_query_wr_data),
        .arena_query_rd_req_valid(arena_query_rd_req_valid),
        .arena_query_rd_req_ready(arena_query_rd_req_ready),
        .arena_query_rd_req_wave(arena_query_rd_req_wave),
        .arena_query_rd_req_addr(arena_query_rd_req_addr),
        .arena_query_rd_rsp_valid(arena_query_rd_rsp_valid),
        .arena_query_rd_rsp_ready(arena_query_rd_rsp_ready),
        .arena_query_rd_rsp_data(arena_query_rd_rsp_data),
        .arena_q8_wr_valid(arena_q8_wr_valid),
        .arena_q8_wr_ready(arena_q8_wr_ready),
        .arena_q8_wr_wave(arena_q8_wr_wave),
        .arena_q8_wr_addr(arena_q8_wr_addr),
        .arena_q8_wr_lane_mask(arena_q8_wr_lane_mask),
        .arena_q8_wr_data(arena_q8_wr_data),
        .arena_q8_rd_req_valid(arena_q8_rd_req_valid),
        .arena_q8_rd_req_ready(arena_q8_rd_req_ready),
        .arena_q8_rd_req_wave(arena_q8_rd_req_wave),
        .arena_q8_rd_req_addr(arena_q8_rd_req_addr),
        .arena_q8_rd_rsp_valid(arena_q8_rd_rsp_valid),
        .arena_q8_rd_rsp_ready(arena_q8_rd_rsp_ready),
        .arena_q8_rd_rsp_data(arena_q8_rd_rsp_data),
        .arena_newkv_wr_valid(arena_newkv_wr_valid),
        .arena_newkv_wr_ready(arena_newkv_wr_ready),
        .arena_newkv_wr_wave(arena_newkv_wr_wave),
        .arena_newkv_wr_addr(arena_newkv_wr_addr),
        .arena_newkv_wr_lane_mask(arena_newkv_wr_lane_mask),
        .arena_newkv_wr_data(arena_newkv_wr_data),
        .arena_newkv_rd_req_valid(arena_newkv_rd_req_valid),
        .arena_newkv_rd_req_ready(arena_newkv_rd_req_ready),
        .arena_newkv_rd_req_wave(arena_newkv_rd_req_wave),
        .arena_newkv_rd_req_addr(arena_newkv_rd_req_addr),
        .arena_newkv_rd_rsp_valid(arena_newkv_rd_rsp_valid),
        .arena_newkv_rd_rsp_ready(arena_newkv_rd_rsp_ready),
        .arena_newkv_rd_rsp_data(arena_newkv_rd_rsp_data),
        .gemm_req_valid(gemm_req_valid), .gemm_req_ready(gemm_req_ready),
        .gemm_req_op(gemm_req_op), .gemm_req_layer(gemm_req_layer),
        .gemm_req_token_count(gemm_req_token_count),
        .gemm_req_lane_mask(gemm_req_lane_mask),
        .gemm_req_addr0(gemm_req_addr0), .gemm_req_addr1(gemm_req_addr1),
        .gemm_req_addr2(gemm_req_addr2), .gemm_req_addr3(gemm_req_addr3),
        .gemm_req_hidden_blocks(gemm_req_hidden_blocks),
        .gemm_req_ffn_blocks(gemm_req_ffn_blocks),
        .gemm_req_q_heads(gemm_req_q_heads),
        .gemm_req_kv_heads(gemm_req_kv_heads),
        .gemm_req_head_dim(gemm_req_head_dim),
        .gemm_req_position_base(gemm_req_position_base),
        .gemm_req_epsilon(gemm_req_epsilon),
        .gemm_req_vocab_rows(gemm_req_vocab_rows),
        .gemm_req_weight_fmt(gemm_req_weight_fmt),
        .gemm_done_valid(gemm_done_valid),
        .gemm_done_ready(gemm_done_ready),
        .gemm_done_error(gemm_done_error),
        .gemm_done_status(gemm_done_status),
        .flash_req_valid(flash_req_valid),
        .flash_req_ready(flash_req_ready),
        .flash_req_layer(flash_req_layer),
        .flash_req_token_count(flash_req_token_count),
        .flash_req_lane_mask(flash_req_lane_mask),
        .flash_req_q_heads(flash_req_q_heads),
        .flash_req_kv_heads(flash_req_kv_heads),
        .flash_req_head_dim(flash_req_head_dim),
        .flash_req_head_group_count(flash_req_head_group_count),
        .flash_req_group_q_heads(flash_req_group_q_heads),
        .flash_req_group_kv_heads(flash_req_group_kv_heads),
        .flash_req_kv_single_pass(flash_req_kv_single_pass),
        .flash_req_position_base(flash_req_position_base),
        .flash_req_kv_base(flash_req_kv_base),
        .flash_done_valid(flash_done_valid),
        .flash_done_ready(flash_done_ready),
        .flash_done_error(flash_done_error),
        .flash_done_status(flash_done_status),
        .vector_req_valid(vector_req_valid),
        .vector_req_ready(vector_req_ready),
        .vector_req_op(vector_req_op), .vector_req_layer(vector_req_layer),
        .vector_req_token_count(vector_req_token_count),
        .vector_req_lane_mask(vector_req_lane_mask),
        .vector_req_token_ids(vector_req_token_ids),
        .vector_req_addr0(vector_req_addr0),
        .vector_req_addr1(vector_req_addr1),
        .vector_req_addr2(vector_req_addr2),
        .vector_req_hidden_blocks(vector_req_hidden_blocks),
        .vector_req_ffn_blocks(vector_req_ffn_blocks),
        .vector_req_position_base(vector_req_position_base),
        .vector_req_kv_base(vector_req_kv_base),
        .vector_req_vocab_rows(vector_req_vocab_rows),
        .vector_req_weight_fmt(vector_req_weight_fmt),
        .vector_done_valid(vector_done_valid),
        .vector_done_ready(vector_done_ready),
        .vector_done_error(vector_done_error),
        .vector_done_status(vector_done_status)
    );

    // Leaf ownership is independent of the controller state encoding.  That
    // keeps response routing fixed while a service drains under backpressure.
    reg gemm_owner_q;
    reg flash_owner_q;
    wire projection_cmd_ready;
    wire projection_done_valid;
    wire projection_done_ready;
    wire projection_done_error;
    wire [7:0] projection_done_status;
    wire attention_cmd_ready;
    wire attention_done_valid;
    wire attention_done_ready;
    wire attention_done_error;
    wire [7:0] attention_done_status;

    assign gemm_req_ready = !gemm_owner_q && projection_cmd_ready;
    assign gemm_done_valid = gemm_owner_q && projection_done_valid;
    assign gemm_done_error = projection_done_error;
    assign gemm_done_status = projection_done_status;
    assign projection_done_ready = gemm_owner_q && gemm_done_ready;

    assign flash_req_ready = !flash_owner_q && attention_cmd_ready;
    assign flash_done_valid = flash_owner_q && attention_done_valid;
    assign flash_done_error = attention_done_error;
    assign flash_done_status = attention_done_status;
    assign attention_done_ready = flash_owner_q && flash_done_ready;

    always @(posedge clk) begin
        if (!rst_n || clear_owner_q) begin
            gemm_owner_q <= 1'b0;
            flash_owner_q <= 1'b0;
        end else begin
            if (gemm_req_valid && gemm_req_ready)
                gemm_owner_q <= 1'b1;
            else if (gemm_done_valid && gemm_done_ready)
                gemm_owner_q <= 1'b0;

            if (flash_req_valid && flash_req_ready)
                flash_owner_q <= 1'b1;
            else if (flash_done_valid && flash_done_ready)
                flash_owner_q <= 1'b0;
        end
    end

    wire embed_cmd_valid;
    wire embed_cmd_ready;
    wire embed_done_valid;
    wire embed_done_ready;
    wire embed_done_error;
    wire [7:0] embed_done_status;
    wire norm_cmd_valid;
    wire norm_cmd_ready;
    wire norm_done_valid;
    wire norm_done_ready;
    wire norm_done_error;
    wire [15:0] norm_done_status;
    wire append_cmd_valid;
    wire append_cmd_ready;
    wire append_done_valid;
    wire append_done_ready;
    wire append_done_error;
    wire [7:0] append_done_status;
    wire [1:0] vector_owner;
    wire vector_dispatch_busy;
    wire vector_dispatch_error;

     vector_dispatch u_vector_dispatch (
        .clk(clk), .rst_n(rst_n), .clear(clear_vector_q),
        .req_valid(vector_req_valid), .req_ready(vector_req_ready),
        .req_op(vector_req_op), .done_valid(vector_done_valid),
        .done_ready(vector_done_ready), .done_error(vector_done_error),
        .done_status(vector_done_status),
        .embed_cmd_valid(embed_cmd_valid),
        .embed_cmd_ready(embed_cmd_ready),
        .embed_done_valid(embed_done_valid),
        .embed_done_ready(embed_done_ready),
        .embed_done_error(embed_done_error),
        .embed_done_status(embed_done_status),
        .norm_cmd_valid(norm_cmd_valid), .norm_cmd_ready(norm_cmd_ready),
        .norm_done_valid(norm_done_valid),
        .norm_done_ready(norm_done_ready),
        .norm_done_error(norm_done_error),
        .norm_done_status(norm_done_status),
        .append_cmd_valid(append_cmd_valid),
        .append_cmd_ready(append_cmd_ready),
        .append_done_valid(append_done_valid),
        .append_done_ready(append_done_ready),
        .append_done_error(append_done_error),
        .append_done_status(append_done_status),
        .owner(vector_owner), .busy(vector_dispatch_busy),
        .protocol_error(vector_dispatch_error)
    );

    // Embedding service and its small-reader client.
    wire embed_read_cmd_valid;
    wire embed_read_cmd_ready;
    wire [63:0] embed_read_cmd_base_addr;
    wire [31:0] embed_read_cmd_port_beats;
    wire [3:0] embed_read_cmd_port_mask;
    wire embed_read_abort;
    wire [511:0] embed_read_data;
    wire embed_read_valid;
    wire embed_read_ready;
    wire embed_read_last;
    wire embed_read_error;
    wire embed_read_busy;
    wire embed_read_done_valid;
    wire embed_read_done_ready;
    wire embed_read_done_error;
    wire [7:0] embed_read_done_status;
    wire embed_r_wr_valid;
    wire embed_r_wr_ready;
    wire embed_r_wr_wave;
    wire [11:0] embed_r_wr_addr;
    wire [3:0] embed_r_wr_lane_mask;
    wire [127:0] embed_r_wr_data;
    wire embed_busy;

     embedding_service u_embedding (
        .clk(clk), .rst_n(rst_n), .clear(clear_embedding_q),
        .abort_run(clear_embedding_q),
        .cmd_valid(embed_cmd_valid), .cmd_ready(embed_cmd_ready),
        .cmd_table_addr(vector_req_addr0),
        .cmd_q1_blocks(vector_req_hidden_blocks[7:2]),
        .cmd_hidden_dim({vector_req_hidden_blocks, 5'b0}),
        .cmd_vocab_rows(vector_req_vocab_rows),
        .cmd_weight_fmt(vector_req_weight_fmt),
        .cmd_token_count(vector_req_token_count),
        .cmd_token_mask(vector_req_lane_mask),
        .cmd_token_ids(vector_req_token_ids),
        .read_cmd_valid(embed_read_cmd_valid),
        .read_cmd_ready(embed_read_cmd_ready),
        .read_cmd_base_addr(embed_read_cmd_base_addr),
        .read_cmd_port_beats(embed_read_cmd_port_beats),
        .read_cmd_port_mask(embed_read_cmd_port_mask),
        .read_abort(embed_read_abort), .read_data(embed_read_data),
        .read_valid(embed_read_valid), .read_ready(embed_read_ready),
        .read_last(embed_read_last), .read_error(embed_read_error),
        .read_busy(embed_read_busy),
        .read_done_valid(embed_read_done_valid),
        .read_done_ready(embed_read_done_ready),
        .read_done_error(embed_read_done_error),
        .read_done_status(embed_read_done_status),
        .r_wr_valid(embed_r_wr_valid), .r_wr_ready(embed_r_wr_ready),
        .r_wr_wave(embed_r_wr_wave), .r_wr_addr(embed_r_wr_addr),
        .r_wr_lane_mask(embed_r_wr_lane_mask), .r_wr_data(embed_r_wr_data),
        .busy(embed_busy), .done_valid(embed_done_valid),
        .done_ready(embed_done_ready), .done_error(embed_done_error),
        .done_status(embed_done_status)
    );

    // Shared vector/sink cluster.  Norm and semantic-projection commands are
    // mutually exclusive by the controller schedule; attention only borrows
    // the cluster's single physical Q8 leaf.
    wire cluster_busy;
    wire cluster_protocol_error;
    wire cluster_abort;
    wire [31:0] norm_done_cycles;
    wire norm_busy;
    wire [3:0] norm_debug_state;
    wire v_gamma_req_valid;
    wire v_gamma_req_ready;
    wire [63:0] v_gamma_req_addr;
    wire [10:0] v_gamma_req_words;
    wire v_gamma_rsp_valid;
    wire v_gamma_rsp_ready;
    wire [127:0] v_gamma_rsp_data;
    wire v_gamma_rsp_last;
    wire v_gamma_rsp_error;
    wire v_r_rd_req_valid;
    wire v_r_rd_req_ready;
    wire v_r_rd_req_wave;
    wire [11:0] v_r_rd_req_addr;
    wire v_r_rd_rsp_valid;
    wire v_r_rd_rsp_ready;
    wire [127:0] v_r_rd_rsp_data;
    wire v_q8_wr_valid;
    wire v_q8_wr_ready;
    wire v_q8_wr_wave;
    wire [8:0] v_q8_wr_addr;
    wire [3:0] v_q8_wr_lane_mask;
    wire [1087:0] v_q8_wr_data;

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
    wire sink_busy;
    wire sink_done_valid;
    wire sink_done_ready;
    wire sink_done_error;
    wire [15:0] sink_done_status;
    wire [31:0] sink_done_cycles;
    wire [4:0] sink_debug_state;
    wire s_gamma_req_valid;
    wire s_gamma_req_ready;
    wire [63:0] s_gamma_req_addr;
    wire [6:0] s_gamma_req_words;
    wire s_gamma_rsp_valid;
    wire s_gamma_rsp_ready;
    wire [127:0] s_gamma_rsp_data;
    wire s_gamma_rsp_last;
    wire s_gamma_rsp_error;
    wire s_rope_req_valid;
    wire s_rope_req_ready;
    wire [63:0] s_rope_req_addr;
    wire [16:0] s_rope_req_position_base;
    wire [7:0] s_rope_req_token_mask;
    wire [7:0] s_rope_req_records;
    wire s_rope_rsp_valid;
    wire s_rope_rsp_ready;
    wire [255:0] s_rope_rsp_data;
    wire s_rope_rsp_last;
    wire s_rope_rsp_error;
    wire sink_proj_valid;
    wire sink_proj_ready;
    wire [2:0] sink_proj_token;
    wire [17:0] sink_proj_row;
    wire [31:0] sink_proj_data_f32;
    wire sink_proj_last;
    wire s_query_wr_valid;
    wire s_query_wr_ready;
    wire s_query_wr_wave;
    wire [11:0] s_query_wr_addr;
    wire [3:0] s_query_wr_lane_mask;
    wire [127:0] s_query_wr_data;
    wire s_newkv_wr_valid;
    wire s_newkv_wr_ready;
    wire s_newkv_wr_wave;
    wire [10:0] s_newkv_wr_addr;
    wire [3:0] s_newkv_wr_lane_mask;
    wire [63:0] s_newkv_wr_data;
    wire s_q8_wr_valid;
    wire s_q8_wr_ready;
    wire s_q8_wr_wave;
    wire [8:0] s_q8_wr_addr;
    wire [3:0] s_q8_wr_lane_mask;
    wire [1087:0] s_q8_wr_data;
    wire s_r_rd_req_valid;
    wire s_r_rd_req_ready;
    wire s_r_rd_req_wave;
    wire [11:0] s_r_rd_req_addr;
    wire s_r_rd_rsp_valid;
    wire s_r_rd_rsp_ready;
    wire [127:0] s_r_rd_rsp_data;
    wire s_r_wr_valid;
    wire s_r_wr_ready;
    wire s_r_wr_wave;
    wire [11:0] s_r_wr_addr;
    wire [3:0] s_r_wr_lane_mask;
    wire [127:0] s_r_wr_data;

    wire a_leaf_q8_cfg_valid;
    wire a_leaf_q8_cfg_ready;
    wire [14:0] a_leaf_q8_cfg_rows;
    wire [3:0] a_leaf_q8_cfg_lane_mask;
    wire a_leaf_q8_abort;
    wire a_leaf_q8_busy;
    wire a_leaf_q8_in_valid;
    wire a_leaf_q8_in_ready;
    wire [127:0] a_leaf_q8_in_data;
    wire a_leaf_q8_out_valid;
    wire a_leaf_q8_out_ready;
    wire [8:0] a_leaf_q8_out_block;
    wire [1087:0] a_leaf_q8_out_data;
    wire [7:0] a_leaf_q8_out_status;
    wire a_leaf_q8_out_last;

    // Global projection clear has a dedicated copy in every dependent island.
    // Only a projection-local failure is allowed to become a cluster abort.
    wire sink_local_abort_run = sink_abort_run && !clear_projection_q;
    assign cluster_abort = sink_local_abort_run;
    vector_cluster u_vector_sink_cluster (
        .clk(clk), .rst_n(rst_n), .clear(clear_cluster_q),
        .abort_run(cluster_abort),
        .cluster_busy(cluster_busy),
        .protocol_error(cluster_protocol_error),
        .v_cmd_valid(norm_cmd_valid), .v_cmd_ready(norm_cmd_ready),
        .v_cmd_op(vector_req_op),
        .v_cmd_token_mask(vector_req_lane_mask),
        .v_cmd_hidden_blocks(vector_req_hidden_blocks),
        .v_cmd_gamma_addr(vector_req_addr0),
        .v_cmd_epsilon(32'h3586_37bd),
        .v_done_valid(norm_done_valid), .v_done_ready(norm_done_ready),
        .v_done_error(norm_done_error), .v_done_status(norm_done_status),
        .v_done_cycles(norm_done_cycles), .v_busy(norm_busy),
        .v_debug_state(norm_debug_state),
        .v_gamma_req_valid(v_gamma_req_valid),
        .v_gamma_req_ready(v_gamma_req_ready),
        .v_gamma_req_addr(v_gamma_req_addr),
        .v_gamma_req_words(v_gamma_req_words),
        .v_gamma_rsp_valid(v_gamma_rsp_valid),
        .v_gamma_rsp_ready(v_gamma_rsp_ready),
        .v_gamma_rsp_data(v_gamma_rsp_data),
        .v_gamma_rsp_last(v_gamma_rsp_last),
        .v_gamma_rsp_error(v_gamma_rsp_error),
        .v_r_rd_req_valid(v_r_rd_req_valid),
        .v_r_rd_req_ready(v_r_rd_req_ready),
        .v_r_rd_req_wave(v_r_rd_req_wave),
        .v_r_rd_req_addr(v_r_rd_req_addr),
        .v_r_rd_rsp_valid(v_r_rd_rsp_valid),
        .v_r_rd_rsp_ready(v_r_rd_rsp_ready),
        .v_r_rd_rsp_data(v_r_rd_rsp_data), .v_r_rd_rsp_error(1'b0),
        .v_q8_wr_valid(v_q8_wr_valid), .v_q8_wr_ready(v_q8_wr_ready),
        .v_q8_wr_wave(v_q8_wr_wave), .v_q8_wr_addr(v_q8_wr_addr),
        .v_q8_wr_lane_mask(v_q8_wr_lane_mask),
        .v_q8_wr_data(v_q8_wr_data),
        .s_cfg_valid(sink_cfg_valid), .s_cfg_ready(sink_cfg_ready),
        .s_cfg_mode(sink_cfg_mode),
        .s_cfg_token_mask(sink_cfg_token_mask),
        .s_cfg_hidden_dim(sink_cfg_hidden_dim),
        .s_cfg_ffn_dim(sink_cfg_ffn_dim),
        .s_cfg_q_heads(sink_cfg_q_heads),
        .s_cfg_kv_heads(sink_cfg_kv_heads),
        .s_cfg_position_base(sink_cfg_position_base),
        .s_cfg_epsilon(sink_cfg_epsilon),
        .s_cfg_q_gamma_addr(sink_cfg_q_gamma_addr),
        .s_cfg_k_gamma_addr(sink_cfg_k_gamma_addr),
        .s_cfg_rope_addr(sink_cfg_rope_addr),
        .s_projection_armed(sink_projection_armed), .s_busy(sink_busy),
        .s_done_valid(sink_done_valid), .s_done_ready(sink_done_ready),
        .s_done_error(sink_done_error), .s_done_status(sink_done_status),
        .s_done_cycles(sink_done_cycles), .s_debug_state(sink_debug_state),
        .s_gamma_req_valid(s_gamma_req_valid),
        .s_gamma_req_ready(s_gamma_req_ready),
        .s_gamma_req_addr(s_gamma_req_addr),
        .s_gamma_req_words(s_gamma_req_words),
        .s_gamma_rsp_valid(s_gamma_rsp_valid),
        .s_gamma_rsp_ready(s_gamma_rsp_ready),
        .s_gamma_rsp_data(s_gamma_rsp_data),
        .s_gamma_rsp_last(s_gamma_rsp_last),
        .s_gamma_rsp_error(s_gamma_rsp_error),
        .s_rope_req_valid(s_rope_req_valid),
        .s_rope_req_ready(s_rope_req_ready),
        .s_rope_req_addr(s_rope_req_addr),
        .s_rope_req_position_base(s_rope_req_position_base),
        .s_rope_req_token_mask(s_rope_req_token_mask),
        .s_rope_req_records(s_rope_req_records),
        .s_rope_rsp_valid(s_rope_rsp_valid),
        .s_rope_rsp_ready(s_rope_rsp_ready),
        .s_rope_rsp_data(s_rope_rsp_data),
        .s_rope_rsp_last(s_rope_rsp_last),
        .s_rope_rsp_error(s_rope_rsp_error),
        .s_proj_valid(sink_proj_valid), .s_proj_ready(sink_proj_ready),
        .s_proj_token(sink_proj_token), .s_proj_row(sink_proj_row),
        .s_proj_data_f32(sink_proj_data_f32),
        .s_proj_last(sink_proj_last),
        .s_query_wr_valid(s_query_wr_valid),
        .s_query_wr_ready(s_query_wr_ready),
        .s_query_wr_wave(s_query_wr_wave),
        .s_query_wr_addr(s_query_wr_addr),
        .s_query_wr_lane_mask(s_query_wr_lane_mask),
        .s_query_wr_data(s_query_wr_data),
        .s_newkv_wr_valid(s_newkv_wr_valid),
        .s_newkv_wr_ready(s_newkv_wr_ready),
        .s_newkv_wr_wave(s_newkv_wr_wave),
        .s_newkv_wr_addr(s_newkv_wr_addr),
        .s_newkv_wr_lane_mask(s_newkv_wr_lane_mask),
        .s_newkv_wr_data(s_newkv_wr_data),
        .s_q8_wr_valid(s_q8_wr_valid), .s_q8_wr_ready(s_q8_wr_ready),
        .s_q8_wr_wave(s_q8_wr_wave), .s_q8_wr_addr(s_q8_wr_addr),
        .s_q8_wr_lane_mask(s_q8_wr_lane_mask), .s_q8_wr_data(s_q8_wr_data),
        .s_r_rd_req_valid(s_r_rd_req_valid),
        .s_r_rd_req_ready(s_r_rd_req_ready),
        .s_r_rd_req_wave(s_r_rd_req_wave),
        .s_r_rd_req_addr(s_r_rd_req_addr),
        .s_r_rd_rsp_valid(s_r_rd_rsp_valid),
        .s_r_rd_rsp_ready(s_r_rd_rsp_ready),
        .s_r_rd_rsp_data(s_r_rd_rsp_data), .s_r_rd_rsp_error(1'b0),
        .s_r_wr_valid(s_r_wr_valid), .s_r_wr_ready(s_r_wr_ready),
        .s_r_wr_wave(s_r_wr_wave), .s_r_wr_addr(s_r_wr_addr),
        .s_r_wr_lane_mask(s_r_wr_lane_mask), .s_r_wr_data(s_r_wr_data),
        .a_leaf_q8_cfg_valid(a_leaf_q8_cfg_valid),
        .a_leaf_q8_cfg_ready(a_leaf_q8_cfg_ready),
        .a_leaf_q8_cfg_rows(a_leaf_q8_cfg_rows),
        .a_leaf_q8_cfg_lane_mask(a_leaf_q8_cfg_lane_mask),
        .a_leaf_q8_abort(a_leaf_q8_abort),
        .a_leaf_q8_busy(a_leaf_q8_busy),
        .a_leaf_q8_in_valid(a_leaf_q8_in_valid),
        .a_leaf_q8_in_ready(a_leaf_q8_in_ready),
        .a_leaf_q8_in_data(a_leaf_q8_in_data),
        .a_leaf_q8_out_valid(a_leaf_q8_out_valid),
        .a_leaf_q8_out_ready(a_leaf_q8_out_ready),
        .a_leaf_q8_out_block(a_leaf_q8_out_block),
        .a_leaf_q8_out_data(a_leaf_q8_out_data),
        .a_leaf_q8_out_status(a_leaf_q8_out_status),
        .a_leaf_q8_out_last(a_leaf_q8_out_last)
    );

    // Low-bit projection service and its outer-reader client.
    wire proj_read_cmd_valid;
    wire proj_read_cmd_ready;
    wire [63:0] proj_read_cmd_base_addr;
    wire [31:0] proj_read_cmd_port_beats;
    wire [3:0] proj_read_cmd_port_mask;
    wire proj_read_abort;
    wire [511:0] proj_read_data;
    wire proj_read_valid;
    wire proj_read_ready;
    wire proj_read_last;
    wire proj_read_error;
    wire proj_read_busy;
    wire proj_read_done_valid;
    wire proj_read_done_ready;
    wire proj_read_done_error;
    wire [7:0] proj_read_done_status;
    wire proj_q8_rd_req_valid;
    wire proj_q8_rd_req_ready;
    wire proj_q8_rd_req_wave;
    wire [8:0] proj_q8_rd_req_addr;
    wire proj_q8_rd_rsp_valid;
    wire proj_q8_rd_rsp_ready;
    wire [1087:0] proj_q8_rd_rsp_data;
    wire projection_busy;
    wire [15:0] projection_derived_k;
    wire [17:0] projection_derived_m;
    wire [15:0] projection_derived_rowblocks;
    wire [31:0] projection_weight_beats;
    wire [31:0] projection_wave_issues;

     projection_service u_projection (
        .clk(clk), .rst_n(rst_n), .clear(clear_projection_q),
        .abort_run(1'b0),
        .cmd_valid(gemm_req_valid && !gemm_owner_q),
        .cmd_ready(projection_cmd_ready), .cmd_op(gemm_req_op),
        .cmd_token_count(gemm_req_token_count),
        .cmd_token_mask(gemm_req_lane_mask), .cmd_addr0(gemm_req_addr0),
        .cmd_addr1(gemm_req_addr1), .cmd_addr2(gemm_req_addr2),
        .cmd_addr3(gemm_req_addr3),
        .cmd_hidden_blocks(gemm_req_hidden_blocks),
        .cmd_ffn_blocks(gemm_req_ffn_blocks),
        .cmd_q_heads(gemm_req_q_heads),
        .cmd_kv_heads(gemm_req_kv_heads),
        .cmd_head_dim(gemm_req_head_dim),
        .cmd_position_base(gemm_req_position_base),
        .cmd_epsilon(gemm_req_epsilon),
        .cmd_vocab_rows(gemm_req_vocab_rows),
        .cmd_weight_fmt(gemm_req_weight_fmt),
        .cmd_emit_full_logits(EMIT_FULL_LOGITS != 0),
        .read_cmd_valid(proj_read_cmd_valid),
        .read_cmd_ready(proj_read_cmd_ready),
        .read_cmd_base_addr(proj_read_cmd_base_addr),
        .read_cmd_port_beats(proj_read_cmd_port_beats),
        .read_cmd_port_mask(proj_read_cmd_port_mask),
        .read_abort(proj_read_abort), .read_data(proj_read_data),
        .read_valid(proj_read_valid), .read_ready(proj_read_ready),
        .read_last(proj_read_last), .read_error(proj_read_error),
        .read_busy(proj_read_busy),
        .read_done_valid(proj_read_done_valid),
        .read_done_ready(proj_read_done_ready),
        .read_done_error(proj_read_done_error),
        .read_done_status(proj_read_done_status),
        .q8_rd_req_valid(proj_q8_rd_req_valid),
        .q8_rd_req_ready(proj_q8_rd_req_ready),
        .q8_rd_req_wave(proj_q8_rd_req_wave),
        .q8_rd_req_addr(proj_q8_rd_req_addr),
        .q8_rd_rsp_valid(proj_q8_rd_rsp_valid),
        .q8_rd_rsp_ready(proj_q8_rd_rsp_ready),
        .q8_rd_rsp_data(proj_q8_rd_rsp_data),
        .sink_cfg_valid(sink_cfg_valid), .sink_cfg_ready(sink_cfg_ready),
        .sink_cfg_mode(sink_cfg_mode),
        .sink_cfg_token_mask(sink_cfg_token_mask),
        .sink_cfg_hidden_dim(sink_cfg_hidden_dim),
        .sink_cfg_ffn_dim(sink_cfg_ffn_dim),
        .sink_cfg_q_heads(sink_cfg_q_heads),
        .sink_cfg_kv_heads(sink_cfg_kv_heads),
        .sink_cfg_position_base(sink_cfg_position_base),
        .sink_cfg_epsilon(sink_cfg_epsilon),
        .sink_cfg_q_gamma_addr(sink_cfg_q_gamma_addr),
        .sink_cfg_k_gamma_addr(sink_cfg_k_gamma_addr),
        .sink_cfg_rope_addr(sink_cfg_rope_addr),
        .sink_abort_run(sink_abort_run),
        .sink_projection_armed(sink_projection_armed),
        .sink_proj_valid(sink_proj_valid),
        .sink_proj_ready(sink_proj_ready),
        .sink_proj_token(sink_proj_token), .sink_proj_row(sink_proj_row),
        .sink_proj_data_f32(sink_proj_data_f32),
        .sink_proj_last(sink_proj_last),
        .sink_done_valid(sink_done_valid),
        .sink_done_ready(sink_done_ready),
        .sink_done_error(sink_done_error),
        .sink_done_status(sink_done_status),
        .logits_valid(logits_valid), .logits_ready(logits_ready),
        .logits_row(logits_row), .logits_data(logits_data),
        .logits_last(logits_last), .result_valid(result_valid),
        .result_ready(result_ready), .result_token(result_token),
        .result_logit(result_logit), .result_error(result_error),
        .result_status(result_status), .busy(projection_busy),
        .done_valid(projection_done_valid),
        .done_ready(projection_done_ready),
        .done_error(projection_done_error),
        .done_status(projection_done_status),
        .derived_k(projection_derived_k),
        .derived_m(projection_derived_m),
        .derived_rowblocks(projection_derived_rowblocks),
        .accepted_weight_beats(projection_weight_beats),
        .activation_wave_issues(projection_wave_issues),
        .metrics_probe(metrics_projection_probe)
    );

    // Layer-local NewKV publication and the sole committed-KV writer.
    wire append_newkv_rd_req_valid;
    wire append_newkv_rd_req_ready;
    wire append_newkv_rd_req_wave;
    wire [10:0] append_newkv_rd_req_addr;
    wire append_newkv_rd_rsp_valid;
    wire append_newkv_rd_rsp_ready;
    wire [63:0] append_newkv_rd_rsp_data;
    wire append_wr_cmd_valid;
    wire append_wr_cmd_ready;
    wire [63:0] append_wr_cmd_addr;
    wire [31:0] append_wr_cmd_segment_beats;
    wire [31:0] append_wr_cmd_stride_bytes;
    wire [16:0] append_wr_cmd_repeats;
    wire [127:0] append_wr_data;
    wire append_wr_valid;
    wire append_wr_ready;
    wire append_wr_last;
    wire append_wr_error;
    wire kv_writer_busy;
    wire kv_writer_done_valid;
    wire kv_writer_done_ready;
    wire kv_writer_done_error;
    wire [7:0] kv_writer_done_status;
    wire append_busy;

     kv_append8 u_append (
        .clk(clk), .rst_n(rst_n), .clear(clear_append_q),
        .abort_run(clear_append_q), .cmd_valid(append_cmd_valid),
        .cmd_ready(append_cmd_ready),
        .cmd_layer_kv_base(vector_req_addr0),
        .cmd_position_base(vector_req_position_base),
        .cmd_token_count(vector_req_token_count),
        .cmd_token_mask(vector_req_lane_mask),
        .newkv_rd_req_valid(append_newkv_rd_req_valid),
        .newkv_rd_req_ready(append_newkv_rd_req_ready),
        .newkv_rd_req_wave(append_newkv_rd_req_wave),
        .newkv_rd_req_addr(append_newkv_rd_req_addr),
        .newkv_rd_rsp_valid(append_newkv_rd_rsp_valid),
        .newkv_rd_rsp_ready(append_newkv_rd_rsp_ready),
        .newkv_rd_rsp_data(append_newkv_rd_rsp_data),
        .newkv_rd_rsp_error(1'b0),
        .wr_cmd_valid(append_wr_cmd_valid),
        .wr_cmd_ready(append_wr_cmd_ready),
        .wr_cmd_addr(append_wr_cmd_addr),
        .wr_cmd_segment_beats(append_wr_cmd_segment_beats),
        .wr_cmd_stride_bytes(append_wr_cmd_stride_bytes),
        .wr_cmd_repeats(append_wr_cmd_repeats),
        .wr_data(append_wr_data), .wr_valid(append_wr_valid),
        .wr_ready(append_wr_ready), .wr_last(append_wr_last),
        .wr_error(append_wr_error), .wr_busy(kv_writer_busy),
        .wr_done_valid(kv_writer_done_valid),
        .wr_done_ready(kv_writer_done_ready),
        .wr_done_error(kv_writer_done_error),
        .wr_done_status(kv_writer_done_status), .busy(append_busy),
        .done_valid(append_done_valid), .done_ready(append_done_ready),
        .done_error(append_done_error), .done_status(append_done_status)
    );

     axi_write128 #(.ADDR_W(ADDR_W)) u_kv_writer (
        .clk(clk), .rst_n(rst_n), .clear(clear_kv_writer_q),
        .abort_run(clear_kv_writer_q), .cmd_valid(append_wr_cmd_valid),
        .cmd_ready(append_wr_cmd_ready), .cmd_addr(append_wr_cmd_addr),
        .cmd_segment_beats(append_wr_cmd_segment_beats),
        .cmd_stride_bytes(append_wr_cmd_stride_bytes),
        .cmd_repeats(append_wr_cmd_repeats), .in_data(append_wr_data),
        .in_valid(append_wr_valid), .in_ready(append_wr_ready),
        .in_last(append_wr_last), .in_error(append_wr_error),
        .busy(kv_writer_busy), .done_valid(kv_writer_done_valid),
        .done_ready(kv_writer_done_ready),
        .done_error(kv_writer_done_error),
        .done_status(kv_writer_done_status),
        .m_axi_awaddr(kv_axi_awaddr), .m_axi_awlen(kv_axi_awlen),
        .m_axi_awsize(kv_axi_awsize), .m_axi_awburst(kv_axi_awburst),
        .m_axi_awvalid(kv_axi_awvalid), .m_axi_awready(kv_axi_awready),
        .m_axi_wdata(kv_axi_wdata), .m_axi_wstrb(kv_axi_wstrb),
        .m_axi_wlast(kv_axi_wlast), .m_axi_wvalid(kv_axi_wvalid),
        .m_axi_wready(kv_axi_wready), .m_axi_bresp(kv_axi_bresp),
        .m_axi_bvalid(kv_axi_bvalid), .m_axi_bready(kv_axi_bready)
    );

    // FlashAttention service and independent K/V history movers.
    wire attention_query_rd_req_valid;
    wire attention_query_rd_req_ready;
    wire attention_query_rd_req_wave;
    wire [11:0] attention_query_rd_req_addr;
    wire attention_query_rd_rsp_valid;
    wire attention_query_rd_rsp_ready;
    wire [127:0] attention_query_rd_rsp_data;
    wire attention_newkv_rd_req_valid;
    wire attention_newkv_rd_req_ready;
    wire attention_newkv_rd_req_wave;
    wire [10:0] attention_newkv_rd_req_addr;
    wire attention_newkv_rd_rsp_valid;
    wire attention_newkv_rd_rsp_ready;
    wire [63:0] attention_newkv_rd_rsp_data;
    wire attention_q8_wr_valid;
    wire attention_q8_wr_ready;
    wire attention_q8_wr_wave;
    wire [8:0] attention_q8_wr_addr;
    wire [3:0] attention_q8_wr_lane_mask;
    wire [1087:0] attention_q8_wr_data;
    wire hist_k_cmd_valid;
    wire hist_k_cmd_ready;
    wire [63:0] hist_k_cmd_addr;
    wire [31:0] hist_k_cmd_segment_beats;
    wire [31:0] hist_k_cmd_stride_bytes;
    wire [16:0] hist_k_cmd_repeats;
    wire hist_k_abort;
    wire [127:0] hist_k_data;
    wire hist_k_valid;
    wire hist_k_ready;
    wire hist_k_last;
    wire hist_k_error;
    wire hist_k_busy;
    wire hist_k_done_valid;
    wire hist_k_done_ready;
    wire hist_k_done_error;
    wire [7:0] hist_k_done_status;
    wire hist_v_cmd_valid;
    wire hist_v_cmd_ready;
    wire [63:0] hist_v_cmd_addr;
    wire [31:0] hist_v_cmd_segment_beats;
    wire [31:0] hist_v_cmd_stride_bytes;
    wire [16:0] hist_v_cmd_repeats;
    wire hist_v_abort;
    wire [127:0] hist_v_data;
    wire hist_v_valid;
    wire hist_v_ready;
    wire hist_v_last;
    wire hist_v_error;
    wire hist_v_busy;
    wire hist_v_done_valid;
    wire hist_v_done_ready;
    wire hist_v_done_error;
    wire [7:0] hist_v_done_status;
    wire attention_busy;

     attention_service u_attention (
        .clk(clk), .rst_n(rst_n), .clear(clear_attention_q),
        .abort_run(clear_attention_q),
        .cmd_valid(flash_req_valid && !flash_owner_q),
        .cmd_ready(attention_cmd_ready),
        .cmd_token_count(flash_req_token_count),
        .cmd_token_mask(flash_req_lane_mask),
        .cmd_q_heads(flash_req_q_heads),
        .cmd_history_len(flash_req_position_base),
        .cmd_scale(32'h3db5_04f3),
        .cmd_layer_kv_base(flash_req_kv_base),
        .query_rd_req_valid(attention_query_rd_req_valid),
        .query_rd_req_ready(attention_query_rd_req_ready),
        .query_rd_req_wave(attention_query_rd_req_wave),
        .query_rd_req_addr(attention_query_rd_req_addr),
        .query_rd_rsp_valid(attention_query_rd_rsp_valid),
        .query_rd_rsp_ready(attention_query_rd_rsp_ready),
        .query_rd_rsp_data(attention_query_rd_rsp_data),
        .newkv_rd_req_valid(attention_newkv_rd_req_valid),
        .newkv_rd_req_ready(attention_newkv_rd_req_ready),
        .newkv_rd_req_wave(attention_newkv_rd_req_wave),
        .newkv_rd_req_addr(attention_newkv_rd_req_addr),
        .newkv_rd_rsp_valid(attention_newkv_rd_rsp_valid),
        .newkv_rd_rsp_ready(attention_newkv_rd_rsp_ready),
        .newkv_rd_rsp_data(attention_newkv_rd_rsp_data),
        .q8_wr_valid(attention_q8_wr_valid),
        .q8_wr_ready(attention_q8_wr_ready),
        .q8_wr_wave(attention_q8_wr_wave),
        .q8_wr_addr(attention_q8_wr_addr),
        .q8_wr_lane_mask(attention_q8_wr_lane_mask),
        .q8_wr_data(attention_q8_wr_data),
        .leaf_q8_cfg_valid(a_leaf_q8_cfg_valid),
        .leaf_q8_cfg_ready(a_leaf_q8_cfg_ready),
        .leaf_q8_cfg_rows(a_leaf_q8_cfg_rows),
        .leaf_q8_cfg_lane_mask(a_leaf_q8_cfg_lane_mask),
        .leaf_q8_abort(a_leaf_q8_abort),
        .leaf_q8_busy(a_leaf_q8_busy),
        .leaf_q8_in_valid(a_leaf_q8_in_valid),
        .leaf_q8_in_ready(a_leaf_q8_in_ready),
        .leaf_q8_in_data(a_leaf_q8_in_data),
        .leaf_q8_out_valid(a_leaf_q8_out_valid),
        .leaf_q8_out_ready(a_leaf_q8_out_ready),
        .leaf_q8_out_block(a_leaf_q8_out_block),
        .leaf_q8_out_data(a_leaf_q8_out_data),
        .leaf_q8_out_status(a_leaf_q8_out_status),
        .leaf_q8_out_last(a_leaf_q8_out_last),
        .hist_k_cmd_valid(hist_k_cmd_valid),
        .hist_k_cmd_ready(hist_k_cmd_ready),
        .hist_k_cmd_addr(hist_k_cmd_addr),
        .hist_k_cmd_segment_beats(hist_k_cmd_segment_beats),
        .hist_k_cmd_stride_bytes(hist_k_cmd_stride_bytes),
        .hist_k_cmd_repeats(hist_k_cmd_repeats),
        .hist_k_abort(hist_k_abort), .hist_k_data(hist_k_data),
        .hist_k_valid(hist_k_valid), .hist_k_ready(hist_k_ready),
        .hist_k_last(hist_k_last), .hist_k_error(hist_k_error),
        .hist_k_busy(hist_k_busy),
        .hist_k_done_valid(hist_k_done_valid),
        .hist_k_done_ready(hist_k_done_ready),
        .hist_k_done_error(hist_k_done_error),
        .hist_k_done_status(hist_k_done_status),
        .hist_v_cmd_valid(hist_v_cmd_valid),
        .hist_v_cmd_ready(hist_v_cmd_ready),
        .hist_v_cmd_addr(hist_v_cmd_addr),
        .hist_v_cmd_segment_beats(hist_v_cmd_segment_beats),
        .hist_v_cmd_stride_bytes(hist_v_cmd_stride_bytes),
        .hist_v_cmd_repeats(hist_v_cmd_repeats),
        .hist_v_abort(hist_v_abort), .hist_v_data(hist_v_data),
        .hist_v_valid(hist_v_valid), .hist_v_ready(hist_v_ready),
        .hist_v_last(hist_v_last), .hist_v_error(hist_v_error),
        .hist_v_busy(hist_v_busy),
        .hist_v_done_valid(hist_v_done_valid),
        .hist_v_done_ready(hist_v_done_ready),
        .hist_v_done_error(hist_v_done_error),
        .hist_v_done_status(hist_v_done_status),
        .busy(attention_busy), .done_valid(attention_done_valid),
        .done_ready(attention_done_ready),
        .done_error(attention_done_error),
        .done_status(attention_done_status)
    );

     axi_read128 #(.ADDR_W(ADDR_W)) u_hist_k_reader (
        .clk(clk), .rst_n(rst_n), .clear(clear_history_q),
        .abort_run(clear_history_q || hist_k_abort),
        .cmd_valid(hist_k_cmd_valid), .cmd_ready(hist_k_cmd_ready),
        .cmd_addr(hist_k_cmd_addr),
        .cmd_segment_beats(hist_k_cmd_segment_beats),
        .cmd_stride_bytes(hist_k_cmd_stride_bytes),
        .cmd_repeats(hist_k_cmd_repeats), .out_data(hist_k_data),
        .out_valid(hist_k_valid), .out_ready(hist_k_ready),
        .out_last(hist_k_last), .out_error(hist_k_error),
        .busy(hist_k_busy), .done_valid(hist_k_done_valid),
        .done_ready(hist_k_done_ready), .done_error(hist_k_done_error),
        .done_status(hist_k_done_status),
        .m_axi_araddr(hist_k_axi_araddr),
        .m_axi_arlen(hist_k_axi_arlen),
        .m_axi_arsize(hist_k_axi_arsize),
        .m_axi_arburst(hist_k_axi_arburst),
        .m_axi_arvalid(hist_k_axi_arvalid),
        .m_axi_arready(hist_k_axi_arready),
        .m_axi_rdata(hist_k_axi_rdata), .m_axi_rresp(hist_k_axi_rresp),
        .m_axi_rlast(hist_k_axi_rlast), .m_axi_rvalid(hist_k_axi_rvalid),
        .m_axi_rready(hist_k_axi_rready)
    );

     axi_read128 #(.ADDR_W(ADDR_W)) u_hist_v_reader (
        .clk(clk), .rst_n(rst_n), .clear(clear_history_q),
        .abort_run(clear_history_q || hist_v_abort),
        .cmd_valid(hist_v_cmd_valid), .cmd_ready(hist_v_cmd_ready),
        .cmd_addr(hist_v_cmd_addr),
        .cmd_segment_beats(hist_v_cmd_segment_beats),
        .cmd_stride_bytes(hist_v_cmd_stride_bytes),
        .cmd_repeats(hist_v_cmd_repeats), .out_data(hist_v_data),
        .out_valid(hist_v_valid), .out_ready(hist_v_ready),
        .out_last(hist_v_last), .out_error(hist_v_error),
        .busy(hist_v_busy), .done_valid(hist_v_done_valid),
        .done_ready(hist_v_done_ready), .done_error(hist_v_done_error),
        .done_status(hist_v_done_status),
        .m_axi_araddr(hist_v_axi_araddr),
        .m_axi_arlen(hist_v_axi_arlen),
        .m_axi_arsize(hist_v_axi_arsize),
        .m_axi_arburst(hist_v_axi_arburst),
        .m_axi_arvalid(hist_v_axi_arvalid),
        .m_axi_arready(hist_v_axi_arready),
        .m_axi_rdata(hist_v_axi_rdata), .m_axi_rresp(hist_v_axi_rresp),
        .m_axi_rlast(hist_v_axi_rlast), .m_axi_rvalid(hist_v_axi_rvalid),
        .m_axi_rready(hist_v_axi_rready)
    );

    // Coefficient and embedding traffic is the low-bandwidth client of the
    // same physical four-port reader used for packed projection weights.
    wire small_cmd_valid;
    wire small_cmd_ready;
    wire [63:0] small_cmd_base_addr;
    wire [31:0] small_cmd_port_beats;
    wire [3:0] small_cmd_port_mask;
    wire small_abort;
    wire [511:0] small_data;
    wire small_valid;
    wire small_ready;
    wire small_last;
    wire small_error;
    wire small_busy;
    wire small_done_valid;
    wire small_done_ready;
    wire small_done_error;
    wire [7:0] small_done_status;

     small_read_mux u_small_read_mux (
        .clk(clk), .rst_n(rst_n), .clear(clear_small_q),
        .abort_run(clear_small_q || sink_local_abort_run),
        .embed_cmd_valid(embed_read_cmd_valid),
        .embed_cmd_ready(embed_read_cmd_ready),
        .embed_cmd_base_addr(embed_read_cmd_base_addr),
        .embed_cmd_port_beats(embed_read_cmd_port_beats),
        .embed_cmd_port_mask(embed_read_cmd_port_mask),
        .embed_abort(embed_read_abort), .embed_data(embed_read_data),
        .embed_valid(embed_read_valid), .embed_ready(embed_read_ready),
        .embed_last(embed_read_last), .embed_error(embed_read_error),
        .embed_busy(embed_read_busy),
        .embed_done_valid(embed_read_done_valid),
        .embed_done_ready(embed_read_done_ready),
        .embed_done_error(embed_read_done_error),
        .embed_done_status(embed_read_done_status),
        .vector_req_valid(v_gamma_req_valid),
        .vector_req_ready(v_gamma_req_ready),
        .vector_req_addr(v_gamma_req_addr),
        .vector_req_words(v_gamma_req_words),
        .vector_rsp_valid(v_gamma_rsp_valid),
        .vector_rsp_ready(v_gamma_rsp_ready),
        .vector_rsp_data(v_gamma_rsp_data),
        .vector_rsp_last(v_gamma_rsp_last),
        .vector_rsp_error(v_gamma_rsp_error),
        .sink_req_valid(s_gamma_req_valid),
        .sink_req_ready(s_gamma_req_ready),
        .sink_req_addr(s_gamma_req_addr),
        .sink_req_words(s_gamma_req_words),
        .sink_rsp_valid(s_gamma_rsp_valid),
        .sink_rsp_ready(s_gamma_rsp_ready),
        .sink_rsp_data(s_gamma_rsp_data),
        .sink_rsp_last(s_gamma_rsp_last),
        .sink_rsp_error(s_gamma_rsp_error),
        .svc_cmd_valid(small_cmd_valid), .svc_cmd_ready(small_cmd_ready),
        .svc_cmd_base_addr(small_cmd_base_addr),
        .svc_cmd_port_beats(small_cmd_port_beats),
        .svc_cmd_port_mask(small_cmd_port_mask),
        .svc_abort_run(small_abort), .svc_data(small_data),
        .svc_valid(small_valid), .svc_ready(small_ready),
        .svc_last(small_last), .svc_error(small_error),
        .svc_busy(small_busy), .svc_done_valid(small_done_valid),
        .svc_done_ready(small_done_ready),
        .svc_done_error(small_done_error),
        .svc_done_status(small_done_status)
    );

    function automatic [3:0] popcount8(input [7:0] mask);
        integer bit_index;
        begin
            popcount8 = 4'd0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                popcount8 = popcount8 + mask[bit_index];
        end
    endfunction

    wire rope_cmd_ready;
    wire [255:0] rope_coeff_data;
    wire rope_coeff_valid;
    wire rope_coeff_ready;
    wire rope_coeff_last;
    wire rope_coeff_error;
    wire rope_busy;
    wire rope_done_valid;
    wire rope_done_error;
    wire [7:0] rope_done_status;
    wire rope_read_cmd_valid;
    wire rope_read_cmd_ready;
    wire [63:0] rope_read_cmd_base_addr;
    wire [31:0] rope_read_cmd_port_beats;
    wire [3:0] rope_read_cmd_port_mask;
    wire rope_read_abort;
    wire [511:0] rope_read_data;
    wire rope_read_valid;
    wire rope_read_ready;
    wire rope_read_last;
    wire rope_read_error;
    wire rope_read_busy;
    wire rope_read_done_valid;
    wire rope_read_done_ready;
    wire rope_read_done_error;
    wire [7:0] rope_read_done_status;

    assign s_rope_req_ready = rope_cmd_ready;
    assign s_rope_rsp_valid = rope_coeff_valid;
    assign rope_coeff_ready = s_rope_rsp_ready;
    assign s_rope_rsp_data = rope_coeff_data;
    assign s_rope_rsp_last = rope_coeff_last;
    assign s_rope_rsp_error = rope_coeff_error;

     rope_fetch4 u_rope_fetch (
        .clk(clk), .rst_n(rst_n), .clear(clear_rope_q),
        .abort_run(clear_rope_q || sink_local_abort_run),
        .cmd_valid(s_rope_req_valid), .cmd_ready(rope_cmd_ready),
        .cmd_table_addr(s_rope_req_addr),
        .cmd_position_base(s_rope_req_position_base),
        .cmd_token_count(popcount8(s_rope_req_token_mask)),
        .coeff_data(rope_coeff_data), .coeff_valid(rope_coeff_valid),
        .coeff_ready(rope_coeff_ready), .coeff_last(rope_coeff_last),
        .coeff_error(rope_coeff_error), .busy(rope_busy),
        .done_valid(rope_done_valid), .done_ready(1'b1),
        .done_error(rope_done_error), .done_status(rope_done_status),
        .read_cmd_valid(rope_read_cmd_valid),
        .read_cmd_ready(rope_read_cmd_ready),
        .read_cmd_base_addr(rope_read_cmd_base_addr),
        .read_cmd_port_beats(rope_read_cmd_port_beats),
        .read_cmd_port_mask(rope_read_cmd_port_mask),
        .read_abort(rope_read_abort), .read_data(rope_read_data),
        .read_valid(rope_read_valid), .read_ready(rope_read_ready),
        .read_last(rope_read_last), .read_error(rope_read_error),
        .read_busy(rope_read_busy),
        .read_done_valid(rope_read_done_valid),
        .read_done_ready(rope_read_done_ready),
        .read_done_error(rope_read_done_error),
        .read_done_status(rope_read_done_status)
    );

    wire quad_cmd_valid;
    wire quad_cmd_ready;
    wire [63:0] quad_cmd_base_addr;
    wire [31:0] quad_cmd_port_beats;
    wire [3:0] quad_cmd_port_mask;
    wire quad_abort;
    wire [511:0] quad_data;
    wire quad_valid;
    wire quad_ready;
    wire quad_last;
    wire quad_error;
    wire quad_busy;
    wire quad_done_valid;
    wire quad_done_ready;
    wire quad_done_error;
    wire [7:0] quad_done_status;

     quad_read_arbiter u_quad_read_arbiter (
        .clk(clk), .rst_n(rst_n), .clear(clear_arbiter_q),
        .proj_cmd_valid(proj_read_cmd_valid),
        .proj_cmd_ready(proj_read_cmd_ready),
        .proj_cmd_base_addr(proj_read_cmd_base_addr),
        .proj_cmd_port_beats(proj_read_cmd_port_beats),
        .proj_cmd_port_mask(proj_read_cmd_port_mask),
        .proj_abort(proj_read_abort), .proj_data(proj_read_data),
        .proj_valid(proj_read_valid), .proj_ready(proj_read_ready),
        .proj_last(proj_read_last), .proj_error(proj_read_error),
        .proj_busy(proj_read_busy),
        .proj_done_valid(proj_read_done_valid),
        .proj_done_ready(proj_read_done_ready),
        .proj_done_error(proj_read_done_error),
        .proj_done_status(proj_read_done_status),
        .rope_cmd_valid(rope_read_cmd_valid),
        .rope_cmd_ready(rope_read_cmd_ready),
        .rope_cmd_base_addr(rope_read_cmd_base_addr),
        .rope_cmd_port_beats(rope_read_cmd_port_beats),
        .rope_cmd_port_mask(rope_read_cmd_port_mask),
        .rope_abort(rope_read_abort), .rope_data(rope_read_data),
        .rope_valid(rope_read_valid), .rope_ready(rope_read_ready),
        .rope_last(rope_read_last), .rope_error(rope_read_error),
        .rope_busy(rope_read_busy),
        .rope_done_valid(rope_read_done_valid),
        .rope_done_ready(rope_read_done_ready),
        .rope_done_error(rope_read_done_error),
        .rope_done_status(rope_read_done_status),
        .small_cmd_valid(small_cmd_valid),
        .small_cmd_ready(small_cmd_ready),
        .small_cmd_base_addr(small_cmd_base_addr),
        .small_cmd_port_beats(small_cmd_port_beats),
        .small_cmd_port_mask(small_cmd_port_mask),
        .small_abort(small_abort), .small_data(small_data),
        .small_valid(small_valid), .small_ready(small_ready),
        .small_last(small_last), .small_error(small_error),
        .small_busy(small_busy), .small_done_valid(small_done_valid),
        .small_done_ready(small_done_ready),
        .small_done_error(small_done_error),
        .small_done_status(small_done_status),
        .svc_cmd_valid(quad_cmd_valid), .svc_cmd_ready(quad_cmd_ready),
        .svc_cmd_base_addr(quad_cmd_base_addr),
        .svc_cmd_port_beats(quad_cmd_port_beats),
        .svc_cmd_port_mask(quad_cmd_port_mask),
        .svc_abort_run(quad_abort), .svc_data(quad_data),
        .svc_valid(quad_valid), .svc_ready(quad_ready),
        .svc_last(quad_last), .svc_error(quad_error),
        .svc_busy(quad_busy), .svc_done_valid(quad_done_valid),
        .svc_done_ready(quad_done_ready),
        .svc_done_error(quad_done_error),
        .svc_done_status(quad_done_status)
    );

     weight_quad128 #(.ADDR_W(ADDR_W)) u_weight_reader (
        .clk(clk), .rst_n(rst_n), .clear(clear_weight_q),
        .abort_run(clear_weight_q || quad_abort), .cmd_valid(quad_cmd_valid),
        .cmd_ready(quad_cmd_ready), .cmd_base_addr(quad_cmd_base_addr),
        .cmd_port_beats(quad_cmd_port_beats),
        .cmd_port_mask(quad_cmd_port_mask), .weight_data(quad_data),
        .weight_valid(quad_valid), .weight_ready(quad_ready),
        .weight_last(quad_last), .weight_error(quad_error),
        .busy(quad_busy), .done_valid(quad_done_valid),
        .done_ready(quad_done_ready), .done_error(quad_done_error),
        .done_status(quad_done_status),
        .m_axi_araddr(weight_axi_araddr),
        .m_axi_arlen(weight_axi_arlen),
        .m_axi_arsize(weight_axi_arsize),
        .m_axi_arburst(weight_axi_arburst),
        .m_axi_arvalid(weight_axi_arvalid),
        .m_axi_arready(weight_axi_arready),
        .m_axi_rdata(weight_axi_rdata), .m_axi_rresp(weight_axi_rresp),
        .m_axi_rlast(weight_axi_rlast),
        .m_axi_rvalid(weight_axi_rvalid),
        .m_axi_rready(weight_axi_rready),
        .metrics_axi_r_beats(metrics_weight_axi_r_beats),
        .metrics_axi_r_gap_ports(metrics_weight_axi_r_gap_ports),
        .metrics_zip_skew(metrics_weight_zip_skew)
    );

    // Locked resident-arena ownership.  No arbiter may switch owners while a
    // response is pending; each owner is the accepted controller leaf request.
    wire vector_owns_embed = vector_owner == VOWNER_EMBED;
    wire vector_owns_norm = vector_owner == VOWNER_NORM;
    wire vector_owns_append = vector_owner == VOWNER_APPEND;

    assign arena_r_wr_valid = vector_owns_embed ? embed_r_wr_valid :
                              gemm_owner_q ? s_r_wr_valid : 1'b0;
    assign arena_r_wr_wave = vector_owns_embed ? embed_r_wr_wave :
                             s_r_wr_wave;
    assign arena_r_wr_addr = vector_owns_embed ? embed_r_wr_addr :
                             s_r_wr_addr;
    assign arena_r_wr_lane_mask = vector_owns_embed ? embed_r_wr_lane_mask :
                                  s_r_wr_lane_mask;
    assign arena_r_wr_data = vector_owns_embed ? embed_r_wr_data :
                             s_r_wr_data;
    assign embed_r_wr_ready = vector_owns_embed && arena_r_wr_ready;
    assign s_r_wr_ready = gemm_owner_q && arena_r_wr_ready;

    assign arena_r_rd_req_valid = vector_owns_norm ? v_r_rd_req_valid :
                                  gemm_owner_q ? s_r_rd_req_valid : 1'b0;
    assign arena_r_rd_req_wave = vector_owns_norm ? v_r_rd_req_wave :
                                 s_r_rd_req_wave;
    assign arena_r_rd_req_addr = vector_owns_norm ? v_r_rd_req_addr :
                                 s_r_rd_req_addr;
    assign v_r_rd_req_ready = vector_owns_norm && arena_r_rd_req_ready;
    assign s_r_rd_req_ready = gemm_owner_q && arena_r_rd_req_ready;
    assign v_r_rd_rsp_valid = vector_owns_norm && arena_r_rd_rsp_valid;
    assign s_r_rd_rsp_valid = gemm_owner_q && arena_r_rd_rsp_valid;
    assign v_r_rd_rsp_data = arena_r_rd_rsp_data;
    assign s_r_rd_rsp_data = arena_r_rd_rsp_data;
    assign arena_r_rd_rsp_ready = vector_owns_norm ? v_r_rd_rsp_ready :
                                  gemm_owner_q ? s_r_rd_rsp_ready : 1'b0;

    assign arena_query_wr_valid = gemm_owner_q && s_query_wr_valid;
    assign arena_query_wr_wave = s_query_wr_wave;
    assign arena_query_wr_addr = s_query_wr_addr;
    assign arena_query_wr_lane_mask = s_query_wr_lane_mask;
    assign arena_query_wr_data = s_query_wr_data;
    assign s_query_wr_ready = gemm_owner_q && arena_query_wr_ready;

    assign arena_query_rd_req_valid = flash_owner_q &&
                                      attention_query_rd_req_valid;
    assign arena_query_rd_req_wave = attention_query_rd_req_wave;
    assign arena_query_rd_req_addr = attention_query_rd_req_addr;
    assign attention_query_rd_req_ready = flash_owner_q &&
                                          arena_query_rd_req_ready;
    assign attention_query_rd_rsp_valid = flash_owner_q &&
                                          arena_query_rd_rsp_valid;
    assign attention_query_rd_rsp_data = arena_query_rd_rsp_data;
    assign arena_query_rd_rsp_ready = flash_owner_q &&
                                      attention_query_rd_rsp_ready;

    assign arena_q8_wr_valid = vector_owns_norm ? v_q8_wr_valid :
                               gemm_owner_q ? s_q8_wr_valid :
                               flash_owner_q ? attention_q8_wr_valid : 1'b0;
    assign arena_q8_wr_wave = vector_owns_norm ? v_q8_wr_wave :
                              gemm_owner_q ? s_q8_wr_wave :
                              attention_q8_wr_wave;
    assign arena_q8_wr_addr = vector_owns_norm ? v_q8_wr_addr :
                              gemm_owner_q ? s_q8_wr_addr :
                              attention_q8_wr_addr;
    assign arena_q8_wr_lane_mask = vector_owns_norm ? v_q8_wr_lane_mask :
                                   gemm_owner_q ? s_q8_wr_lane_mask :
                                   attention_q8_wr_lane_mask;
    assign arena_q8_wr_data = vector_owns_norm ? v_q8_wr_data :
                              gemm_owner_q ? s_q8_wr_data :
                              attention_q8_wr_data;
    assign v_q8_wr_ready = vector_owns_norm && arena_q8_wr_ready;
    assign s_q8_wr_ready = gemm_owner_q && arena_q8_wr_ready;
    assign attention_q8_wr_ready = flash_owner_q && arena_q8_wr_ready;

    assign arena_q8_rd_req_valid = gemm_owner_q && proj_q8_rd_req_valid;
    assign arena_q8_rd_req_wave = proj_q8_rd_req_wave;
    assign arena_q8_rd_req_addr = proj_q8_rd_req_addr;
    assign proj_q8_rd_req_ready = gemm_owner_q && arena_q8_rd_req_ready;
    assign proj_q8_rd_rsp_valid = gemm_owner_q && arena_q8_rd_rsp_valid;
    assign proj_q8_rd_rsp_data = arena_q8_rd_rsp_data;
    assign arena_q8_rd_rsp_ready = gemm_owner_q && proj_q8_rd_rsp_ready;

    assign arena_newkv_wr_valid = gemm_owner_q && s_newkv_wr_valid;
    assign arena_newkv_wr_wave = s_newkv_wr_wave;
    assign arena_newkv_wr_addr = s_newkv_wr_addr;
    assign arena_newkv_wr_lane_mask = s_newkv_wr_lane_mask;
    assign arena_newkv_wr_data = s_newkv_wr_data;
    assign s_newkv_wr_ready = gemm_owner_q && arena_newkv_wr_ready;

    assign arena_newkv_rd_req_valid = vector_owns_append ?
                                      append_newkv_rd_req_valid :
                                      flash_owner_q ?
                                      attention_newkv_rd_req_valid : 1'b0;
    assign arena_newkv_rd_req_wave = vector_owns_append ?
                                     append_newkv_rd_req_wave :
                                     attention_newkv_rd_req_wave;
    assign arena_newkv_rd_req_addr = vector_owns_append ?
                                     append_newkv_rd_req_addr :
                                     attention_newkv_rd_req_addr;
    assign append_newkv_rd_req_ready = vector_owns_append &&
                                       arena_newkv_rd_req_ready;
    assign attention_newkv_rd_req_ready = flash_owner_q &&
                                          arena_newkv_rd_req_ready;
    assign append_newkv_rd_rsp_valid = vector_owns_append &&
                                       arena_newkv_rd_rsp_valid;
    assign attention_newkv_rd_rsp_valid = flash_owner_q &&
                                          arena_newkv_rd_rsp_valid;
    assign append_newkv_rd_rsp_data = arena_newkv_rd_rsp_data;
    assign attention_newkv_rd_rsp_data = arena_newkv_rd_rsp_data;
    assign arena_newkv_rd_rsp_ready = vector_owns_append ?
                                      append_newkv_rd_rsp_ready :
                                      flash_owner_q ?
                                      attention_newkv_rd_rsp_ready : 1'b0;

    // Sample the wide island/owner reduction before it reaches controller
    // admission. This adds only idle recovery latency and keeps held-clear
    // completion fail-closed.
    wire runtime_all_idle = !busy && !gemm_owner_q && !flash_owner_q &&
                            !vector_dispatch_busy && !embed_busy &&
                            !projection_busy && !attention_busy &&
                            !append_busy && !cluster_busy && !rope_busy &&
                            !quad_busy && !hist_k_busy && !hist_v_busy &&
                            !kv_writer_busy;
    always @(posedge clk) begin
        if (!rst_n) begin
            runtime_quiescent_q <= 1'b0;
            clear_done_q <= 1'b0;
        end else begin
            runtime_quiescent_q <= runtime_all_idle;
            if (!clear_distribution_settled)
                clear_done_q <= 1'b0;
            else
                clear_done_q <= runtime_quiescent_q;
        end
    end
    assign clear_done = clear_done_q && clear_distribution_settled;

    // Registered samples at the remaining physical memory islands. Weight
    // traffic is sampled inside weight_quad128; these two clients are already
    // local to this datapath boundary.
    always @(posedge clk) begin
        if (!rst_n || clear_history_q) begin
            metrics_history_axi_r_beats <= 2'd0;
        end else begin
            metrics_history_axi_r_beats <=
                {1'b0, hist_k_axi_rvalid && hist_k_axi_rready} +
                {1'b0, hist_v_axi_rvalid && hist_v_axi_rready};
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear_kv_writer_q)
            metrics_kv_axi_w_beat <= 1'b0;
        else
            metrics_kv_axi_w_beat <= kv_axi_wvalid && kv_axi_wready;
    end

    reg top_protocol_error_q;
    wire flash_contract_bad = flash_req_valid && flash_req_ready &&
        ((flash_req_head_dim != 8'd128) ||
         (flash_req_kv_heads != 4'd8) ||
         (flash_req_group_q_heads != 4'd8) ||
         !flash_req_kv_single_pass ||
         ((flash_req_q_heads == 6'd16) ?
          ((flash_req_head_group_count != 3'd2) ||
           (flash_req_group_kv_heads != 4'd4)) :
          ((flash_req_q_heads != 6'd32) ||
           (flash_req_head_group_count != 3'd4) ||
           (flash_req_group_kv_heads != 4'd2))));
    wire arena_collision =
        (embed_r_wr_valid && s_r_wr_valid) ||
        (v_r_rd_req_valid && s_r_rd_req_valid) ||
        ((v_q8_wr_valid && s_q8_wr_valid) ||
         (v_q8_wr_valid && attention_q8_wr_valid) ||
         (s_q8_wr_valid && attention_q8_wr_valid)) ||
        (append_newkv_rd_req_valid && attention_newkv_rd_req_valid);
    wire owner_collision = (gemm_owner_q && flash_owner_q) ||
                           (gemm_owner_q && vector_dispatch_busy) ||
                           (flash_owner_q && vector_dispatch_busy);
    wire rope_contract_bad = s_rope_req_valid && s_rope_req_ready &&
        ((s_rope_req_records !=
          ((popcount8(s_rope_req_token_mask) > 4) ? 8'd128 : 8'd64)) ||
         (s_rope_req_token_mask == 8'd0));

    always @(posedge clk) begin
        if (!rst_n || clear_owner_q)
            top_protocol_error_q <= 1'b0;
        else if (flash_contract_bad || arena_collision || owner_collision ||
                 rope_contract_bad)
            top_protocol_error_q <= 1'b1;
    end

    assign protocol_error = top_protocol_error_q ||
                            vector_dispatch_error || cluster_protocol_error;

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !run_clear && clear_distribution_settled) begin
            assert(!arena_collision);
            assert(!owner_collision);
            if (flash_req_valid && flash_req_ready)
                assert(!flash_contract_bad);
            if (s_rope_req_valid && s_rope_req_ready)
                assert(!rope_contract_bad);
        end
    end
`endif
endmodule

`default_nettype wire
