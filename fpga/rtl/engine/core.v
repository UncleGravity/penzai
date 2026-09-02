`default_nettype none

`include "engine_defs.vh"

// Clean EXEC_TILE control plane. Arithmetic leaves are explicit registered
// services; this module owns the one legal stage order, layer iteration,
// model_spec immutability, resident arenas, and the sole KV commit boundary.
module engine_core (
    input  wire          clk,
    input  wire          rst_n,
    // Destructive active-tile cancellation. The sealed model model_spec and RAM
    // payloads survive; controller state and all pending arena responses do not.
    input  wire          run_clear,

    // Immutable model_spec load/seal interface.
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

    // One full-model tile command. Lane masks must be low-contiguous.
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

    // This is the only publication point for the speculative NewKV arena.
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

    output wire          busy,
    output wire [5:0]    debug_layer,
    output wire [4:0]    debug_stage,
    output wire          metrics_stage_active,
    output wire [4:0]    metrics_stage,
    output reg           trace_valid,
    output reg  [5:0]    trace_layer,
    output reg  [4:0]    trace_stage,

    // Token-wide resident arena service ports. Arithmetic services connect
    // here; payloads remain observable at this boundary and in synthesis.
    input  wire          arena_r_wr_valid,
    output wire          arena_r_wr_ready,
    input  wire          arena_r_wr_wave,
    input  wire [11:0]   arena_r_wr_addr,
    input  wire [3:0]    arena_r_wr_lane_mask,
    input  wire [127:0]  arena_r_wr_data,
    input  wire          arena_r_rd_req_valid,
    output wire          arena_r_rd_req_ready,
    input  wire          arena_r_rd_req_wave,
    input  wire [11:0]   arena_r_rd_req_addr,
    output wire          arena_r_rd_rsp_valid,
    input  wire          arena_r_rd_rsp_ready,
    output wire [127:0]  arena_r_rd_rsp_data,
    input  wire          arena_query_wr_valid,
    output wire          arena_query_wr_ready,
    input  wire          arena_query_wr_wave,
    input  wire [11:0]   arena_query_wr_addr,
    input  wire [3:0]    arena_query_wr_lane_mask,
    input  wire [127:0]  arena_query_wr_data,
    input  wire          arena_query_rd_req_valid,
    output wire          arena_query_rd_req_ready,
    input  wire          arena_query_rd_req_wave,
    input  wire [11:0]   arena_query_rd_req_addr,
    output wire          arena_query_rd_rsp_valid,
    input  wire          arena_query_rd_rsp_ready,
    output wire [127:0]  arena_query_rd_rsp_data,
    input  wire          arena_q8_wr_valid,
    output wire          arena_q8_wr_ready,
    input  wire          arena_q8_wr_wave,
    input  wire [8:0]    arena_q8_wr_addr,
    input  wire [3:0]    arena_q8_wr_lane_mask,
    input  wire [1087:0] arena_q8_wr_data,
    input  wire          arena_q8_rd_req_valid,
    output wire          arena_q8_rd_req_ready,
    input  wire          arena_q8_rd_req_wave,
    input  wire [8:0]    arena_q8_rd_req_addr,
    output wire          arena_q8_rd_rsp_valid,
    input  wire          arena_q8_rd_rsp_ready,
    output wire [1087:0] arena_q8_rd_rsp_data,
    input  wire          arena_newkv_wr_valid,
    output wire          arena_newkv_wr_ready,
    input  wire          arena_newkv_wr_wave,
    input  wire [10:0]   arena_newkv_wr_addr,
    input  wire [3:0]    arena_newkv_wr_lane_mask,
    input  wire [63:0]   arena_newkv_wr_data,
    input  wire          arena_newkv_rd_req_valid,
    output wire          arena_newkv_rd_req_ready,
    input  wire          arena_newkv_rd_req_wave,
    input  wire [10:0]   arena_newkv_rd_req_addr,
    output wire          arena_newkv_rd_rsp_valid,
    input  wire          arena_newkv_rd_rsp_ready,
    output wire [63:0]   arena_newkv_rd_rsp_data,

    // Low-bit GEMM leaf.
    output wire          gemm_req_valid,
    input  wire          gemm_req_ready,
    output reg  [2:0]    gemm_req_op,
    output wire [5:0]    gemm_req_layer,
    output wire [3:0]    gemm_req_token_count,
    output wire [7:0]    gemm_req_lane_mask,
    output reg  [63:0]   gemm_req_addr0,
    output reg  [63:0]   gemm_req_addr1,
    output reg  [63:0]   gemm_req_addr2,
    output reg  [63:0]   gemm_req_addr3,
    output wire [7:0]    gemm_req_hidden_blocks,
    output wire [9:0]    gemm_req_ffn_blocks,
    output wire [5:0]    gemm_req_q_heads,
    output wire [3:0]    gemm_req_kv_heads,
    output wire [7:0]    gemm_req_head_dim,
    output wire [16:0]   gemm_req_position_base,
    output wire [31:0]   gemm_req_epsilon,
    output wire [17:0]   gemm_req_vocab_rows,
    output wire [1:0]    gemm_req_weight_fmt,
    input  wire          gemm_done_valid,
    output wire          gemm_done_ready,
    input  wire          gemm_done_error,
    input  wire [7:0]    gemm_done_status,

    // FlashAttention leaf.
    output wire          flash_req_valid,
    input  wire          flash_req_ready,
    output wire [5:0]    flash_req_layer,
    output wire [3:0]    flash_req_token_count,
    output wire [7:0]    flash_req_lane_mask,
    output wire [5:0]    flash_req_q_heads,
    output wire [3:0]    flash_req_kv_heads,
    output wire [7:0]    flash_req_head_dim,
    output wire [2:0]    flash_req_head_group_count,
    output wire [3:0]    flash_req_group_q_heads,
    output wire [3:0]    flash_req_group_kv_heads,
    output wire          flash_req_kv_single_pass,
    output wire [16:0]   flash_req_position_base,
    output wire [63:0]   flash_req_kv_base,
    input  wire          flash_done_valid,
    output wire          flash_done_ready,
    input  wire          flash_done_error,
    input  wire [7:0]    flash_done_status,

    // Shared vector leaf: embedding, norms, RoPE, append, activation, residual.
    output wire          vector_req_valid,
    input  wire          vector_req_ready,
    output reg  [3:0]    vector_req_op,
    output wire [5:0]    vector_req_layer,
    output wire [3:0]    vector_req_token_count,
    output wire [7:0]    vector_req_lane_mask,
    output wire [255:0]  vector_req_token_ids,
    output reg  [63:0]   vector_req_addr0,
    output reg  [63:0]   vector_req_addr1,
    output reg  [63:0]   vector_req_addr2,
    output wire [7:0]    vector_req_hidden_blocks,
    output wire [9:0]    vector_req_ffn_blocks,
    output wire [17:0]   vector_req_vocab_rows,
    output wire [1:0]    vector_req_weight_fmt,
    output wire [16:0]   vector_req_position_base,
    output wire [63:0]   vector_req_kv_base,
    input  wire          vector_done_valid,
    output wire          vector_done_ready,
    input  wire          vector_done_error,
    input  wire [7:0]    vector_done_status
);
    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_LOAD_REQ   = 3'd1;
    localparam [2:0] ST_LOAD_WAIT  = 3'd2;
    localparam [2:0] ST_ISSUE      = 3'd3;
    localparam [2:0] ST_WAIT       = 3'd4;
    localparam [2:0] ST_COMMIT     = 3'd5;
    localparam [2:0] ST_ERROR      = 3'd6;

    reg [2:0] state_q;
    reg [5:0] layer_q;
    reg [4:0] stage_q;
    reg [2:0] desc_word_q;
    reg [63:0] layer_desc [0:`MODEL_LAYER_WORDS-1];

    reg [31:0] cmd_tag_q;
    reg [3:0] token_count_q;
    reg [7:0] lane_mask_q;
    reg [255:0] token_ids_q;
    reg [16:0] position_base_q;
    reg [63:0] kv_base_q;
    reg [16:0] kv_capacity_q;
    reg [63:0] layer_kv_base_q;
    reg emit_logits_q;

    reg [15:0] error_code_q;
    reg [7:0] error_detail_q;
    reg [5:0] error_layer_q;
    reg [4:0] error_stage_q;

    wire [5:0] p_layer_count;
    wire [7:0] p_hidden_blocks;
    wire [9:0] p_ffn_blocks;
    wire [5:0] p_q_heads;
    wire [3:0] p_kv_heads;
    wire [7:0] p_head_dim;
    wire [1:0] p_weight_fmt;
    wire [16:0] p_context_limit;
    wire [17:0] p_vocab_rows;
    wire [63:0] p_embed_addr;
    wire [63:0] p_lm_head_addr;
    wire [63:0] p_final_norm_addr;
    wire [63:0] p_rope_table_addr;

    wire layer_rd_req_valid;
    wire layer_rd_req_ready;
    wire [8:0] layer_rd_req_addr;
    wire layer_rd_rsp_valid;
    wire [63:0] layer_rd_rsp_data;

    function [7:0] token_prefix_mask;
        input [3:0] count;
        begin
            case (count)
                4'd1: token_prefix_mask = 8'h01;
                4'd2: token_prefix_mask = 8'h03;
                4'd3: token_prefix_mask = 8'h07;
                4'd4: token_prefix_mask = 8'h0f;
                4'd5: token_prefix_mask = 8'h1f;
                4'd6: token_prefix_mask = 8'h3f;
                4'd7: token_prefix_mask = 8'h7f;
                4'd8: token_prefix_mask = 8'hff;
                default: token_prefix_mask = 8'h00;
            endcase
        end
    endfunction

    function [7:0] token_final_mask;
        input [3:0] count;
        begin
            case (count)
                4'd1: token_final_mask = 8'h01;
                4'd2: token_final_mask = 8'h02;
                4'd3: token_final_mask = 8'h04;
                4'd4: token_final_mask = 8'h08;
                4'd5: token_final_mask = 8'h10;
                4'd6: token_final_mask = 8'h20;
                4'd7: token_final_mask = 8'h40;
                4'd8: token_final_mask = 8'h80;
                default: token_final_mask = 8'h00;
            endcase
        end
    endfunction

    wire stage_is_gemm = (stage_q == `ENGINE_STAGE_QKV_ROPE) ||
                         (stage_q == `ENGINE_STAGE_O_PROJ_RESID) ||
                         (stage_q == `ENGINE_STAGE_GATE_UP_SWIGLU_Q8) ||
                         (stage_q == `ENGINE_STAGE_DOWN_RESID) ||
                         (stage_q == `ENGINE_STAGE_LM_HEAD);
    wire stage_is_flash = stage_q == `ENGINE_STAGE_ATTENTION;
    wire stage_is_vector = !stage_is_gemm && !stage_is_flash;

    wire gemm_issue_fire = gemm_req_valid && gemm_req_ready;
    wire flash_issue_fire = flash_req_valid && flash_req_ready;
    wire vector_issue_fire = vector_req_valid && vector_req_ready;
    wire issue_fire = gemm_issue_fire || flash_issue_fire || vector_issue_fire;

    wire active_done_valid = stage_is_gemm ? gemm_done_valid :
                             stage_is_flash ? flash_done_valid :
                                              vector_done_valid;
    wire active_done_error = stage_is_gemm ? gemm_done_error :
                             stage_is_flash ? flash_done_error :
                                              vector_done_error;
    wire [7:0] active_done_status = stage_is_gemm ? gemm_done_status :
                                          stage_is_flash ? flash_done_status :
                                                           vector_done_status;

    wire [17:0] incoming_context_end =
        {1'b0, cmd_position_base} + {14'd0, cmd_token_count};
    wire command_model_spec_ok = (cmd_model_spec_id == active_model_spec_id) &&
                              (cmd_model_spec_hash == active_model_spec_hash);
    wire command_count_ok = (cmd_token_count >= 4'd1) &&
                            (cmd_token_count <= `ENGINE_MAX_TOKENS);
    wire command_mask_ok = cmd_lane_mask == token_prefix_mask(cmd_token_count);
    wire command_context_ok = (incoming_context_end <= {1'b0, p_context_limit});
    wire command_capacity_ok = (cmd_kv_capacity != 17'd0) &&
                               (cmd_kv_capacity <= p_context_limit);
    wire command_capacity_extent_ok =
        incoming_context_end <= {1'b0, cmd_kv_capacity};
    wire command_kv_ok = (cmd_kv_base != 64'd0) &&
                         (cmd_kv_base[11:0] == 12'd0);

    assign busy = state_q != ST_IDLE;
    assign interface_version = `ENGINE_INTERFACE_VERSION;
    assign layer_layout_hash = `MODEL_LAYOUT_HASH;
    assign debug_layer = layer_q;
    assign debug_stage = stage_q;
    // state_q and stage_q are already registered controller state. These taps
    // add observation fanout only and cannot affect request scheduling.
    assign metrics_stage_active = (state_q == ST_ISSUE) ||
                                  (state_q == ST_WAIT);
    assign metrics_stage = stage_q;
    assign cmd_ready = !run_clear && (state_q == ST_IDLE) && model_spec_sealed &&
                       !model_spec_cfg_error_valid && !model_spec_clear_valid;

    assign commit_valid = state_q == ST_COMMIT;
    assign commit_tag = cmd_tag_q;
    assign commit_model_spec_id = active_model_spec_id;
    assign commit_model_spec_hash = active_model_spec_hash;
    assign commit_token_count = token_count_q;
    assign commit_kv_length = position_base_q + {13'd0, token_count_q};
    assign commit_logits_valid = emit_logits_q;

    assign error_valid = state_q == ST_ERROR;
    assign error_tag = cmd_tag_q;
    assign error_code = error_code_q;
    assign error_detail = error_detail_q;
    assign error_layer = error_layer_q;
    assign error_stage = error_stage_q;

    assign layer_rd_req_valid = state_q == ST_LOAD_REQ;
    assign layer_rd_req_addr = {layer_q, 3'b000} + {6'd0, desc_word_q};

    assign gemm_req_valid = (state_q == ST_ISSUE) && stage_is_gemm;
    assign gemm_req_layer = layer_q;
    assign gemm_req_token_count = token_count_q;
    // Prefill publishes logits only for the final prompt token.  The shared
    // projection datapath accepts arbitrary nonzero masks, so LM-head work is
    // not repeated across all eight resident contexts.
    assign gemm_req_lane_mask = (stage_q == `ENGINE_STAGE_LM_HEAD) ?
                                token_final_mask(token_count_q) : lane_mask_q;
    assign gemm_req_hidden_blocks = p_hidden_blocks;
    assign gemm_req_ffn_blocks = p_ffn_blocks;
    assign gemm_req_q_heads = p_q_heads;
    assign gemm_req_kv_heads = p_kv_heads;
    assign gemm_req_head_dim = p_head_dim;
    assign gemm_req_position_base = position_base_q;
    // Every supported Bonsai model_spec uses RMS epsilon 1e-6. The immutable
    // model_spec ID/hash binds that semantic constant without a hot descriptor.
    assign gemm_req_epsilon = 32'h3586_37bd;
    assign gemm_req_vocab_rows = p_vocab_rows;
    assign gemm_req_weight_fmt = p_weight_fmt;
    assign gemm_done_ready = (state_q == ST_WAIT) && stage_is_gemm;

    assign flash_req_valid = (state_q == ST_ISSUE) && stage_is_flash;
    assign flash_req_layer = layer_q;
    assign flash_req_token_count = token_count_q;
    assign flash_req_lane_mask = lane_mask_q;
    assign flash_req_q_heads = p_q_heads;
    assign flash_req_kv_heads = p_kv_heads;
    assign flash_req_head_dim = p_head_dim;
    // TILE8_HEAD8_LAYOUT schedules disjoint head groups. Each group owns its
    // KV heads, so the complete K/V extent is consumed exactly once per tile.
    assign flash_req_head_group_count = (p_q_heads == 6'd16) ? 3'd2 : 3'd4;
    assign flash_req_group_q_heads = 4'd8;
    assign flash_req_group_kv_heads = (p_q_heads == 6'd16) ? 4'd4 : 4'd2;
    assign flash_req_kv_single_pass = 1'b1;
    assign flash_req_position_base = position_base_q;
    // Keep the compact per-layer KV address as controller state. Advancing the
    // cursor during the descriptor-load interval removes a layer*capacity
    // multiply from every Flash and append request without adding a stage.
    assign flash_req_kv_base = layer_kv_base_q;
    assign flash_done_ready = (state_q == ST_WAIT) && stage_is_flash;

    assign vector_req_valid = (state_q == ST_ISSUE) && stage_is_vector;
    assign vector_req_layer = layer_q;
    assign vector_req_token_count = token_count_q;
    assign vector_req_lane_mask = lane_mask_q;
    assign vector_req_token_ids = token_ids_q;
    assign vector_req_hidden_blocks = p_hidden_blocks;
    assign vector_req_ffn_blocks = p_ffn_blocks;
    assign vector_req_vocab_rows = p_vocab_rows;
    assign vector_req_weight_fmt = p_weight_fmt;
    assign vector_req_position_base = position_base_q;
    assign vector_req_kv_base = kv_base_q;
    assign vector_done_ready = (state_q == ST_WAIT) && stage_is_vector;

    always @* begin
        gemm_req_op = `PROJECTION_OP_QKV;
        gemm_req_addr0 = 64'd0;
        gemm_req_addr1 = 64'd0;
        gemm_req_addr2 = 64'd0;
        gemm_req_addr3 = 64'd0;
        case (stage_q)
            `ENGINE_STAGE_QKV_ROPE: begin
                gemm_req_op = `PROJECTION_OP_QKV;
                // Fused rows are GQA-group ordered per KV head:
                // {all Q heads mapped to this KV head, K, V}.
                gemm_req_addr0 = layer_desc[`MODEL_LAYER_FUSED_QKV];
                gemm_req_addr1 = layer_desc[`MODEL_LAYER_Q_NORM];
                gemm_req_addr2 = layer_desc[`MODEL_LAYER_K_NORM];
                gemm_req_addr3 = p_rope_table_addr;
            end
            `ENGINE_STAGE_O_PROJ_RESID: begin
                gemm_req_op = `PROJECTION_OP_O;
                gemm_req_addr0 = layer_desc[`MODEL_LAYER_O];
            end
            `ENGINE_STAGE_GATE_UP_SWIGLU_Q8: begin
                gemm_req_op = `PROJECTION_OP_GATE_UP;
                gemm_req_addr0 = layer_desc[`MODEL_LAYER_FUSED_GATE_UP];
            end
            `ENGINE_STAGE_DOWN_RESID: begin
                gemm_req_op = `PROJECTION_OP_DOWN;
                gemm_req_addr0 = layer_desc[`MODEL_LAYER_DOWN];
            end
            `ENGINE_STAGE_LM_HEAD: begin
                gemm_req_op = `PROJECTION_OP_LM_HEAD;
                gemm_req_addr0 = p_lm_head_addr;
            end
            default: begin end
        endcase
    end

    always @* begin
        vector_req_op = `VECTOR_OP_EMBED;
        vector_req_addr0 = 64'd0;
        vector_req_addr1 = 64'd0;
        vector_req_addr2 = 64'd0;
        case (stage_q)
            `ENGINE_STAGE_EMBED: begin
                vector_req_op = `VECTOR_OP_EMBED;
                vector_req_addr0 = p_embed_addr;
            end
            `ENGINE_STAGE_ATTN_NORM: begin
                vector_req_op = `VECTOR_OP_ATTN_NORM;
                vector_req_addr0 = layer_desc[`MODEL_LAYER_ATTN_NORM];
            end
            `ENGINE_STAGE_KV_APPEND: begin
                vector_req_op = `VECTOR_OP_KV_APPEND;
                vector_req_addr0 = layer_kv_base_q;
            end
            `ENGINE_STAGE_FFN_NORM: begin
                vector_req_op = `VECTOR_OP_FFN_NORM;
                vector_req_addr0 = layer_desc[`MODEL_LAYER_FFN_NORM];
            end
            `ENGINE_STAGE_FINAL_NORM: begin
                vector_req_op = `VECTOR_OP_FINAL_NORM;
                vector_req_addr0 = p_final_norm_addr;
            end
            default: begin end
        endcase
    end

    model_spec_store u_model_spec (
        .clk(clk), .rst_n(rst_n),
        .mutation_allowed(state_q == ST_IDLE),
        .clear_valid(model_spec_clear_valid), .clear_ready(model_spec_clear_ready),
        .begin_valid(model_spec_begin_valid), .begin_ready(model_spec_begin_ready),
        .begin_model_spec_id(model_spec_begin_id),
        .begin_model_spec_hash(model_spec_begin_hash),
        .begin_layer_count(model_spec_begin_layer_count),
        .begin_hidden_blocks(model_spec_begin_hidden_blocks),
        .begin_ffn_blocks(model_spec_begin_ffn_blocks),
        .begin_q_heads(model_spec_begin_q_heads),
        .begin_kv_heads(model_spec_begin_kv_heads),
        .begin_head_dim(model_spec_begin_head_dim),
        .begin_weight_fmt(model_spec_begin_weight_fmt),
        .begin_context_limit(model_spec_begin_context_limit),
        .begin_vocab_rows(model_spec_begin_vocab_rows),
        .begin_embed_addr(model_spec_begin_embed_addr),
        .begin_lm_head_addr(model_spec_begin_lm_head_addr),
        .begin_final_norm_addr(model_spec_begin_final_norm_addr),
        .begin_rope_table_addr(model_spec_begin_rope_table_addr),
        .layer_wr_valid(model_spec_layer_wr_valid),
        .layer_wr_ready(model_spec_layer_wr_ready),
        .layer_wr_layer(model_spec_layer_wr_layer),
        .layer_wr_word(model_spec_layer_wr_word),
        .layer_wr_data(model_spec_layer_wr_data),
        .seal_valid(model_spec_seal_valid), .seal_ready(model_spec_seal_ready),
        .cfg_error_valid(model_spec_cfg_error_valid),
        .cfg_error_ready(model_spec_cfg_error_ready),
        .cfg_error_code(model_spec_cfg_error_code),
        .cfg_error_layer(model_spec_cfg_error_layer),
        .cfg_error_word(model_spec_cfg_error_word),
        .model_spec_loading(model_spec_loading), .model_spec_sealed(model_spec_sealed),
        .model_spec_id(active_model_spec_id), .model_spec_hash(active_model_spec_hash),
        .model_spec_layer_count(p_layer_count),
        .model_spec_hidden_blocks(p_hidden_blocks),
        .model_spec_ffn_blocks(p_ffn_blocks),
        .model_spec_q_heads(p_q_heads), .model_spec_kv_heads(p_kv_heads),
        .model_spec_head_dim(p_head_dim), .model_spec_weight_fmt(p_weight_fmt),
        .model_spec_context_limit(p_context_limit),
        .model_spec_vocab_rows(p_vocab_rows),
        .model_spec_embed_addr(p_embed_addr),
        .model_spec_lm_head_addr(p_lm_head_addr),
        .model_spec_final_norm_addr(p_final_norm_addr),
        .model_spec_rope_table_addr(p_rope_table_addr),
        .layer_rd_req_valid(layer_rd_req_valid),
        .layer_rd_req_ready(layer_rd_req_ready),
        .layer_rd_req_addr(layer_rd_req_addr),
        .layer_rd_rsp_valid(layer_rd_rsp_valid),
        .layer_rd_rsp_data(layer_rd_rsp_data)
    );

    resident_arenas u_arenas (
        .clk(clk), .rst_n(rst_n), .clear(run_clear),
        .r_wr_valid(arena_r_wr_valid), .r_wr_ready(arena_r_wr_ready),
        .r_wr_wave(arena_r_wr_wave),
        .r_wr_addr(arena_r_wr_addr), .r_wr_lane_mask(arena_r_wr_lane_mask),
        .r_wr_data(arena_r_wr_data),
        .r_rd_req_valid(arena_r_rd_req_valid),
        .r_rd_req_ready(arena_r_rd_req_ready),
        .r_rd_req_wave(arena_r_rd_req_wave),
        .r_rd_req_addr(arena_r_rd_req_addr),
        .r_rd_rsp_valid(arena_r_rd_rsp_valid),
        .r_rd_rsp_ready(arena_r_rd_rsp_ready),
        .r_rd_rsp_data(arena_r_rd_rsp_data),
        .query_wr_valid(arena_query_wr_valid),
        .query_wr_ready(arena_query_wr_ready),
        .query_wr_wave(arena_query_wr_wave),
        .query_wr_addr(arena_query_wr_addr),
        .query_wr_lane_mask(arena_query_wr_lane_mask),
        .query_wr_data(arena_query_wr_data),
        .query_rd_req_valid(arena_query_rd_req_valid),
        .query_rd_req_ready(arena_query_rd_req_ready),
        .query_rd_req_wave(arena_query_rd_req_wave),
        .query_rd_req_addr(arena_query_rd_req_addr),
        .query_rd_rsp_valid(arena_query_rd_rsp_valid),
        .query_rd_rsp_ready(arena_query_rd_rsp_ready),
        .query_rd_rsp_data(arena_query_rd_rsp_data),
        .q8_wr_valid(arena_q8_wr_valid), .q8_wr_ready(arena_q8_wr_ready),
        .q8_wr_wave(arena_q8_wr_wave),
        .q8_wr_addr(arena_q8_wr_addr),
        .q8_wr_lane_mask(arena_q8_wr_lane_mask), .q8_wr_data(arena_q8_wr_data),
        .q8_rd_req_valid(arena_q8_rd_req_valid),
        .q8_rd_req_ready(arena_q8_rd_req_ready),
        .q8_rd_req_wave(arena_q8_rd_req_wave),
        .q8_rd_req_addr(arena_q8_rd_req_addr),
        .q8_rd_rsp_valid(arena_q8_rd_rsp_valid),
        .q8_rd_rsp_ready(arena_q8_rd_rsp_ready),
        .q8_rd_rsp_data(arena_q8_rd_rsp_data),
        .newkv_wr_valid(arena_newkv_wr_valid),
        .newkv_wr_ready(arena_newkv_wr_ready),
        .newkv_wr_wave(arena_newkv_wr_wave),
        .newkv_wr_addr(arena_newkv_wr_addr),
        .newkv_wr_lane_mask(arena_newkv_wr_lane_mask),
        .newkv_wr_data(arena_newkv_wr_data),
        .newkv_rd_req_valid(arena_newkv_rd_req_valid),
        .newkv_rd_req_ready(arena_newkv_rd_req_ready),
        .newkv_rd_req_wave(arena_newkv_rd_req_wave),
        .newkv_rd_req_addr(arena_newkv_rd_req_addr),
        .newkv_rd_rsp_valid(arena_newkv_rd_rsp_valid),
        .newkv_rd_rsp_ready(arena_newkv_rd_rsp_ready),
        .newkv_rd_rsp_data(arena_newkv_rd_rsp_data)
    );

    integer j;
    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            layer_q <= 6'd0;
            stage_q <= `ENGINE_STAGE_EMBED;
            desc_word_q <= 3'd0;
            cmd_tag_q <= 32'd0;
            token_count_q <= 4'd0;
            lane_mask_q <= 8'd0;
            token_ids_q <= 256'd0;
            position_base_q <= 17'd0;
            kv_base_q <= 64'd0;
            kv_capacity_q <= 17'd0;
            layer_kv_base_q <= 64'd0;
            emit_logits_q <= 1'b0;
            error_code_q <= 16'd0;
            error_detail_q <= 8'd0;
            error_layer_q <= 6'd0;
            error_stage_q <= 5'd0;
            trace_valid <= 1'b0;
            trace_layer <= 6'd0;
            trace_stage <= 5'd0;
            for (j = 0; j < `MODEL_LAYER_WORDS; j = j + 1)
                layer_desc[j] <= 64'd0;
        end else if (run_clear) begin
            state_q <= ST_IDLE;
            layer_q <= 6'd0;
            stage_q <= `ENGINE_STAGE_EMBED;
            desc_word_q <= 3'd0;
            cmd_tag_q <= 32'd0;
            token_count_q <= 4'd0;
            lane_mask_q <= 8'd0;
            token_ids_q <= 256'd0;
            position_base_q <= 17'd0;
            kv_base_q <= 64'd0;
            kv_capacity_q <= 17'd0;
            layer_kv_base_q <= 64'd0;
            emit_logits_q <= 1'b0;
            error_code_q <= 16'd0;
            error_detail_q <= 8'd0;
            error_layer_q <= 6'd0;
            error_stage_q <= 5'd0;
            trace_valid <= 1'b0;
            trace_layer <= 6'd0;
            trace_stage <= 5'd0;
        end else begin
            trace_valid <= 1'b0;

            case (state_q)
                ST_IDLE: begin
                    if (cmd_valid && cmd_ready) begin
                        cmd_tag_q <= cmd_tag;
                        token_count_q <= cmd_token_count;
                        lane_mask_q <= cmd_lane_mask;
                        token_ids_q <= cmd_token_ids;
                        position_base_q <= cmd_position_base;
                        kv_base_q <= cmd_kv_base;
                        kv_capacity_q <= cmd_kv_capacity;
                        layer_kv_base_q <= cmd_kv_base;
                        emit_logits_q <= cmd_emit_logits;
                        layer_q <= 6'd0;
                        stage_q <= `ENGINE_STAGE_EMBED;

                        if (!command_model_spec_ok) begin
                            error_code_q <= `ENGINE_ERROR_MODEL_SPEC;
                            error_detail_q <= 8'd0;
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else if (!command_count_ok) begin
                            error_code_q <= `ENGINE_ERROR_TOKEN_COUNT;
                            error_detail_q <= {4'd0, cmd_token_count};
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else if (!command_mask_ok) begin
                            error_code_q <= `ENGINE_ERROR_LANE_MASK;
                            error_detail_q <= cmd_lane_mask;
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else if (!command_context_ok) begin
                            error_code_q <= `ENGINE_ERROR_CONTEXT;
                            error_detail_q <= 8'd0;
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else if (!command_capacity_ok ||
                                     !command_capacity_extent_ok) begin
                            error_code_q <= `ENGINE_ERROR_CONTEXT;
                            error_detail_q <= cmd_kv_capacity[7:0];
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else if (!command_kv_ok) begin
                            error_code_q <= `ENGINE_ERROR_KV_BASE;
                            error_detail_q <= cmd_kv_base[7:0];
                            error_layer_q <= 6'd0;
                            error_stage_q <= `ENGINE_STAGE_EMBED;
                            state_q <= ST_ERROR;
                        end else begin
                            state_q <= ST_ISSUE;
                        end
                    end
                end

                ST_LOAD_REQ: begin
                    if (layer_rd_req_valid && layer_rd_req_ready)
                        state_q <= ST_LOAD_WAIT;
                end

                ST_LOAD_WAIT: begin
                    if (layer_rd_rsp_valid) begin
                        layer_desc[desc_word_q] <= layer_rd_rsp_data;
                        if (desc_word_q == 3'd7) begin
                            desc_word_q <= 3'd0;
                            stage_q <= `ENGINE_STAGE_ATTN_NORM;
                            state_q <= ST_ISSUE;
                        end else begin
                            desc_word_q <= desc_word_q + 1'b1;
                            state_q <= ST_LOAD_REQ;
                        end
                    end
                end

                ST_ISSUE: begin
                    if (issue_fire) begin
                        trace_valid <= 1'b1;
                        trace_layer <= layer_q;
                        trace_stage <= stage_q;
                        state_q <= ST_WAIT;
                    end
                end

                ST_WAIT: begin
                    if (active_done_valid) begin
                        if (active_done_error) begin
                            error_code_q <= `ENGINE_ERROR_LEAF;
                            error_detail_q <= active_done_status;
                            error_layer_q <= layer_q;
                            error_stage_q <= stage_q;
                            state_q <= ST_ERROR;
                        end else begin
                            case (stage_q)
                                `ENGINE_STAGE_EMBED: begin
                                    desc_word_q <= 3'd0;
                                    state_q <= ST_LOAD_REQ;
                                end
                                `ENGINE_STAGE_ATTN_NORM:
                                    stage_q <= `ENGINE_STAGE_QKV_ROPE;
                                `ENGINE_STAGE_QKV_ROPE:
                                    stage_q <= `ENGINE_STAGE_KV_APPEND;
                                `ENGINE_STAGE_KV_APPEND:
                                    stage_q <= `ENGINE_STAGE_ATTENTION;
                                `ENGINE_STAGE_ATTENTION:
                                    stage_q <= `ENGINE_STAGE_O_PROJ_RESID;
                                `ENGINE_STAGE_O_PROJ_RESID:
                                    stage_q <= `ENGINE_STAGE_FFN_NORM;
                                `ENGINE_STAGE_FFN_NORM:
                                    stage_q <= `ENGINE_STAGE_GATE_UP_SWIGLU_Q8;
                                `ENGINE_STAGE_GATE_UP_SWIGLU_Q8:
                                    stage_q <= `ENGINE_STAGE_DOWN_RESID;
                                `ENGINE_STAGE_DOWN_RESID: begin
                                    if (layer_q + 1'b1 < p_layer_count) begin
                                        layer_q <= layer_q + 1'b1;
                                        layer_kv_base_q <= layer_kv_base_q +
                                            {35'd0, kv_capacity_q, 12'd0};
                                        desc_word_q <= 3'd0;
                                        state_q <= ST_LOAD_REQ;
                                    end else if (emit_logits_q) begin
                                        stage_q <= `ENGINE_STAGE_FINAL_NORM;
                                        state_q <= ST_ISSUE;
                                    end else begin
                                        state_q <= ST_COMMIT;
                                    end
                                end
                                `ENGINE_STAGE_FINAL_NORM:
                                    stage_q <= `ENGINE_STAGE_LM_HEAD;
                                `ENGINE_STAGE_LM_HEAD:
                                    state_q <= ST_COMMIT;
                                default: begin
                                    error_code_q <= `ENGINE_ERROR_LEAF;
                                    error_detail_q <= 8'hff;
                                    error_layer_q <= layer_q;
                                    error_stage_q <= stage_q;
                                    state_q <= ST_ERROR;
                                end
                            endcase
                            if ((stage_q != `ENGINE_STAGE_EMBED) &&
                                (stage_q != `ENGINE_STAGE_DOWN_RESID) &&
                                (stage_q != `ENGINE_STAGE_LM_HEAD))
                                state_q <= ST_ISSUE;
                        end
                    end
                end

                ST_COMMIT: begin
                    if (commit_valid && commit_ready)
                        state_q <= ST_IDLE;
                end

                ST_ERROR: begin
                    if (error_valid && error_ready)
                        state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
