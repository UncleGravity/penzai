`default_nettype none

`include "engine_defs.vh"

// One immutable model model_spec and its bounded per-layer address table.
// Loading and execution are mutually exclusive. Payload RAM is not cleared;
// validity is carried only by the header/loading/sealed state and word masks.
module model_spec_store (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         mutation_allowed,

    input  wire         clear_valid,
    output wire         clear_ready,

    input  wire         begin_valid,
    output wire         begin_ready,
    input  wire [31:0]  begin_model_spec_id,
    input  wire [63:0]  begin_model_spec_hash,
    input  wire [5:0]   begin_layer_count,
    input  wire [7:0]   begin_hidden_blocks,
    input  wire [9:0]   begin_ffn_blocks,
    input  wire [5:0]   begin_q_heads,
    input  wire [3:0]   begin_kv_heads,
    input  wire [7:0]   begin_head_dim,
    input  wire [1:0]   begin_weight_fmt,
    input  wire [16:0]  begin_context_limit,
    input  wire [17:0]  begin_vocab_rows,
    input  wire [63:0]  begin_embed_addr,
    input  wire [63:0]  begin_lm_head_addr,
    input  wire [63:0]  begin_final_norm_addr,
    input  wire [63:0]  begin_rope_table_addr,

    input  wire         layer_wr_valid,
    output wire         layer_wr_ready,
    input  wire [5:0]   layer_wr_layer,
    input  wire [2:0]   layer_wr_word,
    input  wire [63:0]  layer_wr_data,

    input  wire         seal_valid,
    output wire         seal_ready,

    output wire         cfg_error_valid,
    input  wire         cfg_error_ready,
    output wire [7:0]   cfg_error_code,
    output wire [5:0]   cfg_error_layer,
    output wire [2:0]   cfg_error_word,

    output wire         model_spec_loading,
    output wire         model_spec_sealed,
    output wire [31:0]  model_spec_id,
    output wire [63:0]  model_spec_hash,
    output wire [5:0]   model_spec_layer_count,
    output wire [7:0]   model_spec_hidden_blocks,
    output wire [9:0]   model_spec_ffn_blocks,
    output wire [5:0]   model_spec_q_heads,
    output wire [3:0]   model_spec_kv_heads,
    output wire [7:0]   model_spec_head_dim,
    output wire [1:0]   model_spec_weight_fmt,
    output wire [16:0]  model_spec_context_limit,
    output wire [17:0]  model_spec_vocab_rows,
    output wire [63:0]  model_spec_embed_addr,
    output wire [63:0]  model_spec_lm_head_addr,
    output wire [63:0]  model_spec_final_norm_addr,
    output wire [63:0]  model_spec_rope_table_addr,

    input  wire         layer_rd_req_valid,
    output wire         layer_rd_req_ready,
    input  wire [8:0]   layer_rd_req_addr,
    output wire         layer_rd_rsp_valid,
    output wire [63:0]  layer_rd_rsp_data
);
    localparam integer TABLE_WORDS = `ENGINE_MAX_LAYERS * `MODEL_LAYER_WORDS;

    (* ram_style = "block" *) reg [63:0] layer_table [0:TABLE_WORDS-1];
    reg [7:0] layer_word_valid [0:`ENGINE_MAX_LAYERS-1];

    reg loading_q;
    reg sealed_q;
    reg [31:0] model_spec_id_q;
    reg [63:0] model_spec_hash_q;
    reg [5:0] layer_count_q;
    reg [7:0] hidden_blocks_q;
    reg [9:0] ffn_blocks_q;
    reg [5:0] q_heads_q;
    reg [3:0] kv_heads_q;
    reg [7:0] head_dim_q;
    reg [1:0] weight_fmt_q;
    reg [16:0] context_limit_q;
    reg [17:0] vocab_rows_q;
    reg [63:0] embed_addr_q;
    reg [63:0] lm_head_addr_q;
    reg [63:0] final_norm_addr_q;
    reg [63:0] rope_table_addr_q;

    reg cfg_error_valid_q;
    reg [7:0] cfg_error_code_q;
    reg [5:0] cfg_error_layer_q;
    reg [2:0] cfg_error_word_q;
    reg layer_rd_rsp_valid_q;
    reg [63:0] layer_rd_rsp_data_q;
    reg layer_wr_pending_q;
    reg [5:0] layer_wr_layer_q;
    reg [2:0] layer_wr_word_q;
    reg [63:0] layer_wr_data_q;
    reg layer_wr_bad_q;

    integer i;
    reg table_complete;
    always @* begin
        table_complete = 1'b1;
        for (i = 0; i < `ENGINE_MAX_LAYERS; i = i + 1) begin
            if ((i < layer_count_q) && (layer_word_valid[i] != 8'hff))
                table_complete = 1'b0;
        end
    end

    wire begin_model_spec_1_7b =
        (begin_model_spec_id == `MODEL_BONSAI_1_7B) &&
        (begin_layer_count == 6'd28) &&
        (begin_hidden_blocks == 8'd64) &&
        (begin_ffn_blocks == 10'd192) &&
        (begin_q_heads == 6'd16) &&
        (begin_kv_heads == 4'd8) &&
        (begin_head_dim == 8'd128) &&
        (begin_context_limit == 17'd32768) &&
        (begin_vocab_rows == `MODEL_BONSAI_VOCAB_ROWS);
    wire begin_model_spec_4b =
        (begin_model_spec_id == `MODEL_BONSAI_4B) &&
        (begin_layer_count == 6'd36) &&
        (begin_hidden_blocks == 8'd80) &&
        (begin_ffn_blocks == 10'd304) &&
        (begin_q_heads == 6'd32) &&
        (begin_kv_heads == 4'd8) &&
        (begin_head_dim == 8'd128) &&
        (begin_context_limit == 17'd32768) &&
        (begin_vocab_rows == `MODEL_BONSAI_VOCAB_ROWS);
    wire begin_model_spec_8b =
        (begin_model_spec_id == `MODEL_BONSAI_8B) &&
        (begin_layer_count == 6'd36) &&
        (begin_hidden_blocks == 8'd128) &&
        (begin_ffn_blocks == 10'd384) &&
        (begin_q_heads == 6'd32) &&
        (begin_kv_heads == 4'd8) &&
        (begin_head_dim == 8'd128) &&
        (begin_context_limit == 17'd65536) &&
        (begin_vocab_rows == `MODEL_BONSAI_VOCAB_ROWS);
    wire begin_model_spec_ok = begin_model_spec_1_7b || begin_model_spec_4b ||
                            begin_model_spec_8b;

    wire begin_shape_ok =
        (begin_model_spec_hash != 64'd0) &&
        begin_model_spec_ok &&
        ((begin_weight_fmt == 2'd1) || (begin_weight_fmt == 2'd2)) &&
        (begin_embed_addr != 64'd0) && (begin_embed_addr[5:0] == 6'd0) &&
        (begin_final_norm_addr != 64'd0) &&
        (begin_final_norm_addr[5:0] == 6'd0) &&
        (begin_lm_head_addr != 64'd0) && (begin_lm_head_addr[5:0] == 6'd0) &&
        (begin_rope_table_addr != 64'd0) &&
        (begin_rope_table_addr[5:0] == 6'd0);

    assign clear_ready = mutation_allowed && !cfg_error_valid_q &&
                         !layer_wr_pending_q;
    assign begin_ready = mutation_allowed && !loading_q && !sealed_q &&
                         !cfg_error_valid_q && !layer_wr_pending_q;
    assign layer_wr_ready = mutation_allowed && loading_q && !sealed_q &&
                            !cfg_error_valid_q && !layer_wr_pending_q;
    assign seal_ready = mutation_allowed && loading_q && !sealed_q &&
                        !cfg_error_valid_q && !layer_wr_pending_q;

    assign cfg_error_valid = cfg_error_valid_q;
    assign cfg_error_code = cfg_error_code_q;
    assign cfg_error_layer = cfg_error_layer_q;
    assign cfg_error_word = cfg_error_word_q;
    assign model_spec_loading = loading_q;
    assign model_spec_sealed = sealed_q;
    assign model_spec_id = model_spec_id_q;
    assign model_spec_hash = model_spec_hash_q;
    assign model_spec_layer_count = layer_count_q;
    assign model_spec_hidden_blocks = hidden_blocks_q;
    assign model_spec_ffn_blocks = ffn_blocks_q;
    assign model_spec_q_heads = q_heads_q;
    assign model_spec_kv_heads = kv_heads_q;
    assign model_spec_head_dim = head_dim_q;
    assign model_spec_weight_fmt = weight_fmt_q;
    assign model_spec_context_limit = context_limit_q;
    assign model_spec_vocab_rows = vocab_rows_q;
    assign model_spec_embed_addr = embed_addr_q;
    assign model_spec_lm_head_addr = lm_head_addr_q;
    assign model_spec_final_norm_addr = final_norm_addr_q;
    assign model_spec_rope_table_addr = rope_table_addr_q;

    assign layer_rd_req_ready = sealed_q;
    assign layer_rd_rsp_valid = layer_rd_rsp_valid_q;
    assign layer_rd_rsp_data = layer_rd_rsp_data_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            loading_q <= 1'b0;
            sealed_q <= 1'b0;
            cfg_error_valid_q <= 1'b0;
            cfg_error_code_q <= 8'd0;
            cfg_error_layer_q <= 6'd0;
            cfg_error_word_q <= 3'd0;
            layer_rd_rsp_valid_q <= 1'b0;
            layer_wr_pending_q <= 1'b0;
            layer_wr_layer_q <= 6'd0;
            layer_wr_word_q <= 3'd0;
            layer_wr_data_q <= 64'd0;
            layer_wr_bad_q <= 1'b0;
            for (i = 0; i < `ENGINE_MAX_LAYERS; i = i + 1)
                layer_word_valid[i] <= 8'd0;
        end else begin
            layer_rd_rsp_valid_q <= 1'b0;
            if (layer_rd_req_valid && layer_rd_req_ready) begin
                layer_rd_rsp_data_q <= layer_table[layer_rd_req_addr];
                layer_rd_rsp_valid_q <= 1'b1;
            end

            if (cfg_error_valid_q && cfg_error_ready)
                cfg_error_valid_q <= 1'b0;

            if (clear_valid && clear_ready) begin
                loading_q <= 1'b0;
                sealed_q <= 1'b0;
                for (i = 0; i < `ENGINE_MAX_LAYERS; i = i + 1)
                    layer_word_valid[i] <= 8'd0;
            end else if (begin_valid && begin_ready) begin
                if (!begin_shape_ok) begin
                    cfg_error_valid_q <= 1'b1;
                    cfg_error_code_q <= `MODEL_SPEC_ERROR_BAD_HEADER;
                    cfg_error_layer_q <= 6'd0;
                    cfg_error_word_q <= 3'd0;
                end else begin
                    model_spec_id_q <= begin_model_spec_id;
                    model_spec_hash_q <= begin_model_spec_hash;
                    layer_count_q <= begin_layer_count;
                    hidden_blocks_q <= begin_hidden_blocks;
                    ffn_blocks_q <= begin_ffn_blocks;
                    q_heads_q <= begin_q_heads;
                    kv_heads_q <= begin_kv_heads;
                    head_dim_q <= begin_head_dim;
                    weight_fmt_q <= begin_weight_fmt;
                    context_limit_q <= begin_context_limit;
                    vocab_rows_q <= begin_vocab_rows;
                    embed_addr_q <= begin_embed_addr;
                    lm_head_addr_q <= begin_lm_head_addr;
                    final_norm_addr_q <= begin_final_norm_addr;
                    rope_table_addr_q <= begin_rope_table_addr;
                    loading_q <= 1'b1;
                    for (i = 0; i < `ENGINE_MAX_LAYERS; i = i + 1)
                        layer_word_valid[i] <= 8'd0;
                end
            end else if (layer_wr_valid && layer_wr_ready) begin
                layer_wr_pending_q <= 1'b1;
                layer_wr_layer_q <= layer_wr_layer;
                layer_wr_word_q <= layer_wr_word;
                layer_wr_data_q <= layer_wr_data;
                layer_wr_bad_q <= (layer_wr_layer >= layer_count_q) ||
                                  (layer_wr_data == 64'd0) ||
                                  (layer_wr_data[5:0] != 6'd0);
            end else if (layer_wr_pending_q) begin
                layer_wr_pending_q <= 1'b0;
                if (layer_wr_bad_q) begin
                    loading_q <= 1'b0;
                    cfg_error_valid_q <= 1'b1;
                    cfg_error_code_q <= `MODEL_SPEC_ERROR_BAD_LAYER_WORD;
                    cfg_error_layer_q <= layer_wr_layer_q;
                    cfg_error_word_q <= layer_wr_word_q;
                end else begin
                    layer_table[{layer_wr_layer_q, layer_wr_word_q}] <=
                        layer_wr_data_q;
                    layer_word_valid[layer_wr_layer_q][layer_wr_word_q] <= 1'b1;
                end
            end else if (seal_valid && seal_ready) begin
                if (!table_complete) begin
                    loading_q <= 1'b0;
                    cfg_error_valid_q <= 1'b1;
                    cfg_error_code_q <= `MODEL_SPEC_ERROR_INCOMPLETE;
                    cfg_error_layer_q <= 6'd0;
                    cfg_error_word_q <= 3'd0;
                end else begin
                    loading_q <= 1'b0;
                    sealed_q <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
