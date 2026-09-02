`default_nettype none

// Transactional core model for the AXI-Lite shell regression. The full
// engine_datapath has its own end-to-end arithmetic/AXI regression; this
// model makes the shell's command and mailbox timing deterministic.
module engine_datapath #(
    parameter integer ADDR_W = 40,
    parameter integer EMIT_FULL_LOGITS = 0
) (
    input wire clk, input wire rst_n, input wire run_clear,
    output wire clear_done,
    input wire model_spec_clear_valid, output wire model_spec_clear_ready,
    input wire model_spec_begin_valid, output wire model_spec_begin_ready,
    input wire [31:0] model_spec_begin_id,
    input wire [63:0] model_spec_begin_hash,
    input wire [5:0] model_spec_begin_layer_count,
    input wire [7:0] model_spec_begin_hidden_blocks,
    input wire [9:0] model_spec_begin_ffn_blocks,
    input wire [5:0] model_spec_begin_q_heads,
    input wire [3:0] model_spec_begin_kv_heads,
    input wire [7:0] model_spec_begin_head_dim,
    input wire [1:0] model_spec_begin_weight_fmt,
    input wire [16:0] model_spec_begin_context_limit,
    input wire [17:0] model_spec_begin_vocab_rows,
    input wire [63:0] model_spec_begin_embed_addr,
    input wire [63:0] model_spec_begin_lm_head_addr,
    input wire [63:0] model_spec_begin_final_norm_addr,
    input wire [63:0] model_spec_begin_rope_table_addr,
    input wire model_spec_layer_wr_valid, output wire model_spec_layer_wr_ready,
    input wire [5:0] model_spec_layer_wr_layer,
    input wire [2:0] model_spec_layer_wr_word,
    input wire [63:0] model_spec_layer_wr_data,
    input wire model_spec_seal_valid, output wire model_spec_seal_ready,
    output reg model_spec_cfg_error_valid,
    input wire model_spec_cfg_error_ready,
    output reg [7:0] model_spec_cfg_error_code,
    output reg [5:0] model_spec_cfg_error_layer,
    output reg [2:0] model_spec_cfg_error_word,
    output reg model_spec_loading, output reg model_spec_sealed,
    output wire [31:0] interface_version,
    output wire [63:0] layer_layout_hash,
    output reg [31:0] active_model_spec_id,
    output reg [63:0] active_model_spec_hash,
    input wire cmd_valid, output wire cmd_ready,
    input wire [31:0] cmd_tag, input wire [31:0] cmd_model_spec_id,
    input wire [63:0] cmd_model_spec_hash,
    input wire [3:0] cmd_token_count,
    input wire [7:0] cmd_lane_mask,
    input wire [255:0] cmd_token_ids,
    input wire [16:0] cmd_position_base,
    input wire [63:0] cmd_kv_base,
    input wire [16:0] cmd_kv_capacity,
    input wire cmd_emit_logits,
    output reg commit_valid, input wire commit_ready,
    output reg [31:0] commit_tag,
    output reg [31:0] commit_model_spec_id,
    output reg [63:0] commit_model_spec_hash,
    output reg [3:0] commit_token_count,
    output reg [16:0] commit_kv_length,
    output reg commit_logits_valid,
    output reg error_valid, input wire error_ready,
    output reg [31:0] error_tag, output reg [15:0] error_code,
    output reg [7:0] error_detail, output reg [5:0] error_layer,
    output reg [4:0] error_stage,
    output wire logits_valid, input wire logits_ready,
    output wire [17:0] logits_row, output wire [31:0] logits_data,
    output wire logits_last,
    output reg result_valid, input wire result_ready,
    output reg [17:0] result_token, output reg [31:0] result_logit,
    output reg result_error, output reg [7:0] result_status,
    output reg busy, output reg [5:0] debug_layer,
    output reg [4:0] debug_stage, output wire trace_valid,
    output wire [5:0] trace_layer, output wire [4:0] trace_stage,
    output wire protocol_error,
    output wire metrics_stage_active,
    output wire [4:0] metrics_stage,
    output wire [12:0] metrics_projection_probe,
    output wire [2:0] metrics_weight_axi_r_beats,
    output wire [2:0] metrics_weight_axi_r_gap_ports,
    output wire metrics_weight_zip_skew,
    output wire [1:0] metrics_history_axi_r_beats,
    output wire metrics_kv_axi_w_beat,
    output wire [4*ADDR_W-1:0] weight_axi_araddr,
    output wire [31:0] weight_axi_arlen,
    output wire [11:0] weight_axi_arsize,
    output wire [7:0] weight_axi_arburst,
    output wire [3:0] weight_axi_arvalid,
    input wire [3:0] weight_axi_arready,
    input wire [511:0] weight_axi_rdata,
    input wire [7:0] weight_axi_rresp,
    input wire [3:0] weight_axi_rlast,
    input wire [3:0] weight_axi_rvalid,
    output wire [3:0] weight_axi_rready,
    output wire [ADDR_W-1:0] hist_k_axi_araddr,
    output wire [7:0] hist_k_axi_arlen,
    output wire [2:0] hist_k_axi_arsize,
    output wire [1:0] hist_k_axi_arburst,
    output wire hist_k_axi_arvalid, input wire hist_k_axi_arready,
    input wire [127:0] hist_k_axi_rdata,
    input wire [1:0] hist_k_axi_rresp,
    input wire hist_k_axi_rlast, input wire hist_k_axi_rvalid,
    output wire hist_k_axi_rready,
    output wire [ADDR_W-1:0] hist_v_axi_araddr,
    output wire [7:0] hist_v_axi_arlen,
    output wire [2:0] hist_v_axi_arsize,
    output wire [1:0] hist_v_axi_arburst,
    output wire hist_v_axi_arvalid, input wire hist_v_axi_arready,
    input wire [127:0] hist_v_axi_rdata,
    input wire [1:0] hist_v_axi_rresp,
    input wire hist_v_axi_rlast, input wire hist_v_axi_rvalid,
    output wire hist_v_axi_rready,
    output wire [ADDR_W-1:0] kv_axi_awaddr,
    output wire [7:0] kv_axi_awlen, output wire [2:0] kv_axi_awsize,
    output wire [1:0] kv_axi_awburst, output wire kv_axi_awvalid,
    input wire kv_axi_awready, output wire [127:0] kv_axi_wdata,
    output wire [15:0] kv_axi_wstrb, output wire kv_axi_wlast,
    output wire kv_axi_wvalid, input wire kv_axi_wready,
    input wire [1:0] kv_axi_bresp, input wire kv_axi_bvalid,
    output wire kv_axi_bready
);
    reg [2:0] clear_cycles_q;
    reg [3:0] exec_cycles_q;
    reg exec_error_q;
    reg result_sent_q;
    reg [31:0] accepted_tag_q;
    reg [31:0] accepted_model_spec_id_q;
    reg [63:0] accepted_model_spec_hash_q;
    reg [3:0] accepted_token_count_q;
    reg [16:0] accepted_position_q;
    reg [16:0] accepted_kv_capacity_q;
    reg accepted_emit_logits_q;

    assign interface_version = 32'h0001_0007;
    assign layer_layout_hash = 64'hc255_c7a5_2fc1_4a79;
    assign clear_done = !run_clear || (clear_cycles_q == 3'd4);
    assign model_spec_clear_ready = !busy && !model_spec_cfg_error_valid;
    assign model_spec_begin_ready = !busy && !model_spec_loading &&
                                 !model_spec_sealed && !model_spec_cfg_error_valid;
    assign model_spec_layer_wr_ready = !busy && model_spec_loading &&
                                    !model_spec_sealed &&
                                    !model_spec_cfg_error_valid;
    assign model_spec_seal_ready = model_spec_layer_wr_ready;
    assign cmd_ready = model_spec_sealed && !busy && !run_clear;
    assign logits_valid = 1'b0;
    assign logits_row = 18'd0;
    assign logits_data = 32'd0;
    assign logits_last = 1'b0;
    assign trace_valid = 1'b0;
    assign trace_layer = 6'd0;
    assign trace_stage = 5'd0;
    assign protocol_error = 1'b0;
    assign metrics_stage_active = busy;
    assign metrics_stage = debug_stage;
    assign metrics_projection_probe = 13'd0;
    assign metrics_weight_axi_r_beats = 3'd0;
    assign metrics_weight_axi_r_gap_ports = 3'd0;
    assign metrics_weight_zip_skew = 1'b0;
    assign metrics_history_axi_r_beats = 2'd0;
    assign metrics_kv_axi_w_beat = 1'b0;

    assign weight_axi_araddr = {4*ADDR_W{1'b0}};
    assign weight_axi_arlen = 32'd0;
    assign weight_axi_arsize = {4{3'd4}};
    assign weight_axi_arburst = {4{2'b01}};
    assign weight_axi_arvalid = 4'd0;
    assign weight_axi_rready = 4'd0;
    assign hist_k_axi_araddr = {ADDR_W{1'b0}};
    assign hist_k_axi_arlen = 8'd0;
    assign hist_k_axi_arsize = 3'd4;
    assign hist_k_axi_arburst = 2'b01;
    assign hist_k_axi_arvalid = 1'b0;
    assign hist_k_axi_rready = 1'b0;
    assign hist_v_axi_araddr = {ADDR_W{1'b0}};
    assign hist_v_axi_arlen = 8'd0;
    assign hist_v_axi_arsize = 3'd4;
    assign hist_v_axi_arburst = 2'b01;
    assign hist_v_axi_arvalid = 1'b0;
    assign hist_v_axi_rready = 1'b0;
    assign kv_axi_awaddr = {ADDR_W{1'b0}};
    assign kv_axi_awlen = 8'd0;
    assign kv_axi_awsize = 3'd4;
    assign kv_axi_awburst = 2'b01;
    assign kv_axi_awvalid = 1'b0;
    assign kv_axi_wdata = 128'd0;
    assign kv_axi_wstrb = 16'hffff;
    assign kv_axi_wlast = 1'b0;
    assign kv_axi_wvalid = 1'b0;
    assign kv_axi_bready = 1'b0;

    always @(posedge clk) begin
        if (!rst_n) begin
            clear_cycles_q <= 3'd0;
            exec_cycles_q <= 4'd0;
            exec_error_q <= 1'b0;
            result_sent_q <= 1'b0;
            accepted_tag_q <= 32'd0;
            accepted_model_spec_id_q <= 32'd0;
            accepted_model_spec_hash_q <= 64'd0;
            accepted_token_count_q <= 4'd0;
            accepted_position_q <= 17'd0;
            accepted_kv_capacity_q <= 17'd0;
            accepted_emit_logits_q <= 1'b0;
            model_spec_cfg_error_valid <= 1'b0;
            model_spec_cfg_error_code <= 8'd0;
            model_spec_cfg_error_layer <= 6'd0;
            model_spec_cfg_error_word <= 3'd0;
            model_spec_loading <= 1'b0;
            model_spec_sealed <= 1'b0;
            active_model_spec_id <= 32'd0;
            active_model_spec_hash <= 64'd0;
            commit_valid <= 1'b0;
            commit_tag <= 32'd0;
            commit_model_spec_id <= 32'd0;
            commit_model_spec_hash <= 64'd0;
            commit_token_count <= 4'd0;
            commit_kv_length <= 17'd0;
            commit_logits_valid <= 1'b0;
            error_valid <= 1'b0;
            error_tag <= 32'd0;
            error_code <= 16'd0;
            error_detail <= 8'd0;
            error_layer <= 6'd0;
            error_stage <= 5'd0;
            result_valid <= 1'b0;
            result_token <= 18'd0;
            result_logit <= 32'd0;
            result_error <= 1'b0;
            result_status <= 8'd0;
            busy <= 1'b0;
            debug_layer <= 6'd0;
            debug_stage <= 5'd0;
        end else begin
            if (run_clear) begin
                if (clear_cycles_q != 3'd4)
                    clear_cycles_q <= clear_cycles_q + 1'b1;
                busy <= 1'b0;
                exec_cycles_q <= 4'd0;
                result_valid <= 1'b0;
                commit_valid <= 1'b0;
                error_valid <= 1'b0;
            end else begin
                clear_cycles_q <= 3'd0;

                if (model_spec_cfg_error_valid && model_spec_cfg_error_ready)
                    model_spec_cfg_error_valid <= 1'b0;
                if (model_spec_clear_valid && model_spec_clear_ready) begin
                    model_spec_loading <= 1'b0;
                    model_spec_sealed <= 1'b0;
                end
                if (model_spec_begin_valid && model_spec_begin_ready) begin
                    model_spec_loading <= 1'b1;
                    active_model_spec_id <= model_spec_begin_id;
                    active_model_spec_hash <= model_spec_begin_hash;
                end
                if (model_spec_layer_wr_valid && model_spec_layer_wr_ready &&
                    (model_spec_layer_wr_data == 64'h0000_00ab_cdef_0000)) begin
                    model_spec_cfg_error_valid <= 1'b1;
                    model_spec_cfg_error_code <= 8'ha5;
                    model_spec_cfg_error_layer <= model_spec_layer_wr_layer;
                    model_spec_cfg_error_word <= model_spec_layer_wr_word;
                end
                if (model_spec_seal_valid && model_spec_seal_ready) begin
                    model_spec_loading <= 1'b0;
                    model_spec_sealed <= 1'b1;
                end

                if (cmd_valid && cmd_ready) begin
                    busy <= 1'b1;
                    exec_cycles_q <= 4'd12;
                    exec_error_q <= cmd_tag[31];
                    result_sent_q <= 1'b0;
                    accepted_tag_q <= cmd_tag;
                    accepted_model_spec_id_q <= cmd_model_spec_id;
                    accepted_model_spec_hash_q <= cmd_model_spec_hash;
                    accepted_token_count_q <= cmd_token_count;
                    accepted_position_q <= cmd_position_base;
                    accepted_kv_capacity_q <= cmd_kv_capacity;
                    accepted_emit_logits_q <= cmd_emit_logits;
                    debug_layer <= 6'd7;
                    debug_stage <= 5'd11;
                end else if (busy && (exec_cycles_q != 4'd0)) begin
                    exec_cycles_q <= exec_cycles_q - 1'b1;
                end else if (busy && exec_error_q && !error_valid) begin
                    error_valid <= 1'b1;
                    error_tag <= accepted_tag_q;
                    error_code <= 16'h1234;
                    error_detail <= 8'h56;
                    error_layer <= 6'd7;
                    error_stage <= 5'd11;
                end else if (busy && !exec_error_q && !result_sent_q &&
                             !result_valid) begin
                    result_valid <= 1'b1;
                    result_token <= 18'h2aaaa;
                    result_logit <= 32'h3f80_0000;
                    result_error <= 1'b0;
                    result_status <= 8'h5a;
                end else if (busy && !exec_error_q && result_sent_q &&
                             !commit_valid) begin
                    commit_valid <= 1'b1;
                    commit_tag <= accepted_tag_q;
                    commit_model_spec_id <= accepted_model_spec_id_q;
                    commit_model_spec_hash <= accepted_model_spec_hash_q;
                    commit_token_count <= accepted_token_count_q;
                    commit_kv_length <= accepted_position_q +
                                        accepted_token_count_q;
                    commit_logits_valid <= accepted_emit_logits_q;
                end

                if (result_valid && result_ready) begin
                    result_valid <= 1'b0;
                    result_sent_q <= 1'b1;
                end
                if (commit_valid && commit_ready) begin
                    commit_valid <= 1'b0;
                    busy <= 1'b0;
                end
                if (error_valid && error_ready) begin
                    error_valid <= 1'b0;
                    busy <= 1'b0;
                end
            end
        end
    end

    wire unused_inputs = &{1'b0, model_spec_begin_layer_count,
        model_spec_begin_hidden_blocks, model_spec_begin_ffn_blocks,
        model_spec_begin_q_heads, model_spec_begin_kv_heads,
        model_spec_begin_head_dim, model_spec_begin_weight_fmt,
        model_spec_begin_context_limit, model_spec_begin_vocab_rows,
        model_spec_begin_embed_addr, model_spec_begin_lm_head_addr,
        model_spec_begin_final_norm_addr, model_spec_begin_rope_table_addr,
        cmd_lane_mask, cmd_token_ids, cmd_kv_base, cmd_kv_capacity,
        accepted_kv_capacity_q, logits_ready,
        weight_axi_arready, weight_axi_rdata, weight_axi_rresp,
        weight_axi_rlast, weight_axi_rvalid, hist_k_axi_arready,
        hist_k_axi_rdata, hist_k_axi_rresp, hist_k_axi_rlast,
        hist_k_axi_rvalid, hist_v_axi_arready, hist_v_axi_rdata,
        hist_v_axi_rresp, hist_v_axi_rlast, hist_v_axi_rvalid,
        kv_axi_awready, kv_axi_wready, kv_axi_bresp, kv_axi_bvalid,
        EMIT_FULL_LOGITS[0]};
endmodule

`default_nettype wire
