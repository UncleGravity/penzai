// Command-owned low-bit projection transaction.
//
// Shape fields are derived from the immutable model dimensions and operation;
// callers cannot provide independent K/M/rowblock/reader counts. Semantic
// projections are streamed to the shared semantic sink. LM-head records stay
// local and are reduced by the exact FP32 logits sink.
//
// Interface v2 canonical signature SHA-256:
// 9d237020507bd649c1fe9e62b99dde2c9ca61bbddde8cdc230e2f29a57929a61
// projection-service-v2|cmd:op,count,prefix-mask,addr0..3,hb,fb,qh,
// kvh,hd,pos17,eps,vocab18,wfmt,full|shape:op-derived-k,m,rowblocks,
// q1-5,q2-9|reader:quad4|acts:q8-four-lane-wave,input0-127,down128-511|
// sink:cfg-arm-f32-done,status16|lm:full-logits+accepted-result,
// mask=count-1|abort:local-drain-q8+reader,global-clear-drop

`default_nettype none

`include "engine_defs.vh"

module projection_service (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,

    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [2:0]    cmd_op,
    input  wire [3:0]    cmd_token_count,
    input  wire [7:0]    cmd_token_mask,
    input  wire [63:0]   cmd_addr0,
    input  wire [63:0]   cmd_addr1,
    input  wire [63:0]   cmd_addr2,
    input  wire [63:0]   cmd_addr3,
    input  wire [7:0]    cmd_hidden_blocks,
    input  wire [9:0]    cmd_ffn_blocks,
    input  wire [5:0]    cmd_q_heads,
    input  wire [3:0]    cmd_kv_heads,
    input  wire [7:0]    cmd_head_dim,
    input  wire [16:0]   cmd_position_base,
    input  wire [31:0]   cmd_epsilon,
    input  wire [17:0]   cmd_vocab_rows,
    input  wire [1:0]    cmd_weight_fmt,
    input  wire          cmd_emit_full_logits,

    // The projection client of the shared four-port packed-data reader.
    output wire          read_cmd_valid,
    input  wire          read_cmd_ready,
    output wire [63:0]   read_cmd_base_addr,
    output wire [31:0]   read_cmd_port_beats,
    output wire [3:0]    read_cmd_port_mask,
    output wire          read_abort,
    input  wire [511:0]  read_data,
    input  wire          read_valid,
    output wire          read_ready,
    input  wire          read_last,
    input  wire          read_error,
    input  wire          read_busy,
    input  wire          read_done_valid,
    output wire          read_done_ready,
    input  wire          read_done_error,
    input  wire [7:0]    read_done_status,

    // Resident Q8 arena activation read port.
    output wire          q8_rd_req_valid,
    input  wire          q8_rd_req_ready,
    output wire          q8_rd_req_wave,
    output wire [8:0]    q8_rd_req_addr,
    input  wire          q8_rd_rsp_valid,
    output wire          q8_rd_rsp_ready,
    input  wire [1087:0] q8_rd_rsp_data,

    // Existing semantic sink boundary. QKV is configured and fully armed
    // before this service acquires the long packed-weight reader transaction.
    output wire          sink_cfg_valid,
    input  wire          sink_cfg_ready,
    output wire [1:0]    sink_cfg_mode,
    output wire [7:0]    sink_cfg_token_mask,
    output wire [12:0]   sink_cfg_hidden_dim,
    output wire [14:0]   sink_cfg_ffn_dim,
    output wire [5:0]    sink_cfg_q_heads,
    output wire [3:0]    sink_cfg_kv_heads,
    output wire [16:0]   sink_cfg_position_base,
    output wire [31:0]   sink_cfg_epsilon,
    output wire [63:0]   sink_cfg_q_gamma_addr,
    output wire [63:0]   sink_cfg_k_gamma_addr,
    output wire [63:0]   sink_cfg_rope_addr,
    output wire          sink_abort_run,
    input  wire          sink_projection_armed,
    output wire          sink_proj_valid,
    input  wire          sink_proj_ready,
    output wire [2:0]    sink_proj_token,
    output wire [17:0]   sink_proj_row,
    output wire [31:0]   sink_proj_data_f32,
    output wire          sink_proj_last,
    input  wire          sink_done_valid,
    output wire          sink_done_ready,
    input  wire          sink_done_error,
    input  wire [15:0]   sink_done_status,

    // Optional full-logit publication and the always-present greedy result.
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
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [7:0]    done_status,
    output wire [15:0]   derived_k,
    output wire [17:0]   derived_m,
    output wire [15:0]   derived_rowblocks,
    output wire [31:0]   accepted_weight_beats,
    output wire [31:0]   activation_wave_issues,
    output reg  [12:0]   metrics_probe
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_PREP  = 3'd1;
    localparam [2:0] ST_ISSUE = 3'd2;
    localparam [2:0] ST_RUN   = 3'd3;
    localparam [2:0] ST_FAIL  = 3'd4;
    localparam [2:0] ST_ABORT = 3'd5;
    localparam [2:0] ST_DONE  = 3'd6;
    localparam [2:0] ST_CONFIG = 3'd7;

    localparam [7:0] STATUS_OK         = 8'h00;
    localparam [7:0] STATUS_BAD_CMD    = 8'h01;
    localparam [7:0] STATUS_READER     = 8'h10;
    localparam [7:0] STATUS_ENGINE     = 8'h20;
    localparam [7:0] STATUS_SINK       = 8'h30;
    localparam [7:0] STATUS_LOGITS     = 8'h40;
    localparam [7:0] STATUS_WEIGHT_LEN = 8'h50;

    localparam signed [7:0] FIXED_EMIN = -8'sd48;
    localparam [1:0] WEIGHT_Q1 = 2'd1;
    localparam [1:0] WEIGHT_Q2 = 2'd2;
    localparam [9:0] Q8_INPUT_BLOCK_LIMIT = 10'd128;
    localparam [9:0] Q8_FFN_BASE = 10'd128;
    localparam [9:0] Q8_FFN_BLOCK_LIMIT = 10'd384;
    localparam [9:0] Q8_ARENA_BLOCKS = 10'd512;

    reg [2:0] state_q;
    reg [2:0] op_q;
    reg [7:0] token_mask_q;
    reg [63:0] weight_addr_q;
    reg [15:0] k_q;
    reg [17:0] m_q;
    reg [15:0] rowblocks_q;
    reg [15:0] k_blocks_q;
    (* use_dsp = "yes" *) reg [31:0] rowblock_k_product_q;
    reg [31:0] port_beats_q;
    reg [1:0] weight_fmt_q;
    reg semantic_q;
    reg child_configured_q;
    reg [12:0] cfg_hidden_dim_q;
    reg [14:0] cfg_ffn_dim_q;
    reg [5:0] cfg_q_heads_q;
    reg [3:0] cfg_kv_heads_q;
    reg [16:0] cfg_position_base_q;
    reg [31:0] cfg_epsilon_q;
    reg [63:0] cfg_q_gamma_addr_q;
    reg [63:0] cfg_k_gamma_addr_q;
    reg [63:0] cfg_rope_addr_q;
    reg [17:0] logits_vocab_rows_q;
    reg logits_emit_full_q;
    reg engine_done_seen_q;
    reg reader_done_seen_q;
    reg child_done_seen_q;
    reg [7:0] done_status_q;
    reg local_abort_q;
    reg [2:0] q8_outstanding_q;
    reg [31:0] weight_stream_count_q;

    function automatic [7:0] prefix_mask8(input [3:0] count);
        begin
            case (count)
                4'd1: prefix_mask8 = 8'h01;
                4'd2: prefix_mask8 = 8'h03;
                4'd3: prefix_mask8 = 8'h07;
                4'd4: prefix_mask8 = 8'h0f;
                4'd5: prefix_mask8 = 8'h1f;
                4'd6: prefix_mask8 = 8'h3f;
                4'd7: prefix_mask8 = 8'h7f;
                4'd8: prefix_mask8 = 8'hff;
                default: prefix_mask8 = 8'h00;
            endcase
        end
    endfunction

    function automatic [7:0] final_mask8(input [3:0] count);
        begin
            case (count)
                4'd1: final_mask8 = 8'h01;
                4'd2: final_mask8 = 8'h02;
                4'd3: final_mask8 = 8'h04;
                4'd4: final_mask8 = 8'h08;
                4'd5: final_mask8 = 8'h10;
                4'd6: final_mask8 = 8'h20;
                4'd7: final_mask8 = 8'h40;
                4'd8: final_mask8 = 8'h80;
                default: final_mask8 = 8'h00;
            endcase
        end
    endfunction

    function automatic [7:0] sink_status_code(input [15:0] status);
        begin
            sink_status_code = 8'h3f;
            if (status[0]) sink_status_code = 8'h31;
            if (status[1]) sink_status_code = 8'h32;
            if (status[2]) sink_status_code = 8'h33;
            if (status[3]) sink_status_code = 8'h34;
            if (status[4]) sink_status_code = 8'h35;
            if (status[5]) sink_status_code = 8'h36;
            if (status[6]) sink_status_code = 8'h37;
            if (status[7]) sink_status_code = 8'h38;
            if (status[8]) sink_status_code = 8'h39;
            if (status[9]) sink_status_code = 8'h3a;
        end
    endfunction

    wire op_qkv = cmd_op == `PROJECTION_OP_QKV;
    wire op_o = cmd_op == `PROJECTION_OP_O;
    wire op_gate_up = cmd_op == `PROJECTION_OP_GATE_UP;
    wire op_down = cmd_op == `PROJECTION_OP_DOWN;
    wire op_lm = cmd_op == `PROJECTION_OP_LM_HEAD;
    wire op_valid = op_qkv || op_o || op_gate_up || op_down || op_lm;
    wire op_semantic = !op_lm;

    wire [12:0] cmd_hidden_dim = {cmd_hidden_blocks, 5'b0};
    wire [14:0] cmd_ffn_dim = {cmd_ffn_blocks, 5'b0};
    wire [15:0] attention_dim = {10'd0, cmd_q_heads} << 7;
    wire [15:0] shape_k = op_o ? attention_dim :
                          op_down ? {1'b0, cmd_ffn_dim} :
                                    {3'd0, cmd_hidden_dim};
    wire [17:0] qkv_rows = ({12'd0, cmd_q_heads} +
                            ({14'd0, cmd_kv_heads} << 1)) << 7;
    wire [17:0] shape_m = op_qkv ? qkv_rows :
                          op_gate_up ? ({3'd0, cmd_ffn_dim} << 1) :
                          op_lm ? cmd_vocab_rows :
                                  {5'd0, cmd_hidden_dim};
    wire [15:0] shape_rowblocks =
        {2'd0, shape_m[17:4]} + {15'd0, |shape_m[3:0]};
    wire [15:0] shape_k_blocks = shape_k >> 7;

    wire hidden_ok = (cmd_hidden_blocks == 8'd64) ||
                     (cmd_hidden_blocks == 8'd80) ||
                     (cmd_hidden_blocks == 8'd128);
    wire ffn_ok = (cmd_ffn_blocks == 10'd192) ||
                  (cmd_ffn_blocks == 10'd304) ||
                  (cmd_ffn_blocks == 10'd384);
    wire heads_ok = ((cmd_q_heads == 6'd16) ||
                     (cmd_q_heads == 6'd32)) &&
                    (cmd_kv_heads == 4'd8) &&
                    (cmd_head_dim == 8'd128);
    wire token_ok = (cmd_token_count >= 4'd1) &&
                    (cmd_token_count <= 4'd8) &&
                    (cmd_token_mask == (op_lm ?
                     final_mask8(cmd_token_count) :
                     prefix_mask8(cmd_token_count)));
    wire address_ok = (cmd_addr0 != 64'd0) &&
                      (cmd_addr0[5:0] == 6'd0);
    wire aux_ok = !op_qkv ||
                  ((cmd_addr1 != 64'd0) && (cmd_addr1[5:0] == 6'd0) &&
                   (cmd_addr2 != 64'd0) && (cmd_addr2[5:0] == 6'd0) &&
                   (cmd_addr3 != 64'd0) && (cmd_addr3[5:0] == 6'd0) &&
                   !cmd_epsilon[31] &&
                   (cmd_epsilon[30:23] != 8'd0) &&
                   (cmd_epsilon[30:23] != 8'hff) &&
                   (({1'b0, cmd_position_base} +
                     {14'd0, cmd_token_count}) <= 18'd65536));
    wire vocab_ok = !op_lm || ((cmd_vocab_rows != 18'd0) &&
                                      (cmd_vocab_rows <= 18'd160000));
    wire format_ok = (cmd_weight_fmt == WEIGHT_Q1) ||
                     (cmd_weight_fmt == WEIGHT_Q2);
    wire partition_ok = ({2'd0, cmd_hidden_blocks} <=
                         Q8_INPUT_BLOCK_LIMIT) &&
                        (cmd_ffn_blocks <= Q8_FFN_BLOCK_LIMIT);
    wire command_ok = op_valid && hidden_ok && ffn_ok && heads_ok &&
                      token_ok && address_ok && aux_ok && vocab_ok &&
                      format_ok && partition_ok && (shape_k != 16'd0) &&
                      (shape_k[6:0] == 7'd0) &&
                      (shape_m != 18'd0);

    // Admission owns and captures one complete command locally. Child setup is
    // a later registered transaction, so model_spec validation cannot reach child
    // state controls or the projection sizing DSP enables combinationally.
    wire logits_start_ready;
    wire logits_start_valid = rst_n && !clear && !abort_run &&
                              (state_q == ST_CONFIG) && !semantic_q &&
                              !child_configured_q;
    assign sink_cfg_valid = rst_n && !clear && !abort_run &&
                            (state_q == ST_CONFIG) && semantic_q &&
                            !child_configured_q;
    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    wire command_fire = cmd_valid && cmd_ready;
    wire sink_cfg_fire = sink_cfg_valid && sink_cfg_ready;
    wire logits_start_fire = logits_start_valid && logits_start_ready;

    assign sink_cfg_mode = op_q == `PROJECTION_OP_QKV ? 2'd0 :
                           op_q == `PROJECTION_OP_GATE_UP ? 2'd1 :
                           op_q == `PROJECTION_OP_O ? 2'd2 : 2'd3;
    assign sink_cfg_token_mask = token_mask_q;
    assign sink_cfg_hidden_dim = cfg_hidden_dim_q;
    assign sink_cfg_ffn_dim = cfg_ffn_dim_q;
    assign sink_cfg_q_heads = cfg_q_heads_q;
    assign sink_cfg_kv_heads = cfg_kv_heads_q;
    assign sink_cfg_position_base = cfg_position_base_q;
    assign sink_cfg_epsilon = cfg_epsilon_q;
    assign sink_cfg_q_gamma_addr = cfg_q_gamma_addr_q;
    assign sink_cfg_k_gamma_addr = cfg_k_gamma_addr_q;
    assign sink_cfg_rope_addr = cfg_rope_addr_q;

    wire child_clear = clear || abort_run || local_abort_q;
    assign sink_abort_run = child_clear;

    assign read_cmd_valid = (state_q == ST_ISSUE) && !abort_run;
    assign read_cmd_base_addr = weight_addr_q;
    assign read_cmd_port_beats = port_beats_q;
    assign read_cmd_port_mask = 4'hf;
    assign read_abort = clear || abort_run || local_abort_q ||
                        (state_q == ST_FAIL) || (state_q == ST_ABORT);
    wire read_cmd_fire = read_cmd_valid && read_cmd_ready;
    assign read_done_ready = (state_q == ST_RUN) ||
                             (state_q == ST_FAIL) ||
                             (state_q == ST_ABORT);
    wire read_done_fire = read_done_valid && read_done_ready;

    wire engine_busy;
    wire engine_done;
    wire engine_error;
    wire engine_w_ready;
    wire engine_act_req_valid;
    wire [8:0] engine_act_req_addr;
    wire engine_act_req_wave;
    wire engine_act_rsp_ready;
    wire signed [103:0] engine_out_acc;
    wire signed [7:0] engine_out_emin;
    wire [2:0] engine_out_token;
    wire [17:0] engine_out_row;
    wire engine_out_last;
    wire engine_out_valid;
    wire engine_out_ready;
    wire [31:0] engine_weight_beats;
    wire [31:0] engine_wave_issues;
    wire engine_metrics_selector_full;
    wire [2:0] engine_metrics_selector_level;
    wire engine_metrics_drain;
    wire engine_metrics_bank_wait;

    assign read_ready = (state_q == ST_RUN) && !child_clear &&
                        engine_w_ready;
    wire weight_fire = read_valid && read_ready;

    assign q8_rd_req_valid = (state_q == ST_RUN) && !child_clear &&
                             engine_act_req_valid;
    wire [9:0] q8_rd_req_addr_ext = {1'b0, engine_act_req_addr} +
                                    (op_q == `PROJECTION_OP_DOWN ?
                                     Q8_FFN_BASE : 10'd0);
    assign q8_rd_req_addr = q8_rd_req_addr_ext[8:0];
    assign q8_rd_req_wave = engine_act_req_wave;
    assign q8_rd_rsp_ready = (state_q == ST_RUN) && !child_clear ?
                             engine_act_rsp_ready :
                             ((state_q == ST_FAIL) ||
                              (state_q == ST_ABORT) || clear || abort_run);
    wire q8_req_fire = q8_rd_req_valid && q8_rd_req_ready;
    wire q8_rsp_fire = q8_rd_rsp_valid && q8_rd_rsp_ready;

     projection_engine u_engine (
        .clk(clk), .rst_n(rst_n), .clear(child_clear),
        .start(read_cmd_fire), .model_spec_k(k_q), .model_spec_m(m_q),
        .model_spec_rowblocks(rowblocks_q), .weight_fmt(weight_fmt_q),
        .emin(FIXED_EMIN), .token_mask(token_mask_q),
        .busy(engine_busy), .done(engine_done), .error(engine_error),
        .w_data(read_data), .w_valid(read_valid && (state_q == ST_RUN)),
        .w_ready(engine_w_ready),
        .act_req_valid(engine_act_req_valid),
        .act_req_ready(q8_rd_req_ready && (state_q == ST_RUN) &&
                       !child_clear),
        .act_req_addr(engine_act_req_addr),
        .act_req_wave(engine_act_req_wave),
        .act_rsp_valid(q8_rd_rsp_valid && (state_q == ST_RUN) &&
                       !child_clear),
        .act_rsp_ready(engine_act_rsp_ready),
        .act_rsp_data(q8_rd_rsp_data),
        .out_acc(engine_out_acc), .out_emin(engine_out_emin),
        .out_token(engine_out_token), .out_row(engine_out_row),
        .out_last(engine_out_last), .out_valid(engine_out_valid),
        .out_ready(engine_out_ready),
        .weight_beat_count(engine_weight_beats),
        .wave_issue_count(engine_wave_issues),
        .metrics_selector_full(engine_metrics_selector_full),
        .metrics_selector_level(engine_metrics_selector_level),
        .metrics_drain(engine_metrics_drain),
        .metrics_bank_wait(engine_metrics_bank_wait)
    );

    wire emit_out_valid;
    wire emit_out_ready;
    wire [31:0] emit_out_data;
    wire [2:0] emit_out_token;
    wire [17:0] emit_out_row;
    wire emit_out_last;
    wire [3:0] emit_reserved;

     emit_stream #(.DEPTH(8)) u_emit (
        .clk(clk), .rst_n(rst_n), .clear(child_clear),
        .in_valid(engine_out_valid), .in_ready(engine_out_ready),
        .in_acc(engine_out_acc), .in_emin(engine_out_emin),
        .in_token(engine_out_token), .in_row(engine_out_row),
        .in_last(engine_out_last), .out_valid(emit_out_valid),
        .out_ready(emit_out_ready), .out_data(emit_out_data),
        .out_token(emit_out_token), .out_row(emit_out_row),
        .out_last(emit_out_last), .reserved(emit_reserved)
    );

    wire logits_in_ready;
    wire logits_busy;
    wire logits_result_valid;
    wire logits_result_error;
    wire [7:0] logits_result_status;
    wire [31:0] logits_accepted;
    wire route_live = (state_q == ST_RUN) && !child_clear;

    assign sink_proj_valid = route_live && semantic_q && emit_out_valid;
    assign sink_proj_token = emit_out_token;
    assign sink_proj_row = emit_out_row;
    assign sink_proj_data_f32 = emit_out_data;
    assign sink_proj_last = emit_out_last;
    assign emit_out_ready = route_live &&
                            (semantic_q ? sink_proj_ready : logits_in_ready);
    assign sink_done_ready = semantic_q &&
                             (((state_q == ST_CONFIG) &&
                               child_configured_q) ||
                              (state_q == ST_RUN));

     logits_sink u_logits (
        .clk(clk), .rst_n(rst_n), .clear(child_clear),
        .start_valid(logits_start_valid),
        .start_ready(logits_start_ready), .vocab_rows(logits_vocab_rows_q),
        .emit_full_logits(logits_emit_full_q),
        .in_valid(route_live && !semantic_q && emit_out_valid),
        .in_ready(logits_in_ready), .in_row(emit_out_row),
        .in_data(emit_out_data), .in_last(emit_out_last),
        .logits_valid(logits_valid), .logits_ready(logits_ready),
        .logits_row(logits_row), .logits_data(logits_data),
        .logits_last(logits_last), .result_valid(logits_result_valid),
        .result_ready(result_ready), .result_token(result_token),
        .result_logit(result_logit), .result_error(logits_result_error),
        .result_status(logits_result_status),
        .accepted_logits(logits_accepted), .busy(logits_busy)
    );
    assign result_valid = logits_result_valid;
    assign result_error = logits_result_error;
    assign result_status = logits_result_status;

    wire engine_done_after = engine_done_seen_q || engine_done;
    wire reader_done_after = reader_done_seen_q || read_done_fire;
    wire semantic_done_fire = sink_done_valid && sink_done_ready;
    wire logits_result_fire = logits_result_valid && result_ready;
    wire child_done_after = child_done_seen_q ||
                            (semantic_q ? semantic_done_fire :
                                          logits_result_fire);
    wire all_complete = engine_done_after && reader_done_after &&
                        child_done_after && !engine_busy &&
                        (emit_reserved == 4'd0) && !emit_out_valid;

    assign busy = (state_q == ST_PREP) || (state_q == ST_CONFIG) ||
                  (state_q == ST_ISSUE) || (state_q == ST_RUN) ||
                  (state_q == ST_FAIL) || (state_q == ST_ABORT);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_status_q != STATUS_OK;
    assign done_status = done_status_q;
    assign derived_k = k_q;
    assign derived_m = m_q;
    assign derived_rowblocks = rowblocks_q;
    assign accepted_weight_beats = weight_stream_count_q;
    assign activation_wave_issues = engine_wave_issues;

    // Register the complete observation bundle at the projection island. The
    // recorder consumes the previous cycle's sample and adds one seal cycle at
    // command termination to retain the final sample.
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            metrics_probe <= 13'd0;
        end else begin
            metrics_probe[0] <= weight_fire;
            metrics_probe[1] <= read_ready && !read_valid;
            metrics_probe[2] <= (state_q == ST_RUN) && read_valid &&
                                !read_ready;
            metrics_probe[3] <= engine_metrics_selector_full;
            metrics_probe[6:4] <= engine_metrics_selector_level;
            metrics_probe[7] <= q8_req_fire;
            metrics_probe[8] <= q8_rsp_fire;
            metrics_probe[9] <= q8_rd_req_valid && !q8_rd_req_ready;
            metrics_probe[10] <= (q8_outstanding_q != 3'd0) &&
                                 !q8_rsp_fire;
            metrics_probe[11] <= engine_metrics_drain;
            metrics_probe[12] <= engine_metrics_bank_wait;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            q8_outstanding_q <= 3'd0;
        end else begin
            case ({q8_req_fire, q8_rsp_fire})
                2'b10: q8_outstanding_q <= q8_outstanding_q + 1'b1;
                2'b01: q8_outstanding_q <= q8_outstanding_q - 1'b1;
                default: ;
            endcase
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            op_q <= `PROJECTION_OP_QKV;
            token_mask_q <= 8'd0;
            weight_addr_q <= 64'd0;
            k_q <= 16'd0;
            m_q <= 18'd0;
            rowblocks_q <= 16'd0;
            k_blocks_q <= 16'd0;
            rowblock_k_product_q <= 32'd0;
            port_beats_q <= 32'd0;
            weight_fmt_q <= WEIGHT_Q1;
            semantic_q <= 1'b1;
            child_configured_q <= 1'b0;
            cfg_hidden_dim_q <= 13'd0;
            cfg_ffn_dim_q <= 15'd0;
            cfg_q_heads_q <= 6'd0;
            cfg_kv_heads_q <= 4'd0;
            cfg_position_base_q <= 17'd0;
            cfg_epsilon_q <= 32'd0;
            cfg_q_gamma_addr_q <= 64'd0;
            cfg_k_gamma_addr_q <= 64'd0;
            cfg_rope_addr_q <= 64'd0;
            logits_vocab_rows_q <= 18'd0;
            logits_emit_full_q <= 1'b0;
            engine_done_seen_q <= 1'b0;
            reader_done_seen_q <= 1'b0;
            child_done_seen_q <= 1'b0;
            done_status_q <= STATUS_OK;
            local_abort_q <= 1'b0;
            weight_stream_count_q <= 32'd0;
        end else begin
            local_abort_q <= 1'b0;

            if (clear) begin
                engine_done_seen_q <= 1'b0;
                reader_done_seen_q <= 1'b0;
                child_done_seen_q <= 1'b0;
                child_configured_q <= 1'b0;
                done_status_q <= STATUS_OK;
                weight_stream_count_q <= 32'd0;
                state_q <= ST_IDLE;
            end else if (abort_run) begin
                engine_done_seen_q <= 1'b0;
                reader_done_seen_q <= 1'b0;
                child_done_seen_q <= 1'b0;
                child_configured_q <= 1'b0;
                done_status_q <= STATUS_OK;
                weight_stream_count_q <= 32'd0;
                state_q <= (read_busy || (q8_outstanding_q != 0)) ?
                           ST_ABORT : ST_IDLE;
            end else begin
                case (state_q)
                    ST_IDLE: if (command_fire) begin
                        op_q <= cmd_op;
                        token_mask_q <= cmd_token_mask;
                        weight_addr_q <= cmd_addr0;
                        k_q <= shape_k;
                        m_q <= shape_m;
                        rowblocks_q <= shape_rowblocks;
                        k_blocks_q <= shape_k_blocks;
                        weight_fmt_q <= cmd_weight_fmt;
                        semantic_q <= op_semantic;
                        child_configured_q <= 1'b0;
                        cfg_hidden_dim_q <= cmd_hidden_dim;
                        cfg_ffn_dim_q <= cmd_ffn_dim;
                        cfg_q_heads_q <= cmd_q_heads;
                        cfg_kv_heads_q <= cmd_kv_heads;
                        cfg_position_base_q <= cmd_position_base;
                        cfg_epsilon_q <= cmd_epsilon;
                        cfg_q_gamma_addr_q <= cmd_addr1;
                        cfg_k_gamma_addr_q <= cmd_addr2;
                        cfg_rope_addr_q <= cmd_addr3;
                        logits_vocab_rows_q <= cmd_vocab_rows;
                        logits_emit_full_q <= cmd_emit_full_logits;
                        engine_done_seen_q <= 1'b0;
                        reader_done_seen_q <= 1'b0;
                        child_done_seen_q <= 1'b0;
                        weight_stream_count_q <= 32'd0;
                        done_status_q <= command_ok ? STATUS_OK :
                                                      STATUS_BAD_CMD;
                        state_q <= command_ok ? ST_PREP : ST_DONE;
                    end

                    ST_PREP: begin
                        rowblock_k_product_q <= rowblocks_q * k_blocks_q;
                        state_q <= ST_CONFIG;
                    end

                    ST_CONFIG: begin
                        port_beats_q <= weight_fmt_q == WEIGHT_Q1 ?
                            (rowblock_k_product_q +
                             (rowblock_k_product_q << 2)) :
                            (rowblock_k_product_q +
                             (rowblock_k_product_q << 3));
                        if (semantic_q && child_configured_q &&
                            sink_done_valid) begin
                            done_status_q <= sink_status_code(
                                sink_done_status);
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (semantic_q && !child_configured_q) begin
                            if (sink_cfg_fire)
                                child_configured_q <= 1'b1;
                        end else if (semantic_q && sink_projection_armed) begin
                            state_q <= ST_ISSUE;
                        end else if (!semantic_q && logits_start_fire) begin
                            child_configured_q <= 1'b1;
                            state_q <= ST_ISSUE;
                        end
                    end

                    ST_ISSUE: if (read_cmd_fire)
                        state_q <= ST_RUN;

                    ST_RUN: begin
                        if (weight_fire)
                            weight_stream_count_q <=
                                weight_stream_count_q + 1'b1;
                        if (engine_done)
                            engine_done_seen_q <= 1'b1;
                        if (read_done_fire)
                            reader_done_seen_q <= 1'b1;
                        if (semantic_q && semantic_done_fire)
                            child_done_seen_q <= 1'b1;
                        if (!semantic_q && logits_result_fire)
                            child_done_seen_q <= 1'b1;

                        if (weight_fire && read_last &&
                            ((weight_stream_count_q + 1'b1) != port_beats_q)) begin
                            done_status_q <= STATUS_WEIGHT_LEN;
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (read_error && read_valid) begin
                            done_status_q <= STATUS_READER;
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (read_done_fire && read_done_error) begin
                            done_status_q <= read_done_status == 8'd0 ?
                                             STATUS_READER :
                                             (STATUS_READER |
                                              {4'd0,
                                               read_done_status[3:0]});
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (engine_done && engine_error) begin
                            done_status_q <= STATUS_ENGINE;
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (semantic_q && semantic_done_fire &&
                                     sink_done_error) begin
                            done_status_q <= sink_status_code(
                                sink_done_status);
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (!semantic_q && logits_result_fire &&
                                     logits_result_error) begin
                            done_status_q <= STATUS_LOGITS |
                                             {4'd0,
                                              logits_result_status[3:0]};
                            local_abort_q <= 1'b1;
                            state_q <= ST_FAIL;
                        end else if (all_complete) begin
                            if ((weight_stream_count_q != port_beats_q) ||
                                (engine_weight_beats != port_beats_q))
                                done_status_q <= STATUS_WEIGHT_LEN;
                            state_q <= ST_DONE;
                        end
                    end

                    ST_FAIL: begin
                        if (!read_busy && read_cmd_ready &&
                            (q8_outstanding_q == 0))
                            state_q <= ST_DONE;
                    end

                    ST_ABORT: begin
                        if (!read_busy && read_cmd_ready &&
                            (q8_outstanding_q == 0) &&
                            sink_cfg_ready && !engine_busy &&
                            (emit_reserved == 0))
                            state_q <= ST_IDLE;
                    end

                    ST_DONE: if (done_ready) begin
                        child_configured_q <= 1'b0;
                        state_q <= ST_IDLE;
                    end

                    default: state_q <= ST_IDLE;
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && (q8_outstanding_q > 3'd2))
            $fatal(1, " projection_service Q8 arena over-issued");
        if (rst_n && read_cmd_valid && semantic_q &&
            !sink_projection_armed)
            $fatal(1, " projection_service reader acquired before sink");
        if (rst_n && read_cmd_valid && !child_configured_q)
            $fatal(1, " projection_service reader acquired before child config");
        if (rst_n && sink_cfg_valid && !semantic_q)
            $fatal(1, " projection_service semantic config on LM command");
        if (rst_n && logits_start_valid && semantic_q)
            $fatal(1, " projection_service logits config on semantic command");
        if (rst_n && sink_proj_valid && !semantic_q)
            $fatal(1, " projection_service LM record reached semantic sink");
        if (rst_n && command_fire && command_ok &&
            ({2'd0, cmd_hidden_blocks} > Q8_INPUT_BLOCK_LIMIT))
            $fatal(1, " projection_service hidden Q8 partition overflow");
        if (rst_n && command_fire && command_ok &&
            (cmd_ffn_blocks > Q8_FFN_BLOCK_LIMIT))
            $fatal(1, " projection_service FFN Q8 partition overflow");
        if (rst_n && q8_rd_req_valid &&
            (q8_rd_req_addr_ext >= Q8_ARENA_BLOCKS))
            $fatal(1, " projection_service Q8 read outside arena");
        if (rst_n && q8_rd_req_valid &&
            ({1'b0, engine_act_req_addr} >= (k_q >> 5)))
            $fatal(1, " projection_service activation block outside K");
    end
`endif

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n && !clear && !abort_run) begin
            assert(!(sink_cfg_valid && logits_start_valid));
            if (read_cmd_valid)
                assert(child_configured_q);
        end
        if (rst_n)
            cover(abort_run && (state_q == ST_CONFIG));
        if (rst_n && !clear && !abort_run &&
            $past(rst_n && !clear && !abort_run)) begin
            if ($past(sink_cfg_valid && !sink_cfg_ready)) begin
                assert(sink_cfg_valid);
                assert($stable({sink_cfg_mode, sink_cfg_token_mask,
                                sink_cfg_hidden_dim, sink_cfg_ffn_dim,
                                sink_cfg_q_heads, sink_cfg_kv_heads,
                                sink_cfg_position_base, sink_cfg_epsilon,
                                sink_cfg_q_gamma_addr,
                                sink_cfg_k_gamma_addr,
                                sink_cfg_rope_addr}));
            end
            if ($past(logits_start_valid && !logits_start_ready)) begin
                assert(logits_start_valid);
                assert($stable({logits_vocab_rows_q,
                                logits_emit_full_q}));
            end
            if ($past(command_fire && !command_ok)) begin
                assert(state_q == ST_DONE);
                assert(!sink_cfg_valid && !logits_start_valid);
            end
        end
    end
`endif
endmodule

`default_nettype wire
