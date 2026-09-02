// Semantic sink for the token projection stream.
//
// The first implemented mode is fused QKV in immutable GQA-group order:
//   [Q heads mapped to KV head, K head, V head], repeated for KV heads 0..7.
// Projection records arrive rowblock -> active token -> row. One complete tile-8
// head is buffered, then Q/K are normalized in FP32 and rotated through the
// single shared RoPE leaf. Q is written to Query as FP32; K and V are written
// to NewKV as IEEE binary16. No raw-K model-wide arena exists. Gate/Up
// results occupy Q8 blocks 128..511 so they cannot alias the normalized
// projection input in blocks 0..127 while the GEMM is still consuming it.

`default_nettype none

module projection_sink (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [1:0]    cfg_mode,
    input  wire [7:0]    cfg_token_mask,
    input  wire [12:0]   cfg_hidden_dim,
    input  wire [14:0]   cfg_ffn_dim,
    input  wire [5:0]    cfg_q_heads,
    input  wire [3:0]    cfg_kv_heads,
    input  wire [16:0]   cfg_position_base,
    input  wire [31:0]   cfg_epsilon,
    input  wire [63:0]   cfg_q_gamma_addr,
    input  wire [63:0]   cfg_k_gamma_addr,
    input  wire [63:0]   cfg_rope_addr,
    input  wire          abort_run,

    output wire          projection_armed,
    output wire          busy,
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [15:0]   done_status,
    output wire [31:0]   done_cycles,
    output wire [4:0]    debug_state,

    // Two exact 32-record four-lane gamma bursts: Q, then K.
    output wire          gamma_req_valid,
    input  wire          gamma_req_ready,
    output wire [63:0]   gamma_req_addr,
    output wire [6:0]    gamma_req_words,
    input  wire          gamma_rsp_valid,
    output wire          gamma_rsp_ready,
    input  wire [127:0]  gamma_rsp_data,
    input  wire          gamma_rsp_last,
    input  wire          gamma_rsp_error,

    // One coefficient burst. Active waves are returned in wave order, with
    // 64 {cos,sin} four-lane records per non-empty wave.
    output wire          rope_req_valid,
    input  wire          rope_req_ready,
    output wire [63:0]   rope_req_addr,
    output wire [16:0]   rope_req_position_base,
    output wire [7:0]    rope_req_token_mask,
    output wire [7:0]    rope_req_records,
    input  wire          rope_rsp_valid,
    output wire          rope_rsp_ready,
    input  wire [255:0]  rope_rsp_data,
    input  wire          rope_rsp_last,
    input  wire          rope_rsp_error,

    // FP32 stream after gemm_emit. Ordering is checked, not inferred.
    input  wire          proj_valid,
    output wire          proj_ready,
    input  wire [2:0]    proj_token,
    input  wire [17:0]   proj_row,
    input  wire [31:0]   proj_data_f32,
    input  wire          proj_last,

    output wire          query_wr_valid,
    input  wire          query_wr_ready,
    output wire          query_wr_wave,
    output wire [11:0]   query_wr_addr,
    output wire [3:0]    query_wr_lane_mask,
    output wire [127:0]  query_wr_data,

    output wire          newkv_wr_valid,
    input  wire          newkv_wr_ready,
    output wire          newkv_wr_wave,
    output wire [10:0]   newkv_wr_addr,
    output wire [3:0]    newkv_wr_lane_mask,
    output wire [63:0]   newkv_wr_data,

    output wire          q8_wr_valid,
    input  wire          q8_wr_ready,
    output wire          q8_wr_wave,
    output wire [8:0]    q8_wr_addr,
    output wire [3:0]    q8_wr_lane_mask,
    output wire [1087:0] q8_wr_data,

    // Shared Q8 service client. Gate/Up requests one 32-row transaction per
    // tile wave and retains ownership through the single out_last record.
    output wire          leaf_q8_cfg_valid,
    input  wire          leaf_q8_cfg_ready,
    output wire [14:0]   leaf_q8_cfg_rows,
    output wire [3:0]    leaf_q8_cfg_lane_mask,
    input  wire          leaf_q8_busy,
    output wire          leaf_q8_in_valid,
    input  wire          leaf_q8_in_ready,
    output wire [127:0]  leaf_q8_in_data,
    input  wire          leaf_q8_out_valid,
    output wire          leaf_q8_out_ready,
    input  wire [8:0]    leaf_q8_out_block,
    input  wire [1087:0] leaf_q8_out_data,
    input  wire [7:0]    leaf_q8_out_status,
    input  wire          leaf_q8_out_last,

    output wire          r_rd_req_valid,
    input  wire          r_rd_req_ready,
    output wire          r_rd_req_wave,
    output wire [11:0]   r_rd_req_addr,
    input  wire          r_rd_rsp_valid,
    output wire          r_rd_rsp_ready,
    input  wire [127:0]  r_rd_rsp_data,
    input  wire          r_rd_rsp_error,
    output wire          r_wr_valid,
    input  wire          r_wr_ready,
    output wire          r_wr_wave,
    output wire [11:0]   r_wr_addr,
    output wire [3:0]    r_wr_lane_mask,
    output wire [127:0]  r_wr_data
);
    localparam [1:0] MODE_QKV = 2'd0;
    localparam [1:0] MODE_GATE_UP = 2'd1;
    localparam [1:0] MODE_O_DELTA = 2'd2;
    localparam [1:0] MODE_DOWN_DELTA = 2'd3;
    localparam [8:0] Q8_FFN_BASE = 9'd128;
    localparam [14:0] Q8_FFN_BLOCK_LIMIT = 15'd384;

    localparam [4:0] ST_IDLE        = 5'd0;
    localparam [4:0] ST_GAMMA_REQ   = 5'd1;
    localparam [4:0] ST_GAMMA_LOAD  = 5'd2;
    localparam [4:0] ST_ROPE_REQ    = 5'd3;
    localparam [4:0] ST_ROPE_LOAD   = 5'd4;
    localparam [4:0] ST_INGEST      = 5'd5;
    localparam [4:0] ST_REDUCE_CFG  = 5'd6;
    localparam [4:0] ST_REDUCE_WAIT = 5'd7;
    localparam [4:0] ST_APPLY_CFG   = 5'd8;
    localparam [4:0] ST_APPLY_RUN   = 5'd9;
    localparam [4:0] ST_V_RUN       = 5'd10;
    localparam [4:0] ST_DONE        = 5'd11;
    localparam [4:0] ST_ABORT       = 5'd12;
    localparam [4:0] ST_RESID_CFG   = 5'd13;
    localparam [4:0] ST_RESID_RUN   = 5'd14;
    localparam [4:0] ST_GATE_Q8_CFG = 5'd15;
    localparam [4:0] ST_GATE_Q8_RUN = 5'd16;

    localparam [1:0] HEAD_Q = 2'd0;
    localparam [1:0] HEAD_K = 2'd1;
    localparam [1:0] HEAD_V = 2'd2;

    localparam [15:0] STATUS_BAD_CFG      = 16'h0001;
    localparam [15:0] STATUS_GAMMA_STREAM = 16'h0002;
    localparam [15:0] STATUS_ROPE_STREAM  = 16'h0004;
    localparam [15:0] STATUS_PROJ_ORDER   = 16'h0008;
    localparam [15:0] STATUS_PROJ_FRAME   = 16'h0010;
    localparam [15:0] STATUS_REDUCE       = 16'h0020;
    localparam [15:0] STATUS_INTERNAL     = 16'h0040;
    localparam [15:0] STATUS_RESID        = 16'h0080;
    localparam [15:0] STATUS_SWIGLU       = 16'h0100;
    localparam [15:0] STATUS_Q8           = 16'h0200;

    reg [4:0] state_q;
    reg [1:0] mode_q;
    reg [7:0] token_mask_q;
    reg [12:0] hidden_dim_q;
    reg [14:0] ffn_dim_q;
    reg [5:0] q_heads_q;
    reg [16:0] position_base_q;
    reg [31:0] epsilon_q;
    reg [63:0] q_gamma_addr_q;
    reg [63:0] k_gamma_addr_q;
    reg [63:0] rope_addr_q;
    reg [15:0] projection_rows_q;
    reg armed_q;
    reg error_q;
    reg [15:0] status_q;
    reg [31:0] cycles_q;

    reg gamma_kind_q;
    reg [5:0] gamma_count_q;
    reg gamma_active_q;
    reg gamma_stream_error_q;
    reg [7:0] rope_count_q;
    reg rope_active_q;
    reg rope_stream_error_q;

    (* ram_style = "distributed" *) reg [127:0] gamma_mem [0:63];
    (* ram_style = "block" *) reg [255:0] rope_mem [0:127];

    // Four physical lane banks, with wave folded into the address MSB.
    (* ram_style = "block" *) reg [31:0] head_lane0 [0:255];
    (* ram_style = "block" *) reg [31:0] head_lane1 [0:255];
    (* ram_style = "block" *) reg [31:0] head_lane2 [0:255];
    (* ram_style = "block" *) reg [31:0] head_lane3 [0:255];
    reg head_wr_valid_q;
    reg [1:0] head_wr_lane_q;
    reg [7:0] head_wr_addr_q;
    reg [31:0] head_wr_data_q;

    // One 32-coordinate tile-8 SwiGLU tile feeds the single shared Q8 leaf.
    (* ram_style = "distributed" *) reg [31:0] gate_lane0 [0:63];
    (* ram_style = "distributed" *) reg [31:0] gate_lane1 [0:63];
    (* ram_style = "distributed" *) reg [31:0] gate_lane2 [0:63];
    (* ram_style = "distributed" *) reg [31:0] gate_lane3 [0:63];

    reg [17:0] expect_block_base_q;
    reg [2:0] expect_token_q;
    reg [3:0] expect_row_lane_q;
    reg ingress_complete_q;
    reg [1:0] head_kind_q;
    reg [5:0] head_index_q;
    reg wave_q;
    reg [127:0] inv_rms_q;

    function automatic [2:0] first_active_token(input [7:0] mask);
        begin
            casez (mask)
                8'b???????1: first_active_token = 3'd0;
                8'b??????10: first_active_token = 3'd1;
                8'b?????100: first_active_token = 3'd2;
                8'b????1000: first_active_token = 3'd3;
                8'b???10000: first_active_token = 3'd4;
                8'b??100000: first_active_token = 3'd5;
                8'b?1000000: first_active_token = 3'd6;
                default: first_active_token = 3'd7;
            endcase
        end
    endfunction

    function automatic [2:0] last_active_token(input [7:0] mask);
        integer scan;
        begin
            last_active_token = 3'd0;
            for (scan = 0; scan < 8; scan = scan + 1)
                if (mask[scan]) last_active_token = scan[2:0];
        end
    endfunction

    function automatic [2:0] next_active_token(
        input [7:0] mask,
        input [2:0] current
    );
        integer scan;
        reg found;
        begin
            next_active_token = current;
            found = 1'b0;
            for (scan = 0; scan < 8; scan = scan + 1) begin
                if (!found && (scan > current) && mask[scan]) begin
                    next_active_token = scan[2:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function automatic [3:0] wave_mask(
        input [7:0] mask,
        input wave
    );
        wave_mask = wave ? mask[7:4] : mask[3:0];
    endfunction

    function automatic [31:0] gamma_scalar(
        input [127:0] word,
        input [1:0] select
    );
        begin
            case (select)
                2'd0: gamma_scalar = word[31:0];
                2'd1: gamma_scalar = word[63:32];
                2'd2: gamma_scalar = word[95:64];
                default: gamma_scalar = word[127:96];
            endcase
        end
    endfunction

    function automatic finite4(input [127:0] value);
        finite4 = (value[30:23] != 8'hff) &&
                  (value[62:55] != 8'hff) &&
                  (value[94:87] != 8'hff) &&
                  (value[126:119] != 8'hff);
    endfunction

    wire cfg_hidden_ok = (cfg_hidden_dim == 13'd2048) ||
                         (cfg_hidden_dim == 13'd2560) ||
                         (cfg_hidden_dim == 13'd4096);
    wire cfg_ffn_ok = (cfg_ffn_dim == 15'd6144) ||
                      (cfg_ffn_dim == 15'd9728) ||
                      (cfg_ffn_dim == 15'd12288);
    wire cfg_common_ok = (cfg_token_mask != 8'd0) && cfg_hidden_ok &&
                         cfg_ffn_ok;
    wire cfg_qkv_ok = ((cfg_q_heads == 6'd16) ||
                       (cfg_q_heads == 6'd32)) &&
                      (cfg_kv_heads == 4'd8) &&
                      !cfg_epsilon[31] &&
                      (cfg_epsilon[30:23] != 8'd0) &&
                      (cfg_epsilon[30:23] != 8'hff) &&
                      (cfg_q_gamma_addr != 64'd0) &&
                      (cfg_q_gamma_addr[5:0] == 6'd0) &&
                      (cfg_k_gamma_addr != 64'd0) &&
                      (cfg_k_gamma_addr[5:0] == 6'd0) &&
                      (cfg_rope_addr != 64'd0) &&
                      (cfg_rope_addr[5:0] == 6'd0) &&
                      (({1'b0, cfg_position_base} +
                        {14'd0, last_active_token(cfg_token_mask)} +
                        18'd1) <= 18'd65536);
    wire cfg_mode_ok = (cfg_mode == MODE_QKV) ? cfg_qkv_ok :
                       ((cfg_mode == MODE_GATE_UP) ||
                        (cfg_mode == MODE_O_DELTA) ||
                        (cfg_mode == MODE_DOWN_DELTA));
    wire cfg_shape_ok = cfg_common_ok && cfg_mode_ok;

    assign cfg_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    wire cfg_fire = cfg_valid && cfg_ready;
    assign projection_armed = armed_q;
    assign busy = state_q != ST_IDLE;
    assign done_valid = rst_n && !clear && !abort_run &&
                        (state_q == ST_DONE);
    assign done_error = error_q;
    assign done_status = status_q;
    assign done_cycles = cycles_q;
    assign debug_state = state_q;

    assign gamma_req_valid = !clear && !abort_run &&
                             (state_q == ST_GAMMA_REQ);
    assign gamma_req_addr = gamma_kind_q ? k_gamma_addr_q : q_gamma_addr_q;
    assign gamma_req_words = 7'd32;
    wire gamma_req_fire = gamma_req_valid && gamma_req_ready;
    assign gamma_rsp_ready = !clear && ((state_q == ST_GAMMA_LOAD) ||
                                       (state_q == ST_ABORT));
    wire gamma_rsp_fire = gamma_rsp_valid && gamma_rsp_ready;
    wire gamma_expected_last = gamma_count_q == 6'd31;

    wire [1:0] active_waves = {|token_mask_q[7:4],
                               |token_mask_q[3:0]};
    wire [7:0] expected_rope_records =
        active_waves == 2'b11 ? 8'd128 : 8'd64;
    wire [7:0] rope_store_index =
        active_waves == 2'b10 ? (8'd64 + rope_count_q) : rope_count_q;
    assign rope_req_valid = !clear && !abort_run &&
                            (state_q == ST_ROPE_REQ);
    assign rope_req_addr = rope_addr_q;
    assign rope_req_position_base = position_base_q;
    assign rope_req_token_mask = token_mask_q;
    assign rope_req_records = expected_rope_records;
    wire rope_req_fire = rope_req_valid && rope_req_ready;
    assign rope_rsp_ready = !clear && ((state_q == ST_ROPE_LOAD) ||
                                      (state_q == ST_ABORT));
    wire rope_rsp_fire = rope_rsp_valid && rope_rsp_ready;
    wire rope_expected_last =
        (rope_count_q + 1'b1) == expected_rope_records;

    // Constant-bound row classification avoids a synthesized divider for the
    // 768-row Q32 GQA group.
    reg [2:0] proj_group;
    reg [9:0] proj_group_offset;
    reg [1:0] proj_head_kind;
    reg [5:0] proj_head_index;
    reg [6:0] proj_head_dim;
    always @(*) begin
        proj_group = 3'd0;
        proj_group_offset = 10'd0;
        if (q_heads_q == 6'd16) begin
            proj_group = proj_row[11:9];
            proj_group_offset = {1'b0, proj_row[8:0]};
            if (proj_group_offset < 10'd256) begin
                proj_head_kind = HEAD_Q;
                proj_head_index = {2'd0, proj_group, 1'b0} +
                                  {5'd0, proj_group_offset[7]};
                proj_head_dim = proj_group_offset[6:0];
            end else if (proj_group_offset < 10'd384) begin
                proj_head_kind = HEAD_K;
                proj_head_index = {3'd0, proj_group};
                proj_head_dim = proj_group_offset[6:0];
            end else begin
                proj_head_kind = HEAD_V;
                proj_head_index = {3'd0, proj_group};
                proj_head_dim = proj_group_offset[6:0];
            end
        end else begin
            if (proj_row < 18'd768) begin proj_group = 3'd0; proj_group_offset = proj_row[9:0]; end
            else if (proj_row < 18'd1536) begin proj_group = 3'd1; proj_group_offset = proj_row[9:0] - 10'd768; end
            else if (proj_row < 18'd2304) begin proj_group = 3'd2; proj_group_offset = proj_row[9:0] - 10'd512; end
            else if (proj_row < 18'd3072) begin proj_group = 3'd3; proj_group_offset = proj_row[9:0] - 10'd256; end
            else if (proj_row < 18'd3840) begin proj_group = 3'd4; proj_group_offset = proj_row[9:0]; end
            else if (proj_row < 18'd4608) begin proj_group = 3'd5; proj_group_offset = proj_row[9:0] - 10'd768; end
            else if (proj_row < 18'd5376) begin proj_group = 3'd6; proj_group_offset = proj_row[9:0] - 10'd512; end
            else begin proj_group = 3'd7; proj_group_offset = proj_row[9:0] - 10'd256; end

            if (proj_group_offset < 10'd512) begin
                proj_head_kind = HEAD_Q;
                proj_head_index = {1'd0, proj_group, 2'b00} +
                                  {4'd0, proj_group_offset[8:7]};
                proj_head_dim = proj_group_offset[6:0];
            end else if (proj_group_offset < 10'd640) begin
                proj_head_kind = HEAD_K;
                proj_head_index = {3'd0, proj_group};
                proj_head_dim = proj_group_offset[6:0];
            end else begin
                proj_head_kind = HEAD_V;
                proj_head_index = {3'd0, proj_group};
                proj_head_dim = proj_group_offset[6:0];
            end
        end
    end

    wire [2:0] first_token = first_active_token(token_mask_q);
    wire [2:0] last_token = last_active_token(token_mask_q);
    wire proj_order_ok = (proj_token == expect_token_q) &&
                         (proj_row == (expect_block_base_q +
                                      {14'd0, expect_row_lane_q})) &&
                         token_mask_q[proj_token] &&
                         (proj_row < {2'd0, projection_rows_q});
    wire proj_is_final = (proj_row + 1'b1 ==
                          {2'd0, projection_rows_q}) &&
                         (proj_token == last_token);
    wire proj_frame_ok = proj_last == proj_is_final;

    reg gate_hold_valid_q;
    reg [31:0] gate_hold_data_q;
    wire swiglu_in_ready;
    wire swiglu_in_ready_core;
    wire swiglu_out_valid;
    wire swiglu_out_ready;
    wire [31:0] swiglu_out_data;
    wire swiglu_out_last;
    wire [1:0] swiglu_out_status;
    wire gate_expects_up = expect_row_lane_q[0];
    wire gate_proj_ready = gate_expects_up ?
                           (gate_hold_valid_q && swiglu_in_ready) :
                           !gate_hold_valid_q;
    assign proj_ready = rst_n && !clear && !abort_run &&
                        (state_q == ST_INGEST) &&
                        ((mode_q != MODE_GATE_UP) || gate_proj_ready);
    wire proj_fire = proj_valid && proj_ready;
    wire swiglu_in_valid = (state_q == ST_INGEST) &&
                           (mode_q == MODE_GATE_UP) && proj_valid &&
                           gate_expects_up && gate_hold_valid_q;

    swiglu #(.RESERVE_DEPTH(64)) u_swiglu (
        .clk(clk), .rst_n(rst_n),
        .abort(clear || abort_run || (state_q == ST_ABORT)),
        .in_valid(swiglu_in_valid), .in_ready(swiglu_in_ready),
        .in_ready_core(swiglu_in_ready_core),
        .in_gate(gate_hold_data_q), .in_up(proj_data_f32),
        .in_last(proj_last), .out_valid(swiglu_out_valid),
        .out_ready(swiglu_out_ready), .out_data(swiglu_out_data),
        .out_last(swiglu_out_last), .out_status(swiglu_out_status)
    );

    reg [8:0] gate_tile_block_q;
    reg [1:0] gate_group_q;
    reg [2:0] gate_out_token_q;
    reg [2:0] gate_coord_lane_q;
    wire [4:0] gate_tile_coord = {gate_group_q,
                                  gate_coord_lane_q};
    wire [5:0] gate_store_addr = {gate_out_token_q[2],
                                  gate_tile_coord};
    assign swiglu_out_ready = !clear && (state_q == ST_INGEST) &&
                              (mode_q == MODE_GATE_UP);
    wire swiglu_out_fire = swiglu_out_valid && swiglu_out_ready;
    wire gate_output_final =
        (({6'd0, gate_tile_block_q} + 15'd1) == (ffn_dim_q >> 5)) &&
        (gate_group_q == 2'd3) &&
        (gate_out_token_q == last_token) &&
        (gate_coord_lane_q == 3'd7);
    wire gate_tile_complete = swiglu_out_fire &&
        (gate_group_q == 2'd3) &&
        (gate_out_token_q == last_token) &&
        (gate_coord_lane_q == 3'd7);
    wire qkv_head_complete = proj_fire && (mode_q == MODE_QKV) &&
                             (proj_head_dim == 7'd127) &&
                             (proj_token == last_token);
    wire delta_block_complete = proj_fire &&
        ((mode_q == MODE_O_DELTA) || (mode_q == MODE_DOWN_DELTA)) &&
        (proj_row[3:0] == 4'd15) && (proj_token == last_token);
    wire [7:0] proj_head_write_addr = mode_q == MODE_QKV ?
        {proj_token[2], proj_head_dim} :
        {proj_token[2], 3'd0, proj_row[3:0]};

    // ---- Head-local RMS reduction ----
    wire reduce_cfg_ready;
    wire reduce_busy;
    wire reduce_src_req_valid;
    wire reduce_src_req_ready;
    wire [11:0] reduce_src_req_addr;
    wire reduce_src_rsp_valid;
    wire reduce_src_rsp_ready;
    wire [127:0] reduce_src_rsp_data;
    wire reduce_result_valid;
    wire reduce_result_error;
    wire [7:0] reduce_result_status;
    wire [127:0] reduce_result_inv;
    wire child_abort = clear || abort_run || (state_q == ST_ABORT);

     rms_reduce4 u_reduce (
        .clk(clk), .rst_n(rst_n && !clear),
        .cfg_valid(!clear && (state_q == ST_REDUCE_CFG)),
        .cfg_ready(reduce_cfg_ready), .cfg_rows(13'd128),
        .cfg_lane_mask(wave_mask(token_mask_q, wave_q)),
        .cfg_epsilon(epsilon_q), .abort_run(child_abort),
        .busy(reduce_busy),
        .src_req_valid(reduce_src_req_valid),
        .src_req_ready(reduce_src_req_ready),
        .src_req_addr(reduce_src_req_addr),
        .src_rsp_valid(reduce_src_rsp_valid),
        .src_rsp_ready(reduce_src_rsp_ready),
        .src_rsp_data(reduce_src_rsp_data), .src_rsp_error(1'b0),
        .result_valid(reduce_result_valid),
        .result_ready(!clear && (state_q == ST_REDUCE_WAIT)),
        .result_error(reduce_result_error),
        .result_status(reduce_result_status),
        .result_inv_rms(reduce_result_inv)
    );

    // The inferred BRAM read port feeds an explicit elastic output register.
    // Tags follow the data so NEOX dimension order remains exact under stalls.
    reg head_mem_valid_q;
    reg [127:0] head_mem_data_q;
    reg [6:0] head_mem_dim_q;
    reg head_rsp_valid_q;
    reg [127:0] head_rsp_data_q;
    reg [6:0] head_rsp_dim_q;
    wire head_req_ready;
    wire head_rsp_ready;
    wire head_rsp_fire = head_rsp_valid_q && head_rsp_ready;
    wire reduce_owns_head = (state_q == ST_REDUCE_CFG) ||
                            (state_q == ST_REDUCE_WAIT);
    wire reduce_drains_head = reduce_owns_head ||
                              ((state_q == ST_ABORT) && reduce_busy);
    assign reduce_src_req_ready = reduce_owns_head && head_req_ready;
    wire reduce_req_fire = reduce_src_req_valid && reduce_src_req_ready;
    assign reduce_src_rsp_valid = reduce_drains_head && head_rsp_valid_q;
    assign reduce_src_rsp_data = head_rsp_data_q;

    // ---- Shared normalized apply and single-flight RoPE ----
    wire apply_cfg_ready;
    wire apply_busy;
    reg apply_stage_valid_q;
    reg [127:0] apply_stage_data_q;
    reg [6:0] apply_stage_dim_q;
    reg [127:0] apply_stage_gamma_q;
    reg [7:0] apply_issue_count_q;
    reg [7:0] apply_output_count_q;
    wire apply_in_ready;
    wire apply_out_valid;
    wire apply_out_ready;
    wire [127:0] apply_out_data;
    wire apply_out_last;

     norm_apply4 u_apply (
        .clk(clk), .rst_n(rst_n && !clear),
        .cfg_valid(!clear && (state_q == ST_APPLY_CFG)),
        .cfg_ready(apply_cfg_ready), .cfg_rows(13'd128),
        .cfg_lane_mask(wave_mask(token_mask_q, wave_q)),
        .cfg_inv_rms(inv_rms_q), .abort_run(child_abort),
        .busy(apply_busy), .in_valid(apply_stage_valid_q),
        .in_ready(apply_in_ready), .in_data(apply_stage_data_q),
        .in_gamma(apply_stage_gamma_q),
        .out_valid(apply_out_valid), .out_ready(apply_out_ready),
        .out_data(apply_out_data), .out_last(apply_out_last)
    );
    wire apply_in_fire = apply_stage_valid_q && apply_in_ready;
    wire apply_stage_capacity = !apply_stage_valid_q || apply_in_fire;
    wire apply_read_want = (state_q == ST_APPLY_RUN) &&
                           (apply_issue_count_q < 8'd128);
    wire apply_issue = apply_read_want && head_req_ready;
    wire apply_head_fire = (state_q == ST_APPLY_RUN) && head_rsp_fire;
    // Qwen3 uses NEOX RoPE: pair p consumes dimensions p and p+64.  Walk the
    // resident head in that order so the shared RoPE leaf still sees adjacent
    // x0/x1 records without another head-sized buffer.
    wire [6:0] apply_issue_dim = {
        apply_issue_count_q[0], apply_issue_count_q[6:1]
    };

    reg even_valid_q;
    reg [127:0] even_data_q;
    reg [255:0] pair_coeff_q;
    reg rope_inflight_q;
    reg rope_hold_valid_q;
    reg [255:0] rope_hold_data_q;
    reg rope_write_phase_q;
    reg [6:0] rope_pair_q;

    wire apply_is_odd = apply_output_count_q[0];
    assign apply_out_ready = !apply_is_odd ? !even_valid_q :
                             (even_valid_q && !rope_inflight_q &&
                              !rope_hold_valid_q);
    wire apply_out_fire = apply_out_valid && apply_out_ready;
    wire rope_feed = !clear && (state_q == ST_APPLY_RUN) &&
                     apply_out_fire &&
                     apply_is_odd;
    wire [255:0] rope_pair_data = {
        apply_out_data[127:96], even_data_q[127:96],
        apply_out_data[95:64],  even_data_q[95:64],
        apply_out_data[63:32],  even_data_q[63:32],
        apply_out_data[31:0],   even_data_q[31:0]
    };
    wire rope_pipe_valid;
    wire [255:0] rope_pipe_data;
     rope4 u_rope (
        .clk(clk), .rst_n(rst_n && !child_abort),
        .in_valid(rope_feed), .in_data(rope_pair_data),
        .in_coeff(pair_coeff_q),
        .out_valid(rope_pipe_valid), .out_data(rope_pipe_data)
    );

    wire [127:0] rope_write_fp32 = rope_write_phase_q ? {
        rope_hold_data_q[255:224], rope_hold_data_q[191:160],
        rope_hold_data_q[127:96], rope_hold_data_q[63:32]
    } : {
        rope_hold_data_q[223:192], rope_hold_data_q[159:128],
        rope_hold_data_q[95:64], rope_hold_data_q[31:0]
    };
    wire rope_write_valid = !clear && (state_q == ST_APPLY_RUN) &&
                            rope_hold_valid_q;
    wire rope_write_ready = head_kind_q == HEAD_Q ? query_wr_ready :
                                                    newkv_wr_ready;
    wire rope_write_fire = rope_write_valid && rope_write_ready;
    wire [6:0] rope_write_dim = rope_write_phase_q ?
        {1'b1, rope_pair_q[5:0]} : {1'b0, rope_pair_q[5:0]};
    wire rope_head_done = rope_write_fire && rope_write_phase_q &&
                          (rope_pair_q == 7'd63);

    // ---- Direct V drain ----
    reg [7:0] v_issue_count_q;
    reg v_stage_valid_q;
    reg [127:0] v_stage_data_q;
    reg [6:0] v_stage_dim_q;
    wire v_write_valid = !clear && (state_q == ST_V_RUN) &&
                         v_stage_valid_q;
    wire [127:0] f16_convert_input = v_write_valid ?
                                      v_stage_data_q : rope_write_fp32;
    wire [63:0] shared_write_f16;
    genvar f16_lane;
    generate
        for (f16_lane = 0; f16_lane < 4; f16_lane = f16_lane + 1) begin : g_f16
             f32_to_f16 u_f16 (
                .in(f16_convert_input[f16_lane*32 +: 32]),
                .out(shared_write_f16[f16_lane*16 +: 16])
            );
        end
    endgenerate
    wire v_write_fire = v_write_valid && newkv_wr_ready;
    wire v_stage_capacity = !v_stage_valid_q || v_write_fire;
    wire v_read_want = (state_q == ST_V_RUN) &&
                       (v_issue_count_q < 8'd128);
    wire v_issue = v_read_want && head_req_ready;
    wire v_head_fire = (state_q == ST_V_RUN) && head_rsp_fire;
    wire v_head_done = v_write_fire && (v_stage_dim_q == 7'd127);

    // ---- Direct O/DOWN residual path ----
    reg [11:0] resid_block_base_q;
    reg [4:0] resid_issue_count_q;
    reg [4:0] resid_output_count_q;
    reg r_outstanding_q;
    wire resid_cfg_ready;
    wire resid_busy;
    wire resid_in_ready;
    wire resid_out_valid;
    wire [127:0] resid_out_data;
    wire resid_out_last;
    wire resid_in_valid = (state_q == ST_RESID_RUN) &&
                          head_rsp_valid_q && r_rd_rsp_valid;
    wire resid_in_fire = resid_in_valid && resid_in_ready;
    wire resid_out_fire = resid_out_valid && r_wr_ready;

     residual4 u_residual (
        .clk(clk), .rst_n(rst_n && !clear),
        .cfg_valid(!clear && (state_q == ST_RESID_CFG)),
        .cfg_ready(resid_cfg_ready), .cfg_rows(13'd16),
        .cfg_lane_mask(wave_mask(token_mask_q, wave_q)),
        .abort_run(child_abort), .busy(resid_busy),
        .in_valid(resid_in_valid), .in_ready(resid_in_ready),
        .in_residual(r_rd_rsp_data), .in_delta(head_rsp_data_q),
        .out_valid(resid_out_valid), .out_ready(r_wr_ready),
        .out_data(resid_out_data), .out_last(resid_out_last)
    );

    assign r_rd_req_valid = rst_n && !clear && !abort_run &&
        (state_q == ST_RESID_RUN) && !r_outstanding_q &&
        (resid_issue_count_q < 5'd16) && head_req_ready;
    assign r_rd_req_wave = wave_q;
    assign r_rd_req_addr = resid_block_base_q +
                           {7'd0, resid_issue_count_q[3:0]};
    wire r_req_fire = r_rd_req_valid && r_rd_req_ready;
    assign r_rd_rsp_ready = !clear && (state_q == ST_ABORT ? 1'b1 :
                            ((state_q == ST_RESID_RUN) &&
                             head_rsp_valid_q && resid_in_ready));
    wire r_rsp_fire = r_rd_rsp_valid && r_rd_rsp_ready;

    assign r_wr_valid = !clear && (state_q == ST_RESID_RUN) &&
                        resid_out_valid;
    assign r_wr_wave = wave_q;
    assign r_wr_addr = resid_block_base_q +
                       {8'd0, resid_output_count_q[3:0]};
    assign r_wr_lane_mask = current_lane_mask;
    assign r_wr_data = resid_out_data;
    wire resid_wave_done = resid_out_fire && resid_out_last;

    wire [3:0] current_lane_mask = wave_mask(token_mask_q, wave_q);
    wire has_next_wave = !wave_q && (|token_mask_q[7:4]);

    reg [5:0] gate_q8_issue_count_q;
    reg gate_q8_stage_valid_q;
    reg [127:0] gate_q8_stage_data_q;
    wire gate_q8_stage_fire = gate_q8_stage_valid_q &&
                              leaf_q8_in_ready;
    wire gate_q8_stage_capacity = !gate_q8_stage_valid_q ||
                                  gate_q8_stage_fire;
    wire gate_q8_read_issue = (state_q == ST_GATE_Q8_RUN) &&
        (gate_q8_issue_count_q < 6'd32) && gate_q8_stage_capacity;
    wire gate_q8_out_fire = leaf_q8_out_valid && leaf_q8_out_ready;

    assign leaf_q8_cfg_valid = rst_n && !clear && !abort_run &&
                               (state_q == ST_GATE_Q8_CFG);
    assign leaf_q8_cfg_rows = 15'd32;
    assign leaf_q8_cfg_lane_mask = current_lane_mask;
    assign leaf_q8_in_valid = !clear && (state_q == ST_GATE_Q8_RUN) &&
                              gate_q8_stage_valid_q;
    assign leaf_q8_in_data = gate_q8_stage_data_q;
    assign leaf_q8_out_ready = !clear && (state_q == ST_GATE_Q8_RUN) &&
                               q8_wr_ready;

    assign q8_wr_valid = !clear && (state_q == ST_GATE_Q8_RUN) &&
                         leaf_q8_out_valid;
    assign q8_wr_wave = wave_q;
    assign q8_wr_addr = Q8_FFN_BASE + gate_tile_block_q;
    assign q8_wr_lane_mask = current_lane_mask;
    assign q8_wr_data = leaf_q8_out_data;

    assign query_wr_valid = rope_write_valid &&
                            (head_kind_q == HEAD_Q);
    assign query_wr_wave = wave_q;
    assign query_wr_addr = {head_index_q[4:0], 7'd0} +
                           {5'd0, rope_write_dim};
    assign query_wr_lane_mask = current_lane_mask;
    assign query_wr_data = rope_write_fp32;

    assign newkv_wr_valid = (rope_write_valid &&
                             (head_kind_q == HEAD_K)) || v_write_valid;
    assign newkv_wr_wave = wave_q;
    assign newkv_wr_addr = v_write_valid ?
        ({1'b1, head_index_q[2:0], v_stage_dim_q}) :
        ({1'b0, head_index_q[2:0], rope_write_dim});
    assign newkv_wr_lane_mask = current_lane_mask;
    assign newkv_wr_data = shared_write_f16;

    // All head consumers are mutually exclusive states. One explicit read
    // port prevents synthesis from replicating the head buffer. The synchronous
    // BRAM output and held response are independently flow controlled.
    wire head_rsp_slot_available = !head_rsp_valid_q || head_rsp_ready;
    wire head_mem_to_rsp = head_mem_valid_q && head_rsp_slot_available;
    wire head_mem_slot_available = !head_mem_valid_q || head_mem_to_rsp;
    assign head_req_ready = rst_n && !clear && !abort_run &&
                            head_mem_slot_available;
    assign head_rsp_ready = (state_q == ST_ABORT) ?
        (reduce_busy ? reduce_src_rsp_ready : 1'b1) :
        reduce_owns_head ? reduce_src_rsp_ready :
        (state_q == ST_APPLY_RUN) ? apply_stage_capacity :
        (state_q == ST_V_RUN) ? v_stage_capacity :
        (state_q == ST_RESID_RUN) ?
            (r_rd_rsp_valid && resid_in_ready) : 1'b0;
    wire head_req_fire = reduce_req_fire || apply_issue || v_issue ||
                         r_req_fire;
    wire [7:0] head_accept_addr = reduce_req_fire ?
        {wave_q, reduce_src_req_addr[6:0]} :
        apply_issue ? {wave_q, apply_issue_dim} :
        v_issue ? {wave_q, v_issue_count_q[6:0]} :
                  {wave_q, 3'd0, resid_issue_count_q[3:0]};
    wire [6:0] head_accept_dim = reduce_req_fire ?
        reduce_src_req_addr[6:0] :
        apply_issue ? apply_issue_dim :
        v_issue ? v_issue_count_q[6:0] :
                  {3'd0, resid_issue_count_q[3:0]};
    wire [7:0] head_read_addr = head_accept_addr;

    integer lane;
    // Keep payload RAM pins local. A previously accepted write may retire on a
    // clear edge, but the intent valid and every read response are canceled.
    always @(posedge clk) begin
        if (head_wr_valid_q) begin
            case (head_wr_lane_q)
                2'd0: head_lane0[head_wr_addr_q] <= head_wr_data_q;
                2'd1: head_lane1[head_wr_addr_q] <= head_wr_data_q;
                2'd2: head_lane2[head_wr_addr_q] <= head_wr_data_q;
                default: head_lane3[head_wr_addr_q] <= head_wr_data_q;
            endcase
        end
        if (head_req_fire) begin
            head_mem_data_q <= {
                head_lane3[head_accept_addr],
                head_lane2[head_accept_addr],
                head_lane1[head_accept_addr],
                head_lane0[head_accept_addr]
            };
            head_mem_dim_q <= head_accept_dim;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q <= ST_IDLE;
            mode_q <= MODE_QKV;
            token_mask_q <= 8'd0;
            hidden_dim_q <= 13'd0;
            ffn_dim_q <= 15'd0;
            q_heads_q <= 6'd0;
            position_base_q <= 17'd0;
            epsilon_q <= 32'd0;
            q_gamma_addr_q <= 64'd0;
            k_gamma_addr_q <= 64'd0;
            rope_addr_q <= 64'd0;
            projection_rows_q <= 16'd0;
            armed_q <= 1'b0;
            error_q <= 1'b0;
            status_q <= 16'd0;
            cycles_q <= 32'd0;
            gamma_kind_q <= 1'b0;
            gamma_count_q <= 6'd0;
            gamma_active_q <= 1'b0;
            gamma_stream_error_q <= 1'b0;
            rope_count_q <= 8'd0;
            rope_active_q <= 1'b0;
            rope_stream_error_q <= 1'b0;
            expect_block_base_q <= 18'd0;
            expect_token_q <= 3'd0;
            expect_row_lane_q <= 4'd0;
            ingress_complete_q <= 1'b0;
            head_kind_q <= HEAD_Q;
            head_index_q <= 6'd0;
            wave_q <= 1'b0;
            inv_rms_q <= 128'd0;
            head_wr_valid_q <= 1'b0;
            head_mem_valid_q <= 1'b0;
            head_rsp_valid_q <= 1'b0;
            apply_stage_valid_q <= 1'b0;
            apply_issue_count_q <= 8'd0;
            apply_output_count_q <= 8'd0;
            even_valid_q <= 1'b0;
            even_data_q <= 128'd0;
            pair_coeff_q <= 256'd0;
            rope_inflight_q <= 1'b0;
            rope_hold_valid_q <= 1'b0;
            rope_hold_data_q <= 256'd0;
            rope_write_phase_q <= 1'b0;
            rope_pair_q <= 7'd0;
            v_issue_count_q <= 8'd0;
            v_stage_valid_q <= 1'b0;
            resid_block_base_q <= 12'd0;
            resid_issue_count_q <= 5'd0;
            resid_output_count_q <= 5'd0;
            r_outstanding_q <= 1'b0;
            gate_hold_valid_q <= 1'b0;
            gate_hold_data_q <= 32'd0;
            gate_tile_block_q <= 9'd0;
            gate_group_q <= 2'd0;
            gate_out_token_q <= 3'd0;
            gate_coord_lane_q <= 3'd0;
            gate_q8_issue_count_q <= 6'd0;
            gate_q8_stage_valid_q <= 1'b0;
            gate_q8_stage_data_q <= 128'd0;
        end else begin
            head_wr_valid_q <= 1'b0;
            if (state_q != ST_IDLE && state_q != ST_DONE)
                cycles_q <= cycles_q + 1'b1;

            // BRAM output and response stages are independently elastic. Local
            // abort stops acceptance and drains both stages.
            if (head_req_fire) begin
                head_mem_valid_q <= 1'b1;
            end else if (head_mem_to_rsp) begin
                head_mem_valid_q <= 1'b0;
            end
            if (head_mem_to_rsp) begin
                head_rsp_data_q <= head_mem_data_q;
                head_rsp_dim_q <= head_mem_dim_q;
                head_rsp_valid_q <= 1'b1;
            end else if (head_rsp_fire) begin
                head_rsp_valid_q <= 1'b0;
            end
            case ({r_req_fire, r_rsp_fire})
                2'b10: r_outstanding_q <= 1'b1;
                2'b01: r_outstanding_q <= 1'b0;
                default: begin end
            endcase

            if (abort_run && state_q != ST_IDLE) begin
                state_q <= ST_ABORT;
                armed_q <= 1'b0;
                apply_stage_valid_q <= 1'b0;
                even_valid_q <= 1'b0;
                rope_inflight_q <= 1'b0;
                rope_hold_valid_q <= 1'b0;
                v_stage_valid_q <= 1'b0;
                r_outstanding_q <= r_rsp_fire ? 1'b0 : r_outstanding_q;
                gate_hold_valid_q <= 1'b0;
                gate_q8_stage_valid_q <= 1'b0;
                if (gamma_rsp_fire && gamma_rsp_last)
                    gamma_active_q <= 1'b0;
                if (rope_rsp_fire && rope_rsp_last)
                    rope_active_q <= 1'b0;
            end else begin
                if (swiglu_out_fire) begin
                    case (gate_out_token_q[1:0])
                        2'd0: gate_lane0[gate_store_addr] <= swiglu_out_data;
                        2'd1: gate_lane1[gate_store_addr] <= swiglu_out_data;
                        2'd2: gate_lane2[gate_store_addr] <= swiglu_out_data;
                        default: gate_lane3[gate_store_addr] <= swiglu_out_data;
                    endcase
                    if (swiglu_out_status != 2'd0) begin
                        error_q <= 1'b1;
                        status_q <= status_q | STATUS_SWIGLU;
                    end
                    if (swiglu_out_last != gate_output_final) begin
                        error_q <= 1'b1;
                        status_q <= status_q | STATUS_PROJ_FRAME;
                    end

                    if (gate_coord_lane_q != 3'd7) begin
                        gate_coord_lane_q <= gate_coord_lane_q + 1'b1;
                    end else if (gate_out_token_q != last_token) begin
                        gate_coord_lane_q <= 3'd0;
                        gate_out_token_q <= next_active_token(
                            token_mask_q, gate_out_token_q);
                    end else begin
                        gate_coord_lane_q <= 3'd0;
                        gate_out_token_q <= first_token;
                        if (gate_group_q != 2'd3) begin
                            gate_group_q <= gate_group_q + 1'b1;
                        end else begin
                            gate_group_q <= 2'd0;
                            wave_q <= |token_mask_q[3:0] ? 1'b0 : 1'b1;
                            state_q <= ST_GATE_Q8_CFG;
                        end
                    end
                end

                if (gate_q8_read_issue) begin
                    gate_q8_stage_data_q <= {
                        gate_lane3[{wave_q, gate_q8_issue_count_q[4:0]}],
                        gate_lane2[{wave_q, gate_q8_issue_count_q[4:0]}],
                        gate_lane1[{wave_q, gate_q8_issue_count_q[4:0]}],
                        gate_lane0[{wave_q, gate_q8_issue_count_q[4:0]}]
                    };
                    gate_q8_issue_count_q <= gate_q8_issue_count_q + 1'b1;
                    gate_q8_stage_valid_q <= 1'b1;
                end else if (gate_q8_stage_fire) begin
                    gate_q8_stage_valid_q <= 1'b0;
                end

                case (state_q)
                    ST_IDLE: if (cfg_fire) begin
                        mode_q <= cfg_mode;
                        token_mask_q <= cfg_token_mask;
                        hidden_dim_q <= cfg_hidden_dim;
                        ffn_dim_q <= cfg_ffn_dim;
                        q_heads_q <= cfg_q_heads;
                        position_base_q <= cfg_position_base;
                        epsilon_q <= cfg_epsilon;
                        q_gamma_addr_q <= cfg_q_gamma_addr;
                        k_gamma_addr_q <= cfg_k_gamma_addr;
                        rope_addr_q <= cfg_rope_addr;
                        projection_rows_q <= cfg_mode == MODE_QKV ?
                            (cfg_q_heads == 6'd16 ? 16'd4096 : 16'd6144) :
                            cfg_mode == MODE_GATE_UP ?
                                ({1'b0, cfg_ffn_dim} << 1) :
                                {3'd0, cfg_hidden_dim};
                        armed_q <= 1'b0;
                        error_q <= !cfg_shape_ok;
                        status_q <= cfg_shape_ok ? 16'd0 : STATUS_BAD_CFG;
                        cycles_q <= 32'd0;
                        gamma_kind_q <= 1'b0;
                        ingress_complete_q <= 1'b0;
                        expect_block_base_q <= 18'd0;
                        expect_token_q <= first_active_token(cfg_token_mask);
                        expect_row_lane_q <= 4'd0;
                        gate_hold_valid_q <= 1'b0;
                        gate_tile_block_q <= 9'd0;
                        gate_group_q <= 2'd0;
                        gate_out_token_q <= first_active_token(cfg_token_mask);
                        gate_coord_lane_q <= 3'd0;
                        gate_q8_issue_count_q <= 6'd0;
                        gate_q8_stage_valid_q <= 1'b0;
                        state_q <= !cfg_shape_ok ? ST_DONE :
                                   cfg_mode == MODE_QKV ? ST_GAMMA_REQ :
                                                        ST_INGEST;
                        armed_q <= cfg_shape_ok && (cfg_mode != MODE_QKV);
                    end

                    ST_GAMMA_REQ: if (gamma_req_fire) begin
                        gamma_count_q <= 6'd0;
                        gamma_active_q <= 1'b1;
                        gamma_stream_error_q <= 1'b0;
                        state_q <= ST_GAMMA_LOAD;
                    end

                    ST_GAMMA_LOAD: if (gamma_rsp_fire) begin
                        gamma_mem[{gamma_kind_q, gamma_count_q[4:0]}] <=
                            gamma_rsp_data;
                        if (gamma_rsp_error || !finite4(gamma_rsp_data) ||
                            (gamma_rsp_last != gamma_expected_last)) begin
                            gamma_stream_error_q <= 1'b1;
                            status_q <= status_q | STATUS_GAMMA_STREAM;
                        end
                        if (gamma_rsp_last) begin
                            gamma_active_q <= 1'b0;
                            if (gamma_stream_error_q || gamma_rsp_error ||
                                !finite4(gamma_rsp_data) ||
                                !gamma_expected_last) begin
                                error_q <= 1'b1;
                                state_q <= ST_DONE;
                            end else if (!gamma_kind_q) begin
                                gamma_kind_q <= 1'b1;
                                state_q <= ST_GAMMA_REQ;
                            end else begin
                                state_q <= ST_ROPE_REQ;
                            end
                        end else begin
                            gamma_count_q <= gamma_count_q + 1'b1;
                        end
                    end

                    ST_ROPE_REQ: if (rope_req_fire) begin
                        rope_count_q <= 8'd0;
                        rope_active_q <= 1'b1;
                        rope_stream_error_q <= 1'b0;
                        state_q <= ST_ROPE_LOAD;
                    end

                    ST_ROPE_LOAD: if (rope_rsp_fire) begin
                        rope_mem[rope_store_index[6:0]] <= rope_rsp_data;
                        if (rope_rsp_error || !finite4(rope_rsp_data[127:0]) ||
                            !finite4(rope_rsp_data[255:128]) ||
                            (rope_rsp_last != rope_expected_last)) begin
                            rope_stream_error_q <= 1'b1;
                            status_q <= status_q | STATUS_ROPE_STREAM;
                        end
                        if (rope_rsp_last) begin
                            rope_active_q <= 1'b0;
                            if (rope_stream_error_q || rope_rsp_error ||
                                !rope_expected_last) begin
                                error_q <= 1'b1;
                                state_q <= ST_DONE;
                            end else begin
                                armed_q <= 1'b1;
                                state_q <= ST_INGEST;
                            end
                        end else begin
                            rope_count_q <= rope_count_q + 1'b1;
                        end
                    end

                    ST_INGEST: if (proj_fire) begin
                        if (mode_q == MODE_GATE_UP) begin
                            if (!gate_expects_up) begin
                                gate_hold_valid_q <= 1'b1;
                                gate_hold_data_q <= proj_data_f32;
                            end else begin
                                gate_hold_valid_q <= 1'b0;
                            end
                        end else begin
                            head_wr_valid_q <= 1'b1;
                            head_wr_lane_q <= proj_token[1:0];
                            head_wr_addr_q <= proj_head_write_addr;
                            head_wr_data_q <= proj_data_f32;
                        end

                        if (!proj_order_ok) begin
                            error_q <= 1'b1;
                            status_q <= status_q | STATUS_PROJ_ORDER;
                        end
                        if (!proj_frame_ok) begin
                            error_q <= 1'b1;
                            status_q <= status_q | STATUS_PROJ_FRAME;
                        end
                        if (proj_last)
                            ingress_complete_q <= 1'b1;

                        if (expect_row_lane_q != 4'd15) begin
                            expect_row_lane_q <= expect_row_lane_q + 1'b1;
                        end else if (expect_token_q != last_token) begin
                            expect_row_lane_q <= 4'd0;
                            expect_token_q <= next_active_token(token_mask_q,
                                                               expect_token_q);
                        end else begin
                            expect_row_lane_q <= 4'd0;
                            expect_token_q <= first_token;
                            expect_block_base_q <= expect_block_base_q + 18'd16;
                        end

                        if (qkv_head_complete) begin
                            head_kind_q <= proj_head_kind;
                            head_index_q <= proj_head_index;
                            wave_q <= |token_mask_q[3:0] ? 1'b0 : 1'b1;
                            if (proj_head_kind == HEAD_V) begin
                                v_issue_count_q <= 8'd0;
                                v_stage_valid_q <= 1'b0;
                                state_q <= ST_V_RUN;
                            end else begin
                                state_q <= ST_REDUCE_CFG;
                            end
                        end else if (delta_block_complete) begin
                            resid_block_base_q <= {proj_row[11:4], 4'b0000};
                            wave_q <= |token_mask_q[3:0] ? 1'b0 : 1'b1;
                            state_q <= ST_RESID_CFG;
                        end
                    end

                    ST_REDUCE_CFG: if (reduce_cfg_ready)
                        state_q <= ST_REDUCE_WAIT;

                    ST_REDUCE_WAIT: if (reduce_result_valid) begin
                        if (reduce_result_error) begin
                            error_q <= 1'b1;
                            status_q <= status_q | STATUS_REDUCE |
                                        {8'd0, reduce_result_status};
                            state_q <= ST_DONE;
                        end else begin
                            inv_rms_q <= reduce_result_inv;
                            state_q <= ST_APPLY_CFG;
                        end
                    end

                    ST_APPLY_CFG: if (apply_cfg_ready) begin
                        apply_issue_count_q <= 8'd0;
                        apply_output_count_q <= 8'd0;
                        apply_stage_valid_q <= 1'b0;
                        even_valid_q <= 1'b0;
                        rope_inflight_q <= 1'b0;
                        rope_hold_valid_q <= 1'b0;
                        rope_write_phase_q <= 1'b0;
                        rope_pair_q <= 7'd0;
                        state_q <= ST_APPLY_RUN;
                    end

                    ST_APPLY_RUN: begin
                        if (apply_issue)
                            apply_issue_count_q <= apply_issue_count_q + 1'b1;
                        if (apply_head_fire) begin
                            apply_stage_data_q <= head_rsp_data_q;
                            apply_stage_dim_q <= head_rsp_dim_q;
                            apply_stage_gamma_q <= {4{gamma_scalar(
                                gamma_mem[{head_kind_q == HEAD_K,
                                           head_rsp_dim_q[6:2]}],
                                head_rsp_dim_q[1:0])}};
                            apply_stage_valid_q <= 1'b1;
                        end else if (apply_in_fire) begin
                            apply_stage_valid_q <= 1'b0;
                        end

                        if (apply_out_fire) begin
                            apply_output_count_q <= apply_output_count_q + 1'b1;
                            if (!apply_is_odd) begin
                                even_valid_q <= 1'b1;
                                even_data_q <= apply_out_data;
                                pair_coeff_q <= rope_mem[
                                    {wave_q, apply_output_count_q[6:1]}];
                            end else begin
                                even_valid_q <= 1'b0;
                                rope_inflight_q <= 1'b1;
                            end
                        end

                        if (rope_pipe_valid) begin
                            if (rope_hold_valid_q) begin
                                error_q <= 1'b1;
                                status_q <= status_q | STATUS_INTERNAL;
                            end
                            rope_hold_data_q <= rope_pipe_data;
                            rope_hold_valid_q <= 1'b1;
                            rope_inflight_q <= 1'b0;
                            rope_write_phase_q <= 1'b0;
                        end

                        if (rope_write_fire) begin
                            if (!rope_write_phase_q) begin
                                rope_write_phase_q <= 1'b1;
                            end else begin
                                rope_hold_valid_q <= 1'b0;
                                rope_write_phase_q <= 1'b0;
                                if (rope_pair_q != 7'd63)
                                    rope_pair_q <= rope_pair_q + 1'b1;
                            end
                        end

                        if (rope_head_done) begin
                            if (has_next_wave) begin
                                wave_q <= 1'b1;
                                state_q <= ST_REDUCE_CFG;
                            end else if (ingress_complete_q) begin
                                armed_q <= 1'b0;
                                state_q <= ST_DONE;
                            end else begin
                                state_q <= ST_INGEST;
                            end
                        end
                    end

                    ST_V_RUN: begin
                        if (v_issue)
                            v_issue_count_q <= v_issue_count_q + 1'b1;
                        if (v_head_fire) begin
                            v_stage_data_q <= head_rsp_data_q;
                            v_stage_dim_q <= head_rsp_dim_q;
                            v_stage_valid_q <= 1'b1;
                        end else if (v_write_fire) begin
                            v_stage_valid_q <= 1'b0;
                        end

                        if (v_head_done) begin
                            if (has_next_wave) begin
                                wave_q <= 1'b1;
                                v_issue_count_q <= 8'd0;
                                v_stage_valid_q <= 1'b0;
                            end else if (ingress_complete_q) begin
                                armed_q <= 1'b0;
                                state_q <= ST_DONE;
                            end else begin
                                state_q <= ST_INGEST;
                            end
                        end
                    end

                    ST_RESID_CFG: if (resid_cfg_ready) begin
                        resid_issue_count_q <= 5'd0;
                        resid_output_count_q <= 5'd0;
                        r_outstanding_q <= 1'b0;
                        state_q <= ST_RESID_RUN;
                    end

                    ST_RESID_RUN: begin
                        if (r_req_fire)
                            resid_issue_count_q <=
                                resid_issue_count_q + 1'b1;
                        if (resid_out_fire)
                            resid_output_count_q <=
                                resid_output_count_q + 1'b1;
                        if (r_rsp_fire && r_rd_rsp_error) begin
                            error_q <= 1'b1;
                            status_q <= status_q | STATUS_RESID;
                        end
                        if (resid_wave_done) begin
                            if (has_next_wave) begin
                                wave_q <= 1'b1;
                                state_q <= ST_RESID_CFG;
                            end else if (ingress_complete_q) begin
                                armed_q <= 1'b0;
                                state_q <= ST_DONE;
                            end else begin
                                state_q <= ST_INGEST;
                            end
                        end
                    end

                    ST_GATE_Q8_CFG: if (leaf_q8_cfg_ready) begin
                        gate_q8_issue_count_q <= 6'd0;
                        gate_q8_stage_valid_q <= 1'b0;
                        state_q <= ST_GATE_Q8_RUN;
                    end

                    ST_GATE_Q8_RUN: begin
                        if (gate_q8_out_fire) begin
                            if ((leaf_q8_out_block != 9'd0) ||
                                !leaf_q8_out_last ||
                                (leaf_q8_out_status != 8'd0)) begin
                                error_q <= 1'b1;
                                status_q <= status_q | STATUS_Q8;
                            end

                            if (has_next_wave) begin
                                wave_q <= 1'b1;
                                state_q <= ST_GATE_Q8_CFG;
                            end else begin
                                gate_tile_block_q <= gate_tile_block_q + 1'b1;
                                if (({6'd0, gate_tile_block_q} + 15'd1) ==
                                    (ffn_dim_q >> 5)) begin
                                    if (!ingress_complete_q) begin
                                        error_q <= 1'b1;
                                        status_q <= status_q |
                                                    STATUS_PROJ_FRAME;
                                    end
                                    armed_q <= 1'b0;
                                    state_q <= ST_DONE;
                                end else begin
                                    state_q <= ST_INGEST;
                                end
                            end
                        end
                    end

                    ST_DONE: if (done_valid && done_ready) begin
                        state_q <= ST_IDLE;
                        armed_q <= 1'b0;
                        error_q <= 1'b0;
                        status_q <= 16'd0;
                    end

                    ST_ABORT: begin
                        if (gamma_rsp_fire && gamma_rsp_last)
                            gamma_active_q <= 1'b0;
                        if (rope_rsp_fire && rope_rsp_last)
                            rope_active_q <= 1'b0;
                        if (!gamma_active_q && !rope_active_q &&
                            !gamma_rsp_valid && !rope_rsp_valid &&
                            !reduce_busy && !apply_busy && !resid_busy &&
                            !leaf_q8_busy && !swiglu_out_valid &&
                            !r_outstanding_q && !r_rd_rsp_valid &&
                            !head_mem_valid_q &&
                            !head_rsp_valid_q) begin
                            state_q <= ST_IDLE;
                            error_q <= 1'b0;
                            status_q <= 16'd0;
                        end
                    end

                    default: begin
                        error_q <= 1'b1;
                        status_q <= status_q | STATUS_INTERNAL;
                        armed_q <= 1'b0;
                        state_q <= ST_DONE;
                    end
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear && proj_fire && !proj_order_ok)
            $error(" projection_sink projection order mismatch");
        if (rst_n && !clear && proj_fire && !proj_frame_ok)
            $error(" projection_sink projection frame mismatch");
        if (rst_n && !clear && rope_pipe_valid && rope_hold_valid_q)
            $fatal(1, " projection_sink RoPE output collision");
        if (rst_n && !clear && query_wr_valid && newkv_wr_valid)
            $fatal(1, " projection_sink dual arena write");
        if (rst_n && !clear && cfg_fire &&
            ((cfg_hidden_dim >> 5) > 13'd128))
            $fatal(1, " projection_sink hidden Q8 partition overflow");
        if (rst_n && !clear && cfg_fire &&
            ((cfg_ffn_dim >> 5) > Q8_FFN_BLOCK_LIMIT))
            $fatal(1, " projection_sink FFN Q8 partition overflow");
        if (rst_n && !clear && q8_wr_valid &&
            ({1'b0, q8_wr_addr} >= 10'd512))
            $fatal(1, " projection_sink Q8 write outside arena");
    end
`endif
endmodule

`default_nettype wire
