// Controller-facing vector/projection cluster with one physical Q8 service.
// Commands are serialized because transformer layer stages are deterministic;
// vector and projection arithmetic never need to execute concurrently.

`default_nettype none

module vector_cluster (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,
    output wire          cluster_busy,
    output wire          protocol_error,

    input  wire          v_cmd_valid,
    output wire          v_cmd_ready,
    input  wire [3:0]    v_cmd_op,
    input  wire [7:0]    v_cmd_token_mask,
    input  wire [7:0]    v_cmd_hidden_blocks,
    input  wire [63:0]   v_cmd_gamma_addr,
    input  wire [31:0]   v_cmd_epsilon,
    output wire          v_done_valid,
    input  wire          v_done_ready,
    output wire          v_done_error,
    output wire [15:0]   v_done_status,
    output wire [31:0]   v_done_cycles,
    output wire          v_busy,
    output wire [3:0]    v_debug_state,
    output wire          v_gamma_req_valid,
    input  wire          v_gamma_req_ready,
    output wire [63:0]   v_gamma_req_addr,
    output wire [10:0]   v_gamma_req_words,
    input  wire          v_gamma_rsp_valid,
    output wire          v_gamma_rsp_ready,
    input  wire [127:0]  v_gamma_rsp_data,
    input  wire          v_gamma_rsp_last,
    input  wire          v_gamma_rsp_error,
    output wire          v_r_rd_req_valid,
    input  wire          v_r_rd_req_ready,
    output wire          v_r_rd_req_wave,
    output wire [11:0]   v_r_rd_req_addr,
    input  wire          v_r_rd_rsp_valid,
    output wire          v_r_rd_rsp_ready,
    input  wire [127:0]  v_r_rd_rsp_data,
    input  wire          v_r_rd_rsp_error,
    output wire          v_q8_wr_valid,
    input  wire          v_q8_wr_ready,
    output wire          v_q8_wr_wave,
    output wire [8:0]    v_q8_wr_addr,
    output wire [3:0]    v_q8_wr_lane_mask,
    output wire [1087:0] v_q8_wr_data,

    input  wire          s_cfg_valid,
    output wire          s_cfg_ready,
    input  wire [1:0]    s_cfg_mode,
    input  wire [7:0]    s_cfg_token_mask,
    input  wire [12:0]   s_cfg_hidden_dim,
    input  wire [14:0]   s_cfg_ffn_dim,
    input  wire [5:0]    s_cfg_q_heads,
    input  wire [3:0]    s_cfg_kv_heads,
    input  wire [16:0]   s_cfg_position_base,
    input  wire [31:0]   s_cfg_epsilon,
    input  wire [63:0]   s_cfg_q_gamma_addr,
    input  wire [63:0]   s_cfg_k_gamma_addr,
    input  wire [63:0]   s_cfg_rope_addr,
    output wire          s_projection_armed,
    output wire          s_busy,
    output wire          s_done_valid,
    input  wire          s_done_ready,
    output wire          s_done_error,
    output wire [15:0]   s_done_status,
    output wire [31:0]   s_done_cycles,
    output wire [4:0]    s_debug_state,
    output wire          s_gamma_req_valid,
    input  wire          s_gamma_req_ready,
    output wire [63:0]   s_gamma_req_addr,
    output wire [6:0]    s_gamma_req_words,
    input  wire          s_gamma_rsp_valid,
    output wire          s_gamma_rsp_ready,
    input  wire [127:0]  s_gamma_rsp_data,
    input  wire          s_gamma_rsp_last,
    input  wire          s_gamma_rsp_error,
    output wire          s_rope_req_valid,
    input  wire          s_rope_req_ready,
    output wire [63:0]   s_rope_req_addr,
    output wire [16:0]   s_rope_req_position_base,
    output wire [7:0]    s_rope_req_token_mask,
    output wire [7:0]    s_rope_req_records,
    input  wire          s_rope_rsp_valid,
    output wire          s_rope_rsp_ready,
    input  wire [255:0]  s_rope_rsp_data,
    input  wire          s_rope_rsp_last,
    input  wire          s_rope_rsp_error,
    input  wire          s_proj_valid,
    output wire          s_proj_ready,
    input  wire [2:0]    s_proj_token,
    input  wire [17:0]   s_proj_row,
    input  wire [31:0]   s_proj_data_f32,
    input  wire          s_proj_last,
    output wire          s_query_wr_valid,
    input  wire          s_query_wr_ready,
    output wire          s_query_wr_wave,
    output wire [11:0]   s_query_wr_addr,
    output wire [3:0]    s_query_wr_lane_mask,
    output wire [127:0]  s_query_wr_data,
    output wire          s_newkv_wr_valid,
    input  wire          s_newkv_wr_ready,
    output wire          s_newkv_wr_wave,
    output wire [10:0]   s_newkv_wr_addr,
    output wire [3:0]    s_newkv_wr_lane_mask,
    output wire [63:0]   s_newkv_wr_data,
    output wire          s_q8_wr_valid,
    input  wire          s_q8_wr_ready,
    output wire          s_q8_wr_wave,
    output wire [8:0]    s_q8_wr_addr,
    output wire [3:0]    s_q8_wr_lane_mask,
    output wire [1087:0] s_q8_wr_data,
    output wire          s_r_rd_req_valid,
    input  wire          s_r_rd_req_ready,
    output wire          s_r_rd_req_wave,
    output wire [11:0]   s_r_rd_req_addr,
    input  wire          s_r_rd_rsp_valid,
    output wire          s_r_rd_rsp_ready,
    input  wire [127:0]  s_r_rd_rsp_data,
    input  wire          s_r_rd_rsp_error,
    output wire          s_r_wr_valid,
    input  wire          s_r_wr_ready,
    output wire          s_r_wr_wave,
    output wire [11:0]   s_r_wr_addr,
    output wire [3:0]    s_r_wr_lane_mask,
    output wire [127:0]  s_r_wr_data,

    // Attention-output transpose is the third client of the same Q8 leaf.
    input  wire          a_leaf_q8_cfg_valid,
    output wire          a_leaf_q8_cfg_ready,
    input  wire [14:0]   a_leaf_q8_cfg_rows,
    input  wire [3:0]    a_leaf_q8_cfg_lane_mask,
    input  wire          a_leaf_q8_abort,
    output wire          a_leaf_q8_busy,
    input  wire          a_leaf_q8_in_valid,
    output wire          a_leaf_q8_in_ready,
    input  wire [127:0]  a_leaf_q8_in_data,
    output wire          a_leaf_q8_out_valid,
    input  wire          a_leaf_q8_out_ready,
    output wire [8:0]    a_leaf_q8_out_block,
    output wire [1087:0] a_leaf_q8_out_data,
    output wire [7:0]    a_leaf_q8_out_status,
    output wire          a_leaf_q8_out_last
);
    localparam [1:0] OWNER_NONE = 2'd0;
    localparam [1:0] OWNER_VEC  = 2'd1;
    localparam [1:0] OWNER_SINK = 2'd2;

    reg [1:0] command_owner_q;
    reg command_collision_q;
    (* keep = "true", dont_touch = "true" *) reg clear_vector_q;
    (* keep = "true", dont_touch = "true" *) reg clear_sink_q;
    (* keep = "true", dont_touch = "true" *) reg clear_q8_q;
    wire local_clear_active = clear || clear_vector_q || clear_sink_q ||
                              clear_q8_q;
    wire vector_cmd_select = !local_clear_active &&
                             (command_owner_q == OWNER_NONE) && v_cmd_valid;
    wire sink_cmd_select = !local_clear_active &&
                           (command_owner_q == OWNER_NONE) && !v_cmd_valid &&
                           s_cfg_valid;
    wire vector_cmd_ready_core;
    wire sink_cfg_ready_core;
    wire vector_cmd_fire = vector_cmd_select && vector_cmd_ready_core;
    wire sink_cfg_fire = sink_cmd_select && sink_cfg_ready_core;
    assign v_cmd_ready = vector_cmd_select && vector_cmd_ready_core;
    assign s_cfg_ready = sink_cmd_select && sink_cfg_ready_core;

    wire q8_busy;
    wire q8_collision;
    assign cluster_busy = (command_owner_q != OWNER_NONE) || v_busy ||
                          s_busy || q8_busy;
    assign protocol_error = command_collision_q || q8_collision;
    assign a_leaf_q8_busy = q8_busy;

    // One registered clear copy per physical child keeps the top-level pulse
    // out of service state/BRAM enable trees. clear_done observes child busy,
    // so the extra cycle remains fail-closed.
    always @(posedge clk) begin
        if (!rst_n) begin
            clear_vector_q <= 1'b0;
            clear_sink_q <= 1'b0;
            clear_q8_q <= 1'b0;
        end else begin
            clear_vector_q <= clear;
            clear_sink_q <= clear;
            clear_q8_q <= clear;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear || abort_run) begin
            command_owner_q <= OWNER_NONE;
            command_collision_q <= 1'b0;
        end else begin
            if ((command_owner_q == OWNER_NONE) && v_cmd_valid && s_cfg_valid)
                command_collision_q <= 1'b1;
            if (vector_cmd_fire)
                command_owner_q <= OWNER_VEC;
            else if (sink_cfg_fire)
                command_owner_q <= OWNER_SINK;
            if ((command_owner_q == OWNER_VEC) && v_done_valid &&
                v_done_ready)
                command_owner_q <= OWNER_NONE;
            else if ((command_owner_q == OWNER_SINK) && s_done_valid &&
                     s_done_ready)
                command_owner_q <= OWNER_NONE;
        end
    end

    wire v_leaf_cfg_valid;
    wire v_leaf_cfg_ready;
    wire [14:0] v_leaf_cfg_rows;
    wire [3:0] v_leaf_cfg_lane_mask;
    wire v_leaf_in_valid;
    wire v_leaf_in_ready;
    wire [127:0] v_leaf_in_data;
    wire v_leaf_out_valid;
    wire v_leaf_out_ready;
    wire [8:0] v_leaf_out_block;
    wire [1087:0] v_leaf_out_data;
    wire [7:0] v_leaf_out_status;
    wire v_leaf_out_last;

    (* keep_hierarchy = "yes" *)  vector_service u_vector (
        .clk(clk), .rst_n(rst_n), .clear(clear_vector_q),
        .cmd_valid(vector_cmd_select), .cmd_ready(vector_cmd_ready_core),
        .cmd_op(v_cmd_op), .cmd_token_mask(v_cmd_token_mask),
        .cmd_hidden_blocks(v_cmd_hidden_blocks),
        .cmd_gamma_addr(v_cmd_gamma_addr), .cmd_epsilon(v_cmd_epsilon),
        .abort_run(abort_run), .done_valid(v_done_valid),
        .done_ready(v_done_ready), .done_error(v_done_error),
        .done_status(v_done_status), .done_cycles(v_done_cycles),
        .busy(v_busy), .debug_state(v_debug_state),
        .gamma_req_valid(v_gamma_req_valid),
        .gamma_req_ready(v_gamma_req_ready),
        .gamma_req_addr(v_gamma_req_addr),
        .gamma_req_words(v_gamma_req_words),
        .gamma_rsp_valid(v_gamma_rsp_valid),
        .gamma_rsp_ready(v_gamma_rsp_ready),
        .gamma_rsp_data(v_gamma_rsp_data),
        .gamma_rsp_last(v_gamma_rsp_last),
        .gamma_rsp_error(v_gamma_rsp_error),
        .r_rd_req_valid(v_r_rd_req_valid),
        .r_rd_req_ready(v_r_rd_req_ready), .r_rd_req_wave(v_r_rd_req_wave),
        .r_rd_req_addr(v_r_rd_req_addr), .r_rd_rsp_valid(v_r_rd_rsp_valid),
        .r_rd_rsp_ready(v_r_rd_rsp_ready), .r_rd_rsp_data(v_r_rd_rsp_data),
        .r_rd_rsp_error(v_r_rd_rsp_error), .q8_wr_valid(v_q8_wr_valid),
        .q8_wr_ready(v_q8_wr_ready), .q8_wr_wave(v_q8_wr_wave),
        .q8_wr_addr(v_q8_wr_addr), .q8_wr_lane_mask(v_q8_wr_lane_mask),
        .q8_wr_data(v_q8_wr_data), .leaf_q8_cfg_valid(v_leaf_cfg_valid),
        .leaf_q8_cfg_ready(v_leaf_cfg_ready),
        .leaf_q8_cfg_rows(v_leaf_cfg_rows),
        .leaf_q8_cfg_lane_mask(v_leaf_cfg_lane_mask),
        .leaf_q8_busy(q8_busy), .leaf_q8_in_valid(v_leaf_in_valid),
        .leaf_q8_in_ready(v_leaf_in_ready), .leaf_q8_in_data(v_leaf_in_data),
        .leaf_q8_out_valid(v_leaf_out_valid),
        .leaf_q8_out_ready(v_leaf_out_ready),
        .leaf_q8_out_block(v_leaf_out_block),
        .leaf_q8_out_data(v_leaf_out_data),
        .leaf_q8_out_status(v_leaf_out_status),
        .leaf_q8_out_last(v_leaf_out_last)
    );

    wire s_leaf_cfg_valid;
    wire s_leaf_cfg_ready;
    wire [14:0] s_leaf_cfg_rows;
    wire [3:0] s_leaf_cfg_lane_mask;
    wire s_leaf_in_valid;
    wire s_leaf_in_ready;
    wire [127:0] s_leaf_in_data;
    wire s_leaf_out_valid;
    wire s_leaf_out_ready;
    wire [8:0] s_leaf_out_block;
    wire [1087:0] s_leaf_out_data;
    wire [7:0] s_leaf_out_status;
    wire s_leaf_out_last;

    (* keep_hierarchy = "yes" *)  projection_sink u_sink (
        .clk(clk), .rst_n(rst_n), .clear(clear_sink_q),
        .cfg_valid(sink_cmd_select),
        .cfg_ready(sink_cfg_ready_core), .cfg_mode(s_cfg_mode),
        .cfg_token_mask(s_cfg_token_mask),
        .cfg_hidden_dim(s_cfg_hidden_dim), .cfg_ffn_dim(s_cfg_ffn_dim),
        .cfg_q_heads(s_cfg_q_heads), .cfg_kv_heads(s_cfg_kv_heads),
        .cfg_position_base(s_cfg_position_base),
        .cfg_epsilon(s_cfg_epsilon),
        .cfg_q_gamma_addr(s_cfg_q_gamma_addr),
        .cfg_k_gamma_addr(s_cfg_k_gamma_addr),
        .cfg_rope_addr(s_cfg_rope_addr), .abort_run(abort_run),
        .projection_armed(s_projection_armed), .busy(s_busy),
        .done_valid(s_done_valid), .done_ready(s_done_ready),
        .done_error(s_done_error), .done_status(s_done_status),
        .done_cycles(s_done_cycles), .debug_state(s_debug_state),
        .gamma_req_valid(s_gamma_req_valid),
        .gamma_req_ready(s_gamma_req_ready),
        .gamma_req_addr(s_gamma_req_addr),
        .gamma_req_words(s_gamma_req_words),
        .gamma_rsp_valid(s_gamma_rsp_valid),
        .gamma_rsp_ready(s_gamma_rsp_ready),
        .gamma_rsp_data(s_gamma_rsp_data),
        .gamma_rsp_last(s_gamma_rsp_last),
        .gamma_rsp_error(s_gamma_rsp_error),
        .rope_req_valid(s_rope_req_valid), .rope_req_ready(s_rope_req_ready),
        .rope_req_addr(s_rope_req_addr),
        .rope_req_position_base(s_rope_req_position_base),
        .rope_req_token_mask(s_rope_req_token_mask),
        .rope_req_records(s_rope_req_records),
        .rope_rsp_valid(s_rope_rsp_valid),
        .rope_rsp_ready(s_rope_rsp_ready), .rope_rsp_data(s_rope_rsp_data),
        .rope_rsp_last(s_rope_rsp_last),
        .rope_rsp_error(s_rope_rsp_error), .proj_valid(s_proj_valid),
        .proj_ready(s_proj_ready), .proj_token(s_proj_token),
        .proj_row(s_proj_row), .proj_data_f32(s_proj_data_f32),
        .proj_last(s_proj_last), .query_wr_valid(s_query_wr_valid),
        .query_wr_ready(s_query_wr_ready), .query_wr_wave(s_query_wr_wave),
        .query_wr_addr(s_query_wr_addr),
        .query_wr_lane_mask(s_query_wr_lane_mask),
        .query_wr_data(s_query_wr_data), .newkv_wr_valid(s_newkv_wr_valid),
        .newkv_wr_ready(s_newkv_wr_ready), .newkv_wr_wave(s_newkv_wr_wave),
        .newkv_wr_addr(s_newkv_wr_addr),
        .newkv_wr_lane_mask(s_newkv_wr_lane_mask),
        .newkv_wr_data(s_newkv_wr_data), .q8_wr_valid(s_q8_wr_valid),
        .q8_wr_ready(s_q8_wr_ready), .q8_wr_wave(s_q8_wr_wave),
        .q8_wr_addr(s_q8_wr_addr), .q8_wr_lane_mask(s_q8_wr_lane_mask),
        .q8_wr_data(s_q8_wr_data), .leaf_q8_cfg_valid(s_leaf_cfg_valid),
        .leaf_q8_cfg_ready(s_leaf_cfg_ready),
        .leaf_q8_cfg_rows(s_leaf_cfg_rows),
        .leaf_q8_cfg_lane_mask(s_leaf_cfg_lane_mask),
        .leaf_q8_busy(q8_busy), .leaf_q8_in_valid(s_leaf_in_valid),
        .leaf_q8_in_ready(s_leaf_in_ready), .leaf_q8_in_data(s_leaf_in_data),
        .leaf_q8_out_valid(s_leaf_out_valid),
        .leaf_q8_out_ready(s_leaf_out_ready),
        .leaf_q8_out_block(s_leaf_out_block),
        .leaf_q8_out_data(s_leaf_out_data),
        .leaf_q8_out_status(s_leaf_out_status),
        .leaf_q8_out_last(s_leaf_out_last),
        .r_rd_req_valid(s_r_rd_req_valid),
        .r_rd_req_ready(s_r_rd_req_ready), .r_rd_req_wave(s_r_rd_req_wave),
        .r_rd_req_addr(s_r_rd_req_addr), .r_rd_rsp_valid(s_r_rd_rsp_valid),
        .r_rd_rsp_ready(s_r_rd_rsp_ready), .r_rd_rsp_data(s_r_rd_rsp_data),
        .r_rd_rsp_error(s_r_rd_rsp_error), .r_wr_valid(s_r_wr_valid),
        .r_wr_ready(s_r_wr_ready), .r_wr_wave(s_r_wr_wave),
        .r_wr_addr(s_r_wr_addr), .r_wr_lane_mask(s_r_wr_lane_mask),
        .r_wr_data(s_r_wr_data)
    );

    (* keep_hierarchy = "yes" *)  shared_q8 u_shared_q8 (
        .clk(clk), .rst_n(rst_n), .abort_run(clear_q8_q || abort_run),
        .busy(q8_busy),
        .collision_error(q8_collision), .c0_cfg_valid(v_leaf_cfg_valid),
        .c0_cfg_ready(v_leaf_cfg_ready), .c0_cfg_rows(v_leaf_cfg_rows),
        .c0_cfg_lane_mask(v_leaf_cfg_lane_mask),
        .c0_in_valid(v_leaf_in_valid), .c0_in_ready(v_leaf_in_ready),
        .c0_in_data(v_leaf_in_data), .c0_out_valid(v_leaf_out_valid),
        .c0_out_ready(v_leaf_out_ready), .c0_out_block(v_leaf_out_block),
        .c0_out_data(v_leaf_out_data), .c0_out_status(v_leaf_out_status),
        .c0_out_last(v_leaf_out_last), .c0_abort(1'b0),
        .c1_cfg_valid(s_leaf_cfg_valid),
        .c1_cfg_ready(s_leaf_cfg_ready), .c1_cfg_rows(s_leaf_cfg_rows),
        .c1_cfg_lane_mask(s_leaf_cfg_lane_mask),
        .c1_in_valid(s_leaf_in_valid), .c1_in_ready(s_leaf_in_ready),
        .c1_in_data(s_leaf_in_data), .c1_out_valid(s_leaf_out_valid),
        .c1_out_ready(s_leaf_out_ready), .c1_out_block(s_leaf_out_block),
        .c1_out_data(s_leaf_out_data), .c1_out_status(s_leaf_out_status),
        .c1_out_last(s_leaf_out_last), .c1_abort(1'b0),
        .c2_cfg_valid(a_leaf_q8_cfg_valid),
        .c2_cfg_ready(a_leaf_q8_cfg_ready),
        .c2_cfg_rows(a_leaf_q8_cfg_rows),
        .c2_cfg_lane_mask(a_leaf_q8_cfg_lane_mask),
        .c2_in_valid(a_leaf_q8_in_valid),
        .c2_in_ready(a_leaf_q8_in_ready), .c2_in_data(a_leaf_q8_in_data),
        .c2_out_valid(a_leaf_q8_out_valid),
        .c2_out_ready(a_leaf_q8_out_ready),
        .c2_out_block(a_leaf_q8_out_block),
        .c2_out_data(a_leaf_q8_out_data),
        .c2_out_status(a_leaf_q8_out_status),
        .c2_out_last(a_leaf_q8_out_last), .c2_abort(a_leaf_q8_abort)
    );
endmodule

`default_nettype wire
