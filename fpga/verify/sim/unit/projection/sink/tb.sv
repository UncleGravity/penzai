`timescale 1ns/1ps
`default_nettype none

module projection_sink_tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    integer cycle = 0;
    always @(posedge clk) cycle <= cycle + 1;

    reg cfg_valid = 1'b0;
    wire cfg_ready;
    reg [1:0] cfg_mode = 2'd0;
    reg [7:0] cfg_token_mask = 8'd0;
    reg [12:0] cfg_hidden_dim = 13'd2048;
    reg [14:0] cfg_ffn_dim = 15'd6144;
    reg [5:0] cfg_q_heads = 6'd16;
    reg [3:0] cfg_kv_heads = 4'd8;
    reg [16:0] cfg_position_base = 17'd0;
    reg [31:0] cfg_epsilon = 32'h3727_c5ac;
    reg [63:0] cfg_q_gamma_addr = 64'h1000;
    reg [63:0] cfg_k_gamma_addr = 64'h2000;
    reg [63:0] cfg_rope_addr = 64'h3000;
    reg abort_run = 1'b0;

    wire projection_armed;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [15:0] done_status;
    wire [31:0] done_cycles;
    wire [4:0] debug_state;

    wire gamma_req_valid;
    wire gamma_req_ready;
    wire [63:0] gamma_req_addr;
    wire [6:0] gamma_req_words;
    wire gamma_rsp_valid;
    wire gamma_rsp_ready;
    wire [127:0] gamma_rsp_data;
    wire gamma_rsp_last;
    wire gamma_rsp_error = 1'b0;

    wire rope_req_valid;
    wire rope_req_ready;
    wire [63:0] rope_req_addr;
    wire [16:0] rope_req_position_base;
    wire [7:0] rope_req_token_mask;
    wire [7:0] rope_req_records;
    wire rope_rsp_valid;
    wire rope_rsp_ready;
    wire [255:0] rope_rsp_data;
    wire rope_rsp_last;
    wire rope_rsp_error = 1'b0;

    reg proj_valid = 1'b0;
    wire proj_ready;
    reg [2:0] proj_token = 3'd0;
    reg [17:0] proj_row = 18'd0;
    reg [31:0] proj_data_f32 = 32'd0;
    reg proj_last = 1'b0;

    wire query_wr_valid;
    wire query_wr_ready;
    wire query_wr_wave;
    wire [11:0] query_wr_addr;
    wire [3:0] query_wr_lane_mask;
    wire [127:0] query_wr_data;
    wire newkv_wr_valid;
    wire newkv_wr_ready;
    wire newkv_wr_wave;
    wire [10:0] newkv_wr_addr;
    wire [3:0] newkv_wr_lane_mask;
    wire [63:0] newkv_wr_data;
    wire q8_wr_valid;
    wire q8_wr_ready;
    wire q8_wr_wave;
    wire [8:0] q8_wr_addr;
    wire [3:0] q8_wr_lane_mask;
    wire [1087:0] q8_wr_data;
    wire leaf_q8_cfg_valid;
    wire leaf_q8_cfg_ready;
    wire [14:0] leaf_q8_cfg_rows;
    wire [3:0] leaf_q8_cfg_lane_mask;
    wire leaf_q8_busy;
    wire leaf_q8_in_valid;
    wire leaf_q8_in_ready;
    wire [127:0] leaf_q8_in_data;
    wire leaf_q8_out_valid;
    wire leaf_q8_out_ready;
    wire [8:0] leaf_q8_out_block;
    wire [1087:0] leaf_q8_out_data;
    wire [7:0] leaf_q8_out_status;
    wire leaf_q8_out_last;
    wire q8_collision_error;
    wire q8_dummy_cfg_ready;
    wire q8_dummy_in_ready;
    wire q8_dummy_out_valid;
    wire [8:0] q8_dummy_out_block;
    wire [1087:0] q8_dummy_out_data;
    wire [7:0] q8_dummy_out_status;
    wire q8_dummy_out_last;
    wire r_rd_req_valid;
    reg r_rd_req_ready = 1'b0;
    wire r_rd_req_wave;
    wire [11:0] r_rd_req_addr;
    reg r_rd_rsp_valid = 1'b0;
    wire r_rd_rsp_ready;
    reg [127:0] r_rd_rsp_data = 128'd0;
    reg r_rd_rsp_error = 1'b0;
    wire r_wr_valid;
    reg r_wr_ready = 1'b1;
    wire r_wr_wave;
    wire [11:0] r_wr_addr;
    wire [3:0] r_wr_lane_mask;
    wire [127:0] r_wr_data;

    reg neox_test_mode = 1'b0;
    reg [31:0] neox_q_norm [0:127];
    reg [31:0] neox_k_norm [0:127];
    reg [127:0] neox_q_seen = 128'd0;
    reg [127:0] neox_k_seen = 128'd0;
    integer neox_q_norm_count = 0;
    integer neox_k_norm_count = 0;
    integer neox_q_write_count = 0;
    integer neox_k_write_count = 0;
    reg head_stalled_q = 1'b0;
    reg [127:0] stalled_head_data_q = 128'd0;
    reg [6:0] stalled_head_dim_q = 7'd0;
    integer head_stall_count = 0;

    function automatic [31:0] neox_gamma(
        input kind,
        input [6:0] dimension
    );
        // Every one of the 128 dimensions is exactly representable and unique.
        // K uses the next exponent so its numerical path is independently tagged.
        neox_gamma = {1'b0, kind ? 8'h80 : 8'h7f, dimension, 16'd0};
    endfunction

    function automatic [6:0] neox_count_dim(input [7:0] count);
        neox_count_dim = {count[0], count[6:1]};
    endfunction

    wire [6:0] neox_apply_dim = {
        dut.apply_output_count_q[0], dut.apply_output_count_q[6:1]
    };
    wire [6:0] neox_query_dim = query_wr_addr[6:0];
    wire [6:0] neox_query_source_dim = {
        ~neox_query_dim[6], neox_query_dim[5:0]
    };
    wire [31:0] neox_query_source = neox_q_norm[neox_query_source_dim];
    wire [31:0] neox_query_expected = neox_query_dim[6] ?
        neox_query_source : {~neox_query_source[31], neox_query_source[30:0]};
    wire [6:0] neox_k_dim = newkv_wr_addr[6:0];
    wire [6:0] neox_k_source_dim = {~neox_k_dim[6], neox_k_dim[5:0]};
    wire [31:0] neox_k_source = neox_k_norm[neox_k_source_dim];
    wire [31:0] neox_k_expected_f32 = neox_k_dim[6] ?
        neox_k_source : {~neox_k_source[31], neox_k_source[30:0]};
    wire [15:0] neox_k_expected_f16;

     f32_to_f16 u_neox_k_reference (
        .in(neox_k_expected_f32), .out(neox_k_expected_f16)
    );

     projection_sink dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_mode(cfg_mode), .cfg_token_mask(cfg_token_mask),
        .cfg_hidden_dim(cfg_hidden_dim), .cfg_ffn_dim(cfg_ffn_dim),
        .cfg_q_heads(cfg_q_heads), .cfg_kv_heads(cfg_kv_heads),
        .cfg_position_base(cfg_position_base), .cfg_epsilon(cfg_epsilon),
        .cfg_q_gamma_addr(cfg_q_gamma_addr),
        .cfg_k_gamma_addr(cfg_k_gamma_addr), .cfg_rope_addr(cfg_rope_addr),
        .abort_run(abort_run), .projection_armed(projection_armed),
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status),
        .done_cycles(done_cycles), .debug_state(debug_state),
        .gamma_req_valid(gamma_req_valid),
        .gamma_req_ready(gamma_req_ready), .gamma_req_addr(gamma_req_addr),
        .gamma_req_words(gamma_req_words),
        .gamma_rsp_valid(gamma_rsp_valid),
        .gamma_rsp_ready(gamma_rsp_ready), .gamma_rsp_data(gamma_rsp_data),
        .gamma_rsp_last(gamma_rsp_last), .gamma_rsp_error(gamma_rsp_error),
        .rope_req_valid(rope_req_valid), .rope_req_ready(rope_req_ready),
        .rope_req_addr(rope_req_addr),
        .rope_req_position_base(rope_req_position_base),
        .rope_req_token_mask(rope_req_token_mask),
        .rope_req_records(rope_req_records),
        .rope_rsp_valid(rope_rsp_valid), .rope_rsp_ready(rope_rsp_ready),
        .rope_rsp_data(rope_rsp_data), .rope_rsp_last(rope_rsp_last),
        .rope_rsp_error(rope_rsp_error),
        .proj_valid(proj_valid), .proj_ready(proj_ready),
        .proj_token(proj_token), .proj_row(proj_row),
        .proj_data_f32(proj_data_f32), .proj_last(proj_last),
        .query_wr_valid(query_wr_valid), .query_wr_ready(query_wr_ready),
        .query_wr_wave(query_wr_wave), .query_wr_addr(query_wr_addr),
        .query_wr_lane_mask(query_wr_lane_mask),
        .query_wr_data(query_wr_data),
        .newkv_wr_valid(newkv_wr_valid), .newkv_wr_ready(newkv_wr_ready),
        .newkv_wr_wave(newkv_wr_wave), .newkv_wr_addr(newkv_wr_addr),
        .newkv_wr_lane_mask(newkv_wr_lane_mask),
        .newkv_wr_data(newkv_wr_data),
        .q8_wr_valid(q8_wr_valid), .q8_wr_ready(q8_wr_ready),
        .q8_wr_wave(q8_wr_wave), .q8_wr_addr(q8_wr_addr),
        .q8_wr_lane_mask(q8_wr_lane_mask), .q8_wr_data(q8_wr_data),
        .leaf_q8_cfg_valid(leaf_q8_cfg_valid),
        .leaf_q8_cfg_ready(leaf_q8_cfg_ready),
        .leaf_q8_cfg_rows(leaf_q8_cfg_rows),
        .leaf_q8_cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .leaf_q8_busy(leaf_q8_busy),
        .leaf_q8_in_valid(leaf_q8_in_valid),
        .leaf_q8_in_ready(leaf_q8_in_ready),
        .leaf_q8_in_data(leaf_q8_in_data),
        .leaf_q8_out_valid(leaf_q8_out_valid),
        .leaf_q8_out_ready(leaf_q8_out_ready),
        .leaf_q8_out_block(leaf_q8_out_block),
        .leaf_q8_out_data(leaf_q8_out_data),
        .leaf_q8_out_status(leaf_q8_out_status),
        .leaf_q8_out_last(leaf_q8_out_last),
        .r_rd_req_valid(r_rd_req_valid),
        .r_rd_req_ready(r_rd_req_ready), .r_rd_req_wave(r_rd_req_wave),
        .r_rd_req_addr(r_rd_req_addr), .r_rd_rsp_valid(r_rd_rsp_valid),
        .r_rd_rsp_ready(r_rd_rsp_ready), .r_rd_rsp_data(r_rd_rsp_data),
        .r_rd_rsp_error(r_rd_rsp_error), .r_wr_valid(r_wr_valid),
        .r_wr_ready(r_wr_ready), .r_wr_wave(r_wr_wave),
        .r_wr_addr(r_wr_addr), .r_wr_lane_mask(r_wr_lane_mask),
        .r_wr_data(r_wr_data)
    );

     shared_q8 q8_service (
        .clk(clk), .rst_n(rst_n), .abort_run(clear || abort_run),
        .busy(leaf_q8_busy), .collision_error(q8_collision_error),
        .c0_cfg_valid(1'b0), .c0_cfg_ready(q8_dummy_cfg_ready),
        .c0_cfg_rows(15'd0), .c0_cfg_lane_mask(4'd0),
        .c0_in_valid(1'b0), .c0_in_ready(q8_dummy_in_ready),
        .c0_in_data(128'd0), .c0_out_valid(q8_dummy_out_valid),
        .c0_out_ready(1'b0), .c0_out_block(q8_dummy_out_block),
        .c0_out_data(q8_dummy_out_data),
        .c0_out_status(q8_dummy_out_status),
        .c0_out_last(q8_dummy_out_last), .c0_abort(1'b0),
        .c1_cfg_valid(leaf_q8_cfg_valid),
        .c1_cfg_ready(leaf_q8_cfg_ready),
        .c1_cfg_rows(leaf_q8_cfg_rows),
        .c1_cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .c1_in_valid(leaf_q8_in_valid), .c1_in_ready(leaf_q8_in_ready),
        .c1_in_data(leaf_q8_in_data), .c1_out_valid(leaf_q8_out_valid),
        .c1_out_ready(leaf_q8_out_ready),
        .c1_out_block(leaf_q8_out_block), .c1_out_data(leaf_q8_out_data),
        .c1_out_status(leaf_q8_out_status), .c1_out_last(leaf_q8_out_last),
        .c1_abort(1'b0), .c2_cfg_valid(1'b0), .c2_cfg_ready(),
        .c2_cfg_rows(15'd0), .c2_cfg_lane_mask(4'd0),
        .c2_in_valid(1'b0), .c2_in_ready(), .c2_in_data(128'd0),
        .c2_out_valid(), .c2_out_ready(1'b0), .c2_out_block(),
        .c2_out_data(), .c2_out_status(), .c2_out_last(), .c2_abort(1'b0)
    );

    reg gamma_source_active = 1'b0;
    reg [5:0] gamma_source_index = 6'd0;
    reg gamma_source_kind = 1'b0;
    integer gamma_request_count = 0;
    assign gamma_req_ready = !gamma_source_active && (cycle[2:0] != 3'd3);
    assign gamma_rsp_valid = gamma_source_active && (cycle[1:0] != 2'd2);
    wire [6:0] gamma_source_dim0 = {gamma_source_index[4:0], 2'b00};
    assign gamma_rsp_data = neox_test_mode ? {
        neox_gamma(gamma_source_kind, gamma_source_dim0 + 7'd3),
        neox_gamma(gamma_source_kind, gamma_source_dim0 + 7'd2),
        neox_gamma(gamma_source_kind, gamma_source_dim0 + 7'd1),
        neox_gamma(gamma_source_kind, gamma_source_dim0)
    } : {4{32'h3f80_0000}};
    assign gamma_rsp_last = gamma_source_index == 6'd31;

    reg rope_source_active = 1'b0;
    reg [7:0] rope_source_index = 8'd0;
    reg [7:0] rope_source_records = 8'd0;
    integer rope_request_count = 0;
    assign rope_req_ready = !rope_source_active && (cycle[2:0] != 3'd5);
    assign rope_rsp_valid = rope_source_active && (cycle[2:0] != 3'd6);
    // The directed NEOX test uses cos=0,sin=1, making every expected rotated
    // value an exact signed permutation rather than a tolerance-only check.
    assign rope_rsp_data = neox_test_mode ?
        {4{{32'h3f80_0000, 32'h0000_0000}}} :
        {4{{32'h0000_0000, 32'h3f80_0000}}};
    assign rope_rsp_last =
        (rope_source_index + 1'b1) == rope_source_records;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            gamma_source_active <= 1'b0;
            gamma_source_index <= 6'd0;
            gamma_source_kind <= 1'b0;
            rope_source_active <= 1'b0;
            rope_source_index <= 8'd0;
            rope_source_records <= 8'd0;
        end else begin
            if (gamma_req_valid && gamma_req_ready) begin
                if (gamma_req_words != 7'd32 ||
                    ((gamma_req_addr != 64'h1000) &&
                     (gamma_req_addr != 64'h2000)))
                    $fatal(1, "bad gamma request");
                gamma_source_active <= 1'b1;
                gamma_source_index <= 6'd0;
                gamma_source_kind <= gamma_req_addr == 64'h2000;
                gamma_request_count <= gamma_request_count + 1;
            end
            if (gamma_rsp_valid && gamma_rsp_ready) begin
                if (gamma_rsp_last)
                    gamma_source_active <= 1'b0;
                else
                    gamma_source_index <= gamma_source_index + 1'b1;
            end

            if (rope_req_valid && rope_req_ready) begin
                if ((rope_req_addr != 64'h3000) ||
                    (rope_req_token_mask != cfg_token_mask) ||
                    (rope_req_position_base != cfg_position_base) ||
                    ((rope_req_records != 8'd64) &&
                     (rope_req_records != 8'd128)))
                    $fatal(1, "bad RoPE request");
                rope_source_active <= 1'b1;
                rope_source_index <= 8'd0;
                rope_source_records <= rope_req_records;
                rope_request_count <= rope_request_count + 1;
            end
            if (rope_rsp_valid && rope_rsp_ready) begin
                if (rope_rsp_last)
                    rope_source_active <= 1'b0;
                else
                    rope_source_index <= rope_source_index + 1'b1;
            end
        end
    end

    assign query_wr_ready = cycle[2:0] != 3'd4;
    assign newkv_wr_ready = cycle[2:0] != 3'd2;
    assign q8_wr_ready = cycle[2:0] != 3'd6;
    integer query_write_count = 0;
    integer k_write_count = 0;
    integer v_write_count = 0;
    integer proj_record_count = 0;
    integer r_request_count = 0;
    integer r_write_count = 0;
    integer q8_write_count = 0;
    reg [8:0] first_q8_write_addr = 9'h1ff;
    reg [31:0] stream_value = 32'h3f80_0000;
    reg inject_projection_nonfinite = 1'b0;
    reg injected_head_bad_seen = 1'b0;
    reg injected_head_tail_seen = 1'b0;
    integer injected_head_outstanding = 0;
    integer lane;
    integer quant;
    always @(posedge clk) begin
        if (rst_n && proj_valid && proj_ready)
            proj_record_count <= proj_record_count + 1;

        if (!rst_n || clear) begin
            r_rd_req_ready <= 1'b0;
            r_rd_rsp_valid <= 1'b0;
            r_rd_rsp_data <= 128'd0;
            r_rd_rsp_error <= 1'b0;
            r_wr_ready <= 1'b0;
        end else begin
            r_rd_req_ready <= (cycle[2:0] != 3'd3) &&
                              (!r_rd_rsp_valid || r_rd_rsp_ready);
            r_wr_ready <= cycle[2:0] != 3'd5;
            if (r_rd_rsp_valid && r_rd_rsp_ready)
                r_rd_rsp_valid <= 1'b0;
            if (r_rd_req_valid && r_rd_req_ready) begin
                r_rd_rsp_valid <= 1'b1;
                r_rd_rsp_data <= {4{32'h3f80_0000}};
                r_rd_rsp_error <= 1'b0;
                r_request_count <= r_request_count + 1;
            end
        end

        if (rst_n && query_wr_valid && query_wr_ready) begin
            query_write_count <= query_write_count + 1;
            if ((query_wr_lane_mask != (query_wr_wave ?
                    cfg_token_mask[7:4] : cfg_token_mask[3:0])) ||
                ((cfg_q_heads == 6'd16) &&
                 (query_wr_addr >= 12'd2048)))
                $fatal(1, "Query geometry mismatch");
            for (lane = 0; lane < 4; lane = lane + 1)
                if (query_wr_lane_mask[lane] &&
                    ((query_wr_data[lane*32 + 30 -: 8] == 8'hff) ||
                     (query_wr_data[lane*32 +: 31] == 31'd0)))
                    $fatal(1, "Query output nonfinite/zero");
        end

        if (rst_n && newkv_wr_valid && newkv_wr_ready) begin
            if (newkv_wr_addr[10])
                v_write_count <= v_write_count + 1;
            else
                k_write_count <= k_write_count + 1;
            if (newkv_wr_lane_mask != (newkv_wr_wave ?
                    cfg_token_mask[7:4] : cfg_token_mask[3:0]))
                $fatal(1, "NewKV geometry mismatch");
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (newkv_wr_lane_mask[lane] && !neox_test_mode &&
                    (newkv_wr_data[lane*16 +: 16] != 16'h3c00))
                    $fatal(1, "NewKV was not IEEE f16 one");
                if (newkv_wr_lane_mask[lane] && neox_test_mode &&
                    newkv_wr_addr[10] &&
                    (newkv_wr_data[lane*16 +: 16] != 16'h3c00))
                    $fatal(1, "NEOX V changed during Q/K rotation test");
                if (newkv_wr_lane_mask[lane] && neox_test_mode &&
                    !newkv_wr_addr[10] &&
                    ((newkv_wr_data[lane*16 + 10 +: 5] == 5'h1f) ||
                     (newkv_wr_data[lane*16 +: 15] == 15'd0)))
                    $fatal(1, "NEOX K output nonfinite/zero");
            end
        end

        if (rst_n && r_wr_valid && r_wr_ready) begin
            r_write_count <= r_write_count + 1;
            if (r_wr_lane_mask != (r_wr_wave ?
                    cfg_token_mask[7:4] : cfg_token_mask[3:0]))
                $fatal(1, "residual lane mask mismatch");
            for (lane = 0; lane < 4; lane = lane + 1)
                if (r_wr_lane_mask[lane] &&
                    (r_wr_data[lane*32 +: 32] != 32'h4040_0000))
                    $fatal(1, "direct residual was not exactly 1+2=3");
        end

        if (rst_n && q8_wr_valid && q8_wr_ready) begin
            if (q8_write_count == 0)
                first_q8_write_addr <= q8_wr_addr;
            q8_write_count <= q8_write_count + 1;
            if ((integer'(q8_wr_addr) != (128 + q8_write_count /
                                          wavecount(cfg_token_mask))) ||
                (q8_wr_wave != ((|cfg_token_mask[3:0]) ?
                    ((|cfg_token_mask[7:4]) && q8_write_count[0]) :
                    1'b1)) ||
                (q8_wr_lane_mask != (q8_wr_wave ?
                    cfg_token_mask[7:4] : cfg_token_mask[3:0])))
                $fatal(1, "Q8 geometry/order mismatch count=%0d",
                       q8_write_count);
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (q8_wr_lane_mask[lane]) begin
                    if ((q8_wr_data[lane*272 + 256 +: 16] == 16'd0) ||
                        (q8_wr_data[lane*272 + 266 +: 5] == 5'h1f))
                        $fatal(1, "Q8 active lane has bad scale");
                    for (quant = 0; quant < 32; quant = quant + 1)
                        if (q8_wr_data[lane*272 + quant*8 +: 8] != 8'h7f)
                            $fatal(1, "Q8 constant tile was not +127");
                end else if (q8_wr_data[lane*272 +: 272] != 272'd0) begin
                    $fatal(1, "Q8 inactive lane was not zero-filled");
                end
            end
        end
        if (rst_n && q8_collision_error)
            $fatal(1, "unexpected shared Q8 collision");
    end

    wire reduce_rsp_nonfinite =
        (dut.reduce_src_rsp_data[30:23] == 8'hff) ||
        (dut.reduce_src_rsp_data[62:55] == 8'hff) ||
        (dut.reduce_src_rsp_data[94:87] == 8'hff) ||
        (dut.reduce_src_rsp_data[126:119] == 8'hff);
    always @(posedge clk) begin
        if (rst_n && dut.reduce_src_rsp_valid &&
            dut.reduce_src_rsp_ready && reduce_rsp_nonfinite) begin
            injected_head_bad_seen <= 1'b1;
            injected_head_tail_seen <= dut.head_mem_valid_q;
            injected_head_outstanding <=
                integer'(dut.u_reduce.issue_count_q) -
                integer'(dut.u_reduce.rsp_count_q);
        end
    end

    wire neox_head_target = neox_test_mode && !dut.wave_q &&
        (dut.head_index_q == 6'd0) &&
        ((dut.head_kind_q == 2'd0) || (dut.head_kind_q == 2'd1));
    integer neox_lane;
    always @(posedge clk) begin
        if (!rst_n || clear || !neox_test_mode) begin
            neox_q_seen <= 128'd0;
            neox_k_seen <= 128'd0;
            neox_q_norm_count <= 0;
            neox_k_norm_count <= 0;
            neox_q_write_count <= 0;
            neox_k_write_count <= 0;
        end else begin
            if (dut.apply_in_fire && neox_head_target) begin
                for (neox_lane = 0; neox_lane < 4;
                     neox_lane = neox_lane + 1)
                    if (dut.apply_stage_gamma_q[neox_lane*32 +: 32] !==
                        neox_gamma(dut.head_kind_q == 2'd1,
                                   dut.apply_stage_dim_q))
                        $fatal(1,
                            "NEOX gamma permutation mismatch kind=%0d dim=%0d",
                            dut.head_kind_q, dut.apply_stage_dim_q);
            end

            if (dut.apply_issue && neox_head_target) begin
                if (dut.head_read_addr !== {1'b0, dut.apply_issue_dim})
                    $fatal(1,
                        "NEOX head read order mismatch count=%0d addr=%0d dim=%0d",
                        dut.apply_issue_count_q, dut.head_read_addr,
                        dut.apply_issue_dim);
            end

            if (dut.apply_out_fire && neox_head_target) begin
                if (dut.head_kind_q == 2'd0) begin
                    if (neox_q_seen[neox_apply_dim])
                        $fatal(1, "duplicate NEOX Q normalized dim=%0d",
                               neox_apply_dim);
                    neox_q_norm[neox_apply_dim] <= dut.apply_out_data[31:0];
                    neox_q_seen[neox_apply_dim] <= 1'b1;
                    neox_q_norm_count <= neox_q_norm_count + 1;
                end else begin
                    if (neox_k_seen[neox_apply_dim])
                        $fatal(1, "duplicate NEOX K normalized dim=%0d",
                               neox_apply_dim);
                    neox_k_norm[neox_apply_dim] <= dut.apply_out_data[31:0];
                    neox_k_seen[neox_apply_dim] <= 1'b1;
                    neox_k_norm_count <= neox_k_norm_count + 1;
                end
            end

            if (query_wr_valid && query_wr_ready && !query_wr_wave &&
                (query_wr_addr < 12'd128)) begin
                if ((query_wr_addr !==
                     {5'd0, neox_count_dim(neox_q_write_count[7:0])}) ||
                    !neox_q_seen[neox_query_source_dim] ||
                    (query_wr_data[31:0] !== neox_query_expected))
                    $fatal(1,
                        "NEOX Q write mismatch count=%0d addr=%0d data=%h expected=%h",
                        neox_q_write_count, query_wr_addr,
                        query_wr_data[31:0], neox_query_expected);
                neox_q_write_count <= neox_q_write_count + 1;
            end

            if (newkv_wr_valid && newkv_wr_ready && !newkv_wr_wave &&
                !newkv_wr_addr[10] && (newkv_wr_addr[9:7] == 3'd0)) begin
                if ((newkv_wr_addr !==
                     {1'b0, 3'd0,
                      neox_count_dim(neox_k_write_count[7:0])}) ||
                    !neox_k_seen[neox_k_source_dim] ||
                    (newkv_wr_data[15:0] !== neox_k_expected_f16))
                    $fatal(1,
                        "NEOX K write mismatch count=%0d addr=%0d data=%h expected=%h",
                        neox_k_write_count, newkv_wr_addr,
                        newkv_wr_data[15:0], neox_k_expected_f16);
                neox_k_write_count <= neox_k_write_count + 1;
            end
        end
    end

    // The registered head response must remain a stable ready/valid record
    // while any downstream consumer is stalled.
    always @(posedge clk) begin
        if (!rst_n || clear || abort_run) begin
            head_stalled_q <= 1'b0;
            stalled_head_data_q <= 128'd0;
            stalled_head_dim_q <= 7'd0;
        end else begin
            if (head_stalled_q &&
                (!dut.head_rsp_valid_q ||
                 (dut.head_rsp_data_q !== stalled_head_data_q) ||
                 (dut.head_rsp_dim_q !== stalled_head_dim_q)))
                $fatal(1, "head response changed while stalled");
            head_stalled_q <= dut.head_rsp_valid_q && !dut.head_rsp_ready;
            if (dut.head_rsp_valid_q && !dut.head_rsp_ready) begin
                stalled_head_data_q <= dut.head_rsp_data_q;
                stalled_head_dim_q <= dut.head_rsp_dim_q;
                head_stall_count <= head_stall_count + 1;
            end
        end
    end

    function automatic integer popcount8(input [7:0] mask);
        integer bit_index;
        begin
            popcount8 = 0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                popcount8 = popcount8 + mask[bit_index];
        end
    endfunction

    function automatic integer wavecount(input [7:0] mask);
        wavecount = ((|mask[3:0]) ? 1 : 0) + ((|mask[7:4]) ? 1 : 0);
    endfunction

    function automatic integer lastactive(input [7:0] mask);
        integer bit_index;
        begin
            lastactive = 0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                if (mask[bit_index]) lastactive = bit_index;
        end
    endfunction

    function automatic integer firstactive(input [7:0] mask);
        integer bit_index;
        reg found;
        begin
            firstactive = 0;
            found = 1'b0;
            for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
                if (!found && mask[bit_index]) begin
                    firstactive = bit_index;
                    found = 1'b1;
                end
        end
    endfunction

    task automatic submit_cfg(
        input [5:0] q_heads,
        input [7:0] mask,
        input [16:0] position
    );
        begin
            @(negedge clk);
            cfg_q_heads = q_heads;
            cfg_token_mask = mask;
            cfg_position_base = position;
            cfg_valid = 1'b1;
            while (!cfg_ready) @(negedge clk);
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task automatic send_record(
        input [2:0] token,
        input [17:0] row,
        input is_last
    );
        begin
            @(negedge clk);
            proj_token = token;
            proj_row = row;
            proj_data_f32 = inject_projection_nonfinite &&
                            (token == 3'd0) && (row == 18'd8) ?
                            32'h7fc0_0001 : stream_value;
            proj_last = is_last;
            proj_valid = 1'b1;
            do @(posedge clk); while (!proj_ready);
            @(negedge clk);
            proj_valid = 1'b0;
        end
    endtask

    task automatic stream_projection(
        input integer rows,
        input [7:0] mask,
        input corrupt_frame
    );
        integer block_base;
        integer token;
        integer row_lane;
        begin
            while (!projection_armed) @(posedge clk);
            for (block_base = 0; block_base < rows; block_base = block_base + 16)
                for (token = 0; token < 8; token = token + 1)
                    if (mask[token])
                        for (row_lane = 0; row_lane < 16; row_lane = row_lane + 1)
                            send_record(token[2:0],
                                18'(block_base + row_lane),
                                ((block_base + row_lane == rows - 1) &&
                                 (token == lastactive(mask))) ||
                                (corrupt_frame && block_base == 0 &&
                                 row_lane == 0 &&
                                 token == firstactive(mask)));
        end
    endtask

    task automatic clear_counts;
        begin
            gamma_request_count = 0;
            rope_request_count = 0;
            query_write_count = 0;
            k_write_count = 0;
            v_write_count = 0;
            proj_record_count = 0;
            r_request_count = 0;
            r_write_count = 0;
            q8_write_count = 0;
            first_q8_write_addr = 9'h1ff;
            injected_head_bad_seen = 1'b0;
            injected_head_tail_seen = 1'b0;
            injected_head_outstanding = 0;
        end
    endtask

    task automatic run_qkv(
        input [5:0] q_heads,
        input [7:0] mask,
        input [16:0] position
    );
        integer rows;
        integer waves;
        integer q_expected;
        integer kv_expected;
        begin
            rows = q_heads == 16 ? 4096 : 6144;
            waves = wavecount(mask);
            q_expected = q_heads * waves * 128;
            kv_expected = 8 * waves * 128;
            clear_counts();
            cfg_mode = 2'd0;
            if (q_heads == 6'd16) begin
                cfg_hidden_dim = 13'd2048;
                cfg_ffn_dim = 15'd6144;
            end else begin
                cfg_hidden_dim = 13'd2560;
                cfg_ffn_dim = 15'd9728;
            end
            stream_value = 32'h3f80_0000;
            submit_cfg(q_heads, mask, position);
            stream_projection(rows, mask, 1'b0);
            while (!done_valid) @(posedge clk);
            if (done_error || done_status != 16'd0)
                $fatal(1, "sink error status=%h state=%0d", done_status,
                       debug_state);
            if ((gamma_request_count != 2) ||
                (rope_request_count != 1) ||
                (proj_record_count != rows * popcount8(mask)) ||
                (query_write_count != q_expected) ||
                (k_write_count != kv_expected) ||
                (v_write_count != kv_expected) || (q8_write_count != 0))
                $fatal(1, "count mismatch g=%0d rp=%0d p=%0d q=%0d k=%0d v=%0d",
                    gamma_request_count, rope_request_count,
                    proj_record_count, query_write_count,
                    k_write_count, v_write_count);
            $display("QKV Q%0d mask=%h cycles=%0d records=%0d PASS",
                     q_heads, mask, done_cycles, proj_record_count);
            @(posedge clk);
        end
    endtask

    task automatic run_nonfinite_qkv_and_restart;
        integer timeout;
        begin
            clear_counts();
            cfg_mode = 2'd0;
            cfg_hidden_dim = 13'd2048;
            cfg_ffn_dim = 15'd6144;
            stream_value = 32'h3f80_0000;
            inject_projection_nonfinite = 1'b1;
            done_ready = 1'b0;
            submit_cfg(6'd16, 8'h21, 17'd37);
            fork : nonfinite_qkv_fork
                begin
                    stream_projection(4096, 8'h21, 1'b0);
                end
                begin
                    timeout = 0;
                    while (!done_valid && timeout < 100000) begin
                        @(negedge clk);
                        timeout = timeout + 1;
                    end
                    if (!done_valid)
                        $fatal(1, "nonfinite QKV cleanup timed out");
                end
            join_any
            disable nonfinite_qkv_fork;
            proj_valid = 1'b0;
            proj_last = 1'b0;

            if (!done_valid || !done_error || (done_status != 16'h0024))
                $fatal(1, "nonfinite QKV status mismatch %h", done_status);
            if (!injected_head_bad_seen || !injected_head_tail_seen ||
                (injected_head_outstanding < 2))
                $fatal(1,
                    "nonfinite QKV lacked a prefetched head tail out=%0d tail=%0d",
                    injected_head_outstanding, injected_head_tail_seen);
            if (dut.head_mem_valid_q || dut.head_rsp_valid_q ||
                dut.reduce_busy)
                $fatal(1, "QKV error published before head drain");
            repeat (7) begin
                @(negedge clk);
                if (!done_valid || !done_error ||
                    (done_status != 16'h0024) || dut.head_mem_valid_q ||
                    dut.head_rsp_valid_q || dut.reduce_busy)
                    $fatal(1, "QKV error completion/drain was not stable");
            end
            done_ready = 1'b1;
            @(negedge clk);
            inject_projection_nonfinite = 1'b0;
            while (!cfg_ready) @(negedge clk);

            run_qkv(6'd16, 8'h21, 17'd37);
            $display("projection sink nonfinite reducer drain/restart PASS");
        end
    endtask

    task automatic run_neox_qkv;
        integer dimension;
        integer other;
        begin
            @(negedge clk);
            neox_test_mode = 1'b1;
            run_qkv(6'd16, 8'h01, 17'd37);
            if ((neox_q_seen !== {128{1'b1}}) ||
                (neox_k_seen !== {128{1'b1}}) ||
                (neox_q_norm_count != 128) ||
                (neox_k_norm_count != 128) ||
                (neox_q_write_count != 128) ||
                (neox_k_write_count != 128))
                $fatal(1,
                    "NEOX full-head coverage mismatch qn=%0d kn=%0d qw=%0d kw=%0d",
                    neox_q_norm_count, neox_k_norm_count,
                    neox_q_write_count, neox_k_write_count);

            // Distinct gamma for every dimension must survive RMS apply as 128
            // distinct normalized values before the exact cos=0,sin=1 rotate.
            for (dimension = 0; dimension < 128;
                 dimension = dimension + 1)
                for (other = dimension + 1; other < 128;
                     other = other + 1) begin
                    if (neox_q_norm[dimension] === neox_q_norm[other])
                        $fatal(1, "NEOX Q dimensions collapsed %0d/%0d",
                               dimension, other);
                    if (neox_k_norm[dimension] === neox_k_norm[other])
                        $fatal(1, "NEOX K dimensions collapsed %0d/%0d",
                               dimension, other);
                end
            $display("QKV NEOX full-head permutation/numerics PASS");
            @(negedge clk);
            neox_test_mode = 1'b0;
        end
    endtask

    task automatic run_delta(
        input [1:0] mode,
        input [12:0] hidden,
        input [14:0] ffn,
        input [7:0] mask
    );
        integer records;
        integer arena_records;
        begin
            records = hidden * popcount8(mask);
            arena_records = hidden * wavecount(mask);
            clear_counts();
            cfg_mode = mode;
            cfg_hidden_dim = hidden;
            cfg_ffn_dim = ffn;
            stream_value = 32'h4000_0000;
            submit_cfg(6'd16, mask, 17'd0);
            stream_projection(integer'(hidden), mask, 1'b0);
            while (!done_valid) @(posedge clk);
            if (done_error || done_status != 16'd0)
                $fatal(1, "delta sink error status=%h", done_status);
            if ((proj_record_count != records) ||
                (r_request_count != arena_records) ||
                (r_write_count != arena_records) ||
                (gamma_request_count != 0) || (rope_request_count != 0) ||
                (query_write_count != 0) || (k_write_count != 0) ||
                (v_write_count != 0) || (q8_write_count != 0))
                $fatal(1, "delta count mismatch p=%0d rr=%0d rw=%0d",
                       proj_record_count, r_request_count, r_write_count);
            $display("DELTA mode=%0d D=%0d mask=%h cycles=%0d PASS",
                     mode, hidden, mask, done_cycles);
            @(posedge clk);
        end
    endtask

    task automatic run_gate(
        input [12:0] hidden,
        input [14:0] ffn,
        input [7:0] mask,
        input corrupt_frame
    );
        integer records;
        integer q8_records;
        begin
            records = 2 * ffn * popcount8(mask);
            q8_records = (integer'(ffn) / 32) * wavecount(mask);
            clear_counts();
            cfg_mode = 2'd1;
            cfg_hidden_dim = hidden;
            cfg_ffn_dim = ffn;
            stream_value = 32'h3f80_0000;
            submit_cfg(6'd16, mask, 17'd0);
            stream_projection(2 * ffn, mask, corrupt_frame);
            while (!done_valid) @(posedge clk);
            if (corrupt_frame) begin
                if (!done_error || ((done_status & 16'h0010) == 16'd0))
                    $fatal(1, "bad Gate/Up frame was not rejected status=%h",
                           done_status);
            end else if (done_error || done_status != 16'd0) begin
                $fatal(1, "Gate/Up sink error status=%h", done_status);
            end
            if ((proj_record_count != records) ||
                (q8_write_count != q8_records) ||
                (!corrupt_frame && (first_q8_write_addr != 9'd128)) ||
                (gamma_request_count != 0) || (rope_request_count != 0) ||
                (query_write_count != 0) || (k_write_count != 0) ||
                (v_write_count != 0) || (r_request_count != 0) ||
                (r_write_count != 0))
                $fatal(1, "Gate/Up count mismatch p=%0d q8=%0d",
                       proj_record_count, q8_write_count);
            $display("GATE_UP D=%0d F=%0d mask=%h frame_bad=%0d cycles=%0d PASS",
                     hidden, ffn, mask, corrupt_frame, done_cycles);
            @(posedge clk);
        end
    endtask

    task automatic abort_gate_q8;
        begin
            clear_counts();
            cfg_mode = 2'd1;
            cfg_hidden_dim = 13'd2048;
            cfg_ffn_dim = 15'd6144;
            stream_value = 32'h3f80_0000;
            submit_cfg(6'd16, 8'h84, 17'd0);
            fork : gate_abort_fork
                begin
                    stream_projection(12288, 8'h84, 1'b0);
                end
                begin
                    while (debug_state != 5'd16) @(posedge clk);
                    repeat (5) @(posedge clk);
                    @(negedge clk); abort_run = 1'b1;
                    @(negedge clk); abort_run = 1'b0;
                end
            join_any
            disable gate_abort_fork;
            proj_valid = 1'b0;
            proj_last = 1'b0;
            while (!cfg_ready) @(posedge clk);
            if (done_valid) $fatal(1, "Gate/Up abort published completion");
            $display("GATE_UP abort/restart boundary PASS");
        end
    endtask

    task automatic clear_delta_with_outstanding_r;
        begin
            clear_counts();
            cfg_mode = 2'd2;
            cfg_hidden_dim = 13'd2048;
            cfg_ffn_dim = 15'd6144;
            stream_value = 32'h4000_0000;
            submit_cfg(6'd16, 8'h03, 17'd0);
            fork : delta_clear_fork
                begin
                    stream_projection(2048, 8'h03, 1'b0);
                end
                begin
                    while (!(r_rd_req_valid && r_rd_req_ready))
                        @(negedge clk);
                    @(negedge clk);
                    if (!r_rd_rsp_valid)
                        $fatal(1, "clear test did not create sink R response");
                    if (!dut.head_mem_valid_q && !dut.head_rsp_valid_q)
                        $fatal(1, "clear test missed sink head read pipeline");
                    clear = 1'b1;
                    @(negedge clk);
                    clear = 1'b0;
                end
            join_any
            disable delta_clear_fork;
            proj_valid = 1'b0;
            proj_last = 1'b0;
            while (!cfg_ready) @(posedge clk);
            if (r_rd_rsp_valid || dut.head_mem_valid_q ||
                dut.head_rsp_valid_q || done_valid)
                $fatal(1, "global clear retained sink response/completion");
            $display("projection sink global clear with outstanding R PASS");
        end
    endtask

    task automatic abort_delta_with_head_pipeline;
        begin
            clear_counts();
            cfg_mode = 2'd2;
            cfg_hidden_dim = 13'd2048;
            cfg_ffn_dim = 15'd6144;
            stream_value = 32'h4000_0000;
            submit_cfg(6'd16, 8'h81, 17'd0);
            fork : delta_abort_fork
                begin
                    stream_projection(2048, 8'h81, 1'b0);
                end
                begin
                    while (!dut.head_mem_valid_q && !dut.head_rsp_valid_q)
                        @(negedge clk);
                    abort_run = 1'b1;
                    @(negedge clk);
                    abort_run = 1'b0;
                end
            join_any
            disable delta_abort_fork;
            proj_valid = 1'b0;
            proj_last = 1'b0;
            while (!cfg_ready) @(posedge clk);
            if (dut.head_mem_valid_q || dut.head_rsp_valid_q || done_valid)
                $fatal(1, "local abort retained sink head response/completion");
            $display("projection sink head-pipeline abort/drain PASS");
        end
    endtask

    task automatic abort_qkv_with_head_reduce_pipeline;
        begin
            clear_counts();
            cfg_mode = 2'd0;
            cfg_hidden_dim = 13'd2048;
            cfg_ffn_dim = 15'd6144;
            stream_value = 32'h3f80_0000;
            submit_cfg(6'd16, 8'h21, 17'd37);
            fork : qkv_abort_fork
                begin
                    stream_projection(4096, 8'h21, 1'b0);
                end
                begin
                    while ((debug_state != 5'd7) ||
                           (!dut.head_mem_valid_q &&
                            !dut.head_rsp_valid_q))
                        @(negedge clk);
                    abort_run = 1'b1;
                    @(negedge clk);
                    abort_run = 1'b0;
                end
            join_any
            disable qkv_abort_fork;
            proj_valid = 1'b0;
            proj_last = 1'b0;
            while (!cfg_ready) @(posedge clk);
            if (dut.head_mem_valid_q || dut.head_rsp_valid_q ||
                dut.reduce_busy || done_valid)
                $fatal(1,
                    "QKV abort retained reducer/head response/completion");
            $display("projection sink QKV reducer-pipeline abort/drain PASS");
        end
    endtask

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // Abort an in-flight gamma burst and prove its tail is drained before
        // accepting a fresh command.
        clear_counts();
        submit_cfg(6'd16, 8'ha5, 17'd65528);
        while (!gamma_source_active) @(posedge clk);
        repeat (5) @(posedge clk);
        @(negedge clk); abort_run = 1'b1;
        @(negedge clk); abort_run = 1'b0;
        while (!cfg_ready) @(posedge clk);
        if (done_valid) $fatal(1, "abort published completion");

        run_qkv(6'd16, 8'ha5, 17'd65528);
        run_qkv(6'd16, 8'h01, 17'd65535);
        run_qkv(6'd32, 8'ha0, 17'd1024);
        run_nonfinite_qkv_and_restart();
        run_neox_qkv();
        run_delta(2'd2, 13'd2048, 15'd6144, 8'h4a);
        run_delta(2'd3, 13'd2560, 15'd9728, 8'ha0);
        run_delta(2'd2, 13'd4096, 15'd12288, 8'h81);
        clear_delta_with_outstanding_r();
        abort_delta_with_head_pipeline();
        abort_qkv_with_head_reduce_pipeline();
        abort_gate_q8();
        run_gate(13'd2048, 15'd6144, 8'h84, 1'b0);
        run_gate(13'd4096, 15'd12288, 8'h20, 1'b0);

        if (head_stall_count == 0)
            $fatal(1, "head response pipeline was never backpressured");
        $display(" projection_sink_tb PASS total_cycles=%0d head_stalls=%0d",
                 cycle, head_stall_count);
        $finish;
    end

    initial begin
        #6000000;
        $fatal(1, "projection sink timeout state=%0d", debug_state);
    end
endmodule

`default_nettype wire
