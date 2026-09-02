`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [3:0] cmd_token_count = 4'd0;
    reg [7:0] cmd_token_mask = 8'd0;
    reg [5:0] cmd_q_heads = 6'd0;
    reg [16:0] cmd_history_len = 17'd0;
    reg [31:0] cmd_scale = 32'h3db504f3;
    reg [63:0] cmd_layer_kv_base = 64'd0;

    wire query_req_valid;
    wire query_req_ready;
    wire query_req_wave;
    wire [11:0] query_req_addr;
    reg query_rsp_valid = 1'b0;
    wire query_rsp_ready;
    reg [127:0] query_rsp_data = 128'd0;

    wire newkv_req_valid;
    wire newkv_req_ready;
    wire newkv_req_wave;
    wire [10:0] newkv_req_addr;
    reg newkv_rsp_valid = 1'b0;
    wire newkv_rsp_ready;
    reg [63:0] newkv_rsp_data = 64'd0;

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
    wire leaf_q8_abort;
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

    wire k_cmd_valid;
    wire k_cmd_ready;
    wire [63:0] k_cmd_addr;
    wire [31:0] k_cmd_segment_beats;
    wire [31:0] k_cmd_stride_bytes;
    wire [16:0] k_cmd_repeats;
    wire k_abort;
    wire [127:0] k_data;
    wire k_valid;
    wire k_ready;
    wire k_last;
    wire k_error;
    wire k_busy;
    reg k_done_valid = 1'b0;
    wire k_done_ready;
    reg k_done_error = 1'b0;
    reg [7:0] k_done_status = 8'd0;

    wire v_cmd_valid;
    wire v_cmd_ready;
    wire [63:0] v_cmd_addr;
    wire [31:0] v_cmd_segment_beats;
    wire [31:0] v_cmd_stride_bytes;
    wire [16:0] v_cmd_repeats;
    wire v_abort;
    wire [127:0] v_data;
    wire v_valid;
    wire v_ready;
    wire v_last;
    wire v_error;
    wire v_busy;
    reg v_done_valid = 1'b0;
    wire v_done_ready;
    reg v_done_error = 1'b0;
    reg [7:0] v_done_status = 8'd0;
    reg inject_v_done_error = 1'b0;

    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

     attention_service dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_token_count(cmd_token_count),
        .cmd_token_mask(cmd_token_mask), .cmd_q_heads(cmd_q_heads),
        .cmd_history_len(cmd_history_len), .cmd_scale(cmd_scale),
        .cmd_layer_kv_base(cmd_layer_kv_base),
        .query_rd_req_valid(query_req_valid),
        .query_rd_req_ready(query_req_ready),
        .query_rd_req_wave(query_req_wave),
        .query_rd_req_addr(query_req_addr),
        .query_rd_rsp_valid(query_rsp_valid),
        .query_rd_rsp_ready(query_rsp_ready),
        .query_rd_rsp_data(query_rsp_data),
        .newkv_rd_req_valid(newkv_req_valid),
        .newkv_rd_req_ready(newkv_req_ready),
        .newkv_rd_req_wave(newkv_req_wave),
        .newkv_rd_req_addr(newkv_req_addr),
        .newkv_rd_rsp_valid(newkv_rsp_valid),
        .newkv_rd_rsp_ready(newkv_rsp_ready),
        .newkv_rd_rsp_data(newkv_rsp_data),
        .q8_wr_valid(q8_wr_valid), .q8_wr_ready(q8_wr_ready),
        .q8_wr_wave(q8_wr_wave), .q8_wr_addr(q8_wr_addr),
        .q8_wr_lane_mask(q8_wr_lane_mask), .q8_wr_data(q8_wr_data),
        .leaf_q8_cfg_valid(leaf_q8_cfg_valid),
        .leaf_q8_cfg_ready(leaf_q8_cfg_ready),
        .leaf_q8_cfg_rows(leaf_q8_cfg_rows),
        .leaf_q8_cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .leaf_q8_abort(leaf_q8_abort), .leaf_q8_busy(leaf_q8_busy),
        .leaf_q8_in_valid(leaf_q8_in_valid),
        .leaf_q8_in_ready(leaf_q8_in_ready),
        .leaf_q8_in_data(leaf_q8_in_data),
        .leaf_q8_out_valid(leaf_q8_out_valid),
        .leaf_q8_out_ready(leaf_q8_out_ready),
        .leaf_q8_out_block(leaf_q8_out_block),
        .leaf_q8_out_data(leaf_q8_out_data),
        .leaf_q8_out_status(leaf_q8_out_status),
        .leaf_q8_out_last(leaf_q8_out_last),
        .hist_k_cmd_valid(k_cmd_valid), .hist_k_cmd_ready(k_cmd_ready),
        .hist_k_cmd_addr(k_cmd_addr),
        .hist_k_cmd_segment_beats(k_cmd_segment_beats),
        .hist_k_cmd_stride_bytes(k_cmd_stride_bytes),
        .hist_k_cmd_repeats(k_cmd_repeats), .hist_k_abort(k_abort),
        .hist_k_data(k_data), .hist_k_valid(k_valid),
        .hist_k_ready(k_ready), .hist_k_last(k_last),
        .hist_k_error(k_error), .hist_k_busy(k_busy),
        .hist_k_done_valid(k_done_valid),
        .hist_k_done_ready(k_done_ready),
        .hist_k_done_error(k_done_error),
        .hist_k_done_status(k_done_status),
        .hist_v_cmd_valid(v_cmd_valid), .hist_v_cmd_ready(v_cmd_ready),
        .hist_v_cmd_addr(v_cmd_addr),
        .hist_v_cmd_segment_beats(v_cmd_segment_beats),
        .hist_v_cmd_stride_bytes(v_cmd_stride_bytes),
        .hist_v_cmd_repeats(v_cmd_repeats), .hist_v_abort(v_abort),
        .hist_v_data(v_data), .hist_v_valid(v_valid),
        .hist_v_ready(v_ready), .hist_v_last(v_last),
        .hist_v_error(v_error), .hist_v_busy(v_busy),
        .hist_v_done_valid(v_done_valid),
        .hist_v_done_ready(v_done_ready),
        .hist_v_done_error(v_done_error),
        .hist_v_done_status(v_done_status),
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status)
    );

     q8_pack4 u_q8 (
        .clk(clk), .rst_n(rst_n), .cfg_valid(leaf_q8_cfg_valid),
        .cfg_ready(leaf_q8_cfg_ready), .cfg_rows(leaf_q8_cfg_rows),
        .cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .abort_run(leaf_q8_abort), .busy(leaf_q8_busy),
        .in_valid(leaf_q8_in_valid), .in_ready(leaf_q8_in_ready),
        .in_data(leaf_q8_in_data), .out_valid(leaf_q8_out_valid),
        .out_ready(leaf_q8_out_ready), .out_block(leaf_q8_out_block),
        .out_data(leaf_q8_out_data), .out_status(leaf_q8_out_status),
        .out_last(leaf_q8_out_last)
    );

    integer cycles = 0;
    integer query_request_count = 0;
    integer newkv_request_count = 0;
    integer q8_write_count = 0;
    integer k_command_count = 0;
    integer v_command_count = 0;
    integer active_groups = 0;
    integer active_kv_heads = 0;
    integer active_history = 0;
    integer expected_q8_writes = 0;
    reg [63:0] active_layer_base = 64'd0;
    reg check_q8 = 1'b0;

    assign query_req_ready = (!query_rsp_valid || query_rsp_ready) &&
                             (cycles[2:0] != 3'd3);
    assign newkv_req_ready = (!newkv_rsp_valid || newkv_rsp_ready) &&
                             (cycles[2:0] != 3'd5);
    assign q8_wr_ready = cycles[2:0] != 3'd6;

    // The resident arenas have one elastic response slot. All-zero Q/K/V
    // makes the full numeric path deterministic: attention and Q8 are zero.
    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (!rst_n) begin
            query_rsp_valid <= 1'b0;
            newkv_rsp_valid <= 1'b0;
        end else begin
            if (query_rsp_valid && query_rsp_ready)
                query_rsp_valid <= 1'b0;
            if (query_req_valid && query_req_ready) begin
                query_rsp_valid <= 1'b1;
                query_rsp_data <= 128'd0;
                query_request_count <= query_request_count + 1;
            end
            if (newkv_rsp_valid && newkv_rsp_ready)
                newkv_rsp_valid <= 1'b0;
            if (newkv_req_valid && newkv_req_ready) begin
                newkv_rsp_valid <= 1'b1;
                newkv_rsp_data <= 64'd0;
                newkv_request_count <= newkv_request_count + 1;
            end
        end

        if (rst_n && check_q8 && q8_wr_valid && q8_wr_ready) begin
            integer writes_per_group;
            integer group_i;
            integer wave_i;
            integer block_i;
            writes_per_group = cmd_token_count > 4 ? 64 : 32;
            group_i = q8_write_count / writes_per_group;
            wave_i = (q8_write_count / 32) %
                     (cmd_token_count > 4 ? 2 : 1);
            block_i = q8_write_count % 32;
            if (q8_wr_wave != wave_i[0] ||
                q8_wr_addr != group_i * 32 + block_i)
                $fatal(1, "Q8 order mismatch n=%0d wave=%0d addr=%0d",
                       q8_write_count, q8_wr_wave, q8_wr_addr);
            if (q8_wr_lane_mask != (wave_i ?
                    cmd_token_mask[7:4] : cmd_token_mask[3:0]))
                $fatal(1, "Q8 lane mask mismatch n=%0d",
                       q8_write_count);
            if (q8_wr_data !== 1088'd0)
                $fatal(1, "zero attention result did not quantize to zero");
            q8_write_count <= q8_write_count + 1;
        end

        if (cycles > 5000000)
            $fatal(1, "timeout");
    end

    // Two independent behavioral 128-bit history movers.
    reg k_active = 1'b0;
    reg v_active = 1'b0;
    integer k_index = 0;
    integer v_index = 0;
    integer k_total = 0;
    integer v_total = 0;
    assign k_cmd_ready = !k_active && !k_done_valid;
    assign v_cmd_ready = !v_active && !v_done_valid;
    assign k_busy = k_active;
    assign v_busy = v_active;
    assign k_valid = k_active && (cycles[2:0] != 3'd1);
    assign v_valid = v_active && (cycles[2:0] != 3'd4);
    assign k_data = 128'd0;
    assign v_data = 128'd0;
    assign k_last = k_active && (k_index + 1 == k_total);
    assign v_last = v_active && (v_index + 1 == v_total);
    assign k_error = 1'b0;
    assign v_error = 1'b0;

    task automatic check_mover_command(
        input integer kind,
        input integer ordinal,
        input [63:0] addr,
        input [31:0] segment_beats,
        input [31:0] stride_bytes,
        input [16:0] repeats
    );
        integer head_base;
        reg [63:0] expected_addr;
        begin
            head_base = ordinal * active_kv_heads;
            expected_addr = active_layer_base + head_base * 256 +
                            (kind ? 2048 : 0);
            if (addr != expected_addr ||
                segment_beats != active_kv_heads * 16 ||
                stride_bytes != 4096 || repeats != active_history)
                $fatal(1, "bad %s mover cmd n=%0d addr=%h seg=%0d stride=%0d rep=%0d",
                       kind ? "V" : "K", ordinal, addr, segment_beats,
                       stride_bytes, repeats);
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            k_active <= 1'b0;
            v_active <= 1'b0;
            k_done_valid <= 1'b0;
            v_done_valid <= 1'b0;
        end else begin
            if (k_done_valid && k_done_ready)
                k_done_valid <= 1'b0;
            if (v_done_valid && v_done_ready)
                v_done_valid <= 1'b0;

            if (k_abort) begin
                k_active <= 1'b0;
                k_done_valid <= 1'b0;
                k_done_error <= 1'b0;
                k_done_status <= 8'd0;
            end else begin
                if (k_cmd_valid && k_cmd_ready) begin
                    check_mover_command(0, k_command_count, k_cmd_addr,
                        k_cmd_segment_beats, k_cmd_stride_bytes,
                        k_cmd_repeats);
                    k_active <= 1'b1;
                    k_index <= 0;
                    k_total <= k_cmd_segment_beats * k_cmd_repeats;
                    k_command_count <= k_command_count + 1;
                end
                if (k_valid && k_ready) begin
                    if (k_last) begin
                        k_active <= 1'b0;
                        k_done_valid <= 1'b1;
                        k_done_error <= 1'b0;
                        k_done_status <= 8'd0;
                    end else begin
                        k_index <= k_index + 1;
                    end
                end
            end

            if (v_abort) begin
                v_active <= 1'b0;
                v_done_valid <= 1'b0;
                v_done_error <= 1'b0;
                v_done_status <= 8'd0;
            end else begin
                if (v_cmd_valid && v_cmd_ready) begin
                    check_mover_command(1, v_command_count, v_cmd_addr,
                        v_cmd_segment_beats, v_cmd_stride_bytes,
                        v_cmd_repeats);
                    v_active <= 1'b1;
                    v_index <= 0;
                    v_total <= v_cmd_segment_beats * v_cmd_repeats;
                    v_command_count <= v_command_count + 1;
                end
                if (v_valid && v_ready) begin
                    if (v_last) begin
                        v_active <= 1'b0;
                        v_done_valid <= 1'b1;
                        v_done_error <= inject_v_done_error;
                        v_done_status <= inject_v_done_error ? 8'h02 : 8'd0;
                    end else begin
                        v_index <= v_index + 1;
                    end
                end
            end
        end
    end

    task automatic launch(
        input [3:0] tokens,
        input [7:0] mask,
        input [5:0] q_heads,
        input [16:0] history,
        input [63:0] layer_base
    );
        begin
            active_groups = q_heads == 16 ? 2 : 4;
            active_kv_heads = q_heads == 16 ? 4 : 2;
            active_history = history;
            active_layer_base = layer_base;
            @(negedge clk);
            cmd_token_count = tokens;
            cmd_token_mask = mask;
            cmd_q_heads = q_heads;
            cmd_history_len = history;
            cmd_layer_kv_base = layer_base;
            cmd_valid = 1'b1;
            do @(posedge clk); while (!cmd_ready);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic run_good(
        input [3:0] tokens,
        input [7:0] mask,
        input [5:0] q_heads,
        input [16:0] history,
        input [63:0] layer_base
    );
        integer want_commands;
        begin
            q8_write_count = 0;
            k_command_count = 0;
            v_command_count = 0;
            expected_q8_writes = (q_heads / 8) *
                                 (tokens > 4 ? 64 : 32);
            check_q8 = 1'b1;
            launch(tokens, mask, q_heads, history, layer_base);
            wait (done_valid);
            if (done_error || done_status != 8'd0)
                $fatal(1, "attention completion error status=%h",
                       done_status);
            want_commands = history == 0 ? 0 : active_groups;
            if (k_command_count != want_commands ||
                v_command_count != want_commands)
                $fatal(1, "mover command count K=%0d V=%0d expected=%0d",
                       k_command_count, v_command_count, want_commands);
            if (q8_write_count != expected_q8_writes)
                $fatal(1, "Q8 writes %0d expected %0d",
                       q8_write_count, expected_q8_writes);
            check_q8 = 1'b0;
            @(posedge clk);
            wait (cmd_ready);
        end
    endtask

    initial begin
        repeat (8) @(posedge clk);
        rst_n = 1'b1;

        // Decode bypasses both history movers and still executes both groups.
        run_good(4'd1, 8'h01, 6'd16, 17'd0,
                 64'h0000_0000_0010_0000);

        // Abort a live tile-8 history transaction, quarantine arena responses,
        // then prove a clean restart on the largest group shape.
        k_command_count = 0;
        v_command_count = 0;
        launch(4'd8, 8'hff, 6'd32, 17'd2,
               64'h0000_0000_0020_0000);
        wait (k_command_count != 0 && query_request_count > 2100);
        @(negedge clk);
        abort_run = 1'b1;
        @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        wait (cmd_ready);
        if (done_valid || busy)
            $fatal(1, "abort did not return the service to idle");

        // A mover failure aborts every sibling exactly once, reports a
        // source-tagged service status, and leaves no state for the restart.
        k_command_count = 0;
        v_command_count = 0;
        inject_v_done_error = 1'b1;
        launch(4'd1, 8'h01, 6'd16, 17'd1,
               64'h0000_0000_0028_0000);
        wait (done_valid);
        if (!done_error || done_status != 8'h42)
            $fatal(1, "V mover error status=%h error=%0d",
                   done_status, done_error);
        inject_v_done_error = 1'b0;
        @(posedge clk);
        wait (cmd_ready);

        run_good(4'd8, 8'hff, 6'd32, 17'd1,
                 64'h0000_0000_0030_0000);

        // The public contract rejects non-prefix masks before any child runs.
        launch(4'd2, 8'h05, 6'd16, 17'd0,
               64'h0000_0000_0040_0000);
        wait (done_valid);
        if (!done_error || done_status != 8'h01)
            $fatal(1, "invalid command status=%h error=%0d",
                   done_status, done_error);

        $display(" attention_service PASS cycles=%0d query=%0d newkv=%0d",
                 cycles, query_request_count, newkv_request_count);
        $finish;
    end
endmodule

`default_nettype wire
