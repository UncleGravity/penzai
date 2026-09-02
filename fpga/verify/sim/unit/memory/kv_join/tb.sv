`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [16:0] cmd_history_len = 17'd0;
    reg [3:0] cmd_token_count = 4'd0;
    reg [3:0] cmd_kv_head_base = 4'd0;
    reg [2:0] cmd_kv_head_count = 3'd0;

    reg hist_k_enable = 1'b0;
    reg hist_v_enable = 1'b0;
    reg force_k_stream_error = 1'b0;
    reg force_v_stream_error = 1'b0;
    reg force_bad_k_last = 1'b0;
    integer hist_k_index = 0;
    integer hist_v_index = 0;
    integer expected_history_beats = 0;
    wire hist_k_valid;
    wire hist_k_ready;
    wire hist_v_valid;
    wire hist_v_ready;
    wire [127:0] hist_k_data;
    wire [127:0] hist_v_data;
    wire hist_k_last;
    wire hist_v_last;

    wire arena_req_valid;
    wire arena_req_ready;
    wire arena_req_wave;
    wire [10:0] arena_req_addr;
    reg arena_rsp_valid = 1'b0;
    wire arena_rsp_ready;
    reg [63:0] arena_rsp_data = 64'd0;
    reg arena_rsp_error = 1'b0;
    reg inject_arena_error = 1'b0;
    integer arena_req_count = 0;

    wire [127:0] k_data;
    wire k_valid;
    wire k_ready;
    wire k_last;
    wire [127:0] v_data;
    wire v_valid;
    wire v_ready;
    wire v_last;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

    integer cycles = 0;
    integer k_output_index = 0;
    integer v_output_index = 0;
    integer expected_total_beats = 0;
    integer active_history = 0;
    integer active_tokens = 0;
    integer active_head_base = 0;
    integer active_head_count = 0;
    reg check_outputs = 1'b0;
    reg k_stalled_q = 1'b0;
    reg v_stalled_q = 1'b0;
    reg [127:0] k_stall_data_q = 128'd0;
    reg [127:0] v_stall_data_q = 128'd0;
    reg k_stall_last_q = 1'b0;
    reg v_stall_last_q = 1'b0;

    function automatic [127:0] history_word(
        input integer kind,
        input integer index
    );
        reg [31:0] tag;
        begin
            tag = kind ? (32'h7600_0000 + index) :
                         (32'h6b00_0000 + index);
            history_word = {4{tag}};
        end
    endfunction

    function automatic [15:0] arena_word(
        input integer kind,
        input integer head,
        input integer token,
        input integer dim
    );
        begin
            arena_word = {2'b10, kind[0], head[2:0], token[2],
                          token[1:0], dim[6:0]};
        end
    endfunction

    function automatic [127:0] local_word(
        input integer kind,
        input integer index
    );
        integer token;
        integer head;
        integer beat;
        integer scalar;
        integer remainder;
        reg [127:0] value;
        begin
            token = index / (active_head_count * 16);
            remainder = index % (active_head_count * 16);
            head = active_head_base + (remainder / 16);
            beat = remainder % 16;
            value = 128'd0;
            for (scalar = 0; scalar < 8; scalar = scalar + 1)
                value[scalar*16 +: 16] = arena_word(
                    kind, head, token, beat*8 + scalar);
            local_word = value;
        end
    endfunction

    wire [127:0] expected_k_word =
        (k_output_index < expected_history_beats) ?
            history_word(0, k_output_index) :
            local_word(0, k_output_index - expected_history_beats);
    wire [127:0] expected_v_word =
        (v_output_index < expected_history_beats) ?
            history_word(1, v_output_index) :
            local_word(1, v_output_index - expected_history_beats);

    assign hist_k_valid = hist_k_enable &&
        (hist_k_index < expected_history_beats) && (cycles[2:0] != 3'd2);
    assign hist_v_valid = hist_v_enable &&
        (hist_v_index < expected_history_beats) && (cycles[2:0] != 3'd5);
    assign hist_k_data = history_word(0, hist_k_index);
    assign hist_v_data = history_word(1, hist_v_index);
    assign hist_k_last = force_bad_k_last ? (hist_k_index == 3) :
        (hist_k_index + 1 == expected_history_beats);
    assign hist_v_last = hist_v_index + 1 == expected_history_beats;

    assign arena_req_ready = (!arena_rsp_valid || arena_rsp_ready) &&
                             (cycles[2:0] != 3'd3);
    assign k_ready = cycles[2:0] != 3'd6;
    assign v_ready = cycles[1:0] != 2'd1;

     kv_join8 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_history_len(cmd_history_len),
        .cmd_token_count(cmd_token_count),
        .cmd_kv_head_base(cmd_kv_head_base),
        .cmd_kv_head_count(cmd_kv_head_count),
        .hist_k_data(hist_k_data), .hist_k_valid(hist_k_valid),
        .hist_k_ready(hist_k_ready), .hist_k_last(hist_k_last),
        .hist_k_error(force_k_stream_error),
        .hist_v_data(hist_v_data), .hist_v_valid(hist_v_valid),
        .hist_v_ready(hist_v_ready), .hist_v_last(hist_v_last),
        .hist_v_error(force_v_stream_error),
        .newkv_rd_req_valid(arena_req_valid),
        .newkv_rd_req_ready(arena_req_ready),
        .newkv_rd_req_wave(arena_req_wave),
        .newkv_rd_req_addr(arena_req_addr),
        .newkv_rd_rsp_valid(arena_rsp_valid),
        .newkv_rd_rsp_ready(arena_rsp_ready),
        .newkv_rd_rsp_data(arena_rsp_data),
        .newkv_rd_rsp_error(arena_rsp_error),
        .k_data(k_data), .k_valid(k_valid), .k_ready(k_ready),
        .k_last(k_last),
        .v_data(v_data), .v_valid(v_valid), .v_ready(v_ready),
        .v_last(v_last),
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status)
    );

    task automatic issue_command(
        input [16:0] history,
        input [3:0] tokens,
        input [3:0] head_base,
        input [2:0] head_count
    );
        begin
            @(negedge clk);
            cmd_history_len = history;
            cmd_token_count = tokens;
            cmd_kv_head_base = head_base;
            cmd_kv_head_count = head_count;
            cmd_valid = 1'b1;
            do @(posedge clk); while (!cmd_ready);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic prepare_good(
        input integer history,
        input integer tokens,
        input integer head_base,
        input integer head_count
    );
        begin
            active_history = history;
            active_tokens = tokens;
            active_head_base = head_base;
            active_head_count = head_count;
            expected_history_beats = history * head_count * 16;
            expected_total_beats = (history + tokens) * head_count * 16;
            hist_k_index = 0;
            hist_v_index = 0;
            k_output_index = 0;
            v_output_index = 0;
            arena_req_count = 0;
            force_k_stream_error = 1'b0;
            force_v_stream_error = 1'b0;
            force_bad_k_last = 1'b0;
            inject_arena_error = 1'b0;
            hist_k_enable = 1'b1;
            hist_v_enable = 1'b1;
            check_outputs = 1'b1;
        end
    endtask

    task automatic run_good(
        input integer history,
        input integer tokens,
        input integer head_base,
        input integer head_count
    );
        integer expected_arena_requests;
        begin
            prepare_good(history, tokens, head_base, head_count);
            expected_arena_requests = tokens * head_count * 16 * 8 * 2;
            issue_command(history[16:0], tokens[3:0],
                          head_base[3:0], head_count[2:0]);
            wait (done_valid);
            if (done_error || done_status != 8'h00)
                $fatal(1, "unexpected completion error %h", done_status);
            if (k_output_index != expected_total_beats ||
                v_output_index != expected_total_beats)
                $fatal(1, "output counts K=%0d V=%0d expected=%0d",
                       k_output_index, v_output_index, expected_total_beats);
            if (hist_k_index != expected_history_beats ||
                hist_v_index != expected_history_beats)
                $fatal(1, "history counts K=%0d V=%0d expected=%0d",
                       hist_k_index, hist_v_index, expected_history_beats);
            if (arena_req_count != expected_arena_requests)
                $fatal(1, "arena requests %0d expected %0d",
                       arena_req_count, expected_arena_requests);
            check_outputs = 1'b0;
            hist_k_enable = 1'b0;
            hist_v_enable = 1'b0;
            @(posedge clk);
            wait (cmd_ready);
        end
    endtask

    task automatic expect_command_error(input [7:0] status);
        begin
            wait (done_valid);
            if (!done_error || done_status != status)
                $fatal(1, "expected error %h got error=%0d status=%h",
                       status, done_error, done_status);
            @(posedge clk);
            wait (cmd_ready);
        end
    endtask

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (!rst_n) begin
            arena_rsp_valid <= 1'b0;
            arena_rsp_error <= 1'b0;
        end else begin
            if (arena_rsp_valid && arena_rsp_ready) begin
                arena_rsp_valid <= 1'b0;
                arena_rsp_error <= 1'b0;
            end
            if (arena_req_valid && arena_req_ready) begin
                if (arena_req_addr[9:7] + 1 > 8)
                    $fatal(1, "arena head out of range");
                arena_rsp_valid <= 1'b1;
                arena_rsp_data[15:0] <= arena_word(
                    arena_req_addr[10], arena_req_addr[9:7],
                    {arena_req_wave, 2'd0}, arena_req_addr[6:0]);
                arena_rsp_data[31:16] <= arena_word(
                    arena_req_addr[10], arena_req_addr[9:7],
                    {arena_req_wave, 2'd1}, arena_req_addr[6:0]);
                arena_rsp_data[47:32] <= arena_word(
                    arena_req_addr[10], arena_req_addr[9:7],
                    {arena_req_wave, 2'd2}, arena_req_addr[6:0]);
                arena_rsp_data[63:48] <= arena_word(
                    arena_req_addr[10], arena_req_addr[9:7],
                    {arena_req_wave, 2'd3}, arena_req_addr[6:0]);
                arena_rsp_error <= inject_arena_error;
                arena_req_count <= arena_req_count + 1;
            end
        end

        if (hist_k_valid && hist_k_ready)
            hist_k_index <= hist_k_index + 1;
        if (hist_v_valid && hist_v_ready)
            hist_v_index <= hist_v_index + 1;

        if (rst_n && check_outputs && k_valid && k_ready) begin
            if (k_data !== expected_k_word)
                $fatal(1, "K mismatch n=%0d got=%h expected=%h",
                       k_output_index, k_data, expected_k_word);
            if (k_last != (k_output_index + 1 == expected_total_beats))
                $fatal(1, "K last mismatch n=%0d", k_output_index);
            k_output_index <= k_output_index + 1;
        end
        if (rst_n && check_outputs && v_valid && v_ready) begin
            if (v_data !== expected_v_word)
                $fatal(1, "V mismatch n=%0d got=%h expected=%h",
                       v_output_index, v_data, expected_v_word);
            if (v_last != (v_output_index + 1 == expected_total_beats))
                $fatal(1, "V last mismatch n=%0d", v_output_index);
            v_output_index <= v_output_index + 1;
        end

        if (rst_n && check_outputs && !clear && !abort_run) begin
            if (k_stalled_q && (!k_valid || k_data !== k_stall_data_q ||
                                k_last !== k_stall_last_q))
                $fatal(1, "K output changed while stalled");
            if (v_stalled_q && (!v_valid || v_data !== v_stall_data_q ||
                                v_last !== v_stall_last_q))
                $fatal(1, "V output changed while stalled");
        end
        k_stalled_q <= rst_n && check_outputs && !clear && !abort_run &&
                       k_valid && !k_ready;
        v_stalled_q <= rst_n && check_outputs && !clear && !abort_run &&
                       v_valid && !v_ready;
        if (k_valid && !k_ready) begin
            k_stall_data_q <= k_data;
            k_stall_last_q <= k_last;
        end
        if (v_valid && !v_ready) begin
            v_stall_data_q <= v_data;
            v_stall_last_q <= v_last;
        end

        if (cycles > 300000)
            $fatal(1, "timeout");
    end

    initial begin
        repeat (6) @(posedge clk);
        rst_n = 1'b1;

        // Two-group shape: four KV heads, one local token after history.
        run_good(2, 1, 4, 4);
        // Four-group shape: two KV heads, both tile-8 waves, independent stalls.
        run_good(1, 8, 6, 2);
        // No-history decode proves the local source starts directly.
        run_good(0, 1, 0, 2);

        // Exact context arithmetic accepts the final legal tile-8 request. Cancel it
        // before intentionally streaming four million history records.
        prepare_good(65528, 8, 0, 2);
        hist_k_enable = 1'b0;
        hist_v_enable = 1'b0;
        check_outputs = 1'b0;
        issue_command(17'd65528, 4'd8, 4'd0, 3'd2);
        if (!busy || done_valid)
            $fatal(1, "64K boundary command was not accepted");
        @(negedge clk);
        clear = 1'b1;
        repeat (3) @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        wait (cmd_ready);

        issue_command(17'd65535, 4'd2, 4'd0, 3'd2);
        expect_command_error(8'h01);
        issue_command(17'd0, 4'd1, 4'd1, 3'd4);
        expect_command_error(8'h01);

        // Early last is an exact-count failure.
        prepare_good(1, 1, 0, 2);
        force_bad_k_last = 1'b1;
        check_outputs = 1'b0;
        issue_command(17'd1, 4'd1, 4'd0, 3'd2);
        expect_command_error(8'h02);
        hist_k_enable = 1'b0;
        hist_v_enable = 1'b0;
        force_bad_k_last = 1'b0;

        // Mover and arena errors terminate with source-specific status.
        prepare_good(1, 1, 0, 2);
        force_v_stream_error = 1'b1;
        check_outputs = 1'b0;
        issue_command(17'd1, 4'd1, 4'd0, 3'd2);
        expect_command_error(8'h03);
        hist_k_enable = 1'b0;
        hist_v_enable = 1'b0;
        force_v_stream_error = 1'b0;

        prepare_good(0, 1, 0, 2);
        inject_arena_error = 1'b1;
        check_outputs = 1'b0;
        issue_command(17'd0, 4'd1, 4'd0, 3'd2);
        expect_command_error(8'h04);
        inject_arena_error = 1'b0;

        // Abort while a local read is active must drain it and return cleanly.
        prepare_good(0, 8, 0, 2);
        issue_command(17'd0, 4'd8, 4'd0, 3'd2);
        wait (arena_req_count >= 5);
        @(negedge clk);
        abort_run = 1'b1;
        check_outputs = 1'b0;
        repeat (3) @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        wait (cmd_ready);
        run_good(0, 1, 2, 2);

        $display(" kv_join8 PASS cycles=%0d", cycles);
        $finish;
    end
endmodule

`default_nettype wire
