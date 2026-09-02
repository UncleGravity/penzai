`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg cmd_valid = 1'b0;
    reg [3:0] cmd_token_count = 4'd0;
    reg [5:0] cmd_q_heads = 6'd0;
    reg [16:0] cmd_history_len = 17'd0;
    wire cmd_ready;

    wire group_valid;
    wire [1:0] group_index;
    wire [5:0] group_q_base;
    wire [3:0] group_kv_base;
    wire [2:0] group_kv_heads;
    wire [16:0] group_total_kv;

    integer cycles = 0;
    integer q_count = 0;
    integer k_count = 0;
    integer v_count = 0;
    integer q_expected = 0;
    integer k_expected = 0;
    integer v_expected = 0;
    integer q_total = 0;
    integer k_total = 0;
    integer v_total = 0;
    integer group_count = 0;
    integer out_count = 0;
    integer expected_groups = 0;
    integer expected_outputs = 0;
    integer expected_token;
    integer expected_head;
    integer expected_group;
    reg group_active = 1'b0;

    wire [31:0] q_word = 32'h3f000000 + q_count;

    wire q_valid = group_active && q_count < q_expected;
    wire k_valid = group_active && k_count < k_expected;
    wire v_valid = group_active && v_count < v_expected;
    wire q_ready, k_ready, v_ready;
    wire [255:0] out_data;
    wire out_valid;
    wire out_ready = cycles[1:0] != 2'b11;
    wire [2:0] out_token;
    wire [5:0] out_head;
    wire [3:0] out_beat;
    wire out_group_last;
    wire out_last;
    wire busy;
    wire done_valid;
    wire done_error;
    wire [7:0] done_status;

    localparam integer TEST_HEAD_BEATS = 2;

    flash_groups8 #(.HEAD_DIM(16)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_token_count(cmd_token_count), .cmd_q_heads(cmd_q_heads),
        .cmd_history_len(cmd_history_len), .cmd_scale(32'h3f000000),
        .group_req_valid(group_valid), .group_req_ready(1'b1),
        .group_req_index(group_index),
        .group_req_q_head_base(group_q_base),
        .group_req_kv_head_base(group_kv_base),
        .group_req_kv_heads(group_kv_heads),
        .group_req_total_kv(group_total_kv),
        .q_tdata({8{q_word}}), .q_tvalid(q_valid), .q_tready(q_ready),
        .k_tdata(128'd0), .k_tvalid(k_valid), .k_tready(k_ready),
        .v_tdata({8{16'h3c00}}), .v_tvalid(v_valid), .v_tready(v_ready),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),
        .out_token(out_token), .out_head(out_head), .out_beat(out_beat),
        .out_group_last(out_group_last), .out_last(out_last),
        .busy(busy), .done_valid(done_valid), .done_ready(1'b1),
        .done_error(done_error), .done_status(done_status)
    );

    task automatic reset_counters(input integer groups, input integer outputs);
        begin
            q_count = 0; k_count = 0; v_count = 0;
            q_expected = 0; k_expected = 0; v_expected = 0;
            q_total = 0; k_total = 0; v_total = 0;
            group_count = 0; out_count = 0;
            expected_groups = groups; expected_outputs = outputs;
            group_active = 1'b0;
        end
    endtask

    task automatic check_q_map;
        integer token;
        integer head;
        integer beat;
        integer address;
        integer seq_index;
        begin
            for (head = 0; head < 8; head = head + 1)
                for (beat = 0; beat < TEST_HEAD_BEATS; beat = beat + 1)
                    for (token = 0; token < cmd_token_count; token = token + 1) begin
                        address = (token * 8 + head) * TEST_HEAD_BEATS + beat;
                        seq_index = (head * TEST_HEAD_BEATS + beat) *
                                    cmd_token_count + token;
                        if (dut.u_flash.q_buf[address][31:0] !==
                            (32'h3f000000 + seq_index))
                            $fatal(1, "four-lane Q load order mismatch addr=%0d", address);
                    end
        end
    endtask

    task automatic run_good(
        input [5:0] heads,
        input [3:0] tokens,
        input [16:0] history,
        input integer groups
    );
        begin
            reset_counters(groups, heads * tokens * TEST_HEAD_BEATS);
            @(negedge clk);
            cmd_q_heads = heads;
            cmd_token_count = tokens;
            cmd_history_len = history;
            cmd_valid = 1'b1;
            do @(posedge clk); while (!cmd_ready);
            @(negedge clk);
            cmd_valid = 1'b0;
            wait (done_valid);
            check_q_map();
            if (done_error)
                $fatal(1, "unexpected service error %h", done_status);
            if (group_count != expected_groups || out_count != expected_outputs)
                $fatal(1, "count mismatch groups=%0d/%0d outputs=%0d/%0d",
                    group_count, expected_groups, out_count, expected_outputs);
            if (q_total != groups * tokens * 8 * TEST_HEAD_BEATS)
                $fatal(1, "Q total mismatch %0d", q_total);
            if (k_total != groups * (history + tokens) *
                           (8 / groups) * TEST_HEAD_BEATS)
                $fatal(1, "K total mismatch %0d", k_total);
            if (v_total != k_total)
                $fatal(1, "V total mismatch %0d/%0d", v_total, k_total);
            @(posedge clk);
        end
    endtask

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (rst_n && !clear) begin
            if (group_valid) begin
                if (group_count != 0)
                    check_q_map();
                if (group_index != group_count[1:0])
                    $fatal(1, "group index mismatch %0d/%0d", group_index, group_count);
                if (group_q_base != group_count * 8)
                    $fatal(1, "Q head base mismatch");
                if (group_kv_base != group_count * (8 / expected_groups))
                    $fatal(1, "KV head base mismatch");
                if (group_kv_heads != 8 / expected_groups)
                    $fatal(1, "KV group width mismatch");
                q_count <= 0;
                k_count <= 0;
                v_count <= 0;
                q_expected <= cmd_token_count * 8 * TEST_HEAD_BEATS;
                k_expected <= group_total_kv * group_kv_heads * TEST_HEAD_BEATS;
                v_expected <= group_total_kv * group_kv_heads * TEST_HEAD_BEATS;
                group_active <= 1'b1;
                group_count <= group_count + 1;
            end
            if (q_valid && q_ready) begin
                q_count <= q_count + 1;
                q_total <= q_total + 1;
            end
            if (k_valid && k_ready) begin
                k_count <= k_count + 1;
                k_total <= k_total + 1;
            end
            if (v_valid && v_ready) begin
                v_count <= v_count + 1;
                v_total <= v_total + 1;
            end
            if (out_valid && out_ready) begin
                expected_group = out_count /
                                 (cmd_token_count * 8 * TEST_HEAD_BEATS);
                expected_token = (out_count / (8 * TEST_HEAD_BEATS)) %
                                 cmd_token_count;
                expected_head = expected_group * 8 +
                                ((out_count / TEST_HEAD_BEATS) % 8);
                if (out_token != expected_token || out_head != expected_head ||
                    out_beat != (out_count % TEST_HEAD_BEATS))
                    $fatal(1, "output tag mismatch n=%0d tok=%0d head=%0d beat=%0d",
                        out_count, out_token, out_head, out_beat);
                if (out_data !== {8{32'h3f800000}})
                    $fatal(1, "output numeric mismatch n=%0d data=%h", out_count, out_data);
                if (out_group_last != ((out_count + 1) %
                    (cmd_token_count * 8 * TEST_HEAD_BEATS) == 0))
                    $fatal(1, "group last mismatch n=%0d", out_count);
                if (out_last != (out_count + 1 == expected_outputs))
                    $fatal(1, "command last mismatch n=%0d", out_count);
                out_count <= out_count + 1;
            end
            if (cycles > 300000)
                $fatal(1, "timeout");
        end
    end

    initial begin
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        run_good(6'd16, 4'd2, 17'd1, 2);
        run_good(6'd32, 4'd8, 17'd0, 4);

        reset_counters(0, 0);
        @(negedge clk);
        cmd_q_heads = 6'd32;
        cmd_token_count = 4'd2;
        cmd_history_len = 17'd65535;
        cmd_valid = 1'b1;
        do @(posedge clk); while (!cmd_ready);
        @(negedge clk);
        cmd_valid = 1'b0;
        wait (done_valid);
        if (!done_error || done_status != 8'h03)
            $fatal(1, "context overflow was not rejected");

        $display("flash_groups8 PASS cycles=%0d", cycles);
        $finish;
    end
endmodule

`default_nettype wire
