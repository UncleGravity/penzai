`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cfg_valid = 1'b0;
    wire cfg_ready;
    reg [3:0] cfg_token_count = 4'd5;
    reg [7:0] cfg_token_mask = 8'h1f;
    reg [5:0] cfg_q_heads = 6'd16;
    reg [255:0] in_data = 256'd0;
    reg in_valid = 1'b0;
    wire in_ready;
    reg [2:0] in_token = 3'd0;
    reg [5:0] in_head = 6'd0;
    reg [3:0] in_beat = 4'd0;
    reg in_group_last = 1'b0;
    reg in_last = 1'b0;
    wire q8_wr_valid;
    reg q8_wr_ready = 1'b1;
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
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

     flash_output_q8 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_token_count(cfg_token_count),
        .cfg_token_mask(cfg_token_mask), .cfg_q_heads(cfg_q_heads),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
        .in_token(in_token), .in_head(in_head), .in_beat(in_beat),
        .in_group_last(in_group_last), .in_last(in_last),
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
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status)
    );

     q8_pack4 u_q8 (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(leaf_q8_cfg_valid), .cfg_ready(leaf_q8_cfg_ready),
        .cfg_rows(leaf_q8_cfg_rows),
        .cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .abort_run(leaf_q8_abort), .busy(leaf_q8_busy),
        .in_valid(leaf_q8_in_valid), .in_ready(leaf_q8_in_ready),
        .in_data(leaf_q8_in_data), .out_valid(leaf_q8_out_valid),
        .out_ready(leaf_q8_out_ready), .out_block(leaf_q8_out_block),
        .out_data(leaf_q8_out_data), .out_status(leaf_q8_out_status),
        .out_last(leaf_q8_out_last)
    );

    integer cycles = 0;
    integer input_count = 0;
    integer write_count = 0;
    integer group_i;
    integer token_i;
    integer head_i;
    integer beat_i;
    integer expected_group;
    integer expected_block;
    integer expected_wave;

    always @(posedge clk) begin
        cycles <= cycles + 1;
        q8_wr_ready <= (cycles % 7) != 3;
        if (in_valid && in_ready)
            input_count <= input_count + 1;
        if (q8_wr_valid && q8_wr_ready) begin
            expected_group = write_count / 64;
            expected_wave = (write_count / 32) % 2;
            expected_block = write_count % 32;
            if (q8_wr_wave != expected_wave[0] ||
                q8_wr_addr != expected_group * 32 + expected_block)
                $fatal(1, "Q8 address/order mismatch count=%0d", write_count);
            if (q8_wr_lane_mask != (expected_wave ?
                                    cfg_token_mask[7:4] :
                                    cfg_token_mask[3:0]))
                $fatal(1, "Q8 lane mask mismatch count=%0d", write_count);
            if (q8_wr_data != 1088'd0)
                $fatal(1, "zero Flash output did not quantize to zero");
            write_count <= write_count + 1;
        end
    end

    task automatic start_run;
        begin
            while (!cfg_ready) @(posedge clk);
            @(negedge clk);
            cfg_valid = 1'b1;
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task automatic drive_flash_records;
        input integer stop_after;
        integer sent;
        integer target;
        begin
            sent = 0;
            target = stop_after == 0 ? 2 * cfg_token_count * 128 :
                                       stop_after;
            for (group_i = 0; group_i < 2 && sent < target;
                 group_i = group_i + 1)
                for (token_i = 0; token_i < cfg_token_count && sent < target;
                     token_i = token_i + 1)
                    for (head_i = 0; head_i < 8 && sent < target;
                         head_i = head_i + 1)
                        for (beat_i = 0; beat_i < 16 && sent < target;
                             beat_i = beat_i + 1) begin
                            while (!in_ready) @(posedge clk);
                            @(negedge clk);
                            in_valid = 1'b1;
                            in_data = 256'd0;
                            in_token = token_i[2:0];
                            in_head = group_i * 8 + head_i;
                            in_beat = beat_i[3:0];
                            in_group_last =
                                            (token_i[3:0] ==
                                             cfg_token_count - 1'b1) &&
                                            (head_i == 7) && (beat_i == 15);
                            in_last = in_group_last && (group_i == 1);
                            @(negedge clk);
                            in_valid = 1'b0;
                            sent = sent + 1;
                        end
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        start_run();
        drive_flash_records(40);
        @(negedge clk);
        abort_run = 1'b1;
        @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        repeat (5) @(posedge clk);
        if (busy || done_valid)
            $fatal(1, "abort failed to return idle");

        // A clear may coincide with the local write intent retirement. The
        // payload write is harmless, but no valid/control state may survive.
        cfg_token_count = 4'd1;
        cfg_token_mask = 8'h01;
        start_run();
        while (!in_ready) @(posedge clk);
        @(negedge clk);
        in_valid = 1'b1;
        in_token = 3'd0;
        in_head = 6'd0;
        in_beat = 4'd0;
        in_group_last = 1'b0;
        in_last = 1'b0;
        @(negedge clk);
        in_valid = 1'b0;
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        repeat (2) @(posedge clk);
        if (busy || done_valid || dut.ingest_wr_valid_q)
            $fatal(1, "clear retained Flash write intent/state");

        // Clear after a single-token wave has entered the BRAM-output/held pipeline.
        start_run();
        drive_flash_records(128);
        wait (dut.drain_pending_q || dut.drain_mem_valid_q ||
              dut.drain_valid_q);
        @(negedge clk);
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        repeat (3) @(posedge clk);
        if (busy || done_valid || dut.drain_pending_q ||
            dut.drain_mem_valid_q || dut.drain_valid_q ||
            dut.ingest_wr_valid_q)
            $fatal(1, "clear retained Flash read pipeline/state");

        input_count = 0;
        write_count = 0;
        cfg_token_count = 4'd5;
        cfg_token_mask = 8'h1f;
        start_run();
        drive_flash_records(0);
        wait (done_valid);
        if (done_error || done_status != 0)
            $fatal(1, "Flash Q8 adapter failed status=%h", done_status);
        if (input_count != 1280 || write_count != 128)
            $fatal(1, "count mismatch in=%0d out=%0d",
                   input_count, write_count);

        $display(" flash_output_q8 PASS cycles=%0d", cycles);
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1, "timeout");
    end
endmodule
