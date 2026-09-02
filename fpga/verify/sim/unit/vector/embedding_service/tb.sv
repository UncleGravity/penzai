`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;

    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [3:0] cmd_token_count = 4'd2;
    reg [7:0] cmd_token_mask = 8'h03;
    reg [255:0] cmd_token_ids = 256'd0;

    wire read_cmd_valid;
    reg read_cmd_ready = 1'b1;
    wire [63:0] read_cmd_base_addr;
    wire [31:0] read_cmd_port_beats;
    wire [3:0] read_cmd_port_mask;
    wire read_abort;
    reg [511:0] read_data = 512'd0;
    reg read_valid = 1'b0;
    wire read_ready;
    reg read_last = 1'b0;
    reg read_error = 1'b0;
    reg read_busy = 1'b0;
    reg read_done_valid = 1'b0;
    wire read_done_ready;
    reg read_done_error = 1'b0;
    reg [7:0] read_done_status = 8'd0;

    wire r_wr_valid;
    reg r_wr_ready = 1'b1;
    wire r_wr_wave;
    wire [11:0] r_wr_addr;
    wire [3:0] r_wr_lane_mask;
    wire [127:0] r_wr_data;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

    localparam [63:0] BASE = 64'h1000;

     embedding_service dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_table_addr(BASE), .cmd_q1_blocks(6'd1),
        .cmd_hidden_dim(13'd128), .cmd_vocab_rows(18'd16),
        .cmd_weight_fmt(2'd1), .cmd_token_count(cmd_token_count),
        .cmd_token_mask(cmd_token_mask), .cmd_token_ids(cmd_token_ids),
        .read_cmd_valid(read_cmd_valid), .read_cmd_ready(read_cmd_ready),
        .read_cmd_base_addr(read_cmd_base_addr),
        .read_cmd_port_beats(read_cmd_port_beats),
        .read_cmd_port_mask(read_cmd_port_mask), .read_abort(read_abort),
        .read_data(read_data), .read_valid(read_valid),
        .read_ready(read_ready), .read_last(read_last),
        .read_error(read_error), .read_busy(read_busy),
        .read_done_valid(read_done_valid),
        .read_done_ready(read_done_ready),
        .read_done_error(read_done_error),
        .read_done_status(read_done_status),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_wave(r_wr_wave), .r_wr_addr(r_wr_addr),
        .r_wr_lane_mask(r_wr_lane_mask), .r_wr_data(r_wr_data),
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status)
    );

    reg pending_rsp = 1'b0;
    reg pending_done = 1'b0;
    reg [63:0] pending_addr = 64'd0;
    integer writes = 0;
    integer commands = 0;
    integer cycles = 0;

    always @(posedge clk) begin
        cycles <= cycles + 1;
        read_valid <= 1'b0;
        read_last <= 1'b0;
        read_done_valid <= 1'b0;

        if (read_abort) begin
            pending_rsp <= 1'b0;
            pending_done <= 1'b0;
            read_busy <= 1'b0;
        end else begin
            if (read_cmd_valid && read_cmd_ready) begin
                if (read_cmd_port_beats != 1 || read_cmd_port_mask != 4'h1)
                    $fatal(1, "bad shared-reader command");
                pending_addr <= read_cmd_base_addr;
                pending_rsp <= 1'b1;
                read_busy <= 1'b1;
                commands <= commands + 1;
            end
            if (pending_rsp && !read_valid) begin
                read_data <= 512'd0;
                if (pending_addr == BASE) begin
                    read_data[31:0] <= 32'h0000_3c00;
                    read_data[63:32] <= 32'h0000_3c00;
                end else begin
                    read_data[31:0] <= 32'hffff_ffff;
                    read_data[63:32] <= 32'h0000_0000;
                end
                read_valid <= 1'b1;
                read_last <= 1'b1;
            end
            if (read_valid && read_ready) begin
                pending_rsp <= 1'b0;
                pending_done <= 1'b1;
            end
            if (pending_done) begin
                read_done_valid <= 1'b1;
                if (read_done_ready) begin
                    pending_done <= 1'b0;
                    read_busy <= 1'b0;
                end
            end
        end

        if (r_wr_valid && r_wr_ready) begin
            if (r_wr_wave != 1'b0 || r_wr_addr != (writes % 128))
                $fatal(1, "R address/order mismatch write=%0d", writes);
            if (writes < 128) begin
                if (r_wr_lane_mask != 4'h1 || r_wr_data[31:0] != 32'h3f80_0000)
                    $fatal(1, "token0 embedding mismatch");
            end else begin
                if (r_wr_lane_mask != 4'h2 || r_wr_data[63:32] != 32'hbf80_0000)
                    $fatal(1, "token1 embedding mismatch");
            end
            writes <= writes + 1;
        end
    end

    task automatic launch;
        begin
            while (!cmd_ready) @(posedge clk);
            cmd_valid <= 1'b1;
            @(posedge clk);
            cmd_valid <= 1'b0;
        end
    endtask

    initial begin
        cmd_token_ids[31:0] = 32'd0;
        cmd_token_ids[63:32] = 32'd1;
        repeat (4) @(posedge clk);
        rst_n <= 1'b1;

        // The decoder may advance into its response state while arbitration is
        // stalled, but the registered reader command must remain exact and a
        // live clear must discard it without leaking an external transaction.
        read_cmd_ready = 1'b0;
        launch();
        while (!read_cmd_valid) @(posedge clk);
        if (read_cmd_base_addr != BASE)
            $fatal(1, "registered embedding address mismatch");
        repeat (4) begin
            @(posedge clk);
            if (!read_cmd_valid || read_cmd_base_addr != BASE)
                $fatal(1, "stalled embedding command was not stable");
        end
        @(negedge clk);
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        read_cmd_ready = 1'b1;
        repeat (3) @(posedge clk);
        if (busy || done_valid || read_busy || commands != 0)
            $fatal(1, "clear leaked a pending embedding transaction");

        writes = 0;
        commands = 0;
        launch();
        wait (writes >= 20);
        abort_run <= 1'b1;
        @(posedge clk);
        abort_run <= 1'b0;
        repeat (4) @(posedge clk);
        if (busy || done_valid || read_busy)
            $fatal(1, "abort did not return service to idle");

        writes = 0;
        commands = 0;
        launch();
        wait (done_valid);
        if (done_error || done_status != 0 || writes != 256)
            $fatal(1, "embedding service failed status=%h writes=%0d",
                   done_status, writes);
        if (commands != 10)
            $fatal(1, "unexpected packed-table read count %0d", commands);

        $display(" embedding_service PASS cycles=%0d", cycles);
        $finish;
    end

    initial begin
        #2000000;
        $display("timeout state=%0d read_state=%0d decode_state=%0d busy=%0d read_busy=%0d pending_rsp=%0d pending_done=%0d writes=%0d commands=%0d",
                 dut.state_q, dut.read_state_q, dut.u_decode.state_q, busy,
                 read_busy, pending_rsp, pending_done, writes, commands);
        $fatal(1, "timeout");
    end
endmodule
