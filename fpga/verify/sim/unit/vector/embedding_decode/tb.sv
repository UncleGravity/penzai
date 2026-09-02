`timescale 1ns/1ps

module tb;
    localparam [63:0] BASE = 64'h0000_0000_0000_1000;

    reg clk = 1'b0;
    always #1.666 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg start_valid = 1'b0;
    wire start_ready;
    reg [63:0] table_addr = BASE;
    reg [5:0] q1_blocks = 6'd1;
    reg [17:0] vocab_rows = 18'd32;
    reg [1:0] weight_fmt = 2'd1;
    reg [3:0] token_count = 4'd0;
    reg [7:0] token_mask = 8'd0;
    reg [255:0] token_ids = 256'd0;

    wire mem_req_valid;
    reg mem_req_ready;
    wire [63:0] mem_req_addr;
    reg mem_rsp_valid = 1'b0;
    wire mem_rsp_ready;
    reg [127:0] mem_rsp_data = 128'd0;
    reg mem_rsp_error = 1'b0;

    wire out_valid;
    reg out_ready;
    wire [2:0] out_token;
    wire [11:0] out_index;
    wire [127:0] out_data;
    wire out_last;
    wire busy;
    wire done;
    wire error;
    wire [7:0] status;

    reg [7:0] memory [0:4095];
    reg [15:0] scale_lo [0:7];
    reg [15:0] scale_hi [0:7];
    integer cycle = 0;
    integer out_count = 0;
    integer expected_records = 0;
    integer request_count = 0;
    integer i;
    integer j;
    integer token_id_i;
    integer index_i;
    integer code_i;
    integer request_token_i;
    integer request_beat_i;
    reg [63:0] expected_addr_i;
    reg [31:0] bits32;
    reg [63:0] codes64;
    reg saw_done = 1'b0;

     embedding_decode dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .start_valid(start_valid), .start_ready(start_ready),
        .table_addr(table_addr), .q1_blocks(q1_blocks),
        .vocab_rows(vocab_rows), .weight_fmt(weight_fmt),
        .token_count(token_count), .token_mask(token_mask),
        .token_ids(token_ids),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_rsp_valid(mem_rsp_valid),
        .mem_rsp_ready(mem_rsp_ready), .mem_rsp_data(mem_rsp_data),
        .mem_rsp_error(mem_rsp_error),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_token(out_token), .out_index(out_index),
        .out_data(out_data), .out_last(out_last),
        .busy(busy), .done(done), .error(error), .status(status)
    );

    function automatic [31:0] half_to_f32(input [15:0] h);
        reg [7:0] exponent;
        begin
            exponent = {3'd0, h[14:10]} + 8'd112;
            half_to_f32 = (h[14:10] == 0) ? {h[15], 31'd0} :
                          {h[15], exponent, h[9:0], 13'd0};
        end
    endfunction

    function automatic integer block_offset(
        input integer row,
        input integer fmt
    );
        integer rowblocks;
        integer port;
        integer rb;
        integer stride;
        begin
            rowblocks = 2;
            port = (row % 16) / 4;
            rb = row / 16;
            stride = (fmt == 1) ? 80 : 144;
            block_offset = (port * rowblocks + rb) * stride;
        end
    endfunction

    task automatic put32(input integer off, input [31:0] value);
        begin
            memory[off + 0] = value[7:0];
            memory[off + 1] = value[15:8];
            memory[off + 2] = value[23:16];
            memory[off + 3] = value[31:24];
        end
    endtask

    task automatic fill_q1_row(
        input integer row,
        input [15:0] scale
    );
        integer off;
        integer lane;
        integer sub;
        integer k;
        begin
            off = block_offset(row, 1);
            lane = row % 4;
            put32(off + lane * 4, {16'd0, scale});
            for (sub = 0; sub < 4; sub = sub + 1) begin
                bits32 = 32'd0;
                for (k = 0; k < 32; k = k + 1)
                    bits32[k] = ((row + sub * 32 + k) & 1) != 0;
                put32(off + (1 + sub) * 16 + lane * 4, bits32);
            end
        end
    endtask

    task automatic fill_q2_row(
        input integer row,
        input [15:0] lo,
        input [15:0] hi
    );
        integer off;
        integer lane;
        integer sub;
        integer k;
        integer code;
        begin
            off = block_offset(row, 2);
            lane = row % 4;
            put32(off + lane * 4, {hi, lo});
            for (sub = 0; sub < 4; sub = sub + 1) begin
                codes64 = 64'd0;
                for (k = 0; k < 32; k = k + 1) begin
                    code = (row + sub * 32 + k) % 3;
                    codes64[k*2 +: 2] = code[1:0];
                end
                put32(off + (1 + sub * 2) * 16 + lane * 4,
                      codes64[31:0]);
                put32(off + (2 + sub * 2) * 16 + lane * 4,
                      codes64[63:32]);
            end
        end
    endtask

    task automatic launch(input integer fmt, input integer count);
        begin
            @(negedge clk);
            weight_fmt = fmt[1:0];
            token_count = count[3:0];
            token_mask = (9'd1 << count) - 1;
            start_valid = 1'b1;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task automatic await_success;
        integer timeout;
        begin
            timeout = 0;
            saw_done = 1'b0;
            while (!saw_done && timeout < 20000) begin
                @(posedge clk);
                if (done) begin
                    saw_done = 1'b1;
                    if (error) begin
                        $display("unexpected decoder error status=%02x", status);
                        $fatal(1);
                    end
                end
                timeout = timeout + 1;
            end
            if (!saw_done) begin
                $display("embedding timeout busy=%0d", busy);
                $fatal(1);
            end
            if (out_count != expected_records) begin
                $display("record count got=%0d expected=%0d",
                         out_count, expected_records);
                $fatal(1);
            end
        end
    endtask

    always @(*) begin
        mem_req_ready = !mem_rsp_valid && (cycle[1:0] != 2'b01);
        out_ready = (cycle[2:0] != 3'b011);
    end

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (mem_rsp_valid && mem_rsp_ready)
            mem_rsp_valid <= 1'b0;
        if (mem_req_valid && mem_req_ready) begin
            if ((mem_req_addr < BASE) || (mem_req_addr + 16 > BASE + 4096)) begin
                $display("bad memory address %h", mem_req_addr);
                $fatal(1);
            end
            request_beat_i = request_count %
                             ((weight_fmt == 1) ? 5 : 9);
            request_token_i = request_count /
                              ((weight_fmt == 1) ? 5 : 9);
            expected_addr_i = BASE +
                block_offset(token_ids[request_token_i*32 +: 32],
                             weight_fmt) + request_beat_i * 16;
            if (mem_req_addr != expected_addr_i) begin
                $display("embedding address mismatch req=%0d got=%h want=%h",
                         request_count, mem_req_addr, expected_addr_i);
                $fatal(1);
            end
            request_count = request_count + 1;
            for (i = 0; i < 16; i = i + 1)
                mem_rsp_data[i*8 +: 8] <=
                    memory[mem_req_addr - BASE + i];
            mem_rsp_valid <= 1'b1;
        end

        if (out_valid && out_ready) begin
            if (out_token >= token_count) begin
                $display("bad output token %0d", out_token);
                $fatal(1);
            end
            token_id_i = token_ids[out_token*32 +: 32];
            if (out_index != (out_count % 32) * 4) begin
                $display("bad index rec=%0d got=%0d", out_count, out_index);
                $fatal(1);
            end
            for (j = 0; j < 4; j = j + 1) begin
                index_i = out_index + j;
                if (weight_fmt == 1) begin
                    bits32 = half_to_f32(scale_lo[out_token]);
                    if (((token_id_i + index_i) & 1) == 0)
                        bits32 = bits32 ^ 32'h8000_0000;
                end else begin
                    bits32 = half_to_f32(index_i < 64 ?
                                         scale_lo[out_token] :
                                         scale_hi[out_token]);
                    code_i = (token_id_i + index_i) % 3;
                    if (code_i == 0)
                        bits32 = bits32 ^ 32'h8000_0000;
                    else if (code_i == 1)
                        bits32 = 32'd0;
                end
                if (out_data[j*32 +: 32] !== bits32) begin
                    $display("data mismatch token=%0d idx=%0d got=%h want=%h",
                             out_token, index_i,
                             out_data[j*32 +: 32], bits32);
                    $fatal(1);
                end
            end
            if (out_last != (out_count + 1 == expected_records)) begin
                $display("bad last rec=%0d last=%0d", out_count, out_last);
                $fatal(1);
            end
            out_count = out_count + 1;
        end
    end

    initial begin
        for (i = 0; i < 4096; i = i + 1)
            memory[i] = 8'd0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        // Q1: two rows from different rowblocks and physical ports.
        token_ids = 256'd0;
        token_ids[0 +: 32] = 32'd5;
        token_ids[32 +: 32] = 32'd18;
        scale_lo[0] = 16'h3c00;
        scale_lo[1] = 16'h4000;
        fill_q1_row(5, scale_lo[0]);
        fill_q1_row(18, scale_lo[1]);
        out_count = 0;
        request_count = 0;
        expected_records = 64;
        launch(1, 2);
        await_success();
        if (request_count != 10)
            $fatal(1, "Q1 embedding request boundary count mismatch");

        // Abort after the scale request has left the decoder but before its
        // response is consumed.  The stale response must be drained before a
        // new start is accepted.
        token_ids = 256'd0;
        token_ids[0 +: 32] = 32'd5;
        token_mask = 8'h01;
        out_count = 0;
        request_count = 0;
        expected_records = 0;
        launch(1, 1);
        while (!(mem_req_valid && mem_req_ready)) @(posedge clk);
        @(negedge clk);
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        while (!start_ready) @(posedge clk);
        if (busy || mem_rsp_valid)
            $fatal(1, "embedding abort did not drain the memory response");
        out_count = 0;
        request_count = 0;
        expected_records = 32;
        launch(1, 1);
        await_success();
        if (request_count != 5)
            $fatal(1, "Q1 restart request count mismatch");

        // Q2: all eight logical contexts cover both rowblocks and all ports.
        for (i = 0; i < 4096; i = i + 1)
            memory[i] = 8'd0;
        for (i = 0; i < 8; i = i + 1) begin
            token_ids[i*32 +: 32] = i * 5 - (i >= 4 ? 4 : 0);
            scale_lo[i] = 16'h3c00;
            scale_hi[i] = 16'h4000;
            fill_q2_row(token_ids[i*32 +: 32], scale_lo[i], scale_hi[i]);
        end
        out_count = 0;
        request_count = 0;
        expected_records = 256;
        launch(2, 8);
        await_success();
        if (request_count != 72)
            $fatal(1, "Q2 boundary request count mismatch");

        $display(" embedding_decode: Q1 tile-2 + Q2 tile-8 PASS cycles=%0d", cycle);
        $finish;
    end
endmodule
