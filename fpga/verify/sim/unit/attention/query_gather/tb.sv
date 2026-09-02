`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg start_valid = 1'b0;
    reg [5:0] start_head_base = 6'd0;
    reg [3:0] start_tokens = 4'd0;
    wire start_ready;

    wire rd_valid;
    wire rd_ready = cycles[2:0] != 3'b101;
    wire rd_wave;
    wire [11:0] rd_addr;
    reg rsp_valid = 1'b0;
    wire rsp_ready;
    reg [127:0] rsp_data = 128'd0;
    reg pending = 1'b0;
    reg pending_wave = 1'b0;
    reg [11:0] pending_addr = 12'd0;

    wire [255:0] q_data;
    wire q_valid;
    wire q_ready = cycles[1:0] != 2'b10;
    wire busy;
    wire done_valid;
    wire done_error;

    integer cycles = 0;
    integer q_count = 0;
    integer expected_count = 0;
    integer token;
    integer head;
    integer beat;
    integer dim;
    integer global_head;
    reg [31:0] expected_word;

    function automatic [31:0] arena_word(
        input wave,
        input [1:0] lane,
        input [11:0] address
    );
        begin
            arena_word = {wave, lane, 17'd0, address};
        end
    endfunction

    flash_query_gather dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .start_valid(start_valid), .start_ready(start_ready),
        .start_q_head_base(start_head_base),
        .start_token_count(start_tokens),
        .query_rd_req_valid(rd_valid), .query_rd_req_ready(rd_ready),
        .query_rd_req_wave(rd_wave), .query_rd_req_addr(rd_addr),
        .query_rd_rsp_valid(rsp_valid), .query_rd_rsp_ready(rsp_ready),
        .query_rd_rsp_data(rsp_data),
        .q_tdata(q_data), .q_tvalid(q_valid), .q_tready(q_ready),
        .busy(busy), .done_valid(done_valid), .done_ready(1'b1),
        .done_error(done_error)
    );

    task automatic run_case(input [5:0] base, input [3:0] tokens);
        begin
            q_count = 0;
            expected_count = tokens * 8 * 16;
            @(negedge clk);
            start_head_base = base;
            start_tokens = tokens;
            start_valid = 1'b1;
            do @(posedge clk); while (!start_ready);
            @(negedge clk);
            start_valid = 1'b0;
            wait (done_valid);
            if (done_error || q_count != expected_count)
                $fatal(1, "gather completion mismatch error=%0d q=%0d/%0d",
                    done_error, q_count, expected_count);
            @(posedge clk);
        end
    endtask

    always @(posedge clk) begin
        cycles <= cycles + 1;

        if (!rst_n || clear) begin
            pending <= 1'b0;
            rsp_valid <= 1'b0;
        end else begin
            if (rsp_ready)
                rsp_valid <= pending;
            if (rsp_ready && pending) begin
                rsp_data[31:0] <= arena_word(pending_wave, 2'd0, pending_addr);
                rsp_data[63:32] <= arena_word(pending_wave, 2'd1, pending_addr);
                rsp_data[95:64] <= arena_word(pending_wave, 2'd2, pending_addr);
                rsp_data[127:96] <= arena_word(pending_wave, 2'd3, pending_addr);
            end
            if (rsp_ready)
                pending <= rd_valid && rd_ready;
            if (rd_valid && rd_ready) begin
                pending_wave <= rd_wave;
                pending_addr <= rd_addr;
            end
        end

        if (rst_n && q_valid && q_ready) begin
            token = q_count % start_tokens;
            beat = (q_count / start_tokens) % 16;
            head = (q_count / (start_tokens * 16)) % 8;
            global_head = start_head_base + head;
            for (dim = 0; dim < 8; dim = dim + 1) begin
                expected_word = arena_word(token >= 4, token[1:0],
                    global_head * 128 + beat * 8 + dim);
                if (q_data[dim*32 +: 32] !== expected_word)
                    $fatal(1, "gather data mismatch q=%0d dim=%0d got=%h want=%h",
                        q_count, dim, q_data[dim*32 +: 32], expected_word);
            end
            q_count <= q_count + 1;
        end
        if (cycles > 100000)
            $fatal(1, "timeout");
    end

    initial begin
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        run_case(6'd0, 4'd1);
        run_case(6'd8, 4'd4);
        run_case(6'd24, 4'd8);
        $display("flash_query_gather PASS cycles=%0d", cycles);
        $finish;
    end
endmodule

`default_nettype wire
