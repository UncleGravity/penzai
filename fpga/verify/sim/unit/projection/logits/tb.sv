`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg start_valid = 1'b0;
    wire start_ready;
    reg [17:0] vocab_rows = 18'd0;
    reg emit_full_logits = 1'b0;
    reg in_valid = 1'b0;
    wire in_ready;
    reg [17:0] in_row = 18'd0;
    reg [31:0] in_data = 32'd0;
    reg in_last = 1'b0;
    wire logits_valid;
    reg logits_ready = 1'b0;
    wire [17:0] logits_row;
    wire [31:0] logits_data;
    wire logits_last;
    wire result_valid;
    reg result_ready = 1'b0;
    wire [17:0] result_token;
    wire [31:0] result_logit;
    wire result_error;
    wire [7:0] result_status;
    wire [31:0] accepted_logits;
    wire busy;
    integer cycle = 0;
    integer passed = 0;

     logits_sink dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .start_valid(start_valid), .start_ready(start_ready),
        .vocab_rows(vocab_rows), .emit_full_logits(emit_full_logits),
        .in_valid(in_valid), .in_ready(in_ready), .in_row(in_row),
        .in_data(in_data), .in_last(in_last),
        .logits_valid(logits_valid), .logits_ready(logits_ready),
        .logits_row(logits_row), .logits_data(logits_data),
        .logits_last(logits_last),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_logit(result_logit),
        .result_error(result_error), .result_status(result_status),
        .accepted_logits(accepted_logits), .busy(busy)
    );

    task automatic start_run(input integer rows, input bit full);
        begin
            @(negedge clk);
            vocab_rows = rows;
            emit_full_logits = full;
            start_valid = 1'b1;
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task automatic send(
        input integer row,
        input [31:0] value,
        input bit last
    );
        begin
            @(negedge clk);
            in_row = row;
            in_data = value;
            in_last = last;
            in_valid = 1'b1;
            while (!in_ready) @(negedge clk);
            @(negedge clk);
            in_valid = 1'b0;
        end
    endtask

    task automatic expect_result(
        input integer token,
        input [31:0] value,
        input bit want_error,
        input [7:0] want_status
    );
        begin
            while (!result_valid) @(negedge clk);
            if ((result_token !== token[17:0]) ||
                (result_logit !== value) ||
                (result_error !== want_error) ||
                (result_status !== want_status)) begin
                $display("bad result token=%0d/%0d value=%h/%h err=%0d/%0d status=%h/%h",
                         result_token, token, result_logit, value,
                         result_error, want_error, result_status, want_status);
                $fatal(1);
            end
            result_ready = 1'b1;
            @(negedge clk);
            result_ready = 1'b0;
            passed = passed + 1;
        end
    endtask

    always @(posedge clk) begin
        cycle <= cycle + 1;
        logits_ready <= cycle[1:0] != 2'b10;
        if (logits_valid && !logits_ready) begin
            // Protocol stability is checked by Verilator assertions below.
        end
    end

    reg held_valid = 1'b0;
    reg [50:0] held_payload;
    always @(posedge clk) begin
        if (held_valid && ({logits_last, logits_row, logits_data} !== held_payload)) begin
            $display("logits changed under backpressure");
            $fatal(1);
        end
        held_valid <= logits_valid && !logits_ready;
        if (logits_valid && !logits_ready)
            held_payload <= {logits_last, logits_row, logits_data};
    end

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        // Full stream, NaN skipped, and equal maxima keep the first row.
        start_run(6, 1'b1);
        send(0, 32'hbf80_0000, 1'b0); // -1
        send(1, 32'h7fc0_0000, 1'b0); // NaN
        send(2, 32'h40a0_0000, 1'b0); // 5
        send(3, 32'h4000_0000, 1'b0); // 2
        send(4, 32'h40a0_0000, 1'b0); // 5, first tie wins
        send(5, 32'hc0a0_0000, 1'b1); // -5
        expect_result(2, 32'h40a0_0000, 1'b0, 8'h00);
        if (accepted_logits != 6) $fatal(1, "accepted count");

        // Greedy-only mode never waits on the disabled logits output.
        logits_ready = 1'b0;
        start_run(3, 1'b0);
        send(0, 32'h8000_0000, 1'b0); // -0
        send(1, 32'h0000_0000, 1'b0); // +0: numeric tie, first wins
        send(2, 32'hff80_0000, 1'b1); // -inf
        expect_result(0, 32'h8000_0000, 1'b0, 8'h00);

        // Framing violations fail closed.
        start_run(2, 1'b0);
        send(1, 32'h3f80_0000, 1'b0);
        expect_result(0, 32'hff80_0000, 1'b1, 8'h02);

        if (passed != 3) $fatal(1, "missing cases");
        $display(" logits_sink: full/greedy/framing PASS cycles=%0d", cycle);
        $finish;
    end
endmodule
