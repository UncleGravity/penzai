`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg in_valid = 1'b0;
    wire in_ready;
    reg signed [103:0] in_acc = 104'sd0;
    reg signed [7:0] in_emin = -8'sd23;
    reg [2:0] in_token = 3'd0;
    reg [17:0] in_row = 18'd0;
    reg in_last = 1'b0;
    wire out_valid;
    reg out_ready;
    wire [31:0] out_data;
    wire [2:0] out_token;
    wire [17:0] out_row;
    wire out_last;
    wire [3:0] reserved;
    integer cycle = 0;
    integer sent = 0;
    integer received = 0;
    reg [53:0] expected [0:31];
    reg held = 1'b0;
    reg [53:0] held_payload;

     emit_stream #(.DEPTH(8)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .in_valid(in_valid), .in_ready(in_ready), .in_acc(in_acc),
        .in_emin(in_emin), .in_token(in_token), .in_row(in_row),
        .in_last(in_last), .out_valid(out_valid), .out_ready(out_ready),
        .out_data(out_data), .out_token(out_token), .out_row(out_row),
        .out_last(out_last), .reserved(reserved)
    );

    function automatic [31:0] expected_f32(input integer n);
        begin
            case (n % 4)
                0: expected_f32 = 32'h0000_0000;
                1: expected_f32 = 32'h3f80_0000;
                2: expected_f32 = 32'hbf80_0000;
                default: expected_f32 = 32'h3fc0_0000;
            endcase
        end
    endfunction

    always @(*) out_ready = cycle[2:0] != 3'b101;

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (held && ({out_last, out_token, out_row, out_data} !== held_payload))
            $fatal(1, "output changed under stall");
        held <= out_valid && !out_ready;
        if (out_valid && !out_ready)
            held_payload <= {out_last, out_token, out_row, out_data};

        if (out_valid && out_ready) begin
            if ({out_last, out_token, out_row, out_data} !== expected[received]) begin
                $display("emit mismatch n=%0d got=%h want=%h",
                         received, {out_last, out_token, out_row, out_data},
                         expected[received]);
                $fatal(1);
            end
            received <= received + 1;
        end
        if (reserved > 8) $fatal(1, "credit overflow");
    end

    task automatic send_one(input integer n);
        begin
            @(negedge clk);
            case (n % 4)
                0: in_acc = 104'sd0;
                1: in_acc = 104'sd8388608;
                2: in_acc = -104'sd8388608;
                default: in_acc = 104'sd12582912;
            endcase
            in_token = n % 8;
            in_row = n;
            in_last = n == 19;
            expected[n] = {in_last, in_token, in_row, expected_f32(n)};
            in_valid = 1'b1;
            while (!in_ready) @(negedge clk);
            @(negedge clk);
            in_valid = 1'b0;
            sent = sent + 1;
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        for (integer n = 0; n < 20; n = n + 1)
            send_one(n);
        while (received != sent && cycle < 1000) @(posedge clk);
        if (received != 20 || reserved != 0)
            $fatal(1, "drain failed sent=%0d received=%0d reserved=%0d",
                   sent, received, reserved);
        $display(" emit_stream: exact tags/backpressure PASS cycles=%0d", cycle);
        $finish;
    end
endmodule
