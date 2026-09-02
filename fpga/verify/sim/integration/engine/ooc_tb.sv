`timescale 1ns/1ps
`default_nettype none

module engine_ooc_tb;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start_valid = 1'b0;
    wire start_ready;
    reg [31:0] seed = 32'd0;
    wire result_valid;
    reg result_ready = 1'b0;
    wire result_error;
    wire [31:0] result_signature;

    reg [31:0] first_signature;
    integer cycles = 0;

    always #1 clk = ~clk;
    always @(posedge clk) begin
        if (rst_n)
            cycles <= cycles + 1;
    end

    engine_ooc dut (.*);

    task automatic execute(input [31:0] value);
        integer timeout;
        begin
            @(negedge clk);
            seed = value;
            start_valid = 1'b1;
            while (!start_ready)
                @(negedge clk);
            @(negedge clk);
            start_valid = 1'b0;

            timeout = 0;
            while (!result_valid && timeout < 10000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!result_valid)
                $fatal(1, "registered OOC shell timed out");
            if (result_error)
                $fatal(1, "registered OOC shell reported execution error");
            @(negedge clk);
            result_ready = 1'b1;
            @(negedge clk);
            result_ready = 1'b0;
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst_n = 1'b1;

        execute(32'h1234_5604);
        first_signature = result_signature;
        execute(32'h89ab_cdef);
        if (result_signature == first_signature)
            $fatal(1, "OOC signature did not depend on registered inputs");

        $display("engine_ooc_tb PASS cycles=%0d", cycles);
        $finish;
    end
endmodule

`default_nettype wire
