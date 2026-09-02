`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */
/* verilator lint_off TIMESCALEMOD */

module tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg in_valid = 1'b0;
    reg [255:0] in_data = 256'd0;
    reg [255:0] in_coeff = 256'd0;
    wire out_valid;
    wire [255:0] out_data;

    integer sent = 0;
    integer seen = 0;
    integer cycles = 0;
    reg [255:0] expected [0:3];

     rope4 dut (
        .clk(clk), .rst_n(rst_n),
        .in_valid(in_valid), .in_data(in_data), .in_coeff(in_coeff),
        .out_valid(out_valid), .out_data(out_data)
    );

    function automatic [31:0] f32_integer(input integer value);
        begin
            case (value)
                -8: f32_integer = 32'hc1000000;
                -7: f32_integer = 32'hc0e00000;
                -6: f32_integer = 32'hc0c00000;
                -5: f32_integer = 32'hc0a00000;
                -4: f32_integer = 32'hc0800000;
                -3: f32_integer = 32'hc0400000;
                -2: f32_integer = 32'hc0000000;
                -1: f32_integer = 32'hbf800000;
                 0: f32_integer = 32'h00000000;
                 1: f32_integer = 32'h3f800000;
                 2: f32_integer = 32'h40000000;
                 3: f32_integer = 32'h40400000;
                 4: f32_integer = 32'h40800000;
                 5: f32_integer = 32'h40a00000;
                 6: f32_integer = 32'h40c00000;
                 7: f32_integer = 32'h40e00000;
                 8: f32_integer = 32'h41000000;
                default: f32_integer = 32'h7fc00000;
            endcase
        end
    endfunction

    task automatic set_lane(
        input integer lane,
        input integer x0,
        input integer x1,
        input [31:0] c,
        input [31:0] s,
        input integer y0,
        input integer y1
    );
        begin
            in_data[lane*64 +: 32] = f32_integer(x0);
            in_data[lane*64 + 32 +: 32] = f32_integer(x1);
            in_coeff[lane*64 +: 32] = c;
            in_coeff[lane*64 + 32 +: 32] = s;
            expected[sent][lane*64 +: 32] = f32_integer(y0);
            expected[sent][lane*64 + 32 +: 32] = f32_integer(y1);
        end
    endtask

    task automatic send_record(input integer mode);
        integer lane;
        integer x0;
        integer x1;
        reg [31:0] c;
        reg [31:0] s;
        integer y0;
        integer y1;
        begin
            @(negedge clk);
            in_data = 256'd0;
            in_coeff = 256'd0;
            expected[sent] = 256'd0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                x0 = lane + 1;
                x1 = lane + 5;
                case (mode)
                    0: begin
                        c = 32'h3f800000; s = 32'h00000000;
                        y0 = x0; y1 = x1;
                    end
                    1: begin
                        c = 32'h00000000; s = 32'h3f800000;
                        y0 = -x1; y1 = x0;
                    end
                    2: begin
                        c = 32'h3f000000; s = 32'h3f000000;
                        y0 = -2; y1 = x0 + 2;
                    end
                    default: begin
                        c = 32'hbf000000; s = 32'h3f000000;
                        y0 = -(x0 + 2); y1 = -2;
                    end
                endcase
                set_lane(lane, x0, x1, c, s, y0, y1);
            end
            in_valid = 1'b1;
            sent = sent + 1;
        end
    endtask

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (out_valid) begin
            if (out_data !== expected[seen])
                $fatal(1, "RoPE record %0d mismatch got=%h expected=%h",
                    seen, out_data, expected[seen]);
            seen <= seen + 1;
        end
        if (cycles > 300) $fatal(1, "timeout");
    end

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        send_record(0);
        send_record(1);
        send_record(2);
        send_record(3);
        @(negedge clk);
        in_valid = 1'b0;
        in_data = 256'd0;
        in_coeff = 256'd0;
        wait (seen == 4);
        repeat (2) @(posedge clk);
        $display(" rope4 PASS records=%0d cycles=%0d", seen, cycles);
        $finish;
    end
endmodule

`default_nettype wire
