`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */
/* verilator lint_off TIMESCALEMOD */

module digit_accum_tb;
    localparam integer DIGITS = 7;
    localparam integer BIN_W = 26;
    localparam integer DIGIT_W = DIGITS*BIN_W;
    localparam integer CONTRIBUTIONS = 12288/32;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #1 clk = ~clk;

    reg mul_in_valid = 1'b0;
    reg mul_clear_in = 1'b0;
    reg signed [11:0] ws_sig = 12'sd0;
    reg signed [11:0] as_sig = 12'sd0;
    reg signed [7:0] p_exp = 8'sd0;
    reg signed [7:0] emin = -8'sd20;
    reg signed [13:0] dot_sum = 14'sd0;
    wire mul_valid;
    wire signed [8:0] mul_coarse;
    wire [63:0] mul_chunks;
     digit_mul u_mul (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(mul_in_valid),
        .ws_sig(ws_sig),
        .as_sig(as_sig),
        .p_exp(p_exp),
        .emin(emin),
        .dot_sum(dot_sum),
        .valid_out(mul_valid),
        .coarse_digit(mul_coarse),
        .fine_chunks(mul_chunks)
    );

    wire mul_clear;
     delay #(.WIDTH(1), .DEPTH(4)) u_clear_delay (
        .clk(clk),
        .din(mul_clear_in),
        .dout(mul_clear)
    );

    reg direct_mode = 1'b0;
    reg direct_valid = 1'b0;
    reg direct_clear = 1'b0;
    reg signed [8:0] direct_coarse = 9'sd0;
    reg [63:0] direct_chunks = 64'd0;
    wire cell_valid = direct_mode ? direct_valid : mul_valid;
    wire cell_clear = direct_mode ? direct_clear : mul_clear;
    wire signed [8:0] cell_coarse =
        direct_mode ? direct_coarse : mul_coarse;
    wire [63:0] cell_chunks = direct_mode ? direct_chunks : mul_chunks;
    wire signed [DIGIT_W-1:0] drain_digits;
    wire cell_update_pending;
     digit_cell u_cell (
        .clk(clk),
        .rst_n(rst_n),
        .clear(1'b0),
        .update_valid(cell_valid),
        .update_clear(cell_clear),
        .update_bank(1'b0),
        .update_token(3'd0),
        .update_coarse(cell_coarse),
        .update_chunks(cell_chunks),
        .drain_bank(1'b0),
        .drain_token(3'd0),
        .drain_digits(drain_digits),
        .update_pending(cell_update_pending)
    );

    reg norm_in_valid = 1'b0;
    wire norm_in_ready;
    reg [7:0] norm_in_meta = 8'd0;
    wire norm_out_valid;
    reg norm_out_ready = 1'b0;
    wire signed [103:0] norm_out_acc;
    wire [7:0] norm_out_meta;
    wire norm_busy;
     digit_normalize #(.META_W(8)) u_normalize (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(norm_in_valid),
        .in_ready(norm_in_ready),
        .in_digits(drain_digits),
        .in_meta(norm_in_meta),
        .out_valid(norm_out_valid),
        .out_ready(norm_out_ready),
        .out_acc(norm_out_acc),
        .out_meta(norm_out_meta),
        .busy(norm_busy)
    );

    function automatic signed [103:0] oracle_contribution;
        input integer ws_value;
        input integer as_value;
        input integer sum_value;
        input integer shift_value;
        reg signed [11:0] ws_local;
        reg signed [11:0] as_local;
        reg signed [13:0] sum_local;
        reg signed [23:0] pair_product;
        reg signed [37:0] full_product;
        reg signed [103:0] extended_product;
        begin
            ws_local = ws_value;
            as_local = as_value;
            sum_local = sum_value;
            pair_product = ws_local * as_local;
            full_product = pair_product * sum_local;
            extended_product = {{(104-38){full_product[37]}}, full_product};
            if (shift_value < 0)
                oracle_contribution = extended_product >>> (-shift_value);
            else
                oracle_contribution = extended_product <<< shift_value;
        end
    endfunction

    task automatic normalize_and_check;
        input signed [103:0] expected;
        input [7:0] expected_meta;
        reg signed [103:0] held;
        begin
            @(negedge clk);
            norm_out_ready = 1'b0;
            norm_in_meta = expected_meta;
            norm_in_valid = 1'b1;
            do @(posedge clk); while (!norm_in_ready);
            @(negedge clk);
            norm_in_valid = 1'b0;
            wait (norm_out_valid);
            @(negedge clk);
            held = norm_out_acc;
            if ((norm_out_acc !== expected) ||
                (norm_out_meta !== expected_meta))
                $fatal(1,
                    "digit normalize mismatch meta=%0d got=%0d expected=%0d",
                    expected_meta, $signed(norm_out_acc), $signed(expected));
            repeat (3) begin
                @(negedge clk);
                if (!norm_out_valid || (norm_out_acc !== held))
                    $fatal(1, "digit output changed while stalled");
            end
            norm_out_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            norm_out_ready = 1'b0;
            if (norm_out_valid)
                $fatal(1, "digit output did not retire");
        end
    endtask

    integer contribution;
    integer digit;
    integer seq_index;
    integer ws_value;
    integer as_value;
    integer sum_value;
    integer shift_value;
    reg signed [103:0] expected;
    reg signed [103:0] chunk;
    initial begin
        if (CONTRIBUTIONS*65535 >= (1 << 25))
            $fatal(1, "K=12288 digit bound exceeds 25 unsigned bits");

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // Exercise the exact worst positive per-bin bound directly.  Signed
        // 26-bit storage is required because 25 bits describe the magnitude,
        // while the fourth source chunk may also be negative.
        direct_mode = 1'b1;
        direct_chunks = 64'd0;
        direct_chunks[15:0] = 16'hffff;
        expected = 104'sd0;
        for (contribution = 0; contribution < CONTRIBUTIONS;
             contribution = contribution + 1) begin
            for (digit = 0; digit < DIGITS; digit = digit + 1) begin
                @(negedge clk);
                direct_valid = 1'b1;
                direct_clear = (contribution == 0) && (digit == 0);
                direct_coarse = digit;
                chunk = 104'sd65535;
                expected = expected + (chunk <<< (16*digit));
            end
        end
        @(negedge clk);
        direct_valid = 1'b0;
        direct_clear = 1'b0;
        normalize_and_check(expected, 8'ha5);

        // Twelve deterministic K=12288 sequences sweep positive and negative
        // products and coarse shifts from -96 through +48.  The oracle is the
        // original signed 104-bit shift-and-add behavior.
        direct_mode = 1'b0;
        for (seq_index = 0; seq_index < 12; seq_index = seq_index + 1) begin
            expected = 104'sd0;
            for (contribution = 0; contribution < CONTRIBUTIONS;
                 contribution = contribution + 1) begin
                ws_value = ((contribution*73 + seq_index*17) % 1023) - 511;
                as_value = ((contribution*47 + seq_index*31) % 1009) - 504;
                sum_value = ((contribution*61 + seq_index*43) % 1019) - 509;
                shift_value =
                    ((contribution*29 + seq_index*11) % 145) - 96;
                expected = expected + oracle_contribution(
                    ws_value, as_value, sum_value, shift_value);
                @(negedge clk);
                mul_in_valid = 1'b1;
                mul_clear_in = (contribution == 0);
                ws_sig = ws_value;
                as_sig = as_value;
                dot_sum = sum_value;
                p_exp = -20 + shift_value;
            end
            @(negedge clk);
            mul_in_valid = 1'b0;
            mul_clear_in = 1'b0;
            repeat (6) @(posedge clk);
            normalize_and_check(expected, seq_index[7:0]);
        end

        $display("PASS radix-digit K=12288 bound/random exact equivalence");
        $finish;
    end
endmodule

`default_nettype wire
