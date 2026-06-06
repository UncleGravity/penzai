`timescale 1ns / 1ps

module axis_pattern_check_tb;
    localparam integer DATA_BYTES = 16;

    logic aclk;
    logic aresetn;
    logic start;
    logic [63:0] length_bytes;
    logic [63:0] base_index;
    logic [7:0] seed;

    logic busy;
    logic done;
    logic error_seen;
    logic [63:0] first_error_index;
    logic [7:0] expected;
    logic [7:0] actual;
    logic [63:0] bytes_checked;
    logic [63:0] cycles;

    logic [127:0] s_axis_tdata;
    logic [15:0] s_axis_tkeep;
    logic s_axis_tvalid;
    logic s_axis_tready;
    logic s_axis_tlast;

    axis_pattern_check dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .start(start),
        .length_bytes(length_bytes),
        .base_index(base_index),
        .seed(seed),
        .busy(busy),
        .done(done),
        .error_seen(error_seen),
        .first_error_index(first_error_index),
        .expected(expected),
        .actual(actual),
        .bytes_checked(bytes_checked),
        .cycles(cycles),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast)
    );

    initial begin
        aclk = 1'b0;
    end

    /* verilator lint_off BLKSEQ */
    always #5 aclk = ~aclk;
    /* verilator lint_on BLKSEQ */

    function automatic logic [7:0] pattern_byte(
        input logic [7:0] index,
        input logic [7:0] seed_value
    );
        begin
            pattern_byte = (index * 8'd7) + seed_value;
        end
    endfunction

    function automatic logic [127:0] pattern_beat(
        input logic [7:0] start_index_low,
        input logic [7:0] seed_value
    );
        logic [127:0] data;
        integer lane;
        begin
            data = 128'd0;
            for (lane = 0; lane < DATA_BYTES; lane = lane + 1) begin
                data[(lane * 8) +: 8] = pattern_byte(start_index_low + lane[7:0], seed_value);
            end
            pattern_beat = data;
        end
    endfunction

    task automatic clear_inputs;
        begin
            start = 1'b0;
            length_bytes = 64'd0;
            base_index = 64'd0;
            seed = 8'd0;
            s_axis_tdata = 128'd0;
            s_axis_tkeep = 16'd0;
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
        end
    endtask

    task automatic reset_dut;
        begin
            clear_inputs();
            aresetn = 1'b0;
            repeat (4) @(posedge aclk);
            aresetn = 1'b1;
            repeat (2) @(posedge aclk);
        end
    endtask

    task automatic start_check(
        input logic [63:0] case_length_bytes,
        input logic [63:0] case_base_index,
        input logic [7:0] case_seed
    );
        begin
            @(negedge aclk);
            length_bytes = case_length_bytes;
            base_index = case_base_index;
            seed = case_seed;
            start = 1'b1;
            @(negedge aclk);
            start = 1'b0;
            if (!busy) begin
                $fatal(1, "checker did not enter busy state");
            end
        end
    endtask

    task automatic send_raw_beat(
        input logic [127:0] data,
        input logic [15:0] keep,
        input logic last
    );
        begin
            @(negedge aclk);
            s_axis_tdata = data;
            s_axis_tkeep = keep;
            s_axis_tlast = last;
            s_axis_tvalid = 1'b1;
            @(posedge aclk);
            if (!s_axis_tready) begin
                $fatal(1, "checker did not accept stream beat");
            end
            @(negedge aclk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tkeep = 16'd0;
            s_axis_tdata = 128'd0;
        end
    endtask

    task automatic send_pattern_beat(
        input logic [7:0] beat_base_index_low,
        input logic [15:0] keep,
        input logic last
    );
        begin
            send_raw_beat(pattern_beat(beat_base_index_low, seed), keep, last);
        end
    endtask

    task automatic expect_clean(input logic [63:0] expected_bytes_checked);
        begin
            repeat (2) @(posedge aclk);
            if (!done) begin
                $fatal(1, "checker did not finish");
            end
            if (busy) begin
                $fatal(1, "checker stayed busy after final beat");
            end
            if (error_seen) begin
                $fatal(
                    1,
                    "unexpected checker error index=%0d expected=0x%02x actual=0x%02x",
                    first_error_index,
                    expected,
                    actual
                );
            end
            if (bytes_checked != expected_bytes_checked) begin
                $fatal(
                    1,
                    "bytes_checked mismatch expected=%0d actual=%0d",
                    expected_bytes_checked,
                    bytes_checked
                );
            end
            if (cycles == 64'd0) begin
                $fatal(1, "checker cycle counter did not advance");
            end
        end
    endtask

    task automatic expect_error(
        input logic [63:0] error_index,
        input logic [7:0] error_expected,
        input logic [7:0] error_actual
    );
        begin
            repeat (2) @(posedge aclk);
            if (!done) begin
                $fatal(1, "checker did not finish after expected error");
            end
            if (!error_seen) begin
                $fatal(1, "checker did not report expected error");
            end
            if (first_error_index != error_index) begin
                $fatal(
                    1,
                    "error index mismatch expected=%0d actual=%0d",
                    error_index,
                    first_error_index
                );
            end
            if (expected != error_expected || actual != error_actual) begin
                $fatal(
                    1,
                    "error bytes mismatch expected=0x%02x/0x%02x actual=0x%02x/0x%02x",
                    error_expected,
                    error_actual,
                    expected,
                    actual
                );
            end
        end
    endtask

    initial begin
        logic [127:0] corrupt;
        logic [7:0] corrupt_expected;
        logic [7:0] corrupt_actual;

        reset_dut();
        start_check(64'd32, 64'd0, 8'h00);
        send_pattern_beat(8'd0, 16'h0000, 1'b0);
        send_pattern_beat(8'd16, 16'h0000, 1'b1);
        expect_clean(64'd32);

        reset_dut();
        start_check(64'd16, 64'd16, 8'h11);
        corrupt = pattern_beat(8'd16, 8'h11);
        corrupt_expected = pattern_byte(8'd19, 8'h11);
        corrupt_actual = corrupt_expected ^ 8'hff;
        corrupt[(3 * 8) +: 8] = corrupt_actual;
        send_raw_beat(corrupt, 16'hffff, 1'b1);
        expect_error(64'd19, corrupt_expected, corrupt_actual);

        reset_dut();
        start_check(64'd16, 64'd0, 8'h22);
        send_pattern_beat(8'd0, 16'hffff, 1'b0);
        expect_error(64'd16, 8'h01, 8'h00);

        $display("axis_pattern_check_tb ok");
        $finish;
    end
endmodule
