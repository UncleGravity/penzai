`timescale 1ns/1ps
`default_nettype none

module engine_metrics_tb;
    logic clk = 1'b0;
    always #1 clk = ~clk;
    logic rst_n = 1'b0;

    logic start = 1'b0;
    logic [31:0] start_tag = 32'd0;
    logic finish = 1'b0;
    logic [1:0] finish_outcome = 2'd0;
    logic acknowledge = 1'b0;
    logic core_stage_active = 1'b0;
    logic [4:0] core_stage = 5'd0;
    logic core_stage_call = 1'b0;
    logic [4:0] core_stage_call_id = 5'd0;
    logic [12:0] projection_probe = 13'd0;
    logic [2:0] weight_axi_r_beats = 3'd0;
    logic [2:0] weight_axi_r_gap_ports = 3'd0;
    logic weight_zip_skew = 1'b0;
    logic [1:0] history_axi_r_beats = 2'd0;
    logic kv_axi_w_beat = 1'b0;
    logic [6:0] read_index = 7'd0;
    wire [31:0] read_data_lo;
    wire [31:0] read_data_hi;
    wire [31:0] schema;
    wire [31:0] capabilities;
    wire [31:0] status;
    wire [31:0] snapshot_tag;
    wire [31:0] overflow0;
    wire [31:0] overflow1;
    wire [31:0] overflow2;
    wire [31:0] overflow3;
    wire [63:0] total_cycles;
    wire recording;
    wire sealing;
    wire snapshot_valid;

    engine_metrics #(
        .COUNTER_W(4),
        .CALL_COUNTER_W(4)
    ) dut (.*);

    task automatic expect_metric(input logic [6:0] id,
                                 input logic [31:0] expected);
        begin
            read_index = id;
            #0.1;
            if (read_data_lo !== expected)
                $fatal(1, "metric %0d got %0d expected %0d",
                       id, read_data_lo, expected);
        end
    endtask

    task automatic clear_probes;
        begin
            core_stage_active = 1'b0;
            core_stage = 5'd0;
            core_stage_call = 1'b0;
            core_stage_call_id = 5'd0;
            projection_probe = 13'd0;
            weight_axi_r_beats = 3'd0;
            weight_axi_r_gap_ports = 3'd0;
            weight_zip_skew = 1'b0;
            history_axi_r_beats = 2'd0;
            kv_axi_w_beat = 1'b0;
        end
    endtask

    integer cycle;
    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        #0.1;
        assert(schema == 32'h0001_0000);
        assert(capabilities == 32'h0028_0b1d);
        assert(status == 32'h0000_0008);

        @(negedge clk);
        start_tag = 32'h1234_5678;
        start = 1'b1;
        @(posedge clk); #0.1;
        assert(recording && !sealing && !snapshot_valid);
        @(negedge clk);
        start = 1'b0;

        // First recorded cycle: EMBED call and four physical weight beats.
        core_stage_active = 1'b1;
        core_stage = 5'd0;
        core_stage_call = 1'b1;
        core_stage_call_id = 5'd0;
        projection_probe[0] = 1'b1;
        projection_probe[6:4] = 3'd2;
        projection_probe[7] = 1'b1;
        weight_axi_r_beats = 3'd4;
        weight_axi_r_gap_ports = 3'd1;
        history_axi_r_beats = 2'd1;
        @(posedge clk);

        // Second cycle: QKV call plus selector/Q8 starvation signals.
        @(negedge clk);
        clear_probes();
        core_stage_active = 1'b1;
        core_stage = 5'd2;
        core_stage_call = 1'b1;
        core_stage_call_id = 5'd2;
        projection_probe[1] = 1'b1;
        projection_probe[3] = 1'b1;
        projection_probe[6:4] = 3'd4;
        projection_probe[9] = 1'b1;
        projection_probe[11] = 1'b1;
        weight_axi_r_gap_ports = 3'd3;
        weight_zip_skew = 1'b1;
        history_axi_r_beats = 2'd2;
        kv_axi_w_beat = 1'b1;
        @(posedge clk);

        // The terminal controller cycle is control time. Its locally registered
        // event sample is followed by one final seal sample below.
        @(negedge clk);
        clear_probes();
        projection_probe[2] = 1'b1;
        projection_probe[8] = 1'b1;
        projection_probe[10] = 1'b1;
        projection_probe[12] = 1'b1;
        weight_axi_r_beats = 3'd2;
        kv_axi_w_beat = 1'b1;
        finish_outcome = 2'd1;
        finish = 1'b1;
        @(posedge clk); #0.1;
        assert(!recording && sealing && !snapshot_valid);

        @(negedge clk);
        clear_probes();
        finish = 1'b0;
        projection_probe[8] = 1'b1;
        weight_axi_r_beats = 3'd1;
        weight_axi_r_gap_ports = 3'd4;
        weight_zip_skew = 1'b1;
        history_axi_r_beats = 2'd1;
        kv_axi_w_beat = 1'b1;
        @(posedge clk); #0.1;
        assert(!recording && !sealing && snapshot_valid);
        assert(snapshot_tag == 32'h1234_5678);
        assert(status == 32'h0000_0029);
        assert(total_cycles == 64'd3);

        expect_metric(0, 3);
        assert(read_data_hi == 32'd0);
        expect_metric(1, 1);
        expect_metric(2, 1);
        expect_metric(4, 1);
        expect_metric(13, 1);
        expect_metric(15, 1);
        expect_metric(24, 1);
        expect_metric(25, 1);
        expect_metric(26, 1);
        expect_metric(27, 1);
        expect_metric(28, 4);
        expect_metric(29, 1);
        expect_metric(30, 2);
        expect_metric(31, 1);
        expect_metric(32, 1);
        expect_metric(33, 1);
        expect_metric(34, 1);
        expect_metric(35, 7);
        expect_metric(36, 8);
        expect_metric(37, 2);
        expect_metric(38, 4);
        expect_metric(39, 3);
        assert(overflow0 == 32'd0 && overflow1 == 32'd0 &&
               overflow2 == 32'd0 && overflow3 == 32'd0);

        // Frozen values ignore later probes until explicitly acknowledged.
        @(negedge clk);
        projection_probe = 13'h1fff;
        weight_axi_r_beats = 3'd4;
        repeat (3) @(posedge clk);
        expect_metric(24, 1);
        expect_metric(35, 7);

        @(negedge clk);
        acknowledge = 1'b1;
        @(posedge clk); #0.1;
        assert(!snapshot_valid);
        @(negedge clk);
        acknowledge = 1'b0;
        clear_probes();

        // Reduced-width counters saturate and mark overflow instead of wrapping.
        start_tag = 32'hcafe_0002;
        start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        start = 1'b0;
        core_stage_active = 1'b1;
        core_stage = 5'd0;
        core_stage_call = 1'b1;
        core_stage_call_id = 5'd0;
        projection_probe[0] = 1'b1;
        for (cycle = 0; cycle < 19; cycle = cycle + 1)
            @(posedge clk);
        @(negedge clk);
        finish = 1'b1;
        finish_outcome = 2'd3;
        @(posedge clk);
        @(negedge clk);
        finish = 1'b0;
        @(posedge clk); #0.1;
        assert(snapshot_valid && (status[6:5] == 2'd3));
        assert(total_cycles == 64'd20);
        expect_metric(2, 15);
        expect_metric(13, 15);
        expect_metric(24, 15);
        assert(overflow0[2] && overflow0[13] && overflow0[24]);

        $display("PASS engine_metrics lifecycle/saturation/snapshot");
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "engine_metrics timeout");
    end
endmodule

`default_nettype wire
