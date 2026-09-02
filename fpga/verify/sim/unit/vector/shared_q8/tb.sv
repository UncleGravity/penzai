`timescale 1ns/1ps
`default_nettype none

module shared_q8_tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;
    reg rst_n = 1'b0;
    reg abort_run = 1'b0;
    wire busy;
    wire collision_error;

    reg c0_cfg_valid = 1'b0;
    wire c0_cfg_ready;
    reg [14:0] c0_cfg_rows = 15'd32;
    reg [3:0] c0_cfg_lane_mask = 4'h1;
    reg c0_in_valid = 1'b0;
    wire c0_in_ready;
    reg [127:0] c0_in_data = {4{32'h3f80_0000}};
    wire c0_out_valid;
    reg c0_out_ready = 1'b1;
    wire [8:0] c0_out_block;
    wire [1087:0] c0_out_data;
    wire [7:0] c0_out_status;
    wire c0_out_last;
    reg c0_abort = 1'b0;

    reg c1_cfg_valid = 1'b0;
    wire c1_cfg_ready;
    reg [14:0] c1_cfg_rows = 15'd32;
    reg [3:0] c1_cfg_lane_mask = 4'h2;
    reg c1_in_valid = 1'b0;
    wire c1_in_ready;
    reg [127:0] c1_in_data = {4{32'h3f80_0000}};
    wire c1_out_valid;
    reg c1_out_ready = 1'b1;
    wire [8:0] c1_out_block;
    wire [1087:0] c1_out_data;
    wire [7:0] c1_out_status;
    wire c1_out_last;
    reg c1_abort = 1'b0;

    reg c2_cfg_valid = 1'b0;
    wire c2_cfg_ready;
    reg [14:0] c2_cfg_rows = 15'd32;
    reg [3:0] c2_cfg_lane_mask = 4'h8;
    reg c2_in_valid = 1'b0;
    wire c2_in_ready;
    reg [127:0] c2_in_data = {4{32'h3f80_0000}};
    wire c2_out_valid;
    reg c2_out_ready = 1'b1;
    wire [8:0] c2_out_block;
    wire [1087:0] c2_out_data;
    wire [7:0] c2_out_status;
    wire c2_out_last;
    reg c2_abort = 1'b0;

     shared_q8 dut (.*);

    task automatic start_client(input integer client);
        begin
            @(negedge clk);
            case (client)
                0: c0_cfg_valid = 1'b1;
                1: c1_cfg_valid = 1'b1;
                default: c2_cfg_valid = 1'b1;
            endcase
            #0.1;
            case (client)
                0: while (!c0_cfg_ready) begin @(negedge clk); #0.1; end
                1: while (!c1_cfg_ready) begin @(negedge clk); #0.1; end
                default: while (!c2_cfg_ready) begin
                    @(negedge clk); #0.1;
                end
            endcase
            @(posedge clk);
            @(negedge clk);
            c0_cfg_valid = 1'b0;
            c1_cfg_valid = 1'b0;
            c2_cfg_valid = 1'b0;
        end
    endtask

    task automatic feed_client(input integer client, input integer count);
        integer sent;
        begin
            sent = 0;
            while (sent < count) begin
                @(negedge clk);
                case (client)
                    0: c0_in_valid = 1'b1;
                    1: c1_in_valid = 1'b1;
                    default: c2_in_valid = 1'b1;
                endcase
                case (client)
                    0: if (c0_in_ready) sent = sent + 1;
                    1: if (c1_in_ready) sent = sent + 1;
                    default: if (c2_in_ready) sent = sent + 1;
                endcase
            end
            @(negedge clk);
            c0_in_valid = 1'b0;
            c1_in_valid = 1'b0;
            c2_in_valid = 1'b0;
        end
    endtask

    task automatic finish_client(input integer client);
        reg [1087:0] payload;
        reg [7:0] status;
        reg [8:0] block_index;
        reg is_last;
        reg [3:0] lane_mask;
        integer lane;
        begin
            case (client)
                0: while (!c0_out_valid) @(negedge clk);
                1: while (!c1_out_valid) @(negedge clk);
                default: while (!c2_out_valid) @(negedge clk);
            endcase
            case (client)
                0: begin payload = c0_out_data; status = c0_out_status;
                    block_index = c0_out_block; is_last = c0_out_last;
                    lane_mask = c0_cfg_lane_mask; end
                1: begin payload = c1_out_data; status = c1_out_status;
                    block_index = c1_out_block; is_last = c1_out_last;
                    lane_mask = c1_cfg_lane_mask; end
                default: begin payload = c2_out_data; status = c2_out_status;
                    block_index = c2_out_block; is_last = c2_out_last;
                    lane_mask = c2_cfg_lane_mask; end
            endcase
            if ((status != 8'd0) || (block_index != 9'd0) || !is_last)
                $fatal(1, "client %0d bad Q8 framing/status", client);
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (lane_mask[lane]) begin
                    if (payload[lane*272 +: 256] != {32{8'h7f}})
                        $fatal(1, "client %0d bad quant payload", client);
                end else if (payload[lane*272 +: 272] != 272'd0) begin
                    $fatal(1, "client %0d inactive lane was nonzero", client);
                end
            end
            @(negedge clk);
            if (busy)
                $fatal(1, "client %0d ownership did not retire", client);
        end
    endtask

    task automatic run_client(input integer client);
        begin
            $display("shared Q8 client %0d start", client);
            start_client(client);
            feed_client(client, 32);
            finish_client(client);
            $display("shared Q8 client %0d PASS", client);
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        run_client(0);
        run_client(1);
        run_client(2);

        // A non-owner abort cannot disrupt the active vector client.
        $display("shared Q8 non-owner abort test");
        start_client(0);
        @(negedge clk); c2_abort = 1'b1;
        @(negedge clk); c2_abort = 1'b0;
        if (!busy) $fatal(1, "non-owner abort released Q8");
        feed_client(0, 32);
        finish_client(0);

        // Abort after lane 0 has populated part of the record. The next
        // command must initialize that invisible stale payload before output.
        $display("shared Q8 owner partial-record abort/restart test");
        c2_cfg_lane_mask = 4'h9;
        start_client(2);
        feed_client(2, 32);
        while ((dut.u_q8.state_q != 3'd2) ||
               (dut.u_q8.current_lane_q != 2'd3)) @(negedge clk);
        if (dut.u_q8.record_q[271:0] == 272'd0)
            $fatal(1, "owner abort test never populated the partial record");
        @(negedge clk); c2_abort = 1'b1;
        @(negedge clk); c2_abort = 1'b0;
        repeat (4) @(negedge clk);
        if (busy || c2_out_valid)
            $fatal(1, "attention owner abort did not flush Q8");
        c2_cfg_lane_mask = 4'h8;
        run_client(1);

        // Global abort has the same payload-visibility contract and must also
        // permit an exact clean restart after a partially assembled record.
        $display("shared Q8 global partial-record abort/restart test");
        c0_cfg_lane_mask = 4'h3;
        start_client(0);
        feed_client(0, 32);
        while ((dut.u_q8.state_q != 3'd2) ||
               (dut.u_q8.current_lane_q != 2'd1)) @(negedge clk);
        if (dut.u_q8.record_q[271:0] == 272'd0)
            $fatal(1, "global abort test never populated the partial record");
        @(negedge clk); abort_run = 1'b1;
        @(negedge clk); abort_run = 1'b0;
        repeat (4) @(negedge clk);
        if (busy || c0_out_valid)
            $fatal(1, "global abort did not flush partial Q8 record");
        c0_cfg_lane_mask = 4'h1;
        run_client(1);

        // Simultaneous requests are serialized with c0 priority and reported.
        $display("shared Q8 collision/priority test");
        @(negedge clk);
        c0_cfg_valid = 1'b1;
        c2_cfg_valid = 1'b1;
        #0.1;
        if (!c0_cfg_ready || c2_cfg_ready)
            $fatal(1, "shared Q8 priority was not deterministic");
        @(negedge clk);
        c0_cfg_valid = 1'b0;
        c2_cfg_valid = 1'b0;
        @(negedge clk);
        if (!collision_error)
            $fatal(1, "simultaneous request was not reported");
        feed_client(0, 32);
        finish_client(0);
        abort_run = 1'b1;
        @(negedge clk); abort_run = 1'b0;
        @(negedge clk);
        if (collision_error)
            $fatal(1, "global abort did not clear collision status");

        $display(" shared_q8_tb PASS");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "shared Q8 timeout");
    end
endmodule

`default_nettype wire
