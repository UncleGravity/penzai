`timescale 1ns/1ps
`default_nettype none

module rms_reduce4_tb;
    localparam [1:0] INJECT_NONE      = 2'd0;
    localparam [1:0] INJECT_SOURCE    = 2'd1;
    localparam [1:0] INJECT_NONFINITE = 2'd2;

    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg cfg_valid = 1'b0;
    wire cfg_ready;
    reg [12:0] cfg_rows = 13'd128;
    reg [3:0] cfg_lane_mask = 4'b1010;
    reg [31:0] cfg_epsilon = 32'h3727_c5ac;
    reg abort_run = 1'b0;
    wire busy;

    wire src_req_valid;
    wire src_req_ready;
    wire [11:0] src_req_addr;
    wire src_rsp_valid;
    wire src_rsp_ready;
    wire [127:0] src_rsp_data;
    wire src_rsp_error;

    wire result_valid;
    reg result_ready = 1'b0;
    wire result_error;
    wire [7:0] result_status;
    wire [127:0] result_inv_rms;

     rms_reduce4 dut (.*);

    integer cycle = 0;
    always @(posedge clk)
        if (rst_n) cycle <= cycle + 1;

    // A throttled 32-entry response queue deliberately allows the reducer to
    // issue well ahead of the first response. This models the arena pipeline
    // plus interconnect backpressure and leaves a real prefetched tail behind
    // the injected bad record.
    reg [11:0] source_fifo [0:31];
    reg [4:0] source_wr_q = 5'd0;
    reg [4:0] source_rd_q = 5'd0;
    reg [5:0] source_count_q = 6'd0;
    reg source_drain_started_q = 1'b0;
    reg [1:0] inject_mode_q = INJECT_NONE;
    reg [11:0] inject_addr_q = 12'd5;
    reg oracle_mode_q = 1'b0;
    reg oracle_checked_q = 1'b0;
    reg bad_seen_q = 1'b0;
    integer bad_outstanding_q = 0;
    integer request_count_q = 0;
    integer response_count_q = 0;

    wire source_push = src_req_valid && src_req_ready;
    wire [11:0] source_head_addr = source_fifo[source_rd_q];
    wire source_is_bad = (inject_mode_q != INJECT_NONE) &&
                         (source_head_addr == inject_addr_q);
    assign src_req_ready = source_count_q != 6'd32;
    assign src_rsp_valid = source_drain_started_q &&
                           (source_count_q != 6'd0) &&
                           (cycle[2:0] != 3'd3);
    assign src_rsp_error = src_rsp_valid && source_is_bad &&
                           (inject_mode_q == INJECT_SOURCE);
    function automatic [31:0] oracle_value(
        input [11:0] addr,
        input integer lane
    );
        begin
            case (lane)
                0: oracle_value = 32'h3f80_0000; // 1.0
                1: oracle_value = addr[0] ?
                                  32'h3f00_0000 : 32'h3f80_0000;
                2: oracle_value = 32'h3fc0_0000; // 1.5
                default: oracle_value = 32'hbf80_0000;
            endcase
        end
    endfunction

    wire [127:0] oracle_rsp_data = {
        oracle_value(source_head_addr, 3),
        oracle_value(source_head_addr, 2),
        oracle_value(source_head_addr, 1),
        oracle_value(source_head_addr, 0)
    };
    assign src_rsp_data = (src_rsp_valid && source_is_bad &&
                           (inject_mode_q == INJECT_NONFINITE)) ?
                          {4{32'h7fc0_0001}} :
                          (oracle_mode_q ? oracle_rsp_data :
                                           {4{32'h3f80_0000}});
    wire source_pop = src_rsp_valid && src_rsp_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            source_wr_q <= 5'd0;
            source_rd_q <= 5'd0;
            source_count_q <= 6'd0;
            source_drain_started_q <= 1'b0;
            bad_seen_q <= 1'b0;
            bad_outstanding_q <= 0;
            request_count_q <= 0;
            response_count_q <= 0;
            oracle_checked_q <= 1'b0;
        end else begin
            if (cfg_valid && cfg_ready) begin
                if (source_count_q != 0)
                    $fatal(1, "new reduction accepted with stale source data");
                source_drain_started_q <= 1'b0;
                bad_seen_q <= 1'b0;
                bad_outstanding_q <= 0;
                request_count_q <= 0;
                response_count_q <= 0;
                oracle_checked_q <= 1'b0;
            end

            if (oracle_mode_q && !oracle_checked_q &&
                (dut.state_q == 5'd9)) begin
                // Independent closed-form sums for 128 rows after the exact
                // exponent-aligned quantizer: 1.0 -> 2^17, 0.5 -> 2^16,
                // and 1.5 -> 3*2^16.
                if ((dut.inverse_sum_q[0] !== 48'h0200_0000_0000) ||
                    (dut.inverse_sum_q[1] !== 48'h0140_0000_0000) ||
                    (dut.inverse_sum_q[2] !== 48'h0480_0000_0000) ||
                    (dut.inverse_sum_q[3] !== 48'd0))
                    $fatal(1,
                        "reducer exact sum oracle mismatch %h %h %h %h",
                        dut.inverse_sum_q[0], dut.inverse_sum_q[1],
                        dut.inverse_sum_q[2], dut.inverse_sum_q[3]);
                oracle_checked_q <= 1'b1;
            end

            if (source_push) begin
                source_fifo[source_wr_q] <= src_req_addr;
                source_wr_q <= source_wr_q + 1'b1;
                request_count_q <= request_count_q + 1;
                if (bad_seen_q)
                    $fatal(1, "reducer issued after bad source response");
            end
            if (source_pop) begin
                source_rd_q <= source_rd_q + 1'b1;
                response_count_q <= response_count_q + 1;
                if (source_is_bad) begin
                    bad_seen_q <= 1'b1;
                    bad_outstanding_q <= integer'(dut.issue_count_q) -
                                         integer'(dut.rsp_count_q);
                end
            end

            case ({source_push, source_pop})
                2'b10: source_count_q <= source_count_q + 1'b1;
                2'b01: source_count_q <= source_count_q - 1'b1;
                default: source_count_q <= source_count_q;
            endcase
            if (source_count_q >= 6'd8)
                source_drain_started_q <= 1'b1;
        end
    end

    task automatic submit;
        begin
            @(negedge clk);
            cfg_valid = 1'b1;
            while (!cfg_ready) @(negedge clk);
            @(negedge clk);
            cfg_valid = 1'b0;
        end
    endtask

    task automatic run_oracle_success;
        integer timeout;
        begin
            inject_mode_q = INJECT_NONE;
            oracle_mode_q = 1'b1;
            cfg_lane_mask = 4'b0111;
            result_ready = 1'b0;
            submit();
            timeout = 0;
            while (!result_valid && timeout < 5000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!result_valid || result_error || !oracle_checked_q ||
                (request_count_q != 256) || (response_count_q != 256))
                $fatal(1,
                    "oracle reduction failed result=%0d/%0d checked=%0d req=%0d rsp=%0d",
                    result_valid, result_error, oracle_checked_q,
                    request_count_q, response_count_q);
            result_ready = 1'b1;
            @(negedge clk);
            result_ready = 1'b0;
            while (!cfg_ready) @(negedge clk);
            oracle_mode_q = 1'b0;
            cfg_lane_mask = 4'b1010;
        end
    endtask

    task automatic run_error(
        input [1:0] mode,
        input [7:0] expected_status,
        input cancel_drain
    );
        integer timeout;
        reg [7:0] held_status;
        begin
            inject_mode_q = mode;
            result_ready = 1'b0;
            submit();
            timeout = 0;
            while (!bad_seen_q && timeout < 1000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!bad_seen_q || (bad_outstanding_q < 4))
                $fatal(1, "bad response lacked prefetched tail outstanding=%0d",
                       bad_outstanding_q);

            if (cancel_drain) begin
                abort_run = 1'b1;
                @(negedge clk);
                abort_run = 1'b0;
                timeout = 0;
                while (!cfg_ready && timeout < 1000) begin
                    if (result_valid)
                        $fatal(1, "aborted error drain published a result");
                    @(negedge clk);
                    timeout = timeout + 1;
                end
                if (!cfg_ready || source_count_q != 0 || result_valid)
                    $fatal(1, "abort did not drain source tail cleanly");
            end else begin
                timeout = 0;
                while (!result_valid && timeout < 1000) begin
                    @(negedge clk);
                    timeout = timeout + 1;
                end
                if (!result_valid || !result_error ||
                    (result_status != expected_status))
                    $fatal(1, "bad reduction result error=%0d status=%h",
                           result_error, result_status);
                if ((source_count_q != 0) ||
                    (dut.issue_count_q != dut.rsp_count_q))
                    $fatal(1, "error published before source drain");

                held_status = result_status;
                repeat (7) begin
                    @(negedge clk);
                    if (!result_valid || !result_error ||
                        (result_status != held_status) ||
                        (source_count_q != 0))
                        $fatal(1, "error result was not stable under backpressure");
                end
                result_ready = 1'b1;
                @(negedge clk);
                result_ready = 1'b0;
                while (!cfg_ready) @(negedge clk);
            end
        end
    endtask

    task automatic run_success;
        integer timeout;
        begin
            inject_mode_q = INJECT_NONE;
            result_ready = 1'b0;
            submit();
            timeout = 0;
            while (!result_valid && timeout < 5000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!result_valid || result_error || (result_status != 0) ||
                (request_count_q != 256) || (response_count_q != 256) ||
                (source_count_q != 0))
                $fatal(1,
                    "restart failed err=%0d status=%h req=%0d rsp=%0d queued=%0d",
                    result_error, result_status, request_count_q,
                    response_count_q, source_count_q);
            result_ready = 1'b1;
            @(negedge clk);
            result_ready = 1'b0;
            while (!cfg_ready) @(negedge clk);
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        run_error(INJECT_SOURCE, 8'h02, 1'b0);
        run_success();
        run_error(INJECT_NONFINITE, 8'h04, 1'b0);
        run_success();
        run_error(INJECT_SOURCE, 8'h02, 1'b1);
        run_success();
        run_oracle_success();

        $display(" rms_reduce4_tb PASS cycles=%0d", cycle);
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "rms reducer timeout state=%0d", dut.state_q);
    end
endmodule

`default_nettype wire
