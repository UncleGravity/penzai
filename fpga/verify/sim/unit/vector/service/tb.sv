`timescale 1ns/1ps
`default_nettype none
`include "vector_defs.vh"

module vector_service_tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;

    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [3:0] cmd_op = `VECTOR_OP_ATTN_NORM;
    reg [7:0] cmd_token_mask = 8'h01;
    reg [7:0] cmd_hidden_blocks = 8'd64;
    reg [63:0] cmd_gamma_addr = 64'h1000;
    reg [31:0] cmd_epsilon = 32'h3727_c5ac;
    wire done_valid;
    reg done_ready = 1'b0;
    wire done_error;
    wire [15:0] done_status;
    wire [31:0] done_cycles;
    wire busy;
    wire [3:0] debug_state;

    wire gamma_req_valid;
    wire gamma_req_ready;
    wire [63:0] gamma_req_addr;
    wire [10:0] gamma_req_words;
    wire gamma_rsp_valid;
    wire gamma_rsp_ready;
    wire [127:0] gamma_rsp_data;
    wire gamma_rsp_last;
    wire gamma_rsp_error = 1'b0;

    wire r_rd_req_valid;
    wire r_rd_req_ready;
    wire r_rd_req_wave;
    wire [11:0] r_rd_req_addr;
    reg r_rd_rsp_valid = 1'b0;
    wire r_rd_rsp_ready;
    reg [127:0] r_rd_rsp_data = 128'd0;
    reg r_rd_rsp_error = 1'b0;
    reg r_mem_pending = 1'b0;
    reg r_mem_pending_error = 1'b0;
    reg inject_r_source_error = 1'b0;
    integer inject_r_request = 8;
    reg injected_error_seen = 1'b0;
    integer injected_error_outstanding = 0;

    wire q8_wr_valid;
    wire q8_wr_ready;
    wire q8_wr_wave;
    wire [8:0] q8_wr_addr;
    wire [3:0] q8_wr_lane_mask;
    wire [1087:0] q8_wr_data;

    wire leaf_q8_cfg_valid;
    wire leaf_q8_cfg_ready;
    wire [14:0] leaf_q8_cfg_rows;
    wire [3:0] leaf_q8_cfg_lane_mask;
    wire leaf_q8_busy;
    wire leaf_q8_in_valid;
    wire leaf_q8_in_ready;
    wire [127:0] leaf_q8_in_data;
    wire leaf_q8_out_valid;
    wire leaf_q8_out_ready;
    wire [8:0] leaf_q8_out_block;
    wire [1087:0] leaf_q8_out_data;
    wire [7:0] leaf_q8_out_status;
    wire leaf_q8_out_last;
    wire q8_collision_error;

     vector_service dut (.*);

     shared_q8 q8_service (
        .clk(clk), .rst_n(rst_n), .abort_run(clear || abort_run),
        .busy(leaf_q8_busy), .collision_error(q8_collision_error),
        .c0_cfg_valid(leaf_q8_cfg_valid),
        .c0_cfg_ready(leaf_q8_cfg_ready),
        .c0_cfg_rows(leaf_q8_cfg_rows),
        .c0_cfg_lane_mask(leaf_q8_cfg_lane_mask),
        .c0_in_valid(leaf_q8_in_valid), .c0_in_ready(leaf_q8_in_ready),
        .c0_in_data(leaf_q8_in_data), .c0_out_valid(leaf_q8_out_valid),
        .c0_out_ready(leaf_q8_out_ready),
        .c0_out_block(leaf_q8_out_block), .c0_out_data(leaf_q8_out_data),
        .c0_out_status(leaf_q8_out_status), .c0_out_last(leaf_q8_out_last),
        .c0_abort(1'b0),
        .c1_cfg_valid(1'b0), .c1_cfg_ready(), .c1_cfg_rows(15'd0),
        .c1_cfg_lane_mask(4'd0), .c1_in_valid(1'b0), .c1_in_ready(),
        .c1_in_data(128'd0), .c1_out_valid(), .c1_out_ready(1'b0),
        .c1_out_block(), .c1_out_data(), .c1_out_status(), .c1_out_last(),
        .c1_abort(1'b0), .c2_cfg_valid(1'b0), .c2_cfg_ready(),
        .c2_cfg_rows(15'd0), .c2_cfg_lane_mask(4'd0),
        .c2_in_valid(1'b0), .c2_in_ready(), .c2_in_data(128'd0),
        .c2_out_valid(), .c2_out_ready(1'b0), .c2_out_block(),
        .c2_out_data(), .c2_out_status(), .c2_out_last(), .c2_abort(1'b0)
    );

    integer cycle = 0;
    integer r_req_count = 0;
    integer q8_write_count = 0;
    integer gamma_request_count = 0;
    integer expected_q8_block = 0;
    reg expected_q8_wave = 1'b0;
    integer lane;
    always @(posedge clk)
        if (rst_n) cycle <= cycle + 1;

    wire r_rsp_slot_available = !r_rd_rsp_valid || r_rd_rsp_ready;
    wire r_pending_to_rsp = r_mem_pending && r_rsp_slot_available;
    assign r_rd_req_ready = (cycle[2:0] != 3'd2) &&
                            (!r_mem_pending || r_pending_to_rsp);
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            r_mem_pending <= 1'b0;
            r_mem_pending_error <= 1'b0;
            r_rd_rsp_valid <= 1'b0;
            r_rd_rsp_error <= 1'b0;
        end else begin
            if (r_pending_to_rsp) begin
                r_rd_rsp_valid <= 1'b1;
                r_rd_rsp_data <= {4{32'h3f80_0000}};
                r_rd_rsp_error <= r_mem_pending_error;
            end else if (r_rd_rsp_valid && r_rd_rsp_ready) begin
                r_rd_rsp_valid <= 1'b0;
                r_rd_rsp_error <= 1'b0;
            end
            if (r_rd_req_valid && r_rd_req_ready) begin
                r_mem_pending <= 1'b1;
                r_mem_pending_error <= inject_r_source_error &&
                                       (r_req_count == inject_r_request);
                r_req_count <= r_req_count + 1;
            end else if (r_pending_to_rsp) begin
                r_mem_pending <= 1'b0;
                r_mem_pending_error <= 1'b0;
            end
            if (r_rd_rsp_valid && r_rd_rsp_ready && r_rd_rsp_error) begin
                injected_error_seen <= 1'b1;
                injected_error_outstanding <=
                    integer'(dut.r_outstanding_q);
            end
        end
    end

    reg gamma_active = 1'b0;
    reg [10:0] gamma_words_q = 11'd0;
    reg [10:0] gamma_index_q = 11'd0;
    assign gamma_req_ready = !gamma_active && (cycle[1:0] != 2'd1);
    assign gamma_rsp_valid = gamma_active && (cycle[2:0] != 3'd5);
    assign gamma_rsp_data = {4{32'h3f80_0000}};
    assign gamma_rsp_last = gamma_active &&
                            ((gamma_index_q + 1'b1) == gamma_words_q);
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            gamma_active <= 1'b0;
            gamma_words_q <= 11'd0;
            gamma_index_q <= 11'd0;
        end else begin
            if (gamma_req_valid && gamma_req_ready) begin
                gamma_active <= 1'b1;
                gamma_words_q <= gamma_req_words;
                gamma_index_q <= 11'd0;
                gamma_request_count <= gamma_request_count + 1;
                if ((gamma_req_addr != cmd_gamma_addr) ||
                    (gamma_req_words != ({3'd0, cmd_hidden_blocks} << 3)))
                    $fatal(1, "bad gamma request contract");
            end
            if (gamma_rsp_valid && gamma_rsp_ready) begin
                if (gamma_rsp_last)
                    gamma_active <= 1'b0;
                else
                    gamma_index_q <= gamma_index_q + 1'b1;
            end
        end
    end

    assign q8_wr_ready = cycle[2:0] != 3'd1;
    always @(posedge clk) begin
        if (rst_n && q8_wr_valid && q8_wr_ready) begin
            q8_write_count <= q8_write_count + 1;
            if (q8_wr_wave != expected_q8_wave) begin
                if (!q8_wr_wave || (q8_wr_addr != 9'd0))
                    $fatal(1, "Q8 wave transition/order mismatch");
                expected_q8_wave <= 1'b1;
                expected_q8_block <= 1;
            end else begin
                if (q8_wr_addr != expected_q8_block[8:0])
                    $fatal(1, "Q8 block order mismatch");
                expected_q8_block <= expected_q8_block + 1;
            end
            if (q8_wr_lane_mask != (q8_wr_wave ?
                    cmd_token_mask[7:4] : cmd_token_mask[3:0]))
                $fatal(1, "Q8 sparse lane mask mismatch");
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (q8_wr_lane_mask[lane]) begin
                    if (q8_wr_data[lane*272 +: 256] !== {32{8'h7f}})
                        $fatal(1, "constant norm did not quantize to +127");
                    if (q8_wr_data[lane*272 + 256 +: 16] == 16'd0)
                        $fatal(1, "constant norm emitted zero Q8 scale");
                end else if (q8_wr_data[lane*272 +: 272] != 272'd0) begin
                    $fatal(1, "inactive Q8 lane was not zero");
                end
            end
        end
        if (rst_n && q8_collision_error)
            $fatal(1, "unexpected Q8 client collision");
    end

    function automatic integer wavecount(input [7:0] mask);
        wavecount = ((|mask[3:0]) ? 1 : 0) + ((|mask[7:4]) ? 1 : 0);
    endfunction

    task automatic clear_counts(input [7:0] mask);
        begin
            r_req_count = 0;
            q8_write_count = 0;
            gamma_request_count = 0;
            expected_q8_block = 0;
            expected_q8_wave = !(|mask[3:0]);
            injected_error_seen = 1'b0;
            injected_error_outstanding = 0;
        end
    endtask

    task automatic submit;
        begin
            @(negedge clk);
            cmd_valid = 1'b1;
            while (!cmd_ready) @(negedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic run_source_error_and_restart;
        integer timeout;
        begin
            clear_counts(8'h05);
            cmd_op = `VECTOR_OP_ATTN_NORM;
            cmd_hidden_blocks = 8'd64;
            cmd_token_mask = 8'h05;
            inject_r_source_error = 1'b1;
            done_ready = 1'b0;
            submit();
            timeout = 0;
            while (!done_valid && timeout < 10000) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!done_valid || !done_error ||
                (done_status != (`VECTOR_STATUS_REDUCE | 16'h0002)))
                $fatal(1, "vector source error mismatch status=%h",
                       done_status);
            if (!injected_error_seen ||
                (injected_error_outstanding < 2))
                $fatal(1,
                    "vector source error lacked multiple outstanding reads (%0d)",
                    injected_error_outstanding);
            if ((dut.r_outstanding_q != 0) || r_mem_pending ||
                r_rd_rsp_valid || dut.reduce_busy)
                $fatal(1, "vector published error before R drain");

            repeat (7) begin
                @(negedge clk);
                if (!done_valid || !done_error ||
                    (done_status != (`VECTOR_STATUS_REDUCE | 16'h0002)) ||
                    r_mem_pending || r_rd_rsp_valid ||
                    (dut.r_outstanding_q != 0))
                    $fatal(1, "vector error result/drain was not stable");
            end
            done_ready = 1'b1;
            @(negedge clk);
            done_ready = 1'b0;
            inject_r_source_error = 1'b0;
            while (!cmd_ready) @(negedge clk);

            run_norm(`VECTOR_OP_ATTN_NORM, 8'd64, 8'h05);
            $display("vector reducer source-error drain/restart PASS");
        end
    endtask

    task automatic wait_success(input integer limit);
        integer timeout;
        begin
            timeout = 0;
            while (!done_valid && timeout < limit) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!done_valid)
                $fatal(1, "vector timeout state=%0d cycles=%0d",
                       debug_state, done_cycles);
            if (done_error || done_status != 16'd0)
                $fatal(1, "vector command failed status=%h", done_status);
            @(negedge clk); done_ready = 1'b1;
            @(negedge clk); done_ready = 1'b0;
        end
    endtask

    task automatic run_norm(
        input [3:0] op,
        input [7:0] blocks,
        input [7:0] mask
    );
        integer rows;
        integer waves;
        begin
            clear_counts(mask);
            cmd_op = op;
            cmd_hidden_blocks = blocks;
            cmd_token_mask = mask;
            rows = blocks * 32;
            waves = wavecount(mask);
            submit();
            wait_success(250000);
            if ((gamma_request_count != 1) ||
                (r_req_count != rows * waves * 3) ||
                (q8_write_count != blocks * waves))
                $fatal(1, "norm count mismatch D=%0d g=%0d r=%0d q8=%0d",
                       rows, gamma_request_count, r_req_count,
                       q8_write_count);
            $display("norm op=%0d D=%0d mask=%h cycles=%0d PASS",
                     op, rows, mask, done_cycles);
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        // Removed post-store QK opcode is rejected without memory traffic.
        clear_counts(8'h01);
        cmd_op = 4'd2;
        cmd_hidden_blocks = 8'd64;
        cmd_token_mask = 8'h01;
        submit();
        while (!done_valid) @(negedge clk);
        if (!done_error || done_status != `VECTOR_STATUS_BAD_CMD ||
            gamma_request_count != 0 || r_req_count != 0)
            $fatal(1, "unsupported QK opcode was not rejected");
        @(negedge clk); done_ready = 1'b1;
        @(negedge clk); done_ready = 1'b0;

        run_norm(`VECTOR_OP_ATTN_NORM, 8'd64, 8'ha5);
        run_norm(`VECTOR_OP_FFN_NORM, 8'd80, 8'h0a);
        run_norm(`VECTOR_OP_FINAL_NORM, 8'd128, 8'ha0);
        run_source_error_and_restart();

        // Abort during the two-pass reducer, drain the outstanding R response,
        // and prove a sparse upper-wave-only restart is clean.
        clear_counts(8'ha5);
        cmd_op = `VECTOR_OP_ATTN_NORM;
        cmd_hidden_blocks = 8'd128;
        cmd_token_mask = 8'ha5;
        submit();
        while ((debug_state != 4'd5) || (r_req_count < 20))
            @(negedge clk);
        abort_run = 1'b1;
        @(negedge clk);
        abort_run = 1'b0;
        repeat (200) begin
            if (cmd_ready) break;
            if (done_valid)
                $fatal(1, "aborted vector command produced completion");
            @(negedge clk);
        end
        if (!cmd_ready)
            $fatal(1,
                "vector abort cleanup did not restore readiness state=%0d out=%0d model=%0d/%0d reduce=%0d apply=%0d q8=%0d gamma=%0d",
                debug_state, dut.r_outstanding_q, r_mem_pending,
                r_rd_rsp_valid, dut.reduce_busy, dut.apply_busy,
                leaf_q8_busy, dut.gamma_active_q);

        // Global clear invalidates an accepted arena request/response instead
        // of waiting for a response that the arena deliberately drops.
        clear_counts(8'h03);
        cmd_op = `VECTOR_OP_ATTN_NORM;
        cmd_hidden_blocks = 8'd64;
        cmd_token_mask = 8'h03;
        submit();
        while (!(r_rd_req_valid && r_rd_req_ready)) @(negedge clk);
        @(negedge clk);
        if (!r_mem_pending && !r_rd_rsp_valid)
            $fatal(1, "clear test did not create pipelined R response");
        clear = 1'b1;
        @(negedge clk);
        clear = 1'b0;
        while (!cmd_ready) @(negedge clk);
        if (r_mem_pending || r_rd_rsp_valid || done_valid)
            $fatal(1, "global clear retained vector response/completion");
        $display("vector global clear with outstanding R PASS");

        run_norm(`VECTOR_OP_FINAL_NORM, 8'd64, 8'h80);

        $display(" vector_service_tb PASS total_cycles=%0d", cycle);
        $finish;
    end

    initial begin
        #3000000;
        $fatal(1, "vector service timeout state=%0d", debug_state);
    end
endmodule

`default_nettype wire
