`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [63:0] cmd_addr = 64'd0;
    reg [31:0] cmd_segment_beats = 32'd0;
    reg [31:0] cmd_stride_bytes = 32'd0;
    reg [16:0] cmd_repeats = 17'd0;
    wire [127:0] out_data;
    wire out_valid;
    reg out_ready = 1'b0;
    wire out_last;
    wire out_error;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

    wire [39:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [1:0] m_axi_arburst;
    wire m_axi_arvalid;
    reg m_axi_arready = 1'b0;
    reg [127:0] m_axi_rdata = 128'd0;
    reg [1:0] m_axi_rresp = 2'b00;
    reg m_axi_rlast = 1'b0;
    reg m_axi_rvalid = 1'b0;
    wire m_axi_rready;

     axi_read128 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_addr(cmd_addr),
        .cmd_segment_beats(cmd_segment_beats),
        .cmd_stride_bytes(cmd_stride_bytes), .cmd_repeats(cmd_repeats),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready),
        .out_last(out_last), .out_error(out_error), .busy(busy),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    // A second one-beat instance models a slave that presents RVALID in the
    // same cycle it accepts AR. It targets the canceled stalled-AR corner.
    reg zero_clear = 1'b0;
    reg zero_cmd_valid = 1'b0;
    wire zero_cmd_ready;
    wire zero_out_valid;
    wire zero_busy;
    wire zero_done_valid;
    wire [39:0] zero_araddr;
    wire [7:0] zero_arlen;
    wire [2:0] zero_arsize;
    wire [1:0] zero_arburst;
    wire zero_arvalid;
    reg zero_arready = 1'b0;
    wire zero_rready;
    reg zero_rsp_pending_q = 1'b0;
    wire zero_rvalid = zero_rsp_pending_q ||
                       (zero_arvalid && zero_arready);
    wire zero_r_fire = zero_rvalid && zero_rready;

     axi_read128 zero_dut (
        .clk(clk), .rst_n(rst_n), .clear(zero_clear), .abort_run(1'b0),
        .cmd_valid(zero_cmd_valid), .cmd_ready(zero_cmd_ready),
        .cmd_addr(64'h0000_0000_0000_d000), .cmd_segment_beats(32'd1),
        .cmd_stride_bytes(32'd0), .cmd_repeats(17'd1),
        .out_data(), .out_valid(zero_out_valid), .out_ready(1'b1),
        .out_last(), .out_error(), .busy(zero_busy),
        .done_valid(zero_done_valid), .done_ready(1'b1),
        .done_error(), .done_status(), .m_axi_araddr(zero_araddr),
        .m_axi_arlen(zero_arlen), .m_axi_arsize(zero_arsize),
        .m_axi_arburst(zero_arburst), .m_axi_arvalid(zero_arvalid),
        .m_axi_arready(zero_arready), .m_axi_rdata(128'hd00d),
        .m_axi_rresp(2'b00), .m_axi_rlast(1'b1),
        .m_axi_rvalid(zero_rvalid), .m_axi_rready(zero_rready)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            zero_rsp_pending_q <= 1'b0;
        end else if (zero_r_fire) begin
            zero_rsp_pending_q <= 1'b0;
        end else if (zero_arvalid && zero_arready) begin
            zero_rsp_pending_q <= 1'b1;
        end
    end

    function automatic [127:0] data_for(input [39:0] addr);
        data_for = {
            24'hc0ffee, addr,
            8'h5a, addr,
            8'ha5, addr
        };
    endfunction

    reg slave_active = 1'b0;
    reg force_ar_stall = 1'b0;
    reg [39:0] slave_addr = 40'd0;
    integer slave_left = 0;
    integer slave_beat = 0;
    integer inject_error_beat = -1;
    integer ar_count = 0;
    reg [39:0] expected_araddr[$];
    reg [8:0] expected_arbeats[$];

    // One-outstanding-burst AXI memory model with stable R payload under stall.
    always @(posedge clk) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b0;
            m_axi_rvalid <= 1'b0;
            m_axi_rdata <= 128'd0;
            m_axi_rresp <= 2'b00;
            m_axi_rlast <= 1'b0;
            slave_active <= 1'b0;
            slave_addr <= 40'd0;
            slave_left <= 0;
            slave_beat <= 0;
            ar_count <= 0;
        end else begin
            m_axi_arready <= !force_ar_stall && !slave_active &&
                             !m_axi_rvalid;

            if (m_axi_arvalid && m_axi_arready) begin
                reg [39:0] want_araddr;
                reg [8:0] want_arbeats;
                if ((expected_araddr.size() == 0) ||
                    (expected_arbeats.size() == 0))
                    $fatal(1, "unexpected AXI read burst addr=%h len=%0d",
                           m_axi_araddr, {1'b0, m_axi_arlen} + 9'd1);
                want_araddr = expected_araddr.pop_front();
                want_arbeats = expected_arbeats.pop_front();
                if ((m_axi_araddr != want_araddr) ||
                    ({1'b0, m_axi_arlen} + 9'd1 != want_arbeats))
                    $fatal(1,
                        "AXI burst plan mismatch got=%h/%0d want=%h/%0d",
                        m_axi_araddr, {1'b0, m_axi_arlen} + 9'd1,
                        want_araddr, want_arbeats);
                if (m_axi_arsize != 3'd4 || m_axi_arburst != 2'b01)
                    $fatal(1, "bad AR attributes");
                if (m_axi_araddr[3:0] != 4'd0)
                    $fatal(1, "unaligned AR address");
                if ({1'b0, m_axi_araddr[11:4]} +
                    {1'b0, m_axi_arlen} + 10'd1 > 10'd256)
                    $fatal(1, "burst crossed 4 KiB boundary");
                slave_active <= 1'b1;
                slave_addr <= m_axi_araddr;
                slave_left <= m_axi_arlen + 1;
                ar_count <= ar_count + 1;
            end

            if (slave_active && !m_axi_rvalid) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata <= data_for(slave_addr);
                m_axi_rresp <= slave_beat == inject_error_beat ?
                               2'b10 : 2'b00;
                m_axi_rlast <= slave_left == 1;
            end else if (m_axi_rvalid && m_axi_rready) begin
                slave_beat <= slave_beat + 1;
                if (m_axi_rlast) begin
                    m_axi_rvalid <= 1'b0;
                    m_axi_rlast <= 1'b0;
                    m_axi_rresp <= 2'b00;
                    slave_active <= 1'b0;
                end else begin
                    slave_addr <= slave_addr + 40'd16;
                    slave_left <= slave_left - 1;
                    m_axi_rdata <= data_for(slave_addr + 40'd16);
                    m_axi_rresp <= slave_beat + 1 == inject_error_beat ?
                                   2'b10 : 2'b00;
                    m_axi_rlast <= slave_left == 2;
                end
            end
        end
    end

    reg [39:0] expected_addr[$];
    reg expected_error[$];
    integer accepted_outputs = 0;
    reg random_ready = 1'b1;

    always @(negedge clk) begin
        if (!rst_n)
            out_ready <= 1'b0;
        else if (random_ready)
            out_ready <= ($urandom_range(0, 3) != 0);
    end

    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            reg [39:0] want_addr;
            reg want_error;
            reg want_last;
            if (expected_addr.size() == 0)
                $fatal(1, "unexpected output beat");
            want_addr = expected_addr.pop_front();
            want_error = expected_error.pop_front();
            want_last = expected_addr.size() == 0;
            if (out_data !== data_for(want_addr))
                $fatal(1, "data mismatch addr=%h", want_addr);
            if (out_error !== want_error)
                $fatal(1, "error mismatch addr=%h got=%b", want_addr,
                       out_error);
            if (out_last !== want_last)
                $fatal(1, "last mismatch addr=%h got=%b want=%b",
                       want_addr, out_last, want_last);
            accepted_outputs <= accepted_outputs + 1;
        end
    end

    task automatic expect_records(
        input [39:0] base,
        input integer segment_beats,
        input integer stride_bytes,
        input integer repeats,
        input integer error_beat
    );
        integer rep_i;
        integer beat_i;
        integer ordinal;
        begin
            ordinal = 0;
            for (rep_i = 0; rep_i < repeats; rep_i = rep_i + 1)
                for (beat_i = 0; beat_i < segment_beats;
                     beat_i = beat_i + 1) begin
                    expected_addr.push_back(base + rep_i * stride_bytes +
                                            beat_i * 16);
                    expected_error.push_back(ordinal == error_beat);
                    ordinal = ordinal + 1;
                end
        end
    endtask

    task automatic expect_burst(
        input [39:0] addr,
        input [8:0] beats
    );
        begin
            expected_araddr.push_back(addr);
            expected_arbeats.push_back(beats);
        end
    endtask

    task automatic launch(
        input [63:0] addr,
        input [31:0] segment_beats,
        input [31:0] stride_bytes,
        input [16:0] repeats
    );
        begin
            while (!cmd_ready) @(posedge clk);
            @(negedge clk);
            cmd_addr = addr;
            cmd_segment_beats = segment_beats;
            cmd_stride_bytes = stride_bytes;
            cmd_repeats = repeats;
            cmd_valid = 1'b1;
            @(posedge clk);
            if (!cmd_ready)
                $fatal(1, "command was not accepted");
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic wait_done(
        input reg want_error,
        input [7:0] want_status
    );
        begin
            while (!done_valid) @(posedge clk);
            if (done_error !== want_error || done_status !== want_status)
                $fatal(1, "done mismatch error=%b/%b status=%h/%h",
                       done_error, want_error, done_status, want_status);
            @(posedge clk);
        end
    endtask

    initial begin
        integer before_ar;
        integer before_outputs;
        reg [39:0] held_araddr;
        reg [7:0] held_arlen;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // A 600-beat segment starts one beat before a 4 KiB boundary.  It must
        // issue 1, 256, 256, and 87-beat legal bursts.
        inject_error_beat = -1;
        slave_beat = 0;
        before_ar = ar_count;
        expect_burst(40'h0000000ff0, 9'd1);
        expect_burst(40'h0000001000, 9'd256);
        expect_burst(40'h0000002000, 9'd256);
        expect_burst(40'h0000003000, 9'd87);
        expect_records(40'h0000000ff0, 600, 0, 1, -1);
        launch(64'h0000000000000ff0, 600, 0, 1);
        wait_done(1'b0, 8'h00);
        if (ar_count - before_ar != 4)
            $fatal(1, "expected four boundary-split bursts, got %0d",
                   ar_count - before_ar);
        if (expected_addr.size() != 0)
            $fatal(1, "contiguous records remain");

        // Four short strided segments exercise the KV-history shape.
        inject_error_beat = -1;
        slave_beat = 0;
        before_ar = ar_count;
        expect_burst(40'h0000002000, 9'd3);
        expect_burst(40'h0000002080, 9'd3);
        expect_burst(40'h0000002100, 9'd3);
        expect_burst(40'h0000002180, 9'd3);
        expect_records(40'h0000002000, 3, 16'h0080, 4, -1);
        launch(64'h0000000000002000, 3, 16'h0080, 4);
        wait_done(1'b0, 8'h00);
        if (ar_count - before_ar != 4)
            $fatal(1, "expected one burst per strided segment");

        // A slave error is attached to the affected record and the command.
        inject_error_beat = 3;
        slave_beat = 0;
        expect_burst(40'h0000005000, 9'd8);
        expect_records(40'h0000005000, 8, 0, 1, 3);
        launch(64'h0000000000005000, 8, 0, 1);
        wait_done(1'b1, 8'h02);
        inject_error_beat = -1;

        // Cancellation in the registered command-preparation cycle must not
        // leak an AR request or any completion from the staged payload.
        before_ar = ar_count;
        launch(64'h0000000000007000, 13, 0, 1);
        if (dut.state_q != 4'd8)
            $fatal(1, "command did not enter registered preparation");
        abort_run = 1'b1;
        @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        while (!cmd_ready) @(posedge clk);
        if ((ar_count != before_ar) || out_valid || done_valid)
            $fatal(1, "pre-AR abort leaked staged command state");

        // Abort with a buffered output.  The reader must drain the live burst,
        // discard every stale record, and accept a clean restart.
        slave_beat = 0;
        before_outputs = accepted_outputs;
        expect_burst(40'h0000008000, 9'd40);
        expect_records(40'h0000008000, 40, 0, 1, -1);
        launch(64'h0000000000008000, 40, 0, 1);
        while (accepted_outputs - before_outputs < 5) @(posedge clk);
        @(negedge clk);
        random_ready = 1'b0;
        out_ready = 1'b0;
        abort_run = 1'b1;
        expected_addr.delete();
        expected_error.delete();
        @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        while (!cmd_ready) @(posedge clk);
        if (out_valid)
            $fatal(1, "aborted output remained visible");
        random_ready = 1'b1;
        slave_beat = 0;
        expect_burst(40'h000000a000, 9'd7);
        expect_records(40'h000000a000, 7, 0, 1, -1);
        launch(64'h000000000000a000, 7, 0, 1);
        wait_done(1'b0, 8'h00);

        // Abort cannot withdraw an AR request already visible to a stalled
        // slave. The held request must handshake unchanged, then drain.
        force_ar_stall = 1'b1;
        @(posedge clk);
        expect_burst(40'h000000b000, 9'd9);
        launch(64'h000000000000b000, 9, 0, 1);
        wait(m_axi_arvalid);
        held_araddr = m_axi_araddr;
        held_arlen = m_axi_arlen;
        @(negedge clk);
        abort_run = 1'b1;
        repeat (3) begin
            @(posedge clk); #0.1;
            if (!m_axi_arvalid || m_axi_araddr != held_araddr ||
                m_axi_arlen != held_arlen)
                $fatal(1, "stalled AR changed across abort");
        end
        @(negedge clk);
        abort_run = 1'b0;
        force_ar_stall = 1'b0;
        while (!cmd_ready) @(posedge clk);
        if (out_valid)
            $fatal(1, "aborted stalled-AR output remained visible");

        // Global clear has the same AXI hold obligation and suppresses all
        // completion while the accepted burst is quarantined.
        force_ar_stall = 1'b1;
        @(posedge clk);
        expect_burst(40'h000000c000, 9'd11);
        launch(64'h000000000000c000, 11, 0, 1);
        wait(m_axi_arvalid);
        held_araddr = m_axi_araddr;
        held_arlen = m_axi_arlen;
        @(negedge clk);
        clear = 1'b1;
        repeat (3) begin
            @(posedge clk); #0.1;
            if (!m_axi_arvalid || m_axi_araddr != held_araddr ||
                m_axi_arlen != held_arlen)
                $fatal(1, "stalled AR changed across clear");
        end
        @(negedge clk);
        force_ar_stall = 1'b0;
        while (busy || slave_active || m_axi_rvalid) @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        while (!cmd_ready) @(posedge clk);
        if (out_valid || done_valid)
            $fatal(1, "cleared stalled-AR completion remained visible");

        // A zero-latency one-beat R response may be presented alongside the
        // eventual canceled AR handshake. RREADY must remain low until the
        // reader enters its post-AR drain state, or RLAST would be lost.
        zero_arready = 1'b0;
        @(negedge clk);
        zero_cmd_valid = 1'b1;
        @(posedge clk);
        assert(zero_cmd_ready);
        @(negedge clk);
        zero_cmd_valid = 1'b0;
        wait(zero_arvalid);
        zero_clear = 1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        zero_arready = 1'b1;
        #0.1;
        if (!zero_arvalid || !zero_rvalid || zero_rready)
            $fatal(1, "zero-latency canceled AR was not held safely");
        @(posedge clk); #0.1;
        if (!zero_rsp_pending_q || !zero_rready || !zero_busy)
            $fatal(1, "zero-latency response did not enter drain");
        @(posedge clk); #0.1;
        if (zero_rsp_pending_q || zero_busy || zero_out_valid ||
            zero_done_valid)
            $fatal(1, "zero-latency clear did not retire cleanly");
        @(negedge clk);
        zero_clear = 1'b0;
        zero_arready = 1'b0;
        while (!zero_cmd_ready) @(posedge clk);

        // Invalid commands terminate without touching AXI.
        before_ar = ar_count;
        launch(64'h000000000000a003, 7, 0, 1);
        wait_done(1'b1, 8'h01);
        if (ar_count != before_ar)
            $fatal(1, "invalid command issued AXI traffic");
        if ((expected_araddr.size() != 0) ||
            (expected_arbeats.size() != 0))
            $fatal(1, "unconsumed exact AXI burst expectations");

        $display(" axi_read128: boundary/stride/error/abort/clear PASS outputs=%0d",
                 accepted_outputs);
        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "timeout");
    end
endmodule
