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

    reg [127:0] in_data = 128'd0;
    reg in_valid = 1'b0;
    wire in_ready;
    reg in_last = 1'b0;
    reg in_error = 1'b0;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b0;
    wire done_error;
    wire [7:0] done_status;

    wire [39:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [1:0] m_axi_awburst;
    wire m_axi_awvalid;
    reg m_axi_awready = 1'b0;
    wire [127:0] m_axi_wdata;
    wire [15:0] m_axi_wstrb;
    wire m_axi_wlast;
    wire m_axi_wvalid;
    reg m_axi_wready = 1'b0;
    reg [1:0] m_axi_bresp = 2'b00;
    reg m_axi_bvalid = 1'b0;
    wire m_axi_bready;

     axi_write128 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_addr(cmd_addr),
        .cmd_segment_beats(cmd_segment_beats),
        .cmd_stride_bytes(cmd_stride_bytes), .cmd_repeats(cmd_repeats),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
        .in_last(in_last), .in_error(in_error), .busy(busy),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready), .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready)
    );

    function automatic [127:0] payload_for(input integer ordinal);
        payload_for = {
            32'hc0010000 ^ ordinal,
            32'h5a5a0000 + ordinal,
            32'ha5a50000 - ordinal,
            ordinal[31:0]
        };
    endfunction

    reg [39:0] expected_aw_addr[$];
    integer expected_aw_beats[$];
    reg [39:0] expected_w_addr[$];
    reg [127:0] expected_w_data[$];
    reg [15:0] expected_w_strb[$];

    task automatic expect_command(
        input [39:0] base,
        input integer segment_beats,
        input integer stride_bytes,
        input integer repeats,
        input integer data_seed
    );
        integer rep_i;
        integer remain;
        integer burst;
        integer boundary;
        integer beat_i;
        integer ordinal;
        reg [39:0] current;
        begin
            ordinal = 0;
            for (rep_i = 0; rep_i < repeats; rep_i = rep_i + 1) begin
                current = base + rep_i * stride_bytes;
                remain = segment_beats;
                while (remain != 0) begin
                    boundary = 32'd256 - {24'd0, current[11:4]};
                    burst = remain > 256 ? 256 : remain;
                    if (burst > boundary)
                        burst = boundary;
                    expected_aw_addr.push_back(current);
                    expected_aw_beats.push_back(burst);
                    for (beat_i = 0; beat_i < burst;
                         beat_i = beat_i + 1) begin
                        expected_w_addr.push_back(current + beat_i * 16);
                        expected_w_data.push_back(
                            payload_for(data_seed + ordinal));
                        expected_w_strb.push_back(16'hffff);
                        ordinal = ordinal + 1;
                    end
                    current = current + burst * 16;
                    remain = remain - burst;
                end
            end
        end
    endtask

    task automatic null_expectations_from(input integer first);
        integer i;
        begin
            for (i = first; i < expected_w_strb.size(); i = i + 1)
                expected_w_strb[i] = 16'h0000;
        end
    endtask

    task automatic require_empty;
        begin
            if ((expected_aw_addr.size() != 0) ||
                (expected_aw_beats.size() != 0) ||
                (expected_w_addr.size() != 0) ||
                (expected_w_data.size() != 0) ||
                (expected_w_strb.size() != 0))
                $fatal(1, "expected AXI records remain aw=%0d w=%0d",
                       expected_aw_addr.size(), expected_w_addr.size());
        end
    endtask

    reg slave_active = 1'b0;
    reg [39:0] slave_addr = 40'd0;
    integer slave_left = 0;
    integer active_aw_ordinal = -1;
    integer inject_bresp_aw = -1;
    integer aw_count = 0;
    integer w_count = 0;
    integer b_count = 0;
    integer null_w_count = 0;
    reg force_aw_stall = 1'b0;
    reg force_w_stall = 1'b0;

    // Randomized one-outstanding AXI slave.  The expectation queues validate
    // both burst partitioning and every write address/strobe.
    always @(negedge clk) begin
        if (!rst_n) begin
            m_axi_awready = 1'b0;
            m_axi_wready = 1'b0;
        end else begin
            m_axi_awready = !force_aw_stall && !slave_active &&
                            !m_axi_bvalid &&
                            ($urandom_range(0, 3) != 0);
            m_axi_wready = !force_w_stall && slave_active &&
                           !m_axi_bvalid &&
                           ($urandom_range(0, 3) != 0);
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            slave_active <= 1'b0;
            slave_addr <= 40'd0;
            slave_left <= 0;
            active_aw_ordinal <= -1;
            m_axi_bvalid <= 1'b0;
            m_axi_bresp <= 2'b00;
            aw_count <= 0;
            w_count <= 0;
            b_count <= 0;
            null_w_count <= 0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                reg [39:0] want_addr;
                integer want_beats;
                if (slave_active || m_axi_bvalid)
                    $fatal(1, "more than one write burst outstanding");
                if (expected_aw_addr.size() == 0)
                    $fatal(1, "unexpected AW");
                want_addr = expected_aw_addr.pop_front();
                want_beats = expected_aw_beats.pop_front();
                if (m_axi_awaddr !== want_addr ||
                    m_axi_awlen !== (want_beats[7:0] - 8'd1))
                    $fatal(1, "AW mismatch addr=%h/%h beats=%0d/%0d",
                           m_axi_awaddr, want_addr, m_axi_awlen + 1,
                           want_beats);
                if (m_axi_awsize != 3'd4 || m_axi_awburst != 2'b01)
                    $fatal(1, "bad AW attributes");
                if ({1'b0, m_axi_awaddr[11:4]} +
                    {1'b0, m_axi_awlen} + 10'd1 > 10'd256)
                    $fatal(1, "AW burst crossed 4 KiB boundary");
                slave_active <= 1'b1;
                slave_addr <= m_axi_awaddr;
                slave_left <= {24'd0, m_axi_awlen} + 32'd1;
                active_aw_ordinal <= aw_count;
                aw_count <= aw_count + 1;
            end

            if (m_axi_wvalid && m_axi_wready) begin
                reg [39:0] want_addr;
                reg [127:0] want_data;
                reg [15:0] want_strb;
                if (!slave_active)
                    $fatal(1, "W arrived without an AW");
                if (expected_w_addr.size() == 0)
                    $fatal(1, "unexpected W");
                want_addr = expected_w_addr.pop_front();
                want_data = expected_w_data.pop_front();
                want_strb = expected_w_strb.pop_front();
                if (slave_addr !== want_addr)
                    $fatal(1, "W address mismatch %h/%h",
                           slave_addr, want_addr);
                if (m_axi_wstrb !== want_strb)
                    $fatal(1, "WSTRB mismatch addr=%h got=%h want=%h",
                           slave_addr, m_axi_wstrb, want_strb);
                if ((want_strb != 16'h0000) &&
                    (m_axi_wdata !== want_data))
                    $fatal(1, "WDATA mismatch addr=%h", slave_addr);
                if (m_axi_wlast !== (slave_left == 1))
                    $fatal(1, "WLAST mismatch left=%0d got=%b",
                           slave_left, m_axi_wlast);
                if (m_axi_wstrb == 16'h0000)
                    null_w_count <= null_w_count + 1;
                w_count <= w_count + 1;
                if (slave_left == 1) begin
                    slave_active <= 1'b0;
                    slave_left <= 0;
                    m_axi_bvalid <= 1'b1;
                    m_axi_bresp <=
                        active_aw_ordinal == inject_bresp_aw ? 2'b10 : 2'b00;
                end else begin
                    slave_addr <= slave_addr + 40'd16;
                    slave_left <= slave_left - 1;
                end
            end

            if (m_axi_bvalid && m_axi_bready) begin
                m_axi_bvalid <= 1'b0;
                m_axi_bresp <= 2'b00;
                b_count <= b_count + 1;
            end
        end
    end

    reg prior_aw_stall = 1'b0;
    reg [39:0] prior_awaddr = 40'd0;
    reg [7:0] prior_awlen = 8'd0;
    reg prior_w_stall = 1'b0;
    reg [127:0] prior_wdata = 128'd0;
    reg [15:0] prior_wstrb = 16'd0;
    reg prior_wlast = 1'b0;

    // Explicitly check the stability contract under randomized stalls and
    // while abort changes the mover's control state.
    always @(posedge clk) begin
        if (!rst_n) begin
            prior_aw_stall <= 1'b0;
            prior_w_stall <= 1'b0;
        end else begin
            if (prior_aw_stall &&
                (!m_axi_awvalid || m_axi_awaddr !== prior_awaddr ||
                 m_axi_awlen !== prior_awlen))
                $fatal(1, "AW changed while stalled");
            if (prior_w_stall &&
                (!m_axi_wvalid || m_axi_wdata !== prior_wdata ||
                 m_axi_wstrb !== prior_wstrb ||
                 m_axi_wlast !== prior_wlast))
                $fatal(1, "W changed while stalled");
            prior_aw_stall <= m_axi_awvalid && !m_axi_awready;
            prior_awaddr <= m_axi_awaddr;
            prior_awlen <= m_axi_awlen;
            prior_w_stall <= m_axi_wvalid && !m_axi_wready;
            prior_wdata <= m_axi_wdata;
            prior_wstrb <= m_axi_wstrb;
            prior_wlast <= m_axi_wlast;
            if (done_valid && busy)
                $fatal(1, "DONE was reported busy");
        end
    end

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

    task automatic send_records(
        input integer count,
        input integer data_seed,
        input integer last_index,
        input integer error_index
    );
        integer i;
        begin
            for (i = 0; i < count; i = i + 1) begin
                @(negedge clk);
                in_data = payload_for(data_seed + i);
                in_last = i == last_index;
                in_error = i == error_index;
                in_valid = 1'b1;
                @(posedge clk);
                while (!in_ready) @(posedge clk);
            end
            @(negedge clk);
            in_valid = 1'b0;
            in_last = 1'b0;
            in_error = 1'b0;
        end
    endtask

    task automatic wait_done(
        input reg want_error,
        input [7:0] want_status
    );
        begin
            while (!done_valid) @(negedge clk);
            if (done_error !== want_error || done_status !== want_status)
                $fatal(1, "done mismatch error=%b/%b status=%h/%h",
                       done_error, want_error, done_status, want_status);
            repeat (2) @(posedge clk);
            if (!done_valid || done_error !== want_error ||
                done_status !== want_status)
                $fatal(1, "done response was not held");
            @(negedge clk);
            done_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            done_ready = 1'b0;
        end
    endtask

    initial begin
        integer before_aw;
        integer before_w;
        integer i;
        reg [127:0] held_wdata;
        reg [15:0] held_wstrb;
        reg held_wlast;

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // One beat remains before the first 4 KiB boundary.  Legal bursts
        // are 1, 256, 256, and 87 beats.
        expect_command(40'h0000000ff0, 600, 0, 1, 1000);
        launch(64'h0000000000000ff0, 600, 0, 1);
        send_records(600, 1000, 599, -1);
        wait_done(1'b0, 8'h00);
        require_empty();

        // Four independent KV-shaped records use one burst per stride.
        expect_command(40'h0000005000, 3, 32'h00000080, 4, 2000);
        launch(64'h0000000000005000, 3, 32'h00000080, 4);
        send_records(12, 2000, 11, -1);
        wait_done(1'b0, 8'h00);
        require_empty();

        // A BRESP error stops the command after the failed live burst.
        before_aw = aw_count;
        inject_bresp_aw = before_aw;
        expect_command(40'h0000008000, 8, 0, 1, 3000);
        launch(64'h0000000000008000, 8, 0, 1);
        send_records(8, 3000, 7, -1);
        wait_done(1'b1, 8'h02);
        require_empty();
        inject_bresp_aw = -1;

        // Early LAST suppresses that record and null-fills the accepted AW.
        expect_command(40'h0000009000, 8, 0, 1, 4000);
        null_expectations_from(2);
        launch(64'h0000000000009000, 8, 0, 1);
        send_records(3, 4000, 2, -1);
        wait_done(1'b1, 8'h03);
        require_empty();

        // Missing LAST on the final record is also a framing error.
        expect_command(40'h000000a000, 6, 0, 1, 5000);
        null_expectations_from(5);
        launch(64'h000000000000a000, 6, 0, 1);
        send_records(6, 5000, -1, -1);
        wait_done(1'b1, 8'h03);
        require_empty();

        // Upstream errors have a distinct status and use identical draining.
        expect_command(40'h000000b000, 7, 0, 1, 6000);
        null_expectations_from(2);
        launch(64'h000000000000b000, 7, 0, 1);
        send_records(3, 6000, -1, 2);
        wait_done(1'b1, 8'h04);
        require_empty();

        // Abort while W is held.  The already-visible record remains stable;
        // every following record is a zero-strobe quarantine write.
        expect_command(40'h000000c000, 40, 0, 1, 7000);
        launch(64'h000000000000c000, 40, 0, 1);
        before_w = w_count;
        fork : abort_sender
            send_records(40, 7000, 39, -1);
        join_none
        while (w_count - before_w < 5) @(posedge clk);
        @(negedge clk);
        force_w_stall = 1'b1;
        @(posedge clk);
        @(negedge clk);
        while (!(m_axi_wvalid && !m_axi_wready)) @(negedge clk);
        if (expected_w_strb.size() == 0)
            $fatal(1, "no held W expectation at abort");
        for (i = 1; i < expected_w_strb.size(); i = i + 1)
            expected_w_strb[i] = 16'h0000;
        held_wdata = m_axi_wdata;
        held_wstrb = m_axi_wstrb;
        held_wlast = m_axi_wlast;
        abort_run = 1'b1;
        repeat (2) begin
            @(posedge clk);
            if (!m_axi_wvalid || m_axi_wdata !== held_wdata ||
                m_axi_wstrb !== held_wstrb ||
                m_axi_wlast !== held_wlast)
                $fatal(1, "abort mutated stalled W");
        end
        @(negedge clk);
        abort_run = 1'b0;
        disable abort_sender;
        in_valid = 1'b0;
        in_last = 1'b0;
        in_error = 1'b0;
        force_w_stall = 1'b0;
        while (!cmd_ready) @(posedge clk);
        require_empty();
        if (done_valid)
            $fatal(1, "external abort leaked a completion");

        // A fresh command cannot inherit the aborted stream or B response.
        expect_command(40'h000000d000, 7, 0, 1, 8000);
        launch(64'h000000000000d000, 7, 0, 1);
        send_records(7, 8000, 6, -1);
        wait_done(1'b0, 8'h00);
        require_empty();

        // Clear cannot retract a stalled AW.  Hold the exact AW record until
        // acceptance, then issue a complete zero-strobe burst and return idle.
        expect_command(40'h000000e000, 4, 0, 1, 9000);
        null_expectations_from(0);
        force_aw_stall = 1'b1;
        launch(64'h000000000000e000, 4, 0, 1);
        while (!(m_axi_awvalid && !m_axi_awready)) @(negedge clk);
        clear = 1'b1;
        repeat (2) @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        force_aw_stall = 1'b0;
        while (!cmd_ready) @(posedge clk);
        require_empty();
        if (done_valid)
            $fatal(1, "clear leaked a completion");

        // Invalid commands complete locally without touching AXI.
        before_aw = aw_count;
        launch(64'h000000000000e003, 7, 0, 1);
        wait_done(1'b1, 8'h01);
        if (aw_count != before_aw)
            $fatal(1, "invalid command issued AW");
        require_empty();

        $display(" axi_write128: boundary/stride/BRESP/LAST/abort/clear PASS aw=%0d w=%0d null=%0d",
                 aw_count, w_count, null_w_count);
        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "timeout");
    end
endmodule
