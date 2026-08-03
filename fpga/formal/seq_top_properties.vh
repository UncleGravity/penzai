    // These properties are included inside seq_top only for formal builds, so
    // they can observe the BRAM address contract without adding production ports.
    reg f_top_past_valid = 1'b0;
    reg f_run_active = 1'b0;
    reg [COUNT_W-1:0] f_run_start = {COUNT_W{1'b0}};
    reg f_rewrote_start = 1'b0;

    always @(posedge clk) begin
        f_top_past_valid <= 1'b1;

        if (!rst_n) begin
            f_run_active <= 1'b0;
            f_run_start <= {COUNT_W{1'b0}};
            f_rewrote_start <= 1'b0;
        end else if (!core_rst_n) begin
            f_run_active <= 1'b0;
            f_rewrote_start <= 1'b0;
        end else begin
            if (go_strobe && !busy) begin
                f_run_active <= 1'b1;
                f_run_start <= run_start;
                f_rewrote_start <= 1'b0;
            end else if (done) begin
                f_run_active <= 1'b0;
            end

            if (write_commit && !cmd_sel && awaddr_q == OFF_RUN_START && busy)
                f_rewrote_start <= 1'b1;
        end

        if (rst_n && core_rst_n) begin
            if (desc_req) begin
                assert(f_run_active);
                assert(rd_idx == f_run_start[CMD_DEPTH_LOG2-1:0] +
                                 desc_idx[CMD_DEPTH_LOG2-1:0]);
            end

            assert(!(busy && done));
            if (go_strobe && busy)
                assert($stable(f_run_start));

            // ABORT intentionally resets this master mid-transaction. Away from that
            // documented reset boundary, it must obey AXI backpressure stability.
            if (f_top_past_valid && $past(core_rst_n) &&
                $past(reg_awvalid && !reg_awready)) begin
                assert(reg_awvalid);
                assert(reg_awaddr == $past(reg_awaddr));
            end
            if (f_top_past_valid && $past(core_rst_n) &&
                $past(reg_wvalid && !reg_wready)) begin
                assert(reg_wvalid);
                assert(reg_wdata == $past(reg_wdata));
                assert(reg_wstrb == $past(reg_wstrb));
            end
            if (f_top_past_valid && $past(core_rst_n) &&
                $past(reg_arvalid && !reg_arready)) begin
                assert(reg_arvalid);
                assert(reg_araddr == $past(reg_araddr));
            end
        end

        if (f_top_past_valid && !$past(core_rst_n)) begin
            assert(!busy && !done);
            assert(!err_timeout && !err_watchdog);
            assert(!desc_req && !reg_req);
        end

        cover(rst_n && f_run_active && desc_req);
        cover(rst_n && f_rewrote_start && desc_req);
        cover(rst_n && done && !err_timeout && !err_watchdog);
        cover(rst_n && done && err_timeout);
        cover(rst_n && done && err_watchdog);
        cover(rst_n && abort_strobe && busy);
    end
