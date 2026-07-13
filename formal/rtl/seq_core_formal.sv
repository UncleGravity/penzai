`default_nettype none

module seq_core_formal (
    input wire clk
);
    localparam integer ADDR_W = 32;
    localparam integer COUNT_W = 4;
    localparam integer POLL_TIMEOUT = 2;
    localparam integer WATCHDOG_TIMEOUT = 4;

    (* anyseq *) reg                  rst_n;
    (* anyseq *) reg                  go;
    (* anyseq *) reg [COUNT_W-1:0]    desc_count;
    (* anyseq *) reg                  desc_gnt;
    (* anyseq *) reg [127:0]          desc_data;
    (* anyseq *) reg                  reg_gnt;
    (* anyseq *) reg [31:0]           reg_rdata;

    wire                 busy;
    wire                 done;
    wire                 err_timeout;
    wire                 err_watchdog;
    wire [COUNT_W-1:0]   err_index;
    wire                 desc_req;
    wire [COUNT_W-1:0]   desc_idx;
    wire                 reg_req;
    wire                 reg_we;
    wire [ADDR_W-1:0]    reg_addr;
    wire [31:0]          reg_wdata;

    seq_core #(
        .ADDR_W(ADDR_W),
        .COUNT_W(COUNT_W),
        .POLL_TIMEOUT(POLL_TIMEOUT),
        .WATCHDOG_TIMEOUT(WATCHDOG_TIMEOUT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .go(go),
        .desc_count(desc_count),
        .busy(busy),
        .done(done),
        .err_timeout(err_timeout),
        .err_watchdog(err_watchdog),
        .err_index(err_index),
        .desc_req(desc_req),
        .desc_idx(desc_idx),
        .desc_gnt(desc_gnt),
        .desc_data(desc_data),
        .reg_req(reg_req),
        .reg_we(reg_we),
        .reg_addr(reg_addr),
        .reg_wdata(reg_wdata),
        .reg_gnt(reg_gnt),
        .reg_rdata(reg_rdata)
    );

    reg f_past_valid = 1'b0;
    reg run_active = 1'b0;
    reg [COUNT_W-1:0] run_limit = {COUNT_W{1'b0}};
    reg saw_desc = 1'b0;
    reg [COUNT_W-1:0] last_desc_idx = {COUNT_W{1'b0}};

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        // Both downstream ports use the documented req/gnt protocol. Grants
        // are one-cycle responses and cannot appear without a live request.
        if (desc_gnt)
            assume(desc_req);
        if (reg_gnt)
            assume(reg_req);
        if (f_past_valid && $past(desc_gnt))
            assume(!desc_gnt);
        if (f_past_valid && $past(reg_gnt))
            assume(!reg_gnt);

        // `go` is a pulse. Pulses while busy are allowed and must be ignored.
        if (f_past_valid && $past(go))
            assume(!go);

        if (!rst_n) begin
            run_active <= 1'b0;
            run_limit <= {COUNT_W{1'b0}};
            saw_desc <= 1'b0;
            last_desc_idx <= {COUNT_W{1'b0}};
        end else begin
            if (go && !busy) begin
                run_active <= 1'b1;
                run_limit <= desc_count;
                saw_desc <= 1'b0;
            end else if (done) begin
                run_active <= 1'b0;
            end

            if (desc_req && desc_gnt) begin
                if (saw_desc)
                    assert(desc_idx == last_desc_idx + {{(COUNT_W-1){1'b0}}, 1'b1});
                else
                    assert(desc_idx == {COUNT_W{1'b0}});
                saw_desc <= 1'b1;
                last_desc_idx <= desc_idx;
            end
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!busy && !done);
            assert(!err_timeout && !err_watchdog);
            assert(!desc_req && !reg_req);
        end

        if (rst_n) begin
            assert(!(busy && done));
            assert(!(err_timeout && err_watchdog));
            assert(!(desc_req && reg_req));
            if (desc_req || reg_req)
                assert(busy);

            // A run is defined by the count sampled with its accepted go.
            if (desc_req) begin
                assert(run_active);
                assert(desc_idx < run_limit);
            end
            if (err_timeout || err_watchdog) begin
                assert(err_index < run_limit);
            end

            // Requests and their payloads remain stable until grant.
            if (f_past_valid && $past(rst_n) && $past(desc_req && !desc_gnt)) begin
                assert(desc_req || err_watchdog);
                if (desc_req)
                    assert(desc_idx == $past(desc_idx));
            end
            if (f_past_valid && $past(rst_n) && $past(reg_req && !reg_gnt)) begin
                assert(reg_req || err_watchdog);
                if (reg_req) begin
                    assert(reg_we == $past(reg_we));
                    assert(reg_addr == $past(reg_addr));
                    assert(reg_wdata == $past(reg_wdata));
                end
            end

            // Completion and errors are levels until the next accepted run.
            if (f_past_valid && $past(rst_n) && $past(done && !go))
                assert(done);
            if (f_past_valid && $past(rst_n) && $past(err_timeout && !go))
                assert(err_timeout);
            if (f_past_valid && $past(rst_n) && $past(err_watchdog && !go))
                assert(err_watchdog);
            if (f_past_valid && $past(rst_n) && $past(err_timeout || err_watchdog) &&
                (err_timeout || err_watchdog)) begin
                assert(done);
                assert(!busy);
                assert(!desc_req && !reg_req);
            end
        end

        // Cover every semantic exit and both register operation types.
        cover(rst_n && done && run_limit == 0 && !err_timeout && !err_watchdog);
        cover(rst_n && done && saw_desc && !err_timeout && !err_watchdog);
        cover(rst_n && reg_req && reg_we);
        cover(rst_n && reg_req && !reg_we);
        cover(rst_n && done && err_timeout);
        cover(rst_n && done && err_watchdog);
    end
endmodule

`default_nettype wire
