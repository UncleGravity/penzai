`default_nettype none

module flash_kernel_harness(input wire clk);
    localparam integer HEAD_DIM_MAX = 16;
    localparam integer MAX_HEADS = 4;
    localparam integer MAX_HEAD_KV = 2;
    localparam integer LANES = 8;
    localparam integer QBEATS = 2;
    localparam integer VBEATS = 2;
    localparam integer N_HEADS = 2;
    localparam integer N_HEAD_KV = 1;
    // Two resident queries expose sparse query-slot ordering. Two KV positions
    // exhaust the finite/all-masked transition pairs under arbitrary prove masks;
    // the directed trace uses a mixed KV followed by an all-masked KV.
    localparam integer N_KV = 2;
    localparam integer N_TOKENS = 2;
    localparam integer EXPECT_Q = N_TOKENS * N_HEADS * QBEATS;
    localparam integer EXPECT_K = N_KV * N_HEAD_KV * QBEATS;
    localparam integer EXPECT_V = N_KV * N_HEAD_KV * VBEATS;
    localparam integer EXPECT_MASK = N_TOKENS * N_KV;
    localparam integer EXPECT_O = N_TOKENS * N_HEADS * VBEATS;
`ifdef FORMAL_DIRECTED_COVER
    localparam integer EXPECT_PROCESSED_PAIRS = 1;
`else
    localparam integer EXPECT_PROCESSED_PAIRS = N_TOKENS * N_KV;
`endif

    (* anyseq *) reg rst_n;
    (* anyseq *) reg start;
    (* anyseq *) reg [15:0] head_dim_q;
    (* anyseq *) reg [15:0] head_dim_v;
    (* anyseq *) reg [15:0] n_heads;
    (* anyseq *) reg [15:0] n_head_kv;
    (* anyseq *) reg [15:0] head_ratio;
    (* anyseq *) reg [15:0] n_kv;
    (* anyseq *) reg [15:0] n_tokens;
    (* anyseq *) reg [31:0] scale;
    (* anyseq *) reg [15:0] mask_tdata;
    (* anyseq *) reg o_tready;

    wire [255:0] q_tdata = 256'd0;
    wire q_tvalid = 1'b1;
    wire [127:0] k_tdata = 128'd0;
    wire k_tvalid = 1'b1;
    wire [127:0] v_tdata = 128'd0;
    wire v_tvalid = 1'b1;
    wire mask_tvalid = 1'b1;

    wire busy, done;
    wire q_tready, k_tready, v_tready, mask_tready;
    wire [255:0] o_tdata;
    wire o_tvalid, o_tlast;
    wire [31:0] o_tkeep;

    flash_kernel #(
        .HEAD_DIM_MAX(HEAD_DIM_MAX),
        .MAX_HEADS(MAX_HEADS),
        .MAX_HEAD_KV(MAX_HEAD_KV),
        .MAX_TOKENS(4),
        // Two narrow-layout queries occupy slots 0..1 and 16..17. A 32-slot
        // instance is the smallest faithful controller proof; the dedicated
        // max-shape harness proves the production 64-slot map separately.
        .MAX_SLOTS(32),
        .LANES(LANES)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .head_dim_q(head_dim_q), .head_dim_v(head_dim_v),
        .n_heads(n_heads), .n_head_kv(n_head_kv), .head_ratio(head_ratio),
        .n_kv(n_kv), .n_tokens(n_tokens), .scale(scale),
        .busy(busy), .done(done),
        .q_tdata(q_tdata), .q_tvalid(q_tvalid), .q_tready(q_tready),
        .k_tdata(k_tdata), .k_tvalid(k_tvalid), .k_tready(k_tready),
        .v_tdata(v_tdata), .v_tvalid(v_tvalid), .v_tready(v_tready),
        .mask_tdata(mask_tdata), .mask_tvalid(mask_tvalid), .mask_tready(mask_tready),
        .o_tdata(o_tdata), .o_tvalid(o_tvalid), .o_tready(o_tready),
        .q_tlast(1'b0), .q_tkeep(32'hffffffff),
        .k_tlast(1'b0), .k_tkeep(16'hffff),
        .v_tlast(1'b0), .v_tkeep(16'hffff),
        .mask_tlast(1'b0), .mask_tkeep(2'b11),
        .o_tlast(o_tlast), .o_tkeep(o_tkeep)
    );

    reg f_past_valid = 1'b0;
    reg f_run_active = 1'b0;
    reg [7:0] f_q_count = 0;
    reg [7:0] f_k_count = 0;
    reg [7:0] f_v_count = 0;
    reg [7:0] f_mask_count = 0;
    reg [7:0] f_o_count = 0;
    reg [3:0] f_processed_pairs = 0;
    reg f_cover_stalled = 1'b0;
    reg f_cover_mixed = 1'b0;
    reg f_cover_all_masked = 1'b0;
    reg [1:0] f_kv_finite_count = 0;
    reg [N_TOKENS-1:0] f_query_ever_finite = 0;
    reg [1:0] f_o_stall_count = 0;
    reg [9:0] f_run_cycles = 0;

    wire q_fire = q_tvalid && q_tready;
    wire k_fire = k_tvalid && k_tready;
    wire v_fire = v_tvalid && v_tready;
    wire mask_fire = mask_tvalid && mask_tready;
    wire o_fire = o_tvalid && o_tready;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) begin
            assume(!rst_n);
            assume(!start);
        end else begin
            assume(rst_n);
            assume(start == !$past(rst_n));
        end
`ifdef FORMAL_DIRECTED_COVER
        // KV-major masks for two causal queries:
        //   kv0: q0 masked, q1 finite; kv1: both masked.
        // Thus q0 is all-masked for the whole tile while q1 has a finite causal
        // prefix. This reaches sparse-slot and all-masked-KV paths in one trace.
        assume(mask_tdata == ((f_mask_count == 1) ?
                              16'h0000 : 16'hfc00));
        // Force exactly one output stall, then drain the packet immediately.
        if (o_tvalid && !f_cover_stalled)
            assume(!o_tready);
        else
            assume(o_tready);
`else
        assume(mask_tdata == 16'h0000 || mask_tdata == 16'hfc00);
`endif
        if (start && !busy) begin
            assume(head_dim_q == 16'd16);
            assume(head_dim_v == 16'd16);
            assume(n_heads == 16'd2);
            assume(n_head_kv == 16'd1);
            assume(head_ratio == 16'd2);
            assume(n_kv == N_KV);
            assume(n_tokens == N_TOKENS);
            assume(scale == 32'h3f000000);
        end

        // The control proof drives input streams continuously. Randomized input
        // bubbles and independent stream skew are exercised by RTL cosim; formal
        // keeps arbitrary output backpressure and both mask branches.
        if (f_past_valid && $past(rst_n) && $past(mask_tvalid && !mask_tready)) begin
            assume(mask_tdata == $past(mask_tdata));
        end

        if (!rst_n) begin
            f_run_active <= 1'b0;
            f_q_count <= 0;
            f_k_count <= 0;
            f_v_count <= 0;
            f_mask_count <= 0;
            f_o_count <= 0;
            f_processed_pairs <= 0;
            f_cover_stalled <= 1'b0;
            f_cover_mixed <= 1'b0;
            f_cover_all_masked <= 1'b0;
            f_kv_finite_count <= 0;
            f_query_ever_finite <= 0;
            f_o_stall_count <= 0;
            f_run_cycles <= 0;
        end else begin
            if (start && !busy) begin
                f_run_active <= 1'b1;
                f_q_count <= 0;
                f_k_count <= 0;
                f_v_count <= 0;
                f_mask_count <= 0;
                f_o_count <= 0;
                f_processed_pairs <= 0;
                f_cover_mixed <= 1'b0;
                f_cover_all_masked <= 1'b0;
                f_kv_finite_count <= 0;
                f_query_ever_finite <= 0;
                f_run_cycles <= 0;
            end else if (done) begin
                f_run_active <= 1'b0;
            end

            if (f_run_active)
                f_run_cycles <= f_run_cycles + 1'b1;

            // A bounded fairness assumption is necessary for a liveness proof:
            // the output consumer may stall, but not forever. Stream inputs are
            // explicitly always valid above.
            if (o_tvalid && !o_tready)
                f_o_stall_count <= f_o_stall_count + 1'b1;
            else
                f_o_stall_count <= 0;
            if (o_tvalid && f_o_stall_count == 2)
                assume(o_tready);

            if (q_fire) f_q_count <= f_q_count + 1'b1;
            if (k_fire) f_k_count <= f_k_count + 1'b1;
            if (v_fire) f_v_count <= f_v_count + 1'b1;
            if (mask_fire) begin
                f_mask_count <= f_mask_count + 1'b1;
                if (mask_tdata != 16'hfc00) begin
                    f_processed_pairs <= f_processed_pairs + 1'b1;
                    f_kv_finite_count <= f_kv_finite_count + 1'b1;
                    f_query_ever_finite[f_mask_count % N_TOKENS] <= 1'b1;
                end
                if ((f_mask_count % N_TOKENS) + 1 == N_TOKENS) begin
                    if (f_kv_finite_count + (mask_tdata != 16'hfc00) == 1)
                        f_cover_mixed <= 1'b1;
                    if (f_kv_finite_count == 0 && mask_tdata == 16'hfc00)
                        f_cover_all_masked <= 1'b1;
                    f_kv_finite_count <= 0;
                end
            end
            if (o_fire) f_o_count <= f_o_count + 1'b1;
            if (o_tvalid && !o_tready) f_cover_stalled <= 1'b1;
        end

`ifndef FORMAL_LIVENESS_ONLY
`ifndef FORMAL_COMPLETION_ONLY
        if (f_past_valid && !$past(rst_n)) begin
            assert(!busy && !done);
            assert(!o_tvalid);
        end

        if (rst_n) begin
            assert(!(busy && done));
            assert(!q_tready || busy);
            assert(!k_tready || busy);
            assert(!v_tready || busy);
            assert(!mask_tready || busy);
            assert(!o_tvalid || f_run_active);

            assert(f_q_count <= EXPECT_Q);
            assert(f_k_count <= EXPECT_K);
            assert(f_v_count <= EXPECT_V);
            assert(f_mask_count <= EXPECT_MASK);
            assert(f_o_count <= EXPECT_O);
            assert(f_processed_pairs <= EXPECT_MASK);
            // Before accepting the first mask for a KV position, the KV-outer
            // scheduler has retired exactly one preceding K/V block per completed
            // KV position, independent of query count or mask density.
            if (mask_fire) begin
                assert(f_k_count == (f_mask_count / N_TOKENS) * N_HEAD_KV * QBEATS);
                assert(f_v_count == (f_mask_count / N_TOKENS) * N_HEAD_KV * VBEATS);
            end

            if (o_tvalid) begin
                assert(o_tkeep == 32'hffffffff);
                assert(o_tlast == (f_o_count + 1'b1 == EXPECT_O));
            end else begin
                assert(!o_tlast);
            end

            if (done) begin
                assert(f_q_count == EXPECT_Q);
                assert(f_k_count == EXPECT_K);
                assert(f_mask_count == EXPECT_MASK);
                assert(f_o_count == EXPECT_O);
            end

            if (f_past_valid && $past(rst_n) &&
                $past(o_tvalid && !o_tready)) begin
                assert(o_tvalid);
                assert(o_tdata == $past(o_tdata));
                assert(o_tkeep == $past(o_tkeep));
                assert(o_tlast == $past(o_tlast));
            end
        end

        cover(rst_n && mask_fire && mask_tdata == 16'hfc00);
        cover(rst_n && mask_fire && mask_tdata != 16'hfc00);
        cover(rst_n && o_tvalid && !o_tready);
        cover(rst_n && done);
        cover(rst_n && done && f_processed_pairs == EXPECT_PROCESSED_PAIRS);
        cover(rst_n && done && f_cover_mixed && f_cover_all_masked);
        cover(rst_n && done && !f_query_ever_finite[0] && f_query_ever_finite[1] &&
              f_o_count == EXPECT_O);
`else
        // The seventh late completion property is the external V-stream total.
        if (rst_n && done)
            assert(f_v_count == EXPECT_V);

        // Close the harness half of the terminal transition. Controller state and
        // output closure are asserted beside the internal completion properties.
        if (f_past_valid && $past(rst_n) && $past(done))
            assert(!f_run_active);
`endif
`endif

        // This watchdog is a bounded-liveness property, not an inductive
        // invariant. The dedicated liveness BMC exhausts the complete run before
        // this 10-bit counter could wrap. Its reduced cone contains the real
        // controller plus only the fairness assumptions needed for progress.
`ifdef FORMAL_BOUNDED_LIVENESS
        if (rst_n) assert(f_run_cycles < 10'd256);
`endif
    end
endmodule

`default_nettype wire
