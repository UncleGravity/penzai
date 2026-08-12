`default_nettype none

module flash_kernel_formal(input wire clk);
    localparam integer HEAD_DIM_MAX = 16;
    localparam integer MAX_HEADS = 4;
    localparam integer MAX_HEAD_KV = 2;
    localparam integer LANES = 8;
    localparam integer QBEATS = 2;
    localparam integer VBEATS = 2;
    localparam integer N_HEADS = 2;
    localparam integer N_HEAD_KV = 1;
    localparam integer N_KV = 2;
    localparam integer N_TOKENS = 1;
    localparam integer EXPECT_Q = N_TOKENS * N_HEADS * QBEATS;
    localparam integer EXPECT_K = N_TOKENS * N_KV * N_HEAD_KV * QBEATS;
    localparam integer EXPECT_V = N_TOKENS * N_KV * N_HEAD_KV * VBEATS;
    localparam integer EXPECT_MASK = N_TOKENS * N_KV;
    localparam integer EXPECT_O = N_TOKENS * N_HEADS * VBEATS;
`ifdef FORMAL_DIRECTED_COVER
    localparam integer EXPECT_PROCESSED_KV = 1;
`else
    localparam integer EXPECT_PROCESSED_KV = N_KV;
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
    reg [1:0] f_processed_kv = 0;
    reg f_cover_stalled = 1'b0;

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
        // One processed pair followed by one skipped pair reaches both mask
        // branches and the inter-KV barrier without a symbolic cover search.
        assume(mask_tdata == ((f_mask_count == 0) ? 16'h0000 : 16'hfc00));
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
            assume(n_kv == 16'd2);
            assume(n_tokens == 16'd1);
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
            f_processed_kv <= 0;
            f_cover_stalled <= 1'b0;
        end else begin
            if (start && !busy) begin
                f_run_active <= 1'b1;
                f_q_count <= 0;
                f_k_count <= 0;
                f_v_count <= 0;
                f_mask_count <= 0;
                f_o_count <= 0;
                f_processed_kv <= 0;
            end else if (done) begin
                f_run_active <= 1'b0;
            end

            if (q_fire) f_q_count <= f_q_count + 1'b1;
            if (k_fire) f_k_count <= f_k_count + 1'b1;
            if (v_fire) f_v_count <= f_v_count + 1'b1;
            if (mask_fire) begin
                f_mask_count <= f_mask_count + 1'b1;
                if (mask_tdata != 16'hfc00)
                    f_processed_kv <= f_processed_kv + 1'b1;
            end
            if (o_fire) f_o_count <= f_o_count + 1'b1;
            if (o_tvalid && !o_tready) f_cover_stalled <= 1'b1;
        end

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

            // Requesting the next mask is the externally visible KV barrier. K and
            // V for every preceding KV item, including masked items, must already
            // have been consumed in full.
            if (mask_fire) begin
                assert(f_k_count == f_mask_count * N_HEAD_KV * QBEATS);
                assert(f_v_count == f_mask_count * N_HEAD_KV * VBEATS);
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
                assert(f_v_count == EXPECT_V);
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
        cover(rst_n && done && f_processed_kv == EXPECT_PROCESSED_KV);
    end
endmodule

`default_nettype wire
