// flash_kernel - flash attention, kv-major (v3).
//
// Consumes the KV cache in its NATIVE layout — kv-position-major, n_head_kv heads,
// NOT GQA-replicated — so the device DMAs Q/K/V/mask straight from the resident
// tensors with no host gather (the v1 gather was ~89% of a decode call). GQA is done
// here: one kv-head's K/V fans out to its `head_ratio` query heads. All n_heads carry
// an independent online-softmax state (m,l,acc) at once, so streaming kv-major updates
// every head per kv. The dot is pipelined (stream head_dim/8 beats through fp_dot into
// a 16-deep buffer, sum with numeric/reduce — no per-beat recurrence), replacing v1's
// ~320-cyc/kv sequential dot. O is emitted 8-wide packed.
//
//   load Q[token] (all heads, resident); init pool m=-inf,l=0,acc=0
//   for kv:
//     read mask[kv] (one value, all heads); if -inf: consume+skip K[kv],V[kv]
//     load K[kv] block (n_head_kv heads); for each query head h (kvh=h/ratio):
//        dot = treeSum(fp_dot beats of Q[h]·K[kvh]); score=dot*scale+mask
//        (m[h],l[h],p[h],corr[h]) = softmax(m[h],l[h],score)
//     load V[kv] block; for each query head h:  acc[h] = acc[h]*corr[h] + p[h]·V[kvh]
//   finalize per head:  O[h] = acc[h] * recip(l[h])  -> M_AXIS (8 f32 / beat)
//
// Heads are streamed through the dot/score/softmax and AXPY pipelines while one KV
// position is active.  A hard barrier after the final tagged AXPY writeback preserves
// the online-softmax recurrence before the next KV position starts.  Cosim-checked
// bit-for-bit against the previous sequential schedule and against flash_ref.

`default_nettype none

module flash_kernel #(
    parameter integer HEAD_DIM_MAX = 128,
    parameter integer MAX_HEADS    = 32,
    parameter integer MAX_HEAD_KV  = 8,
    parameter integer LANES        = 8
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [15:0] head_dim_q,
    input  wire [15:0] head_dim_v,
    input  wire [15:0] n_heads,
    input  wire [15:0] n_head_kv,
    input  wire [15:0] head_ratio,   // n_heads / n_head_kv (device-computed; no divider here)
    input  wire [15:0] n_kv,
    input  wire [15:0] n_tokens,
    input  wire [31:0] scale,
    output reg         busy,
    output reg         done,

    input  wire [255:0] q_tdata,    input wire q_tvalid,    output wire q_tready,
    input  wire [127:0] k_tdata,    input wire k_tvalid,    output wire k_tready,
    input  wire [127:0] v_tdata,    input wire v_tvalid,    output wire v_tready,
    input  wire [15:0]  mask_tdata, input wire mask_tvalid, output wire mask_tready,
    output wire [255:0] o_tdata,    output wire o_tvalid,   input wire o_tready,
    // Sideband the DMA/converter path expects; the kernel counts beats from the shape
    // registers, so input TLAST/TKEEP are unused. Output TLAST marks the final O beat
    // and TKEEP is all-valid for clean S2MM / data-width-converter packet framing.
    input  wire        q_tlast,    input  wire [31:0] q_tkeep,
    input  wire        k_tlast,    input  wire [15:0] k_tkeep,
    input  wire        v_tlast,    input  wire [15:0] v_tkeep,
    input  wire        mask_tlast, input  wire [1:0]  mask_tkeep,
    output wire        o_tlast,    output wire [31:0] o_tkeep
);
    `include "fmt.vh"
    localparam integer MAXB  = HEAD_DIM_MAX / LANES;         // 16 max Q/K/V beats per head
    localparam integer LSH   = $clog2(LANES);                // 3
    localparam integer LMAXB = $clog2(MAXB);                 // 4 (beat occupies the low bits)
    localparam integer AW    = $clog2(MAX_HEADS * MAXB);     // acc/q_buf addr width (9)
    localparam integer KW    = $clog2(MAX_HEAD_KV * MAXB);   // k_buf/v_buf addr width (7)
    localparam integer HW    = $clog2(MAX_HEADS);            // head-index width for the per-head pools
    localparam [3:0] SOFT_HOLD_RELOAD = 4'd7;
    localparam [31:0]  NEG_INF = 32'hFF800000;
    localparam [15:0]  F16_NEG_INF = 16'hFC00;

    // Snapshot the accepted command.  The AXI-Lite wrapper normally holds these
    // registers stable while busy, but the kernel must not depend on that convention.
    reg [15:0] head_dim_q_r, head_dim_v_r, n_heads_r, n_head_kv_r;
    reg [15:0] head_ratio_r, n_kv_r, n_tokens_r;
    reg [31:0] scale_r;
    wire [15:0] qbeats = head_dim_q_r >> LSH;
    wire [15:0] vbeats = head_dim_v_r >> LSH;

    // ---- resident / pooled storage (indexed [head*MAXB + beat], MAXB a power of 2) ----
    // Lever 2: q/k/v/acc are forced to block RAM (synchronous read below) so they stop
    // burning LUTs as distributed RAM. ram_style=block makes the intent explicit.
    (* ram_style = "block" *) reg [255:0] q_buf [0:MAX_HEADS*MAXB-1];   // Q for the current token, 8 f32/beat
    (* ram_style = "block" *) reg [127:0] k_buf [0:MAX_HEAD_KV*MAXB-1]; // K for the current kv, 8 f16/beat
    (* ram_style = "block" *) reg [127:0] v_buf [0:MAX_HEAD_KV*MAXB-1]; // V for the current kv, 8 f16/beat
    (* ram_style = "block" *) reg [255:0] acc   [0:MAX_HEADS*MAXB-1];   // running output accumulator, 8 f32/beat
    // Synchronous read-data registers for the four block RAMs (data lands one cycle
    // after the address; consumers' valids are delayed one cycle to match).
    reg [255:0] q_rd;
    reg [127:0] k_rd;
    reg [127:0] v_rd;
    reg [255:0] acc_rd;
    reg [31:0]  mpool [0:MAX_HEADS-1];        // running max per head
    reg [31:0]  lpool [0:MAX_HEADS-1];        // running denom per head
    reg [31:0]  ppool [0:MAX_HEADS-1];        // this-kv softmax weight per head
    reg [31:0]  cpool [0:MAX_HEADS-1];        // this-kv rescale factor per head
    reg [31:0]  score_pool [0:MAX_HEADS-1];    // completed scores awaiting softmax issue
    reg [31:0]  dot_parts [0:MAXB-1];         // per-beat fp_dot partials for the active head

    reg [15:0] tok_i, kv_i, head_i;
    reg [15:0] ld_a, ld_b;     // two-level load/clear counters (outer head/kvh, inner beat)
    reg [15:0] dot_req_head, dot_req_kvh, dot_req_ratio, dot_req_beat;
    reg [15:0] dot_cap_head, dot_cap_beat, score_head, soft_req_head, soft_wb_head;
    reg [15:0] ax_req_head, ax_req_kvh, ax_req_ratio, ax_req_beat;
    reg [15:0] ax_wb_head, ax_wb_beat;
    reg        dot_req_done, tree_pending, v_load_done, ax_req_done;
    reg [MAX_HEADS-1:0] score_ready, pc_ready;
    reg [3:0] soft_hold;
    reg [15:0] bi;             // emit beat index
    reg [31:0] mask_f32_q, inv_l_q;
    reg [31:0] soft_score_q, soft_m_q, soft_l_q;
    reg [255:0] emit_buf;

    localparam [4:0]
        S_IDLE   = 5'd0,  S_QLOAD  = 5'd1,  S_TINIT  = 5'd2,  S_MASK   = 5'd3,
        S_KLOAD  = 5'd4,  S_PAIR   = 5'd5,
        S_KSKIP  = 5'd16, S_VSKIP  = 5'd17, S_KVNEXT = 5'd18, S_FRECF  = 5'd19,
        S_FRECW  = 5'd20, S_FEMITF = 5'd21, S_FEMITW = 5'd22, S_FEMITE = 5'd23,
        S_TNEXT  = 5'd24, S_DONE   = 5'd25;
    reg [4:0] state;

    // ---- combinational addressing: address = head·MAXB + beat. MAXB is a power of
    // two and beat < MAXB, so this is a concat (head/kv-head upper, beat low LMAXB
    // bits) — no multiplier, exact width. ----
    wire [AW-1:0] q_dot_addr = {dot_req_head[AW-LMAXB-1:0], dot_req_beat[LMAXB-1:0]};
    wire [KW-1:0] k_dot_addr = {dot_req_kvh[KW-LMAXB-1:0],  dot_req_beat[LMAXB-1:0]};
    wire [AW-1:0] q_ld_addr  = {ld_a[AW-LMAXB-1:0],   ld_b[LMAXB-1:0]};
    wire [KW-1:0] kv_ld_addr = {ld_a[KW-LMAXB-1:0],   ld_b[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_rd  = {ax_req_head[AW-LMAXB-1:0], ax_req_beat[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_wr  = {ax_wb_head[AW-LMAXB-1:0],  ax_wb_beat[LMAXB-1:0]};
    wire [AW-1:0] acc_em_rd  = {head_i[AW-LMAXB-1:0], bi[LMAXB-1:0]};
    wire [KW-1:0] v_ax_addr  = {ax_req_kvh[KW-LMAXB-1:0], ax_req_beat[LMAXB-1:0]};

    // ---- fire pulses + last-beat flags ----
    wire dot_fire  = (state == S_PAIR) && !dot_req_done;
    wire tree_fire = tree_pending;
    wire smul_fire = tree_v;
    wire sadd_fire = smul_v;
    reg  soft_fire;
    wire rec_fire  = (state == S_FRECF);
    wire axpy_fire = (state == S_PAIR) && v_load_done && !ax_req_done &&
                     pc_ready[ax_req_head[HW-1:0]];
    wire emit_fire = (state == S_FEMITF);

    wire last_head = (head_i + 16'd1 == n_heads_r);
    wire last_kv   = (kv_i  + 16'd1 == n_kv_r);
    wire last_tok  = (tok_i + 16'd1 == n_tokens_r);

    // ---- fp_dot: one beat's Q·K (feed-forward, latency 16). q_buf/k_buf are
    // synchronous-read BRAMs (q_rd/k_rd land one cycle after the address), so the issue
    // strobe is delayed one cycle to match. The dot loop is self-timed off dot_v, so the
    // extra cycle just shifts the whole pipeline by one. ----
    reg dot_fire_q;
    always @(posedge clk) begin
        if (!rst_n) dot_fire_q <= 1'b0;
        else        dot_fire_q <= dot_fire;
    end
    wire        dot_v;  wire [31:0] dot_beat;
    fp_dot u_dot (.clk(clk), .rst_n(rst_n), .valid_in(dot_fire_q),
        .q(q_rd), .k(k_rd), .valid_out(dot_v), .sum(dot_beat));

    // ---- reduction tree: sum the per-beat partials (zero-padded, latency 16) ----
    wire [511:0] tree_in;
    genvar g;
    generate
        for (g = 0; g < MAXB; g = g + 1) begin : g_tree_in
            assign tree_in[g*32 +: 32] = (g < qbeats) ? dot_parts[g] : 32'd0;
        end
    endgenerate
    wire tree_v;  wire [31:0] tree_sum;
    reduce #(.MANT_W(FMT_FP32_MANT), .N(MAXB)) u_tree (.clk(clk), .rst_n(rst_n), .valid_in(tree_fire),
        .in(tree_in), .valid_out(tree_v), .sum(tree_sum));

    // ---- score = dot*scale + mask ----
    wire smul_v; wire [31:0] smul_out;
    fmul #(.MANT_W(FMT_FP32_MANT)) u_smul (.clk(clk), .rst_n(rst_n), .valid_in(smul_fire),
        .a(tree_sum), .b(scale_r), .valid_out(smul_v), .out(smul_out));
    wire sadd_v; wire [31:0] sadd_out;
    fadd #(.MANT_W(FMT_FP32_MANT)) u_sadd (.clk(clk), .rst_n(rst_n), .valid_in(sadd_fire),
        .a(smul_out), .b(mask_f32_q), .valid_out(sadd_v), .out(sadd_out));

    // ---- online softmax step (per head) ----
    wire soft_v; wire [31:0] soft_m, soft_l, soft_p, soft_corr; wire soft_grew;
    flash_softmax u_soft (.clk(clk), .rst_n(rst_n), .valid_in(soft_fire),
        .m_in(soft_m_q), .l_in(soft_l_q), .score(soft_score_q), .valid_out(soft_v),
        .m_out(soft_m), .l_out(soft_l), .p(soft_p), .corr(soft_corr), .grew(soft_grew));

    // ---- reciprocal of the denominator (per head, at finalize) ----
    wire rec_v; wire [31:0] rec_out;
    recip u_recip (.clk(clk), .rst_n(rst_n), .valid_in(rec_fire),
        .l(lpool[head_i[HW-1:0]]), .valid_out(rec_v), .y(rec_out));

    // ---- shared 8-wide axpy: acc*corr + p·V (accumulate) OR acc*inv_l (emit) ----
    // acc and V are now synchronous-read BRAMs (acc_rd / v_rd arrive one cycle after the
    // address is presented), so the matching control strobes are delayed one cycle to
    // realign with the data. ax_axpy_q selects v_rd (accumulate) vs zero (emit).
    wire [31:0]  ax_s1 = emit_fire ? inv_l_q : cpool[ax_req_head[HW-1:0]];
    wire [31:0]  ax_p  = emit_fire ? 32'd0   : ppool[ax_req_head[HW-1:0]];
    wire ax_fire = axpy_fire | emit_fire;
    reg        ax_fire_q, ax_axpy_q;
    reg [31:0] ax_s1_q, ax_p_q;
    always @(posedge clk) begin
        if (!rst_n) begin ax_fire_q <= 1'b0; ax_axpy_q <= 1'b0; end
        else        begin ax_fire_q <= ax_fire; ax_axpy_q <= axpy_fire; end
        ax_s1_q <= ax_s1;
        ax_p_q  <= ax_p;
    end
    wire [127:0] ax_v_eff = ax_axpy_q ? v_rd : 128'd0;
    wire ax_v;  wire [255:0] ax_out;
    fp_axpy8 u_axpy (.clk(clk), .rst_n(rst_n), .valid_in(ax_fire_q),
        .acc(acc_rd), .v(ax_v_eff), .s1(ax_s1_q), .p(ax_p_q), .valid_out(ax_v), .out(ax_out));

    // ---- block-RAM storage: each array gets one write port + one synchronous read
    // port — the template Vivado maps to BRAM instead of LUTRAM. ----
    // q_buf: written while loading Q; read by the head-streamed dot sequencer.
    always @(posedge clk) begin
        if ((state == S_QLOAD) && q_tvalid) q_buf[q_ld_addr] <= q_tdata;
        q_rd <= q_buf[q_dot_addr];
    end
    // k_buf: written while loading K (S_KLOAD); read during the dot pass.
    always @(posedge clk) begin
        if ((state == S_KLOAD) && k_tvalid) k_buf[kv_ld_addr] <= k_tdata;
        k_rd <= k_buf[k_dot_addr];
    end
    // v_buf: filled beside the score pipeline, then read by the AXPY sequencer.
    always @(posedge clk) begin
        if ((state == S_PAIR) && v_tvalid && v_tready) v_buf[kv_ld_addr] <= v_tdata;
        v_rd <= v_buf[v_ax_addr];
    end
    // acc: write port muxes the per-token zero-init (S_TINIT) and the axpy writeback
    // (S_AXPY, on ax_v); read port muxes the axpy and emit addresses. Issue leads
    // writeback by the axpy latency, so a beat's read and its write never share a cycle.
    wire          acc_we    = (state == S_TINIT) || ((state == S_PAIR) && ax_v);
    wire [AW-1:0] acc_waddr = (state == S_TINIT) ? q_ld_addr : acc_ax_wr;
    wire [255:0]  acc_wdata = (state == S_TINIT) ? 256'd0    : ax_out;
    wire [AW-1:0] acc_raddr = emit_fire ? acc_em_rd : acc_ax_rd;
    always @(posedge clk) begin
        if (acc_we) acc[acc_waddr] <= acc_wdata;
        acc_rd <= acc[acc_raddr];
    end

    wire [31:0] mask_f32_w;
    cvt_f16_f32 u_maskw (.in(mask_tdata), .out(mask_f32_w));

    // ---- outputs ----
    assign o_tdata  = emit_buf;
    assign o_tvalid = (state == S_FEMITE);
    assign o_tkeep  = 32'hFFFFFFFF;
    assign o_tlast  = (state == S_FEMITE) && (bi + 16'd1 == vbeats) && last_head && last_tok;
    assign q_tready    = (state == S_QLOAD);
    assign k_tready    = (state == S_KLOAD) || (state == S_KSKIP);
    assign v_tready    = ((state == S_PAIR) && !v_load_done) || (state == S_VSKIP);
    assign mask_tready = (state == S_MASK);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            dot_req_done <= 1'b0; tree_pending <= 1'b0;
            v_load_done <= 1'b0; ax_req_done <= 1'b0;
            soft_fire <= 1'b0;
            score_ready <= {MAX_HEADS{1'b0}}; pc_ready <= {MAX_HEADS{1'b0}};
        end else begin
            done <= 1'b0;
            // The score boundary is deliberately registered.  `score_head` is the
            // ordered completion tag; arithmetic pipelines never reorder results.
            soft_fire <= 1'b0;
            tree_pending <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    head_dim_q_r <= head_dim_q;
                    head_dim_v_r <= head_dim_v;
                    n_heads_r <= n_heads;
                    n_head_kv_r <= n_head_kv;
                    head_ratio_r <= head_ratio;
                    n_kv_r <= n_kv;
                    n_tokens_r <= n_tokens;
                    scale_r <= scale;
                    busy <= 1'b1; tok_i <= 0; ld_a <= 0; ld_b <= 0; state <= S_QLOAD;
                end

                // ---- load Q for the current token: n_heads × qbeats beats ----
                S_QLOAD: if (q_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_heads_r) begin ld_a <= 0; ld_b <= 0; state <= S_TINIT; end
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- init the per-head pool for this token (acc=0, m=-inf, l=0) ----
                S_TINIT: begin
                    if (ld_b == 16'd0) begin mpool[ld_a[HW-1:0]] <= NEG_INF; lpool[ld_a[HW-1:0]] <= 32'd0; end
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_heads_r) begin kv_i <= 0; state <= S_MASK; end
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- read this kv's mask (shared by all heads); branch skip/process ----
                S_MASK: if (mask_tvalid) begin
                    mask_f32_q <= mask_f32_w;
                    ld_a <= 0; ld_b <= 0;
                    state <= (mask_tdata == F16_NEG_INF) ? S_KSKIP : S_KLOAD;
                end

                // ---- load K[kv]: n_head_kv heads × qbeats beats (native, non-replicated) ----
                S_KLOAD: if (k_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv_r) begin
                            ld_a <= 0; ld_b <= 0;
                            dot_req_head <= 0; dot_req_kvh <= 0; dot_req_ratio <= 0; dot_req_beat <= 0;
                            dot_cap_head <= 0; dot_cap_beat <= 0; score_head <= 0;
                            soft_req_head <= 0; soft_wb_head <= 0; soft_hold <= 0;
                            dot_req_done <= 1'b0; tree_pending <= 1'b0; soft_fire <= 1'b0;
                            score_ready <= {MAX_HEADS{1'b0}}; pc_ready <= {MAX_HEADS{1'b0}};
                            v_load_done <= 1'b0;
                            ax_req_head <= 0; ax_req_kvh <= 0; ax_req_ratio <= 0; ax_req_beat <= 0;
                            ax_wb_head <= 0; ax_wb_beat <= 0; ax_req_done <= 1'b0;
                            state <= S_PAIR;
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- one KV phase: stream heads through score and accumulator pipelines ----
                S_PAIR: begin
                    // Dot requests are contiguous across head boundaries.  Q/K are
                    // synchronous BRAMs, so dot_fire_q supplies the matching valid.
                    if (dot_fire) begin
                        if (dot_req_beat + 16'd1 == qbeats) begin
                            dot_req_beat <= 0;
                            if (dot_req_head + 16'd1 == n_heads_r) begin
                                dot_req_done <= 1'b1;
                            end else begin
                                dot_req_head <= dot_req_head + 16'd1;
                                if (dot_req_ratio + 16'd1 == head_ratio_r) begin
                                    dot_req_ratio <= 0;
                                    dot_req_kvh <= dot_req_kvh + 16'd1;
                                end else dot_req_ratio <= dot_req_ratio + 16'd1;
                            end
                        end else dot_req_beat <= dot_req_beat + 16'd1;
                    end

                    // fp_dot completions are ordered.  On the cycle after a head's
                    // final partial is captured, the tree samples the old dot_parts
                    // bank while beat zero of the next head may be written.
                    if (dot_v) begin
                        dot_parts[dot_cap_beat[LMAXB-1:0]] <= dot_beat;
                        if (dot_cap_beat + 16'd1 == qbeats) begin
                            dot_cap_beat <= 0;
                            tree_pending <= 1'b1;
                            if (dot_cap_head + 16'd1 != n_heads_r)
                                dot_cap_head <= dot_cap_head + 16'd1;
                        end else dot_cap_beat <= dot_cap_beat + 16'd1;
                    end

                    if (sadd_v) begin
                        score_pool[score_head[HW-1:0]] <= sadd_out;
                        score_ready[score_head[HW-1:0]] <= 1'b1;
                        if (score_head + 16'd1 != n_heads_r)
                            score_head <= score_head + 16'd1;
                    end

                    // The existing softmax leaf requires its input metadata to remain
                    // stable for several composition seams.  Scores queue per head and
                    // launch no closer than eight cycles; production heads arrive every
                    // 16 cycles, so this constraint is free at the supported geometry.
                    if (soft_hold != 0) begin
                        soft_hold <= soft_hold - 1'b1;
                    end else if (score_ready[soft_req_head[HW-1:0]]) begin
                        soft_score_q <= score_pool[soft_req_head[HW-1:0]];
                        soft_m_q <= mpool[soft_req_head[HW-1:0]];
                        soft_l_q <= lpool[soft_req_head[HW-1:0]];
                        score_ready[soft_req_head[HW-1:0]] <= 1'b0;
                        soft_fire <= 1'b1;
                        soft_hold <= SOFT_HOLD_RELOAD;
                        if (soft_req_head + 16'd1 != n_heads_r)
                            soft_req_head <= soft_req_head + 16'd1;
                    end

                    if (soft_v) begin
                        mpool[soft_wb_head[HW-1:0]] <= soft_m;
                        lpool[soft_wb_head[HW-1:0]] <= soft_l;
                        ppool[soft_wb_head[HW-1:0]] <= soft_p;
                        cpool[soft_wb_head[HW-1:0]] <= soft_corr;
                        pc_ready[soft_wb_head[HW-1:0]] <= 1'b1;
                        if (soft_wb_head + 16'd1 != n_heads_r)
                            soft_wb_head <= soft_wb_head + 16'd1;
                    end

                    // V is loaded through the other BRAM port while scores run.
                    if (v_tvalid && v_tready) begin
                        if (ld_b + 16'd1 == vbeats) begin
                            ld_b <= 0;
                            if (ld_a + 16'd1 == n_head_kv_r) begin
                                ld_a <= 0;
                                v_load_done <= 1'b1;
                            end else ld_a <= ld_a + 16'd1;
                        end else ld_b <= ld_b + 16'd1;
                    end

                    // A head starts only after all V data and its own p/c commit are
                    // visible.  Requests and writebacks use independent ordered tags.
                    if (axpy_fire) begin
                        if (ax_req_beat + 16'd1 == vbeats) begin
                            ax_req_beat <= 0;
                            if (ax_req_head + 16'd1 == n_heads_r) begin
                                ax_req_done <= 1'b1;
                            end else begin
                                ax_req_head <= ax_req_head + 16'd1;
                                if (ax_req_ratio + 16'd1 == head_ratio_r) begin
                                    ax_req_ratio <= 0;
                                    ax_req_kvh <= ax_req_kvh + 16'd1;
                                end else ax_req_ratio <= ax_req_ratio + 16'd1;
                            end
                        end else ax_req_beat <= ax_req_beat + 16'd1;
                    end

                    if (ax_v) begin
                        if (ax_wb_beat + 16'd1 == vbeats) begin
                            ax_wb_beat <= 0;
                            if (ax_wb_head + 16'd1 == n_heads_r) begin
                                // Hard KV barrier: the final accumulator write occurs
                                // on this edge before KVNEXT can accept another mask.
                                state <= S_KVNEXT;
                            end else ax_wb_head <= ax_wb_head + 16'd1;
                        end else ax_wb_beat <= ax_wb_beat + 16'd1;
                    end
                end

                // ---- masked kv: consume + discard the K and V blocks ----
                S_KSKIP: if (k_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv_r) begin ld_a <= 0; ld_b <= 0; state <= S_VSKIP; end
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end
                S_VSKIP: if (v_tvalid) begin
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv_r) state <= S_KVNEXT;
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                S_KVNEXT: if (last_kv) begin head_i <= 0; bi <= 0; state <= S_FRECF; end
                          else begin kv_i <= kv_i + 16'd1; state <= S_MASK; end

                // ---- finalize per head: O[h] = acc[h] * recip(l[h]), 8-wide packed ----
                S_FRECF: begin bi <= 0; state <= S_FRECW; end
                S_FRECW: if (rec_v) begin inv_l_q <= rec_out; state <= S_FEMITF; end
                S_FEMITF: state <= S_FEMITW;              // emit_fire pulses this cycle
                S_FEMITW: if (ax_v) begin emit_buf <= ax_out; state <= S_FEMITE; end
                S_FEMITE: if (o_tready) begin
                    if (bi + 16'd1 == vbeats) begin
                        if (last_head) state <= S_TNEXT;
                        else begin head_i <= head_i + 16'd1; state <= S_FRECF; end
                    end else begin bi <= bi + 16'd1; state <= S_FEMITF; end
                end

                S_TNEXT: if (last_tok) state <= S_DONE;
                         else begin tok_i <= tok_i + 16'd1; ld_a <= 0; ld_b <= 0; state <= S_QLOAD; end

                S_DONE: begin busy <= 1'b0; done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused = &{1'b0, soft_grew, dot_v, smul_v, sadd_v, tree_v,
        q_tlast, k_tlast, v_tlast, mask_tlast, q_tkeep, k_tkeep, v_tkeep, mask_tkeep};

`ifdef FORMAL
`include "flash_kernel_properties.vh"
`endif
endmodule
