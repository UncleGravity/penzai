// flash_kernel - query-blocked flash attention, KV-major (v4).
//
// Consumes the KV cache in its NATIVE layout -- kv-position-major, n_head_kv heads,
// NOT GQA-replicated. K/V therefore DMA directly once per query tile; the driver stages
// Q query-major and transposes mask values to KV-major/query-inner order. GQA is done
// here: one kv-head's K/V fans out to its `head_ratio` query heads. A 64-slot adaptive
// store carries four query rows for up to 16 heads or two rows for up to 32 heads; each
// slot has independent online-softmax state (m,l,acc). The dot is pipelined (stream
// head_dim/8 beats through fp_dot into a 16-deep buffer, sum with reduce -- no
// per-beat recurrence). O is emitted query-major, 8-wide.
//
//   load Q tile (query,head,beat); init every (query,head) m=-inf,l=0,acc=0
//   for kv:
//     read one mask per query; load K[kv] once
//     for each finite (query,head): score=Q·K*scale+mask; update m,l,p,corr
//     load V[kv] once; update each finite accumulator with corr and p*V
//     wait for the final tagged accumulator writeback before accepting the next mask
//   finalize query-major: O=query/head acc*recip(l); l=0 emits exact zero
//
// Heads are streamed through the dot/score/softmax and AXPY pipelines while one KV
// position is active.  A hard barrier after the final tagged AXPY writeback preserves
// the online-softmax recurrence before the next KV position starts. Simulation checks
// the arithmetic and schedule bit-for-bit.

`default_nettype none

module flash_kernel #(
    parameter integer HEAD_DIM_MAX = 128,
    parameter integer MAX_HEADS    = 32,
    parameter integer MAX_HEAD_KV  = 8,
    parameter integer MAX_TOKENS   = 4,
    parameter integer MAX_SLOTS    = 64,
    parameter integer LANES        = 8,
    // The token engine schedules at most eight query heads at a time.  Eight
    // tokens x eight heads fills the existing 64 slots without growing the
    // numeric datapath or rereading any K/V head. When disabled, slot mapping
    // derives from the runtime token and head dimensions.
    parameter integer TILE8_HEAD8_LAYOUT = 0,
    // Four-lane resident arenas naturally produce all token lanes for one
    // {head,beat}. The token engine uses head/beat/token Q load order to avoid
    // a transpose buffer or repeated arena reads. When disabled, the input
    // stream order is token/head/beat.
    parameter integer TILE8_LANE4_Q_ORDER = 0
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [15:0] head_dim_q,
    input  wire [15:0] head_dim_v,
    input  wire [15:0] n_heads,
    input  wire [15:0] n_head_kv,
    input  wire [15:0] head_ratio,   // n_heads / n_head_kv (device-computed; no divider here)
    // 17 bits are required to represent the completed 65,536-position
    // context used by the 8B model_spec. Individual position indices remain
    // 16-bit; the extent/count has the inclusive upper value 65,536.
    input  wire [16:0] n_kv,
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
    localparam integer TW    = $clog2(MAX_TOKENS);           // query-index width (2)
    localparam integer SW    = $clog2(MAX_SLOTS);            // adaptive (query,head) slot width (6)
    localparam integer AW    = $clog2(MAX_SLOTS*MAXB);       // acc/q_buf addr width (10)
    localparam integer KW    = $clog2(MAX_HEAD_KV * MAXB);   // k_buf/v_buf addr width (7)
    localparam integer NSLOTS = MAX_SLOTS;
    localparam [31:0]  NEG_INF = 32'hFF800000;
    localparam [15:0]  F16_NEG_INF = 16'hFC00;

    // Snapshot the accepted command.  The AXI-Lite wrapper normally holds these
    // registers stable while busy, but the kernel must not depend on that convention.
    reg [15:0] head_dim_q_r, head_dim_v_r, n_heads_r, n_head_kv_r;
    reg [15:0] head_ratio_r, n_tokens_r;
    reg [16:0] n_kv_r;
    reg [31:0] scale_r;
    reg        wide_heads_r;
    wire [15:0] qbeats = head_dim_q_r >> LSH;
    wire [15:0] vbeats = head_dim_v_r >> LSH;

    // ---- resident / pooled storage (indexed [head*MAXB + beat], MAXB a power of 2) ----
    // Lever 2: q/k/v/acc are forced to block RAM (synchronous read below) so they stop
    // burning LUTs as distributed RAM. ram_style=block makes the intent explicit.
    (* ram_style = "block" *) reg [255:0] q_buf [0:MAX_SLOTS*MAXB-1]; // resident query tile
    (* ram_style = "block" *) reg [127:0] k_buf [0:MAX_HEAD_KV*MAXB-1]; // K for the current kv, 8 f16/beat
    (* ram_style = "block" *) reg [127:0] v_buf [0:MAX_HEAD_KV*MAXB-1]; // V for the current kv, 8 f16/beat
    (* ram_style = "block" *) reg [255:0] acc   [0:MAX_SLOTS*MAXB-1]; // per-(query,head) accumulator
    // First-stage synchronous BRAM outputs plus a true fabric register boundary.
    // The second stage isolates the numeric engines from BRAM placement and routing.
    reg [255:0] q_rd, q_rd_q;
    reg [127:0] k_rd, k_rd_q;
    reg [127:0] v_rd, v_rd_q;
    reg [255:0] acc_rd, acc_rd_q;
    reg [31:0] mpool [0:NSLOTS-1]; // running max per (query,head)
    reg [31:0] lpool [0:NSLOTS-1]; // running denom per (query,head)
    reg [31:0] ppool [0:NSLOTS-1]; // this-kv softmax weight
    reg [31:0] cpool [0:NSLOTS-1]; // this-kv rescale factor
    reg [31:0] score_pool [0:NSLOTS-1];
    reg [31:0]  dot_parts [0:MAXB-1];         // per-beat fp_dot partials for the active head
    reg [31:0]  mask_f32_pool [0:MAX_TOKENS-1];
    reg [MAX_TOKENS-1:0] mask_finite;

    reg [15:0] tok_i, head_i;
    reg [16:0] kv_i;
    reg [15:0] ld_tok, ld_a, ld_b;
    reg [15:0] dot_req_tok, dot_req_head, dot_req_kvh, dot_req_ratio, dot_req_beat;
    reg [15:0] dot_cap_tok, dot_cap_head, dot_cap_beat;
    reg [15:0] score_add_tok, score_add_head, score_tok, score_head;
    reg [15:0] soft_req_tok, soft_req_head, soft_wb_tok, soft_wb_head;
    reg [15:0] ax_req_tok, ax_req_head, ax_req_kvh, ax_req_ratio, ax_req_beat;
    reg [15:0] ax_wb_tok, ax_wb_head, ax_wb_beat;
    reg        dot_req_done, tree_pending, v_load_done, ax_req_done;
    reg [NSLOTS-1:0] score_ready, pc_ready;
    reg [15:0] bi;             // emit beat index
    reg [31:0] inv_l_q;
    reg [31:0] soft_score_q, soft_m_q, soft_l_q;
    reg [255:0] emit_buf;

    localparam [4:0]
        S_IDLE   = 5'd0,  S_QLOAD  = 5'd1,  S_TINIT  = 5'd2,  S_MASK   = 5'd3,
        S_KLOAD  = 5'd4,  S_PAIR   = 5'd5, S_VSKIP = 5'd17,
        S_KVNEXT = 5'd18, S_FRECF  = 5'd19, S_FRECW  = 5'd20,
        S_FEMITF = 5'd21, S_FEMITW = 5'd22, S_FEMITE = 5'd23, S_DONE = 5'd25;
    reg [4:0] state;

    function automatic [15:0] firstValidToken;
        input [MAX_TOKENS-1:0] valid;
        input [15:0] limit;
        integer t;
        reg found;
        begin
            firstValidToken = limit;
            found = 1'b0;
            for (t = 0; t < MAX_TOKENS; t = t + 1)
                if (!found && t < limit && valid[t]) begin
                    firstValidToken = t[15:0];
                    found = 1'b1;
                end
        end
    endfunction

    function automatic [15:0] nextValidToken;
        input [15:0] current;
        input [MAX_TOKENS-1:0] valid;
        input [15:0] limit;
        integer t;
        reg found;
        begin
            nextValidToken = limit;
            found = 1'b0;
            for (t = 0; t < MAX_TOKENS; t = t + 1)
                if (!found && t > current && t < limit && valid[t]) begin
                    nextValidToken = t[15:0];
                    found = 1'b1;
                end
        end
    endfunction

    // At most 64 query/head states are resident.  Small-head models use all four
    // logical query rows; models above 16 heads use two rows.  Both layouts are
    // concatenations, so pool and scratch addressing does not infer a multiplier.
    function automatic [SW-1:0] slotIndex;
        input [15:0] token;
        input [15:0] head;
        begin
            if (TILE8_HEAD8_LAYOUT != 0)
                slotIndex = {token[2:0], head[2:0]};
            else if (wide_heads_r)
                slotIndex = {token[0], head[4:0]};
            else
                slotIndex = {token[1:0], head[3:0]};
        end
    endfunction

    wire [15:0] first_valid_tok = firstValidToken(mask_finite, n_tokens_r);
    wire [15:0] dot_req_next_tok = nextValidToken(dot_req_tok, mask_finite, n_tokens_r);
    wire [15:0] dot_cap_next_tok = nextValidToken(dot_cap_tok, mask_finite, n_tokens_r);
    wire [15:0] score_add_next_tok = nextValidToken(score_add_tok, mask_finite, n_tokens_r);
    wire [15:0] score_next_tok = nextValidToken(score_tok, mask_finite, n_tokens_r);
    wire [15:0] soft_req_next_tok = nextValidToken(soft_req_tok, mask_finite, n_tokens_r);
    wire [15:0] soft_wb_next_tok = nextValidToken(soft_wb_tok, mask_finite, n_tokens_r);
    wire [15:0] ax_req_next_tok = nextValidToken(ax_req_tok, mask_finite, n_tokens_r);
    wire [15:0] ax_wb_next_tok = nextValidToken(ax_wb_tok, mask_finite, n_tokens_r);

    wire [SW-1:0] soft_req_slot = slotIndex(soft_req_tok, soft_req_head);
    wire [SW-1:0] soft_wb_slot  = slotIndex(soft_wb_tok, soft_wb_head);
    wire [SW-1:0] score_slot    = slotIndex(score_tok, score_head);
    wire [SW-1:0] ax_req_slot   = slotIndex(ax_req_tok, ax_req_head);
    wire [SW-1:0] ax_wb_slot    = slotIndex(ax_wb_tok, ax_wb_head);
    wire [SW-1:0] emit_slot     = slotIndex(tok_i, head_i);
    wire [SW-1:0] init_slot     = slotIndex(ld_tok, ld_a);
    wire [SW-1:0] dot_req_slot  = slotIndex(dot_req_tok, dot_req_head);
    wire [SW-1:0] load_slot     = slotIndex(ld_tok, ld_a);

    // Slot/beat fields concatenate because both capacities are powers of two.
    wire [AW-1:0] q_dot_addr = {dot_req_slot, dot_req_beat[LMAXB-1:0]};
    wire [KW-1:0] k_dot_addr = {dot_req_kvh[KW-LMAXB-1:0],  dot_req_beat[LMAXB-1:0]};
    wire [AW-1:0] q_ld_addr  = {load_slot, ld_b[LMAXB-1:0]};
    wire [KW-1:0] kv_ld_addr = {ld_a[KW-LMAXB-1:0],   ld_b[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_rd  = {ax_req_slot, ax_req_beat[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_wr  = {ax_wb_slot, ax_wb_beat[LMAXB-1:0]};
    wire [AW-1:0] acc_em_rd  = {emit_slot, bi[LMAXB-1:0]};
    wire [KW-1:0] v_ax_addr  = {ax_req_kvh[KW-LMAXB-1:0], ax_req_beat[LMAXB-1:0]};

    // ---- fire pulses + last-beat flags ----
    wire dot_fire  = (state == S_PAIR) && !dot_req_done;
    wire tree_fire = tree_pending;
    wire smul_fire = tree_v;
    wire sadd_fire = smul_v;
    reg  soft_fire;
    wire rec_fire  = (state == S_FRECF) && (lpool[emit_slot] != 32'd0);
    wire axpy_fire = (state == S_PAIR) && v_load_done && !ax_req_done &&
                     pc_ready[ax_req_slot];
    wire emit_fire = (state == S_FEMITF);

    wire last_head = (head_i + 16'd1 == n_heads_r);
    wire last_kv   = (kv_i + 17'd1 == n_kv_r);
    wire last_tok  = (tok_i + 16'd1 == n_tokens_r);

    // ---- fp_dot: one beat's Q·K (feed-forward, latency 16). q_buf/k_buf are
    // synchronous-read BRAMs followed by a fabric register, so issue is delayed two
    // cycles to match. The ordered capture cursor makes this latency transparent. ----
    reg dot_fire_q, dot_fire_qq;
    always @(posedge clk) begin
        if (!rst_n) begin dot_fire_q <= 1'b0; dot_fire_qq <= 1'b0; end
        else begin dot_fire_q <= dot_fire; dot_fire_qq <= dot_fire_q; end
    end
    wire        dot_v;  wire [31:0] dot_beat;
    fp_dot u_dot (.clk(clk), .rst_n(rst_n), .valid_in(dot_fire_qq),
        .q(q_rd_q), .k(k_rd_q), .valid_out(dot_v), .sum(dot_beat));

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
        .a(smul_out), .b(mask_f32_pool[score_add_tok[TW-1:0]]), .valid_out(sadd_v), .out(sadd_out));

    // ---- online softmax step (per head) ----
    wire soft_v; wire [31:0] soft_m, soft_l, soft_p, soft_corr; wire soft_grew;
    flash_softmax u_soft (.clk(clk), .rst_n(rst_n), .valid_in(soft_fire),
        .m_in(soft_m_q), .l_in(soft_l_q), .score(soft_score_q), .valid_out(soft_v),
        .m_out(soft_m), .l_out(soft_l), .p(soft_p), .corr(soft_corr), .grew(soft_grew));

    // ---- reciprocal of the denominator (per head, at finalize) ----
    wire rec_v; wire [31:0] rec_out;
    recip u_recip (.clk(clk), .rst_n(rst_n), .valid_in(rec_fire),
        .l(lpool[emit_slot]), .valid_out(rec_v), .y(rec_out));

    // ---- shared 8-wide axpy: acc*corr + p·V (accumulate) OR acc*inv_l (emit) ----
    // acc and V use the same BRAM-output plus fabric-register boundary as Q/K. Valid,
    // mode, and scalar operands take two stages to remain cycle-aligned with the data.
    wire [31:0]  ax_s1 = emit_fire ? inv_l_q : cpool[ax_req_slot];
    wire [31:0]  ax_p  = emit_fire ? 32'd0   : ppool[ax_req_slot];
    wire ax_fire = axpy_fire | emit_fire;
    reg        ax_fire_q, ax_fire_qq, ax_axpy_q, ax_axpy_qq;
    reg [31:0] ax_s1_q, ax_s1_qq, ax_p_q, ax_p_qq;
    always @(posedge clk) begin
        if (!rst_n) begin
            ax_fire_q <= 1'b0; ax_fire_qq <= 1'b0;
            ax_axpy_q <= 1'b0; ax_axpy_qq <= 1'b0;
        end else begin
            ax_fire_q <= ax_fire; ax_fire_qq <= ax_fire_q;
            ax_axpy_q <= axpy_fire; ax_axpy_qq <= ax_axpy_q;
        end
        ax_s1_q <= ax_s1;
        ax_s1_qq <= ax_s1_q;
        ax_p_q  <= ax_p;
        ax_p_qq <= ax_p_q;
    end
    wire [127:0] ax_v_eff = ax_axpy_qq ? v_rd_q : 128'd0;
    wire ax_v;  wire [255:0] ax_out;
    fp_axpy8 u_axpy (.clk(clk), .rst_n(rst_n), .valid_in(ax_fire_qq),
        .acc(acc_rd_q), .v(ax_v_eff), .s1(ax_s1_qq), .p(ax_p_qq), .valid_out(ax_v), .out(ax_out));

    // ---- block-RAM storage: each array gets one write port + one synchronous read
    // port — the template Vivado maps to BRAM instead of LUTRAM. ----
    // q_buf: written while loading Q; read by the head-streamed dot sequencer.
    always @(posedge clk) begin
        if ((state == S_QLOAD) && q_tvalid) q_buf[q_ld_addr] <= q_tdata;
        q_rd <= q_buf[q_dot_addr];
        q_rd_q <= q_rd;
    end
    // k_buf: written while loading K (S_KLOAD); read during the dot pass.
    always @(posedge clk) begin
        if ((state == S_KLOAD) && k_tvalid) k_buf[kv_ld_addr] <= k_tdata;
        k_rd <= k_buf[k_dot_addr];
        k_rd_q <= k_rd;
    end
    // v_buf: filled beside the score pipeline, then read by the AXPY sequencer.
    always @(posedge clk) begin
        if ((state == S_PAIR) && v_tvalid && v_tready) v_buf[kv_ld_addr] <= v_tdata;
        v_rd <= v_buf[v_ax_addr];
        v_rd_q <= v_rd;
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
        acc_rd_q <= acc_rd;
    end

    wire [31:0] mask_f32_w;
    cvt_f16_f32 u_maskw (.in(mask_tdata), .out(mask_f32_w));

    // ---- outputs ----
    assign o_tdata  = emit_buf;
    assign o_tvalid = (state == S_FEMITE);
    assign o_tkeep  = 32'hFFFFFFFF;
    assign o_tlast  = (state == S_FEMITE) && (bi + 16'd1 == vbeats) && last_head && last_tok;
    assign q_tready    = (state == S_QLOAD);
    assign k_tready    = (state == S_KLOAD);
    assign v_tready    = ((state == S_PAIR) && !v_load_done) || (state == S_VSKIP);
    assign mask_tready = (state == S_MASK);

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            dot_req_done <= 1'b0; tree_pending <= 1'b0;
            v_load_done <= 1'b0; ax_req_done <= 1'b0;
            soft_fire <= 1'b0; mask_finite <= 0;
            score_ready <= 0; pc_ready <= 0;
        end else begin
            done <= 1'b0;
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
                    wide_heads_r <= (n_heads > 16'd16);
                    busy <= 1'b1; tok_i <= 0; ld_tok <= 0; ld_a <= 0; ld_b <= 0;
                    state <= S_QLOAD;
                end

                // One tile of Q is resident for the entire KV-outer walk.
                S_QLOAD: if (q_tvalid) begin
                    if (TILE8_LANE4_Q_ORDER != 0) begin
                        // Token-inner order: head, beat, token. This matches one
                        // Four-lane arena gather without changing q_buf addressing.
                        if (ld_tok + 16'd1 == n_tokens_r) begin
                            ld_tok <= 0;
                            if (ld_b + 16'd1 == qbeats) begin
                                ld_b <= 0;
                                if (ld_a + 16'd1 == n_heads_r) begin
                                    ld_a <= 0; state <= S_TINIT;
                                end else ld_a <= ld_a + 16'd1;
                            end else ld_b <= ld_b + 16'd1;
                        end else ld_tok <= ld_tok + 16'd1;
                    end else begin
                        // Token/head/beat input order.
                        if (ld_b + 16'd1 == qbeats) begin
                            ld_b <= 0;
                            if (ld_a + 16'd1 == n_heads_r) begin
                                ld_a <= 0;
                                if (ld_tok + 16'd1 == n_tokens_r) begin
                                    ld_tok <= 0; state <= S_TINIT;
                                end else ld_tok <= ld_tok + 16'd1;
                            end else ld_a <= ld_a + 16'd1;
                        end else ld_b <= ld_b + 16'd1;
                    end
                end

                // Initialize every (query,head) state and accumulator beat.
                S_TINIT: begin
                    if (ld_b == 0) begin
                        mpool[init_slot] <= NEG_INF;
                        lpool[init_slot] <= 32'd0;
                    end
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_heads_r) begin
                            ld_a <= 0;
                            if (ld_tok + 16'd1 == n_tokens_r) begin
                                ld_tok <= 0; kv_i <= 0; mask_finite <= 0; state <= S_MASK;
                            end else ld_tok <= ld_tok + 16'd1;
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // Mask stream order is KV outer, query inner. Keep every finite bias;
                // exact f16 -inf suppresses only that query's work for this KV.
                S_MASK: if (mask_tvalid) begin
                    mask_f32_pool[ld_tok[TW-1:0]] <= mask_f32_w;
                    mask_finite[ld_tok[TW-1:0]] <= (mask_tdata != F16_NEG_INF);
                    if (ld_tok + 16'd1 == n_tokens_r) begin
                        ld_tok <= 0; ld_a <= 0; ld_b <= 0; state <= S_KLOAD;
                    end else ld_tok <= ld_tok + 16'd1;
                end

                // K is consumed once per KV regardless of query count or mask density.
                S_KLOAD: if (k_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv_r) begin
                            ld_a <= 0; ld_b <= 0;
                            if (first_valid_tok == n_tokens_r) begin
                                // Still consume V once so the four DMA streams retain a
                                // shape-only contract even when all queries are masked.
                                state <= S_VSKIP;
                            end else begin
                                dot_req_tok <= first_valid_tok; dot_req_head <= 0;
                                dot_req_kvh <= 0; dot_req_ratio <= 0; dot_req_beat <= 0;
                                dot_cap_tok <= first_valid_tok; dot_cap_head <= 0; dot_cap_beat <= 0;
                                score_add_tok <= first_valid_tok; score_add_head <= 0;
                                score_tok <= first_valid_tok; score_head <= 0;
                                soft_req_tok <= first_valid_tok; soft_req_head <= 0;
                                soft_wb_tok <= first_valid_tok; soft_wb_head <= 0;
                                ax_req_tok <= first_valid_tok; ax_req_head <= 0;
                                ax_req_kvh <= 0; ax_req_ratio <= 0; ax_req_beat <= 0;
                                ax_wb_tok <= first_valid_tok; ax_wb_head <= 0; ax_wb_beat <= 0;
                                dot_req_done <= 1'b0; tree_pending <= 1'b0;
                                score_ready <= 0; pc_ready <= 0;
                                v_load_done <= 1'b0; ax_req_done <= 1'b0;
                                state <= S_PAIR;
                            end
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                S_PAIR: begin
                    // Ordered request tags: query-major, then GQA-mapped head, then beat.
                    if (dot_fire) begin
                        if (dot_req_beat + 16'd1 == qbeats) begin
                            dot_req_beat <= 0;
                            if (dot_req_head + 16'd1 == n_heads_r) begin
                                if (dot_req_next_tok == n_tokens_r) begin
                                    dot_req_done <= 1'b1;
                                end else begin
                                    dot_req_tok <= dot_req_next_tok; dot_req_head <= 0;
                                    dot_req_kvh <= 0; dot_req_ratio <= 0;
                                end
                            end else begin
                                dot_req_head <= dot_req_head + 16'd1;
                                if (dot_req_ratio + 16'd1 == head_ratio_r) begin
                                    dot_req_ratio <= 0; dot_req_kvh <= dot_req_kvh + 16'd1;
                                end else dot_req_ratio <= dot_req_ratio + 16'd1;
                            end
                        end else dot_req_beat <= dot_req_beat + 16'd1;
                    end

                    if (dot_v) begin
                        dot_parts[dot_cap_beat[LMAXB-1:0]] <= dot_beat;
                        if (dot_cap_beat + 16'd1 == qbeats) begin
                            dot_cap_beat <= 0; tree_pending <= 1'b1;
                            if (dot_cap_head + 16'd1 == n_heads_r) begin
                                if (dot_cap_next_tok != n_tokens_r) begin
                                    dot_cap_tok <= dot_cap_next_tok; dot_cap_head <= 0;
                                end
                            end else dot_cap_head <= dot_cap_head + 16'd1;
                        end else dot_cap_beat <= dot_cap_beat + 16'd1;
                    end

                    // Tree, score multiply, and add are ordered fixed-latency pipes;
                    // advance their tags on the corresponding completion pulses.
                    if (smul_v) begin
                        if (score_add_head + 16'd1 == n_heads_r) begin
                            if (score_add_next_tok != n_tokens_r) begin
                                score_add_tok <= score_add_next_tok; score_add_head <= 0;
                            end
                        end else score_add_head <= score_add_head + 16'd1;
                    end
                    if (sadd_v) begin
                        score_pool[score_slot] <= sadd_out;
                        score_ready[score_slot] <= 1'b1;
                        if (score_head + 16'd1 == n_heads_r) begin
                            if (score_next_tok != n_tokens_r) begin
                                score_tok <= score_next_tok; score_head <= 0;
                            end
                        end else score_head <= score_head + 16'd1;
                    end

                    // The softmax is a stateless II=1 pipe. Slot tags stay ordered even
                    // when a masked query creates a gap in the logical tile.
                    if (score_ready[soft_req_slot]) begin
                        soft_score_q <= score_pool[soft_req_slot];
                        soft_m_q <= mpool[soft_req_slot];
                        soft_l_q <= lpool[soft_req_slot];
                        score_ready[soft_req_slot] <= 1'b0;
                        soft_fire <= 1'b1;
                        if (soft_req_head + 16'd1 == n_heads_r) begin
                            if (soft_req_next_tok != n_tokens_r) begin
                                soft_req_tok <= soft_req_next_tok; soft_req_head <= 0;
                            end
                        end else soft_req_head <= soft_req_head + 16'd1;
                    end

                    if (soft_v) begin
                        mpool[soft_wb_slot] <= soft_m; lpool[soft_wb_slot] <= soft_l;
                        ppool[soft_wb_slot] <= soft_p; cpool[soft_wb_slot] <= soft_corr;
                        pc_ready[soft_wb_slot] <= 1'b1;
                        if (soft_wb_head + 16'd1 == n_heads_r) begin
                            if (soft_wb_next_tok != n_tokens_r) begin
                                soft_wb_tok <= soft_wb_next_tok; soft_wb_head <= 0;
                            end
                        end else soft_wb_head <= soft_wb_head + 16'd1;
                    end

                    if (v_tvalid && v_tready) begin
                        if (ld_b + 16'd1 == vbeats) begin
                            ld_b <= 0;
                            if (ld_a + 16'd1 == n_head_kv_r) begin
                                ld_a <= 0; v_load_done <= 1'b1;
                            end else ld_a <= ld_a + 16'd1;
                        end else ld_b <= ld_b + 16'd1;
                    end

                    if (axpy_fire) begin
                        if (ax_req_beat + 16'd1 == vbeats) begin
                            ax_req_beat <= 0;
                            if (ax_req_head + 16'd1 == n_heads_r) begin
                                if (ax_req_next_tok == n_tokens_r) begin
                                    ax_req_done <= 1'b1;
                                end else begin
                                    ax_req_tok <= ax_req_next_tok; ax_req_head <= 0;
                                    ax_req_kvh <= 0; ax_req_ratio <= 0;
                                end
                            end else begin
                                ax_req_head <= ax_req_head + 16'd1;
                                if (ax_req_ratio + 16'd1 == head_ratio_r) begin
                                    ax_req_ratio <= 0; ax_req_kvh <= ax_req_kvh + 16'd1;
                                end else ax_req_ratio <= ax_req_ratio + 16'd1;
                            end
                        end else ax_req_beat <= ax_req_beat + 16'd1;
                    end

                    if (ax_v) begin
                        if (ax_wb_beat + 16'd1 == vbeats) begin
                            ax_wb_beat <= 0;
                            if (ax_wb_head + 16'd1 == n_heads_r) begin
                                if (ax_wb_next_tok == n_tokens_r) begin
                                    // Final registered accumulator write occurs on this
                                    // edge; only the next state may request another mask.
                                    state <= S_KVNEXT;
                                end else begin
                                    ax_wb_tok <= ax_wb_next_tok; ax_wb_head <= 0;
                                end
                            end else ax_wb_head <= ax_wb_head + 16'd1;
                        end else ax_wb_beat <= ax_wb_beat + 16'd1;
                    end
                end

                S_VSKIP: if (v_tvalid) begin
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv_r) begin
                            ld_a <= 0; state <= S_KVNEXT;
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                S_KVNEXT: if (last_kv) begin
                    tok_i <= 0; head_i <= 0; bi <= 0; state <= S_FRECF;
                end else begin
                    kv_i <= kv_i + 17'd1; ld_tok <= 0; mask_finite <= 0; state <= S_MASK;
                end

                // l==0 is the all-masked query contract: emit exact zero and never
                // wait for or use recip(0). Other rows retain normal finalization.
                S_FRECF: begin
                    bi <= 0;
                    if (lpool[emit_slot] == 32'd0) begin
                        inv_l_q <= 32'd0; state <= S_FEMITF;
                    end else state <= S_FRECW;
                end
                S_FRECW: if (rec_v) begin inv_l_q <= rec_out; state <= S_FEMITF; end
                S_FEMITF: state <= S_FEMITW;
                S_FEMITW: if (ax_v) begin emit_buf <= ax_out; state <= S_FEMITE; end
                S_FEMITE: if (o_tready) begin
                    if (bi + 16'd1 == vbeats) begin
                        if (last_head) begin
                            head_i <= 0;
                            if (last_tok) state <= S_DONE;
                            else begin tok_i <= tok_i + 16'd1; state <= S_FRECF; end
                        end else begin head_i <= head_i + 16'd1; state <= S_FRECF; end
                    end else begin bi <= bi + 16'd1; state <= S_FEMITF; end
                end

                S_DONE: begin busy <= 1'b0; done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused = &{1'b0, soft_grew, dot_v, smul_v, sadd_v, tree_v,
        q_tlast, k_tlast, v_tlast, mask_tlast, q_tkeep, k_tkeep, v_tkeep, mask_tkeep};

`ifdef FORMAL
`include "properties.vh"
`endif
endmodule
