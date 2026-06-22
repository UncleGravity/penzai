// flash_kernel - flash attention, kv-major (v2).
//
// Consumes the KV cache in its NATIVE layout — kv-position-major, n_head_kv heads,
// NOT GQA-replicated — so the device DMAs Q/K/V/mask straight from the resident
// tensors with no host gather (the v1 gather was ~89% of a decode call). GQA is done
// here: one kv-head's K/V fans out to its `head_ratio` query heads. All n_heads carry
// an independent online-softmax state (m,l,acc) at once, so streaming kv-major updates
// every head per kv. The dot is pipelined (stream head_dim/8 beats through fp_dot into
// a 16-deep buffer, sum with fp_addtree — no per-beat recurrence), replacing v1's
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
// Sequential-handshake FSM (each fp op completes before the next; indices stable
// across an op) like v1, except the dot and the per-beat axpy/emit, which pipeline.
// Cosim-checked vs flash_ref.attendHead.

`default_nettype none

module flash_kernel #(
    parameter integer HEAD_DIM_MAX = 128,
    parameter integer MAX_HEADS    = 16,
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
    localparam integer MAXB  = HEAD_DIM_MAX / LANES;         // 16 max Q/K/V beats per head
    localparam integer LSH   = $clog2(LANES);                // 3
    localparam integer LMAXB = $clog2(MAXB);                 // 4 (beat occupies the low bits)
    localparam integer AW    = $clog2(MAX_HEADS * MAXB);     // acc/q_buf addr width (8)
    localparam integer KW    = $clog2(MAX_HEAD_KV * MAXB);   // k_buf/v_buf addr width (7)
    localparam [31:0]  NEG_INF = 32'hFF800000;
    localparam [15:0]  F16_NEG_INF = 16'hFC00;

    wire [15:0] qbeats = head_dim_q >> LSH;   // head dims are multiples of LANES (v1 invariant)
    wire [15:0] vbeats = head_dim_v >> LSH;

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
    reg [31:0]  dot_parts [0:MAXB-1];         // per-beat fp_dot partials for the active head

    reg [15:0] tok_i, kv_i, head_i, kvh_i, ratio_cnt;
    reg [15:0] ld_a, ld_b;     // two-level load/clear counters (outer head/kvh, inner beat)
    reg [15:0] di, ci;         // dot issue / capture indices
    reg [15:0] ax_iss, ax_wb;  // axpy issue / writeback indices
    reg [15:0] bi;             // emit beat index
    reg [31:0] mask_f32_q, dot_q, score_q, inv_l_q;
    reg [255:0] emit_buf;

    localparam [4:0]
        S_IDLE   = 5'd0,  S_QLOAD  = 5'd1,  S_TINIT  = 5'd2,  S_MASK   = 5'd3,
        S_KLOAD  = 5'd4,  S_DOTISS = 5'd5,  S_DOTTF  = 5'd6,  S_DOTTW  = 5'd7,
        S_SCRF   = 5'd8,  S_SCRW   = 5'd9,  S_ADDF   = 5'd10, S_ADDW   = 5'd11,
        S_SOFTF  = 5'd12, S_SOFTW  = 5'd13, S_VLOAD  = 5'd14, S_AXPY   = 5'd15,
        S_KSKIP  = 5'd16, S_VSKIP  = 5'd17, S_KVNEXT = 5'd18, S_FRECF  = 5'd19,
        S_FRECW  = 5'd20, S_FEMITF = 5'd21, S_FEMITW = 5'd22, S_FEMITE = 5'd23,
        S_TNEXT  = 5'd24, S_DONE   = 5'd25;
    reg [4:0] state;

    // ---- combinational addressing: address = head·MAXB + beat. MAXB is a power of
    // two and beat < MAXB, so this is a concat (head/kv-head upper, beat low LMAXB
    // bits) — no multiplier, exact width. ----
    wire [AW-1:0] q_dot_addr = {head_i[AW-LMAXB-1:0], di[LMAXB-1:0]};
    wire [KW-1:0] k_dot_addr = {kvh_i[KW-LMAXB-1:0],  di[LMAXB-1:0]};
    wire [AW-1:0] q_ld_addr  = {ld_a[AW-LMAXB-1:0],   ld_b[LMAXB-1:0]};
    wire [KW-1:0] kv_ld_addr = {ld_a[KW-LMAXB-1:0],   ld_b[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_rd  = {head_i[AW-LMAXB-1:0], ax_iss[LMAXB-1:0]};
    wire [AW-1:0] acc_ax_wr  = {head_i[AW-LMAXB-1:0], ax_wb[LMAXB-1:0]};
    wire [AW-1:0] acc_em_rd  = {head_i[AW-LMAXB-1:0], bi[LMAXB-1:0]};
    wire [KW-1:0] v_ax_addr  = {kvh_i[KW-LMAXB-1:0],  ax_iss[LMAXB-1:0]};

    // ---- fire pulses + last-beat flags ----
    wire dot_fire  = (state == S_DOTISS) && (di < qbeats);
    wire tree_fire = (state == S_DOTTF);
    wire smul_fire = (state == S_SCRF);
    wire sadd_fire = (state == S_ADDF);
    wire soft_fire = (state == S_SOFTF);
    wire rec_fire  = (state == S_FRECF);
    wire axpy_fire = (state == S_AXPY) && (ax_iss < vbeats);
    wire emit_fire = (state == S_FEMITF);

    wire last_head = (head_i + 16'd1 == n_heads);
    wire last_kv   = (kv_i  + 16'd1 == n_kv);
    wire last_tok  = (tok_i + 16'd1 == n_tokens);

    // ---- fp_dot: one beat's Q·K (feed-forward, latency 14). q_buf/k_buf are now
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

    // ---- fp_addtree: sum the per-beat partials (zero-padded, latency 16) ----
    wire [511:0] tree_in;
    genvar g;
    generate
        for (g = 0; g < MAXB; g = g + 1) begin : g_tree_in
            assign tree_in[g*32 +: 32] = (g < qbeats) ? dot_parts[g] : 32'd0;
        end
    endgenerate
    wire tree_v;  wire [31:0] tree_sum;
    fp_addtree u_tree (.clk(clk), .rst_n(rst_n), .valid_in(tree_fire),
        .in(tree_in), .valid_out(tree_v), .sum(tree_sum));

    // ---- score = dot*scale + mask ----
    wire smul_v; wire [31:0] smul_out;
    fp32_mul_pipe u_smul (.clk(clk), .rst_n(rst_n), .valid_in(smul_fire),
        .a(dot_q), .b(scale), .valid_out(smul_v), .out(smul_out));
    wire sadd_v; wire [31:0] sadd_out;
    fp32_add_pipe u_sadd (.clk(clk), .rst_n(rst_n), .valid_in(sadd_fire),
        .a(score_q), .b(mask_f32_q), .valid_out(sadd_v), .out(sadd_out));

    // ---- online softmax step (per head) ----
    wire soft_v; wire [31:0] soft_m, soft_l, soft_p, soft_corr; wire soft_grew;
    flash_softmax u_soft (.clk(clk), .rst_n(rst_n), .valid_in(soft_fire),
        .m_in(mpool[head_i[3:0]]), .l_in(lpool[head_i[3:0]]), .score(score_q), .valid_out(soft_v),
        .m_out(soft_m), .l_out(soft_l), .p(soft_p), .corr(soft_corr), .grew(soft_grew));

    // ---- reciprocal of the denominator (per head, at finalize) ----
    wire rec_v; wire [31:0] rec_out;
    fp_recip u_recip (.clk(clk), .rst_n(rst_n), .valid_in(rec_fire),
        .l(lpool[head_i[3:0]]), .valid_out(rec_v), .y(rec_out));

    // ---- shared 8-wide axpy: acc*corr + p·V (accumulate) OR acc*inv_l (emit) ----
    // acc and V are now synchronous-read BRAMs (acc_rd / v_rd arrive one cycle after the
    // address is presented), so the matching control strobes are delayed one cycle to
    // realign with the data. ax_axpy_q selects v_rd (accumulate) vs zero (emit).
    wire [31:0]  ax_s1 = emit_fire ? inv_l_q : cpool[head_i[3:0]];
    wire [31:0]  ax_p  = emit_fire ? 32'd0   : ppool[head_i[3:0]];
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
    // q_buf: written while loading Q (S_QLOAD); read during the dot pass (S_DOTISS).
    always @(posedge clk) begin
        if ((state == S_QLOAD) && q_tvalid) q_buf[q_ld_addr] <= q_tdata;
        q_rd <= q_buf[q_dot_addr];
    end
    // k_buf: written while loading K (S_KLOAD); read during the dot pass.
    always @(posedge clk) begin
        if ((state == S_KLOAD) && k_tvalid) k_buf[kv_ld_addr] <= k_tdata;
        k_rd <= k_buf[k_dot_addr];
    end
    // v_buf: written while loading V (S_VLOAD); read during the axpy pass (S_AXPY).
    always @(posedge clk) begin
        if ((state == S_VLOAD) && v_tvalid) v_buf[kv_ld_addr] <= v_tdata;
        v_rd <= v_buf[v_ax_addr];
    end
    // acc: write port muxes the per-token zero-init (S_TINIT) and the axpy writeback
    // (S_AXPY, on ax_v); read port muxes the axpy and emit addresses. Issue leads
    // writeback by the axpy latency, so a beat's read and its write never share a cycle.
    wire          acc_we    = (state == S_TINIT) || ((state == S_AXPY) && ax_v);
    wire [AW-1:0] acc_waddr = (state == S_TINIT) ? q_ld_addr : acc_ax_wr;
    wire [255:0]  acc_wdata = (state == S_TINIT) ? 256'd0    : ax_out;
    wire [AW-1:0] acc_raddr = emit_fire ? acc_em_rd : acc_ax_rd;
    always @(posedge clk) begin
        if (acc_we) acc[acc_waddr] <= acc_wdata;
        acc_rd <= acc[acc_raddr];
    end

    wire [31:0] mask_f32_w;
    fp16_to_fp32 u_maskw (.in(mask_tdata), .out(mask_f32_w));

    // ---- outputs ----
    assign o_tdata  = emit_buf;
    assign o_tvalid = (state == S_FEMITE);
    assign o_tkeep  = 32'hFFFFFFFF;
    assign o_tlast  = (state == S_FEMITE) && (bi + 16'd1 == vbeats) && last_head && last_tok;
    assign q_tready    = (state == S_QLOAD);
    assign k_tready    = (state == S_KLOAD) || (state == S_KSKIP);
    assign v_tready    = (state == S_VLOAD) || (state == S_VSKIP);
    assign mask_tready = (state == S_MASK);

    // ---- helper: advance head_i + kvh_i (kvh increments every head_ratio heads) ----
    // applied inline at the dot-pass and axpy-pass head boundaries.

    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    busy <= 1'b1; tok_i <= 0; ld_a <= 0; ld_b <= 0; state <= S_QLOAD;
                end

                // ---- load Q for the current token: n_heads × qbeats beats ----
                S_QLOAD: if (q_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_heads) begin ld_a <= 0; ld_b <= 0; state <= S_TINIT; end
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- init the per-head pool for this token (acc=0, m=-inf, l=0) ----
                S_TINIT: begin
                    if (ld_b == 16'd0) begin mpool[ld_a[3:0]] <= NEG_INF; lpool[ld_a[3:0]] <= 32'd0; end
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_heads) begin kv_i <= 0; state <= S_MASK; end
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
                        if (ld_a + 16'd1 == n_head_kv) begin
                            head_i <= 0; kvh_i <= 0; ratio_cnt <= 0; di <= 0; ci <= 0; state <= S_DOTISS;
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- dot-pass: pipelined Q[h]·K[kvh] for the active head ----
                S_DOTISS: begin
                    if (di < qbeats) di <= di + 16'd1;
                    if (dot_v) begin
                        dot_parts[ci[3:0]] <= dot_beat;
                        if (ci + 16'd1 == qbeats) state <= S_DOTTF;
                        else ci <= ci + 16'd1;
                    end
                end
                S_DOTTF: state <= S_DOTTW;                 // tree_fire pulses this cycle
                S_DOTTW: if (tree_v) begin dot_q <= tree_sum; state <= S_SCRF; end

                S_SCRF: state <= S_SCRW;
                S_SCRW: if (smul_v) begin score_q <= smul_out; state <= S_ADDF; end
                S_ADDF: state <= S_ADDW;
                S_ADDW: if (sadd_v) begin score_q <= sadd_out; state <= S_SOFTF; end

                S_SOFTF: state <= S_SOFTW;
                S_SOFTW: if (soft_v) begin
                    mpool[head_i[3:0]] <= soft_m; lpool[head_i[3:0]] <= soft_l;
                    ppool[head_i[3:0]] <= soft_p; cpool[head_i[3:0]] <= soft_corr;
                    if (last_head) begin ld_a <= 0; ld_b <= 0; state <= S_VLOAD; end
                    else begin
                        head_i <= head_i + 16'd1;
                        if (ratio_cnt + 16'd1 == head_ratio) begin ratio_cnt <= 0; kvh_i <= kvh_i + 16'd1; end
                        else ratio_cnt <= ratio_cnt + 16'd1;
                        di <= 0; ci <= 0; state <= S_DOTISS;
                    end
                end

                // ---- load V[kv]: n_head_kv heads × vbeats beats ----
                S_VLOAD: if (v_tvalid) begin
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv) begin
                            head_i <= 0; kvh_i <= 0; ratio_cnt <= 0; ax_iss <= 0; ax_wb <= 0; state <= S_AXPY;
                        end else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end

                // ---- axpy-pass: acc[h] = acc[h]*corr[h] + p[h]·V[kvh], pipelined per beat ----
                S_AXPY: begin
                    if (ax_iss < vbeats) ax_iss <= ax_iss + 16'd1;
                    if (ax_v) begin
                        if (ax_wb + 16'd1 == vbeats) begin
                            if (last_head) state <= S_KVNEXT;
                            else begin
                                head_i <= head_i + 16'd1;
                                if (ratio_cnt + 16'd1 == head_ratio) begin ratio_cnt <= 0; kvh_i <= kvh_i + 16'd1; end
                                else ratio_cnt <= ratio_cnt + 16'd1;
                                ax_iss <= 0; ax_wb <= 0;
                            end
                        end else ax_wb <= ax_wb + 16'd1;
                    end
                end

                // ---- masked kv: consume + discard the K and V blocks ----
                S_KSKIP: if (k_tvalid) begin
                    if (ld_b + 16'd1 == qbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv) begin ld_a <= 0; ld_b <= 0; state <= S_VSKIP; end
                        else ld_a <= ld_a + 16'd1;
                    end else ld_b <= ld_b + 16'd1;
                end
                S_VSKIP: if (v_tvalid) begin
                    if (ld_b + 16'd1 == vbeats) begin
                        ld_b <= 0;
                        if (ld_a + 16'd1 == n_head_kv) state <= S_KVNEXT;
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
endmodule
