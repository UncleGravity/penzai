// flash_kernel - flash attention, correctness-first v1.
//
// Composes the verified leaves (fp_dot, flash_softmax, fp_recip) into the full
// online-softmax attention. For each (token, head): load Q (resident), stream K/V
// per kv, run the online softmax, accumulate p·V, normalize by 1/l, emit O.
//
//   for token, for head:
//     load q_buf <- Q stream (head_dim_q/8 beats of 8×f32)
//     m=-inf, l=0, acc[*]=0
//     for kv in 0..n_kv:
//       mask <- mask stream (1×f16);  if -inf: consume+discard K/V, continue
//       dot   = Σ over head_dim_q/8 beats fp_dot(q_buf, K beat)
//       score = dot*scale + mask
//       (m,l,p,corr) = flash_softmax(m,l,score)
//       v_buf <- V stream ; acc[d] = acc[d]*corr + p*f32(v_buf[d])
//     inv_l = recip(l) ; O[d] = acc[d]*inv_l -> O stream (1 f32/beat in lane 0)
//
// v1 is fully sequential + handshake-driven: combinational `fire` signals decoded
// from the FSM state, each fp op completing before the next starts, indices held
// stable across an op's FIRE→WAIT. Slow but robust and bit-checkable vs
// flash_ref.attendHead. GQA is in the feed order, not here. Perf (interleave, GQA
// reuse, pipelined walks) is the next pass.

`default_nettype none

module flash_kernel #(
    parameter integer HEAD_DIM_MAX = 128,
    parameter integer LANES        = 8
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [15:0] head_dim_q,
    input  wire [15:0] head_dim_v,
    input  wire [15:0] n_heads,
    input  wire [15:0] n_kv,
    input  wire [15:0] n_tokens,
    input  wire [31:0] scale,
    output reg         busy,
    output reg         done,

    input  wire [255:0] q_tdata,    input wire q_tvalid,    output wire q_tready,
    input  wire [127:0] k_tdata,    input wire k_tvalid,    output wire k_tready,
    input  wire [127:0] v_tdata,    input wire v_tvalid,    output wire v_tready,
    input  wire [15:0]  mask_tdata, input wire mask_tvalid, output wire mask_tready,
    output wire [255:0] o_tdata,    output wire o_tvalid,   input wire o_tready
);
    localparam integer MAXB = HEAD_DIM_MAX / LANES;
    localparam integer BW   = (MAXB <= 1) ? 1 : $clog2(MAXB);
    localparam integer EW   = $clog2(HEAD_DIM_MAX);
    localparam integer LSH  = $clog2(LANES);
    localparam [31:0]  NEG_INF = 32'hFF800000;
    localparam [15:0]  F16_NEG_INF = 16'hFC00;

    wire [15:0] qbeats = head_dim_q >> LSH;  // head_dim is a multiple of LANES (v1)
    wire [15:0] vbeats = head_dim_v >> LSH;

    reg [255:0] q_buf [0:MAXB-1];
    reg [31:0]  acc   [0:HEAD_DIM_MAX-1];
    reg [15:0]  v_buf [0:HEAD_DIM_MAX-1];

    reg [31:0] m_q, l_q, dot_q, score_q, corr_q, p_q, inv_l_q, mask_f32_q, dot_beat_q;
    reg [15:0] tok_i, head_i, kv_i, beat_i, elem_i;
    reg        walk_axpy;

    localparam [4:0]
        S_IDLE  = 5'd0,  S_QLOAD = 5'd1,  S_KVINIT= 5'd2,  S_CLR   = 5'd3,
        S_MASK  = 5'd4,  S_DOTF  = 5'd5,  S_DOTW  = 5'd6,  S_DACCF = 5'd7,
        S_DACCW = 5'd8,  S_SKIPK = 5'd9,  S_SCRF  = 5'd10, S_SCRW  = 5'd11,
        S_ADDF  = 5'd12, S_ADDW  = 5'd13, S_SOFTF = 5'd14, S_SOFTW = 5'd15,
        S_VLOAD = 5'd16, S_SKIPV = 5'd17, S_AXPYF = 5'd18, S_AXPYW = 5'd19,
        S_KVNXT = 5'd20, S_NORMF = 5'd21, S_NORMW = 5'd22, S_EMITF = 5'd23,
        S_EMITW = 5'd24, S_HEADN = 5'd25, S_DONE  = 5'd26;
    reg [4:0] state;

    wire [BW-1:0] beat = beat_i[BW-1:0];
    wire [EW-1:0] elem = elem_i[EW-1:0];
    wire last_kv   = (kv_i + 16'd1 == n_kv);
    wire last_head = (head_i + 16'd1 == n_heads);
    wire last_tok  = (tok_i + 16'd1 == n_tokens);
    wire last_qb   = (beat_i + 16'd1 == qbeats);
    wire last_vb   = (beat_i + 16'd1 == vbeats);
    wire last_elem = (elem_i + 16'd1 == head_dim_v);

    // ---- combinational fire signals (one-cycle pulses decoded from state) ----
    wire dot_fire  = (state == S_DOTF) && k_tvalid;
    wire dacc_fire = (state == S_DACCF);
    wire smul_fire = (state == S_SCRF);
    wire sadd_fire = (state == S_ADDF);
    wire soft_fire = (state == S_SOFTF);
    wire rec_fire  = (state == S_NORMF);
    wire walk_fire = (state == S_AXPYF) || (state == S_EMITF);

    // ---- fp blocks ----
    wire        dot_v;  wire [31:0] dot_beat;
    fp_dot u_dot (.clk(clk), .rst_n(rst_n), .valid_in(dot_fire),
        .q(q_buf[beat]), .k(k_tdata), .valid_out(dot_v), .sum(dot_beat));

    wire        dacc_v; wire [31:0] dacc_out;
    fp32_add_pipe u_dacc (.clk(clk), .rst_n(rst_n), .valid_in(dacc_fire),
        .a(dot_q), .b(dot_beat_q), .valid_out(dacc_v), .out(dacc_out));

    wire        smul_v; wire [31:0] smul_out;
    fp32_mul_pipe u_smul (.clk(clk), .rst_n(rst_n), .valid_in(smul_fire),
        .a(dot_q), .b(scale), .valid_out(smul_v), .out(smul_out));
    wire        sadd_v; wire [31:0] sadd_out;
    fp32_add_pipe u_sadd (.clk(clk), .rst_n(rst_n), .valid_in(sadd_fire),
        .a(score_q), .b(mask_f32_q), .valid_out(sadd_v), .out(sadd_out));

    wire soft_v; wire [31:0] soft_m, soft_l, soft_p, soft_corr; wire soft_grew;
    flash_softmax u_soft (.clk(clk), .rst_n(rst_n), .valid_in(soft_fire),
        .m_in(m_q), .l_in(l_q), .score(score_q), .valid_out(soft_v),
        .m_out(soft_m), .l_out(soft_l), .p(soft_p), .corr(soft_corr), .grew(soft_grew));

    wire rec_v; wire [31:0] rec_out;
    fp_recip u_recip (.clk(clk), .rst_n(rst_n), .valid_in(rec_fire),
        .l(l_q), .valid_out(rec_v), .y(rec_out));

    wire [31:0] mask_f32_w;
    fp16_to_fp32 u_maskw (.in(mask_tdata), .out(mask_f32_w));

    // ---- per-element walk: acc[d] = acc[d]*(corr|inv_l) + (axpy ? p*f32(v[d]) : 0) ----
    wire [31:0] vf32_w;
    fp16_to_fp32 u_vw (.in(v_buf[elem]), .out(vf32_w));
    wire m1v, m2v, wav;  wire [31:0] m1o, m2o, wao;
    fp32_mul_pipe u_w1 (.clk(clk), .rst_n(rst_n), .valid_in(walk_fire),
        .a(acc[elem]), .b(walk_axpy ? corr_q : inv_l_q), .valid_out(m1v), .out(m1o));
    fp32_mul_pipe u_w2 (.clk(clk), .rst_n(rst_n), .valid_in(walk_fire),
        .a(walk_axpy ? p_q : 32'd0), .b(walk_axpy ? vf32_w : 32'd0), .valid_out(m2v), .out(m2o));
    fp32_add_pipe u_w3 (.clk(clk), .rst_n(rst_n), .valid_in(m1v),
        .a(m1o), .b(m2o), .valid_out(wav), .out(wao));

    assign o_tdata  = {224'd0, wao};
    assign o_tvalid = (state == S_EMITW) && wav;

    assign q_tready    = (state == S_QLOAD);
    assign k_tready    = (state == S_DOTF) || (state == S_SKIPK);
    assign v_tready    = (state == S_VLOAD) || (state == S_SKIPV);
    assign mask_tready = (state == S_MASK);

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    busy <= 1'b1; tok_i <= 0; head_i <= 0; beat_i <= 0; state <= S_QLOAD;
                end

                S_QLOAD: if (q_tvalid) begin
                    q_buf[beat] <= q_tdata;
                    if (last_qb) begin beat_i <= 0; state <= S_KVINIT; end
                    else beat_i <= beat_i + 16'd1;
                end

                S_KVINIT: begin m_q <= NEG_INF; l_q <= 32'd0; kv_i <= 0; elem_i <= 0; state <= S_CLR; end
                S_CLR: begin
                    acc[elem] <= 32'd0;
                    if (last_elem) begin elem_i <= 0; state <= S_MASK; end
                    else elem_i <= elem_i + 16'd1;
                end

                S_MASK: if (mask_tvalid) begin
                    mask_f32_q <= mask_f32_w; dot_q <= 32'd0; beat_i <= 0;
                    state <= (mask_tdata == F16_NEG_INF) ? S_SKIPK : S_DOTF;
                end

                // dot: per beat, fp_dot then accumulate
                S_DOTF: if (k_tvalid) state <= S_DOTW;             // dot_fire pulses this cycle
                S_DOTW: if (dot_v) begin dot_beat_q <= dot_beat; state <= S_DACCF; end
                S_DACCF: state <= S_DACCW;
                S_DACCW: if (dacc_v) begin
                    dot_q <= dacc_out;
                    if (last_qb) state <= S_SCRF;
                    else begin beat_i <= beat_i + 16'd1; state <= S_DOTF; end
                end
                S_SKIPK: if (k_tvalid) begin
                    if (last_qb) begin beat_i <= 0; state <= S_SKIPV; end
                    else beat_i <= beat_i + 16'd1;
                end

                // score = dot*scale + mask
                S_SCRF: state <= S_SCRW;
                S_SCRW: if (smul_v) begin score_q <= smul_out; state <= S_ADDF; end
                S_ADDF: state <= S_ADDW;
                S_ADDW: if (sadd_v) begin score_q <= sadd_out; state <= S_SOFTF; end

                // online softmax
                S_SOFTF: state <= S_SOFTW;
                S_SOFTW: if (soft_v) begin
                    m_q <= soft_m; l_q <= soft_l; p_q <= soft_p; corr_q <= soft_corr;
                    beat_i <= 0; state <= S_VLOAD;
                end

                // load V, then axpy walk
                S_VLOAD: if (v_tvalid) begin
                    for (i = 0; i < LANES; i = i + 1)
                        v_buf[beat_i*LANES + i] <= v_tdata[i*16 +: 16];
                    if (last_vb) begin elem_i <= 0; walk_axpy <= 1'b1; state <= S_AXPYF; end
                    else beat_i <= beat_i + 16'd1;
                end
                S_SKIPV: if (v_tvalid) begin
                    if (last_vb) begin beat_i <= 0; state <= S_KVNXT; end
                    else beat_i <= beat_i + 16'd1;
                end

                S_AXPYF: state <= S_AXPYW;
                S_AXPYW: if (wav) begin
                    acc[elem] <= wao;
                    if (last_elem) state <= S_KVNXT;
                    else begin elem_i <= elem_i + 16'd1; state <= S_AXPYF; end
                end

                S_KVNXT: if (last_kv) state <= S_NORMF;
                         else begin kv_i <= kv_i + 16'd1; state <= S_MASK; end

                // normalize + emit
                S_NORMF: state <= S_NORMW;
                S_NORMW: if (rec_v) begin inv_l_q <= rec_out; elem_i <= 0; walk_axpy <= 1'b0; state <= S_EMITF; end
                S_EMITF: state <= S_EMITW;
                S_EMITW: if (wav) begin
                    // O[elem] = wao on o_tdata (valid this cycle); assume sink ready.
                    if (last_elem) state <= S_HEADN;
                    else begin elem_i <= elem_i + 16'd1; state <= S_EMITF; end
                end

                S_HEADN: begin
                    beat_i <= 0;
                    if (last_head) begin
                        head_i <= 0;
                        if (last_tok) state <= S_DONE;
                        else begin tok_i <= tok_i + 16'd1; state <= S_QLOAD; end
                    end else begin head_i <= head_i + 16'd1; state <= S_QLOAD; end
                end

                S_DONE: begin busy <= 1'b0; done <= 1'b1; state <= S_IDLE; end
                default: state <= S_IDLE;
            endcase
        end
    end

    wire _unused = &{1'b0, soft_grew, m2v, dot_v, o_tready};
endmodule
