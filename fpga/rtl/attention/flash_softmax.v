// flash_softmax - one online-softmax step, as a stateless pipelined transform.
//
//   m_new = max(m_in, score)              grew = score > m_in
//   corr  = exp(m_in  - m_new)            (1 when not grown; 0 at first kv, m_in=-inf)
//   p     = exp(score - m_new)
//   l_out = l_in * corr + p               m_out = m_new
//
// Implements the online-softmax recurrence consumed by flash_kernel. The kernel
// holds the per-head (m,l) pool and feeds m_out/l_out back as the next kv's
// m_in/l_in. The loop-carried recurrence is closed there and hidden by the n_heads
// interleave, so this unit is purely feed-forward. The kernel always applies
// `acc *= corr; acc += p·V` (corr=1 is a no-op), so `grew` is only an optional hint
// to skip the rescale walk.
//
// Latency valid_in -> valid_out: subtract4 + handoff1 + exp19 + multiply3 +
// handoff1 + add4 + output1 = 33 registered stages. The two handoff stages are
// explicit in the metadata depths below; burst cosim guards their alignment.

`default_nettype none

module flash_softmax (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,
    input  wire [31:0] m_in,
    input  wire [31:0] l_in,
    input  wire [31:0] score,
    output reg         valid_out,
    output reg  [31:0] m_out,
    output reg  [31:0] l_out,
    output reg  [31:0] p,
    output reg  [31:0] corr,
    output reg         grew
);
    `include "fmt.vh"
    localparam integer L_SUB = 4;  // fadd
    localparam integer L_EXP = 19; // exp (mul3 + fx1 + lut1 + interp13 + out1)
    localparam integer L_MUL = 3;  // fmul
    localparam integer L_ADD = 4;  // fadd
    localparam integer D1 = L_SUB + 1 + L_EXP;    // 24: input -> exp output
    localparam integer D2 = L_MUL + 1 + L_ADD;    // 8 : exp output -> valid_out

    integer i;

    // ---- stage 0: fp max + grew (combinational) ----
    function automatic fpGe; // a >= b for fp32 (handles -inf as the smallest value)
        input [31:0] a;
        input [31:0] b;
        begin
            if (a[31] != b[31])      fpGe = (a[31] == 1'b0);      // non-negative wins
            else if (a[31] == 1'b0)  fpGe = (a[30:0] >= b[30:0]); // both >= 0
            else                     fpGe = (a[30:0] <= b[30:0]); // both < 0: smaller mag is larger
        end
    endfunction

    wire        ge0    = fpGe(m_in, score);
    wire        grew0  = ~ge0;                       // score > m_in
    wire [31:0] m_new0 = grew0 ? score : m_in;
    wire [31:0] neg_m_new0 = {~m_new0[31], m_new0[30:0]};

    // ---- subtracts: d_corr = m_in - m_new ; d_p = score - m_new ----
    wire        dcv, dpv;
    wire [31:0] d_corr, d_p;
    fadd #(.MANT_W(FMT_FP32_MANT)) u_dcorr (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a(m_in),  .b(neg_m_new0), .valid_out(dcv), .out(d_corr));
    fadd #(.MANT_W(FMT_FP32_MANT)) u_dp    (.clk(clk), .rst_n(rst_n), .valid_in(valid_in), .a(score), .b(neg_m_new0), .valid_out(dpv), .out(d_p));

    // ---- exp ----
    wire        cv, pv;
    wire [31:0] corr_e, p_e;
    exp u_ec (.clk(clk), .rst_n(rst_n), .valid_in(dcv), .x(d_corr), .valid_out(cv), .y(corr_e));
    exp u_ep (.clk(clk), .rst_n(rst_n), .valid_in(dpv), .x(d_p),    .valid_out(pv), .y(p_e));

    // ---- carry {m_new, grew, l_in} to the exp output (D1 = 24) ----
    reg [31:0] mnew_pre [0:D1-1];
    reg        grew_pre [0:D1-1];
    reg [31:0] lin_pre  [0:D1-1];
    always @(posedge clk) begin
        mnew_pre[0] <= m_new0;
        grew_pre[0] <= grew0;
        lin_pre[0]  <= l_in;
        for (i = 1; i < D1; i = i + 1) begin
            mnew_pre[i] <= mnew_pre[i-1];
            grew_pre[i] <= grew_pre[i-1];
            lin_pre[i]  <= lin_pre[i-1];
        end
    end
    wire [31:0] m_new_e = mnew_pre[D1-1];
    wire        grew_e  = grew_pre[D1-1];
    wire [31:0] l_in_e  = lin_pre[D1-1];

    // ---- l_out = l_in*corr + p ----
    wire        lmv, lav;
    wire [31:0] l_mul, l_sum;
    fmul #(.MANT_W(FMT_FP32_MANT)) u_lmul (.clk(clk), .rst_n(rst_n), .valid_in(cv),  .a(l_in_e), .b(corr_e), .valid_out(lmv), .out(l_mul));

    // carry {m_new, grew, corr, p} from the exp output to valid_out (D2 = 8);
    // p is also tapped at L_MUL (delay 3) to meet l_mul at the adder.
    reg [31:0] mnew_post [0:D2-1];
    reg        grew_post [0:D2-1];
    reg [31:0] corr_post [0:D2-1];
    reg [31:0] p_post    [0:D2-1];
    always @(posedge clk) begin
        mnew_post[0] <= m_new_e;
        grew_post[0] <= grew_e;
        corr_post[0] <= corr_e;
        p_post[0]    <= p_e;
        for (i = 1; i < D2; i = i + 1) begin
            mnew_post[i] <= mnew_post[i-1];
            grew_post[i] <= grew_post[i-1];
            corr_post[i] <= corr_post[i-1];
            p_post[i]    <= p_post[i-1];
        end
    end
    wire [31:0] p_lmul = p_post[L_MUL-1]; // p delayed L_MUL, aligned with l_mul
    fadd #(.MANT_W(FMT_FP32_MANT)) u_ladd (.clk(clk), .rst_n(rst_n), .valid_in(lmv), .a(l_mul), .b(p_lmul), .valid_out(lav), .out(l_sum));

    // ---- register outputs (all aligned at +33) ----
    always @(posedge clk) begin
        if (!rst_n) valid_out <= 1'b0;
        else        valid_out <= lav;
        m_out <= mnew_post[D2-1];
        l_out <= l_sum;
        p     <= p_post[D2-1];
        corr  <= corr_post[D2-1];
        grew  <= grew_post[D2-1];
    end

endmodule
