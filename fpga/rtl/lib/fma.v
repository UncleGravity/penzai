// Fixed-point multiply-accumulate used by the projection GEMM.
//
// Per contribution: m = ws_sig·as_sig·s_sum (two DSP multiplies) → shift into the window
// by (p_exp - emin) → accumulate into a wide single-cycle integer accumulator. `clear`
// marks the first contribution of a new output (resets acc). The accumulate is a
// single-cycle recurrence. Emit (acc -> f16 normalize) is a separate conversion, once per
// output. Inside the window the integer adds are exact and order-independent.
//
// The multiply and shift (s0..s3) are exposed as `fma_mulshift`; this module adds the
// saturating accumulate (s4). The split lets the GEMM prefill path reuse one mul+shift
// per row against a COLS_MAX accumulator bank. The DSP-backed multiply is not
// replicated per column.

`default_nettype none

// fma_mulshift - the multiply + window-shift (fma s0..s3), exposing the windowed value to
// accumulate. Latency valid_in -> valid_out/shifted = 4 cycles. Designed around the DSP
// (AREG/BREG/MREG/PREG): the two multiplies run register-to-register.
module fma_mulshift #(
    parameter integer SIG_W   = 12,
    parameter integer S_W     = 14,
    parameter integer EXP_W   = 8,
    parameter integer ACC_W   = 104,  // full-f16-range fixed window (matmul_ref.ACC_W_BITS)
    parameter integer SHAMT_W = 7
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,
    input  wire                    valid_in,
    input  wire signed [SIG_W-1:0] ws_sig,
    input  wire signed [SIG_W-1:0] as_sig,
    input  wire signed [EXP_W-1:0] p_exp,
    input  wire signed [EXP_W-1:0] emin,
    input  wire signed [S_W-1:0]   s_sum,
    output reg                     valid_out,
    output reg                     clear_out,
    output reg  signed [ACC_W-1:0] shifted
);
    localparam integer PSIG_W = 2 * SIG_W;        // ws_sig·as_sig
    localparam integer PROD_W = PSIG_W + S_W;     // ·s_sum
    // Replicate the shift-direction select at the s2 boundary so one register does not
    // drive every bit of the final wide mux. At ACC_W=104 this makes seven 14/15-bit
    // regions; the clamp keeps other supported widths to a small, bounded register set.
    localparam integer SHIFT_DIR_REPLICAS_RAW = (ACC_W + 15) / 16;
    localparam integer SHIFT_DIR_REPLICAS =
        (SHIFT_DIR_REPLICAS_RAW < 4) ? 4 :
        (SHIFT_DIR_REPLICAS_RAW > 8) ? 8 : SHIFT_DIR_REPLICAS_RAW;
    localparam integer SHIFT_DIR_SLICE_W =
        (ACC_W + SHIFT_DIR_REPLICAS - 1) / SHIFT_DIR_REPLICAS;

    // ---- s0: register inputs (DSP A/B input regs) ----
    reg                    v0, clr0;
    reg signed [SIG_W-1:0] ws0, as0;
    reg signed [EXP_W-1:0] pexp0, emin0;
    reg signed [S_W-1:0]   s0;
    always @(posedge clk) begin
        if (!rst_n) begin v0<=1'b0; clr0<=1'b0; ws0<=0; as0<=0; pexp0<=0; emin0<=0; s0<=0; end
        else begin
            v0 <= valid_in; clr0 <= clear;
            ws0 <= ws_sig; as0 <= as_sig;
            pexp0 <= p_exp; emin0 <= emin; s0 <= s_sum;
        end
    end

    // ---- s1: psig = ws·as (DSP); shamt = p_exp - emin ----
    reg                  v1, clr1;
    (* use_dsp = "yes" *) reg signed [PSIG_W-1:0] psig1;
    reg signed [EXP_W:0] shamt1;
    reg signed [S_W-1:0] s1;
    always @(posedge clk) begin
        if (!rst_n) begin v1<=1'b0; clr1<=1'b0; psig1<=0; shamt1<=0; s1<=0; end
        else begin
            v1 <= v0; clr1 <= clr0;
            psig1  <= ws0 * as0;
            shamt1 <= $signed({pexp0[EXP_W-1], pexp0}) - $signed({emin0[EXP_W-1], emin0});
            s1 <= s0;
        end
    end

    // ---- s2: prod = psig·s (DSP) ----
    reg                  v2, clr2;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod2;
    reg signed [EXP_W:0] shamt2;
    (* keep = "true" *) reg [SHIFT_DIR_REPLICAS-1:0] shift_dir2_rep;
    always @(posedge clk) begin
        if (!rst_n) begin
            v2<=1'b0; clr2<=1'b0; prod2<=0; shamt2<=0; shift_dir2_rep<=0;
        end else begin
            v2<=v1; clr2<=clr1; prod2 <= psig1 * s1; shamt2 <= shamt1;
            shift_dir2_rep <= {SHIFT_DIR_REPLICAS{shamt1[EXP_W]}};
        end
    end

    // ---- s3: shift into the window (barrel; arithmetic right when below floor) ----
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-PROD_W){prod2[PROD_W-1]}}, prod2};
    wire signed [EXP_W:0]   neg_shamt = -shamt2;
    wire [SHAMT_W-1:0]      up   = shamt2[SHAMT_W-1:0];
    wire [SHAMT_W-1:0]      down = neg_shamt[SHAMT_W-1:0];
    wire signed [ACC_W-1:0] shifted_right = prod_ext >>> down;
    wire signed [ACC_W-1:0] shifted_left  = prod_ext <<< up;
    wire signed [ACC_W-1:0] shifted_next;

    genvar shift_slice;
    generate
        for (shift_slice = 0; shift_slice < SHIFT_DIR_REPLICAS; shift_slice = shift_slice + 1) begin : g_shift_dir
            if (shift_slice * SHIFT_DIR_SLICE_W < ACC_W) begin : g_used
                localparam integer LO = shift_slice * SHIFT_DIR_SLICE_W;
                localparam integer HI = (((shift_slice + 1) * SHIFT_DIR_SLICE_W) > ACC_W) ?
                                        (ACC_W - 1) : (((shift_slice + 1) * SHIFT_DIR_SLICE_W) - 1);
                assign shifted_next[HI:LO] = shift_dir2_rep[shift_slice] ?
                                             shifted_right[HI:LO] : shifted_left[HI:LO];
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin valid_out<=1'b0; clear_out<=1'b0; shifted<=0; end
        else begin
            valid_out <= v2; clear_out <= clr2;
            shifted   <= shifted_next;
        end
    end
endmodule

module fma #(
    parameter integer SIG_W   = 12,   // signed significand (11-bit f16 mag + sign)
    parameter integer S_W     = 14,   // signed sub-block sum
    parameter integer EXP_W   = 8,    // signed combined exponent and window floor
    parameter integer ACC_W   = 104,  // wide fixed-point accumulator (full-f16-range fixed window)
    parameter integer SHAMT_W = 7     // window shift magnitude
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,    // first contribution of a new output
    input  wire                    valid_in,
    input  wire signed [SIG_W-1:0] ws_sig,
    input  wire signed [SIG_W-1:0] as_sig,
    input  wire signed [EXP_W-1:0] p_exp,    // e_ws + e_as
    input  wire signed [EXP_W-1:0] emin,     // window floor (calibration)
    input  wire signed [S_W-1:0]   s_sum,
    output reg  signed [ACC_W-1:0] acc
);
    // Saturation limits ±(2^(ACC_W-1)-1), matching matmul_ref.windowedRow. A calibrated
    // window never overflows the wide accumulator; the clamp is the SAFETY NET so an
    // unexpected overflow (mis-calibration / outlier) degrades to a bounded value instead
    // of silently WRAPPING (two's-complement sign flip → garbage logits).
    localparam signed [ACC_W-1:0] LIM_POS = {1'b0, {(ACC_W-1){1'b1}}};        // +2^(ACC_W-1)-1
    localparam signed [ACC_W-1:0] LIM_NEG = {1'b1, {(ACC_W-2){1'b0}}, 1'b1};  // -(2^(ACC_W-1)-1)

    wire                    v3, clr3;
    wire signed [ACC_W-1:0] shifted3;
    fma_mulshift #(.SIG_W(SIG_W), .S_W(S_W), .EXP_W(EXP_W), .ACC_W(ACC_W), .SHAMT_W(SHAMT_W)) u_ms (
        .clk(clk), .rst_n(rst_n), .clear(clear), .valid_in(valid_in),
        .ws_sig(ws_sig), .as_sig(as_sig), .p_exp(p_exp), .emin(emin), .s_sum(s_sum),
        .valid_out(v3), .clear_out(clr3), .shifted(shifted3)
    );

    // ---- s4: single-cycle SATURATING wide accumulate (CARRY8 add + sign-based clamp) ----
    // Two's-complement overflow (operands same sign, result sign flips) → clamp to ±LIM
    // instead of wrapping. The detect is a sign XOR + mux — cheap, and off the DSP path
    // (the critical recurrence path), so the clamp does not add a separate timing stage. The GEMM
    // prefill bank (gemm_rowblock) replicates this same clamp per row; both are gated vs
    // windowedRow's ACC_W_BITS saturation.
    wire signed [ACC_W-1:0] base    = clr3 ? {ACC_W{1'b0}} : acc;
    wire signed [ACC_W-1:0] raw_sum = base + shifted3;
    wire ovf = (base[ACC_W-1] == shifted3[ACC_W-1]) && (raw_sum[ACC_W-1] != base[ACC_W-1]);
    wire signed [ACC_W-1:0] sat_sum = ovf ? (base[ACC_W-1] ? LIM_NEG : LIM_POS) : raw_sum;
    always @(posedge clk) begin
        if (!rst_n)    acc <= {ACC_W{1'b0}};
        else if (clr3) acc <= v3 ? sat_sum : {ACC_W{1'b0}};
        else if (v3)   acc <= sat_sum;
    end
endmodule
