// gemm - the DSP-resident fixed-point matmul datapath (plan-fpga-7.md §gemm.v).
//
// The collapse: this file replaces the matmul_reducer fp32 tail + the matmul_rowblock
// fp32 accumulate-recurrence + the ROWS*(ACCUM_DEPTH-1) fp32 emit tree with the
// "cheap multiply, exact wide accumulate" datapath — the WEIGHT_FMT front-end (Σ±a → int
// S) + f16-scale decompose feeding ROWS numeric/fma lanes (single-cycle wide fixed-point
// accumulate). Built bottom-up and cosim-gated, the same way fma landed:
//
//   increment 1 (here): gemm_rowblock — wire -> per-row 72-bit accumulator, BIT-EXACT vs
//                       matmul_ref.windowedRow (exact-in-window). The genuinely-new part
//                       is gemm_f16_decompose; the Σ±a reduction is re-extracted from
//                       matmul_reducer; fma is the proven leaf.
//   increment 2: the emit normalize (acc -> f32) vs windowedFixedOutput.
//   increment 3: the COLS_MAX column mux + the AXIS feed FSM (re-extract matmul_kernel,
//                delete the recurrence-hiding machinery) vs windowedFixedOutput.
//
// The front-end (decompose + reduce) is gemm-internal — NOT a numeric/ leaf
// (plan-numeric-leaves.md): attention widens f16 with cvt_f16_f32, it never decomposes
// a scale to (significand, exponent). So these helpers live with the consumer.

`default_nettype none

// f16 -> exact (signed significand, signed exponent): value = sig · 2^e, no rounding.
// MUST match matmul_ref.decompose exactly (the cosim oracle):
//   normal (exp_field 1..30): sig = ±(1024+mant), e = exp_field − 25
//   zero/subnormal (exp_field 0): sig = ±mant,    e = −24
// Quant scales are finite/normal, so exp_field==31 (inf/nan) is not represented — the
// cosim sweeps every other f16 pattern.
module gemm_f16_decompose #(
    parameter integer SIG_W = 12,   // signed significand: 11-bit magnitude (max 2047) + sign
    parameter integer EXP_W = 8     // signed exponent: small, range [−24, 5]
) (
    input  wire [15:0]            f16,
    output wire signed [SIG_W-1:0] sig,
    output wire signed [EXP_W-1:0] e
);
    wire        sign        = f16[15];
    wire [4:0]  exp_field   = f16[14:10];
    wire [9:0]  mant        = f16[9:0];
    wire        is_zero_sub = (exp_field == 5'd0);

    // magnitude: 11-bit (1024+mant <= 2047 normal; mant <= 1023 subnormal/zero).
    wire [10:0] mag = is_zero_sub ? {1'b0, mant} : (11'd1024 + {1'b0, mant});

    assign sig = sign ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
    // zero-extend the 5-bit magnitude exp_field before $signed (it is non-negative),
    // then bias. The integer constants sign-extend into the signed [EXP_W-1:0] result.
    assign e   = is_zero_sub ? -24
                             : ($signed({{(EXP_W-5){1'b0}}, exp_field}) - 25);
endmodule

// gemm_emit - the per-output fixed->fp emit: wide accumulator (value = acc · 2^emin) ->
// truncated fp32. The "expensive normalize moves from per-add to per-output"
// (plan-fpga-7.md:114): one leading-zero-detect + shift here, not in the recurrence.
//
// Generalizes cvt_i2f to a wide signed input with an exponent offset: align the leading 1
// to bit 23, take the 23 bits below it toward zero (truncating — the house style of
// fmul/fadd/cvt), and bias the exponent by (msb + emin). The new path the plan keeps with
// the gemm datapath (oracle-gated vs windowedFixedOutput), NOT in numeric/cvt.
//
// PIPELINED over GEMM_EMIT_LATENCY=3 stages (negate | leading-zero-detect | shift+compose):
// the combinational form OOC'd at 198 MHz (WNS −1.7ns — the 72b LZD + barrel shift + negate
// in series, ~23 logic levels, fails f300). Each stage is one of those blocks. valid_in →
// valid_out tracks the pipeline; consumers fill once per output (off the throughput path),
// so the latency is free. Output value unchanged (bit-exact vs matmul_ref.emitTrunc).
module gemm_emit #(
    parameter integer ACC_W = 104,  // full-f16-range fixed window (matmul_ref.ACC_W_BITS)
    parameter integer EXP_W = 8
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    valid_in,
    input  wire signed [ACC_W-1:0] acc,
    input  wire signed [EXP_W-1:0] emin,
    output reg                     valid_out,
    output reg  [31:0]             f32
);
    localparam integer LZW = $clog2(ACC_W);   // leading-1 index width (0..ACC_W-1)

    // ---- s1: |acc| (the 72-bit negate) ----
    wire             sign0 = acc[ACC_W-1];
    wire [ACC_W-1:0] mag0  = sign0 ? -acc : acc;
    reg                    v1, sign1, zero1;
    reg [ACC_W-1:0]        mag1;
    reg signed [EXP_W-1:0] emin1;
    always @(posedge clk) begin
        if (!rst_n) begin v1<=1'b0; sign1<=1'b0; zero1<=1'b0; mag1<=0; emin1<=0; end
        else begin
            v1 <= valid_in; sign1 <= sign0; zero1 <= (mag0 == {ACC_W{1'b0}});
            mag1 <= mag0; emin1 <= emin;
        end
    end

    // ---- s2: leading-1 position (the priority encoder) ----
    reg [LZW-1:0] msb_c;
    integer i;
    always @(*) begin
        msb_c = {LZW{1'b0}};
        for (i = 0; i < ACC_W; i = i + 1)
            if (mag1[i]) msb_c = i[LZW-1:0];
    end
    reg                    v2, sign2, zero2;
    reg [ACC_W-1:0]        mag2;
    reg signed [EXP_W-1:0] emin2;
    reg [LZW-1:0]          msb2;
    always @(posedge clk) begin
        if (!rst_n) begin v2<=1'b0; sign2<=1'b0; zero2<=1'b0; mag2<=0; emin2<=0; msb2<=0; end
        else begin
            v2 <= v1; sign2 <= sign1; zero2 <= zero1; mag2 <= mag1; emin2 <= emin1; msb2 <= msb_c;
        end
    end

    // ---- s3: shift to the window + exponent + compose ----
    // align the leading 1 to bit 23: right-shift drops low bits (truncate) when msb>=23;
    // left-shift is lossless when msb<23. The hidden bit lands at 23; mantissa = [22:0].
    wire [LZW-1:0]   sh_r = (msb2 >= 23) ? (msb2 - 7'd23) : {LZW{1'b0}};
    wire [LZW-1:0]   sh_l = (msb2 <  23) ? (7'd23 - msb2) : {LZW{1'b0}};
    wire [ACC_W-1:0] norm = (msb2 >= 23) ? (mag2 >> sh_r) : (mag2 << sh_l);
    wire [22:0]      mant = norm[22:0];
    // unbiased exponent = msb + emin; biased = + 127. Sign-extend both to a generous width.
    wire signed [15:0] msb_ext  = $signed({{(16-LZW){1'b0}}, msb2});
    wire signed [15:0] emin_ext = $signed({{(16-EXP_W){emin2[EXP_W-1]}}, emin2});
    wire signed [15:0] e_b      = msb_ext + emin_ext + 16'sd127;
    wire underflow = (e_b <= 16'sd0);
    wire overflow  = (e_b >= 16'sd255);
    wire [31:0] f32_c = zero2     ? 32'd0
                      : underflow ? {sign2, 31'd0}
                      : overflow  ? {sign2, 8'hFE, 23'h7FFFFF}   // saturate to max finite
                                  : {sign2, e_b[7:0], mant};
    always @(posedge clk) begin
        if (!rst_n) begin valid_out<=1'b0; f32<=32'd0; end
        else begin valid_out <= v2; f32 <= f32_c; end
    end
endmodule

// gemm_rowblock - one rowblock's fixed-point compute, COLS_MAX-banked.
//
// Per issue (valid_in): every row folds one contribution into accumulator acc[row][col_idx].
// `clear` marks the first sub-block of a new output (resets that column's acc). Decode (C=1)
// uses one accumulator per row (col 0); prefill (C>1) holds a weight beat and sweeps
// columns — col_idx selects which of the COLS_MAX per-row accumulators, the weight-reuse
// axis. This is the matmul_rowblock acc[ROWS][COLS_MAX] pattern, but the accumulate is a
// SINGLE-CYCLE saturating add (no ACCUM_DEPTH pool, no issue_gap): re-issuing the same
// column N cycles later is safe because the write is registered and visible next cycle.
//
// The multiply+shift (fma_mulshift) runs ONCE per issue per row — the DSP cost is NOT
// replicated per column (prefill reuses weights, one multiply/cycle). Its `shifted` output
// feeds a per-row saturating accumulate into acc[row][wcol], where wcol is col_idx delayed
// to align (FE_LAT + the mulshift latency). Output acc_flat = acc[*][read_col] (ROWS×ACC_W),
// the emit reads one column at a time. Total latency input→acc ≈ FE_LAT + fma latency.
module gemm_rowblock #(
    parameter integer ROWS     = 16,
    parameter integer COLS_MAX = 8,   // prefill column fan-out (decode uses col 0 only)
    // numeric/fma fixed-point contract (kept equal to fma's defaults).
    parameter integer SIG_W    = 12,  // f16 signed significand
    parameter integer S_W      = 14,  // signed sub-block sum (|S| <= 32·128 = 4096)
    parameter integer EXP_W    = 8,   // signed combined exponent / window floor
    parameter integer ACC_W    = 104, // wide fixed-point accumulator (full-f16-range fixed window)
    parameter integer SHAMT_W  = 7,   // window shift magnitude
    parameter integer CW       = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,        // first sub-block of a new output
    input  wire                    valid_in,
    input  wire [CW-1:0]           col_idx,      // accumulator column this issue targets
    input  wire signed [EXP_W-1:0] emin,         // calibration window floor
    input  wire [ROWS*32-1:0]      weight_bits_flat,    // per row: 32 sign bits (this sub-block)
    input  wire [ROWS*16-1:0]      weight_scales_flat,  // per row: f16 weight scale
    input  wire [255:0]            acts_packed,         // 32 int8 activations (shared)
    input  wire [15:0]             act_scale,           // f16 activation scale (shared)
    input  wire [CW-1:0]           read_col,            // which column acc_flat exposes
    output wire [ROWS*ACC_W-1:0]   acc_flat
);
    // Σ±a front-end, PIPELINED over FE_LAT register stages (matmul_reducer's 8→4→1
    // balance) so the 32-element add tree closes timing: the combinational version
    // OOC'd at 212 MHz (fails f300; worst path acts→tree→DSP, 13 logic levels). The
    // fast scale decode + clear/valid/col are delayed FE_LAT to meet the pipelined sub-sum
    // at fma_mulshift. Output is the SAME integer sub-sum (add is associative — bit-exact
    // vs matmul_ref.subSum), just FE_LAT cycles later.
    //
    // FE_LAT=4 = ONE input-register stage (acts_r/wb_r) + the 3-deep quad pipeline. The
    // input stage exists because in the full kernel the activations come from a BRAM
    // (acts_mem) whose ~1.2ns clock-to-out, in series with the Σ±a tree, missed f300 by
    // 0.3ns (full-kernel OOC). Registering the quad inputs gives the BRAM read its own
    // cycle. The small scale-decode tolerates the BRAM delay, so it stays on the raw
    // inputs (delayed FE_LAT) — keeping every alignment index uniform at FE_LAT-1.
    localparam integer FE_LAT       = 4;
    // fma_mulshift's valid_in→shifted/valid_out latency (s0..s3). The column index is
    // delayed FE_LAT+MULSHIFT_LAT so wcol selects the right accumulator when `shifted`
    // lands; the cosim's C>1 sweep gates this alignment (a wrong depth → wrong column).
    localparam integer MULSHIFT_LAT = 4;
    localparam integer COL_LAT      = FE_LAT + MULSHIFT_LAT;

    // The accumulate is CARRY-SAVE (see the bank below): no full-width add, so no saturating
    // add here. The fixed full-f16 window (matmul_ref.fixedWindow, ACC_W=104) cannot overflow,
    // so the redundant pair resolves exactly with no clamp.

    // One quad's Σ±a over 4 int8 (combinational; 8 quads tile the 32-wide sub-block).
    function automatic signed [S_W-1:0] quad_sum;
        input [3:0]  wb;
        input [31:0] a4;
        integer e;
        reg signed [S_W-1:0] s, a;
        begin
            s = {S_W{1'b0}};
            for (e = 0; e < 4; e = e + 1) begin
                a = {{(S_W-8){a4[e*8+7]}}, a4[e*8 +: 8]}; // sext int8 -> S_W
                s = s + (wb[e] ? a : -a);
            end
            quad_sum = s;
        end
    endfunction

    // shared act-scale decompose (combinational, once), delayed FE_LAT to align at fma.
    wire signed [SIG_W-1:0] as_sig_c;
    wire signed [EXP_W-1:0] as_e_c;
    gemm_f16_decompose #(.SIG_W(SIG_W), .EXP_W(EXP_W)) u_as (
        .f16(act_scale), .sig(as_sig_c), .e(as_e_c)
    );

    // input-register stage for the quad-sum inputs (the BRAM-acts f300 fix): register the
    // activations and weight sign-bits so the BRAM read and the Σ±a tree don't share a cycle.
    reg [255:0]       acts_r;
    reg [ROWS*32-1:0] wb_r;
    always @(posedge clk) begin
        acts_r <= acts_packed;
        wb_r   <= weight_bits_flat;
    end

    // shared control + as_sig delay lines (FE_LAT deep). The scale decode runs on the RAW
    // inputs (its small logic absorbs the BRAM clock-to-out), so it aligns at FE_LAT too.
    reg [FE_LAT-1:0]       clear_sr, valid_sr;
    reg signed [SIG_W-1:0] as_sig_d [0:FE_LAT-1];
    integer d;
    always @(posedge clk) begin
        if (!rst_n) begin
            clear_sr <= {FE_LAT{1'b0}};
            valid_sr <= {FE_LAT{1'b0}};
        end else begin
            clear_sr <= {clear_sr[FE_LAT-2:0], clear};
            valid_sr <= {valid_sr[FE_LAT-2:0], valid_in};
        end
        as_sig_d[0] <= as_sig_c;
        for (d = 1; d < FE_LAT; d = d + 1) as_sig_d[d] <= as_sig_d[d-1];
    end

    // col_idx delayed COL_LAT to align with `shifted`/`valid_out` at the accumulate.
    reg [CW-1:0] col_pipe [0:COL_LAT-1];
    integer cp;
    always @(posedge clk) begin
        col_pipe[0] <= col_idx;
        for (cp = 1; cp < COL_LAT; cp = cp + 1) col_pipe[cp] <= col_pipe[cp-1];
    end
    wire [CW-1:0] wcol = col_pipe[COL_LAT-1];

    // Build the accumulator write enable before the existing control pipeline, then carry
    // it alongside col_pipe. The last stage is duplicated per lane below. This avoids a
    // timing path from a replicated col_pipe bit through a 3,105-load combinational decode.
    wire [COLS_MAX-1:0] acc_write_onehot;
    genvar wc;
    generate
        for (wc = 0; wc < COLS_MAX; wc = wc + 1) begin : gen_acc_write_decode
            assign acc_write_onehot[wc] = (valid_in || clear) &&
                                           (col_idx == wc[CW-1:0]);
        end
    endgenerate

    // Seven shared stages plus the lane-local leaf register form the same COL_LAT=8 delay
    // as col_pipe and valid_out/clear_out. Reset only this small control pipeline; accS/accC
    // remain unreset and are initialized by their explicit first-contribution clear.
    reg [COLS_MAX-1:0] acc_write_pipe [0:COL_LAT-2];
    integer awp;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (awp = 0; awp < COL_LAT-1; awp = awp + 1)
                acc_write_pipe[awp] <= {COLS_MAX{1'b0}};
        end else begin
            acc_write_pipe[0] <= acc_write_onehot;
            for (awp = 1; awp < COL_LAT-1; awp = awp + 1)
                acc_write_pipe[awp] <= acc_write_pipe[awp-1];
        end
    end
    wire [COLS_MAX-1:0] acc_write_preleaf = acc_write_pipe[COL_LAT-2];

    genvar r, g, cc;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : gen_lane
            // per-row weight-scale decompose (combinational), delayed FE_LAT.
            wire signed [SIG_W-1:0] ws_sig_c;
            wire signed [EXP_W-1:0] ws_e_c;
            gemm_f16_decompose #(.SIG_W(SIG_W), .EXP_W(EXP_W)) u_ws (
                .f16(weight_scales_flat[r*16 +: 16]), .sig(ws_sig_c), .e(ws_e_c)
            );
            wire signed [EXP_W-1:0] p_exp_c = ws_e_c + as_e_c;
            reg signed [SIG_W-1:0] ws_sig_d [0:FE_LAT-1];
            reg signed [EXP_W-1:0] p_exp_d  [0:FE_LAT-1];
            integer k;
            always @(posedge clk) begin
                ws_sig_d[0] <= ws_sig_c;
                p_exp_d[0]  <= p_exp_c;
                for (k = 1; k < FE_LAT; k = k + 1) begin
                    ws_sig_d[k] <= ws_sig_d[k-1];
                    p_exp_d[k]  <= p_exp_d[k-1];
                end
            end

            // quad partials over the REGISTERED inputs (acts_r/wb_r): input reg (stage 0)
            // then 8 quad partials (comb) -> ps1 ; 8->4 -> l1_2 ; 4->1 -> s_sum_3 (3 inner).
            wire signed [S_W-1:0] ps_c [0:7];
            for (g = 0; g < 8; g = g + 1) begin : gen_quad
                assign ps_c[g] = quad_sum(wb_r[r*32 + g*4 +: 4],
                                          acts_r[g*32 +: 32]);
            end
            reg signed [S_W-1:0] ps1 [0:7];
            reg signed [S_W-1:0] l1_2 [0:3];
            reg signed [S_W-1:0] s_sum_3;
            integer j;
            always @(posedge clk) begin
                for (j = 0; j < 8; j = j + 1) ps1[j] <= ps_c[j];
                for (j = 0; j < 4; j = j + 1) l1_2[j] <= ps1[2*j] + ps1[2*j+1];
                s_sum_3 <= (l1_2[0] + l1_2[1]) + (l1_2[2] + l1_2[3]);
            end

            // multiply + window-shift (one per row, reused across columns).
            wire                    vout, clr_out;
            wire signed [ACC_W-1:0] shifted;
            fma_mulshift #(
                .SIG_W(SIG_W), .S_W(S_W), .EXP_W(EXP_W), .ACC_W(ACC_W), .SHAMT_W(SHAMT_W)
            ) u_ms (
                .clk(clk), .rst_n(rst_n),
                .clear(clear_sr[FE_LAT-1]), .valid_in(valid_sr[FE_LAT-1]),
                .ws_sig(ws_sig_d[FE_LAT-1]), .as_sig(as_sig_d[FE_LAT-1]),
                .p_exp(p_exp_d[FE_LAT-1]), .emin(emin), .s_sum(s_sum_3),
                .valid_out(vout), .clear_out(clr_out), .shifted(shifted)
            );

            // Intentionally duplicate the final registered control per lane and CSA bank.
            // Each leaf bit directly drives ACC_W=104 clock enables. KEEP prevents synthesis
            // from merging these equivalent registers back into a device-spanning control net;
            // unlike DONT_TOUCH, it still permits placement-aware physical optimization.
            (* keep = "true" *) reg [COLS_MAX-1:0] accS_write_enable;
            (* keep = "true" *) reg [COLS_MAX-1:0] accC_write_enable;
            always @(posedge clk) begin
                if (!rst_n) begin
                    accS_write_enable <= {COLS_MAX{1'b0}};
                    accC_write_enable <= {COLS_MAX{1'b0}};
                end else begin
                    accS_write_enable <= acc_write_preleaf;
                    accC_write_enable <= acc_write_preleaf;
                end
            end

            // COLS_MAX-wide accumulator bank, in CARRY-SAVE form: the value of column cc is the
            // redundant pair accS[cc] + accC[cc]. Each contribution folds in with a 3:2 CSA
            // (sum = a^b^c, carry = majority(a,b,c)<<1) — ONE logic level, NO carry chain — so
            // the single-cycle recurrence has no 104-bit carry-propagate add on its critical
            // path (that was the f250 worst path). The one real add (accS+accC) happens once per
            // output at readout (acc_flat), off the throughput path. The full f16 window never
            // overflows, so the redundant pair resolves exactly with no saturation.
            reg signed [ACC_W-1:0] accS [0:COLS_MAX-1];
            reg signed [ACC_W-1:0] accC [0:COLS_MAX-1];
            wire signed [ACC_W-1:0] baseS = clr_out ? {ACC_W{1'b0}} : accS[wcol];
            wire signed [ACC_W-1:0] baseC = clr_out ? {ACC_W{1'b0}} : accC[wcol];
            // CSA(baseS, baseC, shifted): newS + newC == baseS + baseC + shifted (no carry prop).
            wire signed [ACC_W-1:0] csaS = baseS ^ baseC ^ shifted;
            wire signed [ACC_W-1:0] csaC = ((baseS & baseC) | (baseS & shifted) | (baseC & shifted)) <<< 1;
            wire signed [ACC_W-1:0] nextS = vout ? csaS : baseS; // clear-only (clr_out & !vout): hold 0
            wire signed [ACC_W-1:0] nextC = vout ? csaC : baseC;
            // Do not reset the accumulator data bank. The first contribution to every active
            // column arrives with clr_out asserted, which selects zero above before writing the
            // new CSA pair. Resetting these otherwise-invalid data registers created a 26,624-FF
            // branch on the design-wide reset net and consumed ordinary data-routing resources.
            for (cc = 0; cc < COLS_MAX; cc = cc + 1) begin : gen_acc
                always @(posedge clk) begin
                    if (accS_write_enable[cc])
                        accS[cc] <= nextS;
                    if (accC_write_enable[cc])
                        accC[cc] <= nextC;
                end
            end
            // resolve the redundant pair for readout, PIPELINED: register the read_col-muxed pair
            // (rdS_q/rdC_q) then add, so the single 104-bit carry-propagate add is a clean
            // FF→add→FF — no read_col mux in series with it. Off the throughput path; costs +1
            // readout latency, which gemm_kernel realigns (pc_beat/pc_vld). The accumulate
            // recurrence (the CSA above) is untouched and stays single-cycle.
            reg signed [ACC_W-1:0] rdS_q, rdC_q;
            always @(posedge clk) begin
                rdS_q <= accS[read_col];
                rdC_q <= accC[read_col];
            end
            assign acc_flat[r*ACC_W +: ACC_W] = rdS_q + rdC_q;
        end
    endgenerate
endmodule
