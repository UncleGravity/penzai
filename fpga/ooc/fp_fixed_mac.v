// fp_fixed_mac - plan-7 phase-2 DSP-inference derisk: the candidate per-row lane.
//
// Models the "cheap multiply, exact wide accumulate" datapath that REPLACES the
// matmul_reducer fp32 tail + the fp32 acc-recurrence + the emit tree. One lane =
// one output row's fixed-point accumulator (the decode/single_col case).
//
// Per sub-block contribution (faithful to plan-fpga-7.md Part 1):
//   1. P     = ws_sig * as_sig        (f16 significands)  -> DSP  (11x11 -> 22b)
//   2. prod  = P * s_sum              (s_sum = int ternary sum)   -> DSP  (23x14 -> 37b signed)
//   3. shift prod into the window by  exp(P) - EMIN                -> barrel shift (LUT)
//   4. acc  += shifted                single-cycle wide add        -> CARRY8 (the bet)
//
// The whole point: the accumulate is a SINGLE-CYCLE recurrence (acc <- acc + x),
// so there is no pool and no emit tree. The two multiplies are forced into DSPs
// (as the existing fp32_mul_pipe already does) so we keep ~the same DSP count and
// pay the area only in the barrel shift -- which the OOC synth quantifies.
//
// This is a throwaway probe (not synthesized into any bitstream); it exists to put
// real DSP / LUT / CARRY8 / Fmax numbers on the phase-2 bet before a 30-min build.

`default_nettype none

module fp_fixed_mac #(
    parameter integer WS_SIG_W = 11,   // f16 significand incl. hidden 1
    parameter integer AS_SIG_W = 11,
    parameter integer S_W      = 14,   // signed ternary/int sub-block sum (|S| <= 4096)
    parameter integer EXP_W    = 5,    // f16 exponent
    parameter integer ACC_W    = 72,   // wide fixed-point accumulator
    parameter integer SHAMT_W  = 7,    // window shift amount (0..127)
    parameter integer EMIN     = 0     // window floor (calibration; arbitrary for synth)
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear,      // start a new output: zero acc
    input  wire                     valid_in,   // a sub-block contribution is present
    input  wire [WS_SIG_W-1:0]      ws_sig,
    input  wire [AS_SIG_W-1:0]      as_sig,
    input  wire [EXP_W-1:0]         ws_exp,
    input  wire [EXP_W-1:0]         as_exp,
    input  wire signed [S_W-1:0]    s_sum,
    output reg  signed [ACC_W-1:0]  acc
);
    localparam integer P_W    = WS_SIG_W + AS_SIG_W;        // 22
    localparam integer PROD_W = (P_W + 1) + S_W;            // 23 (signed P) + 14 = 37
    localparam integer EXPSUM_W = EXP_W + 1;                // ws_exp + as_exp

    // ---- s0: register inputs (these become the DSP A/B input registers) ----
    reg                  v0, clr0;
    reg [WS_SIG_W-1:0]   ws_sig0;
    reg [AS_SIG_W-1:0]   as_sig0;
    reg [EXP_W-1:0]      ws_exp0, as_exp0;
    reg signed [S_W-1:0] s_sum0;
    always @(posedge clk) begin
        if (!rst_n) begin
            v0 <= 1'b0; clr0 <= 1'b0;
            ws_sig0 <= 0; as_sig0 <= 0; ws_exp0 <= 0; as_exp0 <= 0; s_sum0 <= 0;
        end else begin
            v0 <= valid_in; clr0 <= clear;
            ws_sig0 <= ws_sig; as_sig0 <= as_sig;
            ws_exp0 <= ws_exp; as_exp0 <= as_exp; s_sum0 <= s_sum;
        end
    end

    // ---- s1: P = ws_sig * as_sig (DSP), exp_p = ws_exp + as_exp ----
    reg                       v1, clr1;
    (* use_dsp = "yes" *) reg [P_W-1:0] p_sig1;
    reg [EXPSUM_W-1:0]        exp_p1;
    reg signed [S_W-1:0]      s_sum1;
    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0; clr1 <= 1'b0; p_sig1 <= 0; exp_p1 <= 0; s_sum1 <= 0;
        end else begin
            v1     <= v0; clr1 <= clr0;
            p_sig1 <= ws_sig0 * as_sig0;
            exp_p1 <= {1'b0, ws_exp0} + {1'b0, as_exp0};
            s_sum1 <= s_sum0;
        end
    end

    // ---- s2: prod = P * s_sum (DSP, signed); shamt = exp_p - EMIN (floored at 0) ----
    // exp_p max = 2*(2^EXP_W-1) so shamt fits SHAMT_W; only the negative floor matters.
    reg                          v2, clr2;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] prod2;
    reg [SHAMT_W-1:0]            shamt2;
    localparam signed [EXPSUM_W:0] EMIN_S = EMIN[EXPSUM_W:0];
    wire signed [EXPSUM_W:0]     shamt_raw = $signed({1'b0, exp_p1}) - EMIN_S;
    always @(posedge clk) begin
        if (!rst_n) begin
            v2 <= 1'b0; clr2 <= 1'b0; prod2 <= 0; shamt2 <= 0;
        end else begin
            v2    <= v1; clr2 <= clr1;
            prod2 <= $signed({1'b0, p_sig1}) * s_sum1;
            shamt2 <= shamt_raw[EXPSUM_W] ? {SHAMT_W{1'b0}} : shamt_raw[SHAMT_W-1:0];
        end
    end

    // ---- s3: barrel-shift prod into the wide window (the new LUT cost) ----
    reg                     v3, clr3;
    reg signed [ACC_W-1:0]  shifted3;
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-PROD_W){prod2[PROD_W-1]}}, prod2};
    always @(posedge clk) begin
        if (!rst_n) begin
            v3 <= 1'b0; clr3 <= 1'b0; shifted3 <= 0;
        end else begin
            v3 <= v2; clr3 <= clr2;
            shifted3 <= prod_ext <<< shamt2;
        end
    end

    // ---- s4: single-cycle wide accumulate (the bet -> CARRY8 chain) ----
    always @(posedge clk) begin
        if (!rst_n)        acc <= {ACC_W{1'b0}};
        else if (clr3)     acc <= v3 ? shifted3 : {ACC_W{1'b0}};
        else if (v3)       acc <= acc + shifted3;
    end
endmodule
