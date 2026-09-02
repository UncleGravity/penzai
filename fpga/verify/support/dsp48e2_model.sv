// Minimal functional model for the exact DSP48E2 mode used by  dot4.
// This is a simulation-only compile check for the explicit primitive branch;
// Vivado uses UNISIM instead.  Unsupported attributes or OPMODEs fail loudly.

`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */
/* verilator lint_off UNUSEDSIGNAL */

module DSP48E2 #(
    parameter integer ACASCREG = 1,
    parameter integer ADREG = 1,
    parameter integer ALUMODEREG = 1,
    parameter AMULTSEL = "A",
    parameter integer AREG = 1,
    parameter A_INPUT = "DIRECT",
    parameter integer BCASCREG = 1,
    parameter BMULTSEL = "B",
    parameter integer BREG = 1,
    parameter B_INPUT = "DIRECT",
    parameter integer CARRYINREG = 1,
    parameter integer CARRYINSELREG = 1,
    parameter integer CREG = 1,
    parameter integer DREG = 1,
    parameter integer INMODEREG = 1,
    parameter integer MREG = 1,
    parameter integer OPMODEREG = 1,
    parameter PREADDINSEL = "A",
    parameter integer PREG = 1,
    parameter USE_MULT = "MULTIPLY",
    parameter USE_PATTERN_DETECT = "NO_PATDET",
    parameter USE_SIMD = "ONE48",
    parameter USE_WIDEXOR = "FALSE"
) (
    output wire [29:0] ACOUT,
    output wire [17:0] BCOUT,
    output wire CARRYCASCOUT,
    output wire [3:0] CARRYOUT,
    output wire MULTSIGNOUT,
    output wire OVERFLOW,
    output reg  [47:0] P,
    output wire PATTERNBDETECT,
    output wire PATTERNDETECT,
    output wire [47:0] PCOUT,
    output wire UNDERFLOW,
    output wire [7:0] XOROUT,
    input  wire [29:0] A,
    input  wire [29:0] ACIN,
    input  wire [3:0] ALUMODE,
    input  wire [17:0] B,
    input  wire [17:0] BCIN,
    input  wire [47:0] C,
    input  wire CARRYCASCIN,
    input  wire CARRYIN,
    input  wire [2:0] CARRYINSEL,
    input  wire CEA1,
    input  wire CEA2,
    input  wire CEAD,
    input  wire CEALUMODE,
    input  wire CEB1,
    input  wire CEB2,
    input  wire CEC,
    input  wire CECARRYIN,
    input  wire CECTRL,
    input  wire CED,
    input  wire CEINMODE,
    input  wire CEM,
    input  wire CEP,
    input  wire CLK,
    input  wire [26:0] D,
    input  wire [4:0] INMODE,
    input  wire MULTSIGNIN,
    input  wire [8:0] OPMODE,
    input  wire [47:0] PCIN,
    input  wire RSTA,
    input  wire RSTALLCARRYIN,
    input  wire RSTALUMODE,
    input  wire RSTB,
    input  wire RSTC,
    input  wire RSTCTRL,
    input  wire RSTD,
    input  wire RSTINMODE,
    input  wire RSTM,
    input  wire RSTP
);
    initial begin
        if (USE_SIMD != "FOUR12") $fatal(1, "model requires FOUR12");
        if (USE_MULT != "NONE") $fatal(1, "model requires USE_MULT=NONE");
        if (PREG != 1 || CREG != 0 || OPMODEREG != 0 || ALUMODEREG != 0)
            $fatal(1, "model requires the registered-PC configuration");
    end

    wire first_active = (OPMODE == 9'b000001100);
    wire first_pass = (OPMODE == 9'b000000000);
    wire signed [11:0] z0 = first_active ? 12'sd0 : $signed(PCIN[11:0]);
    wire signed [11:0] z1 = first_active ? 12'sd0 : $signed(PCIN[23:12]);
    wire signed [11:0] z2 = first_active ? 12'sd0 : $signed(PCIN[35:24]);
    wire signed [11:0] z3 = first_active ? 12'sd0 : $signed(PCIN[47:36]);
    wire signed [11:0] y0 = C[11:0];
    wire signed [11:0] y1 = C[23:12];
    wire signed [11:0] y2 = C[35:24];
    wire signed [11:0] y3 = C[47:36];
    wire subtract = (ALUMODE == 4'b0011);
    wire signed [11:0] n0 = subtract ? (z0 - y0) : (z0 + y0);
    wire signed [11:0] n1 = subtract ? (z1 - y1) : (z1 + y1);
    wire signed [11:0] n2 = subtract ? (z2 - y2) : (z2 + y2);
    wire signed [11:0] n3 = subtract ? (z3 - y3) : (z3 + y3);

    always @(posedge CLK) begin
        if (RSTP)
            P <= 48'd0;
        else if (CEP) begin
            if ((OPMODE != 9'b000011100) &&
                (OPMODE != 9'b000010000) &&
                !first_active && !first_pass)
                $fatal(1, "unexpected OPMODE %b", OPMODE);
            if ((ALUMODE != 4'b0000) && (ALUMODE != 4'b0011))
                $fatal(1, "unexpected ALUMODE %b", ALUMODE);
            if (first_pass)
                P <= 48'd0;
            else if (OPMODE == 9'b000010000)
                P <= PCIN;
            else
                P <= {n3, n2, n1, n0};
        end
    end

    assign PCOUT = P;
    assign ACOUT = 30'd0;
    assign BCOUT = 18'd0;
    assign CARRYCASCOUT = 1'b0;
    assign CARRYOUT = 4'd0;
    assign MULTSIGNOUT = 1'b0;
    assign OVERFLOW = 1'b0;
    assign PATTERNBDETECT = 1'b0;
    assign PATTERNDETECT = 1'b0;
    assign UNDERFLOW = 1'b0;
    assign XOROUT = 8'd0;
endmodule

`default_nettype wire
