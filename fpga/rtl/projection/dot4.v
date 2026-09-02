// Four-token, sixteen-row ternary dot product for the projection engine.
//
// One input record contains 32 signed Q8 activation elements for each of four
// physical token lanes and 32 shared Q1/Q2 selectors for each output row.  The
// result is 64 exact signed 14-bit dot products.  The dot is split into
// 11/11/10-term groups because a complete 32-term Q8 sum does not fit in one
// DSP48E2 FOUR12 lane.  Every group is a registered PC cascade, so records may
// be accepted on consecutive clocks; the shorter final group is realigned
// before the registered three-way merge.
//
// Define PENZAI_XILINX_DSP48E2 for Vivado synthesis.  Without it, the leaf below
// implements the same registered, lane-isolated FOUR12 operation in portable
// RTL.  Verilator and Yosys therefore test real arithmetic and pipeline timing
// without needing a UNISIM installation.

`default_nettype none

module delay #(
    parameter integer WIDTH = 1,
    parameter integer DEPTH = 0
) (
    input  wire             clk,
    input  wire [WIDTH-1:0] din,
    output wire [WIDTH-1:0] dout
);
    generate
        if (DEPTH == 0) begin : g_passthrough
            assign dout = din;
        end else begin : g_delay
            reg [WIDTH-1:0] pipe [0:DEPTH-1];
            integer i;
            always @(posedge clk) begin
                pipe[0] <= din;
                for (i = 1; i < DEPTH; i = i + 1)
                    pipe[i] <= pipe[i-1];
            end
            assign dout = pipe[DEPTH-1];
        end
    endgenerate
endmodule

// Preserve the requested delay while giving each four-row DSP cluster a local
// final register.  The shared prefix keeps the replication cost to one stage.
module delay_cluster4 #(
    parameter integer WIDTH = 1,
    parameter integer DEPTH = 0
) (
    input  wire               clk,
    input  wire [WIDTH-1:0]   din,
    output wire [4*WIDTH-1:0] dout
);
    generate
        if (DEPTH == 0) begin : g_passthrough
            assign dout = {4{din}};
        end else begin : g_clustered
            wire [WIDTH-1:0] trunk;
            (* keep = "true" *) reg [WIDTH-1:0] leaf0;
            (* keep = "true" *) reg [WIDTH-1:0] leaf1;
            (* keep = "true" *) reg [WIDTH-1:0] leaf2;
            (* keep = "true" *) reg [WIDTH-1:0] leaf3;

             delay #(.WIDTH(WIDTH), .DEPTH(DEPTH-1)) u_trunk (
                .clk(clk),
                .din(din),
                .dout(trunk)
            );
            always @(posedge clk) begin
                leaf0 <= trunk;
                leaf1 <= trunk;
                leaf2 <= trunk;
                leaf3 <= trunk;
            end
            assign dout = {leaf3, leaf2, leaf1, leaf0};
        end
    endgenerate
endmodule

// One registered DSP ALU stage.  Each 12-bit lane is an independent signed
// two's-complement add/subtract, matching DSP48E2 USE_SIMD="FOUR12".
module dsp48e2_addsub4 #(
    parameter integer FIRST_STAGE = 0
) (
    input  wire        clk,
    input  wire [47:0] pcin,
    input  wire [47:0] term,
    input  wire        enable,
    input  wire        subtract,
    output wire [47:0] p,
    output wire [47:0] pcout
);
`ifdef PENZAI_XILINX_DSP48E2
    // Later stages use W=zero, Z=PCIN, Y=C, X=zero.  The first stage must use
    // Z=zero, Y=C: Vivado rejects a PCIN-selecting OPMODE when there is no
    // upstream DSP even if the RTL PCIN net is tied to zero.  ALUMODE 0000
    // computes Z+Y; 0011 computes Z-Y.  USE_SIMD prevents carries from
    // crossing the four 12-bit token lanes.
    wire [8:0] active_opmode = FIRST_STAGE ?
        9'b000001100 : 9'b000011100;
    wire [8:0] pass_opmode = FIRST_STAGE ?
        9'b000000000 : 9'b000010000;
    DSP48E2 #(
        .ACASCREG(0),
        .ADREG(0),
        .ALUMODEREG(0),
        .AMULTSEL("A"),
        .AREG(0),
        .A_INPUT("DIRECT"),
        .BCASCREG(0),
        .BMULTSEL("B"),
        .BREG(0),
        .B_INPUT("DIRECT"),
        .CARRYINREG(0),
        .CARRYINSELREG(0),
        .CREG(0),
        .DREG(0),
        .INMODEREG(0),
        .MREG(0),
        .OPMODEREG(0),
        .PREADDINSEL("A"),
        .PREG(1),
        .USE_MULT("NONE"),
        .USE_PATTERN_DETECT("NO_PATDET"),
        .USE_SIMD("FOUR12"),
        .USE_WIDEXOR("FALSE")
    ) u_dsp (
        .ACOUT(),
        .BCOUT(),
        .CARRYCASCOUT(),
        .CARRYOUT(),
        .MULTSIGNOUT(),
        .OVERFLOW(),
        .P(p),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT(pcout),
        .UNDERFLOW(),
        .XOROUT(),
        .A(30'd0),
        .ACIN(30'd0),
        .ALUMODE(subtract ? 4'b0011 : 4'b0000),
        .B(18'd0),
        .BCIN(18'd0),
        .C(term),
        .CARRYCASCIN(1'b0),
        .CARRYIN(1'b0),
        .CARRYINSEL(3'b000),
        .CEA1(1'b0),
        .CEA2(1'b0),
        .CEAD(1'b0),
        .CEALUMODE(1'b0),
        .CEB1(1'b0),
        .CEB2(1'b0),
        .CEC(1'b0),
        .CECARRYIN(1'b0),
        .CECTRL(1'b0),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),
        .CEP(1'b1),
        .CLK(clk),
        .D(27'd0),
        .INMODE(5'b00000),
        .MULTSIGNIN(1'b0),
        .OPMODE(enable ? active_opmode : pass_opmode),
        .PCIN(pcin),
        .RSTA(1'b0),
        .RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),
        .RSTB(1'b0),
        .RSTC(1'b0),
        .RSTCTRL(1'b0),
        .RSTD(1'b0),
        .RSTINMODE(1'b0),
        .RSTM(1'b0),
        .RSTP(1'b0)
    );
`else
    wire signed [11:0] z0 = pcin[11:0];
    wire signed [11:0] z1 = pcin[23:12];
    wire signed [11:0] z2 = pcin[35:24];
    wire signed [11:0] z3 = pcin[47:36];
    wire signed [11:0] y0 = term[11:0];
    wire signed [11:0] y1 = term[23:12];
    wire signed [11:0] y2 = term[35:24];
    wire signed [11:0] y3 = term[47:36];
    wire signed [11:0] n0 = subtract ? (z0 - y0) : (z0 + y0);
    wire signed [11:0] n1 = subtract ? (z1 - y1) : (z1 + y1);
    wire signed [11:0] n2 = subtract ? (z2 - y2) : (z2 + y2);
    wire signed [11:0] n3 = subtract ? (z3 - y3) : (z3 + y3);
    reg [47:0] p_q;
    always @(posedge clk) begin
        if (enable)
            p_q <= {n3, n2, n1, n0};
        else
            p_q <= pcin;
    end
    assign p = p_q;
    assign pcout = p_q;
`endif
endmodule

module dot4 #(
    parameter integer ROWS  = 16,
    parameter integer LANES = 4,
    parameter integer SUM_W = 14
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         in_valid,
    input  wire [ROWS*32-1:0]           weight_sign,
    input  wire [ROWS*32-1:0]           weight_nonzero,
    input  wire [LANES*256-1:0]         acts_flat,
    output wire                         out_valid,
    output reg  [ROWS*LANES*SUM_W-1:0] sums_flat
);
    localparam integer GROUPS        = 3;
    localparam integer SIMD_W        = 12;
    localparam integer PACK_W        = LANES * SIMD_W;
    localparam integer GROUP0_BASE   = 0;
    localparam integer GROUP1_BASE   = 11;
    localparam integer GROUP2_BASE   = 22;
    localparam integer GROUP0_LEN    = 11;
    localparam integer GROUP1_LEN    = 11;
    localparam integer GROUP2_LEN    = 10;
    localparam integer MAX_GROUP_LEN = 11;
    localparam integer DOT_LATENCY   = MAX_GROUP_LEN + 1;

    // The architecture is intentionally fixed at four FOUR12 lanes.  Keep the
    // parameters on the interface for hierarchy accounting, but fail loudly if
    // a caller attempts to turn this into a different datapath accidentally.
    initial begin
        if (ROWS != 16)  $error(" dot4 requires ROWS=16");
        if (LANES != 4)  $error(" dot4 requires LANES=4");
        if (SUM_W != 14) $error(" dot4 requires SUM_W=14");
    end

    wire [GROUPS*ROWS*LANES*SIMD_W-1:0] group_sums;

    genvar group_idx;
    generate
        for (group_idx = 0; group_idx < GROUPS; group_idx = group_idx + 1) begin : g_group
            localparam integer BASE = (group_idx == 0) ? GROUP0_BASE :
                                      (group_idx == 1) ? GROUP1_BASE : GROUP2_BASE;
            localparam integer LEN  = (group_idx == 0) ? GROUP0_LEN :
                                      (group_idx == 1) ? GROUP1_LEN : GROUP2_LEN;
            localparam integer ROW_CLUSTERS = 4;
            localparam integer ROWS_PER_CLUSTER = ROWS / ROW_CLUSTERS;
            wire [ROW_CLUSTERS*LEN*PACK_W-1:0] acts_skewed;

            genvar stage_idx;
            for (stage_idx = 0; stage_idx < LEN; stage_idx = stage_idx + 1) begin : g_act_skew
                localparam integer ELEMENT = BASE + stage_idx;
                wire [PACK_W-1:0] packed_act;
                genvar lane_idx;
                for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin : g_pack
                    wire [7:0] act = acts_flat[lane_idx*256 + ELEMENT*8 +: 8];
                    assign packed_act[lane_idx*SIMD_W +: SIMD_W] =
                        {{(SIMD_W-8){act[7]}}, act};
                end
                 delay_cluster4 #(
                    .WIDTH(PACK_W),
                    .DEPTH(stage_idx)
                ) u_act_delay (
                    .clk(clk),
                    .din(packed_act),
                    .dout(acts_skewed[
                        stage_idx*ROW_CLUSTERS*PACK_W +:
                        ROW_CLUSTERS*PACK_W])
                );
            end

            genvar row_idx;
            for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin : g_row
                localparam integer ROW_CLUSTER = row_idx / ROWS_PER_CLUSTER;
                wire [LEN*PACK_W-1:0] p_chain;
                wire [LEN*PACK_W-1:0] p_fabric;
                for (stage_idx = 0; stage_idx < LEN; stage_idx = stage_idx + 1) begin : g_stage
                    localparam integer ELEMENT = BASE + stage_idx;
                    wire sign_d;
                    wire nonzero_d;
                    wire [PACK_W-1:0] pcin = (stage_idx == 0) ? {PACK_W{1'b0}} :
                        p_chain[(stage_idx-1)*PACK_W +: PACK_W];
                    wire [PACK_W-1:0] term =
                        acts_skewed[(stage_idx*ROW_CLUSTERS + ROW_CLUSTER)*
                                    PACK_W +: PACK_W];

                     delay #(.WIDTH(1), .DEPTH(stage_idx)) u_sign_delay (
                        .clk(clk),
                        .din(weight_sign[row_idx*32 + ELEMENT]),
                        .dout(sign_d)
                    );
                     delay #(.WIDTH(1), .DEPTH(stage_idx)) u_nonzero_delay (
                        .clk(clk),
                        .din(weight_nonzero[row_idx*32 + ELEMENT]),
                        .dout(nonzero_d)
                    );
                     dsp48e2_addsub4 #(
                        .FIRST_STAGE(stage_idx == 0)
                    ) u_stage (
                        .clk(clk),
                        .pcin(pcin),
                        .term(term),
                        .enable(nonzero_d),
                        .subtract(nonzero_d && !sign_d),
                        .p(p_fabric[stage_idx*PACK_W +: PACK_W]),
                        .pcout(p_chain[stage_idx*PACK_W +: PACK_W])
                    );
                end
                assign group_sums[(group_idx*ROWS + row_idx)*PACK_W +: PACK_W] =
                    p_fabric[(LEN-1)*PACK_W +: PACK_W];
            end
        end
    endgenerate

    // Group 2 is one DSP stage shorter than groups 0 and 1.
    wire [ROWS*PACK_W-1:0] group2_aligned;
     delay #(.WIDTH(ROWS*PACK_W), .DEPTH(1)) u_group2_align (
        .clk(clk),
        .din(group_sums[(2*ROWS)*PACK_W +: ROWS*PACK_W]),
        .dout(group2_aligned)
    );

    wire [ROWS*LANES*SUM_W-1:0] merged_sums;
    genvar merge_row;
    generate
        for (merge_row = 0; merge_row < ROWS; merge_row = merge_row + 1) begin : g_merge_row
            genvar merge_lane;
            for (merge_lane = 0; merge_lane < LANES; merge_lane = merge_lane + 1) begin : g_merge_lane
                wire signed [SIMD_W-1:0] partial0 =
                    group_sums[(0*ROWS + merge_row)*PACK_W +
                               merge_lane*SIMD_W +: SIMD_W];
                wire signed [SIMD_W-1:0] partial1 =
                    group_sums[(1*ROWS + merge_row)*PACK_W +
                               merge_lane*SIMD_W +: SIMD_W];
                wire signed [SIMD_W-1:0] partial2 =
                    group2_aligned[merge_row*PACK_W +
                                   merge_lane*SIMD_W +: SIMD_W];
                assign merged_sums[(merge_row*LANES + merge_lane)*SUM_W +: SUM_W] =
                    {{(SUM_W-SIMD_W){partial0[SIMD_W-1]}}, partial0} +
                    {{(SUM_W-SIMD_W){partial1[SIMD_W-1]}}, partial1} +
                    {{(SUM_W-SIMD_W){partial2[SIMD_W-1]}}, partial2};
            end
        end
    endgenerate
    always @(posedge clk) begin
        sums_flat <= merged_sums;
    end

    reg [DOT_LATENCY-1:0] valid_pipe;
    always @(posedge clk) begin
        if (!rst_n)
            valid_pipe <= {DOT_LATENCY{1'b0}};
        else
            valid_pipe <= {valid_pipe[DOT_LATENCY-2:0], in_valid};
    end
    assign out_valid = valid_pipe[DOT_LATENCY-1];
endmodule

`default_nettype wire
