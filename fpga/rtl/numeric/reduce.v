// reduce - feed-forward sum of N lanes via a pipelined fadd tree.
//
//   sum = Σ_{i<N} in[i]   over log2(N) levels of fadd, NO recurrence.
//
// Parameterized N (power of 2) and MANT_W; composes numeric/fadd. The pairing order is
// fixed as (0,1)(2,3)…, then pairs of those (fp add is non-associative, so order is part
// of the contract). Latency = log2(N) · FADD_LATENCY. Unused lanes are the caller's job to
// zero-pad (fp `+0` is exact). MODE is `tree` for now (iterative/dsp_cascade later,
// behind the same param — plan-fpga-7.md numeric/reduce).

`default_nettype none

module reduce #(
    parameter integer MANT_W = 23,
    parameter integer N      = 16          // power of two, >= 2
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     valid_in,
    input  wire [N*(MANT_W+9)-1:0]  in,     // N lanes, lane i = in[i*W +: W]
    output wire                     valid_out,
    output wire [MANT_W+8:0]        sum
);
    localparam integer W      = MANT_W + 9;
    localparam integer LEVELS = $clog2(N);

    // Heap-shaped node array: node[L][i] is lane i at level L (level 0 = inputs).
    // Over-allocated to N per level for a rectangular declaration; only the low
    // N>>L entries of each level are driven/used.
    wire [W-1:0] node [0:LEVELS][0:N-1];
    wire         nval [0:LEVELS][0:N-1];

    genvar L, i;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_in
            assign node[0][i] = in[i*W +: W];
            assign nval[0][i] = valid_in;
        end
        for (L = 0; L < LEVELS; L = L + 1) begin : g_lvl
            for (i = 0; i < (N >> (L + 1)); i = i + 1) begin : g_node
                fadd #(.MANT_W(MANT_W)) u (
                    .clk(clk), .rst_n(rst_n),
                    .valid_in(nval[L][2*i]),
                    .a(node[L][2*i]), .b(node[L][2*i+1]),
                    .valid_out(nval[L+1][i]), .out(node[L+1][i])
                );
            end
        end
    endgenerate

    assign sum       = node[LEVELS][0];
    assign valid_out = nval[LEVELS][0];
endmodule
