// q1a8_rowblock_mc - multi-column rowblock: accumulate up to COLS_MAX activation
// columns against ROWS rows, time-multiplexing the SAME ROWS reducers.
//
// A weight beat (ROWS x one Q8 sub-block) is held by the caller while it sweeps
// columns: each cycle `valid_in` is high, `col_idx` selects which column's acts
// are presented, and the ROWS reducers fold that (weights, column) pair. The
// reducer is a fixed-latency pipeline that preserves input order, so `col_idx`
// shifted through a matching delay tells each output which column it belongs to.
// Per (row, col) accumulator: acc[row*COLS_MAX + col].
//
// Weights are thus read once per sub-block but MAC'd against every column, which
// is what makes prefill (C columns) cost one weight stream instead of C.
//
// Output: `results_flat` exposes acc[*][read_col] (ROWS fp32, lane-major), so the
// caller reads out one column at a time during emit. Consumers must wait for
// `done`, which is asserted after the final contribution has been written.

`default_nettype none

module q1a8_rowblock_mc #(
    parameter integer ROWS     = 8,
    parameter integer COLS_MAX = 8,
    parameter integer CW       = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX)
) (
    input  wire                       clk,
    input  wire                       rst_n,

    input  wire                       start,

    input  wire                       valid_in,
    input  wire                       last_in,
    // single_col: cols==1 (decode). The kernel issues sub-blocks back to back
    // (no issue_gap); this rowblock ping-pongs them across acc[*][0]/acc[*][1] so
    // each accumulator is revisited at >=2-cycle spacing (the forward's reach),
    // then sums the two partials at emit. acc[*][1] is otherwise idle at cols==1.
    input  wire                       single_col,
    input  wire [CW-1:0]              col_idx,
    input  wire [ROWS*32-1:0]         weight_bits_flat,
    input  wire [ROWS*16-1:0]         weight_scales_flat,
    input  wire [255:0]               acts_packed,
    input  wire [15:0]                act_scale,

    input  wire [CW-1:0]              read_col,
    output reg                        done,
    output wire [ROWS*32-1:0]         results_flat
);
    // q1a8_reducer plus one cycle for this rowblock to sample its registered
    // output contribution.
    localparam integer LAT = 4;

    wire [ROWS-1:0]    reducer_valid;
    wire [ROWS*32-1:0] contributions_flat;
    wire [ROWS*32-1:0] add_results_flat;
    reg  [ROWS*32-1:0] add_acc_q;
    reg  [ROWS*32-1:0] add_contribution_q;
    reg  [ROWS-1:0]    add_valid_q;
    reg  [COLS_MAX-1:0] add_col_oh_q;
    reg                add_last_q;
    reg  [ROWS*32-1:0] add_result_q;
    reg  [ROWS-1:0]    add_result_valid_q;
    reg  [COLS_MAX-1:0] add_result_col_oh_q;
    reg                add_result_last_q;
    reg  [COLS_MAX-1:0] col_seen_q;

    // 2-D accumulator: acc[row][col]. A 2-D array with the native 3-bit column
    // index synthesizes to a clean per-row COLS_MAX:1 mux, instead of a flat
    // 64-deep array with a wide dynamic index (which fans out across every bit
    // and fails timing).
    reg [31:0] acc [0:ROWS-1][0:COLS_MAX-1];

    // single_col ping-pong: toggles each issue so consecutive sub-blocks land in
    // alternating accumulators. The effective accumulator column is the parity at
    // cols==1, else the real matmul column. Everything downstream (the col_pipe
    // delay, the forward, the writeback) keys off eff_col, so the d>=2 spacing is
    // satisfied without an issue_gap.
    reg          issue_parity;
    wire [CW-1:0] eff_col = single_col ? {{(CW-1){1'b0}}, issue_parity} : col_idx;

    // col_idx delayed to align with reducer_valid / contribution.
    reg [CW-1:0] col_pipe [0:LAT-1];
    wire [CW-1:0] wcol = col_pipe[LAT-1];
    wire wcol_seen = col_seen_q[wcol];

    reg [COLS_MAX-1:0] wcol_oh;
    always @(*) begin
        wcol_oh = {COLS_MAX{1'b0}};
        wcol_oh[wcol] = 1'b1;
    end

    // last_in delayed to time `done` one reducer-latency after the final feed.
    reg [LAT:0] last_pipe;

    genvar row;
    genvar col_g;
    generate
        for (row = 0; row < ROWS; row = row + 1) begin : gen_lanes
            wire [31:0] add_acc = add_acc_q[row*32 +: 32];
            wire [31:0] add_contribution = add_contribution_q[row*32 +: 32];

            q1a8_reducer u_reducer (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in),
                .weight_bits(weight_bits_flat[row*32 +: 32]),
                .acts_packed(acts_packed),
                .weight_scale(weight_scales_flat[row*16 +: 16]),
                .act_scale(act_scale),
                .valid_out(reducer_valid[row]),
                .contribution(contributions_flat[row*32 +: 32])
            );

            fp32_add u_acc_add (
                .a(add_acc),
                .b(add_contribution),
                .out(add_results_flat[row*32 +: 32])
            );

            // At cols==1 the result is the sum of the two ping-pong partials;
            // otherwise it is the single accumulator for read_col. The add sits in
            // the emit readout path, not the accumulate recurrence, so it is off
            // the critical timing loop.
            wire [31:0] emit_sum;
            fp32_add u_emit_sum (
                .a(acc[row][0]),
                .b(acc[row][1]),
                .out(emit_sum)
            );
            assign results_flat[row*32 +: 32] = single_col ? emit_sum : acc[row][read_col];

            for (col_g = 0; col_g < COLS_MAX; col_g = col_g + 1) begin : gen_acc_cols
                always @(posedge clk) begin
                    if (add_result_valid_q[row] && add_result_col_oh_q[col_g]) begin
                        acc[row][col_g] <= add_result_q[row*32 +: 32];
                    end
                end
            end
        end
    endgenerate

    integer i;
    integer p;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (p = 0; p < LAT; p = p + 1) col_pipe[p] <= {CW{1'b0}};
            add_acc_q          <= {ROWS*32{1'b0}};
            add_contribution_q <= {ROWS*32{1'b0}};
            add_valid_q        <= {ROWS{1'b0}};
            add_col_oh_q       <= {COLS_MAX{1'b0}};
            add_last_q         <= 1'b0;
            add_result_q        <= {ROWS*32{1'b0}};
            add_result_valid_q  <= {ROWS{1'b0}};
            add_result_col_oh_q <= {COLS_MAX{1'b0}};
            add_result_last_q   <= 1'b0;
            col_seen_q         <= {COLS_MAX{1'b0}};
            last_pipe <= {(LAT+1){1'b0}};
            issue_parity <= 1'b0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;

            // Shift the (effective) column index alongside the reducer's valid
            // pipeline. eff_col folds in the single_col ping-pong parity.
            col_pipe[0] <= eff_col;
            for (p = 1; p < LAT; p = p + 1) col_pipe[p] <= col_pipe[p-1];

            // Flip the ping-pong parity on every issue so consecutive sub-blocks
            // alternate accumulators (only meaningful when single_col).
            if (valid_in) issue_parity <= ~issue_parity;

            last_pipe <= {last_pipe[LAT-1:0], (valid_in && last_in)};

            if (start) begin
                add_acc_q          <= {ROWS*32{1'b0}};
                add_contribution_q <= {ROWS*32{1'b0}};
                add_valid_q        <= {ROWS{1'b0}};
                add_col_oh_q       <= {COLS_MAX{1'b0}};
                add_last_q         <= 1'b0;
                add_result_q        <= {ROWS*32{1'b0}};
                add_result_valid_q  <= {ROWS{1'b0}};
                add_result_col_oh_q <= {COLS_MAX{1'b0}};
                add_result_last_q   <= 1'b0;
                col_seen_q         <= {COLS_MAX{1'b0}};
                last_pipe          <= {(LAT+1){1'b0}};
                issue_parity       <= 1'b0;
            end else begin
                if (add_result_last_q) done <= 1'b1;

                // Register the fp32 add output before it fans into the
                // accumulator bank. This cuts the routed add->acc path; the
                // capture below forwards this stage when a column is revisited
                // before the write lands.
                add_result_q        <= add_results_flat;
                add_result_valid_q  <= add_valid_q;
                add_result_col_oh_q <= add_col_oh_q;
                add_result_last_q   <= add_last_q;

                // Capture the next add inputs. The kernel spaces repeated
                // accesses to a column far enough apart that the only bypass
                // needed here is from the registered writeback stage.
                add_valid_q <= reducer_valid;
                add_col_oh_q <= wcol_oh;
                add_last_q  <= reducer_valid[0] && last_pipe[LAT-1];
                if (reducer_valid[0]) col_seen_q[wcol] <= 1'b1;
                for (i = 0; i < ROWS; i = i + 1) begin
                    if (reducer_valid[i]) begin
                        add_acc_q[i*32 +: 32] <=
                            (add_result_valid_q[i] && add_result_col_oh_q[wcol])
                                ? add_result_q[i*32 +: 32]
                                : (wcol_seen ? acc[i][wcol] : 32'd0);
                        add_contribution_q[i*32 +: 32] <= contributions_flat[i*32 +: 32];
                    end
                end
            end
        end
    end

endmodule
