// gemm_kernel - the fixed-point matmul kernel FSM (plan-fpga-7.md §gemm.v, increment 3).
//
// Owns the AXIS weight/act feed, BRAM-backed activation storage, and COLS_MAX column
// sweep (one weight beat held across `num_cols` columns —
// the prefill weight-reuse axis). What it DROPS is the recurrence-hiding machinery the
// fp32 accumulate needed: no ACCUM_DEPTH pool, no issue_gap, no single_col round-robin, no
// done_pipe. The fixed-point accumulate is single-cycle, so sub-blocks issue back-to-back
// into one accumulator and `done` is just a fixed drain after the last issue.
//
// Datapath = gemm_rowblock (the COLS_MAX-banked fixed-point MAC) + gemm_emit (the per-output
// fixed→fp32 normalize, PIPELINED — fails f300 combinationally). The emit pipeline is filled
// in a PRECOMPUTE phase (free-running, no backpressure) into a result buffer, which ST_EMIT
// then streams 2 fp32/beat with AXIS backpressure. `emin` is stable for a run. Output
// stream layout is per rowblock, per column, ROWS/2 beats of 2 fp32 (lane-major).
//
// Gated vs matmul_ref.windowedFixedOutput (C=1 decode and C>1 prefill) — bit-exact, since
// the oracle emits via the truncating emitTrunc that models gemm_emit.

`default_nettype none

module gemm_kernel #(
    parameter integer ROWS          = 16,
    parameter integer COLS_MAX      = 8,
    parameter integer MAX_SUB_INDEX = 64,
    parameter integer ACC_W         = 104  // full-f16-range fixed window (matmul_ref.ACC_W_BITS)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [15:0]           num_rowblocks,
    input  wire [31:0]           num_rows,
    input  wire [15:0]           num_cols,
    input  wire [1:0]            weight_fmt,
    input  wire signed [7:0]     emin,            // global window floor (calibration, set once)
    output reg                   kernel_done,
    output wire                  busy,

    input  wire [ROWS*32-1:0]    s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    input  wire [63:0]           s_axis_acts_tdata,
    input  wire                  s_axis_acts_tvalid,
    output wire                  s_axis_acts_tready,

    output wire [63:0]           m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast,
    output wire [7:0]            m_axis_tkeep,

    output wire [3:0]            dbg_state
);
    localparam integer CW = (COLS_MAX <= 1) ? 1 : $clog2(COLS_MAX);
    localparam integer ROWW = (ROWS <= 1) ? 1 : $clog2(ROWS);

    localparam [3:0] ST_IDLE        = 4'd0;
    localparam [3:0] ST_LOAD_ACTS   = 4'd1;
    localparam [3:0] ST_LOAD_ASCALE = 4'd2;
    localparam [3:0] ST_WSCALE      = 4'd3;
    localparam [3:0] ST_WISSUE      = 4'd4;
    localparam [3:0] ST_WAIT_DONE   = 4'd5;
    localparam [3:0] ST_PRECOMPUTE  = 4'd6;
    localparam [3:0] ST_EMIT        = 4'd7;
    localparam [3:0] ST_FINISH      = 4'd8;
    localparam [3:0] ST_TLOAD       = 4'd9;
    localparam [3:0] ST_TCODE0      = 4'd10;
    localparam [3:0] ST_TCODE1      = 4'd11;

    // gemm_rowblock latency input→acc = FE_LAT(4) + fma(5) = 9. Wait this (plus the issue
    // register + margin) after the last issue before reading acc; the cosim drain gates it.
    localparam integer ROWBLOCK_LAT = 9;
    localparam integer WAIT_CYCLES  = ROWBLOCK_LAT + 3;
    localparam integer WW           = $clog2(WAIT_CYCLES + 1);

    // Result emit: 64-bit stream (2 fp32 lane-major per beat), ROWS/2 beats per (rowblock,col).
    localparam integer EMIT_BEATS = ROWS / 2;
    localparam integer EBW        = (EMIT_BEATS <= 1) ? 1 : $clog2(EMIT_BEATS);
    localparam integer EMIT_LAST  = EMIT_BEATS - 1;
    // Result buffer: one fp32-pair per (column, emit beat), filled by the emit pipeline in
    // PRECOMPUTE, streamed in EMIT.
    localparam integer BUF_DEPTH = COLS_MAX * EMIT_BEATS;
    localparam integer BUFAW     = $clog2(BUF_DEPTH);      // buffer address width
    localparam integer WRW       = $clog2(BUF_DEPTH + 1);  // write count (holds BUF_DEPTH)

    reg [3:0]  state;
    reg        busy_q;
    reg [15:0] run_num_q1_blocks;
    reg [15:0] run_num_rowblocks;
    reg [31:0] run_num_rows;
    reg [15:0] run_num_cols;
    reg signed [7:0] run_emin;
    reg [1:0] run_weight_fmt;
    reg [15:0] rowblock_remaining;
    reg [13:0] q1_idx;
    reg [1:0]  sub;
    reg [15:0] col;          // column being issued
    reg [WW-1:0] wait_cnt;
    reg [EBW-1:0] emit_beat;
    reg [15:0] emit_col;

    // precompute walk + result buffer.
    reg [15:0]      pc_col;
    reg [EBW-1:0]   pc_beat;
    reg [WRW-1:0]   wr_idx;
    reg [63:0]      result_buf [0:BUF_DEPTH-1];

    reg [ROWS*16-1:0]  weight_scales_q;
    wire [ROWS*16-1:0] weight_scales_from_slots;
    wire [ROWS*16-1:0] tern_scales_hi_from_slots;

    // Acts BRAM: COLS_MAX columns, each MAX_SUB_INDEX subblock entries.
    localparam integer ACT_DEPTH = MAX_SUB_INDEX * COLS_MAX;
    reg [255:0] acts_mem      [0:ACT_DEPTH-1];
    reg [15:0]  act_scale_mem [0:ACT_DEPTH-1];
    localparam integer SUB_W = $clog2(MAX_SUB_INDEX);

    wire [SUB_W-1:0]  issue_sub  = {q1_idx[SUB_W-3:0], sub};
    wire [31:0]       issue_addr = issue_sub * COLS_MAX + {16'd0, col};
    wire [255:0] acts_packed_w = acts_mem[issue_addr];
    wire [15:0]  act_scale_w   = act_scale_mem[issue_addr];

    reg                 issue_valid_q;
    reg                 issue_clear_q;
    reg [CW-1:0]        issue_col_q;
    reg [ROWS*32-1:0]   issue_weight_bits_q;
    reg [ROWS*32-1:0]   issue_weight_nonzero_q;
    reg [ROWS*16-1:0]   issue_weight_scales_q;
    reg [255:0]         issue_acts_packed_q;
    reg [15:0]          issue_act_scale_q;

    // Ternary blocks stream as one dual-scale beat, then two issue-ordered
    // two-bit code beats per 32-weight sub-block.
    reg [ROWS*32-1:0] tern_code_first_q;
    reg [ROWS*32-1:0] tern_code_second_q;
    reg [ROWS*16-1:0] tern_scales_lo;
    reg [ROWS*16-1:0] tern_scales_hi;

    // Acts load scratch.
    reg [255:0] acts_load_accum;
    reg [2:0]   acts_load_beat;
    reg [13:0]  acts_load_q1;
    reg [1:0]   acts_load_sub;
    reg [15:0]  acts_load_col;
    wire [SUB_W-1:0] acts_load_sub_idx = {acts_load_q1[SUB_W-3:0], acts_load_sub};
    wire [31:0] acts_load_addr = acts_load_sub_idx * COLS_MAX + {16'd0, acts_load_col};

    wire [ROWS*ACC_W-1:0] rowblock_acc;

    wire [15:0] q1_idx_wide = {2'b00, q1_idx};
    wire last_q1   = (q1_idx_wide == run_num_q1_blocks - 16'd1);
    wire last_col  = (col == run_num_cols - 16'd1);
    wire is_first_sub = (q1_idx == 14'd0) && (sub == 2'd0); // first sub-block of the rowblock

    assign busy = busy_q;
    assign dbg_state = state;
    // Binary weights hold each issue beat across the column sweep. Ternary
    // weights are captured as two code beats before that sweep.
    assign s_axis_tready =
        busy_q && (((run_weight_fmt == 2'd2) &&
                   ((state == ST_TLOAD) || (state == ST_TCODE0) || (state == ST_TCODE1))) ||
                   ((run_weight_fmt == 2'd1) &&
                    ((state == ST_WSCALE) || ((state == ST_WISSUE) && last_col))));
    assign s_axis_acts_tready =
        busy_q && ((state == ST_LOAD_ACTS) || (state == ST_LOAD_ASCALE));

    wire acts_beat_accept = s_axis_acts_tvalid && s_axis_acts_tready;
    wire start_pulse      = start_kernel && !busy_q;
    wire weight_issue_valid = (run_weight_fmt == 2'd2) ? 1'b1 : s_axis_tvalid;

    wire [ROWS*32-1:0] tern_decoded_sign;
    wire [ROWS*32-1:0] tern_decoded_nonzero;
    genvar tern_row;
    generate
        for (tern_row = 0; tern_row < ROWS; tern_row = tern_row + 1) begin : gen_tern_sub
            gemm_ternary_select32 u_ternary_select (
                .codes({tern_code_second_q[tern_row*32 +: 32],
                        tern_code_first_q[tern_row*32 +: 32]}),
                .sign(tern_decoded_sign[tern_row*32 +: 32]),
                .nonzero(tern_decoded_nonzero[tern_row*32 +: 32])
            );
        end
    endgenerate

    genvar scale_lane;
    generate
        for (scale_lane = 0; scale_lane < ROWS; scale_lane = scale_lane + 1) begin : gen_scale_slots
            assign weight_scales_from_slots[scale_lane*16 +: 16] =
                s_axis_tdata[scale_lane*32 +: 16];
            assign tern_scales_hi_from_slots[scale_lane*16 +: 16] =
                s_axis_tdata[scale_lane*32 + 16 +: 16];
        end
    endgenerate

    gemm_rowblock #(.ROWS(ROWS), .COLS_MAX(COLS_MAX), .ACC_W(ACC_W)) u_rowblock (
        .clk(clk),
        .rst_n(rst_n),
        .clear(issue_clear_q),
        .valid_in(issue_valid_q),
        .col_idx(issue_col_q),
        .emin(run_emin),
        .weight_bits_flat(issue_weight_bits_q),
        .weight_nonzero_flat(issue_weight_nonzero_q),
        .weight_scales_flat(issue_weight_scales_q),
        .acts_packed(issue_acts_packed_q),
        .act_scale(issue_act_scale_q),
        .read_col(pc_col[CW-1:0]),
        .acc_flat(rowblock_acc)
    );

    // PRECOMPUTE readout: walk (pc_col, pc_beat) over the accumulator bank → gemm_emit, filling
    // result_buf. The acc[ROWS][COLS_MAX] bank is ~9k scattered FFs, so the read_col + emit_beat
    // gather muxes route long — the full-build worst path was acc → read_col mux → emit_beat mux
    // → gemm_emit in ONE cycle (61% routing, missed f250). PIPELINE the readout in two stages so
    // the placer can break those gathers: R1 registers the read_col-muxed bank (rowblock_acc =
    // acc[*][pc_col]) + the beat select + valid; R2 registers the beat-muxed lane pair + valid;
    // then gemm_emit. Off the throughput path (one-time fill), so +2 cycles are free; gemm_emit's
    // valid_in→valid_out and the wr_idx collection preserve order regardless of the latency.
    wire pc_presenting = (state == ST_PRECOMPUTE) && (pc_col < run_num_cols);

    reg [ROWS*ACC_W-1:0] rb_acc_q;    // R1: read_col-muxed bank (gemm_rowblock acc[*][pc_col])
    reg [EBW-1:0]        pc_beat_d, pc_beat_q;
    reg                  pc_vld_a, pc_vld_q;
    wire [2*ACC_W-1:0]   acc_pair_c = rb_acc_q[pc_beat_q*(2*ACC_W) +: 2*ACC_W];
    reg [2*ACC_W-1:0]    acc_pair_q;  // R2: beat-muxed lane pair
    reg                  pc_vld_q2;
    // gemm_rowblock now resolves accS+accC through one register (the readout-resolve pipeline),
    // so rowblock_acc lags read_col (=pc_col) by one extra cycle. Match it: pc_beat is delayed 2
    // (pc_beat_d→pc_beat_q) to align with rb_acc_q's column, and the valid carries one extra stage
    // (pc_vld_a). The wr_idx collector is latency-tolerant, so nothing else moves.
    always @(posedge clk) begin
        if (!rst_n) begin pc_vld_a <= 1'b0; pc_vld_q <= 1'b0; pc_vld_q2 <= 1'b0; end
        else        begin pc_vld_a <= pc_presenting; pc_vld_q <= pc_vld_a; pc_vld_q2 <= pc_vld_q; end
        rb_acc_q   <= rowblock_acc;
        pc_beat_d  <= pc_beat;
        pc_beat_q  <= pc_beat_d;
        acc_pair_q <= acc_pair_c;
    end

    wire        emit_vo, emit_vo_hi;
    wire [31:0] emit_f32_lo, emit_f32_hi;
    gemm_emit #(.ACC_W(ACC_W), .EXP_W(8)) u_emit_lo (
        .clk(clk), .rst_n(rst_n), .valid_in(pc_vld_q2),
        .acc($signed(acc_pair_q[ACC_W-1:0])), .emin(run_emin),
        .valid_out(emit_vo), .f32(emit_f32_lo)
    );
    gemm_emit #(.ACC_W(ACC_W), .EXP_W(8)) u_emit_hi (
        .clk(clk), .rst_n(rst_n), .valid_in(pc_vld_q2),
        .acc($signed(acc_pair_q[2*ACC_W-1:ACC_W])), .emin(run_emin),
        .valid_out(emit_vo_hi), .f32(emit_f32_hi)   // same timing as emit_vo
    );

    wire [15:0] n_total = run_num_cols * EMIT_BEATS[15:0];

    // EMIT: stream the buffer 2 fp32/beat, lane-major, with AXIS backpressure. NUM_ROWS
    // shortens only the final rowblock; zero or an exact multiple of ROWS preserves the
    // historical full-rowblock stream. Invalid high-lane bytes never reach S2MM.
    localparam [ROWW:0] ROWS_VALUE = ROWS[ROWW:0];
    wire [ROWW-1:0] final_row_remainder = run_num_rows[ROWW-1:0];
    wire [ROWW:0] final_row_count = (final_row_remainder == {ROWW{1'b0}}) ?
        ROWS_VALUE : {1'b0, final_row_remainder};
    wire [ROWW:0] final_emit_index = (final_row_count - 1'b1) >> 1;
    wire [EBW-1:0] final_emit_last = final_emit_index[EBW-1:0];
    wire [EBW-1:0] active_emit_last = (rowblock_remaining == 16'd1) ?
        final_emit_last : EMIT_LAST[EBW-1:0];

    // The buffer index is {col, beat} since EMIT_BEATS = 2^EBW (matches the
    // precompute walk order).
    wire [BUFAW-1:0] rd_idx = {emit_col[CW-1:0], emit_beat};
    assign m_axis_tdata  = result_buf[rd_idx];
    assign m_axis_tvalid = (state == ST_EMIT);
    assign m_axis_tlast  = (state == ST_EMIT) && (emit_beat == active_emit_last) &&
                           (emit_col == run_num_cols - 16'd1) && (rowblock_remaining == 16'd1);
    assign m_axis_tkeep  = ((state == ST_EMIT) && (rowblock_remaining == 16'd1) &&
                            (emit_beat == active_emit_last) && run_num_rows[0]) ? 8'h0F : 8'hFF;

    always @(posedge clk) begin
        if (!rst_n) begin
            state               <= ST_IDLE;
            busy_q              <= 1'b0;
            run_num_q1_blocks   <= 16'd0;
            run_num_rowblocks   <= 16'd0;
            run_num_rows        <= 32'd0;
            run_num_cols        <= 16'd0;
            run_emin            <= 8'sd0;
            run_weight_fmt      <= 2'd1;
            rowblock_remaining  <= 16'd0;
            q1_idx              <= 14'd0;
            sub                 <= 2'd0;
            col                 <= 16'd0;
            wait_cnt            <= {WW{1'b0}};
            emit_beat           <= {EBW{1'b0}};
            emit_col            <= 16'd0;
            pc_col              <= 16'd0;
            pc_beat             <= {EBW{1'b0}};
            wr_idx              <= {WRW{1'b0}};
            weight_scales_q     <= {ROWS*16{1'b0}};
            issue_valid_q       <= 1'b0;
            issue_clear_q       <= 1'b0;
            issue_col_q         <= {CW{1'b0}};
            issue_weight_bits_q <= {ROWS*32{1'b0}};
            issue_weight_nonzero_q <= {ROWS*32{1'b0}};
            issue_weight_scales_q <= {ROWS*16{1'b0}};
            issue_acts_packed_q <= 256'd0;
            issue_act_scale_q   <= 16'd0;
            kernel_done         <= 1'b0;
            acts_load_accum     <= 256'd0;
            acts_load_beat      <= 3'd0;
            acts_load_q1        <= 14'd0;
            acts_load_sub       <= 2'd0;
            acts_load_col       <= 16'd0;
            tern_code_first_q   <= {ROWS*32{1'b0}};
            tern_code_second_q  <= {ROWS*32{1'b0}};
            tern_scales_lo      <= {ROWS*16{1'b0}};
            tern_scales_hi      <= {ROWS*16{1'b0}};
        end else begin
            issue_valid_q <= 1'b0;
            issue_clear_q <= 1'b0;
            kernel_done   <= 1'b0;

            // Collect emit-pipeline outputs into the result buffer (active in PRECOMPUTE).
            // wr_idx counts in feed order = {col, beat}, matching rd_idx; slice to the
            // address width (wr_idx is one bit wider to hold the BUF_DEPTH terminal count).
            if (emit_vo) begin
                result_buf[wr_idx[BUFAW-1:0]] <= {emit_f32_hi, emit_f32_lo};
                wr_idx <= wr_idx + 1'b1;
            end

            if (start_pulse) begin
                busy_q         <= 1'b1;
                run_num_q1_blocks <= num_q1_blocks;
                run_num_rowblocks <= num_rowblocks;
                run_num_rows      <= num_rows;
                run_num_cols      <= num_cols;
                run_emin          <= emin;
                run_weight_fmt    <= weight_fmt;
                acts_load_beat <= 3'd0;
                acts_load_q1   <= 14'd0;
                acts_load_sub  <= 2'd0;
                acts_load_col  <= 16'd0;
                state          <= ST_LOAD_ACTS;
            end else if (busy_q) begin
                case (state)
                    ST_LOAD_ACTS: begin
                        if (acts_beat_accept) begin
                            case (acts_load_beat)
                                3'd0: acts_load_accum[ 63:  0] <= s_axis_acts_tdata;
                                3'd1: acts_load_accum[127: 64] <= s_axis_acts_tdata;
                                3'd2: acts_load_accum[191:128] <= s_axis_acts_tdata;
                                3'd3: acts_load_accum[255:192] <= s_axis_acts_tdata;
                                default: ;
                            endcase
                            if (acts_load_beat == 3'd3) begin
                                acts_load_beat <= 3'd4;
                                state          <= ST_LOAD_ASCALE;
                            end else begin
                                acts_load_beat <= acts_load_beat + 3'd1;
                            end
                        end
                    end

                    ST_LOAD_ASCALE: begin
                        if (acts_beat_accept) begin
                            acts_mem[acts_load_addr]      <= acts_load_accum;
                            act_scale_mem[acts_load_addr] <= s_axis_acts_tdata[15:0];
                            acts_load_beat <= 3'd0;
                            if (acts_load_sub == 2'd3) begin
                                acts_load_sub <= 2'd0;
                                if ({2'b00, acts_load_q1} + 16'd1 == run_num_q1_blocks) begin
                                    acts_load_q1 <= 14'd0;
                                    if (acts_load_col + 16'd1 == run_num_cols) begin
                                        // All columns loaded - begin matmul.
                                        rowblock_remaining <= run_num_rowblocks;
                                        q1_idx             <= 14'd0;
                                        sub                <= 2'd0;
                                        col                <= 16'd0;
                                        state              <= (run_weight_fmt == 2'd2) ? ST_TLOAD : ST_WSCALE;
                                    end else begin
                                        acts_load_col <= acts_load_col + 16'd1;
                                        state         <= ST_LOAD_ACTS;
                                    end
                                end else begin
                                    acts_load_q1 <= acts_load_q1 + 14'd1;
                                    state        <= ST_LOAD_ACTS;
                                end
                            end else begin
                                acts_load_sub <= acts_load_sub + 2'd1;
                                state         <= ST_LOAD_ACTS;
                            end
                        end
                    end

                    ST_WSCALE: begin
                        if (s_axis_tvalid) begin
                            weight_scales_q <= weight_scales_from_slots;
                            sub             <= 2'd0;
                            col             <= 16'd0;
                            state           <= ST_WISSUE;
                        end
                    end

                    ST_TLOAD: begin
                        if (s_axis_tvalid) begin
                            tern_scales_lo <= weight_scales_from_slots;
                            tern_scales_hi <= tern_scales_hi_from_slots;
                            sub            <= 2'd0;
                            col            <= 16'd0;
                            state          <= ST_TCODE0;
                        end
                    end

                    ST_TCODE0: begin
                        if (s_axis_tvalid) begin
                            tern_code_first_q <= s_axis_tdata;
                            state             <= ST_TCODE1;
                        end
                    end

                    ST_TCODE1: begin
                        if (s_axis_tvalid) begin
                            tern_code_second_q <= s_axis_tdata;
                            col                <= 16'd0;
                            state              <= ST_WISSUE;
                        end
                    end

                    // Hold each wbit beat across the column sweep; advance only on the last
                    // column (s_axis_tready pulses there). Single-cycle accumulate → no gap.
                    ST_WISSUE: begin
                        if (weight_issue_valid) begin
                            issue_valid_q       <= 1'b1;
                            issue_clear_q       <= is_first_sub;
                            issue_col_q         <= col[CW-1:0];
                            issue_weight_bits_q <= (run_weight_fmt == 2'd2) ? tern_decoded_sign : s_axis_tdata;
                            issue_weight_nonzero_q <= (run_weight_fmt == 2'd2) ? tern_decoded_nonzero : {ROWS*32{1'b1}};
                            issue_weight_scales_q <= (run_weight_fmt == 2'd2) ?
                                ((sub < 2) ? tern_scales_lo : tern_scales_hi) : weight_scales_q;
                            issue_acts_packed_q <= acts_packed_w;
                            issue_act_scale_q   <= act_scale_w;
                            if (last_col) begin
                                col <= 16'd0;
                                if (sub == 2'd3) begin
                                    sub <= 2'd0;
                                    if (last_q1) begin
                                        wait_cnt <= WAIT_CYCLES[WW-1:0];
                                        state    <= ST_WAIT_DONE;
                                    end else begin
                                        q1_idx <= q1_idx + 14'd1;
                                        state  <= (run_weight_fmt == 2'd2) ? ST_TLOAD : ST_WSCALE;
                                    end
                                end else begin
                                    sub <= sub + 2'd1;
                                    if (run_weight_fmt == 2'd2)
                                        state <= ST_TCODE0;
                                end
                            end else begin
                                col <= col + 16'd1;
                            end
                        end
                    end

                    // Fixed drain: wait for the last issue to propagate through the rowblock
                    // (FE_LAT + fma) so every accumulator is final, then precompute the emits.
                    ST_WAIT_DONE: begin
                        if (wait_cnt == {WW{1'b0}}) begin
                            pc_col  <= 16'd0;
                            pc_beat <= {EBW{1'b0}};
                            wr_idx  <= {WRW{1'b0}};
                            state   <= ST_PRECOMPUTE;
                        end else begin
                            wait_cnt <= wait_cnt - 1'b1;
                        end
                    end

                    // Walk every (col, beat) through the emit pipeline into result_buf; the
                    // wr_idx collector (above) fills it on valid_out. Done once all N written.
                    ST_PRECOMPUTE: begin
                        if (pc_col < run_num_cols) begin
                            if (pc_beat == EMIT_LAST[EBW-1:0]) begin
                                pc_beat <= {EBW{1'b0}};
                                pc_col  <= pc_col + 16'd1;
                            end else begin
                                pc_beat <= pc_beat + 1'b1;
                            end
                        end
                        if (wr_idx == n_total[WRW-1:0]) begin
                            emit_beat <= {EBW{1'b0}};
                            emit_col  <= 16'd0;
                            state     <= ST_EMIT;
                        end
                    end

                    ST_EMIT: begin
                        if (m_axis_tready) begin
                            if (emit_beat == active_emit_last) begin
                                emit_beat <= {EBW{1'b0}};
                                if (emit_col + 16'd1 == run_num_cols) begin
                                    if (rowblock_remaining == 16'd1) begin
                                        state <= ST_FINISH;
                                    end else begin
                                        rowblock_remaining <= rowblock_remaining - 16'd1;
                                        q1_idx             <= 14'd0;
                                        sub                <= 2'd0;
                                        col                <= 16'd0;
                                        state              <= (run_weight_fmt == 2'd2) ? ST_TLOAD : ST_WSCALE;
                                    end
                                end else begin
                                    emit_col <= emit_col + 16'd1;
                                end
                            end else begin
                                emit_beat <= emit_beat + 1'b1;
                            end
                        end
                    end

                    ST_FINISH: begin
                        kernel_done <= 1'b1;
                        busy_q      <= 1'b0;
                        state       <= ST_IDLE;
                    end

                    default: begin
                        state  <= ST_IDLE;
                        busy_q <= 1'b0;
                    end
                endcase
            end
        end
    end

`ifdef FORMAL
`include "gemm_kernel_properties.vh"
`endif

endmodule
