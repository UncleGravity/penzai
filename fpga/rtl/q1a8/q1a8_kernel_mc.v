// q1a8_kernel_mc - multi-column Q1A8 matmul: one weight stream, up to COLS_MAX
// activation columns. Generalizes q1a8_kernel_wide (which is the COLS=1 case).
//
// Weight stream (ROWS*32-bit beats), per q1block per rowblock: 1 scale beat then
// 4 wbit beats (one Q8 subblock for all ROWS). Each wbit beat is HELD while the
// kernel sweeps the columns (s_axis_tready pulses only on the last column), so
// the weights are read once and MAC'd against every column -- this is what makes
// prefill (C columns) cost a single weight stream instead of C.
//
// Acts stream (64-bit): num_cols columns, each column the same layout as the
// single-column kernel (per q1block per subblock: 4 act beats + 1 scale beat),
// columns back to back. Result stream (64-bit): per rowblock, per column, 4 beats
// of 8 fp32 (2/beat, lane-major) -- layout [rowblock][col][row].

`default_nettype none

module q1a8_kernel_mc #(
    parameter integer ROWS          = 16,
    parameter integer COLS_MAX      = 8,
    parameter integer MAX_SUB_INDEX = 64,
    // Decode accumulator-pool depth (see q1a8_rowblock_mc): >= the accumulate
    // recurrence latency, <= COLS_MAX.
    parameter integer ACCUM_DEPTH   = 4
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [15:0]           num_rowblocks,
    input  wire [15:0]           num_cols,
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

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_LOAD_ACTS  = 4'd1;
    localparam [3:0] ST_LOAD_ASCALE= 4'd2;
    localparam [3:0] ST_WSCALE     = 4'd3;
    localparam [3:0] ST_WISSUE     = 4'd4;
    localparam [3:0] ST_WAIT_DONE  = 4'd5;
    localparam [3:0] ST_DRAIN      = 4'd6;
    localparam [3:0] ST_EMIT       = 4'd7;
    localparam [3:0] ST_FINISH     = 4'd8;

    // Result emit: the stream is 64-bit (2 fp32 lane-major per beat), so a
    // rowblock's ROWS results take ROWS/2 beats. ROWS-parameterized so a wider
    // array (ROWS=16 -> 512-bit weights, 8 emit beats) needs no emit rewrite.
    localparam integer EMIT_BEATS = ROWS / 2;
    localparam integer EBW        = (EMIT_BEATS <= 1) ? 1 : $clog2(EMIT_BEATS);
    localparam integer EMIT_LAST  = EMIT_BEATS - 1;

    reg [3:0]  state;
    reg        busy_q;
    reg [15:0] rowblock_remaining;
    reg [13:0] q1_idx;
    reg [1:0]  sub;
    reg [15:0] col;          // column being issued (matmul)
    reg [1:0]  drain_cnt;
    reg [EBW-1:0] emit_beat;
    reg [15:0] emit_col;

    reg [ROWS*16-1:0] weight_scales_q;
    wire [ROWS*16-1:0] weight_scales_from_slots;

    // Acts BRAM: COLS_MAX columns, each MAX_SUB_INDEX subblock entries.
    localparam integer ACT_DEPTH = MAX_SUB_INDEX * COLS_MAX;
    reg [255:0] acts_mem      [0:ACT_DEPTH-1];
    reg [15:0]  act_scale_mem [0:ACT_DEPTH-1];
    localparam integer SUB_W = $clog2(MAX_SUB_INDEX);

    // subblock index = q1_idx*4 + sub; full address folds in the column.
    wire [SUB_W-1:0]  issue_sub  = {q1_idx[SUB_W-3:0], sub};
    wire [31:0]       issue_addr = issue_sub * COLS_MAX + {16'd0, col};
    wire [255:0] acts_packed_w = acts_mem[issue_addr];
    wire [15:0]  act_scale_w   = act_scale_mem[issue_addr];

    // Acts load scratch.
    reg [255:0] acts_load_accum;
    reg [2:0]   acts_load_beat;
    reg [13:0]  acts_load_q1;
    reg [1:0]   acts_load_sub;
    reg [15:0]  acts_load_col;
    wire [SUB_W-1:0] acts_load_sub_idx = {acts_load_q1[SUB_W-3:0], acts_load_sub};
    wire [31:0] acts_load_addr = acts_load_sub_idx * COLS_MAX + {16'd0, acts_load_col};

    reg  rowblock_start;
    wire rowblock_done;
    wire [ROWS*32-1:0] rowblock_results;

    wire [15:0] q1_idx_wide = {2'b00, q1_idx};
    wire last_q1   = (q1_idx_wide == num_q1_blocks - 16'd1);
    wire last_col  = (col == num_cols - 16'd1);
    wire issue_now = (state == ST_WISSUE) && s_axis_tvalid;
    wire rowblock_last = issue_now && last_q1 && (sub == 2'd3) && last_col;
    wire single_col = (num_cols == 16'd1);

    assign busy = busy_q;
    assign dbg_state = state;
    // Consume the scale beat in WSCALE; consume each wbit beat on the last column.
    assign s_axis_tready =
        busy_q && ((state == ST_WSCALE) || ((state == ST_WISSUE) && last_col));
    assign s_axis_acts_tready =
        busy_q && ((state == ST_LOAD_ACTS) || (state == ST_LOAD_ASCALE));

    wire acts_beat_accept = s_axis_acts_tvalid && s_axis_acts_tready;
    wire start_pulse      = start_kernel && !busy_q;

    genvar scale_lane;
    generate
        for (scale_lane = 0; scale_lane < ROWS; scale_lane = scale_lane + 1) begin : gen_scale_slots
            assign weight_scales_from_slots[scale_lane*16 +: 16] =
                s_axis_tdata[scale_lane*32 +: 16];
        end
    endgenerate

    q1a8_rowblock_mc #(.ROWS(ROWS), .COLS_MAX(COLS_MAX), .ACCUM_DEPTH(ACCUM_DEPTH)) u_rowblock (
        .clk(clk),
        .rst_n(rst_n),
        .start(rowblock_start),
        .valid_in(issue_now),
        .last_in(rowblock_last),
        .single_col(single_col),
        .col_idx(col[CW-1:0]),
        .weight_bits_flat(s_axis_tdata),
        .weight_scales_flat(weight_scales_q),
        .acts_packed(acts_packed_w),
        .act_scale(act_scale_w),
        .read_col(emit_col[CW-1:0]),
        .done(rowblock_done),
        .results_flat(rowblock_results)
    );

    // Beat `emit_beat` carries result lanes [2*emit_beat, 2*emit_beat+1] (64 bits,
    // {emit_beat,6'b0} == emit_beat*64). Generalizes the fixed 4-beat mux to any ROWS.
    wire [63:0] emit_word = rowblock_results[{emit_beat, 6'b0} +: 64];

    assign m_axis_tdata  = emit_word;
    assign m_axis_tvalid = (state == ST_EMIT);
    assign m_axis_tlast  = (state == ST_EMIT) && (emit_beat == EMIT_LAST[EBW-1:0]) &&
                           (emit_col == num_cols - 16'd1) && (rowblock_remaining == 16'd1);
    assign m_axis_tkeep  = 8'hFF;

    always @(posedge clk) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            busy_q             <= 1'b0;
            rowblock_remaining <= 16'd0;
            q1_idx             <= 14'd0;
            sub                <= 2'd0;
            col                <= 16'd0;
            drain_cnt          <= 2'd0;
            emit_beat          <= {EBW{1'b0}};
            emit_col           <= 16'd0;
            weight_scales_q    <= {ROWS*16{1'b0}};
            rowblock_start     <= 1'b0;
            kernel_done        <= 1'b0;
            acts_load_accum    <= 256'd0;
            acts_load_beat     <= 3'd0;
            acts_load_q1       <= 14'd0;
            acts_load_sub      <= 2'd0;
            acts_load_col      <= 16'd0;
        end else begin
            rowblock_start <= 1'b0;
            kernel_done    <= 1'b0;

            if (start_pulse) begin
                busy_q         <= 1'b1;
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
                                if ({2'b00, acts_load_q1} + 16'd1 == num_q1_blocks) begin
                                    acts_load_q1 <= 14'd0;
                                    if (acts_load_col + 16'd1 == num_cols) begin
                                        // All columns loaded - begin matmul.
                                        rowblock_remaining <= num_rowblocks;
                                        q1_idx             <= 14'd0;
                                        sub                <= 2'd0;
                                        col                <= 16'd0;
                                        rowblock_start     <= 1'b1;
                                        state              <= ST_WSCALE;
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

                    // Hold each wbit beat across the column sweep; advance only on
                    // the last column (s_axis_tready pulses there).
                    ST_WISSUE: begin
                        if (s_axis_tvalid) begin
                            if (last_col) begin
                                col <= 16'd0;
                                if (sub == 2'd3) begin
                                    sub <= 2'd0;
                                    if (last_q1) begin
                                        state <= ST_WAIT_DONE;
                                    end else begin
                                        q1_idx <= q1_idx + 14'd1;
                                        state  <= ST_WSCALE;
                                    end
                                end else begin
                                    sub <= sub + 2'd1;
                                end
                            end else begin
                                col <= col + 16'd1;
                            end
                        end
                    end

                    ST_WAIT_DONE: begin
                        if (rowblock_done) begin
                            drain_cnt <= 2'd2;
                            state     <= ST_DRAIN;
                        end
                    end

                    ST_DRAIN: begin
                        if (drain_cnt == 2'd0) begin
                            emit_beat <= {EBW{1'b0}};
                            emit_col  <= 16'd0;
                            state     <= ST_EMIT;
                        end else begin
                            drain_cnt <= drain_cnt - 2'd1;
                        end
                    end

                    ST_EMIT: begin
                        if (m_axis_tready) begin
                            if (emit_beat == EMIT_LAST[EBW-1:0]) begin
                                emit_beat <= {EBW{1'b0}};
                                if (emit_col + 16'd1 == num_cols) begin
                                    if (rowblock_remaining == 16'd1) begin
                                        state <= ST_FINISH;
                                    end else begin
                                        rowblock_remaining <= rowblock_remaining - 16'd1;
                                        q1_idx             <= 14'd0;
                                        sub                <= 2'd0;
                                        col                <= 16'd0;
                                        rowblock_start     <= 1'b1;
                                        state              <= ST_WSCALE;
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

endmodule
