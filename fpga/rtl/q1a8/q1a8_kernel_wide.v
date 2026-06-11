// q1a8_kernel_wide - M6 prototype: wide-ingestion Q1A8 matmul.
//
// The narrow kernel was weight-ingestion-bound (63% of cycles loading wbits,
// 4 beats x 64-bit per subblock). This version widens the weight stream to
// ROWS*32 bits (256 for ROWS=8) so a full subblock for all ROWS loads in ONE
// beat, and issues one subblock per cycle into the SAME q1a8_rowblock core.
// Throughput ceiling ~= ROWS*32 MAC/cycle (=256), vs ~40 for the narrow kernel.
//
// Weight stream layout (256-bit beats), per q1block per rowblock:
//   1 scale beat : ROWS fp16 weight scales in [ROWS*16-1:0]
//   4 wbit beats : ROWS*32 weight bits (one Q8 subblock for all rows)
// Acts stream (64-bit) and result stream (64-bit) are unchanged from the narrow
// kernel. NOTE: the cosim feeds beats with no stall, so this measures the
// kernel's intrinsic rate; real throughput is still capped by the DDR/HP weight
// delivery (~286 weight-bits/cycle across 4 HP ports).

`default_nettype none

module q1a8_kernel_wide #(
    parameter integer ROWS          = 8,
    parameter integer MAX_SUB_INDEX = 256
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [15:0]           num_rowblocks,
    output reg                   kernel_done,
    output wire                  busy,

    // AXIS slave: weights (scale beat + wbit beats), ROWS*32 bits wide.
    input  wire [ROWS*32-1:0]    s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    // AXIS slave: acts (acts + act_scales), once per column. 64-bit.
    input  wire [63:0]           s_axis_acts_tdata,
    input  wire                  s_axis_acts_tvalid,
    output wire                  s_axis_acts_tready,

    // AXIS master: 8 fp32 results per rowblock, 2 fp32 per 64-bit beat.
    output wire [63:0]           m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast,
    output wire [7:0]            m_axis_tkeep,

    output wire [3:0]            dbg_state
);

    localparam integer Q8_SUBBLOCKS = 4;

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_LOAD_ACTS  = 4'd1;
    localparam [3:0] ST_LOAD_ASCALE= 4'd2;
    localparam [3:0] ST_WSCALE     = 4'd3;
    localparam [3:0] ST_WISSUE     = 4'd4;
    localparam [3:0] ST_WAIT_DONE  = 4'd5;
    localparam [3:0] ST_DRAIN      = 4'd6;
    localparam [3:0] ST_EMIT       = 4'd7;
    localparam [3:0] ST_FINISH     = 4'd8;

    reg [3:0]  state;
    reg        busy_q;
    reg [15:0] rowblock_remaining;
    reg [13:0] q1_idx;       // current q1block within rowblock (acts addressing)
    reg [1:0]  sub;          // current subblock 0..3
    reg [1:0]  drain_cnt;
    reg [1:0]  emit_beat;

    reg [ROWS*16-1:0] weight_scales_q;

    // -- Acts BRAM (one column, broadcast across rowblocks) --
    reg [255:0] acts_mem      [0:MAX_SUB_INDEX-1];
    reg [15:0]  act_scale_mem [0:MAX_SUB_INDEX-1];
    localparam integer ADDR_W   = $clog2(MAX_SUB_INDEX);
    localparam integer Q1_IDX_W = $clog2(MAX_SUB_INDEX/4);

    // Combinational read addressed by the subblock being issued.
    wire [ADDR_W-1:0] issue_addr = {q1_idx[Q1_IDX_W-1:0], sub};
    wire [255:0] acts_packed_w = acts_mem[issue_addr];
    wire [15:0]  act_scale_w   = act_scale_mem[issue_addr];

    // -- Acts load scratch --
    reg [255:0] acts_load_accum;
    reg [2:0]   acts_load_beat;
    reg [13:0]  acts_load_q1;
    reg [1:0]   acts_load_sub;
    wire [ADDR_W-1:0] acts_load_addr = {acts_load_q1[Q1_IDX_W-1:0], acts_load_sub};

    reg  rowblock_start;
    wire rowblock_done;
    wire [ROWS*32-1:0] rowblock_results;

    wire last_q1   = (q1_idx == num_q1_blocks - 14'd1);
    wire issue_now = (state == ST_WISSUE) && s_axis_tvalid;
    wire rowblock_valid = issue_now;
    wire rowblock_last  = issue_now && last_q1 && (sub == 2'd3);

    assign busy = busy_q;
    assign dbg_state = state;
    assign s_axis_tready =
        busy_q && ((state == ST_WSCALE) || (state == ST_WISSUE));
    assign s_axis_acts_tready =
        busy_q && ((state == ST_LOAD_ACTS) || (state == ST_LOAD_ASCALE));

    wire acts_beat_accept = s_axis_acts_tvalid && s_axis_acts_tready;
    wire start_pulse      = start_kernel && !busy_q;

    q1a8_rowblock #(.ROWS(ROWS)) u_rowblock (
        .clk(clk),
        .rst_n(rst_n),
        .start(rowblock_start),
        .valid_in(rowblock_valid),
        .last_in(rowblock_last),
        .weight_bits_flat(s_axis_tdata),          // current wbit beat
        .weight_scales_flat(weight_scales_q),
        .acts_packed(acts_packed_w),
        .act_scale(act_scale_w),
        .done(rowblock_done),
        .results_flat(rowblock_results)
    );

    reg [63:0] emit_word;
    always @(*) begin
        case (emit_beat)
            2'd0:    emit_word = {rowblock_results[ 63: 32], rowblock_results[ 31:  0]};
            2'd1:    emit_word = {rowblock_results[127: 96], rowblock_results[ 95: 64]};
            2'd2:    emit_word = {rowblock_results[191:160], rowblock_results[159:128]};
            default: emit_word = {rowblock_results[255:224], rowblock_results[223:192]};
        endcase
    end

    assign m_axis_tdata  = emit_word;
    assign m_axis_tvalid = (state == ST_EMIT);
    assign m_axis_tlast  = (state == ST_EMIT) && (emit_beat == 2'd3) &&
                           (rowblock_remaining == 16'd1);
    assign m_axis_tkeep  = 8'hFF;

    always @(posedge clk) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            busy_q             <= 1'b0;
            rowblock_remaining <= 16'd0;
            q1_idx             <= 14'd0;
            sub                <= 2'd0;
            drain_cnt          <= 2'd0;
            emit_beat          <= 2'd0;
            weight_scales_q    <= {ROWS*16{1'b0}};
            rowblock_start     <= 1'b0;
            kernel_done        <= 1'b0;
            acts_load_accum    <= 256'd0;
            acts_load_beat     <= 3'd0;
            acts_load_q1       <= 14'd0;
            acts_load_sub      <= 2'd0;
        end else begin
            rowblock_start <= 1'b0;
            kernel_done    <= 1'b0;

            if (start_pulse) begin
                busy_q         <= 1'b1;
                acts_load_beat <= 3'd0;
                acts_load_q1   <= 14'd0;
                acts_load_sub  <= 2'd0;
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
                                    // All acts loaded - begin matmul.
                                    rowblock_remaining <= num_rowblocks;
                                    q1_idx             <= 14'd0;
                                    sub                <= 2'd0;
                                    rowblock_start     <= 1'b1;
                                    state              <= ST_WSCALE;
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

                    // One weight beat carries ROWS fp16 scales for this q1block.
                    ST_WSCALE: begin
                        if (s_axis_tvalid) begin
                            weight_scales_q <= s_axis_tdata[ROWS*16-1:0];
                            sub             <= 2'd0;
                            state           <= ST_WISSUE;
                        end
                    end

                    // One weight beat per cycle = one subblock for all ROWS,
                    // issued straight into the rowblock core.
                    ST_WISSUE: begin
                        if (s_axis_tvalid) begin
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
                            emit_beat <= 2'd0;
                            state     <= ST_EMIT;
                        end else begin
                            drain_cnt <= drain_cnt - 2'd1;
                        end
                    end

                    ST_EMIT: begin
                        if (m_axis_tready) begin
                            if (emit_beat == 2'd3) begin
                                if (rowblock_remaining == 16'd1) begin
                                    state <= ST_FINISH;
                                end else begin
                                    rowblock_remaining <= rowblock_remaining - 16'd1;
                                    q1_idx             <= 14'd0;
                                    sub                <= 2'd0;
                                    rowblock_start     <= 1'b1;
                                    state              <= ST_WSCALE;
                                end
                            end else begin
                                emit_beat <= emit_beat + 2'd1;
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
