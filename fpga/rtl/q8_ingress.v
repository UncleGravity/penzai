// q8_ingress - adapt a 64-bit FP32 AXIS stream to gemm_kernel's native Q8_0 stream.
//
// Raw activations arrive column-major, two FP32 values per beat. The exact
// q8_quantizer consumes one 32-value block and this adapter emits the existing
// five-beat GEMM record: four packed-int8 beats followed by one f16 scale beat.
// The adapter is deliberately sequential; grouped projections amortize its work.

`default_nettype none

module q8_ingress (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        raw_mode,
    input  wire        internal_mode,
    input  wire [15:0] num_q1_blocks,
    input  wire [15:0] num_cols,

    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    // Section-local SwiGLU produces FP32 scalars.  They share the one canonical
    // q8_quantizer below with external raw-FP32 ingress.
    input  wire [31:0]  internal_data,
    input  wire         internal_last,
    input  wire [1:0]   internal_status,
    input  wire         internal_valid,
    output wire         internal_ready,
    output wire         internal_record_done,

    output wire [63:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,

    output wire        activation_abort,
    output wire [5:0]  quantizer_status
);
    localparam [3:0] ST_IDLE  = 4'd0;
    localparam [3:0] ST_INPUT = 4'd1;
    localparam [3:0] ST_LOW   = 4'd2;
    localparam [3:0] ST_HIGH  = 4'd3;
    localparam [3:0] ST_WAIT  = 4'd4;
    localparam [3:0] ST_EMIT  = 4'd5;
    localparam [3:0] ST_DONE  = 4'd6;
    localparam [3:0] ST_ERROR = 4'd7;
    localparam [3:0] ST_INTERNAL = 4'd8;

    reg [3:0]  state;
    reg [31:0] upper_scalar;
    reg [31:0] quant_scalar;
    reg        quant_scalar_valid;
    reg        quant_scalar_last;
    reg [5:0]  scalar_index;
    reg [1:0]  block_sub;
    reg [15:0] block_q1;
    reg [15:0] block_col;
    reg [15:0] last_q1;
    reg [15:0] last_col;
    reg [2:0]  emit_index;
    reg [63:0] emit_data;
    reg [5:0]  error_status;
    reg        internal_run;
    reg [1:0]  internal_status_q;
    reg        internal_valid_q;

    assign s_axis_tready = state == ST_INPUT;

    wire q_in_ready;
    wire q_out_valid;
    wire [255:0] q_out_quants;
    wire [15:0] q_out_scale;
    wire [3:0] q_out_status;
    wire internal_frame_ok = quant_scalar_last == (scalar_index == 6'd31);
    wire internal_stage_ok = (internal_status_q == 2'd0) && internal_frame_ok;
    wire internal_accept = internal_run && (state == ST_INTERNAL) &&
                           internal_valid_q && q_in_ready;
    // A diagnosed scalar is consumed by the section controller but is not
    // allowed to mutate the quantizer's partial block before fail-closed abort.
    // Both raw and internal modes use the same final scalar registers, keeping
    // the mode select and SwiGLU arithmetic out of the quantizer's amax/control
    // cone. The stage is elastic within a block: when the quantizer accepts a
    // healthy non-final scalar, the source may replace it on the same edge.
    // Block-final and diagnosed scalars retain the fail-closed boundary.
    assign internal_ready = rst_n && !start && internal_run &&
                            (state == ST_INTERNAL) &&
                            (!internal_valid_q ||
                             (q_in_ready && internal_stage_ok &&
                              (scalar_index != 6'd31)));
    wire internal_source_accept = internal_valid && internal_ready;
    wire internal_replace = internal_source_accept && internal_valid_q;
    // Completion means all five native beats for this block were accepted by
    // gemm_kernel. The scratch walker uses this to retire queued blocks.
    assign internal_record_done = internal_run && (state == ST_EMIT) &&
                                  (emit_index == 3'd4) && m_axis_tready;
    // Keep the leaf's complete record stable through all five output beats. The
    // fifth accepted beat releases it for the next block.
    wire q_out_ready = (state == ST_EMIT) && (emit_index == 3'd4) && m_axis_tready;

    // Reset the leaf at every kernel launch. This also makes an aborted raw run
    // recoverable without resetting the complete accelerator.
    wire q_rst_n = rst_n && !start;
    q8_quantizer u_quantizer (
        .clk(clk),
        .rst_n(q_rst_n),
        .in_valid(quant_scalar_valid),
        .in_ready(q_in_ready),
        .in_data(quant_scalar),
        .in_last(quant_scalar_last),
        .out_valid(q_out_valid),
        .out_ready(q_out_ready),
        .out_quants(q_out_quants),
        .out_scale(q_out_scale),
        .out_status(q_out_status)
    );

    assign m_axis_tvalid = state == ST_EMIT;
    assign m_axis_tdata = emit_data;
    assign activation_abort = state == ST_ERROR;
    assign quantizer_status = error_status;

    wire final_block = (block_sub == 2'd3) &&
                       (block_q1 == last_q1) &&
                       (block_col == last_col);
    wire expected_input_last = (scalar_index == 6'd30) && final_block;

    always @(posedge clk) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            quant_scalar_valid <= 1'b0;
            quant_scalar_last  <= 1'b0;
            scalar_index       <= 6'd0;
            block_sub          <= 2'd0;
            block_q1           <= 16'd0;
            block_col          <= 16'd0;
            last_q1            <= 16'd0;
            last_col           <= 16'd0;
            emit_index         <= 3'd0;
            error_status       <= 6'd0;
            internal_run       <= 1'b0;
            internal_valid_q   <= 1'b0;
        end else if (start) begin
            quant_scalar_valid <= 1'b0;
            quant_scalar_last  <= 1'b0;
            scalar_index       <= 6'd0;
            block_sub          <= 2'd0;
            block_q1           <= 16'd0;
            block_col          <= 16'd0;
            last_q1            <= num_q1_blocks - 16'd1;
            last_col           <= num_cols - 16'd1;
            emit_index         <= 3'd0;
            error_status       <= 6'd0;
            internal_run       <= internal_mode;
            internal_valid_q   <= 1'b0;
            state              <= (raw_mode && num_q1_blocks != 0 && num_cols != 0) ?
                                  (internal_mode ? ST_INTERNAL : ST_INPUT) : ST_IDLE;
        end else begin
            case (state)
                ST_INPUT: if (s_axis_tvalid) begin
                    upper_scalar <= s_axis_tdata[63:32];
                    if (s_axis_tlast != expected_input_last) begin
                        quant_scalar_valid <= 1'b0;
                        error_status <= 6'b000100;
                        state <= ST_ERROR;
                    end else begin
                        // Register the scalar boundary before the quantizer. LOW
                        // presents this word; its handshake loads the saved high word.
                        quant_scalar <= s_axis_tdata[31:0];
                        quant_scalar_last <= 1'b0;
                        quant_scalar_valid <= 1'b1;
                        state <= ST_LOW;
                    end
                end

                ST_LOW: if (q_in_ready) begin
                    quant_scalar <= upper_scalar;
                    quant_scalar_last <= scalar_index == 6'd30;
                    scalar_index <= scalar_index + 6'd1;
                    state <= ST_HIGH;
                end

                ST_HIGH: if (q_in_ready) begin
                    quant_scalar_valid <= 1'b0;
                    quant_scalar_last <= 1'b0;
                    if (scalar_index == 6'd31) begin
                        scalar_index <= 6'd0;
                        state <= ST_WAIT;
                    end else begin
                        scalar_index <= scalar_index + 6'd1;
                        state <= ST_INPUT;
                    end
                end

                ST_INTERNAL: begin
                    if (internal_accept) begin
                        if (!internal_stage_ok) begin
                            internal_valid_q   <= 1'b0;
                            quant_scalar_valid <= 1'b0;
                            quant_scalar_last  <= 1'b0;
                            if (internal_status_q != 2'd0)
                                error_status <= {internal_status_q, 4'd0};
                            else
                                error_status <= 6'b000100;
                            state <= ST_ERROR;
                        end else if (scalar_index == 6'd31) begin
                            internal_valid_q   <= 1'b0;
                            quant_scalar_valid <= 1'b0;
                            quant_scalar_last  <= 1'b0;
                            scalar_index <= 6'd0;
                            state <= ST_WAIT;
                        end else begin
                            scalar_index <= scalar_index + 1'b1;
                            if (internal_replace) begin
                                quant_scalar      <= internal_data;
                                quant_scalar_last <= internal_last;
                                quant_scalar_valid <= (internal_status == 2'd0) &&
                                                      (internal_last ==
                                                       (scalar_index == 6'd30));
                                internal_status_q <= internal_status;
                                internal_valid_q  <= 1'b1;
                            end else begin
                                internal_valid_q   <= 1'b0;
                                quant_scalar_valid <= 1'b0;
                                quant_scalar_last  <= 1'b0;
                            end
                        end
                    end else if (internal_source_accept) begin
                        quant_scalar      <= internal_data;
                        quant_scalar_last <= internal_last;
                        quant_scalar_valid <= (internal_status == 2'd0) &&
                                              (internal_last ==
                                               (scalar_index == 6'd31));
                        internal_status_q <= internal_status;
                        internal_valid_q  <= 1'b1;
                    end
                end

                ST_WAIT: if (q_out_valid) begin
                    if (q_out_status != 4'd0) begin
                        error_status <= {2'd0, q_out_status};
                        state <= ST_ERROR;
                    end else begin
                        emit_data <= q_out_quants[63:0];
                        emit_index <= 3'd0;
                        state <= ST_EMIT;
                    end
                end

                ST_EMIT: if (m_axis_tready) begin
                    if (emit_index == 3'd4) begin
                        emit_index <= 3'd0;
                        if (final_block) begin
                            state <= ST_DONE;
                        end else begin
                            if (block_sub == 2'd3) begin
                                block_sub <= 2'd0;
                                if (block_q1 == last_q1) begin
                                    block_q1 <= 16'd0;
                                    block_col <= block_col + 16'd1;
                                end else begin
                                    block_q1 <= block_q1 + 16'd1;
                                end
                            end else begin
                                block_sub <= block_sub + 2'd1;
                            end
                            state <= internal_run ? ST_INTERNAL : ST_INPUT;
                        end
                    end else begin
                        case (emit_index)
                            3'd0: emit_data <= q_out_quants[127:64];
                            3'd1: emit_data <= q_out_quants[191:128];
                            3'd2: emit_data <= q_out_quants[255:192];
                            default: emit_data <= {48'd0, q_out_scale};
                        endcase
                        emit_index <= emit_index + 3'd1;
                    end
                end

                ST_DONE: ;
                ST_ERROR: ;
                default: state <= ST_IDLE;
            endcase
        end
    end

`ifdef FORMAL
    reg        f_past_valid = 1'b0;
    reg [5:0]  f_scalar_count = 6'd0;
    reg [2:0]  f_native_beat = 3'd0;
    reg        f_record_pending = 1'b0;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!rst_n || start) begin
            f_scalar_count <= 6'd0;
            f_native_beat <= 3'd0;
            f_record_pending <= 1'b0;
        end else if (internal_run) begin
            if (internal_accept) begin
                assert(!f_record_pending);
                if ((internal_status_q == 2'd0) && internal_frame_ok) begin
                    if (f_scalar_count == 6'd31) begin
                        f_scalar_count <= 6'd0;
                        f_record_pending <= 1'b1;
                    end else begin
                        f_scalar_count <= f_scalar_count + 1'b1;
                    end
                end
            end

            if (m_axis_tvalid && m_axis_tready) begin
                assert(f_record_pending);
                assert(internal_record_done == (f_native_beat == 3'd4));
                if (f_native_beat == 3'd4) begin
                    f_native_beat <= 3'd0;
                    f_record_pending <= 1'b0;
                end else begin
                    f_native_beat <= f_native_beat + 1'b1;
                end
            end else begin
                assert(!internal_record_done);
            end

            assert(s_axis_tready == 1'b0);
            assert(f_scalar_count <= 6'd31);
            assert(f_native_beat <= 3'd4);
            if (f_record_pending)
                assert(!internal_ready);
        end

        if (rst_n) begin
            assert(state <= ST_INTERNAL);
            assert(internal_record_done ==
                   (internal_run && (state == ST_EMIT) &&
                    (emit_index == 3'd4) && m_axis_tready));

            if (f_past_valid && $past(rst_n) &&
                $past(m_axis_tvalid && !m_axis_tready)) begin
                assert(m_axis_tvalid);
                assert(m_axis_tdata == $past(m_axis_tdata));
                assert(emit_index == $past(emit_index));
            end

            if (f_past_valid && $past(rst_n) && !$past(start) &&
                $past(internal_accept &&
                      ((internal_status_q != 2'd0) || !internal_frame_ok))) begin
                assert(state == ST_ERROR);
                assert(activation_abort);
                assert(!m_axis_tvalid && !internal_ready);
            end

            if (f_past_valid && !start && $past(rst_n && !start &&
                                                internal_replace)) begin
                assert(state == ST_INTERNAL);
                assert(internal_valid_q);
                assert(quant_scalar == $past(internal_data));
                assert(quant_scalar_last == $past(internal_last));
                assert(internal_status_q == $past(internal_status));
                assert(scalar_index == $past(scalar_index) + 1'b1);
                assert(quant_scalar_valid ==
                       (($past(internal_status) == 2'd0) &&
                        ($past(internal_last) ==
                         ($past(scalar_index) == 6'd30))));
            end

            if (f_past_valid && $past(rst_n && start && raw_mode &&
                                      internal_mode &&
                                      (num_q1_blocks != 0) &&
                                      (num_cols != 0))) begin
                assert(internal_run);
                assert(state == ST_INTERNAL);
                assert(quantizer_status == 6'd0);
            end
        end

        cover(rst_n && internal_record_done);
        cover(rst_n && internal_replace);
        cover(rst_n && internal_run && activation_abort);
    end
`endif
endmodule

`default_nettype wire
