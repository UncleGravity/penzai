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
    input  wire [15:0] num_q1_blocks,
    input  wire [15:0] num_cols,

    input  wire [63:0] s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,
    input  wire        s_axis_tlast,

    output wire [63:0] m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,

    output wire        activation_abort,
    output wire [3:0]  quantizer_status
);
    localparam [3:0] ST_IDLE  = 4'd0;
    localparam [3:0] ST_INPUT = 4'd1;
    localparam [3:0] ST_LOW   = 4'd2;
    localparam [3:0] ST_HIGH  = 4'd3;
    localparam [3:0] ST_WAIT  = 4'd4;
    localparam [3:0] ST_EMIT  = 4'd5;
    localparam [3:0] ST_DONE  = 4'd6;
    localparam [3:0] ST_ERROR = 4'd7;

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
    reg [3:0]  error_status;

    assign s_axis_tready = state == ST_INPUT;

    wire q_in_ready;
    wire q_out_valid;
    wire [255:0] q_out_quants;
    wire [15:0] q_out_scale;
    wire [3:0] q_out_status;
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
            error_status       <= 4'd0;
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
            error_status       <= 4'd0;
            state              <= (raw_mode && num_q1_blocks != 0 && num_cols != 0) ?
                                  ST_INPUT : ST_IDLE;
        end else begin
            case (state)
                ST_INPUT: if (s_axis_tvalid) begin
                    upper_scalar <= s_axis_tdata[63:32];
                    if (s_axis_tlast != expected_input_last) begin
                        quant_scalar_valid <= 1'b0;
                        error_status <= 4'b0100;
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

                ST_WAIT: if (q_out_valid) begin
                    if (q_out_status != 4'd0) begin
                        error_status <= q_out_status;
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
                            state <= ST_INPUT;
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
endmodule

`default_nettype wire
