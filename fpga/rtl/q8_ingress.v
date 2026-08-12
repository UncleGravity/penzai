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
    input  wire [31:0] total_blocks,

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

    reg [3:0] state;
    reg [63:0] input_beat;
    reg [5:0] scalar_index;
    reg [31:0] input_beats;
    reg [31:0] expected_beats;
    reg [31:0] completed_blocks;
    reg [2:0] emit_index;
    reg [255:0] block_quants;
    reg [15:0] block_scale;
    reg [3:0] error_status;

    assign s_axis_tready = state == ST_INPUT;

    wire q_in_valid = (state == ST_LOW) || (state == ST_HIGH);
    wire [31:0] q_in_data = (state == ST_LOW) ? input_beat[31:0] : input_beat[63:32];
    wire q_in_last = (state == ST_HIGH) && (scalar_index == 6'd31);
    wire q_in_ready;
    wire q_out_valid;
    wire [255:0] q_out_quants;
    wire [15:0] q_out_scale;
    wire [3:0] q_out_status;
    wire q_out_ready = state == ST_WAIT;

    // Reset the leaf at every kernel launch. This also makes an aborted raw run
    // recoverable without resetting the complete accelerator.
    wire q_rst_n = rst_n && !start;
    q8_quantizer u_quantizer (
        .clk(clk),
        .rst_n(q_rst_n),
        .in_valid(q_in_valid),
        .in_ready(q_in_ready),
        .in_data(q_in_data),
        .in_last(q_in_last),
        .out_valid(q_out_valid),
        .out_ready(q_out_ready),
        .out_quants(q_out_quants),
        .out_scale(q_out_scale),
        .out_status(q_out_status)
    );

    assign m_axis_tvalid = state == ST_EMIT;
    assign m_axis_tdata = (emit_index == 3'd0) ? block_quants[63:0] :
                         (emit_index == 3'd1) ? block_quants[127:64] :
                         (emit_index == 3'd2) ? block_quants[191:128] :
                         (emit_index == 3'd3) ? block_quants[255:192] :
                         {48'd0, block_scale};
    assign activation_abort = state == ST_ERROR;
    assign quantizer_status = error_status;

    wire expected_input_last = input_beats + 32'd1 == expected_beats;

    always @(posedge clk) begin
        if (!rst_n) begin
            state            <= ST_IDLE;
            scalar_index     <= 6'd0;
            input_beats      <= 32'd0;
            expected_beats   <= 32'd0;
            completed_blocks <= 32'd0;
            emit_index       <= 3'd0;
            error_status     <= 4'd0;
        end else if (start) begin
            scalar_index     <= 6'd0;
            input_beats      <= 32'd0;
            expected_beats   <= total_blocks << 4;
            completed_blocks <= 32'd0;
            emit_index       <= 3'd0;
            error_status     <= 4'd0;
            state            <= (raw_mode && total_blocks != 0) ? ST_INPUT : ST_IDLE;
        end else begin
            case (state)
                ST_INPUT: if (s_axis_tvalid) begin
                    input_beat <= s_axis_tdata;
                    input_beats <= input_beats + 32'd1;
                    if (s_axis_tlast != expected_input_last) begin
                        error_status <= 4'b0100;
                        state <= ST_ERROR;
                    end else begin
                        state <= ST_LOW;
                    end
                end

                ST_LOW: if (q_in_ready) begin
                    scalar_index <= scalar_index + 6'd1;
                    state <= ST_HIGH;
                end

                ST_HIGH: if (q_in_ready) begin
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
                        block_quants <= q_out_quants;
                        block_scale <= q_out_scale;
                        emit_index <= 3'd0;
                        state <= ST_EMIT;
                    end
                end

                ST_EMIT: if (m_axis_tready) begin
                    if (emit_index == 3'd4) begin
                        emit_index <= 3'd0;
                        completed_blocks <= completed_blocks + 32'd1;
                        if (completed_blocks + 32'd1 == total_blocks)
                            state <= ST_DONE;
                        else
                            state <= ST_INPUT;
                    end else begin
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
