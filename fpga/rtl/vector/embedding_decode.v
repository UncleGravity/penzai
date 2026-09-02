// Random-row Q1/Q2 embedding lookup for the resident token engine.
//
// The embedding table stays in the same four-port issue layout used by the
// projection engine and, for tied models, by the LM head.  A token selects one
// row lane from one 128-bit port stream.  The decoder emits four consecutive
// FP32 values per beat and processes the active tokens in order.  No duplicate
// row-major embedding image is required.

`default_nettype none

module embedding_decode (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,

    input  wire          start_valid,
    output wire          start_ready,
    input  wire [63:0]   table_addr,
    // Number of 128-scalar source blocks (D/128), not the token engine's
    // 32-scalar Q8 block count.  The integration boundary performs /4.
    input  wire [5:0]    q1_blocks,
    input  wire [17:0]   vocab_rows,
    input  wire [1:0]    weight_fmt,
    input  wire [3:0]    token_count,
    input  wire [7:0]    token_mask,
    input  wire [255:0]  token_ids,

    output wire          mem_req_valid,
    input  wire          mem_req_ready,
    output wire [63:0]   mem_req_addr,
    input  wire          mem_rsp_valid,
    output wire          mem_rsp_ready,
    input  wire [127:0]  mem_rsp_data,
    input  wire          mem_rsp_error,

    output wire          out_valid,
    input  wire          out_ready,
    output wire [2:0]    out_token,
    output wire [11:0]   out_index,
    output wire [127:0]  out_data,
    output wire          out_last,

    output reg           busy,
    output reg           done,
    output reg           error,
    output reg  [7:0]    status
);
    localparam [1:0] WEIGHT_Q1 = 2'd1;
    localparam [1:0] WEIGHT_Q2 = 2'd2;

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_TOKEN      = 4'd1;
    localparam [3:0] ST_GROUP      = 4'd2;
    localparam [3:0] ST_BLOCK      = 4'd3;
    localparam [3:0] ST_SCALE_REQ  = 4'd4;
    localparam [3:0] ST_SCALE_RSP  = 4'd5;
    localparam [3:0] ST_CODE0_REQ  = 4'd6;
    localparam [3:0] ST_CODE0_RSP  = 4'd7;
    localparam [3:0] ST_CODE1_REQ  = 4'd8;
    localparam [3:0] ST_CODE1_RSP  = 4'd9;
    localparam [3:0] ST_EMIT       = 4'd10;
    localparam [3:0] ST_DRAIN      = 4'd11;

    localparam [7:0] ERR_CONFIG   = 8'h01;
    localparam [7:0] ERR_TOKEN    = 8'h02;
    localparam [7:0] ERR_SCALE    = 8'h04;
    localparam [7:0] ERR_RESERVED = 8'h08;
    localparam [7:0] ERR_MEMORY   = 8'h10;

    reg [3:0] state_q;
    reg [63:0] table_addr_q;
    reg [5:0] q1_blocks_q;
    reg [17:0] vocab_rows_q;
    reg [1:0] weight_fmt_q;
    reg [3:0] token_count_q;
    reg [7:0] token_mask_q;
    reg [255:0] token_ids_q;

    reg [2:0] token_q;
    reg [4:0] q1_q;
    reg [1:0] sub_q;
    reg [2:0] chunk_q;
    reg [1:0] port_lane_q;
    reg [17:0] linear_rowblock_q;
    reg [25:0] linear_group_q;
    reg [63:0] mem_req_addr_q;
    reg [15:0] scale_lo_q;
    reg [15:0] scale_hi_q;
    reg [31:0] q1_bits_q;
    reg [31:0] q2_code_lo_q;
    reg [63:0] q2_codes_q;
    reg mem_outstanding_q;

    wire [31:0] selected_token_id = token_ids_q[token_q*32 +: 32];
    wire [17:0] rowblock_count = (vocab_rows_q + 18'd15) >> 4;
    wire [1:0] selected_port = selected_token_id[3:2];
    wire [17:0] selected_rowblock = {4'd0, selected_token_id[17:4]};
    wire [25:0] group_product = linear_rowblock_q * q1_blocks_q;
    wire [63:0] q1_group_offset =
        ({38'd0, linear_group_q} << 6) +
        ({38'd0, linear_group_q} << 4);
    wire [63:0] q2_group_offset =
        ({38'd0, linear_group_q} << 7) +
        ({38'd0, linear_group_q} << 4);

    wire config_ok = (table_addr != 64'd0) && (table_addr[5:0] == 6'd0) &&
                     (q1_blocks != 6'd0) && (q1_blocks <= 6'd32) &&
                     (vocab_rows != 18'd0) &&
                     ((weight_fmt == WEIGHT_Q1) ||
                      (weight_fmt == WEIGHT_Q2)) &&
                     (token_count >= 4'd1) && (token_count <= 4'd8) &&
                     ({1'b0, token_mask} ==
                      ((9'd1 << token_count) - 1'b1));

    assign start_ready = !busy && !mem_outstanding_q;
    wire start_fire = start_valid && start_ready;

    assign mem_req_valid = busy && !clear &&
                                   ((state_q == ST_SCALE_REQ) ||
                                    (state_q == ST_CODE0_REQ) ||
                                    (state_q == ST_CODE1_REQ));
    assign mem_rsp_ready = busy && ((state_q == ST_DRAIN) ||
                                    (state_q == ST_SCALE_RSP) ||
                                    (state_q == ST_CODE0_RSP) ||
                                    (state_q == ST_CODE1_RSP));
    wire mem_req_fire = mem_req_valid && mem_req_ready;
    wire mem_rsp_fire = mem_rsp_valid && mem_rsp_ready;

    wire [31:0] selected_slot = mem_rsp_data[port_lane_q*32 +: 32];
    wire [63:0] incoming_codes = {selected_slot, q2_code_lo_q};
    wire incoming_reserved =
        |(incoming_codes & (incoming_codes >> 1) &
          64'h5555_5555_5555_5555);

    // Packed embedding records are physically sequential within each source
    // block: one scale beat followed by four Q1 or eight Q2 code beats. Keeping
    // the next address registered removes sub-index arithmetic from the shared
    // reader command path.
    assign mem_req_addr = mem_req_addr_q;

    wire [15:0] selected_scale =
        ((weight_fmt_q == WEIGHT_Q2) && sub_q[1]) ?
        scale_hi_q : scale_lo_q;
    wire [31:0] scale_f32;
    cvt_f16_f32 u_scale_widen (
        .in(selected_scale),
        .out(scale_f32)
    );

    wire [31:0] q1_window = q1_bits_q >> (chunk_q * 4);
    wire [63:0] q2_window = q2_codes_q >> (chunk_q * 8);
    wire [1:0] code0 = q2_window[1:0];
    wire [1:0] code1 = q2_window[3:2];
    wire [1:0] code2 = q2_window[5:4];
    wire [1:0] code3 = q2_window[7:6];

    function [31:0] decoded_value;
        input [31:0] scale;
        input         q1_sign;
        input [1:0]   q2_code;
        input         ternary;
        begin
            if (ternary) begin
                case (q2_code)
                    2'd0: decoded_value = scale ^ 32'h8000_0000;
                    2'd1: decoded_value = 32'd0;
                    2'd2: decoded_value = scale;
                    default: decoded_value = 32'h7fc0_0000;
                endcase
            end else begin
                decoded_value = q1_sign ? scale :
                                (scale ^ 32'h8000_0000);
            end
        end
    endfunction

    wire ternary = weight_fmt_q == WEIGHT_Q2;
    assign out_valid = busy && (state_q == ST_EMIT);
    assign out_token = token_q;
    assign out_index = {q1_q, 7'd0} +
                       {5'd0, sub_q, 5'd0} +
                       {7'd0, chunk_q, 2'd0};
    assign out_data = {
        decoded_value(scale_f32, q1_window[3], code3, ternary),
        decoded_value(scale_f32, q1_window[2], code2, ternary),
        decoded_value(scale_f32, q1_window[1], code1, ternary),
        decoded_value(scale_f32, q1_window[0], code0, ternary)
    };
    assign out_last = ({1'b0, token_q} + 1'b1 == token_count_q) &&
                      ({1'b0, q1_q} + 1'b1 == q1_blocks_q) &&
                      (sub_q == 2'd3) && (chunk_q == 3'd7);
    wire out_fire = out_valid && out_ready;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 8'd0;
            token_q <= 3'd0;
            q1_q <= 5'd0;
            sub_q <= 2'd0;
            chunk_q <= 3'd0;
            mem_req_addr_q <= 64'd0;
            mem_outstanding_q <= 1'b0;
        end else begin
            done <= 1'b0;

            if (mem_req_fire)
                mem_outstanding_q <= 1'b1;
            if (mem_rsp_fire)
                mem_outstanding_q <= 1'b0;

            if (clear) begin
                done <= 1'b0;
                error <= 1'b0;
                status <= 8'd0;
                if (mem_outstanding_q && !(mem_rsp_fire)) begin
                    busy <= 1'b1;
                    state_q <= ST_DRAIN;
                end else begin
                    busy <= 1'b0;
                    state_q <= ST_IDLE;
                end
            end else if (start_fire) begin
                table_addr_q <= table_addr;
                q1_blocks_q <= q1_blocks;
                vocab_rows_q <= vocab_rows;
                weight_fmt_q <= weight_fmt;
                token_count_q <= token_count;
                token_mask_q <= token_mask;
                token_ids_q <= token_ids;
                token_q <= 3'd0;
                q1_q <= 5'd0;
                sub_q <= 2'd0;
                chunk_q <= 3'd0;
                error <= 1'b0;
                status <= 8'd0;
                if (!config_ok) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                    status <= ERR_CONFIG;
                    state_q <= ST_IDLE;
                end else begin
                    busy <= 1'b1;
                    state_q <= ST_TOKEN;
                end
            end else if (busy) begin
                case (state_q)
                    ST_TOKEN: begin
                        if ((selected_token_id >= {14'd0, vocab_rows_q}) ||
                            !token_mask_q[token_q]) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            error <= 1'b1;
                            status <= ERR_TOKEN;
                            state_q <= ST_IDLE;
                        end else begin
                            port_lane_q <= selected_token_id[1:0];
                            linear_rowblock_q <= selected_rowblock +
                                selected_port * rowblock_count;
                            state_q <= ST_GROUP;
                        end
                    end

                    ST_GROUP: begin
                        linear_group_q <= group_product;
                        state_q <= ST_BLOCK;
                    end

                    ST_BLOCK: begin
                        mem_req_addr_q <= table_addr_q +
                            ((weight_fmt_q == WEIGHT_Q1) ?
                             q1_group_offset : q2_group_offset);
                        q1_q <= 5'd0;
                        sub_q <= 2'd0;
                        state_q <= ST_SCALE_REQ;
                    end

                    ST_SCALE_REQ: begin
                        if (mem_req_fire)
                            state_q <= ST_SCALE_RSP;
                    end

                    ST_SCALE_RSP: begin
                        if (mem_rsp_fire) begin
                            if (mem_rsp_error) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= ERR_MEMORY;
                                state_q <= ST_IDLE;
                            end else if ((selected_slot[14:10] == 5'h1f) ||
                                ((weight_fmt_q == WEIGHT_Q2) &&
                                 (selected_slot[30:26] == 5'h1f))) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= ERR_SCALE;
                                state_q <= ST_IDLE;
                            end else begin
                                scale_lo_q <= selected_slot[15:0];
                                scale_hi_q <= selected_slot[31:16];
                                mem_req_addr_q <= mem_req_addr_q + 64'd16;
                                state_q <= ST_CODE0_REQ;
                            end
                        end
                    end

                    ST_CODE0_REQ: begin
                        if (mem_req_fire)
                            state_q <= ST_CODE0_RSP;
                    end

                    ST_CODE0_RSP: begin
                        if (mem_rsp_fire) begin
                            if (mem_rsp_error) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= ERR_MEMORY;
                                state_q <= ST_IDLE;
                            end else if (weight_fmt_q == WEIGHT_Q1) begin
                                q1_bits_q <= selected_slot;
                                chunk_q <= 3'd0;
                                mem_req_addr_q <= mem_req_addr_q + 64'd16;
                                state_q <= ST_EMIT;
                            end else begin
                                q2_code_lo_q <= selected_slot;
                                mem_req_addr_q <= mem_req_addr_q + 64'd16;
                                state_q <= ST_CODE1_REQ;
                            end
                        end
                    end

                    ST_CODE1_REQ: begin
                        if (mem_req_fire)
                            state_q <= ST_CODE1_RSP;
                    end

                    ST_CODE1_RSP: begin
                        if (mem_rsp_fire) begin
                            if (mem_rsp_error) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= ERR_MEMORY;
                                state_q <= ST_IDLE;
                            end else if (incoming_reserved) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                error <= 1'b1;
                                status <= ERR_RESERVED;
                                state_q <= ST_IDLE;
                            end else begin
                                q2_codes_q <= incoming_codes;
                                chunk_q <= 3'd0;
                                mem_req_addr_q <= mem_req_addr_q + 64'd16;
                                state_q <= ST_EMIT;
                            end
                        end
                    end

                    ST_EMIT: begin
                        if (out_fire) begin
                            if (chunk_q != 3'd7) begin
                                chunk_q <= chunk_q + 1'b1;
                            end else if (sub_q != 2'd3) begin
                                sub_q <= sub_q + 1'b1;
                                state_q <= ST_CODE0_REQ;
                            end else if ({1'b0, q1_q} + 1'b1 < q1_blocks_q) begin
                                q1_q <= q1_q + 1'b1;
                                sub_q <= 2'd0;
                                state_q <= ST_SCALE_REQ;
                            end else if ({1'b0, token_q} + 1'b1 < token_count_q) begin
                                token_q <= token_q + 1'b1;
                                q1_q <= 5'd0;
                                sub_q <= 2'd0;
                                state_q <= ST_TOKEN;
                            end else begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                state_q <= ST_IDLE;
                            end
                        end
                    end

                    ST_DRAIN: begin
                        if (mem_rsp_fire) begin
                            busy <= 1'b0;
                            state_q <= ST_IDLE;
                        end
                    end

                    default: begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= ERR_CONFIG;
                        state_q <= ST_IDLE;
                    end
                endcase
            end
        end
    end
endmodule

`default_nettype wire
