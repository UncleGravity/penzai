// Tile-8-resident, four-lane physical low-bit projection engine.
//
// Eight logical token contexts share each weight beat.  The activation service
// returns four token records at a time; a selector beat is retained while wave
// 0 (tokens 0..3) and wave 1 (tokens 4..7) pass through one  dot4 fabric.
// Empty waves are skipped, so decode uses exactly the same datapath with one
// active lane.  Activation responses are in request order.
//
// Projection service stream contract v1.
// Canonical signature SHA-256:
// bf2eef48663192c3b8b125cfce52349f4e93e1d9f3f494786a7b8cdaef67bf9f
// - Activation requests address one 32-coordinate Q8 block with a 9-bit block
//   index and one four-lane wave. Responses are in request order and contain four
//   lane-major {fp16 scale, 32 signed-Q8 values} records (4*272 bits).  A service
//   may enforce single-outstanding operation by deasserting act_req_ready.
// - Outputs are ordered rowblock -> active token -> logical row.  out_acc is an
//   exact signed fixed-window value whose numeric value is out_acc*2^out_emin;
//   out_valid suppresses M-tail rows and out_last marks the final live record.
//   The existing gemm_emit leaf performs the subsequent FP32 conversion.
// - clear synchronously abandons the run and every live handshake/pipeline
//   record.  A wrapper must also discard any activation response outstanding at
//   that boundary.  Payload memories are not reset and the first K record of a
//   restarted run replaces every accumulator digit.

`default_nettype none

module projection_engine (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  clear,

    input  wire                  start,
    input  wire [15:0]           model_spec_k,
    input  wire [17:0]           model_spec_m,
    input  wire [15:0]           model_spec_rowblocks,
    input  wire [1:0]            weight_fmt,
    input  wire signed [7:0]     emin,
    input  wire [7:0]            token_mask,
    output reg                   busy,
    output reg                   done,
    output reg                   error,

    input  wire [511:0]          w_data,
    input  wire                  w_valid,
    output wire                  w_ready,

    output wire                  act_req_valid,
    input  wire                  act_req_ready,
    output wire [8:0]            act_req_addr,
    output wire                  act_req_wave,
    input  wire                  act_rsp_valid,
    output wire                  act_rsp_ready,
    input  wire [1087:0]         act_rsp_data,

    output wire signed [103:0]   out_acc,
    output wire signed [7:0]     out_emin,
    output wire [2:0]            out_token,
    output wire [17:0]           out_row,
    output wire                  out_last,
    output wire                  out_valid,
    input  wire                  out_ready,

    output reg [31:0]            weight_beat_count,
    output reg [31:0]            wave_issue_count,
    output wire                  metrics_selector_full,
    output wire [2:0]            metrics_selector_level,
    output wire                  metrics_drain,
    output wire                  metrics_bank_wait
);
    localparam integer ROWS        = 16;
    localparam integer TOKENS      = 8;
    localparam integer LANES       = 4;
    localparam integer SUM_W       = 14;
    localparam integer SIG_W       = 12;
    localparam integer EXP_W       = 8;
    localparam integer ACC_W       = 104;
    localparam integer DIGITS      = 7;
    localparam integer BIN_W       = 26;
    localparam integer DIGIT_W     = DIGITS*BIN_W;
    localparam integer DOT_LATENCY = 12;
    localparam integer MUL_LATENCY = 4;
    localparam integer SERIAL_FIFO_DEPTH = 4;
    localparam integer SINGLE_TOKEN_SELECTOR_FIFO_DEPTH = 4;
    localparam integer SINGLE_TOKEN_SERIAL_CREDIT_LIMIT =
        DOT_LATENCY + SERIAL_FIFO_DEPTH + 2;

    wire pipe_rst_n = rst_n && !clear;

    localparam [1:0] WEIGHT_Q1 = 2'd1;
    localparam [1:0] WEIGHT_Q2 = 2'd2;

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_SCALE      = 4'd1;
    localparam [3:0] ST_Q1_WEIGHT  = 4'd2;
    localparam [3:0] ST_Q2_CODE0   = 4'd3;
    localparam [3:0] ST_Q2_CODE1   = 4'd4;
    localparam [3:0] ST_WAVES      = 4'd5;
    localparam [3:0] ST_DRAIN      = 4'd6;
    localparam [3:0] ST_WAIT_BANK  = 4'd7;
    localparam [3:0] ST_FINAL      = 4'd8;

    reg [3:0] state;
    reg [15:0] run_k;
    reg [17:0] run_m;
    reg [15:0] run_k_blocks;
    reg [15:0] run_rowblocks;
    reg [1:0] run_weight_fmt;
    reg signed [7:0] run_emin;
    reg [7:0] run_token_mask;

    reg [15:0] rowblock_idx;
    reg [15:0] q1_index;
    reg [1:0] sub_index;
    reg compute_bank;

    reg [ROWS*16-1:0] weight_scales_lo;
    reg [ROWS*16-1:0] weight_scales_hi;
    reg [511:0] q2_code0;
    reg [ROWS*32-1:0] held_weight_sign;
    reg [ROWS*32-1:0] held_weight_nonzero;

    // A one-token run can prepare selectors independently of the resident-Q8
    // round trip.  Entries remain live until their ordered activation response
    // is accepted, so separate pointers track weight intake, requests, and
    // responses without attaching a wide tag to the arena interface.
    wire single_token_fast_mode = (run_token_mask != 8'd0) &&
        ((run_token_mask & (run_token_mask - 8'd1)) == 8'd0);
    wire single_token_active_wave = |run_token_mask[7:4];
    wire [3:0] single_token_active_lane_mask = single_token_active_wave ?
        run_token_mask[7:4] : run_token_mask[3:0];
    wire [1:0] single_token_active_lane = single_token_active_lane_mask[0] ? 2'd0 :
                                single_token_active_lane_mask[1] ? 2'd1 :
                                single_token_active_lane_mask[2] ? 2'd2 : 2'd3;
    (* ram_style = "distributed" *) reg [ROWS*32-1:0]
        single_token_selector_weight_sign [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    (* ram_style = "distributed" *) reg [ROWS*32-1:0]
        single_token_selector_weight_nonzero [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    (* ram_style = "distributed" *) reg [ROWS*16-1:0]
        single_token_selector_weight_scales [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg [8:0] single_token_selector_addr [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg single_token_selector_wave [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg single_token_selector_clear [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg single_token_selector_bank [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg [LANES-1:0] single_token_selector_lane_mask [0:SINGLE_TOKEN_SELECTOR_FIFO_DEPTH-1];
    reg [1:0] single_token_selector_enq_ptr;
    reg [1:0] single_token_selector_req_ptr;
    reg [1:0] single_token_selector_rsp_ptr;
    reg [2:0] single_token_selector_fifo_count;
    reg [2:0] single_token_selector_unrequested_count;
    wire single_token_selector_fifo_full =
        (single_token_selector_fifo_count == SINGLE_TOKEN_SELECTOR_FIFO_DEPTH);
    assign metrics_selector_full = single_token_selector_fifo_full;
    assign metrics_selector_level = single_token_selector_fifo_count;
    assign metrics_drain = state == ST_DRAIN;
    assign metrics_bank_wait = state == ST_WAIT_BANK;

    wire [15:0] expected_model_spec_rowblocks =
        {2'b00, model_spec_m[17:4]} + {15'd0, (|model_spec_m[3:0])};
    wire model_spec_shape_valid =
        (model_spec_k != 16'd0) && (model_spec_k <= 16'd12288) &&
        (model_spec_k[6:0] == 7'd0) &&
        (model_spec_m != 18'd0) &&
        (model_spec_rowblocks == expected_model_spec_rowblocks) &&
        ((weight_fmt == WEIGHT_Q1) || (weight_fmt == WEIGHT_Q2)) &&
        (token_mask != 8'd0);

    wire single_token_selector_capture_state = (state == ST_Q1_WEIGHT) ||
                                     (state == ST_Q2_CODE1);
    assign w_ready = busy && ((state == ST_SCALE) ||
                              (state == ST_Q1_WEIGHT) ||
                              (state == ST_Q2_CODE0) ||
                              (state == ST_Q2_CODE1)) &&
                     (!single_token_fast_mode || !single_token_selector_capture_state ||
                      !single_token_selector_fifo_full);
    wire weight_fire = w_valid && w_ready;

    // Current Q2 code pair is decoded while the second half is presented, then
    // captured with the weight handshake for both token waves.
    wire [ROWS*32-1:0] q2_sign_comb;
    wire [ROWS*32-1:0] q2_nonzero_comb;
    genvar decode_row;
    generate
        for (decode_row = 0; decode_row < ROWS; decode_row = decode_row + 1) begin : g_q2_decode
            ternary_select32 u_decode (
                .codes({w_data[decode_row*32 +: 32],
                        q2_code0[decode_row*32 +: 32]}),
                .sign(q2_sign_comb[decode_row*32 +: 32]),
                .nonzero(q2_nonzero_comb[decode_row*32 +: 32])
            );
        end
    endgenerate

    wire q1_weight_capture = weight_fire && (state == ST_Q1_WEIGHT);
    wire q2_weight_capture = weight_fire && (state == ST_Q2_CODE1);
    wire single_token_selector_push = single_token_fast_mode &&
        (q1_weight_capture || q2_weight_capture);
    wire single_token_selector_pop;
    always @(posedge clk) begin
        if (q1_weight_capture) begin
            held_weight_sign <= w_data;
            held_weight_nonzero <= {ROWS*32{1'b1}};
        end else if (q2_weight_capture) begin
            held_weight_sign <= q2_sign_comb;
            held_weight_nonzero <= q2_nonzero_comb;
        end
    end

    // Tile-8 retains one selector until both active waves return. A single-token
    // the prepared-selector FIFO above: requests and responses advance through
    // the same entries in order while weight intake continues independently.
    wire [1:0] active_wave_mask = {(|run_token_mask[7:4]),
                                   (|run_token_mask[3:0])};
    reg [1:0] wave_requested;
    reg [1:0] wave_received;
    reg [4:0] serial_credit_used;
    wire serial_pop;
    wire request_wave0 = active_wave_mask[0] && !wave_requested[0];
    wire request_wave1 = active_wave_mask[1] && !wave_requested[1];
    wire selected_request_wave = !request_wave0 && request_wave1;
    wire tile8_serial_credit_available =
        (serial_credit_used < SERIAL_FIFO_DEPTH) || serial_pop;
    wire single_token_serial_credit_available =
        (serial_credit_used < SINGLE_TOKEN_SERIAL_CREDIT_LIMIT) || serial_pop;
    wire single_token_request_available =
        (single_token_selector_unrequested_count != 3'd0);
    assign act_req_valid = busy && (single_token_fast_mode ?
        (single_token_request_available && single_token_serial_credit_available) :
        ((state == ST_WAVES) && (request_wave0 || request_wave1) &&
         tile8_serial_credit_available));
    assign act_req_wave = single_token_fast_mode ?
        single_token_selector_wave[single_token_selector_req_ptr] : selected_request_wave;
    assign act_req_addr = single_token_fast_mode ?
        single_token_selector_addr[single_token_selector_req_ptr] :
        ({q1_index[6:0], 2'b00} + {7'd0, sub_index});
    wire act_req_fire = act_req_valid && act_req_ready;

    wire response_wave0 = wave_requested[0] && !wave_received[0];
    wire response_wave1 = wave_requested[1] && !wave_received[1];
    wire selected_response_wave = !response_wave0 && response_wave1;
    wire single_token_response_tag_valid =
        (single_token_selector_fifo_count > single_token_selector_unrequested_count);
    assign act_rsp_ready = rst_n && !clear && busy && (single_token_fast_mode ?
        single_token_response_tag_valid :
        ((state == ST_WAVES) && (response_wave0 || response_wave1)));
    wire act_rsp_fire = act_rsp_valid && act_rsp_ready;
    assign single_token_selector_pop = single_token_fast_mode && act_rsp_fire;
    wire [1:0] response_onehot = selected_response_wave ? 2'b10 : 2'b01;
    wire [1:0] received_after_response = wave_received | response_onehot;
    wire response_finishes_weight = !single_token_fast_mode && act_rsp_fire &&
        ((received_after_response & active_wave_mask) == active_wave_mask);

    wire issue_wave = single_token_fast_mode ?
        single_token_selector_wave[single_token_selector_rsp_ptr] : selected_response_wave;
    wire [3:0] issue_lane_mask = single_token_fast_mode ?
        single_token_selector_lane_mask[single_token_selector_rsp_ptr] :
        (selected_response_wave ? run_token_mask[7:4] :
                                  run_token_mask[3:0]);
    wire issue_clear = single_token_fast_mode ?
        single_token_selector_clear[single_token_selector_rsp_ptr] :
        ((q1_index == 16'd0) && (sub_index == 2'd0));
    wire issue_bank = single_token_fast_mode ?
        single_token_selector_bank[single_token_selector_rsp_ptr] : compute_bank;
    wire [ROWS*16-1:0] issue_weight_scales =
        single_token_fast_mode ? single_token_selector_weight_scales[single_token_selector_rsp_ptr] :
        (((run_weight_fmt == WEIGHT_Q2) && sub_index[1]) ?
         weight_scales_hi : weight_scales_lo);
    wire [ROWS*32-1:0] issue_weight_sign = single_token_fast_mode ?
        single_token_selector_weight_sign[single_token_selector_rsp_ptr] : held_weight_sign;
    wire [ROWS*32-1:0] issue_weight_nonzero = single_token_fast_mode ?
        single_token_selector_weight_nonzero[single_token_selector_rsp_ptr] :
        held_weight_nonzero;

    always @(posedge clk) begin
        if (single_token_selector_push) begin
            single_token_selector_weight_sign[single_token_selector_enq_ptr] <=
                q1_weight_capture ? w_data : q2_sign_comb;
            single_token_selector_weight_nonzero[single_token_selector_enq_ptr] <=
                q1_weight_capture ? {ROWS*32{1'b1}} : q2_nonzero_comb;
            single_token_selector_weight_scales[single_token_selector_enq_ptr] <=
                ((run_weight_fmt == WEIGHT_Q2) && sub_index[1]) ?
                weight_scales_hi : weight_scales_lo;
            single_token_selector_addr[single_token_selector_enq_ptr] <=
                {q1_index[6:0], 2'b00} + {7'd0, sub_index};
            single_token_selector_wave[single_token_selector_enq_ptr] <= single_token_active_wave;
            single_token_selector_clear[single_token_selector_enq_ptr] <=
                (q1_index == 16'd0) && (sub_index == 2'd0);
            single_token_selector_bank[single_token_selector_enq_ptr] <= compute_bank;
            single_token_selector_lane_mask[single_token_selector_enq_ptr] <=
                single_token_active_lane_mask;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || clear || (start && !busy)) begin
            single_token_selector_enq_ptr <= 2'd0;
            single_token_selector_req_ptr <= 2'd0;
            single_token_selector_rsp_ptr <= 2'd0;
            single_token_selector_fifo_count <= 3'd0;
            single_token_selector_unrequested_count <= 3'd0;
        end else begin
            if (single_token_selector_push)
                single_token_selector_enq_ptr <= single_token_selector_enq_ptr + 1'b1;
            if (single_token_fast_mode && act_req_fire)
                single_token_selector_req_ptr <= single_token_selector_req_ptr + 1'b1;
            if (single_token_selector_pop)
                single_token_selector_rsp_ptr <= single_token_selector_rsp_ptr + 1'b1;

            case ({single_token_selector_push, single_token_selector_pop})
                2'b10: single_token_selector_fifo_count <=
                    single_token_selector_fifo_count + 1'b1;
                2'b01: single_token_selector_fifo_count <=
                    single_token_selector_fifo_count - 1'b1;
                default: ;
            endcase
            case ({single_token_selector_push, single_token_fast_mode && act_req_fire})
                2'b10: single_token_selector_unrequested_count <=
                    single_token_selector_unrequested_count + 1'b1;
                2'b01: single_token_selector_unrequested_count <=
                    single_token_selector_unrequested_count - 1'b1;
                default: ;
            endcase
        end
    end

    // The resident arena's BRAM output must not drive the first DSP stage
    // directly.  This full-throughput input register samples one response and
    // its ownership metadata per cycle; control still retires the external
    // handshake immediately, preserving request and selector cadence.  Payload
    // registers are deliberately free of clear/reset fanout.  Their valid bit
    // is the sole ownership boundary and quarantines stale data on abort.
    reg act_rsp_pipe_valid_q;
    (* keep = "true", max_fanout = 4 *) reg [1087:0] act_rsp_pipe_data_q;
    reg act_rsp_pipe_bank_q;
    reg act_rsp_pipe_wave_q;
    reg act_rsp_pipe_clear_q;
    reg [LANES-1:0] act_rsp_pipe_lane_mask_q;
    reg [ROWS*16-1:0] act_rsp_pipe_weight_scales_q;
    reg [ROWS*32-1:0] act_rsp_pipe_weight_sign_q;
    reg [ROWS*32-1:0] act_rsp_pipe_weight_nonzero_q;
    wire dot_issue_valid = rst_n && !clear && act_rsp_pipe_valid_q;

    always @(posedge clk) begin
        if (!rst_n || clear || (start && !busy))
            act_rsp_pipe_valid_q <= 1'b0;
        else
            act_rsp_pipe_valid_q <= act_rsp_fire;

        act_rsp_pipe_data_q <= act_rsp_data;
        act_rsp_pipe_bank_q <= issue_bank;
        act_rsp_pipe_wave_q <= issue_wave;
        act_rsp_pipe_clear_q <= issue_clear;
        act_rsp_pipe_lane_mask_q <= issue_lane_mask;
        act_rsp_pipe_weight_scales_q <= issue_weight_scales;
        act_rsp_pipe_weight_sign_q <= issue_weight_sign;
        act_rsp_pipe_weight_nonzero_q <= issue_weight_nonzero;
    end

    wire [1023:0] act_rsp_values_q;
    wire [63:0] act_rsp_scales_q;
    genvar act_lane;
    generate
        for (act_lane = 0; act_lane < LANES; act_lane = act_lane + 1) begin : g_act_unpack
            assign act_rsp_values_q[act_lane*256 +: 256] =
                act_rsp_pipe_data_q[act_lane*272 +: 256];
            assign act_rsp_scales_q[act_lane*16 +: 16] =
                act_rsp_pipe_data_q[act_lane*272 + 256 +: 16];
        end
    endgenerate

    wire dot_out_valid;
    wire [ROWS*LANES*SUM_W-1:0] dot_sums;
     dot4 u_dot (
        .clk(clk),
        .rst_n(pipe_rst_n),
        .in_valid(dot_issue_valid),
        .weight_sign(act_rsp_pipe_weight_sign_q),
        .weight_nonzero(act_rsp_pipe_weight_nonzero_q),
        .acts_flat(act_rsp_values_q),
        .out_valid(dot_out_valid),
        .sums_flat(dot_sums)
    );

    // Metadata follows its dot record through the twelve-cycle SIMD fabric.
    localparam integer DOT_META_W = 1 + 1 + 1 + LANES + ROWS*16 + LANES*16;
    wire [DOT_META_W-1:0] dot_meta_in = {
        act_rsp_pipe_bank_q,
        act_rsp_pipe_wave_q,
        act_rsp_pipe_clear_q,
        act_rsp_pipe_lane_mask_q,
        act_rsp_pipe_weight_scales_q,
        act_rsp_scales_q
    };
    wire [DOT_META_W-1:0] dot_meta_out;
     delay #(.WIDTH(DOT_META_W), .DEPTH(DOT_LATENCY)) u_dot_meta_delay (
        .clk(clk),
        .din(dot_meta_in),
        .dout(dot_meta_out)
    );
    // Dot records are credit-limited before entering the fixed-latency fabric.
    // Four records cover a worst-case tile-8 burst while the row-shared post-dot
    // path serializes its four physical lanes.  single-token consumes one active lane per
    // record and carries enough request credit to fill the fixed dot pipeline.
    // Payload RAM is not reset; the count and pointers define live entries.
    localparam integer DOT_DATA_W = ROWS*LANES*SUM_W;
    localparam integer DOT_PACKET_W = DOT_META_W + DOT_DATA_W;
    (* ram_style = "distributed" *) reg [DOT_PACKET_W-1:0] dot_fifo
        [0:SERIAL_FIFO_DEPTH-1];
    reg [1:0] dot_fifo_wr_ptr;
    reg [1:0] dot_fifo_rd_ptr;
    reg [2:0] dot_fifo_count;
    reg serial_active;
    reg [1:0] serial_lane;
    reg [DOT_PACKET_W-1:0] serial_packet;
    wire dot_fifo_push = dot_out_valid;
    wire serial_record_last = single_token_fast_mode || (serial_lane == 2'd3);
    assign serial_pop = (dot_fifo_count != 0) &&
                        (!serial_active || serial_record_last);

    always @(posedge clk) begin
        if (dot_fifo_push)
            dot_fifo[dot_fifo_wr_ptr] <= {dot_meta_out, dot_sums};
    end

    always @(posedge clk) begin
        if (!rst_n || clear || (start && !busy)) begin
            dot_fifo_wr_ptr <= 2'd0;
            dot_fifo_rd_ptr <= 2'd0;
            dot_fifo_count <= 3'd0;
            serial_active <= 1'b0;
            serial_lane <= 2'd0;
            serial_credit_used <= 5'd0;
        end else begin
            case ({act_req_fire, serial_pop})
                2'b10: serial_credit_used <= serial_credit_used + 1'b1;
                2'b01: serial_credit_used <= serial_credit_used - 1'b1;
                default: ;
            endcase

            if (dot_fifo_push)
                dot_fifo_wr_ptr <= dot_fifo_wr_ptr + 1'b1;
            if (serial_pop)
                dot_fifo_rd_ptr <= dot_fifo_rd_ptr + 1'b1;
            case ({dot_fifo_push, serial_pop})
                2'b10: dot_fifo_count <= dot_fifo_count + 1'b1;
                2'b01: dot_fifo_count <= dot_fifo_count - 1'b1;
                default: ;
            endcase

            if (serial_pop) begin
                serial_packet <= dot_fifo[dot_fifo_rd_ptr];
                serial_active <= 1'b1;
                serial_lane <= single_token_fast_mode ? single_token_active_lane : 2'd0;
            end else if (serial_active) begin
                if (serial_record_last)
                    serial_active <= 1'b0;
                else
                    serial_lane <= serial_lane + 1'b1;
            end
        end
    end

    wire [DOT_DATA_W-1:0] serial_dot_sums =
        serial_packet[DOT_DATA_W-1:0];
    wire [DOT_META_W-1:0] serial_meta =
        serial_packet[DOT_PACKET_W-1:DOT_DATA_W];
    wire [LANES*16-1:0] serial_act_scales =
        serial_meta[LANES*16-1:0];
    wire [ROWS*16-1:0] serial_weight_scales =
        serial_meta[LANES*16 +: ROWS*16];
    wire [LANES-1:0] serial_lane_mask =
        serial_meta[LANES*16 + ROWS*16 +: LANES];
    wire serial_clear = serial_meta[LANES*16 + ROWS*16 + LANES];
    wire serial_wave = serial_meta[LANES*16 + ROWS*16 + LANES + 1];
    wire serial_bank = serial_meta[DOT_META_W-1];
    wire [2:0] serial_token = {serial_wave, serial_lane};
    wire serial_lane_active = serial_lane_mask[serial_lane];
    wire [15:0] serial_act_scale =
        serial_act_scales[serial_lane*16 +: 16];

    wire signed [ROWS*SIG_W-1:0] ws_sig_flat;
    wire signed [ROWS*EXP_W-1:0] ws_exp_flat;
    genvar scale_row;
    generate
        for (scale_row = 0; scale_row < ROWS; scale_row = scale_row + 1) begin : g_ws_decode
            gemm_f16_decompose #(.SIG_W(SIG_W), .EXP_W(EXP_W)) u_ws (
                .f16(serial_weight_scales[scale_row*16 +: 16]),
                .sig(ws_sig_flat[scale_row*SIG_W +: SIG_W]),
                .e(ws_exp_flat[scale_row*EXP_W +: EXP_W])
            );
        end
    endgenerate

    wire signed [SIG_W-1:0] as_sig;
    wire signed [EXP_W-1:0] as_exp;
    gemm_f16_decompose #(.SIG_W(SIG_W), .EXP_W(EXP_W)) u_as (
        .f16(serial_act_scale),
        .sig(as_sig),
        .e(as_exp)
    );

    // One two-DSP scale/align path per output row consumes token lanes 0..3
    // over four clocks for tile-8. A single-token request selects its sole live lane and advances to
    // the next record immediately.  The dot itself remains four-lane hardware.
    wire [ROWS-1:0] mul_valid_flat;
    wire signed [ROWS*(EXP_W+1)-1:0] digit_coarse_flat;
    wire [ROWS*64-1:0] digit_chunks_flat;
    genvar mul_row;
    generate
        for (mul_row = 0; mul_row < ROWS; mul_row = mul_row + 1) begin : g_mul_row
            wire signed [EXP_W-1:0] product_exp =
                $signed(ws_exp_flat[mul_row*EXP_W +: EXP_W]) +
                $signed(as_exp);
            wire signed [SUM_W-1:0] selected_dot_sum =
                (serial_lane == 2'd0) ? $signed(serial_dot_sums[
                    (mul_row*LANES + 0)*SUM_W +: SUM_W]) :
                (serial_lane == 2'd1) ? $signed(serial_dot_sums[
                    (mul_row*LANES + 1)*SUM_W +: SUM_W]) :
                (serial_lane == 2'd2) ? $signed(serial_dot_sums[
                    (mul_row*LANES + 2)*SUM_W +: SUM_W]) :
                    $signed(serial_dot_sums[
                    (mul_row*LANES + 3)*SUM_W +: SUM_W]);
             digit_mul #(
                .SIG_W(SIG_W),
                .SUM_W(SUM_W),
                .EXP_W(EXP_W),
                .DIGITS(DIGITS),
                .BIN_W(BIN_W)
            ) u_mul (
                .clk(clk),
                .rst_n(pipe_rst_n),
                .valid_in(serial_active),
                .ws_sig($signed(ws_sig_flat[mul_row*SIG_W +: SIG_W])),
                .as_sig(as_sig),
                .p_exp(product_exp),
                .emin(run_emin),
                .dot_sum(selected_dot_sum),
                .valid_out(mul_valid_flat[mul_row]),
                .coarse_digit(digit_coarse_flat[
                    mul_row*(EXP_W+1) +: (EXP_W+1)]),
                .fine_chunks(digit_chunks_flat[mul_row*64 +: 64])
            );
        end
    endgenerate

    wire mul_valid = mul_valid_flat[0];
    localparam integer MUL_TAG_W = 1 + 3 + 1 + 1 + 1;
    wire [MUL_TAG_W-1:0] mul_tag_in = {
        serial_bank, serial_token, serial_clear, serial_record_last,
        serial_lane_active
    };
    wire [MUL_TAG_W-1:0] mul_tag_out;
     delay #(.WIDTH(MUL_TAG_W), .DEPTH(MUL_LATENCY)) u_mul_tag_delay (
        .clk(clk),
        .din(mul_tag_in),
        .dout(mul_tag_out)
    );
    wire mul_lane_active = mul_tag_out[0];
    wire mul_record_last = mul_tag_out[1];
    wire mul_clear = mul_tag_out[2];
    wire [2:0] mul_token = mul_tag_out[5:3];
    wire mul_bank = mul_tag_out[6];

    reg drain_bank;
    reg [2:0] drain_token_q;
    reg [3:0] drain_row_q;
    wire [ROWS*DIGIT_W-1:0] cell_drain_flat;
    wire [ROWS-1:0] cell_update_pending;
    genvar cell_row;
    generate
        for (cell_row = 0; cell_row < ROWS; cell_row = cell_row + 1) begin : g_cell_row
             digit_cell #(
                .DIGITS(DIGITS),
                .BIN_W(BIN_W)
            ) u_cell (
                .clk(clk),
                .rst_n(rst_n),
                .clear(clear),
                .update_valid(mul_valid && mul_lane_active),
                .update_clear(mul_clear),
                .update_bank(mul_bank),
                .update_token(mul_token),
                .update_coarse(digit_coarse_flat[
                    cell_row*(EXP_W+1) +: (EXP_W+1)]),
                .update_chunks(digit_chunks_flat[cell_row*64 +: 64]),
                .drain_bank(drain_bank),
                .drain_token(drain_token_q),
                .drain_digits(cell_drain_flat[
                    cell_row*DIGIT_W +: DIGIT_W]),
                .update_pending(cell_update_pending[cell_row])
            );
        end
    endgenerate

    // Token selection is absorbed by each row's accumulator address, leaving
    // only the sixteen-row drain mux in front of the normalizer.
    wire [DIGIT_W-1:0] drain_row_digits [0:ROWS-1];
    genvar mux_row;
    generate
        for (mux_row = 0; mux_row < ROWS; mux_row = mux_row + 1) begin : g_drain_row
            assign drain_row_digits[mux_row] =
                cell_drain_flat[mux_row*DIGIT_W +: DIGIT_W];
        end
    endgenerate
    reg [DIGIT_W-1:0] drain_digits_mux;
    always @(*) begin
        case (drain_row_q)
            4'd0:  drain_digits_mux = drain_row_digits[0];
            4'd1:  drain_digits_mux = drain_row_digits[1];
            4'd2:  drain_digits_mux = drain_row_digits[2];
            4'd3:  drain_digits_mux = drain_row_digits[3];
            4'd4:  drain_digits_mux = drain_row_digits[4];
            4'd5:  drain_digits_mux = drain_row_digits[5];
            4'd6:  drain_digits_mux = drain_row_digits[6];
            4'd7:  drain_digits_mux = drain_row_digits[7];
            4'd8:  drain_digits_mux = drain_row_digits[8];
            4'd9:  drain_digits_mux = drain_row_digits[9];
            4'd10: drain_digits_mux = drain_row_digits[10];
            4'd11: drain_digits_mux = drain_row_digits[11];
            4'd12: drain_digits_mux = drain_row_digits[12];
            4'd13: drain_digits_mux = drain_row_digits[13];
            4'd14: drain_digits_mux = drain_row_digits[14];
            default: drain_digits_mux = drain_row_digits[15];
        endcase
    end

    reg [5:0] inflight_waves;
    wire mul_record_done = mul_valid && mul_record_last;
    always @(posedge clk) begin
        if (!rst_n || clear || (start && !busy)) begin
            inflight_waves <= 6'd0;
        end else begin
            case ({act_rsp_fire, mul_record_done})
                2'b10: inflight_waves <= inflight_waves + 1'b1;
                2'b01: inflight_waves <= inflight_waves - 1'b1;
                default: ;
            endcase
        end
    end

    function [2:0] first_active_token;
        input [7:0] mask;
        begin
            casez (mask)
                8'b???????1: first_active_token = 3'd0;
                8'b??????10: first_active_token = 3'd1;
                8'b?????100: first_active_token = 3'd2;
                8'b????1000: first_active_token = 3'd3;
                8'b???10000: first_active_token = 3'd4;
                8'b??100000: first_active_token = 3'd5;
                8'b?1000000: first_active_token = 3'd6;
                default:     first_active_token = 3'd7;
            endcase
        end
    endfunction

    function [2:0] next_active_token;
        input [7:0] mask;
        input [2:0] current;
        integer token;
        reg found;
        begin
            next_active_token = current;
            found = 1'b0;
            for (token = 0; token < TOKENS; token = token + 1) begin
                if (!found && (token > current) && mask[token]) begin
                    next_active_token = token[2:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    function [2:0] last_active_token;
        input [7:0] mask;
        integer token;
        begin
            last_active_token = 3'd0;
            for (token = 0; token < TOKENS; token = token + 1)
                if (mask[token]) last_active_token = token[2:0];
        end
    endfunction

    reg drain_active;
    reg pending_valid;
    reg pending_bank;
    reg [15:0] bank_rowblock0;
    reg [15:0] bank_rowblock1;
    wire [15:0] drain_rowblock = drain_bank ?
        bank_rowblock1 : bank_rowblock0;
    wire [17:0] drain_rowblock_base =
        {drain_rowblock[13:0], 4'b0000};
    wire [4:0] drain_valid_rows =
        ((drain_rowblock_base + 18'd16) > run_m) ?
        {1'b0, run_m[3:0]} : 5'd16;
    wire [2:0] final_active_token = last_active_token(run_token_mask);
    wire drain_last_row =
        ({1'b0, drain_row_q} + 5'd1 == drain_valid_rows);
    wire drain_last_token = (drain_token_q == final_active_token);
    wire drain_last_rowblock =
        (drain_rowblock + 16'd1 == run_rowblocks);

    localparam integer NORM_META_W = 1 + 3 + 18 + 8;
    wire [NORM_META_W-1:0] drain_meta_comb = {
        drain_last_row && drain_last_token && drain_last_rowblock,
        drain_token_q,
        drain_rowblock_base + {14'd0, drain_row_q},
        run_emin
    };
    wire norm_in_ready;
    wire [NORM_META_W-1:0] norm_meta_out;
    wire norm_busy;
    reg drain_stage_valid;
    reg [DIGIT_W-1:0] drain_stage_digits;
    reg [NORM_META_W-1:0] drain_stage_meta;
    wire drain_stage_ready = !drain_stage_valid || norm_in_ready;
    wire drain_fire = drain_active && drain_stage_ready;

    // Capture the asynchronous row mux and rowblock metadata before the
    // backpressureable normalizer.  The stage is elastic, so drain throughput
    // remains one row per cycle while the long metadata/control path is cut.
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            drain_stage_valid <= 1'b0;
        end else if (drain_stage_ready) begin
            drain_stage_valid <= drain_active;
            if (drain_active) begin
                drain_stage_digits <= drain_digits_mux;
                drain_stage_meta <= drain_meta_comb;
            end
        end
    end

     digit_normalize #(
        .DIGITS(DIGITS),
        .BIN_W(BIN_W),
        .ACC_W(ACC_W),
        .META_W(NORM_META_W)
    ) u_normalize (
        .clk(clk),
        .rst_n(pipe_rst_n),
        .in_valid(drain_stage_valid),
        .in_ready(norm_in_ready),
        .in_digits(drain_stage_digits),
        .in_meta(drain_stage_meta),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_acc(out_acc),
        .out_meta(norm_meta_out),
        .busy(norm_busy)
    );
    assign out_emin = norm_meta_out[7:0];
    assign out_row = norm_meta_out[25:8];
    assign out_token = norm_meta_out[28:26];
    assign out_last = norm_meta_out[29];

    wire last_subblock = (sub_index == 2'd3);
    wire last_q1_block = (q1_index + 16'd1 == run_k_blocks);
    wire last_compute_rowblock =
        (rowblock_idx + 16'd1 == run_rowblocks);
    integer scale_capture_row;
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            state <= ST_IDLE;
            run_k <= 16'd0;
            run_m <= 18'd0;
            run_k_blocks <= 16'd0;
            run_rowblocks <= 16'd0;
            run_weight_fmt <= WEIGHT_Q1;
            run_emin <= 8'sd0;
            run_token_mask <= 8'd0;
            rowblock_idx <= 16'd0;
            q1_index <= 16'd0;
            sub_index <= 2'd0;
            compute_bank <= 1'b0;
            wave_requested <= 2'b00;
            wave_received <= 2'b00;
            drain_active <= 1'b0;
            drain_bank <= 1'b0;
            drain_token_q <= 3'd0;
            drain_row_q <= 4'd0;
            pending_valid <= 1'b0;
            pending_bank <= 1'b0;
            bank_rowblock0 <= 16'd0;
            bank_rowblock1 <= 16'd0;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            weight_beat_count <= 32'd0;
            wave_issue_count <= 32'd0;
        end else begin
            done <= 1'b0;

            if (start && !busy) begin
                run_k <= model_spec_k;
                run_m <= model_spec_m;
                run_k_blocks <= model_spec_k >> 7;
                run_rowblocks <= model_spec_rowblocks;
                run_weight_fmt <= weight_fmt;
                run_emin <= emin;
                run_token_mask <= token_mask;
                rowblock_idx <= 16'd0;
                q1_index <= 16'd0;
                sub_index <= 2'd0;
                compute_bank <= 1'b0;
                wave_requested <= 2'b00;
                wave_received <= 2'b00;
                drain_active <= 1'b0;
                drain_bank <= 1'b0;
                drain_token_q <= first_active_token(token_mask);
                drain_row_q <= 4'd0;
                pending_valid <= 1'b0;
                pending_bank <= 1'b0;
                weight_beat_count <= 32'd0;
                wave_issue_count <= 32'd0;
                error <= !model_spec_shape_valid;
                if (model_spec_shape_valid) begin
                    busy <= 1'b1;
                    state <= ST_SCALE;
                end else begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= ST_IDLE;
                end
            end else if (busy) begin
                if (weight_fire)
                    weight_beat_count <= weight_beat_count + 1'b1;
                if (act_rsp_fire)
                    wave_issue_count <= wave_issue_count + 1'b1;

                if (drain_fire) begin
                    if (!drain_last_row) begin
                        drain_row_q <= drain_row_q + 1'b1;
                    end else if (!drain_last_token) begin
                        drain_token_q <= next_active_token(
                            run_token_mask, drain_token_q);
                        drain_row_q <= 4'd0;
                    end else begin
                        drain_active <= 1'b0;
                    end
                end

                case (state)
                    ST_SCALE: begin
                        if (weight_fire) begin
                            for (scale_capture_row = 0;
                                 scale_capture_row < ROWS;
                                 scale_capture_row = scale_capture_row + 1) begin
                                weight_scales_lo[scale_capture_row*16 +: 16] <=
                                    w_data[scale_capture_row*32 +: 16];
                                weight_scales_hi[scale_capture_row*16 +: 16] <=
                                    w_data[scale_capture_row*32 + 16 +: 16];
                            end
                            state <= (run_weight_fmt == WEIGHT_Q2) ?
                                     ST_Q2_CODE0 : ST_Q1_WEIGHT;
                        end
                    end

                    ST_Q1_WEIGHT: begin
                        if (weight_fire) begin
                            if (single_token_fast_mode) begin
                                if (last_subblock) begin
                                    sub_index <= 2'd0;
                                    if (last_q1_block) begin
                                        state <= ST_DRAIN;
                                    end else begin
                                        q1_index <= q1_index + 1'b1;
                                        state <= ST_SCALE;
                                    end
                                end else begin
                                    sub_index <= sub_index + 1'b1;
                                    state <= ST_Q1_WEIGHT;
                                end
                            end else begin
                                wave_requested <= 2'b00;
                                wave_received <= 2'b00;
                                state <= ST_WAVES;
                            end
                        end
                    end

                    ST_Q2_CODE0: begin
                        if (weight_fire) begin
                            q2_code0 <= w_data;
                            state <= ST_Q2_CODE1;
                        end
                    end

                    ST_Q2_CODE1: begin
                        if (weight_fire) begin
                            if (single_token_fast_mode) begin
                                if (last_subblock) begin
                                    sub_index <= 2'd0;
                                    if (last_q1_block) begin
                                        state <= ST_DRAIN;
                                    end else begin
                                        q1_index <= q1_index + 1'b1;
                                        state <= ST_SCALE;
                                    end
                                end else begin
                                    sub_index <= sub_index + 1'b1;
                                    state <= ST_Q2_CODE0;
                                end
                            end else begin
                                wave_requested <= 2'b00;
                                wave_received <= 2'b00;
                                state <= ST_WAVES;
                            end
                        end
                    end

                    ST_WAVES: begin
                        if (act_req_fire)
                            wave_requested[selected_request_wave] <= 1'b1;
                        if (act_rsp_fire)
                            wave_received[selected_response_wave] <= 1'b1;

                        if (response_finishes_weight) begin
                            wave_requested <= 2'b00;
                            wave_received <= 2'b00;
                            if (last_subblock) begin
                                sub_index <= 2'd0;
                                if (last_q1_block) begin
                                    state <= ST_DRAIN;
                                end else begin
                                    q1_index <= q1_index + 1'b1;
                                    state <= ST_SCALE;
                                end
                            end else begin
                                sub_index <= sub_index + 1'b1;
                                state <= (run_weight_fmt == WEIGHT_Q2) ?
                                         ST_Q2_CODE0 : ST_Q1_WEIGHT;
                            end
                        end
                    end

                    ST_DRAIN: begin
                        // The counter reaches zero one edge after the final
                        // digit write.  The completed bank can then drain at
                        // one output/cycle while the other bank computes.
                        if ((single_token_selector_fifo_count == 3'd0) &&
                            (single_token_selector_unrequested_count == 3'd0) &&
                            (inflight_waves == 6'd0) && !mul_valid &&
                            !cell_update_pending[0]) begin
                            if (compute_bank)
                                bank_rowblock1 <= rowblock_idx;
                            else
                                bank_rowblock0 <= rowblock_idx;

                            if (!drain_active) begin
                                drain_active <= 1'b1;
                                drain_bank <= compute_bank;
                                drain_token_q <= first_active_token(
                                    run_token_mask);
                                drain_row_q <= 4'd0;
                                if (last_compute_rowblock) begin
                                    state <= ST_FINAL;
                                end else begin
                                    compute_bank <= !compute_bank;
                                    rowblock_idx <= rowblock_idx + 1'b1;
                                    q1_index <= 16'd0;
                                    sub_index <= 2'd0;
                                    state <= ST_SCALE;
                                end
                            end else begin
                                pending_valid <= 1'b1;
                                pending_bank <= compute_bank;
                                state <= last_compute_rowblock ?
                                         ST_FINAL : ST_WAIT_BANK;
                            end
                        end
                    end

                    ST_WAIT_BANK: begin
                        // The old drain bank is reusable once its records have
                        // entered the normalizer; downstream output may still
                        // be active because the digit payload is now captured.
                        if (!drain_active && pending_valid) begin
                            drain_active <= 1'b1;
                            drain_bank <= pending_bank;
                            drain_token_q <= first_active_token(
                                run_token_mask);
                            drain_row_q <= 4'd0;
                            pending_valid <= 1'b0;
                            compute_bank <= !pending_bank;
                            rowblock_idx <= rowblock_idx + 1'b1;
                            q1_index <= 16'd0;
                            sub_index <= 2'd0;
                            state <= ST_SCALE;
                        end
                    end

                    ST_FINAL: begin
                        if (!drain_active) begin
                            if (pending_valid) begin
                                drain_active <= 1'b1;
                                drain_bank <= pending_bank;
                                drain_token_q <= first_active_token(
                                    run_token_mask);
                                drain_row_q <= 4'd0;
                                pending_valid <= 1'b0;
                            end else if (!drain_stage_valid && !norm_busy) begin
                                busy <= 1'b0;
                                done <= 1'b1;
                                state <= ST_IDLE;
                            end
                        end
                    end

                    default: begin
                        error <= 1'b1;
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                endcase
            end
        end
    end

`ifdef FORMAL
    integer formal_lane;
    reg formal_weight_payload_fresh;
    reg [2:0] formal_fifo_fresh_count;
    reg formal_act_rsp_pipe_fresh;
    always @(posedge clk) begin
        if (!rst_n || clear || (start && !busy)) begin
            formal_weight_payload_fresh <= 1'b0;
            formal_fifo_fresh_count <= 3'd0;
            formal_act_rsp_pipe_fresh <= 1'b0;
        end else begin
            formal_act_rsp_pipe_fresh <= act_rsp_fire;
            if (!single_token_fast_mode &&
                (q1_weight_capture || q2_weight_capture))
                formal_weight_payload_fresh <= 1'b1;
            else if (response_finishes_weight)
                formal_weight_payload_fresh <= 1'b0;

            case ({dot_fifo_push, serial_pop})
                2'b10: formal_fifo_fresh_count <=
                    formal_fifo_fresh_count + 1'b1;
                2'b01: formal_fifo_fresh_count <=
                    formal_fifo_fresh_count - 1'b1;
                default: ;
            endcase
        end

        if (rst_n && !clear && busy) begin
            if (act_rsp_fire) assert(issue_lane_mask != 4'd0);
            if (act_rsp_fire && !single_token_fast_mode)
                assert(formal_weight_payload_fresh);
            if (single_token_selector_push)
                assert(!single_token_selector_fifo_full);
            if (single_token_fast_mode && act_req_fire)
                assert(single_token_selector_unrequested_count != 0);
            if (single_token_selector_pop)
                assert(single_token_response_tag_valid);
            assert(single_token_selector_unrequested_count <=
                   single_token_selector_fifo_count);
            assert(single_token_selector_fifo_count <= SINGLE_TOKEN_SELECTOR_FIFO_DEPTH);
            if (dot_issue_valid) assert(formal_act_rsp_pipe_fresh);
            assert(serial_credit_used <= (single_token_fast_mode ?
                SINGLE_TOKEN_SERIAL_CREDIT_LIMIT : SERIAL_FIFO_DEPTH));
            assert(dot_fifo_count <= SERIAL_FIFO_DEPTH);
            if (dot_fifo_push && !serial_pop)
                assert(dot_fifo_count < SERIAL_FIFO_DEPTH);
            if (serial_pop) begin
                assert(serial_credit_used != 0);
                assert(formal_fifo_fresh_count != 0);
            end
            cover(dot_fifo_push && serial_pop);
            for (formal_lane = 1; formal_lane < ROWS;
                 formal_lane = formal_lane + 1) begin
                assert(mul_valid_flat[formal_lane] == mul_valid);
                assert(cell_update_pending[formal_lane] ==
                       cell_update_pending[0]);
            end
            if (out_valid) begin
                assert(run_token_mask[out_token]);
                assert(out_row < run_m);
            end
        end
        if (rst_n) begin
            cover(act_rsp_valid && !act_rsp_ready);
            cover(clear && act_rsp_pipe_valid_q);
        end
    end
`endif
endmodule

`default_nettype wire
