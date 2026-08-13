// section_ffn_pairer - pair streamed GATE groups with resident UP groups.
//
// GEMM produces each 32-row block as two native 16-row sweeps:
//
//   token0 g0, g1; ...; tokenN g0, g1;
//   token0 g2, g3; ...; tokenN g2, g3
//
// The lower groups are retained in a small block-RAM reorder store. After each
// token's upper pair arrives, the store replays g0..g3 in canonical order. Each
// replayed GATE group issues the matching X1/UP scratch request and the registered
// response is serialized as eight scalar pairs for section_swiglu. Input TLAST is
// the logical 32-scalar boundary and is therefore required on every group three.
//
// The reorder read, scratch request, response FIFO, and scalar output are separate
// registered elastic boundaries. Payload memories are intentionally not reset.

`default_nettype none

module section_ffn_pairer (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          start_valid,
    output wire          start_ready,
    input  wire [2:0]    start_tokens,
    input  wire [8:0]    start_blocks,
    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,

    // One complete eight-scalar GATE group with an explicit logical tag.
    input  wire [255:0]  s_axis_tdata,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,
    input  wire [1:0]    s_axis_token,
    input  wire [8:0]    s_axis_block,
    input  wire [1:0]    s_axis_group,

    // Directly compatible with section_f32_scratch's read request/response.
    // The role is fixed to X1 (UP); group zero denotes rows 0..7.
    output wire          rd_req_valid,
    input  wire          rd_req_ready,
    output wire [1:0]    rd_req_role,
    output wire [2:0]    rd_req_token,
    output wire [10:0]   rd_req_group,
    input  wire          rd_rsp_valid,
    output wire          rd_rsp_ready,
    input  wire [255:0]  rd_rsp_data,
    input  wire          rd_rsp_error,

    // Directly compatible with section_swiglu's input stream.
    output wire          out_valid,
    input  wire          out_ready,
    output wire [31:0]   out_gate,
    output wire [31:0]   out_up,
    output wire          out_last
`ifdef FORMAL
    , output wire [1:0]  formal_gate_count
    , output wire [1:0]  formal_pair_count
    , output wire        formal_read_inflight
    , output wire        formal_orphan
    , output wire [2:0]  formal_emit_lane
`endif
);
    localparam [1:0] ROLE_X1 = 2'd2;
    localparam [8:0] BLOCK_CAPACITY = 9'd384;

    reg busy_q;
    reg [2:0] run_tokens_q;
    reg [8:0] run_blocks_q;
    reg [8:0] expect_block_q;
    reg expect_upper_q;
    reg [1:0] expect_token_q;
    reg expect_group_odd_q;
    reg input_complete_q;

    assign busy = busy_q;
    wire start_shape_ok = (start_tokens != 3'd0) &&
                          (start_tokens <= 3'd4) &&
                          (start_blocks != 9'd0) &&
                          (start_blocks <= BLOCK_CAPACITY);

    // An aborted request can still return from scratch. Do not start a new run
    // until that response has been consumed, or it could be mistaken for new data.
    reg orphan_q;
    assign start_ready = rst_n && !abort_run && !busy_q && !orphan_q && !rd_rsp_valid;
    wire start_accept = start_valid && start_ready;

    // ---- Native-order ingress and registered block-RAM reorder ----

    // Four groups per token x four tokens. Only lower groups persist for an
    // extended interval, but placing all groups in one shallow RAM avoids a wide
    // upper-half register bank and keeps the canonical replay mechanism uniform.
    (* ram_style = "block" *) reg [255:0] reorder_mem [0:15];

    reg replay_active_q;
    reg [8:0] replay_block_q;
    reg [1:0] replay_token_q;
    reg [1:0] replay_group_q;

    reg reorder_rd_valid_q;
    reg [255:0] reorder_rd_data_q;
    reg [8:0] reorder_rd_block_q;
    reg [1:0] reorder_rd_token_q;
    reg [1:0] reorder_rd_group_q;

    wire [1:0] expected_group = {expect_upper_q, expect_group_odd_q};
    assign s_axis_tready = rst_n && busy_q && !abort_run && !input_complete_q &&
                           !replay_active_q;
    wire input_accept = s_axis_tvalid && s_axis_tready;
    wire input_tag_ok = (s_axis_token == expect_token_q) &&
                        (s_axis_block == expect_block_q) &&
                        (s_axis_group == expected_group) &&
                        ({1'b0, s_axis_token} < run_tokens_q) &&
                        (s_axis_block < run_blocks_q);
    wire input_frame_ok = s_axis_tlast == (s_axis_group == 2'd3);
    wire input_fault = input_accept && (!input_tag_ok || !input_frame_ok);
    wire reorder_write = input_accept && input_tag_ok && input_frame_ok;
    // ---- Registered two-entry canonical GATE FIFO ----

    reg [255:0] gate_data_mem [0:1];
    reg [1:0]   gate_token_mem [0:1];
    reg [8:0]   gate_block_mem [0:1];
    reg [1:0]   gate_group_mem [0:1];
    reg gate_wr_ptr_q;
    reg gate_rd_ptr_q;
    reg [1:0] gate_count_q;

    wire gate_fifo_slot = gate_count_q < 2'd2;
    wire gate_fifo_push = reorder_rd_valid_q && gate_fifo_slot;
    // A consumed reorder result can be replaced by the next synchronous RAM read
    // on the same edge. No consumer-ready signal reaches the RAM address boundary.
    wire reorder_read_issue = busy_q && !abort_run && replay_active_q &&
                              (!reorder_rd_valid_q || gate_fifo_push);

    // ---- Registered scratch request and one in-flight read ----

    reg req_valid_q;
    reg [255:0] req_gate_q;
    reg [1:0] req_token_q;
    reg [8:0] req_block_q;
    reg [1:0] req_group_q;
    reg read_inflight_q;

    wire gate_fifo_pop = busy_q && !abort_run && !req_valid_q &&
                         !read_inflight_q && (gate_count_q != 2'd0);

    assign rd_req_valid = busy_q && !abort_run && req_valid_q;
    assign rd_req_role = ROLE_X1;
    assign rd_req_token = {1'b0, req_token_q};
    assign rd_req_group = {req_block_q, 2'b00} + {9'd0, req_group_q};
    wire rd_req_accept = rd_req_valid && rd_req_ready;

    // ---- Registered two-entry paired-group FIFO ----

    reg [255:0] pair_gate_mem [0:1];
    reg [255:0] pair_up_mem [0:1];
    reg [1:0]   pair_token_mem [0:1];
    reg [8:0]   pair_block_mem [0:1];
    reg [1:0]   pair_group_mem [0:1];
    reg pair_wr_ptr_q;
    reg pair_rd_ptr_q;
    reg [1:0] pair_count_q;

    // Orphan/idle responses are drained. A live response is accepted only into
    // registered storage with a free slot; downstream ready never reaches here.
    assign rd_rsp_ready = rst_n &&
                          (abort_run || orphan_q || !busy_q || !read_inflight_q ||
                           (pair_count_q < 2'd2));
    wire rd_rsp_accept = rd_rsp_valid && rd_rsp_ready;
    wire live_rsp_accept = rd_rsp_accept && busy_q && read_inflight_q && !abort_run;
    wire rsp_fault = live_rsp_accept && rd_rsp_error;
    wire unexpected_rsp_fault = rd_rsp_accept && busy_q && !read_inflight_q &&
                                !orphan_q && !abort_run;
    wire pair_push = live_rsp_accept && !rd_rsp_error;

    // ---- Registered group serializer and registered consumer output ----

    reg emit_active_q;
    reg [255:0] emit_gate_q;
    reg [255:0] emit_up_q;
    reg [1:0] emit_token_q;
    reg [8:0] emit_block_q;
    reg [1:0] emit_group_q;
    reg [2:0] emit_lane_q;

    wire pair_pop = busy_q && !abort_run && !emit_active_q &&
                    (pair_count_q != 2'd0);

    reg out_valid_q;
    reg [31:0] out_gate_q;
    reg [31:0] out_up_q;
    reg out_last_q;
    reg out_run_last_q;

    assign out_valid = busy_q && !abort_run && out_valid_q;
    assign out_gate = out_gate_q;
    assign out_up = out_up_q;
    assign out_last = out_last_q;
    wire out_accept = out_valid && out_ready;
    wire out_slot_open = !out_valid_q || out_ready;
    wire run_complete = out_accept && out_run_last_q;

    wire run_fault = input_fault || rsp_fault || unexpected_rsp_fault;
    wire unanswered_read = read_inflight_q || rd_req_accept;
    wire response_consumes_read = rd_rsp_accept && read_inflight_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            run_tokens_q <= 3'd0;
            run_blocks_q <= 9'd0;
            expect_block_q <= 9'd0;
            expect_upper_q <= 1'b0;
            expect_token_q <= 2'd0;
            expect_group_odd_q <= 1'b0;
            input_complete_q <= 1'b0;
            orphan_q <= 1'b0;
            replay_active_q <= 1'b0;
            replay_block_q <= 9'd0;
            replay_token_q <= 2'd0;
            replay_group_q <= 2'd0;
            reorder_rd_valid_q <= 1'b0;
            reorder_rd_block_q <= 9'd0;
            reorder_rd_token_q <= 2'd0;
            reorder_rd_group_q <= 2'd0;
            gate_wr_ptr_q <= 1'b0;
            gate_rd_ptr_q <= 1'b0;
            gate_count_q <= 2'd0;
            req_valid_q <= 1'b0;
            read_inflight_q <= 1'b0;
            pair_wr_ptr_q <= 1'b0;
            pair_rd_ptr_q <= 1'b0;
            pair_count_q <= 2'd0;
            emit_active_q <= 1'b0;
            emit_lane_q <= 3'd0;
            out_valid_q <= 1'b0;
            out_gate_q <= 32'd0;
            out_up_q <= 32'd0;
            out_last_q <= 1'b0;
            out_run_last_q <= 1'b0;
        end else begin
            done <= 1'b0;

            if (orphan_q && rd_rsp_accept)
                orphan_q <= 1'b0;

            if (abort_run) begin
                if (busy_q) begin
                    done <= 1'b1;
                    error <= 1'b1;
                end
                busy_q <= 1'b0;
                input_complete_q <= 1'b0;
                replay_active_q <= 1'b0;
                reorder_rd_valid_q <= 1'b0;
                gate_wr_ptr_q <= 1'b0;
                gate_rd_ptr_q <= 1'b0;
                gate_count_q <= 2'd0;
                req_valid_q <= 1'b0;
                read_inflight_q <= 1'b0;
                pair_wr_ptr_q <= 1'b0;
                pair_rd_ptr_q <= 1'b0;
                pair_count_q <= 2'd0;
                emit_active_q <= 1'b0;
                emit_lane_q <= 3'd0;
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
                out_run_last_q <= 1'b0;
                if (unanswered_read && !response_consumes_read)
                    orphan_q <= 1'b1;
            end else if (run_fault) begin
                busy_q <= 1'b0;
                done <= 1'b1;
                error <= 1'b1;
                input_complete_q <= 1'b0;
                replay_active_q <= 1'b0;
                reorder_rd_valid_q <= 1'b0;
                gate_wr_ptr_q <= 1'b0;
                gate_rd_ptr_q <= 1'b0;
                gate_count_q <= 2'd0;
                req_valid_q <= 1'b0;
                read_inflight_q <= 1'b0;
                pair_wr_ptr_q <= 1'b0;
                pair_rd_ptr_q <= 1'b0;
                pair_count_q <= 2'd0;
                emit_active_q <= 1'b0;
                emit_lane_q <= 3'd0;
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
                out_run_last_q <= 1'b0;
                if (unanswered_read && !response_consumes_read)
                    orphan_q <= 1'b1;
            end else if (start_accept) begin
                error <= !start_shape_ok;
                done <= !start_shape_ok;
                busy_q <= start_shape_ok;
                run_tokens_q <= start_tokens;
                run_blocks_q <= start_blocks;
                expect_block_q <= 9'd0;
                expect_upper_q <= 1'b0;
                expect_token_q <= 2'd0;
                expect_group_odd_q <= 1'b0;
                input_complete_q <= 1'b0;
                replay_active_q <= 1'b0;
                replay_block_q <= 9'd0;
                replay_token_q <= 2'd0;
                replay_group_q <= 2'd0;
                reorder_rd_valid_q <= 1'b0;
                gate_wr_ptr_q <= 1'b0;
                gate_rd_ptr_q <= 1'b0;
                gate_count_q <= 2'd0;
                req_valid_q <= 1'b0;
                read_inflight_q <= 1'b0;
                pair_wr_ptr_q <= 1'b0;
                pair_rd_ptr_q <= 1'b0;
                pair_count_q <= 2'd0;
                emit_active_q <= 1'b0;
                emit_lane_q <= 3'd0;
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
                out_run_last_q <= 1'b0;
            end else if (busy_q) begin
                if (reorder_write) begin
                    reorder_mem[{s_axis_token, s_axis_group}] <= s_axis_tdata;

                    if (!expect_group_odd_q) begin
                        expect_group_odd_q <= 1'b1;
                    end else begin
                        expect_group_odd_q <= 1'b0;
                        if (!expect_upper_q) begin
                            if ({1'b0, expect_token_q} + 1'b1 == run_tokens_q) begin
                                expect_upper_q <= 1'b1;
                                expect_token_q <= 2'd0;
                            end else begin
                                expect_token_q <= expect_token_q + 1'b1;
                            end
                        end else begin
                            replay_active_q <= 1'b1;
                            replay_block_q <= expect_block_q;
                            replay_token_q <= expect_token_q;
                            replay_group_q <= 2'd0;

                            if ({1'b0, expect_token_q} + 1'b1 == run_tokens_q) begin
                                expect_upper_q <= 1'b0;
                                expect_token_q <= 2'd0;
                                if (expect_block_q + 1'b1 == run_blocks_q) begin
                                    input_complete_q <= 1'b1;
                                end else begin
                                    expect_block_q <= expect_block_q + 1'b1;
                                end
                            end else begin
                                expect_token_q <= expect_token_q + 1'b1;
                            end
                        end
                    end
                end

                if (reorder_read_issue) begin
                    reorder_rd_data_q <= reorder_mem[{replay_token_q, replay_group_q}];
                    reorder_rd_block_q <= replay_block_q;
                    reorder_rd_token_q <= replay_token_q;
                    reorder_rd_group_q <= replay_group_q;
                    reorder_rd_valid_q <= 1'b1;
                    if (replay_group_q == 2'd3) begin
                        replay_active_q <= 1'b0;
                        replay_group_q <= 2'd0;
                    end else begin
                        replay_group_q <= replay_group_q + 1'b1;
                    end
                end else if (gate_fifo_push) begin
                    reorder_rd_valid_q <= 1'b0;
                end

                if (gate_fifo_push) begin
                    gate_data_mem[gate_wr_ptr_q] <= reorder_rd_data_q;
                    gate_token_mem[gate_wr_ptr_q] <= reorder_rd_token_q;
                    gate_block_mem[gate_wr_ptr_q] <= reorder_rd_block_q;
                    gate_group_mem[gate_wr_ptr_q] <= reorder_rd_group_q;
                    gate_wr_ptr_q <= gate_wr_ptr_q + 1'b1;
                end

                if (gate_fifo_pop) begin
                    req_gate_q <= gate_data_mem[gate_rd_ptr_q];
                    req_token_q <= gate_token_mem[gate_rd_ptr_q];
                    req_block_q <= gate_block_mem[gate_rd_ptr_q];
                    req_group_q <= gate_group_mem[gate_rd_ptr_q];
                    gate_rd_ptr_q <= gate_rd_ptr_q + 1'b1;
                    req_valid_q <= 1'b1;
                end

                case ({gate_fifo_push, gate_fifo_pop})
                    2'b10: gate_count_q <= gate_count_q + 1'b1;
                    2'b01: gate_count_q <= gate_count_q - 1'b1;
                    default: gate_count_q <= gate_count_q;
                endcase

                if (rd_req_accept) begin
                    req_valid_q <= 1'b0;
                    read_inflight_q <= 1'b1;
                end

                if (live_rsp_accept) begin
                    read_inflight_q <= 1'b0;
                    if (!rd_rsp_error) begin
                        pair_gate_mem[pair_wr_ptr_q] <= req_gate_q;
                        pair_up_mem[pair_wr_ptr_q] <= rd_rsp_data;
                        pair_token_mem[pair_wr_ptr_q] <= req_token_q;
                        pair_block_mem[pair_wr_ptr_q] <= req_block_q;
                        pair_group_mem[pair_wr_ptr_q] <= req_group_q;
                        pair_wr_ptr_q <= pair_wr_ptr_q + 1'b1;
                    end
                end

                if (pair_pop) begin
                    emit_gate_q <= pair_gate_mem[pair_rd_ptr_q];
                    emit_up_q <= pair_up_mem[pair_rd_ptr_q];
                    emit_token_q <= pair_token_mem[pair_rd_ptr_q];
                    emit_block_q <= pair_block_mem[pair_rd_ptr_q];
                    emit_group_q <= pair_group_mem[pair_rd_ptr_q];
                    emit_lane_q <= 3'd0;
                    emit_active_q <= 1'b1;
                    pair_rd_ptr_q <= pair_rd_ptr_q + 1'b1;
                end

                case ({pair_push, pair_pop})
                    2'b10: pair_count_q <= pair_count_q + 1'b1;
                    2'b01: pair_count_q <= pair_count_q - 1'b1;
                    default: pair_count_q <= pair_count_q;
                endcase

                if (out_slot_open) begin
                    if (emit_active_q) begin
                        out_valid_q <= 1'b1;
                        out_gate_q <= emit_gate_q[{emit_lane_q, 5'b00000} +: 32];
                        out_up_q <= emit_up_q[{emit_lane_q, 5'b00000} +: 32];
                        out_last_q <= (emit_group_q == 2'd3) &&
                                      (emit_lane_q == 3'd7);
                        out_run_last_q <=
                            (emit_block_q + 1'b1 == run_blocks_q) &&
                            ({1'b0, emit_token_q} + 1'b1 == run_tokens_q) &&
                            (emit_group_q == 2'd3) && (emit_lane_q == 3'd7);
                        if (emit_lane_q == 3'd7) begin
                            emit_active_q <= 1'b0;
                            emit_lane_q <= 3'd0;
                        end else begin
                            emit_lane_q <= emit_lane_q + 1'b1;
                        end
                    end else begin
                        out_valid_q <= 1'b0;
                        out_last_q <= 1'b0;
                        out_run_last_q <= 1'b0;
                    end
                end

                if (run_complete) begin
                    busy_q <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b0;
                    out_valid_q <= 1'b0;
                    out_last_q <= 1'b0;
                    out_run_last_q <= 1'b0;
                end
            end else if (rd_rsp_accept) begin
                // Drain a stale response while idle. start_ready remains low for
                // the entire cycle in which rd_rsp_valid is asserted.
                read_inflight_q <= 1'b0;
            end
        end
    end

`ifdef FORMAL
    assign formal_gate_count = gate_count_q;
    assign formal_pair_count = pair_count_q;
    assign formal_read_inflight = read_inflight_q;
    assign formal_orphan = orphan_q;
    assign formal_emit_lane = emit_lane_q;
`endif

endmodule

`default_nettype wire
