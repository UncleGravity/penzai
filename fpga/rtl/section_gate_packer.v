// section_gate_packer - pack native GEMM GATE results for section_ffn_pairer.
//
// gemm_kernel emits one 16-row rowblock at a time. Within each rowblock it
// visits every token, emitting eight 64-bit FP32-pair beats per token:
//
//   rowblock r, token t, pair p = 0..7
//
// Four adjacent beats become one 256-bit/eight-scalar group. The explicit tag
// maps the two 16-row sweeps back to the pairer's 32-row block contract:
//
//   block = r >> 1, group = {r[0], p[2]}, token = t
//
// Input TLAST is the physical end of the complete GEMM run. Output TLAST is a
// logical 32-scalar boundary and is asserted on every group three. Every input
// beat must have all eight bytes enabled. Any framing error fails closed.

`default_nettype none

module section_gate_packer (
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

    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    output wire          s_axis_tready_core,
    input  wire          s_axis_tlast,

    output wire [255:0]  m_axis_tdata,
    output wire          m_axis_tvalid,
    input  wire          m_axis_tready,
    output wire          m_axis_tlast,
    output wire [1:0]    m_axis_token,
    output wire [8:0]    m_axis_block,
    output wire [1:0]    m_axis_group
);
    localparam [8:0] BLOCK_CAPACITY = 9'd384;

    reg busy_q;
    reg [2:0] run_tokens_q;
    reg [9:0] run_rowblocks_q;
    reg [9:0] rowblock_q;
    reg [2:0] token_q;
    reg [2:0] pair_q;
    reg input_complete_q;

    // The assembly register is also the elastic output payload. A stalled output
    // closes ingress, while a draining output may accept the next group's first
    // beat on the same edge after the old payload has handshaken.
    reg [255:0] data_q;
    reg out_valid_q;
    reg out_last_q;
    reg out_run_last_q;
    reg [1:0] out_token_q;
    reg [8:0] out_block_q;
    reg [1:0] out_group_q;

    assign busy = busy_q;
    assign start_ready = rst_n && !abort_run && !busy_q;
    wire start_accept = start_valid && start_ready;
    wire start_shape_ok = (start_tokens != 3'd0) &&
                          (start_tokens <= 3'd4) &&
                          (start_blocks != 9'd0) &&
                          (start_blocks <= BLOCK_CAPACITY);

    assign m_axis_tdata = data_q;
    assign m_axis_tvalid = busy_q && !abort_run && out_valid_q;
    assign m_axis_tlast = out_last_q;
    assign m_axis_token = out_token_q;
    assign m_axis_block = out_block_q;
    assign m_axis_group = out_group_q;
    wire output_accept = m_axis_tvalid && m_axis_tready;
    wire output_slot_open = !out_valid_q || m_axis_tready;

    // Do not accept beyond the physical end of the run. Holding ingress while
    // a completed group is stalled also guarantees fail-closed output stability.
    assign s_axis_tready_core = rst_n && busy_q && !input_complete_q &&
                                output_slot_open;
    assign s_axis_tready = s_axis_tready_core && !abort_run;
    wire input_accept = s_axis_tvalid && s_axis_tready;
    wire expected_input_last =
        (rowblock_q + 1'b1 == run_rowblocks_q) &&
        (token_q + 1'b1 == run_tokens_q) &&
        (pair_q == 3'd7);
    wire input_frame_ok = (s_axis_tkeep == 8'hff) &&
                          (s_axis_tlast == expected_input_last);
    wire input_fault = input_accept && !input_frame_ok;
    wire group_final = pair_q[1:0] == 2'd3;
    wire [1:0] input_group = {rowblock_q[0], pair_q[2]};

    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            run_tokens_q <= 3'd0;
            run_rowblocks_q <= 10'd0;
            rowblock_q <= 10'd0;
            token_q <= 3'd0;
            pair_q <= 3'd0;
            input_complete_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_run_last_q <= 1'b0;
            out_token_q <= 2'd0;
            out_block_q <= 9'd0;
            out_group_q <= 2'd0;
        end else if (abort_run && busy_q) begin
            busy_q <= 1'b0;
            done <= 1'b1;
            error <= 1'b1;
            input_complete_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_run_last_q <= 1'b0;
        end else if (input_fault) begin
            busy_q <= 1'b0;
            done <= 1'b1;
            error <= 1'b1;
            input_complete_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_run_last_q <= 1'b0;
        end else if (start_accept) begin
            done <= !start_shape_ok;
            error <= !start_shape_ok;
            busy_q <= start_shape_ok;
            run_tokens_q <= start_tokens;
            run_rowblocks_q <= {start_blocks, 1'b0};
            rowblock_q <= 10'd0;
            token_q <= 3'd0;
            pair_q <= 3'd0;
            input_complete_q <= 1'b0;
            out_valid_q <= 1'b0;
            out_last_q <= 1'b0;
            out_run_last_q <= 1'b0;
            out_token_q <= 2'd0;
            out_block_q <= 9'd0;
            out_group_q <= 2'd0;
        end else if (busy_q) begin
            if (output_accept) begin
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
                out_run_last_q <= 1'b0;
            end

            if (input_accept) begin
                case (pair_q[1:0])
                    2'd0: data_q[ 63:  0] <= s_axis_tdata;
                    2'd1: data_q[127: 64] <= s_axis_tdata;
                    2'd2: data_q[191:128] <= s_axis_tdata;
                    default: data_q[255:192] <= s_axis_tdata;
                endcase

                if (group_final) begin
                    out_valid_q <= 1'b1;
                    out_last_q <= input_group == 2'd3;
                    out_run_last_q <= expected_input_last;
                    out_token_q <= token_q[1:0];
                    out_block_q <= rowblock_q[9:1];
                    out_group_q <= input_group;
                end

                if (pair_q == 3'd7) begin
                    pair_q <= 3'd0;
                    if (token_q + 1'b1 == run_tokens_q) begin
                        token_q <= 3'd0;
                        if (rowblock_q + 1'b1 == run_rowblocks_q) begin
                            input_complete_q <= 1'b1;
                        end else begin
                            rowblock_q <= rowblock_q + 1'b1;
                        end
                    end else begin
                        token_q <= token_q + 1'b1;
                    end
                end else begin
                    pair_q <= pair_q + 1'b1;
                end
            end

            if (output_accept && out_run_last_q) begin
                busy_q <= 1'b0;
                done <= 1'b1;
                error <= 1'b0;
                input_complete_q <= 1'b0;
                out_valid_q <= 1'b0;
                out_last_q <= 1'b0;
                out_run_last_q <= 1'b0;
            end
        end
    end

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;

        if (rst_n) begin
            assert(s_axis_tready ==
                   (s_axis_tready_core && !abort_run));
            assert(rowblock_q < (run_rowblocks_q == 0 ? 10'd1 : run_rowblocks_q));
            assert(token_q < (run_tokens_q == 0 ? 3'd1 : run_tokens_q));
            assert(pair_q <= 3'd7);
            if (m_axis_tvalid) begin
                assert(m_axis_token < run_tokens_q);
                assert(m_axis_block < run_rowblocks_q[9:1]);
                assert(m_axis_tlast == (m_axis_group == 2'd3));
            end
            if (!busy_q) begin
                assert(!m_axis_tvalid);
                assert(!s_axis_tready);
            end
            if (abort_run)
                assert(!s_axis_tready && !m_axis_tvalid);
        end

        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && m_axis_tvalid &&
                                           !m_axis_tready && !abort_run)) begin
            assert(m_axis_tvalid);
            assert(m_axis_tdata == $past(m_axis_tdata));
            assert(m_axis_tlast == $past(m_axis_tlast));
            assert(m_axis_token == $past(m_axis_token));
            assert(m_axis_block == $past(m_axis_block));
            assert(m_axis_group == $past(m_axis_group));
        end

        if (f_past_valid && $past(rst_n && input_fault)) begin
            assert(!busy);
            assert(done && error);
            assert(!m_axis_tvalid);
        end

        if (f_past_valid && $past(rst_n && abort_run && busy_q)) begin
            assert(!busy);
            assert(done && error);
            assert(!m_axis_tvalid);
        end
    end
`endif
endmodule

`default_nettype wire
