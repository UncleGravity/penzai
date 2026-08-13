// Four-bank FP32 section scratch.
//
// The physical layout is the P2 section contract packed into one address space:
//
//   bank = (even_row / 2) % 4
//   addr = role_base + token * groups_per_token + even_row / 8
//   word = { odd_row, even_row }
//
// GEMM emits [rowblock][token][row-pair], with 16 rows per rowblock.  The write
// sink consumes that stream directly; no transpose or token-major copy is needed.
// The read side returns all four banks at one address as eight adjacent FP32s.
//
// Roles are encoded as 0=R, 1=X0, 2=X1, 3=X2.  A section controller must not
// schedule a read of the same physical group that is being written.  Such a
// request is accepted but returns rd_rsp_error with zero data.

`default_nettype none

module section_f32_scratch (
    input  wire          clk,
    input  wire          rst_n,

    // One configuration describes one complete GEMM result stream.
    input  wire          wr_cfg_valid,
    output wire          wr_cfg_ready,
    input  wire [1:0]    wr_cfg_role,
    input  wire [13:0]   wr_cfg_rows,
    input  wire [2:0]    wr_cfg_tokens,
    input  wire          wr_abort,
    output wire          wr_busy,
    output reg           wr_done,
    output reg           wr_error,

    // Directly compatible with gemm_kernel.m_axis_*.
    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    // Accepted memory writes, exposed for section-control accounting/formal.
    output wire          wr_commit_valid,
    output wire [1:0]    wr_commit_bank,
    output wire [13:0]   wr_commit_address,

    // One role-local row group request.  group zero denotes rows 0..7.
    input  wire          rd_req_valid,
    output wire          rd_req_ready,
    input  wire [1:0]    rd_req_role,
    input  wire [2:0]    rd_req_token,
    input  wire [10:0]   rd_req_group,

    // Accepted bounded group-read address.  The physical memory access follows
    // from this registered address on the next cycle.
    output wire          rd_issue_valid,
    output wire [13:0]   rd_issue_address,

    // data[63:0] is bank 0 (rows 0/1), data[255:192] is bank 3
    // (rows 6/7).  The response is elastic and remains stable while stalled.
    output wire          rd_rsp_valid,
    input  wire          rd_rsp_ready,
    output wire [255:0]  rd_rsp_data,
    output reg           rd_rsp_error
);
    localparam [1:0] ROLE_R  = 2'd0;
    localparam [1:0] ROLE_X0 = 2'd1;
    localparam [1:0] ROLE_X1 = 2'd2;
    localparam [1:0] ROLE_X2 = 2'd3;

    // Four 64-bit banks x 16384 words = 512 KiB.  At 16384x64 each bank maps
    // to four depth-cascaded URAM288s on XCK26, for 16 URAM288s total.
    localparam integer BANK_DEPTH = 16384;
    (* ram_style = "ultra" *) reg [63:0] bank0_mem [0:BANK_DEPTH-1];
    (* ram_style = "ultra" *) reg [63:0] bank1_mem [0:BANK_DEPTH-1];
    (* ram_style = "ultra" *) reg [63:0] bank2_mem [0:BANK_DEPTH-1];
    (* ram_style = "ultra" *) reg [63:0] bank3_mem [0:BANK_DEPTH-1];

    // ---- Write configuration and GEMM-order address generation ----

    reg         wr_busy_q;
    reg         run_half_final;
    reg [2:0]   run_tokens;
    reg [13:0]  run_role_base;
    reg [10:0]  run_groups_per_token;
    reg [9:0]   run_rowblocks;
    reg [9:0]   wr_rowblock;
    reg [2:0]   wr_token;
    reg [2:0]   wr_pair;
    reg [13:0]  wr_address_q;

    reg [13:0] cfg_capacity;
    reg [13:0] cfg_role_base;
    always @* begin
        case (wr_cfg_role)
            ROLE_R: begin
                cfg_capacity  = 14'd4096;
                cfg_role_base = 14'd0;
            end
            ROLE_X0: begin
                cfg_capacity  = 14'd12288;
                cfg_role_base = 14'd2048;
            end
            ROLE_X1: begin
                cfg_capacity  = 14'd12288;
                cfg_role_base = 14'd8192;
            end
            default: begin // ROLE_X2
                cfg_capacity  = 14'd4096;
                cfg_role_base = 14'd14336;
            end
        endcase
    end

    // Section shapes are multiples of eight rows.  Requiring a complete
    // 256-bit final group keeps every accepted GEMM word at tkeep=0xff.
    wire cfg_shape_ok = (wr_cfg_rows != 14'd0) &&
                        (wr_cfg_rows <= cfg_capacity) &&
                        (wr_cfg_rows[2:0] == 3'b000) &&
                        (wr_cfg_tokens != 3'd0) &&
                        (wr_cfg_tokens <= 3'd4);
    wire cfg_accept = wr_cfg_valid && wr_cfg_ready;

    assign wr_cfg_ready = rst_n && !wr_busy_q && !wr_abort;
    assign wr_busy      = wr_busy_q;
    // Abort wins combinationally so a beat presented in the abort cycle is not
    // accidentally committed before the controller observes completion.
    assign s_axis_tready = rst_n && wr_busy_q && !wr_abort;

    wire wr_accept = s_axis_tvalid && s_axis_tready;
    wire wr_final_rowblock = (wr_rowblock == run_rowblocks - 1'b1);
    // With rows constrained to a multiple of eight, a partial final 16-row
    // block contains exactly eight rows.  Otherwise it contains all sixteen.
    wire [2:0] wr_active_last_pair = (wr_final_rowblock && run_half_final) ? 3'd3 : 3'd7;
    wire wr_expected_last = wr_final_rowblock &&
                            (wr_token + 1'b1 == run_tokens) &&
                            (wr_pair == wr_active_last_pair);
    wire wr_frame_bad = (s_axis_tkeep != 8'hff) ||
                        (s_axis_tlast != wr_expected_last);

    // Keep the physical write address registered at the URAM boundary.  A
    // 16-row rowblock contributes two groups: pairs 0..3 share one address and
    // pairs 4..7 share the next.  The sequential updates below jump directly
    // between tokens and rowblocks without putting that arithmetic on the URAM
    // address pins.
    wire [13:0] wr_address = wr_address_q;
    wire [13:0] wr_next_rowblock_address =
        run_role_base + {3'b000, wr_rowblock, 1'b0} + 14'd2;
    wire [1:0] wr_bank = wr_pair[1:0];
    // A partial byte word is consumed and diagnosed, but never mutates storage.
    wire wr_mem_write = wr_accept && (s_axis_tkeep == 8'hff);
    assign wr_commit_valid   = wr_mem_write;
    assign wr_commit_bank    = wr_bank;
    assign wr_commit_address = wr_address;

    // ---- Elastic synchronous 256-bit group read ----

    reg [13:0] rd_role_base;
    reg [10:0] rd_role_groups;
    always @* begin
        case (rd_req_role)
            ROLE_R: begin
                rd_role_base   = 14'd0;
                rd_role_groups = 11'd512;
            end
            ROLE_X0: begin
                rd_role_base   = 14'd2048;
                rd_role_groups = 11'd1536;
            end
            ROLE_X1: begin
                rd_role_base   = 14'd8192;
                rd_role_groups = 11'd1536;
            end
            default: begin // ROLE_X2
                rd_role_base   = 14'd14336;
                rd_role_groups = 11'd512;
            end
        endcase
    end

    reg [13:0] rd_token_offset;
    always @* begin
        case (rd_req_role)
            ROLE_R, ROLE_X2:
                rd_token_offset = {2'b00, rd_req_token, 9'b0};
            default:
                rd_token_offset = {1'b0, rd_req_token, 10'b0} +
                                  {2'b00, rd_req_token, 9'b0};
        endcase
    end

    wire rd_request_bad = (rd_req_token >= 3'd4) ||
                          (rd_req_group >= rd_role_groups);
    wire [13:0] rd_address_unchecked = rd_role_base + rd_token_offset +
                                       {3'b000, rd_req_group};
    // Invalid requests still receive a bounded, deterministic response.  The
    // data is masked to zero by rd_rsp_error below.
    wire [13:0] rd_address = rd_request_bad ? 14'd0 : rd_address_unchecked;

    reg         rd_pending_q;
    reg [13:0]  rd_address_q;
    reg         rd_pending_error_q;
    reg         rd_rsp_valid_q;
    reg [63:0]  rd_bank0_q;
    reg [63:0]  rd_bank1_q;
    reg [63:0]  rd_bank2_q;
    reg [63:0]  rd_bank3_q;

    // The accepted request is registered before it reaches the four-deep URAM
    // cascades.  Keeping only one request pending also preserves the single
    // elastic response slot without a response FIFO.
    assign rd_req_ready = rst_n && !rd_pending_q &&
                          (!rd_rsp_valid_q || rd_rsp_ready);
    wire rd_accept = rd_req_valid && rd_req_ready;
    wire rd_accept_collision = rd_accept && !rd_request_bad && wr_mem_write &&
                               (rd_address == wr_address);
    wire rd_issue_collision = rd_pending_q &&
                              wr_mem_write && (rd_address_q == wr_address);

    assign rd_rsp_valid = rd_rsp_valid_q;
    assign rd_issue_valid = rd_accept;
    assign rd_issue_address = rd_address;
    assign rd_rsp_data = rd_rsp_error ? 256'd0 :
                         {rd_bank3_q, rd_bank2_q, rd_bank1_q, rd_bank0_q};

    // This is the simple-dual-port UltraRAM inference template: one narrow
    // bank-selecting write port and one synchronous read port per bank.  The
    // memories themselves are intentionally not reset.
    always @(posedge clk) begin
        if (wr_mem_write) begin
            case (wr_bank)
                2'd0: bank0_mem[wr_address] <= s_axis_tdata;
                2'd1: bank1_mem[wr_address] <= s_axis_tdata;
                2'd2: bank2_mem[wr_address] <= s_axis_tdata;
                default: bank3_mem[wr_address] <= s_axis_tdata;
            endcase
        end

        if (rd_pending_q) begin
            rd_bank0_q <= bank0_mem[rd_address_q];
            rd_bank1_q <= bank1_mem[rd_address_q];
            rd_bank2_q <= bank2_mem[rd_address_q];
            rd_bank3_q <= bank3_mem[rd_address_q];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            wr_busy_q       <= 1'b0;
            wr_done         <= 1'b0;
            wr_error        <= 1'b0;
            run_half_final  <= 1'b0;
            run_tokens      <= 3'd0;
            run_role_base   <= 14'd0;
            run_groups_per_token <= 11'd0;
            run_rowblocks   <= 10'd0;
            wr_rowblock     <= 10'd0;
            wr_token        <= 3'd0;
            wr_pair         <= 3'd0;
            wr_address_q    <= 14'd0;
            rd_pending_q    <= 1'b0;
            rd_rsp_valid_q  <= 1'b0;
            rd_rsp_error    <= 1'b0;
        end else begin
            wr_done <= 1'b0;

            if (cfg_accept) begin
                wr_error      <= !cfg_shape_ok;
                wr_rowblock   <= 10'd0;
                wr_token      <= 3'd0;
                wr_pair       <= 3'd0;
                wr_address_q  <= cfg_role_base;
                if (cfg_shape_ok) begin
                    wr_busy_q     <= 1'b1;
                    run_half_final <= wr_cfg_rows[3];
                    run_tokens    <= wr_cfg_tokens;
                    run_role_base <= cfg_role_base;
                    run_groups_per_token <=
                        ((wr_cfg_role == ROLE_R) || (wr_cfg_role == ROLE_X2)) ?
                            11'd512 : 11'd1536;
                    run_rowblocks <= wr_cfg_rows[13:4] + {{9{1'b0}}, wr_cfg_rows[3]};
                end else begin
                    wr_busy_q <= 1'b0;
                    wr_done   <= 1'b1;
                end
            end else if (wr_abort && wr_busy_q) begin
                // Scratch is section-private and writes are not transactional.
                // Already committed words remain, but no abort-cycle beat is
                // accepted and the run completes with a sticky error.
                wr_busy_q <= 1'b0;
                wr_done   <= 1'b1;
                wr_error  <= 1'b1;
            end else if (wr_accept) begin
                if (wr_frame_bad)
                    wr_error <= 1'b1;

                if (wr_pair == wr_active_last_pair) begin
                    wr_pair <= 3'd0;
                    if (wr_token + 1'b1 == run_tokens) begin
                        wr_token <= 3'd0;
                        if (wr_final_rowblock) begin
                            wr_busy_q <= 1'b0;
                            wr_done   <= 1'b1;
                        end else begin
                            wr_rowblock <= wr_rowblock + 1'b1;
                            wr_address_q <= wr_next_rowblock_address;
                        end
                    end else begin
                        wr_token <= wr_token + 1'b1;
                        if (wr_active_last_pair == 3'd3)
                            wr_address_q <= wr_address_q +
                                            {3'b000, run_groups_per_token};
                        else
                            wr_address_q <= wr_address_q +
                                            {3'b000, run_groups_per_token} - 14'd1;
                    end
                end else begin
                    wr_pair <= wr_pair + 1'b1;
                    if (wr_pair == 3'd3)
                        wr_address_q <= wr_address_q + 1'b1;
                end
            end

            if (rd_accept) begin
                rd_pending_q       <= 1'b1;
                rd_address_q       <= rd_address;
                rd_pending_error_q <= rd_request_bad || rd_accept_collision;
            end else if (rd_pending_q) begin
                rd_pending_q <= 1'b0;
            end

            if (rd_pending_q) begin
                rd_rsp_valid_q <= 1'b1;
                rd_rsp_error   <= rd_pending_error_q || rd_issue_collision;
            end else if (rd_rsp_valid_q && rd_rsp_ready) begin
                rd_rsp_valid_q <= 1'b0;
                rd_rsp_error   <= 1'b0;
            end
        end
    end

endmodule
