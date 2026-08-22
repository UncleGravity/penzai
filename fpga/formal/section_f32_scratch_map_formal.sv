`default_nettype none

// Production-width address/control proof for section_f32_scratch. The read data
// is intentionally outside the assertion cone, allowing the four 1-Mbit banks to
// be pruned before PDR. Data retention is checked by the separate storage BMC.
module section_f32_scratch_map_formal(input wire clk);
    localparam [1:0] ROLE_R  = 2'd0;
    localparam [1:0] ROLE_X0 = 2'd1;
    localparam [1:0] ROLE_X1 = 2'd2;
    localparam [1:0] ROLE_X2 = 2'd3;

    function automatic [13:0] role_base(input [1:0] role);
        case (role)
            ROLE_R:  role_base = 14'd0;
            ROLE_X0: role_base = 14'd2048;
            ROLE_X1: role_base = 14'd8192;
            default: role_base = 14'd14336;
        endcase
    endfunction

    function automatic [13:0] role_capacity(input [1:0] role);
        case (role)
            ROLE_R, ROLE_X2: role_capacity = 14'd4096;
            default:         role_capacity = 14'd12288;
        endcase
    endfunction

    function automatic [10:0] role_groups(input [1:0] role);
        case (role)
            ROLE_R, ROLE_X2: role_groups = 11'd512;
            default:         role_groups = 11'd1536;
        endcase
    endfunction

    function automatic [13:0] token_offset(input [1:0] role, input [2:0] token);
        case (role)
            ROLE_R, ROLE_X2:
                token_offset = {2'b00, token, 9'b0};
            default:
                token_offset = {1'b0, token, 10'b0} +
                               {2'b00, token, 9'b0};
        endcase
    endfunction

    function automatic [13:0] logical_address(
        input [1:0] role,
        input [2:0] token,
        input [10:0] group
    );
        logical_address = role_base(role) + token_offset(role, token) +
                          {3'b000, group};
    endfunction

    (* anyconst *) reg [1:0]  cfg_role;
    (* anyconst *) reg [13:0] cfg_rows;
    (* anyconst *) reg [2:0]  cfg_tokens;

    // Pairwise symbolic locations prove injectivity across the complete
    // contract-v1 address space without enumerating a full GEMM stream.
    (* anyconst *) reg [1:0]  loc_a_role, loc_b_role;
    (* anyconst *) reg [2:0]  loc_a_token, loc_b_token;
    (* anyconst *) reg [10:0] loc_a_group, loc_b_group;
    (* anyconst *) reg [1:0]  loc_a_bank, loc_b_bank;

    (* anyseq *) reg          stream_valid;
    (* anyseq *) reg          abort_run;
    (* anyseq *) reg          rd_req_valid;
    (* anyseq *) reg [1:0]    rd_req_role;
    (* anyseq *) reg [2:0]    rd_req_token;
    (* anyseq *) reg [10:0]   rd_req_group;
    (* anyseq *) reg          rd_rsp_ready;
    (* anyseq *) reg          rst_n;

    reg f_past_valid = 1'b0;
    reg cfg_sent = 1'b0;
    reg run_active = 1'b0;
    reg [9:0] ref_rowblock = 10'd0;
    reg [2:0] ref_token = 3'd0;
    reg [2:0] ref_pair = 3'd0;
    reg ref_rd_pending = 1'b0;
    reg [13:0] ref_rd_address = 14'd0;
    reg ref_rd_pending_error = 1'b0;
    reg ref_prev_mem_write = 1'b0;
    reg [13:0] ref_prev_mem_write_address = 14'd0;
    reg ref_rd_result_pending = 1'b0;
    reg ref_rd_result_error = 1'b0;
    reg ref_rsp_expected = 1'b0;
    reg ref_rsp_error = 1'b0;

    wire wr_cfg_valid = rst_n && !cfg_sent;
    wire wr_cfg_ready;
    wire wr_busy;
    wire wr_done;
    wire wr_error;
    wire s_axis_tready;
    wire wr_commit_valid;
    wire [1:0] wr_commit_bank;
    wire [13:0] wr_commit_address;
    wire rd_req_ready;
    wire rd_quiescent;
    wire rd_admission_idle;
    wire rd_issue_valid;
    wire [13:0] rd_issue_address;
    wire rd_rsp_valid;
    wire [255:0] rd_rsp_data_unused;
    wire rd_rsp_error;

    wire cfg_accept = wr_cfg_valid && wr_cfg_ready;
    wire stream_accept = stream_valid && s_axis_tready;
    wire [9:0] ref_rowblocks = (cfg_rows + 14'd15) >> 4;
    wire ref_final_rowblock = ref_rowblock + 1'b1 == ref_rowblocks;
    wire [2:0] ref_last_pair = (ref_final_rowblock && cfg_rows[3]) ? 3'd3 : 3'd7;
    wire ref_expected_last = ref_final_rowblock &&
                             (ref_token + 1'b1 == cfg_tokens) &&
                             (ref_pair == ref_last_pair);
    wire [10:0] ref_group = {ref_rowblock, 1'b0} +
                            ((ref_pair >= 3'd4) ? 11'd1 : 11'd0);
    wire [13:0] ref_address = logical_address(cfg_role, ref_token, ref_group);

    wire rd_request_bad_ref = (rd_req_token >= 3'd4) ||
                              (rd_req_group >= role_groups(rd_req_role));
    wire [13:0] rd_address_ref = rd_request_bad_ref ? 14'd0 :
                                 logical_address(rd_req_role, rd_req_token,
                                                 rd_req_group);

    wire [13:0] loc_a_address = logical_address(loc_a_role, loc_a_token,
                                                loc_a_group);
    wire [13:0] loc_b_address = logical_address(loc_b_role, loc_b_token,
                                                loc_b_group);

    section_f32_scratch dut (
        .clk(clk), .rst_n(rst_n),
        .wr_cfg_valid(wr_cfg_valid), .wr_cfg_ready(wr_cfg_ready),
        .wr_cfg_role(cfg_role), .wr_cfg_rows(cfg_rows),
        .wr_cfg_tokens(cfg_tokens), .wr_abort(abort_run),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error),
        .s_axis_tdata(64'd0), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(stream_valid), .s_axis_tready(s_axis_tready),
        .s_axis_tready_core(),
`ifdef MAP_PROVE
        .s_axis_tlast(1'b0),
`else
        .s_axis_tlast(ref_expected_last),
`endif
        .wr_commit_valid(wr_commit_valid), .wr_commit_bank(wr_commit_bank),
        .wr_commit_address(wr_commit_address),
        .r_wr_abort(1'b0), .r_wr_valid(1'b0), .r_wr_ready(), .r_wr_bank(2'd0),
        .r_wr_address(14'd0), .r_wr_data(64'd0), .r_wr_error(),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_admission_idle(rd_admission_idle),
        .rd_quiescent(rd_quiescent),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group),
        .rd_issue_valid(rd_issue_valid), .rd_issue_address(rd_issue_address),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready),
        .rd_rsp_data(rd_rsp_data_unused), .rd_rsp_error(rd_rsp_error)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) assume(!rst_n);
        else               assume(rst_n);

        assume(cfg_rows != 14'd0);
        assume(cfg_rows <= role_capacity(cfg_role));
        assume(cfg_rows[2:0] == 3'b000);
        assume(cfg_tokens != 3'd0 && cfg_tokens <= 3'd4);

        assume(loc_a_token < 3'd4);
        assume(loc_b_token < 3'd4);
        assume(loc_a_group < role_groups(loc_a_role));
        assume(loc_b_group < role_groups(loc_b_role));
        assert({1'b0, loc_a_address} < 15'd16384);
        assert({1'b0, loc_b_address} < 15'd16384);
        if ((loc_a_address == loc_b_address) &&
            (loc_a_bank == loc_b_bank)) begin
            assert(loc_a_role == loc_b_role);
            assert(loc_a_token == loc_b_token);
            assert(loc_a_group == loc_b_group);
            assert(loc_a_bank == loc_b_bank);
        end

        if (!rst_n) begin
            cfg_sent <= 1'b0;
            run_active <= 1'b0;
            ref_rowblock <= 10'd0;
            ref_token <= 3'd0;
            ref_pair <= 3'd0;
            ref_rd_pending <= 1'b0;
            ref_rd_result_pending <= 1'b0;
            ref_rsp_expected <= 1'b0;
            ref_prev_mem_write <= 1'b0;
        end else begin
            ref_prev_mem_write <= wr_commit_valid;
            if (wr_commit_valid)
                ref_prev_mem_write_address <= wr_commit_address;
            ref_rsp_expected <= ref_rd_result_pending;
            if (ref_rd_result_pending)
                ref_rsp_error <= ref_rd_result_error;

            ref_rd_result_pending <= ref_rd_pending;
            if (ref_rd_pending) begin
                ref_rd_result_error <= ref_rd_pending_error ||
                                       (ref_prev_mem_write &&
                                        (ref_rd_address ==
                                         ref_prev_mem_write_address)) ||
                                       (wr_commit_valid &&
                                        (ref_rd_address == wr_commit_address));
            end

            ref_rd_pending <= rd_issue_valid;
            if (rd_issue_valid) begin
                ref_rd_address <= rd_issue_address;
                ref_rd_pending_error <= rd_request_bad_ref;
            end

            if (cfg_accept) begin
                cfg_sent <= 1'b1;
                run_active <= 1'b1;
                ref_rowblock <= 10'd0;
                ref_token <= 3'd0;
                ref_pair <= 3'd0;
            end else if (abort_run && run_active) begin
                run_active <= 1'b0;
            end else if (stream_accept) begin
                if (ref_pair == ref_last_pair) begin
                    ref_pair <= 3'd0;
                    if (ref_token + 1'b1 == cfg_tokens) begin
                        ref_token <= 3'd0;
                        if (ref_final_rowblock) begin
                            run_active <= 1'b0;
                        end else begin
                            ref_rowblock <= ref_rowblock + 1'b1;
                        end
                    end else begin
                        ref_token <= ref_token + 1'b1;
                    end
                end else begin
                    ref_pair <= ref_pair + 1'b1;
                end
            end
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!wr_busy && !wr_done && !wr_error);
            assert(!wr_commit_valid && !rd_rsp_valid);
        end

        if (rst_n && f_past_valid && $past(rst_n)) begin
            assert(wr_cfg_ready == (!wr_busy && !abort_run));
`ifndef MAP_PROVE
            assert(wr_busy == run_active);
            if (run_active)
                assert(wr_commit_address == ref_address);
`endif
            assert(s_axis_tready == (wr_busy && !abort_run));
            assert(wr_commit_valid == stream_accept);

            if (wr_commit_valid) begin
`ifndef MAP_PROVE
                assert(wr_commit_bank == ref_pair[1:0]);
                assert(wr_commit_address == ref_address);
                assert({1'b0, wr_commit_address} < 15'd16384);
                case (cfg_role)
                    ROLE_R:  assert(wr_commit_address < 14'd2048);
                    ROLE_X0: assert(wr_commit_address >= 14'd2048 &&
                                    wr_commit_address < 14'd8192);
                    ROLE_X1: assert(wr_commit_address >= 14'd8192 &&
                                    wr_commit_address < 14'd14336);
                    default: assert(wr_commit_address >= 14'd14336);
                endcase
`endif
            end

            assert(rd_issue_valid == (rd_req_valid && rd_req_ready));
            if (rd_issue_valid) begin
                assert(rd_issue_address == rd_address_ref);
                assert({1'b0, rd_issue_address} < 15'd16384);
                if (!rd_request_bad_ref) begin
                    case (rd_req_role)
                        ROLE_R:  assert(rd_issue_address < 14'd2048);
                        ROLE_X0: assert(rd_issue_address >= 14'd2048 &&
                                        rd_issue_address < 14'd8192);
                        ROLE_X1: assert(rd_issue_address >= 14'd8192 &&
                                        rd_issue_address < 14'd14336);
                        default: assert(rd_issue_address >= 14'd14336);
                    endcase
                end
            end

            if (abort_run && wr_busy) begin
                assert(!s_axis_tready);
                assert(!wr_commit_valid);
            end

            if ($past(abort_run && wr_busy)) begin
                assert(wr_done && wr_error && !wr_busy);
            end

`ifndef MAP_PROVE
            if ($past(stream_accept && ref_expected_last)) begin
                assert(wr_done && !wr_error && !wr_busy);
            end
`endif

            if (ref_rd_pending || ref_rd_result_pending) begin
                assert(!rd_req_ready);
                assert(!rd_rsp_valid);
            end
            if (ref_rsp_expected) begin
                assert(rd_rsp_valid);
`ifdef MAP_PROVE
                if (ref_rsp_error) assert(rd_rsp_error);
`else
                assert(rd_rsp_error == ref_rsp_error);
`endif
            end
            if ($past(rd_rsp_valid && !rd_rsp_ready)) begin
                assert(rd_rsp_valid);
                assert(rd_rsp_error == $past(rd_rsp_error));
            end
        end

        cover(rst_n && cfg_role == ROLE_X0 && cfg_rows == 14'd8 &&
              cfg_tokens == 3'd1 && wr_done && !wr_error);
        cover(rst_n && wr_done && wr_error && !wr_busy);
        cover(rst_n && rd_issue_valid && rd_request_bad_ref);
        cover(rst_n && rd_rsp_valid && !rd_rsp_ready);
        cover(rst_n && rd_issue_valid && wr_commit_valid &&
              rd_issue_address == wr_commit_address);
        cover(rst_n && ref_rd_pending && wr_commit_valid &&
              ref_rd_address == wr_commit_address);
    end
endmodule

`default_nettype wire
