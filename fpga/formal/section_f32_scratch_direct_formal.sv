`default_nettype none

// Focused proof for the RMSNorm loader's token-major R write port. The real
// memories retain two complete R groups and one legacy X0 group, then exercise
// malformed-address suppression, ownership, abort, and both read-collision
// windows before reading every modified location back.
module section_f32_scratch_direct_formal(input wire clk);
    localparam [1:0] ROLE_R  = 2'd0;
    localparam [1:0] ROLE_X0 = 2'd1;

    localparam [4:0] PH_R0_WRITE       = 5'd0;
    localparam [4:0] PH_R1_WRITE       = 5'd1;
    localparam [4:0] PH_R0_REQ         = 5'd2;
    localparam [4:0] PH_R0_WAIT        = 5'd3;
    localparam [4:0] PH_R1_REQ         = 5'd4;
    localparam [4:0] PH_R1_WAIT        = 5'd5;
    localparam [4:0] PH_X0_CFG         = 5'd6;
    localparam [4:0] PH_X0_BLOCK       = 5'd7;
    localparam [4:0] PH_X0_WRITE       = 5'd8;
    localparam [4:0] PH_BAD_WRITE      = 5'd9;
    localparam [4:0] PH_X0_REQ         = 5'd10;
    localparam [4:0] PH_X0_WAIT        = 5'd11;
    localparam [4:0] PH_ABORT_WRITE    = 5'd12;
    localparam [4:0] PH_ABORT_REQ      = 5'd13;
    localparam [4:0] PH_ABORT_WAIT     = 5'd14;
    localparam [4:0] PH_ACCEPT_COLLIDE = 5'd15;
    localparam [4:0] PH_ACCEPT_WAIT    = 5'd16;
    localparam [4:0] PH_ACCEPT_REQ     = 5'd17;
    localparam [4:0] PH_ACCEPT_CHECK   = 5'd18;
    localparam [4:0] PH_ISSUE_REQ      = 5'd19;
    localparam [4:0] PH_ISSUE_WRITE    = 5'd20;
    localparam [4:0] PH_ISSUE_WAIT     = 5'd21;
    localparam [4:0] PH_ISSUE_CHECK_REQ = 5'd22;
    localparam [4:0] PH_ISSUE_CHECK    = 5'd23;
    localparam [4:0] PH_DONE           = 5'd24;

    function automatic [63:0] r_word(input token, input [1:0] bank);
        r_word = {8'h10, 53'd0, token, bank};
    endfunction

    function automatic [63:0] x0_word(input [1:0] bank);
        x0_word = {8'h20, 54'd0, bank};
    endfunction

    localparam [63:0] ACCEPT_WORD = 64'h3000_0000_0000_0000;
    localparam [63:0] ISSUE_WORD  = 64'h4000_0000_0000_0001;
    localparam [63:0] ABORT_WORD  = 64'h5000_0000_0000_0002;
    localparam [63:0] BAD_WORD    = 64'h6000_0000_0000_0000;

    (* anyseq *) reg rst_n;
    reg f_past_valid = 1'b0;
    reg [4:0] phase = PH_R0_WRITE;
    reg [1:0] index = 2'd0;

    wire wr_cfg_valid = phase == PH_X0_CFG;
    wire wr_cfg_ready;
    wire wr_busy;
    wire wr_done;
    wire wr_error;
    wire stream_valid = phase == PH_X0_WRITE;
    wire stream_ready;
    wire stream_last = stream_valid && index == 2'd3;
    wire wr_commit_valid;
    wire [1:0] wr_commit_bank;
    wire [13:0] wr_commit_address;

    wire r_wr_valid = (phase == PH_R0_WRITE) ||
                      (phase == PH_R1_WRITE) ||
                      (phase == PH_X0_BLOCK) ||
                      (phase == PH_BAD_WRITE) ||
                      (phase == PH_ABORT_WRITE) ||
                      (phase == PH_ACCEPT_COLLIDE) ||
                      (phase == PH_ISSUE_WRITE);
    wire r_wr_ready;
    wire [1:0] r_wr_bank = ((phase == PH_R0_WRITE) ||
                            (phase == PH_R1_WRITE)) ? index :
                           (phase == PH_ISSUE_WRITE) ? 2'd1 : 2'd0;
    wire [13:0] r_wr_address = (phase == PH_R1_WRITE ||
                                phase == PH_X0_BLOCK ||
                                phase == PH_ABORT_WRITE) ? 14'd512 :
                               (phase == PH_BAD_WRITE) ? 14'd2048 : 14'd0;
    wire [63:0] r_wr_data = (phase == PH_R0_WRITE) ? r_word(1'b0, index) :
                              (phase == PH_R1_WRITE) ? r_word(1'b1, index) :
                              (phase == PH_ABORT_WRITE) ? ABORT_WORD :
                              (phase == PH_BAD_WRITE) ? BAD_WORD :
                              (phase == PH_ACCEPT_COLLIDE) ? ACCEPT_WORD :
                              (phase == PH_ISSUE_WRITE) ? ISSUE_WORD : BAD_WORD;
    wire r_wr_error;
    wire r_wr_fire = r_wr_valid && r_wr_ready;

    wire rd_req_valid = (phase == PH_R0_REQ) ||
                        (phase == PH_R1_REQ) ||
                        (phase == PH_X0_REQ) ||
                        (phase == PH_ABORT_REQ) ||
                        (phase == PH_ACCEPT_COLLIDE) ||
                        (phase == PH_ACCEPT_REQ) ||
                        (phase == PH_ISSUE_REQ) ||
                        (phase == PH_ISSUE_CHECK_REQ);
    wire rd_req_ready;
    wire [1:0] rd_req_role = (phase == PH_X0_REQ) ? ROLE_X0 : ROLE_R;
    wire [2:0] rd_req_token = ((phase == PH_R1_REQ) ||
                               (phase == PH_ABORT_REQ)) ? 3'd1 : 3'd0;
    wire rd_issue_valid;
    wire [13:0] rd_issue_address;
    wire rd_rsp_valid;
    wire [255:0] rd_rsp_data;
    wire rd_rsp_error;

    wire wr_abort = phase == PH_ABORT_WRITE;

    section_f32_scratch dut (
        .clk(clk), .rst_n(rst_n),
        .wr_cfg_valid(wr_cfg_valid), .wr_cfg_ready(wr_cfg_ready),
        .wr_cfg_role(ROLE_X0), .wr_cfg_rows(14'd8),
        .wr_cfg_tokens(3'd1), .wr_abort(wr_abort),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error),
        .s_axis_tdata(x0_word(index)), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(stream_valid), .s_axis_tready(stream_ready),
        .s_axis_tlast(stream_last),
        .wr_commit_valid(wr_commit_valid), .wr_commit_bank(wr_commit_bank),
        .wr_commit_address(wr_commit_address),
        .r_wr_valid(r_wr_valid), .r_wr_ready(r_wr_ready),
        .r_wr_bank(r_wr_bank), .r_wr_address(r_wr_address),
        .r_wr_data(r_wr_data), .r_wr_error(r_wr_error),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(11'd0),
        .rd_issue_valid(rd_issue_valid), .rd_issue_address(rd_issue_address),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(1'b1),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) assume(!rst_n);
        else               assume(rst_n);

        if (!rst_n) begin
            phase <= PH_R0_WRITE;
            index <= 2'd0;
        end else begin
            case (phase)
                PH_R0_WRITE: if (r_wr_fire) begin
                    if (index == 2'd3) begin phase <= PH_R1_WRITE; index <= 2'd0; end
                    else index <= index + 1'b1;
                end
                PH_R1_WRITE: if (r_wr_fire) begin
                    if (index == 2'd3) begin phase <= PH_R0_REQ; index <= 2'd0; end
                    else index <= index + 1'b1;
                end
                PH_R0_REQ: if (rd_issue_valid) phase <= PH_R0_WAIT;
                PH_R0_WAIT: if (rd_rsp_valid) phase <= PH_R1_REQ;
                PH_R1_REQ: if (rd_issue_valid) phase <= PH_R1_WAIT;
                PH_R1_WAIT: if (rd_rsp_valid) phase <= PH_X0_CFG;
                PH_X0_CFG: if (wr_cfg_valid && wr_cfg_ready) phase <= PH_X0_BLOCK;
                PH_X0_BLOCK: phase <= PH_X0_WRITE;
                PH_X0_WRITE: if (stream_valid && stream_ready) begin
                    if (index == 2'd3) begin phase <= PH_BAD_WRITE; index <= 2'd0; end
                    else index <= index + 1'b1;
                end
                PH_BAD_WRITE: if (r_wr_fire) phase <= PH_X0_REQ;
                PH_X0_REQ: if (rd_issue_valid) phase <= PH_X0_WAIT;
                PH_X0_WAIT: if (rd_rsp_valid) phase <= PH_ABORT_WRITE;
                PH_ABORT_WRITE: phase <= PH_ABORT_REQ;
                PH_ABORT_REQ: if (rd_issue_valid) phase <= PH_ABORT_WAIT;
                PH_ABORT_WAIT: if (rd_rsp_valid) phase <= PH_ACCEPT_COLLIDE;
                PH_ACCEPT_COLLIDE: if (rd_issue_valid && r_wr_fire)
                    phase <= PH_ACCEPT_WAIT;
                PH_ACCEPT_WAIT: if (rd_rsp_valid) phase <= PH_ACCEPT_REQ;
                PH_ACCEPT_REQ: if (rd_issue_valid) phase <= PH_ACCEPT_CHECK;
                PH_ACCEPT_CHECK: if (rd_rsp_valid) phase <= PH_ISSUE_REQ;
                PH_ISSUE_REQ: if (rd_issue_valid) phase <= PH_ISSUE_WRITE;
                PH_ISSUE_WRITE: if (r_wr_fire) phase <= PH_ISSUE_WAIT;
                PH_ISSUE_WAIT: if (rd_rsp_valid) phase <= PH_ISSUE_CHECK_REQ;
                PH_ISSUE_CHECK_REQ: if (rd_issue_valid) phase <= PH_ISSUE_CHECK;
                PH_ISSUE_CHECK: if (rd_rsp_valid) phase <= PH_DONE;
                default: ;
            endcase
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!wr_busy && !wr_done && !wr_error);
            assert(!wr_commit_valid && !r_wr_error);
            assert(!rd_issue_valid && !rd_rsp_valid);
        end

        if (rst_n && f_past_valid && $past(rst_n)) begin
            assert(r_wr_ready == (!wr_busy && !wr_abort));
            assert(r_wr_error == (r_wr_valid && r_wr_ready &&
                                  r_wr_address >= 14'd2048));
            assert(wr_cfg_ready == (!wr_busy && !wr_abort && !r_wr_valid));

            if (phase == PH_X0_BLOCK) begin
                assert(wr_busy);
                assert(!r_wr_ready);
                assert(!r_wr_error);
            end
            if (phase == PH_ABORT_WRITE) begin
                assert(!r_wr_ready);
                assert(!r_wr_error);
            end
            if (phase == PH_BAD_WRITE) begin
                assert(r_wr_ready);
                assert(r_wr_error);
            end
            if (wr_commit_valid) begin
                assert(phase == PH_X0_WRITE);
                assert(wr_commit_bank == index);
                assert(wr_commit_address == 14'd2048);
            end

            if (rd_rsp_valid) begin
                case (phase)
                    PH_R0_WAIT:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {r_word(1'b0, 2'd3), r_word(1'b0, 2'd2),
                                r_word(1'b0, 2'd1), r_word(1'b0, 2'd0)});
                    PH_R1_WAIT:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {r_word(1'b1, 2'd3), r_word(1'b1, 2'd2),
                                r_word(1'b1, 2'd1), r_word(1'b1, 2'd0)});
                    PH_ABORT_WAIT:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {r_word(1'b1, 2'd3), r_word(1'b1, 2'd2),
                                r_word(1'b1, 2'd1), r_word(1'b1, 2'd0)});
                    PH_X0_WAIT:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {x0_word(2'd3), x0_word(2'd2),
                                x0_word(2'd1), x0_word(2'd0)});
                    PH_ACCEPT_WAIT:
                        assert(rd_rsp_error && rd_rsp_data == 256'd0);
                    PH_ISSUE_WAIT:
                        assert(rd_rsp_error && rd_rsp_data == 256'd0);
                    PH_ACCEPT_CHECK:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {r_word(1'b0, 2'd3), r_word(1'b0, 2'd2),
                                r_word(1'b0, 2'd1), ACCEPT_WORD});
                    PH_ISSUE_CHECK:
                        assert(!rd_rsp_error && rd_rsp_data ==
                               {r_word(1'b0, 2'd3), r_word(1'b0, 2'd2),
                                ISSUE_WORD, ACCEPT_WORD});
                    default: assert(1'b0);
                endcase
            end
        end

        cover(rst_n && phase == PH_DONE);
    end
endmodule

`default_nettype wire
