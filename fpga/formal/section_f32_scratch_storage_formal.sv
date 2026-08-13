`default_nettype none

// Reduced-lifecycle symbolic-array proof.  It drives the real full-size memories,
// but only reaches two 16-row/two-token roles: X0 commits completely before X1,
// then every group is read back.  Address correctness for all production shapes is
// proved separately by section_f32_scratch_map_formal.
module section_f32_scratch_storage_formal(input wire clk);
    localparam [1:0] ROLE_X0 = 2'd1;
    localparam [1:0] ROLE_X1 = 2'd2;

    localparam [3:0] PH_RESET  = 4'd0;
    localparam [3:0] PH_CFG_X0 = 4'd1;
    localparam [3:0] PH_WR_X0  = 4'd2;
    localparam [3:0] PH_CFG_X1 = 4'd3;
    localparam [3:0] PH_WR_X1  = 4'd4;
    localparam [3:0] PH_RD_X0  = 4'd5;
    localparam [3:0] PH_RD_X1  = 4'd6;
    localparam [3:0] PH_DRAIN  = 4'd7;
    localparam [3:0] PH_DONE   = 4'd8;

    (* anyseq *) reg [63:0] stream_data;
    (* anyseq *) reg        rst_n;

    reg f_past_valid = 1'b0;
    reg [3:0] phase = PH_RESET;
    reg [4:0] write_index = 5'd0;
    reg [2:0] read_index = 3'd0;
    reg pending_read_d1 = 1'b0;
    reg pending_read_d2 = 1'b0;
    reg pending_read_d3 = 1'b0;
    reg pending_role_x1_d1 = 1'b0;
    reg pending_role_x1_d2 = 1'b0;
    reg pending_role_x1_d3 = 1'b0;
    reg [2:0] pending_index_d1 = 3'd0;
    reg [2:0] pending_index_d2 = 3'd0;
    reg [2:0] pending_index_d3 = 3'd0;
    reg x0_valid = 1'b0;
    reg x1_valid = 1'b0;

    // Four banks x two row groups x two tokens for each role.
    reg [63:0] expected_x0 [0:15];
    reg [63:0] expected_x1 [0:15];

    wire wr_cfg_valid = (phase == PH_CFG_X0) || (phase == PH_CFG_X1);
    wire [1:0] wr_cfg_role = (phase == PH_CFG_X1) ? ROLE_X1 : ROLE_X0;
    wire wr_cfg_ready;
    wire wr_busy;
    wire wr_done;
    wire wr_error;

    wire writing_x0 = phase == PH_WR_X0;
    wire writing_x1 = phase == PH_WR_X1;
    wire stream_valid = writing_x0 || writing_x1;
    wire stream_ready;
    wire stream_last = stream_valid && (write_index == 5'd15);
    wire wr_commit_valid;
    wire [1:0] wr_commit_bank;
    wire [13:0] wr_commit_address;

    wire reading_x0 = phase == PH_RD_X0;
    wire reading_x1 = phase == PH_RD_X1;
    wire rd_req_valid = reading_x0 || reading_x1;
    wire [1:0] rd_req_role = reading_x1 ? ROLE_X1 : ROLE_X0;
    wire [2:0] rd_req_token = {2'b00, read_index[1]};
    wire [10:0] rd_req_group = {10'd0, read_index[0]};
    wire rd_req_ready;
    wire rd_issue_valid;
    wire [13:0] rd_issue_address;
    wire rd_rsp_valid;
    wire [255:0] rd_rsp_data;
    wire rd_rsp_error;

    section_f32_scratch dut (
        .clk(clk), .rst_n(rst_n),
        .wr_cfg_valid(wr_cfg_valid), .wr_cfg_ready(wr_cfg_ready),
        .wr_cfg_role(wr_cfg_role), .wr_cfg_rows(14'd16),
        .wr_cfg_tokens(3'd2), .wr_abort(1'b0),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error),
        .s_axis_tdata(stream_data), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(stream_valid), .s_axis_tready(stream_ready),
        .s_axis_tlast(stream_last),
        .wr_commit_valid(wr_commit_valid), .wr_commit_bank(wr_commit_bank),
        .wr_commit_address(wr_commit_address),
        .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role), .rd_req_token(rd_req_token),
        .rd_req_group(rd_req_group),
        .rd_issue_valid(rd_issue_valid), .rd_issue_address(rd_issue_address),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(1'b1),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error)
    );

    integer i;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) assume(!rst_n);
        else               assume(rst_n);

        if (!rst_n) begin
            phase <= PH_CFG_X0;
            write_index <= 5'd0;
            read_index <= 3'd0;
            pending_read_d1 <= 1'b0;
            pending_read_d2 <= 1'b0;
            pending_read_d3 <= 1'b0;
            pending_role_x1_d1 <= 1'b0;
            pending_role_x1_d2 <= 1'b0;
            pending_role_x1_d3 <= 1'b0;
            pending_index_d1 <= 3'd0;
            pending_index_d2 <= 3'd0;
            pending_index_d3 <= 3'd0;
            x0_valid <= 1'b0;
            x1_valid <= 1'b0;
        end else begin
            pending_read_d3 <= pending_read_d2;
            pending_role_x1_d3 <= pending_role_x1_d2;
            pending_index_d3 <= pending_index_d2;
            pending_read_d2 <= pending_read_d1;
            pending_role_x1_d2 <= pending_role_x1_d1;
            pending_index_d2 <= pending_index_d1;
            pending_read_d1 <= rd_issue_valid;
            if (rd_issue_valid) begin
                pending_role_x1_d1 <= rd_req_role == ROLE_X1;
                pending_index_d1 <= read_index;
            end

            case (phase)
                PH_CFG_X0: if (wr_cfg_valid && wr_cfg_ready) begin
                    phase <= PH_WR_X0;
                    write_index <= 5'd0;
                    x0_valid <= 1'b0;
                end

                PH_WR_X0: if (stream_valid && stream_ready) begin
                    expected_x0[write_index[3:0]] <= stream_data;
                    if (write_index == 5'd15) begin
                        phase <= PH_CFG_X1;
                        write_index <= 5'd0;
                        x0_valid <= 1'b1;
                    end else begin
                        write_index <= write_index + 1'b1;
                    end
                end

                PH_CFG_X1: if (wr_cfg_valid && wr_cfg_ready) begin
                    phase <= PH_WR_X1;
                    write_index <= 5'd0;
                    x1_valid <= 1'b0;
                end

                PH_WR_X1: if (stream_valid && stream_ready) begin
                    expected_x1[write_index[3:0]] <= stream_data;
                    if (write_index == 5'd15) begin
                        phase <= PH_RD_X0;
                        write_index <= 5'd0;
                        read_index <= 3'd0;
                        x1_valid <= 1'b1;
                    end else begin
                        write_index <= write_index + 1'b1;
                    end
                end

                PH_RD_X0: if (rd_issue_valid) begin
                    if (read_index == 3'd3) begin
                        phase <= PH_RD_X1;
                        read_index <= 3'd0;
                    end else begin
                        read_index <= read_index + 1'b1;
                    end
                end

                PH_RD_X1: if (rd_issue_valid) begin
                    if (read_index == 3'd3) begin
                        phase <= PH_DRAIN;
                        read_index <= 3'd0;
                    end else begin
                        read_index <= read_index + 1'b1;
                    end
                end

                PH_DRAIN: if (rd_rsp_valid) phase <= PH_DONE;
                default: ;
            endcase
        end

        if (f_past_valid && !$past(rst_n)) begin
            assert(!wr_busy && !wr_done && !wr_error);
            assert(!wr_commit_valid && !rd_issue_valid && !rd_rsp_valid);
        end

        if (rst_n && f_past_valid && $past(rst_n)) begin
            // This harness is the ownership boundary: a role becomes readable
            // only after its complete, clean GEMM stream has committed.
            if (reading_x0) assert(x0_valid);
            if (reading_x1) assert(x0_valid && x1_valid);
            assert(!(x1_valid && !x0_valid));

            if (wr_commit_valid) begin
                assert(writing_x0 || writing_x1);
                assert(wr_commit_bank == write_index[1:0]);
                if (writing_x0) begin
                    assert(!x0_valid);
                    assert(wr_commit_address == 14'd2048 +
                           (write_index[3] ? 14'd1536 : 14'd0) +
                           (write_index[2] ? 14'd1 : 14'd0));
                end else begin
                    assert(x0_valid && !x1_valid);
                    assert(wr_commit_address == 14'd8192 +
                           (write_index[3] ? 14'd1536 : 14'd0) +
                           (write_index[2] ? 14'd1 : 14'd0));
                end
            end

            if (rd_issue_valid) begin
                assert(x0_valid && x1_valid);
                assert(rd_issue_address ==
                       (reading_x1 ? 14'd8192 : 14'd2048) +
                       (read_index[1] ? 14'd1536 : 14'd0) +
                       (read_index[0] ? 14'd1 : 14'd0));
            end

            if (pending_read_d1) begin
                assert(!rd_req_ready);
                assert(!rd_rsp_valid);
            end

            if (pending_read_d2) begin
                assert(!rd_req_ready);
                assert(!rd_rsp_valid);
            end

            if (pending_read_d3) begin
                assert(rd_rsp_valid);
                assert(!rd_rsp_error);
                for (i = 0; i < 4; i = i + 1) begin
                    if (pending_role_x1_d3)
                        assert(rd_rsp_data[i*64 +: 64] ==
                               expected_x1[{pending_index_d3, 2'b00} + i]);
                    else
                        assert(rd_rsp_data[i*64 +: 64] ==
                               expected_x0[{pending_index_d3, 2'b00} + i]);
                end
            end

            assert(!wr_error);
        end

        cover(rst_n && phase == PH_DONE && x0_valid && x1_valid);
    end
endmodule

`default_nettype wire
