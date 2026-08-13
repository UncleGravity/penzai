// Registered-boundary OOC probe for the complete contract-v1 FP32 scratch.
// Expected XCK26 inference is exactly 16 URAM288s, with no BRAM or LUTRAM.

`default_nettype none

module section_f32_scratch_ooc (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         wr_cfg_valid,
    input  wire [1:0]   wr_cfg_role,
    input  wire [13:0]  wr_cfg_rows,
    input  wire [2:0]   wr_cfg_tokens,
    input  wire         wr_abort,
    input  wire [63:0]  s_axis_tdata,
    input  wire [7:0]   s_axis_tkeep,
    input  wire         s_axis_tvalid,
    input  wire         s_axis_tlast,
    input  wire         rd_req_valid,
    input  wire [1:0]   rd_req_role,
    input  wire [2:0]   rd_req_token,
    input  wire [10:0]  rd_req_group,
    input  wire         rd_rsp_ready,
    output reg          wr_cfg_ready_q,
    output reg          wr_busy_q,
    output reg          wr_done_q,
    output reg          wr_error_q,
    output reg          s_axis_tready_q,
    output reg          wr_commit_valid_q,
    output reg [1:0]    wr_commit_bank_q,
    output reg [13:0]   wr_commit_address_q,
    output reg          rd_req_ready_q,
    output reg          rd_issue_valid_q,
    output reg [13:0]   rd_issue_address_q,
    output reg          rd_rsp_valid_q,
    output reg [255:0]  rd_rsp_data_q,
    output reg          rd_rsp_error_q
);
    reg wr_cfg_valid_i;
    reg [1:0] wr_cfg_role_i;
    reg [13:0] wr_cfg_rows_i;
    reg [2:0] wr_cfg_tokens_i;
    reg wr_abort_i;
    reg [63:0] s_axis_tdata_i;
    reg [7:0] s_axis_tkeep_i;
    reg s_axis_tvalid_i;
    reg s_axis_tlast_i;
    reg rd_req_valid_i;
    reg [1:0] rd_req_role_i;
    reg [2:0] rd_req_token_i;
    reg [10:0] rd_req_group_i;
    reg rd_rsp_ready_i;

    always @(posedge clk) begin
        wr_cfg_valid_i <= wr_cfg_valid;
        wr_cfg_role_i <= wr_cfg_role;
        wr_cfg_rows_i <= wr_cfg_rows;
        wr_cfg_tokens_i <= wr_cfg_tokens;
        wr_abort_i <= wr_abort;
        s_axis_tdata_i <= s_axis_tdata;
        s_axis_tkeep_i <= s_axis_tkeep;
        s_axis_tvalid_i <= s_axis_tvalid;
        s_axis_tlast_i <= s_axis_tlast;
        rd_req_valid_i <= rd_req_valid;
        rd_req_role_i <= rd_req_role;
        rd_req_token_i <= rd_req_token;
        rd_req_group_i <= rd_req_group;
        rd_rsp_ready_i <= rd_rsp_ready;
    end

    wire wr_cfg_ready;
    wire wr_busy;
    wire wr_done;
    wire wr_error;
    wire s_axis_tready;
    wire wr_commit_valid;
    wire [1:0] wr_commit_bank;
    wire [13:0] wr_commit_address;
    wire rd_req_ready;
    wire rd_issue_valid;
    wire [13:0] rd_issue_address;
    wire rd_rsp_valid;
    wire [255:0] rd_rsp_data;
    wire rd_rsp_error;

    section_f32_scratch u_scratch (
        .clk(clk), .rst_n(rst_n),
        .wr_cfg_valid(wr_cfg_valid_i), .wr_cfg_ready(wr_cfg_ready),
        .wr_cfg_role(wr_cfg_role_i), .wr_cfg_rows(wr_cfg_rows_i),
        .wr_cfg_tokens(wr_cfg_tokens_i), .wr_abort(wr_abort_i),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error),
        .s_axis_tdata(s_axis_tdata_i), .s_axis_tkeep(s_axis_tkeep_i),
        .s_axis_tvalid(s_axis_tvalid_i), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast_i),
        .wr_commit_valid(wr_commit_valid), .wr_commit_bank(wr_commit_bank),
        .wr_commit_address(wr_commit_address),
        .rd_req_valid(rd_req_valid_i), .rd_req_ready(rd_req_ready),
        .rd_req_role(rd_req_role_i), .rd_req_token(rd_req_token_i),
        .rd_req_group(rd_req_group_i),
        .rd_issue_valid(rd_issue_valid), .rd_issue_address(rd_issue_address),
        .rd_rsp_valid(rd_rsp_valid), .rd_rsp_ready(rd_rsp_ready_i),
        .rd_rsp_data(rd_rsp_data), .rd_rsp_error(rd_rsp_error)
    );

    always @(posedge clk) begin
        wr_cfg_ready_q <= wr_cfg_ready;
        wr_busy_q <= wr_busy;
        wr_done_q <= wr_done;
        wr_error_q <= wr_error;
        s_axis_tready_q <= s_axis_tready;
        wr_commit_valid_q <= wr_commit_valid;
        wr_commit_bank_q <= wr_commit_bank;
        wr_commit_address_q <= wr_commit_address;
        rd_req_ready_q <= rd_req_ready;
        rd_issue_valid_q <= rd_issue_valid;
        rd_issue_address_q <= rd_issue_address;
        rd_rsp_valid_q <= rd_rsp_valid;
        rd_rsp_data_q <= rd_rsp_data;
        rd_rsp_error_q <= rd_rsp_error;
    end
endmodule

`default_nettype wire
