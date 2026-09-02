// Three-client transaction arbiter around the engine's sole Q8 leaf.
// A grant is held from cfg acceptance through the final output handshake.

`default_nettype none

module shared_q8 (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          abort_run,
    output wire          busy,
    output reg           collision_error,

    input  wire          c0_cfg_valid,
    output wire          c0_cfg_ready,
    input  wire [14:0]   c0_cfg_rows,
    input  wire [3:0]    c0_cfg_lane_mask,
    input  wire          c0_in_valid,
    output wire          c0_in_ready,
    input  wire [127:0]  c0_in_data,
    output wire          c0_out_valid,
    input  wire          c0_out_ready,
    output wire [8:0]    c0_out_block,
    output wire [1087:0] c0_out_data,
    output wire [7:0]    c0_out_status,
    output wire          c0_out_last,
    input  wire          c0_abort,

    input  wire          c1_cfg_valid,
    output wire          c1_cfg_ready,
    input  wire [14:0]   c1_cfg_rows,
    input  wire [3:0]    c1_cfg_lane_mask,
    input  wire          c1_in_valid,
    output wire          c1_in_ready,
    input  wire [127:0]  c1_in_data,
    output wire          c1_out_valid,
    input  wire          c1_out_ready,
    output wire [8:0]    c1_out_block,
    output wire [1087:0] c1_out_data,
    output wire [7:0]    c1_out_status,
    output wire          c1_out_last,
    input  wire          c1_abort,

    input  wire          c2_cfg_valid,
    output wire          c2_cfg_ready,
    input  wire [14:0]   c2_cfg_rows,
    input  wire [3:0]    c2_cfg_lane_mask,
    input  wire          c2_in_valid,
    output wire          c2_in_ready,
    input  wire [127:0]  c2_in_data,
    output wire          c2_out_valid,
    input  wire          c2_out_ready,
    output wire [8:0]    c2_out_block,
    output wire [1087:0] c2_out_data,
    output wire [7:0]    c2_out_status,
    output wire          c2_out_last,
    input  wire          c2_abort
);
    localparam [1:0] OWNER_NONE = 2'd0;
    localparam [1:0] OWNER_C0   = 2'd1;
    localparam [1:0] OWNER_C1   = 2'd2;
    localparam [1:0] OWNER_C2   = 2'd3;

    reg [1:0] owner_q;
    wire leaf_cfg_ready;
    wire leaf_busy;
    wire leaf_in_ready;
    wire leaf_out_valid;
    wire [8:0] leaf_out_block;
    wire [1087:0] leaf_out_data;
    wire [7:0] leaf_out_status;
    wire leaf_out_last;

    wire choose_c0 = (owner_q == OWNER_NONE) && c0_cfg_valid;
    wire choose_c1 = (owner_q == OWNER_NONE) && !c0_cfg_valid &&
                     c1_cfg_valid;
    wire choose_c2 = (owner_q == OWNER_NONE) && !c0_cfg_valid &&
                     !c1_cfg_valid && c2_cfg_valid;
    wire leaf_cfg_valid = choose_c0 || choose_c1 || choose_c2;
    wire leaf_cfg_fire = leaf_cfg_valid && leaf_cfg_ready;
    wire leaf_out_ready = owner_q == OWNER_C0 ? c0_out_ready :
                          owner_q == OWNER_C1 ? c1_out_ready :
                          owner_q == OWNER_C2 ? c2_out_ready : 1'b0;
    wire leaf_out_fire = leaf_out_valid && leaf_out_ready;
    wire owner_abort = (owner_q == OWNER_C0 && c0_abort) ||
                       (owner_q == OWNER_C1 && c1_abort) ||
                       (owner_q == OWNER_C2 && c2_abort);
    wire leaf_abort = abort_run || owner_abort;

    assign c0_cfg_ready = rst_n && !abort_run && choose_c0 &&
                          leaf_cfg_ready;
    assign c1_cfg_ready = rst_n && !abort_run && choose_c1 &&
                          leaf_cfg_ready;
    assign c2_cfg_ready = rst_n && !abort_run && choose_c2 &&
                          leaf_cfg_ready;
    assign c0_in_ready = (owner_q == OWNER_C0) && leaf_in_ready;
    assign c1_in_ready = (owner_q == OWNER_C1) && leaf_in_ready;
    assign c2_in_ready = (owner_q == OWNER_C2) && leaf_in_ready;
    assign c0_out_valid = (owner_q == OWNER_C0) && leaf_out_valid;
    assign c1_out_valid = (owner_q == OWNER_C1) && leaf_out_valid;
    assign c2_out_valid = (owner_q == OWNER_C2) && leaf_out_valid;
    assign c0_out_block = leaf_out_block;
    assign c1_out_block = leaf_out_block;
    assign c2_out_block = leaf_out_block;
    assign c0_out_data = leaf_out_data;
    assign c1_out_data = leaf_out_data;
    assign c2_out_data = leaf_out_data;
    assign c0_out_status = leaf_out_status;
    assign c1_out_status = leaf_out_status;
    assign c2_out_status = leaf_out_status;
    assign c0_out_last = leaf_out_last;
    assign c1_out_last = leaf_out_last;
    assign c2_out_last = leaf_out_last;
    assign busy = (owner_q != OWNER_NONE) || leaf_busy;

     q8_pack4 u_q8 (
        .clk(clk), .rst_n(rst_n), .cfg_valid(leaf_cfg_valid),
        .cfg_ready(leaf_cfg_ready),
        .cfg_rows(choose_c0 ? c0_cfg_rows :
                  choose_c1 ? c1_cfg_rows : c2_cfg_rows),
        .cfg_lane_mask(choose_c0 ? c0_cfg_lane_mask :
                       choose_c1 ? c1_cfg_lane_mask :
                                   c2_cfg_lane_mask),
        .abort_run(leaf_abort), .busy(leaf_busy),
        .in_valid(owner_q == OWNER_C0 ? c0_in_valid :
                  owner_q == OWNER_C1 ? c1_in_valid :
                  owner_q == OWNER_C2 ? c2_in_valid : 1'b0),
        .in_ready(leaf_in_ready),
        .in_data(owner_q == OWNER_C0 ? c0_in_data :
                 owner_q == OWNER_C1 ? c1_in_data : c2_in_data),
        .out_valid(leaf_out_valid), .out_ready(leaf_out_ready),
        .out_block(leaf_out_block), .out_data(leaf_out_data),
        .out_status(leaf_out_status), .out_last(leaf_out_last)
    );

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            owner_q <= OWNER_NONE;
            collision_error <= 1'b0;
        end else begin
            if ((owner_q == OWNER_NONE) &&
                ((c0_cfg_valid && c1_cfg_valid) ||
                 (c0_cfg_valid && c2_cfg_valid) ||
                 (c1_cfg_valid && c2_cfg_valid)))
                collision_error <= 1'b1;
            if (leaf_cfg_fire)
                owner_q <= choose_c0 ? OWNER_C0 :
                           choose_c1 ? OWNER_C1 : OWNER_C2;
            if (leaf_out_fire && leaf_out_last)
                owner_q <= OWNER_NONE;
            if (owner_abort)
                owner_q <= OWNER_NONE;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && leaf_out_valid && (owner_q == OWNER_NONE))
            $fatal(1, " shared_q8 output without owner");
        if (rst_n && (owner_q != OWNER_C1) && c1_in_valid)
            $error(" shared_q8 non-owner c1 presented data");
        if (rst_n && (owner_q != OWNER_C0) && c0_in_valid)
            $error(" shared_q8 non-owner c0 presented data");
        if (rst_n && (owner_q != OWNER_C2) && c2_in_valid)
            $error(" shared_q8 non-owner c2 presented data");
    end
`endif
endmodule

`default_nettype wire
