`default_nettype none

// Maximum-shape harness for the adaptive query/head slot map. Arithmetic and the
// scheduler are intentionally outside this proof cone: the regular flash harness
// proves the complete two-query schedule, while this harness proves that the actual
// production mapping is injective and in range at both 4x16 and 2x32 maxima.
module flash_slots_formal(input wire clk);
    (* anyseq *) reg rst_n;
    (* anyseq *) reg start;
    (* anyconst *) reg wide_mode;

    wire [15:0] command_heads = wide_mode ? 16'd32 : 16'd16;
    wire [15:0] command_kv_heads = wide_mode ? 16'd8 : 16'd4;
    wire [15:0] command_tokens = wide_mode ? 16'd2 : 16'd4;

    wire busy, done;
    wire q_tready, k_tready, v_tready, mask_tready;
    wire [255:0] o_tdata;
    wire o_tvalid, o_tlast;
    wire [31:0] o_tkeep;

    flash_kernel #(
        .HEAD_DIM_MAX(128),
        .MAX_HEADS(32),
        .MAX_HEAD_KV(8),
        .MAX_TOKENS(4),
        .MAX_SLOTS(64),
        .LANES(8)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .head_dim_q(16'd128), .head_dim_v(16'd128),
        .n_heads(command_heads), .n_head_kv(command_kv_heads), .head_ratio(16'd4),
        .n_kv(16'd1), .n_tokens(command_tokens), .scale(32'h3f000000),
        .busy(busy), .done(done),
        .q_tdata(256'd0), .q_tvalid(1'b0), .q_tready(q_tready),
        .k_tdata(128'd0), .k_tvalid(1'b0), .k_tready(k_tready),
        .v_tdata(128'd0), .v_tvalid(1'b0), .v_tready(v_tready),
        .mask_tdata(16'd0), .mask_tvalid(1'b0), .mask_tready(mask_tready),
        .o_tdata(o_tdata), .o_tvalid(o_tvalid), .o_tready(1'b0),
        .q_tlast(1'b0), .q_tkeep(32'hffffffff),
        .k_tlast(1'b0), .k_tkeep(16'hffff),
        .v_tlast(1'b0), .v_tkeep(16'hffff),
        .mask_tlast(1'b0), .mask_tkeep(2'b11),
        .o_tlast(o_tlast), .o_tkeep(o_tkeep)
    );

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) begin
            assume(!rst_n);
            assume(!start);
        end else begin
            assume(rst_n);
            assume(start == !$past(rst_n));
        end
    end

    wire _unused = &{1'b0, done, q_tready, k_tready, v_tready,
        mask_tready, o_tdata, o_tvalid, o_tlast, o_tkeep};
endmodule

`default_nettype wire
