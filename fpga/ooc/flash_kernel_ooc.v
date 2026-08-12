// Full flash-kernel OOC timing/resource probe. Registered boundaries keep the
// controller, BRAM paths, and composed numeric pipelines visible when top-level
// I/O is false-pathed by ooc_synth.tcl.

`default_nettype none

module flash_kernel_ooc (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [15:0]  head_dim_q,
    input  wire [15:0]  head_dim_v,
    input  wire [15:0]  n_heads,
    input  wire [15:0]  n_head_kv,
    input  wire [15:0]  head_ratio,
    input  wire [15:0]  n_kv,
    input  wire [15:0]  n_tokens,
    input  wire [31:0]  scale,
    input  wire [255:0] q_tdata,
    input  wire         q_tvalid,
    input  wire         q_tlast,
    input  wire [31:0]  q_tkeep,
    input  wire [127:0] k_tdata,
    input  wire         k_tvalid,
    input  wire         k_tlast,
    input  wire [15:0]  k_tkeep,
    input  wire [127:0] v_tdata,
    input  wire         v_tvalid,
    input  wire         v_tlast,
    input  wire [15:0]  v_tkeep,
    input  wire [15:0]  mask_tdata,
    input  wire         mask_tvalid,
    input  wire         mask_tlast,
    input  wire [1:0]   mask_tkeep,
    input  wire         o_tready,
    output reg          busy_q,
    output reg          done_q,
    output reg          q_tready_q,
    output reg          k_tready_q,
    output reg          v_tready_q,
    output reg          mask_tready_q,
    output reg  [255:0] o_tdata_q,
    output reg          o_tvalid_q,
    output reg          o_tlast_q,
    output reg  [31:0]  o_tkeep_q
);
    reg         start_i;
    reg [15:0]  head_dim_q_i, head_dim_v_i, n_heads_i, n_head_kv_i;
    reg [15:0]  head_ratio_i, n_kv_i, n_tokens_i;
    reg [31:0]  scale_i;
    reg [255:0] q_tdata_i;
    reg         q_tvalid_i, q_tlast_i;
    reg [31:0]  q_tkeep_i;
    reg [127:0] k_tdata_i, v_tdata_i;
    reg         k_tvalid_i, k_tlast_i, v_tvalid_i, v_tlast_i;
    reg [15:0]  k_tkeep_i, v_tkeep_i;
    reg [15:0]  mask_tdata_i;
    reg         mask_tvalid_i, mask_tlast_i;
    reg [1:0]   mask_tkeep_i;
    reg         o_tready_i;

    always @(posedge clk) begin
        start_i       <= start;
        head_dim_q_i  <= head_dim_q;
        head_dim_v_i  <= head_dim_v;
        n_heads_i     <= n_heads;
        n_head_kv_i   <= n_head_kv;
        head_ratio_i  <= head_ratio;
        n_kv_i        <= n_kv;
        n_tokens_i    <= n_tokens;
        scale_i       <= scale;
        q_tdata_i     <= q_tdata;
        q_tvalid_i    <= q_tvalid;
        q_tlast_i     <= q_tlast;
        q_tkeep_i     <= q_tkeep;
        k_tdata_i     <= k_tdata;
        k_tvalid_i    <= k_tvalid;
        k_tlast_i     <= k_tlast;
        k_tkeep_i     <= k_tkeep;
        v_tdata_i     <= v_tdata;
        v_tvalid_i    <= v_tvalid;
        v_tlast_i     <= v_tlast;
        v_tkeep_i     <= v_tkeep;
        mask_tdata_i  <= mask_tdata;
        mask_tvalid_i <= mask_tvalid;
        mask_tlast_i  <= mask_tlast;
        mask_tkeep_i  <= mask_tkeep;
        o_tready_i    <= o_tready;
    end

    wire         busy, done, q_ready, k_ready, v_ready, mask_ready;
    wire [255:0] o_data;
    wire         o_valid, o_last;
    wire [31:0]  o_keep;

    flash_kernel #(
        .HEAD_DIM_MAX(128), .MAX_HEADS(32), .MAX_HEAD_KV(8), .LANES(8)
    ) u_kernel (
        .clk(clk), .rst_n(rst_n),
        .start(start_i),
        .head_dim_q(head_dim_q_i), .head_dim_v(head_dim_v_i),
        .n_heads(n_heads_i), .n_head_kv(n_head_kv_i), .head_ratio(head_ratio_i),
        .n_kv(n_kv_i), .n_tokens(n_tokens_i), .scale(scale_i),
        .busy(busy), .done(done),
        .q_tdata(q_tdata_i), .q_tvalid(q_tvalid_i), .q_tready(q_ready),
        .k_tdata(k_tdata_i), .k_tvalid(k_tvalid_i), .k_tready(k_ready),
        .v_tdata(v_tdata_i), .v_tvalid(v_tvalid_i), .v_tready(v_ready),
        .mask_tdata(mask_tdata_i), .mask_tvalid(mask_tvalid_i), .mask_tready(mask_ready),
        .o_tdata(o_data), .o_tvalid(o_valid), .o_tready(o_tready_i),
        .q_tlast(q_tlast_i), .q_tkeep(q_tkeep_i),
        .k_tlast(k_tlast_i), .k_tkeep(k_tkeep_i),
        .v_tlast(v_tlast_i), .v_tkeep(v_tkeep_i),
        .mask_tlast(mask_tlast_i), .mask_tkeep(mask_tkeep_i),
        .o_tlast(o_last), .o_tkeep(o_keep)
    );

    always @(posedge clk) begin
        busy_q       <= busy;
        done_q       <= done;
        q_tready_q   <= q_ready;
        k_tready_q   <= k_ready;
        v_tready_q   <= v_ready;
        mask_tready_q <= mask_ready;
        o_tdata_q    <= o_data;
        o_tvalid_q   <= o_valid;
        o_tlast_q    <= o_last;
        o_tkeep_q    <= o_keep;
    end
endmodule
