`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1 clk = ~clk;

    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire busy;
    wire done;

    localparam integer Q_BEATS = 8 * 8;
    localparam integer KV_BEATS = 2 * 4;
    localparam integer MASK_BEATS = 2 * 8;
    localparam integer O_BEATS = 8 * 8;

    integer cycles = 0;
    integer q_count = 0;
    integer k_count = 0;
    integer v_count = 0;
    integer mask_count = 0;
    integer o_count = 0;

    wire q_valid = q_count < Q_BEATS;
    wire k_valid = k_count < KV_BEATS;
    wire v_valid = v_count < KV_BEATS;
    wire mask_valid = mask_count < MASK_BEATS;
    wire q_ready, k_ready, v_ready, mask_ready;
    wire [255:0] o_data;
    wire o_valid;
    wire o_ready = cycles[1:0] != 2'b11;
    wire o_last;
    wire [31:0] o_keep;

    function automatic [15:0] half_int(input integer value);
        begin
            case (value)
                1: half_int = 16'h3c00;
                2: half_int = 16'h4000;
                3: half_int = 16'h4200;
                4: half_int = 16'h4400;
                5: half_int = 16'h4500;
                6: half_int = 16'h4600;
                7: half_int = 16'h4700;
                default: half_int = 16'h4800;
            endcase
        end
    endfunction

    function automatic [31:0] float_int(input integer value);
        begin
            case (value)
                1: float_int = 32'h3f800000;
                2: float_int = 32'h40000000;
                3: float_int = 32'h40400000;
                4: float_int = 32'h40800000;
                5: float_int = 32'h40a00000;
                6: float_int = 32'h40c00000;
                7: float_int = 32'h40e00000;
                default: float_int = 32'h41000000;
            endcase
        end
    endfunction

    wire [15:0] v_lane = half_int(1 + (v_count / 4) * 4 + (v_count % 4));
    wire [127:0] v_payload = {8{v_lane}};
    wire [15:0] mask_payload = ((mask_count % 8) % 2 == mask_count / 8)
        ? 16'h0000 : 16'hfc00;
    integer expected_value;
    reg [255:0] expected_payload;

    flash_kernel #(
        .HEAD_DIM_MAX(128),
        .MAX_HEADS(8),
        .MAX_HEAD_KV(8),
        .MAX_TOKENS(8),
        .MAX_SLOTS(64),
        .LANES(8),
        .TILE8_HEAD8_LAYOUT(1)
    ) dut (
        .clk(clk), .rst_n(rst_n), .start(start),
        .head_dim_q(16'd8), .head_dim_v(16'd8),
        .n_heads(16'd8), .n_head_kv(16'd4), .head_ratio(16'd2),
        .n_kv(17'd2), .n_tokens(16'd8), .scale(32'h3e800000),
        .busy(busy), .done(done),
        .q_tdata(256'd0), .q_tvalid(q_valid), .q_tready(q_ready),
        .q_tlast(q_count + 1 == Q_BEATS), .q_tkeep(32'hffffffff),
        .k_tdata(128'd0), .k_tvalid(k_valid), .k_tready(k_ready),
        .k_tlast(k_count + 1 == KV_BEATS), .k_tkeep(16'hffff),
        .v_tdata(v_payload), .v_tvalid(v_valid), .v_tready(v_ready),
        .v_tlast(v_count + 1 == KV_BEATS), .v_tkeep(16'hffff),
        .mask_tdata(mask_payload), .mask_tvalid(mask_valid), .mask_tready(mask_ready),
        .mask_tlast(mask_count + 1 == MASK_BEATS), .mask_tkeep(2'b11),
        .o_tdata(o_data), .o_tvalid(o_valid), .o_tready(o_ready),
        .o_tlast(o_last), .o_tkeep(o_keep)
    );

    always @(posedge clk) begin
        cycles <= cycles + 1;
        if (rst_n) begin
            if (q_valid && q_ready) q_count <= q_count + 1;
            if (k_valid && k_ready) k_count <= k_count + 1;
            if (v_valid && v_ready) v_count <= v_count + 1;
            if (mask_valid && mask_ready) mask_count <= mask_count + 1;
            if (o_valid && o_ready) begin
                expected_value = 1 + ((o_count / 8) % 2) * 4 + ((o_count % 8) / 2);
                expected_payload = {8{float_int(expected_value)}};
                if (o_data !== expected_payload)
                    $fatal(1, "bad output beat %0d value=%0d got=%h",
                        o_count, expected_value, o_data);
                if (o_keep !== 32'hffffffff) $fatal(1, "bad keep beat %0d", o_count);
                if (o_last !== (o_count + 1 == O_BEATS))
                    $fatal(1, "bad last beat %0d", o_count);
                o_count <= o_count + 1;
            end
            if (done) begin
                if (q_count != Q_BEATS || k_count != KV_BEATS ||
                    v_count != KV_BEATS || mask_count != MASK_BEATS ||
                    o_count != O_BEATS)
                    $fatal(1, "count mismatch q=%0d k=%0d v=%0d mask=%0d o=%0d",
                        q_count, k_count, v_count, mask_count, o_count);
                $display("attention-kernel PASS cycles=%0d q=%0d k=%0d v=%0d o=%0d",
                    cycles, q_count, k_count, v_count, o_count);
                $finish;
            end
            if (cycles > 200000) $fatal(1, "timeout");
        end
    end

    initial begin
        repeat (6) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
    end
endmodule

`default_nettype wire
