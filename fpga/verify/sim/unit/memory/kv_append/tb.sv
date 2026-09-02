`timescale 1ns/1ps

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;

    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [63:0] cmd_layer_kv_base = 64'h0010_0000;
    reg [16:0] cmd_position_base = 17'd7;
    reg [3:0] cmd_token_count = 4'd5;
    reg [7:0] cmd_token_mask = 8'h1f;

    wire newkv_rd_req_valid;
    reg newkv_rd_req_ready = 1'b1;
    wire newkv_rd_req_wave;
    wire [10:0] newkv_rd_req_addr;
    reg newkv_rd_rsp_valid = 1'b0;
    wire newkv_rd_rsp_ready;
    reg [63:0] newkv_rd_rsp_data = 64'd0;
    reg newkv_rd_rsp_error = 1'b0;

    wire wr_cmd_valid;
    reg wr_cmd_ready = 1'b1;
    wire [63:0] wr_cmd_addr;
    wire [31:0] wr_cmd_segment_beats;
    wire [31:0] wr_cmd_stride_bytes;
    wire [16:0] wr_cmd_repeats;
    wire [127:0] wr_data;
    wire wr_valid;
    reg wr_ready = 1'b1;
    wire wr_last;
    wire wr_error;
    reg wr_busy = 1'b0;
    reg wr_done_valid = 1'b0;
    wire wr_done_ready;
    reg wr_done_error = 1'b0;
    reg [7:0] wr_done_status = 8'd0;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;

     kv_append8 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_layer_kv_base(cmd_layer_kv_base),
        .cmd_position_base(cmd_position_base),
        .cmd_token_count(cmd_token_count), .cmd_token_mask(cmd_token_mask),
        .newkv_rd_req_valid(newkv_rd_req_valid),
        .newkv_rd_req_ready(newkv_rd_req_ready),
        .newkv_rd_req_wave(newkv_rd_req_wave),
        .newkv_rd_req_addr(newkv_rd_req_addr),
        .newkv_rd_rsp_valid(newkv_rd_rsp_valid),
        .newkv_rd_rsp_ready(newkv_rd_rsp_ready),
        .newkv_rd_rsp_data(newkv_rd_rsp_data),
        .newkv_rd_rsp_error(newkv_rd_rsp_error),
        .wr_cmd_valid(wr_cmd_valid), .wr_cmd_ready(wr_cmd_ready),
        .wr_cmd_addr(wr_cmd_addr),
        .wr_cmd_segment_beats(wr_cmd_segment_beats),
        .wr_cmd_stride_bytes(wr_cmd_stride_bytes),
        .wr_cmd_repeats(wr_cmd_repeats), .wr_data(wr_data),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_last(wr_last),
        .wr_error(wr_error), .wr_busy(wr_busy),
        .wr_done_valid(wr_done_valid), .wr_done_ready(wr_done_ready),
        .wr_done_error(wr_done_error), .wr_done_status(wr_done_status),
        .busy(busy), .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status)
    );

    function automatic [15:0] scalar_value(
        input [2:0] token,
        input kind,
        input [2:0] head,
        input [3:0] beat,
        input [2:0] scalar
    );
        scalar_value = {2'b01, kind, token, head, beat, scalar};
    endfunction

    integer cycles = 0;
    integer req_delay = 0;
    reg req_pending = 1'b0;
    reg req_wave_q = 1'b0;
    reg [10:0] req_addr_q = 11'd0;
    reg hold_responses = 1'b0;
    integer lane;

    always @(posedge clk) begin
        cycles <= cycles + 1;
        newkv_rd_req_ready <= !req_pending &&
                              (($urandom_range(0, 3) != 0));
        if (newkv_rd_req_valid && newkv_rd_req_ready) begin
            req_pending <= 1'b1;
            req_wave_q <= newkv_rd_req_wave;
            req_addr_q <= newkv_rd_req_addr;
            req_delay <= $urandom_range(0, 3);
        end
        if (newkv_rd_rsp_valid && newkv_rd_rsp_ready) begin
            newkv_rd_rsp_valid <= 1'b0;
            req_pending <= 1'b0;
        end else if (req_pending && !newkv_rd_rsp_valid &&
                     !hold_responses) begin
            if (req_delay != 0) begin
                req_delay <= req_delay - 1;
            end else begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    newkv_rd_rsp_data[lane*16 +: 16] <= scalar_value(
                        {req_wave_q, lane[1:0]}, req_addr_q[10],
                        req_addr_q[9:7], req_addr_q[6:3], req_addr_q[2:0]);
                newkv_rd_rsp_valid <= 1'b1;
            end
        end
    end

    integer writer_count = 0;
    integer expected_count = 0;
    integer done_delay = 0;
    reg writer_expect_run = 1'b0;

    function automatic [127:0] expected_beat(input integer ordinal);
        integer token_i;
        integer rem_i;
        integer kind_i;
        integer head_i;
        integer beat_i;
        integer scalar_i;
        begin
            token_i = ordinal / 256;
            rem_i = ordinal % 256;
            kind_i = rem_i / 128;
            rem_i = rem_i % 128;
            head_i = rem_i / 16;
            beat_i = rem_i % 16;
            expected_beat = 128'd0;
            for (scalar_i = 0; scalar_i < 8; scalar_i = scalar_i + 1)
                expected_beat[scalar_i*16 +: 16] = scalar_value(
                    token_i[2:0], kind_i[0], head_i[2:0],
                    beat_i[3:0], scalar_i[2:0]);
        end
    endfunction

    always @(posedge clk) begin
        wr_cmd_ready <= !wr_busy;
        wr_ready <= wr_busy && (($urandom_range(0, 4) != 0));
        if (abort_run) begin
            wr_busy <= 1'b0;
            wr_done_valid <= 1'b0;
            writer_expect_run <= 1'b0;
        end else begin
            if (wr_cmd_valid && wr_cmd_ready) begin
                if (wr_cmd_addr != 64'h0010_7000 ||
                    wr_cmd_segment_beats != 32'd256 ||
                    wr_cmd_stride_bytes != 32'd4096 ||
                    wr_cmd_repeats != {13'd0, cmd_token_count})
                    $fatal(1, "writer command mismatch");
                wr_busy <= 1'b1;
                writer_count <= 0;
                expected_count <= cmd_token_count * 256;
                writer_expect_run <= 1'b1;
            end
            if (wr_valid && wr_ready) begin
                if (!wr_busy || wr_error)
                    $fatal(1, "unexpected writer input error");
                if (wr_data !== expected_beat(writer_count))
                    $fatal(1, "KV beat mismatch ordinal=%0d", writer_count);
                if (wr_last != (writer_count + 1 == expected_count))
                    $fatal(1, "KV last mismatch ordinal=%0d", writer_count);
                writer_count <= writer_count + 1;
                if (writer_count + 1 == expected_count) begin
                    wr_busy <= 1'b0;
                    done_delay <= 3;
                end
            end
            if (!wr_busy && writer_expect_run && !wr_done_valid) begin
                if (done_delay != 0)
                    done_delay <= done_delay - 1;
                else
                    wr_done_valid <= 1'b1;
            end
            if (wr_done_valid && wr_done_ready) begin
                wr_done_valid <= 1'b0;
                writer_expect_run <= 1'b0;
            end
        end
    end

    task automatic start_command;
        begin
            while (!cmd_ready) @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b1;
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    initial begin
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        // Force a real outstanding arena response across abort, then restart.
        start_command();
        wait (newkv_rd_req_valid && newkv_rd_req_ready);
        hold_responses = 1'b1;
        @(negedge clk);
        abort_run = 1'b1;
        @(posedge clk);
        @(negedge clk);
        abort_run = 1'b0;
        repeat (3) @(posedge clk);
        hold_responses = 1'b0;
        wait (!busy && cmd_ready);

        start_command();
        wait (done_valid);
        if (done_error || done_status != 0 || writer_count != 1280)
            $fatal(1, "KV append failed status=%h count=%0d",
                   done_status, writer_count);

        $display(" kv_append8 PASS cycles=%0d", cycles);
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1, "timeout");
    end
endmodule
