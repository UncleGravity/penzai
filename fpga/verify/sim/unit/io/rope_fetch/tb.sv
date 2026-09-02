`timescale 1ns/1ps

module rope_mem_port #(
    parameter integer PORT = 0,
    parameter [39:0] BASE = 40'h0010000000
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire [39:0]  araddr,
    input  wire [7:0]   arlen,
    input  wire         arvalid,
    output reg          arready,
    output reg [127:0]  rdata,
    output reg [1:0]    rresp,
    output reg          rlast,
    output reg          rvalid,
    input  wire         rready,
    output integer      requests
);
    reg active;
    reg [39:0] addr_q;
    integer left_q;

    function automatic [31:0] cos_value(
        input integer position,
        input integer pair
    );
        cos_value = 32'h3f00_0000 + (position << 7) + pair;
    endfunction
    function automatic [31:0] sin_value(
        input integer position,
        input integer pair
    );
        sin_value = 32'hbf00_0000 + (position << 7) + pair;
    endfunction
    function automatic [127:0] memory_data(input [39:0] addr);
        integer byte_offset;
        integer position;
        integer beat;
        integer pair0;
        begin
            byte_offset = addr - BASE;
            position = byte_offset / 512;
            beat = (byte_offset % 512) / 16;
            pair0 = beat * 2;
            memory_data = {
                sin_value(position, pair0 + 1),
                cos_value(position, pair0 + 1),
                sin_value(position, pair0),
                cos_value(position, pair0)
            };
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            arready <= 1'b0;
            rdata <= 128'd0;
            rresp <= 2'b00;
            rlast <= 1'b0;
            rvalid <= 1'b0;
            active <= 1'b0;
            addr_q <= 40'd0;
            left_q <= 0;
            requests <= 0;
        end else begin
            arready <= !active && !rvalid;
            if (arvalid && arready) begin
                if (araddr < BASE || araddr[8:0] != 9'd0 || arlen != 8'd31)
                    $fatal(1, "port %0d bad RoPE request addr=%h len=%0d",
                           PORT, araddr, arlen);
                active <= 1'b1;
                addr_q <= araddr;
                left_q <= 32;
                requests <= requests + 1;
            end
            if (active && !rvalid) begin
                rdata <= memory_data(addr_q);
                rresp <= 2'b00;
                rlast <= left_q == 1;
                rvalid <= 1'b1;
            end else if (rvalid && rready) begin
                if (rlast) begin
                    rvalid <= 1'b0;
                    rlast <= 1'b0;
                    active <= 1'b0;
                end else begin
                    addr_q <= addr_q + 40'd16;
                    left_q <= left_q - 1;
                    rdata <= memory_data(addr_q + 40'd16);
                    rlast <= left_q == 2;
                end
            end
        end
    end
endmodule

module tb;
    localparam [39:0] BASE = 40'h0010000000;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;
    reg cmd_valid = 1'b0;
    wire cmd_ready;
    reg [63:0] cmd_table_addr = {24'd0, BASE};
    reg [16:0] cmd_position_base = 17'd0;
    reg [3:0] cmd_token_count = 4'd0;
    wire [255:0] coeff_data;
    wire coeff_valid;
    reg coeff_ready = 1'b0;
    wire coeff_last;
    wire coeff_error;
    wire busy;
    wire done_valid;
    reg done_ready = 1'b1;
    wire done_error;
    wire [7:0] done_status;
    wire [159:0] araddr;
    wire [31:0] arlen;
    wire [11:0] arsize;
    wire [7:0] arburst;
    wire [3:0] arvalid;
    wire [3:0] arready;
    wire [511:0] rdata;
    wire [7:0] rresp;
    wire [3:0] rlast;
    wire [3:0] rvalid;
    wire [3:0] rready;
    wire [127:0] request_counts;
    wire read_cmd_valid;
    wire read_cmd_ready;
    wire [63:0] read_cmd_base_addr;
    wire [31:0] read_cmd_port_beats;
    wire [3:0] read_cmd_port_mask;
    wire read_abort;
    wire [511:0] read_data;
    wire read_valid;
    wire read_ready;
    wire read_last;
    wire read_error;
    wire read_busy;
    wire read_done_valid;
    wire read_done_ready;
    wire read_done_error;
    wire [7:0] read_done_status;

     rope_fetch4 dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_table_addr(cmd_table_addr),
        .cmd_position_base(cmd_position_base),
        .cmd_token_count(cmd_token_count),
        .coeff_data(coeff_data), .coeff_valid(coeff_valid),
        .coeff_ready(coeff_ready), .coeff_last(coeff_last),
        .coeff_error(coeff_error), .busy(busy),
        .done_valid(done_valid), .done_ready(done_ready),
        .done_error(done_error), .done_status(done_status),
        .read_cmd_valid(read_cmd_valid), .read_cmd_ready(read_cmd_ready),
        .read_cmd_base_addr(read_cmd_base_addr),
        .read_cmd_port_beats(read_cmd_port_beats),
        .read_cmd_port_mask(read_cmd_port_mask), .read_abort(read_abort),
        .read_data(read_data), .read_valid(read_valid),
        .read_ready(read_ready), .read_last(read_last),
        .read_error(read_error), .read_busy(read_busy),
        .read_done_valid(read_done_valid),
        .read_done_ready(read_done_ready),
        .read_done_error(read_done_error),
        .read_done_status(read_done_status)
    );

     weight_quad128 u_quad (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .abort_run(read_abort), .cmd_valid(read_cmd_valid),
        .cmd_ready(read_cmd_ready), .cmd_base_addr(read_cmd_base_addr),
        .cmd_port_beats(read_cmd_port_beats),
        .cmd_port_mask(read_cmd_port_mask), .weight_data(read_data),
        .weight_valid(read_valid), .weight_ready(read_ready),
        .weight_last(read_last), .weight_error(read_error),
        .busy(read_busy), .done_valid(read_done_valid),
        .done_ready(read_done_ready), .done_error(read_done_error),
        .done_status(read_done_status),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen),
        .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(arready),
        .m_axi_rdata(rdata), .m_axi_rresp(rresp),
        .m_axi_rlast(rlast), .m_axi_rvalid(rvalid),
        .m_axi_rready(rready)
    );

    genvar p;
    generate
        for (p = 0; p < 4; p = p + 1) begin : g_mem
            rope_mem_port #(.PORT(p), .BASE(BASE)) u_mem (
                .clk(clk), .rst_n(rst_n),
                .araddr(araddr[p*40 +: 40]),
                .arlen(arlen[p*8 +: 8]),
                .arvalid(arvalid[p]), .arready(arready[p]),
                .rdata(rdata[p*128 +: 128]),
                .rresp(rresp[p*2 +: 2]), .rlast(rlast[p]),
                .rvalid(rvalid[p]), .rready(rready[p]),
                .requests(request_counts[p*32 +: 32])
            );
        end
    endgenerate

    function automatic [31:0] cos_value(
        input integer position,
        input integer pair
    );
        cos_value = 32'h3f00_0000 + (position << 7) + pair;
    endfunction
    function automatic [31:0] sin_value(
        input integer position,
        input integer pair
    );
        sin_value = 32'hbf00_0000 + (position << 7) + pair;
    endfunction

    integer output_count = 0;
    integer expected_outputs = 0;
    integer active_position_base = 0;
    integer active_tokens = 0;
    reg random_ready = 1'b1;

    always @(negedge clk) begin
        if (!rst_n)
            coeff_ready <= 1'b0;
        else if (random_ready)
            coeff_ready <= ($urandom_range(0, 3) != 0);
    end

    always @(posedge clk) begin
        integer ordinal;
        integer wave;
        integer pair;
        integer lane;
        integer token;
        reg [63:0] want;
        if (rst_n && coeff_valid && coeff_ready) begin
            ordinal = output_count;
            wave = ordinal / 64;
            pair = ordinal % 64;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                token = wave * 4 + lane;
                want = token < active_tokens ?
                    {sin_value(active_position_base + token, pair),
                     cos_value(active_position_base + token, pair)} : 64'd0;
                if (coeff_data[lane*64 +: 64] !== want)
                    $fatal(1, "coeff mismatch ord=%0d lane=%0d got=%h want=%h",
                           ordinal, lane, coeff_data[lane*64 +: 64], want);
            end
            if (coeff_error)
                $fatal(1, "unexpected coefficient error");
            if (coeff_last != (output_count + 1 == expected_outputs))
                $fatal(1, "last mismatch output=%0d", output_count);
            output_count <= output_count + 1;
        end
    end

    task automatic launch(input integer position, input integer tokens);
        begin
            while (!cmd_ready) @(posedge clk);
            @(negedge clk);
            cmd_position_base = position[16:0];
            cmd_token_count = tokens[3:0];
            cmd_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic run_success(input integer position, input integer tokens);
        begin
            output_count = 0;
            expected_outputs = tokens > 4 ? 128 : 64;
            active_position_base = position;
            active_tokens = tokens;
            launch(position, tokens);
            while (!done_valid) @(posedge clk);
            if (done_error || done_status != 8'd0)
                $fatal(1, "RoPE command failed status=%h", done_status);
            if (output_count != expected_outputs)
                $fatal(1, "output count %0d/%0d", output_count,
                       expected_outputs);
            @(posedge clk);
        end
    endtask

    initial begin
        integer requests_before;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        run_success(7, 1);
        if (request_counts[31:0] != 1 || request_counts[127:32] != 0)
            $fatal(1, "single-token accessed inactive RoPE ports");

        requests_before = request_counts[31:0];
        run_success(20, 8);
        if (request_counts[31:0] - requests_before != 2 ||
            request_counts[63:32] != 2 || request_counts[95:64] != 2 ||
            request_counts[127:96] != 2)
            $fatal(1, "tile-8 did not issue two four-lane waves");

        run_success(65533, 3);

        // Overflow is rejected without memory traffic.
        requests_before = request_counts[31:0];
        launch(65534, 3);
        while (!done_valid) @(posedge clk);
        if (!done_error || done_status != 8'h01)
            $fatal(1, "overflow command was not rejected");
        if (request_counts[31:0] != requests_before)
            $fatal(1, "overflow command touched memory");
        @(posedge clk);

        $display(" rope_fetch4: single-token/tile-8/64K PASS outputs=%0d",
                 output_count);
        $finish;
    end

    initial begin
        #5000000;
        $fatal(1, "timeout");
    end
endmodule
