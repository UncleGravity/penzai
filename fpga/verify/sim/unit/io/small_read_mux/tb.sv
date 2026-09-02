`timescale 1ns/1ps
`default_nettype none

module tb;
    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg abort_run = 1'b0;

    reg e_cmd_valid = 0, e_abort = 0, e_ready = 1, e_done_ready = 1;
    reg [63:0] e_addr = 64'h1000;
    reg [31:0] e_beats = 3;
    reg [3:0] e_mask = 4'h1;
    wire e_cmd_ready, e_valid, e_last, e_error, e_busy;
    wire [511:0] e_data;
    wire e_done_valid, e_done_error;
    wire [7:0] e_done_status;

    reg v_req_valid = 0, v_rsp_ready = 1;
    reg [63:0] v_addr = 64'h2000;
    reg [10:0] v_words = 4;
    wire v_req_ready, v_rsp_valid, v_rsp_last, v_rsp_error;
    wire [127:0] v_rsp_data;

    reg s_req_valid = 0, s_rsp_ready = 1;
    reg [63:0] s_addr = 64'h3000;
    reg [6:0] s_words = 2;
    wire s_req_ready, s_rsp_valid, s_rsp_last, s_rsp_error;
    wire [127:0] s_rsp_data;

    wire svc_cmd_valid, svc_abort, svc_ready, svc_done_ready;
    reg svc_cmd_ready = 1, svc_valid = 0, svc_last = 0, svc_error = 0;
    wire [63:0] svc_addr;
    wire [31:0] svc_beats;
    wire [3:0] svc_mask;
    reg [511:0] svc_data = 0;
    reg svc_busy = 0, svc_done_valid = 0, svc_done_error = 0;
    reg [7:0] svc_done_status = 0;

     small_read_mux dut (
        .clk(clk), .rst_n(rst_n), .clear(clear), .abort_run(abort_run),
        .embed_cmd_valid(e_cmd_valid), .embed_cmd_ready(e_cmd_ready),
        .embed_cmd_base_addr(e_addr), .embed_cmd_port_beats(e_beats),
        .embed_cmd_port_mask(e_mask), .embed_abort(e_abort),
        .embed_data(e_data), .embed_valid(e_valid), .embed_ready(e_ready),
        .embed_last(e_last), .embed_error(e_error), .embed_busy(e_busy),
        .embed_done_valid(e_done_valid), .embed_done_ready(e_done_ready),
        .embed_done_error(e_done_error), .embed_done_status(e_done_status),
        .vector_req_valid(v_req_valid), .vector_req_ready(v_req_ready),
        .vector_req_addr(v_addr), .vector_req_words(v_words),
        .vector_rsp_valid(v_rsp_valid), .vector_rsp_ready(v_rsp_ready),
        .vector_rsp_data(v_rsp_data), .vector_rsp_last(v_rsp_last),
        .vector_rsp_error(v_rsp_error), .sink_req_valid(s_req_valid),
        .sink_req_ready(s_req_ready), .sink_req_addr(s_addr),
        .sink_req_words(s_words), .sink_rsp_valid(s_rsp_valid),
        .sink_rsp_ready(s_rsp_ready), .sink_rsp_data(s_rsp_data),
        .sink_rsp_last(s_rsp_last), .sink_rsp_error(s_rsp_error),
        .svc_cmd_valid(svc_cmd_valid), .svc_cmd_ready(svc_cmd_ready),
        .svc_cmd_base_addr(svc_addr), .svc_cmd_port_beats(svc_beats),
        .svc_cmd_port_mask(svc_mask), .svc_abort_run(svc_abort),
        .svc_data(svc_data), .svc_valid(svc_valid), .svc_ready(svc_ready),
        .svc_last(svc_last), .svc_error(svc_error), .svc_busy(svc_busy),
        .svc_done_valid(svc_done_valid), .svc_done_ready(svc_done_ready),
        .svc_done_error(svc_done_error),
        .svc_done_status(svc_done_status)
    );

    task automatic tick;
        begin @(posedge clk); #0.1; end
    endtask

    task automatic finish_service;
        begin
            svc_done_valid = 1;
            tick();
            svc_done_valid = 0;
            svc_busy = 0;
            tick();
        end
    endtask

    initial begin
        repeat (3) tick();
        rst_n = 1;
        tick();

        // Priority is embedding, then vector, then sink.
        e_cmd_valid = 1; v_req_valid = 1; s_req_valid = 1;
        #0.1;
        if (!e_cmd_ready || v_req_ready || s_req_ready ||
            svc_addr != e_addr || svc_beats != 3 || svc_mask != 4'h1)
            $fatal(1, "embedding priority/command mismatch");
        tick();
        e_cmd_valid = 0; v_req_valid = 0; s_req_valid = 0;
        svc_busy = 1;
        svc_data = {4{128'h1234}};
        svc_valid = 1; svc_last = 1;
        #0.1;
        if (!e_valid || v_rsp_valid || s_rsp_valid || !svc_ready ||
            e_data[127:0] != 128'h1234)
            $fatal(1, "embedding response routing mismatch");
        tick();
        svc_valid = 0; svc_last = 0;
        finish_service();

        // Vector coefficient stream uses port zero and low 128 bits.
        v_req_valid = 1; s_req_valid = 1;
        #0.1;
        if (!v_req_ready || s_req_ready || svc_addr != v_addr ||
            svc_beats != 4 || svc_mask != 4'h1)
            $fatal(1, "vector command mismatch");
        tick();
        v_req_valid = 0; s_req_valid = 0;
        svc_busy = 1;
        svc_data[127:0] = 128'h5678;
        svc_valid = 1; svc_last = 1;
        #0.1;
        if (!v_rsp_valid || s_rsp_valid || v_rsp_data != 128'h5678 ||
            !v_rsp_last)
            $fatal(1, "vector response mismatch");
        tick();
        svc_valid = 0; svc_last = 0;
        finish_service();

        // Global clear releases local ownership immediately. The outer reader
        // remains responsible for quarantining any already accepted AXI work.
        s_req_valid = 1;
        #0.1;
        if (!s_req_ready)
            $fatal(1, "clear setup command not ready");
        tick();
        s_req_valid = 0;
        svc_busy = 1;
        clear = 1;
        tick();
        if (dut.owner_q != 0 || dut.aborting_q || !svc_abort)
            $fatal(1, "global clear retained hidden mux ownership");
        clear = 0;
        svc_busy = 0;
        tick();

        // Sink is selected once higher-priority requests are absent.
        s_req_valid = 1;
        #0.1;
        if (!s_req_ready || svc_addr != s_addr || svc_beats != 2)
            $fatal(1, "sink command mismatch");
        tick();
        s_req_valid = 0;
        svc_busy = 1;
        abort_run = 1;
        tick();
        abort_run = 0;
        if (!svc_abort || s_rsp_valid)
            $fatal(1, "abort was not quarantined");
        repeat (2) tick();
        svc_busy = 0;
        tick();
        if (svc_abort)
            $fatal(1, "abort did not drain");

        // Clean restart after the aborted owner.
        s_req_valid = 1;
        #0.1;
        if (!s_req_ready)
            $fatal(1, "restart not ready");
        tick();
        s_req_valid = 0;
        svc_busy = 1;
        svc_data[127:0] = 128'h9abc;
        svc_valid = 1; svc_last = 1;
        #0.1;
        if (!s_rsp_valid || s_rsp_data != 128'h9abc)
            $fatal(1, "restart response mismatch");
        tick();
        svc_valid = 0; svc_last = 0;
        finish_service();

        $display(" small_read_mux PASS");
        $finish;
    end
endmodule

`default_nettype wire
