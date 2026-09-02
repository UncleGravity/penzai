`timescale 1ns/1ps

// Models the externally visible  weight_quad128 contract: one accepted
// command at a time, stable ready/valid data, held completion, and abort that
// quarantines data while a live transaction drains back to idle without done.
module quad_service_model (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    input  wire          abort_run,
    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [63:0]   cmd_base_addr,
    input  wire [31:0]   cmd_port_beats,
    input  wire [3:0]    cmd_port_mask,
    output wire [511:0]  data,
    output wire          valid,
    input  wire          ready,
    output wire          last,
    output wire          error,
    output wire          busy,
    output wire          done_valid,
    input  wire          done_ready,
    output wire          done_error,
    output wire [7:0]    done_status,
    output reg  [31:0]   command_count
);
    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_RUN   = 3'd1;
    localparam [2:0] ST_DRAIN = 3'd2;
    localparam [2:0] ST_DONE  = 3'd3;

    reg [2:0] state_q;
    reg [63:0] base_q;
    reg [31:0] beats_q;
    reg [3:0] mask_q;
    reg [31:0] beat_q;
    reg [2:0] drain_left_q;
    reg [511:0] data_q;
    reg valid_q;
    reg last_q;
    reg error_q;
    reg done_error_q;
    reg [7:0] done_status_q;

    function automatic [511:0] make_data(
        input [63:0] base,
        input [31:0] beat,
        input [3:0] mask
    );
        integer port;
        reg [63:0] word;
        begin
            make_data = 512'd0;
            for (port = 0; port < 4; port = port + 1) begin
                word = base + {28'd0, beat, 4'd0} +
                       (port * 64'h0000_0001_0000_0000);
                if (mask[port])
                    make_data[port*128 +: 128] = {~word, word};
            end
        end
    endfunction

    assign cmd_ready = rst_n && !clear && !abort_run &&
                       (state_q == ST_IDLE);
    assign data = data_q;
    assign valid = valid_q;
    assign last = last_q;
    assign error = error_q;
    assign busy = (state_q == ST_RUN) || (state_q == ST_DRAIN);
    assign done_valid = state_q == ST_DONE;
    assign done_error = done_error_q;
    assign done_status = done_status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            base_q <= 64'd0;
            beats_q <= 32'd0;
            mask_q <= 4'd0;
            beat_q <= 32'd0;
            drain_left_q <= 3'd0;
            data_q <= 512'd0;
            valid_q <= 1'b0;
            last_q <= 1'b0;
            error_q <= 1'b0;
            done_error_q <= 1'b0;
            done_status_q <= 8'd0;
            command_count <= 32'd0;
        end else if (clear) begin
            state_q <= ST_IDLE;
            valid_q <= 1'b0;
            drain_left_q <= 3'd0;
            done_error_q <= 1'b0;
            done_status_q <= 8'd0;
        end else begin
            case (state_q)
                ST_IDLE: begin
                    valid_q <= 1'b0;
                    if (cmd_valid && cmd_ready) begin
                        base_q <= cmd_base_addr;
                        beats_q <= cmd_port_beats;
                        mask_q <= cmd_port_mask;
                        beat_q <= 32'd0;
                        done_error_q <= cmd_base_addr[7:0] == 8'he0;
                        done_status_q <= cmd_base_addr[7:0] == 8'he0 ?
                                         8'h52 : 8'h00;
                        command_count <= command_count + 32'd1;
                        state_q <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (abort_run) begin
                        valid_q <= 1'b0;
                        drain_left_q <= 3'd4;
                        state_q <= ST_DRAIN;
                    end else if (!valid_q) begin
                        data_q <= make_data(base_q, beat_q, mask_q);
                        valid_q <= 1'b1;
                        last_q <= beat_q + 32'd1 == beats_q;
                        error_q <= done_error_q &&
                                   (beat_q + 32'd1 == beats_q);
                    end else if (ready) begin
                        valid_q <= 1'b0;
                        if (last_q) begin
                            state_q <= ST_DONE;
                        end else begin
                            beat_q <= beat_q + 32'd1;
                        end
                    end
                end

                ST_DRAIN: begin
                    valid_q <= 1'b0;
                    if (drain_left_q == 3'd1) begin
                        drain_left_q <= 3'd0;
                        state_q <= ST_IDLE;
                    end else begin
                        drain_left_q <= drain_left_q - 3'd1;
                    end
                end

                ST_DONE: begin
                    valid_q <= 1'b0;
                    if (abort_run || done_ready)
                        state_q <= ST_IDLE;
                end

                default: state_q <= ST_IDLE;
            endcase
        end
    end
endmodule

module tb;
    localparam integer PROJ  = 1;
    localparam integer ROPE  = 2;
    localparam integer SMALL = 3;

    reg clk = 1'b0;
    always #1.666 clk = ~clk;
    reg rst_n = 1'b0;
    reg clear = 1'b0;

    reg proj_cmd_valid = 1'b0;
    wire proj_cmd_ready;
    reg [63:0] proj_cmd_base_addr = 64'd0;
    reg [31:0] proj_cmd_port_beats = 32'd0;
    reg [3:0] proj_cmd_port_mask = 4'd0;
    reg proj_abort = 1'b0;
    wire [511:0] proj_data;
    wire proj_valid;
    reg proj_ready = 1'b0;
    wire proj_last;
    wire proj_error;
    wire proj_busy;
    wire proj_done_valid;
    reg proj_done_ready = 1'b1;
    wire proj_done_error;
    wire [7:0] proj_done_status;

    reg rope_cmd_valid = 1'b0;
    wire rope_cmd_ready;
    reg [63:0] rope_cmd_base_addr = 64'd0;
    reg [31:0] rope_cmd_port_beats = 32'd0;
    reg [3:0] rope_cmd_port_mask = 4'd0;
    reg rope_abort = 1'b0;
    wire [511:0] rope_data;
    wire rope_valid;
    reg rope_ready = 1'b0;
    wire rope_last;
    wire rope_error;
    wire rope_busy;
    wire rope_done_valid;
    reg rope_done_ready = 1'b1;
    wire rope_done_error;
    wire [7:0] rope_done_status;

    reg small_cmd_valid = 1'b0;
    wire small_cmd_ready;
    reg [63:0] small_cmd_base_addr = 64'd0;
    reg [31:0] small_cmd_port_beats = 32'd0;
    reg [3:0] small_cmd_port_mask = 4'd0;
    reg small_abort = 1'b0;
    wire [511:0] small_data;
    wire small_valid;
    reg small_ready = 1'b0;
    wire small_last;
    wire small_error;
    wire small_busy;
    wire small_done_valid;
    reg small_done_ready = 1'b1;
    wire small_done_error;
    wire [7:0] small_done_status;

    wire svc_cmd_valid;
    wire svc_cmd_ready;
    wire [63:0] svc_cmd_base_addr;
    wire [31:0] svc_cmd_port_beats;
    wire [3:0] svc_cmd_port_mask;
    wire svc_abort_run;
    wire [511:0] svc_data;
    wire svc_valid;
    wire svc_ready;
    wire svc_last;
    wire svc_error;
    wire svc_busy;
    wire svc_done_valid;
    wire svc_done_ready;
    wire svc_done_error;
    wire [7:0] svc_done_status;
    wire [31:0] service_command_count;

     quad_read_arbiter dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .proj_cmd_valid(proj_cmd_valid),
        .proj_cmd_ready(proj_cmd_ready),
        .proj_cmd_base_addr(proj_cmd_base_addr),
        .proj_cmd_port_beats(proj_cmd_port_beats),
        .proj_cmd_port_mask(proj_cmd_port_mask),
        .proj_abort(proj_abort), .proj_data(proj_data),
        .proj_valid(proj_valid), .proj_ready(proj_ready),
        .proj_last(proj_last), .proj_error(proj_error),
        .proj_busy(proj_busy), .proj_done_valid(proj_done_valid),
        .proj_done_ready(proj_done_ready),
        .proj_done_error(proj_done_error),
        .proj_done_status(proj_done_status),
        .rope_cmd_valid(rope_cmd_valid),
        .rope_cmd_ready(rope_cmd_ready),
        .rope_cmd_base_addr(rope_cmd_base_addr),
        .rope_cmd_port_beats(rope_cmd_port_beats),
        .rope_cmd_port_mask(rope_cmd_port_mask),
        .rope_abort(rope_abort), .rope_data(rope_data),
        .rope_valid(rope_valid), .rope_ready(rope_ready),
        .rope_last(rope_last), .rope_error(rope_error),
        .rope_busy(rope_busy), .rope_done_valid(rope_done_valid),
        .rope_done_ready(rope_done_ready),
        .rope_done_error(rope_done_error),
        .rope_done_status(rope_done_status),
        .small_cmd_valid(small_cmd_valid),
        .small_cmd_ready(small_cmd_ready),
        .small_cmd_base_addr(small_cmd_base_addr),
        .small_cmd_port_beats(small_cmd_port_beats),
        .small_cmd_port_mask(small_cmd_port_mask),
        .small_abort(small_abort), .small_data(small_data),
        .small_valid(small_valid), .small_ready(small_ready),
        .small_last(small_last), .small_error(small_error),
        .small_busy(small_busy), .small_done_valid(small_done_valid),
        .small_done_ready(small_done_ready),
        .small_done_error(small_done_error),
        .small_done_status(small_done_status),
        .svc_cmd_valid(svc_cmd_valid), .svc_cmd_ready(svc_cmd_ready),
        .svc_cmd_base_addr(svc_cmd_base_addr),
        .svc_cmd_port_beats(svc_cmd_port_beats),
        .svc_cmd_port_mask(svc_cmd_port_mask),
        .svc_abort_run(svc_abort_run), .svc_data(svc_data),
        .svc_valid(svc_valid), .svc_ready(svc_ready),
        .svc_last(svc_last), .svc_error(svc_error),
        .svc_busy(svc_busy), .svc_done_valid(svc_done_valid),
        .svc_done_ready(svc_done_ready),
        .svc_done_error(svc_done_error),
        .svc_done_status(svc_done_status)
    );

    quad_service_model u_service (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .abort_run(svc_abort_run), .cmd_valid(svc_cmd_valid),
        .cmd_ready(svc_cmd_ready), .cmd_base_addr(svc_cmd_base_addr),
        .cmd_port_beats(svc_cmd_port_beats),
        .cmd_port_mask(svc_cmd_port_mask), .data(svc_data),
        .valid(svc_valid), .ready(svc_ready), .last(svc_last),
        .error(svc_error), .busy(svc_busy),
        .done_valid(svc_done_valid), .done_ready(svc_done_ready),
        .done_error(svc_done_error), .done_status(svc_done_status),
        .command_count(service_command_count)
    );

    function automatic [511:0] expected_data(
        input [63:0] base,
        input [31:0] beat,
        input [3:0] mask
    );
        integer port;
        reg [63:0] word;
        begin
            expected_data = 512'd0;
            for (port = 0; port < 4; port = port + 1) begin
                word = base + {28'd0, beat, 4'd0} +
                       (port * 64'h0000_0001_0000_0000);
                if (mask[port])
                    expected_data[port*128 +: 128] = {~word, word};
            end
        end
    endfunction

    integer cycle_count = 0;
    integer expected_owner = 0;
    integer expected_beat = 0;
    integer expected_beats = 0;
    reg [63:0] expected_base = 64'd0;
    reg [3:0] expected_mask = 4'd0;
    reg expected_done_error = 1'b0;
    reg [7:0] expected_done_status = 8'd0;
    reg expected_aborted = 1'b0;

    always @(posedge clk)
        cycle_count <= cycle_count + 1;

    // Deterministic output stalls exercise ownership stability without making
    // the test seed-dependent.
    always @(negedge clk) begin
        if (!rst_n) begin
            proj_ready <= 1'b0;
            rope_ready <= 1'b0;
            small_ready <= 1'b0;
        end else begin
            proj_ready <= cycle_count[1:0] != 2'b00;
            rope_ready <= cycle_count[2:0] != 3'b001;
            small_ready <= cycle_count[1:0] != 2'b10;
        end
    end

    always @(posedge clk) begin
        integer fire_owner;
        reg [511:0] fire_data;
        reg fire_last;
        reg fire_error;
        integer done_owner;

        fire_owner = 0;
        fire_data = 512'd0;
        fire_last = 1'b0;
        fire_error = 1'b0;
        if (proj_valid && proj_ready) begin
            fire_owner = PROJ;
            fire_data = proj_data;
            fire_last = proj_last;
            fire_error = proj_error;
        end
        if (rope_valid && rope_ready) begin
            if (fire_owner != 0)
                $fatal(1, "multiple clients accepted one data beat");
            fire_owner = ROPE;
            fire_data = rope_data;
            fire_last = rope_last;
            fire_error = rope_error;
        end
        if (small_valid && small_ready) begin
            if (fire_owner != 0)
                $fatal(1, "multiple clients accepted one data beat");
            fire_owner = SMALL;
            fire_data = small_data;
            fire_last = small_last;
            fire_error = small_error;
        end

        if (fire_owner != 0) begin
            if (expected_aborted || fire_owner != expected_owner)
                $fatal(1, "data routed to owner %0d, expected %0d",
                       fire_owner, expected_owner);
            if (fire_data !== expected_data(expected_base,
                                             expected_beat[31:0],
                                             expected_mask))
                $fatal(1, "data mismatch owner=%0d beat=%0d",
                       fire_owner, expected_beat);
            if (fire_last !== (expected_beat + 1 == expected_beats))
                $fatal(1, "last mismatch owner=%0d beat=%0d",
                       fire_owner, expected_beat);
            if (fire_error !== (expected_done_error && fire_last))
                $fatal(1, "stream error mismatch owner=%0d beat=%0d",
                       fire_owner, expected_beat);
            expected_beat = expected_beat + 1;
        end

        done_owner = 0;
        if (proj_done_valid)
            done_owner = PROJ;
        if (rope_done_valid) begin
            if (done_owner != 0)
                $fatal(1, "multiple clients saw done");
            done_owner = ROPE;
        end
        if (small_done_valid) begin
            if (done_owner != 0)
                $fatal(1, "multiple clients saw done");
            done_owner = SMALL;
        end
        if (done_owner != 0) begin
            if (expected_aborted || done_owner != expected_owner)
                $fatal(1, "done routed to owner %0d, expected %0d",
                       done_owner, expected_owner);
            if (expected_beat != expected_beats)
                $fatal(1, "done before all data: %0d/%0d",
                       expected_beat, expected_beats);
            case (done_owner)
                PROJ: if (proj_done_error !== expected_done_error ||
                          proj_done_status !== expected_done_status)
                    $fatal(1, "projection done payload mismatch");
                ROPE: if (rope_done_error !== expected_done_error ||
                          rope_done_status !== expected_done_status)
                    $fatal(1, "RoPE done payload mismatch");
                SMALL: if (small_done_error !== expected_done_error ||
                           small_done_status !== expected_done_status)
                    $fatal(1, "small done payload mismatch");
            endcase
        end

        if (expected_owner != PROJ &&
            (proj_valid || proj_busy || proj_done_valid))
            $fatal(1, "projection response leaked without ownership");
        if (expected_owner != ROPE &&
            (rope_valid || rope_busy || rope_done_valid))
            $fatal(1, "RoPE response leaked without ownership");
        if (expected_owner != SMALL &&
            (small_valid || small_busy || small_done_valid))
            $fatal(1, "small response leaked without ownership");
    end

    task automatic set_expectation(
        input integer owner,
        input [63:0] base,
        input integer beats,
        input [3:0] mask
    );
        begin
            expected_owner = owner;
            expected_beat = 0;
            expected_beats = beats;
            expected_base = base;
            expected_mask = mask;
            expected_done_error = base[7:0] == 8'he0;
            expected_done_status = base[7:0] == 8'he0 ? 8'h52 : 8'h00;
            expected_aborted = 1'b0;
        end
    endtask

    task automatic drive_client(
        input integer owner,
        input reg value
    );
        begin
            case (owner)
                PROJ: proj_cmd_valid = value;
                ROPE: rope_cmd_valid = value;
                SMALL: small_cmd_valid = value;
                default: $fatal(1, "bad owner");
            endcase
        end
    endtask

    task automatic set_command(
        input integer owner,
        input [63:0] base,
        input [31:0] beats,
        input [3:0] mask
    );
        begin
            case (owner)
                PROJ: begin
                    proj_cmd_base_addr = base;
                    proj_cmd_port_beats = beats;
                    proj_cmd_port_mask = mask;
                end
                ROPE: begin
                    rope_cmd_base_addr = base;
                    rope_cmd_port_beats = beats;
                    rope_cmd_port_mask = mask;
                end
                SMALL: begin
                    small_cmd_base_addr = base;
                    small_cmd_port_beats = beats;
                    small_cmd_port_mask = mask;
                end
                default: $fatal(1, "bad owner");
            endcase
        end
    endtask

    task automatic launch_one(
        input integer owner,
        input [63:0] base,
        input [31:0] beats,
        input [3:0] mask
    );
        reg accepted;
        begin
            set_expectation(owner, base, beats, mask);
            @(negedge clk);
            set_command(owner, base, beats, mask);
            drive_client(owner, 1'b1);
            accepted = 1'b0;
            while (!accepted) begin
                @(posedge clk);
                case (owner)
                    PROJ: accepted = proj_cmd_ready;
                    ROPE: accepted = rope_cmd_ready;
                    SMALL: accepted = small_cmd_ready;
                endcase
            end
            if (svc_cmd_base_addr !== base ||
                svc_cmd_port_beats !== beats ||
                svc_cmd_port_mask !== mask)
                $fatal(1, "service command payload mismatch owner=%0d", owner);
            @(negedge clk);
            drive_client(owner, 1'b0);
        end
    endtask

    task automatic wait_done(input integer owner);
        reg seen;
        begin
            seen = 1'b0;
            while (!seen) begin
                @(posedge clk);
                case (owner)
                    PROJ: seen = proj_done_valid;
                    ROPE: seen = rope_done_valid;
                    SMALL: seen = small_done_valid;
                endcase
            end
            @(posedge clk);
            expected_owner = 0;
        end
    endtask

    task automatic launch_simultaneous(
        input reg include_proj,
        input reg include_rope,
        input reg include_small,
        input integer winner
    );
        begin
            set_expectation(winner,
                winner == PROJ ? 64'h0000_0000_0000_4000 :
                winner == ROPE ? 64'h0000_0000_0000_5000 :
                                 64'h0000_0000_0000_6000,
                3, winner == PROJ ? 4'hf :
                   winner == ROPE ? 4'h7 : 4'h3);
            @(negedge clk);
            set_command(PROJ, 64'h0000_0000_0000_4000, 32'd3, 4'hf);
            set_command(ROPE, 64'h0000_0000_0000_5000, 32'd3, 4'h7);
            set_command(SMALL, 64'h0000_0000_0000_6000, 32'd3, 4'h3);
            proj_cmd_valid = include_proj;
            rope_cmd_valid = include_rope;
            small_cmd_valid = include_small;
            @(posedge clk);
            if ((proj_cmd_ready !== (winner == PROJ)) ||
                (rope_cmd_ready !== (winner == ROPE)) ||
                (small_cmd_ready !== (winner == SMALL)))
                $fatal(1, "simultaneous priority mismatch winner=%0d", winner);
            @(negedge clk);
            proj_cmd_valid = 1'b0;
            rope_cmd_valid = 1'b0;
            small_cmd_valid = 1'b0;
        end
    endtask

    initial begin
        integer commands_before;
        integer beats_before_abort;
        integer drain_cycles;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);
        if (!proj_cmd_ready || !rope_cmd_ready || !small_cmd_ready)
            $fatal(1, "all idle clients must observe ready");

        // Each owner receives only its own data and completion.  The RoPE
        // command also proves error/status routing.
        launch_one(PROJ, 64'h0000_0000_0000_1000, 5, 4'hf);
        wait_done(PROJ);
        launch_one(ROPE, 64'h0000_0000_0000_20e0, 4, 4'h7);
        wait_done(ROPE);
        launch_one(SMALL, 64'h0000_0000_0000_3000, 2, 4'h1);
        wait_done(SMALL);

        // Fixed priority is deterministic for requests arriving together.
        launch_simultaneous(1'b1, 1'b1, 1'b1, PROJ);
        wait_done(PROJ);
        launch_simultaneous(1'b0, 1'b1, 1'b1, ROPE);
        wait_done(ROPE);

        // Completion backpressure keeps ownership locked; a waiting client
        // cannot be accepted until the owner's done handshake.
        proj_done_ready = 1'b0;
        launch_one(PROJ, 64'h0000_0000_0000_7000, 3, 4'hf);
        while (!proj_done_valid) @(posedge clk);
        @(negedge clk);
        set_command(ROPE, 64'h0000_0000_0000_7100, 2, 4'h3);
        rope_cmd_valid = 1'b1;
        repeat (3) begin
            @(posedge clk);
            if (rope_cmd_ready)
                $fatal(1, "client accepted before owner consumed done");
        end
        @(negedge clk);
        rope_cmd_valid = 1'b0;
        proj_done_ready = 1'b1;
        @(posedge clk);
        @(posedge clk);
        expected_owner = 0;

        // Abort after visible traffic.  Data is hidden immediately, the
        // contender remains blocked throughout drain, and a held abort is
        // allowed to observe idle ready before a clean restart.
        launch_one(SMALL, 64'h0000_0000_0000_8000, 9, 4'hf);
        while (expected_beat < 2) @(posedge clk);
        beats_before_abort = expected_beat;
        commands_before = service_command_count;
        @(negedge clk);
        expected_aborted = 1'b1;
        small_abort = 1'b1;
        set_command(PROJ, 64'h0000_0000_0000_9000, 4, 4'hf);
        proj_cmd_valid = 1'b1;
        drain_cycles = 0;
        while (svc_busy) begin
            @(posedge clk);
            drain_cycles = drain_cycles + 1;
            if (proj_cmd_ready)
                $fatal(1, "contender accepted during abort drain");
            if (proj_valid || rope_valid || small_valid ||
                proj_done_valid || rope_done_valid || small_done_valid)
                $fatal(1, "aborted response was not quarantined");
        end
        if (drain_cycles < 4)
            $fatal(1, "abort did not remain locked through drain");
        @(negedge clk);
        proj_cmd_valid = 1'b0;
        while (!small_cmd_ready) begin
            @(posedge clk);
            if (service_command_count != commands_before)
                $fatal(1, "service accepted a command during abort");
        end
        if (expected_beat != beats_before_abort)
            $fatal(1, "aborted data escaped quarantine");
        if (svc_abort_run)
            $fatal(1, "service abort remained asserted after owner release");
        @(negedge clk);
        small_abort = 1'b0;
        expected_owner = 0;

        launch_one(PROJ, 64'h0000_0000_0000_a000, 4, 4'hd);
        wait_done(PROJ);
        if (service_command_count != commands_before + 1)
            $fatal(1, "restart command count mismatch");

        $display("PASS  quad_read_arbiter cycles=%0d commands=%0d",
                 cycle_count, service_command_count);
        $finish;
    end

    initial begin
        repeat (3000) @(posedge clk);
        $fatal(1, "timeout");
    end
endmodule
