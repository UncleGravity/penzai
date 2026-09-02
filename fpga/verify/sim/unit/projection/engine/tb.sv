`timescale 1ns/1ps
`default_nettype none
/* verilator lint_off DECLFILENAME */
/* verilator lint_off TIMESCALEMOD */

module projection_engine_tb;
    localparam [1:0] WEIGHT_Q1 = 2'd1;
    localparam [1:0] WEIGHT_Q2 = 2'd2;
    localparam integer MAX_TEST_M = 32;
    localparam integer ACT_MODEL_DEPTH = 16;
    localparam integer MAX_CADENCE_GROUPS = 32;
    localparam integer DOT_PACKET_W =
        (1 + 1 + 1 + 4 + 16*16 + 4*16) + (16*4*14);

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg clear = 1'b0;
    reg start = 1'b0;
    reg [15:0] model_spec_k = 16'd0;
    reg [17:0] model_spec_m = 18'd0;
    reg [15:0] model_spec_rowblocks = 16'd0;
    reg [1:0] weight_fmt = WEIGHT_Q1;
    reg signed [7:0] emin = -8'sd20;
    reg [7:0] token_mask = 8'd0;
    wire busy;
    wire done;
    wire error;

    reg [511:0] w_data = 512'd0;
    reg w_valid = 1'b0;
    wire w_ready;

    wire act_req_valid;
    wire act_req_ready;
    wire [8:0] act_req_addr;
    wire act_req_wave;
    wire act_rsp_valid;
    wire act_rsp_ready;
    wire [1087:0] act_rsp_data;
    reg auto_act_rsp_valid_q = 1'b0;
    reg [1087:0] auto_act_rsp_data_q = 1088'd0;
    reg manual_act_rsp = 1'b0;
    reg manual_act_rsp_valid = 1'b0;
    reg [1087:0] manual_act_rsp_data = 1088'd0;
    reg randomize_act_service = 1'b0;
    reg hold_act_responses = 1'b0;
    reg tagged_activations = 1'b0;
    reg [15:0] act_lfsr = 16'h1ace;
    reg [8:0] act_model_addr [0:ACT_MODEL_DEPTH-1];
    reg act_model_wave [0:ACT_MODEL_DEPTH-1];
    reg [3:0] act_model_wr_ptr = 4'd0;
    reg [3:0] act_model_rd_ptr = 4'd0;
    reg [4:0] act_model_count = 5'd0;

    wire signed [103:0] out_acc;
    wire signed [7:0] out_emin;
    wire [2:0] out_token;
    wire [17:0] out_row;
    wire out_last;
    wire out_valid;
    reg out_ready = 1'b1;
    wire [31:0] weight_beat_count;
    wire [31:0] wave_issue_count;

    integer cycle = 0;
    integer expected_fmt = 0;
    integer expected_m = 0;
    integer expected_k_blocks = 0;
    integer expected_rowblocks = 0;
    reg [7:0] expected_token_mask = 8'd0;
    integer observed_outputs = 0;
    reg [MAX_TEST_M*8-1:0] seen_outputs = {MAX_TEST_M*8{1'b0}};
    reg enable_output_stalls = 1'b0;
    reg stalled_last_cycle = 1'b0;
    reg signed [103:0] stalled_acc;
    reg signed [7:0] stalled_emin;
    reg [2:0] stalled_token;
    reg [17:0] stalled_row;
    reg stalled_last;
    integer fifo_push_pop_events = 0;
    integer fifo_push_pop_before;
    integer response_backpressure_cycles = 0;
    integer request_backpressure_cycles = 0;
    integer queued_response_delay_cycles = 0;
    integer request_backpressure_before;
    integer response_delay_before;
    reg prior_act_rsp_fire = 1'b0;
    integer observed_act_requests = 0;
    integer observed_act_responses = 0;
    integer expected_request_addr;
    integer expected_request_wave;
    integer active_wave_count;
    integer request_selector_index;
    integer request_wave_index;
    integer request_within_rowblock;
    reg held_act_rsp_last_cycle = 1'b0;
    reg [1087:0] held_act_rsp_data;
    integer single_token_fifo_full_cycles = 0;
    integer single_token_fifo_push_pop_events = 0;
    integer single_token_fifo_full_cycles_before;
    reg measure_single_token_cadence = 1'b0;
    integer cadence_weight_index = 0;
    integer cadence_group_count = 0;
    integer cadence_response_count = 0;
    integer cadence_last_scale_cycle = -1;
    integer cadence_min_scale_delta = 32'h7fffffff;
    integer cadence_max_scale_delta = 0;
    integer cadence_max_completion_span = 0;
    integer cadence_delta;
    integer cadence_span;
    integer cadence_response_group;
    integer cadence_scale_cycles [0:MAX_CADENCE_GROUPS-1];
    reg weight_driver_done = 1'b0;
    reg [511:0] pre_clear_weight_sign;
    reg [511:0] pre_clear_weight_nonzero;
    reg [DOT_PACKET_W-1:0] stale_fifo0;
    reg [DOT_PACKET_W-1:0] stale_fifo1;
    reg [DOT_PACKET_W-1:0] stale_fifo2;
    reg [DOT_PACKET_W-1:0] stale_fifo3;

    always #1 clk = ~clk;

     projection_engine dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear(clear),
        .start(start),
        .model_spec_k(model_spec_k),
        .model_spec_m(model_spec_m),
        .model_spec_rowblocks(model_spec_rowblocks),
        .weight_fmt(weight_fmt),
        .emin(emin),
        .token_mask(token_mask),
        .busy(busy),
        .done(done),
        .error(error),
        .w_data(w_data),
        .w_valid(w_valid),
        .w_ready(w_ready),
        .act_req_valid(act_req_valid),
        .act_req_ready(act_req_ready),
        .act_req_addr(act_req_addr),
        .act_req_wave(act_req_wave),
        .act_rsp_valid(act_rsp_valid),
        .act_rsp_ready(act_rsp_ready),
        .act_rsp_data(act_rsp_data),
        .out_acc(out_acc),
        .out_emin(out_emin),
        .out_token(out_token),
        .out_row(out_row),
        .out_last(out_last),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .weight_beat_count(weight_beat_count),
        .wave_issue_count(wave_issue_count)
    );

    function automatic [1087:0] activation_wave;
        input [8:0] addr;
        input wave;
        integer lane;
        integer element;
        integer logical_token;
        integer activation_value;
        reg [1087:0] value;
        begin
            value = 1088'd0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                logical_token = (wave ? 4 : 0) + lane;
                activation_value = logical_token + 1;
                if (tagged_activations)
                    activation_value = activation_value + addr;
                for (element = 0; element < 32; element = element + 1)
                    value[lane*272 + element*8 +: 8] = activation_value;
                value[lane*272 + 256 +: 16] = 16'h3c00;
            end
            activation_wave = value;
        end
    endfunction

    wire auto_act_req_fire = act_req_valid && act_req_ready &&
                             !manual_act_rsp;
    wire auto_act_rsp_slot_ready = !auto_act_rsp_valid_q || act_rsp_ready;
    wire auto_act_release = !hold_act_responses &&
                            (!randomize_act_service ||
                             (act_lfsr[1] && act_lfsr[4]));
    wire auto_act_queue_pop = auto_act_rsp_slot_ready && auto_act_release &&
                              (act_model_count != 0);
    wire auto_act_req_bypass = auto_act_rsp_slot_ready && auto_act_release &&
                               (act_model_count == 0) && auto_act_req_fire;
    wire auto_act_queue_push = auto_act_req_fire && !auto_act_req_bypass;

    assign act_req_ready = manual_act_rsp ? 1'b1 :
                           (act_model_count < ACT_MODEL_DEPTH) &&
                           (!randomize_act_service ||
                            act_lfsr[0] || act_lfsr[3]);
    assign act_rsp_valid = manual_act_rsp ? manual_act_rsp_valid :
                                           auto_act_rsp_valid_q;
    assign act_rsp_data = manual_act_rsp ? manual_act_rsp_data :
                                          auto_act_rsp_data_q;

    // In-order model of the resident activation service.  The empty-queue
    // bypass retains the production service's one-cycle best-case latency;
    // queued requests permit deterministic response delay and backpressure.
    always @(posedge clk) begin
        if (!rst_n || clear) begin
            auto_act_rsp_valid_q <= 1'b0;
            act_model_wr_ptr <= 4'd0;
            act_model_rd_ptr <= 4'd0;
            act_model_count <= 5'd0;
            act_lfsr <= 16'h1ace;
        end else if (manual_act_rsp) begin
            auto_act_rsp_valid_q <= 1'b0;
            act_lfsr <= {act_lfsr[14:0], act_lfsr[15] ^ act_lfsr[13] ^
                         act_lfsr[12] ^ act_lfsr[10]};
        end else begin
            act_lfsr <= {act_lfsr[14:0], act_lfsr[15] ^ act_lfsr[13] ^
                         act_lfsr[12] ^ act_lfsr[10]};

            if (auto_act_queue_push) begin
                act_model_addr[act_model_wr_ptr] <= act_req_addr;
                act_model_wave[act_model_wr_ptr] <= act_req_wave;
                act_model_wr_ptr <= act_model_wr_ptr + 1'b1;
            end
            if (auto_act_queue_pop)
                act_model_rd_ptr <= act_model_rd_ptr + 1'b1;
            case ({auto_act_queue_push, auto_act_queue_pop})
                2'b10: act_model_count <= act_model_count + 1'b1;
                2'b01: act_model_count <= act_model_count - 1'b1;
                default: ;
            endcase

            if (auto_act_rsp_slot_ready) begin
                if (auto_act_queue_pop) begin
                    auto_act_rsp_valid_q <= 1'b1;
                    auto_act_rsp_data_q <= activation_wave(
                        act_model_addr[act_model_rd_ptr],
                        act_model_wave[act_model_rd_ptr]);
                end else if (auto_act_req_bypass) begin
                    auto_act_rsp_valid_q <= 1'b1;
                    auto_act_rsp_data_q <= activation_wave(
                        act_req_addr, act_req_wave);
                end else begin
                    auto_act_rsp_valid_q <= 1'b0;
                end
            end
        end
    end

    function automatic [511:0] unit_scales;
        integer row;
        reg [511:0] value;
        begin
            value = 512'd0;
            for (row = 0; row < 16; row = row + 1) begin
                value[row*32 +: 16] = 16'h3c00;
                value[row*32 + 16 +: 16] = 16'h3c00;
            end
            unit_scales = value;
        end
    endfunction

    function automatic integer q1_target;
        input integer global_row;
        begin
            q1_target = -30 + 2*(global_row & 15);
        end
    endfunction

    function automatic [511:0] q1_bits;
        input integer rowblock;
        input integer subblock;
        integer row;
        integer lane;
        integer positives;
        integer element;
        reg [511:0] value;
        begin
            value = 512'd0;
            for (row = 0; row < 16; row = row + 1) begin
                positives = (q1_target(rowblock*16 + row) + 128) / 2;
                for (lane = 0; lane < 32; lane = lane + 1) begin
                    element = subblock*32 + lane;
                    value[row*32 + lane] = (element < positives);
                end
            end
            q1_bits = value;
        end
    endfunction

    function automatic integer q2_target;
        input integer global_row;
        begin
            q2_target = (global_row % 17) - 8;
        end
    endfunction

    function automatic [1:0] q2_code;
        input integer global_row;
        input integer element;
        integer target;
        begin
            target = q2_target(global_row);
            if ((target < 0) && (element < -target))
                q2_code = 2'd0;
            else if ((target > 0) && (element < target))
                q2_code = 2'd2;
            else
                q2_code = 2'd1;
        end
    endfunction

    function automatic [511:0] q2_codes;
        input integer rowblock;
        input integer subblock;
        input integer half;
        integer row;
        integer lane;
        integer element;
        reg [511:0] value;
        begin
            value = 512'd0;
            for (row = 0; row < 16; row = row + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    element = subblock*32 + half*16 + lane;
                    value[row*32 + lane*2 +: 2] =
                        q2_code(rowblock*16 + row, element);
                end
            end
            q2_codes = value;
        end
    endfunction

    task automatic send_weight;
        input [511:0] beat;
        begin
            @(negedge clk);
            w_data = beat;
            w_valid = 1'b1;
            do @(posedge clk); while (!w_ready);
            @(negedge clk);
            w_valid = 1'b0;
            w_data = 512'd0;
        end
    endtask

    task automatic launch_run;
        input [1:0] fmt;
        input [15:0] k;
        input [17:0] m;
        input [15:0] rowblocks;
        input [7:0] mask;
        begin
            expected_fmt = fmt;
            expected_m = m;
            expected_k_blocks = k >> 7;
            expected_rowblocks = rowblocks;
            expected_token_mask = mask;
            observed_outputs = 0;
            observed_act_requests = 0;
            observed_act_responses = 0;
            seen_outputs = {MAX_TEST_M*8{1'b0}};
            cadence_weight_index = 0;
            cadence_group_count = 0;
            cadence_response_count = 0;
            cadence_last_scale_cycle = -1;
            cadence_min_scale_delta = 32'h7fffffff;
            cadence_max_scale_delta = 0;
            cadence_max_completion_span = 0;
            @(negedge clk);
            model_spec_k = k;
            model_spec_m = m;
            model_spec_rowblocks = rowblocks;
            weight_fmt = fmt;
            token_mask = mask;
            emin = -8'sd20;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            if (!busy) $fatal(1, "run was not accepted");
        end
    endtask

    task automatic send_q1_projection;
        input integer rowblocks;
        integer rb;
        integer sub;
        begin
            for (rb = 0; rb < rowblocks; rb = rb + 1) begin
                send_weight(unit_scales());
                for (sub = 0; sub < 4; sub = sub + 1)
                    send_weight(q1_bits(rb, sub));
            end
        end
    endtask

    // Present the first response in the same cycle as its request.  The source
    // must be backpressured until the engine has registered the request tag,
    // then the complete projection must retain the normal exact result.
    task automatic send_q1_projection_early_first_response;
        integer sub;
        integer backpressure_before;
        reg [1087:0] held_response;
        begin
            send_weight(unit_scales());
            send_weight(q1_bits(0, 0));
            if (!act_req_valid)
                $fatal(1, "early-response test has no activation request");

            held_response = activation_wave(act_req_addr, act_req_wave);
            manual_act_rsp = 1'b1;
            manual_act_rsp_valid = 1'b1;
            manual_act_rsp_data = held_response;
            backpressure_before = response_backpressure_cycles;
            #0.1;
            if (act_rsp_ready)
                $fatal(1, "untagged activation response was accepted");

            // The request is accepted here; the held response becomes ready on
            // the following cycle without changing payload.
            @(posedge clk);
            @(negedge clk);
            if (!act_rsp_valid || !act_rsp_ready ||
                (act_rsp_data !== held_response))
                $fatal(1, "held activation response changed under backpressure");
            @(posedge clk);
            @(negedge clk);
            manual_act_rsp_valid = 1'b0;
            manual_act_rsp = 1'b0;
            if (response_backpressure_cycles == backpressure_before)
                $fatal(1, "early activation response was never backpressured");

            for (sub = 1; sub < 4; sub = sub + 1)
                send_weight(q1_bits(0, sub));
        end
    endtask

    task automatic send_q2_projection;
        input integer rowblocks;
        integer rb;
        integer sub;
        begin
            for (rb = 0; rb < rowblocks; rb = rb + 1) begin
                send_weight(unit_scales());
                for (sub = 0; sub < 4; sub = sub + 1) begin
                    send_weight(q2_codes(rb, sub, 0));
                    send_weight(q2_codes(rb, sub, 1));
                end
            end
        end
    endtask

    task automatic stream_weight_beat;
        input [511:0] beat;
        begin
            w_data = beat;
            do @(posedge clk); while (!w_ready);
            @(negedge clk);
        end
    endtask

    task automatic send_q1_projection_contiguous;
        input integer rowblocks;
        input integer kblocks;
        integer rb;
        integer kb;
        integer sub;
        begin
            @(negedge clk);
            w_valid = 1'b1;
            for (rb = 0; rb < rowblocks; rb = rb + 1) begin
                for (kb = 0; kb < kblocks; kb = kb + 1) begin
                    stream_weight_beat(unit_scales());
                    for (sub = 0; sub < 4; sub = sub + 1)
                        stream_weight_beat(q1_bits(rb, sub));
                end
            end
            w_valid = 1'b0;
            w_data = 512'd0;
            weight_driver_done = 1'b1;
        end
    endtask

    task automatic send_q2_projection_contiguous;
        input integer rowblocks;
        input integer kblocks;
        integer rb;
        integer kb;
        integer sub;
        begin
            @(negedge clk);
            w_valid = 1'b1;
            for (rb = 0; rb < rowblocks; rb = rb + 1) begin
                for (kb = 0; kb < kblocks; kb = kb + 1) begin
                    stream_weight_beat(unit_scales());
                    for (sub = 0; sub < 4; sub = sub + 1) begin
                        stream_weight_beat(q2_codes(rb, sub, 0));
                        stream_weight_beat(q2_codes(rb, sub, 1));
                    end
                end
            end
            w_valid = 1'b0;
            w_data = 512'd0;
            weight_driver_done = 1'b1;
        end
    endtask

    function automatic integer expected_target;
        input integer fmt;
        input integer row;
        begin
            expected_target = (fmt == WEIGHT_Q1) ? q1_target(row) : q2_target(row);
        end
    endfunction

    function automatic integer q1_sub_target;
        input integer global_row;
        input integer subblock;
        integer lane;
        integer element;
        integer positives;
        integer value;
        begin
            positives = (q1_target(global_row) + 128) / 2;
            value = 0;
            for (lane = 0; lane < 32; lane = lane + 1) begin
                element = subblock*32 + lane;
                value = value + ((element < positives) ? 1 : -1);
            end
            q1_sub_target = value;
        end
    endfunction

    function automatic integer q2_sub_target;
        input integer global_row;
        input integer subblock;
        integer lane;
        integer element;
        integer value;
        begin
            value = 0;
            for (lane = 0; lane < 32; lane = lane + 1) begin
                element = subblock*32 + lane;
                case (q2_code(global_row, element))
                    2'd0: value = value - 1;
                    2'd2: value = value + 1;
                    default: ;
                endcase
            end
            q2_sub_target = value;
        end
    endfunction

    function automatic integer expected_projection;
        input integer fmt;
        input integer row;
        input integer token;
        integer kblock;
        integer subblock;
        integer activation_value;
        integer weight_sum;
        integer value;
        begin
            value = 0;
            for (kblock = 0; kblock < expected_k_blocks;
                 kblock = kblock + 1) begin
                for (subblock = 0; subblock < 4;
                     subblock = subblock + 1) begin
                    weight_sum = (fmt == WEIGHT_Q1) ?
                        q1_sub_target(row, subblock) :
                        q2_sub_target(row, subblock);
                    activation_value = token + 1;
                    if (tagged_activations)
                        activation_value = activation_value +
                                           kblock*4 + subblock;
                    value = value + weight_sum*activation_value;
                end
            end
            expected_projection = value;
        end
    endfunction

    reg signed [103:0] expected_acc;
    integer expected_integer;
    integer seen_index;
    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (cycle > 100000)
            $fatal(1,
                "timeout state=%0d inflight=%0d credit=%0d fifo=%0d serial=%0d lane=%0d requested=%b received=%b reqv=%b rspv=%b rspr=%b",
                dut.state, dut.inflight_waves, dut.serial_credit_used,
                dut.dot_fifo_count, dut.serial_active, dut.serial_lane,
                dut.wave_requested, dut.wave_received, act_req_valid,
                act_rsp_valid, act_rsp_ready);

        if (!rst_n || clear || (start && !busy)) begin
            prior_act_rsp_fire <= 1'b0;
        end else begin
            if (dut.dot_issue_valid !== prior_act_rsp_fire)
                $fatal(1,
                    "activation input register latency changed issue=%b prior_fire=%b",
                    dut.dot_issue_valid, prior_act_rsp_fire);
            prior_act_rsp_fire <= act_rsp_valid && act_rsp_ready;
        end
        if (rst_n && act_rsp_valid && !act_rsp_ready)
            response_backpressure_cycles <=
                response_backpressure_cycles + 1;
        if (rst_n && act_req_valid && !act_req_ready)
            request_backpressure_cycles <=
                request_backpressure_cycles + 1;
        if (rst_n && randomize_act_service &&
            (act_model_count != 0) && !auto_act_rsp_valid_q &&
            !auto_act_release)
            queued_response_delay_cycles <=
                queued_response_delay_cycles + 1;

        if (!rst_n || clear) begin
            held_act_rsp_last_cycle <= 1'b0;
        end else begin
            if (held_act_rsp_last_cycle) begin
                if (!act_rsp_valid || (act_rsp_data !== held_act_rsp_data))
                    $fatal(1, "activation response changed under backpressure");
            end
            held_act_rsp_last_cycle <= act_rsp_valid && !act_rsp_ready;
            if (act_rsp_valid && !act_rsp_ready)
                held_act_rsp_data <= act_rsp_data;
        end

        if (act_req_valid && act_req_ready) begin
            active_wave_count = (|expected_token_mask[3:0]) +
                                (|expected_token_mask[7:4]);
            if (active_wave_count == 0)
                $fatal(1, "activation request issued for an empty token mask");
            request_selector_index = observed_act_requests /
                                     active_wave_count;
            request_wave_index = observed_act_requests % active_wave_count;
            request_within_rowblock = request_selector_index %
                                      (expected_k_blocks*4);
            expected_request_addr = request_within_rowblock;
            if ((|expected_token_mask[3:0]) &&
                (|expected_token_mask[7:4]))
                expected_request_wave = request_wave_index;
            else
                expected_request_wave = |expected_token_mask[7:4];
            if ((act_req_addr !== expected_request_addr[8:0]) ||
                (act_req_wave !== expected_request_wave[0]))
                $fatal(1,
                    "activation request order mismatch index=%0d got=%0d/%0d expected=%0d/%0d",
                    observed_act_requests, act_req_addr, act_req_wave,
                    expected_request_addr, expected_request_wave);
            observed_act_requests = observed_act_requests + 1;
        end
        if (act_rsp_valid && act_rsp_ready) begin
            if (observed_act_responses >= observed_act_requests)
                $fatal(1, "activation response accepted without request tag");
            observed_act_responses = observed_act_responses + 1;
        end

        if (measure_single_token_cadence && w_valid && w_ready) begin
            if (cadence_weight_index == 0) begin
                if (cadence_group_count >= MAX_CADENCE_GROUPS)
                    $fatal(1, "cadence group scoreboard overflow");
                cadence_scale_cycles[cadence_group_count] = cycle;
                if (cadence_last_scale_cycle >= 0) begin
                    cadence_delta = cycle - cadence_last_scale_cycle;
                    if (cadence_delta < cadence_min_scale_delta)
                        cadence_min_scale_delta = cadence_delta;
                    if (cadence_delta > cadence_max_scale_delta)
                        cadence_max_scale_delta = cadence_delta;
                end
                cadence_last_scale_cycle = cycle;
                cadence_group_count = cadence_group_count + 1;
            end
            if (cadence_weight_index ==
                ((expected_fmt == WEIGHT_Q1) ? 4 : 8))
                cadence_weight_index = 0;
            else
                cadence_weight_index = cadence_weight_index + 1;
        end
        if (measure_single_token_cadence && act_rsp_valid && act_rsp_ready) begin
            if ((cadence_response_count % 4) == 3) begin
                cadence_response_group = cadence_response_count / 4;
                if (cadence_response_group >= cadence_group_count)
                    $fatal(1, "selector response completed before group scale");
                cadence_span = cycle -
                    cadence_scale_cycles[cadence_response_group] + 1;
                if (cadence_span > cadence_max_completion_span)
                    cadence_max_completion_span = cadence_span;
            end
            cadence_response_count = cadence_response_count + 1;
        end

        if (enable_output_stalls)
            out_ready <= ((cycle % 7) != 2) && ((cycle % 7) != 3) &&
                         ((cycle % 11) != 5);
        else
            out_ready <= 1'b1;

        if (stalled_last_cycle) begin
            if (!out_valid || (out_acc !== stalled_acc) ||
                (out_emin !== stalled_emin) || (out_token !== stalled_token) ||
                (out_row !== stalled_row) || (out_last !== stalled_last))
                $fatal(1, "output changed under backpressure");
        end
        stalled_last_cycle <= out_valid && !out_ready;
        if (out_valid && !out_ready) begin
            stalled_acc <= out_acc;
            stalled_emin <= out_emin;
            stalled_token <= out_token;
            stalled_row <= out_row;
            stalled_last <= out_last;
        end

        if (out_valid && out_ready) begin
            if (out_row >= expected_m)
                $fatal(1, "row tag out of range: %0d", out_row);
            if (!expected_token_mask[out_token])
                $fatal(1, "inactive token emitted: %0d", out_token);
            if (out_emin != -8'sd20)
                $fatal(1, "wrong output exponent: %0d", out_emin);
            seen_index = out_token*MAX_TEST_M + out_row;
            if (seen_outputs[seen_index])
                $fatal(1, "duplicate output token=%0d row=%0d", out_token, out_row);
            seen_outputs[seen_index] = 1'b1;
            expected_integer = expected_projection(
                expected_fmt, out_row, out_token);
            expected_acc = expected_integer;
            expected_acc = expected_acc <<< 20;
            if (out_acc !== expected_acc)
                $fatal(1,
                    "bad acc fmt=%0d token=%0d row=%0d got=%0d expected=%0d",
                    expected_fmt, out_token, out_row, $signed(out_acc),
                    $signed(expected_acc));
            observed_outputs = observed_outputs + 1;
            if (out_last != (observed_outputs ==
                ($countones(expected_token_mask) * expected_m)))
                $fatal(1, "out_last mismatch at output %0d", observed_outputs);
        end

        if (rst_n) begin
            if (dut.dot_fifo_push && dut.serial_pop)
                fifo_push_pop_events <= fifo_push_pop_events + 1;
            if (!dut.single_token_fast_mode && (dut.serial_credit_used > 4))
                $fatal(1, "serializer credit overflow: %0d",
                    dut.serial_credit_used);
            if (dut.single_token_fast_mode && (dut.serial_credit_used > 18))
                $fatal(1, "single-token serializer credit overflow: %0d",
                    dut.serial_credit_used);
            if (dut.dot_fifo_count > 4)
                $fatal(1, "dot FIFO overflow: %0d", dut.dot_fifo_count);
            if (dut.dot_fifo_push && !dut.serial_pop &&
                (dut.dot_fifo_count == 4))
                $fatal(1, "dot record arrived without FIFO capacity");
            if (dut.serial_pop && (dut.serial_credit_used == 0))
                $fatal(1, "serializer popped an unreserved record");
            if (dut.single_token_fast_mode && busy) begin
                if (dut.single_token_selector_fifo_count > 4)
                    $fatal(1, "single-token selector FIFO overflow: %0d",
                        dut.single_token_selector_fifo_count);
                if (dut.single_token_selector_unrequested_count >
                    dut.single_token_selector_fifo_count)
                    $fatal(1, "single-token unrequested count exceeds live selectors");
                if (dut.single_token_selector_fifo_full !==
                    (dut.single_token_selector_fifo_count == 4))
                    $fatal(1, "single-token selector FIFO full flag mismatch");
                if (dut.single_token_selector_pop !==
                    (act_rsp_valid && act_rsp_ready))
                    $fatal(1, "single-token selector response/tag pop mismatch");
                if ((act_req_valid && act_req_ready) &&
                    (dut.single_token_selector_unrequested_count == 0))
                    $fatal(1, "single-token request issued without an unrequested tag");
                if (dut.single_token_selector_pop &&
                    (dut.single_token_selector_fifo_count <=
                     dut.single_token_selector_unrequested_count))
                    $fatal(1, "single-token response popped an unissued selector tag");
                if (dut.single_token_selector_push && !dut.single_token_selector_pop &&
                    (dut.single_token_selector_fifo_count == 4))
                    $fatal(1, "single-token selector pushed without FIFO capacity");
                if (dut.single_token_selector_fifo_full && dut.single_token_selector_pop &&
                    dut.single_token_selector_push)
                    $fatal(1, "full single-token selector FIFO accepted a same-cycle push");
                if (dut.single_token_selector_fifo_full)
                    single_token_fifo_full_cycles <= single_token_fifo_full_cycles + 1;
                if (dut.single_token_selector_push && dut.single_token_selector_pop)
                    single_token_fifo_push_pop_events <=
                        single_token_fifo_push_pop_events + 1;
            end
            if (busy && (dut.state == 4'd6) &&
                (dut.single_token_selector_fifo_count == 0) &&
                (dut.single_token_selector_unrequested_count == 0) &&
                (dut.inflight_waves == 0) && !dut.mul_valid &&
                !dut.cell_update_pending[0]) begin
                if ((dut.single_token_selector_enq_ptr != dut.single_token_selector_req_ptr) ||
                    (dut.single_token_selector_req_ptr != dut.single_token_selector_rsp_ptr) ||
                    (dut.serial_credit_used != 0) ||
                    (dut.dot_fifo_count != 0) || dut.serial_active ||
                    dut.act_rsp_pipe_valid_q || dut.dot_issue_valid)
                    $fatal(1, "projection released ST_DRAIN with live work");
            end
            if (act_model_count > ACT_MODEL_DEPTH)
                $fatal(1, "activation model request FIFO overflow: %0d",
                    act_model_count);
        end
    end

    initial begin
        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        // A contiguous producer exposes the selector pipeline's steady-state
        // intake rate.  Address-dependent activations make every selector's
        // response distinguishable, so exact arithmetic also checks tag order.
        tagged_activations = 1'b1;
        measure_single_token_cadence = 1'b1;
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q1, 16'd512, 18'd16, 16'd1, 8'h01);
        send_q1_projection_contiguous(1, 4);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q1/single-token cadence run reported error");
        if ((weight_beat_count != 32'd20) ||
            (wave_issue_count != 32'd16) ||
            (observed_outputs != 16))
            $fatal(1, "Q1/single-token cadence run count mismatch");
        if ((cadence_group_count != 4) ||
            (cadence_response_count != 16) ||
            (cadence_min_scale_delta != 5) ||
            (cadence_max_scale_delta != 5))
            $fatal(1,
                "Q1/single-token cadence groups=%0d responses=%0d scale delta=%0d..%0d",
                cadence_group_count, cadence_response_count,
                cadence_min_scale_delta, cadence_max_scale_delta);
        if ((cadence_max_completion_span == 0) ||
            (cadence_max_completion_span > 9))
            $fatal(1, "Q1/single-token completion span %0d exceeds 9 cycles",
                cadence_max_completion_span);
        if ((observed_act_requests != 16) ||
            (observed_act_responses != 16))
            $fatal(1, "Q1/single-token activation count mismatch req=%0d rsp=%0d",
                observed_act_requests, observed_act_responses);
        $display("single-token Q1 cadence: scale delta %0d, completion span %0d",
            cadence_max_scale_delta, cadence_max_completion_span);

        // Repeat on physical lane 3 of the upper wave and the Q2 decoder.  This
        // catches lane-selection mistakes that a lane-0-only test would hide.
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q2, 16'd512, 18'd16, 16'd1, 8'h80);
        send_q2_projection_contiguous(1, 4);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q2/single-token cadence run reported error");
        if ((weight_beat_count != 32'd36) ||
            (wave_issue_count != 32'd16) ||
            (observed_outputs != 16))
            $fatal(1, "Q2/single-token cadence run count mismatch");
        if ((cadence_group_count != 4) ||
            (cadence_response_count != 16) ||
            (cadence_min_scale_delta != 9) ||
            (cadence_max_scale_delta != 9))
            $fatal(1,
                "Q2/single-token cadence groups=%0d responses=%0d scale delta=%0d..%0d",
                cadence_group_count, cadence_response_count,
                cadence_min_scale_delta, cadence_max_scale_delta);
        if ((cadence_max_completion_span == 0) ||
            (cadence_max_completion_span > 13))
            $fatal(1, "Q2/single-token completion span %0d exceeds 13 cycles",
                cadence_max_completion_span);
        $display("single-token Q2 cadence: scale delta %0d, completion span %0d",
            cadence_max_scale_delta, cadence_max_completion_span);
        measure_single_token_cadence = 1'b0;

        // Hold all Q8 responses until the four-entry selector FIFO fills.  The
        // following group's scale may pass, but its first completing selector
        // beat must remain stable and backpressured until a response retires.
        hold_act_responses = 1'b1;
        single_token_fifo_full_cycles_before = single_token_fifo_full_cycles;
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q1, 16'd256, 18'd16, 16'd1, 8'h01);
        fork
            send_q1_projection_contiguous(1, 2);
        join_none
        wait (dut.single_token_selector_fifo_full);
        repeat (2) @(posedge clk);
        if (dut.single_token_selector_fifo_count != 4)
            $fatal(1, "single-token selector FIFO did not remain full");
        if (!w_valid || w_ready)
            $fatal(1, "full selector FIFO did not backpressure weight input");
        hold_act_responses = 1'b0;
        wait (weight_driver_done);
        wait (done);
        @(posedge clk);
        if (error || (observed_outputs != 16))
            $fatal(1, "single-token selector FIFO full run failed");
        if (single_token_fifo_full_cycles == single_token_fifo_full_cycles_before)
            $fatal(1, "single-token selector FIFO full condition was not observed");

        // Deterministic pseudo-random request stalls and response delays stress
        // ordered tags across rowblocks while output backpressure is also live.
        randomize_act_service = 1'b1;
        enable_output_stalls = 1'b1;
        request_backpressure_before = request_backpressure_cycles;
        response_delay_before = queued_response_delay_cycles;
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q1, 16'd512, 18'd17, 16'd2, 8'h80);
        send_q1_projection_contiguous(2, 4);
        wait (done);
        @(posedge clk);
        if (error || (weight_beat_count != 32'd40) ||
            (wave_issue_count != 32'd32) || (observed_outputs != 17))
            $fatal(1, "random-delay single-token run failed");
        if (request_backpressure_cycles == request_backpressure_before)
            $fatal(1, "random service never backpressured a Q8 request");
        if (queued_response_delay_cycles == response_delay_before)
            $fatal(1, "random service never delayed a queued Q8 response");
        randomize_act_service = 1'b0;
        enable_output_stalls = 1'b0;

        // Abort with four decoded selectors and their Q8 requests live, then
        // restart without resetting payload RAM.  Only valid/count ownership is
        // cleared, and no stale selector may reach the dot or accumulator path.
        hold_act_responses = 1'b1;
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'h01);
        send_q1_projection_contiguous(1, 1);
        wait (dut.single_token_selector_fifo_count == 4);
        wait (dut.single_token_selector_unrequested_count == 0);
        wait (act_model_count == 4);
        @(negedge clk);
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        hold_act_responses = 1'b0;
        if (busy || out_valid || act_req_valid || act_rsp_valid ||
            (dut.single_token_selector_fifo_count != 0) ||
            (dut.single_token_selector_unrequested_count != 0) ||
            (dut.single_token_selector_enq_ptr != 0) ||
            (dut.single_token_selector_req_ptr != 0) ||
            (dut.single_token_selector_rsp_ptr != 0) ||
            (dut.serial_credit_used != 0) || (act_model_count != 0))
            $fatal(1, "clear did not flush queued single-token selector ownership");
        tagged_activations = 1'b0;
        weight_driver_done = 1'b0;
        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'h01);
        send_q1_projection_contiguous(1, 1);
        wait (done);
        @(posedge clk);
        if (error || (observed_outputs != 16))
            $fatal(1, "queued-selector clear/restart run failed");

        // single-token decode-shaped Q1 run on the upper physical wave.  The first arena
        // response is presented with zero request latency and held until the
        // request tag makes the engine ready.
        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'h10);
        send_q1_projection_early_first_response();
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q1/single-token run reported error");
        if (weight_beat_count != 32'd5)
            $fatal(1, "Q1/single-token weight count %0d != 5", weight_beat_count);
        if (wave_issue_count != 32'd4)
            $fatal(1, "Q1/single-token wave count %0d != 4", wave_issue_count);
        if (observed_outputs != 16)
            $fatal(1, "Q1/single-token output count %0d != 16", observed_outputs);

        // Minimum-cadence Q1 stress: a one-cycle activation service returns
        // both four-lane waves as soon as each request is issued. Credit reservation
        // must throttle requests without overflowing the post-dot FIFO.
        fifo_push_pop_before = fifo_push_pop_events;
        launch_run(WEIGHT_Q1, 16'd128, 18'd1, 16'd1, 8'hff);
        send_q1_projection(1);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q1 tile-8 cadence run reported error");
        if (weight_beat_count != 32'd5)
            $fatal(1, "Q1 tile-8 weight count %0d != 5", weight_beat_count);
        if (wave_issue_count != 32'd8)
            $fatal(1, "Q1 tile-8 wave count %0d != 8", wave_issue_count);
        if (observed_outputs != 8)
            $fatal(1, "Q1 tile-8 output count %0d != 8", observed_outputs);
        if (fifo_push_pop_events == fifo_push_pop_before)
            $fatal(1, "Q1 tile-8 cadence never exercised FIFO push+pop");

        // Clear while an accepted arena response is resident in the new input
        // register but has not yet entered dot stage 0.  Its valid ownership and
        // every downstream valid must be quarantined before restart.
        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'h01);
        send_weight(unit_scales());
        send_weight(~q1_bits(0, 0));
        wait (dut.act_rsp_pipe_valid_q);
        @(negedge clk);
        if (!dut.dot_issue_valid)
            $fatal(1, "clear test missed registered activation response");
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        if (dut.act_rsp_pipe_valid_q || dut.dot_issue_valid || busy || out_valid)
            $fatal(1, "clear retained registered activation ownership");
        repeat (14) begin
            @(posedge clk);
            if (dut.dot_fifo_push || out_valid)
                $fatal(1, "cleared activation response reached output pipeline");
        end

        // Abort with activation responses and dot records live, then restart
        // without resetting accumulator payload.  The restarted exact result
        // proves that clear flushes ownership/control and first-K replacement
        // prevents stale digit state from leaking into the next projection.
        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'hff);
        send_weight(unit_scales());
        send_weight(~q1_bits(0, 0));
        wait (dut.dot_fifo_push);
        @(posedge clk);
        pre_clear_weight_sign = dut.held_weight_sign;
        pre_clear_weight_nonzero = dut.held_weight_nonzero;
        @(negedge clk);
        if (dut.dot_fifo_count == 0)
            $fatal(1, "clear test did not leave a live FIFO payload");
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        if ((dut.held_weight_sign !== pre_clear_weight_sign) ||
            (dut.held_weight_nonzero !== pre_clear_weight_nonzero))
            $fatal(1, "clear modified held-weight payload storage");
        if (busy || done || error || out_valid || act_req_valid ||
            act_rsp_ready || (dut.dot_fifo_count != 0) ||
            (dut.dot_fifo_wr_ptr != 0) || (dut.dot_fifo_rd_ptr != 0) ||
            (dut.serial_credit_used != 0) || (dut.inflight_waves != 0) ||
            dut.drain_stage_valid || dut.cell_update_pending[0])
            $fatal(1, "clear did not flush live projection state");

        stale_fifo0 = dut.dot_fifo[0];
        stale_fifo1 = dut.dot_fifo[1];
        stale_fifo2 = dut.dot_fifo[2];
        stale_fifo3 = dut.dot_fifo[3];

        launch_run(WEIGHT_Q1, 16'd128, 18'd16, 16'd1, 8'h01);
        repeat (4) begin
            @(posedge clk);
            if (act_req_valid || act_rsp_ready || dut.dot_fifo_push ||
                dut.serial_pop || dut.serial_active || out_valid ||
                (dut.dot_fifo_count != 0) ||
                (dut.dot_fifo_wr_ptr != 0) || (dut.dot_fifo_rd_ptr != 0))
                $fatal(1, "restart exposed stale projection payload");
        end
        if ((dut.held_weight_sign !== pre_clear_weight_sign) ||
            (dut.held_weight_nonzero !== pre_clear_weight_nonzero) ||
            (dut.dot_fifo[0] !== stale_fifo0) ||
            (dut.dot_fifo[1] !== stale_fifo1) ||
            (dut.dot_fifo[2] !== stale_fifo2) ||
            (dut.dot_fifo[3] !== stale_fifo3))
            $fatal(1, "start modified stale payload storage");
        send_q1_projection(1);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q1 clear/restart run reported error");
        if (observed_outputs != 16)
            $fatal(1, "Q1 clear/restart output count %0d != 16",
                observed_outputs);

        // Tile-8 prefill-shaped Q2 run: two physical waves per selector beat,
        // two rowblocks with a one-row M tail, and sustained output stalls.
        enable_output_stalls = 1'b1;
        launch_run(WEIGHT_Q2, 16'd128, 18'd17, 16'd2, 8'hff);
        send_q2_projection(2);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q2 tile-8 run reported error");
        if (weight_beat_count != 32'd18)
            $fatal(1, "Q2 tile-8 weight count %0d != 18", weight_beat_count);
        if (wave_issue_count != 32'd16)
            $fatal(1, "Q2 tile-8 wave count %0d != 16", wave_issue_count);
        if (observed_outputs != 136)
            $fatal(1, "Q2 tile-8 output count %0d != 136", observed_outputs);

        // LM-head style final-token projection: only wave 1/lane 3 is live.
        // This also checks sparse-mask request, clear, drain, and output tags
        // under sustained downstream backpressure.
        enable_output_stalls = 1'b1;
        launch_run(WEIGHT_Q1, 16'd128, 18'd17, 16'd2, 8'h80);
        send_q1_projection(2);
        wait (done);
        @(posedge clk);
        if (error) $fatal(1, "Q1/token7 run reported error");
        if (weight_beat_count != 32'd10)
            $fatal(1, "Q1/token7 weight count %0d != 10", weight_beat_count);
        if (wave_issue_count != 32'd8)
            $fatal(1, "Q1/token7 wave count %0d != 8", wave_issue_count);
        if (observed_outputs != 17)
            $fatal(1, "Q1/token7 output count %0d != 17", observed_outputs);

        // An empty mask is the sole invalid token selection.
        enable_output_stalls = 1'b0;
        @(negedge clk);
        model_spec_k = 16'd128;
        model_spec_m = 18'd16;
        model_spec_rowblocks = 16'd1;
        weight_fmt = WEIGHT_Q1;
        token_mask = 8'h00;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        if (busy || !done || !error)
            $fatal(1, "empty token mask was not rejected");

        // The LM-head model_spec is wider than the hidden/FFN projections.  This
        // directed shape test proves the full 18-bit M value and 9,480
        // rowblocks survive command capture.  Clear aborts before streaming the
        // deliberately enormous projection.
        launch_run(WEIGHT_Q1, 16'd128, 18'd151669, 16'd9480, 8'h80);
        if (dut.run_m != 18'd151669)
            $fatal(1, "LM-head M truncated to %0d", dut.run_m);
        if (dut.run_rowblocks != 16'd9480)
            $fatal(1, "LM-head rowblock count changed to %0d", dut.run_rowblocks);
        @(negedge clk);
        clear = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear = 1'b0;
        if (busy) $fatal(1, "clear did not abort LM-head shape test");

        $display("PASS  projection_engine exact Q1/Q2 single-token/tile-8 backpressure");
        $finish;
    end
endmodule

`default_nettype wire
