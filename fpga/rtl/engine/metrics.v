`default_nettype none

// Low-fanout execution recorder. Event inputs are registered at their source;
// this module only observes them and never feeds a functional ready/valid path.
// A terminal edge is followed by one seal cycle so the final registered event
// sample is included before the single snapshot becomes visible to software.
module engine_metrics #(
    parameter integer COUNTER_W = 32,
    parameter integer CALL_COUNTER_W = 8
) (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start,
    input  wire [31:0] start_tag,
    input  wire        finish,
    input  wire [1:0]  finish_outcome,
    input  wire        acknowledge,

    input  wire        core_stage_active,
    input  wire [4:0]  core_stage,
    input  wire        core_stage_call,
    input  wire [4:0]  core_stage_call_id,

    // [0] weight beat, [1] weight source empty, [2] weight consumer blocked,
    // [3] selector full, [6:4] selector level, [7] Q8 request,
    // [8] Q8 response, [9] Q8 request wait, [10] Q8 response wait,
    // [11] drain, [12] bank wait.
    input  wire [12:0] projection_probe,
    input  wire [2:0]  weight_axi_r_beats,
    input  wire [2:0]  weight_axi_r_gap_ports,
    input  wire        weight_zip_skew,
    input  wire [1:0]  history_axi_r_beats,
    input  wire        kv_axi_w_beat,

    input  wire [6:0]  read_index,
    output reg  [31:0] read_data_lo,
    output reg  [31:0] read_data_hi,
    output wire [31:0] schema,
    output wire [31:0] capabilities,
    output wire [31:0] status,
    output wire [31:0] snapshot_tag,
    output wire [31:0] overflow0,
    output wire [31:0] overflow1,
    output wire [31:0] overflow2,
    output wire [31:0] overflow3,
    output wire [63:0] total_cycles,
    output wire        recording,
    output wire        sealing,
    output wire        snapshot_valid
);
    localparam integer METRIC_COUNT = 40;
    localparam [31:0] METRICS_SCHEMA = 32'h0001_0000;
    localparam [31:0] METRICS_CAPABILITIES = 32'h0028_0b1d;

    // Cycle/event counters are u32, stage calls are u8, and selector occupancy
    // is u3 in production. Width parameters are reduced only by verification
    // harnesses so saturation remains reachable in short tests and proofs.
    reg [COUNTER_W-1:0] cycle_counters_q [1:12];
    reg [CALL_COUNTER_W-1:0] call_counters_q [13:23];
    reg [COUNTER_W-1:0] event_lo_q [24:27];
    reg [COUNTER_W-1:0] event_hi_q [29:39];
    reg [2:0] selector_high_water_q;
    reg [39:0] overflow_q;
    reg [63:0] live_total_q;
    reg [63:0] last_total_q;
    reg [31:0] tag_q;
    reg [1:0] outcome_q;
    reg recording_q;
    reg sealing_q;
    reg snapshot_valid_q;

    // Controller cycles and calls are one-hot, so each bank shares one adder.
    // Physical event counters remain independent because they can coincide.
    reg [5:0] cycle_metric_id;
    reg [5:0] call_metric_id;
    reg [COUNTER_W-1:0] cycle_counter_value;
    reg [CALL_COUNTER_W-1:0] call_counter_value;
    always @* begin
        cycle_metric_id = 6'd1;
        if (core_stage_active) begin
            case (core_stage)
                5'd0:  cycle_metric_id = 6'd2;
                5'd1:  cycle_metric_id = 6'd3;
                5'd2:  cycle_metric_id = 6'd4;
                5'd4:  cycle_metric_id = 6'd5;
                5'd5:  cycle_metric_id = 6'd6;
                5'd6:  cycle_metric_id = 6'd7;
                5'd8:  cycle_metric_id = 6'd8;
                5'd9:  cycle_metric_id = 6'd9;
                5'd11: cycle_metric_id = 6'd10;
                5'd13: cycle_metric_id = 6'd11;
                5'd14: cycle_metric_id = 6'd12;
                default: cycle_metric_id = 6'd1;
            endcase
        end

        case (cycle_metric_id)
            6'd2:  cycle_counter_value = cycle_counters_q[2];
            6'd3:  cycle_counter_value = cycle_counters_q[3];
            6'd4:  cycle_counter_value = cycle_counters_q[4];
            6'd5:  cycle_counter_value = cycle_counters_q[5];
            6'd6:  cycle_counter_value = cycle_counters_q[6];
            6'd7:  cycle_counter_value = cycle_counters_q[7];
            6'd8:  cycle_counter_value = cycle_counters_q[8];
            6'd9:  cycle_counter_value = cycle_counters_q[9];
            6'd10: cycle_counter_value = cycle_counters_q[10];
            6'd11: cycle_counter_value = cycle_counters_q[11];
            6'd12: cycle_counter_value = cycle_counters_q[12];
            default: cycle_counter_value = cycle_counters_q[1];
        endcase

        call_metric_id = 6'd0;
        if (core_stage_call) begin
            case (core_stage_call_id)
                5'd0:  call_metric_id = 6'd13;
                5'd1:  call_metric_id = 6'd14;
                5'd2:  call_metric_id = 6'd15;
                5'd4:  call_metric_id = 6'd16;
                5'd5:  call_metric_id = 6'd17;
                5'd6:  call_metric_id = 6'd18;
                5'd8:  call_metric_id = 6'd19;
                5'd9:  call_metric_id = 6'd20;
                5'd11: call_metric_id = 6'd21;
                5'd13: call_metric_id = 6'd22;
                5'd14: call_metric_id = 6'd23;
                default: call_metric_id = 6'd0;
            endcase
        end

        case (call_metric_id)
            6'd13: call_counter_value = call_counters_q[13];
            6'd14: call_counter_value = call_counters_q[14];
            6'd15: call_counter_value = call_counters_q[15];
            6'd16: call_counter_value = call_counters_q[16];
            6'd17: call_counter_value = call_counters_q[17];
            6'd18: call_counter_value = call_counters_q[18];
            6'd19: call_counter_value = call_counters_q[19];
            6'd20: call_counter_value = call_counters_q[20];
            6'd21: call_counter_value = call_counters_q[21];
            6'd22: call_counter_value = call_counters_q[22];
            6'd23: call_counter_value = call_counters_q[23];
            default: call_counter_value = {CALL_COUNTER_W{1'b0}};
        endcase
    end

    wire [COUNTER_W:0] cycle_counter_sum =
        {1'b0, cycle_counter_value} + {{COUNTER_W{1'b0}}, 1'b1};
    wire [CALL_COUNTER_W:0] call_counter_sum =
        {1'b0, call_counter_value} + {{CALL_COUNTER_W{1'b0}}, 1'b1};

    reg [2:0] increment [24:METRIC_COUNT-1];
    integer increment_index;
    always @* begin
        for (increment_index = 24; increment_index < METRIC_COUNT;
             increment_index = increment_index + 1)
            increment[increment_index] = 3'd0;

        // Sealing samples only registered physical events. Controller time and
        // stage calls ended on the terminal edge and are not extended here.
        if (recording_q || sealing_q) begin
            increment[24] = {2'd0, projection_probe[0]};
            increment[25] = {2'd0, projection_probe[1]};
            increment[26] = {2'd0, projection_probe[2]};
            increment[27] = {2'd0, projection_probe[3]};
            increment[29] = {2'd0, projection_probe[7]};
            increment[30] = {2'd0, projection_probe[8]};
            increment[31] = {2'd0, projection_probe[9]};
            increment[32] = {2'd0, projection_probe[10]};
            increment[33] = {2'd0, projection_probe[11]};
            increment[34] = {2'd0, projection_probe[12]};
            increment[35] = weight_axi_r_beats;
            increment[36] = weight_axi_r_gap_ports;
            increment[37] = {2'd0, weight_zip_skew};
            increment[38] = {1'b0, history_axi_r_beats};
            increment[39] = {2'd0, kv_axi_w_beat};
        end
    end

    integer counter_index;
    reg [COUNTER_W:0] counter_sum;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (counter_index = 1; counter_index <= 12;
                 counter_index = counter_index + 1)
                cycle_counters_q[counter_index] <= {COUNTER_W{1'b0}};
            for (counter_index = 13; counter_index <= 23;
                 counter_index = counter_index + 1)
                call_counters_q[counter_index] <=
                    {CALL_COUNTER_W{1'b0}};
            for (counter_index = 24; counter_index <= 27;
                 counter_index = counter_index + 1)
                event_lo_q[counter_index] <= {COUNTER_W{1'b0}};
            for (counter_index = 29; counter_index < METRIC_COUNT;
                 counter_index = counter_index + 1)
                event_hi_q[counter_index] <= {COUNTER_W{1'b0}};
            selector_high_water_q <= 3'd0;
            overflow_q <= 40'd0;
            live_total_q <= 64'd0;
            last_total_q <= 64'd0;
            tag_q <= 32'd0;
            outcome_q <= 2'd0;
            recording_q <= 1'b0;
            sealing_q <= 1'b0;
            snapshot_valid_q <= 1'b0;
        end else begin
            if (acknowledge && snapshot_valid_q)
                snapshot_valid_q <= 1'b0;

            if (start) begin
                for (counter_index = 1; counter_index <= 12;
                     counter_index = counter_index + 1)
                    cycle_counters_q[counter_index] <= {COUNTER_W{1'b0}};
                for (counter_index = 13; counter_index <= 23;
                     counter_index = counter_index + 1)
                    call_counters_q[counter_index] <=
                        {CALL_COUNTER_W{1'b0}};
                for (counter_index = 24; counter_index <= 27;
                     counter_index = counter_index + 1)
                    event_lo_q[counter_index] <= {COUNTER_W{1'b0}};
                for (counter_index = 29; counter_index < METRIC_COUNT;
                     counter_index = counter_index + 1)
                    event_hi_q[counter_index] <= {COUNTER_W{1'b0}};
                selector_high_water_q <= 3'd0;
                overflow_q <= 40'd0;
                live_total_q <= 64'd0;
                tag_q <= start_tag;
                outcome_q <= 2'd0;
                recording_q <= 1'b1;
                sealing_q <= 1'b0;
                snapshot_valid_q <= 1'b0;
            end else begin
                if (recording_q)
                    live_total_q <= live_total_q + 1'b1;

                if (recording_q) begin
                    if (cycle_counter_sum[COUNTER_W]) begin
                        overflow_q[cycle_metric_id] <= 1'b1;
                    end else begin
                        case (cycle_metric_id)
                            6'd1: cycle_counters_q[1] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd2: cycle_counters_q[2] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd3: cycle_counters_q[3] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd4: cycle_counters_q[4] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd5: cycle_counters_q[5] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd6: cycle_counters_q[6] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd7: cycle_counters_q[7] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd8: cycle_counters_q[8] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd9: cycle_counters_q[9] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd10: cycle_counters_q[10] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd11: cycle_counters_q[11] <= cycle_counter_sum[COUNTER_W-1:0];
                            6'd12: cycle_counters_q[12] <= cycle_counter_sum[COUNTER_W-1:0];
                            default: ;
                        endcase
                    end

                    if (call_metric_id != 6'd0) begin
                        if (call_counter_sum[CALL_COUNTER_W]) begin
                            overflow_q[call_metric_id] <= 1'b1;
                        end else begin
                            case (call_metric_id)
                                6'd13: call_counters_q[13] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd14: call_counters_q[14] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd15: call_counters_q[15] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd16: call_counters_q[16] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd17: call_counters_q[17] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd18: call_counters_q[18] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd19: call_counters_q[19] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd20: call_counters_q[20] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd21: call_counters_q[21] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd22: call_counters_q[22] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                6'd23: call_counters_q[23] <= call_counter_sum[CALL_COUNTER_W-1:0];
                                default: ;
                            endcase
                        end
                    end
                end

                for (counter_index = 24; counter_index <= 27;
                     counter_index = counter_index + 1) begin
                    if (increment[counter_index] != 3'd0) begin
                        counter_sum = {1'b0, event_lo_q[counter_index]} +
                                      increment[counter_index];
                        if (counter_sum[COUNTER_W]) begin
                            event_lo_q[counter_index] <= {COUNTER_W{1'b1}};
                            overflow_q[counter_index] <= 1'b1;
                        end else begin
                            event_lo_q[counter_index] <=
                                counter_sum[COUNTER_W-1:0];
                        end
                    end
                end

                for (counter_index = 29; counter_index < METRIC_COUNT;
                     counter_index = counter_index + 1) begin
                    if (increment[counter_index] != 3'd0) begin
                        counter_sum = {1'b0, event_hi_q[counter_index]} +
                                      increment[counter_index];
                        if (counter_sum[COUNTER_W]) begin
                            event_hi_q[counter_index] <= {COUNTER_W{1'b1}};
                            overflow_q[counter_index] <= 1'b1;
                        end else begin
                            event_hi_q[counter_index] <=
                                counter_sum[COUNTER_W-1:0];
                        end
                    end
                end

                if ((recording_q || sealing_q) &&
                    (projection_probe[6:4] > selector_high_water_q))
                    selector_high_water_q <= projection_probe[6:4];

                if (finish && recording_q) begin
                    recording_q <= 1'b0;
                    sealing_q <= 1'b1;
                    outcome_q <= finish_outcome;
                end

                if (sealing_q) begin
                    last_total_q <= live_total_q;
                    sealing_q <= 1'b0;
                    snapshot_valid_q <= 1'b1;
                end
            end
        end
    end

    function [31:0] read_counter;
        input [6:0] metric_id;
        begin
            case (metric_id)
                7'd1:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[1]};
                7'd2:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[2]};
                7'd3:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[3]};
                7'd4:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[4]};
                7'd5:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[5]};
                7'd6:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[6]};
                7'd7:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[7]};
                7'd8:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[8]};
                7'd9:  read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[9]};
                7'd10: read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[10]};
                7'd11: read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[11]};
                7'd12: read_counter = {{(32-COUNTER_W){1'b0}}, cycle_counters_q[12]};
                7'd13: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[13]};
                7'd14: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[14]};
                7'd15: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[15]};
                7'd16: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[16]};
                7'd17: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[17]};
                7'd18: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[18]};
                7'd19: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[19]};
                7'd20: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[20]};
                7'd21: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[21]};
                7'd22: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[22]};
                7'd23: read_counter = {{(32-CALL_COUNTER_W){1'b0}}, call_counters_q[23]};
                7'd24: read_counter = {{(32-COUNTER_W){1'b0}}, event_lo_q[24]};
                7'd25: read_counter = {{(32-COUNTER_W){1'b0}}, event_lo_q[25]};
                7'd26: read_counter = {{(32-COUNTER_W){1'b0}}, event_lo_q[26]};
                7'd27: read_counter = {{(32-COUNTER_W){1'b0}}, event_lo_q[27]};
                7'd28: read_counter = {29'd0, selector_high_water_q};
                7'd29: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[29]};
                7'd30: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[30]};
                7'd31: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[31]};
                7'd32: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[32]};
                7'd33: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[33]};
                7'd34: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[34]};
                7'd35: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[35]};
                7'd36: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[36]};
                7'd37: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[37]};
                7'd38: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[38]};
                7'd39: read_counter = {{(32-COUNTER_W){1'b0}}, event_hi_q[39]};
                default: read_counter = 32'd0;
            endcase
        end
    endfunction

    always @* begin
        read_data_lo = 32'd0;
        read_data_hi = 32'd0;
        if (read_index == 7'd0) begin
            read_data_lo = last_total_q[31:0];
            read_data_hi = last_total_q[63:32];
        end else if (read_index < METRIC_COUNT)
            read_data_lo = read_counter(read_index);
    end

    assign schema = METRICS_SCHEMA;
    assign capabilities = METRICS_CAPABILITIES;
    assign status = {25'd0, outcome_q, |overflow_q, 1'b1, sealing_q,
                     recording_q, snapshot_valid_q};
    assign snapshot_tag = tag_q;
    assign overflow0 = overflow_q[31:0];
    assign overflow1 = {24'd0, overflow_q[39:32]};
    assign overflow2 = 32'd0;
    assign overflow3 = 32'd0;
    assign total_cycles = last_total_q;
    assign recording = recording_q;
    assign sealing = sealing_q;
    assign snapshot_valid = snapshot_valid_q;

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    wire [63:0] stage_total =
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[1]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[2]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[3]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[4]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[5]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[6]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[7]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[8]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[9]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[10]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[11]} +
        {{(64-COUNTER_W){1'b0}}, cycle_counters_q[12]};
    integer formal_index;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(recording_q && sealing_q));
            assert(!(recording_q && snapshot_valid_q));
            assert(!(sealing_q && snapshot_valid_q));
            if (snapshot_valid_q && !(|overflow_q[12:1]))
                assert(last_total_q == stage_total);
            if (core_stage_active)
                assert((core_stage == 5'd0) || (core_stage == 5'd1) ||
                       (core_stage == 5'd2) || (core_stage == 5'd4) ||
                       (core_stage == 5'd5) || (core_stage == 5'd6) ||
                       (core_stage == 5'd8) || (core_stage == 5'd9) ||
                       (core_stage == 5'd11) || (core_stage == 5'd13) ||
                       (core_stage == 5'd14));
            if (core_stage_call)
                assert((core_stage_call_id == 5'd0) ||
                       (core_stage_call_id == 5'd1) ||
                       (core_stage_call_id == 5'd2) ||
                       (core_stage_call_id == 5'd4) ||
                       (core_stage_call_id == 5'd5) ||
                       (core_stage_call_id == 5'd6) ||
                       (core_stage_call_id == 5'd8) ||
                       (core_stage_call_id == 5'd9) ||
                       (core_stage_call_id == 5'd11) ||
                       (core_stage_call_id == 5'd13) ||
                       (core_stage_call_id == 5'd14));
        end
        if (f_past_valid && rst_n && $past(rst_n)) begin
            if ($past(snapshot_valid_q && !acknowledge)) begin
                assert(snapshot_valid_q);
                assert($stable(last_total_q));
                assert($stable(tag_q));
                assert($stable(outcome_q));
                assert($stable(overflow_q));
                for (formal_index = 1; formal_index <= 12;
                     formal_index = formal_index + 1)
                    assert($stable(cycle_counters_q[formal_index]));
                for (formal_index = 13; formal_index <= 23;
                     formal_index = formal_index + 1)
                    assert($stable(call_counters_q[formal_index]));
                for (formal_index = 24; formal_index <= 27;
                     formal_index = formal_index + 1)
                    assert($stable(event_lo_q[formal_index]));
                assert($stable(selector_high_water_q));
                for (formal_index = 29; formal_index < METRIC_COUNT;
                     formal_index = formal_index + 1)
                    assert($stable(event_hi_q[formal_index]));
            end
            if ($past(finish && recording_q))
                assert(sealing_q);
            if ($past(sealing_q))
                assert(snapshot_valid_q);
        end
    end
`endif
endmodule

`default_nettype wire
