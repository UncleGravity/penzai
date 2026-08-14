`default_nettype none

// Rows=8 retains all eight exponent lanes while making each accepted group a
// complete token. Four tokens exercise the production terminal-width boundary.
module section_rmsnorm_maxexp_formal(input wire clk);
    localparam [3:0] SC_CLEAN      = 4'd0;
    localparam [3:0] SC_EARLY_LAST = 4'd1;
    localparam [3:0] SC_LATE_LAST = 4'd2;
    localparam [3:0] SC_SCRATCH   = 4'd3;
    localparam [3:0] SC_NONFINITE = 4'd4;
    localparam [3:0] SC_SUBNORMAL = 4'd5;
    localparam [3:0] SC_BAD_CFG   = 4'd6;
    localparam [3:0] SC_ABORT     = 4'd7;

    localparam [5:0] ST_BAD_CFG   = 6'h01;
    localparam [5:0] ST_NONFINITE = 6'h02;
    localparam [5:0] ST_FRAME     = 6'h04;
    localparam [5:0] ST_SCRATCH   = 6'h08;
    localparam [5:0] ST_SUBNORMAL = 6'h20;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg input_allow;
    (* anyseq *) reg result_ready_any;
    (* anyconst *) reg [3:0] scenario;
    (* anyconst *) reg abort_phase;
    (* anyconst *) reg [255:0] arbitrary_data0;
    (* anyconst *) reg [255:0] arbitrary_data1;
    (* anyconst *) reg [255:0] arbitrary_data2;
    (* anyconst *) reg [255:0] arbitrary_data3;

`ifdef FORMAL_BMC
    wire [3:0] active_scenario = SC_CLEAN;
`else
    wire [3:0] active_scenario = scenario;
`endif

    reg cfg_pending_q;
    reg [2:0] sent_groups_q;
    reg [2:0] accepted_results_q;
    reg aborted_q;
    reg restarted_q;
    reg f_past_valid = 1'b0;

    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [13:0] cfg_rows = active_scenario == SC_BAD_CFG ? 14'd7 : 14'd8;
    wire busy;
    wire done;
    wire error;
    wire [5:0] status;
    wire group_ready;
    wire result_valid;
    wire [1:0] result_token;
    wire [7:0] result_max_exp;
    wire [13:0] result_rows;
    wire result_subnormal_warning;
    wire result_final;

    function automatic [255:0] healthy_group(input [1:0] token);
        reg [255:0] value;
        reg [7:0] exponent;
        integer lane;
        begin
            value = 256'd0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                exponent = 8'd120 + {4'd0, token} + lane[2:0];
                value[lane*32 +: 32] = {
                    token[0], exponent, token, lane[2:0], 18'h2a155
                };
            end
            healthy_group = value;
        end
    endfunction

    function automatic [255:0] arbitrary_group(input [1:0] token);
        case (token)
            2'd0: arbitrary_group = arbitrary_data0;
            2'd1: arbitrary_group = arbitrary_data1;
            2'd2: arbitrary_group = arbitrary_data2;
            default: arbitrary_group = arbitrary_data3;
        endcase
    endfunction

    function automatic [255:0] scenario_group(
        input [3:0] which,
        input [1:0] token
    );
        reg [255:0] value;
        begin
`ifdef FORMAL_BMC
            value = arbitrary_group(token);
`else
            value = healthy_group(token);
`endif
            if ((which == SC_NONFINITE) && (token == 0))
                value[2*32 +: 32] = 32'h7fc0_0001;
            if ((which == SC_SUBNORMAL) && (token == 0))
                value[2*32 +: 32] = 32'h0000_0001;
            scenario_group = value;
        end
    endfunction

    function automatic [7:0] expected_max(input [255:0] value);
        reg [7:0] maximum;
        reg [7:0] exponent;
        integer lane;
        begin
            maximum = 8'd0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                exponent = value[lane*32 + 23 +: 8];
                if ((exponent != 8'hff) && (exponent > maximum))
                    maximum = exponent;
            end
            expected_max = maximum;
        end
    endfunction

    function automatic has_nonfinite(input [255:0] value);
        reg found;
        integer lane;
        begin
            found = 1'b0;
            for (lane = 0; lane < 8; lane = lane + 1)
                if (value[lane*32 + 23 +: 8] == 8'hff)
                    found = 1'b1;
            has_nonfinite = found;
        end
    endfunction

    function automatic has_subnormal(input [255:0] value);
        reg found;
        reg [31:0] bits;
        integer lane;
        begin
            found = 1'b0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                bits = value[lane*32 +: 32];
                if ((bits[30:23] == 8'd0) && (bits[22:0] != 23'd0))
                    found = 1'b1;
            end
            has_subnormal = found;
        end
    endfunction

`ifdef FORMAL_BMC
    wire clean_has_subnormal = has_subnormal(arbitrary_data0) ||
                               has_subnormal(arbitrary_data1) ||
                               has_subnormal(arbitrary_data2) ||
                               has_subnormal(arbitrary_data3);
`else
    wire clean_has_subnormal =
        has_subnormal(scenario_group(SC_CLEAN, 2'd0)) ||
        has_subnormal(scenario_group(SC_CLEAN, 2'd1)) ||
        has_subnormal(scenario_group(SC_CLEAN, 2'd2)) ||
        has_subnormal(scenario_group(SC_CLEAN, 2'd3));
`endif

    wire [255:0] group_data = scenario_group(
        active_scenario, sent_groups_q[1:0]
    );
`ifdef FORMAL_BMC
    wire group_valid = busy && (sent_groups_q < 4) && input_allow;
    wire result_ready = result_ready_any;
`else
    wire group_valid = busy && (sent_groups_q < 4);
    wire result_ready = 1'b1;
`endif
    wire group_last = (active_scenario == SC_EARLY_LAST) ?
                      (sent_groups_q == 0) :
                      (active_scenario == SC_LATE_LAST) ? 1'b0 :
                      (sent_groups_q == 3);
    wire group_error = (active_scenario == SC_SCRATCH) &&
                       (sent_groups_q == 0);

    wire abort_point = abort_phase ?
                       (busy && sent_groups_q == 1 &&
                        accepted_results_q == 0) :
                       (busy && sent_groups_q == 0);
`ifdef FORMAL_COVER
    wire abort_run = (active_scenario == SC_ABORT) && !aborted_q && abort_point;
`else
    wire abort_run = 1'b0;
`endif

    wire cfg_fire = cfg_valid && cfg_ready;
    wire group_fire = group_valid && group_ready;
    wire result_fire = result_valid && result_ready;

    section_rmsnorm_maxexp dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(3'd4),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_group_data(group_data), .s_group_error(group_error),
        .s_group_valid(group_valid), .s_group_ready(group_ready),
        .s_group_last(group_last),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_max_exp(result_max_exp),
        .result_rows(result_rows),
        .result_subnormal_warning(result_subnormal_warning),
        .result_final(result_final)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

`ifdef FORMAL_FAULTS
        assume(scenario >= SC_EARLY_LAST && scenario <= SC_BAD_CFG);
`elsif FORMAL_COVER
        assume(scenario <= SC_ABORT);
`else
        assume(scenario == SC_CLEAN);
`endif

`ifdef FORMAL_BMC
        assume(!has_nonfinite(arbitrary_data0));
        assume(!has_nonfinite(arbitrary_data1));
        assume(!has_nonfinite(arbitrary_data2));
        assume(!has_nonfinite(arbitrary_data3));
`endif

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            sent_groups_q <= 3'd0;
            accepted_results_q <= 3'd0;
            aborted_q <= 1'b0;
            restarted_q <= 1'b0;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                sent_groups_q <= 3'd0;
                accepted_results_q <= 3'd0;
                if (aborted_q) restarted_q <= 1'b1;
            end
            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                sent_groups_q <= 3'd0;
                accepted_results_q <= 3'd0;
                aborted_q <= 1'b1;
            end else begin
                if (group_fire)
                    sent_groups_q <= sent_groups_q + 1'b1;
                if (result_fire)
                    accepted_results_q <= accepted_results_q + 1'b1;
            end
        end

        if (rst_n) begin
            assert(accepted_results_q <= 4);
            assert(!status[4]);
            if (result_valid) begin
                assert(result_token == accepted_results_q[1:0]);
                assert(result_max_exp == expected_max(
                    scenario_group(active_scenario, result_token)
                ));
                assert(result_rows == 8);
                assert(result_subnormal_warning == has_subnormal(
                    scenario_group(active_scenario, result_token)
                ));
                assert(result_final == (result_token == 3));
            end
            if (done) begin
                if ((active_scenario == SC_CLEAN) ||
                    (active_scenario == SC_SUBNORMAL) ||
                    (active_scenario == SC_ABORT))
                    assert(!error);
                else
                    assert(error);
            end
            if (done && !error) begin
                assert(accepted_results_q == 4);
                if (active_scenario == SC_SUBNORMAL)
                    assert(status == ST_SUBNORMAL);
                if (active_scenario == SC_CLEAN)
                    assert(status == (clean_has_subnormal ?
                                      ST_SUBNORMAL : 6'd0));
            end
            if (done && error) begin
                assert(!result_valid);
                if ((active_scenario == SC_EARLY_LAST) ||
                    (active_scenario == SC_LATE_LAST))
                    assert(status == ST_FRAME);
                if (active_scenario == SC_SCRATCH)
                    assert(status == ST_SCRATCH);
                if (active_scenario == SC_NONFINITE)
                    assert(status == ST_NONFINITE);
                if (active_scenario == SC_BAD_CFG)
                    assert(status == ST_BAD_CFG);

                if ((active_scenario == SC_EARLY_LAST) ||
                    (active_scenario == SC_SCRATCH) ||
                    (active_scenario == SC_NONFINITE) ||
                    (active_scenario == SC_BAD_CFG))
                    assert(accepted_results_q == 0);
                if (active_scenario == SC_LATE_LAST)
                    assert(accepted_results_q == 3);
            end
        end

        if (f_past_valid && rst_n && busy && !abort_run &&
            $past(rst_n && !abort_run && group_valid && !group_ready)) begin
            assume(group_valid);
            assume(group_data == $past(group_data));
            assume(group_last == $past(group_last));
            assume(group_error == $past(group_error));
        end

        cover(rst_n && (active_scenario == SC_CLEAN) && done && !error &&
              accepted_results_q == 4);
        cover(rst_n && (active_scenario == SC_EARLY_LAST) && done && error);
        cover(rst_n && (active_scenario == SC_LATE_LAST) && done && error &&
              accepted_results_q == 3);
        cover(rst_n && (active_scenario == SC_SCRATCH) && done && error);
        cover(rst_n && (active_scenario == SC_NONFINITE) && done && error);
        cover(rst_n && (active_scenario == SC_SUBNORMAL) && done && !error &&
              status == ST_SUBNORMAL);
        cover(rst_n && (active_scenario == SC_BAD_CFG) && done && error);
        cover(rst_n && (active_scenario == SC_ABORT) && !abort_phase &&
              aborted_q && restarted_q && done && !error &&
              accepted_results_q == 4);
        cover(rst_n && (active_scenario == SC_ABORT) && abort_phase &&
              aborted_q && restarted_q && done && !error &&
              accepted_results_q == 4);
    end
endmodule

`default_nettype wire
