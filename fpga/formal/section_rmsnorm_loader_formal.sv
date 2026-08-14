`default_nettype none

// Rows=8 makes each token exactly one four-word scratch/scanner group. Two
// tokens exercise the token stride without making the bounded proof large.
module section_rmsnorm_loader_formal(input wire clk);
    localparam [2:0] SC_CLEAN       = 3'd0;
    localparam [2:0] SC_EARLY_LAST  = 3'd1;
    localparam [2:0] SC_LATE_LAST   = 3'd2;
    localparam [2:0] SC_SINK        = 3'd3;
    localparam [2:0] SC_BAD_CFG     = 3'd4;
    localparam [2:0] SC_ABORT_INPUT = 3'd5;
    localparam [2:0] SC_ABORT_GROUP = 3'd6;

    localparam [3:0] ST_BAD_CFG = 4'h1;
    localparam [3:0] ST_FRAME   = 4'h2;
    localparam [3:0] ST_SINK    = 4'h4;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg source_allow_any;
    (* anyseq *) reg wr_ready_any;
    (* anyseq *) reg group_ready_any;
    (* anyconst *) reg [2:0] scenario;

`ifdef FORMAL_BMC
    wire [2:0] active_scenario = SC_CLEAN;
`else
    wire [2:0] active_scenario = scenario;
`endif

    reg cfg_pending_q;
    reg [3:0] sent_words_q;
    reg [3:0] accepted_writes_q;
    reg [1:0] accepted_groups_q;
    reg aborted_q;
    reg restarted_q;
    reg f_past_valid = 1'b0;

    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [13:0] cfg_rows =
        ((active_scenario == SC_BAD_CFG) && !restarted_q) ? 14'd7 : 14'd8;
    wire [2:0] cfg_tokens = 3'd2;

    wire busy;
    wire done;
    wire error;
    wire [3:0] status;
    wire input_ready;
    wire wr_valid;
    wire [1:0] wr_bank;
    wire [13:0] wr_address;
    wire [63:0] wr_data;
    wire group_valid;
    wire [255:0] group_data;
    wire group_last;

    wire initial_run = !aborted_q && !restarted_q;
    wire effective_clean = restarted_q || (active_scenario == SC_CLEAN);

`ifdef FORMAL_BMC
    wire source_allow = source_allow_any;
    wire wr_ready = wr_ready_any;
    wire group_ready = group_ready_any;
`else
    wire source_allow = 1'b1;
    wire wr_ready = 1'b1;
    wire group_ready = !((active_scenario == SC_ABORT_GROUP) && initial_run);
`endif

    wire abort_input_point = busy && (sent_words_q == 4'd2);
    wire abort_group_point = busy && (sent_words_q == 4'd4);
`ifdef FORMAL_COVER
    wire abort_run = initial_run &&
        (((active_scenario == SC_ABORT_INPUT) && abort_input_point) ||
         ((active_scenario == SC_ABORT_GROUP) && abort_group_point));
`else
    wire abort_run = 1'b0;
`endif

    function automatic [63:0] word_value(input [3:0] ordinal);
        word_value = {
            24'hc35a71, ordinal, 4'hd,
            24'h3ca58e, ordinal, 4'h2
        };
    endfunction

    function automatic [255:0] group_value(input [0:0] token);
        reg [3:0] base;
        begin
            base = token ? 4'd4 : 4'd0;
            group_value = {
                word_value(base + 4'd3), word_value(base + 4'd2),
                word_value(base + 4'd1), word_value(base)
            };
        end
    endfunction

    wire input_valid = busy && (sent_words_q < 4'd8) && source_allow;
    wire [63:0] input_data = word_value(sent_words_q);
    wire input_last = (active_scenario == SC_EARLY_LAST) && initial_run ?
                      (sent_words_q == 4'd0) :
                      (active_scenario == SC_LATE_LAST) && initial_run ?
                      1'b0 : (sent_words_q == 4'd7);
    wire wr_error = (active_scenario == SC_SINK) && initial_run &&
                    (sent_words_q == 4'd1);

    wire cfg_fire = cfg_valid && cfg_ready;
    wire input_fire = input_valid && input_ready;
    wire wr_fire = wr_valid && wr_ready;
    wire group_fire = group_valid && group_ready;

    section_rmsnorm_loader dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_axis_tdata(input_data), .s_axis_tkeep(8'hff),
        .s_axis_tvalid(input_valid), .s_axis_tready(input_ready),
        .s_axis_tlast(input_last),
        .wr_valid(wr_valid), .wr_ready(wr_ready), .wr_error(wr_error),
        .wr_bank(wr_bank), .wr_address(wr_address), .wr_data(wr_data),
        .group_valid(group_valid), .group_ready(group_ready),
        .group_data(group_data), .group_last(group_last)
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
        assume(scenario <= SC_ABORT_GROUP);
`else
        assume(scenario == SC_CLEAN);
`endif

        // A source bubble may not retract a word that was already presented.
        if (f_past_valid && rst_n && busy &&
            $past(rst_n && busy && !abort_run && input_valid && !input_ready))
            assume(source_allow);

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            sent_words_q <= 4'd0;
            accepted_writes_q <= 4'd0;
            accepted_groups_q <= 2'd0;
            aborted_q <= 1'b0;
            restarted_q <= 1'b0;
        end else begin
            if (abort_run) begin
                cfg_pending_q <= 1'b1;
                sent_words_q <= 4'd0;
                accepted_writes_q <= 4'd0;
                accepted_groups_q <= 2'd0;
                aborted_q <= 1'b1;
            end else begin
                if (cfg_fire) begin
                    cfg_pending_q <= 1'b0;
                    sent_words_q <= 4'd0;
                    accepted_writes_q <= 4'd0;
                    accepted_groups_q <= 2'd0;
                    if (aborted_q)
                        restarted_q <= 1'b1;
                end
                if (input_fire)
                    sent_words_q <= sent_words_q + 1'b1;
                if (wr_fire)
                    accepted_writes_q <= accepted_writes_q + 1'b1;
                if (group_fire)
                    accepted_groups_q <= accepted_groups_q + 1'b1;
            end
        end

        if (rst_n) begin
            assert(input_fire == wr_fire);
            assert(sent_words_q == accepted_writes_q);
            assert(sent_words_q <= 4'd8);
            assert(accepted_groups_q <= 2'd2);
            assert(!(done && busy));
            assert(!status[3]);

            if (abort_run && (active_scenario == SC_ABORT_GROUP))
                assert(accepted_groups_q == 2'd0);

            if (wr_valid) begin
                assert(wr_bank == sent_words_q[1:0]);
                assert(wr_address == (sent_words_q < 4'd4 ? 14'd0 : 14'd512));
                assert(wr_data == word_value(sent_words_q));
            end

            if (group_valid) begin
                assert(group_data == group_value(accepted_groups_q[0]));
                assert(group_last == (accepted_groups_q == 2'd1));
            end

            if (done && !error) begin
                assert(effective_clean);
                assert(status == 4'd0);
                assert(sent_words_q == 4'd8);
                assert(accepted_writes_q == 4'd8);
                assert(accepted_groups_q == 2'd2);
            end

            if (done && error) begin
                assert(initial_run);
                case (active_scenario)
                    SC_EARLY_LAST: begin
                        assert(status == ST_FRAME);
                        assert(sent_words_q == 4'd1);
                        assert(accepted_groups_q == 2'd0);
                    end
                    SC_LATE_LAST: begin
                        assert(status == ST_FRAME);
                        assert(sent_words_q == 4'd8);
                        assert(accepted_groups_q == 2'd1);
                    end
                    SC_SINK: begin
                        assert(status == ST_SINK);
                        assert(sent_words_q == 4'd2);
                        assert(accepted_groups_q == 2'd0);
                    end
                    SC_BAD_CFG: begin
                        assert(status == ST_BAD_CFG);
                        assert(sent_words_q == 4'd0);
                        assert(accepted_groups_q == 2'd0);
                    end
                    default: assert(1'b0);
                endcase
            end
        end

        if (f_past_valid && rst_n && $past(rst_n && abort_run)) begin
            assert(!busy);
            assert(!done);
            assert(!error);
            assert(status == 4'd0);
            assert(!group_valid);
        end

`ifdef FORMAL_COVER
        cover(rst_n && (active_scenario == SC_CLEAN) && done && !error);
        cover(rst_n && (active_scenario == SC_EARLY_LAST) && done && error &&
              (status == ST_FRAME));
        cover(rst_n && (active_scenario == SC_LATE_LAST) && done && error &&
              (status == ST_FRAME));
        cover(rst_n && (active_scenario == SC_SINK) && done && error &&
              (status == ST_SINK));
        cover(rst_n && (active_scenario == SC_BAD_CFG) && done && error &&
              (status == ST_BAD_CFG));
        cover(rst_n && (active_scenario == SC_ABORT_INPUT) && aborted_q &&
              restarted_q && done && !error);
        cover(rst_n && (active_scenario == SC_ABORT_GROUP) && aborted_q &&
              restarted_q && done && !error);
`endif
    end
endmodule

`default_nettype wire
