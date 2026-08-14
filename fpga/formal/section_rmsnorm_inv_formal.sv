`default_nettype none

module section_rmsnorm_inv_formal(input wire clk);
    localparam [3:0] SC_CLEAN       = 4'd0;
    localparam [3:0] SC_BAD_CFG     = 4'd1;
    localparam [3:0] SC_BAD_TOKEN   = 4'd2;
    localparam [3:0] SC_BAD_ROWS    = 4'd3;
    localparam [3:0] SC_BAD_FINAL   = 4'd4;
    localparam [3:0] SC_BAD_RECORD  = 4'd5;
    localparam [3:0] SC_ARITHMETIC  = 4'd6;
    localparam [3:0] SC_ABORT       = 4'd7;

    localparam [3:0] STATUS_BAD_CFG    = 4'b0001;
    localparam [3:0] STATUS_FRAME      = 4'b0010;
    localparam [3:0] STATUS_ARITHMETIC = 4'b0100;

    localparam [3:0] DUT_ST_INPUT     = 4'd1;
    localparam [3:0] DUT_ST_ADD_WAIT  = 4'd6;
    localparam [3:0] DUT_ST_SQ_WAIT   = 4'd8;
    localparam [3:0] DUT_ST_CORR_WAIT = 4'd12;
    localparam [3:0] DUT_ST_RESULT    = 4'd15;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg input_allow_any;
    (* anyseq *) reg result_ready_any;
    (* anyconst *) reg [3:0] scenario;
    (* anyconst *) reg [2:0] abort_phase;

`ifdef FORMAL_BMC
    wire [3:0] active_scenario = SC_CLEAN;
`else
    wire [3:0] active_scenario = scenario;
`endif

    reg cfg_pending_q;
    reg [2:0] sent_q;
    reg [2:0] accepted_q;
    reg aborted_q;
    reg restarted_q;
    reg f_past_valid = 1'b0;

    wire initial_run = !aborted_q && !restarted_q;
    wire effective_clean = restarted_q || (active_scenario == SC_CLEAN);
    wire cfg_valid = cfg_pending_q && rst_n;
    wire cfg_ready;
    wire [13:0] cfg_rows = (initial_run &&
        (active_scenario == SC_BAD_CFG)) ? 14'd24 : 14'd8;
    wire [2:0] cfg_tokens = 3'd2;
    wire [31:0] cfg_eps = 32'h3586_37bd;

    wire busy;
    wire done;
    wire error;
    wire [3:0] status;
    wire input_ready;
    wire result_valid;
    wire [1:0] result_token;
    wire [31:0] result_inv_rms;
    wire result_final;
    wire [3:0] formal_state;

`ifdef FORMAL_BMC
    wire input_allow = input_allow_any;
    wire result_ready = result_ready_any;
`else
    wire input_allow = 1'b1;
    wire result_ready = 1'b1;
`endif

    wire [1:0] clean_token = sent_q[1:0];
    wire [1:0] input_token = initial_run &&
        (active_scenario == SC_BAD_TOKEN) ? 2'd1 : clean_token;
    wire [13:0] input_rows = initial_run &&
        (active_scenario == SC_BAD_ROWS) ? 14'd16 : 14'd8;
    wire input_final = initial_run &&
        (active_scenario == SC_BAD_FINAL) ? 1'b1 : (sent_q == 3'd1);
    wire [7:0] input_max_exp = initial_run &&
        (active_scenario == SC_BAD_RECORD) ? 8'd0 :
        initial_run && (active_scenario == SC_ARITHMETIC) ? 8'd254 :
        (sent_q == 0 ? 8'd127 : 8'd128);
    wire [47:0] input_sum_sq = initial_run &&
        (active_scenario == SC_BAD_RECORD) ? (48'd1 << 37) :
        initial_run && (active_scenario == SC_ARITHMETIC) ? 48'hffff_ffff_ffff :
        (48'd1 << 37);
    wire input_valid = busy && (sent_q < 3'd2) && input_allow;

    wire cfg_fire = cfg_valid && cfg_ready;
    wire input_fire = input_valid && input_ready;
    wire result_fire = result_valid && result_ready;

    section_rmsnorm_inv dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_valid(cfg_valid), .cfg_ready(cfg_ready),
        .cfg_rows(cfg_rows), .cfg_tokens(cfg_tokens), .cfg_eps(cfg_eps),
        .abort_run(abort_run), .busy(busy), .done(done),
        .error(error), .status(status),
        .s_valid(input_valid), .s_ready(input_ready),
        .s_token(input_token), .s_max_exp(input_max_exp),
        .s_sum_sq(input_sum_sq), .s_rows(input_rows), .s_final(input_final),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_token(result_token), .result_inv_rms(result_inv_rms),
        .result_final(result_final), .formal_state(formal_state)
    );

    wire abort_point =
        (abort_phase == 0) ? (formal_state == DUT_ST_INPUT) :
        (abort_phase == 1) ? (formal_state == DUT_ST_ADD_WAIT) :
        (abort_phase == 2) ? (formal_state == DUT_ST_SQ_WAIT) :
        (abort_phase == 3) ? (formal_state == DUT_ST_CORR_WAIT) :
                             (formal_state == DUT_ST_RESULT);
`ifdef FORMAL_COVER
    wire abort_run = initial_run && (active_scenario == SC_ABORT) && abort_point;
`else
    wire abort_run = 1'b0;
`endif

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

`ifdef FORMAL_FAULTS
        assume(scenario >= SC_BAD_CFG && scenario <= SC_ARITHMETIC);
`elsif FORMAL_COVER
        assume(scenario <= SC_ABORT);
        assume(abort_phase <= 3'd4);
`else
        assume(scenario == SC_CLEAN);
`endif

        // A presented record and a stalled result remain owned by their source.
        if (f_past_valid && rst_n && busy &&
            $past(rst_n && busy && !abort_run && input_valid && !input_ready))
            assume(input_allow);

        if (!rst_n) begin
            cfg_pending_q <= 1'b1;
            sent_q <= 3'd0;
            accepted_q <= 3'd0;
            aborted_q <= 1'b0;
            restarted_q <= 1'b0;
        end else if (abort_run) begin
            cfg_pending_q <= 1'b1;
            sent_q <= 3'd0;
            accepted_q <= 3'd0;
            aborted_q <= 1'b1;
        end else begin
            if (cfg_fire) begin
                cfg_pending_q <= 1'b0;
                sent_q <= 3'd0;
                accepted_q <= 3'd0;
                if (aborted_q)
                    restarted_q <= 1'b1;
            end
            if (input_fire)
                sent_q <= sent_q + 1'b1;
            if (result_fire)
                accepted_q <= accepted_q + 1'b1;
        end

        if (rst_n) begin
            assert(sent_q <= 3'd2);
            assert(accepted_q <= sent_q);
            assert(!(done && busy));
            assert(!status[3]);
            if (result_valid) begin
                assert(result_token == accepted_q[1:0]);
                assert(result_inv_rms == 32'h3f80_0000);
                assert(result_final == (accepted_q == 3'd1));
            end

            if (done && !error) begin
                assert(effective_clean);
                assert(status == 4'd0);
                assert(sent_q == 3'd2);
                assert(accepted_q == 3'd2);
            end
            if (done && error) begin
                assert(initial_run);
                assert(accepted_q == 3'd0);
                case (active_scenario)
                    SC_BAD_CFG: begin
                        assert(status == STATUS_BAD_CFG);
                        assert(sent_q == 3'd0);
                    end
                    SC_BAD_TOKEN: begin
                        assert(status == STATUS_FRAME);
                        assert(sent_q == 3'd1);
                    end
                    SC_BAD_ROWS: begin
                        assert(status == STATUS_FRAME);
                        assert(sent_q == 3'd1);
                    end
                    SC_BAD_FINAL: begin
                        assert(status == STATUS_FRAME);
                        assert(sent_q == 3'd1);
                    end
                    SC_BAD_RECORD: begin
                        assert(status == STATUS_FRAME);
                        assert(sent_q == 3'd1);
                    end
                    SC_ARITHMETIC: begin
                        assert(status == STATUS_ARITHMETIC);
                        assert(sent_q == 3'd1);
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
            assert(!result_valid);
        end

`ifdef FORMAL_COVER
        cover(rst_n && (active_scenario == SC_CLEAN) && done && !error);
        cover(rst_n && (active_scenario == SC_BAD_CFG) && done && error &&
              status == STATUS_BAD_CFG);
        cover(rst_n && (active_scenario == SC_BAD_TOKEN) && done && error &&
              status == STATUS_FRAME);
        cover(rst_n && (active_scenario == SC_BAD_ROWS) && done && error &&
              status == STATUS_FRAME);
        cover(rst_n && (active_scenario == SC_BAD_FINAL) && done && error &&
              status == STATUS_FRAME);
        cover(rst_n && (active_scenario == SC_BAD_RECORD) && done && error &&
              status == STATUS_FRAME);
        cover(rst_n && (active_scenario == SC_ARITHMETIC) && done && error &&
              status == STATUS_ARITHMETIC);
        cover(rst_n && (active_scenario == SC_ABORT) && aborted_q &&
              restarted_q && done && !error && abort_phase == 0);
        cover(rst_n && (active_scenario == SC_ABORT) && aborted_q &&
              restarted_q && done && !error && abort_phase == 1);
        cover(rst_n && (active_scenario == SC_ABORT) && aborted_q &&
              restarted_q && done && !error && abort_phase == 2);
        cover(rst_n && (active_scenario == SC_ABORT) && aborted_q &&
              restarted_q && done && !error && abort_phase == 3);
        cover(rst_n && (active_scenario == SC_ABORT) && aborted_q &&
              restarted_q && done && !error && abort_phase == 4);
`endif
    end
endmodule

`default_nettype wire
