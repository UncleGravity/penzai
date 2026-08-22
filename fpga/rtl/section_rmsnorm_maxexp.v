// Section-local RMSNorm maximum-exponent scan.
//
// The residual load stream is token-major in 256-bit groups. This leaf records
// the largest finite normal binary32 exponent field for each token. Zeros and
// subnormals do not contribute; subnormals set a warning that integration must
// reject before architectural publication. Result records are tentative until
// the final record handshakes and done pulses.

`default_nettype none

module section_rmsnorm_maxexp (
    input  wire          clk,
    input  wire          rst_n,

    input  wire          cfg_valid,
    output wire          cfg_ready,
    input  wire [13:0]   cfg_rows,
    input  wire [2:0]    cfg_tokens,

    input  wire          abort_run,
    output wire          busy,
    output reg           done,
    output reg           error,
    // bit 0 BAD_CFG, 1 NONFINITE, 2 FRAME, 3 SCRATCH,
    // bit 4 INTERNAL, 5 SUBNORMAL_WARNING.
    output reg  [5:0]    status,

    input  wire [255:0]  s_group_data,
    input  wire          s_group_error,
    input  wire          s_group_valid,
    output wire          s_group_ready,
    input  wire          s_group_last,

    output wire          result_valid,
    input  wire          result_ready,
    output wire [1:0]    result_token,
    output reg  [7:0]    result_max_exp,
    output wire [13:0]   result_rows,
    output reg           result_subnormal_warning,
    output wire          result_final
`ifdef VERILATOR
    , input wire         sim_force_summary_fatal
`endif
);
    localparam [5:0] STATUS_BAD_CFG           = 6'b000001;
    localparam [5:0] STATUS_NONFINITE         = 6'b000010;
    localparam [5:0] STATUS_FRAME             = 6'b000100;
    localparam [5:0] STATUS_SCRATCH           = 6'b001000;
    localparam [5:0] STATUS_INTERNAL          = 6'b010000;
    localparam [5:0] STATUS_SUBNORMAL_WARNING = 6'b100000;

    localparam [1:0] ST_IDLE   = 2'd0;
    localparam [1:0] ST_INPUT  = 2'd1;
    localparam [1:0] ST_DRAIN  = 2'd2;
    localparam [1:0] ST_RESULT = 2'd3;

    reg [1:0] state_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [9:0] run_groups_q;
    reg [1:0] token_q;
    reg [9:0] group_q;
    reg [7:0] token_max_exp_q;
    reg token_subnormal_q;
    reg tree_valid_q;
    reg [7:0] tree_max03_q;
    reg [7:0] tree_max47_q;
    reg tree_nonfinite_q;
    reg tree_subnormal_q;
    reg tree_frame_bad_q;
    reg tree_scratch_q;
    reg summary_valid_q;
    reg [7:0] summary_max_exp_q;
    reg summary_nonfinite_q;
    reg summary_subnormal_q;
    reg summary_frame_bad_q;
    reg summary_scratch_q;

    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) &&
                        (cfg_rows[2:0] == 3'b000) &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4);
    wire cfg_accept = cfg_valid && cfg_ready;
    wire summary_nonfinite_live = summary_nonfinite_q
`ifdef VERILATOR
                                  || sim_force_summary_fatal
`endif
                                  ;
    wire summary_fatal = summary_valid_q &&
                         (summary_scratch_q || summary_frame_bad_q ||
                          summary_nonfinite_live);
    wire tree_fatal = tree_valid_q &&
                      (tree_scratch_q || tree_frame_bad_q ||
                       tree_nonfinite_q);

    assign cfg_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign busy = state_q != ST_IDLE;
    // A registered fatal entry already makes the run irrevocable. Keep READY
    // local to this input stage; the priority branches below discard any
    // younger beat presented while the fatal entry retires.
    assign s_group_ready = rst_n && !abort_run && (state_q == ST_INPUT);
    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_token = token_q;
    assign result_rows = run_rows_q;
    assign result_final = ({1'b0, token_q} + 3'd1) == run_tokens_q;

    wire group_accept = s_group_valid && s_group_ready;
    wire group_final = group_q + 10'd1 == run_groups_q;
    wire expected_group_last = result_final && group_final;
    wire group_frame_bad = s_group_last != expected_group_last;

    wire [7:0] exp0 = s_group_data[30:23];
    wire [7:0] exp1 = s_group_data[62:55];
    wire [7:0] exp2 = s_group_data[94:87];
    wire [7:0] exp3 = s_group_data[126:119];
    wire [7:0] exp4 = s_group_data[158:151];
    wire [7:0] exp5 = s_group_data[190:183];
    wire [7:0] exp6 = s_group_data[222:215];
    wire [7:0] exp7 = s_group_data[254:247];

    wire [22:0] mant0 = s_group_data[22:0];
    wire [22:0] mant1 = s_group_data[54:32];
    wire [22:0] mant2 = s_group_data[86:64];
    wire [22:0] mant3 = s_group_data[118:96];
    wire [22:0] mant4 = s_group_data[150:128];
    wire [22:0] mant5 = s_group_data[182:160];
    wire [22:0] mant6 = s_group_data[214:192];
    wire [22:0] mant7 = s_group_data[246:224];

    function automatic [7:0] max2(input [7:0] a, input [7:0] b);
        max2 = a > b ? a : b;
    endfunction

    wire [7:0] max01 = max2(exp0, exp1);
    wire [7:0] max23 = max2(exp2, exp3);
    wire [7:0] max45 = max2(exp4, exp5);
    wire [7:0] max67 = max2(exp6, exp7);
    wire [7:0] max03 = max2(max01, max23);
    wire [7:0] max47 = max2(max45, max67);
    // Split the three-comparator group tree after its second level. The tree
    // and summary stages advance together, retaining one accepted group per
    // cycle while keeping the loader boundary to two comparator levels.
    wire [7:0] tree_max_exp = max2(tree_max03_q, tree_max47_q);
    wire [7:0] next_max_exp = max2(token_max_exp_q, summary_max_exp_q);

    wire group_nonfinite = (exp0 == 8'hff) || (exp1 == 8'hff) ||
                           (exp2 == 8'hff) || (exp3 == 8'hff) ||
                           (exp4 == 8'hff) || (exp5 == 8'hff) ||
                           (exp6 == 8'hff) || (exp7 == 8'hff);
    wire group_subnormal = ((exp0 == 8'd0) && (mant0 != 23'd0)) ||
                           ((exp1 == 8'd0) && (mant1 != 23'd0)) ||
                           ((exp2 == 8'd0) && (mant2 != 23'd0)) ||
                           ((exp3 == 8'd0) && (mant3 != 23'd0)) ||
                           ((exp4 == 8'd0) && (mant4 != 23'd0)) ||
                           ((exp5 == 8'd0) && (mant5 != 23'd0)) ||
                           ((exp6 == 8'd0) && (mant6 != 23'd0)) ||
                           ((exp7 == 8'd0) && (mant7 != 23'd0));

    task automatic fail_run(input [5:0] failure);
        begin
            state_q <= ST_IDLE;
            done <= 1'b1;
            error <= 1'b1;
            status <= status | failure;
            token_max_exp_q <= 8'd0;
            token_subnormal_q <= 1'b0;
            tree_valid_q <= 1'b0;
            summary_valid_q <= 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_groups_q <= 10'd0;
            token_q <= 2'd0;
            group_q <= 10'd0;
            token_max_exp_q <= 8'd0;
            token_subnormal_q <= 1'b0;
            tree_valid_q <= 1'b0;
            tree_max03_q <= 8'd0;
            tree_max47_q <= 8'd0;
            tree_nonfinite_q <= 1'b0;
            tree_subnormal_q <= 1'b0;
            tree_frame_bad_q <= 1'b0;
            tree_scratch_q <= 1'b0;
            summary_valid_q <= 1'b0;
            summary_max_exp_q <= 8'd0;
            summary_nonfinite_q <= 1'b0;
            summary_subnormal_q <= 1'b0;
            summary_frame_bad_q <= 1'b0;
            summary_scratch_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 6'd0;
            result_max_exp <= 8'd0;
            result_subnormal_warning <= 1'b0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            token_q <= 2'd0;
            group_q <= 10'd0;
            token_max_exp_q <= 8'd0;
            token_subnormal_q <= 1'b0;
            tree_valid_q <= 1'b0;
            tree_max03_q <= 8'd0;
            tree_max47_q <= 8'd0;
            tree_nonfinite_q <= 1'b0;
            tree_subnormal_q <= 1'b0;
            tree_frame_bad_q <= 1'b0;
            tree_scratch_q <= 1'b0;
            summary_valid_q <= 1'b0;
            summary_max_exp_q <= 8'd0;
            summary_nonfinite_q <= 1'b0;
            summary_subnormal_q <= 1'b0;
            summary_frame_bad_q <= 1'b0;
            summary_scratch_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 6'd0;
            result_max_exp <= 8'd0;
            result_subnormal_warning <= 1'b0;
        end else begin
            done <= 1'b0;

            case (state_q)
                ST_IDLE: if (cfg_accept) begin
                    error <= 1'b0;
                    status <= 6'd0;
                    token_q <= 2'd0;
                    group_q <= 10'd0;
                    token_max_exp_q <= 8'd0;
                    token_subnormal_q <= 1'b0;
                    tree_valid_q <= 1'b0;
                    tree_max03_q <= 8'd0;
                    tree_max47_q <= 8'd0;
                    tree_nonfinite_q <= 1'b0;
                    tree_subnormal_q <= 1'b0;
                    tree_frame_bad_q <= 1'b0;
                    tree_scratch_q <= 1'b0;
                    summary_valid_q <= 1'b0;
                    summary_max_exp_q <= 8'd0;
                    summary_nonfinite_q <= 1'b0;
                    summary_subnormal_q <= 1'b0;
                    summary_frame_bad_q <= 1'b0;
                    summary_scratch_q <= 1'b0;
                    result_max_exp <= 8'd0;
                    result_subnormal_warning <= 1'b0;
                    if (!cfg_shape_ok) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                        error <= 1'b1;
                        status <= STATUS_BAD_CFG;
                    end else begin
                        run_rows_q <= cfg_rows;
                        run_tokens_q <= cfg_tokens;
                        run_groups_q <= cfg_rows[12:3];
                        state_q <= ST_INPUT;
                    end
                end

                ST_INPUT: begin
                    if (summary_fatal) begin
                        if (summary_scratch_q)
                            fail_run(STATUS_SCRATCH);
                        else if (summary_frame_bad_q)
                            fail_run(STATUS_FRAME);
                        else
                            fail_run(STATUS_NONFINITE);
                    end else if (tree_fatal) begin
                        if (tree_scratch_q)
                            fail_run(STATUS_SCRATCH |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                        else if (tree_frame_bad_q)
                            fail_run(STATUS_FRAME |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                        else
                            fail_run(STATUS_NONFINITE |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                    end else begin
                        tree_valid_q <= group_accept;
                        summary_valid_q <= tree_valid_q;
                        if (summary_valid_q) begin
                            token_max_exp_q <= next_max_exp;
                            if (summary_subnormal_q) begin
                                token_subnormal_q <= 1'b1;
                                status <= status | STATUS_SUBNORMAL_WARNING;
                            end
                        end
                        if (tree_valid_q) begin
                            summary_max_exp_q <= tree_max_exp;
                            summary_nonfinite_q <= tree_nonfinite_q;
                            summary_subnormal_q <= tree_subnormal_q;
                            summary_frame_bad_q <= tree_frame_bad_q;
                            summary_scratch_q <= tree_scratch_q;
                        end
                        if (group_accept) begin
                            tree_max03_q <= max03;
                            tree_max47_q <= max47;
                            tree_nonfinite_q <= group_nonfinite;
                            tree_subnormal_q <= group_subnormal;
                            tree_frame_bad_q <= group_frame_bad;
                            tree_scratch_q <= s_group_error;
                            if (group_final)
                                state_q <= ST_DRAIN;
                            else
                                group_q <= group_q + 1'b1;
                        end
                    end
                end

                ST_DRAIN: begin
                    tree_valid_q <= 1'b0;
                    if (summary_fatal) begin
                        if (summary_scratch_q)
                            fail_run(STATUS_SCRATCH);
                        else if (summary_frame_bad_q)
                            fail_run(STATUS_FRAME);
                        else
                            fail_run(STATUS_NONFINITE);
                    end else if (tree_fatal) begin
                        if (tree_scratch_q)
                            fail_run(STATUS_SCRATCH |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                        else if (tree_frame_bad_q)
                            fail_run(STATUS_FRAME |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                        else
                            fail_run(STATUS_NONFINITE |
                                     (summary_valid_q && summary_subnormal_q ?
                                      STATUS_SUBNORMAL_WARNING : 6'd0));
                    end else if (tree_valid_q) begin
                        summary_valid_q <= 1'b1;
                        summary_max_exp_q <= tree_max_exp;
                        summary_nonfinite_q <= tree_nonfinite_q;
                        summary_subnormal_q <= tree_subnormal_q;
                        summary_frame_bad_q <= tree_frame_bad_q;
                        summary_scratch_q <= tree_scratch_q;
                        if (summary_valid_q) begin
                            token_max_exp_q <= next_max_exp;
                            if (summary_subnormal_q) begin
                                token_subnormal_q <= 1'b1;
                                status <= status | STATUS_SUBNORMAL_WARNING;
                            end
                        end
                    end else if (!summary_valid_q) begin
                        fail_run(STATUS_INTERNAL);
                    end else if (summary_scratch_q) begin
                        fail_run(STATUS_SCRATCH);
                    end else if (summary_frame_bad_q) begin
                        fail_run(STATUS_FRAME);
                    end else if (summary_nonfinite_q) begin
                        fail_run(STATUS_NONFINITE);
                    end else begin
                        summary_valid_q <= 1'b0;
                        result_max_exp <= next_max_exp;
                        result_subnormal_warning <=
                            token_subnormal_q || summary_subnormal_q;
                        if (summary_subnormal_q) begin
                            token_subnormal_q <= 1'b1;
                            status <= status | STATUS_SUBNORMAL_WARNING;
                        end
                        state_q <= ST_RESULT;
                    end
                end

                ST_RESULT: if (result_ready) begin
                    if (result_final) begin
                        state_q <= ST_IDLE;
                        done <= 1'b1;
                    end else begin
                        token_q <= token_q + 1'b1;
                        group_q <= 10'd0;
                        token_max_exp_q <= 8'd0;
                        token_subnormal_q <= 1'b0;
                        tree_valid_q <= 1'b0;
                        summary_valid_q <= 1'b0;
                        state_q <= ST_INPUT;
                    end
                end

                default: fail_run(STATUS_INTERNAL);
            endcase
        end
    end

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    always @* begin
        assert(s_group_ready ==
               (rst_n && !abort_run && (state_q == ST_INPUT)));
    end

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(!(result_valid && error));
            assert(!(s_group_ready && result_valid));
            if (busy) begin
                assert(run_rows_q >= 14'd8 && run_rows_q <= 14'd4096);
                assert(run_rows_q[2:0] == 3'b000);
                assert(run_tokens_q >= 3'd1 && run_tokens_q <= 3'd4);
                assert(token_q < run_tokens_q);
                assert(group_q < run_groups_q);
            end
            if (tree_valid_q || summary_valid_q)
                assert(state_q == ST_INPUT || state_q == ST_DRAIN);
            if (result_valid) begin
                assert(result_token < run_tokens_q);
                assert(result_rows == run_rows_q);
            end
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && (state_q == ST_INPUT) &&
                  (summary_fatal || tree_fatal))) begin
            assert(done && error && !busy);
            assert(!tree_valid_q && !summary_valid_q);
            assert(group_q == $past(group_q));
            assert(token_q == $past(token_q));
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_token == $past(result_token));
            assert(result_max_exp == $past(result_max_exp));
            assert(result_rows == $past(result_rows));
            assert(result_subnormal_warning ==
                   $past(result_subnormal_warning));
            assert(result_final == $past(result_final));
        end
        if (f_past_valid && rst_n && $past(rst_n && abort_run)) begin
            assert(!busy);
            assert(!result_valid);
            assert(!error);
            assert(status == 6'd0);
        end
    end
`endif

endmodule

`default_nettype wire
