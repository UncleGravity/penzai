`default_nettype none

// Focused q8-ingress failure proof: aborting an in-flight raw activation load
// must fail closed, invalidate the resident activation, and finish promptly.
module gemm_activation_abort_formal(input wire clk);
    localparam integer ROWS = 4;
    localparam integer COLS_MAX = 1;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg [ROWS*32-1:0] weight_data;
    (* anyseq *) reg [63:0] act_data;
    reg [3:0] step = 0;
    reg f_past_valid = 1'b0;
    reg abort_seen = 1'b0;

    wire start_kernel = (step == 4'd1);
    wire activation_abort = (step == 4'd3);
    wire busy, kernel_done, weight_ready, act_ready;
    wire [63:0] result_data;
    wire result_valid, result_last;
    wire [7:0] result_keep;
    wire [3:0] dbg_state;
    wire activation_error, activation_valid;
    wire [31:0] loaded_act_epoch;
    wire [15:0] loaded_act_q1_blocks, loaded_act_cols;

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(COLS_MAX),
        .MAX_SUB_INDEX(8), .ACC_W(16)) dut (
        .clk(clk), .rst_n(rst_n), .start_kernel(start_kernel),
        .num_q1_blocks(16'd1), .num_rowblocks(16'd1), .num_rows(32'd4),
        .num_cols(16'd1), .weight_fmt(2'd1),
        .act_mode(2'd2), .act_epoch(32'hAB07_0001), .emin(-8'sd8),
        .activation_abort(activation_abort),
        .kernel_done(kernel_done), .activation_error(activation_error),
        .activation_valid(activation_valid), .loaded_act_epoch(loaded_act_epoch),
        .loaded_act_q1_blocks(loaded_act_q1_blocks), .loaded_act_cols(loaded_act_cols),
        .busy(busy),
        .s_axis_tdata(weight_data), .s_axis_tvalid(1'b1), .s_axis_tready(weight_ready),
        .s_axis_acts_tdata(act_data), .s_axis_acts_tvalid(1'b1),
        .s_axis_acts_tready(act_ready),
        .m_axis_tdata(result_data), .m_axis_tvalid(result_valid), .m_axis_tready(1'b1),
        .m_axis_tlast(result_last), .m_axis_tkeep(result_keep), .dbg_state(dbg_state)
    );

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) begin
            assume(!rst_n);
            step <= 0;
            abort_seen <= 1'b0;
        end else begin
            assume(rst_n);
            if (step != 4'd15) step <= step + 1'b1;
            if (activation_abort && busy) abort_seen <= 1'b1;
        end

        if (rst_n) begin
            // The scenario reaches the raw activation-load state before aborting.
            if (activation_abort) begin
                assert(busy);
                assert(dbg_state == 4'd1 || dbg_state == 4'd2);
            end

            // No weight can be consumed and no result can be emitted from abort
            // through the terminating done pulse.
            if ((activation_abort && busy) || (abort_seen && !kernel_done)) begin
                assert(!weight_ready);
                assert(!result_valid);
            end

            // The first clock after abort is the fail-closed error state.
            if (f_past_valid && $past(rst_n) && $past(activation_abort && busy)) begin
                assert(busy);
                assert(dbg_state == 4'd12);
                assert(activation_error);
                assert(!activation_valid);
                assert(!weight_ready && !act_ready && !result_valid);
            end

            // The following clock must expose done while already idle. Resident
            // activation metadata remain invalid and no stream can transfer.
            if (abort_seen && $past(rst_n, 2) && $past(activation_abort && busy, 2)) begin
                assert(kernel_done);
                assert(!busy);
                assert(dbg_state == 4'd0);
                assert(activation_error);
                assert(!activation_valid);
                assert(!weight_ready && !act_ready && !result_valid);
            end
        end

        cover(rst_n && kernel_done && activation_error && abort_seen);
    end

    wire _unused = &{
        1'b0,
        result_data,
        result_last,
        result_keep,
        loaded_act_epoch,
        loaded_act_q1_blocks,
        loaded_act_cols
    };
endmodule

`default_nettype wire
