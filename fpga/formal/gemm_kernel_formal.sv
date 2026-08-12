`default_nettype none

module gemm_kernel_formal(input wire clk);
    localparam integer ROWS = 4;
    localparam integer COLS_MAX = 2;
    localparam integer MAX_SUB_INDEX = 8;
    localparam integer ACC_W = 16;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg start_kernel;
    (* anyseq *) reg [15:0] num_q1_blocks;
    (* anyseq *) reg [15:0] num_rowblocks;
    (* anyseq *) reg [31:0] num_rows;
    (* anyseq *) reg [15:0] num_cols;
    (* anyseq *) reg signed [7:0] emin;
    (* anyseq *) reg [1:0] weight_fmt;
    // Preserve the original aggregate proof's single packed-load scope. Reuse
    // rejection has a focused harness and successful reuse is covered in cosim.
    wire [1:0] act_mode = 2'd0;
    wire [31:0] act_epoch = 32'd0;
    (* anyseq *) reg [ROWS*32-1:0] s_axis_tdata;
    (* anyseq *) reg s_axis_tvalid;
    (* anyseq *) reg [63:0] s_axis_acts_tdata;
    (* anyseq *) reg s_axis_acts_tvalid;
    (* anyseq *) reg m_axis_tready;

    wire busy, kernel_done, s_axis_tready, s_axis_acts_tready;
    wire [63:0] m_axis_tdata;
    wire m_axis_tvalid, m_axis_tlast;
    wire [7:0] m_axis_tkeep;
    wire [3:0] dbg_state;
    wire activation_error, activation_valid;
    wire [31:0] loaded_act_epoch;
    wire [15:0] loaded_act_q1_blocks, loaded_act_cols;

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(COLS_MAX),
        .MAX_SUB_INDEX(MAX_SUB_INDEX), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst_n(rst_n), .start_kernel(start_kernel),
        .num_q1_blocks(num_q1_blocks), .num_rowblocks(num_rowblocks),
        .num_rows(num_rows),
        .num_cols(num_cols), .emin(emin), .kernel_done(kernel_done), .busy(busy),
        .weight_fmt(weight_fmt), .act_mode(act_mode), .act_epoch(act_epoch),
        .activation_abort(1'b0),
        .activation_error(activation_error), .activation_valid(activation_valid),
        .loaded_act_epoch(loaded_act_epoch),
        .loaded_act_q1_blocks(loaded_act_q1_blocks), .loaded_act_cols(loaded_act_cols),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_acts_tdata(s_axis_acts_tdata), .s_axis_acts_tvalid(s_axis_acts_tvalid),
        .s_axis_acts_tready(s_axis_acts_tready), .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast), .m_axis_tkeep(m_axis_tkeep), .dbg_state(dbg_state)
    );

    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid) assume(!rst_n); else assume(rst_n);

        if (start_kernel && !busy) begin
            assume(num_q1_blocks > 0 && num_q1_blocks <= 2);
            assume(num_rowblocks > 0 && num_rowblocks <= 2);
            assume(num_rows == 0 ||
                   (num_rows > (num_rowblocks - 1'b1) * ROWS &&
                    num_rows <= num_rowblocks * ROWS));
            assume(num_cols > 0 && num_cols <= COLS_MAX);
            assume(weight_fmt == 1 || weight_fmt == 2);
        end
        if (f_past_valid && $past(start_kernel)) assume(!start_kernel);

        if (f_past_valid && $past(rst_n) && $past(s_axis_tvalid && !s_axis_tready)) begin
            assume(s_axis_tvalid);
            assume(s_axis_tdata == $past(s_axis_tdata));
        end
        if (f_past_valid && $past(rst_n) &&
            $past(s_axis_acts_tvalid && !s_axis_acts_tready)) begin
            assume(s_axis_acts_tvalid);
            assume(s_axis_acts_tdata == $past(s_axis_acts_tdata));
        end
    end
endmodule

`default_nettype wire
