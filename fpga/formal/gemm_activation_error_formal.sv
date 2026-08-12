`default_nettype none

// Focused fail-closed proof: a reuse request without resident activation must finish
// with an error while accepting no weight/activation beats and emitting no result.
module gemm_activation_error_formal(input wire clk);
    localparam integer ROWS = 4;
    localparam integer COLS_MAX = 1;

    (* anyseq *) reg rst_n;
    (* anyseq *) reg [ROWS*32-1:0] weight_data;
    (* anyseq *) reg [63:0] act_data;
    reg [2:0] step = 0;
    reg f_past_valid = 1'b0;

    wire start_kernel = (step == 3'd1);
    wire busy, kernel_done, weight_ready, act_ready;
    wire [63:0] result_data;
    wire result_valid, result_last;
    wire [7:0] result_keep;
    wire [3:0] dbg_state;
    wire activation_error, activation_valid;
    wire [31:0] loaded_act_epoch;
    wire [15:0] loaded_act_q1_blocks, loaded_act_cols;

    gemm_kernel #(.ROWS(ROWS), .COLS_MAX(COLS_MAX),
        .MAX_SUB_INDEX(4), .ACC_W(16)) dut (
        .clk(clk), .rst_n(rst_n), .start_kernel(start_kernel),
        .num_q1_blocks(16'd1), .num_rowblocks(16'd1), .num_rows(32'd4),
        .num_cols(16'd1), .weight_fmt(2'd1),
        .act_mode(2'd1), .act_epoch(32'hBAD0_E001), .emin(-8'sd8),
        .activation_abort(1'b0),
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
        end else begin
            assume(rst_n);
            if (step != 3'd7) step <= step + 1'b1;
        end

        if (rst_n) begin
            assert(!weight_ready && !act_ready && !result_valid);
            assert(!activation_valid);
            assert(loaded_act_epoch == 0);
            assert(loaded_act_q1_blocks == 0);
            assert(loaded_act_cols == 0);
            if (activation_error) assert(!busy || dbg_state == 4'd12);
            if (kernel_done) begin
                assert(activation_error);
                assert(!busy);
            end
        end

        cover(rst_n && kernel_done && activation_error);
    end

    wire _unused = &{1'b0, result_data, result_last, result_keep};
endmodule

`default_nettype wire
