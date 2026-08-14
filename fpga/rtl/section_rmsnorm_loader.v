// Token-major residual loader for the P3d RMSNorm section.
//
// Each accepted 64-bit input word is atomically presented to the direct R
// scratch write port. Four adjacent words are also packed into one 256-bit
// group for the maximum-exponent scanner. The scratch mapping is:
//
//   bank = row_pair % 4
//   addr = token * 512 + row_pair / 4
//
// Results written before a later framing/sink failure are tentative. The
// section controller must invalidate R on any error or abort.

`default_nettype none

module section_rmsnorm_loader (
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
    // bit 0 BAD_CFG, 1 FRAME, 2 SINK, 3 INTERNAL.
    output reg  [3:0]    status,

    input  wire [63:0]   s_axis_tdata,
    input  wire [7:0]    s_axis_tkeep,
    input  wire          s_axis_tvalid,
    output wire          s_axis_tready,
    input  wire          s_axis_tlast,

    output wire          wr_valid,
    input  wire          wr_ready,
    input  wire          wr_error,
    output wire [1:0]    wr_bank,
    output wire [13:0]   wr_address,
    output wire [63:0]   wr_data,

    output wire          group_valid,
    input  wire          group_ready,
    output wire [255:0]  group_data,
    output wire          group_last
);
    localparam [3:0] STATUS_BAD_CFG  = 4'b0001;
    localparam [3:0] STATUS_FRAME    = 4'b0010;
    localparam [3:0] STATUS_SINK     = 4'b0100;
    localparam [3:0] STATUS_INTERNAL = 4'b1000;

    reg active_q;
    reg input_complete_q;
    reg [13:0] run_rows_q;
    reg [2:0] run_tokens_q;
    reg [11:0] run_words_q;
    reg [1:0] token_q;
    reg [10:0] word_q;
    reg [191:0] pack_q;
    reg group_valid_q;
    reg [255:0] group_data_q;
    reg group_last_q;

    wire cfg_shape_ok = (cfg_rows >= 14'd8) &&
                        (cfg_rows <= 14'd4096) &&
                        (cfg_rows[2:0] == 3'b000) &&
                        (cfg_tokens != 3'd0) &&
                        (cfg_tokens <= 3'd4);
    wire cfg_accept = cfg_valid && cfg_ready;

    assign cfg_ready = rst_n && !abort_run && !active_q;
    assign busy = active_q;
    assign group_valid = rst_n && !abort_run && group_valid_q;
    assign group_data = group_data_q;
    assign group_last = group_last_q;

    wire group_slot_ready = !group_valid_q || group_ready;
    wire word_completes_group = word_q[1:0] == 2'd3;
    wire input_admissible = rst_n && active_q && !abort_run &&
                            !input_complete_q &&
                            (!word_completes_group || group_slot_ready);
    assign wr_valid = input_admissible && s_axis_tvalid;
    assign s_axis_tready = input_admissible && wr_ready;
    assign wr_bank = word_q[1:0];
    assign wr_address = {3'b000, token_q, 9'b0} +
                        {5'b00000, word_q[10:2]};
    assign wr_data = s_axis_tdata;

    wire input_fire = s_axis_tvalid && s_axis_tready;
    wire wr_fire = wr_valid && wr_ready;
    wire group_fire = group_valid && group_ready;
    wire word_final = {1'b0, word_q} + 12'd1 == run_words_q;
    wire input_final = word_final &&
                       ({1'b0, token_q} + 3'd1 == run_tokens_q);
    wire frame_bad = (s_axis_tkeep != 8'hff) ||
                     (s_axis_tlast != input_final);

    task automatic fail_run(input [3:0] failure);
        begin
            active_q <= 1'b0;
            input_complete_q <= 1'b0;
            group_valid_q <= 1'b0;
            done <= 1'b1;
            error <= 1'b1;
            status <= status | failure;
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            active_q <= 1'b0;
            input_complete_q <= 1'b0;
            run_rows_q <= 14'd0;
            run_tokens_q <= 3'd0;
            run_words_q <= 12'd0;
            token_q <= 2'd0;
            word_q <= 11'd0;
            pack_q <= 192'd0;
            group_valid_q <= 1'b0;
            group_data_q <= 256'd0;
            group_last_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else if (abort_run) begin
            active_q <= 1'b0;
            input_complete_q <= 1'b0;
            token_q <= 2'd0;
            word_q <= 11'd0;
            pack_q <= 192'd0;
            group_valid_q <= 1'b0;
            group_data_q <= 256'd0;
            group_last_q <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            status <= 4'd0;
        end else begin
            done <= 1'b0;

            if (group_fire)
                group_valid_q <= 1'b0;

            if (cfg_accept) begin
                error <= 1'b0;
                status <= 4'd0;
                input_complete_q <= 1'b0;
                token_q <= 2'd0;
                word_q <= 11'd0;
                pack_q <= 192'd0;
                group_valid_q <= 1'b0;
                group_last_q <= 1'b0;
                if (!cfg_shape_ok) begin
                    active_q <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                    status <= STATUS_BAD_CFG;
                end else begin
                    active_q <= 1'b1;
                    run_rows_q <= cfg_rows;
                    run_tokens_q <= cfg_tokens;
                    run_words_q <= cfg_rows[12:1];
                end
            end else if (input_fire) begin
                if (!wr_fire) begin
                    fail_run(STATUS_INTERNAL);
                end else if (wr_error) begin
                    fail_run(STATUS_SINK);
                end else if (frame_bad) begin
                    fail_run(STATUS_FRAME);
                end else begin
                    case (word_q[1:0])
                        2'd0: pack_q[63:0] <= s_axis_tdata;
                        2'd1: pack_q[127:64] <= s_axis_tdata;
                        2'd2: pack_q[191:128] <= s_axis_tdata;
                        default: begin
                            group_data_q <= {s_axis_tdata, pack_q};
                            group_last_q <= input_final;
                            group_valid_q <= 1'b1;
                        end
                    endcase

                    if (input_final) begin
                        input_complete_q <= 1'b1;
                    end else if (word_final) begin
                        token_q <= token_q + 1'b1;
                        word_q <= 11'd0;
                    end else begin
                        word_q <= word_q + 1'b1;
                    end
                end
            end

            if (group_fire && group_last_q) begin
                active_q <= 1'b0;
                input_complete_q <= 1'b0;
                done <= 1'b1;
            end
        end
    end

`ifdef FORMAL
    reg f_past_valid = 1'b0;
    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (rst_n) begin
            assert(input_fire == wr_fire);
            if (active_q) begin
                assert(run_rows_q >= 14'd8 && run_rows_q <= 14'd4096);
                assert(run_rows_q[2:0] == 3'b000);
                assert(run_tokens_q >= 3'd1 && run_tokens_q <= 3'd4);
                assert({1'b0, token_q} < run_tokens_q);
                assert({1'b0, word_q} < run_words_q);
            end
            if (wr_valid) begin
                assert(s_axis_tvalid);
                assert(wr_bank == word_q[1:0]);
                assert(wr_address < 14'd2048);
            end
            if (group_valid)
                assert(group_last_q == input_complete_q);
        end
        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && group_valid && !group_ready)) begin
            assert(group_valid);
            assert(group_data == $past(group_data));
            assert(group_last == $past(group_last));
        end
        if (f_past_valid && rst_n && $past(rst_n && abort_run)) begin
            assert(!busy);
            assert(!group_valid);
            assert(!error);
            assert(status == 4'd0);
        end
    end
`endif

    wire _unused = &{1'b0, run_rows_q};
endmodule

`default_nettype wire
