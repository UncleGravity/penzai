`default_nettype none

// Control-only proof boundary for swiglu. Arithmetic values and FIFO
// contents are abstract, while production latency-valid chains, reservation
// accounting, queue control, output elasticity, and abort handling are retained.
module swiglu_harness(input wire clk);
    // A reduced reservation bound makes unbounded induction tractable while
    // preserving the production counter/FIFO logic. RTL cosim separately fills
    // and releases the default 64-credit configuration.
    localparam [6:0] CAPACITY = 7'd4;

    (* anyseq *) reg        rst_n;
    (* anyseq *) reg        abort;
    (* anyseq *) reg        in_valid;
    (* anyseq *) reg [31:0] in_gate;
    (* anyseq *) reg [31:0] in_up;
    (* anyseq *) reg        in_last;
    (* anyseq *) reg        out_ready;

    wire        in_ready;
    wire        in_ready_core;
    wire        out_valid;
    wire [31:0] out_data;
    wire        out_last;
    wire [1:0]  out_status;
    wire [6:0]  formal_reserved_count;
    wire [9:0]  formal_fifo_count;
    wire        formal_fifo_write_enable;
    wire        formal_fifo_read_enable;
    wire input_fire = in_valid && in_ready;
    wire output_fire = out_valid && out_ready;

    swiglu #(.RESERVE_DEPTH(CAPACITY)) dut (
        .clk(clk), .rst_n(rst_n), .abort(abort),
        .in_valid(in_valid), .in_ready(in_ready),
        .in_ready_core(in_ready_core),
        .in_gate(in_gate), .in_up(in_up), .in_last(in_last),
        .out_valid(out_valid), .out_ready(out_ready),
        .out_data(out_data), .out_last(out_last), .out_status(out_status),
        .formal_reserved_count(formal_reserved_count),
        .formal_fifo_count(formal_fifo_count),
        .formal_fifo_write_enable(formal_fifo_write_enable),
        .formal_fifo_read_enable(formal_fifo_read_enable)
    );

    reg f_past_valid = 1'b0;
    reg f_saw_stall = 1'b0;
    reg f_saw_abort = 1'b0;

    always @(posedge clk) begin
        f_past_valid <= 1'b1;
        if (!f_past_valid)
            assume(!rst_n);
        else
            assume(rst_n);

        if (!rst_n) begin
            f_saw_stall <= 1'b0;
            f_saw_abort <= 1'b0;
        end else if (abort) begin
            f_saw_stall <= 1'b0;
            f_saw_abort <= 1'b1;
        end else begin
            if (out_valid && !out_ready)
                f_saw_stall <= 1'b1;
        end

        if (rst_n) begin
            assert(formal_reserved_count <= CAPACITY);
            assert(in_ready_core == (formal_reserved_count < CAPACITY));
            assert(in_ready == (in_ready_core && !abort));
            assert(formal_fifo_count <= CAPACITY);
            if (input_fire)
                assert(formal_reserved_count < CAPACITY);
            if (formal_reserved_count == CAPACITY)
                assert(!in_ready);
            if (abort) begin
                assert(!in_ready);
                assert(!formal_fifo_write_enable);
                assert(!formal_fifo_read_enable);
            end
        end

        if (f_past_valid && rst_n && !abort &&
            $past(rst_n && !abort && out_valid && !out_ready)) begin
            assert(out_valid);
            assert(out_data == $past(out_data));
            assert(out_last == $past(out_last));
            assert(out_status == $past(out_status));
        end

        if (f_past_valid && rst_n && $past(rst_n && abort)) begin
            assert(!out_valid);
            assert(formal_reserved_count == 0);
            assert(formal_fifo_count == 0);
        end

        cover(rst_n && formal_reserved_count == CAPACITY);
        cover(rst_n && f_saw_stall && output_fire);
        cover(rst_n && f_saw_abort && !out_valid && in_ready);
    end
endmodule

`default_nettype wire
