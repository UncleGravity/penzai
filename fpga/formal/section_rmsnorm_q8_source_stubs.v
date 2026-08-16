`default_nettype none

// Fixed-latency control abstraction of the independently proven exact FP32
// multiplier.  The noncommutative function preserves operand-order coverage.
module section_rmsnorm_mul_rne (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        abort_run,
    output wire        busy,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire [31:0] s_a,
    input  wire [31:0] s_b,
    output wire        result_valid,
    input  wire        result_ready,
    output wire [31:0] result_data,
    output wire [1:0]  result_status
);
    reg        busy_q;
    reg [2:0]  age_q;
    reg        valid_q;
    reg [31:0] data_q;
    reg [1:0]  status_q;

    function automatic [31:0] model_data(
        input [31:0] a,
        input [31:0] b
    );
        model_data = a ^ {b[15:0], b[31:16]} ^ 32'h6d2b_79f5;
    endfunction

    function automatic [1:0] model_status(
        input [31:0] a,
        input [31:0] b
    );
        begin
            if (a == 32'h7f80_0000 || b == 32'h7f80_0000)
                model_status = 2'b01;
            else if (a == 32'h7f7f_ffff || b == 32'h7f7f_ffff)
                model_status = 2'b10;
            else
                model_status = 2'b00;
        end
    endfunction

    assign busy = busy_q;
    assign s_ready = rst_n && !abort_run && !busy_q;
    assign result_valid = valid_q;
    assign result_data = data_q;
    assign result_status = status_q;

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            busy_q <= 1'b0;
            age_q <= 3'd0;
            valid_q <= 1'b0;
            data_q <= 32'd0;
            status_q <= 2'd0;
        end else begin
            if (s_valid && s_ready) begin
                busy_q <= 1'b1;
                age_q <= 3'd5;
                valid_q <= 1'b0;
                data_q <= model_data(s_a, s_b);
                status_q <= model_status(s_a, s_b);
            end else if (busy_q && !valid_q) begin
                if (age_q == 3'd1) begin
                    age_q <= 3'd0;
                    valid_q <= 1'b1;
                end else begin
                    age_q <= age_q - 1'b1;
                end
            end else if (valid_q && result_ready) begin
                busy_q <= 1'b0;
                valid_q <= 1'b0;
                data_q <= 32'd0;
                status_q <= 2'd0;
            end
        end
    end
endmodule

// Stream-accurate abstraction of q8_quantizer.  It consumes exactly 32 framed
// scalars, waits a fixed three cycles, and retains one complete native record
// until the fifth serialized beat releases it through out_ready.
module q8_quantizer (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         in_valid,
    output wire         in_ready,
    input  wire [31:0]  in_data,
    input  wire         in_last,
    output reg          out_valid,
    input  wire         out_ready,
    output reg  [255:0] out_quants,
    output reg  [15:0]  out_scale,
    output reg  [3:0]   out_status
);
    localparam [31:0] FAULT_WORD = 32'hdead_c0de;

    reg [5:0] accepted_q;
    reg [2:0] latency_q;
    reg       pending_q;
    reg       fault_q;
    reg [8:0] record_q;

    wire input_fire = in_valid && in_ready;
    wire input_fault = in_data == FAULT_WORD;
    assign in_ready = rst_n && !pending_q && !out_valid;

    function automatic [63:0] record_beat(
        input [2:0] beat,
        input [8:0] record_index
    );
        begin
            case (beat)
                3'd0: record_beat = 64'h1100_0000_0000_0000 |
                                      {55'd0, record_index};
                3'd1: record_beat = 64'h2200_0000_0000_0000 |
                                      {55'd0, record_index};
                3'd2: record_beat = 64'h3300_0000_0000_0000 |
                                      {55'd0, record_index};
                default: record_beat = 64'h4400_0000_0000_0000 |
                                       {55'd0, record_index};
            endcase
        end
    endfunction

    always @(posedge clk) begin
        if (!rst_n) begin
            accepted_q <= 6'd0;
            latency_q <= 3'd0;
            pending_q <= 1'b0;
            fault_q <= 1'b0;
            record_q <= 9'd0;
            out_valid <= 1'b0;
            out_quants <= 256'd0;
            out_scale <= 16'd0;
            out_status <= 4'd0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
                out_quants <= 256'd0;
                out_scale <= 16'd0;
                out_status <= 4'd0;
                record_q <= record_q + 1'b1;
            end

            if (input_fire) begin
                assert(in_last == (accepted_q == 6'd31));
                fault_q <= fault_q || input_fault;
                if (accepted_q == 6'd31) begin
                    accepted_q <= 6'd0;
                    latency_q <= 3'd3;
                    pending_q <= 1'b1;
                end else begin
                    accepted_q <= accepted_q + 1'b1;
                end
            end else if (pending_q) begin
                if (latency_q == 3'd1) begin
                    latency_q <= 3'd0;
                    pending_q <= 1'b0;
                    out_valid <= 1'b1;
                    out_quants <= {
                        record_beat(3'd3, record_q),
                        record_beat(3'd2, record_q),
                        record_beat(3'd1, record_q),
                        record_beat(3'd0, record_q)
                    };
                    out_scale <= 16'h5000 | {7'd0, record_q};
                    out_status <= fault_q ? 4'b0010 : 4'd0;
                    fault_q <= 1'b0;
                end else begin
                    latency_q <= latency_q - 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
