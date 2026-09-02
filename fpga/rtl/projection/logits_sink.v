// Streaming LM-head publication and greedy selection.
//
// Every accepted finite value participates in an exact FP32 argmax.  Equal
// maxima choose the first row and NaNs are ignored, matching llama's host
// greedy sampler. Full logits may be passed through for numerical validation; the
// normal greedy path disables that stream and returns only the selected token.

`default_nettype none

module logits_sink (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,

    input  wire         start_valid,
    output wire         start_ready,
    input  wire [17:0]  vocab_rows,
    input  wire         emit_full_logits,

    input  wire         in_valid,
    output wire         in_ready,
    input  wire [17:0]  in_row,
    input  wire [31:0]  in_data,
    input  wire         in_last,

    output wire         logits_valid,
    input  wire         logits_ready,
    output wire [17:0]  logits_row,
    output wire [31:0]  logits_data,
    output wire         logits_last,

    output wire         result_valid,
    input  wire         result_ready,
    output wire [17:0]  result_token,
    output wire [31:0]  result_logit,
    output wire         result_error,
    output wire [7:0]   result_status,
    output wire [31:0]  accepted_logits,
    output wire         busy
);
    localparam [7:0] ERR_CONFIG = 8'h01;
    localparam [7:0] ERR_ORDER  = 8'h02;
    localparam [7:0] ERR_LAST   = 8'h04;

    reg busy_q;
    reg result_valid_q;
    reg result_error_q;
    reg [7:0] result_status_q;
    reg [17:0] vocab_rows_q;
    reg emit_full_q;
    reg [17:0] expected_row_q;
    reg [31:0] accepted_q;
    reg [31:0] best_value_q;
    reg [31:0] best_key_q;
    reg [17:0] best_row_q;

    assign start_ready = !busy_q && !result_valid_q;
    wire start_fire = start_valid && start_ready;

    assign logits_valid = busy_q && emit_full_q && in_valid;
    assign logits_row = in_row;
    assign logits_data = in_data;
    assign logits_last = in_last;
    assign in_ready = busy_q && (!emit_full_q || logits_ready);
    wire in_fire = in_valid && in_ready;

    wire in_nan = (in_data[30:23] == 8'hff) && (in_data[22:0] != 0);
    wire in_zero = in_data[30:0] == 31'd0;
    wire [31:0] in_key = in_zero ? 32'h8000_0000 :
                           (in_data[31] ? ~in_data :
                                          (in_data ^ 32'h8000_0000));
    wire take_value = !in_nan && (in_key > best_key_q);
    wire order_ok = in_row == expected_row_q;
    wire expected_last = expected_row_q + 1'b1 == vocab_rows_q;
    wire last_ok = in_last == expected_last;

    assign result_valid = result_valid_q;
    assign result_token = best_row_q;
    assign result_logit = best_value_q;
    assign result_error = result_error_q;
    assign result_status = result_status_q;
    assign accepted_logits = accepted_q;
    assign busy = busy_q;

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            busy_q <= 1'b0;
            result_valid_q <= 1'b0;
            result_error_q <= 1'b0;
            result_status_q <= 8'd0;
            vocab_rows_q <= 18'd0;
            emit_full_q <= 1'b0;
            expected_row_q <= 18'd0;
            accepted_q <= 32'd0;
            best_value_q <= 32'hff80_0000;
            best_key_q <= 32'h007f_ffff;
            best_row_q <= 18'd0;
        end else begin
            if (result_valid_q && result_ready)
                result_valid_q <= 1'b0;

            if (start_fire) begin
                vocab_rows_q <= vocab_rows;
                emit_full_q <= emit_full_logits;
                expected_row_q <= 18'd0;
                accepted_q <= 32'd0;
                best_value_q <= 32'hff80_0000;
                best_key_q <= 32'h007f_ffff;
                best_row_q <= 18'd0;
                result_error_q <= 1'b0;
                result_status_q <= 8'd0;
                if (vocab_rows == 18'd0) begin
                    busy_q <= 1'b0;
                    result_valid_q <= 1'b1;
                    result_error_q <= 1'b1;
                    result_status_q <= ERR_CONFIG;
                end else begin
                    busy_q <= 1'b1;
                end
            end else if (in_fire) begin
                accepted_q <= accepted_q + 1'b1;
                if (take_value) begin
                    best_value_q <= in_data;
                    best_key_q <= in_key;
                    best_row_q <= in_row;
                end

                if (!order_ok || !last_ok) begin
                    busy_q <= 1'b0;
                    result_valid_q <= 1'b1;
                    result_error_q <= 1'b1;
                    result_status_q <= !order_ok ? ERR_ORDER : ERR_LAST;
                    best_value_q <= 32'hff80_0000;
                    best_key_q <= 32'h007f_ffff;
                    best_row_q <= 18'd0;
                end else if (expected_last) begin
                    busy_q <= 1'b0;
                    result_valid_q <= 1'b1;
                    // Nonblocking updates would otherwise publish the previous
                    // maximum when the final logit wins.
                    if (take_value) begin
                        best_value_q <= in_data;
                        best_row_q <= in_row;
                    end
                end else begin
                    expected_row_q <= expected_row_q + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
