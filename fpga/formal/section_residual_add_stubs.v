`default_nettype none

// Lifecycle-faithful arithmetic boundary for the residual controller proof.
// The real exact-RNE leaf is covered by exhaustive/differential Verilator
// cosim; this stub keeps request exclusion, fixed latency, result retention,
// status propagation, abort, and restart behavior in the control state space.
module section_residual_add_rne (
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
    localparam [31:0] FAULT_WORD = 32'h7f80_0000;

    reg        busy_q;
    reg [1:0]  age_q;
    reg        valid_q;
    reg [31:0] data_q;
    reg [1:0]  status_q;

    assign busy = busy_q;
    assign s_ready = rst_n && !abort_run && !busy_q;
    assign result_valid = rst_n && !abort_run && valid_q;
    assign result_data = data_q;
    assign result_status = status_q;

    always @(posedge clk) begin
        if (!rst_n || abort_run) begin
            busy_q <= 1'b0;
            age_q <= 2'd0;
            valid_q <= 1'b0;
            data_q <= 32'd0;
            status_q <= 2'd0;
        end else if (s_valid && s_ready) begin
            busy_q <= 1'b1;
            age_q <= 2'd2;
            valid_q <= 1'b0;
            data_q <= s_a ^ s_b;
            status_q <= ((s_a == FAULT_WORD) || (s_b == FAULT_WORD)) ?
                        2'b01 : 2'b00;
        end else if (busy_q && !valid_q) begin
            if (age_q == 2'd1) begin
                age_q <= 2'd0;
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
endmodule

`default_nettype wire
