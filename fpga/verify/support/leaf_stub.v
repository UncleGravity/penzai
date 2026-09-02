`default_nettype none

// Deterministic registered service used only for focused EXEC_TILE controller
// tests. Request metadata stays in the engine; the stage is supplied here
// solely for focused failure injection in simulation.
module leaf_stub #(
    parameter integer LATENCY = 3
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       req_valid,
    output wire       req_ready,
    input  wire [4:0] req_stage,
    input  wire       fail_enable,
    input  wire [4:0] fail_stage,
    input  wire [7:0] fail_status,
    output wire       done_valid,
    input  wire       done_ready,
    output wire       done_error,
    output wire [7:0] done_status
);
    localparam integer COUNT_W = (LATENCY < 2) ? 1 : $clog2(LATENCY + 1);

    reg busy_q;
    reg done_valid_q;
    reg [COUNT_W-1:0] count_q;
    reg fail_q;
    reg [7:0] status_q;

    assign req_ready = !busy_q && !done_valid_q;
    assign done_valid = done_valid_q;
    assign done_error = fail_q;
    assign done_status = status_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q <= 1'b0;
            done_valid_q <= 1'b0;
            count_q <= {COUNT_W{1'b0}};
            fail_q <= 1'b0;
            status_q <= 8'd0;
        end else begin
            if (done_valid_q && done_ready)
                done_valid_q <= 1'b0;

            if (req_valid && req_ready) begin
                busy_q <= 1'b1;
                count_q <= LATENCY[COUNT_W-1:0];
                fail_q <= fail_enable && (req_stage == fail_stage);
                status_q <= (fail_enable && (req_stage == fail_stage)) ?
                            fail_status : 8'd0;
            end else if (busy_q) begin
                if (count_q == {{(COUNT_W-1){1'b0}}, 1'b1}) begin
                    busy_q <= 1'b0;
                    done_valid_q <= 1'b1;
                    count_q <= {COUNT_W{1'b0}};
                end else begin
                    count_q <= count_q - 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
