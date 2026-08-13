`default_nettype none

// Control-accurate abstraction for proving q8_ingress framing and serialization.
// Numeric correctness remains owned by the standalone q8_quantizer cosim.
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
    reg [5:0] accepted;
    assign in_ready = !out_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            accepted <= 6'd0;
            out_valid <= 1'b0;
            out_quants <= 256'd0;
            out_scale <= 16'd0;
            out_status <= 4'd0;
        end else begin
            if (out_valid && out_ready)
                out_valid <= 1'b0;
            if (in_valid && in_ready) begin
                assert(in_last == (accepted == 6'd31));
                if (accepted == 6'd31) begin
                    accepted <= 6'd0;
                    out_valid <= 1'b1;
                end else begin
                    accepted <= accepted + 1'b1;
                end
            end
        end
    end

    wire _unused = &{1'b0, in_data};
endmodule

`default_nettype wire
