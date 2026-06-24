// reduce_ooc - OOC area/timing probe for numeric/reduce (N=16, fp32) — the flash dot's
// u_tree, the module whose internal fadd normalize was the f300 limiter (lead_pos_s3 ->
// out). Registers ALL inputs and outputs so the timed path is reg -> reduce -> reg
// (ooc_synth.tcl false-paths the top I/O). Throwaway, not in any bitstream.
//
//   ssh $VM "cd penzai-ooc && ooc.bat reduce_ooc 3.333 redu reduce_ooc.v reduce.v fadd.v"

`default_nettype none

module reduce_ooc (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             valid_in,
    input  wire [16*32-1:0] in,         // 16 × f32 lanes
    output reg              valid_out_q,
    output reg  [31:0]      sum_q
);
    reg             vin_q;
    reg [16*32-1:0] in_q;
    always @(posedge clk) begin
        vin_q <= valid_in;
        in_q  <= in;
    end

    wire vout;  wire [31:0] sum;
    reduce #(.MANT_W(23), .N(16)) u (
        .clk(clk), .rst_n(rst_n), .valid_in(vin_q),
        .in(in_q), .valid_out(vout), .sum(sum)
    );

    always @(posedge clk) begin
        valid_out_q <= vout;
        sum_q       <= sum;
    end
endmodule
