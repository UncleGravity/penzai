// axpy_ooc - OOC area/timing probe for the shipping bf16 fp_axpy8 (p·V in bf16).
// Registers ALL inputs and outputs so the timed path is reg -> fp_axpy8 -> reg
// (ooc_synth.tcl false-paths the top I/O). Throwaway, not in any bitstream.
//
//   ssh $VM "cd penzai-ooc && ooc.bat axpy_ooc 3.333 axpybf16 axpy_ooc.v fp_axpy8.v cvt.v fmul.v fadd.v"

`default_nettype none

module axpy_ooc (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,
    input  wire [255:0] acc,
    input  wire [127:0] v,
    input  wire [31:0]  s1,
    input  wire [31:0]  p,
    output reg          valid_out_q,
    output reg  [255:0] out_q
);
    reg         vin_q;
    reg [255:0] acc_q;
    reg [127:0] v_q;
    reg [31:0]  s1_q, p_q;
    always @(posedge clk) begin
        vin_q <= valid_in; acc_q <= acc; v_q <= v; s1_q <= s1; p_q <= p;
    end

    wire vout;  wire [255:0] out;
    fp_axpy8 u (
        .clk(clk), .rst_n(rst_n), .valid_in(vin_q),
        .acc(acc_q), .v(v_q), .s1(s1_q), .p(p_q),
        .valid_out(vout), .out(out)
    );

    always @(posedge clk) begin
        valid_out_q <= vout; out_q <= out;
    end
endmodule
