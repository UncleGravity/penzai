// Registered out-of-context shell for routing the explicit 512-DSP dot fabric.

`default_nettype none

module dot4_ooc (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          in_valid,
    input  wire [511:0]  weight_sign,
    input  wire [511:0]  weight_nonzero,
    input  wire [1023:0] acts_flat,
    output wire          out_valid,
    output wire [895:0]  sums_flat
);
    reg in_valid_q;
    reg [511:0] weight_sign_q;
    reg [511:0] weight_nonzero_q;
    reg [1023:0] acts_flat_q;
    always @(posedge clk) begin
        if (!rst_n)
            in_valid_q <= 1'b0;
        else
            in_valid_q <= in_valid;
        weight_sign_q <= weight_sign;
        weight_nonzero_q <= weight_nonzero;
        acts_flat_q <= acts_flat;
    end

     dot4 u_dot (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid_q),
        .weight_sign(weight_sign_q),
        .weight_nonzero(weight_nonzero_q),
        .acts_flat(acts_flat_q),
        .out_valid(out_valid),
        .sums_flat(sums_flat)
    );
endmodule

`default_nettype wire
