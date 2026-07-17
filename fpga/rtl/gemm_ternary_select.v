`default_nettype none

// Translate 32 upstream Q2_0 codes into the GEMM selector controls.
// 0 => -1, 1 => 0, 2 => +1. The low bit is the zero mask and the high
// bit is the sign selector. Reserved code 3 is therefore disabled by
// nonzero; the software packer rejects it before weights reach the device.
module gemm_ternary_select32 (
    input  wire [63:0] codes,
    output wire [31:0] sign,
    output wire [31:0] nonzero
);
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : gen_select
            assign nonzero[i] = ~codes[i*2];
            assign sign[i] = codes[i*2 + 1];
        end
    endgenerate
endmodule

`default_nettype wire
