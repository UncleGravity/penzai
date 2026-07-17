`default_nettype none

module gemm_ternary_select_formal;
    (* anyconst *) reg [63:0] codes;
    wire [31:0] sign;
    wire [31:0] nonzero;

    gemm_ternary_select32 dut (
        .codes(codes),
        .sign(sign),
        .nonzero(nonzero)
    );

    integer i;
    always @* begin
        for (i = 0; i < 32; i = i + 1) begin
            case (codes[i*2 +: 2])
                2'd0: assert (nonzero[i] && !sign[i]);
                2'd1: assert (!nonzero[i] && !sign[i]);
                2'd2: assert (nonzero[i] && sign[i]);
                2'd3: assert (!nonzero[i] && sign[i]);
            endcase
        end

        cover (codes == 64'h0000_0000_0000_0000 && nonzero == 32'hffff_ffff && sign == 32'd0);
        cover (codes == 64'h5555_5555_5555_5555 && nonzero == 32'd0);
        cover (codes == 64'haaaa_aaaa_aaaa_aaaa && nonzero == 32'hffff_ffff && sign == 32'hffff_ffff);
        cover (codes == 64'hffff_ffff_ffff_ffff && nonzero == 32'd0 && sign == 32'hffff_ffff);
    end
endmodule

`default_nettype wire
