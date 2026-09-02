// Combinational IEEE binary32 to binary16 conversion, round-to-nearest-even.
//
// Exact binary16 subnormals are preserved here. The current Flash read-side
// converter deliberately flushes binary16 subnormals to signed zero.

`default_nettype none

module f32_to_f16 (
    input  wire [31:0] in,
    output wire [15:0] out
);
    reg [15:0] result;
    reg [7:0] exponent;
    reg [22:0] fraction;
    reg [23:0] significand;
    reg [10:0] fraction_rounded;
    reg [10:0] sub_rounded;
    reg [10:0] sub_base;
    reg sub_guard;
    reg sub_sticky;
    reg increment;
    reg [8:0] half_exp;

    always @(*) begin
        exponent = in[30:23];
        fraction = in[22:0];
        significand = {1'b1, fraction};
        fraction_rounded = 11'd0;
        sub_rounded = 11'd0;
        sub_base = 11'd0;
        sub_guard = 1'b0;
        sub_sticky = 1'b0;
        increment = 1'b0;
        half_exp = 9'd0;
        result = {in[31], 15'd0};

        if (exponent == 8'hff) begin
            result = (fraction == 23'd0) ? {in[31], 15'h7c00}
                                         : {in[31], 15'h7e00};
        end else if (exponent >= 8'd143) begin
            result = {in[31], 15'h7c00};
        end else if (exponent >= 8'd113) begin
            increment = fraction[12] && ((|fraction[11:0]) ||
                                         fraction[13]);
            fraction_rounded = {1'b0, fraction[22:13]} +
                               {{10{1'b0}}, increment};
            half_exp = {1'b0, exponent} - 9'd112;
            if (fraction_rounded[10]) begin
                half_exp = half_exp + 9'd1;
                if (half_exp >= 9'd31)
                    result = {in[31], 15'h7c00};
                else
                    result = {in[31], half_exp[4:0], 10'd0};
            end else begin
                result = {in[31], half_exp[4:0],
                          fraction_rounded[9:0]};
            end
        end else if (exponent >= 8'd102) begin
            case (exponent)
                8'd112: begin sub_base = {1'd0, significand[23:14]}; sub_guard = significand[13]; sub_sticky = |significand[12:0]; end
                8'd111: begin sub_base = {2'd0, significand[23:15]}; sub_guard = significand[14]; sub_sticky = |significand[13:0]; end
                8'd110: begin sub_base = {3'd0, significand[23:16]}; sub_guard = significand[15]; sub_sticky = |significand[14:0]; end
                8'd109: begin sub_base = {4'd0, significand[23:17]}; sub_guard = significand[16]; sub_sticky = |significand[15:0]; end
                8'd108: begin sub_base = {5'd0, significand[23:18]}; sub_guard = significand[17]; sub_sticky = |significand[16:0]; end
                8'd107: begin sub_base = {6'd0, significand[23:19]}; sub_guard = significand[18]; sub_sticky = |significand[17:0]; end
                8'd106: begin sub_base = {7'd0, significand[23:20]}; sub_guard = significand[19]; sub_sticky = |significand[18:0]; end
                8'd105: begin sub_base = {8'd0, significand[23:21]}; sub_guard = significand[20]; sub_sticky = |significand[19:0]; end
                8'd104: begin sub_base = {9'd0, significand[23:22]}; sub_guard = significand[21]; sub_sticky = |significand[20:0]; end
                8'd103: begin sub_base = {10'd0, significand[23]}; sub_guard = significand[22]; sub_sticky = |significand[21:0]; end
                default: begin sub_base = 11'd0; sub_guard = significand[23]; sub_sticky = |significand[22:0]; end
            endcase
            increment = sub_guard && (sub_sticky || sub_base[0]);
            sub_rounded = sub_base + {{10{1'b0}}, increment};
            if (sub_rounded[10])
                result = {in[31], 15'h0400};
            else
                result = {in[31], 5'd0, sub_rounded[9:0]};
        end
    end

    assign out = result;
endmodule

`default_nettype wire
