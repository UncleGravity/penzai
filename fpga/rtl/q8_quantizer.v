// q8_quantizer - canonical FP32 -> Q8_0 block quantizer.
//
// One block is exactly 32 FP32 values. The leaf buffers the block, finds the
// largest finite magnitude, and emits the GEMM-native record: 32 signed bytes
// plus one IEEE binary16 scale. Arithmetic matches shared/layout.zig:
//
//   scale = binary32(amax / 127)
//   stored_scale = binary16_rne(scale)
//   inv = binary32(1 / scale)
//   q[i] = clamp_i8(round_even(binary32(value[i] * inv)))
//
// The existing floating-point leaves are deliberately truncating or LUT based,
// so this block keeps its exact-RNE divider/multiplier local. Division is iterative
// (27 quotient bits) and multiplication is time-multiplexed across the 32 lanes.
// The surrounding ingress and section controller buffer this block latency while
// keeping this leaf as the sole exact canonical conversion implementation.
//
// Status bits are sticky for the completed block:
//   [0] a non-finite input was replaced with zero in the quant stream
//   [1] the stored f16 scale is infinite (the GEMM cannot use it)
//   [2] input TLAST did not coincide with lane 31
//   [3] exact normal-range division was impossible (f32 under/overflow)
// A zero block is valid and emits an all-zero record. out_* remains stable while
// out_valid && !out_ready. Datapath storage is not reset; valid state owns it.

`default_nettype none

// The exact divider is intentionally colocated with its sole consumer so the
// arithmetic contract is one review unit rather than a generic numeric-library API.
/* verilator lint_off DECLFILENAME */

module q8_fp32_div_rne_pos (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output reg         busy,
    output reg         done,
    output reg         invalid,
    output reg  [31:0] out
);
    wire a_normal = !a[31] && (a[30:23] != 8'd0) && (a[30:23] != 8'hff);
    wire b_normal = !b[31] && (b[30:23] != 8'd0) && (b[30:23] != 8'hff);
    wire [23:0] a_sig = {1'b1, a[22:0]};
    wire [23:0] b_sig = {1'b1, b[22:0]};
    wire normalize_left = a_sig < b_sig;

    reg [24:0] remainder;
    reg [23:0] divisor;
    reg [26:0] quotient;
    reg [4:0]  bit_index;
    reg signed [10:0] result_exp;
    reg finalize;
    reg result_pending;
    reg [22:0] rounded_frac_q;
    reg signed [10:0] rounded_exp_q;

    wire div_ge = remainder >= {1'b0, divisor};
    wire [24:0] remainder_sub = div_ge ? remainder - {1'b0, divisor} : remainder;

    reg [26:0] quotient_next;
    reg [22:0] rounded_frac;
    reg signed [10:0] rounded_exp;
    reg [24:0] rounded_ext;
    reg round_up;
    always @(*) begin
        quotient_next = quotient;
        quotient_next[bit_index] = div_ge;

        // Final rounding consumes the quotient/remainder registered by the last
        // divide iteration. This is a deliberate cycle boundary after the
        // compare/subtract path.
        round_up = quotient[2] &&
            (quotient[1] || quotient[0] || (remainder != 25'd0) || quotient[3]);
        rounded_ext = {1'b0, quotient[26:3]} + {{24{1'b0}}, round_up};
        rounded_exp = result_exp;
        if (rounded_ext[24]) begin
            rounded_frac = rounded_ext[23:1];
            rounded_exp = result_exp + 11'sd1;
        end else begin
            rounded_frac = rounded_ext[22:0];
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            invalid <= 1'b0;
            finalize <= 1'b0;
            result_pending <= 1'b0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                invalid <= !(a_normal && b_normal);
                finalize <= 1'b0;
                result_pending <= 1'b0;
                if (!(a_normal && b_normal)) begin
                    out  <= 32'd0;
                    done <= 1'b1;
                end else begin
                    busy       <= 1'b1;
                    remainder  <= normalize_left ? {a_sig, 1'b0} : {1'b0, a_sig};
                    divisor    <= b_sig;
                    quotient   <= 27'd0;
                    bit_index  <= 5'd26;
                    finalize   <= 1'b0;
                    result_exp <= $signed({3'b000, a[30:23]}) -
                                  $signed({3'b000, b[30:23]}) + 11'sd127 -
                                  (normalize_left ? 11'sd1 : 11'sd0);
                end
            end else if (busy && !finalize && !result_pending) begin
                quotient <= quotient_next;
                if (bit_index == 5'd0) begin
                    remainder <= remainder_sub;
                    finalize <= 1'b1;
                end else begin
                    remainder <= remainder_sub << 1;
                    bit_index <= bit_index - 5'd1;
                end
            end else if (busy && finalize) begin
                // Register the exact rounded result before range/status decode.
                // This removes the quotient/remainder carry chain from the
                // output-register control pins at the cost of one fixed cycle.
                rounded_frac_q <= rounded_frac;
                rounded_exp_q <= rounded_exp;
                finalize <= 1'b0;
                result_pending <= 1'b1;
            end else if (busy && result_pending) begin
                busy <= 1'b0;
                done <= 1'b1;
                result_pending <= 1'b0;
                if (rounded_exp_q <= 11'sd0) begin
                    // Subnormal division is outside the resident-Q8 contract.
                    invalid <= 1'b1;
                    out <= 32'd0;
                end else if (rounded_exp_q >= 11'sd255) begin
                    invalid <= 1'b1;
                    out <= 32'h7f80_0000;
                end else begin
                    invalid <= 1'b0;
                    out <= {1'b0, rounded_exp_q[7:0], rounded_frac_q};
                end
            end
        end
    end
endmodule

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
    localparam [3:0] ST_LOAD       = 4'd0;
    localparam [3:0] ST_SCALE_GO   = 4'd1;
    localparam [3:0] ST_SCALE_WAIT = 4'd2;
    localparam [3:0] ST_INV_GO     = 4'd3;
    localparam [3:0] ST_INV_WAIT   = 4'd4;
    localparam [3:0] ST_QUANT_MUL  = 4'd5;
    localparam [3:0] ST_QUANT_NORM = 4'd6;
    localparam [3:0] ST_QUANT_RNE  = 4'd7;
    localparam [3:0] ST_QUANT_OUT  = 4'd8;
    localparam [3:0] ST_HOLD       = 4'd9;
    localparam [3:0] ST_SCALE_CVT  = 4'd10;
    localparam [3:0] ST_SCALE_STAT = 4'd11;
    localparam [3:0] ST_QUANT_FETCH = 4'd12;
    localparam [3:0] ST_QUANT_PIPE = 4'd13;

    localparam [3:0] STATUS_NONFINITE = 4'b0001;
    localparam [3:0] STATUS_SCALE      = 4'b0010;
    localparam [3:0] STATUS_FRAME      = 4'b0100;
    localparam [3:0] STATUS_ARITH      = 4'b1000;

    reg [3:0] state;
    reg [5:0] load_index;
    reg [5:0] quant_index;
    reg [31:0] values [0:31];
    reg [30:0] amax;
    reg        seen_finite_nonzero_q;
    reg        seen_finite_normal_q;
    reg [31:0] scale_f32;
    reg [31:0] inv_scale;
    reg [31:0] quant_value_q;

    wire input_finite = in_data[30:23] != 8'hff;
    wire [30:0] input_abs = in_data[30:0];
    wire [30:0] amax_next = input_finite && input_abs > amax ? input_abs : amax;
    wire seen_finite_nonzero_next = seen_finite_nonzero_q ||
                                    (input_finite && (input_abs != 31'd0));
    wire seen_finite_normal_next = seen_finite_normal_q ||
                                   (input_finite &&
                                    (input_abs[30:23] != 8'd0));
    wire frame_error = in_last != (load_index == 6'd31);
    wire [3:0] load_status_next = out_status |
        (!input_finite ? STATUS_NONFINITE : 4'd0) |
        (frame_error ? STATUS_FRAME : 4'd0);

    assign in_ready = state == ST_LOAD;

    wire div_start = (state == ST_SCALE_GO) || (state == ST_INV_GO);
    wire [31:0] div_a = state == ST_SCALE_GO ? {1'b0, amax} : 32'h3f80_0000;
    wire [31:0] div_b = state == ST_SCALE_GO ? 32'h42fe_0000 : scale_f32;
    wire div_busy;
    wire div_done;
    wire div_invalid;
    wire [31:0] div_out;
    q8_fp32_div_rne_pos u_div (
        .clk(clk), .rst_n(rst_n), .start(div_start), .a(div_a), .b(div_b),
        .busy(div_busy), .done(div_done), .invalid(div_invalid), .out(div_out)
    );

    // Positive finite binary32 -> binary16, round-to-nearest-even. This also
    // returns exact subnormals and infinity so diagnostics can distinguish them.
    function [15:0] fp32_to_f16_rne;
        input [31:0] x;
        reg [7:0] e;
        reg [22:0] f;
        reg [23:0] sig;
        reg [10:0] fraction_rounded;
        reg [10:0] sub_rounded;
        reg [10:0] sub_base;
        reg sub_guard;
        reg sub_sticky;
        reg increment;
        reg [8:0] half_exp;
        begin
            e = x[30:23];
            f = x[22:0];
            sig = {1'b1, f};
            fp32_to_f16_rne = 16'd0;
            if (e == 8'hff) begin
                fp32_to_f16_rne = (f == 23'd0) ? {x[31], 15'h7c00} : {x[31], 15'h7e00};
            end else if (e >= 8'd143) begin
                fp32_to_f16_rne = {x[31], 15'h7c00};
            end else if (e >= 8'd113) begin
                increment = f[12] && ((|f[11:0]) || f[13]);
                fraction_rounded = {1'b0, f[22:13]} +
                                   {{10{1'b0}}, increment};
                half_exp = {1'b0, e} - 9'd112;
                if (fraction_rounded[10]) begin
                    half_exp = half_exp + 9'd1;
                    if (half_exp >= 9'd31)
                        fp32_to_f16_rne = {x[31], 15'h7c00};
                    else
                        fp32_to_f16_rne = {x[31], half_exp[4:0], 10'd0};
                end else begin
                    fp32_to_f16_rne = {x[31], half_exp[4:0], fraction_rounded[9:0]};
                end
            end else if (e >= 8'd102) begin
                sub_base = 11'd0;
                sub_guard = 1'b0;
                sub_sticky = 1'b0;
                case (e)
                    8'd112: begin sub_base = {1'd0, sig[23:14]};  sub_guard = sig[13]; sub_sticky = |sig[12:0]; end
                    8'd111: begin sub_base = {2'd0, sig[23:15]};  sub_guard = sig[14]; sub_sticky = |sig[13:0]; end
                    8'd110: begin sub_base = {3'd0, sig[23:16]};  sub_guard = sig[15]; sub_sticky = |sig[14:0]; end
                    8'd109: begin sub_base = {4'd0, sig[23:17]};  sub_guard = sig[16]; sub_sticky = |sig[15:0]; end
                    8'd108: begin sub_base = {5'd0, sig[23:18]};  sub_guard = sig[17]; sub_sticky = |sig[16:0]; end
                    8'd107: begin sub_base = {6'd0, sig[23:19]};  sub_guard = sig[18]; sub_sticky = |sig[17:0]; end
                    8'd106: begin sub_base = {7'd0, sig[23:20]};  sub_guard = sig[19]; sub_sticky = |sig[18:0]; end
                    8'd105: begin sub_base = {8'd0, sig[23:21]};  sub_guard = sig[20]; sub_sticky = |sig[19:0]; end
                    8'd104: begin sub_base = {9'd0, sig[23:22]};  sub_guard = sig[21]; sub_sticky = |sig[20:0]; end
                    8'd103: begin sub_base = {10'd0, sig[23]};    sub_guard = sig[22]; sub_sticky = |sig[21:0]; end
                    default: begin sub_base = 11'd0;              sub_guard = sig[23]; sub_sticky = |sig[22:0]; end
                endcase
                increment = sub_guard && (sub_sticky || sub_base[0]);
                sub_rounded = sub_base + {{10{1'b0}}, increment};
                if (sub_rounded[10])
                    fp32_to_f16_rne = {x[31], 15'h0400};
                else
                    fp32_to_f16_rne = {x[31], 5'd0, sub_rounded[9:0]};
            end
        end
    endfunction

    wire [15:0] stored_scale = fp32_to_f16_rne(scale_f32);

    // The post-multiply integer rounding has only ten relevant exponent classes:
    // below 0.5, [0.5,1), seven integer-bit positions, and saturation. Spell
    // those out instead of synthesizing a 24-bit variable barrel shift.
    function [8:0] rounded_f32_to_magnitude;
        input [23:0] sig;
        input signed [10:0] exponent;
        reg [8:0] whole;
        reg guard_bit;
        reg sticky_bit;
        reg increment;
        begin
            whole = 9'd0;
            guard_bit = 1'b0;
            sticky_bit = 1'b0;
            if (exponent < 11'sd126) begin
                rounded_f32_to_magnitude = 9'd0;
            end else if (exponent == 11'sd126) begin
                // Exactly 0.5 ties to even zero; every larger significand rounds 1.
                rounded_f32_to_magnitude = sig == 24'h80_0000 ? 9'd0 : 9'd1;
            end else if (exponent >= 11'sd135) begin
                rounded_f32_to_magnitude = 9'h1ff;
            end else begin
                case (exponent)
                    11'sd127: begin whole = {8'd0, sig[23]};    guard_bit = sig[22]; sticky_bit = |sig[21:0]; end
                    11'sd128: begin whole = {7'd0, sig[23:22]}; guard_bit = sig[21]; sticky_bit = |sig[20:0]; end
                    11'sd129: begin whole = {6'd0, sig[23:21]}; guard_bit = sig[20]; sticky_bit = |sig[19:0]; end
                    11'sd130: begin whole = {5'd0, sig[23:20]}; guard_bit = sig[19]; sticky_bit = |sig[18:0]; end
                    11'sd131: begin whole = {4'd0, sig[23:19]}; guard_bit = sig[18]; sticky_bit = |sig[17:0]; end
                    11'sd132: begin whole = {3'd0, sig[23:18]}; guard_bit = sig[17]; sticky_bit = |sig[16:0]; end
                    11'sd133: begin whole = {2'd0, sig[23:17]}; guard_bit = sig[16]; sticky_bit = |sig[15:0]; end
                    default:  begin whole = {1'd0, sig[23:16]}; guard_bit = sig[15]; sticky_bit = |sig[14:0]; end
                endcase
                increment = guard_bit && (sticky_bit || whole[0]);
                rounded_f32_to_magnitude = whole + {{8{1'b0}}, increment};
            end
        end
    endfunction

    wire [31:0] quant_value = values[quant_index[4:0]];
    wire quant_finite = quant_value_q[30:23] != 8'hff;
    wire quant_normal = quant_value_q[30:23] != 8'd0;
    wire inv_normal = !inv_scale[31] &&
                      inv_scale[30:23] != 8'd0 && inv_scale[30:23] != 8'hff;
    wire signed [10:0] quant_exp_pre =
        $signed({3'b000, quant_value_q[30:23]}) +
        $signed({3'b000, inv_scale[30:23]}) - 11'sd127;

    // A pure register after the inferred 24x24 product lets Vivado place a
    // pipeline boundary inside the two-DSP cascade without changing the value.
    (* use_dsp = "yes" *) reg [47:0] quant_product_raw;
    reg [47:0] quant_product;
    reg quant_sign;
    reg signed [10:0] quant_product_exp;
    reg [23:0] quant_rounded_sig;
    reg signed [10:0] quant_rounded_exp;
    reg [8:0] quant_magnitude;

    wire product_renormalizes = quant_product[47];
    wire [23:0] product_sig = product_renormalizes ?
        quant_product[47:24] : quant_product[46:23];
    wire product_guard = product_renormalizes ? quant_product[23] : quant_product[22];
    wire product_sticky = product_renormalizes ?
        |quant_product[22:0] : |quant_product[21:0];
    wire product_round_up = product_guard && (product_sticky || product_sig[0]);
    wire [24:0] product_rounded_ext =
        {1'b0, product_sig} + {{24{1'b0}}, product_round_up};
    wire product_round_overflow = product_rounded_ext[24];
    wire [23:0] product_rounded_sig = product_round_overflow ?
        product_rounded_ext[24:1] : product_rounded_ext[23:0];
    wire signed [10:0] product_rounded_exp = quant_product_exp +
        (product_renormalizes ? 11'sd1 : 11'sd0) +
        (product_round_overflow ? 11'sd1 : 11'sd0);

    wire [7:0] quant_byte = !quant_sign ?
        (quant_magnitude > 9'd127 ? 8'h7f : quant_magnitude[7:0]) :
        (quant_magnitude >= 9'd128 ? 8'h80 : (~quant_magnitude[7:0]) + 8'd1);

    // Product data is intentionally valid-owned and unreset. The FSM waits for
    // both registers before observing quant_product.
    always @(posedge clk) begin
        quant_product_raw <= {1'b1, quant_value_q[22:0]} *
                             {1'b1, inv_scale[22:0]};
        quant_product <= quant_product_raw;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= ST_LOAD;
            load_index  <= 6'd0;
            quant_index <= 6'd0;
            amax        <= 31'd0;
            seen_finite_nonzero_q <= 1'b0;
            seen_finite_normal_q <= 1'b0;
            out_valid   <= 1'b0;
            out_status  <= 4'd0;
        end else begin
            case (state)
                ST_LOAD: begin
                    out_valid <= 1'b0;
                    if (in_valid) begin
                        values[load_index[4:0]] <= in_data;
                        amax <= amax_next;
                        seen_finite_nonzero_q <= seen_finite_nonzero_next;
                        seen_finite_normal_q <= seen_finite_normal_next;
                        out_status <= load_status_next;

                        if (load_index == 6'd31) begin
                            out_quants <= 256'd0;
                            load_index <= 6'd0;
                            if (!seen_finite_nonzero_next) begin
                                out_scale <= 16'd0;
                                out_valid <= 1'b1;
                                state <= ST_HOLD;
                            end else if (!seen_finite_normal_next) begin
                                out_scale <= 16'd0;
                                out_status <= load_status_next | STATUS_ARITH;
                                out_valid <= 1'b1;
                                state <= ST_HOLD;
                            end else begin
                                state <= ST_SCALE_GO;
                            end
                        end else begin
                            load_index <= load_index + 6'd1;
                        end
                    end
                end

                ST_SCALE_GO: if (!div_busy) state <= ST_SCALE_WAIT;

                ST_SCALE_WAIT: if (div_done) begin
                    scale_f32 <= div_out;
                    if (div_invalid) begin
                        out_status <= out_status | STATUS_ARITH;
                        out_valid <= 1'b1;
                        state <= ST_HOLD;
                    end else begin
                        state <= ST_SCALE_CVT;
                    end
                end

                ST_SCALE_CVT: begin
                    out_scale <= stored_scale;
                    state <= ST_SCALE_STAT;
                end

                ST_SCALE_STAT: begin
                    // Zero and subnormal f16 scales are exact inputs to the
                    // GEMM scale decomposer. Only infinity is unsupported.
                    if (out_scale[14:10] == 5'h1f)
                        out_status <= out_status | STATUS_SCALE;
                    state <= ST_INV_GO;
                end

                ST_INV_GO: if (!div_busy) state <= ST_INV_WAIT;

                ST_INV_WAIT: if (div_done) begin
                    inv_scale <= div_out;
                    quant_index <= 6'd0;
                    if (div_invalid) begin
                        out_status <= out_status | STATUS_ARITH;
                        out_valid <= 1'b1;
                        state <= ST_HOLD;
                    end else begin
                        state <= ST_QUANT_FETCH;
                    end
                end

                ST_QUANT_FETCH: begin
                    quant_value_q <= quant_value;
                    state <= ST_QUANT_MUL;
                end

                ST_QUANT_MUL: begin
                    if (!quant_finite || !quant_normal || !inv_normal) begin
                        out_quants[quant_index[4:0] * 8 +: 8] <= 8'd0;
                        if (quant_index == 6'd31) begin
                            out_valid <= 1'b1;
                            state <= ST_HOLD;
                        end else begin
                            quant_index <= quant_index + 6'd1;
                            state <= ST_QUANT_FETCH;
                        end
                    end else begin
                        quant_sign <= quant_value_q[31];
                        quant_product_exp <= quant_exp_pre;
                        state <= ST_QUANT_PIPE;
                    end
                end

                ST_QUANT_PIPE: state <= ST_QUANT_NORM;

                ST_QUANT_NORM: begin
                    quant_rounded_sig <= product_rounded_sig;
                    quant_rounded_exp <= product_rounded_exp;
                    state <= ST_QUANT_RNE;
                end

                ST_QUANT_RNE: begin
                    quant_magnitude <= rounded_f32_to_magnitude(
                        quant_rounded_sig,
                        quant_rounded_exp
                    );
                    state <= ST_QUANT_OUT;
                end

                ST_QUANT_OUT: begin
                    out_quants[quant_index[4:0] * 8 +: 8] <= quant_byte;
                    if (quant_index == 6'd31) begin
                        out_valid <= 1'b1;
                        state <= ST_HOLD;
                    end else begin
                            quant_index <= quant_index + 6'd1;
                            state <= ST_QUANT_FETCH;
                    end
                end

                ST_HOLD: if (out_valid && out_ready) begin
                    state <= ST_LOAD;
                    amax <= 31'd0;
                    seen_finite_nonzero_q <= 1'b0;
                    seen_finite_normal_q <= 1'b0;
                    out_status <= 4'd0;
                    out_valid <= 1'b0;
                end

                default: begin
                    state <= ST_LOAD;
                    load_index <= 6'd0;
                    amax <= 31'd0;
                    seen_finite_nonzero_q <= 1'b0;
                    seen_finite_normal_q <= 1'b0;
                    out_status <= 4'd0;
                    out_valid <= 1'b0;
                end
            endcase
        end
    end

`ifdef FORMAL
    always @(posedge clk) begin
        if (rst_n) begin
            assert(seen_finite_nonzero_q == (amax != 31'd0));
            assert(seen_finite_normal_q == (amax[30:23] != 8'd0));
        end
    end
`endif
endmodule

/* verilator lint_on DECLFILENAME */
`default_nettype wire
