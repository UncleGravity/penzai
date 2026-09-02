// Fixed Q8_0 block producer for the vector engine.
//
// Every 32 accepted FP32 scalars form one block. Two input banks allow the next
// block to load while the previous block is quantized. Quant bytes use the
// direct ratio
//
//     q = round_even(value * 127 / finite_absmax)
//
// rather than reproducing the host's two rounded FP32 divisions. The stored
// scale is round-to-nearest-even binary16(absmax / 127). This is the intended
// Internal numeric contract: the maximum finite magnitude maps exactly to
// +/-127. It can differ from a two-step FP32 division only for values on a
// quantization rounding boundary.
//
// out_status[0] reports sanitized NaN/Inf input. out_status[1] reports a
// nonzero scale outside the usable binary16 range. A bad scale publishes an
// all-zero record. Payload remains stable while out_valid && !out_ready.

`default_nettype none
/* verilator lint_off DECLFILENAME */

// One exact real-ratio lane. Because |value| <= absmax, only seven quotient
// bits are required. One registered compare/subtract per bit keeps the divider
// at II=1 without a general-purpose divider or reciprocal.
module q8_ratio_lane (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] in_value,
    input  wire [30:0] in_amax,
    input  wire [4:0]  in_index,
    output wire        out_valid,
    output wire [7:0]  out_quant,
    output wire [4:0]  out_index
);
    wire [7:0] value_exp = in_value[30:23];
    wire [7:0] amax_exp = in_amax[30:23];
    wire [23:0] value_sig = {1'b1, in_value[22:0]};
    wire [23:0] amax_sig = {1'b1, in_amax[22:0]};
    wire [7:0] exp_delta = amax_exp - value_exp;

    wire value_normal = (value_exp != 8'd0) && (value_exp != 8'hff);
    wire amax_normal = (amax_exp != 8'd0) && (amax_exp != 8'hff);
    wire ratio_active = value_normal && amax_normal && (exp_delta <= 8'd8);

    // 127*x is a shift/subtract constant multiply. The largest numerator is
    // below 2^31. An exponent delta above eight always rounds to zero.
    wire [30:0] numerator_narrow = {value_sig, 7'd0} -
                                   {{7{1'b0}}, value_sig};
    wire [38:0] numerator = {{8{1'b0}}, numerator_narrow};
    wire [38:0] denominator = {{15{1'b0}}, amax_sig} << exp_delta[3:0];

    // Isolate LUTRAM reads and FP field decoding from the first divide step.
    // The divider remains fully pipelined: this adds latency, not cadence.
    reg [38:0] decoded_numerator_q;
    reg [38:0] decoded_denominator_q;
    reg decoded_sign_q;
    reg decoded_zero_q;
    reg decoded_valid_q;
    reg [4:0] decoded_index_q;

    always @(posedge clk) begin
        if (!rst_n)
            decoded_valid_q <= 1'b0;
        else
            decoded_valid_q <= in_valid;

        decoded_numerator_q <= ratio_active ? numerator : 39'd0;
        decoded_denominator_q <= ratio_active ? denominator : 39'd1;
        decoded_sign_q <= in_value[31];
        decoded_zero_q <= !ratio_active;
        decoded_index_q <= in_index;
    end

    reg [38:0] remainder_q [0:6];
    reg [38:0] denominator_q [0:6];
    reg [6:0] quotient_q [0:6];
    reg sign_q [0:6];
    reg zero_q [0:6];
    reg valid_q [0:6];
    reg [4:0] index_q [0:6];

    genvar stage;
    generate
        for (stage = 0; stage < 7; stage = stage + 1) begin : g_divide
            localparam integer QUOTIENT_BIT = 6 - stage;
            wire [38:0] source_remainder;
            wire [38:0] source_denominator;
            wire [6:0] source_quotient;
            wire source_sign;
            wire source_zero;
            wire source_valid;
            wire [4:0] source_index;

            if (stage == 0) begin : g_first
                assign source_remainder = decoded_numerator_q;
                assign source_denominator = decoded_denominator_q;
                assign source_quotient = 7'd0;
                assign source_sign = decoded_sign_q;
                assign source_zero = decoded_zero_q;
                assign source_valid = decoded_valid_q;
                assign source_index = decoded_index_q;
            end else begin : g_later
                assign source_remainder = remainder_q[stage-1];
                assign source_denominator = denominator_q[stage-1];
                assign source_quotient = quotient_q[stage-1];
                assign source_sign = sign_q[stage-1];
                assign source_zero = zero_q[stage-1];
                assign source_valid = valid_q[stage-1];
                assign source_index = index_q[stage-1];
            end

            wire [38:0] trial = source_denominator << QUOTIENT_BIT;
            wire take = source_remainder >= trial;
            wire [6:0] quotient_bit = 7'd1 << QUOTIENT_BIT;

            always @(posedge clk) begin
                if (!rst_n)
                    valid_q[stage] <= 1'b0;
                else
                    valid_q[stage] <= source_valid;
                remainder_q[stage] <= take ? source_remainder - trial
                                           : source_remainder;
                denominator_q[stage] <= source_denominator;
                quotient_q[stage] <= source_quotient |
                                     (take ? quotient_bit : 7'd0);
                sign_q[stage] <= source_sign;
                zero_q[stage] <= source_zero;
                index_q[stage] <= source_index;
            end
        end
    endgenerate

    wire [39:0] twice_remainder = {remainder_q[6], 1'b0};
    wire [39:0] extended_denominator = {1'b0, denominator_q[6]};
    wire round_up = (twice_remainder > extended_denominator) ||
                    ((twice_remainder == extended_denominator) && quotient_q[6][0]);
    wire [7:0] rounded_magnitude = {1'b0, quotient_q[6]} +
                                   {{7{1'b0}}, round_up};
    wire [7:0] magnitude = rounded_magnitude > 8'd127 ?
                           8'd127 : rounded_magnitude;

    assign out_valid = valid_q[6];
    assign out_index = index_q[6];
    assign out_quant = zero_q[6] ? 8'd0 :
                       (sign_q[6] ? (~magnitude + 8'd1) : magnitude);
endmodule

// Exact binary16 rounding of absmax/127. The eleven-bit significand division
// is iterative and runs in parallel with the two ratio lanes, so it does not
// affect block cadence.
module q8_scale (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [30:0] amax,
    output reg         done,
    output reg  [15:0] scale,
    output reg         bad
);
    reg busy;
    reg [3:0] bit_index;
    reg [39:0] remainder;
    reg [39:0] denominator;
    reg [39:0] trial;
    reg [10:0] quotient;
    reg subnormal;
    reg signed [10:0] half_exp;
    reg        finalize_q;
    reg        round_pending_q;
    reg [39:0] final_remainder_q;
    reg [10:0] final_quotient_q;
    reg [11:0] rounded_q;

    wire [7:0] amax_exp = amax[30:23];
    wire [23:0] amax_sig = {1'b1, amax[22:0]};
    wire needs_exp17 = amax_sig >= 24'd16646144; // 127 * 2^17
    wire signed [10:0] initial_half_exp =
        $signed({3'b000, amax_exp}) + (needs_exp17 ? 11'sd17 : 11'sd16) - 11'sd135;
    // This value is used only for normal FP32 inputs with exponent 108..119.
    wire [5:0] subnormal_shift = 6'd62 - amax_exp[5:0];

    wire take = remainder >= trial;
    wire [39:0] remainder_next = take ? remainder - trial : remainder;
    wire [10:0] quotient_next = quotient |
                                (take ? (11'd1 << bit_index) : 11'd0);
    wire [40:0] twice_remainder = {final_remainder_q, 1'b0};
    wire [40:0] extended_denominator = {1'b0, denominator};
    wire round_up = (twice_remainder > extended_denominator) ||
                    ((twice_remainder == extended_denominator) && final_quotient_q[0]);
    wire [11:0] rounded = {1'b0, final_quotient_q} +
                          {{11{1'b0}}, round_up};

    always @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            scale <= 16'd0;
            bad <= 1'b0;
            finalize_q <= 1'b0;
            round_pending_q <= 1'b0;
            final_remainder_q <= 40'd0;
            final_quotient_q <= 11'd0;
            rounded_q <= 12'd0;
        end else begin
            done <= 1'b0;

            if (finalize_q) begin
                finalize_q <= 1'b0;
                done <= 1'b1;
                if (subnormal) begin
                    if (rounded_q >= 12'd1024)
                        scale <= 16'h0400;
                    else
                        scale <= {6'd0, rounded_q[9:0]};
                    bad <= rounded_q == 12'd0;
                end else if (rounded_q >= 12'd2048) begin
                    if (half_exp >= 11'sd30) begin
                        scale <= 16'h7c00;
                        bad <= 1'b1;
                    end else begin
                        scale <= {1'b0, (half_exp[4:0] + 5'd1), 10'd0};
                    end
                end else begin
                    scale <= {1'b0, half_exp[4:0], rounded_q[9:0]};
                end
            end else if (round_pending_q) begin
                round_pending_q <= 1'b0;
                finalize_q <= 1'b1;
                rounded_q <= rounded;
            end else if (start && !busy) begin
                bad <= 1'b0;
                quotient <= 11'd0;
                bit_index <= 4'd10;
                remainder <= {16'd0, amax_sig};
                half_exp <= initial_half_exp;
                subnormal <= initial_half_exp <= 11'sd0;

                if (amax == 31'd0) begin
                    scale <= 16'd0;
                    done <= 1'b1;
                end else if ((amax_exp == 8'd0) || (amax_exp == 8'hff)) begin
                    scale <= 16'd0;
                    bad <= 1'b1;
                    done <= 1'b1;
                end else if (initial_half_exp >= 11'sd31) begin
                    scale <= 16'h7c00;
                    bad <= 1'b1;
                    done <= 1'b1;
                end else if ((initial_half_exp <= 11'sd0) &&
                             (amax_exp < 8'd108)) begin
                    // The correctly rounded binary16 scale is zero.
                    scale <= 16'd0;
                    bad <= 1'b1;
                    done <= 1'b1;
                end else begin
                    busy <= 1'b1;
                    if (initial_half_exp <= 11'sd0)
                        denominator <= 40'd127 << subnormal_shift;
                    else if (needs_exp17)
                        denominator <= 40'd16256; // 127 * 2^7
                    else
                        denominator <= 40'd8128;  // 127 * 2^6

                    if (initial_half_exp <= 11'sd0)
                        trial <= (40'd127 << subnormal_shift) << 10;
                    else if (needs_exp17)
                        trial <= 40'd16256 << 10;
                    else
                        trial <= 40'd8128 << 10;
                end
            end else if (busy) begin
                remainder <= remainder_next;
                quotient <= quotient_next;
                trial <= trial >> 1;

                if (bit_index == 4'd0) begin
                    busy <= 1'b0;
                    round_pending_q <= 1'b1;
                    final_remainder_q <= remainder_next;
                    final_quotient_q <= quotient_next;
                end else begin
                    bit_index <= bit_index - 4'd1;
                end
            end
        end
    end
endmodule

module q8_block (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         in_valid,
    output wire         in_ready,
    input  wire [31:0]  in_data,

    output reg          out_valid,
    input  wire         out_ready,
    output reg  [255:0] out_quants,
    output reg  [15:0]  out_scale,
    output reg  [1:0]   out_status
);
    reg [31:0] values0 [0:31];
    reg [31:0] values1 [0:31];

    reg fill_bank;
    reg [4:0] fill_index;
    reg [30:0] fill_amax;
    reg fill_nonfinite;

    reg bank0_full;
    reg bank1_full;
    reg [30:0] bank0_amax;
    reg [30:0] bank1_amax;
    reg bank0_nonfinite;
    reg bank1_nonfinite;

    wire input_finite = in_data[30:23] != 8'hff;
    wire [31:0] sanitized_input = input_finite ? in_data : 32'd0;
    wire [30:0] input_abs = sanitized_input[30:0];
    wire [30:0] next_amax = input_abs > fill_amax ? input_abs : fill_amax;
    wire input_fire = in_valid && in_ready;

    assign in_ready = fill_bank ? !bank1_full : !bank0_full;

    reg proc_busy;
    reg proc_bank;
    reg proc_next_bank;
    reg [30:0] proc_amax;
    reg proc_nonfinite;
    reg feeding;
    reg [3:0] feed_pair;
    reg [255:0] work_quants;
    reg quant_done;
    reg scale_ready;
    reg [15:0] work_scale;
    reg work_scale_bad;
    reg scale_start;

    wire any_full = bank0_full || bank1_full;
    wire preferred_full = proc_next_bank ? bank1_full : bank0_full;
    wire selected_bank = preferred_full ? proc_next_bank : ~proc_next_bank;
    wire start_process = !proc_busy && any_full;

    wire [4:0] feed_index0 = {feed_pair, 1'b0};
    wire [4:0] feed_index1 = {feed_pair, 1'b0} + 5'd1;
    wire [31:0] feed_value0 = proc_bank ? values1[feed_index0] : values0[feed_index0];
    wire [31:0] feed_value1 = proc_bank ? values1[feed_index1] : values0[feed_index1];
    wire ratio_in_valid = proc_busy && feeding;

    wire ratio0_valid;
    wire [7:0] ratio0_quant;
    wire [4:0] ratio0_index;
    wire ratio1_valid;
    wire [7:0] ratio1_quant;
    wire [4:0] ratio1_index;

    q8_ratio_lane u_ratio0 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(ratio_in_valid), .in_value(feed_value0),
        .in_amax(proc_amax), .in_index(feed_index0),
        .out_valid(ratio0_valid), .out_quant(ratio0_quant),
        .out_index(ratio0_index)
    );
    q8_ratio_lane u_ratio1 (
        .clk(clk), .rst_n(rst_n),
        .in_valid(ratio_in_valid), .in_value(feed_value1),
        .in_amax(proc_amax), .in_index(feed_index1),
        .out_valid(ratio1_valid), .out_quant(ratio1_quant),
        .out_index(ratio1_index)
    );

    wire scale_done;
    wire [15:0] scale_value;
    wire scale_bad;
    q8_scale u_scale (
        .clk(clk), .rst_n(rst_n), .start(scale_start), .amax(proc_amax),
        .done(scale_done), .scale(scale_value), .bad(scale_bad)
    );

    wire output_slot_free = !out_valid || out_ready;
    wire publish = proc_busy && quant_done && scale_ready && output_slot_free;

    always @(posedge clk) begin
        if (!rst_n) begin
            fill_bank <= 1'b0;
            fill_index <= 5'd0;
            fill_amax <= 31'd0;
            fill_nonfinite <= 1'b0;
            bank0_full <= 1'b0;
            bank1_full <= 1'b0;
            proc_busy <= 1'b0;
            proc_bank <= 1'b0;
            proc_next_bank <= 1'b0;
            proc_amax <= 31'd0;
            proc_nonfinite <= 1'b0;
            feeding <= 1'b0;
            feed_pair <= 4'd0;
            work_quants <= 256'd0;
            quant_done <= 1'b0;
            scale_ready <= 1'b0;
            work_scale <= 16'd0;
            work_scale_bad <= 1'b0;
            scale_start <= 1'b0;
            out_valid <= 1'b0;
            out_quants <= 256'd0;
            out_scale <= 16'd0;
            out_status <= 2'd0;
        end else begin
            scale_start <= 1'b0;

            if (out_valid && out_ready)
                out_valid <= 1'b0;

            if (input_fire) begin
                if (fill_bank)
                    values1[fill_index] <= sanitized_input;
                else
                    values0[fill_index] <= sanitized_input;

                if (fill_index == 5'd31) begin
                    if (fill_bank) begin
                        bank1_full <= 1'b1;
                        bank1_amax <= next_amax;
                        bank1_nonfinite <= fill_nonfinite || !input_finite;
                    end else begin
                        bank0_full <= 1'b1;
                        bank0_amax <= next_amax;
                        bank0_nonfinite <= fill_nonfinite || !input_finite;
                    end
                    fill_bank <= ~fill_bank;
                    fill_index <= 5'd0;
                    fill_amax <= 31'd0;
                    fill_nonfinite <= 1'b0;
                end else begin
                    fill_index <= fill_index + 5'd1;
                    fill_amax <= next_amax;
                    fill_nonfinite <= fill_nonfinite || !input_finite;
                end
            end

            if (start_process) begin
                proc_busy <= 1'b1;
                proc_bank <= selected_bank;
                proc_next_bank <= ~selected_bank;
                proc_amax <= selected_bank ? bank1_amax : bank0_amax;
                proc_nonfinite <= selected_bank ? bank1_nonfinite : bank0_nonfinite;
                feeding <= 1'b1;
                feed_pair <= 4'd0;
                work_quants <= 256'd0;
                quant_done <= 1'b0;
                scale_ready <= 1'b0;
                work_scale_bad <= 1'b0;
                scale_start <= 1'b1;
            end

            if (proc_busy && feeding) begin
                if (feed_pair == 4'd15) begin
                    feeding <= 1'b0;
                    if (proc_bank)
                        bank1_full <= 1'b0;
                    else
                        bank0_full <= 1'b0;
                end else begin
                    feed_pair <= feed_pair + 4'd1;
                end
            end

            if (ratio0_valid && ratio1_valid) begin
                work_quants[ratio0_index * 8 +: 8] <= ratio0_quant;
                work_quants[ratio1_index * 8 +: 8] <= ratio1_quant;
                if (ratio1_index == 5'd31)
                    quant_done <= 1'b1;
            end

            if (scale_done) begin
                scale_ready <= 1'b1;
                work_scale <= scale_value;
                work_scale_bad <= scale_bad;
            end

            if (publish) begin
                out_valid <= 1'b1;
                out_quants <= work_scale_bad ? 256'd0 : work_quants;
                out_scale <= work_scale;
                out_status <= {work_scale_bad, proc_nonfinite};
                proc_busy <= 1'b0;
                quant_done <= 1'b0;
                scale_ready <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
