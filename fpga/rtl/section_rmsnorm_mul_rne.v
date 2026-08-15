// Serialized exact finite binary32 multiplier for section RMSNorm.
//
// Every finite operand, including subnormals and signed zero, is multiplied with
// IEEE round-to-nearest-even semantics. Non-finite operands are outside the
// section contract and return deterministic +0 with status. A finite overflow
// returns signed infinity with status. One request may be outstanding; the
// result remains stable under backpressure and abort discards every pipeline
// stage.

`default_nettype none

module section_rmsnorm_mul_rne (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        abort_run,

    output wire        busy,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire [31:0] s_a,
    input  wire [31:0] s_b,

    output wire        result_valid,
    input  wire        result_ready,
    output wire [31:0] result_data,
    // bit 0 NONFINITE_INPUT, bit 1 OVERFLOW.
    output wire [1:0]  result_status
`ifdef FORMAL
    , output wire [2:0] formal_state
    , output wire [23:0] formal_sig_a
    , output wire [23:0] formal_sig_b
    , output wire [47:0] formal_product
    , output wire [8:0] formal_exp_sum
    , output wire formal_sign
    , output wire formal_nonfinite
    , output wire formal_zero
`endif
);
    localparam [1:0] STATUS_NONFINITE = 2'b01;
    localparam [1:0] STATUS_OVERFLOW  = 2'b10;

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_MUL    = 3'd1;
    localparam [2:0] ST_PIPE   = 3'd2;
    localparam [2:0] ST_SCAN   = 3'd3;
    localparam [2:0] ST_SHIFT  = 3'd4;
    localparam [2:0] ST_FINAL  = 3'd5;
    localparam [2:0] ST_RESULT = 3'd6;

    reg [2:0] state_q;

    reg        sign_q;
    reg        nonfinite_q;
    reg        zero_q;
    reg [23:0] sig_a_q;
    reg [23:0] sig_b_q;
    reg [8:0]  exp_sum_q;

    // A pure register after the inferred 24x24 product lets Vivado place a
    // pipeline boundary inside the two-DSP cascade without changing the value.
    (* use_dsp = "yes" *) reg [47:0] product_raw_q;
    reg [47:0] product_q;

    reg signed [10:0] biased_q;
    reg               normal_q;
    reg               shift_left_q;
    reg [8:0]         shift_amount_q;
    reg [23:0]        round_base_q;
    reg               round_up_q;

    reg [31:0] result_data_q;
    reg [1:0]  result_status_q;

    wire [7:0] exp_a = s_a[30:23];
    wire [7:0] exp_b = s_b[30:23];
    wire [22:0] frac_a = s_a[22:0];
    wire [22:0] frac_b = s_b[22:0];
    wire input_nonfinite = (exp_a == 8'hff) || (exp_b == 8'hff);
    wire input_zero = ((exp_a == 8'd0) && (frac_a == 23'd0)) ||
                      ((exp_b == 8'd0) && (frac_b == 23'd0));
    wire [8:0] effective_exp_a = exp_a == 8'd0 ? 9'd1 : {1'b0, exp_a};
    wire [8:0] effective_exp_b = exp_b == 8'd0 ? 9'd1 : {1'b0, exp_b};

    assign busy = state_q != ST_IDLE;
    assign s_ready = rst_n && !abort_run && (state_q == ST_IDLE);
    assign result_valid = rst_n && !abort_run && (state_q == ST_RESULT);
    assign result_data = result_data_q;
    assign result_status = result_status_q;
`ifdef FORMAL
    assign formal_state = state_q;
    assign formal_sig_a = sig_a_q;
    assign formal_sig_b = sig_b_q;
    assign formal_product = product_q;
    assign formal_exp_sum = exp_sum_q;
    assign formal_sign = sign_q;
    assign formal_nonfinite = nonfinite_q;
    assign formal_zero = zero_q;
`endif

    function automatic [5:0] lead_one48(input [47:0] value);
        integer bit_index;
        begin
            lead_one48 = 6'd0;
            for (bit_index = 0; bit_index < 48; bit_index = bit_index + 1)
                if (value[bit_index])
                    lead_one48 = bit_index[5:0];
        end
    endfunction

    // Product data is intentionally valid-owned and unreset. The FSM waits for
    // both registers before observing product_q.
    always @(posedge clk) begin
        product_raw_q <= sig_a_q * sig_b_q;
        product_q <= product_raw_q;
    end

    wire [5:0] scan_lead = lead_one48(product_q);
    wire signed [10:0] scan_exp_sum = $signed({2'b00, exp_sum_q});
    wire signed [10:0] scan_lead_signed = $signed({5'b00000, scan_lead});
    wire signed [10:0] scan_biased = scan_exp_sum +
                                            scan_lead_signed - 11'sd173;
    wire signed [10:0] scan_subnormal_shift = 11'sd151 - scan_exp_sum;
    wire [10:0] scan_subnormal_left_amount =
        $unsigned(-scan_subnormal_shift);

    reg [23:0] shift_base_comb;
    reg        shift_round_up_comb;
    reg [48:0] shift_quotient_comb;
    reg [48:0] shift_remainder_mask_comb;
    reg [48:0] shift_remainder_comb;
    reg [48:0] shift_halfway_comb;
    reg [48:0] shift_left_comb;
    always @(*) begin
        shift_base_comb = 24'd0;
        shift_round_up_comb = 1'b0;
        shift_quotient_comb = 49'd0;
        shift_remainder_mask_comb = 49'd0;
        shift_remainder_comb = 49'd0;
        shift_halfway_comb = 49'd0;
        shift_left_comb = 49'd0;

        if (shift_left_q) begin
            // The reachable normal left shift is at most 23. The wide guard
            // keeps the explicitly handled signed-subnormal branch bounded too.
            if (shift_amount_q >= 9'd49) begin
                shift_base_comb = product_q == 48'd0 ? 24'd0 : 24'h80_0000;
            end else begin
                shift_left_comb = {1'b0, product_q} << shift_amount_q;
                shift_base_comb = |shift_left_comb[48:24] ?
                                  24'h80_0000 : shift_left_comb[23:0];
            end
        end else if (shift_amount_q == 9'd0) begin
            shift_base_comb = product_q[23:0];
        end else if (shift_amount_q <= 9'd48) begin
            shift_quotient_comb = {1'b0, product_q} >> shift_amount_q;
            shift_remainder_mask_comb =
                (49'd1 << shift_amount_q) - 1'b1;
            shift_remainder_comb = {1'b0, product_q} &
                                   shift_remainder_mask_comb;
            shift_halfway_comb = 49'd1 << (shift_amount_q - 1'b1);
            shift_base_comb = shift_quotient_comb[23:0];
            shift_round_up_comb =
                (shift_remainder_comb > shift_halfway_comb) ||
                ((shift_remainder_comb == shift_halfway_comb) &&
                 shift_quotient_comb[0]);
        end
        // A right shift above the exact 48-bit product rounds to zero.
    end

    wire [24:0] rounded_ext = {1'b0, round_base_q} + round_up_q;
    wire normal_round_carry = rounded_ext[24];
    wire [23:0] normal_rounded_sig = normal_round_carry ?
                                     rounded_ext[24:1] : rounded_ext[23:0];
    wire signed [11:0] final_biased =
        $signed({biased_q[10], biased_q}) +
        (normal_round_carry ? 12'sd1 : 12'sd0);

    always @(posedge clk) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            result_data_q <= 32'd0;
            result_status_q <= 2'd0;
        end else if (abort_run) begin
            state_q <= ST_IDLE;
            result_data_q <= 32'd0;
            result_status_q <= 2'd0;
        end else begin
            case (state_q)
                ST_IDLE: if (s_valid) begin
                    sign_q <= s_a[31] ^ s_b[31];
                    nonfinite_q <= input_nonfinite;
                    zero_q <= input_zero;
                    sig_a_q <= {exp_a != 8'd0, frac_a};
                    sig_b_q <= {exp_b != 8'd0, frac_b};
                    exp_sum_q <= effective_exp_a + effective_exp_b;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                    state_q <= ST_MUL;
                end

                ST_MUL: state_q <= ST_PIPE;

                ST_PIPE: state_q <= ST_SCAN;

                ST_SCAN: begin
                    biased_q <= scan_biased;
                    normal_q <= scan_biased >= 11'sd1;
                    if (scan_biased >= 11'sd1) begin
                        if (scan_lead >= 6'd23) begin
                            shift_left_q <= 1'b0;
                            shift_amount_q <= {3'd0, scan_lead} - 9'd23;
                        end else begin
                            shift_left_q <= 1'b1;
                            shift_amount_q <= 9'd23 - {3'd0, scan_lead};
                        end
                    end else if (scan_subnormal_shift > 11'sd0) begin
                        shift_left_q <= 1'b0;
                        shift_amount_q <= scan_subnormal_shift[8:0];
                    end else begin
                        shift_left_q <= 1'b1;
                        shift_amount_q <= scan_subnormal_left_amount[8:0];
                    end
                    state_q <= ST_SHIFT;
                end

                ST_SHIFT: begin
                    round_base_q <= shift_base_comb;
                    round_up_q <= shift_round_up_comb;
                    state_q <= ST_FINAL;
                end

                ST_FINAL: begin
                    if (nonfinite_q) begin
                        result_data_q <= 32'd0;
                        result_status_q <= STATUS_NONFINITE;
                    end else if (zero_q) begin
                        result_data_q <= {sign_q, 31'd0};
                        result_status_q <= 2'd0;
                    end else if (normal_q) begin
                        if (final_biased >= 12'sd255) begin
                            result_data_q <= {sign_q, 8'hff, 23'd0};
                            result_status_q <= STATUS_OVERFLOW;
                        end else begin
                            result_data_q <= {
                                sign_q,
                                final_biased[7:0],
                                normal_rounded_sig[22:0]
                            };
                            result_status_q <= 2'd0;
                        end
                    end else begin
                        // Rounding the largest subnormal may carry exactly into
                        // the minimum normal representation.
                        result_data_q <= |rounded_ext[24:23] ?
                            {sign_q, 8'd1, 23'd0} :
                            {sign_q, 8'd0, rounded_ext[22:0]};
                        result_status_q <= 2'd0;
                    end
                    state_q <= ST_RESULT;
                end

                ST_RESULT: if (result_ready) begin
                    state_q <= ST_IDLE;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                end

                default: begin
                    state_q <= ST_IDLE;
                    result_data_q <= 32'd0;
                    result_status_q <= 2'd0;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
