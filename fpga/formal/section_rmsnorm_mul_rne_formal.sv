`default_nettype none

// Independent arithmetic and lifecycle proof boundary for the serialized
// binary32 RNE multiplier. The product task proves the registered product and
// captured metadata independently. The data task then passes that proven exact
// 48-bit product to a u64 packer that shares no normalization logic with RTL.
module section_rmsnorm_mul_rne_formal(input wire clk);
    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_MUL    = 3'd1;
    localparam [2:0] ST_PIPE   = 3'd2;
    localparam [2:0] ST_SCAN   = 3'd3;
    localparam [2:0] ST_SHIFT  = 3'd4;
    localparam [2:0] ST_FINAL  = 3'd5;
    localparam [2:0] ST_RESULT = 3'd6;

    localparam [1:0] STATUS_OK        = 2'b00;
    localparam [1:0] STATUS_NONFINITE = 2'b01;
    localparam [1:0] STATUS_OVERFLOW  = 2'b10;

    localparam [3:0] SC_NORMAL       = 4'd0;
    localparam [3:0] SC_SUBNORMAL    = 4'd1;
    localparam [3:0] SC_UNDERFLOW    = 4'd2;
    localparam [3:0] SC_SUB_TO_NORM  = 4'd3;
    localparam [3:0] SC_OVERFLOW     = 4'd4;
    localparam [3:0] SC_NONFINITE    = 4'd5;
    localparam [3:0] SC_STALL        = 4'd6;
    localparam [3:0] SC_ABORT        = 4'd7;
    localparam [3:0] SC_BACK_TO_BACK = 4'd8;

    // Return {status, result}. All temporaries are wider than binary32 so
    // signed scale arithmetic, shift 48 ties, and subnormal left shifts are
    // explicit rather than relying on host floating-point behavior.
    function automatic [33:0] mul_rne_pack_reference(
        input [31:0] a,
        input [31:0] b,
        input [47:0] exact_product
    );
        reg sign;
        reg [7:0] exp_a;
        reg [7:0] exp_b;
        reg [22:0] frac_a;
        reg [22:0] frac_b;
        reg [47:0] product;
        reg [63:0] wide_product;
        reg [63:0] quotient;
        reg [63:0] remainder;
        reg [63:0] remainder_mask;
        reg [63:0] halfway;
        reg [63:0] rounded;
        reg round_up;
        integer scale_a;
        integer scale_b;
        integer scale;
        integer msb;
        integer unbiased;
        integer biased;
        integer shift;
        integer left_shift;
        integer i;
        begin
            sign = a[31] ^ b[31];
            exp_a = a[30:23];
            exp_b = b[30:23];
            frac_a = a[22:0];
            frac_b = b[22:0];

            product = exact_product;
            wide_product = 64'd0;
            quotient = 64'd0;
            remainder = 64'd0;
            remainder_mask = 64'd0;
            halfway = 64'd0;
            rounded = 64'd0;
            round_up = 1'b0;
            scale_a = 0;
            scale_b = 0;
            scale = 0;
            msb = -1;
            unbiased = 0;
            biased = 0;
            shift = 0;
            left_shift = 0;

            // A nonfinite operand dominates every finite special case,
            // including zero times infinity, and produces deterministic +0.
            if (exp_a == 8'hff || exp_b == 8'hff) begin
                mul_rne_pack_reference = {
                    STATUS_NONFINITE, 32'h0000_0000
                };
            end else if ((exp_a == 0 && frac_a == 0) ||
                         (exp_b == 0 && frac_b == 0)) begin
                mul_rne_pack_reference = {STATUS_OK, sign, 31'd0};
            end else begin
                scale_a = exp_a == 0 ? -149 : exp_a - 150;
                scale_b = exp_b == 0 ? -149 : exp_b - 150;
                scale = scale_a + scale_b;
                wide_product = {16'd0, product};
                for (i = 0; i < 48; i = i + 1)
                    if (product[i])
                        msb = i;
                unbiased = scale + msb;

                if (unbiased >= -126) begin
                    // Round the exact product to a 24-bit normal significand.
                    shift = msb - 23;
                    if (shift > 0) begin
                        quotient = wide_product >> shift;
                        remainder_mask = (64'd1 << shift) - 1'b1;
                        remainder = wide_product & remainder_mask;
                        halfway = 64'd1 << (shift - 1);
                        round_up = (remainder > halfway) ||
                                   ((remainder == halfway) && quotient[0]);
                        rounded = quotient + round_up;
                    end else begin
                        left_shift = -shift;
                        rounded = wide_product << left_shift;
                    end

                    if (rounded[24]) begin
                        rounded = rounded >> 1;
                        unbiased = unbiased + 1;
                    end
                    if (unbiased >= 128) begin
                        mul_rne_pack_reference = {
                            STATUS_OVERFLOW, sign, 8'hff, 23'd0
                        };
                    end else begin
                        biased = unbiased + 127;
                        mul_rne_pack_reference = {
                            STATUS_OK, sign, biased[7:0], rounded[22:0]
                        };
                    end
                end else begin
                    // Round directly in units of the minimum subnormal 2^-149.
                    // The wide branches make shifts >=64 and <=0 well-defined;
                    // shift==48 retains the exact halfway bit in the u64 oracle.
                    shift = -(scale + 149);
                    if (shift <= 0) begin
                        left_shift = -shift;
                        rounded = wide_product << left_shift;
                    end else if (shift >= 64) begin
                        rounded = 64'd0;
                    end else begin
                        quotient = wide_product >> shift;
                        remainder_mask = (64'd1 << shift) - 1'b1;
                        remainder = wide_product & remainder_mask;
                        halfway = 64'd1 << (shift - 1);
                        round_up = (remainder > halfway) ||
                                   ((remainder == halfway) && quotient[0]);
                        rounded = quotient + round_up;
                    end

                    if (rounded >= 64'h0000_0000_0080_0000) begin
                        mul_rne_pack_reference = {
                            STATUS_OK, sign, 8'd1, 23'd0
                        };
                    end else begin
                        mul_rne_pack_reference = {
                            STATUS_OK, sign, 8'd0, rounded[22:0]
                        };
                    end
                end
            end
        end
    endfunction

    reg f_past_valid = 1'b0;
    wire rst_n = f_past_valid;

    (* anyseq *) reg f_s_valid;
    (* anyseq *) reg f_result_ready;
    (* anyseq *) reg f_abort_run;
    (* anyconst *) reg [31:0] f_operand_a;
    (* anyconst *) reg [31:0] f_operand_b;
    (* anyconst *) reg [3:0] f_scenario;
    (* anyconst *) reg [2:0] f_abort_phase;

    reg data_want_input_q = 1'b1;
    reg cover_want_input_q = 1'b1;
    reg cover_saw_stall_q = 1'b0;
    reg cover_saw_abort_q = 1'b0;
    reg cover_restarted_q = 1'b0;
    reg [2:0] cover_accepts_q = 3'd0;
    reg [2:0] cover_retires_q = 3'd0;

    wire busy;
    wire s_ready;
    wire result_valid;
    wire [31:0] result_data;
    wire [1:0] result_status;
    wire [2:0] formal_state;
    wire [23:0] formal_sig_a;
    wire [23:0] formal_sig_b;
    wire [47:0] formal_product;
    wire [8:0] formal_exp_sum;
    wire formal_sign;
    wire formal_nonfinite;
    wire formal_zero;

`ifdef FORMAL_CONTROL
    wire s_valid = f_s_valid;
    wire [31:0] s_a = 32'h3fc0_0000;
    wire [31:0] s_b = 32'hc000_0000;
    wire result_ready = f_result_ready;
    wire abort_run = f_abort_run;
`elsif FORMAL_DATA
    wire s_valid = rst_n && data_want_input_q;
    wire [31:0] s_a = f_operand_a;
    wire [31:0] s_b = f_operand_b;
    // Control PDR separately leaves result_ready arbitrary and proves held
    // output stability. Keeping it asserted here isolates the one arbitrary
    // 64-bit arithmetic transaction and its fixed C0-to-result latency.
    wire result_ready = 1'b1;
    wire abort_run = 1'b0;
`elsif FORMAL_PRODUCT
    wire s_valid = rst_n && data_want_input_q;
    wire [31:0] s_a = f_operand_a;
    wire [31:0] s_b = f_operand_b;
    wire result_ready = 1'b1;
    wire abort_run = 1'b0;
`else
    wire cover_after_abort = cover_saw_abort_q && !cover_restarted_q;
    wire [31:0] scenario_a =
        (f_scenario == SC_SUBNORMAL)   ? 32'h0080_0000 :
        (f_scenario == SC_UNDERFLOW)   ? 32'h8000_0001 :
        (f_scenario == SC_SUB_TO_NORM) ? 32'h007f_ffff :
        (f_scenario == SC_OVERFLOW)    ? 32'h7f7f_ffff :
        (f_scenario == SC_NONFINITE)   ? 32'h0000_0000 :
                                          32'h3fc0_0000;
    wire [31:0] scenario_b =
        (f_scenario == SC_SUBNORMAL)   ? 32'h3f00_0000 :
        (f_scenario == SC_UNDERFLOW)   ? 32'h3f00_0000 :
        (f_scenario == SC_SUB_TO_NORM) ? 32'h3f80_0001 :
        (f_scenario == SC_OVERFLOW)    ? 32'h4000_0000 :
        (f_scenario == SC_NONFINITE)   ? 32'hff80_0000 :
                                          32'hc000_0000;
    wire s_valid = rst_n && cover_want_input_q;
    wire [31:0] s_a = cover_after_abort ? 32'h3fc0_0000 : scenario_a;
    wire [31:0] s_b = cover_after_abort ? 32'hc000_0000 : scenario_b;
    wire result_ready = (f_scenario == SC_STALL) ? cover_saw_stall_q : 1'b1;
    wire abort_run = rst_n && (f_scenario == SC_ABORT) &&
                     !cover_saw_abort_q && busy &&
                     (formal_state == f_abort_phase);
`endif

    wire input_fire = s_valid && s_ready;
    wire result_fire = result_valid && result_ready;

    section_rmsnorm_mul_rne dut (
        .clk(clk), .rst_n(rst_n), .abort_run(abort_run), .busy(busy),
        .s_valid(s_valid), .s_ready(s_ready), .s_a(s_a), .s_b(s_b),
        .result_valid(result_valid), .result_ready(result_ready),
        .result_data(result_data), .result_status(result_status),
        .formal_state(formal_state),
        .formal_sig_a(formal_sig_a), .formal_sig_b(formal_sig_b),
        .formal_product(formal_product), .formal_exp_sum(formal_exp_sum),
        .formal_sign(formal_sign), .formal_nonfinite(formal_nonfinite),
        .formal_zero(formal_zero)
    );

    reg model_pending_q = 1'b0;
    reg [31:0] model_a_q = 32'd0;
    reg [31:0] model_b_q = 32'd0;
    reg [3:0] model_age_q = 4'd0;
    wire [33:0] expected = mul_rne_pack_reference(
        model_a_q, model_b_q, formal_product
    );
    wire [23:0] model_sig_a = {
        model_a_q[30:23] != 8'd0, model_a_q[22:0]
    };
    wire [23:0] model_sig_b = {
        model_b_q[30:23] != 8'd0, model_b_q[22:0]
    };
    wire [8:0] model_effective_exp_a = model_a_q[30:23] == 8'd0 ?
        9'd1 : {1'b0, model_a_q[30:23]};
    wire [8:0] model_effective_exp_b = model_b_q[30:23] == 8'd0 ?
        9'd1 : {1'b0, model_b_q[30:23]};
    wire model_nonfinite = (model_a_q[30:23] == 8'hff) ||
                           (model_b_q[30:23] == 8'hff);
    wire model_zero = ((model_a_q[30:23] == 8'd0) &&
                       (model_a_q[22:0] == 23'd0)) ||
                      ((model_b_q[30:23] == 8'd0) &&
                       (model_b_q[22:0] == 23'd0));

    always @(posedge clk) begin
        f_past_valid <= 1'b1;

`ifdef FORMAL_COVER
        assume(f_scenario <= SC_BACK_TO_BACK);
        assume(f_abort_phase >= ST_MUL && f_abort_phase <= ST_RESULT);
`endif

        if (!rst_n) begin
            data_want_input_q <= 1'b1;
            cover_want_input_q <= 1'b1;
            cover_saw_stall_q <= 1'b0;
            cover_saw_abort_q <= 1'b0;
            cover_restarted_q <= 1'b0;
            cover_accepts_q <= 3'd0;
            cover_retires_q <= 3'd0;
            model_pending_q <= 1'b0;
            model_a_q <= 32'd0;
            model_b_q <= 32'd0;
            model_age_q <= 4'd0;
        end else begin
            if (input_fire) begin
                data_want_input_q <= 1'b0;
                cover_want_input_q <= 1'b0;
                cover_accepts_q <= cover_accepts_q + 1'b1;
                if (cover_saw_abort_q)
                    cover_restarted_q <= 1'b1;
                model_pending_q <= 1'b1;
                model_a_q <= s_a;
                model_b_q <= s_b;
                model_age_q <= 4'd0;
            end else if (model_pending_q && !result_valid) begin
                model_age_q <= model_age_q + 1'b1;
            end

            if (result_fire) begin
                model_pending_q <= 1'b0;
                cover_retires_q <= cover_retires_q + 1'b1;
                if (f_scenario == SC_BACK_TO_BACK &&
                    cover_retires_q == 0)
                    cover_want_input_q <= 1'b1;
            end

            if (result_valid && !result_ready)
                cover_saw_stall_q <= 1'b1;

            if (abort_run) begin
                cover_saw_abort_q <= 1'b1;
                cover_want_input_q <= 1'b1;
                model_pending_q <= 1'b0;
                model_age_q <= 4'd0;
            end
        end

        if (rst_n) begin
            assert(formal_state <= ST_RESULT);
            assert(busy == (formal_state != ST_IDLE));
            assert(s_ready == (!abort_run && !busy));
            assert(result_valid == (!abort_run &&
                                    formal_state == ST_RESULT));
            assert(result_status != 2'b11);
            assert(!(s_ready && result_valid));
            if (busy)
                assert(!s_ready);
            if (input_fire)
                assert(!model_pending_q);
            if (result_valid) begin
                assert(model_pending_q);
                assert(model_age_q == 4'd5);
`ifdef FORMAL_CONTROL
                assert(result_data == 32'hc040_0000);
                assert(result_status == STATUS_OK);
`elsif FORMAL_PRODUCT
                // Product exactness and metadata capture are the only numeric
                // guarantees in this task. Packing is proved by FORMAL_DATA.
`else
                assert({result_status, result_data} == expected);
`endif
            end
        end

`ifdef FORMAL_PRODUCT
        if (rst_n && model_pending_q) begin
            assert(formal_sig_a == model_sig_a);
            assert(formal_sig_b == model_sig_b);
            assert(formal_exp_sum ==
                   model_effective_exp_a + model_effective_exp_b);
            assert(formal_sign == (model_a_q[31] ^ model_b_q[31]));
            assert(formal_nonfinite == model_nonfinite);
            assert(formal_zero == model_zero);
            if (formal_state >= ST_SCAN)
                assert(formal_product == formal_sig_a * formal_sig_b);
        end
`endif

        // These transitions prove the fixed C0 -> post-C5 latency and that a
        // result stall cannot admit a second request.
        if (f_past_valid && $past(rst_n)) begin
            if ($past(abort_run)) begin
                assert(formal_state == ST_IDLE);
                assert(!busy);
                assert(!result_valid);
            end else begin
                case ($past(formal_state))
                    ST_IDLE: begin
                        if ($past(input_fire))
                            assert(formal_state == ST_MUL);
                        else
                            assert(formal_state == ST_IDLE);
                    end
                    ST_MUL:    assert(formal_state == ST_PIPE);
                    ST_PIPE:   assert(formal_state == ST_SCAN);
                    ST_SCAN:   assert(formal_state == ST_SHIFT);
                    ST_SHIFT:  assert(formal_state == ST_FINAL);
                    ST_FINAL:  assert(formal_state == ST_RESULT);
                    ST_RESULT: begin
                        if ($past(result_fire))
                            assert(formal_state == ST_IDLE);
                        else
                            assert(formal_state == ST_RESULT);
                    end
                    default: assert(1'b0);
                endcase
            end
        end

        if (f_past_valid && rst_n && !abort_run &&
            $past(rst_n && !abort_run && result_valid && !result_ready)) begin
            assert(result_valid);
            assert(result_data == $past(result_data));
            assert(result_status == $past(result_status));
        end

`ifdef FORMAL_COVER
        cover(rst_n && f_scenario == SC_NORMAL && result_fire &&
              result_data == 32'hc040_0000 && result_status == STATUS_OK);
        cover(rst_n && f_scenario == SC_SUBNORMAL && result_fire &&
              result_data == 32'h0040_0000 && result_status == STATUS_OK);
        cover(rst_n && f_scenario == SC_UNDERFLOW && result_fire &&
              result_data == 32'h8000_0000 && result_status == STATUS_OK);
        cover(rst_n && f_scenario == SC_SUB_TO_NORM && result_fire &&
              result_data == 32'h0080_0000 && result_status == STATUS_OK);
        cover(rst_n && f_scenario == SC_OVERFLOW && result_fire &&
              result_data == 32'h7f80_0000 &&
              result_status == STATUS_OVERFLOW);
        cover(rst_n && f_scenario == SC_NONFINITE && result_fire &&
              result_data == 32'h0000_0000 &&
              result_status == STATUS_NONFINITE);
        cover(rst_n && f_scenario == SC_STALL && cover_saw_stall_q &&
              cover_retires_q == 1);
        cover(rst_n && f_scenario == SC_BACK_TO_BACK &&
              cover_accepts_q == 2 && cover_retires_q == 2);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_MUL);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_PIPE);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_SCAN);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_SHIFT);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_FINAL);
        cover(rst_n && f_scenario == SC_ABORT && cover_saw_abort_q &&
              cover_restarted_q && cover_retires_q == 1 &&
              f_abort_phase == ST_RESULT);
`endif
    end
endmodule

`default_nettype wire
