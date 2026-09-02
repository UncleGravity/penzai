// Exact base-2^16 accumulation for the tile-8, four-lane projection engine.
//
// A scale product is split into seven redundant radix digits.  Each digit is
// accumulated independently, avoiding both a 104-bit barrel shifter and a
// 104-bit carry recurrence in every row/lane.  The maximum supported K=12288
// contributes 384 dot records.  A 16-bit unsigned chunk therefore sums to at
// most 384*65535=25165440; signed 26-bit bins cover that positive bound and
// the signed high-chunk bound without overflow.

`default_nettype none

module digit_mul #(
    parameter integer SIG_W   = 12,
    parameter integer SUM_W   = 14,
    parameter integer EXP_W   = 8,
    parameter integer DIGITS  = 7,
    parameter integer BIN_W   = 26
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         valid_in,
    input  wire signed [SIG_W-1:0]      ws_sig,
    input  wire signed [SIG_W-1:0]      as_sig,
    input  wire signed [EXP_W-1:0]      p_exp,
    input  wire signed [EXP_W-1:0]      emin,
    input  wire signed [SUM_W-1:0]      dot_sum,
    output reg                          valid_out,
    output reg signed [EXP_W:0]         coarse_digit,
    output reg [63:0]                   fine_chunks
);
    localparam integer PSIG_W = 2*SIG_W;
    // gemm_f16_decompose magnitudes are at most 2048 and a 32-element
    // signed-Q8 dot is in [-4096, 4064].  Their exact product therefore fits
    // signed 35 bits, including the -2^34 endpoint.
    localparam integer PROD_W = 35;
    localparam integer FINE_W = 64;

    initial begin
        if (DIGITS != 7 || BIN_W != 26 || SIG_W != 12 || SUM_W != 14)
            $error(" digit_mul parameter set is unsupported");
    end

    reg v0;
    reg signed [SIG_W-1:0] ws0;
    reg signed [SIG_W-1:0] as0;
    reg signed [EXP_W-1:0] pexp0;
    reg signed [EXP_W-1:0] emin0;
    reg signed [SUM_W-1:0] sum0;
    always @(posedge clk) begin
        if (!rst_n) begin
            v0 <= 1'b0;
        end else begin
            v0 <= valid_in;
            ws0 <= ws_sig;
            as0 <= as_sig;
            pexp0 <= p_exp;
            emin0 <= emin;
            sum0 <= dot_sum;
        end
    end

    reg v1;
    (* use_dsp = "yes" *) reg signed [PSIG_W-1:0] psig1;
    reg signed [EXP_W:0] shift1;
    reg signed [SUM_W-1:0] sum1;
    always @(posedge clk) begin
        if (!rst_n) begin
            v1 <= 1'b0;
        end else begin
            v1 <= v0;
            psig1 <= ws0 * as0;
            shift1 <= $signed({pexp0[EXP_W-1], pexp0}) -
                      $signed({emin0[EXP_W-1], emin0});
            sum1 <= sum0;
        end
    end

    reg v2;
    (* use_dsp = "yes" *) reg signed [PROD_W-1:0] product2;
    reg signed [EXP_W:0] coarse2;
    reg [3:0] fine_shift2;
    always @(posedge clk) begin
        if (!rst_n) begin
            v2 <= 1'b0;
        end else begin
            v2 <= v1;
            product2 <= psig1 * sum1;
            // Arithmetic division by 16 plus a non-negative remainder gives
            // the same result for positive and negative window shifts.
            coarse2 <= shift1 >>> 4;
            fine_shift2 <= shift1[3:0];
        end
    end

    wire signed [FINE_W-1:0] fine_product =
        $signed({{(FINE_W-PROD_W){product2[PROD_W-1]}}, product2})
            <<< fine_shift2;

    // The consumer stores digits by index modulo four.  Keeping the four
    // consecutive chunks plus their coarse digit is much smaller than
    // building a seven-way shifted vector in every multiplier lane.
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
        end else begin
            valid_out <= v2;
            coarse_digit <= coarse2;
            fine_chunks <= fine_product;
        end
    end

`ifdef FORMAL
    always @(posedge clk)
        if (rst_n && valid_in) begin
            assert(dot_sum >= -14'sd4096);
            assert(dot_sum <= 14'sd4064);
        end
`endif
endmodule

// Sixteen resident records per physical row: two rowblock banks by eight
// logical tokens.  There is one update port and an independent asynchronous
// drain read.  The payload has no reset; clear replaces every digit on the
// first K record.
module digit_cell #(
    parameter integer DIGITS = 7,
    parameter integer BIN_W  = 26
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          clear,
    input  wire                          update_valid,
    input  wire                          update_clear,
    input  wire                          update_bank,
    input  wire [2:0]                    update_token,
    input  wire signed [8:0]             update_coarse,
    input  wire [63:0]                   update_chunks,
    input  wire                          drain_bank,
    input  wire [2:0]                    drain_token,
    output wire signed [DIGITS*BIN_W-1:0] drain_digits,
    output wire                          update_pending
);
    wire [3:0] update_addr = {update_bank, update_token};
    wire [3:0] drain_addr = {drain_bank, drain_token};

    // Consecutive source chunks map one-to-one onto these four modulo banks.
    // A bank therefore needs only one adder and one record write per beat.
    (* ram_style = "distributed" *) reg [2*BIN_W-1:0] records0 [0:15];
    (* ram_style = "distributed" *) reg [2*BIN_W-1:0] records1 [0:15];
    (* ram_style = "distributed" *) reg [2*BIN_W-1:0] records2 [0:15];
    (* ram_style = "distributed" *) reg [BIN_W-1:0]   records3 [0:15];

    reg [1:0] source0, source1, source2, source3;
    reg signed [BIN_W-1:0] add0, add1, add2, add3;
    reg valid0, valid1, valid2, valid3;
    reg high0, high1, high2;
    reg update_valid_q;
    reg update_clear_q;
    // Keep one address register beside each LUTRAM group.  A single copy drives
    // hundreds of RAMD32 address pins after flattening and becomes a long route
    // into the read-modify-write carry chain.
    (* keep = "true" *) reg [3:0] update_addr0_q;
    (* keep = "true" *) reg [3:0] update_addr1_q;
    (* keep = "true" *) reg [3:0] update_addr2_q;
    (* keep = "true" *) reg [3:0] update_addr3_q;
    reg signed [BIN_W-1:0] add0_q, add1_q, add2_q, add3_q;
    reg valid0_q, valid1_q, valid2_q, valid3_q;
    reg high0_q, high1_q, high2_q;
    reg [2*BIN_W-1:0] next0, next1, next2;
    reg [BIN_W-1:0] next3;

    function signed [BIN_W-1:0] extend_chunk;
        input [1:0] source;
        reg [15:0] raw;
        begin
            case (source)
                2'd0: raw = update_chunks[15:0];
                2'd1: raw = update_chunks[31:16];
                2'd2: raw = update_chunks[47:32];
                default: raw = update_chunks[63:48];
            endcase
            extend_chunk = (source == 2'd3) ?
                {{(BIN_W-16){raw[15]}}, raw} :
                {{(BIN_W-16){1'b0}}, raw};
        end
    endfunction

    always @(*) begin
        source0 = 2'd0;
        source1 = 2'd0;
        source2 = 2'd0;
        source3 = 2'd0;
        valid0 = 1'b0;
        valid1 = 1'b0;
        valid2 = 1'b0;
        valid3 = 1'b0;
        high0 = 1'b0;
        high1 = 1'b0;
        high2 = 1'b0;
        case ($signed(update_coarse))
            -9'sd3: begin
                source0 = 2'd3; valid0 = 1'b1;
            end
            -9'sd2: begin
                source0 = 2'd2; valid0 = 1'b1;
                source1 = 2'd3; valid1 = 1'b1;
            end
            -9'sd1: begin
                source0 = 2'd1; valid0 = 1'b1;
                source1 = 2'd2; valid1 = 1'b1;
                source2 = 2'd3; valid2 = 1'b1;
            end
            9'sd0: begin
                source0 = 2'd0; valid0 = 1'b1;
                source1 = 2'd1; valid1 = 1'b1;
                source2 = 2'd2; valid2 = 1'b1;
                source3 = 2'd3; valid3 = 1'b1;
            end
            9'sd1: begin
                source0 = 2'd3; valid0 = 1'b1; high0 = 1'b1;
                source1 = 2'd0; valid1 = 1'b1;
                source2 = 2'd1; valid2 = 1'b1;
                source3 = 2'd2; valid3 = 1'b1;
            end
            9'sd2: begin
                source0 = 2'd2; valid0 = 1'b1; high0 = 1'b1;
                source1 = 2'd3; valid1 = 1'b1; high1 = 1'b1;
                source2 = 2'd0; valid2 = 1'b1;
                source3 = 2'd1; valid3 = 1'b1;
            end
            9'sd3: begin
                source0 = 2'd1; valid0 = 1'b1; high0 = 1'b1;
                source1 = 2'd2; valid1 = 1'b1; high1 = 1'b1;
                source2 = 2'd3; valid2 = 1'b1; high2 = 1'b1;
                source3 = 2'd0; valid3 = 1'b1;
            end
            9'sd4: begin
                source0 = 2'd0; valid0 = 1'b1; high0 = 1'b1;
                source1 = 2'd1; valid1 = 1'b1; high1 = 1'b1;
                source2 = 2'd2; valid2 = 1'b1; high2 = 1'b1;
            end
            9'sd5: begin
                source1 = 2'd0; valid1 = 1'b1; high1 = 1'b1;
                source2 = 2'd1; valid2 = 1'b1; high2 = 1'b1;
            end
            9'sd6: begin
                source2 = 2'd0; valid2 = 1'b1; high2 = 1'b1;
            end
            default: begin
                if (update_coarse[8] && update_chunks[63]) begin
                    valid0 = 1'b1;
                    source0 = 2'd0;
                end
            end
        endcase
        add0 = extend_chunk(source0);
        add1 = extend_chunk(source1);
        add2 = extend_chunk(source2);
        add3 = extend_chunk(source3);

        // Preserve an arithmetic-right-shifted negative value when all four
        // chunks fall below the fixed accumulation window.
        if (update_coarse[8] && (update_coarse != -9'sd3) &&
            (update_coarse != -9'sd2) && (update_coarse != -9'sd1) &&
            update_chunks[63]) begin
            valid0 = 1'b1;
            high0 = 1'b0;
            add0 = {BIN_W{1'b1}};
        end

    end

    // Register the decoded contribution before the LUTRAM read-modify-write.
    // This separates the coarse-digit decode/mux from the 26-bit carry chain.
    // Payload registers and memories intentionally have no reset.
    always @(*) begin
        next0 = update_clear_q ? {2*BIN_W{1'b0}} : records0[update_addr0_q];
        next1 = update_clear_q ? {2*BIN_W{1'b0}} : records1[update_addr1_q];
        next2 = update_clear_q ? {2*BIN_W{1'b0}} : records2[update_addr2_q];
        next3 = update_clear_q ? {BIN_W{1'b0}} : records3[update_addr3_q];
        if (valid0_q) begin
            if (high0_q)
                next0[BIN_W +: BIN_W] =
                    $signed(next0[BIN_W +: BIN_W]) + add0_q;
            else
                next0[0 +: BIN_W] =
                    $signed(next0[0 +: BIN_W]) + add0_q;
        end
        if (valid1_q) begin
            if (high1_q)
                next1[BIN_W +: BIN_W] =
                    $signed(next1[BIN_W +: BIN_W]) + add1_q;
            else
                next1[0 +: BIN_W] =
                    $signed(next1[0 +: BIN_W]) + add1_q;
        end
        if (valid2_q) begin
            if (high2_q)
                next2[BIN_W +: BIN_W] =
                    $signed(next2[BIN_W +: BIN_W]) + add2_q;
            else
                next2[0 +: BIN_W] =
                    $signed(next2[0 +: BIN_W]) + add2_q;
        end
        if (valid3_q)
            next3 = $signed(next3) + add3_q;
    end

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            update_valid_q <= 1'b0;
        end else begin
            update_valid_q <= update_valid;
            if (update_valid) begin
                update_clear_q <= update_clear;
                update_addr0_q <= update_addr;
                update_addr1_q <= update_addr;
                update_addr2_q <= update_addr;
                update_addr3_q <= update_addr;
                add0_q <= add0;
                add1_q <= add1;
                add2_q <= add2;
                add3_q <= add3;
                valid0_q <= valid0;
                valid1_q <= valid1;
                valid2_q <= valid2;
                valid3_q <= valid3;
                high0_q <= high0;
                high1_q <= high1;
                high2_q <= high2;
            end
            if (update_valid_q) begin
                records0[update_addr0_q] <= next0;
                records1[update_addr1_q] <= next1;
                records2[update_addr2_q] <= next2;
                records3[update_addr3_q] <= next3;
            end
        end
    end

    assign update_pending = update_valid_q;

    assign drain_digits[0*BIN_W +: BIN_W] =
        records0[drain_addr][0 +: BIN_W];
    assign drain_digits[1*BIN_W +: BIN_W] =
        records1[drain_addr][0 +: BIN_W];
    assign drain_digits[2*BIN_W +: BIN_W] =
        records2[drain_addr][0 +: BIN_W];
    assign drain_digits[3*BIN_W +: BIN_W] = records3[drain_addr];
    assign drain_digits[4*BIN_W +: BIN_W] =
        records0[drain_addr][BIN_W +: BIN_W];
    assign drain_digits[5*BIN_W +: BIN_W] =
        records1[drain_addr][BIN_W +: BIN_W];
    assign drain_digits[6*BIN_W +: BIN_W] =
        records2[drain_addr][BIN_W +: BIN_W];
endmodule

// Four-stage radix carry normalizer.  It accepts and emits one logical output
// per cycle.  A single global clock-enable makes every payload and tag stable
// under output backpressure.
module digit_normalize #(
    parameter integer DIGITS = 7,
    parameter integer BIN_W  = 26,
    parameter integer ACC_W  = 104,
    parameter integer META_W = 30
) (
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          in_valid,
    output wire                          in_ready,
    input  wire signed [DIGITS*BIN_W-1:0] in_digits,
    input  wire [META_W-1:0]             in_meta,
    output wire                          out_valid,
    input  wire                          out_ready,
    output wire signed [ACC_W-1:0]       out_acc,
    output wire [META_W-1:0]             out_meta,
    output wire                          busy
);
    localparam integer TMP_W = BIN_W+1;

    initial begin
        if (DIGITS != 7 || BIN_W != 26 || ACC_W != 104)
            $error(" digit_normalize parameter set is unsupported");
    end

    reg v0, v1, v2, v3;
    wire advance = !v3 || out_ready;
    assign in_ready = advance;
    assign out_valid = v3;
    assign busy = v0 || v1 || v2 || v3;

    wire signed [BIN_W-1:0] in_d0 =
        $signed(in_digits[0*BIN_W +: BIN_W]);
    wire signed [BIN_W-1:0] in_d1 =
        $signed(in_digits[1*BIN_W +: BIN_W]);
    wire signed [TMP_W-1:0] in_t0 =
        $signed({in_d0[BIN_W-1], in_d0});
    wire signed [TMP_W-1:0] in_t1 =
        $signed({in_d1[BIN_W-1], in_d1}) + (in_t0 >>> 16);

    reg signed [TMP_W-1:0] carry0;
    reg [31:0] low0;
    reg [(DIGITS-2)*BIN_W-1:0] remain0;
    reg [META_W-1:0] meta0;

    wire signed [BIN_W-1:0] s1_d2 =
        $signed(remain0[0*BIN_W +: BIN_W]);
    wire signed [BIN_W-1:0] s1_d3 =
        $signed(remain0[1*BIN_W +: BIN_W]);
    wire signed [TMP_W-1:0] s1_t2 =
        $signed({s1_d2[BIN_W-1], s1_d2}) + carry0;
    wire signed [TMP_W-1:0] s1_t3 =
        $signed({s1_d3[BIN_W-1], s1_d3}) + (s1_t2 >>> 16);

    reg signed [TMP_W-1:0] carry1;
    reg [63:0] low1;
    reg [(DIGITS-4)*BIN_W-1:0] remain1;
    reg [META_W-1:0] meta1;

    wire signed [BIN_W-1:0] s2_d4 =
        $signed(remain1[0*BIN_W +: BIN_W]);
    wire signed [BIN_W-1:0] s2_d5 =
        $signed(remain1[1*BIN_W +: BIN_W]);
    wire signed [TMP_W-1:0] s2_t4 =
        $signed({s2_d4[BIN_W-1], s2_d4}) + carry1;
    wire signed [TMP_W-1:0] s2_t5 =
        $signed({s2_d5[BIN_W-1], s2_d5}) + (s2_t4 >>> 16);

    reg signed [TMP_W-1:0] carry2;
    reg [95:0] low2;
    reg signed [BIN_W-1:0] final_digit2;
    reg [META_W-1:0] meta2;

    wire signed [TMP_W-1:0] s3_t6 =
        $signed({final_digit2[BIN_W-1], final_digit2}) + carry2;
    reg signed [ACC_W-1:0] acc3;
    reg [META_W-1:0] meta3;

    always @(posedge clk) begin
        if (!rst_n) begin
            v0 <= 1'b0;
            v1 <= 1'b0;
            v2 <= 1'b0;
            v3 <= 1'b0;
        end else if (advance) begin
            v0 <= in_valid;
            v1 <= v0;
            v2 <= v1;
            v3 <= v2;

            low0 <= {in_t1[15:0], in_t0[15:0]};
            carry0 <= in_t1 >>> 16;
            remain0 <= in_digits[DIGITS*BIN_W-1:2*BIN_W];
            meta0 <= in_meta;

            low1 <= {s1_t3[15:0], s1_t2[15:0], low0};
            carry1 <= s1_t3 >>> 16;
            remain1 <= remain0[(DIGITS-2)*BIN_W-1:2*BIN_W];
            meta1 <= meta0;

            low2 <= {s2_t5[15:0], s2_t4[15:0], low1};
            carry2 <= s2_t5 >>> 16;
            final_digit2 <= remain1[2*BIN_W +: BIN_W];
            meta2 <= meta1;

            acc3 <= {s3_t6[7:0], low2};
            meta3 <= meta2;
        end
    end

    assign out_acc = acc3;
    assign out_meta = meta3;
endmodule

`default_nettype wire
