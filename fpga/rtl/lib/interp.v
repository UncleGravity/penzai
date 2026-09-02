// Linear interpolation between two fp32 LUT endpoints with an aligned metadata
// bus. Composes fadd/fmul/cvt_i2f with delay depths derived from fmt.vh.
// The +1 terms are the registered-module handoff: a producer's valid/data become
// visible only after the edge on which the consumer sampled its inputs. Burst cosim
// checks these sideband depths directly; isolated transactions cannot expose an
// off-by-one because their payload remains parked on the input wires.
//
//   frac = lo + (t/256)·(hi - lo)

`default_nettype none

module interp #(
    parameter integer META_W = 16
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire [31:0]       lo,
    input  wire [31:0]       hi,
    input  wire [7:0]        t,
    input  wire [META_W-1:0] meta,
    output wire              valid_out,
    output wire [31:0]       frac,
    output wire [META_W-1:0] meta_out
);
    `include "fmt.vh"
    localparam integer ADD_LAT = FP32_ADD_LATENCY;
    localparam integer MUL_LAT = FP32_MUL_LATENCY;
    localparam integer TOTAL   = ADD_LAT + MUL_LAT + ADD_LAT + 2;
    localparam [31:0]  INV256  = 32'h3B800000; // 2^-8

    integer i;

    // delta = hi - lo (+ADD_LAT)
    wire [31:0] lo_neg = {~lo[31], lo[30:0]};
    wire        delta_v;
    wire [31:0] delta;
    fadd #(.MANT_W(FMT_FP32_MANT)) u_delta (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(hi), .b(lo_neg), .valid_out(delta_v), .out(delta)
    );

    // t_scaled = fp32(t) · 2^-8 (+MUL_LAT)
    wire [31:0] t_f32;
    cvt_i2f #(.WIDTH(9)) u_t (.in({1'b0, t}), .out(t_f32));
    wire        ts_v;
    wire [31:0] t_scaled;
    fmul #(.MANT_W(FMT_FP32_MANT)) u_ts (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .a(t_f32), .b(INV256), .valid_out(ts_v), .out(t_scaled)
    );
    localparam integer TS_DLY = ADD_LAT - MUL_LAT + 1;
    reg [31:0] ts_q [0:TS_DLY-1];
    always @(posedge clk) begin
        ts_q[0] <= t_scaled;
        for (i = 1; i < TS_DLY; i = i + 1) ts_q[i] <= ts_q[i-1];
    end

    // prod = t_scaled · delta (delta_v marks both aligned, +ADD_LAT)
    wire        prod_v;
    wire [31:0] prod;
    fmul #(.MANT_W(FMT_FP32_MANT)) u_prod (
        .clk(clk), .rst_n(rst_n), .valid_in(delta_v),
        .a(ts_q[TS_DLY-1]), .b(delta), .valid_out(prod_v), .out(prod)
    );

    // lo delayed to meet prod (+ADD_LAT+MUL_LAT)
    localparam integer LO_DLY = ADD_LAT + MUL_LAT + 1;
    reg [31:0] lo_q [0:LO_DLY-1];
    always @(posedge clk) begin
        lo_q[0] <= lo;
        for (i = 1; i < LO_DLY; i = i + 1) lo_q[i] <= lo_q[i-1];
    end

    // frac = lo + prod
    fadd #(.MANT_W(FMT_FP32_MANT)) u_frac (
        .clk(clk), .rst_n(rst_n), .valid_in(prod_v),
        .a(lo_q[LO_DLY-1]), .b(prod), .valid_out(valid_out), .out(frac)
    );

    // META delayed the full TOTAL to land with frac.
    reg [META_W-1:0] meta_q [0:TOTAL-1];
    always @(posedge clk) begin
        meta_q[0] <= meta;
        for (i = 1; i < TOTAL; i = i + 1) meta_q[i] <= meta_q[i-1];
    end
    assign meta_out = meta_q[TOTAL-1];
endmodule
