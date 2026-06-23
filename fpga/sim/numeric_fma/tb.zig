//! Oracle cosim for numeric/fma (the fixed-point DSP-MAC). Builds a matmul problem with
//! realistic narrow-window f16 scales, feeds each output row's sub-block contributions
//! through one fma instance (clear on the first), and checks the wide accumulator
//! BIT-EXACT against matmul_ref.windowedRow — the exact-in-window integer truth. This is
//! the gemm datapath's numeric gate, proven on real RTL (no board, no build).

const std = @import("std");
const ref = @import("matmul_ref");
const layout = @import("layout");
const c = @cImport(@cInclude("shim.h"));

const Dut = struct {
    h: *c.Dut,
    fn init() Dut {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Dut) void {
        c.dut_free(self.h);
    }
    fn step(self: *Dut) void {
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }
};

// A positive normal f16 with exponent in a narrow band — realistic model scales, well
// inside the fixed [-48,+10] window the 104-bit accumulator covers exactly.
fn normalScale(rnd: std.Random) f16 {
    const exp: u16 = rnd.intRangeAtMost(u16, 12, 18);
    const mant: u16 = rnd.intRangeLessThan(u16, 0, 1024);
    return @bitCast((exp << 10) | mant);
}

fn readAcc(h: *c.Dut) i128 {
    const w0: u128 = c.dut_acc0(h);
    const w1: u128 = c.dut_acc1(h);
    const w2: u128 = c.dut_acc2(h);
    const w3: u128 = c.dut_acc3(h);
    var raw: u128 = w0 | (w1 << 32) | (w2 << 64) | (w3 << 96);
    raw &= (@as(u128, 1) << 104) - 1;
    if (raw >> 103 != 0) {
        return @as(i128, @intCast(raw)) - (@as(i128, 1) << 104);
    }
    return @intCast(raw);
}

// Feed one row's sub-blocks through the single fma instance (clear on the first), drain,
// and return the wide accumulator. Shared by the covering and overflow phases.
fn feedRow(dut: *Dut, p: ref.Problem, row: usize, emin: i32) i128 {
    var first = true;
    var blk: usize = 0;
    while (blk < p.q1_blocks) : (blk += 1) {
        const pw = ref.decompose(p.weight_scales[row * p.q1_blocks + blk]);
        var sub: usize = 0;
        while (sub < layout.Q8_SUBBLOCKS) : (sub += 1) {
            const pa = ref.decompose(p.act_scales[blk * layout.Q8_SUBBLOCKS + sub]);
            const e: i32 = pw.e + pa.e;
            const s: i32 = ref.subSum(p, row, blk, sub);
            c.dut_set_clear(dut.h, if (first) 1 else 0);
            c.dut_set_valid(dut.h, 1);
            c.dut_set_ws_sig(dut.h, @bitCast(@as(i32, @intCast(pw.sig))));
            c.dut_set_as_sig(dut.h, @bitCast(@as(i32, @intCast(pa.sig))));
            c.dut_set_p_exp(dut.h, @bitCast(e));
            c.dut_set_emin(dut.h, @bitCast(emin));
            c.dut_set_s_sum(dut.h, @bitCast(s));
            dut.step();
            first = false;
        }
    }
    c.dut_set_valid(dut.h, 0);
    c.dut_set_clear(dut.h, 0);
    var d: usize = 0;
    while (d < 8) : (d += 1) dut.step();
    return readAcc(dut.h);
}

pub fn main() !void {
    const rows = layout.ROWS;
    const B: usize = 4;
    const Q1 = layout.Q1_BLOCK;
    const SUBS = layout.Q8_SUBBLOCKS;

    var prng = std.Random.DefaultPrng.init(0x123F3A);
    const rnd = prng.random();

    var wbits: [rows * B]u128 = undefined;
    var wscales: [rows * B]f16 = undefined;
    var aquants: [B * Q1]i8 = undefined;
    var ascales: [B * SUBS]f16 = undefined;
    for (&wbits) |*x| x.* = rnd.int(u128);
    for (&wscales) |*x| x.* = normalScale(rnd);
    for (&aquants) |*x| x.* = rnd.intRangeAtMost(i8, -127, 127);
    for (&ascales) |*x| x.* = normalScale(rnd);

    const p = ref.Problem{
        .rows = rows,
        .q1_blocks = B,
        .weight_bits = &wbits,
        .weight_scales = &wscales,
        .act_quants = &aquants,
        .act_scales = &ascales,
    };
    const w = ref.fixedWindow(); // the deployed fixed window (emin -48, ACC_W 104) — no calibration

    var dut = Dut.init();
    defer dut.deinit();

    // reset
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_clear(dut.h, 0);
    c.dut_set_valid(dut.h, 0);
    c.dut_set_ws_sig(dut.h, 0);
    c.dut_set_as_sig(dut.h, 0);
    c.dut_set_p_exp(dut.h, 0);
    c.dut_set_emin(dut.h, @bitCast(w.emin));
    c.dut_set_s_sum(dut.h, 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);

    var checked: usize = 0;
    var mism: usize = 0;
    var sats: usize = 0;

    // ---- phase 1: exact-in-window (covering window, the clamp must NOT engage) ----
    for (0..rows) |row| {
        const got = feedRow(&dut, p, row, w.emin);
        const exp_row = ref.windowedRow(p, row, w);
        sats += exp_row.sats;
        checked += 1;
        if (got != exp_row.acc) {
            mism += 1;
            if (mism <= 8)
                std.debug.print("  row {d}: fma acc={d} expected={d}\n", .{ row, got, exp_row.acc });
        }
    }
    if (mism != 0) {
        std.debug.print("  FAIL: fma !== windowedRow (exact-in-window broken)\n", .{});
        return error.ResultMismatch;
    }
    if (sats != 0) {
        std.debug.print("  FAIL: unexpected saturations in the covering window\n", .{});
        return error.WindowTooNarrow;
    }

    // ---- phase 2: saturation safety net. The fixed window never saturates in production
    //      (that is the whole point), so drive an ADVERSARIAL window to force ACCUMULATOR
    //      overflow and confirm the clamp SATURATES (bounded) instead of WRAPPING
    //      (two's-complement sign flip → garbage), bit-exact vs windowedRow. e=-8, emin=-72 ->
    //      up=64 -> ~2^98 each; 64/row -> >2^103, exceeding the 104-bit accumulator. ----
    const OB: usize = 16; // OB*SUBS = 64 contributions/row
    var owb = [_]u128{~@as(u128, 0)} ** (rows * OB); // all sign bits set (+)
    var ows = [_]f16{@bitCast(@as(u16, (21 << 10) | 1023))} ** (rows * OB); // e=-4, sig=2047
    var oaq = [_]i8{127} ** (OB * Q1); // acts +127 -> subSum 32*127 = 4064
    var oas = [_]f16{@bitCast(@as(u16, (21 << 10) | 1023))} ** (OB * SUBS);
    const op = ref.Problem{
        .rows = rows,
        .q1_blocks = OB,
        .weight_bits = &owb,
        .weight_scales = &ows,
        .act_quants = &oaq,
        .act_scales = &oas,
    };
    const ow = ref.Window{ .emin = -72, .emax = -8, .acc_w = ref.ACC_W_BITS };
    const limit: i128 = (@as(i128, 1) << (ref.ACC_W_BITS - 1)) - 1;

    const sgot = feedRow(&dut, op, 0, ow.emin);
    const sexp = ref.windowedRow(op, 0, ow);
    var sat_fail: usize = 0;
    if (sexp.sats == 0) {
        std.debug.print("  FAIL: overflow construction didn't saturate (sexp.sats=0)\n", .{});
        return error.NoSaturation;
    }
    if (sgot != sexp.acc) {
        sat_fail += 1;
        std.debug.print("  sat: fma acc={d} != windowedRow {d}\n", .{ sgot, sexp.acc });
    }
    if (sgot != limit) {
        sat_fail += 1;
        std.debug.print("  sat: fma acc={d} not clamped to +limit {d} (WRAPPED?)\n", .{ sgot, limit });
    }

    std.debug.print(
        "\n  fma oracle cosim:\n" ++
            "    exact-in-window : {d} rows, mismatches {d}, sats {d}\n" ++
            "    saturation      : fma={d} windowedRow={d} limit={d} (clamped {d} contribs)\n",
        .{ checked, mism, sats, sgot, sexp.acc, limit, sexp.sats },
    );
    if (sat_fail != 0) {
        std.debug.print("  FAIL: fma clamp !== windowedRow (saturation broken / wrapped)\n", .{});
        return error.SaturationMismatch;
    }
    std.debug.print("  fma oracle cosim passed (exact in window + saturating clamp, no wrap)\n\n", .{});
}
