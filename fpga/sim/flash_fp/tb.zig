//! Micro-cosim for the migrated numeric leaves: drives exp and recip (wrapped in
//! flash_fp_top with fp_dot) and checks each RTL result against the validated
//! `flash_ref` software model (softmaxExp / recip, B=8). The pipes are fixed-latency
//! and in-order, so the k-th valid_out is the k-th input — collect in order.
//!
//! ε note: the RTL ranges-reduce on |x| (2^-af) while the model uses the 2^+f
//! convention; the two are algebraically equal but interpolate at reflected points,
//! so they agree to ~LUT precision, not bit-for-bit. The threshold below is far
//! tighter than any structural bug (which shows as >>1% error) yet absorbs that gap.

const std = @import("std");
const ref = @import("flash_ref");

const c = @cImport(@cInclude("shim.h"));

const N: usize = 512;

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

fn f32bits(v: f32) u32 {
    return @bitCast(v);
}
fn bitsf32(v: u32) f32 {
    return @bitCast(v);
}

fn err(expected: f32, got: f32) struct { abs: f64, rel: f64 } {
    const e: f64 = expected;
    const g: f64 = got;
    const d = @abs(e - g);
    const rel = if (@abs(e) < 1e-30) 0.0 else d / @abs(e);
    return .{ .abs = d, .rel = rel };
}

pub fn main() !void {
    var xs: [N]f32 = undefined;
    var ls: [N]f32 = undefined;
    var exp_e: [N]f32 = undefined;
    var rec_e: [N]f32 = undefined;

    for (0..N) |k| {
        const frac = @as(f32, @floatFromInt(k)) / @as(f32, N);
        xs[k] = -87.0 * frac; // (-87, 0]
        ls[k] = 1.0 + 8191.0 * frac; // [1, 8192)
    }
    // Guard / boundary cases.
    xs[0] = 1.0; // x > 0     -> 1
    xs[1] = 0.0; // x == 0    -> 1
    xs[2] = -100.0; // x < -87 -> 0
    xs[3] = -87.0; // boundary -> 0
    ls[0] = 1.0; // 1/1 == 1
    for (0..N) |k| {
        exp_e[k] = ref.softmaxExp(8, xs[k]);
        rec_e[k] = ref.recip(8, ls[k]);
    }

    // fp_dot inputs: 8 (f32 Q, f16 K) lanes per beat; expected = Σ q·f32(k).
    var dq_words: [N][8]u32 = undefined;
    var dk_words: [N][4]u32 = undefined;
    var dot_e: [N]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xD07);
    const rnd = prng.random();
    for (0..N) |k| {
        var acc: f32 = 0;
        for (0..4) |w| dk_words[k][w] = 0;
        for (0..8) |i| {
            const qv = (rnd.float(f32) - 0.5) * 4.0;
            const kv: f16 = @floatCast((rnd.float(f32) - 0.5) * 4.0);
            dq_words[k][i] = @bitCast(qv);
            const kbits: u16 = @bitCast(kv);
            dk_words[k][i / 2] |= @as(u32, kbits) << @intCast((i % 2) * 16);
            acc += qv * @as(f32, @floatCast(kv));
        }
        dot_e[k] = acc;
    }

    var dut = Dut.init();
    defer dut.deinit();

    // Reset.
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_valid(dut.h, 0);
    c.dut_set_x(dut.h, 0);
    c.dut_set_l(dut.h, 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();

    var ec: usize = 0; // exp outputs captured
    var rc: usize = 0; // recip outputs captured
    var dc: usize = 0; // dot outputs captured
    var exp_abs: f64 = 0;
    var exp_rel: f64 = 0;
    var rec_abs: f64 = 0;
    var rec_rel: f64 = 0;
    var dot_rel: f64 = 0;
    var worst_exp_x: f32 = 0;
    var worst_rec_l: f32 = 0;

    // Single-shot: one input, drain fully, capture each leaf's lone valid_out.
    // Unambiguous (no pipeline-ordering assumption) — isolates datapath correctness.
    for (0..N) |k| {
        c.dut_set_x(dut.h, f32bits(xs[k]));
        c.dut_set_l(dut.h, f32bits(ls[k]));
        c.dut_set_dq(dut.h, &dq_words[k]);
        c.dut_set_dk(dut.h, &dk_words[k]);
        c.dut_set_valid(dut.h, 1);
        dut.step();
        c.dut_set_valid(dut.h, 0);

        var got_e: ?f32 = null;
        var got_r: ?f32 = null;
        var got_d: ?f32 = null;
        var d: usize = 0;
        while (d < 40 and (got_e == null or got_r == null or got_d == null)) : (d += 1) {
            dut.step();
            if (got_e == null and c.dut_exp_valid(dut.h) != 0) got_e = bitsf32(c.dut_exp_y(dut.h));
            if (got_r == null and c.dut_recip_valid(dut.h) != 0) got_r = bitsf32(c.dut_recip_y(dut.h));
            if (got_d == null and c.dut_dot_valid(dut.h) != 0) got_d = bitsf32(c.dut_dot_sum(dut.h));
        }

        if (got_e) |g| {
            const e = err(exp_e[k], g);
            if (e.rel > exp_rel) {
                exp_rel = e.rel;
                worst_exp_x = xs[k];
            }
            exp_abs = @max(exp_abs, e.abs);
            ec += 1;
        }
        if (got_r) |g| {
            const e = err(rec_e[k], g);
            if (e.rel > rec_rel) {
                rec_rel = e.rel;
                worst_rec_l = ls[k];
            }
            rec_abs = @max(rec_abs, e.abs);
            rc += 1;
        }
        if (got_d) |g| {
            // Divide by max(|exp|,1) like the matmul cosim — robust to cancellation;
            // the gap is fp truncation + the tree-vs-sequential reduction order.
            const rel = @abs(@as(f64, dot_e[k]) - @as(f64, g)) / @max(@abs(@as(f64, dot_e[k])), 1.0);
            dot_rel = @max(dot_rel, rel);
            dc += 1;
        }
    }

    std.debug.print(
        "\n  flash fp leaf cosim: exp {d} / recip {d} / dot {d} outputs (of {d})\n",
        .{ ec, rc, dc, N },
    );
    std.debug.print(
        "    exp      : max_rel={e:.3} (@x={d:.3})  max_abs={e:.3}\n",
        .{ exp_rel, worst_exp_x, exp_abs },
    );
    std.debug.print(
        "    recip    : max_rel={e:.3} (@l={d:.3})  max_abs={e:.3}\n",
        .{ rec_rel, worst_rec_l, rec_abs },
    );
    std.debug.print(
        "    fp_dot   : max_rel={e:.3} (8-lane Q·K)\n",
        .{dot_rel},
    );

    if (ec != N or rc != N or dc != N) {
        std.debug.print("  FAIL: missing outputs (pipe stalled?)\n", .{});
        return error.MissingOutputs;
    }
    // Structural bugs show as >>1% error; the exp/recip LUT gap is ~1e-5, the dot
    // truncation + reduction-order gap is ~1e-5.
    if (exp_rel > 1e-3 or rec_rel > 1e-3 or dot_rel > 1e-2) {
        std.debug.print("  FAIL: error exceeds threshold\n", .{});
        return error.ResultMismatch;
    }
    std.debug.print("  all flash fp leaf cosim cases passed\n\n", .{});
}
