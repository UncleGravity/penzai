//! Cosim for flash_softmax: the online-softmax transform step. Checks the RTL
//! (m_out, l_out, p, corr, grew) against a software reference that uses the same
//! HW-modeled exp (flash_ref.softmaxExp, B=8), over both random (m,l,score) triples
//! and realistic kv sequences (m starts -inf, l=0; outputs fed forward as the next
//! step's inputs, closing the recurrence the kernel will own). m_out/grew are exact
//! (a select / compare); l_out/p/corr are ε (fp truncation + the exp LUT gap).

const std = @import("std");
const ref = @import("flash_ref");

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

fn f32bits(v: f32) u32 {
    return @bitCast(v);
}
fn bitsf32(v: u32) f32 {
    return @bitCast(v);
}

const Step = struct { m: f32, l: f32, p: f32, corr: f32, grew: bool };

fn refStep(m_in: f32, l_in: f32, score: f32) Step {
    const grew = score > m_in;
    const m_new = @max(m_in, score);
    const corr = ref.softmaxExp(8, m_in - m_new);
    const p = ref.softmaxExp(8, score - m_new);
    return .{ .m = m_new, .l = l_in * corr + p, .p = p, .corr = corr, .grew = grew };
}

const Out = struct { m: f32, l: f32, p: f32, corr: f32, grew: bool };

fn drive(dut: *Dut, m_in: f32, l_in: f32, score: f32) Out {
    c.dut_set_m_in(dut.h, f32bits(m_in));
    c.dut_set_l_in(dut.h, f32bits(l_in));
    c.dut_set_score(dut.h, f32bits(score));
    c.dut_set_valid(dut.h, 1);
    dut.step();
    c.dut_set_valid(dut.h, 0);
    var d: usize = 0;
    while (d < 48) : (d += 1) {
        dut.step();
        if (c.dut_valid_out(dut.h) != 0) {
            return .{
                .m = bitsf32(c.dut_m_out(dut.h)),
                .l = bitsf32(c.dut_l_out(dut.h)),
                .p = bitsf32(c.dut_p(dut.h)),
                .corr = bitsf32(c.dut_corr(dut.h)),
                .grew = c.dut_grew(dut.h) != 0,
            };
        }
    }
    return .{ .m = std.math.nan(f32), .l = 0, .p = 0, .corr = 0, .grew = false };
}

const Acc = struct {
    l_rel: f64 = 0,
    p_rel: f64 = 0,
    corr_rel: f64 = 0,
    m_bad: usize = 0,
    grew_bad: usize = 0,
    n: usize = 0,

    fn record(self: *Acc, got: Out, exp: Step) void {
        self.n += 1;
        if (f32bits(got.m) != f32bits(exp.m)) self.m_bad += 1;
        if (got.grew != exp.grew) self.grew_bad += 1;
        self.l_rel = @max(self.l_rel, rel(exp.l, got.l));
        self.p_rel = @max(self.p_rel, rel(exp.p, got.p));
        self.corr_rel = @max(self.corr_rel, rel(exp.corr, got.corr));
    }
};

fn rel(e: f32, g: f32) f64 {
    const d = @abs(@as(f64, e) - @as(f64, g));
    return if (@abs(@as(f64, e)) < 1e-6) d else d / @abs(@as(f64, e));
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();

    // Reset.
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_valid(dut.h, 0);
    c.dut_set_m_in(dut.h, 0);
    c.dut_set_l_in(dut.h, 0);
    c.dut_set_score(dut.h, 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();

    var acc: Acc = .{};
    var prng = std.Random.DefaultPrng.init(0x50F7);
    const rnd = prng.random();

    // 1) Random triples (breadth). m_in occasionally -inf (the first-kv case).
    for (0..400) |i| {
        const m_in: f32 = if (i % 17 == 0) -std.math.inf(f32) else (rnd.float(f32) - 0.5) * 20.0;
        const l_in: f32 = rnd.float(f32) * 8.0;
        const score: f32 = (rnd.float(f32) - 0.5) * 20.0;
        acc.record(drive(&dut, m_in, l_in, score), refStep(m_in, l_in, score));
    }

    // 2) Realistic kv sequences: m=-inf, l=0; feed outputs forward (recurrence).
    for (0..24) |seq| {
        var m: f32 = -std.math.inf(f32);
        var l: f32 = 0;
        var sp = std.Random.DefaultPrng.init(0xA77 + @as(u64, seq));
        const sr = sp.random();
        for (0..40) |_| {
            // Mix: mostly random, sometimes a new running max (frequent grew).
            const score: f32 = (sr.float(f32) - 0.5) * 16.0;
            const got = drive(&dut, m, l, score);
            acc.record(got, refStep(m, l, score));
            m = got.m; // feed RTL outputs forward, as the kernel will
            l = got.l;
        }
    }

    std.debug.print("\n  flash_softmax cosim: {d} steps\n", .{acc.n});
    std.debug.print("    m_out exact mismatches : {d}\n", .{acc.m_bad});
    std.debug.print("    grew  exact mismatches : {d}\n", .{acc.grew_bad});
    std.debug.print("    l_out max_rel={e:.3}  p max_rel={e:.3}  corr max_rel={e:.3}\n", .{ acc.l_rel, acc.p_rel, acc.corr_rel });

    if (acc.m_bad != 0 or acc.grew_bad != 0) {
        std.debug.print("  FAIL: m_out/grew must be bit-exact\n", .{});
        return error.ExactMismatch;
    }
    if (acc.l_rel > 1e-3 or acc.p_rel > 1e-3 or acc.corr_rel > 1e-3) {
        std.debug.print("  FAIL: fp error exceeds threshold\n", .{});
        return error.ResultMismatch;
    }
    std.debug.print("  all flash_softmax cosim cases passed\n\n", .{});
}
