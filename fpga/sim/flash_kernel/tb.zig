//! Cosim for flash_kernel: full flash attention, composed from the verified leaves,
//! checked end-to-end against a software reference that mirrors flash_ref.attendHead
//! (using the HW-modeled softmaxExp/recip, B=8). The kernel streams Q/K/V/mask via
//! AXIS in its own consumption order; the tb feeds each stream reactively (provide a
//! beat whenever the kernel asserts that stream's tready) and collects O. GQA lives
//! in the feed (kv_head = head/head_ratio); the kernel is GQA-agnostic.

const std = @import("std");
const ref = @import("flash_ref");

const c = @cImport(@cInclude("shim.h"));

const Cfg = struct {
    hdq: usize,
    hdv: usize,
    nh: usize,
    nhkv: usize,
    nkv: usize,
    ntok: usize,
    scale: f32,
    masked_kv: i32, // a kv index to mask (-inf), or -1 for none
};

fn f32bits(v: f32) u32 {
    return @bitCast(v);
}
fn bitsf32(v: u32) f32 {
    return @bitCast(v);
}
fn f16bits(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}
fn f16val(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

const Dut = struct {
    h: *c.Dut,
    fn init() Dut {
        return .{ .h = c.dut_new().? };
    }
    fn deinit(self: *Dut) void {
        c.dut_free(self.h);
    }
    fn eval(self: *Dut) void {
        c.dut_eval(self.h);
    }
    fn step(self: *Dut) void {
        c.dut_set_clk(self.h, 1);
        c.dut_eval(self.h);
        c.dut_set_clk(self.h, 0);
        c.dut_eval(self.h);
    }
};

fn runCfg(comptime cfg: Cfg, dut: *Dut, seed: u64) !void {
    const HDQ = cfg.hdq;
    const HDV = cfg.hdv;
    const NH = cfg.nh;
    const NHKV = cfg.nhkv;
    const NKV = cfg.nkv;
    const NTOK = cfg.ntok;
    const HEAD_RATIO = NH / NHKV;
    const QB = HDQ / 8;
    const VB = HDV / 8;
    const SCALE: f32 = cfg.scale;

    // ---- source tensors, flash_ref layout ----
    var q: [NTOK * NH * HDQ]f32 = undefined;
    var k: [NKV * NHKV * HDQ]u16 = undefined; // f16 bits
    var v: [NKV * NHKV * HDV]u16 = undefined;
    var mask: [NTOK * NKV]u16 = undefined;

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (&q) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (&k) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (&v) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (0..NTOK) |t| {
        for (0..NKV) |kv| mask[t * NKV + kv] = if (@as(i32, @intCast(kv)) == cfg.masked_kv) f16bits(-std.math.inf(f32)) else f16bits(0.0);
    }

    // ---- reference O (mirror flash_ref.attendHead with HW exp/recip) ----
    var o_exp: [NTOK * NH * HDV]f32 = undefined; // bf16 p·V (matches the RTL datapath)
    var o_fp32: [NTOK * NH * HDV]f32 = undefined; // fp32 p·V (the bf16-vs-fp32 baseline)
    for (0..NTOK) |t| {
        for (0..NH) |h| {
            const kvh = h / HEAD_RATIO;
            var acc = [_]f32{0} ** HDV; // bf16 p·V (mirrors the RTL)
            var acc_fp32 = [_]f32{0} ** HDV; // fp32 p·V baseline — shares l/corr/inv_l
            var m: f32 = -std.math.inf(f32);
            var l: f32 = 0;
            for (0..NKV) |kv| {
                const mv = f16val(mask[t * NKV + kv]);
                if (!std.math.isFinite(mv) and mv < 0) continue;
                var dot: f32 = 0;
                for (0..HDQ) |d| dot += q[(t * NH + h) * HDQ + d] * f16val(k[(kv * NHKV + kvh) * HDQ + d]);
                const score = dot * SCALE + mv;
                const m_new = @max(m, score);
                if (m_new != m) {
                    const corr = ref.softmaxExp(8, m - m_new);
                    l *= corr;
                    for (&acc) |*a| a.* *= corr;
                    for (&acc_fp32) |*a| a.* *= corr;
                    m = m_new;
                }
                const p = ref.softmaxExp(8, score - m);
                l += p;
                for (0..HDV) |d| {
                    const vd = f16val(v[(kv * NHKV + kvh) * HDV + d]);
                    acc[d] += ref.bf16MulPV(p, vd); // bf16 — matches fp_axpy8 u_m2
                    acc_fp32[d] += p * vd; // fp32 reference
                }
            }
            const inv_l = ref.recip(8, l);
            for (0..HDV) |d| {
                o_exp[(t * NH + h) * HDV + d] = acc[d] * inv_l;
                o_fp32[(t * NH + h) * HDV + d] = acc_fp32[d] * inv_l;
            }
        }
    }

    // ---- build input streams in the kv-major kernel's consumption order ----
    // Q: per (token, head) resident. K/V/mask: kv-major, NON-replicated (one per
    // kv-head — the native cache layout the device DMAs directly). GQA is done in the
    // kernel, so heads sharing a kv-head are no longer expanded in the feed.
    var q_seq: [NTOK * NH * QB][8]u32 = undefined;
    var k_seq: [NTOK * NKV * NHKV * QB][4]u32 = undefined;
    var v_seq: [NTOK * NKV * NHKV * VB][4]u32 = undefined;
    var mask_seq: [NTOK * NKV]u16 = undefined;
    var qi: usize = 0;
    var ki: usize = 0;
    var vi: usize = 0;
    var mi: usize = 0;
    for (0..NTOK) |t| {
        for (0..NH) |h| {
            for (0..QB) |b| {
                for (0..8) |i| q_seq[qi][i] = f32bits(q[(t * NH + h) * HDQ + b * 8 + i]);
                qi += 1;
            }
        }
        for (0..NKV) |kv| {
            mask_seq[mi] = mask[t * NKV + kv];
            mi += 1;
            for (0..NHKV) |kvh| {
                for (0..QB) |b| {
                    for (0..4) |w| k_seq[ki][w] = @as(u32, k[(kv * NHKV + kvh) * HDQ + b * 8 + 2 * w]) |
                        (@as(u32, k[(kv * NHKV + kvh) * HDQ + b * 8 + 2 * w + 1]) << 16);
                    ki += 1;
                }
            }
            for (0..NHKV) |kvh| {
                for (0..VB) |b| {
                    for (0..4) |w| v_seq[vi][w] = @as(u32, v[(kv * NHKV + kvh) * HDV + b * 8 + 2 * w]) |
                        (@as(u32, v[(kv * NHKV + kvh) * HDV + b * 8 + 2 * w + 1]) << 16);
                    vi += 1;
                }
            }
        }
    }

    // config + start pulse (dut already reset by the caller; returns to IDLE after done)
    c.dut_set_config(dut.h, @intCast(HDQ), @intCast(HDV), @intCast(NH), @intCast(NHKV), @intCast(HEAD_RATIO), @intCast(NKV), @intCast(NTOK), f32bits(SCALE));
    c.dut_set_start(dut.h, 1);
    dut.step();
    c.dut_set_start(dut.h, 0);

    var qc: usize = 0;
    var kc: usize = 0;
    var vc: usize = 0;
    var mc: usize = 0;
    var oc: usize = 0;
    var o_got: [NTOK * NH * HDV]f32 = undefined;

    var cyc: usize = 0;
    const LIMIT: usize = 200000;
    while (cyc < LIMIT) : (cyc += 1) {
        const qv = qc < q_seq.len;
        const kv_ = kc < k_seq.len;
        const vv = vc < v_seq.len;
        const mv = mc < mask_seq.len;
        c.dut_set_q(dut.h, if (qv) &q_seq[qc] else &[_]u32{0} ** 8, @intFromBool(qv));
        c.dut_set_k(dut.h, if (kv_) &k_seq[kc] else &[_]u32{0} ** 4, @intFromBool(kv_));
        c.dut_set_v(dut.h, if (vv) &v_seq[vc] else &[_]u32{0} ** 4, @intFromBool(vv));
        c.dut_set_mask(dut.h, if (mv) mask_seq[mc] else 0, @intFromBool(mv));
        c.dut_set_o_ready(dut.h, 1);
        dut.eval();

        const qf = qv and c.dut_q_ready(dut.h) != 0;
        const kf = kv_ and c.dut_k_ready(dut.h) != 0;
        const vf = vv and c.dut_v_ready(dut.h) != 0;
        const mf = mv and c.dut_mask_ready(dut.h) != 0;
        if (c.dut_o_valid(dut.h) != 0 and oc + 8 <= o_got.len) {
            var beat: [8]u32 = undefined;
            c.dut_o_data(dut.h, &beat);
            for (0..8) |i| {
                o_got[oc] = bitsf32(beat[i]);
                oc += 1;
            }
        }
        const is_done = c.dut_done(dut.h) != 0;

        dut.step();
        if (qf) qc += 1;
        if (kf) kc += 1;
        if (vf) vc += 1;
        if (mf) mc += 1;
        if (is_done) break;
    }

    if (oc != o_got.len) {
        std.debug.print("  FAIL [{d}x{d} nh{d}/{d} nkv{d} ntok{d}]: expected {d} O, got {d}\n", .{ HDQ, HDV, NH, NHKV, NKV, NTOK, o_got.len, oc });
        return error.MissingOutputs;
    }
    var max_rel: f64 = 0;
    var worst: usize = 0;
    var bf16_vs_fp32: f64 = 0;
    for (0..o_got.len) |i| {
        const e: f64 = o_exp[i];
        const g: f64 = o_got[i];
        const r = @abs(e - g) / @max(@abs(e), 1.0);
        if (r > max_rel) {
            max_rel = r;
            worst = i;
        }
        const dvg = @abs(@as(f64, o_exp[i]) - @as(f64, o_fp32[i])) / @max(@abs(@as(f64, o_fp32[i])), 1.0);
        if (dvg > bf16_vs_fp32) bf16_vs_fp32 = dvg;
    }
    std.debug.print("    cfg hdq{d} hdv{d} nh{d} nhkv{d} nkv{d} ntok{d} mask{d}: {d} cyc, O {d}, RTL-vs-ref max_rel={e:.3}, bf16-vs-fp32={e:.3}\n", .{ HDQ, HDV, NH, NHKV, NKV, NTOK, cfg.masked_kv, cyc, oc, max_rel, bf16_vs_fp32 });
    if (max_rel > 0.02) {
        std.debug.print("  FAIL: O exceeds tolerance (worst i={d} exp={d:.5} got={d:.5})\n", .{ worst, o_exp[worst], o_got[worst] });
        return error.ResultMismatch;
    }
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();

    // reset once; the kernel returns to IDLE after each run, so configs reuse it.
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_start(dut.h, 0);
    c.dut_set_q(dut.h, &[_]u32{0} ** 8, 0);
    c.dut_set_k(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_v(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_mask(dut.h, 0, 0);
    c.dut_set_o_ready(dut.h, 1);
    c.dut_set_clk(dut.h, 0);
    dut.eval();
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();

    std.debug.print("\n  flash_kernel cosim (vs flash_ref.attendHead):\n", .{});
    // GQA r2 + masked kv; hdq!=hdv + n_tok>1 + nhkv>1 + no mask; single head + leading mask.
    try runCfg(.{ .hdq = 16, .hdv = 16, .nh = 2, .nhkv = 1, .nkv = 4, .ntok = 1, .scale = 0.25, .masked_kv = 2 }, &dut, 0xF1A54EE7);
    try runCfg(.{ .hdq = 24, .hdv = 8, .nh = 4, .nhkv = 2, .nkv = 3, .ntok = 2, .scale = 0.125, .masked_kv = -1 }, &dut, 0xBEEF01);
    try runCfg(.{ .hdq = 8, .hdv = 8, .nh = 1, .nhkv = 1, .nkv = 5, .ntok = 1, .scale = 1.0, .masked_kv = 0 }, &dut, 0xC0FFEE);
    // Decode shape: head_dim 128 (qbeats=16, full adder tree, no zero-pad),
    // 16 query heads over 8 kv-heads (GQA r2, half the pool), real 1/sqrt(128) scale.
    try runCfg(.{ .hdq = 128, .hdv = 128, .nh = 16, .nhkv = 8, .nkv = 6, .ntok = 1, .scale = 0.08838835, .masked_kv = 4 }, &dut, 0xD00D);
    // 32 query heads over 8 kv-heads (GQA r4): exercises the full MAX_HEADS pool and the
    // head-index bits above [3:0] — the aliasing the [3:0] pool index used to cause.
    try runCfg(.{ .hdq = 128, .hdv = 128, .nh = 32, .nhkv = 8, .nkv = 6, .ntok = 1, .scale = 0.08838835, .masked_kv = 4 }, &dut, 0x32EAD5);
    std.debug.print("  all flash_kernel cosim configs passed\n\n", .{});
}
