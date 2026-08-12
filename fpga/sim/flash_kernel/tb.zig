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
    expected_hash: u64 = 0,
};

const Traffic = enum { baseline, stressed };

const RunResult = struct {
    cycles: usize,
    hash: u64,
};

const zero8 = [_]u32{0} ** 8;
const zero4 = [_]u32{0} ** 4;

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

fn hashOutputs(values: []const f32) u64 {
    var hash: u64 = 14695981039346656037;
    for (values) |value| {
        const word = f32bits(value);
        inline for (0..4) |byte_i| {
            hash ^= @as(u8, @truncate(word >> (byte_i * 8)));
            hash *%= 1099511628211;
        }
    }
    return hash;
}

fn runCfg(comptime cfg: Cfg, dut: *Dut, seed: u64, traffic: Traffic) !RunResult {
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

    var data_prng = std.Random.DefaultPrng.init(seed);
    const rnd = data_prng.random();
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
    var o_beats: usize = 0;
    var o_got: [NTOK * NH * HDV]f32 = undefined;

    // Each source independently decides when to present its next beat. Once TVALID
    // rises, it stays asserted with the same beat until the DUT accepts it.
    var q_active = false;
    var k_active = false;
    var v_active = false;
    var mask_active = false;
    var traffic_prng = std.Random.DefaultPrng.init(seed ^ 0xA519_2F61_7C3D_EB47);
    const traffic_rnd = traffic_prng.random();

    var stalled_o = false;
    var stalled_data: [8]u32 = undefined;
    var stalled_keep: u32 = undefined;
    var stalled_last = false;
    var last_count: usize = 0;
    var saw_done = false;

    var cyc: usize = 0;
    const LIMIT: usize = 500000;
    while (cyc < LIMIT) : (cyc += 1) {
        if (!q_active and qc < q_seq.len and (traffic == .baseline or traffic_rnd.int(u2) != 0)) q_active = true;
        if (!k_active and kc < k_seq.len and (traffic == .baseline or traffic_rnd.int(u2) != 0)) k_active = true;
        if (!v_active and vc < v_seq.len and (traffic == .baseline or traffic_rnd.int(u2) != 0)) v_active = true;
        if (!mask_active and mc < mask_seq.len and (traffic == .baseline or traffic_rnd.int(u2) != 0)) mask_active = true;

        const q_last = q_active and qc + 1 == q_seq.len;
        const k_last = k_active and kc + 1 == k_seq.len;
        const v_last = v_active and vc + 1 == v_seq.len;
        const mask_last = mask_active and mc + 1 == mask_seq.len;
        c.dut_set_q(dut.h, if (q_active) &q_seq[qc] else &zero8, @intFromBool(q_active), @intFromBool(q_last));
        c.dut_set_k(dut.h, if (k_active) &k_seq[kc] else &zero4, @intFromBool(k_active), @intFromBool(k_last));
        c.dut_set_v(dut.h, if (v_active) &v_seq[vc] else &zero4, @intFromBool(v_active), @intFromBool(v_last));
        c.dut_set_mask(dut.h, if (mask_active) mask_seq[mc] else 0, @intFromBool(mask_active), @intFromBool(mask_last));
        const o_ready = traffic == .baseline or traffic_rnd.int(u2) != 0;
        c.dut_set_o_ready(dut.h, @intFromBool(o_ready));
        dut.eval();

        const qf = q_active and c.dut_q_ready(dut.h) != 0;
        const kf = k_active and c.dut_k_ready(dut.h) != 0;
        const vf = v_active and c.dut_v_ready(dut.h) != 0;
        const mf = mask_active and c.dut_mask_ready(dut.h) != 0;
        const o_valid = c.dut_o_valid(dut.h) != 0;
        if (stalled_o and !o_valid) return error.OutputValidDropped;
        if (o_valid) {
            var beat: [8]u32 = undefined;
            c.dut_o_data(dut.h, &beat);
            const keep = c.dut_o_keep(dut.h);
            const last = c.dut_o_last(dut.h) != 0;
            if (keep != 0xFFFF_FFFF) return error.BadOutputKeep;
            if (stalled_o and (!std.mem.eql(u32, &beat, &stalled_data) or keep != stalled_keep or last != stalled_last)) {
                return error.OutputChangedWhileStalled;
            }
            if (o_ready) {
                if (o_beats >= NTOK * NH * VB or oc + 8 > o_got.len) return error.ExtraOutputs;
                const expect_last = o_beats + 1 == NTOK * NH * VB;
                if (last != expect_last) return error.BadOutputLast;
                if (last) last_count += 1;
                for (0..8) |i| {
                    o_got[oc] = bitsf32(beat[i]);
                    oc += 1;
                }
                o_beats += 1;
                stalled_o = false;
            } else if (!stalled_o) {
                stalled_data = beat;
                stalled_keep = keep;
                stalled_last = last;
                stalled_o = true;
            }
        }
        const is_done = c.dut_done(dut.h) != 0;

        if (is_done and (qc != q_seq.len or kc != k_seq.len or vc != v_seq.len or mc != mask_seq.len or o_beats != NTOK * NH * VB or stalled_o)) {
            return error.PrematureDone;
        }

        dut.step();
        if (qf) {
            qc += 1;
            q_active = false;
        }
        if (kf) {
            kc += 1;
            k_active = false;
        }
        if (vf) {
            vc += 1;
            v_active = false;
        }
        if (mf) {
            mc += 1;
            mask_active = false;
        }
        if (is_done) {
            saw_done = true;
            break;
        }
    }

    if (!saw_done) return error.Timeout;
    dut.eval();
    if (c.dut_done(dut.h) != 0 or c.dut_busy(dut.h) != 0 or c.dut_o_valid(dut.h) != 0) return error.BadIdleAfterDone;

    if (qc != q_seq.len or kc != k_seq.len or vc != v_seq.len or mc != mask_seq.len) return error.InputCountMismatch;
    if (oc != o_got.len or o_beats != NTOK * NH * VB or last_count != 1) {
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
    const output_hash = hashOutputs(&o_got);
    std.debug.print("    {s: <8} hdq{d} hdv{d} nh{d}/{d} nkv{d} ntok{d} mask{d}: {d} cyc, hash={x:0>16}, RTL-ref={e:.3}, bf16-fp32={e:.3}\n", .{ @tagName(traffic), HDQ, HDV, NH, NHKV, NKV, NTOK, cfg.masked_kv, cyc, output_hash, max_rel, bf16_vs_fp32 });
    if (max_rel > 0.02) {
        std.debug.print("  FAIL: O exceeds tolerance (worst i={d} exp={d:.5} got={d:.5})\n", .{ worst, o_exp[worst], o_got[worst] });
        return error.ResultMismatch;
    }
    if (cfg.expected_hash != 0 and output_hash != cfg.expected_hash) return error.OutputHashMismatch;
    return .{ .cycles = cyc, .hash = output_hash };
}

fn runPair(comptime cfg: Cfg, dut: *Dut, seed: u64) !void {
    const baseline = try runCfg(cfg, dut, seed, .baseline);
    const stressed = try runCfg(cfg, dut, seed, .stressed);
    if (stressed.hash != baseline.hash) return error.StressedOutputMismatch;
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();

    // reset once; the kernel returns to IDLE after each run, so configs reuse it.
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_start(dut.h, 0);
    c.dut_set_q(dut.h, &zero8, 0, 0);
    c.dut_set_k(dut.h, &zero4, 0, 0);
    c.dut_set_v(dut.h, &zero4, 0, 0);
    c.dut_set_mask(dut.h, 0, 0, 0);
    c.dut_set_o_ready(dut.h, 1);
    c.dut_set_clk(dut.h, 0);
    dut.eval();
    for (0..6) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();

    std.debug.print("\n  flash_kernel cosim (vs flash_ref.attendHead):\n", .{});
    // GQA r2 + masked kv; fastest one-beat multi-head producer; hdq!=hdv +
    // n_tok>1 + nhkv>1 + no mask; single head + leading mask.
    try runPair(.{ .hdq = 16, .hdv = 16, .nh = 2, .nhkv = 1, .nkv = 4, .ntok = 1, .scale = 0.25, .masked_kv = 2, .expected_hash = 0x571c_b97c_8fc5_a57a }, &dut, 0xF1A54EE7);
    try runPair(.{ .hdq = 8, .hdv = 8, .nh = 2, .nhkv = 1, .nkv = 4, .ntok = 1, .scale = 0.35355338, .masked_kv = -1, .expected_hash = 0xe455_4143_ad16_f43c }, &dut, 0x8011);
    try runPair(.{ .hdq = 24, .hdv = 8, .nh = 4, .nhkv = 2, .nkv = 3, .ntok = 2, .scale = 0.125, .masked_kv = -1, .expected_hash = 0xb5ae_b111_cbf4_8077 }, &dut, 0xBEEF01);
    try runPair(.{ .hdq = 8, .hdv = 8, .nh = 1, .nhkv = 1, .nkv = 5, .ntok = 1, .scale = 1.0, .masked_kv = 0, .expected_hash = 0x7d76_2a47_b5e4_31f3 }, &dut, 0xC0FFEE);
    // Non-power-of-two head count, q/v widths, and KV extent. The interior mask also
    // verifies that skip traffic does not disturb ordered head tags or output framing.
    try runPair(.{ .hdq = 24, .hdv = 16, .nh = 3, .nhkv = 1, .nkv = 5, .ntok = 1, .scale = 0.2, .masked_kv = 2, .expected_hash = 0x14e4_ef7c_e532_1e81 }, &dut, 0x3EAD5);
    // Decode shape: head_dim 128 (qbeats=16, full adder tree, no zero-pad),
    // 16 query heads over 8 kv-heads (GQA r2, half the pool), real 1/sqrt(128) scale.
    try runPair(.{ .hdq = 128, .hdv = 128, .nh = 16, .nhkv = 8, .nkv = 6, .ntok = 1, .scale = 0.08838835, .masked_kv = 4, .expected_hash = 0x041a_157f_8f65_bcef }, &dut, 0xD00D);
    // Scheduler benchmark: enough all-valid KV positions to amortize Q load,
    // accumulator initialization, and final output emission.
    try runPair(.{ .hdq = 128, .hdv = 128, .nh = 16, .nhkv = 8, .nkv = 16, .ntok = 1, .scale = 0.08838835, .masked_kv = -1, .expected_hash = 0x8d9f_b8a9_24f4_8301 }, &dut, 0x16BEEF);
    // 32 query heads over 8 kv-heads (GQA r4): exercises the full MAX_HEADS pool and the
    // head-index bits above [3:0] — the aliasing the [3:0] pool index used to cause.
    try runPair(.{ .hdq = 128, .hdv = 128, .nh = 32, .nhkv = 8, .nkv = 6, .ntok = 1, .scale = 0.08838835, .masked_kv = 4, .expected_hash = 0xcb4b_8407_4344_2339 }, &dut, 0x32EAD5);
    std.debug.print("  all flash_kernel cosim configs passed\n\n", .{});
}
