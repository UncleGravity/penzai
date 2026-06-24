//! Cosim for flash_top: the AXI-Lite/DMA wrapper around flash_kernel. Configures the
//! kernel via AXI-Lite register writes, streams Q/K/V/mask in and O out (same feed as
//! the kernel cosim), checks O vs flash_ref, then reads STATUS + the counter bank back
//! over AXI-Lite. Register offsets come from the generated regmap (single source).

const std = @import("std");
const ref = @import("flash_ref");
const regmap = @import("regmap");

const c = @cImport(@cInclude("shim.h"));

const HDQ = 16;
const HDV = 16;
const NH = 2;
const NHKV = 1;
const NKV = 4;
const NTOK = 1;
const HEAD_RATIO = NH / NHKV;
const QB = HDQ / 8;
const VB = HDV / 8;
const SCALE: f32 = 0.25;

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
    fn axiWrite(self: *Dut, off: u32, data: u32) void {
        c.dut_axi_write(self.h, @intCast(off), data, 1);
        self.step();
        self.step();
        c.dut_axi_idle(self.h);
        self.step();
    }
    fn axiRead(self: *Dut, off: u32) u32 {
        c.dut_axi_read(self.h, @intCast(off), 1);
        var i: usize = 0;
        var val: u32 = 0;
        while (i < 16) : (i += 1) {
            self.step();
            if (c.dut_axi_rvalid(self.h) != 0) {
                val = c.dut_axi_rdata(self.h);
                break;
            }
        }
        c.dut_axi_idle(self.h);
        self.step();
        return val;
    }
};

pub fn main() !void {
    // ---- source tensors + reference (mirror flash_ref.attendHead) ----
    var q: [NTOK * NH * HDQ]f32 = undefined;
    var k: [NKV * NHKV * HDQ]u16 = undefined;
    var v: [NKV * NHKV * HDV]u16 = undefined;
    var mask: [NTOK * NKV]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(0x70F70F);
    const rnd = prng.random();
    for (&q) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (&k) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (&v) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (0..NTOK) |t| for (0..NKV) |kv| {
        mask[t * NKV + kv] = if (kv == 2) f16bits(-std.math.inf(f32)) else f16bits(0.0);
    };

    var o_exp: [NTOK * NH * HDV]f32 = undefined;
    for (0..NTOK) |t| for (0..NH) |h| {
        const kvh = h / HEAD_RATIO;
        var acc = [_]f32{0} ** HDV;
        var m: f32 = -std.math.inf(f32);
        var l: f32 = 0;
        for (0..NKV) |kv| {
            const mvv = f16val(mask[t * NKV + kv]);
            if (!std.math.isFinite(mvv) and mvv < 0) continue;
            var dot: f32 = 0;
            for (0..HDQ) |d| dot += q[(t * NH + h) * HDQ + d] * f16val(k[(kv * NHKV + kvh) * HDQ + d]);
            const score = dot * SCALE + mvv;
            const m_new = @max(m, score);
            if (m_new != m) {
                const corr = ref.softmaxExp(8, m - m_new);
                l *= corr;
                for (&acc) |*a| a.* *= corr;
                m = m_new;
            }
            const p = ref.softmaxExp(8, score - m);
            l += p;
            for (0..HDV) |d| acc[d] += ref.bf16MulPV(p, f16val(v[(kv * NHKV + kvh) * HDV + d])); // bf16 p·V (matches fp_axpy8 u_m2)
        }
        const inv_l = ref.recip(8, l);
        for (0..HDV) |d| o_exp[(t * NH + h) * HDV + d] = acc[d] * inv_l;
    };

    // ---- input streams in the kv-major kernel's consumption order ----
    // Q per (token, head); K/V/mask kv-major, NON-replicated (one per kv-head).
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

    var dut = Dut.init();
    defer dut.deinit();

    // reset
    c.dut_set_rst_n(dut.h, 0);
    c.dut_axi_idle(dut.h);
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

    // ---- configure via AXI-Lite, then start ----
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_Q"), HDQ);
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_V"), HDV);
    dut.axiWrite(regmap.offsetOf("N_HEADS"), NH);
    dut.axiWrite(regmap.offsetOf("N_HEAD_KV"), NHKV);
    dut.axiWrite(regmap.offsetOf("HEAD_RATIO"), HEAD_RATIO);
    dut.axiWrite(regmap.offsetOf("N_KV"), NKV);
    dut.axiWrite(regmap.offsetOf("N_TOKENS"), NTOK);
    dut.axiWrite(regmap.offsetOf("SCALE"), f32bits(SCALE));
    // readback sanity on one config register
    const hdq_rb = dut.axiRead(regmap.offsetOf("HEAD_DIM_Q"));
    dut.axiWrite(regmap.offsetOf("CTRL"), 1); // start strobe
    c.dut_axi_idle(dut.h);

    // ---- feed streams, collect O ----
    var qc: usize = 0;
    var kc: usize = 0;
    var vc: usize = 0;
    var mc: usize = 0;
    var oc: usize = 0;
    var o_got: [NTOK * NH * HDV]f32 = undefined;
    var cyc: usize = 0;
    while (cyc < 100000 and oc < o_got.len) : (cyc += 1) {
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
        dut.step();
        if (qf) qc += 1;
        if (kf) kc += 1;
        if (vf) vc += 1;
        if (mf) mc += 1;
    }
    // streams idle, then let the kernel finish (last O beat → S_TNEXT → S_DONE → IDLE)
    // before polling status/counters.
    c.dut_set_q(dut.h, &[_]u32{0} ** 8, 0);
    c.dut_set_k(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_v(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_mask(dut.h, 0, 0);
    for (0..8) |_| dut.step();

    const status = dut.axiRead(regmap.offsetOf("STATUS"));
    const q_beats = dut.axiRead(regmap.offsetOf("Q_BEATS"));
    const k_beats = dut.axiRead(regmap.offsetOf("K_BEATS"));
    const v_beats = dut.axiRead(regmap.offsetOf("V_BEATS"));
    const o_beats = dut.axiRead(regmap.offsetOf("O_BEATS"));
    const id = dut.axiRead(regmap.offsetOf("ID"));

    std.debug.print("\n  flash_top cosim: {d} cyc, O {d}/{d}\n", .{ cyc, oc, o_got.len });
    std.debug.print("    config readback HEAD_DIM_Q={d} (exp {d}); ID=0x{X:0>8}\n", .{ hdq_rb, HDQ, id });
    std.debug.print("    STATUS=0x{X} (done bit {d}); counters q={d} k={d} v={d} o={d}\n", .{ status, (status >> 1) & 1, q_beats, k_beats, v_beats, o_beats });

    if (oc != o_got.len) {
        std.debug.print("  FAIL: O incomplete ({d}/{d})\n", .{ oc, o_got.len });
        return error.MissingOutputs;
    }
    var max_rel: f64 = 0;
    for (0..o_got.len) |i| {
        const r = @abs(@as(f64, o_exp[i]) - @as(f64, o_got[i])) / @max(@abs(@as(f64, o_exp[i])), 1.0);
        max_rel = @max(max_rel, r);
    }
    std.debug.print("    O max_rel={e:.4}\n", .{max_rel});

    var ok = true;
    if (max_rel > 0.02) {
        std.debug.print("  FAIL: O exceeds tolerance\n", .{});
        ok = false;
    }
    if (hdq_rb != HDQ) {
        std.debug.print("  FAIL: config readback {d} != {d}\n", .{ hdq_rb, HDQ });
        ok = false;
    }
    if (id != regmap.resetOf("ID")) {
        std.debug.print("  FAIL: ID 0x{X} != regmap 0x{X}\n", .{ id, regmap.resetOf("ID") });
        ok = false;
    }
    if ((status >> 1) & 1 != 1) {
        std.debug.print("  FAIL: STATUS done bit not set\n", .{});
        ok = false;
    }
    const exp_q = NTOK * NH * QB;
    const exp_o = NTOK * NH * VB; // packed O: VB beats/head (8 f32 each), not HDV
    const exp_k = NTOK * NKV * NHKV * QB; // kv-major, non-replicated; all kv consumed (incl. masked skip)
    const exp_v = NTOK * NKV * NHKV * VB;
    if (q_beats != exp_q or o_beats != exp_o or k_beats != exp_k or v_beats != exp_v) {
        std.debug.print("  FAIL: counters q={d}(exp {d}) k={d}(exp {d}) v={d}(exp {d}) o={d}(exp {d})\n", .{ q_beats, exp_q, k_beats, exp_k, v_beats, exp_v, o_beats, exp_o });
        ok = false;
    }
    if (!ok) return error.Failed;
    std.debug.print("  all flash_top cosim cases passed\n\n", .{});
}
