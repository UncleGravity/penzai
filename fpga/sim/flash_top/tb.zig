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

const QSeq = [NTOK * NH * QB][8]u32;
const KSeq = [NTOK * NKV * NHKV * QB][4]u32;
const VSeq = [NTOK * NKV * NHKV * VB][4]u32;
const MaskSeq = [NTOK * NKV]u16;
const Output = [NTOK * NH * HDV]f32;

const RunStats = struct {
    cycles: u32,
    k_stalls: u32,
    v_stalls: u32,
    o_stalls: u32,
};

fn resetDut(dut: *Dut) void {
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
}

fn runOnce(
    dut: *Dut,
    name: []const u8,
    q_seq: *const QSeq,
    k_seq: *const KSeq,
    v_seq: *const VSeq,
    mask_seq: *const MaskSeq,
    o_exp: *const Output,
    inject_stalls: bool,
) !RunStats {
    // Reconfigure and start every run. In particular, CTRL.start must clear status
    // and all performance counters without relying on reset between invocations.
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_Q"), HDQ);
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_V"), HDV);
    dut.axiWrite(regmap.offsetOf("N_HEADS"), NH);
    dut.axiWrite(regmap.offsetOf("N_HEAD_KV"), NHKV);
    dut.axiWrite(regmap.offsetOf("HEAD_RATIO"), HEAD_RATIO);
    dut.axiWrite(regmap.offsetOf("N_KV"), NKV);
    dut.axiWrite(regmap.offsetOf("N_TOKENS"), NTOK);
    dut.axiWrite(regmap.offsetOf("SCALE"), f32bits(SCALE));
    const hdq_rb = dut.axiRead(regmap.offsetOf("HEAD_DIM_Q"));
    dut.axiWrite(regmap.offsetOf("CTRL"), 1);
    c.dut_axi_idle(dut.h);

    var qc: usize = 0;
    var kc: usize = 0;
    var vc: usize = 0;
    var mc: usize = 0;
    var oc: usize = 0;
    var output_beat: usize = 0;
    var o_got: Output = undefined;

    // Each selected beat is withheld for exactly one cycle in which the consumer
    // is ready. This makes the expected K/V/O stall counts independent of the
    // kernel's internal cycle schedule.
    const no_beat: usize = std.math.maxInt(usize);
    var k_bubbled_for: usize = no_beat;
    var v_bubbled_for: usize = no_beat;
    var o_stalled_for: usize = no_beat;
    var observed_k_stalls: u32 = 0;
    var observed_v_stalls: u32 = 0;
    var observed_o_stalls: u32 = 0;

    var held_valid = false;
    var held_data: [8]u32 = undefined;
    var held_keep: u32 = 0;
    var held_last = false;
    var framing_ok = true;

    var loop_cycles: usize = 0;
    while (loop_cycles < 100000 and oc < o_got.len) : (loop_cycles += 1) {
        const q_valid = qc < q_seq.len;
        const k_pending = kc < k_seq.len;
        const v_pending = vc < v_seq.len;
        const mask_valid = mc < mask_seq.len;
        const k_bubble = inject_stalls and k_pending and kc % 2 == 0 and k_bubbled_for != kc;
        const v_bubble = inject_stalls and v_pending and vc % 2 == 1 and v_bubbled_for != vc;
        const o_stall = inject_stalls and output_beat % 2 == 0 and o_stalled_for != output_beat;
        const k_valid = k_pending and !k_bubble;
        const v_valid = v_pending and !v_bubble;
        const o_ready = !o_stall;

        c.dut_set_q(dut.h, if (q_valid) &q_seq[qc] else &[_]u32{0} ** 8, @intFromBool(q_valid));
        c.dut_set_k(dut.h, if (k_valid) &k_seq[kc] else &[_]u32{0} ** 4, @intFromBool(k_valid));
        c.dut_set_v(dut.h, if (v_valid) &v_seq[vc] else &[_]u32{0} ** 4, @intFromBool(v_valid));
        c.dut_set_mask(dut.h, if (mask_valid) mask_seq[mc] else 0, @intFromBool(mask_valid));
        c.dut_set_o_ready(dut.h, @intFromBool(o_ready));
        dut.eval();

        const q_ready = c.dut_q_ready(dut.h) != 0;
        const k_ready = c.dut_k_ready(dut.h) != 0;
        const v_ready = c.dut_v_ready(dut.h) != 0;
        const mask_ready = c.dut_mask_ready(dut.h) != 0;
        const o_valid = c.dut_o_valid(dut.h) != 0;

        if (k_ready and !k_valid) {
            observed_k_stalls += 1;
            if (k_bubble) k_bubbled_for = kc;
        }
        if (v_ready and !v_valid) {
            observed_v_stalls += 1;
            if (v_bubble) v_bubbled_for = vc;
        }

        var beat: [8]u32 = undefined;
        if (o_valid) c.dut_o_data(dut.h, &beat);
        const o_keep = c.dut_o_keep(dut.h);
        const o_last = c.dut_o_last(dut.h) != 0;

        if (held_valid) {
            if (!o_valid or !std.mem.eql(u32, held_data[0..], beat[0..]) or
                held_keep != o_keep or held_last != o_last)
            {
                std.debug.print("  FAIL {s}: output changed while backpressured\n", .{name});
                framing_ok = false;
            }
        }
        if (o_valid and !o_ready) {
            observed_o_stalls += 1;
            o_stalled_for = output_beat;
            if (!held_valid) {
                held_valid = true;
                held_data = beat;
                held_keep = o_keep;
                held_last = o_last;
            }
        } else if (o_valid and o_ready) {
            const expect_last = output_beat + 1 == NTOK * NH * VB;
            if (o_keep != 0xFFFFFFFF or o_last != expect_last) {
                std.debug.print("  FAIL {s}: O beat {d} keep=0x{X:0>8} last={any}, expected keep=all last={any}\n", .{ name, output_beat, o_keep, o_last, expect_last });
                framing_ok = false;
            }
            if (oc + 8 > o_got.len) return error.ExtraOutputs;
            for (0..8) |i| {
                o_got[oc] = bitsf32(beat[i]);
                oc += 1;
            }
            output_beat += 1;
            held_valid = false;
        } else if (held_valid) {
            std.debug.print("  FAIL {s}: output valid dropped while backpressured\n", .{name});
            framing_ok = false;
        }

        dut.step();
        if (q_valid and q_ready) qc += 1;
        if (k_valid and k_ready) kc += 1;
        if (v_valid and v_ready) vc += 1;
        if (mask_valid and mask_ready) mc += 1;
    }

    c.dut_set_q(dut.h, &[_]u32{0} ** 8, 0);
    c.dut_set_k(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_v(dut.h, &[_]u32{0} ** 4, 0);
    c.dut_set_mask(dut.h, 0, 0);
    c.dut_set_o_ready(dut.h, 1);
    for (0..8) |_| dut.step();

    const status = dut.axiRead(regmap.offsetOf("STATUS"));
    const cycles = dut.axiRead(regmap.offsetOf("CYCLES"));
    const q_beats = dut.axiRead(regmap.offsetOf("Q_BEATS"));
    const k_beats = dut.axiRead(regmap.offsetOf("K_BEATS"));
    const k_stalls = dut.axiRead(regmap.offsetOf("K_STALL"));
    const v_beats = dut.axiRead(regmap.offsetOf("V_BEATS"));
    const v_stalls = dut.axiRead(regmap.offsetOf("V_STALL"));
    const o_beats = dut.axiRead(regmap.offsetOf("O_BEATS"));
    const o_stalls = dut.axiRead(regmap.offsetOf("O_STALL"));
    const id = dut.axiRead(regmap.offsetOf("ID"));

    var max_rel: f64 = 0;
    if (oc == o_got.len) {
        for (0..o_got.len) |i| {
            const relative = @abs(@as(f64, o_exp[i]) - @as(f64, o_got[i])) /
                @max(@abs(@as(f64, o_exp[i])), 1.0);
            max_rel = @max(max_rel, relative);
        }
    }

    std.debug.print("\n  flash_top {s}: loop={d} cycles_reg={d}, O {d}/{d}, max_rel={e:.4}\n", .{ name, loop_cycles, cycles, oc, o_got.len, max_rel });
    std.debug.print("    beats q/k/v/o={d}/{d}/{d}/{d}; stalls k/v/o={d}/{d}/{d}\n", .{ q_beats, k_beats, v_beats, o_beats, k_stalls, v_stalls, o_stalls });

    var ok = framing_ok;
    if (oc != o_got.len) {
        std.debug.print("  FAIL {s}: O incomplete ({d}/{d})\n", .{ name, oc, o_got.len });
        ok = false;
    }
    if (qc != q_seq.len or kc != k_seq.len or vc != v_seq.len or mc != mask_seq.len) {
        std.debug.print("  FAIL {s}: input streams incomplete q/k/v/m={d}/{d}/{d}/{d}\n", .{ name, qc, kc, vc, mc });
        ok = false;
    }
    if (max_rel > 0.02) {
        std.debug.print("  FAIL {s}: O exceeds tolerance\n", .{name});
        ok = false;
    }
    if (hdq_rb != HDQ or id != regmap.resetOf("ID")) {
        std.debug.print("  FAIL {s}: register readback hdq={d}, id=0x{X:0>8}\n", .{ name, hdq_rb, id });
        ok = false;
    }
    if ((status >> 1) & 1 != 1 or status & 1 != 0) {
        std.debug.print("  FAIL {s}: terminal STATUS=0x{X}\n", .{ name, status });
        ok = false;
    }

    const exp_q = NTOK * NH * QB;
    const exp_k = NTOK * NKV * NHKV * QB;
    const exp_v = NTOK * NKV * NHKV * VB;
    const exp_o = NTOK * NH * VB;
    if (q_beats != exp_q or k_beats != exp_k or v_beats != exp_v or o_beats != exp_o) {
        std.debug.print("  FAIL {s}: beat counters expected q/k/v/o={d}/{d}/{d}/{d}\n", .{ name, exp_q, exp_k, exp_v, exp_o });
        ok = false;
    }
    if (k_stalls != observed_k_stalls or v_stalls != observed_v_stalls or o_stalls != observed_o_stalls) {
        std.debug.print("  FAIL {s}: observed stalls k/v/o={d}/{d}/{d}\n", .{ name, observed_k_stalls, observed_v_stalls, observed_o_stalls });
        ok = false;
    }
    if (cycles == 0) {
        std.debug.print("  FAIL {s}: cycle counter is zero\n", .{name});
        ok = false;
    }
    if (inject_stalls and (k_stalls == 0 or v_stalls == 0 or o_stalls == 0)) {
        std.debug.print("  FAIL {s}: deterministic stalls did not exercise every counted stream\n", .{name});
        ok = false;
    }
    if (!inject_stalls and (k_stalls != 0 or v_stalls != 0 or o_stalls != 0)) {
        std.debug.print("  FAIL {s}: per-run stall counters were not clear\n", .{name});
        ok = false;
    }
    if (!ok) return error.Failed;

    return .{ .cycles = cycles, .k_stalls = k_stalls, .v_stalls = v_stalls, .o_stalls = o_stalls };
}

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
    var q_seq: QSeq = undefined;
    var k_seq: KSeq = undefined;
    var v_seq: VSeq = undefined;
    var mask_seq: MaskSeq = undefined;
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
    resetDut(&dut);

    const stalled = try runOnce(&dut, "stalled", &q_seq, &k_seq, &v_seq, &mask_seq, &o_exp, true);
    const clean = try runOnce(&dut, "clean-after-stalled", &q_seq, &k_seq, &v_seq, &mask_seq, &o_exp, false);
    if (stalled.cycles <= clean.cycles) {
        std.debug.print("  FAIL: stalled run cycles {d} did not exceed clean run {d}\n", .{ stalled.cycles, clean.cycles });
        return error.Failed;
    }
    if (clean.k_stalls != 0 or clean.v_stalls != 0 or clean.o_stalls != 0) return error.Failed;
    std.debug.print("  all flash_top integration checks passed\n\n", .{});
}
