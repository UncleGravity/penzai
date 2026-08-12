//! Cosim for flash_top: the AXI-Lite/DMA wrapper around flash_kernel. Configures the
//! kernel via AXI-Lite register writes, streams Q/K/V/mask in and O out (same feed as
//! the kernel cosim), checks O vs flash_ref, then reads STATUS + the counter bank back
//! over AXI-Lite. Register offsets come from the generated regmap (single source).

const std = @import("std");
const ref = @import("flash_ref");
const regmap = @import("regmap");

const c = @cImport(@cInclude("shim.h"));

const MaskKind = enum { interior, causal };

const Cfg = struct {
    hdq: usize,
    hdv: usize,
    nh: usize,
    nhkv: usize,
    nkv: usize,
    ntok: usize,
    scale: f32,
    mask_kind: MaskKind,
    expected_hash: u64 = 0,
};

const legacy_cfg = Cfg{
    .hdq = 16,
    .hdv = 16,
    .nh = 2,
    .nhkv = 1,
    .nkv = 4,
    .ntok = 1,
    .scale = 0.25,
    .mask_kind = .interior,
};

const query_blocked_cfg = Cfg{
    .hdq = 8,
    .hdv = 8,
    .nh = 2,
    .nhkv = 1,
    .nkv = 5,
    .ntok = 4,
    .scale = 0.25,
    .mask_kind = .causal,
    .expected_hash = 0x07f8_6b6d_92f2_c961,
};

const wide_heads_cfg = Cfg{
    .hdq = 8,
    .hdv = 8,
    .nh = 32,
    .nhkv = 8,
    .nkv = 4,
    .ntok = 2,
    .scale = 0.25,
    .mask_kind = .causal,
    .expected_hash = 0x595f_47b4_3bdf_df06,
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

const RunStats = struct {
    cycles: u32,
    k_stalls: u32,
    v_stalls: u32,
    o_stalls: u32,
    hash: u64,
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
    comptime cfg: Cfg,
    dut: *Dut,
    name: []const u8,
    q_seq: []const [8]u32,
    k_seq: []const [4]u32,
    v_seq: []const [4]u32,
    mask_seq: []const u16,
    o_exp: []const f32,
    inject_stalls: bool,
) !RunStats {
    const qb = cfg.hdq / 8;
    const vb = cfg.hdv / 8;
    const head_ratio = cfg.nh / cfg.nhkv;

    // Reconfigure and start every run. In particular, CTRL.start must clear status
    // and all performance counters without relying on reset between invocations.
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_Q"), @intCast(cfg.hdq));
    dut.axiWrite(regmap.offsetOf("HEAD_DIM_V"), @intCast(cfg.hdv));
    dut.axiWrite(regmap.offsetOf("N_HEADS"), @intCast(cfg.nh));
    dut.axiWrite(regmap.offsetOf("N_HEAD_KV"), @intCast(cfg.nhkv));
    dut.axiWrite(regmap.offsetOf("HEAD_RATIO"), @intCast(head_ratio));
    dut.axiWrite(regmap.offsetOf("N_KV"), @intCast(cfg.nkv));
    dut.axiWrite(regmap.offsetOf("N_TOKENS"), @intCast(cfg.ntok));
    dut.axiWrite(regmap.offsetOf("SCALE"), f32bits(cfg.scale));
    const hdq_rb = dut.axiRead(regmap.offsetOf("HEAD_DIM_Q"));
    dut.axiWrite(regmap.offsetOf("CTRL"), 1);
    c.dut_axi_idle(dut.h);

    var qc: usize = 0;
    var kc: usize = 0;
    var vc: usize = 0;
    var mc: usize = 0;
    var oc: usize = 0;
    var output_beat: usize = 0;
    var o_got: [cfg.ntok * cfg.nh * cfg.hdv]f32 = undefined;

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
            const expect_last = output_beat + 1 == cfg.ntok * cfg.nh * vb;
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
    const version = dut.axiRead(regmap.offsetOf("VERSION"));

    var max_rel: f64 = 0;
    if (oc == o_got.len) {
        for (0..o_got.len) |i| {
            const relative = @abs(@as(f64, o_exp[i]) - @as(f64, o_got[i])) /
                @max(@abs(@as(f64, o_exp[i])), 1.0);
            max_rel = @max(max_rel, relative);
        }
    }

    const output_hash = if (oc == o_got.len) hashOutputs(&o_got) else 0;
    std.debug.print("\n  flash_top {s}: loop={d} cycles_reg={d}, O {d}/{d}, hash={x:0>16}, max_rel={e:.4}\n", .{ name, loop_cycles, cycles, oc, o_got.len, output_hash, max_rel });
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
    const query_slots = dut.axiRead(regmap.offsetOf("QUERY_SLOTS"));
    if (hdq_rb != @as(u32, @intCast(cfg.hdq)) or id != regmap.resetOf("ID") or
        version != regmap.resetOf("VERSION") or query_slots != regmap.resetOf("QUERY_SLOTS"))
    {
        std.debug.print("  FAIL {s}: register readback hdq={d}, id=0x{X:0>8}, version={d}, slots={d}\n", .{ name, hdq_rb, id, version, query_slots });
        ok = false;
    }
    if ((status >> 1) & 1 != 1 or status & 1 != 0) {
        std.debug.print("  FAIL {s}: terminal STATUS=0x{X}\n", .{ name, status });
        ok = false;
    }

    const exp_q = cfg.ntok * cfg.nh * qb;
    const exp_k = cfg.nkv * cfg.nhkv * qb;
    const exp_v = cfg.nkv * cfg.nhkv * vb;
    const exp_o = cfg.ntok * cfg.nh * vb;
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
    if (cfg.expected_hash != 0 and output_hash != cfg.expected_hash) {
        std.debug.print("  FAIL {s}: output hash 0x{x:0>16}, expected 0x{x:0>16}\n", .{ name, output_hash, cfg.expected_hash });
        ok = false;
    }
    if (!ok) return error.Failed;

    return .{ .cycles = cycles, .k_stalls = k_stalls, .v_stalls = v_stalls, .o_stalls = o_stalls, .hash = output_hash };
}

fn runCfg(comptime cfg: Cfg, dut: *Dut, seed: u64, name: []const u8, inject_stalls: bool) !RunStats {
    comptime {
        if (cfg.hdq % 8 != 0 or cfg.hdv % 8 != 0) @compileError("head dimensions must be multiples of 8");
        if (cfg.nhkv == 0 or cfg.nh % cfg.nhkv != 0) @compileError("invalid GQA head ratio");
        if (cfg.nkv == 0 or cfg.ntok == 0) @compileError("empty attention shape");
    }

    const qb = cfg.hdq / 8;
    const vb = cfg.hdv / 8;
    const head_ratio = cfg.nh / cfg.nhkv;

    // Source tensors use the runtime layout: Q is query/head-major, K/V are
    // historical-KV/KV-head-major, and the mask is query/KV-major in memory.
    var q: [cfg.ntok * cfg.nh * cfg.hdq]f32 = undefined;
    var k: [cfg.nkv * cfg.nhkv * cfg.hdq]u16 = undefined;
    var v: [cfg.nkv * cfg.nhkv * cfg.hdv]u16 = undefined;
    var mask: [cfg.ntok * cfg.nkv]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (&q) |*x| x.* = (rnd.float(f32) - 0.5) * 2.0;
    for (&k) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (&v) |*x| x.* = f16bits((rnd.float(f32) - 0.5) * 2.0);
    for (0..cfg.ntok) |t| for (0..cfg.nkv) |kv| {
        const value: f32 = switch (cfg.mask_kind) {
            .interior => if (kv == @min(2, cfg.nkv - 1)) -std.math.inf(f32) else 0.0,
            .causal => if (kv <= t) 0.0 else -std.math.inf(f32),
        };
        mask[t * cfg.nkv + kv] = f16bits(value);
    };

    // Mirror the hardware's online-softmax approximation and BF16 p*V split.
    var o_exp: [cfg.ntok * cfg.nh * cfg.hdv]f32 = undefined;
    for (0..cfg.ntok) |t| for (0..cfg.nh) |h| {
        const kvh = h / head_ratio;
        var acc = [_]f32{0} ** cfg.hdv;
        var m: f32 = -std.math.inf(f32);
        var l: f32 = 0;
        for (0..cfg.nkv) |kv| {
            const mvv = f16val(mask[t * cfg.nkv + kv]);
            if (!std.math.isFinite(mvv) and mvv < 0) continue;
            var dot: f32 = 0;
            for (0..cfg.hdq) |d| dot += q[(t * cfg.nh + h) * cfg.hdq + d] *
                f16val(k[(kv * cfg.nhkv + kvh) * cfg.hdq + d]);
            const score = dot * cfg.scale + mvv;
            const m_new = @max(m, score);
            if (m_new != m) {
                const corr = ref.softmaxExp(8, m - m_new);
                l *= corr;
                for (&acc) |*a| a.* *= corr;
                m = m_new;
            }
            const p = ref.softmaxExp(8, score - m);
            l += p;
            for (0..cfg.hdv) |d| acc[d] += ref.bf16MulPV(
                p,
                f16val(v[(kv * cfg.nhkv + kvh) * cfg.hdv + d]),
            );
        }
        if (l == 0) {
            for (0..cfg.hdv) |d| o_exp[(t * cfg.nh + h) * cfg.hdv + d] = 0;
        } else {
            const inv_l = ref.recip(8, l);
            for (0..cfg.hdv) |d| o_exp[(t * cfg.nh + h) * cfg.hdv + d] = acc[d] * inv_l;
        }
    };

    var q_seq: [cfg.ntok * cfg.nh * qb][8]u32 = undefined;
    var k_seq: [cfg.nkv * cfg.nhkv * qb][4]u32 = undefined;
    var v_seq: [cfg.nkv * cfg.nhkv * vb][4]u32 = undefined;
    var mask_seq: [cfg.ntok * cfg.nkv]u16 = undefined;
    var qi: usize = 0;
    var ki: usize = 0;
    var vi: usize = 0;
    var mi: usize = 0;

    for (0..cfg.ntok) |t| for (0..cfg.nh) |h| {
        for (0..qb) |b| {
            for (0..8) |i| q_seq[qi][i] = f32bits(q[(t * cfg.nh + h) * cfg.hdq + b * 8 + i]);
            qi += 1;
        }
    };

    // The v4 wire contract is KV outermost. Each KV position supplies all query
    // masks, then one native K/V block per KV head. K/V traffic is independent of
    // query count; the scheduler reuses the resident blocks for every query.
    for (0..cfg.nkv) |kv| {
        for (0..cfg.ntok) |t| {
            mask_seq[mi] = mask[t * cfg.nkv + kv];
            mi += 1;
        }
        for (0..cfg.nhkv) |kvh| for (0..qb) |b| {
            for (0..4) |w| k_seq[ki][w] = @as(u32, k[(kv * cfg.nhkv + kvh) * cfg.hdq + b * 8 + 2 * w]) |
                (@as(u32, k[(kv * cfg.nhkv + kvh) * cfg.hdq + b * 8 + 2 * w + 1]) << 16);
            ki += 1;
        };
        for (0..cfg.nhkv) |kvh| for (0..vb) |b| {
            for (0..4) |w| v_seq[vi][w] = @as(u32, v[(kv * cfg.nhkv + kvh) * cfg.hdv + b * 8 + 2 * w]) |
                (@as(u32, v[(kv * cfg.nhkv + kvh) * cfg.hdv + b * 8 + 2 * w + 1]) << 16);
            vi += 1;
        };
    }

    return runOnce(cfg, dut, name, &q_seq, &k_seq, &v_seq, &mask_seq, &o_exp, inject_stalls);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    resetDut(&dut);

    // Preserve the original wrapper reset regression: start a clean run directly
    // after a stalled ntok=1 invocation and require all per-run state to clear.
    const stalled = try runCfg(legacy_cfg, &dut, 0x70F70F, "ntok1-stalled", true);
    const clean = try runCfg(legacy_cfg, &dut, 0x70F70F, "ntok1-clean-after-stalled", false);
    if (stalled.cycles <= clean.cycles or stalled.hash != clean.hash) {
        std.debug.print("  FAIL: stalled ntok=1 run ({d} cycles) did not reproduce clean run ({d} cycles)\n", .{ stalled.cycles, clean.cycles });
        return error.Failed;
    }
    if (clean.k_stalls != 0 or clean.v_stalls != 0 or clean.o_stalls != 0) return error.Failed;

    // Cross the complete v4 wrapper boundary with several causal queries. This
    // catches a query-scaled K/V feed, token-outer mask order, and broken TLAST or
    // performance-counter plumbing that the direct kernel cosim cannot observe.
    _ = try runCfg(query_blocked_cfg, &dut, 0x4B10C, "ntok4-causal-stalled", true);

    // Cross the adaptive wide-head mapping through AXI-Lite as well: two 32-head
    // queries occupy all 64 reported physical state slots without aliasing.
    _ = try runCfg(wide_heads_cfg, &dut, 0x64A07, "ntok2-wide-heads", false);

    std.debug.print("  all flash_top integration checks passed\n\n", .{});
}
