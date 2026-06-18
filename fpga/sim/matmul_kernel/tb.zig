//! Cosim for matmul_kernel: one weight stream, num_cols activation columns.
//! Drives the kernel at C=1 (must match the wide kernel) and C>1, checking every
//! column's results against matmul_ref. The win it proves: weights are read once
//! regardless of C (prefill amortization), and the column-pipeline alignment in
//! matmul_rowblock is bit-exact-within-ε.

const std = @import("std");
const layout = @import("layout");
const pack = @import("pack");
const ref = @import("matmul_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 4_000_000;
const ROWS = layout.ROWS;
const RESULT_BYTES_PER_RB = layout.RESULT_BYTES_PER_ROWBLOCK;

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

/// Per-run handshake counters, mirroring the silicon bank in
/// matmul_top.v:160-172 exactly: counted only while `busy`, beats =
/// valid&&ready, stalls = ready&&!valid. Lets the cosim print the same MAC/cyc,
/// util%, and W_STALL the board's `--prof` matmul detail shows — no translation.
const Counters = struct {
    busy: usize = 0,
    w_beats: usize = 0,
    a_beats: usize = 0,
    r_beats: usize = 0,
    w_stall: usize = 0,
    a_stall: usize = 0,
    r_stall: usize = 0,
};

/// Models a bandwidth/latency-limited weight feed (the AXI DMA + HP port) so the
/// cosim can reproduce the silicon W_STALL the perfect-feed default never shows.
/// Token bucket: `w_rate` beats accrue per cycle, capped at `w_burst`; a weight
/// beat is presented only when a whole token is available, spent when consumed.
/// The default (inf rate, inf burst) is "never the limit" — the original
/// always-valid behavior — so the unthrottled cases are unchanged.
const Feed = struct {
    w_rate: f64 = std.math.inf(f64),
    w_burst: f64 = std.math.inf(f64),
};

const Run = struct { res: []u8, ctr: Counters };

fn runKernel(a: std.mem.Allocator, rows: usize, blocks: usize, num_cols: usize, w_bytes: []const u8, a_bytes: []const u8, feed: Feed) !Run {
    const num_rb = rows / ROWS;
    var dut = Dut.init();
    defer dut.deinit();

    const a_beats = std.mem.bytesAsSlice(u64, a_bytes);
    const w_beats = w_bytes.len / pack.WIDE_BEAT_BYTES;
    const res = try a.alloc(u8, num_rb * num_cols * RESULT_BYTES_PER_RB);

    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_start(dut.h, 0);
    c.dut_set_a(dut.h, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
    var zero = [_]u32{0} ** ROWS;
    c.dut_set_w(dut.h, &zero, @intCast(ROWS), 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..4) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    c.dut_set_num_q1(dut.h, @intCast(blocks));
    c.dut_set_num_rb(dut.h, @intCast(num_rb));
    c.dut_set_num_cols(dut.h, @intCast(num_cols));

    var wi: usize = 0;
    var ai: usize = 0;
    var ri: usize = 0;
    var ctr: Counters = .{};
    var w_tokens: f64 = 0;
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        c.dut_set_start(dut.h, if (cycle == 0) 1 else 0);

        // Weight feed throttle: the token bucket caps how fast beats arrive, so a
        // beat already prepared may be withheld (tvalid=0) to model DMA starvation.
        w_tokens = @min(w_tokens + feed.w_rate, feed.w_burst);
        const w_have = wi < w_beats;
        const w_valid = w_have and w_tokens >= 1.0;
        var word = [_]u32{0} ** ROWS;
        if (w_have) {
            const base = wi * pack.WIDE_BEAT_BYTES;
            for (0..ROWS) |k| word[k] = std.mem.readInt(u32, w_bytes[base + k * 4 ..][0..4], .little);
        }
        c.dut_set_w(dut.h, &word, @intCast(ROWS), @intFromBool(w_valid));

        const a_valid = ai < a_beats.len;
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid));
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);

        const w_ready = c.dut_w_ready(dut.h) != 0;
        const a_ready = c.dut_a_ready(dut.h) != 0;
        const m_valid = c.dut_m_valid(dut.h) != 0;
        const w_fire = w_valid and w_ready;
        const a_fire = a_valid and a_ready;
        if (m_valid) {
            if (ri + 8 > res.len) return error.TooManyResultBeats;
            std.mem.writeInt(u64, res[ri..][0..8], c.dut_m_data(dut.h), .little);
            ri += 8;
        }
        // Counter bank, sampled post-eval/pre-edge — the values the RTL's posedge
        // logic in matmul_top.v:160-172 would latch. m_ready is tied high
        // here, so r_stall stays 0 (the S2MM sink is always ready), as on silicon.
        if (c.dut_busy(dut.h) != 0) {
            ctr.busy += 1;
            if (w_fire) ctr.w_beats += 1;
            if (a_fire) ctr.a_beats += 1;
            if (m_valid) ctr.r_beats += 1;
            if (w_ready and !w_valid) ctr.w_stall += 1;
            if (a_ready and !a_valid) ctr.a_stall += 1;
        }

        dut.step();
        if (w_fire) {
            wi += 1;
            w_tokens -= 1.0;
        }
        if (a_fire) ai += 1;
        if (c.dut_done(dut.h) != 0) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.KernelTimeout;
    if (ri != res.len) return error.MissingResultBeats;
    return .{ .res = res, .ctr = ctr };
}

fn runCase(a: std.mem.Allocator, rows: usize, blocks: usize, num_cols: usize, seed: u64, feed: Feed, note: []const u8) !void {
    const num_rb = rows / ROWS;
    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (wscales) |*s| s.* = @floatCast(0.01 + rnd.float(f32) * 0.2);

    const w_bytes = try a.alloc(u8, pack.weightBytesWide(num_rb, blocks));
    defer a.free(w_bytes);
    pack.packWeightsWide(rows, blocks, bits, wscales, w_bytes);

    // num_cols activation columns, packed back to back; expected per column.
    const a_bytes = try a.alloc(u8, num_cols * pack.actBytes(blocks));
    defer a.free(a_bytes);
    const expected = try a.alloc(f32, num_cols * rows);
    defer a.free(expected);
    const column = try a.alloc(f32, blocks * layout.Q1_BLOCK);
    defer a.free(column);
    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * layout.Q8_SUBBLOCKS);
    defer a.free(ascales);

    for (0..num_cols) |col| {
        for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        pack.quantizeActs(column, aquants, ascales);
        pack.packActs(blocks, aquants, ascales, a_bytes[col * pack.actBytes(blocks) ..][0..pack.actBytes(blocks)]);
        ref.scaledOutput(.{
            .rows = rows,
            .q1_blocks = blocks,
            .weight_bits = bits,
            .weight_scales = wscales,
            .act_quants = aquants,
            .act_scales = ascales,
        }, expected[col * rows ..][0..rows]);
    }

    const run = try runKernel(a, rows, blocks, num_cols, w_bytes, a_bytes, feed);
    defer a.free(run.res);

    var max_rel: f32 = 0;
    for (0..num_cols) |col| {
        for (0..num_rb) |rb| {
            const chunk = run.res[(rb * num_cols + col) * RESULT_BYTES_PER_RB ..][0..RESULT_BYTES_PER_RB];
            for (0..ROWS / 2) |beat| {
                const w = std.mem.readInt(u64, chunk[beat * 8 ..][0..8], .little);
                const got0: f32 = @bitCast(@as(u32, @truncate(w)));
                const got1: f32 = @bitCast(@as(u32, @truncate(w >> 32)));
                const row0 = rb * ROWS + beat * 2;
                const e0 = expected[col * rows + row0];
                const e1 = expected[col * rows + row0 + 1];
                max_rel = @max(max_rel, @abs(got0 - e0) / @max(@abs(e0), 1.0));
                max_rel = @max(max_rel, @abs(got1 - e1) / @max(@abs(e1), 1.0));
            }
        }
    }

    const ctr = run.ctr;
    const macs = rows * blocks * layout.Q1_BLOCK * num_cols;
    const fbusy: f64 = @floatFromInt(ctr.busy);
    const bpc = @as(f64, @floatFromInt(macs)) / fbusy;
    const max_stall = @max(ctr.w_stall, @max(ctr.a_stall, ctr.r_stall));
    const util = 100.0 * (fbusy - @as(f64, @floatFromInt(max_stall))) / fbusy;
    const wstall_pct = 100.0 * @as(f64, @floatFromInt(ctr.w_stall)) / fbusy;
    std.debug.print("case rows={d} blocks={d} cols={d} feed={d:.2} ok={d} max_rel={d:.4} busy={d} MAC/cyc={d:.1} util={d:.1}% w_stall={d}({d:.1}%) w_beats={d} a_stall={d}  {s}\n", .{
        rows, blocks, num_cols, feed.w_rate, @intFromBool(max_rel <= 0.02), max_rel,
        ctr.busy, bpc, util, ctr.w_stall, wstall_pct, ctr.w_beats, ctr.a_stall, note,
    });
    if (max_rel > 0.02) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    // rows = num_rb * ROWS, so the cases scale with the array width (at ROWS=8
    // these are the familiar 8/24/64/2048-row shapes; at ROWS=16 they double).
    const Case = struct {
        num_rb: usize,
        blocks: usize,
        cols: usize,
        feed: Feed = .{},
        note: []const u8 = "",
    };
    const cases = [_]Case{
        .{ .num_rb = 1, .blocks = 1, .cols = 1, .note = "C=1 must match wide" },
        .{ .num_rb = 1, .blocks = 2, .cols = 4 },
        .{ .num_rb = 3, .blocks = 2, .cols = 3, .note = "multi-rb, odd cols" },
        .{ .num_rb = 8, .blocks = 16, .cols = 1, .note = "decode, full feed" },
        .{ .num_rb = 8, .blocks = 16, .cols = 8, .note = "prefill, full feed" },
        // Bonsai attn-size matmul so the cosim MAC/cyc is comparable to the
        // silicon decode aggregate. Full feed = the cols=1 compute ceiling.
        .{ .num_rb = 256, .blocks = 16, .cols = 1, .note = "decode attn-size, full feed" },
        // Supply sweep = the HP-port lever (beats here are ROWS*32-bit, so the
        // bandwidth a rate maps to scales with ROWS). Full-feed vs throttled shows
        // the compute ceiling vs the feed wall.
        .{ .num_rb = 256, .blocks = 16, .cols = 1, .feed = .{ .w_rate = 0.5, .w_burst = 1.0 }, .note = "decode attn @ 0.5 beat/cyc" },
        .{ .num_rb = 256, .blocks = 16, .cols = 1, .feed = .{ .w_rate = 1.0, .w_burst = 1.0 }, .note = "decode attn @ 1.0 beat/cyc" },
        // Throttled feed doubles as a backpressure-correctness test, and shows the
        // decode/prefill feed asymmetry: decode (no reuse) starves, prefill shrugs.
        .{ .num_rb = 8, .blocks = 16, .cols = 1, .feed = .{ .w_rate = 0.5, .w_burst = 1.0 }, .note = "decode @ 0.5 beat/cyc" },
        .{ .num_rb = 8, .blocks = 16, .cols = 8, .feed = .{ .w_rate = 0.5, .w_burst = 1.0 }, .note = "prefill @ 0.5 beat/cyc" },
    };
    for (cases, 0..) |cs, i| try runCase(a, cs.num_rb * ROWS, cs.blocks, cs.cols, 0x2000 + i, cs.feed, cs.note);
    std.debug.print("all matmul cosim cases passed\n", .{});
}
