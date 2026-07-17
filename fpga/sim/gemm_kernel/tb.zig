//! Cosim for gemm_kernel: the full fixed-point matmul kernel (FSM + banked gemm_rowblock +
//! time-muxed gemm_emit). Drives the AXIS weight/act streams and the global `emin`, reads the
//! result stream, and checks every (rowblock, column) BIT-EXACT against
//! matmul_ref.windowedFixedOutput — exact, not ε, because the oracle emits via the truncating
//! emitTrunc that models gemm_emit. C=1 is decode (one accumulator/row); C>1 is prefill (one
//! weight stream MAC'd against COLS_MAX columns — the bank + column sweep). Reuses pack.zig,
//! the deployed GEMM wire contract.

const std = @import("std");
const layout = @import("layout");
const pack = @import("pack");
const ref = @import("matmul_ref");
const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 8_000_000;
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

fn runKernel(a: std.mem.Allocator, rows: usize, blocks: usize, num_cols: usize, emin: i32, weight_fmt: u2, w_bytes: []const u8, a_bytes: []const u8) ![]u8 {
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
    c.dut_set_emin(dut.h, @intCast(emin));
    c.dut_set_weight_fmt(dut.h, weight_fmt);

    var wi: usize = 0;
    var ai: usize = 0;
    var ri: usize = 0;
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        c.dut_set_start(dut.h, if (cycle == 0) 1 else 0);

        const w_valid = wi < w_beats;
        var word = [_]u32{0} ** ROWS;
        if (w_valid) {
            const base = wi * pack.WIDE_BEAT_BYTES;
            for (0..ROWS) |k| word[k] = std.mem.readInt(u32, w_bytes[base + k * 4 ..][0..4], .little);
        }
        c.dut_set_w(dut.h, &word, @intCast(ROWS), @intFromBool(w_valid));

        const a_valid = ai < a_beats.len;
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid));
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h);

        const w_fire = w_valid and c.dut_w_ready(dut.h) != 0;
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (c.dut_m_valid(dut.h) != 0) {
            if (ri + 8 > res.len) return error.TooManyResultBeats;
            std.mem.writeInt(u64, res[ri..][0..8], c.dut_m_data(dut.h), .little);
            ri += 8;
        }

        dut.step();
        if (w_fire) wi += 1;
        if (a_fire) ai += 1;
        if (c.dut_done(dut.h) != 0) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.KernelTimeout;
    if (ri != res.len) return error.MissingResultBeats;
    return res;
}

fn runCase(a: std.mem.Allocator, num_rb: usize, blocks: usize, num_cols: usize, ternary: bool, seed: u64, note: []const u8) !void {
    const rows = num_rb * ROWS;
    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);
    const wscales_hi = try a.alloc(f16, rows * blocks);
    defer a.free(wscales_hi);
    const nonzero = try a.alloc(u128, rows * blocks);
    defer a.free(nonzero);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    if (ternary) {
        for (bits, nonzero) |*signs, *nz| {
            signs.* = 0;
            nz.* = 0;
            for (0..layout.Q1_BLOCK) |i| {
                const digit = rnd.intRangeLessThan(u2, 0, 3);
                if (digit != 1) nz.* |= @as(u128, 1) << @intCast(i);
                if (digit == 2) signs.* |= @as(u128, 1) << @intCast(i);
            }
        }
    } else {
        for (bits, nonzero) |*b, *nz| {
            b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
            nz.* = std.math.maxInt(u128);
        }
    }
    // narrow-band positive normal f16 weight scales (a realistic calibrated window).
    for (wscales) |*s| {
        const e: u16 = rnd.intRangeAtMost(u16, 13, 17);
        s.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }
    for (wscales_hi) |*s| {
        const e: u16 = rnd.intRangeAtMost(u16, 13, 17);
        s.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }

    const w_len = if (ternary) pack.ternaryWeightBytesWide(num_rb, blocks) else pack.weightBytesWide(num_rb, blocks);
    const w_bytes = try a.alloc(u8, w_len);
    defer a.free(w_bytes);
    if (ternary)
        pack.packTernaryWeightsWide(rows, blocks, bits, nonzero, wscales, wscales_hi, w_bytes)
    else
        pack.packWeightsWide(rows, blocks, bits, wscales, w_bytes);

    // num_cols activation columns, packed back to back; one Problem per column (shared weights).
    const a_bytes = try a.alloc(u8, num_cols * pack.actBytes(blocks));
    defer a.free(a_bytes);
    const aquants = try a.alloc(i8, num_cols * blocks * layout.Q1_BLOCK);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, num_cols * blocks * layout.Q8_SUBBLOCKS);
    defer a.free(ascales);
    const column = try a.alloc(f32, blocks * layout.Q1_BLOCK);
    defer a.free(column);

    const aq_per = blocks * layout.Q1_BLOCK;
    const as_per = blocks * layout.Q8_SUBBLOCKS;
    for (0..num_cols) |col| {
        for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        const aq = aquants[col * aq_per ..][0..aq_per];
        const as_ = ascales[col * as_per ..][0..as_per];
        pack.quantizeActs(column, aq, as_);
        pack.packActs(blocks, aq, as_, a_bytes[col * pack.actBytes(blocks) ..][0..pack.actBytes(blocks)]);
    }

    var probs = try a.alloc(ref.Problem, num_cols);
    defer a.free(probs);
    for (0..num_cols) |col| probs[col] = .{
        .rows = rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_nonzero = if (ternary) nonzero else null,
        .weight_scales = wscales,
        .weight_scales_hi = if (ternary) wscales_hi else null,
        .act_quants = aquants[col * aq_per ..][0..aq_per],
        .act_scales = ascales[col * as_per ..][0..as_per],
    };
    const w = ref.fixedWindow(); // the deployed fixed window (emin -48, ACC_W 104) — no calibration

    const expected = try a.alloc(f32, num_cols * rows);
    defer a.free(expected);
    var sats: usize = 0;
    for (0..num_cols) |col| {
        var s: usize = 0;
        ref.windowedFixedOutput(probs[col], w, expected[col * rows ..][0..rows], &s);
        sats += s;
    }

    const res = try runKernel(a, rows, blocks, num_cols, w.emin, if (ternary) 2 else 1, w_bytes, a_bytes);
    defer a.free(res);

    var mism: usize = 0;
    for (0..num_cols) |col| {
        for (0..num_rb) |rb| {
            const chunk = res[(rb * num_cols + col) * RESULT_BYTES_PER_RB ..][0..RESULT_BYTES_PER_RB];
            for (0..ROWS / 2) |beat| {
                const word = std.mem.readInt(u64, chunk[beat * 8 ..][0..8], .little);
                const got0: u32 = @truncate(word);
                const got1: u32 = @truncate(word >> 32);
                const row0 = rb * ROWS + beat * 2;
                const e0: u32 = @bitCast(expected[col * rows + row0]);
                const e1: u32 = @bitCast(expected[col * rows + row0 + 1]);
                if (got0 != e0 or got1 != e1) {
                    mism += 1;
                    if (mism <= 8)
                        std.debug.print("  col {d} rb {d} beat {d}: 0x{X:0>8}/0x{X:0>8} vs 0x{X:0>8}/0x{X:0>8}\n", .{ col, rb, beat, got0, got1, e0, e1 });
                }
            }
        }
    }

    std.debug.print("  case rows={d} blocks={d} cols={d} window=[{d},{d}] acc_w={d} sats={d} mism={d}  {s}\n", .{ rows, blocks, num_cols, w.emin, w.emax, w.acc_w, sats, mism, note });
    if (sats != 0) {
        std.debug.print("  FAIL: window did not cover the test problem (sats {d})\n", .{sats});
        return error.WindowTooNarrow;
    }
    if (mism != 0) {
        std.debug.print("  FAIL: gemm_kernel !== windowedFixedOutput\n", .{});
        return error.ResultMismatch;
    }
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const Case = struct { num_rb: usize, blocks: usize, cols: usize, ternary: bool = false, note: []const u8 = "" };
    const cases = [_]Case{
        .{ .num_rb = 1, .blocks = 1, .cols = 1, .note = "decode, 1 rb" },
        .{ .num_rb = 1, .blocks = 2, .cols = 4, .note = "prefill C=4" },
        .{ .num_rb = 3, .blocks = 2, .cols = 3, .note = "multi-rb, odd cols" },
        .{ .num_rb = 2, .blocks = 4, .cols = 8, .note = "prefill C=8 (full bank)" },
        .{ .num_rb = 8, .blocks = 16, .cols = 1, .note = "decode, attn-ish" },
        .{ .num_rb = 4, .blocks = 8, .cols = 8, .note = "prefill, larger" },
        .{ .num_rb = 1, .blocks = 2, .cols = 3, .ternary = true, .note = "ternary decode + prefill" },
    };
    for (cases, 0..) |cs, i| try runCase(a, cs.num_rb, cs.blocks, cs.cols, cs.ternary, 0x3000 + i, cs.note);
    std.debug.print("all gemm_kernel cosim cases passed (gemm_kernel === windowedFixedOutput, bit-exact)\n", .{});
}
