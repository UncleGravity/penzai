//! Cosim for q1a8_kernel_mc: one weight stream, num_cols activation columns.
//! Drives the kernel at C=1 (must match the wide kernel) and C>1, checking every
//! column's results against matmul_ref. The win it proves: weights are read once
//! regardless of C (prefill amortization), and the column-pipeline alignment in
//! q1a8_rowblock_mc is bit-exact-within-ε.

const std = @import("std");
const q1a8 = @import("q1a8");
const pack = @import("pack");
const ref = @import("matmul_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 4_000_000;
const ROWS = q1a8.ROWS;
const RESULT_BYTES_PER_RB = q1a8.RESULT_BYTES_PER_ROWBLOCK;

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

const Run = struct { res: []u8, busy_cycles: usize };

fn runKernel(a: std.mem.Allocator, rows: usize, blocks: usize, num_cols: usize, w_bytes: []const u8, a_bytes: []const u8) !Run {
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
    var zero = [_]u32{0} ** 8;
    c.dut_set_w(dut.h, &zero, 0);
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
    var busy_cycles: usize = 0;
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        c.dut_set_start(dut.h, if (cycle == 0) 1 else 0);

        const w_valid = wi < w_beats;
        var word = [_]u32{0} ** 8;
        if (w_valid) {
            const base = wi * pack.WIDE_BEAT_BYTES;
            for (0..8) |k| word[k] = std.mem.readInt(u32, w_bytes[base + k * 4 ..][0..4], .little);
        }
        c.dut_set_w(dut.h, &word, @intFromBool(w_valid));

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
        if (c.dut_busy(dut.h) != 0) busy_cycles += 1;

        dut.step();
        if (w_fire) wi += 1;
        if (a_fire) ai += 1;
        if (c.dut_done(dut.h) != 0) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.KernelTimeout;
    if (ri != res.len) return error.MissingResultBeats;
    return .{ .res = res, .busy_cycles = busy_cycles };
}

fn runCase(a: std.mem.Allocator, rows: usize, blocks: usize, num_cols: usize, seed: u64) !void {
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
    const column = try a.alloc(f32, blocks * q1a8.Q1_BLOCK);
    defer a.free(column);
    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * q1a8.Q8_SUBBLOCKS);
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

    const run = try runKernel(a, rows, blocks, num_cols, w_bytes, a_bytes);
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

    const macs = rows * blocks * q1a8.Q1_BLOCK * num_cols;
    const bpc = @as(f64, @floatFromInt(macs)) / @as(f64, @floatFromInt(run.busy_cycles));
    std.debug.print("case rows={d} blocks={d} cols={d} ok={d} max_rel={d:.4} busy_cycles={d} MAC/cycle={d:.1}\n", .{
        rows, blocks, num_cols, @intFromBool(max_rel <= 0.02), max_rel, run.busy_cycles, bpc,
    });
    if (max_rel > 0.02) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const Case = struct { rows: usize, blocks: usize, cols: usize };
    const cases = [_]Case{
        .{ .rows = 8, .blocks = 1, .cols = 1 }, // C=1 must match the wide kernel
        .{ .rows = 8, .blocks = 2, .cols = 4 },
        .{ .rows = 24, .blocks = 2, .cols = 3 }, // multi-rowblock, odd cols
        .{ .rows = 64, .blocks = 16, .cols = 1 }, // Bonsai-rep decode
        .{ .rows = 64, .blocks = 16, .cols = 8 }, // Bonsai-rep prefill tile
    };
    for (cases, 0..) |cs, i| try runCase(a, cs.rows, cs.blocks, cs.cols, 0x2000 + i);
    std.debug.print("all mc cosim cases passed\n", .{});
}
