//! M2 cosim: drive the Verilated q1a8_kernel from Zig, feed it pack.zig output,
//! and check the result stream against matmul_ref. The integer matmul is the
//! invariant (matmul_ref tests it `==`); the kernel folds in fp16 scales with
//! round-toward-zero custom FP, so at the kernel boundary we compare the final
//! fp32 within a relative tolerance (plan §11: float scaling is ε, not `==`).

const std = @import("std");
const q1a8 = @import("q1a8");
const pack = @import("pack");
const ref = @import("matmul_ref");

const c = @cImport(@cInclude("shim.h"));

const CYCLE_LIMIT: usize = 2_000_000;

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

const Run = struct { out: []f32, busy_cycles: usize, state_hist: [16]usize };

const state_names = [_][]const u8{ "IDLE", "LOAD_ACTS", "LOAD_SCALE", "SCALES", "WBITS", "ISSUE", "WAIT_DONE", "DRAIN", "EMIT", "FINISH" };

/// Run one matmul through the DUT, returning per-row fp32 results and the
/// number of cycles the kernel was busy. Streams are never stalled, so
/// busy_cycles is the kernel's intrinsic (compute-bound) time for this shape.
fn runKernel(
    a: std.mem.Allocator,
    rows: usize,
    blocks: usize,
    w_bytes: []const u8,
    a_bytes: []const u8,
) !Run {
    const num_rb = rows / q1a8.ROWS;
    var dut = Dut.init();
    defer dut.deinit();

    // weight/act streams as 64-bit beats
    const w_beats = std.mem.bytesAsSlice(u64, w_bytes);
    const a_beats = std.mem.bytesAsSlice(u64, a_bytes);
    const res_bytes = try a.alloc(u8, pack.resultBytes(num_rb));
    defer a.free(res_bytes);

    // reset
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_start(dut.h, 0);
    c.dut_set_w(dut.h, 0, 0);
    c.dut_set_a(dut.h, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..4) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    c.dut_set_num_q1(dut.h, @intCast(blocks));
    c.dut_set_num_rb(dut.h, @intCast(num_rb));

    var wi: usize = 0;
    var ai: usize = 0;
    var ri: usize = 0; // result byte cursor
    var busy_cycles: usize = 0;
    var state_hist = [_]usize{0} ** 16;
    var cycle: usize = 0;
    while (cycle < CYCLE_LIMIT) : (cycle += 1) {
        c.dut_set_start(dut.h, if (cycle == 0) 1 else 0);

        const w_valid = wi < w_beats.len;
        const a_valid = ai < a_beats.len;
        c.dut_set_w(dut.h, if (w_valid) w_beats[wi] else 0, @intFromBool(w_valid));
        c.dut_set_a(dut.h, if (a_valid) a_beats[ai] else 0, @intFromBool(a_valid));
        c.dut_set_m_ready(dut.h, 1);
        c.dut_eval(dut.h); // settle combinational at clk=0

        const w_fire = w_valid and c.dut_w_ready(dut.h) != 0;
        const a_fire = a_valid and c.dut_a_ready(dut.h) != 0;
        if (c.dut_m_valid(dut.h) != 0) {
            if (ri + 8 > res_bytes.len) return error.TooManyResultBeats;
            std.mem.writeInt(u64, res_bytes[ri..][0..8], c.dut_m_data(dut.h), .little);
            ri += 8;
        }

        if (c.dut_busy(dut.h) != 0) {
            busy_cycles += 1;
            state_hist[@intCast(c.dut_state(dut.h) & 0xf)] += 1;
        }
        dut.step(); // rising edge commits transfers, then settle low

        if (w_fire) wi += 1;
        if (a_fire) ai += 1;
        if (c.dut_done(dut.h) != 0) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.KernelTimeout;
    if (ri != res_bytes.len) return error.MissingResultBeats;

    const out = try a.alloc(f32, rows);
    pack.unpackResults(num_rb, res_bytes, out);
    return .{ .out = out, .busy_cycles = busy_cycles, .state_hist = state_hist };
}

fn runCase(a: std.mem.Allocator, rows: usize, blocks: usize, seed: u64) !void {
    const num_rb = rows / q1a8.ROWS;

    const bits = try a.alloc(u128, rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, rows * blocks);
    defer a.free(wscales);
    const column = try a.alloc(f32, blocks * q1a8.Q1_BLOCK);
    defer a.free(column);

    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (bits) |*b| b.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
    for (wscales) |*s| s.* = @floatCast(0.01 + rnd.float(f32) * 0.2);
    for (column) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;

    const aquants = try a.alloc(i8, column.len);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, blocks * q1a8.Q8_SUBBLOCKS);
    defer a.free(ascales);
    pack.quantizeActs(column, aquants, ascales);

    const w_bytes = try a.alloc(u8, pack.weightBytes(num_rb, blocks));
    defer a.free(w_bytes);
    pack.packWeights(rows, blocks, bits, wscales, w_bytes);
    const a_bytes = try a.alloc(u8, pack.actBytes(blocks));
    defer a.free(a_bytes);
    pack.packActs(blocks, aquants, ascales, a_bytes);

    const expected = try a.alloc(f32, rows);
    defer a.free(expected);
    ref.scaledOutput(.{
        .rows = rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_scales = wscales,
        .act_quants = aquants,
        .act_scales = ascales,
    }, expected);

    const run = try runKernel(a, rows, blocks, w_bytes, a_bytes);
    defer a.free(run.out);

    var max_rel: f32 = 0;
    for (run.out, expected, 0..) |g, e, i| {
        const denom = @max(@abs(e), 1.0);
        const rel = @abs(g - e) / denom;
        max_rel = @max(max_rel, rel);
        if (rel > 0.02) {
            std.debug.print(
                "  MISMATCH row={d} got={d:.5} exp={d:.5} rel={d:.4}\n",
                .{ i, g, e, rel },
            );
        }
    }
    // Intrinsic throughput: weight-bits (= MACs) processed per busy cycle. The
    // four-port DDR fixture delivers 12.1 GB/s; minus the ~11% fp16 weight-scale
    // overhead (128/144 of the stream) that is ~286 weight-bits/cycle at 300 MHz.
    // Acts (int8 + scales) and result writes are <~1% of decode DDR traffic (the
    // K-vector is read once and reused across all M rows), so weights set the
    // budget. bits/cycle < ~286 means the array is compute-bound.
    const weight_bits = rows * blocks * q1a8.Q1_BLOCK;
    const bpc = @as(f64, @floatFromInt(weight_bits)) / @as(f64, @floatFromInt(run.busy_cycles));
    std.debug.print(
        "case rows={d} blocks={d} ok={d} max_rel={d:.4} busy_cycles={d} MAC/cycle={d:.1} (DDR budget ~286)\n",
        .{ rows, blocks, @intFromBool(max_rel <= 0.02), max_rel, run.busy_cycles, bpc },
    );
    if (rows >= 64) {
        std.debug.print("  cycle breakdown:", .{});
        for (run.state_hist[0..state_names.len], state_names) |n, name| {
            if (n > 0) std.debug.print(" {s}={d}({d}%)", .{ name, n, n * 100 / run.busy_cycles });
        }
        std.debug.print("\n", .{});
    }
    if (max_rel > 0.02) return error.ResultMismatch;
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const Case = struct { rows: usize, blocks: usize };
    const cases = [_]Case{
        .{ .rows = 8, .blocks = 1 }, // 1 rowblock, K=128
        .{ .rows = 8, .blocks = 4 }, // multi-block accumulation
        .{ .rows = 24, .blocks = 2 }, // multi-rowblock broadcast
        .{ .rows = 64, .blocks = 16 }, // Bonsai-representative (M=64, K=2048)
    };
    for (cases, 0..) |cs, i| try runCase(a, cs.rows, cs.blocks, 0x1000 + i);
    std.debug.print("all cosim cases passed\n", .{});
}
