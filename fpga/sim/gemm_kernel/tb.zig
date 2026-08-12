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
const ACT_PACKED_LOAD: u2 = 0;
const ACT_REUSE: u2 = 1;

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

const KernelRun = struct {
    result: []u8,
    cycles: usize,
    weight_beats: usize,
    act_beats: usize,
    activation_error: bool,
    activation_valid: bool,
    loaded_epoch: u32,
    loaded_blocks: u16,
    loaded_cols: u16,
};

fn resetKernel(dut: *Dut) void {
    c.dut_set_rst_n(dut.h, 0);
    c.dut_set_start(dut.h, 0);
    c.dut_set_a(dut.h, 0, 0);
    c.dut_set_m_ready(dut.h, 1);
    c.dut_set_act_mode(dut.h, ACT_PACKED_LOAD);
    c.dut_set_act_epoch(dut.h, 0);
    c.dut_set_activation_abort(dut.h, 0);
    var zero = [_]u32{0} ** ROWS;
    c.dut_set_w(dut.h, &zero, @intCast(ROWS), 0);
    c.dut_set_clk(dut.h, 0);
    c.dut_eval(dut.h);
    for (0..4) |_| dut.step();
    c.dut_set_rst_n(dut.h, 1);
    dut.step();
}

fn runKernelOp(
    a: std.mem.Allocator,
    dut: *Dut,
    physical_rows: usize,
    logical_rows: usize,
    program_rows: usize,
    blocks: usize,
    num_cols: usize,
    emin: i32,
    weight_fmt: u2,
    act_mode: u2,
    act_epoch: u32,
    w_bytes: []const u8,
    a_bytes: []const u8,
    expect_error: bool,
) !KernelRun {
    const num_rb = physical_rows / ROWS;

    const a_beats = std.mem.bytesAsSlice(u64, a_bytes);
    const w_beats = w_bytes.len / pack.WIDE_BEAT_BYTES;
    const result_len = if (expect_error) 0 else logical_rows * num_cols * @sizeOf(f32);
    const res = try a.alloc(u8, result_len);
    errdefer a.free(res);

    c.dut_set_num_q1(dut.h, @intCast(blocks));
    c.dut_set_num_rb(dut.h, @intCast(num_rb));
    c.dut_set_num_rows(dut.h, @intCast(program_rows));
    c.dut_set_num_cols(dut.h, @intCast(num_cols));
    c.dut_set_emin(dut.h, @intCast(emin));
    c.dut_set_weight_fmt(dut.h, weight_fmt);
    c.dut_set_act_mode(dut.h, act_mode);
    c.dut_set_act_epoch(dut.h, act_epoch);
    c.dut_set_activation_abort(dut.h, 0);

    var wi: usize = 0;
    var ai: usize = 0;
    var ri: usize = 0;
    var saw_last = false;
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
        if (act_mode == ACT_REUSE and !expect_error and c.dut_a_ready(dut.h) != 0)
            return error.ReuseRequestedActivationData;
        if (c.dut_m_valid(dut.h) != 0) {
            const keep: u8 = @intCast(c.dut_m_keep(dut.h));
            const beat_bytes: usize = switch (keep) {
                0xFF => 8,
                0x0F => 4,
                else => return error.InvalidTkeep,
            };
            if (ri + beat_bytes > res.len) return error.UnexpectedResultBeat;
            var beat: [8]u8 = undefined;
            std.mem.writeInt(u64, &beat, c.dut_m_data(dut.h), .little);
            @memcpy(res[ri..][0..beat_bytes], beat[0..beat_bytes]);
            ri += beat_bytes;
            if (c.dut_m_last(dut.h) != 0) {
                if (saw_last or ri != res.len) return error.InvalidTlast;
                saw_last = true;
            }
        }

        dut.step();
        if (w_fire) wi += 1;
        if (a_fire) ai += 1;
        if (c.dut_done(dut.h) != 0) break;
    }
    if (cycle >= CYCLE_LIMIT) return error.KernelTimeout;
    const activation_error = c.dut_activation_error(dut.h) != 0;
    if (expect_error) {
        if (!activation_error or wi != 0 or ai != 0 or ri != 0 or saw_last)
            return error.InvalidReuseDidNotFailClosed;
    } else if (activation_error or ri != res.len or !saw_last) {
        return error.MissingResultBytes;
    }
    return .{
        .result = res,
        .cycles = cycle + 1,
        .weight_beats = wi,
        .act_beats = ai,
        .activation_error = activation_error,
        .activation_valid = c.dut_activation_valid(dut.h) != 0,
        .loaded_epoch = c.dut_loaded_act_epoch(dut.h),
        .loaded_blocks = @intCast(c.dut_loaded_act_q1_blocks(dut.h)),
        .loaded_cols = @intCast(c.dut_loaded_act_cols(dut.h)),
    };
}

fn runKernel(a: std.mem.Allocator, physical_rows: usize, logical_rows: usize, program_rows: usize, blocks: usize, num_cols: usize, emin: i32, weight_fmt: u2, w_bytes: []const u8, a_bytes: []const u8) ![]u8 {
    var dut = Dut.init();
    defer dut.deinit();
    resetKernel(&dut);
    const run = try runKernelOp(a, &dut, physical_rows, logical_rows, program_rows, blocks, num_cols, emin, weight_fmt, ACT_PACKED_LOAD, 0, w_bytes, a_bytes, false);
    return run.result;
}

fn runCase(a: std.mem.Allocator, num_rb: usize, logical_rows_raw: ?usize, program_rows_raw: ?usize, blocks: usize, num_cols: usize, ternary: bool, seed: u64, note: []const u8) !void {
    const physical_rows = num_rb * ROWS;
    const logical_rows = logical_rows_raw orelse physical_rows;
    const program_rows = program_rows_raw orelse logical_rows;
    if (logical_rows <= physical_rows - ROWS or logical_rows > physical_rows) return error.InvalidLogicalRows;
    const bits = try a.alloc(u128, physical_rows * blocks);
    defer a.free(bits);
    const wscales = try a.alloc(f16, physical_rows * blocks);
    defer a.free(wscales);
    const wscales_hi = try a.alloc(f16, physical_rows * blocks);
    defer a.free(wscales_hi);
    const nonzero = try a.alloc(u128, physical_rows * blocks);
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
        pack.packTernaryWeightsWide(physical_rows, blocks, bits, nonzero, wscales, wscales_hi, w_bytes)
    else
        pack.packWeightsWide(physical_rows, blocks, bits, wscales, w_bytes);

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
        .rows = physical_rows,
        .q1_blocks = blocks,
        .weight_bits = bits,
        .weight_nonzero = if (ternary) nonzero else null,
        .weight_scales = wscales,
        .weight_scales_hi = if (ternary) wscales_hi else null,
        .act_quants = aquants[col * aq_per ..][0..aq_per],
        .act_scales = ascales[col * as_per ..][0..as_per],
    };
    const w = ref.fixedWindow(); // the deployed fixed window (emin -48, ACC_W 104) — no calibration

    const expected = try a.alloc(f32, num_cols * physical_rows);
    defer a.free(expected);
    var sats: usize = 0;
    for (0..num_cols) |col| {
        var s: usize = 0;
        ref.windowedFixedOutput(probs[col], w, expected[col * physical_rows ..][0..physical_rows], &s);
        sats += s;
    }

    const res = try runKernel(a, physical_rows, logical_rows, program_rows, blocks, num_cols, w.emin, if (ternary) 2 else 1, w_bytes, a_bytes);
    defer a.free(res);

    var mism: usize = 0;
    var result_offset: usize = 0;
    for (0..num_rb) |rb| {
        const row0 = rb * ROWS;
        const valid_rows = @min(ROWS, logical_rows - row0);
        for (0..num_cols) |col| {
            for (0..valid_rows) |lane| {
                const got = std.mem.readInt(u32, res[result_offset..][0..4], .little);
                result_offset += 4;
                const expected_bits: u32 = @bitCast(expected[col * physical_rows + row0 + lane]);
                if (got != expected_bits) {
                    mism += 1;
                    if (mism <= 8)
                        std.debug.print("  col {d} rb {d} lane {d}: 0x{X:0>8} vs 0x{X:0>8}\n", .{ col, rb, lane, got, expected_bits });
                }
            }
        }
    }
    if (result_offset != res.len) return error.InvalidResultLength;

    std.debug.print("  case rows={d}/{d} blocks={d} cols={d} window=[{d},{d}] acc_w={d} sats={d} mism={d}  {s}\n", .{ logical_rows, physical_rows, blocks, num_cols, w.emin, w.emax, w.acc_w, sats, mism, note });
    if (sats != 0) {
        std.debug.print("  FAIL: window did not cover the test problem (sats {d})\n", .{sats});
        return error.WindowTooNarrow;
    }
    if (mism != 0) {
        std.debug.print("  FAIL: gemm_kernel !== windowedFixedOutput\n", .{});
        return error.ResultMismatch;
    }
}

fn runResidentReuseCase(a: std.mem.Allocator) !void {
    const blocks: usize = 2;
    const cols: usize = 3;
    const rows: usize = ROWS;
    const epoch: u32 = 0xA17C_0042;
    var prng = std.Random.DefaultPrng.init(0xA17C_5EED);
    const rnd = prng.random();

    const bits_load = try a.alloc(u128, rows * blocks);
    defer a.free(bits_load);
    const bits_reuse = try a.alloc(u128, rows * blocks);
    defer a.free(bits_reuse);
    const scales_load = try a.alloc(f16, rows * blocks);
    defer a.free(scales_load);
    const scales_reuse = try a.alloc(f16, rows * blocks);
    defer a.free(scales_reuse);
    for (bits_load, bits_reuse) |*first, *second| {
        first.* = (@as(u128, rnd.int(u64)) << 64) | rnd.int(u64);
        second.* = ~first.*;
    }
    for (scales_load, scales_reuse) |*first, *second| {
        const e: u16 = rnd.intRangeAtMost(u16, 13, 17);
        first.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
        second.* = @bitCast((e << 10) | rnd.intRangeLessThan(u16, 0, 1024));
    }

    const weight_len = pack.weightBytesWide(1, blocks);
    const weights_load = try a.alloc(u8, weight_len);
    defer a.free(weights_load);
    const weights_reuse = try a.alloc(u8, weight_len);
    defer a.free(weights_reuse);
    pack.packWeightsWide(rows, blocks, bits_load, scales_load, weights_load);
    pack.packWeightsWide(rows, blocks, bits_reuse, scales_reuse, weights_reuse);

    const aq_per = blocks * layout.Q1_BLOCK;
    const as_per = blocks * layout.Q8_SUBBLOCKS;
    const act_bytes = try a.alloc(u8, cols * pack.actBytes(blocks));
    defer a.free(act_bytes);
    const aquants = try a.alloc(i8, cols * aq_per);
    defer a.free(aquants);
    const ascales = try a.alloc(f16, cols * as_per);
    defer a.free(ascales);
    const column = try a.alloc(f32, aq_per);
    defer a.free(column);
    for (0..cols) |col| {
        for (column) |*value| value.* = (rnd.float(f32) - 0.5) * 4.0;
        const aq = aquants[col * aq_per ..][0..aq_per];
        const as_ = ascales[col * as_per ..][0..as_per];
        pack.quantizeActs(column, aq, as_);
        pack.packActs(blocks, aq, as_, act_bytes[col * pack.actBytes(blocks) ..][0..pack.actBytes(blocks)]);
    }

    const window = ref.fixedWindow();
    const expected_load = try a.alloc(f32, cols * rows);
    defer a.free(expected_load);
    const expected_reuse = try a.alloc(f32, cols * rows);
    defer a.free(expected_reuse);
    var saturations: usize = 0;
    for (0..cols) |col| {
        const common = .{
            .rows = rows,
            .q1_blocks = blocks,
            .weight_nonzero = @as(?[]const u128, null),
            .weight_scales_hi = @as(?[]const f16, null),
            .act_quants = aquants[col * aq_per ..][0..aq_per],
            .act_scales = ascales[col * as_per ..][0..as_per],
        };
        var sat: usize = 0;
        ref.windowedFixedOutput(.{
            .rows = common.rows,
            .q1_blocks = common.q1_blocks,
            .weight_bits = bits_load,
            .weight_nonzero = common.weight_nonzero,
            .weight_scales = scales_load,
            .weight_scales_hi = common.weight_scales_hi,
            .act_quants = common.act_quants,
            .act_scales = common.act_scales,
        }, window, expected_load[col * rows ..][0..rows], &sat);
        saturations += sat;
        sat = 0;
        ref.windowedFixedOutput(.{
            .rows = common.rows,
            .q1_blocks = common.q1_blocks,
            .weight_bits = bits_reuse,
            .weight_nonzero = common.weight_nonzero,
            .weight_scales = scales_reuse,
            .weight_scales_hi = common.weight_scales_hi,
            .act_quants = common.act_quants,
            .act_scales = common.act_scales,
        }, window, expected_reuse[col * rows ..][0..rows], &sat);
        saturations += sat;
    }
    if (saturations != 0) return error.WindowTooNarrow;

    var dut = Dut.init();
    defer dut.deinit();
    resetKernel(&dut);

    const load = try runKernelOp(a, &dut, rows, rows, rows, blocks, cols, window.emin, 1, ACT_PACKED_LOAD, epoch, weights_load, act_bytes, false);
    defer a.free(load.result);
    if (!std.mem.eql(u8, load.result, std.mem.sliceAsBytes(expected_load)) or
        !load.activation_valid or load.loaded_epoch != epoch or
        load.loaded_blocks != blocks or load.loaded_cols != cols or
        load.weight_beats != weight_len / pack.WIDE_BEAT_BYTES or
        load.act_beats != std.mem.bytesAsSlice(u64, act_bytes).len)
        return error.ResidentLoadMismatch;

    const reuse = try runKernelOp(a, &dut, rows, rows, rows, blocks, cols, window.emin, 1, ACT_REUSE, epoch, weights_reuse, &.{}, false);
    defer a.free(reuse.result);
    if (!std.mem.eql(u8, reuse.result, std.mem.sliceAsBytes(expected_reuse)) or
        reuse.weight_beats != weight_len / pack.WIDE_BEAT_BYTES or reuse.act_beats != 0 or
        reuse.activation_error or !reuse.activation_valid or reuse.loaded_epoch != epoch or
        reuse.loaded_blocks != blocks or reuse.loaded_cols != cols)
        return error.ResidentReuseMismatch;

    const bad_epoch = try runKernelOp(a, &dut, rows, rows, rows, blocks, cols, window.emin, 1, ACT_REUSE, epoch + 1, weights_reuse, act_bytes, true);
    defer a.free(bad_epoch.result);
    const bad_shape = try runKernelOp(a, &dut, rows, rows, rows, blocks, cols - 1, window.emin, 1, ACT_REUSE, epoch, weights_reuse, act_bytes, true);
    defer a.free(bad_shape.result);
    if (!bad_epoch.activation_valid or bad_epoch.loaded_epoch != epoch or
        bad_epoch.loaded_blocks != blocks or bad_epoch.loaded_cols != cols or
        !bad_shape.activation_valid or
        bad_shape.loaded_epoch != epoch or bad_shape.loaded_blocks != blocks or
        bad_shape.loaded_cols != cols)
        return error.InvalidReuseCorruptedResidentState;

    std.debug.print("  resident activation load/reuse passed: epoch=0x{X:0>8}, cycles={d}/{d}, hashes=0x{X:0>16}/0x{X:0>16}, load A={d}, reuse A={d}, invalid requests consumed W/A=0/0\n", .{
        epoch,                                load.cycles,                           reuse.cycles,
        std.hash.Wyhash.hash(0, load.result), std.hash.Wyhash.hash(0, reuse.result), load.act_beats,
        reuse.act_beats,
    });
}

pub fn main() !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();

    const Case = struct { num_rb: usize, logical_rows: ?usize = null, program_rows: ?usize = null, blocks: usize, cols: usize, ternary: bool = false, note: []const u8 = "" };
    const cases = [_]Case{
        .{ .num_rb = 1, .blocks = 1, .cols = 1, .note = "decode, 1 rb" },
        .{ .num_rb = 1, .blocks = 2, .cols = 4, .note = "prefill C=4" },
        .{ .num_rb = 3, .blocks = 2, .cols = 3, .note = "multi-rb, odd cols" },
        .{ .num_rb = 2, .blocks = 4, .cols = 8, .note = "prefill C=8 (full bank)" },
        .{ .num_rb = 8, .blocks = 16, .cols = 1, .note = "decode, attn-ish" },
        .{ .num_rb = 4, .blocks = 8, .cols = 8, .note = "prefill, larger" },
        .{ .num_rb = 1, .blocks = 2, .cols = 3, .ternary = true, .note = "ternary decode + prefill" },
        .{ .num_rb = 1, .program_rows = 0, .blocks = 2, .cols = 1, .note = "legacy zero NUM_ROWS emits full rowblock" },
        .{ .num_rb = 1, .logical_rows = 1, .blocks = 1, .cols = 1, .note = "one logical row, partial TKEEP" },
        .{ .num_rb = 1, .logical_rows = 6, .blocks = 2, .cols = 1, .note = "even partial row, full final TKEEP" },
        .{ .num_rb = 2, .logical_rows = 21, .blocks = 2, .cols = 3, .note = "odd partial final rowblock, C=3" },
        .{ .num_rb = 2, .logical_rows = 21, .blocks = 2, .cols = 1, .ternary = true, .note = "ternary odd partial final rowblock" },
    };
    for (cases, 0..) |cs, i| try runCase(a, cs.num_rb, cs.logical_rows, cs.program_rows, cs.blocks, cs.cols, cs.ternary, 0x3000 + i, cs.note);
    try runResidentReuseCase(a);
    std.debug.print("all gemm_kernel cosim cases passed (gemm_kernel === windowedFixedOutput, bit-exact)\n", .{});
}
