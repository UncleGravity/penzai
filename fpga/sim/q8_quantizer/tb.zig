//! Bit-exact cosim for the production FP32 -> Q8_0 block quantizer.
//!
//! The oracle is shared/layout.zig itself. Directed rounding cases, randomized
//! normal-range blocks, input bubbles, output backpressure, framing diagnostics,
//! and the explicitly unsupported arithmetic range are all exercised.

const std = @import("std");
const layout = @import("shared_layout");
const c = @cImport(@cInclude("shim.h"));

const block = layout.q8_block;
const status_nonfinite: u8 = 1 << 0;
const status_scale: u8 = 1 << 1;
const status_frame: u8 = 1 << 2;
const status_arith: u8 = 1 << 3;

comptime {
    if (block != 32) @compileError("q8_quantizer RTL is fixed to Q8_0 blocks of 32");
}

const Dut = struct {
    handle: *c.Dut,

    fn init() Dut {
        return .{ .handle = c.dut_new().? };
    }

    fn deinit(self: *Dut) void {
        c.dut_free(self.handle);
    }

    fn step(self: *Dut) void {
        c.dut_set_clk(self.handle, 1);
        c.dut_eval(self.handle);
        c.dut_set_clk(self.handle, 0);
        c.dut_eval(self.handle);
    }

    fn reset(self: *Dut) void {
        c.dut_set_rst_n(self.handle, 0);
        c.dut_set_input(self.handle, 0, 0, 0);
        c.dut_set_out_ready(self.handle, 0);
        c.dut_set_clk(self.handle, 0);
        c.dut_eval(self.handle);
        for (0..4) |_| self.step();
        c.dut_set_rst_n(self.handle, 1);
        self.step();
    }
};

const Result = struct {
    quants: [block]i8,
    scale_bits: u16,
    status: u8,
    cycles: usize,
};

fn runBlock(
    dut: *Dut,
    values: *const [block]f32,
    bubble_seed: ?u64,
    malformed_last: ?usize,
    hold_output_cycles: usize,
) !Result {
    var prng = std.Random.DefaultPrng.init(bubble_seed orelse 0);
    const rnd = prng.random();
    var sent: usize = 0;
    var cycles: usize = 0;

    while (sent < block) : (cycles += 1) {
        if (cycles > 20_000) return error.InputTimeout;
        const ready = c.dut_in_ready(dut.handle) != 0;
        const bubble = bubble_seed != null and rnd.uintLessThan(u8, 4) == 0;
        const send = ready and !bubble;
        if (send) {
            const last_index = malformed_last orelse block - 1;
            c.dut_set_input(
                dut.handle,
                1,
                @bitCast(values[sent]),
                @intFromBool(sent == last_index),
            );
            sent += 1;
        } else {
            c.dut_set_input(dut.handle, 0, 0, 0);
        }
        dut.step();
    }
    c.dut_set_input(dut.handle, 0, 0, 0);

    while (c.dut_out_valid(dut.handle) == 0) : (cycles += 1) {
        if (cycles > 20_000) return error.OutputTimeout;
        dut.step();
    }

    var held_quants: [block]u8 = undefined;
    for (0..block) |i| held_quants[i] = c.dut_out_quant(dut.handle, @intCast(i));
    const held_scale = c.dut_out_scale(dut.handle);
    const held_status = c.dut_out_status(dut.handle);
    for (0..hold_output_cycles) |_| {
        c.dut_set_out_ready(dut.handle, 0);
        dut.step();
        if (c.dut_out_valid(dut.handle) == 0) return error.ValidDroppedUnderBackpressure;
        if (c.dut_out_scale(dut.handle) != held_scale or c.dut_out_status(dut.handle) != held_status)
            return error.MetadataChangedUnderBackpressure;
        for (0..block) |i| {
            if (c.dut_out_quant(dut.handle, @intCast(i)) != held_quants[i])
                return error.QuantsChangedUnderBackpressure;
        }
    }

    var result: Result = .{
        .quants = undefined,
        .scale_bits = held_scale,
        .status = held_status,
        .cycles = cycles,
    };
    for (0..block) |i| result.quants[i] = @bitCast(held_quants[i]);

    c.dut_set_out_ready(dut.handle, 1);
    dut.step();
    c.dut_set_out_ready(dut.handle, 0);
    if (c.dut_out_valid(dut.handle) != 0) return error.ValidDidNotRetire;
    return result;
}

fn expectCanonical(
    dut: *Dut,
    values: *const [block]f32,
    expected_status: u8,
    seed: ?u64,
    hold_cycles: usize,
) !usize {
    var expected_quants: [block]i8 = undefined;
    var expected_scales: [1]f16 = undefined;
    try layout.quantizeQ8_0(values, &expected_quants, &expected_scales);
    const got = try runBlock(dut, values, seed, null, hold_cycles);

    if (got.status != expected_status) {
        std.debug.print("status mismatch: got=0x{x} expected=0x{x}\n", .{ got.status, expected_status });
        return error.StatusMismatch;
    }
    const expected_scale_bits: u16 = @bitCast(expected_scales[0]);
    if (got.scale_bits != expected_scale_bits) {
        std.debug.print("scale mismatch: got=0x{x:0>4} expected=0x{x:0>4}\n", .{ got.scale_bits, expected_scale_bits });
        return error.ScaleMismatch;
    }
    for (expected_quants, got.quants, 0..) |expected, actual, i| {
        if (actual != expected) {
            std.debug.print(
                "quant mismatch lane {d}: value={e} got={d} expected={d} scale=0x{x:0>4}\n",
                .{ i, values[i], actual, expected, got.scale_bits },
            );
            return error.QuantMismatch;
        }
    }
    return got.cycles;
}

fn normalValue(rnd: std.Random, min_exp: u8, max_exp: u8) f32 {
    const sign: u32 = @as(u32, rnd.int(u1)) << 31;
    const exponent: u32 = rnd.intRangeAtMost(u8, min_exp, max_exp);
    const fraction = rnd.int(u23);
    return @bitCast(sign | (exponent << 23) | fraction);
}

pub fn main() !void {
    var dut = Dut.init();
    defer dut.deinit();
    dut.reset();

    var cases: usize = 0;
    var max_cycles: usize = 0;

    var zero = [_]f32{0} ** block;
    max_cycles = @max(max_cycles, try expectCanonical(&dut, &zero, 0, null, 5));
    cases += 1;

    var ties = [_]f32{0} ** block;
    ties[0] = 127;
    ties[1] = 2.5;
    ties[2] = -2.5;
    ties[3] = 3.5;
    ties[4] = -3.5;
    max_cycles = @max(max_cycles, try expectCanonical(&dut, &ties, 0, 0x71e5, 3));
    cases += 1;

    // This is the canonical regression proving q uses the unrounded f32 scale,
    // even though the stored f16 scale is subnormal for this block.
    var unrounded = [_]f32{0} ** block;
    unrounded[0] = 0.001;
    unrounded[1] = 0.0000118094488;
    max_cycles = @max(max_cycles, try expectCanonical(&dut, &unrounded, 0, 0x51ca1e, 0));
    cases += 1;

    // A finite f32 block can have a scale above the f16 finite range. The
    // quantized bytes remain deterministic, but the resident GEMM scale would
    // be infinite, so the ingress path must reject the block.
    var scale_overflow = [_]f32{0} ** block;
    scale_overflow[0] = 1.0e8;
    max_cycles = @max(max_cycles, try expectCanonical(&dut, &scale_overflow, status_scale, null, 0));
    cases += 1;

    var broad = [_]f32{0} ** block;
    for (&broad, 0..) |*value, i| {
        const signed_i: i32 = @intCast(i);
        value.* = @as(f32, @floatFromInt(signed_i - 16)) * 17.125;
    }
    broad[0] = 65504;
    max_cycles = @max(max_cycles, try expectCanonical(&dut, &broad, 0, 0xb10c, 7));
    cases += 1;

    var random = std.Random.DefaultPrng.init(0x0a80_f32f_16ee_0001);
    const rnd = random.random();
    for (0..1024) |case_index| {
        var values: [block]f32 = undefined;
        // At least one value keeps scale in the current GEMM's normal-f16 range;
        // the other lanes sweep a much wider normal-f32 exponent range.
        values[0] = normalValue(rnd, 120, 140);
        for (values[1..], 1..) |*value, i| {
            value.* = if ((case_index + i) % 19 == 0) 0 else normalValue(rnd, 80, 140);
        }
        const seed = rnd.int(u64);
        max_cycles = @max(max_cycles, try expectCanonical(&dut, &values, 0, seed, case_index % 4));
        cases += 1;
    }

    // Non-finite inputs do not participate in absmax. The canonical Zig routine
    // currently has no defined int conversion for a mixed NaN/Inf block, so the
    // leaf deterministically emits q=0 for those lanes and advertises the event.
    var nonfinite = ties;
    nonfinite[5] = std.math.nan(f32);
    nonfinite[6] = std.math.inf(f32);
    var sanitized = nonfinite;
    sanitized[5] = 0;
    sanitized[6] = 0;
    var expected_q: [block]i8 = undefined;
    var expected_scale: [1]f16 = undefined;
    try layout.quantizeQ8_0(&sanitized, &expected_q, &expected_scale);
    const nonfinite_got = try runBlock(&dut, &nonfinite, 0x1f1f, null, 2);
    if (nonfinite_got.status != status_nonfinite) return error.NonfiniteStatusMissing;
    if (nonfinite_got.scale_bits != @as(u16, @bitCast(expected_scale[0]))) return error.ScaleMismatch;
    try std.testing.expectEqualSlices(i8, &expected_q, &nonfinite_got.quants);
    cases += 1;

    // TLAST is diagnostic, not a source of block length. An early marker must be
    // sticky while the fixed 32-value transaction still completes normally.
    const malformed = try runBlock(&dut, &ties, null, 9, 0);
    if (malformed.status != status_frame) return error.FrameStatusMissing;
    cases += 1;

    // A normal f32 amax whose /127 result is subnormal cannot be represented by
    // the exact-normal divider. The leaf must fail closed instead of fabricating
    // a resident activation record.
    var tiny = [_]f32{0} ** block;
    tiny[0] = @bitCast(@as(u32, 0x0080_0000)); // minimum positive normal f32
    const tiny_got = try runBlock(&dut, &tiny, null, null, 0);
    if (tiny_got.status & status_arith == 0) return error.ArithmeticStatusMissing;
    cases += 1;

    std.debug.print(
        "\n  q8 quantizer cosim: {d} blocks, exact quants/scales, max {d} cycles/block\n" ++
            "  directed RNE, unrounded reciprocal, bubbles, backpressure, status: passed\n\n",
        .{ cases, max_cycles },
    );
}
