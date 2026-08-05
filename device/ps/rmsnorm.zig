const std = @import("std");

pub const RmsNormError = error{
    InvalidLength,
    InvalidEpsilon,
};

// Vectorized over fixed-width lanes (portable @Vector → NEON on the A53). The
// sum-of-squares accumulates in f64 lanes to match the scalar path's precision;
// the reduction reorders the sum, so the result matches the scalar reference
// within the op's fp tolerance, not bit-exact. The final scale is per-element
// independent, so that part is bit-exact.
const lanes = 16;
const Vf32 = @Vector(lanes, f32);
const Vf64 = @Vector(lanes, f64);

pub fn runF32(input: []const f32, dst: []f32, rows_raw: u32, cols_raw: u32, eps: f32) RmsNormError!void {
    if (rows_raw == 0 or cols_raw == 0) return error.InvalidLength;
    if (eps < 0 or !std.math.isFinite(eps)) return error.InvalidEpsilon;
    const rows: usize = @intCast(rows_raw);
    const cols: usize = @intCast(cols_raw);
    const elements = std.math.mul(usize, rows, cols) catch return error.InvalidLength;
    if (input.len != elements or dst.len != elements) return error.InvalidLength;

    for (0..cols) |col| {
        const base = col * rows;
        var sum_sq: f64 = 0;
        for (0..rows) |row| {
            const x = input[base + row];
            const sq: f32 = x * x;
            sum_sq += sq;
        }

        const mean_sq: f32 = @floatCast(sum_sq / @as(f64, @floatFromInt(rows)));
        const inv_rms = 1.0 / @sqrt(mean_sq + eps);
        for (0..rows) |row| {
            const index = base + row;
            dst[index] = input[index] * inv_rms;
        }
    }
}

pub fn runBytes(input: []const u8, dst: []u8, rows_raw: u32, cols_raw: u32, eps: f32) RmsNormError!void {
    try runBytesImpl(false, input, &.{}, dst, rows_raw, cols_raw, eps);
}

pub fn runWeightedBytes(input: []const u8, weight: []const u8, dst: []u8, rows_raw: u32, cols_raw: u32, eps: f32) RmsNormError!void {
    try runBytesImpl(true, input, weight, dst, rows_raw, cols_raw, eps);
}

fn runBytesImpl(
    comptime has_weight: bool,
    input: []const u8,
    weight: []const u8,
    dst: []u8,
    rows_raw: u32,
    cols_raw: u32,
    eps: f32,
) RmsNormError!void {
    if (rows_raw == 0 or cols_raw == 0) return error.InvalidLength;
    if (eps < 0 or !std.math.isFinite(eps)) return error.InvalidEpsilon;
    const rows: usize = @intCast(rows_raw);
    const cols: usize = @intCast(cols_raw);
    const elements = std.math.mul(usize, rows, cols) catch return error.InvalidLength;
    const total_bytes = std.math.mul(usize, elements, @sizeOf(f32)) catch return error.InvalidLength;
    const weight_bytes = std.math.mul(usize, rows, @sizeOf(f32)) catch return error.InvalidLength;
    if (input.len != total_bytes or dst.len != total_bytes) return error.InvalidLength;
    if (has_weight) {
        if (weight.len != weight_bytes) return error.InvalidLength;
    } else if (weight.len != 0) return error.InvalidLength;

    const in = std.mem.bytesAsSlice(f32, input);
    const gamma = if (has_weight) std.mem.bytesAsSlice(f32, weight) else &.{};
    const out = std.mem.bytesAsSlice(f32, dst);
    for (0..cols) |col| {
        const x = in[col * rows ..][0..rows];
        const y = out[col * rows ..][0..rows];

        // sum of squares, f64 lanes for precision
        var accv: Vf64 = @splat(0);
        var i: usize = 0;
        while (i + lanes <= rows) : (i += lanes) {
            const xf: Vf32 = x[i..][0..lanes].*;
            const xd: Vf64 = @floatCast(xf);
            accv += xd * xd;
        }
        var sum_sq: f64 = @reduce(.Add, accv);
        while (i < rows) : (i += 1) {
            const xd: f64 = x[i];
            sum_sq += xd * xd;
        }

        const mean_sq: f32 = @floatCast(sum_sq / @as(f64, @floatFromInt(rows)));
        const inv_rms = 1.0 / @sqrt(mean_sq + eps);

        // scale (per-element, bit-exact)
        const rv: Vf32 = @splat(inv_rms);
        i = 0;
        while (i + lanes <= rows) : (i += lanes) {
            const v: Vf32 = x[i..][0..lanes].*;
            const normalized: Vf32 = v * rv;
            y[i..][0..lanes].* = if (has_weight)
                normalized * @as(Vf32, gamma[i..][0..lanes].*)
            else
                normalized;
        }
        while (i < rows) : (i += 1) {
            const normalized: f32 = x[i] * inv_rms;
            y[i] = if (has_weight) normalized * gamma[i] else normalized;
        }
    }
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

test "rmsnorm normalizes by root mean square" {
    const input = [_]f32{ 3, 4 };
    var dst: [2]f32 = undefined;

    try runF32(&input, &dst, 2, 1, 0);

    try expectApprox(0.84852815, dst[0], 0.000001);
    try expectApprox(1.1313709, dst[1], 0.000001);
}

test "rmsnorm normalizes each contiguous vector independently" {
    const input = [_]f32{ 3, 4, 1, 1 };
    var dst: [4]f32 = undefined;

    try runF32(&input, &dst, 2, 2, 0);

    try expectApprox(0.84852815, dst[0], 0.000001);
    try expectApprox(1.1313709, dst[1], 0.000001);
    try expectApprox(1, dst[2], 0.000001);
    try expectApprox(1, dst[3], 0.000001);
}

test "rmsnorm byte wrapper uses little-endian f32 values" {
    var input: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&input, 0, 3);
    writeF32(&input, 1, 4);

    try runBytes(&input, &dst, 2, 1, 0);

    try expectApprox(0.84852815, readF32(&dst, 0), 0.000001);
    try expectApprox(1.1313709, readF32(&dst, 1), 0.000001);
}

test "rmsnorm vector byte path matches scalar reference over wide rows" {
    const rows = 40; // > lanes (16), with remainder
    const cols = 2;
    var in_f: [rows * cols]f32 = undefined;
    var in_b: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var ref: [rows * cols]f32 = undefined;
    var out_b: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0x12345);
    const rnd = prng.random();
    for (0..rows * cols) |i| {
        const v = rnd.float(f32) * 2 - 1;
        in_f[i] = v;
        writeF32(&in_b, i, v);
    }

    try runF32(&in_f, &ref, rows, cols, 1e-5);
    try runBytes(&in_b, &out_b, rows, cols, 1e-5);

    for (0..rows * cols) |i| try expectApprox(ref[i], readF32(&out_b, i), 1e-4);
}

test "fused rmsnorm weight matches standalone then multiply across columns and remainder" {
    const rows = 35;
    const cols = 3;
    var input: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var weight: [rows * @sizeOf(f32)]u8 = undefined;
    var normalized: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var fused: [rows * cols * @sizeOf(f32)]u8 = undefined;

    for (0..rows) |row| {
        writeF32(&weight, row, 0.25 + @as(f32, @floatFromInt(row)) * 0.03125);
    }
    for (0..cols) |col| {
        for (0..rows) |row| {
            const value = @as(f32, @floatFromInt(1 + col * rows + row)) / 17.0 - 2.0;
            writeF32(&input, col * rows + row, value);
        }
    }

    try runBytes(&input, &normalized, rows, cols, 1e-4);
    try runWeightedBytes(&input, &weight, &fused, rows, cols, 1e-4);

    for (0..cols) |col| {
        for (0..rows) |row| {
            const index = col * rows + row;
            const expected = readF32(&normalized, index) * readF32(&weight, row);
            try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(readF32(&fused, index))));
        }
    }
}

test "rmsnorm byte paths validate shape lengths and epsilon" {
    var input: [8]u8 = undefined;
    var weight: [8]u8 = undefined;
    var dst: [8]u8 = undefined;

    try std.testing.expectError(error.InvalidLength, runBytes(&input, &dst, 0, 1, 0));
    try std.testing.expectError(error.InvalidLength, runBytes(&input, &dst, 2, 0, 0));
    try std.testing.expectError(error.InvalidLength, runBytes(input[0..4], &dst, 2, 1, 0));
    try std.testing.expectError(error.InvalidLength, runBytes(&input, dst[0..4], 2, 1, 0));
    try std.testing.expectError(error.InvalidLength, runWeightedBytes(&input, weight[0..4], &dst, 2, 1, 0));
    try std.testing.expectError(error.InvalidLength, runWeightedBytes(&input, &weight, dst[0..4], 2, 1, 0));
    try std.testing.expectError(error.InvalidLength, runWeightedBytes(&.{}, &.{}, &.{}, std.math.maxInt(u32), std.math.maxInt(u32), 0));
    try std.testing.expectError(error.InvalidEpsilon, runBytes(&input, &dst, 2, 1, -0.001));
    try std.testing.expectError(error.InvalidEpsilon, runBytes(&input, &dst, 2, 1, std.math.nan(f32)));
    try std.testing.expectError(error.InvalidEpsilon, runWeightedBytes(&input, &weight, &dst, 2, 1, std.math.inf(f32)));
}
