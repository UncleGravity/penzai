const std = @import("std");

pub const RmsNormError = error{
    InvalidLength,
    InvalidEpsilon,
};

pub fn runF32(input: []const f32, weight: []const f32, dst: []f32, eps: f32) RmsNormError!void {
    if (input.len == 0 or weight.len != input.len or dst.len != input.len) return error.InvalidLength;
    if (eps < 0 or !std.math.isFinite(eps)) return error.InvalidEpsilon;

    var sum_sq: f64 = 0;
    for (input) |x| {
        const wide: f64 = x;
        sum_sq += wide * wide;
    }

    const mean_sq: f32 = @floatCast(sum_sq / @as(f64, @floatFromInt(input.len)));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
    for (input, weight, dst) |x, w, *out| {
        out.* = x * inv_rms * w;
    }
}

pub fn runBytes(input: []const u8, weight: []const u8, dst: []u8, eps: f32) RmsNormError!void {
    const count = try f32Count(input);
    if (weight.len != input.len or dst.len != input.len) return error.InvalidLength;
    if (eps < 0 or !std.math.isFinite(eps)) return error.InvalidEpsilon;

    var sum_sq: f64 = 0;
    for (0..count) |i| {
        const wide: f64 = readF32(input, i);
        sum_sq += wide * wide;
    }

    const mean_sq: f32 = @floatCast(sum_sq / @as(f64, @floatFromInt(count)));
    const inv_rms = 1.0 / @sqrt(mean_sq + eps);
    for (0..count) |i| {
        writeF32(dst, i, readF32(input, i) * inv_rms * readF32(weight, i));
    }
}

fn f32Count(bytes: []const u8) RmsNormError!usize {
    if (bytes.len == 0 or bytes.len % @sizeOf(f32) != 0) return error.InvalidLength;
    return bytes.len / @sizeOf(f32);
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
    const weight = [_]f32{ 1, 1 };
    var dst: [2]f32 = undefined;

    try runF32(&input, &weight, &dst, 0);

    try expectApprox(0.84852815, dst[0], 0.000001);
    try expectApprox(1.1313709, dst[1], 0.000001);
}

test "rmsnorm applies per-channel weights" {
    const input = [_]f32{ 1, -1, 1, -1 };
    const weight = [_]f32{ 1, 2, 3, 4 };
    var dst: [4]f32 = undefined;

    try runF32(&input, &weight, &dst, 0);

    try expectApprox(1, dst[0], 0.000001);
    try expectApprox(-2, dst[1], 0.000001);
    try expectApprox(3, dst[2], 0.000001);
    try expectApprox(-4, dst[3], 0.000001);
}

test "rmsnorm byte wrapper uses little-endian f32 values" {
    var input: [8]u8 = undefined;
    var weight: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&input, 0, 3);
    writeF32(&input, 1, 4);
    writeF32(&weight, 0, 1);
    writeF32(&weight, 1, 1);

    try runBytes(&input, &weight, &dst, 0);

    try expectApprox(0.84852815, readF32(&dst, 0), 0.000001);
    try expectApprox(1.1313709, readF32(&dst, 1), 0.000001);
}
