const std = @import("std");

pub const RmsNormError = error{
    InvalidLength,
    InvalidEpsilon,
};

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
    if (rows_raw == 0 or cols_raw == 0) return error.InvalidLength;
    if (eps < 0 or !std.math.isFinite(eps)) return error.InvalidEpsilon;
    const rows: usize = @intCast(rows_raw);
    const cols: usize = @intCast(cols_raw);
    const elements = std.math.mul(usize, rows, cols) catch return error.InvalidLength;
    const total_bytes = std.math.mul(usize, elements, @sizeOf(f32)) catch return error.InvalidLength;
    if (input.len != total_bytes or dst.len != total_bytes) return error.InvalidLength;

    for (0..cols) |col| {
        const base = col * rows;
        var sum_sq: f64 = 0;
        for (0..rows) |row| {
            const x = readF32(input, base + row);
            const sq: f32 = x * x;
            sum_sq += sq;
        }

        const mean_sq: f32 = @floatCast(sum_sq / @as(f64, @floatFromInt(rows)));
        const inv_rms = 1.0 / @sqrt(mean_sq + eps);
        for (0..rows) |row| {
            const index = base + row;
            writeF32(dst, index, readF32(input, index) * inv_rms);
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
