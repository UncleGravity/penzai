const std = @import("std");

pub const RopeError = error{
    InvalidLength,
    InvalidTheta,
};

pub fn applyF32(input: []const f32, dst: []f32, position: u32, theta: f32) RopeError!void {
    if (input.len == 0 or input.len % 2 != 0 or dst.len != input.len) return error.InvalidLength;
    if (theta <= 0 or !std.math.isFinite(theta)) return error.InvalidTheta;

    const dim: f32 = @floatFromInt(input.len);
    const pos: f32 = @floatFromInt(position);
    const log_theta = @log(theta);

    var i: usize = 0;
    while (i < input.len) : (i += 2) {
        const x0 = input[i];
        const x1 = input[i + 1];
        const exponent = -@as(f32, @floatFromInt(i)) / dim;
        const angle = pos * @exp(log_theta * exponent);
        const c = @cos(angle);
        const s = @sin(angle);
        dst[i] = x0 * c - x1 * s;
        dst[i + 1] = x0 * s + x1 * c;
    }
}

pub fn applyInPlaceF32(values: []f32, position: u32, theta: f32) RopeError!void {
    try applyF32(values, values, position, theta);
}

pub fn applyBytes(input: []const u8, dst: []u8, position: u32, theta: f32) RopeError!void {
    const count = try f32Count(input);
    if (count % 2 != 0 or dst.len != input.len) return error.InvalidLength;
    if (theta <= 0 or !std.math.isFinite(theta)) return error.InvalidTheta;

    const dim: f32 = @floatFromInt(count);
    const pos: f32 = @floatFromInt(position);
    const log_theta = @log(theta);

    var i: usize = 0;
    while (i < count) : (i += 2) {
        const x0 = readF32(input, i);
        const x1 = readF32(input, i + 1);
        const exponent = -@as(f32, @floatFromInt(i)) / dim;
        const angle = pos * @exp(log_theta * exponent);
        const c = @cos(angle);
        const s = @sin(angle);
        writeF32(dst, i, x0 * c - x1 * s);
        writeF32(dst, i + 1, x0 * s + x1 * c);
    }
}

fn f32Count(bytes: []const u8) RopeError!usize {
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

test "rope leaves values unchanged at position zero" {
    const input = [_]f32{ 1, 2, 3, 4 };
    var dst: [4]f32 = undefined;

    try applyF32(&input, &dst, 0, 10000);

    for (input, dst) |expected, actual| {
        try expectApprox(expected, actual, 0.000001);
    }
}

test "rope rotates first pair by position radians" {
    const input = [_]f32{ 1, 0 };
    var dst: [2]f32 = undefined;

    try applyF32(&input, &dst, 1, 10000);

    try expectApprox(@cos(@as(f32, 1)), dst[0], 0.000001);
    try expectApprox(@sin(@as(f32, 1)), dst[1], 0.000001);
}

test "rope byte wrapper supports in-place rotation" {
    var bytes: [8]u8 = undefined;
    writeF32(&bytes, 0, 1);
    writeF32(&bytes, 1, 0);

    try applyBytes(&bytes, &bytes, 1, 10000);

    try expectApprox(@cos(@as(f32, 1)), readF32(&bytes, 0), 0.000001);
    try expectApprox(@sin(@as(f32, 1)), readF32(&bytes, 1), 0.000001);
}
