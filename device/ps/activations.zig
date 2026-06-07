const std = @import("std");

pub const ActivationError = error{
    InvalidLength,
};

pub fn siluScalar(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn siluF32(input: []const f32, dst: []f32) ActivationError!void {
    if (dst.len != input.len) return error.InvalidLength;
    for (input, dst) |x, *out| {
        out.* = siluScalar(x);
    }
}

pub fn swigluF32(gate: []const f32, up: []const f32, dst: []f32) ActivationError!void {
    if (up.len != gate.len or dst.len != gate.len) return error.InvalidLength;
    for (gate, up, dst) |g, u, *out| {
        out.* = siluScalar(g) * u;
    }
}

pub fn siluBytes(input: []const u8, dst: []u8) ActivationError!void {
    const count = try f32Count(input);
    if (dst.len != input.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, siluScalar(readF32(input, i)));
    }
}

pub fn swigluBytes(gate: []const u8, up: []const u8, dst: []u8) ActivationError!void {
    const count = try f32Count(gate);
    if (up.len != gate.len or dst.len != gate.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, siluScalar(readF32(gate, i)) * readF32(up, i));
    }
}

fn f32Count(bytes: []const u8) ActivationError!usize {
    if (bytes.len % @sizeOf(f32) != 0) return error.InvalidLength;
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

test "silu applies sigmoid-weighted linear activation" {
    const input = [_]f32{ 0, 1, -1 };
    var dst: [3]f32 = undefined;

    try siluF32(&input, &dst);

    try expectApprox(0, dst[0], 0.000001);
    try expectApprox(0.7310586, dst[1], 0.000001);
    try expectApprox(-0.26894143, dst[2], 0.000001);
}

test "swiglu multiplies up projection by silu gate" {
    const gate = [_]f32{ 0, 1 };
    const up = [_]f32{ 10, 2 };
    var dst: [2]f32 = undefined;

    try swigluF32(&gate, &up, &dst);

    try expectApprox(0, dst[0], 0.000001);
    try expectApprox(1.4621172, dst[1], 0.000001);
}

test "activation byte wrappers use little-endian f32 values" {
    var gate: [8]u8 = undefined;
    var up: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&gate, 0, 0);
    writeF32(&gate, 1, 1);
    writeF32(&up, 0, 10);
    writeF32(&up, 1, 2);

    try swigluBytes(&gate, &up, &dst);

    try expectApprox(0, readF32(&dst, 0), 0.000001);
    try expectApprox(1.4621172, readF32(&dst, 1), 0.000001);
}
