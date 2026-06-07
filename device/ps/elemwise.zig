const std = @import("std");

pub const ElemwiseError = error{
    InvalidLength,
};

pub fn copyF32(src: []const f32, dst: []f32) ElemwiseError!void {
    if (dst.len != src.len) return error.InvalidLength;
    for (src, dst) |x, *out| {
        out.* = x;
    }
}

pub fn addF32(lhs: []const f32, rhs: []const f32, dst: []f32) ElemwiseError!void {
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (lhs, rhs, dst) |a, b, *out| {
        out.* = a + b;
    }
}

pub fn mulF32(lhs: []const f32, rhs: []const f32, dst: []f32) ElemwiseError!void {
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (lhs, rhs, dst) |a, b, *out| {
        out.* = a * b;
    }
}

pub fn scaleF32(input: []const f32, scale: f32, dst: []f32) ElemwiseError!void {
    if (dst.len != input.len) return error.InvalidLength;
    for (input, dst) |x, *out| {
        out.* = x * scale;
    }
}

pub fn addScaledF32(lhs: []const f32, rhs: []const f32, rhs_scale: f32, dst: []f32) ElemwiseError!void {
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (lhs, rhs, dst) |a, b, *out| {
        out.* = a + b * rhs_scale;
    }
}

pub fn copyBytes(src: []const u8, dst: []u8) ElemwiseError!void {
    if (dst.len != src.len) return error.InvalidLength;
    for (src, dst) |x, *out| {
        out.* = x;
    }
}

pub fn addBytes(lhs: []const u8, rhs: []const u8, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, readF32(lhs, i) + readF32(rhs, i));
    }
}

pub fn mulBytes(lhs: []const u8, rhs: []const u8, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, readF32(lhs, i) * readF32(rhs, i));
    }
}

pub fn scaleBytes(input: []const u8, scale: f32, dst: []u8) ElemwiseError!void {
    const count = try f32Count(input);
    if (dst.len != input.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, readF32(input, i) * scale);
    }
}

pub fn addScaledBytes(lhs: []const u8, rhs: []const u8, rhs_scale: f32, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    for (0..count) |i| {
        writeF32(dst, i, readF32(lhs, i) + readF32(rhs, i) * rhs_scale);
    }
}

fn f32Count(bytes: []const u8) ElemwiseError!usize {
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

test "elemwise add, multiply, and scale f32 slices" {
    const lhs = [_]f32{ 1, 2, 3 };
    const rhs = [_]f32{ 10, 20, 30 };
    var dst: [3]f32 = undefined;

    try addF32(&lhs, &rhs, &dst);
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33 }, &dst);

    try mulF32(&lhs, &rhs, &dst);
    try std.testing.expectEqualSlices(f32, &.{ 10, 40, 90 }, &dst);

    try scaleF32(&lhs, 0.5, &dst);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1, 1.5 }, &dst);
}

test "elemwise add scaled supports residual-style updates" {
    const lhs = [_]f32{ 1, 2, 3 };
    const rhs = [_]f32{ 10, 20, 30 };
    var dst: [3]f32 = undefined;

    try addScaledF32(&lhs, &rhs, 0.1, &dst);

    try expectApprox(2, dst[0], 0.000001);
    try expectApprox(4, dst[1], 0.000001);
    try expectApprox(6, dst[2], 0.000001);
}

test "elemwise byte wrappers use little-endian f32 values" {
    var lhs: [8]u8 = undefined;
    var rhs: [8]u8 = undefined;
    var dst: [8]u8 = undefined;
    writeF32(&lhs, 0, 1);
    writeF32(&lhs, 1, 2);
    writeF32(&rhs, 0, 10);
    writeF32(&rhs, 1, 20);

    try addBytes(&lhs, &rhs, &dst);

    try expectApprox(11, readF32(&dst, 0), 0.000001);
    try expectApprox(22, readF32(&dst, 1), 0.000001);
}
