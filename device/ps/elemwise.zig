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

pub fn f32ToF16Bytes(src: []const u8, dst: []u8) ElemwiseError!void {
    const count = try f32Count(src);
    if (dst.len != count * @sizeOf(f16)) return error.InvalidLength;
    for (0..count) |i| {
        const value: f16 = @floatCast(readF32(src, i));
        std.mem.writeInt(u16, dst[i * @sizeOf(f16) ..][0..2], @bitCast(value), .little);
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

pub fn add2dBytes(lhs: []const u8, rhs: []const u8, dst: []u8, rows: u32, cols: u32, rhs_row_broadcast: bool) ElemwiseError!void {
    try binary2dBytes(.add, lhs, rhs, dst, rows, cols, rhs_row_broadcast);
}

pub fn mul2dBytes(lhs: []const u8, rhs: []const u8, dst: []u8, rows: u32, cols: u32, rhs_row_broadcast: bool) ElemwiseError!void {
    try binary2dBytes(.mul, lhs, rhs, dst, rows, cols, rhs_row_broadcast);
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

const BinaryOp = enum { add, mul };

fn binary2dBytes(
    comptime op: BinaryOp,
    lhs: []const u8,
    rhs: []const u8,
    dst: []u8,
    rows_raw: u32,
    cols_raw: u32,
    rhs_row_broadcast: bool,
) ElemwiseError!void {
    if (rows_raw == 0 or cols_raw == 0) return error.InvalidLength;
    const rows: usize = @intCast(rows_raw);
    const cols: usize = @intCast(cols_raw);
    const elements = std.math.mul(usize, rows, cols) catch return error.InvalidLength;
    const total_bytes = std.math.mul(usize, elements, @sizeOf(f32)) catch return error.InvalidLength;
    const rhs_elements = if (rhs_row_broadcast) rows else elements;
    const rhs_bytes = std.math.mul(usize, rhs_elements, @sizeOf(f32)) catch return error.InvalidLength;
    if (lhs.len != total_bytes or rhs.len != rhs_bytes or dst.len != total_bytes) return error.InvalidLength;

    for (0..cols) |col| {
        const base = col * rows;
        for (0..rows) |row| {
            const index = base + row;
            const rhs_index = if (rhs_row_broadcast) row else index;
            const a = readF32(lhs, index);
            const b = readF32(rhs, rhs_index);
            const value = switch (op) {
                .add => a + b,
                .mul => a * b,
            };
            writeF32(dst, index, value);
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

test "elemwise converts little-endian f32 bytes to f16 bytes" {
    var src: [8]u8 = undefined;
    var dst: [4]u8 = undefined;
    writeF32(&src, 0, 1.5);
    writeF32(&src, 1, -2);

    try f32ToF16Bytes(&src, &dst);

    const a: f16 = @bitCast(std.mem.readInt(u16, dst[0..2], .little));
    const b: f16 = @bitCast(std.mem.readInt(u16, dst[2..4], .little));
    try expectApprox(1.5, @floatCast(a), 0);
    try expectApprox(-2, @floatCast(b), 0);
}

test "elemwise 2d byte wrappers support rhs row broadcast" {
    var lhs: [6 * @sizeOf(f32)]u8 = undefined;
    var rhs: [3 * @sizeOf(f32)]u8 = undefined;
    var dst: [6 * @sizeOf(f32)]u8 = undefined;
    for (&[_]f32{ 1, 2, 3, 10, 20, 30 }, 0..) |value, i| {
        writeF32(&lhs, i, value);
    }
    for (&[_]f32{ 100, 200, 300 }, 0..) |value, i| {
        writeF32(&rhs, i, value);
    }

    try add2dBytes(&lhs, &rhs, &dst, 3, 2, true);

    try expectApprox(101, readF32(&dst, 0), 0.000001);
    try expectApprox(202, readF32(&dst, 1), 0.000001);
    try expectApprox(303, readF32(&dst, 2), 0.000001);
    try expectApprox(110, readF32(&dst, 3), 0.000001);
    try expectApprox(220, readF32(&dst, 4), 0.000001);
    try expectApprox(330, readF32(&dst, 5), 0.000001);
}
