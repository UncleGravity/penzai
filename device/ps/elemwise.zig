const std = @import("std");

pub const ElemwiseError = error{
    InvalidLength,
};

// Element-wise f32 kernels vectorize over fixed-width lanes; portable @Vector
// lowers to NEON on the A53 board. Each lane is an independent IEEE add/mul/scale
// with no cross-lane reduction, so the vector path is bit-exact vs the scalar one.
const lanes = 16;
const Vf32 = @Vector(lanes, f32);

const BinaryOp = enum { add, mul };

/// dst[0..n] = lhs[0..n] (op) rhs[0..n], over little-endian f32 bytes (n elements
/// from the start of each slice). Tail past the last full lane group runs scalar.
fn binaryVecBytes(comptime op: BinaryOp, lhs_b: []const u8, rhs_b: []const u8, dst_b: []u8, n: usize) void {
    const lhs = std.mem.bytesAsSlice(f32, lhs_b);
    const rhs = std.mem.bytesAsSlice(f32, rhs_b);
    const dst = std.mem.bytesAsSlice(f32, dst_b);
    var i: usize = 0;
    while (i + lanes <= n) : (i += lanes) {
        const a: Vf32 = lhs[i..][0..lanes].*;
        const b: Vf32 = rhs[i..][0..lanes].*;
        dst[i..][0..lanes].* = switch (op) {
            .add => a + b,
            .mul => a * b,
        };
    }
    while (i < n) : (i += 1) {
        dst[i] = switch (op) {
            .add => lhs[i] + rhs[i],
            .mul => lhs[i] * rhs[i],
        };
    }
}

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
    const in = std.mem.bytesAsSlice(f32, src);
    const out = std.mem.bytesAsSlice(f16, dst);
    var i: usize = 0;
    while (i + lanes <= count) : (i += lanes) {
        const fv: Vf32 = in[i..][0..lanes].*;
        const hv: @Vector(lanes, f16) = @floatCast(fv);
        out[i..][0..lanes].* = hv;
    }
    while (i < count) : (i += 1) out[i] = @floatCast(in[i]);
}

pub fn addBytes(lhs: []const u8, rhs: []const u8, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    binaryVecBytes(.add, lhs, rhs, dst, count);
}

pub fn mulBytes(lhs: []const u8, rhs: []const u8, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    binaryVecBytes(.mul, lhs, rhs, dst, count);
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
    const in = std.mem.bytesAsSlice(f32, input);
    const out = std.mem.bytesAsSlice(f32, dst);
    const sv: Vf32 = @splat(scale);
    var i: usize = 0;
    while (i + lanes <= count) : (i += lanes) {
        const v: Vf32 = in[i..][0..lanes].*;
        out[i..][0..lanes].* = v * sv;
    }
    while (i < count) : (i += 1) out[i] = in[i] * scale;
}

pub fn addScaledBytes(lhs: []const u8, rhs: []const u8, rhs_scale: f32, dst: []u8) ElemwiseError!void {
    const count = try f32Count(lhs);
    if (rhs.len != lhs.len or dst.len != lhs.len) return error.InvalidLength;
    const l = std.mem.bytesAsSlice(f32, lhs);
    const r = std.mem.bytesAsSlice(f32, rhs);
    const out = std.mem.bytesAsSlice(f32, dst);
    const sv: Vf32 = @splat(rhs_scale);
    var i: usize = 0;
    while (i + lanes <= count) : (i += lanes) {
        const a: Vf32 = l[i..][0..lanes].*;
        const b: Vf32 = r[i..][0..lanes].*;
        out[i..][0..lanes].* = a + b * sv;
    }
    while (i < count) : (i += 1) out[i] = l[i] + r[i] * rhs_scale;
}

fn f32Count(bytes: []const u8) ElemwiseError!usize {
    if (bytes.len % @sizeOf(f32) != 0) return error.InvalidLength;
    return bytes.len / @sizeOf(f32);
}

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

    if (!rhs_row_broadcast) {
        // Same-shape: one flat vectorized pass.
        binaryVecBytes(op, lhs, rhs, dst, elements);
        return;
    }
    // Row-broadcast: rhs (length `rows`) repeats across columns, so each column is
    // an independent element-wise op of `rows` elements against the same rhs.
    const row_bytes = rows * @sizeOf(f32);
    for (0..cols) |col| {
        const off = col * row_bytes;
        binaryVecBytes(op, lhs[off..], rhs, dst[off..], rows);
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

test "elemwise vector path is bit-exact vs scalar over a 35-wide span (vector + remainder)" {
    const n = 35; // > lanes (16), with a non-multiple remainder
    var lhs: [n * @sizeOf(f32)]u8 = undefined;
    var rhs: [n * @sizeOf(f32)]u8 = undefined;
    var dst: [n * @sizeOf(f32)]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0xE12C0DE);
    const rnd = prng.random();
    for (0..n) |i| {
        writeF32(&lhs, i, rnd.float(f32) * 2 - 1);
        writeF32(&rhs, i, rnd.float(f32) * 2 - 1);
    }

    try addBytes(&lhs, &rhs, &dst);
    for (0..n) |i| try std.testing.expectEqual(readF32(&lhs, i) + readF32(&rhs, i), readF32(&dst, i));

    try mulBytes(&lhs, &rhs, &dst);
    for (0..n) |i| try std.testing.expectEqual(readF32(&lhs, i) * readF32(&rhs, i), readF32(&dst, i));

    try scaleBytes(&lhs, 0.375, &dst);
    for (0..n) |i| try std.testing.expectEqual(readF32(&lhs, i) * 0.375, readF32(&dst, i));

    try addScaledBytes(&lhs, &rhs, 0.25, &dst);
    for (0..n) |i| try std.testing.expectEqual(readF32(&lhs, i) + readF32(&rhs, i) * 0.25, readF32(&dst, i));

    // Same-shape 2d (rows*cols = 35) uses the flat vector pass.
    try add2dBytes(&lhs, &rhs, &dst, 7, 5, false);
    for (0..n) |i| try std.testing.expectEqual(readF32(&lhs, i) + readF32(&rhs, i), readF32(&dst, i));

    var half: [n * @sizeOf(f16)]u8 = undefined;
    try f32ToF16Bytes(&lhs, &half);
    for (0..n) |i| {
        const got: f16 = @bitCast(std.mem.readInt(u16, half[i * @sizeOf(f16) ..][0..2], .little));
        try std.testing.expectEqual(@as(f16, @floatCast(readF32(&lhs, i))), got);
    }
}

test "elemwise row-broadcast vector path is bit-exact over wide rows" {
    const rows = 20; // > lanes, exercises the per-column vector pass
    const cols = 3;
    var lhs: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var rhs: [rows * @sizeOf(f32)]u8 = undefined;
    var dst: [rows * cols * @sizeOf(f32)]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(0xB0AD);
    const rnd = prng.random();
    for (0..rows * cols) |i| writeF32(&lhs, i, rnd.float(f32) * 2 - 1);
    for (0..rows) |i| writeF32(&rhs, i, rnd.float(f32) * 2 - 1);

    try mul2dBytes(&lhs, &rhs, &dst, rows, cols, true);
    for (0..cols) |col| {
        for (0..rows) |row| {
            const idx = col * rows + row;
            try std.testing.expectEqual(readF32(&lhs, idx) * readF32(&rhs, row), readF32(&dst, idx));
        }
    }
}
