const std = @import("std");
const wire = @import("wire");

pub const RowsError = error{
    InvalidShape,
    InvalidStride,
    NegativeIndex,
    OutOfBounds,
};

pub fn setRowsF32ToF16Bytes(command: wire.SetRows, src: []const u8, indices: []const u8, dst: []u8) RowsError!void {
    const head_dim: usize = try positive(command.head_dim);
    const ne01: usize = try positive(command.ne01);
    const ne02: usize = try positive(command.ne02);
    const ne03: usize = try positive(command.ne03);
    const ne11: usize = try positive(command.ne11);
    const ne12: usize = try positive(command.ne12);
    const index_size = indexSize(command.index_type);
    const src_row_bytes = try checkedMul(head_dim, @sizeOf(f32));
    const dst_row_bytes = try checkedMul(head_dim, @sizeOf(f16));

    try validateStride(command.src_nb1, src_row_bytes);
    try validateStride(command.indices_nb1, try checkedMul(ne01, index_size));
    try validateStride(command.dst_nb1, dst_row_bytes);

    for (0..ne03) |d3| {
        const idx_dim12 = d3 % ne12;
        for (0..ne02) |d2| {
            const idx_dim11 = d2 % ne11;
            for (0..ne01) |i| {
                const index_offset = try checkedAdd(
                    try checkedMul(i, index_size),
                    try checkedAdd(
                        try checkedMul(idx_dim11, command.indices_nb1),
                        try checkedMul(idx_dim12, command.indices_nb2),
                    ),
                );
                if (index_offset > indices.len or index_size > indices.len - index_offset) return error.OutOfBounds;

                const row = try readIndex(command.index_type, indices[index_offset..][0..index_size]);
                const src_offset = try checkedAdd(
                    try checkedMul(i, command.src_nb1),
                    try checkedAdd(
                        try checkedMul(d2, command.src_nb2),
                        try checkedMul(d3, command.src_nb3),
                    ),
                );
                const dst_offset = try checkedAdd(
                    try checkedMul(row, command.dst_nb1),
                    try checkedAdd(
                        try checkedMul(d2, command.dst_nb2),
                        try checkedMul(d3, command.dst_nb3),
                    ),
                );
                if (src_offset > src.len or src_row_bytes > src.len - src_offset) return error.OutOfBounds;
                if (dst_offset > dst.len or dst_row_bytes > dst.len - dst_offset) return error.OutOfBounds;
                copyRowF32ToF16(src[src_offset..][0..src_row_bytes], dst[dst_offset..][0..dst_row_bytes], head_dim);
            }
        }
    }
}

fn copyRowF32ToF16(src: []const u8, dst: []u8, head_dim: usize) void {
    for (0..head_dim) |d| {
        const value = readF32(src, d);
        writeF16(dst, d, @floatCast(value));
    }
}

fn readIndex(index_type: wire.IndexType, bytes: []const u8) RowsError!usize {
    return switch (index_type) {
        .i32 => blk: {
            const value = std.mem.readInt(i32, bytes[0..4], .little);
            if (value < 0) return error.NegativeIndex;
            break :blk @intCast(value);
        },
        .i64 => blk: {
            const value = std.mem.readInt(i64, bytes[0..8], .little);
            if (value < 0) return error.NegativeIndex;
            if (value > std.math.maxInt(usize)) return error.OutOfBounds;
            break :blk @intCast(value);
        },
    };
}

fn indexSize(index_type: wire.IndexType) usize {
    return switch (index_type) {
        .i32 => @sizeOf(i32),
        .i64 => @sizeOf(i64),
    };
}

fn validateStride(stride: u64, required: usize) RowsError!void {
    if (stride < required) return error.InvalidStride;
}

fn positive(value: u32) RowsError!usize {
    if (value == 0) return error.InvalidShape;
    return @intCast(value);
}

fn checkedAdd(a: anytype, b: anytype) RowsError!usize {
    const lhs = try checkedUsize(a);
    const rhs = try checkedUsize(b);
    return std.math.add(usize, lhs, rhs) catch return error.OutOfBounds;
}

fn checkedMul(a: anytype, b: anytype) RowsError!usize {
    const lhs = try checkedUsize(a);
    const rhs = try checkedUsize(b);
    return std.math.mul(usize, lhs, rhs) catch return error.OutOfBounds;
}

fn checkedUsize(value: anytype) RowsError!usize {
    return std.math.cast(usize, value) orelse error.OutOfBounds;
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * @sizeOf(f32) ..][0..4], .little));
}

fn writeF32(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * @sizeOf(f32) ..][0..4], @bitCast(value), .little);
}

fn readF16(bytes: []const u8, index: usize) f32 {
    const half: f16 = @bitCast(std.mem.readInt(u16, bytes[index * @sizeOf(f16) ..][0..2], .little));
    return @floatCast(half);
}

fn writeF16(bytes: []u8, index: usize, value: f16) void {
    std.mem.writeInt(u16, bytes[index * @sizeOf(f16) ..][0..2], @bitCast(value), .little);
}

fn writeI32(bytes: []u8, index: usize, value: i32) void {
    std.mem.writeInt(i32, bytes[index * @sizeOf(i32) ..][0..4], value, .little);
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

test "set_rows writes indexed f32 rows into f16 backing span" {
    const head_dim = 4;
    const src_nb1 = head_dim * @sizeOf(f32);
    const src_nb2 = src_nb1 * 2;
    var src: [src_nb2 * 2]u8 = undefined;
    @memset(&src, 0);
    const values = [_]f32{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    };
    for (values, 0..) |value, i| writeF32(&src, i, value);

    var indices: [2 * @sizeOf(i32)]u8 = undefined;
    writeI32(&indices, 0, 2);
    writeI32(&indices, 1, 0);

    const dst_nb1 = head_dim * @sizeOf(f16);
    const dst_nb2 = dst_nb1 * 3;
    var dst: [dst_nb2 * 2]u8 = undefined;
    @memset(&dst, 0);

    try setRowsF32ToF16Bytes(.{
        .src = .{ .handle = 1, .offset = 0, .nbytes = src.len },
        .indices = .{ .handle = 2, .offset = 0, .nbytes = indices.len },
        .dst = .{ .handle = 3, .offset = 0, .nbytes = dst.len },
        .index_type = .i32,
        .head_dim = head_dim,
        .ne01 = 2,
        .ne02 = 2,
        .ne03 = 1,
        .ne11 = 1,
        .ne12 = 1,
        .src_nb1 = src_nb1,
        .src_nb2 = src_nb2,
        .src_nb3 = src_nb2 * 2,
        .indices_nb1 = indices.len,
        .indices_nb2 = indices.len,
        .dst_nb1 = dst_nb1,
        .dst_nb2 = dst_nb2,
        .dst_nb3 = dst_nb2 * 2,
    }, &src, &indices, &dst);

    try expectApprox(1.0, readF16(&dst, 2 * head_dim), 0.0);
    try expectApprox(5.0, readF16(&dst, 0), 0.0);
    try expectApprox(9.0, readF16(&dst, (dst_nb2 / @sizeOf(f16)) + 2 * head_dim), 0.0);
    try expectApprox(13.0, readF16(&dst, dst_nb2 / @sizeOf(f16)), 0.0);
}

test "set_rows rejects negative indices" {
    var src: [4]u8 = undefined;
    writeF32(&src, 0, 1.0);
    var indices: [4]u8 = undefined;
    writeI32(&indices, 0, -1);
    var dst: [2]u8 = undefined;

    try std.testing.expectError(error.NegativeIndex, setRowsF32ToF16Bytes(.{
        .src = .{ .handle = 1, .offset = 0, .nbytes = src.len },
        .indices = .{ .handle = 2, .offset = 0, .nbytes = indices.len },
        .dst = .{ .handle = 3, .offset = 0, .nbytes = dst.len },
        .index_type = .i32,
        .head_dim = 1,
        .ne01 = 1,
        .ne02 = 1,
        .ne03 = 1,
        .ne11 = 1,
        .ne12 = 1,
        .src_nb1 = 4,
        .src_nb2 = 4,
        .src_nb3 = 4,
        .indices_nb1 = 4,
        .indices_nb2 = 4,
        .dst_nb1 = 2,
        .dst_nb2 = 2,
        .dst_nb3 = 2,
    }, &src, &indices, &dst));
}
