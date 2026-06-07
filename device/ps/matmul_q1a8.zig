const std = @import("std");
const q1a8 = @import("q1a8");

pub const MatmulError = error{
    InvalidShape,
    InvalidLength,
    OutOfMemory,
};

pub fn runQ1A8(
    allocator: std.mem.Allocator,
    packed_weights: []const u8,
    acts_f32: []const u8,
    dst_f32: []u8,
    rows: u32,
    cols: u32,
    k: u32,
) MatmulError!void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    const k_usize: usize = @intCast(k);
    if (rows_usize == 0 or cols_usize == 0) return error.InvalidShape;
    const q1_blocks = q1a8.blocksPerRow(k_usize) catch return error.InvalidShape;
    if (packed_weights.len != (q1a8.packedWeightBytes(rows_usize, k_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }
    if (acts_f32.len != (q1a8.actsF32Bytes(cols_usize, k_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }
    if (dst_f32.len != (q1a8.outputF32Bytes(rows_usize, cols_usize) catch return error.InvalidShape)) {
        return error.InvalidLength;
    }

    const quants = try allocator.alloc(i8, k_usize);
    defer allocator.free(quants);
    const act_scales = try allocator.alloc(f16, q1_blocks * q1a8.q8_subblocks);
    defer allocator.free(act_scales);
    const column = try allocator.alloc(f32, k_usize);
    defer allocator.free(column);

    for (0..cols_usize) |col| {
        for (0..k_usize) |i| {
            column[i] = readF32(acts_f32, (col * k_usize + i) * @sizeOf(f32));
        }
        q1a8.quantizeQ8_0(column, quants, act_scales) catch return error.InvalidShape;

        for (0..rows_usize) |row| {
            var acc: f32 = 0;
            for (0..q1_blocks) |q1| {
                const weight_scale: f32 = @floatCast(q1a8.packedWeightScale(packed_weights, rows_usize, k_usize, row, q1) catch return error.InvalidLength);
                if (weight_scale == 0) continue;
                for (0..q1a8.q8_subblocks) |sub| {
                    const act_scale: f32 = @floatCast(act_scales[q1 * q1a8.q8_subblocks + sub]);
                    if (act_scale == 0) continue;
                    const bits = q1a8.packedWeightBits(packed_weights, rows_usize, k_usize, row, q1, sub) catch return error.InvalidLength;
                    var sum: i32 = 0;
                    for (0..q1a8.q8_block) |i| {
                        const act: i32 = quants[q1 * q1a8.q1_block + sub * q1a8.q8_block + i];
                        sum += if (((bits >> @intCast(i)) & 1) != 0) act else -act;
                    }
                    acc += weight_scale * act_scale * @as(f32, @floatFromInt(sum));
                }
            }
            writeF32(dst_f32, (col * rows_usize + row) * @sizeOf(f32), acc);
        }
    }
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

test "all positive weights and 127 activations" {
    const rows = q1a8.rows_per_block;
    const k = q1a8.q1_block;
    const q1_blocks = comptime q1a8.blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime q1a8.packedWeightBytes(rows, k) catch unreachable;
    var bits: [logical_len]u128 = [_]u128{std.math.maxInt(u128)} ** logical_len;
    var scales: [logical_len]f16 = [_]f16{1} ** logical_len;
    var weights: [packed_len]u8 = undefined;
    try q1a8.packWeightsFromLogical(rows, k, &bits, &scales, &weights);

    var acts: [k * 4]u8 = undefined;
    for (0..k) |i| writeF32(&acts, i * 4, 127);
    var dst: [rows * 4]u8 = undefined;
    try runQ1A8(std.testing.allocator, &weights, &acts, &dst, @intCast(rows), 1, @intCast(k));

    for (0..rows) |row| {
        try std.testing.expectEqual(@as(f32, 127 * q1a8.q1_block), readF32(&dst, row * 4));
    }
}
