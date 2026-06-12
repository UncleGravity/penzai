const std = @import("std");
const shared = @import("shared");

const q1a8 = shared.q1a8;

pub const MatmulError = error{
    InvalidShape,
    InvalidLength,
    OutOfMemory,
};

/// Q1A8 matmul reference oracle: Y[rows,cols] = W (q1, resident packed layout) x
/// A (int8, quantized from f32 acts). Reads weights through the layout accessors,
/// so it follows the packing wherever it lives. This is the correctness oracle
/// (golden tests, PL verify, bring-up fallback) -- clarity over speed; the PL
/// kernel is the fast path. Integer sub-sums are exact; the fp16-scale apply is
/// within the documented ε. Output is column-major: dst[col*rows + row].
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

    const q8_blocks = q1_blocks * q1a8.q8_subblocks;
    const column = try allocator.alloc(f32, k_usize);
    defer allocator.free(column);
    const quants = try allocator.alloc(i8, k_usize);
    defer allocator.free(quants);
    const act_scales = try allocator.alloc(f32, q8_blocks);
    defer allocator.free(act_scales);

    for (0..cols_usize) |col| {
        for (0..k_usize) |i| column[i] = readF32(acts_f32, (col * k_usize + i) * @sizeOf(f32));
        q1a8.quantizeQ8_0F32Scales(column, quants, act_scales) catch return error.InvalidShape;

        for (0..rows_usize) |row| {
            var acc: f32 = 0;
            for (0..q1_blocks) |q1| {
                const wscale: f32 = @floatCast(q1a8.packedWeightScale(packed_weights, rows_usize, k_usize, row, q1) catch return error.InvalidShape);
                if (wscale == 0) continue;
                for (0..q1a8.q8_subblocks) |sub| {
                    const ascale = act_scales[q1 * q1a8.q8_subblocks + sub];
                    if (ascale == 0) continue;
                    const bits = q1a8.packedWeightBits(packed_weights, rows_usize, k_usize, row, q1, sub) catch return error.InvalidShape;
                    const base = q1 * q1a8.q1_block + sub * q1a8.q8_block;
                    var sum: i32 = 0;
                    for (0..q1a8.q8_block) |i| {
                        const a: i32 = quants[base + i];
                        sum += if (((bits >> @intCast(i)) & 1) != 0) a else -a;
                    }
                    acc = @mulAdd(f32, @as(f32, @floatFromInt(sum)), wscale * ascale, acc);
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

test "rowblock tail and multiple columns match logical reference" {
    const rows = q1a8.rows_per_block + 3;
    const cols = 2;
    const k = q1a8.q1_block * 2;
    const q1_blocks = comptime q1a8.blocksPerRow(k) catch unreachable;
    const q8_blocks = q1_blocks * q1a8.q8_subblocks;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime q1a8.packedWeightBytes(rows, k) catch unreachable;

    var bits: [logical_len]u128 = undefined;
    var scales: [logical_len]f16 = undefined;
    for (0..rows) |row| {
        for (0..q1_blocks) |q1| {
            var mask: u128 = 0;
            for (0..q1a8.q1_block) |bit| {
                if (((row * 17 + q1 * 31 + bit * 7) % 5) < 2) mask |= @as(u128, 1) << @intCast(bit);
            }
            bits[row * q1_blocks + q1] = mask;
            scales[row * q1_blocks + q1] = @floatCast(0.0625 * @as(f32, @floatFromInt((row + 1) * (q1 + 2))));
        }
    }

    var weights: [packed_len]u8 = undefined;
    try q1a8.packWeightsFromLogical(rows, k, &bits, &scales, &weights);

    var acts: [cols * k * @sizeOf(f32)]u8 = undefined;
    for (0..cols) |col| {
        for (0..k) |i| {
            const centered = @as(i32, @intCast((i * 13 + col * 7) % 41)) - 20;
            writeF32(&acts, (col * k + i) * @sizeOf(f32), @as(f32, @floatFromInt(centered)) / 7.0);
        }
    }

    var dst: [cols * rows * @sizeOf(f32)]u8 = undefined;
    try runQ1A8(std.testing.allocator, &weights, &acts, &dst, @intCast(rows), @intCast(cols), @intCast(k));

    var column: [k]f32 = undefined;
    var quants: [k]i8 = undefined;
    var act_scales: [q8_blocks]f16 = undefined;
    for (0..cols) |col| {
        for (0..k) |i| column[i] = readF32(&acts, (col * k + i) * @sizeOf(f32));
        try q1a8.quantizeQ8_0(&column, &quants, &act_scales);

        for (0..rows) |row| {
            var expected: f32 = 0;
            for (0..q1_blocks) |q1| {
                const weight_scale: f32 = @floatCast(scales[row * q1_blocks + q1]);
                if (weight_scale == 0) continue;
                for (0..q1a8.q8_subblocks) |sub| {
                    const act_scale: f32 = @floatCast(act_scales[q1 * q1a8.q8_subblocks + sub]);
                    if (act_scale == 0) continue;
                    const sub_bits: u32 = @truncate(bits[row * q1_blocks + q1] >> @intCast(sub * q1a8.q8_block));
                    const base = q1 * q1a8.q1_block + sub * q1a8.q8_block;
                    var sum: i32 = 0;
                    for (0..q1a8.q8_block) |i| {
                        const act: i32 = quants[base + i];
                        sum += if (((sub_bits >> @intCast(i)) & 1) != 0) act else -act;
                    }
                    expected = @mulAdd(f32, @as(f32, @floatFromInt(sum)), weight_scale * act_scale, expected);
                }
            }
            const got = readF32(&dst, (col * rows + row) * @sizeOf(f32));
            try std.testing.expectApproxEqAbs(expected, got, 0.001);
        }
    }
}
