const std = @import("std");
const shared = @import("shared");
const simd = @import("matmul_q1a8_simd.zig");

const q1a8 = shared.q1a8;

const q8_lookup_bytes: usize = 256;
const q8_lookup_lanes: usize = q1a8.q8_block / 8;
const q8_lookup_entries_per_block: usize = q8_lookup_lanes * q8_lookup_bytes;

comptime {
    if (q1a8.q8_block != 32) @compileError("q8 lookup reducer assumes 32-wide q8 blocks");
}

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
    if (comptime simd.available) {
        return simd.runQ1A8(allocator, packed_weights, acts_f32, dst_f32, rows, cols, k);
    }
    return runQ1A8Scalar(allocator, packed_weights, acts_f32, dst_f32, rows, cols, k);
}

pub fn runQ1A8Scalar(
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
    const q8_blocks = q1_blocks * q1a8.q8_subblocks;
    const act_scales = try allocator.alloc(f32, q8_blocks);
    defer allocator.free(act_scales);
    const q8_sums = try allocator.alloc(i16, q8_blocks);
    defer allocator.free(q8_sums);
    const q8_lookups = try allocator.alloc(i16, q8_blocks * q8_lookup_entries_per_block);
    defer allocator.free(q8_lookups);
    const column = try allocator.alloc(f32, k_usize);
    defer allocator.free(column);

    const rowblocks = q1a8.rowblocksFor(rows_usize);
    const packed_per_rowblock = q1_blocks * q1a8.packed_per_q1_block;

    for (0..cols_usize) |col| {
        for (0..k_usize) |i| {
            column[i] = readF32(acts_f32, (col * k_usize + i) * @sizeOf(f32));
        }
        q1a8.quantizeQ8_0F32Scales(column, quants, act_scales) catch return error.InvalidShape;
        buildQ8Lookups(quants, q8_lookups, q8_sums);

        for (0..rowblocks) |rb| {
            const row_start = rb * q1a8.rows_per_block;
            const row_count = @min(q1a8.rows_per_block, rows_usize - row_start);
            const rb_base = rb * packed_per_rowblock;
            var accs = [_]f32{0} ** q1a8.rows_per_block;
            for (0..q1_blocks) |q1| {
                const block_base = rb_base + q1 * q1a8.packed_per_q1_block;
                var weight_scales = [_]f32{0} ** q1a8.rows_per_block;
                for (0..row_count) |lane| {
                    weight_scales[lane] = @floatCast(readF16(packed_weights, block_base + lane * @sizeOf(f16)));
                }

                for (0..q1a8.q8_subblocks) |sub| {
                    const q8 = q1 * q1a8.q8_subblocks + sub;
                    const act_scale = act_scales[q8];
                    if (act_scale == 0) continue;

                    const bits_base = block_base + q1a8.scales_bytes + sub * q1a8.wbits_bytes;
                    const lookup = q8_lookups[q8 * q8_lookup_entries_per_block ..][0..q8_lookup_entries_per_block];
                    const total = q8_sums[q8];
                    for (0..row_count) |lane| {
                        const weight_scale = weight_scales[lane];
                        if (weight_scale == 0) continue;

                        const bits = readU32(packed_weights, bits_base + lane * @sizeOf(u32));
                        const sum = lookupSignedSum(bits, lookup, total);
                        const scale = weight_scale * act_scale;
                        accs[lane] = @mulAdd(f32, @as(f32, @floatFromInt(sum)), scale, accs[lane]);
                    }
                }
            }
            for (0..row_count) |lane| {
                writeF32(dst_f32, (col * rows_usize + row_start + lane) * @sizeOf(f32), accs[lane]);
            }
        }
    }
}

fn buildQ8Lookups(quants: []const i8, lookups: []i16, totals: []i16) void {
    for (0..totals.len) |q8| {
        const quants_base = q8 * q1a8.q8_block;
        var total: i32 = 0;
        for (0..q1a8.q8_block) |i| {
            total += quants[quants_base + i];
        }
        totals[q8] = @intCast(total);

        const lookup_base = q8 * q8_lookup_entries_per_block;
        for (0..q8_lookup_lanes) |lane| {
            const lane_quants = quants[quants_base + lane * 8 ..][0..8];
            const table = lookups[lookup_base + lane * q8_lookup_bytes ..][0..q8_lookup_bytes];
            table[0] = 0;
            for (1..q8_lookup_bytes) |mask| {
                const previous = mask & (mask - 1);
                const bit: usize = @intCast(@ctz(mask));
                table[mask] = table[previous] + @as(i16, lane_quants[bit]);
            }
        }
    }
}

fn lookupSignedSum(bits: u32, lookup: []const i16, total: i16) i32 {
    const byte0: usize = @intCast(bits & 0xff);
    const byte1: usize = @intCast((bits >> 8) & 0xff);
    const byte2: usize = @intCast((bits >> 16) & 0xff);
    const byte3: usize = @intCast((bits >> 24) & 0xff);
    const selected =
        @as(i32, lookup[byte0]) +
        @as(i32, lookup[q8_lookup_bytes + byte1]) +
        @as(i32, lookup[2 * q8_lookup_bytes + byte2]) +
        @as(i32, lookup[3 * q8_lookup_bytes + byte3]);
    return selected * 2 - @as(i32, total);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn readF16(bytes: []const u8, offset: usize) f16 {
    return @bitCast(std.mem.readInt(u16, bytes[offset..][0..2], .little));
}

fn readU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
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
                if (((row * 17 + q1 * 31 + bit * 7) % 5) < 2) {
                    mask |= @as(u128, 1) << @intCast(bit);
                }
            }
            bits[row * q1_blocks + q1] = mask;
            const scale = 0.0625 * @as(f32, @floatFromInt((row + 1) * (q1 + 2)));
            scales[row * q1_blocks + q1] = @floatCast(scale);
        }
    }

    var weights: [packed_len]u8 = undefined;
    try q1a8.packWeightsFromLogical(rows, k, &bits, &scales, &weights);

    var acts: [cols * k * @sizeOf(f32)]u8 = undefined;
    for (0..cols) |col| {
        for (0..k) |i| {
            const centered = @as(i32, @intCast((i * 13 + col * 7) % 41)) - 20;
            const value = @as(f32, @floatFromInt(centered)) / 7.0;
            writeF32(&acts, (col * k + i) * @sizeOf(f32), value);
        }
    }

    var dst: [cols * rows * @sizeOf(f32)]u8 = undefined;
    try runQ1A8(std.testing.allocator, &weights, &acts, &dst, @intCast(rows), @intCast(cols), @intCast(k));
    var scalar_dst: [cols * rows * @sizeOf(f32)]u8 = undefined;
    try runQ1A8Scalar(std.testing.allocator, &weights, &acts, &scalar_dst, @intCast(rows), @intCast(cols), @intCast(k));
    try std.testing.expectEqualSlices(u8, &scalar_dst, &dst);

    var column: [k]f32 = undefined;
    var quants: [k]i8 = undefined;
    var act_scales: [q8_blocks]f16 = undefined;
    for (0..cols) |col| {
        for (0..k) |i| {
            column[i] = readF32(&acts, (col * k + i) * @sizeOf(f32));
        }
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

test "large rowblock split matches scalar fallback" {
    const rows = q1a8.rows_per_block * 64;
    const cols = 1;
    const k = q1a8.q1_block * 8;
    const q1_blocks = comptime q1a8.blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime q1a8.packedWeightBytes(rows, k) catch unreachable;

    const bits = try std.testing.allocator.alloc(u128, logical_len);
    defer std.testing.allocator.free(bits);
    const scales = try std.testing.allocator.alloc(f16, logical_len);
    defer std.testing.allocator.free(scales);
    for (0..rows) |row| {
        for (0..q1_blocks) |q1| {
            var mask: u128 = 0;
            for (0..q1a8.q1_block) |bit| {
                if (((row * 11 + q1 * 19 + bit * 23) % 7) < 3) {
                    mask |= @as(u128, 1) << @intCast(bit);
                }
            }
            bits[row * q1_blocks + q1] = mask;
            scales[row * q1_blocks + q1] = @floatCast(0.03125 * @as(f32, @floatFromInt((row % 13) + q1 + 1)));
        }
    }

    const weights = try std.testing.allocator.alloc(u8, packed_len);
    defer std.testing.allocator.free(weights);
    try q1a8.packWeightsFromLogical(rows, k, bits, scales, weights);

    const acts = try std.testing.allocator.alloc(u8, cols * k * @sizeOf(f32));
    defer std.testing.allocator.free(acts);
    for (0..k) |i| {
        const centered = @as(i32, @intCast((i * 29) % 53)) - 26;
        writeF32(acts, i * @sizeOf(f32), @as(f32, @floatFromInt(centered)) / 13.0);
    }

    const dst = try std.testing.allocator.alloc(u8, cols * rows * @sizeOf(f32));
    defer std.testing.allocator.free(dst);
    const scalar_dst = try std.testing.allocator.alloc(u8, cols * rows * @sizeOf(f32));
    defer std.testing.allocator.free(scalar_dst);

    try runQ1A8(std.testing.allocator, weights, acts, dst, @intCast(rows), @intCast(cols), @intCast(k));
    try runQ1A8Scalar(std.testing.allocator, weights, acts, scalar_dst, @intCast(rows), @intCast(cols), @intCast(k));
    try std.testing.expectEqualSlices(u8, scalar_dst, dst);
}
