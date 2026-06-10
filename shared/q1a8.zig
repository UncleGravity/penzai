const std = @import("std");

pub const rows_per_block: usize = 8;
pub const q1_block: usize = 128;
pub const q1_block_bytes: usize = 18;
pub const q8_block: usize = 32;
pub const q8_subblocks: usize = q1_block / q8_block;
pub const beat_bytes: usize = 8;

pub const scale_beats: usize = (rows_per_block + 3) / 4;
pub const wbits_beats: usize = (rows_per_block + 1) / 2;
pub const scales_bytes: usize = scale_beats * beat_bytes;
pub const wbits_bytes: usize = wbits_beats * beat_bytes;
pub const packed_per_q1_block: usize = scales_bytes + q8_subblocks * wbits_bytes;
pub const acts_per_q1_block: usize = q8_subblocks * (q8_block + beat_bytes);

comptime {
    if (packed_per_q1_block != rows_per_block * q1_block_bytes) {
        @compileError("Q1A8 packed weight layout no longer matches Q1_0 size");
    }
    if (packed_per_q1_block != 144) @compileError("Q1A8 packed block size drifted");
    if (acts_per_q1_block != 160) @compileError("Q1A8 acts block size drifted");
}

pub const LayoutError = error{
    InvalidRows,
    InvalidK,
    InvalidLength,
};

pub fn rowblocksFor(rows: usize) usize {
    return (rows + rows_per_block - 1) / rows_per_block;
}

pub fn blocksPerRow(k: usize) LayoutError!usize {
    if (k == 0 or k % q1_block != 0) return error.InvalidK;
    return k / q1_block;
}

pub fn packedWeightBytes(rows: usize, k: usize) LayoutError!usize {
    if (rows == 0) return error.InvalidRows;
    return rowblocksFor(rows) * try blocksPerRow(k) * packed_per_q1_block;
}

pub fn actsF32Bytes(cols: usize, k: usize) LayoutError!usize {
    _ = try blocksPerRow(k);
    return cols * k * @sizeOf(f32);
}

pub fn outputF32Bytes(rows: usize, cols: usize) LayoutError!usize {
    if (rows == 0) return error.InvalidRows;
    return rows * cols * @sizeOf(f32);
}

pub fn packWeightsFromLogical(
    rows: usize,
    k: usize,
    weight_bits: []const u128,
    weight_scales: []const f16,
    out: []u8,
) LayoutError!void {
    const q1_blocks = try blocksPerRow(k);
    if (weight_bits.len != rows * q1_blocks or weight_scales.len != rows * q1_blocks) {
        return error.InvalidLength;
    }
    if (out.len != try packedWeightBytes(rows, k)) return error.InvalidLength;

    @memset(out, 0);
    var cursor: usize = 0;
    for (0..rowblocksFor(rows)) |rb| {
        const row_start = rb * rows_per_block;
        const row_count = @min(rows_per_block, rows - row_start);
        for (0..q1_blocks) |q1| {
            for (0..scale_beats) |beat| {
                var word: u64 = 0;
                for (0..4) |local| {
                    const lane = beat * 4 + local;
                    if (lane < row_count) {
                        const scale_bits: u64 = @as(u16, @bitCast(weight_scales[(row_start + lane) * q1_blocks + q1]));
                        word |= scale_bits << @intCast(local * 16);
                    }
                }
                putU64(out, &cursor, word);
            }

            for (0..q8_subblocks) |sub| {
                for (0..wbits_beats) |beat| {
                    var word: u64 = 0;
                    for (0..2) |local| {
                        const lane = beat * 2 + local;
                        if (lane < row_count) {
                            const bits = weight_bits[(row_start + lane) * q1_blocks + q1];
                            const part: u64 = @as(u32, @truncate(bits >> @intCast(sub * q8_block)));
                            word |= part << @intCast(local * 32);
                        }
                    }
                    putU64(out, &cursor, word);
                }
            }
        }
    }
}

pub fn packWeightsFromGgmlQ1_0(
    rows: usize,
    k: usize,
    q1_0_weights: []const u8,
    out: []u8,
) LayoutError!void {
    const q1_blocks = try blocksPerRow(k);
    if (q1_0_weights.len != rows * q1_blocks * q1_block_bytes) return error.InvalidLength;
    if (out.len != try packedWeightBytes(rows, k)) return error.InvalidLength;

    @memset(out, 0);
    var cursor: usize = 0;
    const source_row_bytes = q1_blocks * q1_block_bytes;
    for (0..rowblocksFor(rows)) |rb| {
        const row_start = rb * rows_per_block;
        const row_count = @min(rows_per_block, rows - row_start);
        for (0..q1_blocks) |q1| {
            for (0..scale_beats) |beat| {
                var word: u64 = 0;
                for (0..4) |local| {
                    const lane = beat * 4 + local;
                    if (lane < row_count) {
                        const source_offset = (row_start + lane) * source_row_bytes + q1 * q1_block_bytes;
                        const scale = std.mem.readInt(u16, q1_0_weights[source_offset..][0..2], .little);
                        word |= @as(u64, scale) << @intCast(local * 16);
                    }
                }
                putU64(out, &cursor, word);
            }

            for (0..q8_subblocks) |sub| {
                for (0..wbits_beats) |beat| {
                    var word: u64 = 0;
                    for (0..2) |local| {
                        const lane = beat * 2 + local;
                        if (lane < row_count) {
                            const source_offset = (row_start + lane) * source_row_bytes +
                                q1 * q1_block_bytes +
                                @sizeOf(f16) +
                                sub * (q8_block / 8);
                            const bits = std.mem.readInt(u32, q1_0_weights[source_offset..][0..4], .little);
                            word |= @as(u64, bits) << @intCast(local * 32);
                        }
                    }
                    putU64(out, &cursor, word);
                }
            }
        }
    }
}

pub fn packedWeightScale(weights: []const u8, rows: usize, k: usize, row: usize, q1: usize) LayoutError!f16 {
    const q1_blocks = try blocksPerRow(k);
    if (weights.len != try packedWeightBytes(rows, k)) return error.InvalidLength;
    const rb = row / rows_per_block;
    const lane = row % rows_per_block;
    const block_base = (rb * q1_blocks + q1) * packed_per_q1_block;
    const beat = lane / 4;
    const local = lane % 4;
    const word = readU64(weights, block_base + beat * beat_bytes);
    return @bitCast(@as(u16, @truncate(word >> @intCast(local * 16))));
}

pub fn packedWeightBits(weights: []const u8, rows: usize, k: usize, row: usize, q1: usize, sub: usize) LayoutError!u32 {
    const q1_blocks = try blocksPerRow(k);
    if (weights.len != try packedWeightBytes(rows, k)) return error.InvalidLength;
    const rb = row / rows_per_block;
    const lane = row % rows_per_block;
    const block_base = (rb * q1_blocks + q1) * packed_per_q1_block;
    const sub_base = block_base + scales_bytes + sub * wbits_bytes;
    const beat = lane / 2;
    const local = lane % 2;
    const word = readU64(weights, sub_base + beat * beat_bytes);
    return @truncate(word >> @intCast(local * 32));
}

pub fn quantizeQ8_0(column: []const f32, out_quants: []i8, out_scales: []f16) LayoutError!void {
    return quantizeQ8_0WithScaleType(f16, column, out_quants, out_scales);
}

pub fn quantizeQ8_0F32Scales(column: []const f32, out_quants: []i8, out_scales: []f32) LayoutError!void {
    return quantizeQ8_0WithScaleType(f32, column, out_quants, out_scales);
}

fn quantizeQ8_0WithScaleType(comptime Scale: type, column: []const f32, out_quants: []i8, out_scales: []Scale) LayoutError!void {
    comptime {
        if (Scale != f16 and Scale != f32) @compileError("unsupported q8 scale type");
    }
    if (column.len == 0 or column.len % q8_block != 0) return error.InvalidK;
    if (out_quants.len != column.len or out_scales.len != column.len / q8_block) return error.InvalidLength;

    for (0..out_scales.len) |block_index| {
        const base = block_index * q8_block;
        var amax: f32 = 0;
        for (column[base..][0..q8_block]) |value| {
            if (std.math.isFinite(value)) amax = @max(amax, @abs(value));
        }
        if (amax == 0) {
            @memset(out_quants[base..][0..q8_block], 0);
            out_scales[block_index] = 0;
            continue;
        }

        const scale = amax / 127.0;
        const scale_f16: f16 = @floatCast(scale);
        out_scales[block_index] = if (Scale == f16) scale_f16 else @floatCast(scale_f16);
        const inv_scale: f32 = 1.0 / scale;
        for (column[base..][0..q8_block], 0..) |value, i| {
            const quantized = roundNearestEven(value * inv_scale);
            out_quants[base + i] = @intCast(std.math.clamp(quantized, -128, 127));
        }
    }
}

fn roundNearestEven(value: f32) i32 {
    const floored = @floor(value);
    const whole: i32 = @intFromFloat(floored);
    const frac = value - floored;
    if (frac < 0.5) return whole;
    if (frac > 0.5) return whole + 1;
    return if (@mod(whole, 2) == 0) whole else whole + 1;
}

fn putU64(out: []u8, cursor: *usize, value: u64) void {
    std.mem.writeInt(u64, out[cursor.*..][0..8], value, .little);
    cursor.* += 8;
}

fn readU64(bytes: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

test "pack logical weights and read back lanes" {
    const rows = rows_per_block + 3;
    const k = q1_block;
    const q1_blocks = comptime blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime packedWeightBytes(rows, k) catch unreachable;
    var bits: [logical_len]u128 = [_]u128{0} ** logical_len;
    var scales: [logical_len]f16 = [_]f16{0} ** logical_len;
    for (&bits, 0..) |*b, i| b.* = @as(u128, i + 1) << 32 | 0xFFFF_FFFF;
    for (&scales, 0..) |*s, i| s.* = @floatCast(@as(f32, @floatFromInt(i + 1)));

    var packed_buf: [packed_len]u8 = undefined;
    try packWeightsFromLogical(rows, k, &bits, &scales, &packed_buf);

    try std.testing.expectEqual(scales[0], try packedWeightScale(&packed_buf, rows, k, 0, 0));
    try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), try packedWeightBits(&packed_buf, rows, k, 0, 0, 0));
    try std.testing.expectEqual(scales[rows - 1], try packedWeightScale(&packed_buf, rows, k, rows - 1, 0));
}

test "pack ggml q1_0 bytes matches logical packer" {
    const rows = rows_per_block + 1;
    const k = q1_block;
    const q1_blocks = comptime blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const raw_len = rows * q1_blocks * q1_block_bytes;
    const packed_len = comptime packedWeightBytes(rows, k) catch unreachable;

    var bits: [logical_len]u128 = undefined;
    var scales: [logical_len]f16 = undefined;
    var raw: [raw_len]u8 = undefined;
    for (0..logical_len) |i| {
        bits[i] = (@as(u128, 0xABCD_EF01) << 96) | (@as(u128, i + 1) << 32) | 0xFFFF_FFFF;
        scales[i] = @floatCast(@as(f32, @floatFromInt(i + 1)));
        const off = i * q1_block_bytes;
        std.mem.writeInt(u16, raw[off..][0..2], @bitCast(scales[i]), .little);
        for (0..4) |sub| {
            std.mem.writeInt(u32, raw[off + 2 + sub * 4 ..][0..4], @truncate(bits[i] >> @intCast(sub * 32)), .little);
        }
    }

    var from_logical: [packed_len]u8 = undefined;
    var from_raw: [packed_len]u8 = undefined;
    try packWeightsFromLogical(rows, k, &bits, &scales, &from_logical);
    try packWeightsFromGgmlQ1_0(rows, k, &raw, &from_raw);
    try std.testing.expectEqualSlices(u8, &from_logical, &from_raw);
}

test "quantize exact scale when amax is 127" {
    var column = [_]f32{127} ** q1_block;
    var quants: [q1_block]i8 = undefined;
    var scales: [q8_subblocks]f16 = undefined;
    try quantizeQ8_0(&column, &quants, &scales);
    for (quants) |q| try std.testing.expectEqual(@as(i8, 127), q);
    for (scales) |s| try std.testing.expectEqual(@as(f16, 1), s);
}

test "quantize f32 scales match f16 scale output" {
    var column = [_]f32{0} ** q1_block;
    for (&column, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) * 0.03125;
    }
    var quants_f16: [q1_block]i8 = undefined;
    var quants_f32: [q1_block]i8 = undefined;
    var scales_f16: [q8_subblocks]f16 = undefined;
    var scales_f32: [q8_subblocks]f32 = undefined;

    try quantizeQ8_0(&column, &quants_f16, &scales_f16);
    try quantizeQ8_0F32Scales(&column, &quants_f32, &scales_f32);

    try std.testing.expectEqualSlices(i8, &quants_f16, &quants_f32);
    for (scales_f16, scales_f32) |scale_f16, scale_f32| {
        try std.testing.expectEqual(@as(f32, @floatCast(scale_f16)), scale_f32);
    }
}

test "quantize uses unrounded scale reciprocal" {
    var column = [_]f32{0} ** q1_block;
    column[0] = 0.001;
    column[1] = 0.0000118094488;
    var quants: [q1_block]i8 = undefined;
    var scales: [q8_subblocks]f16 = undefined;

    try quantizeQ8_0(&column, &quants, &scales);

    try std.testing.expectEqual(@as(i8, 127), quants[0]);
    try std.testing.expectEqual(@as(i8, 1), quants[1]);
}

test "quantize rounds ties to nearest even" {
    var column = [_]f32{0} ** q1_block;
    column[0] = 127;
    column[1] = 2.5;
    column[2] = -2.5;
    var quants: [q1_block]i8 = undefined;
    var scales: [q8_subblocks]f16 = undefined;

    try quantizeQ8_0(&column, &quants, &scales);

    try std.testing.expectEqual(@as(i8, 2), quants[1]);
    try std.testing.expectEqual(@as(i8, -2), quants[2]);
}
