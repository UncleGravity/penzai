const std = @import("std");

pub const rows_per_block: usize = 16;
pub const weight_ports: usize = 2;
pub const rows_per_port: usize = rows_per_block / weight_ports;
pub const q1_block: usize = 128;
pub const q1_block_bytes: usize = 18; // ggml Q1_0 block size (2 B scale + 16 B bits)
pub const q8_block: usize = 32;
pub const q8_subblocks: usize = q1_block / q8_block;
pub const beat_bytes: usize = 8; // activation AXIS beat (64-bit)

// Resident weight layout for the v7 two-port PL kernel. Each rowblock has 16
// rows split into two contiguous 8-row port streams. Within one port stream, per
// (rowblock, q1block), there is one scale beat then one beat per Q8 sub-block.
// Every row lane owns one 32-bit slot in each beat; scale beats store the fp16
// scale in the low half of that slot. This makes the RTL combiner a pure zip:
// {port1_256b, port0_256b} -> one 512-bit kernel beat.
pub const weight_beat_bytes: usize = rows_per_block * 4; // ROWS * 32 bits = 512
pub const weight_port_beat_bytes: usize = rows_per_port * 4; // 8 lanes * 32 bits = 256
pub const packed_per_port_q1_block: usize = (1 + q8_subblocks) * weight_port_beat_bytes;
pub const packed_per_q1_block: usize = (1 + q8_subblocks) * weight_beat_bytes; // = 320
pub const acts_per_q1_block: usize = q8_subblocks * (q8_block + beat_bytes);

comptime {
    if (rows_per_block % weight_ports != 0) @compileError("Q1A8 rows must split evenly across ports");
    if (rows_per_port != 8) @compileError("Q1A8 v7 expects 8 rows per weight port");
    if (weight_port_beat_bytes != 32) @compileError("Q1A8 port beat size drifted");
    if (packed_per_port_q1_block != 160) @compileError("Q1A8 port block size drifted");
    if (packed_per_q1_block != 320) @compileError("Q1A8 packed block size drifted");
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

pub fn packedWeightPortBytes(rows: usize, k: usize) LayoutError!usize {
    if (rows == 0) return error.InvalidRows;
    return rowblocksFor(rows) * try blocksPerRow(k) * packed_per_port_q1_block;
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
    for (0..weight_ports) |port| {
        for (0..rowblocksFor(rows)) |rb| {
            const row_start = rb * rows_per_block + port * rows_per_port;
            const row_count = if (row_start < rows) @min(rows_per_port, rows - row_start) else 0;
            for (0..q1_blocks) |q1| {
                for (0..row_count) |lane| {
                    const scale_bits: u16 = @bitCast(weight_scales[(row_start + lane) * q1_blocks + q1]);
                    std.mem.writeInt(u16, out[cursor + lane * 4 ..][0..2], scale_bits, .little);
                }
                cursor += weight_port_beat_bytes;
                for (0..q8_subblocks) |sub| {
                    for (0..row_count) |lane| {
                        const bits = weight_bits[(row_start + lane) * q1_blocks + q1];
                        const part: u32 = @truncate(bits >> @intCast(sub * q8_block));
                        std.mem.writeInt(u32, out[cursor + lane * 4 ..][0..4], part, .little);
                    }
                    cursor += weight_port_beat_bytes;
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
    for (0..weight_ports) |port| {
        for (0..rowblocksFor(rows)) |rb| {
            const row_start = rb * rows_per_block + port * rows_per_port;
            const row_count = if (row_start < rows) @min(rows_per_port, rows - row_start) else 0;
            for (0..q1_blocks) |q1| {
                for (0..row_count) |lane| {
                    const source_offset = (row_start + lane) * source_row_bytes + q1 * q1_block_bytes;
                    const scale = std.mem.readInt(u16, q1_0_weights[source_offset..][0..2], .little);
                    std.mem.writeInt(u16, out[cursor + lane * 4 ..][0..2], scale, .little);
                }
                cursor += weight_port_beat_bytes;
                for (0..q8_subblocks) |sub| {
                    for (0..row_count) |lane| {
                        const source_offset = (row_start + lane) * source_row_bytes +
                            q1 * q1_block_bytes + @sizeOf(f16) + sub * (q8_block / 8);
                        const bits = std.mem.readInt(u32, q1_0_weights[source_offset..][0..4], .little);
                        std.mem.writeInt(u32, out[cursor + lane * 4 ..][0..4], bits, .little);
                    }
                    cursor += weight_port_beat_bytes;
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
    const port = lane / rows_per_port;
    const port_lane = lane % rows_per_port;
    const port_stride = rowblocksFor(rows) * q1_blocks * packed_per_port_q1_block;
    const block_base = port * port_stride + (rb * q1_blocks + q1) * packed_per_port_q1_block;
    return @bitCast(std.mem.readInt(u16, weights[block_base + port_lane * 4 ..][0..2], .little));
}

pub fn packedWeightBits(weights: []const u8, rows: usize, k: usize, row: usize, q1: usize, sub: usize) LayoutError!u32 {
    const q1_blocks = try blocksPerRow(k);
    if (weights.len != try packedWeightBytes(rows, k)) return error.InvalidLength;
    const rb = row / rows_per_block;
    const lane = row % rows_per_block;
    const port = lane / rows_per_port;
    const port_lane = lane % rows_per_port;
    const port_stride = rowblocksFor(rows) * q1_blocks * packed_per_port_q1_block;
    const block_base = port * port_stride + (rb * q1_blocks + q1) * packed_per_port_q1_block;
    const sub_base = block_base + weight_port_beat_bytes + sub * weight_port_beat_bytes;
    return std.mem.readInt(u32, weights[sub_base + port_lane * 4 ..][0..4], .little);
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

/// Vectorized Q8_0 quantizer, bit-identical to `quantizeQ8_0` (round-to-nearest
/// -even via the magic-number trick, same f32 scale reciprocal). Portable
/// `@Vector` code: lowers to NEON on the A53 board (where it replaces the scalar
/// per-element round on the PL activation feed) and to scalar elsewhere. Pinned
/// equal to the scalar oracle by the fuzz test below, so PL and PS still feed
/// bit-identical int8.
pub fn quantizeQ8_0Simd(column: []const f32, out_quants: []i8, out_scales: []f16) LayoutError!void {
    if (column.len == 0 or column.len % q8_block != 0) return error.InvalidK;
    if (out_quants.len != column.len or out_scales.len != column.len / q8_block) return error.InvalidLength;

    const Vec = @Vector(q8_block, f32);
    // 1.5 * 2^23: for |x| < 2^22, (x + magic) - magic rounds x to the nearest
    // integer using the FPU's default round-to-nearest-even mode.
    const magic: Vec = @splat(12582912.0);
    const inf_v: Vec = @splat(std.math.inf(f32));
    const zero_v: Vec = @splat(@as(f32, 0));

    for (0..out_scales.len) |block_index| {
        const base = block_index * q8_block;
        const v: Vec = column[base..][0..q8_block].*;
        const absv = @abs(v);
        // Non-finite values do not contribute to amax (abs >= inf is false for
        // both inf and NaN), matching the scalar isFinite filter.
        const cand = @select(f32, absv < inf_v, absv, zero_v);
        const amax = @reduce(.Max, cand);
        if (amax == 0) {
            @memset(out_quants[base..][0..q8_block], 0);
            out_scales[block_index] = 0;
            continue;
        }
        const scale = amax / 127.0;
        out_scales[block_index] = @floatCast(scale);
        const inv: Vec = @splat(1.0 / scale);
        const rounded = (v * inv + magic) - magic;
        const clamped = @max(@min(rounded, @as(Vec, @splat(@as(f32, 127)))), @as(Vec, @splat(@as(f32, -128))));
        const ints: @Vector(q8_block, i8) = @intFromFloat(clamped);
        out_quants[base..][0..q8_block].* = ints;
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

test "simd quantizer is bit-identical to scalar across magnitudes" {
    const blocks = 7;
    const n = blocks * q8_block;
    var prng = std.Random.DefaultPrng.init(0xC0DE_8A8);
    const rnd = prng.random();
    var col: [n]f32 = undefined;
    var qs: [n]i8 = undefined;
    var qv: [n]i8 = undefined;
    var ss: [blocks]f16 = undefined;
    var sv: [blocks]f16 = undefined;

    for (0..300) |_| {
        // Span a wide magnitude range so the round-half-to-even boundary and the
        // clamp are both exercised hard.
        for (&col) |*c| {
            const mag = std.math.pow(f32, 10.0, rnd.float(f32) * 8.0 - 4.0);
            c.* = (rnd.float(f32) - 0.5) * 2.0 * mag;
        }
        try quantizeQ8_0(&col, &qs, &ss);
        try quantizeQ8_0Simd(&col, &qv, &sv);
        try std.testing.expectEqualSlices(i8, &qs, &qv);
        try std.testing.expectEqualSlices(f16, &ss, &sv);
    }

    // Tie cases land exactly on .5 boundaries (amax = 127 makes scale = 1).
    for (&col, 0..) |*c, i| c.* = @floatFromInt(@as(i32, @intCast(i % 9)) - 4);
    col[0] = 127;
    try quantizeQ8_0(&col, &qs, &ss);
    try quantizeQ8_0Simd(&col, &qv, &sv);
    try std.testing.expectEqualSlices(i8, &qs, &qv);
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
