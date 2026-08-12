const std = @import("std");
const shared = @import("shared");

const layout = shared.layout;
const wire = shared.wire;

pub const MatmulError = error{
    InvalidShape,
    InvalidLength,
    OutOfMemory,
};

pub const GroupProjection = struct {
    packed_weights: []const u8,
    dst_f32: []u8,
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
    var projections = [_]GroupProjection{.{ .packed_weights = packed_weights, .dst_f32 = dst_f32 }};
    return runGrouped(allocator, &projections, acts_f32, rows, cols, k, .w1a8);
}

pub fn runW158A8(
    allocator: std.mem.Allocator,
    packed_weights: []const u8,
    acts_f32: []const u8,
    dst_f32: []u8,
    rows: u32,
    cols: u32,
    k: u32,
) MatmulError!void {
    var projections = [_]GroupProjection{.{ .packed_weights = packed_weights, .dst_f32 = dst_f32 }};
    return runGrouped(allocator, &projections, acts_f32, rows, cols, k, .w158a8);
}

/// Reference oracle for the fixed two-projection command. Quantization is
/// performed once per activation column and reused by both projections. The
/// projection accumulation order is identical to two standalone calls.
pub fn runGroup2(
    allocator: std.mem.Allocator,
    projections: [2]GroupProjection,
    acts_f32: []const u8,
    rows: u32,
    cols: u32,
    k: u32,
    weight_fmt: wire.WeightFormat,
) MatmulError!void {
    return runGrouped(allocator, &projections, acts_f32, rows, cols, k, weight_fmt);
}

fn runGrouped(
    allocator: std.mem.Allocator,
    projections: []const GroupProjection,
    acts_f32: []const u8,
    rows: u32,
    cols: u32,
    k: u32,
    weight_fmt: wire.WeightFormat,
) MatmulError!void {
    const rows_usize: usize = @intCast(rows);
    const cols_usize: usize = @intCast(cols);
    const k_usize: usize = @intCast(k);
    if (projections.len == 0 or rows_usize == 0 or cols_usize == 0) return error.InvalidShape;
    const q1_blocks = layout.blocksPerRow(k_usize) catch return error.InvalidShape;
    const packed_len = switch (weight_fmt) {
        .w1a8 => layout.packedWeightBytes(rows_usize, k_usize),
        .w158a8 => layout.packedTernaryWeightBytes(rows_usize, k_usize),
    } catch return error.InvalidShape;
    const dst_len = layout.outputF32Bytes(rows_usize, cols_usize) catch return error.InvalidShape;
    for (projections) |projection| {
        if (projection.packed_weights.len != packed_len or projection.dst_f32.len != dst_len) return error.InvalidLength;
    }
    if (acts_f32.len != (layout.actsF32Bytes(cols_usize, k_usize) catch return error.InvalidShape)) return error.InvalidLength;

    const q8_blocks = q1_blocks * layout.q8_subblocks;
    const column = try allocator.alloc(f32, k_usize);
    defer allocator.free(column);
    const quants = try allocator.alloc(i8, k_usize);
    defer allocator.free(quants);
    const act_scales = try allocator.alloc(f32, q8_blocks);
    defer allocator.free(act_scales);

    for (0..cols_usize) |col| {
        for (0..k_usize) |i| column[i] = readF32(acts_f32, (col * k_usize + i) * @sizeOf(f32));
        layout.quantizeQ8_0F32Scales(column, quants, act_scales) catch return error.InvalidShape;

        for (projections) |projection| {
            switch (weight_fmt) {
                .w1a8 => try computeQ1Column(projection, quants, act_scales, rows_usize, k_usize, q1_blocks, col),
                .w158a8 => try computeW158Column(projection, quants, act_scales, rows_usize, k_usize, q1_blocks, col),
            }
        }
    }
}

fn computeQ1Column(
    projection: GroupProjection,
    quants: []const i8,
    act_scales: []const f32,
    rows: usize,
    k: usize,
    q1_blocks: usize,
    col: usize,
) MatmulError!void {
    for (0..rows) |row| {
        var acc: f32 = 0;
        for (0..q1_blocks) |q1| {
            const wscale: f32 = @floatCast(layout.packedWeightScale(projection.packed_weights, rows, k, row, q1) catch return error.InvalidShape);
            if (wscale == 0) continue;
            for (0..layout.q8_subblocks) |sub| {
                const ascale = act_scales[q1 * layout.q8_subblocks + sub];
                if (ascale == 0) continue;
                const bits = layout.packedWeightBits(projection.packed_weights, rows, k, row, q1, sub) catch return error.InvalidShape;
                const base = q1 * layout.q1_block + sub * layout.q8_block;
                var sum: i32 = 0;
                for (0..layout.q8_block) |i| {
                    const a: i32 = quants[base + i];
                    sum += if (((bits >> @intCast(i)) & 1) != 0) a else -a;
                }
                acc = @mulAdd(f32, @as(f32, @floatFromInt(sum)), wscale * ascale, acc);
            }
        }
        writeF32(projection.dst_f32, (col * rows + row) * @sizeOf(f32), acc);
    }
}

fn computeW158Column(
    projection: GroupProjection,
    quants: []const i8,
    act_scales: []const f32,
    rows: usize,
    k: usize,
    q1_blocks: usize,
    col: usize,
) MatmulError!void {
    for (0..rows) |row| {
        var acc: f32 = 0;
        for (0..q1_blocks) |q1| {
            for (0..layout.q8_subblocks) |sub| {
                const wscale: f32 = @floatCast(layout.packedTernaryWeightScale(projection.packed_weights, rows, k, row, q1, sub / 2) catch return error.InvalidShape);
                if (wscale == 0) continue;
                const ascale = act_scales[q1 * layout.q8_subblocks + sub];
                if (ascale == 0) continue;
                const base = q1 * layout.q1_block + sub * layout.q8_block;
                const codes = layout.packedTernaryWeightCodes(projection.packed_weights, rows, k, row, q1, sub) catch return error.InvalidShape;
                var sum: i32 = 0;
                for (0..layout.q8_block) |i| {
                    const code: u2 = @truncate(codes >> @intCast(i * 2));
                    const act: i32 = quants[base + i];
                    sum += switch (code) {
                        0 => -act,
                        1 => 0,
                        2 => act,
                        3 => unreachable, // rejected by packedTernaryWeightCodes
                    };
                }
                acc = @mulAdd(f32, @as(f32, @floatFromInt(sum)), wscale * ascale, acc);
            }
        }
        writeF32(projection.dst_f32, (col * rows + row) * @sizeOf(f32), acc);
    }
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

test "all positive weights and 127 activations" {
    const rows = layout.rows_per_block;
    const k = layout.q1_block;
    const q1_blocks = comptime layout.blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime layout.packedWeightBytes(rows, k) catch unreachable;
    var bits: [logical_len]u128 = [_]u128{std.math.maxInt(u128)} ** logical_len;
    var scales: [logical_len]f16 = [_]f16{1} ** logical_len;
    var weights: [packed_len]u8 = undefined;
    try layout.packWeightsFromLogical(rows, k, &bits, &scales, &weights);

    var acts: [k * 4]u8 = undefined;
    for (0..k) |i| writeF32(&acts, i * 4, 127);
    var dst: [rows * 4]u8 = undefined;
    try runQ1A8(std.testing.allocator, &weights, &acts, &dst, @intCast(rows), 1, @intCast(k));

    for (0..rows) |row| {
        try std.testing.expectEqual(@as(f32, 127 * layout.q1_block), readF32(&dst, row * 4));
    }
}

test "rowblock tail and multiple columns match logical reference" {
    const rows = layout.rows_per_block + 3;
    const cols = 2;
    const k = layout.q1_block * 2;
    const q1_blocks = comptime layout.blocksPerRow(k) catch unreachable;
    const q8_blocks = q1_blocks * layout.q8_subblocks;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime layout.packedWeightBytes(rows, k) catch unreachable;

    var bits: [logical_len]u128 = undefined;
    var scales: [logical_len]f16 = undefined;
    for (0..rows) |row| {
        for (0..q1_blocks) |q1| {
            var mask: u128 = 0;
            for (0..layout.q1_block) |bit| {
                if (((row * 17 + q1 * 31 + bit * 7) % 5) < 2) mask |= @as(u128, 1) << @intCast(bit);
            }
            bits[row * q1_blocks + q1] = mask;
            scales[row * q1_blocks + q1] = @floatCast(0.0625 * @as(f32, @floatFromInt((row + 1) * (q1 + 2))));
        }
    }

    var weights: [packed_len]u8 = undefined;
    try layout.packWeightsFromLogical(rows, k, &bits, &scales, &weights);

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
        try layout.quantizeQ8_0(&column, &quants, &act_scales);

        for (0..rows) |row| {
            var expected: f32 = 0;
            for (0..q1_blocks) |q1| {
                const weight_scale: f32 = @floatCast(scales[row * q1_blocks + q1]);
                if (weight_scale == 0) continue;
                for (0..layout.q8_subblocks) |sub| {
                    const act_scale: f32 = @floatCast(act_scales[q1 * layout.q8_subblocks + sub]);
                    if (act_scale == 0) continue;
                    const sub_bits: u32 = @truncate(bits[row * q1_blocks + q1] >> @intCast(sub * layout.q8_block));
                    const base = q1 * layout.q1_block + sub * layout.q8_block;
                    var sum: i32 = 0;
                    for (0..layout.q8_block) |i| {
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

test "ternary resident oracle applies minus zero plus selectors" {
    const rows = layout.rows_per_block;
    const k = layout.q1_block;
    const raw_blocks_per_row = k / layout.q2_source_block;
    const raw_len = rows * raw_blocks_per_row * layout.q2_source_block_bytes;
    const packed_len = comptime layout.packedTernaryWeightBytes(rows, k) catch unreachable;
    var raw: [raw_len]u8 = [_]u8{0} ** raw_len;
    for (0..rows) |row| {
        for (0..raw_blocks_per_row) |block| {
            const off = (row * raw_blocks_per_row + block) * layout.q2_source_block_bytes;
            std.mem.writeInt(u16, raw[off..][0..2], @bitCast(@as(f16, 1)), .little);
            for (0..layout.q2_source_block) |i| {
                const code: u8 = @intCast((block * layout.q2_source_block + i) % 3);
                raw[off + 2 + i / 4] |= code << @intCast((i % 4) * 2);
            }
        }
    }
    var weights: [packed_len]u8 = undefined;
    try layout.packWeightsFromGgmlQ2_0(rows, k, &raw, &weights);

    var acts: [k * @sizeOf(f32)]u8 = undefined;
    for (0..k) |i| writeF32(&acts, i * @sizeOf(f32), 127);
    var dst: [rows * @sizeOf(f32)]u8 = undefined;
    try runW158A8(std.testing.allocator, &weights, &acts, &dst, rows, 1, k);
    for (0..rows) |row| try std.testing.expectEqual(@as(f32, -127), readF32(&dst, row * @sizeOf(f32)));
}

test "grouped binary projections are bit-identical to standalone calls" {
    const rows = layout.rows_per_block;
    const cols = 2;
    const k = layout.q1_block;
    const logical_len = rows;
    const packed_len = comptime layout.packedWeightBytes(rows, k) catch unreachable;
    var bits0: [logical_len]u128 = [_]u128{std.math.maxInt(u128)} ** logical_len;
    var bits1: [logical_len]u128 = undefined;
    var scales0: [logical_len]f16 = [_]f16{1} ** logical_len;
    var scales1: [logical_len]f16 = [_]f16{0.5} ** logical_len;
    for (0..rows) |row| bits1[row] = if (@mod(row, 2) == 0) 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa else 0x55555555555555555555555555555555;
    var weights0: [packed_len]u8 = undefined;
    var weights1: [packed_len]u8 = undefined;
    try layout.packWeightsFromLogical(rows, k, &bits0, &scales0, &weights0);
    try layout.packWeightsFromLogical(rows, k, &bits1, &scales1, &weights1);

    var acts: [cols * k * @sizeOf(f32)]u8 = undefined;
    for (0..cols * k) |i| writeF32(&acts, i * @sizeOf(f32), @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 5.0);
    var standalone0: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var standalone1: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var grouped0: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var grouped1: [cols * rows * @sizeOf(f32)]u8 = undefined;
    try runQ1A8(std.testing.allocator, &weights0, &acts, &standalone0, rows, cols, k);
    try runQ1A8(std.testing.allocator, &weights1, &acts, &standalone1, rows, cols, k);
    try runGroup2(std.testing.allocator, .{
        .{ .packed_weights = &weights0, .dst_f32 = &grouped0 },
        .{ .packed_weights = &weights1, .dst_f32 = &grouped1 },
    }, &acts, rows, cols, k, .w1a8);
    try std.testing.expectEqualSlices(u8, &standalone0, &grouped0);
    try std.testing.expectEqualSlices(u8, &standalone1, &grouped1);
}

test "grouped ternary projections are bit-identical to standalone calls" {
    const rows = layout.rows_per_block;
    const cols = 2;
    const k = layout.q1_block;
    const raw_blocks_per_row = k / layout.q2_source_block;
    const raw_len = rows * raw_blocks_per_row * layout.q2_source_block_bytes;
    const packed_len = comptime layout.packedTernaryWeightBytes(rows, k) catch unreachable;
    var raw0: [raw_len]u8 = [_]u8{0} ** raw_len;
    var raw1: [raw_len]u8 = [_]u8{0} ** raw_len;
    for (0..rows) |row| {
        for (0..raw_blocks_per_row) |block| {
            const off = (row * raw_blocks_per_row + block) * layout.q2_source_block_bytes;
            std.mem.writeInt(u16, raw0[off..][0..2], @bitCast(@as(f16, 1)), .little);
            std.mem.writeInt(u16, raw1[off..][0..2], @bitCast(@as(f16, 0.5)), .little);
            for (0..layout.q2_source_block) |i| {
                raw0[off + 2 + i / 4] |= @as(u8, @intCast(i % 3)) << @intCast((i % 4) * 2);
                raw1[off + 2 + i / 4] |= @as(u8, @intCast((i + row + 1) % 3)) << @intCast((i % 4) * 2);
            }
        }
    }
    var weights0: [packed_len]u8 = undefined;
    var weights1: [packed_len]u8 = undefined;
    try layout.packWeightsFromGgmlQ2_0(rows, k, &raw0, &weights0);
    try layout.packWeightsFromGgmlQ2_0(rows, k, &raw1, &weights1);

    var acts: [cols * k * @sizeOf(f32)]u8 = undefined;
    for (0..cols * k) |i| writeF32(&acts, i * @sizeOf(f32), @as(f32, @floatFromInt(@as(i32, @intCast(i % 31)) - 15)) / 7.0);
    var standalone0: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var standalone1: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var grouped0: [cols * rows * @sizeOf(f32)]u8 = undefined;
    var grouped1: [cols * rows * @sizeOf(f32)]u8 = undefined;
    try runW158A8(std.testing.allocator, &weights0, &acts, &standalone0, rows, cols, k);
    try runW158A8(std.testing.allocator, &weights1, &acts, &standalone1, rows, cols, k);
    try runGroup2(std.testing.allocator, .{
        .{ .packed_weights = &weights0, .dst_f32 = &grouped0 },
        .{ .packed_weights = &weights1, .dst_f32 = &grouped1 },
    }, &acts, rows, cols, k, .w158a8);
    try std.testing.expectEqualSlices(u8, &standalone0, &grouped0);
    try std.testing.expectEqualSlices(u8, &standalone1, &grouped1);
}
