const std = @import("std");
const q1a8 = @import("q1a8");
const wire = @import("wire");
const runtime_mod = @import("runtime");
const link_mod = @import("link");

test "fake link alloc upload copy download" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const src = try link.alloc(16, 64);
    const dst = try link.alloc(16, 64);
    try link.upload(src, "abcdefghijklmnop");
    try link.runGraph(&.{.{ .copy = .{ .src = src, .dst = dst } }});

    var out: [16]u8 = undefined;
    try link.download(dst, &out);
    try std.testing.expectEqualSlices(u8, "abcdefghijklmnop", &out);
}

test "fake link q1a8 matmul reference" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const rows = q1a8.rows_per_block;
    const cols = 1;
    const k = q1a8.q1_block;
    const q1_blocks = comptime q1a8.blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime q1a8.packedWeightBytes(rows, k) catch unreachable;

    var bits: [logical_len]u128 = [_]u128{std.math.maxInt(u128)} ** logical_len;
    var scales: [logical_len]f16 = [_]f16{1} ** logical_len;
    var packed_buf: [packed_len]u8 = undefined;
    try q1a8.packWeightsFromLogical(rows, k, &bits, &scales, &packed_buf);

    var acts: [k * @sizeOf(f32)]u8 = undefined;
    for (0..k) |i| writeF32(&acts, i * @sizeOf(f32), 127);

    const weights_range = try link.alloc(packed_buf.len, 64);
    const acts_range = try link.alloc(acts.len, 64);
    const dst_range = try link.alloc(rows * cols * @sizeOf(f32), 64);
    try link.upload(weights_range, &packed_buf);
    try link.upload(acts_range, &acts);

    try link.runGraph(&.{.{ .matmul_q1a8 = .{
        .weights = weights_range,
        .acts = acts_range,
        .dst = dst_range,
        .rows = @intCast(rows),
        .cols = @intCast(cols),
        .k = @intCast(k),
    } }});

    var out: [rows * @sizeOf(f32)]u8 = undefined;
    try link.download(dst_range, &out);
    for (0..rows) |row| {
        try std.testing.expectEqual(@as(f32, 127 * q1a8.q1_block), readF32(&out, row * @sizeOf(f32)));
    }
}

test "fake link ps f32 command graph" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const a = try allocTensor(&link, 2);
    const b = try allocTensor(&link, 2);
    const weight = try allocTensor(&link, 2);
    const out_rms = try allocTensor(&link, 2);
    const out_rope = try allocTensor(&link, 2);
    const out_softmax = try allocTensor(&link, 2);
    const out_silu = try allocTensor(&link, 2);
    const out_swiglu = try allocTensor(&link, 2);
    const out_add = try allocTensor(&link, 2);
    const out_mul = try allocTensor(&link, 2);
    const out_scale = try allocTensor(&link, 2);
    const out_add_scaled = try allocTensor(&link, 2);

    try uploadTensor(&link, a, &.{ 3, 4 });
    try uploadTensor(&link, b, &.{ 1, 2 });
    try uploadTensor(&link, weight, &.{ 1, 1 });

    try link.runGraph(&.{
        .{ .rmsnorm = .{ .input = a, .weight = weight, .dst = out_rms, .eps = 0 } },
        .{ .rope = .{ .input = b, .dst = out_rope, .position = 1, .theta = 10000 } },
        .{ .softmax = .{ .src = b, .dst = out_softmax } },
        .{ .silu = .{ .src = b, .dst = out_silu } },
        .{ .swiglu = .{ .lhs = b, .rhs = a, .dst = out_swiglu } },
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = out_add } },
        .{ .mul_f32 = .{ .lhs = a, .rhs = b, .dst = out_mul } },
        .{ .scale_f32 = .{ .src = b, .dst = out_scale, .scale = 0.5 } },
        .{ .add_scaled_f32 = .{ .lhs = a, .rhs = b, .dst = out_add_scaled, .rhs_scale = 0.5 } },
    });

    try expectTensor(&link, out_rms, &.{ 0.84852815, 1.1313709 });
    try expectTensor(&link, out_rope, &.{ -1.1426396, 1.9220756 });
    try expectTensor(&link, out_softmax, &.{ 0.26894143, 0.7310586 });
    try expectTensor(&link, out_silu, &.{ 0.7310586, 1.761594 });
    try expectTensor(&link, out_swiglu, &.{ 2.1931758, 7.046376 });
    try expectTensor(&link, out_add, &.{ 4, 6 });
    try expectTensor(&link, out_mul, &.{ 3, 8 });
    try expectTensor(&link, out_scale, &.{ 0.5, 1 });
    try expectTensor(&link, out_add_scaled, &.{ 3.5, 5 });
}

fn allocTensor(link: *link_mod.FakeLink, len: usize) !wire.TensorRange {
    return link.alloc(len * @sizeOf(f32), @alignOf(f32));
}

fn uploadTensor(link: *link_mod.FakeLink, range: wire.TensorRange, values: []const f32) !void {
    const bytes = try std.testing.allocator.alloc(u8, values.len * @sizeOf(f32));
    defer std.testing.allocator.free(bytes);
    for (values, 0..) |value, i| {
        writeF32(bytes, i * @sizeOf(f32), value);
    }
    try link.upload(range, bytes);
}

fn expectTensor(link: *link_mod.FakeLink, range: wire.TensorRange, expected: []const f32) !void {
    const bytes = try std.testing.allocator.alloc(u8, expected.len * @sizeOf(f32));
    defer std.testing.allocator.free(bytes);
    try link.download(range, bytes);
    for (expected, 0..) |value, i| {
        try expectApprox(value, readF32(bytes, i * @sizeOf(f32)), 0.000001);
    }
}

fn expectApprox(expected: f32, actual: f32, tolerance: f32) !void {
    try std.testing.expect(@abs(expected - actual) <= tolerance);
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}
