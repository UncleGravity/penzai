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

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}
