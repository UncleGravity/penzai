const std = @import("std");
const q1a8 = @import("q1a8");
const protocol_transport = @import("protocol_transport");
const wire = @import("wire");
const runtime_mod = @import("runtime");
const link_mod = @import("link");

pub const RunError = error{
    InvalidShape,
    OutOfMemory,
    Protocol,
    RemoteFailed,
    Transport,
};

pub const RunOptions = struct {
    rows: u32 = @intCast(q1a8.rows_per_block),
    cols: u32 = 1,
    k: u32 = @intCast(q1a8.q1_block),
    heap_mib: u32 = 16,
};

pub const RunResult = struct {
    rows: u32,
    cols: u32,
    k: u32,
    command_count: u64,
    weights_nbytes: usize,
    acts_nbytes: usize,
    dst_nbytes: usize,
    expected: f32,
    max_abs_diff: f32,
};

pub fn runFakeMatmul(allocator: std.mem.Allocator, options: RunOptions) RunError!RunResult {
    var runtime = try runtime_mod.Runtime.init(allocator, try heapBytes(options.heap_mib));
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(allocator, &runtime);
    return runMatmulWithLink(allocator, options, &link);
}

pub fn runTcpMatmul(io: std.Io, allocator: std.mem.Allocator, spec: protocol_transport.TcpSpec, options: RunOptions) RunError!RunResult {
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    return runMatmulWithLink(allocator, options, &link);
}

fn runMatmulWithLink(allocator: std.mem.Allocator, options: RunOptions, link: anytype) RunError!RunResult {
    const rows = try validatePositive(options.rows);
    const cols = try validatePositive(options.cols);
    const k = try validatePositive(options.k);

    const q1_blocks = q1a8.blocksPerRow(k) catch return error.InvalidShape;
    const logical_len = try checkedMul(rows, q1_blocks);
    const act_count = try checkedMul(cols, k);
    const output_count = try checkedMul(rows, cols);
    const weights_nbytes = q1a8.packedWeightBytes(rows, k) catch return error.InvalidShape;
    const acts_nbytes = q1a8.actsF32Bytes(cols, k) catch return error.InvalidShape;
    const dst_nbytes = q1a8.outputF32Bytes(rows, cols) catch return error.InvalidShape;

    const weight_bits = try allocator.alloc(u128, logical_len);
    defer allocator.free(weight_bits);
    const weight_scales = try allocator.alloc(f16, logical_len);
    defer allocator.free(weight_scales);
    @memset(weight_bits, std.math.maxInt(u128));
    @memset(weight_scales, 1);

    const packed_weights = try allocator.alloc(u8, weights_nbytes);
    defer allocator.free(packed_weights);
    q1a8.packWeightsFromLogical(rows, k, weight_bits, weight_scales, packed_weights) catch return error.InvalidShape;

    const acts = try allocator.alloc(u8, acts_nbytes);
    defer allocator.free(acts);
    for (0..act_count) |i| writeF32(acts, i * @sizeOf(f32), 127);

    const out = try allocator.alloc(u8, dst_nbytes);
    defer allocator.free(out);
    @memset(out, 0);

    try link.hello();

    const weights_range = try link.alloc(@intCast(packed_weights.len), 64);
    defer link.free(weights_range) catch {};
    const acts_range = try link.alloc(@intCast(acts.len), 64);
    defer link.free(acts_range) catch {};
    const dst_range = try link.alloc(@intCast(out.len), 64);
    defer link.free(dst_range) catch {};

    try link.upload(weights_range, packed_weights);
    try link.upload(acts_range, acts);
    try link.runGraph(&.{.{ .matmul_q1a8 = .{
        .weights = weights_range,
        .acts = acts_range,
        .dst = dst_range,
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
    } }});
    try link.download(dst_range, out);

    const expected = 127.0 * @as(f32, @floatFromInt(k));
    var max_abs_diff: f32 = 0;
    for (0..output_count) |i| {
        const got = readF32(out, i * @sizeOf(f32));
        max_abs_diff = @max(max_abs_diff, @abs(got - expected));
    }

    return .{
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
        .command_count = 1,
        .weights_nbytes = weights_nbytes,
        .acts_nbytes = acts_nbytes,
        .dst_nbytes = dst_nbytes,
        .expected = expected,
        .max_abs_diff = max_abs_diff,
    };
}

fn heapBytes(heap_mib: u32) RunError!usize {
    const mib = try validatePositive(heap_mib);
    const kib = try checkedMul(mib, 1024);
    return checkedMul(kib, 1024);
}

fn validatePositive(value: u32) RunError!usize {
    if (value == 0) return error.InvalidShape;
    return @intCast(value);
}

fn checkedMul(a: usize, b: usize) RunError!usize {
    return std.math.mul(usize, a, b) catch return error.InvalidShape;
}

fn writeF32(bytes: []u8, offset: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}

test "fake q1a8 matmul smoke succeeds" {
    const result = try runFakeMatmul(std.testing.allocator, .{});
    try std.testing.expectEqual(@as(u32, 8), result.rows);
    try std.testing.expectEqual(@as(u32, 1), result.cols);
    try std.testing.expectEqual(@as(f32, 127 * q1a8.q1_block), result.expected);
    try std.testing.expectEqual(@as(f32, 0), result.max_abs_diff);
}

test "fake q1a8 matmul rejects invalid k" {
    try std.testing.expectError(error.InvalidShape, runFakeMatmul(std.testing.allocator, .{ .k = 64 }));
}
