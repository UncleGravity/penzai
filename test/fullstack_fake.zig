const std = @import("std");
const build_options = @import("build_options");
const shared = @import("shared");
const runtime_mod = @import("runtime");
const link_mod = @import("link");

const layout = shared.layout;
const wire = shared.wire;
const profiling = shared.profiling;

test "fake link returns deployment and ABI capabilities" {
    const receipt = try shared.capabilities.Receipt.parse(
        "key\tvalue\n" ++
            "receipt_schema_version\t1\n" ++
            "manifest_schema_version\t1\n" ++
            "run_id\ttest-clean-run\n" ++
            "variant\tw512-p4-f300\n" ++
            "git_commit\tdeadbeef1234\n" ++
            "git_dirty\t0\n" ++
            "build_mode\tclean\n" ++
            "source_sha256\t1111\n" ++
            "manifest_sha256\t2222\n" ++
            "bitstream_sha256\t3333\n" ++
            "bitstream_hash_verified\t1\n",
    );
    var runtime = try runtime_mod.Runtime.initWithReceipt(std.testing.allocator, 1024 * 1024, receipt);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const report = try link.capabilities();
    try std.testing.expectEqual(shared.capabilities.ReceiptStatus.loaded, report.receipt_status);
    try std.testing.expectEqual(wire.version, report.wire_abi);
    try std.testing.expectEqual(profiling.version, report.profile_abi);
    try std.testing.expectEqualStrings("test-clean-run", report.run_id.slice());
    try std.testing.expectEqual(@as(u32, 0), report.engine_mask);
}

test "fake link alloc upload copy download" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const src = (try link.alloc(16, 64)).range;
    const dst = (try link.alloc(16, 64)).range;
    _ = try link.upload(src, "abcdefghijklmnop");
    try link.runGraph(&.{.{ .copy = .{ .src = src, .dst = dst } }});

    var out: [16]u8 = undefined;
    _ = try link.download(dst, &out);
    try std.testing.expectEqualSlices(u8, "abcdefghijklmnop", &out);
}

test "fake link run_graph preload applies before commands" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const src = (try link.alloc(16, 64)).range;
    const dst = (try link.alloc(16, 64)).range;

    var preload: [64]u8 = undefined;
    const preload_len = try wire.encodePreloadEntry(&preload, src, "preload-works-ok");
    try link.runGraphPreload(preload[0..preload_len], &.{.{ .copy = .{ .src = src, .dst = dst } }});

    var out: [16]u8 = undefined;
    _ = try link.download(dst, &out);
    try std.testing.expectEqualSlices(u8, "preload-works-ok", &out);
}

test "fake link fill download" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const range = (try link.alloc(16, 64)).range;
    _ = try link.fill(.{ .handle = range.handle, .offset = 4, .nbytes = 8 }, 0xab);

    var out: [16]u8 = undefined;
    _ = try link.download(range, &out);
    try std.testing.expectEqualSlices(u8, &([_]u8{0} ** 4), out[0..4]);
    try std.testing.expectEqualSlices(u8, &([_]u8{0xab} ** 8), out[4..12]);
    try std.testing.expectEqualSlices(u8, &([_]u8{0} ** 4), out[12..16]);
}

test "fake link q1a8 matmul reference" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const rows = layout.rows_per_block;
    const cols = 1;
    const k = layout.q1_block;
    const q1_blocks = comptime layout.blocksPerRow(k) catch unreachable;
    const logical_len = rows * q1_blocks;
    const packed_len = comptime layout.packedWeightBytes(rows, k) catch unreachable;

    var bits: [logical_len]u128 = [_]u128{std.math.maxInt(u128)} ** logical_len;
    var scales: [logical_len]f16 = [_]f16{1} ** logical_len;
    var packed_buf: [packed_len]u8 = undefined;
    try layout.packWeightsFromLogical(rows, k, &bits, &scales, &packed_buf);

    var acts: [k * @sizeOf(f32)]u8 = undefined;
    for (0..k) |i| writeF32(&acts, i * @sizeOf(f32), 127);

    const weights_range = (try link.alloc(packed_buf.len, 64)).range;
    const acts_range = (try link.alloc(acts.len, 64)).range;
    const dst_range = (try link.alloc(rows * cols * @sizeOf(f32), 64)).range;
    _ = try link.upload(weights_range, &packed_buf);
    _ = try link.upload(acts_range, &acts);

    try link.runGraph(&.{.{ .matmul_q1a8 = .{
        .weights = weights_range,
        .acts = acts_range,
        .dst = dst_range,
        .rows = @intCast(rows),
        .cols = @intCast(cols),
        .k = @intCast(k),
    } }});

    var out: [rows * @sizeOf(f32)]u8 = undefined;
    _ = try link.download(dst_range, &out);
    for (0..rows) |row| {
        try std.testing.expectEqual(@as(f32, 127 * layout.q1_block), readF32(&out, row * @sizeOf(f32)));
    }
}

test "fake link ps f32 command graph" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(std.testing.allocator, &runtime);

    const a = try allocTensor(&link, 2);
    const b = try allocTensor(&link, 2);
    const positions = (try link.alloc(@sizeOf(i32), @alignOf(i32))).range;
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
    var position_bytes: [4]u8 = undefined;
    writeI32(&position_bytes, 0, 1);
    _ = try link.upload(positions, &position_bytes);

    try link.runGraph(&.{
        .{ .rmsnorm = .{ .input = a, .dst = out_rms, .rows = 2, .cols = 1, .eps = 0 } },
        .{ .rope = .{
            .input = b,
            .positions = positions,
            .dst = out_rope,
            .head_dim = 2,
            .n_heads = 1,
            .n_tokens = 1,
            .n_dims = 2,
            .mode = .normal,
            .n_ctx_orig = 0,
            .freq_base = 10000,
            .freq_scale = 1,
            .ext_factor = 0,
            .attn_factor = 1,
            .beta_fast = 0,
            .beta_slow = 0,
        } },
        .{ .softmax = .{ .src = b, .dst = out_softmax } },
        .{ .silu = .{ .src = b, .dst = out_silu } },
        .{ .swiglu = .{ .lhs = b, .rhs = a, .dst = out_swiglu } },
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = out_add, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .mul_f32 = .{ .lhs = a, .rhs = b, .dst = out_mul, .rows = 2, .cols = 1, .mode = .same_shape } },
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

test "fake link runGraphProfile reports per-op aggregates" {
    if (!build_options.enable_profiling) return error.SkipZigTest;
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 1024 * 1024);
    defer runtime.deinit();
    var link = link_mod.FakeLink.initWithIo(std.testing.allocator, &runtime, std.testing.io);

    const a = try allocTensor(&link, 2);
    const b = try allocTensor(&link, 2);
    const out_add = try allocTensor(&link, 2);
    const out_scale = try allocTensor(&link, 2);
    try uploadTensor(&link, a, &.{ 3, 4 });
    try uploadTensor(&link, b, &.{ 1, 2 });

    const commands = [_]wire.Command{
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = out_add, .rows = 2, .cols = 1, .mode = .same_shape } },
        .{ .scale_f32 = .{ .src = b, .dst = out_scale, .scale = 0.5 } },
    };

    var profiled = try link.runGraphProfile(&commands, .aggregate);
    defer profiled.deinit(); // testing.allocator fails the test if the report leaks.

    try std.testing.expectEqual(@as(u32, 2), profiled.report.summary.command_count);
    try std.testing.expect(profiled.rpc.request_bytes > 0);
    try std.testing.expect(profiled.rpc.response_bytes > 0);
    try std.testing.expect(profiled.rpc.round_trip_ns >= profiled.rpc.deviceRpcNs());
    try std.testing.expect(profiled.rpc.device_service_ns >= profiled.report.summary.profile_span_ns);
    try std.testing.expect(profiled.report.summary.profile_span_ns >= profiled.report.summary.stagesNs());
    try std.testing.expectEqual(@as(u32, 0), profiled.report.summary.accounting_violations);

    const add = findAggregate(profiled.report, @intFromEnum(wire.OpTag.add_f32)) orelse return error.MissingAggregate;
    try std.testing.expectEqual(@as(u32, 1), add.count);
    try std.testing.expectEqual(a.nbytes + b.nbytes + out_add.nbytes, add.bytes);

    const scale = findAggregate(profiled.report, @intFromEnum(wire.OpTag.scale_f32)) orelse return error.MissingAggregate;
    try std.testing.expectEqual(@as(u32, 1), scale.count);
    try std.testing.expectEqual(b.nbytes + out_scale.nbytes, scale.bytes);
}

fn findAggregate(report: profiling.Report, tag: u16) ?profiling.Aggregate {
    for (report.aggregates) |aggregate| {
        if (aggregate.tag == tag) return aggregate;
    }
    return null;
}

fn allocTensor(link: *link_mod.FakeLink, len: usize) !wire.TensorRange {
    return (try link.alloc(len * @sizeOf(f32), @alignOf(f32))).range;
}

fn uploadTensor(link: *link_mod.FakeLink, range: wire.TensorRange, values: []const f32) !void {
    const bytes = try std.testing.allocator.alloc(u8, values.len * @sizeOf(f32));
    defer std.testing.allocator.free(bytes);
    for (values, 0..) |value, i| {
        writeF32(bytes, i * @sizeOf(f32), value);
    }
    _ = try link.upload(range, bytes);
}

fn expectTensor(link: *link_mod.FakeLink, range: wire.TensorRange, expected: []const f32) !void {
    const bytes = try std.testing.allocator.alloc(u8, expected.len * @sizeOf(f32));
    defer std.testing.allocator.free(bytes);
    _ = try link.download(range, bytes);
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

fn writeI32(bytes: []u8, offset: usize, value: i32) void {
    std.mem.writeInt(i32, bytes[offset..][0..4], value, .little);
}

fn readF32(bytes: []const u8, offset: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[offset..][0..4], .little));
}
