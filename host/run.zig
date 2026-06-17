const std = @import("std");
const build_options = @import("build_options");
const shared = @import("shared");
const prof_report = @import("prof_report.zig");
const runtime_mod = @import("runtime");
const link_mod = @import("link");
const llama_mod = if (build_options.enable_llama) @import("llama.zig") else struct {
    pub const Error = error{};
    pub const Options = struct {};
};

const q1a8 = shared.q1a8;
const protocol_transport = shared.protocol_transport;
const profiling = shared.profiling;
const wire = shared.wire;

pub const RunError = error{
    InvalidShape,
    OutOfMemory,
    Protocol,
    RemoteFailed,
    Transport,
    LlamaDisabled,
} || llama_mod.Error;

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

pub const BenchOptions = struct {
    rows: u32 = @intCast(q1a8.rows_per_block),
    cols: u32 = 1,
    k: u32 = @intCast(q1a8.q1_block),
    heap_mib: u32 = 16,
    warmup: u32 = 3,
    iters: u32 = 30,
    profile: bool = false,
};

pub const BenchResult = struct {
    rows: u32,
    cols: u32,
    k: u32,
    warmup: u32,
    iters: u32,
    profiled: bool,
    weights_nbytes: usize,
    acts_nbytes: usize,
    dst_nbytes: usize,
    host_total_ns: u64,
    host_min_ns: u64,
    host_max_ns: u64,
    expected: f32,
    max_abs_diff: f32,
    profile: BenchProfile = .{},
};

/// Bench run_graph profiling is exactly the shared cross-call rollup.
pub const BenchProfile = prof_report.RunGraphTotals;

pub const LlamaOptions = struct {
    model_path: []const u8 = build_options.default_model_path,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 16,
    heap_mib: u32 = 768,
    census: bool = false,
    logits_tolerance: f32 = 0.25,
    chat_template: bool = true,
    enable_thinking: bool = false,
    profile: bool = false,
    device_label: []const u8 = "fake",
    backend_sampling: bool = false,
};

pub fn runFakeLlama(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var runtime = try runtime_mod.Runtime.init(allocator, try heapBytes(options.heap_mib));
    defer runtime.deinit();
    var link = link_mod.FakeLink.initWithIo(allocator, &runtime, io);
    const client = link_mod.Client.init(&link);
    return llama_mod.runPrompt(io, allocator, writer, client, .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .max_tokens = options.max_tokens,
        .census = options.census,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
        .profile = options.profile,
        .device_label = options.device_label,
        .backend_sampling = options.backend_sampling,
    });
}

pub fn runTcpLlama(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    spec: protocol_transport.TcpSpec,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    const client = link_mod.Client.init(&link);
    return llama_mod.runPrompt(io, allocator, writer, client, .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .max_tokens = options.max_tokens,
        .census = options.census,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
        .profile = options.profile,
        .device_label = options.device_label,
        .backend_sampling = options.backend_sampling,
    });
}

pub fn runFakeLogitsCheck(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var runtime = try runtime_mod.Runtime.init(allocator, try heapBytes(options.heap_mib));
    defer runtime.deinit();
    var link = link_mod.FakeLink.init(allocator, &runtime);
    const client = link_mod.Client.init(&link);
    return llama_mod.runLogitsCheck(allocator, writer, client, .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .max_tokens = options.max_tokens,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
    });
}

pub fn runTcpLogitsCheck(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    spec: protocol_transport.TcpSpec,
    options: LlamaOptions,
) RunError!void {
    if (!build_options.enable_llama) return error.LlamaDisabled;
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    const client = link_mod.Client.init(&link);
    return llama_mod.runLogitsCheck(allocator, writer, client, .{
        .model_path = options.model_path,
        .prompt = options.prompt,
        .max_tokens = options.max_tokens,
        .logits_tolerance = options.logits_tolerance,
        .chat_template = options.chat_template,
        .enable_thinking = options.enable_thinking,
    });
}

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

pub fn benchFakeMatmulQ1A8(io: std.Io, allocator: std.mem.Allocator, options: BenchOptions) RunError!BenchResult {
    var runtime = try runtime_mod.Runtime.init(allocator, try heapBytes(options.heap_mib));
    defer runtime.deinit();
    var link = link_mod.FakeLink.initWithIo(allocator, &runtime, io);
    return benchMatmulQ1A8WithLink(io, allocator, options, &link);
}

pub fn benchTcpMatmulQ1A8(
    io: std.Io,
    allocator: std.mem.Allocator,
    spec: protocol_transport.TcpSpec,
    options: BenchOptions,
) RunError!BenchResult {
    var link = try link_mod.TcpLink.connect(allocator, io, spec);
    defer link.deinit();
    return benchMatmulQ1A8WithLink(io, allocator, options, &link);
}

fn runMatmulWithLink(allocator: std.mem.Allocator, options: RunOptions, link: anytype) RunError!RunResult {
    var fixture = try MatmulFixture.init(allocator, options.rows, options.cols, options.k);
    defer fixture.deinit(allocator);

    try link.hello();

    const weights_range = (try link.alloc(@intCast(fixture.packed_weights.len), 64)).range;
    defer freeQuietly(link, weights_range);
    const acts_range = (try link.alloc(@intCast(fixture.acts.len), 64)).range;
    defer freeQuietly(link, acts_range);
    const dst_range = (try link.alloc(@intCast(fixture.out.len), 64)).range;
    defer freeQuietly(link, dst_range);

    _ = try link.upload(weights_range, fixture.packed_weights);
    _ = try link.upload(acts_range, fixture.acts);
    try link.runGraph(&.{.{ .matmul_q1a8 = .{
        .weights = weights_range,
        .acts = acts_range,
        .dst = dst_range,
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
    } }});
    _ = try link.download(dst_range, fixture.out);
    const max_abs_diff = fixture.maxAbsDiff();

    return .{
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
        .command_count = 1,
        .weights_nbytes = fixture.packed_weights.len,
        .acts_nbytes = fixture.acts.len,
        .dst_nbytes = fixture.out.len,
        .expected = fixture.expected,
        .max_abs_diff = max_abs_diff,
    };
}

fn benchMatmulQ1A8WithLink(
    io: std.Io,
    allocator: std.mem.Allocator,
    options: BenchOptions,
    link: anytype,
) RunError!BenchResult {
    const warmup: usize = @intCast(options.warmup);
    const iters = try validatePositive(options.iters);
    var fixture = try MatmulFixture.init(allocator, options.rows, options.cols, options.k);
    defer fixture.deinit(allocator);

    try link.hello();

    const weights_range = (try link.alloc(@intCast(fixture.packed_weights.len), 64)).range;
    defer freeQuietly(link, weights_range);
    const acts_range = (try link.alloc(@intCast(fixture.acts.len), 64)).range;
    defer freeQuietly(link, acts_range);
    const dst_range = (try link.alloc(@intCast(fixture.out.len), 64)).range;
    defer freeQuietly(link, dst_range);

    _ = try link.upload(weights_range, fixture.packed_weights);
    _ = try link.upload(acts_range, fixture.acts);

    const commands = [_]wire.Command{.{ .matmul_q1a8 = .{
        .weights = weights_range,
        .acts = acts_range,
        .dst = dst_range,
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
    } }};

    for (0..warmup) |_| try link.runGraph(&commands);

    var result = BenchResult{
        .rows = options.rows,
        .cols = options.cols,
        .k = options.k,
        .warmup = options.warmup,
        .iters = options.iters,
        .profiled = options.profile,
        .weights_nbytes = fixture.packed_weights.len,
        .acts_nbytes = fixture.acts.len,
        .dst_nbytes = fixture.out.len,
        .host_total_ns = 0,
        .host_min_ns = std.math.maxInt(u64),
        .host_max_ns = 0,
        .expected = fixture.expected,
        .max_abs_diff = 0,
    };

    for (0..iters) |_| {
        const start_ns = profiling.nowNs(io);
        if (options.profile) {
            var profiled = try link.runGraphProfile(&commands, .aggregate);
            const end_ns = profiling.nowNs(io);
            result.profile.record(profiled);
            profiled.deinit();
            recordIteration(&result, profiling.elapsed(start_ns, end_ns));
        } else {
            try link.runGraph(&commands);
            const end_ns = profiling.nowNs(io);
            recordIteration(&result, profiling.elapsed(start_ns, end_ns));
        }
    }

    _ = try link.download(dst_range, fixture.out);
    result.max_abs_diff = fixture.maxAbsDiff();
    return result;
}

const MatmulFixture = struct {
    rows: usize,
    cols: usize,
    k: usize,
    packed_weights: []u8,
    acts: []u8,
    out: []u8,
    expected: f32,

    fn init(allocator: std.mem.Allocator, rows_value: u32, cols_value: u32, k_value: u32) RunError!MatmulFixture {
        const rows = try validatePositive(rows_value);
        const cols = try validatePositive(cols_value);
        const k = try validatePositive(k_value);

        const q1_blocks = q1a8.blocksPerRow(k) catch return error.InvalidShape;
        const logical_len = try checkedMul(rows, q1_blocks);
        const act_count = try checkedMul(cols, k);
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
        errdefer allocator.free(packed_weights);
        q1a8.packWeightsFromLogical(rows, k, weight_bits, weight_scales, packed_weights) catch return error.InvalidShape;

        const acts = try allocator.alloc(u8, acts_nbytes);
        errdefer allocator.free(acts);
        for (0..act_count) |i| writeF32(acts, i * @sizeOf(f32), 127);

        const out = try allocator.alloc(u8, dst_nbytes);
        errdefer allocator.free(out);
        @memset(out, 0);

        return .{
            .rows = rows,
            .cols = cols,
            .k = k,
            .packed_weights = packed_weights,
            .acts = acts,
            .out = out,
            .expected = 127.0 * @as(f32, @floatFromInt(k)),
        };
    }

    fn deinit(self: *MatmulFixture, allocator: std.mem.Allocator) void {
        allocator.free(self.packed_weights);
        allocator.free(self.acts);
        allocator.free(self.out);
        self.* = undefined;
    }

    fn maxAbsDiff(self: *const MatmulFixture) f32 {
        var max_abs_diff: f32 = 0;
        for (0..self.rows * self.cols) |i| {
            const got = readF32(self.out, i * @sizeOf(f32));
            max_abs_diff = @max(max_abs_diff, @abs(got - self.expected));
        }
        return max_abs_diff;
    }
};

/// Best-effort cleanup free: this is the smoke/bench harness, so a failed free
/// on teardown is not actionable. Discards both the error and the op timing.
fn freeQuietly(link: anytype, range: wire.TensorRange) void {
    _ = link.free(range) catch return;
}

fn recordIteration(self: *BenchResult, ns: u64) void {
    self.host_total_ns += ns;
    self.host_min_ns = @min(self.host_min_ns, ns);
    self.host_max_ns = @max(self.host_max_ns, ns);
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
    try std.testing.expectEqual(@as(u32, @intCast(q1a8.rows_per_block)), result.rows);
    try std.testing.expectEqual(@as(u32, 1), result.cols);
    try std.testing.expectEqual(@as(f32, 127 * q1a8.q1_block), result.expected);
    try std.testing.expectEqual(@as(f32, 0), result.max_abs_diff);
}

test "fake q1a8 matmul rejects invalid k" {
    try std.testing.expectError(error.InvalidShape, runFakeMatmul(std.testing.allocator, .{ .k = 64 }));
}

test "fake q1a8 matmul bench succeeds" {
    const result = try benchFakeMatmulQ1A8(std.testing.io, std.testing.allocator, .{ .warmup = 1, .iters = 2 });
    try std.testing.expectEqual(@as(u32, @intCast(q1a8.rows_per_block)), result.rows);
    try std.testing.expectEqual(@as(u32, 2), result.iters);
    try std.testing.expectEqual(@as(f32, 0), result.max_abs_diff);
    try std.testing.expect(result.host_total_ns > 0);
}
