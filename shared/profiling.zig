const std = @import("std");

pub const version: u16 = 6;
pub const max_op_tag = 64;
pub const max_matmul_buckets = 32;
pub const max_flash_buckets = 16;

const magic: u32 = 0x5046_5250; // "PRFP", little-endian on the wire.
const header_len: usize = 104;
const aggregate_len: usize = 24;
const matmul_stat_len: usize = 160;
const flash_stat_len: usize = 256;

pub const AccountingViolation = struct {
    pub const wrapper_segments: u32 = 1 << 0;
    pub const command_children: u32 = 1 << 1;
    pub const device_stages: u32 = 1 << 2;
    pub const rpc_budget: u32 = 1 << 3;
};

pub const Backend = enum(u8) {
    ps = 1,
    pl = 2,
};

pub const ExecutionPath = enum(u8) {
    software = 1,
    direct = 2,
    staged = 3,
};

pub const DecodeError = error{
    Truncated,
    BadMagic,
    UnsupportedVersion,
    InvalidLength,
    InvalidEnum,
    OutOfMemory,
};

/// The profile payload subdivides device service time. The response metadata,
/// not this summary, remains authoritative for the complete RPC device time.
pub const Summary = struct {
    /// Covered portion of device service, from request decoding through profile
    /// serialization. ResponseMeta.device_service_ns is the authoritative parent.
    profile_span_ns: u64 = 0,
    request_decode_ns: u64 = 0,
    preload_ns: u64 = 0,
    command_decode_ns: u64 = 0,
    execute_ns: u64 = 0,
    profile_encode_ns: u64 = 0,
    preload_bytes: u64 = 0,
    command_bytes: u64 = 0,
    command_count: u32 = 0,
    device_fclk_hz: u32 = 0,
    accounting_violations: u32 = 0,
    matmul_bucket_overflow: u32 = 0,
    flash_bucket_overflow: u32 = 0,

    pub fn stagesNs(self: Summary) u64 {
        return self.request_decode_ns +| self.preload_ns +| self.command_decode_ns +| self.execute_ns +| self.profile_encode_ns;
    }
};

pub const Aggregate = struct {
    tag: u16 = 0,
    count: u32 = 0,
    total_ns: u64 = 0,
    bytes: u64 = 0,
};

/// One bounded bucket of like-shaped GEMM commands. Command time belongs to the
/// runtime; wrapper stages and hardware counters belong to the selected PL path.
pub const MatmulStat = struct {
    backend: Backend = .ps,
    path: ExecutionPath = .software,
    fmt: u16 = 0,
    rows: u32 = 0,
    cols: u32 = 0,
    k: u32 = 0,
    count: u32 = 0,
    kernel_runs: u32 = 0,
    macs: u64 = 0,
    command_ns: u64 = 0,
    wrapper_ns: u64 = 0,
    quantize_pack_ns: u64 = 0,
    sync_to_ns: u64 = 0,
    setup_ns: u64 = 0,
    wait_ns: u64 = 0,
    sync_from_ns: u64 = 0,
    result_layout_ns: u64 = 0,
    cycles: u64 = 0,
    w_stall_cycles: u64 = 0,
    a_stall_cycles: u64 = 0,
    r_stall_cycles: u64 = 0,
    w_beats: u64 = 0,
    a_beats: u64 = 0,
    r_beats: u64 = 0,

    pub fn sameKey(a: MatmulStat, b: MatmulStat) bool {
        return a.backend == b.backend and a.path == b.path and a.fmt == b.fmt and
            a.rows == b.rows and a.cols == b.cols and a.k == b.k;
    }

    pub fn wrapperChildrenNs(self: MatmulStat) u64 {
        return self.quantize_pack_ns +| self.sync_to_ns +| self.setup_ns +|
            self.wait_ns +| self.sync_from_ns +| self.result_layout_ns;
    }
};

/// One bounded bucket of like-shaped flash-attention commands. KV extents report
/// loop bounds; query-KV pairs separately report padded work, mask entries other
/// than f16 -inf, and the work actually walked by the selected backend. Pair counts
/// do not include query heads; use the query-head/KV helpers for engine work.
pub const FlashStat = struct {
    backend: Backend = .ps,
    path: ExecutionPath = .software,
    n_heads: u16 = 0,
    n_head_kv: u16 = 0,
    head_dim_q: u16 = 0,
    head_dim_v: u16 = 0,
    n_tokens: u32 = 0,
    count: u32 = 0,
    kernel_runs: u32 = 0,
    command_ns: u64 = 0,
    wrapper_ns: u64 = 0,
    prepare_ns: u64 = 0,
    sync_to_ns: u64 = 0,
    setup_ns: u64 = 0,
    wait_ns: u64 = 0,
    sync_from_ns: u64 = 0,
    result_layout_ns: u64 = 0,
    requested_n_kv_sum: u64 = 0,
    valid_n_kv_sum: u64 = 0,
    processed_n_kv_sum: u64 = 0,
    requested_qkv_pairs: u64 = 0,
    valid_qkv_pairs: u64 = 0,
    processed_qkv_pairs: u64 = 0,
    q_bytes: u64 = 0,
    k_bytes: u64 = 0,
    v_bytes: u64 = 0,
    mask_bytes: u64 = 0,
    o_bytes: u64 = 0,
    cycles: u64 = 0,
    q_beats: u64 = 0,
    k_beats: u64 = 0,
    k_stall_cycles: u64 = 0,
    v_beats: u64 = 0,
    v_stall_cycles: u64 = 0,
    o_beats: u64 = 0,
    o_stall_cycles: u64 = 0,
    requested_n_kv_max: u32 = 0,
    valid_n_kv_max: u32 = 0,
    processed_n_kv_max: u32 = 0,

    pub fn sameKey(a: FlashStat, b: FlashStat) bool {
        return a.backend == b.backend and a.path == b.path and
            a.n_heads == b.n_heads and a.n_head_kv == b.n_head_kv and
            a.head_dim_q == b.head_dim_q and a.head_dim_v == b.head_dim_v and
            a.n_tokens == b.n_tokens;
    }

    pub fn wrapperChildrenNs(self: FlashStat) u64 {
        return self.prepare_ns +| self.sync_to_ns +| self.setup_ns +| self.wait_ns +|
            self.sync_from_ns +| self.result_layout_ns;
    }

    pub fn validQueryHeadKvUpdates(self: FlashStat) u64 {
        return self.valid_qkv_pairs *| @as(u64, self.n_heads);
    }

    pub fn processedQueryHeadKvUpdates(self: FlashStat) u64 {
        return self.processed_qkv_pairs *| @as(u64, self.n_heads);
    }

    pub fn cyclesPerValidUpdate(self: FlashStat) ?f64 {
        const updates = self.validQueryHeadKvUpdates();
        if (self.cycles == 0 or updates == 0) return null;
        return @as(f64, @floatFromInt(self.cycles)) / @as(f64, @floatFromInt(updates));
    }

    pub fn cyclesPerProcessedUpdate(self: FlashStat) ?f64 {
        const updates = self.processedQueryHeadKvUpdates();
        if (self.cycles == 0 or updates == 0) return null;
        return @as(f64, @floatFromInt(self.cycles)) / @as(f64, @floatFromInt(updates));
    }
};

pub const ReportView = struct {
    summary: Summary,
    aggregates: []const Aggregate = &.{},
    matmul_stats: []const MatmulStat = &.{},
    flash_stats: []const FlashStat = &.{},
};

pub const Report = struct {
    summary: Summary,
    aggregates: []Aggregate = &.{},
    matmul_stats: []MatmulStat = &.{},
    flash_stats: []FlashStat = &.{},

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.aggregates);
        allocator.free(self.matmul_stats);
        allocator.free(self.flash_stats);
        self.* = undefined;
    }
};

pub fn nowNs(io: ?std.Io) u64 {
    const active = io orelse return 0;
    const ns = std.Io.Timestamp.now(active, .awake).nanoseconds;
    return std.math.cast(u64, ns) orelse 0;
}

pub fn elapsed(start_ns: u64, end_ns: u64) u64 {
    return if (end_ns >= start_ns) end_ns - start_ns else 0;
}

pub fn encodedLen(report: ReportView) usize {
    return header_len + report.aggregates.len * aggregate_len +
        report.matmul_stats.len * matmul_stat_len + report.flash_stats.len * flash_stat_len;
}

pub fn encode(report: ReportView, out: []u8) error{ OutputTooSmall, InvalidLayout }!usize {
    const want = encodedLen(report);
    if (out.len < want) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU32(out, &cursor, magic);
    putU16(out, &cursor, version);
    putU16(out, &cursor, 0);
    putSummary(out, &cursor, report.summary);
    putU32(out, &cursor, @intCast(report.aggregates.len));
    putU32(out, &cursor, @intCast(report.matmul_stats.len));
    putU32(out, &cursor, @intCast(report.flash_stats.len));

    for (report.aggregates) |aggregate| {
        putU16(out, &cursor, aggregate.tag);
        putU16(out, &cursor, 0);
        putU32(out, &cursor, aggregate.count);
        putU64(out, &cursor, aggregate.total_ns);
        putU64(out, &cursor, aggregate.bytes);
    }
    for (report.matmul_stats) |stat| putMatmul(out, &cursor, stat);
    for (report.flash_stats) |stat| putFlash(out, &cursor, stat);
    if (cursor != want) return error.InvalidLayout;
    return cursor;
}

/// Patch fields only known after profile serialization without re-encoding the
/// bounded bucket payload.
pub fn patchProfileSpan(out: []u8, profile_span_ns: u64, profile_encode_ns: u64, extra_violations: u32) DecodeError!void {
    if (out.len < header_len) return error.Truncated;
    if (std.mem.readInt(u32, out[0..4], .little) != magic) return error.BadMagic;
    if (std.mem.readInt(u16, out[4..6], .little) != version) return error.UnsupportedVersion;
    std.mem.writeInt(u64, out[8..16], profile_span_ns, .little);
    std.mem.writeInt(u64, out[48..56], profile_encode_ns, .little);
    const old = std.mem.readInt(u32, out[80..84], .little);
    std.mem.writeInt(u32, out[80..84], old | extra_violations, .little);
}

pub fn encodeAlloc(allocator: std.mem.Allocator, report: ReportView) std.mem.Allocator.Error![]u8 {
    const out = try allocator.alloc(u8, encodedLen(report));
    errdefer allocator.free(out);
    _ = encode(report, out) catch unreachable;
    return out;
}

pub fn decodeAlloc(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!Report {
    if (bytes.len < header_len) return error.Truncated;
    var cursor: usize = 0;
    if (try takeU32(bytes, &cursor) != magic) return error.BadMagic;
    if (try takeU16(bytes, &cursor) != version) return error.UnsupportedVersion;
    _ = try takeU16(bytes, &cursor);
    const summary = try takeSummary(bytes, &cursor);
    const aggregate_count = try takeU32(bytes, &cursor);
    const matmul_stat_count = try takeU32(bytes, &cursor);
    const flash_stat_count = try takeU32(bytes, &cursor);

    const aggregate_bytes = std.math.mul(usize, @intCast(aggregate_count), aggregate_len) catch return error.InvalidLength;
    const matmul_bytes = std.math.mul(usize, @intCast(matmul_stat_count), matmul_stat_len) catch return error.InvalidLength;
    const flash_bytes = std.math.mul(usize, @intCast(flash_stat_count), flash_stat_len) catch return error.InvalidLength;
    var total = std.math.add(usize, header_len, aggregate_bytes) catch return error.InvalidLength;
    total = std.math.add(usize, total, matmul_bytes) catch return error.InvalidLength;
    total = std.math.add(usize, total, flash_bytes) catch return error.InvalidLength;
    if (bytes.len != total) return error.InvalidLength;

    const aggregates = allocator.alloc(Aggregate, @intCast(aggregate_count)) catch return error.OutOfMemory;
    errdefer allocator.free(aggregates);
    const matmul_stats = allocator.alloc(MatmulStat, @intCast(matmul_stat_count)) catch return error.OutOfMemory;
    errdefer allocator.free(matmul_stats);
    const flash_stats = allocator.alloc(FlashStat, @intCast(flash_stat_count)) catch return error.OutOfMemory;
    errdefer allocator.free(flash_stats);

    for (aggregates) |*aggregate| {
        aggregate.* = .{
            .tag = try takeU16(bytes, &cursor),
            .count = blk: {
                _ = try takeU16(bytes, &cursor);
                break :blk try takeU32(bytes, &cursor);
            },
            .total_ns = try takeU64(bytes, &cursor),
            .bytes = try takeU64(bytes, &cursor),
        };
    }
    for (matmul_stats) |*stat| stat.* = try takeMatmul(bytes, &cursor);
    for (flash_stats) |*stat| stat.* = try takeFlash(bytes, &cursor);
    if (cursor != bytes.len) return error.InvalidLength;

    return .{ .summary = summary, .aggregates = aggregates, .matmul_stats = matmul_stats, .flash_stats = flash_stats };
}

fn putSummary(out: []u8, cursor: *usize, s: Summary) void {
    putU64(out, cursor, s.profile_span_ns);
    putU64(out, cursor, s.request_decode_ns);
    putU64(out, cursor, s.preload_ns);
    putU64(out, cursor, s.command_decode_ns);
    putU64(out, cursor, s.execute_ns);
    putU64(out, cursor, s.profile_encode_ns);
    putU64(out, cursor, s.preload_bytes);
    putU64(out, cursor, s.command_bytes);
    putU32(out, cursor, s.command_count);
    putU32(out, cursor, s.device_fclk_hz);
    putU32(out, cursor, s.accounting_violations);
    putU32(out, cursor, s.matmul_bucket_overflow);
    putU32(out, cursor, s.flash_bucket_overflow);
}

fn takeSummary(bytes: []const u8, cursor: *usize) DecodeError!Summary {
    return .{
        .profile_span_ns = try takeU64(bytes, cursor),
        .request_decode_ns = try takeU64(bytes, cursor),
        .preload_ns = try takeU64(bytes, cursor),
        .command_decode_ns = try takeU64(bytes, cursor),
        .execute_ns = try takeU64(bytes, cursor),
        .profile_encode_ns = try takeU64(bytes, cursor),
        .preload_bytes = try takeU64(bytes, cursor),
        .command_bytes = try takeU64(bytes, cursor),
        .command_count = try takeU32(bytes, cursor),
        .device_fclk_hz = try takeU32(bytes, cursor),
        .accounting_violations = try takeU32(bytes, cursor),
        .matmul_bucket_overflow = try takeU32(bytes, cursor),
        .flash_bucket_overflow = try takeU32(bytes, cursor),
    };
}

fn putMatmul(out: []u8, cursor: *usize, s: MatmulStat) void {
    putU8(out, cursor, @intFromEnum(s.backend));
    putU8(out, cursor, @intFromEnum(s.path));
    putU16(out, cursor, s.fmt);
    putU32(out, cursor, s.rows);
    putU32(out, cursor, s.cols);
    putU32(out, cursor, s.k);
    putU32(out, cursor, s.count);
    putU32(out, cursor, s.kernel_runs);
    putU32(out, cursor, 0);
    putU32(out, cursor, 0);
    inline for (.{ s.macs, s.command_ns, s.wrapper_ns, s.quantize_pack_ns, s.sync_to_ns, s.setup_ns, s.wait_ns, s.sync_from_ns, s.result_layout_ns, s.cycles, s.w_stall_cycles, s.a_stall_cycles, s.r_stall_cycles, s.w_beats, s.a_beats, s.r_beats }) |value| putU64(out, cursor, value);
}

fn takeMatmul(bytes: []const u8, cursor: *usize) DecodeError!MatmulStat {
    const backend = try takeBackend(bytes, cursor);
    const path = try takePath(bytes, cursor);
    var s = MatmulStat{ .backend = backend, .path = path, .fmt = try takeU16(bytes, cursor), .rows = try takeU32(bytes, cursor), .cols = try takeU32(bytes, cursor), .k = try takeU32(bytes, cursor), .count = try takeU32(bytes, cursor), .kernel_runs = try takeU32(bytes, cursor) };
    _ = try takeU32(bytes, cursor);
    _ = try takeU32(bytes, cursor);
    inline for (.{ "macs", "command_ns", "wrapper_ns", "quantize_pack_ns", "sync_to_ns", "setup_ns", "wait_ns", "sync_from_ns", "result_layout_ns", "cycles", "w_stall_cycles", "a_stall_cycles", "r_stall_cycles", "w_beats", "a_beats", "r_beats" }) |field| @field(s, field) = try takeU64(bytes, cursor);
    return s;
}

fn putFlash(out: []u8, cursor: *usize, s: FlashStat) void {
    putU8(out, cursor, @intFromEnum(s.backend));
    putU8(out, cursor, @intFromEnum(s.path));
    putU16(out, cursor, s.n_heads);
    putU16(out, cursor, s.n_head_kv);
    putU16(out, cursor, s.head_dim_q);
    putU16(out, cursor, s.head_dim_v);
    putU16(out, cursor, 0);
    putU32(out, cursor, s.count);
    putU32(out, cursor, s.kernel_runs);
    putU32(out, cursor, s.n_tokens);
    inline for (.{ s.command_ns, s.wrapper_ns, s.prepare_ns, s.sync_to_ns, s.setup_ns, s.wait_ns, s.sync_from_ns, s.result_layout_ns, s.requested_n_kv_sum, s.valid_n_kv_sum, s.processed_n_kv_sum, s.requested_qkv_pairs, s.valid_qkv_pairs, s.processed_qkv_pairs, s.q_bytes, s.k_bytes, s.v_bytes, s.mask_bytes, s.o_bytes, s.cycles, s.q_beats, s.k_beats, s.k_stall_cycles, s.v_beats, s.v_stall_cycles, s.o_beats, s.o_stall_cycles }) |value| putU64(out, cursor, value);
    putU32(out, cursor, s.requested_n_kv_max);
    putU32(out, cursor, s.valid_n_kv_max);
    putU32(out, cursor, s.processed_n_kv_max);
    putU32(out, cursor, 0);
}

fn takeFlash(bytes: []const u8, cursor: *usize) DecodeError!FlashStat {
    const backend = try takeBackend(bytes, cursor);
    const path = try takePath(bytes, cursor);
    var s = FlashStat{ .backend = backend, .path = path, .n_heads = try takeU16(bytes, cursor), .n_head_kv = try takeU16(bytes, cursor), .head_dim_q = try takeU16(bytes, cursor), .head_dim_v = try takeU16(bytes, cursor) };
    _ = try takeU16(bytes, cursor);
    s.count = try takeU32(bytes, cursor);
    s.kernel_runs = try takeU32(bytes, cursor);
    s.n_tokens = try takeU32(bytes, cursor);
    inline for (.{ "command_ns", "wrapper_ns", "prepare_ns", "sync_to_ns", "setup_ns", "wait_ns", "sync_from_ns", "result_layout_ns", "requested_n_kv_sum", "valid_n_kv_sum", "processed_n_kv_sum", "requested_qkv_pairs", "valid_qkv_pairs", "processed_qkv_pairs", "q_bytes", "k_bytes", "v_bytes", "mask_bytes", "o_bytes", "cycles", "q_beats", "k_beats", "k_stall_cycles", "v_beats", "v_stall_cycles", "o_beats", "o_stall_cycles" }) |field| @field(s, field) = try takeU64(bytes, cursor);
    s.requested_n_kv_max = try takeU32(bytes, cursor);
    s.valid_n_kv_max = try takeU32(bytes, cursor);
    s.processed_n_kv_max = try takeU32(bytes, cursor);
    _ = try takeU32(bytes, cursor);
    return s;
}

fn putU8(out: []u8, cursor: *usize, value: u8) void {
    out[cursor.*] = value;
    cursor.* += 1;
}

fn putU16(out: []u8, cursor: *usize, value: u16) void {
    std.mem.writeInt(u16, out[cursor.*..][0..2], value, .little);
    cursor.* += 2;
}

fn putU32(out: []u8, cursor: *usize, value: u32) void {
    std.mem.writeInt(u32, out[cursor.*..][0..4], value, .little);
    cursor.* += 4;
}

fn putU64(out: []u8, cursor: *usize, value: u64) void {
    std.mem.writeInt(u64, out[cursor.*..][0..8], value, .little);
    cursor.* += 8;
}

fn takeU8(bytes: []const u8, cursor: *usize) DecodeError!u8 {
    if (cursor.* >= bytes.len) return error.Truncated;
    defer cursor.* += 1;
    return bytes[cursor.*];
}

fn takeBackend(bytes: []const u8, cursor: *usize) DecodeError!Backend {
    return switch (try takeU8(bytes, cursor)) {
        @intFromEnum(Backend.ps) => .ps,
        @intFromEnum(Backend.pl) => .pl,
        else => error.InvalidEnum,
    };
}

fn takePath(bytes: []const u8, cursor: *usize) DecodeError!ExecutionPath {
    return switch (try takeU8(bytes, cursor)) {
        @intFromEnum(ExecutionPath.software) => .software,
        @intFromEnum(ExecutionPath.direct) => .direct,
        @intFromEnum(ExecutionPath.staged) => .staged,
        else => error.InvalidEnum,
    };
}

fn takeU16(bytes: []const u8, cursor: *usize) DecodeError!u16 {
    if (cursor.* + 2 > bytes.len) return error.Truncated;
    defer cursor.* += 2;
    return std.mem.readInt(u16, bytes[cursor.*..][0..2], .little);
}

fn takeU32(bytes: []const u8, cursor: *usize) DecodeError!u32 {
    if (cursor.* + 4 > bytes.len) return error.Truncated;
    defer cursor.* += 4;
    return std.mem.readInt(u32, bytes[cursor.*..][0..4], .little);
}

fn takeU64(bytes: []const u8, cursor: *usize) DecodeError!u64 {
    if (cursor.* + 8 > bytes.len) return error.Truncated;
    defer cursor.* += 8;
    return std.mem.readInt(u64, bytes[cursor.*..][0..8], .little);
}

test "profile report roundtrips" {
    const aggregates = [_]Aggregate{.{ .tag = 2, .count = 3, .total_ns = 99, .bytes = 1234 }};
    const matmul_stats = [_]MatmulStat{.{ .backend = .pl, .path = .direct, .fmt = 1, .rows = 2, .cols = 3, .k = 4, .count = 5, .kernel_runs = 5, .macs = 9000, .command_ns = 42, .wrapper_ns = 40, .wait_ns = 30, .cycles = 1000, .w_stall_cycles = 100, .r_beats = 16 }};
    const flash_stats = [_]FlashStat{.{ .backend = .pl, .path = .staged, .count = 28, .kernel_runs = 28, .n_heads = 16, .n_head_kv = 8, .head_dim_q = 128, .head_dim_v = 128, .n_tokens = 8, .requested_n_kv_sum = 1024, .valid_n_kv_sum = 784, .processed_n_kv_sum = 784, .requested_qkv_pairs = 8192, .valid_qkv_pairs = 3136, .processed_qkv_pairs = 3136, .requested_n_kv_max = 64, .valid_n_kv_max = 54, .processed_n_kv_max = 54, .command_ns = 56000, .wrapper_ns = 55000, .q_bytes = 1234, .cycles = 9000 }};
    const view = ReportView{
        .summary = .{ .profile_span_ns = 100, .request_decode_ns = 1, .preload_ns = 2, .command_decode_ns = 6, .execute_ns = 80, .profile_encode_ns = 11, .preload_bytes = 64, .command_bytes = 256, .command_count = 3, .device_fclk_hz = 250_000_000, .matmul_bucket_overflow = 2 },
        .aggregates = &aggregates,
        .matmul_stats = &matmul_stats,
        .flash_stats = &flash_stats,
    };
    const encoded = try encodeAlloc(std.testing.allocator, view);
    defer std.testing.allocator.free(encoded);
    try std.testing.expectEqual(encodedLen(view), encoded.len);
    var decoded = try decodeAlloc(std.testing.allocator, encoded);
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqualDeep(view.summary, decoded.summary);
    try std.testing.expectEqualDeep(matmul_stats[0], decoded.matmul_stats[0]);
    try std.testing.expectEqualDeep(flash_stats[0], decoded.flash_stats[0]);
}

test "flash efficiency counts query-head KV updates and handles empty masks" {
    const stat = FlashStat{
        .n_heads = 16,
        .valid_qkv_pairs = 100,
        .processed_qkv_pairs = 125,
        .cycles = 24_000,
    };
    try std.testing.expectEqual(@as(u64, 1600), stat.validQueryHeadKvUpdates());
    try std.testing.expectEqual(@as(u64, 2000), stat.processedQueryHeadKvUpdates());
    try std.testing.expectApproxEqAbs(@as(f64, 15), stat.cyclesPerValidUpdate().?, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 12), stat.cyclesPerProcessedUpdate().?, 0.0001);

    const all_masked = FlashStat{ .n_heads = 16, .processed_qkv_pairs = 8, .cycles = 100 };
    try std.testing.expect(all_masked.cyclesPerValidUpdate() == null);
    try std.testing.expect(all_masked.cyclesPerProcessedUpdate() != null);
}

test "profile decoder rejects unknown backend tags" {
    const stats = [_]MatmulStat{.{ .backend = .pl, .path = .direct }};
    const encoded = try encodeAlloc(std.testing.allocator, .{ .summary = .{}, .matmul_stats = &stats });
    defer std.testing.allocator.free(encoded);
    encoded[header_len] = 0xff;
    try std.testing.expectError(error.InvalidEnum, decodeAlloc(std.testing.allocator, encoded));
}

test "profile service accounting can be patched after serialization" {
    const encoded = try encodeAlloc(std.testing.allocator, .{ .summary = .{} });
    defer std.testing.allocator.free(encoded);
    try patchProfileSpan(encoded, 123, 17, AccountingViolation.device_stages);
    var decoded = try decodeAlloc(std.testing.allocator, encoded);
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u64, 123), decoded.summary.profile_span_ns);
    try std.testing.expectEqual(@as(u64, 17), decoded.summary.profile_encode_ns);
    try std.testing.expectEqual(AccountingViolation.device_stages, decoded.summary.accounting_violations);
}
