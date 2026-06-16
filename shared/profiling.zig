const std = @import("std");

pub const version: u16 = 3;
pub const max_spans = 8192;
pub const max_op_tag = 64;
/// Distinct weight formats a matmul can carry (wire.WeightFormat values are
/// 1-based; this bounds the per-format MatmulStat table indexed by that value).
pub const max_weight_fmt = 8;

const magic: u32 = 0x5046_5250; // "PRFP", little-endian on the wire.
const header_len: usize = 64;
const aggregate_len: usize = 24;
const span_len: usize = 28;
const matmul_stat_len: usize = 104;
const flash_stat_len: usize = 40;

pub const DecodeError = error{
    Truncated,
    BadMagic,
    UnsupportedVersion,
    InvalidLength,
    OutOfMemory,
};

pub const Summary = struct {
    device_total_ns: u64 = 0,
    decode_ns: u64 = 0,
    execute_ns: u64 = 0,
    command_count: u32 = 0,
    span_count: u32 = 0,
    span_dropped: u32 = 0,
    /// Fabric clock the PL kernel ran at (from its CLK_HZ register). Zero on a
    /// PS-only run, so the host falls back to wall-time bandwidth estimates and
    /// leaves the clock out of the variant label.
    device_fclk_hz: u32 = 0,
};

pub const Aggregate = struct {
    tag: u16 = 0,
    count: u32 = 0,
    total_ns: u64 = 0,
    bytes: u64 = 0,
};

pub const Span = struct {
    tag: u16 = 0,
    start_ns: u64 = 0,
    end_ns: u64 = 0,
    bytes: u64 = 0,
};

/// Per-(matmul format) rollup. `count`/`macs`/`total_ns` cover every matmul of
/// this format (CPU or PL). The `pl_*` fields and the cycle/stall/beat counters
/// cover only the PL-executed subset — so MAC/cycle (`pl_macs/cycles`) and PL
/// wall throughput (`pl_macs/pl_ns`) are not polluted by CPU-fallback matmuls
/// that share the format bucket. All `pl_*` and counter fields stay zero for a
/// CPU-only run. `fmt` is a wire.WeightFormat value, kept as a raw u16 so shared/
/// stays free of the wire enum.
pub const MatmulStat = struct {
    fmt: u16 = 0,
    count: u32 = 0,
    macs: u64 = 0,
    total_ns: u64 = 0,
    pl_count: u32 = 0,
    pl_macs: u64 = 0,
    pl_ns: u64 = 0,
    cycles: u64 = 0,
    w_stall_cycles: u64 = 0,
    a_stall_cycles: u64 = 0,
    r_stall_cycles: u64 = 0,
    w_beats: u64 = 0,
    a_beats: u64 = 0,
    r_beats: u64 = 0,
};

/// Per-phase flash-attention shape rollup. There is one flash op kind, so this
/// is a single accumulator (the wire carries 0 or 1 of them). The shape fields
/// (`n_heads`/`n_head_kv`/`head_dim_*`) are constant for a model, captured as-is;
/// only `n_kv` varies per call (it grows over decode), so `sum_n_kv`/`max_n_kv`
/// describe its distribution. The host derives avg n_kv, ns/call, ns per
/// (head·kv) inner iteration, and effective bandwidth — enough to tell a padded
/// `n_kv` walk apart from genuine per-iteration cost.
pub const FlashStat = struct {
    count: u32 = 0,
    n_heads: u16 = 0,
    n_head_kv: u16 = 0,
    head_dim_q: u16 = 0,
    head_dim_v: u16 = 0,
    sum_n_kv: u64 = 0,
    max_n_kv: u32 = 0,
    total_ns: u64 = 0,
    bytes: u64 = 0,
};

pub const ReportView = struct {
    summary: Summary,
    aggregates: []const Aggregate = &.{},
    spans: []const Span = &.{},
    matmul_stats: []const MatmulStat = &.{},
    flash_stats: []const FlashStat = &.{},
};

pub const Report = struct {
    summary: Summary,
    aggregates: []Aggregate = &.{},
    spans: []Span = &.{},
    matmul_stats: []MatmulStat = &.{},
    flash_stats: []FlashStat = &.{},

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.aggregates);
        allocator.free(self.spans);
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
    return header_len + report.aggregates.len * aggregate_len + report.spans.len * span_len +
        report.matmul_stats.len * matmul_stat_len + report.flash_stats.len * flash_stat_len;
}

pub fn encode(report: ReportView, out: []u8) error{OutputTooSmall}!usize {
    const want = encodedLen(report);
    if (out.len < want) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU32(out, &cursor, magic);
    putU16(out, &cursor, version);
    putU16(out, &cursor, 0);
    putU64(out, &cursor, report.summary.device_total_ns);
    putU64(out, &cursor, report.summary.decode_ns);
    putU64(out, &cursor, report.summary.execute_ns);
    putU32(out, &cursor, report.summary.command_count);
    putU32(out, &cursor, report.summary.span_count);
    putU32(out, &cursor, report.summary.span_dropped);
    putU32(out, &cursor, report.summary.device_fclk_hz);
    putU32(out, &cursor, @intCast(report.aggregates.len));
    putU32(out, &cursor, @intCast(report.spans.len));
    putU32(out, &cursor, @intCast(report.matmul_stats.len));
    putU32(out, &cursor, @intCast(report.flash_stats.len));

    for (report.aggregates) |aggregate| {
        putU16(out, &cursor, aggregate.tag);
        putU16(out, &cursor, 0);
        putU32(out, &cursor, aggregate.count);
        putU64(out, &cursor, aggregate.total_ns);
        putU64(out, &cursor, aggregate.bytes);
    }
    for (report.spans) |span| {
        putU16(out, &cursor, span.tag);
        putU16(out, &cursor, 0);
        putU64(out, &cursor, span.start_ns);
        putU64(out, &cursor, span.end_ns);
        putU64(out, &cursor, span.bytes);
    }
    for (report.matmul_stats) |stat| {
        putU16(out, &cursor, stat.fmt);
        putU16(out, &cursor, 0);
        putU32(out, &cursor, stat.count);
        putU64(out, &cursor, stat.macs);
        putU64(out, &cursor, stat.total_ns);
        putU32(out, &cursor, stat.pl_count);
        putU32(out, &cursor, 0);
        putU64(out, &cursor, stat.pl_macs);
        putU64(out, &cursor, stat.pl_ns);
        putU64(out, &cursor, stat.cycles);
        putU64(out, &cursor, stat.w_stall_cycles);
        putU64(out, &cursor, stat.a_stall_cycles);
        putU64(out, &cursor, stat.r_stall_cycles);
        putU64(out, &cursor, stat.w_beats);
        putU64(out, &cursor, stat.a_beats);
        putU64(out, &cursor, stat.r_beats);
    }
    for (report.flash_stats) |stat| {
        putU32(out, &cursor, stat.count);
        putU16(out, &cursor, stat.n_heads);
        putU16(out, &cursor, stat.n_head_kv);
        putU16(out, &cursor, stat.head_dim_q);
        putU16(out, &cursor, stat.head_dim_v);
        putU64(out, &cursor, stat.sum_n_kv);
        putU32(out, &cursor, stat.max_n_kv);
        putU64(out, &cursor, stat.total_ns);
        putU64(out, &cursor, stat.bytes);
    }
    return cursor;
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
    const summary = Summary{
        .device_total_ns = try takeU64(bytes, &cursor),
        .decode_ns = try takeU64(bytes, &cursor),
        .execute_ns = try takeU64(bytes, &cursor),
        .command_count = try takeU32(bytes, &cursor),
        .span_count = try takeU32(bytes, &cursor),
        .span_dropped = try takeU32(bytes, &cursor),
        .device_fclk_hz = try takeU32(bytes, &cursor),
    };
    const aggregate_count = try takeU32(bytes, &cursor);
    const span_count = try takeU32(bytes, &cursor);
    const matmul_stat_count = try takeU32(bytes, &cursor);
    const flash_stat_count = try takeU32(bytes, &cursor);

    const aggregate_bytes = std.math.mul(usize, @intCast(aggregate_count), aggregate_len) catch return error.InvalidLength;
    const spans_bytes = std.math.mul(usize, @intCast(span_count), span_len) catch return error.InvalidLength;
    const matmul_bytes = std.math.mul(usize, @intCast(matmul_stat_count), matmul_stat_len) catch return error.InvalidLength;
    const flash_bytes = std.math.mul(usize, @intCast(flash_stat_count), flash_stat_len) catch return error.InvalidLength;
    var total = std.math.add(usize, header_len, aggregate_bytes) catch return error.InvalidLength;
    total = std.math.add(usize, total, spans_bytes) catch return error.InvalidLength;
    total = std.math.add(usize, total, matmul_bytes) catch return error.InvalidLength;
    total = std.math.add(usize, total, flash_bytes) catch return error.InvalidLength;
    if (bytes.len != total) return error.InvalidLength;

    const aggregates = allocator.alloc(Aggregate, @intCast(aggregate_count)) catch return error.OutOfMemory;
    errdefer allocator.free(aggregates);
    const spans = allocator.alloc(Span, @intCast(span_count)) catch return error.OutOfMemory;
    errdefer allocator.free(spans);
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
    for (spans) |*span| {
        span.* = .{
            .tag = try takeU16(bytes, &cursor),
            .start_ns = blk: {
                _ = try takeU16(bytes, &cursor);
                break :blk try takeU64(bytes, &cursor);
            },
            .end_ns = try takeU64(bytes, &cursor),
            .bytes = try takeU64(bytes, &cursor),
        };
    }
    for (matmul_stats) |*stat| {
        stat.* = .{
            .fmt = try takeU16(bytes, &cursor),
            .count = blk: {
                _ = try takeU16(bytes, &cursor);
                break :blk try takeU32(bytes, &cursor);
            },
            .macs = try takeU64(bytes, &cursor),
            .total_ns = try takeU64(bytes, &cursor),
            .pl_count = blk: {
                const c = try takeU32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk c;
            },
            .pl_macs = try takeU64(bytes, &cursor),
            .pl_ns = try takeU64(bytes, &cursor),
            .cycles = try takeU64(bytes, &cursor),
            .w_stall_cycles = try takeU64(bytes, &cursor),
            .a_stall_cycles = try takeU64(bytes, &cursor),
            .r_stall_cycles = try takeU64(bytes, &cursor),
            .w_beats = try takeU64(bytes, &cursor),
            .a_beats = try takeU64(bytes, &cursor),
            .r_beats = try takeU64(bytes, &cursor),
        };
    }
    for (flash_stats) |*stat| {
        stat.* = .{
            .count = try takeU32(bytes, &cursor),
            .n_heads = try takeU16(bytes, &cursor),
            .n_head_kv = try takeU16(bytes, &cursor),
            .head_dim_q = try takeU16(bytes, &cursor),
            .head_dim_v = try takeU16(bytes, &cursor),
            .sum_n_kv = try takeU64(bytes, &cursor),
            .max_n_kv = try takeU32(bytes, &cursor),
            .total_ns = try takeU64(bytes, &cursor),
            .bytes = try takeU64(bytes, &cursor),
        };
    }

    return .{ .summary = summary, .aggregates = aggregates, .spans = spans, .matmul_stats = matmul_stats, .flash_stats = flash_stats };
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
    const aggregates = [_]Aggregate{
        .{ .tag = 2, .count = 3, .total_ns = 99, .bytes = 1234 },
    };
    const spans = [_]Span{
        .{ .tag = 2, .start_ns = 10, .end_ns = 20, .bytes = 64 },
    };
    const matmul_stats = [_]MatmulStat{
        .{ .fmt = 1, .count = 5, .macs = 9000, .total_ns = 42, .pl_count = 3, .pl_macs = 6000, .pl_ns = 30, .cycles = 1000, .w_stall_cycles = 100, .a_stall_cycles = 20, .r_stall_cycles = 5, .w_beats = 700, .a_beats = 80, .r_beats = 16 },
    };
    const flash_stats = [_]FlashStat{
        .{ .count = 28, .n_heads = 16, .n_head_kv = 8, .head_dim_q = 128, .head_dim_v = 128, .sum_n_kv = 784, .max_n_kv = 54, .total_ns = 56000, .bytes = 1_234_567 },
    };
    const view = ReportView{
        .summary = .{
            .device_total_ns = 100,
            .decode_ns = 7,
            .execute_ns = 90,
            .command_count = 3,
            .span_count = spans.len,
            .span_dropped = 1,
            .device_fclk_hz = 250_000_000,
        },
        .aggregates = &aggregates,
        .spans = &spans,
        .matmul_stats = &matmul_stats,
        .flash_stats = &flash_stats,
    };
    const encoded = try encodeAlloc(std.testing.allocator, view);
    defer std.testing.allocator.free(encoded);
    var decoded = try decodeAlloc(std.testing.allocator, encoded);
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(view.summary.device_total_ns, decoded.summary.device_total_ns);
    try std.testing.expectEqual(view.summary.span_dropped, decoded.summary.span_dropped);
    try std.testing.expectEqual(view.summary.device_fclk_hz, decoded.summary.device_fclk_hz);
    try std.testing.expectEqual(aggregates[0].bytes, decoded.aggregates[0].bytes);
    try std.testing.expectEqual(spans[0].end_ns, decoded.spans[0].end_ns);
    try std.testing.expectEqual(@as(usize, 1), decoded.matmul_stats.len);
    try std.testing.expectEqual(matmul_stats[0].macs, decoded.matmul_stats[0].macs);
    try std.testing.expectEqual(matmul_stats[0].pl_macs, decoded.matmul_stats[0].pl_macs);
    try std.testing.expectEqual(matmul_stats[0].pl_ns, decoded.matmul_stats[0].pl_ns);
    try std.testing.expectEqual(matmul_stats[0].w_stall_cycles, decoded.matmul_stats[0].w_stall_cycles);
    try std.testing.expectEqual(matmul_stats[0].r_beats, decoded.matmul_stats[0].r_beats);
    try std.testing.expectEqual(@as(usize, 1), decoded.flash_stats.len);
    try std.testing.expectEqual(flash_stats[0].sum_n_kv, decoded.flash_stats[0].sum_n_kv);
    try std.testing.expectEqual(flash_stats[0].max_n_kv, decoded.flash_stats[0].max_n_kv);
    try std.testing.expectEqual(flash_stats[0].n_heads, decoded.flash_stats[0].n_heads);
    try std.testing.expectEqual(flash_stats[0].total_ns, decoded.flash_stats[0].total_ns);
    try std.testing.expectEqual(flash_stats[0].bytes, decoded.flash_stats[0].bytes);
}
