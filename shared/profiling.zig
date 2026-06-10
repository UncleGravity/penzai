const std = @import("std");

pub const version: u16 = 1;
pub const max_spans = 8192;
pub const max_op_tag = 64;

const magic: u32 = 0x5046_5250; // "PRFP", little-endian on the wire.
const header_len: usize = 56;
const aggregate_len: usize = 24;
const span_len: usize = 28;

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

pub const ReportView = struct {
    summary: Summary,
    aggregates: []const Aggregate = &.{},
    spans: []const Span = &.{},
};

pub const Report = struct {
    summary: Summary,
    aggregates: []Aggregate = &.{},
    spans: []Span = &.{},

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.aggregates);
        allocator.free(self.spans);
        self.* = undefined;
    }
};

pub fn encodedLen(report: ReportView) usize {
    return header_len + report.aggregates.len * aggregate_len + report.spans.len * span_len;
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
    putU32(out, &cursor, @intCast(report.aggregates.len));
    putU32(out, &cursor, @intCast(report.spans.len));
    putU32(out, &cursor, 0);

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
    };
    const aggregate_count = try takeU32(bytes, &cursor);
    const span_count = try takeU32(bytes, &cursor);
    _ = try takeU32(bytes, &cursor);

    const aggregate_bytes = std.math.mul(usize, @intCast(aggregate_count), aggregate_len) catch return error.InvalidLength;
    const spans_bytes = std.math.mul(usize, @intCast(span_count), span_len) catch return error.InvalidLength;
    const expected = std.math.add(usize, header_len, aggregate_bytes) catch return error.InvalidLength;
    const total = std.math.add(usize, expected, spans_bytes) catch return error.InvalidLength;
    if (bytes.len != total) return error.InvalidLength;

    const aggregates = allocator.alloc(Aggregate, @intCast(aggregate_count)) catch return error.OutOfMemory;
    errdefer allocator.free(aggregates);
    const spans = allocator.alloc(Span, @intCast(span_count)) catch return error.OutOfMemory;
    errdefer allocator.free(spans);

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

    return .{ .summary = summary, .aggregates = aggregates, .spans = spans };
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
    const view = ReportView{
        .summary = .{
            .device_total_ns = 100,
            .decode_ns = 7,
            .execute_ns = 90,
            .command_count = 3,
            .span_count = spans.len,
            .span_dropped = 1,
        },
        .aggregates = &aggregates,
        .spans = &spans,
    };
    const encoded = try encodeAlloc(std.testing.allocator, view);
    defer std.testing.allocator.free(encoded);
    var decoded = try decodeAlloc(std.testing.allocator, encoded);
    defer decoded.deinit(std.testing.allocator);
    try std.testing.expectEqual(view.summary.device_total_ns, decoded.summary.device_total_ns);
    try std.testing.expectEqual(view.summary.span_dropped, decoded.summary.span_dropped);
    try std.testing.expectEqual(aggregates[0].bytes, decoded.aggregates[0].bytes);
    try std.testing.expectEqual(spans[0].end_ns, decoded.spans[0].end_ns);
}
