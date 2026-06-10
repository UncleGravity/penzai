const std = @import("std");
const profiling = @import("profiling");
const prof_report = @import("prof_report");

// Trace capture file: a small magic header followed by one record per profiled
// run_graph. Each record is {host_base_ns: u64}{payload_len: u32}{payload}, where
// payload is a profiling.Report encoded with the shared wire codec — so the
// offline converter reuses exactly the decode path the device produced. host_base_ns
// is the host monotonic clock at the call, used to lay each graph on a global timeline.
const magic = "PZTRACE\x01";
const record_header_len = 12;

pub const TraceError = error{
    BadCapture,
} || std.mem.Allocator.Error;

/// In-memory accumulator for trace records during a run. Kept in memory (not
/// streamed) so the deep llama/backend path never touches files; the host writes
/// the whole buffer once at the end.
pub const Capture = struct {
    allocator: std.mem.Allocator,
    data: std.ArrayList(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) Capture {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Capture) void {
        self.data.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn append(self: *Capture, report: profiling.Report, host_base_ns: u64) std.mem.Allocator.Error!void {
        const payload = try profiling.encodeAlloc(self.allocator, .{
            .summary = report.summary,
            .aggregates = report.aggregates,
            .spans = report.spans,
        });
        defer self.allocator.free(payload);
        var header: [record_header_len]u8 = undefined;
        std.mem.writeInt(u64, header[0..8], host_base_ns, .little);
        std.mem.writeInt(u32, header[8..12], @intCast(payload.len), .little);
        try self.data.appendSlice(self.allocator, &header);
        try self.data.appendSlice(self.allocator, payload);
    }

    pub fn writeFile(self: *const Capture, io: std.Io, path: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer file.close(io);
        try file.writeStreamingAll(io, magic);
        try file.writeStreamingAll(io, self.data.items);
    }
};

/// Read a capture file and emit a Chrome Trace Event JSON array to `writer`.
pub fn convertFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
    writer: *std.Io.Writer,
) !void {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(raw);
    try convertBytes(allocator, raw, writer);
}

/// Decode capture bytes and emit Chrome Trace Event JSON. Each device span becomes
/// one "complete" event placed at host_base + span offset (origin shifted to zero).
pub fn convertBytes(allocator: std.mem.Allocator, raw: []const u8, writer: *std.Io.Writer) !void {
    if (raw.len < magic.len or !std.mem.eql(u8, raw[0..magic.len], magic)) return error.BadCapture;

    try writer.writeAll("[\n");
    var cursor: usize = magic.len;
    var origin: ?u64 = null;
    var first = true;
    while (cursor < raw.len) {
        if (cursor + record_header_len > raw.len) return error.BadCapture;
        const host_base = std.mem.readInt(u64, raw[cursor..][0..8], .little);
        const payload_len = std.mem.readInt(u32, raw[cursor + 8 ..][0..4], .little);
        cursor += record_header_len;
        if (cursor + payload_len > raw.len) return error.BadCapture;

        var report = profiling.decodeAlloc(allocator, raw[cursor..][0..payload_len]) catch return error.BadCapture;
        defer report.deinit(allocator);
        cursor += payload_len;

        if (origin == null) origin = host_base;
        const base = host_base - origin.?;
        for (report.spans) |span| {
            const ts_us = usFromNs(base +| span.start_ns);
            const dur_us = usFromNs(span.end_ns -| span.start_ns);
            if (!first) try writer.writeAll(",\n");
            first = false;
            try writer.print(
                "{{\"name\":\"{s}\",\"ph\":\"X\",\"pid\":1,\"tid\":1,\"ts\":{d:.3},\"dur\":{d:.3},\"args\":{{\"bytes\":{d}}}}}",
                .{ prof_report.opName(span.tag), ts_us, dur_us, span.bytes },
            );
        }
    }
    try writer.writeAll("\n]\n");
}

fn usFromNs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1000.0;
}

test "trace capture roundtrips through chrome json" {
    const wire = @import("wire");
    const allocator = std.testing.allocator;
    var capture = Capture.init(allocator);
    defer capture.deinit();

    const spans = [_]profiling.Span{
        .{ .tag = @intFromEnum(wire.OpTag.matmul_q1a8), .start_ns = 10, .end_ns = 60, .bytes = 4096 },
    };
    const report = profiling.Report{
        .summary = .{ .device_total_ns = 100, .command_count = 1, .span_count = 1 },
        .aggregates = &.{},
        .spans = @constCast(spans[0..]),
    };
    try capture.append(report, 1_000);
    try capture.append(report, 2_000);

    // Reconstruct the on-disk bytes (magic + records) that writeFile would produce.
    const raw = try allocator.alloc(u8, magic.len + capture.data.items.len);
    defer allocator.free(raw);
    @memcpy(raw[0..magic.len], magic);
    @memcpy(raw[magic.len..], capture.data.items);

    var buf: [4096]u8 = undefined;
    var stream = std.Io.Writer.fixed(&buf);
    try convertBytes(allocator, raw, &stream);
    const json = stream.buffered();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"matmul_q1a8\"") != null);
    // origin shift: first record at ts 0.010us, second at 1.000+0.010.
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ts\":0.010") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ts\":1.010") != null);
    try std.testing.expectError(error.BadCapture, convertBytes(allocator, "not a capture", &stream));
}
