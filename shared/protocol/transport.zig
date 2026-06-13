const std = @import("std");
const framing = @import("framing.zig");

pub const TcpSpec = struct {
    host: []const u8,
    port: u16,
};

pub const DeviceSpec = union(enum) {
    fake,
    tcp: TcpSpec,
};

pub const ParseError = error{
    InvalidDevice,
    InvalidTcpSpec,
    MissingTcpHost,
    MissingTcpPort,
    InvalidTcpPort,
};

pub fn parseDeviceSpec(text: []const u8) ParseError!DeviceSpec {
    if (std.mem.eql(u8, text, "fake")) return .fake;
    if (std.mem.startsWith(u8, text, "tcp:")) return .{ .tcp = try parseTcpSpec(text) };
    return error.InvalidDevice;
}

pub fn parseTcpSpec(text: []const u8) ParseError!TcpSpec {
    const rest = if (std.mem.startsWith(u8, text, "tcp:")) text["tcp:".len..] else text;
    const sep = std.mem.lastIndexOfScalar(u8, rest, ':') orelse return error.InvalidTcpSpec;
    const host = rest[0..sep];
    const port_text = rest[sep + 1 ..];
    if (host.len == 0) return error.MissingTcpHost;
    if (port_text.len == 0) return error.MissingTcpPort;
    const port = std.fmt.parseInt(u16, port_text, 10) catch return error.InvalidTcpPort;
    return .{ .host = host, .port = port };
}

pub const StreamError = error{
    OutOfMemory,
    BadFrame,
    EndOfStream,
    ReadFailed,
    WriteFailed,
};

pub const FrameIntoResult = struct {
    metadata: []u8,
    payload_len: usize,
    payload_copied: bool,

    pub fn deinit(self: *FrameIntoResult, allocator: std.mem.Allocator) void {
        allocator.free(self.metadata);
        self.* = undefined;
    }
};

pub fn readFrameAlloc(allocator: std.mem.Allocator, reader: *std.Io.Reader) StreamError![]u8 {
    var header_buf: [framing.header_len]u8 = undefined;
    reader.readSliceAll(&header_buf) catch |err| switch (err) {
        error.EndOfStream => return error.EndOfStream,
        else => return error.ReadFailed,
    };

    const header = framing.decodeHeader(&header_buf) catch return error.BadFrame;
    const frame = try allocator.alloc(u8, header.total_len);
    errdefer allocator.free(frame);
    @memcpy(frame[0..framing.header_len], &header_buf);
    reader.readSliceAll(frame[framing.header_len..]) catch |err| switch (err) {
        error.EndOfStream => return error.EndOfStream,
        else => return error.ReadFailed,
    };
    _ = framing.decode(frame) catch return error.BadFrame;
    return frame;
}

pub fn readFrameInto(allocator: std.mem.Allocator, reader: *std.Io.Reader, payload_out: []u8) StreamError!FrameIntoResult {
    var header_buf: [framing.header_len]u8 = undefined;
    reader.readSliceAll(&header_buf) catch |err| switch (err) {
        error.EndOfStream => return error.EndOfStream,
        else => return error.ReadFailed,
    };

    const header = framing.decodeHeader(&header_buf) catch return error.BadFrame;
    const metadata = try allocator.alloc(u8, header.metadata_len);
    errdefer allocator.free(metadata);
    reader.readSliceAll(metadata) catch |err| switch (err) {
        error.EndOfStream => return error.EndOfStream,
        else => return error.ReadFailed,
    };

    const payload_copied = header.payload_len == payload_out.len;
    if (payload_copied) {
        reader.readSliceAll(payload_out) catch |err| switch (err) {
            error.EndOfStream => return error.EndOfStream,
            else => return error.ReadFailed,
        };
    } else {
        reader.discardAll(header.payload_len) catch |err| switch (err) {
            error.EndOfStream => return error.EndOfStream,
            else => return error.ReadFailed,
        };
    }

    return .{
        .metadata = metadata,
        .payload_len = header.payload_len,
        .payload_copied = payload_copied,
    };
}

pub fn writeFrame(writer: *std.Io.Writer, frame: []const u8) StreamError!void {
    writer.writeAll(frame) catch return error.WriteFailed;
    writer.flush() catch return error.WriteFailed;
}

test "parse fake and tcp device specs" {
    try std.testing.expectEqual(.fake, try parseDeviceSpec("fake"));
    const device = try parseDeviceSpec("tcp:127.0.0.1:9000");
    try std.testing.expectEqualStrings("127.0.0.1", device.tcp.host);
    try std.testing.expectEqual(@as(u16, 9000), device.tcp.port);
}

test "reject malformed tcp specs" {
    try std.testing.expectError(error.MissingTcpHost, parseTcpSpec("tcp::9000"));
    try std.testing.expectError(error.MissingTcpPort, parseTcpSpec("tcp:127.0.0.1:"));
    try std.testing.expectError(error.InvalidTcpPort, parseTcpSpec("tcp:127.0.0.1:nope"));
}

test "read frame into caller payload buffer" {
    const metadata = "meta";
    const payload = "payload";
    var frame_buf: [128]u8 = undefined;
    const frame_len = try framing.encode(metadata, payload, &frame_buf);

    var reader = std.Io.Reader.fixed(frame_buf[0..frame_len]);
    var out: [payload.len]u8 = undefined;
    var result = try readFrameInto(std.testing.allocator, &reader, &out);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(metadata, result.metadata);
    try std.testing.expectEqualSlices(u8, payload, &out);
    try std.testing.expectEqual(payload.len, result.payload_len);
    try std.testing.expect(result.payload_copied);
    try std.testing.expectEqual(reader.end, reader.seek);
}

test "read frame into drains unexpected payload length" {
    const metadata = "meta";
    const payload = "payload";
    var frame_buf: [128]u8 = undefined;
    const frame_len = try framing.encode(metadata, payload, &frame_buf);

    var reader = std.Io.Reader.fixed(frame_buf[0..frame_len]);
    var out: [3]u8 = undefined;
    var result = try readFrameInto(std.testing.allocator, &reader, &out);
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(metadata, result.metadata);
    try std.testing.expectEqual(payload.len, result.payload_len);
    try std.testing.expect(!result.payload_copied);
    try std.testing.expectEqual(reader.end, reader.seek);
}
