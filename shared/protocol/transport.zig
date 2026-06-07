const std = @import("std");
const framing = @import("framing");

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
