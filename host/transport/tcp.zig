const std = @import("std");
const shared = @import("shared");

const protocol_transport = shared.protocol_transport;

const net = std.Io.net;

pub const Error = error{
    OutOfMemory,
    InvalidAddress,
    Transport,
    Protocol,
};

/// Stream buffer size. Frames larger than this still work (the buffered
/// reader/writer drain in chunks); this just bounds syscalls per frame.
const buffer_size = 64 * 1024;

pub const Endpoint = struct {
    const Self = @This();

    io: std.Io,
    stream: net.Stream,
    // Persistent stream buffers: the Endpoint is pinned for its lifetime (held by
    // TcpLink), so `call` can hand the stream reader/writer these stable buffers
    // instead of zero-length ones (which forced a syscall per fragment).
    read_buf: [buffer_size]u8 = undefined,
    write_buf: [buffer_size]u8 = undefined,

    pub fn connect(io: std.Io, spec: protocol_transport.TcpSpec) Error!Self {
        const stream = connectStream(io, spec) catch |err| switch (err) {
            error.InvalidAddress => return error.InvalidAddress,
            else => return error.Transport,
        };
        setNoDelay(stream.socket.handle);
        return .{ .io = io, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        self.stream.close(self.io);
        self.* = undefined;
    }

    pub fn call(self: *Self, allocator: std.mem.Allocator, request_frame: []const u8) Error![]u8 {
        var writer = self.stream.writer(self.io, &self.write_buf);
        protocol_transport.writeFrame(&writer.interface, request_frame) catch return error.Transport;

        // Safe to recreate the reader per call: the protocol is strict
        // request/response, so the socket holds no bytes past this response frame
        // and the reader never strands cross-frame read-ahead.
        var reader = self.stream.reader(self.io, &self.read_buf);
        return protocol_transport.readFrameAlloc(allocator, &reader.interface) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.BadFrame => error.Protocol,
            else => error.Transport,
        };
    }
};

/// Disable Nagle: our request/response ping-pong of small frames otherwise eats
/// ~40 ms delayed-ACK stalls per round trip. Best-effort — a platform that
/// rejects the option still works, just slower.
fn setNoDelay(handle: std.posix.socket_t) void {
    const one: c_int = 1;
    std.posix.setsockopt(handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
}

fn connectStream(io: std.Io, spec: protocol_transport.TcpSpec) !net.Stream {
    if (std.mem.eql(u8, spec.host, "localhost")) {
        const address: net.IpAddress = .{ .ip4 = net.Ip4Address.loopback(spec.port) };
        return address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    }
    if (net.IpAddress.parse(spec.host, spec.port)) |address| {
        return address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    } else |_| {}
    const host_name = net.HostName.init(spec.host) catch return error.InvalidAddress;
    return net.HostName.connect(host_name, io, spec.port, .{ .mode = .stream, .protocol = .tcp });
}
