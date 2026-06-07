const std = @import("std");
const protocol_transport = @import("protocol_transport");

const net = std.Io.net;

pub const Error = error{
    OutOfMemory,
    InvalidAddress,
    Transport,
    Protocol,
};

pub const Endpoint = struct {
    const Self = @This();

    io: std.Io,
    stream: net.Stream,

    pub fn connect(io: std.Io, spec: protocol_transport.TcpSpec) Error!Self {
        const stream = connectStream(io, spec) catch |err| switch (err) {
            error.InvalidAddress => return error.InvalidAddress,
            else => return error.Transport,
        };
        return .{ .io = io, .stream = stream };
    }

    pub fn deinit(self: *Self) void {
        self.stream.close(self.io);
        self.* = undefined;
    }

    pub fn call(self: *Self, allocator: std.mem.Allocator, request_frame: []const u8) Error![]u8 {
        var write_buf: [0]u8 = .{};
        var writer = self.stream.writer(self.io, &write_buf);
        protocol_transport.writeFrame(&writer.interface, request_frame) catch return error.Transport;

        var read_buf: [0]u8 = .{};
        var reader = self.stream.reader(self.io, &read_buf);
        return protocol_transport.readFrameAlloc(allocator, &reader.interface) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.BadFrame => error.Protocol,
            else => error.Transport,
        };
    }
};

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
