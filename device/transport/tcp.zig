const std = @import("std");
const protocol_transport = @import("protocol_transport");
const runtime_mod = @import("runtime");
const server_mod = @import("server");
const xrt_bo = @import("xrt_bo");

const net = std.Io.net;

pub const ServeError = error{
    OutOfMemory,
    InvalidAddress,
    InvalidShape,
    Transport,
    Protocol,
};

pub const ServeOptions = struct {
    heap_mib: u32 = 16,
    max_requests: ?u32 = null,
    memory: MemoryBackend = .fake,
};

pub const MemoryBackend = enum {
    fake,
    xrt,
};

pub fn serve(io: std.Io, allocator: std.mem.Allocator, spec: protocol_transport.TcpSpec, options: ServeOptions) ServeError!void {
    const heap_size = try heapBytes(options.heap_mib);

    const address = resolveListenAddress(spec) catch return error.InvalidAddress;
    var listener = address.listen(io, .{ .reuse_address = true }) catch return error.Transport;
    defer listener.deinit(io);

    switch (options.memory) {
        .fake => try serveWithRuntime(runtime_mod.Runtime, io, allocator, &listener, heap_size, options),
        .xrt => try serveWithRuntime(runtime_mod.RuntimeFor(xrt_bo.Heap), io, allocator, &listener, heap_size, options),
    }
}

fn serveWithRuntime(
    comptime Runtime: type,
    io: std.Io,
    allocator: std.mem.Allocator,
    listener: *net.Server,
    heap_size: usize,
    options: ServeOptions,
) ServeError!void {
    var runtime = Runtime.init(allocator, heap_size) catch |err| return mapInitError(err);
    defer runtime.deinit();

    var remaining = options.max_requests;
    while (remaining == null or remaining.? > 0) {
        var stream = listener.accept(io) catch return error.Transport;
        defer stream.close(io);
        const handled = try serveStream(io, allocator, &runtime, stream, remaining);
        if (remaining) |*n| {
            if (handled >= n.*) {
                n.* = 0;
            } else {
                n.* -= handled;
            }
        }
    }
}

fn mapInitError(err: anyerror) ServeError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.Transport,
    };
}

fn serveStream(
    io: std.Io,
    allocator: std.mem.Allocator,
    runtime: anytype,
    stream: net.Stream,
    max_requests: ?u32,
) ServeError!u32 {
    var handled: u32 = 0;
    var read_buf: [0]u8 = .{};
    var reader = stream.reader(io, &read_buf);
    var write_buf: [0]u8 = .{};
    var writer = stream.writer(io, &write_buf);

    while (max_requests == null or handled < max_requests.?) {
        {
            var frame = protocol_transport.readFrameAlloc(allocator, &reader.interface) catch |err| switch (err) {
                error.EndOfStream => return handled,
                error.OutOfMemory => return error.OutOfMemory,
                error.BadFrame => return error.Protocol,
                else => return error.Transport,
            };
            defer frame = undefined;
            defer allocator.free(frame);

            var response = server_mod.handleFrame(allocator, runtime, frame) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.BadFrame => return error.Protocol,
            };
            defer response = undefined;
            defer allocator.free(response);

            protocol_transport.writeFrame(&writer.interface, response) catch return error.Transport;
        }
        handled += 1;
    }
    return handled;
}

fn heapBytes(heap_mib: u32) ServeError!usize {
    if (heap_mib == 0) return error.InvalidShape;
    const kib = std.math.mul(usize, @intCast(heap_mib), 1024) catch return error.InvalidShape;
    return std.math.mul(usize, kib, 1024) catch return error.InvalidShape;
}

fn resolveListenAddress(spec: protocol_transport.TcpSpec) !net.IpAddress {
    if (std.mem.eql(u8, spec.host, "localhost")) {
        return .{ .ip4 = net.Ip4Address.loopback(spec.port) };
    }
    return net.IpAddress.parse(spec.host, spec.port);
}
