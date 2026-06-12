const std = @import("std");
const shared = @import("shared");
const runtime_mod = @import("runtime");
const server_mod = @import("server");
const xrt_bo = @import("../mem/xrt_bo.zig");

const protocol_transport = shared.protocol_transport;

const net = std.Io.net;

/// Per-connection stream buffer size. Larger frames still work (drained in chunks).
const buffer_size = 64 * 1024;

pub const ServeError = error{
    OutOfMemory,
    InvalidAddress,
    InvalidShape,
    Transport,
    Protocol,
    XrtOpenFailed,
    XrtSymbolMissing,
    XrtDeviceOpenFailed,
    XrtBOAllocFailed,
    XrtBOMapFailed,
    XrtBOSyncFailed,
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
        error.XrtOpenFailed => error.XrtOpenFailed,
        error.XrtSymbolMissing => error.XrtSymbolMissing,
        error.XrtDeviceOpenFailed => error.XrtDeviceOpenFailed,
        error.XrtBOAllocFailed => error.XrtBOAllocFailed,
        error.XrtBOMapFailed => error.XrtBOMapFailed,
        error.XrtBOSyncFailed => error.XrtBOSyncFailed,
        else => error.Transport,
    };
}

test "runtime init errors preserve XRT causes" {
    try std.testing.expectEqual(error.XrtOpenFailed, mapInitError(error.XrtOpenFailed));
    try std.testing.expectEqual(error.XrtSymbolMissing, mapInitError(error.XrtSymbolMissing));
    try std.testing.expectEqual(error.XrtDeviceOpenFailed, mapInitError(error.XrtDeviceOpenFailed));
    try std.testing.expectEqual(error.XrtBOAllocFailed, mapInitError(error.XrtBOAllocFailed));
    try std.testing.expectEqual(error.XrtBOMapFailed, mapInitError(error.XrtBOMapFailed));
    try std.testing.expectEqual(error.XrtBOSyncFailed, mapInitError(error.XrtBOSyncFailed));
    try std.testing.expectEqual(error.Transport, mapInitError(error.Unexpected));
}

fn serveStream(
    io: std.Io,
    allocator: std.mem.Allocator,
    runtime: anytype,
    stream: net.Stream,
    max_requests: ?u32,
) ServeError!u32 {
    setNoDelay(stream.socket.handle);
    var handled: u32 = 0;
    // Persistent per-connection stream buffers (and reader/writer), so request
    // decode and response framing aren't a syscall per fragment. The reader is
    // reused across requests so any read-ahead is retained, not stranded.
    var read_buf: [buffer_size]u8 = undefined;
    var reader = stream.reader(io, &read_buf);
    var write_buf: [buffer_size]u8 = undefined;
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

            var response = server_mod.handleFrame(io, allocator, runtime, frame) catch |err| switch (err) {
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

/// Disable Nagle on the accepted connection: the host/device protocol is a
/// request/response ping-pong, so delayed-ACK + Nagle otherwise stalls every
/// round trip. Best-effort.
fn setNoDelay(handle: std.posix.socket_t) void {
    const one: c_int = 1;
    std.posix.setsockopt(handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
}

fn resolveListenAddress(spec: protocol_transport.TcpSpec) !net.IpAddress {
    if (std.mem.eql(u8, spec.host, "localhost")) {
        return .{ .ip4 = net.Ip4Address.loopback(spec.port) };
    }
    return net.IpAddress.parse(spec.host, spec.port);
}
