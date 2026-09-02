const std = @import("std");
const builtin = @import("builtin");
const shared = @import("shared");
const runtime_mod = @import("runtime");
const server_mod = @import("server");

const protocol_transport = shared.protocol_transport;
const capabilities = shared.capabilities;
const framing = shared.framing;
const wire = shared.wire;

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
    EngineOpenFailed,
};

pub const ServeOptions = struct {
    heap_mib: u32 = 1500,
    max_requests: ?u32 = null,
    memory: MemoryBackend = .xrt,
    receipt_path: []const u8 = "",
};

test "board daemon defaults to the development heap" {
    const options: ServeOptions = .{};
    try std.testing.expectEqual(@as(u32, 1500), options.heap_mib);
}

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
        .xrt => try serveWithRuntime(runtime_mod.XrtRuntime, io, allocator, &listener, heap_size, options),
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
    const receipt = loadReceipt(io, options.receipt_path);
    var runtime = Runtime.initWithReceipt(allocator, heap_size, receipt) catch |err| return mapInitError(err);
    defer runtime.deinit();

    var remaining = options.max_requests;
    while (remaining == null or remaining.? > 0) {
        var stream = listener.accept(io) catch |err| switch (err) {
            // A peer can disappear between the kernel queuing and accepting it.
            // That is a client failure, not a listener failure.
            error.ConnectionAborted => continue,
            else => return error.Transport,
        };
        const result = serveStream(io, allocator, &runtime, stream, remaining);
        stream.close(io);
        const handled = result catch |err| switch (err) {
            // One client owns the runtime at a time. A malformed frame or a
            // broken socket ends only that client's turn, then accept resumes.
            error.Protocol, error.Transport => {
                if (!builtin.is_test)
                    std.debug.print("device client ended with {s}; accepting the next client\n", .{@errorName(err)});
                continue;
            },
            error.OutOfMemory => return error.OutOfMemory,
        };
        if (remaining) |*n| {
            if (handled >= n.*) {
                n.* = 0;
            } else {
                n.* -= handled;
            }
        }
    }
}

const max_receipt_bytes = 4096;

fn loadReceipt(io: std.Io, path: []const u8) capabilities.Receipt {
    if (path.len == 0) return .{};
    const file = std.Io.Dir.openFileAbsolute(io, path, .{ .allow_directory = false }) catch return .{};
    defer file.close(io);

    var buf: [max_receipt_bytes]u8 = undefined;
    var used: u64 = 0;
    while (used < buf.len) {
        var dst = [_][]u8{buf[@intCast(used)..]};
        const n = io.vtable.fileReadPositional(io.userdata, file, &dst, used) catch return capabilities.Receipt.invalid();
        if (n == 0) break;
        used += n;
    }
    if (used == buf.len) {
        var extra: [1]u8 = undefined;
        var dst = [_][]u8{extra[0..]};
        const n = io.vtable.fileReadPositional(io.userdata, file, &dst, used) catch return capabilities.Receipt.invalid();
        if (n != 0) return capabilities.Receipt.invalid();
    }
    return capabilities.Receipt.parse(buf[0..@intCast(used)]) catch capabilities.Receipt.invalid();
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
        error.EngineOpenFailed => error.EngineOpenFailed,
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
) error{ OutOfMemory, Transport, Protocol }!u32 {
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

test "native TCP reconnect, malformed isolation, and sequential reuse" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const listen_address: net.IpAddress = .{ .ip4 = net.Ip4Address.loopback(0) };
    var listener = try listen_address.listen(io, .{ .reuse_address = true });
    var listener_open = true;
    defer if (listener_open) listener.deinit(io);

    const TestServer = struct {
        io: std.Io,
        allocator: std.mem.Allocator,
        listener: *net.Server,
        result: ?ServeError = null,

        fn run(self: *@This()) void {
            serveWithRuntime(
                runtime_mod.Runtime,
                self.io,
                self.allocator,
                self.listener,
                4096,
                .{ .memory = .fake, .max_requests = 4 },
            ) catch |err| {
                self.result = err;
            };
        }
    };

    var test_server = TestServer{
        .io = io,
        .allocator = allocator,
        .listener = &listener,
    };
    const thread = try std.Thread.spawn(.{}, TestServer.run, .{&test_server});
    var thread_joined = false;
    defer if (!thread_joined) {
        if (listener_open) {
            listener.deinit(io);
            listener_open = false;
        }
        thread.join();
    };

    const address = listener.socket.address;

    // A successful request may disconnect and a later client reuses the same
    // runtime state. Keep an allocation alive across the reconnect to prove it.
    var first = try address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    var request_meta: [64]u8 = undefined;
    const alloc_len = try wire.encodeAlloc(&request_meta, 1, 64, 64);
    const alloc_response = try callTestClient(io, allocator, first, request_meta[0..alloc_len], "");
    try std.testing.expectEqual(wire.Status.ok, alloc_response.status);
    try std.testing.expect(alloc_response.handle != 0);
    first.close(io);

    // A complete but invalid framing header must close only this connection.
    var malformed = try address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    try writeTestBytes(io, malformed, &([_]u8{0} ** framing.header_len));
    malformed.close(io);

    // A disconnect in the middle of a header is isolated in the same way.
    var partial = try address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    try writeTestBytes(io, partial, "PNZ1\x01\x00\x00\x00");
    partial.close(io);

    // The next connection can free the first client's allocation and can issue
    // multiple sequential requests without reconnecting.
    var reused = try address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    const free_len = try wire.encodeFree(&request_meta, 2, alloc_response.handle);
    const free_response = try callTestClient(io, allocator, reused, request_meta[0..free_len], "");
    try std.testing.expectEqual(wire.Status.ok, free_response.status);
    const hello_len = try wire.encodeHello(&request_meta, 3);
    const hello_response = try callTestClient(io, allocator, reused, request_meta[0..hello_len], "");
    try std.testing.expectEqual(@as(u64, 3), hello_response.request_id);
    try std.testing.expectEqual(wire.Status.ok, hello_response.status);
    reused.close(io);

    // A final reconnect proves accept resumed after both clean and bad clients.
    var final_client = try address.connect(io, .{ .mode = .stream, .protocol = .tcp });
    defer final_client.close(io);
    const capabilities_len = try wire.encodeCapabilities(&request_meta, 4);
    const capabilities_response = try callTestClient(
        io,
        allocator,
        final_client,
        request_meta[0..capabilities_len],
        "",
    );
    try std.testing.expectEqual(@as(u64, 4), capabilities_response.request_id);
    try std.testing.expectEqual(wire.Status.ok, capabilities_response.status);

    thread.join();
    thread_joined = true;
    if (test_server.result) |err| return err;
    listener.deinit(io);
    listener_open = false;
}

fn callTestClient(
    io: std.Io,
    allocator: std.mem.Allocator,
    stream: net.Stream,
    metadata: []const u8,
    payload: []const u8,
) !wire.ResponseMeta {
    var request: [framing.header_len + 64]u8 = undefined;
    const request_len = try framing.encode(metadata, payload, &request);
    var write_buf: [256]u8 = undefined;
    var writer = stream.writer(io, &write_buf);
    try protocol_transport.writeFrame(&writer.interface, request[0..request_len]);

    var read_buf: [256]u8 = undefined;
    var reader = stream.reader(io, &read_buf);
    const response_bytes = try protocol_transport.readFrameAlloc(allocator, &reader.interface);
    defer allocator.free(response_bytes);
    const response = try framing.decode(response_bytes);
    return wire.decodeResponseMeta(response.metadata);
}

fn writeTestBytes(io: std.Io, stream: net.Stream, bytes: []const u8) !void {
    var write_buf: [64]u8 = undefined;
    var writer = stream.writer(io, &write_buf);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}
