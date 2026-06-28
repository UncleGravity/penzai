//! In-process fake transport — services each request by driving the device
//! runtime directly through `server.handleFrame`, no socket. This is the only
//! file under link/ that pulls in the device simulator (runtime + server);
//! isolating it here is what keeps the wire-only path (link.zig + tcp.zig) free
//! of the device for a future out-of-tree `.so`. Correctness / oracle use only —
//! never trust it for transport perf (§3 #10).
const std = @import("std");
const shared = @import("shared");
const runtime_mod = @import("runtime");
const server = @import("server");
const link = @import("link.zig");

const framing = shared.framing;
const protocol_transport = shared.protocol_transport;

pub const FakeTransport = struct {
    const Self = @This();

    runtime: *runtime_mod.Runtime,
    io_handle: ?std.Io,

    pub fn init(runtime: *runtime_mod.Runtime, io: ?std.Io) Self {
        return .{ .runtime = runtime, .io_handle = io };
    }

    pub fn ioHandle(self: *Self) ?std.Io {
        return self.io_handle;
    }

    pub fn call(self: *Self, allocator: std.mem.Allocator, request_frame: []const u8) link.LinkError![]u8 {
        return server.handleFrame(self.io_handle, allocator, self.runtime, request_frame) catch return error.Protocol;
    }

    /// No socket to read into, so service the request and copy the response payload
    /// into the caller's buffer — the same observable result as the wire's zero-copy
    /// `callInto`, so `Link.download` stays single-path.
    pub fn callInto(
        self: *Self,
        allocator: std.mem.Allocator,
        request_frame: []const u8,
        payload_out: []u8,
    ) link.LinkError!protocol_transport.FrameIntoResult {
        const response_frame = try self.call(allocator, request_frame);
        defer allocator.free(response_frame);
        const frame = framing.decode(response_frame) catch return error.Protocol;
        const metadata = allocator.dupe(u8, frame.metadata) catch return error.OutOfMemory;
        errdefer allocator.free(metadata);
        const payload_copied = frame.payload.len == payload_out.len;
        if (payload_copied) @memcpy(payload_out, frame.payload);
        return .{
            .metadata = metadata,
            .payload_len = frame.payload.len,
            .payload_copied = payload_copied,
        };
    }
};
