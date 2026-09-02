const std = @import("std");
const shared = @import("shared");
const runtime_mod = @import("runtime");

const framing = shared.framing;
const wire = shared.wire;

pub const ServerError = error{
    OutOfMemory,
    BadFrame,
};

pub fn handleFrame(
    io: ?std.Io,
    allocator: std.mem.Allocator,
    runtime: anytype,
    frame_bytes: []const u8,
) ServerError![]u8 {
    const start_ns = nowNs(io);

    const frame = framing.decode(frame_bytes) catch
        return finish(io, allocator, start_ns, errorMeta(0, .invalid_request), "");
    const request = wire.decodeRequest(frame.metadata, frame.payload) catch
        return finish(io, allocator, start_ns, errorMeta(0, .invalid_request), "");
    const request_decode_ns = elapsed(start_ns, nowNs(io));
    var result = runtime.dispatchMeasured(request, io, request_decode_ns) catch |err| {
        std.debug.print("device request {s} failed: {s}\n", .{ @tagName(std.meta.activeTag(request)), @errorName(err) });
        return finish(io, allocator, start_ns, errorMeta(request.requestId(), runtime_mod.errorCode(err)), "");
    };
    defer result.deinit(allocator);
    return finish(io, allocator, start_ns, result.meta, result.payload);
}

fn errorMeta(request_id: u64, code: wire.ErrorCode) wire.ResponseMeta {
    return .{ .request_id = request_id, .status = .failed, .error_code = code };
}

/// Single exit: stamp service and response-encoding time for every request.
fn finish(
    io: ?std.Io,
    allocator: std.mem.Allocator,
    start_ns: u64,
    meta_in: wire.ResponseMeta,
    payload: []const u8,
) ServerError![]u8 {
    var meta = meta_in;
    meta.device_service_ns = elapsed(start_ns, nowNs(io));
    const encode_start = nowNs(io);
    const out = try encodeResponse(allocator, meta, payload);
    const encode_ns = elapsed(encode_start, nowNs(io));
    const off = framing.header_len + wire.device_encode_ns_offset;
    if (out.len >= off + 8)
        std.mem.writeInt(u64, out[off..][0..8], encode_ns, .little);
    return out;
}

fn nowNs(io: ?std.Io) u64 {
    const active = io orelse return 0;
    const nanoseconds = std.Io.Timestamp.now(active, .awake).nanoseconds;
    return std.math.cast(u64, nanoseconds) orelse 0;
}

fn elapsed(start_ns: u64, end_ns: u64) u64 {
    return if (end_ns >= start_ns) end_ns - start_ns else 0;
}

fn encodeResponse(
    allocator: std.mem.Allocator,
    meta: wire.ResponseMeta,
    payload: []const u8,
) ServerError![]u8 {
    var meta_buf: [wire.response_meta_len]u8 = undefined;
    const meta_len = wire.encodeResponseMeta(&meta_buf, meta) catch unreachable;
    const frame_len = framing.encodedLen(meta_len, payload.len) catch return error.BadFrame;
    const out = try allocator.alloc(u8, frame_len);
    errdefer allocator.free(out);
    _ = framing.encode(meta_buf[0..meta_len], payload, out) catch unreachable;
    return out;
}

test "server frames a successful runtime response" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    var metadata: [32]u8 = undefined;
    const metadata_len = try wire.encodeHello(&metadata, 42);
    var request: [framing.header_len + metadata.len]u8 = undefined;
    const request_len = try framing.encode(
        metadata[0..metadata_len],
        "",
        &request,
    );

    const response_bytes = try handleFrame(
        std.testing.io,
        std.testing.allocator,
        &runtime,
        request[0..request_len],
    );
    defer std.testing.allocator.free(response_bytes);
    const response = try framing.decode(response_bytes);
    const meta = try wire.decodeResponseMeta(response.metadata);
    try std.testing.expectEqual(@as(u64, 42), meta.request_id);
    try std.testing.expectEqual(wire.Status.ok, meta.status);
    try std.testing.expectEqual(@as(usize, 0), response.payload.len);
}

test "server turns malformed frames into protocol errors" {
    var runtime = try runtime_mod.Runtime.init(std.testing.allocator, 4096);
    defer runtime.deinit();

    const response_bytes = try handleFrame(
        std.testing.io,
        std.testing.allocator,
        &runtime,
        "not a frame",
    );
    defer std.testing.allocator.free(response_bytes);
    const response = try framing.decode(response_bytes);
    const meta = try wire.decodeResponseMeta(response.metadata);
    try std.testing.expectEqual(wire.Status.failed, meta.status);
    try std.testing.expectEqual(wire.ErrorCode.invalid_request, meta.error_code);
}
