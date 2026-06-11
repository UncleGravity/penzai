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
    const frame = framing.decode(frame_bytes) catch return error.BadFrame;
    const request = wire.decodeRequest(frame.metadata, frame.payload) catch return encodeError(
        allocator,
        0,
        .invalid_request,
    );
    var result = runtime.dispatch(request, io) catch |err| return encodeError(
        allocator,
        request.requestId(),
        runtime_mod.errorCode(err),
    );
    defer result.deinit(allocator);
    return encodeResponse(allocator, result.meta, result.payload);
}

fn encodeError(
    allocator: std.mem.Allocator,
    request_id: u64,
    code: wire.ErrorCode,
) ServerError![]u8 {
    return encodeResponse(allocator, .{
        .request_id = request_id,
        .status = .failed,
        .error_code = code,
    }, "");
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
