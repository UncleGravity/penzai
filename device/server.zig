const std = @import("std");
const build_options = @import("build_options");
const shared = @import("shared");
const runtime_mod = @import("runtime");

const framing = shared.framing;
const wire = shared.wire;
const profiling = shared.profiling;

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
    // Device service time brackets request decode + dispatch for every request,
    // so the host can split any op's round trip into device vs transport. Gated
    // by enable_profiling to keep the `-Dprofiling=false` zero-cost contract.
    const start_ns = if (build_options.enable_profiling) profiling.nowNs(io) else 0;

    const frame = framing.decode(frame_bytes) catch
        return finish(io, allocator, start_ns, errorMeta(0, .invalid_request), "");
    const request = wire.decodeRequest(frame.metadata, frame.payload) catch
        return finish(io, allocator, start_ns, errorMeta(0, .invalid_request), "");
    var result = runtime.dispatch(request, io) catch |err|
        return finish(io, allocator, start_ns, errorMeta(request.requestId(), runtime_mod.errorCode(err)), "");
    defer result.deinit(allocator);
    return finish(io, allocator, start_ns, result.meta, result.payload);
}

fn errorMeta(request_id: u64, code: wire.ErrorCode) wire.ResponseMeta {
    return .{ .request_id = request_id, .status = .failed, .error_code = code };
}

/// Single exit: stamp device service time (if profiling) and frame the response.
fn finish(
    io: ?std.Io,
    allocator: std.mem.Allocator,
    start_ns: u64,
    meta_in: wire.ResponseMeta,
    payload: []const u8,
) ServerError![]u8 {
    var meta = meta_in;
    if (build_options.enable_profiling) {
        meta.device_service_ns = profiling.elapsed(start_ns, profiling.nowNs(io));
    }
    const encode_start = if (build_options.enable_profiling) profiling.nowNs(io) else 0;
    const out = try encodeResponse(allocator, meta, payload);
    // device_service_ns brackets request servicing only; the response framing +
    // payload copy (large for downloads and the profile payload) happen here. Stamp
    // them by patching device_encode_ns — the meta's last u64, at a fixed offset in
    // the framed buffer — so the host can keep its transport bucket as pure wire time.
    if (build_options.enable_profiling) {
        const encode_ns = profiling.elapsed(encode_start, profiling.nowNs(io));
        const off = framing.header_len + wire.response_meta_len - 8;
        if (out.len >= off + 8) std.mem.writeInt(u64, out[off..][0..8], encode_ns, .little);
    }
    return out;
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
