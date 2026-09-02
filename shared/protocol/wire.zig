//! Versioned host-to-daemon control protocol for the resident inference engine.
//!
//! Model execution is intentionally opaque here. The only data-plane operation
//! is `inference`, whose payload is defined by `shared/engine/rpc.zig`.
const std = @import("std");

pub const version: u16 = 18;
pub const response_meta_len: usize = 64;
pub const device_encode_ns_offset: usize = response_meta_len - @sizeOf(u64);

pub const RequestTag = enum(u16) {
    hello = 1,
    alloc = 2,
    free = 3,
    upload = 4,
    capabilities = 8,
    inference = 9,
};

pub const InferenceAction = enum(u16) {
    install_model = 1,
    uninstall_model = 2,
    open_session = 3,
    close_session = 4,
    reset_session = 5,
    execute = 6,
};

pub const Status = enum(u16) {
    ok = 0,
    failed = 1,
};

pub const ErrorCode = enum(u16) {
    none = 0,
    invalid_request = 1,
    out_of_memory = 2,
    unknown_handle = 3,
    out_of_bounds = 4,
    unsupported_op = 5,
    backend_failure = 6,
};

pub const DecodeError = error{
    Truncated,
    UnsupportedVersion,
    InvalidTag,
    InvalidStatus,
    InvalidErrorCode,
    InvalidLength,
    InvalidFlags,
};

pub const EncodeError = error{OutputTooSmall};

pub const TensorRange = struct {
    handle: u64,
    offset: u64,
    nbytes: u64,
};

pub const AllocRequest = struct {
    request_id: u64,
    nbytes: u64,
    alignment: u32,
};

pub const FreeRequest = struct {
    request_id: u64,
    handle: u64,
};

pub const TransferRequest = struct {
    request_id: u64,
    range: TensorRange,
    bytes: []const u8 = &.{},
};

pub const InferenceRequest = struct {
    request_id: u64,
    action: InferenceAction,
    bytes: []const u8,
};

pub const Request = union(RequestTag) {
    hello: u64,
    alloc: AllocRequest,
    free: FreeRequest,
    upload: TransferRequest,
    capabilities: u64,
    inference: InferenceRequest,

    pub fn requestId(self: Request) u64 {
        return switch (self) {
            .hello, .capabilities => |request_id| request_id,
            .alloc => |request| request.request_id,
            .free => |request| request.request_id,
            .upload => |request| request.request_id,
            .inference => |request| request.request_id,
        };
    }
};

pub const ResponseMeta = struct {
    request_id: u64,
    status: Status,
    error_code: ErrorCode = .none,
    handle: u64 = 0,
    nbytes: u64 = 0,
    value0: u64 = 0,
    value1: u64 = 0,
    device_service_ns: u64 = 0,
    device_encode_ns: u64 = 0,
};

pub fn encodeHello(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .hello, request_id);
}

pub fn encodeCapabilities(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .capabilities, request_id);
}

pub fn encodeAlloc(
    out: []u8,
    request_id: u64,
    nbytes: u64,
    alignment: u32,
) EncodeError!usize {
    const len = 32;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .alloc, request_id);
    putU64(out, &cursor, nbytes);
    putU32(out, &cursor, alignment);
    putU32(out, &cursor, 0);
    return cursor;
}

pub fn encodeFree(out: []u8, request_id: u64, handle: u64) EncodeError!usize {
    const len = 24;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .free, request_id);
    putU64(out, &cursor, handle);
    return cursor;
}

pub fn encodeUpload(
    out: []u8,
    request_id: u64,
    range: TensorRange,
) EncodeError!usize {
    return encodeRangeRequest(out, .upload, request_id, range);
}

pub fn encodeInference(
    out: []u8,
    request_id: u64,
    action: InferenceAction,
) EncodeError!usize {
    const len = 24;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .inference, request_id);
    putU16(out, &cursor, @intFromEnum(action));
    putU16(out, &cursor, 0);
    putU32(out, &cursor, 0);
    return cursor;
}

pub fn decodeRequest(metadata: []const u8, payload: []const u8) DecodeError!Request {
    var cursor: usize = 0;
    if (try takeU16(metadata, &cursor) != version)
        return error.UnsupportedVersion;
    const raw_tag = try takeU16(metadata, &cursor);
    if (try takeU32(metadata, &cursor) != 0) return error.InvalidFlags;
    const request_id = try takeU64(metadata, &cursor);
    const tag = enumFromInt(RequestTag, raw_tag) orelse return error.InvalidTag;

    return switch (tag) {
        .hello => blk: {
            try expectEnd(metadata, cursor, payload.len);
            break :blk .{ .hello = request_id };
        },
        .capabilities => blk: {
            try expectEnd(metadata, cursor, payload.len);
            break :blk .{ .capabilities = request_id };
        },
        .alloc => blk: {
            const nbytes = try takeU64(metadata, &cursor);
            const alignment = try takeU32(metadata, &cursor);
            if (try takeU32(metadata, &cursor) != 0) return error.InvalidFlags;
            try expectEnd(metadata, cursor, payload.len);
            break :blk .{ .alloc = .{
                .request_id = request_id,
                .nbytes = nbytes,
                .alignment = alignment,
            } };
        },
        .free => blk: {
            const handle = try takeU64(metadata, &cursor);
            try expectEnd(metadata, cursor, payload.len);
            break :blk .{ .free = .{
                .request_id = request_id,
                .handle = handle,
            } };
        },
        .upload => blk: {
            const range = try takeRange(metadata, &cursor);
            if (cursor != metadata.len or
                payload.len != try checkedUsize(range.nbytes))
            {
                return error.InvalidLength;
            }
            break :blk .{ .upload = .{
                .request_id = request_id,
                .range = range,
                .bytes = payload,
            } };
        },
        .inference => blk: {
            const action = enumFromInt(
                InferenceAction,
                try takeU16(metadata, &cursor),
            ) orelse return error.InvalidTag;
            if (try takeU16(metadata, &cursor) != 0 or
                try takeU32(metadata, &cursor) != 0)
            {
                return error.InvalidFlags;
            }
            if (cursor != metadata.len) return error.InvalidLength;
            break :blk .{ .inference = .{
                .request_id = request_id,
                .action = action,
                .bytes = payload,
            } };
        },
    };
}

pub fn encodeResponseMeta(out: []u8, meta: ResponseMeta) EncodeError!usize {
    if (out.len < response_meta_len) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU16(out, &cursor, version);
    putU16(out, &cursor, @intFromEnum(meta.status));
    putU16(out, &cursor, @intFromEnum(meta.error_code));
    putU16(out, &cursor, 0);
    putU64(out, &cursor, meta.request_id);
    putU64(out, &cursor, meta.handle);
    putU64(out, &cursor, meta.nbytes);
    putU64(out, &cursor, meta.value0);
    putU64(out, &cursor, meta.value1);
    putU64(out, &cursor, meta.device_service_ns);
    putU64(out, &cursor, meta.device_encode_ns);
    return cursor;
}

pub fn decodeResponseMeta(bytes: []const u8) DecodeError!ResponseMeta {
    if (bytes.len != response_meta_len) return error.InvalidLength;
    var cursor: usize = 0;
    if (try takeU16(bytes, &cursor) != version)
        return error.UnsupportedVersion;
    const status = enumFromInt(Status, try takeU16(bytes, &cursor)) orelse
        return error.InvalidStatus;
    const error_code = enumFromInt(
        ErrorCode,
        try takeU16(bytes, &cursor),
    ) orelse return error.InvalidErrorCode;
    if (try takeU16(bytes, &cursor) != 0) return error.InvalidFlags;
    return .{
        .request_id = try takeU64(bytes, &cursor),
        .status = status,
        .error_code = error_code,
        .handle = try takeU64(bytes, &cursor),
        .nbytes = try takeU64(bytes, &cursor),
        .value0 = try takeU64(bytes, &cursor),
        .value1 = try takeU64(bytes, &cursor),
        .device_service_ns = try takeU64(bytes, &cursor),
        .device_encode_ns = try takeU64(bytes, &cursor),
    };
}

const range_len: usize = 24;

fn encodeHeader(out: []u8, tag: RequestTag, request_id: u64) EncodeError!usize {
    const len = 16;
    if (out.len < len) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU16(out, &cursor, version);
    putU16(out, &cursor, @intFromEnum(tag));
    putU32(out, &cursor, 0);
    putU64(out, &cursor, request_id);
    return cursor;
}

fn encodeRangeRequest(
    out: []u8,
    tag: RequestTag,
    request_id: u64,
    range: TensorRange,
) EncodeError!usize {
    const len = 16 + range_len;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, tag, request_id);
    putRange(out, &cursor, range);
    return cursor;
}

fn expectEnd(metadata: []const u8, cursor: usize, payload_len: usize) DecodeError!void {
    if (cursor != metadata.len or payload_len != 0) return error.InvalidLength;
}

fn checkedUsize(value: u64) DecodeError!usize {
    return std.math.cast(usize, value) orelse error.InvalidLength;
}

fn enumFromInt(comptime E: type, value: anytype) ?E {
    inline for (std.meta.fields(E)) |field| {
        if (value == field.value) return @enumFromInt(field.value);
    }
    return null;
}

fn putRange(out: []u8, cursor: *usize, range: TensorRange) void {
    putU64(out, cursor, range.handle);
    putU64(out, cursor, range.offset);
    putU64(out, cursor, range.nbytes);
}

fn takeRange(bytes: []const u8, cursor: *usize) DecodeError!TensorRange {
    return .{
        .handle = try takeU64(bytes, cursor),
        .offset = try takeU64(bytes, cursor),
        .nbytes = try takeU64(bytes, cursor),
    };
}

fn putU16(out: []u8, cursor: *usize, value: u16) void {
    std.mem.writeInt(u16, out[cursor.*..][0..2], value, .little);
    cursor.* += 2;
}

fn putU32(out: []u8, cursor: *usize, value: u32) void {
    std.mem.writeInt(u32, out[cursor.*..][0..4], value, .little);
    cursor.* += 4;
}

fn putU64(out: []u8, cursor: *usize, value: u64) void {
    std.mem.writeInt(u64, out[cursor.*..][0..8], value, .little);
    cursor.* += 8;
}

fn takeU16(bytes: []const u8, cursor: *usize) DecodeError!u16 {
    if (bytes.len - cursor.* < 2) return error.Truncated;
    const value = std.mem.readInt(u16, bytes[cursor.*..][0..2], .little);
    cursor.* += 2;
    return value;
}

fn takeU32(bytes: []const u8, cursor: *usize) DecodeError!u32 {
    if (bytes.len - cursor.* < 4) return error.Truncated;
    const value = std.mem.readInt(u32, bytes[cursor.*..][0..4], .little);
    cursor.* += 4;
    return value;
}

fn takeU64(bytes: []const u8, cursor: *usize) DecodeError!u64 {
    if (bytes.len - cursor.* < 8) return error.Truncated;
    const value = std.mem.readInt(u64, bytes[cursor.*..][0..8], .little);
    cursor.* += 8;
    return value;
}

test "control requests roundtrip" {
    var metadata: [64]u8 = undefined;

    const hello_len = try encodeHello(&metadata, 1);
    try std.testing.expectEqual(@as(u64, 1), (try decodeRequest(
        metadata[0..hello_len],
        "",
    )).hello);

    const alloc_len = try encodeAlloc(&metadata, 2, 4096, 64);
    const allocation = (try decodeRequest(metadata[0..alloc_len], "")).alloc;
    try std.testing.expectEqual(@as(u64, 4096), allocation.nbytes);
    try std.testing.expectEqual(@as(u32, 64), allocation.alignment);

    const free_len = try encodeFree(&metadata, 3, 7);
    try std.testing.expectEqual(@as(u64, 7), (try decodeRequest(
        metadata[0..free_len],
        "",
    )).free.handle);

    const range: TensorRange = .{ .handle = 7, .offset = 64, .nbytes = 4 };
    const upload_len = try encodeUpload(&metadata, 4, range);
    const upload = (try decodeRequest(
        metadata[0..upload_len],
        &.{ 1, 2, 3, 4 },
    )).upload;
    try std.testing.expectEqual(range, upload.range);

    const capabilities_len = try encodeCapabilities(&metadata, 5);
    try std.testing.expectEqual(@as(u64, 5), (try decodeRequest(
        metadata[0..capabilities_len],
        "",
    )).capabilities);
}

test "inference request carries a bounded action and opaque payload" {
    var metadata: [24]u8 = undefined;
    const metadata_len = try encodeInference(&metadata, 77, .execute);
    const payload = [_]u8{ 1, 2, 3, 4 };
    const request = (try decodeRequest(
        metadata[0..metadata_len],
        &payload,
    )).inference;
    try std.testing.expectEqual(@as(u64, 77), request.request_id);
    try std.testing.expectEqual(InferenceAction.execute, request.action);
    try std.testing.expectEqualSlices(u8, &payload, request.bytes);

    var invalid = metadata;
    invalid[18] = 1;
    try std.testing.expectError(
        error.InvalidFlags,
        decodeRequest(invalid[0..metadata_len], &payload),
    );
}

test "unknown request tag is rejected" {
    var metadata: [16]u8 = undefined;
    var cursor: usize = 0;
    putU16(&metadata, &cursor, version);
    putU16(&metadata, &cursor, 6);
    putU32(&metadata, &cursor, 0);
    putU64(&metadata, &cursor, 1);
    try std.testing.expectError(error.InvalidTag, decodeRequest(&metadata, ""));
}

test "response metadata roundtrips and pins encode timing offset" {
    var bytes: [response_meta_len]u8 = undefined;
    const expected: ResponseMeta = .{
        .request_id = 42,
        .status = .ok,
        .handle = 7,
        .nbytes = 4096,
        .value0 = 3,
        .device_service_ns = 123_456,
        .device_encode_ns = 789,
    };
    try std.testing.expectEqual(response_meta_len, try encodeResponseMeta(
        &bytes,
        expected,
    ));
    try std.testing.expectEqual(expected, try decodeResponseMeta(&bytes));
    try std.testing.expectEqual(
        expected.device_encode_ns,
        std.mem.readInt(
            u64,
            bytes[device_encode_ns_offset..][0..8],
            .little,
        ),
    );
}
