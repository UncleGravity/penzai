const std = @import("std");

pub const version: u16 = 1;
pub const response_meta_len: usize = 48;

pub const RequestTag = enum(u16) {
    hello = 1,
    alloc = 2,
    free = 3,
    upload = 4,
    download = 5,
    run_graph = 6,
};

pub const OpTag = enum(u16) {
    copy = 1,
    matmul_q1a8 = 2,
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
    TrailingBytes,
};

pub const EncodeError = error{ OutputTooSmall, TooManyCommands };

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

pub const RunGraphRequest = struct {
    request_id: u64,
    command_bytes: []const u8,
};

pub const Request = union(RequestTag) {
    hello: u64,
    alloc: AllocRequest,
    free: FreeRequest,
    upload: TransferRequest,
    download: TransferRequest,
    run_graph: RunGraphRequest,

    pub fn requestId(self: Request) u64 {
        return switch (self) {
            .hello => |id| id,
            .alloc => |r| r.request_id,
            .free => |r| r.request_id,
            .upload => |r| r.request_id,
            .download => |r| r.request_id,
            .run_graph => |r| r.request_id,
        };
    }
};

pub const Copy = struct {
    src: TensorRange,
    dst: TensorRange,
};

pub const MatmulQ1A8 = struct {
    weights: TensorRange,
    acts: TensorRange,
    dst: TensorRange,
    rows: u32,
    cols: u32,
    k: u32,
};

pub const Command = union(OpTag) {
    copy: Copy,
    matmul_q1a8: MatmulQ1A8,
};

pub const ResponseMeta = struct {
    request_id: u64,
    status: Status,
    error_code: ErrorCode = .none,
    handle: u64 = 0,
    nbytes: u64 = 0,
    value0: u64 = 0,
    value1: u64 = 0,
};

pub fn encodeHello(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .hello, request_id);
}

pub fn encodeAlloc(out: []u8, request_id: u64, nbytes: u64, alignment: u32) EncodeError!usize {
    const len = 28;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .alloc, request_id);
    putU64(out, &cursor, nbytes);
    putU32(out, &cursor, alignment);
    putU32(out, &cursor, 0);
    return cursor;
}

pub fn encodeFree(out: []u8, request_id: u64, handle: u64) EncodeError!usize {
    const len = 20;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .free, request_id);
    putU64(out, &cursor, handle);
    return cursor;
}

pub fn encodeUpload(out: []u8, request_id: u64, range: TensorRange) EncodeError!usize {
    return encodeRangeRequest(out, .upload, request_id, range);
}

pub fn encodeDownload(out: []u8, request_id: u64, range: TensorRange) EncodeError!usize {
    return encodeRangeRequest(out, .download, request_id, range);
}

pub fn encodeRunGraph(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .run_graph, request_id);
}

pub fn decodeRequest(metadata: []const u8, payload: []const u8) DecodeError!Request {
    var cursor: usize = 0;
    const ver = try takeU16(metadata, &cursor);
    if (ver != version) return error.UnsupportedVersion;
    const raw_tag = try takeU16(metadata, &cursor);
    _ = try takeU32(metadata, &cursor);
    const request_id = try takeU64(metadata, &cursor);
    const tag = enumFromInt(RequestTag, raw_tag) orelse return error.InvalidTag;

    return switch (tag) {
        .hello => blk: {
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            break :blk .{ .hello = request_id };
        },
        .alloc => blk: {
            const nbytes = try takeU64(metadata, &cursor);
            const alignment = try takeU32(metadata, &cursor);
            _ = try takeU32(metadata, &cursor);
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            break :blk .{ .alloc = .{ .request_id = request_id, .nbytes = nbytes, .alignment = alignment } };
        },
        .free => blk: {
            const handle = try takeU64(metadata, &cursor);
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            break :blk .{ .free = .{ .request_id = request_id, .handle = handle } };
        },
        .upload => blk: {
            const range = try takeRange(metadata, &cursor);
            if (cursor != metadata.len or payload.len != try checkedUsize(range.nbytes)) return error.InvalidLength;
            break :blk .{ .upload = .{ .request_id = request_id, .range = range, .bytes = payload } };
        },
        .download => blk: {
            const range = try takeRange(metadata, &cursor);
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            break :blk .{ .download = .{ .request_id = request_id, .range = range } };
        },
        .run_graph => blk: {
            if (cursor != metadata.len) return error.InvalidLength;
            break :blk .{ .run_graph = .{ .request_id = request_id, .command_bytes = payload } };
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
    return cursor;
}

pub fn decodeResponseMeta(bytes: []const u8) DecodeError!ResponseMeta {
    if (bytes.len != response_meta_len) return error.InvalidLength;
    var cursor: usize = 0;
    const ver = try takeU16(bytes, &cursor);
    if (ver != version) return error.UnsupportedVersion;
    const status = enumFromInt(Status, try takeU16(bytes, &cursor)) orelse return error.InvalidStatus;
    const error_code = enumFromInt(ErrorCode, try takeU16(bytes, &cursor)) orelse return error.InvalidErrorCode;
    _ = try takeU16(bytes, &cursor);
    return .{
        .request_id = try takeU64(bytes, &cursor),
        .status = status,
        .error_code = error_code,
        .handle = try takeU64(bytes, &cursor),
        .nbytes = try takeU64(bytes, &cursor),
        .value0 = try takeU64(bytes, &cursor),
        .value1 = try takeU64(bytes, &cursor),
    };
}

pub fn commandBufferLen(commands: []const Command) EncodeError!usize {
    if (commands.len > std.math.maxInt(u32)) return error.TooManyCommands;
    var len: usize = 4;
    for (commands) |command| {
        len += switch (command) {
            .copy => 4 + rangeLen * 2,
            .matmul_q1a8 => 4 + rangeLen * 3 + 16,
        };
    }
    return len;
}

pub fn encodeCommandBuffer(commands: []const Command, out: []u8) EncodeError!usize {
    const want = try commandBufferLen(commands);
    if (out.len < want) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU32(out, &cursor, @intCast(commands.len));
    for (commands) |command| switch (command) {
        .copy => |copy| {
            putU16(out, &cursor, @intFromEnum(OpTag.copy));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, copy.src);
            putRange(out, &cursor, copy.dst);
        },
        .matmul_q1a8 => |matmul| {
            putU16(out, &cursor, @intFromEnum(OpTag.matmul_q1a8));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, matmul.weights);
            putRange(out, &cursor, matmul.acts);
            putRange(out, &cursor, matmul.dst);
            putU32(out, &cursor, matmul.rows);
            putU32(out, &cursor, matmul.cols);
            putU32(out, &cursor, matmul.k);
            putU32(out, &cursor, 0);
        },
    };
    return cursor;
}

pub fn decodeCommandBuffer(allocator: std.mem.Allocator, bytes: []const u8) (DecodeError || error{OutOfMemory})![]Command {
    var cursor: usize = 0;
    const count = try takeU32(bytes, &cursor);
    const commands = try allocator.alloc(Command, @intCast(count));
    errdefer allocator.free(commands);

    for (commands) |*command| {
        const raw_tag = try takeU16(bytes, &cursor);
        _ = try takeU16(bytes, &cursor);
        const tag = enumFromInt(OpTag, raw_tag) orelse return error.InvalidTag;
        command.* = switch (tag) {
            .copy => .{ .copy = .{ .src = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
            .matmul_q1a8 => blk: {
                const weights = try takeRange(bytes, &cursor);
                const acts = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const k = try takeU32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .matmul_q1a8 = .{ .weights = weights, .acts = acts, .dst = dst, .rows = rows, .cols = cols, .k = k } };
            },
        };
    }
    if (cursor != bytes.len) return error.TrailingBytes;
    return commands;
}

const rangeLen: usize = 24;

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

fn encodeRangeRequest(out: []u8, tag: RequestTag, request_id: u64, range: TensorRange) EncodeError!usize {
    const len = 16 + rangeLen;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, tag, request_id);
    putRange(out, &cursor, range);
    return cursor;
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

fn checkedUsize(value: u64) DecodeError!usize {
    if (value > std.math.maxInt(usize)) return error.InvalidLength;
    return @intCast(value);
}

fn enumFromInt(comptime E: type, value: anytype) ?E {
    inline for (std.meta.fields(E)) |field| {
        if (value == field.value) return @enumFromInt(field.value);
    }
    return null;
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

test "alloc request roundtrip" {
    var meta: [64]u8 = undefined;
    const n = try encodeAlloc(&meta, 7, 4096, 64);
    const req = try decodeRequest(meta[0..n], "");
    try std.testing.expectEqual(@as(u64, 7), req.requestId());
    try std.testing.expectEqual(@as(u64, 4096), req.alloc.nbytes);
    try std.testing.expectEqual(@as(u32, 64), req.alloc.alignment);
}

test "command buffer roundtrip" {
    const commands = [_]Command{
        .{ .copy = .{
            .src = .{ .handle = 1, .offset = 2, .nbytes = 3 },
            .dst = .{ .handle = 4, .offset = 5, .nbytes = 3 },
        } },
        .{ .matmul_q1a8 = .{
            .weights = .{ .handle = 1, .offset = 0, .nbytes = 144 },
            .acts = .{ .handle = 2, .offset = 0, .nbytes = 512 },
            .dst = .{ .handle = 3, .offset = 0, .nbytes = 32 },
            .rows = 8,
            .cols = 1,
            .k = 128,
        } },
    };
    var buf: [256]u8 = undefined;
    const n = try encodeCommandBuffer(&commands, &buf);
    const got = try decodeCommandBuffer(std.testing.allocator, buf[0..n]);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqual(commands[0].copy.src.handle, got[0].copy.src.handle);
    try std.testing.expectEqual(commands[1].matmul_q1a8.k, got[1].matmul_q1a8.k);
}
