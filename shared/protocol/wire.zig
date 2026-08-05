const std = @import("std");

pub const version: u16 = 13;
pub const response_meta_len: usize = 64;

pub const RequestTag = enum(u16) {
    hello = 1,
    alloc = 2,
    free = 3,
    upload = 4,
    download = 5,
    run_graph = 6,
    fill = 7,
    capabilities = 8,
};

pub const OpTag = enum(u16) {
    copy = 1,
    matmul_q1a8 = 2,
    rmsnorm = 3,
    rope = 4,
    softmax = 5,
    silu = 6,
    swiglu = 7,
    add_f32 = 8,
    mul_f32 = 9,
    scale_f32 = 10,
    add_scaled_f32 = 11,
    set_rows = 12,
    get_rows = 13,
    flash_attn_f32 = 14,
    cpy_f32_to_f16 = 15,
    argmax = 16,
    pad = 17,
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
    TrailingBytes,
};

pub const EncodeError = error{ OutputTooSmall, TooManyCommands, InvalidLength };

pub const TensorRange = struct {
    handle: u64,
    offset: u64,
    nbytes: u64,
};

pub const IndexType = enum(u32) {
    i32 = 1,
    i64 = 2,
};

pub const GetRowsSrcType = enum(u32) {
    f32 = 1,
    q1_0 = 2,
    q2_0 = 3,
};

pub const BinaryF32Mode = enum(u32) {
    same_shape = 1,
    rhs_row_broadcast = 2,
};

pub const RopeMode = enum(u32) {
    normal = 1,
    neox = 2,
};

/// Weight encoding for a matmul. Tags the command so the device picks the
/// decoder/kernel path and the host keys per-format profiling. `w1a8` is the
/// Runtime weight format selected by each matmul command.
pub const WeightFormat = enum(u32) {
    w1a8 = 1,
    w158a8 = 2,
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

pub const FillRequest = struct {
    request_id: u64,
    range: TensorRange,
    value: u8,
};

/// What a run_graph request asks the device to collect. `aggregate` accrues
/// per-op totals (cheap, fixed size); `off` collects nothing. Wire flag bit0 =
/// aggregate; all other bit patterns are reserved and rejected.
pub const ProfileTier = enum(u8) {
    off,
    aggregate,

    fn flags(self: ProfileTier) u32 {
        return switch (self) {
            .off => 0,
            .aggregate => 0b01,
        };
    }

    fn fromFlags(value: u32) ?ProfileTier {
        return switch (value) {
            0 => .off,
            0b01 => .aggregate,
            else => null,
        };
    }
};

pub const RunGraphRequest = struct {
    request_id: u64,
    command_bytes: []const u8,
    tier: ProfileTier = .off,
    /// Tensor writes to apply immediately before executing the command buffer.
    /// Empty means no folded uploads. The payload format is parsed separately so
    /// the command buffer remains the same closed op list used by the runtime.
    preload_bytes: []const u8 = &.{},
};

/// One folded upload in a run_graph request: destination range plus exact bytes.
pub const PreloadEntry = struct {
    range: TensorRange,
    bytes: []const u8,
};

pub const PreloadIterator = struct {
    bytes: []const u8,
    cursor: usize = 0,

    pub fn next(self: *PreloadIterator) DecodeError!?PreloadEntry {
        if (self.cursor == self.bytes.len) return null;
        const range = try takeRange(self.bytes, &self.cursor);
        const n = try checkedUsize(range.nbytes);
        if (self.bytes.len - self.cursor < n) return error.Truncated;
        const payload = self.bytes[self.cursor..][0..n];
        self.cursor += n;
        return .{ .range = range, .bytes = payload };
    }
};

pub fn preloadEntryLen(nbytes: usize) EncodeError!usize {
    return std.math.add(usize, rangeLen, nbytes) catch error.OutputTooSmall;
}

pub fn encodePreloadEntry(out: []u8, range: TensorRange, bytes: []const u8) EncodeError!usize {
    if (range.nbytes != bytes.len) return error.InvalidLength;
    const want = try preloadEntryLen(bytes.len);
    if (out.len < want) return error.OutputTooSmall;
    var cursor: usize = 0;
    putRange(out, &cursor, range);
    @memcpy(out[cursor..][0..bytes.len], bytes);
    return want;
}

pub fn runGraphPayloadLen(preload_len: usize, command_len: usize) EncodeError!usize {
    const with_commands = std.math.add(usize, preload_len, command_len) catch return error.OutputTooSmall;
    return std.math.add(usize, 4, with_commands) catch error.OutputTooSmall;
}

/// run_graph payload = u32 preload_len, then preload entries, then command bytes.
pub fn encodeRunGraphPayload(out: []u8, preload_bytes: []const u8, command_bytes: []const u8) EncodeError!usize {
    if (preload_bytes.len > std.math.maxInt(u32)) return error.OutputTooSmall;
    const want = try runGraphPayloadLen(preload_bytes.len, command_bytes.len);
    if (out.len < want) return error.OutputTooSmall;
    var cursor: usize = 0;
    putU32(out, &cursor, @intCast(preload_bytes.len));
    @memcpy(out[cursor..][0..preload_bytes.len], preload_bytes);
    cursor += preload_bytes.len;
    @memcpy(out[cursor..][0..command_bytes.len], command_bytes);
    return want;
}

pub const Request = union(RequestTag) {
    hello: u64,
    alloc: AllocRequest,
    free: FreeRequest,
    upload: TransferRequest,
    download: TransferRequest,
    run_graph: RunGraphRequest,
    fill: FillRequest,
    capabilities: u64,

    pub fn requestId(self: Request) u64 {
        return switch (self) {
            .hello => |id| id,
            .alloc => |r| r.request_id,
            .free => |r| r.request_id,
            .upload => |r| r.request_id,
            .download => |r| r.request_id,
            .run_graph => |r| r.request_id,
            .fill => |r| r.request_id,
            .capabilities => |id| id,
        };
    }
};

pub const Copy = struct {
    src: TensorRange,
    dst: TensorRange,
};

pub const CpyF32ToF16 = struct {
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
    weight_fmt: WeightFormat = .w1a8,
};

pub const UnaryF32 = struct {
    src: TensorRange,
    dst: TensorRange,
};

pub const BinaryF32 = struct {
    lhs: TensorRange,
    rhs: TensorRange,
    dst: TensorRange,
};

pub const BinaryBroadcastF32 = struct {
    lhs: TensorRange,
    rhs: TensorRange,
    dst: TensorRange,
    rows: u32,
    cols: u32,
    mode: BinaryF32Mode,
};

pub const RmsNorm = struct {
    input: TensorRange,
    weight: TensorRange = .{ .handle = 0, .offset = 0, .nbytes = 0 },
    dst: TensorRange,
    rows: u32,
    cols: u32,
    eps: f32,
    has_weight: bool = false,
};

pub const Rope = struct {
    input: TensorRange,
    positions: TensorRange,
    dst: TensorRange,
    head_dim: u32,
    n_heads: u32,
    n_tokens: u32,
    n_dims: u32,
    mode: RopeMode,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
};

pub const FlashAttnF32 = struct {
    q: TensorRange,
    k: TensorRange,
    v: TensorRange,
    mask: TensorRange,
    dst: TensorRange,
    has_mask: bool,
    head_dim_q: u32,
    head_dim_v: u32,
    n_heads: u32,
    n_head_kv: u32,
    n_kv: u32,
    n_tokens: u32,
    scale: f32,
    q_nb1: u64,
    q_nb2: u64,
    k_nb1: u64,
    k_nb2: u64,
    v_nb1: u64,
    v_nb2: u64,
    mask_nb1: u64,
    dst_nb1: u64,
    dst_nb2: u64,
};

pub const ScaleF32 = struct {
    src: TensorRange,
    dst: TensorRange,
    scale: f32,
};

pub const AddScaledF32 = struct {
    lhs: TensorRange,
    rhs: TensorRange,
    dst: TensorRange,
    rhs_scale: f32,
};

pub const SetRows = struct {
    src: TensorRange,
    indices: TensorRange,
    dst: TensorRange,
    index_type: IndexType,
    head_dim: u32,
    ne01: u32,
    ne02: u32,
    ne03: u32,
    ne11: u32,
    ne12: u32,
    src_nb1: u64,
    src_nb2: u64,
    src_nb3: u64,
    indices_nb1: u64,
    indices_nb2: u64,
    dst_nb1: u64,
    dst_nb2: u64,
    dst_nb3: u64,
};

pub const GetRows = struct {
    src: TensorRange,
    indices: TensorRange,
    dst: TensorRange,
    src_type: GetRowsSrcType,
    row_width: u32,
    src_rows: u32,
    ne10: u32,
    ne11: u32,
    ne12: u32,
    /// Used as byte strides for .f32 sources. Quantized sources use their
    /// format-specific resident matmul layout instead of the raw ggml strides.
    src_nb1: u64,
    src_nb2: u64,
    src_nb3: u64,
    indices_nb1: u64,
    indices_nb2: u64,
    dst_nb1: u64,
    dst_nb2: u64,
    dst_nb3: u64,
};

/// f32 argmax along each contiguous row (ggml `GGML_OP_ARGMAX`): reduce `cols`
/// elements per row over `rows` rows, writing one i32 index per row. `pad` reuses
/// `UnaryF32` (src/dst ranges only).
pub const Argmax = struct {
    src: TensorRange,
    dst: TensorRange,
    rows: u32,
    cols: u32,
};

pub const Command = union(OpTag) {
    copy: Copy,
    matmul_q1a8: MatmulQ1A8,
    rmsnorm: RmsNorm,
    rope: Rope,
    softmax: UnaryF32,
    silu: UnaryF32,
    swiglu: BinaryF32,
    add_f32: BinaryBroadcastF32,
    mul_f32: BinaryBroadcastF32,
    scale_f32: ScaleF32,
    add_scaled_f32: AddScaledF32,
    set_rows: SetRows,
    get_rows: GetRows,
    flash_attn_f32: FlashAttnF32,
    cpy_f32_to_f16: CpyF32ToF16,
    argmax: Argmax,
    pad: UnaryF32,
};

pub const ResponseMeta = struct {
    request_id: u64,
    status: Status,
    error_code: ErrorCode = .none,
    handle: u64 = 0,
    nbytes: u64 = 0,
    value0: u64 = 0,
    value1: u64 = 0,
    /// Device wall time spent servicing this request (request decode + dispatch),
    /// in device-clock ns. Stamped by the device on every response so the host can
    /// split any op's round trip into device-service vs transport. Zero when the
    /// device was built with profiling disabled. See device/server.zig.
    device_service_ns: u64 = 0,
    /// Device time spent encoding + copying the response (framing + payload memcpy),
    /// in device-clock ns. Separate from device_service_ns (which brackets request
    /// servicing only), so without it the response-encode cost — large for downloads
    /// and the profile payload — would hide in the host's transport bucket. The host
    /// folds it into device time. Patched in post-encode; see device/server.zig.
    device_encode_ns: u64 = 0,
};

/// Byte offset of `device_encode_ns` inside an encoded ResponseMeta: it is the
/// last u64 `encodeResponseMeta` writes. The device stamps the field in place
/// after encoding (it can only measure encode time once encoding is done), at
/// `framing.header_len + device_encode_ns_offset` in the framed buffer. Pinned by
/// a golden test against the encoder, so adding or reordering a field fails the
/// test instead of silently corrupting the wire.
pub const device_encode_ns_offset: usize = response_meta_len - @sizeOf(u64);

pub fn encodeHello(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .hello, request_id);
}

pub fn encodeCapabilities(out: []u8, request_id: u64) EncodeError!usize {
    return encodeHeader(out, .capabilities, request_id);
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

pub fn encodeFill(out: []u8, request_id: u64, range: TensorRange, value: u8) EncodeError!usize {
    const len = 16 + rangeLen + 8;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .fill, request_id);
    putRange(out, &cursor, range);
    putU32(out, &cursor, value);
    putU32(out, &cursor, 0);
    return cursor;
}

pub fn encodeRunGraph(out: []u8, request_id: u64, tier: ProfileTier) EncodeError!usize {
    const len = 24;
    if (out.len < len) return error.OutputTooSmall;
    var cursor = try encodeHeader(out, .run_graph, request_id);
    putU32(out, &cursor, tier.flags());
    putU32(out, &cursor, 0);
    return cursor;
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
        .capabilities => blk: {
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            break :blk .{ .capabilities = request_id };
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
        .fill => blk: {
            const range = try takeRange(metadata, &cursor);
            const value = try takeU32(metadata, &cursor);
            const reserved = try takeU32(metadata, &cursor);
            if (cursor != metadata.len or payload.len != 0) return error.InvalidLength;
            if (value > std.math.maxInt(u8) or reserved != 0) return error.InvalidFlags;
            break :blk .{ .fill = .{ .request_id = request_id, .range = range, .value = @intCast(value) } };
        },
        .run_graph => blk: {
            var tier: ProfileTier = .off;
            if (cursor != metadata.len) {
                const flags = try takeU32(metadata, &cursor);
                const reserved = try takeU32(metadata, &cursor);
                if (cursor != metadata.len) return error.InvalidLength;
                if (reserved != 0) return error.InvalidFlags;
                tier = ProfileTier.fromFlags(flags) orelse return error.InvalidFlags;
            }
            var preload_bytes: []const u8 = &.{};
            var command_bytes: []const u8 = &.{};
            if (payload.len != 0) {
                var payload_cursor: usize = 0;
                const preload_len = try checkedUsize(try takeU32(payload, &payload_cursor));
                if (payload.len - payload_cursor < preload_len) return error.InvalidLength;
                preload_bytes = payload[payload_cursor..][0..preload_len];
                command_bytes = payload[payload_cursor + preload_len ..];
            }
            break :blk .{ .run_graph = .{
                .request_id = request_id,
                .command_bytes = command_bytes,
                .tier = tier,
                .preload_bytes = preload_bytes,
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
        .device_service_ns = try takeU64(bytes, &cursor),
        .device_encode_ns = try takeU64(bytes, &cursor),
    };
}

pub fn commandBufferLen(commands: []const Command) EncodeError!usize {
    if (commands.len > std.math.maxInt(u32)) return error.TooManyCommands;
    var len: usize = 4;
    for (commands) |command| {
        len += switch (command) {
            .copy => 4 + rangeLen * 2,
            .cpy_f32_to_f16 => 4 + rangeLen * 2,
            .matmul_q1a8 => 4 + rangeLen * 3 + 16,
            .rmsnorm => 4 + rangeLen * 3 + 16,
            .rope => 4 + rangeLen * 3 + 48,
            .softmax, .silu => 4 + rangeLen * 2,
            .swiglu => 4 + rangeLen * 3,
            .add_f32, .mul_f32 => 4 + rangeLen * 3 + 16,
            .scale_f32 => 4 + rangeLen * 2 + 8,
            .add_scaled_f32 => 4 + rangeLen * 3 + 8,
            .set_rows => 4 + rangeLen * 3 + 32 + 64,
            .get_rows => 4 + rangeLen * 3 + 32 + 64,
            .flash_attn_f32 => 4 + rangeLen * 5 + 32 + 72,
            .argmax => 4 + rangeLen * 2 + 8,
            .pad => 4 + rangeLen * 2,
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
        .cpy_f32_to_f16 => |copy| {
            putU16(out, &cursor, @intFromEnum(OpTag.cpy_f32_to_f16));
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
            putU32(out, &cursor, @intFromEnum(matmul.weight_fmt));
        },
        .rmsnorm => |rmsnorm| {
            putU16(out, &cursor, @intFromEnum(OpTag.rmsnorm));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, rmsnorm.input);
            putRange(out, &cursor, rmsnorm.weight);
            putRange(out, &cursor, rmsnorm.dst);
            putU32(out, &cursor, rmsnorm.rows);
            putU32(out, &cursor, rmsnorm.cols);
            putF32(out, &cursor, rmsnorm.eps);
            putU32(out, &cursor, @intFromBool(rmsnorm.has_weight));
        },
        .rope => |rope| {
            putU16(out, &cursor, @intFromEnum(OpTag.rope));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, rope.input);
            putRange(out, &cursor, rope.positions);
            putRange(out, &cursor, rope.dst);
            putU32(out, &cursor, rope.head_dim);
            putU32(out, &cursor, rope.n_heads);
            putU32(out, &cursor, rope.n_tokens);
            putU32(out, &cursor, rope.n_dims);
            putU32(out, &cursor, @intFromEnum(rope.mode));
            putU32(out, &cursor, rope.n_ctx_orig);
            putF32(out, &cursor, rope.freq_base);
            putF32(out, &cursor, rope.freq_scale);
            putF32(out, &cursor, rope.ext_factor);
            putF32(out, &cursor, rope.attn_factor);
            putF32(out, &cursor, rope.beta_fast);
            putF32(out, &cursor, rope.beta_slow);
        },
        .softmax => |unary| {
            putU16(out, &cursor, @intFromEnum(OpTag.softmax));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, unary.src);
            putRange(out, &cursor, unary.dst);
        },
        .silu => |unary| {
            putU16(out, &cursor, @intFromEnum(OpTag.silu));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, unary.src);
            putRange(out, &cursor, unary.dst);
        },
        .swiglu => |binary| {
            putU16(out, &cursor, @intFromEnum(OpTag.swiglu));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, binary.lhs);
            putRange(out, &cursor, binary.rhs);
            putRange(out, &cursor, binary.dst);
        },
        .add_f32 => |binary| {
            putU16(out, &cursor, @intFromEnum(OpTag.add_f32));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, binary.lhs);
            putRange(out, &cursor, binary.rhs);
            putRange(out, &cursor, binary.dst);
            putU32(out, &cursor, binary.rows);
            putU32(out, &cursor, binary.cols);
            putU32(out, &cursor, @intFromEnum(binary.mode));
            putU32(out, &cursor, 0);
        },
        .mul_f32 => |binary| {
            putU16(out, &cursor, @intFromEnum(OpTag.mul_f32));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, binary.lhs);
            putRange(out, &cursor, binary.rhs);
            putRange(out, &cursor, binary.dst);
            putU32(out, &cursor, binary.rows);
            putU32(out, &cursor, binary.cols);
            putU32(out, &cursor, @intFromEnum(binary.mode));
            putU32(out, &cursor, 0);
        },
        .scale_f32 => |scale| {
            putU16(out, &cursor, @intFromEnum(OpTag.scale_f32));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, scale.src);
            putRange(out, &cursor, scale.dst);
            putF32(out, &cursor, scale.scale);
            putU32(out, &cursor, 0);
        },
        .add_scaled_f32 => |add_scaled| {
            putU16(out, &cursor, @intFromEnum(OpTag.add_scaled_f32));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, add_scaled.lhs);
            putRange(out, &cursor, add_scaled.rhs);
            putRange(out, &cursor, add_scaled.dst);
            putF32(out, &cursor, add_scaled.rhs_scale);
            putU32(out, &cursor, 0);
        },
        .set_rows => |set_rows| {
            putU16(out, &cursor, @intFromEnum(OpTag.set_rows));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, set_rows.src);
            putRange(out, &cursor, set_rows.indices);
            putRange(out, &cursor, set_rows.dst);
            putU32(out, &cursor, @intFromEnum(set_rows.index_type));
            putU32(out, &cursor, set_rows.head_dim);
            putU32(out, &cursor, set_rows.ne01);
            putU32(out, &cursor, set_rows.ne02);
            putU32(out, &cursor, set_rows.ne03);
            putU32(out, &cursor, set_rows.ne11);
            putU32(out, &cursor, set_rows.ne12);
            putU32(out, &cursor, 0);
            putU64(out, &cursor, set_rows.src_nb1);
            putU64(out, &cursor, set_rows.src_nb2);
            putU64(out, &cursor, set_rows.src_nb3);
            putU64(out, &cursor, set_rows.indices_nb1);
            putU64(out, &cursor, set_rows.indices_nb2);
            putU64(out, &cursor, set_rows.dst_nb1);
            putU64(out, &cursor, set_rows.dst_nb2);
            putU64(out, &cursor, set_rows.dst_nb3);
        },
        .get_rows => |get_rows| {
            putU16(out, &cursor, @intFromEnum(OpTag.get_rows));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, get_rows.src);
            putRange(out, &cursor, get_rows.indices);
            putRange(out, &cursor, get_rows.dst);
            putU32(out, &cursor, @intFromEnum(get_rows.src_type));
            putU32(out, &cursor, get_rows.row_width);
            putU32(out, &cursor, get_rows.src_rows);
            putU32(out, &cursor, get_rows.ne10);
            putU32(out, &cursor, get_rows.ne11);
            putU32(out, &cursor, get_rows.ne12);
            putU32(out, &cursor, 0);
            putU32(out, &cursor, 0);
            putU64(out, &cursor, get_rows.src_nb1);
            putU64(out, &cursor, get_rows.src_nb2);
            putU64(out, &cursor, get_rows.src_nb3);
            putU64(out, &cursor, get_rows.indices_nb1);
            putU64(out, &cursor, get_rows.indices_nb2);
            putU64(out, &cursor, get_rows.dst_nb1);
            putU64(out, &cursor, get_rows.dst_nb2);
            putU64(out, &cursor, get_rows.dst_nb3);
        },
        .flash_attn_f32 => |attn| {
            putU16(out, &cursor, @intFromEnum(OpTag.flash_attn_f32));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, attn.q);
            putRange(out, &cursor, attn.k);
            putRange(out, &cursor, attn.v);
            putRange(out, &cursor, attn.mask);
            putRange(out, &cursor, attn.dst);
            putU32(out, &cursor, if (attn.has_mask) 1 else 0);
            putU32(out, &cursor, attn.head_dim_q);
            putU32(out, &cursor, attn.head_dim_v);
            putU32(out, &cursor, attn.n_heads);
            putU32(out, &cursor, attn.n_head_kv);
            putU32(out, &cursor, attn.n_kv);
            putU32(out, &cursor, attn.n_tokens);
            putF32(out, &cursor, attn.scale);
            putU64(out, &cursor, attn.q_nb1);
            putU64(out, &cursor, attn.q_nb2);
            putU64(out, &cursor, attn.k_nb1);
            putU64(out, &cursor, attn.k_nb2);
            putU64(out, &cursor, attn.v_nb1);
            putU64(out, &cursor, attn.v_nb2);
            putU64(out, &cursor, attn.mask_nb1);
            putU64(out, &cursor, attn.dst_nb1);
            putU64(out, &cursor, attn.dst_nb2);
        },
        .argmax => |op| {
            putU16(out, &cursor, @intFromEnum(OpTag.argmax));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, op.src);
            putRange(out, &cursor, op.dst);
            putU32(out, &cursor, op.rows);
            putU32(out, &cursor, op.cols);
        },
        .pad => |unary| {
            putU16(out, &cursor, @intFromEnum(OpTag.pad));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, unary.src);
            putRange(out, &cursor, unary.dst);
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
            .cpy_f32_to_f16 => .{ .cpy_f32_to_f16 = .{ .src = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
            .matmul_q1a8 => blk: {
                const weights = try takeRange(bytes, &cursor);
                const acts = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const k = try takeU32(bytes, &cursor);
                const weight_fmt = enumFromInt(WeightFormat, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                break :blk .{ .matmul_q1a8 = .{ .weights = weights, .acts = acts, .dst = dst, .rows = rows, .cols = cols, .k = k, .weight_fmt = weight_fmt } };
            },
            .rmsnorm => blk: {
                const input = try takeRange(bytes, &cursor);
                const weight = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const eps = try takeF32(bytes, &cursor);
                const has_weight_raw = try takeU32(bytes, &cursor);
                if (has_weight_raw > 1) return error.InvalidFlags;
                if (has_weight_raw == 0 and !emptyRange(weight)) return error.InvalidFlags;
                break :blk .{ .rmsnorm = .{
                    .input = input,
                    .weight = weight,
                    .dst = dst,
                    .rows = rows,
                    .cols = cols,
                    .eps = eps,
                    .has_weight = has_weight_raw == 1,
                } };
            },
            .rope => blk: {
                const input = try takeRange(bytes, &cursor);
                const positions = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const head_dim = try takeU32(bytes, &cursor);
                const n_heads = try takeU32(bytes, &cursor);
                const n_tokens = try takeU32(bytes, &cursor);
                const n_dims = try takeU32(bytes, &cursor);
                const mode = enumFromInt(RopeMode, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                const n_ctx_orig = try takeU32(bytes, &cursor);
                const freq_base = try takeF32(bytes, &cursor);
                const freq_scale = try takeF32(bytes, &cursor);
                const ext_factor = try takeF32(bytes, &cursor);
                const attn_factor = try takeF32(bytes, &cursor);
                const beta_fast = try takeF32(bytes, &cursor);
                const beta_slow = try takeF32(bytes, &cursor);
                break :blk .{ .rope = .{
                    .input = input,
                    .positions = positions,
                    .dst = dst,
                    .head_dim = head_dim,
                    .n_heads = n_heads,
                    .n_tokens = n_tokens,
                    .n_dims = n_dims,
                    .mode = mode,
                    .n_ctx_orig = n_ctx_orig,
                    .freq_base = freq_base,
                    .freq_scale = freq_scale,
                    .ext_factor = ext_factor,
                    .attn_factor = attn_factor,
                    .beta_fast = beta_fast,
                    .beta_slow = beta_slow,
                } };
            },
            .softmax => .{ .softmax = .{ .src = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
            .silu => .{ .silu = .{ .src = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
            .swiglu => .{ .swiglu = .{ .lhs = try takeRange(bytes, &cursor), .rhs = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
            .add_f32 => blk: {
                const lhs = try takeRange(bytes, &cursor);
                const rhs = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const mode = enumFromInt(BinaryF32Mode, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .add_f32 = .{ .lhs = lhs, .rhs = rhs, .dst = dst, .rows = rows, .cols = cols, .mode = mode } };
            },
            .mul_f32 => blk: {
                const lhs = try takeRange(bytes, &cursor);
                const rhs = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const mode = enumFromInt(BinaryF32Mode, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .mul_f32 = .{ .lhs = lhs, .rhs = rhs, .dst = dst, .rows = rows, .cols = cols, .mode = mode } };
            },
            .scale_f32 => blk: {
                const src = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const scale = try takeF32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .scale_f32 = .{ .src = src, .dst = dst, .scale = scale } };
            },
            .add_scaled_f32 => blk: {
                const lhs = try takeRange(bytes, &cursor);
                const rhs = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rhs_scale = try takeF32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .add_scaled_f32 = .{ .lhs = lhs, .rhs = rhs, .dst = dst, .rhs_scale = rhs_scale } };
            },
            .set_rows => blk: {
                const src = try takeRange(bytes, &cursor);
                const indices = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const index_type = enumFromInt(IndexType, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                const head_dim = try takeU32(bytes, &cursor);
                const ne01 = try takeU32(bytes, &cursor);
                const ne02 = try takeU32(bytes, &cursor);
                const ne03 = try takeU32(bytes, &cursor);
                const ne11 = try takeU32(bytes, &cursor);
                const ne12 = try takeU32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .set_rows = .{
                    .src = src,
                    .indices = indices,
                    .dst = dst,
                    .index_type = index_type,
                    .head_dim = head_dim,
                    .ne01 = ne01,
                    .ne02 = ne02,
                    .ne03 = ne03,
                    .ne11 = ne11,
                    .ne12 = ne12,
                    .src_nb1 = try takeU64(bytes, &cursor),
                    .src_nb2 = try takeU64(bytes, &cursor),
                    .src_nb3 = try takeU64(bytes, &cursor),
                    .indices_nb1 = try takeU64(bytes, &cursor),
                    .indices_nb2 = try takeU64(bytes, &cursor),
                    .dst_nb1 = try takeU64(bytes, &cursor),
                    .dst_nb2 = try takeU64(bytes, &cursor),
                    .dst_nb3 = try takeU64(bytes, &cursor),
                } };
            },
            .get_rows => blk: {
                const src = try takeRange(bytes, &cursor);
                const indices = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const src_type = enumFromInt(GetRowsSrcType, try takeU32(bytes, &cursor)) orelse return error.InvalidTag;
                const row_width = try takeU32(bytes, &cursor);
                const src_rows = try takeU32(bytes, &cursor);
                const ne10 = try takeU32(bytes, &cursor);
                const ne11 = try takeU32(bytes, &cursor);
                const ne12 = try takeU32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .get_rows = .{
                    .src = src,
                    .indices = indices,
                    .dst = dst,
                    .src_type = src_type,
                    .row_width = row_width,
                    .src_rows = src_rows,
                    .ne10 = ne10,
                    .ne11 = ne11,
                    .ne12 = ne12,
                    .src_nb1 = try takeU64(bytes, &cursor),
                    .src_nb2 = try takeU64(bytes, &cursor),
                    .src_nb3 = try takeU64(bytes, &cursor),
                    .indices_nb1 = try takeU64(bytes, &cursor),
                    .indices_nb2 = try takeU64(bytes, &cursor),
                    .dst_nb1 = try takeU64(bytes, &cursor),
                    .dst_nb2 = try takeU64(bytes, &cursor),
                    .dst_nb3 = try takeU64(bytes, &cursor),
                } };
            },
            .flash_attn_f32 => blk: {
                const q = try takeRange(bytes, &cursor);
                const k = try takeRange(bytes, &cursor);
                const v = try takeRange(bytes, &cursor);
                const mask = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const has_mask_raw = try takeU32(bytes, &cursor);
                if (has_mask_raw > 1) return error.InvalidTag;
                const head_dim_q = try takeU32(bytes, &cursor);
                const head_dim_v = try takeU32(bytes, &cursor);
                const n_heads = try takeU32(bytes, &cursor);
                const n_head_kv = try takeU32(bytes, &cursor);
                const n_kv = try takeU32(bytes, &cursor);
                const n_tokens = try takeU32(bytes, &cursor);
                const scale = try takeF32(bytes, &cursor);
                break :blk .{ .flash_attn_f32 = .{
                    .q = q,
                    .k = k,
                    .v = v,
                    .mask = mask,
                    .dst = dst,
                    .has_mask = has_mask_raw == 1,
                    .head_dim_q = head_dim_q,
                    .head_dim_v = head_dim_v,
                    .n_heads = n_heads,
                    .n_head_kv = n_head_kv,
                    .n_kv = n_kv,
                    .n_tokens = n_tokens,
                    .scale = scale,
                    .q_nb1 = try takeU64(bytes, &cursor),
                    .q_nb2 = try takeU64(bytes, &cursor),
                    .k_nb1 = try takeU64(bytes, &cursor),
                    .k_nb2 = try takeU64(bytes, &cursor),
                    .v_nb1 = try takeU64(bytes, &cursor),
                    .v_nb2 = try takeU64(bytes, &cursor),
                    .mask_nb1 = try takeU64(bytes, &cursor),
                    .dst_nb1 = try takeU64(bytes, &cursor),
                    .dst_nb2 = try takeU64(bytes, &cursor),
                } };
            },
            .argmax => blk: {
                const src = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                break :blk .{ .argmax = .{ .src = src, .dst = dst, .rows = rows, .cols = cols } };
            },
            .pad => .{ .pad = .{ .src = try takeRange(bytes, &cursor), .dst = try takeRange(bytes, &cursor) } },
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

fn emptyRange(range: TensorRange) bool {
    return range.handle == 0 and range.offset == 0 and range.nbytes == 0;
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

fn putF32(out: []u8, cursor: *usize, value: f32) void {
    putU32(out, cursor, @bitCast(value));
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

fn takeF32(bytes: []const u8, cursor: *usize) DecodeError!f32 {
    return @bitCast(try takeU32(bytes, cursor));
}

fn takeU64(bytes: []const u8, cursor: *usize) DecodeError!u64 {
    if (bytes.len - cursor.* < 8) return error.Truncated;
    const value = std.mem.readInt(u64, bytes[cursor.*..][0..8], .little);
    cursor.* += 8;
    return value;
}

test "run_graph profile tier roundtrip and rejection" {
    var meta: [32]u8 = undefined;

    // Legacy 16-byte header (no flag word) decodes with tier = off.
    const legacy_len = try encodeHeader(&meta, .run_graph, 1);
    try std.testing.expectEqual(ProfileTier.off, (try decodeRequest(meta[0..legacy_len], "")).run_graph.tier);

    // Each tier round-trips through its flag bits.
    inline for (.{ ProfileTier.off, ProfileTier.aggregate }) |tier| {
        const len = try encodeRunGraph(&meta, 2, tier);
        try std.testing.expectEqual(tier, (try decodeRequest(meta[0..len], "")).run_graph.tier);
    }

    const len = try encodeRunGraph(&meta, 2, .aggregate);

    // Unknown flag bits (e.g. 0b10, the old spans bit) are rejected, not coerced.
    var bad_flags = meta;
    std.mem.writeInt(u32, bad_flags[16..20], 0b10, .little);
    try std.testing.expectError(error.InvalidFlags, decodeRequest(bad_flags[0..len], ""));
    std.mem.writeInt(u32, bad_flags[16..20], 0b100, .little);
    try std.testing.expectError(error.InvalidFlags, decodeRequest(bad_flags[0..len], ""));

    // Reserved word must be zero.
    var bad_reserved = meta;
    std.mem.writeInt(u32, bad_reserved[20..24], 1, .little);
    try std.testing.expectError(error.InvalidFlags, decodeRequest(bad_reserved[0..len], ""));
}

test "run_graph preload payload roundtrips and bounds-checks" {
    var meta: [32]u8 = undefined;
    const meta_len = try encodeRunGraph(&meta, 7, .aggregate);

    const r0: TensorRange = .{ .handle = 3, .offset = 16, .nbytes = 4 };
    const r1: TensorRange = .{ .handle = 5, .offset = 0, .nbytes = 2 };
    var preload: [128]u8 = undefined;
    var preload_len: usize = 0;
    preload_len += try encodePreloadEntry(preload[preload_len..], r0, &[_]u8{ 1, 2, 3, 4 });
    preload_len += try encodePreloadEntry(preload[preload_len..], r1, &[_]u8{ 9, 9 });

    const commands = [_]u8{ 0, 0, 0, 0 };
    var payload: [256]u8 = undefined;
    const payload_len = try encodeRunGraphPayload(&payload, preload[0..preload_len], &commands);

    const req = (try decodeRequest(meta[0..meta_len], payload[0..payload_len])).run_graph;
    try std.testing.expectEqual(@as(u64, 7), req.request_id);
    try std.testing.expectEqual(ProfileTier.aggregate, req.tier);
    try std.testing.expectEqualSlices(u8, &commands, req.command_bytes);

    var it: PreloadIterator = .{ .bytes = req.preload_bytes };
    const e0 = (try it.next()).?;
    try std.testing.expectEqual(r0, e0.range);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4 }, e0.bytes);
    const e1 = (try it.next()).?;
    try std.testing.expectEqual(r1, e1.range);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 9, 9 }, e1.bytes);
    try std.testing.expect((try it.next()) == null);

    var bad = payload;
    std.mem.writeInt(u32, bad[0..4], @intCast(payload_len), .little);
    try std.testing.expectError(error.InvalidLength, decodeRequest(meta[0..meta_len], bad[0..payload_len]));

    try std.testing.expectError(error.InvalidLength, encodePreloadEntry(preload[0..], .{
        .handle = 1,
        .offset = 0,
        .nbytes = 8,
    }, &[_]u8{ 1, 2 }));

    var short: [rangeLen + 2]u8 = undefined;
    var short_cursor: usize = 0;
    putRange(&short, &short_cursor, .{ .handle = 1, .offset = 0, .nbytes = 8 });
    short[short_cursor] = 1;
    short[short_cursor + 1] = 2;
    var short_it: PreloadIterator = .{ .bytes = &short };
    try std.testing.expectError(error.Truncated, short_it.next());
}

test "response meta roundtrips including device service time" {
    var buf: [response_meta_len]u8 = undefined;
    const meta: ResponseMeta = .{
        .request_id = 42,
        .status = .ok,
        .handle = 7,
        .nbytes = 4096,
        .value0 = 3,
        .device_service_ns = 123_456,
        .device_encode_ns = 789,
    };
    const n = try encodeResponseMeta(&buf, meta);
    try std.testing.expectEqual(response_meta_len, n);
    const decoded = try decodeResponseMeta(buf[0..n]);
    try std.testing.expectEqual(meta, decoded);
}

test "device_encode_ns_offset locates the field in an encoded ResponseMeta" {
    // Pins the constant the device patches at (server.zig) to the actual encoder
    // layout: a field added/reordered before device_encode_ns breaks this, not the
    // silent on-wire stamp.
    var buf: [response_meta_len]u8 = undefined;
    const sentinel: u64 = 0xDEAD_BEEF_F00D_1234;
    const meta: ResponseMeta = .{
        .request_id = 1,
        .status = .ok,
        .handle = 2,
        .nbytes = 3,
        .value0 = 4,
        .value1 = 5,
        .device_service_ns = 6,
        .device_encode_ns = sentinel,
    };
    const n = try encodeResponseMeta(&buf, meta);
    try std.testing.expectEqual(response_meta_len, n);
    // The named offset must point exactly at the encoded device_encode_ns bytes.
    try std.testing.expectEqual(sentinel, std.mem.readInt(u64, buf[device_encode_ns_offset..][0..8], .little));

    // The device's in-place post-encode patch must then round-trip through decode.
    const patched: u64 = 0x0102_0304_0506_0708;
    std.mem.writeInt(u64, buf[device_encode_ns_offset..][0..8], patched, .little);
    const decoded = try decodeResponseMeta(buf[0..n]);
    try std.testing.expectEqual(patched, decoded.device_encode_ns);
    // ...and only that field changed.
    try std.testing.expectEqual(@as(u64, 6), decoded.device_service_ns);
}

test "alloc request roundtrip" {
    var meta: [64]u8 = undefined;
    const n = try encodeAlloc(&meta, 7, 4096, 64);
    const req = try decodeRequest(meta[0..n], "");
    try std.testing.expectEqual(@as(u64, 7), req.requestId());
    try std.testing.expectEqual(@as(u64, 4096), req.alloc.nbytes);
    try std.testing.expectEqual(@as(u32, 64), req.alloc.alignment);
}

test "capabilities request roundtrip" {
    var meta: [32]u8 = undefined;
    const n = try encodeCapabilities(&meta, 17);
    const req = try decodeRequest(meta[0..n], "");
    try std.testing.expectEqual(@as(u64, 17), req.requestId());
    try std.testing.expectEqual(@as(u64, 17), req.capabilities);
}

test "fill request roundtrip" {
    var meta: [64]u8 = undefined;
    const range: TensorRange = .{ .handle = 3, .offset = 128, .nbytes = 4096 };
    const n = try encodeFill(&meta, 9, range, 0xab);
    const req = try decodeRequest(meta[0..n], "");
    try std.testing.expectEqual(@as(u64, 9), req.requestId());
    try std.testing.expectEqual(range, req.fill.range);
    try std.testing.expectEqual(@as(u8, 0xab), req.fill.value);
}

test "command buffer roundtrip" {
    const a: TensorRange = .{ .handle = 1, .offset = 2, .nbytes = 16 };
    const b: TensorRange = .{ .handle = 2, .offset = 4, .nbytes = 16 };
    const c: TensorRange = .{ .handle = 3, .offset = 6, .nbytes = 16 };
    const commands = [_]Command{
        .{ .copy = .{
            .src = a,
            .dst = b,
        } },
        .{ .cpy_f32_to_f16 = .{
            .src = a,
            .dst = b,
        } },
        .{ .matmul_q1a8 = .{
            .weights = .{ .handle = 1, .offset = 0, .nbytes = 144 },
            .acts = .{ .handle = 2, .offset = 0, .nbytes = 512 },
            .dst = .{ .handle = 3, .offset = 0, .nbytes = 32 },
            .rows = 8,
            .cols = 1,
            .k = 128,
            .weight_fmt = .w158a8,
        } },
        .{ .rmsnorm = .{ .input = a, .dst = c, .rows = 4, .cols = 2, .eps = 0.00001 } },
        .{ .rope = .{
            .input = a,
            .positions = b,
            .dst = c,
            .head_dim = 8,
            .n_heads = 2,
            .n_tokens = 3,
            .n_dims = 8,
            .mode = .neox,
            .n_ctx_orig = 4096,
            .freq_base = 10000,
            .freq_scale = 1,
            .ext_factor = 0,
            .attn_factor = 1,
            .beta_fast = 32,
            .beta_slow = 1,
        } },
        .{ .softmax = .{ .src = a, .dst = b } },
        .{ .silu = .{ .src = a, .dst = b } },
        .{ .swiglu = .{ .lhs = a, .rhs = b, .dst = c } },
        .{ .add_f32 = .{ .lhs = a, .rhs = b, .dst = c, .rows = 4, .cols = 1, .mode = .same_shape } },
        .{ .mul_f32 = .{ .lhs = a, .rhs = b, .dst = c, .rows = 4, .cols = 2, .mode = .rhs_row_broadcast } },
        .{ .scale_f32 = .{ .src = a, .dst = b, .scale = 0.5 } },
        .{ .add_scaled_f32 = .{ .lhs = a, .rhs = b, .dst = c, .rhs_scale = 0.25 } },
        .{ .set_rows = .{
            .src = a,
            .indices = b,
            .dst = c,
            .index_type = .i32,
            .head_dim = 4,
            .ne01 = 2,
            .ne02 = 3,
            .ne03 = 1,
            .ne11 = 1,
            .ne12 = 1,
            .src_nb1 = 16,
            .src_nb2 = 32,
            .src_nb3 = 96,
            .indices_nb1 = 8,
            .indices_nb2 = 8,
            .dst_nb1 = 8,
            .dst_nb2 = 64,
            .dst_nb3 = 192,
        } },
        .{ .get_rows = .{
            .src = a,
            .indices = b,
            .dst = c,
            .src_type = .f32,
            .row_width = 4,
            .src_rows = 8,
            .ne10 = 2,
            .ne11 = 3,
            .ne12 = 1,
            .src_nb1 = 16,
            .src_nb2 = 128,
            .src_nb3 = 384,
            .indices_nb1 = 8,
            .indices_nb2 = 24,
            .dst_nb1 = 16,
            .dst_nb2 = 32,
            .dst_nb3 = 96,
        } },
        .{ .flash_attn_f32 = .{
            .q = a,
            .k = b,
            .v = c,
            .mask = a,
            .dst = b,
            .has_mask = true,
            .head_dim_q = 4,
            .head_dim_v = 4,
            .n_heads = 8,
            .n_head_kv = 2,
            .n_kv = 16,
            .n_tokens = 3,
            .scale = 0.5,
            .q_nb1 = 16,
            .q_nb2 = 48,
            .k_nb1 = 8,
            .k_nb2 = 128,
            .v_nb1 = 8,
            .v_nb2 = 128,
            .mask_nb1 = 32,
            .dst_nb1 = 16,
            .dst_nb2 = 128,
        } },
        .{ .argmax = .{ .src = a, .dst = b, .rows = 1, .cols = 4 } },
        .{ .pad = .{ .src = a, .dst = c } },
    };
    var buf: [2048]u8 = undefined;
    const n = try encodeCommandBuffer(&commands, &buf);
    const got = try decodeCommandBuffer(std.testing.allocator, buf[0..n]);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(commands.len, got.len);
    try std.testing.expectEqual(commands[0].copy.src.handle, got[0].copy.src.handle);
    try std.testing.expectEqual(commands[1].cpy_f32_to_f16.dst.handle, got[1].cpy_f32_to_f16.dst.handle);
    try std.testing.expectEqual(commands[2].matmul_q1a8.k, got[2].matmul_q1a8.k);
    try std.testing.expectEqual(commands[2].matmul_q1a8.weight_fmt, got[2].matmul_q1a8.weight_fmt);
    try std.testing.expectEqual(commands[3].rmsnorm.eps, got[3].rmsnorm.eps);
    try std.testing.expectEqual(commands[4].rope.positions.handle, got[4].rope.positions.handle);
    try std.testing.expectEqual(commands[4].rope.mode, got[4].rope.mode);
    try std.testing.expectEqual(commands[4].rope.freq_base, got[4].rope.freq_base);
    try std.testing.expectEqual(commands[5].softmax.dst.offset, got[5].softmax.dst.offset);
    try std.testing.expectEqual(commands[6].silu.src.handle, got[6].silu.src.handle);
    try std.testing.expectEqual(commands[7].swiglu.rhs.handle, got[7].swiglu.rhs.handle);
    try std.testing.expectEqual(commands[8].add_f32.dst.handle, got[8].add_f32.dst.handle);
    try std.testing.expectEqual(commands[8].add_f32.mode, got[8].add_f32.mode);
    try std.testing.expectEqual(commands[9].mul_f32.lhs.offset, got[9].mul_f32.lhs.offset);
    try std.testing.expectEqual(commands[9].mul_f32.cols, got[9].mul_f32.cols);
    try std.testing.expectEqual(commands[9].mul_f32.mode, got[9].mul_f32.mode);
    try std.testing.expectEqual(commands[10].scale_f32.scale, got[10].scale_f32.scale);
    try std.testing.expectEqual(commands[11].add_scaled_f32.rhs_scale, got[11].add_scaled_f32.rhs_scale);
    try std.testing.expectEqual(commands[12].set_rows.index_type, got[12].set_rows.index_type);
    try std.testing.expectEqual(commands[12].set_rows.dst_nb3, got[12].set_rows.dst_nb3);
    try std.testing.expectEqual(commands[13].get_rows.src_rows, got[13].get_rows.src_rows);
    try std.testing.expectEqual(commands[13].get_rows.dst_nb3, got[13].get_rows.dst_nb3);
    try std.testing.expectEqual(commands[14].flash_attn_f32.has_mask, got[14].flash_attn_f32.has_mask);
    try std.testing.expectEqual(commands[14].flash_attn_f32.n_head_kv, got[14].flash_attn_f32.n_head_kv);
    try std.testing.expectEqual(commands[14].flash_attn_f32.scale, got[14].flash_attn_f32.scale);
    try std.testing.expectEqual(commands[14].flash_attn_f32.dst_nb2, got[14].flash_attn_f32.dst_nb2);
    try std.testing.expectEqual(commands[15].argmax.cols, got[15].argmax.cols);
    try std.testing.expectEqual(commands[15].argmax.rows, got[15].argmax.rows);
    try std.testing.expectEqual(commands[15].argmax.dst.handle, got[15].argmax.dst.handle);
    try std.testing.expectEqual(commands[16].pad.src.handle, got[16].pad.src.handle);
    try std.testing.expectEqual(commands[16].pad.dst.offset, got[16].pad.dst.offset);
}

test "rmsnorm optional weight encoding is exact and rejects invalid flags" {
    const input: TensorRange = .{ .handle = 1, .offset = 16, .nbytes = 96 };
    const weight: TensorRange = .{ .handle = 2, .offset = 32, .nbytes = 12 };
    const dst: TensorRange = .{ .handle = 3, .offset = 48, .nbytes = 96 };
    const commands = [_]Command{
        .{ .rmsnorm = .{ .input = input, .dst = dst, .rows = 3, .cols = 8, .eps = 1e-5 } },
        .{ .rmsnorm = .{
            .input = input,
            .weight = weight,
            .dst = dst,
            .rows = 3,
            .cols = 8,
            .eps = 1e-5,
            .has_weight = true,
        } },
    };
    const one_command_len = 4 + 4 + rangeLen * 3 + 16;
    var buf: [256]u8 = undefined;
    const n = try encodeCommandBuffer(&commands, &buf);
    try std.testing.expectEqual(@as(usize, one_command_len * 2 - 4), n);

    const got = try decodeCommandBuffer(std.testing.allocator, buf[0..n]);
    defer std.testing.allocator.free(got);
    try std.testing.expect(!got[0].rmsnorm.has_weight);
    try std.testing.expect(emptyRange(got[0].rmsnorm.weight));
    try std.testing.expect(got[1].rmsnorm.has_weight);
    try std.testing.expectEqual(weight, got[1].rmsnorm.weight);
    try std.testing.expectEqual(dst, got[1].rmsnorm.dst);

    try std.testing.expectError(error.Truncated, decodeCommandBuffer(std.testing.allocator, buf[0 .. n - 1]));
    buf[n] = 0;
    try std.testing.expectError(error.TrailingBytes, decodeCommandBuffer(std.testing.allocator, buf[0 .. n + 1]));

    var bad_flag = buf;
    const first_flag_offset = one_command_len - @sizeOf(u32);
    std.mem.writeInt(u32, bad_flag[first_flag_offset..][0..4], 2, .little);
    try std.testing.expectError(error.InvalidFlags, decodeCommandBuffer(std.testing.allocator, bad_flag[0..n]));

    var bad_absent_weight = buf;
    const first_weight_offset = 4 + 4 + rangeLen;
    std.mem.writeInt(u64, bad_absent_weight[first_weight_offset..][0..8], 9, .little);
    try std.testing.expectError(error.InvalidFlags, decodeCommandBuffer(std.testing.allocator, bad_absent_weight[0..n]));
}
