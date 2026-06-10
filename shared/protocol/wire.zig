const std = @import("std");

pub const version: u16 = 5;
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

pub const EncodeError = error{ OutputTooSmall, TooManyCommands };

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
};

pub const BinaryF32Mode = enum(u32) {
    same_shape = 1,
    rhs_row_broadcast = 2,
};

pub const RopeMode = enum(u32) {
    normal = 1,
    neox = 2,
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

/// What a run_graph request asks the device to collect.
/// `aggregate` accrues per-op totals (cheap, fixed size); `trace` additionally
/// records per-command spans (O(commands), opt-in). Wire flags: bit0 = aggregate,
/// bit1 = spans. Spans imply aggregate, so 0b10 alone is invalid.
pub const ProfileTier = enum(u8) {
    off,
    aggregate,
    trace,

    fn flags(self: ProfileTier) u32 {
        return switch (self) {
            .off => 0,
            .aggregate => 0b01,
            .trace => 0b11,
        };
    }

    fn fromFlags(value: u32) ?ProfileTier {
        return switch (value) {
            0 => .off,
            0b01 => .aggregate,
            0b11 => .trace,
            else => null,
        };
    }
};

pub const RunGraphRequest = struct {
    request_id: u64,
    command_bytes: []const u8,
    tier: ProfileTier = .off,
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
    dst: TensorRange,
    rows: u32,
    cols: u32,
    eps: f32,
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
    /// Used as byte strides for .f32 sources. For .q1_0, src is resident
    /// q1a8-packed data and the device derives row addresses from q1a8 layout.
    src_nb1: u64,
    src_nb2: u64,
    src_nb3: u64,
    indices_nb1: u64,
    indices_nb2: u64,
    dst_nb1: u64,
    dst_nb2: u64,
    dst_nb3: u64,
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
            var tier: ProfileTier = .off;
            if (cursor != metadata.len) {
                const flags = try takeU32(metadata, &cursor);
                const reserved = try takeU32(metadata, &cursor);
                if (cursor != metadata.len) return error.InvalidLength;
                if (reserved != 0) return error.InvalidFlags;
                tier = ProfileTier.fromFlags(flags) orelse return error.InvalidFlags;
            }
            break :blk .{ .run_graph = .{ .request_id = request_id, .command_bytes = payload, .tier = tier } };
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
            .cpy_f32_to_f16 => 4 + rangeLen * 2,
            .matmul_q1a8 => 4 + rangeLen * 3 + 16,
            .rmsnorm => 4 + rangeLen * 2 + 16,
            .rope => 4 + rangeLen * 3 + 48,
            .softmax, .silu => 4 + rangeLen * 2,
            .swiglu => 4 + rangeLen * 3,
            .add_f32, .mul_f32 => 4 + rangeLen * 3 + 16,
            .scale_f32 => 4 + rangeLen * 2 + 8,
            .add_scaled_f32 => 4 + rangeLen * 3 + 8,
            .set_rows => 4 + rangeLen * 3 + 32 + 64,
            .get_rows => 4 + rangeLen * 3 + 32 + 64,
            .flash_attn_f32 => 4 + rangeLen * 5 + 32 + 72,
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
            putU32(out, &cursor, 0);
        },
        .rmsnorm => |rmsnorm| {
            putU16(out, &cursor, @intFromEnum(OpTag.rmsnorm));
            putU16(out, &cursor, 0);
            putRange(out, &cursor, rmsnorm.input);
            putRange(out, &cursor, rmsnorm.dst);
            putU32(out, &cursor, rmsnorm.rows);
            putU32(out, &cursor, rmsnorm.cols);
            putF32(out, &cursor, rmsnorm.eps);
            putU32(out, &cursor, 0);
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
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .matmul_q1a8 = .{ .weights = weights, .acts = acts, .dst = dst, .rows = rows, .cols = cols, .k = k } };
            },
            .rmsnorm => blk: {
                const input = try takeRange(bytes, &cursor);
                const dst = try takeRange(bytes, &cursor);
                const rows = try takeU32(bytes, &cursor);
                const cols = try takeU32(bytes, &cursor);
                const eps = try takeF32(bytes, &cursor);
                _ = try takeU32(bytes, &cursor);
                break :blk .{ .rmsnorm = .{ .input = input, .dst = dst, .rows = rows, .cols = cols, .eps = eps } };
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
    inline for (.{ ProfileTier.off, ProfileTier.aggregate, ProfileTier.trace }) |tier| {
        const len = try encodeRunGraph(&meta, 2, tier);
        try std.testing.expectEqual(tier, (try decodeRequest(meta[0..len], "")).run_graph.tier);
    }

    const len = try encodeRunGraph(&meta, 2, .trace);

    // Spans-without-aggregate (0b10) and unknown bits are rejected, not coerced.
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

test "alloc request roundtrip" {
    var meta: [64]u8 = undefined;
    const n = try encodeAlloc(&meta, 7, 4096, 64);
    const req = try decodeRequest(meta[0..n], "");
    try std.testing.expectEqual(@as(u64, 7), req.requestId());
    try std.testing.expectEqual(@as(u64, 4096), req.alloc.nbytes);
    try std.testing.expectEqual(@as(u32, 64), req.alloc.alignment);
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
    };
    var buf: [2048]u8 = undefined;
    const n = try encodeCommandBuffer(&commands, &buf);
    const got = try decodeCommandBuffer(std.testing.allocator, buf[0..n]);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqual(commands.len, got.len);
    try std.testing.expectEqual(commands[0].copy.src.handle, got[0].copy.src.handle);
    try std.testing.expectEqual(commands[1].cpy_f32_to_f16.dst.handle, got[1].cpy_f32_to_f16.dst.handle);
    try std.testing.expectEqual(commands[2].matmul_q1a8.k, got[2].matmul_q1a8.k);
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
}
