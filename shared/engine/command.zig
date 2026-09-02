//! Fixed EXEC_TILE command and transactional session/KV watermark contract.
//!
//! Decode and prefill share this command. `valid_tokens == 1` is decode;
//! `2...8` is a prefill tile. A failed or aborted tile may leave uncommitted DDR
//! bytes behind, but it cannot advance the visible session watermark.

const std = @import("std");
pub const model_spec = @import("model_spec.zig");
const metrics = @import("metrics.zig");

pub const command_magic: u32 = 0x315A_4E50; // "PNZ1" in little-endian bytes.
pub const command_version: u16 = 2;
pub const encoded_len: usize = 64;

pub const Flags = struct {
    pub const emit_logits: u16 = 1 << 0;
    pub const defined_mask: u16 = emit_logits;
};

pub const ExecutionMode = enum {
    decode,
    prefill,
};

pub const CommandError = error{
    OutputTooSmall,
    InvalidLength,
    InvalidMagic,
    UnsupportedVersion,
    InvalidFlags,
    NonCanonicalEncoding,
    InvalidRequestId,
    InvalidTileSize,
    InvalidTokenId,
    InvalidModelSlot,
    InvalidSessionSlot,
    InvalidSessionEpoch,
    InvalidWatermark,
    ContextOverflow,
    SessionBusy,
    SessionIdle,
    RequestMismatch,
    SpecMismatch,
    InvalidKvCapacity,
    InvalidKvAddress,
    IncompleteLayerWalk,
    KvWritesNotFenced,
    Overflow,
};

fn checkedAdd(a: u32, b: u32) CommandError!u32 {
    return std.math.add(u32, a, b) catch error.Overflow;
}

fn checkedAdd64(a: u64, b: u64) CommandError!u64 {
    return std.math.add(u64, a, b) catch error.Overflow;
}

fn checkedMul64(a: u64, b: u64) CommandError!u64 {
    return std.math.mul(u64, a, b) catch error.Overflow;
}

pub const ExecuteTile = struct {
    request_id: u64,
    model_slot: u16,
    session_slot: u16,
    session_epoch: u32,
    first_position: u32,
    valid_tokens: u8,
    flags: u16 = 0,
    metrics_level: metrics.Level = .none,
    token_ids: [model_spec.token_tile_max]u32,

    pub fn mode(self: ExecuteTile) ExecutionMode {
        return if (self.valid_tokens == 1) .decode else .prefill;
    }

    pub fn endPosition(self: ExecuteTile) CommandError!u32 {
        return checkedAdd(self.first_position, self.valid_tokens);
    }

    pub fn validate(
        self: ExecuteTile,
        spec: model_spec.ModelSpec,
        session: *const SessionState,
    ) CommandError!void {
        if (self.request_id == 0) return error.InvalidRequestId;
        if (self.flags & ~Flags.defined_mask != 0) return error.InvalidFlags;
        if (self.valid_tokens == 0 or
            self.valid_tokens > model_spec.token_tile_max)
            return error.InvalidTileSize;
        if (session.in_flight) return error.SessionBusy;
        if (self.model_slot != session.model_slot) return error.InvalidModelSlot;
        if (self.session_slot != session.session_slot)
            return error.InvalidSessionSlot;
        if (self.session_epoch != session.epoch)
            return error.InvalidSessionEpoch;
        if (spec.id != session.spec_id) return error.SpecMismatch;
        if (self.first_position != session.committed_tokens)
            return error.InvalidWatermark;
        const end = try self.endPosition();
        if (end > session.kv_capacity_tokens or end > spec.context_length)
            return error.ContextOverflow;
        for (self.token_ids, 0..) |token_id, index| {
            if (index < self.valid_tokens) {
                if (token_id >= spec.vocab_size) return error.InvalidTokenId;
            } else if (token_id != 0) {
                return error.NonCanonicalEncoding;
            }
        }
    }
};

pub fn encode(command: ExecuteTile, out: []u8) CommandError![]u8 {
    if (out.len < encoded_len) return error.OutputTooSmall;
    if (command.request_id == 0) return error.InvalidRequestId;
    if (command.flags & ~Flags.defined_mask != 0) return error.InvalidFlags;
    if (command.valid_tokens == 0 or
        command.valid_tokens > model_spec.token_tile_max)
        return error.InvalidTileSize;
    for (command.token_ids[command.valid_tokens..]) |token_id|
        if (token_id != 0) return error.NonCanonicalEncoding;

    const bytes = out[0..encoded_len];
    @memset(bytes, 0);
    std.mem.writeInt(u32, bytes[0..4], command_magic, .little);
    std.mem.writeInt(u16, bytes[4..6], command_version, .little);
    std.mem.writeInt(u16, bytes[6..8], command.flags, .little);
    std.mem.writeInt(u64, bytes[8..16], command.request_id, .little);
    std.mem.writeInt(u16, bytes[16..18], command.model_slot, .little);
    std.mem.writeInt(u16, bytes[18..20], command.session_slot, .little);
    std.mem.writeInt(u32, bytes[20..24], command.session_epoch, .little);
    std.mem.writeInt(u32, bytes[24..28], command.first_position, .little);
    bytes[28] = command.valid_tokens;
    bytes[29] = @intFromEnum(command.metrics_level);
    for (command.token_ids, 0..) |token_id, index|
        std.mem.writeInt(u32, bytes[32 + index * 4 ..][0..4], token_id, .little);
    return bytes;
}

pub fn decode(bytes: []const u8) CommandError!ExecuteTile {
    if (bytes.len != encoded_len) return error.InvalidLength;
    if (std.mem.readInt(u32, bytes[0..4], .little) != command_magic)
        return error.InvalidMagic;
    if (std.mem.readInt(u16, bytes[4..6], .little) != command_version)
        return error.UnsupportedVersion;
    const flags = std.mem.readInt(u16, bytes[6..8], .little);
    if (flags & ~Flags.defined_mask != 0) return error.InvalidFlags;
    const metrics_level = std.enums.fromInt(metrics.Level, bytes[29]) orelse
        return error.NonCanonicalEncoding;
    if (bytes[30] != 0 or bytes[31] != 0)
        return error.NonCanonicalEncoding;
    const valid_tokens = bytes[28];
    if (valid_tokens == 0 or valid_tokens > model_spec.token_tile_max)
        return error.InvalidTileSize;

    var token_ids: [model_spec.token_tile_max]u32 = undefined;
    for (&token_ids, 0..) |*token_id, index|
        token_id.* = std.mem.readInt(u32, bytes[32 + index * 4 ..][0..4], .little);
    for (token_ids[valid_tokens..]) |token_id|
        if (token_id != 0) return error.NonCanonicalEncoding;
    const request_id = std.mem.readInt(u64, bytes[8..16], .little);
    if (request_id == 0) return error.InvalidRequestId;
    return .{
        .request_id = request_id,
        .model_slot = std.mem.readInt(u16, bytes[16..18], .little),
        .session_slot = std.mem.readInt(u16, bytes[18..20], .little),
        .session_epoch = std.mem.readInt(u32, bytes[20..24], .little),
        .first_position = std.mem.readInt(u32, bytes[24..28], .little),
        .valid_tokens = valid_tokens,
        .flags = flags,
        .metrics_level = metrics_level,
        .token_ids = token_ids,
    };
}

pub const ExecuteCompletion = struct {
    request_id: u64,
    completed_layers: u16,
    kv_writes_fenced: bool,
};

/// Persistent session metadata. `committed_tokens` is the only visibility
/// watermark. Pending rows are ignored after abort and overwritten on retry.
pub const SessionState = struct {
    model_slot: u16,
    session_slot: u16,
    spec_id: model_spec.SpecId,
    epoch: u32,
    kv_base: u64,
    kv_capacity_tokens: u32,
    committed_tokens: u32 = 0,
    in_flight: bool = false,
    pending_request_id: u64 = 0,
    pending_end: u32 = 0,

    pub fn init(
        model_slot: u16,
        session_slot: u16,
        spec: model_spec.ModelSpec,
        epoch: u32,
        kv_base: u64,
        kv_capacity_tokens: u32,
    ) CommandError!SessionState {
        if (epoch == 0) return error.InvalidSessionEpoch;
        if (kv_capacity_tokens == 0 or
            kv_capacity_tokens > spec.context_length)
            return error.InvalidKvCapacity;
        const bytes = spec.kvCacheBytes(kv_capacity_tokens) catch
            return error.Overflow;
        const record_bytes = spec.kvRecordBytes() catch
            return error.Overflow;
        if (kv_base % record_bytes != 0) return error.InvalidKvAddress;
        const span = model_spec.DdrSpan{
            .address = kv_base,
            .bytes = bytes,
        };
        span.validate() catch return error.InvalidKvAddress;
        return .{
            .model_slot = model_slot,
            .session_slot = session_slot,
            .spec_id = spec.id,
            .epoch = epoch,
            .kv_base = kv_base,
            .kv_capacity_tokens = kv_capacity_tokens,
        };
    }

    pub fn begin(
        self: *SessionState,
        command: ExecuteTile,
        spec: model_spec.ModelSpec,
    ) CommandError!void {
        try command.validate(spec, self);
        self.in_flight = true;
        self.pending_request_id = command.request_id;
        self.pending_end = try command.endPosition();
    }

    /// Publish the tile only after every layer completed and every asynchronous
    /// KV append is globally visible to the next tile.
    pub fn commit(
        self: *SessionState,
        spec: model_spec.ModelSpec,
        completion: ExecuteCompletion,
    ) CommandError!void {
        if (!self.in_flight) return error.SessionIdle;
        if (spec.id != self.spec_id) return error.SpecMismatch;
        if (completion.request_id != self.pending_request_id)
            return error.RequestMismatch;
        if (completion.completed_layers != spec.layers)
            return error.IncompleteLayerWalk;
        if (!completion.kv_writes_fenced) return error.KvWritesNotFenced;
        self.committed_tokens = self.pending_end;
        self.clearPending();
    }

    pub fn abort(self: *SessionState, request_id: u64) CommandError!void {
        if (!self.in_flight) return error.SessionIdle;
        if (request_id != self.pending_request_id) return error.RequestMismatch;
        self.clearPending();
    }

    pub fn reset(self: *SessionState, next_epoch: u32) CommandError!void {
        if (self.in_flight) return error.SessionBusy;
        if (next_epoch == 0 or next_epoch == self.epoch)
            return error.InvalidSessionEpoch;
        self.epoch = next_epoch;
        self.committed_tokens = 0;
        self.clearPending();
    }

    fn clearPending(self: *SessionState) void {
        self.in_flight = false;
        self.pending_request_id = 0;
        self.pending_end = 0;
    }

    pub fn kvAllocation(self: SessionState, spec: model_spec.ModelSpec) CommandError!model_spec.DdrSpan {
        if (spec.id != self.spec_id) return error.SpecMismatch;
        const bytes = spec.kvCacheBytes(self.kv_capacity_tokens) catch
            return error.Overflow;
        return .{ .address = self.kv_base, .bytes = bytes };
    }

    /// Layer-major layout: [layer][capacity position][contiguous K then V].
    pub fn kvRecordAddress(
        self: SessionState,
        spec: model_spec.ModelSpec,
        layer: u16,
        position: u32,
    ) CommandError!u64 {
        if (spec.id != self.spec_id) return error.SpecMismatch;
        if (layer >= spec.layers or position >= self.kv_capacity_tokens)
            return error.InvalidKvAddress;
        const record_bytes = spec.kvRecordBytes() catch return error.Overflow;
        const record_index = try checkedAdd64(
            try checkedMul64(layer, self.kv_capacity_tokens),
            position,
        );
        return checkedAdd64(
            self.kv_base,
            try checkedMul64(record_index, record_bytes),
        );
    }
};

pub fn tileKvBytesPerLayer(
    spec: model_spec.ModelSpec,
    valid_tokens: u8,
) CommandError!u64 {
    if (valid_tokens == 0 or valid_tokens > model_spec.token_tile_max)
        return error.InvalidTileSize;
    return checkedMul64(
        valid_tokens,
        spec.kvRecordBytes() catch return error.Overflow,
    );
}

pub fn tileKvWriteBytes(
    spec: model_spec.ModelSpec,
    valid_tokens: u8,
) CommandError!u64 {
    return checkedMul64(
        spec.layers,
        try tileKvBytesPerLayer(spec, valid_tokens),
    );
}

fn commandFor(
    session: SessionState,
    request_id: u64,
    valid_tokens: u8,
) ExecuteTile {
    var ids: [model_spec.token_tile_max]u32 = @splat(0);
    for (ids[0..valid_tokens], 0..) |*token_id, index| token_id.* = @intCast(index + 10);
    return .{
        .request_id = request_id,
        .model_slot = session.model_slot,
        .session_slot = session.session_slot,
        .session_epoch = session.epoch,
        .first_position = session.committed_tokens,
        .valid_tokens = valid_tokens,
        .flags = Flags.emit_logits,
        .metrics_level = .summary,
        .token_ids = ids,
    };
}

test "EXEC_TILE has one exact 64-byte canonical encoding" {
    const spec = model_spec.bonsai_4b;
    const session = try SessionState.init(3, 9, spec, 7, 0x4000, 1024);
    const command = commandFor(session, 0x1122_3344_5566_7788, 8);
    var storage: [encoded_len]u8 = undefined;
    const bytes = try encode(command, &storage);
    try std.testing.expectEqual(@as(usize, 64), bytes.len);
    try std.testing.expectEqualSlices(u8, "PNZ1", bytes[0..4]);
    const decoded = try decode(bytes);
    try std.testing.expectEqual(command.request_id, decoded.request_id);
    try std.testing.expectEqual(command.token_ids, decoded.token_ids);
    try std.testing.expectEqual(metrics.Level.summary, decoded.metrics_level);
    try std.testing.expectEqual(ExecutionMode.prefill, decoded.mode());
    try decoded.validate(spec, &session);
}

test "decode is the same command with one valid lane" {
    const spec = model_spec.bonsai_1_7b;
    const session = try SessionState.init(1, 2, spec, 1, 0x8000, 32);
    const command = commandFor(session, 1, 1);
    try std.testing.expectEqual(ExecutionMode.decode, command.mode());
    try command.validate(spec, &session);
}

test "watermark commits only after a complete fenced layer walk" {
    const spec = model_spec.bonsai_8b;
    var session = try SessionState.init(1, 4, spec, 12, 0x10_0000, 64);
    const command = commandFor(session, 99, 8);
    try session.begin(command, spec);
    try std.testing.expectEqual(@as(u32, 0), session.committed_tokens);
    try std.testing.expectError(error.IncompleteLayerWalk, session.commit(spec, .{
        .request_id = 99,
        .completed_layers = spec.layers - 1,
        .kv_writes_fenced = true,
    }));
    try std.testing.expectError(error.KvWritesNotFenced, session.commit(spec, .{
        .request_id = 99,
        .completed_layers = spec.layers,
        .kv_writes_fenced = false,
    }));
    try session.commit(spec, .{
        .request_id = 99,
        .completed_layers = spec.layers,
        .kv_writes_fenced = true,
    });
    try std.testing.expectEqual(@as(u32, 8), session.committed_tokens);
    try std.testing.expect(!session.in_flight);
}

test "8B final T8 tile commits the 65536 watermark exactly" {
    const spec = model_spec.bonsai_8b;
    var session = try SessionState.init(
        1,
        4,
        spec,
        12,
        0x10_0000,
        spec.context_length,
    );
    session.committed_tokens = spec.context_length - 8;
    const final = commandFor(session, 100, 8);
    try final.validate(spec, &session);
    try session.begin(final, spec);
    try session.commit(spec, .{
        .request_id = 100,
        .completed_layers = spec.layers,
        .kv_writes_fenced = true,
    });
    try std.testing.expectEqual(@as(u32, 65_536), session.committed_tokens);

    const overflow = commandFor(session, 101, 1);
    try std.testing.expectError(
        error.ContextOverflow,
        overflow.validate(spec, &session),
    );
}

test "abort preserves watermark and retry overwrites pending rows" {
    const spec = model_spec.bonsai_1_7b;
    var session = try SessionState.init(0, 0, spec, 3, 0x20_0000, 16);
    const first = commandFor(session, 41, 8);
    try session.begin(first, spec);
    try session.abort(41);
    try std.testing.expectEqual(@as(u32, 0), session.committed_tokens);
    const retry = commandFor(session, 42, 8);
    try session.begin(retry, spec);
    try std.testing.expectEqual(first.first_position, retry.first_position);
}

test "layer-major KV addresses and tile write accounting are exact" {
    const spec = model_spec.bonsai_4b;
    const session = try SessionState.init(0, 0, spec, 1, 0x40_0000, 1024);
    const record_bytes = try spec.kvRecordBytes();
    try std.testing.expectEqual(@as(u64, 4096), record_bytes);
    try std.testing.expectEqual(
        session.kv_base + (@as(u64, 7) * 1024 + 19) * record_bytes,
        try session.kvRecordAddress(spec, 7, 19),
    );
    try std.testing.expectEqual(@as(u64, 32 * 1024), try tileKvBytesPerLayer(spec, 8));
    try std.testing.expectEqual(@as(u64, 36 * 32 * 1024), try tileKvWriteBytes(spec, 8));
}

test "stale, over-capacity, and noncanonical tiles fail closed" {
    const spec = model_spec.bonsai_1_7b;
    const session = try SessionState.init(2, 5, spec, 8, 0x80_0000, 8);
    var command = commandFor(session, 1, 8);
    command.session_epoch = 7;
    try std.testing.expectError(error.InvalidSessionEpoch, command.validate(spec, &session));
    command.session_epoch = 8;
    command.token_ids[7] = spec.vocab_size;
    try std.testing.expectError(error.InvalidTokenId, command.validate(spec, &session));
    command.token_ids[7] = 0;
    command.valid_tokens = 7;
    command.token_ids[7] = 1;
    try std.testing.expectError(error.NonCanonicalEncoding, command.validate(spec, &session));
    command.token_ids[7] = 0;
    command.first_position = 1;
    try std.testing.expectError(error.InvalidWatermark, command.validate(spec, &session));
}

test "KV arena base is aligned to one physical position record" {
    const spec = model_spec.bonsai_1_7b;
    try std.testing.expectError(
        error.InvalidKvAddress,
        SessionState.init(0, 0, spec, 1, 0x10_0040, 8),
    );
    _ = try SessionState.init(0, 0, spec, 1, 0x10_1000, 8);
}

test "decoder rejects reserved bytes without changing the command length" {
    const spec = model_spec.bonsai_1_7b;
    const session = try SessionState.init(0, 0, spec, 1, 0x1000, 8);
    const command = commandFor(session, 1, 1);
    var storage: [encoded_len]u8 = undefined;
    _ = try encode(command, &storage);
    storage[31] = 1;
    try std.testing.expectError(error.NonCanonicalEncoding, decode(&storage));
    storage[31] = 0;
    storage[29] = 3;
    try std.testing.expectError(error.NonCanonicalEncoding, decode(&storage));
    storage[29] = @intFromEnum(metrics.Level.summary);
    storage[36] = 1;
    try std.testing.expectError(error.NonCanonicalEncoding, decode(&storage));
}
