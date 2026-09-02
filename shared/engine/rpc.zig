//! Direct host/daemon control records for the resident inference engine.

const std = @import("std");
const wire = @import("../protocol/wire.zig");
const command = @import("command.zig");
const metrics = @import("metrics.zig");
const spec = @import("model_spec.zig");

pub const Action = wire.InferenceAction;

pub const Error = command.CommandError || spec.SpecError || error{
    InvalidLength,
    InvalidMetricsSnapshot,
    InvalidReserved,
    InvalidSpec,
    InvalidWeightFormat,
    InvalidRange,
    OutputTooSmall,
};

pub const ModelInstall = struct {
    image: wire.TensorRange,
    model_hash: u64,
    model_slot: u16,
    spec_id: spec.SpecId,
    weight_format: spec.WeightFormat,
};

pub const OpenSession = struct {
    model_slot: u16,
    session_slot: u16,
    capacity_tokens: u32,
    epoch: u32,
};

pub const ResetSession = struct {
    session_slot: u16,
    next_epoch: u32,
};

pub const Slot = struct {
    slot: u16,
};

pub const Payload = union(Action) {
    install_model: ModelInstall,
    uninstall_model: Slot,
    open_session: OpenSession,
    close_session: Slot,
    reset_session: ResetSession,
    execute: command.ExecuteTile,
};

pub const install_len: usize = 48;
pub const open_len: usize = 16;
pub const slot_len: usize = 8;
pub const reset_len: usize = 8;

pub fn encodedLen(payload: Payload) usize {
    return switch (payload) {
        .install_model => install_len,
        .open_session => open_len,
        .uninstall_model, .close_session => slot_len,
        .reset_session => reset_len,
        .execute => command.encoded_len,
    };
}

pub fn encode(payload: Payload, out: []u8) Error![]u8 {
    const len = encodedLen(payload);
    if (out.len < len) return error.OutputTooSmall;
    const bytes = out[0..len];
    @memset(bytes, 0);
    switch (payload) {
        .install_model => |request| {
            if (request.image.handle == 0 or request.image.nbytes == 0)
                return error.InvalidRange;
            const selected = spec.get(request.spec_id);
            const plan = try spec.planModelImage(
                request.spec_id,
                request.weight_format,
                0,
            );
            if (request.image.nbytes != plan.image_bytes)
                return error.InvalidRange;
            _ = selected;
            putRange(bytes, 0, request.image);
            std.mem.writeInt(u64, bytes[24..32], request.model_hash, .little);
            std.mem.writeInt(u16, bytes[32..34], request.model_slot, .little);
            std.mem.writeInt(u16, bytes[34..36], @intFromEnum(request.spec_id), .little);
            bytes[36] = @intFromEnum(request.weight_format);
        },
        .open_session => |request| {
            std.mem.writeInt(u16, bytes[0..2], request.model_slot, .little);
            std.mem.writeInt(u16, bytes[2..4], request.session_slot, .little);
            std.mem.writeInt(u32, bytes[4..8], request.capacity_tokens, .little);
            std.mem.writeInt(u32, bytes[8..12], request.epoch, .little);
        },
        .uninstall_model, .close_session => |request| {
            std.mem.writeInt(u16, bytes[0..2], request.slot, .little);
        },
        .reset_session => |request| {
            std.mem.writeInt(u16, bytes[0..2], request.session_slot, .little);
            std.mem.writeInt(u32, bytes[4..8], request.next_epoch, .little);
        },
        .execute => |request| return command.encode(request, out),
    }
    return bytes;
}

pub fn decode(action: Action, bytes: []const u8) Error!Payload {
    return switch (action) {
        .install_model => blk: {
            if (bytes.len != install_len) return error.InvalidLength;
            try reservedZero(bytes[37..]);
            const spec_id = std.enums.fromInt(
                spec.SpecId,
                std.mem.readInt(u16, bytes[34..36], .little),
            ) orelse return error.InvalidSpec;
            const weight_format = std.enums.fromInt(
                spec.WeightFormat,
                bytes[36],
            ) orelse return error.InvalidWeightFormat;
            const image = takeRange(bytes, 0);
            const plan = try spec.planModelImage(spec_id, weight_format, 0);
            if (image.handle == 0 or image.nbytes != plan.image_bytes)
                return error.InvalidRange;
            break :blk .{ .install_model = .{
                .image = image,
                .model_hash = std.mem.readInt(u64, bytes[24..32], .little),
                .model_slot = std.mem.readInt(u16, bytes[32..34], .little),
                .spec_id = spec_id,
                .weight_format = weight_format,
            } };
        },
        .open_session => blk: {
            if (bytes.len != open_len) return error.InvalidLength;
            try reservedZero(bytes[12..]);
            break :blk .{ .open_session = .{
                .model_slot = std.mem.readInt(u16, bytes[0..2], .little),
                .session_slot = std.mem.readInt(u16, bytes[2..4], .little),
                .capacity_tokens = std.mem.readInt(u32, bytes[4..8], .little),
                .epoch = std.mem.readInt(u32, bytes[8..12], .little),
            } };
        },
        .uninstall_model, .close_session => blk: {
            if (bytes.len != slot_len) return error.InvalidLength;
            try reservedZero(bytes[2..]);
            const value = Slot{ .slot = std.mem.readInt(u16, bytes[0..2], .little) };
            break :blk if (action == .uninstall_model)
                .{ .uninstall_model = value }
            else
                .{ .close_session = value };
        },
        .reset_session => blk: {
            if (bytes.len != reset_len) return error.InvalidLength;
            try reservedZero(bytes[2..4]);
            break :blk .{ .reset_session = .{
                .session_slot = std.mem.readInt(u16, bytes[0..2], .little),
                .next_epoch = std.mem.readInt(u32, bytes[4..8], .little),
            } };
        },
        .execute => .{ .execute = try command.decode(bytes) },
    };
}

pub const ResultFlags = struct {
    pub const has_token: u16 = 1 << 0;
    pub const defined_mask: u16 = has_token;
};

pub const ExecuteResult = struct {
    session_slot: u16,
    flags: u16,
    epoch: u32,
    committed_tokens: u32,
    token_id: u32,
    logit: f32,
    status: u32,
    cycles: u64,
    metrics_level: metrics.Level,
    metrics_snapshot: ?MetricsSnapshot = null,

    pub fn token(self: ExecuteResult) ?u32 {
        return if (self.flags & ResultFlags.has_token != 0) self.token_id else null;
    }
};

pub const MetricsSnapshot = struct {
    schema: u32,
    capabilities: u32,
    status: u32,
    tag: u32,
    overflow: [4]u32,
    values: [metrics.metric_count]u64,
};

pub const MetricsStatus = struct {
    pub const valid: u32 = 1 << 0;
    pub const recording: u32 = 1 << 1;
    pub const sealing: u32 = 1 << 2;
    pub const enabled: u32 = 1 << 3;
    pub const any_overflow: u32 = 1 << 4;
    pub const outcome_shift: u5 = 5;
    pub const outcome_mask: u32 = 0x3 << outcome_shift;
    pub const defined_mask: u32 = valid | recording | sealing | enabled |
        any_overflow | outcome_mask;
};

pub fn validMetricsCapabilities(value: u32) bool {
    return value == metrics.compiled_hardware_capabilities or
        value == (metrics.compiled_hardware_capabilities |
            metrics.HardwareCapability.full_bank);
}

pub fn validateMetricsEnvelope(
    schema: u32,
    capabilities: u32,
    status: u32,
    tag: u32,
    expected_outcome: ?metrics.Outcome,
) error{InvalidMetricsSnapshot}!void {
    if (schema != metrics.recorder_schema or
        !validMetricsCapabilities(capabilities) or
        tag == 0 or
        status & ~MetricsStatus.defined_mask != 0 or
        status & MetricsStatus.valid == 0 or
        status & MetricsStatus.enabled == 0 or
        status & (MetricsStatus.recording | MetricsStatus.sealing) != 0)
        return error.InvalidMetricsSnapshot;

    const outcome: metrics.Outcome = @enumFromInt(@as(u2, @truncate(
        (status & MetricsStatus.outcome_mask) >> MetricsStatus.outcome_shift,
    )));
    if (outcome == .none or
        (expected_outcome != null and outcome != expected_outcome.?))
        return error.InvalidMetricsSnapshot;
}

pub fn validateMetricsSnapshot(
    snapshot: MetricsSnapshot,
    expected_cycles: u64,
) error{InvalidMetricsSnapshot}!void {
    try validateMetricsEnvelope(
        snapshot.schema,
        snapshot.capabilities,
        snapshot.status,
        snapshot.tag,
        .commit,
    );
    if (snapshot.overflow[1] & ~@as(u32, 0xff) != 0 or
        snapshot.overflow[2] != 0 or snapshot.overflow[3] != 0)
        return error.InvalidMetricsSnapshot;

    const overflow_present = snapshot.overflow[0] != 0 or
        snapshot.overflow[1] != 0;
    if ((snapshot.status & MetricsStatus.any_overflow != 0) != overflow_present)
        return error.InvalidMetricsSnapshot;

    inline for (@typeInfo(metrics.MetricId).@"enum".fields) |field| {
        const id: metrics.MetricId = @enumFromInt(field.value);
        const max_value: u64 = switch (metrics.metricStorageBits(id)) {
            3 => 0x7,
            8 => 0xff,
            32 => std.math.maxInt(u32),
            64 => std.math.maxInt(u64),
            else => unreachable,
        };
        if (snapshot.values[field.value] > max_value)
            return error.InvalidMetricsSnapshot;
    }
    if (snapshot.values[@intFromEnum(metrics.MetricId.total_cycles)] !=
        expected_cycles)
        return error.InvalidMetricsSnapshot;
}

pub const result_base_len: usize = 40;
pub const metrics_snapshot_len: usize = 32 + @as(usize, metrics.metric_count) * 8;

pub fn resultLen(level: metrics.Level) usize {
    return result_base_len + if (level == .none) 0 else metrics_snapshot_len;
}

pub fn encodeResult(result: ExecuteResult, out: []u8) Error![]u8 {
    if (result.metrics_level == .full) return error.InvalidMetricsSnapshot;
    const len = resultLen(result.metrics_level);
    if (out.len < len) return error.OutputTooSmall;
    if (result.flags & ~ResultFlags.defined_mask != 0)
        return error.InvalidReserved;
    if (result.status != 0) return error.InvalidReserved;
    if (result.token() == null and (result.token_id != 0 or result.logit != 0))
        return error.InvalidReserved;
    if ((result.metrics_level == .none) != (result.metrics_snapshot == null))
        return error.InvalidReserved;
    const bytes = out[0..len];
    @memset(bytes, 0);
    std.mem.writeInt(u16, bytes[0..2], result.session_slot, .little);
    std.mem.writeInt(u16, bytes[2..4], result.flags, .little);
    std.mem.writeInt(u32, bytes[4..8], result.epoch, .little);
    std.mem.writeInt(u32, bytes[8..12], result.committed_tokens, .little);
    std.mem.writeInt(u32, bytes[12..16], result.token_id, .little);
    std.mem.writeInt(u32, bytes[16..20], @bitCast(result.logit), .little);
    std.mem.writeInt(u32, bytes[20..24], result.status, .little);
    std.mem.writeInt(u64, bytes[24..32], result.cycles, .little);
    bytes[32] = @intFromEnum(result.metrics_level);
    if (result.metrics_snapshot) |snapshot| {
        try validateMetricsSnapshot(snapshot, result.cycles);
        var cursor: usize = result_base_len;
        putU32(bytes, &cursor, snapshot.schema);
        putU32(bytes, &cursor, snapshot.capabilities);
        putU32(bytes, &cursor, snapshot.status);
        putU32(bytes, &cursor, snapshot.tag);
        for (snapshot.overflow) |value| putU32(bytes, &cursor, value);
        for (snapshot.values) |value| putU64(bytes, &cursor, value);
        std.debug.assert(cursor == len);
    }
    return bytes;
}

pub fn decodeResult(bytes: []const u8) Error!ExecuteResult {
    if (bytes.len < result_base_len) return error.InvalidLength;
    const metrics_level = std.enums.fromInt(metrics.Level, bytes[32]) orelse
        return error.InvalidReserved;
    if (metrics_level == .full) return error.InvalidMetricsSnapshot;
    if (bytes.len != resultLen(metrics_level)) return error.InvalidLength;
    try reservedZero(bytes[33..result_base_len]);
    var result = ExecuteResult{
        .session_slot = std.mem.readInt(u16, bytes[0..2], .little),
        .flags = std.mem.readInt(u16, bytes[2..4], .little),
        .epoch = std.mem.readInt(u32, bytes[4..8], .little),
        .committed_tokens = std.mem.readInt(u32, bytes[8..12], .little),
        .token_id = std.mem.readInt(u32, bytes[12..16], .little),
        .logit = @bitCast(std.mem.readInt(u32, bytes[16..20], .little)),
        .status = std.mem.readInt(u32, bytes[20..24], .little),
        .cycles = std.mem.readInt(u64, bytes[24..32], .little),
        .metrics_level = metrics_level,
    };
    if (metrics_level != .none) {
        var cursor: usize = result_base_len;
        var snapshot: MetricsSnapshot = .{
            .schema = takeU32(bytes, &cursor),
            .capabilities = takeU32(bytes, &cursor),
            .status = takeU32(bytes, &cursor),
            .tag = takeU32(bytes, &cursor),
            .overflow = undefined,
            .values = undefined,
        };
        for (&snapshot.overflow) |*value| value.* = takeU32(bytes, &cursor);
        for (&snapshot.values) |*value| value.* = takeU64(bytes, &cursor);
        if (cursor != bytes.len)
            return error.InvalidReserved;
        try validateMetricsSnapshot(snapshot, result.cycles);
        result.metrics_snapshot = snapshot;
    }
    if (result.flags & ~ResultFlags.defined_mask != 0)
        return error.InvalidReserved;
    if (result.status != 0) return error.InvalidReserved;
    if (result.token() == null and (result.token_id != 0 or result.logit != 0))
        return error.InvalidReserved;
    return result;
}

fn putU32(out: []u8, cursor: *usize, value: u32) void {
    std.mem.writeInt(u32, out[cursor.*..][0..4], value, .little);
    cursor.* += 4;
}

fn putU64(out: []u8, cursor: *usize, value: u64) void {
    std.mem.writeInt(u64, out[cursor.*..][0..8], value, .little);
    cursor.* += 8;
}

fn takeU32(bytes: []const u8, cursor: *usize) u32 {
    defer cursor.* += 4;
    return std.mem.readInt(u32, bytes[cursor.*..][0..4], .little);
}

fn takeU64(bytes: []const u8, cursor: *usize) u64 {
    defer cursor.* += 8;
    return std.mem.readInt(u64, bytes[cursor.*..][0..8], .little);
}

fn putRange(out: []u8, offset: usize, range: wire.TensorRange) void {
    std.mem.writeInt(u64, out[offset..][0..8], range.handle, .little);
    std.mem.writeInt(u64, out[offset + 8 ..][0..8], range.offset, .little);
    std.mem.writeInt(u64, out[offset + 16 ..][0..8], range.nbytes, .little);
}

fn takeRange(bytes: []const u8, offset: usize) wire.TensorRange {
    return .{
        .handle = std.mem.readInt(u64, bytes[offset..][0..8], .little),
        .offset = std.mem.readInt(u64, bytes[offset + 8 ..][0..8], .little),
        .nbytes = std.mem.readInt(u64, bytes[offset + 16 ..][0..8], .little),
    };
}

fn reservedZero(bytes: []const u8) Error!void {
    for (bytes) |byte| if (byte != 0) return error.InvalidReserved;
}

test "direct model session and tile records are canonical" {
    const plan = try spec.planModelImage(.bonsai_4b, .q1_0, 0);
    const install = Payload{ .install_model = .{
        .image = .{ .handle = 9, .offset = 64, .nbytes = plan.image_bytes },
        .model_hash = 0x1122_3344_5566_7788,
        .model_slot = 2,
        .spec_id = .bonsai_4b,
        .weight_format = .q1_0,
    } };
    var buffer: [64]u8 = undefined;
    const install_bytes = try encode(install, &buffer);
    try std.testing.expectEqual(@as(usize, install_len), install_bytes.len);
    try std.testing.expectEqualDeep(install, try decode(.install_model, install_bytes));
    buffer[47] = 1;
    try std.testing.expectError(
        error.InvalidReserved,
        decode(.install_model, buffer[0..install_len]),
    );

    const open = Payload{ .open_session = .{
        .model_slot = 2,
        .session_slot = 7,
        .capacity_tokens = 4096,
        .epoch = 3,
    } };
    try std.testing.expectEqualDeep(
        open,
        try decode(.open_session, try encode(open, &buffer)),
    );

    const exec = command.ExecuteTile{
        .request_id = 99,
        .model_slot = 2,
        .session_slot = 7,
        .session_epoch = 3,
        .first_position = 0,
        .valid_tokens = 2,
        .flags = command.Flags.emit_logits,
        .token_ids = .{ 11, 12, 0, 0, 0, 0, 0, 0 },
    };
    const exec_payload = Payload{ .execute = exec };
    try std.testing.expectEqualDeep(
        exec_payload,
        try decode(.execute, try encode(exec_payload, &buffer)),
    );
}

test "tile result distinguishes no-logit tiles from token publication" {
    var bytes: [result_base_len + metrics_snapshot_len]u8 = undefined;
    const with_token = ExecuteResult{
        .session_slot = 4,
        .flags = ResultFlags.has_token,
        .epoch = 8,
        .committed_tokens = 9,
        .token_id = 123,
        .logit = 1.5,
        .status = 0,
        .cycles = 456,
        .metrics_level = .none,
    };
    try std.testing.expectEqualDeep(
        with_token,
        try decodeResult(try encodeResult(with_token, &bytes)),
    );
    var bad_status = with_token;
    bad_status.status = 1;
    try std.testing.expectError(
        error.InvalidReserved,
        encodeResult(bad_status, &bytes),
    );
    const canonical = try encodeResult(with_token, &bytes);
    canonical[20] = 1;
    try std.testing.expectError(error.InvalidReserved, decodeResult(canonical));
    const no_token = ExecuteResult{
        .session_slot = 4,
        .flags = 0,
        .epoch = 8,
        .committed_tokens = 8,
        .token_id = 0,
        .logit = 0,
        .status = 0,
        .cycles = 400,
        .metrics_level = .none,
    };
    try std.testing.expect((try decodeResult(try encodeResult(no_token, &bytes))).token() == null);
}

test "metrics snapshot is present only for instrumented results" {
    var values: [metrics.metric_count]u64 = @splat(0);
    values[@intFromEnum(metrics.MetricId.total_cycles)] = 1234;
    values[@intFromEnum(metrics.MetricId.projection_drain_cycles)] = 17;
    const result: ExecuteResult = .{
        .session_slot = 1,
        .flags = 0,
        .epoch = 2,
        .committed_tokens = 8,
        .token_id = 0,
        .logit = 0,
        .status = 0,
        .cycles = 1234,
        .metrics_level = .summary,
        .metrics_snapshot = .{
            .schema = metrics.recorder_schema,
            .capabilities = metrics.compiled_hardware_capabilities,
            .status = 0x39,
            .tag = 7,
            .overflow = .{ 0, 1, 0, 0 },
            .values = values,
        },
    };
    var bytes: [result_base_len + metrics_snapshot_len]u8 = undefined;
    const encoded = try encodeResult(result, &bytes);
    try std.testing.expectEqual(resultLen(.summary), encoded.len);
    try std.testing.expectEqualDeep(result, try decodeResult(encoded));
    try std.testing.expectError(error.InvalidLength, decodeResult(encoded[0 .. encoded.len - 1]));
}

test "metrics snapshots enforce the canonical lean-bank contract" {
    var values: [metrics.metric_count]u64 = @splat(0);
    values[@intFromEnum(metrics.MetricId.total_cycles)] = 100;
    var snapshot: MetricsSnapshot = .{
        .schema = metrics.recorder_schema,
        .capabilities = metrics.compiled_hardware_capabilities,
        .status = MetricsStatus.valid | MetricsStatus.enabled |
            (@as(u32, @intFromEnum(metrics.Outcome.commit)) << MetricsStatus.outcome_shift),
        .tag = 1,
        .overflow = @splat(0),
        .values = values,
    };
    try validateMetricsSnapshot(snapshot, 100);

    snapshot.values[@intFromEnum(metrics.MetricId.projection_selector_high_water)] = 8;
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.values[@intFromEnum(metrics.MetricId.projection_selector_high_water)] = 0;

    snapshot.values[@intFromEnum(metrics.MetricId.embed_calls)] = 256;
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.values[@intFromEnum(metrics.MetricId.embed_calls)] = 0;

    snapshot.values[@intFromEnum(metrics.MetricId.control_cycles)] =
        @as(u64, std.math.maxInt(u32)) + 1;
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.values[@intFromEnum(metrics.MetricId.control_cycles)] = 0;

    snapshot.overflow[1] = 1 << 8;
    snapshot.status |= MetricsStatus.any_overflow;
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.overflow = @splat(0);
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.status &= ~MetricsStatus.any_overflow;

    snapshot.capabilities |= 1 << 31;
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        validateMetricsSnapshot(snapshot, 100),
    );
    snapshot.capabilities = metrics.compiled_hardware_capabilities |
        metrics.HardwareCapability.full_bank;
    try validateMetricsSnapshot(snapshot, 100);
}

test "full metrics results remain reserved until their payload is versioned" {
    var bytes: [result_base_len + metrics_snapshot_len]u8 = undefined;
    var values: [metrics.metric_count]u64 = @splat(0);
    values[@intFromEnum(metrics.MetricId.total_cycles)] = 1;
    const result: ExecuteResult = .{
        .session_slot = 1,
        .flags = 0,
        .epoch = 1,
        .committed_tokens = 1,
        .token_id = 0,
        .logit = 0,
        .status = 0,
        .cycles = 1,
        .metrics_level = .full,
        .metrics_snapshot = .{
            .schema = metrics.recorder_schema,
            .capabilities = metrics.compiled_hardware_capabilities |
                metrics.HardwareCapability.full_bank,
            .status = MetricsStatus.valid | MetricsStatus.enabled |
                (@as(u32, @intFromEnum(metrics.Outcome.commit)) << MetricsStatus.outcome_shift),
            .tag = 1,
            .overflow = @splat(0),
            .values = values,
        },
    };
    try std.testing.expectError(
        error.InvalidMetricsSnapshot,
        encodeResult(result, &bytes),
    );
}
