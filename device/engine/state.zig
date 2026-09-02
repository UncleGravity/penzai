//! Bounded daemon-side ownership and transaction state for the inference engine.
//!
//! This module deliberately has no runtime, transport, allocator, or register
//! dependencies. RPC decoding can populate the fixed-width request records;
//! the eventual PL driver consumes the immutable records returned by lookup.

const std = @import("std");
const shared = @import("shared");

pub const model_spec = shared.engine.model_spec;
pub const command = shared.engine.command;

pub const interface_version: u32 = 0x0001_0007;
pub const image_layout_hash: u64 = 0xc255_c7a5_2fc1_4a79;
pub const default_model_slots: usize = 4;
pub const default_session_slots: usize = 64;

pub const Error = model_spec.SpecError || command.CommandError || error{
    InterfaceMismatch,
    LayoutMismatch,
    InvalidModelHash,
    NonCanonicalRecord,
    InvalidPhysicalAddress,
    PhysicalAddressConflict,
    ModelSlotOccupied,
    ModelSlotEmpty,
    ModelInUse,
    SessionSlotOccupied,
    SessionSlotEmpty,
    EngineBusy,
    NotActiveSession,
};

/// Fixed 64-byte install header. Exactly `layer_count` canonical 64-byte
/// LayerAddresses records follow it in an RPC payload.
pub const ModelInstallHeader = extern struct {
    model_hash: u64,
    image_layout_hash: u64,
    embedding: u64,
    lm_head: u64,
    output_norm: u64,
    rope_table: u64,
    interface_version: u32,
    model_slot: u16,
    spec_id: u16,
    layer_count: u16,
    weight_format: u8,
    reserved: [5]u8 = @splat(0),
};

/// Fixed 32-byte session-open record. A newly opened session always starts
/// with a zero committed watermark.
pub const OpenSessionRequest = extern struct {
    kv_base: u64,
    kv_capacity_tokens: u32,
    epoch: u32,
    model_slot: u16,
    session_slot: u16,
    reserved: [12]u8 = @splat(0),
};

/// Fixed 16-byte form of shared.TileCompletion. Integers replace bools so the
/// record has one canonical wire representation.
pub const CommitRequest = extern struct {
    request_id: u64,
    session_slot: u16,
    completed_layers: u16,
    kv_writes_fenced: u8,
    reserved: [3]u8 = @splat(0),
};

pub const AbortRequest = extern struct {
    request_id: u64,
    session_slot: u16,
    reserved: [6]u8 = @splat(0),
};

pub const ResetSessionRequest = extern struct {
    next_epoch: u32,
    session_slot: u16,
    reserved: [2]u8 = @splat(0),
};

/// Stable session status record for responses and diagnostics.
pub const SessionSnapshot = extern struct {
    kv_base: u64,
    pending_request_id: u64,
    epoch: u32,
    kv_capacity_tokens: u32,
    committed_tokens: u32,
    pending_end: u32,
    model_slot: u16,
    session_slot: u16,
    spec_id: u16,
    present: u8,
    in_flight: u8,
    reserved: [8]u8 = @splat(0),
};

/// Everything needed to bind one accepted execute request to the immutable model
/// and mutable session allocation. The token IDs remain in the canonical
/// shared 64-byte command rather than being copied into this lease.
pub const ExecutionLease = extern struct {
    request_id: u64,
    model_hash: u64,
    kv_base: u64,
    first_position: u32,
    pending_end: u32,
    kv_capacity_tokens: u32,
    session_epoch: u32,
    model_slot: u16,
    session_slot: u16,
    spec_id: u16,
    contract_version: u16,
    weight_format: u8,
    valid_tokens: u8,
    flags: u16,
    committed_tokens: u32,
    reserved: [4]u8 = @splat(0),
};

comptime {
    if (@sizeOf(ModelInstallHeader) != 64)
        @compileError("inference model install header must remain 64 bytes");
    if (@sizeOf(OpenSessionRequest) != 32)
        @compileError("inference session-open record must remain 32 bytes");
    if (@sizeOf(CommitRequest) != 16 or @sizeOf(AbortRequest) != 16)
        @compileError("inference completion and abort records must remain 16 bytes");
    if (@sizeOf(ResetSessionRequest) != 8)
        @compileError("inference session-reset record must remain 8 bytes");
    if (@sizeOf(SessionSnapshot) != 48)
        @compileError("inference session snapshot must remain 48 bytes");
    if (@sizeOf(ExecutionLease) != 64)
        @compileError("inference execution lease must remain 64 bytes");
}

pub const ModelInstall = struct {
    header: ModelInstallHeader,
    layers: []const model_spec.LayerAddresses,
};

/// Immutable after publication. Unused layer records are zeroed so a complete
/// record has deterministic contents when inspected or serialized.
pub const ModelRecord = struct {
    header: ModelInstallHeader,
    layer_storage: [model_spec.max_layers]model_spec.LayerAddresses,

    pub fn spec(self: *const ModelRecord) model_spec.ModelSpec {
        return model_spec.get(@enumFromInt(self.header.spec_id));
    }

    pub fn format(self: *const ModelRecord) model_spec.WeightFormat {
        return @enumFromInt(self.header.weight_format);
    }

    pub fn layers(self: *const ModelRecord) []const model_spec.LayerAddresses {
        return self.layer_storage[0..self.header.layer_count];
    }

    pub fn addressTable(self: *const ModelRecord) model_spec.ModelAddressTable {
        return .{
            .spec_id = @enumFromInt(self.header.spec_id),
            .weight_format = @enumFromInt(self.header.weight_format),
            .embedding = self.header.embedding,
            .lm_head = self.header.lm_head,
            .output_norm = self.header.output_norm,
            .rope_table = self.header.rope_table,
            .layers = self.layers(),
        };
    }
};

const zero_layer = model_spec.LayerAddresses{
    .fused_qkv = 0,
    .attention_output = 0,
    .fused_gate_up = 0,
    .ffn_down = 0,
    .attention_norm = 0,
    .attention_q_norm = 0,
    .attention_k_norm = 0,
    .ffn_norm = 0,
};

const ModelIdentity = struct {
    spec: model_spec.ModelSpec,
    format: model_spec.WeightFormat,
};

pub fn Registry(
    comptime model_slot_count: usize,
    comptime session_slot_count: usize,
) type {
    if (model_slot_count == 0 or model_slot_count > std.math.maxInt(u16))
        @compileError("inference model slot count must fit u16 and be nonzero");
    if (session_slot_count == 0 or
        session_slot_count > std.math.maxInt(u16))
        @compileError("inference session slot count must fit u16 and be nonzero");

    return struct {
        const Self = @This();

        models: [model_slot_count]?ModelRecord,
        sessions: [session_slot_count]?command.SessionState,
        active_session_slot: ?u16,

        pub fn init() Self {
            return .{
                .models = @splat(null),
                .sessions = @splat(null),
                .active_session_slot = null,
            };
        }

        pub fn validateInstall(
            self: *const Self,
            install_request: ModelInstall,
        ) Error!void {
            const model_index = try modelIndex(install_request.header.model_slot);
            if (self.models[model_index] != null) return error.ModelSlotOccupied;
            const identity = try validateModelInstall(install_request);

            var candidate_storage: [model_spec.ModelAddressTable.max_spans]model_spec.DdrSpan = undefined;
            const candidate = try collectModelSpans(
                install_request.header,
                install_request.layers,
                identity,
                &candidate_storage,
            );
            try self.rejectConflicts(candidate, null);
        }

        pub fn install(
            self: *Self,
            install_request: ModelInstall,
        ) Error!*const ModelRecord {
            try self.validateInstall(install_request);
            const index = try modelIndex(install_request.header.model_slot);
            var record = ModelRecord{
                .header = install_request.header,
                .layer_storage = @splat(zero_layer),
            };
            @memcpy(
                record.layer_storage[0..install_request.layers.len],
                install_request.layers,
            );
            self.models[index] = record;
            return &self.models[index].?;
        }

        pub fn model(
            self: *const Self,
            model_slot: u16,
        ) Error!*const ModelRecord {
            const index = try modelIndex(model_slot);
            return if (self.models[index]) |*record|
                record
            else
                error.ModelSlotEmpty;
        }

        pub fn uninstall(self: *Self, model_slot: u16) Error!void {
            const index = try modelIndex(model_slot);
            if (self.models[index] == null) return error.ModelSlotEmpty;
            for (self.sessions) |maybe_session| {
                if (maybe_session) |session_state|
                    if (session_state.model_slot == model_slot)
                        return error.ModelInUse;
            }
            self.models[index] = null;
        }

        pub fn open(
            self: *Self,
            request: OpenSessionRequest,
        ) Error!SessionSnapshot {
            try requireZero(&request.reserved);
            const session_index = try sessionIndex(request.session_slot);
            if (self.sessions[session_index] != null)
                return error.SessionSlotOccupied;
            const model_record = try self.model(request.model_slot);
            if (request.kv_base == 0) return error.InvalidPhysicalAddress;

            const session_state = try command.SessionState.init(
                request.model_slot,
                request.session_slot,
                model_record.spec(),
                request.epoch,
                request.kv_base,
                request.kv_capacity_tokens,
            );
            const kv_span = try session_state.kvAllocation(model_record.spec());
            try self.rejectConflicts(&.{kv_span}, request.session_slot);
            self.sessions[session_index] = session_state;
            return snapshot(session_state);
        }

        pub fn session(
            self: *const Self,
            session_slot: u16,
        ) Error!SessionSnapshot {
            const index = try sessionIndex(session_slot);
            return if (self.sessions[index]) |session_state|
                snapshot(session_state)
            else
                error.SessionSlotEmpty;
        }

        pub fn close(self: *Self, session_slot: u16) Error!void {
            const index = try sessionIndex(session_slot);
            const session_state = self.sessions[index] orelse
                return error.SessionSlotEmpty;
            if (session_state.in_flight or
                self.active_session_slot == session_slot)
                return error.SessionBusy;
            self.sessions[index] = null;
        }

        pub fn begin(
            self: *Self,
            execute: command.ExecuteTile,
        ) Error!ExecutionLease {
            const session_index = try sessionIndex(execute.session_slot);
            const session_state = if (self.sessions[session_index]) |*value|
                value
            else
                return error.SessionSlotEmpty;
            const model_record = try self.model(execute.model_slot);
            if (self.active_session_slot != null) return error.EngineBusy;

            const committed_before = session_state.committed_tokens;
            try session_state.begin(execute, model_record.spec());
            self.active_session_slot = execute.session_slot;
            return .{
                .request_id = execute.request_id,
                .model_hash = model_record.header.model_hash,
                .kv_base = session_state.kv_base,
                .first_position = execute.first_position,
                .pending_end = session_state.pending_end,
                .kv_capacity_tokens = session_state.kv_capacity_tokens,
                .session_epoch = session_state.epoch,
                .model_slot = execute.model_slot,
                .session_slot = execute.session_slot,
                .spec_id = model_record.header.spec_id,
                .contract_version = model_spec.contract_version,
                .weight_format = model_record.header.weight_format,
                .valid_tokens = execute.valid_tokens,
                .flags = execute.flags,
                .committed_tokens = committed_before,
            };
        }

        pub fn commit(
            self: *Self,
            request: CommitRequest,
        ) Error!SessionSnapshot {
            try requireZero(&request.reserved);
            if (request.kv_writes_fenced > 1)
                return error.NonCanonicalRecord;
            const session_index = try sessionIndex(request.session_slot);
            if (self.active_session_slot != request.session_slot)
                return error.NotActiveSession;
            const session_state = if (self.sessions[session_index]) |*value|
                value
            else
                return error.SessionSlotEmpty;
            const model_record = try self.model(session_state.model_slot);
            try session_state.commit(model_record.spec(), .{
                .request_id = request.request_id,
                .completed_layers = request.completed_layers,
                .kv_writes_fenced = request.kv_writes_fenced == 1,
            });
            self.active_session_slot = null;
            return snapshot(session_state.*);
        }

        pub fn abort(
            self: *Self,
            request: AbortRequest,
        ) Error!SessionSnapshot {
            try requireZero(&request.reserved);
            const session_index = try sessionIndex(request.session_slot);
            if (self.active_session_slot != request.session_slot)
                return error.NotActiveSession;
            const session_state = if (self.sessions[session_index]) |*value|
                value
            else
                return error.SessionSlotEmpty;
            try session_state.abort(request.request_id);
            self.active_session_slot = null;
            return snapshot(session_state.*);
        }

        pub fn reset(
            self: *Self,
            request: ResetSessionRequest,
        ) Error!SessionSnapshot {
            try requireZero(&request.reserved);
            const session_index = try sessionIndex(request.session_slot);
            const session_state = if (self.sessions[session_index]) |*value|
                value
            else
                return error.SessionSlotEmpty;
            try session_state.reset(request.next_epoch);
            return snapshot(session_state.*);
        }

        fn modelIndex(slot: u16) Error!usize {
            const index: usize = slot;
            if (index >= model_slot_count) return error.InvalidModelSlot;
            return index;
        }

        fn sessionIndex(slot: u16) Error!usize {
            const index: usize = slot;
            if (index >= session_slot_count)
                return error.InvalidSessionSlot;
            return index;
        }

        fn rejectConflicts(
            self: *const Self,
            candidate: []const model_spec.DdrSpan,
            skip_session_slot: ?u16,
        ) Error!void {
            var model_span_storage: [model_spec.ModelAddressTable.max_spans]model_spec.DdrSpan = undefined;
            for (&self.models) |*entry| {
                if (entry.*) |*record| {
                    const identity = ModelIdentity{
                        .spec = record.spec(),
                        .format = record.format(),
                    };
                    const existing = try collectModelSpans(
                        record.header,
                        record.layers(),
                        identity,
                        &model_span_storage,
                    );
                    try rejectOverlap(candidate, existing);
                }
            }

            for (self.sessions) |maybe_session| {
                if (maybe_session) |session_state| {
                    if (skip_session_slot != null and
                        skip_session_slot.? == session_state.session_slot)
                        continue;
                    const session_spec = model_spec.get(session_state.spec_id);
                    const kv_span = try session_state.kvAllocation(session_spec);
                    try rejectOverlap(candidate, &.{kv_span});
                }
            }
        }
    };
}

pub const State = Registry(default_model_slots, default_session_slots);

fn validateModelInstall(install_request: ModelInstall) Error!ModelIdentity {
    const header = install_request.header;
    try requireZero(&header.reserved);
    if (header.interface_version != interface_version)
        return error.InterfaceMismatch;
    if (header.image_layout_hash != image_layout_hash)
        return error.LayoutMismatch;
    if (header.model_hash == 0) return error.InvalidModelHash;

    const spec_id = std.enums.fromInt(
        model_spec.SpecId,
        header.spec_id,
    ) orelse return error.UnsupportedSpec;
    const format = std.enums.fromInt(
        model_spec.WeightFormat,
        header.weight_format,
    ) orelse return error.UnsupportedWeightFormat;
    const spec = model_spec.get(spec_id);
    try spec.validate();
    if (header.layer_count != spec.layers or
        install_request.layers.len != spec.layers)
        return error.InvalidLayerCount;

    const table = model_spec.ModelAddressTable{
        .spec_id = spec_id,
        .weight_format = format,
        .embedding = header.embedding,
        .lm_head = header.lm_head,
        .output_norm = header.output_norm,
        .rope_table = header.rope_table,
        .layers = install_request.layers,
    };
    try table.validate();
    return .{ .spec = spec, .format = format };
}

fn collectModelSpans(
    header: ModelInstallHeader,
    layers: []const model_spec.LayerAddresses,
    identity: ModelIdentity,
    storage: *[model_spec.ModelAddressTable.max_spans]model_spec.DdrSpan,
) Error![]const model_spec.DdrSpan {
    var count: usize = 0;
    const embedding_bytes = try model_spec.packedMatrixBytes(
        identity.format,
        identity.spec.vocab_size,
        identity.spec.model_dim,
    );
    try appendPhysicalSpan(storage, &count, .{
        .address = header.embedding,
        .bytes = embedding_bytes,
    });
    if (!identity.spec.tied_embeddings) {
        try appendPhysicalSpan(storage, &count, .{
            .address = header.lm_head,
            .bytes = embedding_bytes,
        });
    }
    try appendPhysicalSpan(storage, &count, .{
        .address = header.output_norm,
        .bytes = @as(u64, identity.spec.model_dim) * model_spec.f32_bytes,
    });
    try appendPhysicalSpan(storage, &count, .{
        .address = header.rope_table,
        .bytes = try identity.spec.ropeTableBytes(),
    });
    for (layers) |layer| {
        const spans = try layer.spans(identity.spec, identity.format);
        for (spans) |span| try appendPhysicalSpan(storage, &count, span);
    }
    return storage[0..count];
}

fn appendPhysicalSpan(
    storage: *[model_spec.ModelAddressTable.max_spans]model_spec.DdrSpan,
    count: *usize,
    span: model_spec.DdrSpan,
) Error!void {
    if (span.address == 0) return error.InvalidPhysicalAddress;
    try span.validate();
    storage[count.*] = span;
    count.* += 1;
}

fn rejectOverlap(
    candidate: []const model_spec.DdrSpan,
    existing: []const model_spec.DdrSpan,
) Error!void {
    for (candidate) |candidate_span|
        for (existing) |existing_span|
            if (try candidate_span.overlaps(existing_span))
                return error.PhysicalAddressConflict;
}

fn requireZero(bytes: []const u8) Error!void {
    for (bytes) |byte| if (byte != 0) return error.NonCanonicalRecord;
}

fn snapshot(session_state: command.SessionState) SessionSnapshot {
    return .{
        .kv_base = session_state.kv_base,
        .pending_request_id = session_state.pending_request_id,
        .epoch = session_state.epoch,
        .kv_capacity_tokens = session_state.kv_capacity_tokens,
        .committed_tokens = session_state.committed_tokens,
        .pending_end = session_state.pending_end,
        .model_slot = session_state.model_slot,
        .session_slot = session_state.session_slot,
        .spec_id = @intFromEnum(session_state.spec_id),
        .present = 1,
        .in_flight = @intFromBool(session_state.in_flight),
    };
}
