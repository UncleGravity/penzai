const std = @import("std");
const state = @import("state.zig");

const model_spec = state.model_spec;
const command = state.command;

const TestModel = struct {
    header: state.ModelInstallHeader,
    layer_storage: [model_spec.max_layers]model_spec.LayerAddresses,
    end_address: u64,

    fn install(self: *const TestModel) state.ModelInstall {
        return .{
            .header = self.header,
            .layers = self.layer_storage[0..self.header.layer_count],
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

fn take(cursor: *u64, bytes: u64) !u64 {
    const address = std.mem.alignForward(u64, cursor.*, model_spec.ddr_alignment);
    cursor.* = try std.math.add(u64, address, bytes);
    return address;
}

fn makeModel(
    spec: model_spec.ModelSpec,
    format: model_spec.WeightFormat,
    model_slot: u16,
    base: u64,
    model_hash: u64,
) !TestModel {
    var cursor = base;
    const table_bytes = try model_spec.packedMatrixBytes(
        format,
        spec.vocab_size,
        spec.model_dim,
    );
    const embedding = try take(&cursor, table_bytes);
    const lm_head = if (spec.tied_embeddings)
        embedding
    else
        try take(&cursor, table_bytes);
    var result = TestModel{
        .header = .{
            .model_hash = model_hash,
            .image_layout_hash = state.image_layout_hash,
            .embedding = embedding,
            .lm_head = lm_head,
            .output_norm = try take(
                &cursor,
                @as(u64, spec.model_dim) * model_spec.f32_bytes,
            ),
            .rope_table = try take(&cursor, try spec.ropeTableBytes()),
            .interface_version = state.interface_version,
            .model_slot = model_slot,
            .spec_id = @intFromEnum(spec.id),
            .layer_count = spec.layers,
            .weight_format = @intFromEnum(format),
        },
        .layer_storage = @splat(zero_layer),
        .end_address = 0,
    };
    for (result.layer_storage[0..spec.layers]) |*layer| {
        layer.* = .{
            .fused_qkv = try take(
                &cursor,
                try spec.packedQkvBundleBytes(format),
            ),
            .attention_output = try take(
                &cursor,
                try model_spec.packedMatrixBytes(
                    format,
                    spec.model_dim,
                    spec.attentionValueWidth(),
                ),
            ),
            .fused_gate_up = try take(
                &cursor,
                try spec.packedGateUpBundleBytes(format),
            ),
            .ffn_down = try take(
                &cursor,
                try model_spec.packedMatrixBytes(
                    format,
                    spec.model_dim,
                    spec.ffn_dim,
                ),
            ),
            .attention_norm = try take(
                &cursor,
                @as(u64, spec.model_dim) * model_spec.f32_bytes,
            ),
            .attention_q_norm = try take(
                &cursor,
                @as(u64, spec.key_head_dim) * model_spec.f32_bytes,
            ),
            .attention_k_norm = try take(
                &cursor,
                @as(u64, spec.key_head_dim) * model_spec.f32_bytes,
            ),
            .ffn_norm = try take(
                &cursor,
                @as(u64, spec.model_dim) * model_spec.f32_bytes,
            ),
        };
    }
    result.end_address = cursor;
    return result;
}

fn alignedAfter(address: u64, gap: u64) u64 {
    return std.mem.alignForward(u64, address + gap, model_spec.ddr_alignment);
}

fn makeSession(
    model_slot: u16,
    session_slot: u16,
    epoch: u32,
    kv_base: u64,
    capacity: u32,
) state.OpenSessionRequest {
    return .{
        .kv_base = kv_base,
        .kv_capacity_tokens = capacity,
        .epoch = epoch,
        .model_slot = model_slot,
        .session_slot = session_slot,
    };
}

fn makeCommand(
    session: state.SessionSnapshot,
    request_id: u64,
    valid_tokens: u8,
) command.ExecuteTile {
    var token_ids: [model_spec.token_tile_max]u32 = @splat(0);
    for (token_ids[0..valid_tokens], 0..) |*token_id, index|
        token_id.* = @intCast(index + 100);
    return .{
        .request_id = request_id,
        .model_slot = session.model_slot,
        .session_slot = session.session_slot,
        .session_epoch = session.epoch,
        .first_position = session.committed_tokens,
        .valid_tokens = valid_tokens,
        .flags = command.Flags.emit_logits,
        .token_ids = token_ids,
    };
}

test "daemon records have fixed canonical serialization layouts" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(state.ModelInstallHeader));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(state.ModelInstallHeader, "model_hash"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(state.ModelInstallHeader, "image_layout_hash"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(state.ModelInstallHeader, "embedding"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(state.ModelInstallHeader, "interface_version"));
    try std.testing.expectEqual(@as(usize, 52), @offsetOf(state.ModelInstallHeader, "model_slot"));
    try std.testing.expectEqual(@as(usize, 58), @offsetOf(state.ModelInstallHeader, "weight_format"));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(model_spec.LayerAddresses));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(state.OpenSessionRequest));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(state.CommitRequest));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(state.AbortRequest));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(state.ResetSessionRequest));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(state.SessionSnapshot));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(state.ExecutionLease));
}

test "model install validates the complete immutable physical image" {
    const Registry = state.Registry(2, 4);
    var registry = Registry.init();
    var first = try makeModel(
        model_spec.bonsai_1_7b,
        .q1_0,
        0,
        0x0010_0000,
        0x1111_2222_3333_4444,
    );
    try registry.validateInstall(first.install());
    const installed = try registry.install(first.install());
    const original_qkv = installed.layers()[0].fused_qkv;
    first.layer_storage[0].fused_qkv += model_spec.ddr_alignment;
    try std.testing.expectEqual(
        original_qkv,
        (try registry.model(0)).layers()[0].fused_qkv,
    );
    try std.testing.expectEqual(
        model_spec.SpecId.bonsai_1_7b,
        (try registry.model(0)).spec().id,
    );
    try std.testing.expectError(
        error.ModelSlotOccupied,
        registry.install(first.install()),
    );

    var malformed = try makeModel(
        model_spec.bonsai_4b,
        .q2_0,
        1,
        alignedAfter(first.end_address, 0x10000),
        0x5555_6666_7777_8888,
    );
    malformed.header.reserved[0] = 1;
    try std.testing.expectError(
        error.NonCanonicalRecord,
        registry.validateInstall(malformed.install()),
    );
    malformed.header.reserved[0] = 0;
    malformed.header.interface_version +%= 1;
    try std.testing.expectError(
        error.InterfaceMismatch,
        registry.validateInstall(malformed.install()),
    );
    malformed.header.interface_version = state.interface_version;
    malformed.header.image_layout_hash ^= 1;
    try std.testing.expectError(
        error.LayoutMismatch,
        registry.validateInstall(malformed.install()),
    );
    malformed.header.image_layout_hash = state.image_layout_hash;
    malformed.header.model_hash = 0;
    try std.testing.expectError(
        error.InvalidModelHash,
        registry.validateInstall(malformed.install()),
    );
    malformed.header.model_hash = 9;
    malformed.header.embedding = 0;
    malformed.header.lm_head = 0;
    try std.testing.expectError(
        error.InvalidPhysicalAddress,
        registry.validateInstall(malformed.install()),
    );

    var overlap = try makeModel(
        model_spec.bonsai_1_7b,
        .q1_0,
        1,
        0x0010_0000,
        10,
    );
    try std.testing.expectError(
        error.PhysicalAddressConflict,
        registry.validateInstall(overlap.install()),
    );
    overlap.header.spec_id = 99;
    try std.testing.expectError(
        error.UnsupportedSpec,
        registry.validateInstall(overlap.install()),
    );
}

test "model and KV allocations cannot alias across slots" {
    const Registry = state.Registry(2, 4);
    var registry = Registry.init();
    var model = try makeModel(
        model_spec.bonsai_1_7b,
        .q1_0,
        0,
        0x0020_0000,
        1,
    );
    _ = try registry.install(model.install());
    const kv_base = alignedAfter(model.end_address, 0x20_0000);
    const opened = try registry.open(makeSession(0, 0, 7, kv_base, 32));
    try std.testing.expectEqual(@as(u32, 0), opened.committed_tokens);
    try std.testing.expectEqual(@as(u8, 1), opened.present);

    try std.testing.expectError(
        error.SessionSlotOccupied,
        registry.open(makeSession(0, 0, 8, kv_base + 0x1000_0000, 32)),
    );
    try std.testing.expectError(
        error.PhysicalAddressConflict,
        registry.open(makeSession(0, 1, 1, kv_base, 32)),
    );
    try std.testing.expectError(
        error.PhysicalAddressConflict,
        registry.open(makeSession(0, 1, 1, model.header.embedding, 1)),
    );
    var noncanonical = makeSession(0, 1, 1, kv_base + 0x1000_0000, 32);
    noncanonical.reserved[3] = 1;
    try std.testing.expectError(
        error.NonCanonicalRecord,
        registry.open(noncanonical),
    );
    try std.testing.expectError(
        error.ModelSlotEmpty,
        registry.open(makeSession(1, 1, 1, kv_base + 0x1000_0000, 32)),
    );

    var colliding_model = try makeModel(
        model_spec.bonsai_1_7b,
        .q2_0,
        1,
        kv_base,
        2,
    );
    try std.testing.expectError(
        error.PhysicalAddressConflict,
        registry.install(colliding_model.install()),
    );
    colliding_model.header.model_slot = 9;
    try std.testing.expectError(
        error.InvalidModelSlot,
        registry.install(colliding_model.install()),
    );
}

test "session close and model uninstall preserve physical ownership" {
    const Registry = state.Registry(1, 2);
    var registry = Registry.init();
    var model = try makeModel(
        model_spec.bonsai_1_7b,
        .q1_0,
        0,
        0x0030_0000,
        0x55aa,
    );
    _ = try registry.install(model.install());
    const kv_base = alignedAfter(model.end_address, 0x20_0000);
    const session = try registry.open(makeSession(0, 0, 3, kv_base, 32));

    try std.testing.expectError(error.ModelInUse, registry.uninstall(0));
    _ = try registry.begin(makeCommand(session, 77, 1));
    try std.testing.expectError(error.SessionBusy, registry.close(0));
    try std.testing.expectError(error.ModelInUse, registry.uninstall(0));

    _ = try registry.abort(.{ .request_id = 77, .session_slot = 0 });
    try registry.close(0);
    try std.testing.expectError(error.SessionSlotEmpty, registry.close(0));
    try registry.uninstall(0);
    try std.testing.expectError(error.ModelSlotEmpty, registry.uninstall(0));
    try std.testing.expectError(error.ModelSlotEmpty, registry.model(0));
}

test "begin commit abort and reset preserve transactional watermarks" {
    const Registry = state.Registry(1, 3);
    var registry = Registry.init();
    var model = try makeModel(
        model_spec.bonsai_1_7b,
        .q1_0,
        0,
        0x0040_0000,
        0x1234,
    );
    _ = try registry.install(model.install());
    const kv0 = alignedAfter(model.end_address, 0x20_0000);
    const kv_bytes = try model_spec.bonsai_1_7b.kvCacheBytes(32);
    var session0 = try registry.open(makeSession(0, 0, 7, kv0, 32));
    const session1 = try registry.open(makeSession(
        0,
        1,
        9,
        alignedAfter(kv0 + kv_bytes, 0x20_0000),
        32,
    ));

    const first = makeCommand(session0, 1001, 8);
    const lease = try registry.begin(first);
    try std.testing.expectEqual(first.request_id, lease.request_id);
    try std.testing.expectEqual(@as(u32, 0), lease.committed_tokens);
    try std.testing.expectEqual(@as(u32, 8), lease.pending_end);
    try std.testing.expectEqual(model.header.model_hash, lease.model_hash);
    try std.testing.expectEqual(@as(u16, model_spec.contract_version), lease.contract_version);
    try std.testing.expectEqual(@as(u8, 8), lease.valid_tokens);
    try std.testing.expectEqual(@as(u8, 1), (try registry.session(0)).in_flight);

    try std.testing.expectError(
        error.EngineBusy,
        registry.begin(makeCommand(session1, 2001, 1)),
    );
    try std.testing.expectError(
        error.SessionBusy,
        registry.reset(.{ .next_epoch = 8, .session_slot = 0 }),
    );
    try std.testing.expectError(
        error.NotActiveSession,
        registry.commit(.{
            .request_id = 2001,
            .session_slot = 1,
            .completed_layers = model_spec.bonsai_1_7b.layers,
            .kv_writes_fenced = 1,
        }),
    );
    try std.testing.expectError(
        error.IncompleteLayerWalk,
        registry.commit(.{
            .request_id = 1001,
            .session_slot = 0,
            .completed_layers = model_spec.bonsai_1_7b.layers - 1,
            .kv_writes_fenced = 1,
        }),
    );
    try std.testing.expectError(
        error.KvWritesNotFenced,
        registry.commit(.{
            .request_id = 1001,
            .session_slot = 0,
            .completed_layers = model_spec.bonsai_1_7b.layers,
            .kv_writes_fenced = 0,
        }),
    );
    try std.testing.expectError(
        error.NonCanonicalRecord,
        registry.commit(.{
            .request_id = 1001,
            .session_slot = 0,
            .completed_layers = model_spec.bonsai_1_7b.layers,
            .kv_writes_fenced = 2,
        }),
    );
    session0 = try registry.commit(.{
        .request_id = 1001,
        .session_slot = 0,
        .completed_layers = model_spec.bonsai_1_7b.layers,
        .kv_writes_fenced = 1,
    });
    try std.testing.expectEqual(@as(u32, 8), session0.committed_tokens);
    try std.testing.expectEqual(@as(u8, 0), session0.in_flight);
    try std.testing.expectError(error.InvalidWatermark, registry.begin(first));

    const second = makeCommand(session0, 1002, 1);
    _ = try registry.begin(second);
    try std.testing.expectError(
        error.RequestMismatch,
        registry.abort(.{ .request_id = 9999, .session_slot = 0 }),
    );
    session0 = try registry.abort(.{
        .request_id = 1002,
        .session_slot = 0,
    });
    try std.testing.expectEqual(@as(u32, 8), session0.committed_tokens);
    try std.testing.expectEqual(@as(u64, 0), session0.pending_request_id);

    _ = try registry.begin(makeCommand(session0, 1003, 1));
    try std.testing.expectError(
        error.SessionBusy,
        registry.reset(.{ .next_epoch = 8, .session_slot = 0 }),
    );
    _ = try registry.abort(.{ .request_id = 1003, .session_slot = 0 });
    session0 = try registry.reset(.{ .next_epoch = 8, .session_slot = 0 });
    try std.testing.expectEqual(@as(u32, 0), session0.committed_tokens);
    try std.testing.expectEqual(@as(u32, 8), session0.epoch);
    try std.testing.expectError(
        error.InvalidSessionEpoch,
        registry.reset(.{ .next_epoch = 8, .session_slot = 0 }),
    );

    var stale = makeCommand(session0, 1004, 1);
    stale.session_epoch = 7;
    try std.testing.expectError(error.InvalidSessionEpoch, registry.begin(stale));
    var bad_reset = state.ResetSessionRequest{
        .next_epoch = 10,
        .session_slot = 0,
    };
    bad_reset.reserved[0] = 1;
    try std.testing.expectError(
        error.NonCanonicalRecord,
        registry.reset(bad_reset),
    );
}
