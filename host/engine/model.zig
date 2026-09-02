//! Host-only semantic model binding and immutable inference engine image plan.

const std = @import("std");
const shared = @import("shared");

pub const contract = shared.engine.model_spec;

pub const Error = contract.SpecError || error{
    InvalidTensor,
    InvalidTensorShape,
    InvalidTensorLength,
    MixedWeightFormat,
    DuplicateTensor,
    SourceAlias,
    OutputTooSmall,
};

pub const TensorShape = struct {
    ne: [4]u32,

    pub fn matrix(k: u32, rows: u32) TensorShape {
        return .{ .ne = .{ k, rows, 1, 1 } };
    }

    pub fn vector(values: u32) TensorShape {
        return .{ .ne = .{ values, 1, 1, 1 } };
    }

    pub fn isMatrix(self: TensorShape, k: u32, rows: u32) bool {
        return std.meta.eql(self.ne, [4]u32{ k, rows, 1, 1 });
    }

    pub fn isVector(self: TensorShape, values: u32) bool {
        return std.meta.eql(self.ne, [4]u32{ values, 1, 1, 1 });
    }
};

/// A source tensor identity is the authoritative loader's tensor pointer cast
/// to usize. Its bytes are borrowed only during synchronous model provisioning.
pub const LowBitTensor = struct {
    identity: usize,
    format: contract.WeightFormat,
    shape: TensorShape,
    bytes: []const u8,

    pub fn validateMatrix(self: LowBitTensor, format: contract.WeightFormat, rows: u32, k: u32) Error!void {
        if (self.identity == 0) return error.InvalidTensor;
        if (self.format != format) return error.MixedWeightFormat;
        if (!self.shape.isMatrix(k, rows)) return error.InvalidTensorShape;
        const expected = try contract.sourceMatrixBytes(format, rows, k);
        if (self.bytes.len != expected) return error.InvalidTensorLength;
    }
};

pub const F32Tensor = struct {
    identity: usize,
    shape: TensorShape,
    bytes: []const u8,

    pub fn validateVector(self: F32Tensor, values: u32) Error!void {
        if (self.identity == 0) return error.InvalidTensor;
        if (!self.shape.isVector(values)) return error.InvalidTensorShape;
        const expected = std.math.mul(u64, values, @sizeOf(f32)) catch
            return error.Overflow;
        if (self.bytes.len != expected) return error.InvalidTensorLength;
    }
};

/// Semantic roles are supplied by the model loader, not inferred from names.
pub const LayerSources = struct {
    query: LowBitTensor,
    key: LowBitTensor,
    value: LowBitTensor,
    attention_output: LowBitTensor,
    gate: LowBitTensor,
    up: LowBitTensor,
    ffn_down: LowBitTensor,
    attention_norm: F32Tensor,
    attention_q_norm: F32Tensor,
    attention_k_norm: F32Tensor,
    ffn_norm: F32Tensor,

    pub fn validate(self: LayerSources, spec: contract.ModelSpec, format: contract.WeightFormat) Error!void {
        try self.query.validateMatrix(format, spec.queryWidth(), spec.model_dim);
        try self.key.validateMatrix(format, spec.kvKeyWidth(), spec.model_dim);
        try self.value.validateMatrix(format, spec.kvValueWidth(), spec.model_dim);
        try self.attention_output.validateMatrix(format, spec.model_dim, spec.attentionValueWidth());
        try self.gate.validateMatrix(format, spec.ffn_dim, spec.model_dim);
        try self.up.validateMatrix(format, spec.ffn_dim, spec.model_dim);
        try self.ffn_down.validateMatrix(format, spec.model_dim, spec.ffn_dim);
        try self.attention_norm.validateVector(spec.model_dim);
        try self.attention_q_norm.validateVector(spec.key_head_dim);
        try self.attention_k_norm.validateVector(spec.key_head_dim);
        try self.ffn_norm.validateVector(spec.model_dim);
    }
};

pub const ModelSources = struct {
    metadata: contract.GgufMetadata,
    weight_format: contract.WeightFormat,
    embedding: LowBitTensor,
    /// Absent or backed by the exact embedding bytes for tied 1.7B/4B models;
    /// required and physically distinct for 8B. The loader may expose separate
    /// ggml tensor identities for the two tied semantic roles.
    lm_head: ?LowBitTensor,
    output_norm: F32Tensor,
    layers: []const LayerSources,
    pub fn spec(self: ModelSources) Error!contract.ModelSpec {
        return contract.matchGgufMetadata(self.metadata);
    }

    /// Validate the complete semantic model and all source ownership before any
    /// resident allocation or upload begins.
    pub fn validate(self: ModelSources) Error!contract.ModelSpec {
        const selected = try self.spec();
        try selected.validate();
        if (self.layers.len != selected.layers) return error.InvalidLayerCount;
        try self.embedding.validateMatrix(
            self.weight_format,
            selected.vocab_size,
            selected.model_dim,
        );
        try validateLmHead(selected, self.weight_format, self.embedding, self.lm_head);
        try self.output_norm.validateVector(selected.model_dim);
        for (self.layers) |layer| try layer.validate(selected, self.weight_format);
        try validateNoSourceAliases(self);
        return selected;
    }
};

pub fn validateLmHead(
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    embedding: LowBitTensor,
    lm_head: ?LowBitTensor,
) Error!void {
    try embedding.validateMatrix(format, spec.vocab_size, spec.model_dim);
    if (lm_head) |head| {
        try head.validateMatrix(format, spec.vocab_size, spec.model_dim);
        const exact_alias = sameBackingSource(embedding, head);
        if (spec.tied_embeddings != exact_alias)
            return error.InvalidWeightTying;
    } else if (!spec.tied_embeddings) {
        return error.InvalidWeightTying;
    }
}

test "LM-head source policy is exact for tied and untied specs" {
    var first: [72]u8 = undefined;
    var second: [72]u8 = undefined;
    const embedding = LowBitTensor{
        .identity = 1,
        .format = .q1_0,
        .shape = TensorShape.matrix(128, 4),
        .bytes = &first,
    };
    const same = embedding;
    const tied_role = LowBitTensor{
        .identity = 3,
        .format = .q1_0,
        .shape = TensorShape.matrix(128, 4),
        .bytes = &first,
    };
    const distinct = LowBitTensor{
        .identity = 2,
        .format = .q1_0,
        .shape = TensorShape.matrix(128, 4),
        .bytes = &second,
    };
    var spec = contract.ModelSpec{
        .id = .bonsai_1_7b,
        .name = "synthetic",
        .layers = 1,
        .context_length = 1,
        .model_dim = 128,
        .ffn_dim = 128,
        .query_heads = 1,
        .kv_heads = 1,
        .key_head_dim = 128,
        .value_head_dim = 128,
        .vocab_size = 4,
        .tied_embeddings = true,
        .rope_original_context = 1,
        .rope_scaling_factor = 1,
        .rope_freq_base = 10_000,
        .rms_epsilon = 0.000001,
    };
    try validateLmHead(spec, .q1_0, embedding, null);
    try validateLmHead(spec, .q1_0, embedding, same);
    try validateLmHead(spec, .q1_0, embedding, tied_role);
    try std.testing.expectError(error.InvalidWeightTying, validateLmHead(spec, .q1_0, embedding, distinct));

    spec.tied_embeddings = false;
    try std.testing.expectError(error.InvalidWeightTying, validateLmHead(spec, .q1_0, embedding, null));
    try std.testing.expectError(error.InvalidWeightTying, validateLmHead(spec, .q1_0, embedding, same));
    try std.testing.expectError(error.InvalidWeightTying, validateLmHead(spec, .q1_0, embedding, tied_role));
    try validateLmHead(spec, .q1_0, embedding, distinct);
}

fn sameSource(a: LowBitTensor, b: LowBitTensor) bool {
    return a.identity == b.identity and
        sameBackingSource(a, b);
}

fn sameBackingSource(a: LowBitTensor, b: LowBitTensor) bool {
    return a.bytes.ptr == b.bytes.ptr and
        a.bytes.len == b.bytes.len and
        a.format == b.format and
        std.meta.eql(a.shape.ne, b.shape.ne);
}

pub const SourceRef = struct {
    identity: usize,
    bytes: []const u8,
};

pub const max_source_tensors: usize = 3 + @as(usize, contract.max_layers) * 11;

fn appendSource(refs: *[max_source_tensors]SourceRef, count: *usize, source: anytype) void {
    refs[count.*] = .{ .identity = source.identity, .bytes = source.bytes };
    count.* += 1;
}

fn validateNoSourceAliases(sources: ModelSources) Error!void {
    var refs: [max_source_tensors]SourceRef = undefined;
    const collected = collectSourceRefs(sources, &refs);
    const tied_alias = if (sources.lm_head) |head|
        if (sameBackingSource(sources.embedding, head) and
            sources.embedding.identity != head.identity)
            [2]usize{ sources.embedding.identity, head.identity }
        else
            null
    else
        null;
    try validateDistinctSourcesAllowing(collected, tied_alias);
}

/// Copy the complete unique semantic source set into caller-owned storage.
/// The returned spans remain borrowed from the authoritative model mapping.
pub fn collectSourceRefs(
    sources: ModelSources,
    refs: *[max_source_tensors]SourceRef,
) []const SourceRef {
    var count: usize = 0;
    appendSource(refs, &count, sources.embedding);
    if (sources.lm_head) |head|
        if (!sameSource(sources.embedding, head)) appendSource(refs, &count, head);
    appendSource(refs, &count, sources.output_norm);
    for (sources.layers) |layer| {
        inline for (std.meta.fields(LayerSources)) |field|
            appendSource(refs, &count, @field(layer, field.name));
    }
    return refs[0..count];
}

fn validateDistinctSources(refs: []const SourceRef) Error!void {
    return validateDistinctSourcesAllowing(refs, null);
}

fn validateDistinctSourcesAllowing(
    refs: []const SourceRef,
    exact_alias: ?[2]usize,
) Error!void {
    for (refs, 0..) |current, index| {
        if (current.identity == 0 or current.bytes.len == 0) return error.InvalidTensor;
        const current_start = @intFromPtr(current.bytes.ptr);
        const current_end = std.math.add(usize, current_start, current.bytes.len) catch
            return error.InvalidTensorLength;
        for (refs[0..index]) |prior| {
            if (current.identity == prior.identity) return error.DuplicateTensor;
            const prior_start = @intFromPtr(prior.bytes.ptr);
            const prior_end = std.math.add(usize, prior_start, prior.bytes.len) catch
                return error.InvalidTensorLength;
            if (current_start < prior_end and prior_start < current_end) {
                const allowed = if (exact_alias) |pair|
                    ((current.identity == pair[0] and prior.identity == pair[1]) or
                        (current.identity == pair[1] and prior.identity == pair[0])) and
                        current_start == prior_start and current.bytes.len == prior.bytes.len
                else
                    false;
                if (!allowed) return error.SourceAlias;
            }
        }
    }
}

pub const ImagePlan = struct {
    spec_id: contract.SpecId,
    weight_format: contract.WeightFormat,
    base_address: u64,
    image_bytes: u64,
    embedding: u64,
    lm_head: u64,
    output_norm: u64,
    rope_table: u64,
    layer_count: u16,
    layer_storage: [contract.max_layers]contract.LayerAddresses,

    pub fn spec(self: ImagePlan) contract.ModelSpec {
        return contract.get(self.spec_id);
    }

    pub fn layers(self: *const ImagePlan) []const contract.LayerAddresses {
        return self.layer_storage[0..self.layer_count];
    }

    pub fn table(self: *const ImagePlan) contract.ModelAddressTable {
        return .{
            .spec_id = self.spec_id,
            .weight_format = self.weight_format,
            .embedding = self.embedding,
            .lm_head = self.lm_head,
            .output_norm = self.output_norm,
            .rope_table = self.rope_table,
            .layers = self.layers(),
        };
    }

    pub fn addressTableBytes(self: ImagePlan) usize {
        return @intCast(contract.addressTablePayloadBytes(self.spec()));
    }

    /// Physical payload: four global u64 addresses, followed by exactly one
    /// 64-byte LayerAddresses record per layer, all little-endian.
    pub fn encodeAddressTable(self: *const ImagePlan, out: []u8) Error![]u8 {
        const needed = self.addressTableBytes();
        if (out.len < needed) return error.OutputTooSmall;
        var cursor: usize = 0;
        putU64(out, &cursor, self.embedding);
        putU64(out, &cursor, self.lm_head);
        putU64(out, &cursor, self.output_norm);
        putU64(out, &cursor, self.rope_table);
        for (self.layers()) |layer| {
            inline for (std.meta.fields(contract.LayerAddresses)) |field|
                putU64(out, &cursor, @field(layer, field.name));
        }
        std.debug.assert(cursor == needed);
        return out[0..needed];
    }
};

fn putU64(out: []u8, cursor: *usize, value: u64) void {
    std.mem.writeInt(u64, out[cursor.*..][0..8], value, .little);
    cursor.* += 8;
}

pub fn planImage(spec: contract.ModelSpec, format: contract.WeightFormat, base_address: u64) Error!ImagePlan {
    try spec.validate();
    if (!sameSpecShape(spec, contract.get(spec.id)))
        return error.UnsupportedSpec;
    const shared_plan = try contract.planModelImage(spec.id, format, base_address);
    const result = ImagePlan{
        .spec_id = shared_plan.spec_id,
        .weight_format = shared_plan.weight_format,
        .base_address = shared_plan.base_address,
        .image_bytes = shared_plan.image_bytes,
        .embedding = shared_plan.embedding,
        .lm_head = shared_plan.lm_head,
        .output_norm = shared_plan.output_norm,
        .rope_table = shared_plan.rope_table,
        .layer_count = shared_plan.layer_count,
        .layer_storage = shared_plan.layer_storage,
    };
    try result.table().validate();
    return result;
}

fn sameSpecShape(a: contract.ModelSpec, b: contract.ModelSpec) bool {
    return a.id == b.id and
        a.layers == b.layers and
        a.context_length == b.context_length and
        a.model_dim == b.model_dim and
        a.ffn_dim == b.ffn_dim and
        a.query_heads == b.query_heads and
        a.kv_heads == b.kv_heads and
        a.key_head_dim == b.key_head_dim and
        a.value_head_dim == b.value_head_dim and
        a.vocab_size == b.vocab_size and
        a.tied_embeddings == b.tied_embeddings and
        a.rope_original_context == b.rope_original_context and
        a.rope_scaling_factor == b.rope_scaling_factor and
        a.rope_freq_base == b.rope_freq_base and
        a.rms_epsilon == b.rms_epsilon;
}

test "all real specs produce non-aliasing Q1 and Q2 physical tables" {
    for (contract.all_specs) |spec| {
        for ([_]contract.WeightFormat{ .q1_0, .q2_0 }) |format| {
            const plan = try planImage(spec, format, 0x4000);
            try plan.table().validate();
            try std.testing.expectEqual(spec.layers, plan.layer_count);
            try std.testing.expectEqual(
                @as(usize, 32 + @as(usize, spec.layers) * 64),
                plan.addressTableBytes(),
            );
            try std.testing.expectEqual(
                try std.math.add(
                    u64,
                    try std.math.add(
                        u64,
                        try spec.modelPackedWeightBytes(format),
                        try spec.modelNormBytes(),
                    ),
                    try spec.ropeTableBytes(),
                ),
                plan.image_bytes,
            );
            try std.testing.expect(plan.image_bytes <= contract.k26_ddr_bytes);
            try std.testing.expectEqual(spec.tied_embeddings, plan.embedding == plan.lm_head);
        }
    }
}

test "physical table encoding is fixed and little-endian" {
    const plan = try planImage(contract.bonsai_4b, .q1_0, 0x1000);
    var bytes: [32 + contract.max_layers * 64]u8 = undefined;
    const encoded = try plan.encodeAddressTable(&bytes);
    try std.testing.expectEqual(plan.addressTableBytes(), encoded.len);
    try std.testing.expectEqual(plan.embedding, std.mem.readInt(u64, encoded[0..8], .little));
    try std.testing.expectEqual(plan.lm_head, std.mem.readInt(u64, encoded[8..16], .little));
    try std.testing.expectEqual(plan.output_norm, std.mem.readInt(u64, encoded[16..24], .little));
    try std.testing.expectEqual(plan.rope_table, std.mem.readInt(u64, encoded[24..32], .little));
    try std.testing.expectEqual(plan.layer_storage[0].fused_qkv, std.mem.readInt(u64, encoded[32..40], .little));
    try std.testing.expectEqual(plan.layer_storage[0].ffn_norm, std.mem.readInt(u64, encoded[88..96], .little));
}

test "source descriptors reject wrong shapes formats identities and aliases" {
    var bytes: [72]u8 = undefined;
    const tensor = LowBitTensor{
        .identity = 1,
        .format = .q1_0,
        .shape = TensorShape.matrix(128, 4),
        .bytes = &bytes,
    };
    try tensor.validateMatrix(.q1_0, 4, 128);
    try std.testing.expectError(error.MixedWeightFormat, tensor.validateMatrix(.q2_0, 4, 128));
    try std.testing.expectError(error.InvalidTensorShape, tensor.validateMatrix(.q1_0, 3, 128));

    try std.testing.expectError(error.DuplicateTensor, validateDistinctSources(&.{
        .{ .identity = 7, .bytes = bytes[0..16] },
        .{ .identity = 7, .bytes = bytes[32..48] },
    }));
    try std.testing.expectError(error.SourceAlias, validateDistinctSources(&.{
        .{ .identity = 7, .bytes = bytes[0..16] },
        .{ .identity = 8, .bytes = bytes[8..24] },
    }));
    try validateDistinctSources(&.{
        .{ .identity = 7, .bytes = bytes[0..16] },
        .{ .identity = 8, .bytes = bytes[16..32] },
    });
}

test "image planner rejects spec-shaped impostors" {
    var wrong = contract.bonsai_4b;
    wrong.model_dim = 4000;
    try std.testing.expectError(error.InvalidSpec, planImage(wrong, .q1_0, 0));

    wrong = contract.bonsai_4b;
    wrong.context_length /= 2;
    try std.testing.expectError(error.UnsupportedSpec, planImage(wrong, .q1_0, 0));
}

test "8B RoPE plan covers the 17-bit terminal context extent" {
    const spec = contract.bonsai_8b;
    const plan = try planImage(spec, .q1_0, 0);
    const row_bytes = @as(u64, spec.key_head_dim) * @sizeOf(f32);
    const last_position = @as(u64, spec.context_length) - 1;
    const last_row = plan.rope_table + last_position * row_bytes;
    try std.testing.expectEqual(@as(u32, 65_536), spec.context_length);
    try std.testing.expectEqual(
        plan.rope_table + try spec.ropeTableBytes(),
        last_row + row_bytes,
    );
}
