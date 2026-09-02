//! Clean current inference-engine model and resident-memory contract.
//!
//! The specs are intentionally bounded to the three observed Bonsai/Qwen3
//! shapes. Dimensions select loop bounds; they are not inferred from one
//! another. In particular, Bonsai-4B has D=2560 and Q/O width=4096.

const std = @import("std");

pub const contract_version: u16 = 5;
pub const token_tile_max: u8 = 8;
pub const physical_token_lanes: u8 = 4;
pub const weight_rows_per_block: u32 = 16;
pub const weight_group: u32 = 128;
pub const q8_group: u32 = 32;
pub const q8_padded_record_bytes: u32 = 40;
pub const ddr_alignment: u64 = 64;
pub const f16_bytes: u64 = 2;
pub const f32_bytes: u64 = 4;

pub const k26_uram_count: u64 = 64;
pub const k26_bram36_count: u64 = 144;
pub const k26_raw_onchip_bytes: u64 =
    k26_uram_count * 36 * 1024 + k26_bram36_count * 4608;
pub const k26_ddr_bytes: u64 = 4 * 1024 * 1024 * 1024;

pub const SpecId = enum(u16) {
    bonsai_1_7b = 1,
    bonsai_4b = 2,
    bonsai_8b = 3,
};

pub const WeightFormat = enum(u8) {
    q1_0 = 1,
    q2_0 = 2,

    pub fn fromGgufName(name: []const u8) error{UnsupportedWeightFormat}!WeightFormat {
        if (std.mem.eql(u8, name, "Q1_0")) return .q1_0;
        if (std.mem.eql(u8, name, "Q2_0")) return .q2_0;
        return error.UnsupportedWeightFormat;
    }

    pub fn issueBytesPerRowBlockGroup(self: WeightFormat) u32 {
        return switch (self) {
            // One 32-bit scale slot and 128 sign bits per row.
            .q1_0 => 320,
            // Two f16 scales and 256 two-bit codes per row.
            .q2_0 => 576,
        };
    }

    pub fn sourceGroup(self: WeightFormat) u32 {
        return switch (self) {
            .q1_0 => 128,
            .q2_0 => 64,
        };
    }

    pub fn sourceBlockBytes(self: WeightFormat) u32 {
        return switch (self) {
            .q1_0, .q2_0 => 18,
        };
    }
};

pub const SpecError = error{
    UnsupportedSpec,
    InvalidSpec,
    InvalidMatrixShape,
    UnsupportedWeightFormat,
    MetadataMismatch,
    Overflow,
    InvalidLayerCount,
    UnalignedAddress,
    EmptyAddress,
    AddressOverlap,
    InvalidWeightTying,
};

fn checkedAdd(a: u64, b: u64) SpecError!u64 {
    return std.math.add(u64, a, b) catch error.Overflow;
}

fn checkedMul(a: u64, b: u64) SpecError!u64 {
    return std.math.mul(u64, a, b) catch error.Overflow;
}

fn vectorBytes(count: u32) SpecError!u64 {
    return checkedMul(count, f32_bytes);
}

pub fn packedMatrixBytes(
    format: WeightFormat,
    rows: u32,
    k: u32,
) SpecError!u64 {
    if (rows == 0 or k == 0 or k % weight_group != 0)
        return error.InvalidMatrixShape;
    const rowblocks = (@as(u64, rows) + weight_rows_per_block - 1) /
        weight_rows_per_block;
    const groups = k / weight_group;
    return checkedMul(
        try checkedMul(rowblocks, groups),
        format.issueBytesPerRowBlockGroup(),
    );
}

/// Bytes in the source GGUF tensor before issue-order rowblock padding.
pub fn sourceMatrixBytes(
    format: WeightFormat,
    rows: u32,
    k: u32,
) SpecError!u64 {
    if (rows == 0 or k == 0 or k % format.sourceGroup() != 0)
        return error.InvalidMatrixShape;
    return checkedMul(
        try checkedMul(rows, k / format.sourceGroup()),
        format.sourceBlockBytes(),
    );
}

pub fn paddedQ8Bytes(tokens: u8, values_per_token: u32) SpecError!u64 {
    if (tokens == 0 or tokens > token_tile_max or
        values_per_token == 0 or values_per_token % q8_group != 0)
        return error.InvalidMatrixShape;
    return checkedMul(
        try checkedMul(tokens, values_per_token / q8_group),
        q8_padded_record_bytes,
    );
}

pub const ModelSpec = struct {
    id: SpecId,
    name: []const u8,
    layers: u16,
    context_length: u32,
    model_dim: u32,
    ffn_dim: u32,
    query_heads: u16,
    kv_heads: u16,
    key_head_dim: u16,
    value_head_dim: u16,
    vocab_size: u32,
    tied_embeddings: bool,
    rope_original_context: u32,
    rope_scaling_factor: f32,
    rope_freq_base: f32,
    rms_epsilon: f32,

    pub fn validate(self: ModelSpec) SpecError!void {
        if (self.layers == 0 or self.layers > max_layers or
            self.context_length == 0 or self.model_dim == 0 or
            self.ffn_dim == 0 or self.query_heads == 0 or
            self.kv_heads == 0 or self.key_head_dim == 0 or
            self.value_head_dim == 0 or self.vocab_size == 0 or
            self.rope_original_context == 0 or
            self.rope_original_context > self.context_length or
            !std.math.isFinite(self.rope_scaling_factor) or
            self.rope_scaling_factor < 1 or
            !std.math.isFinite(self.rope_freq_base) or
            self.rope_freq_base <= 0 or
            !std.math.isFinite(self.rms_epsilon) or
            self.rms_epsilon <= 0 or
            self.query_heads % self.kv_heads != 0 or
            self.model_dim % weight_group != 0 or
            self.ffn_dim % weight_group != 0 or
            self.queryWidth() % weight_group != 0 or
            self.kvKeyWidth() % weight_group != 0 or
            self.kvValueWidth() % weight_group != 0)
            return error.InvalidSpec;
    }

    pub fn queryWidth(self: ModelSpec) u32 {
        return @as(u32, self.query_heads) * self.key_head_dim;
    }

    pub fn attentionValueWidth(self: ModelSpec) u32 {
        return @as(u32, self.query_heads) * self.value_head_dim;
    }

    pub fn kvKeyWidth(self: ModelSpec) u32 {
        return @as(u32, self.kv_heads) * self.key_head_dim;
    }

    pub fn kvValueWidth(self: ModelSpec) u32 {
        return @as(u32, self.kv_heads) * self.value_head_dim;
    }

    pub fn queryHeadsPerKv(self: ModelSpec) u16 {
        return self.query_heads / self.kv_heads;
    }

    pub fn qkvRowsPerKvGroup(self: ModelSpec) u32 {
        return @as(u32, self.queryHeadsPerKv()) * self.key_head_dim +
            self.key_head_dim + self.value_head_dim;
    }

    pub fn qkvBundleRows(self: ModelSpec) u32 {
        return @as(u32, self.kv_heads) * self.qkvRowsPerKvGroup();
    }

    pub fn gateUpBundleRows(self: ModelSpec) u32 {
        return 2 * self.ffn_dim;
    }

    pub fn packedQkvBundleBytes(
        self: ModelSpec,
        format: WeightFormat,
    ) SpecError!u64 {
        return packedMatrixBytes(format, self.qkvBundleRows(), self.model_dim);
    }

    pub fn packedGateUpBundleBytes(
        self: ModelSpec,
        format: WeightFormat,
    ) SpecError!u64 {
        return packedMatrixBytes(format, self.gateUpBundleRows(), self.model_dim);
    }

    /// One committed position for one layer, with contiguous f16 K then V.
    pub fn kvRecordBytes(self: ModelSpec) SpecError!u64 {
        return checkedMul(
            try checkedAdd(self.kvKeyWidth(), self.kvValueWidth()),
            f16_bytes,
        );
    }

    pub fn kvCacheBytes(self: ModelSpec, capacity_tokens: u32) SpecError!u64 {
        if (capacity_tokens == 0 or capacity_tokens > self.context_length)
            return error.InvalidSpec;
        return checkedMul(
            try checkedMul(self.layers, capacity_tokens),
            try self.kvRecordBytes(),
        );
    }

    /// One FP32 {cos,sin} pair per rotary pair and absolute position. The
    /// table is shared by Q and K across every layer.
    pub fn ropeTableBytes(self: ModelSpec) SpecError!u64 {
        return checkedMul(
            try checkedMul(self.context_length, self.key_head_dim),
            f32_bytes,
        );
    }

    pub fn layerPackedWeightBytes(
        self: ModelSpec,
        format: WeightFormat,
    ) SpecError!u64 {
        var total: u64 = 0;
        const shapes = [_]MatrixShape{
            .{ .rows = self.qkvBundleRows(), .k = self.model_dim },
            .{ .rows = self.model_dim, .k = self.attentionValueWidth() },
            .{ .rows = self.gateUpBundleRows(), .k = self.model_dim },
            .{ .rows = self.model_dim, .k = self.ffn_dim },
        };
        for (shapes) |shape|
            total = try checkedAdd(total, try packedMatrixBytes(format, shape.rows, shape.k));
        return total;
    }

    /// Low-bit bytes resident for one model. Smaller specs tie the input
    /// embedding and LM-head tables; Bonsai-8B carries distinct tensors.
    pub fn modelPackedWeightBytes(
        self: ModelSpec,
        format: WeightFormat,
    ) SpecError!u64 {
        const embedding = try packedMatrixBytes(format, self.vocab_size, self.model_dim);
        const global_tables = if (self.tied_embeddings)
            embedding
        else
            try checkedMul(2, embedding);
        return checkedAdd(
            global_tables,
            try checkedMul(self.layers, try self.layerPackedWeightBytes(format)),
        );
    }

    pub fn modelNormBytes(self: ModelSpec) SpecError!u64 {
        const per_layer = try checkedAdd(
            try checkedMul(2, try vectorBytes(self.model_dim)),
            try checkedMul(2, try vectorBytes(self.key_head_dim)),
        );
        return checkedAdd(
            try vectorBytes(self.model_dim),
            try checkedMul(self.layers, per_layer),
        );
    }

    pub fn minimumDdrBytes(
        self: ModelSpec,
        format: WeightFormat,
        capacity_tokens: u32,
    ) SpecError!u64 {
        return checkedAdd(
            try checkedAdd(
                try self.modelPackedWeightBytes(format),
                try checkedAdd(try self.modelNormBytes(), try self.ropeTableBytes()),
            ),
            try self.kvCacheBytes(capacity_tokens),
        );
    }

    pub fn workingSet(self: ModelSpec, tokens: u8) SpecError!WorkingSet {
        if (tokens == 0 or tokens > token_tile_max)
            return error.InvalidMatrixShape;
        const residual = try checkedMul(try checkedMul(tokens, self.model_dim), f32_bytes);
        const query = try checkedMul(try checkedMul(tokens, self.queryWidth()), f32_bytes);
        const flash_accumulator = try checkedMul(
            try checkedMul(tokens, self.attentionValueWidth()),
            f32_bytes,
        );
        const normalized_q8 = try paddedQ8Bytes(tokens, self.model_dim);
        const down_q8 = try paddedQ8Bytes(tokens, self.ffn_dim);
        const new_kv = try checkedMul(tokens, try self.kvRecordBytes());
        return .{
            .residual = residual,
            .query = query,
            .flash_accumulator = flash_accumulator,
            .normalized_q8 = normalized_q8,
            .down_q8 = down_q8,
            .new_kv = new_kv,
        };
    }
};

pub const MatrixShape = struct {
    rows: u32,
    k: u32,
};

pub const WorkingSet = struct {
    residual: u64,
    query: u64,
    flash_accumulator: u64,
    normalized_q8: u64,
    down_q8: u64,
    new_kv: u64,

    pub fn totalWithoutAliasing(self: WorkingSet) SpecError!u64 {
        var total: u64 = 0;
        inline for (std.meta.fields(WorkingSet)) |field|
            total = try checkedAdd(total, @field(self, field.name));
        return total;
    }
};

pub const max_layers: u16 = 36;
pub const vocabulary_size: u32 = 151_669;

pub const bonsai_1_7b = ModelSpec{
    .id = .bonsai_1_7b,
    .name = "Bonsai-1.7B",
    .layers = 28,
    .context_length = 32_768,
    .model_dim = 2048,
    .ffn_dim = 6144,
    .query_heads = 16,
    .kv_heads = 8,
    .key_head_dim = 128,
    .value_head_dim = 128,
    .vocab_size = vocabulary_size,
    .tied_embeddings = true,
    .rope_original_context = 8192,
    .rope_scaling_factor = 4,
    .rope_freq_base = 1_000_000,
    .rms_epsilon = 0.000001,
};

pub const bonsai_4b = ModelSpec{
    .id = .bonsai_4b,
    .name = "Bonsai-4B",
    .layers = 36,
    .context_length = 32_768,
    .model_dim = 2560,
    .ffn_dim = 9728,
    .query_heads = 32,
    .kv_heads = 8,
    .key_head_dim = 128,
    .value_head_dim = 128,
    .vocab_size = vocabulary_size,
    .tied_embeddings = true,
    .rope_original_context = 8192,
    .rope_scaling_factor = 4,
    .rope_freq_base = 5_000_000,
    .rms_epsilon = 0.000001,
};

pub const bonsai_8b = ModelSpec{
    .id = .bonsai_8b,
    .name = "Bonsai-8B",
    .layers = 36,
    .context_length = 65_536,
    .model_dim = 4096,
    .ffn_dim = 12_288,
    .query_heads = 32,
    .kv_heads = 8,
    .key_head_dim = 128,
    .value_head_dim = 128,
    .vocab_size = vocabulary_size,
    .tied_embeddings = false,
    .rope_original_context = 16_384,
    .rope_scaling_factor = 4,
    .rope_freq_base = 1_000_000,
    .rms_epsilon = 0.000001,
};

pub const all_specs = [_]ModelSpec{
    bonsai_1_7b,
    bonsai_4b,
    bonsai_8b,
};

pub fn get(id: SpecId) ModelSpec {
    return switch (id) {
        .bonsai_1_7b => bonsai_1_7b,
        .bonsai_4b => bonsai_4b,
        .bonsai_8b => bonsai_8b,
    };
}

/// Metadata already decoded by the model loader. The token contract matches
/// values; it does not add a second GGUF binary parser.
pub const GgufMetadata = struct {
    architecture: []const u8,
    block_count: u32,
    context_length: u32,
    embedding_length: u32,
    feed_forward_length: u32,
    attention_head_count: u32,
    attention_head_count_kv: u32,
    attention_key_length: u32,
    attention_value_length: u32,
    vocabulary_size: u32,
    rope_original_context: u32,
    rope_scaling_factor: f32,
    rope_freq_base: f32,
    rms_epsilon: f32,
};

pub fn matchGgufMetadata(metadata: GgufMetadata) SpecError!ModelSpec {
    if (!std.mem.eql(u8, metadata.architecture, "qwen3"))
        return error.MetadataMismatch;
    for (all_specs) |spec| {
        if (metadata.block_count == spec.layers and
            metadata.context_length == spec.context_length and
            metadata.embedding_length == spec.model_dim and
            metadata.feed_forward_length == spec.ffn_dim and
            metadata.attention_head_count == spec.query_heads and
            metadata.attention_head_count_kv == spec.kv_heads and
            metadata.attention_key_length == spec.key_head_dim and
            metadata.attention_value_length == spec.value_head_dim and
            metadata.vocabulary_size == spec.vocab_size and
            metadata.rope_original_context == spec.rope_original_context and
            metadata.rope_scaling_factor == spec.rope_scaling_factor and
            metadata.rope_freq_base == spec.rope_freq_base and
            metadata.rms_epsilon == spec.rms_epsilon)
            return spec;
    }
    return error.MetadataMismatch;
}

pub const DdrSpan = struct {
    address: u64,
    bytes: u64,

    pub fn end(self: DdrSpan) SpecError!u64 {
        return checkedAdd(self.address, self.bytes);
    }

    pub fn validate(self: DdrSpan) SpecError!void {
        // Zero is a valid offset when a model or KV arena is addressed relative
        // to its immutable BO base. An empty span remains invalid.
        if (self.bytes == 0) return error.EmptyAddress;
        if (self.address % ddr_alignment != 0) return error.UnalignedAddress;
        _ = try self.end();
    }

    pub fn overlaps(self: DdrSpan, other: DdrSpan) SpecError!bool {
        return self.address < try other.end() and other.address < try self.end();
    }
};

/// Addresses are immutable after model-slot publication. Tensor lengths are
/// derived from the selected spec and format, so stale sizes cannot enter
/// the execution contract.
pub const LayerAddresses = extern struct {
    /// GQA issue order, repeated per KV head: its query heads, K head, V head.
    fused_qkv: u64,
    attention_output: u64,
    /// Fused issue rows `[gate0, up0, gate1, up1, ...]`.
    fused_gate_up: u64,
    ffn_down: u64,
    attention_norm: u64,
    attention_q_norm: u64,
    attention_k_norm: u64,
    ffn_norm: u64,

    pub const count: usize = 8;
    pub const encoded_bytes: usize = count * @sizeOf(u64);

    pub fn spans(
        self: LayerAddresses,
        spec: ModelSpec,
        format: WeightFormat,
    ) SpecError![count]DdrSpan {
        return .{
            .{ .address = self.fused_qkv, .bytes = try spec.packedQkvBundleBytes(format) },
            .{ .address = self.attention_output, .bytes = try packedMatrixBytes(format, spec.model_dim, spec.attentionValueWidth()) },
            .{ .address = self.fused_gate_up, .bytes = try spec.packedGateUpBundleBytes(format) },
            .{ .address = self.ffn_down, .bytes = try packedMatrixBytes(format, spec.model_dim, spec.ffn_dim) },
            .{ .address = self.attention_norm, .bytes = try vectorBytes(spec.model_dim) },
            .{ .address = self.attention_q_norm, .bytes = try vectorBytes(spec.key_head_dim) },
            .{ .address = self.attention_k_norm, .bytes = try vectorBytes(spec.key_head_dim) },
            .{ .address = self.ffn_norm, .bytes = try vectorBytes(spec.model_dim) },
        };
    }
};

comptime {
    if (@sizeOf(LayerAddresses) != LayerAddresses.encoded_bytes)
        @compileError("engine layer address record must remain exactly 64 bytes");
}

pub fn addressTablePayloadBytes(spec: ModelSpec) u64 {
    // Four global u64 addresses followed by fixed 64-byte layer records.
    return 4 * @sizeOf(u64) + @as(u64, spec.layers) * LayerAddresses.encoded_bytes;
}

pub const ModelAddressTable = struct {
    spec_id: SpecId,
    weight_format: WeightFormat,
    embedding: u64,
    lm_head: u64,
    output_norm: u64,
    rope_table: u64,
    layers: []const LayerAddresses,

    pub const max_spans: usize = 4 + @as(usize, max_layers) * LayerAddresses.count;

    pub fn validate(self: ModelAddressTable) SpecError!void {
        const spec = get(self.spec_id);
        try spec.validate();
        if (self.layers.len != spec.layers) return error.InvalidLayerCount;

        var spans_buffer: [max_spans]DdrSpan = undefined;
        var span_count: usize = 0;
        if (spec.tied_embeddings != (self.embedding == self.lm_head))
            return error.InvalidWeightTying;

        const table_bytes = try packedMatrixBytes(
            self.weight_format,
            spec.vocab_size,
            spec.model_dim,
        );
        spans_buffer[span_count] = .{
            .address = self.embedding,
            .bytes = table_bytes,
        };
        span_count += 1;
        if (!spec.tied_embeddings) {
            spans_buffer[span_count] = .{
                .address = self.lm_head,
                .bytes = table_bytes,
            };
            span_count += 1;
        }
        spans_buffer[span_count] = .{
            .address = self.output_norm,
            .bytes = try vectorBytes(spec.model_dim),
        };
        span_count += 1;
        spans_buffer[span_count] = .{
            .address = self.rope_table,
            .bytes = try spec.ropeTableBytes(),
        };
        span_count += 1;
        for (self.layers) |layer| {
            const layer_spans = try layer.spans(spec, self.weight_format);
            for (layer_spans) |span| {
                spans_buffer[span_count] = span;
                span_count += 1;
            }
        }

        for (spans_buffer[0..span_count], 0..) |span, i| {
            try span.validate();
            for (spans_buffer[0..i]) |prior|
                if (try span.overlaps(prior)) return error.AddressOverlap;
        }
    }
};

/// Deterministic physical model image. The host writes this layout and the
/// device derives the same addresses from the BO base; no hot address table is
/// accepted from a client.
pub const ModelImageLayout = struct {
    spec_id: SpecId,
    weight_format: WeightFormat,
    base_address: u64,
    image_bytes: u64,
    embedding: u64,
    lm_head: u64,
    output_norm: u64,
    rope_table: u64,
    layer_count: u16,
    layer_storage: [max_layers]LayerAddresses,

    pub fn spec(self: ModelImageLayout) ModelSpec {
        return get(self.spec_id);
    }

    pub fn layers(self: *const ModelImageLayout) []const LayerAddresses {
        return self.layer_storage[0..self.layer_count];
    }

    pub fn table(self: *const ModelImageLayout) ModelAddressTable {
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
};

pub fn planModelImage(
    spec_id: SpecId,
    format: WeightFormat,
    base_address: u64,
) SpecError!ModelImageLayout {
    const spec = get(spec_id);
    try spec.validate();
    if (base_address % ddr_alignment != 0) return error.UnalignedAddress;

    var cursor = base_address;
    const global_table_bytes = try packedMatrixBytes(
        format,
        spec.vocab_size,
        spec.model_dim,
    );
    const embedding = try alignedTake(&cursor, global_table_bytes);
    const lm_head = if (spec.tied_embeddings)
        embedding
    else
        try alignedTake(&cursor, global_table_bytes);

    var result = ModelImageLayout{
        .spec_id = spec_id,
        .weight_format = format,
        .base_address = base_address,
        .image_bytes = 0,
        .embedding = embedding,
        .lm_head = lm_head,
        .output_norm = try alignedTake(
            &cursor,
            try vectorBytes(spec.model_dim),
        ),
        .rope_table = try alignedTake(&cursor, try spec.ropeTableBytes()),
        .layer_count = spec.layers,
        .layer_storage = undefined,
    };

    for (&result.layer_storage) |*layer| layer.* = .{
        .fused_qkv = 0,
        .attention_output = 0,
        .fused_gate_up = 0,
        .ffn_down = 0,
        .attention_norm = 0,
        .attention_q_norm = 0,
        .attention_k_norm = 0,
        .ffn_norm = 0,
    };
    for (result.layer_storage[0..spec.layers]) |*layer|
        layer.* = try makeLayerAddresses(spec, format, &cursor);

    result.image_bytes = std.math.sub(u64, cursor, base_address) catch
        return error.Overflow;
    try result.table().validate();
    return result;
}

fn metadataFor(spec: ModelSpec) GgufMetadata {
    return .{
        .architecture = "qwen3",
        .block_count = spec.layers,
        .context_length = spec.context_length,
        .embedding_length = spec.model_dim,
        .feed_forward_length = spec.ffn_dim,
        .attention_head_count = spec.query_heads,
        .attention_head_count_kv = spec.kv_heads,
        .attention_key_length = spec.key_head_dim,
        .attention_value_length = spec.value_head_dim,
        .vocabulary_size = spec.vocab_size,
        .rope_original_context = spec.rope_original_context,
        .rope_scaling_factor = spec.rope_scaling_factor,
        .rope_freq_base = spec.rope_freq_base,
        .rms_epsilon = spec.rms_epsilon,
    };
}

fn alignedTake(cursor: *u64, bytes: u64) SpecError!u64 {
    const address = (try checkedAdd(cursor.*, ddr_alignment - 1)) & ~(ddr_alignment - 1);
    cursor.* = try checkedAdd(address, bytes);
    return address;
}

fn makeLayerAddresses(
    spec: ModelSpec,
    format: WeightFormat,
    cursor: *u64,
) SpecError!LayerAddresses {
    var result = LayerAddresses{
        .fused_qkv = ddr_alignment,
        .attention_output = ddr_alignment,
        .fused_gate_up = ddr_alignment,
        .ffn_down = ddr_alignment,
        .attention_norm = ddr_alignment,
        .attention_q_norm = ddr_alignment,
        .attention_k_norm = ddr_alignment,
        .ffn_norm = ddr_alignment,
    };
    const lengths = try result.spans(spec, format);
    inline for (std.meta.fields(LayerAddresses), 0..) |field, i| {
        @field(result, field.name) = try alignedTake(cursor, lengths[i].bytes);
    }
    return result;
}

test "observed GGUF metadata binds all three bounded specs" {
    for (all_specs) |expected| {
        const matched = try matchGgufMetadata(metadataFor(expected));
        try std.testing.expectEqual(expected.id, matched.id);
        try matched.validate();
    }
    var invalid = metadataFor(bonsai_4b);
    invalid.embedding_length = 4096;
    try std.testing.expectError(error.MetadataMismatch, matchGgufMetadata(invalid));
    try std.testing.expectEqual(@as(u32, 2560), bonsai_4b.model_dim);
    try std.testing.expectEqual(@as(u32, 4096), bonsai_4b.queryWidth());

    try std.testing.expectEqual(@as(f32, 1_000_000), bonsai_1_7b.rope_freq_base);
    try std.testing.expectEqual(@as(f32, 5_000_000), bonsai_4b.rope_freq_base);
    try std.testing.expectEqual(@as(f32, 1_000_000), bonsai_8b.rope_freq_base);
    invalid = metadataFor(bonsai_1_7b);
    invalid.rope_freq_base = 5_000_000;
    try std.testing.expectError(error.MetadataMismatch, matchGgufMetadata(invalid));
}

test "Q1 and Q2 issue-packed sizes match deployed 1.7B accounting" {
    try std.testing.expectEqual(
        @as(u64, 48_537_600),
        try packedMatrixBytes(.q1_0, vocabulary_size, 2048),
    );
    try std.testing.expectEqual(
        @as(u64, 87_367_680),
        try packedMatrixBytes(.q2_0, vocabulary_size, 2048),
    );
    try std.testing.expectEqual(
        @as(u64, 268_738_560),
        try bonsai_1_7b.modelPackedWeightBytes(.q1_0),
    );
    try std.testing.expectEqual(
        @as(u64, 483_729_408),
        try bonsai_1_7b.modelPackedWeightBytes(.q2_0),
    );
    try std.testing.expectEqual(
        @as(u64, 4_096),
        try bonsai_8b.kvRecordBytes(),
    );
}

test "fused QKV geometry is GQA-grouped and byte exact" {
    try std.testing.expectEqual(@as(u16, 2), bonsai_1_7b.queryHeadsPerKv());
    try std.testing.expectEqual(@as(u32, 512), bonsai_1_7b.qkvRowsPerKvGroup());
    try std.testing.expectEqual(@as(u32, 4096), bonsai_1_7b.qkvBundleRows());
    try std.testing.expectEqual(
        @as(u64, 1_310_720),
        try bonsai_1_7b.packedQkvBundleBytes(.q1_0),
    );

    try std.testing.expectEqual(@as(u16, 4), bonsai_4b.queryHeadsPerKv());
    try std.testing.expectEqual(@as(u32, 768), bonsai_4b.qkvRowsPerKvGroup());
    try std.testing.expectEqual(@as(u32, 6144), bonsai_4b.qkvBundleRows());
    try std.testing.expectEqual(
        @as(u64, 2_457_600),
        try bonsai_4b.packedQkvBundleBytes(.q1_0),
    );
    try std.testing.expectEqual(
        @as(u64, 4_423_680),
        try bonsai_4b.packedQkvBundleBytes(.q2_0),
    );
}

test "physical layer address record is eight ordered u64 words" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(LayerAddresses));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(LayerAddresses, "fused_qkv"));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(LayerAddresses, "attention_output"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(LayerAddresses, "fused_gate_up"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(LayerAddresses, "ffn_down"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(LayerAddresses, "attention_norm"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(LayerAddresses, "attention_q_norm"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(LayerAddresses, "attention_k_norm"));
    try std.testing.expectEqual(@as(usize, 56), @offsetOf(LayerAddresses, "ffn_norm"));
    try std.testing.expectEqual(@as(u64, 32 + 28 * 64), addressTablePayloadBytes(bonsai_1_7b));
}

test "T8 max-spec working set fits raw K26 block memory" {
    const work = try bonsai_8b.workingSet(token_tile_max);
    try std.testing.expectEqual(@as(u64, 128 * 1024), work.residual);
    try std.testing.expectEqual(@as(u64, 120 * 1024), work.down_q8);
    try std.testing.expectEqual(@as(u64, 32 * 1024), work.new_kv);
    try std.testing.expect((try work.totalWithoutAliasing()) < k26_raw_onchip_bytes);
}

test "full-context f16 KV exposes K26 DDR capacity limits" {
    const four = try bonsai_4b.minimumDdrBytes(.q1_0, bonsai_4b.context_length);
    const eight = try bonsai_8b.minimumDdrBytes(.q1_0, bonsai_8b.context_length);
    try std.testing.expect(four > k26_ddr_bytes);
    try std.testing.expect(eight > k26_ddr_bytes);
}

test "immutable address table derives lengths and rejects aliases" {
    const spec = bonsai_1_7b;
    var cursor: u64 = 0x1000;
    const embedding = try alignedTake(
        &cursor,
        try packedMatrixBytes(.q1_0, spec.vocab_size, spec.model_dim),
    );
    const output_norm = try alignedTake(&cursor, try vectorBytes(spec.model_dim));
    const rope_table = try alignedTake(&cursor, try spec.ropeTableBytes());
    var layers: [28]LayerAddresses = undefined;
    for (&layers) |*layer| layer.* = try makeLayerAddresses(spec, .q1_0, &cursor);
    const table = ModelAddressTable{
        .spec_id = .bonsai_1_7b,
        .weight_format = .q1_0,
        .embedding = embedding,
        .lm_head = embedding,
        .output_norm = output_norm,
        .rope_table = rope_table,
        .layers = &layers,
    };
    try table.validate();

    var invalid_tie = table;
    invalid_tie.lm_head += ddr_alignment;
    try std.testing.expectError(error.InvalidWeightTying, invalid_tie.validate());

    layers[7].ffn_down = layers[7].fused_gate_up;
    try std.testing.expectError(error.AddressOverlap, table.validate());
}

test "shared model planner derives every bounded image exactly" {
    const expected = [_][2]u64{
        .{ 286_011_392, 501_002_240 },
        .{ 645_939_200, 1_148_641_280 },
        .{ 1_314_213_888, 2_337_755_136 },
    };
    for (all_specs, 0..) |spec, spec_index| {
        for ([_]WeightFormat{ .q1_0, .q2_0 }, 0..) |format, format_index| {
            const plan = try planModelImage(spec.id, format, 0x4000);
            try std.testing.expectEqual(expected[spec_index][format_index], plan.image_bytes);
            try std.testing.expectEqual(spec.layers, plan.layer_count);
            try std.testing.expectEqual(spec.tied_embeddings, plan.embedding == plan.lm_head);
            try plan.table().validate();
        }
    }
}
