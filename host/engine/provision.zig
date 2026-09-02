//! One-shot inference model provisioning. Source spans are borrowed, each packed
//! tensor is freed immediately after upload, and only the immutable plan remains.

const std = @import("std");
const shared = @import("shared");
const model = @import("model.zig");

const packing = @import("packing.zig");
const contract = shared.engine.model_spec;

pub const Error = model.Error || packing.LayoutError || std.mem.Allocator.Error || error{
    UploadFailed,
};

pub const Writer = struct {
    context: *anyopaque,
    write_fn: *const fn (context: *anyopaque, address: u64, bytes: []const u8) error{UploadFailed}!void,

    pub fn write(self: Writer, address: u64, bytes: []const u8) error{UploadFailed}!void {
        return self.write_fn(self.context, address, bytes);
    }
};

fn packedMatrixBytes(format: contract.WeightFormat, rows: u32, k: u32) Error!usize {
    return @intCast(try contract.packedMatrixBytes(format, rows, k));
}

pub fn packMatrixAlloc(
    allocator: std.mem.Allocator,
    source: model.LowBitTensor,
    rows: u32,
    k: u32,
) Error![]u8 {
    try source.validateMatrix(source.format, rows, k);
    const out = try allocator.alloc(u8, try packedMatrixBytes(source.format, rows, k));
    errdefer allocator.free(out);
    switch (source.format) {
        .q1_0 => try packing.packWeightsFromGgmlQ1_0(rows, k, source.bytes, out),
        .q2_0 => try packing.packWeightsFromGgmlQ2_0(rows, k, source.bytes, out),
    }
    return out;
}

pub fn packGqaQkvQ1Alloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error![]u8 {
    try query.validateMatrix(.q1_0, spec.queryWidth(), spec.model_dim);
    try key.validateMatrix(.q1_0, spec.kvKeyWidth(), spec.model_dim);
    try value.validateMatrix(.q1_0, spec.kvValueWidth(), spec.model_dim);
    const out = try allocator.alloc(u8, @intCast(try spec.packedQkvBundleBytes(.q1_0)));
    errdefer allocator.free(out);
    try packing.packFusedQkvWeightsFromGgmlQ1_0(
        spec.query_heads,
        spec.kv_heads,
        spec.key_head_dim,
        spec.model_dim,
        query.bytes,
        key.bytes,
        value.bytes,
        out,
    );
    return out;
}

pub fn packGqaQkvQ2Alloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error![]u8 {
    try query.validateMatrix(.q2_0, spec.queryWidth(), spec.model_dim);
    try key.validateMatrix(.q2_0, spec.kvKeyWidth(), spec.model_dim);
    try value.validateMatrix(.q2_0, spec.kvValueWidth(), spec.model_dim);
    const out = try allocator.alloc(u8, @intCast(try spec.packedQkvBundleBytes(.q2_0)));
    errdefer allocator.free(out);
    try packing.packFusedQkvWeightsFromGgmlQ2_0(
        spec.query_heads,
        spec.kv_heads,
        spec.key_head_dim,
        spec.model_dim,
        query.bytes,
        key.bytes,
        value.bytes,
        out,
    );
    return out;
}

pub fn packGqaQkvAlloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error![]u8 {
    return switch (format) {
        .q1_0 => packGqaQkvQ1Alloc(allocator, spec, query, key, value),
        .q2_0 => packGqaQkvQ2Alloc(allocator, spec, query, key, value),
    };
}

pub fn packGateUpQ1Alloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error![]u8 {
    try gate.validateMatrix(.q1_0, spec.ffn_dim, spec.model_dim);
    try up.validateMatrix(.q1_0, spec.ffn_dim, spec.model_dim);
    const out = try allocator.alloc(u8, @intCast(try spec.packedGateUpBundleBytes(.q1_0)));
    errdefer allocator.free(out);
    try packing.packFusedGateUpWeightsFromGgmlQ1_0(
        spec.ffn_dim,
        spec.model_dim,
        gate.bytes,
        up.bytes,
        out,
    );
    return out;
}

pub fn packGateUpQ2Alloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error![]u8 {
    try gate.validateMatrix(.q2_0, spec.ffn_dim, spec.model_dim);
    try up.validateMatrix(.q2_0, spec.ffn_dim, spec.model_dim);
    const out = try allocator.alloc(u8, @intCast(try spec.packedGateUpBundleBytes(.q2_0)));
    errdefer allocator.free(out);
    try packing.packFusedGateUpWeightsFromGgmlQ2_0(
        spec.ffn_dim,
        spec.model_dim,
        gate.bytes,
        up.bytes,
        out,
    );
    return out;
}

pub fn packGateUpAlloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error![]u8 {
    return switch (format) {
        .q1_0 => packGateUpQ1Alloc(allocator, spec, gate, up),
        .q2_0 => packGateUpQ2Alloc(allocator, spec, gate, up),
    };
}

fn uploadOwned(allocator: std.mem.Allocator, writer: Writer, address: u64, bytes: []u8) Error!void {
    defer allocator.free(bytes);
    try writer.write(address, bytes);
}

const RopeFormula = struct {
    inv_n_dims: f32,
    freq_scale: f32,
    freq_base: f32,
    corr_low: f32,
    corr_high: f32,
    mscale: f32,

    fn init(spec: contract.ModelSpec) RopeFormula {
        const n_dims: f32 = @floatFromInt(spec.key_head_dim);
        const context: f32 = @floatFromInt(spec.rope_original_context);
        const freq_scale = 1.0 / spec.rope_scaling_factor;
        return .{
            .inv_n_dims = 1.0 / n_dims,
            .freq_scale = freq_scale,
            .freq_base = spec.rope_freq_base,
            .corr_low = @max(0.0, @floor(yarnCorrDim(
                n_dims,
                context,
                32.0,
                spec.rope_freq_base,
            ))),
            .corr_high = @min(n_dims - 1.0, @ceil(yarnCorrDim(
                n_dims,
                context,
                1.0,
                spec.rope_freq_base,
            ))),
            .mscale = 1.0 + 0.1 * @log(1.0 / freq_scale),
        };
    }

    fn coefficient(self: RopeFormula, position: usize, pair: usize) struct { cos: f32, sin: f32 } {
        // llama.cpp advances i0 by two even for NEOX. NEOX changes the data
        // coordinates to pair,pair+64; it does not change the frequency index.
        const frequency_dimension = 2 * pair;
        const theta_extrap = @as(f32, @floatFromInt(position)) * std.math.pow(
            f32,
            self.freq_base,
            -@as(f32, @floatFromInt(frequency_dimension)) * self.inv_n_dims,
        );
        const theta_interp = self.freq_scale * theta_extrap;
        const ramp = yarnRamp(
            self.corr_low,
            self.corr_high,
            frequency_dimension,
        );
        const theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
        return .{
            .cos = @cos(theta) * self.mscale,
            .sin = @sin(theta) * self.mscale,
        };
    }
};

/// Build the immutable head-independent YaRN table consumed by the token
/// engine. Each position stores interleaved FP32 `{ cos, sin }` pairs for all
/// rotary dimensions, in little-endian byte order.
pub fn generateRopeTableAlloc(
    allocator: std.mem.Allocator,
    spec: contract.ModelSpec,
) Error![]u8 {
    try spec.validate();
    const out = try allocator.alloc(u8, @intCast(try spec.ropeTableBytes()));
    errdefer allocator.free(out);

    const formula = RopeFormula.init(spec);
    const pairs: usize = spec.key_head_dim / 2;

    for (0..spec.context_length) |position| {
        for (0..pairs) |pair| {
            const dimension = 2 * pair;
            const coefficient = formula.coefficient(position, pair);
            const element = position * @as(usize, spec.key_head_dim) + dimension;
            writeF32(out, element, coefficient.cos);
            writeF32(out, element + 1, coefficient.sin);
        }
    }
    return out;
}

fn yarnCorrDim(n_dims: f32, context: f32, rotations: f32, base: f32) f32 {
    return n_dims * @log(context / (rotations * 2.0 * std.math.pi)) /
        (2.0 * @log(base));
}

fn yarnRamp(low: f32, high: f32, dimension: usize) f32 {
    const denominator = @max(0.001, high - low);
    const value = (@as(f32, @floatFromInt(dimension / 2)) - low) / denominator;
    return 1.0 - std.math.clamp(value, 0.0, 1.0);
}

fn writeF32(out: []u8, index: usize, value: f32) void {
    std.mem.writeInt(
        u32,
        out[index * @sizeOf(f32) ..][0..@sizeOf(f32)],
        @bitCast(value),
        .little,
    );
}

fn readF32(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(
        u32,
        bytes[index * @sizeOf(f32) ..][0..@sizeOf(f32)],
        .little,
    ));
}

pub fn provisionGqaQkvQ1(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error!void {
    try uploadOwned(
        allocator,
        writer,
        address,
        try packGqaQkvQ1Alloc(allocator, spec, query, key, value),
    );
}

pub fn provisionGqaQkvQ2(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error!void {
    try uploadOwned(
        allocator,
        writer,
        address,
        try packGqaQkvQ2Alloc(allocator, spec, query, key, value),
    );
}

pub fn provisionGqaQkv(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    query: model.LowBitTensor,
    key: model.LowBitTensor,
    value: model.LowBitTensor,
) Error!void {
    return switch (format) {
        .q1_0 => provisionGqaQkvQ1(allocator, writer, address, spec, query, key, value),
        .q2_0 => provisionGqaQkvQ2(allocator, writer, address, spec, query, key, value),
    };
}

pub fn provisionGateUpQ1(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error!void {
    try uploadOwned(
        allocator,
        writer,
        address,
        try packGateUpQ1Alloc(allocator, spec, gate, up),
    );
}

pub fn provisionGateUpQ2(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error!void {
    try uploadOwned(
        allocator,
        writer,
        address,
        try packGateUpQ2Alloc(allocator, spec, gate, up),
    );
}

pub fn provisionGateUp(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    gate: model.LowBitTensor,
    up: model.LowBitTensor,
) Error!void {
    return switch (format) {
        .q1_0 => provisionGateUpQ1(allocator, writer, address, spec, gate, up),
        .q2_0 => provisionGateUpQ2(allocator, writer, address, spec, gate, up),
    };
}

fn provisionMatrix(
    allocator: std.mem.Allocator,
    writer: Writer,
    address: u64,
    source: model.LowBitTensor,
    rows: u32,
    k: u32,
) Error!void {
    try uploadOwned(allocator, writer, address, try packMatrixAlloc(allocator, source, rows, k));
}

/// Validate and upload a complete model image. The caller publishes the returned
/// address table only after this function succeeds; a partial image is inert.
pub fn provisionModel(
    allocator: std.mem.Allocator,
    writer: Writer,
    base_address: u64,
    sources: model.ModelSources,
) Error!model.ImagePlan {
    const spec = try sources.validate();
    const plan = try model.planImage(spec, sources.weight_format, base_address);

    try provisionGlobalTables(
        allocator,
        writer,
        plan.embedding,
        plan.lm_head,
        spec,
        sources.weight_format,
        sources.embedding,
        sources.lm_head,
    );
    try writer.write(plan.output_norm, sources.output_norm.bytes);
    try uploadOwned(
        allocator,
        writer,
        plan.rope_table,
        try generateRopeTableAlloc(allocator, spec),
    );

    for (sources.layers, 0..) |layer, index| {
        const addresses = plan.layer_storage[index];
        try provisionGqaQkv(
            allocator,
            writer,
            addresses.fused_qkv,
            spec,
            sources.weight_format,
            layer.query,
            layer.key,
            layer.value,
        );
        try provisionMatrix(
            allocator,
            writer,
            addresses.attention_output,
            layer.attention_output,
            spec.model_dim,
            spec.attentionValueWidth(),
        );
        try provisionGateUp(
            allocator,
            writer,
            addresses.fused_gate_up,
            spec,
            sources.weight_format,
            layer.gate,
            layer.up,
        );
        try provisionMatrix(
            allocator,
            writer,
            addresses.ffn_down,
            layer.ffn_down,
            spec.model_dim,
            spec.ffn_dim,
        );
        try writer.write(addresses.attention_norm, layer.attention_norm.bytes);
        try writer.write(addresses.attention_q_norm, layer.attention_q_norm.bytes);
        try writer.write(addresses.attention_k_norm, layer.attention_k_norm.bytes);
        try writer.write(addresses.ffn_norm, layer.ffn_norm.bytes);
    }
    return plan;
}

pub fn provisionGlobalTables(
    allocator: std.mem.Allocator,
    writer: Writer,
    embedding_address: u64,
    lm_head_address: u64,
    spec: contract.ModelSpec,
    format: contract.WeightFormat,
    embedding: model.LowBitTensor,
    lm_head: ?model.LowBitTensor,
) Error!void {
    try model.validateLmHead(spec, format, embedding, lm_head);
    if (spec.tied_embeddings != (embedding_address == lm_head_address))
        return error.InvalidWeightTying;
    try provisionMatrix(
        allocator,
        writer,
        embedding_address,
        embedding,
        spec.vocab_size,
        spec.model_dim,
    );
    if (!spec.tied_embeddings) {
        try provisionMatrix(
            allocator,
            writer,
            lm_head_address,
            lm_head orelse return error.InvalidWeightTying,
            spec.vocab_size,
            spec.model_dim,
        );
    }
}

fn sourceBytes(format: contract.WeightFormat, rows: u32, k: u32) usize {
    return @intCast(contract.sourceMatrixBytes(format, rows, k) catch unreachable);
}

fn lowBit(identity: usize, format: contract.WeightFormat, rows: u32, k: u32, bytes: []const u8) model.LowBitTensor {
    return .{
        .identity = identity,
        .format = format,
        .shape = model.TensorShape.matrix(k, rows),
        .bytes = bytes,
    };
}

const CaptureWriter = struct {
    writes: usize = 0,
    last_address: u64 = 0,
    last: std.ArrayList(u8) = .empty,

    fn write(context: *anyopaque, address: u64, bytes: []const u8) error{UploadFailed}!void {
        const self: *CaptureWriter = @ptrCast(@alignCast(context));
        self.last.clearRetainingCapacity();
        self.last.appendSlice(std.testing.allocator, bytes) catch return error.UploadFailed;
        self.last_address = address;
        self.writes += 1;
    }

    fn interface(self: *CaptureWriter) Writer {
        return .{ .context = self, .write_fn = write };
    }
};

test "Q1 and Q2 GQA packs preserve grouped source roles" {
    const spec = contract.ModelSpec{
        .id = .bonsai_1_7b,
        .name = "synthetic",
        .layers = 1,
        .context_length = 16,
        .model_dim = 128,
        .ffn_dim = 128,
        .query_heads = 2,
        .kv_heads = 1,
        .key_head_dim = 128,
        .value_head_dim = 128,
        .vocab_size = 128,
        .tied_embeddings = true,
        .rope_original_context = 8,
        .rope_scaling_factor = 2,
        .rope_freq_base = 10_000,
        .rms_epsilon = 0.000001,
    };

    for ([_]contract.WeightFormat{ .q1_0, .q2_0 }) |format| {
        const q_len = sourceBytes(format, spec.queryWidth(), spec.model_dim);
        const kv_len = sourceBytes(format, spec.kvKeyWidth(), spec.model_dim);
        const q = try std.testing.allocator.alloc(u8, q_len);
        defer std.testing.allocator.free(q);
        const k = try std.testing.allocator.alloc(u8, kv_len);
        defer std.testing.allocator.free(k);
        const v = try std.testing.allocator.alloc(u8, kv_len);
        defer std.testing.allocator.free(v);
        @memset(q, 0);
        @memset(k, 0);
        @memset(v, 0);
        const q_scale: f16 = 1;
        const k_scale: f16 = 2;
        const v_scale: f16 = 3;
        std.mem.writeInt(u16, q[0..2], @bitCast(q_scale), .little);
        std.mem.writeInt(u16, k[0..2], @bitCast(k_scale), .little);
        std.mem.writeInt(u16, v[0..2], @bitCast(v_scale), .little);

        const resident = try packGqaQkvAlloc(
            std.testing.allocator,
            spec,
            format,
            lowBit(1, format, spec.queryWidth(), spec.model_dim, q),
            lowBit(2, format, spec.kvKeyWidth(), spec.model_dim, k),
            lowBit(3, format, spec.kvValueWidth(), spec.model_dim, v),
        );
        defer std.testing.allocator.free(resident);
        try std.testing.expectEqual(@as(usize, @intCast(try spec.packedQkvBundleBytes(format))), resident.len);
        switch (format) {
            .q1_0 => {
                try std.testing.expectEqual(q_scale, try packing.packedWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 0, 0));
                try std.testing.expectEqual(k_scale, try packing.packedWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 256, 0));
                try std.testing.expectEqual(v_scale, try packing.packedWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 384, 0));
            },
            .q2_0 => {
                try std.testing.expectEqual(q_scale, try packing.packedTernaryWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 0, 0, 0));
                try std.testing.expectEqual(k_scale, try packing.packedTernaryWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 256, 0, 0));
                try std.testing.expectEqual(v_scale, try packing.packedTernaryWeightScale(resident, spec.qkvBundleRows(), spec.model_dim, 384, 0, 0));
            },
        }
    }
}

test "fused gate up provisioning uploads once and releases scratch" {
    const spec = contract.ModelSpec{
        .id = .bonsai_1_7b,
        .name = "synthetic",
        .layers = 1,
        .context_length = 16,
        .model_dim = 128,
        .ffn_dim = 128,
        .query_heads = 1,
        .kv_heads = 1,
        .key_head_dim = 128,
        .value_head_dim = 128,
        .vocab_size = 128,
        .tied_embeddings = true,
        .rope_original_context = 8,
        .rope_scaling_factor = 2,
        .rope_freq_base = 10_000,
        .rms_epsilon = 0.000001,
    };
    const len = sourceBytes(.q1_0, spec.ffn_dim, spec.model_dim);
    const gate = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(gate);
    const up = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(up);
    @memset(gate, 0);
    @memset(up, 0);
    const gate_scale: f16 = 4;
    const up_scale: f16 = 5;
    std.mem.writeInt(u16, gate[0..2], @bitCast(gate_scale), .little);
    std.mem.writeInt(u16, up[0..2], @bitCast(up_scale), .little);

    var capture: CaptureWriter = .{};
    defer capture.last.deinit(std.testing.allocator);
    try provisionGateUp(
        std.testing.allocator,
        capture.interface(),
        0x8000,
        spec,
        .q1_0,
        lowBit(10, .q1_0, spec.ffn_dim, spec.model_dim, gate),
        lowBit(11, .q1_0, spec.ffn_dim, spec.model_dim, up),
    );
    try std.testing.expectEqual(@as(usize, 1), capture.writes);
    try std.testing.expectEqual(@as(u64, 0x8000), capture.last_address);
    try std.testing.expectEqual(gate_scale, try packing.packedWeightScale(capture.last.items, spec.gateUpBundleRows(), spec.model_dim, 0, 0));
    try std.testing.expectEqual(up_scale, try packing.packedWeightScale(capture.last.items, spec.gateUpBundleRows(), spec.model_dim, 1, 0));
}

test "global table provisioning uploads once when tied and twice for 8B policy" {
    var embedding_bytes: [72]u8 = @splat(0);
    var head_bytes: [72]u8 = @splat(0);
    const embedding = lowBit(21, .q1_0, 4, 128, &embedding_bytes);
    const head = lowBit(22, .q1_0, 4, 128, &head_bytes);
    var spec = contract.ModelSpec{
        .id = .bonsai_1_7b,
        .name = "synthetic",
        .layers = 1,
        .context_length = 16,
        .model_dim = 128,
        .ffn_dim = 128,
        .query_heads = 1,
        .kv_heads = 1,
        .key_head_dim = 128,
        .value_head_dim = 128,
        .vocab_size = 4,
        .tied_embeddings = true,
        .rope_original_context = 8,
        .rope_scaling_factor = 2,
        .rope_freq_base = 10_000,
        .rms_epsilon = 0.000001,
    };

    var tied: CaptureWriter = .{};
    defer tied.last.deinit(std.testing.allocator);
    try provisionGlobalTables(
        std.testing.allocator,
        tied.interface(),
        0x1000,
        0x1000,
        spec,
        .q1_0,
        embedding,
        null,
    );
    try std.testing.expectEqual(@as(usize, 1), tied.writes);
    try std.testing.expectEqual(@as(u64, 0x1000), tied.last_address);

    spec.tied_embeddings = false;
    var untied: CaptureWriter = .{};
    defer untied.last.deinit(std.testing.allocator);
    try provisionGlobalTables(
        std.testing.allocator,
        untied.interface(),
        0x2000,
        0x3000,
        spec,
        .q1_0,
        embedding,
        head,
    );
    try std.testing.expectEqual(@as(usize, 2), untied.writes);
    try std.testing.expectEqual(@as(u64, 0x3000), untied.last_address);
    try std.testing.expectError(error.InvalidWeightTying, provisionGlobalTables(
        std.testing.allocator,
        untied.interface(),
        0x2000,
        0x2000,
        spec,
        .q1_0,
        embedding,
        head,
    ));
}

test "RoPE table is head-independent interleaved YaRN cos sin" {
    const spec = contract.ModelSpec{
        .id = .bonsai_1_7b,
        .name = "synthetic",
        .layers = 1,
        .context_length = 16,
        .model_dim = 128,
        .ffn_dim = 128,
        .query_heads = 1,
        .kv_heads = 1,
        .key_head_dim = 128,
        .value_head_dim = 128,
        .vocab_size = 4,
        .tied_embeddings = true,
        .rope_original_context = 8,
        .rope_scaling_factor = 2,
        .rope_freq_base = 10_000,
        .rms_epsilon = 0.000001,
    };
    const table = try generateRopeTableAlloc(std.testing.allocator, spec);
    defer std.testing.allocator.free(table);
    try std.testing.expectEqual(@as(usize, @intCast(try spec.ropeTableBytes())), table.len);

    const mscale: f32 = 1.0 + 0.1 * @log(spec.rope_scaling_factor);
    for (0..spec.key_head_dim / 2) |pair| {
        try std.testing.expectApproxEqAbs(mscale, readF32(table, 2 * pair), 1e-6);
        try std.testing.expectEqual(@as(f32, 0), readF32(table, 2 * pair + 1));
    }
    // Pair zero is in YaRN's extrapolation region, so position one uses theta 1.
    const position_one: usize = spec.key_head_dim;
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)) * mscale, readF32(table, position_one), 1e-6);
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)) * mscale, readF32(table, position_one + 1), 1e-6);
}

test "sealed Qwen3 specs match llama YaRN NEOX pair frequencies" {
    const expected_original = [_]u32{ 8192, 8192, 16_384 };
    const expected_base = [_]f32{ 1_000_000, 5_000_000, 1_000_000 };
    const sample_pairs = [_]usize{ 0, 1, 17, 63 };

    for (contract.all_specs, 0..) |spec, spec_index| {
        try std.testing.expectEqual(@as(u16, 128), spec.key_head_dim);
        try std.testing.expectEqual(@as(f32, 4), spec.rope_scaling_factor);
        try std.testing.expectEqual(
            expected_original[spec_index],
            spec.rope_original_context,
        );
        try std.testing.expectEqual(
            expected_base[spec_index],
            spec.rope_freq_base,
        );

        const formula = RopeFormula.init(spec);
        const position: usize = @intCast(spec.rope_original_context + 37);
        const n_dims: f32 = @floatFromInt(spec.key_head_dim);
        const context: f32 = @floatFromInt(spec.rope_original_context);
        const corr_low = @max(0.0, @floor(
            n_dims * @log(context / (32.0 * 2.0 * std.math.pi)) /
                (2.0 * @log(spec.rope_freq_base)),
        ));
        const corr_high = @min(n_dims - 1.0, @ceil(
            n_dims * @log(context / (1.0 * 2.0 * std.math.pi)) /
                (2.0 * @log(spec.rope_freq_base)),
        ));
        const freq_scale: f32 = 0.25;
        const mscale = 1.0 + 0.1 * @log(1.0 / freq_scale);

        for (sample_pairs) |pair| {
            // llama.cpp's NEOX data pair is [pair,pair+64], while its YaRN
            // frequency/ramp index remains i0=2*pair.
            const dim0 = pair;
            const dim1 = pair + @as(usize, spec.key_head_dim / 2);
            const frequency_dimension = 2 * pair;
            try std.testing.expectEqual(pair + 64, dim1);
            try std.testing.expect(dim0 < 64 and dim1 < 128);

            const exponent = -@as(f32, @floatFromInt(frequency_dimension)) /
                n_dims;
            const theta_extrap = @as(f32, @floatFromInt(position)) *
                std.math.pow(f32, spec.rope_freq_base, exponent);
            const theta_interp = freq_scale * theta_extrap;
            const ramp_y = (@as(f32, @floatFromInt(pair)) - corr_low) /
                @max(0.001, corr_high - corr_low);
            const ramp = 1.0 - std.math.clamp(ramp_y, 0.0, 1.0);
            const theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
            const expected_cos = @cos(theta) * mscale;
            const expected_sin = @sin(theta) * mscale;
            const actual = formula.coefficient(position, pair);
            try std.testing.expectApproxEqAbs(expected_cos, actual.cos, 0.00002);
            try std.testing.expectApproxEqAbs(expected_sin, actual.sin, 0.00002);
        }
    }
}
