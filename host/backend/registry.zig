//! Penzai ggml registry and `penzai.inference.v1` provider.
//!
//! llama.cpp selects the device, discovers the versioned whole-token contract,
//! and delegates model/session execution through it.
const std = @import("std");
const c = @import("c_ggml");
const shared = @import("shared");
const link_mod = @import("link");

pub const engine = @import("inference_engine");

const capabilities = shared.capabilities;
const wire = shared.wire;
const engine_metrics = shared.engine.metrics;
const model_spec = shared.engine.model_spec;

pub const inference_engine_id: u32 = 0xB05A_4000;
pub const inference_interface_version: u32 = 0x0001_0007;
const resident_model_slot: u16 = 0;
const resident_session_slot: u16 = 0;
const inference_features: u64 = (1 << 0) | (1 << 1) | (1 << 2);

const Status = struct {
    const ok: c_int = 0;
    const invalid_argument: c_int = -1;
    const unsupported: c_int = -2;
    const io: c_int = -3;
    const device: c_int = -4;
    const state: c_int = -5;
};

pub const BackendError = error{ HandshakeFailed, MetricsUnsupported, OutOfMemory };

pub const MetricsAggregate = struct {
    level: engine_metrics.Level = .none,
    executions: u64 = 0,
    device_cycles: u64 = 0,
    overflow: [4]u32 = @splat(0),
    values: [engine_metrics.metric_count]u64 = @splat(0),

    fn add(self: *MetricsAggregate, result: shared.engine.rpc.ExecuteResult) void {
        if (self.level == .none) return;
        self.executions +|= 1;
        self.device_cycles +|= result.cycles;
        const snapshot = result.metrics_snapshot orelse return;
        for (&self.overflow, snapshot.overflow) |*out, value| out.* |= value;
        for (&self.values, snapshot.values, 0..) |*out, value, index| {
            if (index == @intFromEnum(engine_metrics.MetricId.projection_selector_high_water)) {
                out.* = @max(out.*, value);
            } else {
                out.* +|= value;
            }
        }
    }
};

const ResidencyPhase = enum { idle, prepared, ambiguous, installed };

const Residency = struct {
    phase: ResidencyPhase = .idle,
    spec_id: model_spec.SpecId = .bonsai_1_7b,
    weight_format: model_spec.WeightFormat = .q1_0,
    model_hash: u64 = 0,
    image_bytes: u64 = 0,
    image: ?wire.TensorRange = null,

    fn reset(self: *Residency) void {
        self.* = .{};
    }
};

pub const Device = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    link: link_mod.Client,
    device: c.ggml_backend_device = undefined,
    reg: c.ggml_backend_reg = undefined,
    inference_capable: bool,
    inference_format_mask: u32,
    feature_mask: u32,
    engine_clock_hz: u32,
    metrics_level: engine_metrics.Level = .none,
    metrics_aggregate: MetricsAggregate = .{},
    mock_executor: bool = false,
    residency: Residency = .{},
    name: [*:0]const u8 = "penzai",
    description: [*:0]const u8 = "Penzai whole-token accelerator",

    pub fn create(allocator: std.mem.Allocator, link: link_mod.Client) BackendError!*Self {
        link.hello() catch return error.HandshakeFailed;
        const report = link.capabilities() catch return error.HandshakeFailed;
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .link = link,
            .inference_capable = hasInference(report),
            .inference_format_mask = report.format_mask,
            .feature_mask = report.feature_mask,
            .engine_clock_hz = report.engine.clock_hz,
        };
        self.reg = .{
            .api_version = c.GGML_BACKEND_API_VERSION,
            .iface = registry_interface,
            .context = self,
        };
        self.device = .{
            .iface = device_interface,
            .reg = &self.reg,
            .context = self,
        };
        return self;
    }

    pub fn destroy(self: *Self) void {
        self.releaseModel();
        self.allocator.destroy(self);
    }

    pub fn ggmlDevice(self: *Self) c.ggml_backend_dev_t {
        return &self.device;
    }

    pub fn ggmlReg(self: *Self) c.ggml_backend_reg_t {
        return &self.reg;
    }

    pub fn enableMockExecutor(self: *Self) void {
        self.mock_executor = true;
    }

    pub fn setMetricsLevel(
        self: *Self,
        level: engine_metrics.Level,
    ) BackendError!void {
        const required: u32 = switch (level) {
            .none => 0,
            .summary => capabilities.Feature.metrics_summary,
            .full => return error.MetricsUnsupported,
        };
        if (self.feature_mask & required != required)
            return error.MetricsUnsupported;
        self.metrics_level = level;
        self.metrics_aggregate = .{ .level = level };
    }

    pub fn takeMetrics(self: *Self) MetricsAggregate {
        const result = self.metrics_aggregate;
        self.metrics_aggregate = .{ .level = self.metrics_level };
        return result;
    }

    pub fn metricsTimeNs(self: *const Self, cycles: u64) u64 {
        if (self.engine_clock_hz == 0) return 0;
        return @intCast((@as(u128, cycles) * std.time.ns_per_s) /
            self.engine_clock_hz);
    }

    fn releaseModel(self: *Self) void {
        const state = &self.residency;
        if (state.phase == .installed or state.phase == .ambiguous) {
            state.phase = .ambiguous;
            _ = self.link.inference(.{ .uninstall_model = .{
                .slot = resident_model_slot,
            } }) catch return;
        }
        if (state.image) |image| freeQuietly(self, image);
        state.reset();
    }
};

fn hasInference(report: capabilities.Report) bool {
    return report.wire_abi == wire.version and
        report.metrics_schema == engine_metrics.schema_version and
        report.feature_mask & capabilities.Feature.inference != 0 and
        report.engine.interface_id == inference_engine_id and
        report.engine.interface_version == inference_interface_version and
        report.engine.token_tile_max == model_spec.token_tile_max and
        report.engine.token_lanes == model_spec.physical_token_lanes and
        report.engine.model_spec_count == model_spec.all_specs.len and
        report.engine.context_tokens_max == model_spec.bonsai_8b.context_length and
        report.engine.address_record_bytes == model_spec.LayerAddresses.encoded_bytes;
}

fn formatSupported(mask: u32, format: model_spec.WeightFormat) bool {
    const required: u32 = switch (format) {
        .q1_0 => capabilities.Format.weight_q1_0,
        .q2_0 => capabilities.Format.weight_q2_0_g64,
    };
    return mask & required != 0;
}

fn mixHash(hash: *u64, value: u64) void {
    var remaining = value;
    for (0..8) |_| {
        hash.* = (hash.* ^ @as(u8, @truncate(remaining))) *% 0x0000_0100_0000_01b3;
        remaining >>= 8;
    }
}

fn modelHash(spec: model_spec.ModelSpec, format: model_spec.WeightFormat) u64 {
    var hash: u64 = 0xcbf2_9ce4_8422_2325;
    mixHash(&hash, inference_interface_version);
    mixHash(&hash, 0xc255_c7a5_2fc1_4a79);
    mixHash(&hash, @intFromEnum(spec.id));
    mixHash(&hash, @intFromEnum(format));
    inline for (.{
        spec.layers,
        spec.context_length,
        spec.model_dim,
        spec.ffn_dim,
        spec.query_heads,
        spec.kv_heads,
        spec.key_head_dim,
        spec.value_head_dim,
        spec.vocab_size,
        spec.rope_original_context,
    }) |value| mixHash(&hash, value);
    mixHash(&hash, @intFromBool(spec.tied_embeddings));
    mixHash(&hash, @as(u32, @bitCast(spec.rope_scaling_factor)));
    mixHash(&hash, @as(u32, @bitCast(spec.rope_freq_base)));
    mixHash(&hash, @as(u32, @bitCast(spec.rms_epsilon)));
    return if (hash == 0) 1 else hash;
}

const ImageWriter = struct {
    device: *Device,
    image: wire.TensorRange,

    fn write(context: *anyopaque, offset: u64, bytes: []const u8) error{UploadFailed}!void {
        const self: *ImageWriter = @ptrCast(@alignCast(context));
        const len: u64 = @intCast(bytes.len);
        if (offset > self.image.nbytes or len > self.image.nbytes - offset)
            return error.UploadFailed;
        const absolute = std.math.add(u64, self.image.offset, offset) catch
            return error.UploadFailed;
        _ = self.device.link.upload(.{
            .handle = self.image.handle,
            .offset = absolute,
            .nbytes = len,
        }, bytes) catch return error.UploadFailed;
    }

    fn interface(self: *ImageWriter) engine.provision.Writer {
        return .{ .context = self, .write_fn = write };
    }
};

fn installModel(self: *Device, sources: engine.model.ModelSources) !void {
    if (!self.inference_capable) return error.Unsupported;
    if (self.residency.phase != .idle) return error.ModelAlreadyPrepared;
    if (!formatSupported(self.inference_format_mask, sources.weight_format))
        return error.UnsupportedWeightFormat;

    const spec = try sources.validate();
    const plan = try engine.model.planImage(spec, sources.weight_format, 0);
    const allocation = try self.link.alloc(plan.image_bytes, @intCast(model_spec.ddr_alignment));
    const image = allocation.range;
    errdefer if (self.residency.phase != .ambiguous) freeQuietly(self, image);

    self.residency = .{
        .phase = .prepared,
        .spec_id = spec.id,
        .weight_format = sources.weight_format,
        .model_hash = modelHash(spec, sources.weight_format),
        .image_bytes = plan.image_bytes,
        .image = image,
    };
    errdefer if (self.residency.phase == .prepared) self.residency.reset();

    var writer = ImageWriter{ .device = self, .image = image };
    const written = try engine.provision.provisionModel(
        self.allocator,
        writer.interface(),
        0,
        sources,
    );
    if (written.image_bytes != image.nbytes) return error.InvalidImage;

    self.residency.phase = .ambiguous;
    _ = try self.link.inference(.{ .install_model = .{
        .image = image,
        .model_hash = self.residency.model_hash,
        .model_slot = resident_model_slot,
        .spec_id = self.residency.spec_id,
        .weight_format = self.residency.weight_format,
    } });
    self.residency.phase = .installed;
}

fn freeQuietly(device: *Device, range: wire.TensorRange) void {
    _ = device.link.free(range) catch return;
}

const CTensor = c.struct_penzai_inference_tensor_source_v1;
const CLayer = c.struct_penzai_inference_layer_v1;
const CDescriptor = c.struct_penzai_inference_model_descriptor_v1;
const CModelResult = c.struct_penzai_inference_model_result_v1;
const CSessionParams = c.struct_penzai_inference_session_params_v1;
const CExecuteRequest = c.struct_penzai_inference_execute_request_v1;
const CExecuteResult = c.struct_penzai_inference_execute_result_v1;
const CApi = c.struct_penzai_inference_v1;

const ModelHandle = struct {
    device: *Device,
    context_tokens: u32,
    vocabulary_size: u32,
    mock: bool,
    session_open: bool = false,
};

const SessionHandle = struct {
    model: *ModelHandle,
    capacity_tokens: u32,
    epoch: u32 = 1,
    committed_tokens: u32 = 0,
    next_request_id: u64 = 1,
    script_cursor: usize = 0,
    open: bool = true,
};

fn weightFormat(raw: u32) !model_spec.WeightFormat {
    return switch (raw) {
        3 => .q1_0,
        4 => .q2_0,
        else => error.UnsupportedWeightFormat,
    };
}

fn tensorBytes(source: *const CTensor) ![]const u8 {
    const data = source.data orelse return error.InvalidTensor;
    const len = std.math.cast(usize, source.data_size) orelse return error.InvalidTensorLength;
    if (len == 0) return error.InvalidTensorLength;
    const pointer: [*]const u8 = @ptrCast(data);
    return pointer[0..len];
}

fn tensorShape(source: *const CTensor, rank: u32) !engine.model.TensorShape {
    if (source.struct_size < @sizeOf(CTensor) or source.rank != rank)
        return error.InvalidTensorShape;
    var dimensions: [4]u32 = undefined;
    for (&dimensions, source.dimensions) |*out, value|
        out.* = std.math.cast(u32, value) orelse return error.InvalidTensorShape;
    return .{ .ne = dimensions };
}

fn lowBit(source: *const CTensor, expected: model_spec.WeightFormat) !engine.model.LowBitTensor {
    const format = try weightFormat(source.format);
    if (format != expected) return error.MixedWeightFormat;
    const bytes = try tensorBytes(source);
    return .{
        .identity = @intFromPtr(bytes.ptr),
        .format = format,
        .shape = try tensorShape(source, 2),
        .bytes = bytes,
    };
}

fn floatTensor(source: *const CTensor) !engine.model.F32Tensor {
    if (source.format != 1) return error.MixedWeightFormat;
    const bytes = try tensorBytes(source);
    return .{
        .identity = @intFromPtr(bytes.ptr),
        .shape = try tensorShape(source, 1),
        .bytes = bytes,
    };
}

const ConvertedSources = struct {
    sources: engine.model.ModelSources,
    layers: []engine.model.LayerSources,

    fn deinit(self: ConvertedSources, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

fn convertSources(allocator: std.mem.Allocator, descriptor: *const CDescriptor) !ConvertedSources {
    if (descriptor.struct_size < @sizeOf(CDescriptor) or
        descriptor.architecture == null or descriptor.architecture_size == 0 or
        descriptor.layers == null or descriptor.layer_count == 0 or
        descriptor.layer_count != descriptor.block_count)
        return error.InvalidTensor;

    const format = try weightFormat(descriptor.embedding.format);
    const source_layers = descriptor.layers[0..descriptor.layer_count];
    const layers = try allocator.alloc(engine.model.LayerSources, source_layers.len);
    errdefer allocator.free(layers);
    for (layers, source_layers) |*out, source| {
        if (source.struct_size < @sizeOf(CLayer)) return error.InvalidTensor;
        out.* = .{
            .query = try lowBit(&source.query, format),
            .key = try lowBit(&source.key, format),
            .value = try lowBit(&source.value, format),
            .attention_output = try lowBit(&source.attention_output, format),
            .gate = try lowBit(&source.gate, format),
            .up = try lowBit(&source.up, format),
            .ffn_down = try lowBit(&source.ffn_down, format),
            .attention_norm = try floatTensor(&source.attention_norm),
            .attention_q_norm = try floatTensor(&source.attention_q_norm),
            .attention_k_norm = try floatTensor(&source.attention_k_norm),
            .ffn_norm = try floatTensor(&source.ffn_norm),
        };
    }

    const tied_output = descriptor.flags & (1 << 1) != 0;
    return .{
        .sources = .{
            .metadata = .{
                .architecture = descriptor.architecture[0..descriptor.architecture_size],
                .block_count = descriptor.block_count,
                .context_length = descriptor.context_length,
                .embedding_length = descriptor.embedding_length,
                .feed_forward_length = descriptor.feed_forward_length,
                .attention_head_count = descriptor.attention_head_count,
                .attention_head_count_kv = descriptor.attention_head_count_kv,
                .attention_key_length = descriptor.attention_key_length,
                .attention_value_length = descriptor.attention_value_length,
                .vocabulary_size = descriptor.vocabulary_size,
                .rope_original_context = descriptor.rope_original_context,
                .rope_scaling_factor = descriptor.rope_scaling_factor,
                .rope_freq_base = descriptor.rope_frequency_base,
                .rms_epsilon = descriptor.rms_epsilon,
            },
            .weight_format = format,
            .embedding = try lowBit(&descriptor.embedding, format),
            .lm_head = if (tied_output) null else try lowBit(&descriptor.output, format),
            .output_norm = try floatTensor(&descriptor.output_norm),
            .layers = layers,
        },
        .layers = layers,
    };
}

fn appendAudit(line: []const u8) void {
    const path = std.c.getenv("PENZAI_EXECUTOR_AUDIT") orelse return;
    const file = std.c.fopen(path, "a") orelse return;
    defer _ = std.c.fclose(file);
    _ = std.c.fwrite(line.ptr, 1, line.len, file);
}

fn auditSimple(comptime event: []const u8) void {
    appendAudit("{\"event\":\"" ++ event ++ "\"}\n");
}

fn auditSessionOpen(capacity_tokens: u32) void {
    var buffer: [96]u8 = undefined;
    const line = std.fmt.bufPrint(
        &buffer,
        "{{\"event\":\"session.open\",\"capacity_tokens\":{d}}}\n",
        .{capacity_tokens},
    ) catch return;
    appendAudit(line);
}

fn auditExecute(first: u32, count: u32, emit: bool, token: u32, committed: u32) void {
    var buffer: [256]u8 = undefined;
    const line = std.fmt.bufPrint(
        &buffer,
        "{{\"event\":\"execute\",\"first_position\":{d},\"token_count\":{d},\"emit\":{s},\"token_id\":{d},\"committed_tokens\":{d}}}\n",
        .{ first, count, if (emit) "true" else "false", token, committed },
    ) catch return;
    appendAudit(line);
}

fn scriptedToken(cursor: usize) u32 {
    const text = if (std.c.getenv("PENZAI_EXECUTOR_TOKENS")) |value|
        std.mem.span(value)
    else
        "2";
    var values: [64]u32 = undefined;
    var count: usize = 0;
    var fields = std.mem.splitScalar(u8, text, ',');
    while (fields.next()) |field| {
        if (count == values.len) break;
        values[count] = std.fmt.parseInt(u32, std.mem.trim(u8, field, " \t"), 10) catch
            continue;
        count += 1;
    }
    return if (count == 0) 2 else values[cursor % count];
}

fn loadModel(
    device_context: ?*anyopaque,
    descriptor: ?*const CDescriptor,
    result: ?*CModelResult,
) callconv(.c) c_int {
    const raw_device = device_context orelse return Status.invalid_argument;
    const source = descriptor orelse return Status.invalid_argument;
    const out = result orelse return Status.invalid_argument;
    if (out.struct_size < @sizeOf(CModelResult)) return Status.invalid_argument;
    out.model = null;
    out.max_context_tokens = 0;
    out.max_tile_tokens = 0;
    out.features = 0;
    if (source.context_length == 0 or source.vocabulary_size == 0)
        return Status.invalid_argument;

    const ggml_device: c.ggml_backend_dev_t = @ptrCast(@alignCast(raw_device));
    if (ggml_device == null or ggml_device.?.*.context == null)
        return Status.invalid_argument;
    const device = deviceOf(ggml_device.?.*.context);
    if (ggml_device != &device.device or !device.inference_capable)
        return Status.unsupported;

    const handle = device.allocator.create(ModelHandle) catch return Status.io;
    if (!device.mock_executor) {
        const converted = convertSources(device.allocator, source) catch |err| {
            device.allocator.destroy(handle);
            return errorStatus(err);
        };
        defer converted.deinit(device.allocator);
        installModel(device, converted.sources) catch |err| {
            device.allocator.destroy(handle);
            return errorStatus(err);
        };
    }

    handle.* = .{
        .device = device,
        .context_tokens = source.context_length,
        .vocabulary_size = source.vocabulary_size,
        .mock = device.mock_executor,
    };
    out.model = handle;
    out.max_context_tokens = source.context_length;
    out.max_tile_tokens = if (device.mock_executor) 4 else model_spec.token_tile_max;
    out.features = inference_features;
    auditSimple("model.load");
    return Status.ok;
}

fn unloadModel(model_context: ?*anyopaque) callconv(.c) void {
    const raw = model_context orelse return;
    const model: *ModelHandle = @ptrCast(@alignCast(raw));
    if (!model.mock) model.device.releaseModel();
    auditSimple("model.unload");
    model.device.allocator.destroy(model);
}

fn openSession(
    model_context: ?*anyopaque,
    params: ?*const CSessionParams,
    result: ?*?*anyopaque,
) callconv(.c) c_int {
    const raw_model = model_context orelse return Status.invalid_argument;
    const config = params orelse return Status.invalid_argument;
    const out = result orelse return Status.invalid_argument;
    out.* = null;
    if (config.struct_size < @sizeOf(CSessionParams) or config.flags != 0 or
        config.capacity_tokens == 0)
        return Status.invalid_argument;
    const model: *ModelHandle = @ptrCast(@alignCast(raw_model));
    if (config.capacity_tokens > model.context_tokens) return Status.invalid_argument;
    if (model.session_open) return Status.state;

    const session = model.device.allocator.create(SessionHandle) catch return Status.io;
    session.* = .{ .model = model, .capacity_tokens = config.capacity_tokens };
    if (!model.mock) {
        _ = model.device.link.inference(.{ .open_session = .{
            .model_slot = resident_model_slot,
            .session_slot = resident_session_slot,
            .capacity_tokens = config.capacity_tokens,
            .epoch = session.epoch,
        } }) catch {
            model.device.allocator.destroy(session);
            return Status.device;
        };
    }
    model.session_open = true;
    out.* = session;
    auditSessionOpen(config.capacity_tokens);
    return Status.ok;
}

fn closeSession(session_context: ?*anyopaque) callconv(.c) void {
    const raw = session_context orelse return;
    const session: *SessionHandle = @ptrCast(@alignCast(raw));
    if (session.open and !session.model.mock) {
        _ = session.model.device.link.inference(.{ .close_session = .{
            .slot = resident_session_slot,
        } }) catch {};
    }
    session.open = false;
    session.model.session_open = false;
    auditSimple("session.close");
    session.model.device.allocator.destroy(session);
}

fn resetSession(session_context: ?*anyopaque) callconv(.c) c_int {
    const raw = session_context orelse return Status.invalid_argument;
    const session: *SessionHandle = @ptrCast(@alignCast(raw));
    if (!session.open) return Status.state;
    const epoch = session.epoch +% 1;
    if (epoch == 0) return Status.state;
    if (!session.model.mock) {
        _ = session.model.device.link.inference(.{ .reset_session = .{
            .session_slot = resident_session_slot,
            .next_epoch = epoch,
        } }) catch return Status.device;
    }
    session.epoch = epoch;
    session.committed_tokens = 0;
    session.script_cursor = 0;
    auditSimple("session.reset");
    return Status.ok;
}

fn execute(
    session_context: ?*anyopaque,
    request: ?*const CExecuteRequest,
    result: ?*CExecuteResult,
) callconv(.c) c_int {
    const raw = session_context orelse return Status.invalid_argument;
    const command = request orelse return Status.invalid_argument;
    const out = result orelse return Status.invalid_argument;
    const session: *SessionHandle = @ptrCast(@alignCast(raw));
    if (!session.open) return Status.state;
    if (command.struct_size < @sizeOf(CExecuteRequest) or
        out.struct_size < @sizeOf(CExecuteResult) or command.token_ids == null or
        command.token_count == 0 or command.token_count > model_spec.token_tile_max or
        command.first_position != session.committed_tokens or
        command.flags & ~@as(u64, 1) != 0 or
        command.first_position > session.capacity_tokens or
        command.token_count > session.capacity_tokens - command.first_position)
        return Status.invalid_argument;

    var tokens = [_]u32{0} ** model_spec.token_tile_max;
    for (tokens[0..command.token_count], command.token_ids[0..command.token_count]) |*dst, token| {
        if (token >= session.model.vocabulary_size) return Status.invalid_argument;
        dst.* = token;
    }
    const emit = command.flags & 1 != 0;
    const committed = command.first_position + command.token_count;
    var winner: u32 = 0;
    var winner_logit: f32 = 0;
    var cycles: u64 = 0;

    if (session.model.mock) {
        if (emit) {
            winner = scriptedToken(session.script_cursor);
            session.script_cursor += 1;
            if (winner >= session.model.vocabulary_size) return Status.device;
            winner_logit = 1.0;
        }
    } else {
        const reply = session.model.device.link.inference(.{ .execute = .{
            .request_id = session.next_request_id,
            .model_slot = resident_model_slot,
            .session_slot = resident_session_slot,
            .session_epoch = session.epoch,
            .first_position = command.first_position,
            .valid_tokens = @intCast(command.token_count),
            .flags = if (emit) shared.engine.command.Flags.emit_logits else 0,
            .metrics_level = session.model.device.metrics_level,
            .token_ids = tokens,
        } }) catch return Status.device;
        const response = reply.result orelse return Status.device;
        if (response.session_slot != resident_session_slot or
            response.epoch != session.epoch or response.committed_tokens != committed or
            (response.token() != null) != emit or
            response.metrics_level != session.model.device.metrics_level or
            (response.metrics_snapshot != null) !=
                (session.model.device.metrics_level != .none))
            return Status.device;
        if (emit) {
            winner = response.token().?;
            winner_logit = response.logit;
        }
        cycles = response.cycles;
        session.model.device.metrics_aggregate.add(response);
        session.next_request_id +%= 1;
        if (session.next_request_id == 0) session.next_request_id = 1;
    }

    session.committed_tokens = committed;
    out.committed_tokens = committed;
    out.flags = if (emit) 1 else 0;
    out.token_id = winner;
    out.winning_logit = winner_logit;
    out.device_cycles = cycles;
    out.device_time_ns = if (session.model.device.engine_clock_hz == 0)
        0
    else
        @intCast((@as(u128, cycles) * std.time.ns_per_s) /
            session.model.device.engine_clock_hz);
    auditExecute(command.first_position, command.token_count, emit, winner, committed);
    return Status.ok;
}

fn statusMessage(status: c_int) callconv(.c) [*c]const u8 {
    return switch (status) {
        Status.ok => "ok",
        Status.invalid_argument => "invalid argument",
        Status.unsupported => "unsupported",
        Status.io => "host allocation or transport failure",
        Status.device => "device execution failure",
        Status.state => "invalid executor state",
        else => "unknown inference status",
    };
}

fn errorStatus(err: anyerror) c_int {
    return switch (err) {
        error.OutOfMemory => Status.io,
        error.Unsupported, error.UnsupportedWeightFormat => Status.unsupported,
        error.Transport, error.RemoteFailed, error.RemoteBackendFailure => Status.device,
        error.ModelAlreadyPrepared, error.ModelNotPrepared => Status.state,
        else => Status.invalid_argument,
    };
}

const api = CApi{
    .abi_version = 1,
    .struct_size = @sizeOf(CApi),
    .features = inference_features,
    .load_model = &loadModel,
    .unload_model = &unloadModel,
    .open_session = &openSession,
    .close_session = &closeSession,
    .reset_session = &resetSession,
    .execute = &execute,
    .status_message = &statusMessage,
    .reserved = [_]?*anyopaque{null} ** 8,
};

fn query() callconv(.c) ?*const CApi {
    return &api;
}

fn deviceOf(context: ?*anyopaque) *Device {
    return @ptrCast(@alignCast(context.?));
}

fn deviceName(device: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceOf(device.?.*.context).name;
}

fn deviceDescription(device: c.ggml_backend_dev_t) callconv(.c) [*c]const u8 {
    return deviceOf(device.?.*.context).description;
}

fn deviceMemory(device: c.ggml_backend_dev_t, free: ?*usize, total: ?*usize) callconv(.c) void {
    _ = device;
    if (free) |value| value.* = 4 * 1024 * 1024 * 1024;
    if (total) |value| value.* = 4 * 1024 * 1024 * 1024;
}

fn deviceType(device: c.ggml_backend_dev_t) callconv(.c) c.enum_ggml_backend_dev_type {
    _ = device;
    return c.GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

fn deviceProperties(device: c.ggml_backend_dev_t, properties: [*c]c.ggml_backend_dev_props) callconv(.c) void {
    properties.*.name = deviceName(device);
    properties.*.description = deviceDescription(device);
    properties.*.type = deviceType(device);
    properties.*.device_id = null;
    deviceMemory(device, &properties.*.memory_free, &properties.*.memory_total);
    properties.*.caps = .{
        .async = false,
        .host_buffer = false,
        .buffer_from_host_ptr = false,
        .events = false,
    };
}

fn noBackend(device: c.ggml_backend_dev_t, params: [*c]const u8) callconv(.c) c.ggml_backend_t {
    _ = device;
    _ = params;
    return null;
}

fn noBufferType(device: c.ggml_backend_dev_t) callconv(.c) c.ggml_backend_buffer_type_t {
    _ = device;
    return null;
}

fn noOperation(device: c.ggml_backend_dev_t, operation: ?*const c.ggml_tensor) callconv(.c) bool {
    _ = device;
    _ = operation;
    return false;
}

fn noBufferSupport(device: c.ggml_backend_dev_t, buffer_type: c.ggml_backend_buffer_type_t) callconv(.c) bool {
    _ = device;
    _ = buffer_type;
    return false;
}

const device_interface = c.ggml_backend_device_i{
    .get_name = &deviceName,
    .get_description = &deviceDescription,
    .get_memory = &deviceMemory,
    .get_type = &deviceType,
    .get_props = &deviceProperties,
    .init_backend = &noBackend,
    .get_buffer_type = &noBufferType,
    .get_host_buffer_type = null,
    .buffer_from_host_ptr = null,
    .supports_op = &noOperation,
    .supports_buft = &noBufferSupport,
    .offload_op = &noOperation,
    .event_new = null,
    .event_free = null,
    .event_synchronize = null,
};

fn registryName(registry: c.ggml_backend_reg_t) callconv(.c) [*c]const u8 {
    return deviceOf(registry.?.*.context).name;
}

fn registryDeviceCount(registry: c.ggml_backend_reg_t) callconv(.c) usize {
    _ = registry;
    return 1;
}

fn registryDevice(registry: c.ggml_backend_reg_t, index: usize) callconv(.c) c.ggml_backend_dev_t {
    if (index != 0) return null;
    return &deviceOf(registry.?.*.context).device;
}

fn registryProcedure(
    registry: c.ggml_backend_reg_t,
    name: [*c]const u8,
) callconv(.c) ?*anyopaque {
    _ = registry;
    if (name == null or !std.mem.eql(u8, std.mem.span(name), "penzai.inference.v1"))
        return null;
    return @ptrCast(@constCast(&query));
}

const registry_interface = c.ggml_backend_reg_i{
    .get_name = &registryName,
    .get_device_count = &registryDeviceCount,
    .get_device = &registryDevice,
    .get_proc_address = &registryProcedure,
};

test "full metrics remains host-reserved despite a reported hardware bit" {
    var device: Device = undefined;
    device.feature_mask = capabilities.Feature.metrics_summary |
        capabilities.Feature.metrics_full;
    device.metrics_level = .none;
    device.metrics_aggregate = .{};
    try std.testing.expectError(
        error.MetricsUnsupported,
        device.setMetricsLevel(.full),
    );
}

test "metrics aggregate preserves phase totals and overflow" {
    var values: [engine_metrics.metric_count]u64 = @splat(0);
    values[@intFromEnum(engine_metrics.MetricId.total_cycles)] = 100;
    values[@intFromEnum(engine_metrics.MetricId.projection_drain_cycles)] = 7;
    values[@intFromEnum(engine_metrics.MetricId.projection_selector_high_water)] = 5;
    var aggregate: MetricsAggregate = .{ .level = .summary };
    var result: shared.engine.rpc.ExecuteResult = .{
        .session_slot = 0,
        .flags = 0,
        .epoch = 1,
        .committed_tokens = 8,
        .token_id = 0,
        .logit = 0,
        .status = 0,
        .cycles = 100,
        .metrics_level = .summary,
        .metrics_snapshot = .{
            .schema = engine_metrics.recorder_schema,
            .capabilities = engine_metrics.compiled_hardware_capabilities,
            .status = 0x29,
            .tag = 1,
            .overflow = .{ 0, 1 << 1, 0, 0 },
            .values = values,
        },
    };
    aggregate.add(result);
    result.cycles = 50;
    if (result.metrics_snapshot) |*snapshot| {
        snapshot.values[@intFromEnum(engine_metrics.MetricId.total_cycles)] = 50;
        snapshot.values[@intFromEnum(engine_metrics.MetricId.projection_drain_cycles)] = 11;
        snapshot.values[@intFromEnum(engine_metrics.MetricId.projection_selector_high_water)] = 4;
    }
    aggregate.add(result);

    try std.testing.expectEqual(@as(u64, 2), aggregate.executions);
    try std.testing.expectEqual(@as(u64, 150), aggregate.device_cycles);
    try std.testing.expectEqual(
        aggregate.device_cycles,
        aggregate.values[@intFromEnum(engine_metrics.MetricId.total_cycles)],
    );
    try std.testing.expectEqual(
        @as(u64, 18),
        aggregate.values[@intFromEnum(engine_metrics.MetricId.projection_drain_cycles)],
    );
    try std.testing.expectEqual(
        @as(u64, 5),
        aggregate.values[@intFromEnum(engine_metrics.MetricId.projection_selector_high_water)],
    );
    try std.testing.expectEqual(@as(u32, 2), aggregate.overflow[1]);
}
