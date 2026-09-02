//! Thin llama.cpp integration for the default whole-token executor.
//!
//! Model loading and context execution intentionally use llama.cpp's public
//! APIs. The selected Penzai ggml device advertises `penzai.inference.v1`, so
//! this CLI, llama-cli, and llama-server all exercise the same patched path.
const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");
const shared = @import("shared");
const backend_mod = @import("backend");
const link_mod = @import("link");

const clock = shared.timing;
const metrics = shared.engine.metrics;

pub const Error = error{
    MissingModel,
    ModelLoadFailed,
    ContextInitFailed,
    TokenizeFailed,
    ChatTemplateFailed,
    DecodeFailed,
    LogitsMissing,
    LogitMismatch,
    SamplerInitFailed,
    PieceDecodeFailed,
    BackendHandshakeFailed,
    MetricsUnsupported,
    InvalidSampledResult,
} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const Options = struct {
    model_path: []const u8 = build_options.default_model_path,
    prompt: []const u8 = "Hello",
    prompt_tokens: ?u32 = null,
    max_tokens: u32 = 16,
    n_ctx: u32 = 4096,
    n_batch: u32 = 32,
    n_ubatch: u32 = 16,
    threads: u32 = 1,
    logits_tolerance: f32 = 0.25,
    chat_template: bool = true,
    enable_thinking: bool = false,
    metrics: metrics.Level = .none,
    hardware_metrics: bool = false,
    device_label: []const u8 = "tcp:127.0.0.1:29092",
    exact_tokens: bool = false,
    token_ids: bool = false,
};

const PreparedPrompt = struct {
    allocator: std.mem.Allocator,
    allocation: []u8,
    bytes: []const u8,
    parse_special: bool,

    fn deinit(self: PreparedPrompt) void {
        self.allocator.free(self.allocation);
    }
};

const TokenBuffer = struct {
    allocator: std.mem.Allocator,
    tokens: []c.llama_token,
    prompt_len: usize,

    fn deinit(self: TokenBuffer) void {
        self.allocator.free(self.tokens);
    }
};

const GeneratedOutput = struct {
    writer: *std.Io.Writer,
    vocab: *const c.llama_vocab,
    token_ids: bool,
    emitted: u32 = 0,

    fn emit(self: *@This(), token: c.llama_token) Error!void {
        if (self.token_ids) {
            const token_id = std.math.cast(u32, token) orelse
                return error.PieceDecodeFailed;
            try self.writer.print(
                "generated_token schema=1 index={d} token_id={d}\n",
                .{ self.emitted, token_id },
            );
        } else {
            try writePiece(self.writer, self.vocab, token);
        }
        self.emitted += 1;
    }
};

pub fn runPrompt(
    io: std.Io,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    link: link_mod.Client,
    options: Options,
) Error!void {
    if (options.model_path.len == 0) return error.MissingModel;

    const device = backend_mod.Device.create(allocator, link) catch
        return error.BackendHandshakeFailed;
    defer device.destroy();
    device.setMetricsLevel(options.metrics) catch |err| switch (err) {
        error.MetricsUnsupported => return error.MetricsUnsupported,
        else => return error.BackendHandshakeFailed,
    };
    const model_path = try allocator.dupeZ(u8, options.model_path);
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    var model_params = acceleratedModelParams();
    var devices = [_]c.ggml_backend_dev_t{ @ptrCast(device.ggmlDevice()), null };
    model_params.devices = &devices;
    const load_start = clock.nowNs(io);
    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse
        return error.ModelLoadFailed;
    const model_load_ns = clock.elapsed(load_start, clock.nowNs(io));
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse return error.ModelLoadFailed;
    const ctx_params = contextParams(options);
    const ctx = c.llama_init_from_model(model, ctx_params) orelse
        return error.ContextInitFailed;
    defer c.llama_free(ctx);

    var sampler_params = c.llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    const sampler = c.llama_sampler_chain_init(sampler_params) orelse
        return error.SamplerInitFailed;
    defer c.llama_sampler_free(sampler);
    c.llama_sampler_chain_add(
        sampler,
        c.llama_sampler_init_greedy() orelse return error.SamplerInitFailed,
    );

    const prepared = try preparePrompt(
        allocator,
        model,
        options.prompt,
        options.chat_template,
        options.enable_thinking,
    );
    defer prepared.deinit();
    var token_buffer = try tokenizePrompt(
        allocator,
        vocab,
        prepared,
        options.max_tokens,
        options.prompt_tokens,
    );
    defer token_buffer.deinit();
    const tokens = token_buffer.tokens;
    var n_past = token_buffer.prompt_len;
    if (n_past + options.max_tokens > options.n_ctx) return error.ContextInitFailed;

    const prefill_start = clock.nowNs(io);
    _ = device.takeMetrics();
    try decodeTokenBatches(ctx, tokens[0..n_past], 0, options.n_batch);
    const prefill_end = clock.nowNs(io);
    const prefill_ns = clock.elapsed(prefill_start, prefill_end);
    const prefill_metrics = device.takeMetrics();

    var output = GeneratedOutput{
        .writer = writer,
        .vocab = vocab,
        .token_ids = options.token_ids,
    };
    var generated: u32 = 0;
    var first_output_ns: u64 = 0;
    const generation_start = prefill_end;
    while (generated < options.max_tokens) {
        const token = c.llama_sampler_sample(sampler, ctx, -1);
        c.llama_sampler_accept(sampler, token);
        if (!options.exact_tokens and c.llama_vocab_is_eog(vocab, token)) break;

        try output.emit(token);
        generated += 1;
        if (generated == 1) {
            try writer.flush();
            first_output_ns = clock.nowNs(io);
        }

        // The winner returned by the final requested decode is the final
        // output. Do not execute it when no successor is needed.
        if (generated == options.max_tokens) break;
        tokens[n_past] = token;
        n_past += 1;
        try decodeTokens(ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
    }
    const generation_end = clock.nowNs(io);
    const generation_ns = clock.elapsed(generation_start, generation_end);
    const decode_metrics = device.takeMetrics();
    const ttft_ns = if (first_output_ns == 0)
        0
    else
        clock.elapsed(prefill_start, first_output_ns);

    if (!options.token_ids and generated > 0) try writer.writeByte('\n');
    try writer.print(
        "benchmark_result schema=2 device={s} prompt_tokens={d} generated_tokens={d}" ++
            " model_load_ns={d} prefill_wall_ns={d} generation_wall_ns={d}" ++
            " output_ttft_ns={d} metrics={s}\n",
        .{
            options.device_label,
            token_buffer.prompt_len,
            generated,
            model_load_ns,
            prefill_ns,
            generation_ns,
            ttft_ns,
            @tagName(options.metrics),
        },
    );
    if (options.metrics != .none) {
        const summary: metrics.Summary = .{
            .prompt_tokens = token_buffer.prompt_len,
            .generated_tokens = generated,
            .prefill_tiles = prefill_metrics.executions,
            .decode_executions = decode_metrics.executions,
            .prefill_wall_ns = prefill_ns,
            .prefill_device_ns = device.metricsTimeNs(prefill_metrics.device_cycles),
            .decode_wall_ns = generation_ns,
            .decode_device_ns = device.metricsTimeNs(decode_metrics.device_cycles),
            .first_token_ns = if (generated == 0) 0 else prefill_ns,
            .output_first_token_ns = ttft_ns,
            .engine_cycles = prefill_metrics.device_cycles +| decode_metrics.device_cycles,
        };
        try writeMetricsSummary(writer, summary, options.metrics);
        if (options.hardware_metrics) {
            try writeHardwareMetrics(writer, "prefill", prefill_metrics);
            try writeHardwareMetrics(writer, "decode", decode_metrics);
        }
    }
}

pub fn runLogitsCheck(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    link: link_mod.Client,
    options: Options,
) Error!void {
    if (options.model_path.len == 0) return error.MissingModel;
    if (options.max_tokens == 0) return error.LogitMismatch;

    const model_path = try allocator.dupeZ(u8, options.model_path);
    defer allocator.free(model_path);
    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    var cpu_params = c.llama_model_default_params();
    cpu_params.n_gpu_layers = 0;
    const cpu_model = c.llama_model_load_from_file(model_path.ptr, cpu_params) orelse
        return error.ModelLoadFailed;
    defer c.llama_model_free(cpu_model);

    const device = backend_mod.Device.create(allocator, link) catch
        return error.BackendHandshakeFailed;
    defer device.destroy();
    device.setMetricsLevel(options.metrics) catch |err| switch (err) {
        error.MetricsUnsupported => return error.MetricsUnsupported,
        else => return error.BackendHandshakeFailed,
    };
    var accelerated_params = acceleratedModelParams();
    var devices = [_]c.ggml_backend_dev_t{ @ptrCast(device.ggmlDevice()), null };
    accelerated_params.devices = &devices;
    const accelerated_model = c.llama_model_load_from_file(model_path.ptr, accelerated_params) orelse
        return error.ModelLoadFailed;
    defer c.llama_model_free(accelerated_model);

    const cpu_vocab = c.llama_model_get_vocab(cpu_model) orelse return error.ModelLoadFailed;
    const accelerated_vocab = c.llama_model_get_vocab(accelerated_model) orelse
        return error.ModelLoadFailed;
    const vocab_count_raw = c.llama_vocab_n_tokens(cpu_vocab);
    if (vocab_count_raw <= 0 or vocab_count_raw != c.llama_vocab_n_tokens(accelerated_vocab))
        return error.ModelLoadFailed;
    const vocab_count: usize = @intCast(vocab_count_raw);

    const cpu_context_params = contextParams(options);
    const cpu_ctx = c.llama_init_from_model(cpu_model, cpu_context_params) orelse
        return error.ContextInitFailed;
    defer c.llama_free(cpu_ctx);
    const accelerated_context_params = contextParams(options);
    const accelerated_ctx = c.llama_init_from_model(accelerated_model, accelerated_context_params) orelse
        return error.ContextInitFailed;
    defer c.llama_free(accelerated_ctx);

    const prepared = try preparePrompt(
        allocator,
        cpu_model,
        options.prompt,
        options.chat_template,
        options.enable_thinking,
    );
    defer prepared.deinit();
    var token_buffer = try tokenizePrompt(
        allocator,
        cpu_vocab,
        prepared,
        options.max_tokens,
        options.prompt_tokens,
    );
    defer token_buffer.deinit();
    const tokens = token_buffer.tokens;
    var n_past = token_buffer.prompt_len;
    if (n_past + options.max_tokens > options.n_ctx) return error.ContextInitFailed;

    _ = device.takeMetrics();
    try decodeTokenBatches(cpu_ctx, tokens[0..n_past], 0, options.n_batch);
    try decodeTokenBatches(accelerated_ctx, tokens[0..n_past], 0, options.n_batch);

    try writer.writeAll("penzai verify logits\n");
    try writer.print("prompt_tokens={d} requested_steps={d} vocab={d}\n", .{
        token_buffer.prompt_len,
        options.max_tokens,
        vocab_count,
    });
    var steps: u32 = 0;
    var mismatches: u32 = 0;
    var max_winner_diff: f32 = 0;
    while (steps < options.max_tokens) : (steps += 1) {
        const cpu_logits = try currentRawLogits(cpu_ctx, vocab_count);
        const accelerated_logits = try currentRawLogits(accelerated_ctx, vocab_count);
        const cpu_winner = argmaxIndex(cpu_logits);
        const accelerated_raw_winner = argmaxIndex(accelerated_logits);
        const accelerated_sampled_token = try currentSampledToken(
            accelerated_ctx,
            vocab_count,
        );
        if (accelerated_sampled_token != accelerated_raw_winner) {
            return error.InvalidSampledResult;
        }
        const accelerated_winner = accelerated_sampled_token;
        const accelerated_winner_logit = accelerated_logits[accelerated_winner];
        const winner_diff = @abs(cpu_logits[cpu_winner] - accelerated_winner_logit);
        max_winner_diff = @max(max_winner_diff, winner_diff);
        if (cpu_winner != accelerated_winner) mismatches += 1;
        try writer.print("step={d} cpu_token={d} device_token={d} winner_abs_diff={d:.6}\n", .{
            steps,
            cpu_winner,
            accelerated_winner,
            winner_diff,
        });
        try writer.print(
            "winner_logits schema=1 step={d} cpu={d:.9} device={d:.9}" ++
                " cpu_bits=0x{X:0>8} device_bits=0x{X:0>8}\n",
            .{
                steps,
                cpu_logits[cpu_winner],
                accelerated_winner_logit,
                @as(u32, @bitCast(cpu_logits[cpu_winner])),
                @as(u32, @bitCast(accelerated_winner_logit)),
            },
        );

        if (shouldStopVerification(
            options.exact_tokens,
            steps + 1,
            options.max_tokens,
            c.llama_vocab_is_eog(cpu_vocab, @intCast(cpu_winner)),
        )) break;
        tokens[n_past] = @intCast(cpu_winner);
        n_past += 1;
        try decodeTokens(cpu_ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
        try decodeTokens(accelerated_ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
    }

    try writer.print("token_mismatches={d} max_winner_abs_diff={d:.6} tolerance={d:.6}\n", .{
        mismatches,
        max_winner_diff,
        options.logits_tolerance,
    });
    if (mismatches != 0 or !std.math.isFinite(max_winner_diff) or
        max_winner_diff > options.logits_tolerance)
    {
        try writer.flush();
        return error.LogitMismatch;
    }
    try writer.writeAll("check=ok\n");
    if (options.metrics != .none) {
        const measured = device.takeMetrics();
        try writer.print(
            "verification_metrics schema=1 level={s} executions={d} device_cycles={d} device_ns={d}\n",
            .{
                @tagName(options.metrics),
                measured.executions,
                measured.device_cycles,
                device.metricsTimeNs(measured.device_cycles),
            },
        );
        if (options.hardware_metrics)
            try writeHardwareMetrics(writer, "verification", measured);
    }
}

fn shouldStopVerification(
    exact_tokens: bool,
    completed_steps: u32,
    max_steps: u32,
    winner_is_eog: bool,
) bool {
    return completed_steps == max_steps or (!exact_tokens and winner_is_eog);
}

test "exact logits verification does not stop at EOG" {
    try std.testing.expect(shouldStopVerification(false, 2, 8, true));
    try std.testing.expect(!shouldStopVerification(true, 2, 8, true));
    try std.testing.expect(shouldStopVerification(true, 8, 8, false));
}

fn writeMetricsSummary(
    writer: *std.Io.Writer,
    summary: metrics.Summary,
    level: metrics.Level,
) std.Io.Writer.Error!void {
    try writer.print(
        "metrics_summary schema=1 level={s} prompt_tokens={d} generated_tokens={d}" ++
            " prefill_tiles={d} decode_executions={d} prefill_wall_ns={d}" ++
            " prefill_device_ns={d} decode_wall_ns={d} decode_device_ns={d}" ++
            " first_token_ns={d} output_first_token_ns={d} engine_cycles={d}\n",
        .{
            @tagName(level),
            summary.prompt_tokens,
            summary.generated_tokens,
            summary.prefill_tiles,
            summary.decode_executions,
            summary.prefill_wall_ns,
            summary.prefill_device_ns,
            summary.decode_wall_ns,
            summary.decode_device_ns,
            summary.first_token_ns,
            summary.output_first_token_ns,
            summary.engine_cycles,
        },
    );
}

fn writeHardwareMetrics(
    writer: *std.Io.Writer,
    phase: []const u8,
    aggregate: backend_mod.MetricsAggregate,
) std.Io.Writer.Error!void {
    inline for (@typeInfo(metrics.MetricId).@"enum".fields) |field| {
        const id: metrics.MetricId = @enumFromInt(field.value);
        const index: usize = @intFromEnum(id);
        const overflow_word = index / 32;
        const overflow_bit: u5 = @intCast(index % 32);
        try writer.print(
            "hardware_metric schema=1 phase={s} id={d} name={s} value={d} overflow={}\n",
            .{
                phase,
                index,
                metrics.metricName(id),
                aggregate.values[index],
                aggregate.overflow[overflow_word] & (@as(u32, 1) << overflow_bit) != 0,
            },
        );
    }
}

fn acceleratedModelParams() c.llama_model_params {
    var params = c.llama_model_default_params();
    params.n_gpu_layers = 999;
    params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;
    return params;
}

fn contextParams(options: Options) c.llama_context_params {
    var params = c.llama_context_default_params();
    params.n_ctx = options.n_ctx;
    params.n_batch = options.n_batch;
    params.n_ubatch = options.n_ubatch;
    params.n_seq_max = 1;
    params.n_threads = @intCast(options.threads);
    params.n_threads_batch = @intCast(options.threads);
    params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.op_offload = false;
    return params;
}

fn preparePrompt(
    allocator: std.mem.Allocator,
    model: *const c.llama_model,
    prompt: []const u8,
    use_chat_template: bool,
    enable_thinking: bool,
) Error!PreparedPrompt {
    if (!use_chat_template) return rawPrompt(allocator, prompt);
    const prompt_z = try allocator.dupeZ(u8, prompt);
    defer allocator.free(prompt_z);
    const needed_raw = c.penzai_chat_format_user(model, prompt_z.ptr, enable_thinking, null, 0);
    if (needed_raw == c.PENZAI_CHAT_NO_TEMPLATE) return rawPrompt(allocator, prompt);
    if (needed_raw < 0) return error.ChatTemplateFailed;
    const needed: usize = @intCast(needed_raw);
    const allocation = try allocator.alloc(u8, try checkedAdd(needed, 1));
    errdefer allocator.free(allocation);
    const written = c.penzai_chat_format_user(
        model,
        prompt_z.ptr,
        enable_thinking,
        @ptrCast(allocation.ptr),
        try i32Len(allocation.len),
    );
    if (written != needed_raw) return error.ChatTemplateFailed;
    return .{
        .allocator = allocator,
        .allocation = allocation,
        .bytes = allocation[0..needed],
        .parse_special = true,
    };
}

fn rawPrompt(allocator: std.mem.Allocator, prompt: []const u8) Error!PreparedPrompt {
    const allocation = try allocator.dupe(u8, prompt);
    return .{
        .allocator = allocator,
        .allocation = allocation,
        .bytes = allocation,
        .parse_special = false,
    };
}

fn tokenizePrompt(
    allocator: std.mem.Allocator,
    vocab: *const c.llama_vocab,
    prepared: PreparedPrompt,
    max_decode_tokens: usize,
    forced_prompt_tokens: ?u32,
) Error!TokenBuffer {
    const needed_raw = c.llama_tokenize(
        vocab,
        prepared.bytes.ptr,
        try i32Len(prepared.bytes.len),
        null,
        0,
        true,
        prepared.parse_special,
    );
    if (needed_raw >= 0) return error.TokenizeFailed;
    const tokenized_len: usize = @intCast(-@as(i64, needed_raw));
    if (tokenized_len == 0) return error.TokenizeFailed;
    const prompt_len: usize = if (forced_prompt_tokens) |n| @intCast(n) else tokenized_len;
    if (prompt_len == 0) return error.TokenizeFailed;

    const source_or_prompt_len = @max(tokenized_len, prompt_len);
    const total_len = try checkedAdd(try checkedAdd(source_or_prompt_len, max_decode_tokens), 1);
    const tokens = try allocator.alloc(c.llama_token, total_len);
    errdefer allocator.free(tokens);
    const written = c.llama_tokenize(
        vocab,
        prepared.bytes.ptr,
        try i32Len(prepared.bytes.len),
        tokens.ptr,
        try i32Len(tokenized_len),
        true,
        prepared.parse_special,
    );
    if (written <= 0 or @as(usize, @intCast(written)) != tokenized_len)
        return error.TokenizeFailed;
    if (prompt_len > tokenized_len) {
        const repeat_start: usize = if (tokenized_len > 1) 1 else 0;
        const repeat_len = tokenized_len - repeat_start;
        for (tokenized_len..prompt_len) |index|
            tokens[index] = tokens[repeat_start + (index - tokenized_len) % repeat_len];
    }
    return .{ .allocator = allocator, .tokens = tokens, .prompt_len = prompt_len };
}

fn decodeTokenBatches(
    ctx: *c.llama_context,
    tokens: []c.llama_token,
    start_position: usize,
    batch_tokens: u32,
) Error!void {
    if (batch_tokens == 0) return error.DecodeFailed;
    var offset: usize = 0;
    const limit: usize = @intCast(batch_tokens);
    while (offset < tokens.len) {
        const count = @min(limit, tokens.len - offset);
        const end = offset + count;
        try decodeTokens(ctx, tokens[offset..end], start_position + offset, end == tokens.len);
        offset = end;
    }
}

fn decodeTokens(
    ctx: *c.llama_context,
    tokens: []c.llama_token,
    start_position: usize,
    want_logits: bool,
) Error!void {
    var batch = c.llama_batch_init(@intCast(tokens.len), 0, 1);
    defer c.llama_batch_free(batch);
    batch.n_tokens = @intCast(tokens.len);
    for (tokens, 0..) |token, index| {
        batch.token[index] = token;
        batch.pos[index] = @intCast(start_position + index);
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = if (want_logits and index + 1 == tokens.len) 1 else 0;
    }
    if (c.llama_decode(ctx, batch) != 0) return error.DecodeFailed;
}

fn currentRawLogits(ctx: *c.llama_context, count: usize) Error![]const f32 {
    const logits = c.llama_get_logits(ctx) orelse return error.LogitsMissing;
    return logits[0..count];
}

fn currentSampledToken(ctx: *c.llama_context, vocab_count: usize) Error!usize {
    return validateSampledToken(c.llama_get_sampled_token_ith(ctx, -1), vocab_count);
}

fn validateSampledToken(
    token_raw: c.llama_token,
    vocab_count: usize,
) Error!usize {
    const token = std.math.cast(usize, token_raw) orelse
        return error.InvalidSampledResult;
    if (token >= vocab_count) return error.InvalidSampledResult;
    return token;
}

test "sampled winner is a bounded token, not a vocabulary span" {
    try std.testing.expectEqual(@as(usize, 2), try validateSampledToken(2, 7));
    try std.testing.expectError(
        error.InvalidSampledResult,
        validateSampledToken(7, 7),
    );
}

fn argmaxIndex(values: []const f32) usize {
    var best: usize = 0;
    var best_value = values[0];
    for (values[1..], 1..) |value, index| {
        if (value > best_value) {
            best = index;
            best_value = value;
        }
    }
    return best;
}

fn checkedAdd(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch return error.OutOfMemory;
}

fn i32Len(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse error.TokenizeFailed;
}

fn writePiece(
    writer: *std.Io.Writer,
    vocab: *const c.llama_vocab,
    token: c.llama_token,
) Error!void {
    var buffer: [256]u8 = undefined;
    const written = c.llama_token_to_piece(
        vocab,
        token,
        @ptrCast(buffer[0..].ptr),
        @intCast(buffer.len),
        0,
        false,
    );
    if (written < 0) return error.PieceDecodeFailed;
    try writer.writeAll(buffer[0..@intCast(written)]);
}

fn quietLog(
    level: c.enum_ggml_log_level,
    text: [*c]const u8,
    user_data: ?*anyopaque,
) callconv(.c) void {
    _ = user_data;
    if (level == c.GGML_LOG_LEVEL_ERROR and text != null)
        std.debug.print("llama: {s}", .{std.mem.span(text)});
}
