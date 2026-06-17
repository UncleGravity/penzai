const std = @import("std");
const c = @import("c");
const build_options = @import("build_options");
const backend_mod = @import("backend.zig");
const census_mod = @import("census.zig");
const link_mod = @import("link");
const prof_collector = @import("prof/collector.zig");
const prof_render = @import("prof/render.zig");

pub const Error = error{
    MissingModel,
    ModelPathTooLong,
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
} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const Options = struct {
    model_path: []const u8 = build_options.default_model_path,
    prompt: []const u8 = "Hello",
    max_tokens: u32 = 16,
    census: bool = false,
    n_ctx: u32 = 128,
    n_batch: u32 = 32,
    n_ubatch: u32 = 16,
    threads: u32 = 1,
    logits_tolerance: f32 = 0.25,
    chat_template: bool = true,
    enable_thinking: bool = false,
    profile: bool = false,
    device_label: []const u8 = "fake",
    /// Build the greedy sampler on the llama.cpp backend (bound to seq 0) so the
    /// argmax runs in the device compute graph and only the sampled token id is
    /// transferred back, instead of the full f32 logits vector. Requires device
    /// support for GGML_OP_ARGMAX and GGML_OP_PAD (the static-graph dummy row).
    backend_sampling: bool = false,
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

const DecodeTrace = struct {
    allocator: std.mem.Allocator,
    logits: []f32,
    generated: []c.llama_token,
    vocab_count: usize,
    prompt_tokens: usize,
    steps: usize,

    fn deinit(self: DecodeTrace) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.generated);
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

    const want_profile = options.profile and !options.census;
    var profile_store = prof_collector.Collector.init(io);
    const profile: ?*prof_collector.Collector = if (want_profile) &profile_store else null;
    const device = backend_mod.Device.create(allocator, link) catch return error.BackendHandshakeFailed;
    defer device.destroy();
    device.profile = profile;
    var census: census_mod.Census = .{};
    if (options.census) device.census = &census;

    const model_path = try allocator.dupeZ(u8, options.model_path);
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = 999;
    model_params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;
    var devices = [_]c.ggml_backend_dev_t{ device.ggmlDevice(), null };
    model_params.devices = &devices;

    if (profile) |p| p.setPhase(.model_load);
    const model_load_start = if (profile) |p| p.now() else 0;
    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return error.ModelLoadFailed;
    if (profile) |p| p.addWall(.model_load, p.elapsedSince(model_load_start));
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse return error.ModelLoadFailed;

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = options.n_ctx;
    ctx_params.n_batch = options.n_batch;
    ctx_params.n_ubatch = options.n_ubatch;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = @intCast(options.threads);
    ctx_params.n_threads_batch = @intCast(options.threads);
    ctx_params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // op_offload copies a CPU-side weight to the device to run one op there. With
    // resident weights that backfires: the embedding get_rows re-copies the whole
    // token_embd table every token. Off → get_rows uses the resident table.
    ctx_params.op_offload = false;

    // Greedy sampler, built as a chain up front. With backend sampling it is bound
    // to seq 0 through ctx_params *before* context init, so llama.cpp folds the
    // sampling graph (argmax over the logits) into the compute graph at reserve
    // time and runs it on the device — only the sampled token id returns instead of
    // the full logits vector. Without the flag it is an ordinary host-side greedy
    // driven by llama_sampler_sample. The chain is freed after the context, which
    // references it under backend sampling (single defer → no double free on the
    // context-init error path).
    var sparams = c.llama_sampler_chain_default_params();
    sparams.no_perf = true;
    const sampler = c.llama_sampler_chain_init(sparams) orelse return error.SamplerInitFailed;
    defer c.llama_sampler_free(sampler);
    c.llama_sampler_chain_add(sampler, c.llama_sampler_init_greedy() orelse return error.SamplerInitFailed);

    var sampler_config: c.llama_sampler_seq_config = .{ .seq_id = 0, .sampler = sampler };
    if (options.backend_sampling) {
        ctx_params.samplers = &sampler_config;
        ctx_params.n_samplers = 1;
    }

    if (profile) |p| p.setPhase(.context_init);
    const context_start = if (profile) |p| p.now() else 0;
    const ctx = c.llama_init_from_model(model, ctx_params) orelse return error.ContextInitFailed;
    if (profile) |p| p.addWall(.context_init, p.elapsedSince(context_start));
    defer c.llama_free(ctx);

    const prepared = try preparePrompt(allocator, model, options.prompt, options.chat_template, options.enable_thinking);
    defer prepared.deinit();

    var token_buffer = try tokenizePrompt(allocator, vocab, prepared, @intCast(options.max_tokens));
    defer token_buffer.deinit();
    const tokens = token_buffer.tokens;
    var n_past = token_buffer.prompt_len;

    if (profile) |p| p.setPhase(.prefill);
    const prefill_start = if (profile) |p| p.now() else 0;
    try decodeTokens(ctx, tokens[0..n_past], 0, true);
    if (profile) |p| p.addWall(.prefill, p.elapsedSince(prefill_start));

    if (profile) |p| p.setPhase(.decode);
    var generated: u32 = 0;
    while (generated < options.max_tokens) : (generated += 1) {
        // Bracket the whole per-token step (sample + detokenize + decode) so the
        // decode phase wall closes; the non-decode part lands in host residual.
        const step_start = if (profile) |p| p.now() else 0;
        const token = c.llama_sampler_sample(sampler, ctx, -1);
        c.llama_sampler_accept(sampler, token);
        if (c.llama_vocab_is_eog(vocab, token)) break;

        if (!options.census) try writePiece(writer, vocab, token);
        tokens[n_past] = token;
        n_past += 1;
        try decodeTokens(ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
        if (profile) |p| p.recordDecodeToken(generated == 0, p.elapsedSince(step_start));
    }

    if (options.census) {
        try census.report(writer);
    } else {
        try writer.writeByte('\n');
        if (profile) |p| try prof_render.writeProfile(writer, p, options.model_path, options.device_label);
    }
}

pub fn runLogitsCheck(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    link: link_mod.Client,
    options: Options,
) Error!void {
    if (options.model_path.len == 0) return error.MissingModel;

    const model_path = try allocator.dupeZ(u8, options.model_path);
    defer allocator.free(model_path);

    c.llama_backend_init();
    defer c.llama_backend_free();
    c.llama_log_set(&quietLog, null);

    const cpu = try collectTrace(allocator, model_path, options, null, null);
    defer cpu.deinit();

    const device = backend_mod.Device.create(allocator, link) catch return error.BackendHandshakeFailed;
    defer device.destroy();
    const accelerated = try collectTrace(allocator, model_path, options, device, cpu.generated[0 .. cpu.steps - 1]);
    defer accelerated.deinit();

    if (cpu.vocab_count != accelerated.vocab_count or cpu.steps != accelerated.steps) return error.LogitMismatch;
    const compared = cpu.steps * cpu.vocab_count;
    const diff = maxAbsDiff(cpu.logits[0..compared], accelerated.logits[0..compared]);

    try writer.print("penzai logits\n", .{});
    try writer.print("prompt_tokens={d} steps={d} vocab={d}\n", .{
        cpu.prompt_tokens,
        cpu.steps,
        cpu.vocab_count,
    });
    try writer.print("max_abs_diff={d:.6} tolerance={d:.6}\n", .{ diff, options.logits_tolerance });
    try writer.writeAll("step_max_abs_diff");
    for (0..cpu.steps) |step| {
        const start = step * cpu.vocab_count;
        const step_diff = maxAbsDiff(
            cpu.logits[start..][0..cpu.vocab_count],
            accelerated.logits[start..][0..cpu.vocab_count],
        );
        try writer.print(" {d}={d:.6}", .{ step, step_diff });
    }
    try writer.writeByte('\n');
    try writer.writeAll("step_argmax");
    var token_mismatches: usize = 0;
    for (0..cpu.steps) |step| {
        const start = step * cpu.vocab_count;
        const cpu_argmax = argmaxIndex(cpu.logits[start..][0..cpu.vocab_count]);
        const accel_argmax = argmaxIndex(accelerated.logits[start..][0..cpu.vocab_count]);
        if (cpu_argmax != accel_argmax) token_mismatches += 1;
        try writer.print(" {d}={d}/{d}", .{ step, cpu_argmax, accel_argmax });
    }
    try writer.writeByte('\n');
    try writer.print("token_mismatches={d}\n", .{token_mismatches});
    if (token_mismatches != 0 or diff > options.logits_tolerance) {
        try writer.flush();
        return error.LogitMismatch;
    }
    try writer.print("check=ok\n", .{});
}

fn collectTrace(
    allocator: std.mem.Allocator,
    model_path: [:0]const u8,
    options: Options,
    device: ?*backend_mod.Device,
    forced_tokens: ?[]const c.llama_token,
) Error!DecodeTrace {
    var model_params = c.llama_model_default_params();
    model_params.n_gpu_layers = if (device != null) 999 else 0;
    model_params.split_mode = c.LLAMA_SPLIT_MODE_LAYER;
    var devices = [_]c.ggml_backend_dev_t{ if (device) |d| d.ggmlDevice() else null, null };
    if (device != null) model_params.devices = &devices;

    const model = c.llama_model_load_from_file(model_path.ptr, model_params) orelse return error.ModelLoadFailed;
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse return error.ModelLoadFailed;
    const vocab_count_raw = c.llama_vocab_n_tokens(vocab);
    if (vocab_count_raw <= 0) return error.ModelLoadFailed;
    const vocab_count: usize = @intCast(vocab_count_raw);

    var ctx_params = c.llama_context_default_params();
    ctx_params.n_ctx = options.n_ctx;
    ctx_params.n_batch = options.n_batch;
    ctx_params.n_ubatch = options.n_ubatch;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = @intCast(options.threads);
    ctx_params.n_threads_batch = @intCast(options.threads);
    ctx_params.flash_attn_type = c.LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // op_offload copies a CPU-side weight to the device to run one op there. With
    // resident weights that backfires: the embedding get_rows re-copies the whole
    // token_embd table every token. Off → get_rows uses the resident table.
    ctx_params.op_offload = false;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse return error.ContextInitFailed;
    defer c.llama_free(ctx);

    const sampler = if (forced_tokens == null) c.llama_sampler_init_greedy() orelse return error.SamplerInitFailed else null;
    defer if (sampler) |s| c.llama_sampler_free(s);

    const max_decode_tokens: usize = if (forced_tokens) |tokens| tokens.len else @intCast(options.max_tokens);
    const max_steps = max_decode_tokens + 1;
    const logits = try allocator.alloc(f32, try checkedMul(max_steps, vocab_count));
    errdefer allocator.free(logits);
    const generated = try allocator.alloc(c.llama_token, max_decode_tokens);
    errdefer allocator.free(generated);

    const prepared = try preparePrompt(allocator, model, options.prompt, options.chat_template, options.enable_thinking);
    defer prepared.deinit();

    var token_buffer = try tokenizePrompt(allocator, vocab, prepared, max_decode_tokens);
    defer token_buffer.deinit();
    const tokens = token_buffer.tokens;
    var n_past = token_buffer.prompt_len;

    try decodeTokens(ctx, tokens[0..n_past], 0, true);
    try copyCurrentLogits(ctx, logits[0..vocab_count]);

    var generated_len: usize = 0;
    while (generated_len < max_decode_tokens) : (generated_len += 1) {
        const token = if (forced_tokens) |forced| forced[generated_len] else c.llama_sampler_sample(sampler.?, ctx, -1);
        if (forced_tokens == null) {
            c.llama_sampler_accept(sampler.?, token);
            if (c.llama_vocab_is_eog(vocab, token)) break;
        }

        generated[generated_len] = token;
        tokens[n_past] = token;
        n_past += 1;
        try decodeTokens(ctx, tokens[n_past - 1 .. n_past], n_past - 1, true);
        const step = generated_len + 1;
        try copyCurrentLogits(ctx, logits[step * vocab_count ..][0..vocab_count]);
    }

    return .{
        .allocator = allocator,
        .logits = logits,
        .generated = generated,
        .vocab_count = vocab_count,
        .prompt_tokens = token_buffer.prompt_len,
        .steps = generated_len + 1,
    };
}

fn preparePrompt(
    allocator: std.mem.Allocator,
    model: *const c.llama_model,
    prompt: []const u8,
    use_chat_template: bool,
    enable_thinking: bool,
) Error!PreparedPrompt {
    if (!use_chat_template) {
        return rawPrompt(allocator, prompt);
    }

    const prompt_z = try allocator.dupeZ(u8, prompt);
    defer allocator.free(prompt_z);

    const needed_raw = c.penzai_chat_format_user(model, prompt_z.ptr, enable_thinking, null, 0);
    if (needed_raw == c.PENZAI_CHAT_NO_TEMPLATE) return rawPrompt(allocator, prompt);
    if (needed_raw < 0) return error.ChatTemplateFailed;
    const needed: usize = @intCast(needed_raw);

    const allocation = try allocator.alloc(u8, try checkedAdd(needed, 1));
    errdefer allocator.free(allocation);

    const out_len = std.math.cast(c_int, allocation.len) orelse return error.ChatTemplateFailed;
    const written = c.penzai_chat_format_user(
        model,
        prompt_z.ptr,
        enable_thinking,
        @ptrCast(allocation.ptr),
        out_len,
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
    const prompt_len: usize = @intCast(-@as(i64, needed_raw));
    if (prompt_len == 0) return error.TokenizeFailed;

    const total_len = try checkedAdd(try checkedAdd(prompt_len, max_decode_tokens), 1);
    const tokens = try allocator.alloc(c.llama_token, total_len);
    errdefer allocator.free(tokens);

    const written = c.llama_tokenize(
        vocab,
        prepared.bytes.ptr,
        try i32Len(prepared.bytes.len),
        tokens.ptr,
        try i32Len(prompt_len),
        true,
        prepared.parse_special,
    );
    if (written <= 0 or @as(usize, @intCast(written)) != prompt_len) return error.TokenizeFailed;
    return .{
        .allocator = allocator,
        .tokens = tokens,
        .prompt_len = prompt_len,
    };
}

fn decodeTokens(ctx: *c.llama_context, tokens: []c.llama_token, start_pos: usize, want_logits: bool) Error!void {
    var batch = c.llama_batch_init(@intCast(tokens.len), 0, 1);
    defer c.llama_batch_free(batch);
    batch.n_tokens = @intCast(tokens.len);
    for (tokens, 0..) |token, i| {
        batch.token[i] = token;
        batch.pos[i] = @intCast(start_pos + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = if (want_logits and i + 1 == tokens.len) 1 else 0;
    }
    if (c.llama_decode(ctx, batch) != 0) return error.DecodeFailed;
}

fn copyCurrentLogits(ctx: *c.llama_context, dst: []f32) Error!void {
    const logits_ptr = c.llama_get_logits_ith(ctx, -1) orelse return error.LogitsMissing;
    @memcpy(dst, logits_ptr[0..dst.len]);
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var max: f32 = 0;
    for (a, b) |x, y| {
        const diff = @abs(x - y);
        if (diff > max) max = diff;
    }
    return max;
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

fn checkedMul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch return error.OutOfMemory;
}

fn checkedAdd(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch return error.OutOfMemory;
}

fn i32Len(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse error.TokenizeFailed;
}

fn writePiece(writer: *std.Io.Writer, vocab: *const c.llama_vocab, token: c.llama_token) Error!void {
    var buf: [256]u8 = undefined;
    const n = c.llama_token_to_piece(vocab, token, @ptrCast(buf[0..].ptr), @intCast(buf.len), 0, false);
    if (n < 0) return error.PieceDecodeFailed;
    try writer.writeAll(buf[0..@intCast(n)]);
}

fn quietLog(level: c.enum_ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = level;
    _ = text;
    _ = user_data;
}
